from __future__ import annotations

from csv import DictReader
from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np

from sparse_recon.hs_timeseries import load_structured_simulation_snapshot


@dataclass
class StructureFunctionInput:
    """Prepared pairwise samples for a structure-function calculation."""

    pair_positions: np.ndarray
    pair_fields: np.ndarray
    metadata: dict
    n_steps: int | None = None


@dataclass
class StructureFunctionResult:
    """Structured output for binned structure-function statistics."""

    lambda_bin_centers: np.ndarray
    lambda_bin_edges: np.ndarray
    counts: np.ndarray
    structure_functions: np.ndarray
    orders: np.ndarray
    metadata: dict
    fitted_exponents: dict | None = None
    kurtosis: np.ndarray | None = None

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of the result."""
        return {
            "lambda_bin_centers": self.lambda_bin_centers.tolist(),
            "lambda_bin_edges": self.lambda_bin_edges.tolist(),
            "counts": self.counts.tolist(),
            "structure_functions": self.structure_functions.tolist(),
            "orders": self.orders.tolist(),
            "metadata": self.metadata,
            "fitted_exponents": self.fitted_exponents,
            "kurtosis": None if self.kurtosis is None else self.kurtosis.tolist(),
        }


def _initialize_bin_summary(n_bins: int) -> dict[str, np.ndarray]:
    return {
        "lambda_sum": np.zeros(n_bins, dtype=float),
        "lambda_min": np.full(n_bins, np.inf, dtype=float),
        "lambda_max": np.full(n_bins, -np.inf, dtype=float),
        "delta_bx_sum": np.zeros(n_bins, dtype=float),
        "delta_bx_min": np.full(n_bins, np.inf, dtype=float),
        "delta_bx_max": np.full(n_bins, -np.inf, dtype=float),
        "local_b_sum": np.zeros(n_bins, dtype=float),
        "local_b_min": np.full(n_bins, np.inf, dtype=float),
        "local_b_max": np.full(n_bins, -np.inf, dtype=float),
        "r_sum": np.zeros(n_bins, dtype=float),
        "r_min": np.full(n_bins, np.inf, dtype=float),
        "r_max": np.full(n_bins, -np.inf, dtype=float),
    }


def _update_bin_summary(
    summary: dict[str, np.ndarray],
    *,
    bin_index: int,
    lambda_value: float,
    delta_bx_value: float,
    local_b_value: float,
    separation_value: float,
) -> None:
    summary["lambda_sum"][bin_index] += lambda_value
    summary["lambda_min"][bin_index] = min(summary["lambda_min"][bin_index], lambda_value)
    summary["lambda_max"][bin_index] = max(summary["lambda_max"][bin_index], lambda_value)

    summary["delta_bx_sum"][bin_index] += delta_bx_value
    summary["delta_bx_min"][bin_index] = min(summary["delta_bx_min"][bin_index], delta_bx_value)
    summary["delta_bx_max"][bin_index] = max(summary["delta_bx_max"][bin_index], delta_bx_value)

    summary["local_b_sum"][bin_index] += local_b_value
    summary["local_b_min"][bin_index] = min(summary["local_b_min"][bin_index], local_b_value)
    summary["local_b_max"][bin_index] = max(summary["local_b_max"][bin_index], local_b_value)

    summary["r_sum"][bin_index] += separation_value
    summary["r_min"][bin_index] = min(summary["r_min"][bin_index], separation_value)
    summary["r_max"][bin_index] = max(summary["r_max"][bin_index], separation_value)


def _finalize_bin_summary(
    summary: dict[str, np.ndarray],
    counts: np.ndarray,
) -> dict[str, list[float | None]]:
    counts = np.asarray(counts, dtype=int)
    output: dict[str, list[float | None]] = {}
    for key, values in summary.items():
        result_values: list[float | None] = []
        for index, value in enumerate(values):
            if counts[index] <= 0:
                result_values.append(None)
            elif key.endswith("_sum"):
                result_values.append(float(value / counts[index]))
            elif np.isfinite(value):
                result_values.append(float(value))
            else:
                result_values.append(None)
        output[key.replace("_sum", "_mean")] = result_values
    return output


def _compute_pair_diagnostics(
    pair_positions: np.ndarray,
    pair_fields: np.ndarray,
    *,
    component_index: int = 0,
) -> dict[str, np.ndarray]:
    separation = pair_positions[:, 1, :] - pair_positions[:, 0, :]
    local_b = 0.5 * (pair_fields[:, 0, :] + pair_fields[:, 1, :])
    return {
        "delta_bx": np.abs(pair_fields[:, 1, component_index] - pair_fields[:, 0, component_index]),
        "local_b_magnitude": np.linalg.norm(local_b, axis=1),
        "separation_magnitude": np.linalg.norm(separation, axis=1),
    }


def _classify_undersampled_bins(
    counts: np.ndarray,
    *,
    fraction: float = 0.05,
) -> tuple[np.ndarray, int, float]:
    if fraction < 0.0:
        raise ValueError(f"undersampled fraction must be non-negative; got {fraction}")

    counts = np.asarray(counts, dtype=int)
    counts_max = int(np.max(counts)) if counts.size else 0
    threshold = float(fraction * counts_max)
    if counts_max == 0:
        return np.ones_like(counts, dtype=bool), counts_max, threshold
    return counts < threshold, counts_max, threshold


def _extract_field_metadata(input_metadata: dict | None) -> tuple[dict | None, str | None]:
    if not input_metadata:
        return None, None
    if "field" in input_metadata and isinstance(input_metadata["field"], dict):
        return input_metadata["field"], "input.field"

    timeseries_metadata = input_metadata.get("timeseries_metadata")
    if isinstance(timeseries_metadata, dict):
        field_metadata = timeseries_metadata.get("field")
        if isinstance(field_metadata, dict):
            return field_metadata, "input.timeseries_metadata.field"
    return None, None


def _derive_grid_spacing_metadata(input_metadata: dict | None) -> tuple[dict | None, str | None]:
    field_metadata, source = _extract_field_metadata(input_metadata)
    if field_metadata is None:
        return None, None

    sim_box_rho_p = field_metadata.get("sim_box_rho_p")
    array_shape = field_metadata.get("array_shape")
    if sim_box_rho_p is None or array_shape is None:
        return None, None
    if len(sim_box_rho_p) != 3 or len(array_shape) < 3:
        return None, None

    spacings = {}
    for axis_name, box_length, axis_count in zip(("x", "y", "z"), sim_box_rho_p, array_shape[:3]):
        box_length = float(box_length)
        axis_count = int(axis_count)
        if axis_count <= 1:
            spacing = np.nan
        else:
            spacing = box_length / float(axis_count - 1)
        spacings[axis_name] = spacing

    finite_spacings = [value for value in spacings.values() if np.isfinite(value) and value > 0.0]
    if not finite_spacings:
        return None, source

    return {
        "x": float(spacings["x"]),
        "y": float(spacings["y"]),
        "z": float(spacings["z"]),
        "min": float(min(finite_spacings)),
        "box_perp_max": float(max(float(sim_box_rho_p[0]), float(sim_box_rho_p[1]))),
    }, source


def _resolve_lambda_min_floor(grid_spacing_rho_p: dict | None) -> float | None:
    if not isinstance(grid_spacing_rho_p, dict):
        return None

    grid_spacing_min = grid_spacing_rho_p.get("min")
    if grid_spacing_min is None or not np.isfinite(grid_spacing_min) or grid_spacing_min <= 0.0:
        return None
    return 0.5 * float(grid_spacing_min)


def _build_unordered_pairs(array: np.ndarray) -> np.ndarray:
    n_samples = len(array)
    if n_samples < 2:
        return np.empty((0, 2) + array.shape[1:], dtype=array.dtype)

    left_index, right_index = np.triu_indices(n_samples, k=1)
    return np.stack([array[left_index], array[right_index]], axis=1)


def _flatten_structured_snapshot(field) -> tuple[np.ndarray, np.ndarray]:
    x_axis = np.asarray(field.axes["x"], dtype=float)
    y_axis = np.asarray(field.axes["y"], dtype=float)
    z_axis = np.asarray(field.axes["z"], dtype=float)
    xx, yy, zz = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    values = np.asarray(field.values, dtype=float).reshape(-1, field.values.shape[-1])
    return coords, values


def _compute_pair_lambda_values(
    pair_positions: np.ndarray,
    pair_fields: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    finite_positions_mask = np.isfinite(pair_positions).all(axis=(1, 2))
    finite_fields_mask = np.isfinite(pair_fields).all(axis=(1, 2))

    separation = pair_positions[:, 1, :] - pair_positions[:, 0, :]
    local_b = 0.5 * (pair_fields[:, 0, :] + pair_fields[:, 1, :])
    local_b_magnitude = np.linalg.norm(local_b, axis=1)
    valid_local_b_mask = np.isfinite(local_b_magnitude) & (local_b_magnitude > 0.0)

    unit_local_b = np.zeros_like(local_b)
    unit_local_b[valid_local_b_mask] = (
        local_b[valid_local_b_mask] / local_b_magnitude[valid_local_b_mask, None]
    )
    parallel_projection = np.sum(separation * unit_local_b, axis=1)
    separation_perp = separation - parallel_projection[:, None] * unit_local_b
    lambda_values = np.linalg.norm(separation_perp, axis=1)
    valid_lambda_mask = np.isfinite(lambda_values) & (lambda_values > 0.0)
    base_valid_mask = (
        finite_positions_mask
        & finite_fields_mask
        & valid_local_b_mask
        & valid_lambda_mask
    )
    return lambda_values, base_valid_mask


def _resolve_simulation_cube_lambda_range(
    field_metadata: dict,
    *,
    lambda_min: float | None,
    lambda_max: float | None,
) -> tuple[float, float]:
    grid_spacing_rho_p, _ = _derive_grid_spacing_metadata({"field": field_metadata})
    if grid_spacing_rho_p is None:
        raise ValueError(
            "Simulation cube metadata does not provide enough information to resolve lambda range"
        )
    lambda_min_floor = _resolve_lambda_min_floor(grid_spacing_rho_p)

    resolved_lambda_min = float(
        grid_spacing_rho_p["min"] if lambda_min is None else lambda_min
    )
    if lambda_min_floor is not None:
        resolved_lambda_min = max(resolved_lambda_min, lambda_min_floor)
    resolved_lambda_max = float(
        grid_spacing_rho_p["box_perp_max"] if lambda_max is None else lambda_max
    )
    if resolved_lambda_min <= 0.0 or resolved_lambda_max <= 0.0:
        raise ValueError("Resolved simulation-cube lambda range must be positive")
    if resolved_lambda_max < resolved_lambda_min:
        raise ValueError("Resolved simulation-cube lambda_max must be greater than or equal to lambda_min")
    if np.isclose(resolved_lambda_min, resolved_lambda_max):
        resolved_lambda_min *= 0.99
        resolved_lambda_max *= 1.01
    return resolved_lambda_min, resolved_lambda_max


def _generate_stratified_random_pairs(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    lambda_bin_edges: np.ndarray,
    candidate_pairs: int,
    target_pairs_per_bin: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if candidate_pairs <= 0:
        raise ValueError(f"candidate_pairs must be positive; got {candidate_pairs}")
    if target_pairs_per_bin <= 0:
        raise ValueError(
            f"target_pairs_per_bin must be positive; got {target_pairs_per_bin}"
        )

    n_points = len(coords)
    if n_points < 2:
        raise ValueError("Simulation cube must contain at least two points for pair sampling")

    rng = np.random.default_rng(random_seed)
    n_bins = len(lambda_bin_edges) - 1
    accepted_counts = np.zeros(n_bins, dtype=int)
    candidate_counts = np.zeros(n_bins, dtype=int)
    candidate_summary = _initialize_bin_summary(n_bins)
    accepted_summary = _initialize_bin_summary(n_bins)
    accepted_pair_positions: list[np.ndarray] = []
    accepted_pair_fields: list[np.ndarray] = []
    seen_pair_ids: set[int] = set()

    raw_pairs_drawn = 0
    unique_pairs_considered = 0
    valid_pairs_in_range = 0

    while raw_pairs_drawn < candidate_pairs and np.any(accepted_counts < target_pairs_per_bin):
        remaining_draws = candidate_pairs - raw_pairs_drawn
        batch_size = min(
            remaining_draws,
            max(4096, 8 * int(np.sum(target_pairs_per_bin - accepted_counts))),
        )
        left_index = rng.integers(0, n_points, size=batch_size, dtype=np.int64)
        right_index = rng.integers(0, n_points, size=batch_size, dtype=np.int64)
        raw_pairs_drawn += batch_size

        distinct_mask = left_index != right_index
        if not np.any(distinct_mask):
            continue
        left_index = left_index[distinct_mask]
        right_index = right_index[distinct_mask]

        ordered_left = np.minimum(left_index, right_index)
        ordered_right = np.maximum(left_index, right_index)
        pair_ids = ordered_left * np.int64(n_points) + ordered_right
        _, unique_index = np.unique(pair_ids, return_index=True)
        ordered_left = ordered_left[unique_index]
        ordered_right = ordered_right[unique_index]
        pair_ids = pair_ids[unique_index]

        is_new_mask = np.fromiter(
            (int(pair_id) not in seen_pair_ids for pair_id in pair_ids),
            dtype=bool,
            count=len(pair_ids),
        )
        if not np.any(is_new_mask):
            continue

        ordered_left = ordered_left[is_new_mask]
        ordered_right = ordered_right[is_new_mask]
        pair_ids = pair_ids[is_new_mask]
        for pair_id in pair_ids:
            seen_pair_ids.add(int(pair_id))

        unique_pairs_considered += len(pair_ids)
        batch_pair_positions = np.stack(
            [coords[ordered_left], coords[ordered_right]],
            axis=1,
        )
        batch_pair_fields = np.stack(
            [values[ordered_left], values[ordered_right]],
            axis=1,
        )
        lambda_values, base_valid_mask = _compute_pair_lambda_values(
            batch_pair_positions,
            batch_pair_fields,
        )
        pair_diagnostics = _compute_pair_diagnostics(
            batch_pair_positions,
            batch_pair_fields,
        )
        in_range_mask = (
            base_valid_mask
            & (lambda_values >= lambda_bin_edges[0])
            & (lambda_values <= lambda_bin_edges[-1])
        )
        if not np.any(in_range_mask):
            continue

        valid_pairs_in_range += int(np.sum(in_range_mask))
        valid_lambda_values = lambda_values[in_range_mask]
        valid_positions = batch_pair_positions[in_range_mask]
        valid_fields = batch_pair_fields[in_range_mask]
        valid_delta_bx = pair_diagnostics["delta_bx"][in_range_mask]
        valid_local_b = pair_diagnostics["local_b_magnitude"][in_range_mask]
        valid_separation = pair_diagnostics["separation_magnitude"][in_range_mask]
        valid_bin_index = (
            np.searchsorted(lambda_bin_edges, valid_lambda_values, side="right") - 1
        )
        valid_bin_index = np.clip(valid_bin_index, 0, n_bins - 1)
        candidate_counts += np.bincount(valid_bin_index, minlength=n_bins)
        for lambda_value, delta_bx_value, local_b_value, separation_value, bin_index in zip(
            valid_lambda_values,
            valid_delta_bx,
            valid_local_b,
            valid_separation,
            valid_bin_index,
        ):
            _update_bin_summary(
                candidate_summary,
                bin_index=int(bin_index),
                lambda_value=float(lambda_value),
                delta_bx_value=float(delta_bx_value),
                local_b_value=float(local_b_value),
                separation_value=float(separation_value),
            )

        for pair_position, pair_field, bin_index, lambda_value, delta_bx_value, local_b_value, separation_value in zip(
            valid_positions,
            valid_fields,
            valid_bin_index,
            valid_lambda_values,
            valid_delta_bx,
            valid_local_b,
            valid_separation,
        ):
            if accepted_counts[bin_index] >= target_pairs_per_bin:
                continue
            accepted_pair_positions.append(pair_position)
            accepted_pair_fields.append(pair_field)
            accepted_counts[bin_index] += 1
            _update_bin_summary(
                accepted_summary,
                bin_index=int(bin_index),
                lambda_value=float(lambda_value),
                delta_bx_value=float(delta_bx_value),
                local_b_value=float(local_b_value),
                separation_value=float(separation_value),
            )
            if np.all(accepted_counts >= target_pairs_per_bin):
                break

    if not accepted_pair_positions:
        raise ValueError(
            "No simulation-cube candidate pairs were accepted within the requested lambda range"
        )

    metadata = {
        "random_seed": int(random_seed),
        "candidate_pairs_requested": int(candidate_pairs),
        "candidate_pairs_drawn": int(raw_pairs_drawn),
        "unique_pairs_considered": int(unique_pairs_considered),
        "valid_pairs_in_range": int(valid_pairs_in_range),
        "candidate_acceptance_fraction": (
            float(np.sum(accepted_counts)) / float(valid_pairs_in_range)
            if valid_pairs_in_range > 0
            else 0.0
        ),
        "target_pairs_per_bin": int(target_pairs_per_bin),
        "candidate_counts_per_bin": candidate_counts.tolist(),
        "accepted_counts_per_bin": accepted_counts.tolist(),
        "candidate_bin_summary": _finalize_bin_summary(candidate_summary, candidate_counts),
        "accepted_bin_summary": _finalize_bin_summary(accepted_summary, accepted_counts),
        "full_bin_coverage_achieved": bool(
            np.all(accepted_counts >= target_pairs_per_bin)
        ),
        "sampling": "direct_random_pairs_stratified_by_lambda",
        "pair_uniqueness": (
            "unordered_pairs_without_replacement_among_considered_candidates"
        ),
    }
    return (
        np.stack(accepted_pair_positions, axis=0),
        np.stack(accepted_pair_fields, axis=0),
        metadata,
    )


def _generate_local_reference_pairs(
    coords: np.ndarray,
    values: np.ndarray,
    grid_shape: tuple[int, int, int],
    *,
    max_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_offset <= 0:
        raise ValueError(f"max_offset must be positive; got {max_offset}")

    nx, ny, nz = grid_shape
    coord_grid = coords.reshape(nx, ny, nz, 3)
    value_grid = values.reshape(nx, ny, nz, values.shape[-1])

    pair_positions: list[np.ndarray] = []
    pair_fields: list[np.ndarray] = []
    for dx in range(0, max_offset + 1):
        for dy in range(-max_offset, max_offset + 1):
            for dz in range(-max_offset, max_offset + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                if dx == 0 and dy < 0:
                    continue
                if dx == 0 and dy == 0 and dz <= 0:
                    continue

                x_stop = nx - dx
                y_start = max(0, -dy)
                y_stop = ny - max(0, dy)
                z_start = max(0, -dz)
                z_stop = nz - max(0, dz)
                if x_stop <= 0 or y_start >= y_stop or z_start >= z_stop:
                    continue

                base_pos = coord_grid[:x_stop, y_start:y_stop, z_start:z_stop]
                shifted_pos = coord_grid[
                    dx : dx + x_stop,
                    y_start + dy : y_stop + dy,
                    z_start + dz : z_stop + dz,
                ]
                base_val = value_grid[:x_stop, y_start:y_stop, z_start:z_stop]
                shifted_val = value_grid[
                    dx : dx + x_stop,
                    y_start + dy : y_stop + dy,
                    z_start + dz : z_stop + dz,
                ]
                pair_positions.append(
                    np.stack(
                        [base_pos.reshape(-1, 3), shifted_pos.reshape(-1, 3)],
                        axis=1,
                    )
                )
                pair_fields.append(
                    np.stack(
                        [base_val.reshape(-1, values.shape[-1]), shifted_val.reshape(-1, values.shape[-1])],
                        axis=1,
                    )
                )

    if not pair_positions:
        return (
            np.empty((0, 2, 3), dtype=float),
            np.empty((0, 2, values.shape[-1]), dtype=float),
        )
    return np.concatenate(pair_positions, axis=0), np.concatenate(pair_fields, axis=0)


def _build_cube_sampling_diagnostics(
    sampling_metadata: dict,
    *,
    lambda_bin_edges: np.ndarray,
    grid_spacing_rho_p: dict | None,
    left_edge_fraction_threshold: float = 0.5,
) -> tuple[dict, list[dict]]:
    candidate_counts = np.asarray(sampling_metadata["candidate_counts_per_bin"], dtype=int)
    accepted_counts = np.asarray(sampling_metadata["accepted_counts_per_bin"], dtype=int)
    target_pairs_per_bin = int(sampling_metadata["target_pairs_per_bin"])
    occupancy_fraction = (
        accepted_counts / float(target_pairs_per_bin)
        if target_pairs_per_bin > 0
        else np.zeros_like(accepted_counts, dtype=float)
    )
    accepted_fraction = np.divide(
        accepted_counts,
        candidate_counts,
        out=np.zeros_like(accepted_counts, dtype=float),
        where=candidate_counts > 0,
    )

    first_populated_bin = int(np.argmax(accepted_counts > 0)) if np.any(accepted_counts > 0) else None
    left_edge_bins = min(3, len(accepted_counts))
    left_edge_undercovered = bool(
        np.any(occupancy_fraction[:left_edge_bins] < left_edge_fraction_threshold)
    )

    candidate_below_grid_multiples = {}
    if grid_spacing_rho_p is not None:
        grid_spacing = float(grid_spacing_rho_p["min"])
        lambda_centers = np.sqrt(lambda_bin_edges[:-1] * lambda_bin_edges[1:])
        for factor in (1.0, 2.0, 4.0):
            candidate_below_grid_multiples[f"{factor:g}x"] = int(
                np.sum(candidate_counts[lambda_centers <= factor * grid_spacing])
            )
    warnings_records = []
    if left_edge_undercovered:
        message = (
            "Simulation-cube diagnostic indicates the left-edge lambda bins are "
            "under-covered relative to the target occupancy."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        warnings_records.append(
            {
                "kind": "cube_left_edge_undercovered",
                "message": message,
                "left_edge_bins": left_edge_bins,
                "occupancy_fraction": occupancy_fraction[:left_edge_bins].tolist(),
                "threshold": float(left_edge_fraction_threshold),
            }
        )
    if grid_spacing_rho_p is not None and candidate_below_grid_multiples.get("1x", 0) == 0:
        message = (
            "Simulation-cube diagnostic found no candidate pairs at or below one grid spacing; "
            "the requested left edge is likely unsupported by candidate coverage."
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        warnings_records.append(
            {
                "kind": "cube_no_gridscale_candidates",
                "message": message,
                "candidate_below_grid_spacing": candidate_below_grid_multiples,
            }
        )

    return {
        "candidate_counts_per_bin": candidate_counts.tolist(),
        "accepted_counts_per_bin": accepted_counts.tolist(),
        "candidate_to_accepted_fraction_per_bin": accepted_fraction.tolist(),
        "occupancy_fraction_per_bin": occupancy_fraction.tolist(),
        "accepted_bin_summary": sampling_metadata["accepted_bin_summary"],
        "candidate_bin_summary": sampling_metadata["candidate_bin_summary"],
        "first_populated_bin": first_populated_bin,
        "first_populated_lambda_center": (
            float(np.sqrt(lambda_bin_edges[first_populated_bin] * lambda_bin_edges[first_populated_bin + 1]))
            if first_populated_bin is not None
            else None
        ),
        "left_edge_bins_undercovered": left_edge_undercovered,
        "left_edge_fraction_threshold": float(left_edge_fraction_threshold),
        "candidate_below_grid_spacing_multiples": candidate_below_grid_multiples,
    }, warnings_records


def _compare_small_lambda_results(
    production_result: StructureFunctionResult,
    reference_result: StructureFunctionResult,
    *,
    max_bins: int = 3,
    relative_difference_threshold: float = 0.5,
) -> tuple[dict, list[dict]]:
    warnings_records = []
    production_counts = np.asarray(production_result.counts, dtype=int)
    reference_counts = np.asarray(reference_result.counts, dtype=int)
    production_first = int(np.argmax(production_counts > 0)) if np.any(production_counts > 0) else None
    reference_first = int(np.argmax(reference_counts > 0)) if np.any(reference_counts > 0) else None

    order_differences = {}
    if production_first is not None and reference_first is not None:
        for order_index, order in enumerate(production_result.orders):
            diffs = []
            for bin_index in range(min(max_bins, len(production_counts))):
                prod_value = production_result.structure_functions[order_index, bin_index]
                ref_value = reference_result.structure_functions[order_index, bin_index]
                if not np.isfinite(prod_value) or not np.isfinite(ref_value) or ref_value == 0.0:
                    diffs.append(None)
                else:
                    diffs.append(float((prod_value - ref_value) / ref_value))
            order_differences[str(int(order))] = diffs

        first_order_diffs = [
            value
            for value in order_differences.get("1", [])
            if value is not None
        ]
        if first_order_diffs and np.max(np.abs(first_order_diffs)) > relative_difference_threshold:
            message = (
                "Simulation-cube diagnostic found material disagreement between the "
                "production sampler and local reference at small lambda."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            warnings_records.append(
                {
                    "kind": "cube_local_reference_disagreement",
                    "message": message,
                    "relative_difference_threshold": float(relative_difference_threshold),
                    "order_differences": order_differences,
                }
            )

    return {
        "production_first_populated_bin": production_first,
        "reference_first_populated_bin": reference_first,
        "production_counts_first_bins": production_counts[:max_bins].tolist(),
        "reference_counts_first_bins": reference_counts[:max_bins].tolist(),
        "relative_differences_by_order_first_bins": order_differences,
    }, warnings_records


def prepare_simulation_cube_structure_function_input(
    simulation_path: str | Path,
    *,
    sim_box_rho_p: tuple[float, float, float],
    n_lambda_bins: int,
    vector_variables: tuple[str, ...] | list[str] | None = None,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
    candidate_pairs: int = 200000,
    target_pairs_per_bin: int = 256,
    random_seed: int = 0,
) -> StructureFunctionInput:
    """Build stratified random point pairs directly from a structured simulation cube."""

    field = load_structured_simulation_snapshot(
        simulation_path,
        sim_box_rho_p=sim_box_rho_p,
        vector_variables=vector_variables,
    )
    coords, values = _flatten_structured_snapshot(field)
    resolved_lambda_min, resolved_lambda_max = _resolve_simulation_cube_lambda_range(
        dict(field.metadata or {}),
        lambda_min=lambda_min,
        lambda_max=lambda_max,
    )
    lambda_bin_edges = np.logspace(
        np.log10(resolved_lambda_min),
        np.log10(resolved_lambda_max),
        n_lambda_bins + 1,
    )
    grid_spacing_rho_p, _ = _derive_grid_spacing_metadata({"field": dict(field.metadata or {})})
    pair_positions, pair_fields, sampling_metadata = _generate_stratified_random_pairs(
        coords,
        values,
        lambda_bin_edges=lambda_bin_edges,
        candidate_pairs=candidate_pairs,
        target_pairs_per_bin=target_pairs_per_bin,
        random_seed=random_seed,
    )
    diagnostics_metadata, diagnostic_warnings = _build_cube_sampling_diagnostics(
        sampling_metadata,
        lambda_bin_edges=lambda_bin_edges,
        grid_spacing_rho_p=grid_spacing_rho_p,
    )

    metadata = {
        "input_mode": "simulation_cube",
        "field": dict(field.metadata or {}),
        "pair_generation": "stratified_random_pairs",
        "sampling": {
            **sampling_metadata,
            "resolved_lambda_min": float(resolved_lambda_min),
            "resolved_lambda_max": float(resolved_lambda_max),
            "lambda_bin_edges": lambda_bin_edges.tolist(),
            "n_lambda_bins": int(n_lambda_bins),
        },
        "diagnostics": {
            "cube_sampling": diagnostics_metadata,
            "warnings": diagnostic_warnings,
        },
        "field_component": "bx",
        "local_b_definition": "pair_midpoint",
        "position_units": "rho_p",
    }
    return StructureFunctionInput(
        pair_positions=pair_positions,
        pair_fields=pair_fields,
        metadata=metadata,
        n_steps=None,
    )


def prepare_simulation_cube_local_reference_input(
    simulation_path: str | Path,
    *,
    sim_box_rho_p: tuple[float, float, float],
    vector_variables: tuple[str, ...] | list[str] | None = None,
    max_offset: int,
) -> StructureFunctionInput:
    """Build deterministic local-offset reference pairs for simulation-cube diagnostics."""

    field = load_structured_simulation_snapshot(
        simulation_path,
        sim_box_rho_p=sim_box_rho_p,
        vector_variables=vector_variables,
    )
    coords, values = _flatten_structured_snapshot(field)
    pair_positions, pair_fields = _generate_local_reference_pairs(
        coords,
        values,
        field.grid_shape,
        max_offset=max_offset,
    )
    return StructureFunctionInput(
        pair_positions=pair_positions,
        pair_fields=pair_fields,
        metadata={
            "input_mode": "simulation_cube_local_reference",
            "field": dict(field.metadata or {}),
            "pair_generation": "local_reference_offsets",
            "reference_max_offset": int(max_offset),
            "field_component": "bx",
            "local_b_definition": "pair_midpoint",
            "position_units": "rho_p",
        },
        n_steps=None,
    )


def _load_timeseries_metadata(metadata_path: str | Path | None) -> dict | None:
    if metadata_path is None:
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_timeseries_structure_function_input(
    csv_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> StructureFunctionInput:
    """Build synchronized same-timestep spacecraft pairs from a time-series CSV."""

    csv_path = Path(csv_path)
    metadata_json = _load_timeseries_metadata(metadata_path)

    rows: list[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"Time-series CSV is empty: {csv_path}")

    metadata_labels = None
    if metadata_json is not None:
        metadata_labels = metadata_json.get("helioswarm", {}).get("spacecraft_labels")

    inferred_labels: list[str] = []
    for row in rows:
        label = row["spacecraft_label"]
        if label not in inferred_labels:
            inferred_labels.append(label)
    spacecraft_labels = list(metadata_labels) if metadata_labels is not None else inferred_labels
    label_to_index = {label: index for index, label in enumerate(spacecraft_labels)}

    step_values = sorted({int(row["step"]) for row in rows})
    step_to_index = {step: index for index, step in enumerate(step_values)}
    n_steps = len(step_values)
    n_spacecraft = len(spacecraft_labels)

    times = np.full(n_steps, np.nan, dtype=float)
    positions = np.full((n_steps, n_spacecraft, 3), np.nan, dtype=float)
    fields = np.full((n_steps, n_spacecraft, 3), np.nan, dtype=float)

    for row in rows:
        step = int(row["step"])
        label = row["spacecraft_label"]
        if label not in label_to_index:
            raise ValueError(f"Unknown spacecraft label '{label}' in {csv_path}")
        step_index = step_to_index[step]
        spacecraft_index = label_to_index[label]
        times[step_index] = float(row["time_seconds"])
        positions[step_index, spacecraft_index] = [
            float(row["x_rho_p"]),
            float(row["y_rho_p"]),
            float(row["z_rho_p"]),
        ]
        fields[step_index, spacecraft_index] = [
            float(row["bx"]),
            float(row["by"]),
            float(row["bz"]),
        ]

    if np.isnan(positions).any() or np.isnan(fields).any():
        raise ValueError(
            "Time-series CSV does not contain a complete rectangular grid of "
            "(step, spacecraft_label) samples"
        )

    pair_positions = []
    pair_fields = []
    for step_index in range(n_steps):
        pair_positions.append(_build_unordered_pairs(positions[step_index]))
        pair_fields.append(_build_unordered_pairs(fields[step_index]))

    metadata = {
        "input_mode": "interpolated_timeseries",
        "field_component": "bx",
        "local_b_definition": "pair_midpoint",
        "pair_generation": "synchronized_same_timestep_spacecraft_pairs",
        "position_units": "rho_p",
        "spacecraft_labels": spacecraft_labels,
        "time_seconds": times.tolist(),
        "input": {
            "timeseries_csv": str(csv_path),
            "timeseries_metadata": None if metadata_path is None else str(metadata_path),
        },
    }
    if metadata_json is not None:
        metadata["timeseries_metadata"] = metadata_json

    return StructureFunctionInput(
        pair_positions=np.concatenate(pair_positions, axis=0),
        pair_fields=np.concatenate(pair_fields, axis=0),
        metadata=metadata,
        n_steps=n_steps,
    )


def _validate_component(component: str) -> int:
    if component != "bx":
        raise ValueError(f"Unsupported field component '{component}'. Only 'bx' is supported in v1.")
    return 0


def _warn_for_short_timeseries(*, n_steps: int, orders: np.ndarray) -> list[dict]:
    warning_records = []
    for order in orders:
        threshold = float(order * np.log10(order))
        if n_steps < threshold:
            message = (
                f"Structure-function order {int(order)} uses n_steps={n_steps}, which is "
                f"below the configured warning threshold {threshold:.3f} = p*log10(p)."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            warning_records.append(
                {
                    "order": int(order),
                    "n_steps": int(n_steps),
                    "threshold": threshold,
                    "formula": "p*log10(p)",
                    "message": message,
                }
            )
    return warning_records


def _resolve_lambda_edges(
    lambda_values: np.ndarray,
    *,
    n_lambda_bins: int,
    lambda_min: float | None,
    lambda_max: float | None,
    grid_spacing_rho_p: dict | None = None,
) -> np.ndarray:
    if n_lambda_bins <= 0:
        raise ValueError(f"n_lambda_bins must be positive; got {n_lambda_bins}")

    valid_lambda = lambda_values[np.isfinite(lambda_values) & (lambda_values > 0.0)]
    if valid_lambda.size == 0:
        raise ValueError("No positive finite lambda values are available for logarithmic binning")

    resolved_lambda_min = float(valid_lambda.min() if lambda_min is None else lambda_min)
    lambda_min_floor = _resolve_lambda_min_floor(grid_spacing_rho_p)
    if lambda_min_floor is not None:
        resolved_lambda_min = max(resolved_lambda_min, lambda_min_floor)
    resolved_lambda_max = float(valid_lambda.max() if lambda_max is None else lambda_max)

    if resolved_lambda_min <= 0.0 or resolved_lambda_max <= 0.0:
        raise ValueError("lambda_min and lambda_max must be positive for logarithmic bins")
    if resolved_lambda_max < resolved_lambda_min:
        raise ValueError("lambda_max must be greater than or equal to lambda_min")

    if np.isclose(resolved_lambda_min, resolved_lambda_max):
        resolved_lambda_min *= 0.99
        resolved_lambda_max *= 1.01

    return np.logspace(
        np.log10(resolved_lambda_min),
        np.log10(resolved_lambda_max),
        n_lambda_bins + 1,
    )


def compute_structure_functions(
    pair_positions: np.ndarray,
    pair_fields: np.ndarray,
    *,
    component: str = "bx",
    max_order: int = 4,
    local_b_definition: str = "pair_midpoint",
    n_lambda_bins: int = 24,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
    input_metadata: dict | None = None,
    n_steps: int | None = None,
    undersampled_fraction: float = 0.05,
) -> StructureFunctionResult:
    """
    Compute binned multipoint structure functions for `B_x`.

    Pair semantics in v1:
    - `delta_bx = bx_j - bx_i`
    - `B_local = 0.5 * (B_i + B_j)`
    - `lambda = |r_perp|`, where `r_perp` is the separation projected
      perpendicular to `B_local`
    - `S_p(lambda)` is the mean of `|delta_bx|**p` within each log-spaced bin
    """

    component_index = _validate_component(component)
    if local_b_definition != "pair_midpoint":
        raise ValueError(
            f"Unsupported local_b_definition '{local_b_definition}'. "
            "Only 'pair_midpoint' is supported in v1."
        )
    if max_order <= 0:
        raise ValueError(f"max_order must be positive; got {max_order}")

    pair_positions = np.asarray(pair_positions, dtype=float)
    pair_fields = np.asarray(pair_fields, dtype=float)
    if pair_positions.shape[:2] != pair_fields.shape[:2]:
        raise ValueError("pair_positions and pair_fields must contain the same pair count and width")
    if pair_positions.ndim != 3 or pair_positions.shape[1:] != (2, 3):
        raise ValueError(
            "pair_positions must be shaped (n_pairs, 2, 3); "
            f"got {pair_positions.shape}"
        )
    if pair_fields.ndim != 3 or pair_fields.shape[1:] != (2, 3):
        raise ValueError(
            "pair_fields must be shaped (n_pairs, 2, 3); "
            f"got {pair_fields.shape}"
        )

    orders = np.arange(1, max_order + 1, dtype=int)
    warning_records = (
        _warn_for_short_timeseries(n_steps=n_steps, orders=orders)
        if n_steps is not None
        else []
    )

    lambda_values, base_valid_mask = _compute_pair_lambda_values(
        pair_positions,
        pair_fields,
    )
    finite_positions_mask = np.isfinite(pair_positions).all(axis=(1, 2))
    finite_fields_mask = np.isfinite(pair_fields).all(axis=(1, 2))
    local_b = 0.5 * (pair_fields[:, 0, :] + pair_fields[:, 1, :])
    local_b_magnitude = np.linalg.norm(local_b, axis=1)
    valid_local_b_mask = np.isfinite(local_b_magnitude) & (local_b_magnitude > 0.0)
    valid_lambda_mask = np.isfinite(lambda_values) & (lambda_values > 0.0)
    valid_lambda_values = lambda_values[base_valid_mask]
    grid_spacing_rho_p, grid_spacing_source = _derive_grid_spacing_metadata(input_metadata)
    lambda_bin_edges = _resolve_lambda_edges(
        valid_lambda_values,
        n_lambda_bins=n_lambda_bins,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        grid_spacing_rho_p=grid_spacing_rho_p,
    )
    lambda_bin_centers = np.sqrt(lambda_bin_edges[:-1] * lambda_bin_edges[1:])

    in_range_mask = (
        base_valid_mask
        & (lambda_values >= lambda_bin_edges[0])
        & (lambda_values <= lambda_bin_edges[-1])
    )
    valid_pair_index = np.nonzero(in_range_mask)[0]
    valid_bin_index = np.searchsorted(lambda_bin_edges, lambda_values[in_range_mask], side="right") - 1
    valid_bin_index = np.clip(valid_bin_index, 0, n_lambda_bins - 1)

    counts = np.bincount(valid_bin_index, minlength=n_lambda_bins).astype(int)
    structure_functions = np.full((len(orders), n_lambda_bins), np.nan, dtype=float)
    delta_bx = np.abs(pair_fields[:, 1, component_index] - pair_fields[:, 0, component_index])

    for order_index, order in enumerate(orders):
        powered_increment = delta_bx[valid_pair_index] ** order
        sums = np.bincount(valid_bin_index, weights=powered_increment, minlength=n_lambda_bins)
        populated_mask = counts > 0
        structure_functions[order_index, populated_mask] = sums[populated_mask] / counts[populated_mask]

    kurtosis = None
    if max_order >= 4:
        kurtosis = np.full(n_lambda_bins, np.nan, dtype=float)
        second_order = structure_functions[1]
        fourth_order = structure_functions[3]
        valid_kurtosis_mask = (
            (counts > 0)
            & np.isfinite(second_order)
            & np.isfinite(fourth_order)
            & (second_order != 0.0)
        )
        kurtosis[valid_kurtosis_mask] = (
            fourth_order[valid_kurtosis_mask] / (second_order[valid_kurtosis_mask] ** 2)
        )

    undersampled_bin_mask, counts_max, undersampled_threshold = _classify_undersampled_bins(
        counts,
        fraction=undersampled_fraction,
    )
    diagnostics_metadata = (
        dict(input_metadata.get("diagnostics", {}))
        if isinstance(input_metadata, dict) and "diagnostics" in input_metadata
        else {}
    )
    if diagnostics_metadata.get("warnings"):
        warning_records.extend(diagnostics_metadata["warnings"])

    excluded_count = int(len(pair_positions) - len(valid_pair_index))
    metadata = {
        "analysis": {
            "field_component": component,
            "max_order": int(max_order),
            "orders": orders.tolist(),
            "local_b_definition": local_b_definition,
            "lambda_definition": "norm_of_separation_projected_perpendicular_to_pair_midpoint_local_B",
            "lambda_units": "rho_p",
            "increment_definition": "delta_bx = bx_j - bx_i",
            "bin_statistic": "mean(abs(delta_bx)**order) within each logarithmic lambda bin",
            "n_lambda_bins": int(n_lambda_bins),
            "lambda_min": float(lambda_bin_edges[0]),
            "lambda_max": float(lambda_bin_edges[-1]),
            "binning": "logspace",
            "counts_max": counts_max,
            "undersampled_bin_mask": undersampled_bin_mask.tolist(),
            "undersampled_count_threshold_mode": "relative_to_max",
            "undersampled_count_fraction": float(undersampled_fraction),
            "undersampled_count_threshold": float(undersampled_threshold),
            "grid_spacing_rho_p": grid_spacing_rho_p,
            "grid_spacing_definition": (
                "minimum structured-grid point spacing from sim_box_rho_p/(n_axis-1)"
                if grid_spacing_rho_p is not None
                else None
            ),
            "grid_spacing_source": grid_spacing_source,
            "grid_reference_line_orientation": "vertical",
        },
        "input": dict(input_metadata or {}),
        "filtering": {
            "total_pairs": int(len(pair_positions)),
            "kept_pairs": int(len(valid_pair_index)),
            "excluded_pairs": excluded_count,
            "excluded_nonfinite_positions": int((~finite_positions_mask).sum()),
            "excluded_nonfinite_fields": int((~finite_fields_mask).sum()),
            "excluded_invalid_local_b": int((~valid_local_b_mask & finite_positions_mask & finite_fields_mask).sum()),
            "excluded_invalid_lambda": int((~valid_lambda_mask & finite_positions_mask & finite_fields_mask & valid_local_b_mask).sum()),
            "excluded_outside_lambda_range": int((base_valid_mask & ~in_range_mask).sum()),
        },
        "extensions": {
            "fitted_exponents": None,
            "kurtosis": None if kurtosis is None else kurtosis.tolist(),
        },
        "diagnostics": diagnostics_metadata,
        "warnings": warning_records,
    }

    return StructureFunctionResult(
        lambda_bin_centers=lambda_bin_centers,
        lambda_bin_edges=lambda_bin_edges,
        counts=counts,
        structure_functions=structure_functions,
        orders=orders,
        metadata=metadata,
        fitted_exponents=None,
        kurtosis=kurtosis,
    )


def plot_structure_functions(
    result: StructureFunctionResult,
    *,
    title: str = "",
    mask_undersampled: bool = True,
    undersampled_fraction: float | None = None,
):
    """Render a minimal log-log plot of the populated structure-function bins."""
    analysis_metadata = result.metadata.get("analysis", {})
    lambda_units = analysis_metadata.get("lambda_units", "rho_p")
    if undersampled_fraction is None:
        undersampled_fraction = analysis_metadata.get("undersampled_count_fraction", 0.05)
    undersampled_bin_mask, _, _ = _classify_undersampled_bins(
        result.counts,
        fraction=float(undersampled_fraction),
    )
    has_kurtosis = result.kurtosis is not None
    if has_kurtosis:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
        structure_ax, kurtosis_ax = axes
    else:
        fig, structure_ax = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
        axes = structure_ax
        kurtosis_ax = None

    def _add_reference_lines(ax) -> None:
        grid_spacing_rho_p = analysis_metadata.get("grid_spacing_rho_p")
        if not isinstance(grid_spacing_rho_p, dict):
            return

        grid_spacing_min = grid_spacing_rho_p.get("min")
        if grid_spacing_min is not None and np.isfinite(grid_spacing_min) and grid_spacing_min > 0.0:
            ax.axvline(
                float(grid_spacing_min),
                color="#7A7A7A",
                linestyle="--",
                linewidth=1.6,
                label=f"grid spacing (min Δ = {float(grid_spacing_min):.3g} {lambda_units})",
            )
        box_perp_max = grid_spacing_rho_p.get("box_perp_max")
        if box_perp_max is not None and np.isfinite(box_perp_max) and box_perp_max > 0.0:
            ax.axvline(
                float(box_perp_max),
                color="#A0A0A0",
                linestyle="--",
                linewidth=1.6,
                label=f"perp box size (max Lxy = {float(box_perp_max):.3g} {lambda_units})",
            )

    for order_index, order in enumerate(result.orders):
        values = result.structure_functions[order_index]
        valid_mask = np.isfinite(values) & (result.counts > 0)
        if mask_undersampled:
            valid_mask &= ~undersampled_bin_mask
        if not np.any(valid_mask):
            continue
        structure_ax.plot(
            result.lambda_bin_centers[valid_mask],
            values[valid_mask],
            marker="o",
            linewidth=1.8,
            label=fr"$S_{int(order)}$",
        )

    _add_reference_lines(structure_ax)
    structure_ax.set_xscale("log")
    structure_ax.set_yscale("log")
    structure_ax.set_xlabel(fr"$\lambda$ ({lambda_units})")
    structure_ax.set_ylabel(r"$S_n(\lambda)$")
    structure_ax.grid(alpha=0.3, which="both")
    structure_ax.legend(loc="best")

    if kurtosis_ax is not None:
        kurtosis_values = np.asarray(result.kurtosis, dtype=float)
        valid_mask = np.isfinite(kurtosis_values) & (result.counts > 0)
        if mask_undersampled:
            valid_mask &= ~undersampled_bin_mask
        if np.any(valid_mask):
            kurtosis_ax.plot(
                result.lambda_bin_centers[valid_mask],
                kurtosis_values[valid_mask],
                marker="o",
                linewidth=1.8,
                color="#0072B2",
                label=r"$S_4(\lambda) / S_2(\lambda)^2$",
            )
        _add_reference_lines(kurtosis_ax)
        kurtosis_ax.set_xscale("log")
        kurtosis_ax.set_xlabel(fr"$\lambda$ ({lambda_units})")
        kurtosis_ax.set_ylabel(r"$S_4(\lambda) / S_2(\lambda)^2$")
        kurtosis_ax.grid(alpha=0.3, which="both")
        kurtosis_ax.legend(loc="best")

    if title:
        if has_kurtosis:
            fig.suptitle(title)
        else:
            structure_ax.set_title(title)
    return fig, axes


def plot_cube_sampling_diagnostics(
    result: StructureFunctionResult,
    *,
    reference_result: StructureFunctionResult | None = None,
    title: str = "",
):
    """Render diagnostic plots for simulation-cube lambda coverage and small-scale comparison."""

    diagnostics = result.metadata.get("diagnostics", {}).get("cube_sampling", {})
    lambda_centers = np.asarray(result.lambda_bin_centers, dtype=float)
    candidate_counts = np.asarray(
        diagnostics.get("candidate_counts_per_bin", np.zeros_like(result.counts)),
        dtype=float,
    )
    accepted_counts = np.asarray(
        diagnostics.get("accepted_counts_per_bin", np.zeros_like(result.counts)),
        dtype=float,
    )
    occupancy_fraction = np.asarray(
        diagnostics.get("occupancy_fraction_per_bin", np.zeros_like(result.counts, dtype=float)),
        dtype=float,
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    axes[0, 0].plot(lambda_centers, candidate_counts, marker="o", linewidth=1.6)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Candidate Counts")
    axes[0, 0].set_xlabel(r"$\lambda$ (rho_p)")
    axes[0, 0].set_ylabel("count")
    axes[0, 0].grid(alpha=0.3, which="both")

    axes[0, 1].plot(lambda_centers, accepted_counts, marker="o", linewidth=1.6, color="#D55E00")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Accepted Counts")
    axes[0, 1].set_xlabel(r"$\lambda$ (rho_p)")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].grid(alpha=0.3, which="both")

    axes[1, 0].plot(lambda_centers, occupancy_fraction, marker="o", linewidth=1.6, color="#009E73")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("Occupancy Fraction")
    axes[1, 0].set_xlabel(r"$\lambda$ (rho_p)")
    axes[1, 0].set_ylabel("accepted / target")
    axes[1, 0].grid(alpha=0.3, which="both")

    left_edge_undercovered = diagnostics.get("left_edge_bins_undercovered", False)
    for order_index, order in enumerate(result.orders):
        values = result.structure_functions[order_index]
        valid_mask = np.isfinite(values) & (result.counts > 0)
        if np.any(valid_mask):
            axes[1, 1].plot(
                lambda_centers[valid_mask],
                values[valid_mask],
                marker="o",
                linewidth=1.6,
                label=fr"$S_{int(order)}$",
            )
    if reference_result is not None:
        ref_values = reference_result.structure_functions[0]
        valid_mask = np.isfinite(ref_values) & (reference_result.counts > 0)
        if np.any(valid_mask):
            axes[1, 1].plot(
                reference_result.lambda_bin_centers[valid_mask],
                ref_values[valid_mask],
                marker="s",
                linestyle="--",
                linewidth=1.4,
                color="#444444",
                label="local reference $S_1$",
            )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title(
        "Structure Functions"
        + (" (left edge under-covered)" if left_edge_undercovered else "")
    )
    axes[1, 1].set_xlabel(r"$\lambda$ (rho_p)")
    axes[1, 1].set_ylabel(r"$S_n(\lambda)$")
    axes[1, 1].grid(alpha=0.3, which="both")
    axes[1, 1].legend(loc="best")

    if title:
        fig.suptitle(title)
    return fig, axes
