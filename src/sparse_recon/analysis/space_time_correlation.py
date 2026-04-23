from __future__ import annotations

from csv import DictReader
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass
class SpaceTimeCorrelationInput:
    """Prepared interpolated time-series data for space-time correlation analysis."""

    times: np.ndarray
    positions: np.ndarray
    wrapped_positions: np.ndarray | None
    fields: np.ndarray
    spacecraft_labels: list[str]
    metadata: dict


@dataclass
class SpaceTimeCorrelationResult:
    """
    Structured output for a binned space-time correlation map.

    Array orientation:
    - `counts`, `correlation`, and `valid_mask` are shaped `(n_tau_bins, n_r_bins)`
    - axis 0 follows `tau_bin_centers`
    - axis 1 follows `r_bin_centers`
    """

    r_bin_edges: np.ndarray
    r_bin_centers: np.ndarray
    tau_bin_edges: np.ndarray
    tau_bin_centers: np.ndarray
    counts: np.ndarray
    correlation: np.ndarray
    valid_mask: np.ndarray
    metadata: dict

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation of the result."""
        return {
            "r_bin_edges": self.r_bin_edges.tolist(),
            "r_bin_centers": self.r_bin_centers.tolist(),
            "tau_bin_edges": self.tau_bin_edges.tolist(),
            "tau_bin_centers": self.tau_bin_centers.tolist(),
            "counts": self.counts.tolist(),
            "correlation": self.correlation.tolist(),
            "valid_mask": self.valid_mask.tolist(),
            "metadata": self.metadata,
        }


@dataclass
class HubAutocorrelationResult:
    """Structured output for the Hub-only single-spacecraft autocorrelation."""

    tau_values: np.ndarray
    counts: np.ndarray
    correlation: np.ndarray
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "tau_values": self.tau_values.tolist(),
            "counts": self.counts.tolist(),
            "correlation": self.correlation.tolist(),
            "metadata": self.metadata,
        }


def _compute_component_fluctuation_statistics(
    fields: np.ndarray,
    *,
    component_index: int,
    spacecraft_labels: list[str],
) -> dict[str, np.ndarray]:
    """Compute per-spacecraft mean-removed fluctuations and standard deviations."""

    component_values = np.asarray(fields[:, :, component_index], dtype=float)
    if component_values.ndim != 2:
        raise ValueError("Component field values must be shaped (n_steps, n_spacecraft)")

    means = np.nanmean(component_values, axis=0)
    fluctuations = component_values - means[None, :]
    variances = np.nanmean(fluctuations**2, axis=0)
    stddevs = np.sqrt(variances)

    if np.any(~np.isfinite(means)):
        raise ValueError("Per-spacecraft means for delta bx contain non-finite values")
    if np.any(~np.isfinite(stddevs)):
        raise ValueError("Per-spacecraft standard deviations for delta bx contain non-finite values")
    zero_variance_labels = [
        label
        for label, stddev in zip(spacecraft_labels, stddevs)
        if stddev <= 0.0
    ]
    if zero_variance_labels:
        raise ValueError(
            "Cannot compute normalized delta bx correlation because these spacecraft have "
            "zero variance: " + ", ".join(zero_variance_labels)
        )

    return {
        "means": means,
        "fluctuations": fluctuations,
        "variances": variances,
        "stddevs": stddevs,
    }


def _load_timeseries_metadata(metadata_path: str | Path | None) -> dict | None:
    if metadata_path is None:
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_spacecraft_subset(
    spacecraft_labels: list[str] | tuple[str, ...] | None,
    *,
    available_labels: list[str],
) -> list[str]:
    if spacecraft_labels is None:
        return list(available_labels)
    if not isinstance(spacecraft_labels, (list, tuple)):
        raise ValueError(
            "spacecraft_labels must be provided as a list or tuple of labels such as "
            "['H', 'N1', 'N2']"
        )
    normalized_labels = [str(label).strip() for label in spacecraft_labels]
    if not normalized_labels or any(not label for label in normalized_labels):
        raise ValueError("spacecraft_labels must contain at least one non-empty label")
    if len(set(normalized_labels)) != len(normalized_labels):
        raise ValueError("spacecraft_labels must not contain duplicates")
    if "H" not in normalized_labels:
        raise ValueError("spacecraft_labels must include the Hub spacecraft 'H'")

    unknown_labels = [label for label in normalized_labels if label not in available_labels]
    if unknown_labels:
        raise ValueError(
            "Unknown spacecraft labels requested: " + ", ".join(sorted(unknown_labels))
        )
    return normalized_labels


def _reconstruct_unwrapped_positions_from_metadata(
    *,
    times: np.ndarray,
    metadata_json: dict,
    selected_labels: list[str],
    available_labels: list[str],
) -> tuple[np.ndarray, dict]:
    motion_metadata = metadata_json.get("motion")
    if not isinstance(motion_metadata, dict):
        raise ValueError(
            "Time-series metadata must contain a 'motion' block to reconstruct unwrapped positions"
        )

    initial_unwrapped = motion_metadata.get("initial_unwrapped_coords_rho_p")
    velocity_rho_p_s = motion_metadata.get("velocity_rho_p_s")
    dt_seconds = motion_metadata.get("dt_seconds")
    n_steps = motion_metadata.get("n_steps")
    if initial_unwrapped is None or velocity_rho_p_s is None or dt_seconds is None or n_steps is None:
        raise ValueError(
            "Time-series metadata motion block must include initial_unwrapped_coords_rho_p, "
            "velocity_rho_p_s, dt_seconds, and n_steps"
        )

    initial_unwrapped = np.asarray(initial_unwrapped, dtype=float)
    velocity_rho_p_s = np.asarray(velocity_rho_p_s, dtype=float)
    times = np.asarray(times, dtype=float)
    if initial_unwrapped.ndim != 2 or initial_unwrapped.shape[1] != 3:
        raise ValueError("initial_unwrapped_coords_rho_p must be shaped (n_spacecraft, 3)")
    if velocity_rho_p_s.shape != (3,):
        raise ValueError("velocity_rho_p_s must contain three components")
    if len(times) != int(n_steps):
        raise ValueError(
            f"CSV times contain {len(times)} steps but metadata motion block declares {n_steps}"
        )
    if not np.allclose(times, np.arange(len(times), dtype=float) * float(dt_seconds)):
        raise ValueError("CSV times are inconsistent with metadata motion dt_seconds")

    metadata_labels = metadata_json.get("helioswarm", {}).get("spacecraft_labels")
    if metadata_labels is not None and list(metadata_labels) != list(available_labels):
        raise ValueError(
            "Time-series metadata spacecraft_labels do not match the CSV/available label order"
        )
    if initial_unwrapped.shape[0] != len(available_labels):
        raise ValueError(
            "initial_unwrapped_coords_rho_p row count must match the available spacecraft labels"
        )

    selected_indices = np.array([available_labels.index(label) for label in selected_labels], dtype=int)
    unwrapped_positions = (
        initial_unwrapped[None, :, :]
        + times[:, None, None] * velocity_rho_p_s[None, None, :]
    )[:, selected_indices, :]
    return unwrapped_positions, {
        "position_geometry": "unwrapped_physical_trajectory",
        "position_geometry_source": "reconstructed_from_timeseries_metadata.motion",
        "position_geometry_metadata_fields": [
            "initial_unwrapped_coords_rho_p",
            "velocity_rho_p_s",
            "dt_seconds",
            "n_steps",
        ],
    }


def _validate_component(component: str) -> int:
    if component != "bx":
        raise ValueError(f"Unsupported field component '{component}'. Only 'bx' is supported in v1.")
    return 0


def _resolve_uniform_cadence(times: np.ndarray, *, rtol: float = 1e-8, atol: float = 1e-10) -> float:
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("At least two time samples are required to resolve the native cadence")
    dt_values = np.diff(times)
    if np.any(~np.isfinite(dt_values)) or np.any(dt_values <= 0.0):
        raise ValueError("Time samples must be finite and strictly increasing")
    reference_dt = float(dt_values[0])
    if not np.allclose(dt_values, reference_dt, rtol=rtol, atol=atol):
        raise ValueError("Time samples must use a uniform native cadence in v1")
    return reference_dt


def _resolve_max_lag_steps(
    n_steps: int,
    *,
    max_lag_steps: int | None,
    max_lag_fraction: float,
) -> int:
    if n_steps < 2:
        raise ValueError("At least two time samples are required for lagged correlations")
    if max_lag_steps is not None:
        if max_lag_steps < 0:
            raise ValueError(f"max_lag_steps must be non-negative; got {max_lag_steps}")
        if max_lag_steps >= n_steps:
            raise ValueError(
                f"max_lag_steps must be smaller than the record length {n_steps}; got {max_lag_steps}"
            )
        return int(max_lag_steps)
    if not 0.0 < max_lag_fraction <= 1.0:
        raise ValueError(
            f"max_lag_fraction must lie in the interval (0, 1]; got {max_lag_fraction}"
        )
    return max(1, int(np.floor((n_steps - 1) * float(max_lag_fraction))))


def _build_centered_linear_edges(
    centers: np.ndarray,
    *,
    spacing: float,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or len(centers) == 0:
        raise ValueError("centers must be a non-empty one-dimensional array")
    if spacing <= 0.0:
        raise ValueError(f"spacing must be positive; got {spacing}")
    edges = np.empty(len(centers) + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * spacing
    edges[-1] = centers[-1] + 0.5 * spacing
    return edges


def _resolve_linear_bin_edges(
    *,
    values_min: float,
    values_max: float,
    n_bins: int | None,
    explicit_edges: np.ndarray | None,
    default_bins: int,
) -> np.ndarray:
    if explicit_edges is not None:
        edges = np.asarray(explicit_edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError("Explicit bin edges must be a one-dimensional array with at least two entries")
        if np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
            raise ValueError("Explicit bin edges must be finite and strictly increasing")
        return edges

    resolved_bins = default_bins if n_bins is None else int(n_bins)
    if resolved_bins <= 0:
        raise ValueError(f"Number of bins must be positive; got {resolved_bins}")
    if not np.isfinite(values_min) or not np.isfinite(values_max):
        raise ValueError("Cannot resolve linear bin edges from non-finite data bounds")
    if values_max < values_min:
        raise ValueError("values_max must be greater than or equal to values_min")
    if np.isclose(values_min, values_max):
        span = max(abs(values_min), 1.0)
        values_min -= 0.5 * span
        values_max += 0.5 * span
    return np.linspace(values_min, values_max, resolved_bins + 1)


def _iter_unordered_pairs_with_self(n_spacecraft: int):
    for left_index in range(n_spacecraft):
        yield left_index, left_index
    for left_index in range(n_spacecraft):
        for right_index in range(left_index + 1, n_spacecraft):
            yield left_index, right_index


def prepare_timeseries_space_time_correlation_input(
    csv_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    spacecraft_labels: list[str] | tuple[str, ...] | None = None,
) -> SpaceTimeCorrelationInput:
    """
    Load an interpolated spacecraft time-series CSV into dense arrays.

    The returned arrays are shaped `(n_steps, n_spacecraft, 3)`. The spacecraft subset
    must use repo-native labels such as `["H", "N1", "N2"]`; numeric indices are not
    accepted in v1.
    """

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
    available_labels = list(metadata_labels) if metadata_labels is not None else inferred_labels
    selected_labels = _validate_spacecraft_subset(
        spacecraft_labels,
        available_labels=available_labels,
    )

    label_to_index = {label: index for index, label in enumerate(available_labels)}
    step_values = sorted({int(row["step"]) for row in rows})
    step_to_index = {step: index for index, step in enumerate(step_values)}
    n_steps = len(step_values)
    n_spacecraft_full = len(available_labels)

    times = np.full(n_steps, np.nan, dtype=float)
    positions_full = np.full((n_steps, n_spacecraft_full, 3), np.nan, dtype=float)
    fields_full = np.full((n_steps, n_spacecraft_full, 3), np.nan, dtype=float)

    for row in rows:
        step = int(row["step"])
        label = row["spacecraft_label"]
        if label not in label_to_index:
            raise ValueError(f"Unknown spacecraft label '{label}' in {csv_path}")
        step_index = step_to_index[step]
        spacecraft_index = label_to_index[label]
        times[step_index] = float(row["time_seconds"])
        positions_full[step_index, spacecraft_index] = [
            float(row["x_rho_p"]),
            float(row["y_rho_p"]),
            float(row["z_rho_p"]),
        ]
        fields_full[step_index, spacecraft_index] = [
            float(row["bx"]),
            float(row["by"]),
            float(row["bz"]),
        ]

    if np.isnan(positions_full).any() or np.isnan(fields_full).any():
        raise ValueError(
            "Time-series CSV does not contain a complete rectangular grid of "
            "(step, spacecraft_label) samples"
        )

    selected_indices = np.array([label_to_index[label] for label in selected_labels], dtype=int)
    positions = positions_full[:, selected_indices, :]
    fields = fields_full[:, selected_indices, :]
    wrapped_positions = positions.copy()

    if metadata_json is None:
        raise ValueError(
            "Space-time correlation analysis requires timeseries_metadata so unwrapped "
            "spacecraft trajectories can be reconstructed"
        )
    positions, geometry_metadata = _reconstruct_unwrapped_positions_from_metadata(
        times=times,
        metadata_json=metadata_json,
        selected_labels=selected_labels,
        available_labels=available_labels,
    )

    metadata = {
        "input_mode": "interpolated_timeseries",
        "spacecraft_labels": selected_labels,
        "available_spacecraft_labels": available_labels,
        "field_component": "bx",
        "position_units": "rho_p",
        "time_units": "seconds",
        **geometry_metadata,
        "input": {
            "timeseries_csv": str(csv_path),
            "timeseries_metadata": None if metadata_path is None else str(metadata_path),
        },
    }
    if metadata_json is not None:
        metadata["timeseries_metadata"] = metadata_json

    return SpaceTimeCorrelationInput(
        times=times,
        positions=positions,
        wrapped_positions=wrapped_positions,
        fields=fields,
        spacecraft_labels=selected_labels,
        metadata=metadata,
    )


def iter_space_time_correlation_samples(
    positions: np.ndarray,
    fields: np.ndarray,
    *,
    spacecraft_labels: list[str] | None = None,
    times: np.ndarray,
    component: str = "bx",
    max_lag_steps: int | None = None,
    max_lag_fraction: float = 0.5,
):
    """
    Yield raw `(r, tau, normalized_delta_bx_product)` sample chunks for the multipoint map.

    Pair semantics in v1:
    - self-pairs `(i, i)` are included
    - cross-spacecraft pairs use a single unordered convention `i < j`
    - symmetric duplicates `(j, i)` are not added separately

    Each yielded chunk corresponds to one integer lag `k`, with `tau = k * dt`.
    """

    component_index = _validate_component(component)
    positions = np.asarray(positions, dtype=float)
    fields = np.asarray(fields, dtype=float)
    times = np.asarray(times, dtype=float)

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must be shaped (n_steps, n_spacecraft, 3)")
    if fields.ndim != 3 or fields.shape[-1] != 3:
        raise ValueError("fields must be shaped (n_steps, n_spacecraft, 3)")
    if positions.shape != fields.shape:
        raise ValueError("positions and fields must have matching shapes")
    if len(times) != positions.shape[0]:
        raise ValueError("times length must match the first dimension of positions")
    if spacecraft_labels is None:
        spacecraft_labels = [f"S{index}" for index in range(positions.shape[1])]
    if len(spacecraft_labels) != positions.shape[1]:
        raise ValueError("spacecraft_labels length must match the spacecraft dimension")

    dt_seconds = _resolve_uniform_cadence(times)
    resolved_max_lag_steps = _resolve_max_lag_steps(
        positions.shape[0],
        max_lag_steps=max_lag_steps,
        max_lag_fraction=max_lag_fraction,
    )
    fluctuation_stats = _compute_component_fluctuation_statistics(
        fields,
        component_index=component_index,
        spacecraft_labels=list(spacecraft_labels),
    )
    fluctuations = fluctuation_stats["fluctuations"]
    stddevs = fluctuation_stats["stddevs"]

    pair_indices = tuple(_iter_unordered_pairs_with_self(positions.shape[1]))
    for lag_step in range(resolved_max_lag_steps + 1):
        left_count = positions.shape[0] - lag_step
        base_positions = positions[:left_count]
        lagged_positions = positions[lag_step:]
        base_fluctuations = fluctuations[:left_count]
        lagged_fluctuations = fluctuations[lag_step:]

        r_chunks = []
        normalized_product_chunks = []
        for left_index, right_index in pair_indices:
            separation = lagged_positions[:, right_index, :] - base_positions[:, left_index, :]
            r_chunks.append(np.linalg.norm(separation, axis=1))
            normalized_product_chunks.append(
                (
                    base_fluctuations[:, left_index]
                    * lagged_fluctuations[:, right_index]
                )
                / (stddevs[left_index] * stddevs[right_index])
            )

        if r_chunks:
            r_values = np.concatenate(r_chunks, axis=0)
            normalized_product_values = np.concatenate(normalized_product_chunks, axis=0)
        else:
            r_values = np.empty(0, dtype=float)
            normalized_product_values = np.empty(0, dtype=float)

        finite_mask = np.isfinite(r_values) & np.isfinite(normalized_product_values)
        tau_value = float(lag_step * dt_seconds)
        yield {
            "lag_step": lag_step,
            "tau": tau_value,
            "r": r_values[finite_mask],
            "normalized_product": normalized_product_values[finite_mask],
            "sample_count": int(np.sum(finite_mask)),
        }


def _resolve_default_tau_edges(*, dt_seconds: float, max_lag_steps: int) -> np.ndarray:
    tau_centers = np.arange(max_lag_steps + 1, dtype=float) * dt_seconds
    return _build_centered_linear_edges(tau_centers, spacing=dt_seconds)


def compute_space_time_correlation(
    prepared_input: SpaceTimeCorrelationInput,
    *,
    component: str = "bx",
    max_lag_steps: int | None = None,
    max_lag_fraction: float = 0.5,
    r_bin_edges: np.ndarray | None = None,
    tau_bin_edges: np.ndarray | None = None,
    n_r_bins: int | None = None,
    n_tau_bins: int | None = None,
    min_count_threshold: int | None = None,
    min_count_fraction: float = 0.01,
) -> SpaceTimeCorrelationResult:
    """
    Compute a normalized multipoint space-time correlation map from interpolated time series.

    The implemented v1 definition is:

    `C(r, tau) = < bx_i(t) * bx_j(t + tau) > / < bx^2 >`

    where:
    - `tau` is restricted to integer multiples of the native cadence
    - `r = |x_j(t + tau) - x_i(t)|`
    - self-pairs are included
    - cross-spacecraft pairs are counted once using the unordered convention `i < j`
    - samples are grouped into linear bins in `r` and `tau`
    """

    if not isinstance(prepared_input, SpaceTimeCorrelationInput):
        raise TypeError("prepared_input must be a SpaceTimeCorrelationInput instance")
    component_index = _validate_component(component)
    positions = np.asarray(prepared_input.positions, dtype=float)
    fields = np.asarray(prepared_input.fields, dtype=float)
    times = np.asarray(prepared_input.times, dtype=float)
    fluctuation_stats = _compute_component_fluctuation_statistics(
        fields,
        component_index=component_index,
        spacecraft_labels=prepared_input.spacecraft_labels,
    )

    dt_seconds = _resolve_uniform_cadence(times)
    resolved_max_lag_steps = _resolve_max_lag_steps(
        len(times),
        max_lag_steps=max_lag_steps,
        max_lag_fraction=max_lag_fraction,
    )

    observed_r_max = 0.0
    sample_count_total = 0
    for chunk in iter_space_time_correlation_samples(
        positions,
        fields,
        spacecraft_labels=prepared_input.spacecraft_labels,
        times=times,
        component=component,
        max_lag_steps=resolved_max_lag_steps,
        max_lag_fraction=max_lag_fraction,
    ):
        if chunk["sample_count"] <= 0:
            continue
        observed_r_max = max(observed_r_max, float(np.max(chunk["r"])))
        sample_count_total += int(chunk["sample_count"])

    if sample_count_total == 0:
        raise ValueError("No finite space-time correlation samples were produced")

    resolved_r_edges = _resolve_linear_bin_edges(
        values_min=0.0,
        values_max=observed_r_max,
        n_bins=n_r_bins,
        explicit_edges=r_bin_edges,
        default_bins=24,
    )
    resolved_tau_edges = _resolve_linear_bin_edges(
        values_min=float(_resolve_default_tau_edges(dt_seconds=dt_seconds, max_lag_steps=resolved_max_lag_steps)[0]),
        values_max=float(_resolve_default_tau_edges(dt_seconds=dt_seconds, max_lag_steps=resolved_max_lag_steps)[-1]),
        n_bins=n_tau_bins,
        explicit_edges=tau_bin_edges,
        default_bins=resolved_max_lag_steps + 1,
    )

    n_tau = len(resolved_tau_edges) - 1
    n_r = len(resolved_r_edges) - 1
    counts = np.zeros((n_tau, n_r), dtype=int)
    normalized_product_sums = np.zeros((n_tau, n_r), dtype=float)

    for chunk in iter_space_time_correlation_samples(
        positions,
        fields,
        spacecraft_labels=prepared_input.spacecraft_labels,
        times=times,
        component=component,
        max_lag_steps=resolved_max_lag_steps,
        max_lag_fraction=max_lag_fraction,
    ):
        if chunk["sample_count"] <= 0:
            continue
        tau_values = np.full(chunk["sample_count"], chunk["tau"], dtype=float)
        r_index = np.searchsorted(resolved_r_edges, chunk["r"], side="right") - 1
        tau_index = np.searchsorted(resolved_tau_edges, tau_values, side="right") - 1
        in_range_mask = (
            (r_index >= 0)
            & (r_index < n_r)
            & (tau_index >= 0)
            & (tau_index < n_tau)
        )
        if not np.any(in_range_mask):
            continue

        flat_index = tau_index[in_range_mask] * n_r + r_index[in_range_mask]
        counts += np.bincount(flat_index, minlength=n_tau * n_r).reshape(n_tau, n_r)
        normalized_product_sums += np.bincount(
            flat_index,
            weights=chunk["normalized_product"][in_range_mask],
            minlength=n_tau * n_r,
        ).reshape(n_tau, n_r)

    correlation = np.full((n_tau, n_r), np.nan, dtype=float)
    populated_mask = counts > 0
    correlation[populated_mask] = (
        normalized_product_sums[populated_mask] / counts[populated_mask]
    )

    counts_max = int(np.max(counts)) if counts.size else 0
    if min_count_threshold is not None:
        if min_count_threshold < 0:
            raise ValueError(
                f"min_count_threshold must be non-negative; got {min_count_threshold}"
            )
        resolved_min_count_threshold = int(min_count_threshold)
        threshold_mode = "absolute"
    else:
        if min_count_fraction < 0.0:
            raise ValueError(
                f"min_count_fraction must be non-negative; got {min_count_fraction}"
            )
        resolved_min_count_threshold = int(np.ceil(float(min_count_fraction) * counts_max))
        threshold_mode = "fraction_of_max"
    valid_mask = populated_mask & (counts >= resolved_min_count_threshold)

    r_centers = 0.5 * (resolved_r_edges[:-1] + resolved_r_edges[1:])
    tau_centers = 0.5 * (resolved_tau_edges[:-1] + resolved_tau_edges[1:])
    raw_cloud_metadata = {
        "generation_mode": "streamed_two_pass",
        "tau_definition": "integer_multiples_of_native_cadence",
        "r_definition": "norm(x_j(t+tau) - x_i(t))",
        "position_geometry": prepared_input.metadata.get("position_geometry"),
        "position_geometry_source": prepared_input.metadata.get("position_geometry_source"),
        "pair_semantics": "self_pairs_included_cross_pairs_counted_once_with_i_lt_j",
        "sample_definition": "delta_bx_i(t) * delta_bx_j(t+tau) / (sigma_i * sigma_j)",
        "sample_count_total": int(sample_count_total),
    }
    metadata = {
        "analysis": {
            "field_component": component,
            "correlation_definition": (
                "mean(delta_bx_i(t) * delta_bx_j(t+tau) / (sigma_i * sigma_j)) "
                "within each linear (r, tau) bin"
            ),
            "delta_bx_definition": "per_spacecraft_mean_subtracted_over_analyzed_record",
            "sigma_definition": "per_spacecraft_std_of_delta_bx_over_analyzed_record",
            "bin_statistic": "mean_of_pre_normalized_fluctuation_products_within_each_linear_(r,tau)_bin",
            "normalization_choice": "per_spacecraft_standard_deviation",
            "binning": "linear_in_r_and_tau",
            "r_units": "rho_p",
            "tau_units": "seconds",
            "dt_seconds": dt_seconds,
            "max_lag_steps": int(resolved_max_lag_steps),
            "max_lag_seconds": float(resolved_max_lag_steps * dt_seconds),
            "n_r_bins": int(n_r),
            "n_tau_bins": int(n_tau),
            "minimum_count_threshold_mode": threshold_mode,
            "minimum_count_fraction": None if min_count_threshold is not None else float(min_count_fraction),
            "minimum_count_threshold": int(resolved_min_count_threshold),
            "counts_max": counts_max,
            "valid_bin_fraction": (
                float(np.sum(valid_mask)) / float(valid_mask.size)
                if valid_mask.size
                else 0.0
            ),
        },
        "input": dict(prepared_input.metadata),
        "fluctuation_statistics": {
            "spacecraft_labels": list(prepared_input.spacecraft_labels),
            "means": fluctuation_stats["means"].tolist(),
            "variances": fluctuation_stats["variances"].tolist(),
            "stddevs": fluctuation_stats["stddevs"].tolist(),
        },
        "raw_cloud": raw_cloud_metadata,
        "masking": {
            "valid_mask_shape": list(valid_mask.shape),
            "invalid_bins_zero_count": int(np.sum(~populated_mask)),
            "invalid_bins_below_threshold": int(np.sum(populated_mask & ~valid_mask)),
            "valid_bins": int(np.sum(valid_mask)),
        },
    }

    return SpaceTimeCorrelationResult(
        r_bin_edges=resolved_r_edges,
        r_bin_centers=r_centers,
        tau_bin_edges=resolved_tau_edges,
        tau_bin_centers=tau_centers,
        counts=counts,
        correlation=correlation,
        valid_mask=valid_mask,
        metadata=metadata,
    )


def compute_hub_autocorrelation(
    prepared_input: SpaceTimeCorrelationInput,
    *,
    component: str = "bx",
    max_lag_steps: int | None = None,
    max_lag_fraction: float = 0.5,
) -> HubAutocorrelationResult:
    """Compute a Hub-only single-spacecraft autocorrelation with the same lag convention."""

    component_index = _validate_component(component)
    if "H" not in prepared_input.spacecraft_labels:
        raise ValueError("Hub autocorrelation requires spacecraft label 'H' in the prepared input")

    times = np.asarray(prepared_input.times, dtype=float)
    dt_seconds = _resolve_uniform_cadence(times)
    resolved_max_lag_steps = _resolve_max_lag_steps(
        len(times),
        max_lag_steps=max_lag_steps,
        max_lag_fraction=max_lag_fraction,
    )
    hub_index = prepared_input.spacecraft_labels.index("H")
    fluctuation_stats = _compute_component_fluctuation_statistics(
        prepared_input.fields,
        component_index=component_index,
        spacecraft_labels=prepared_input.spacecraft_labels,
    )
    delta_bx_hub = np.asarray(fluctuation_stats["fluctuations"][:, hub_index], dtype=float)
    sigma_hub = float(fluctuation_stats["stddevs"][hub_index])

    tau_values = np.arange(resolved_max_lag_steps + 1, dtype=float) * dt_seconds
    counts = np.zeros(resolved_max_lag_steps + 1, dtype=int)
    correlation = np.full(resolved_max_lag_steps + 1, np.nan, dtype=float)

    for lag_step in range(resolved_max_lag_steps + 1):
        products = (
            delta_bx_hub[: len(delta_bx_hub) - lag_step]
            * delta_bx_hub[lag_step:]
        ) / (sigma_hub**2)
        finite_mask = np.isfinite(products)
        counts[lag_step] = int(np.sum(finite_mask))
        if counts[lag_step] > 0:
            correlation[lag_step] = float(np.mean(products[finite_mask]))

    return HubAutocorrelationResult(
        tau_values=tau_values,
        counts=counts,
        correlation=correlation,
        metadata={
            "field_component": component,
            "spacecraft_label": "H",
            "delta_bx_definition": "per_spacecraft_mean_subtracted_over_analyzed_record",
            "sigma_definition": "per_spacecraft_std_of_delta_bx_over_analyzed_record",
            "dt_seconds": dt_seconds,
            "max_lag_steps": int(resolved_max_lag_steps),
            "max_lag_seconds": float(resolved_max_lag_steps * dt_seconds),
            "normalization_choice": "hub_variance_from_mean_removed_hub_series",
            "mean": float(fluctuation_stats["means"][hub_index]),
            "variance": float(fluctuation_stats["variances"][hub_index]),
            "stddev": sigma_hub,
        },
    )


def interpolate_correlation_map(
    result: SpaceTimeCorrelationResult,
    *,
    n_r: int = 200,
    n_tau: int = 200,
) -> dict[str, np.ndarray]:
    """Interpolate the masked correlation map onto a finer regular grid."""

    if len(result.r_bin_centers) < 2 or len(result.tau_bin_centers) < 2:
        raise ValueError("At least two populated bins along each axis are required for interpolation")

    masked_values = np.where(result.valid_mask, result.correlation, np.nan)
    interpolator = RegularGridInterpolator(
        (result.tau_bin_centers, result.r_bin_centers),
        masked_values,
        bounds_error=False,
        fill_value=np.nan,
    )
    tau_fine = np.linspace(result.tau_bin_centers[0], result.tau_bin_centers[-1], n_tau)
    r_fine = np.linspace(result.r_bin_centers[0], result.r_bin_centers[-1], n_r)
    tau_mesh, r_mesh = np.meshgrid(tau_fine, r_fine, indexing="ij")
    fine_values = interpolator(np.column_stack([tau_mesh.ravel(), r_mesh.ravel()])).reshape(
        n_tau,
        n_r,
    )
    return {
        "tau_values": tau_fine,
        "r_values": r_fine,
        "correlation": fine_values,
    }


def estimate_decorrelation_contour(
    result: SpaceTimeCorrelationResult,
    *,
    level: float = float(np.exp(-1.0)),
    interpolation_shape: tuple[int, int] = (200, 200),
) -> dict | None:
    """Estimate the longest available decorrelation contour from the masked map."""

    if (
        not np.any(result.valid_mask)
        or len(result.r_bin_centers) < 2
        or len(result.tau_bin_centers) < 2
    ):
        return None
    interpolated = interpolate_correlation_map(
        result,
        n_r=int(interpolation_shape[1]),
        n_tau=int(interpolation_shape[0]),
    )
    contour_figure, contour_ax = plt.subplots()
    contour_set = contour_ax.contour(
        interpolated["tau_values"],
        interpolated["r_values"],
        interpolated["correlation"].T,
        levels=[level],
    )
    plt.close(contour_figure)
    segments = contour_set.allsegs[0] if contour_set.allsegs else []
    segments = [np.asarray(segment, dtype=float) for segment in segments if len(segment) > 1]
    if not segments:
        return None

    vertices = max(segments, key=len)
    return {
        "level": float(level),
        "tau_values": vertices[:, 0].tolist(),
        "r_values": vertices[:, 1].tolist(),
        "interpolation_shape": [int(interpolation_shape[0]), int(interpolation_shape[1])],
    }


def extract_axis_cuts(result: SpaceTimeCorrelationResult) -> dict:
    """Extract the nearest available `C(r, 0)` and `C(0, tau)` cuts from the binned map."""

    tau_index = int(np.argmin(np.abs(result.tau_bin_centers)))
    r_index = int(np.argmin(np.abs(result.r_bin_centers)))
    return {
        "spatial_cut": {
            "r_values": result.r_bin_centers.tolist(),
            "correlation": result.correlation[tau_index].tolist(),
            "counts": result.counts[tau_index].tolist(),
            "valid_mask": result.valid_mask[tau_index].tolist(),
            "selected_tau_index": tau_index,
            "selected_tau_value": float(result.tau_bin_centers[tau_index]),
        },
        "temporal_cut": {
            "tau_values": result.tau_bin_centers.tolist(),
            "correlation": result.correlation[:, r_index].tolist(),
            "counts": result.counts[:, r_index].tolist(),
            "valid_mask": result.valid_mask[:, r_index].tolist(),
            "selected_r_index": r_index,
            "selected_r_value": float(result.r_bin_centers[r_index]),
        },
    }


def _estimate_first_crossing(
    x_values: np.ndarray,
    y_values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    level: float,
) -> float | None:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    finite_mask = valid_mask & np.isfinite(x_values) & np.isfinite(y_values)
    x_valid = x_values[finite_mask]
    y_valid = y_values[finite_mask]
    if len(x_valid) < 2:
        return None

    for index in range(1, len(x_valid)):
        left_y = y_valid[index - 1]
        right_y = y_valid[index]
        if left_y == level:
            return float(x_valid[index - 1])
        if (left_y - level) * (right_y - level) > 0.0:
            continue
        if np.isclose(right_y, left_y):
            return float(x_valid[index])
        fraction = (level - left_y) / (right_y - left_y)
        return float(x_valid[index - 1] + fraction * (x_valid[index] - x_valid[index - 1]))
    return None


def estimate_decorrelation_scales_from_cuts(
    result: SpaceTimeCorrelationResult,
    *,
    level: float = float(np.exp(-1.0)),
) -> dict:
    """Estimate 1/e-like decorrelation scales from the nearest spatial and temporal axis cuts."""

    cuts = extract_axis_cuts(result)
    spatial_cut = cuts["spatial_cut"]
    temporal_cut = cuts["temporal_cut"]
    return {
        "level": float(level),
        "decorrelation_length": _estimate_first_crossing(
            np.asarray(spatial_cut["r_values"], dtype=float),
            np.asarray(spatial_cut["correlation"], dtype=float),
            np.asarray(spatial_cut["valid_mask"], dtype=bool),
            level=level,
        ),
        "decorrelation_time": _estimate_first_crossing(
            np.asarray(temporal_cut["tau_values"], dtype=float),
            np.asarray(temporal_cut["correlation"], dtype=float),
            np.asarray(temporal_cut["valid_mask"], dtype=bool),
            level=level,
        ),
        "axis_cut_reference": {
            "tau_for_spatial_cut": spatial_cut["selected_tau_value"],
            "r_for_temporal_cut": temporal_cut["selected_r_value"],
        },
    }


def plot_space_time_correlation(
    result: SpaceTimeCorrelationResult,
    *,
    hub_autocorrelation: HubAutocorrelationResult | None = None,
    show_contour: bool = True,
    contour_level: float = float(np.exp(-1.0)),
    title: str = "",
):
    """Render a Fig.-13-like diagnostic with the 2D map and Hub-only autocorrelation."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    map_ax, auto_ax = axes

    masked_map = np.ma.masked_where(~result.valid_mask, result.correlation)
    finite_values = masked_map.compressed()
    if finite_values.size:
        value_limit = float(np.max(np.abs(finite_values)))
        if value_limit == 0.0:
            value_limit = 1.0
        norm = TwoSlopeNorm(vmin=-value_limit, vcenter=0.0, vmax=value_limit)
    else:
        norm = None

    image = map_ax.pcolormesh(
        result.tau_bin_edges,
        result.r_bin_edges,
        masked_map.T,
        shading="auto",
        cmap="RdBu_r",
        norm=norm,
    )
    fig.colorbar(image, ax=map_ax, shrink=0.88, label=r"$C(r, \tau)$")
    map_ax.set_xlabel(r"$\tau$ (s)")
    map_ax.set_ylabel(r"$r$ ($\rho_p$)")
    map_ax.set_title("Multipoint Space-Time Correlation")
    map_ax.grid(alpha=0.25)

    if show_contour and np.any(result.valid_mask):
        contour = estimate_decorrelation_contour(result, level=contour_level)
        if contour is not None:
            map_ax.plot(
                contour["tau_values"],
                contour["r_values"],
                color="#111111",
                linewidth=1.5,
                label=fr"$C = e^{{-1}}$",
            )
            map_ax.legend(loc="best")

    if hub_autocorrelation is None:
        auto_ax.set_title("Hub Autocorrelation")
    else:
        valid_hub = np.isfinite(hub_autocorrelation.correlation) & (hub_autocorrelation.counts > 0)
        auto_ax.plot(
            hub_autocorrelation.tau_values[valid_hub],
            hub_autocorrelation.correlation[valid_hub],
            color="#0072B2",
            marker="o",
            linewidth=1.8,
        )
        auto_ax.axhline(contour_level, color="#666666", linestyle="--", linewidth=1.2)
        auto_ax.set_title("Hub-Only Autocorrelation")
    auto_ax.set_xlabel(r"$\tau$ (s)")
    auto_ax.set_ylabel(r"$C_H(\tau)$")
    auto_ax.grid(alpha=0.3)

    if title:
        fig.suptitle(title)
    return fig, axes
