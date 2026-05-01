from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_json(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_selected_labels(
    requested_labels: list[str] | tuple[str, ...] | None,
    *,
    available_labels: list[str],
) -> list[str]:
    if requested_labels is None:
        return list(available_labels)
    normalized = [str(label).strip() for label in requested_labels if str(label).strip()]
    if not normalized:
        raise ValueError("spacecraft_labels must contain at least one non-empty label")
    if len(set(normalized)) != len(normalized):
        raise ValueError("spacecraft_labels must not contain duplicates")
    missing = [label for label in normalized if label not in available_labels]
    if missing:
        raise ValueError(
            "Unknown spacecraft labels requested: " + ", ".join(sorted(missing))
        )
    return [label for label in available_labels if label in normalized]


def _reconstruct_unwrapped_positions_from_metadata(
    *,
    times: np.ndarray,
    metadata_json: dict,
    selected_labels: list[str],
    available_labels: list[str],
) -> np.ndarray:
    motion_metadata = metadata_json.get("motion")
    if not isinstance(motion_metadata, dict):
        raise ValueError("timeseries metadata must contain a 'motion' block")

    initial_unwrapped = np.asarray(
        motion_metadata.get("initial_unwrapped_coords_rho_p"),
        dtype=float,
    )
    velocity_rho_p_s = np.asarray(motion_metadata.get("velocity_rho_p_s"), dtype=float)
    dt_seconds = motion_metadata.get("dt_seconds")
    n_steps = motion_metadata.get("n_steps")
    if (
        initial_unwrapped.ndim != 2
        or initial_unwrapped.shape[1] != 3
        or velocity_rho_p_s.shape != (3,)
        or dt_seconds is None
        or n_steps is None
    ):
        raise ValueError(
            "timeseries metadata motion block must include valid initial positions, velocity, "
            "dt_seconds, and n_steps"
        )
    if initial_unwrapped.shape[0] != len(available_labels):
        raise ValueError(
            "initial_unwrapped_coords_rho_p row count must match spacecraft_labels"
        )
    if len(times) != int(n_steps):
        raise ValueError(
            f"pair-product times contain {len(times)} steps but metadata declares {n_steps}"
        )
    expected_times = np.arange(len(times), dtype=float) * float(dt_seconds)
    if not np.allclose(times, expected_times):
        raise ValueError("pair-product times are inconsistent with metadata dt_seconds")

    selected_indices = np.array(
        [available_labels.index(label) for label in selected_labels],
        dtype=int,
    )
    positions = (
        initial_unwrapped[None, :, :]
        + times[:, None, None] * velocity_rho_p_s[None, None, :]
    )
    return positions[:, selected_indices, :]


def _resolve_time_index(
    times: np.ndarray,
    *,
    time_index: int | None,
    time_seconds: float | None,
) -> int:
    if time_index is not None and time_seconds is not None:
        raise ValueError("Provide either time_index or time_seconds, not both")
    if time_index is None and time_seconds is None:
        return 0
    if time_index is not None:
        resolved = int(time_index)
        if resolved < 0 or resolved >= len(times):
            raise ValueError(
                f"time_index must lie in [0, {len(times) - 1}], got {resolved}"
            )
        return resolved
    return int(np.argmin(np.abs(np.asarray(times, dtype=float) - float(time_seconds))))


def _compute_shape_metrics(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    barycenter = np.mean(vertices, axis=0)
    centered = vertices - barycenter[None, :]
    volumetric_tensor = centered.T @ centered / float(len(vertices))
    eigenvalues = np.sort(np.linalg.eigvalsh(volumetric_tensor))[::-1]
    leading = float(eigenvalues[0])
    middle = float(max(eigenvalues[1], 0.0))
    trailing = float(max(eigenvalues[2], 0.0))
    eps = float(np.finfo(float).tiny)
    if not np.isfinite(leading) or leading <= 0.0 or not np.isfinite(middle) or middle < 0.0:
        elongation = np.nan
        planarity = np.nan
    else:
        elongation = float(1.0 - np.sqrt(np.clip(middle / max(leading, eps), 0.0, 1.0)))
        if not np.isfinite(trailing) or trailing < 0.0:
            planarity = np.nan
        else:
            planarity = float(1.0 - np.sqrt(np.clip(trailing / max(middle, eps), 0.0, 1.0)))
    d_ep = float(np.sqrt(elongation**2 + planarity**2))
    return barycenter, volumetric_tensor, eigenvalues, elongation, planarity, d_ep


def _offset_overlapping_points(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    span: float,
    atol: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return display coordinates with tiny deterministic offsets for overlaps."""
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    display_x = x_array.copy()
    display_y = y_array.copy()
    offsets = np.zeros((len(x_array), 2), dtype=float)
    visited: set[int] = set()
    had_overlaps = False
    offset_radius = max(float(span), 1.0) * 0.018

    for index in range(len(x_array)):
        if index in visited:
            continue
        group = [
            other
            for other in range(len(x_array))
            if other not in visited
            and np.isclose(x_array[index], x_array[other], rtol=1.0e-9, atol=atol)
            and np.isclose(y_array[index], y_array[other], rtol=1.0e-9, atol=atol)
        ]
        for other in group:
            visited.add(other)
        if len(group) <= 1:
            continue
        had_overlaps = True
        for group_position, point_index in enumerate(group):
            angle = (2.0 * math.pi * group_position / len(group)) + (math.pi / 4.0)
            offsets[point_index, 0] = offset_radius * math.cos(angle)
            offsets[point_index, 1] = offset_radius * math.sin(angle)
            display_x[point_index] += offsets[point_index, 0]
            display_y[point_index] += offsets[point_index, 1]

    return display_x, display_y, offsets, had_overlaps


def _finite_time_average(vectors: np.ndarray) -> np.ndarray | None:
    array = np.asarray(vectors, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("Expected vector time series shaped (n_steps, 3)")
    finite_mask = np.all(np.isfinite(array), axis=1)
    if not np.any(finite_mask):
        return None
    return np.mean(array[finite_mask], axis=0)


def _compute_yaglom_timeseries(
    delta_zplus_timeseries: np.ndarray,
    delta_zminus_timeseries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dz_plus = np.asarray(delta_zplus_timeseries, dtype=float)
    dz_minus = np.asarray(delta_zminus_timeseries, dtype=float)
    plus_norm_sq = np.sum(dz_plus * dz_plus, axis=1, keepdims=True)
    minus_norm_sq = np.sum(dz_minus * dz_minus, axis=1, keepdims=True)
    return dz_minus * plus_norm_sq, dz_plus * minus_norm_sq


def _estimate_vector_gradient_on_tetrahedron(
    vertices: np.ndarray,
    values: np.ndarray,
    *,
    condition_number_max: float = 1.0e12,
) -> tuple[np.ndarray | None, str | None]:
    """Estimate grad Y for a vector field sampled at 4 tetrahedron vertices.

    Uses an affine reconstruction Y(l) = a + G (l - l0), solved exactly from the
    four vertex values when the tetrahedron geometry is non-degenerate.
    """
    vertex_array = np.asarray(vertices, dtype=float)
    field_values = np.asarray(values, dtype=float)
    if vertex_array.shape != (4, 3) or field_values.shape != (4, 3):
        raise ValueError("vertices and values must both be shaped (4, 3)")
    if not np.all(np.isfinite(vertex_array)):
        return None, "non_finite_vertices"
    if not np.all(np.isfinite(field_values)):
        return None, "non_finite_vertex_values"

    origin = vertex_array[0]
    geometry_matrix = (vertex_array[1:] - origin[None, :]).T
    try:
        condition_number = float(np.linalg.cond(geometry_matrix))
    except np.linalg.LinAlgError:
        return None, "singular_geometry"
    if not np.isfinite(condition_number) or condition_number > float(condition_number_max):
        return None, "ill_conditioned_geometry"

    value_differences = (field_values[1:] - field_values[0][None, :]).T
    try:
        gradient = value_differences @ np.linalg.inv(geometry_matrix)
    except np.linalg.LinAlgError:
        return None, "singular_geometry"
    return gradient, None


def _summarize_metric(values: np.ndarray) -> dict[str, float | None]:
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "std": float(np.std(finite_values)),
    }


@dataclass
class DirectedLagPoint:
    left_index: int
    right_index: int
    left_label: str
    right_label: str
    pair_label: str
    lag_vector: np.ndarray
    lag_magnitude: float
    key: str
    reflected_key: str
    delta_zplus: np.ndarray | None = None
    delta_zminus: np.ndarray | None = None
    delta_zplus_timeseries: np.ndarray | None = None
    delta_zminus_timeseries: np.ndarray | None = None
    y_plus_timeseries: np.ndarray | None = None
    y_minus_timeseries: np.ndarray | None = None
    y_plus: np.ndarray | None = None
    y_minus: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "left_index": self.left_index,
            "right_index": self.right_index,
            "left_label": self.left_label,
            "right_label": self.right_label,
            "pair_label": self.pair_label,
            "lag_vector": np.asarray(self.lag_vector, dtype=float).tolist(),
            "lag_magnitude": float(self.lag_magnitude),
            "key": self.key,
            "reflected_key": self.reflected_key,
            "delta_zplus": (
                None if self.delta_zplus is None else np.asarray(self.delta_zplus, dtype=float).tolist()
            ),
            "delta_zminus": (
                None if self.delta_zminus is None else np.asarray(self.delta_zminus, dtype=float).tolist()
            ),
            "delta_zplus_timeseries_shape": (
                None
                if self.delta_zplus_timeseries is None
                else list(np.asarray(self.delta_zplus_timeseries, dtype=float).shape)
            ),
            "delta_zminus_timeseries_shape": (
                None
                if self.delta_zminus_timeseries is None
                else list(np.asarray(self.delta_zminus_timeseries, dtype=float).shape)
            ),
            "y_plus": None if self.y_plus is None else np.asarray(self.y_plus, dtype=float).tolist(),
            "y_minus": None if self.y_minus is None else np.asarray(self.y_minus, dtype=float).tolist(),
        }


@dataclass
class LagTetrahedron:
    points: tuple[DirectedLagPoint, DirectedLagPoint, DirectedLagPoint, DirectedLagPoint]
    pair_labels: tuple[str, str, str, str]
    vertex_array: np.ndarray
    barycenter: np.ndarray
    volumetric_tensor: np.ndarray
    eigenvalues: np.ndarray
    elongation: float
    planarity: float
    d_ep: float
    is_redundant: bool
    is_zero_barycenter: bool
    passes_quality_cut: bool
    mesocenter_magnitude: float | None = None
    y_plus_vertices: np.ndarray | None = None
    y_minus_vertices: np.ndarray | None = None
    grad_y_plus: np.ndarray | None = None
    grad_y_minus: np.ndarray | None = None
    div_y_plus: float | None = None
    div_y_minus: float | None = None
    epsilon_plus: float | None = None
    epsilon_minus: float | None = None
    estimated_gradients: np.ndarray | None = None
    future_divergence: float | None = None
    is_valid_for_gradient: bool = False
    invalid_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "point_keys": [point.key for point in self.points],
            "pair_labels": list(self.pair_labels),
            "vertex_array": np.asarray(self.vertex_array, dtype=float).tolist(),
            "barycenter": np.asarray(self.barycenter, dtype=float).tolist(),
            "volumetric_tensor": np.asarray(self.volumetric_tensor, dtype=float).tolist(),
            "eigenvalues": np.asarray(self.eigenvalues, dtype=float).tolist(),
            "elongation": float(self.elongation),
            "planarity": float(self.planarity),
            "d_ep": float(self.d_ep),
            "is_redundant": bool(self.is_redundant),
            "is_zero_barycenter": bool(self.is_zero_barycenter),
            "passes_quality_cut": bool(self.passes_quality_cut),
            "mesocenter_magnitude": self.mesocenter_magnitude,
            "y_plus_vertices": (
                None
                if self.y_plus_vertices is None
                else np.asarray(self.y_plus_vertices, dtype=float).tolist()
            ),
            "y_minus_vertices": (
                None
                if self.y_minus_vertices is None
                else np.asarray(self.y_minus_vertices, dtype=float).tolist()
            ),
            "grad_y_plus": (
                None if self.grad_y_plus is None else np.asarray(self.grad_y_plus, dtype=float).tolist()
            ),
            "grad_y_minus": (
                None if self.grad_y_minus is None else np.asarray(self.grad_y_minus, dtype=float).tolist()
            ),
            "div_y_plus": self.div_y_plus,
            "div_y_minus": self.div_y_minus,
            "epsilon_plus": self.epsilon_plus,
            "epsilon_minus": self.epsilon_minus,
            "estimated_gradients": (
                None
                if self.estimated_gradients is None
                else np.asarray(self.estimated_gradients, dtype=float).tolist()
            ),
            "future_divergence": self.future_divergence,
            "is_valid_for_gradient": bool(self.is_valid_for_gradient),
            "invalid_reason": self.invalid_reason,
        }


@dataclass
class LagTetrahedraInput:
    time_index: int
    time_seconds: float
    spacecraft_labels: list[str]
    spacecraft_positions: np.ndarray
    unordered_pair_indices: np.ndarray
    unordered_pair_labels: np.ndarray
    separation_vectors: np.ndarray
    separation_magnitudes: np.ndarray
    delta_zplus_timeseries: np.ndarray
    delta_zminus_timeseries: np.ndarray
    delta_zplus: np.ndarray
    delta_zminus: np.ndarray
    metadata: dict


@dataclass
class LagTetrahedraResult:
    spacecraft_labels: list[str]
    selected_time_index: int
    selected_time_seconds: float
    spacecraft_positions: np.ndarray
    unordered_baselines: list[dict]
    directed_lag_points: list[DirectedLagPoint]
    raw_tetrahedra: int
    removed_redundant: int
    removed_zero_barycenter: int
    retained_tetrahedra: list[LagTetrahedron]
    summary_statistics: dict
    metadata: dict

    def to_dict(self) -> dict:
        preview_limit = int(self.metadata.get("tetrahedron_preview_limit", 64))
        preview = [tetra.to_dict() for tetra in self.retained_tetrahedra[:preview_limit]]
        return {
            "spacecraft_labels": list(self.spacecraft_labels),
            "selected_time_index": int(self.selected_time_index),
            "selected_time_seconds": float(self.selected_time_seconds),
            "spacecraft_positions": np.asarray(self.spacecraft_positions, dtype=float).tolist(),
            "unordered_baselines": list(self.unordered_baselines),
            "directed_lag_points": [point.to_dict() for point in self.directed_lag_points],
            "counts": {
                "unordered_baselines": len(self.unordered_baselines),
                "directed_lag_points": len(self.directed_lag_points),
                "raw_tetrahedra": int(self.raw_tetrahedra),
                "removed_redundant": int(self.removed_redundant),
                "removed_zero_barycenter": int(self.removed_zero_barycenter),
                "retained_tetrahedra": len(self.retained_tetrahedra),
            },
            "summary_statistics": self.summary_statistics,
            "epsilon_plus": [
                tetra.epsilon_plus for tetra in self.retained_tetrahedra
            ],
            "epsilon_minus": [
                tetra.epsilon_minus for tetra in self.retained_tetrahedra
            ],
            "mesocenters": [
                np.asarray(tetra.barycenter, dtype=float).tolist()
                for tetra in self.retained_tetrahedra
            ],
            "mesocenter_magnitudes": [
                tetra.mesocenter_magnitude for tetra in self.retained_tetrahedra
            ],
            "quality_cut_mask": [
                bool(tetra.passes_quality_cut and tetra.is_valid_for_gradient)
                for tetra in self.retained_tetrahedra
            ],
            "lag_point_y_plus": {
                point.key: (
                    None if point.y_plus is None else np.asarray(point.y_plus, dtype=float).tolist()
                )
                for point in self.directed_lag_points
            },
            "lag_point_y_minus": {
                point.key: (
                    None if point.y_minus is None else np.asarray(point.y_minus, dtype=float).tolist()
                )
                for point in self.directed_lag_points
            },
            "retained_tetrahedra_preview": preview,
            "retained_tetrahedra_preview_limit": preview_limit,
            "retained_tetrahedra_truncated": len(self.retained_tetrahedra) > preview_limit,
            "metadata": self.metadata,
        }


def prepare_saved_elsasser_lag_tetrahedra_input(
    metadata_path: str | Path,
    elsasser_pairs_npz_path: str | Path,
    *,
    elsasser_pairs_json_path: str | Path | None = None,
    spacecraft_labels: list[str] | tuple[str, ...] | None = None,
    time_index: int | None = None,
    time_seconds: float | None = None,
) -> LagTetrahedraInput:
    metadata_path = Path(metadata_path)
    npz_path = Path(elsasser_pairs_npz_path)
    metadata_json = _load_json(metadata_path)
    pair_json = _load_json(elsasser_pairs_json_path)
    if metadata_json is None:
        raise ValueError("metadata_path is required to prepare lag tetrahedra input")

    payload = np.load(npz_path, allow_pickle=False)
    required_arrays = {
        "times",
        "pair_indices",
        "pair_labels",
        "separation_vectors",
        "separation_magnitudes",
        "delta_zplus",
        "delta_zminus",
    }
    missing = sorted(required_arrays - set(payload.files))
    if missing:
        raise ValueError(
            f"Elsasser pair product {npz_path} is missing required arrays: {missing}"
        )

    available_labels = list(metadata_json.get("helioswarm", {}).get("spacecraft_labels", []))
    if not available_labels:
        raise ValueError("timeseries metadata must include helioswarm.spacecraft_labels")
    selected_labels = _resolve_selected_labels(
        spacecraft_labels,
        available_labels=available_labels,
    )

    times = np.asarray(payload["times"], dtype=float)
    selected_time_index = _resolve_time_index(
        times,
        time_index=time_index,
        time_seconds=time_seconds,
    )
    positions = _reconstruct_unwrapped_positions_from_metadata(
        times=times,
        metadata_json=metadata_json,
        selected_labels=selected_labels,
        available_labels=available_labels,
    )

    full_pair_indices = np.asarray(payload["pair_indices"], dtype=int)
    full_pair_labels = np.asarray(payload["pair_labels"]).astype(str)
    full_separation_vectors = np.asarray(payload["separation_vectors"], dtype=float)
    full_separation_magnitudes = np.asarray(payload["separation_magnitudes"], dtype=float)
    full_delta_zplus = np.asarray(payload["delta_zplus"], dtype=float)
    full_delta_zminus = np.asarray(payload["delta_zminus"], dtype=float)

    selected_full_indices = np.array(
        [available_labels.index(label) for label in selected_labels],
        dtype=int,
    )
    full_to_selected = {int(full_index): local_index for local_index, full_index in enumerate(selected_full_indices)}
    selected_index_set = set(int(index) for index in selected_full_indices)
    keep_mask = np.array(
        [
            int(left_index) in selected_index_set and int(right_index) in selected_index_set
            for left_index, right_index in full_pair_indices
        ],
        dtype=bool,
    )
    pair_indices = np.asarray(
        [
            [full_to_selected[int(left_index)], full_to_selected[int(right_index)]]
            for left_index, right_index in full_pair_indices[keep_mask]
        ],
        dtype=int,
    )
    pair_labels = full_pair_labels[keep_mask]

    return LagTetrahedraInput(
        time_index=selected_time_index,
        time_seconds=float(times[selected_time_index]),
        spacecraft_labels=selected_labels,
        spacecraft_positions=positions[selected_time_index],
        unordered_pair_indices=pair_indices,
        unordered_pair_labels=pair_labels,
        separation_vectors=full_separation_vectors[selected_time_index, keep_mask, :],
        separation_magnitudes=full_separation_magnitudes[selected_time_index, keep_mask],
        delta_zplus_timeseries=full_delta_zplus[:, keep_mask, :],
        delta_zminus_timeseries=full_delta_zminus[:, keep_mask, :],
        delta_zplus=full_delta_zplus[selected_time_index, keep_mask, :],
        delta_zminus=full_delta_zminus[selected_time_index, keep_mask, :],
        metadata={
            "input_mode": "saved_helioswarm_elsasser_pairs",
            "available_spacecraft_labels": available_labels,
            "spacecraft_labels": selected_labels,
            "helioswarm": metadata_json.get("helioswarm"),
            "selected_time_index": int(selected_time_index),
            "selected_time_seconds": float(times[selected_time_index]),
            "input": {
                "timeseries_metadata": str(metadata_path),
                "elsasser_pairs_npz": str(npz_path),
                "elsasser_pairs_json": None if elsasser_pairs_json_path is None else str(elsasser_pairs_json_path),
            },
            "pair_json_metadata": pair_json,
            "position_geometry": "unwrapped_physical_trajectory",
            "pair_convention_in_saved_product": "unordered_pairs_with_i_lt_j_stored_as_j_minus_i",
            "time_averaging_interval": {
                "n_steps": int(len(times)),
                "start_seconds": float(times[0]),
                "end_seconds": float(times[-1]),
            },
        },
    )


def construct_lag_tetrahedra(
    prepared_input: LagTetrahedraInput,
    *,
    zero_barycenter_atol: float = 1e-12,
    max_d_ep: float | None = None,
    tetrahedron_preview_limit: int = 64,
) -> LagTetrahedraResult:
    if len(prepared_input.spacecraft_labels) < 4:
        raise ValueError("At least four spacecraft are required to build lag tetrahedra")

    spacecraft_positions = np.asarray(prepared_input.spacecraft_positions, dtype=float)
    pair_indices = np.asarray(prepared_input.unordered_pair_indices, dtype=int)
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError("unordered_pair_indices must be shaped (n_pairs, 2)")

    unordered_baselines: list[dict] = []
    directed_lag_points: list[DirectedLagPoint] = []
    reflection_map: list[int] = []

    for pair_index, ((left_index, right_index), pair_label, separation_vector, separation_magnitude, delta_zplus, delta_zminus) in enumerate(
        zip(
            pair_indices,
            prepared_input.unordered_pair_labels,
            prepared_input.separation_vectors,
            prepared_input.separation_magnitudes,
            prepared_input.delta_zplus,
            prepared_input.delta_zminus,
        )
    ):
        delta_zplus_timeseries = np.asarray(
            prepared_input.delta_zplus_timeseries[:, pair_index, :],
            dtype=float,
        )
        delta_zminus_timeseries = np.asarray(
            prepared_input.delta_zminus_timeseries[:, pair_index, :],
            dtype=float,
        )
        left_label = prepared_input.spacecraft_labels[int(left_index)]
        right_label = prepared_input.spacecraft_labels[int(right_index)]
        unordered_baselines.append(
            {
                "left_index": int(left_index),
                "right_index": int(right_index),
                "left_label": left_label,
                "right_label": right_label,
                "pair_label": str(pair_label),
                "separation_vector_saved": np.asarray(separation_vector, dtype=float).tolist(),
                "separation_magnitude": float(separation_magnitude),
                "delta_zplus_saved": np.asarray(delta_zplus, dtype=float).tolist(),
                "delta_zminus_saved": np.asarray(delta_zminus, dtype=float).tolist(),
            }
        )

        reverse_key = f"{left_label}__{right_label}"
        canonical_key = f"{right_label}__{left_label}"
        canonical_point = DirectedLagPoint(
            left_index=int(right_index),
            right_index=int(left_index),
            left_label=right_label,
            right_label=left_label,
            pair_label=str(pair_label),
            lag_vector=np.asarray(separation_vector, dtype=float),
            lag_magnitude=float(separation_magnitude),
            key=canonical_key,
            reflected_key=reverse_key,
            delta_zplus=np.asarray(delta_zplus, dtype=float),
            delta_zminus=np.asarray(delta_zminus, dtype=float),
            delta_zplus_timeseries=delta_zplus_timeseries,
            delta_zminus_timeseries=delta_zminus_timeseries,
        )
        reflected_point = DirectedLagPoint(
            left_index=int(left_index),
            right_index=int(right_index),
            left_label=left_label,
            right_label=right_label,
            pair_label=str(pair_label),
            lag_vector=-np.asarray(separation_vector, dtype=float),
            lag_magnitude=float(separation_magnitude),
            key=reverse_key,
            reflected_key=canonical_key,
            delta_zplus=-np.asarray(delta_zplus, dtype=float),
            delta_zminus=-np.asarray(delta_zminus, dtype=float),
            delta_zplus_timeseries=-delta_zplus_timeseries,
            delta_zminus_timeseries=-delta_zminus_timeseries,
        )
        canonical_y_plus_ts, canonical_y_minus_ts = _compute_yaglom_timeseries(
            canonical_point.delta_zplus_timeseries,
            canonical_point.delta_zminus_timeseries,
        )
        canonical_point.y_plus_timeseries = canonical_y_plus_ts
        canonical_point.y_minus_timeseries = canonical_y_minus_ts
        canonical_point.y_plus = _finite_time_average(canonical_y_plus_ts)
        canonical_point.y_minus = _finite_time_average(canonical_y_minus_ts)

        reflected_y_plus_ts, reflected_y_minus_ts = _compute_yaglom_timeseries(
            reflected_point.delta_zplus_timeseries,
            reflected_point.delta_zminus_timeseries,
        )
        reflected_point.y_plus_timeseries = reflected_y_plus_ts
        reflected_point.y_minus_timeseries = reflected_y_minus_ts
        reflected_point.y_plus = _finite_time_average(reflected_y_plus_ts)
        reflected_point.y_minus = _finite_time_average(reflected_y_minus_ts)
        canonical_idx = len(directed_lag_points)
        directed_lag_points.append(canonical_point)
        reflected_idx = len(directed_lag_points)
        directed_lag_points.append(reflected_point)
        reflection_map.extend([reflected_idx, canonical_idx])

    n_directed_points = len(directed_lag_points)
    raw_tetrahedra = math.comb(n_directed_points, 4)
    seen_reflection_keys: set[tuple[int, int, int, int]] = set()
    removed_redundant = 0
    removed_zero_barycenter = 0
    retained_tetrahedra: list[LagTetrahedron] = []

    vertices_by_index = np.stack(
        [np.asarray(point.lag_vector, dtype=float) for point in directed_lag_points],
        axis=0,
    )
    for combination_indices in combinations(range(n_directed_points), 4):
        reflected_indices = tuple(sorted(reflection_map[index] for index in combination_indices))
        canonical_key = min(combination_indices, reflected_indices)
        if canonical_key in seen_reflection_keys:
            removed_redundant += 1
            continue
        seen_reflection_keys.add(canonical_key)

        vertices = vertices_by_index[np.array(combination_indices, dtype=int)]
        barycenter, volumetric_tensor, eigenvalues, elongation, planarity, d_ep = (
            _compute_shape_metrics(vertices)
        )
        if np.linalg.norm(barycenter) <= float(zero_barycenter_atol):
            removed_zero_barycenter += 1
            continue
        passes_quality_cut = True if max_d_ep is None else bool(d_ep < float(max_d_ep))
        points = tuple(directed_lag_points[index] for index in combination_indices)
        y_plus_vertices = np.array(
            [
                np.full(3, np.nan, dtype=float)
                if point.y_plus is None
                else np.asarray(point.y_plus, dtype=float)
                for point in points
            ],
            dtype=float,
        )
        y_minus_vertices = np.array(
            [
                np.full(3, np.nan, dtype=float)
                if point.y_minus is None
                else np.asarray(point.y_minus, dtype=float)
                for point in points
            ],
            dtype=float,
        )
        grad_y_plus, plus_error = _estimate_vector_gradient_on_tetrahedron(vertices, y_plus_vertices)
        grad_y_minus, minus_error = _estimate_vector_gradient_on_tetrahedron(vertices, y_minus_vertices)
        div_y_plus = None if grad_y_plus is None else float(np.trace(grad_y_plus))
        div_y_minus = None if grad_y_minus is None else float(np.trace(grad_y_minus))
        epsilon_plus = None if div_y_plus is None else float(-0.25 * div_y_plus)
        epsilon_minus = None if div_y_minus is None else float(-0.25 * div_y_minus)
        invalid_reasons = [reason for reason in (plus_error, minus_error) if reason is not None]
        retained_tetrahedra.append(
            LagTetrahedron(
                points=points,  # type: ignore[arg-type]
                pair_labels=tuple(point.pair_label for point in points),  # type: ignore[arg-type]
                vertex_array=vertices,
                barycenter=barycenter,
                volumetric_tensor=volumetric_tensor,
                eigenvalues=eigenvalues,
                elongation=elongation,
                planarity=planarity,
                d_ep=d_ep,
                is_redundant=False,
                is_zero_barycenter=False,
                passes_quality_cut=passes_quality_cut,
                mesocenter_magnitude=float(np.linalg.norm(barycenter)),
                y_plus_vertices=y_plus_vertices,
                y_minus_vertices=y_minus_vertices,
                grad_y_plus=grad_y_plus,
                grad_y_minus=grad_y_minus,
                div_y_plus=div_y_plus,
                div_y_minus=div_y_minus,
                epsilon_plus=epsilon_plus,
                epsilon_minus=epsilon_minus,
                estimated_gradients=None
                if grad_y_plus is None and grad_y_minus is None
                else np.stack(
                    [
                        np.full((3, 3), np.nan, dtype=float) if grad_y_plus is None else grad_y_plus,
                        np.full((3, 3), np.nan, dtype=float) if grad_y_minus is None else grad_y_minus,
                    ],
                    axis=0,
                ),
                future_divergence=None,
                is_valid_for_gradient=(grad_y_plus is not None and grad_y_minus is not None),
                invalid_reason=None if not invalid_reasons else ",".join(invalid_reasons),
            )
        )

    summary_statistics = {
        "elongation": _summarize_metric(np.array([tetra.elongation for tetra in retained_tetrahedra])),
        "planarity": _summarize_metric(np.array([tetra.planarity for tetra in retained_tetrahedra])),
        "d_ep": _summarize_metric(np.array([tetra.d_ep for tetra in retained_tetrahedra])),
        "epsilon_plus": _summarize_metric(
            np.array(
                [
                    np.nan if tetra.epsilon_plus is None else tetra.epsilon_plus
                    for tetra in retained_tetrahedra
                    if tetra.passes_quality_cut and tetra.is_valid_for_gradient
                ],
                dtype=float,
            )
        ),
        "epsilon_minus": _summarize_metric(
            np.array(
                [
                    np.nan if tetra.epsilon_minus is None else tetra.epsilon_minus
                    for tetra in retained_tetrahedra
                    if tetra.passes_quality_cut and tetra.is_valid_for_gradient
                ],
                dtype=float,
            )
        ),
    }

    return LagTetrahedraResult(
        spacecraft_labels=list(prepared_input.spacecraft_labels),
        selected_time_index=int(prepared_input.time_index),
        selected_time_seconds=float(prepared_input.time_seconds),
        spacecraft_positions=spacecraft_positions,
        unordered_baselines=unordered_baselines,
        directed_lag_points=directed_lag_points,
        raw_tetrahedra=raw_tetrahedra,
        removed_redundant=removed_redundant,
        removed_zero_barycenter=removed_zero_barycenter,
        retained_tetrahedra=retained_tetrahedra,
        summary_statistics=summary_statistics,
        metadata={
            **dict(prepared_input.metadata),
            "geometry": {
                "lag_vector_definition": "r_ij = r_i - r_j",
                "directed_point_delta_assignment": (
                    "saved canonical j_minus_i assigned to directed point j__i; "
                    "reflected partner receives the negated increment"
                ),
                "volumetric_tensor_definition": (
                    "mean outer product of vertex offsets from tetrahedron barycenter"
                ),
                "shape_metric_definition": {
                    "elongation": "1 - sqrt(lambda2 / lambda1)",
                    "planarity": "1 - sqrt(lambda3 / lambda2)",
                    "d_ep": "sqrt(E^2 + P^2)",
                },
                "yaglom_definition": {
                    "y_plus": "delta_zminus * |delta_zplus|^2",
                    "y_minus": "delta_zplus * |delta_zminus|^2",
                    "averaging": "time_average_over_full_saved_interval",
                },
                "divergence_method": (
                    "affine tetrahedral reconstruction Y(l) = a + G(l-l0); "
                    "divergence is trace(G)"
                ),
                "epsilon_definition": "-0.25 * div_l <Y>",
            },
            "filtering": {
                "zero_barycenter_atol": float(zero_barycenter_atol),
                "reflection_redundancy": "tetrahedra equivalent under full vertex reflection",
                "quality_cut_applied": max_d_ep is not None,
                "max_d_ep": None if max_d_ep is None else float(max_d_ep),
            },
            "counts": {
                "unordered_baselines": len(unordered_baselines),
                "directed_lag_points": len(directed_lag_points),
                "raw_tetrahedra": int(raw_tetrahedra),
                "removed_redundant": int(removed_redundant),
                "removed_zero_barycenter": int(removed_zero_barycenter),
                "retained_tetrahedra": len(retained_tetrahedra),
                "valid_tetrahedra_for_gradient": int(
                    sum(tetra.is_valid_for_gradient for tetra in retained_tetrahedra)
                ),
                "passing_quality_cut": int(
                    sum(
                        tetra.passes_quality_cut and tetra.is_valid_for_gradient
                        for tetra in retained_tetrahedra
                    )
                ),
            },
            "tetrahedron_preview_limit": int(tetrahedron_preview_limit),
        },
    )


def plot_lag_tetrahedra_ep_scatter(
    result: LagTetrahedraResult,
    *,
    color_by_log_size: bool = True,
    highlight_tetrahedron_index: int | None = None,
    title: str = "",
):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    scatter_ax, cdf_ax, meso_ax = axes
    cdf_twin_ax = cdf_ax.twinx()
    cdf_ax._cdf_twin_ax = cdf_twin_ax
    helioswarm_metadata = result.metadata.get("helioswarm")
    transform_metadata = (
        helioswarm_metadata.get("transform")
        if isinstance(helioswarm_metadata, dict)
        else None
    )
    rho_p_km = (
        transform_metadata.get("rho_p_km")
        if isinstance(transform_metadata, dict)
        else None
    )
    if rho_p_km is None:
        raise ValueError(
            "Lag-tetrahedra metadata must include helioswarm.transform.rho_p_km for mesocenter kilometer plots"
        )
    rho_p_km = float(rho_p_km)
    if not np.isfinite(rho_p_km) or rho_p_km <= 0.0:
        raise ValueError("helioswarm.transform.rho_p_km must be a positive finite number")
    e_values = np.array([tetra.elongation for tetra in result.retained_tetrahedra], dtype=float)
    p_values = np.array([tetra.planarity for tetra in result.retained_tetrahedra], dtype=float)
    d_ep_values = np.array([tetra.d_ep for tetra in result.retained_tetrahedra], dtype=float)
    mesocenter_magnitudes = np.array(
        [tetra.mesocenter_magnitude for tetra in result.retained_tetrahedra],
        dtype=float,
    )
    log_meso_km_values = np.log10(
        np.maximum(mesocenter_magnitudes * rho_p_km, np.finfo(float).tiny)
    )
    size_values = np.array(
        [
            np.mean(
                np.linalg.norm(
                    np.asarray(tetra.vertex_array, dtype=float)
                    - np.asarray(tetra.barycenter, dtype=float)[None, :],
                    axis=1,
                )
            )
            for tetra in result.retained_tetrahedra
        ],
        dtype=float,
    )
    log_size_values = np.log10(np.maximum(size_values, np.finfo(float).tiny))

    if len(e_values) == 0:
        scatter_ax.text(0.5, 0.5, "No retained lag tetrahedra", ha="center", va="center")
        cdf_ax.text(0.5, 0.5, "No retained lag tetrahedra", ha="center", va="center")
        meso_ax.text(0.5, 0.5, "No retained lag tetrahedra", ha="center", va="center")
    else:
        scatter = scatter_ax.scatter(
            e_values,
            p_values,
            c=log_size_values if color_by_log_size else "#1f77b4",
            cmap="viridis" if color_by_log_size else None,
            s=12,
            alpha=0.85,
            edgecolors="none",
        )
        if color_by_log_size:
            colorbar = fig.colorbar(scatter, ax=scatter_ax, shrink=0.9)
            colorbar.set_label(r"$\log_{10}(L_{\mathrm{tetra}})$")
        sorted_indices = np.argsort(d_ep_values, kind="stable")
        sorted_d_ep = d_ep_values[sorted_indices]
        cumulative_fraction = np.arange(1, len(sorted_d_ep) + 1, dtype=float) / float(len(sorted_d_ep))
        histogram_bins = min(20, max(5, int(np.sqrt(len(d_ep_values)))))
        histogram_counts, histogram_edges, _ = cdf_ax.hist(
            d_ep_values,
            bins=histogram_bins,
            color="#9ecae1",
            edgecolor="#4c72b0",
            alpha=0.85,
        )
        cdf_twin_ax.step(sorted_d_ep, cumulative_fraction, where="post", color="#1f77b4", linewidth=1.8)
        meso_ax.scatter(
            d_ep_values,
            log_meso_km_values,
            s=12,
            alpha=0.85,
            edgecolors="none",
            color="#1f77b4",
        )
        if highlight_tetrahedron_index is not None:
            if highlight_tetrahedron_index < 0 or highlight_tetrahedron_index >= len(result.retained_tetrahedra):
                raise ValueError(
                    f"highlight_tetrahedron_index must lie in [0, {len(result.retained_tetrahedra) - 1}]"
                )
            highlighted = result.retained_tetrahedra[int(highlight_tetrahedron_index)]
            scatter_ax.scatter(
                [highlighted.elongation],
                [highlighted.planarity],
                s=90,
                facecolors="none",
                edgecolors="crimson",
                linewidths=1.8,
                zorder=4,
            )
            scatter_ax.text(
                highlighted.elongation,
                highlighted.planarity,
                f"  #{highlight_tetrahedron_index}",
                color="crimson",
                va="center",
                fontsize=9,
            )
            highlighted_rank = float(np.count_nonzero(d_ep_values <= highlighted.d_ep)) / float(len(d_ep_values))
            histogram_index = np.searchsorted(histogram_edges, highlighted.d_ep, side="right") - 1
            histogram_index = int(np.clip(histogram_index, 0, len(histogram_counts) - 1))
            highlighted_count = float(histogram_counts[histogram_index])
            cdf_ax.scatter(
                [highlighted.d_ep],
                [highlighted_count],
                s=90,
                facecolors="none",
                edgecolors="crimson",
                linewidths=1.8,
                zorder=4,
            )
            cdf_twin_ax.scatter(
                [highlighted.d_ep],
                [highlighted_rank],
                s=90,
                facecolors="none",
                edgecolors="crimson",
                linewidths=1.8,
                zorder=4,
            )
            cdf_twin_ax.text(
                highlighted.d_ep,
                highlighted_rank,
                f"  #{highlight_tetrahedron_index}",
                color="crimson",
                va="center",
                fontsize=9,
            )
            highlighted_log_meso_km = np.log10(
                max(float(highlighted.mesocenter_magnitude) * rho_p_km, np.finfo(float).tiny)
            )
            meso_ax.scatter(
                [highlighted.d_ep],
                [highlighted_log_meso_km],
                s=90,
                facecolors="none",
                edgecolors="crimson",
                linewidths=1.8,
                zorder=4,
            )
            meso_ax.text(
                highlighted.d_ep,
                highlighted_log_meso_km,
                f"  #{highlight_tetrahedron_index}",
                color="crimson",
                va="center",
                fontsize=9,
            )

    scatter_ax.set_xlabel("Elongation E")
    scatter_ax.set_ylabel("Planarity P")
    scatter_ax.grid(alpha=0.3)
    cdf_ax.set_xlabel(r"$d_{EP} = \sqrt{E^2 + P^2}$")
    cdf_ax.set_ylabel("Count")
    cdf_ax.grid(alpha=0.3)
    cdf_twin_ax.set_ylabel("Cumulative Fraction")
    if len(d_ep_values) > 0:
        cdf_twin_ax.set_ylim(0.0, 1.0)
    meso_ax.set_xlabel(r"$d_{EP} = \sqrt{E^2 + P^2}$")
    meso_ax.set_ylabel(r"$\log_{10}(|L_{\mathrm{meso}}| / \mathrm{km})$")
    meso_ax.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    return fig, axes


def plot_lag_tetrahedra_yaglom_flux(
    result: LagTetrahedraResult,
    *,
    max_arrows: int = 24,
    display_fraction_of_lag_span: float = 0.18,
    highlight_tetrahedron_index: int | None = None,
    title: str = "",
):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    eligible_points = [
        point
        for point in result.directed_lag_points
        if point.y_plus is not None and point.y_minus is not None
    ]
    eligible_points.sort(key=lambda point: (point.lag_magnitude, point.key))
    step = max(1, int(math.ceil(len(eligible_points) / max(max_arrows, 1))))
    selected_points = eligible_points[::step][:max_arrows]

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    axes = [
        fig.add_subplot(1, 2, 1, projection="3d"),
        fig.add_subplot(1, 2, 2, projection="3d"),
    ]
    fields = [
        (r"$\langle Y^+ \rangle$", "y_plus"),
        (r"$\langle Y^- \rangle$", "y_minus"),
    ]
    highlight_metadata = {
        "highlighted_point_keys": None,
        "highlighted_vertex_coordinates": None,
    }

    for ax, (label, attr_name) in zip(axes, fields):
        extent_points: list[np.ndarray] = []
        if not selected_points:
            ax.text2D(0.5, 0.5, "No valid lag-space flux points", transform=ax.transAxes, ha="center", va="center")
        else:
            origins = np.array([point.lag_vector for point in selected_points], dtype=float)
            vectors = np.array([getattr(point, attr_name) for point in selected_points], dtype=float)
            extent_points.append(origins)
            vector_magnitudes = np.linalg.norm(vectors, axis=1)
            max_vector_magnitude = float(np.max(vector_magnitudes))
            lag_span = float(
                np.max(np.ptp(origins, axis=0))
                if origins.shape[0] > 1
                else max(np.linalg.norm(origins[0]), 1.0)
            )
            display_scale = 1.0
            if np.isfinite(max_vector_magnitude) and max_vector_magnitude > 0.0:
                target_display_length = max(
                    float(display_fraction_of_lag_span) * max(lag_span, 1.0),
                    1.0e-12,
                )
                display_scale = target_display_length / max_vector_magnitude
            display_vectors = vectors * display_scale
            ax.quiver(
                origins[:, 0],
                origins[:, 1],
                origins[:, 2],
                display_vectors[:, 0],
                display_vectors[:, 1],
                display_vectors[:, 2],
                normalize=False,
                linewidth=0.8,
                arrow_length_ratio=0.15,
                color="#1f77b4",
            )
            ax.text2D(
                0.03,
                0.96,
                f"Display scale x{display_scale:.2e}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )
        if highlight_tetrahedron_index is not None:
            if highlight_tetrahedron_index < 0 or highlight_tetrahedron_index >= len(result.retained_tetrahedra):
                raise ValueError(
                    f"highlight_tetrahedron_index must lie in [0, {len(result.retained_tetrahedra) - 1}]"
                )
            highlighted = result.retained_tetrahedra[int(highlight_tetrahedron_index)]
            vertices = np.asarray(highlighted.vertex_array, dtype=float)
            extent_points.append(vertices)
            highlight_metadata = {
                "highlighted_point_keys": [point.key for point in highlighted.points],
                "highlighted_vertex_coordinates": vertices.tolist(),
            }
            cycle_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            for left, right in cycle_edges:
                edge = vertices[[left, right], :]
                ax.plot(
                    edge[:, 0],
                    edge[:, 1],
                    edge[:, 2],
                    color="crimson",
                    linewidth=1.5,
                    alpha=0.9,
                )
            ax.scatter(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                s=58,
                facecolors="#fff1f2",
                edgecolors="crimson",
                linewidths=1.7,
                depthshade=False,
                zorder=5,
            )
            offset_scale = max(np.max(np.ptp(vertices, axis=0)), 1.0) * 0.025
            for vertex_index, vertex in enumerate(vertices, start=1):
                ax.text(
                    vertex[0] + offset_scale,
                    vertex[1] + offset_scale,
                    vertex[2] + offset_scale,
                    f"v{vertex_index}",
                    color="crimson",
                    fontsize=9,
                    zorder=6,
                )
        if extent_points:
            combined_points = np.concatenate(extent_points, axis=0)
            min_corner = np.min(combined_points, axis=0)
            max_corner = np.max(combined_points, axis=0)
            center = 0.5 * (min_corner + max_corner)
            half_span = 0.5 * np.max(np.maximum(max_corner - min_corner, 1.0e-12))
            padded_half_span = max(half_span * 1.08, 1.0)
            ax.set_xlim(center[0] - padded_half_span, center[0] + padded_half_span)
            ax.set_ylim(center[1] - padded_half_span, center[1] + padded_half_span)
            ax.set_zlim(center[2] - padded_half_span, center[2] + padded_half_span)
            ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.scatter([0.0], [0.0], [0.0], color="black", s=18)
        ax.set_xlabel(r"$\ell_x$ ($\rho_p$)")
        ax.set_ylabel(r"$\ell_y$ ($\rho_p$)")
        ax.set_zlabel(r"$\ell_z$ ($\rho_p$)")
        ax.set_title(label)
    if title:
        fig.suptitle(title)
    return fig, axes, {
        "selected_keys": [point.key for point in selected_points],
        **highlight_metadata,
    }


def plot_lag_tetrahedra_epsilon_diagnostics(
    result: LagTetrahedraResult,
    *,
    highlight_tetrahedron_index: int | None = None,
    lag_scale_units: str = "rho_p",
    lag_scale_transform: str = "linear",
    log_floor: float = 1.0e-12,
    title: str = "",
):
    if lag_scale_units not in {"rho_p", "km"}:
        raise ValueError("lag_scale_units must be 'rho_p' or 'km'")
    if lag_scale_transform not in {"linear", "log10"}:
        raise ValueError("lag_scale_transform must be 'linear' or 'log10'")

    rho_p_km = None
    if lag_scale_units == "km":
        helioswarm_metadata = result.metadata.get("helioswarm")
        transform_metadata = (
            helioswarm_metadata.get("transform")
            if isinstance(helioswarm_metadata, dict)
            else None
        )
        rho_p_km = (
            transform_metadata.get("rho_p_km")
            if isinstance(transform_metadata, dict)
            else None
        )
        if rho_p_km is None:
            raise ValueError(
                "Lag-tetrahedra metadata must include helioswarm.transform.rho_p_km for kilometer plots"
            )
        rho_p_km = float(rho_p_km)
        if not np.isfinite(rho_p_km) or rho_p_km <= 0.0:
            raise ValueError("helioswarm.transform.rho_p_km must be a positive finite number")

    def _transform_lag_scale(values) -> np.ndarray:
        transformed = np.asarray(values, dtype=float)
        if lag_scale_units == "km":
            transformed = transformed * float(rho_p_km)
        if lag_scale_transform == "log10":
            transformed = np.log10(np.maximum(transformed, float(log_floor)))
        return transformed

    if lag_scale_units == "rho_p" and lag_scale_transform == "linear":
        x_label = r"$|\ell_{\mathrm{meso}}|$ ($\rho_p$)"
    elif lag_scale_units == "km" and lag_scale_transform == "linear":
        x_label = r"$|\ell_{\mathrm{meso}}|$ (km)"
    elif lag_scale_units == "rho_p" and lag_scale_transform == "log10":
        x_label = r"$\log_{10}(|\ell_{\mathrm{meso}}| / \rho_p)$"
    else:
        x_label = r"$\log_{10}(|\ell_{\mathrm{meso}}| / \mathrm{km})$"

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    subfigs = fig.subfigures(1, 2)
    plot_specs = [
        (r"$\epsilon^+$", "epsilon_plus"),
        (r"$\epsilon^-$", "epsilon_minus"),
    ]
    filtered_tetrahedra = [
        tetra
        for tetra in result.retained_tetrahedra
        if tetra.passes_quality_cut and tetra.is_valid_for_gradient
    ]
    values_by_attr = {
        attr_name: np.array(
            [
                getattr(tetra, attr_name)
                for tetra in filtered_tetrahedra
                if getattr(tetra, attr_name) is not None
            ],
            dtype=float,
        )
        for _, attr_name in plot_specs
    }
    finite_y_values = np.concatenate(
        [
            values[np.isfinite(values)]
            for values in values_by_attr.values()
            if values.size > 0
        ]
    ) if any(values.size > 0 for values in values_by_attr.values()) else np.array([], dtype=float)
    if finite_y_values.size == 0:
        shared_y_extent = 1.0
    else:
        shared_y_extent = float(np.max(np.abs(finite_y_values)))
        if not np.isfinite(shared_y_extent) or shared_y_extent <= 0.0:
            shared_y_extent = 1.0
    shared_y_limits = (-shared_y_extent, shared_y_extent)

    for subfig, (label, attr_name) in zip(subfigs, plot_specs):
        axes = subfig.subplots(1, 2, gridspec_kw={"width_ratios": [4, 1]}, sharey=True)
        ax_scatter, ax_hist = axes
        values = values_by_attr[attr_name]
        lag_scales = np.array(
            [
                tetra.mesocenter_magnitude
                for tetra in filtered_tetrahedra
                if getattr(tetra, attr_name) is not None
            ],
            dtype=float,
        )
        if values.size == 0:
            ax_scatter.text(0.5, 0.5, "No tetrahedra pass quality cut", ha="center", va="center", transform=ax_scatter.transAxes)
            ax_hist.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_hist.transAxes)
        else:
            mean_value = float(np.mean(values))
            std_value = float(np.std(values))
            transformed_lag_scales = _transform_lag_scale(lag_scales)
            ax_scatter.scatter(transformed_lag_scales, values, s=10, alpha=0.8, edgecolors="none")
            ax_scatter.axhline(mean_value, color="tab:red", linestyle="--", linewidth=1.0)
            ax_scatter.axhspan(mean_value - std_value, mean_value + std_value, color="tab:red", alpha=0.15)
            histogram_counts, histogram_edges = np.histogram(
                values,
                bins=min(20, max(5, int(np.sqrt(values.size)))),
                range=shared_y_limits,
            )
            positive_mask = histogram_counts > 0
            if np.any(positive_mask):
                histogram_centers = 0.5 * (histogram_edges[:-1] + histogram_edges[1:])
                histogram_heights = np.diff(histogram_edges)
                ax_hist.barh(
                    histogram_centers[positive_mask],
                    np.log10(histogram_counts[positive_mask].astype(float)),
                    height=histogram_heights[positive_mask],
                    color="#4c72b0",
                    alpha=0.85,
                )
            ax_hist.axhline(mean_value, color="tab:red", linestyle="--", linewidth=1.0)
            ax_hist.axhspan(mean_value - std_value, mean_value + std_value, color="tab:red", alpha=0.15)
        if highlight_tetrahedron_index is not None:
            if highlight_tetrahedron_index < 0 or highlight_tetrahedron_index >= len(result.retained_tetrahedra):
                raise ValueError(
                    f"highlight_tetrahedron_index must lie in [0, {len(result.retained_tetrahedra) - 1}]"
                )
            highlighted = result.retained_tetrahedra[int(highlight_tetrahedron_index)]
            highlighted_value = getattr(highlighted, attr_name)
            if highlighted_value is not None and highlighted.mesocenter_magnitude is not None:
                highlighted_lag_scale = _transform_lag_scale([highlighted.mesocenter_magnitude])
                ax_scatter.scatter(
                    highlighted_lag_scale,
                    [highlighted_value],
                    s=90,
                    facecolors="none",
                    edgecolors="crimson",
                    linewidths=1.8,
                    zorder=4,
                )
                ax_hist.scatter(
                    [0.5],
                    [highlighted_value],
                    s=70,
                    facecolors="none",
                    edgecolors="crimson",
                    linewidths=1.8,
                    zorder=4,
                    transform=ax_hist.get_yaxis_transform(),
                )
        ax_scatter.set_xlabel(x_label)
        ax_scatter.set_ylabel(label)
        ax_scatter.grid(alpha=0.3)
        ax_hist.grid(alpha=0.3)
        ax_scatter.set_ylim(*shared_y_limits)
        ax_hist.set_ylim(*shared_y_limits)
        ax_hist.set_xlabel(r"$\log_{10}(\mathrm{count})$")
        subfig.suptitle(label)
    if title:
        fig.suptitle(title)
    return fig, subfigs


def plot_lag_tetrahedra_baseline_projections(
    result: LagTetrahedraResult,
    *,
    floor: float = 1.0e-12,
    highlight_tetrahedron_index: int | None = None,
    title: str = "",
):
    baselines = np.array(
        [baseline["separation_vector_saved"] for baseline in result.unordered_baselines],
        dtype=float,
    )
    abs_baselines = np.maximum(np.abs(baselines), float(floor))
    log_components = np.log10(abs_baselines)

    fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(grid[0, 0])
    axes[0, 1] = fig.add_subplot(grid[0, 1])
    axes[1, 0] = fig.add_subplot(grid[1, 0])
    axes[1, 1] = fig.add_subplot(grid[1, 1])
    axes[0, 1].axis("off")

    panel_specs = [
        (axes[0, 0], 0, 2, r"$\log_{10}|\ell_x / \rho_p|$", r"$\log_{10}|\ell_z / \rho_p|$"),
        (axes[1, 0], 0, 1, r"$\log_{10}|\ell_x / \rho_p|$", r"$\log_{10}|\ell_y / \rho_p|$"),
        (axes[1, 1], 2, 1, r"$\log_{10}|\ell_z / \rho_p|$", r"$\log_{10}|\ell_y / \rho_p|$"),
    ]
    finite_log_values = log_components[np.isfinite(log_components)]
    if finite_log_values.size == 0:
        global_log_min = -12.0
        global_log_max = 0.0
    else:
        global_log_min = float(np.min(finite_log_values))
        global_log_max = float(np.max(finite_log_values))
    if not np.isfinite(global_log_min) or not np.isfinite(global_log_max):
        global_log_min, global_log_max = -12.0, 0.0
    if np.isclose(global_log_min, global_log_max):
        global_log_min -= 0.5
        global_log_max += 0.5
    shared_log_limits = (global_log_min, global_log_max)

    for ax, x_index, y_index, x_label, y_label in panel_specs:
        ax.scatter(
            log_components[:, x_index],
            log_components[:, y_index],
            s=20,
            color="#111111",
            alpha=0.85,
            edgecolors="none",
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.3)
        ax.set_xlim(*shared_log_limits)
        ax.set_ylim(*shared_log_limits)
        ax.set_aspect("equal", adjustable="box")

    baseline_plotting = {
        "global_log_range": list(shared_log_limits),
    }

    if highlight_tetrahedron_index is not None:
        if highlight_tetrahedron_index < 0 or highlight_tetrahedron_index >= len(result.retained_tetrahedra):
            raise ValueError(
                f"highlight_tetrahedron_index must lie in [0, {len(result.retained_tetrahedra) - 1}]"
            )
        highlighted = result.retained_tetrahedra[int(highlight_tetrahedron_index)]
        highlighted_vertices = np.asarray(highlighted.vertex_array, dtype=float)
        if len(highlighted_vertices) > 0:
            highlighted_abs_vertices = np.maximum(np.abs(highlighted_vertices), float(floor))
            highlighted_components = np.log10(highlighted_abs_vertices)
            highlighted_panels = []
            any_highlight_offsets = False
            log_span = global_log_max - global_log_min
            for ax, x_index, y_index, _, _ in panel_specs:
                x_values = highlighted_components[:, x_index]
                y_values = highlighted_components[:, y_index]
                display_x, display_y, offsets, had_overlaps = _offset_overlapping_points(
                    x_values,
                    y_values,
                    span=log_span,
                )
                any_highlight_offsets = any_highlight_offsets or had_overlaps
                ax.scatter(
                    display_x,
                    display_y,
                    s=60,
                    facecolors="none",
                    edgecolors="crimson",
                    linewidths=1.5,
                    zorder=4,
                )
                for vertex_index, (x_value, y_value) in enumerate(zip(display_x, display_y), start=1):
                    ax.annotate(
                        f"v{vertex_index}",
                        (x_value, y_value),
                        xytext=(4, 4),
                        textcoords="offset points",
                        color="crimson",
                        fontsize=9,
                    )
                highlighted_panels.append(
                    {
                        "component_indices": [int(x_index), int(y_index)],
                        "projected_coordinates": np.column_stack([x_values, y_values]).tolist(),
                        "display_coordinates": np.column_stack([display_x, display_y]).tolist(),
                        "display_offsets": offsets.tolist(),
                        "had_overlaps": bool(had_overlaps),
                    }
                )
            baseline_plotting.update(
                {
                    "highlighted_pair_labels": list(highlighted.pair_labels),
                    "highlighted_vertex_coordinates": highlighted_vertices.tolist(),
                    "highlighted_log_components": highlighted_components.tolist(),
                    "highlighted_projection_panels": highlighted_panels,
                    "highlighted_projection_had_overlaps": bool(any_highlight_offsets),
                }
            )

    if title:
        fig.suptitle(title)
    return fig, axes, baseline_plotting
