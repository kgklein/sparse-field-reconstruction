from __future__ import annotations

from csv import DictWriter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


HS_COLORS = [
    "#56B4E9",
    "#E69F00",
    "#000000",
    "#888888",
    "#F0E442",
    "#D55E00",
    "#009E73",
    "#8D00B2",
    "#CC79A7",
]

TIMESERIES_FIELDNAMES = [
    "step",
    "time_seconds",
    "spacecraft_label",
    "x_rho_p",
    "y_rho_p",
    "z_rho_p",
    "bx",
    "by",
    "bz",
]


@dataclass
class StructuredFieldSnapshot:
    values: np.ndarray
    axes: dict[str, np.ndarray]
    grid_shape: tuple[int, int, int]
    metadata: dict | None = None


def build_hs_color_map(spacecraft_labels: list[str]) -> dict[str, str]:
    if len(spacecraft_labels) > len(HS_COLORS):
        raise ValueError(
            f"HelioSwarm color map supports at most {len(HS_COLORS)} spacecraft; "
            f"got {len(spacecraft_labels)}"
        )
    return {label: HS_COLORS[i] for i, label in enumerate(spacecraft_labels)}


def load_structured_simulation_snapshot(
    path: str | Path,
    *,
    sim_box_rho_p: tuple[float, float, float],
) -> StructuredFieldSnapshot:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place the file locally or provide a valid simulation snapshot path."
        )
    if path.suffix.lower() != ".npy":
        raise ValueError(
            f"Unsupported simulation snapshot format: {path.suffix}. Expected .npy"
        )

    field = np.load(path, allow_pickle=False)
    if field.ndim != 4:
        raise ValueError(
            "Simulation snapshot must be a 4D array shaped (nx, ny, nz, 3); "
            f"got shape {field.shape}"
        )
    if field.shape[-1] != 3:
        raise ValueError(
            "Simulation snapshot last dimension must have size 3 for vector components; "
            f"got shape {field.shape}"
        )

    sim_box = np.asarray(sim_box_rho_p, dtype=float)
    if sim_box.shape != (3,) or np.any(sim_box <= 0):
        raise ValueError(
            "sim_box_rho_p must contain three positive axis lengths; "
            f"got {sim_box_rho_p}"
        )

    nx, ny, nz, _ = field.shape
    axes = {
        "x": np.linspace(0.0, sim_box[0], nx),
        "y": np.linspace(0.0, sim_box[1], ny),
        "z": np.linspace(0.0, sim_box[2], nz),
    }
    metadata = {
        "source": "simulation",
        "field_kind": "simulation_snapshot",
        "file_path": str(path),
        "array_shape": list(field.shape),
        "dtype": str(field.dtype),
        "grid_convention": "uniform_box_rho_p",
        "sim_box_rho_p": sim_box.tolist(),
    }
    return StructuredFieldSnapshot(
        values=field,
        axes=axes,
        grid_shape=(nx, ny, nz),
        metadata=metadata,
    )


def generate_moving_spacecraft_trajectory(
    initial_coords_rho_p: np.ndarray,
    *,
    velocity_km_s: tuple[float, float, float],
    rho_p_km: float,
    sim_box_rho_p: tuple[float, float, float],
    dt_seconds: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    initial_coords_rho_p = np.asarray(initial_coords_rho_p, dtype=float)
    if initial_coords_rho_p.ndim != 2 or initial_coords_rho_p.shape[1] != 3:
        raise ValueError(
            "initial_coords_rho_p must be an array shaped (n_spacecraft, 3); "
            f"got {initial_coords_rho_p.shape}"
        )
    if rho_p_km <= 0:
        raise ValueError(f"rho_p_km must be positive; got {rho_p_km}")
    if dt_seconds <= 0:
        raise ValueError(f"dt_seconds must be positive; got {dt_seconds}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive; got {n_steps}")

    box = np.asarray(sim_box_rho_p, dtype=float)
    if box.shape != (3,) or np.any(box <= 0):
        raise ValueError(
            "sim_box_rho_p must contain three positive axis lengths; "
            f"got {sim_box_rho_p}"
        )

    velocity_km_s_arr = np.asarray(velocity_km_s, dtype=float)
    if velocity_km_s_arr.shape != (3,):
        raise ValueError(
            "velocity_km_s must contain three components; "
            f"got {velocity_km_s}"
        )

    velocity_rho_p_s = velocity_km_s_arr / rho_p_km
    time_seconds = np.arange(n_steps, dtype=float) * dt_seconds
    displacement = time_seconds[:, None, None] * velocity_rho_p_s[None, None, :]
    unwrapped_coords = initial_coords_rho_p[None, :, :] + displacement
    wrapped_coords = np.mod(unwrapped_coords, box[None, None, :])

    metadata = {
        "velocity_km_s": velocity_km_s_arr.tolist(),
        "velocity_rho_p_s": velocity_rho_p_s.tolist(),
        "dt_seconds": float(dt_seconds),
        "n_steps": int(n_steps),
        "sim_box_rho_p": box.tolist(),
        "wrapping": "periodic_per_axis",
    }
    return time_seconds, unwrapped_coords, wrapped_coords, metadata


def sample_structured_field_nearest(
    field: StructuredFieldSnapshot,
    sample_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sample_coords = np.asarray(sample_coords, dtype=float)
    if sample_coords.ndim != 2 or sample_coords.shape[1] != 3:
        raise ValueError(
            "sample_coords must be shaped (n_samples, 3); "
            f"got {sample_coords.shape}"
        )

    axes = field.axes
    index_arrays = []
    for axis_name, coords in zip(("x", "y", "z"), sample_coords.T):
        axis = np.asarray(axes[axis_name], dtype=float)
        if len(axis) < 2:
            idx = np.zeros(len(coords), dtype=int)
        else:
            spacing = float(axis[1] - axis[0])
            idx = np.rint((coords - axis[0]) / spacing).astype(int)
            idx = np.clip(idx, 0, len(axis) - 1)
        index_arrays.append(idx)

    ix, iy, iz = index_arrays
    values = field.values[ix, iy, iz]
    return sample_coords, values


def sample_structured_field_trilinear(
    field: StructuredFieldSnapshot,
    sample_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sample_coords = np.asarray(sample_coords, dtype=float)
    if sample_coords.ndim != 2 or sample_coords.shape[1] != 3:
        raise ValueError(
            "sample_coords must be shaped (n_samples, 3); "
            f"got {sample_coords.shape}"
        )

    lower_indices = []
    upper_indices = []
    weights = []
    for axis_name, coords in zip(("x", "y", "z"), sample_coords.T):
        axis = np.asarray(field.axes[axis_name], dtype=float)
        n_axis = len(axis)
        if n_axis == 0:
            raise ValueError(f"Axis {axis_name} is empty")
        if n_axis == 1:
            lower = np.zeros(len(coords), dtype=int)
            upper = np.zeros(len(coords), dtype=int)
            weight = np.zeros(len(coords), dtype=float)
        else:
            spacing = float(axis[1] - axis[0])
            if spacing <= 0:
                raise ValueError(f"Axis {axis_name} spacing must be positive")
            scaled = (coords - axis[0]) / spacing
            lower = np.floor(scaled).astype(int)
            lower = np.mod(lower, n_axis)
            upper = np.mod(lower + 1, n_axis)
            weight = scaled - np.floor(scaled)
        lower_indices.append(lower)
        upper_indices.append(upper)
        weights.append(weight)

    lx, ly, lz = lower_indices
    ux, uy, uz = upper_indices
    wx, wy, wz = weights

    values = np.zeros((len(sample_coords), field.values.shape[-1]), dtype=float)
    for x_index, x_weight in ((lx, 1.0 - wx), (ux, wx)):
        for y_index, y_weight in ((ly, 1.0 - wy), (uy, wy)):
            for z_index, z_weight in ((lz, 1.0 - wz), (uz, wz)):
                corner_weight = (x_weight * y_weight * z_weight)[:, None]
                values += corner_weight * field.values[x_index, y_index, z_index]

    return sample_coords, values


def sample_structured_field(
    field: StructuredFieldSnapshot,
    sample_coords: np.ndarray,
    *,
    method: str = "nearest",
) -> tuple[np.ndarray, np.ndarray]:
    if method == "nearest":
        return sample_structured_field_nearest(field, sample_coords)
    if method == "trilinear":
        return sample_structured_field_trilinear(field, sample_coords)
    raise ValueError(f"Unknown sampling method '{method}'")


def iter_time_series_records(
    *,
    initial_coords_rho_p: np.ndarray,
    velocity_km_s: tuple[float, float, float],
    rho_p_km: float,
    sim_box_rho_p: tuple[float, float, float],
    dt_seconds: float,
    n_steps: int,
    field: StructuredFieldSnapshot,
    spacecraft_labels: list[str],
    sampling_method: str = "nearest",
    progress_callback=None,
):
    initial_coords_rho_p = np.asarray(initial_coords_rho_p, dtype=float)
    sim_box = np.asarray(sim_box_rho_p, dtype=float)
    velocity_rho_p_s = np.asarray(velocity_km_s, dtype=float) / float(rho_p_km)

    for step in range(n_steps):
        time_value = float(step * dt_seconds)
        unwrapped_coords = initial_coords_rho_p + time_value * velocity_rho_p_s[None, :]
        wrapped_coords = np.mod(unwrapped_coords, sim_box[None, :])
        if progress_callback is not None:
            progress_callback(
                step=step,
                n_steps=n_steps,
                time_seconds=time_value,
            )
        sampled_coords, sampled_values = sample_structured_field(
            field,
            wrapped_coords,
            method=sampling_method,
        )
        rows = []
        for label, coord, value in zip(spacecraft_labels, sampled_coords, sampled_values):
            rows.append(
                {
                    "step": int(step),
                    "time_seconds": time_value,
                    "spacecraft_label": label,
                    "x_rho_p": float(coord[0]),
                    "y_rho_p": float(coord[1]),
                    "z_rho_p": float(coord[2]),
                    "bx": float(value[0]),
                    "by": float(value[1]),
                    "bz": float(value[2]),
                }
            )
        yield {
            "step": step,
            "time_seconds": time_value,
            "unwrapped_coords": unwrapped_coords,
            "wrapped_coords": wrapped_coords,
            "rows": rows,
            "values": sampled_values,
        }


def stream_timeseries_to_csv(
    output_path: str | Path,
    *,
    initial_coords_rho_p: np.ndarray,
    velocity_km_s: tuple[float, float, float],
    rho_p_km: float,
    sim_box_rho_p: tuple[float, float, float],
    dt_seconds: float,
    n_steps: int,
    field: StructuredFieldSnapshot,
    spacecraft_labels: list[str],
    sampling_method: str = "nearest",
    progress_callback=None,
) -> dict:
    output_path = Path(output_path)
    n_spacecraft = len(spacecraft_labels)
    times = np.empty(n_steps, dtype=float)
    bx = np.empty((n_spacecraft, n_steps), dtype=float)
    by = np.empty((n_spacecraft, n_steps), dtype=float)
    bz = np.empty((n_spacecraft, n_steps), dtype=float)

    final_unwrapped_coords = None
    final_wrapped_coords = None
    row_count = 0

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = DictWriter(f, fieldnames=TIMESERIES_FIELDNAMES)
        writer.writeheader()

        for chunk in iter_time_series_records(
            initial_coords_rho_p=initial_coords_rho_p,
            velocity_km_s=velocity_km_s,
            rho_p_km=rho_p_km,
            sim_box_rho_p=sim_box_rho_p,
            dt_seconds=dt_seconds,
            n_steps=n_steps,
            field=field,
            spacecraft_labels=spacecraft_labels,
            sampling_method=sampling_method,
            progress_callback=progress_callback,
        ):
            writer.writerows(chunk["rows"])
            row_count += len(chunk["rows"])
            step = chunk["step"]
            times[step] = chunk["time_seconds"]
            bx[:, step] = chunk["values"][:, 0]
            by[:, step] = chunk["values"][:, 1]
            bz[:, step] = chunk["values"][:, 2]
            final_unwrapped_coords = chunk["unwrapped_coords"]
            final_wrapped_coords = chunk["wrapped_coords"]

    return {
        "row_count": row_count,
        "times": times,
        "bx": bx,
        "by": by,
        "bz": bz,
        "final_unwrapped_coords": (
            final_unwrapped_coords.tolist() if final_unwrapped_coords is not None else None
        ),
        "final_wrapped_coords": (
            final_wrapped_coords.tolist() if final_wrapped_coords is not None else None
        ),
    }


def sample_timeseries_from_trajectory(
    field: StructuredFieldSnapshot,
    *,
    time_seconds: np.ndarray,
    wrapped_coords: np.ndarray,
    spacecraft_labels: list[str],
    sampling_method: str = "nearest",
    progress_callback=None,
) -> list[dict]:
    time_seconds = np.asarray(time_seconds, dtype=float)
    wrapped_coords = np.asarray(wrapped_coords, dtype=float)

    if wrapped_coords.ndim != 3 or wrapped_coords.shape[-1] != 3:
        raise ValueError(
            "wrapped_coords must be shaped (n_steps, n_spacecraft, 3); "
            f"got {wrapped_coords.shape}"
        )
    if wrapped_coords.shape[0] != len(time_seconds):
        raise ValueError("time_seconds length must match wrapped_coords first dimension")
    if wrapped_coords.shape[1] != len(spacecraft_labels):
        raise ValueError("spacecraft_labels length must match wrapped_coords second dimension")

    records: list[dict] = []
    for step, (time_value, coords_step) in enumerate(zip(time_seconds, wrapped_coords)):
        if progress_callback is not None:
            progress_callback(
                step=step,
                n_steps=len(time_seconds),
                time_seconds=float(time_value),
            )
        sampled_coords, sampled_values = sample_structured_field(
            field,
            coords_step,
            method=sampling_method,
        )
        for label, coord, value in zip(spacecraft_labels, sampled_coords, sampled_values):
            records.append(
                {
                    "step": int(step),
                    "time_seconds": float(time_value),
                    "spacecraft_label": label,
                    "x_rho_p": float(coord[0]),
                    "y_rho_p": float(coord[1]),
                    "z_rho_p": float(coord[2]),
                    "bx": float(value[0]),
                    "by": float(value[1]),
                    "bz": float(value[2]),
                }
            )
    return records


def write_timeseries_metadata(metadata: dict, output_path: str | Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
