from __future__ import annotations

from csv import DictWriter
import json
from pathlib import Path

import numpy as np

from sparse_recon.sampling.sampler import sample_field_nearest
from sparse_recon.types import FieldSnapshot


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


def build_hs_color_map(spacecraft_labels: list[str]) -> dict[str, str]:
    if len(spacecraft_labels) > len(HS_COLORS):
        raise ValueError(
            f"HelioSwarm color map supports at most {len(HS_COLORS)} spacecraft; "
            f"got {len(spacecraft_labels)}"
        )
    return {label: HS_COLORS[i] for i, label in enumerate(spacecraft_labels)}


def rescale_field_snapshot_to_box(
    field: FieldSnapshot,
    *,
    sim_box_rho_p: tuple[float, float, float],
) -> FieldSnapshot:
    sim_box = np.asarray(sim_box_rho_p, dtype=float)
    if sim_box.shape != (3,) or np.any(sim_box <= 0):
        raise ValueError(
            "sim_box_rho_p must contain three positive axis lengths; "
            f"got {sim_box_rho_p}"
        )

    coords = np.asarray(field.coords, dtype=float) * sim_box[None, :]
    axes = None
    if field.axes is not None and {"x", "y", "z"}.issubset(field.axes):
        axes = {
            "x": np.asarray(field.axes["x"], dtype=float) * sim_box[0],
            "y": np.asarray(field.axes["y"], dtype=float) * sim_box[1],
            "z": np.asarray(field.axes["z"], dtype=float) * sim_box[2],
        }

    metadata = dict(field.metadata or {})
    metadata["grid_convention"] = "uniform_box_rho_p"
    metadata["sim_box_rho_p"] = sim_box.tolist()

    return FieldSnapshot(
        coords=coords,
        values=np.asarray(field.values),
        grid_shape=field.grid_shape,
        axes=axes,
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


def sample_timeseries_from_trajectory(
    field: FieldSnapshot,
    *,
    time_seconds: np.ndarray,
    wrapped_coords: np.ndarray,
    spacecraft_labels: list[str],
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
        sample_set = sample_field_nearest(field, coords_step)
        for label, coord, value in zip(spacecraft_labels, sample_set.coords, sample_set.values):
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


def write_timeseries_csv(records: list[dict], output_path: str | Path) -> None:
    fieldnames = [
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
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_timeseries_metadata(metadata: dict, output_path: str | Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
