from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cdflib import CDF, cdfepoch

FILL_VALUE_THRESHOLD = -1e30


@dataclass
class HelioSwarmTrajectoryData:
    epoch_tt2000: np.ndarray
    epoch_datetime: np.ndarray
    positions_km: np.ndarray
    spacecraft_labels: list[str]
    baselines_km: np.ndarray | None
    source_files: list[str]


@dataclass
class HelioSwarmFormationSnapshot:
    requested_time: str
    selected_time: str
    source_file: str
    spacecraft_labels: list[str]
    raw_positions_km: np.ndarray
    relative_positions_km: np.ndarray
    metadata: dict


def _list_cdf_files(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() != ".cdf":
            raise ValueError(f"Expected a CDF file, got: {path}")
        return [path]
    if path.is_dir():
        files = sorted(path.glob("*.cdf"))
        if not files:
            raise FileNotFoundError(f"No CDF files found in directory: {path}")
        return files
    raise FileNotFoundError(f"HelioSwarm path does not exist: {path}")


def _read_single_cdf(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    cdf = CDF(str(path))
    try:
        epoch_tt2000 = np.asarray(cdf.varget("Epoch"), dtype=np.int64)
        positions_km = np.asarray(cdf.varget("Position"), dtype=float)
    except ValueError as exc:
        if "No records found for variable" in str(exc):
            return np.array([], dtype=np.int64), np.empty((0, 0, 3)), [], None
        raise
    labels = np.asarray(cdf.varget("Spacecraft_Label")).astype(str).ravel().tolist()

    baselines_km = None
    if "Baseline" in cdf.cdf_info().zVariables:
        try:
            baselines_km = np.asarray(cdf.varget("Baseline"), dtype=float)
        except ValueError as exc:
            if "No records found for variable" not in str(exc):
                raise

    return epoch_tt2000, positions_km, labels, baselines_km


def load_helioswarm_trajectory_data(path: str | Path) -> HelioSwarmTrajectoryData:
    files = _list_cdf_files(path)

    all_epochs = []
    all_positions = []
    all_baselines = []
    source_files = []
    spacecraft_labels = None

    for cdf_path in files:
        epoch_tt2000, positions_km, labels, baselines_km = _read_single_cdf(cdf_path)
        if epoch_tt2000.size == 0:
            continue
        if spacecraft_labels is None:
            spacecraft_labels = labels
        elif spacecraft_labels != labels:
            raise ValueError("Spacecraft labels do not match across HelioSwarm CDF files")

        all_epochs.append(epoch_tt2000)
        all_positions.append(positions_km)
        if baselines_km is not None:
            all_baselines.append(baselines_km)
        source_files.extend([str(cdf_path)] * len(epoch_tt2000))

    if not all_epochs:
        raise ValueError(f"No HelioSwarm samples with Epoch records were found in: {path}")

    epoch_tt2000 = np.concatenate(all_epochs, axis=0)
    positions_km = np.concatenate(all_positions, axis=0)
    baselines_km = np.concatenate(all_baselines, axis=0) if all_baselines else None

    sort_idx = np.argsort(epoch_tt2000)
    epoch_tt2000 = epoch_tt2000[sort_idx]
    positions_km = positions_km[sort_idx]
    source_files = [source_files[i] for i in sort_idx]
    if baselines_km is not None:
        baselines_km = baselines_km[sort_idx]

    epoch_datetime = cdfepoch.to_datetime(epoch_tt2000)

    return HelioSwarmTrajectoryData(
        epoch_tt2000=epoch_tt2000,
        epoch_datetime=epoch_datetime,
        positions_km=positions_km,
        spacecraft_labels=spacecraft_labels or [],
        baselines_km=baselines_km,
        source_files=source_files,
    )


def _normalize_requested_time(requested_time: str) -> np.datetime64:
    normalized = requested_time.strip().replace("T", " ")
    if len(normalized) == 13:
        normalized = f"{normalized}:00:00"
    elif len(normalized) == 16:
        normalized = f"{normalized}:00"
    return np.datetime64(normalized)


def select_helioswarm_hour(
    data: HelioSwarmTrajectoryData,
    requested_time: str,
    *,
    include_hub: bool = False,
    spacecraft_subset: list[str] | None = None,
) -> HelioSwarmFormationSnapshot:
    requested_dt = _normalize_requested_time(requested_time)
    deltas = np.abs(data.epoch_datetime - requested_dt)
    index = int(np.argmin(deltas))

    labels = list(data.spacecraft_labels)
    positions = np.asarray(data.positions_km[index], dtype=float)

    try:
        hub_index = labels.index("H")
    except ValueError as exc:
        raise ValueError("HelioSwarm labels do not contain hub spacecraft 'H'") from exc

    valid_mask = np.all(positions > FILL_VALUE_THRESHOLD, axis=1)
    valid_mask &= np.array([label != "N/A" for label in labels], dtype=bool)
    if not np.any(valid_mask):
        raise ValueError("No valid HelioSwarm spacecraft positions found at selected time")

    relative_positions_km = positions - positions[hub_index]

    valid_labels = [label for label, is_valid in zip(labels, valid_mask) if is_valid]
    selected_labels = valid_labels
    if spacecraft_subset is not None:
        missing = sorted(set(spacecraft_subset) - set(valid_labels))
        if missing:
            raise ValueError(f"Unknown spacecraft labels requested: {', '.join(missing)}")
        selected_labels = spacecraft_subset
    elif not include_hub:
        selected_labels = [label for label in valid_labels if label != "H"]

    selection_idx = [labels.index(label) for label in selected_labels]
    raw_positions_selected = positions[selection_idx]
    relative_positions_selected = relative_positions_km[selection_idx]

    selected_time = str(data.epoch_datetime[index])
    return HelioSwarmFormationSnapshot(
        requested_time=str(requested_dt),
        selected_time=selected_time,
        source_file=data.source_files[index],
        spacecraft_labels=selected_labels,
        raw_positions_km=raw_positions_selected,
        relative_positions_km=relative_positions_selected,
        metadata={
            "requested_time": str(requested_dt),
            "selected_time": selected_time,
            "source_file": data.source_files[index],
            "spacecraft_labels": selected_labels,
            "frame": "GSE",
            "units": "km",
            "include_hub": include_hub,
            "spacecraft_count": len(selected_labels),
        },
    )


def scale_formation_to_box(
    coords_km: np.ndarray,
    *,
    box_center: float = 0.5,
    box_half_span: float = 0.45,
) -> tuple[np.ndarray, dict]:
    coords_km = np.asarray(coords_km, dtype=float)
    center_km = coords_km.mean(axis=0)
    centered = coords_km - center_km

    max_extent = float(np.max(np.abs(centered)))
    scale = box_half_span / max_extent if max_extent > 0 else 1.0
    scaled_coords = centered * scale + box_center
    scaled_coords = np.clip(scaled_coords, 0.0, 1.0)

    metadata = {
        "center_km": center_km.tolist(),
        "scale_factor": scale,
        "box_center": box_center,
        "box_half_span": box_half_span,
        "max_extent_km": max_extent,
        "original_extent_km": np.ptp(coords_km, axis=0).tolist(),
        "scaled_min": scaled_coords.min(axis=0).tolist(),
        "scaled_max": scaled_coords.max(axis=0).tolist(),
    }
    return scaled_coords, metadata


def place_formation_in_simulation_box(
    coords_km: np.ndarray,
    *,
    rho_p_km: float,
    sim_box_rho_p: tuple[float, float, float],
) -> tuple[np.ndarray, dict]:
    coords_km = np.asarray(coords_km, dtype=float)
    if rho_p_km <= 0:
        raise ValueError(f"rho_p_km must be positive; got {rho_p_km}")

    sim_box = np.asarray(sim_box_rho_p, dtype=float)
    if sim_box.shape != (3,) or np.any(sim_box <= 0):
        raise ValueError(
            "sim_box_rho_p must contain three positive axis lengths; "
            f"got {sim_box_rho_p}"
        )

    coords_rho_p = coords_km / rho_p_km
    centroid_rho_p = coords_rho_p.mean(axis=0)
    box_center_rho_p = sim_box / 2.0
    translated_coords = coords_rho_p - centroid_rho_p + box_center_rho_p

    metadata = {
        "mode": "km_to_rho_p_centered_box",
        "rho_p_km": float(rho_p_km),
        "sim_box_rho_p": sim_box.tolist(),
        "formation_centroid_rho_p": centroid_rho_p.tolist(),
        "placement_center_rho_p": box_center_rho_p.tolist(),
        "coords_rho_p_before_translation": coords_rho_p.tolist(),
        "coords_rho_p_after_translation": translated_coords.tolist(),
        "translated_min_rho_p": translated_coords.min(axis=0).tolist(),
        "translated_max_rho_p": translated_coords.max(axis=0).tolist(),
    }
    return translated_coords, metadata


def load_helioswarm_sample_coords(
    path: str | Path,
    requested_time: str,
    *,
    include_hub: bool = False,
    spacecraft_subset: list[str] | None = None,
    rho_p_km: float | None = None,
    sim_box_rho_p: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, HelioSwarmFormationSnapshot, dict]:
    data = load_helioswarm_trajectory_data(path)
    formation = select_helioswarm_hour(
        data,
        requested_time,
        include_hub=include_hub,
        spacecraft_subset=spacecraft_subset,
    )
    if rho_p_km is not None or sim_box_rho_p is not None:
        if rho_p_km is None or sim_box_rho_p is None:
            raise ValueError(
                "Both rho_p_km and sim_box_rho_p are required for simulation-space "
                "HelioSwarm placement"
            )
        sample_coords_box, transform_metadata = place_formation_in_simulation_box(
            formation.relative_positions_km,
            rho_p_km=rho_p_km,
            sim_box_rho_p=sim_box_rho_p,
        )
    else:
        sample_coords_box, transform_metadata = scale_formation_to_box(
            formation.relative_positions_km,
        )
    return sample_coords_box, formation, transform_metadata
