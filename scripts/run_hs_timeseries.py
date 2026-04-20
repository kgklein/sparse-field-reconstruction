from __future__ import annotations

import argparse
from pathlib import Path

from sparse_recon.datasets.helioswarm import load_helioswarm_sample_coords
from sparse_recon.datasets.simulation_snapshot import SimulationSnapshotDataset
from sparse_recon.hs_timeseries import (
    HS_COLORS,
    build_hs_color_map,
    generate_moving_spacecraft_trajectory,
    rescale_field_snapshot_to_box,
    sample_timeseries_from_trajectory,
    write_timeseries_csv,
    write_timeseries_metadata,
)
from sparse_recon.visualization import plot_hs_timeseries_components


def _validate_positive(value, *, name: str) -> float:
    if value is None or value <= 0:
        raise ValueError(f"{name} must be provided as a positive value")
    return float(value)


def _load_initial_hs_formation(args):
    sim_box_rho_p = (
        _validate_positive(args.sim_box_x, name="--sim-box-x"),
        _validate_positive(args.sim_box_y, name="--sim-box-y"),
        _validate_positive(args.sim_box_z, name="--sim-box-z"),
    )
    rho_p_km = _validate_positive(args.rho_p_km, name="--rho-p-km")

    coords_rho_p, formation, transform = load_helioswarm_sample_coords(
        args.hs_path,
        args.hs_time,
        include_hub=True,
        rho_p_km=rho_p_km,
        sim_box_rho_p=sim_box_rho_p,
    )

    if len(formation.spacecraft_labels) != 9:
        raise ValueError(
            "Moving-observatory HelioSwarm runs require exactly 9 valid spacecraft "
            f"including the hub; got {len(formation.spacecraft_labels)}"
        )
    if "H" not in formation.spacecraft_labels:
        raise ValueError("Moving-observatory HelioSwarm runs require hub spacecraft 'H'")

    return coords_rho_p, formation, transform, rho_p_km, sim_box_rho_p


def run_hs_timeseries(args) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_coords_rho_p, formation, transform, rho_p_km, sim_box_rho_p = _load_initial_hs_formation(
        args
    )
    field = rescale_field_snapshot_to_box(
        SimulationSnapshotDataset(args.simulation_path).load(),
        sim_box_rho_p=sim_box_rho_p,
    )

    time_seconds, unwrapped_coords, wrapped_coords, motion_metadata = generate_moving_spacecraft_trajectory(
        initial_coords_rho_p,
        velocity_km_s=(args.vx_kms, args.vy_kms, args.vz_kms),
        rho_p_km=rho_p_km,
        sim_box_rho_p=sim_box_rho_p,
        dt_seconds=_validate_positive(args.dt_seconds, name="--dt-seconds"),
        n_steps=int(_validate_positive(args.n_steps, name="--n-steps")),
    )

    spacecraft_colors = build_hs_color_map(formation.spacecraft_labels)
    records = sample_timeseries_from_trajectory(
        field,
        time_seconds=time_seconds,
        wrapped_coords=wrapped_coords,
        spacecraft_labels=formation.spacecraft_labels,
    )

    metadata = {
        "run_type": "helioswarm_timeseries",
        "sampling_mode": "nearest",
        "input": {
            "simulation_path": args.simulation_path,
            "hs_path": args.hs_path,
            "hs_time": args.hs_time,
        },
        "helioswarm": {
            **formation.metadata,
            "spacecraft_labels": formation.spacecraft_labels,
            "spacecraft_colors": spacecraft_colors,
            "color_sequence": HS_COLORS[: len(formation.spacecraft_labels)],
            "raw_positions_km": formation.raw_positions_km.tolist(),
            "relative_positions_km": formation.relative_positions_km.tolist(),
            "initial_coords_rho_p": initial_coords_rho_p.tolist(),
            "transform": transform,
        },
        "motion": {
            **motion_metadata,
            "initial_unwrapped_coords_rho_p": initial_coords_rho_p.tolist(),
            "final_unwrapped_coords_rho_p": unwrapped_coords[-1].tolist(),
            "final_wrapped_coords_rho_p": wrapped_coords[-1].tolist(),
        },
        "field": dict(field.metadata or {}),
        "output": {
            "csv_path": str(output_dir / "helioswarm_timeseries.csv"),
            "metadata_path": str(output_dir / "helioswarm_timeseries_metadata.json"),
            "plot_path": (
                str(output_dir / "helioswarm_timeseries.png")
                if args.plot_timeseries
                else None
            ),
        },
    }

    write_timeseries_csv(records, output_dir / "helioswarm_timeseries.csv")
    write_timeseries_metadata(metadata, output_dir / "helioswarm_timeseries_metadata.json")

    if args.plot_timeseries:
        fig, _ = plot_hs_timeseries_components(
            records,
            spacecraft_labels=formation.spacecraft_labels,
            spacecraft_colors=spacecraft_colors,
            title="HelioSwarm Moving Observatory Time Series",
        )
        fig.savefig(output_dir / "helioswarm_timeseries.png", dpi=150)

    return {
        "records": records,
        "metadata": metadata,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample a static simulation snapshot along a moving HelioSwarm formation."
    )
    parser.add_argument("--simulation-path", required=True)
    parser.add_argument("--hs-path", required=True)
    parser.add_argument("--hs-time", required=True)
    parser.add_argument("--rho-p-km", type=float, required=True)
    parser.add_argument("--sim-box-x", type=float, required=True)
    parser.add_argument("--sim-box-y", type=float, required=True)
    parser.add_argument("--sim-box-z", type=float, required=True)
    parser.add_argument("--vx-kms", type=float, required=True)
    parser.add_argument("--vy-kms", type=float, required=True)
    parser.add_argument("--vz-kms", type=float, required=True)
    parser.add_argument("--dt-seconds", type=float, required=True)
    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument("--plot-timeseries", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    result = run_hs_timeseries(args)

    print(
        f"Saved {len(result['records'])} time-series samples to "
        f"{result['metadata']['output']['csv_path']}"
    )


if __name__ == "__main__":
    main()
