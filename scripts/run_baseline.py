import argparse
import json
from pathlib import Path

from sparse_recon.datasets.synthetic import create_synthetic_field
from sparse_recon.datasets.helioswarm import load_helioswarm_sample_coords
from sparse_recon.methods.linear import LinearMethod
from sparse_recon.methods.nearest import NearestMethod
from sparse_recon.methods.rbf import RBFMethod
from sparse_recon.pipeline import run_sampling_experiment
from sparse_recon.sampling.geometries import generate_sampling_points
from sparse_recon.visualization import (
    plot_point_cloud_3d,
    plot_reconstruction_overview_2d,
    plot_reconstruction_overview_3d,
)


def build_method(name: str):
    if name == "nearest":
        return NearestMethod()
    if name == "linear":
        return LinearMethod()
    if name == "rbf":
        return RBFMethod()
    raise ValueError(f"Unknown method '{name}'")


def _parse_csv(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def save_experiment_figure(field, samples, predicted_values, title: str, output_path: Path):
    dim = field.coords.shape[1]
    if dim == 2:
        fig, _ = plot_reconstruction_overview_2d(
            field,
            samples,
            predicted_values,
            title=title,
        )
    elif dim == 3:
        fig, _ = plot_reconstruction_overview_3d(
            field,
            samples,
            predicted_values,
            title=title,
        )
    else:
        raise ValueError(f"Visualization is not implemented for dim={dim}")
    fig.savefig(output_path, dpi=150)


def save_helioswarm_formation_figures(
    formation,
    scaled_coords,
    output_dir: Path,
):
    physical_fig, _ = plot_point_cloud_3d(
        formation.relative_positions_km,
        labels=formation.spacecraft_labels,
        title="HelioSwarm Formation (km, hub-relative)",
        axis_labels=("dX (km)", "dY (km)", "dZ (km)"),
    )
    physical_fig.savefig(output_dir / "helioswarm_physical.png", dpi=150)

    scaled_fig, _ = plot_point_cloud_3d(
        scaled_coords,
        labels=formation.spacecraft_labels,
        title="HelioSwarm Formation (scaled to box)",
        axis_labels=("x", "y", "z"),
    )
    scaled_fig.savefig(output_dir / "helioswarm_scaled.png", dpi=150)


def run_benchmark_matrix(args) -> list[dict]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.hs_path) != bool(args.hs_time):
        raise ValueError("HelioSwarm mode requires both --hs-path and --hs-time")

    records = []
    experiment_index = 0

    methods = _parse_csv(args.methods, str)
    sample_counts = _parse_csv(args.sample_counts, int)
    geometries = _parse_csv(args.geometries, str)
    noise_levels = _parse_csv(args.noise_levels, float)

    field = create_synthetic_field(
        kind=args.field_kind,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        seed=args.field_seed,
        noise_sigma=args.field_noise_sigma,
    )
    dim = field.coords.shape[1]
    use_helioswarm = bool(args.hs_path and args.hs_time)

    helioswarm_coords = None
    helioswarm_formation = None
    helioswarm_transform = None
    if use_helioswarm:
        helioswarm_coords, helioswarm_formation, helioswarm_transform = load_helioswarm_sample_coords(
            args.hs_path,
            args.hs_time,
            include_hub=args.include_hub,
        )
        if helioswarm_coords.shape[1] != dim:
            raise ValueError(
                f"HelioSwarm coordinates are {helioswarm_coords.shape[1]}D but field is {dim}D"
            )

    sample_count_values = [helioswarm_coords.shape[0]] if use_helioswarm else sample_counts
    geometry_values = ["helioswarm"] if use_helioswarm else geometries

    for method_name in methods:
        for sample_count in sample_count_values:
            for geometry in geometry_values:
                for noise_sigma in noise_levels:
                    experiment_index += 1
                    experiment_name = (
                        f"{experiment_index:03d}_{args.field_kind}_{method_name}_{geometry}"
                        f"_{sample_count}_noise{noise_sigma:g}"
                    )
                    experiment_dir = output_dir / experiment_name
                    experiment_dir.mkdir(parents=True, exist_ok=True)

                    if use_helioswarm:
                        sample_coords = helioswarm_coords
                        save_helioswarm_formation_figures(
                            helioswarm_formation,
                            helioswarm_coords,
                            experiment_dir,
                        )
                    else:
                        sample_coords = generate_sampling_points(
                            geometry=geometry,
                            n_points=sample_count,
                            dim=dim,
                            seed=args.sample_seed + experiment_index,
                        )
                    method = build_method(method_name)
                    samples, result = run_sampling_experiment(
                        field,
                        sample_coords,
                        method,
                        noise_sigma=noise_sigma,
                        noise_seed=args.noise_seed + experiment_index,
                        metadata={
                            "experiment": {
                                "field_kind": args.field_kind,
                                "geometry": geometry,
                                "sample_count": sample_count,
                                "noise_sigma": noise_sigma,
                                "experiment_name": experiment_name,
                                "dim": dim,
                            }
                        }
                        | (
                            {
                                "helioswarm": {
                                    **helioswarm_formation.metadata,
                                    "raw_positions_km": helioswarm_formation.raw_positions_km.tolist(),
                                    "relative_positions_km": helioswarm_formation.relative_positions_km.tolist(),
                                    "scaled_positions_box": helioswarm_coords.tolist(),
                                    "transform": helioswarm_transform,
                                }
                            }
                            if use_helioswarm
                            else {}
                        ),
                    )

                    record = {
                        "experiment_name": experiment_name,
                        "method": result.method,
                        "metrics": result.metrics,
                        "metadata": result.metadata,
                    }
                    records.append(record)

                    with open(experiment_dir / "metrics.json", "w", encoding="utf-8") as f:
                        json.dump(record, f, indent=2)

                    save_experiment_figure(
                        field,
                        samples,
                        result.predicted_values,
                        title=experiment_name,
                        output_path=experiment_dir / "overview.png",
                    )

    with open(output_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic sparse reconstruction benchmarks.")
    parser.add_argument("--field-kind", default="smooth")
    parser.add_argument("--field-seed", type=int, default=0)
    parser.add_argument("--field-noise-sigma", type=float, default=0.05)
    parser.add_argument("--methods", default="rbf,linear,nearest")
    parser.add_argument("--sample-counts", default="50")
    parser.add_argument("--geometries", default="random,clustered,multi_probe_like")
    parser.add_argument("--noise-levels", default="0.0,0.02")
    parser.add_argument("--sample-seed", type=int, default=1)
    parser.add_argument("--noise-seed", type=int, default=11)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=24)
    parser.add_argument("--hs-path", default=None)
    parser.add_argument("--hs-time", default=None)
    parser.add_argument("--include-hub", action="store_true")
    parser.add_argument("--output-dir", default="results/baseline_demo")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    records = run_benchmark_matrix(args)

    print(f"Saved {len(records)} experiment records to {args.output_dir}")
    for record in records:
        print(record["experiment_name"], record["metrics"])


if __name__ == "__main__":
    main()
