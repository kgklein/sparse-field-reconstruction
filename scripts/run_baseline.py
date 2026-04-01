import argparse
import json
from pathlib import Path

from sparse_recon.datasets.synthetic import create_synthetic_field
from sparse_recon.methods.linear import LinearMethod
from sparse_recon.methods.nearest import NearestMethod
from sparse_recon.methods.rbf import RBFMethod
from sparse_recon.pipeline import run_sampling_experiment
from sparse_recon.sampling.geometries import generate_sampling_points
from sparse_recon.visualization import plot_reconstruction_overview_2d


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


def run_benchmark_matrix(args) -> list[dict]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        seed=args.field_seed,
        noise_sigma=args.field_noise_sigma,
    )

    for method_name in methods:
        for sample_count in sample_counts:
            for geometry in geometries:
                for noise_sigma in noise_levels:
                    experiment_index += 1
                    experiment_name = (
                        f"{experiment_index:03d}_{args.field_kind}_{method_name}"
                        f"_{geometry}_{sample_count}_noise{noise_sigma:g}"
                    )
                    experiment_dir = output_dir / experiment_name
                    experiment_dir.mkdir(parents=True, exist_ok=True)

                    sample_coords = generate_sampling_points(
                        geometry=geometry,
                        n_points=sample_count,
                        dim=2,
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
                            }
                        },
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

                    fig, _ = plot_reconstruction_overview_2d(
                        field,
                        samples,
                        result.predicted_values,
                        title=experiment_name,
                    )
                    fig.savefig(experiment_dir / "overview.png", dpi=150)

    with open(output_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic sparse reconstruction benchmarks.")
    parser.add_argument("--field-kind", default="smooth")
    parser.add_argument("--field-seed", type=int, default=0)
    parser.add_argument("--field-noise-sigma", type=float, default=0.05)
    parser.add_argument("--methods", default="rbf,linear,nearest,cubic_spline")
    parser.add_argument("--sample-counts", default="50")
    parser.add_argument("--geometries", default="random,clustered,multi_probe_like,flyby")
    parser.add_argument("--noise-levels", default="0.0,0.02")
    parser.add_argument("--sample-seed", type=int, default=1)
    parser.add_argument("--noise-seed", type=int, default=11)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
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
