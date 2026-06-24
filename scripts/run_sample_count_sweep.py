import json
from pathlib import Path

import matplotlib.pyplot as plt

from sparse_recon.datasets.synthetic import create_synthetic_field
from sparse_recon.methods.cubic_spline import CubicSplineMethod
from sparse_recon.methods.linear import LinearMethod
from sparse_recon.methods.nearest import NearestMethod
from sparse_recon.methods.rbf import RBFMethod
from sparse_recon.pipeline import run_sampling_experiment
from sparse_recon.sampling.geometries import generate_sampling_points

SAMPLE_COUNTS = [10, 25, 50, 100, 200]
METHODS = [
    ("rbf", "#378ADD"),
    ("linear", "#1D9E75"),
    ("nearest", "#888780"),
    ("cubic_spline", "#D85A30"),
]
OUTPUT_DIR = Path("results/sample_count_sweep")


def build_method(name):
    if name == "rbf":
        return RBFMethod()
    if name == "linear":
        return LinearMethod()
    if name == "nearest":
        return NearestMethod()
    if name == "cubic_spline":
        return CubicSplineMethod()
    raise ValueError(f"Unknown method: {name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    field = create_synthetic_field(kind="smooth", nx=64, ny=64, seed=0)

    results = {name: [] for name, _ in METHODS}

    for name, _ in METHODS:
        for n in SAMPLE_COUNTS:
            sample_coords = generate_sampling_points(
                geometry="random", n_points=n, dim=2, seed=42
            )
            method = build_method(name)
            _, result = run_sampling_experiment(
                field,
                sample_coords,
                method,
                noise_sigma=0.0,
                noise_seed=0,
            )
            rel_l2 = result.metrics["relative_l2"]
            valid = result.metrics["valid_fraction"]
            results[name].append({"n": n, "relative_l2": rel_l2, "valid_fraction": valid})
            print(f"{name:<15} n={n:4d}  relative_l2={rel_l2:.4f}  valid={valid:.2f}")

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    colors = {name: color for name, color in METHODS}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for name, _ in METHODS:
        ns = [r["n"] for r in results[name]]
        errors = [r["relative_l2"] for r in results[name]]
        valid = [r["valid_fraction"] for r in results[name]]
        color = colors[name]
        ax1.plot(ns, errors, marker="o", linewidth=2, label=name, color=color)
        ax2.plot(ns, valid, marker="o", linewidth=2, label=name, color=color)

    ax1.set_xlabel("Number of sample points")
    ax1.set_ylabel("Relative L2 error")
    ax1.set_title("Reconstruction error vs sample count")
    ax1.set_xticks(SAMPLE_COUNTS)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Number of sample points")
    ax2.set_ylabel("Valid fraction")
    ax2.set_title("Domain coverage vs sample count")
    ax2.set_xticks(SAMPLE_COUNTS)
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("RBF vs Linear vs Nearest vs Cubic Spline — random geometry, smooth field", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "error_vs_sample_count.png", dpi=150)
    print(f"\nPlot saved to {OUTPUT_DIR / 'error_vs_sample_count.png'}")


if __name__ == "__main__":
    main()