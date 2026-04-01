from pathlib import Path
import json

from sparse_recon.datasets.synthetic import make_smooth_2d_vector_field
from sparse_recon.sampling.geometries import random_points_in_box
from sparse_recon.sampling.sampler import sample_field_nearest
from sparse_recon.methods.rbf import RBFMethod
from sparse_recon.pipeline import run_reconstruction
from sparse_recon.visualization import plot_field_and_samples_2d


def main():
    field = make_smooth_2d_vector_field(nx=64, ny=64, seed=0)
    sample_coords = random_points_in_box(n_points=50, dim=2, seed=1)
    samples = sample_field_nearest(field, sample_coords)

    method = RBFMethod(kernel="multiquadric")
    result = run_reconstruction(field, samples, method)

    outdir = Path("results/baseline_demo")
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "metrics.json", "w") as f:
        json.dump(result.metrics, f, indent=2)

    fig, _ = plot_field_and_samples_2d(field, samples, title="Baseline demo")
    fig.savefig(outdir / "samples.png", dpi=150)

    print("Saved results to", outdir)
    print(result.metrics)


if __name__ == "__main__":
    main()
