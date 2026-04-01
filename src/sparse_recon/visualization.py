import matplotlib.pyplot as plt
import numpy as np

from sparse_recon.types import FieldSnapshot, SampleSet


def plot_field_and_samples_2d(
    field: FieldSnapshot,
    samples: SampleSet,
    title: str = "",
):
    nx, ny = field.grid_shape

    xx = field.coords[:, 0].reshape(nx, ny)
    yy = field.coords[:, 1].reshape(nx, ny)
    bx = field.values[:, 0].reshape(nx, ny)
    by = field.values[:, 1].reshape(nx, ny)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.quiver(xx, yy, bx, by, alpha=0.5)
    ax.scatter(samples.coords[:, 0], samples.coords[:, 1], s=30)
    ax.set_title(title or "Field and sparse samples")
    return fig, ax


def plot_reconstruction_overview_2d(
    field: FieldSnapshot,
    samples: SampleSet,
    predicted_values: np.ndarray,
    title: str = "",
):
    nx, ny = field.grid_shape
    xx = field.coords[:, 0].reshape(nx, ny)
    yy = field.coords[:, 1].reshape(nx, ny)

    truth_bx = field.values[:, 0].reshape(nx, ny)
    truth_by = field.values[:, 1].reshape(nx, ny)
    pred_bx = predicted_values[:, 0].reshape(nx, ny)
    pred_by = predicted_values[:, 1].reshape(nx, ny)

    error_mag = np.linalg.norm(predicted_values - field.values, axis=1).reshape(nx, ny)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    axes[0].quiver(xx, yy, truth_bx, truth_by, alpha=0.5)
    axes[0].scatter(samples.coords[:, 0], samples.coords[:, 1], s=16, color="tab:red")
    axes[0].set_title("Truth + Samples")

    axes[1].quiver(xx, yy, pred_bx, pred_by, alpha=0.5)
    axes[1].set_title("Prediction")

    im = axes[2].pcolormesh(xx, yy, error_mag, shading="auto", cmap="magma")
    axes[2].set_title("Error Magnitude")
    fig.colorbar(im, ax=axes[2], shrink=0.85)

    for ax in axes:
        ax.set_xlim(float(xx.min()), float(xx.max()))
        ax.set_ylim(float(yy.min()), float(yy.max()))
        ax.set_aspect("equal")

    if title:
        fig.suptitle(title)
    return fig, axes
