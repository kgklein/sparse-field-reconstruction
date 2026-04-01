import matplotlib.pyplot as plt
import numpy as np
from sparse_recon.types import FieldSnapshot, SampleSet


def plot_field_and_samples_2d(field: FieldSnapshot, samples: SampleSet, title: str = ""):
    nx, ny = field.grid_shape
    x = field.axes["x"]
    y = field.axes["y"]

    xx = field.coords[:, 0].reshape(nx, ny)
    yy = field.coords[:, 1].reshape(nx, ny)
    bx = field.values[:, 0].reshape(nx, ny)
    by = field.values[:, 1].reshape(nx, ny)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.quiver(xx, yy, bx, by, alpha=0.5)
    ax.scatter(samples.coords[:, 0], samples.coords[:, 1], s=30)
    ax.set_title(title or "Field and sparse samples")
    return fig, ax
