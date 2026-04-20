import matplotlib.pyplot as plt
import numpy as np

from sparse_recon.types import FieldSnapshot, SampleSet


def _scatter_projected_samples(ax, projected_samples: np.ndarray, samples: SampleSet):
    projected_samples = np.asarray(projected_samples, dtype=float)
    sample_metadata = samples.metadata or {}
    spacecraft_labels = sample_metadata.get("spacecraft_labels")
    include_hub = bool(sample_metadata.get("include_hub", False))

    hub_index = None
    if spacecraft_labels:
        try:
            hub_index = list(spacecraft_labels).index("H")
        except ValueError:
            hub_index = None
    elif include_hub and len(projected_samples) == 9:
        hub_index = len(projected_samples) - 1

    node_mask = np.ones(len(projected_samples), dtype=bool)
    if hub_index is not None:
        node_mask[hub_index] = False

    if np.any(node_mask):
        ax.scatter(
            projected_samples[node_mask, 0],
            projected_samples[node_mask, 1],
            s=52,
            facecolors="#f4f1de",
            edgecolors="#111111",
            linewidths=1.25,
            alpha=0.95,
            zorder=3,
        )
    if hub_index is not None:
        ax.scatter(
            projected_samples[hub_index, 0],
            projected_samples[hub_index, 1],
            s=90,
            marker="*",
            facecolors="#ff5a36",
            edgecolors="#111111",
            linewidths=1.4,
            alpha=1.0,
            zorder=4,
        )


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


def plot_reconstruction_overview_3d(
    field: FieldSnapshot,
    samples: SampleSet,
    predicted_values: np.ndarray,
    title: str = "",
):
    nx, ny, nz = field.grid_shape
    x = np.asarray(field.axes["x"], dtype=float)
    y = np.asarray(field.axes["y"], dtype=float)
    z = np.asarray(field.axes["z"], dtype=float)
    truth_mag = np.linalg.norm(field.values, axis=1).reshape(nx, ny, nz)
    pred_mag = np.linalg.norm(predicted_values, axis=1).reshape(nx, ny, nz)
    error_mag = np.linalg.norm(predicted_values - field.values, axis=1).reshape(nx, ny, nz)

    ix = nx // 2
    iy = ny // 2
    iz = nz // 2

    slices = [
        (
            "xy",
            truth_mag[:, :, iz].T,
            pred_mag[:, :, iz].T,
            error_mag[:, :, iz].T,
            samples.coords[:, :2],
            [x[0], x[-1], y[0], y[-1]],
            ("x", "y"),
        ),
        (
            "xz",
            truth_mag[:, iy, :].T,
            pred_mag[:, iy, :].T,
            error_mag[:, iy, :].T,
            samples.coords[:, [0, 2]],
            [x[0], x[-1], z[0], z[-1]],
            ("x", "z"),
        ),
        (
            "yz",
            truth_mag[ix, :, :].T,
            pred_mag[ix, :, :].T,
            error_mag[ix, :, :].T,
            samples.coords[:, 1:],
            [y[0], y[-1], z[0], z[-1]],
            ("y", "z"),
        ),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    for row, (
        label,
        truth_slice,
        pred_slice,
        error_slice,
        projected_samples,
        extent,
        axis_labels,
    ) in enumerate(slices):
        im0 = axes[row, 0].imshow(
            truth_slice,
            origin="lower",
            cmap="viridis",
            aspect="auto",
            extent=extent,
        )
        axes[row, 0].set_title(f"{label.upper()} Truth | n={len(samples.coords)}")
        fig.colorbar(im0, ax=axes[row, 0], shrink=0.8)

        im1 = axes[row, 1].imshow(
            pred_slice,
            origin="lower",
            cmap="viridis",
            aspect="auto",
            extent=extent,
        )
        axes[row, 1].set_title(f"{label.upper()} Prediction")
        fig.colorbar(im1, ax=axes[row, 1], shrink=0.8)

        im2 = axes[row, 2].imshow(
            error_slice,
            origin="lower",
            cmap="magma",
            aspect="auto",
            extent=extent,
        )
        axes[row, 2].set_title(f"{label.upper()} Error")
        fig.colorbar(im2, ax=axes[row, 2], shrink=0.8)

        for col in range(3):
            _scatter_projected_samples(axes[row, col], projected_samples, samples)
            axes[row, col].set_xlim(extent[0], extent[1])
            axes[row, col].set_ylim(extent[2], extent[3])
            axes[row, col].set_xlabel(axis_labels[0])
            axes[row, col].set_ylabel(axis_labels[1])

    if title:
        fig.suptitle(title)
    return fig, axes


def plot_point_cloud_3d(
    coords: np.ndarray,
    *,
    labels: list[str] | None = None,
    title: str = "",
    axis_labels: tuple[str, str, str] = ("X", "Y", "Z"),
):
    coords = np.asarray(coords, dtype=float)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=50, c=np.arange(len(coords)), cmap="tab10")

    if labels is not None:
        for label, point in zip(labels, coords):
            ax.text(point[0], point[1], point[2], label)

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.set_title(title)
    return fig, ax


def plot_hs_timeseries_components(
    records: list[dict],
    *,
    spacecraft_labels: list[str],
    spacecraft_colors: dict[str, str],
    title: str = "",
):
    component_specs = [
        ("bx", "Bx"),
        ("by", "By"),
        ("bz", "Bz"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)

    for ax, (key, label) in zip(axes, component_specs):
        for spacecraft_label in spacecraft_labels:
            label_records = [record for record in records if record["spacecraft_label"] == spacecraft_label]
            times = [record["time_seconds"] for record in label_records]
            values = [record[key] for record in label_records]
            ax.plot(
                times,
                values,
                label=spacecraft_label,
                color=spacecraft_colors[spacecraft_label],
                linewidth=1.8,
            )
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("time_seconds")
    axes[0].legend(loc="upper right", ncol=3, fontsize=9)
    if title:
        fig.suptitle(title)
    return fig, axes
