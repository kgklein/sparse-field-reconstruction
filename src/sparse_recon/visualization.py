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
    times: np.ndarray,
    *,
    spacecraft_labels: list[str],
    spacecraft_colors: dict[str, str],
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
    title: str = "",
):
    component_specs = [
        (bx, "Bx"),
        (by, "By"),
        (bz, "Bz"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)

    times = np.asarray(times, dtype=float)
    for ax, (values_by_spacecraft, label) in zip(axes, component_specs):
        values_by_spacecraft = np.asarray(values_by_spacecraft, dtype=float)
        for index, spacecraft_label in enumerate(spacecraft_labels):
            ax.plot(
                times,
                values_by_spacecraft[index],
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


def _plot_hs_projected_positions(
    ax,
    coords: np.ndarray,
    *,
    spacecraft_labels: list[str],
    spacecraft_colors: dict[str, str],
    x_index: int,
    y_index: int,
    x_label: str,
    y_label: str,
    title: str,
    axis_limit: float,
):
    coords = np.asarray(coords, dtype=float)
    for label, point in zip(spacecraft_labels, coords):
        is_hub = label == "H"
        ax.scatter(
            point[x_index],
            point[y_index],
            s=78 if is_hub else 64,
            marker="o",
            facecolors=spacecraft_colors[label],
            edgecolors="#111111",
            linewidths=1.2 if is_hub else 1.0,
            zorder=3,
        )

    ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.35, zorder=1)
    ax.axvline(0.0, color="#444444", linewidth=0.8, alpha=0.35, zorder=1)
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.set_aspect("equal")


def _segment_wrapped_path(coords: np.ndarray, *, box_spans: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=float)
    x_values = coords[:, 0].astype(float, copy=True)
    z_values = coords[:, 2].astype(float, copy=True)

    if len(coords) < 2:
        return x_values, z_values

    x_span, z_span = box_spans
    jumps = np.zeros(len(coords), dtype=bool)
    if x_span > 0.0:
        jumps[1:] |= np.abs(np.diff(x_values)) > 0.5 * x_span
    if z_span > 0.0:
        jumps[1:] |= np.abs(np.diff(z_values)) > 0.5 * z_span

    x_values[jumps] = np.nan
    z_values[jumps] = np.nan
    return x_values, z_values


def plot_hs_timeseries_geometry(
    field,
    *,
    spacecraft_labels: list[str],
    spacecraft_colors: dict[str, str],
    hub_relative_positions_km: np.ndarray,
    initial_coords_rho_p: np.ndarray,
    trajectory_coords_rho_p: np.ndarray,
    title: str = "",
):
    hub_relative_positions_km = np.asarray(hub_relative_positions_km, dtype=float)
    initial_coords_rho_p = np.asarray(initial_coords_rho_p, dtype=float)
    trajectory_coords_rho_p = np.asarray(trajectory_coords_rho_p, dtype=float)

    if hub_relative_positions_km.ndim != 2 or hub_relative_positions_km.shape[1] != 3:
        raise ValueError(
            "hub_relative_positions_km must be shaped (n_spacecraft, 3); "
            f"got {hub_relative_positions_km.shape}"
        )
    if initial_coords_rho_p.ndim != 2 or initial_coords_rho_p.shape[1] != 3:
        raise ValueError(
            "initial_coords_rho_p must be shaped (n_spacecraft, 3); "
            f"got {initial_coords_rho_p.shape}"
        )
    if trajectory_coords_rho_p.ndim != 3 or trajectory_coords_rho_p.shape[-1] != 3:
        raise ValueError(
            "trajectory_coords_rho_p must be shaped (n_steps, n_spacecraft, 3); "
            f"got {trajectory_coords_rho_p.shape}"
        )

    if len(spacecraft_labels) != len(hub_relative_positions_km):
        raise ValueError("spacecraft_labels length must match hub_relative_positions_km rows")
    if initial_coords_rho_p.shape[0] != len(spacecraft_labels):
        raise ValueError("spacecraft_labels length must match initial_coords_rho_p rows")
    if trajectory_coords_rho_p.shape[1] != len(spacecraft_labels):
        raise ValueError("spacecraft_labels length must match trajectory_coords_rho_p columns")

    x = np.asarray(field.axes["x"], dtype=float)
    z = np.asarray(field.axes["z"], dtype=float)
    magnitude = np.linalg.norm(np.asarray(field.values, dtype=float), axis=-1)
    y_index = field.grid_shape[1] // 2
    xz_slice = magnitude[:, y_index, :].T
    relative_extent = float(np.max(np.abs(hub_relative_positions_km)))
    axis_limit = relative_extent * 1.05 if relative_extent > 0.0 else 1.0

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 0.05))
    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(grid[0, 0])
    axes[1, 0] = fig.add_subplot(grid[1, 0])
    axes[0, 1] = fig.add_subplot(grid[0, 1])
    axes[1, 1] = fig.add_subplot(grid[1, 1])
    colorbar_ax = fig.add_subplot(grid[0, 2])

    _plot_hs_projected_positions(
        axes[0, 0],
        hub_relative_positions_km,
        spacecraft_labels=spacecraft_labels,
        spacecraft_colors=spacecraft_colors,
        x_index=0,
        y_index=2,
        x_label="Delta X (km)",
        y_label="Delta Z (km)",
        title="Hub-Relative X-Z",
        axis_limit=axis_limit,
    )
    _plot_hs_projected_positions(
        axes[1, 0],
        hub_relative_positions_km,
        spacecraft_labels=spacecraft_labels,
        spacecraft_colors=spacecraft_colors,
        x_index=0,
        y_index=1,
        x_label="Delta X (km)",
        y_label="Delta Y (km)",
        title="Hub-Relative X-Y",
        axis_limit=axis_limit,
    )
    _plot_hs_projected_positions(
        axes[1, 1],
        hub_relative_positions_km,
        spacecraft_labels=spacecraft_labels,
        spacecraft_colors=spacecraft_colors,
        x_index=1,
        y_index=2,
        x_label="Delta Y (km)",
        y_label="Delta Z (km)",
        title="Hub-Relative Y-Z",
        axis_limit=axis_limit,
    )

    extent = [float(x[0]), float(x[-1]), float(z[0]), float(z[-1])]
    box_spans = (extent[1] - extent[0], extent[3] - extent[2])
    im = axes[0, 1].imshow(
        xz_slice,
        origin="lower",
        cmap="viridis",
        aspect="auto",
        extent=extent,
    )
    fig.colorbar(im, cax=colorbar_ax, label="|B|")
    trajectory_x = trajectory_coords_rho_p[:, :, 0]
    trajectory_z = trajectory_coords_rho_p[:, :, 2]
    x_min = min(extent[0], float(np.min(trajectory_x)))
    x_max = max(extent[1], float(np.max(trajectory_x)))
    z_min = min(extent[2], float(np.min(trajectory_z)))
    z_max = max(extent[3], float(np.max(trajectory_z)))
    x_pad = 0.03 * max(x_max - x_min, 1.0)
    z_pad = 0.03 * max(z_max - z_min, 1.0)
    for index, label in enumerate(spacecraft_labels):
        coords = trajectory_coords_rho_p[:, index, :]
        is_hub = label == "H"
        x_path, z_path = _segment_wrapped_path(coords, box_spans=box_spans)
        axes[0, 1].plot(
            x_path,
            z_path,
            color=spacecraft_colors[label],
            linewidth=1.8,
            alpha=0.95,
            zorder=3,
        )
        axes[0, 1].scatter(
            initial_coords_rho_p[index, 0],
            initial_coords_rho_p[index, 2],
            s=80 if is_hub else 65,
            marker="o",
            facecolors=spacecraft_colors[label],
            edgecolors="#111111",
            linewidths=1.2 if is_hub else 1.0,
            zorder=4,
        )
    axes[0, 1].set_xlabel("X (rho_p)")
    axes[0, 1].set_ylabel("Z (rho_p)")
    axes[0, 1].set_title("Simulation X-Z Slice")
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].set_xlim(x_min - x_pad, x_max + x_pad)
    axes[0, 1].set_ylim(z_min - z_pad, z_max + z_pad)
    axes[0, 1].set_aspect("equal")
    axes[0, 1].set_anchor("C")

    if title:
        fig.suptitle(title)
    return fig, axes
