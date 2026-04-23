from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse_recon.analysis.lag_tetrahedra import (
    construct_lag_tetrahedra,
    plot_lag_tetrahedra_baseline_projections,
    plot_lag_tetrahedra_epsilon_diagnostics,
    plot_lag_tetrahedra_ep_scatter,
    plot_lag_tetrahedra_yaglom_flux,
    prepare_saved_elsasser_lag_tetrahedra_input,
)


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    normalized = [item.strip() for item in value.split(",") if item.strip()]
    return normalized or None


def run_lag_tetrahedra_analysis(args) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing lag-tetrahedra run in {output_dir}", flush=True)

    prepared_input = prepare_saved_elsasser_lag_tetrahedra_input(
        args.timeseries_metadata,
        args.elsasser_pairs_npz,
        elsasser_pairs_json_path=args.elsasser_pairs_json,
        spacecraft_labels=_parse_csv(args.spacecraft_labels),
        time_index=args.time_index,
        time_seconds=args.time_seconds,
    )
    print(
        f"Loaded saved Elsasser products for {len(prepared_input.spacecraft_labels)} spacecraft "
        f"at time index {prepared_input.time_index}",
        flush=True,
    )

    result = construct_lag_tetrahedra(
        prepared_input,
        zero_barycenter_atol=float(args.zero_barycenter_atol),
        max_d_ep=args.dep_max,
    )
    payload = result.to_dict()
    payload["metadata"]["output"] = {
        "json_path": str(output_dir / "lag_tetrahedra.json"),
        "ep_plot_path": (
            str(output_dir / "lag_tetrahedra_ep_scatter.png")
            if args.plot
            else None
        ),
        "yaglom_flux_plot_path": (
            str(output_dir / "lag_tetrahedra_yaglom_flux.png")
            if args.plot
            else None
        ),
        "epsilon_plot_path": (
            str(output_dir / "lag_tetrahedra_epsilon_diagnostics.png")
            if args.plot
            else None
        ),
        "baseline_projection_plot_path": (
            str(output_dir / "lag_tetrahedra_baseline_projections.png")
            if args.plot
            else None
        ),
    }
    with open(output_dir / "lag_tetrahedra.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote lag-tetrahedra summary to {output_dir / 'lag_tetrahedra.json'}", flush=True)
    print(
        f"Valid tetrahedra: {payload['metadata']['counts']['valid_tetrahedra_for_gradient']}",
        flush=True,
    )
    print(
        f"Passing d_EP cut: {payload['metadata']['counts']['passing_quality_cut']}",
        flush=True,
    )
    print(
        f"epsilon+ mean/std: {payload['summary_statistics']['epsilon_plus']['mean']} / "
        f"{payload['summary_statistics']['epsilon_plus']['std']}",
        flush=True,
    )
    print(
        f"epsilon- mean/std: {payload['summary_statistics']['epsilon_minus']['mean']} / "
        f"{payload['summary_statistics']['epsilon_minus']['std']}",
        flush=True,
    )

    if args.plot:
        fig, _ = plot_lag_tetrahedra_ep_scatter(
            result,
            highlight_tetrahedron_index=args.highlight_tetrahedron_index,
            title="Lag-Tetrahedra E-P Diagnostic",
        )
        fig.savefig(output_dir / "lag_tetrahedra_ep_scatter.png", dpi=150)
        arrow_display_fraction = float(getattr(args, "arrow_display_fraction", 0.25))
        flux_fig, _, flux_plotting = plot_lag_tetrahedra_yaglom_flux(
            result,
            max_arrows=int(args.max_arrows),
            display_fraction_of_lag_span=arrow_display_fraction,
            highlight_tetrahedron_index=args.highlight_tetrahedron_index,
            title="Lag-Space Time-Averaged Yaglom Flux",
        )
        flux_fig.savefig(output_dir / "lag_tetrahedra_yaglom_flux.png", dpi=150)
        payload["metadata"]["plotting"] = {
            "yaglom_flux_selected_keys": flux_plotting["selected_keys"],
            "yaglom_flux_selection_rule": (
                "sort directed lag points by lag magnitude then key; keep every kth point up to max_arrows"
            ),
            "yaglom_flux_max_arrows": int(args.max_arrows),
            "yaglom_flux_arrow_display_fraction": arrow_display_fraction,
            "highlight_tetrahedron_index": args.highlight_tetrahedron_index,
            "highlighted_point_keys": flux_plotting["highlighted_point_keys"],
            "highlighted_vertex_coordinates": flux_plotting["highlighted_vertex_coordinates"],
        }
        epsilon_fig, _ = plot_lag_tetrahedra_epsilon_diagnostics(
            result,
            highlight_tetrahedron_index=args.highlight_tetrahedron_index,
            title="Cascade-Rate Diagnostics",
        )
        epsilon_fig.savefig(output_dir / "lag_tetrahedra_epsilon_diagnostics.png", dpi=150)
        baseline_fig, _, baseline_plotting = plot_lag_tetrahedra_baseline_projections(
            result,
            highlight_tetrahedron_index=args.highlight_tetrahedron_index,
            title="Lag-Baseline Log-Projection Diagnostic",
        )
        baseline_fig.savefig(output_dir / "lag_tetrahedra_baseline_projections.png", dpi=150)
        payload["metadata"]["plotting"]["baseline_projection_global_log_range"] = baseline_plotting[
            "global_log_range"
        ]
        for key in (
            "highlighted_pair_labels",
            "highlighted_vertex_coordinates",
            "highlighted_log_components",
            "highlighted_projection_panels",
            "highlighted_projection_had_overlaps",
        ):
            if key in baseline_plotting:
                payload["metadata"]["plotting"][f"baseline_projection_{key}"] = baseline_plotting[key]
        if args.highlight_tetrahedron_index is not None and "highlighted_pair_labels" in baseline_plotting:
            highlighted_pair_labels = baseline_plotting["highlighted_pair_labels"]
            unique_pair_count = len(set(highlighted_pair_labels))
            overlap_label = "yes" if baseline_plotting["highlighted_projection_had_overlaps"] else "no"
            print(
                "Baseline projection highlight: "
                f"{len(highlighted_pair_labels)} vertices across {unique_pair_count} unique pair labels; "
                f"overlapping projected vertices: {overlap_label}; "
                f"pair labels: {', '.join(highlighted_pair_labels)}",
                flush=True,
            )
        print(
            f"Saved lag-tetrahedra diagnostic plot to "
            f"{output_dir / 'lag_tetrahedra_ep_scatter.png'}",
            flush=True,
        )
        print(
            f"Saved lag-space Yaglom flux plot to "
            f"{output_dir / 'lag_tetrahedra_yaglom_flux.png'}",
            flush=True,
        )
        print(
            f"Saved epsilon diagnostics plot to "
            f"{output_dir / 'lag_tetrahedra_epsilon_diagnostics.png'}",
            flush=True,
        )
        print(
            f"Saved baseline projection plot to "
            f"{output_dir / 'lag_tetrahedra_baseline_projections.png'}",
            flush=True,
        )
        with open(output_dir / "lag_tetrahedra.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return {
        "json_path": str(output_dir / "lag_tetrahedra.json"),
        "ep_plot_path": (
            str(output_dir / "lag_tetrahedra_ep_scatter.png")
            if args.plot
            else None
        ),
        "yaglom_flux_plot_path": (
            str(output_dir / "lag_tetrahedra_yaglom_flux.png")
            if args.plot
            else None
        ),
        "epsilon_plot_path": (
            str(output_dir / "lag_tetrahedra_epsilon_diagnostics.png")
            if args.plot
            else None
        ),
        "baseline_projection_plot_path": (
            str(output_dir / "lag_tetrahedra_baseline_projections.png")
            if args.plot
            else None
        ),
        "metadata": payload["metadata"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct reusable lag tetrahedra from saved HelioSwarm Elsasser products."
    )
    parser.add_argument("--timeseries-metadata", required=True)
    parser.add_argument("--elsasser-pairs-npz", required=True)
    parser.add_argument("--elsasser-pairs-json", default=None)
    parser.add_argument("--time-index", type=int, default=None)
    parser.add_argument("--time-seconds", type=float, default=None)
    parser.add_argument(
        "--spacecraft-labels",
        default=None,
        help="Optional comma-separated spacecraft subset in metadata order.",
    )
    parser.add_argument(
        "--zero-barycenter-atol",
        type=float,
        default=1e-12,
        help="Tolerance for classifying tetrahedra as zero-barycenter; default is 1e-12.",
    )
    parser.add_argument(
        "--dep-max",
        type=float,
        default=0.85,
        help="Quality threshold used for epsilon statistics/plots; tetrahedra with d_EP < threshold pass.",
    )
    parser.add_argument(
        "--max-arrows",
        type=int,
        default=24,
        help="Maximum number of lag-space Yaglom quiver arrows to display.",
    )
    parser.add_argument(
        "--arrow-display-fraction",
        type=float,
        default=0.25,
        help=(
            "Target length of the longest Yaglom arrow as a fraction of the plotted lag span; "
            "larger values make arrows visually longer."
        ),
    )
    parser.add_argument(
        "--highlight-tetrahedron-index",
        type=int,
        default=None,
        help="Optional retained-tetrahedron index to highlight across diagnostics.",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    result = run_lag_tetrahedra_analysis(args)
    print(f"Saved lag-tetrahedra outputs to {result['json_path']}")


if __name__ == "__main__":
    main()
