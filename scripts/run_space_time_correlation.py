from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sparse_recon.analysis.space_time_correlation import (
    compute_hub_autocorrelation,
    compute_space_time_correlation,
    estimate_decorrelation_contour,
    estimate_decorrelation_scales_from_cuts,
    extract_axis_cuts,
    plot_space_time_correlation,
    prepare_timeseries_space_time_correlation_input,
)


def _parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or None


def _parse_float_csv(value: str | None) -> np.ndarray | None:
    if value is None:
        return None
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    return np.asarray(parsed, dtype=float) if parsed else None


def _resolve_max_lag_steps_from_args(args, *, times: np.ndarray) -> int | None:
    if args.max_tau_seconds is None:
        return args.max_lag_steps
    if args.max_tau_seconds < 0.0:
        raise ValueError("--max-tau-seconds must be non-negative")
    if len(times) < 2:
        raise ValueError("At least two time samples are required to resolve --max-tau-seconds")
    dt_seconds = float(times[1] - times[0])
    if dt_seconds <= 0.0:
        raise ValueError("Time samples must be strictly increasing to resolve --max-tau-seconds")
    max_lag_steps = int(np.floor(args.max_tau_seconds / dt_seconds))
    return max_lag_steps


def run_space_time_correlation_analysis(args) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing space-time correlation run in {output_dir}", flush=True)

    prepared_input = prepare_timeseries_space_time_correlation_input(
        args.timeseries_csv,
        metadata_path=args.timeseries_metadata,
        spacecraft_labels=_parse_csv(args.spacecraft_labels),
    )
    print(
        f"Loaded interpolated time-series input with {len(prepared_input.times)} time steps "
        f"and {len(prepared_input.spacecraft_labels)} spacecraft",
        flush=True,
    )
    resolved_max_lag_steps = _resolve_max_lag_steps_from_args(
        args,
        times=prepared_input.times,
    )

    result = compute_space_time_correlation(
        prepared_input,
        component=args.field_component,
        max_lag_steps=resolved_max_lag_steps,
        max_lag_fraction=args.max_lag_fraction,
        r_bin_edges=_parse_float_csv(args.r_bin_edges),
        tau_bin_edges=_parse_float_csv(args.tau_bin_edges),
        n_r_bins=args.n_r_bins,
        n_tau_bins=args.n_tau_bins,
        min_count_threshold=args.min_count_threshold,
        min_count_fraction=args.min_count_fraction,
    )
    hub_autocorrelation = compute_hub_autocorrelation(
        prepared_input,
        component=args.field_component,
        max_lag_steps=result.metadata["analysis"]["max_lag_steps"],
    )
    axis_cuts = extract_axis_cuts(result)
    decorrelation_scales = estimate_decorrelation_scales_from_cuts(result)
    contour = estimate_decorrelation_contour(result) if args.plot_contour else None

    output_payload = result.to_dict()
    output_payload["hub_autocorrelation"] = hub_autocorrelation.to_dict()
    output_payload["axis_cuts"] = axis_cuts
    output_payload["decorrelation_scales"] = decorrelation_scales
    output_payload["decorrelation_contour"] = contour
    output_payload["metadata"]["output"] = {
        "json_path": str(output_dir / "space_time_correlation.json"),
        "plot_path": (
            str(output_dir / "space_time_correlation.png")
            if args.plot
            else None
        ),
    }

    with open(output_dir / "space_time_correlation.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(
        f"Wrote space-time correlation data to {output_dir / 'space_time_correlation.json'}",
        flush=True,
    )

    if args.plot:
        fig, _ = plot_space_time_correlation(
            result,
            hub_autocorrelation=hub_autocorrelation,
            show_contour=args.plot_contour,
            title="Space-Time Decorrelation",
        )
        fig.savefig(output_dir / "space_time_correlation.png", dpi=150)
        print(
            f"Saved space-time correlation plot to {output_dir / 'space_time_correlation.png'}",
            flush=True,
        )

    return {
        "json_path": str(output_dir / "space_time_correlation.json"),
        "plot_path": (
            str(output_dir / "space_time_correlation.png")
            if args.plot
            else None
        ),
        "metadata": output_payload["metadata"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute a multipoint space-time decorrelation map from interpolated HelioSwarm time series."
    )
    parser.add_argument("--timeseries-csv", required=True)
    parser.add_argument("--timeseries-metadata", default=None)
    parser.add_argument(
        "--spacecraft-labels",
        required=True,
        help="Comma-separated label list such as H,N1,N2. The Hub H must be included.",
    )
    parser.add_argument("--field-component", default="bx", choices=["bx"])
    parser.add_argument("--max-lag-steps", type=int, default=None)
    parser.add_argument(
        "--max-tau-seconds",
        type=float,
        default=None,
        help=(
            "Maximum tau range in seconds. Converted to integer lag steps using the "
            "native cadence and takes precedence over --max-lag-steps when provided."
        ),
    )
    parser.add_argument("--max-lag-fraction", type=float, default=0.5)
    parser.add_argument("--n-r-bins", type=int, default=24)
    parser.add_argument("--n-tau-bins", type=int, default=None)
    parser.add_argument("--r-bin-edges", default=None)
    parser.add_argument("--tau-bin-edges", default=None)
    parser.add_argument("--min-count-threshold", type=int, default=None)
    parser.add_argument("--min-count-fraction", type=float, default=0.01)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-contour", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    result = run_space_time_correlation_analysis(args)
    print(f"Saved space-time correlation outputs to {result['json_path']}")


if __name__ == "__main__":
    main()
