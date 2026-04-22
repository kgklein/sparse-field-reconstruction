from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparse_recon.analysis.structure_functions import (
    _compare_small_lambda_results,
    compute_structure_functions,
    plot_cube_sampling_diagnostics,
    plot_structure_functions,
    prepare_simulation_cube_local_reference_input,
    prepare_simulation_cube_structure_function_input,
    prepare_timeseries_structure_function_input,
)


def _validate_positive(value, *, name: str) -> float:
    if value is None or value <= 0:
        raise ValueError(f"{name} must be provided as a positive value")
    return float(value)


def _get_simulation_box_args(args) -> tuple[float, float, float]:
    return (
        _validate_positive(args.sim_box_x, name="--sim-box-x"),
        _validate_positive(args.sim_box_y, name="--sim-box-y"),
        _validate_positive(args.sim_box_z, name="--sim-box-z"),
    )


def run_structure_function_analysis(args) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing structure-function run in {output_dir}", flush=True)

    if args.input_mode == "interpolated_timeseries":
        if not args.timeseries_csv:
            raise ValueError("Interpolated time-series mode requires --timeseries-csv")
        prepared_input = prepare_timeseries_structure_function_input(
            args.timeseries_csv,
            metadata_path=args.timeseries_metadata,
        )
    elif args.input_mode == "simulation_cube":
        if not args.simulation_path:
            raise ValueError("Simulation-cube mode requires --simulation-path")
        prepared_input = prepare_simulation_cube_structure_function_input(
            args.simulation_path,
            sim_box_rho_p=_get_simulation_box_args(args),
            n_lambda_bins=args.n_lambda_bins,
            lambda_min=args.lambda_min,
            lambda_max=args.lambda_max,
            candidate_pairs=args.cube_candidate_pairs,
            target_pairs_per_bin=args.cube_target_pairs_per_bin,
            random_seed=args.cube_random_seed,
        )
    else:
        raise ValueError(f"Unknown input mode '{args.input_mode}'")

    print(
        f"Loaded {prepared_input.metadata['input_mode']} input with "
        f"{len(prepared_input.pair_positions)} candidate pairs",
        flush=True,
    )
    resolved_lambda_min = args.lambda_min
    resolved_lambda_max = args.lambda_max
    if args.input_mode == "simulation_cube":
        sampling_metadata = prepared_input.metadata.get("sampling", {})
        resolved_lambda_min = sampling_metadata.get("resolved_lambda_min", resolved_lambda_min)
        resolved_lambda_max = sampling_metadata.get("resolved_lambda_max", resolved_lambda_max)

    result = compute_structure_functions(
        prepared_input.pair_positions,
        prepared_input.pair_fields,
        component=args.field_component,
        max_order=args.max_order,
        local_b_definition="pair_midpoint",
        n_lambda_bins=args.n_lambda_bins,
        lambda_min=resolved_lambda_min,
        lambda_max=resolved_lambda_max,
        input_metadata=prepared_input.metadata,
        n_steps=prepared_input.n_steps,
    )

    reference_result = None
    if args.input_mode == "simulation_cube" and args.cube_compare_local_reference:
        reference_input = prepare_simulation_cube_local_reference_input(
            args.simulation_path,
            sim_box_rho_p=_get_simulation_box_args(args),
            max_offset=args.cube_reference_max_offset,
        )
        reference_result = compute_structure_functions(
            reference_input.pair_positions,
            reference_input.pair_fields,
            component=args.field_component,
            max_order=args.max_order,
            local_b_definition="pair_midpoint",
            n_lambda_bins=args.n_lambda_bins,
            lambda_min=resolved_lambda_min,
            lambda_max=resolved_lambda_max,
            input_metadata=reference_input.metadata,
            n_steps=reference_input.n_steps,
        )
        comparison_metadata, comparison_warnings = _compare_small_lambda_results(
            result,
            reference_result,
        )
        result.metadata.setdefault("diagnostics", {})["local_reference_comparison"] = comparison_metadata
        result.metadata["warnings"].extend(comparison_warnings)

    output_payload = result.to_dict()
    output_payload["metadata"]["output"] = {
        "json_path": str(output_dir / "structure_functions.json"),
        "plot_path": (
            str(output_dir / "structure_functions.png")
            if args.plot
            else None
        ),
        "diagnostics_json_path": (
            str(output_dir / "structure_functions_diagnostics.json")
            if args.cube_diagnostics and args.input_mode == "simulation_cube"
            else None
        ),
        "diagnostics_plot_path": (
            str(output_dir / "structure_functions_diagnostics.png")
            if args.cube_diagnostics and args.input_mode == "simulation_cube"
            else None
        ),
    }
    with open(output_dir / "structure_functions.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Wrote structure-function data to {output_dir / 'structure_functions.json'}", flush=True)

    if args.cube_diagnostics and args.input_mode == "simulation_cube":
        diagnostics_payload = {
            "cube_sampling": result.metadata.get("diagnostics", {}).get("cube_sampling"),
            "local_reference_comparison": result.metadata.get("diagnostics", {}).get("local_reference_comparison"),
            "warnings": result.metadata.get("warnings", []),
        }
        with open(output_dir / "structure_functions_diagnostics.json", "w", encoding="utf-8") as f:
            json.dump(diagnostics_payload, f, indent=2)
        print(
            f"Wrote structure-function diagnostics to {output_dir / 'structure_functions_diagnostics.json'}",
            flush=True,
        )

    if args.plot:
        fig, _ = plot_structure_functions(
            result,
            title=f"Structure Functions ({args.input_mode})",
        )
        fig.savefig(output_dir / "structure_functions.png", dpi=150)
        print(f"Saved structure-function plot to {output_dir / 'structure_functions.png'}", flush=True)
    if args.cube_diagnostics and args.input_mode == "simulation_cube":
        diagnostics_fig, _ = plot_cube_sampling_diagnostics(
            result,
            reference_result=reference_result,
            title="Simulation-Cube Sampling Diagnostics",
        )
        diagnostics_fig.savefig(output_dir / "structure_functions_diagnostics.png", dpi=150)
        print(
            f"Saved structure-function diagnostics plot to {output_dir / 'structure_functions_diagnostics.png'}",
            flush=True,
        )

    return {
        "json_path": str(output_dir / "structure_functions.json"),
        "plot_path": (
            str(output_dir / "structure_functions.png")
            if args.plot
            else None
        ),
        "metadata": output_payload["metadata"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute turbulence structure functions from interpolated time series or simulation cubes."
    )
    parser.add_argument(
        "--input-mode",
        required=True,
        choices=["interpolated_timeseries", "simulation_cube"],
    )
    parser.add_argument("--timeseries-csv", default=None)
    parser.add_argument("--timeseries-metadata", default=None)
    parser.add_argument("--simulation-path", default=None)
    parser.add_argument("--sim-box-x", type=float, default=None)
    parser.add_argument("--sim-box-y", type=float, default=None)
    parser.add_argument("--sim-box-z", type=float, default=None)
    parser.add_argument("--field-component", default="bx", choices=["bx"])
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--n-lambda-bins", type=int, default=24)
    parser.add_argument("--lambda-min", type=float, default=None)
    parser.add_argument("--lambda-max", type=float, default=None)
    parser.add_argument("--cube-candidate-pairs", type=int, default=200000)
    parser.add_argument("--cube-target-pairs-per-bin", type=int, default=256)
    parser.add_argument("--cube-random-seed", type=int, default=0)
    parser.add_argument("--cube-diagnostics", action="store_true")
    parser.add_argument("--cube-compare-local-reference", action="store_true")
    parser.add_argument("--cube-reference-max-offset", type=int, default=1)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    result = run_structure_function_analysis(args)
    print(f"Saved structure-function outputs to {result['json_path']}")


if __name__ == "__main__":
    main()
