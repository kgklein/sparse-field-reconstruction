from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sparse_recon.datasets.helioswarm import load_helioswarm_sample_coords
from sparse_recon.hs_timeseries import (
    HS_COLORS,
    build_hs_color_map,
    generate_moving_spacecraft_trajectory,
    load_structured_simulation_snapshot,
    sample_timeseries_from_trajectory,
    stream_timeseries_to_csv,
    write_timeseries_metadata,
)
from sparse_recon.visualization import (
    plot_hs_timeseries_components,
    plot_hs_timeseries_geometry,
)


def _validate_positive(value, *, name: str) -> float:
    if value is None or value <= 0:
        raise ValueError(f"{name} must be provided as a positive value")
    return float(value)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve_component_labels(args) -> tuple[str, str, str]:
    if args.simulation_component_labels:
        labels = _parse_csv(args.simulation_component_labels)
        if len(labels) != 3:
            raise ValueError("--simulation-component-labels must contain exactly three labels")
        return tuple(labels)
    if args.simulation_density_var:
        return (r"$u_x$", r"$u_y$", r"$u_z$")
    return (r"$B_x$", r"$B_y$", r"$B_z$")


def _resolve_density_label(args) -> str | None:
    if args.simulation_density_var:
        return r"$\rho$"
    return None


def _resolve_secondary_component_labels(args) -> tuple[str, str, str]:
    if args.secondary_timeseries_component_labels:
        labels = _parse_csv(args.secondary_timeseries_component_labels)
        if len(labels) != 3:
            raise ValueError(
                "--secondary-timeseries-component-labels must contain exactly three labels"
            )
        return tuple(labels)
    if args.secondary_timeseries_vector_vars:
        labels = _parse_csv(args.secondary_timeseries_vector_vars)
        if labels == ["bx", "by", "bz"]:
            return (r"$B_x$", r"$B_y$", r"$B_z$")
        return tuple(labels)
    return (r"$B_x$", r"$B_y$", r"$B_z$")


def _resolve_elsasser_component_labels(sign: str) -> tuple[str, str, str]:
    if sign == "+":
        return (r"$z^+_x$", r"$z^+_y$", r"$z^+_z$")
    if sign == "-":
        return (r"$z^-_x$", r"$z^-_y$", r"$z^-_z$")
    raise ValueError(f"Unsupported Elsasser sign '{sign}'")


def _resolve_geometry_vector_vars(args) -> list[str] | None:
    if args.geometry_vector_vars:
        return _parse_csv(args.geometry_vector_vars)
    return None


def _resolve_field_background_lua_path(args) -> str | None:
    return args.background_b_lua_path


def _looks_like_moment_variables(variable_names: list[str]) -> bool:
    return all(name in {"n", "nux", "nuy", "nuz", "p"} for name in variable_names)


def _secondary_series_is_magnetic(args) -> bool:
    if not args.secondary_timeseries_vector_vars:
        return False
    return [name.lower() for name in _parse_csv(args.secondary_timeseries_vector_vars)] == [
        "bx",
        "by",
        "bz",
    ]


def _compute_elsasser_series(
    *,
    velocity_result: dict,
    magnetic_result: dict,
) -> dict:
    density = velocity_result["density"]
    if density is None:
        raise ValueError("Elsasser calculation requires sampled density values")

    sqrt_rho = density**0.5
    magnetic_over_sqrt_rho = {
        component: magnetic_result[component] / sqrt_rho
        for component in ("bx", "by", "bz")
    }
    return {
        "zplus": {
            component: velocity_result[component] + magnetic_over_sqrt_rho[component]
            for component in ("bx", "by", "bz")
        },
        "zminus": {
            component: velocity_result[component] - magnetic_over_sqrt_rho[component]
            for component in ("bx", "by", "bz")
        },
    }


def _compute_elsasser_pairwise_increments(
    *,
    elsasser_result: dict,
    times: np.ndarray,
    unwrapped_positions: np.ndarray,
    spacecraft_labels: list[str],
) -> dict:
    times = np.asarray(times, dtype=float)
    unwrapped_positions = np.asarray(unwrapped_positions, dtype=float)
    if unwrapped_positions.ndim != 3 or unwrapped_positions.shape[-1] != 3:
        raise ValueError(
            "unwrapped_positions must be shaped (n_steps, n_spacecraft, 3); "
            f"got {unwrapped_positions.shape}"
        )
    if unwrapped_positions.shape[0] != len(times):
        raise ValueError("times length must match unwrapped_positions first dimension")
    if unwrapped_positions.shape[1] != len(spacecraft_labels):
        raise ValueError("spacecraft_labels length must match unwrapped_positions second dimension")

    pair_indices = np.array(
        [
            [left_index, right_index]
            for left_index in range(len(spacecraft_labels))
            for right_index in range(left_index + 1, len(spacecraft_labels))
        ],
        dtype=int,
    )
    pair_labels = np.array(
        [
            f"{spacecraft_labels[left_index]}__{spacecraft_labels[right_index]}"
            for left_index, right_index in pair_indices
        ],
        dtype=str,
    )

    separation_vectors = (
        unwrapped_positions[:, pair_indices[:, 1], :]
        - unwrapped_positions[:, pair_indices[:, 0], :]
    )
    separation_magnitudes = np.linalg.norm(separation_vectors, axis=-1)

    def _pairwise_delta(series: dict[str, np.ndarray]) -> np.ndarray:
        values = np.stack([series["bx"], series["by"], series["bz"]], axis=-1)
        values = np.transpose(values, (1, 0, 2))
        return values[:, pair_indices[:, 1], :] - values[:, pair_indices[:, 0], :]

    return {
        "times": times,
        "pair_indices": pair_indices,
        "pair_labels": pair_labels,
        "separation_vectors": separation_vectors,
        "separation_magnitudes": separation_magnitudes,
        "delta_zplus": _pairwise_delta(elsasser_result["zplus"]),
        "delta_zminus": _pairwise_delta(elsasser_result["zminus"]),
    }


def _write_elsasser_pair_outputs(
    *,
    output_dir: Path,
    pairwise_result: dict,
) -> tuple[str, str]:
    npz_path = output_dir / "helioswarm_timeseries_elsasser_pairs.npz"
    json_path = output_dir / "helioswarm_timeseries_elsasser_pairs.json"

    np.savez(
        npz_path,
        times=np.asarray(pairwise_result["times"], dtype=float),
        pair_indices=np.asarray(pairwise_result["pair_indices"], dtype=int),
        pair_labels=np.asarray(pairwise_result["pair_labels"], dtype=str),
        separation_vectors=np.asarray(pairwise_result["separation_vectors"], dtype=float),
        separation_magnitudes=np.asarray(pairwise_result["separation_magnitudes"], dtype=float),
        delta_zplus=np.asarray(pairwise_result["delta_zplus"], dtype=float),
        delta_zminus=np.asarray(pairwise_result["delta_zminus"], dtype=float),
    )
    json_payload = {
        "times_shape": list(np.asarray(pairwise_result["times"]).shape),
        "pair_indices_shape": list(np.asarray(pairwise_result["pair_indices"]).shape),
        "pair_labels": np.asarray(pairwise_result["pair_labels"], dtype=str).tolist(),
        "separation_vectors_shape": list(np.asarray(pairwise_result["separation_vectors"]).shape),
        "separation_magnitudes_shape": list(
            np.asarray(pairwise_result["separation_magnitudes"]).shape
        ),
        "delta_zplus_shape": list(np.asarray(pairwise_result["delta_zplus"]).shape),
        "delta_zminus_shape": list(np.asarray(pairwise_result["delta_zminus"]).shape),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    return str(npz_path), str(json_path)


def _load_initial_hs_formation(args):
    sim_box_rho_p = (
        _validate_positive(args.sim_box_x, name="--sim-box-x"),
        _validate_positive(args.sim_box_y, name="--sim-box-y"),
        _validate_positive(args.sim_box_z, name="--sim-box-z"),
    )
    rho_p_km = _validate_positive(args.rho_p_km, name="--rho-p-km")

    coords_rho_p, formation, transform = load_helioswarm_sample_coords(
        args.hs_path,
        args.hs_time,
        include_hub=True,
        rho_p_km=rho_p_km,
        sim_box_rho_p=sim_box_rho_p,
    )

    if len(formation.spacecraft_labels) != 9:
        raise ValueError(
            "Moving-observatory HelioSwarm runs require exactly 9 valid spacecraft "
            f"including the hub; got {len(formation.spacecraft_labels)}"
        )
    if "H" not in formation.spacecraft_labels:
        raise ValueError("Moving-observatory HelioSwarm runs require hub spacecraft 'H'")

    return coords_rho_p, formation, transform, rho_p_km, sim_box_rho_p


def run_hs_timeseries(args) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing moving-observatory run in {output_dir}", flush=True)

    initial_coords_rho_p, formation, transform, rho_p_km, sim_box_rho_p = _load_initial_hs_formation(
        args
    )
    print(
        f"Loaded HelioSwarm formation with {len(formation.spacecraft_labels)} spacecraft "
        f"at {formation.selected_time}",
        flush=True,
    )
    dt_seconds = _validate_positive(args.dt_seconds, name="--dt-seconds")
    n_steps = int(_validate_positive(args.n_steps, name="--n-steps"))
    sampling_method = args.sampling_method
    primary_vector_vars = _parse_csv(args.simulation_vector_vars)
    background_b_lua_path = _resolve_field_background_lua_path(args)

    if _looks_like_moment_variables(primary_vector_vars):
        if not args.ion_moments_path:
            raise ValueError(
                "Moment-like vector variables require --ion-moments-path so the code knows "
                "which packed moments file to read."
            )
        field = load_structured_simulation_snapshot(
            args.ion_moments_path,
            sim_box_rho_p=sim_box_rho_p,
            vector_variables=primary_vector_vars,
            packed_schema="moments",
        )
    else:
        field = load_structured_simulation_snapshot(
            args.simulation_path,
            sim_box_rho_p=sim_box_rho_p,
            vector_variables=primary_vector_vars,
            packed_schema="field" if str(args.simulation_path).lower().endswith(".bp") else None,
            background_b_lua_path=background_b_lua_path,
        )

    geometry_field = load_structured_simulation_snapshot(
        args.simulation_path,
        sim_box_rho_p=sim_box_rho_p,
        vector_variables=["bx", "by", "bz"],
        packed_schema="field" if str(args.simulation_path).lower().endswith(".bp") else None,
        background_b_lua_path=background_b_lua_path,
    )
    geometry_vector_vars = _resolve_geometry_vector_vars(args)
    if geometry_vector_vars is not None and geometry_vector_vars != ["bx", "by", "bz"]:
        geometry_field = load_structured_simulation_snapshot(
            args.simulation_path,
            sim_box_rho_p=sim_box_rho_p,
            vector_variables=geometry_vector_vars,
            packed_schema="field" if str(args.simulation_path).lower().endswith(".bp") else None,
            background_b_lua_path=background_b_lua_path,
        )
    normalization_field = None
    if args.simulation_density_var:
        if not args.ion_moments_path:
            raise ValueError("--simulation-density-var requires --ion-moments-path")
        normalization_field = load_structured_simulation_snapshot(
            args.ion_moments_path,
            sim_box_rho_p=sim_box_rho_p,
            vector_variables=None,
            scalar_variable=args.simulation_density_var,
            packed_schema="moments",
        )
    secondary_timeseries_field = None
    if args.secondary_timeseries_vector_vars:
        secondary_timeseries_vars = _parse_csv(args.secondary_timeseries_vector_vars)
        if geometry_vector_vars is not None and secondary_timeseries_vars == geometry_vector_vars:
            secondary_timeseries_field = geometry_field
        else:
            secondary_path = args.simulation_path
            secondary_schema = "field" if str(args.simulation_path).lower().endswith(".bp") else None
            if _looks_like_moment_variables(secondary_timeseries_vars):
                if not args.ion_moments_path:
                    raise ValueError(
                        "Moment-like secondary time-series variables require --ion-moments-path."
                    )
                secondary_path = args.ion_moments_path
                secondary_schema = "moments"
            secondary_timeseries_field = load_structured_simulation_snapshot(
                secondary_path,
                sim_box_rho_p=sim_box_rho_p,
                vector_variables=secondary_timeseries_vars,
                packed_schema=secondary_schema,
                background_b_lua_path=(
                    background_b_lua_path if secondary_schema == "field" else None
                ),
            )
    print("Loaded simulation field on a structured physical box", flush=True)

    _, trajectory_unwrapped_coords, trajectory_wrapped_coords, motion_metadata = (
        generate_moving_spacecraft_trajectory(
        initial_coords_rho_p,
        velocity_km_s=(args.vx_kms, args.vy_kms, args.vz_kms),
        rho_p_km=rho_p_km,
        sim_box_rho_p=sim_box_rho_p,
        dt_seconds=dt_seconds,
        n_steps=n_steps,
        )
    )
    print(
        f"Prepared motion metadata for {n_steps} timesteps; starting field sampling",
        flush=True,
    )

    spacecraft_colors = build_hs_color_map(formation.spacecraft_labels)
    sampling_result = stream_timeseries_to_csv(
        output_dir / "helioswarm_timeseries.csv",
        initial_coords_rho_p=initial_coords_rho_p,
        velocity_km_s=(args.vx_kms, args.vy_kms, args.vz_kms),
        rho_p_km=rho_p_km,
        sim_box_rho_p=sim_box_rho_p,
        dt_seconds=dt_seconds,
        n_steps=n_steps,
        field=field,
        normalization_field=normalization_field,
        spacecraft_labels=formation.spacecraft_labels,
        sampling_method=sampling_method,
        progress_callback=lambda *, step, n_steps, time_seconds: print(
            f"Sampling timestep {step + 1}/{n_steps} at t={time_seconds:.3f} s",
            flush=True,
        ),
    )
    secondary_sampling_result = None
    if secondary_timeseries_field is not None:
        secondary_sampling_result = sample_timeseries_from_trajectory(
            secondary_timeseries_field,
            time_seconds=sampling_result["times"],
            wrapped_coords=trajectory_wrapped_coords,
            spacecraft_labels=formation.spacecraft_labels,
            sampling_method=sampling_method,
        )
    elsasser_result = None
    if (
        args.simulation_density_var
        and secondary_sampling_result is not None
        and _secondary_series_is_magnetic(args)
    ):
        elsasser_result = _compute_elsasser_series(
            velocity_result=sampling_result,
            magnetic_result=secondary_sampling_result,
        )
    elsasser_pairwise_result = None
    elsasser_pair_npz_path = None
    elsasser_pair_json_path = None
    if elsasser_result is not None:
        elsasser_pairwise_result = _compute_elsasser_pairwise_increments(
            elsasser_result=elsasser_result,
            times=sampling_result["times"],
            unwrapped_positions=trajectory_unwrapped_coords,
            spacecraft_labels=formation.spacecraft_labels,
        )
        elsasser_pair_npz_path, elsasser_pair_json_path = _write_elsasser_pair_outputs(
            output_dir=output_dir,
            pairwise_result=elsasser_pairwise_result,
        )

    metadata = {
        "run_type": "helioswarm_timeseries",
        "sampling_mode": sampling_method,
        "interpolation_boundary": "periodic",
        "input": {
            "simulation_path": args.simulation_path,
            "ion_moments_path": args.ion_moments_path,
            "background_b_lua_path": args.background_b_lua_path,
            "hs_path": args.hs_path,
            "hs_time": args.hs_time,
        },
        "helioswarm": {
            **formation.metadata,
            "spacecraft_labels": formation.spacecraft_labels,
            "spacecraft_colors": spacecraft_colors,
            "color_sequence": HS_COLORS[: len(formation.spacecraft_labels)],
            "raw_positions_km": formation.raw_positions_km.tolist(),
            "relative_positions_km": formation.relative_positions_km.tolist(),
            "initial_coords_rho_p": initial_coords_rho_p.tolist(),
            "transform": transform,
        },
        "motion": {
            **motion_metadata,
            "initial_unwrapped_coords_rho_p": initial_coords_rho_p.tolist(),
            "final_unwrapped_coords_rho_p": sampling_result["final_unwrapped_coords"],
            "final_wrapped_coords_rho_p": sampling_result["final_wrapped_coords"],
        },
        "field": dict(field.metadata or {}),
        "geometry_field": dict(geometry_field.metadata or {}),
        "secondary_timeseries_field": (
            None
            if secondary_timeseries_field is None
            else dict(secondary_timeseries_field.metadata or {})
        ),
        "derived_quantity": (
            {
                "type": "velocity_from_momentum_over_density",
                "momentum_variables": _parse_csv(args.simulation_vector_vars),
                "density_variable": args.simulation_density_var,
                "component_labels": list(_resolve_component_labels(args)),
                "timeseries_plot_includes_density": True,
            }
            if args.simulation_density_var
            else None
        ),
        "elsasser": {
            "available": elsasser_result is not None,
            "formula": "u +/- B/sqrt(rho)",
            "density_variable": args.simulation_density_var if elsasser_result is not None else None,
            "magnetic_source_variables": (
                None
                if elsasser_result is None
                else _parse_csv(args.secondary_timeseries_vector_vars)
            ),
            "magnetic_source_metadata": (
                None
                if elsasser_result is None or secondary_timeseries_field is None
                else dict(secondary_timeseries_field.metadata or {})
            ),
            "component_labels": (
                None
                if elsasser_result is None
                else {
                    "zplus": list(_resolve_elsasser_component_labels("+")),
                    "zminus": list(_resolve_elsasser_component_labels("-")),
                }
            ),
        },
        "elsasser_pairs": {
            "available": elsasser_pairwise_result is not None,
            "formula": "delta z^\u00b1 = z^\u00b1_j - z^\u00b1_i",
            "pair_convention": "unordered_pairs_with_i_lt_j_stored_as_j_minus_i",
            "separation_geometry": "unwrapped_physical_positions",
            "density_variable": args.simulation_density_var if elsasser_pairwise_result is not None else None,
            "n_spacecraft": len(formation.spacecraft_labels),
            "n_pairs": (
                None
                if elsasser_pairwise_result is None
                else int(len(elsasser_pairwise_result["pair_labels"]))
            ),
            "pair_indices_shape": (
                None
                if elsasser_pairwise_result is None
                else list(np.asarray(elsasser_pairwise_result["pair_indices"]).shape)
            ),
            "pair_labels": (
                None
                if elsasser_pairwise_result is None
                else np.asarray(elsasser_pairwise_result["pair_labels"], dtype=str).tolist()
            ),
            "times_shape": (
                None
                if elsasser_pairwise_result is None
                else list(np.asarray(elsasser_pairwise_result["times"]).shape)
            ),
            "separation_vectors_shape": (
                None
                if elsasser_pairwise_result is None
                else list(np.asarray(elsasser_pairwise_result["separation_vectors"]).shape)
            ),
            "delta_zplus_shape": (
                None
                if elsasser_pairwise_result is None
                else list(np.asarray(elsasser_pairwise_result["delta_zplus"]).shape)
            ),
            "delta_zminus_shape": (
                None
                if elsasser_pairwise_result is None
                else list(np.asarray(elsasser_pairwise_result["delta_zminus"]).shape)
            ),
            "npz_path": elsasser_pair_npz_path,
            "json_path": elsasser_pair_json_path,
        },
        "timeseries_plot": {
            "mode": (
                "velocity_plus_density"
                if args.simulation_density_var
                else "vector_only"
            ),
            "component_labels": list(_resolve_component_labels(args)),
            "density_label": _resolve_density_label(args),
            "secondary_component_labels": (
                None
                if secondary_timeseries_field is None
                else list(_resolve_secondary_component_labels(args))
            ),
            "elsasser_available": elsasser_result is not None,
        },
        "output": {
            "csv_path": str(output_dir / "helioswarm_timeseries.csv"),
            "metadata_path": str(output_dir / "helioswarm_timeseries_metadata.json"),
            "geometry_plot_path": str(output_dir / "helioswarm_timeseries_geometry.png"),
            "plot_path": (
                str(output_dir / "helioswarm_timeseries.png")
                if args.plot_timeseries
                else None
            ),
            "secondary_plot_path": (
                str(output_dir / "helioswarm_timeseries_secondary.png")
                if args.plot_timeseries and secondary_timeseries_field is not None
                else None
            ),
            "zplus_plot_path": (
                str(output_dir / "helioswarm_timeseries_zplus.png")
                if args.plot_timeseries and elsasser_result is not None
                else None
            ),
            "zminus_plot_path": (
                str(output_dir / "helioswarm_timeseries_zminus.png")
                if args.plot_timeseries and elsasser_result is not None
                else None
            ),
            "elsasser_pairs_npz_path": elsasser_pair_npz_path,
            "elsasser_pairs_json_path": elsasser_pair_json_path,
        },
    }

    print("Rendering HelioSwarm geometry plot", flush=True)
    geometry_fig, _ = plot_hs_timeseries_geometry(
        geometry_field,
        spacecraft_labels=formation.spacecraft_labels,
        spacecraft_colors=spacecraft_colors,
        hub_relative_positions_km=formation.relative_positions_km,
        initial_coords_rho_p=initial_coords_rho_p,
        trajectory_coords_rho_p=trajectory_wrapped_coords,
        title="HelioSwarm Geometry and Simulation Slice",
    )
    geometry_fig.savefig(output_dir / "helioswarm_timeseries_geometry.png", dpi=150)
    print(
        f"Saved HelioSwarm geometry plot to "
        f"{output_dir / 'helioswarm_timeseries_geometry.png'}",
        flush=True,
    )

    write_timeseries_metadata(metadata, output_dir / "helioswarm_timeseries_metadata.json")
    print("Wrote CSV and metadata outputs", flush=True)

    if args.plot_timeseries:
        print("Rendering time-series plot", flush=True)
        fig, _ = plot_hs_timeseries_components(
            sampling_result["times"],
            spacecraft_labels=formation.spacecraft_labels,
            spacecraft_colors=spacecraft_colors,
            bx=sampling_result["bx"],
            by=sampling_result["by"],
            bz=sampling_result["bz"],
            component_labels=_resolve_component_labels(args),
            density=sampling_result["density"],
            density_label=_resolve_density_label(args) or r"$n$",
            title="HelioSwarm Moving Observatory Time Series",
        )
        fig.savefig(output_dir / "helioswarm_timeseries.png", dpi=150)
        print(f"Saved time-series plot to {output_dir / 'helioswarm_timeseries.png'}", flush=True)
        if secondary_sampling_result is not None:
            print("Rendering secondary time-series plot", flush=True)
            secondary_fig, _ = plot_hs_timeseries_components(
                secondary_sampling_result["times"],
                spacecraft_labels=formation.spacecraft_labels,
                spacecraft_colors=spacecraft_colors,
                bx=secondary_sampling_result["bx"],
                by=secondary_sampling_result["by"],
                bz=secondary_sampling_result["bz"],
                component_labels=_resolve_secondary_component_labels(args),
                title="HelioSwarm Secondary Time Series",
            )
            secondary_fig.savefig(output_dir / "helioswarm_timeseries_secondary.png", dpi=150)
            print(
                f"Saved secondary time-series plot to "
                f"{output_dir / 'helioswarm_timeseries_secondary.png'}",
                flush=True,
            )
        if elsasser_result is not None:
            print("Rendering Elsasser z+ time-series plot", flush=True)
            zplus_fig, _ = plot_hs_timeseries_components(
                sampling_result["times"],
                spacecraft_labels=formation.spacecraft_labels,
                spacecraft_colors=spacecraft_colors,
                bx=elsasser_result["zplus"]["bx"],
                by=elsasser_result["zplus"]["by"],
                bz=elsasser_result["zplus"]["bz"],
                component_labels=_resolve_elsasser_component_labels("+"),
                title="HelioSwarm Elsasser z+ Time Series",
            )
            zplus_fig.savefig(output_dir / "helioswarm_timeseries_zplus.png", dpi=150)
            print(
                f"Saved Elsasser z+ time-series plot to "
                f"{output_dir / 'helioswarm_timeseries_zplus.png'}",
                flush=True,
            )
            print("Rendering Elsasser z- time-series plot", flush=True)
            zminus_fig, _ = plot_hs_timeseries_components(
                sampling_result["times"],
                spacecraft_labels=formation.spacecraft_labels,
                spacecraft_colors=spacecraft_colors,
                bx=elsasser_result["zminus"]["bx"],
                by=elsasser_result["zminus"]["by"],
                bz=elsasser_result["zminus"]["bz"],
                component_labels=_resolve_elsasser_component_labels("-"),
                title="HelioSwarm Elsasser z- Time Series",
            )
            zminus_fig.savefig(output_dir / "helioswarm_timeseries_zminus.png", dpi=150)
            print(
                f"Saved Elsasser z- time-series plot to "
                f"{output_dir / 'helioswarm_timeseries_zminus.png'}",
                flush=True,
            )

    return {
        "record_count": sampling_result["row_count"],
        "metadata": metadata,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample a static simulation snapshot along a moving HelioSwarm formation."
    )
    parser.add_argument("--simulation-path", required=True)
    parser.add_argument(
        "--ion-moments-path",
        default=None,
        help="Optional ion moments .bp file used for n and nu quantities.",
    )
    parser.add_argument(
        "--background-b-lua-path",
        default=None,
        help="Optional Lua config used to compute and subtract the uniform guide field from B_z.",
    )
    parser.add_argument(
        "--simulation-vector-vars",
        default="bx,by,bz",
        help="Comma-separated vector component names to read from .bp snapshots.",
    )
    parser.add_argument(
        "--simulation-density-var",
        default=None,
        help="Optional scalar density variable used to normalize the sampled vector field.",
    )
    parser.add_argument(
        "--geometry-vector-vars",
        default=None,
        help="Optional comma-separated vector component names used only for the geometry plot background.",
    )
    parser.add_argument(
        "--simulation-component-labels",
        default=None,
        help="Optional comma-separated plot labels for the sampled vector components.",
    )
    parser.add_argument(
        "--secondary-timeseries-vector-vars",
        default=None,
        help="Optional comma-separated vector component names to sample into a second time-series figure.",
    )
    parser.add_argument(
        "--secondary-timeseries-component-labels",
        default=None,
        help="Optional comma-separated plot labels for the secondary time-series vector components.",
    )
    parser.add_argument("--hs-path", required=True)
    parser.add_argument("--hs-time", required=True)
    parser.add_argument("--rho-p-km", type=float, required=True)
    parser.add_argument("--sim-box-x", type=float, required=True)
    parser.add_argument("--sim-box-y", type=float, required=True)
    parser.add_argument("--sim-box-z", type=float, required=True)
    parser.add_argument("--vx-kms", type=float, required=True)
    parser.add_argument("--vy-kms", type=float, required=True)
    parser.add_argument("--vz-kms", type=float, required=True)
    parser.add_argument("--dt-seconds", type=float, required=True)
    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument("--sampling-method", choices=["nearest", "trilinear"], default="trilinear")
    parser.add_argument("--plot-timeseries", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    result = run_hs_timeseries(args)

    print(
        f"Saved {result['record_count']} time-series samples to "
        f"{result['metadata']['output']['csv_path']}"
    )


if __name__ == "__main__":
    main()
