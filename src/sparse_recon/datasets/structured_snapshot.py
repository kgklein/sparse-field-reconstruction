from __future__ import annotations

import ast
from pathlib import Path

import numpy as np


DEFAULT_VECTOR_VARIABLES = ("bx", "by", "bz")
PACKED_CART_GRID_FIELD_COMPONENTS = {
    "ex": 0,
    "ey": 1,
    "ez": 2,
    "bx": 3,
    "by": 4,
    "bz": 5,
    "pote": 6,
    "potb": 7,
}
PACKED_CART_GRID_MOMENT_COMPONENTS = {
    "n": 0,
    "nux": 1,
    "nuy": 2,
    "nuz": 3,
    "p": 4,
}

PACKED_CART_GRID_SCHEMAS = {
    "field": PACKED_CART_GRID_FIELD_COMPONENTS,
    "moments": PACKED_CART_GRID_MOMENT_COMPONENTS,
}

_LUA_BACKGROUND_REQUIRED_SYMBOLS = ("mu0", "n0", "elcMass", "vAe", "B0")


def _build_axes(
    grid_shape: tuple[int, int, int],
    *,
    sim_box_rho_p: tuple[float, float, float] | None,
) -> tuple[dict[str, np.ndarray], str, list[float] | None]:
    if sim_box_rho_p is None:
        lengths = (1.0, 1.0, 1.0)
        grid_convention = "uniform_unit_box"
        box_metadata = None
    else:
        box = np.asarray(sim_box_rho_p, dtype=float)
        if box.shape != (3,) or np.any(box <= 0.0):
            raise ValueError(
                "sim_box_rho_p must contain three positive axis lengths; "
                f"got {sim_box_rho_p}"
            )
        lengths = tuple(float(value) for value in box)
        grid_convention = "uniform_box_rho_p"
        box_metadata = list(lengths)

    axes = {
        axis_name: np.linspace(0.0, axis_length, axis_count)
        for axis_name, axis_length, axis_count in zip(("x", "y", "z"), lengths, grid_shape)
    }
    return axes, grid_convention, box_metadata


def _normalize_vector_variables(
    vector_variables: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if vector_variables is None:
        return DEFAULT_VECTOR_VARIABLES

    normalized = tuple(str(name).strip() for name in vector_variables if str(name).strip())
    if not normalized:
        raise ValueError("vector_variables must contain at least one non-empty variable name")
    if len(normalized) != 3:
        raise ValueError(
            "vector_variables must contain exactly three component names, "
            f"got {normalized}"
        )
    return normalized


def _normalize_variable_selection(
    *,
    vector_variables: tuple[str, ...] | list[str] | None,
    scalar_variable: str | None,
) -> tuple[tuple[str, ...] | None, str | None]:
    if scalar_variable is not None:
        scalar_name = str(scalar_variable).strip()
        if not scalar_name:
            raise ValueError("scalar_variable must be a non-empty string when provided")
        if vector_variables is not None:
            raise ValueError("Provide either vector_variables or scalar_variable, not both")
        return None, scalar_name
    return _normalize_vector_variables(vector_variables), None


def _normalize_packed_schema(packed_schema: str | None) -> str | None:
    if packed_schema is None:
        return None
    normalized = str(packed_schema).strip().lower()
    if normalized not in PACKED_CART_GRID_SCHEMAS:
        raise ValueError(
            f"Unsupported packed_schema '{packed_schema}'. "
            f"Expected one of {sorted(PACKED_CART_GRID_SCHEMAS)}"
        )
    return normalized


def _parse_lua_assignments(path: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("--", 1)[0].strip()
        if not line or "=" not in line:
            continue
        name, expression = line.split("=", 1)
        normalized_name = name.strip()
        normalized_expression = expression.strip()
        if not normalized_name or not normalized_expression:
            continue
        assignments[normalized_name] = normalized_expression
    return assignments


def _evaluate_lua_expression(
    expression: str,
    *,
    assignments: dict[str, str],
    cache: dict[str, float],
    stack: set[str],
) -> float:
    expression = expression.strip()
    parsed = ast.parse(expression, mode="eval")

    def _evaluate_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _evaluate_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            name = node.id
            if name in cache:
                return cache[name]
            if name in stack:
                raise ValueError(f"Encountered circular Lua assignment reference for '{name}'")
            if name not in assignments:
                raise ValueError(f"Lua expression references unknown symbol '{name}'")
            stack.add(name)
            value = _evaluate_lua_expression(
                assignments[name],
                assignments=assignments,
                cache=cache,
                stack=stack,
            )
            stack.remove(name)
            cache[name] = value
            return value
        if isinstance(node, ast.UnaryOp):
            operand = _evaluate_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError(f"Unsupported Lua unary operator in '{expression}'")
        if isinstance(node, ast.BinOp):
            left = _evaluate_node(node.left)
            right = _evaluate_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise ValueError(f"Unsupported Lua binary operator in '{expression}'")
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "math"
                and node.func.attr == "sqrt"
                and len(node.args) == 1
            ):
                return float(np.sqrt(_evaluate_node(node.args[0])))
            raise ValueError(f"Unsupported Lua function call in '{expression}'")
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "math" and node.attr == "pi":
                return float(np.pi)
            raise ValueError(f"Unsupported Lua attribute access in '{expression}'")
        raise ValueError(f"Unsupported Lua expression syntax in '{expression}'")

    return float(_evaluate_node(parsed))


def _load_background_b0_from_lua(path: Path) -> float:
    assignments = _parse_lua_assignments(path)
    missing = [symbol for symbol in _LUA_BACKGROUND_REQUIRED_SYMBOLS if symbol not in assignments]
    if missing:
        raise ValueError(
            f"Lua file {path} is missing required background-field symbols: {missing}"
        )
    cache: dict[str, float] = {}
    for symbol in _LUA_BACKGROUND_REQUIRED_SYMBOLS:
        cache[symbol] = _evaluate_lua_expression(
            assignments[symbol],
            assignments=assignments,
            cache=cache,
            stack={symbol},
        )
    return float(cache["B0"])


def _apply_magnetic_background_subtraction(
    values: np.ndarray,
    *,
    read_variables: list[str],
    packed_schema: str | None,
    background_b_lua_path: Path | None,
) -> tuple[np.ndarray, dict | None]:
    if background_b_lua_path is None:
        return values, None
    if packed_schema not in (None, "field"):
        return values, None

    normalized_names = [name.lower() for name in read_variables]
    if "bz" not in normalized_names:
        return values, None

    b0 = _load_background_b0_from_lua(background_b_lua_path)
    adjusted_values = np.array(values, copy=True)
    adjusted_values[..., normalized_names.index("bz")] -= b0
    metadata = {
        "background_b_lua_path": str(background_b_lua_path),
        "background_b0": float(b0),
        "subtracted_component": "bz",
        "subtracted_direction": "+z",
        "bz_represents_fluctuation": True,
    }
    return adjusted_values, metadata


def _load_npy_snapshot(
    path: Path,
    *,
    sim_box_rho_p: tuple[float, float, float] | None,
    vector_variables: tuple[str, ...] | None,
    scalar_variable: str | None,
    packed_schema: str | None,
    background_b_lua_path: Path | None,
) -> tuple[np.ndarray, dict[str, np.ndarray], tuple[int, int, int], dict]:
    if packed_schema is not None:
        raise ValueError(".npy simulation snapshots do not support packed_schema selection")
    if scalar_variable is not None:
        raise ValueError(
            ".npy simulation snapshots only support the default vector field layout; "
            "scalar variable selection is only available for .bp inputs"
        )
    if vector_variables is not None and tuple(vector_variables) != DEFAULT_VECTOR_VARIABLES:
        raise ValueError(
            ".npy simulation snapshots only support the default vector variable set "
            "'bx,by,bz'"
        )

    field = np.load(path, allow_pickle=False)
    if field.ndim != 4:
        raise ValueError(
            "Simulation snapshot must be a 4D array shaped (nx, ny, nz, 3); "
            f"got shape {field.shape}"
        )
    if field.shape[-1] != 3:
        raise ValueError(
            "Simulation snapshot last dimension must have size 3 for vector components; "
            f"got shape {field.shape}"
        )

    read_variables = list(DEFAULT_VECTOR_VARIABLES)
    field, preprocessing_metadata = _apply_magnetic_background_subtraction(
        field,
        read_variables=read_variables,
        packed_schema=None,
        background_b_lua_path=background_b_lua_path,
    )

    grid_shape = tuple(int(value) for value in field.shape[:3])
    axes, grid_convention, sim_box_metadata = _build_axes(
        grid_shape,
        sim_box_rho_p=sim_box_rho_p,
    )
    metadata = {
        "source": "simulation",
        "field_kind": "simulation_snapshot",
        "source_format": "npy",
        "file_path": str(path),
        "array_shape": list(field.shape),
        "dtype": str(field.dtype),
        "grid_convention": grid_convention,
        "sim_box_rho_p": sim_box_metadata,
        "available_variables": list(DEFAULT_VECTOR_VARIABLES),
        "read_variables": read_variables,
        "preprocessing": preprocessing_metadata,
    }
    return field, axes, grid_shape, metadata


def _load_bp_snapshot(
    path: Path,
    *,
    sim_box_rho_p: tuple[float, float, float] | None,
    vector_variables: tuple[str, ...] | None,
    scalar_variable: str | None,
    packed_schema: str | None,
    background_b_lua_path: Path | None,
) -> tuple[np.ndarray, dict[str, np.ndarray], tuple[int, int, int], dict]:
    try:
        import adios2
    except ImportError as exc:
        raise ImportError(
            "Reading .bp simulation snapshots requires the 'adios2' package to be installed."
        ) from exc

    packed_mapping = (
        None if packed_schema is None else PACKED_CART_GRID_SCHEMAS[packed_schema]
    )
    packed_component_names: list[str] = []
    with adios2.Stream(str(path), "r") as stream:
        available = stream.available_variables()
        available_names = sorted(str(name) for name in available.keys())
        if "CartGridField" in available:
            packed_component_names = [
                name
                for name, index in sorted(
                    (packed_mapping or {}).items(),
                    key=lambda item: item[1],
                )
            ]

        if scalar_variable is not None:
            requested_names = [scalar_variable]
        else:
            requested_names = list(vector_variables or DEFAULT_VECTOR_VARIABLES)

        available_lookup = {name.lower(): name for name in available_names}
        requested_lower = [name.lower() for name in requested_names]
        direct_names = [available_lookup[name] for name in requested_lower if name in available_lookup]
        packed_names = [
            name for name in requested_lower
            if packed_mapping is not None and name in packed_mapping and "CartGridField" in available
        ]
        missing = [
            original for original, lowered in zip(requested_names, requested_lower)
            if lowered not in available_lookup and lowered not in packed_names
        ]
        if missing:
            advertised_names = sorted(set(available_names + packed_component_names))
            raise ValueError(
                f"Requested .bp variables {missing} were not found in {path}. "
                f"Available variables: {advertised_names}"
            )

        cart_grid_field = None
        if packed_names:
            cart_grid_field = np.asarray(stream.read("CartGridField"))
            if cart_grid_field.ndim != 4:
                raise ValueError(
                    "Packed CartGridField variable must be 4D with trailing component axis; "
                    f"got shape {cart_grid_field.shape}"
                )

        arrays = []
        for requested_name, lowered_name in zip(requested_names, requested_lower):
            if lowered_name in available_lookup:
                arrays.append(np.asarray(stream.read(available_lookup[lowered_name])))
            else:
                component_index = packed_mapping[lowered_name]
                if component_index >= cart_grid_field.shape[-1]:
                    raise ValueError(
                        f"Packed CartGridField in {path} does not contain component "
                        f"'{requested_name}' at index {component_index}"
                    )
                arrays.append(np.asarray(cart_grid_field[..., component_index]))

    first_shape = arrays[0].shape
    if len(first_shape) != 3:
        raise ValueError(
            "Only structured 3D .bp variables are currently supported; "
            f"got shape {first_shape} for variable '{requested_names[0]}'"
        )
    for name, array in zip(requested_names[1:], arrays[1:]):
        if array.shape != first_shape:
            raise ValueError(
                "All requested .bp variables must share the same 3D shape; "
                f"got {array.shape} for '{name}' and {first_shape} for '{requested_names[0]}'"
            )

    values = np.stack(arrays, axis=-1)
    values, preprocessing_metadata = _apply_magnetic_background_subtraction(
        values,
        read_variables=requested_names,
        packed_schema=packed_schema,
        background_b_lua_path=background_b_lua_path,
    )
    grid_shape = tuple(int(value) for value in first_shape)
    axes, grid_convention, sim_box_metadata = _build_axes(
        grid_shape,
        sim_box_rho_p=sim_box_rho_p,
    )
    metadata = {
        "source": "simulation",
        "field_kind": "simulation_snapshot",
        "source_format": "bp",
        "file_path": str(path),
        "array_shape": list(values.shape),
        "dtype": str(values.dtype),
        "grid_convention": grid_convention,
        "sim_box_rho_p": sim_box_metadata,
        "available_variables": sorted(set(available_names + packed_component_names)),
        "read_variables": requested_names,
        "packed_source_variable": "CartGridField" if packed_component_names else None,
        "packed_schema_kind": packed_schema if packed_component_names else None,
        "preprocessing": preprocessing_metadata,
    }
    return values, axes, grid_shape, metadata


def load_structured_snapshot_data(
    path: str | Path,
    *,
    sim_box_rho_p: tuple[float, float, float] | None = None,
    vector_variables: tuple[str, ...] | list[str] | None = None,
    scalar_variable: str | None = None,
    packed_schema: str | None = None,
    background_b_lua_path: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], tuple[int, int, int], dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place the file locally or provide a valid simulation snapshot path."
        )

    normalized_vector_variables, normalized_scalar_variable = _normalize_variable_selection(
        vector_variables=vector_variables,
        scalar_variable=scalar_variable,
    )
    normalized_packed_schema = _normalize_packed_schema(packed_schema)
    normalized_background_path = (
        None if background_b_lua_path is None else Path(background_b_lua_path)
    )
    if normalized_background_path is not None and not normalized_background_path.exists():
        raise FileNotFoundError(
            f"Lua background-field config not found: {normalized_background_path}"
        )

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _load_npy_snapshot(
            path,
            sim_box_rho_p=sim_box_rho_p,
            vector_variables=normalized_vector_variables,
            scalar_variable=normalized_scalar_variable,
            packed_schema=normalized_packed_schema,
            background_b_lua_path=normalized_background_path,
        )
    if suffix == ".bp":
        return _load_bp_snapshot(
            path,
            sim_box_rho_p=sim_box_rho_p,
            vector_variables=normalized_vector_variables,
            scalar_variable=normalized_scalar_variable,
            packed_schema=normalized_packed_schema,
            background_b_lua_path=normalized_background_path,
        )

    raise ValueError(
        f"Unsupported simulation snapshot format: {suffix}. Expected .npy or .bp"
    )
