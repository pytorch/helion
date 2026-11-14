"""Helpers for checking tunable declarations and their selected values."""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import cast
from unittest.mock import patch

import torch

from .._compiler.inductor_lowering import APIFuncLowering
from ..autotuner.config_fragment import ConfigSpecFragment
from ..language.tunable_ops import _get_tunable
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .._compiler.aten_lowering import LoweringContext
    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel


class IdentityFragment(ConfigSpecFragment):
    def __init__(self, value: float | bool = 3) -> None:
        self.value = value

    def default(self) -> int | float | bool:
        return self.value


def host_functions(
    bound: BoundKernel[object], config: Config
) -> tuple[ast.FunctionDef, list[ast.FunctionDef]]:
    return _host_functions(ast.parse(bound.to_code(config)), bound.kernel.fn.__name__)


def _host_functions(
    module: ast.Module, name: str
) -> tuple[ast.FunctionDef, list[ast.FunctionDef]]:
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    host = next(node for node in functions if node.name == name)
    return host, functions


def host_assignment_values(
    bound: BoundKernel[object], config: Config, *names: str
) -> list[ast.expr]:
    """Read selected-value assignments from the emitted host function."""
    host, _ = host_functions(bound, config)
    return [
        node.value
        for node in ast.walk(host)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        )
    ]


def load_host_function(
    bound: BoundKernel[object],
    config: Config,
    launcher: Callable[..., None],
    *,
    scalar_values: bool = False,
) -> Callable[..., Any]:
    """Execute host code, optionally exposing boxed Pallas scalar values.

    Raw launch arguments are the default. Scalar observation decodes only
    parameters whose generated device preamble explicitly unboxes element zero;
    genuine scalar tensors and one-element tensor inputs remain tensors.
    """
    source = ast.parse(bound.to_code(config))
    host, functions = _host_functions(source, bound.kernel.fn.__name__)
    launcher_default = next(
        default
        for arg, default in zip(
            host.args.kwonlyargs, host.args.kw_defaults, strict=True
        )
        if arg.arg == "_launcher"
    )
    assert isinstance(launcher_default, ast.Name)
    namespace: dict[str, Any] = dict(bound.kernel.fn.__globals__)
    for node in source.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            exec(compile(ast.Module([node], []), "<host_imports>", "exec"), namespace)
    namespace.update(torch=torch, hl=hl)
    device_objects = {
        node.name: SimpleNamespace() for node in functions if node is not host
    }
    if scalar_values and bound.settings.backend == "pallas":
        scalar_positions = []
        for node in functions:
            if node is host:
                continue
            positions = {arg.arg: i for i, arg in enumerate(node.args.args)}
            unboxed = tuple(
                positions[stmt.targets[0].id]
                for stmt in node.body
                if isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id in positions
                and isinstance(stmt.value, ast.Subscript)
                and isinstance(stmt.value.value, ast.Name)
                and stmt.value.value.id == stmt.targets[0].id
                and isinstance(stmt.value.slice, ast.Constant)
                and stmt.value.slice.value == 0
            )
            scalar_positions.append((device_objects[node.name], unboxed))

        def observe_scalars(
            kernel: object, grid: object, *args: object, **kwargs: object
        ) -> None:
            (indices,) = [
                indices for device, indices in scalar_positions if device is kernel
            ]
            observed = list(args)
            for index in indices:
                value = observed[index]
                assert isinstance(value, torch.Tensor) and value.shape == (1,)
                observed[index] = value.item()
            launcher(kernel, grid, *observed, **kwargs)

        observed_launcher = observe_scalars
    else:
        observed_launcher = launcher
    namespace[launcher_default.id] = (
        pallas_host_observer(observed_launcher)
        if bound.settings.backend == "pallas"
        else observed_launcher
    )
    namespace.update(device_objects)
    future = ast.ImportFrom("__future__", [ast.alias("annotations")], 0)
    module = ast.fix_missing_locations(ast.Module([future, host], []))
    exec(compile(module, "<selected_host>", "exec"), namespace)
    return cast("Callable[..., Any]", namespace[host.name])


def pallas_host_observer(observer: Callable[..., None]) -> Callable[..., object]:
    """Observe a CPU host call using Pallas's returned-output contract."""

    def launch(kernel: object, grid: object, *args: object, **kwargs: object) -> object:
        inplace_indices = set(cast("Sequence[int]", kwargs["_inplace_indices"]))
        output_indices = [
            index
            for index in cast("Sequence[int]", kwargs["_output_indices"])
            if index not in inplace_indices
        ]
        launch_args = list(args)
        for index in output_indices:
            output = launch_args[index]
            assert isinstance(output, torch.Tensor)
            if output.is_meta:
                launch_args[index] = torch.empty_strided(
                    output.size(),
                    output.stride(),
                    dtype=output.dtype,
                    device="cpu",
                    requires_grad=output.requires_grad,
                )
        observer(kernel, grid, *launch_args, **kwargs)
        outputs = tuple(launch_args[index] for index in output_indices)
        if len(outputs) == 1:
            return outputs[0]
        return outputs or None

    return launch


def check_host_selected_value(
    bound: BoundKernel[object],
    args: Sequence[object],
    values: Sequence[int | float | bool],
) -> None:
    for value in values:
        config = bound.config_spec.default_config()
        config.config["value"] = value
        launches = []

        def observe_launch(
            kernel: object,
            grid: object,
            x: torch.Tensor,
            out: torch.Tensor,
            *selected: float | bool,
            records: list[tuple[int | float | bool, ...]] = launches,
            expected: float | bool = value,
            **kwargs: object,
        ) -> None:
            records.append(selected)
            assert selected
            assert all(
                type(item) is type(expected) and item == expected for item in selected
            )

        # Execute actual emitted host selection and argument publication,
        # replacing only the native launch. Both declarations select the same
        # typed value, whether the launcher combines or separates their inputs.
        load_host_function(bound, config, observe_launch)(*args)
        assert len(launches) == 1


def single_tunable_value(
    bound: BoundKernel[object], name: str
) -> torch.SymInt | torch.SymFloat | torch.SymBool:
    """Require one symbolic value despite repeated visits to a declaration."""
    (value,) = [
        value
        for (registered_name, _), value in bound.env.tunable_values.items()
        if registered_name == name
    ]
    return value


def render_tunable_config(
    bound: BoundKernel[object], name: str, value: float | bool
) -> str:
    """Render a config and check the real lowering emits its typed constant."""
    original = APIFuncLowering.codegen
    emitted: list[object] = []

    def observe(
        self: APIFuncLowering, ctx: LoweringContext, node: torch.fx.Node
    ) -> object:
        result = original(self, ctx, node)
        if node.target is _get_tunable and node.args == (name,):
            emitted.append(result)
        return result

    config = bound.config_spec.default_config()
    config.config[name] = value
    with patch.object(APIFuncLowering, "codegen", observe):
        source = bound.to_code(config)
    assert emitted
    assert all(
        isinstance(result, ast.Constant)
        and type(result.value) is type(value)
        and result.value == value
        for result in emitted
    )
    ast.parse(source)
    return source
