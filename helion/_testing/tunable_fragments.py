"""Helpers for checking tunable declarations and their selected values."""

from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import cast

import torch

import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..runtime.config import Config
    from ..runtime.kernel import BoundKernel


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
