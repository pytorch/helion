from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _cpu_only() -> Generator[None, None, None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _underscore_tensor(_: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(_)
    for tile in hl.tile(_.numel()):
        out[tile] = _[tile] * 2
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _underscore_dimension(x: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
    _ = shape.size(0)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + _
    return out


@pytest.mark.parametrize("dimension", [7, 11])
@pytest.mark.parametrize("tensor_argument", [False, True])
def test_underscore_device_binding_preserves_host_argument(
    dimension: int, tensor_argument: bool
) -> None:
    x = torch.empty(19)
    inputs = (x,) if tensor_argument else (x, torch.empty(dimension))
    kernel = _underscore_tensor if tensor_argument else _underscore_dimension
    bound = _cpu_bind(kernel, inputs)
    source = bound.to_code(helion.Config(block_sizes=[32], num_threads=[32]))
    module = ast.parse(source)
    device = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    assert all(argument.arg != "_" for argument in device.args.args)
    assert not any(
        isinstance(node, ast.Name) and node.id == "_" for node in ast.walk(device)
    )

    # Execute the generated host wrapper with a capture launcher. The device
    # binding is renamed, but Python's original `_` still supplies its value.
    module.body.remove(device)
    marker = object()
    namespace: dict[str, object] = {device.name: marker}
    exec(compile(module, "<host-wrapper>", "exec"), namespace)
    captured: list[tuple[object, ...]] = []

    def capture(
        device_kernel: object, grid: object, *arguments: object, **kwargs: object
    ) -> None:
        assert device_kernel is marker
        captured.append(arguments)

    cast("Callable[..., object]", namespace[kernel.name])(*inputs, _launcher=capture)
    assert len(captured) == 1
    launch = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    )
    underscore_index = next(
        index
        for index, argument in enumerate(launch.args[2:])
        if isinstance(argument, ast.Name) and argument.id == "_"
    )
    value = captured[0][underscore_index]
    if tensor_argument:
        assert value is x
    else:
        assert value == dimension
