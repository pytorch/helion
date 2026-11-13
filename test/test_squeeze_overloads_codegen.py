"""CPU coverage for squeeze overloads that must bypass Inductor views."""

from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._compiler.aten_lowering import squeeze_lowering
from helion._compiler.backend_registry import get_backend_class
from helion._testing import cpu_render_target
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.squeeze_overloads import squeeze_overloads
from helion.runtime.settings import _get_backend

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["triton", "tileir", "cute"])]


def test_squeeze_routing_keeps_existing_backend_scope() -> None:
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    # Both logical-view overloads share routing; squeeze.dim keeps its existing
    # ordinary lowering. Unopted-in backends must not acquire new lowering.
    for target, args, routed in (
        (torch.ops.aten.squeeze.default, (value,), True),
        (torch.ops.aten.squeeze.dims, (value, [0]), True),
        (torch.ops.aten.squeeze.dim, (value, 0), False),
    ):
        node = graph.call_function(target, args)
        for name in ("triton", "tileir", "cute", "metal", "pallas", "flydsl"):
            backend = get_backend_class(name)()
            lowering = backend.pre_inductor_lowering(node)
            expected = routed and name in ("triton", "tileir", "cute", "metal")
            assert lowering is (squeeze_lowering if expected else None), name


@pytest.mark.parametrize("mode", [0, 1, 2, 3])
def test_squeeze_overloads_use_traced_shape(mode: int) -> None:
    with cpu_render_target():
        bound = cpu_bind(squeeze_overloads, (torch.randn(7, 13), mode))
        code = bound.to_code(helion.Config(block_sizes=[4, 8]))
    calls = [node for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Call)]
    # Nontrivial overloads restore the tile shape; an identity reshape may
    # legitimately disappear from the mode-zero no-op control.
    if mode != 0 and _get_backend() in ("triton", "tileir"):
        assert any(
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tl"
            and node.func.attr == "reshape"
            and len(node.args) == 2
            and isinstance(node.args[1], ast.List)
            and [ast.unparse(axis) for axis in node.args[1].elts]
            == ["_BLOCK_SIZE_0", "_BLOCK_SIZE_1"]
            for node in calls
        )
