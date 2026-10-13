from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_loop_workspace import _loop_plan
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute import chained_loop
from helion._compiler.cute import chained_matmul as chain
from helion.language import _tracing_ops
from helion.language import memory_ops


def fixture(size, dtype):
    graph = Graph()
    state = _input(graph, "state", (32, 32), torch.float32)
    vector = _input(graph, "vector", (size,), dtype)
    scalar = _input(graph, "scalar", (), torch.float32)
    left = _call(graph, _tracing_ops._host_tensor, ("left",), (32, 32), torch.bfloat16)
    right = _call(
        graph, _tracing_ops._host_tensor, ("right",), (32, 32), torch.bfloat16
    )
    result = _dot(graph, left, right, state)
    sink = _call(graph, _tracing_ops._host_tensor, ("sink",), (32, 32), torch.float32)
    graph.call_function(memory_ops.store, (sink, [None, None], state))
    plan = replace(_loop_plan(graph, (state,), (result,)), strategy="warp")
    return plan, vector, scalar


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("size", [32, 64, 129])
def test_vector_capture_is_typed_boundary_and_storage_is_separate(size, dtype):
    plan, vector, scalar = fixture(size, dtype)
    assert plan.loop is not None
    assert plan.loop.captures() == {scalar: "chain_loop_capture_2"}
    assert plan.loop.vector_captures() == {vector: "chain_loop_vector_1"}
    assert plan.loop.boundaries()[vector] == "chain_loop_vector_1"
    with patch.object(chain, "_shape", lambda node: tuple(node.meta["val"].shape)):
        assert (
            chained_loop.capture_shared_bytes(plan.loop)
            == (size * torch.tensor([], dtype=dtype).element_size() + 127) // 128 * 128
        )
        assert chained_loop.carry_shared_bytes(plan.loop) == 32 * 32 * 4


@pytest.mark.parametrize("size", [64, 129])
def test_original_vector_expression_fills_once_before_existing_join(size):
    plan, vector, _ = fixture(size, torch.float32)
    assert plan.loop is not None
    seen = []

    class Expression:
        def __init__(self, *args):
            self.coordinate_names = set()
            self.lines = []

        def value(self, node, coordinates):
            if node.meta["val"].ndim == 1:
                assert len(coordinates) == 1
                seen.append(node)
                return f"original_value({coordinates[0]})"
            return "3.0"

        def scalar(self, node):
            return str(node)

    with (
        patch.object(chain, "_shape", lambda node: tuple(node.meta["val"].shape)),
        patch.object(chain, "_Expression", Expression),
        patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(
                backend=SimpleNamespace(dtype_str=lambda dtype: "cutlass.Float32")
            ),
        ),
    ):
        lines = chained_loop.initialize_loop(
            cast("Any", object()),
            plan,
            cast("Any", object()),
            resident_carries=frozenset((0,)),
        )
    assert seen == [plan.loop.inputs[1]]
    assert lines[-1] == "cute.arch.sync_threads()"
    assert not any("chain_loop_carry_0" in line for line in lines)
    visits = []
    buffer = [None] * size
    barriers = []
    cute = SimpleNamespace(
        arch=SimpleNamespace(
            alloc_smem=lambda *args, **kwargs: buffer,
            sync_threads=lambda: barriers.append(True),
        ),
        make_tensor=lambda storage, layout: storage,
        make_layout=lambda shape: shape,
    )

    def value(index):
        visits.append(index)
        return index * 0.125 - 7

    for thread in range(plan.threads):
        namespace = {
            "chain_thread": thread,
            "cute": cute,
            "cutlass": SimpleNamespace(Float32=float, range_constexpr=range),
            "original_value": value,
        }
        exec(
            compile("\n".join(lines), "<bounded-original-capture-fill>", "exec"),
            namespace,
        )
    assert sorted(visits) == list(range(size))
    assert buffer == [i * 0.125 - 7 for i in range(size)]
    assert len(barriers) == plan.threads
    assert plan.loop.boundaries()[vector] == "chain_loop_vector_1"
