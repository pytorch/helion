from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

import helion
from helion import Config
from helion._compiler.device_function import DeviceFunction
from helion._compiler.pallas.codegen import _loop_offset_alignment
from helion._compiler.pallas.memory_access import MemoryAccessKind
from helion._compiler.pallas.memory_access import build_memory_access
from helion._compiler.pallas.plan_tiling import ArbitrarySlicePattern
from helion._compiler.pallas.plan_tiling import IndexingPattern
from helion._compiler.pallas.plan_tiling import TensorIndexPattern
from helion._compiler.pallas.plan_tiling import TilePattern
from helion._compiler.pallas.tensorcore_plan import OneHotGatherPlan
from helion._compiler.pallas.tensorcore_plan import OneHotScatterPlan
from helion._compiler.pallas.tensorcore_plan import select_tensorcore_plan
from helion._compiler.pallas.tracing_ops import _annotate_provable_sublane_alignment
from helion._compiler.tile_strategy import LoopDimInfo
from helion._testing import DEVICE
from helion._testing import onlyBackends
import helion.language as hl
from helion.language import memory_ops
from helion.language.atomic_ops import atomic_add

pytestmark = pytest.mark.cpu_only

if TYPE_CHECKING:
    from helion._compiler.inductor_lowering import CodegenState


@helion.kernel(static_shapes=False, autotune_effort="none")
def _store_broadcast_tail(out, values, kind: hl.constexpr):
    axis = 1 if kind == "indexed" else 0
    for tile in hl.tile(out.size(axis)):
        if kind == "indexed":
            out[0, tile] = 3.0
        elif kind == "scalar":
            out[tile, :] = 3.0
        else:
            out[tile, :] = values[None, :]
    return out


@onlyBackends(["pallas"])
@pytest.mark.parametrize("kind", ["scalar", "vector", "indexed"])
def test_stores_broadcast_to_runtime_clamped_refs(kind):
    values = torch.arange(8, dtype=torch.float32, device=DEVICE)
    initial_shape = (1, 4) if kind == "indexed" else (4, 8)
    initial = torch.empty(initial_shape, device=DEVICE)
    bound = _store_broadcast_tail.bind((initial, values, kind))
    compiled = bound.compile_config(Config(block_sizes=[128]))
    # Reuse the same generated program across short refs, a singleton, and a
    # full tile. The full tile also respects Mosaic's memory tiling.
    for size in (4, 1, 128, 4):
        shape = (1, size) if kind == "indexed" else (size, 8)
        out = torch.full(shape, float("nan"), device=DEVICE)
        actual = compiled(out, values, kind)
        expected = (
            values.expand_as(out) if kind == "vector" else torch.full_like(out, 3.0)
        )
        assert actual is out
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@helion.kernel(static_shapes=False, autotune_effort="none")
def _dynamic_clamped_memory(x, out, operation: hl.constexpr):
    for tile in hl.tile(x.size(0)):
        value = hl.full([tile], 1.0, dtype=torch.float32)
        if operation == "load":
            out[tile] = x[tile] + value
        elif operation == "add":
            previous = hl.atomic_add(out, [tile], value)
            x[tile] = previous + value
        elif operation == "add_scalar":
            previous = hl.atomic_add(out, [tile], 1.0)
            x[tile] = previous + value
        elif operation == "cas_scalar":
            previous = hl.atomic_cas(out, [tile], 0.0, 1.0)
            x[tile] = previous + value
        else:
            expected = hl.full([tile], 0.0, dtype=torch.float32)
            previous = hl.atomic_cas(out, [tile], expected, value)
            x[tile] = previous + value
    return out


@onlyBackends(["pallas"])
@pytest.mark.parametrize(
    "operation", ["load", "add", "cas", "add_scalar", "cas_scalar"]
)
def test_dynamic_clamped_loads_and_atomics(operation):
    x = torch.arange(128, dtype=torch.float32, device=DEVICE)
    out = torch.zeros_like(x)
    compiled = _dynamic_clamped_memory.bind((x, out, operation)).compile_config(
        Config(block_sizes=[128])
    )
    # One generated program must handle full, partial, and singleton refs,
    # including a return to the partial shape after changing the launcher key.
    for size in (128, 65, 1, 65):
        x = torch.arange(size, dtype=torch.float32, device=DEVICE)
        out = x.clone()
        # Materialize an independent oracle before the output is donated.
        # A lazy TPU clone can still refer to the donated device buffer.
        before = out.cpu().clone()
        actual = compiled(x, out, operation)
        expected = (
            torch.where(before == 0, 1.0, before)
            if operation in ("cas", "cas_scalar")
            else before + 1
        )
        assert actual is out
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        if operation != "load":
            torch.testing.assert_close(x.cpu(), before + 1, rtol=0, atol=0)


def _placeholder(
    graph: torch.fx.Graph, name: str, value: torch.Tensor
) -> torch.fx.Node:
    node = graph.placeholder(name)
    node.meta["val"] = value
    return node


def test_memory_access_snapshots_semantic_patterns() -> None:
    graph = torch.fx.Graph()
    table = torch.empty(128, 32)
    index = torch.empty(16, dtype=torch.int32)
    table_node = _placeholder(graph, "table", table)
    index_node = _placeholder(graph, "index", index)
    fx_subscript = (index_node, slice(None))
    subscript: list[object] = list(fx_subscript)
    load = graph.call_function(memory_ops.load, (table_node, fx_subscript))
    load.meta["val"] = torch.empty(16, 32)
    patterns: list[IndexingPattern] = [
        TensorIndexPattern(),
        ArbitrarySlicePattern(slice(None)),
    ]

    access = build_memory_access(load, table, subscript, patterns)
    # Later tiling changes must not alter recorded memory semantics.
    patterns[0] = TilePattern(0)

    assert access.kind is MemoryAccessKind.LOAD
    assert access.tensor is table
    assert access.value_node is None
    assert isinstance(access.patterns[0], TensorIndexPattern)


def test_store_and_atomic_memory_accesses_record_values() -> None:
    graph = torch.fx.Graph()
    output = torch.empty(64, 32)
    value = torch.empty(16, 32)
    output_node = _placeholder(graph, "output", output)
    value_node = _placeholder(graph, "value", value)
    fx_subscript = (slice(None), slice(None))
    subscript: list[object] = list(fx_subscript)
    patterns: list[IndexingPattern] = [
        TilePattern(0),
        ArbitrarySlicePattern(slice(None)),
    ]

    store = graph.call_function(
        memory_ops.store, (output_node, fx_subscript, value_node)
    )
    atomic = graph.call_function(atomic_add, (output_node, fx_subscript, value_node))

    store_access = build_memory_access(store, output, subscript, patterns)
    atomic_access = build_memory_access(atomic, output, subscript, patterns)

    assert store_access.kind is MemoryAccessKind.STORE
    assert atomic_access.kind is MemoryAccessKind.ATOMIC
    assert store_access.value_node is value_node
    assert atomic_access.value_node is value_node


# We normally avoid testing compiler internals. However, here we make an
# exception because lying to Mosaic about alignment can lead to UB.
def test_loop_offset_uses_only_proven_window_alignment() -> None:
    block_id = 3
    # The loop start has no known alignment unless an aligned window is recorded.
    loop = SimpleNamespace(
        block_id_to_info={
            block_id: LoopDimInfo(begin_var_name="runtime_begin", begin_expr=None)
        }
    )

    def make_state(aligned_tiles: dict[int, int], block_size: int) -> CodegenState:
        # A fake device function, so the real predicate has to be bound onto it
        # below rather than inherited.
        device_function = SimpleNamespace(
            aligned_tiles=aligned_tiles,
            resolved_block_size=lambda _block_id: block_size,
        )
        device_function.proven_sublane_alignment = functools.partial(
            DeviceFunction.proven_sublane_alignment, device_function
        )
        return cast(
            "CodegenState",
            SimpleNamespace(
                device_function=device_function,
                codegen=SimpleNamespace(active_device_loops={block_id: [loop]}),
            ),
        )

    aligned_state = make_state({block_id: 16}, block_size=32)
    assert _loop_offset_alignment(block_id, aligned_state) == 16
    assert (
        _annotate_provable_sublane_alignment(aligned_state, block_id, "offset")
        == "pl.multiple_of(offset, 16)"
    )

    # A block size that is not a multiple of the window alignment proves
    # nothing: offsets 16, 40, 64, ... are not all multiples of 16.
    unstepped_state = make_state({block_id: 16}, block_size=24)
    assert _loop_offset_alignment(block_id, unstepped_state) is None
    assert (
        _annotate_provable_sublane_alignment(unstepped_state, block_id, "offset")
        == "offset"
    )

    # Without an aligned window, leave the offset unannotated.
    unaligned_state = make_state({}, block_size=32)
    assert _loop_offset_alignment(block_id, unaligned_state) is None
    assert (
        _annotate_provable_sublane_alignment(unaligned_state, block_id, "offset")
        == "offset"
    )


def test_tensorcore_plan_owns_indirect_fallbacks() -> None:
    graph = torch.fx.Graph()
    table = torch.empty(128, 32)
    index = torch.empty(16, dtype=torch.int32)
    value = torch.empty(16, 32)
    table_node = _placeholder(graph, "table", table)
    index_node = _placeholder(graph, "index", index)
    value_node = _placeholder(graph, "value", value)
    fx_subscript = (index_node, slice(None))
    subscript: list[object] = list(fx_subscript)
    patterns: list[IndexingPattern] = [
        TensorIndexPattern(),
        ArbitrarySlicePattern(slice(None)),
    ]
    load = graph.call_function(memory_ops.load, (table_node, fx_subscript))
    load.meta["val"] = value
    store = graph.call_function(
        memory_ops.store, (table_node, fx_subscript, value_node)
    )
    load_access = build_memory_access(load, table, subscript, patterns)
    store_access = build_memory_access(store, table, subscript, patterns)

    gather_fallback = object()
    scatter_fallback = object()
    with (
        patch(
            "helion._compiler.pallas.gather.build_gather_plan",
            return_value=gather_fallback,
        ),
        patch(
            "helion._compiler.pallas.gather.build_scatter_plan",
            return_value=scatter_fallback,
        ),
    ):
        gather = select_tensorcore_plan(load_access, Config())
        scatter = select_tensorcore_plan(store_access, Config())

    assert isinstance(gather, OneHotGatherPlan)
    assert isinstance(scatter, OneHotScatterPlan)
    assert gather.plan is gather_fallback
    assert scatter.plan is scatter_fallback
