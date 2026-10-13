from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph
from torch.fx import Node

from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_loop_workspace import _loop_plan
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute.chained_warp_loop_carry import plan_warp_loop_carry
from helion._compiler.cute.chained_warp_loop_prefetch import RawOperand
from helion._compiler.cute.chained_warp_loop_prefetch import WarpLoopPrefetchEmission
from helion._compiler.cute.chained_warp_loop_prefetch import _future_index
from helion._compiler.cute.chained_warp_loop_prefetch import pending_groups
from helion._compiler.cute.chained_warp_loop_prefetch import plan_warp_loop_prefetch
from helion.autotuner.config_spec import ConfigSpec
from helion.exc import InvalidConfig
from helion.language import _tracing_ops
from helion.language import memory_ops


def fixture(mode="plain"):
    graph = Graph()
    state = _input(graph, "state", (32, 32), torch.float32)
    raw = []
    for name in ("left", "right"):
        source = _call(
            graph, _tracing_ops._host_tensor, (name,), (32, 32), torch.bfloat16
        )
        raw.append(
            _call(
                graph, memory_ops.load, (source, [None, None]), (32, 32), torch.bfloat16
            )
        )
    left = _call(graph, torch.ops.aten.t.default, (raw[0],), (32, 32), torch.bfloat16)
    right = _convert(graph, raw[1], torch.float32)
    right = _call(
        graph, torch.ops.aten.mul.Scalar, (right, 0.75), (32, 32), torch.float32
    )
    right = _convert(graph, right, torch.bfloat16)
    sink = _call(graph, _tracing_ops._host_tensor, ("sink",), (32, 32), torch.bfloat16)
    if mode == "alias":
        sink.meta["val"] = cast("Node", raw[1].args[0]).meta["val"]
    if mode == "gather":
        right = _call(
            graph,
            torch.ops.aten.roll.default,
            (right, [1], [0]),
            (32, 32),
            torch.bfloat16,
        )
    graph.call_function(
        memory_ops.store, (sink, [None, None], _convert(graph, state, torch.bfloat16))
    )
    seed = _call(
        graph, torch.ops.aten.mul.Scalar, (state, 0.5), (32, 32), torch.float32
    )
    output = _dot(graph, left, right, seed)
    plan = replace(_loop_plan(graph, (state,), (output,)), strategy="warp")
    carry = plan_warp_loop_carry(plan, 2)
    assert carry is not None
    return replace(plan, warp_loop_carry=carry)


def test_raw_readonly_plan_keeps_direct_and_transformed_ownership_separate():
    with patch.object(chain, "_shape", lambda node: tuple(node.meta["val"].shape)):
        plan = fixture()
        selected = plan_warp_loop_prefetch(plan, 3)
        assert selected is not None
        selected.check(plan)
        assert selected.direct_roles == frozenset({"a"})
        assert [(x.direct, x.swapped, x.inner, x.pitch) for x in selected.operands] == [
            (True, True, 0, 40),
            (False, False, 0, 32),
        ]
        assert selected.shared_bytes == 3 * (32 * 40 + 32 * 32) * 2


@pytest.mark.parametrize("mode", ["alias", "gather"])
def test_raw_mutation_and_cross_coordinate_reads_decline(mode):
    with patch.object(chain, "_shape", lambda node: tuple(node.meta["val"].shape)):
        assert plan_warp_loop_prefetch(fixture(mode), 3) is None


@pytest.mark.parametrize(
    "shape,threads", [((64, 32), 128), ((32, 64), 128), ((32, 16), 64)]
)
def test_actual_cute_raw_packets_match_original_pointwise_coordinates(shape, threads):
    cutlass = pytest.importorskip("cutlass")
    cute = pytest.importorskip("cutlass.cute")
    ir = pytest.importorskip("cutlass._mlir.ir")
    columns = min(threads, shape[1] // 8)
    covered = []
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            tiled = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    cutlass.BFloat16,
                    num_bits_per_copy=128,
                ),
                cute.make_layout((threads // columns, columns), stride=(columns, 1)),
                cute.make_layout((1, 8)),
            )
            for thread in range(threads):
                partition = tiled.get_slice(thread).partition_S(
                    cute.make_identity_tensor(shape)
                )
                for step in range(shape[0] * shape[1] // (threads * 8)):
                    packet = thread + step * threads
                    values = partition[None, step, 0]
                    assert cute.size(values) == 8
                    for lane in range(8):
                        actual = tuple(int(x) for x in values[lane])
                        expected = (
                            packet // (shape[1] // 8),
                            packet % (shape[1] // 8) * 8 + lane,
                        )
                        assert actual == expected
                        covered.append(actual)
    assert len(covered) == len(set(covered)) == shape[0] * shape[1]


def spec():
    return cast(
        "ConfigSpec",
        SimpleNamespace(
            cute_async_load_pipeline_enabled=False,
            cute_chained_matmul_search_enabled=True,
            cute_chained_loop_search_enabled=True,
        ),
    )


def test_explicit_stages_is_additive_and_other_async_controls_stay_rejected():
    base: dict[str, object] = {
        "cute_chained_mma_schedule": "cp_async",
        "cute_chained_fragment_reuse": True,
    }
    for stages in (0, 3):
        config = base | {"cute_async_load_stages": stages}
        ConfigSpec._normalize_cute_async_load_pipeline(
            spec(), config, fix_invalid=False
        )
        assert config == (base if stages == 0 else base | {"cute_async_load_stages": 3})
    for key, value in (
        ("cute_async_load_stages", 4),
        ("cute_async_load_stages", True),
        ("cute_async_load_lookahead", 2),
    ):
        with pytest.raises(InvalidConfig):
            ConfigSpec._normalize_cute_async_load_pipeline(
                spec(), base | {key: value}, fix_invalid=False
            )


@pytest.mark.parametrize("trips", [0, 1, 2, 3, 4, 32])
@pytest.mark.parametrize("leaves", [1, 2])
@pytest.mark.parametrize("fallback", [False, True])
def test_group_budget_preserves_each_slot_until_last_reader(trips, leaves, fallback):
    depth = 3
    pending = []
    slots = {}
    retired = set()

    def issue(ordinal):
        slot = ordinal % depth
        if ordinal >= depth:
            assert ordinal - depth in retired
        for leaf in range(leaves):
            value = (slot, leaf, ordinal)
            if fallback:
                slots[slot, leaf] = ordinal
            # A scalar fallback still commits an empty asynchronous group.
            pending.append(None if fallback else value)

    for ordinal in range(min(depth, trips)):
        issue(ordinal)
    for ordinal in range(trips):
        keep = pending_groups(depth, leaves, trips - ordinal - 1)
        assert len(pending) <= 8
        while len(pending) > keep:
            value = pending.pop(0)
            if value is not None:
                slot, leaf, published = value
                slots[slot, leaf] = published
        assert all(slots[ordinal % depth, leaf] == ordinal for leaf in range(leaves))
        retired.add(ordinal)
        if ordinal + depth < trips:
            issue(ordinal + depth)
    assert not pending


def test_future_rebind_changes_only_owned_load_bytes():
    lines = (
        'label = "chain_loop_index" # café',
        "x = ((chain_loop_index * 64) + -3) / 2",
    )
    assert _future_index(lines) == (
        lines[0],
        "x = ((chain_prefetch_index * 64) + -3) / 2",
    )
    with pytest.raises(chain._UnsupportedChain, match="writes lexical index"):
        _future_index(["chain_loop_index = 3"])


@pytest.mark.parametrize(
    "depth,leaves,remaining", [(0, 2, 0), (3, 0, 0), (3, 3, 0), (3, 2, -1)]
)
def test_pending_group_capacity_rejects(depth, leaves, remaining):
    with pytest.raises(ValueError):
        pending_groups(depth, leaves, remaining)


@pytest.mark.parametrize("shape", [(32, 64), (64, 32)])
@pytest.mark.parametrize(
    "role,swapped", [("a", False), ("a", True), ("b", False), ("b", True)]
)
def test_actual_weighted_target_matches_every_role_and_transpose(shape, role, swapped):
    graph = Graph()
    leaf = _input(graph, "leaf", shape, torch.bfloat16)
    logical = shape[::-1] if swapped else shape
    operand = _input(graph, "operand", logical, torch.bfloat16)
    item = RawOperand(operand, leaf, role, shape, swapped, False, 8)
    contraction = (
        (logical[0], 32, logical[1]) if role == "a" else (32, logical[1], logical[0])
    )
    plan = SimpleNamespace(
        threads=128, shapes=(contraction,), operand_dtype=lambda stage: torch.bfloat16
    )
    emission = WarpLoopPrefetchEmission.__new__(WarpLoopPrefetchEmission)
    emission.cg = cast("Any", object())
    emission.plan = cast("Any", plan)
    emission.selected = cast(
        "Any", SimpleNamespace(depth=3, operands=(item,), check=lambda plan: None)
    )
    emission.copies = (("cute.arch.cp_async_commit_group()",),)
    emission.emitted = False
    visited = []

    class Expression:
        def __init__(self, *args):
            self.coordinate_names = set()
            self.fragments = {}
            self.lines = []

        def value(self, node, coordinates):
            assert node is operand and leaf in self.fragments
            visited.append(coordinates)
            return self.fragments[leaf][1]

    with (
        patch.object(chain, "_Expression", Expression),
        patch.object(chain, "_operand_domain", return_value=()),
        patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(
                backend=SimpleNamespace(dtype_str=lambda dtype: "cutlass.BFloat16")
            ),
        ),
    ):
        result = emission.operands()
    assert result.matches() and len(visited) == 1
    target = next(
        node.value
        for node in ast.walk(ast.parse("\n".join(result.lines)))
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == f"chain_raw_{role}_weighted_target"
    )
    assert isinstance(target, ast.Call)
    target_layout = cast("ast.Call", target.args[1])
    target_shape = ast.literal_eval(target_layout.args[0])
    target_stride = ast.literal_eval(target_layout.keywords[0].value)
    actual = result.layouts[0]
    assert target_shape == shape
    for row in range(shape[0]):
        for column in range(shape[1]):
            flat = row * shape[1] + column
            scope = {
                f"chain_raw_{role}_packet": flat // 8,
                f"chain_raw_{role}_lane": flat % 8,
            }
            original_coordinates = tuple(eval(text, {}, scope) for text in visited[0])
            physical = (
                original_coordinates if role == "a" else original_coordinates[::-1]
            )
            assert all(
                0 <= value < size
                for value, size in zip(physical, actual.shape, strict=True)
            )
            store_offset = row * target_stride[0] + column * target_stride[1]
            mma_offset = sum(
                value * stride
                for value, stride in zip(physical, actual.stride, strict=True)
            )
            assert store_offset == mma_offset


@pytest.mark.parametrize("dtype", ["BFloat16", "Float16"])
@pytest.mark.parametrize("aligned", [False, True])
def test_dynamic_ring_pointer_copy_ir_retains_proved_alignment(dtype, aligned):
    cutlass = pytest.importorskip("cutlass")
    cute = pytest.importorskip("cutlass.cute")
    ir = pytest.importorskip("cutlass._mlir.ir")
    func = pytest.importorskip("cutlass._mlir.dialects.func")
    native_dtype = getattr(cutlass, dtype)
    item = RawOperand(
        cast("Any", None), cast("Any", None), "b", (64, 32), False, False, 8
    )
    assert item.slot_elements * 2 % 128 == 0
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(
                "raw_packet", ([ir.IntegerType.get_signless(32)], [])
            )
        with ir.InsertionPoint(function.add_entry_block()):
            slot = cutlass.Int32(function.arguments[0])
            ring = cute.make_ptr(
                native_dtype, 16384, cute.AddressSpace.smem, assumed_align=128
            )
            pointer = (
                eval(
                    item.slot_pointer("slot"),
                    {},
                    {"chain_raw_b_ring": ring, "slot": slot},
                )
                if aligned
                else ring + slot * item.slot_elements
            )
            raw = cute.make_tensor(
                pointer, cute.make_layout(item.shape, stride=(32, 1))
            )
            target = cute.make_tensor(
                cute.make_ptr(
                    native_dtype, 32768, cute.AddressSpace.smem, assumed_align=128
                ),
                cute.make_layout(item.shape, stride=(40, 1)),
            )
            copy = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), native_dtype, num_bits_per_copy=128
                ),
                cute.make_layout((32, 4), stride=(4, 1)),
                cute.make_layout((1, 8)),
            )
            thread = copy.get_slice(0)
            source = thread.partition_S(raw)
            destination = thread.partition_D(target)
            global_source = cute.make_tensor(
                cute.make_ptr(
                    native_dtype, 0x100000, cute.AddressSpace.gmem, assumed_align=16
                ),
                cute.make_layout(item.shape, stride=(32, 1)),
            )
            ingress = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(),
                    native_dtype,
                    num_bits_per_copy=128,
                ),
                cute.make_layout((32, 4), stride=(4, 1)),
                cute.make_layout((1, 8)),
            )
            ingress_thread = ingress.get_slice(0)
            cute.copy(
                ingress,
                ingress_thread.partition_S(global_source),
                ingress_thread.partition_D(raw),
            )
            registers = cute.make_rmem_tensor_like(source[None, 0, 0])
            cute.copy(copy, source[None, 0, 0], registers)
            cute.copy(copy, registers, destination[None, 0, 0])
            func.ReturnOp([])
        if aligned:
            assert module.operation.verify()
        else:
            with pytest.raises(ir.MLIRError, match="alignment"):
                module.operation.verify()
