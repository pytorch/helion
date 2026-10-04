from __future__ import annotations

import ast
import copy
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_pipeline import _config
import helion
from helion import exc
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_tcgen05 as root_stage
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_loop_tmem_carry_transport import (
    LoopTmemCarryTransport,
)
from helion._compiler.cute.chained_tcgen05 import _load_result
from helion._compiler.cute.chained_tmem_snapshots import PackedSnapshotPanels
from helion._compiler.cute.chained_tmem_snapshots import emit_streamed_snapshot
from helion._compiler.cute.chained_tmem_transport import emit_packed_tmem_fragment
import helion.language as hl


def _emit(shape=(128, 128), dtype="cutlass.BFloat16", execution=None):
    return emit_streamed_snapshot(
        PackedSnapshotPanels(shape, 32, 320, 448),
        "snapshot",
        "original",
        dtype,
        ("row", "column"),
        (f"value = {dtype}(original_values[snapshot_index])",),
        "value",
        execution=execution or ChainedExecution(128),
    )


@pytest.mark.parametrize("width", [64, 96, 128, 160, 192, 224, 256])
def test_complete_n32_geometry_and_original_full_layout(width):
    panels = PackedSnapshotPanels((128, width), 32, 320, 448)
    assert panels.count == width // 32
    source = "\n".join(_emit((128, width)))
    tree = ast.parse(source)
    loops = [n for n in tree.body if isinstance(n, ast.For)]
    assert len(loops) == 1
    loop = loops[0]
    assert ast.unparse(loop.iter) == f"cutlass.range({width // 32}, unroll=1)"
    assert "make_rmem_tensor" not in ast.unparse(loop)
    assert "sync_threads" not in ast.unparse(loop)
    assert source.count("make_rmem_tensor") == 2
    assert source.count("sync_threads()") == 2
    assert source.count("fence_view_async_tmem_load()") == 1
    assert source.count("fence_view_async_tmem_store()") == 1
    assert f"make_identity_tensor((128, {width}))" in source
    assert f"original_acc.layout, cute.make_layout((128, {width // 2}))" in source
    assert "row, column = original_coords[snapshot_index]" in source
    assert "snapshot_panel * 32" in source
    assert source.endswith("fence_view_async_tmem_store()\ncute.arch.sync_threads()")
    setup = tree.body[: tree.body.index(loop)]
    setup_names = {
        name.id
        for statement in setup
        for name in ast.walk(statement)
        if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)
    }
    dynamic_names = {
        name.id
        for name in ast.walk(loop)
        if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)
    }
    # Mutating buffer elements is allowed, but static layout containers must
    # not become dynamic loop-carried values in the actual CuTe AST converter.
    assert not setup_names.intersection(dynamic_names)


@pytest.mark.parametrize(
    "shape,source,destination,allocation",
    [
        ((64, 128), 0, 256, 512),
        ((128, 32), 0, 256, 512),
        ((128, 80), 0, 256, 512),
        ((128, 288), 0, 320, 512),
        ((128,), 0, 256, 512),
        ([128, 128], 0, 256, 512),
        ((128, 128.0), 0, 256, 512),
        ((True, 128), 0, 256, 512),
        ((128, 128), True, 256, 512),
        ((128, 128), -16, 256, 512),
        ((128, 128), 8, 256, 512),
        ((128, 128), 0, 272, 512),
        ((128, 128), 0, 256.0, 512),
        ((128, 128), 0, 0, 512),
        ((128, 128), 0, 96, 512),
        ((128, 128), 16, 0, 512),
        ((128, 128), 400, 0, 512),
        ((128, 128), 0, 480, 512),
        ((128, 128), 0, 256, 513),
        ((128, 128), 0, 256, True),
    ],
)
def test_invalid_geometry_bounds_and_all_overlap_reject(
    shape, source, destination, allocation
):
    with pytest.raises(ValueError):
        PackedSnapshotPanels(shape, source, destination, allocation)


@pytest.mark.parametrize("dtype", ["cutlass.Float32", "cutlass.Int16", "bf16"])
def test_non_half_dtype_rejects(dtype):
    with pytest.raises(ValueError, match="BF16 or FP16"):
        _emit(dtype=dtype)


@pytest.mark.parametrize("threads", [32, 64, 96, 160, 256])
def test_core_requires_exact_participant_team(threads):
    with pytest.raises(ValueError, match="exactly 128"):
        _emit(execution=ChainedExecution(threads))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _snapshot_sequence(a, b, c, d, initial):
    steps, rows, reduction = a.shape
    width = initial.shape[-1]
    history = torch.empty((steps, rows, width), device=a.device)
    final = torch.empty_like(initial)
    for rr in hl.tile(rows, block_size=128):
        state = initial[rr, :]
        for step in hl.tile(steps, block_size=1):
            kk, jj = hl.arange(reduction), hl.arange(width)
            prepared = hl.dot(c[step.id, jj, kk], d[step.id, kk, jj]).to(a.dtype)
            projected = hl.dot(state.to(a.dtype), prepared, out_dtype=torch.float32)
            state = hl.dot(a[step.id, rr, kk], b[step.id, kk, jj], acc=state * 0.5)
            history[step.id, rr, jj] = projected
        final[rr, :] = state
    return history, final


def _args(dtype, width, steps):
    return (
        *(
            torch.empty(shape, dtype=dtype)
            for shape in (
                (steps, 128, 16),
                (steps, 16, width),
                (steps, width, 16),
                (steps, 16, width),
            )
        ),
        torch.empty((128, width), dtype=torch.float32),
    )


@pytest.mark.parametrize(
    "dtype,width,steps,warps",
    [
        (torch.bfloat16, 64, 0, 4),
        (torch.float16, 64, 1, 8),
        (torch.bfloat16, 64, 5, 4),
    ],
)
def test_real_carry_adapter_defaults_exact_and_explicit_stream(
    dtype, width, steps, warps
):
    original = LoopTmemCarryTransport.snapshot
    records = []

    def observe(self, execution):
        records.append((self, execution))
        return original(self, execution)

    args, config = (
        _args(dtype, width, steps),
        _config(16, pipeline=True, consumer_warps=warps),
    )
    config.config["cute_chained_warp_mma_rows"] = width
    with patch.object(LoopTmemCarryTransport, "snapshot", observe):
        before = _source(_snapshot_sequence, args, config)
    assert len(records) == 1
    carry, execution = records[0]
    prefix = f"{carry.prefix}_snapshot"
    local = ChainedExecution(
        128,
        thread=execution.thread,
        warp=execution.warp,
        sync="chain_tmem_barrier.arrive_and_wait()",
    )
    old_body = [
        *_load_result(carry.prefix, carry.candidate.snapshot_shape, execution=local),
        *emit_packed_tmem_fragment(
            prefix,
            carry.prefix,
            carry.candidate.snapshot_shape,
            carry.dtype,
            (f"{prefix}_row", f"{prefix}_column"),
            carry.snapshot_lines,
            carry.snapshot_value,
            execution=local,
            destination=f"(chain_tptr + {carry.snapshot_offset})",
        ),
    ]
    from helion._compiler.cute.chained_matmul import _indent

    expected = (
        [f"if {execution.thread} < 128:", _indent(old_body)]
        if execution.threads > 128
        else old_body
    ) + [execution.sync]
    assert (
        original(carry, execution)
        == original(carry, execution, tile_columns=0)
        == expected
    )

    def streamed(self, execution):
        return original(self, execution, tile_columns=32)

    with patch.object(LoopTmemCarryTransport, "snapshot", streamed):
        after = _source(_snapshot_sequence, args, config)
    old = "\n".join(expected)
    new = "\n".join(original(carry, execution, tile_columns=32))
    # Compare the entire original kernel after reversing exactly this action.
    before_ast, after_ast = ast.parse(before), ast.parse(after)
    assert ast.dump(before_ast) != ast.dump(after_ast)
    old_statements, new_statements = ast.parse(old).body, ast.parse(new).body
    changed = 0

    def match(expected, actual, names):
        if type(expected) is not type(actual):
            return False
        if isinstance(expected, ast.Name) and expected.id.startswith("chain_value"):
            if not actual.id.startswith("chain_value"):
                return False
            if expected.id not in names and actual.id in names.values():
                return False
            return names.setdefault(expected.id, actual.id) == actual.id
        if isinstance(expected, ast.AST):
            return all(
                match(old, new, names)
                for (_, old), (_, new) in zip(
                    ast.iter_fields(expected), ast.iter_fields(actual), strict=True
                )
            )
        if isinstance(expected, list):
            return len(expected) == len(actual) and all(
                match(old, new, names)
                for old, new in zip(expected, actual, strict=True)
            )
        return expected == actual

    def restore(node):
        nonlocal changed
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for index in range(len(value) - len(new_statements) + 1):
                    selected = value[index : index + len(new_statements)]
                    names = {}
                    if match(new_statements, selected, names):
                        replacement = copy.deepcopy(old_statements)
                        for statement in replacement:
                            for child in ast.walk(statement):
                                if isinstance(child, ast.Name) and child.id in names:
                                    child.id = names[child.id]
                        value[index : index + len(new_statements)] = replacement
                        changed += 1
                        break
                for item in value:
                    if isinstance(item, ast.AST):
                        restore(item)
            elif isinstance(value, ast.AST):
                restore(value)

    restore(after_ast)
    assert changed == 1
    assert ast.dump(after_ast) == ast.dump(before_ast)
    assert (
        old.count("chain_tmem_barrier.arrive_and_wait()")
        == new.count("chain_tmem_barrier.arrive_and_wait()")
        == 2
    )
    assert new.endswith(execution.sync)
    assert "if " + execution.thread + " < 128:" in new if warps > 4 else True
    invalid_values: tuple[Any, ...] = (True, 1, 16, 64, 32.0)
    for invalid in invalid_values:
        with pytest.raises(exc.BackendUnsupported):
            original(carry, execution, tile_columns=invalid)
    with pytest.raises(exc.BackendUnsupported, match="disjoint"):
        original(
            replace(carry, snapshot_offset=carry.arena_offset),
            execution,
            tile_columns=32,
        )
    with pytest.raises(exc.BackendUnsupported, match="128-thread"):
        original(carry, ChainedExecution(160), tile_columns=32)
    with pytest.raises(exc.BackendUnsupported, match="expression proof"):
        original(
            replace(carry, snapshot_value="cutlass.BFloat16(0)"),
            execution,
            tile_columns=32,
        )
    with pytest.raises(exc.BackendUnsupported, match="expression proof"):
        original(replace(carry, snapshot_proof=None), execution, tile_columns=32)
    source_node = carry.candidate.input
    metadata = source_node.meta["val"]
    try:
        source_node.meta["val"] = torch.empty((1,), dtype=torch.float16)
        with pytest.raises(exc.BackendUnsupported, match="expression proof"):
            original(carry, execution, tile_columns=32)
    finally:
        source_node.meta["val"] = metadata


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _side_input_snapshot(a, b, c, side, masked: hl.constexpr):
    rows, reduction = a.shape
    width = b.shape[-1]
    output = torch.empty((rows, width), dtype=a.dtype, device=a.device)
    for rr, jj in hl.tile([rows, width], block_size=[128, 64]):
        kk = hl.arange(reduction)
        value = hl.dot(a[rr, kk], b[kk, jj])
        if masked:
            coefficient = hl.load(
                side, [rr, jj], extra_mask=(jj.index % 3 != 0)[None, :]
            )
        else:
            coefficient = hl.load(side, [rr, jj])
        image = (value * torch.exp(coefficient.float()) + 0.25).to(a.dtype)
        ll = hl.arange(width)
        output[rr, jj] = hl.dot(image, c[ll, jj]).to(a.dtype)
    return output


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("masked", [False, True])
def test_original_pointwise_side_input_mask_and_cross_coordinate_guard(dtype, masked):
    original = root_stage._bridge
    captured = []

    def observe(cg, plan, boundaries, scans, stage, dtype):
        coords = ("snapshot_row", "snapshot_column")
        source = plan.dots[stage - 1]
        operand = plan.dots[stage].args[0]
        expression = chain._Expression(cg, plan, boundaries)
        expression.coordinate_names.update(coords)
        expression.fragments[source] = (coords, "original_values[snapshot_index]")
        value = expression.value(operand, coords)
        domain = chain._operand_domain(cg, operand, coords, plan)
        assert expression.loaded_inputs  # Original immutable input, not a new copy.
        assert "exp2" in "\n".join(expression.lines)
        with pytest.raises(chain._UnsupportedChain, match="cross-coordinate"):
            expression.value(source, (coords[0], f"({coords[1]} + 1)"))
        rendered = emit_streamed_snapshot(
            PackedSnapshotPanels((128, 64), 0, 128, 256),
            "snapshot",
            "original",
            dtype,
            coords,
            expression.lines,
            chain._masked_operand(value, dtype, domain),
            execution=ChainedExecution(128),
        )
        loop = next(
            node
            for node in ast.parse("\n".join(rendered)).body
            if isinstance(node, ast.For)
        )
        values_loop = next(node for node in loop.body if isinstance(node, ast.For))
        assert [ast.dump(node) for node in values_loop.body[1:-1]] == [
            ast.dump(node) for node in ast.parse("\n".join(expression.lines)).body
        ]
        captured.append(rendered)
        return original(cg, plan, boundaries, scans, stage, dtype)

    args = (
        *(
            torch.empty(shape, dtype=dtype)
            for shape in ((128, 32), (32, 64), (64, 64), (128, 64))
        ),
        masked,
    )
    config = helion.Config(num_warps=4, cute_chained_mma_schedule="tcgen05_tmem")
    with patch.object(root_stage, "_bridge", observe):
        _source(_side_input_snapshot, args, config)
    assert len(captured) == 1
