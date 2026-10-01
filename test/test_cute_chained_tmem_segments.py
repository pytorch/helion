from __future__ import annotations

import ast
import importlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_group_guards import _plan
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
from helion._compiler.cute.chained_tcgen_stage import _publish_result
from helion._compiler.cute.chained_tmem_segments import emit_tmem_segment_load

_SEGMENTS = (
    (144, 0, 16),
    (144, 16, 128),
    (144, 96, 48),
    (160, 0, 32),
    (160, 32, 128),
    (160, 16, 48),
    (256, 128, 64),
    (256, 208, 48),
    (256, 240, 16),
    (256, 16, 32),
    (256, 0, 128),
    (256, 128, 128),
)


@pytest.mark.parametrize("total,offset,width", _SEGMENTS)
def test_segment_source_keeps_global_coordinate_origin_and_fp32(total, offset, width):
    lines = emit_tmem_segment_load("member", "group", (128, total), offset, width)
    source = "\n".join(lines)
    assert source.count(f"cute.domain_offset(((0, {offset}), 0, 0)") == 2
    assert "group_acc" in lines[0]
    assert (
        f"group_slice.partition_C(cute.make_identity_tensor((128, {total})))"
        in lines[1]
    )
    assert f"tcgen05.Repetition({min(32, width & -width)})" in source
    assert "member_coords = member_thread.partition_D(member_identity)" in source
    assert (
        "member_values = cute.make_rmem_tensor(member_coords.shape, cutlass.Float32)"
        in source
    )
    assert lines[-1] == "cute.arch.fence_view_async_tmem_load()"
    assert "BFloat16" not in source and "Float16" not in source
    assert "mbarrier" not in source and "arrive_and_wait" not in source
    assert "sync_threads" not in source and "alloc_smem" not in source
    assert not any(
        isinstance(node, (ast.For, ast.If, ast.While))
        for node in ast.walk(ast.parse(source))
    )


@pytest.mark.parametrize("threads", [128, 160, 256, 512, 1024])
def test_execution_only_selects_thread_name_not_role_barriers_or_predicates(threads):
    execution = ChainedExecution(
        threads, thread="consumer_thread", sync="must_not_emit.arrive_and_wait()"
    )
    lines = emit_tmem_segment_load(
        "member", "group", (128, 160), 32, 128, execution=execution
    )
    source = "\n".join(lines)
    assert "get_slice(consumer_thread)" in source
    assert "must_not_emit" not in source and "if " not in source
    assert "chain_thread" not in source


@pytest.mark.parametrize("threads", [32, 64, 96])
def test_insufficient_tmem_participants_reject(threads):
    with pytest.raises(ValueError, match="at least 128"):
        emit_tmem_segment_load(
            "member", "group", (128, 160), 32, 128, execution=ChainedExecution(threads)
        )


@pytest.mark.parametrize(
    "shape,offset,width",
    [
        ((64, 160), 0, 32),
        ((128, 0), 0, 16),
        ((128, 144), 0, 0),
        ((128, 160), -16, 32),
        ((128, 160), 8, 32),
        ((128, 160), 0, 8),
        ((128, 160), 144, 32),
        ((128, 160), 160, 16),
        ((128, 272), 0, 16),
        ((128, 150), 0, 32),
        ((128,), 0, 32),
        ((128, 160.0), 0, 32),
        ((True, 160), 0, 32),
        ((128, 160), True, 32),
        ((128, 160), 0, True),
        ((128, 160), 0.0, 32),
        ((128, 160), 0, 32.0),
    ],
)
def test_invalid_physical_segments_fail_closed(shape, offset, width):
    with pytest.raises(ValueError, match="contained full-M128"):
        emit_tmem_segment_load("member", "group", shape, offset, width)


def _depends_on(value: Any, root: Any) -> bool:
    """Trace actual MLIR SSA operands back to the nonzero caller pointer."""
    ir = importlib.import_module("cutlass._mlir.ir")
    pending, seen = [value], set()
    while pending:
        current = ir.Value(pending.pop())
        if current == root:
            return True
        if current in seen:
            continue
        seen.add(current)
        for operand in current.owner.operation.operands:
            # CuTe's registered MLIR value casters wrap memrefs/layouts when
            # generic Operation.operands exposes them to Python.
            pending.extend(
                [operand]
                if isinstance(operand, ir.Value)
                else operand.__extract_mlir_values__()
            )
    return False


@pytest.mark.parametrize("dtype_name", ["BFloat16", "Float16"])
@pytest.mark.parametrize("total,offset,width", _SEGMENTS)
def test_actual_cute_member_load_coverage_and_source_base_cpu(
    dtype_name, total, offset, width
):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05
    from cutlass.utils import blackwell_helpers

    ir = importlib.import_module("cutlass._mlir.ir")
    dtype = {"BFloat16": cutlass.BFloat16, "Float16": cutlass.Float16}[dtype_name]
    before = torch.cuda.is_initialized()
    lines = emit_tmem_segment_load("member", "group", (128, total), offset, width)
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = blackwell_helpers.make_trivial_tiled_mma(
                dtype,
                dtype,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                (128, total),
                tcgen05.OperandSource.SMEM,
            )
            layout = mma.make_fragment_C(mma.partition_shape_C((128, total))).layout
            base = cute.make_ptr(cutlass.Float32, 224, cute.AddressSpace.tmem)
            base_ir = base.__extract_mlir_values__()[0]
            accumulator = cute.make_tensor(base, layout)
            coverage = set()
            for thread in range(128):
                namespace: dict[str, Any] = {
                    "cutlass": cutlass,
                    "cute": cute,
                    "tcgen05": tcgen05,
                    "group_acc": accumulator,
                    "group_slice": mma.get_slice(0),
                    "chain_thread": thread,
                }
                # Execute the actual source helper, including its real load-copy
                # operation and load fence. Nothing is launched on a device.
                exec("\n".join(lines), namespace)
                segment, coords, values = (
                    namespace[f"member_{name}"]
                    for name in ("segment", "coords", "values")
                )
                assert segment.element_type == values.element_type == cutlass.Float32
                assert int(cute.size(coords)) == int(cute.size(values)) == width
                assert _depends_on(
                    segment.iterator.__extract_mlir_values__()[0], base_ir
                )
                for index in range(width):
                    row, column = map(int, coords[index])
                    assert row == thread and offset <= column < offset + width
                    assert (row, column) not in coverage
                    coverage.add((row, column))
                    local = int(segment.layout(((row, column - offset), 0, 0)))
                    original = int(layout(((row, column), 0, 0)))
                    assert 224 + offset + local == 224 + original
            assert coverage == {
                (row, column)
                for row in range(128)
                for column in range(offset, offset + width)
            }
        assert module.operation.verify()
        text = str(module)
        assert "tmem_load<f32" in text and "tmem_store" not in text
    assert torch.cuda.is_initialized() == before


@pytest.mark.parametrize("transpose", [False, True])
def test_global_member_coordinates_feed_existing_publication_without_guard_changes(
    transpose,
):
    graph = Graph()
    m, n = (48, 128) if transpose else (128, 48)
    left = _input(graph, "left", (m, 16), torch.bfloat16)
    right = _input(graph, "right", (16, n), torch.bfloat16)
    node = _dot(graph, left, right)
    plan = _plan(graph, (node,))
    boundaries = {}
    lines = _publish_result(
        plan, boundaries, "member", ((0, StageGeometry((m, n, 16), transpose), 96),)
    )
    assert boundaries[node] == "chain_0_c"
    coordinates = [(row, column) for row in range(128) for column in range(96, 144)]
    output = {}
    namespace = {
        "cutlass": SimpleNamespace(range_constexpr=range),
        "cute": SimpleNamespace(size=len),
        "member_values": list(range(len(coordinates))),
        "member_coords": coordinates,
        "chain_0_c": output,
    }
    exec("\n".join(lines), namespace)
    assert output == {
        ((column - 96, row) if transpose else (row, column - 96)): index
        for index, (row, column) in enumerate(coordinates)
    }
