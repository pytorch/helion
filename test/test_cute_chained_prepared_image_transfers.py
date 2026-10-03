from __future__ import annotations

from dataclasses import replace
import importlib
import re
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_group_guards import _plan
from .test_cute_chained_operand_retention import _case
from helion._compiler.cute.chained_contraction_groups import ContractionGroup
from helion._compiler.cute.chained_operand_retention import discover_operand_retention
from helion._compiler.cute.chained_preparation_cut import PreparationImage
from helion._compiler.cute.chained_preparation_frame import PreparationAction
from helion._compiler.cute.chained_preparation_frame import PreparationBuffer
from helion._compiler.cute.chained_prepared_image_emission import (
    bind_raw_widening_transfers,
)
from helion._compiler.cute.chained_prepared_image_transfers import (
    discover_native_identity_crops,
)
from helion._compiler.cute.chained_prepared_image_transfers import (
    discover_raw_widening_transfers,
)
from helion._compiler.cute.chained_prepared_image_transfers import (
    plan_native_identity_crops,
)
from helion._compiler.cute.chained_tcgen_stage import StageGeometry
from helion._compiler.cute.warp_specialized_plan import SharedBufferRegion
from helion._compiler.cute.warp_specialized_plan import SharedMemoryLayoutPlan
from helion.language import memory_ops


def _native_case(dtype=torch.bfloat16, *, member=1):
    plan, frame, shapes, values = _case(dtype)
    assert plan.region is not None and plan.contraction_groups is not None
    graph = plan.region.graph
    output = next(node for node in graph.nodes if node.op == "output")
    old_outputs = output.args[0]
    graph.erase_node(output)
    state = _input(graph, "state", (128, 128), dtype)
    result = _dot(graph, values[member], state)
    fresh = _plan(graph, (*old_outputs, result))
    group = ContractionGroup((2,), (StageGeometry((32, 128, 128), True),))
    plan = replace(
        fresh,
        strategy=plan.strategy,
        contraction_groups=(*plan.contraction_groups, group),
        warp_mma_stages=plan.warp_mma_stages,
    )
    assert plan.region is not None
    buffers = tuple(
        replace(buffer, node=values[member], dtype=dtype)
        if buffer.name == "frontier_1"
        else buffer
        for buffer in frame.buffers
    )
    actions = tuple(
        replace(action, nodes=(values[member],), reads=(f"raw_{member}",))
        if action.event == 4
        else action
        for action in frame.actions
    )
    images = tuple(
        PreparationImage(
            buffer.node,
            buffer.dtype,
            tuple(buffer.node.meta["val"].shape),
            (result,) if buffer.node is values[member] else (),
        )
        for buffer in buffers
        if buffer.kind == "frontier" and buffer.node is not None
    )
    cut = replace(
        frame.cut,
        region=plan.region,
        recurrence=(result,),
        images=images,
    )
    frame = replace(frame, cut=cut, buffers=buffers, actions=actions)
    shapes.update({state: (128, 128), result: (32, 128)})
    return plan, frame, shapes, values[member]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("member", [1, 2])
def test_native_crop_keeps_full_owner_and_k_panel_stride(dtype, member):
    plan, frame, shapes, node = _native_case(dtype, member=member)
    available = discover_native_identity_crops(plan, frame, shapes)
    assert available is not None and len(available) == 1
    crop = available[0]
    assert crop.source.node is node
    assert crop.source.full_shape == (64, 128)
    assert crop.source.row_offset == 32 * (member - 1)
    assert crop.alias_lines("full_native_b", "image") == (
        f"image = cute.local_tile(full_native_b, (32, 128), ({member - 1}, 0))",
    )
    candidates = discover_operand_retention(plan, frame, shapes)
    assert candidates is not None
    bound = plan_native_identity_crops(
        plan,
        frame,
        shapes,
        candidates,
        available,
        capacity_bytes=frame.layout.allocated_bytes,
    )
    assert bound is not None and bound.matches(plan, frame, shapes)
    old = frame.layout.region(crop.source.owner)
    new = bound.retention.frame.layout.region(crop.source.owner)
    assert new.byte_size == old.byte_size == 16384
    assert new.live_from == old.live_from
    assert new.live_until == len(frame.actions)
    assert tuple(r.name for r in bound.retention.frame.layout.regions) == tuple(
        r.name for r in frame.layout.regions
    )
    assert frame.layout.region(crop.source.owner) is old
    assert not replace(bound, crops=()).matches(plan, frame, shapes)
    assert not replace(
        bound, retention=replace(bound.retention, candidates=())
    ).matches(plan, frame, shapes)
    assert (
        plan_native_identity_crops(
            plan, frame, shapes, candidates, available, capacity_bytes=16384
        )
        is None
    )


@pytest.mark.parametrize("change", ["cast", "mask", "other_user", "dtype", "shape"])
def test_native_crop_rejects_changed_identity_or_recurrence_uses(change):
    plan, frame, shapes, node = _native_case()
    assert plan.region is not None
    if change == "dtype":
        buffers = tuple(
            replace(buffer, dtype=torch.float32) if buffer.node is node else buffer
            for buffer in frame.buffers
        )
        frame = replace(frame, buffers=buffers)
    elif change == "shape":
        shapes = {**shapes, node: (16, 128)}
    else:
        graph = node.graph
        output = next(n for n in graph.nodes if n.op == "output")
        with graph.inserting_before(output):
            user = (
                _convert(graph, node, torch.float32)
                if change == "cast"
                else _call(
                    graph,
                    torch.ops.aten.neg.default,
                    (node,),
                    (32, 128),
                    node.meta["val"].dtype,
                )
            )
        # A graph edit invalidates the retained revision before any aliasing.
        assert user not in plan.region.nodes
    assert not discover_native_identity_crops(plan, frame, shapes)


def _raw_case(dtype=torch.bfloat16, *, masked=True):
    plan, frame, shapes, _ = _case()
    assert plan.region is not None
    graph = plan.region.graph
    output = next(node for node in graph.nodes if node.op == "output")
    old_outputs = output.args[0]
    graph.erase_node(output)
    host = _input(graph, "host", (32, 128), dtype)
    rows = _input(graph, "rows", (32, 1), torch.int32)
    columns = _input(graph, "columns", (1, 128), torch.int32)
    mask = _input(graph, "mask", (32, 128), torch.bool) if masked else None
    raw = _call(
        graph, memory_ops.load, (host, [rows, columns], mask, None), (32, 128), dtype
    )
    widening = _convert(graph, raw, torch.float32)
    use = _call(
        graph, torch.ops.aten.sub.Tensor, (widening, 0.25), (32, 128), torch.float32
    )
    fresh = _plan(graph, (*old_outputs, use))
    plan = replace(plan, region=fresh.region, store=fresh.store)
    assert plan.region is not None
    buffer = PreparationBuffer(
        "raw_image", "frontier", widening, torch.float32, (32, 128)
    )
    event = len(frame.actions) - 1
    region = SharedBufferRegion(
        buffer.name, frame.layout.allocated_bytes, 16384, event, event + 2, 128
    )
    frame = replace(
        frame,
        cut=replace(
            frame.cut,
            region=plan.region,
            preparation=(*frame.cut.preparation, raw, widening),
            recurrence=(use,),
            images=(PreparationImage(widening, torch.float32, (32, 128), (use,)),),
        ),
        buffers=(*frame.buffers, buffer),
        layout=SharedMemoryLayoutPlan(
            (
                *(
                    replace(item, live_until=event + 2)
                    if item.live_until == event + 1
                    else item
                    for item in frame.layout.regions
                ),
                region,
            ),
            region.byte_end,
        ),
        actions=(
            *frame.actions[:-1],
            PreparationAction(
                "frontier", event, (widening,), (), None, (), (buffer.name,)
            ),
            replace(frame.actions[-1], event=event + 1),
        ),
    )
    shapes.update(
        {
            n: tuple(n.meta["val"].shape)
            for n in (host, rows, columns, raw, widening, use)
        }
    )
    if mask is not None:
        shapes[mask] = (32, 128)
    return plan, frame, shapes, raw, widening, use


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("masked", [False, True])
def test_raw_transfer_retains_original_typed_masked_load(dtype, masked):
    plan, frame, shapes, raw, widening, use = _raw_case(dtype, masked=masked)
    transfers = discover_raw_widening_transfers(plan, frame, shapes)
    assert transfers is not None and len(transfers) == 1
    transfer = transfers[0]
    assert (transfer.source, transfer.widening, transfer.consumer) == (
        raw,
        widening,
        use,
    )
    assert transfer.dtype == dtype and raw.meta["val"].dtype == dtype
    assert widening.meta["val"].dtype == transfer.buffer.dtype == torch.float32
    assert transfer.matches(plan, frame, shapes)
    assert not replace(transfer, source=widening).matches(plan, frame, shapes)
    assert not replace(transfer, dtype=torch.float32).matches(plan, frame, shapes)
    original_mask = raw.args[2]
    indices = raw.args[1]
    assert isinstance(indices, (list, tuple))
    raw.args = (*raw.args[:2], None if masked else indices[0], raw.args[3])
    assert raw.args[2] is not original_mask
    assert not transfer.matches(plan, frame, shapes)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("byte_size", [16384, 16383, 8192, 128])
def test_raw_transfer_requires_full_original_fp32_allocation(dtype, byte_size):
    plan, frame, shapes, _, _, _ = _raw_case(dtype)
    original = discover_raw_widening_transfers(plan, frame, shapes)
    assert original is not None and len(original) == 1
    changed = replace(
        frame,
        layout=replace(
            frame.layout,
            regions=tuple(
                replace(region, byte_size=byte_size)
                if region.name == "raw_image"
                else region
                for region in frame.layout.regions
            ),
        ),
    )
    found = discover_raw_widening_transfers(plan, changed, shapes)
    if byte_size == 16384:
        assert found is not None and len(found) == 1
        bound = bind_raw_widening_transfers(plan, changed, shapes, found)
        assert bound is not None and bound.matches(plan)
        assert bound.bindings[0].region.byte_size == 16384
    else:
        if byte_size == 16383:
            # The original frame validator rejects unaligned allocation sizes
            # before the transfer-specific full-payload guard is reached.
            assert found is None
        else:
            assert found == ()
        assert not original[0].matches(plan, changed, shapes)
        assert bind_raw_widening_transfers(plan, changed, shapes, original) is None


@pytest.mark.parametrize(
    "change",
    [
        "source_dtype",
        "cast_dtype",
        "multiuse",
        "nonlinear",
        "domain",
        "preparation_user",
    ],
)
def test_raw_transfer_rejects_non_identity_or_escaping_values(change):
    plan, frame, shapes, raw, widening, use = _raw_case()
    if change == "source_dtype":
        raw.meta["val"] = torch.empty((32, 128), dtype=torch.float32)
    elif change == "cast_dtype":
        widening.args = (raw, torch.float16)
    elif change == "domain":
        shapes[raw] = (16, 128)
    elif change == "preparation_user":
        frame = replace(frame, cut=replace(frame.cut, recurrence=()))
    else:
        output = next(n for n in raw.graph.nodes if n.op == "output")
        with raw.graph.inserting_before(output):
            _call(
                raw.graph,
                torch.ops.aten.exp.default,
                (raw if change == "nonlinear" else widening,),
                (32, 128),
                torch.float32,
            )
    assert not discover_raw_widening_transfers(plan, frame, shapes)


@pytest.mark.parametrize(
    "change",
    [
        "raw_second_user",
        "widened_second_user",
        "mask_dtype",
        "nonlinear",
        "load_keywords",
    ],
)
def test_recollected_raw_graph_still_rejects_unproved_transfers(change):
    from helion._compiler.cute.contraction_region import collect_contraction_region
    from helion._compiler.device_ir import RootGraphInfo

    plan, frame, shapes, raw, widening, use = _raw_case()
    assert plan.region is not None
    graph = plan.region.graph
    if change == "mask_dtype":
        mask = raw.args[2]
        assert isinstance(mask, torch.fx.Node)
        mask.meta["val"] = torch.empty((32, 128), dtype=torch.float32)
    elif change == "load_keywords":
        raw.kwargs = {"unexpected": True}
    else:
        with graph.inserting_before(
            widening
            if change == "nonlinear"
            else next(n for n in graph.nodes if n.op == "output")
        ):
            extra = _call(
                graph,
                torch.ops.aten.exp.default,
                (widening if change == "widened_second_user" else raw,),
                (32, 128),
                torch.float32,
            )
        if change == "nonlinear":
            widening.args = (extra, torch.float32)
        shapes[extra] = (32, 128)
    region = collect_contraction_region(RootGraphInfo(0, graph))
    assert region is not None
    plan = replace(plan, region=region)
    frame = replace(frame, cut=replace(frame.cut, region=region))
    assert discover_raw_widening_transfers(plan, frame, shapes) == ()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_raw_binding_keeps_semantic_dtype_allocation_and_original_guard(dtype):
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.cute import chained_matmul as chain
    from helion._compiler.host_function import HostFunction

    plan, frame, shapes, raw, widening, _ = _raw_case(dtype)
    transfers = discover_raw_widening_transfers(plan, frame, shapes)
    assert transfers
    bound = bind_raw_widening_transfers(plan, frame, shapes, transfers)
    assert bound is not None and bound.matches(plan)
    binding = bound.bindings[0]
    assert binding.buffer.node is raw and binding.buffer.dtype == dtype
    assert binding.region.byte_size == 16384
    assert bound.frame is frame
    original = {widening: transfers[0].buffer.name}
    boundaries = bound.recurrence_boundaries(original)
    assert original == {widening: "raw_image"}
    assert boundaries == {raw: "raw_image_raw"}
    typed_plan = replace(plan, prepared_widenings=bound)
    cg: Any = SimpleNamespace()
    expression = chain._Expression(cg, typed_plan, boundaries)
    dtype_name = "cutlass.BFloat16" if dtype == torch.bfloat16 else "cutlass.Float16"
    function = (
        "bfloat16_to_float32" if dtype == torch.bfloat16 else "float16_to_float32"
    )
    imported = []

    def import_from_module(scope, name):
        imported.append((scope["__name__"], name))
        return SimpleNamespace(host_str=lambda: f"typed_conversion.{name}")

    with (
        patch.object(expression, "bind", side_effect=lambda value: value),
        patch.object(
            chain, "_shape", side_effect=lambda node: tuple(node.meta["val"].shape)
        ),
        patch.object(
            CompileEnvironment,
            "current",
            return_value=SimpleNamespace(
                backend=SimpleNamespace(dtype_str=lambda value: dtype_name)
            ),
        ),
        patch.object(
            HostFunction,
            "current",
            return_value=SimpleNamespace(import_from_module=import_from_module),
        ),
    ):
        value = expression.value(widening, ("row", "column"))
    original_read = chain._materialized_value(
        "raw_image_raw", (32, 128), ("row", "column"), dtype_name
    )
    assert value == f"typed_conversion.{function}({original_read})"
    assert imported == [("helion._compiler.cute.chained_half_widening", function)]
    assert f"else {dtype_name}(0)" in value
    assert raw.meta["val"].dtype == dtype
    assert widening.meta["val"].dtype == torch.float32
    assert "stride=(128, 1)" in binding.view_lines("frame_bytes")[0]
    with pytest.raises(chain._UnsupportedChain, match="original boundary"):
        bound.recurrence_boundaries({widening: "wrong_version"})
    assert not replace(bound, bindings=()).matches(plan)
    assert not bound.matches(replace(plan, threads=256))


@pytest.mark.parametrize("dtype_name,ptx", [("BFloat16", "bf16"), ("Float16", "f16")])
def test_original_widening_actual_mlir_contract(dtype_name, ptx):
    import cutlass
    from cutlass._mlir.dialects import func

    from helion._compiler.cute import chained_half_widening as widening

    ir = importlib.import_module("cutlass._mlir.ir")

    dtype = {"BFloat16": cutlass.BFloat16, "Float16": cutlass.Float16}[dtype_name]
    helper = (
        widening.bfloat16_to_float32 if ptx == "bf16" else widening.float16_to_float32
    )
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        for name in ("original", "opaque"):
            with ir.InsertionPoint(module.body):
                function = func.FuncOp(
                    name, ([dtype.mlir_type], [cutlass.Float32.mlir_type])
                )
            with ir.InsertionPoint(function.add_entry_block()):
                value = dtype(function.arguments[0])
                result = cutlass.Float32(value) if name == "original" else helper(value)
                func.ReturnOp([result.ir_value()])
        assert module.operation.verify()
        original, opaque = tuple(module.body.operations)
        original_ops = tuple(original.regions[0].blocks[0].operations)
        opaque_ops = tuple(opaque.regions[0].blocks[0].operations)
        assert tuple(op.operation.name for op in original_ops) == (
            "arith.extf",
            "func.return",
        )
        assert tuple(op.operation.name for op in opaque_ops) == (
            "llvm.bitcast",
            "llvm.inline_asm",
            "func.return",
        )
        carrier, assembly = (op.operation for op in opaque_ops[:2])
        assert str(carrier.operands[0].type) == ptx
        assert str(carrier.results[0].type) == "i16"
        assert ir.Value(assembly.operands[0]) == ir.Value(carrier.results[0])
        assert str(assembly.results[0].type) == "f32"
        assert assembly.attributes["asm_string"].value == f"cvt.f32.{ptx} $0, $1;"
        assert assembly.attributes["constraints"].value == "=f,h"
        assert "has_side_effects" not in assembly.attributes
        assert "is_align_stack" not in assembly.attributes


@pytest.mark.parametrize("dtype_name", ["BFloat16", "Float16"])
@pytest.mark.parametrize("columns", [64, 128, 256])
@pytest.mark.parametrize("row_offset", [0, 32])
def test_actual_dynamic_native_crop_scalar_load_matches_full_owner(
    dtype_name, columns, row_offset
):
    import cutlass
    from cutlass._mlir.dialects import func
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05

    ir = importlib.import_module("cutlass._mlir.ir")
    passmanager = importlib.import_module("cutlass._mlir.passmanager")

    dtype = {"BFloat16": cutlass.BFloat16, "Float16": cutlass.Float16}[dtype_name]
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(
                "read",
                (
                    [
                        ir.IntegerType.get_signless(64),
                        ir.IntegerType.get_signless(32),
                        ir.IntegerType.get_signless(32),
                    ],
                    [dtype.mlir_type, dtype.mlir_type],
                ),
            )
        with ir.InsertionPoint(function.add_entry_block()):
            base = cutlass.Int64(function.arguments[0])
            row, column = (cutlass.Int32(arg) for arg in function.arguments[1:])
            layout = cute.tile_to_shape(
                tcgen05.make_smem_layout_atom(
                    tcgen05.SmemLayoutAtomKind.K_SW128, dtype
                ),
                (64, columns),
                order=(0, 1),
            )
            pointer = cute.make_ptr(
                dtype, base, cute.AddressSpace.smem, assumed_align=128
            )
            full = cute.make_tensor(
                cute.recast_ptr(pointer, layout.inner, dtype=dtype), layout.outer
            )
            crop = cute.local_tile(full, (32, columns), (row_offset // 32, 0))
            left, right = full[row + row_offset, column], crop[row, column]
            func.ReturnOp([left.ir_value(), right.ir_value()])
        passes = "builtin.module(cute-desugar,cute-fold-static,cute-expand-ops,convert-cute-to-core,canonicalize,cse)"
        passmanager.PassManager.parse(passes).run(module.operation)
        assert module.operation.verify()
        operations = tuple(function.regions[0].blocks[0].operations)
        returned = operations[-1].operation
        assert returned.name == "func.return"
        # Actual scalar Tensor loads with dynamic row/column/base lower to the
        # very same SSA result, including the swizzled pointer and outer panel.
        program = _native_program(function)
        for phase in range(8):
            for row_value in range(32):
                for column_value in range(columns):
                    base_value = (1 << 20) + phase * 128
                    actual, cropped = _run_native_program(
                        program, (base_value, row_value, column_value)
                    )
                    assert actual == cropped
                    assert base_value <= actual < base_value + 64 * columns * 2


def _native_program(function):
    """Compile the actual lowered scalar address operations into a tiny oracle."""
    ir = importlib.import_module("cutlass._mlir.ir")

    indices = {ir.Value(value): index for index, value in enumerate(function.arguments)}
    program = []
    for wrapped in function.regions[0].blocks[0].operations:
        op = wrapped.operation
        operands = tuple(indices[ir.Value(value)] for value in op.operands)
        result = len(indices)
        for value in op.results:
            indices[ir.Value(value)] = result
        if op.name == "arith.constant":
            extra = int(op.attributes["value"].value)
        elif op.name == "llvm.getelementptr":
            # Element type is half; the only supported fixed offset is read
            # from the actual LLVM operation, not reconstructed from geometry.
            match = re.search(r"\[(\d+)\]", str(op))
            extra = int(match[1]) if match else None
        else:
            extra = None
        program.append((op.name, result, operands, extra))
    return program


def _run_native_program(program, arguments):
    values = list(arguments)
    for name, result, operands, extra in program:
        args = [values[index] for index in operands]
        if name == "arith.constant":
            value = extra
        elif name == "arith.addi":
            value = args[0] + args[1]
        elif name == "arith.muli":
            value = args[0] * args[1]
        elif name in ("arith.divsi", "arith.floordivsi"):
            value = args[0] // args[1]
        elif name == "arith.remsi":
            value = args[0] % args[1]
        elif name == "arith.andi":
            value = args[0] & args[1]
        elif name == "arith.shrui":
            value = args[0] >> args[1]
        elif name == "arith.xori":
            value = args[0] ^ args[1]
        elif name in ("llvm.inttoptr", "llvm.ptrtoint", "llvm.load"):
            value = args[0]
        elif name == "llvm.getelementptr":
            value = args[0] + 2 * (extra if extra is not None else args[1])
        elif name == "llvm.intr.assume":
            continue
        elif name == "func.return":
            return tuple(args)
        else:
            raise AssertionError(f"unsupported native address operation: {name}")
        assert result == len(values)
        values.append(value)
    raise AssertionError("no returned addresses")
