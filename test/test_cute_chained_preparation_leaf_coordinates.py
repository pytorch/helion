"""Device-free checks of coordinates consumed by the real TMA copy operation.

These tests expand CuTe's copy to its architecture operation, then return that
operation's coordinate operands from a CPU-only MLIR probe. They do not compile
PTX, allocate a descriptor on a GPU, or exercise the runtime barrier protocol.
"""

from __future__ import annotations

import ast
import importlib
import operator
from types import SimpleNamespace
from unittest.mock import patch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
import pytest
import torch
from torch.fx import Graph

from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_preparation_leaves import PreparationLeaf
from helion._compiler.cute.chained_rectangular_leaf import prove_rectangular_leaf

ir = importlib.import_module("cutlass._mlir.ir")
passmanager = importlib.import_module("cutlass._mlir.passmanager")
func = importlib.import_module("cutlass._mlir.dialects.func")


def _leaf(dtype):
    # A compact [token, head, channel] source, flattened to a pitched 2D TMA
    # descriptor. The channel offset is deliberately not a 64-element tile.
    proof = prove_rectangular_leaf(
        ("start + row", "head", "8 + column"),
        {},
        row="row",
        column="column",
        uniform_names={"start", "head"},
        tile_shape=(16, 64),
        shape=(512, 8, 128),
        strides=(1024, 128, 1),
        dtype=dtype,
    )
    assert proof is not None
    return PreparationLeaf(
        Graph().placeholder("raw"),
        "leaf",
        proof,
        0,
        1,
        (1,),
        {"kernel_args": ["leaf_atom", "leaf_tensor"]},
    )


def _source(leaf):
    # Only scalar fallback-expression construction is mocked. The actual fast
    # branch, layout, partition, and copy are all emitted by production code.
    expression = SimpleNamespace(
        coordinate_names=set(), lines=[], value=lambda *args: "fallback_value"
    )
    execution = ChainedExecution(
        384, thread="prep_thread", warp="prep_warp", sync="prep_sync()"
    )
    with patch(
        "helion._compiler.cute.chained_matmul._Expression",
        return_value=expression,
    ) as expression_type:
        tree = ast.parse(
            "\n".join(leaf.emit(None, None, {}, execution, "barrier", "phase"))
        )
        expression_type.assert_called_once_with(None, None, {})
        return tree


def _walk(operation):
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk(child.operation)


def _copy_coordinate_probe(leaf, *, grid_truncation=False):
    transactions = max(1, leaf.proof.tile_shape[1] * leaf.proof.element_bytes // 128)
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        i32 = ir.IntegerType.get_signless(32)
        function = func.FuncOp("coordinates", ([i32, i32], [i32] * (2 * transactions)))
    with ir.InsertionPoint(function.add_entry_block()):
        namespace = {
            "cutlass": cutlass,
            "cute": cute,
            "tcgen05": tcgen05,
            "start": cutlass.Int32(function.entry_block.arguments[0]),
            "head": cutlass.Int32(function.entry_block.arguments[1]),
            "leaf_bytes": cute.make_ptr(
                cutlass.Int8, 0, cute.AddressSpace.smem, assumed_align=128
            ),
            "barrier": cute.make_ptr(
                cutlass.Int64, 64, cute.AddressSpace.smem, assumed_align=16
            ),
        }
        exec("\n".join(leaf.view("leaf_bytes", 128)), namespace)
        dtype = eval(leaf.proof.dtype, namespace)
        host = cute.make_tensor(
            cute.make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=16),
            cute.make_layout(leaf.proof.view_shape, stride=(leaf.proof.pitch, 1)),
        )
        namespace["leaf_atom"], namespace["leaf_tensor"] = (
            cute.nvgpu.cpasync.make_tiled_tma_atom(
                cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
                host,
                namespace["leaf_layout"],
                leaf.proof.tile_shape,
            )
        )
        root = _source(leaf)
        assert isinstance(root.body[0], ast.If)
        warp = next(
            node
            for node in root.body[0].body
            if isinstance(node, ast.If) and ast.unparse(node.test) == "prep_warp == 0"
        )
        body = ast.Module(body=warp.body, type_ignores=[])
        if grid_truncation:
            # Negative control: the tempting but wrong grid-coordinate API.
            # Keep production's computed element origin, then floor it to a
            # tile before local_tile. The real-copy operand check must fail.
            row, column = leaf.proof.origin
            body = ast.parse(
                "\n".join(
                    [
                        f"leaf_source = cute.local_tile(leaf_tensor, (16, 64), (cutlass.Int32({row}) // 16, cutlass.Int32({column}) // 64))",
                        *(ast.unparse(node) for node in warp.body[2:]),
                    ]
                )
            )
        exec(compile(body, "<PreparationLeaf.emit>", "exec"), namespace)
        func.ReturnOp([cutlass.Int32(0).ir_value()] * (2 * transactions))
    assert module.operation.verify()
    passmanager.PassManager.parse(
        "builtin.module(cute-desugar,cute-fold-static,cute-expand-ops,"
        "canonicalize,cute-fold-static,cute-desugar,canonicalize)"
    ).run(module.operation)
    operations = list(_walk(module.operation))
    copies = [
        op for op in operations if op.name == "cute_nvgpu.arch.copy.SM100.tma_load"
    ]
    assert len(copies) == transactions
    # Export the actual architecture-copy coordinates, not independently
    # reconstructed offsets. Erasing the copy only after capturing its SSA
    # operands lets dead atom state disappear without kernel-argument ABI
    # lowering; the remaining coordinate arithmetic lowers entirely on CPU.
    ret = next(op for op in operations if op.name == "func.return")
    for transaction, copy in enumerate(copies):
        assert len(copy.operands) == 6  # descriptor, SMEM, barrier, x, y, cache policy
        # Do not iterate all operands: CuTe's Python value caster cannot construct
        # a typed _Pointer for the architecture op's untyped descriptor address.
        assert [str(copy.operands[index].type) for index in (3, 4)] == ["i32"] * 2
        assert "mode = <tiled> num_cta = 1" in str(copy)
        ret.operands[transaction * 2] = copy.operands[3]
        ret.operands[transaction * 2 + 1] = copy.operands[4]
        copy.erase()
    passmanager.PassManager.parse(
        "builtin.module(canonicalize,convert-cute-to-core,canonicalize)"
    ).run(module.operation)
    assert module.operation.verify()
    return module, function


def _evaluate_coordinates(module, function, start, head):
    operations = list(_walk(module.operation))
    values = dict(zip(function.entry_block.arguments, (start, head), strict=True))
    definitions = {result: op for op in operations for result in op.results}

    def evaluate(value):
        if value in values:
            return values[value]
        op = definitions[value]
        args = [evaluate(operand) for operand in op.operands]
        name = op.name
        if name == "arith.constant":
            result = int(ir.IntegerAttr(op.attributes["value"]).value)
        elif name in ("arith.extsi", "arith.trunci"):
            result = args[0]
        elif name in ("arith.addi", "arith.subi", "arith.muli"):
            result = {
                "arith.addi": operator.add,
                "arith.subi": operator.sub,
                "arith.muli": operator.mul,
            }[name](*args)
        elif name in ("arith.divsi", "arith.remsi"):
            quotient = abs(args[0]) // abs(args[1])
            if (args[0] < 0) != (args[1] < 0):
                quotient = -quotient
            result = quotient if name == "arith.divsi" else args[0] - quotient * args[1]
        elif name == "arith.floordivsi":
            result = args[0] // args[1]
        elif name == "arith.select":
            result = args[1] if args[0] else args[2]
        elif name == "arith.cmpi":
            predicate = int(ir.IntegerAttr(op.attributes["predicate"]).value)
            result = (
                operator.eq,
                operator.ne,
                operator.lt,
                operator.le,
                operator.gt,
                operator.ge,
            )[predicate](*args)
        elif name in ("arith.andi", "arith.ori", "arith.xori"):
            result = {
                "arith.andi": operator.and_,
                "arith.ori": operator.or_,
                "arith.xori": operator.xor,
            }[name](*args)
        else:
            pytest.fail(f"unmodeled coordinate operation: {op}")
        bits = int(str(value.type).removeprefix("i"))
        result = (
            int(result)
            if bits == 1
            else (result + 2 ** (bits - 1)) % 2**bits - 2 ** (bits - 1)
        )
        values[value] = result
        return result

    ret = next(op for op in operations if op.name == "func.return")
    return tuple(evaluate(value) for value in ret.operands)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_actual_tma_copy_preserves_dynamic_element_origins(dtype):
    with ir.Context(), ir.Location.unknown():
        module, function = _copy_coordinate_probe(_leaf(dtype))
        for start in (1, 17, 97):
            for head in (0, 1, 7):
                # TMA orders the contiguous dimension first. Both origin
                # components are intentionally non-tile-aligned.
                coordinates = _evaluate_coordinates(module, function, start, head)
                width = min(64, 128 // dtype.itemsize)
                assert coordinates == tuple(
                    value
                    for column in range(0, 64, width)
                    for value in (head * 128 + 8 + column, start)
                )
                addresses = {
                    (coordinates[transaction + 1] + row) * 1024
                    + coordinates[transaction]
                    + column
                    for transaction in range(0, len(coordinates), 2)
                    for row in range(16)
                    for column in range(width)
                }
                assert addresses == {
                    (start + row) * 1024 + head * 128 + 8 + column
                    for row in range(16)
                    for column in range(64)
                }


def test_actual_copy_probe_detects_grid_truncation():
    with ir.Context(), ir.Location.unknown():
        module, function = _copy_coordinate_probe(
            _leaf(torch.bfloat16), grid_truncation=True
        )
        assert _evaluate_coordinates(module, function, 17, 1) == (128, 16)
        assert _evaluate_coordinates(module, function, 17, 1) != (136, 17)


def test_leaf_copy_is_whole_warp_and_publication_wait_is_unconditional():
    source = _source(_leaf(torch.bfloat16))
    guard = source.body[0]
    assert isinstance(guard, ast.If)
    assert [
        ast.unparse(node.test) for node in guard.body if isinstance(node, ast.If)
    ] == ["prep_thread == 0", "prep_warp == 0"]
    warp = next(
        node
        for node in guard.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "prep_warp == 0"
    )
    assert len(warp.body) == 4
    assert not any(isinstance(node, ast.If) for node in ast.walk(warp.body[-1]))
    assert ast.unparse(warp.body[-1]).startswith("cute.copy(")
    arrival = next(
        node
        for node in guard.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "prep_thread == 0"
    )
    # Reused frame bytes can have prior generic writers. Every preparation
    # participant crosses the proxy fence and role barrier before leader issue.
    assert [ast.unparse(node) for node in guard.body[:2]] == [
        "cute.arch.fence_view_async_shared()",
        "prep_sync()",
    ]
    assert guard.body.index(arrival) >= 2
    assert guard.body.index(warp) > guard.body.index(arrival)
    assert ast.unparse(arrival.body[0]) == (
        "cute.arch.mbarrier_arrive_and_expect_tx(barrier, 2048)"
    )
    assert [ast.unparse(node) for node in source.body[1:]] == [
        "cute.arch.mbarrier_wait(barrier, phase)",
        "prep_sync()",
    ]
    assert ast.unparse(guard.orelse[-2]) == "prep_sync()"
    assert ast.unparse(guard.orelse[-1]) == (
        "if prep_thread == 0:\n    cute.arch.mbarrier_arrive(barrier)"
    )
