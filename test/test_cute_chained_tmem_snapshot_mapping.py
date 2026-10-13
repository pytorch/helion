from __future__ import annotations

import ast
import importlib
from typing import Any
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_tmem_snapshots import _emit


class _StaticRanges(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cutlass"
            and node.func.attr == "range_constexpr"
        ):
            node.func = ast.Name(id="range", ctx=ast.Load())
        return node


def _code(statements):
    tree = _StaticRanges().visit(ast.Module(body=statements, type_ignores=[]))
    return compile(ast.fix_missing_locations(tree), "<snapshot emitter>", "exec")


def _native_module(dtype_name, width):
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05
    from cutlass.utils import blackwell_helpers

    ir = importlib.import_module("cutlass._mlir.ir")
    func = importlib.import_module("cutlass._mlir.dialects.func")
    scf = importlib.import_module("cutlass._mlir.dialects.scf")
    dtype = {"BFloat16": cutlass.BFloat16, "Float16": cutlass.Float16}[dtype_name]
    source = "\n".join(_emit((128, width), f"cutlass.{dtype_name}"))
    tree = ast.parse(source)
    index = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.For))
    loop = tree.body[index]
    assert isinstance(loop, ast.For)
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        function = func.FuncOp("snapshot", ([ir.IntegerType.get_signless(32)] * 3, []))
    with ir.InsertionPoint(function.add_entry_block()):
        thread, unused_panel, origin = map(cutlass.Int32, function.arguments)
        mma = blackwell_helpers.make_trivial_tiled_mma(
            dtype,
            dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (128, width),
            tcgen05.OperandSource.SMEM,
        )
        layout = mma.make_fragment_C(mma.partition_shape_C((128, width))).layout
        base = cute.make_ptr(cutlass.Float32, origin, cute.AddressSpace.tmem)
        namespace: dict[str, Any] = {
            "cutlass": cutlass,
            "cute": cute,
            "tcgen05": tcgen05,
            "original_acc": cute.make_tensor(base + 32, layout),
            "original_slice": mma.get_slice(0),
            "chain_thread": thread,
            "chain_tptr": base,
        }
        exec(_code(tree.body[:index]), namespace)
        assert int(cute.size(namespace["original_values"])) == 32
        assert int(cute.size(namespace["snapshot_packed"])) == 16
        native_loop = scf.ForOp(
            *(cutlass.Int32(v).ir_value() for v in (0, width // 32, 1))
        )
        with ir.InsertionPoint(native_loop.body):
            namespace["snapshot_panel"] = cutlass.Int32(native_loop.induction_variable)
            exec(_code(loop.body), namespace)
            scf.YieldOp([])
        exec(_code(tree.body[index + 1 :]), namespace)
        func.ReturnOp([])
    assert module.operation.verify()
    return module


@pytest.mark.parametrize("dtype_name", ["BFloat16", "Float16"])
@pytest.mark.parametrize("width", [64, 128, 256])
def test_actual_dynamic_emitter_load_pack_store_cpu(dtype_name, width):
    ir = importlib.import_module("cutlass._mlir.ir")
    passmanager = importlib.import_module("cutlass._mlir.passmanager")
    before = torch.cuda.is_initialized()
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
        ir.Context(),
        ir.Location.unknown(),
    ):
        module = _native_module(dtype_name, width)
        passmanager.PassManager.parse(
            "builtin.module(cute-desugar,cute-fold-static,cute-expand-ops,convert-cute-to-core,canonicalize)"
        ).run(module.operation)
        assert module.operation.verify()
        function = next(iter(module.body.operations)).operation
        operations = [
            view.operation for view in function.regions[0].blocks[0].operations
        ]
        constants = {
            op.results[0]: op.attributes["value"].value
            for op in operations
            if op.name == "arith.constant"
        }
        assert [
            constants[op.operands[0]] for op in operations if op.name == "llvm.alloca"
        ] == [32, 16]
        effects = [
            op
            for op in operations
            if op.name
            in ("nvvm.barrier0", "nvvm.barrier", "scf.for", "nvvm.tcgen05.wait")
        ]
        assert [op.name for op in effects] in (
            ["nvvm.barrier0", "scf.for", "nvvm.tcgen05.wait", "nvvm.barrier0"],
            ["nvvm.barrier", "scf.for", "nvvm.tcgen05.wait", "nvvm.barrier"],
        )
        body = [view.operation for view in effects[1].regions[0].blocks[0].operations]
        assert not any(
            op.name in ("llvm.alloca", "nvvm.barrier0", "nvvm.barrier") for op in body
        )
        assert len([op for op in body if op.name == "nvvm.tcgen05.ld"]) == 1
        assert len([op for op in body if op.name == "nvvm.tcgen05.st"]) == 1
        assert len([op for op in body if op.name == "arith.truncf"]) == 32
        assert "<store>" in str(effects[2])
    assert torch.cuda.is_initialized() == before
