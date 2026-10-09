from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from test.cute_register_model import gather_registers
from test.test_cute_register_tensor import gather
from test.test_cute_register_tensor import graph

from helion._compiler.cute.register_tensor import emit_register_tensor
from helion._compiler.cute.register_tensor import plan_register_tensor
from helion._compiler.cute.row_fragment import RowFragment
from helion._compiler.cute.row_fragment import RowFragmentLayout
from helion._compiler.cute.row_fragment import emit_replicated_register_gather


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.bool_])
@pytest.mark.parametrize(
    "extent,capacity,lanes,width,owners",
    [
        (16, 16, 2, 8, None),
        (16, 16, 2, 1, None),
        (23, 32, 2, 4, None),
        (3, 16, 8, 1, None),
        (3, 16, 8, 4, None),
        (17, 32, 4, 2, (2, 0, 3, 1)),
        (1, 4, 32, 1, tuple(reversed(range(32)))),
        (9, 16, 1, 4, None),
    ],
)
def test_replicated_gather_ownership_and_padding(
    dtype, extent, capacity, lanes, width, owners
):
    layout = RowFragmentLayout(lanes, width, "logical_lane", owners)
    mapping = layout.replicated_gather_map(extent, capacity)
    count = 32 * capacity
    values = (np.arange(count) + 1).astype(dtype).reshape(32, capacity)
    if dtype == np.int64:
        values += 2**40
    elif dtype == np.bool_:
        values = np.arange(count).reshape(32, capacity) % 3 == 1
    elif dtype == np.float32:
        # Include signed zero and a noncanonical NaN payload; preserve raw bits.
        bits = np.array([0, 0x80000000, 0x7FC12345, 0x3F800000], dtype=np.uint32)
        values = np.resize(bits, count).view(np.float32).reshape(32, capacity)
    original = values.copy()
    owners = tuple(range(lanes)) if owners is None else owners
    for thread in range(32):
        logical_lane = owners.index(thread % lanes)
        register = np.arange(layout.num_registers(extent))
        column = (register // width * lanes + logical_lane) * width + register % width
        expected = np.zeros(register.size, dtype=dtype)
        expected[column < extent] = values[thread, column[column < extent]]
        actual = gather_registers(values[thread], mapping, thread)
        np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))
        assert not np.shares_memory(actual, values)
    np.testing.assert_array_equal(values.view(np.uint8), original.view(np.uint8))


def test_replicated_gather_emission_preserves_plan_import_and_names():
    statements, modules, names = [], [], []

    def new_var(prefix):
        name = prefix + str(len(names))
        names.append(name)
        return name

    cg: Any = SimpleNamespace(
        module_statements=modules,
        add_statement=statements.append,
        device_function=SimpleNamespace(new_var=new_var),
    )
    layout = RowFragmentLayout(4, 2, "logical_lane", (2, 0, 3, 1))
    arguments = {
        "extent": 13,
        "source_registers": 16,
        "layout": layout,
        "physical_lane": "physical_lane",
    }
    first = emit_replicated_register_gather(cg, "first", **arguments)
    second = emit_replicated_register_gather(cg, "second", **arguments)
    assert first.endswith(", physical_lane)") and "logical_lane" not in first
    assert first.replace("first", "second") == second
    assert len(modules) == 2
    fragment = RowFragment("x", torch.int64, 32, RowFragmentLayout(4, 8, "lane"))
    plan = plan_register_tensor(
        graph(lambda value: gather(value)[:, :2]), [fragment], lanes=4
    )
    assert plan is not None
    emit_register_tensor(cg, plan, [fragment], lane_expr="lane")
    imported = {
        alias.name
        for statement in modules
        if isinstance(statement, ast.ImportFrom)
        for alias in statement.names
    }
    assert {"_cute_gather_registers", "_cute_execute_register_plan"} <= imported
    assert sum(isinstance(statement, ast.Assign) for statement in modules) == 2


@pytest.mark.parametrize("dtype_name", ["Int32", "Int64", "Float32", "Boolean"])
def test_sdk_local_gather_has_unconditional_materialization(dtype_name, tmp_path):
    cutlass = pytest.importorskip("cutlass")
    from cutlass._mlir.dialects import func
    import cutlass.cute as cute

    from helion.runtime.cute.register_tensor import _cute_gather_registers

    ir: Any = pytest.importorskip("cutlass._mlir.ir")
    pass_manager: Any = pytest.importorskip("cutlass._mlir.passmanager")
    dtype = getattr(cutlass, dtype_name)
    layout = RowFragmentLayout(2, 1, "lane")
    mapping = layout.replicated_gather_map(13, 16)
    initialized = torch.cuda.is_initialized()
    with ir.Context(), ir.Location.unknown():
        emitted = ir.Module.create()
        with ir.InsertionPoint(emitted.body):
            function = func.FuncOp(
                "entry",
                (
                    [ir.VectorType.get([16], dtype.mlir_type), cutlass.Int32.mlir_type],
                    [ir.VectorType.get([7], dtype.mlir_type)],
                ),
            )
            block = function.add_entry_block()
            with ir.InsertionPoint(block):
                values = cute.TensorSSA(block.arguments[0], (16,), dtype)
                buffer = cute.make_rmem_tensor((16,), dtype)
                buffer.store(values)
                output = _cute_gather_registers(
                    buffer, mapping, cutlass.Int32(block.arguments[1])
                )
                func.ReturnOp([output.load().ir_value()])
        pass_manager.PassManager.parse("builtin.module(canonicalize,cse)").run(
            emitted.operation
        )
        assert emitted.operation.verify()
        source = str(emitted)
        (tmp_path / "register_gather.mlir").write_text(source)
        assert "arith.select" in source
        assert "scf.if" not in source
        assert "nvvm.shfl" not in source
        assert source.count("cute.memref.store_vec") == 2
    assert torch.cuda.is_initialized() == initialized
