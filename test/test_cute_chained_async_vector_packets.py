from __future__ import annotations

import ast
import importlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from .test_cute_chained_async_vector_contract import _capture
from .test_cute_chained_async_vector_contract import _Pointer
from helion._compiler.cute.chained_vector_leaf import emit_vector_leaf
from helion._compiler.cute.chained_vector_leaf import prove_vector_leaf


def _emission(dtype: torch.dtype, *, shared: bool = True):
    plan = prove_vector_leaf(
        ("row", "base + element"),
        {},
        element="element",
        uniform_names={"row", "base", "valid"},
        shape=(19, 127),
        strides=(127, 1),
        dtype=dtype,
        mask="row < valid",
    )
    assert plan is not None
    emission = emit_vector_leaf(
        plan,
        tensor="tensor",
        prefix="leaf",
        pointer_for_indices=lambda indices: (
            f"tensor.iterator + {indices[0]} * 127 + {indices[1]}"
        ),
        scalar_for_element=lambda element: ([], f"original_scalar({element})"),
        shared_pointer="shared.iterator + row * 128 + base" if shared else None,
    )
    return plan, emission, ast.parse("\n".join(emission.lines))


def _shape(tree: ast.AST, target: str) -> tuple[int, ...]:
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(name, ast.Name) and name.id == target for name in node.targets
        )
    ]
    assert len(nodes) == 1
    call = nodes[0].value
    assert isinstance(call, ast.Call) and ast.unparse(call.func) == "cute.make_tensor"
    layout = call.args[1]
    assert isinstance(layout, ast.Call)
    return ast.literal_eval(layout.args[0])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_async_packet_profile_matches_actual_native_atom(dtype: torch.dtype) -> None:
    import cutlass
    from cutlass._mlir.dialects import func
    import cutlass.cute as cute

    ir = importlib.import_module("cutlass._mlir.ir")
    plan, _, tree = _emission(dtype)
    shape = _shape(tree, "leaf_source")
    assert _shape(tree, "leaf_sink") == shape
    native_dtype = {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: cutlass.Float32,
    }[dtype]
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp("actual_emitted_packet_profile", ([], []))
        with ir.InsertionPoint(function.add_entry_block()):
            atom = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(), native_dtype, num_bits_per_copy=128
            )
            atom_values = cute.size(atom.layout_src_tv, mode=[1])
            assert shape[0] == atom_values == 16 // plan.element_bytes
            layout = cute.make_layout(shape)
            count = shape[1] if len(shape) == 2 else 1
            coordinates = [
                cute.crd2idx((value, packet) if len(shape) == 2 else value, layout)
                for packet in range(count)
                for value in range(atom_values)
            ]
            assert coordinates == list(range(8))
            assert count * atom_values * plan.element_bytes == 8 * plan.element_bytes
            source = cute.make_tensor(
                cute.make_ptr(
                    native_dtype, 0x100000, cute.AddressSpace.gmem, assumed_align=16
                ),
                layout,
            )
            sink = cute.make_tensor(
                cute.make_ptr(
                    native_dtype, 16384, cute.AddressSpace.smem, assumed_align=16
                ),
                layout,
            )
            cute.copy(atom, source, sink)
            func.ReturnOp([])
        assert module.operation.verify()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_ordinary_leaf_retains_original_flat_register_copy(dtype: torch.dtype) -> None:
    _, emission, tree = _emission(dtype, shared=False)
    assert _shape(tree, "leaf_source") == (8,)
    source = "\n".join(emission.lines)
    assert "CopyUniversalOp()" in source and "CopyG2SOp" not in source
    assert "cute.copy(leaf_copy, leaf_source, leaf_values)" in source


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("valid", [0, 13, 19])
def test_two_row_iterations_keep_complete_exclusive_packet_domains(
    dtype: torch.dtype, valid: int
) -> None:
    plan, _, tree = _emission(dtype)
    shape = _shape(tree, "leaf_source")
    packet_values = shape[0]
    packets = shape[1] if len(shape) == 2 else 1
    outer = next(node for node in tree.body if isinstance(node, ast.If))
    inner = next(node for node in outer.body if isinstance(node, ast.If))
    assignments: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, ast.Assign)
    ]
    assignments = assignments[2:]
    destinations: set[tuple[int, int]] = set()
    fast_rows: set[int] = set()
    for step in range(2):
        for thread in range(256):
            row, base = thread // 16 + step * 16, thread % 16 * 8
            env: dict[str, Any] = {
                "cutlass": SimpleNamespace(Int64=int),
                "row": row,
                "base": base,
                "valid": valid,
                "tensor": SimpleNamespace(
                    shape=(19, 127),
                    layout=SimpleNamespace(stride=(127, 1)),
                    iterator=_Pointer(0x100000, plan.element_bytes),
                ),
                "shared": SimpleNamespace(iterator=_Pointer(16384, plan.element_bytes)),
            }
            exec(compile(ast.Module(assignments, []), "indices", "exec"), env)
            fast = eval(compile(ast.Expression(outer.test), "bounds", "eval"), env)
            if fast:
                setup = outer.body[: outer.body.index(inner)]
                exec(compile(ast.Module(setup, []), "pointers", "exec"), env)
                fast = eval(
                    compile(ast.Expression(inner.test), "alignment", "eval"), env
                )
            offsets = list(range(8))
            if fast:
                fast_rows.add(row)
                offsets = [
                    p * packet_values + v
                    for p in range(packets)
                    for v in range(packet_values)
                ]
                assert row < valid and base + 7 < 127
                for p in range(packets):
                    source = env["leaf_pointer"] + p * packet_values
                    sink = env["leaf_shared_pointer"] + p * packet_values
                    assert source.toint() % 16 == sink.toint() % 16 == 0
                    assert (
                        source.toint() + 16
                        <= env["leaf_pointer"].toint() + 8 * plan.element_bytes
                    )
            assert offsets == list(range(8))
            for offset in offsets:
                coordinate = (row, base + offset)
                assert coordinate not in destinations
                destinations.add(coordinate)
    assert destinations == {(row, column) for row in range(32) for column in range(128)}
    assert bool(fast_rows) is bool(valid)
    assert (16 in fast_rows) is (valid == 19)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_original_public_client_has_live_second_row_and_original_join(
    dtype: torch.dtype,
) -> None:
    source, observations = _capture(
        dtype, warps=16, consumer_warps=8, columns=127, steps=3, valid=19
    )
    selected = [
        text
        for actual, text, sink in observations
        if actual == dtype and "CopyG2SOp" in text
    ]
    assert selected and "cute.arch.cp_async_wait_group(0)" in source
    for text in selected:
        assert "range(2, unroll=1)" in text
        assert "cp_async_commit_group()" in text
        assert text.endswith(
            "cute.arch.cp_async_wait_group(0)\nchain_prep_barrier.arrive_and_wait()"
        )
        assert (
            "cute.make_layout((4, 2))" in text
            if dtype == torch.float32
            else "cute.make_layout((8,))" in text
        )
