from __future__ import annotations

import ast
import importlib
import inspect
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from helion._compiler.cute.chained_warp_mma import emit_warp_mma


def _helper_tree():
    module = importlib.import_module("helion._compiler.cute.prepared_warp_contraction")
    return ast.parse(inspect.getsource(module))


def _host_executor():
    """Execute the exact device schedule as a symbolic trace, not numeric MMA."""
    tree = _helper_tree()
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for node in functions:
        node.decorator_list = []
    namespace: dict[str, Any] = {
        "cutlass": SimpleNamespace(
            const_expr=lambda value: value, range_constexpr=range
        )
    }
    program = ast.Module(
        body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0), *functions],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(program), "<physical-schedule>", "exec"),
        namespace,
    )
    return namespace


def test_exactly_one_increasing_k_executor_and_closed_leaves():
    functions = {
        node.name: node
        for node in _helper_tree().body
        if isinstance(node, ast.FunctionDef)
    }
    executor = functions["execute_prepared_warp_k"]
    loops = [node for node in ast.walk(executor) if isinstance(node, ast.For)]
    assert len(loops) == 1
    assert ast.unparse(loops[0].iter) == "cutlass.range_constexpr(K_STEPS)"
    calls = []
    for node in loops[0].body:
        assert isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        calls.append(ast.unparse(node.value.func))
    assert calls == [
        "_load_a",
        "_load_b",
        "_issue",
    ]
    assert ast.unparse(executor.body[-1]) == "return accumulator"
    for name in ("_load_x4", "_load_a", "_load_b", "_issue"):
        assert not any(
            isinstance(node, (ast.For, ast.While)) for node in ast.walk(functions[name])
        )
        text = ast.unparse(functions[name])
        for forbidden in ("sync_threads", "wait_group", "store", "callback"):
            assert forbidden not in text


@pytest.mark.parametrize("steps", (1, 2, 4, 8, 16))
@pytest.mark.parametrize("atoms", (1, 2))
def test_original_accumulator_versions_and_low_before_high(steps, atoms):
    namespace = _host_executor()
    events: list[tuple[object, ...]] = []
    accumulator = tuple(("fp32-zero", slot) for slot in range(4 * atoms))

    def load_a(source, k, packed, transposed=False):
        assert transposed is False
        assert source == "a" and packed is True
        events.append(("a", k))
        return tuple(("a", k, word) for word in range(4))

    def load_b(source, k, packed, count, transposed=False):
        assert transposed is False
        assert source == "b" and packed is True and count == atoms
        events.append(("b", k))
        return tuple(("b", k, word) for word in range(2 * atoms))

    def issue(*values):
        k, panel = values[4][1], values[4][2] // 2
        assert values[:4] == tuple(("a", k, word) for word in range(4))
        assert values[4:6] == tuple(("b", k, 2 * panel + word) for word in range(2))
        expected = (
            tuple(("fp32-zero", 4 * panel + slot) for slot in range(4))
            if k == 0
            else tuple(("result", k - 1, panel, slot) for slot in range(4))
        )
        assert values[6:] == expected
        events.append(("mma", k, panel))
        return tuple(("result", k, panel, slot) for slot in range(4))

    namespace.update(_load_a=load_a, _load_b=load_b, mma_m16n8k16_bf16=issue)
    result = namespace["execute_prepared_warp_k"](
        "a", "b", accumulator, None, steps, True, atoms
    )
    assert events == [
        event
        for k in range(steps)
        for event in [
            ("a", k),
            ("b", k),
            *(("mma", k, panel) for panel in range(atoms)),
        ]
    ]
    assert result == tuple(
        ("result", steps - 1, panel, slot)
        for panel in range(atoms)
        for slot in range(4)
    )


@pytest.mark.parametrize(
    "steps,packed,atoms", ((0, True, 1), (-1, True, 1), (8, False, 2), (8, True, 3))
)
def test_unsupported_physical_schedule_fails_before_load(steps, packed, atoms):
    namespace = _host_executor()
    with pytest.raises(AssertionError):
        namespace["execute_prepared_warp_k"](
            None, None, None, None, steps, packed, atoms
        )


@pytest.mark.parametrize("threads", (32, 64, 128, 256))
@pytest.mark.parametrize("axes", ({"a": 1, "b": 1}, {"a": 0, "b": 1}, {"a": 1, "b": 0}))
def test_common_one_atom_geometry_uses_original_setup_and_shared_executor(
    threads, axes
):
    lines = emit_warp_mma(
        "test",
        "cutlass.BFloat16",
        (16, threads // 4, 128),
        threads,
        axes,
        ["test_acc.fill(0.0)"],
    )
    assert sum("execute_prepared_warp_k(" in line for line in lines) == 1
    assert not any(line.startswith("for ") for line in lines)
    assert lines[-3] == "test_acc.fill(0.0)"
    assert "test_thr.partition_A(test_a)" in "\n".join(lines)
    assert f"transpose={axes['b'] == 0}, num_matrices=2" in "\n".join(lines)


@pytest.mark.parametrize(
    "dtype,shape,seed",
    (
        ("cutlass.Float16", (16, 16, 128), ["test_acc.fill(0.0)"]),
        ("cutlass.BFloat16", (32, 16, 128), ["test_acc.fill(0.0)"]),
        ("cutlass.BFloat16", (16, 32, 128), ["test_acc.fill(0.0)"]),
        ("cutlass.BFloat16", (16, 16, 128), ["test_acc.fill(1.0)"]),
    ),
)
def test_other_original_schedules_retain_the_exact_k_loop(dtype, shape, seed):
    lines = emit_warp_mma("test", dtype, shape, 64, {"a": 1, "b": 1}, seed)
    assert not any("execute_prepared_warp_k" in line for line in lines)
    assert lines[-4:] == [
        "for test_kk in cutlass.range_constexpr(cute.size(test_sa, mode=[2])):",
        "    cute.copy(test_copy_a, test_copy_sa[None, None, test_kk], test_copy_ra[None, None, 0])",
        "    cute.copy(test_copy_b, test_copy_sb[None, None, test_kk], test_copy_rb[None, None, 0])",
        "    cute.gemm(test_mma, test_acc, test_ra[None, None, 0], test_rb[None, None, 0], test_acc)",
    ]


@pytest.mark.parametrize("warps", (2, 8))
def test_actual_cute_atom_slots_cover_cb_full_tile_and_ordered_k(warps):
    import cutlass
    import cutlass.cute as cute

    ir = importlib.import_module("cutlass._mlir.ir")
    initialized = torch.cuda.is_initialized()
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma = cute.make_tiled_mma(
                cute.make_mma_atom(
                    cute.nvgpu.warp.MmaF16BF16Op(
                        cutlass.BFloat16, cutlass.Float32, (16, 8, 16)
                    )
                ),
                atom_layout_mnk=(1, warps, 1),
            )
            result = set()
            for thread in range(32 * warps):
                owner = mma.get_slice(thread)
                a = owner.partition_A(cute.make_identity_tensor((16, 128)))
                b = owner.partition_B(cute.make_identity_tensor((8 * warps, 128)))
                c = owner.partition_C(cute.make_identity_tensor((16, 8 * warps)))
                assert cute.size(a, mode=[1]) == cute.size(b, mode=[1]) == 1
                assert cute.size(a, mode=[2]) == cute.size(b, mode=[2]) == 8
                assert cute.size(c) == 4
                for slot in range(4):
                    row, col = map(int, c[slot])
                    result.add((row, col))
                    assert col // 8 == thread // 32
                for k in range(8):
                    for value in (a[None, None, k], b[None, None, k]):
                        assert all(
                            int(value[index][1]) // 16 == k
                            for index in range(cute.size(value))
                        )
                # Preserve actual b16 packed-word order, rather than narrowing C.
                registers = mma.make_fragment_A((cute.shape(a)[0], 1, 1))
                words = cute.recast_tensor(registers, cutlass.Int32)
                assert cute.size(registers) == 8 and cute.size(words) == 4
                assert [int(registers.layout(i)) for i in range(8)] == list(range(8))
                assert [int(words.layout(i)) for i in range(4)] == list(range(4))
            assert result == {
                (row, col) for row in range(16) for col in range(8 * warps)
            }
        assert module.operation.verify()
    assert torch.cuda.is_initialized() is initialized


def test_direct_adapters_retain_history_after_complete_shared_projection():
    module = importlib.import_module("helion._compiler.cute.short_affine_scan_mma")
    tree = ast.parse(inspect.getsource(module))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    for name in ("_project_m16n8_bf16", "project_retain_affine_m16n16_bf16"):
        assert not any(isinstance(node, ast.For) for node in ast.walk(functions[name]))
        assert ast.unparse(functions[name]).count("execute_prepared_warp_k(") == 1
    calls = [
        ast.unparse(node.value.func)
        for node in functions["project_retain_affine_m16n16_bf16"].body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
    ]
    assert calls == ["execute_prepared_warp_k", "_retain_state_history_bf16"]
