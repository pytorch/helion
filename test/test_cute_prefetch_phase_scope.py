from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute.chained_warp_loop_prefetch import _copy_scope

COPY = (
    "pointer = (ring + slot * 2560).align(16)",
    "tensor = cute.make_tensor(pointer, layout)",
    "copied = cutlass.Boolean(False)",
    "if address % 16 == 0:",
    "    source = cute.make_tensor(global_pointer, layout)",
    "    cute.copy(atom, source, tensor)",
    "    copied = cutlass.Boolean(True)",
    "if not copied:",
    "    for lane in cutlass.range_constexpr(8):",
    "        value = ((lane * 0.5) + 1.0) * scale",
    "        tensor[lane] = value",
    "cute.arch.cp_async_commit_group()",
)


def _names(lines, context):
    return {
        node.id
        for node in ast.walk(ast.parse("\n".join(lines)))
        if isinstance(node, ast.Name) and isinstance(node.ctx, context)
    }


@pytest.mark.parametrize("phase", ["warm", "refill"])
def test_copy_scope_changes_only_owned_name_spans(phase):
    original = (*COPY, 'label = "pointer"  # café: pointer', "é = value")
    scoped = _copy_scope(original, phase)
    local = _names(original, ast.Store)
    expected = {name: f"chain_prefetch_{phase}_{name}" for name in local}
    assert _names(scoped, ast.Store) == set(expected.values())
    assert _names(original, ast.Load) - local == _names(scoped, ast.Load) - set(
        expected.values()
    )
    assert scoped[-2] == f'chain_prefetch_{phase}_label = "pointer"  # café: pointer'

    class Inverse(ast.NodeTransformer):
        def visit_Name(self, node):
            inverse = {value: key for key, value in expected.items()}
            if node.id in inverse:
                node.id = inverse[node.id]
            return node

    changed = Inverse().visit(ast.parse("\n".join(scoped)))
    assert ast.dump(changed) == ast.dump(ast.parse("\n".join(original)))


def test_copy_scope_rejects_collision_and_lexical_index_writes():
    with pytest.raises(chain._UnsupportedChain, match="collision"):
        _copy_scope(["value = chain_prefetch_warm_value"], "warm")
    with pytest.raises(chain._UnsupportedChain, match="writes lexical index"):
        _copy_scope(["chain_loop_index = 1"], "refill")


def test_refill_binding_reads_current_index_and_keeps_current_slot_external():
    assert _copy_scope(
        [
            "chain_prefetch_index = chain_loop_index + 3",
            "copy(chain_prefetch_index, chain_prefetch_slot)",
        ],
        "refill",
    ) == (
        "chain_prefetch_refill_chain_prefetch_index = chain_loop_index + 3",
        "copy(chain_prefetch_refill_chain_prefetch_index, chain_prefetch_slot)",
    )


def test_actual_sdk_refill_analysis_does_not_carry_warm_or_compute_locals():
    sdk = pytest.importorskip("cutlass.base_dsl.ast_preprocessor")

    def writes(lines, active):
        region = ast.If(
            test=ast.Name(id="refill", ctx=ast.Load()),
            body=ast.parse("\n".join(lines)).body,
            orelse=[],
        )
        names, count, closures = sdk.DSLPreprocessor.analyze_region_variables(
            SimpleNamespace(_has_pyir_support=lambda: False), region, [active], []
        )
        assert closures == []
        return names[:count]

    original_locals = _names(COPY, ast.Store)
    assert "pointer" in writes(COPY, original_locals)
    warm = _copy_scope(COPY, "warm")
    refill = _copy_scope(COPY, "refill")
    assert writes(refill, _names(warm, ast.Store) | original_locals) == []
    assert not (_names(warm, ast.Store) & _names(refill, ast.Store))


@pytest.mark.parametrize("dtype", ["BFloat16", "Float16"])
def test_actual_sdk_constant_and_dynamic_pointer_container_structure(dtype):
    cutlass = pytest.importorskip("cutlass")
    cute = pytest.importorskip("cutlass.cute")
    ir = pytest.importorskip("cutlass._mlir.ir")
    func = pytest.importorskip("cutlass._mlir.dialects.func")
    sdk = pytest.importorskip("cutlass.cutlass_dsl.cutlass")
    tree = pytest.importorskip("cutlass.base_dsl.utils.tree_utils")
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(
                "phase_types", ([ir.IntegerType.get_signless(32)], [])
            )
        with ir.InsertionPoint(function.add_entry_block()):
            slot = cutlass.Int32(function.arguments[0])
            ring = cute.make_ptr(
                getattr(cutlass, dtype),
                16384,
                cute.AddressSpace.smem,
                assumed_align=128,
            )
            warm = (ring + 2 * 2560).align(16)
            refill = (ring + slot * 2560).align(16)
            assert warm.alignment == 128 and refill.alignment == 16
            _, before = sdk.unpack_to_irvalue([warm], "if", 1, ["pointer"])
            _, after = sdk.unpack_to_irvalue([refill], "if", 1, ["pointer"])
            assert tree.check_tree_equal(before, after) == 0
            # Distinct local names are absent from the actual SDK write-carry
            # list, so no 128B-to-16B pointer is incorrectly threaded through it.
            _, empty_before = sdk.unpack_to_irvalue([], "if", 0, [])
            _, empty_after = sdk.unpack_to_irvalue([], "if", 0, [])
            assert tree.check_tree_equal(empty_before, empty_after) == -1
            func.ReturnOp([])
        assert module.operation.verify()
