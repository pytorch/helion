from __future__ import annotations

import ast

from helion._compiler.cute.proven_loop_bounds import simplify_proven_loop_bounds


def rewrite(source: str) -> str:
    body = simplify_proven_loop_bounds(
        ast.parse(source).body,
        enabled=True,
        thread_block_dims=(128, 1, 1),
        block_grid_dims=(64, 1, 1),
        constexpr_values={},
    )
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def test_augassign_preserves_operation_and_unrelated_proof() -> None:
    source = """
acc = cutlass.Float32(0)
for column in cutlass.range_constexpr(8):
    acc += read(column)
    output[column] = value if 0 <= column + 8 < 64 else zero
"""
    result = rewrite(source)
    assert "acc += read(column)" in result
    assert "output[column] = value" in result
    assert "else zero" not in result


def test_augassign_forgets_updated_target() -> None:
    source = """
index = 0
index += dynamic
output[0] = value if 0 <= index < 8 else zero
"""
    result = rewrite(source)
    assert "index += dynamic" in result
    assert "0 <= index < 8" in result


def test_augassign_does_not_rewrite_rhs_or_effects() -> None:
    source = """
index = 0
index += read() if 0 <= index < 8 else another_read()
output[0] = value if 0 <= index < 8 else zero
"""
    assert rewrite(source) == ast.unparse(ast.parse(source))


def test_augassign_loop_carried_value_is_not_an_entry_constant() -> None:
    source = """
index = 0
for step in cutlass.range_constexpr(8):
    output[step] = value if 0 <= index < 8 else zero
    index += 100
output[0] = value if 0 <= index < 8 else zero
"""
    assert rewrite(source).count("0 <= index < 8") == 2


def test_augassign_keeps_immutable_integer_snapshot() -> None:
    source = """
index = 0
snapshot = index
index += dynamic
output[0] = value if 0 <= snapshot < 8 else zero
output[1] = value if 0 <= index < 8 else zero
"""
    result = rewrite(source)
    assert "output[0] = value" in result
    assert "output[1] = value if 0 <= index < 8 else zero" in result


def test_memory_augassign_still_rejects_pass() -> None:
    source = """
for column in cutlass.range_constexpr(8):
    accumulator[column] += value
    output[column] = value if 0 <= column < 8 else zero
"""
    assert rewrite(source) == ast.unparse(ast.parse(source))


def test_augassign_intrinsic_rebinding_rejects_pass() -> None:
    source = """
cutlass += value
output[0] = value if 0 <= cutlass.Int32(1) < 8 else zero
"""
    assert rewrite(source) == ast.unparse(ast.parse(source))


def test_chained_comparison_all_edges_required() -> None:
    source = """
for index in cutlass.range_constexpr(8):
    output[index] = value if 0 <= index < 7 else zero
"""
    assert "0 <= index < 7" in rewrite(source)


def test_chained_comparison_keeps_side_effecting_middle() -> None:
    source = "output[0] = value if 0 <= read() < 8 else zero"
    assert rewrite(source) == ast.unparse(ast.parse(source))


def test_chained_comparison_keeps_overflowing_middle() -> None:
    source = """
index = cutlass.Int32(2147483647)
output[0] = value if 0 <= index + 1 < 8 else zero
"""
    assert "0 <= index + 1 < 8" in rewrite(source)


def test_chained_comparison_keeps_unsupported_operator() -> None:
    source = """
index = 1
output[0] = value if 0 <= index != 3 else zero
"""
    assert "0 <= index != 3" in rewrite(source)
