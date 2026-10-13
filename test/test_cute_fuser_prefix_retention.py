from __future__ import annotations

import ast

import pytest

from helion._compiler.cute.fuse_two_pass_loads import _CuteFuseTwoPassLoads
from helion._compiler.cute.fuse_two_pass_loads import fuse_two_pass_loads


@pytest.mark.parametrize("mode", ("register", "smem"))
def test_successive_scalar_groups_retain_strict_prefix(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    lines = ["base = input_offset"]
    for group in range(5):
        for sweep in range(2):
            lines.extend(
                (
                    f"for synthetic_lane_{group} in range(8):",
                    f"    address = base + synthetic_lane_{group}",
                    f"    v_{group}_{sweep} = (x.iterator + address).load()",
                )
            )
    freezes = 0
    original = _CuteFuseTwoPassLoads._freeze_binding

    def freeze(
        self: _CuteFuseTwoPassLoads,
        stmt: ast.stmt,
        value: ast.expr,
        previous: dict[str, ast.expr],
    ) -> ast.expr:
        nonlocal freezes
        if isinstance(stmt, ast.Assign) and ast.unparse(stmt.targets[0]) == "base":
            freezes += 1
        return original(self, stmt, value, previous)

    monkeypatch.setattr(_CuteFuseTwoPassLoads, "_freeze_binding", freeze)
    result = fuse_two_pass_loads(
        ast.parse("\n".join(lines)).body,
        tensor_dtypes={"x": "cutlass.Float32"},
        reload_modes=dict.fromkeys(range(5), mode),
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert code.count(".load()") == 5
    assert freezes == 2  # child traversal and first enclosing matching prefix


@pytest.mark.parametrize("mode", ("register", "smem"))
def test_changed_prefix_after_first_fusion_is_recomputed(mode: str) -> None:
    source = """
base = initial
for synthetic_lane_0 in range(8):
    first = (x.iterator + base + synthetic_lane_0).load()
for synthetic_lane_0 in range(8):
    second = (x.iterator + base + synthetic_lane_0).load()
for synthetic_lane_1 in range(8):
    address = base + synthetic_lane_1
    third = (x.iterator + address).load()
base = different
for synthetic_lane_1 in range(8):
    address = base + synthetic_lane_1
    fourth = (x.iterator + address).load()
"""
    result = fuse_two_pass_loads(
        ast.parse(source).body,
        tensor_dtypes={"x": "cutlass.Float32"},
        reload_modes={0: mode, 1: mode},
    )
    assert ast.unparse(ast.Module(body=result, type_ignores=[])).count(".load()") == 3


def test_vector_extract_rewrite_keeps_full_prefix_invalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = """
base = input_offset
for synthetic_lane_0 in range(8):
    packet0 = cute.arch.load(x.iterator + base + synthetic_lane_0 * 4, ir.VectorType.get([4], cutlass.Float32.mlir_type))
    first = packet0[0]
for synthetic_lane_0 in range(8):
    packet1 = cute.arch.load(x.iterator + base + synthetic_lane_0 * 4, ir.VectorType.get([4], cutlass.Float32.mlir_type))
    second = packet1[0]
for synthetic_lane_1 in range(8):
    third = (x.iterator + base + synthetic_lane_1).load()
for synthetic_lane_1 in range(8):
    fourth = (x.iterator + base + synthetic_lane_1).load()
"""
    freezes = 0
    original = _CuteFuseTwoPassLoads._freeze_binding

    def freeze(
        self: _CuteFuseTwoPassLoads,
        stmt: ast.stmt,
        value: ast.expr,
        previous: dict[str, ast.expr],
    ) -> ast.expr:
        nonlocal freezes
        if isinstance(stmt, ast.Assign) and ast.unparse(stmt.targets[0]) == "base":
            freezes += 1
        return original(self, stmt, value, previous)

    monkeypatch.setattr(_CuteFuseTwoPassLoads, "_freeze_binding", freeze)
    result = fuse_two_pass_loads(
        ast.parse(source).body,
        tensor_dtypes={"x": "cutlass.Float32"},
        reload_modes={0: "register", 1: "register"},
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert "packet1 = cute.arch.load" not in code
    assert "second = _fuse_cache_" in code
    assert freezes == 3
