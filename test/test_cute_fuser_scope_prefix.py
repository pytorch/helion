from __future__ import annotations

import ast
from textwrap import indent

import pytest

from helion._compiler.cute.fuse_two_pass_loads import _CuteFuseTwoPassLoads
from helion._compiler.cute.fuse_two_pass_loads import fuse_two_pass_loads


@pytest.mark.parametrize("mode", ("auto", "register", "gmem"))
@pytest.mark.parametrize("nested", (False, True))
def test_unmatched_sweeps_do_not_rebuild_enclosing_binding(
    monkeypatch: pytest.MonkeyPatch, mode: str, nested: bool
) -> None:
    lines = ["base = input_offset"]
    for sweep in range(12):
        lines.extend(
            (
                "for synthetic_lane_7 in range(8):",
                f"    address = base + synthetic_lane_7 + {32 * sweep}",
                f"    value_{sweep} = (x.iterator + address).load()",
            )
        )
    source = "\n".join(lines)
    if nested:
        source = "if active:\n" + indent(source, "    ")
    freezes = 0
    original = _CuteFuseTwoPassLoads._freeze_binding

    def freeze(
        self: _CuteFuseTwoPassLoads,
        stmt: ast.stmt,
        value: ast.expr,
        previous: dict[str, ast.expr],
    ) -> ast.expr:
        nonlocal freezes
        if (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "base"
        ):
            freezes += 1
        return original(self, stmt, value, previous)

    monkeypatch.setattr(_CuteFuseTwoPassLoads, "_freeze_binding", freeze)
    result = fuse_two_pass_loads(
        ast.parse(source).body,
        tensor_dtypes={"x": "cutlass.Float32"},
        reload_modes={7: mode},
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert code.count(".load()") == 12
    assert "cute.make_rmem_tensor" not in code
    # One enclosing scan for child traversal and one for matching sweeps,
    # regardless of the number of unmatched suffix groups.
    assert freezes <= 2


@pytest.mark.parametrize("mode", ("register", "smem"))
def test_later_group_uses_rewritten_prefix(mode: str) -> None:
    source = """
for synthetic_lane_7 in range(8):
    a = (x.iterator + synthetic_lane_7).load()
for synthetic_lane_7 in range(8):
    b = (x.iterator + synthetic_lane_7).load()
    c = (y.iterator + synthetic_lane_7).load()
for synthetic_lane_7 in range(8):
    d = (y.iterator + synthetic_lane_7).load()
"""
    result = fuse_two_pass_loads(
        ast.parse(source).body,
        tensor_dtypes={"x": "cutlass.Float32", "y": "cutlass.Float32"},
        reload_modes={7: mode},
        proven_disjoint_tensor_pairs={frozenset(("x", "y"))},
    )
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))
    assert code.count(".load()") == 2
    if mode == "register":
        assert code.count("cute.make_rmem_tensor") == 2
    else:
        assert code.count("cute.arch.sync_threads()") == 2


def test_child_rewrite_is_visible_to_later_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {}

    def rewrite(
        self: _CuteFuseTwoPassLoads,
        body: list[ast.stmt],
        *,
        allow_smem: bool = True,
        inherited: dict[str, ast.expr] | None = None,
    ) -> list[ast.stmt] | None:
        if len(body) == 1 and isinstance(body[0], ast.Expr):
            marker = ast.unparse(body[0].value)
            if marker == "first_marker()":
                return [*body, *ast.parse("saved = 99").body]
            if marker == "second_marker()":
                assert inherited is not None
                observed.update(inherited)
        return None

    monkeypatch.setattr(_CuteFuseTwoPassLoads, "_try_fuse", rewrite)
    fuse_two_pass_loads(
        ast.parse(
            "saved = 1\nif first:\n    first_marker()\nif second:\n    second_marker()"
        ).body
    )
    # A definition inserted into a conditional predecessor cannot replace
    # the entering value with either constant at the later sibling.
    assert isinstance(observed["saved"], ast.Name)
    assert observed["saved"].id.startswith("_fuser_binding_")
