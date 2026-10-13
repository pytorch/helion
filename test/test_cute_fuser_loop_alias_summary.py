from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import pytest

from helion._compiler.cute import fuse_two_pass_loads as fuser

if TYPE_CHECKING:
    from collections.abc import Iterable


def _loop(body: str) -> ast.For:
    return cast("ast.For", ast.parse(body).body[0])


def test_summary_preserves_breadth_first_and_assignment_target_order() -> None:
    loop = _loop(
        """
for synthetic_lane_7 in range(8):
    if active:
        reduction_lane_base_deep = 1
        for vec_lane_deep in range(2):
            pass
    reduction_lane_base_first = reduction_lane_base_second = 2
    reduction_vec_lane_first = reduction_vec_lane_second = 3
    lane_base_first = 4
    lane_base_later = 5
    for vec_lane_first in range(2):
        pass
    for vec_lane_later in range(2):
        pass
"""
    )
    assert fuser._loop_alias_names(loop) == (
        "reduction_lane_base_first",
        "reduction_vec_lane_first",
        "lane_base_first",
        "vec_lane_first",
    )


def test_summary_ignores_non_assignment_binding_forms() -> None:
    loop = _loop(
        """
for vec_lane_outer in range(8):
    reduction_lane_base_annotation: int = 1
    reduction_lane_base_augmented += 1
    (reduction_vec_lane_tuple, other) = values
    lane_base_subscript[0] = 2
    use(lane_base_read)
    if (lane_base_named := 3):
        pass
    for lane_base_induction in range(2):
        pass
"""
    )
    assert fuser._loop_alias_names(loop) == (None, None, None, "vec_lane_outer")


def test_direct_alias_calls_do_not_keep_summaries_after_mutation() -> None:
    first = _loop("for lane in range(8):\n    lane_base_first = 0")
    second = _loop("for lane in range(8):\n    lane_base_second = 0")
    engine = fuser._CuteFuseTwoPassLoads()
    assert engine._build_second_alias(first, second) == {
        "lane_base_second": "lane_base_first"
    }
    first.body.insert(0, ast.parse("lane_base_changed = 0").body[0])
    assert engine._build_second_alias(first, second) == {
        "lane_base_second": "lane_base_changed"
    }


def test_unmatched_suffixes_summarize_each_loop_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[ast.For] = []
    original = fuser._loop_alias_names

    def observe(loop: ast.For) -> fuser._LoopAliasNames:
        calls.append(loop)
        return original(loop)

    monkeypatch.setattr(fuser, "_loop_alias_names", observe)
    source = "\n".join(
        f"for synthetic_lane_7 in range(8):\n"
        f"    value_{i} = (x.iterator + synthetic_lane_7 + {i * 32}).load()"
        for i in range(12)
    )
    result = fuser.fuse_two_pass_loads(
        ast.parse(source).body,
        tensor_dtypes={"x": "cutlass.Float32"},
        reload_modes={7: "register"},
    )
    assert len(calls) == len(set(calls)) == 12
    assert ast.dump(ast.Module(body=result, type_ignores=[])) == ast.dump(
        ast.parse(source)
    )


@pytest.mark.parametrize("mode", ("register", "smem"))
def test_successful_rewrite_retires_all_loop_summaries(mode: str) -> None:
    # Mutate a summarized loop at the real end-of-rewrite hook. Later suffix
    # groups must see a fresh scope, including changes to first-match names.
    second_loops: list[ast.For] = []
    mutations: list[ast.For] = []
    checks = 0

    class Observe(fuser._CuteFuseTwoPassLoads):
        def _build_second_alias(
            self,
            first_loop: ast.For,
            second_loop: ast.For,
            *,
            summaries: dict[ast.For, fuser._LoopAliasNames] | None = None,
        ) -> dict[str, str]:
            nonlocal checks
            assert summaries is not None
            for loop, names in summaries.items():
                assert names == fuser._loop_alias_names(loop)
            checks += 1
            second_loops.append(second_loop)
            return super()._build_second_alias(
                first_loop, second_loop, summaries=summaries
            )

    class Declarations(list[ast.stmt]):
        def extend(self, values: Iterable[ast.stmt]) -> None:
            super().extend(values)
            if not mutations:
                loop = second_loops[0]
                loop.body.insert(0, ast.parse("lane_base_new = 17").body[0])
                mutations.append(loop)

    engine = Observe(
        tensor_dtypes=dict.fromkeys(("x", "y", "z"), "cutlass.Float32"),
        reload_modes={7: mode},
        thread_block_dims=(32, 2, 1),
        proven_disjoint_tensor_pairs={
            frozenset(pair) for pair in (("x", "y"), ("x", "z"), ("y", "z"))
        },
    )
    engine._root_declarations = Declarations()
    source = """
for synthetic_lane_7 in range(8):
    a = (x.iterator + synthetic_lane_7).load()
for synthetic_lane_7 in range(8):
    b = (x.iterator + synthetic_lane_7).load()
    c = (y.iterator + synthetic_lane_7).load()
for synthetic_lane_7 in range(8):
    d = (y.iterator + synthetic_lane_7).load()
    e = (z.iterator + synthetic_lane_7).load()
for synthetic_lane_7 in range(8):
    f = (z.iterator + synthetic_lane_7).load()
"""
    result = engine.transform(ast.parse(source).body)
    assert len(mutations) == 1
    assert checks == 6
    assert ast.unparse(ast.Module(body=result, type_ignores=[])).count(".load()") == 3
