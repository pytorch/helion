from __future__ import annotations

import ast
import itertools

import pytest

from .test_cute_computed_sweep_cache import _body
from .test_cute_computed_sweep_cache import _declarations
from .test_cute_computed_sweep_cache import _run


def _marked_body(*, inplace: bool = False, shift: int = 0) -> list[ast.stmt]:
    body = _body()
    for loop in (node for node in body if isinstance(node, ast.For)):
        loop.iter = ast.parse("cutlass.range_constexpr(8)", mode="eval").body
        load = loop.body[0]
        assert isinstance(load, ast.Assign)
        load.value = ast.parse(
            "_helion_persistent_branch_vec_load(7, 8, 'cutlass.Float32', '', "
            "x.iterator + synthetic_lane_7, "
            "(x.iterator + synthetic_lane_7).load())",
            mode="eval",
        ).body
    second = body[-1]
    assert isinstance(second, ast.For)
    target = "x" if inplace else "out"
    second.body[-1] = ast.parse(
        "_helion_persistent_branch_vec_store(7, 8, 'cutlass.Float32', "
        f"{target}.iterator + synthetic_lane_7 + {shift}, current + total, None)"
    ).body[0]
    return body


@pytest.mark.parametrize("inplace", (False, True))
def test_computed_cache_preserves_vector_markers_and_store_phase(
    inplace: bool,
) -> None:
    source = _run(_marked_body(inplace=inplace))
    assert len(_declarations(source)) == 1
    assert source.count("_helion_persistent_branch_vec_load(") == 1
    assert source.count("_helion_persistent_branch_vec_store(") == 1
    loops = [node for node in ast.parse(source).body if isinstance(node, ast.For)]
    assert "_computed_cache_" in ast.unparse(loops[1].body[0])
    assert "_helion_persistent_branch_vec_store" in ast.unparse(loops[1].body[-1])


def test_shifted_inplace_vector_store_cannot_consume_snapshot() -> None:
    source = _run(_marked_body(inplace=True, shift=1))
    assert not _declarations(source)


@pytest.mark.parametrize("mutation", ("store", "unknown", "mask"))
def test_vector_marker_composition_keeps_memory_and_control_barriers(
    mutation: str,
) -> None:
    body = _marked_body()
    if mutation == "mask":
        second = body[-1]
        assert isinstance(second, ast.For)
        second.body = [
            ast.If(
                test=ast.Name(id="active", ctx=ast.Load()),
                body=second.body,
                orelse=[],
            )
        ]
    else:
        text = (
            "(x.iterator + 1).store(cutlass.Float32(9))"
            if mutation == "store"
            else "unknown_effect()"
        )
        body.insert(-1, ast.parse(text).body[0])
    assert not _declarations(_run(body))


@pytest.mark.parametrize("mode", ("auto", "register", "gmem"))
def test_original_reload_modes_do_not_enable_computed_vector_cache(mode: str) -> None:
    assert "_computed_cache_" not in _run(_marked_body(), mode)


def _indexed_body() -> list[ast.stmt]:
    source = ast.unparse(ast.Module(body=_marked_body(inplace=True), type_ignores=[]))
    source = source.replace(
        "x.iterator + synthetic_lane_7", "x.iterator + slot * 8 + synthetic_lane_7"
    ).replace(
        "for synthetic_lane_7 in cutlass.range_constexpr(8):",
        "for synthetic_lane_7 in cutlass.range_constexpr(8):\n"
        "    slot = cute.arch.load(indices.iterator + batch, cutlass.Int64)",
    )
    return ast.parse(source).body


def _run_indexed(
    body: list[ast.stmt],
    *,
    disjoint: bool = True,
    constexpr_values: dict[str, int] | None = None,
) -> str:
    tensors = {"x": "cutlass.Float32", "indices": "cutlass.Int64"}
    return _run(
        body,
        tensor_dtypes=tensors,
        constexpr_values=constexpr_values,
        proven_disjoint_tensor_pairs=(
            {frozenset(pair) for pair in itertools.combinations(tensors, 2)}
            if disjoint
            else set()
        ),
    )


def test_immutable_index_loads_move_to_each_positive_sweep_entry() -> None:
    source = _run_indexed(_indexed_body())
    body = ast.parse(source).body
    loops = [node for node in body if isinstance(node, ast.For)]
    assert len(_declarations(source)) == 1
    assert source.count("slot = cute.arch.load(") == 2
    assert all("slot = cute.arch.load(" not in ast.unparse(loop) for loop in loops)
    for loop in loops:
        assert "slot = cute.arch.load(" in ast.unparse(body[body.index(loop) - 1])


@pytest.mark.parametrize(
    "barrier",
    (
        "alias",
        "write",
        "incoming",
        "redefine",
        "varying",
        "opaque",
        "volatile",
        "extra_argument",
        "range_bound",
    ),
)
def test_index_load_hoisting_retains_alias_binding_and_effect_proofs(
    barrier: str,
) -> None:
    body = _indexed_body()
    loops = [node for node in body if isinstance(node, ast.For)]
    for loop in loops:
        if barrier == "write":
            loop.body.append(
                ast.parse("indices.iterator.store(cutlass.Int64(1))").body[0]
            )
        elif barrier == "incoming":
            loop.body.insert(0, ast.parse("saved = slot").body[0])
        elif barrier == "redefine":
            loop.body.append(ast.parse("slot = cutlass.Int64(1)").body[0])
        elif barrier == "varying":
            load = loop.body[0]
            assert isinstance(load, ast.Assign)
            load.value = ast.parse(
                "cute.arch.load(indices.iterator + synthetic_lane_7, cutlass.Int64)",
                mode="eval",
            ).body
        elif barrier == "opaque":
            loop.body.append(ast.parse("unknown_effect()").body[0])
        elif barrier in ("volatile", "extra_argument"):
            load = loop.body[0]
            assert isinstance(load, ast.Assign)
            extra = ", volatile=True" if barrier == "volatile" else ", 1"
            load.value = ast.parse(
                f"cute.arch.load(indices.iterator + batch, cutlass.Int64{extra})",
                mode="eval",
            ).body
        elif barrier == "range_bound":
            loop.iter = ast.parse("range(slot)", mode="eval").body
    source = _run_indexed(
        body,
        disjoint=barrier != "alias",
        constexpr_values={"slot": 8} if barrier == "range_bound" else None,
    )
    final_loops = [node for node in ast.parse(source).body if isinstance(node, ast.For)]
    assert all("slot = cute.arch.load(" in ast.unparse(loop) for loop in final_loops)
