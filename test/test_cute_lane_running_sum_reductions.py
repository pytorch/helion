"""Keep a proved scalar matmul fold beside a complete owned lane reduction."""

from __future__ import annotations

import ast
import copy
from types import SimpleNamespace
from typing import Any
from typing import TypedDict
from typing import Unpack
from typing import cast

import numpy as np
import pytest

import helion
from helion._compiler import tile_strategy as lanes
from helion._compiler.ast_read_writes import ast_rename

_RENAMES = {"acc_next": "acc", "extra_next": "extra"}


def _body(source: str) -> list[ast.AST]:
    return list(ast.parse(source).body)


def _source(body: list[ast.AST]) -> str:
    return ast.unparse(
        ast.fix_missing_locations(
            ast.Module(body=cast("list[ast.stmt]", body), type_ignores=[])
        )
    )


def _marker(owner: str | None = "lane", value: str = "partial") -> str:
    return lanes._lane_reduce_marker_expr(
        value, "sum", "cutlass.Float32(0)", 1, owner_lane=owner
    )


def _loop(
    extent: int = 4, *, marker_first: bool = False, owner: str | None = "lane"
) -> ast.For:
    reduction = f"partial = aux[offset + lane]\nreduced = {_marker(owner)}\n"
    matmul = (
        "product = values[offset + lane]\n"
        "fold = fold + product\nacc_next = base + fold\n"
    )
    return lanes._create_lane_loop(
        "lane",
        extent,
        _body(
            "acc_copy = acc\nextra_copy = extra\n"
            + (reduction + matmul if marker_first else matmul + reduction)
            + "extra_next = extra_copy + reduced\n"
        ),
    )


class _SplitOptions(TypedDict, total=False):
    proven_disjoint_tensor_pairs: set[frozenset[str]]


def _split(loop: ast.For, **kwargs: Unpack[_SplitOptions]) -> list[ast.AST]:
    return lanes.split_lane_loop_reductions(
        [loop], rename_groups=_RENAMES, running_sums={"fold"}, **kwargs
    )


def _execute(body: list[ast.AST], **values: object) -> dict[str, Any]:
    module = ast.parse(_source(body))
    ast_rename(module, _RENAMES)
    namespace = {"cutlass": SimpleNamespace(Float32=np.float32, Int32=int), **values}
    exec(compile(module, "<certified-lane-fold>", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("extent", [2, 5, 16])
@pytest.mark.parametrize("marker_first", [False, True])
def test_full_lane_sum_and_matmul_fold_have_distinct_scopes(
    extent: int, marker_first: bool
) -> None:
    loop = _loop(extent, marker_first=marker_first)
    outer = ast.For(
        target=ast.Name(id="offset", ctx=ast.Store()),
        iter=ast.parse(f"range(0, {3 * extent}, {extent})", mode="eval").body,
        body=cast("list[ast.stmt]", [loop]),
        orelse=[],
    )
    ast.fix_missing_locations(outer)
    body = lanes.split_lane_loop_reductions(
        [outer], rename_groups=_RENAMES, running_sums={"fold"}
    )
    values = np.resize(np.array([2**24, 1, -(2**24), 3, -2], np.float32), 3 * extent)
    aux = np.arange(1, 3 * extent + 1, dtype=np.float32)
    expected_fold = np.float32(7)
    expected_extra = np.float32(11)
    for offset in range(0, values.size, extent):
        partial = np.float32(0)
        for lane in range(extent):
            expected_fold = expected_fold + values[offset + lane]
            partial = partial + aux[offset + lane]
        expected_extra = expected_extra + partial
    actual = _execute(
        body,
        values=values,
        aux=aux,
        fold=np.float32(7),
        base=np.float32(5),
        acc=np.float32(5),
        extra=np.float32(11),
    )
    assert actual["fold"].tobytes() == expected_fold.tobytes()
    assert actual["acc"] == np.float32(5) + expected_fold
    assert actual["extra"] == expected_extra
    assert "_helion_lane_reduce" not in _source(body)
    # One inner sweep per tile. The two original finalizers are after it;
    # the exact self-add stays at its original position among the producers.
    lowered_outer = body[0]
    assert isinstance(lowered_outer, ast.For)
    inner = next(stmt for stmt in lowered_outer.body if isinstance(stmt, ast.For))
    assert ast.dump(inner.iter) == ast.dump(loop.iter)
    names = [lanes._plain_assignment_name(stmt) for stmt in inner.body]
    assert names.count("fold") == 1
    assert names.index("product") < names.index("fold")
    assert (names.index("fold") < names.index("reduced_lane_acc")) != marker_first
    assert "acc_next" not in names and "extra_next" not in names
    assert sum(isinstance(stmt, ast.For) for stmt in lowered_outer.body) == 1


def test_fp32_running_additions_are_not_regrouped_into_a_fresh_sum() -> None:
    values = np.array([1, 1, -(2**24), 1], np.float32)
    initial = np.float32(2**24)
    serial = initial
    regrouped_partial = np.float32(0)
    for value in values:
        serial = serial + value
        regrouped_partial = regrouped_partial + value
    regrouped = initial + regrouped_partial
    assert serial != regrouped
    actual = _execute(
        _split(_loop()),
        offset=0,
        values=values,
        aux=np.ones(4, np.float32),
        fold=initial,
        base=np.float32(0),
        acc=np.float32(0),
        extra=np.float32(0),
    )
    assert actual["fold"].tobytes() == serial.tobytes()


def test_provenance_does_not_depend_on_a_temporary_name_convention() -> None:
    loop = _loop()
    ast_rename(loop, {"fold": "opaque_temporary"})
    lowered = lanes.split_lane_loop_reductions(
        [loop], rename_groups=_RENAMES, running_sums={"opaque_temporary"}
    )
    assert "opaque_temporary = opaque_temporary + product" in _source(lowered)


@pytest.mark.parametrize("provenance", [None, set(), {"other_name"}])
def test_a_self_add_without_compiler_provenance_still_declines(
    provenance: set[str] | None,
) -> None:
    with pytest.raises(helion.exc.BackendUnsupported, match="loop-carried value"):
        lanes.split_lane_loop_reductions(
            [_loop()], rename_groups=_RENAMES, running_sums=provenance
        )


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("fold = fold + product", "if flag:\n    fold = fold + product"),
        ("fold = fold + product", "fold = fold + product\nfold = fold + product"),
        ("fold = fold + product", "fold += product"),
        ("fold = fold + product", "fold = fold * product"),
        ("fold = fold + product", "fold = fold + fold"),
        ("fold = fold + product", "snapshot = fold\nfold = fold + product"),
        ("product = values[offset + lane]", "if flag:\n    product = lane + 1"),
        (
            "product = values[offset + lane]",
            "product = lane + 1\nproduct = product * 2",
        ),
        ("product = values[offset + lane]", "product = acc_copy + lane"),
        ("product = values[offset + lane]", "product = reduced + lane"),
        (
            "product = values[offset + lane]",
            "alias = reduced\nproduct = alias + lane",
        ),
        ("product = values[offset + lane]", "product = unknown_call(lane)"),
        (
            "product = values[offset + lane]",
            "product = cute.arch.warp_reduction_sum(lane)",
        ),
        ("product = values[offset + lane]", "product = producer\nproducer = lane"),
        ("partial = aux[offset + lane]", "partial = fold"),
        ("acc_next = base + fold", "acc_next = base + fold + lane"),
        ("acc_next = base + fold", "if flag:\n    acc_next = base + fold"),
        ("acc_next = base + fold", "(out.iterator + lane).store(fold)"),
        ("acc_next = base + fold", "side_effect()\nacc_next = base + fold"),
        ("acc_next = base + fold", "acc_next = base + fold\nother = other + product"),
        (
            "extra_next = extra_copy + reduced",
            "extra_next = extra_copy + reduced + partial",
        ),
    ],
)
def test_uncertified_inputs_carries_and_observations_remain_rejected(
    old: str, new: str
) -> None:
    loop = _loop()
    source = _source(list(loop.body))
    assert source.count(old) == 1
    loop.body = cast("list[ast.stmt]", _body(source.replace(old, new)))
    with pytest.raises(helion.exc.BackendUnsupported):
        _split(loop)


def test_multiple_provenance_accumulators_need_a_separate_schedule() -> None:
    loop = _loop()
    loop.body.insert(4, ast.parse("second = second + product").body[0])
    with pytest.raises(helion.exc.BackendUnsupported, match="one scalar accumulator"):
        lanes.split_lane_loop_reductions(
            [loop], rename_groups=_RENAMES, running_sums={"fold", "second"}
        )


@pytest.mark.parametrize("iterator", ["range(0)", "range(1)", "range(dynamic)"])
def test_only_complete_static_lane_iterators_are_certified(iterator: str) -> None:
    loop = _loop()
    loop.iter = ast.parse(iterator, mode="eval").body
    with pytest.raises(
        helion.exc.BackendUnsupported, match="complete static lane loop"
    ):
        _split(loop)


def test_alias_check_includes_matmul_only_input_loads() -> None:
    def make() -> ast.For:
        loop = _loop()
        source = _source(list(loop.body)).replace(
            "product = values[offset + lane]",
            "product = (x.iterator + lane).load()",
        )
        source += "\n(out.iterator + lane + 1).store(partial)"
        loop.body = cast("list[ast.stmt]", _body(source))
        return loop

    with pytest.raises(helion.exc.BackendUnsupported, match="aliasing write"):
        _split(make())
    # A separate allocation permits the ordinary second pass. Its store does
    # not observe a matmul prefix; only the marker's independent input is replayed.
    result = _split(make(), proven_disjoint_tensor_pairs={frozenset(("x", "out"))})
    assert sum(isinstance(stmt, ast.For) for stmt in result) == 2
    assert _source(result).count("(x.iterator + lane).load()") == 1
    assert _source(result).count("fold = fold + product") == 1


def test_shared_pure_producer_is_kept_once_in_the_accumulation_pass() -> None:
    loop = _loop()
    loop.body = cast(
        "list[ast.stmt]",
        _body(
            _source(list(loop.body)).replace(
                "partial = aux[offset + lane]", "partial = product"
            )
        ),
    )
    result = _split(loop)
    first = next(stmt for stmt in result if isinstance(stmt, ast.For))
    assert _source(list(first.body)).count("product = values[offset + lane]") == 1


@pytest.mark.parametrize("constexpr", [False, True])
def test_iterator_form_is_preserved(constexpr: bool) -> None:
    loop = _loop()
    if constexpr:
        loop.iter = ast.parse("cutlass.range_constexpr(4)", mode="eval").body
    result = _split(loop)
    phase1 = next(stmt for stmt in result if isinstance(stmt, ast.For))
    assert ast.dump(phase1.iter) == ast.dump(loop.iter)


def test_unowned_marker_and_no_marker_behaviors_are_unchanged() -> None:
    legacy = lanes.split_lane_loop_reductions(
        [_loop(owner=None)], rename_groups=_RENAMES
    )
    certified_arg = _split(_loop(owner=None))
    assert _source(legacy) == _source(certified_arg)
    assert "reduced = partial" in _source(legacy)
    body = _body("product = lane + 1\nfold = fold + product")
    loop = lanes._create_lane_loop("lane", 4, body)
    assert _source(_split(loop)) == _source([loop])


@pytest.mark.parametrize(
    "fault",
    [
        "omitted_update",
        "duplicate_update",
        "tail_update",
        "conditional_update",
        "changed_iterator",
        "producer_order",
        "omitted_producer",
        "regrouped_add",
        "duplicate_sweep",
        "early_finalization",
        "missing_finalization",
        "early_consumer",
        "omitted_other_carry",
        "repeated_other_carry",
    ],
)
def test_final_schedule_proof_rejects_corrupted_placements(fault: str) -> None:
    loop = _loop()
    original = cast("list[ast.AST]", copy.deepcopy(list(loop.body)))
    markers = [
        (index, marker)
        for index, stmt in enumerate(original)
        if (marker := lanes._is_lane_reduce_marker_assign(stmt)) is not None
    ]
    certificate = lanes._certify_lane_running_sum(
        loop, "lane", markers, _RENAMES, {"fold"}
    )
    assert certificate is not None
    result = _split(loop)
    phase1 = next(stmt for stmt in result if isinstance(stmt, ast.For))
    position = next(
        index
        for index, stmt in enumerate(phase1.body)
        if lanes._plain_assignment_name(stmt) == "fold"
    )
    update = phase1.body[position]
    if fault == "omitted_update":
        phase1.body.pop(position)
    elif fault == "duplicate_update":
        phase1.body.insert(position, copy.deepcopy(update))
    elif fault == "tail_update":
        phase1.body.pop(position)
        result.append(update)
    elif fault == "conditional_update":
        phase1.body[position] = ast.If(
            test=ast.Name(id="flag", ctx=ast.Load()), body=[update], orelse=[]
        )
    elif fault == "changed_iterator":
        phase1.iter = ast.parse("range(3)", mode="eval").body
    elif fault == "producer_order":
        phase1.body[position - 1], phase1.body[position] = (
            phase1.body[position],
            phase1.body[position - 1],
        )
    elif fault == "omitted_producer":
        phase1.body.pop(position - 1)
    elif fault == "regrouped_add":
        phase1.body[position] = ast.parse("fold = fold + product * 4").body[0]
    elif fault == "duplicate_sweep":
        result.append(lanes._clone_stmt(phase1))
    elif fault in ("early_finalization", "missing_finalization"):
        index = next(
            index
            for index, stmt in enumerate(result)
            if lanes._plain_assignment_name(stmt) == "reduced"
        )
        finalized = result.pop(index)
        if fault == "early_finalization":
            result.insert(0, finalized)
    elif fault == "early_consumer":
        index = next(
            index
            for index, stmt in enumerate(result)
            if lanes._plain_assignment_name(stmt) == "acc_next"
        )
        result.insert(0, result.pop(index))
    else:
        index = next(
            index
            for index, stmt in enumerate(result)
            if lanes._plain_assignment_name(stmt) == "extra_next"
        )
        other = result.pop(index)
        if fault == "repeated_other_carry":
            phase1.body.append(cast("ast.stmt", other))
    with pytest.raises(helion.exc.BackendUnsupported):
        lanes._validate_owned_lane_carry_schedule(
            original,
            result,
            "lane",
            {index for index, _ in markers},
            _RENAMES,
            running_sum=certificate,
        )
