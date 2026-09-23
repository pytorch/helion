from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Any
from typing import cast

import numpy as np
import pytest
import torch

import helion
from helion._compiler import tile_strategy as lanes
from helion._compiler.ast_read_writes import ast_rename
from helion._compiler.cute import matmul_fallback

_RENAMES = {"next_high": "high", "next_mass": "mass", "next_total": "total"}


def _marker(value: str, operation: str, *, product: bool = False) -> str:
    identity = "float('-inf')" if operation == "max" else "0"
    return lanes._lane_reduce_marker_expr(
        value,
        operation,
        f"cutlass.Float32({identity})",
        1,
        owner_lane="lane",
        matmul_contribution=product,
    )


def _body(*, product: bool = True) -> str:
    return f"""old_high = high
old_mass = mass
old_total = total
index = offset + lane
score_input = cutlass.Float32(scores[index] / 64)
score = _cute_grouped_reduce_shared_two_stage(score_input, 'sum', cutlass.Float32(0), 0, 0, 0, pre=1, group_span=64, group_count=1)
peak = {_marker("score", "max")}
high_value = cute.math.max(old_high, peak, propagate_nan=True)
weight = cute.math.exp2(score - high_value)
weight_sum = {_marker("weight", "sum")}
alpha = cute.math.exp2(old_high - high_value)
scaled_mass = old_mass * alpha
next_mass = scaled_mass + weight_sum
value = values[index]
product = cutlass.Float32(cutlass.Float32(cutlass.Float16(weight)) * cutlass.Float32(value))
product_sum = {_marker("product", "sum", product=product)}
scaled_total = old_total * alpha
next_total = scaled_total + product_sum
next_high = high_value
"""


def _loop(extent: int = 8, source: str | None = None) -> ast.For:
    return lanes._create_lane_loop(
        "lane", extent, list(ast.parse(source if source is not None else _body()).body)
    )


def _source(body: list[ast.AST]) -> str:
    return ast.unparse(
        ast.fix_missing_locations(
            ast.Module(body=cast("list[ast.stmt]", body), type_ignores=[])
        )
    )


def _lower(loop: ast.For) -> list[ast.AST]:
    return lanes.split_lane_loop_reductions([loop], rename_groups=_RENAMES)


def _execute(body: list[ast.AST], **values: object) -> tuple[dict[str, Any], int]:
    calls = 0

    def collective(
        value: np.float32,
        operation: str,
        identity: np.float32,
        lane: int,
        lane_in_group: int,
        lane_mod_pre: int,
        *,
        pre: int,
        group_span: int,
        group_count: int,
    ) -> np.float32:
        nonlocal calls
        assert operation == "sum" and identity == 0
        assert lane == lane_in_group == lane_mod_pre == 0
        assert pre == group_count == 1 and group_span == 64
        calls += 1
        # The test uses 64 identical exactly representable score/64 values.
        return np.float32(value * 64)

    namespace = {
        "cutlass": SimpleNamespace(Float32=np.float32, Float16=np.float16, Int32=int),
        "cute": SimpleNamespace(
            make_rmem_tensor=lambda extent, dtype: np.zeros(extent, dtype=dtype),
            math=SimpleNamespace(
                exp2=lambda value: np.float32(np.exp2(value)),
                max=lambda left, right, **kwargs: np.maximum(left, right),
            ),
        ),
        "_cute_grouped_reduce_shared_two_stage": collective,
        **values,
    }
    module = ast.parse(_source(body))
    ast_rename(module, _RENAMES)
    exec(compile(module, "<staged-matmul-recurrence>", "exec"), namespace)
    return namespace, calls


@pytest.mark.parametrize("extent", [2, 5, 16, 128])
@pytest.mark.parametrize("initial_high", [np.float32(-np.inf), np.float32(1)])
def test_staged_products_and_rescale_preserve_FP32_recurrence(
    extent: int, initial_high: np.float32
) -> None:
    loop = _loop(extent)
    outer = ast.For(
        target=ast.Name(id="offset", ctx=ast.Store()),
        iter=ast.parse(f"range(0, {3 * extent}, {extent})", mode="eval").body,
        body=[loop],
        orelse=[],
    )
    ast.fix_missing_locations(outer)
    lowered = lanes.split_lane_loop_reductions([outer], rename_groups=_RENAMES)
    scores = np.resize(np.array([-2, -1, 0, 1, 2], np.float32), 3 * extent)
    scores[extent : 2 * extent] += 4
    scores[2 * extent :] += 8
    values = np.resize(np.array([13, -11, 7, -3, 0.25], np.float16), 3 * extent)
    high, mass, total = initial_high, np.float32(2), np.float32(17)
    for offset in range(0, scores.size, extent):
        next_high = np.maximum(high, scores[offset : offset + extent].max())
        alpha = np.float32(np.exp2(high - next_high))
        weight_sum = np.float32(0)
        product_sum = np.float32(0)
        for lane in range(extent):
            weight = np.float32(np.exp2(scores[offset + lane] - next_high))
            weight_sum = np.float32(weight_sum + weight)
            product = np.float32(np.float16(weight)) * np.float32(values[offset + lane])
            product_sum = np.float32(product_sum + product)
        total = np.float32(np.float32(total * alpha) + product_sum)
        mass = np.float32(np.float32(mass * alpha) + weight_sum)
        high = next_high
    actual, calls = _execute(
        lowered,
        scores=scores,
        values=values,
        high=initial_high,
        mass=np.float32(2),
        total=np.float32(17),
    )
    for name, expected in (("high", high), ("mass", mass), ("total", total)):
        assert actual[name].tobytes() == expected.tobytes()
    assert calls == 3 * extent
    assert "_helion_lane_reduce" not in _source(lowered)
    assert _source(lowered).count("next_total = scaled_total + product_sum") == 1
    assert _source(lowered).count("alpha = cute.math.exp2") == 1


def test_each_FP32_product_is_accumulated_without_half_rounding() -> None:
    source = _body().replace("cutlass.Float16(weight)", "weight")
    lowered = _lower(_loop(4, source))
    values = np.array([2**24, 1, -(2**24), 3], np.float32)
    actual, calls = _execute(
        lowered,
        offset=0,
        scores=np.zeros(4, np.float32),
        values=values,
        high=np.float32(0),
        mass=np.float32(0),
        total=np.float32(5),
    )
    assert calls == 4
    assert actual["product_sum"] == np.float32(3)
    assert actual["total"] == np.float32(8)


def test_matmul_provenance_is_required_for_the_new_stash_schedule() -> None:
    with pytest.raises(helion.exc.BackendUnsupported):
        _lower(_loop(source=_body(product=False)))


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (
            "next_total = scaled_total + product_sum",
            "next_total = scaled_total + product_sum + product",
        ),
        (
            "next_total = scaled_total + product_sum",
            "if flag:\n    next_total = scaled_total + product_sum",
        ),
        (
            "next_total = scaled_total + product_sum",
            "next_total = scaled_total + product_sum\nnext_total = next_total + 1",
        ),
        (
            "next_total = scaled_total + product_sum",
            "next_total = scaled_total + product_sum\nextra = extra + product",
        ),
        ("value = values[index]", "value = unknown_call(index)"),
        ("value = values[index]", "values[index] = 0\nvalue = values[index]"),
        (
            "score_input = cutlass.Float32(scores[index] / 64)",
            "score_input = cutlass.Float32(scores[index] / 64)\nscore_input = score_input + 1",
        ),
        ("'sum', cutlass.Float32(0), 0, 0, 0", "'sum', cutlass.Float16(0), 0, 0, 0"),
        (
            "score = _cute_grouped_reduce_shared_two_stage",
            "score = 1 + _cute_grouped_reduce_shared_two_stage",
        ),
        (
            "next_high = high_value",
            "next_high = high_value\n(out.iterator + lane).store(next_total)",
        ),
    ],
)
def test_unproved_carries_effects_and_collectives_still_decline(
    before: str, after: str
) -> None:
    source = _body()
    assert source.count(before) == 1
    with pytest.raises(helion.exc.BackendUnsupported):
        _lower(_loop(source=source.replace(before, after)))


def test_raw_product_is_not_a_complete_matmul_contribution() -> None:
    source = _body().replace(_marker("product", "sum", product=True), "product")
    with pytest.raises(helion.exc.BackendUnsupported, match="loop-carried value"):
        _lower(_loop(source=source))


@pytest.mark.parametrize(
    "iterator", ["range(0)", "range(1)", "range(dynamic)", "range(257)"]
)
def test_staging_requires_a_bounded_complete_lane_sweep(iterator: str) -> None:
    loop = _loop()
    loop.iter = ast.parse(iterator, mode="eval").body
    with pytest.raises(helion.exc.BackendUnsupported):
        _lower(loop)


@pytest.mark.parametrize("fault", ["drop", "repeat", "inside", "early"])
def test_existing_final_carry_validator_rejects_corrupt_staged_schedules(
    fault: str,
) -> None:
    loop = _loop()
    original = list(loop.body)
    marker_indices = {
        index
        for index, statement in enumerate(original)
        if lanes._is_lane_reduce_marker_assign(statement) is not None
    }
    lowered = _lower(loop)
    update_index = next(
        index
        for index, statement in enumerate(lowered)
        if lanes._plain_assignment_name(statement) == "next_total"
    )
    update = lowered[update_index]
    if fault == "drop":
        lowered.pop(update_index)
    elif fault == "repeat":
        lowered.append(lanes._clone_stmt(update))
    elif fault == "inside":
        lowered[update_index] = lanes._create_lane_loop("lane", 8, [update])
    else:
        lowered.pop(update_index)
        lowered.insert(0, update)
    with pytest.raises(helion.exc.BackendUnsupported, match="once-per-tile"):
        lanes._validate_owned_lane_carry_schedule(
            original, lowered, "lane", marker_indices, _RENAMES
        )


@pytest.mark.parametrize("bad", ["1", "'matmul'", "None"])
def test_product_provenance_is_a_typed_internal_marker(bad: str) -> None:
    marker = _marker("product", "sum", product=True).replace(", True)", f", {bad})")
    with pytest.raises(helion.exc.BackendUnsupported, match="matmul contribution"):
        lanes._is_lane_reduce_marker_assign(ast.parse(f"result = {marker}").body[0])


@pytest.mark.parametrize(
    ("axes", "axis", "threads", "pre", "span", "count"),
    [
        ({}, None, 1, 1, 0, 1),
        ({0: 8}, 0, 8, 1, 0, 1),
        ({0: 4, 1: 8}, 1, 8, 4, 32, 1),
        ({0: 64, 1: 2}, 0, 64, 1, 64, 2),
        ({0: 4, 1: 32, 2: 2}, 1, 32, 4, 128, 2),
    ],
)
def test_product_marker_preserves_physical_group_and_serial_owner(
    axes: dict[int, int],
    axis: int | None,
    threads: int,
    pre: int,
    span: int,
    count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = SimpleNamespace(
        dtype_str=lambda dtype: "cutlass.Float32",
        thread_linear_index_expr=lambda sizes: "physical_thread",
    )
    monkeypatch.setattr(
        matmul_fallback.CompileEnvironment,
        "current",
        lambda: SimpleNamespace(backend=backend),
    )
    monkeypatch.setattr(
        matmul_fallback,
        "_cute_active_thread_layout",
        lambda cg: (axes, {7: axis} if axis is not None else {}),
    )
    expression = matmul_fallback._emit_cute_owned_product_sum(
        SimpleNamespace(max_thread_block_dims=tuple(axes.values())),
        "contribution",
        value_dtype=torch.float32,
        loop_state=SimpleNamespace(block_thread_axes={}),
        k_block_id=7,
        owner_lane="contraction_lane",
    )
    marker = lanes._is_lane_reduce_marker_assign(
        ast.parse(f"result = {expression}").body[0]
    )
    assert marker is not None
    assert marker.input_name == "contribution" and marker.matmul_contribution
    assert marker.owner_lane == "contraction_lane" and marker.reduction_type == "sum"
    assert (
        marker.threads_in_group,
        marker.group_pre,
        marker.group_span,
        marker.group_count,
    ) == (threads, pre, span, count)
    assert marker.group_lane_expr == ("physical_thread" if span else "")


@pytest.mark.parametrize("fault", ["dtype", "overlaunch", "span", "missing_index"])
def test_product_marker_declines_unproved_thread_groups(
    fault: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    axes = {0: 3, 1: 16} if fault == "span" else {0: 4, 1: 32}
    backend = SimpleNamespace(
        dtype_str=lambda dtype: "cutlass.Float32",
        thread_linear_index_expr=lambda sizes: (
            None if fault == "missing_index" else "physical_thread"
        ),
    )
    monkeypatch.setattr(
        matmul_fallback.CompileEnvironment,
        "current",
        lambda: SimpleNamespace(backend=backend),
    )
    monkeypatch.setattr(
        matmul_fallback, "_cute_active_thread_layout", lambda cg: (axes, {7: 1})
    )
    with pytest.raises(helion.exc.BackendUnsupported):
        matmul_fallback._emit_cute_owned_product_sum(
            SimpleNamespace(
                max_thread_block_dims=(64,)
                if fault == "overlaunch"
                else tuple(axes.values())
            ),
            "contribution",
            value_dtype=torch.float16 if fault == "dtype" else torch.float32,
            loop_state=SimpleNamespace(block_thread_axes={}),
            k_block_id=7,
            owner_lane="contraction_lane",
        )
