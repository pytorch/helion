from __future__ import annotations

import ast
from dataclasses import replace
from typing import Literal
from unittest.mock import patch

import pytest

from ._cute_aux import _cpu_codegen
from .test_cute_chained_loop_collectives import _inputs
from .test_cute_chained_loop_collectives import _prefix_coefficient_recurrence
from .test_cute_direct_affine_replay import _feature_reduction_loop
import helion
from helion._compiler import tile_strategy as lanes
from helion._compiler.ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from helion._compiler.cute import chained_collectives
from helion._compiler.cute import producer_phase as phase_impl
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.producer_phase import CollectiveProducer
from helion._compiler.cute.producer_phase import LaneProducer
from helion._compiler.cute.producer_phase import PointGuard
from helion._compiler.cute.producer_phase import PointIR
from helion._compiler.cute.producer_phase import PointLoop
from helion._compiler.cute.producer_phase import PointStore
from helion._compiler.cute.producer_phase import PointValue
from helion._compiler.cute.producer_phase import _copy_ir
from helion._compiler.cute.producer_phase import collective_body
from helion._compiler.cute.producer_phase import emit_point_actions
from helion._compiler.cute.producer_phase import emit_producer_phase
from helion._compiler.cute.producer_phase import split_producer_reductions


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _source(statements: tuple[ast.stmt, ...]) -> str:
    return "\n".join(ast.unparse(ast.fix_missing_locations(s)) for s in statements)


def _point(dtype: str) -> PointValue:
    return PointValue(
        tuple(
            ast.parse(
                f"loaded = src[lane * 4 + element] if valid else cutlass.{dtype}(0)\n"
                "wide = cutlass.Float32(loaded)\n"
                "value = cute.math.exp2(wide, fastmath=False)"
            ).body
        ),
        _expr("value"),
        PointStore(
            _expr("dst.iterator + cutlass.Int32(lane * 4 + element)"),
            pointer=True,
            cast_to=_expr("cutlass.Float32"),
        ),
    )


def _lane(dtype: str = "BFloat16") -> LaneProducer:
    scalar = PointValue(
        tuple(ast.parse("scale = scale_source[row]").body),
        _expr("scale"),
        PointStore(_expr("scales[row]")),
    )
    return LaneProducer(
        _expr("warp == 2"),
        "lane",
        "element",
        4,
        (_point(dtype),),
        ((_expr("lane == 0"), scalar),),
        frozenset(("dst", "scales")),
        frozenset(),
    )


@pytest.mark.parametrize("dtype", ("BFloat16", "Float16", "Float32"))
def test_lane_scalar_ir_masks_casts_order_and_whole_role_join(dtype: str) -> None:
    program = _lane(dtype)
    before = tuple(
        ast.dump(s, include_attributes=True) for s in program.values[0].statements
    )
    phase = emit_producer_phase(
        (program,), ast.parse("cute.arch.sync_threads()").body[0]
    )
    assert phase is not None
    source = _source(phase)
    assert f"else cutlass.{dtype}(0)" in source
    assert "wide = cutlass.Float32(loaded)" in source
    assert "cute.math.exp2(wide, fastmath=False)" in source
    assert ".store(cutlass.Float32(value))" in source
    assert source.index("scale = scale_source[row]") < source.index("if lane == 0:")
    assert isinstance(phase[0], ast.If)
    assert ast.unparse(phase[-1]) == "cute.arch.sync_threads()"
    assert source.count("cute.arch.sync_threads()") == 1
    assert before == tuple(
        ast.dump(s, include_attributes=True) for s in program.values[0].statements
    )


def test_evaluation_instances_are_not_deduplicated() -> None:
    program = _lane()
    two = replace(program, values=(program.values[0], program.values[0]))
    phase = emit_producer_phase((two,), ast.parse("join()").body[0])
    assert phase is not None
    source = _source(phase)
    assert source.count("loaded =") == 2
    assert source.count(".store(cutlass.Float32(value))") == 2


def test_original_marker_wrap_and_extended_metadata_are_preserved() -> None:
    loop = _feature_reduction_loop()
    marker = loop.body[2]
    assert isinstance(marker, ast.Assign)
    marker.value = ast.Call(
        func=_expr("cutlass.Float32"), args=[marker.value], keywords=[]
    )
    before = ast.dump(loop, include_attributes=True)
    copied = _copy_ir(loop)
    assert copied is not loop
    assert ast.dump(copied, include_attributes=True) == before
    assert vars(copied).keys() == vars(loop).keys()
    emitted = split_producer_reductions(
        loop, frozenset(("scratch",)), "lane", frozenset()
    )
    assert emitted is not None
    source = _source(emitted)
    assert "cutlass.Float32(cute.arch.warp_reduction_sum(" in source
    assert source.count("for element in range(4)") == 2
    assert "_helion_lane_reduce" not in source
    assert ast.dump(loop, include_attributes=True) == before


@pytest.mark.parametrize("failure", ("carry", "name"))
def test_failed_marker_split_retains_original_ir(failure: str) -> None:
    loop = _feature_reduction_loop("carry += value" if failure == "carry" else "")
    before = ast.dump(loop, include_attributes=True)
    reserved = frozenset(("reduced_lane_acc",)) if failure == "name" else frozenset()
    assert (
        split_producer_reductions(loop, frozenset(("scratch",)), "lane", reserved)
        is None
    )
    assert ast.dump(loop, include_attributes=True) == before


def _owned_marker(owner: str) -> str:
    return lanes._lane_reduce_marker_expr(
        "value", "sum", "cutlass.Float32(0)", 32, owner_lane=owner
    )


def _owned_feature_reduction_loop(owner: str) -> ast.For:
    loop = ast.parse(
        "for element in range(4):\n"
        "    feature = lane * 4 + element\n"
        "    value = source.iterator.load() + feature\n"
        f"    reduced = {_owned_marker(owner)}\n"
        "    (scratch.iterator + feature).store(value + reduced)"
    ).body[0]
    assert isinstance(loop, ast.For)
    setattr(loop, HELION_LANE_LOOP_VAR_ATTR, "element")
    return loop


@pytest.mark.parametrize(
    "source_lane,accepted",
    (("feature_lane", True), ("other_lane", False), (None, False)),
)
def test_marker_source_lane_owner_is_rebound_or_rejected(
    source_lane: str | None, accepted: bool
) -> None:
    loop = _owned_feature_reduction_loop("feature_lane")
    before = ast.dump(loop, include_attributes=True)
    emitted = split_producer_reductions(
        loop,
        frozenset(("scratch",)),
        "lane",
        frozenset(),
        source_lane=source_lane,
    )
    assert ast.dump(loop, include_attributes=True) == before
    if not accepted:
        assert emitted is None
        return
    assert emitted is not None
    source = _source(emitted)
    assert source.count("cute.arch.warp_reduction_sum(") == 1
    assert source.count("for element in range(4)") == 2
    assert "_helion_lane_reduce" not in source


def test_marker_source_lane_requires_destination_lane_loop() -> None:
    loop = _owned_feature_reduction_loop("feature_lane")
    delattr(loop, HELION_LANE_LOOP_VAR_ATTR)
    assert (
        split_producer_reductions(
            loop,
            frozenset(("scratch",)),
            "lane",
            frozenset(),
            source_lane="feature_lane",
        )
        is None
    )


@pytest.mark.parametrize("source_lane", ("feature_lane", "other_lane", None))
def test_lane_producer_threads_marker_source_lane(source_lane: str | None) -> None:
    point = PointValue(
        tuple(
            ast.parse(
                "value = src[lane * 4 + element]\n"
                f"reduced = {_owned_marker('feature_lane')}"
            ).body
        ),
        _expr("value + reduced"),
        PointStore(
            _expr("dst.iterator + cutlass.Int32(lane * 4 + element)"),
            pointer=True,
            cast_to=_expr("cutlass.Float32"),
        ),
    )
    program = replace(_lane(), values=(point,), source_lane=source_lane)
    phase = emit_producer_phase((program,), ast.parse("join()").body[0])
    if source_lane != "feature_lane":
        assert phase is None
        return
    assert phase is not None
    source = _source(phase)
    assert source.count("cute.arch.warp_reduction_sum(") == 1
    assert "_helion_lane_reduce" not in source
    assert ast.unparse(phase[-1]) == "join()"


@pytest.mark.parametrize("extent", (0, 1, 17, 32, 33, 128))
@pytest.mark.parametrize("axis", (0, 1))
def test_sum_recipe_retains_ordered_parts_zero_and_five_down_adds(
    extent: int, axis: int
) -> None:
    coords = ("p_position", "p_vector") if axis == 0 else ("p_vector", "p_position")
    program = CollectiveProducer(
        "p",
        "sum",
        extent,
        17,
        coords,
        ("p_vector",),
        ("loaded = original[p_position] if valid else cutlass.Float32(0)",),
        "loaded",
        ChainedExecution(128),
    )
    phase = emit_producer_phase((program,), ast.parse("role_join()").body[0])
    assert phase is not None
    source = _source(phase)
    assert "p_acc = cutlass.Float32(0)" in source
    assert "p_position = chain_thread % 32 + p_part * 32" in source
    offsets = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "cute.arch.shuffle_sync_down"
        ):
            offset = node.keywords[0].value
            assert isinstance(offset, ast.Constant)
            offsets.append(offset.value)
    assert offsets == [16, 8, 4, 2, 1]
    assert source.count("p_acc += loaded") == 1
    assert source.endswith("role_join()")


@pytest.mark.parametrize("extent", (1, 17, 32, 33, 128))
def test_serial_and_warp_prefix_are_distinct_original_programs(extent: int) -> None:
    program = CollectiveProducer(
        "p",
        "serial_scan",
        extent,
        17,
        ("p_vector", "p_position"),
        ("p_vector",),
        ("loaded = source[p_vector, p_position]",),
        "loaded",
        ChainedExecution(128),
    )
    serial = "\n".join(collective_body(program))
    warp = "\n".join(collective_body(replace(program, kind="warp_scan")))
    assert f"cutlass.range({extent}, unroll=1)" in serial
    assert "shuffle_sync" not in serial
    assert "p_acc = cutlass.Float32(0) + loaded" in warp
    assert [warp.index(f"offset={offset}") for offset in (1, 2, 4, 8, 16)] == sorted(
        warp.index(f"offset={offset}") for offset in (1, 2, 4, 8, 16)
    )
    assert ("p_acc = p_carry + p_acc" in warp) == (extent > 32)


@pytest.mark.parametrize("axis", (0, 1))
@pytest.mark.parametrize("scan", ("serial", "warp"))
def test_actual_common_loop_uses_phase_executor(axis: int, scan: str) -> None:
    calls: list[tuple[LaneProducer | CollectiveProducer, ...]] = []

    def record(programs, completion):
        calls.append(programs)
        return emit_producer_phase(programs, completion)

    with (
        _cpu_codegen(),
        patch.object(chained_collectives, "emit_producer_phase", record),
    ):
        bound = _prefix_coefficient_recurrence._bind_isolated(_inputs("cpu", 3, axis))
        source = bound.to_code(
            helion.Config(
                cute_chained_mma_schedule="tcgen05_tmem",
                cute_chained_scan_schedule=scan,
                num_warps=4,
            )
        )
    assert len(calls) == 3
    assert [
        p.kind for group in calls for p in group if isinstance(p, CollectiveProducer)
    ] == [f"{scan}_scan", f"{scan}_scan", "sum"]
    assert "chain_collective_2_acc" in source


@pytest.mark.parametrize("kind", ("lane", "sum", "serial_scan", "warp_scan"))
def test_all_recipes_share_loop_guard_scalar_and_store_interpreter(
    kind: Literal["lane", "sum", "serial_scan", "warp_scan"],
) -> None:
    program = (
        _lane()
        if kind == "lane"
        else CollectiveProducer(
            "p",
            kind,
            17,
            16,
            ("p_vector", "p_position"),
            ("p_vector",),
            ("x = original[p_vector, p_position]",),
            "x",
            ChainedExecution(128),
        )
    )
    seen = []
    original = phase_impl.emit_point_actions

    def observe(actions):
        seen.extend(type(action) for action in actions)
        return original(actions)

    stores = []
    original_store = PointStore.statement

    def store(target, value):
        stores.append(target.pointer)
        return original_store(target, value)

    with (
        patch.object(phase_impl, "emit_point_actions", observe),
        patch.object(PointStore, "statement", store),
    ):
        body = emit_producer_phase((program,), ast.parse("join()").body[0])
    assert body is not None
    assert {PointLoop, PointGuard, PointIR, PointValue} <= set(seen)
    assert stores == ([True, False] if kind == "lane" else [False])


@pytest.mark.parametrize(
    "source",
    (
        "for column in range(4):\n    result = source[column]",
        "while active:\n    result += source",
        "output[row] = value",
        "output.iterator.store(value)",
        "if valid:\n    for column in range(4):\n        result = source[column]",
    ),
)
def test_point_ir_cannot_hide_rendered_loops_or_publication(source: str) -> None:
    point = PointIR(tuple(ast.parse(source).body))
    before = tuple(
        ast.dump(statement, include_attributes=True) for statement in point.statements
    )
    assert emit_point_actions((point,)) is None
    assert before == tuple(
        ast.dump(statement, include_attributes=True) for statement in point.statements
    )


def test_failure_after_first_action_does_not_mutate_scalar_ir_or_emit_join() -> None:
    point = _point("Float16")
    bad = replace(
        point, statements=tuple(ast.parse("for i in range(2):\n    x = i").body)
    )
    program = replace(_lane("Float16"), values=(point, bad))
    original = tuple(ast.dump(statement) for statement in point.statements)
    completion = ast.parse("join()").body[0]
    assert emit_producer_phase((program,), completion) is None
    assert original == tuple(ast.dump(statement) for statement in point.statements)
    assert ast.unparse(completion) == "join()"
