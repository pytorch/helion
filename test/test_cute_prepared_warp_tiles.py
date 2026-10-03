from __future__ import annotations

import ast
from contextlib import contextmanager
from dataclasses import replace
import importlib
import inspect
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.cute.kda_prefill_fused_bt32 import kda_prefill_native_math_bt32
import pytest
import torch

from test._cute_aux import _cpu_codegen

import helion
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chunk_prefill_prepared_warp_tiles as adapter
from helion._compiler.cute.chunk_prefill_prepared_inverse import bind_fast_inverse
from helion._compiler.cute.chunk_prefill_prepared_issue import bind_fast_recurrence
from helion._compiler.cute.chunk_prefill_prepared_pairwise import bind_fast_pairwise
from helion._compiler.device_function import DeviceFunction

pytest.importorskip("cutlass.cute")


@contextmanager
def _capture(strategy):
    q = torch.empty((1, 64, 8, 128), dtype=torch.bfloat16)
    state = torch.empty((2, 8, 128, 128), dtype=torch.float32)
    args = (
        q,
        torch.empty_like(q),
        torch.empty_like(q),
        torch.empty_like(q),
        torch.empty((1, 64, 8), dtype=torch.bfloat16),
        torch.empty((8,), dtype=torch.float32),
        torch.empty((8, 128), dtype=torch.float32),
        state,
        torch.empty_like(q),
        torch.empty_like(state),
        torch.tensor([0, 32, 64], dtype=torch.int64),
        128**-0.5,
        -5 * 1.4426950408889634,
    )
    original = adapter.bind_fast_warp_tiles
    observed = []

    def observe(pairwise, inverse, strategy_argument="independent"):
        schedule = original(pairwise, inverse, strategy)
        observed.append((DeviceFunction.current(), schedule, inverse))
        return schedule

    with (
        _cpu_codegen(),
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 3)),
        patch("cutlass.cute.compile", side_effect=AssertionError("native forbidden")),
        patch.object(adapter, "bind_fast_warp_tiles", observe),
    ):
        bound = kda_prefill_native_math_bt32._bind_isolated(args)
        source = bound.to_code(helion.Config(block_sizes=[64]))
        ((device, schedule, inverse),) = observed
        assert bound.host_function is not None
        with bound.env, bound.host_function, device:
            schedule.check()
            yield bound, schedule, inverse, source


@pytest.fixture(scope="module", params=("original", "independent"))
def capture_state(request):
    with _capture(request.param) as values:
        return request.param, *values, DeviceFunction.current()


@pytest.fixture
def captured(capture_state):
    strategy, bound, schedule, inverse, source, device = capture_state
    with bound.env, bound.host_function, device:
        yield strategy, bound, schedule, inverse, source


def test_actual_partition_publication_and_serialized_program(captured):
    strategy, bound, schedule, inverse, source = captured
    plans = [
        ast.literal_eval(node.value)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "_helion_cute_wrapper_plans"
            for target in node.targets
        )
    ]
    assert len(plans) == 1 and len(plans[0]) == 1
    plan = plans[0][0]
    assert plan["prepared_warp_tile_program"] == schedule.payload()
    assert plan["prepared_pairwise_program"] == inverse.pairwise.payload()
    assert schedule.partition.specs == (inverse.pairwise.qk,)
    assert schedule.publication.publication is inverse.owner.output_update.lhs
    assert len(schedule.result.bytes()) == 2048
    assert schedule.payload()[0] == (
        (1, 1, 1, 1) if strategy == "original" else (0, 3, 1, 0)
    )
    assert schedule.payload()[5:] == ((10, 128), ((2, None, 16), (1, 2, 16)))
    assert any(
        (lease.begin, lease.end, lease.first_phase, lease.last_phase)
        == (41984, 42500, 0, 2)
        for lease in schedule.interval.leases
    )
    names = {node: node.name for node in schedule.partition.graph.region.graph.nodes}
    try:
        for index, node in enumerate(names):
            node.name = f"anonymous_{index}"
        schedule.check()
    finally:
        for node, name in names.items():
            node.name = name


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "duplicate",
        "foreign_warp",
        "short_k",
        "axis",
        "missing_join",
        "partial_team",
        "forged_team",
        "barrier",
        "result_alias",
        "relocation",
        "live_gate_prefix",
        "fixed_consumer",
        "missing_fixed",
        "omitted_tile",
        "missing_restore_lease",
    ),
)
def test_actual_schedule_rejects_illegal_ownership(captured, mutation):
    strategy, bound, schedule, inverse, source = captured
    if mutation == "missing":
        changed = replace(schedule, tiles=schedule.tiles[:-1])
    elif mutation == "duplicate":
        changed = replace(schedule, tiles=(*schedule.tiles, schedule.tiles[0]))
    elif mutation == "foreign_warp":
        changed = replace(
            schedule, tiles=(replace(schedule.tiles[0], warp=4), *schedule.tiles[1:])
        )
    elif mutation == "short_k":
        changed = replace(
            schedule,
            tiles=(replace(schedule.tiles[0], k_interval=(0, 64)), *schedule.tiles[1:]),
        )
    elif mutation == "axis":
        changed = replace(
            schedule, axes=(schedule.axes[0], replace(schedule.axes[1], scale=8))
        )
    elif mutation == "missing_join":
        changed = replace(schedule, interval=replace(schedule.interval, join_phase=0))
    elif mutation == "partial_team":
        changed = replace(schedule, interval=replace(schedule.interval, teams=4))
    elif mutation == "forged_team":
        changed = replace(
            schedule,
            interval=replace(
                schedule.interval,
                teams=2,
                execution=replace(schedule.interval.execution, threads=320),
            ),
        )
    elif mutation == "barrier":
        changed = replace(
            schedule, interval=replace(schedule.interval, named_barrier_base=9)
        )
    elif mutation == "result_alias":
        changed = replace(
            schedule, result=replace(schedule.result, offset=1024, origin=(0, 0))
        )
    elif mutation == "relocation":
        changed = replace(
            schedule,
            result=replace(schedule.result, offset=schedule.result.offset + 2048),
        )
    elif mutation == "live_gate_prefix":
        changed = replace(
            schedule,
            interval=replace(
                schedule.interval,
                leases=(
                    replace(schedule.interval.leases[0], last_phase=1),
                    *schedule.interval.leases[1:],
                ),
            ),
        )
    elif mutation == "fixed_consumer":
        changed = replace(schedule, fixed=(inverse.owner.output_update,))
    elif mutation == "missing_fixed":
        changed = replace(schedule, fixed=())
    elif mutation == "missing_restore_lease":
        changed = replace(
            schedule,
            interval=replace(
                schedule.interval,
                leases=tuple(
                    lease for lease in schedule.interval.leases if lease.begin != 41984
                ),
            ),
        )
    else:
        tiles = tuple(
            replace(tile, omit_contraction=False) if tile.omit_contraction else tile
            for tile in schedule.tiles
        )
        changed = replace(schedule, tiles=tiles)
    with pytest.raises(chain._UnsupportedChain):
        changed.check()
    schedule.check()


@pytest.mark.parametrize(
    "field,value",
    (
        ("origin", (0, 1)),
        ("row_bytes", 64),
        ("columns_per_tile", 32),
        ("rows_per_tile", 16),
        ("phase_mask", 3),
        ("transposed", True),
    ),
)
def test_only_actual_ordered_k_input_image_is_admitted(captured, field, value):
    strategy, bound, schedule, inverse, source = captured
    changed = replace(
        schedule,
        interval=replace(
            schedule.interval,
            inputs=(
                replace(schedule.interval.inputs[0], **{field: value}),
                schedule.interval.inputs[1],
            ),
        ),
    )
    with pytest.raises(chain._UnsupportedChain):
        changed.check()


@pytest.mark.parametrize("mutation", ("cast", "negative_zero", "nonzero", "mask"))
def test_exact_original_typed_causal_publication(captured, mutation):
    strategy, bound, schedule, inverse, source = captured
    publication = schedule.publication
    cast = publication.publication
    mask, _, zero = cast.args[0].args
    node = cast if mutation == "cast" else mask if mutation == "mask" else zero
    args = node.args
    try:
        if mutation == "cast":
            node.args = (args[0], torch.float16)
        elif mutation == "mask":
            node.args = tuple(reversed(args))
        else:
            node.args = (-0.0 if mutation == "negative_zero" else 1.0,)
        with pytest.raises(chain._UnsupportedChain):
            publication.check()
        # Python structural equality identifies -0.0 with +0.0; the explicit
        # positive-zero publication proof still rejects the changed sign.
        expected = "positive-zero" if mutation == "negative_zero" else "graph changed"
        with pytest.raises(chain._UnsupportedChain, match=expected):
            schedule.check()
    finally:
        node.args = args
    schedule.check()


def test_fresh_real_graph_dependency_cannot_cross_warp_interval():
    with _capture("independent") as (bound, schedule, inverse, source):
        qk, kk = inverse.pairwise.qk, inverse.pairwise.kk
        graph = schedule.partition.graph.region.graph
        with graph.inserting_before(qk.node):
            scalar = graph.call_function(torch.ops.aten.sum.default, (kk.node,))
            scalar.meta["val"] = torch.empty((), dtype=torch.float32)
            changed = graph.call_function(torch.ops.aten.add.Tensor, (qk.lhs, scalar))
            changed.meta = dict(qk.lhs.meta)
        qk.node.args = (changed, qk.rhs, None, qk.requested_out_dtype)
        loop = next(
            item
            for item in bound.host_function.device_ir.cute_semantic_graphs
            if item.graph_id == inverse.owner.region.loop_graph_id
        )
        owner = bind_fast_recurrence(inverse.owner.region, loop)
        pairwise = bind_fast_pairwise(owner)
        fixed = bind_fast_inverse(owner, pairwise)
        owner.graph.check()
        assert any(
            edge.producer is pairwise.kk and edge.consumer is pairwise.qk
            for edge in owner.graph.dependencies
        )
        with pytest.raises(chain._UnsupportedChain, match="dependency"):
            adapter.bind_fast_warp_tiles(pairwise, fixed)


def _executor():
    module = importlib.import_module(
        "helion._compiler.cute.prepared_warp_tile_executor"
    )
    tree = ast.parse(inspect.getsource(module))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_warp_tile_count", "execute_prepared_warp_tiles")
    ]
    for node in functions:
        node.decorator_list = []
    program = ast.Module(
        body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0), *functions],
        type_ignores=[],
    )
    return compile(ast.fix_missing_locations(program), "<actual-warp-tiles>", "exec")


def test_actual_executor_consumes_axes_counts_and_every_warp_lane(captured):
    strategy, bound, schedule, inverse, source = captured
    program = schedule.payload()
    expected = (
        {0: [(0, 0)], 1: [(0, 16)], 2: [(16, 0)], 3: [(16, 16)]}
        if strategy == "original"
        else {0: [], 1: [(0, 16), (16, 0), (16, 16)], 2: [(0, 0)], 3: []}
    )

    def runtime_range(count, *, unroll):
        assert unroll == 1
        return range(count)

    executor = _executor()
    for warp in range(4):
        for lane in range(32):
            trace = []

            def publish(
                smem,
                stage,
                row,
                col,
                actual_lane,
                operand,
                store,
                publication,
                expected_lane=lane,
                events=trace,
            ):
                assert (smem, stage, actual_lane) == ("smem", 41984, expected_lane)
                assert (operand, store, publication) == (program[3], program[4], "cut")
                events.append((row, col))

            namespace = {
                "cutlass": SimpleNamespace(
                    Int32=int,
                    const_expr=bool,
                    range_constexpr=range,
                    range=runtime_range,
                ),
                "publish_prepared_warp_tile": publish,
            }
            exec(executor, namespace)
            namespace["execute_prepared_warp_tiles"](
                "smem", 41984, warp, lane, program, "cut"
            )
            assert trace == expected[warp]
