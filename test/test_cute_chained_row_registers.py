from __future__ import annotations

import ast
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.fx import Graph
from torch.fx import Node

from ._cute_aux import _cpu_codegen
from .test_cute_chained_group_guards import _call
from .test_cute_chained_group_guards import _convert
from .test_cute_chained_group_guards import _dot
from .test_cute_chained_group_guards import _input
from .test_cute_chained_group_guards import _plan
import helion
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_row_registers as rows
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_row_register_emission import RowRegisterEmission
from helion._compiler.cute.chained_row_register_emission import _zero_adds
from helion._compiler.cute.chained_row_registers import KEY
from helion._compiler.cute.chained_row_registers import MAX_ROW_REGISTER_SLOTS
from helion._compiler.cute.chained_row_registers import has_row_register_candidate
from helion._compiler.cute.chained_row_registers import plan_row_registers
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.exc import InvalidConfig
import helion.language as hl
from helion.language import view_ops

if TYPE_CHECKING:
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.device_ir import GraphInfo
    from helion._compiler.generate_ast import GenerateAST


def _selection(graph: Graph, source: Node, position: int, axis: int) -> Node:
    shape = tuple(source.meta["val"].shape)
    n = shape[axis]
    idx = _call(graph, torch.ops.prims.iota.default, (n,), (n,), torch.int32)
    idx.kwargs = {"start": 0, "step": 1, "dtype": torch.int32}
    equal = _call(graph, torch.ops.aten.eq.Scalar, (idx, position), (n,), torch.bool)
    selector = [slice(None), None] if axis == 0 else [None, slice(None)]
    mask = _call(
        graph,
        view_ops.subscript,
        (equal, selector),
        (n, 1) if axis == 0 else (1, n),
        torch.bool,
    )
    zero = _call(graph, torch.ops.aten.scalar_tensor.default, (0.0,), (), torch.float32)
    selected = _call(
        graph, torch.ops.aten.where.self, (mask, source, zero), shape, torch.float32
    )
    return _call(
        graph,
        torch.ops.aten.sum.dim_IntList,
        (selected, [axis], False),
        (shape[1 - axis],),
        torch.float32,
    )


def _fixture(
    rows: int = 16, columns: int = 16, *, fanout: bool = False
) -> tuple[ChainedMatmulPlan, list[Node], list[Node]]:
    graph = Graph()
    left = _input(graph, "left", (rows, 16), torch.bfloat16)
    right = _input(graph, "right", (16, columns), torch.bfloat16)
    original = _dot(graph, left, right)
    state = original
    old = []
    reductions = []
    for position in range(1, 4):
        row = _selection(graph, state, position, 0)
        column = _selection(graph, original, position, 1)
        reductions.extend((row, column))
        rb = _call(
            graph,
            view_ops.subscript,
            (row, [None, slice(None)]),
            (1, columns),
            torch.float32,
        )
        cb = _call(
            graph,
            view_ops.subscript,
            (column, [slice(None), None]),
            (rows, 1),
            torch.float32,
        )
        product = _call(
            graph, torch.ops.aten.mul.Tensor, (rb, cb), (rows, columns), torch.float32
        )
        state = _call(
            graph,
            torch.ops.aten.add.Tensor,
            (state, product),
            (rows, columns),
            torch.float32,
        )
        old.append(state)
    rhs = _input(graph, "rhs", (columns, 32), torch.bfloat16)
    final = _dot(graph, _convert(graph, state, torch.bfloat16), rhs)
    plan = _plan(graph, (final, old[0]) if fanout else (final,))
    return plan, reductions, old


@pytest.mark.parametrize("shape", [(8, 8), (8, 16), (16, 8), (16, 16), (32, 32)])
def test_closed_versioned_rows_have_complete_nonoverlapping_images(
    shape: tuple[int, int],
) -> None:
    plan, _, _ = _fixture(*shape)
    proposal = plan_row_registers(plan)
    assert len(proposal.islands) == 1
    proposal.check(plan)
    island = proposal.islands[0]
    assert island.shape == shape
    assert len(island.reductions) == 6
    assert island.register_slots <= MAX_ROW_REGISTER_SLOTS
    slots = {event.node: event for event in island.events}
    for index, event in enumerate(island.events):
        for source in event.reads:
            old = slots[source]
            assert old.last_use >= index
            assert (
                event.offset + event.size <= old.offset
                or old.offset + old.size <= event.offset
            )
    assert {
        (lane, col) for lane in range(32) if lane < shape[0] for col in range(shape[1])
    } == {(r, c) for r in range(shape[0]) for c in range(shape[1])}


@pytest.mark.parametrize(
    "change", ["fanout", "extent", "dynamic", "negative", "zero", "unknown", "cache"]
)
def test_incomplete_or_unknown_components_decline(change: str) -> None:
    plan, reductions, old = _fixture(fanout=change == "fanout")
    if change == "extent":
        plan, reductions, old = _fixture(64, 16)
    elif change in ("dynamic", "negative", "zero"):
        for reduction in reductions:
            selected = reduction.args[0]
            assert isinstance(selected, Node)
            mask, _, zero = selected.args
            assert isinstance(mask, Node) and isinstance(zero, Node)
            equality = mask.args[0]
            assert isinstance(equality, Node)
            if change == "zero":
                zero.args = (-0.0,)
            else:
                equality.args = (
                    equality.args[0],
                    equality.args[0] if change == "dynamic" else -1,
                )
    elif change == "unknown":
        old[0].target = torch.ops.aten.flip.default
    elif change == "cache":
        from helion._compiler.cute.chained_pointwise_residency import PointwiseCachePlan

        plan = replace(plan, pointwise_cache=PointwiseCachePlan())
    assert not plan_row_registers(plan).islands


@pytest.mark.parametrize(
    "extent,expected", [(1, False), (8, True), (32, True), (64, False)]
)
def test_discovery_requires_repeated_full_small_panels(
    extent: int, expected: bool
) -> None:
    plan, _, _ = _fixture(extent, extent)
    graph = SimpleNamespace(loop=None, region=plan.region)
    # Static FX fixtures need no active compile context or implicit GPU query.
    env = cast("CompileEnvironment", object())
    with patch.object(chain, "_classify_chained_graph", return_value=graph):
        assert has_row_register_candidate([], env) is expected
        graph.loop = object()
        assert not has_row_register_candidate([], env)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_panel_discovery_does_not_claim_fp32_rows(dtype: torch.dtype) -> None:
    plan, reductions, _ = _fixture()
    for reduction in reductions:
        source = cast("Node", reduction.args[0])
        source.meta["val"] = source.meta["val"].to(dtype)
    graph = SimpleNamespace(loop=None, region=plan.region)
    with patch.object(chain, "_classify_chained_graph", return_value=graph):
        assert not has_row_register_candidate([], cast("CompileEnvironment", object()))
    assert not plan_row_registers(plan).islands


@pytest.mark.parametrize("store", [False, True])
def test_independent_effect_before_publication_cannot_be_crossed(store: bool) -> None:
    from helion.language import memory_ops

    plan, reductions, _ = _fixture()
    assert plan.region is not None
    graph = plan.region.graph
    source = next(node for node in graph.nodes if node.op == "placeholder")
    with graph.inserting_after(reductions[0]):
        if store:
            effect = graph.call_function(
                memory_ops.store, (source, [slice(None), slice(None)], source)
            )
        else:
            effect = graph.call_function(torch.ops.aten.add_.Tensor, (source, 1.0))
        effect.meta["val"] = source.meta["val"]
    changed = replace(plan, region=replace(plan.region, nodes=tuple(graph.nodes)))
    assert not plan_row_registers(changed).islands


def test_register_payload_limit_and_fewer_collectives_decline() -> None:
    plan, _, _ = _fixture()
    assert plan_row_registers(plan).islands[0].register_slots == 65
    with patch.object(rows, "MAX_ROW_REGISTER_SLOTS", 64):
        assert not plan_row_registers(plan).islands
    assert plan.region is not None
    graph = SimpleNamespace(loop=None, region=replace(plan.region, reductions=()))
    with patch.object(chain, "_classify_chained_graph", return_value=graph):
        assert not has_row_register_candidate([], cast("CompileEnvironment", object()))


@pytest.mark.parametrize(
    "change", ["users", "dtype", "position", "offset", "last_use", "register_slots"]
)
def test_graph_and_physical_record_mutations_invalidate(change: str) -> None:
    plan, reductions, old = _fixture()
    proposal = plan_row_registers(plan)
    island = proposal.islands[0]
    if change == "users":
        assert plan.region is not None
        plan.region.graph.output((old[0],))
    elif change == "dtype":
        old[0].meta["val"] = old[0].meta["val"].to(torch.float16)
    elif change == "position":
        reduction = island.events[1].reduction
        assert reduction is not None
        object.__setattr__(reduction, "position", 31)
    elif change == "register_slots":
        object.__setattr__(island, change, 1)
    else:
        object.__setattr__(island.events[0], change, 99)
    with pytest.raises(chain._UnsupportedChain, match="changed"):
        proposal.check(plan)


@pytest.mark.parametrize("failure", [False, True])
def test_publication_requires_completed_first_collective(failure: bool) -> None:
    plan, _, _ = _fixture()
    proposal = plan_row_registers(plan)
    island = proposal.islands[0]
    emitter = RowRegisterEmission(proposal)
    cg = cast("GenerateAST", None)
    bindings: dict[Node, str] = {}
    execution = ChainedExecution(plan.threads)
    with pytest.raises(chain._UnsupportedChain, match="precedes"):
        emitter.emit_collective(cg, plan, bindings, island.reductions[1], execution)
    with pytest.raises(chain._UnsupportedChain, match="not published"):
        emitter.validate(plan)
    with patch(
        "helion._compiler.cute.chained_row_register_emission._emit_island",
        side_effect=chain._UnsupportedChain("failed typed recipe") if failure else None,
        return_value=["original_typed_body", execution.sync],
    ):
        if failure:
            with pytest.raises(chain._UnsupportedChain, match="failed typed"):
                emitter.emit_collective(
                    cg, plan, bindings, island.first_collective, execution
                )
            assert not bindings and not emitter.completed
            return
        assert emitter.emit_collective(
            cg, plan, bindings, island.first_collective, execution
        ) == ["original_typed_body", execution.sync]
    assert bindings == {island.output: island.name}
    for node in island.reductions[1:]:
        assert emitter.emit_collective(cg, plan, bindings, node, execution) == []
    emitter.validate(plan)
    with pytest.raises(chain._UnsupportedChain, match="duplicate"):
        emitter.emit_collective(cg, plan, bindings, island.first_collective, execution)


@pytest.mark.parametrize("position", [0, 1, 7, 15, 31])
def test_owned_shuffle_preserves_original_six_additions_and_poison(
    position: int,
) -> None:
    plan, _, _ = _fixture(32, 32)
    proof = plan_row_registers(plan).islands[0].events[1].reduction
    assert proof is not None
    proof = replace(
        proof,
        position=position,
        zero_sides=tuple(
            "left" if position & step else "right" for step in (16, 8, 4, 2, 1)
        ),
    )
    values = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1.125], dtype=np.float32)
    owners = np.full((32, values.size), np.float32(np.nan))
    owners[position] = values
    selected = np.where(np.arange(32)[:, None] == position, owners, np.float32(0))
    original = np.float32(0) + selected
    for offset in (16, 8, 4, 2, 1):
        previous = original.copy()
        for lane in range(32):
            peer = lane + offset
            original[lane] = previous[lane] + (
                previous[peer] if peer < 32 else previous[lane]
            )
    statements = _zero_adds("value", "acc", proof.zero_sides)
    namespace = {
        "cutlass": SimpleNamespace(Float32=np.float32),
        "value": owners[position],
    }
    exec("\n".join(statements), namespace)
    actual = namespace["acc"]
    np.testing.assert_array_equal(actual.view(np.uint32), original[0].view(np.uint32))
    assert len(statements) == 6


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sequence(
    a: torch.Tensor,
    b: torch.Tensor,
    rhs: torch.Tensor,
    triangular: hl.constexpr,
) -> torch.Tensor:
    output = torch.empty((a.shape[0], 16, 32), device=a.device, dtype=a.dtype)
    for batch in hl.tile(a.shape[0], block_size=1):
        base = hl.dot(a[batch.begin, :, :], b[batch.begin, :, :])
        state = base
        index = hl.arange(16)
        for position in hl.static_range(1, 4):
            row = torch.where(index[:, None] == position, state, 0.0).sum(0)
            column = torch.where(index[None, :] == position, base, 0.0).sum(1)
            state = torch.tanh(state * 0.5 + column[:, None] * row[None, :] + 0.0625)
            if triangular:
                state = torch.where(index[:, None] >= index[None, :], state, 0.0)
        output[batch.begin, :, :] = hl.dot(
            state.to(a.dtype), rhs[batch.begin, :, :]
        ).to(a.dtype)
    return output


def _args(dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(shape, dtype=dtype)
        for shape in ((3, 16, 16), (3, 16, 16), (3, 16, 32))
    )


def _config(mode: str = "shared", **extra: object) -> helion.Config:
    return helion.Config.from_dict(
        {"num_warps": 4, "cute_chained_mma_schedule": "coalesced", KEY: mode, **extra}
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_actual_source_default_seed_and_register_publication(
    dtype: torch.dtype,
) -> None:
    plans = []
    original = chain.plan_chained_matmul

    def observe(graphs: list[GraphInfo]) -> ChainedMatmulPlan | None:
        plan = original(graphs)
        if plan is not None:
            plans.append(plan)
        return plan

    with _cpu_codegen(), patch.object(chain, "plan_chained_matmul", observe):
        bound = _sequence._bind_isolated((*_args(dtype), False))
        spec = bound.config_spec
        off = bound.to_code(_config())
        assert off == bound.to_code(
            helion.Config(num_warps=4, cute_chained_mma_schedule="coalesced")
        )
        on = bound.to_code(_config("warp_rows"))
        seeds = spec.compiler_seed_configs
        added = [s for s in seeds if s.config.get(KEY) == "warp_rows"]
        assert len(added) == 1
        assert seeds[-1].config["cute_chained_mma_schedule"] == "cp_async"
        assert [s.config.get(KEY) for s in seeds[-3:-1]] == [
            "warp_rows",
            "warp_rows_parallel",
        ]
        base = [s for s in seeds if KEY not in s]
        with patch.dict(bound.env.settings.autotune_config_overrides, {KEY: "shared"}):
            assert (
                CuteChainedMatmulHeuristic._with_row_register_seed(bound.env, base)
                == base
            )
        for mode in ("shared", "warp_rows"):
            config = spec.normalized_config(_config(mode))
            generation = spec.create_config_generation()
            assert generation.unflatten(generation.flatten(config)) == config
    assert plans[-1].row_registers is not None
    assert "shuffle_sync" in on and "_values = cute.make_rmem_tensor" in on
    assert on.count("sync_threads()") == off.count("sync_threads()") - 5

    def mma_calls(source: str) -> list[str]:
        return [
            ast.dump(node)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) in ("cute.gemm", "execute_prepared_warp_k")
        ]

    assert mma_calls(on) == mma_calls(off)
    assert len(mma_calls(on)) == 2
    ast.parse(on)


def test_explicit_invalid_or_competing_schedule_is_not_erased() -> None:
    with _cpu_codegen():
        spec = _sequence._bind_isolated((*_args(), False)).config_spec
        for mode in (True, 1, None, "register", "WARP_ROWS"):
            with pytest.raises(InvalidConfig):
                spec.normalized_config({"num_warps": 4, KEY: mode})
        for extra in (
            {"cute_chained_mma_schedule": "tcgen05_tmem"},
            {"cute_chained_pointwise_cache_bytes": 1024},
            {"cute_chained_typed_operands": True},
            {"cute_chained_fragment_epilogues": True},
        ):
            with pytest.raises(InvalidConfig):
                spec.normalized_config(_config("warp_rows", **extra))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("triangular", [False, True])
@skipUnlessBackends(["cute"])
def test_gpu_row_images_match_original_and_independent_recurrence(
    dtype: torch.dtype,
    triangular: bool,
) -> None:
    torch.manual_seed(2301)
    args = tuple(
        torch.randn(x.shape, device=DEVICE, dtype=dtype) * 0.125 for x in _args(dtype)
    )
    saved = tuple(x.clone() for x in args)
    expected = []
    for batch in range(3):
        a, b, rhs = (x[batch].double() for x in args)
        base = a @ b
        state = base
        for position in range(1, 4):
            state = torch.tanh(
                state * 0.5
                + base[:, position, None] * state[position, None, :]
                + 0.0625
            )
            if triangular:
                state = torch.tril(state)
        expected.append(state.to(dtype).double() @ rhs)
    oracle = torch.stack(expected).to(dtype)
    results = []
    for mode in ("shared", "warp_rows"):
        bound = _sequence._bind_isolated((*args, triangular))
        bound.set_config(_config(mode))
        actual = bound(*args, triangular)
        torch.testing.assert_close(actual, oracle, atol=0.002, rtol=0.01)
        results.append(actual.clone())
    torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)
    for original, actual in zip(saved, args, strict=True):
        torch.testing.assert_close(actual, original, atol=0, rtol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _poisoned_sequence(
    a: torch.Tensor, b: torch.Tensor, rhs: torch.Tensor
) -> torch.Tensor:
    output = torch.empty((a.shape[0], 16, 32), device=a.device, dtype=a.dtype)
    for batch in hl.tile(a.shape[0], block_size=1):
        base = hl.dot(a[batch.begin, :, :], b[batch.begin, :, :])
        index = hl.arange(16)
        state = torch.where(index[:, None] < 4, base, float("nan"))
        for position in hl.static_range(1, 4):
            row = torch.where(index[:, None] == position, state, 0.0).sum(0)
            column = torch.where(index[None, :] == position, base, 0.0).sum(1)
            state = torch.tanh(state * 0.5 + column[:, None] * row[None, :] + 0.0625)
        state = torch.where(index[:, None] < 4, state, 0.0)
        output[batch.begin, :, :] = hl.dot(
            state.to(a.dtype), rhs[batch.begin, :, :]
        ).to(a.dtype)
    return output


def test_poisoned_unselected_rows_retain_register_codegen() -> None:
    with _cpu_codegen():
        bound = _poisoned_sequence._bind_isolated(_args(torch.float16))
        source = bound.to_code(_config("warp_rows"))
    assert "chain_row_register" in source and "shuffle_sync" in source
    assert "nan" in source
    ast.parse(source)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@skipUnlessBackends(["cute"])
def test_gpu_poisoned_unselected_rows_do_not_enter_selected_updates(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(1703)
    args = tuple(
        torch.randint(-2, 3, value.shape, device=DEVICE).to(dtype) / 16
        for value in _args(dtype)
    )
    saved = tuple(value.clone() for value in args)
    expected = []
    for batch in range(3):
        a, b, rhs = (value[batch].double() for value in args)
        base = a @ b
        state = base.clone()
        state[4:, :] = torch.nan
        for position in range(1, 4):
            state = torch.tanh(
                state * 0.5
                + base[:, position, None] * state[position, None, :]
                + 0.0625
            )
        state[4:, :] = 0
        expected.append(state.to(dtype).double() @ rhs)
    oracle = torch.stack(expected).to(dtype)
    assert torch.isfinite(oracle).all()
    results = []
    for mode in ("shared", "warp_rows"):
        bound = _poisoned_sequence._bind_isolated(args)
        bound.set_config(_config(mode))
        actual = bound(*args)
        torch.testing.assert_close(actual, oracle, atol=0.002, rtol=0.01)
        results.append(actual.clone())
    torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)
    for original, actual in zip(saved, args, strict=True):
        torch.testing.assert_close(actual, original, atol=0, rtol=0)
