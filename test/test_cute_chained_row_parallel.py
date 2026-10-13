from __future__ import annotations

import ast
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

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
from .test_cute_chained_row_registers import _config
from .test_cute_chained_row_registers import _selection
import helion
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_row_parallel import RowParallelEmission
from helion._compiler.cute.chained_row_parallel import plan_row_parallel
from helion._compiler.cute.chained_row_registers import KEY
from helion._compiler.cute.chained_row_registers import plan_row_registers
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language import view_ops

if TYPE_CHECKING:
    from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
    from helion._compiler.generate_ast import GenerateAST


def _island(graph: Graph, base: Node, factor: float) -> Node:
    shape = tuple(base.meta["val"].shape)
    state = _call(
        graph, torch.ops.aten.mul.Scalar, (base, factor), shape, torch.float32
    )
    original = state
    for position in (1, 2):
        row = _selection(graph, state, position, 0)
        col = _selection(graph, original, position, 1)
        rb = _call(
            graph,
            view_ops.subscript,
            (row, [None, slice(None)]),
            (1, shape[1]),
            torch.float32,
        )
        cb = _call(
            graph,
            view_ops.subscript,
            (col, [slice(None), None]),
            (shape[0], 1),
            torch.float32,
        )
        product = _call(
            graph, torch.ops.aten.mul.Tensor, (rb, cb), shape, torch.float32
        )
        state = _call(
            graph, torch.ops.aten.add.Tensor, (state, product), shape, torch.float32
        )
    return state


def _fixture(
    count: int = 4, *, future: bool = False, shape: tuple[int, int] = (16, 16)
) -> ChainedMatmulPlan:
    graph = Graph()
    left = _input(graph, "left", (shape[0], 16), torch.bfloat16)
    right = _input(graph, "right", (16, shape[1]), torch.bfloat16)
    rhs = _input(graph, "rhs", (shape[1], 32), torch.bfloat16)
    base = _dot(graph, left, right)
    outputs = []
    for index in range(count):
        if future and index == count - 1:
            base = _dot(graph, left, right)
        state = _island(graph, base, 1 + index / 8)
        outputs.append(_dot(graph, _convert(graph, state, torch.bfloat16), rhs))
    plan = _plan(graph, tuple(outputs))
    return replace(plan, row_registers=plan_row_registers(plan))


@pytest.mark.parametrize("count,sizes", [(2, [2]), (4, [4]), (5, [4, 1])])
@pytest.mark.parametrize("shape", [(8, 16), (16, 8), (16, 16), (32, 32)])
def test_common_readiness_batches_original_disjoint_images(
    count: int, sizes: list[int], shape: tuple[int, int]
) -> None:
    plan = _fixture(count, shape=shape)
    proof = plan_row_parallel(plan)
    assert proof is not None
    proof.check(plan)
    assert [len(batch.islands) for batch in proof.batches] == sizes
    assert plan.row_registers is not None
    assert (
        tuple(item.island for batch in proof.batches for item in batch.islands)
        == plan.row_registers.islands
    )
    assert all(
        item.materialized == (plan.dots[0],)
        for batch in proof.batches
        for item in batch.islands
    )
    for batch in proof.batches:
        assert len({item.island.name for item in batch.islands}) == len(batch.islands)
        assert len(batch.islands) <= plan.threads // 32
    assert any(node in plan.dots for node in proof.batches[0].crossed)


def test_future_source_stays_at_original_event_and_singleton_is_rejected() -> None:
    plan = _fixture(future=True)
    proof = plan_row_parallel(plan)
    assert proof is not None
    assert [len(batch.islands) for batch in proof.batches] == [3, 1]
    assert proof.batches[1].event is proof.batches[1].islands[0].island.first_collective
    assert plan_row_parallel(_fixture(1)) is None
    assert plan_row_parallel(replace(plan, threads=33)) is None
    assert plan_row_parallel(replace(plan, threads=32)) is None


@pytest.mark.parametrize("effect", ["unknown", "store", "mutable"])
def test_movement_rejects_unknown_or_mutating_initializer(effect: str) -> None:
    from helion.language import memory_ops

    plan = _fixture()
    assert plan.region is not None and plan.row_registers is not None
    graph = plan.region.graph
    original = plan.dots[0]
    with graph.inserting_after(original):
        target = {
            "unknown": torch.ops.aten.flip.default,
            "store": memory_ops.store,
            "mutable": torch.ops.aten.add_.Tensor,
        }[effect]
        node = graph.call_function(target, (original, 1))
        node.meta["val"] = original.meta["val"]
    changed = replace(plan, region=replace(plan.region, nodes=tuple(graph.nodes)))
    changed = replace(changed, row_registers=plan_row_registers(changed))
    assert plan_row_parallel(changed) is None


@pytest.mark.parametrize("change", ["graph", "storage", "batch", "read"])
def test_stale_graph_storage_and_movement_records_reject(change: str) -> None:
    plan = _fixture()
    proof = plan_row_parallel(plan)
    assert proof is not None
    if change == "graph":
        plan.dots[0].args = (*plan.dots[0].args[:2], plan.dots[1], torch.float32)
    elif change == "storage":
        object.__setattr__(plan, "root_workspace", object())
    elif change == "batch":
        object.__setattr__(proof.batches[0], "event", plan.dots[1])
    else:
        object.__setattr__(proof.batches[0].islands[0], "materialized", ())
    with pytest.raises(chain._UnsupportedChain, match="changed"):
        proof.check(plan)


def _emission() -> tuple[ChainedMatmulPlan, RowParallelEmission, dict[Node, str]]:
    plan = _fixture()
    proof = plan_row_parallel(plan)
    assert proof is not None
    return plan, RowParallelEmission(proof), {plan.dots[0]: "original_c"}


@pytest.mark.parametrize(
    "kind", ["name_only", "no_join", "foreign", "binding", "reorder", "failed_recipe"]
)
def test_only_original_joined_same_attempt_inputs_grant_readiness(kind: str) -> None:
    plan, emission, boundaries = _emission()
    execution = ChainedExecution(plan.threads)
    first = emission.parallel.batches[0].event
    assert emission.parallel.events[:2] == (plan.dots[0], first)
    if kind == "reorder":
        with pytest.raises(chain._UnsupportedChain, match="order"):
            emission.emit_collective(
                cast("GenerateAST", None), plan, boundaries, first, execution
            )
        return
    if kind == "no_join":
        with pytest.raises(chain._UnsupportedChain, match="joined phase"):
            emission.complete_original(
                plan, plan.dots[0], "original_c", ("publication",), execution
            )
        assert not emission.completed
        return
    emission.complete_original(
        plan, plan.dots[0], "original_c", ("publication", execution.sync), execution
    )
    if kind == "name_only":
        emission.completed.clear()
    elif kind == "foreign":
        emission.completed[plan.dots[0]] = replace(
            emission.completed[plan.dots[0]], owner=object()
        )
    elif kind == "binding":
        boundaries[plan.dots[0]] = "reused_slot"
    with (
        patch(
            "helion._compiler.cute.chained_row_register_emission._emit_island",
            side_effect=chain._UnsupportedChain("failed recipe"),
        ),
        pytest.raises(
            chain._UnsupportedChain,
            match="failed recipe" if kind == "failed_recipe" else "not completed",
        ),
    ):
        emission.emit_collective(
            cast("GenerateAST", None), plan, boundaries, first, execution
        )
    assert not emission.batches
    assert len(boundaries) == 1


def test_emission_accounts_every_original_event_and_one_join_per_batch() -> None:
    plan, emission, boundaries = _emission()
    execution = ChainedExecution(plan.threads)
    emitted = []
    with patch(
        "helion._compiler.cute.chained_row_register_emission._emit_island",
        side_effect=lambda cg, p, b, island, e, *, warp: [
            f"warp_{warp}_original_{island.name}"
        ],
    ):
        for node in emission.parallel.events:
            if node in plan.dots:
                boundaries[node] = node.name
                emission.complete_original(
                    plan, node, node.name, ("store", execution.sync), execution
                )
            else:
                phase = emission.emit_collective(
                    cast("GenerateAST", None), plan, boundaries, node, execution
                )
                assert phase is not None
                emitted.extend(phase)
    emission.validate(plan)
    assert emitted.count(execution.sync) == 1
    assert len(emission.batches) == 1 and len(emission.batches[0].batch.islands) == 4
    with pytest.raises(chain._UnsupportedChain, match="order"):
        emission.emit_collective(
            cast("GenerateAST", None),
            plan,
            boundaries,
            emission.parallel.batches[0].event,
            execution,
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _parallel_sequence(
    a: torch.Tensor,
    b: torch.Tensor,
    rhs: torch.Tensor,
    count: hl.constexpr,
    triangular: hl.constexpr,
) -> torch.Tensor:
    output = torch.empty((a.size(0), 16, 32), device=a.device, dtype=a.dtype)
    for batch in hl.tile(a.size(0), block_size=1):
        base = hl.dot(a[batch.begin, :, :], b[batch.begin, :, :])
        index = hl.arange(16)
        total = hl.full([16, 32], 0, torch.float32)
        for island in hl.static_range(cast("int", count)):
            original = base * (1 + island / 8)
            state = original
            for position in hl.static_range(1, 3):
                row = torch.where(index[:, None] == position, state, 0.0).sum(0)
                col = torch.where(index[None, :] == position, original, 0.0).sum(1)
                state = torch.tanh(state * 0.5 + col[:, None] * row[None, :] + 0.0625)
                if triangular:
                    state = torch.where(index[:, None] >= index[None, :], state, 0.0)
            total = hl.dot(state.to(a.dtype), rhs[batch.begin, :, :], acc=total)
        output[batch.begin, :, :] = total.to(a.dtype)
    return output


def _args(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(shape, dtype=dtype)
        for shape in ((3, 16, 16), (3, 16, 16), (3, 16, 32))
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_real_source_preserves_math_old_modes_and_complete_seed_prefix(
    dtype: torch.dtype,
) -> None:
    plans = []
    original = chain.plan_chained_matmul

    def observe(graphs):
        plan = original(graphs)
        if plan is not None:
            plans.append(plan)
        return plan

    with _cpu_codegen(), patch.object(chain, "plan_chained_matmul", observe):
        bound = _parallel_sequence._bind_isolated((*_args(dtype), 4, True))
        shared = bound.to_code(_config())
        serial = bound.to_code(_config("warp_rows"))
        parallel = bound.to_code(_config("warp_rows_parallel"))
        assert shared == bound.to_code(
            helion.Config(num_warps=4, cute_chained_mma_schedule="coalesced")
        )
        seeds = bound.config_spec.compiler_seed_configs
        assert seeds[-1].config["cute_chained_mma_schedule"] == "cp_async"
        assert (
            CuteChainedMatmulHeuristic._with_row_async_copy_seed(bound.env, seeds[:-1])
            == seeds
        )
        # The new copy choice follows the entire previous seed sequence.
        seeds = seeds[:-1]
        added = [s for s in seeds if s.config.get(KEY) == "warp_rows_parallel"]
        assert len(added) == 1 and seeds[-1] is added[0]
        old = seeds[:-1]
        assert (
            CuteChainedMatmulHeuristic._with_parallel_row_seed(bound.env, old) == seeds
        )
        for mode in ("shared", "warp_rows"):
            with patch.dict(bound.env.settings.autotune_config_overrides, {KEY: mode}):
                assert (
                    CuteChainedMatmulHeuristic._with_parallel_row_seed(bound.env, old)
                    == old
                )
        for mode in ("shared", "warp_rows", "warp_rows_parallel"):
            config = bound.config_spec.normalized_config(_config(mode))
            generation = bound.config_spec.create_config_generation()
            assert generation.unflatten(generation.flatten(config)) == config
    accepted = next(plan for plan in plans if plan.row_parallel is not None)
    assert accepted.row_parallel is not None
    assert [len(batch.islands) for batch in accepted.row_parallel.batches] == [4]
    assert parallel.count("sync_threads()") == serial.count("sync_threads()") - 3

    def calls(source):
        return [
            ast.dump(node)
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and ast.unparse(node.func)
            in ("cute.gemm", "execute_prepared_warp_k", "cute.arch.shuffle_sync")
        ]

    assert calls(parallel) == calls(serial)
    assert parallel.count("cute.arch.alloc_smem") == serial.count(
        "cute.arch.alloc_smem"
    )

    def bodies(source: str) -> dict[str, str]:
        result = {}
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.If) or not node.body:
                continue
            first = node.body[0]
            if (
                isinstance(first, ast.Assign)
                and isinstance(first.targets[0], ast.Name)
                and first.targets[0].id.startswith("chain_row_register_")
                and first.targets[0].id.endswith("_row")
            ):
                # Only the proven warp origin may change. Everything after
                # it, including zero-adds/masks/casts/shuffles, is exact.
                first.value = ast.Name(id="chain_thread", ctx=ast.Load())
                key = first.targets[0].id
                body = ast.Module(body=node.body, type_ignores=[])
                # Typed lowering allocates fresh local temporary names in
                # emission order. Normalize only definitions inside this
                # closed body; input pointers and shared bindings stay exact.
                locals_: dict[str, str] = {}
                for name in ast.walk(body):
                    if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store):
                        locals_.setdefault(name.id, f"local_{len(locals_)}")
                for name in ast.walk(body):
                    if isinstance(name, ast.Name) and name.id in locals_:
                        name.id = locals_[name.id]
                result[key] = ast.dump(body)
        return result

    assert len(bodies(parallel)) == 4
    assert bodies(parallel) == bodies(serial)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("triangular", [False, True])
@pytest.mark.parametrize("schedule", ["coalesced", "cp_async"])
@skipUnlessBackends(["cute"])
def test_gpu_parallel_islands_match_original_and_double_reference(
    dtype: torch.dtype, triangular: bool, schedule: str
) -> None:
    torch.manual_seed(4137)
    shapes = (
        ((3, 16, 64), (3, 64, 16), (3, 16, 32))
        if schedule == "cp_async"
        else tuple(x.shape for x in _args(dtype))
    )
    args = tuple(
        torch.randn(shape, device=DEVICE, dtype=dtype) * 0.125 for shape in shapes
    )
    guarded = []
    if triangular and schedule == "cp_async":
        # A two-byte offset must use the original scalar fallback. Preserve
        # sentinels around each actual input view as well as its contents.
        views = []
        for value in args:
            storage = torch.full((value.numel() + 2,), 17, device=DEVICE, dtype=dtype)
            view = storage[1:-1].view(value.shape)
            view.copy_(value)
            guarded.append(storage)
            views.append(view)
        args = tuple(views)
    snapshots = tuple(x.clone() for x in args)
    expected = []
    for batch in range(3):
        a, b, rhs = (x[batch].double() for x in args)
        base = a @ b
        total = torch.zeros((16, 32), device=DEVICE, dtype=torch.float64)
        for island in range(4):
            initial = base * (1 + island / 8)
            state = initial
            for position in (1, 2):
                state = torch.tanh(
                    state * 0.5
                    + initial[:, position, None] * state[position, None, :]
                    + 0.0625
                )
                if triangular:
                    state = state.tril()
            total += state.to(dtype).double() @ rhs
        expected.append(total.to(dtype))
    results = []
    for mode in ("warp_rows", "warp_rows_parallel"):
        bound = _parallel_sequence._bind_isolated((*args, 4, triangular))
        config = _config(mode, cute_chained_mma_schedule=schedule)
        if schedule == "cp_async":
            config.config["cute_native_matmul_metadata"] = True
            assert "CopyG2SOp(" in bound.to_code(config)
        bound.set_config(config)
        result = bound(*args, 4, triangular)
        torch.testing.assert_close(result, torch.stack(expected), atol=0.02, rtol=0.02)
        results.append(result)
    torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)
    for value, snapshot in zip(args, snapshots, strict=True):
        torch.testing.assert_close(value, snapshot, atol=0, rtol=0)
    for storage in guarded:
        assert storage[0] == 17 and storage[-1] == 17
