from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_residency_search import _root_residency
import helion
from helion._compiler.cute.chained_preparation_cut import plan_preparation_cut
from helion._compiler.cute.contraction_region import _domain
from helion._testing import import_path
import helion.language as hl
from helion.language import memory_ops
from helion.language.matmul_ops import dot

if TYPE_CHECKING:
    from helion._compiler.cute.chained_preparation_cut import PreparationCut
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _typed_sequence(a, b, c, initial, recurrent_first: hl.constexpr):
    steps, size, _ = a.shape
    history = torch.empty((steps, size, size), dtype=torch.float32, device=a.device)
    final = torch.empty_like(initial)
    for rows in hl.tile(size, block_size=16):
        state = initial[rows, :]
        for step in hl.tile(steps, block_size=1):
            cols = hl.arange(size)
            kk = hl.arange(size)
            operand = c[step.id, kk, cols].to(torch.float16)
            projected = state
            if recurrent_first:
                projected = hl.dot(
                    state.to(torch.float16), operand, out_dtype=torch.float32
                )
            first = hl.dot(
                a[step.id, rows, kk], b[step.id, kk, cols], out_dtype=torch.float32
            )
            rounded = first.to(torch.float16)
            snapshot = hl.dot(rounded, operand, out_dtype=torch.float32).to(
                torch.bfloat16
            )
            if not recurrent_first:
                projected = hl.dot(
                    state.to(torch.float16), operand, out_dtype=torch.float32
                )
            state = hl.dot(
                snapshot,
                c[step.id, kk, cols],
                acc=projected + state * 0.5 + rounded.float(),
                out_dtype=torch.float32,
            )
            history[step.id, rows, cols] = state
        final[rows, :] = state
    return history, final


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _dependency_sequence(a, b, initial, dependency: hl.constexpr):
    steps, size, _ = a.shape
    history = torch.empty((steps, size, size), dtype=torch.float32, device=a.device)
    final = torch.empty_like(initial)
    for rows in hl.tile(size, block_size=16):
        state = initial[rows, :]
        for step in hl.tile(steps, block_size=1):
            cols = hl.arange(size)
            kk = hl.arange(size)
            if dependency == 0:
                left = hl.load(
                    a, [step.id, rows, kk], extra_mask=(state[:, 0] > 0)[:, None]
                )
                first = hl.dot(left, b[step.id, kk, cols], out_dtype=torch.float32)
            elif dependency == 1:
                indices = state[:, 0].to(torch.int32) % size
                left = hl.load(a, [step.id, indices, kk])
                first = hl.dot(left, b[step.id, kk, cols], out_dtype=torch.float32)
            else:
                first = hl.dot(
                    a[step.id, rows, kk],
                    b[step.id, kk, cols],
                    acc=state,
                    out_dtype=torch.float32,
                )
            independent = hl.dot(
                a[step.id, rows, kk], b[step.id, kk, cols], out_dtype=torch.float32
            )
            state = first + independent
            history[step.id, rows, cols] = state
        final[rows, :] = state
    return history, final


def _inputs(extra: object, *, typed: bool) -> tuple[object, ...]:
    tensors = tuple(
        torch.empty((3, 16, 16), dtype=torch.bfloat16) for _ in range(3 if typed else 2)
    )
    return (*tensors, torch.empty((16, 16), dtype=torch.float32), extra)


def _runtime_values(kernel: Kernel, args: tuple[object, ...]) -> dict[str, object]:
    return dict(inspect.signature(kernel.fn).bind(*args).arguments)


def _cut(
    bound: BoundKernel, kernel: Kernel, args: tuple[object, ...]
) -> PreparationCut | None:
    assert bound.host_function is not None
    with (
        bound.env,
        bound.host_function,
        bound.env.use_runtime_arg_values(_runtime_values(kernel, args)),
    ):
        return plan_preparation_cut(bound.host_function.device_ir.graphs)


@pytest.mark.parametrize("recurrent_first", [False, True])
def test_full_dag_partition_retains_typed_frontier(recurrent_first: bool) -> None:
    args = _inputs(recurrent_first, typed=True)
    with _cpu_codegen():
        bound = _typed_sequence._bind_isolated(args)
        cut = _cut(bound, _typed_sequence, args)
    assert cut is not None
    nodes = cut.region.nodes
    prep, recurrent, shared = map(
        set, (cut.preparation, cut.recurrence, cut.shared_inputs)
    )
    assert prep | recurrent | shared == set(nodes)
    assert not (prep & recurrent or prep & shared or recurrent & shared)
    dots = tuple(spec.node for spec in cut.region.contractions)
    assert [node in prep for node in dots] == (
        [False, True, True, False] if recurrent_first else [True, True, False, False]
    )
    assert all(carry.input in recurrent for carry in cut.carries)
    assert set(cut.captures) <= shared
    assert all(store in recurrent for store in cut.region.stores)
    assert {image.dtype for image in cut.images} >= {
        torch.float16,
        torch.bfloat16,
        torch.float32,
    }
    for image in cut.images:
        assert image.node in prep
        assert image.logical_domain == _domain(image.node.meta["val"])
        assert image.dtype == image.node.meta["val"].dtype
        assert image.consumers
        assert all(
            consumer in recurrent and image.node in consumer.all_input_nodes
            for consumer in image.consumers
        )
    # Shape queries on the carry are metadata, not reads of carried values.
    assert any(node.target is torch.ops.aten.sym_size.int for node in shared)
    with pytest.raises(FrozenInstanceError):
        cut.images = ()  # pyrefly: ignore [read-only]


@pytest.mark.parametrize("dependency", [0, 1, 2])
def test_carry_taints_mask_index_and_explicit_accumulator(dependency: int) -> None:
    args = _inputs(dependency, typed=False)
    with _cpu_codegen():
        bound = _dependency_sequence._bind_isolated(args)
        cut = _cut(bound, _dependency_sequence, args)
    assert cut is not None
    first, independent = (spec.node for spec in cut.region.contractions)
    assert first in cut.recurrence
    assert independent in cut.preparation  # It appears after the recurrent dot.
    if dependency < 2:
        assert first.args[0] in cut.recurrence
        assert first.args[0].target is memory_ops.load
    else:
        assert cut.region.contractions[0].accumulator in cut.recurrence


def test_missing_runtime_binding_fails_closed_without_changing_graph() -> None:
    args = _inputs(True, typed=True)
    with _cpu_codegen():
        bound = _typed_sequence._bind_isolated(args)
        assert bound.host_function is not None
        graphs = bound.host_function.device_ir.graphs
        before = tuple(
            (node, node.op, node.target, node.args, dict(node.kwargs))
            for graph in graphs
            for node in graph.graph.nodes
        )
        with bound.env, bound.host_function:
            assert plan_preparation_cut(graphs) is None
        assert _cut(bound, _typed_sequence, args) is not None
        after = tuple(
            (node, node.op, node.target, node.args, dict(node.kwargs))
            for graph in graphs
            for node in graph.graph.nodes
        )
        assert after == before


def _opaque(value):
    return value


@pytest.mark.parametrize(
    "target",
    [
        _opaque,
        torch.ops.aten.copy_.default,
        torch.ops.aten.rand_like.default,
        torch.ops.aten._assert_async.msg,
    ],
)
def test_unknown_mutable_and_nondeterministic_effects_reject(target) -> None:
    args = _inputs(False, typed=True)
    with _cpu_codegen():
        bound = _typed_sequence._bind_isolated(args)
        cut = _cut(bound, _typed_sequence, args)
        assert cut is not None
        # Preserve the old lowering annotation: it is not a purity certificate.
        node = next(
            node for node in cut.recurrence if node.target is torch.ops.aten.mul.Tensor
        )
        node.target = target
        assert _cut(bound, _typed_sequence, args) is None


def test_hidden_dependency_kwargs_are_not_dropped() -> None:
    args = _inputs(True, typed=True)
    with _cpu_codegen():
        bound = _typed_sequence._bind_isolated(args)
        original = _cut(bound, _typed_sequence, args)
        assert original is not None
        node = next(node for node in original.preparation if node.target is dot)
        # Dependency-only kwargs are graph edges even when expression lowering
        # omits them. Use a pointwise cast to keep the normalized dot contract.
        narrowed = next(
            user for user in node.users if user.meta["val"].dtype == torch.float16
        )
        narrowed.kwargs = {**narrowed.kwargs, "_extra_deps": original.carries[0].input}
        cut = _cut(bound, _typed_sequence, args)
        assert cut is not None
        assert narrowed in cut.recurrence


def _kda_fixture() -> tuple[Kernel, tuple[object, ...]]:
    module = import_path(
        Path(__file__).resolve().parents[1]
        / "benchmarks/cute/kda_prefill_fused_bt32.py"
    )
    original = module.kda_prefill_native_math_bt32
    kernel = helion.kernel(
        original.fn,
        backend="cute",
        static_shapes=True,
        fast_math=True,
        autotune_config_overrides={
            "cute_chained_group_contractions": True,
            "cute_chained_mma_schedule": "tcgen05_tmem",
        },
    )
    q, k, v, gate = (
        torch.empty((1, 64, 2, 128), dtype=torch.bfloat16) for _ in range(4)
    )
    state = torch.empty((2, 2, 128, 128), dtype=torch.float32)
    args = (
        q,
        k,
        v,
        gate,
        torch.empty((1, 64, 2), dtype=torch.bfloat16),
        torch.empty((2,), dtype=torch.float32),
        torch.empty((2, 128), dtype=torch.float32),
        state,
        torch.empty_like(v),
        torch.empty_like(state),
        torch.tensor([0, 32, 64]),
        128**-0.5,
        -5 * 1.4426950408889634,
    )
    return kernel, args


def test_actual_kda_bt32_admitted_graph_cut() -> None:
    kernel, args = _kda_fixture()
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        cut = _cut(bound, kernel, args)
    assert cut is not None
    dots = tuple(spec.node for spec in cut.region.contractions)
    assert len(dots) == 15
    assert all(node in cut.preparation for node in dots[:10])
    assert all(node in cut.recurrence for node in dots[10:])
    assert any(
        cut.region.nodes.index(image.node) > cut.region.nodes.index(dots[10])
        for image in cut.images
    )
    assert all(node in cut.preparation for node in cut.region.scans)
    assert all(node in cut.preparation for node in cut.region.reductions)
    assert any(image.dtype == torch.bfloat16 for image in cut.images)
    assert any(image.dtype == torch.float32 for image in cut.images)
    assert any(
        consumer.target is memory_ops.store
        for image in cut.images
        for consumer in image.consumers
    )
    assert not set(cut.preparation).intersection(carry.input for carry in cut.carries)


def test_runtime_alias_replay_rejects_read_ahead() -> None:
    kernel, args = _kda_fixture()
    with _cpu_codegen():
        bound = kernel._bind_isolated(args)
        assert _cut(bound, kernel, args) is not None
        aliased = list(args)
        aliased[8] = args[2]  # Caller-provided output aliases the loaded V input.
        assert _cut(bound, kernel, tuple(aliased)) is None
        assert _cut(bound, kernel, args) is not None


def test_straight_line_region_is_not_a_loop_pipeline() -> None:
    args = (
        torch.empty((128, 16), dtype=torch.bfloat16),
        torch.empty((16, 32), dtype=torch.bfloat16),
    )
    with _cpu_codegen():
        bound = _root_residency._bind_isolated(args)
        assert _cut(bound, _root_residency, args) is None


def test_malformed_shape_metadata_is_not_a_shared_port() -> None:
    args = _inputs(False, typed=True)
    with _cpu_codegen():
        bound = _typed_sequence._bind_isolated(args)
        original = _cut(bound, _typed_sequence, args)
        assert original is not None
        shape = next(
            node
            for node in original.shared_inputs
            if node.target is torch.ops.aten.sym_size.int
        )
        shape.meta["val"] = 17
        assert _cut(bound, _typed_sequence, args) is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_untyped_crossing_value_fails_closed(dtype: torch.dtype) -> None:
    args = _inputs(False, typed=True)
    with _cpu_codegen():
        bound = _typed_sequence._bind_isolated(args)
        original = _cut(bound, _typed_sequence, args)
        assert original is not None
        image = next(image for image in original.images if image.dtype == dtype)
        image.node.meta["val"] = None
        assert _cut(bound, _typed_sequence, args) is None
