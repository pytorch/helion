from __future__ import annotations

import ast
from dataclasses import replace
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_m64_loop import _args
from .test_cute_chained_native_async_vector import _selected
from .test_cute_chained_preparation_cut import _runtime_values
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_independent_publication
from helion._compiler.cute import chained_pipeline_storage
from helion._compiler.cute import chained_prepared_values
from helion._compiler.cute.chained_async_groups import ASYNC_GROUP_LIMITS
from helion._compiler.cute.chained_async_groups import MAX_ASYNC_GROUPS
from helion._compiler.cute.chained_async_groups import WAIT
from helion._compiler.cute.chained_async_groups import AsyncCopyAttempt
from helion._compiler.cute.chained_async_groups import AsyncPreparationGroups
from helion._compiler.cute.chained_preparation_reads import EnqueuedPreparationFrontier
from helion._compiler.cute.chained_preparation_reads import (
    PreparationFrontierCompletion,
)
import helion.language as hl

_KEY = "cute_chained_async_group_limit"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _many_raw_images(
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
    transform: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    initial: torch.Tensor,
    output: torch.Tensor,
    final: torch.Tensor,
    extra: torch.Tensor,
    count: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    steps = hl.specialize(value.size(1))
    extra_count = cast("int", count)
    for batch in hl.tile(initial.size(0), block_size=1):
        state = initial[batch.id, :, :]
        for step in hl.grid(steps):
            rows = hl.arange(value.size(2))
            columns = hl.arange(value.size(3))
            raw = hl.load(
                value,
                [batch.id, step, rows, columns],
                extra_mask=(rows < step * 17 + 3)[:, None],
            )
            entering = state.to(left.dtype)
            corrected = raw.float() - hl.dot(right[batch.id, step, :, :], entering)
            if extra_count >= 1:
                corrected = corrected + extra[0, batch.id, step, :, :].float()
            if extra_count >= 2:
                corrected = corrected + extra[1, batch.id, step, :, :].float()
            if extra_count >= 3:
                corrected = corrected + extra[2, batch.id, step, :, :].float()
            if extra_count >= 4:
                corrected = corrected + extra[3, batch.id, step, :, :].float()
            narrowed = corrected.to(left.dtype)
            cross = hl.dot(left[batch.id, step, :, :], entering)
            cross = cross * gate[batch.id, step, :][:, None]
            result = hl.dot(transform[batch.id, step, :, :], narrowed, acc=cross)
            output[batch.id, step, :, :] = result.to(output.dtype)
            state = hl.dot(
                update[batch.id, step, :, :],
                narrowed,
                acc=state * gate[batch.id, step, -1],
            )
        final[batch.id, :, :] = state
    return output, final


def _inputs(
    count: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    *,
    broadcast_extra: bool = False,
) -> tuple[Any, ...]:
    args = _args(device, dtype=dtype)
    shape = list(args[2].shape)
    if broadcast_extra:
        shape[-2] = 1
    extra = torch.empty((4, *shape), device=device, dtype=dtype)
    return (*args, extra, count)


def _config(limit: int = 8, **options: object) -> helion.Config:
    return _selected(
        True,
        cute_chained_independent_publication=True,
        cute_chained_late_widening=True,
        **({_KEY: limit} | options),
    )


def _source(config: helion.Config, args: tuple[Any, ...] | None = None) -> str:
    args = _inputs() if args is None else args
    with _cpu_codegen():
        bound = _many_raw_images._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_many_raw_images, args)):
            return bound.to_code(config)


def _without_waits(source: str) -> str:
    class Remove(ast.NodeTransformer):
        def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:
            return None if ast.unparse(node) == WAIT else node

    return ast.dump(Remove().visit(ast.parse(source)))


@pytest.mark.parametrize("limit", [2, 4, 8])
@pytest.mark.parametrize("copies", [0, 1, 2, 4, 5, 6])
def test_actual_copy_groups_count_empty_commits_and_settle_at_boundaries(
    limit: int, copies: int
) -> None:
    owners = []
    initialize = AsyncPreparationGroups.__init__
    original = chained_prepared_values.emit_vector_expression

    def capture(self: AsyncPreparationGroups, *args: Any) -> None:
        initialize(self, *args)
        owners.append(self)

    def choose(*args: Any, **kwargs: Any) -> Any:
        attempt = kwargs.get("async_copy")
        if attempt is not None and len(attempt.owner.enqueued) >= copies:
            kwargs["async_staging"] = replace(
                kwargs["async_staging"], async_enabled=False
            )
        return original(*args, **kwargs)

    with (
        patch.object(AsyncPreparationGroups, "__init__", capture),
        patch.object(chained_prepared_values, "emit_vector_expression", choose),
    ):
        if copies < 2:
            with pytest.raises(
                exc.BackendUnsupported, match="at least two original copies"
            ):
                _source(_config(limit))
            return
        source = _source(_config(limit))
    (owner,) = owners
    assert owner.finished and not owner.pending and owner.active is None
    assert len(owner.enqueued) == copies
    assert source.count("cp_async_commit_group()") == copies
    assert source.count("cp_async_wait_group(0)") == (copies + limit - 1) // limit
    assert [len(item.groups) for item in owner.settled] == [
        min(limit, copies - start) for start in range(0, copies, limit)
    ]
    assert owner.ready_lines == (WAIT,)
    assert all(copy.matches() for copy in owner.enqueued)


@pytest.mark.parametrize("copies", [8, 9])
def test_architectural_eight_group_boundary_uses_real_extra_images(copies: int) -> None:
    assert MAX_ASYNC_GROUPS == 8 and ASYNC_GROUP_LIMITS == (1, 2, 4, 8)
    source = _source(_config(8), _inputs(copies - 5, broadcast_extra=copies > 8))
    assert source.count("cp_async_commit_group()") == copies
    assert source.count("cp_async_wait_group(0)") == (copies + 7) // 8


@pytest.mark.parametrize("limit", [2, 4, 8])
def test_only_wait_placement_changes_original_math_fallbacks_and_protocol(
    limit: int,
) -> None:
    before, after = _source(_config(1)), _source(_config(limit))
    assert _without_waits(before) == _without_waits(after)
    for marker in (
        "cute.gemm(",
        "mbarrier_wait(",
        "arrive_mbarrier(",
        "cp_async_commit_group()",
        "chain_prep_barrier.arrive_and_wait()",
        "fence_view_async_shared()",
    ):
        assert before.count(marker) == after.count(marker)
    # Every producer READY handoff follows wait, proxy fence, and role join.
    assert (
        "cute.arch.cp_async_wait_group(0)\n            cute.arch.fence_view_async_shared()\n            chain_prep_barrier.arrive_and_wait()"
        in after
    )


def test_default_and_explicit_one_are_byte_identical() -> None:
    config = _config(1).config
    config.pop(_KEY)
    assert _source(helion.Config.from_dict(config)) == _source(_config(1))


@pytest.mark.parametrize(
    "fault", ["source", "ordinal", "missing", "completed", "early_consume"]
)
def test_enqueued_frontier_cannot_be_forged_or_consumed_as_complete(fault: str) -> None:
    record = PreparationFrontierCompletion.record

    def changed(self: PreparationFrontierCompletion, *args: Any, **kwargs: Any) -> None:
        copy = kwargs.get("enqueued")
        if copy is not None:
            if fault == "source":
                kwargs["enqueued"] = replace(copy, body=())
            elif fault == "ordinal":
                kwargs["enqueued"] = replace(copy, ordinal=copy.ordinal + 1)
            elif fault == "missing":
                return
            elif fault == "completed":
                kwargs.pop("enqueued")
        record(self, *args, **kwargs)
        if copy is not None and fault == "early_consume":
            p = self.publication
            assert isinstance(p, EnqueuedPreparationFrontier)
            self.consume(p.cg, p.plan, p.execution, p.inputs, p.outputs, p.lines)

    with (
        patch.object(PreparationFrontierCompletion, "record", changed),
        pytest.raises(
            (exc.BackendUnsupported, exc.InternalError), match="frontier|enqueue"
        ),
    ):
        _source(_config())


@pytest.mark.parametrize(
    "fault", ["groups", "boundary", "wait", "pending", "limit", "ready"]
)
def test_settlement_revision_and_ready_are_mandatory(fault: str) -> None:
    finish = AsyncPreparationGroups.finish

    def changed(self: AsyncPreparationGroups) -> None:
        if fault == "pending":
            self.pending.append(self.enqueued[0])
        elif fault == "limit":
            self.limit = 9
        elif fault == "ready":
            self.ready_lines = ()
        else:
            old = self.settled[0]
            if fault == "groups":
                self.settled[0] = replace(old, groups=tuple(reversed(old.groups)))
            elif fault == "boundary":
                self.settled[0] = replace(old, before=old.before + 1)
            else:
                self.settled[0] = replace(old, lines=())
        finish(self)

    with (
        patch.object(AsyncPreparationGroups, "finish", changed),
        pytest.raises(exc.BackendUnsupported, match="async preparation groups"),
    ):
        _source(_config(2))


def test_actual_shared_read_still_vetoes_pending_publication() -> None:
    with (
        patch.object(
            chained_independent_publication,
            "resolved_preparation_reads",
            return_value=("chain_prepared_0",),
        ),
        pytest.raises(exc.BackendUnsupported, match="reads a shared image"),
    ):
        _source(_config())


@pytest.mark.parametrize("fault", ["role", "source", "owner"])
def test_original_vector_owner_is_mandatory(fault: str) -> None:
    emit = AsyncCopyAttempt.emit

    def changed(self: AsyncCopyAttempt, *args: Any, **kwargs: Any) -> list[str]:
        values = list(args)
        if fault == "role":
            values[3] = replace(values[3], threads=values[3].threads // 2)
        elif fault == "source":
            values[2] = self.owner.publication.pipeline.frame.cut.carries[0].input
        else:
            self.owner.active = None
        return emit(self, *values, **kwargs)

    with (
        patch.object(AsyncCopyAttempt, "emit", changed),
        pytest.raises(exc.BackendUnsupported, match="original copy owner"),
    ):
        _source(_config())


def test_final_storage_and_release_ownership_cannot_be_bypassed() -> None:
    with (
        patch.object(
            chained_pipeline_storage, "finalize_pipeline_storage", return_value=None
        ),
        pytest.raises(exc.BackendUnsupported, match="post-transport allocation"),
    ):
        _source(_config())


@pytest.mark.parametrize("value", [True, 0, 3, 9, "8"])
def test_limit_is_exact_supported_integer(value: object) -> None:
    with _cpu_codegen(), pytest.raises(exc.InvalidConfig, match="must be one of"):
        bound = _many_raw_images._bind_isolated(_inputs())
        bound.config_spec.normalized_config(_config().config | {_KEY: value})


def test_one_seed_eight_preserves_all_parents_and_overrides() -> None:
    with _cpu_codegen():
        bound = _many_raw_images._bind_isolated(_inputs())
        with bound.env:
            make = CuteChainedMatmulHeuristic._with_async_group_seed
            parents = [_selected(), _config(1)]
            seeds = make(bound.env, parents)
            assert seeds[:-1] == parents and len(seeds) == len(parents) + 1
            assert seeds[-1].config == parents[-1].config | {_KEY: 8}
            assert bound.config_spec.normalized_config(seeds[-1])[_KEY] == 8
            assert make(bound.env, seeds) == seeds
            with patch.dict(bound.env.settings.autotune_config_overrides, {_KEY: 1}):
                assert make(bound.env, parents) is parents


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("limit,misaligned", [(2, True), (8, False)])
def test_multiple_masked_frames_raw_casts_and_state_gpu(
    dtype: torch.dtype, limit: int, misaligned: bool
) -> None:
    _check_masked_frames(dtype, misaligned, (_config(1), _config(limit)))


def _check_masked_frames(
    dtype: torch.dtype, misaligned: bool, configurations: tuple[helion.Config, ...]
) -> None:
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("TCGEN05 requires SM100+")
    inputs = list(_inputs(1, "cuda", dtype))
    if misaligned:
        for index in (0, 2):
            storage = torch.empty(inputs[index].numel() + 1, device="cuda", dtype=dtype)
            inputs[index] = storage[1:].view(inputs[index].shape)
    left, right, value, transform, update, gate, initial, output, final, extra, _ = (
        inputs
    )
    generator = torch.Generator(device="cuda").manual_seed(3005)
    readonly = [*inputs[:7], extra]
    for tensor in (*inputs[:5], initial, extra):
        tensor.copy_(
            torch.randint(-2, 3, tensor.shape, device="cuda", generator=generator)
            * 0.015625
        )
    gate.copy_(
        torch.randint(1, 4, gate.shape, device="cuda", generator=generator) * 0.25
    )
    unchanged = [tensor.clone() for tensor in readonly]
    expected, state = output.clone(), initial.double()
    rows = torch.arange(value.size(2), device="cuda")
    for step in range(3):
        raw = torch.where(
            (rows < step * 17 + 3)[None, :, None], value[:, step].double(), 0.0
        )
        entering = state.to(dtype).double()
        corrected = (
            (raw - right[:, step].double() @ entering + extra[0, :, step].double())
            .to(dtype)
            .double()
        )
        cross = (left[:, step].double() @ entering) * gate[:, step, :, None].double()
        expected[:, step] = (transform[:, step].double() @ corrected + cross).to(dtype)
        state = (
            update[:, step].double() @ corrected
            + state * gate[:, step, -1, None, None].double()
        )
    results = []
    for selected in configurations:
        output.fill_(19.0)
        final.fill_(float("nan"))
        result = _many_raw_images._bind_isolated(tuple(inputs)).compile_config(
            selected
        )(*inputs)
        results.append(tuple(tensor.clone() for tensor in result))
        torch.testing.assert_close(result[0], expected, atol=0.002, rtol=0.01)
        torch.testing.assert_close(result[1], state.float(), atol=0.002, rtol=0.01)
    for before, after in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    for before, after in zip(unchanged, readonly, strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
