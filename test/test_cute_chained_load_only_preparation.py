from __future__ import annotations

import ast
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_preparation_cut import _cut
from .test_cute_chained_preparation_cut import _inputs
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_cut import _typed_sequence
from .test_cute_chained_preparation_frame import _capture
from .test_cute_chained_preparation_frame import _safe
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_pipeline_storage
from helion._compiler.cute.chained_preparation_cut import is_load_only_preparation
from helion._compiler.cute.chained_preparation_frame import plan_preparation_frame
from helion._compiler.cute.chained_preparation_pipeline import plan_preparation_pipeline
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _load_only_recurrence(
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
    transform: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    initial: torch.Tensor,
    output: torch.Tensor,
    final: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    steps = hl.specialize(value.size(1))
    for batch in hl.tile(initial.size(0), block_size=1):
        state = initial[batch.id, :, :]
        for step in hl.grid(steps):
            entering = state.to(left.dtype)
            corrected = value[batch.id, step, :, :].float() - hl.dot(
                right[batch.id, step, :, :], entering
            )
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


def _args(
    device: str = "cpu", *, steps: int = 3, dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, ...]:
    batch, rows, depth, columns = 2, 32, 64, 128
    shapes = (
        (batch, steps, rows, depth),
        (batch, steps, rows, depth),
        (batch, steps, rows, columns),
        (batch, steps, rows, rows),
        (batch, steps, depth, rows),
    )
    return (
        *(torch.empty(shape, device=device, dtype=dtype) for shape in shapes),
        torch.empty((batch, steps, rows), device=device),
        torch.empty((batch, depth, columns), device=device),
        torch.full((batch, steps + 1, rows, columns), 19.0, device=device, dtype=dtype),
        torch.empty((batch, depth, columns), device=device),
    )


def _config(*, pipeline: bool = True) -> helion.Config:
    return helion.Config(
        num_warps=8,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_group_contractions=True,
        cute_chained_preparation_pipeline=pipeline,
    )


def test_load_only_discovery_and_finite_typed_frame() -> None:
    with _cpu_codegen():
        bound = _load_only_recurrence._bind_isolated(_args())
        assert bound.config_spec.cute_chained_load_only_preparation_search_enabled
        assert bound.config_spec.cute_chained_preparation_pipeline_search_enabled
        normalized = bound.config_spec.normalized_config(dict(_config()))
        assert "cute_chained_warp_mma_rows" not in normalized
    plan, cut, shapes = _capture(
        _load_only_recurrence, _args(), _config(pipeline=False)
    )
    assert is_load_only_preparation(cut)
    frame = plan_preparation_frame(plan, cut, shapes)
    assert frame is not None and not frame.stages
    assert {action.kind for action in frame.actions} == {"frontier", "ready"}
    _safe(frame)
    pipeline = plan_preparation_pipeline(plan, cut, shapes, 1)
    assert pipeline is not None and pipeline.deferred_capacity
    assert pipeline.shared_bytes > 1


@pytest.mark.parametrize("mutation", ["empty", "dtype", "carry"])
def test_load_only_rejects_invalid_cut_or_typed_image(mutation: str) -> None:
    plan, cut, shapes = _capture(
        _load_only_recurrence, _args(), _config(pipeline=False)
    )
    if mutation == "empty":
        cut = replace(cut, images=())
        assert not is_load_only_preparation(cut)
    elif mutation == "dtype":
        first = cut.images[0]
        dtype = torch.int32 if first.dtype != torch.int32 else torch.float32
        cut = replace(cut, images=(replace(first, dtype=dtype), *cut.images[1:]))
    else:
        carried = cut.carries[0].input
        cut = replace(
            cut,
            preparation=(*cut.preparation, carried),
            recurrence=tuple(node for node in cut.recurrence if node is not carried),
        )
    assert plan_preparation_pipeline(plan, cut, shapes, 232448) is None


def test_load_only_read_ahead_rejects_output_alias() -> None:
    args = list(_args())
    args[0] = args[7][..., :64]
    with _cpu_codegen():
        bound = _load_only_recurrence._bind_isolated(tuple(args))
        assert not bound.config_spec.cute_chained_preparation_pipeline_search_enabled
        with pytest.raises(exc.InvalidConfig):
            bound.config_spec.normalized_config(dict(_config()))


def test_load_only_emission_requires_accepted_final_storage() -> None:
    accepted = []
    original = chained_pipeline_storage.finalize_pipeline_storage

    def observe(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        accepted.append(result)
        return result

    with _cpu_codegen():
        args = _args()
        bound = _load_only_recurrence._bind_isolated(args)
        with (
            bound.env.use_runtime_arg_values(
                _runtime_values(_load_only_recurrence, args)
            ),
            patch.object(
                chained_pipeline_storage, "finalize_pipeline_storage", observe
            ),
        ):
            source = bound.to_code(_config())
    assert len(accepted) == 1 and accepted[0] is not None
    assert accepted[0].charged_bytes <= 232448
    tree = ast.parse(source)
    roles = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and node.orelse
        and ast.unparse(node.test) == "chain_thread < 128"
    )
    prep = ast.unparse(ast.Module(body=roles.body, type_ignores=[]))
    recurrence = ast.unparse(ast.Module(body=roles.orelse, type_ignores=[]))
    assert "cute.gemm(" not in prep
    assert "cute.gemm(" in recurrence
    assert "chain_prep_barrier.arrive_and_wait()" in prep
    assert "mbarrier_wait" in prep and "mbarrier_wait" in recurrence


@pytest.mark.parametrize("failure", ["missing", "capacity"])
def test_load_only_failed_final_storage_has_no_emission_fallback(failure: str) -> None:
    original = chained_pipeline_storage.finalize_pipeline_storage

    def reject(*args: Any, **kwargs: Any) -> Any:
        if failure == "missing":
            return None
        kwargs["capacity_bytes"] = 1
        return original(*args, **kwargs)

    with _cpu_codegen():
        args = _args()
        bound = _load_only_recurrence._bind_isolated(args)
        with (
            bound.env.use_runtime_arg_values(
                _runtime_values(_load_only_recurrence, args)
            ),
            patch.object(chained_pipeline_storage, "finalize_pipeline_storage", reject),
            pytest.raises(exc.BackendUnsupported, match="post-transport allocation"),
        ):
            bound.to_code(_config())


@pytest.mark.parametrize(
    "target", [torch.ops.aten.copy_.default, torch.ops.aten.rand_like.default]
)
def test_load_only_unknown_effects_do_not_become_read_ahead(target: Any) -> None:
    with _cpu_codegen():
        args = _args()
        bound = _load_only_recurrence._bind_isolated(args)
        cut = _cut(bound, _load_only_recurrence, args)
        assert cut is not None
        image = next(image for image in cut.images if image.dtype == torch.bfloat16)
        image.node.target = target
        assert _cut(bound, _load_only_recurrence, args) is None


def test_load_only_seed_is_additive_and_mma_preparation_stays_strict() -> None:
    parents = [
        _config(pipeline=False),
        helion.Config(num_warps=4, cute_chained_mma_schedule="coalesced"),
    ]
    with _cpu_codegen():
        bound = _load_only_recurrence._bind_isolated(_args())
        parents = [bound.config_spec.normalized_config(dict(seed)) for seed in parents]
        seeds = CuteChainedMatmulHeuristic._with_preparation_pipeline_seeds(
            bound.env, parents
        )
        assert seeds[:2] == parents and len(seeds) == 3
        assert seeds[-1].config["cute_chained_preparation_pipeline"] is True
        assert "cute_chained_warp_mma_rows" not in seeds[-1].config
        bound.config_spec.normalized_config(dict(seeds[-1]))
        old = _typed_sequence._bind_isolated(_inputs(False, typed=True))
        assert old.config_spec.cute_chained_preparation_pipeline_search_enabled
        assert not old.config_spec.cute_chained_load_only_preparation_search_enabled
        with pytest.raises(exc.InvalidConfig, match="positive warp MMA rows"):
            old.config_spec.normalized_config(dict(_config()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("steps", [1, 3])
def test_load_only_changing_frames_and_state_gpu(
    dtype: torch.dtype, steps: int
) -> None:
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("TCGEN05 requires SM100+")
    args = _args("cuda", steps=steps, dtype=dtype)
    torch.manual_seed(7201)
    for tensor in (*args[:5], args[6]):
        tensor.copy_(torch.randint(-2, 3, tensor.shape, device="cuda") * 0.03125)
    args[5].copy_(
        torch.randint(1, 4, args[5].shape, device="cuda", dtype=torch.int32) * 0.25
    )
    left, right, value, transform, update, gate, initial, output, final = args
    originals = [tensor.clone() for tensor in args[:7]]
    expected = output.clone()
    state = initial.double()
    for step in range(steps):
        entering = state.to(dtype).double()
        corrected = value[:, step].double() - right[:, step].double() @ entering
        narrowed = corrected.to(dtype).double()
        cross = (left[:, step].double() @ entering) * gate[:, step, :, None].double()
        expected[:, step] = (transform[:, step].double() @ narrowed + cross).to(dtype)
        state = (
            update[:, step].double() @ narrowed
            + state * gate[:, step, -1, None, None].double()
        )
    actual = _load_only_recurrence._bind_isolated(args).compile_config(_config())(*args)
    torch.testing.assert_close(actual[0], expected, atol=0.002, rtol=0.01)
    torch.testing.assert_close(actual[1], state.float(), atol=0.002, rtol=0.01)
    torch.testing.assert_close(actual[0][:, -1], expected[:, -1], atol=0, rtol=0)
    for tensor, original in zip(args[:7], originals, strict=True):
        torch.testing.assert_close(tensor, original, atol=0, rtol=0)
