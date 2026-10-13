from __future__ import annotations

import ast
from dataclasses import replace
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch.fx import Node

from ._cute_aux import _cpu_codegen
from .test_cute_chained_load_only_preparation import _load_only_recurrence
from .test_cute_chained_m64_loop import _args
from .test_cute_chained_m64_loop import _source
from .test_cute_chained_native_async_vector import _selected
from .test_cute_chained_preparation_cut import _runtime_values
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_pipeline_storage
from helion._compiler.cute.chained_preparation_cut import late_widening_candidates
from helion._compiler.cute.chained_preparation_cut import plan_preparation_cut
from helion._compiler.cute.contraction_region import _domain
import helion.language as hl
from helion.language import memory_ops

if TYPE_CHECKING:
    from helion._compiler.cute.chained_preparation_cut import PreparationCut
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

_KEY = "cute_chained_late_widening"


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _masked_widening(
    left: torch.Tensor,
    right: torch.Tensor,
    value: torch.Tensor,
    transform: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    initial: torch.Tensor,
    output: torch.Tensor,
    final: torch.Tensor,
    fanout: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    steps = hl.specialize(value.size(1))
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
            widened = raw.float()
            entering = state.to(left.dtype)
            corrected = widened - hl.dot(right[batch.id, step, :, :], entering)
            narrowed = corrected.to(left.dtype)
            cross = hl.dot(left[batch.id, step, :, :], entering)
            if fanout:
                cross = cross + widened
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


def _config(enabled: bool = True, *, asynchronous: bool = True) -> helion.Config:
    return _selected(asynchronous, **{_KEY: enabled})


def _cut(
    bound: BoundKernel,
    kernel: Kernel,
    args: tuple[object, ...],
    *,
    enabled: bool = False,
) -> PreparationCut | None:
    assert bound.host_function is not None
    with (
        bound.env,
        bound.host_function,
        bound.env.use_runtime_arg_values(_runtime_values(kernel, args)),
    ):
        return plan_preparation_cut(
            bound.host_function.device_ir.graphs, late_widening=enabled
        )


def _facts(cut: PreparationCut) -> tuple[object, ...]:
    return tuple(
        (node, node.target, node.args, dict(node.kwargs), tuple(node.users))
        for node in cut.region.nodes
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fanout", [False, True])
def test_cut_moves_only_original_cast_and_preserves_mask_and_all_users(
    dtype: torch.dtype, fanout: bool
) -> None:
    args = (*_args(dtype=dtype), fanout)
    with _cpu_codegen():
        bound = _masked_widening._bind_isolated(args)
        before = _cut(bound, _masked_widening, args)
        assert before is not None
        facts = _facts(before)
        moved = late_widening_candidates(before)
        assert len(moved) == 1
        widening = moved[0]
        source = widening.args[0]
        assert isinstance(source, Node)
        assert source.target is memory_ops.load
        assert source.args[2] is not None
        assert len(widening.users) == (2 if fanout else 1)
        after = _cut(bound, _masked_widening, args, enabled=True)
    assert after is not None and _facts(after) == facts
    assert after.region == before.region
    assert after.storage_proof_key == before.storage_proof_key
    assert (
        after.carries == before.carries and after.shared_inputs == before.shared_inputs
    )
    assert set(after.preparation) == set(before.preparation) - set(moved)
    assert set(after.recurrence) == set(before.recurrence) | set(moved)
    assert source in after.preparation and widening in after.recurrence
    images = {image.node: image for image in after.images}
    assert widening not in images and images[source].consumers == (widening,)
    assert images[source].dtype == dtype
    assert images[source].logical_domain == _domain(source.meta["val"])
    for image in after.images:
        assert image.dtype == image.node.meta["val"].dtype
        assert image.logical_domain == _domain(image.node.meta["val"])
        assert set(image.consumers) == set(image.node.users) & set(after.recurrence)


@pytest.mark.parametrize(
    "fault",
    [
        "cast_target",
        "cast_dtype",
        "cast_kwargs",
        "cast_arity",
        "cast_domain",
        "raw_dtype",
        "raw_host_dtype",
        "raw_fanout",
        "raw_kwargs",
        "raw_arity",
        "raw_indices",
        "mask_dtype",
        "preparation_use",
        "hidden_dependency",
        "image_dtype",
        "image_domain",
        "image_consumers",
    ],
)
def test_unsupported_cast_domain_and_dependency_cannot_move(fault: str) -> None:
    args = (*_args(), False)
    with _cpu_codegen():
        bound = _masked_widening._bind_isolated(args)
        cut = _cut(bound, _masked_widening, args)
        assert cut is not None
        (node,) = late_widening_candidates(cut)
        raw = node.args[0]
        assert isinstance(raw, Node)
        host, _, mask, _ = raw.args
        assert isinstance(host, Node) and isinstance(mask, Node)
        image = next(image for image in cut.images if image.node is node)
        if fault == "cast_target":
            node.target = torch.ops.aten._to_copy.default
        elif fault == "cast_dtype":
            node.args = (raw, torch.float16)
        elif fault == "cast_kwargs":
            node.kwargs = {"dtype": torch.float32}
        elif fault == "cast_arity":
            node.args = (*node.args, False)
        elif fault == "cast_domain":
            node.meta["val"] = torch.empty((32, 128))
        elif fault == "raw_dtype":
            raw.meta["val"] = torch.empty_like(raw.meta["val"], dtype=torch.float32)
        elif fault == "raw_host_dtype":
            host.meta["val"] = torch.empty_like(host.meta["val"], dtype=torch.float32)
        elif fault == "raw_fanout":
            user = next(iter(node.users))
            user.kwargs = {"extra": raw}
        elif fault == "raw_kwargs":
            raw.kwargs = {"other": 1}
        elif fault == "raw_arity":
            raw.args = (*raw.args, False)
        elif fault == "raw_indices":
            raw.args = (raw.args[0], 0, *raw.args[2:])
        elif fault == "mask_dtype":
            mask.meta["val"] = torch.empty((64, 1), dtype=torch.int32)
        elif fault == "preparation_use":
            user = next(iter(node.users))
            cut = replace(
                cut, recurrence=tuple(n for n in cut.recurrence if n is not user)
            )
        elif fault == "hidden_dependency":
            node.kwargs = {"_extra_deps": cut.carries[0].input}
        else:
            changes: dict[str, Any] = {
                "image_dtype": {"dtype": torch.bfloat16},
                "image_domain": {"logical_domain": (1,)},
                "image_consumers": {"consumers": ()},
            }[fault]
            cut = replace(
                cut,
                images=tuple(
                    replace(x, **changes) if x is image else x for x in cut.images
                ),
            )
        assert not late_widening_candidates(cut)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_true_raw_frame_and_consumer_cast_keep_finalizer_and_protocol(
    asynchronous: bool,
) -> None:
    pipelines = []
    finalize = chained_pipeline_storage.finalize_pipeline_storage

    def observe(*args: Any, **kwargs: Any) -> Any:
        result = finalize(*args, **kwargs)
        assert result is not None
        pipelines.append((args[1], result))
        return result

    with patch.object(chained_pipeline_storage, "finalize_pipeline_storage", observe):
        before = _source(_config(False, asynchronous=asynchronous))
        after = _source(_config(asynchronous=asynchronous))
    first, second = (item[0] for item in pipelines)
    before_casts = late_widening_candidates(first.frame.cut)
    assert len(before_casts) == 1
    (cast,) = before_casts
    before_image = next(b for b in first.frame.buffers if b.node is cast)
    # Each isolated bind has distinct FX nodes: identify the source by its
    # original target/name, not cross-bind Python identity.
    raw = cast.args[0]
    assert isinstance(raw, Node)
    raw_name = raw.name
    after_image = next(
        b
        for b in second.frame.buffers
        if b.node is not None and b.node.name == raw_name
    )
    assert before_image.dtype == torch.float32 and after_image.dtype == torch.bfloat16
    assert after_image.shape == before_image.shape
    assert second.frame.layout.allocated_bytes < first.frame.layout.allocated_bytes
    assert all(storage.charged_bytes <= 232448 for _, storage in pipelines)
    tree = ast.parse(after)
    loaded = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Name)
            and value.value.id == after_image.name
            for value in ast.walk(node.value)
        )
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cutlass.Float32"
        and any(
            isinstance(value, ast.Name) and value.id in loaded for value in node.args
        )
        for node in ast.walk(tree)
    )
    if asynchronous:
        assert after.count("cpasync.CopyG2SOp") == before.count("cpasync.CopyG2SOp") + 1
    for operation in (
        "cute.gemm(",
        "mbarrier_wait(",
        "arrive_mbarrier(",
        "chain_prep_barrier.arrive_and_wait()",
    ):
        assert before.count(operation) == after.count(operation)
    ast.parse(after)


def test_default_source_and_false_selection_are_identical() -> None:
    assert _source(_selected()) == _source(_config(False))


@pytest.mark.parametrize("enabled", [False, True])
def test_masked_runtime_fixture_codegen_has_original_predicate_and_fallback(
    enabled: bool,
) -> None:
    args = (*_args(), False)
    with _cpu_codegen():
        bound = _masked_widening._bind_isolated(args)
        with bound.env.use_runtime_arg_values(_runtime_values(_masked_widening, args)):
            source = bound.to_code(_config(enabled))
    assert "17" in source and "if not chain_prepared_" in source
    assert source.count("cpasync.CopyG2SOp") == (5 if enabled else 4)
    ast.parse(source)


def test_no_widening_does_not_silently_accept_option() -> None:
    args = list(_args())
    args[2] = args[2].float()
    with _cpu_codegen():
        bound = _load_only_recurrence._bind_isolated(tuple(args))
        assert not bound.config_spec.cute_chained_late_widening_search_enabled
        with pytest.raises(exc.InvalidConfig, match="raw widening"):
            bound.config_spec.normalized_config(_config())
        # Even if a coarse search fact is stale, the exact planner declines.
        bound.config_spec.cute_chained_late_widening_search_enabled = True
        with (
            bound.env.use_runtime_arg_values(
                _runtime_values(_load_only_recurrence, tuple(args))
            ),
            pytest.raises(
                exc.BackendUnsupported, match="admitted common loop pipeline"
            ),
        ):
            bound.to_code(_config())


@pytest.mark.parametrize("fault", ["missing", "capacity"])
def test_late_widening_still_requires_complete_final_storage(fault: str) -> None:
    finalize = chained_pipeline_storage.finalize_pipeline_storage

    def changed(*args: Any, **kwargs: Any) -> Any:
        if fault == "missing":
            return None
        kwargs["capacity_bytes"] = 1
        return finalize(*args, **kwargs)

    with (
        patch.object(chained_pipeline_storage, "finalize_pipeline_storage", changed),
        pytest.raises(exc.BackendUnsupported, match="post-transport allocation"),
    ):
        _source(_config())


def test_read_ahead_alias_and_absent_runtime_binding_remain_rejected() -> None:
    args = _args()
    with _cpu_codegen():
        bound = _load_only_recurrence._bind_isolated(args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            assert (
                plan_preparation_cut(
                    bound.host_function.device_ir.graphs, late_widening=True
                )
                is None
            )
        aliased = (*args[:2], args[7][:, :3], *args[3:])
        bound = _load_only_recurrence._bind_isolated(aliased)
        assert not bound.config_spec.cute_chained_late_widening_search_enabled
        with pytest.raises(exc.InvalidConfig):
            bound.config_spec.normalized_config(_config())


@pytest.mark.parametrize("value", [1, "true"])
def test_option_requires_bool(value: object) -> None:
    with _cpu_codegen(), pytest.raises(exc.InvalidConfig, match="must be bool"):
        bound = _load_only_recurrence._bind_isolated(_args())
        bound.config_spec.normalized_config(_selected().config | {_KEY: value})


def test_one_seed_preserves_parents_and_explicit_override() -> None:
    with _cpu_codegen():
        bound = _load_only_recurrence._bind_isolated(_args())
        assert bound.config_spec.cute_chained_late_widening_search_enabled
        with bound.env:
            parents = [_selected(False), _selected(True)]
            make = CuteChainedMatmulHeuristic._with_late_widening_seed
            seeds = make(bound.env, parents)
            assert seeds[:-1] == parents and len(seeds) == len(parents) + 1
            assert seeds[-1].config == parents[-1].config | {_KEY: True}
            assert bound.config_spec.normalized_config(seeds[-1])[_KEY] is True
            assert make(bound.env, seeds) == seeds
            with patch.dict(
                bound.env.settings.autotune_config_overrides, {_KEY: False}
            ):
                assert make(bound.env, parents) is parents


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("misaligned", [False, True])
def test_changing_masked_raw_images_and_fp32_state_gpu(
    dtype: torch.dtype, misaligned: bool
) -> None:
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("TCGEN05 requires SM100+")
    args = list(_args("cuda", dtype=dtype, steps=3))
    if misaligned:
        storage = torch.empty(args[2].numel() + 1, dtype=dtype, device="cuda")
        args[2] = storage[1:].view(args[2].shape)
    generator = torch.Generator(device="cuda").manual_seed(2905)
    for tensor in (*args[:5], args[6]):
        tensor.copy_(
            torch.randint(-2, 3, tensor.shape, generator=generator, device="cuda")
            * 0.015625
        )
    args[5].copy_(
        torch.randint(1, 4, args[5].shape, generator=generator, device="cuda") * 0.25
    )
    left, right, value, transform, update, gate, initial, output, final = args
    unchanged = [tensor.clone() for tensor in args[:7]]
    expected, state = output.clone(), initial.double()
    rows = torch.arange(value.size(2), device="cuda")
    for step in range(3):
        raw = torch.where(
            (rows < step * 17 + 3)[None, :, None], value[:, step].double(), 0.0
        )
        entering = state.to(dtype).double()
        corrected = (raw - right[:, step].double() @ entering).to(dtype).double()
        cross = (left[:, step].double() @ entering) * gate[:, step, :, None].double()
        expected[:, step] = (transform[:, step].double() @ corrected + cross).to(dtype)
        state = (
            update[:, step].double() @ corrected
            + state * gate[:, step, -1, None, None].double()
        )
    results = []
    inputs = (*args, False)
    for enabled in (False, True):
        output.fill_(19.0)
        final.fill_(float("nan"))
        result = _masked_widening._bind_isolated(inputs).compile_config(
            _config(enabled)
        )(*inputs)
        results.append(tuple(tensor.clone() for tensor in result))
        torch.testing.assert_close(result[0], expected, atol=0.002, rtol=0.01)
        torch.testing.assert_close(result[1], state.float(), atol=0.002, rtol=0.01)
    for before, after in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    for before, after in zip(unchanged, args[:7], strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
