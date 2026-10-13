from __future__ import annotations

import ast
from dataclasses import replace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _cpu_codegen
from .test_cute_chained_load_only_preparation import _load_only_recurrence
from .test_cute_chained_m64_loop import _args
from .test_cute_chained_m64_loop import _source
from .test_cute_chained_native_async_vector import _masked_raw_recurrence
from .test_cute_chained_native_async_vector import _selected
from .test_cute_chained_preparation_cut import _runtime_values
import helion
from helion import exc
from helion._compiler.autotuner_heuristics.cute import CuteChainedMatmulHeuristic
from helion._compiler.cute import chained_independent_publication
from helion._compiler.cute import chained_pipeline_storage
from helion._compiler.cute.chained_independent_publication import IndependentPublication
from helion._compiler.cute.chained_preparation_reads import (
    PreparationFrontierCompletion,
)

_KEY = "cute_chained_independent_publication"


def _config(enabled: bool = True, *, asynchronous: bool = True) -> helion.Config:
    return _selected(asynchronous, **{_KEY: enabled})


def _without_producer_joins(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            joins = [
                item
                for item in node.body
                if ast.unparse(item) == "chain_prep_barrier.arrive_and_wait()"
            ]
            if joins:
                node.body = [item for item in node.body if item not in joins[:-1]]
    return ast.dump(tree, include_attributes=False)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_only_independent_joins_change_not_device_math_or_storage(
    asynchronous: bool,
) -> None:
    allocations = []
    finalize = chained_pipeline_storage.finalize_pipeline_storage

    def observe(*args: Any, **kwargs: Any) -> Any:
        result = finalize(*args, **kwargs)
        assert result is not None
        allocations.append(result.allocations)
        return result

    with patch.object(chained_pipeline_storage, "finalize_pipeline_storage", observe):
        before = _source(_config(False, asynchronous=asynchronous))
        after = _source(_config(asynchronous=asynchronous))
    assert allocations[0] == allocations[1]
    assert before.count("chain_prep_barrier.arrive_and_wait()") == 8
    assert after.count("chain_prep_barrier.arrive_and_wait()") == 1
    assert _without_producer_joins(before) == ast.dump(ast.parse(after))
    for operation in (
        "mbarrier_wait(",
        "arrive_mbarrier(",
        "fence_view_async_shared()",
        "cp_async_commit_group()",
        "cp_async_wait_group(0)",
    ):
        assert before.count(operation) == after.count(operation)


def test_default_source_and_false_config_are_identical() -> None:
    assert _source(_selected()) == _source(_config(False))


@pytest.mark.parametrize("fault", ["overlap", "read", "duplicate", "collective"])
def test_semantic_or_physical_dependency_rejects_join_elimination(fault: str) -> None:
    initialize = IndependentPublication.__init__

    def changed(self: IndependentPublication, *args: Any) -> None:
        cg, plan, pipeline, execution, storage = args
        frame = pipeline.frame
        first, second, *remaining = frame.actions
        if fault == "overlap":
            regions = list(frame.layout.regions)
            regions[1] = replace(regions[1], byte_offset=regions[0].byte_offset)
            frame = replace(frame, layout=replace(frame.layout, regions=tuple(regions)))
        elif fault == "read":
            second = replace(second, reads=first.writes)
        elif fault == "duplicate":
            second = replace(second, writes=first.writes)
        else:
            second = replace(second, kind="collective")
        frame = replace(frame, actions=(first, second, *remaining))
        initialize(self, cg, plan, replace(pipeline, frame=frame), execution, storage)

    with (
        patch.object(IndependentPublication, "__init__", changed),
        pytest.raises(exc.BackendUnsupported, match="independent publication"),
    ):
        _source(_config())


@pytest.mark.parametrize("fault", ["missing", "changed_source", "changed_writes"])
def test_only_actual_unchanged_frontier_completion_can_remove_join(fault: str) -> None:
    record = PreparationFrontierCompletion.record

    def changed(self: PreparationFrontierCompletion, *args: Any) -> None:
        if fault == "missing":
            return
        record(self, *args)
        assert self.publication is not None
        if fault == "changed_source":
            self.publication = replace(self.publication, lines=())
        else:
            self.publication = replace(self.publication, buffers=())

    with (
        patch.object(PreparationFrontierCompletion, "record", changed),
        pytest.raises(exc.InternalError, match="frontier completion"),
    ):
        _source(_config())


def test_late_unowned_read_rejects_nominally_independent_action() -> None:
    accept = IndependentPublication.accept

    def changed(self: IndependentPublication, *args: Any) -> list[str]:
        action, stop, before, after, completion, lines = args
        node = self.pipeline.frame.buffers[0].node
        assert node is not None
        # The completed source must agree with this new boundary too. Keeping
        # the old completion prevents a nominal action label authorizing it.
        before = dict(before) | {node: self.pipeline.frame.buffers[0].name}
        return accept(self, action, stop, before, after, completion, lines)

    with (
        patch.object(IndependentPublication, "accept", changed),
        pytest.raises(exc.InternalError, match="frontier completion"),
    ):
        _source(_config())


def test_actual_shared_read_vetoes_completed_frontier() -> None:
    with (
        patch.object(
            chained_independent_publication,
            "resolved_preparation_reads",
            return_value=("chain_prepared_0",),
        ),
        pytest.raises(exc.BackendUnsupported, match="reads a shared image"),
    ):
        _source(_config())


@pytest.mark.parametrize("enabled", [False, True])
def test_unsupported_vector_grouping_keeps_its_original_veto(enabled: bool) -> None:
    with pytest.raises(exc.BackendUnsupported, match="coordinate-owned operands"):
        _source(_selected(False, cute_chained_vector_group=True, **{_KEY: enabled}))


@pytest.mark.parametrize("fault", ["capacity", "missing"])
def test_complete_finalizer_remains_mandatory(fault: str) -> None:
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


@pytest.mark.parametrize("fault", ["join", "repeated", "alias"])
def test_final_ready_join_and_semantic_revision_cannot_be_lost(fault: str) -> None:
    finish = IndependentPublication.finish

    def changed(self: IndependentPublication, action: Any, lines: list[str]) -> None:
        if fault == "join":
            lines = lines[:-1]
        elif fault == "repeated":
            finish(self, action, lines)
        else:
            key = next(iter(self.plan.tensor_aliases))
            self.plan.tensor_aliases[key] = "different_storage"
        finish(self, action, lines)

    with (
        patch.object(IndependentPublication, "finish", changed),
        pytest.raises(exc.BackendUnsupported, match="final READY join"),
    ):
        _source(_config())


@pytest.mark.parametrize("value", [1, "true"])
def test_selection_requires_explicit_bool(value: object) -> None:
    with _cpu_codegen(), pytest.raises(exc.InvalidConfig, match="must be bool"):
        bound = _load_only_recurrence._bind_isolated(_args())
        bound.config_spec.normalized_config(_selected().config | {_KEY: value})


def test_one_structural_seed_keeps_every_existing_parent_and_override() -> None:
    args = _args()
    with _cpu_codegen():
        bound = _load_only_recurrence._bind_isolated(args)
        with (
            bound.env,
            bound.env.use_runtime_arg_values(
                _runtime_values(_load_only_recurrence, args)
            ),
        ):
            parents = [_selected(False), _selected(True)]
            make = CuteChainedMatmulHeuristic._with_independent_publication_seed
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
@pytest.mark.parametrize("asynchronous", [False, True])
def test_independent_changing_masked_frames_and_state_gpu(
    dtype: torch.dtype,
    asynchronous: bool,
) -> None:
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("TCGEN05 requires SM100+")
    args = list(_args("cuda", dtype=dtype, steps=3))
    # Force the scalar fallback for one raw member while other images retain
    # their exact guarded asynchronous transport.
    left_storage = torch.empty(args[0].numel() + 1, dtype=dtype, device="cuda")
    args[0] = left_storage[1:].view(args[0].shape)
    generator = torch.Generator(device="cuda").manual_seed(2605)
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
    rows = torch.arange(left.size(2), device="cuda")
    for step in range(3):
        mask = (rows < step * 17 + 3)[None, :, None]
        entering = state.to(dtype).double()
        lhs = torch.where(mask, left[:, step].double(), 0.0)
        rhs = torch.where(mask, right[:, step].double(), 0.0)
        corrected = (value[:, step].double() - rhs @ entering).to(dtype).double()
        cross = (lhs @ entering) * gate[:, step, :, None].double()
        expected[:, step] = (transform[:, step].double() @ corrected + cross).to(dtype)
        state = (
            update[:, step].double() @ corrected
            + state * gate[:, step, -1, None, None].double()
        )
    results = []
    for enabled in (False, True):
        bound = _masked_raw_recurrence._bind_isolated(tuple(args))
        result = bound.compile_config(_config(enabled, asynchronous=asynchronous))(
            *args
        )
        results.append(tuple(tensor.clone() for tensor in result))
        torch.testing.assert_close(result[0], expected, atol=0.002, rtol=0.01)
        torch.testing.assert_close(result[1], state.float(), atol=0.002, rtol=0.01)
    for before, after in zip(results[0], results[1], strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    for before, after in zip(unchanged, args[:7], strict=True):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert torch.equal(final, results[1][1])
