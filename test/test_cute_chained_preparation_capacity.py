from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import create_autospec
from unittest.mock import patch

import pytest
import torch

from .test_cute_chained_compact_preparation import _twenty
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_scan_producer import _scan_sequence
from .test_cute_chained_scan_producer import _sequence_args
from .test_cute_chained_scan_producer import _sequence_config
from helion._compiler.cute import chained_pipeline_storage as storage
from helion._compiler.cute.chained_matmul import ChainedMatmulPlan
from helion._compiler.cute.chained_preparation_pipeline import PreparationPipeline
from helion._compiler.cute.chained_preparation_pipeline import _frame_planning_capacity
from helion._compiler.cute.chained_preparation_storage import bind_preparation_storage
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config


@pytest.mark.parametrize("capacity", (232448, 233023))
@pytest.mark.parametrize("slots", (2, 3, 4))
@pytest.mark.parametrize("has_cohorts", (False, True))
@pytest.mark.parametrize("compact", (False, True))
def test_semantic_frame_ceiling_is_not_a_physical_slot_quota(
    capacity, slots, has_cohorts, compact
):
    plan = create_autospec(ChainedMatmulPlan, instance=True)
    plan.dots = (SimpleNamespace(meta={"val": torch.empty(0)}),)
    pipeline = create_autospec(PreparationPipeline, instance=True)
    pipeline.slots = slots
    pipeline.cohorts = object() if has_cohorts else None
    with patch.object(
        CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=capacity
    ):
        result = _frame_planning_capacity(plan, pipeline, compact_preparation=compact)
        default = _frame_planning_capacity(plan, pipeline)
    assert default == capacity // slots // 128 * 128
    divisor = 1 if compact and has_cohorts else slots
    assert result == capacity // divisor // 128 * 128
    assert result <= capacity and result % 128 == 0


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_compact_finalizer_still_charges_all_slots_and_exact_capacity(dtype):
    config = _sequence_config()
    config.config.update(_twenty().config)
    original = storage.finalize_pipeline_storage
    checked = []

    def observe(plan, pipeline, stages, **kwargs):
        result = original(plan, pipeline, stages, **kwargs)
        assert result is not None and result.preparation is kwargs["preparation"]
        preparation = result.preparation
        assert preparation is not None
        assert dict(result.allocations)["frames"] == 4 * preparation.stride
        assert result.charged_bytes > 4 * preparation.stride
        for quota, succeeds in (
            (result.charged_bytes - 1, False),
            (result.charged_bytes, True),
        ):
            fresh = bind_preparation_storage(
                plan,
                pipeline,
                preparation.physical,
                cache_layouts=preparation.cache_layouts,
            )
            assert fresh is not None
            trial_kwargs = dict(kwargs)
            trial_kwargs.update(preparation=fresh, capacity_bytes=quota)
            trial = original(
                plan,
                pipeline,
                stages,
                **trial_kwargs,
            )
            assert (trial is not None) is succeeds
            assert (fresh._state.finalized is not None) is succeeds
            assert not fresh._state.consumed
        checked.append(result)
        return result

    with patch.object(storage, "finalize_pipeline_storage", observe):
        source = _source(_scan_sequence, _sequence_args(dtype), config)
    assert "block=(640, 1, 1)" in source
    assert len(checked) == 1
    preparation = checked[0].preparation
    assert preparation is not None and preparation._state.consumed
