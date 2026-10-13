from __future__ import annotations

import pytest
import torch

from .test_cute_chained_loop_tmem_carry_transport import _resident_carry_sequence
from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_pipeline import _config


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width,cohorts,consumers", [(32, 2, 8), (64, 3, 4)])
def test_cohort_endpoint_fixture_exercises_both_storage_pools_cpu(
    dtype: torch.dtype,
    width: int,
    cohorts: int,
    consumers: int,
) -> None:
    args = (
        *(
            torch.zeros(shape, dtype=dtype)
            for shape in (
                (7, 128, 16),
                (7, 16, width),
                (7, width, 16),
                (7, 16, width),
            )
        ),
        torch.zeros((128, width)),
    )
    config = _config(16, pipeline=True, consumer_warps=consumers)
    config.config.update(
        cute_chained_warp_mma_rows=width,
        cute_chained_preparation_cohorts=cohorts,
        cute_chained_preparation_unroll=1,
    )
    source = _source(_resident_carry_sequence, args, config)
    assert "chain_resident_carry_" in source
    assert ("chain_endpoints =" in source) is (width == 32)
