from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cute_optional_lse import bind_paged_attention_length_replay
from helion._testing.cute_optional_lse import check_optional_lse_replay
from helion._testing.cute_optional_lse import check_paged_attention_length_replay
from helion._testing.cute_paged_positive_zero import positive_zero_attention_fixture
from helion._testing.cute_paged_v_lag import paired_v_lag_config


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "query_len,post_scale,probability,lag,query_tma,sleep,capture,fast_math,lse,overlap",
    [
        (1, -14, "serial", 0, False, False, False, False, True, False),
        (4, -22, "serial", 3, True, True, False, True, False, True),
        (5, -22, "early", 1, True, False, True, False, True, True),
        (1, -22, "early_tmem", 2, False, True, False, True, True, True),
        (4, -14, "early", 2, False, False, False, False, False, False),
        (5, -22, "early_tmem", 3, True, True, True, True, True, False),
    ],
)
def test_metadata_schedule_preserves_source_and_changed_replay(
    query_len,
    post_scale,
    probability,
    lag,
    query_tma,
    sleep,
    capture,
    fast_math,
    lse,
    overlap,
):
    # Six configurations cover N8/N32/N40, all four V lags, the three
    # probability schedules and both source exponent boundaries. Optional query
    # TMA, sleep and capture overlap are paired rather than multiplied.
    # Each metadata-read placement covers N8/N32/N40 and all three probability
    # schedules. The slot and TMA fixtures also exercise both placements directly.
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention uses SM100 tensor memory")
    kernel, initial = positive_zero_attention_fixture(query_len, post_scale, DEVICE)
    initial = (*initial[:9], lse, *initial[10:])
    bound, args = bind_paged_attention_length_replay(
        kernel, initial, "mapped", fast_math=fast_math, reverse_pages=True
    )
    config = paired_v_lag_config(lag)
    config.config.update(
        block_sizes=[helion.next_power_of_2(query_len * 8)],
        cute_mma_dag_value_metadata_skip=True,
        cute_mma_dag_value_metadata_read_overlap=overlap,
        cute_mma_dag_probability_schedule=probability,
        cute_mma_dag_query_tma=query_tma,
        cute_mma_dag_sleep_waits=sleep,
        cute_pdl=capture,
        cute_pdl_capture_overlap=capture,
    )
    source = bound.to_code(config)
    assert "paged_attention_paired_body(" in source
    assert "value_metadata_scales=cute.make_tensor(" in source
    assert "value_positive_zero=True" in source
    assert ("value_metadata_read_overlap=True" in source) is overlap
    options = {
        "reference_scores": True,
        "reverse_pages": True,
        "base2_scores": True,
        "output_reciprocal": True,
        "key_post_scale_log2": -6,
        "value_post_scale_log2": post_scale,
        "output_scale_log2": -post_scale,
        "value_positive_zero": True,
    }
    # Shared references retain the exact BF16 decode/postscale/+0 ordering;
    # each configuration compiles once, then mutates inputs and graph outputs.
    check_optional_lse_replay(bound, config, args, "mapped", lse, **options)
    check_paged_attention_length_replay(bound, config, args, "mapped", lse, **options)
