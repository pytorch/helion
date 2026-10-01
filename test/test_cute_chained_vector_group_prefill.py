from __future__ import annotations

import pytest

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_prefill import _run_preparation_prefill
from .test_cute_chained_seed_tile_integration import _config
from helion._testing import skipUnlessBackends


@pytest.mark.parametrize("value_tile", [64, 128])
@pytest.mark.parametrize("consumer_warps", [4, 8])
def test_vector_group_prefill_cpu_uses_common_three_output_producer(
    value_tile, consumer_warps
):
    kernel, args = _kda_fixture()
    config = _config(32, pipeline=True, value_tile=value_tile)
    config.config.update(
        cute_chained_vector_group=True,
        cute_chained_pipeline_consumer_warps=consumer_warps,
        cute_chained_pointwise_cache_layout="xor",
    )
    source = _source(kernel, args, config)
    assert "chain_0_vector_group_output_2" in source
    assert "chain_0_a_0_vector_step" not in source
    assert "chain_0_b_1_vector_step" not in source
    assert "chain_slot_bars" in source


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("bt32", [False, True])
@pytest.mark.parametrize("value_tile", [64, 128])
@pytest.mark.parametrize("consumer_warps", [4, 8])
def test_vector_group_prefill_gpu_preserves_fp32_state_and_ragged_masks(
    bt32, value_tile, consumer_warps
):
    _run_preparation_prefill(
        bt32,
        value_tile,
        consumer_warps,
        seed_columns=32,
        cache_layout="xor",
        vector_group=True,
    )


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("bt32", [False, True])
def test_vector_group_prefill_gpu_keeps_rectangular_tma_protocol(bt32):
    _run_preparation_prefill(
        bt32,
        128,
        4,
        rectangular_leaf=True,
        seed_columns=32,
        cache_layout="xor",
        vector_group=True,
    )
