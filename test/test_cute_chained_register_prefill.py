from __future__ import annotations

import pytest

from .test_cute_chained_loop_tmem_transport import _source
from .test_cute_chained_preparation_cut import _kda_fixture
from .test_cute_chained_preparation_prefill import _run_preparation_prefill
from .test_cute_chained_register_emission import _register_config
from helion._testing import skipUnlessBackends


@pytest.mark.parametrize("tma", [False, True])
@pytest.mark.parametrize("value_tile,consumer_warps,cohorts", [(64, 8, 2), (128, 4, 3)])
def test_register_island_binds_after_final_native_and_tma_frame_cpu(
    tma, value_tile, consumer_warps, cohorts
):
    kernel, args = _kda_fixture()
    config = _register_config(
        cohorts,
        block_sizes=[value_tile],
        cute_chained_pipeline_consumer_warps=consumer_warps,
        cute_chained_seed_tile_columns=32,
        cute_chained_pointwise_cache_layout="xor",
        cute_chained_vector_group=True,
        cute_chained_leaf_pipeline="rectangular_tma" if tma else "legacy",
        cute_chained_register_islands=True,
    )
    source = _source(kernel, args, config)
    assert "chain_register_island" in source
    assert "chain_2_a_ptr" not in source
    assert "chain_generation = chain_iteration //" in source
    assert ("mbarrier_arrive_and_expect_tx" in source) is tma


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("tma", [False, True])
@pytest.mark.parametrize("value_tile,consumer_warps,cohorts", [(64, 8, 2), (128, 4, 3)])
def test_register_island_prefill_keeps_original_fp32_and_replay_oracles_gpu(
    tma, value_tile, consumer_warps, cohorts
):
    _run_preparation_prefill(
        True,
        value_tile,
        consumer_warps,
        rectangular_leaf=tma,
        seed_columns=32,
        cache_layout="xor",
        vector_group=True,
        cohorts=cohorts,
        preparation_unroll=1,
        register_islands=True,
    )
