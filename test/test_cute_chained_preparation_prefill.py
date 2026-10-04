from __future__ import annotations

from benchmarks.cute.kda_prefill_fused import kda_prefill_native_math
from benchmarks.cute.kda_prefill_fused_bt32 import kda_prefill_native_math_bt32
import pytest
import torch

from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_pipeline import _pipeline_plan
import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("bt32", [False, True])
@pytest.mark.parametrize("value_tile", [64, 128])
@pytest.mark.parametrize("consumer_warps", [4, 8])
def test_preparation_prefill_gpu_preserves_ragged_source_and_reused_slot(
    bt32, value_tile, consumer_warps
):
    _run_preparation_prefill(bt32, value_tile, consumer_warps)


def _run_preparation_prefill(
    bt32,
    value_tile,
    consumer_warps,
    *,
    rectangular_leaf=False,
    cache_entries=1,
    seed_columns=0,
    cache_layout="auto",
    vector_group=False,
    cohorts=1,
    preparation_unroll=0,
    register_islands=False,
    leaf_count=1,
    collective_retention=False,
    operand_retention=False,
    frontier_tile_columns=0,
    native_vector_reads=False,
):
    torch.manual_seed(935)
    lengths = (0, 1, 97, 161)
    width, heads = 128, 2
    shape = (1, sum(lengths), heads, width)
    q, k, v = (
        torch.randn(shape, device=DEVICE, dtype=torch.bfloat16) for _ in range(3)
    )
    initial = torch.randn((4, heads, width, width), device=DEVICE) * 0.1
    args = (
        q,
        k,
        v,
        torch.full_like(q, -8.0),
        torch.randn(shape[:-1], device=DEVICE, dtype=torch.bfloat16),
        torch.zeros(heads, device=DEVICE),
        torch.zeros((heads, width), device=DEVICE),
        initial,
        torch.empty_like(v),
        torch.empty_like(initial),
        torch.tensor((0, 0, 1, 98, 259), device=DEVICE, dtype=torch.int64),
        width**-0.5,
        -5 * 1.4426950408889634,
    )
    saved = tuple(value.clone() for value in (*args[:8], args[10]))
    reference_args = (
        *args[:8],
        torch.empty_like(args[8]),
        torch.empty_like(args[9]),
        *args[10:],
    )
    kernel = kda_prefill_native_math_bt32 if bt32 else kda_prefill_native_math
    reference = (
        helion.kernel(
            kernel.fn,
            backend="triton",
            static_shapes=True,
            fast_math=True,
            autotune_effort="none",
        )
        ._bind_isolated(reference_args)
        .compile_config(
            helion.Config(
                block_sizes=[64],
                num_warps=4,
                num_stages=2,
                indexing="pointer",
                pid_type="flat",
            )
        )
    )
    reference(*reference_args)
    candidate = helion.kernel(
        kernel.fn,
        backend="cute",
        static_shapes=True,
        fast_math=True,
        autotune_config_overrides={
            "cute_chained_mma_schedule": "tcgen05_tmem",
            "cute_chained_group_contractions": True,
        },
    )
    bound = candidate._bind_isolated(args)
    config = helion.Config(
        block_sizes=[value_tile],
        num_warps=16,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_group_contractions=True,
        cute_chained_warp_mma_rows=32,
        cute_chained_pointwise_cache_bytes=4096,
        cute_chained_pointwise_cache_entries=cache_entries,
        cute_chained_pointwise_cache_layout=cache_layout,
        cute_chained_seed_tile_columns=seed_columns,
        cute_chained_scan_schedule="warp",
        cute_chained_scratch_layout="xor",
        cute_chained_pointwise_vectorize=True,
        cute_chained_vector_group=vector_group,
        cute_chained_pointwise_unroll=8,
        cute_chained_preparation_pipeline=True,
        cute_chained_pipeline_consumer_warps=consumer_warps,
        cute_chained_leaf_pipeline="rectangular_tma" if rectangular_leaf else "legacy",
    )
    if cohorts != 1:
        config.config["cute_chained_preparation_cohorts"] = cohorts
    if preparation_unroll:
        config.config["cute_chained_preparation_unroll"] = preparation_unroll
    if register_islands:
        config.config["cute_chained_register_islands"] = True
    if leaf_count != 1:
        config.config["cute_chained_leaf_count"] = leaf_count
    if collective_retention:
        config.config["cute_chained_collective_retention"] = True
    if operand_retention:
        config.config["cute_chained_operand_retention"] = True
    if frontier_tile_columns:
        config.config["cute_chained_frontier_tile_columns"] = frontier_tile_columns
    if native_vector_reads:
        config.config["cute_chained_native_vector_reads"] = True
    with (
        bound.env.use_runtime_arg_values(_runtime_values(candidate, args)),
        _pipeline_plan(),
    ):
        source = bound.to_code(config)
        assert "chain_slot_bars" in source and "chunk_prefill" not in source
        assert ("chain_register_island" in source) is register_islands
        assert ("chain_retained_operand_" in source) is operand_retention
        if frontier_tile_columns:
            assert f"_group_step * {frontier_tile_columns}" in source
        if native_vector_reads:
            assert "chain_prepared_5_group_input_0_values" in source
        if collective_retention:
            assert "chain_row_collective_" in source
        elif vector_group:
            assert "chain_0_vector_group_output_2" in source
        if rectangular_leaf:
            assert "chained_rectangular_leaf_tma" in source
            assert "mbarrier_arrive_and_expect_tx" in source
        if cache_entries > 1:
            assert "chain_pointwise_cache_1" in source
        compiled = bound.compile_config(config)
    compiled(*args)
    torch.testing.assert_close(args[8], reference_args[8], atol=0.005, rtol=0.02)
    torch.testing.assert_close(args[9], reference_args[9], atol=0.01, rtol=0.02)
    torch.testing.assert_close(args[9][0], initial[0], atol=0, rtol=0)
    output, final = args[8].clone(), args[9].clone()
    args[8].fill_(float("nan"))
    args[9].fill_(float("nan"))
    compiled(*args)
    torch.testing.assert_close(args[8], output, atol=0, rtol=0)
    torch.testing.assert_close(args[9], final, atol=0, rtol=0)
    torch.testing.assert_close((*args[:8], args[10]), saved, atol=0, rtol=0)
