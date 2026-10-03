from __future__ import annotations

import pytest
import torch

from .test_cute_chained_loop_tmem_carry_transport import _resident_carry_sequence
from .test_cute_chained_preparation_cut import _runtime_values
from .test_cute_chained_preparation_prefill import _run_preparation_prefill
from .test_cute_chained_seed_tile_integration import _config
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("bt32", [False, True])
@pytest.mark.parametrize("value_tile", [64, 128])
@pytest.mark.parametrize("consumer_warps", [4, 8])
def test_seed_tile_prefill_gpu_preserves_ragged_source_and_fp32_state(
    bt32, value_tile, consumer_warps
):
    _run_preparation_prefill(bt32, value_tile, consumer_warps, seed_columns=32)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("bt32", [False, True])
def test_seed_tiles_cache_sets_and_tma_gpu(bt32):
    _run_preparation_prefill(
        bt32, 128, 4, rectangular_leaf=True, cache_entries=4, seed_columns=32
    )


@skipUnlessBackends(["cute"])
def test_seed_tile64_prefill_gpu():
    _run_preparation_prefill(True, 128, 4, seed_columns=64)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_seed_tiles_resident_fp32_gpu_are_bitwise_equal(dtype):
    torch.manual_seed(9962)
    steps, width = 5, 64
    args = (
        *(
            torch.randn(shape, device=DEVICE, dtype=dtype) * 0.125
            for shape in (
                (steps, 128, 16),
                (steps, 16, width),
                (steps, width, 16),
                (steps, 16, width),
            )
        ),
        torch.randn((128, width), device=DEVICE) * 0.125,
    )
    saved = tuple(arg.clone() for arg in args)
    compiled = []
    for columns in (0, 32):
        config = _config(columns, pipeline=True)
        config.config.pop("block_sizes")
        # This graph has no shared-LHS contraction group or XOR-eligible scratch.
        config.config.pop("cute_chained_group_contractions")
        config.config.pop("cute_chained_scratch_layout")
        config.config.update(
            cute_chained_warp_mma_rows=64,
            cute_chained_pointwise_cache_bytes=0,
            cute_chained_pointwise_unroll=1,
            cute_chained_pointwise_vectorize=False,
            cute_chained_scan_schedule="serial",
        )
        bound = _resident_carry_sequence._bind_isolated(args)
        with bound.env.use_runtime_arg_values(
            _runtime_values(_resident_carry_sequence, args)
        ):
            source = bound.to_code(config)
            assert "chain_resident_carry_" in source
            if columns:
                assert source.count("chain_2_seed_2_segment =") == 2
            compiled.append(bound.compile_config(config))
    expected, actual = compiled[0](*args), compiled[1](*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(compiled[1](*args), actual, atol=0, rtol=0)
    torch.testing.assert_close(args, saved, atol=0, rtol=0)
