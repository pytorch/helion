from __future__ import annotations

from benchmarks.cute.kda_prefill_fused import kda_prefill_native_math
from benchmarks.cute.kda_prefill_fused_bt32 import kda_prefill_native_math_bt32
import pytest
import torch

from ._cute_aux import _cpu_codegen
import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends


def _inputs(device: str | torch.device) -> tuple:
    lengths = (0, 1, 17, 33)
    tokens, heads, width = sum(lengths), 2, 128
    shape = (1, tokens, heads, width)
    q, k, v = (
        torch.randn(shape, device=device, dtype=torch.bfloat16) for _ in range(3)
    )
    gate = torch.full_like(q, -8.0)
    beta = torch.randn(shape[:-1], device=device, dtype=torch.bfloat16)
    a_log = torch.zeros(heads, device=device)
    bias = torch.zeros((heads, width), device=device)
    initial = torch.randn((len(lengths), heads, width, width), device=device) * 0.1
    cu = torch.tensor((0, 0, 1, 18, 51), dtype=torch.int64, device=device)
    return (
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        bias,
        initial,
        torch.empty_like(v),
        torch.empty_like(initial),
        cu,
        width**-0.5,
        -5 * 1.4426950408889634,
    )


def _config(value_tile: int = 64) -> helion.Config:
    return helion.Config(
        block_sizes=[value_tile],
        num_warps=4,
        cute_chained_mma_schedule="tcgen05_tmem",
        cute_chained_group_contractions=True,
    )


@pytest.mark.parametrize("bt32", [False, True])
def test_prefill_uses_shared_loop_without_dispatch_mock(bt32: bool) -> None:
    kernel = kda_prefill_native_math_bt32 if bt32 else kda_prefill_native_math
    with _cpu_codegen():
        bound = kernel._bind_isolated(_inputs("cpu"))
        source = bound.to_code(_config())
    assert "chain_loop_index" in source and "chain_0_mma" in source
    assert "chunk_prefill" not in source


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("bt32", [False, True])
@pytest.mark.parametrize("value_tile", [64, 128])
@pytest.mark.parametrize("cache_bytes", [0, 4096])
def test_shared_prefill_gpu_preserves_source_and_ragged_state(
    bt32: bool, value_tile: int, cache_bytes: int, warp_rows: int = 0
) -> None:
    torch.manual_seed(938)
    kernel = kda_prefill_native_math_bt32 if bt32 else kda_prefill_native_math
    args = _inputs(DEVICE)
    frozen = tuple(value.clone() for value in (*args[:8], args[10]))
    reference_args = (
        *args[:8],
        torch.empty_like(args[8]),
        torch.empty_like(args[9]),
        *args[10:],
    )
    reference_kernel = helion.kernel(
        kernel.fn,
        backend="triton",
        static_shapes=True,
        fast_math=True,
        autotune_effort="none",
    )
    reference = reference_kernel._bind_isolated(reference_args).compile_config(
        helion.Config(
            block_sizes=[64],
            num_warps=4,
            num_stages=2,
            indexing="pointer",
            pid_type="flat",
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
    config = _config(value_tile)
    if cache_bytes:
        config = helion.Config.from_dict(
            {
                **config.config,
                "num_warps": 16,
                "cute_chained_pointwise_cache_bytes": cache_bytes,
                "cute_chained_scan_schedule": "warp",
                "cute_chained_scratch_layout": "xor",
                "cute_chained_pointwise_vectorize": True,
            }
        )
    if warp_rows:
        config = helion.Config.from_dict(
            config.config | {"cute_chained_warp_mma_rows": warp_rows}
        )
    source = bound.to_code(config)
    assert "chain_loop_index" in source and "chunk_prefill" not in source
    assert ("chain_pointwise_cache_0" in source) == bool(cache_bytes)
    assert ("chain_0_warp_mma" in source) == bool(warp_rows)
    compiled = bound.compile_config(config)
    compiled(*args)
    torch.testing.assert_close(args[8], reference_args[8], atol=0.005, rtol=0.02)
    torch.testing.assert_close(args[9], reference_args[9], atol=0.01, rtol=0.02)
    torch.testing.assert_close(args[9][0], args[7][0], atol=0, rtol=0)
    output, final = args[8].clone(), args[9].clone()
    args[8].fill_(float("nan"))
    args[9].fill_(float("nan"))
    compiled(*args)
    torch.testing.assert_close(args[8], output, atol=0, rtol=0)
    torch.testing.assert_close(args[9], final, atol=0, rtol=0)
    for actual, original in zip((*args[:8], args[10]), frozen, strict=True):
        torch.testing.assert_close(actual, original, atol=0, rtol=0)
