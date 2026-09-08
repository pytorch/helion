"""Standalone reproducer for the head_dim=128 causal CuTe flash attention win.

Runs `examples.attention.causal_attention_output` on the GB300 (sm103) shape the
PR is measured on, with the exact config the compiler-promoted resident seed
resolves to, and reports TFLOP/s against torch SDPA.

    python benchmarks/cute/kernel.py

BEST_CONFIG below is the compiler-promoted resident seed for this shape, exactly
as the harness resolves it from ``config_spec.compiler_seed_configs``. It must be
passed verbatim: the resident softmax lowering only engages when the config
matches the promoted seed exactly (``causal_seed_matches``), so changing any
single field silently drops the kernel to the STANDARD lowering and costs ~11%.
"""

from __future__ import annotations

import argparse
import math

import torch

import helion
from examples.attention import causal_attention_output

# Shape the PR measures: z=2, h=32, head_dim=128, FP16 causal.
Z, H, SEQ_LEN, HEAD_DIM = 2, 32, 65536, 128

# Effective config resolved by the promoted resident seed on sm103 for this
# shape. Captured from the harness JSON of the measured run.
#
# Note the resident lowering rewrites several of these internally (see
# `_flash_resident_softmax_config`): softmax_disc -> False, disc_pipe -> 1,
# e2e_schedule -> "xu", stat_transport -> "single". The values below are the
# requested config, which is what `causal_seed_matches` compares against.
BEST_CONFIG = helion.Config(
    block_sizes=[1, 128, 128],
    cute_flash_causal_kv_order="descending",
    cute_flash_causal_loop_split=True,
    cute_flash_causal_lpt_swizzle=1,
    cute_flash_clc_heads_per_batch=0,
    cute_flash_clc_pdl=False,
    cute_flash_clc_stages=1,
    cute_flash_corr_regs=64,
    cute_flash_corr_tile_size=16,
    cute_flash_disc_pipe=2,
    cute_flash_e2e_offset=0,
    cute_flash_e2e_offset0=0,
    cute_flash_e2e_schedule="8/2",
    cute_flash_epi_stg=False,
    cute_flash_epi_stg_gmem="stage",
    cute_flash_epi_stg_store="slice",
    cute_flash_epi_tma=False,
    cute_flash_epi_tma_setup="shared",
    cute_flash_exp2_packet="1x1",
    cute_flash_first_load_order=2,
    cute_flash_kv_order="ascending",
    cute_flash_kv_stage=3,
    cute_flash_masked_e2e_schedule="inherit",
    cute_flash_mma_interleave=True,
    cute_flash_other_regs=48,
    cute_flash_p_store_rep=16,
    cute_flash_packed_reduce=True,
    cute_flash_persistent=False,
    cute_flash_persistent_ctas_per_sm=1,
    cute_flash_persistent_loop="while",
    cute_flash_pipeline_family="fa4",
    cute_flash_precompute_qk_desc=False,
    cute_flash_q_tile_count=2,
    cute_flash_recompute_tile_coords=False,
    cute_flash_rescale_chunk_cols=16,
    cute_flash_rescale_threshold=8.0,
    cute_flash_role_chain=False,
    cute_flash_role_map="fa4",
    cute_flash_s_load_rep=32,
    cute_flash_s_stage=2,
    cute_flash_skip_rescale_stats=False,
    cute_flash_small_biased=True,
    cute_flash_softmax_disc=True,
    cute_flash_softmax_regs=200,
    cute_flash_softmax_setup="shared",
    cute_flash_sp_row_sum="fragment",
    cute_flash_split_p_arrive=True,
    cute_flash_stat_transport="ring2",
    cute_flash_wait_hint=0,
)


def causal_flops(z: int, h: int, seq_len: int, head_dim: int) -> float:
    """QK and PV matmuls only, halved for the causal mask (harness FLOP model)."""
    return 4.0 * z * h * seq_len * seq_len * head_dim * 0.5


def benchmark(fn, *args, warmup_ms: int = 1000, rep_ms: int = 500) -> float:
    """Median milliseconds via Triton do_bench, matching the harness protocol."""
    from triton.testing import do_bench

    return do_bench(lambda: fn(*args), warmup=warmup_ms, rep=rep_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = parser.parse_args()

    torch.manual_seed(0)
    shape = (Z, H, args.seq_len, HEAD_DIM)
    q, k, v = (
        torch.randn(shape, dtype=torch.float16, device="cuda") for _ in range(3)
    )

    kernel = helion.kernel(
        causal_attention_output.fn,
        config=BEST_CONFIG,
        static_shapes=True,
        backend="cute",
    )

    out = kernel(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=1.0 / math.sqrt(HEAD_DIM), is_causal=True
    )
    torch.testing.assert_close(out, expected, atol=5e-2, rtol=2e-2)

    flops = causal_flops(Z, H, args.seq_len, HEAD_DIM)
    helion_ms = benchmark(kernel, q, k, v)
    sdpa_ms = benchmark(
        lambda a, b, c: torch.nn.functional.scaled_dot_product_attention(
            a, b, c, scale=1.0 / math.sqrt(HEAD_DIM), is_causal=True
        ),
        q,
        k,
        v,
    )

    print(f"shape z={Z} h={H} seq_len={args.seq_len} head_dim={HEAD_DIM} fp16 causal")
    print(f"helion-cute         {flops / (helion_ms * 1e9):8.1f} TF/s")
    print(f"torch SDPA          {flops / (sdpa_ms * 1e9):8.1f} TF/s")


if __name__ == "__main__":
    main()
