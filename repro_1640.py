"""
Standalone reproduction for issue #1640: Suboptimal config from autotuning.

Reproduces the silu_mul_fp8 kernel from vLLM without vLLM dependencies.
Tests problematic shapes (intermediate_size 2048/4096, num_tokens < 128)
comparing the autotuner-style configs (small block_sizes like [1,1]) vs
manually specified good configs from the vLLM PR fix.

Usage:
    # Compare bad (autotuner-picked) vs good (manual) configs — fast, no autotuning
    python repro_1640.py

    # Actually run autotuning on a single shape to see what the tuner picks
    python repro_1640.py --autotune --tokens 16 --isize 2048

    # AOT flow
    python -m helion.experimental.aot_runner -- python repro_1640.py --aot
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from triton.testing import do_bench

import helion
import helion.language as hl

# ============================================================================
# Kernel: silu_mul_fp8 copied from vLLM (no vLLM deps)
# ============================================================================


@helion.kernel(static_shapes=False)
def silu_mul_fp8(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    original_shape = input.shape
    two_d = hl.specialize(original_shape[-1])
    d = two_d // 2
    output_shape = original_shape[:-1] + (d,)

    input_2d = input.view(-1, original_shape[-1])
    m = input_2d.shape[0]

    out = torch.empty((m, d), device=input.device, dtype=torch.float8_e4m3fn)

    input_part_a = input_2d[:, :d]
    input_part_b = input_2d[:, d:]

    assert scale.numel() == 1, "Scale must be a scalar Tensor"

    for tile_m, tile_n in hl.tile([m, d]):
        a_vals = input_part_a[tile_m, tile_n]
        silu_result = torch.nn.functional.silu(a_vals)
        b_vals = input_part_b[tile_m, tile_n]
        result = silu_result * b_vals
        result_f32 = result.to(torch.float32)
        scale_val = hl.load(scale, [0])
        inv_scale = 1.0 / scale_val
        result_scaled = result_f32 * inv_scale
        out[tile_m, tile_n] = result_scaled.to(out.dtype)

    return out.view(output_shape)


# ============================================================================
# AOT kernel variant
# ============================================================================

try:
    import helion.experimental

    @helion.experimental.aot_kernel(batched=[[0, None], None])
    def silu_mul_fp8_aot(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        original_shape = input.shape
        two_d = hl.specialize(original_shape[-1])
        d = two_d // 2
        output_shape = original_shape[:-1] + (d,)

        input_2d = input.view(-1, original_shape[-1])
        m = input_2d.shape[0]

        out = torch.empty((m, d), device=input.device, dtype=torch.float8_e4m3fn)

        input_part_a = input_2d[:, :d]
        input_part_b = input_2d[:, d:]

        assert scale.numel() == 1, "Scale must be a scalar Tensor"

        for tile_m, tile_n in hl.tile([m, d]):
            a_vals = input_part_a[tile_m, tile_n]
            silu_result = torch.nn.functional.silu(a_vals)
            b_vals = input_part_b[tile_m, tile_n]
            result = silu_result * b_vals
            result_f32 = result.to(torch.float32)
            scale_val = hl.load(scale, [0])
            inv_scale = 1.0 / scale_val
            result_scaled = result_f32 * inv_scale
            out[tile_m, tile_n] = result_scaled.to(out.dtype)

        return out.view(output_shape)

    HAS_AOT = True
except Exception:
    HAS_AOT = False


# ============================================================================
# Configs
# ============================================================================

# "Bad" configs: actual configs from vLLM's original autotuning on H200
# (before PR #36062 fix). These were produced by helion's autotuner but
# are suboptimal — small block_n, wrong num_warps for the workload size,
# unnecessary multi-staging. Root cause: measurement noise at microsecond
# runtimes makes the autotuner unable to reliably distinguish configs.
BAD_CONFIGS: dict[tuple[int, int], dict] = {
    (1, 2048): {"block_sizes": [1, 16], "num_warps": 16, "num_stages": 2},
    (2, 2048): {"block_sizes": [2, 16], "num_warps": 2, "num_stages": 1},
    (4, 2048): {"block_sizes": [1, 256], "num_warps": 32, "num_stages": 2},
    (8, 2048): {"block_sizes": [8, 256], "num_warps": 32, "num_stages": 1},
    (16, 2048): {"block_sizes": [16, 64], "num_warps": 4, "num_stages": 2},
    (32, 2048): {"block_sizes": [32, 64], "num_warps": 1, "num_stages": 1},
    (64, 2048): {"block_sizes": [16, 128], "num_warps": 8, "num_stages": 1},
    (128, 2048): {"block_sizes": [32, 64], "num_warps": 2, "num_stages": 2},
    (1, 4096): {"block_sizes": [1, 32], "num_warps": 4, "num_stages": 2},
    (2, 4096): {"block_sizes": [2, 32], "num_warps": 8, "num_stages": 2},
    (4, 4096): {"block_sizes": [4, 16], "num_warps": 4, "num_stages": 1},
    (8, 4096): {"block_sizes": [2, 32], "num_warps": 32, "num_stages": 1},
    (16, 4096): {"block_sizes": [16, 32], "num_warps": 8, "num_stages": 2},
    (32, 4096): {"block_sizes": [32, 16], "num_warps": 1, "num_stages": 2},
    (64, 4096): {"block_sizes": [2, 16], "num_warps": 1, "num_stages": 1},
    (128, 4096): {"block_sizes": [128, 32], "num_warps": 32, "num_stages": 2},
}

# "Good" configs: from vLLM PR #36062 manual re-tuning.
# Key improvements: larger block_n, appropriate num_warps, num_stages=1.
GOOD_CONFIGS: dict[tuple[int, int], dict] = {
    (1, 2048): {"block_sizes": [1, 256], "num_warps": 2, "num_stages": 1},
    (2, 2048): {"block_sizes": [2, 128], "num_warps": 2, "num_stages": 1},
    (4, 2048): {"block_sizes": [4, 64], "num_warps": 2, "num_stages": 1},
    (8, 2048): {"block_sizes": [8, 32], "num_warps": 2, "num_stages": 1},
    (16, 2048): {"block_sizes": [8, 512], "num_warps": 16, "num_stages": 2},
    (32, 2048): {"block_sizes": [32, 16], "num_warps": 4, "num_stages": 1},
    (64, 2048): {"block_sizes": [64, 32], "num_warps": 8, "num_stages": 1},
    (128, 2048): {"block_sizes": [128, 16], "num_warps": 8, "num_stages": 1},
    (1, 4096): {"block_sizes": [1, 256], "num_warps": 2, "num_stages": 1},
    (2, 4096): {"block_sizes": [2, 128], "num_warps": 2, "num_stages": 1},
    (4, 4096): {"block_sizes": [4, 64], "num_warps": 2, "num_stages": 1},
    (8, 4096): {"block_sizes": [8, 32], "num_warps": 2, "num_stages": 1},
    (16, 4096): {"block_sizes": [16, 256], "num_warps": 32, "num_stages": 1},
    (32, 4096): {"block_sizes": [4, 4096], "num_warps": 32, "num_stages": 5},
    (64, 4096): {"block_sizes": [64, 32], "num_warps": 8, "num_stages": 1},
    (128, 4096): {"block_sizes": [128, 32], "num_warps": 8, "num_stages": 1},
}


def make_inputs(num_tokens: int, intermediate_size: int):
    input_tensor = torch.randn(
        num_tokens,
        2 * intermediate_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    return input_tensor, scale


def bench_with_config(num_tokens, intermediate_size, config):
    """Benchmark a single (shape, config) pair. Returns time in ms."""
    inp, scale = make_inputs(num_tokens, intermediate_size)
    # Reset to avoid BoundKernel cache aliasing between different configs
    silu_mul_fp8.reset()
    bound = silu_mul_fp8.bind((inp, scale))
    compiled_fn = bound.compile_config(config)
    compiled_fn(inp, scale)  # warmup
    return do_bench(lambda: compiled_fn(inp, scale))


def run_comparison() -> None:
    """Compare bad (autotuner-style) vs good (manual) configs."""
    intermediate_sizes = [2048, 4096]
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128]

    print("=" * 72)
    print("Issue #1640: silu_mul_fp8 — autotuner-style vs manual configs")
    print("=" * 72)
    print()
    print("'bad' = original autotuned configs from vLLM (before PR #36062)")
    print("'good' = manually re-tuned configs from vLLM PR #36062")
    print()
    print(
        f"{'tokens':>8} {'isize':>6} {'bad_ms':>9} {'good_ms':>9} "
        f"{'ratio':>7} {'status':>8}"
    )
    print("-" * 52)

    for intermediate_size in intermediate_sizes:
        for num_tokens in num_tokens_list:
            key = (num_tokens, intermediate_size)
            bad_cfg = BAD_CONFIGS[key]
            good_cfg = GOOD_CONFIGS[key]

            t_bad = bench_with_config(num_tokens, intermediate_size, bad_cfg)
            t_good = bench_with_config(num_tokens, intermediate_size, good_cfg)

            ratio = t_bad / t_good if t_good > 0 else float("inf")
            status = "OK" if ratio < 1.2 else "SLOW" if ratio < 2.0 else "BAD"
            print(
                f"{num_tokens:>8} {intermediate_size:>6} {t_bad:>9.4f} "
                f"{t_good:>9.4f} {ratio:>7.2f}x {status:>8}"
            )
        print()


def run_autotune_single(num_tokens: int, intermediate_size: int) -> None:
    """Autotune a single shape and show what the autotuner picks."""
    print(f"Autotuning: num_tokens={num_tokens}, intermediate_size={intermediate_size}")
    print()

    inp, scale = make_inputs(num_tokens, intermediate_size)

    # This triggers autotuning
    out = silu_mul_fp8(inp, scale)
    t_auto = do_bench(lambda: silu_mul_fp8(inp, scale))

    print(f"\nAutotuned time: {t_auto:.4f} ms")

    # Compare with manual config
    key = (num_tokens, intermediate_size)
    good_cfg = GOOD_CONFIGS.get(key)
    if good_cfg:
        bound_good = silu_mul_fp8.bind((inp, scale))
        bound_good.set_config(good_cfg)
        bound_good(inp, scale)  # warmup
        t_good = do_bench(lambda: bound_good(inp, scale))
        ratio = t_auto / t_good
        print(f"Manual config time: {t_good:.4f} ms")
        print(f"Manual config: {good_cfg}")
        print(f"Ratio (auto/manual): {ratio:.2f}x")
        if ratio > 1.2:
            print(">>> AUTOTUNER PICKED A SUBOPTIMAL CONFIG <<<")
        else:
            print("Autotuner found a good config!")


def run_block_size_sweep() -> None:
    """Sweep block sizes for a single shape to show impact of block_sizes."""
    num_tokens, intermediate_size = 128, 4096
    d = intermediate_size  # half of 2*intermediate_size
    inp, scale = make_inputs(num_tokens, intermediate_size)

    configs = [
        ("tiny [1,1]", {"block_sizes": [1, 1], "num_warps": 4, "num_stages": 1}),
        ("small [2,16]", {"block_sizes": [2, 16], "num_warps": 1, "num_stages": 1}),
        ("small [4,8]", {"block_sizes": [4, 8], "num_warps": 2, "num_stages": 1}),
        ("med [32,64]", {"block_sizes": [32, 64], "num_warps": 8, "num_stages": 1}),
        ("good [128,32]", {"block_sizes": [128, 32], "num_warps": 8, "num_stages": 1}),
        (
            "vllm_old w32s2",
            {"block_sizes": [128, 32], "num_warps": 32, "num_stages": 2},
        ),
    ]

    print(f"Block size sweep: tokens={num_tokens}, isize={intermediate_size}")
    print(f"{'config':<20} {'time_ms':>10} {'num_blocks':>12}")
    print("-" * 45)
    for name, cfg in configs:
        bound = silu_mul_fp8.bind((inp, scale))
        bound.set_config(cfg)
        bound(inp, scale)
        t = do_bench(lambda: bound(inp, scale))
        bm, bn = cfg["block_sizes"]
        nblocks = ((num_tokens + bm - 1) // bm) * ((d + bn - 1) // bn)
        print(f"{name:<20} {t:>10.4f} {nblocks:>12}")


def run_aot_benchmark() -> None:
    """Run benchmark using the AOT kernel (for use with aot_runner)."""
    if not HAS_AOT:
        print("AOT support not available")
        sys.exit(1)

    intermediate_sizes = [2048, 4096]
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128]

    print(f"{'tokens':>8} {'isize':>8} {'time_ms':>10}")
    print("-" * 30)
    for intermediate_size in intermediate_sizes:
        for num_tokens in num_tokens_list:
            inp, scale = make_inputs(num_tokens, intermediate_size)
            out = silu_mul_fp8_aot(inp, scale)
            t = do_bench(lambda: silu_mul_fp8_aot(inp, scale))
            print(f"{num_tokens:>8} {intermediate_size:>8} {t:>10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #1640 reproduction")
    parser.add_argument(
        "--autotune", action="store_true", help="Run full autotuning on a single shape"
    )
    parser.add_argument(
        "--tokens", type=int, default=1, help="num_tokens for --autotune mode"
    )
    parser.add_argument(
        "--isize", type=int, default=2048, help="intermediate_size for --autotune mode"
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Sweep block sizes to show impact"
    )
    parser.add_argument("--aot", action="store_true", help="Use AOT kernel variant")
    args = parser.parse_args()

    aot_mode = os.environ.get("HELION_AOT_MODE", "disabled")

    if args.aot or aot_mode != "disabled":
        run_aot_benchmark()
    elif args.sweep:
        run_block_size_sweep()
    elif args.autotune:
        run_autotune_single(args.tokens, args.isize)
    else:
        run_comparison()


if __name__ == "__main__":
    main()
