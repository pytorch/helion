"""
AOT Benchmark for Linear Attention Kernels
===========================================

Exercises all Helion linear attention kernels (forward + backward) at multiple
problem sizes for AOT autotuning via helion.experimental.aot_kernel.

Run the full AOT workflow::

    python -m helion.experimental.aot_runner \\
        -- python -m examples.linear.aot_benchmark

Individual phases::

    python -m helion.experimental.aot_runner --phase collect \\
        -- python -m examples.linear.aot_benchmark
    python -m helion.experimental.aot_runner --phase measure \\
        -- python -m examples.linear.aot_benchmark
    python -m helion.experimental.aot_runner --phase build \\
        -- python -m examples.linear.aot_benchmark
    python -m helion.experimental.aot_runner --phase evaluate \\
        -- python -m examples.linear.aot_benchmark

Multi-GPU parallel collect (8 GPUs, each gets a slice of configs)::

    for gpu in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$gpu python -m examples.linear.aot_benchmark \\
            --shard $gpu --num-shards 8 &
    done
    wait
"""

from __future__ import annotations

import argparse
import os

import torch
from triton.testing import do_bench

from .linear_attention_engine import chunked_linear_attn
from .linear_attention_utils import make_delta_rule_inputs
from .linear_attention_utils import make_full_gla_inputs
from .linear_attention_utils import make_gated_delta_rule_inputs
from .linear_attention_utils import make_mamba2_inputs
from .linear_attention_utils import make_retention_inputs
from .linear_attention_utils import make_simple_gla_inputs
from .linear_attention_utils import make_vanilla_linear_attn_inputs
from helion._testing import DEVICE

CONFIGS = [
    # (B, H, T, D, DV) — focused set for fast AOT pipeline
    (1, 8, 512, 64, 32),
    (1, 32, 1024, 128, 128),
    (1, 32, 2048, 128, 128),
]

CHUNK_SIZES = [64]

VARIANTS = {
    "simple_gla": (make_simple_gla_inputs, False),
    "full_gla": (make_full_gla_inputs, False),
    "delta_rule": (make_delta_rule_inputs, True),
    "gated_delta": (make_gated_delta_rule_inputs, True),
    "vanilla": (make_vanilla_linear_attn_inputs, False),
    "retention": (make_retention_inputs, False),
    "mamba2": (make_mamba2_inputs, False),
}


def _run_fwd_bwd(make_fn, has_beta, B, H, T, D, DV, C):
    """Run forward + backward to exercise all kernels for a config."""
    dtype = torch.bfloat16
    torch.manual_seed(0)

    if has_beta:
        q, k, v, g, beta, scale = make_fn(
            B, H, T, D, DV, dtype=dtype, device=DEVICE, requires_grad=True
        )
        beta = beta.detach().requires_grad_(True)
    else:
        q, k, v, g, scale = make_fn(
            B, H, T, D, DV, dtype=dtype, device=DEVICE, requires_grad=True
        )
        beta = None

    g = g.detach().requires_grad_(True)
    grad_out = torch.randn(B, H, T, DV, device=DEVICE, dtype=dtype)

    # Forward
    if beta is not None:
        o = chunked_linear_attn(q * scale, k, v, g, beta=beta, C=C)
    else:
        o = chunked_linear_attn(q * scale, k, v, g, C=C)

    # Backward (exercises all backward kernels)
    o.backward(grad_out)

    return o


def bench_variant(name, make_fn, has_beta, configs, chunk_sizes):
    """Benchmark a variant: fwd, bwd, fwd+bwd."""
    dtype = torch.bfloat16

    for B, H, T, D, DV in configs:
        for C in chunk_sizes:
            if T % C != 0:
                continue

            config_str = f"B={B},H={H},T={T},D={D},C={C}"

            try:
                # Warmup (compiles all kernels for this shape)
                _run_fwd_bwd(make_fn, has_beta, B, H, T, D, DV, C)
                torch.cuda.synchronize()

                # Forward-only benchmark
                torch.manual_seed(0)
                if has_beta:
                    q, k, v, g, beta, scale = make_fn(
                        B, H, T, D, DV, dtype=dtype, device=DEVICE
                    )
                    fwd_ms = do_bench(
                        lambda q=q, k=k, v=v, g=g, beta=beta, scale=scale: (
                            chunked_linear_attn(q * scale, k, v, g, beta=beta, C=C)
                        )
                    )
                else:
                    q, k, v, g, scale = make_fn(
                        B, H, T, D, DV, dtype=dtype, device=DEVICE
                    )
                    fwd_ms = do_bench(
                        lambda q=q, k=k, v=v, g=g, scale=scale: chunked_linear_attn(
                            q * scale, k, v, g, C=C
                        )
                    )

                # Fwd+bwd benchmark
                if has_beta:
                    q_b = q.clone().requires_grad_(True)
                    k_b = k.clone().requires_grad_(True)
                    v_b = v.clone().requires_grad_(True)
                    go = torch.randn(B, H, T, DV, device=DEVICE, dtype=dtype)

                    def fwd_bwd(q_b=q_b, k_b=k_b, v_b=v_b, g=g, beta=beta, go=go):
                        o = chunked_linear_attn(
                            q_b * scale, k_b, v_b, g, beta=beta, C=C
                        )
                        o.backward(go)
                        q_b.grad = k_b.grad = v_b.grad = None
                else:
                    q_b = q.clone().requires_grad_(True)
                    k_b = k.clone().requires_grad_(True)
                    v_b = v.clone().requires_grad_(True)
                    go = torch.randn(B, H, T, DV, device=DEVICE, dtype=dtype)

                    def fwd_bwd(q_b=q_b, k_b=k_b, v_b=v_b, g=g, go=go):
                        o = chunked_linear_attn(q_b * scale, k_b, v_b, g, C=C)
                        o.backward(go)
                        q_b.grad = k_b.grad = v_b.grad = None

                bwd_ms = do_bench(fwd_bwd)

                print(
                    f"  {name:<16} {config_str:<36} "
                    f"fwd={fwd_ms:>7.3f}ms  fwd+bwd={bwd_ms:>7.3f}ms"
                )
            except Exception as e:
                print(f"  {name:<16} {config_str:<36} ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="AOT Benchmark for Linear Attention")
    parser.add_argument(
        "--shard",
        type=int,
        default=None,
        help="Shard index for multi-GPU parallel (0..num_shards-1)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=8,
        help="Total number of shards (default: 8)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Run only a specific variant",
    )
    args = parser.parse_args()

    aot_mode = os.environ.get("HELION_AOT_MODE", "disabled")
    print(f"AOT Mode: {aot_mode}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    variants = VARIANTS
    if args.variant:
        variants = {args.variant: VARIANTS[args.variant]}

    # Build full list of (variant, config, chunk_size) tuples
    work_items = []
    for vname in variants:
        for cfg in CONFIGS:
            for cs in CHUNK_SIZES:
                work_items.append((vname, cfg, cs))

    # Shard if requested
    if args.shard is not None:
        work_items = [
            w for i, w in enumerate(work_items) if i % args.num_shards == args.shard
        ]
        print(f"Shard {args.shard}/{args.num_shards}: {len(work_items)} work items")

    print(f"{'Variant':<16} {'Config':<36} {'Forward':>10}  {'Fwd+Bwd':>12}")
    print("-" * 82)

    for vname, (B, H, T, D, DV), C in work_items:
        make_fn, has_beta = variants[vname]
        if T % C != 0:
            continue
        bench_variant(vname, make_fn, has_beta, [(B, H, T, D, DV)], [C])

    print()
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
