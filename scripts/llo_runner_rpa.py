"""Single-config runner for msl_tpu_kernel.kernels.ragged_paged_attention.

Designed to produce a clean LLO dump for one fixed config so the roofline
predictor can be calibrated against a production RPA kernel.

Usage (on the TPU pod, msl-tpu-kernel package on PYTHONPATH):
    rm -rf /tmp/llo_dump && mkdir -p /tmp/llo_dump
    LIBTPU_INIT_ARGS='--xla_jf_dump_to=/tmp/llo_dump' \\
        ALLOW_MULTIPLE_LIBTPU_LOAD=1 TPU_VISIBLE_CHIPS=<N> \\
        PYTHONPATH=/path/to/msl-tpu-kernel:$PYTHONPATH \\
        python scripts/llo_runner_rpa.py --mode decode \\
        --num-seqs 32 --kv-per-seq 2048 --num-q-heads 8 --num-kv-heads 1 --head-dim 128 --page-size 64

After the run, the named kernel dump will be
`/tmp/llo_dump/*RPA-bq_<bq>-bkvp_<bkvp>-p_<p>-*final_bundles.txt` plus a
matching `*-final_hlo-static-per-bundle-utilization.txt`. Copy the two files
into `llo/rpa_<mode>_*/<entry>/{final_bundles,utilization}.txt` then run:
    python scripts/tpu_roofline.py llo/<entry> --inputs '...' --outputs '...' \\
        --inner-loop-iters <K>     # RPA has dynamic trip count — supply K
                                   # ≈ runtime kv-block iterations

Two modes:
    - decode:  many short sequences (q=1 each), exercises the decode path
    - prefill: a few longer sequences, exercises the prefill path
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
from msl_tpu_kernel.kernels.ragged_paged_attention import get_kv_cache_shape
from msl_tpu_kernel.kernels.ragged_paged_attention import ragged_paged_attention
import numpy as np


def _bf16(
    shape: tuple[int, ...], rng: np.random.Generator, scale: float = 1.0
) -> object:
    arr = rng.random(shape, dtype=np.float32) * scale
    return jnp.array(arr.astype(jnp.bfloat16))


def make_decode_inputs(
    num_seqs: int,
    kv_per_seq: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
) -> tuple[object, ...]:
    rng = np.random.default_rng(0)
    q_len = num_seqs
    kv_len = num_seqs * kv_per_seq
    pages_per_seq = (kv_per_seq + page_size - 1) // page_size
    total_pages = num_seqs * pages_per_seq + 4

    queries = _bf16((max(q_len, kv_len), num_q_heads, head_dim), rng)
    keys = _bf16((kv_len, num_kv_heads, head_dim), rng)
    values = _bf16((kv_len, num_kv_heads, head_dim), rng)

    kv_cache_shape = get_kv_cache_shape(
        total_pages, page_size, num_kv_heads, head_dim, jnp.bfloat16
    )
    kv_cache = _bf16(kv_cache_shape, rng, scale=1e-3)

    kv_lens = jnp.full((num_seqs,), kv_per_seq, dtype=jnp.int32)
    page_indices = jnp.arange(num_seqs * pages_per_seq, dtype=jnp.int32)
    cu_q_lens = jnp.arange(num_seqs + 1, dtype=jnp.int32)
    # All decode: distribution = (num_seqs, num_seqs, num_seqs)
    distribution = jnp.array([num_seqs, num_seqs, num_seqs], dtype=jnp.int32)

    return (
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )


def make_prefill_inputs(
    num_seqs: int,
    q_per_seq: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
) -> tuple[object, ...]:
    rng = np.random.default_rng(0)
    q_len = num_seqs * q_per_seq
    kv_len = q_len  # prefill: kv == q
    pages_per_seq = (q_per_seq + page_size - 1) // page_size
    total_pages = num_seqs * pages_per_seq + 4

    queries = _bf16((max(q_len, kv_len), num_q_heads, head_dim), rng)
    keys = _bf16((kv_len, num_kv_heads, head_dim), rng)
    values = _bf16((kv_len, num_kv_heads, head_dim), rng)

    kv_cache_shape = get_kv_cache_shape(
        total_pages, page_size, num_kv_heads, head_dim, jnp.bfloat16
    )
    kv_cache = _bf16(kv_cache_shape, rng, scale=1e-3)

    kv_lens = jnp.full((num_seqs,), q_per_seq, dtype=jnp.int32)
    page_indices = jnp.arange(num_seqs * pages_per_seq, dtype=jnp.int32)
    cu_q_lens = jnp.arange(0, q_len + 1, q_per_seq, dtype=jnp.int32)
    # All prefill: distribution = (0, 0, num_seqs) (decodes=0, chunked-prefill=0)
    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)

    return (
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["decode", "prefill"], default="decode")
    p.add_argument("--num-seqs", type=int, default=32)
    p.add_argument("--kv-per-seq", type=int, default=2048)
    p.add_argument("--q-per-seq", type=int, default=512)
    p.add_argument("--num-q-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=1)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    print(f"Devices: {jax.devices()}")
    if args.mode == "decode":
        inputs = make_decode_inputs(
            args.num_seqs,
            args.kv_per_seq,
            args.num_q_heads,
            args.num_kv_heads,
            args.head_dim,
            args.page_size,
        )
        print(
            f"Decode: {args.num_seqs} seqs x {args.kv_per_seq} kv, "
            f"q={args.num_q_heads}h kv={args.num_kv_heads}h d={args.head_dim}"
        )
    else:
        inputs = make_prefill_inputs(
            args.num_seqs,
            args.q_per_seq,
            args.num_q_heads,
            args.num_kv_heads,
            args.head_dim,
            args.page_size,
        )
        print(
            f"Prefill: {args.num_seqs} seqs x {args.q_per_seq} q, "
            f"q={args.num_q_heads}h kv={args.num_kv_heads}h d={args.head_dim}"
        )

    fn = jax.jit(ragged_paged_attention)

    out, _ = fn(*inputs)
    for _ in range(max(0, args.warmup - 1)):
        out, _ = fn(*inputs)
    out.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(max(1, args.iters)):
        out, _ = fn(*inputs)
    out.block_until_ready()
    elapsed = (time.perf_counter() - t0) / args.iters

    print(f"Avg latency: {elapsed * 1e6:.2f} us")


if __name__ == "__main__":
    main()
