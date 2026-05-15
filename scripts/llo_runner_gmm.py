"""Single-config runner for msl_tpu_kernel.kernels.megablox.gmm_v2.

Designed to produce a clean LLO dump for one fixed config so the roofline
predictor can be calibrated against a production GMM kernel.

Usage (on the TPU pod, msl-tpu-kernel package on PYTHONPATH):
    rm -rf /tmp/llo_dump && mkdir -p /tmp/llo_dump
    LIBTPU_INIT_ARGS='--xla_jf_dump_to=/tmp/llo_dump' \\
        ALLOW_MULTIPLE_LIBTPU_LOAD=1 TPU_VISIBLE_CHIPS=<N> \\
        PYTHONPATH=/path/to/msl-tpu-kernel:$PYTHONPATH \\
        python scripts/llo_runner_gmm.py --M 1024 --K 4096 --N 4096 --G 8

After the run, the named kernel dump will be
`/tmp/llo_dump/*gmm_v2-g_<G>-m_<M>-k_<K>-n_<N>-*final_bundles.txt`
plus a matching `*-final_hlo-static-per-bundle-utilization.txt`. Copy the
two files into `llo/gmm_v2_M..K..N..G.._bf16/{final_bundles,utilization}.txt`
to add an entry to the LLO database, then run the predictor:
    python scripts/tpu_roofline.py llo/<entry> --inputs '...' --outputs '...'

The default config (M=1024, K=4096, N=4096, G=8, bf16) is representative of
prefill-time GMM in MoE workloads.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
from msl_tpu_kernel.kernels.megablox.gmm_v2 import gmm_v2


def make_inputs(
    M: int, K: int, N: int, G: int, seed: int = 0
) -> tuple[object, object, object]:
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    lhs = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    rhs = jax.random.normal(k2, (G, K, N), dtype=jnp.bfloat16)
    # Even distribution across groups.
    base = M // G
    rem = M - base * G
    sizes = [base + (1 if i < rem else 0) for i in range(G)]
    group_sizes = jnp.array(sizes, dtype=jnp.int32)
    return lhs, rhs, group_sizes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=1024)
    p.add_argument("--K", type=int, default=4096)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--G", type=int, default=8)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()

    print(f"Devices: {jax.devices()}")
    print(f"Config: M={args.M} K={args.K} N={args.N} G={args.G} bf16")

    lhs, rhs, group_sizes = make_inputs(args.M, args.K, args.N, args.G)

    fn = jax.jit(gmm_v2)
    # Warmup (triggers compilation + LLO dump).
    out = fn(lhs, rhs, group_sizes)
    for _ in range(max(0, args.warmup - 1)):
        out = fn(lhs, rhs, group_sizes)
    out.block_until_ready()

    # Timed.
    t0 = time.perf_counter()
    for _ in range(max(1, args.iters)):
        out = fn(lhs, rhs, group_sizes)
    out.block_until_ready()
    elapsed = (time.perf_counter() - t0) / args.iters

    flops = 2 * args.M * args.K * args.N
    tflops = flops / elapsed / 1e12
    print(f"Avg latency: {elapsed * 1e6:.2f} us")
    print(f"BF16 TFLOPS:  {tflops:.2f}")


if __name__ == "__main__":
    main()
