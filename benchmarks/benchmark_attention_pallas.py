"""
Benchmark: Standard Attention via JAX and Pallas Kernels
========================================================

V0: Compiled JAX (XLA softmax -- optimal when seq fits in VMEM)
V1: Naive tiled FlashAttention (fori_loop + pl.ds ref slicing)
V2: memory_space=ANY + emit_pipeline with ARBITRARY semantics
V3: Explicit pltpu async DMA double-buffer

Each Pallas kernel autotuned over 16 configs.
"""

from __future__ import annotations

import functools
import itertools
import math
import time
from typing import Any
from typing import NamedTuple

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

BATCH = 2
HEADS = 4
SEQ_LEN = 1024
HEAD_DIM = 64
DTYPE = jnp.float16
NUM_WARMUP = 3
NUM_ITERS = 10


class AutotuneConfig(NamedTuple):
    block_q: int
    block_kv: int


def generate_configs(max_configs: int = 16) -> list[AutotuneConfig]:
    configs = []
    for bq, bkv in itertools.product([32, 64, 128, 256], repeat=2):
        if bq <= SEQ_LEN and bkv <= SEQ_LEN:
            configs.append(AutotuneConfig(block_q=bq, block_kv=bkv))
        if len(configs) >= max_configs:
            break
    return configs


DEFAULT_CONFIGS = generate_configs()


def make_inputs(key: jax.Array):
    keys = jax.random.split(key, 3)
    shape = (BATCH, HEADS, SEQ_LEN, HEAD_DIM)
    return tuple(jax.random.normal(k, shape, dtype=DTYPE) for k in keys)


def benchmark(fn, q, k, v, label: str):
    for _ in range(NUM_WARMUP):
        fn(q, k, v).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(NUM_ITERS):
        fn(q, k, v).block_until_ready()
    elapsed = (time.perf_counter() - t0) / NUM_ITERS
    print(f"  {label:50s}  {elapsed * 1e3:8.3f} ms")


def autotune_and_bench(factory, q, k, v, label: str):
    best_time, best_cfg, best_fn = float("inf"), None, None
    for cfg in DEFAULT_CONFIGS:
        try:
            fn = factory(cfg)
            fn(q, k, v).block_until_ready()
            t0 = time.perf_counter()
            for _ in range(3):
                fn(q, k, v).block_until_ready()
            t = (time.perf_counter() - t0) / 3
            if t < best_time:
                best_time, best_cfg, best_fn = t, cfg, fn
        except Exception:
            continue
    if best_fn is None:
        print(f"  {label:50s}  FAILED")
        return
    print(f"  [best: block_q={best_cfg.block_q}, block_kv={best_cfg.block_kv}]")
    benchmark(best_fn, q, k, v, label)


# ===========================================================================
# V0: Compiled JAX (XLA path)
# ===========================================================================


@functools.partial(jax.jit, donate_argnums=())
def attention_jax_compiled(q, k, v):
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    attn = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * sm_scale
    attn = jax.nn.softmax(attn.astype(jnp.float32), axis=-1).astype(q.dtype)
    return jnp.matmul(attn, v)


# ===========================================================================
# V1: Naive tiled FlashAttention (fori_loop + pl.ds ref slicing)
#
# Flat 2D layout: reshape [B,H,S,D] -> [B*H*S, D]. Grid (bh, num_q_blocks).
# K/V get full-sequence blocks; kernel iterates KV blocks via pl.ds slicing.
# No singleton leading dims (avoids TPU tiling issues).
# ===========================================================================


def make_kernel_v1_naive(cfg: AutotuneConfig):
    block_q, block_kv = cfg
    num_kv_blocks = SEQ_LEN // block_kv
    num_q_blocks = SEQ_LEN // block_q
    qk_scale = (1.0 / math.sqrt(HEAD_DIM)) * 1.44269504

    def kernel(q_ref, k_ref, v_ref, o_ref):
        # q_ref: [block_q, D], k/v_ref: [SEQ_LEN, D], o_ref: [block_q, D]
        q = q_ref[...].astype(jnp.float32)
        m_i = jnp.full((block_q,), -1e30, dtype=jnp.float32)
        l_i = jnp.ones((block_q,), dtype=jnp.float32)
        acc = jnp.zeros((block_q, HEAD_DIM), dtype=jnp.float32)

        def body(j, carry):
            m_i, l_i, acc = carry
            # pl.ds ref slicing (not lax.dynamic_slice) for TPU compat
            k_blk = k_ref[pl.ds(j * block_kv, block_kv), :].astype(jnp.float32)
            qk = jnp.dot(q, k_blk.T)
            m_ij = jnp.maximum(m_i, jnp.max(qk, axis=-1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = jnp.exp2(qk)
            l_ij = jnp.sum(p, axis=-1)
            alpha = jnp.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            v_blk = v_ref[pl.ds(j * block_kv, block_kv), :].astype(jnp.float32)
            acc = acc + jnp.dot(p.astype(jnp.float16), v_blk.astype(jnp.float32))
            return m_ij, l_i, acc

        m_i, l_i, acc = lax.fori_loop(0, num_kv_blocks, body, (m_i, l_i, acc))
        o_ref[...] = (acc / l_i[:, None]).astype(o_ref.dtype)

    @jax.jit
    def run(q, k, v):
        bh = BATCH * HEADS
        # Flatten to 2D: [bh * SEQ_LEN, HEAD_DIM]
        q_ = q.reshape(bh * SEQ_LEN, HEAD_DIM)
        k_ = k.reshape(bh * SEQ_LEN, HEAD_DIM)
        v_ = v.reshape(bh * SEQ_LEN, HEAD_DIM)
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((bh * SEQ_LEN, HEAD_DIM), q.dtype),
            grid=(bh, num_q_blocks),
            in_specs=[
                # Q: block_q rows per grid cell
                pl.BlockSpec(
                    (block_q, HEAD_DIM),
                    lambda i, j: (i * num_q_blocks + j, 0),
                ),
                # K: full sequence per batch-head
                pl.BlockSpec(
                    (SEQ_LEN, HEAD_DIM),
                    lambda i, j: (i, 0),
                ),
                # V: full sequence per batch-head
                pl.BlockSpec(
                    (SEQ_LEN, HEAD_DIM),
                    lambda i, j: (i, 0),
                ),
            ],
            out_specs=pl.BlockSpec(
                (block_q, HEAD_DIM),
                lambda i, j: (i * num_q_blocks + j, 0),
            ),
        )(q_, k_, v_).reshape(BATCH, HEADS, SEQ_LEN, HEAD_DIM)

    return run


# ===========================================================================
# V2: memory_space=ANY + emit_pipeline with ARBITRARY semantics
#
# Per-head via vmap. K/V in memory_space=ANY (block=full, trivial index_map)
# so the compiler chooses HBM vs VMEM placement. Inner KV loop via
# emit_pipeline with ARBITRARY semantics (sequential for online softmax).
# Pipeline auto-synthesizes double-buffering for HBM→VMEM transfers.
# State (m_i, l_i, acc) lives in VMEM scratch refs -- the pipeline body
# reads/writes scratch directly (no captured traced values).
# ===========================================================================


def make_kernel_v2_emit_pipeline(cfg: AutotuneConfig):
    block_q, block_kv = cfg
    num_kv_blocks = SEQ_LEN // block_kv
    num_q_blocks = SEQ_LEN // block_q
    qk_scale = (1.0 / math.sqrt(HEAD_DIM)) * 1.44269504

    def kernel(q_ref, k_ref, v_ref, o_ref, m_scr, l_scr, acc_scr):
        # q_ref: [block_q, D], k/v_ref: [SEQ_LEN, D] (ANY)
        # m_scr: [block_q] VMEM, l_scr: [block_q] VMEM, acc_scr: [block_q, D] VMEM
        m_scr[...] = jnp.full((block_q,), -1e30, dtype=jnp.float32)
        l_scr[...] = jnp.ones((block_q,), dtype=jnp.float32)
        acc_scr[...] = jnp.zeros((block_q, HEAD_DIM), dtype=jnp.float32)

        def pipeline_body(q_tile_ref, k_tile_ref, v_tile_ref):
            # q_tile_ref: [block_q, D] (same block every iter)
            # k/v_tile_ref: [block_kv, D] (changes each iter)
            q = q_tile_ref[...].astype(jnp.float32) * qk_scale
            m_i = m_scr[...]
            l_i = l_scr[...]
            acc = acc_scr[...]

            k_blk = k_tile_ref[...].astype(jnp.float32)
            qk = jnp.dot(q, k_blk.T)
            m_ij = jnp.maximum(m_i, jnp.max(qk, axis=-1))
            p = jnp.exp2(qk - m_ij[:, None])
            l_ij = jnp.sum(p, axis=-1)
            alpha = jnp.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            v_blk = v_tile_ref[...].astype(jnp.float32)
            acc = acc + jnp.dot(p.astype(jnp.float16), v_blk.astype(jnp.float32))

            m_scr[...] = m_ij
            l_scr[...] = l_i
            acc_scr[...] = acc

        # Pass q_ref as a pipeline input that repeats every iteration
        # (block=full, trivial index). K/V are tiled along the seq dim.
        pltpu.emit_pipeline(
            pipeline_body,
            grid=(num_kv_blocks,),
            in_specs=[
                pl.BlockSpec((block_q, HEAD_DIM), lambda j: (0, 0)),  # q: same each iter
                pl.BlockSpec((block_kv, HEAD_DIM), lambda j: (j, 0)),  # k: tiled
                pl.BlockSpec((block_kv, HEAD_DIM), lambda j: (j, 0)),  # v: tiled
            ],
            out_specs=[],
            dimension_semantics=(pltpu.GridDimensionSemantics.ARBITRARY,),
        )(q_ref, k_ref, v_ref)

        acc = acc_scr[...]
        l_i = l_scr[...]
        o_ref[...] = (acc / l_i[:, None]).astype(o_ref.dtype)

    def single_head(q_seq, k_seq, v_seq):
        # q/k/v_seq: [SEQ_LEN, HEAD_DIM] for one batch-head
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((SEQ_LEN, HEAD_DIM), q_seq.dtype),
            grid=(num_q_blocks,),
            in_specs=[
                pl.BlockSpec((block_q, HEAD_DIM), lambda j: (j, 0)),
                pl.BlockSpec(
                    (SEQ_LEN, HEAD_DIM),
                    lambda j: (0, 0),
                    memory_space=pl.ANY,
                ),
                pl.BlockSpec(
                    (SEQ_LEN, HEAD_DIM),
                    lambda j: (0, 0),
                    memory_space=pl.ANY,
                ),
            ],
            out_specs=pl.BlockSpec((block_q, HEAD_DIM), lambda j: (j, 0)),
            scratch_shapes=[
                pltpu.VMEM((block_q,), dtype=jnp.float32),       # m_i
                pltpu.VMEM((block_q,), dtype=jnp.float32),       # l_i
                pltpu.VMEM((block_q, HEAD_DIM), dtype=jnp.float32),  # acc
            ],
        )(q_seq, k_seq, v_seq)

    @jax.jit
    def run(q, k, v):
        # vmap over [B, H] dims; each call handles [SEQ_LEN, HEAD_DIM]
        return jax.vmap(jax.vmap(single_head))(q, k, v)

    return run


# ===========================================================================
# V3: Explicit pltpu async DMA double-buffer
#
# Per-head via vmap. K/V in memory_space=ANY (full array, trivial index).
# Manual double-buffering: 2-slot VMEM scratch for K+V, DMA semaphores.
# Prologue starts first copy; each fori_loop iteration prefetches next
# block into the alternate slot while computing on the current slot.
# ===========================================================================


def make_kernel_v3_async_dma(cfg: AutotuneConfig):
    block_q, block_kv = cfg
    num_kv_blocks = SEQ_LEN // block_kv
    num_q_blocks = SEQ_LEN // block_q
    qk_scale = (1.0 / math.sqrt(HEAD_DIM)) * 1.44269504

    def kernel(q_ref, k_ref, v_ref, o_ref, k_buf, v_buf, k_sem, v_sem):
        # q_ref: [block_q, D], k/v_ref: [SEQ_LEN, D] (ANY/HBM)
        # k_buf/v_buf: [2, block_kv, D] VMEM double-buffer
        # k_sem/v_sem: [2] DMA semaphores
        q = q_ref[...].astype(jnp.float32) * qk_scale
        m_i = jnp.full((block_q,), -1e30, dtype=jnp.float32)
        l_i = jnp.ones((block_q,), dtype=jnp.float32)
        acc = jnp.zeros((block_q, HEAD_DIM), dtype=jnp.float32)

        # Prologue: start first KV fetch into slot 0
        pltpu.make_async_copy(
            k_ref.at[pl.ds(0, block_kv), :], k_buf.at[0], k_sem.at[0]
        ).start()
        pltpu.make_async_copy(
            v_ref.at[pl.ds(0, block_kv), :], v_buf.at[0], v_sem.at[0]
        ).start()

        def body(j, carry):
            m_i, l_i, acc = carry
            slot = j % 2
            next_slot = (j + 1) % 2
            next_j = j + 1

            # Prefetch next block into alternate slot
            def do_prefetch():
                pltpu.make_async_copy(
                    k_ref.at[pl.ds(next_j * block_kv, block_kv), :],
                    k_buf.at[next_slot],
                    k_sem.at[next_slot],
                ).start()
                pltpu.make_async_copy(
                    v_ref.at[pl.ds(next_j * block_kv, block_kv), :],
                    v_buf.at[next_slot],
                    v_sem.at[next_slot],
                ).start()

            lax.cond(next_j < num_kv_blocks, do_prefetch, lambda: None)

            # Wait for current slot to finish loading
            pltpu.make_async_copy(
                k_ref.at[pl.ds(j * block_kv, block_kv), :],
                k_buf.at[slot],
                k_sem.at[slot],
            ).wait()
            pltpu.make_async_copy(
                v_ref.at[pl.ds(j * block_kv, block_kv), :],
                v_buf.at[slot],
                v_sem.at[slot],
            ).wait()

            # Compute online softmax on current VMEM block
            k_blk = k_buf[slot].astype(jnp.float32)
            qk = jnp.dot(q, k_blk.T)
            m_ij = jnp.maximum(m_i, jnp.max(qk, axis=-1))
            p = jnp.exp2(qk - m_ij[:, None])
            l_ij = jnp.sum(p, axis=-1)
            alpha = jnp.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            v_blk = v_buf[slot].astype(jnp.float32)
            acc = acc + jnp.dot(p.astype(jnp.float16), v_blk.astype(jnp.float32))
            return m_ij, l_i, acc

        m_i, l_i, acc = lax.fori_loop(0, num_kv_blocks, body, (m_i, l_i, acc))
        o_ref[...] = (acc / l_i[:, None]).astype(o_ref.dtype)

    def single_head(q_seq, k_seq, v_seq):
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((SEQ_LEN, HEAD_DIM), q_seq.dtype),
            grid=(num_q_blocks,),
            in_specs=[
                pl.BlockSpec((block_q, HEAD_DIM), lambda j: (j, 0)),
                pl.BlockSpec(
                    (SEQ_LEN, HEAD_DIM),
                    lambda j: (0, 0),
                    memory_space=pl.ANY,
                ),
                pl.BlockSpec(
                    (SEQ_LEN, HEAD_DIM),
                    lambda j: (0, 0),
                    memory_space=pl.ANY,
                ),
            ],
            out_specs=pl.BlockSpec((block_q, HEAD_DIM), lambda j: (j, 0)),
            scratch_shapes=[
                pltpu.VMEM((2, block_kv, HEAD_DIM), dtype=DTYPE),  # k_buf
                pltpu.VMEM((2, block_kv, HEAD_DIM), dtype=DTYPE),  # v_buf
                pltpu.SemaphoreType.DMA((2,)),                     # k_sem
                pltpu.SemaphoreType.DMA((2,)),                     # v_sem
            ],
        )(q_seq, k_seq, v_seq)

    @jax.jit
    def run(q, k, v):
        return jax.vmap(jax.vmap(single_head))(q, k, v)

    return run


# ===========================================================================
# Main
# ===========================================================================


def check_correctness(fn, q, k, v, ref, label: str, atol: float = 1e-2):
    try:
        out = fn(q, k, v)
        err = jnp.max(jnp.abs(out - ref)).item()
        print(
            f"  {label:50s}  max_err={err:.6f}"
            f"  [{'PASS' if err < atol else 'FAIL'}]"
        )
    except Exception as e:
        print(f"  {label:50s}  ERROR: {e}")


def main():
    print(f"B={BATCH} H={HEADS} SEQ={SEQ_LEN} D={HEAD_DIM} dtype={DTYPE}")
    print(f"{len(DEFAULT_CONFIGS)} autotuning configs per kernel\n")

    q, k, v = make_inputs(jax.random.PRNGKey(42))
    ref = attention_jax_compiled(q, k, v)

    variants: list[tuple[str, Any]] = [
        ("V1: naive tiled FlashAttn", make_kernel_v1_naive),
        ("V2: ANY + emit_pipeline ARBITRARY", make_kernel_v2_emit_pipeline),
        ("V3: async DMA double-buffer", make_kernel_v3_async_dma),
    ]

    print("=== Correctness ===")
    check_correctness(attention_jax_compiled, q, k, v, ref, "V0: compiled JAX (XLA)")
    for label, factory in variants:
        try:
            check_correctness(factory(DEFAULT_CONFIGS[0]), q, k, v, ref, label)
        except Exception as e:
            print(f"  {label:50s}  ERROR: {e}")

    print("\n=== Benchmarks ===")
    benchmark(attention_jax_compiled, q, k, v, "V0: compiled JAX (XLA)")
    for label, factory in variants:
        autotune_and_bench(factory, q, k, v, label)

    print("\nDone.")


if __name__ == "__main__":
    main()
