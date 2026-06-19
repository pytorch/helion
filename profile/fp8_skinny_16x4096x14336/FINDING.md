# fp8_gemm (16, 4096, 14336) — skinny-M underperformance diagnosis

## Result (cold-L2, tritonbench do_bench_wrapper cudagraph, B200)

| impl | latency | vs torch |
|---|---|---|
| torch._scaled_mm (cuBLAS) | 26.5 µs | 1.00× |
| helion fp8_gemm_skinny_m (autotuned / any config) | ~542 µs | **0.049× (≈20× slower)** |

Numerics correct (relerr 0). This is a severe, real underperformance — not a
measurement artifact. (An autotune-internal timing reported 1197µs; the stable
cudagraph p50 is ~542µs and is flat across every skinny config tried.)

## Root cause: M=16 falls off the tcgen05 MMA path → scalar SIMT GEMV

The problem is memory-bound: A=[16,14336] (0.23MB) + B=[14336,4096] weights
(58.7MB) → arithmetic intensity ~32 FLOP/byte. cuBLAS streams the 58.7MB
weights with a GEMV/small-M kernel at a few TB/s (26µs).

helion cannot use tensor cores here: **matmul_ops.py gates tcgen05 on
`static_m >= 64`** (line ~331). M=16 < 64, so BOTH `fp8_gemm` and
`fp8_gemm_skinny_m` fall to a **pure scalar SIMT loop**. The generated skinny
kernel is, per inner K iteration:

    load   = cute.arch.load(x, Uint8)          # one fp8 byte, scalar
    load_1 = cute.arch.load(y, Uint8)          # one fp8 byte, scalar
    dot_acc += fp8_to_f32(load) * fp8_to_f32(load_1)   # scalar FMA

NCU confirms the pathology: **L1TEX 78% (saturated) while DRAM 1.4% / L2 1.6%
(idle)** — the kernel re-reads through L1 element-by-element instead of
streaming the weights from DRAM. ~0.22 waves, 32 regs/thread.

Every config converges to ~542µs: block_n ∈ {8..32}, block_k ∈ {128..512},
cute_vector_widths up to [4,8] (vk capped at 8) — the scalar per-element
fp8 convert+FMA over K=14336 is the hard floor; no tiling/vec knob escapes it.

## Fixes attempted

1. **split-K** (3D tile [m,n,k] + atomic_add, split_k 8–64): correct but
   ~2100–2500µs — *worse*. Still scalar (M=16 < 64), and atomic_add overhead
   on top. Split-K parallelizes the read but each CTA is still scalar.

2. **Pad M=16 → 128 and force the tcgen05 path** (the principled fix — the
   58.7MB weight read is identical at M=64/128, but the MMA path engages and
   uses TMA/SMEM streaming + tensor cores). This is the right idea and would
   close most of the gap, BUT: the padded-M tcgen05 fp8 kernel **crashes at
   runtime (hard CUDA fault, no Python exception) specifically for this large-K
   / large-N shape** (K=14336, N=4096). The same M=128 tcgen05 config runs fine
   on K=4096/N=2048. So there is a separate tcgen05 codegen/runtime bug at large
   K that blocks the proper fix — beyond config tuning.

## Conclusion / where the fix lives

The 20× gap is a **kernel-path limitation, not a tuning problem**: skinny M is
served by a scalar SIMT GEMV because tcgen05 is M>=64-gated. Closing it requires
either:
- (a) a tcgen05 small-M path that pads M to the MMA tile (mask the extra rows)
  and emits TMA/SMEM-staged weight loads — once the large-K tcgen05 runtime
  crash is fixed; or
- (b) a proper vectorized/SMEM-staged SIMT GEMV codegen for the skinny kernel
  (wide cp.async loads of the weight tile into SMEM, then a register GEMV),
  instead of the current scalar one-byte-at-a-time loop.

Both are codegen-authoring changes in the CuTe backend, not expressible via a
helion.Config, so they could not be landed as a config/seed tweak like the
small-grid work. This commit records the diagnosis + a reproduction harness
(profile/fp8_skinny_16x4096x14336/).
