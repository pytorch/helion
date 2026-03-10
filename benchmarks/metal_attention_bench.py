"""Benchmark: Helion fused attention vs PyTorch SDPA on MPS.

Single-dispatch fused attention using MPP matmul2d + SIMD-shuffle softmax.
Materializes the full [M, N] attention matrix in a scratch buffer:
    softmax(Q @ K^T / sqrt(d)) @ V

Usage (on a Mac with MPS):
    python benchmarks/metal_attention_bench.py
"""

from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn.functional as F


def bench(
    fn: Callable[..., object],
    *args: torch.Tensor,
    warmup: int = 25,
    rep: int = 100,
) -> float:
    """Wall-clock benchmark with MPS synchronization. Returns median ms."""
    for _ in range(warmup):
        fn(*args)
    torch.mps.synchronize()

    times = []
    for _ in range(rep):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        fn(*args)
        torch.mps.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)

    times.sort()
    return times[len(times) // 2]


def _build_fused_attention_msl(M: int, K: int, N: int, D: int, NUM_SG: int = 4) -> str:
    """Build batched fused attention MSL: single dispatch for all B*H heads.

    Grid: (tpg, grid_m, num_heads). Each threadgroup processes one
    (head, tile_m) combination via:
      1. Multi-SG matmul: Q_tile @ K^T -> scratch  (scores)
      2. Threadgroup-cooperative softmax on scratch  (SIMD shuffles + shared mem)
      3. Multi-SG matmul: scratch @ V -> output     (weighted sum)
    """
    return f"""
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
constant int _M = {M};
constant int _N = {N};
constant int _K = {K};
constant int _OUT_D = {D};
constant int _TILE_M = 32;
constant int _NUM_SG = {NUM_SG};
constant int _HEAD_STRIDE_QVO = {M * D};
constant int _HEAD_STRIDE_KT = {K * N};
constant int _SCRATCH_STRIDE = {M * N};

kernel void fused_attention(
    device float* q [[buffer(0)]],
    device float* kt [[buffer(1)]],
    device float* v [[buffer(2)]],
    device float* out_buf [[buffer(3)]],
    device float* scratch [[buffer(4)]],
    uint3 tgid [[thread_position_in_grid]],
    uint _tpos [[thread_index_in_threadgroup]]) {{

    uint _head = tgid.z;
    device float* _hq = q + _head * _HEAD_STRIDE_QVO;
    device float* _hkt = kt + _head * _HEAD_STRIDE_KT;
    device float* _hv = v + _head * _HEAD_STRIDE_QVO;
    device float* _hout = out_buf + _head * _HEAD_STRIDE_QVO;
    device float* _hs = scratch + _head * _SCRATCH_STRIDE;

    auto _t_q = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        _hq, dextents<int32_t, 2>({K}, {M}));
    auto _t_kt = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        _hkt, dextents<int32_t, 2>({N}, {K}));
    auto _t_v = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        _hv, dextents<int32_t, 2>({D}, {N}));
    auto _t_out = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        _hout, dextents<int32_t, 2>({D}, {M}));
    auto _t_s = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        _hs, dextents<int32_t, 2>({N}, {M}));

    // --- Stage 1: Scores = Q_tile @ K^T (multi-SG matmul) ---
    constexpr auto _mm1 = matmul2d_descriptor(
        _TILE_M, _N, dynamic_length_v<int>,
        false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<_mm1, execution_simdgroups<_NUM_SG>> _mm1Op;
    auto _As = _t_q.slice(0, tgid.y * _TILE_M);
    auto _Bs = _t_kt.slice(0, 0);
    auto _Ss = _t_s.slice(0, tgid.y * _TILE_M);
    _mm1Op.run(_As, _Bs, _Ss);
    threadgroup_barrier(mem_flags::mem_device);

    // --- Stage 2: Softmax (threadgroup-cooperative SIMD + shared mem) ---
    threadgroup float _shared[32];
    uint _lane = _tpos % 32;
    uint _sg = _tpos / 32;
    uint _nsg = _NUM_SG;
    uint _tsz = _nsg * 32;
    float _scale = rsqrt((float)_K);

    for (int _r = 0; _r < _TILE_M; _r++) {{
        int _rb = (tgid.y * _TILE_M + _r) * _N;

        // Scale + row max
        float _lm = -INFINITY;
        for (int _c = (int)_tpos; _c < _N; _c += (int)_tsz) {{
            float _v = (float)_hs[_rb + _c] * _scale;
            _hs[_rb + _c] = _v;
            _lm = max(_lm, _v);
        }}
        for (uint _o = 16; _o > 0; _o >>= 1) _lm = max(_lm, simd_shuffle_down(_lm, _o));
        if (_lane == 0) _shared[_sg] = _lm;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (_sg == 0) {{
            float _v = (_lane < _nsg) ? _shared[_lane] : -INFINITY;
            for (uint _o = 16; _o > 0; _o >>= 1) _v = max(_v, simd_shuffle_down(_v, _o));
            _shared[0] = _v;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _mx = _shared[0];

        // Exp + row sum
        float _ls = 0.0f;
        for (int _c = (int)_tpos; _c < _N; _c += (int)_tsz) {{
            float _v = exp((float)_hs[_rb + _c] - _mx);
            _hs[_rb + _c] = _v;
            _ls += _v;
        }}
        for (uint _o = 16; _o > 0; _o >>= 1) _ls += simd_shuffle_down(_ls, _o);
        if (_lane == 0) _shared[_sg] = _ls;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (_sg == 0) {{
            float _v = (_lane < _nsg) ? _shared[_lane] : 0.0f;
            for (uint _o = 16; _o > 0; _o >>= 1) _v += simd_shuffle_down(_v, _o);
            _shared[0] = _v;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float _sm = _shared[0];

        // Normalize
        for (int _c = (int)_tpos; _c < _N; _c += (int)_tsz) _hs[_rb + _c] /= _sm;
        threadgroup_barrier(mem_flags::mem_device);
    }}

    // --- Stage 3: Output = Weights @ V (multi-SG matmul) ---
    constexpr auto _mm2 = matmul2d_descriptor(
        _TILE_M, _OUT_D, dynamic_length_v<int>,
        false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<_mm2, execution_simdgroups<_NUM_SG>> _mm2Op;
    auto _Vs = _t_v.slice(0, 0);
    auto _Os = _t_out.slice(0, tgid.y * _TILE_M);
    _mm2Op.run(_Ss, _Vs, _Os);
}}
"""


_shader_cache: dict[tuple[int, ...], object] = {}


def helion_fused_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Single-dispatch batched fused attention. Q/K/V: [B, H, M, D]."""
    batch, heads, seq_len, head_dim = q.shape
    num_heads = batch * heads
    NUM_SG = 4
    TILE_M = 32
    grid_m = (seq_len + TILE_M - 1) // TILE_M
    tpg = 32 * NUM_SG

    q3 = q.reshape(num_heads, seq_len, head_dim).contiguous()
    k3 = k.reshape(num_heads, seq_len, head_dim).contiguous()
    v3 = v.reshape(num_heads, seq_len, head_dim).contiguous()
    kt3 = k3.transpose(1, 2).contiguous()

    cache_key = (seq_len, head_dim, NUM_SG)
    if cache_key not in _shader_cache:
        msl = _build_fused_attention_msl(seq_len, head_dim, seq_len, head_dim, NUM_SG)
        _shader_cache[cache_key] = torch.mps.compile_shader(msl)

    lib = _shader_cache[cache_key]
    out = torch.empty_like(q3)
    scratch = torch.empty(num_heads * seq_len * seq_len, device=q.device, dtype=q.dtype)

    lib.fused_attention(
        q3,
        kt3,
        v3,
        out,
        scratch,
        threads=[tpg, grid_m, num_heads],
        group_size=[tpg, 1, 1],
    )
    return out.view(batch, heads, seq_len, head_dim)


def eager_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v)


def main() -> None:
    device = torch.device("mps")
    batch, heads, head_dim = 2, 8, 64
    seq_lens = [64, 128, 256, 512]

    print(f"{'Config':<26} {'Eager SDPA':>12} {'Helion Fused':>13} {'Speedup':>9}")
    print("-" * 66)

    for seq_len in seq_lens:
        q = torch.randn(batch, heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device)

        # Correctness check
        expected = F.scaled_dot_product_attention(q, k, v)
        result = helion_fused_attention(q, k, v)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

        t_eager = bench(eager_sdpa, q, k, v)
        t_helion = bench(helion_fused_attention, q, k, v)
        speedup = t_eager / t_helion

        config = f"B={batch} H={heads} S={seq_len} D={head_dim}"
        print(f"{config:<26} {t_eager:>10.3f}ms {t_helion:>11.3f}ms {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
