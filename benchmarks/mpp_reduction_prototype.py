"""Prototype: Standalone softmax via MPP cooperative_tensor reduce_rows.

Tests whether we can use MPP's cooperative_tensor load() + reduce_rows() +
map_iterator() for standalone reductions (no matmul), and benchmarks it
against PyTorch eager and the current Helion SIMD shuffle approach.

The key idea: create a matmul2d descriptor with K=1 as a "dummy" just to
get access to cooperative_tensor infrastructure. Load data via coop.load(),
reduce via reduce_rows(), normalize via begin()/end() + map_iterator().
"""

from __future__ import annotations

import time

import torch


def build_mpp_softmax_msl(M: int, N: int) -> str:
    """Build MSL for softmax using MPP cooperative_tensor.

    One threadgroup per row, one simdgroup per threadgroup.
    Uses execution_simdgroup scope (required for reduce_rows).
    """
    return f"""\
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant int _M = {M};
constant int _N = {N};

kernel void mpp_softmax(
    device float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint tgid [[threadgroup_position_in_grid]]
) {{
    uint _row = tgid;
    if (_row >= (uint)_M) return;

    auto _t_x = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        x, dextents<int32_t, 2>(_N, _M));
    auto _t_out = tensor<device float, dextents<int32_t, 2>, tensor_inline>(
        out, dextents<int32_t, 2>(_N, _M));

    // Degenerate matmul descriptor: 1 × N with K=1
    // We never call run() — just use cooperative_tensor infrastructure
    constexpr auto _desc = matmul2d_descriptor(
        1, _N, 1,
        false, false, false,
        matmul2d_descriptor::mode::multiply);
    matmul2d<_desc, execution_simdgroup> _op;

    auto _x_slice = _t_x.slice(0, _row);

    // Load row into cooperative_tensor
    auto _coop = _op.get_destination_cooperative_tensor<
        decltype(_x_slice), decltype(_x_slice), float>();
    _coop.load(_x_slice);

    // Row max
    auto _cMax = _op.get_row_reduction_destination_cooperative_tensor<
        decltype(_x_slice), decltype(_x_slice), float>();
    reduce_rows(_coop, _cMax, reduction_operation::max,
        metal::numeric_limits<float>::lowest());

    // exp(x - max)
    for (auto _it = _coop.begin(); _it != _coop.end(); _it++) {{
        auto _m = _cMax.map_iterator(_it);
        *_it = exp(*_it - *_m);
    }}

    // Row sum
    auto _cSum = _op.get_row_reduction_destination_cooperative_tensor<
        decltype(_x_slice), decltype(_x_slice), float>();
    reduce_rows(_coop, _cSum, reduction_operation::sum, 0.0f);

    // Normalize
    for (auto _it = _coop.begin(); _it != _coop.end(); _it++) {{
        auto _s = _cSum.map_iterator(_it);
        *_it *= (1.0f / *_s);
    }}

    auto _o_slice = _t_out.slice(0, _row);
    _coop.store(_o_slice);
}}
"""


def bench(fn, *args, warmup: int = 20, rep: int = 100) -> float:
    """Benchmark with per-call sync, return median ms."""
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
    # IQR filtering
    q1, q3 = times[len(times) // 4], times[3 * len(times) // 4]
    iqr = q3 - q1
    filtered = [t for t in times if q1 - 1.5 * iqr <= t <= q3 + 1.5 * iqr]
    if not filtered:
        filtered = times
    filtered.sort()
    return filtered[len(filtered) // 2]


def main() -> None:
    device = torch.device("mps")

    shapes = [(256, 1024), (1024, 1024), (1024, 4096), (4096, 2560)]

    print("MPP cooperative_tensor softmax prototype")
    print("=" * 80)
    print(
        f"{'Shape':<16} {'Eager':>10} {'MPP coop':>10} "
        f"{'Speedup':>10} {'Accuracy':>12}"
    )
    print("-" * 80)

    for M, N in shapes:
        x = torch.randn(M, N, device=device, dtype=torch.float32)
        expected = torch.nn.functional.softmax(x, dim=-1)

        # Build and compile MPP softmax
        msl = build_mpp_softmax_msl(M, N)

        try:
            lib = torch.mps.compile_shader(msl)
        except Exception as e:
            print(f"({M}, {N})       COMPILE FAIL: {e}")
            continue

        out = torch.empty_like(x)

        try:
            # 1D dispatch: one threadgroup (32 threads) per row
            lib.mpp_softmax(x, out, threads=M * 32, group_size=32)
            torch.mps.synchronize()
        except Exception as e:
            print(f"({M}, {N})       DISPATCH FAIL: {e}")
            continue

        # Check correctness
        max_diff = (out - expected).abs().max().item()
        if max_diff < 1e-5:
            status = "PASS"
        elif max_diff < 1e-3:
            status = f"~OK ({max_diff:.1e})"
        else:
            status = f"FAIL ({max_diff:.1e})"

        # Benchmark
        def eager_fn(_x=x):
            return torch.nn.functional.softmax(_x, dim=-1)

        def mpp_fn(_x=x, _out=out, _lib=lib, _m=M):
            _lib.mpp_softmax(_x, _out, threads=_m * 32, group_size=32)

        t_eager = bench(eager_fn)
        t_mpp = bench(mpp_fn)
        speedup = t_eager / t_mpp

        print(
            f"({M}, {N})"
            f"{t_eager:>14.4f}ms"
            f"{t_mpp:>10.4f}ms"
            f"{speedup:>9.2f}x"
            f"{status:>12}"
        )


if __name__ == "__main__":
    main()
