"""FP8 scaled matrix multiply (``c = scale_a * (a @ b) * scale_b + bias``).

Pretuned for the small-token-count (decode) FP8 GEMM shapes used by vLLM
serving.  The kernel and its checked-in heuristic are ported from
https://github.com/vllm-project/vllm/pull/46522.
"""

from __future__ import annotations

import math

import torch
import triton.testing as tt

import helion
import helion.experimental
import helion.language as hl


@helion.experimental.aot_kernel(
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def scaled_mm(
    c: torch.Tensor,  # [M, N]
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [1]/[1, 1]/[M]/[M, 1]
    scale_b: torch.Tensor,  # [1]/[1, 1]/[N]/[N, 1]
    bias: torch.Tensor | None = None,  # [N]
) -> None:
    M, K = a.shape
    N = b.shape[1]
    hl.specialize(K)
    hl.specialize(N)

    assert N > 0 and K > 0 and M > 0
    assert b.shape[0] == K
    assert a.dtype == b.dtype
    assert a.stride(1) == 1
    assert b.stride(0) == 1

    scale_a = scale_a.reshape(-1, 1) if scale_a.dim() <= 1 else scale_a
    scale_b = scale_b.reshape(-1, 1) if scale_b.dim() <= 1 else scale_b

    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()
    assert scale_a.shape[1] == 1 and (scale_a.shape[0] == 1 or scale_a.shape[0] == M)
    assert scale_b.shape[1] == 1 and (scale_b.shape[0] == 1 or scale_b.shape[0] == N)
    hl.specialize(scale_b.shape[1])

    out_dtype = c.dtype
    assert out_dtype.is_floating_point
    if bias is not None:
        assert bias.numel() == N and bias.dtype == out_dtype

    acc_dtype = torch.float32 if a.is_floating_point() else torch.int32

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], acc_dtype)
        for tile_k in hl.tile(K):
            acc = hl.dot(
                a[tile_m, tile_k],
                b[tile_k, tile_n],
                acc=acc,
                out_dtype=acc_dtype,
            )

        acc = acc.to(torch.float32)

        if scale_a.shape[0] == M:
            scale_a_blk = scale_a[tile_m, :]
        else:
            scale_a_blk = scale_a[0, 0].expand(tile_m.block_size, 1)
        acc = scale_a_blk * acc

        if scale_b.shape[0] == N:
            scale_b_blk = scale_b[tile_n, :]
        else:
            scale_b_blk = scale_b[0, 0].expand(tile_n.block_size, 1)
        acc = scale_b_blk.T * acc

        c_blk = acc.to(out_dtype)

        if bias is not None:
            c_blk += bias[tile_n]

        c[tile_m, tile_n] = c_blk


def _scaled_mm_torch(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> None:
    out = torch._scaled_mm(
        a,
        b,
        scale_a=scale_a,
        scale_b=scale_b,
        bias=bias,
        out_dtype=c.dtype,
    )
    c.copy_(out)


def main() -> None:
    # (K, N) weight shapes pulled from the vLLM Qwen3 FP8 sweep (TP=1).
    kn_shapes = [
        (2048, 4096),
        (2048, 2048),
        (2048, 12288),
        (6144, 2048),
        (4096, 6144),
        (4096, 4096),
        (4096, 24576),
        (12288, 4096),
        (5120, 10240),
        (5120, 5120),
        (5120, 51200),
        (25600, 5120),
    ]
    m_sizes = [16, 64]
    shapes = [(M, K, N) for M in m_sizes for (K, N) in kn_shapes]

    fp8_dtype = torch.float8_e4m3fn
    out_dtype = torch.bfloat16

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        f"{'M':>6s}  {'K':>6s}  {'N':>6s}  {'helion (us)':>12s}  "
        f"{'torch (us)':>12s}  {'speedup':>8s}"
    )
    print("-" * 63)

    speedups: list[float] = []
    helion_wins = 0
    best_speedup = 0.0
    best_shape = (0, 0, 0)
    for M, K, N in shapes:
        scale = 1.0 / math.sqrt(K)
        a = (scale * (0.5 + torch.rand(M, K, device="cuda"))).to(fp8_dtype)
        b = (scale * (0.5 + torch.rand(N, K, device="cuda"))).to(fp8_dtype).t()
        c = torch.empty((M, N), dtype=out_dtype, device="cuda")
        scale_a = torch.rand((1, 1), dtype=torch.float32, device="cuda") + 0.5
        scale_b = torch.rand((1, 1), dtype=torch.float32, device="cuda") + 0.5
        bias = 0.5 * (torch.rand(N, dtype=out_dtype, device="cuda") - 0.5)

        scaled_mm(c, a, b, scale_a, scale_b, bias)  # warmup
        ms_helion = tt.do_bench(
            lambda c=c, a=a, b=b, sa=scale_a, sb=scale_b, bias=bias: scaled_mm(
                c, a, b, sa, sb, bias
            ),
            warmup=25,
            rep=100,
            return_mode="median",
        )
        ms_torch = tt.do_bench(
            lambda c=c, a=a, b=b, sa=scale_a, sb=scale_b, bias=bias: _scaled_mm_torch(
                c, a, b, sa, sb, bias
            ),
            warmup=25,
            rep=100,
            return_mode="median",
        )
        speedup = ms_torch / ms_helion if ms_helion > 0 else float("nan")
        speedups.append(speedup)
        if speedup > 1.0:
            helion_wins += 1
        if speedup > best_speedup:
            best_speedup = speedup
            best_shape = (M, K, N)
        print(
            f"{M:>6d}  {K:>6d}  {N:>6d}  {ms_helion * 1000:>12.2f}  "
            f"{ms_torch * 1000:>12.2f}  {speedup:>7.2f}x"
        )

    geomean = math.exp(
        sum(math.log(s) for s in speedups if s > 0) / max(len(speedups), 1)
    )
    print(
        f"\nHelion faster on {helion_wins}/{len(shapes)} shapes; "
        f"geomean speedup {geomean:.3f}x; "
        f"best speedup {best_speedup:.2f}x at (M, K, N)={best_shape}."
    )
    print(
        f"SUMMARY: helion_wins={helion_wins} total={len(shapes)} "
        f"geomean={geomean:.4f} best_speedup={best_speedup:.4f}"
    )


if __name__ == "__main__":
    main()
