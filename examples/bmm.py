from __future__ import annotations

import torch

import helion
import helion.language as hl


# static_shapes=True gives a performance boost for matmuls
@helion.kernel(static_shapes=True)
def bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [B, M, K], B: [B, K, N], Out: [B, M, N]   # dense bmm
    b, m, k = A.size()
    b, k, n = B.size()
    out = torch.empty([b, m, n], device=A.device, dtype=torch.promote_types(A.dtype, B.dtype))
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(acc, A[tile_b, tile_m, tile_k], B[tile_b, tile_k, tile_n])
        out[tile_b, tile_m, tile_n] = acc
    return out


def check(b: int, m: int, k: int, n: int) -> None:
    from triton.testing import do_bench

    x = torch.randn([b, m, k], device="cuda", dtype=torch.float16)
    y = torch.randn([b, k, n], device="cuda", dtype=torch.float16)
    result = bmm(x, y)
    torch.testing.assert_close(result, x @ y, rtol=1e-2, atol=1e-1)
    sec = do_bench(lambda: bmm(x, y))
    baseline_sec = do_bench(lambda: torch.bmm(x, y))
    print(
        f"Helion time: {sec:.4f}s, torch time: {baseline_sec:.4f}, speedup: {baseline_sec / sec:.2f}x"
    )


if __name__ == "__main__":
    check(8, 512, 768, 1024)
