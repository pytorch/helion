from __future__ import annotations

import torch

import helion
import helion.language as hl


# static_shapes=True gives a performance boost for matmuls
@helion.kernel(static_shapes=True)
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


def check(n: int, k: int, m: int) -> None:
    from triton.testing import do_bench

    x = torch.randn([n, k], device="cuda", dtype=torch.float16)
    y = torch.randn([k, m], device="cuda", dtype=torch.float16)
    result = matmul(x, y)
    torch.testing.assert_close(result, x @ y, rtol=1e-2, atol=1e-1)
    sec = do_bench(lambda: matmul(x, y))
    baseline_sec = do_bench(lambda: torch.matmul(x, y))
    print(
        f"Helion time: {sec:.4f}s, torch time: {baseline_sec:.4f}, speedup: {baseline_sec / sec:.2f}x"
    )


if __name__ == "__main__":
    check(1024, 1024, 1024)
