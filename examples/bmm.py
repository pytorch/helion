from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


# static_shapes=True gives a performance boost for matmuls
@helion.kernel(static_shapes=True)
def bmm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [B, M, K], B: [B, K, N], Out: [B, M, N]   # dense bmm
    b, m, k = A.size()
    b, k, n = B.size()
    out = torch.empty(
        [b, m, n], device=A.device, dtype=torch.promote_types(A.dtype, B.dtype)
    )
    for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc, A[tile_b, tile_m, tile_k], B[tile_b, tile_k, tile_n]
            )
        out[tile_b, tile_m, tile_n] = acc
    return out


def check(b: int, m: int, k: int, n: int) -> None:
    x = torch.randn([b, m, k], device="cuda", dtype=torch.float16)
    y = torch.randn([b, k, n], device="cuda", dtype=torch.float16)
    run_example(bmm, torch.bmm, (x, y))


def main() -> None:
    # torch.baddbmm support for 16-bit tensors requires torch 2.8+
    assert torch.__version__.split(".")[:2] >= ["2", "8"], "Requires torch 2.8+"
    check(16, 512, 768, 1024)


if __name__ == "__main__":
    main()
