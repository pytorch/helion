from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel()
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # match pytorch broadcasting rules
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty(
        x.shape,
        # match type promotion of torch.add
        dtype=torch.promote_types(x.dtype, y.dtype),
        device=x.device,
    )
    # tile will be a tuple of blocks
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


def check(m: int, n: int) -> None:
    from triton.testing import do_bench

    x = torch.randn([m, n], device="cuda", dtype=torch.float16)
    y = torch.randn([m, n], device="cuda", dtype=torch.float16)
    result = add(x, y)
    torch.testing.assert_close(result, x + y, rtol=1e-2, atol=1e-1)
    sec = do_bench(lambda: add(x, y))
    baseline_sec = do_bench(lambda: torch.add(x, y))
    print(
        f"Helion time: {sec:.4f}s, torch time: {baseline_sec:.4f}, speedup: {baseline_sec / sec:.2f}x"
    )


if __name__ == "__main__":
    check(1024, 1024)
