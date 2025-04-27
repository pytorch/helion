from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(config={"block_size": 1})
def softmax(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        out[tile_n, :] = torch.nn.functional.softmax(x[tile_n, :], dim=1)
    return out


# This generates the same code as the above, but avoids using the pytorch softmax decomposition
@helion.kernel(config={"block_size": 1})
def softmax_decomposed(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        values = x[tile_n, :]
        amax = torch.amax(values, dim=1, keepdim=True)
        exp = torch.exp(values - amax)
        sum_exp = torch.sum(exp, dim=1, keepdim=True)
        out[tile_n, :] = exp / sum_exp
    return out


# TODO(jansel): we should add support for view ops (or broadcasting) so the `keepdim` arg isn't needed
# TODO(jansel): we should add support for constexpr args and make dim a constexpr arg


def check(m: int, n: int) -> None:
    from triton.testing import do_bench

    x = torch.randn([m, n], device="cuda", dtype=torch.float16)
    result = softmax(x)
    torch.testing.assert_close(
        result, torch.nn.functional.softmax(x, dim=1), rtol=1e-2, atol=1e-1
    )
    sec = do_bench(lambda: softmax(x))
    baseline_sec = do_bench(lambda: torch.nn.functional.softmax(x, dim=1))
    print(
        f"Helion time: {sec:.4f}s, torch time: {baseline_sec:.4f}, speedup: {baseline_sec / sec:.2f}x"
    )


if __name__ == "__main__":
    check(1024, 1024)
