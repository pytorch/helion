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


# This optimization does softmax in fewer passes, but is less numerically stable
@helion.kernel(config={"block_sizes": [1, 128]})
def softmax_two_pass(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m, 1], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m, block_size_n], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1, keepdim=True)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(values - mi_next)
            mi = mi_next
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi) / di
    return out


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
