from __future__ import annotations

import torch
import torch.nn.functional as F

import helion
import helion.language as hl


# static_shapes=True gives a performance boost for matmuls
@helion.kernel(static_shapes=True)
def matmul_ln(
    x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {n}"
    assert bias.size(0) == n, f"bias size mismatch {bias.size(0)} != {n}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n], block_size=[None, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            mm = torch.matmul(x[tile_m, tile_k], y[tile_k, tile_n])
            acc = acc + mm
        acc = F.layer_norm(
            acc,
            [acc.size(1)],
            weight[tile_n].to(torch.float32),
            bias[tile_n].to(torch.float32),
        )
        out[tile_m, tile_n] = acc
    return out


def matmul_ln_pytorch(
    x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    matmul_out = torch.matmul(x, y)

    ln_out = F.layer_norm(
        matmul_out.to(torch.float32),
        normalized_shape=(matmul_out.shape[-1],),
        weight=weight.to(torch.float32),
        bias=bias.to(torch.float32),
    )

    return ln_out.to(torch.promote_types(x.dtype, y.dtype))


def check(m: int, k: int, n: int) -> None:
    from triton.testing import do_bench

    x = torch.randn([m, k], device="cuda", dtype=torch.float16)
    y = torch.randn([k, n], device="cuda", dtype=torch.float16)
    weight = torch.randn([n], device="cuda", dtype=torch.float16)
    bias = torch.randn([n], device="cuda", dtype=torch.float16)
    result = matmul_ln(x, y, weight, bias)
    expected = matmul_ln_pytorch(x, y, weight, bias)
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-1)
    sec = do_bench(lambda: matmul_ln(x, y, weight, bias))
    baseline_sec = do_bench(lambda: matmul_ln_pytorch(x, y, weight, bias))
    print(
        f"Helion time: {sec:.4f}s, torch time: {baseline_sec:.4f}, speedup: {baseline_sec / sec:.2f}x"
    )


def main() -> None:
    check(32, 64, 128)


if __name__ == "__main__":
    main()
