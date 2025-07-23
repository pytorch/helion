from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


# static_shapes=True gives a performance boost for matmuls
@helion.kernel(static_shapes=True)
def matmul_no_bias(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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


@helion.kernel(static_shapes=True)
def matmul_with_bias(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    bias_size = bias.size(0)
    assert bias_size == n, f"bias size mismatch, expected {n}, got {bias_size}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        # Add bias
        acc = acc + bias[tile_n]
        out[tile_m, tile_n] = acc
    return out


def matmul(x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """Wrapper function for tritonbench that dispatches based on bias presence."""
    if bias is None:
        return matmul_no_bias(x, y)
    else:
        return matmul_with_bias(x, y, bias)


def check(m: int, k: int, n: int) -> None:
    x = torch.randn([m, k], device="cuda", dtype=torch.float16)
    y = torch.randn([k, n], device="cuda", dtype=torch.float16)
    
    # Test without bias
    run_example(matmul_no_bias, torch.matmul, (x, y))
    
    # Test with bias
    bias = torch.randn([n], device="cuda", dtype=torch.float16)
    expected_with_bias = lambda x, y, bias: torch.matmul(x, y) + bias
    run_example(matmul_with_bias, expected_with_bias, (x, y, bias))


def main() -> None:
    check(1024, 1024, 1024)


if __name__ == "__main__":
    main()
