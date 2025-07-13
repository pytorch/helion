from __future__ import annotations

import torch

import helion
from helion._testing import run_example
import helion.language as hl


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def gather_gemv(w: torch.Tensor, idx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    batch_size, s, s2 = w.size()
    num_indices = idx.size(0)
    assert s == s2, f"size mismatch {s} != {s2}"
    assert x.size(0) == s, f"vector size mismatch {x.size(0)} != {s}"
    
    out = torch.empty([num_indices, s], dtype=x.dtype, device=w.device)
    
    # Handle negative indices by wrapping around
    idx_wrapped = torch.where(idx < 0, idx + batch_size, idx)
    
    for tile_i, tile_j in hl.tile([num_indices, s]):
        acc = hl.zeros([tile_i, tile_j], dtype=torch.float32)
        for tile_k in hl.tile(s):
            # Get the indices for this tile
            indices_tile = idx_wrapped[tile_i]
            # Gather matrix elements: w[indices_tile, tile_j, tile_k]
            # This gives us shape [tile_i, tile_j, tile_k]
            # We need to sum over k dimension
            w_tile = w[indices_tile, tile_j, tile_k]
            x_tile = x[tile_k]
            # Multiply and accumulate
            # w_tile has shape [tile_i, tile_j, tile_k]
            # x_tile has shape [tile_k]
            # We want to compute: out[tile_i, tile_j] += sum_k(w_tile[:, :, k] * x_tile[k])
            acc = acc + (w_tile.to(torch.float32) * x_tile.to(torch.float32)).sum(dim=-1)
        out[tile_i, tile_j] = acc.to(x.dtype)
    
    return out


def gather_gemv_tritonbench(
    w: torch.Tensor, idx: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Wrapper for tritonbench that matches its interface."""
    return gather_gemv(w, idx, x)


def gather_gemv_ref(w: torch.Tensor, idx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reference implementation for testing."""
    return w[idx].to(x.dtype) @ x


def main() -> None:
    s = 2048
    # Create int8 tensor by generating float and converting
    w_float = torch.randn([8, s, s], device="cuda", dtype=torch.float32)
    w = (w_float * 127).to(torch.int8)
    idx = torch.randint(0, 8, [2], device="cuda", dtype=torch.int64)
    x = torch.randn([s], device="cuda", dtype=torch.bfloat16)
    
    run_example(gather_gemv, gather_gemv_ref, (w, idx, x), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    main()