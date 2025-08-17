"""Debug the mixed types test to see generated code."""

import torch
import helion
import helion.language as hl
from helion._testing import code_and_output, DEVICE

@helion.kernel(use_default_config=True)
def kernel(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    M, N = src.shape
    block_size_m = hl.register_block_size(32)
    block_size_n = hl.register_block_size(16)
    
    for tile_m in hl.tile(M, block_size=block_size_m):
        for tile_n in hl.tile(N, block_size=block_size_n):
            if tile_m.begin == 0 and tile_n.begin == 0:
                # Test block_size_m - block_size_n + 1 arithmetic in slice
                # This creates a slice with SymInt arithmetic: [tile_m.begin:tile_m.begin + block_size_m - block_size_n + 1]
                # Which is [0:0+32-16+1] = [0:17]
                end_idx1 = tile_m.begin + block_size_m - block_size_n + 1
                dst[tile_m.begin : end_idx1, tile_n] = src[
                    tile_m.begin : end_idx1, tile_n
                ]
            elif tile_m.begin == 32 and tile_n.begin == 0:
                # Test block_size_m - block_size_n + 16 arithmetic in slice
                # This creates a slice with SymInt arithmetic: [tile_m.begin:tile_m.begin + block_size_m - block_size_n + 16]
                # Which is [32:32+32-16+16] = [32:64]
                # This should align with block_size_m and potentially use block_ptr
                end_idx2 = tile_m.begin + block_size_m - block_size_n + 16
                dst[tile_m.begin : end_idx2, tile_n] = src[
                    tile_m.begin : end_idx2, tile_n
                ]
            else:
                # Regular copy for other tiles
                dst[tile_m, tile_n] = src[tile_m, tile_n]
    
    return dst

M, N = 64, 32
src = torch.ones([M, N], device=DEVICE) * 2.0
dst = torch.zeros_like(src)

# Enable debug
import os
os.environ["HELION_DEBUG_SLICE"] = "1"

code, result = code_and_output(
    kernel,
    (src, dst),
    indexing="pointer",
    block_size=[32, 16],
)

print("Generated Triton code:")
print(code)
print("\nResult shape:", result.shape)
print("Result[0:17, 0:16] unique values:", torch.unique(result[0:17, 0:16]))
print("Expected all 2.0, got zeros at:", torch.where(result[0:17, 0:16] == 0))