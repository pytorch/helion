#!/usr/bin/env python3
"""Simple test of slice with symbolic bounds in device loop"""

import torch
import helion
import helion.language as hl

@helion.kernel(use_default_config=True)
def simple_slice_kernel(x: torch.Tensor) -> torch.Tensor:
    N = x.size(0)
    block_size = hl.register_block_size(N)
    
    out = torch.zeros_like(x)
    
    # Device loop with slice inside
    for tile in hl.tile(N, block_size=block_size):
        # This creates slice(0, block_size) where block_size is SymInt
        # The slice is inside the device loop, so FX needs to trace it
        data = x[tile][0:block_size]  # Slice with symbolic bound
        out[tile] = data
    
    return out

# Test it
x = torch.randn(64, device='cuda')
print("Input shape:", x.shape)

try:
    result = simple_slice_kernel(x)
    print("Success! Output shape:", result.shape)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()