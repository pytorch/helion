"""Debug script to understand why SliceProxy.to_slice() might fail."""

import torch
import helion
import helion.language as hl
from helion._compiler.compile_environment import CompileEnvironment
from helion.language.slice_proxy import SliceProxy

# First, let's try to recreate the scenario
@helion.kernel(use_default_config=True)
def test_kernel(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    M, N = src.shape
    block_size_m = hl.register_block_size(32)
    block_size_n = hl.register_block_size(16)
    
    # Try to create a slice like in the test
    end_idx = block_size_m - block_size_n + 1  # Should be 32 - 16 + 1 = 17
    
    # This should create a SliceProxy
    # Let's see if we can examine it
    import builtins
    slice_obj = builtins.slice(0, end_idx)
    print(f"Slice object type: {type(slice_obj)}")
    print(f"Slice bounds: start={slice_obj.start}, stop={slice_obj.stop}, step={slice_obj.step}")
    
    # Now let's see what happens when we use it
    for tile_m in hl.tile(M, block_size=block_size_m):
        for tile_n in hl.tile(N, block_size=block_size_n):
            if tile_m.begin == 0 and tile_n.begin == 0:
                dst[0:end_idx, tile_n] = src[0:end_idx, tile_n]
            else:
                dst[tile_m, tile_n] = src[tile_m, tile_n]
    
    return dst

# Trace the kernel to see what happens
M, N = 64, 32
src = torch.ones([M, N], device="cuda") * 2.0
dst = torch.zeros_like(src)

try:
    # Enable debug mode if available
    import os
    os.environ["HELION_DEBUG"] = "1"
    
    # Compile with pointer indexing
    from helion._compiler.compile import compile_for_args
    code, _ = compile_for_args(
        test_kernel,
        (src, dst),
        indexing="pointer",
        block_size=[32, 16],
    )
    print("Generated code:")
    print(code)
except Exception as e:
    print(f"Error during compilation: {e}")
    import traceback
    traceback.print_exc()