"""
Element-wise Addition Example
=============================

This example demonstrates how to implement an element-wise addition kernel using Helion.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# Addition Kernel
# ---------------


# %%
@helion.kernel(config=helion.Config(block_sizes=[128]))
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # tile will be a tuple of blocks
    for tile in hl.tile(x.size()):
        hl.start_async_remote_copy(x[tile], y[tile], device_id=0)

 

# %%
def main() -> None:
    x = torch.randn([1024], device=DEVICE, dtype=torch.bfloat16)
    y = torch.randn([1024], device=DEVICE, dtype=torch.bfloat16)
    add(x, y)
    print(y)

if __name__ == "__main__":
    main()
