# Issue #447: HELION_INTERPRET=1 silently hides HELION_PRINT_OUTPUT_CODE=1

## Metadata
- **State**: OPEN
- **Author**: [xmfan](https://github.com/xmfan)
- **Created**: August 07, 2025 at 21:13 UTC
- **Updated**: August 09, 2025 at 16:02 UTC

## Description

`HELION_INTERPRET=1 HELION_PRINT_OUTPUT_CODE=1 python mykernel.py` behaves like `HELION_INTERPRET=1 python mykernel.py`. Maybe we could print the interpreter's eager code.

```python
import logging

from typing import Tuple
import helion
import helion.language as hl
import torch
from torch import Tensor

# If you set this to info you will see the output Triton Code
logging.getLogger().setLevel(logging.WARNING)

# The @helion.kernel decorator marks this function for Helion compilation.
# `use_default_config=True` will skip autotuning and is handy for quick development iteration, but the generated kernel will be slow.
# To get best performance, leave it empty (i.e. `@helion.kernel`) to enable autotuning.
@helion.kernel(use_default_config=True)
def example_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Host code: Standard PyTorch operations
    m, n = x.size()
    out = torch.empty_like(x)  # Allocate output tensor

    # The hl.tile loop (device loop) defines the parallel execution structure
    for tile_m, tile_n in hl.tile([m, n]):
        # Device code: Everything inside the hl.tile loop runs on GPU
        out[tile_m, tile_n] = (
            x[tile_m, tile_n] + y[tile_m, tile_n]
        )  # Simple element-wise addition expressed w/ pytorch ops

    return out  # Return the result back to the host


# Alternatively, you can use @helion.kernel(config=helion.Config(block_sizes = [32, 32])) to manually specify block size and other configs

# Create some sample data
x = torch.randn(512, 512, device="cuda")
y = torch.randn(512, 512, device="cuda")

# Run the kernel
result = example_add(x, y)

# Verify result
expected = x + y
torch.testing.assert_close(result, expected)
print("✅ Results Match ✅")
```

## Comments

### Comment 1 by [jansel](https://github.com/jansel)
*Posted on August 09, 2025 at 16:02 UTC*

The eager code is just your original kernel.  There is no generated code.

---
