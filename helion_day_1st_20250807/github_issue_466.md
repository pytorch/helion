# Issue #466: HELION_INTERPRET=1 only prints once for fixed block_size

## Metadata
- **State**: OPEN
- **Author**: [pianpwk](https://github.com/pianpwk)
- **Created**: August 08, 2025 at 02:03 UTC
- **Updated**: August 09, 2025 at 15:58 UTC

## Description

Maybe I misunderstand, or this is a known issue, but this program only prints once with HELION_INTERPRET=1, I expected it to print 16 times (following TRITON_INTERPRET behavior)?
```python
import torch

import helion
from helion._testing import run_example
import helion.language as hl
from torch._C import device


@helion.kernel()
def abc_kernel(x: torch.Tensor) -> torch.Tensor:
    out = hl.zeros(x.shape, dtype=x.dtype)
    for tile in hl.tile(x.size(0), block_size=8):
        out[tile] += x[tile] * 2
        print("tile", tile)
    return out


if __name__ == "__main__":
    x = torch.randn(128, device="cuda")
    abc_kernel(x)
```

output:
```
tile RefTile(slice(0, 128, None))
```

## Comments

### Comment 1 by [oulgen](https://github.com/oulgen)
*Posted on August 08, 2025 at 04:28 UTC*

@yf225 
this might be expected since interp mode runs everything on the entire input

---

### Comment 2 by [jansel](https://github.com/jansel)
*Posted on August 09, 2025 at 15:58 UTC*

This is expected in order to get fast eager code you can trace through.

---
