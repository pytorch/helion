"""Autotuned config -> Triton codegen."""
import os
os.environ["HELION_AUTOTUNE_EFFORT"] = "quick"
os.environ["HELION_SKIP_CACHE"] = "1"

import torch
import helion
from helion._testing import DEVICE
import helion.language as hl


@helion.kernel(static_shapes=True)
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        rms = torch.sqrt(torch.mean(row * row, dim=-1) + eps)
        out[tile_m, :] = (row / rms[:, None] * weight[:].to(torch.float32)).to(out.dtype)
    return out


x = torch.randn(2048, 4096, device=DEVICE, dtype=torch.bfloat16)
w = torch.randn(4096, device=DEVICE, dtype=torch.bfloat16)

result = rms_norm(x, w)
row = x.float()
rms = torch.sqrt(torch.mean(row * row, dim=-1, keepdim=True) + 1e-5)
assert torch.allclose(result, ((row / rms) * w.float()).to(torch.bfloat16), atol=1e-2, rtol=1e-2)
print("Correctness: PASSED")
