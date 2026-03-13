"""Pallas backend: same rms_norm kernel targeting TPU via JAX/Pallas."""
import torch
import helion
from helion._testing import DEVICE
import helion.language as hl

CONFIG = helion.Config(block_sizes=[1], num_warps=4)


@helion.kernel(config=CONFIG, backend="pallas", static_shapes=True)
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        rms = torch.sqrt(torch.mean(row * row, dim=-1) + eps)
        out[tile_m, :] = (row / rms[:, None] * weight[:].to(torch.float32)).to(out.dtype)
    return out


x = torch.randn(256, 1024, device=DEVICE, dtype=torch.bfloat16)
w = torch.randn(1024, device=DEVICE, dtype=torch.bfloat16)
print(rms_norm.bind((x, w)).to_triton_code(CONFIG))
