"""Helion + torch.compile: rms_norm + quantization via closure
epilogue (works today) vs Inductor fusion (coming soon).
"""
import torch
import helion
from helion._testing import DEVICE
import helion.language as hl

CONFIG = helion.Config(block_sizes=[1], num_warps=4)


# Approach 1: closure epilogue (works today, fused in one kernel)
@helion.kernel(config=CONFIG, static_shapes=True)
def rms_norm(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor,
             epilogue=lambda out, tile: out, eps: float = 1e-5) -> torch.Tensor:
    m, n = x.size()
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        rms = torch.sqrt(torch.mean(row * row, dim=-1) + eps)
        normalized = row / rms[:, None] * weight[:].to(torch.float32)
        out[tile_m, :] = epilogue(normalized, tile_m).to(out.dtype)
    return out


# Approach 2: Inductor fusion (coming soon)
@helion.kernel(config=CONFIG, static_shapes=True)
def rms_norm_plain(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        rms = torch.sqrt(torch.mean(row * row, dim=-1) + eps)
        out[tile_m, :] = (row / rms[:, None] * weight[:].to(torch.float32)).to(out.dtype)
    return out


def rms_norm_quantized_inductor(x, weight, scale):
    normed = rms_norm_plain(x, weight)  # Helion kernel
    return torch.clamp(torch.round(normed.float() * scale), -128, 127).to(torch.int8)
# compiled = torch.compile(rms_norm_quantized_inductor, backend="inductor")


x = torch.randn(256, 1024, device=DEVICE, dtype=torch.bfloat16)
w = torch.randn(1024, device=DEVICE, dtype=torch.bfloat16)
scale = 127.0

print("-- Approach 1: closure epilogue (works today) --")
print("Quantize fused inside the kernel, no extra memory traffic.\n")
out = torch.empty(256, 1024, dtype=torch.int8, device=DEVICE)
bound = rms_norm.bind((x, w, out, lambda n, t: torch.clamp(torch.round(n * scale), -128, 127)))
print(bound.to_triton_code(CONFIG))

result = rms_norm(x, w, out, lambda n, t: torch.clamp(torch.round(n * scale), -128, 127))
row = x.float()
rms = torch.sqrt(torch.mean(row * row, dim=-1, keepdim=True) + 1e-5)
expected = torch.clamp(torch.round((row / rms * w.float()) * scale), -128, 127).to(torch.int8)
assert result.dtype == torch.int8
assert torch.allclose(result.float(), expected.float(), atol=1, rtol=0)
print(f"Output dtype: {result.dtype}")
print("Correctness: PASSED")

print("\n-- Approach 2: Inductor fusion (coming soon) --")
print("torch.compile auto-fuses quantize epilogue into Helion launch.")
print("  compiled = torch.compile(rms_norm_quantized_inductor, backend='inductor')")
