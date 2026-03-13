"""Inline Triton and inline PTX for fine-grained numerics control.
Shows both hl.inline_triton and hl.inline_asm_elementwise in one kernel.
"""
import torch
import helion
from helion._testing import DEVICE
import helion.language as hl

CONFIG = helion.Config(block_sizes=[1], num_warps=4)


@helion.kernel(config=CONFIG, static_shapes=True)
def rms_norm_inline(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        mean_sq = torch.mean(x[tile_m, :].to(torch.float32) ** 2, dim=-1)
        # inline_triton: call tl.math.rsqrt directly
        inv_rms = hl.inline_triton(
            "tl.math.rsqrt({v} + {eps})",
            args={"v": mean_sq, "eps": eps}, output_like=mean_sq,
        )
        # Could also use PTX: hl.inline_asm_elementwise(
        #     "rsqrt.approx.f32 $0, $1;", "=f,f",
        #     [mean_sq + eps], dtype=torch.float32, is_pure=True, pack=1)
        out[tile_m, :] = (x[tile_m, :].to(torch.float32) * inv_rms[:, None]
                          * weight[:].to(torch.float32)).to(out.dtype)
    return out


x = torch.randn(256, 1024, device=DEVICE, dtype=torch.bfloat16)
w = torch.randn(1024, device=DEVICE, dtype=torch.bfloat16)
print(rms_norm_inline.bind((x, w)).to_triton_code(CONFIG))

row = x.float()
rms = torch.sqrt(torch.mean(row * row, dim=-1, keepdim=True) + 1e-5)
expected = ((row / rms) * w.float()).to(torch.bfloat16)
assert torch.allclose(rms_norm_inline(x, w), expected, atol=1e-1, rtol=1e-1)
print("Correctness: PASSED")
