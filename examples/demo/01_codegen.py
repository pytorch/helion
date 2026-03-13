"""Manual config -> Triton codegen."""
import torch
import helion
from helion._testing import DEVICE
import helion.language as hl

CONFIG = helion.Config(
    block_sizes=[1], num_warps=4,
    indexing=["block_ptr", "pointer", "block_ptr",
              "block_ptr", "block_ptr", "block_ptr", "block_ptr"],
    load_eviction_policies=["last", "first", "", "last", ""],
    pid_type="persistent_blocked",
    reduction_loops=[256],
)


@helion.kernel(config=CONFIG, static_shapes=True)
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
print(f"Config: {CONFIG}\n")
print(rms_norm.bind((x, w)).to_triton_code(CONFIG))

result = rms_norm(x, w)
row = x.float()
rms = torch.sqrt(torch.mean(row * row, dim=-1, keepdim=True) + 1e-5)
assert torch.allclose(result, ((row / rms) * w.float()).to(torch.bfloat16), atol=1e-2, rtol=1e-2)
print("Correctness: PASSED")
