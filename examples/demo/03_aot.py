"""AOT tuning -> find best config for any shape."""
import os, subprocess, sys
import torch
from triton.testing import do_bench
from helion._testing import DEVICE
import helion.experimental
import helion.language as hl


@helion.experimental.aot_kernel(batched=[[0, None], [None], None])
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        rms = torch.sqrt(torch.mean(row * row, dim=-1) + eps)
        out[tile_m, :] = (row / rms[:, None] * weight[:].to(torch.float32)).to(out.dtype)
    return out


def benchmark_rms_norm() -> None:
    print(f"{'Shape':>16} {'ms':>8} {'GB/s':>8}")
    for b, h in [(512, 2048), (1024, 4096), (2048, 4096)]:
        x = torch.randn(b, h, device=DEVICE, dtype=torch.bfloat16)
        w = torch.randn(h, device=DEVICE, dtype=torch.bfloat16)
        rms_norm(x, w)
        ms = do_bench(lambda x=x, w=w: rms_norm(x, w))
        print(f"{(b,h)!s:>16} {ms:>8.4f} {x.numel()*x.element_size()*2/ms*1e-6:>8.1f}")


aot_mode = os.environ.get("HELION_AOT_MODE", "disabled")
if aot_mode != "disabled":
    print(f"AOT mode: {aot_mode}")
    benchmark_rms_norm()
else:
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_aot_data")
    print(f"Launching aot_runner -> {output_dir}", flush=True)
    env = {**os.environ, "HELION_AUTOTUNE_EFFORT": "quick"}
    cmd = [sys.executable, "-m", "helion.experimental.aot_runner",
           "--output-dir", output_dir, "--", sys.executable, __file__]
    sys.exit(subprocess.run(cmd, env=env).returncode)
