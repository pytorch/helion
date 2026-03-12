"""Quick test: emit_pipeline with looped reduction."""
from __future__ import annotations

import torch
import helion
import helion.language as hl
import torch_tpu.api


DEVICE = torch_tpu.api.tpu_device()


@helion.kernel(
    backend="pallas",
    static_shapes=True,
    config=helion.Config(
        block_sizes=[1024],
        reduction_loops=[128],
        pallas_loop_type="emit_pipeline",
    ),
)
def sum_kernel(x: torch.Tensor) -> torch.Tensor:
    m, _ = x.size()
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(-1)
    return out


def main() -> None:
    x = torch.randn(2048, 512, device=DEVICE, dtype=torch.float32)
    result = sum_kernel(x)
    expected = x.sum(-1)
    err = (result - expected).abs().max().item()
    print(f"max error: {err}")
    assert err < 1e-4, f"error too large: {err}"
    print("PASSED")


if __name__ == "__main__":
    main()
