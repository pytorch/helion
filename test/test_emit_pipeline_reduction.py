"""Test: emit_pipeline with looped reduction on TPU."""
from __future__ import annotations

import unittest

import torch

import helion
import helion.language as hl
from helion._testing import skipUnlessPallas


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


class TestEmitPipelineReduction(unittest.TestCase):
    @skipUnlessPallas("requires TPU + Pallas")
    def test_sum(self) -> None:
        import torch_tpu.api  # type: ignore[import-not-found]

        device = torch_tpu.api.tpu_device()
        x = torch.randn(2048, 512, device=device, dtype=torch.float32)
        result = sum_kernel(x)
        expected = x.sum(-1)
        err = (result - expected).abs().max().item()
        self.assertLess(err, 1e-4, f"max error too large: {err}")


if __name__ == "__main__":
    unittest.main()
