"""How a Metal kernel takes its tensor arguments (``metal/metal_jit.py``):
each as a device pointer, a 0-d one included.  Renders on CPU tensors
(``helion._testing.metal.render_msl``), with no Apple GPU; ``test_metal.py``
runs the same kernel."""

from __future__ import annotations

import unittest

import pytest
import torch

import helion
from helion._testing import skipUnlessBackends
from helion._testing.metal import render_msl
from helion._testing.metal import scale_by_tensor
import helion.language as hl

# Metal codegen only: other backends' CPU lanes skip the module.
pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]


@helion.kernel(backend="metal", static_shapes=True)
def double_phase_named(helion_phase: torch.Tensor) -> torch.Tensor:
    """Twice its argument, named like the phase helpers' namespace."""
    out = torch.empty_like(helion_phase)
    for tile in hl.tile(helion_phase.size(0)):
        out[tile] = helion_phase[tile] * 2.0
    return out


class TestMetalKernelArgs(unittest.TestCase):
    def test_zero_dim_tensor(self) -> None:
        """A 0-d tensor is a device pointer whose element the body reads
        where it loads it, not a scalar copied at kernel start, which
        ``scale[0]`` could not index."""
        for dtype, metal in ((torch.float32, "float"), (torch.float16, "half")):
            with self.subTest(dtype=dtype):
                msl = render_msl(
                    scale_by_tensor,
                    (torch.randn(64, dtype=dtype), torch.ones((), dtype=dtype)),
                    helion.Config(block_sizes=[64]),
                )
                self.assertIn(f"device {metal}* scale [[buffer(1)]]", msl)
                self.assertIn("scale[0]", msl)
                self.assertNotIn("_buf_scale", msl)

    def test_phase_namespace_argument(self) -> None:
        """An argument named like the phase helpers' namespace reaches a
        kernel without hl.barrier() renamed, which renders without the
        phase helpers."""
        msl = render_msl(
            double_phase_named, (torch.randn(64),), helion.Config(block_sizes=[64])
        )
        self.assertNotIn("helion_phase [[buffer(0)]]", msl)
        self.assertNotIn("namespace helion_phase", msl)


if __name__ == "__main__":
    unittest.main()
