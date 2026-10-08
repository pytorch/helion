from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessHostTensorDescriptor
from helion.runtime.triton import cache_hints


@triton.jit
def _streamed_copy(x_desc, out):
    tile = cache_hints.descriptor_load(x_desc, [0, 0], eviction_policy="evict_first")
    rows = tl.arange(0, 16)
    tl.store(out + rows[:, None] * 16 + rows[None, :], tile)


@onlyBackends(["triton"])
class TestCacheHints(TestCase):
    @skipIfNotCUDA()
    @skipIfCudaCapabilityLessThan((9, 0), reason="TMA needs sm90")
    @skipUnlessHostTensorDescriptor("host tensor descriptors are required")
    def test_tma_load_keeps_its_eviction_policy(self) -> None:
        x = torch.randn((16, 16), device=DEVICE, dtype=torch.float32)
        out = torch.empty_like(x)
        compiled = _streamed_copy[(1,)](TensorDescriptor.from_tensor(x, [16, 16]), out)

        torch.testing.assert_close(out, x)
        ptx = compiled.asm["ptx"]
        self.assertIn("createpolicy.fractional.L2::evict_first", ptx)
        self.assertIn(".L2::cache_hint", ptx)
