from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfCudaCapabilityLessThan
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessHostTensorDescriptor
import helion.language as hl
from helion.runtime.triton import cache_hints


@triton.jit
def _streamed_copy(x_desc, out):
    tile = cache_hints.descriptor_load(x_desc, [0, 0], eviction_policy="evict_first")
    rows = tl.arange(0, 16)
    tl.store(out + rows[:, None] * 16 + rows[None, :], tile)


@triton.jit
def _warmed_copy(x, out):
    cache_hints.code_warm()
    offsets = tl.arange(0, 128)
    tl.store(out + offsets, tl.load(x + offsets))


@helion.kernel(static_shapes=True, autotune_effort="none")
def two_roots(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile_a in hl.tile(x.size(0), block_size=1):
        y[tile_a, :] = x[tile_a, :] + 1
    for tile_b in hl.tile(x.size(0), block_size=1):
        out[tile_b, :] = y[tile_b, :] * 2
    return out


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

    @skipIfNotCUDA()
    def test_marked_kernel_warms_its_code(self) -> None:
        x = torch.randn(128, device=DEVICE, dtype=torch.float32)
        out = torch.empty_like(x)
        compiled = _warmed_copy[(1,)](x, out)

        torch.testing.assert_close(out, x)
        ptx = compiled.asm["ptx"]
        self.assertIn(".global .align 8 .u64 helion_codewarm_tab[2]", ptx)
        self.assertIn("ld.global.cg.L2::cache_hint.u32", ptx)

    @skipIfNotCUDA()
    @skipIfRefEager("persistent tile-dependency codegen is unavailable")
    def test_megakernel_is_marked_for_code_warm(self) -> None:
        x = torch.randn((8, 64), device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            two_roots,
            (x,),
            pid_type="persistent_blocked",
            cross_loop_pipeline="dynamic",
        )

        torch.testing.assert_close(out, (x + 1) * 2)
        self.assertIn("helion_cache_hints.code_warm()", code)
