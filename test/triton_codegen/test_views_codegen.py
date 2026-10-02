"""Emitted-code assertions for the portable view tests.

The numerics halves of these kernels run on every backend in
``test/portable/test_views.py``; the assertions below are about the Triton
source Helion produces, so they live here behind ``@onlyBackends``,
``@skipIfNotTriton``, and ``@skipIfRefEager`` (ref-eager returns Python source,
not emitted Triton, and tileir shares the triton backend alias but may diverge
in lowering).
"""

from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import skipIfNotTriton
from helion._testing import skipIfRefEager
import helion.language as hl


@skipIfRefEager(
    "code_and_output returns Python source in ref-eager, not emitted Triton"
)
@skipIfNotTriton("tl.* assertions are triton-specific codegen, not shared with tileir")
class TestViewsCodegen(TestCase):
    def test_split_join_roundtrip_lowers_to_split_join(self):
        @helion.kernel(config={"block_size": 64})
        def fn(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.empty_like(x)
            for tile in hl.tile(n):
                lo, hi = hl.split(x[tile, :])
                out[tile, :] = hl.join(hi, lo)
            return out

        x = torch.randn([256, 2], device=DEVICE)
        code, _result = code_and_output(fn, (x,))
        self.assertIn("tl.split", code)
        self.assertIn("tl.join", code)

    def test_join_broadcast_scalar_lowers_to_join(self):
        @helion.kernel(config={"block_size": 64})
        def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.empty([n, 2], dtype=x.dtype, device=x.device)
            for tile in hl.tile(n):
                scalar = hl.load(y, [0])
                out[tile, :] = hl.join(x[tile], scalar)
            return out

        x = torch.randn([128], device=DEVICE)
        y = torch.randn([1], device=DEVICE)
        code, _result = code_and_output(fn, (x, y))
        self.assertIn("tl.join", code)

    def test_view_blocksize_constexpr_lowers_to_reshape(self):
        @helion.kernel(static_shapes=True, autotune_effort="none")
        def fn(x: torch.Tensor) -> torch.Tensor:
            N = x.shape[0]
            N = hl.specialize(N)
            out = x.new_empty(N // 2)
            for (n_tile,) in hl.tile([N]):
                val = x[n_tile]
                val = val.view(n_tile.block_size // 2, 2)
                val_a, val_b = hl.split(val)
                out[n_tile.begin + hl.arange(0, n_tile.block_size // 2)] = val_a + val_b
            return out

        x = torch.randn(1024, dtype=torch.bfloat16, device=DEVICE)
        code, result = code_and_output(fn, (x,))
        self.assertEqual(result.numel(), x.numel() // 2)
        self.assertIn("tl.reshape", code)

    def test_view_dtype_reinterpret_bitcasts_to_int16(self):
        @helion.kernel(static_shapes=True)
        def fn(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.empty_like(x)
            for tile in hl.tile(n):
                val = x[tile]
                val_as_int = val.view(dtype=torch.int16)
                val_as_int = val_as_int + 1
                val_back = val_as_int.view(dtype=torch.bfloat16)
                out[tile] = val_back
            return out

        x = torch.randn(1024, dtype=torch.bfloat16, device=DEVICE)
        code, _result = code_and_output(fn, (x,))
        self.assertTrue(
            ".to(tl.int16)" in code or "tl.cast(" in code,
            "Expected bitcast to int16 via .to() or tl.cast()",
        )
