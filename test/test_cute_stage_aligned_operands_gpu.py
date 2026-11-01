"""GPU tests of the CuTe unaligned-operand staging pass.

A GEMM over odd-sized rows (``K = 1025`` bf16: 2050-byte strides) fails the
TensorMap proof; with the staging pass the host pads the operands and the
tcgen05 default config keeps its TMA producers, and the result matches torch.
"""

from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
import helion.language as hl


@helion.kernel(static_shapes=True)
def _gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@onlyBackends(["cute"])
@skipIfNotCUDA()
class TestCuteStageAlignedOperandsGpu(RefEagerTestDisabled, TestCase):
    def _check(self, a: torch.Tensor, b: torch.Tensor, staged: tuple[str, ...]) -> None:
        bound = _gemm.bind((a, b))
        config = bound.config_spec.default_config()
        code = bound.to_code(config)
        for name in ("a", "b"):
            self.assertEqual(f"{name} = torch.constant_pad_nd(" in code, name in staged)
        if staged:
            self.assertIn("tma_tensor_a", code)
            self.assertIn("tma_tensor_b", code)
        bound.set_config(config)
        torch.testing.assert_close(bound(a, b), a @ b, rtol=1e-2, atol=1e-1)

    def test_odd_rows_both_operands(self) -> None:
        a = torch.randn(1025, 1025, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(1025, 1025, device=DEVICE, dtype=torch.bfloat16)
        self._check(a, b, ("a", "b"))

    def test_odd_k_stages_a_only(self) -> None:
        a = torch.randn(512, 1025, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(1025, 512, device=DEVICE, dtype=torch.bfloat16)
        self._check(a, b, ("a",))

    def test_transposed_odd_operand(self) -> None:
        a = torch.randn(1025, 512, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(512, 1025, device=DEVICE, dtype=torch.bfloat16)
        # ``a.T`` ([512, 1025], strides (1, 512)) has 1024-byte outer strides
        # and stays in place; ``b.T`` ([1025, 512], strides (1, 1025)) has
        # 2050-byte outer strides and is relaid out row-major.
        self._check(a.T, b.T, ("b",))

    def test_aligned_operands_untouched(self) -> None:
        a = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        self._check(a, a, ())


if __name__ == "__main__":
    unittest.main()
