"""GPU tests of the CuTe unaligned-operand staging pass.

A GEMM over odd-sized rows (``K = 1025`` bf16: 2050-byte strides) fails the
TensorMap proof; with the staging pass the host pads the operands and the
tcgen05 default config keeps its TMA producers, and the result matches torch.
"""

from __future__ import annotations

import unittest

import torch

import helion
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing.cute_cp_async_a import gemm_into
from helion._testing.cute_cp_async_a import residual_gemm
from helion._testing.cute_cp_async_a import two_output_gemm
import helion.language as hl

# Both classes check tcgen05 kernels (the TMA producers, the cp.async copies).
_skip_without_tcgen05 = unittest.skipUnless(
    torch.cuda.is_available() and get_cute_mma_support().tcgen05_f16bf16,
    "needs tcgen05 (sm100)",
)


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
@_skip_without_tcgen05
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


_residual_gemm = helion.kernel(residual_gemm, static_shapes=True)
_gemm_into = helion.kernel(gemm_into, static_shapes=True)
_two_output_gemm = helion.kernel(two_output_gemm, static_shapes=True)


@onlyBackends(["cute"])
@skipIfNotCUDA()
@_skip_without_tcgen05
class TestCuteCpAsyncOperandGpu(RefEagerTestDisabled, TestCase):
    """An M-contiguous A no TensorMap can address, copied by the epilogue
    warps with ``cp.async`` (``helion/_compiler/cute/cp_async_operand.py``)."""

    def _check(self, m: int, row_stride: int, block_m: int | None) -> None:
        x = torch.randn(512, row_stride, device=DEVICE, dtype=torch.bfloat16)
        a = x[:, :m].T
        b = torch.randn(512, 128, device=DEVICE, dtype=torch.bfloat16)
        residual = torch.randn(m, 128, device=DEVICE, dtype=torch.float32)
        bound = _residual_gemm.bind((a, b, residual))
        config = bound.config_spec.default_config()
        if block_m is not None:
            config = helion.Config(
                **{
                    **config.config,
                    "block_sizes": [block_m, 128, 64],
                    "tcgen05_ab_stages": 4,
                }
            )
        code = bound.to_code(config)
        self.assertNotIn("constant_pad_nd", code)
        self.assertIn("cp_async_mbarrier_arrive_noinc", code)
        bound.set_config(config)
        torch.testing.assert_close(
            bound(a, b, residual),
            a.float() @ b.float() + residual,
            rtol=1e-3,
            atol=1e-2,
        )

    def test_row_classes_of_four(self) -> None:
        # 2036-byte K rows: 16-, 8- and 4-byte copies by row; the default
        # 128-row tile, whose last tile ends 6 rows past M.
        self._check(1018, 1018, None)

    def test_paired_m_subtiles(self) -> None:
        # The 256-row tile: the last pair's second subtile (rows 896-1023)
        # holds 122 rows inside M.
        self._check(1018, 1018, 256)

    def test_paired_m_subtile_wholly_past_m(self) -> None:
        # M=890: the last pair's second subtile (rows 896-1023) lies wholly
        # past M, as on 30522 rows.
        self._check(890, 890, 256)

    def test_row_classes_of_two_in_a_wider_row(self) -> None:
        # 2040-byte K rows (8-byte copies on odd rows) over 1018 of 1020
        # columns: the last K row's class copies element pairs.
        self._check(1018, 1020, 256)

    def test_output_no_tensormap_addresses(self) -> None:
        # 520-byte output rows take the SIMT store beside the cp.async A.
        a = torch.randn(512, 1018, device=DEVICE, dtype=torch.bfloat16).T
        b = torch.randn(512, 128, device=DEVICE, dtype=torch.bfloat16)
        out = torch.empty(1018, 130, device=DEVICE, dtype=torch.float32)[:, :128]
        bound = _gemm_into.bind((a, b, out))
        config = bound.config_spec.default_config()
        self.assertIn("cp_async_mbarrier_arrive_noinc", bound.to_code(config))
        bound.set_config(config)
        bound(a, b, out)
        torch.testing.assert_close(out, a.float() @ b.float(), rtol=1e-3, atol=1e-2)

    def test_two_outputs_keep_128_row_tiles(self) -> None:
        # Two output stores: the 128-row tile over 239 M tiles.
        a = torch.randn(512, 30522, device=DEVICE, dtype=torch.bfloat16).T
        b = torch.randn(512, 128, device=DEVICE, dtype=torch.bfloat16)
        bound = _two_output_gemm.bind((a, b))
        config = bound.config_spec.default_config()
        self.assertEqual(config.config["block_sizes"], [128, 128, 64])
        bound.set_config(config)
        out, out2 = bound(a, b)
        reference = a.float() @ b.float()
        torch.testing.assert_close(out, reference, rtol=1e-3, atol=1e-2)
        torch.testing.assert_close(out2, reference, rtol=1e-3, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
