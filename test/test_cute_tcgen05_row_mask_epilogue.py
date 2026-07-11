from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import matchesBackends
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

pytestmark = skipUnlessBackends(["cute"])
if matchesBackends(["cute"]):
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute")


def _tcgen05_row_mask_config() -> helion.Config:
    return helion.Config(
        block_sizes=[128, 128, 32],
        l2_groupings=[1],
        loop_orders=[[0, 1]],
        num_stages=2,
        num_warps=8,
        pid_type="persistent_interleaved",
        tcgen05_cluster_m=1,
        tcgen05_ab_stages=2,
        tcgen05_acc_stages=2,
        tcgen05_c_stages=2,
        tcgen05_num_epi_warps=4,
    )


@helion.kernel(backend="cute")
def _row_mask_matmul(
    x: torch.Tensor, y: torch.Tensor, row_ids: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        row_mask = row_ids[tile_m] >= 0
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = torch.where(
            row_mask[:, None],
            acc.to(out.dtype),
            torch.zeros_like(acc).to(out.dtype),
        )
    return out


@helion.kernel(backend="cute")
def _row_mask_matmul_nonzero_false(
    x: torch.Tensor, y: torch.Tensor, row_ids: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        row_mask = row_ids[tile_m] >= 0
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = torch.where(
            row_mask[:, None],
            acc.to(out.dtype),
            torch.ones_like(acc).to(out.dtype),
        )
    return out


class TestCuteTcgen05RowMaskEpilogue(TestCase):
    def _skip_unless_tcgen05(self) -> None:
        if DEVICE.type != "cuda":
            self.skipTest("tcgen05 row-mask splice requires CUDA")
        from helion._compiler.cute.mma_support import get_cute_mma_support

        with torch.cuda.device(DEVICE):
            major, _minor = torch.cuda.get_device_capability(DEVICE)
        if major < 10:
            self.skipTest("tcgen05 row-mask splice requires SM100+")
        if not get_cute_mma_support().tcgen05_f16bf16:
            self.skipTest("tcgen05 F16/BF16 MMA is not supported on this machine")

    def _args(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        x = torch.randn(128, 128, device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn(128, 128, device=DEVICE, dtype=torch.bfloat16)
        row_ids = torch.arange(128, device=DEVICE, dtype=torch.int64)
        row_ids[96:] = -1
        return x, y, row_ids

    def _bound_and_code(
        self, kernel: Kernel, args: tuple[object, ...]
    ) -> tuple[BoundKernel, str]:
        bound = kernel.bind(args)
        bound.env.config_spec.cute_tcgen05_search_enabled = True
        config = _tcgen05_row_mask_config()
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            code = bound.to_triton_code(config)
            bound.set_config(config)
        return bound, code

    def test_tcgen05_row_mask_where_splice_codegen_and_correctness(self) -> None:
        self._skip_unless_tcgen05()
        args = self._args()
        bound, code = self._bound_and_code(_row_mask_matmul, args)

        self.assertIn("tcgen05", code)
        self.assertIn("tcgen05_tma_store_atom", code)
        self.assertIn("tcgen05_tma_store_tensor", code)
        self.assertIn("tcgen05_row_mask", code)
        self.assertIn("cute.where(tcgen05_row_mask", code)
        self.assertNotIn("mask_0", code)
        self.assertNotIn("row_ids.iterator + cutlass.Int32(indices_0)", code)

        x, y, row_ids = args
        with patch.dict(os.environ, {"HELION_CUTE_MMA_IMPL": "tcgen05"}, clear=False):
            out = bound(*args)
        expected_mm = torch.mm(x, y)
        expected = torch.where(
            row_ids[:, None] >= 0,
            expected_mm,
            torch.zeros_like(expected_mm),
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, rtol=1e-1, atol=1e-1)

    def test_tcgen05_row_mask_rejects_nonzero_false_branch(self) -> None:
        self._skip_unless_tcgen05()
        args = self._args()
        with self.assertRaisesRegex(exc.BackendUnsupported, "where"):
            self._bound_and_code(_row_mask_matmul_nonzero_false, args)
