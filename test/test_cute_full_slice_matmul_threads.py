"""Thread plans for the CuTe scalar fallback of a full-slice matmul.

``out[tile, :] = x[tile, :] @ w[:, :]`` reduces K across the threads of K's
reduction strategy.  A flattened row tile leaves the contraction its full
thread extent (as an N-D tile does) and takes a lane loop itself, so at the
default block size (32 rows) a K of 48 or more stays on threads instead of
splitting into synthetic lanes.  A K past the 1024-thread cap folds over the
whole K through the synthetic-lane fold, with the contraction at any position
of the operand (``w[:, :]`` or ``wt[:, :].T``).  A matmul that contracts the
flattened tile's own rows cannot use that tile's lane loop and is refused
rather than keeping one lane's products.
"""

from __future__ import annotations

import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@helion.kernel(backend="cute", static_shapes=True)
def _full_slice_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), w.size(1)], device=x.device)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :] = x[tile_m, :] @ w[:, :]
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _full_slice_matmul_transposed(x: torch.Tensor, wt: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0), wt.size(0)], device=x.device)
    for tile_m in hl.tile(x.size(0)):
        out[tile_m, :] = x[tile_m, :] @ wt[:, :].T
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _row_contraction(
    x: torch.Tensor, g: torch.Tensor, parts: torch.Tensor
) -> torch.Tensor:
    for tile_m in hl.tile(x.size(0)):
        parts[tile_m.id, :, :] = x[tile_m, :].T @ g[tile_m, :]
    return parts


@onlyBackends(["cute"])
class TestCuteFullSliceMatmulThreads(TestCase):
    def test_default_config_keeps_k_on_threads(self) -> None:
        for m, k, n in ((64, 48, 32), (64, 64, 32), (37, 40, 24), (128, 128, 64)):
            torch.manual_seed(0)
            x = torch.randn(m, k, device=DEVICE)
            w = torch.randn(k, n, device=DEVICE)
            for config in ({}, {"block_sizes": [32]}, {"block_sizes": [64]}):
                with self.subTest(shape=(m, k, n), config=config):
                    code, out = code_and_output(_full_slice_matmul, (x, w), **config)
                    torch.testing.assert_close(out, x @ w, rtol=1e-4, atol=1e-4)
                    self.assertNotIn("synthetic_lane_1", code)
                    self.assertNotIn("mm_fold", code)

    def test_k_past_the_thread_cap_folds_either_operand_layout(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(64, 2048, device=DEVICE)
        w = torch.randn(2048, 16, device=DEVICE)
        for fn, rhs in (
            (_full_slice_matmul, w),
            (_full_slice_matmul_transposed, w.T.contiguous()),
        ):
            with self.subTest(fn=fn.name):
                code, out = code_and_output(fn, (x, rhs), block_sizes=[32])
                torch.testing.assert_close(out, x @ w, rtol=1e-3, atol=1e-3)
                self.assertIn("mm_fold_k", code)
                # The fold covers the whole K per output element, so K runs
                # on one thread and the 32 rows and 16 columns keep theirs.
                self.assertIn("block=(16, 32, 1)", code)

    def test_contraction_over_lane_looped_tile_rows_is_refused(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(64, 48, device=DEVICE)
        g = torch.randn(64, 32, device=DEVICE)
        expected = torch.stack([x[:32].T @ g[:32], x[32:].T @ g[32:]])
        _code, parts = code_and_output(
            _row_contraction,
            (x, g, torch.zeros(2, 48, 32, device=DEVICE)),
            block_sizes=[32],
        )
        torch.testing.assert_close(parts, expected, rtol=1e-4, atol=1e-4)
        with self.assertRaisesRegex(exc.BackendUnsupported, "split across a lane loop"):
            code_and_output(
                _row_contraction,
                (x, g, torch.zeros(2, 48, 32, device=DEVICE)),
                block_sizes=[32],
                num_threads=[16, 0, 0],
            )
