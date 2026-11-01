from __future__ import annotations

import contextlib
import io
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
import helion.language as hl


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    n, d = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        row = x[tile_n, :].to(torch.float32)
        scale = torch.sqrt(torch.mean(row * row, dim=-1) + 1e-5)
        out[tile_n, :] = (torch.exp(row / scale[:, None])).to(out.dtype)
    return out


def _expected(x: torch.Tensor) -> torch.Tensor:
    row = x.float()
    scale = torch.sqrt(torch.mean(row * row, dim=-1, keepdim=True) + 1e-5)
    return torch.exp(row / scale).to(x.dtype)


def _stderr_of(kernel: object, x: torch.Tensor) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        _, out = code_and_output(kernel, (x,))  # pyrefly: ignore[bad-argument-type]
    torch.testing.assert_close(out, _expected(x), atol=1e-2, rtol=1e-2)
    return buffer.getvalue()


def _row_max(x: torch.Tensor) -> torch.Tensor:
    n, d = x.size()
    out = x.new_empty([n])
    for tile_n in hl.tile(n):
        out[tile_n] = torch.amax(x[tile_n, :], dim=-1)
    return out


@onlyBackends(["triton", "cute"])
@skipIfTileIR(
    "TileIR keeps the exact forms under fast_math; the warning is not emitted"
)
class TestFastMathWarning(TestCase):
    @skipIfRefEager("Warning emitted in compile mode only")
    def test_max_reduction_counts_on_triton_only(self) -> None:
        # Under fast_math Triton lowers amax/amin and maximum/minimum without
        # NaN propagation; the CuTe backend keeps the propagating forms, so
        # only Triton kernels are told about them.
        from helion.runtime.settings import _get_backend

        x = torch.randn([64, 128], device=DEVICE)
        buffer = io.StringIO()
        with contextlib.redirect_stderr(buffer):
            _, out = code_and_output(helion.kernel(_row_max, static_shapes=True), (x,))
        torch.testing.assert_close(out, x.amax(-1))
        text = buffer.getvalue()
        if _get_backend() == "triton":
            self.assertIn("WARNING[FastMathUnused]", text)
            self.assertIn("max/min reductions", text)
        else:
            self.assertNotIn("FastMathUnused", text)

    @skipIfRefEager("Warning emitted in compile mode only")
    def test_division_and_exp_without_fast_math_warn(self) -> None:
        x = torch.randn([64, 128], device=DEVICE)
        text = _stderr_of(helion.kernel(_normalize_rows, static_shapes=True), x)
        self.assertIn("WARNING[FastMathUnused]", text)
        self.assertIn("exp, fp32 division", text)
        self.assertIn("fast_math=True", text)

    @skipIfRefEager("Warning emitted in compile mode only")
    def test_fast_math_kernel_does_not_warn(self) -> None:
        x = torch.randn([64, 128], device=DEVICE)
        text = _stderr_of(
            helion.kernel(_normalize_rows, static_shapes=True, fast_math=True), x
        )
        self.assertNotIn("FastMathUnused", text)

    @skipIfRefEager("Warning emitted in compile mode only")
    def test_warning_can_be_ignored(self) -> None:
        x = torch.randn([64, 128], device=DEVICE)
        text = _stderr_of(
            helion.kernel(
                _normalize_rows,
                static_shapes=True,
                ignore_warnings=[helion.exc.FastMathUnused],
            ),
            x,
        )
        self.assertNotIn("FastMathUnused", text)


if __name__ == "__main__":
    unittest.main()
