from __future__ import annotations

import contextlib
import io
import unittest

import pytest
import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import TestCase
import helion.language as hl


class TestPrintOutputCode(TestCase):
    def test_ref_eager_mode_code_print_error(self):
        """Test that RefEagerModeCodePrintError is raised when using @helion.kernel with both settings"""

        with pytest.raises(exc.RefEagerModeCodePrintError):

            @helion.kernel(
                autotune_effort="none",
                print_output_code=True,
                ref_mode=helion.RefMode.EAGER,
            )
            def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                x, y = torch.broadcast_tensors(x, y)
                out = torch.empty(
                    x.shape,
                    dtype=torch.promote_types(x.dtype, y.dtype),
                    device=x.device,
                )
                for tile in hl.tile(out.size()):
                    out[tile] = x[tile] + y[tile]
                return out

            x = torch.randn([512, 512], device=DEVICE, dtype=torch.float16)
            y = torch.randn([512, 512], device=DEVICE, dtype=torch.float16)
            torch.testing.assert_close(add(x, y), torch.add(x, y))

    def test_normal_mode_code_print(self):
        """Test that output code is in stderr when using @helion.kernel with normal mode"""

        f = io.StringIO()
        with contextlib.redirect_stderr(f):

            @helion.kernel(
                autotune_effort="none",
                print_output_code=True,
                ref_mode=helion.RefMode.OFF,
            )
            def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                x, y = torch.broadcast_tensors(x, y)
                out = torch.empty(
                    x.shape,
                    dtype=torch.promote_types(x.dtype, y.dtype),
                    device=x.device,
                )
                for tile in hl.tile(out.size()):
                    out[tile] = x[tile] + y[tile]
                return out

            x = torch.randn([512, 512], device=DEVICE, dtype=torch.float16)
            y = torch.randn([512, 512], device=DEVICE, dtype=torch.float16)
            torch.testing.assert_close(add(x, y), torch.add(x, y))

        self.assertNotEqual(
            f.getvalue(),
            "",
            "Output code in stderr should not be empty at normal mode.",
        )


if __name__ == "__main__":
    unittest.main()
