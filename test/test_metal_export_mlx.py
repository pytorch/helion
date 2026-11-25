"""Captured Metal launches run under MLX's ``mx.fast.metal_kernel``
(``MetalLaunch.to_mlx``) reproduce Helion's own launches, bit for bit but for
c10's erf: an elementwise kernel, one with a baked scalar, one with a 0-dim
input, one with torch's c10 headers, one with a reduction's threadgroup
scratch, one that writes part of an output the host zeroes, and an MPP
matmul over device views.  Needs an Apple GPU and mlx."""

from __future__ import annotations

import importlib.util
import sys
from typing import TYPE_CHECKING
from typing import Literal
import unittest

import numpy as np
import pytest
import torch

from helion._testing import skipUnlessBackends
from helion._testing.metal import addmm_k_loop
from helion._testing.metal import mpp_only
from helion._testing.metal import scale_by_0dim
from helion._testing.metal_export import add_one
from helion._testing.metal_export import position_named
from helion._testing.metal_export import row_sum
from helion._testing.metal_export import scatter_even
from helion._testing.metal_export import shift
from helion._testing.metal_preamble import erf_of
from helion.runtime import capture_metal_launches

if TYPE_CHECKING:
    import helion

DEVICE = "mps"

# Metal kernels on macOS only: other backends' suites skip the module.
pytestmark = [
    pytest.mark.skipif(sys.platform != "darwin", reason="Metal tests require macOS"),
    skipUnlessBackends(["metal"]),
]
_requires_mps_and_mlx = unittest.skipUnless(
    sys.platform == "darwin"
    and torch.backends.mps.is_available()
    and importlib.util.find_spec("mlx") is not None,
    "requires a Metal GPU and mlx",
)


@_requires_mps_and_mlx
class TestRunUnderMLX(unittest.TestCase):
    def _check(
        self,
        kernel: helion.Kernel,
        args: tuple[object, ...],
        config: helion.Config | None = None,
        exact: bool = True,
        init_value: float | Literal["captured"] | None = None,
    ) -> None:
        import mlx.core as mx

        def to_mx(t: torch.Tensor) -> mx.array:
            return mx.array(t.detach().cpu().numpy())

        (launch,) = capture_metal_launches(kernel, args, config)
        k = launch.to_mlx(init_value=init_value)
        fn = mx.fast.metal_kernel(
            name=k.name,
            input_names=list(k.input_names),
            output_names=list(k.output_names),
            source=k.source,
            header=k.header,
        )
        (out,) = fn(
            inputs=[to_mx(t) for t in k.inputs],
            grid=k.grid,
            threadgroup=k.threadgroup,
            output_shapes=[list(s) for s in k.output_shapes],
            output_dtypes=[mx.float32 for _ in k.output_dtypes],
            init_value=k.init_value,
        )
        expected = kernel.bind(args).compile_config(config)(*args)
        torch.mps.synchronize()
        got = torch.from_numpy(np.array(out))
        if exact:
            self.assertTrue(torch.equal(got, expected.cpu()))
        else:
            torch.testing.assert_close(got, expected.cpu())

    def test_elementwise(self) -> None:
        self._check(add_one, (torch.randn(40, 72, device=DEVICE),))

    def test_baked_scalar(self) -> None:
        self._check(shift, (torch.randn(1000, device=DEVICE), 3.0))

    def test_0dim_input(self) -> None:
        x = torch.randn(1000, device=DEVICE)
        self._check(scale_by_0dim, (x, torch.tensor(0.5, device=DEVICE)))

    def test_c10_headers(self) -> None:
        """erf, which Helion carries no copy of, takes torch's c10 headers,
        inlined into the header.  Compiled by MLX, c10's erf differs from
        torch's compile in the last bit for a few inputs (27 of 1000 here),
        so this one compares within float32's tolerance."""
        x = torch.randn(1000, device=DEVICE)
        config = erf_of.bind((x,)).config_spec.default_config()
        self._check(erf_of, (x,), config, exact=False)

    def test_threadgroup_scratch(self) -> None:
        (launch,) = capture_metal_launches(row_sum, (torch.randn(16, 256),))
        self.assertTrue(launch.msl.scratch)
        self._check(row_sum, (torch.randn(16, 256, device=DEVICE),))

    def test_partly_written_output(self) -> None:
        """The host zeroes the output, and the kernel writes half of it."""
        x = torch.randn(40, device=DEVICE)
        self._check(scatter_even, (x,), init_value="captured")

    def test_input_named_like_a_position_attribute(self) -> None:
        """An input MLX would declare under the same name as its position
        attribute runs under a name of its own."""
        x = torch.arange(64, dtype=torch.float32, device=DEVICE)
        self._check(position_named, (x,))

    def test_mpp_matmul(self) -> None:
        args = (
            torch.randn(64, 128, device=DEVICE),
            torch.randn(128, 64, device=DEVICE),
        )
        self._check(addmm_k_loop, args, mpp_only(addmm_k_loop, args, None))


if __name__ == "__main__":
    unittest.main()
