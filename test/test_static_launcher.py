from __future__ import annotations

import unittest
from unittest import mock

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
import helion.language as hl
from helion.runtime import _check_static_launcher_available


def _make_add_kernel():
    @helion.kernel(static_shapes=True, autotune_effort="none")
    def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = torch.broadcast_tensors(x, y)
        out = torch.empty_like(x)
        for tile in hl.tile(out.size()):
            out[tile] = x[tile] + y[tile]
        return out

    return add_kernel


@unittest.skipUnless(
    _check_static_launcher_available(),
    "static launcher not available (missing torch._inductor.runtime.static_triton_launcher)",
)
class TestStaticLauncher(TestCase):
    def test_basic(self):
        """Verify the static launcher produces correct results."""
        add_kernel = _make_add_kernel()
        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        y = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        torch.testing.assert_close(add_kernel(x, y), x + y)
        # Second call exercises the cached hot path
        x2 = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        y2 = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        torch.testing.assert_close(add_kernel(x2, y2), x2 + y2)

    def test_static_launch_is_called(self):
        """Verify the static launcher (not Triton's default) is used."""
        from torch._inductor.runtime.static_triton_launcher import (
            StaticallyLaunchedTritonKernel,
        )
        from torch._inductor.runtime.static_triton_launcher import (
            statically_launched_kernel_by_device,
        )

        add_kernel = _make_add_kernel()
        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        y = torch.randn(1024, device=DEVICE, dtype=torch.float32)

        created_launchers = []
        original = statically_launched_kernel_by_device

        def tracking_factory(*args, **kwargs):
            launcher = original(*args, **kwargs)
            created_launchers.append(launcher)
            return launcher

        with mock.patch(
            "torch._inductor.runtime.static_triton_launcher.statically_launched_kernel_by_device",
            side_effect=tracking_factory,
        ):
            # First call creates the static launcher
            result1 = add_kernel(x, y)
            torch.testing.assert_close(result1, x + y)
            self.assertEqual(len(created_launchers), 1)
            self.assertIsInstance(created_launchers[0], StaticallyLaunchedTritonKernel)

            # Second call reuses it (no new launcher created)
            x2 = torch.randn(1024, device=DEVICE, dtype=torch.float32)
            y2 = torch.randn(1024, device=DEVICE, dtype=torch.float32)
            result2 = add_kernel(x2, y2)
            torch.testing.assert_close(result2, x2 + y2)
            self.assertEqual(len(created_launchers), 1)

    def test_disable_static_launcher(self):
        """Verify that _static_launch is NOT called when disabled via env var."""
        add_kernel = _make_add_kernel()
        x = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        y = torch.randn(1024, device=DEVICE, dtype=torch.float32)
        with mock.patch.dict("os.environ", {"HELION_STATIC_LAUNCHER": "0"}):
            # Clear the cached result so the env var takes effect
            _check_static_launcher_available.cache_clear()
            try:
                with mock.patch("helion.runtime._static_launch") as mocked:
                    result = add_kernel(x, y)
                    mocked.assert_not_called()
                torch.testing.assert_close(result, x + y)
            finally:
                _check_static_launcher_available.cache_clear()


if __name__ == "__main__":
    unittest.main()
