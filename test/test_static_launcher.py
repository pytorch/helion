"""Static-kernel-launcher specific tests.

The launcher (``helion.runtime.StaticLauncher``) builds PyTorch
Inductor's ``StaticallyLaunchedCudaKernel`` on first call and
dispatches every subsequent call straight to ``cuLaunchKernel`` via
``torch._C._StaticCudaLauncher``. Falls back to ``default_launcher``
on any priming failure.

These tests pin down behavior unique to the static path: correctness
match against the default path, internal priming state, both env-var
escape hatches (``HELION_SKIP_STATIC_LAUNCHER`` and
``HELION_SKIP_FAST_LAUNCHER``), constexpr-arg filtering before the
C-level dispatch, and the cross-device fall-back. Launcher-agnostic
behaviors (unaligned input / output handling and
``used_global_vals`` mutation) are covered for both launchers via
the parametrized ``TestLauncher`` class in ``test/test_misc.py``.
"""

from __future__ import annotations

import os
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfNotTriton
import helion.language as hl
from helion.runtime import StaticLauncher


@helion.kernel(
    static_shapes=True,
    config=helion.Config(block_sizes=[1024], num_warps=4, num_stages=2),
)
def _static_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for i in hl.tile(out.size(0)):
        out[i] = x[i] + y[i]
    return out


def _installed_launcher(kernel: helion.Kernel) -> StaticLauncher:
    bound = next(iter(kernel._bound_kernels.values()))  # type: ignore[attr-defined]
    return bound._run.__kwdefaults__["_launcher"]


@skipIfNotTriton("static launcher is Triton-specific")
class TestStaticLauncher(RefEagerTestDisabled, TestCase):
    def setUp(self) -> None:
        # Other test files (e.g. TestLauncher in test_misc.py) set these
        # env vars and restore them via addCleanup, but pytest-xdist
        # dispatches tests across workers and pytest-rerunfailures
        # reruns failures — both can land tests on the same worker in
        # orders that briefly expose the env var before cleanup. We
        # snapshot + clear here so each TestStaticLauncher test starts
        # in a known state regardless of what ran before.
        super().setUp()
        for name in ("HELION_SKIP_FAST_LAUNCHER", "HELION_SKIP_STATIC_LAUNCHER"):
            prev = os.environ.pop(name, None)
            if prev is not None:
                self.addCleanup(os.environ.__setitem__, name, prev)

    @skipIfNotCUDA()
    def test_result_matches_default_path(self) -> None:
        """Output from the static-launcher path matches the
        ``default_launcher`` result on the same inputs."""
        _static_add.reset()
        x = torch.randn(4096, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4096, device=DEVICE, dtype=torch.float32)
        expected = (x + y).clone()
        result = _static_add(x, y)
        torch.cuda.synchronize()
        torch.testing.assert_close(result, expected)

    @skipIfNotCUDA()
    def test_static_kernel_attached_after_first_call(self) -> None:
        """First call primes the launcher; the static kernel is
        attached if PyTorch + Triton support it. (No-op on older
        environments — we just check the launcher exposes its state.)
        """
        _static_add.reset()
        x = torch.randn(4096, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4096, device=DEVICE, dtype=torch.float32)
        _static_add(x, y)
        torch.cuda.synchronize()
        launcher = _installed_launcher(_static_add)
        self.assertIsInstance(launcher, StaticLauncher)
        self.assertTrue(launcher._primed)

    @skipIfNotCUDA()
    def test_skip_env_var_uses_default_launcher(self) -> None:
        """``HELION_SKIP_STATIC_LAUNCHER=1`` set BEFORE priming routes
        every call through ``default_launcher``. Result is still
        correct."""
        prev = os.environ.get("HELION_SKIP_STATIC_LAUNCHER")
        os.environ["HELION_SKIP_STATIC_LAUNCHER"] = "1"
        try:
            _static_add.reset()
            x = torch.randn(4096, device=DEVICE, dtype=torch.float32)
            y = torch.randn(4096, device=DEVICE, dtype=torch.float32)
            expected = (x + y).clone()
            result = _static_add(x, y)
            torch.cuda.synchronize()
            torch.testing.assert_close(result, expected)
            launcher = _installed_launcher(_static_add)
            self.assertIsNone(launcher._static_kernel)
        finally:
            if prev is None:
                os.environ.pop("HELION_SKIP_STATIC_LAUNCHER", None)
            else:
                os.environ["HELION_SKIP_STATIC_LAUNCHER"] = prev

    @skipIfNotCUDA()
    def test_constexpr_args_filtered(self) -> None:
        """A kernel that takes a constexpr arg (block size etc.) must
        still produce correct output through the static launcher —
        the launcher's ``_keep_indices`` filters constexpr positions
        out of ``*args`` before the C-level dispatch."""

        @helion.kernel(
            static_shapes=True,
            config=helion.Config(block_sizes=[64, 32], num_warps=4, num_stages=2),
        )
        def _sum_rows(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty(m, device=x.device, dtype=x.dtype)
            for tile_m in hl.tile(m):
                acc = torch.zeros_like(out[tile_m])
                for tile_n in hl.tile(n):
                    acc = acc + x[tile_m, tile_n].sum(-1)
                out[tile_m] = acc
            return out

        x = torch.randn(128, 256, device=DEVICE, dtype=torch.float32)
        expected = x.sum(-1).clone()
        result = _sum_rows(x)
        torch.cuda.synchronize()
        torch.testing.assert_close(result, expected)

    @skipIfNotCUDA()
    def test_skip_fast_launcher_env_var_disables_install(self) -> None:
        """``HELION_SKIP_FAST_LAUNCHER=1`` (the upstream escape hatch)
        leaves the wrapper's ``_launcher`` kwdefault at the
        ``default_launcher`` module-level function — no static
        launcher is installed at all."""
        from helion.runtime import default_launcher

        prev = os.environ.get("HELION_SKIP_FAST_LAUNCHER")
        os.environ["HELION_SKIP_FAST_LAUNCHER"] = "1"
        try:
            _static_add.reset()
            x = torch.randn(64, device=DEVICE, dtype=torch.float32)
            y = torch.randn(64, device=DEVICE, dtype=torch.float32)
            expected = (x + y).clone()
            result = _static_add(x, y)
            torch.cuda.synchronize()
            torch.testing.assert_close(result, expected)
            launcher = _installed_launcher(_static_add)
            self.assertIs(launcher, default_launcher)
        finally:
            if prev is None:
                os.environ.pop("HELION_SKIP_FAST_LAUNCHER", None)
            else:
                os.environ["HELION_SKIP_FAST_LAUNCHER"] = prev

    @skipIfNotCUDA()
    def test_separate_bound_kernels_per_cuda_device(self) -> None:
        """The bound-kernel cache key includes ``device.index``, so
        ``cuda:0`` and ``cuda:1`` get **distinct** ``BoundKernel``
        cache entries — each with its own ``StaticLauncher`` pinned to
        the right device. Verifies the cache shape; per-device numeric
        correctness is covered by ``TestLauncher`` in ``test_misc.py``."""
        if torch.cuda.device_count() < 2:
            self.skipTest("requires >= 2 CUDA devices")
        _static_add.reset()

        device0 = torch.device(DEVICE.type, 0)
        device1 = torch.device(DEVICE.type, 1)

        x0 = torch.randn(1024, device=device0, dtype=torch.float32)
        with torch.cuda.device(device0.index):
            _static_add(x0, x0)
        torch.cuda.synchronize(0)

        x1 = torch.randn(1024, device=device1, dtype=torch.float32)
        with torch.cuda.device(device1.index):
            _static_add(x1, x1)
        torch.cuda.synchronize(1)

        bound_kernels = list(_static_add._bound_kernels.values())  # type: ignore[attr-defined]
        self.assertEqual(len(bound_kernels), 2)
        per_device_index = {
            bk._env.device.index: bk._run.__kwdefaults__["_launcher"]._device_index
            for bk in bound_kernels
        }
        self.assertEqual(per_device_index, {0: 0, 1: 1})


if __name__ == "__main__":
    unittest.main()
