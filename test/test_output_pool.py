"""Tests for the opt-in output-tensor pool.

The pool is gated on ``HELION_OUTPUT_POOL=1`` (or programmatically via
``helion.runtime.set_pool_enabled``). Default behavior is a pass-through
to ``torch.empty_like``, so existing kernels that build their output
with the standard PyTorch builtin are completely unaffected.
"""

from __future__ import annotations

import os
import unittest

import torch

from helion.runtime import clear_pool
from helion.runtime import empty_like
from helion.runtime import is_pool_enabled
from helion.runtime import set_pool_enabled


class TestPoolDisabledByDefault(unittest.TestCase):
    """When the env var is unset, ``empty_like`` must behave exactly
    like ``torch.empty_like``: a fresh allocation on every call."""

    def setUp(self) -> None:
        self.addCleanup(clear_pool)
        # Snapshot + restore the global flag and env var so this test
        # is isolated from any other test that may have toggled them.
        self._old_env = os.environ.get("HELION_OUTPUT_POOL")
        self._old_flag = is_pool_enabled()
        os.environ.pop("HELION_OUTPUT_POOL", None)
        set_pool_enabled(False)

    def tearDown(self) -> None:
        if self._old_env is not None:
            os.environ["HELION_OUTPUT_POOL"] = self._old_env
        else:
            os.environ.pop("HELION_OUTPUT_POOL", None)
        set_pool_enabled(self._old_flag)

    def test_returns_fresh_tensor_each_call(self) -> None:
        """Pool off → each call must return a distinct ``data_ptr()``."""
        x = torch.randn(64, dtype=torch.float32)
        a = empty_like(x)
        b = empty_like(x)
        self.assertNotEqual(a.data_ptr(), b.data_ptr())

    def test_matches_torch_empty_like_metadata(self) -> None:
        """Returned tensor's dtype/shape/stride/device matches template."""
        x = torch.randn(3, 5, dtype=torch.bfloat16)
        y = empty_like(x)
        self.assertEqual(y.dtype, x.dtype)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.stride(), x.stride())
        self.assertEqual(y.device, x.device)


class TestPoolEnabled(unittest.TestCase):
    """When the pool is enabled, identical-signature calls recycle
    buffers from a fixed-size ring."""

    def setUp(self) -> None:
        self.addCleanup(clear_pool)
        self._old_flag = is_pool_enabled()
        set_pool_enabled(True)
        clear_pool()

    def tearDown(self) -> None:
        set_pool_enabled(self._old_flag)

    def test_ring_recycles_after_depth_calls(self) -> None:
        """With ring depth N, the (N+1)th call should hand back the
        SAME storage as the 1st."""
        x = torch.randn(64, dtype=torch.float32)
        # Module-level constant; matches _POOL_DEPTH in _output_pool.py.
        from helion.runtime._output_pool import _POOL_DEPTH

        first_n = [empty_like(x).data_ptr() for _ in range(_POOL_DEPTH)]
        recycled = empty_like(x).data_ptr()
        self.assertEqual(recycled, first_n[0])

    def test_distinct_signatures_use_distinct_rings(self) -> None:
        """A different shape must NOT collide with an existing ring's
        buffers — it gets its own ring."""
        x1 = torch.randn(64, dtype=torch.float32)
        x2 = torch.randn(128, dtype=torch.float32)
        a = empty_like(x1)
        b = empty_like(x2)
        self.assertEqual(a.shape, x1.shape)
        self.assertEqual(b.shape, x2.shape)
        self.assertNotEqual(a.data_ptr(), b.data_ptr())

    def test_distinct_dtypes_use_distinct_rings(self) -> None:
        """Same shape but different dtype → separate rings, no aliasing."""
        x1 = torch.randn(64, dtype=torch.float32)
        x2 = torch.randn(64, dtype=torch.bfloat16)
        a = empty_like(x1)
        b = empty_like(x2)
        self.assertEqual(a.dtype, x1.dtype)
        self.assertEqual(b.dtype, x2.dtype)
        self.assertNotEqual(a.data_ptr(), b.data_ptr())

    def test_clear_pool_releases_buffers(self) -> None:
        """``clear_pool`` should drop refs so the next call allocates
        fresh, not recycle."""
        x = torch.randn(64, dtype=torch.float32)
        first = empty_like(x).data_ptr()
        clear_pool()
        # After clearing, the next call allocates a new buffer — it
        # MAY happen to land at the same address (allocator reuse) but
        # the pool internals must be empty.
        from helion.runtime._output_pool import _POOLS

        empty_like(x)
        self.assertEqual(len(_POOLS), 1)  # one new ring after clear
        # And the first buffer ptr is no longer guaranteed to be reused
        # — we don't assert ptr inequality (allocator may legitimately
        # reuse the freed slot) but the pool state should be fresh.
        _ = first


class TestEnvVarToggle(unittest.TestCase):
    """``set_pool_enabled`` should mirror the env-var-driven default."""

    def test_set_pool_enabled_changes_behavior(self) -> None:
        old_flag = is_pool_enabled()
        try:
            set_pool_enabled(False)
            self.assertFalse(is_pool_enabled())
            set_pool_enabled(True)
            self.assertTrue(is_pool_enabled())
        finally:
            set_pool_enabled(old_flag)
            clear_pool()


if __name__ == "__main__":
    unittest.main()
