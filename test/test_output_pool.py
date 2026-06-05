"""Tests for the opt-in output-tensor pool.

The pool is gated on ``HELION_OUTPUT_POOL=1`` (or programmatically via
``helion.runtime._output_pool._set_pool_enabled``). Default behavior
is a pass-through to ``torch.empty_like``, so existing kernels that
build their output with the standard PyTorch builtin are completely
unaffected.
"""

from __future__ import annotations

import os
import unittest

import torch

from helion.runtime._output_pool import _clear_pool as clear_pool
from helion.runtime._output_pool import _empty_like as empty_like
from helion.runtime._output_pool import _is_pool_enabled as is_pool_enabled
from helion.runtime._output_pool import _set_pool_enabled as set_pool_enabled


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
    """When the pool is enabled, repeat calls for the same
    ``(dtype, shape, device)`` key return the SAME cached buffer."""

    def setUp(self) -> None:
        self.addCleanup(clear_pool)
        self._old_flag = is_pool_enabled()
        set_pool_enabled(True)
        clear_pool()

    def tearDown(self) -> None:
        set_pool_enabled(self._old_flag)

    def test_same_key_returns_same_tensor(self) -> None:
        """Two calls with the same template key return the same Python
        tensor object (and therefore the same storage)."""
        x = torch.randn(64, dtype=torch.float32)
        a = empty_like(x)
        b = empty_like(x)
        self.assertIs(a, b)
        self.assertEqual(a.data_ptr(), b.data_ptr())

    def test_distinct_shapes_get_distinct_buffers(self) -> None:
        """A different shape must NOT collide with an existing cache
        entry — it gets its own buffer."""
        x1 = torch.randn(64, dtype=torch.float32)
        x2 = torch.randn(128, dtype=torch.float32)
        a = empty_like(x1)
        b = empty_like(x2)
        self.assertEqual(a.shape, x1.shape)
        self.assertEqual(b.shape, x2.shape)
        self.assertNotEqual(a.data_ptr(), b.data_ptr())

    def test_distinct_dtypes_get_distinct_buffers(self) -> None:
        """Same shape but different dtype → separate cache entries,
        no aliasing."""
        x1 = torch.randn(64, dtype=torch.float32)
        x2 = torch.randn(64, dtype=torch.bfloat16)
        a = empty_like(x1)
        b = empty_like(x2)
        self.assertEqual(a.dtype, x1.dtype)
        self.assertEqual(b.dtype, x2.dtype)
        self.assertNotEqual(a.data_ptr(), b.data_ptr())

    def test_distinct_slots_get_distinct_buffers(self) -> None:
        """Same ``(dtype, shape, device)`` but different ``_slot`` →
        distinct cached buffers. Disambiguates multiple kernel-output
        allocations in the same generated wrapper."""
        x = torch.randn(64, dtype=torch.float32)
        a = empty_like(x, _slot=0)
        b = empty_like(x, _slot=1)
        # Distinct tensor objects, distinct storage.
        self.assertIsNot(a, b)
        self.assertNotEqual(a.data_ptr(), b.data_ptr())
        # And idempotent per-slot.
        self.assertIs(empty_like(x, _slot=0), a)
        self.assertIs(empty_like(x, _slot=1), b)

    def test_clear_pool_releases_buffers(self) -> None:
        """``clear_pool`` should drop refs so the next call allocates
        fresh, not recycle."""
        x = torch.randn(64, dtype=torch.float32)
        first = empty_like(x)
        clear_pool()
        from helion.runtime._output_pool import _cache

        self.assertEqual(len(_cache()), 0)
        second = empty_like(x)
        # New cache entry after clear; the returned tensor is a fresh
        # allocation (not the same Python object as ``first``).
        self.assertIsNot(first, second)
        self.assertEqual(len(_cache()), 1)


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
