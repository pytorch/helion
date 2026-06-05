"""Tests for autotune-only output-pool enable.

The codegen rewrite is unconditional now (every Helion wrapper calls
``_helion_pool_empty_like`` for output allocations), so the SAME
compiled wrapper is used in autotune mode (pool ON) and production
(pool OFF). The autotuner flips the pool flag around its per-config
benchmark and restores afterwards; the pool helper checks the flag at
each call.
"""

from __future__ import annotations

import unittest

import torch

from helion.runtime import _output_pool as _pool_mod


class TestPoolAutotuneEnable(unittest.TestCase):
    def setUp(self) -> None:
        # Save and force pool off so each test starts clean.
        self._prev = _pool_mod._is_pool_enabled()
        _pool_mod._set_pool_enabled(False)
        _pool_mod._clear_pool()

    def tearDown(self) -> None:
        _pool_mod._set_pool_enabled(self._prev)
        _pool_mod._clear_pool()

    def test_default_is_off(self) -> None:
        """Pool flag defaults to off so production calls hit the
        standard allocator."""
        self.assertFalse(_pool_mod._is_pool_enabled())

    def test_helper_passthrough_when_off(self) -> None:
        """``empty_like`` returns a fresh tensor (no pool) when the
        flag is off."""
        x = torch.randn(64)
        a = _pool_mod._empty_like(x)
        b = _pool_mod._empty_like(x)
        # No pool -> distinct allocations.
        self.assertIsNot(a, b)
        self.assertNotEqual(a.data_ptr(), b.data_ptr())

    def test_helper_recycles_when_on(self) -> None:
        """When the pool is on, repeat calls for the same
        ``(dtype, shape, device)`` triple return the SAME cached
        tensor object."""
        _pool_mod._set_pool_enabled(True)
        try:
            x = torch.randn(64)
            a = _pool_mod._empty_like(x)
            b = _pool_mod._empty_like(x)
            self.assertIs(a, b)
            self.assertEqual(a.data_ptr(), b.data_ptr())
        finally:
            _pool_mod._set_pool_enabled(False)

    def test_enable_pool_context_manager(self) -> None:
        """``enable_pool()`` flips the flag inside the ``with`` block
        and clears the cache + disables the pool on exit. Mirrors how
        ``benchmark_provider._benchmark_function`` scopes the pool to
        its per-config bench loop."""
        observed = []
        _pool_mod._set_pool_enabled(False)
        observed.append(("before", _pool_mod._is_pool_enabled()))
        with _pool_mod._enable_pool():
            observed.append(("during", _pool_mod._is_pool_enabled()))
            # Populate the cache so we can verify it's cleared on exit.
            _pool_mod._empty_like(torch.randn(8))
            self.assertEqual(len(_pool_mod._cache()), 1)
        observed.append(("after", _pool_mod._is_pool_enabled()))
        self.assertEqual(
            observed,
            [("before", False), ("during", True), ("after", False)],
        )
        # Cache cleared on exit so the next caller starts clean.
        self.assertEqual(len(_pool_mod._cache()), 0)

    def test_enable_pool_not_reentrant(self) -> None:
        """``enable_pool()`` raises if the pool is already enabled on
        the calling thread. Nesting would silently clobber the outer
        scope's cached buffers via the exit-time clear, so we fail
        loudly instead."""
        _pool_mod._set_pool_enabled(True)
        try:
            with self.assertRaises(RuntimeError), _pool_mod._enable_pool():
                pass
        finally:
            _pool_mod._set_pool_enabled(False)
            _pool_mod._clear_pool()


if __name__ == "__main__":
    unittest.main()
