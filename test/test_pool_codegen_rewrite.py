"""Tests for the ``HELION_OUTPUT_POOL=1`` codegen rewrite.

The rewrite swaps ``torch.empty_like(x)`` in the generated host wrapper
to ``_helion_pool_empty_like(x)`` (== ``helion.runtime.empty_like``) when
the env var is set. On cache hit, the pool returns a recycled buffer
instead of allocating, saving ~0.7 μs per call.

These tests verify:
- Rewrite fires when the env var is set; absent otherwise.
- End-to-end correctness: output matches a fresh ``torch.empty_like``
  baseline regardless of which path produced the output buffer.
- Pool entries actually accumulate when the wrapper is exercised.
"""

from __future__ import annotations

import inspect
import os
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfNotTriton
import helion.language as hl


def _make_add_kernel():
    """Build a fresh kernel inside each test so the codegen runs under
    the test's current env-var state (the rewrite is decided at compile
    time, not at call time)."""

    @helion.kernel(
        static_shapes=True,
        config=helion.Config(block_sizes=[1024], num_warps=4, num_stages=2),
    )
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(out.size(0)):
            out[tile] = x[tile] + y[tile]
        return out

    return add


@skipIfNotTriton("codegen rewrite is Triton-backend-only")
class TestPoolCodegenRewrite(unittest.TestCase):
    def _toggle_env(self, value: str | None) -> None:
        prev = os.environ.get("HELION_OUTPUT_POOL")
        self.addCleanup(
            lambda: (
                os.environ.__setitem__("HELION_OUTPUT_POOL", prev)
                if prev is not None
                else os.environ.pop("HELION_OUTPUT_POOL", None)
            )
        )
        if value is None:
            os.environ.pop("HELION_OUTPUT_POOL", None)
        else:
            os.environ["HELION_OUTPUT_POOL"] = value

    @skipIfNotCUDA()
    def test_rewrite_fires_when_env_var_set(self) -> None:
        """Wrapper source must contain ``_helion_pool_empty_like`` when
        ``HELION_OUTPUT_POOL=1`` was set at compile time."""
        self._toggle_env("1")
        add = _make_add_kernel()
        x = torch.randn(4096, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4096, device=DEVICE, dtype=torch.float32)
        add(x, y)
        bound = next(iter(add._bound_kernels.values()))  # type: ignore[attr-defined]
        src = inspect.getsource(bound._run)
        # Look at the rewritten call site, not the ``# src[...]``
        # comment that preserves the user's original line for debugging.
        # The actual emitted call should be the pool helper.
        rewritten_line = [
            ln for ln in src.splitlines() if "out = " in ln and "#" not in ln
        ]
        self.assertTrue(rewritten_line, "no rewritten 'out = ...' line found")
        self.assertIn("_helion_pool_empty_like(x)", rewritten_line[0])

    @skipIfNotCUDA()
    def test_rewrite_absent_when_env_var_unset(self) -> None:
        """Without the env var, the wrapper preserves ``torch.empty_like``."""
        self._toggle_env(None)
        add = _make_add_kernel()
        x = torch.randn(4096, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4096, device=DEVICE, dtype=torch.float32)
        add(x, y)
        bound = next(iter(add._bound_kernels.values()))  # type: ignore[attr-defined]
        src = inspect.getsource(bound._run)
        rewritten_line = [
            ln for ln in src.splitlines() if "out = " in ln and "#" not in ln
        ]
        self.assertTrue(rewritten_line, "no 'out = ...' line found")
        self.assertIn("torch.empty_like(x)", rewritten_line[0])
        self.assertNotIn("_helion_pool_empty_like", rewritten_line[0])

    @skipIfNotCUDA()
    def test_end_to_end_correctness_with_rewrite(self) -> None:
        """Same numeric output regardless of which path allocates."""
        x = torch.randn(4096, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4096, device=DEVICE, dtype=torch.float32)
        expected = x + y

        # Compile with rewrite off.
        self._toggle_env(None)
        add_no_pool = _make_add_kernel()
        ref = add_no_pool(x, y)
        torch.cuda.synchronize()
        torch.testing.assert_close(ref, expected)

        # Compile with rewrite on. NOTE: importing helion.runtime here
        # re-uses the module-level pool state — clear it to avoid
        # cross-test interference.
        self._toggle_env("1")
        from helion.runtime import clear_pool
        from helion.runtime import set_pool_enabled

        clear_pool()
        set_pool_enabled(True)
        try:
            add_pool = _make_add_kernel()
            for _ in range(3):
                got = add_pool(x, y)
                torch.cuda.synchronize()
                torch.testing.assert_close(got, expected)
        finally:
            set_pool_enabled(False)
            clear_pool()

    @skipIfNotCUDA()
    def test_pool_populated_after_calls(self) -> None:
        """Multiple calls with the same shape should populate one ring
        in ``_POOLS`` — not allocate fresh per call."""
        self._toggle_env("1")
        from helion.runtime import clear_pool
        from helion.runtime import set_pool_enabled
        from helion.runtime._output_pool import _POOLS

        clear_pool()
        set_pool_enabled(True)
        try:
            add = _make_add_kernel()
            x = torch.randn(4096, device=DEVICE, dtype=torch.float32)
            y = torch.randn(4096, device=DEVICE, dtype=torch.float32)
            for _ in range(5):
                add(x, y)
            # Exactly one ring entry — same (dtype, shape, stride, device).
            self.assertEqual(len(_POOLS), 1)
        finally:
            set_pool_enabled(False)
            clear_pool()


if __name__ == "__main__":
    unittest.main()
