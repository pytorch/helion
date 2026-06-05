"""Tests for the ``HELION_OUTPUT_POOL=1`` codegen rewrite.

The rewrite swaps ``torch.empty_like(x)`` in the generated host wrapper
to ``_helion_pool_empty_like(x)`` (== ``helion.runtime._output_pool._empty_like``) when
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
        self.assertIn("_helion_pool_empty_like(x, _slot=0)", rewritten_line[0])

    @skipIfNotCUDA()
    def test_rewrite_present_even_without_env_var(self) -> None:
        """The rewrite is unconditional now — the wrapper always emits
        ``_helion_pool_empty_like``. When the pool is OFF at runtime
        (default), the helper short-circuits to ``torch.empty_like``.

        Doing the rewrite unconditionally lets a single compiled
        wrapper work for both autotune (pool on) and production (pool
        off) without recompiling.
        """
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
        self.assertIn("_helion_pool_empty_like(x, _slot=0)", rewritten_line[0])

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
        from helion.runtime._output_pool import _clear_pool as clear_pool
        from helion.runtime._output_pool import _set_pool_enabled as set_pool_enabled

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
        """Multiple calls with the same shape should populate one
        ``(dtype, shape, device, _slot)`` entry in the pool cache — not
        allocate fresh per call."""
        self._toggle_env("1")
        from helion.runtime._output_pool import _cache
        from helion.runtime._output_pool import _clear_pool as clear_pool
        from helion.runtime._output_pool import _set_pool_enabled as set_pool_enabled

        clear_pool()
        set_pool_enabled(True)
        try:
            add = _make_add_kernel()
            x = torch.randn(4096, device=DEVICE, dtype=torch.float32)
            y = torch.randn(4096, device=DEVICE, dtype=torch.float32)
            for _ in range(5):
                add(x, y)
            # Exactly one cache entry — same key.
            self.assertEqual(len(_cache()), 1)
        finally:
            set_pool_enabled(False)
            clear_pool()


class TestRewriteSlotDisambiguation(unittest.TestCase):
    """Multiple eligible ``torch.empty_like`` Assigns in the same
    wrapper must each get a UNIQUE ``_slot`` kwarg, so that two
    same-key kernel-output buffers don't collapse onto a single cached
    tensor (which would silently alias them at runtime)."""

    def test_two_outputs_get_distinct_slots(self) -> None:
        import ast

        from helion._compiler.generate_ast import _rewrite_output_allocs_for_pool

        # Build the AST equivalent of:
        #     out1 = torch.empty_like(x)
        #     out2 = torch.empty_like(x)
        #     launcher(x, out1, out2)
        # then mark the launcher Expr as ``_is_kernel_call=True`` (the
        # codegen sets that attribute on kernel-launch statements).
        def _empty_like_assign(target: str) -> ast.Assign:
            return ast.Assign(
                targets=[ast.Name(id=target, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="torch", ctx=ast.Load()),
                        attr="empty_like",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Name(id="x", ctx=ast.Load())],
                    keywords=[],
                ),
            )

        launcher_expr = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="launcher", ctx=ast.Load()),
                args=[
                    ast.Name(id="x", ctx=ast.Load()),
                    ast.Name(id="out1", ctx=ast.Load()),
                    ast.Name(id="out2", ctx=ast.Load()),
                ],
                keywords=[],
            )
        )
        launcher_expr._is_kernel_call = True  # type: ignore[attr-defined]

        stmts = [
            _empty_like_assign("out1"),
            _empty_like_assign("out2"),
            launcher_expr,
        ]
        _rewrite_output_allocs_for_pool(stmts)

        # Both Assigns rewritten to the pool helper with distinct _slot.
        slots: list[int] = []
        for stmt in stmts[:2]:
            assert isinstance(stmt, ast.Assign)
            call = stmt.value
            assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            self.assertEqual(call.func.id, "_helion_pool_empty_like")
            self.assertEqual(len(call.keywords), 1)
            kw = call.keywords[0]
            self.assertEqual(kw.arg, "_slot")
            assert isinstance(kw.value, ast.Constant)
            slots.append(kw.value.value)
        self.assertEqual(slots, [0, 1])
        self.assertEqual(len(set(slots)), 2)


if __name__ == "__main__":
    unittest.main()
