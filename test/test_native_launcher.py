"""Tests for the minimal Rust launcher (Chunk E experiment).

The Rust launcher (``helion._native.CompiledLauncher``) is OPTIONAL
and built manually via ``cargo`` (see ``helion/_native/README.md``).
When present, it can be installed as the ``_launcher`` kwdefault of a
Helion-generated host wrapper to dispatch kernels via a ``__call__``
slot directly into ``compiled_kernel.run``, skipping both the Python
``default_launcher`` frame and Triton's ``JITFunction.run`` pipeline.

The launcher in this commit is intentionally minimal: no multi-spec
cache, no knob/hook re-reads, no ``used_global_vals`` check. Tests
verify correctness on the steady-state case (same args, same knobs)
and the priming + invocation lifecycle.
"""

from __future__ import annotations

import unittest

import torch

import helion
from helion import _native
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfNotTriton
import helion.language as hl


@helion.kernel(
    static_shapes=True,
    config=helion.Config(block_sizes=[1024], num_warps=4, num_stages=2),
)
def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for i in hl.tile(out.size(0)):
        out[i] = x[i] + y[i]
    return out


def _get_jit_function(kernel: helion.Kernel) -> object:
    from triton.runtime.jit import JITFunction

    bound = next(iter(kernel._bound_kernels.values()))  # type: ignore[attr-defined]
    return next(
        v for v in bound._run.__globals__.values() if isinstance(v, JITFunction)
    )


@skipIfNotTriton("Rust launcher is Triton-specific")
class TestRustLauncher(unittest.TestCase):
    def setUp(self) -> None:
        if _native.CompiledLauncher is None:
            self.skipTest("helion._native._launcher extension not built")

    @skipIfNotCUDA()
    def test_priming_then_calling_matches_reference(self) -> None:
        """A primed Rust launcher must produce the same output as a
        Python ``default_launcher`` call on the same args."""
        _add.reset()
        n = 4096
        a = torch.randn(n, device=DEVICE, dtype=torch.float32)
        b = torch.randn(n, device=DEVICE, dtype=torch.float32)
        expected = (a + b).clone()

        # Compile + run once to populate the Triton cache.
        _add(a, b)
        torch.cuda.synchronize()

        jit_fn = _get_jit_function(_add)
        out = torch.empty_like(a)
        grid = ((n + 1024 - 1) // 1024,)
        kernel_args = (a, b, out)

        launcher = _native.CompiledLauncher()
        launcher.prime(jit_fn, grid, kernel_args, num_warps=4, num_stages=2)

        # Direct call through the Rust launcher.
        result_buf = torch.empty_like(a)
        launcher(jit_fn, grid, a, b, result_buf)
        torch.cuda.synchronize()
        torch.testing.assert_close(result_buf, expected)

        # Now install as the wrapper's kwdefault and verify end-to-end.
        bound = next(iter(_add._bound_kernels.values()))  # type: ignore[attr-defined]
        compiled_wrapper = bound._run
        default = compiled_wrapper.__kwdefaults__["_launcher"]
        self.addCleanup(
            lambda: compiled_wrapper.__kwdefaults__.__setitem__("_launcher", default)
        )
        compiled_wrapper.__kwdefaults__["_launcher"] = launcher

        result = _add(a, b)
        torch.cuda.synchronize()
        torch.testing.assert_close(result, expected)

    @skipIfNotCUDA()
    def test_call_without_priming_raises(self) -> None:
        """Calling an unprimed launcher must raise rather than crash."""
        launcher = _native.CompiledLauncher()
        a = torch.empty(8, device=DEVICE, dtype=torch.float32)
        b = torch.empty(8, device=DEVICE, dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            launcher(None, (1,), a, b)

    def test_launcher_type_is_subclassable(self) -> None:
        """``CompiledLauncher`` is declared with ``subclass`` so Python
        subclasses are possible (e.g. wrapping with extra Python-side
        guards)."""
        if _native.CompiledLauncher is None:
            self.skipTest("extension not built")

        class Sub(_native.CompiledLauncher):
            pass

        self.assertTrue(issubclass(Sub, _native.CompiledLauncher))


if __name__ == "__main__":
    unittest.main()
