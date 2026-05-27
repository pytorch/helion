"""Smoke tests for the ``helion._native`` optional-extension shim.

The Rust extension itself is added in a later commit; these tests pin
down the Python-side contract that lets the rest of the codebase use
the ``helion._native`` module safely whether or not the compiled
extension is present.
"""

from __future__ import annotations

import importlib
import unittest


class TestNativeExtensionShim(unittest.TestCase):
    def test_helion_native_imports_cleanly(self) -> None:
        """``import helion._native`` must succeed even when the extension is absent."""
        mod = importlib.import_module("helion._native")
        self.assertTrue(hasattr(mod, "AVAILABLE"))
        self.assertTrue(hasattr(mod, "tensor_key"))
        self.assertTrue(hasattr(mod, "CompiledLauncher"))

    def test_available_flag_is_bool(self) -> None:
        """``AVAILABLE`` must be a bool so callers can use it in
        unambiguous truthiness checks."""
        from helion import _native

        self.assertIsInstance(_native.AVAILABLE, bool)

    def test_symbol_consistency(self) -> None:
        """``tensor_key`` is reserved as ``None`` (no Rust accelerator
        ships in this slot yet). ``CompiledLauncher`` is either a class
        when the extension is built, or ``None`` when it isn't."""
        from helion import _native

        # tensor_key has no Rust implementation; the slot is reserved.
        self.assertIsNone(_native.tensor_key)

        if _native.AVAILABLE:
            self.assertTrue(callable(_native.CompiledLauncher))
        else:
            self.assertIsNone(_native.CompiledLauncher)

    def test_helion_imports_with_extension_absent(self) -> None:
        """Top-level ``import helion`` must work regardless of extension state.

        The extension is optional. If it isn't built (e.g. on a non-CPython
        runtime, a stripped build, or before the Rust source lands),
        ``helion`` must still be importable and functional — falling back to
        pure Python implementations.
        """
        import helion  # noqa: F401

        # Re-import to be sure we don't have a stale state.
        from helion import _native

        self.assertTrue(hasattr(_native, "AVAILABLE"))
        self.assertTrue(hasattr(_native, "tensor_key"))
        self.assertTrue(hasattr(_native, "CompiledLauncher"))


if __name__ == "__main__":
    unittest.main()
