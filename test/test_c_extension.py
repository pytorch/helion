"""Smoke tests for the ``helion._C`` optional-native-extension shim.

The extension itself is added in a later commit; these tests pin down
the Python-side contract that lets the rest of the codebase use the
``helion._C`` module safely whether or not the compiled extension is
present.
"""

from __future__ import annotations

import importlib
import unittest


class TestCExtensionShim(unittest.TestCase):
    def test_helion_c_imports_cleanly(self) -> None:
        """``import helion._C`` must succeed even when the extension is absent."""
        c_mod = importlib.import_module("helion._C")
        self.assertTrue(hasattr(c_mod, "AVAILABLE"))
        self.assertTrue(hasattr(c_mod, "tensor_key"))

    def test_available_flag_is_bool(self) -> None:
        """``AVAILABLE`` must be a bool so callers can use it in
        unambiguous truthiness checks."""
        from helion import _C

        self.assertIsInstance(_C.AVAILABLE, bool)

    def test_symbol_consistency(self) -> None:
        """If ``AVAILABLE`` is ``True``, each public function slot must
        be callable; if ``False``, each must be ``None`` so callers can
        gate on ``is not None``."""
        from helion import _C

        if _C.AVAILABLE:
            # Extension built — all advertised symbols must resolve.
            self.assertTrue(callable(_C.tensor_key))
        else:
            # Extension absent — every slot must be ``None`` exactly,
            # not some other falsy value, so callers can use
            # ``if _C.tensor_key is not None`` unambiguously.
            self.assertIsNone(_C.tensor_key)

    def test_helion_imports_with_extension_absent(self) -> None:
        """Top-level ``import helion`` must work regardless of extension state.

        The extension is optional. If it isn't built (e.g. on a non-CPython
        runtime, a stripped build, or before the C source lands), ``helion``
        must still be importable and functional — falling back to pure
        Python implementations.
        """
        import helion  # noqa: F401

        # Re-import _C to be sure we don't have a stale state.
        from helion import _C

        # Regardless of whether the extension built, _C must have these
        # documented attributes.
        self.assertTrue(hasattr(_C, "AVAILABLE"))
        self.assertTrue(hasattr(_C, "tensor_key"))


if __name__ == "__main__":
    unittest.main()
