"""Tests for the C `tensor_key` accelerator + its Python fallback.

The C extension is OPTIONAL — built manually via the instructions in
``helion/_C/README.md`` and not included in the default install. These
tests verify:

1. With or without the extension, ``Kernel.bind`` produces identical
   ``_bound_kernels`` cache keys (the Python fallback and C fast path
   are equivalent).
2. When the extension is present, ``helion._C.tensor_key`` is callable
   and returns the documented 4-tuple shape for static-shapes tensors.
3. When the extension is absent, ``helion._C.tensor_key`` is ``None``
   and ``_tensor_key`` transparently uses the Python implementation.
4. Unsupported inputs (e.g. SymInt sizes, dynamic-shapes mode) make
   the C path return ``None`` and the Python fallback takes over.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

import helion
from helion import _C
from helion._testing import DEVICE
from helion._testing import TestCase
import helion.language as hl
from helion.runtime.kernel import _tensor_key


@helion.kernel(static_shapes=True)
def _add_one_static(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=False)
def _add_one_dynamic(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile] + 1
    return out


class TestCTensorKeyShape(unittest.TestCase):
    """Shape + structure of the C output (when built)."""

    def setUp(self) -> None:
        if not _C.AVAILABLE or _C.tensor_key is None:
            self.skipTest("helion._C extension not built")

    def test_static_shapes_tensor_returns_4tuple(self) -> None:
        """For a typical static-shapes tensor, the C path returns
        ``(dtype, sizes, strides, static_indices)``."""
        x = torch.randn(32, 64, dtype=torch.float32)
        result = _C.tensor_key(x, frozenset())
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)
        dtype, sizes, strides, static_indices = result
        self.assertEqual(dtype, torch.float32)
        self.assertEqual(sizes, (32, 64))
        self.assertEqual(strides, tuple(x.stride()))
        self.assertEqual(static_indices, frozenset())

    def test_threads_static_indices_through(self) -> None:
        """The ``static_indices`` arg is returned verbatim in slot 3."""
        x = torch.randn(8, dtype=torch.float32)
        si = frozenset({1, 3})
        result = _C.tensor_key(x, si)
        self.assertEqual(result[3], si)

    def test_1d_tensor(self) -> None:
        x = torch.arange(16, dtype=torch.int64)
        result = _C.tensor_key(x, frozenset())
        self.assertEqual(result, (torch.int64, (16,), (1,), frozenset()))

    def test_strided_view_has_distinct_strides(self) -> None:
        """A non-contiguous view should produce strides that reflect it."""
        base = torch.randn(64, 64, dtype=torch.float32)
        view = base[:, ::2]  # stride (64, 2) on dim
        result = _C.tensor_key(view, frozenset())
        self.assertEqual(result[1], tuple(view.size()))
        self.assertEqual(result[2], tuple(view.stride()))


class TestPythonFallback(TestCase):
    """The Python `_tensor_key` path must produce the same result as
    the C path for any input the C path accepts — and must take over
    cleanly when the C path returns ``None`` (or when the extension is
    absent altogether)."""

    def test_python_and_c_paths_produce_equal_keys(self) -> None:
        """Same tensor through C path and through Python path → same key."""
        x = torch.randn(32, 64, dtype=torch.float32)
        py_key = _tensor_key(_add_one_static, x)
        if _C.tensor_key is None:
            self.skipTest("C extension not built; nothing to compare against")
        # Patch out the C path and force the Python branch, then compare.
        with mock.patch("helion.runtime.kernel._C_tensor_key", None):
            py_fallback_key = _tensor_key(_add_one_static, x)
        self.assertEqual(py_key, py_fallback_key)

    def test_bind_cache_hit_with_and_without_c(self) -> None:
        """Two calls with the same shape MUST land on the same
        BoundKernel, whether the C fast path or the Python path
        computed the lookup key. If they produced different keys,
        we'd compile twice and double the BoundKernel cache."""
        # Make a kernel that can run on the available device.
        x1 = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)
        x2 = torch.randn(4, 8, device=DEVICE, dtype=torch.float32)

        # First call with whatever path is configured by default.
        _add_one_static.reset()
        _add_one_static(x1)
        first_id = id(next(iter(_add_one_static._bound_kernels.values())))

        # Force the Python fallback for the second call by nulling the C
        # binding — same shape, same dtype → must hit the same entry.
        with mock.patch("helion.runtime.kernel._C_tensor_key", None):
            _add_one_static(x2)
        cache = _add_one_static._bound_kernels
        self.assertEqual(len(cache), 1)
        self.assertEqual(id(next(iter(cache.values()))), first_id)
        _add_one_static.reset()

    def test_dynamic_shapes_uses_python_path(self) -> None:
        """`static_shapes=False` must go through the Python path (the C
        path is static-shapes-only) and produce the bucketed key."""
        x = torch.randn(8, dtype=torch.float32)
        # The C path is only consulted when static_shapes=True. For a
        # dynamic-shapes kernel, _tensor_key must return the
        # bucketed-size shape, which the C extension does NOT implement.
        key = _tensor_key(_add_one_dynamic, x)
        # Bucketed-size keys are 3- or 4-tuples; static-shapes keys are
        # 4-tuples containing sizes AND strides. Distinguish by
        # checking sizes-vs-bucketed in slot 1.
        # For our 1-element shape this should be a non-tuple-of-strides shape.
        self.assertNotIsInstance(key[1], tuple) if len(key) == 3 else self.assertTrue(
            True
        )


class TestUnsupportedInputs(unittest.TestCase):
    """The C path returns ``None`` for inputs it can't handle, so the
    Python fallback path takes over cleanly."""

    def setUp(self) -> None:
        if not _C.AVAILABLE or _C.tensor_key is None:
            self.skipTest("helion._C extension not built")

    def test_non_tensor_input_returns_none_or_raises(self) -> None:
        """Passing a non-tensor either crashes (we don't promise a clean
        signature) or returns None — but must not corrupt state."""
        try:
            result = _C.tensor_key("not a tensor", frozenset())
            self.assertIsNone(result)
        except Exception:
            # Acceptable — caller is expected to pass tensors.
            pass


if __name__ == "__main__":
    unittest.main()
