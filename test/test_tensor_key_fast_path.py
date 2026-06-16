"""Tests for the ``_concrete_tensor_key`` specialization-key fast path.

``torch.Tensor`` and ``torch.nn.Parameter`` are dispatched (by exact type)
to ``_concrete_tensor_key``, which uses ``tensor.size()`` / ``tensor.stride()``
directly instead of the ``_hashable_dims`` SymInt-normalization wrap that
``_tensor_key`` applies. Anything that can carry a SymInt -- ``FakeTensor``
and ``torch.Tensor`` *subclasses* reached via the ``isinstance`` fallback --
still goes through ``_tensor_key``.

These tests pin down:
1. The fast-path key hashes and compares equal to the wrapped key, so
   existing BoundKernel / on-disk caches don't silently miss.
2. The dispatch table routes concrete tensors, FakeTensors, and the
   subclass-fallback to the right extractor.
3. Tensor subclasses take the SymInt-safe path without error.
4. ``bind()`` still caches/distinguishes correctly across dtype and shape.
"""

from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
import helion.language as hl
from helion.runtime.kernel import _concrete_tensor_key
from helion.runtime.kernel import _specialization_extractors
from helion.runtime.kernel import _tensor_key


@helion.kernel(static_shapes=True)
def _vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(static_shapes=False)
def _vector_add_dynamic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


class TestTensorKeyFastPath(unittest.TestCase):
    def test_dispatch_routes_concrete_tensor_to_fast_path(self) -> None:
        """``torch.Tensor`` / ``torch.nn.Parameter`` dispatch to the fast
        extractor; ``FakeTensor`` and the subclass fallback keep the
        SymInt-safe ``_tensor_key``."""
        from torch._subclasses.fake_tensor import FakeTensor

        self.assertIs(_specialization_extractors[torch.Tensor], _concrete_tensor_key)
        self.assertIs(
            _specialization_extractors[torch.nn.Parameter], _concrete_tensor_key
        )
        self.assertIs(_specialization_extractors[FakeTensor], _tensor_key)
        # The fallback for torch.Tensor subclasses is registered under a
        # string key so the dispatch site stays loosely typed.
        self.assertIs(_specialization_extractors["tensor_subclass"], _tensor_key)

    def test_fast_path_key_hash_matches_wrapped_static_shapes(self) -> None:
        """Fast-path key must hash and compare identically to the wrapped
        key under static_shapes=True; otherwise cache entries created under
        one path silently miss under the other."""
        x = torch.empty(4096, dtype=torch.float32)
        fast = _concrete_tensor_key(_vector_add, x)
        wrapped = _tensor_key(_vector_add, x)
        self.assertEqual(hash(fast), hash(wrapped))
        self.assertEqual(fast, wrapped)

    def test_fast_path_key_hash_matches_wrapped_dynamic_shapes(self) -> None:
        """Equivalence must also hold on the dynamic-shape (bucketed) path,
        where the key carries the int32/int64 index-width bit."""
        x = torch.empty(4096, dtype=torch.float32)
        fast = _concrete_tensor_key(_vector_add_dynamic, x)
        wrapped = _tensor_key(_vector_add_dynamic, x)
        self.assertEqual(hash(fast), hash(wrapped))
        self.assertEqual(fast, wrapped)

    def test_fast_path_key_matches_wrapped_for_strided_tensor(self) -> None:
        """A non-contiguous (transposed) tensor exercises the stride
        component; the fast and wrapped keys must still agree."""
        x = torch.empty(8, 16, dtype=torch.float32).transpose(0, 1)
        self.assertFalse(x.is_contiguous())
        fast = _concrete_tensor_key(_vector_add, x)
        wrapped = _tensor_key(_vector_add, x)
        self.assertEqual(hash(fast), hash(wrapped))
        self.assertEqual(fast, wrapped)

    def test_fast_path_uses_unwrapped_size_under_static_shapes(self) -> None:
        """The size component is the raw ``torch.Size`` (the wrapped form is
        always a plain tuple), confirming the fast path actually skips
        ``_hashable_dims``."""
        x = torch.empty(4096, dtype=torch.float32)
        key = _concrete_tensor_key(_vector_add, x)
        assert isinstance(key, tuple)
        self.assertIs(type(key[1]), torch.Size)
        self.assertEqual(tuple(key[1]), (4096,))
        self.assertEqual(key[2], (1,))

    def test_subclass_routes_to_symint_safe_extractor(self) -> None:
        """A ``torch.Tensor`` subclass misses the exact-type dict and takes
        the ``isinstance`` fallback -> ``_tensor_key`` (not the fast path).
        The computed key must match the fast path for the same shape so a
        subclass and a plain tensor share a BoundKernel."""

        class MyTensor(torch.Tensor):
            pass

        base = torch.empty(64, dtype=torch.float32)
        sub = base.as_subclass(MyTensor)
        self.assertIsNot(type(sub), torch.Tensor)
        # Goes through Kernel._specialization_key's isinstance fallback.
        sub_key = _vector_add._specialization_key(sub)
        plain_key = _vector_add._specialization_key(base)
        self.assertEqual(sub_key, plain_key)
        self.assertEqual(hash(sub_key), hash(plain_key))

    @skipIfNotCUDA()
    def test_bind_caches_across_tensors_with_same_spec(self) -> None:
        """``bind()`` reuses one BoundKernel for distinct tensor objects of
        the same dtype/shape/stride."""
        x1 = torch.randn(64, device=DEVICE)
        y1 = torch.randn(64, device=DEVICE)
        x2 = torch.randn(64, device=DEVICE)
        y2 = torch.randn(64, device=DEVICE)
        self.assertIs(_vector_add.bind((x1, y1)), _vector_add.bind((x2, y2)))

    @skipIfNotCUDA()
    def test_bind_distinguishes_dtype_and_shape(self) -> None:
        x_f32 = torch.randn(64, dtype=torch.float32, device=DEVICE)
        x_f64 = torch.randn(64, dtype=torch.float64, device=DEVICE)
        x_big = torch.randn(128, dtype=torch.float32, device=DEVICE)
        self.assertIsNot(
            _vector_add.bind((x_f32, x_f32)), _vector_add.bind((x_f64, x_f64))
        )
        self.assertIsNot(
            _vector_add.bind((x_f32, x_f32)), _vector_add.bind((x_big, x_big))
        )


if __name__ == "__main__":
    unittest.main()
