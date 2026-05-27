"""Tests for the ``_concrete_tensor_key`` static-shapes fast path.

``torch.Tensor`` and ``torch.nn.Parameter`` are dispatched to a
specialized extractor that uses ``tensor.size()`` and
``tensor.stride()`` directly as cache-key components, skipping the
``_hashable_dims`` wrap that exists only to normalize SymInts (which
appear on ``FakeTensor`` during tracing). The hash of the fast-path key
must match the old wrapped key so existing on-disk caches don't silently
miss.

These tests pin down:
1. Hash and equality of the fast-path key match the old wrapped key.
2. ``bind()`` still caches correctly across calls.
3. The dispatch table routes concrete tensors and FakeTensors to the
   right extractor.
"""

from __future__ import annotations

import unittest

import torch

import helion
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


class TestTensorKeyFastPath(unittest.TestCase):
    def test_dispatch_routes_concrete_tensor_to_fast_path(self) -> None:
        """Concrete ``torch.Tensor`` and ``torch.nn.Parameter`` are
        dispatched to ``_concrete_tensor_key``; ``FakeTensor`` goes
        through the original ``_tensor_key`` so its SymInt sizes still
        get normalized."""
        from torch._subclasses.fake_tensor import FakeTensor

        self.assertIs(_specialization_extractors[torch.Tensor], _concrete_tensor_key)
        self.assertIs(
            _specialization_extractors[torch.nn.Parameter], _concrete_tensor_key
        )
        self.assertIs(_specialization_extractors[FakeTensor], _tensor_key)

    def test_fast_path_key_hash_matches_wrapped(self) -> None:
        """The fast-path key (raw size/stride tuples) must hash and
        compare identically to the old wrapped key (via
        ``_hashable_dims``). Otherwise BoundKernel cache entries created
        under one path would silently miss under the other."""
        x = torch.empty(4096, dtype=torch.float32)
        fast = _concrete_tensor_key(_vector_add, x)
        wrapped = _tensor_key(_vector_add, x)
        self.assertEqual(hash(fast), hash(wrapped))
        self.assertEqual(fast, wrapped)

    def test_fast_path_used_for_static_shapes_real_tensor(self) -> None:
        """Real tensors under ``static_shapes=True`` get the fast-path
        key — verified by checking that the size component is the
        unwrapped ``torch.Size`` (the wrapped form would always be a
        plain ``tuple``)."""
        x = torch.empty(4096, dtype=torch.float32)
        key = _concrete_tensor_key(_vector_add, x)
        self.assertIs(type(key), tuple)
        self.assertIs(type(key[1]), torch.Size)
        self.assertEqual(tuple(key[1]), (4096,))
        self.assertEqual(key[2], (1,))

    def test_bind_caches_across_different_tensors_with_same_spec(self) -> None:
        """``bind()`` must reuse the same ``BoundKernel`` for different
        tensor objects sharing the same dtype/shape/stride."""
        x1 = torch.randn(64, device="cpu")
        y1 = torch.randn(64, device="cpu")
        x2 = torch.randn(64, device="cpu")
        y2 = torch.randn(64, device="cpu")
        bk1 = _vector_add.bind((x1, y1))
        bk2 = _vector_add.bind((x2, y2))
        self.assertIs(bk1, bk2)

    def test_bind_distinguishes_different_dtypes(self) -> None:
        x_f32 = torch.randn(64, device="cpu", dtype=torch.float32)
        y_f32 = torch.randn(64, device="cpu", dtype=torch.float32)
        x_f64 = torch.randn(64, device="cpu", dtype=torch.float64)
        y_f64 = torch.randn(64, device="cpu", dtype=torch.float64)
        bk1 = _vector_add.bind((x_f32, y_f32))
        bk2 = _vector_add.bind((x_f64, y_f64))
        self.assertIsNot(bk1, bk2)

    def test_bind_distinguishes_different_shapes(self) -> None:
        x1 = torch.randn(64, device="cpu")
        y1 = torch.randn(64, device="cpu")
        x2 = torch.randn(128, device="cpu")
        y2 = torch.randn(128, device="cpu")
        bk1 = _vector_add.bind((x1, y1))
        bk2 = _vector_add.bind((x2, y2))
        self.assertIsNot(bk1, bk2)


if __name__ == "__main__":
    unittest.main()
