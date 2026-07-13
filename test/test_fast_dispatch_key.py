"""Tests for ``Kernel._fast_dispatch_key``, the cheap dispatch key that backs
the ``Kernel.__call__`` fast-path cache (``_dispatch_cache``).

The correctness argument for the cache is that the fast key is *strictly finer*
than the full specialization key: any two argument lists that produce different
full keys also produce different fast keys, so a fast-key hit can never dispatch
to a BoundKernel that a full ``bind()`` would not have resolved for those
arguments. "Strictly" finer because the converse fails -- the fast key
distinguishes argument lists (e.g. scalar values, or dynamic-shape sizes that
bucket together) that the full key deliberately collapses.

These tests pin down:
1. Refinement: across a matrix of dtype/shape/stride variants, distinct full
   keys always imply distinct fast keys.
2. Strictness under two witnesses that share a full key but not a fast key:
   scalar value (full key records only ``type``) and dynamic-shape bucketing.
3. The documented ``None`` returns: unhandled argument types and the
   no-tensor-to-pin-the-device case.
4. ``key=`` functions feed into the fast key.

The key is pure argument-metadata bookkeeping, so it needs no compilation and
runs on CPU-only bots.
"""

from __future__ import annotations

import unittest

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def _static_add1(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=False)
def _dynamic_add1(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@helion.kernel(static_shapes=False)
def _dynamic_add_scalar(x: torch.Tensor, s: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + s
    return out


class TestFastDispatchKey(unittest.TestCase):
    def test_refinement_across_dtype_shape_stride(self) -> None:
        """Every pair of argument lists with different *full* keys must also
        have different *fast* keys -- the property that makes a fast-key hit
        safe to dispatch directly to a previously-bound BoundKernel."""
        variants = {
            "f32_64": (torch.empty(64, dtype=torch.float32),),
            "f64_64": (torch.empty(64, dtype=torch.float64),),
            "f32_128": (torch.empty(128, dtype=torch.float32),),
            "f32_8x16T": (torch.empty(8, 16, dtype=torch.float32).transpose(0, 1),),
            "f32_8x16": (torch.empty(8, 16, dtype=torch.float32),),
        }
        full = {
            name: _static_add1.specialization_key(a) for name, a in variants.items()
        }
        fast = {
            name: _static_add1._fast_dispatch_key(a) for name, a in variants.items()
        }

        names = list(variants)
        # There must be at least one distinct pair, else the test is vacuous.
        saw_distinct_full = False
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if full[a] != full[b]:
                    saw_distinct_full = True
                    self.assertNotEqual(
                        fast[a],
                        fast[b],
                        msg=f"{a} vs {b}: full keys differ but fast keys collide",
                    )
        self.assertTrue(saw_distinct_full)

    def test_strictly_finer_via_scalar_value(self) -> None:
        """Witness that the fast key is *strictly* finer: two calls whose only
        difference is a scalar's value share a full key (which records only the
        scalar's ``type``) but get distinct fast keys (which record its value).
        """
        x = torch.empty(64, dtype=torch.float32)
        a = (x, 1)
        b = (x, 2)
        self.assertEqual(
            _dynamic_add_scalar.specialization_key(a),
            _dynamic_add_scalar.specialization_key(b),
        )
        self.assertNotEqual(
            _dynamic_add_scalar._fast_dispatch_key(a),
            _dynamic_add_scalar._fast_dispatch_key(b),
        )

    def test_strictly_finer_via_dynamic_shape_bucketing(self) -> None:
        """Second strictness witness: under ``static_shapes=False`` two nearby
        sizes bucket to the same full key, but the fast key records the exact
        shape and keeps them apart."""
        a = (torch.empty(4096, dtype=torch.float32),)
        b = (torch.empty(4097, dtype=torch.float32),)
        self.assertEqual(
            _dynamic_add1.specialization_key(a),
            _dynamic_add1.specialization_key(b),
        )
        self.assertNotEqual(
            _dynamic_add1._fast_dispatch_key(a),
            _dynamic_add1._fast_dispatch_key(b),
        )

    def test_fast_key_records_exact_tensor_metadata(self) -> None:
        """The per-tensor entry is exactly (dtype, shape, stride, device,
        static-indices) with the raw ``torch.Size`` -- i.e. finer than any
        bucketed/normalized form."""
        x = torch.empty(64, dtype=torch.float32)
        key = _static_add1._fast_dispatch_key((x,))
        assert isinstance(key, tuple)
        entry = key[0]
        self.assertEqual(
            entry,
            (x.dtype, x.shape, x.stride(), x.device, None),
        )
        self.assertIs(type(entry[1]), torch.Size)

    def test_returns_none_for_unhandled_arg_type(self) -> None:
        """A container / unsupported argument type forces the slow ``bind()``
        path by returning ``None``."""
        x = torch.empty(64, dtype=torch.float32)
        self.assertIsNone(_static_add1._fast_dispatch_key((x, [1, 2, 3])))
        self.assertIsNone(_static_add1._fast_dispatch_key((x, object())))

    def test_returns_none_without_tensor_to_pin_device(self) -> None:
        """With no tensor argument there is nothing to pin the device, so the
        key is ``None`` even though the scalars are individually handled."""
        self.assertIsNone(_dynamic_add_scalar._fast_dispatch_key((1, 2)))
        self.assertIsNone(_dynamic_add_scalar._fast_dispatch_key(()))

    def test_scalar_and_none_entries(self) -> None:
        """Handled non-tensor arguments contribute (type, value) for scalars
        and a bare ``None`` for ``None``, provided a tensor pins the device."""
        x = torch.empty(64, dtype=torch.float32)
        key = _dynamic_add_scalar._fast_dispatch_key((x, 7))
        assert isinstance(key, tuple)
        self.assertEqual(key[1], (int, 7))

        none_key = _static_add1._fast_dispatch_key((x, None))
        assert isinstance(none_key, tuple)
        self.assertIsNone(none_key[1])

    def test_key_fn_feeds_into_fast_key(self) -> None:
        """A user ``key=`` function participates in the fast key, so two calls
        the user marks distinct get distinct fast keys."""

        state = {"v": 0}

        @helion.kernel(static_shapes=True, key=lambda x: state["v"])
        def _with_key(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + 1
            return out

        x = torch.empty(64, dtype=torch.float32)
        state["v"] = 0
        k0 = _with_key._fast_dispatch_key((x,))
        state["v"] = 1
        k1 = _with_key._fast_dispatch_key((x,))
        self.assertNotEqual(k0, k1)


if __name__ == "__main__":
    unittest.main()
