"""Tests for ``Kernel._fast_dispatch_key``, the cheap dispatch key that backs
the ``Kernel.__call__`` fast-path cache (``_dispatch_cache``).

The correctness argument for the cache is that the fast key refines the full
specialization key: any two argument lists that produce different full keys also
produce different fast keys, so a fast-key hit can never dispatch to a
BoundKernel that a full ``bind()`` would not have resolved for those arguments.
The fast key remains strictly finer for exact tensor metadata that the dynamic
specialization key deliberately buckets together.

These tests pin down:
1. Refinement: across a matrix of dtype/shape/stride variants, distinct full
   keys always imply distinct fast keys.
2. Runtime scalar values share a key, while constexpr values remain distinct.
3. Strictness under dynamic-shape bucketing.
4. The documented ``None`` returns: unhandled argument types and the
   no-tensor-to-pin-the-device case.
5. ``key=`` functions feed into the fast key.

The key is pure argument-metadata bookkeeping, so it needs no compilation and
runs on CPU-only bots.
"""

from __future__ import annotations

import unittest

import torch

import helion
import helion.language as hl
from helion.runtime.kernel import _make_prepared_arg_guard


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


@helion.kernel(static_shapes=False)
def _dynamic_add_constexpr(x: torch.Tensor, s: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + s
    return out


@helion.kernel(static_shapes=False)
def _dynamic_add_scalar_list(x: torch.Tensor, values: list[int]) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + values[0]
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

    def test_runtime_scalar_values_share_fast_key(self) -> None:
        x = torch.empty(64, dtype=torch.float32)
        a = (x, 1)
        b = (x, 2)
        self.assertEqual(
            _dynamic_add_scalar.specialization_key(a),
            _dynamic_add_scalar.specialization_key(b),
        )
        self.assertEqual(
            _dynamic_add_scalar._fast_dispatch_key(a),
            _dynamic_add_scalar._fast_dispatch_key(b),
        )
        self.assertEqual(
            len(
                {
                    _dynamic_add_scalar._fast_dispatch_key((x, value))
                    for value in (True, 1, 1.0)
                }
            ),
            3,
        )

    def test_constexpr_scalar_values_have_distinct_fast_keys(self) -> None:
        x = torch.empty(64, dtype=torch.float32)
        self.assertNotEqual(
            _dynamic_add_constexpr._fast_dispatch_key((x, 1)),
            _dynamic_add_constexpr._fast_dispatch_key((x, 2)),
        )
        self.assertNotEqual(
            _dynamic_add_scalar._fast_dispatch_key((x, hl.constexpr(1))),
            _dynamic_add_scalar._fast_dispatch_key((x, hl.constexpr(2))),
        )

    def test_nested_runtime_scalar_values_share_fast_key(self) -> None:
        x = torch.empty(64, dtype=torch.float32)
        self.assertEqual(
            _dynamic_add_scalar_list._fast_dispatch_key((x, [1, 2])),
            _dynamic_add_scalar_list._fast_dispatch_key((x, [3, 4])),
        )

    def test_prepared_guard_preserves_nested_constexpr_values(self) -> None:
        x = torch.empty(64, dtype=torch.float32)
        guard = _make_prepared_arg_guard(_dynamic_add_constexpr, (x, (1, 2)))

        self.assertTrue(guard((x, (1, 2))))
        self.assertFalse(guard((x, (3, 4))))

    def test_prepared_guard_supports_constexpr_wrapped_values(self) -> None:
        x = torch.empty(64, dtype=torch.float32)
        guard = _make_prepared_arg_guard(_dynamic_add_scalar, (x, hl.constexpr(2)))

        self.assertTrue(guard((x, hl.constexpr(2))))
        self.assertFalse(guard((x, hl.constexpr(3))))
        self.assertFalse(guard((x, 2)))

        fguard = _make_prepared_arg_guard(_dynamic_add_scalar, (x, hl.constexpr(0.0)))
        self.assertTrue(fguard((x, hl.constexpr(0.0))))
        self.assertFalse(fguard((x, hl.constexpr(-0.0))))
        with self.assertRaises(TypeError):
            _make_prepared_arg_guard(
                _dynamic_add_scalar, (x, hl.constexpr(float("nan")))
            )

    def test_constexpr_container_with_tensor_returns_none(self) -> None:
        """A constexpr-annotated container specializes on the raw argument, so
        tensor identity drives the full key; a metadata fast key would conflate
        keys the full key distinguishes, so there must be no fast key."""
        x = torch.empty(64, dtype=torch.float32)
        t = torch.empty(8, dtype=torch.float32)
        self.assertIsNone(_dynamic_add_constexpr._fast_dispatch_key((x, (t, 1))))
        self.assertIsNone(
            _dynamic_add_constexpr._fast_dispatch_key((x, {"weights": t}))
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
        """An unsupported nested argument forces the slow ``bind()`` path."""
        x = torch.empty(64, dtype=torch.float32)
        self.assertIsNone(_static_add1._fast_dispatch_key((x, [object()])))
        self.assertIsNone(_static_add1._fast_dispatch_key((x, object())))

    def test_nested_containers_record_structure_and_tensor_metadata(self) -> None:
        x = torch.empty(64, dtype=torch.float32)
        y = torch.empty(128, dtype=torch.float32)

        tuple_key = _static_add1._fast_dispatch_key(((x, y),))
        list_key = _static_add1._fast_dispatch_key(([x, y],))
        dict_key = _static_add1._fast_dispatch_key(({"x": x, "y": y},))
        reordered_dict_key = _static_add1._fast_dispatch_key(({"y": y, "x": x},))
        nested_key = _static_add1._fast_dispatch_key(({"pair": [x, y]},))

        self.assertIsNotNone(tuple_key)
        self.assertIsNotNone(list_key)
        self.assertIsNotNone(nested_key)
        self.assertNotEqual(tuple_key, list_key)
        self.assertEqual(dict_key, reordered_dict_key)
        self.assertNotEqual(
            tuple_key,
            _static_add1._fast_dispatch_key(((x, torch.empty_like(x)),)),
        )

    def test_nested_container_without_tensor_returns_none(self) -> None:
        self.assertIsNone(_dynamic_add_scalar._fast_dispatch_key((([1, 2],),)))

    def test_prepared_guard_recurses_through_nested_containers(self) -> None:
        x = torch.empty(64, dtype=torch.float32)
        y = torch.empty(128, dtype=torch.float32)
        args = ({"pair": [x, y], "scale": 2},)
        guard = _make_prepared_arg_guard(_static_add1, args)

        self.assertTrue(
            guard(
                (
                    {
                        "scale": 3,
                        "pair": [torch.empty_like(x), torch.empty_like(y)],
                    },
                )
            )
        )
        self.assertFalse(guard(({"pair": (x, y), "scale": 2},)))
        self.assertFalse(guard(({"pair": [x], "scale": 2},)))
        self.assertFalse(guard(({"pair": [x, y], "other": 2},)))
        self.assertFalse(guard(({"pair": [x, torch.empty_like(x)], "scale": 2},)))

    def test_returns_none_without_tensor_to_pin_device(self) -> None:
        """With no tensor argument there is nothing to pin the device, so the
        key is ``None`` even though the scalars are individually handled."""
        self.assertIsNone(_dynamic_add_scalar._fast_dispatch_key((1, 2)))
        self.assertIsNone(_dynamic_add_scalar._fast_dispatch_key(()))

    def test_scalar_and_none_entries(self) -> None:
        """Runtime scalars contribute their type and ``None`` stays bare."""
        x = torch.empty(64, dtype=torch.float32)
        key = _dynamic_add_scalar._fast_dispatch_key((x, 7))
        assert isinstance(key, tuple)
        self.assertIs(key[1], int)

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

    def test_key_fn_is_evaluated_once_with_specialization_extras(self) -> None:
        calls = 0

        def user_key(x: torch.Tensor) -> int:
            nonlocal calls
            calls += 1
            return int(x.numel())

        @helion.kernel(static_shapes=True, key=user_key)
        def _with_key(x: torch.Tensor) -> torch.Tensor:
            return x

        args = (torch.empty(64),)
        signature = _with_key._base_specialization_key(args)
        _with_key._specialize_extra[signature] = [lambda values: len(values)]
        _with_key._has_specialization_extras = True
        calls = 0

        key = _with_key._fast_dispatch_key(args)

        self.assertIsNotNone(key)
        self.assertEqual(calls, 1)
        assert isinstance(key, tuple)
        self.assertEqual(key[-2:], (64, (1,)))


if __name__ == "__main__":
    unittest.main()
