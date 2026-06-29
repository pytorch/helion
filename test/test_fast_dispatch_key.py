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
5. Pallas tensor alias relationships participate in both specialization paths.

The key is pure argument-metadata bookkeeping, so it needs no compilation and
runs on CPU-only bots.
"""

from __future__ import annotations

import dataclasses
import unittest
from unittest.mock import Mock

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


@helion.kernel(backend="pallas", static_shapes=True)
def _pallas_static_add_tensors(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
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

    def test_tensor_aliasing_is_part_of_both_keys(self) -> None:
        x = torch.empty(64, dtype=torch.float32)
        alias = x.view_as(x)
        independent = torch.empty_like(x)
        aliased_args = (x, alias)
        independent_args = (x, independent)
        fresh_x = torch.empty_like(x)
        fresh_aliased_args = (fresh_x, fresh_x.view_as(fresh_x))
        fresh_independent_args = (fresh_x, torch.empty_like(fresh_x))

        aliased_full = _pallas_static_add_tensors.specialization_key(aliased_args)
        independent_full = _pallas_static_add_tensors.specialization_key(
            independent_args
        )
        aliased_fast = _pallas_static_add_tensors._fast_dispatch_key(aliased_args)
        independent_fast = _pallas_static_add_tensors._fast_dispatch_key(
            independent_args
        )
        self.assertNotEqual(aliased_full, independent_full)
        self.assertNotEqual(aliased_fast, independent_fast)
        self.assertEqual(
            aliased_full,
            _pallas_static_add_tensors.specialization_key(fresh_aliased_args),
        )
        self.assertEqual(
            independent_full,
            _pallas_static_add_tensors.specialization_key(fresh_independent_args),
        )
        self.assertEqual(
            aliased_fast,
            _pallas_static_add_tensors._fast_dispatch_key(fresh_aliased_args),
        )
        self.assertEqual(
            independent_fast,
            _pallas_static_add_tensors._fast_dispatch_key(fresh_independent_args),
        )

    def test_container_aliasing_uses_specialization_order(self) -> None:
        @dataclasses.dataclass
        class TensorBundle:
            a: torch.Tensor
            b: torch.Tensor
            c: torch.Tensor

        x = torch.empty(64, dtype=torch.float32)
        y = torch.empty_like(x)
        containers = (
            (
                {"a": x, "b": x, "c": y},
                {"a": x, "c": x, "b": y},
            ),
            (
                TensorBundle(a=x, b=x, c=y),
                TensorBundle(a=x, b=y, c=x),
            ),
        )
        for first, second in containers:
            with self.subTest(container_type=type(first).__name__):
                self.assertNotEqual(
                    _pallas_static_add_tensors.specialization_key((first,)),
                    _pallas_static_add_tensors.specialization_key((second,)),
                )

    def test_tensor_aliasing_does_not_require_view_base(self) -> None:
        x = torch.empty(64, dtype=torch.float32)
        detached = x.detach()
        explicit_storage_alias = torch.empty(0, dtype=x.dtype).set_(
            x.untyped_storage(), 0, x.size(), x.stride()
        )
        expected_full = _pallas_static_add_tensors.specialization_key((x, x.view_as(x)))
        expected_fast = _pallas_static_add_tensors._fast_dispatch_key((x, x.view_as(x)))

        for alias in (detached, explicit_storage_alias):
            with self.subTest(alias=alias):
                self.assertIsNone(alias._base)
                self.assertEqual(
                    _pallas_static_add_tensors.specialization_key((x, alias)),
                    expected_full,
                )
                self.assertEqual(
                    _pallas_static_add_tensors._fast_dispatch_key((x, alias)),
                    expected_fast,
                )

    def test_tensor_aliasing_in_function_closure_is_specialized(self) -> None:
        def make_helper(captured: torch.Tensor):
            def helper(value: torch.Tensor) -> torch.Tensor:
                return value + captured

            return helper

        x = torch.empty(64, dtype=torch.float32)
        independent = torch.empty_like(x)
        self.assertNotEqual(
            _pallas_static_add_tensors.specialization_key((x, make_helper(x))),
            _pallas_static_add_tensors.specialization_key(
                (x, make_helper(independent))
            ),
        )

    def test_tensor_default_is_normalized_before_fast_dispatch(self) -> None:
        default = torch.empty(64, dtype=torch.float32)

        @helion.kernel(backend="pallas", static_shapes=True)
        def with_default(x: torch.Tensor, y: torch.Tensor = default) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile] + y[tile]
            return out

        independent = torch.empty_like(default)
        independent_args = with_default.normalize_args(independent)
        aliased_args = with_default.normalize_args(default)
        self.assertNotEqual(
            with_default.specialization_key(independent_args),
            with_default.specialization_key(aliased_args),
        )

        fast_key = with_default._fast_dispatch_key(independent_args)
        assert fast_key is not None
        sentinel = object()
        bound = Mock()
        bound._run = Mock(return_value=sentinel)
        with_default._dispatch_cache[fast_key] = bound
        self.assertIs(with_default(independent), sentinel)
        bound._run.assert_called_once_with(*independent_args)

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
