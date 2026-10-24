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

import dataclasses
import enum
import gc
import types
from typing import NamedTuple
import unittest
import weakref

import torch

import helion
from helion._argument_device import _ArgumentDeviceResolver
from helion._argument_device import _find_argument_device
import helion.language as hl
from helion.language.constexpr import ConstExpr
from helion.runtime.kernel import _contains_tensor
from helion.runtime.kernel import _IdentityKey
from helion.runtime.kernel import _make_prepared_arg_guard

_MODULE_SCALE = 2


def _reads_module_global(t: torch.Tensor) -> torch.Tensor:
    return t * _MODULE_SCALE


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


def _flatten_key(key: object):
    if isinstance(key, (tuple, list)):
        yield key
        for item in key:
            yield from _flatten_key(item)
    else:
        yield key


class _Point(NamedTuple):
    a: int
    b: int


class _Mode(enum.Enum):
    A = 1
    B = 2


def _with_default(value: object):
    def apply(t: torch.Tensor, scale: object = value) -> torch.Tensor:
        return t

    return apply


def _with_kwdefault(value: object):
    def apply(t: torch.Tensor, *, scale: object = value) -> torch.Tensor:
        return t

    return apply


class TestFastDispatchKey(unittest.TestCase):
    def test_argument_device_discovery_preserves_precedence(self) -> None:
        cpu = torch.device("cpu")
        meta = torch.device("meta")
        cpu_tensor = torch.empty(1)
        meta_tensor = torch.empty(1, device=meta)
        cases = (
            ((cpu_tensor, meta_tensor), cpu),
            ((meta_tensor, cpu_tensor), meta),
            ((meta, cpu_tensor), meta),
            ((None, [meta_tensor], cpu_tensor), meta),
            (({"empty": None, "device": meta}, cpu_tensor), meta),
            ((ConstExpr(meta_tensor), cpu_tensor), cpu),
        )
        for index, (args, expected) in enumerate(cases):
            with self.subTest(case=index):
                self.assertEqual(_find_argument_device(args), expected)
                self.assertEqual(
                    _ArgumentDeviceResolver.from_values(args)(args), expected
                )

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

    def test_prepared_guard_does_not_retain_tensors_with_gc_disabled(self) -> None:
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            x = torch.empty(64)
            y = torch.empty(64)
            x_ref = weakref.ref(x)
            y_ref = weakref.ref(y)
            args = ({"pair": [x, y], "alias": (x,)},)
            guard = _make_prepared_arg_guard(_static_add1, args)
            self.assertTrue(guard(args))

            del args, x, y
            # Check before any cyclic collection, while the guard stays alive.
            self.assertIsNone(x_ref())
            self.assertIsNone(y_ref())

            replacement = torch.empty(64)
            self.assertTrue(
                guard(
                    ({"pair": [replacement, torch.empty(64)], "alias": (replacement,)},)
                )
            )
        finally:
            if gc_was_enabled:
                gc.enable()

    def test_prepared_guard_error_does_not_retain_tensors_with_gc_disabled(
        self,
    ) -> None:
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            x = torch.empty(64)
            x_ref = weakref.ref(x)
            with self.assertRaisesRegex(
                TypeError, "unsupported prepared-call argument"
            ):
                _make_prepared_arg_guard(
                    _static_add1, ({"tensor": x, "unsupported": object()},)
                )

            del x
            self.assertIsNone(x_ref())
        finally:
            if gc_was_enabled:
                gc.enable()

    def test_prepared_guard_preserves_nested_tensor_identity_relations(self) -> None:
        x = torch.empty(64)
        y = torch.empty(64)
        guard = _make_prepared_arg_guard(
            _static_add1, ({"pair": [x, y], "alias": (x,)},)
        )

        replacement = torch.empty_like(x)
        other = torch.empty_like(y)
        self.assertTrue(
            guard(({"pair": [replacement, other], "alias": (replacement,)},))
        )
        self.assertFalse(
            guard(({"pair": [replacement, replacement], "alias": (replacement,)},))
        )
        self.assertFalse(
            guard(({"pair": [replacement, other], "alias": (torch.empty_like(x),)},))
        )

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

    def test_function_argument_keys_by_code_object(self) -> None:
        """Callable arguments key by code object (as ``_function_key`` does), so
        a per-call rebuilt function object with the same code keeps the fast
        key, while a different function changes it."""

        def make_double():
            def double(t: torch.Tensor) -> torch.Tensor:
                return t * 2

            return double

        def triple(t: torch.Tensor) -> torch.Tensor:
            return t * 3

        x = torch.empty(64, dtype=torch.float32)
        first = make_double()
        second = make_double()
        self.assertIsNot(first, second)
        key_first = _static_add1._fast_dispatch_key((x, first))
        key_second = _static_add1._fast_dispatch_key((x, second))
        key_triple = _static_add1._fast_dispatch_key((x, triple))
        self.assertIsNotNone(key_first)
        self.assertEqual(key_first, key_second)
        self.assertNotEqual(key_first, key_triple)
        assert isinstance(key_first, tuple)
        self.assertEqual(key_first[1], (types.FunctionType, first.__code__))
        self.assertEqual(
            _static_add1._fast_dispatch_key((x, [first]))[1],  # type: ignore[index]
            (list, ((types.FunctionType, first.__code__),)),
        )
        builtin_key = _static_add1._fast_dispatch_key((x, torch.relu))
        assert isinstance(builtin_key, tuple)
        self.assertEqual(builtin_key[1], (types.BuiltinFunctionType, torch.relu))
        self.assertNotEqual(
            builtin_key, _static_add1._fast_dispatch_key((x, torch.sigmoid))
        )

    def test_function_closure_cells_refine_full_key(self) -> None:
        """Closure cells follow the per-type rules of top-level arguments:
        tensors contribute exact metadata (finer than the full key), runtime
        numbers contribute only their type (matching the full key)."""

        def closing_over(value: object):
            def apply(t: torch.Tensor) -> torch.Tensor:
                return t + value  # type: ignore[operator]

            return apply

        x = torch.empty(64, dtype=torch.float32)
        f32 = closing_over(torch.empty(8, dtype=torch.float32))
        f16 = closing_over(torch.empty(8, dtype=torch.float16))
        wide = closing_over(torch.empty(16, dtype=torch.float32))
        self.assertNotEqual(
            _dynamic_add_scalar.specialization_key((x, f32)),
            _dynamic_add_scalar.specialization_key((x, f16)),
        )
        self.assertNotEqual(
            _static_add1._fast_dispatch_key((x, f32)),
            _static_add1._fast_dispatch_key((x, f16)),
        )
        # Static shapes: the full key already separates the extents, and the
        # fast key must as well.
        self.assertNotEqual(
            _static_add1._fast_dispatch_key((x, f32)),
            _static_add1._fast_dispatch_key((x, wide)),
        )
        two = closing_over(2)
        three = closing_over(3)
        self.assertEqual(
            _dynamic_add_scalar.specialization_key((x, two)),
            _dynamic_add_scalar.specialization_key((x, three)),
        )
        self.assertEqual(
            _static_add1._fast_dispatch_key((x, two)),
            _static_add1._fast_dispatch_key((x, three)),
        )
        self.assertNotEqual(
            _static_add1._fast_dispatch_key((x, two)),
            _static_add1._fast_dispatch_key((x, closing_over(2.0))),
        )
        # Unsupported closure contents keep the slow path.
        self.assertIsNone(_static_add1._fast_dispatch_key((x, closing_over(object()))))
        # A function closing over a tensor pins the device on its own.
        self.assertIsNotNone(_dynamic_add_scalar._fast_dispatch_key((f32, 1)))

    def test_function_default_values_refine_the_keys(self) -> None:
        """Default values are baked into the traced kernel: a rebuilt function
        with a new default is a new specialization in the full key, the fast
        key and the prepared guard, while the same defaults share them; a
        default the constexpr rules cannot key is keyed by identity in the
        full key and keeps the fast path off."""

        def with_defaults(value: object, *, offset: object = 0):
            def apply(
                t: torch.Tensor, scale: object = value, *, shift: object = offset
            ) -> torch.Tensor:
                return t * scale + shift  # type: ignore[operator]

            return apply

        x = torch.empty(64, dtype=torch.float32)
        two = with_defaults(2)
        three = with_defaults(3)
        self.assertIs(two.__code__, three.__code__)
        full_two = _dynamic_add_scalar.specialization_key((x, two))
        self.assertEqual(
            full_two, _dynamic_add_scalar.specialization_key((x, with_defaults(2)))
        )
        self.assertNotEqual(
            full_two, _dynamic_add_scalar.specialization_key((x, three))
        )
        self.assertNotEqual(
            full_two, _dynamic_add_scalar.specialization_key((x, with_defaults(2.0)))
        )
        self.assertNotEqual(
            full_two,
            _dynamic_add_scalar.specialization_key((x, with_defaults(2, offset=1))),
        )
        fast_two = _static_add1._fast_dispatch_key((x, two))
        self.assertIsNotNone(fast_two)
        self.assertEqual(
            fast_two, _static_add1._fast_dispatch_key((x, with_defaults(2)))
        )
        self.assertNotEqual(fast_two, _static_add1._fast_dispatch_key((x, three)))
        self.assertNotEqual(
            fast_two, _static_add1._fast_dispatch_key((x, with_defaults(2.0)))
        )
        self.assertNotEqual(
            fast_two, _static_add1._fast_dispatch_key((x, with_defaults(2, offset=1)))
        )
        tensor = torch.empty(8)
        full_tensor = _dynamic_add_scalar.specialization_key((x, with_defaults(tensor)))
        assert isinstance(full_tensor, tuple)
        # The tensor default keys by identity; its position in the nested key
        # depends on the backend's own key entries, so search for it.
        self.assertTrue(
            any(
                isinstance(entry, _IdentityKey) and entry.value is tensor
                for entry in _flatten_key(full_tensor)
            ),
            full_tensor,
        )
        self.assertEqual(
            full_tensor,
            _dynamic_add_scalar.specialization_key((x, with_defaults(tensor))),
        )
        self.assertIsNone(_static_add1._fast_dispatch_key((x, with_defaults(tensor))))

        guard = _make_prepared_arg_guard(_dynamic_add_scalar, (x, two))
        self.assertTrue(guard((x, two)))
        self.assertTrue(guard((x, with_defaults(2))))
        self.assertFalse(guard((x, three)))
        self.assertFalse(guard((x, with_defaults(2.0))))
        self.assertFalse(guard((x, with_defaults(2, offset=1))))
        with self.assertRaises(TypeError):
            _make_prepared_arg_guard(_dynamic_add_scalar, (x, with_defaults(tensor)))

    def test_tensor_holding_and_unhashable_defaults_key_by_identity(self) -> None:
        """A default that cannot key by its own equality (a container holding
        a tensor, whose ``==`` is not a bool; an unhashable object) keys by
        identity: the full key stays hashable, repeats for the same function
        object, differs for a rebuilt one, and the fast path stays off. A
        container the constexpr rules can key keeps keying by value."""

        @dataclasses.dataclass(frozen=True)
        class Holder:
            weight: torch.Tensor

            def act(self, t: torch.Tensor) -> torch.Tensor:
                return t

        x = torch.empty(64, dtype=torch.float32)
        tensor = torch.empty(8)
        holder = Holder(tensor)
        plain = object()
        # (a default, the same value rebuilt as another object)
        by_identity = (
            ([tensor], [tensor]),
            ({"scale": tensor}, {"scale": tensor}),
            ((tensor,), (tensor,)),
            (frozenset({tensor}), frozenset({tensor})),
            (holder, Holder(tensor)),
            (holder.act, holder.act),
            ([plain], [plain]),
            ({"scale": plain}, {"scale": plain}),
        )
        for factory in (_with_default, _with_kwdefault):
            for value, rebuilt in by_identity:
                self.assertIsNot(value, rebuilt)
                fn = factory(value)
                full = _dynamic_add_scalar.specialization_key((x, fn))
                hash(full)
                self.assertEqual(full, _dynamic_add_scalar.specialization_key((x, fn)))
                self.assertNotEqual(
                    full,
                    _dynamic_add_scalar.specialization_key((x, factory(rebuilt))),
                    value,
                )
                self.assertTrue(
                    any(
                        isinstance(entry, _IdentityKey) and entry.value is value
                        for entry in _flatten_key(full)
                    ),
                    full,
                )
                self.assertIsNone(_static_add1._fast_dispatch_key((x, fn)))
                with self.assertRaises(TypeError):
                    _make_prepared_arg_guard(_dynamic_add_scalar, (x, fn))
            self.assertEqual(
                _dynamic_add_scalar.specialization_key((x, factory([1, 2]))),
                _dynamic_add_scalar.specialization_key((x, factory([1, 2]))),
            )
            self.assertNotEqual(
                _dynamic_add_scalar.specialization_key((x, factory([1, 2]))),
                _dynamic_add_scalar.specialization_key((x, factory([1, 3]))),
            )
            self.assertIsNotNone(_static_add1._fast_dispatch_key((x, factory([1, 2]))))

    def test_cyclic_defaults_key_by_identity(self) -> None:
        """A default that reaches itself (a dataclass whose parent link is
        itself, hashable by identity or not, a tree whose kid points back at
        the root) keys by identity without walking the cycle: the full key
        repeats for the same object, differs for a rebuilt one, and a tensor
        behind the cycle is still found. The fast path stays off."""

        @dataclasses.dataclass
        class Node:
            parent: object = None

        @dataclasses.dataclass(eq=False)
        class Link:
            parent: object = None
            weight: object = None

        @dataclasses.dataclass
        class Tree:
            kids: list[object]
            up: object = None

        def self_parent(cls: type, **fields: object) -> object:
            value = cls(**fields)
            value.parent = value
            return value

        def tree() -> object:
            root = Tree([])
            root.kids.append(Tree([], root))
            return root

        x = torch.empty(64, dtype=torch.float32)
        tensor = torch.empty(8)
        # (builder, keyed through an _IdentityKey wrapper)
        builders = (
            (lambda: self_parent(Node), True),  # unhashable
            (lambda: self_parent(Link), False),  # hashable by identity
            (lambda: self_parent(Link, weight=tensor), True),  # holds a tensor
            (tree, True),  # unhashable, the cycle sits one level down
        )
        self.assertFalse(_contains_tensor(self_parent(Link), set()))
        self.assertTrue(_contains_tensor(self_parent(Link, weight=tensor), set()))
        self.assertFalse(_contains_tensor(tree(), set()))
        for factory in (_with_default, _with_kwdefault):
            for builder, wrapped in builders:
                value = builder()
                fn = factory(value)
                full = _dynamic_add_scalar.specialization_key((x, fn))
                hash(full)
                self.assertEqual(full, _dynamic_add_scalar.specialization_key((x, fn)))
                self.assertNotEqual(
                    full,
                    _dynamic_add_scalar.specialization_key((x, factory(builder()))),
                )
                self.assertEqual(
                    any(
                        isinstance(entry, _IdentityKey) and entry.value is value
                        for entry in _flatten_key(full)
                    ),
                    wrapped,
                    full,
                )
                self.assertIsNone(_static_add1._fast_dispatch_key((x, fn)))
                with self.assertRaises(TypeError):
                    _make_prepared_arg_guard(_dynamic_add_scalar, (x, fn))

    def test_hashable_tensor_free_defaults_key_by_value(self) -> None:
        """A hashable default with no tensor inside keys by its own equality,
        like a constexpr: a frozenset, namedtuple, frozen dataclass, enum
        member, bound method or complex rebuilt per call shares the key (no
        recompile, no pinned object per call) and a different value changes
        it; an object with default equality therefore keys by identity without
        a wrapper. The fast path stays off for all of them."""

        @dataclasses.dataclass(frozen=True)
        class Cfg:
            scale: float

        class Owner:
            def act(self, t: torch.Tensor) -> torch.Tensor:
                return t

        owner = Owner()
        x = torch.empty(64, dtype=torch.float32)
        pairs: list[tuple[object, object, object]] = [
            (frozenset({1, 2}), frozenset({1, 2}), frozenset({1, 3})),
            (_Point(1, 2), _Point(1, 2), _Point(2, 1)),
            (Cfg(0.5), Cfg(0.5), Cfg(0.25)),
            (_Mode.A, _Mode.A, _Mode.B),
            (owner.act, owner.act, Owner().act),
            (complex(1, 2), complex(1, 2), complex(1, 3)),
            ((_Point(1, 2), "a"), (_Point(1, 2), "a"), (_Point(1, 2), "b")),
        ]
        for factory in (_with_default, _with_kwdefault):
            for value, same, other in pairs:
                full = _dynamic_add_scalar.specialization_key((x, factory(value)))
                hash(full)
                self.assertEqual(
                    full,
                    _dynamic_add_scalar.specialization_key((x, factory(same))),
                    value,
                )
                self.assertNotEqual(
                    full,
                    _dynamic_add_scalar.specialization_key((x, factory(other))),
                    value,
                )
                self.assertFalse(
                    any(
                        isinstance(entry, _IdentityKey) for entry in _flatten_key(full)
                    ),
                    full,
                )
                self.assertIsNone(_static_add1._fast_dispatch_key((x, factory(value))))
                with self.assertRaises(TypeError):
                    _make_prepared_arg_guard(_dynamic_add_scalar, (x, factory(value)))
            # a namedtuple is not the plain tuple of its fields
            self.assertNotEqual(
                _dynamic_add_scalar.specialization_key((x, factory(_Point(1, 2)))),
                _dynamic_add_scalar.specialization_key((x, factory((1, 2)))),
            )
            plain = object()
            full = _dynamic_add_scalar.specialization_key((x, factory(plain)))
            self.assertEqual(
                full, _dynamic_add_scalar.specialization_key((x, factory(plain)))
            )
            self.assertNotEqual(
                full, _dynamic_add_scalar.specialization_key((x, factory(object())))
            )

    def test_function_defaults_key_like_function_arguments(self) -> None:
        """A function-valued default keys through ``_function_key``: a lambda
        rebuilt by a factory shares the full key (no recompile per factory
        call) and another code object changes it. The default's closure cells
        are baked into the kernel with it (``lift_closures`` never lifts a
        default), so they key by value: another cell value is another
        specialization. The fast path stays off."""

        def with_scale(scale_fn: object):
            def apply(t: torch.Tensor, scale: object = scale_fn) -> torch.Tensor:
                return scale(t)  # type: ignore[operator]

            return apply

        def make_double():
            return lambda t: t * 2

        def closing_over(value: object):
            return lambda t: t * value

        def closing_over_fn(inner: object):
            return lambda t: inner(t)  # type: ignore[operator]

        x = torch.empty(64, dtype=torch.float32)
        first = with_scale(make_double())
        second = with_scale(make_double())
        assert first.__defaults__ is not None and second.__defaults__ is not None
        self.assertIsNot(first.__defaults__[0], second.__defaults__[0])
        full = _dynamic_add_scalar.specialization_key((x, first))
        hash(full)
        self.assertEqual(full, _dynamic_add_scalar.specialization_key((x, second)))
        self.assertNotEqual(
            full, _dynamic_add_scalar.specialization_key((x, with_scale(lambda t: t)))
        )
        self.assertTrue(
            any(
                entry is first.__defaults__[0].__code__ for entry in _flatten_key(full)
            ),
            full,
        )
        scale_2 = _dynamic_add_scalar.specialization_key(
            (x, with_scale(closing_over(2)))
        )
        self.assertEqual(
            scale_2,
            _dynamic_add_scalar.specialization_key((x, with_scale(closing_over(2)))),
        )
        for other in (3, 2.0, 7.5, "2"):
            self.assertNotEqual(
                scale_2,
                _dynamic_add_scalar.specialization_key(
                    (x, with_scale(closing_over(other)))
                ),
                other,
            )
        scale_7_5 = _dynamic_add_scalar.specialization_key(
            (x, with_scale(closing_over(7.5)))
        )
        self.assertIn((float, 7.5), list(_flatten_key(scale_7_5)))
        self.assertNotEqual(
            scale_7_5,
            _dynamic_add_scalar.specialization_key((x, with_scale(closing_over(2.5)))),
        )
        # a function in a default function's cell is baked too, cells and all
        nested_2 = with_scale(closing_over_fn(closing_over(2)))
        self.assertEqual(
            _dynamic_add_scalar.specialization_key((x, nested_2)),
            _dynamic_add_scalar.specialization_key(
                (x, with_scale(closing_over_fn(closing_over(2))))
            ),
        )
        self.assertNotEqual(
            _dynamic_add_scalar.specialization_key((x, nested_2)),
            _dynamic_add_scalar.specialization_key(
                (x, with_scale(closing_over_fn(closing_over(3))))
            ),
        )
        # a cell of the argument itself stays a runtime input, keyed by type
        self.assertEqual(
            _dynamic_add_scalar.specialization_key((x, closing_over(2))),
            _dynamic_add_scalar.specialization_key((x, closing_over(3))),
        )
        self.assertIsNone(_static_add1._fast_dispatch_key((x, first)))
        with self.assertRaises(TypeError):
            _make_prepared_arg_guard(_dynamic_add_scalar, (x, first))

    def test_function_globals_are_not_keyed(self) -> None:
        """Module globals a callable reads follow the kernel's own global rule
        (read from the module when the kernel runs, specialized on the first
        call's values): rebinding one changes neither key nor the guard."""
        global _MODULE_SCALE
        x = torch.empty(64, dtype=torch.float32)
        full = _dynamic_add_scalar.specialization_key((x, _reads_module_global))
        fast = _static_add1._fast_dispatch_key((x, _reads_module_global))
        guard = _make_prepared_arg_guard(_dynamic_add_scalar, (x, _reads_module_global))
        saved = _MODULE_SCALE
        try:
            _MODULE_SCALE = saved + 1
            self.assertEqual(
                _dynamic_add_scalar.specialization_key((x, _reads_module_global)),
                full,
            )
            self.assertEqual(
                _static_add1._fast_dispatch_key((x, _reads_module_global)), fast
            )
            self.assertTrue(guard((x, _reads_module_global)))
        finally:
            _MODULE_SCALE = saved

    def test_constexpr_function_argument_returns_none(self) -> None:
        """A constexpr-annotated callable specializes on the raw object, which
        a code-object key would conflate."""

        def ident(t: torch.Tensor) -> torch.Tensor:
            return t

        x = torch.empty(64, dtype=torch.float32)
        self.assertIsNone(_dynamic_add_constexpr._fast_dispatch_key((x, ident)))
        self.assertIsNone(_dynamic_add_constexpr._fast_dispatch_key((x, (ident,))))
        with self.assertRaises(TypeError):
            _make_prepared_arg_guard(_dynamic_add_constexpr, (x, ident))

    def test_prepared_guard_function_argument(self) -> None:
        """The prepared guard re-reads the code object and every closure cell."""
        scale = torch.empty(8, dtype=torch.float32)

        def closing_over(value: object):
            def apply(t: torch.Tensor) -> torch.Tensor:
                return t * value  # type: ignore[operator]

            return apply

        def rebinding():
            cell_value: object = scale

            def apply(t: torch.Tensor) -> torch.Tensor:
                return t * cell_value  # type: ignore[operator]

            def rebind(new_value: object) -> None:
                nonlocal cell_value
                cell_value = new_value

            return apply, rebind

        x = torch.empty(64, dtype=torch.float32)
        prepared = closing_over(scale)
        guard = _make_prepared_arg_guard(_dynamic_add_scalar, (x, prepared))
        self.assertTrue(guard((x, prepared)))
        self.assertTrue(guard((x, closing_over(torch.empty_like(scale)))))
        self.assertFalse(guard((x, closing_over(scale.to(torch.float16)))))
        self.assertFalse(guard((x, closing_over(torch.empty(16)))))
        self.assertFalse(guard((x, closing_over(2))))
        self.assertFalse(guard((x, torch.relu)))
        self.assertFalse(guard((x, None)))

        apply, rebind = rebinding()
        guard = _make_prepared_arg_guard(_dynamic_add_scalar, (x, apply))
        self.assertTrue(guard((x, apply)))
        rebind(scale.to(torch.float16))
        self.assertFalse(guard((x, apply)))
        rebind(torch.empty_like(scale))
        self.assertTrue(guard((x, apply)))

        guard = _make_prepared_arg_guard(_dynamic_add_scalar, (x, torch.relu))
        self.assertTrue(guard((x, torch.relu)))
        self.assertFalse(guard((x, torch.sigmoid)))
        guard = _make_prepared_arg_guard(_dynamic_add_scalar, (x, closing_over(2)))
        self.assertTrue(guard((x, closing_over(3))))
        self.assertFalse(guard((x, closing_over(3.0))))


if __name__ == "__main__":
    unittest.main()
