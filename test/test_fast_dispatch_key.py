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
6. The per-prefix schema memo (``Kernel._fast_dispatch_schemas``) leaves the
   key's value unchanged while skipping the base specialization key and every
   extractor exact metadata implies.
7. A prepared call's extra guards (``_partition_prepared_extra_guards``) drop
   the same extractors, looking through omitted-default aliases.

The key is pure argument-metadata bookkeeping, so it needs no compilation and
runs on CPU-only bots.
"""

from __future__ import annotations

import dataclasses
import enum
import gc
import importlib
import operator
import types
from typing import TYPE_CHECKING
from typing import NamedTuple
import unittest
from unittest.mock import patch
import weakref

import pytest
import sympy
import torch
from torch._dynamo.source import LocalSource
from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource

import helion
from helion._argument_device import _ArgumentDeviceResolver
from helion._argument_device import _find_argument_device
from helion._compiler.compile_environment import (
    _concrete_tensor_satisfies_alignment_guard,
)
from helion._compiler.cute.grouped_static_tail import make_tail_extractor
from helion._compiler.cute.runtime_hooks import _CuteReductionExtent
from helion._compiler.cute.runtime_hooks import _CuteReductionExtentExtractor
import helion.language as hl
from helion.language.constexpr import ConstExpr
from helion.runtime.kernel import PreparedMetadataSpecializationExtractor
from helion.runtime.kernel import _contains_tensor
from helion.runtime.kernel import _GuardedInputFact
from helion.runtime.kernel import _IdentityKey
from helion.runtime.kernel import _implied_by_exact_metadata
from helion.runtime.kernel import _make_prepared_arg_guard
from helion.runtime.kernel import _RuntimeInputSpecializationExtractor
from helion.runtime.kernel import _SpecializationAlias
from helion.runtime.kernel import _TensorDescriptorAlignmentExtractor

pytestmark = pytest.mark.cpu_only

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Hashable
    from collections.abc import Sequence

kernel_module = importlib.import_module("helion.runtime.kernel")

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


class _Counted:
    """An extractor that counts its evaluations."""

    def __init__(self, fn: Callable[[Sequence[object]], Hashable]) -> None:
        self.fn = fn
        self.calls = 0

    def __call__(self, values: Sequence[object]) -> Hashable:
        self.calls += 1
        return self.fn(values)


def _make_extras_kernel(key: Callable[..., Hashable] | None = None) -> helion.Kernel:
    @helion.kernel(static_shapes=False, key=key)
    def scale(x: torch.Tensor, y: torch.Tensor, n: int) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile in hl.tile(x.size(0)):
            out[tile] = x[tile] * n
        return out

    return scale


def _storage_offset_is_zero_fact(name: str, params: tuple[str, ...]) -> Hashable:
    """A guard fact over a storage-offset symbol (exact metadata omits it)."""
    offset = sympy.Symbol(f"{name}_offset", integer=True)
    return PreparedMetadataSpecializationExtractor(
        _GuardedInputFact(
            offset,
            sympy.Integer(0),
            (
                (
                    offset,
                    TensorPropertySource(
                        LocalSource(name), TensorProperty.STORAGE_OFFSET
                    ),
                ),
            ),
            False,
            True,
            params,
        )
    )


def _publish(
    kernel: helion.Kernel,
    args: tuple[object, ...],
    extractors: Sequence[Callable[[Sequence[object]], Hashable]],
) -> tuple[Hashable, ...]:
    """Publish ``extractors`` as the schema of ``args``' signature the way a
    first bind does: into a missing slot, without a generation bump."""
    signature = kernel._base_specialization_key(args)
    kernel._specialize_extra.setdefault(signature, list(extractors))
    kernel._has_specialization_extras = True
    return signature


def _unmemoized_key_and_guards(
    kernel: helion.Kernel, args: tuple[object, ...]
) -> tuple[
    tuple[Hashable, ...] | None,
    tuple[tuple[Callable[[Sequence[object]], Hashable], Hashable], ...],
]:
    """The fast key and its extra guards without the memo: an explicit
    signature never reads it."""
    guards: list[tuple[Callable[[Sequence[object]], Hashable], Hashable]] = []
    key = kernel._fast_dispatch_key(
        args, signature=kernel._base_specialization_key(args), _extra_guards=guards
    )
    return key, tuple(guards)


def _unmemoized_key(
    kernel: helion.Kernel, args: tuple[object, ...]
) -> tuple[Hashable, ...] | None:
    """The fast key without the memo."""
    return _unmemoized_key_and_guards(kernel, args)[0]


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


class TestMemoizedFastDispatchKey(unittest.TestCase):
    def _extractors(self) -> dict[str, Callable[[Sequence[object]], Hashable]]:
        params = ("x", "y", "n")
        return {
            "size": PreparedMetadataSpecializationExtractor(
                _Counted(lambda values: values[0].size(0))  # type: ignore[union-attr]
            ),
            "x_offset": _storage_offset_is_zero_fact("x", params),
            "value": _Counted(operator.itemgetter(2)),
            "x_aligned": _TensorDescriptorAlignmentExtractor(
                _Counted(operator.itemgetter(0)), False
            ),
            "y_aligned": _TensorDescriptorAlignmentExtractor(
                operator.itemgetter(1), True
            ),
            "classifier": _RuntimeInputSpecializationExtractor(
                (operator.itemgetter(0),),
                lambda values: values[0].data_ptr() % 32,  # type: ignore[union-attr]
                frozenset(),
            ),
        }

    def test_implied_by_exact_metadata_predicate(self) -> None:
        params = ("x",)
        size = sympy.Symbol("x_size0", integer=True)
        x_size = TensorPropertySource(LocalSource("x"), TensorProperty.SIZE, 0)
        size_fact = _GuardedInputFact(
            size, sympy.Integer(1), ((size, x_size),), False, True, params
        )
        implied = (
            PreparedMetadataSpecializationExtractor(operator.itemgetter(0)),
            PreparedMetadataSpecializationExtractor(size_fact),
            # A CuTe reduction extent's power-of-2 bucket reads input sizes.
            PreparedMetadataSpecializationExtractor(
                _CuteReductionExtentExtractor(
                    _CuteReductionExtent(size, (size,), (x_size,)),
                    params,
                )
            ),
        )
        offset = sympy.Symbol("x_offset", integer=True)
        x_offset = TensorPropertySource(LocalSource("x"), TensorProperty.STORAGE_OFFSET)
        # One storage-offset source makes the whole fact per call.
        offset_equals_size_fact = PreparedMetadataSpecializationExtractor(
            _GuardedInputFact(
                offset - size,
                sympy.Integer(0),
                ((offset, x_offset), (size, x_size)),
                False,
                True,
                params,
            )
        )
        per_call = (
            _storage_offset_is_zero_fact("x", params),
            offset_equals_size_fact,
            _TensorDescriptorAlignmentExtractor(operator.itemgetter(0), False),
            operator.itemgetter(0),
            _RuntimeInputSpecializationExtractor(
                (operator.itemgetter(0),), operator.itemgetter(0), frozenset()
            ),
            _SpecializationAlias({}, (), ()),
            # A CuTe grouped static tail reads an int tensor's contents.
            make_tail_extractor(("cute_grouped_static_tail", 0, 2, 64, None, None)),
        )
        for extractor in implied:
            self.assertTrue(_implied_by_exact_metadata(extractor))
        for extractor in per_call:
            self.assertFalse(_implied_by_exact_metadata(extractor))

    def test_alignment_extractor_matches_concrete_guard(self) -> None:
        storage = torch.empty(80)
        half_storage = torch.empty(80, dtype=torch.float16)
        eight_byte_aligned = storage[2:66]
        two_byte_aligned = half_storage[1:65]
        self.assertEqual(eight_byte_aligned.data_ptr() % 16, 8)
        self.assertEqual(two_byte_aligned.data_ptr() % 16, 2)
        views = (
            storage[:64],
            storage[1:65],
            eight_byte_aligned,
            storage[4:68],
            two_byte_aligned,
            half_storage[8:72],
            torch.nn.Parameter(storage[4:68].clone()),
            torch.nn.Parameter(torch.empty(64)),
        )
        for requires_zero_storage_offset in (False, True):
            extractor = _TensorDescriptorAlignmentExtractor(
                operator.itemgetter(0), requires_zero_storage_offset
            )
            for view in views:
                result = extractor((view,))
                self.assertIs(
                    result,
                    _concrete_tensor_satisfies_alignment_guard(
                        view, requires_zero_storage_offset
                    ),
                )

    def _flat_matrix(self) -> tuple[tuple[object, ...], ...]:
        """Arguments for ``_make_extras_kernel``: exact tensors and
        Parameters, input aliasing, specialized scalars and views whose exact
        metadata matches but whose pointer alignment and storage offset
        differ."""
        storage = torch.empty(160)
        x = torch.empty(64)
        misaligned = storage[1:65]
        aligned_offset = storage[4:68]
        parameter = torch.nn.Parameter(torch.empty(64))
        y = torch.empty(8, 8)
        y_offset = storage[16:80].view(8, 8)
        self.assertEqual(misaligned.stride(), x.stride())
        self.assertNotEqual(misaligned.data_ptr() % 16, 0)
        self.assertEqual(aligned_offset.data_ptr() % 16, 0)
        return (
            (x, y, 2),
            (x, y, 3),
            (misaligned, y, 2),
            (aligned_offset, y, 2),
            (x, y_offset, 2),
            (parameter, y, 2),
            (x, x, 2),
            (y, y, 2),
        )

    def _container_kernel(self) -> tuple[helion.Kernel, tuple[tuple[object, ...], ...]]:
        """A kernel over nested containers with a published schema, and
        arguments for it."""

        @helion.kernel(static_shapes=False)
        def total(xs: list[torch.Tensor], scales: dict[str, int]) -> torch.Tensor:
            out = torch.zeros_like(xs[0])
            for tile in hl.tile(out.size(0)):
                out[tile] = xs[0][tile] * scales["a"]
            return out

        storage = torch.empty(80)
        a = torch.empty(64)
        misaligned = storage[1:65]
        extractors = [
            PreparedMetadataSpecializationExtractor(
                lambda values: tuple(t.size(0) for t in values[0])  # type: ignore[union-attr]
            ),
            lambda values: values[1]["a"],  # type: ignore[index]
            _TensorDescriptorAlignmentExtractor(
                lambda values: values[0][0],  # type: ignore[index]
                True,
            ),
        ]
        matrix = (
            ([a, a], {"a": 1}),
            ([a, a], {"a": 2}),
            ([misaligned, a], {"a": 1}),
            ([a, torch.empty(64)], {"a": 1}),
            ((a, a), {"a": 1}),
            ([torch.nn.Parameter(torch.empty(64)), a], {"a": 1}),
        )
        for args in matrix:
            _publish(total, args, extractors)
        return total, matrix

    def _indexless_device_kernel(self) -> helion.Kernel:
        """A kernel taking a device argument, with a schema published for the
        index-less and the indexed CUDA device."""

        @helion.kernel(static_shapes=False)
        def fill(device: torch.device, x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile]
            return out

        x = torch.empty(64)
        for device in (torch.device("cuda"), torch.device("cuda", 0)):
            _publish(fill, (device, x), [operator.itemgetter(0)])
        return fill

    def test_memoized_key_matches_unmemoized_key(self) -> None:
        """Across the flat argument matrix, the memoized key equals the
        unmemoized one on the miss and on every hit."""
        matrix = self._flat_matrix()
        for key in (None, lambda x, y, n: x.size(0) > 32):
            kernel = _make_extras_kernel(key)
            for args in matrix:
                _publish(kernel, args, list(self._extractors().values()))
            for _ in range(3):
                keys = []
                for args in matrix:
                    fast_key = kernel._fast_dispatch_key(args)
                    self.assertEqual(fast_key, _unmemoized_key(kernel, args))
                    keys.append(fast_key)
                # A Parameter keys like a plain tensor; every other row
                # differs from (x, y, 2) in one key component.
                self.assertEqual(keys[5], keys[0])
                self.assertEqual(len(set(keys)), len(keys) - 1)
            self.assertEqual(
                len(kernel._fast_dispatch_schemas),  # type: ignore[attr-defined]
                len({kernel._fast_dispatch_key(args)[:-1] for args in matrix}),
            )

    def test_memoized_key_matches_unmemoized_key_for_containers(self) -> None:
        total, matrix = self._container_kernel()
        for _ in range(2):
            keys = [total._fast_dispatch_key(args) for args in matrix]
            self.assertEqual(keys, [_unmemoized_key(total, args) for args in matrix])
            # A Parameter in a list keys like a plain tensor there.
            self.assertEqual(keys[5], keys[3])
            self.assertEqual(len(set(keys)), len(keys) - 1)
        self.assertEqual(len(total._fast_dispatch_schemas), 3)  # type: ignore[attr-defined]

    def test_hit_runs_only_per_call_extractors(self) -> None:
        kernel = _make_extras_kernel()
        extractors = self._extractors()
        args = (torch.empty(64), torch.empty(8, 8), 2)
        _publish(kernel, args, list(extractors.values()))
        miss_key = kernel._fast_dispatch_key(args)
        size = extractors["size"].extractor  # type: ignore[attr-defined]
        value = extractors["value"]
        x_aligned = extractors["x_aligned"].extract_tensor  # type: ignore[attr-defined]
        self.assertEqual((size.calls, value.calls, x_aligned.calls), (1, 1, 1))
        with (
            patch.object(
                kernel,
                "_base_specialization_key",
                side_effect=AssertionError("memoized hit rebuilt the base key"),
            ),
            patch.object(
                kernel_module,
                "_input_tensor_aliases",
                side_effect=AssertionError("flat arguments re-walked for aliases"),
            ),
            patch.object(
                kernel_module,
                "_tensor_aliases",
                wraps=kernel_module._tensor_aliases,
            ) as tensor_aliases,
        ):
            for _ in range(3):
                self.assertEqual(kernel._fast_dispatch_key(args), miss_key)
        self.assertEqual(tensor_aliases.call_count, 3)
        self.assertEqual((size.calls, value.calls, x_aligned.calls), (1, 4, 4))

    def test_key_fn_runs_once_per_key_on_miss_and_hit(self) -> None:
        calls = 0

        def key(x: torch.Tensor, y: torch.Tensor, n: int) -> bool:
            nonlocal calls
            calls += 1
            return x.size(0) > 32

        kernel = _make_extras_kernel(key)
        args = (torch.empty(64), torch.empty(8, 8), 2)
        _publish(kernel, args, list(self._extractors().values()))
        for _ in range(3):
            calls = 0
            fast_key = kernel._fast_dispatch_key(args)
            self.assertEqual(calls, 1)
            assert fast_key is not None
            self.assertIs(fast_key[len(args) + 1], True)

        def raising_key(x: torch.Tensor, y: torch.Tensor, n: int) -> bool:
            nonlocal calls
            calls += 1
            raise RuntimeError("key failed")

        schemas = dict(kernel._fast_dispatch_schemas)  # type: ignore[attr-defined]
        calls = 0
        with (
            patch.object(kernel, "_key_fn", raising_key),
            self.assertRaisesRegex(RuntimeError, "key failed"),
        ):
            kernel._fast_dispatch_key(args)
        self.assertEqual(calls, 1)
        self.assertEqual(kernel._fast_dispatch_schemas, schemas)  # type: ignore[attr-defined]

    def test_unpublished_schema_is_not_memoized(self) -> None:
        kernel = _make_extras_kernel()
        args = (torch.empty(64), torch.empty(8, 8), 2)
        kernel._has_specialization_extras = True  # type: ignore[attr-defined]
        prefix = kernel._fast_dispatch_key(args)
        assert prefix is not None
        self.assertFalse(kernel._fast_dispatch_schemas)  # type: ignore[attr-defined]
        # First publication fills the slot without a generation bump.
        _publish(kernel, args, [operator.itemgetter(2)])
        self.assertEqual(kernel._fast_dispatch_key(args), (*prefix, (2,)))
        self.assertEqual(len(kernel._fast_dispatch_schemas), 1)  # type: ignore[attr-defined]

    def test_stale_generation_rebuilds_schema(self) -> None:
        kernel = _make_extras_kernel()
        args = (torch.empty(64), torch.empty(8, 8), 2)
        signature = _publish(kernel, args, [operator.itemgetter(2)])
        prefix = kernel._fast_dispatch_key(args)[:-1]  # type: ignore[index]
        (schema,) = kernel._fast_dispatch_schemas.values()  # type: ignore[attr-defined]
        # A schema extension: replace the published list, bump the generation.
        kernel._specialize_extra[signature] = [  # type: ignore[attr-defined]
            operator.itemgetter(2),
            PreparedMetadataSpecializationExtractor(
                lambda values: values[0].size(0)  # type: ignore[union-attr]
            ),
        ]
        kernel._specialization_generation += 1  # type: ignore[attr-defined]
        self.assertEqual(kernel._fast_dispatch_key(args), (*prefix, (2, 64)))
        rebuilt = kernel._fast_dispatch_schemas[prefix]  # type: ignore[attr-defined]
        self.assertIsNot(rebuilt, schema)
        self.assertEqual(
            rebuilt.generation,
            kernel._specialization_generation,  # type: ignore[attr-defined]
        )

    def test_settings_change_rebuilds_schema(self) -> None:
        kernel = _make_extras_kernel()
        args = (torch.empty(64), torch.empty(8, 8), 2)
        _publish(kernel, args, [operator.itemgetter(2)])
        self.assertEqual(kernel._fast_dispatch_key(args)[-1], (2,))  # type: ignore[index]
        # The static-shape signature has no published schema yet.
        kernel.settings.static_shapes = True
        self.assertEqual(kernel._fast_dispatch_key(args), _unmemoized_key(kernel, args))
        self.assertNotEqual(kernel._fast_dispatch_key(args)[-1], (2,))  # type: ignore[index]

    def test_indexless_device_argument_is_not_memoized(self) -> None:
        fill = self._indexless_device_kernel()
        x = torch.empty(64)
        indexless = (torch.device("cuda"), x)
        # The argument loop decides: such a call costs no memo lookup, schema
        # build or device walk on every launch.
        with patch.object(
            fill,
            "_memoized_fast_dispatch_key",
            side_effect=AssertionError("index-less device entered the memo"),
        ):
            for _ in range(2):
                self.assertEqual(
                    fill._fast_dispatch_key(indexless),
                    _unmemoized_key(fill, indexless),
                )
        self.assertFalse(fill._fast_dispatch_schemas)  # type: ignore[attr-defined]
        # An indexed device resolves the same way on every call.
        indexed = (torch.device("cuda", 0), x)
        self.assertEqual(
            fill._fast_dispatch_key(indexed), _unmemoized_key(fill, indexed)
        )
        self.assertEqual(len(fill._fast_dispatch_schemas), 1)  # type: ignore[attr-defined]

    def test_indexless_device_behind_a_tensor_is_not_memoized(self) -> None:
        """``_find_device`` finds the tensor first, but the argument loop stays
        conservative rather than tracking which device comes first."""

        @helion.kernel(static_shapes=False)
        def fill(x: torch.Tensor, device: torch.device) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile]
            return out

        args = (torch.empty(64), torch.device("cuda"))
        _publish(fill, args, [operator.itemgetter(1)])
        with patch.object(
            fill,
            "_memoized_fast_dispatch_key",
            side_effect=AssertionError("index-less device entered the memo"),
        ):
            self.assertEqual(fill._fast_dispatch_key(args), _unmemoized_key(fill, args))

    def test_indexless_device_in_container_is_not_memoized(self) -> None:
        @helion.kernel(static_shapes=False)
        def fill(items: list[object]) -> torch.Tensor:
            out = torch.empty_like(items[1])
            for tile in hl.tile(out.size(0)):
                out[tile] = items[1][tile]
            return out

        x = torch.empty(64)
        indexless = ([torch.device("cuda"), x],)
        indexed = ([torch.device("cuda", 0), x],)
        for args in (indexless, indexed):
            _publish(fill, args, [lambda values: values[0][0]])  # type: ignore[index]
        for _ in range(2):
            self.assertEqual(
                fill._fast_dispatch_key(indexless), _unmemoized_key(fill, indexless)
            )
        self.assertFalse(fill._fast_dispatch_schemas)  # type: ignore[attr-defined]
        # (With the CPU tensor that is two devices: not memoized either.)
        self.assertEqual(
            fill._fast_dispatch_key(indexed), _unmemoized_key(fill, indexed)
        )
        self.assertFalse(fill._fast_dispatch_schemas)  # type: ignore[attr-defined]
        cpu = ([torch.device("cpu"), x],)
        _publish(fill, cpu, [lambda values: values[0][0]])  # type: ignore[index]
        self.assertEqual(fill._fast_dispatch_key(cpu), _unmemoized_key(fill, cpu))
        self.assertEqual(len(fill._fast_dispatch_schemas), 1)  # type: ignore[attr-defined]

    def test_argument_device_is_fixed(self) -> None:
        is_fixed = kernel_module._argument_device_is_fixed
        x = torch.empty(64)
        meta = torch.empty(64, device=torch.device("meta"))
        cuda, cuda0 = torch.device("cuda"), torch.device("cuda", 0)
        for args, fixed in (
            ((x, [x, (torch.device("cpu"),)], {"a": x}), True),
            (([cuda0], {"a": (cuda0,)}), True),
            (([cuda],), False),
            ((x, [meta]), False),
            (({"a": x, "b": meta},), False),
            (([cuda0, cuda],), False),
            # ``_find_device`` skips what a ConstExpr wraps.
            ((x, [ConstExpr(meta)]), True),
        ):
            with self.subTest(args=args):
                self.assertIs(is_fixed(args), fixed)

    def test_dict_device_insertion_order_is_not_memoized(self) -> None:
        """The prefix sorts dict entries while ``_find_device`` follows their
        insertion order, so with tensors on two devices two dicts share a
        prefix but not a base signature; neither may be memoized."""

        @helion.kernel(static_shapes=False)
        def first(d: dict[str, torch.Tensor]) -> torch.Tensor:
            out = torch.empty_like(d["a"])
            for tile in hl.tile(out.size(0)):
                out[tile] = d["a"][tile]
            return out

        cpu, meta = torch.empty(64), torch.empty(64, device=torch.device("meta"))
        cpu_first = ({"a": cpu, "b": meta},)
        meta_first = ({"b": meta, "a": cpu},)
        self.assertNotEqual(
            first._base_specialization_key(cpu_first),
            first._base_specialization_key(meta_first),
        )
        _publish(first, cpu_first, [lambda values: "cpu first"])
        _publish(first, meta_first, [lambda values: "meta first"])
        for _ in range(2):
            keys = []
            for args in (cpu_first, meta_first):
                keys.append(first._fast_dispatch_key(args))
                self.assertEqual(keys[-1], _unmemoized_key(first, args))
            self.assertEqual(keys[0][:-1], keys[1][:-1])  # type: ignore[index]
            self.assertNotEqual(keys[0][-1], keys[1][-1])  # type: ignore[index]
        self.assertFalse(first._fast_dispatch_schemas)  # type: ignore[attr-defined]
        # On one device, insertion order cannot change the signature.
        one_device = ({"b": torch.empty(64), "a": cpu},)
        for _ in range(2):
            self.assertEqual(
                first._fast_dispatch_key(one_device),
                _unmemoized_key(first, one_device),
            )
        self.assertEqual(len(first._fast_dispatch_schemas), 1)  # type: ignore[attr-defined]

    def test_is_distributed_keys_distinct_memo_entries(self) -> None:
        kernel = _make_extras_kernel()
        args = (torch.empty(64), torch.empty(8, 8), 2)
        for is_distributed in (False, True):
            signature = kernel._base_specialization_key(
                args, is_distributed=is_distributed
            )
            kernel._specialize_extra[signature] = [  # type: ignore[attr-defined]
                PreparedMetadataSpecializationExtractor(
                    lambda values, is_distributed=is_distributed: is_distributed
                ),
                operator.itemgetter(2),
            ]
        kernel._has_specialization_extras = True  # type: ignore[attr-defined]
        for _ in range(2):
            keys = []
            for is_distributed in (False, True):
                keys.append(
                    kernel._fast_dispatch_key(args, is_distributed=is_distributed)
                )
                self.assertEqual(
                    keys[-1],
                    kernel._fast_dispatch_key(
                        args,
                        is_distributed=is_distributed,
                        signature=kernel._base_specialization_key(
                            args, is_distributed=is_distributed
                        ),
                    ),
                )
            self.assertEqual([key[-1] for key in keys], [(False, 2), (True, 2)])  # type: ignore[index]
        self.assertEqual(
            [prefix[len(args)] for prefix in kernel._fast_dispatch_schemas],  # type: ignore[attr-defined]
            [False, True],
        )

    def test_compiling_neither_reads_nor_writes_memo(self) -> None:
        """Dynamo would guard on the memo's contents and replay its writes."""
        kernel = _make_extras_kernel()
        args = (torch.empty(64), torch.empty(8, 8), 2)
        _publish(kernel, args, [operator.itemgetter(2)])
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            compiling_key = kernel._fast_dispatch_key(args)
        self.assertFalse(kernel._fast_dispatch_schemas)  # type: ignore[attr-defined]
        self.assertEqual(compiling_key, _unmemoized_key(kernel, args))
        self.assertEqual(kernel._fast_dispatch_key(args), compiling_key)
        (schema,) = kernel._fast_dispatch_schemas.values()  # type: ignore[attr-defined]
        assert compiling_key is not None
        poisoned = (*compiling_key[:-1], ("poison",))
        schema.results = ("poison",)
        schema.full_key = poisoned
        self.assertEqual(kernel._fast_dispatch_key(args), poisoned)
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            self.assertEqual(kernel._fast_dispatch_key(args), compiling_key)

    def test_constexpr_tensor_argument_is_not_memoized(self) -> None:
        @helion.kernel(static_shapes=False)
        def add(x: torch.Tensor, w: hl.constexpr) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile]
            return out

        x = torch.empty(64)
        first, second = torch.empty(4), torch.empty(4)
        size = _Counted(lambda values: values[1].size(0))  # type: ignore[union-attr]
        for w in (first, second):
            _publish(add, (x, w), [PreparedMetadataSpecializationExtractor(size)])
        for _ in range(2):
            for w in (first, second):
                self.assertEqual(
                    add._fast_dispatch_key((x, w)), _unmemoized_key(add, (x, w))
                )
        self.assertFalse(add._fast_dispatch_schemas)  # type: ignore[attr-defined]

    def _omitted_default_kernel(
        self,
        trailing_default: object,
        canonical: Sequence[Callable[[Sequence[object]], Hashable]] | None,
    ) -> helion.Kernel:
        """A kernel whose omitted-default signature aliases the canonical
        signature for ``(x, trailing_default)``, as ``Kernel._bind`` sets up."""

        @helion.kernel(static_shapes=False)
        def add(x: torch.Tensor, value: object = None) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile]
            return out

        x = torch.empty(64)
        canonical_signature = add._base_specialization_key((x, trailing_default))
        if canonical is not None:
            add._specialize_extra[canonical_signature] = list(canonical)  # type: ignore[attr-defined]
        alias_signature = add._base_specialization_key((x,))
        alias = _SpecializationAlias(
            add._specialize_extra,  # type: ignore[attr-defined]
            canonical_signature,
            (trailing_default,),
        )
        add._specialization_aliases[alias_signature] = alias  # type: ignore[attr-defined]
        add._specialize_extra[alias_signature] = [alias]  # type: ignore[attr-defined]
        add._has_specialization_extras = True  # type: ignore[attr-defined]
        return add

    def test_omitted_default_alias_looks_through_to_canonical_schema(self) -> None:
        size = _Counted(lambda values: values[0].size(0))  # type: ignore[union-attr]
        default = _Counted(operator.itemgetter(1))
        extract = _Counted(operator.itemgetter(0))
        add = self._omitted_default_kernel(
            1e-5,
            [
                PreparedMetadataSpecializationExtractor(size),
                _TensorDescriptorAlignmentExtractor(extract, False),
                default,
            ],
        )
        storage = torch.empty(80)
        aligned, misaligned = torch.empty(64), storage[1:65]
        keys = set()
        for _ in range(3):
            for x in (aligned, misaligned):
                fast_key = add._fast_dispatch_key((x,))
                self.assertEqual(fast_key, _unmemoized_key(add, (x,)))
                keys.add(fast_key)
        self.assertEqual(len(keys), 2)
        # Six memoized keys and six reference keys: the canonical metadata
        # extractor ran for the reference keys and the one miss.
        self.assertEqual((size.calls, default.calls, extract.calls), (7, 12, 12))

    def test_omitted_mutable_default_alias_stays_per_call(self) -> None:
        size = _Counted(lambda values: values[0].size(0))  # type: ignore[union-attr]
        add = self._omitted_default_kernel(
            [1], [PreparedMetadataSpecializationExtractor(size)]
        )
        x = torch.empty(64)
        for _ in range(3):
            self.assertEqual(add._fast_dispatch_key((x,)), _unmemoized_key(add, (x,)))
        self.assertEqual(size.calls, 6)

    def test_alias_without_canonical_schema_fails_closed(self) -> None:
        add = self._omitted_default_kernel(1e-5, None)
        x = torch.empty(64)
        for _ in range(2):
            with self.assertRaises(KeyError):
                add._fast_dispatch_key((x,))
        self.assertFalse(add._fast_dispatch_schemas)  # type: ignore[attr-defined]
        with self.assertRaises(KeyError):
            _unmemoized_key(add, (x,))

    def test_prepared_guards_keep_facts_exact_metadata_does_not_imply(
        self,
    ) -> None:
        """The prepared argument guard pins sizes and strides, not storage
        offsets: a guard fact over a storage offset is checked every call."""
        params = ("x",)
        size = sympy.Symbol("x_size0", integer=True)
        size_fact = PreparedMetadataSpecializationExtractor(
            _GuardedInputFact(
                size,
                sympy.Integer(1),
                (
                    (
                        size,
                        TensorPropertySource(LocalSource("x"), TensorProperty.SIZE, 0),
                    ),
                ),
                False,
                True,
                params,
            )
        )
        offset_fact = _storage_offset_is_zero_fact("x", params)
        args = (torch.empty(64),)
        always_check, fact_guards = kernel_module._partition_prepared_extra_guards(
            args, ((size_fact, size_fact(args)), (offset_fact, offset_fact(args)))
        )
        self.assertEqual(always_check, ((offset_fact, True),))
        self.assertEqual(fact_guards, ())

    def test_prepared_guards_look_through_omitted_default_alias(self) -> None:
        x = torch.empty(64)
        size = PreparedMetadataSpecializationExtractor(
            _Counted(lambda values: values[0].size(0))  # type: ignore[union-attr]
        )
        default = operator.itemgetter(1)

        def partition(
            trailing_default: object,
            canonical: Sequence[Callable[[Sequence[object]], Hashable]] | None,
            expected: Hashable | None = None,
        ) -> tuple[_SpecializationAlias, tuple[object, ...]]:
            add = self._omitted_default_kernel(trailing_default, canonical)
            (alias,) = add._specialization_aliases.values()  # type: ignore[attr-defined]
            if expected is None:
                expected = alias((x,))
            always_check, fact_guards = kernel_module._partition_prepared_extra_guards(
                (x,), ((alias, expected),)
            )
            self.assertEqual(fact_guards, ())
            return alias, always_check

        # Exact metadata implies every canonical extractor.
        _alias, always_check = partition(1e-5, [size], (64,))
        self.assertEqual(always_check, ())
        # Only the default's value is re-read, on the normalized arguments;
        # the implied slot comes from the expected value, unevaluated.
        calls = size.extractor.calls  # type: ignore[attr-defined]
        _alias, always_check = partition(1e-5, [size, default], (64, 1e-5))
        ((dispatch, expected),) = always_check
        self.assertIsInstance(dispatch, kernel_module._SpecializationAliasDispatch)
        self.assertEqual(dispatch.per_call, ((1, default),))  # type: ignore[attr-defined]
        self.assertEqual(dispatch((x,)), expected)  # type: ignore[operator]
        self.assertEqual(size.extractor.calls, calls)  # type: ignore[attr-defined]
        # The whole alias stays for a mutable default, a missing canonical
        # schema, and (failing closed) an expected value that the re-read
        # slots do not reproduce or that does not have the canonical shape.
        for trailing_default, canonical, expected in (
            ([1], [size], None),
            (1e-5, None, (64,)),
            (1e-5, [size, default], (64, 2e-5)),
            (1e-5, [size, default], (64,)),
            (1e-5, [size], 64),
        ):
            alias, always_check = partition(trailing_default, canonical, expected)
            self.assertEqual(len(always_check), 1)
            self.assertIs(always_check[0][0], alias)  # type: ignore[index]

    def test_alias_per_call_split_follows_replaced_canonical_list(self) -> None:
        """The split of the canonical schema is cached per published list,
        which a schema extension replaces rather than mutates."""
        size = PreparedMetadataSpecializationExtractor(
            lambda values: values[0].size(0)  # type: ignore[union-attr]
        )
        default = operator.itemgetter(1)
        add = self._omitted_default_kernel(1e-5, [size])
        (alias,) = add._specialization_aliases.values()  # type: ignore[attr-defined]
        x = torch.empty(64)
        for _ in range(2):
            self.assertIsNone(kernel_module._prepared_alias_guard(alias, (64,), (x,)))
        add._specialize_extra[alias.canonical_signature] = [size, default]  # type: ignore[attr-defined]
        guard = kernel_module._prepared_alias_guard(alias, (64, 1e-5), (x,))
        self.assertIsInstance(guard, kernel_module._SpecializationAliasDispatch)
        self.assertEqual(guard.per_call, ((1, default),))  # type: ignore[union-attr]
        self.assertEqual(
            kernel_module._evaluate_specialization_alias(alias, (x,))[0], (64, 1e-5)
        )

    def _assert_guards_match_unmemoized(
        self,
        kernel: helion.Kernel,
        matrix: Sequence[tuple[object, ...]],
        *,
        memoized: bool = True,
    ) -> None:
        """On each prefix's miss and on every later hit, the extra guards
        collected with the key equal the unmemoized ones."""
        hits = set()
        for _ in range(3):
            for args in matrix:
                key, guards = _unmemoized_key_and_guards(kernel, args)
                assert key is not None
                hits.add(key[:-1] in kernel._fast_dispatch_schemas)  # type: ignore[attr-defined]
                entry = kernel._fast_dispatch_key_and_guards(args)  # type: ignore[attr-defined]
                assert entry is not None
                self.assertEqual(entry.key, key)
                # One guard per published extractor, holding the key's result.
                self.assertEqual(tuple(result for _fn, result in guards), key[-1])
                self.assertEqual(entry.extra_guards, guards)
        self.assertEqual(hits, {False, True} if memoized else {False})

    def test_extra_guards_match_unmemoized_guards(self) -> None:
        """A BoundKernel's direct prepared call guards on the extras its first
        call collected, always a memo miss; misses and hits alike must collect
        exactly the guards of the unmemoized key."""
        matrix = self._flat_matrix()
        for key in (None, lambda x, y, n: x.size(0) > 32):
            with self.subTest(keyed=key is not None):
                kernel = _make_extras_kernel(key)
                for args in matrix:
                    _publish(kernel, args, list(self._extractors().values()))
                self._assert_guards_match_unmemoized(kernel, matrix)
        with self.subTest("containers"):
            self._assert_guards_match_unmemoized(*self._container_kernel())
        storage = torch.empty(80)
        aligned, misaligned = torch.empty(64), storage[1:65]
        with self.subTest("omitted immutable default"):
            add = self._omitted_default_kernel(
                1e-5,
                [
                    PreparedMetadataSpecializationExtractor(
                        lambda values: values[0].size(0)  # type: ignore[union-attr]
                    ),
                    _TensorDescriptorAlignmentExtractor(operator.itemgetter(0), False),
                    operator.itemgetter(1),
                ],
            )
            self._assert_guards_match_unmemoized(add, ((aligned,), (misaligned,)))
        with self.subTest("omitted mutable default"):
            add = self._omitted_default_kernel(
                [1],
                [
                    PreparedMetadataSpecializationExtractor(
                        lambda values: values[0].size(0)  # type: ignore[union-attr]
                    )
                ],
            )
            self._assert_guards_match_unmemoized(add, ((aligned,),))
        with self.subTest("index-less device"):
            self._assert_guards_match_unmemoized(
                self._indexless_device_kernel(),
                ((torch.device("cuda"), aligned),),
                memoized=False,
            )


if __name__ == "__main__":
    unittest.main()
