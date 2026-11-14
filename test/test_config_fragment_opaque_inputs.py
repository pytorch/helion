from __future__ import annotations

from contextlib import nullcontext
from dataclasses import InitVar
from dataclasses import dataclass
from dataclasses import field
from typing import ClassVar
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.fragment_call import _fragment_call_input
from helion._compiler.fragment_call import _fragment_call_inputs
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing.binding_observation import observe_compiled_binding
from helion.autotuner import BooleanFragment
from helion.autotuner.config_fragment import ConfigSpecFragment
import helion.language as hl
from helion.runtime.kernel import BoundKernel
from helion.runtime.kernel import _function_key

pytestmark = pytest.mark.cpu_only

ANCHOR = BooleanFragment()
HELPER_CALLS = []


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_tensor_metadata_calls_do_not_register_fragment_guards(static_shapes):
    @helion.kernel(static_shapes=static_shapes, autotune_effort="none")
    def metadata(x):
        out = torch.empty_like(x)
        for row, column in hl.tile(x.shape):
            out[row, column] = x[row, column] + x.size(1) + x.dim()
        return out

    x = torch.empty(3, 5)[:, :4]
    with cpu_render_target():
        bound = metadata.bind((x,))
        bound.to_code()
    assert not any(
        key.startswith("fragment_call_input:")
        for key in bound.env.runtime_input_specializations
    )


def _inspect_fragment(fragment, boolean):
    HELPER_CALLS.append(fragment)
    if boolean:
        return fragment is ANCHOR
    return 2 if fragment is ANCHOR else 5


@helion.kernel(autotune_effort="none")
def _explicit_input(x, fragment: hl.constexpr, boolean: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        value = _inspect_fragment(fragment, boolean)
        if boolean:
            out[tile] = x[tile] + (2.0 if value else 5.0)
        else:
            out[tile] = x[tile] + value
    return out


@dataclass(frozen=True)
class _Receiver(ConfigSpecFragment):
    value: int = 7

    def default(self):
        return self.value

    def offset(self):
        HELPER_CALLS.append(self)
        return 2.0 if self is RECEIVER_ANCHOR else 5.0


RECEIVER_ANCHOR = _Receiver()


@helion.kernel(autotune_effort="none")
def _receiver_input(x, fragment: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + fragment.offset()
    return out


@helion.kernel(autotune_effort="none")
def _callable_input(x, fn):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + fn()
    return out


def _call_through(fn):
    return fn()


@helion.kernel(autotune_effort="none")
def _passed_callable_input(x, fn):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + _call_through(fn)
    return out


def _default_input(fragment):
    def fn(value=fragment):
        HELPER_CALLS.append(value)
        return 2.0 if value is ANCHOR else 5.0

    return fn


def _closure_input(fragment):
    captured = hl.constexpr(fragment)

    def fn():
        HELPER_CALLS.append(captured)
        return 2.0 if captured is ANCHOR else 5.0

    return fn


def _frozenset_default_input(fragment):
    wrapped = hl.constexpr(frozenset((fragment,)))

    def fn(values=wrapped):
        selected = next(iter(values.value))
        HELPER_CALLS.append(selected)
        return 2.0 if selected is ANCHOR else 5.0

    return fn


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize(
    "kind",
    [
        "bool",
        "int",
        "receiver",
        "default",
        "closure",
        "passed_callable",
        "frozenset_default",
    ],
)
def test_opaque_calls_guard_explicit_and_implicit_fragment_inputs(kind):
    tensor = torch.empty(17)
    if kind in ("bool", "int"):
        fn = _explicit_input.fn
        first = (tensor, ANCHOR, kind == "bool")
        second = (tensor, BooleanFragment(), kind == "bool")
    elif kind == "receiver":
        fn = _receiver_input.fn
        first = (tensor, RECEIVER_ANCHOR)
        second = (tensor, _Receiver())
    else:
        fn = (
            _passed_callable_input.fn
            if kind == "passed_callable"
            else _callable_input.fn
        )
        factory = (
            _frozenset_default_input
            if kind == "frozenset_default"
            else _closure_input
            if kind == "closure"
            else _default_input
        )
        first = (tensor, factory(ANCHOR))
        second = (tensor, factory(BooleanFragment()))
    kernel = helion.kernel(fn, autotune_effort="none")

    def offsets(bound):
        return [
            node.args[1]
            for graph in bound.host_function.device_ir.graphs
            for node in graph.graph.nodes
            if node.op == "call_function" and node.target is torch.ops.aten.add.Tensor
        ]

    def observe(bound, *args):
        before = tuple(bound.env.runtime_input_specializations.values())
        bound.to_code()
        assert tuple(bound.env.runtime_input_specializations.values()) == before
        bound._run = lambda *unused: bound
        return bound

    with cpu_render_target(), patch.object(BoundKernel, "__call__", observe):
        shared = kernel(*first)
        fresh = helion.kernel(fn, autotune_effort="none").bind(second)
        assert offsets(shared) == [2.0]
        assert offsets(fresh) == [5.0]
        distinct = kernel(*second)
        assert offsets(distinct) == [5.0]
        assert shared is not distinct
        assert distinct.to_code() == fresh.to_code()
        assert kernel.specialization_key(first) != kernel.specialization_key(second)
        assert repr(kernel.specialization_key(first)) == repr(
            kernel.specialization_key(second)
        )
        calls = len(HELPER_CALLS)
        # Custom equality is supported by the full binding cache, while the
        # other cases also exercise prepared fast dispatch.
        fast = kernel._fast_dispatch_key(first)
        assert (fast is None) == (kind == "receiver")
        with (
            patch.object(kernel, "bind", side_effect=AssertionError("cache miss"))
            if fast is not None
            else nullcontext()
        ):
            for args, bound in ((first, shared), (second, distinct)):
                assert kernel(*args) is bound
                assert kernel(*args) is bound
        assert len(HELPER_CALLS) == calls


def test_opaque_input_classifier_handles_cycles_and_never_calls_functions():
    other = BooleanFragment()
    cycle = []

    def fn(value=ANCHOR, *, option=other):
        raise AssertionError(cycle)

    cycle.append(fn)
    before = _fragment_call_input((fn,))
    assert [item.value for item in before] == [ANCHOR, other]
    fn.__defaults__ = (other,)
    assert _fragment_call_input((fn,)) != before
    assert [item.value for item in _fragment_call_input((RECEIVER_ANCHOR.offset,))] == [
        RECEIVER_ANCHOR
    ]

    def empty_cell():
        return other

    assert empty_cell.__closure__ is not None
    del empty_cell.__closure__[0].cell_contents
    assert _fragment_call_input((empty_cell,)) == ()
    assert not _fragment_call_inputs((empty_cell,))[1]
    empty_cell.__closure__[0].cell_contents = hl.constexpr(ANCHOR)
    assert _fragment_call_inputs((empty_cell,))[1]


def _plain_offset():
    return 5.0


@helion.kernel(autotune_effort="none")
def _callable_default_input(x, fn=_plain_offset):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + fn()
    return out


@helion.kernel(autotune_effort="none")
def _constexpr_callable_input(x, fn: hl.constexpr):
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile] + fn()
    return out


def _assert_offset(bound, expected):
    assert [
        node.args[1]
        for graph in bound.host_function.device_ir.graphs
        for node in graph.graph.nodes
        if node.op == "call_function" and node.target is torch.ops.aten.add.Tensor
    ] == [expected]
    bound.to_code()


def _has_callable_guard(bound):
    return "fragment_call_input:fn" in bound.env.runtime_input_specializations


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("explicit", [False, True])
def test_fragment_free_function_uses_existing_dispatch_key(explicit):
    x = torch.empty(17)
    kernel = helion.kernel(_callable_default_input.fn, autotune_effort="none")
    args = (x, _plain_offset) if explicit else (x,)
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "__call__", observe_compiled_binding),
    ):
        bound = kernel(*args)
        _assert_offset(bound, 5.0)
        assert not _has_callable_guard(bound)
        # Backend alignment guards may still contribute nonempty extras.
        extras = kernel._stable_specialization_extra_results(
            (x, _plain_offset), kernel._base_specialization_key((x, _plain_offset))
        )
        assert extras is not None and () not in extras
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            assert kernel(*args) is bound
            assert kernel(*args) is bound


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("kind", ["default", "kwdefault", "closure"])
def test_empty_function_input_still_rebinds_when_fragment_is_added(kind):
    if kind == "default":
        fn = _default_input(None)

        def replace(value):
            fn.__defaults__ = (value,)

    elif kind == "kwdefault":

        def fn(*, value=None):
            return 2.0 if value is ANCHOR else 5.0

        def replace(value):
            fn.__kwdefaults__ = {"value": value}

    else:
        captured = hl.constexpr(None)

        def fn():
            return 2.0 if captured is ANCHOR else 5.0

        def replace(value):
            nonlocal captured
            captured = hl.constexpr(value)

    x = torch.empty(17)
    args = (x, fn)
    kernel = helion.kernel(_callable_input.fn, autotune_effort="none")
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "__call__", observe_compiled_binding),
    ):
        initial = kernel(*args)
        _assert_offset(initial, 5.0)
        # A ConstExpr wrapper can later hold an equal custom fragment, even
        # when its current payload is None. Ordinary defaults still elide.
        assert _has_callable_guard(initial) == (kind == "closure")
        replace(ANCHOR)
        with_fragment = kernel(*args)
        assert with_fragment is not initial
        _assert_offset(with_fragment, 2.0)
        assert _has_callable_guard(with_fragment)
        replace(BooleanFragment())
        distinct = kernel(*args)
        fresh = helion.kernel(_callable_input.fn, autotune_effort="none").bind(args)
        assert distinct is not with_fragment
        _assert_offset(distinct, 5.0)
        assert distinct.to_code() == fresh.to_code()
        assert kernel(*args) is distinct


@dataclass
class _OpaqueDefault:
    fragment: object = None


@dataclass
class _InitVarDefault:
    initial: InitVar[object]
    fragment: object = field(init=False)

    def __post_init__(self, initial):
        self.fragment = initial


@dataclass
class _ClassVarDefault:
    fragment: object
    nonstored: ClassVar[object]


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("holder_type", [_InitVarDefault, _ClassVarDefault])
def test_dataclass_pseudo_fields_do_not_enter_opaque_call_guards(holder_type):
    holder = holder_type(ANCHOR)

    def fn(value=holder):
        return 2.0 if value.fragment is ANCHOR else 5.0

    args = (torch.empty(17), fn)
    kernel = helion.kernel(_callable_input.fn, autotune_effort="none")
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "__call__", observe_compiled_binding),
    ):
        first = kernel(*args)
        _assert_offset(first, 2.0)
        assert _has_callable_guard(first)
        holder.fragment = BooleanFragment()
        changed = kernel(*args)
        fresh = helion.kernel(_callable_input.fn, autotune_effort="none").bind(args)
        _assert_offset(changed, 5.0)
        assert changed is not first
        assert changed.to_code() == fresh.to_code()
        assert kernel(*args) is changed
        holder.fragment = ANCHOR
        assert kernel(*args) is first


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize(
    "kind",
    [
        "opaque_default",
        "opaque_kwdefault",
        "constexpr",
        "constexpr_kwdefault",
        "constexpr_closure",
    ],
)
def test_unkeyed_empty_callable_inputs_keep_identity_guard(kind):
    value = _OpaqueDefault()
    if kind in ("opaque_default", "opaque_kwdefault"):
        if kind == "opaque_default":

            def fn(holder=value):
                return 2.0 if holder.fragment is ANCHOR else 5.0

        else:

            def fn(*, holder=value):
                return 2.0 if holder.fragment is ANCHOR else 5.0

        source = _callable_input.fn

        def replace():
            value.fragment = ANCHOR

    elif kind in ("constexpr", "constexpr_kwdefault"):
        if kind == "constexpr":
            fn = _default_input(None)
        else:

            def fn(*, value=None):
                return 2.0 if value is ANCHOR else 5.0

        source = _constexpr_callable_input.fn

        def replace():
            if kind == "constexpr":
                fn.__defaults__ = (ANCHOR,)
            else:
                fn.__kwdefaults__ = {"value": ANCHOR}

    else:
        captured = None

        def fn():
            return 2.0 if captured is ANCHOR else 5.0

        source = _constexpr_callable_input.fn

        def replace():
            nonlocal captured
            captured = ANCHOR

    x = torch.empty(17)
    args = (x, fn)
    kernel = helion.kernel(source, autotune_effort="none")
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "__call__", observe_compiled_binding),
    ):
        initial = kernel(*args)
        _assert_offset(initial, 5.0)
        assert _has_callable_guard(initial)
        replace()
        updated = kernel(*args)
        assert updated is not initial
        _assert_offset(updated, 2.0)
        fresh = helion.kernel(source, autotune_effort="none").bind(args)
        assert updated.to_code() == fresh.to_code()


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("kind", ["default", "closure"])
def test_constexpr_custom_fragment_keeps_callable_identity_guard(kind):
    if kind == "default":
        wrapped = hl.constexpr(RECEIVER_ANCHOR)

        def fn(value=wrapped):
            return 2.0 if value.value is RECEIVER_ANCHOR else 5.0

        def replace(value):
            fn.__defaults__ = (hl.constexpr(value),)

    else:
        captured = hl.constexpr(RECEIVER_ANCHOR)

        def fn():
            return 2.0 if captured is RECEIVER_ANCHOR else 5.0

        def replace(value):
            nonlocal captured
            captured = hl.constexpr(value)

    args = (torch.empty(17), fn)
    kernel = helion.kernel(_callable_input.fn, autotune_effort="none")
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "__call__", observe_compiled_binding),
    ):
        initial_key = kernel._base_specialization_key(args)
        initial = kernel(*args)
        _assert_offset(initial, 2.0)
        prepared = kernel._prepared_call
        assert prepared is not None
        replace(_Receiver())
        # Custom equality stays part of the public constexpr key contract.
        assert kernel._base_specialization_key(args) == initial_key
        fresh = helion.kernel(_callable_input.fn, autotune_effort="none").bind(args)
        _assert_offset(fresh, 5.0)
        updated = kernel(*args)
        _assert_offset(updated, 5.0)
        assert updated is not initial
        assert updated.to_code() == fresh.to_code()
        assert not prepared.matches(kernel, args)
        assert _has_callable_guard(initial)
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            assert kernel(*args) is updated


@dataclass(eq=False)
class _IdentityDefault:
    fragment: object = None


class _NullEqualFragment(ConfigSpecFragment):
    def default(self):
        return 1

    def __eq__(self, other):
        return other is None or isinstance(other, _NullEqualFragment)

    def __hash__(self):
        return hash(None)


@onlyBackends(["triton", "cute"])
def test_primitive_constexpr_can_be_replaced_by_equal_custom_fragment():
    captured = hl.constexpr(None)

    def fn():
        return 2.0 if captured is None else 5.0

    args = (torch.empty(17), fn)
    kernel = helion.kernel(_callable_input.fn, autotune_effort="none")
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "__call__", observe_compiled_binding),
    ):
        initial_key = kernel._base_specialization_key(args)
        initial = kernel(*args)
        _assert_offset(initial, 2.0)
        prepared = kernel._prepared_call
        assert prepared is not None
        captured = hl.constexpr(_NullEqualFragment())
        assert kernel._base_specialization_key(args) == initial_key
        fresh = helion.kernel(_callable_input.fn, autotune_effort="none").bind(args)
        _assert_offset(fresh, 5.0)
        updated = kernel(*args)
        _assert_offset(updated, 5.0)
        assert updated is not initial
        assert not prepared.matches(kernel, args)
        assert _has_callable_guard(initial)
        assert updated.to_code() == fresh.to_code()


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("kind", ["dataclass", "container"])
def test_empty_opaque_constexpr_default_detects_later_fragment(kind):
    holder = _IdentityDefault()
    wrapped = hl.constexpr(holder if kind == "dataclass" else (holder,))
    if kind == "dataclass":

        def fn(value=wrapped):
            return 2.0 if value.value.fragment is ANCHOR else 5.0

    else:

        def fn(value=wrapped):
            return 2.0 if value.value[0].fragment is ANCHOR else 5.0

    args = (torch.empty(17), fn)
    kernel = helion.kernel(_callable_input.fn, autotune_effort="none")
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "__call__", observe_compiled_binding),
    ):
        initial_key = kernel._base_specialization_key(args)
        initial = kernel(*args)
        _assert_offset(initial, 5.0)
        prepared = kernel._prepared_call
        assert prepared is not None
        holder.fragment = ANCHOR
        assert kernel._base_specialization_key(args) == initial_key
        fresh = helion.kernel(_callable_input.fn, autotune_effort="none").bind(args)
        _assert_offset(fresh, 2.0)
        updated = kernel(*args)
        _assert_offset(updated, 2.0)
        assert updated is not initial
        assert not prepared.matches(kernel, args)
        assert _has_callable_guard(initial)
        assert updated.to_code() == fresh.to_code()
        with patch.object(kernel, "bind", side_effect=AssertionError("cache miss")):
            assert kernel(*args) is updated


def _unused_empty_cell_callback():
    value = None

    def callback():
        if False:
            return value
        return 5.0

    assert callback.__closure__ is not None
    del callback.__closure__[0].cell_contents
    return callback


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("kind", ["direct", "opaque", "constexpr", "default"])
def test_empty_callback_cell_binds_and_tracks_later_filling(kind):
    callback = _unused_empty_cell_callback()
    original = callback
    if kind == "default":

        def callback(fn=original):
            return fn()

    template = (
        _passed_callable_input
        if kind == "opaque"
        else _constexpr_callable_input
        if kind == "constexpr"
        else _callable_input
    )
    kernel = helion.kernel(template.fn, autotune_effort="none")
    args = (torch.empty(17), callback)
    cell = original.__closure__[0]
    with (
        cpu_render_target(),
        patch.object(BoundKernel, "__call__", observe_compiled_binding),
    ):
        empty_key = _function_key(kernel, original, baked=(kind == "default"))
        initial = kernel(*args)
        _assert_offset(initial, 5.0)
        assert kernel(*args) is initial
        cell.cell_contents = None
        assert _function_key(kernel, original, baked=(kind == "default")) != empty_key
        filled = kernel(*args)
        _assert_offset(filled, 5.0)
        del cell.cell_contents
        assert _function_key(kernel, original, baked=(kind == "default")) == empty_key
        _assert_offset(kernel(*args), 5.0)


@onlyBackends(["triton", "cute"])
@pytest.mark.parametrize("action", ["fill", "delete"])
def test_lifted_closure_rejects_changes_to_cell_occupancy(action):
    value = None

    def fill():
        nonlocal value
        value = 5.0
        return 5.0

    def delete():
        nonlocal value
        del value
        return 5.0

    callback = fill if action == "fill" else delete
    if action == "fill":
        del callback.__closure__[0].cell_contents
    kernel = helion.kernel(_callable_input.fn, autotune_effort="none")
    with cpu_render_target(), pytest.raises(helion.exc.ClosureMutation):
        kernel.bind((torch.empty(17), callback))
