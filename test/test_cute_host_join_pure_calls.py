from __future__ import annotations

import ast
import math

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
import triton

import helion
from helion._compiler.runtime_tensor_guards import (
    tensor_pairs_with_proven_disjoint_storage,
)
from helion._compiler.type_info import CallableType
from helion._compiler.type_info import LiteralType
from helion._compiler.type_info import _is_known_host_call
from helion._compiler.type_info import _is_known_host_value
from helion._compiler.variable_origin import AttributeOrigin
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _votes_after_host_call(
    x, left, right, select, kind: hl.constexpr, opaque: hl.constexpr
):
    first = torch.empty_like(left)
    second = torch.empty_like(right)
    if select > 0:
        target = first
    else:
        target = second
    if kind == "len":
        _ignored = len(x)
    elif kind == "min":
        _ignored = min(x.size(0), 3)
    elif kind == "max":
        _ignored = max(x.size(0), 3)
    elif kind == "sqrt":
        _ignored = math.sqrt(x.size(0))
    elif kind == "int":
        _ignored = int(x.size(0))
    elif kind == "tuple":
        _ignored = tuple(x.shape)
    elif kind == "range":
        _ignored = range(3)
    elif kind == "abs":
        _ignored = abs(-3)
    elif kind == "float":
        _ignored = float(3)
    elif kind == "bool":
        _ignored = bool(3)
    elif kind == "round":
        _ignored = round(3.5)
    elif kind == "sum":
        _ignored = sum((2, 3))
    elif kind == "list":
        _ignored = list((2, 3))  # noqa: C410 - exercise the host callable
    elif kind == "dict":
        _ignored = dict(value=3)  # noqa: C408 - exercise the host callable
    elif kind == "cdiv":
        _ignored = helion.cdiv(x.size(0), 2)
    elif kind == "next_power":
        _ignored = helion.next_power_of_2(x.size(0))
    elif kind == "triton_cdiv":
        _ignored = triton.cdiv(x.size(0), 2)
    elif kind == "triton_next_power":
        _ignored = triton.next_power_of_2(x.size(0))
    elif kind == "bit_length":
        _ignored = (x.size(0) - 1).bit_length()
    elif kind == "split":
        _ignored = torch.split(x, 1, dim=0)
    elif kind == "method_split":
        _ignored = x.split(1, dim=0)
    elif kind == "native":
        _ignored = torch.neg(x)
    elif kind == "native_method":
        _ignored = target.transpose(0, 1)
    elif kind == "custom_key":
        _ignored = min((1, 2), key=opaque)
    for row in hl.tile(x.size(0), block_size=1):
        values = x[row.id, :, :]
        vote = torch.any(values > 2)
        left[row.id, :] = torch.amax(values, dim=1)
        target[row.id, :] = torch.where(vote, torch.amin(values, dim=1), 0.0)
    return left, target


def _bind(kind, opaque=None):
    kernel = helion.kernel(_votes_after_host_call, backend="cute", static_shapes=True)
    args = (torch.empty(3, 32, 32), torch.empty(3, 32), torch.empty(3, 32))
    return cpu_bind(kernel, (*args, 1, kind, opaque))


# Each case exercises a different callable identity or argument conversion;
# native function/descriptor cases retain the previously supported paths.
@pytest.mark.parametrize(
    "kind",
    [
        "len",
        "min",
        "max",
        "sqrt",
        "int",
        "tuple",
        "range",
        "abs",
        "float",
        "bool",
        "round",
        "sum",
        "list",
        "dict",
        "cdiv",
        "next_power",
        "triton_cdiv",
        "triton_next_power",
        "bit_length",
        "split",
        "method_split",
        "native",
        "native_method",
    ],
)
def test_known_host_calls_preserve_fresh_join_proof(kind):
    with cpu_render_target(fake_device_sm_count=148):
        bound = _bind(kind)
        with bound.env, bound.host_function:
            tensors = {
                name: bound.host_function.local_types[name].fake_value
                for name in ("left", "first", "second", "target")
            }
            pairs = tensor_pairs_with_proven_disjoint_storage(bound.env, tensors)
        assert frozenset(("left", "target")) in pairs
        assert frozenset(("first", "target")) not in pairs
        assert frozenset(("second", "target")) not in pairs
        ast.parse(bound.to_code(bound.config_spec.default_config()))


def _key(value):
    return value


class _ProtocolEffects:
    def __init__(self, first, left):
        self.first = first
        self.left = left
        self.calls = []

    def _touch(self, operation):
        self.calls.append(operation)
        original = self.first.data
        self.first.data = self.left
        self.first.data = original

    def __len__(self):
        self._touch("len")
        return 3

    def __float__(self):
        self._touch("sqrt")
        return 3.0

    def __int__(self):
        self._touch("int")
        return 3

    def __lt__(self, other):
        self._touch("min")
        return False

    def __neg__(self):
        self._touch("unary")
        return 1

    def __getitem__(self, index):
        self._touch("getitem")
        return 1

    def __bool__(self):
        self._touch("truth")
        return True

    @property
    def value(self):
        self._touch("attribute")
        return 1

    def __iter__(self):
        self._touch("unpack")
        return iter((1, 2))

    def __eq__(self, other):
        self._touch("merge")
        return True


@pytest.mark.parametrize(
    "operation", ["unary", "getitem", "truth", "attribute", "unpack", "merge"]
)
def test_literal_protocol_execution_invalidates_existing_join(operation):
    with cpu_render_target(fake_device_sm_count=148):
        bound = _bind("native")
        host = bound.host_function
        with bound.env, host:
            tensors = {
                name: host.local_types[name].fake_value
                for name in ("left", "first", "target")
            }
            assert frozenset(
                ("left", "target")
            ) in tensor_pairs_with_proven_disjoint_storage(bound.env, tensors)
            owner = _ProtocolEffects(tensors["first"], tensors["left"])
            origin = host.local_types["first"].origin
            value = LiteralType(origin, owner)
            if operation == "unary":
                value.propagate_unary(ast.USub(), origin)
            elif operation == "getitem":
                value.propagate_getitem(LiteralType(origin, 0), origin)
            elif operation == "truth":
                value.truth_value()
            elif operation == "attribute":
                value.propagate_attribute("value", AttributeOrigin(origin, "value"))
            elif operation == "unpack":
                value.unpack()
            else:
                value.merge(LiteralType(origin, owner))
            assert owner.calls == [operation]
            assert frozenset(
                ("left", "target")
            ) not in tensor_pairs_with_proven_disjoint_storage(bound.env, tensors)


def test_known_call_classification_never_executes_user_hooks():
    class Hooks:
        def __eq__(self, other):
            raise AssertionError("classification called __eq__")

        def __iter__(self):
            raise AssertionError("classification called __iter__")

        def __float__(self):
            raise AssertionError("classification called __float__")

    custom = Hooks()
    cyclic = []
    cyclic.append(cyclic)
    assert not _is_known_host_value(cyclic)
    assert not _is_known_host_call(custom, (), {})
    # Iterating the exact outer list does not invoke its element's protocols.
    assert _is_known_host_call(tuple, ([custom],), {})
    assert tuple([custom])[0] is custom  # noqa: RUF015 - exercise tuple conversion
    assert not _is_known_host_call(tuple, (custom,), {})

    class Meta(type):
        def __eq__(cls, other):
            raise AssertionError("classification called metaclass equality")

    class CustomList(list, metaclass=Meta):
        pass

    assert not _is_known_host_call(tuple, (CustomList([custom]),), {})
    assert not _is_known_host_call(math.sqrt, (custom,), {})
    assert not _is_known_host_call(min, ((1, 2),), {"key": _key})
    assert not _is_known_host_call(sorted, ([1, 2],), {"key": _key})
    assert not _is_known_host_call(helion.cdiv, (custom, 2), {})
    assert not _is_known_host_call(abs, (custom,), {})
    assert _is_known_host_call(int.bit_length, (3,), {})

    class IntegerSubclass(int):
        pass

    assert not _is_known_host_call(IntegerSubclass(3).bit_length, (), {})
    assert _is_known_host_call(torch.finfo, (torch.float32,), {})
    with FakeTensorMode():
        tensor = torch.empty(3)
        assert not _is_known_host_call(tensor.apply_, (_key,), {})
        assert _is_known_host_call(tensor.clone, (), {})


@pytest.mark.parametrize("operation", ["len", "sqrt", "int", "tuple", "min"])
def test_builtin_user_protocol_runs_before_proof_is_discarded(operation):
    with cpu_render_target(fake_device_sm_count=148):
        bound = _bind("native")
        host = bound.host_function
        with bound.env, host:
            tensors = {
                name: host.local_types[name].fake_value
                for name in ("left", "first", "target")
            }
            assert frozenset(
                ("left", "target")
            ) in tensor_pairs_with_proven_disjoint_storage(bound.env, tensors)
            owner = _ProtocolEffects(tensors["first"], tensors["left"])
            origin = host.local_types["first"].origin
            value = LiteralType(origin, owner)
            fn = {
                "len": len,
                "sqrt": math.sqrt,
                "int": int,
                "tuple": tuple,
                "min": min,
            }[operation]
            args = (LiteralType(origin, 3), value) if operation == "min" else (value,)
            CallableType(origin, fn).propagate_call(args, {}, origin)
            if operation == "tuple":
                # Python 3.14 takes the iterator's length hint; 3.12 also asks
                # the original object. Both must execute the user iterator.
                assert owner.calls in (["unpack"], ["unpack", "len"])
            else:
                assert owner.calls == [operation]
            assert frozenset(
                ("left", "target")
            ) not in tensor_pairs_with_proven_disjoint_storage(bound.env, tensors)


def test_min_key_callback_still_invalidates_join():
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(helion.exc.BackendUnsupported, match="require disjoint storage"),
    ):
        _bind("custom_key", _key)


def test_replaced_library_function_is_not_treated_as_known(monkeypatch):
    def replacement(*args, **kwargs):
        raise AssertionError("classification executed replacement")

    monkeypatch.setattr(torch, "split", replacement)
    with FakeTensorMode():
        tensor = torch.empty(3)
        assert not _is_known_host_call(replacement, (tensor, 1), {})
