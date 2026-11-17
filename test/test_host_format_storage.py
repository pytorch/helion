"""Keep untraced formatting hooks outside the host compiler's effect boundary."""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.tunable_fragments import load_host_function
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@dataclasses.dataclass
class FormatEffect:
    left: torch.Tensor
    right: torch.Tensor

    def __format__(self, spec):
        self.right.set_(self.left)
        return "1"

    def __str__(self):
        self.right.set_(self.left)
        return "string"

    def __repr__(self):
        self.right.set_(self.left)
        return "representation"

    def __bool__(self):
        self.right.set_(self.left)
        return True

    def __len__(self):
        self.right.set_(self.left)
        return 1

    def __or__(self, other):
        self.left.t_()
        return self


class NativeTuple(NamedTuple):
    left: torch.Tensor
    right: torch.Tensor


class TupleEffect(NativeTuple):
    __slots__ = ()

    __format__ = FormatEffect.__format__
    __str__ = FormatEffect.__str__
    __repr__ = FormatEffect.__repr__
    __bool__ = FormatEffect.__bool__
    __or__ = FormatEffect.__or__
    # PyTorch's argument pytree flattening calls a tuple's length before
    # tracing begins. Test custom __len__ on the dataclass instead, so these
    # cases isolate protocols in the compiled host function.


def _formatted_copy(left, right, effect: FormatEffect, kind: hl.constexpr):
    if kind == "format":
        _label = f"{effect}"
    elif kind == "str":
        _label = f"{effect!s}"
    elif kind == "repr":
        _label = f"{effect!r}"
    elif kind == "ascii":
        _label = f"{effect!a}"
    elif kind == "nested":
        _label = f"{(effect,)}"
    elif kind == "spec":
        _label = f"{1:{effect}}"
    else:
        _label = f"{1}:{left.shape}:{1.5!r}"
    for row, col in hl.tile(left.shape):
        right[row, col] = left[row, col] + 1
    return left, right


@pytest.mark.parametrize(
    "kind", ["format", "str", "repr", "ascii", "nested", "spec", "plain"]
)
@forbid_cuda_work()
def test_formatting_rejects_untraced_protocols_and_retains_builtin_values(kind):
    left, right = torch.empty(4, 8), torch.empty(4, 8)
    effect = FormatEffect(left, right)
    args = (left, right, effect, kind)
    kernel = helion.kernel(_formatted_copy, static_shapes=True)
    with cpu_render_target(fake_device_sm_count=148):
        if kind != "plain":
            with pytest.raises(
                helion.exc.TypeInferenceError,
                match="Host formatting with custom protocols is unsupported",
            ):
                cpu_bind(kernel, args)
            assert left.untyped_storage() is not right.untyped_storage()
            return
        bound = cpu_bind(kernel, args)
        with bound.env, bound.host_function:
            tensors = {
                name: bound.host_function.local_types[name].fake_value
                for name in ("left", "right")
            }
            assert (
                tensors["left"].untyped_storage()
                is not tensors["right"].untyped_storage()
            )
            for tensor in tensors.values():
                assert bound.env.tensor_storage_provenance(tensor) is not None
        entry = load_host_function(
            bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
        )
        actual_left, actual_right = entry(*args)
    assert actual_left.untyped_storage() is not actual_right.untyped_storage()


def _opaque_format(effect):
    return str(effect)


def _protocol_copy(left, right, effect, other, kind: hl.constexpr, flag: bool):
    if kind == "str":
        _label = str(effect)
    elif kind == "repr":
        _label = repr(effect)
    elif kind == "ascii":
        _label = ascii(effect)
    elif kind == "format":
        _label = format(effect)
    elif kind == "bool":
        _label = bool(effect)
    elif kind == "len":
        _label = len(effect)
    elif kind == "truth":
        if effect:
            _label = 1
    elif kind == "not":
        _label = not effect
    elif kind == "or":
        _label = effect | other
    elif kind == "percent":
        _label = "%s" % (effect,)  # noqa: UP031 - exercise percent formatting
    elif kind == "nested":
        _label = repr([effect])
    elif kind == "method":
        _label = "{}".format(effect)  # noqa: UP032 - exercise format method
    elif kind == "format_map":
        _label = "{effect}".format_map({"effect": effect})
    elif kind == "bytes_percent":
        _label = b"%r" % (effect,)
    elif kind == "opaque":
        _label = _opaque_format(effect)
    elif kind == "erased_opaque":
        converted = tuple([effect])
        _label = _opaque_format(converted[0])
    elif kind == "merged":
        if flag:
            target = effect
        else:
            target = other
        _label = str(target)
    elif kind == "native_tuple":
        _length = len(effect)
        _truth = bool(effect)
        _inverse = not effect
        if effect:
            _record_shape = effect.left.shape
    else:
        # These inspect the builtin outer container without invoking its
        # elements' protocols. Field access remains supported, too.
        records = [effect, other]
        _length = len(records)
        _truth = bool(records)
        _inverse = not records
        _converted = tuple(records)
        _rank = len(left.shape)
        if records:
            _record_shape = effect.left.shape
    for row, col in hl.tile(left.shape):
        right[row, col] = left[row, col] + 1
    return right


# Every operation has a distinct Python protocol entry point. Both record
# representations are needed: one is a tuple subclass, the other is not iterable.
@pytest.mark.parametrize("record_type", [FormatEffect, TupleEffect])
@pytest.mark.parametrize(
    "kind",
    [
        "str",
        "repr",
        "ascii",
        "format",
        "bool",
        "len",
        "truth",
        "not",
        "or",
        "percent",
        "nested",
        "method",
        "format_map",
        "bytes_percent",
        "opaque",
        "erased_opaque",
        "merged",
        "plain",
    ],
)
@forbid_cuda_work()
def test_record_protocols_are_not_evaluated_on_field_proxies(record_type, kind):
    left, right = torch.empty(4, 8), torch.empty(4, 8)
    args = (left, right, record_type(left, right), record_type(left, right), kind, True)
    kernel = helion.kernel(_protocol_copy, static_shapes=True)
    with cpu_render_target(fake_device_sm_count=148):
        if kind == "plain" or (kind == "len" and record_type is TupleEffect):
            bound = cpu_bind(kernel, args)
            with bound.env, bound.host_function:
                for name in ("left", "right"):
                    tensor = bound.host_function.local_types[name].fake_value
                    assert bound.env.tensor_storage_provenance(tensor) is not None
            entry = load_host_function(
                bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
            )
            assert entry(*args) is right
        else:
            with pytest.raises(
                helion.exc.TypeInferenceError,
                match="Host operations with custom record or container protocols",
            ):
                cpu_bind(kernel, args)
    assert left.untyped_storage() is not right.untyped_storage()
    assert left.shape == right.shape == (4, 8)


@forbid_cuda_work()
def test_namedtuple_native_length_and_truth_remain_supported():
    left, right = torch.empty(4, 8), torch.empty(4, 8)
    args = (left, right, NativeTuple(left, right), None, "native_tuple", True)
    kernel = helion.kernel(_protocol_copy, static_shapes=True)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, args)
        entry = load_host_function(
            bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
        )
        assert entry(*args) is right
    assert left.untyped_storage() is not right.untyped_storage()


def _builtin_record_binary(left, right, effect, use_tuple: hl.constexpr):
    if use_tuple:
        records = (effect,) + (effect,)  # noqa: RUF005 - exercise host concatenation
    else:
        records = [effect] + [effect]  # noqa: RUF005 - exercise host concatenation
    repeated = records * 2
    count = len(repeated)
    for row, col in hl.tile(left.shape):
        right[row, col] = left[row, col] + count
    return right, count


@pytest.mark.parametrize("record_type", [FormatEffect, TupleEffect])
@pytest.mark.parametrize("use_tuple", [False, True])
@forbid_cuda_work()
def test_builtin_binary_operations_do_not_invoke_nested_record_protocols(
    record_type, use_tuple
):
    left, right = torch.empty(4, 8), torch.empty(4, 8)
    args = (left, right, record_type(left, right), use_tuple)
    kernel = helion.kernel(_builtin_record_binary, static_shapes=True)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, args)
        entry = load_host_function(
            bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
        )
        result, count = entry(*args)
    assert result is right and count == 4
    assert left.shape == right.shape == (4, 8)
    assert left.untyped_storage() is not right.untyped_storage()
