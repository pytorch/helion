"""Native record metadata queries do not run user protocols."""

from __future__ import annotations

import dataclasses

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.tunable_fragments import load_host_function
import helion.language as hl

pytestmark = pytest.mark.cpu_only
_events: list[str] = []


@dataclasses.dataclass
class QueryRecord:
    x: torch.Tensor

    def __format__(self, spec):
        _events.append("format")
        self.x.t_()
        return "bad record"


@dataclasses.dataclass
class DefaultQueryRecord:
    x: torch.Tensor
    flag: bool = False
    count: int = 0
    text: str = ""


@dataclasses.dataclass(slots=True)
class SlotDefaultQueryRecord:
    x: torch.Tensor
    flag: bool = False
    count: int = 0
    text: str = ""


def _query_default_field(x, record, field: hl.constexpr):
    answer = hasattr(record, field)
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile]
    return out, answer


@pytest.mark.parametrize("record_type", [DefaultQueryRecord, SlotDefaultQueryRecord])
@pytest.mark.parametrize("field", ["flag", "count", "text"])
@forbid_cuda_work()
def test_declared_class_defaults_preserve_attribute_presence(record_type, field):
    x = torch.empty(8)
    record = record_type(x)
    with cpu_render_target():
        bound = cpu_bind(helion.kernel(_query_default_field), (x, record, field))
        entry = load_host_function(
            bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
        )
        out, answer = entry(x, record, field)
    assert answer is hasattr(record, field) is True
    assert out.shape == x.shape


def _query_record(x, record, form: hl.constexpr):
    if form == "type":
        answer = type(record) is QueryRecord
    elif form == "isinstance":
        answer = isinstance(record, QueryRecord)
    elif form == "hasattr":
        answer = hasattr(record, "x")
    else:
        answer = hasattr(record, "missing")
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        out[tile] = x[tile]
    return out, answer


@pytest.mark.parametrize("form", ["type", "isinstance", "hasattr"])
@forbid_cuda_work()
def test_native_record_queries_bind_and_execute(form):
    x = torch.empty(8)
    record = QueryRecord(x)
    with cpu_render_target():
        bound = cpu_bind(helion.kernel(_query_record), (x, record, form))
        entry = load_host_function(
            bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
        )
        out, answer = entry(x, record, form)
    assert answer is (form != "missing")
    assert out.shape == x.shape


def _sort_tuple(x):
    values, indices = tuple(torch.sort(x))
    out = torch.empty_like(values)
    for tile in hl.tile(x.numel()):
        out[tile] = values[tile]
    return out, values, indices


@forbid_cuda_work()
def test_native_torch_return_tuple_preserves_values():
    x = torch.tensor([3.0, 1.0, 2.0])
    with cpu_render_target():
        bound = cpu_bind(helion.kernel(_sort_tuple), (x,))
        entry = load_host_function(
            bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
        )
        out, values, indices = entry(x)
    expected = torch.sort(x)
    torch.testing.assert_close(values, expected.values)
    torch.testing.assert_close(indices, expected.indices)
    assert out.shape == x.shape


@forbid_cuda_work()
def test_undeclared_attribute_presence_is_not_specialized():
    x = torch.empty(8)
    with (
        cpu_render_target(),
        pytest.raises(
            helion.exc.TypeInferenceError, match="custom record or container protocols"
        ),
    ):
        cpu_bind(helion.kernel(_query_record), (x, QueryRecord(x), "missing"))


class PropertyRecord(QueryRecord):
    @property
    def secret(self):
        _events.append("property")
        self.x.t_()
        return True


class InstanceEffect(type):
    def __instancecheck__(cls, instance):
        _events.append("instancecheck")
        instance.x.t_()
        return True


class EffectTarget(metaclass=InstanceEffect):
    pass


def _effectful_query(x, record, form: hl.constexpr):
    if form == "property":
        answer = hasattr(record, "secret")
    else:
        answer = isinstance(record, EffectTarget)
    out = torch.empty_like(x)
    for row, col in hl.tile(x.shape):
        out[row, col] = x[row, col]
    return out, answer


@pytest.mark.parametrize("form", ["property", "metaclass"])
@forbid_cuda_work()
def test_record_queries_do_not_invoke_custom_hooks(form):
    _events.clear()
    x = torch.empty(4, 8)
    record = PropertyRecord(x) if form == "property" else QueryRecord(x)
    with (
        cpu_render_target(),
        pytest.raises(
            helion.exc.TypeInferenceError, match="pass declared fields explicitly"
        ),
    ):
        cpu_bind(helion.kernel(_effectful_query), (x, record, form))
    assert not _events
    assert x.shape == (4, 8)


def _assert_message(x, record, keep: int):
    assert keep > 0, f"failure: {record}"
    out = torch.empty_like(x)
    for row, col in hl.tile(x.shape):
        out[row, col] = x[row, col]
    return out


def _caught_assert_message(x, record, keep: int):
    try:  # noqa: SIM105 -- exercise the traced try/except syntax
        assert keep > 0, f"failure: {record}"
    except AssertionError:
        pass
    out = torch.empty_like(x)
    for row, col in hl.tile(x.shape):
        out[row, col] = x[row, col]
    return out


def _always_failed_assert(x, record):
    assert False, f"failure: {record}"  # noqa: B011 -- statically false assertion
    out = torch.empty_like(x)
    for row, col in hl.tile(x.shape):
        out[row, col] = x[row, col]
    return out


@pytest.mark.parametrize(
    "fn", [_assert_message, _caught_assert_message, _always_failed_assert]
)
@forbid_cuda_work()
def test_custom_assertion_messages_do_not_run_hooks_during_inference(fn):
    _events.clear()
    x = torch.empty(4, 8)
    record = QueryRecord(x)
    args = (x, record) if fn is _always_failed_assert else (x, record, 1)
    with (
        cpu_render_target(),
        pytest.raises(
            helion.exc.TypeInferenceError, match="pass declared fields explicitly"
        ),
    ):
        cpu_bind(helion.kernel(fn), args)
    assert not _events
    assert x.shape == (4, 8)


def _field_assert_message(x, record, keep: int):
    assert keep > 0, f"failure with shape: {record.x.shape}"
    out = torch.empty_like(x)
    for row, col in hl.tile(x.shape):
        out[row, col] = x[row, col]
    return out


@forbid_cuda_work()
def test_declared_field_assertion_message_preserves_runtime_failure():
    _events.clear()
    x = torch.empty(4, 8)
    record = QueryRecord(x)
    launches = []
    with cpu_render_target():
        bound = cpu_bind(helion.kernel(_field_assert_message), (x, record, 1))
        entry = load_host_function(
            bound,
            bound.config_spec.default_config(),
            lambda *args, **kwargs: launches.append(True),
        )
        assert entry(x, record, 1).shape == (4, 8)
        with pytest.raises(AssertionError, match="failure with shape"):
            entry(x, record, 0)
    assert launches == [True]
    assert not _events
    assert x.shape == (4, 8)
