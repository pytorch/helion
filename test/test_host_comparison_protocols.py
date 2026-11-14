"""Structural record proxies must not hide comparison protocol effects."""

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


@dataclasses.dataclass(eq=False)
class ComparisonRecord:
    x: torch.Tensor

    def __contains__(self, item):
        _events.append("contains")
        self.x.t_()
        return True

    def __eq__(self, other):
        _events.append("eq")
        self.x.t_()
        return True


def _comparison_copy(x, record, other, form: hl.constexpr):
    if form == "record_in":
        member = 0 in record
    elif form == "record_not_in":
        member = 0 not in record
    elif form == "nested_in":
        member = 0 in [record]  # noqa: FURB171 - exercise containment, not direct equality
    elif form == "sequence_eq":
        member = [record] == [other]
    elif form == "identity":
        member = record is other
    elif form == "not_identity":
        member = record is not other
    elif form == "empty_list":
        member = record in []
    elif form == "empty_tuple":
        member = record not in ()
    elif form == "different_lengths":
        member = [record] == []
    else:
        member = "value" in {"value": record}
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    for row, col in hl.tile(x.shape):
        out[row, col] = x[row, col] + 1
    return out, member


@pytest.mark.parametrize(
    "form", ["record_in", "record_not_in", "nested_in", "sequence_eq"]
)
@forbid_cuda_work()
def test_untraced_membership_and_equality_are_refused(form):
    _events.clear()
    reference_x = torch.empty(4, 8)
    record = ComparisonRecord(reference_x)
    if form == "record_in":
        assert 0 in record
    elif form == "record_not_in":
        assert not (0 not in record)
    elif form == "nested_in":
        assert 0 in [record]  # noqa: FURB171 - exercise element equality via containment
    else:
        assert [record] == [ComparisonRecord(reference_x)]
    assert _events == (["contains"] if form.startswith("record_") else ["eq"])
    assert reference_x.shape == (8, 4)
    _events.clear()
    x = torch.empty(4, 8)
    args = (x, ComparisonRecord(x), ComparisonRecord(x), form)
    kernel = helion.kernel(_comparison_copy, static_shapes=True)
    with (
        cpu_render_target(fake_device_sm_count=148),
        pytest.raises(
            helion.exc.TypeInferenceError,
            match="Host operations with custom record or container protocols",
        ),
    ):
        cpu_bind(kernel, args)
    assert not _events
    assert x.shape == (4, 8)


@pytest.mark.parametrize(
    "form,expected",
    [
        ("identity", False),
        ("not_identity", True),
        ("empty_list", False),
        ("empty_tuple", True),
        ("different_lengths", False),
        ("dict_values", True),
    ],
)
@forbid_cuda_work()
def test_proved_builtin_comparisons_do_not_invoke_record_protocols(form, expected):
    _events.clear()
    x = torch.empty(4, 8)
    args = (x, ComparisonRecord(x), ComparisonRecord(x), form)
    kernel = helion.kernel(_comparison_copy, static_shapes=True)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, args)
        entry = load_host_function(
            bound, bound.config_spec.default_config(), lambda *args, **kwargs: None
        )
        out, member = entry(*args)
    assert member is expected
    assert out.shape == x.shape == (4, 8)
    assert not _events
