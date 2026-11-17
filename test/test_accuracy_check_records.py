from __future__ import annotations

import dataclasses
from typing import Literal
import weakref

import pytest
import torch
from torch.utils._pytree import register_pytree_node

from helion.autotuner import AccuracyCheck
from helion.autotuner.accuracy import relative_l2_disagreement
from helion.runtime.settings import Settings

pytestmark = pytest.mark.cpu_only


@dataclasses.dataclass
class _Output:
    backref: _Output | None
    payload: object


def _nested_output(tensor: torch.Tensor, *, cyclic: bool) -> _Output:
    leaf = _Output(None, {"values": [tensor]})
    outer = _Output(None, (leaf,))
    if cyclic:
        # Encounter a backedge before visiting the nested tensor field.
        outer.backref = outer
        leaf.backref = outer
    return outer


@pytest.mark.parametrize("cyclic", [False, True])
@pytest.mark.parametrize(
    "atol_scale,inside,outside",
    [("peak", 0.9, 1.1), ("rms", 0.45, 0.55)],
)
def test_scaled_accuracy_reaches_nested_record_tensors(
    atol_scale: Literal["peak", "rms"],
    inside: float,
    outside: float,
    cyclic: bool,
) -> None:
    # The peak is 100 and the RMS is 50: with atol=.01 the respective
    # absolute floors are 1 and .5, rather than the unscaled .01.
    expected = torch.tensor([100.0, 0.0, 0.0, 0.0])
    actual = expected.clone()
    actual[1] = inside
    expected_output = _nested_output(expected, cyclic=cyclic)
    actual_output = _nested_output(actual, cyclic=cyclic)
    check = AccuracyCheck(atol=0.01, rtol=0.0, atol_scale=atol_scale)

    check(actual_output, expected_output)
    with pytest.raises(AssertionError):
        AccuracyCheck(atol=0.01, rtol=0.0)(actual_output, expected_output)

    # A visited cyclic record must not hide a mismatch on a later call.
    actual[1] = outside
    with pytest.raises(AssertionError):
        check(actual_output, expected_output)


@pytest.mark.parametrize("atol_scale,small_error", [("peak", 0.002), ("rms", 0.011)])
def test_scaled_record_accuracy_keeps_each_tensor_own_scale(
    atol_scale: Literal["peak", "rms"], small_error: float
) -> None:
    large = torch.tensor([100.0, 0.0, 0.0, 0.0])
    small = torch.tensor([0.1, 0.0])
    changed_small = small.clone()
    changed_small[1] = small_error
    expected = _Output(None, [_nested_output(large, cyclic=True), small])
    actual = _Output(None, [_nested_output(large.clone(), cyclic=True), small.clone()])
    check = AccuracyCheck(atol=0.01, rtol=0.0, atol_scale=atol_scale)
    check(actual, expected)

    # The large sibling cannot lend its tolerance to the small tensor.
    actual.payload = [_nested_output(large.clone(), cyclic=True), changed_small]
    with pytest.raises(AssertionError):
        check(actual, expected)


@pytest.mark.parametrize("atol_scale", ["peak", "rms"])
def test_settings_roundtrip_preserves_record_accuracy_callable(
    atol_scale: Literal["peak", "rms"],
) -> None:
    check = AccuracyCheck(
        atol=0.01, rtol=0.0, atol_scale=atol_scale, fp8_accuracy="bitwise"
    )
    settings = Settings(autotune_baseline_accuracy_check_fn=check)
    serialized = settings.to_dict()
    assert serialized["autotune_baseline_accuracy_check_fn"] is check
    restored = Settings(**serialized)
    assert restored.autotune_baseline_accuracy_check_fn is check
    restored_check = restored.autotune_baseline_accuracy_check_fn
    assert restored_check is not None

    expected = torch.tensor([100.0, 0.0, 0.0, 0.0])
    actual = expected.clone()
    actual[1] = 0.4
    expected_output = _nested_output(expected, cyclic=True)
    actual_output = _nested_output(actual, cyclic=True)
    restored_check(actual_output, expected_output)
    actual[1] = 2.0
    with pytest.raises(AssertionError):
        restored_check(actual_output, expected_output)


@pytest.mark.parametrize("comparison", ["accuracy", "relative_l2"])
def test_record_comparison_retains_temporary_pytree_children(comparison: str) -> None:
    @dataclasses.dataclass
    class TemporaryBox:
        payload: object

    children_seen: list[weakref.ReferenceType[_Output]] = []
    earlier_children_alive: list[bool] = []

    def flatten(box: TemporaryBox) -> tuple[list[_Output], None]:
        # Observe lifetime without retaining the children in the test. This
        # makes the regression independent of the allocator reusing an address.
        earlier_children_alive.append(all(ref() is not None for ref in children_seen))
        child = _Output(None, box.payload)
        children_seen.append(weakref.ref(child))
        return [child], None

    register_pytree_node(
        TemporaryBox,
        flatten,
        lambda children, context: TemporaryBox(next(iter(children)).payload),
    )
    expected_tensors = [torch.ones(4) for _ in range(8)]
    actual_tensors = [value.clone() for value in expected_tensors]
    expected = [_Output(None, TemporaryBox(value)) for value in expected_tensors]
    actual = [_Output(None, TemporaryBox(value)) for value in actual_tensors]
    check = AccuracyCheck(atol=0.0, rtol=0.0)

    if comparison == "accuracy":
        check(actual, expected)
    else:
        assert relative_l2_disagreement(actual, expected, 0.0) is None
    assert len(children_seen) == 16
    assert all(earlier_children_alive)

    children_seen.clear()
    earlier_children_alive.clear()
    actual_tensors[5].zero_()
    if comparison == "accuracy":
        with pytest.raises(AssertionError):
            check(actual, expected)
    else:
        assert relative_l2_disagreement(actual, expected, 0.0) is not None
    assert all(earlier_children_alive)


def test_relative_l2_record_cycles_do_not_hide_tensor_disagreement() -> None:
    expected_tensor = torch.ones(4)
    actual_tensor = expected_tensor.clone()
    expected = _nested_output(expected_tensor, cyclic=True)
    actual = _nested_output(actual_tensor, cyclic=True)
    assert relative_l2_disagreement(actual, expected, 0.0) is None
    actual_tensor.zero_()
    assert relative_l2_disagreement(actual, expected, 0.0) is not None
