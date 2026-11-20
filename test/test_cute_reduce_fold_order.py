"""The order in which the grouped reduces fold a group's few partials.

The serial form of the shared two-stage reduce (2 to 8 warp partials read by
every thread) and the cluster exchange of at most 8 slots fold a float sum as
a balanced tree, in the order of the two-stage form's second stage, so a sum
has the same association from 2 warps on.  Every other combine keeps its
chain.  The device results are checked in
``test_cute_persistent_reduce_hints_cuda.py``.
"""

from __future__ import annotations

import operator

import pytest

pytest.importorskip("cutlass")

import cutlass

from helion._compiler.cute import reduce_helpers

pytestmark = pytest.mark.cpu_only


class _Term:
    """A symbolic operand that records how it is combined."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __add__(self, other: _Term) -> _Term:
        return _Term(f"({self.text}+{other.text})")


def _butterfly(count: int) -> str:
    """Lane 0 of a 32-lane descending butterfly (offsets 16 .. 1, partner
    second) over ``count`` values and identity lanes, the identities left
    out: ``_cute_warp_tree_fold``'s association."""
    lanes: list[_Term | None] = [
        _Term(f"w{lane}") if lane < count else None for lane in range(32)
    ]
    offset = 16
    while offset:
        for lane in range(offset):
            partner = lanes[lane + offset]
            if partner is not None:
                mine = lanes[lane]
                lanes[lane] = partner if mine is None else mine + partner
        offset //= 2
    assert lanes[0] is not None
    return lanes[0].text


def test_tree_fold_is_the_two_stage_butterfly_without_identity_lanes() -> None:
    def fold(count: int) -> str:
        terms = [_Term(f"w{index}") for index in range(count)]
        return reduce_helpers._cute_tree_fold(terms, operator.add).text

    assert fold(1) == "w0"
    assert fold(2) == "(w0+w1)"
    assert fold(3) == "((w0+w2)+w1)"
    assert fold(4) == "((w0+w2)+(w1+w3))"
    assert fold(8) == "(((w0+w4)+(w2+w6))+((w1+w5)+(w3+w7)))"
    for count in range(1, 33):
        assert fold(count) == _butterfly(count), count
        # As many combines as a chain: every value enters once.
        assert fold(count).count("+") == count - 1
    # The butterfly starts at offset 16 and would leave a 33rd value out.
    with pytest.raises(AssertionError):
        fold(33)


def test_only_a_float_sum_folds_as_a_tree() -> None:
    folds = reduce_helpers._folds_as_tree
    for dtype in (cutlass.Float32, cutlass.Float64, cutlass.Float16, cutlass.BFloat16):
        assert folds(operator.add, dtype)
        # A product rounds differently in another order and an extremum's
        # compare-select picks another zero: both keep their chain.
        assert not folds(operator.mul, dtype)
        assert not folds(reduce_helpers._cute_nan_max, dtype)
        assert not folds(reduce_helpers._cute_nan_min, dtype)
    # Integer sums and ``any`` have one result in any order.
    for dtype in (cutlass.Int32, cutlass.Int64, cutlass.Uint8):
        assert not folds(operator.add, dtype)
        assert not folds(operator.or_, dtype)
    assert {
        name: folds(combine, cutlass.Float32)
        for name, (_warp_op, combine) in reduce_helpers._OP_TABLE.items()
    } == {"sum": True, "max": False, "min": False, "prod": False, "any": False}
