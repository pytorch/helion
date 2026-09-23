from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from test._cute_paired_sum_model import reference_tree
from test._cute_paired_sum_model import run_literal

from helion.runtime.cute import paired_sum
from helion.runtime.cute.paired_sum import plan_for_pair
from helion.runtime.cute.paired_sum import sum_plan
from helion.runtime.cute.paired_sum import try_paired_sum_cast

if TYPE_CHECKING:
    from collections.abc import Iterator

    from test._cute_paired_sum_model import Value


@pytest.fixture(scope="module", autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield


@pytest.mark.parametrize("layout", ("mapped", "narrow"))
@pytest.mark.parametrize("rows", (2, 64, 65, 128, 257))
@pytest.mark.parametrize("stride,alignment", ((1, 16), (4, 16), (4, 20)))
def test_single_output_column_declines_the_coalesced_input_tree(
    layout: str, rows: int, stride: int, alignment: int
) -> None:
    # Real TensorIterator storage coalesces this output axis for both contiguous
    # and padded rows. These are input-reduction trees, even without input vec4.
    assert sum_plan(rows, 1, stride, alignment, layout) is None


@pytest.mark.parametrize("layout", ("mapped", "narrow"))
@pytest.mark.parametrize("rows", (0, 1))
@pytest.mark.parametrize("stride", (1, 4))
def test_empty_and_one_row_single_column_keep_exact_device_ownership(
    layout: str, rows: int, stride: int
) -> None:
    plan = sum_plan(rows, 1, stride, 16, layout)
    assert plan is not None
    result = run_literal(plan, stride=stride)
    assert result["loads"] == 2 * rows
    assert result["stores"] == 2


@pytest.mark.parametrize("layout", ("mapped", "narrow"))
@pytest.mark.parametrize("rows", (2, 64, 65, 128, 257))
@pytest.mark.parametrize("columns,stride", ((2, 2), (4, 4), (17, 20), (128, 132)))
def test_non_singleton_output_axis_remains_eligible(
    layout: str, rows: int, columns: int, stride: int
) -> None:
    assert sum_plan(rows, columns, stride, 16, layout) is not None


@pytest.mark.parametrize("sign", (1, -1))
def test_original_singleton_cancellation_distinguishes_the_two_fp32_trees(
    sign: int,
) -> None:
    def evaluate(value: Value) -> np.float32:
        if value.op == "positive_zero":
            return np.float32(0)
        if value.op == "input":
            row = cast("int", value.args[1])
            return np.float32(sign * {0: 1e8, 4: -1e8, 16: 1}.get(row, 0))
        assert value.op == "add_rn_f32"
        return np.float32(
            evaluate(cast("Value", value.args[0]))
            + evaluate(cast("Value", value.args[1]))
        )

    # The preserved GPU witness launches ATen's 64-lane x reduction for [64,1]
    # with strides (1,1) and (4,1). Each lane has one input, so its descending x
    # tree has the same edges as this independent 64-lane tree model. The old
    # admitted plan instead folded four sequential chains in one row lane.
    old_output_tree = evaluate(reference_tree(64, 1, 0, 0))
    actual_input_tree = evaluate(reference_tree(64, 64, 0, 0))
    assert float(old_output_tree) == sign
    assert float(actual_input_tree) == 0
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        assert torch.tensor([old_output_tree, actual_input_tree]).to(
            dtype
        ).tolist() == [
            sign,
            0,
        ]


def _forbidden(*args: object, **kwargs: object) -> None:
    raise AssertionError("singleton fallback must precede compatibility and launch")


@pytest.mark.parametrize("layout", ("mapped", "narrow"))
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("stride", (1, 4))
def test_paired_singleton_falls_back_before_allocation_or_compilation(
    layout: str, dtype: torch.dtype, stride: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    left = torch.zeros(64, stride)[:, :1]
    right = torch.zeros(64, stride)[:, :1]
    assert plan_for_pair(left, right, dtype, layout) is None
    monkeypatch.setattr(
        torch.Tensor, "device", property(lambda self: torch.device("cuda"))
    )
    monkeypatch.setattr(paired_sum, "_compatible_aten_sum", _forbidden)
    assert (
        try_paired_sum_cast(
            left,
            right,
            dtype,
            dtype_is_tensor=False,
            layout=layout,
            _launcher=_forbidden,
        )
        is None
    )
