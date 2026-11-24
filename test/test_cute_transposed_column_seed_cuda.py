"""GPU test: the ``cute_transposed_column_reduction`` seed (the register
transpose of a column sum whose input is contiguous along the reduced rows)
copies the permuted input exactly and sums its rows within fp32 rounding."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.cute.autotune.heuristics import (
    CuteTransposedColumnReductionHeuristic,
)
from helion._testing import DEVICE
from helion._testing.transposed_column_kernels import copy_and_column_sums
from helion._testing.transposed_column_kernels import permuted


@pytest.mark.parametrize(
    ("dtype", "rows", "columns"),
    [(torch.bfloat16, 128, 48), (torch.float32, 100, 6)],
    ids=["bf16", "fp32_short_rows_narrow_columns"],
)
def test_the_seed_copies_and_sums(dtype: torch.dtype, rows: int, columns: int) -> None:
    torch.manual_seed(0)
    x = permuted(3, rows, columns, dtype, DEVICE)
    bound = copy_and_column_sums.bind((x,))
    with bound.env:
        seed = CuteTransposedColumnReductionHeuristic.get_seed_config(
            bound.env, bound.host_function.device_ir
        )
    assert seed is not None
    kernel = helion.kernel(
        copy_and_column_sums.fn,
        backend="cute",
        static_shapes=True,
        config=helion.Config(**dict(seed.config)),
    )
    y, total = kernel(x)
    assert torch.equal(y, x.contiguous())
    torch.testing.assert_close(
        total.double(), x.double().sum(dim=1), rtol=1e-4, atol=1e-3
    )
