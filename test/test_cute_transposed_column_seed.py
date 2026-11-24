"""``cute_transposed_column_reduction``: a device-loop column sum whose input
is contiguous along the reduced rows gets the register-transpose seed (two
threads along the rows, each loading one packet of them, sixteen along the
columns).  Seed checks on CPU."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._compiler.cute.autotune.heuristics import (
    CuteTransposedColumnReductionHeuristic,
)
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.transposed_column_kernels import copy_and_column_sums
from helion._testing.transposed_column_kernels import permuted
from helion._testing.transposed_column_kernels import row_sums

pytestmark = pytest.mark.cpu_only


def _seed(
    x: torch.Tensor, kernel: helion.Kernel = copy_and_column_sums
) -> dict[str, object] | None:
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, (x,))
        with bound.env:
            seed = CuteTransposedColumnReductionHeuristic.get_seed_config(
                bound.env, bound.host_function.device_ir
            )
        if seed is None:
            return None
        config = dict(seed.config)
        # the seed is a valid config of the kernel, and renders
        bound.to_code(helion.Config(**config))
        assert any(
            dict(s.config) == config for s in bound.config_spec.compiler_seed_configs
        )
    return {
        key: config[key] for key in ("block_sizes", "num_threads", "cute_vector_widths")
    }


@pytest.mark.parametrize(
    ("dtype", "rows", "expected_rows", "expected_vec"),
    [
        # 32-byte packets of the rows: 16 bf16 or 8 fp32 elements, two threads
        (torch.bfloat16, 128, 32, 16),
        (torch.float32, 128, 16, 8),
        # 100 rows hold no 16- or 8-element packet: four-element ones
        (torch.bfloat16, 100, 8, 4),
    ],
    ids=["bf16", "fp32", "rows_not_a_multiple_of_the_packet"],
)
def test_a_transposed_column_sum_gets_the_register_transpose(
    dtype: torch.dtype, rows: int, expected_rows: int, expected_vec: int
) -> None:
    seed = _seed(permuted(4, rows, 48, dtype, "cpu"))
    assert seed == {
        "block_sizes": [1, 16, expected_rows],
        "num_threads": [1, 16, expected_rows // expected_vec],
        "cute_vector_widths": [1, 1, expected_vec],
    }


def test_narrow_columns_clamp_the_tile() -> None:
    # 6 columns: the column tile takes the smallest block that holds them
    seed = _seed(permuted(4, 128, 6, torch.bfloat16, "cpu"))
    assert seed is not None
    assert seed["block_sizes"] == [1, 8, 32]
    assert seed["num_threads"] == [1, 8, 2]


def test_a_column_contiguous_input_gets_no_seed() -> None:
    # The columns on the stride-1 dim are the column heuristics' case.
    x = torch.randn(4, 128, 48, dtype=torch.bfloat16)
    assert _seed(x) is None


def test_a_row_sum_over_the_contiguous_dim_gets_no_seed() -> None:
    # The loop's tile on the input's stride-1 dim, but summed innermost: the
    # rows are contiguous already (the row heuristics' case).
    x = torch.randn(4, 48, 1493, dtype=torch.bfloat16)
    assert _seed(x, row_sums) is None
