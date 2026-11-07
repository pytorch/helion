"""MMA tile-index expressions retain the selected integer width."""

from __future__ import annotations

import re

import pytest
import torch

import helion
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_mma_indices import causal_dot
from helion._testing.cute_mma_indices import masked_dot
from helion._testing.cute_mma_indices import recurrent_dot

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_mma_root_and_loop_index_width(mode, dtype):
    kernel = helion.kernel(
        masked_dot.fn, backend="cute", static_shapes=True, index_dtype=dtype
    )
    with cpu_target():
        bound = cpu_bind(
            kernel,
            (
                torch.empty(128, 16, dtype=torch.bfloat16),
                torch.empty(16, 16, dtype=torch.bfloat16),
            ),
        )
        assert bound.config_spec.cute_mma_dag_facts is not None
        source = bound.to_code(
            helion.Config(
                block_sizes=[16, 64],
                cute_mma_dag=mode,
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
            )
        )
    assert "_helion_mma_dag" in source
    if dtype == torch.int64:
        assert re.search(r"md_v\w* = cutlass.Int64\(md_a", source)
        assert not re.search(r"md_v\w* = cutlass.Int32\(md_a", source)
        assert "md_a0 = cutlass.Int64(" in source
        assert "cutlass.Int64(md_i) * 64" in source
    else:
        assert re.search(r"md_v\w* = cutlass.Int32\(md_a", source)


@pytest.mark.parametrize("mode", ["warp", "tcgen05"])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_varying_loop_bound_retains_index_width(mode, dtype):
    kernel = helion.kernel(
        causal_dot.fn, backend="cute", static_shapes=True, index_dtype=dtype
    )
    with cpu_target():
        bound = cpu_bind(
            kernel,
            (
                torch.empty(128, 128, dtype=torch.bfloat16),
                torch.empty(128, 16, dtype=torch.bfloat16),
            ),
        )
        source = bound.to_code(
            helion.Config(
                block_sizes=[64, 16, 64],
                cute_mma_dag=mode,
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
            )
        )
    width = "Int64" if dtype == torch.int64 else "Int32"
    assert f"md_end0 = cutlass.{width}(" in source
    assert f"md_trips0 = cutlass.{width}(" in source
    assert "cutlass.range(0, md_trips0)" in source


def test_legacy_tcgen05_loop_base_retains_int64():
    kernel = helion.kernel(
        recurrent_dot.fn, backend="cute", static_shapes=True, index_dtype=torch.int64
    )
    with cpu_target():
        bound = cpu_bind(
            kernel,
            (
                torch.empty(64, 128, dtype=torch.bfloat16),
                torch.empty(128, 16, dtype=torch.bfloat16),
            ),
        )
        source = bound.to_code(
            helion.Config(
                block_sizes=[64, 16, 64],
                cute_mma_dag="tcgen05",
                cute_mma_dag_warps=4,
                cute_mma_dag_stages=2,
            )
        )
    assert "cutlass.Int64(md_i) * 64" in source
    assert "md_round = " in source  # Legacy emitter; regions use md_rd{loop}.
