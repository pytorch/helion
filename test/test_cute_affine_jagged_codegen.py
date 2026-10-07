from __future__ import annotations

from unittest.mock import patch

from examples.jagged_layer_norm import jagged_layer_norm_kernel
import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])

CPU_DEVICE = "cpu"


@pytest.mark.parametrize("columns", [4, 69, 128])
@pytest.mark.parametrize("rows_per_tile", [1, 2])
def test_dynamic_jagged_row_masks_keep_affine_vector_loads(
    columns: int, rows_per_tile: int
) -> None:
    # Match the original CUDA test's odd sequence count and dynamic binding. The
    # last two-row tile predicates the Int64 offset load before the vector loops.
    rows = sum((0, 1, 2, 9, 257, 0, 513, 7, 65))
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        kernel = helion.kernel(
            jagged_layer_norm_kernel.fn,
            backend="cute",
            cute_flatten_nested_reductions=True,
            static_shapes=False,
            autotune_effort="none",
        )
        values = torch.empty(rows * columns, device=CPU_DEVICE).view(rows, columns)
        offsets = torch.empty(10, dtype=torch.int64, device=CPU_DEVICE)
        bound = _cpu_bind(kernel, (values, offsets, 1e-6))
        assert len(bound.config_spec.block_sizes) == 4
        code = bound.to_code(
            helion.Config(
                block_sizes=[rows_per_tile, 512, 512, 512],
                num_threads=[rows_per_tile, 128, 128, 128],
                cute_vector_widths=[1, 4, 4, 4],
                cute_lane_layouts=["strided"] * 4,
            )
        )
    assert "_helion_affine_load" in code
