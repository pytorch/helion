from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import import_example
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

jagged_layer_norm_kernel = import_example("jagged_layer_norm").jagged_layer_norm_kernel
jagged_mean_kernel = import_example("jagged_mean").jagged_mean_kernel

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]

CPU_DEVICE = "cpu"


def test_computed_rows_with_isfinite_pack_the_implicit_feature_axis() -> None:
    @helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
    def kernel(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        batch, rows = weight.shape
        columns = values.size(2)
        out = torch.empty((batch, rows, columns), device=values.device)
        source = values.view(batch * rows * 2, columns)
        output = out.view(batch * rows, columns)
        for request, row in hl.tile([batch, rows], block_size=[1, 8]):
            index = request.id * rows * 2 + row.index
            first = source[index, :]
            second = source[index + rows, :]
            w = weight[request.id, row]
            merged = first * w[:, None] + second * (1.0 - w[:, None])
            result = torch.where(torch.isfinite(merged), merged, 0.0)
            output[request.id * rows + row.index, :] = result
        return out

    # An odd row count exercises the computed row's padding guard while the
    # implicit feature axis has four consecutive scalar values per lane.
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, (torch.empty(3, 38, 128), torch.empty(3, 19)))
        cfg = bound.config_spec.default_config()
        cfg.config.update(num_threads=[32], cute_vector_widths=[4, 1, 1])
        code = bound.to_code(cfg)
    assert "synthetic_lane_" in code
    assert "_helion_affine_load" in code, code
    assert "ir.VectorType.get([4]" in code
    assert "_cute_store_u32_vec" in code


@pytest.mark.parametrize("columns", [4, 69, 128])
@pytest.mark.parametrize("rows_per_tile", [1, 2])
def test_dynamic_jagged_row_masks_keep_affine_vector_loads(
    columns: int, rows_per_tile: int
) -> None:
    # Match the original CUDA test's odd sequence count and dynamic binding. The
    # last two-row tile predicates the Int64 offset load before the vector loops.
    rows = sum((0, 1, 2, 9, 257, 0, 513, 7, 65))
    with (
        mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch(
            "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
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
        bound = cpu_bind(kernel, (values, offsets, 1e-6))
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


@pytest.mark.parametrize("static_shapes", [False, True])
def test_vector_width_on_a_jagged_feature_axis_stays_scalar(
    static_shapes: bool,
) -> None:
    # The feature tile is ``hl.jagged_tile(feature_counts)``: it has no static
    # extent to prove a whole number of packets, so its loads stay scalar.
    # The packet check used to read the jagged block's numel and assert.
    with (
        mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch(
            "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        kernel = helion.kernel(
            jagged_mean_kernel.fn,
            backend="cute",
            static_shapes=static_shapes,
            autotune_effort="none",
        )
        args = (
            torch.empty((731, 128), device=CPU_DEVICE),
            torch.empty(18, dtype=torch.int64, device=CPU_DEVICE),
            torch.empty(17, dtype=torch.int32, device=CPU_DEVICE),
            128,
        )
        bound = cpu_bind(kernel, args)
        code = bound.to_code(
            helion.Config(
                block_sizes=[1, 128, 64],
                num_threads=[1, 32, 1],
                cute_vector_widths=[1, 4, 1],
            )
        )
    assert "for vec_lane_1 in cutlass.range_constexpr(4):" in code
    assert "cute.arch.load(" not in code
