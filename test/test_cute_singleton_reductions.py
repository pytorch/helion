"""Reductions over extent-1 dims on the CuTe backend.

Inductor lowers a reduction whose reduced dims all have extent 1
element-wise, so the reduction's input has more dims than its result:
``x[tile][:, :, None].sum(-1)`` reads a ``[tile0, tile1, 1]`` input for a
``[tile0, tile1]`` result.  That input used to be right-aligned against the
result, pairing ``tile1`` with ``tile0``: the re-binding check raised
``ShapeMismatch`` for unequal block sizes and exchanged the transposed
element through shared memory (``rebind_smem``) for equal ones, and the
literal-dims analysis refused a full-row slice.  The computed-fragment
selection read the input's shape as the result's, sizing a sort of the
result by the unit dim.  The values are checked on the GPU at the end.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING
from typing import Any

import pytest
import torch

import helion
from helion._compiler.cute.fragment_context import _static_selection_extent
from helion._compiler.inductor_lowering import size1_reduction_positions
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sum_over_trailing_unit_dim(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = (x[tile][:, :, None] * 2.0).sum(-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mean_over_leading_and_trailing_unit_dims(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = x[tile][None, :, :, None].mean((0, 3))
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _amax_over_unsqueezed_3d_tile(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = torch.amax(x[tile].unsqueeze(-1) + 1.0, dim=-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _amax_and_mean_over_unit_dims_3d(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        v = x[tile]
        out[tile] = torch.amax(v.unsqueeze(-1) * 2.0, dim=-1) + v.unsqueeze(0).mean(0)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _max_over_trailing_unit_dim(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = torch.max(x[tile][:, :, None] * 2.0, -1).values
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sum_over_unit_dim_of_full_rows(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        rows = hl.zeros([tile_m, n], dtype=torch.float32) + x[tile_m, :]
        out[tile_m, :] = rows[:, :, None].sum(-1) * 2.0
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mean_keepdim_over_unit_dim(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size()):
        out[tile] = (x[tile][:, :, None] * 2.0).mean(-1, keepdim=True).squeeze(-1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _sort_of_sum_over_unit_dim(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.empty_like(x)
    indices = torch.empty_like(x, dtype=torch.int64)
    for row in hl.tile(x.size(0)):
        value, index = torch.sort(
            (x[row, :][:, :, None] * 2.0).sum(-1), dim=-1, stable=True
        )
        values[row, :] = value
        indices[row, :] = index
    return values, indices


@functools.cache
def _bound(kernel: Kernel[Any], shape: tuple[int, ...]) -> BoundKernel[Any]:
    return cpu_bind(kernel, (torch.empty(shape),))


@pytest.mark.parametrize(
    ("kernel", "shape", "block_sizes"),
    [
        # Unequal block sizes raised ShapeMismatch, equal ones exchanged.
        pytest.param(_sum_over_trailing_unit_dim, (16, 32), [8, 16], id="sum"),
        pytest.param(
            _sum_over_trailing_unit_dim, (16, 32), [16, 16], id="sum-equal-blocks"
        ),
        pytest.param(
            _mean_over_leading_and_trailing_unit_dims, (16, 32), [8, 16], id="mean"
        ),
        pytest.param(
            _mean_over_leading_and_trailing_unit_dims,
            (16, 32),
            [16, 16],
            id="mean-equal-blocks",
        ),
        pytest.param(_amax_over_unsqueezed_3d_tile, (4, 8, 16), [4, 8, 8], id="amax"),
        pytest.param(
            _amax_over_unsqueezed_3d_tile, (4, 8, 16), [8, 8, 8], id="amax-equal-blocks"
        ),
        # max(dim)'s values lower the same way, as an element-wise node.
        pytest.param(_max_over_trailing_unit_dim, (16, 32), [8, 16], id="max-dim"),
        # The literal-dims analysis refused it: "combines dim 1 of subscript
        # (shape [u0, 8, 1]), held by the reduction block of its extent, with a
        # dim held by block id 0's tile".
        pytest.param(_sum_over_unit_dim_of_full_rows, (16, 8), [8], id="full-rows"),
        # keepdim keeps the input's rank, so it was aligned already.
        pytest.param(_mean_keepdim_over_unit_dim, (16, 32), [16, 16], id="keepdim"),
    ],
)
def test_reduction_over_unit_dims_reads_own_element(
    kernel: Kernel[Any], shape: tuple[int, ...], block_sizes: list[int]
) -> None:
    code = _bound(kernel, shape).to_code(helion.Config(block_sizes=block_sizes))
    assert "rebind_smem" not in code


def test_selection_extent_of_reduction_over_unit_dim() -> None:
    bound = _bound(_sort_of_sum_over_unit_dim, (8, 16))
    (total,) = [
        node
        for info in bound.host_function.device_ir.graphs
        for node in info.graph.nodes
        if node.target is torch.ops.aten.sum.dim_IntList
    ]
    with bound.env, bound.host_function:
        # Was 1, the trailing dim of the sum's [row, 16, 1] input.
        assert _static_selection_extent(bound.env, total) == 16


def test_size1_reduction_positions_pairs_equal_extents_only() -> None:
    # The input's dims wider than 1 meet the result's in order and its unit
    # dims none; a pair whose extents differ makes no size-1 reduction input.
    with _bound(_sum_over_trailing_unit_dim, (16, 32)).env:
        assert size1_reduction_positions([16, 32, 1], [16, 32]) == [0, 1, -1]
        assert size1_reduction_positions([1, 16, 1, 32, 1], [16, 32]) == [
            -1,
            0,
            -1,
            1,
            -1,
        ]
        assert size1_reduction_positions([16, 32, 1], [32, 16]) is None
        assert size1_reduction_positions([16, 32, 4], [16, 32]) is None


@pytest.mark.cuda
@skipIfNotCUDA()
@onlyBackends(["cute"])
class TestReductionOverUnitDimsOnTheGpu(TestCase):
    def test_reduction_keeps_thread_element(self) -> None:
        # Right-aligned against the result, the input's tile1 met tile0:
        # unequal block sizes raised ShapeMismatch, equal ones exchanged
        # out[i, j] = 2 * x[i, i] within each tile.  Ragged tails exercise the
        # tile masks.
        torch.manual_seed(0)
        x = torch.randn(48, 40, device=DEVICE)
        for block_sizes in ([16, 16], [16, 32]):
            with self.subTest(block_sizes=block_sizes):
                _, out = code_and_output(
                    _sum_over_trailing_unit_dim, (x,), block_sizes=block_sizes
                )
                torch.testing.assert_close(out, (x[:, :, None] * 2.0).sum(-1))
        x = torch.randn(4, 24, 20, device=DEVICE)
        _, out = code_and_output(
            _amax_and_mean_over_unit_dims_3d, (x,), block_sizes=[8, 8, 8]
        )
        torch.testing.assert_close(
            out, torch.amax(x.unsqueeze(-1) * 2.0, dim=-1) + x.unsqueeze(0).mean(0)
        )

    def test_sort_of_reduction(self) -> None:
        # The computed-fragment selection took the sort's width from the
        # sum's [row, 16, 1] input, 1, and filled each row with its first
        # element at index 0.
        torch.manual_seed(0)
        x = torch.randn(8, 16, device=DEVICE)
        _, (values, indices) = code_and_output(_sort_of_sum_over_unit_dim, (x,))
        expected_values, expected_indices = torch.sort(x * 2.0, dim=-1, stable=True)
        torch.testing.assert_close(values, expected_values)
        torch.testing.assert_close(indices, expected_indices)
