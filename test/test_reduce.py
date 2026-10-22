from __future__ import annotations

import operator
import unittest
from unittest.mock import PropertyMock
from unittest.mock import patch

import torch
from torch.fx.experimental.proxy_tensor import make_fx

import helion
from helion import exc
from helion._compiler.autotuner_heuristics.triton import TritonNarrowReductionHeuristic
from helion._compiler.autotuner_heuristics.triton import TritonReductionHeuristic
from helion._compiler.cute.canonicalize_reductions import _independent_combines
from helion._compiler.cute.canonicalize_reductions import canonicalize_reductions
from helion._compiler.device_ir import DeviceIR
from helion._compiler.device_ir import HelperFunctionGraphInfo
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import _get_backend
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfFn
from helion._testing import skipIfRefEager
import helion.language as hl
from helion.language._tracing_ops import _mask_to
from helion.language.reduce_ops import _reduce


def add_combine_fn(x, y):
    """Simple addition combine function for sum reduction."""
    return x + y


def max_combine_fn(x, y):
    """Maximum combine function for max reduction."""
    return torch.maximum(x, y)


def mul_combine_fn(x, y):
    """Multiplication combine function for product reduction."""
    return x * y


def min_combine_fn(x, y):
    """Minimum combine function for min reduction."""
    return torch.minimum(x, y)


def max_min_combine_fn(left_max, left_min, right_max, right_min):
    """Elementwise max of the first operands and min of the second ones."""
    return torch.maximum(left_max, right_max), torch.minimum(left_min, right_min)


def scaled_max_combine_fn(x, y):
    """Ends in ``torch.maximum`` but scales its right operand first."""
    return torch.maximum(x, y * 2)


def keep_left_max_combine_fn(x, y):
    """``torch.maximum`` of the left operand with itself: keeps the first
    element."""
    return torch.maximum(x, x)


def tuple_add_alpha_combine_fn(left_x, left_y, right_x, right_y):
    """Adds the right operand twice to the first tuple element."""
    return torch.add(left_x, right_x, alpha=2), left_y + right_y


def tuple_add_combine_fn(left_tuple, right_tuple):
    """Tuple combine function for tuple reduction."""
    left_values, left_indices = left_tuple
    right_values, right_indices = right_tuple
    combined_values = left_values + right_values
    combined_indices = left_indices + right_indices
    return combined_values, combined_indices


def argmax_combine_fn(left_tuple, right_tuple):
    """Combine function for argmax: returns (value, index) of maximum element."""
    left_value, left_index = left_tuple
    right_value, right_index = right_tuple

    # If right value is greater, take right; otherwise keep left
    take_right = right_value > left_value
    max_value = torch.where(take_right, right_value, left_value)
    max_index = torch.where(take_right, right_index, left_index)

    return max_value, max_index


def tuple_add_combine_unpacked_fn(
    left_values, left_indices, right_values, right_indices
):
    """Tuple combine function for tuple reduction (unpacked format)."""
    combined_values = left_values + right_values
    combined_indices = left_indices + right_indices
    return combined_values, combined_indices


def argmax_combine_unpacked_fn(left_value, left_index, right_value, right_index):
    """Combine function for argmax (unpacked format): returns (value, index) of maximum element."""
    # If right value is greater, take right; otherwise keep left
    take_right = right_value > left_value
    max_value = torch.where(take_right, right_value, left_value)
    max_index = torch.where(take_right, right_index, left_index)

    return max_value, max_index


def argmax_tie_right_fn(left_value, left_index, right_value, right_index):
    """argmax whose ties keep the right operand (the last tied element)."""
    take_left = left_value > right_value
    return (
        torch.where(take_left, left_value, right_value),
        torch.where(take_left, left_index, right_index),
    )


def argmax_ge_fn(left_value, left_index, right_value, right_index):
    """argmax written with ``>=``; ties keep the left operand."""
    take_left = left_value >= right_value
    return (
        torch.where(take_left, left_value, right_value),
        torch.where(take_left, left_index, right_index),
    )


def argmin_le_fn(left_value, left_index, right_value, right_index):
    """argmin written with ``<=``; ties keep the right operand."""
    take_right = right_value <= left_value
    return (
        torch.where(take_right, right_value, left_value),
        torch.where(take_right, right_index, left_index),
    )


def argmax_le_fn(left_value, left_index, right_value, right_index):
    """argmax written with ``<=``; ties keep the right operand."""
    take_right = left_value <= right_value
    return (
        torch.where(take_right, right_value, left_value),
        torch.where(take_right, right_index, left_index),
    )


def argmin_lt_fn(left_value, left_index, right_value, right_index):
    """argmin written with ``<``; ties keep the left operand."""
    take_right = right_value < left_value
    return (
        torch.where(take_right, right_value, left_value),
        torch.where(take_right, right_index, left_index),
    )


def argmin_keep_left_le_fn(left_value, left_index, right_value, right_index):
    """argmin choosing the left operand when its test holds; ties keep it."""
    take_left = left_value <= right_value
    return (
        torch.where(take_left, left_value, right_value),
        torch.where(take_left, left_index, right_index),
    )


def argmin_keep_left_gt_fn(left_value, left_index, right_value, right_index):
    """argmin choosing the left operand when the right one is greater; ties
    keep the right operand."""
    take_left = right_value > left_value
    return (
        torch.where(take_left, left_value, right_value),
        torch.where(take_left, left_index, right_index),
    )


# Every argmax/argmin form ``torch.where`` combines take: either operand kept
# on a tie, and either operand kept when the test is false (as it is for a
# NaN).
ARG_REDUCE_COMBINES = (
    argmax_combine_unpacked_fn,
    argmax_le_fn,
    argmax_ge_fn,
    argmax_tie_right_fn,
    argmin_lt_fn,
    argmin_le_fn,
    argmin_keep_left_le_fn,
    argmin_keep_left_gt_fn,
)


@helion.kernel(autotune_effort="none")
def arg_reduce_combines(
    x: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Column ``k`` of each result reduces ``(x, indices)`` with
    ``ARG_REDUCE_COMBINES[k]``."""
    values = torch.empty([x.size(0), 8], dtype=x.dtype, device=x.device)
    chosen = torch.empty([x.size(0), 8], dtype=indices.dtype, device=x.device)
    for i in hl.tile(x.size(0)):
        pair = (x[i, :], indices[i, :])
        values[i, 0], chosen[i, 0] = hl.reduce(argmax_combine_unpacked_fn, pair, dim=1)
        values[i, 1], chosen[i, 1] = hl.reduce(argmax_le_fn, pair, dim=1)
        values[i, 2], chosen[i, 2] = hl.reduce(argmax_ge_fn, pair, dim=1)
        values[i, 3], chosen[i, 3] = hl.reduce(argmax_tie_right_fn, pair, dim=1)
        values[i, 4], chosen[i, 4] = hl.reduce(argmin_lt_fn, pair, dim=1)
        values[i, 5], chosen[i, 5] = hl.reduce(argmin_le_fn, pair, dim=1)
        values[i, 6], chosen[i, 6] = hl.reduce(argmin_keep_left_le_fn, pair, dim=1)
        values[i, 7], chosen[i, 7] = hl.reduce(argmin_keep_left_gt_fn, pair, dim=1)
    return values, chosen


@helion.kernel(autotune_effort="none")
def arg_reduce_tile_dim(
    x: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``arg_reduce_combines`` with two of the combines, over a tile of each
    row rather than a reduction dim (the tile holds the whole row)."""
    values = torch.empty([x.size(0), 2], dtype=x.dtype, device=x.device)
    chosen = torch.empty([x.size(0), 2], dtype=indices.dtype, device=x.device)
    for i, j in hl.tile([x.size(0), x.size(1)], block_size=[4, 256]):
        pair = (x[i, j], indices[i, j])
        values[i, 0], chosen[i, 0] = hl.reduce(argmax_combine_unpacked_fn, pair, dim=1)
        values[i, 1], chosen[i, 1] = hl.reduce(argmin_keep_left_gt_fn, pair, dim=1)
    return values, chosen


def _rows_with_nans(n: int, dtype: torch.dtype) -> torch.Tensor:
    """Rows of small integers (many ties) with a NaN at the start, inside, at
    the end, two inside, at both ends, everywhere, or nowhere."""
    x = torch.randint(0, 4, (16, n), device=DEVICE).to(dtype)
    nan = float("nan")
    x[0, 0] = nan
    x[1, 3] = nan
    x[2, -1] = nan
    x[3, 1] = x[3, 5] = nan
    x[4, 0] = x[4, -1] = nan
    x[5, :] = nan
    return x


def _left_fold_choice(
    x: torch.Tensor, indices: torch.Tensor, extreme: torch.Tensor, *, last: bool
) -> torch.Tensor:
    """The index operand of the first (or last) element of each row of ``x``
    equal to ``extreme``."""
    n = x.size(1)
    positions = torch.arange(n, device=x.device).expand_as(x)
    tied = x == extreme[:, None]
    if last:
        chosen = torch.where(tied, positions, -1).amax(1)
    else:
        chosen = torch.where(tied, positions, n).amin(1)
    return indices.gather(1, chosen[:, None])[:, 0]


@helion.kernel(autotune_effort="none")
def arg_reductions_with_ties(
    x: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    first_max = torch.empty([x.size(0)], dtype=torch.int64, device=x.device)
    last_max = torch.empty_like(first_max)
    ge_max = torch.empty_like(first_max)
    last_min = torch.empty_like(first_max)
    for i in hl.tile(x.size(0)):
        row = x[i, :]
        row_indices = indices[i, :]
        _, first_max[i] = hl.reduce(
            argmax_combine_unpacked_fn,
            (row, row_indices),
            dim=1,
            other=(float("-inf"), 0),
        )
        _, last_max[i] = hl.reduce(
            argmax_tie_right_fn, (row, row_indices), dim=1, other=(float("-inf"), 0)
        )
        _, ge_max[i] = hl.reduce(
            argmax_ge_fn, (row, row_indices), dim=1, other=(float("-inf"), 0)
        )
        _, last_min[i] = hl.reduce(
            argmin_le_fn, (row, row_indices), dim=1, other=(float("inf"), 0)
        )
    return first_max, last_max, ge_max, last_min


@helion.kernel
def jit_add_combine_fn(x, y):
    """Addition combine function with @helion.kernel decorator (should be ignored)."""
    return x + y


@onlyBackends(["triton", "cute"])
class TestReduce(RefEagerTestBase, TestCase):
    def test_reduce_basic_sum(self):
        """Test basic reduce functionality with sum reduction along a dimension."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]  # Shape: [TILE_SIZE, seq_len]
                result[i] = hl.reduce(add_combine_fn, row_data, dim=1)
            return result

        # Create test input
        x = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
            device=DEVICE,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_reduce_kernel, (x,))

        # Test the actual reduce operation
        expected = torch.tensor([10.0, 26.0, 42.0], device=DEVICE)
        torch.testing.assert_close(result, expected)

        if _get_backend() == "cute":
            self.assertIn("cute.arch.warp_reduction_sum", code)
        else:
            # Check that the generated code contains triton reduce calls
            self.assertIn("tl.reduce", code)
            self.assertIn("add_combine_fn_", code)

    def test_reduce_max(self):
        """Test reduce with maximum operation."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_max_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i] = hl.reduce(max_combine_fn, row_data, dim=1)
            return result

        # Create test input
        x = torch.tensor(
            [[1.0, 4.0, 2.0, 3.0], [8.0, 5.0, 7.0, 6.0], [9.0, 12.0, 10.0, 11.0]],
            device=DEVICE,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_reduce_max_kernel, (x,))

        # Test the actual reduce operation
        expected = torch.tensor([4.0, 8.0, 12.0], device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_reduce_with_keep_dims(self):
        """Test reduce with keep_dims=True."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_keep_dims_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0), 1], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i, :] = hl.reduce(
                    add_combine_fn, row_data, dim=1, keep_dims=True
                )
            return result

        # Create test input
        x = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            device=DEVICE,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_reduce_keep_dims_kernel, (x,))

        # Test the actual reduce operation
        expected = torch.tensor([[10.0], [26.0]], device=DEVICE)
        torch.testing.assert_close(result, expected)

        if _get_backend() != "cute":
            # Triton lowers this via tl.reduce(..., keep_dims=True)
            self.assertIn("keep_dims=True", code)

    def test_reduce_all_dims(self):
        """Test reduce with dim=None (reduce all dimensions)."""

        @helion.kernel(autotune_effort="none")
        def reduce_all_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.grid(x.size(0)):
                result[i] = hl.reduce(add_combine_fn, x[i, :])
            return result

        x = torch.randn(3, 200, device=DEVICE)
        _, result = code_and_output(reduce_all_kernel, (x,))
        torch.testing.assert_close(result, x.sum(1), rtol=1e-4, atol=1e-4)

        @helion.kernel(autotune_effort="none")
        def reduce_all_2d_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.grid(x.size(0)):
                result[i] = hl.reduce(add_combine_fn, x[i, :, :])
            return result

        x = torch.randn(3, 4, 8, device=DEVICE)
        _, result = code_and_output(reduce_all_2d_kernel, (x,))
        torch.testing.assert_close(result, x.sum((1, 2)), rtol=1e-4, atol=1e-4)

    def test_reduce_rows_wider_than_a_warp(self):
        """Built-in combines over rows of 200 elements, wider than one warp."""

        @helion.kernel(autotune_effort="none")
        def wide_rows_kernel(
            x: torch.Tensor, indices: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            row_sum = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            row_max = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            row_argmax = torch.empty([x.size(0)], dtype=torch.int64, device=x.device)
            for i in hl.tile(x.size(0)):
                row = x[i, :]
                row_sum[i] = hl.reduce(add_combine_fn, row, dim=1)
                row_max[i] = hl.reduce(max_combine_fn, row, dim=1, other=float("-inf"))
                _, row_argmax[i] = hl.reduce(
                    argmax_combine_unpacked_fn,
                    (row, indices[i, :]),
                    dim=1,
                    other=(float("-inf"), 0),
                )
            return row_sum, row_max, row_argmax

        x = torch.randn(32, 200, device=DEVICE)
        indices = torch.arange(200, device=DEVICE).expand(32, 200).contiguous()
        _, (row_sum, row_max, row_argmax) = code_and_output(
            wide_rows_kernel, (x, indices), block_size=4
        )
        torch.testing.assert_close(row_sum, x.sum(1), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(row_max, x.amax(1))
        torch.testing.assert_close(row_argmax, x.argmax(1))

    @skipIfFn(
        lambda: _get_backend() != "cute",
        "Triton's tree reduction does not keep hl.reduce's left-fold order "
        "among tied elements",
    )
    def test_reduce_arg_reduction_tie_rules(self):
        """A tied extreme ends the left fold on the first element, or the last
        when the combine keeps the right operand of a tie; the index operand
        of that element is returned even when it does not increase."""
        torch.manual_seed(0)
        for n in (8, 200):
            x = torch.randint(0, 3, (16, n), device=DEVICE).float()
            positions = torch.arange(n, device=DEVICE).expand(16, n)
            for indices in (positions.contiguous(), (n - 1 - positions).contiguous()):
                _, results = code_and_output(
                    arg_reductions_with_ties, (x, indices), block_size=4
                )
                maxes, mins = x.amax(1), x.amin(1)
                first_max, last_max, ge_max, last_min = results
                for result, extreme, last in (
                    (first_max, maxes, False),
                    (last_max, maxes, True),
                    (ge_max, maxes, False),
                    (last_min, mins, True),
                ):
                    torch.testing.assert_close(
                        result, _left_fold_choice(x, indices, extreme, last=last)
                    )

    @skipIfFn(
        lambda: _get_backend() != "cute",
        "Triton's tree reduction does not keep hl.reduce's left-fold order, "
        "which decides where a NaN ends up",
    )
    def test_reduce_arg_reduction_nan_rows_match_left_fold(self):
        """A comparison with a NaN is false, so each ``torch.where`` form
        keeps or drops NaNs as the left fold in ref eager mode does: a value
        and an index operand of the row, never a sentinel."""
        torch.manual_seed(0)
        for kernel, combines, config in (
            (arg_reduce_combines, ARG_REDUCE_COMBINES, {"block_size": 4}),
            (
                arg_reduce_tile_dim,
                (argmax_combine_unpacked_fn, argmin_keep_left_gt_fn),
                {},
            ),
        ):
            reference = helion.kernel(kernel.fn, ref_mode=helion.RefMode.EAGER)
            for n, dtype in ((8, torch.float32), (200, torch.float16)):
                x = _rows_with_nans(n, dtype)
                indices = torch.arange(n, 0, -1, device=DEVICE).expand(16, n)
                indices = indices.contiguous()
                _, (values, chosen) = code_and_output(kernel, (x, indices), **config)
                expected_values, expected_chosen = reference(x.cpu(), indices.cpu())
                for k, combine_fn in enumerate(combines):
                    with self.subTest(
                        kernel=kernel.fn.__name__, combine_fn=combine_fn.__name__, n=n
                    ):
                        torch.testing.assert_close(
                            values[:, k].cpu(),
                            expected_values[:, k],
                            rtol=0,
                            atol=0,
                            equal_nan=True,
                        )
                        torch.testing.assert_close(
                            chosen[:, k].cpu(), expected_chosen[:, k]
                        )

    def test_reduce_builtin_combines_propagate_nan(self):
        """``torch.maximum``/``torch.minimum``, ``+`` and ``*`` combines turn
        a row with a NaN into NaN, as the left fold in ref eager mode does."""

        @helion.kernel(autotune_effort="none")
        def builtin_combines_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty([x.size(0), 6], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row = x[i, :]
                out[i, 0] = hl.reduce(max_combine_fn, row, dim=1, other=float("-inf"))
                out[i, 1] = hl.reduce(min_combine_fn, row, dim=1, other=float("inf"))
                out[i, 2], out[i, 3] = hl.reduce(
                    max_min_combine_fn,
                    (row, row),
                    dim=1,
                    other=(float("-inf"), float("inf")),
                )
                out[i, 4] = hl.reduce(add_combine_fn, row, dim=1)
                out[i, 5] = hl.reduce(mul_combine_fn, row, dim=1, other=1.0)
            return out

        reference = helion.kernel(
            builtin_combines_kernel.fn, ref_mode=helion.RefMode.EAGER
        )
        torch.manual_seed(0)
        for n in (8, 200):
            x = _rows_with_nans(n, torch.float32)
            _, out = code_and_output(builtin_combines_kernel, (x,), block_size=4)
            torch.testing.assert_close(
                out.cpu(), reference(x.cpu()), rtol=1e-5, atol=0, equal_nan=True
            )

    @skipIfRefEager("ref eager mode runs the combine function itself")
    @skipIfFn(
        lambda: _get_backend() != "cute",
        "Triton lowers the combine function itself",
    )
    def test_reduce_refuses_combines_ending_in_a_builtin(self):
        """A combine whose last op is a built-in reduction's, applied to
        anything but the two operands (or with an extra argument), is not that
        reduction: it must be refused rather than lowered as one."""

        @helion.kernel(autotune_effort="none")
        def scaled_max_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                out[i] = hl.reduce(scaled_max_combine_fn, x[i, :], dim=1)
            return out

        @helion.kernel(autotune_effort="none")
        def keep_left_max_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                out[i] = hl.reduce(keep_left_max_combine_fn, x[i, :], dim=1)
            return out

        @helion.kernel(autotune_effort="none")
        def tuple_add_alpha_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                out[i], _ = hl.reduce(
                    tuple_add_alpha_combine_fn, (x[i, :], x[i, :]), dim=1
                )
            return out

        x = torch.randn(8, 16, device=DEVICE)
        for kernel in (scaled_max_kernel, keep_left_max_kernel, tuple_add_alpha_kernel):
            with (
                self.subTest(kernel=kernel.fn.__name__),
                self.assertRaisesRegex(exc.BackendUnsupported, "custom combine"),
            ):
                code_and_output(kernel, (x,), block_size=4)

    def test_reduce_min(self):
        """Test reduce with minimum operation."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_min_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i] = hl.reduce(min_combine_fn, row_data, dim=1)
            return result

        # Create test input
        x = torch.tensor(
            [[4.0, 1.0, 3.0, 2.0], [8.0, 5.0, 7.0, 6.0], [12.0, 9.0, 11.0, 10.0]],
            device=DEVICE,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_reduce_min_kernel, (x,))

        # Test the actual reduce operation
        expected = torch.tensor([1.0, 5.0, 9.0], device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_reduce_product(self):
        """Test reduce with multiplication operation using other=1."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_product_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i] = hl.reduce(mul_combine_fn, row_data, dim=1, other=1.0)
            return result

        # Create test input with non-power-2 size (3 elements)
        x = torch.tensor(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [1.0, 1.0, 5.0]],
            device=DEVICE,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_reduce_product_kernel, (x,))

        # Test the actual reduce operation
        expected = torch.tensor([6.0, 24.0, 5.0], device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_reduce_jit_combine_fn(self):
        """Test reduce with @helion.kernel decorated combine function."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_jit_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i] = hl.reduce(jit_add_combine_fn, row_data, dim=1)
            return result

        # Create test input
        x = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            device=DEVICE,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_reduce_jit_kernel, (x,))

        # Test the actual reduce operation
        expected = torch.tensor([10.0, 26.0], device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_reduce_tuple_input(self):
        """Test reduce with tuple input."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_tuple_kernel(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_x = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            result_y = torch.empty([y.size(0)], dtype=y.dtype, device=y.device)

            for i in hl.tile(x.size(0)):
                row_x = x[i, :]
                row_y = y[i, :]
                input_tuple = (row_x, row_y)
                reduced_tuple = hl.reduce(tuple_add_combine_fn, input_tuple, dim=1)
                result_x[i], result_y[i] = reduced_tuple

            return result_x, result_y

        # Create test input
        x = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            device=DEVICE,
        )
        y = torch.tensor(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            device=DEVICE,
        )

        # Test that the kernel compiles and runs
        code, (result_x, result_y) = code_and_output(test_reduce_tuple_kernel, (x, y))

        # Test the actual reduce operation
        expected_x = torch.tensor([6.0, 15.0], device=DEVICE)
        expected_y = torch.tensor([3.0, 6.0], device=DEVICE)
        torch.testing.assert_close(result_x, expected_x)
        torch.testing.assert_close(result_y, expected_y)

    def test_reduce_different_dtypes(self):
        """Test reduce with different tensor dtypes."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_int_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i] = hl.reduce(add_combine_fn, row_data, dim=1)
            return result

        # Create test input with integer dtype
        x = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            device=DEVICE,
            dtype=torch.int64,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_reduce_int_kernel, (x,))

        # Test the actual reduce operation
        expected = torch.tensor([10, 26], device=DEVICE, dtype=torch.int64)
        torch.testing.assert_close(result, expected)

    def test_reduce_tuple_unpacking_oneline(self):
        """Test tuple unpacking in one line: a, b = hl.reduce(...)"""

        @helion.kernel(autotune_effort="none")
        def test_tuple_oneline_kernel(
            values: torch.Tensor, indices: torch.Tensor
        ) -> torch.Tensor:
            batch_size = values.size(0)
            result = torch.empty([batch_size], dtype=torch.int64, device=values.device)

            for i in hl.tile(batch_size):
                row_values = values[i, :]  # Shape: [TILE_SIZE, seq_len]
                row_indices = indices[i, :]  # Shape: [TILE_SIZE, seq_len]

                # Create tuple of (values, indices) for reduction
                value_index_pairs = (row_values, row_indices)

                # Test one-line tuple unpacking - reduce 2D tensors on dim=1 to get 1D results
                max_value, max_index = hl.reduce(
                    argmax_combine_fn, value_index_pairs, dim=1
                )

                # max_index is now 1D tensor that can be assigned directly
                result[i] = max_index

            return result

        # Create test input with known argmax positions
        values = torch.tensor(
            [
                [1.0, 5.0, 3.0, 2.0],  # max=5.0 at index 1
                [9.0, 2.0, 7.0, 8.0],  # max=9.0 at index 0
                [4.0, 6.0, 8.0, 3.0],  # max=8.0 at index 2
            ],
            device=DEVICE,
        )
        indices = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
            device=DEVICE,
            dtype=torch.int64,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_tuple_oneline_kernel, (values, indices))

        # Test the actual argmax operation
        expected = torch.tensor([1, 0, 2], device=DEVICE, dtype=torch.int64)
        torch.testing.assert_close(result, expected)

        # Verify against PyTorch argmax
        pytorch_result = torch.argmax(values, dim=1)
        torch.testing.assert_close(result, pytorch_result)

        # Check that the generated code contains the expected elements
        if _get_backend() == "cute":
            self.assertIn("cute.arch.warp_reduction_max", code)
        else:
            self.assertIn("tl.reduce", code)
            self.assertIn("argmax_combine_fn_", code)

    def test_reduce_tuple_unpacking_twoline(self):
        """Test tuple unpacking in two lines: result = hl.reduce(...); a, b = result"""

        @helion.kernel(autotune_effort="none")
        def test_tuple_twoline_kernel(
            values: torch.Tensor, indices: torch.Tensor
        ) -> torch.Tensor:
            batch_size = values.size(0)
            result = torch.empty([batch_size], dtype=torch.int64, device=values.device)

            for i in hl.tile(batch_size):
                row_values = values[i, :]  # Shape: [TILE_SIZE, seq_len]
                row_indices = indices[i, :]  # Shape: [TILE_SIZE, seq_len]

                # Create tuple of (values, indices) for reduction
                value_index_pairs = (row_values, row_indices)

                # Test two-line tuple unpacking - reduce 2D tensors on dim=1 to get 1D results
                reduction_result = hl.reduce(
                    argmax_combine_fn, value_index_pairs, dim=1
                )
                max_value, max_index = reduction_result

                # max_index is now 1D tensor that can be assigned directly
                result[i] = max_index

            return result

        # Create test input with known argmax positions
        values = torch.tensor(
            [
                [1.0, 5.0, 3.0, 2.0],  # max=5.0 at index 1
                [9.0, 2.0, 7.0, 8.0],  # max=9.0 at index 0
                [4.0, 6.0, 8.0, 3.0],  # max=8.0 at index 2
            ],
            device=DEVICE,
        )
        indices = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
            device=DEVICE,
            dtype=torch.int64,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_tuple_twoline_kernel, (values, indices))

        # Test the actual argmax operation
        expected = torch.tensor([1, 0, 2], device=DEVICE, dtype=torch.int64)
        torch.testing.assert_close(result, expected)

        # Verify against PyTorch argmax
        pytorch_result = torch.argmax(values, dim=1)
        torch.testing.assert_close(result, pytorch_result)

        # Check that the generated code contains the expected elements
        if _get_backend() == "cute":
            self.assertIn("cute.arch.warp_reduction_max", code)
        else:
            self.assertIn("tl.reduce", code)
            self.assertIn("argmax_combine_fn_", code)

    def test_reduce_argmax_negative_values(self):
        """Test argmax with all negative values using other=(-inf, 0)."""

        @helion.kernel(autotune_effort="none")
        def test_argmax_negative_kernel(
            values: torch.Tensor, indices: torch.Tensor
        ) -> torch.Tensor:
            batch_size = values.size(0)
            result = torch.empty([batch_size], dtype=torch.int64, device=values.device)

            for i in hl.tile(batch_size):
                row_values = values[i, :]  # Shape: [TILE_SIZE, seq_len]
                row_indices = indices[i, :]  # Shape: [TILE_SIZE, seq_len]

                # Create tuple of (values, indices) for reduction
                value_index_pairs = (row_values, row_indices)

                # Test argmax with negative values - use -inf for values, 0 for indices
                max_value, max_index = hl.reduce(
                    argmax_combine_fn,
                    value_index_pairs,
                    dim=1,
                    other=(float("-inf"), 0),
                )

                # max_index is now 1D tensor that can be assigned directly
                result[i] = max_index

            return result

        # Create test input with all negative values
        values = torch.tensor(
            [
                [-5.0, -1.0, -3.0, -2.0],  # max=-1.0 at index 1
                [-9.0, -8.0, -7.0, -10.0],  # max=-7.0 at index 2
                [-4.0, -6.0, -2.0, -3.0],  # max=-2.0 at index 2
            ],
            device=DEVICE,
        )
        indices = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
            device=DEVICE,
            dtype=torch.int64,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_argmax_negative_kernel, (values, indices))

        # Test the actual argmax operation
        expected = torch.tensor([1, 2, 2], device=DEVICE, dtype=torch.int64)
        torch.testing.assert_close(result, expected)

        # Verify against PyTorch argmax
        pytorch_result = torch.argmax(values, dim=1)
        torch.testing.assert_close(result, pytorch_result)

        # Check that the generated code contains the expected elements
        if _get_backend() == "cute":
            self.assertIn("cute.arch.warp_reduction_max", code)
        else:
            self.assertIn("tl.reduce", code)
            self.assertIn("argmax_combine_fn_", code)

    def test_reduce_code_generation(self):
        """Test that reduce generates correct Triton code."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_codegen_kernel(x: torch.Tensor) -> torch.Tensor:
            result = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            for i in hl.tile(x.size(0)):
                row_data = x[i, :]
                result[i] = hl.reduce(add_combine_fn, row_data, dim=1)
            return result

        # Create test input
        x = torch.tensor([[1.0, 2.0, 3.0]], device=DEVICE)

        # Test that the kernel compiles and generates expected code
        code, result = code_and_output(test_reduce_codegen_kernel, (x,))

        if _get_backend() == "cute":
            self.assertIn("cute.kernel", code)
            self.assertIn("cute.arch.warp_reduction_sum", code)
        else:
            # Check that the generated code contains the expected elements
            self.assertIn("tl.reduce", code)
            self.assertIn("add_combine_fn_", code)
            self.assertIn("@triton.jit", code)

        # Verify correctness
        expected = torch.tensor([6.0], device=DEVICE)
        torch.testing.assert_close(result, expected)

    def test_reduce_tuple_unpacked_format(self):
        """Test reduce with tuple input using unpacked format combine function."""

        @helion.kernel(autotune_effort="none")
        def test_reduce_tuple_unpacked_kernel(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_x = torch.empty([x.size(0)], dtype=x.dtype, device=x.device)
            result_y = torch.empty([y.size(0)], dtype=y.dtype, device=y.device)

            for i in hl.tile(x.size(0)):
                row_x = x[i, :]
                row_y = y[i, :]
                input_tuple = (row_x, row_y)
                reduced_tuple = hl.reduce(
                    tuple_add_combine_unpacked_fn, input_tuple, dim=1
                )
                result_x[i], result_y[i] = reduced_tuple

            return result_x, result_y

        # Create test input
        x = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            device=DEVICE,
        )
        y = torch.tensor(
            [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            device=DEVICE,
        )

        # Test that the kernel compiles and runs
        code, (result_x, result_y) = code_and_output(
            test_reduce_tuple_unpacked_kernel, (x, y)
        )

        # Test the actual reduce operation
        expected_x = torch.tensor([6.0, 15.0], device=DEVICE)
        expected_y = torch.tensor([3.0, 6.0], device=DEVICE)
        torch.testing.assert_close(result_x, expected_x)
        torch.testing.assert_close(result_y, expected_y)

    def test_reduce_argmax_unpacked_format(self):
        """Test argmax with unpacked format combine function."""

        @helion.kernel(autotune_effort="none")
        def test_argmax_unpacked_kernel(
            values: torch.Tensor, indices: torch.Tensor
        ) -> torch.Tensor:
            batch_size = values.size(0)
            result = torch.empty([batch_size], dtype=torch.int64, device=values.device)

            for i in hl.tile(batch_size):
                row_values = values[i, :]
                row_indices = indices[i, :]

                # Create tuple of (values, indices) for reduction
                value_index_pairs = (row_values, row_indices)

                # Test unpacked format combine function
                max_value, max_index = hl.reduce(
                    argmax_combine_unpacked_fn, value_index_pairs, dim=1
                )

                result[i] = max_index

            return result

        # Create test input with known argmax positions
        values = torch.tensor(
            [
                [1.0, 5.0, 3.0, 2.0],  # max=5.0 at index 1
                [9.0, 2.0, 7.0, 8.0],  # max=9.0 at index 0
                [4.0, 6.0, 8.0, 3.0],  # max=8.0 at index 2
            ],
            device=DEVICE,
        )
        indices = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ],
            device=DEVICE,
            dtype=torch.int64,
        )

        # Test that the kernel compiles and runs
        code, result = code_and_output(test_argmax_unpacked_kernel, (values, indices))

        # Test the actual argmax operation
        expected = torch.tensor([1, 0, 2], device=DEVICE, dtype=torch.int64)
        torch.testing.assert_close(result, expected)

        # Verify against PyTorch argmax
        pytorch_result = torch.argmax(values, dim=1)
        torch.testing.assert_close(result, pytorch_result)


@onlyBackends(["cute"])
class TestCuteBuiltinReduce(RefEagerTestBase, TestCase):
    def test_nonidentity_padding_is_not_canonicalized(self) -> None:
        combine = make_fx(operator.add)(torch.ones(1), torch.ones(1)).graph
        for other in (0.0, 1.0):
            with self.subTest(other=other):
                ir = DeviceIR()
                combine_id = ir.add_graph(
                    combine,
                    HelperFunctionGraphInfo,
                    node_args=[],
                    original_function_name="add",
                )
                graph = torch.fx.Graph()
                x = graph.placeholder("x")
                masked = graph.call_function(_mask_to, (x, other))
                masked.meta["val"] = torch.empty(1, 65)
                reduction = graph.call_function(
                    _reduce, (combine_id, masked, -1, False, False)
                )
                reduction.meta["val"] = torch.empty(1)
                graph.output(reduction)
                ir.add_root_graph(graph)
                canonicalize_reductions(ir)
                output = graph.find_nodes(op="output")[0]
                self.assertIs(
                    output.args[0].target,
                    torch.ops.aten.sum.dim_IntList if other == 0 else _reduce,
                )

    def test_builtin_combine_requires_exact_operands(self) -> None:
        x, y = torch.ones(1), torch.ones(1)
        for combine, recognized in (
            (operator.add, True),
            (lambda a, b: b * a, True),
            (lambda a, b: a + b + 1, False),
            (lambda a, b: a + a, False),
            (lambda a, b: torch.add(a, b, alpha=2), False),
        ):
            with self.subTest(combine=combine):
                graph = make_fx(combine)(x, y).graph
                self.assertEqual(
                    _independent_combines(graph, 1) is not None, recognized
                )

    def test_builtin_combines_span_multiple_lane_groups(self) -> None:
        @helion.kernel(autotune_effort="none")
        def combined(
            x: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            sums = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
            products = torch.empty_like(sums)
            maxima = torch.empty_like(sums)
            minima = torch.empty_like(sums)
            for row in hl.tile(x.size(0)):
                value = x[row, :]
                sums[row] = hl.reduce(add_combine_fn, value, dim=-1)
                products[row] = hl.reduce(mul_combine_fn, value, dim=-1, other=1.0)
                maxima[row] = hl.reduce(
                    max_combine_fn, value, dim=-1, other=-float("inf")
                )
                minima[row] = hl.reduce(
                    min_combine_fn, value, dim=-1, other=float("inf")
                )
            return sums, products, maxima, minima

        for columns in (65, 513):
            with self.subTest(columns=columns):
                x = torch.randn((17, columns), device=DEVICE) * 0.01 + 1.0
                _, output = code_and_output(combined, (x,))
                expected = (x.sum(-1), x.prod(-1), x.amax(-1), x.amin(-1))
                for actual, reference in zip(output, expected, strict=True):
                    torch.testing.assert_close(actual, reference)

    def test_independent_tuple_combines_preserve_dtypes(self) -> None:
        @helion.kernel(autotune_effort="none")
        def combined(
            x: torch.Tensor, indices: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            values = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
            counts = torch.empty((x.size(0),), dtype=indices.dtype, device=x.device)
            for row in hl.tile(x.size(0)):
                total, count = hl.reduce(
                    tuple_add_combine_fn, (x[row, :], indices[row, :]), dim=-1
                )
                values[row] = total
                counts[row] = count
            return values, counts

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.full((17, 65), 0.5, dtype=dtype, device=DEVICE)
                indices = torch.full(
                    (17, 65), 100000000, dtype=torch.int32, device=DEVICE
                )
                _, (values, counts) = code_and_output(combined, (x, indices))
                torch.testing.assert_close(values, x.sum(-1))
                torch.testing.assert_close(counts, indices.sum(-1, dtype=torch.int32))


# Backend test marks also admit TileIR; these kernels explicitly select Triton.
@onlyBackends(["triton"])
class TestRolledCustomReduce(RefEagerTestBase, TestCase):
    @skipIfRefEager("checks empty compiled custom reductions")
    def test_empty_custom_reduction_matches_persistent_padding(self) -> None:
        @helion.kernel(backend="triton", static_shapes=False, autotune_effort="none")
        def reduce(x: torch.Tensor, other: hl.constexpr) -> torch.Tensor:
            out = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
            for row in hl.tile(x.size(0)):
                out[row] = hl.reduce(max_combine_fn, x[row, :], dim=1, other=other)
            return out

        x = torch.empty((5, 0), device=DEVICE)
        for other in (float("-inf"), -7.0):
            _, result = code_and_output(
                reduce, (x, other), block_sizes=[2], reduction_loops=[256]
            )
            torch.testing.assert_close(result, torch.full_like(result, other))

            rolled = helion.kernel(
                reduce.fn, backend="triton", static_shapes=False, autotune_effort="none"
            )
            bound = rolled.bind((torch.empty((5, 513), device=DEVICE), other))
            run = bound.compile_config(
                helion.Config(block_sizes=[2], reduction_loops=[256])
            )
            self.assertIn(
                "custom_reduce_padding",
                bound.to_code(helion.Config(block_sizes=[2], reduction_loops=[256])),
            )
            torch.testing.assert_close(run(x, other), torch.full_like(result, other))

    @skipIfRefEager("checks masks on rolled custom reduction inputs")
    def test_custom_reduction_keeps_mask_after_native_reduction(self) -> None:
        @helion.kernel(backend="triton", autotune_effort="none")
        def reduce(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty((x.size(0),), device=x.device, dtype=x.dtype)
            for row in hl.tile(x.size(0)):
                values = x[row, :]
                inverse_sum = 1.0 / values.sum(1, keepdim=True)
                out[row] = hl.reduce(
                    max_combine_fn, values * inverse_sum, dim=1, other=0.0
                )
            return out

        # Valid lanes become +/-inf; padded zero lanes become NaN without
        # the mask immediately before the custom maximum.
        x = torch.tensor([1.0, -1.0], device=DEVICE).repeat(3, 257)
        _, result = code_and_output(
            reduce, (x,), block_sizes=[2], reduction_loops=[256]
        )
        torch.testing.assert_close(result, torch.full_like(result, float("inf")))

    @skipIfRefEager("checks custom reduction tuple-axis admission")
    def test_custom_tuple_with_fixed_payload_stays_persistent(self) -> None:
        @helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
        def reduce(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            sums = torch.empty(x.size(0), device=x.device, dtype=x.dtype)
            counts = torch.empty_like(sums)
            for row in hl.grid(x.size(0)):
                values = x[row, :]
                ones = hl.full([256], 1.0, dtype=x.dtype)
                total, count = hl.reduce(tuple_add_combine_fn, (values, ones), dim=0)
                sums[row], counts[row] = total, count
            return sums, counts

        x = torch.randn(3, 256, device=DEVICE)
        bound = reduce.bind((x,))
        self.assertFalse(bound.config_spec.has_custom_reduction_loops)
        _, (sums, counts) = code_and_output(reduce, (x,))
        torch.testing.assert_close(sums, x.sum(1))
        torch.testing.assert_close(counts, torch.full_like(counts, 256))

    @skipIfRefEager("checks unrelated pointwise grid defaults")
    def test_custom_baseline_preserves_unrelated_pointwise_tiles(self) -> None:
        @helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
        def mixed(x: torch.Tensor, y: torch.Tensor):
            pointwise = torch.empty_like(x)
            custom = torch.empty(y.size(0), device=y.device, dtype=y.dtype)
            for row, col in hl.tile(x.shape):
                pointwise[row, col] = x[row, col] + 1
            for row in hl.tile(y.size(0)):
                custom[row] = hl.reduce(add_combine_fn, y[row, :], dim=1)
            return pointwise, custom

        bound = mixed.bind(
            (
                torch.empty(128, 256, device=DEVICE),
                torch.empty(5, 8192, device=DEVICE),
            )
        )
        assert bound.host_function is not None
        spec = bound.config_spec
        default = spec._base_default_config()
        pointwise_ids = bound.host_function.device_ir.grid_block_ids[0]
        for seed in (
            TritonReductionHeuristic.get_seed_configs(
                bound.env, bound.host_function.device_ir
            )[0],
            TritonNarrowReductionHeuristic.get_seed_config(
                bound.env, bound.host_function.device_ir
            ),
        ):
            assert seed is not None
            for block_id in pointwise_ids:
                self.assertEqual(
                    spec.block_sizes.config_get(seed.block_sizes, block_id),
                    spec.block_sizes.config_get(default.block_sizes, block_id),
                )

    @skipIfRefEager("checks mixed reduction heuristic seeds")
    def test_custom_baseline_preserves_unrelated_native_schedule(self) -> None:
        @helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
        def mixed(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            native = torch.empty((x.size(0),), device=x.device, dtype=x.dtype)
            custom = torch.empty_like(native)
            for row in hl.tile(x.size(0)):
                native[row] = x[row, :].sum(1)
            for other_row in hl.tile(y.size(0)):
                custom[other_row] = hl.reduce(add_combine_fn, y[other_row, :], dim=1)
            return native, custom

        bound = mixed.bind(
            (torch.empty(5, 65536, device=DEVICE), torch.empty(5, 8192, device=DEVICE))
        )
        assert bound.host_function is not None
        env, ir = bound.env, bound.host_function.device_ir
        self.assertTrue(bound.config_spec.has_custom_reduction_loops)
        seeds = TritonReductionHeuristic.get_seed_configs(env, ir)
        with patch.object(
            type(bound.config_spec),
            "has_custom_reduction_loops",
            new_callable=PropertyMock,
            return_value=False,
        ):
            native_seed = TritonReductionHeuristic.get_seed_configs(env, ir)[0]
        self.assertEqual(seeds[0], native_seed)
        native_ids = {
            r.block_id
            for r in bound.config_spec.reduction_kernel_fact.reductions
            if not r.is_custom
        }
        for seed in seeds:
            for block_id in native_ids:
                self.assertEqual(
                    bound.config_spec.reduction_loops.config_get(
                        seed.reduction_loops, block_id
                    ),
                    bound.config_spec.reduction_loops.config_get(
                        native_seed.reduction_loops, block_id
                    ),
                )

        narrow = TritonNarrowReductionHeuristic.get_seed_config(env, ir)
        default = bound.config_spec._base_default_config()
        for block_id in native_ids:
            self.assertEqual(
                bound.config_spec.reduction_loops.config_get(
                    narrow.reduction_loops, block_id
                ),
                bound.config_spec.reduction_loops.config_get(
                    default.reduction_loops, block_id
                ),
            )
        self.assertEqual(narrow.block_sizes, default.block_sizes)
        self.assertEqual(narrow.num_warps, default.num_warps)

    @skipIfRefEager("checks generated reduction-loop code")
    def test_nonzero_identity_and_nonidentity_padding(self) -> None:
        @helion.kernel(
            backend="triton", autotune_effort="none", debug_dtype_asserts=True
        )
        def shifted_rows(
            x: torch.Tensor,
            other: hl.constexpr,
            combine: hl.constexpr = lambda left, right: left + right + 1,
        ) -> torch.Tensor:
            out = torch.empty((x.size(0),), device=x.device, dtype=x.dtype)
            for row in hl.tile(x.size(0)):
                out[row] = hl.reduce(combine, x[row, :], dim=1, other=other)
            return out

        x = torch.randint(-10, 10, (5, 2051), device=DEVICE).float()
        for other in (-1, 2):
            code, result = code_and_output(
                shifted_rows, (x, other), block_sizes=[2], reduction_loops=[256]
            )
            padding = 4096 - x.size(1)
            expected = x.sum(1) + x.size(1) - 1 + padding * (other + 1)
            torch.testing.assert_close(result, expected)
            self.assertIn("for roffset_", code)
            self.assertNotIn("custom_reduce_valid_combine", code)

    @skipIfRefEager("Checks dynamic reduction results")
    def test_dynamic_tuple_padding(self) -> None:
        @helion.kernel(
            backend="triton",
            static_shapes=False,
            config=helion.Config(block_sizes=[2], reduction_loops=[256]),
        )
        def padded_pair(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            first = torch.empty((x.size(0),), device=x.device, dtype=x.dtype)
            second = torch.empty_like(first)
            for row in hl.tile(x.size(0)):
                a, b = hl.reduce(
                    tuple_add_combine_fn,
                    (x[row, :], x[row, :]),
                    dim=1,
                    other=(2, 3),
                )
                first[row] = a
                second[row] = b
            return first, second

        storage = torch.randint(-10, 10, (5, 8192), device=DEVICE).float()
        code, _ = code_and_output(
            padded_pair,
            (storage[:, :5000],),
            block_sizes=[2],
            reduction_loops=[256],
        )
        self.assertIn("custom_reduce_valid_combine", code)
        for width in (5000, 2, 33):
            x = storage[:, :width]
            padding = (1 << (width - 1).bit_length()) - width
            torch.testing.assert_close(
                padded_pair(x), (x.sum(1) + padding * 2, x.sum(1) + padding * 3)
            )

    @skipIfRefEager("checks generated reduction-loop code")
    def test_mixed_reduction_fusion(self) -> None:
        @helion.kernel(backend="triton", autotune_effort="none")
        def two_reductions(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            totals = torch.empty((x.size(0),), device=x.device, dtype=x.dtype)
            squares = torch.empty_like(totals)
            for row in hl.tile(x.size(0)):
                values = x[row, :]
                total = hl.reduce(add_combine_fn, values, dim=1, keep_dims=True)
                totals[row] = total.squeeze(1)
                squares[row] = (values * values).sum(1)
            return totals, squares

        x = torch.randn((5, 2051), device=DEVICE)
        bound = two_reductions.bind((x,))
        self.assertTrue(bound.env.config_spec.reduction_fusion_available)
        code, (totals, squares) = code_and_output(
            two_reductions,
            (x,),
            block_sizes=[2],
            reduction_loops=[256],
            reduction_fusion=True,
        )
        torch.testing.assert_close(totals, x.sum(1), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(squares, x.square().sum(1))
        self.assertEqual(code.count("for roffset_"), 1)


if __name__ == "__main__":
    unittest.main()
