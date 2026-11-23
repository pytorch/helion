from __future__ import annotations

import contextlib
import io
import math
import unittest
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import assert_ref_eager_mode
from helion._testing import onlyBackends
import helion.language as hl
from helion.runtime.ref_mode import RefModeContext


def _broadcast_index_forms(x: torch.Tensor, form: hl.constexpr) -> torch.Tensor:
    """One form of ``hl.load`` / ``hl.store`` whose index is a tile beside two
    index tensors that broadcast against each other, a row vector ``[H, 1]``
    and a column vector ``[1, W]``.  The masked forms address each element's
    upper right neighbour, masked where that is outside ``x``; the unmasked
    ones its mirror image.  ``cartesian`` is the masked load with
    one-dimensional index tensors, which take a dim each."""
    planes, height, width = x.shape
    out = torch.zeros_like(x)
    for tile_p, tile_h, tile_w in hl.tile([planes, height, width]):
        row = tile_h.index - 1
        col = tile_w.index + 1
        inside = (row >= 0)[None, :, None] & (col < width)[None, None, :]
        flip_row = height - 1 - tile_h.index
        flip_col = width - 1 - tile_w.index
        if form == "masked_load":
            out[tile_p, tile_h, tile_w] = hl.load(
                x, [tile_p, row[:, None], col[None, :]], extra_mask=inside
            )
        elif form == "load":
            out[tile_p, tile_h, tile_w] = hl.load(
                x, [tile_p, flip_row[:, None], flip_col[None, :]]
            )
        elif form == "masked_store":
            hl.store(
                out,
                [tile_p, row[:, None], col[None, :]],
                x[tile_p, tile_h, tile_w],
                extra_mask=inside,
            )
        elif form == "store":
            hl.store(
                out,
                [tile_p, flip_row[:, None], flip_col[None, :]],
                x[tile_p, tile_h, tile_w],
            )
        else:
            out[tile_p, tile_h, tile_w] = hl.load(
                x, [tile_p, row, col], extra_mask=inside
            )
    return out


def _broadcast_index_forms_expected(x: torch.Tensor, form: str) -> torch.Tensor:
    if form in ("load", "store"):
        return x.flip(1, 2)
    expected = torch.zeros_like(x)
    if form == "masked_store":
        expected[:, :-1, 1:] = x[:, 1:, :-1]
    else:
        expected[:, 1:, :-1] = x[:, :-1, 1:]
    return expected


def _sliced_index_forms(x: torch.Tensor, form: hl.constexpr) -> torch.Tensor:
    """A slice between two tiles in an unmasked ``hl.load`` or ``hl.store``,
    or between two one-dimensional index tensors in a subscript (each copies
    ``x``); or a masked ``hl.store`` whose value, one row of ``x``,
    broadcasts to the rows it indexes."""
    rows, _, cols = x.shape
    out = torch.zeros_like(x)
    for tile_r, tile_c in hl.tile([rows, cols]):
        if form == "slice_load":
            out[tile_r, :, tile_c] = hl.load(x, [tile_r, slice(None), tile_c])
        elif form == "slice_store":
            hl.store(out, [tile_r, slice(None), tile_c], x[tile_r, :, tile_c])
        elif form == "subscript_load":
            out[tile_r, :, tile_c] = x[tile_r.index, :, tile_c.index]
        elif form == "subscript_store":
            out[tile_r.index, :, tile_c.index] = x[tile_r, :, tile_c]
        else:
            hl.store(
                out,
                [tile_r, 0, tile_c],
                x[0, 0, tile_c][None, :],
                extra_mask=(tile_c.index >= 1)[None, :],
            )
    return out


def _sliced_index_forms_expected(x: torch.Tensor, form: str) -> torch.Tensor:
    if form != "broadcast_value_store":
        return x
    expected = torch.zeros_like(x)
    expected[:, 0, 1:] = x[0, 0, 1:]
    return expected


def _tile_between_forms(x: torch.Tensor, form: hl.constexpr) -> torch.Tensor:
    """A subscript with a tile between two one-dimensional index tensors, as
    a load or as a store; each copies ``x``.  The blocks of two leave last
    tiles of one element."""
    shape = [x.size(0), x.size(1), x.size(2)]
    out = torch.zeros_like(x)
    for tile_p, tile_h, tile_w in hl.tile(shape, block_size=[2, 2, 2]):
        if form == "tile_load":
            out[tile_p, tile_h, tile_w] = x[tile_p.index, tile_h, tile_w.index]
        else:
            out[tile_p.index, tile_h, tile_w.index] = x[tile_p, tile_h, tile_w]
    return out


def _copy_expected(x: torch.Tensor, form: str) -> torch.Tensor:
    return x


def _int64_index_forms(x: torch.Tensor, form: hl.constexpr) -> torch.Tensor:
    """A tile, whose index tensor is int32, beside an int64 index tensor (the
    columns in reverse): in ``hl.load`` and ``hl.store``, unmasked or masked
    to all columns of the tile but its first, and in a subscript."""
    rows, cols = x.shape
    reverse = torch.arange(cols - 1, -1, -1, device=x.device)
    out = torch.zeros_like(x)
    for tile_r, tile_c in hl.tile([rows, cols]):
        col = reverse[tile_c]
        keep = (tile_c.index >= 1)[None, :]
        if form == "load":
            out[tile_r, tile_c] = hl.load(x, [tile_r, col])
        elif form == "masked_load":
            out[tile_r, tile_c] = hl.load(x, [tile_r, col], extra_mask=keep)
        elif form == "store":
            hl.store(out, [tile_r, col], x[tile_r, tile_c])
        elif form == "masked_store":
            hl.store(out, [tile_r, col], x[tile_r, tile_c], extra_mask=keep)
        elif form == "subscript_load":
            out[tile_r, tile_c] = x[tile_r.index, col]
        else:
            out[tile_r.index, col] = x[tile_r, tile_c]
    return out


def _int64_index_forms_expected(x: torch.Tensor, form: str) -> torch.Tensor:
    expected = x.flip(1)
    if form == "masked_load":
        expected[:, 0] = 0
    elif form == "masked_store":
        expected[:, -1] = 0
    return expected


def _host_index_forms(x: torch.Tensor, form: hl.constexpr) -> torch.Tensor:
    """A subscript in host code, outside the device loop, which only copies
    the result: one index tensor (the rows in reverse); two, which PyTorch
    pairs (with every other column), before the loop or after it; two around
    a slice; negative indices, which wrap; rows past the end, which raise;
    as a load or as a store."""
    rows, cols = x.shape
    back = torch.arange(rows - 1, -1, -1, device=x.device)
    even = torch.arange(0, 2 * rows, 2, device=x.device)
    wrapped_back = torch.arange(-1, -rows - 1, -1, device=x.device)
    wrapped_even = torch.arange(-cols, 2 * rows - cols, 2, device=x.device)
    past_the_end = torch.arange(rows, 0, -1, device=x.device)
    if form == "one":
        host = x[back, :]
    elif form == "two":
        host = x[back, even]
    elif form == "two_around_slice":
        host = x.reshape(rows, 1, cols)[back, :, even]
    elif form == "negative":
        host = x[wrapped_back, wrapped_even]
    elif form == "past_the_end":
        host = x[past_the_end, :]
    elif form == "store_two":
        host = torch.zeros_like(x)
        host[back, even] = 1.0
    elif form == "store_negative":
        host = torch.zeros_like(x)
        host[wrapped_back, wrapped_even] = 1.0
    elif form == "store_past_the_end":
        host = torch.zeros_like(x)
        host[past_the_end, :] = 1.0
    else:
        host = x
    flat = host.reshape(-1)
    out = torch.empty_like(flat)
    for tile in hl.tile(flat.size(0)):
        out[tile] = flat[tile]
    if form == "two_after_loop":
        return out.reshape(rows, cols)[back, even]
    return out


def _out_of_range_store_forms(
    out: torch.Tensor, x: torch.Tensor, idx: torch.Tensor, form: hl.constexpr
) -> torch.Tensor:
    """A store in a device loop through an index tensor with elements out of
    range, beside a tile: of a tensor or a scalar, as a subscript (with the
    tile or with its index tensor) or as ``hl.store``."""
    for tile_r, tile_c in hl.tile([out.size(0), idx.size(0)]):
        col = idx[tile_c]
        if form == "subscript":
            out[tile_r, col] = x[tile_r, tile_c]
        elif form == "subscript_scalar":
            out[tile_r, col] = 1.5
        elif form == "cartesian":
            out[tile_r.index, col] = x[tile_r, tile_c]
        elif form == "store":
            hl.store(out, [tile_r, col], x[tile_r, tile_c])
        else:
            hl.store(out, [tile_r, col], 1.5)
    return out


def _host_index_forms_expected(x: torch.Tensor, form: str) -> torch.Tensor:
    pairs = [(x.size(0) - 1 - i, 2 * i) for i in range(x.size(0))]
    if form == "one":
        return x.flip(0).reshape(-1)
    if form.startswith("store"):
        expected = torch.zeros_like(x)
        for row, col in pairs:
            expected[row, col] = 1.0
        return expected.reshape(-1)
    return torch.stack([x[row, col] for row, col in pairs])


# A kernel function, its expected result, its forms and two shapes.  For the
# first three kernels a layout that does not follow compiled code can still
# broadcast at the first shape, to other elements (as many planes as rows; a
# slice as long as a tile); at the second it cannot.
_INDEX_FORM_CASES = (
    (
        _broadcast_index_forms,
        _broadcast_index_forms_expected,
        ("masked_load", "load", "masked_store", "store", "cartesian"),
        ((3, 3, 5), (2, 3, 5)),
    ),
    (
        _sliced_index_forms,
        _sliced_index_forms_expected,
        (
            "slice_load",
            "slice_store",
            "subscript_load",
            "subscript_store",
            "broadcast_value_store",
        ),
        ((2, 4, 4), (2, 3, 4)),
    ),
    (
        _tile_between_forms,
        _copy_expected,
        ("tile_load", "tile_store"),
        ((3, 3, 5), (2, 3, 5)),
    ),
    (
        _int64_index_forms,
        _int64_index_forms_expected,
        (
            "load",
            "masked_load",
            "store",
            "masked_store",
            "subscript_load",
            "subscript_store",
        ),
        ((3, 4), (2, 5)),
    ),
    (
        _host_index_forms,
        _host_index_forms_expected,
        (
            "one",
            "two",
            "two_around_slice",
            "negative",
            "store_two",
            "store_negative",
            "two_after_loop",
        ),
        ((3, 5), (2, 4)),
    ),
)


@onlyBackends(["triton", "cute"])
class TestRefEagerMisc(TestCase):
    @pytest.mark.cpu_only
    def test_load_store_index_forms(self):
        """``hl.load``, ``hl.store`` and subscripts lay their result out as
        compiled code does: index tensors that are not all one-dimensional
        broadcast against each other, a tile beside them keeps its own dim, a
        slice or tile between two one-dimensional index tensors stays between
        them, and a masked store's value broadcasts.  One-dimensional index
        tensors stay a Cartesian product in a device loop, whatever their
        integer dtypes; in host code a subscript is plain PyTorch indexing."""
        for fn, expected, forms, shapes in _INDEX_FORM_CASES:
            kernel = helion.kernel(
                fn, static_shapes=True, ref_mode=helion.RefMode.EAGER
            )
            for shape in shapes:
                x = torch.randn(shape)
                for form in forms:
                    with self.subTest(shape=shape, form=form):
                        torch.testing.assert_close(
                            kernel(x, form), expected(x, form), rtol=0, atol=0
                        )

    @pytest.mark.cpu_only
    def test_host_subscript_past_the_end_raises(self):
        """An index past the end raises in host code as in PyTorch: a load
        does not clamp it and a store does not skip it, as a device loop's
        do."""
        kernel = helion.kernel(
            _host_index_forms, static_shapes=True, ref_mode=helion.RefMode.EAGER
        )
        for form in ("past_the_end", "store_past_the_end"):
            with self.subTest(form=form), self.assertRaises(IndexError):
                kernel(torch.randn(3, 5), form)

    @pytest.mark.cpu_only
    def test_atomic_index_tensors_stay_cartesian(self):
        """The reference implementations index through the same mode: in a
        device loop their index tensors still form a Cartesian product."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile_r, tile_c in hl.tile([x.size(0), x.size(1)]):
                hl.atomic_xchg(out, [tile_r.index, tile_c.index], x[tile_r, tile_c])
            return out

        x = torch.randn(3, 5)
        torch.testing.assert_close(fn(x), x, rtol=0, atol=0)

    @pytest.mark.cpu_only
    def test_subscript_store_trims_overrunning_tile(self):
        """A tile that runs past the stored tensor still stores the elements
        inside it when it sits between two index tensors, as it does in a
        store to tiles alone."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def fn(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            for tile_p, tile_h, tile_w in hl.tile([x.size(0), x.size(1), x.size(2)]):
                out[tile_p.index, tile_h, tile_w.index] = x[tile_p, tile_h, tile_w]
            return out

        x = torch.randn(2, 3, 4)
        torch.testing.assert_close(
            fn(x, torch.zeros(2, 2, 4)), x[:, :2, :], rtol=0, atol=0
        )

    @pytest.mark.cpu_only
    def test_store_leaves_out_index_elements_out_of_range(self):
        """A store in a device loop does not store the elements whose index
        is out of range, and they do not disturb a store to the edge element
        they were clamped to.  Compiled code has no mask for them: it stores
        them outside the tensor."""
        kernel = helion.kernel(
            _out_of_range_store_forms,
            static_shapes=True,
            ref_mode=helion.RefMode.EAGER,
        )
        x = torch.randn(2, 5)
        # The last column, one past it (on the last when clamped), one before
        # the first (on the first, which nothing else stores) and two others.
        idx = torch.tensor([3, 4, -1, 2, 1])
        for form in (
            "subscript",
            "subscript_scalar",
            "cartesian",
            "store",
            "store_scalar",
        ):
            with self.subTest(form=form):
                expected = torch.full((2, 4), -7.0)
                for lane, col in enumerate(idx.tolist()):
                    if 0 <= col < 4:
                        expected[:, col] = 1.5 if "scalar" in form else x[:, lane]
                out = kernel(torch.full((2, 4), -7.0), x, idx, form)
                torch.testing.assert_close(out, expected, rtol=0, atol=0)

    @pytest.mark.cpu_only
    def test_subscript_store_refuses_value_past_index_tensor(self):
        """Only a tile's dim is trimmed: a value that is too large along an
        index tensor's dim is still refused."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def fn(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            for tile_p, tile_h, tile_w in hl.tile([x.size(0), x.size(1), x.size(2)]):
                value = x[tile_p, tile_h, tile_w]
                out[tile_p.index, tile_h, tile_w.index] = torch.cat([value, value])
            return out

        x = torch.randn(2, 3, 4)
        with self.assertRaises(RuntimeError):
            fn(x, torch.zeros(2, 3, 4))

    @pytest.mark.cpu_only
    def test_subscript_empty_slice_between_index_tensors(self):
        """An empty slice between two index tensors selects nothing: the load
        is empty along its dim and the store leaves the tensor alone."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def fn(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out = torch.zeros_like(x)
            picked = out
            for tile_r, tile_c in hl.tile([x.size(0), x.size(2)]):
                picked = x[tile_r.index, 2:1, tile_c.index]
                out[tile_r.index, 2:1, tile_c.index] = 1.0
            return picked, out

        picked, out = fn(torch.randn(2, 3, 4))
        self.assertEqual(picked.shape, (2, 0, 4))
        self.assertFalse(out.any())

    @pytest.mark.cpu_only
    def test_subscript_negative_slice_step_between_index_tensors(self):
        """A slice with a negative step between two index tensors is refused
        as in compiled code, in a load and in a store."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def fn(x: torch.Tensor, store: hl.constexpr) -> torch.Tensor:
            out = torch.zeros_like(x)
            for tile_r, tile_c in hl.tile([x.size(0), x.size(2)]):
                if store:
                    out[tile_r.index, ::-1, tile_c.index] = x[tile_r, :, tile_c]
                else:
                    out[tile_r, :, tile_c] = x[tile_r.index, ::-1, tile_c.index]
            return out

        for store in (False, True):
            with (
                self.subTest(store=store),
                self.assertRaises(helion.exc.InvalidSliceStep),
            ):
                fn(torch.randn(2, 3, 4), store)

    def test_load_store_index_forms_match_compiled(self):
        """The compiled kernel and reference mode agree on the forms of
        ``test_load_store_index_forms``."""
        for fn, expected, forms, shapes in _INDEX_FORM_CASES:
            kernels = (
                helion.kernel(
                    fn,
                    static_shapes=True,
                    autotune_effort="none",
                    # The host-code stores are what their forms are about.
                    ignore_warnings=[helion.exc.TensorOperationInWrapper],
                ),
                helion.kernel(fn, static_shapes=True, ref_mode=helion.RefMode.EAGER),
            )
            x = torch.randn(shapes[0], device=DEVICE)
            for form in forms:
                with self.subTest(form=form):
                    for kernel in kernels:
                        torch.testing.assert_close(
                            kernel(x, form), expected(x, form), rtol=0, atol=0
                        )

    def test_masked_slice_with_scalar_index(self):
        @helion.kernel(static_shapes=True, ref_mode=helion.RefMode.EAGER)
        def fn(x: torch.Tensor) -> torch.Tensor:
            out = torch.full_like(x, -7)
            for tile in hl.tile(x.size(0)):
                mask = tile.index < 3
                values = hl.load(x, [tile, 1, slice(None)], extra_mask=mask[:, None])
                hl.store(
                    out,
                    [tile, 0, slice(None)],
                    values + 1,
                    extra_mask=mask[:, None],
                )
            return out

        for shape in ((5, 2, 7), (7, 2, 5)):
            with self.subTest(shape=shape):
                x = torch.randn(shape, device=DEVICE)
                expected = torch.full_like(x, -7)
                expected[:3, 0, :] = x[:3, 1, :] + 1
                result = fn.bind((x,)).run_ref(x)
                torch.testing.assert_close(result, expected)

    def test_context_entry_failure_releases_context(self):
        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def fn(x: torch.Tensor) -> torch.Tensor:
            return x

        x = torch.randn(2, device=DEVICE)
        bound = fn.bind((x,))
        default_device = torch.get_default_device()
        with (
            patch.object(
                RefModeContext, "_get_rng_state", side_effect=RuntimeError("RNG probe")
            ),
            self.assertRaisesRegex(RuntimeError, "RNG probe"),
        ):
            bound.run_ref(x)
        self.assertFalse(RefModeContext.has_current())
        self.assertFalse(CompileEnvironment.has_current())
        self.assertEqual(torch.get_default_device(), default_device)
        torch.testing.assert_close(bound.run_ref(x), x)

    def test_print_intermediate_tensor(self):
        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def print_intermediate_tensor_kernel(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            out = torch.empty_like(x)
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                x_val = x[tile_m, tile_n]
                y_val = y[tile_m, tile_n]
                sum_val = x_val + y_val
                print("x: ", x_val)
                print("y: ", y_val)
                print("sum: ", sum_val)
                out[tile_m, tile_n] = sum_val
            return out

        x = torch.ones([2, 2], device=DEVICE, dtype=torch.float32) * 10.0
        y = torch.ones([2, 2], device=DEVICE, dtype=torch.float32) * 5.0
        expected = x + y

        # Capture stdout to check print output
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            result = print_intermediate_tensor_kernel(x, y)

        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

        # Check that the print statements produced output
        output = captured_output.getvalue()
        self.assertIn("x: ", output)
        self.assertIn("y: ", output)
        self.assertIn("sum: ", output)
        self.assertIn("[[10., 10.]", output)  # x values
        self.assertIn("[[5., 5.]", output)  # y values
        self.assertIn("[[15., 15.]", output)  # sum values

    def test_print_in_invalid_helion_kernel(self):
        """Test that print works even in invalid Helion kernels in ref eager mode."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def incorrect_kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            for tile_m, tile_n in hl.tile([m, n]):
                val = x[tile_m, tile_n]
                print("processing tile: ", val)
                # `pass` below causes this kernel to be invalid.
                # But we show that in ref-eager mode, the `print` statement above still works,
                # which is useful for debugging.
                pass  # noqa: PIE790
            return x

        x = torch.ones([2, 2], device=DEVICE, dtype=torch.float32) * math.pi

        # Capture stdout to check print output
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            _ = incorrect_kernel(x)

        # Check that the print statement produced output
        output = captured_output.getvalue()
        self.assertIn("processing tile: ", output)
        self.assertIn("[[3.14", output)  # The value printed

    def test_ref_eager_kernel_config(self):
        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
            return out

        with assert_ref_eager_mode():
            x = torch.randn(128, 128, device=DEVICE)
            result = kernel(x)
            expected = x * 2.0
            torch.testing.assert_close(result, expected)

    def test_block_size_support(self):
        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.empty_like(x)
            for tile_m, tile_n in hl.tile([m, n], block_size=2):
                out[tile_m, tile_n] = x[tile_m, tile_n] * 2.0
            return out

        with assert_ref_eager_mode():
            x = torch.randn(128, 128, device=DEVICE)
            result = kernel(x)
            expected = x * 2.0
            torch.testing.assert_close(result, expected)

    def test_tile_begin_with_block_size_1(self):
        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            out = torch.empty_like(x)
            for tile in hl.tile(n, block_size=1):
                out[tile] = x[tile] + tile.begin
            return out

        with assert_ref_eager_mode():
            x = torch.zeros(8, device=DEVICE)
            result = kernel(x)
            expected = torch.arange(8, device=DEVICE, dtype=torch.float32)
            torch.testing.assert_close(result, expected)

    def test_tile_over_an_empty_dimension_runs_no_iterations(self):
        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros_like(x)
            for tile_m, tile_n in hl.tile([m, n]):
                out[tile_m, tile_n] = x[tile_m, tile_n] + 1
            for tile_m, tile_n in hl.tile([m, n], block_size=[2, 2]):
                out[tile_m, tile_n] += 1
            return out

        with assert_ref_eager_mode():
            for shape in ((4, 0), (0, 4), (0, 0)):
                with self.subTest(shape=shape):
                    x = torch.randn(shape, device=DEVICE)
                    torch.testing.assert_close(kernel(x), x + 2)

    def test_store_with_duplicate_indices_raises_error(self):
        """Test that hl.store with duplicate indices raises an error in ref mode."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel_with_dup_store(
            out: torch.Tensor, idx: torch.Tensor, val: torch.Tensor
        ):
            mask = torch.ones_like(idx, dtype=torch.bool)
            hl.store(out, [idx], val, extra_mask=mask)

        out = torch.zeros(4, device=DEVICE)
        idx = torch.tensor(
            [0, 0, 1], device=DEVICE, dtype=torch.int64
        )  # duplicate index
        val = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)

        with self.assertRaises(helion.exc.DuplicateStoreIndicesError):
            kernel_with_dup_store(out, idx, val)

    def test_store_dtype_conversion(self):
        """Test that hl.store properly converts dtype in ref eager mode."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            # Output tensor is bfloat16 (same as input)
            out = torch.empty_like(x)
            out_flat = out.view(-1)

            for tile_m, tile_n in hl.tile([m, n]):
                # Accumulator is float32 for numeric precision
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                acc += x[tile_m, tile_n].to(torch.float32)

                # Compute flat indices for store
                flat_indices = tile_m.index[:, None] * n + tile_n.index[None, :]

                # Store float32 acc into bfloat16 out_flat
                # This requires dtype conversion in prepare_args
                hl.store(out_flat, [flat_indices], acc)

            return out

        with assert_ref_eager_mode():
            x = torch.randn(8, 8, device=DEVICE, dtype=torch.bfloat16)
            result = kernel(x)
            torch.testing.assert_close(
                result.to(torch.float32), x.to(torch.float32), atol=1e-2, rtol=1e-2
            )

    def test_store_with_offset_index_on_non_leading_dim(self):
        """Test that storing with tile.index + offset on a non-leading dimension works."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel(x: torch.Tensor, offset: int) -> torch.Tensor:
            m, n = x.shape
            out = torch.zeros([m, n + offset], dtype=x.dtype, device=x.device)
            for tile_row, tile_col in hl.tile([m, n]):
                out[tile_row, tile_col.index + offset] = x[tile_row, tile_col]
            return out

        with assert_ref_eager_mode():
            x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=DEVICE)
            result = kernel(x, 2)
            expected = torch.tensor(
                [[0.0, 0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 4.0, 5.0, 6.0]],
                device=DEVICE,
            )
            torch.testing.assert_close(result, expected)

    def test_store_with_scalar_and_int_tensor_indices(self):
        """Multiple scalar (tile.begin) + int-tensor + tile indices must
        broadcast the mask correctly.  Mirrors the mamba2_chunk_scan pattern
        of [scalar, int_tensor, scalar, tile]."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            batch, seq, heads, feat = x.shape
            out = torch.zeros_like(x)
            for tile_b in hl.tile(batch, block_size=1):
                for tile_h in hl.tile(heads, block_size=1):
                    for tile_s, tile_f in hl.tile([seq, feat]):
                        out[tile_b.begin, tile_s.index, tile_h.begin, tile_f] = x[
                            tile_b.begin, tile_s.index, tile_h.begin, tile_f
                        ]
            return out

        with assert_ref_eager_mode():
            x = torch.randn(2, 8, 2, 4, device=DEVICE)
            result = kernel(x)
            torch.testing.assert_close(result, x)

    def test_load_2d_indexing_without_extra_mask(self):
        """Test that hl.load with two 1D tensor indices produces 2D output in ref eager mode."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel(mask: torch.Tensor) -> torch.Tensor:
            n = mask.size(0)
            out = torch.zeros_like(mask)
            for tile_i, tile_j in hl.tile([n, n]):
                # Load with two 1D tensor indices - should produce [tile_I, tile_J] output
                vals = hl.load(mask, [tile_i.index, tile_j.index])
                out[tile_i, tile_j] = vals
            return out

        with assert_ref_eager_mode():
            mask = torch.tril(torch.ones(4, 4, device=DEVICE, dtype=torch.float32))
            result = kernel(mask)
            torch.testing.assert_close(result, mask)

    def test_load_3d_indexing_without_extra_mask(self):
        """Test that hl.load with three 1D tensor indices produces 3D output in ref eager mode."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            d0, d1, d2 = x.shape
            out = torch.zeros_like(x)
            for tile_i, tile_j, tile_k in hl.tile([d0, d1, d2]):
                # Load with three 1D tensor indices - should produce [tile_I, tile_J, tile_K] output
                vals = hl.load(x, [tile_i.index, tile_j.index, tile_k.index])
                out[tile_i, tile_j, tile_k] = vals
            return out

        with assert_ref_eager_mode():
            x = torch.arange(24, device=DEVICE, dtype=torch.float32).reshape(2, 3, 4)
            result = kernel(x)
            torch.testing.assert_close(result, x)

    def test_load_is_a_value_not_a_view(self):
        """A device load keeps its values when the kernel later stores to the
        loaded region, as in compiled code."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def swap_strided(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            for tile in hl.tile(m):
                a = x[tile, 0::2]
                b = x[tile, 1::2]
                x[tile, 0::2] = b
                x[tile, 1::2] = a
            return x

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def swap_halves(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            for i in hl.grid(m):
                a = x[i, 0 : n // 2]
                b = x[i, n // 2 : n]
                x[i, 0 : n // 2] = b
                x[i, n // 2 : n] = a
            return x

        with assert_ref_eager_mode():
            x = torch.arange(64, device=DEVICE, dtype=torch.float32).reshape(8, 8)
            torch.testing.assert_close(
                swap_strided(x.clone()), x.view(8, 4, 2).flip(-1).reshape(8, 8)
            )
            torch.testing.assert_close(
                swap_halves(x.clone()), x.view(8, 2, 4).flip(1).reshape(8, 8)
            )

    def test_inplace_op_on_load_leaves_input(self):
        """An in-place op on a loaded value does not write the input."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                v = x[tile, :]
                v += 1
                out[tile, :] = v
            return out

        with assert_ref_eager_mode():
            x = torch.randn(8, 8, device=DEVICE)
            x_before = x.clone()
            torch.testing.assert_close(kernel(x), x_before + 1)
            torch.testing.assert_close(x, x_before)

    def test_host_slice_still_aliases(self):
        """A slice taken in host code stays a view of the argument."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def kernel(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            left = x[:, : n // 2]
            for tile in hl.tile(m):
                left[tile, :] = left[tile, :] + 1
            return x

        with assert_ref_eager_mode():
            x = torch.zeros(8, 8, device=DEVICE)
            expected = x.clone()
            expected[:, :4] = 1
            torch.testing.assert_close(kernel(x), expected)

    def test_grid_scalar_swap(self):
        """A 0-d load is a value too: reversing in place with scalar loads."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def reverse_subscript(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            for i in hl.grid(n // 2):
                a = x[i]
                b = x[n - 1 - i]
                x[i] = b
                x[n - 1 - i] = a
            return x

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def reverse_hl_load(x: torch.Tensor) -> torch.Tensor:
            n = x.size(0)
            for i in hl.grid(n // 2):
                a = hl.load(x, [i])
                b = hl.load(x, [n - 1 - i])
                x[i] = b
                x[n - 1 - i] = a
            return x

        with assert_ref_eager_mode():
            x = torch.arange(8, device=DEVICE, dtype=torch.float32)
            torch.testing.assert_close(reverse_subscript(x.clone()), x.flip(0))
            torch.testing.assert_close(reverse_hl_load(x.clone()), x.flip(0))

    def test_store_into_load_raises(self):
        """A store or atomic into a load raises as in compiled code, rather
        than writing into the copy the load is."""

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def hl_store_into_load(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            for t in hl.tile(x.size(0)):
                hl.store(out[0], [t], x[t])
            return out

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def atomic_into_load(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            for t in hl.tile(x.size(0)):
                hl.atomic_add(out[0], [t], x[t])
            return out

        @helion.kernel(ref_mode=helion.RefMode.EAGER)
        def setitem_into_load(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            for t in hl.tile(x.size(0)):
                left = out[:, :4]
                left[t, :] = x[t, :4]
            return out

        with assert_ref_eager_mode():
            x = torch.randn(8, 8, device=DEVICE)
            out = torch.zeros(8, 8, device=DEVICE)
            with self.assertRaises(
                helion.exc.DeviceTensorSubscriptAssignmentNotAllowed
            ):
                hl_store_into_load(x[0], out)
            with self.assertRaises(helion.exc.AtomicOnDeviceTensor):
                atomic_into_load(x[0], out)
            with self.assertRaises(
                helion.exc.DeviceTensorSubscriptAssignmentNotAllowed
            ):
                setitem_into_load(x, out)


if __name__ == "__main__":
    unittest.main()
