"""Consumers of ``hl.split`` view coordinates under CuTe.

A split view of ``x[tile, :]`` and the split's results have no block of their
own: every thread of the source block holds the element at a sub-coordinate
of its lane.  Ops that address such a dim by that sub-coordinate read the
right element; any other op would read the element its block coordinate
names (a bias loaded as ``bias[tile, :]``, a reduction over the split dim, a
reshape that interleaves the halves), so it is refused.  The kernels below
cover both sides.
"""

from __future__ import annotations

import unittest

import torch

import helion
from helion import exc
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@helion.kernel(static_shapes=True)
def bias_after_split(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    n, d = x.shape
    h = d // 2
    out = x.new_empty([n, h])
    for tile in hl.tile(n):
        a, b = hl.split(x[tile, :].reshape([tile, 2, h]).permute(0, 2, 1))
        out[tile, :] = a * bias[tile, :] + b
    return out


@helion.kernel(static_shapes=True)
def amax_after_split(x: torch.Tensor) -> torch.Tensor:
    n, d = x.shape
    h = d // 2
    out = x.new_empty([n, h])
    for tile in hl.tile(n):
        a, b = hl.split(x[tile, :].reshape([tile, 2, h]).permute(0, 2, 1))
        gated = torch.sigmoid(a) * b
        out[tile, :] = gated - gated.amax(-1, keepdim=True)
    return out


@helion.kernel(static_shapes=True)
def cumsum_after_split(x: torch.Tensor) -> torch.Tensor:
    n, d = x.shape
    h = d // 2
    out = x.new_empty([n, h])
    for tile in hl.tile(n):
        a, _ = hl.split(x[tile, :].reshape([tile, 2, h]).permute(0, 2, 1))
        out[tile, :] = torch.cumsum(a, -1)
    return out


@helion.kernel(static_shapes=True)
def interleaving_reshape(x: torch.Tensor) -> torch.Tensor:
    """``join`` reshaped without the permute back: the halves interleave."""
    n, d = x.shape
    h = d // 2
    out = torch.empty_like(x)
    for tile in hl.tile(n):
        a, b = hl.split(x[tile, :].reshape([tile, 2, h]).permute(0, 2, 1))
        out[tile, :] = hl.join(b, a).reshape([tile, d])
    return out


@helion.kernel(static_shapes=True)
def interleaving_stack(x: torch.Tensor) -> torch.Tensor:
    n, d = x.shape
    h = d // 2
    out = torch.empty_like(x)
    for tile in hl.tile(n):
        a, b = hl.split(x[tile, :].reshape([tile, 2, h]).permute(0, 2, 1))
        out[tile, :] = torch.stack((b, a), dim=-1).reshape([tile, d])
    return out


@helion.kernel(static_shapes=True)
def partial_slice_store(x: torch.Tensor) -> torch.Tensor:
    n, d = x.shape
    h = d // 2
    out = torch.zeros_like(x)
    for tile in hl.tile(n):
        _, b = hl.split(x[tile, :].reshape([tile, 2, h]).permute(0, 2, 1))
        out[tile, 0:h] = b
    return out


@helion.kernel(static_shapes=True)
def masked_store_after_split(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    n, d = x.shape
    h = d // 2
    out = x.new_zeros([n, h])
    for tile in hl.tile(n):
        a, b = hl.split(x[tile, :].reshape([tile, 2, h]).permute(0, 2, 1))
        hl.store(out, [tile, slice(None)], a + b, extra_mask=m[tile, :] > 0)
    return out


@helion.kernel(static_shapes=True)
def masked_load_after_split(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n, d = x.shape
    h = d // 2
    out = x.new_zeros([n, h])
    for tile in hl.tile(n):
        a, b = hl.split(x[tile, :].reshape([tile, 2, h]).permute(0, 2, 1))
        out[tile, :] = hl.load(y, [tile, slice(None)], extra_mask=a > 0) + b
    return out


@helion.kernel(static_shapes=True)
def self_masked_store_after_split(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, d = x.shape
    h = d // 2
    out = x.new_zeros([n, h])
    hi = x.new_empty([n, h])
    for tile in hl.tile(n):
        a, b = hl.split(x[tile, :].reshape([tile, 2, h]).permute(0, 2, 1))
        hi[tile, :] = b
        hl.store(out, [tile, slice(None)], a * b, extra_mask=a > 0)
    return out, hi


@helion.kernel(static_shapes=True)
def split_into_loop(x: torch.Tensor) -> torch.Tensor:
    n, d = x.shape
    h = d // 2
    out = x.new_empty([n, h])
    for tile in hl.tile(n):
        a, b = hl.split(x[tile, :].reshape([tile, 2, h]).permute(0, 2, 1))
        acc = hl.zeros([tile, h])
        for _ in hl.tile(2, block_size=1):
            acc = acc + a * b
        out[tile, :] = acc
    return out


@helion.kernel(static_shapes=True)
def broadcast_after_split(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    n, d = x.shape
    h = d // 2
    out = x.new_empty([n, h, w.size(0)])
    for tile in hl.tile(n):
        a, b = hl.split(x[tile, :].reshape([tile, 2, h]).permute(0, 2, 1))
        out[tile, :, :] = (a + b)[:, :, None] * w[None, None, :]
    return out


@helion.kernel(static_shapes=True)
def row_scale_after_split(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n, d = x.shape
    h = d // 2
    lo = x.new_empty([n, h])
    hi = x.new_empty([n, h])
    for tile in hl.tile(n):
        a, b = hl.split(x[tile, :].reshape([tile, 2, h]).permute(0, 2, 1))
        lo[tile, :] = torch.where(a > 0, a * scale[tile, None], b)
        hi[tile, :] = torch.sigmoid(a) * b
    return lo, hi


# One element per thread along the split dim, and two synthetic lanes.
_WIDTHS = (16, 64)


@onlyBackends(["cute"])
class TestCuteSplitViewConsumers(TestCase):
    def _refused(self, kernel: object, args: tuple[object, ...], message: str) -> None:
        with self.assertRaisesRegex(exc.BackendUnsupported, message):
            code_and_output(kernel, args, block_sizes=[32])

    def test_operand_on_another_lane_refused(self) -> None:
        # ``bias[tile, :]`` is distributed by its own full-slice block; the
        # thread holding ``a[c]`` holds ``bias`` at another position.
        for d in _WIDTHS:
            with self.subTest(d=d):
                x = torch.randn([65, d], device=DEVICE)
                bias = torch.randn([65, d // 2], device=DEVICE)
                self._refused(
                    bias_after_split, (x, bias), "with different lane coordinates"
                )

    def test_reduction_and_scan_refused(self) -> None:
        for d in _WIDTHS:
            x = torch.randn([65, d], device=DEVICE)
            for kernel, message in (
                (amax_after_split, "operand masking of a reduction"),
                (cumsum_after_split, "_associative_scan reads"),
            ):
                with self.subTest(d=d, kernel=kernel.fn.__name__):
                    self._refused(kernel, (x,), message)

    def test_interleaving_views_refused(self) -> None:
        for d in _WIDTHS:
            x = torch.randn([65, d], device=DEVICE)
            for kernel, message in (
                (interleaving_reshape, "dims whose lanes are not its coordinates"),
                (interleaving_stack, "aten.stack.default reads"),
            ):
                with self.subTest(d=d, kernel=kernel.fn.__name__):
                    self._refused(kernel, (x,), message)

    def test_partial_slice_store_refused(self) -> None:
        x = torch.randn([65, 64], device=DEVICE)
        self._refused(partial_slice_store, (x,), "through a slice other than ':'")

    def test_mask_on_another_lane_refused(self) -> None:
        # The store is re-addressed by the value's split coordinate while
        # ``m[tile, :]`` is held at its own block's lane; the mask of a load
        # of ``y[tile, :]`` likewise.
        for d in _WIDTHS:
            with self.subTest(d=d):
                x = torch.randn([65, d], device=DEVICE)
                other = torch.randn([65, d // 2], device=DEVICE)
                self._refused(
                    masked_store_after_split,
                    (x, other),
                    "with different lane coordinates",
                )
                self._refused(
                    masked_load_after_split,
                    (x, other),
                    "with different lane coordinates",
                )

    def test_mask_from_the_split_value(self) -> None:
        for d in _WIDTHS:
            with self.subTest(d=d):
                x = torch.randn([65, d], device=DEVICE)
                _code, (result, hi) = code_and_output(
                    self_masked_store_after_split, (x,), block_sizes=[32]
                )
                a, b = x.chunk(2, dim=-1)
                torch.testing.assert_close(
                    result, torch.where(a > 0, a * b, torch.zeros_like(a))
                )
                torch.testing.assert_close(hi, b)

    def test_loop_carry_refused(self) -> None:
        # The device loop's body placeholder carries no view coordinates.
        x = torch.randn([65, 64], device=DEVICE)
        self._refused(split_into_loop, (x,), "_for_loop reads")

    def test_broadcast_keeps_view_coordinates(self) -> None:
        # The broadcasting product keeps the split dim's coordinates, which
        # the store re-addresses.
        w = torch.randn([8], device=DEVICE)
        for d in _WIDTHS:
            with self.subTest(d=d):
                x = torch.randn([65, d], device=DEVICE)
                _code, result = code_and_output(
                    broadcast_after_split, (x, w), block_sizes=[32]
                )
                a, b = x.chunk(2, dim=-1)
                torch.testing.assert_close(result, (a + b)[:, :, None] * w)

    def test_row_scale_and_where(self) -> None:
        for d in _WIDTHS:
            with self.subTest(d=d):
                x = torch.randn([65, d], device=DEVICE)
                scale = torch.randn([65], device=DEVICE)
                _code, (lo, hi) = code_and_output(
                    row_scale_after_split, (x, scale), block_sizes=[32]
                )
                a, b = x.chunk(2, dim=-1)
                torch.testing.assert_close(
                    lo, torch.where(a > 0, a * scale[:, None], b)
                )
                torch.testing.assert_close(hi, torch.sigmoid(a) * b)


if __name__ == "__main__":
    unittest.main()
