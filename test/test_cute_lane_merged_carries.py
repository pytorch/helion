"""A loop carry with a dim on the loop's own lane-looped block.

``acc = hl.zeros([tile_m, bn])`` updated in ``for tile_n in hl.tile(n,
block_size=bn)`` is one scalar per thread on CuTe.  With fewer threads than
``bn`` (a lane loop, or vector lanes) each thread folds its several columns
into it.  A zero-initialized additive update read only by a sum over ``bn``
(the manual looped reduction of ``examples/long_sum.py``) is still exact;
anything else would read the folded lanes and is refused.
"""

from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
import helion.language as hl


@helion.kernel(static_shapes=True)
def carry_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc + x[tile_m, tile_n]
            acc = acc - y[tile_m, tile_n]
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True)
def carry_square_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc += x[tile_m, tile_n]
        out[tile_m] = (acc * acc).sum(-1)
    return out


@helion.kernel(static_shapes=True)
def carry_amax(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc += x[tile_m, tile_n]
        out[tile_m] = acc.amax(-1)
    return out


@helion.kernel(static_shapes=True)
def carry_running_max(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = torch.maximum(acc, x[tile_m, tile_n])
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True)
def carry_ones_init(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.full([tile_m, bn], 1.0, dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc + x[tile_m, tile_n]
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True)
def carry_widened_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    # acc.sum(-1) of an int32 carry is int64.
    dtype = torch.int64 if x.dtype == torch.int32 else x.dtype
    out = torch.zeros([m], dtype=dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc + x[tile_m, tile_n]
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True)
def carry_read_in_loop(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    prefix = torch.zeros_like(x)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            acc = acc + x[tile_m, tile_n]
            prefix[tile_m, tile_n] = acc
        out[tile_m] = acc.sum(-1)
    return out, prefix


@helion.kernel(static_shapes=True)
def carry_branch_add(x: torch.Tensor, beta: float) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            if beta > 0.0:
                acc = acc + x[tile_m, tile_n]
            else:
                acc = acc - x[tile_m, tile_n]
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True)
def carry_branch_max(x: torch.Tensor, beta: float) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            if beta > 0.0:
                acc = acc + x[tile_m, tile_n]
            else:
                acc = torch.maximum(acc, x[tile_m, tile_n])
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True)
def carry_outer_masked_max(x: torch.Tensor) -> torch.Tensor:
    s, r, c = x.shape
    br = hl.register_block_size(r)
    out = torch.zeros([s, c], dtype=x.dtype, device=x.device)
    for tile_s, tile_c in hl.tile([s, c]):
        acc = hl.full([tile_s, br, tile_c], float("-inf"), dtype=x.dtype)
        for tile_r in hl.tile(r, block_size=br):
            flat = tile_s.index[:, None] * r + tile_r.index[None, :]
            valid = (flat % 3 != 0) & (tile_r.index[None, :] < r)
            acc = torch.where(
                valid[:, :, None], torch.maximum(acc, x[tile_s, tile_r, tile_c]), acc
            )
        out[tile_s, tile_c] = acc.amax(1)
    return out


def _config(bn: int, threads: int, vector: int = 1) -> helion.Config:
    return helion.Config(
        block_sizes=[bn, 1],
        num_threads=[threads, 1],
        cute_vector_widths=[vector, 1],
    )


def _columns(x: torch.Tensor, bn: int) -> torch.Tensor:
    return x.view(x.shape[0], -1, bn).sum(1)


@onlyBackends(["cute"])
@skipIfRefEager("checks the CuTe lowering of loop carries")
class TestCuteLaneMergedCarries(TestCase):
    def test_additive_sum_carry_folds_lanes_exactly(self) -> None:
        """acc = acc + x - y, then acc.sum(-1): exact with a lane loop and
        with vector lanes over bn."""
        x = torch.randint(-4, 5, (64, 512), device=DEVICE).float()
        y = torch.randint(-4, 5, (64, 512), device=DEVICE).float()
        for config in (_config(16, 4), _config(32, 8, 2), _config(16, 4, 4)):
            with self.subTest(config=config):
                bound = carry_sum.bind((x, y))
                out = bound.compile_config(config)(x, y)
                torch.testing.assert_close(out, (x - y).sum(-1), atol=0, rtol=0)
                # So is a runtime if whose branches each add or subtract.
                bound = carry_branch_add.bind((x, 0.5))
                for beta, expected in ((0.5, x.sum(-1)), (-0.5, -x.sum(-1))):
                    out = bound.compile_config(config)(x, beta)
                    torch.testing.assert_close(out, expected, atol=0, rtol=0)

    def test_folded_lanes_are_refused(self) -> None:
        """A sum of squares, an amax, a running max (also in one branch of an
        if), a nonzero start and a read of the carry in the loop would see
        the folded lanes."""
        x = torch.randint(-4, 5, (64, 512), device=DEVICE).float()
        for kernel, args in (
            (carry_square_sum, (x,)),
            (carry_amax, (x,)),
            (carry_running_max, (x,)),
            (carry_ones_init, (x,)),
            (carry_read_in_loop, (x,)),
            (carry_branch_max, (x, 0.5)),
        ):
            for config in (_config(16, 4), _config(16, 4, 4)):
                with self.subTest(kernel=kernel.name, config=config):
                    bound = kernel.bind(args)
                    with self.assertRaisesRegex(
                        helion.exc.BackendUnsupported, "own loop's block"
                    ):
                        bound.compile_config(config)

    def test_sums_wider_than_the_carry_are_refused(self) -> None:
        """An int32 carry summed as int64, or a float16 or bfloat16 carry
        whose sum accumulates in float32, would have each thread add its
        columns in the carry's dtype first: 32 tiles of 2**25 fit a column's
        int32, but four columns wrap to 0 in one Int32.  With a thread per
        column nothing folds and the int64 sum is exact."""
        x = torch.full((64, 512), 2**25, device=DEVICE, dtype=torch.int32)
        for dtype in (torch.int32, torch.float16, torch.bfloat16):
            bound = carry_widened_sum.bind((x.to(dtype),))
            for config in (_config(16, 4), _config(16, 4, 4)):
                with (
                    self.subTest(dtype=dtype, config=config),
                    self.assertRaisesRegex(
                        helion.exc.BackendUnsupported, "own loop's block"
                    ),
                ):
                    bound.compile_config(config)
        out = carry_widened_sum.bind((x,)).compile_config(_config(16, 16))(x)
        torch.testing.assert_close(out, x.sum(-1), atol=0, rtol=0)

    def test_one_element_per_thread_is_exact(self) -> None:
        """With a thread per column the carry holds one element, so the
        consumers the lane loop refuses stay exact."""
        x = torch.randint(-4, 5, (64, 512), device=DEVICE).float()
        columns = _columns(x, 16)
        for kernel, expected in (
            (carry_square_sum, (columns * columns).sum(-1)),
            (carry_amax, columns.amax(-1)),
        ):
            with self.subTest(kernel=kernel.name):
                out = kernel.bind((x,)).compile_config(_config(16, 16))(x)
                torch.testing.assert_close(out, expected, atol=0, rtol=0)

    def test_outer_index_mask_keeps_a_thread_per_element(self) -> None:
        """``tile_s.index[:, None]`` allocates a reduction block aliasing
        ``tile_s`` that gets no strategy; it must not take the third thread
        axis from ``br``, whose masked running max then holds one element per
        thread."""
        flat = torch.arange(4, device=DEVICE)[:, None] * 200 + torch.arange(
            200, device=DEVICE
        )
        masked = (flat % 3 == 0)[:, :, None]
        # Distinct negative values, larger than any of them where the mask
        # drops them: the mask, the tail lanes' masked loads of 0 and every
        # lane's own maximum decide the result.
        x = -1.0 - torch.rand(4, 200, 16, device=DEVICE)
        x = torch.where(masked, x + 10.0, x)
        expected = x.masked_fill(masked, float("-inf")).amax(1)
        bound = carry_outer_masked_max.bind((x,))
        out = bound.compile_config(helion.Config(block_sizes=[16, 4, 16]))(x)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
