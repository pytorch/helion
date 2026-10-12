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
