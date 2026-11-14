"""A loop carry with a dim on the loop's own lane-looped block.

``acc = hl.zeros([tile_m, bn])`` updated in ``for tile_n in hl.tile(n,
block_size=bn)`` is one scalar per thread on CuTe.  With fewer threads than
``bn`` (a lane loop, or vector lanes) each thread folds its several columns
into it.  A zero-initialized additive update read only by a sum over ``bn``
in the carry's dtype (the manual looped reduction of
``examples/long_sum.py``) is still exact; anything else would read the
folded lanes and is refused.  The sum may read the carry through the mask
the compiler puts before it when padded lanes would not add zero, if the
update reads the tile's lanes along every block it folds: those lanes never
reach the carry.  An update that is the same in every lane of a folded
block is not repeated per lane, so a thread would add it once for all of
them; it is refused (test_cute_lane_merged_carry_refusals.py).
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
def carry_exp_sum(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            # A padded lane would add exp(0): the sum reads acc through a mask.
            acc = acc + torch.exp(x[tile_m, tile_n])
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True)
def carry_two_block_exp_sum(x: torch.Tensor) -> torch.Tensor:
    m, n, k = x.shape
    bn = hl.register_block_size(n)
    bk = hl.register_block_size(k)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn, bk], dtype=x.dtype)
        for tile_n, tile_k in hl.tile([n, k], block_size=[bn, bk]):
            acc = acc + torch.exp(x[tile_m, tile_n, tile_k])
        out[tile_m] = acc.sum(2).sum(1)
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
def carry_branch_exp(x: torch.Tensor, beta: float) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            if beta > 0.0:
                acc = acc + torch.exp(x[tile_m, tile_n])
            else:
                acc = acc + x[tile_m, tile_n]
        out[tile_m] = acc.sum(-1)
    return out


@helion.kernel(static_shapes=True)
def carry_branch_exp_or_one(x: torch.Tensor, beta: float) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.zeros([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, bn], dtype=x.dtype)
        for tile_n in hl.tile(n, block_size=bn):
            if beta > 0.0:
                acc = acc + torch.exp(x[tile_m, tile_n])
            else:
                acc = acc + 1.0
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


def _assert_sum_close(
    out: torch.Tensor, terms: torch.Tensor, dims: int | tuple[int, ...]
) -> None:
    """``out``, an fp32 sum of ``terms`` over ``dims``, against their exact
    sum.  fp32 additions of ``n`` terms in any order (each thread's lanes,
    then a warp's tree) err by at most ``(n - 1) * 2**-24`` times the sum of
    the terms' magnitudes, and exps within 2 ulp of the exact ones by
    ``4 * 2**-24`` of it more: less than ``(n + 4) * 2**-24`` of it."""
    exact = terms.double()
    error = (out.double() - exact.sum(dims)).abs()
    bound = (terms.numel() // out.numel() + 4) * 2.0**-24 * exact.abs().sum(dims)
    assert bool((error <= bound).all()), (error / bound).max().item()


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

    def test_masked_sum_carry_matches_eager(self) -> None:
        """acc = acc + exp(x), whose padded lanes would add exp(0): summed
        through the mask before acc.sum(-1), with and without a partial tile
        of bn (512 and 500 columns).  So is a runtime if adding exp(x) in one
        branch, whose update is selected, and x or 1.0 in the other: the exp
        branch's loads run the loop body per lane, so the 1.0 reaches every
        lane of all 32 tiles, padded ones included (512 per row)."""
        generator = torch.Generator().manual_seed(0)
        x512, x500 = (
            torch.randn(64, n, generator=generator).to(DEVICE) for n in (512, 500)
        )
        # Every lane form on the partial tile; full tiles under one of them.
        for x, configs in (
            (x500, (_config(16, 4), _config(32, 8, 2), _config(16, 4, 4))),
            (x512, (_config(16, 4, 4),)),
        ):
            bound = carry_exp_sum.bind((x,))
            for config in configs:
                with self.subTest(n=x.shape[1], config=config):
                    out = bound.compile_config(config)(x)
                    _assert_sum_close(out, torch.exp(x.double()), -1)
        # A loop over two blocks folding only bk, whose update reads the lanes
        # of both: the lane loop over bk runs it per lane, partial bk tile
        # (40 = 16 + 16 + 8) included.
        x3 = torch.randn(8, 64, 40, generator=generator).to(DEVICE)
        two_block = helion.Config(block_sizes=[4, 16, 1], num_threads=[4, 4, 1])
        out = carry_two_block_exp_sum.bind((x3,)).compile_config(two_block)(x3)
        _assert_sum_close(out, torch.exp(x3.double()), (1, 2))
        # On the 500 columns the 1.0 reaches the padded lanes too: 512 per row.
        ones = torch.ones(x500.shape[0], 512, dtype=torch.float64, device=DEVICE)
        for kernel, other in (
            (carry_branch_exp, x500.double()),
            (carry_branch_exp_or_one, ones),
        ):
            bound = kernel.bind((x500, 0.5))
            for config in (_config(16, 4), _config(16, 4, 4)):
                compiled = bound.compile_config(config)
                for beta, terms in ((0.5, torch.exp(x500.double())), (-0.5, other)):
                    with self.subTest(kernel=kernel.name, config=config, beta=beta):
                        _assert_sum_close(compiled(x500, beta), terms, -1)

    def test_folded_lanes_are_refused(self) -> None:
        """A sum of squares, an amax, a running max (also in one branch of an
        if), a nonzero start and a read of the carry in the loop would see
        the folded lanes (test_cute_lane_merged_carry_refusals.py has the
        refusals decided by the update's lane reads and the sum's dtype)."""
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
        consumers the lane loop refuses stay exact: a sum of squares, an
        amax, and an int32 carry's int64 sum of 32 tiles of 2**25, which four
        folded columns would wrap to 0 in one Int32
        (test_cute_lane_merged_carry_refusals.py)."""
        x = torch.randint(-4, 5, (64, 512), device=DEVICE).float()
        columns = _columns(x, 16)
        wide = torch.full((64, 512), 2**25, device=DEVICE, dtype=torch.int32)
        for kernel, args, expected in (
            (carry_square_sum, (x,), (columns * columns).sum(-1)),
            (carry_amax, (x,), columns.amax(-1)),
            (carry_widened_sum, (wide,), wide.sum(-1)),
        ):
            with self.subTest(kernel=kernel.name):
                out = kernel.bind(args).compile_config(_config(16, 16))(*args)
                torch.testing.assert_close(out, expected, atol=0, rtol=0)

    def test_outer_index_masked_running_max_is_exact(self) -> None:
        """A running max masked by the outer tile's index
        (``tile_s.index[:, None]``, which allocates no reduction block) leaves
        ``br`` its thread axis, so the masked carry holds one element per
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
