"""Which lowering a Metal matmul takes, read from the MSL a CPU bind renders:
MPP's matmul2d, or the Scalar contraction.  Binds CPU tensors only; the GPU
module ``test_metal_mpp_matmul.py`` compiles and runs MPP kernels."""

from __future__ import annotations

import unittest

import pytest
import torch

import helion
from helion._testing import skipUnlessBackends
from helion._testing.metal import render_msl
import helion.language as hl

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["metal"])]

F32 = torch.float32
I32 = torch.int32


def _float_args() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(32, 64), torch.randn(64, 32)


def _int8_args() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.randint(-128, 128, (32, 64), dtype=torch.int8),
        torch.randint(-128, 128, (64, 32), dtype=torch.int8),
    )


@helion.kernel(backend="metal", static_shapes=True)
def dot_beside_a_branch_product(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Two int8 hl.dot calls in one K loop: one with an acc, and one with
    none under an ``if``, outside its K loop's own body."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=I32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=I32)
        last = hl.zeros([tile_m, tile_n], dtype=I32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
            if tile_k.begin >= 0:
                last = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], out_dtype=I32)
        out[tile_m, tile_n] = acc + last
    return out


@helion.kernel(backend="metal", static_shapes=True)
def carry_reads_its_seed(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A per-tile product updating a carry whose update reads the fill the
    carry was seeded from."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        seed = hl.full([tile_m, tile_n], 0.5, dtype=F32)
        acc = seed
        for tile_k in hl.tile(k):
            acc = acc + torch.mm(x[tile_m, tile_k], y[tile_k, tile_n]) * seed
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def per_tile_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """acc + an hl.dot without an acc argument, per K tile, in the acc's dtype."""
    m, k = x.size()
    n = y.size(1)
    dtype = I32 if x.dtype == torch.int8 else F32
    out = torch.empty([m, n], dtype=dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=dtype)
        for tile_k in hl.tile(k):
            acc = acc + hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], out_dtype=dtype)
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def one_seed_product_first(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """One hl.zeros seeding two K loops' carries: a per-tile product's
    (acc + mm), then an accumulator's (addmm)."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        seed = hl.zeros([tile_m, tile_n], dtype=F32)
        acc1 = seed
        acc2 = seed
        for tile_k in hl.tile(k):
            acc1 = acc1 + torch.mm(x[tile_m, tile_k], y[tile_k, tile_n])
        for tile_k in hl.tile(k):
            acc2 = torch.addmm(acc2, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc1 + acc2
    return out


@helion.kernel(backend="metal", static_shapes=True)
def one_seed_group_first(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """one_seed_product_first with the two K loops swapped."""
    m, k = x.size()
    n = y.size(1)
    out = torch.empty([m, n], dtype=F32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        seed = hl.zeros([tile_m, tile_n], dtype=F32)
        acc1 = seed
        acc2 = seed
        for tile_k in hl.tile(k):
            acc2 = torch.addmm(acc2, x[tile_m, tile_k], y[tile_k, tile_n])
        for tile_k in hl.tile(k):
            acc1 = acc1 + torch.mm(x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc1 + acc2
    return out


class TestMppCodegen(unittest.TestCase):
    def test_dot_the_planner_skips_keeps_every_dot_scalar(self) -> None:
        """An hl.dot without an acc argument outside its K loop's body is no
        per-tile product the planner takes, so neither dot of the kernel runs
        on MPP: both stay on the Scalar contraction, at a config whose
        threads MPP would reject."""
        config = helion.Config(block_sizes=[16, 16, 16], num_threads=[1, 1, 16])
        msl = render_msl(dot_beside_a_branch_product, _int8_args(), config)
        self.assertNotIn("_helion_mpp_run", msl)
        self.assertNotIn("matmul2d_descriptor", msl)

    def test_carry_reading_its_seed_is_not_folded(self) -> None:
        """The fill is the carry's own storage, which a fold updates in place,
        so the carry is no fold: the product is read through its
        materialization, and the MSL generates.  (Through a value computed
        from the seed: TestMppFold.test_kept_in_the_scalar_layout.)"""
        config = helion.Config(block_sizes=[16, 16, 16])
        msl = render_msl(carry_reads_its_seed, _float_args(), config)
        self.assertIn("matmul2d_descriptor", msl)
        self.assertIn("_mpp_mat", msl)

    def test_fold_beside_a_group_of_its_seed(self) -> None:
        """A seed the accumulator group's op updates in place is no storage
        for a fold: the per-tile product is read through its
        materialization, whichever loop comes first in the source (the fold
        is planned after every group has claimed its storage)."""
        config = helion.Config(block_sizes=[16, 16, 16, 16], metal_matmul="mpp")
        for kernel in (one_seed_product_first, one_seed_group_first):
            msl = render_msl(kernel, _float_args(), config)
            self.assertIn("matmul2d_descriptor", msl)
            self.assertIn("_mpp_mat", msl)

    def test_per_tile_dot(self) -> None:
        """A per-K-tile hl.dot takes MPP from int8 operands; a float one
        keeps the scalar contraction, which lowers it at every config."""
        config = helion.Config(block_sizes=[16, 16, 16])
        msl = render_msl(per_tile_dot, _int8_args(), config)
        self.assertIn("matmul2d_descriptor", msl)
        msl = render_msl(per_tile_dot, _float_args(), config)
        self.assertNotIn("matmul2d_descriptor", msl)


if __name__ == "__main__":
    unittest.main()
