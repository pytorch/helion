"""The CTA-pair seeds and AB ring of an fp32 GEMM computed as tf32.

CPU code-generation tests: the seeds depend only on the shape facts, the
operand width and the GB300 SMEM model, which the bind patches pin.
"""

from __future__ import annotations

import contextlib
import functools
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._compiler.cute.tcgen05_config import seeds
from helion._compiler.cute.tcgen05_config import smem
from helion._testing import TestCase
from helion._testing import default_cute_mma_support
from helion._testing import onlyBackends
from helion._testing import patch_cute_mma_support
from helion._testing import patch_target_facts
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import gb300_target
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only

# The tf32 MMA is off in the default test record: fp32 operands would take no
# tcgen05 MMA at all.
TF32_SUPPORT = default_cute_mma_support(tcgen05_tf32=True)


@contextlib.contextmanager
def _gb300() -> Iterator[None]:
    """Bind, seed and render on the GB300 stand-in: its SMEM model and SM
    count, and an MMA record with the tf32 MMA."""
    with (
        gb300_target(),
        patch_cute_mma_support(TF32_SUPPORT),
        patch_target_facts(num_sm=152),
        forbid_cuda_work(),
    ):
        yield


@helion.kernel(backend="cute", static_shapes=True, dot_precision="tf32")
def _gemm(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    m, k = x.size()
    _, n = y.size()
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, dot_precision="tf32")
def _k64_gemm(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    """The same GEMM over a K block bounded to 64 elements."""
    m, k = x.size()
    _, n = y.size()
    block_k = hl.register_block_size(64, 64)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)


@helion.kernel(backend="cute", static_shapes=True, dot_precision="tf32")
def _k128_gemm(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    """The same GEMM over a K block bounded to 128 elements."""
    m, k = x.size()
    _, n = y.size()
    block_k = hl.register_block_size(128, 128)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k, block_size=block_k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)


_KERNELS = {"gemm": _gemm, "k64": _k64_gemm, "k128": _k128_gemm}


@functools.cache
def _bound(
    dtype: torch.dtype, m: int, n: int, k: int, kernel: str = "gemm"
) -> BoundKernel:
    """One bind per problem for the module (the tests only read, seed and
    render): ``x`` row-major, ``y`` K-major (a Linear's weight), as
    ``x @ w.t()``."""
    with _gb300():
        return _KERNELS[kernel]._bind_isolated(
            (
                torch.empty((m, k), dtype=dtype),
                torch.empty((n, k), dtype=dtype).T,
                torch.empty((m, n), dtype=dtype),
            )
        )


def _bk_choices(bound: BoundKernel) -> tuple[int, ...]:
    with _gb300():
        return seeds.cluster_m2_seed_bk_choices(bound.config_spec.tcgen05)


def _pair_seeds(bound: BoundKernel) -> list[dict[str, object]]:
    """The kernel's normalized CTA-pair seeds (compiler and autotune)."""
    spec = bound.config_spec
    found = []
    with _gb300():
        for seed in [
            *spec.compiler_seed_configs,
            *seeds.autotune_seed_configs(spec.tcgen05),
        ]:
            fixed = helion.Config.from_dict(dict(seed.config))
            spec.normalize(fixed, _fix_invalid=True)
            if fixed.config.get("tcgen05_cluster_m") == 2:
                found.append(dict(fixed.config))
    return found


@onlyBackends(["cute"])
class TestCuteTcgen05Tf32Seeds(TestCase):
    def test_fp32_ab_ring_is_sized_by_the_smem_budget(self) -> None:
        # A 32-deep fp32 K step stages the bytes of a 64-deep 16-bit one.
        self.assertEqual(smem._get_dtype_ab_stages_hard_cap(4), 12)
        self.assertEqual(
            smem._get_dtype_ab_stages_hard_cap(4), smem._get_dtype_ab_stages_hard_cap(2)
        )

    def test_tf32_pair_seeds_stage_the_16_bit_seeds_bytes(self) -> None:
        # 16384x3072x768: whole 256x256 tiles, an even count of N tiles (the
        # 2x2 seeds keep their cluster).
        pairs = _pair_seeds(_bound(torch.float32, 16384, 3072, 768))
        self.assertTrue(pairs)
        rings = {(seed["block_sizes"][2], seed["tcgen05_ab_stages"]) for seed in pairs}
        # Three 64-deep and six 32-deep stages: 192 KiB of ring either way,
        # the bytes of the 16-bit seeds' three 128-deep and six 64-deep ones.
        self.assertEqual(rings, {(64, 3), (32, 6)})
        for seed in pairs:
            self.assertEqual(seed["block_sizes"][:2], [256, 256])
        # cuBLAS's tf32 2-CTA tile, 256x256x32 x 6, as the 2x1 and the 2x2
        # cluster, static and CLC-persistent.
        deep = {
            (seed["tcgen05_cluster_n"], seed["tcgen05_persistence_model"])
            for seed in pairs
            if seed["block_sizes"][2] == 32
        }
        self.assertLessEqual(
            {(1, "static_persistent"), (2, "static_persistent"), (1, "clc_persistent")},
            deep,
        )

    def test_tf32_seeds_lower_to_the_tcgen05_pair(self) -> None:
        bound = _bound(torch.float32, 16384, 3072, 768)
        for seed in _pair_seeds(bound):
            if seed["block_sizes"][2] != 32 or seed["tcgen05_cluster_n"] != 1:
                continue
            with _gb300():
                code = bound.to_code(helion.Config.from_dict(seed))
            # The six-stage TF32 ring, not the SIMT fallback a 256 KiB ring
            # lowers to.
            self.assertIn("make_tma_umma_pipeline(num_stages=6", code)
            self.assertIn("'input_dtype': 'cutlass.TFloat32'", code)

    def test_tf32_clc_2x2_seed_takes_the_deep_ring(self) -> None:
        # The deep CLC seed's 2x2 twin and its fallback-cluster twin, beside
        # the 2x2 seed on the 64-deep ring (an even count of N tiles: 12).
        clc_2x2 = {
            (
                seed["block_sizes"][2],
                seed["tcgen05_ab_stages"],
                seed["l2_groupings"][0],
                seed["tcgen05_fallback_cluster"],
            )
            for seed in _pair_seeds(_bound(torch.float32, 16384, 3072, 768))
            if seed["tcgen05_cluster_n"] == 2
            and seed["tcgen05_persistence_model"] == "clc_persistent"
        }
        self.assertLessEqual(
            {(32, 6, 4, False), (32, 6, 4, True), (64, 3, 8, False), (64, 3, 8, True)},
            clc_2x2,
        )
        # An odd count of N tiles (3) pairs no N tiles: no 2x2 seed.
        self.assertFalse(
            [
                seed
                for seed in _pair_seeds(_bound(torch.float32, 16384, 768, 768))
                if seed["tcgen05_cluster_n"] == 2
            ]
        )

    def test_tf32_pair_seeds_stay_within_the_k_block(self) -> None:
        # The pair seeds' K tile stages the 16-bit seeds' 128-deep step in
        # fp32 bytes, 64 elements, within the K block's bounds: a K block
        # bounded to 64 takes it whole, and one bounded to 128, above it,
        # keeps its own floor rather than losing every pair seed.
        for kernel, block_k in (("k64", 64), ("k128", 128)):
            bound = _bound(torch.float32, 16384, 3072, 768, kernel)
            self.assertEqual(_bk_choices(bound), (block_k,), kernel)
        # A K of one 64-deep step takes it whole, three stages deep (the bytes
        # of the 16-bit seeds' three 128-deep stages).
        bound = _bound(torch.float32, 16384, 3072, 64)
        self.assertEqual(_bk_choices(bound), (64,))
        rings = {
            (seed["block_sizes"][2], seed["tcgen05_ab_stages"])
            for seed in _pair_seeds(bound)
        }
        self.assertIn((64, 3), rings)

    def test_tf32_edge_seeds_take_the_64_deep_tile(self) -> None:
        # The edge + K-tail family (M and N ragged, a K tail): its 128-deep
        # seed would be a 256 KiB ring at the family's stages, the SIMT
        # fallback; the 64-deep tile is one the family admits.
        bound = _bound(torch.float32, 3000, 3000, 1000)
        self.assertEqual(_bk_choices(bound), (64,))
        pairs = _pair_seeds(bound)
        self.assertTrue(pairs)
        self.assertNotIn(128, {seed["block_sizes"][2] for seed in pairs})

    def test_16_bit_seeds_keep_their_k_tiles(self) -> None:
        bound = _bound(torch.bfloat16, 16384, 3072, 768)
        with _gb300():
            self.assertEqual(seeds.seed_block_k(bound.config_spec.tcgen05, 128), 128)
        self.assertEqual(_bk_choices(bound)[0], 128)
        # The CLC 2x2 seeds keep the 128-deep ring (no deep CLC 2x2 twin).
        self.assertFalse(
            [
                seed
                for seed in _pair_seeds(bound)
                if seed["tcgen05_cluster_n"] == 2
                and seed["tcgen05_persistence_model"] == "clc_persistent"
                and seed["block_sizes"][2] != 128
            ]
        )
