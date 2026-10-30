"""The deep (bk=64) seeds of the M-paired block_m=512 tcgen05 tile.

CPU code-generation tests: the seeds depend only on the shape facts and
the B200 SMEM model, which the bind patches pin.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from ._cute_binding import _cpu_bind
import helion
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from helion._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import patch_cute_mma_support
import helion.language as hl

PAIR_BM = 2 * TCGEN05_TWO_CTA_BLOCK_M


def _make_plain_gemm() -> helion.Kernel:
    """A fresh kernel per bind: ``Kernel.bind`` memoizes by argument key."""

    @helion.kernel(backend="cute", static_shapes=True)
    def _plain_gemm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.size()
        _, n = y.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            out[tile_m, tile_n] = acc.to(x.dtype)
        return out

    return _plain_gemm


def _make_bias_gemm() -> helion.Kernel:
    @helion.kernel(backend="cute", static_shapes=True)
    def _bias_gemm(
        x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        m, k = x.size()
        _, n = y.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            out[tile_m, tile_n] = (acc + bias[tile_m, tile_n]).to(x.dtype)
        return out

    return _bias_gemm


def _make_batched_gemm() -> helion.Kernel:
    @helion.kernel(backend="cute", static_shapes=True)
    def _batched_gemm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, m, k = x.size()
        _, _, n = y.size()
        out = torch.empty([b, m, n], dtype=x.dtype, device=x.device)
        for tile_b, tile_m, tile_n in hl.tile([b, m, n]):
            acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.baddbmm(
                    acc, x[tile_b, tile_m, tile_k], y[tile_b, tile_k, tile_n]
                )
            out[tile_b, tile_m, tile_n] = acc.to(x.dtype)
        return out

    return _batched_gemm


def _bind(kernel: helion.Kernel, args: tuple[torch.Tensor, ...]) -> object:
    """Bind on a 148-SM B200 with its 227 KiB opt-in SMEM, on any host."""
    with (
        patch_cute_mma_support(),
        patch("helion.language.matmul_ops._cuda_num_sms_or_zero", return_value=148),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        return _cpu_bind(kernel, args)


def _m_pair_seeds(spec: object) -> list[dict[str, object]]:
    seeds = [dict(seed.config) for seed in spec.autotune_seed_configs()]  # pyrefly: ignore[missing-attribute]
    return [
        seed
        for seed in seeds
        if seed["block_sizes"][0] == PAIR_BM and seed.get("tcgen05_cluster_m") == 2
    ]


@onlyBackends(["cute"])
class TestCuteTcgen05MPairSeeds(TestCase):
    def _assert_deep_m_pair_seeds(self, args: tuple[torch.Tensor, ...]) -> None:
        spec = _bind(_make_plain_gemm(), args).config_spec
        tcgen05 = spec._cute_tcgen05_config
        self.assertTrue(tcgen05._m_pair_block_m_is_valid())
        m_pair = _m_pair_seeds(spec)
        deep = [seed for seed in m_pair if seed["block_sizes"][2] == 64]
        shallow = [seed for seed in m_pair if seed["block_sizes"][2] == 128]
        # The bk=128 pair seeds (2x1 and the 2x2 super-tile) come first and
        # keep their shallow ring; the deep ring follows on both clusters.
        self.assertEqual(len(shallow), 2)
        self.assertEqual(
            [seed["tcgen05_cluster_n"] for seed in deep],
            [2, 1],
        )
        all_seeds = [dict(seed.config) for seed in spec.autotune_seed_configs()]
        self.assertLess(
            max(all_seeds.index(seed) for seed in shallow),
            min(all_seeds.index(seed) for seed in deep),
        )
        for seed in deep:
            self.assertEqual(
                seed["block_sizes"], [PAIR_BM, TCGEN05_TWO_CTA_BLOCK_N, 64]
            )
            self.assertEqual(seed["pid_type"], "persistent_interleaved")
            self.assertEqual(seed["tcgen05_cluster_m"], 2)
            # Three 48 KiB stages plus the four-deep (128, 32) C ring fit
            # the 199 KiB AB budget; four stages do not fit even a two-deep
            # ring.
            self.assertEqual(seed["tcgen05_ab_stages"], 3)
            self.assertEqual(seed["tcgen05_acc_stages"], 2)
            self.assertEqual(seed["tcgen05_c_stages"], 4)
            self.assertEqual(seed["l2_groupings"], [1])
            self.assertEqual(seed["indexing"], ["tensor_descriptor"] * 3)
            # Like every tcgen05 seed where the knob is searched (sm_100).
            self.assertIs(seed["cute_pdl"], True)
            self.assertNotIn("tcgen05_strategy", seed)
            self.assertNotIn("tcgen05_persistence_model", seed)
            # The seed is a valid config as written and the search's
            # projection leaves it alone.
            strict = dict(seed)
            spec.normalize(strict, _fix_invalid=False)
            projected = dict(seed)
            spec.normalize(projected, _fix_invalid=True)
            for key in (
                "block_sizes",
                "tcgen05_cluster_m",
                "tcgen05_cluster_n",
                "tcgen05_ab_stages",
                "tcgen05_acc_stages",
                "tcgen05_c_stages",
                "pid_type",
            ):
                self.assertEqual(projected[key], seed[key], key)
            self.assertEqual(projected["tcgen05_strategy"], "role_local_monolithic")
            self.assertEqual(
                projected["tcgen05_persistence_model"], "static_persistent"
            )
        self.assertTrue(
            tcgen05.c_stages_fits(
                bm=PAIR_BM,
                bn=TCGEN05_TWO_CTA_BLOCK_N,
                bk=64,
                cluster_m=2,
                ab_stages=3,
                c_stages=4,
                has_source_c=False,
            )
        )
        self.assertFalse(
            tcgen05.c_stages_fits(
                bm=PAIR_BM,
                bn=TCGEN05_TWO_CTA_BLOCK_N,
                bk=64,
                cluster_m=2,
                ab_stages=4,
                c_stages=2,
                has_source_c=False,
            )
        )

    def test_gradient_gemms_seed_the_deep_pair_ring(self) -> None:
        """Both 2048x4096x2048 gradient GEMMs (K-major B, M-major A).

        Their 512x256 pair grid is one wave of 128 CTAs; the seeded ring is
        the config both cold searches that reached it selected.
        """
        m, k, n = 2048, 4096, 2048
        mat1 = torch.empty([m, k], dtype=torch.float16)
        mat2 = torch.empty([k, n], dtype=torch.float16)
        grad_out = torch.empty([m, n], dtype=torch.float16)
        self._assert_deep_m_pair_seeds((grad_out, mat2.T))
        self._assert_deep_m_pair_seeds((mat1.T, grad_out))

    def test_flattened_batch_gemm_seeds_the_deep_pair_ring(self) -> None:
        """A [B*M, K] x [K, N] GEMM (broadcast_matmul: 8192x1024x768)."""
        x = torch.empty([16 * 512, 768], dtype=torch.float16)
        w = torch.empty([768, 1024], dtype=torch.float16)
        self._assert_deep_m_pair_seeds((x, w))

    def test_square_gemm_keeps_the_shallow_pair_seeds_and_adds_the_deep_ring(
        self,
    ) -> None:
        x = torch.empty([4096, 4096], dtype=torch.bfloat16)
        y = torch.empty([4096, 4096], dtype=torch.bfloat16)
        self._assert_deep_m_pair_seeds((x, y))

    def test_no_deep_pair_seeds_where_the_pair_tile_is_invalid(self) -> None:
        """Rows not divisible by 512, a 64-misaligned K, an exact-shape
        epilogue and a batched GEMM keep the seed list as it was."""
        cases: list[tuple[helion.Kernel, tuple[torch.Tensor, ...]]] = [
            (
                _make_plain_gemm(),
                (
                    torch.empty([1792, 2048], dtype=torch.float16),
                    torch.empty([2048, 2048], dtype=torch.float16),
                ),
            ),
            (
                _make_plain_gemm(),
                (
                    torch.empty([2048, 2016], dtype=torch.float16),
                    torch.empty([2016, 4096], dtype=torch.float16),
                ),
            ),
            (
                _make_bias_gemm(),
                (
                    torch.empty([2048, 4096], dtype=torch.float16),
                    torch.empty([4096, 2048], dtype=torch.float16),
                    torch.empty([2048, 2048], dtype=torch.float16),
                ),
            ),
            (
                _make_batched_gemm(),
                (
                    torch.empty([4, 1024, 512], dtype=torch.float16),
                    torch.empty([4, 512, 1024], dtype=torch.float16),
                ),
            ),
        ]
        for kernel, args in cases:
            spec = _bind(kernel, args).config_spec
            tcgen05 = spec._cute_tcgen05_config
            self.assertEqual(tcgen05._m_pair_deep_seed_configs(), [])
            self.assertFalse(
                any(seed["block_sizes"][2] == 64 for seed in _m_pair_seeds(spec)),
                [tuple(t.shape) for t in args],
            )

    def test_deep_pair_seeds_need_the_b200_smem_budget(self) -> None:
        """Without a recorded AB budget the ring cannot be sized: no seeds."""
        x = torch.empty([2048, 2048], dtype=torch.float16)
        y = torch.empty([2048, 4096], dtype=torch.float16)
        spec = _bind(_make_plain_gemm(), (x, y)).config_spec
        tcgen05 = spec._cute_tcgen05_config
        self.assertEqual(len(tcgen05._m_pair_deep_seed_configs()), 2)
        with patch.object(tcgen05, "ab_stages_three_search_constraints", None):
            self.assertEqual(tcgen05._m_pair_deep_seed_configs(), [])


if __name__ == "__main__":
    unittest.main()
