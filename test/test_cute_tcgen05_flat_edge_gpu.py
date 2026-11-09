"""GPU tests of tcgen05 kernels over ragged tiles.

The flat one-CTA pipeline, the persistent one-CTA N-edge tiles, the staged
rows' one-tile lookahead and the ragged two-CTA search.

Partial M / N output tiles and a K tail used to take the one-warp scalar
SMEM producer and store 16-bit scalars; the flat pipeline now loads every
box through TMA (zero-filled past the extents) and stores interior tiles as
vectors and fringe tiles chunk-predicated. These tests pin flat tcgen05
configs over odd extents for the plain, transposed, bias, fp16 and batched
GEMMs and check the results against torch.
"""

from __future__ import annotations

import itertools
import unittest

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing.cute_aux import rank_two_aux
from helion._testing.cute_dense_tcgen05 import bias_mm_into
from helion._testing.cute_dense_tcgen05 import plain_mm_into
from helion._testing.cute_dense_tcgen05 import promoted_bias_mm_into
from helion._testing.cute_dense_tcgen05 import scale_bias_mm_into
import helion.language as hl

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="tcgen05 requires an sm_100 GPU",
)


@helion.kernel(static_shapes=True)
def _gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def _bias_gemm(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = (acc + bias[tile_n]).to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def _bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    batch, m, k = a.size()
    _, _, n = b.size()
    out = torch.empty([batch, m, n], dtype=a.dtype, device=a.device)
    for tile_b, tile_m, tile_n in hl.tile([batch, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc, a[tile_b, tile_m, tile_k], b[tile_b, tile_k, tile_n]
            )
        out[tile_b, tile_m, tile_n] = acc.to(out.dtype)
    return out


@helion.kernel(static_shapes=True)
def _gemm_into(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def _flat_config(block_sizes: list[int], stages: int = 4) -> helion.Config:
    return helion.Config(
        block_sizes=block_sizes,
        indexing=["tensor_descriptor"] * 3,
        pid_type="flat",
        tcgen05_cluster_m=1,
        tcgen05_ab_stages=stages,
        l2_groupings=[1],
    )


@onlyBackends(["cute"])
@skipIfNotCUDA()
class TestCuteTcgen05FlatEdgeGpu(RefEagerTestDisabled, TestCase):
    def _check(
        self,
        kernel: helion.Kernel,
        args: tuple[torch.Tensor, ...],
        expected: torch.Tensor,
        block_sizes: list[int],
        *,
        stages: int = 4,
        rtol: float = 1e-2,
        atol: float = 1e-1,
    ) -> None:
        bound = kernel.bind(args)
        config = _flat_config(block_sizes, stages)
        code = bound.to_code(config)
        # TMA producers for A and B, no scalar SMEM producer fallback.
        self.assertIn("tma_tensor_a", code)
        self.assertIn("tma_tensor_b", code)
        self.assertNotIn("tcgen05_scalar_smem_mbar", code)
        bound.set_config(config)
        torch.testing.assert_close(bound(*args), expected, rtol=rtol, atol=atol)

    def test_double_edge_and_k_tail(self) -> None:
        a = torch.randn(1000, 1000, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(1000, 1000, device=DEVICE, dtype=torch.bfloat16)
        self._check(_gemm, (a, b), a @ b, [128, 256, 64])
        self._check(_gemm, (a, b), a @ b, [128, 128, 64])
        # 256-byte K packets (bk=128) over the same ragged tiles (three
        # stages: the SMEM budget of the validated non-TVM-FFI path).
        self._check(_gemm, (a, b), a @ b, [128, 128, 128], stages=3)

    def test_k_tail_only(self) -> None:
        a = torch.randn(1024, 1000, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(1000, 1024, device=DEVICE, dtype=torch.bfloat16)
        self._check(_gemm, (a, b), a @ b, [128, 256, 64])

    def test_m_edge_only_and_n_edge_only(self) -> None:
        a = torch.randn(1000, 1024, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        self._check(_gemm, (a, b), a @ b, [128, 256, 64])
        a = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(1024, 1000, device=DEVICE, dtype=torch.bfloat16)
        self._check(_gemm, (a, b), a @ b, [128, 256, 64])

    def test_n_not_a_vector_multiple(self) -> None:
        # 999 columns of a contiguous output: the store's vector width divides
        # the row stride (999 itself), so whole predicated chunks cover the
        # N edge and the element tail stays dead.
        a = torch.randn(777, 555, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(555, 999, device=DEVICE, dtype=torch.bfloat16)
        self._check(_gemm, (a, b), a @ b, [128, 128, 64])

    def test_n_edge_straddles_a_vector_of_a_padded_output(self) -> None:
        # A 999-column output with a 1024-element row stride takes 16-byte
        # vectors, so the last chunk of every row straddles the N edge and
        # takes the element-predicated tail copy.
        a = torch.randn(777, 555, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(555, 999, device=DEVICE, dtype=torch.bfloat16)
        out = torch.zeros(777, 1024, device=DEVICE, dtype=torch.bfloat16)[:, :999]
        bound = _gemm_into.bind((a, b, out))
        config = _flat_config([128, 128, 64])
        code = bound.to_code(config)
        # The chunk-predicated fringe store and its element tail.
        self.assertIn("tcgen05_edge_pred", code)
        self.assertIn("tcgen05_edge_tail_pred", code)
        bound.set_config(config)
        bound(a, b, out)
        torch.testing.assert_close(out, a @ b, rtol=1e-2, atol=1e-1)

    def test_transposed_operands_and_fp16(self) -> None:
        a = torch.randn(1000, 1000, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(1000, 1000, device=DEVICE, dtype=torch.bfloat16)
        self._check(_gemm, (a.T, b), a.T @ b, [128, 256, 64])
        self._check(_gemm, (a, b.T), a @ b.T, [128, 256, 64])
        a16 = torch.randn(1000, 1000, device=DEVICE, dtype=torch.float16)
        b16 = torch.randn(1000, 1000, device=DEVICE, dtype=torch.float16)
        self._check(_gemm, (a16, b16), a16 @ b16, [128, 256, 64])

    def test_bias_epilogue(self) -> None:
        a = torch.randn(1000, 1000, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(1000, 1000, device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(1000, device=DEVICE, dtype=torch.bfloat16)
        self._check(_bias_gemm, (a, b, bias), a @ b + bias, [128, 256, 64])

    def test_rows_below_the_tile(self) -> None:
        # Decode-shaped GEMMs: M below the 64-row tile is one zero-filled tile.
        for rows in (16, 2):
            a = torch.randn(rows, 4096, device=DEVICE, dtype=torch.bfloat16)
            b = torch.randn(4096, 2048, device=DEVICE, dtype=torch.bfloat16)
            self._check(_gemm, (a, b), a @ b, [64, 64, 64])

    def test_ragged_batched(self) -> None:
        a = torch.randn(4, 256, 1000, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(4, 1000, 257, device=DEVICE, dtype=torch.bfloat16)
        self._check(_bmm, (a, b), torch.bmm(a, b), [1, 128, 128, 64])
        a = torch.randn(2, 1000, 1000, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(2, 1000, 1000, device=DEVICE, dtype=torch.bfloat16)
        self._check(_bmm, (a, b), torch.bmm(a, b), [1, 128, 256, 64])

    def test_ragged_batched_unaligned_rows_take_the_scalar_producer(self) -> None:
        # Rows of 481 bf16 in a 484-element stride (968 bytes, not a 16-byte
        # multiple) fail the TMA proof and sit below the staging floor: the
        # flat pipeline keeps its one-warp scalar SMEM producer, through
        # eight K tiles with in-loop stages, and still matches torch.
        a = torch.randn(2, 70, 484, device=DEVICE, dtype=torch.bfloat16)[:, :, :481]
        b = torch.randn(2, 481, 48, device=DEVICE, dtype=torch.bfloat16)
        bound = _bmm.bind((a, b))
        config = _flat_config([1, 64, 32, 64])
        code = bound.to_code(config)
        self.assertNotIn("tma_tensor_a", code)
        self.assertIn("tcgen05_scalar_smem_mbar", code)
        bound.set_config(config)
        torch.testing.assert_close(bound(a, b), torch.bmm(a, b), rtol=1e-2, atol=1e-1)

    def test_ragged_batched_m_major_a_stays_off_the_search(self) -> None:
        # The batched planner admits K-major operands only; an M-major A
        # ([B, K, M] transposed) keeps the search off at full and ragged
        # shapes alike, so the flat ragged pipeline never meets the shared
        # scalar producer over partial batched tiles.
        for m in (1024, 1000):
            a = torch.randn(2, 1000, m, device=DEVICE, dtype=torch.bfloat16)
            b = torch.randn(2, 1000, 1000, device=DEVICE, dtype=torch.bfloat16)
            bound = _bmm.bind((a.transpose(1, 2), b))
            self.assertFalse(bound.config_spec.cute_tcgen05_search_enabled)
            with self.assertRaises(helion.exc.InvalidConfig):
                bound.to_code(_flat_config([1, 128, 256, 64]))


@onlyBackends(["cute"])
@skipIfNotCUDA()
class TestCuteTcgen05OneCtaNEdgeTmaStore(RefEagerTestDisabled, TestCase):
    def test_persistent_n_edge_tiles_store_through_tma(self) -> None:
        # One CTA, 128x256 tiles: N=576 ends 64 columns into its third tile and
        # 8192 rows give 192 tiles, so CTAs run more than one tile and some of
        # the partial ones. Every tile takes the TMA store (the D descriptor
        # drops the columns past N): a plain store, a bias row staged through
        # the predicated copy, and an exact-shape residual read behind the
        # full-tile guard.
        m, n, k = 8192, 576, 192
        a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(n, device=DEVICE, dtype=torch.bfloat16)
        res = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16)
        out = torch.empty(m, n, device=DEVICE, dtype=torch.bfloat16)
        acc = a.float() @ b.float()
        cases = (
            (plain_mm_into, (a, b, out), acc),
            (bias_mm_into, (a, b, bias, out), acc + bias.float()),
            (rank_two_aux, (a, b, res), acc + res.float()),
        )
        # the 128x128 tile is the decode default's (one row of M tiles)
        for (kernel, args, expected), bn in itertools.product(cases, (256, 128)):
            bound = kernel.bind(args)
            config = bound.config_spec.default_config()
            config.config.update(
                block_sizes=[128, bn, 64],
                pid_type="persistent_interleaved",
                tcgen05_cluster_m=1,
                tcgen05_ab_stages=3,
                tcgen05_aux_load_placement="post_acc_wait",
            )
            bound.config_spec.normalize(config, _fix_invalid=True)
            self.assertEqual(config.config["pid_type"], "persistent_interleaved")
            code = bound.to_code(config)
            self.assertIn("tcgen05_tma_store_atom", code)
            self.assertNotIn("cute.nvgpu.CopyR2GOp()", code)
            bound.set_config(config)
            out.zero_()
            result = bound(*args)
            torch.testing.assert_close(
                out if result is None else result,
                expected.to(out.dtype),
                rtol=1e-2,
                atol=1e-1,
            )

    def test_plain_column_major_output_defaults_to_the_n_edge_tiles(self) -> None:
        # A plain column-major output (no auxiliary operand) takes the
        # persistent one-CTA N-edge tiles by default and stores every tile
        # through TMA.
        m, n, k = 25216, 576, 192
        a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
        out = torch.empty(n, m, device=DEVICE, dtype=torch.bfloat16).T
        bound = plain_mm_into.bind((a, b, out))
        config = bound.config_spec.default_config()
        self.assertEqual(config.config["pid_type"], "persistent_interleaved")
        self.assertEqual(config.block_sizes, [128, 256, 64])
        self.assertIn("tcgen05_tma_store_atom", bound.to_code(config))
        bound.set_config(config)
        bound(a, b, out)
        torch.testing.assert_close(
            out, (a.float() @ b.float()).to(out.dtype), rtol=1e-2, atol=1e-1
        )

    def test_output_wider_than_n_keeps_its_columns(self) -> None:
        # The GEMM covers 576 of the output's 640 columns: N-edge tiles would
        # store zeros past N (edge stores clip at the destination's shape),
        # so the default keeps the flat whole-tile kernel and the columns
        # past N keep their sentinel.
        m, n, k = 25216, 576, 192
        a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
        out = torch.full((m, 640), 7.0, device=DEVICE, dtype=torch.bfloat16)
        bound = plain_mm_into.bind((a, b, out))
        self.assertFalse(bound.config_spec.cute_tcgen05_matmul_output_matches_extents)
        config = bound.config_spec.default_config()
        bound.set_config(config)
        bound(a, b, out)
        torch.testing.assert_close(
            out[:, :n], (a.float() @ b.float()).to(out.dtype), rtol=1e-2, atol=1e-1
        )
        self.assertTrue(bool((out[:, n:] == 7).all()))

    def test_column_major_output_bias_stays_off_the_n_edge_tiles(self) -> None:
        # A column-major output with a bias row: the partial-output body's
        # auxiliary reads do not take the accumulator's layout there, so the
        # default keeps the flat kernel on whole tiles instead of the
        # persistent one-CTA N-edge tiles, and matches torch.
        m, n, k = 25216, 576, 192
        a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(n, device=DEVICE, dtype=torch.bfloat16)
        out = torch.empty(n, m, device=DEVICE, dtype=torch.bfloat16).T
        bound = bias_mm_into.bind((a, b, bias, out))
        config = bound.config_spec.default_config()
        self.assertEqual(config.config["pid_type"], "flat")
        bound.set_config(config)
        bound(a, b, bias, out)
        torch.testing.assert_close(
            out,
            (a.float() @ b.float() + bias.float()).to(out.dtype),
            rtol=1e-2,
            atol=1e-1,
        )


@onlyBackends(["cute"])
@skipIfNotCUDA()
class TestCuteTcgen05RowLookahead(RefEagerTestDisabled, TestCase):
    def test_staged_rows_load_one_tile_ahead(self) -> None:
        # N=3000 on 256x256 pairs: 12 N tiles, the last ending 184 columns in;
        # 2048 rows give 96 tiles over 74 pairs (one or two tiles each), 4096
        # rows 192. Every tile takes the partial-output TMA store, whose
        # staged rows the epilogue role copies one tile ahead: one bias row,
        # two rows (one cp.async group per tile), and one row on the swizzled
        # walk (swizzle 8 pads the 12 N tiles to 16; at 4096 rows a pair
        # meets a padding slot before its next tile, which the lookahead
        # skips in step with the role).
        n, k = 3000, 128
        cases = (
            (bias_mm_into, 2048, 1),
            (scale_bias_mm_into, 2048, 1),
            (bias_mm_into, 4096, 8),
        )
        for kernel, m, swizzle in cases:
            a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
            b = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
            scale = torch.randn(n, device=DEVICE, dtype=torch.bfloat16)
            bias = torch.randn(n, device=DEVICE, dtype=torch.bfloat16)
            out = torch.empty(m, n, device=DEVICE, dtype=torch.bfloat16)
            acc = a.float() @ b.float()
            if kernel is scale_bias_mm_into:
                args = (a, b, scale, bias, out)
                expected = acc * scale.float() + bias.float()
            else:
                args = (a, b, bias, out)
                expected = acc + bias.float()
            bound = kernel.bind(args)
            config = bound.config_spec.default_config()
            config.config["tcgen05_l2_swizzle_size"] = swizzle
            bound.config_spec.normalize(config, _fix_invalid=True)
            self.assertEqual(config.config["pid_type"], "persistent_interleaved")
            self.assertEqual(config.config["tcgen05_l2_swizzle_size"], swizzle)
            code = bound.to_code(config)
            self.assertIn("row_lookahead_sched", code)
            self.assertIn("cute.arch.cp_async_wait_group(1)", code)
            bound.set_config(config)
            bound(*args)
            torch.testing.assert_close(
                out, expected.to(out.dtype), rtol=1e-2, atol=1e-1
            )

    def test_rows_off_their_vector_alignment_keep_the_synchronous_copy(self) -> None:
        # A bias view 2 or 8 bytes past a 16-byte boundary: the lookahead's
        # 128-bit cp.async copy needs the vector alignment the synchronous
        # copy does not, so the row keeps the per-tile copy.
        m, n, k = 2048, 3000, 128
        a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
        buf = torch.randn(n + 8, device=DEVICE, dtype=torch.bfloat16)
        for shift in (1, 4):
            bias = buf[shift : shift + n]
            out = torch.empty(m, n, device=DEVICE, dtype=torch.bfloat16)
            bound = bias_mm_into.bind((a, b, bias, out))
            config = bound.config_spec.default_config()
            code = bound.to_code(config)
            self.assertNotIn("cute.arch.cp_async_wait_group(1)", code)
            bound.set_config(config)
            bound(a, b, bias, out)
            torch.testing.assert_close(
                out,
                (a.float() @ b.float() + bias.float()).to(out.dtype),
                rtol=1e-2,
                atol=1e-1,
            )


@onlyBackends(["cute"])
@skipIfNotCUDA()
class TestCuteTcgen05RaggedTwoCtaSearch(RefEagerTestDisabled, TestCase):
    def test_ragged_gemm_admits_the_two_cta_edge_search(self) -> None:
        # Ragged GEMMs below the old 1024 floor keep the two-CTA edge family
        # in the search once its 128x128 tiles fill a quarter of the SMs;
        # the planner counts those tiles for 16-bit operands, not 256x256
        # pairs (16 of those at 1000^3).
        for m, n, k in ((1000, 1000, 1000), (777, 999, 555)):
            a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
            b = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
            bound = _gemm.bind((a, b))
            spec = bound.config_spec
            self.assertTrue(spec.cute_tcgen05_search_enabled)
            self.assertIn(2, spec._cute_tcgen05_config.cluster_m_search_choices)
            self.assertIn("persistent_interleaved", spec.allowed_pid_types)
            config = helion.Config(
                block_sizes=[128, 128, 64],
                indexing=["tensor_descriptor"] * 3,
                pid_type="persistent_interleaved",
                tcgen05_cta_group="two",
                tcgen05_cluster_m=2,
                tcgen05_cluster_n=1,
                tcgen05_ab_stages=5,
                tcgen05_acc_stages=2,
                tcgen05_c_stages=2,
                l2_groupings=[4],
            )
            code = bound.to_code(config)
            self.assertIn("CtaGroup.TWO", code)
            bound.set_config(config)
            torch.testing.assert_close(
                bound(a, b), (a.float() @ b.float()).to(a.dtype), rtol=1e-2, atol=1e-1
            )

    def test_the_wide_k_pair_seed_runs_a_k_tail(self) -> None:
        # The [128, 128, 128] ab6 twin of the plain edge seed on a shape whose
        # K (555) is not a multiple of the K tile.
        a = torch.randn(777, 555, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(555, 999, device=DEVICE, dtype=torch.bfloat16)
        bound = _gemm.bind((a, b))
        seeds = [
            seed
            for seed in bound.config_spec.compiler_seed_configs
            if seed.get("tcgen05_cta_group") == "two"
            and seed.get("block_sizes") == [128, 128, 128]
        ]
        self.assertEqual(len(seeds), 1)
        code = bound.to_code(seeds[0])
        self.assertIn("CtaGroup.TWO", code)
        bound.set_config(seeds[0])
        torch.testing.assert_close(
            bound(a, b), (a.float() @ b.float()).to(a.dtype), rtol=1e-2, atol=1e-1
        )

    def test_bias_epilogue_runs_the_256_row_pair_family(self) -> None:
        # Auxiliary epilogues are detected after planning, so a bf16 bias GEMM
        # at 1000^3 is admitted to the edge family on the 128x128 tile count
        # but runs the family's 256x256 pairs (the 128-row pair excludes aux
        # kernels): its cluster_m=2 seeds compile and match torch.
        a = torch.randn(1000, 1000, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(1000, 1000, device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(1000, device=DEVICE, dtype=torch.bfloat16)
        bound = _bias_gemm.bind((a, b, bias))
        spec = bound.config_spec
        self.assertTrue(spec.cute_tcgen05_aux_kernel_detected)
        self.assertIn(2, spec._cute_tcgen05_config.cluster_m_search_choices)
        seeds = [
            seed
            for seed in spec.compiler_seed_configs
            if seed.get("tcgen05_cluster_m") == 2
        ]
        self.assertTrue(seeds)
        expected = (a.float() @ b.float() + bias.float()).to(a.dtype)
        for seed in seeds:
            self.assertEqual(seed.get("block_sizes")[:2], [256, 256])
            bound.set_config(seed)
            torch.testing.assert_close(
                bound(a, b, bias), expected, rtol=1e-2, atol=1e-1
            )

    def test_k_tail_on_whole_tiles_runs_the_edge_family(self) -> None:
        # A 16-bit K tail on whole M and N tiles runs the edge + K-tail
        # family's pairs (TMA zero-fills past K; nothing is clipped): its
        # cluster_m=2 seeds compile and match torch.
        a = torch.randn(1024, 1000, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(1000, 1024, device=DEVICE, dtype=torch.bfloat16)
        bound = _gemm.bind((a, b))
        state = bound.config_spec._cute_tcgen05_config
        self.assertIn(2, state.cluster_m_search_choices)
        seeds = [
            seed
            for seed in bound.config_spec.compiler_seed_configs
            if seed.get("tcgen05_cluster_m") == 2
        ]
        self.assertTrue(seeds)
        expected = (a.float() @ b.float()).to(a.dtype)
        for seed in seeds:
            bound.set_config(seed)
            torch.testing.assert_close(bound(a, b), expected, rtol=1e-2, atol=1e-1)

    def test_output_edge_bias_stores_every_tile_through_tma(self) -> None:
        # Register auxiliary loads: the ragged N=3000 tiles take the TMA store
        # like the full tiles (the D descriptor drops the columns past N)
        # and the bias row stages through a predicated copy.
        a = torch.randn(2048, 768, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(768, 3000, device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(3000, device=DEVICE, dtype=torch.bfloat16)
        out = torch.empty(2048, 3000, device=DEVICE, dtype=torch.bfloat16)
        expected = (a.float() @ b.float() + bias.float()).to(a.dtype)
        bound = bias_mm_into.bind((a, b, bias, out))
        config = bound.config_spec.default_config()
        config.config.update(
            tcgen05_aux_load_placement="post_acc_wait", tcgen05_ab_stages=5
        )
        code = bound.to_code(config)
        self.assertIn("tcgen05_tma_store_atom", code)
        self.assertNotIn("cute.nvgpu.CopyR2GOp()", code)
        bound.set_config(config)
        out.zero_()
        bound(a, b, bias, out)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-1)

    def test_column_major_output_edge_bias_keeps_the_simt_edge_store(self) -> None:
        # A column-major output: the partial-output body's register auxiliary
        # reads do not match its accumulator layout there (CuTe: "profile of
        # input tuples doesn't match"), so the ragged N=3000 tiles keep the
        # SIMT edge store beside the full tiles' TMA store.
        a = torch.randn(2048, 768, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(768, 3000, device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(3000, device=DEVICE, dtype=torch.bfloat16)
        out = torch.empty(3000, 2048, device=DEVICE, dtype=torch.bfloat16).T
        expected = (a.float() @ b.float() + bias.float()).to(a.dtype)
        bound = bias_mm_into.bind((a, b, bias, out))
        config = bound.config_spec.default_config()
        config.config.update(
            block_sizes=[256, 256, 64],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_aux_load_placement="post_acc_wait",
            tcgen05_ab_stages=5,
        )
        code = bound.to_code(config)
        self.assertIn("cute.nvgpu.CopyR2GOp()", code)
        self.assertIn("tcgen05_tma_store_atom", code)
        bound.set_config(config)
        out.zero_()
        bound(a, b, bias, out)
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-1)

    def test_staged_row_on_the_hybrid_store_edge(self) -> None:
        # A ``pre_acc_wait`` row stage lives in the full-tile branch of the
        # hybrid TMA/SIMT store; the SIMT edge tiles (N=3000 is ragged) read
        # the row from global memory. Inductor's ``bias.to(float32) + acc``
        # stages promoted like the bare ``acc + bias``.
        a = torch.randn(2048, 128, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(128, 3000, device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(3000, device=DEVICE, dtype=torch.bfloat16)
        out = torch.empty(2048, 3000, device=DEVICE, dtype=torch.bfloat16)
        expected = (a.float() @ b.float() + bias.float()).to(a.dtype)
        for kernel in (promoted_bias_mm_into, bias_mm_into):
            bound = kernel.bind((a, b, bias, out))
            config = bound.config_spec.default_config()
            # ab5 leaves SMEM for the C ring and the row stage: the hybrid store
            config.config.update(
                tcgen05_aux_load_placement="pre_acc_wait", tcgen05_ab_stages=5
            )
            code = bound.to_code(config)
            self.assertIn("tcgen05_tma_store_atom", code)
            self.assertIn("tcgen05_aux_rowvec_smem_0", code)
            bound.set_config(config)
            out.zero_()
            bound(a, b, bias, out)
            torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-1)


if __name__ == "__main__":
    unittest.main()
