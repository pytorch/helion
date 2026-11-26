"""GPU tests of tcgen05 kernels over ragged tiles, and of the autotune-off
default tiles.

The flat one-CTA pipeline, the persistent one-CTA N-edge tiles, the staged
rows' one-tile lookahead and the ragged two-CTA search; autotune-off default
tiles where the B200 defaults promote (``TestCuteTcgen05DefaultTilesGpu``).

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
from unittest.mock import patch

import pytest
import torch

from test._cute_kernels import _gemm

import helion
from helion._compiler.cute.autotune.heuristics.cute_tcgen05_default import (
    CuteTcgen05DefaultTileHeuristic,
)
from helion._compiler.cute.dense_aligned_n import dense_aligned_n_defaults
from helion._hardware import get_hardware_info
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing.cute_aux import rank_two_aux
from helion._testing.cute_aux import two_residual_aux
from helion._testing.cute_dense_tcgen05 import bias_mm_into
from helion._testing.cute_dense_tcgen05 import column_scale_mm_into
from helion._testing.cute_dense_tcgen05 import plain_mm_into
from helion._testing.cute_dense_tcgen05 import promoted_bias_mm_into
from helion._testing.cute_dense_tcgen05 import scale_bias_mm_into
from helion._testing.cute_dense_tcgen05 import scaled_bmm_args
from helion._testing.cute_dense_tcgen05 import scaled_bmm_into
import helion.language as hl

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="tcgen05 requires an sm_100 GPU",
)


def _promotes_b200_defaults() -> bool:
    """The default-tile heuristic promotes on its targets only
    (``CuteTcgen05DefaultTileHeuristic.PROMOTE_TARGETS``), and the default
    tests pin wave quantization over a B200's 148 SMs."""
    if not torch.cuda.is_available():
        return False
    device = torch.device(DEVICE)
    hardware = get_hardware_info(device)
    return (
        hardware.device_kind,
        hardware.compute_capability,
    ) in CuteTcgen05DefaultTileHeuristic.PROMOTE_TARGETS and (
        torch.cuda.get_device_properties(device).multi_processor_count >= 148
    )


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
def _bmm_into(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    batch, m, k = a.size()
    _, _, n = b.size()
    for tile_b, tile_m, tile_n in hl.tile([batch, m, n]):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc, a[tile_b, tile_m, tile_k], b[tile_b, tile_k, tile_n]
            )
        out[tile_b, tile_m, tile_n] = acc.to(out.dtype)
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

    def test_ragged_batched_matrices_run_one_pair_each(self) -> None:
        # One persistent 256x256 CTA pair per ragged matrix (the planner's
        # batched_pair_matrix): TMA zero-fills past each matrix and the 3-D
        # TMA store clips at its edges, the batch being its own TensorMap
        # dim. The ragged N keeps CuTe's four (128, 64) rounds rather than
        # eight (128, 32) ones (``tcgen05_plain_narrow_subtile_covers_n``):
        # the rule counts one matrix's tiles (one here, not a wave's worth),
        # and one K block per tile leaves the epilogue the critical path.
        # 300 matrices give the pairs several tiles each; 192 x 176 has whole
        # 64-row heights in M and a ragged N.
        for batch, m, n in ((300, 200, 200), (6, 192, 176)):
            a = torch.randn(batch, m, 64, device=DEVICE, dtype=torch.bfloat16)
            b = torch.randn(batch, 64, n, device=DEVICE, dtype=torch.bfloat16)
            out = torch.full(
                (batch, m, n), float("nan"), device=DEVICE, dtype=torch.bfloat16
            )
            bound = _bmm_into.bind((a, b, out))
            config = bound.config_spec.default_config()
            config.config.update(
                block_sizes=[1, 256, 256, 64],
                pid_type="persistent_interleaved",
                tcgen05_persistence_model="static_persistent",
                tcgen05_cluster_m=2,
                tcgen05_ab_stages=4,
            )
            code = bound.to_code(config)
            self.assertIn("tcgen05_gC3d", code)
            self.assertIn("compute_epilogue_tile_shape((256, 256)", code)
            self.assertNotIn("_helion_tcgen05_persistent_total_tiles", code)
            bound.set_config(config)
            bound(a, b, out)
            torch.testing.assert_close(out, torch.bmm(a, b), rtol=1e-2, atol=1e-1)

    def test_ragged_batched_matrices_run_on_several_pairs(self) -> None:
        # Ragged matrices larger than one pair (the planner's
        # batched_pair_matrix past one pair tile): the persistent 256x256
        # pairs walk every matrix's 4 x 4 (1000 x 1000, B column-major) or
        # 3 x 3 (600 x 520, B row-major) tiles; the last row and column of
        # tiles in each matrix load zeros past its edges and the 3-D TMA store
        # clips them, so no tile writes into the next matrix.
        for batch, m, n, b_col_major in ((8, 1000, 1000, True), (4, 600, 520, False)):
            a = torch.randn(batch, m, 64, device=DEVICE, dtype=torch.bfloat16)
            if b_col_major:
                b = torch.randn(batch, n, 64, device=DEVICE, dtype=torch.bfloat16)
                b = b.transpose(1, 2)
            else:
                b = torch.randn(batch, 64, n, device=DEVICE, dtype=torch.bfloat16)
            out = torch.full(
                (batch, m, n), float("nan"), device=DEVICE, dtype=torch.bfloat16
            )
            bound = _bmm_into.bind((a, b, out))
            config = bound.config_spec.default_config()
            config.config.update(
                block_sizes=[1, 256, 256, 64],
                pid_type="persistent_interleaved",
                tcgen05_persistence_model="static_persistent",
                tcgen05_cluster_m=2,
                tcgen05_ab_stages=6,
                tcgen05_batch_raster="batch_slowest",
            )
            code = bound.to_code(config)
            self.assertIn("CtaGroup.TWO", code)
            self.assertIn("tcgen05_gC3d", code)
            self.assertNotIn("_helion_tcgen05_persistent_total_tiles", code)
            bound.set_config(config)
            bound(a, b, out)
            torch.testing.assert_close(out, torch.bmm(a, b), rtol=1e-2, atol=1e-1)

    def test_ragged_batched_pairs_run_k_tails_and_b_multicast(self) -> None:
        # The persistent pairs over a batch's K tail (on whole tiles, beside
        # an M edge, on a double edge) and under a tcgen05_cluster_n=2 B
        # multicast (a K tail on whole tiles, an M edge): TMA zero-fills K
        # past each matrix's last k-block and the rows and columns past its
        # edges, and the 3-D TMA store clips each matrix
        # (``_ragged_batched_pair_config``). The fp16 operands' rows (K = 128
        # or 136, N = 256, 304 or 512) are TensorMap-aligned. Outputs start as
        # NaN.
        for (batch, m, n, k), cluster_n in (
            ((2, 256, 512, 136), 2),
            ((2, 300, 512, 128), 2),
            ((2, 300, 256, 136), 1),
            ((2, 300, 304, 136), 1),
        ):
            a = torch.randn(batch, m, k, device=DEVICE, dtype=torch.float16)
            b = torch.randn(batch, k, n, device=DEVICE, dtype=torch.float16)
            out = torch.full(
                (batch, m, n), float("nan"), device=DEVICE, dtype=torch.float32
            )
            bound = _bmm_into.bind((a, b, out))
            config = helion.Config(
                block_sizes=[1, 256, 256, 64],
                l2_groupings=[1],
                pid_type="persistent_interleaved",
                tcgen05_persistence_model="static_persistent",
                tcgen05_cluster_m=2,
                tcgen05_cluster_n=cluster_n,
                tcgen05_ab_stages=2,
            )
            code = bound.to_code(config)
            self.assertIn("CtaGroup.TWO", code)
            self.assertIn("tcgen05_gC3d", code)
            self.assertNotIn("_helion_tcgen05_persistent_total_tiles", code)
            bound.set_config(config)
            bound(a, b, out)
            torch.testing.assert_close(
                out, torch.bmm(a.float(), b.float()), rtol=1e-2, atol=1e-1
            )

    def test_ragged_batched_pairs_split_full_and_fringe_tiles(self) -> None:
        # Attention scores over 300-row matrices on the pairs, with the
        # scheduler warp and auxiliary loads ahead of the accumulator wait:
        # the scheduler publishes each matrix's interior tiles to the TMA
        # store and its fringe tiles to the predicated SIMT store, by the
        # tile the consumers decode (the batch decoded out of the virtual
        # pid), plain or L2-grouped. The [300, 512] mask is read unguarded on
        # interior tiles only. Outputs start as NaN.
        for l2_grouping in (1, 2):
            (a, b, mask, out), _ = scaled_bmm_args(
                2, 300, 512, 128, row=512, device=DEVICE
            )
            out.fill_(float("nan"))
            bound = scaled_bmm_into.bind((a, b, mask, out))
            config = helion.Config(
                block_sizes=[1, 256, 256, 64],
                l2_groupings=[l2_grouping],
                pid_type="persistent_interleaved",
                tcgen05_persistence_model="static_persistent",
                tcgen05_cluster_m=2,
                tcgen05_ab_stages=2,
                tcgen05_strategy="role_local_with_scheduler",
                tcgen05_warp_spec_scheduler_warps=1,
                tcgen05_aux_load_placement="pre_acc_wait",
            )
            code = bound.to_code(config)
            self.assertIn("CtaGroup.TWO", code)
            self.assertIn("tcgen05_scheduler_warp_full_tile", code)
            bound.set_config(config)
            bound(a, b, mask, out)
            expected = torch.bmm(a.float(), b.float()) * 0.125 + mask.float()
            torch.testing.assert_close(
                out, expected.to(out.dtype), rtol=1e-2, atol=1e-1
            )

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

    def test_ragged_batched_column_major_operands_load_through_tma(self) -> None:
        # An M-major A ([B, K, M] transposed) and a K-major B ([B, N, K]
        # transposed) load through their (M, K, L) and (N, K, L) TensorMap
        # views, at full and ragged shapes alike, and match torch.
        for m in (1024, 1000):
            a = torch.randn(2, 1000, m, device=DEVICE, dtype=torch.bfloat16)
            b = torch.randn(2, 1000, 1000, device=DEVICE, dtype=torch.bfloat16)
            for lhs, rhs in ((a.transpose(1, 2), b), (a.transpose(1, 2), b.mT)):
                bound = _bmm.bind((lhs, rhs))
                self.assertTrue(bound.config_spec.tcgen05.search_enabled)
                config = _flat_config([1, 128, 256, 64])
                code = bound.to_code(config)
                self.assertIn("'a_k_major': False", code)
                self.assertEqual("'b_k_major': True" in code, rhs.stride(1) == 1)
                self.assertIn("tma_tensor_a", code)
                bound.set_config(config)
                torch.testing.assert_close(
                    bound(lhs, rhs), torch.bmm(lhs, rhs), rtol=1e-2, atol=1e-1
                )

    def test_ragged_batched_output_wider_than_n_keeps_its_columns(self) -> None:
        # Matrices 96 columns wide written over N = 80 (the flat edge store)
        # and over N = 64 (whole tiles, the TMA store): both stores stop at
        # the GEMM's N, so the columns past it keep their NaN, as eager
        # leaves them.
        a = torch.randn(4, 128, 128, device=DEVICE, dtype=torch.bfloat16)
        config = _flat_config([1, 128, 64, 64])
        for n in (80, 64):
            with self.subTest(n=n):
                b = torch.randn(4, 128, n, device=DEVICE, dtype=torch.bfloat16)
                out = torch.full(
                    (4, 128, 96), float("nan"), device=DEVICE, dtype=torch.bfloat16
                )
                bound = _bmm_into.bind((a, b, out))
                self.assertEqual(
                    "cute.copy(tcgen05_tma_store_atom" in bound.to_code(config),
                    n == 64,
                )
                bound.set_config(config)
                bound(a, b, out)
                torch.testing.assert_close(
                    out[..., :n],
                    torch.bmm(a.float(), b.float()).to(out.dtype),
                    rtol=1e-2,
                    atol=1e-1,
                )
                self.assertTrue(bool(out[..., n:].isnan().all()))


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

    def test_output_wider_than_n_keeps_its_columns_on_the_simt_store(self) -> None:
        # The flat tile's SIMT edge store bounds the GEMM's (M, N) too: a
        # 1001-wide output over N=1000 (2002-byte rows), and a 1008-wide one
        # over N=1001 with a K-major B, whose vector edge copy writes the
        # column past the last whole vector (1000) and nothing beyond.
        config = helion.Config(block_sizes=[128, 128, 64], l2_groupings=[1])
        for n, width, rhs_k_major in ((1000, 1001, False), (1001, 1008, True)):
            with self.subTest(n=n, width=width):
                a = torch.randn(1000, 512, device=DEVICE, dtype=torch.bfloat16)
                b = (
                    torch.randn(n, 512, device=DEVICE, dtype=torch.bfloat16).T
                    if rhs_k_major
                    else torch.randn(512, n, device=DEVICE, dtype=torch.bfloat16)
                )
                out = torch.full(
                    (1000, width), 7.0, device=DEVICE, dtype=torch.bfloat16
                )
                bound = plain_mm_into.bind((a, b, out))
                bound.set_config(config)
                bound(a, b, out)
                torch.testing.assert_close(
                    out[:, :n],
                    (a.float() @ b.float()).to(out.dtype),
                    rtol=1e-2,
                    atol=1e-1,
                )
                self.assertTrue(bool((out[:, n:] == 7).all()))

    def test_output_wider_than_n_keeps_its_columns(self) -> None:
        # The GEMM covers 576 of the output's 640 columns, row- or
        # column-major: the edge stores stop at N (the TMA store's TensorMap
        # views the GEMM's extents), so the columns past N keep their
        # sentinel on the default (25216 rows: ragged for the pairs), the
        # pair's partial-output store (25600 rows) and the persistent
        # one-CTA N-edge tiles.
        n, k = 576, 192
        pair = helion.Config(
            block_sizes=[256, 256, 64],
            l2_groupings=[1],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=2,
            tcgen05_ab_stages=4,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
        )
        one_cta = helion.Config(
            block_sizes=[128, 256, 64],
            l2_groupings=[1],
            pid_type="persistent_interleaved",
            tcgen05_cluster_m=1,
            tcgen05_ab_stages=3,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
        )
        cases = [
            (column_major, m, config)
            for column_major in (False, True)
            for m, config in ((25216, None), (25600, pair), (25600, one_cta))
        ]
        for column_major, m, config in cases:
            with self.subTest(column_major=column_major, m=m, config=config):
                a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
                b = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
                out = (
                    torch.full((640, m), 7.0, device=DEVICE, dtype=torch.bfloat16).T
                    if column_major
                    else torch.full((m, 640), 7.0, device=DEVICE, dtype=torch.bfloat16)
                )
                bound = plain_mm_into.bind((a, b, out))
                self.assertFalse(
                    bound.config_spec.tcgen05.matmul_output_matches_extents
                )
                bound.set_config(
                    config if config is not None else bound.config_spec.default_config()
                )
                bound(a, b, out)
                torch.testing.assert_close(
                    out[:, :n],
                    (a.float() @ b.float()).to(out.dtype),
                    rtol=1e-2,
                    atol=1e-1,
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
        # pairs (16 of those at 1000^3). 1000x1000x3000 runs the deepest
        # ring the pair's allocation proof admits, which the search
        # projection takes (13 stages; 46 whole K tiles and a 56-deep tail).
        for (m, n, k), ab_stages in (
            ((1000, 1000, 1000), 5),
            ((777, 999, 555), 5),
            ((1000, 1000, 3000), 13),
        ):
            a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
            b = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
            bound = _gemm.bind((a, b))
            spec = bound.config_spec
            self.assertTrue(spec.tcgen05.search_enabled)
            self.assertIn(2, spec._cute_tcgen05_config.cluster_m_search_choices)
            self.assertIn("persistent_interleaved", spec.allowed_pid_types)
            config = helion.Config(
                block_sizes=[128, 128, 64],
                indexing=["tensor_descriptor"] * 3,
                pid_type="persistent_interleaved",
                tcgen05_cta_group="two",
                tcgen05_cluster_m=2,
                tcgen05_cluster_n=1,
                tcgen05_ab_stages=ab_stages,
                tcgen05_acc_stages=2,
                tcgen05_c_stages=2,
                l2_groupings=[4],
            )
            code = bound.to_code(config)
            self.assertIn("CtaGroup.TWO", code)
            self.assertIn(f"num_stages={ab_stages}", code)
            bound.set_config(config)
            torch.testing.assert_close(
                bound(a, b), (a.float() @ b.float()).to(a.dtype), rtol=1e-2, atol=1e-1
            )

    def test_one_wave_grid_runs_the_128_row_pair(self) -> None:
        # A one-wave shape (1000 rows clipped, whole N and K tiles) runs the
        # 128-row pair at the 13-stage ring its seed takes.
        a = torch.randn(1000, 1024, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(1024, 1024, device=DEVICE, dtype=torch.bfloat16).T
        out = torch.empty(1000, 1024, device=DEVICE, dtype=torch.bfloat16)
        bound = plain_mm_into.bind((a, b, out))
        config = helion.Config(
            block_sizes=[128, 128, 64],
            pid_type="persistent_interleaved",
            tcgen05_cta_group="two",
            tcgen05_cluster_m=2,
            tcgen05_cluster_n=1,
            tcgen05_ab_stages=13,
            tcgen05_acc_stages=2,
            tcgen05_c_stages=2,
            l2_groupings=[1],
        )
        code = bound.to_code(config)
        self.assertIn("CtaGroup.TWO", code)
        self.assertIn("num_stages=13", code)
        bound.set_config(config)
        out.zero_()
        bound(a, b, out)
        torch.testing.assert_close(
            out, (a.float() @ b.float()).to(out.dtype), rtol=1e-2, atol=1e-1
        )

    @unittest.skipUnless(_promotes_b200_defaults(), "pins B200 default tiles")
    def test_one_wave_default_runs_the_64_wide_128_row_pair(self) -> None:
        # bf16 1500x384x384: 36 pairs of 128x128 would leave half the SMs
        # idle; the autotune-off default runs 72 pairs of 128x64 (144 CTAs,
        # one wave from 148 SMs up), the last M tile (1500 = 11 x 128 + 92
        # rows) clipped by the TMA store.
        a = torch.randn(1500, 384, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(384, 384, device=DEVICE, dtype=torch.bfloat16).T
        out = torch.empty(1500, 384, device=DEVICE, dtype=torch.bfloat16)
        bound = plain_mm_into.bind((a, b, out))
        config = bound.config_spec.default_config()
        self.assertEqual(config.config["block_sizes"][:3], [128, 64, 64])
        self.assertEqual(config.config["tcgen05_cta_group"], "two")
        self.assertIn("CtaGroup.TWO, (128, 64)", bound.to_code(config))
        bound.set_config(config)
        out.zero_()
        bound(a, b, out)
        torch.testing.assert_close(
            out, (a.float() @ b.float()).to(out.dtype), rtol=1e-2, atol=1e-1
        )

    @unittest.skipUnless(_promotes_b200_defaults(), "pins B200 default tiles")
    def test_single_edge_k_tail_defaults_run_the_128_row_pair(self) -> None:
        # One ragged output edge with a K tail (240 = 3 x 64 + 48, 1000 =
        # 15 x 64 + 40): the 128-row pair's producer issues the partial last
        # K tile as a whole box (zero-filled past K) and the TMA store clips
        # the edge, over a ragged N (8192x240), a ragged M (240x8192) and a
        # longer K loop (1280x1600x1000).
        for m, n, k in ((8192, 240, 240), (240, 8192, 240), (1280, 1600, 1000)):
            a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
            b = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16).T
            out = torch.empty(m, n, device=DEVICE, dtype=torch.bfloat16)
            bound = plain_mm_into.bind((a, b, out))
            config = bound.config_spec.default_config()
            self.assertEqual(config.config["tcgen05_cta_group"], "two")
            self.assertEqual(config.config["block_sizes"][:2], [128, 128])
            bound.set_config(config)
            out.zero_()
            bound(a, b, out)
            torch.testing.assert_close(
                out, (a.float() @ b.float()).to(out.dtype), rtol=1e-2, atol=1e-1
            )

    def test_single_edge_k_tail_runs_the_256_row_pair(self) -> None:
        # The same single ragged edge with a K tail on the 256-row pair the
        # edge family projects a sampled two-CTA tile to (each CTA 128 rows,
        # the edge clipped by the TMA store, the partial last K tile
        # zero-filled), plain and with a bias row (whose kernels the
        # 128-row pair does not run).
        pair = {
            "pid_type": "persistent_interleaved",
            "tcgen05_cluster_m": 2,
            "tcgen05_cluster_n": 1,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": 2,
            "l2_groupings": [1],
        }
        for m, n, k in ((8192, 240, 240), (240, 8192, 240)):
            a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
            b = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16).T
            bias = torch.randn(n, device=DEVICE, dtype=torch.bfloat16)
            expected = a.float() @ b.float()
            for kernel, args, reference, tile, stages in (
                (plain_mm_into, (a, b), expected, [256, 256, 64], 4),
                (
                    bias_mm_into,
                    (a, b, bias),
                    expected + bias.float(),
                    [256, 256, 128],
                    2,
                ),
            ):
                out = torch.zeros(m, n, device=DEVICE, dtype=torch.bfloat16)
                bound = kernel.bind((*args, out))
                config = helion.Config(
                    block_sizes=tile, tcgen05_ab_stages=stages, **pair
                )
                code = bound.to_code(config)
                self.assertNotIn("_helion_tcgen05_persistent_total_tiles", code)
                self.assertIn(f"< cutlass.Int32({k}))", code)
                bound.set_config(config)
                bound(*args, out)
                torch.testing.assert_close(
                    out, reference.to(out.dtype), rtol=1e-2, atol=1e-1
                )

    def test_aligned_width_runs_the_128_row_pair(self) -> None:
        # 16-bit aligned widths on 128-row CTA pairs (64 rows per CTA) over a
        # row-major output whose N one tile spans: N = 192 in one tile
        # (carrier 256) over whole 128-row M tiles and over a partial last one
        # (3104 = 24 x 128 + 32, clipped by the TMA store), and every other
        # width the family reaches there: 96 (N = 192 in two tiles), 160,
        # 176 and 208 (neither 32 nor 48 divides them: the per-round store,
        # no last-tile staging on the 128-row pair), 224 and 240 (whose CTA
        # stages 120 B rows per K step).
        for (m, n, k), width, stages in (
            ((3200, 192, 768), 192, 10),
            ((3104, 192, 768), 192, 10),
            ((1152, 192, 512), 96, 8),
            ((1152, 160, 512), 160, 8),
            ((1152, 176, 512), 176, 8),
            ((1152, 208, 512), 208, 8),
            ((1152, 224, 512), 224, 8),
            ((1152, 240, 512), 240, 8),
        ):
            a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
            b = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16).T
            out = torch.empty(m, n, device=DEVICE, dtype=torch.bfloat16)
            bound = plain_mm_into.bind((a, b, out))
            config = helion.Config(
                **dense_aligned_n_defaults(),
                tcgen05_tile_n=width,
                block_sizes=[128, 256 if width > 128 else 128, 64],
                pid_type="persistent_interleaved",
                tcgen05_ab_stages=stages,
                tcgen05_c_stages=2,
            )
            self.assertIn(f"CtaGroup.TWO, (128, {width})", bound.to_code(config))
            bound.set_config(config)
            out.zero_()
            bound(a, b, out)
            torch.testing.assert_close(
                out, (a.float() @ b.float()).to(out.dtype), rtol=1e-2, atol=1e-1
            )

    def test_output_edge_pairs_clip_the_partial_tiles(self) -> None:
        # The 16-bit pairs over a partial last M tile (1000 rows: four 256-row
        # pair tiles, the last clipped by the TMA store): the one-wave 256x128
        # pair, a 256x112 pair (56 B rows per CTA, an odd count of 8-row
        # swizzle atoms: the stage-0 B box is the SMEM layout's ``select``),
        # the 256x160 pair over whole N tiles (2880 = 18 x 160) and, ragged
        # in N as well, the 256x144 pair (2560 = 17.8 x 144); the 256x144 and
        # (over whole M tiles, 2000 = 8.3 x 240) 256x240 pairs store (128, 48)
        # subtiles. The widths 32 does not divide otherwise (112, 176, 208)
        # stage their CTA's last tile in the A ring its MMAs drained, the
        # earlier ones in the C ring: 76 256x112 tiles give most CTAs one
        # (staged) tile, 150 256x176 tiles most CTAs one of each, and 80
        # 256x208 tiles (partial in N) a few CTAs two. The 256x256 pair over
        # whole K tiles stores its clipped tiles through (128, 32) subtiles,
        # as whole tiles do, over a partial last N tile too where its CTAs run
        # more than one tile over a K loop past the tile's fixed cost (8192 x
        # 1000 = 32 x 3.9 tiles over eight K blocks), and so does the 256x192
        # pair (1000 x 4000 = 4 x 20.8) at the seven AB stages that ring
        # leaves room for: rounds wholly past N store nothing. The output
        # starts as NaN, so a tile left unstored fails the check.
        cases = (
            ((1000, 2304, 256), [256, 128, 64], 0, "auto", 4),
            ((1000, 2048, 256), [256, 256, 64], 0, "two", 4),
            ((8192, 1000, 512), [256, 256, 64], 0, "two", 4),
            ((1000, 4000, 512), [256, 256, 64], 192, "two", 7),
            ((1000, 2048, 256), [256, 128, 64], 112, "two", 4),
            ((1000, 2880, 256), [256, 256, 64], 160, "two", 4),
            ((1000, 2560, 256), [256, 256, 64], 144, "two", 4),
            ((2048, 2000, 256), [256, 256, 64], 240, "two", 4),
            ((6272, 1024, 512), [256, 256, 64], 176, "two", 8),
            ((2048, 2000, 256), [256, 256, 64], 208, "two", 4),
        )
        for (m, n, k), block_sizes, width, cta_group, ab_stages in cases:
            a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
            b = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16).T
            out = torch.full((m, n), float("nan"), device=DEVICE, dtype=torch.bfloat16)
            bound = plain_mm_into.bind((a, b, out))
            config = helion.Config(
                block_sizes=block_sizes,
                pid_type="persistent_interleaved",
                tcgen05_cta_group=cta_group,
                tcgen05_cluster_m=2,
                tcgen05_cluster_n=1,
                tcgen05_ab_stages=ab_stages,
                tcgen05_acc_stages=2,
                tcgen05_c_stages=2,
                tcgen05_l2_swizzle_size=1,
                l2_groupings=[1],
                **({"tcgen05_tile_n": width} if width else {}),
            )
            code = bound.to_code(config)
            physical = width or block_sizes[1]
            self.assertIn(f"CtaGroup.TWO, (256, {physical})", code)
            if physical // 2 % 16:
                self.assertIn("sB_tma_layout = cute.select(sB_layout", code)
            if physical in (144, 192, 240, 256):
                self.assertIn(
                    "tcgen05_store_epi_tile = (cute.make_layout(128), "
                    f"cute.make_layout({48 if physical % 32 else 32}))",
                    code,
                )
            self.assertEqual(
                "if not tcgen05_epi_next_work.is_valid_tile:" in code,
                physical in (112, 176, 208),
            )
            bound.set_config(config)
            bound(a, b, out)
            torch.testing.assert_close(
                out, (a.float() @ b.float()).to(out.dtype), rtol=1e-2, atol=1e-1
            )

    def test_column_major_outputs_clip_the_partial_tiles(self) -> None:
        # The same pairs storing a column-major output: the transposed product
        # out.T = b.T @ a.T of a tall GEMM, whose pairs run 256 rows of its N
        # and a width of its M. 224 over 2000 columns (8.9 tiles) clips the
        # last tile, as 1000 rows of N clip the last 256-row tile; 176 and
        # 192 divide 2112 and 3072. A per-column FP32 scale (one per row of
        # ``out``) over the clipped 224-wide tiles takes the split store (TMA
        # for whole tiles, SIMT for the last), and an FP32 output its own
        # with-source C tile. The 224-wide pair also runs seven AB stages
        # (231 596 of the 232 448-byte opt-in, ``pipeline_smem_bytes``'
        # ``plain_layout``).
        cases = (
            ((2000, 1024, 256), 224, "plain"),
            ((2000, 1024, 512), 224, "seven"),
            ((1792, 1000, 256), 224, "plain"),
            ((2112, 1024, 256), 176, "plain"),
            ((3072, 512, 256), 192, "plain"),
            ((2000, 1024, 256), 224, "scale"),
            ((2112, 1024, 256), 176, "fp32"),
        )
        for (m, n, k), width, kind in cases:
            a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
            b = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16).T
            out_dtype = torch.float32 if kind == "fp32" else torch.bfloat16
            out = torch.empty(m, n, device=DEVICE, dtype=out_dtype)
            expected = a.float() @ b.float()
            if kind == "scale":
                scale = torch.randn(m, device=DEVICE, dtype=torch.float32)
                expected = expected * scale[:, None]
                args = (b.T, a.T, scale, out.T)
                bound = column_scale_mm_into.bind(args)
            else:
                args = (b.T, a.T, out.T)
                bound = plain_mm_into.bind(args)
            config = helion.Config(
                block_sizes=[256, 256, 64],
                pid_type="persistent_interleaved",
                tcgen05_cta_group="two",
                tcgen05_cluster_m=2,
                tcgen05_cluster_n=1,
                tcgen05_ab_stages=7 if kind == "seven" else 4,
                tcgen05_acc_stages=2,
                tcgen05_c_stages=2,
                tcgen05_l2_swizzle_size=1,
                l2_groupings=[1],
                tcgen05_tile_n=width,
            )
            code = bound.to_code(config)
            self.assertIn(f"CtaGroup.TWO, (256, {width})", code)
            self.assertIn("LayoutEnum.COL_MAJOR", code)
            if kind == "seven":
                self.assertIn("num_stages=7", code)
            bound.set_config(config)
            out.zero_()
            bound(*args)
            torch.testing.assert_close(
                out, expected.to(out.dtype), rtol=2e-2, atol=1e-1
            )

    @unittest.skipUnless(_promotes_b200_defaults(), "pins B200 default tiles")
    def test_double_edge_defaults_run_the_pairs(self) -> None:
        # Ragged in both M and N with whole K tiles and an M no one-CTA tile
        # height divides: the autotune-off default is a 256-row pair whose
        # TMA store clips both edges; plain, with a bias row, and into a
        # column-major output.
        m, n, k = 3000, 3000, 3072
        a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16).T
        bias = torch.randn(n, device=DEVICE, dtype=torch.bfloat16)
        acc = a.float() @ b.float()
        row_major = torch.empty(m, n, device=DEVICE, dtype=torch.bfloat16)
        column_major = torch.empty(n, m, device=DEVICE, dtype=torch.bfloat16).T
        cases = (
            (plain_mm_into, (a, b, row_major), acc),
            (bias_mm_into, (a, b, bias, row_major), acc + bias.float()),
            (plain_mm_into, (a, b, column_major), acc),
        )
        for kernel, args, expected in cases:
            bound = kernel.bind(args)
            config = bound.config_spec.default_config()
            self.assertEqual(config.config["tcgen05_cluster_m"], 2)
            self.assertIn("CtaGroup.TWO", bound.to_code(config))
            bound.set_config(config)
            args[-1].zero_()
            bound(*args)
            torch.testing.assert_close(
                args[-1], expected.to(torch.bfloat16), rtol=1e-2, atol=1e-1
            )

    @unittest.skipUnless(_promotes_b200_defaults(), "pins B200 default tiles")
    def test_narrow_n_m_edge_default_runs_the_256xn_pair(self) -> None:
        # One 128-wide N tile over an M edge (19000 = 74 pairs + 56 rows),
        # M-major A, FP32 residual: the default 256x128 pair matches torch.
        # The default is the 148-SM one (149 128x128 tiles fill the grid of
        # a device with more SMs, which takes the small-grid pair).
        m, n, k = 19000, 128, 256
        a = torch.randn(k, m, device=DEVICE, dtype=torch.bfloat16).T
        b = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
        residual = torch.randn(m, n, device=DEVICE, dtype=torch.float32)
        with patch(
            "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
            return_value=148,
        ):
            bound = rank_two_aux.bind((a, b, residual))
            config = bound.config_spec.default_config()
        self.assertEqual(config.block_sizes, [256, 128, 64])
        self.assertEqual(config.config["tcgen05_cluster_m"], 2)
        self.assertIn("CtaGroup.TWO", bound.to_code(config))
        bound.set_config(config)
        torch.testing.assert_close(
            bound(a, b, residual),
            a.float() @ b.float() + residual,
            rtol=1e-3,
            atol=1e-2,
        )

    def test_two_ab_load_warps_match_torch(self) -> None:
        # ``tcgen05_warp_spec_ab_load_warps=2``: warps 5 and 6 walk one rolled
        # K loop and issue alternate k-blocks into the shared AB ring, on the
        # autotune-off default pairs: one tile per CTA over a partial last M
        # tile with an odd K-block count (1000x512x2880, 128x64 pair, 45
        # k-blocks: the warps' shares differ), the 13-stage 128x128 pair
        # (1000x1024x2048), a K tail the last box zero-fills (8192x240x240,
        # 128x128 pair) and several tiles per CTA (4096x4096x448, 256 tiles
        # of 256x256 on 76 SM pairs: the ring state walks on into the next
        # tile's 7 k-blocks); and one tile per CTA over whole tiles
        # (1024x1024x2048: the TMA-load role hoisted ahead of the init
        # rendezvous, which both A/B-load warps join).
        for m, n, k in (
            (1000, 512, 2880),
            (1000, 1024, 2048),
            (8192, 240, 240),
            (4096, 4096, 448),
            (1024, 1024, 2048),
        ):
            a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
            b = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16).T
            out = torch.zeros(m, n, device=DEVICE, dtype=torch.bfloat16)
            bound = plain_mm_into.bind((a, b, out))
            values = dict(bound.config_spec.default_config().config)
            self.assertEqual(values["tcgen05_cluster_m"], 2)
            values["tcgen05_warp_spec_ab_load_warps"] = 2
            # (the flat 8-warp role layout keeps one A/B-load warp)
            values["tcgen05_flat_role_coordinates"] = False
            config = helion.Config(**values)
            code = bound.to_code(config)
            self.assertIn("block=(32, 7, 1)", code)
            self.assertIn("% cutlass.Int32(2) == tcgen05_ab_load_rank:", code)
            bound.set_config(config)
            bound(a, b, out)
            torch.testing.assert_close(
                out, (a.float() @ b.float()).to(out.dtype), rtol=1e-2, atol=1e-1
            )
        # The aligned width 112 (1000x2048x2304, 256x112 pair: an odd count
        # of 8-row swizzle atoms per CTA, the stage-0 ``select`` B box).
        a = torch.randn(1000, 2304, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(2048, 2304, device=DEVICE, dtype=torch.bfloat16).T
        out = torch.zeros(1000, 2048, device=DEVICE, dtype=torch.bfloat16)
        bound = plain_mm_into.bind((a, b, out))
        config = helion.Config(
            **{
                **dense_aligned_n_defaults(),
                "tcgen05_tile_n": 112,
                "block_sizes": [256, 128, 64],
                "pid_type": "persistent_interleaved",
                "tcgen05_ab_stages": 9,
                "tcgen05_c_stages": 2,
                "tcgen05_warp_spec_ab_load_warps": 2,
            }
        )
        self.assertIn(
            "% cutlass.Int32(2) == tcgen05_ab_load_rank:", bound.to_code(config)
        )
        bound.set_config(config)
        bound(a, b, out)
        torch.testing.assert_close(
            out, (a.float() @ b.float()).to(out.dtype), rtol=1e-2, atol=1e-1
        )

    @unittest.skipUnless(_promotes_b200_defaults(), "pins B200 default tiles")
    def test_seventeen_stage_pair_ring_matches_torch(self) -> None:
        # The 128x64 bf16 pair's ring fills shared memory at 17 stages
        # (``TCGEN05_PAIRED_MAX_AB_STAGES``): the autotune-off default of
        # 1000x512x2880 (45 k-blocks, a partial last M tile), with one and two
        # A/B-load warps (an odd ring: a warp's slots change every lap), and
        # an M-major A / N-major B product (768x768x2048, x.T @ y).
        a = torch.randn(1000, 2880, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(512, 2880, device=DEVICE, dtype=torch.bfloat16).T
        out = torch.zeros(1000, 512, device=DEVICE, dtype=torch.bfloat16)
        bound = plain_mm_into.bind((a, b, out))
        values = dict(bound.config_spec.default_config().config)
        self.assertEqual(values["block_sizes"][:3], [128, 64, 64])
        self.assertEqual(values["tcgen05_ab_stages"], 17)
        for warps in (1, 2):
            config = helion.Config(
                **{**values, "tcgen05_warp_spec_ab_load_warps": warps}
            )
            self.assertIn("num_stages=17", bound.to_code(config))
            out.zero_()
            bound.set_config(config)
            bound(a, b, out)
            torch.testing.assert_close(
                out, (a.float() @ b.float()).to(out.dtype), rtol=1e-2, atol=1e-1
            )
        x = torch.randn(2048, 768, device=DEVICE, dtype=torch.bfloat16).T
        y = torch.randn(2048, 768, device=DEVICE, dtype=torch.bfloat16)
        out = torch.zeros(768, 768, device=DEVICE, dtype=torch.bfloat16)
        bound = plain_mm_into.bind((x, y, out))
        config = helion.Config(
            **{
                **dict(bound.config_spec.default_config().config),
                "block_sizes": [128, 64, 64],
                "tcgen05_cluster_m": 2,
                "tcgen05_cta_group": "two",
                "pid_type": "persistent_interleaved",
                "tcgen05_ab_stages": 17,
                "tcgen05_c_stages": 2,
            }
        )
        self.assertIn("num_stages=17", bound.to_code(config))
        bound.set_config(config)
        bound(x, y, out)
        torch.testing.assert_close(
            out, (x.float() @ y.float()).to(out.dtype), rtol=1e-2, atol=1e-1
        )

    @unittest.skipUnless(_promotes_b200_defaults(), "pins B200 default tiles")
    def test_l2_warm_up_from_idle_sms_matches_torch(self) -> None:
        # The grid's first eight CTAs warm the first k-blocks of every A and
        # B row into L2 from the SMs the grid leaves idle
        # (``_build_pdl_l2_warmup``): the autotune-off defaults of bf16
        # 1000x1024x1280 (64 pairs of 128x128) and of 1000x512x1024 reading
        # the first one's output (64 pairs of 128x64; at 512 its footprint
        # would take the one-tile prefetch instead), each leaving 20 or more
        # of the at least 148 SMs idle. One CUDA graph replays three such
        # pairs over their own inputs and buffers: each dependent launch
        # starts its first CTAs while the previous kernel still runs, so each
        # second kernel prefetches its ``mid`` while the first one is still
        # writing it (NaN-filled before the replay), and the loads after its
        # dependent-launch wait must read the written values.
        pairs = 3
        a = [
            torch.randn(1000, 1280, device=DEVICE, dtype=torch.bfloat16)
            for _ in range(pairs)
        ]
        b = torch.randn(1024, 1280, device=DEVICE, dtype=torch.bfloat16).T
        c = torch.randn(512, 1024, device=DEVICE, dtype=torch.bfloat16).T
        mid = [
            torch.empty(1000, 1024, device=DEVICE, dtype=torch.bfloat16)
            for _ in range(pairs)
        ]
        out = [
            torch.empty(1000, 512, device=DEVICE, dtype=torch.bfloat16)
            for _ in range(pairs)
        ]
        first = plain_mm_into.bind((a[0], b, mid[0]))
        second = plain_mm_into.bind((mid[0], c, out[0]))
        for bound in (first, second):
            config = bound.config_spec.default_config()
            self.assertIn(
                "tcgen05_l2_warm_cta < cutlass.Int32(8)", bound.to_code(config)
            )
            bound.set_config(config)
        # compile and load both outside the capture
        first(a[0], b, mid[0])
        second(mid[0], c, out[0])
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(pairs):
                first(a[i], b, mid[i])
                second(mid[i], c, out[i])
        for i in range(pairs):
            mid[i].fill_(float("nan"))
            out[i].fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        for i in range(pairs):
            expected_mid = (a[i].float() @ b.float()).to(torch.bfloat16)
            torch.testing.assert_close(mid[i], expected_mid, rtol=1e-2, atol=1e-1)
            # against the ``mid`` the first kernel wrote (a stale or NaN read
            # of it would show as whole wrong tiles), not torch's rounding of
            # it, which the second product would amplify past the tolerance
            torch.testing.assert_close(
                out[i],
                (mid[i].float() @ c.float()).to(torch.bfloat16),
                rtol=1e-2,
                atol=1e-1,
            )

    @unittest.skipUnless(_promotes_b200_defaults(), "pins B200 default tiles")
    def test_multi_wave_l2_warm_up_matches_torch(self) -> None:
        # A persistent grid with more tiles than CTA pairs warms from the SMs
        # its previous call's last wave leaves idle
        # (``_tcgen05_last_wave_idle_sms``): bf16 2048x3072x768 on the 256x256
        # pair (96 tiles over 74-76 pairs, so each CTA's release arrive
        # stays at its first tile), two calls back to back in one CUDA graph
        # over NaN-filled outputs.
        a = torch.randn(2048, 768, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(3072, 768, device=DEVICE, dtype=torch.bfloat16).T
        outs = [
            torch.empty(2048, 3072, device=DEVICE, dtype=torch.bfloat16)
            for _ in range(2)
        ]
        bound = plain_mm_into.bind((a, b, outs[0]))
        config = helion.Config(
            **{
                **dict(bound.config_spec.default_config().config),
                "block_sizes": [256, 256, 64],
                "tcgen05_tile_n": 0,
                "tcgen05_cluster_m": 2,
                "tcgen05_cta_group": "two",
                "pid_type": "persistent_interleaved",
                "tcgen05_ab_stages": 6,
            }
        )
        code = bound.to_code(config)
        self.assertIn("_BLOCK_SIZE_1 = 256", code)
        self.assertIn("tcgen05_l2_warm_cta < cutlass.Int32(8)", code)
        self.assertIn("advance_to_next_work()", code)
        bound.set_config(config)
        bound(a, b, outs[0])
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for out in outs:
                bound(a, b, out)
        for out in outs:
            out.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        expected = (a.float() @ b.float()).to(torch.bfloat16)
        for out in outs:
            torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-1)

    def test_one_tile_operand_l2_prefetch_matches_torch(self) -> None:
        # A one-tile CTA whose A and B rows hold at most 768 lines prefetches
        # them into L2 ahead of its dependent-launch wait: 1500x384x384 (a
        # partial last M tile) and 768x768x384 (whole tiles, the TMA-load role
        # hoisted), with one and two A/B-load warps, and over padded rows.
        for (m, n, k), warps, pad in (
            *itertools.product(((1500, 384, 384), (768, 768, 384)), (1, 2), (0,)),
            # padded K-major rows: the prefetch reads each operand's stride
            ((1500, 384, 384), 2, 128),
        ):
            a = torch.randn(m, k + pad, device=DEVICE, dtype=torch.bfloat16)[:, :k]
            b = torch.randn(n, k + pad, device=DEVICE, dtype=torch.bfloat16)[:, :k].T
            out = torch.zeros(m, n, device=DEVICE, dtype=torch.bfloat16)
            bound = plain_mm_into.bind((a, b, out))
            values = dict(bound.config_spec.default_config().config)
            values["tcgen05_warp_spec_ab_load_warps"] = warps
            config = helion.Config(**values)
            # (the one-tile prefetch itself, not the L2 warm-up)
            self.assertIn("tcgen05_l2_pf_", bound.to_code(config))
            bound.set_config(config)
            bound(a, b, out)
            torch.testing.assert_close(
                out, (a.float() @ b.float()).to(out.dtype), rtol=1e-2, atol=1e-1
            )

    def test_mn_major_one_tile_operand_l2_prefetch_matches_torch(self) -> None:
        # The same prefetch along K for an M-major A and an N-major B (a
        # weight gradient): 1000x768x128 (a partial last M tile; 2000-byte A
        # columns, two lines each, the last one clamped to row 999) with one
        # and two A/B-load warps, and 1024x768x128 (line-aligned columns) on
        # the 128x128 pair.
        for m, warps in ((1000, 1), (1000, 2), (1024, 1)):
            a = torch.randn(128, m, device=DEVICE, dtype=torch.bfloat16).T
            b = torch.randn(128, 768, device=DEVICE, dtype=torch.bfloat16)
            out = torch.zeros(m, 768, device=DEVICE, dtype=torch.bfloat16)
            bound = plain_mm_into.bind((a, b, out))
            values = dict(bound.config_spec.default_config().config)
            values.update(
                block_sizes=[128, 128, 64],
                tcgen05_cluster_m=2,
                tcgen05_cta_group="two",
                pid_type="persistent_interleaved",
                tcgen05_ab_stages=2,
                tcgen05_warp_spec_ab_load_warps=warps,
            )
            config = helion.Config(**values)
            # (the one-tile prefetch itself, not the L2 warm-up)
            self.assertIn("tcgen05_l2_pf_", bound.to_code(config))
            bound.set_config(config)
            bound(a, b, out)
            torch.testing.assert_close(
                out, (a.float() @ b.float()).to(out.dtype), rtol=1e-2, atol=1e-1
            )

    def test_transposed_product_stages_the_swapped_pairs_last_tile(self) -> None:
        # helion.inductor's transposed product of a bf16 RC mm, out.T (n x m,
        # column-major) = b.T @ a.T, on the aligned-N 128x112 pair (each CTA
        # stages its 64-row C tile in the drained A ring): its autotune-off
        # default over 1000x1024x2048 (72 pairs, one tile each) with one and
        # two A/B-load warps, and 2000x1024x2048 (144 tiles on 76 SM pairs:
        # the ring stores every tile but each CTA's last); and the 256x112
        # pair over 1000x2048x2304 (72 pairs, a 128-row C tile per CTA).
        pair = {
            **dense_aligned_n_defaults(),
            "tcgen05_tile_n": 112,
            "pid_type": "persistent_interleaved",
            "tcgen05_c_stages": 2,
        }
        pair128 = {**pair, "block_sizes": [128, 128, 64], "tcgen05_ab_stages": 14}
        pair256 = {**pair, "block_sizes": [256, 128, 64], "tcgen05_ab_stages": 9}
        for (m, n, k), warps, explicit in (
            ((1000, 1024, 2048), 1, None),
            ((1000, 1024, 2048), 2, None),
            ((2000, 1024, 2048), 2, pair128),
            ((1000, 2048, 2304), 1, pair256),
        ):
            a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
            b = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16).T
            out = torch.zeros(m, n, device=DEVICE, dtype=torch.bfloat16)
            args = (b.T, a.T, out.T)
            bound = plain_mm_into.bind(args)
            if explicit is not None:
                values = dict(explicit)
            else:
                values = dict(bound.config_spec.default_config().config)
                self.assertEqual(values["tcgen05_tile_n"], 112)
                self.assertEqual(values["block_sizes"][0], 128)
            values["tcgen05_warp_spec_ab_load_warps"] = warps
            config = helion.Config(**values)
            self.assertIn(
                "if not tcgen05_epi_next_work.is_valid_tile:", bound.to_code(config)
            )
            bound.set_config(config)
            bound(*args)
            torch.testing.assert_close(
                out, (a.float() @ b.float()).to(out.dtype), rtol=1e-2, atol=1e-1
            )

    def test_a_single_n_edge_default_runs_the_aligned_128_row_pair(self) -> None:
        # A row-major product with one N edge over whole 256-row M tiles takes
        # the one-wave search and its aligned 128-row pair as the autotune-off
        # default (4608x224x1024: 72 pairs of 128x112, N 224 = 2 x 112, whose
        # 144 CTAs fit one wave on a B200's 148 SMs as on GB300's 152; the 152
        # CTAs of 4864 rows take the 224-wide pair on a B200).
        a = torch.randn(4608, 1024, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(224, 1024, device=DEVICE, dtype=torch.bfloat16).T
        out = torch.zeros(4608, 224, device=DEVICE, dtype=torch.bfloat16)
        bound = plain_mm_into.bind((a, b, out))
        config = bound.config_spec.default_config()
        self.assertEqual(config.config["tcgen05_tile_n"], 112)
        self.assertEqual(config.config["block_sizes"][0], 128)
        bound.set_config(config)
        bound(a, b, out)
        torch.testing.assert_close(
            out, (a.float() @ b.float()).to(out.dtype), rtol=1e-2, atol=1e-1
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
        self.assertTrue(spec.tcgen05.aux_kernel_detected)
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

    def test_k_tail_defaults_above_the_sm_gate_match_torch(self) -> None:
        # The edge + K-tail default on grids that fill the machine: the
        # 128-deep two-stage pair (bf16 3000x3000x2880, FP8 3000x3000x3008)
        # and the 64-deep deep-ring pair (bf16 2048x2560x8008).
        @helion.kernel(static_shapes=True)
        def fp8_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            m, k = a.size()
            _, n = b.size()
            out = torch.empty([m, n], dtype=torch.bfloat16, device=a.device)
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = hl.dot(a[tile_m, tile_k], b[tile_k, tile_n], acc=acc)
                out[tile_m, tile_n] = acc.to(torch.bfloat16)
            return out

        cases = (
            (_gemm, (3000, 3000, 2880), torch.bfloat16, 128),
            (_gemm, (2048, 2560, 8008), torch.bfloat16, 64),
            (fp8_gemm, (3000, 3000, 3008), torch.float8_e4m3fn, 128),
        )
        for kernel, (m, n, k), dtype, bk in cases:
            with self.subTest(shape=(m, n, k), dtype=dtype):
                a = (torch.randn(m, k, device=DEVICE) * 0.5).to(dtype)
                b = (torch.randn(n, k, device=DEVICE) * 0.5).to(dtype).T
                bound = kernel.bind((a, b))
                config = bound.config_spec.default_config()
                self.assertEqual(config.block_sizes, [256, 256, bk])
                self.assertEqual(config.config["tcgen05_cluster_m"], 2)
                bound.set_config(config)
                torch.testing.assert_close(
                    bound(a, b).float(), a.float() @ b.float(), rtol=3e-2, atol=1.0
                )

    def test_k_tail_on_whole_tiles_runs_the_edge_family(self) -> None:
        # A 16-bit K tail on whole M and N tiles runs the edge + K-tail
        # family's pairs (TMA zero-fills past K; nothing is clipped): its
        # plain 256x256x64 pair seed and its 128-row pair seed compile and
        # match torch.
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
        chosen = [
            next(
                seed for seed in seeds if seed.get("block_sizes")[:3] == [256, 256, 64]
            ),
            next(seed for seed in seeds if seed.get("block_sizes")[0] == 128),
        ]
        expected = (a.float() @ b.float()).to(a.dtype)
        for seed in chosen:
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


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fp8_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    batch, m, k = a.shape
    n = b.shape[2]
    out = torch.empty((batch, m, n), dtype=torch.bfloat16, device=a.device)
    for tile_b, tile_m, tile_n in hl.tile((batch, m, n)):
        acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.baddbmm(
                acc, a[tile_b, tile_m, tile_k], b[tile_b, tile_k, tile_n]
            )
        out[tile_b, tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


@onlyBackends(["cute"])
@skipIfNotCUDA()
class TestCuteTcgen05SkinnyTransposedGpu(RefEagerTestDisabled, TestCase):
    """helion.inductor's transposed skinny products: out.T = b.T @ a.T, an
    N-contiguous output, the operands as views of a and b."""

    def test_small_n_runs_one_zero_filled_tile(self) -> None:
        # A transposed GEMV (N = 4): the default's 8-column tcgen05 tile,
        # whose edge TMA zero-fills a.T's rows 4-7, over a flat and a
        # split-K grid.
        m, n, k = 4, 4096, 1024
        a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16).T
        # the result row-major, so that out.T is Inductor's (1, n) buffer
        out = torch.empty(m, n, device=DEVICE, dtype=torch.bfloat16)
        bound = plain_mm_into.bind((b.T, a.T, out.T))
        default = bound.config_spec.default_config()
        self.assertEqual(default.config["block_sizes"][1], 8)
        for split_k in (1, 4):
            config = helion.Config(**{**default.config, "tcgen05_split_k": split_k})
            bound.set_config(config)
            out.fill_(float("nan"))
            bound(b.T, a.T, out.T)
            torch.testing.assert_close(
                out, (a.float() @ b.float()).to(out.dtype), rtol=1e-2, atol=1e-1
            )

    def test_persistent_one_cta_k_tail(self) -> None:
        # A K tail over whole output tiles (K = 1000) on persistent one-CTA
        # tiles: the role-local producer issues the partial last K tile,
        # which TMA zero-fills past K; an N-major b (b.T MN-major) and a
        # K-major one, a 128- and a 256-deep K tile, SIMT and TMA stores, a
        # plain store and Inductor's addmm epilogue (a bias over the
        # kernel's N, the result's rows here).
        m, n, k = 128, 768, 1000
        a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(m, device=DEVICE, dtype=torch.bfloat16)
        for b in (
            torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16),
            torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16).T,
        ):
            # the result row-major, so that out.T is Inductor's (1, n) buffer
            out = torch.empty(m, n, device=DEVICE, dtype=torch.bfloat16)
            acc = a.float() @ b.float()
            for kernel, args, expected in (
                (plain_mm_into, (b.T, a.T), acc),
                (bias_mm_into, (b.T, a.T, bias), acc + bias.float()[:, None]),
            ):
                bound = kernel.bind((*args, out.T))
                for bk, stages, store in ((128, 8, "direct"), (256, 4, "normal")):
                    config = helion.Config(
                        **{
                            **bound.config_spec.default_config().config,
                            "block_sizes": [64, 16, bk],
                            "pid_type": "persistent_interleaved",
                            "tcgen05_persistence_model": "static_persistent",
                            "tcgen05_ab_stages": stages,
                            "tcgen05_c_store_mode": store,
                        }
                    )
                    code = bound.to_code(config)
                    self.assertNotIn("_helion_tcgen05_persistent_total_tiles", code)
                    bound.set_config(config)
                    out.fill_(float("nan"))
                    bound(*args, out.T)
                    torch.testing.assert_close(
                        out, expected.to(out.dtype), rtol=1e-2, atol=1e-1
                    )

    def test_flat_one_cta_tiles_prefetch_their_footprint(self) -> None:
        # bf16 128x192x1000 with b stored [K, N], transposed onto flat
        # [64, 8, 128] one-CTA tiles (48 CTAs): the epilogue warps prefetch
        # each CTA's b.T lines and its 8 rows of a (2000 bytes apart, so a
        # row starts 0, 80, 32, ... bytes into a line and may reach a 17th
        # line) into L2 ahead of the dependent-launch wait. With 132 rows of a, the last tile's rows
        # past them prefetch nothing.
        for m in (128, 132):
            n, k = 192, 1000
            a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
            b = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
            out = torch.empty(m, n, device=DEVICE, dtype=torch.bfloat16)
            bound = plain_mm_into.bind((b.T, a.T, out.T))
            config = helion.Config(
                **{
                    **bound.config_spec.default_config().config,
                    "block_sizes": [64, 8, 128],
                    "pid_type": "flat",
                    "tcgen05_ab_stages": 8,
                    "cute_pdl": True,
                }
            )
            code = bound.to_code(config)
            self.assertIn("cache_level='L2'", code)
            bound.set_config(config)
            out.fill_(float("nan"))
            bound(b.T, a.T, out.T)
            torch.testing.assert_close(
                out, (a.float() @ b.float()).to(out.dtype), rtol=1e-2, atol=1e-1
            )


@onlyBackends(["cute"])
@skipIfNotCUDA()
@unittest.skipUnless(_promotes_b200_defaults(), "pins B200 default tiles")
class TestCuteTcgen05DefaultTilesGpu(RefEagerTestDisabled, TestCase):
    def setUp(self) -> None:
        super().setUp()
        # the defaults' wave model counts B200's 148 SMs; a device with
        # more would pick other tiles
        patcher = patch(
            "helion._compiler.cute.tcgen05.search_plan._cuda_num_sms_or_zero",
            return_value=148,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_direct_entry_bias_default_stages_the_row_and_matches_torch(
        self,
    ) -> None:
        # Whole 256x256 tiles of a 16-bit bias GEMM: the direct-entry pair's
        # default stages the promoted bias row per tile (``pre_acc_wait``).
        m, n, k = 8192, 2304, 768
        a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(n, device=DEVICE, dtype=torch.bfloat16)
        expected = (a.float() @ b.float() + bias.float()).to(torch.bfloat16)
        for kernel in (bias_mm_into, promoted_bias_mm_into):
            with self.subTest(kernel=kernel.name):
                out = torch.empty(m, n, device=DEVICE, dtype=torch.bfloat16)
                bound = kernel.bind((a, b, bias, out))
                config = bound.config_spec.default_config()
                self.assertTrue(config.config["tcgen05_tvm_ffi_launch"])
                self.assertEqual(
                    config.config["tcgen05_aux_load_placement"], "pre_acc_wait"
                )
                bound.set_config(config)
                bound(a, b, bias, out)
                torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-1)

    def test_fp8_bmm_defaults_match_torch(self) -> None:
        # FP8 bmms with a K-major B take tcgen05 at their defaults, a K tail
        # (K = 48 on 32-deep k-blocks) included.
        for k in (48, 64):
            with self.subTest(k=k):
                a = torch.randn(64, 512, k, device=DEVICE).to(torch.float8_e4m3fn)
                b = (
                    torch.randn(64, 64, k, device=DEVICE)
                    .to(torch.float8_e4m3fn)
                    .transpose(1, 2)
                )
                bound = _fp8_bmm.bind((a, b))
                bound.set_config(bound.config_spec.default_config())
                torch.testing.assert_close(
                    bound(a, b),
                    torch.bmm(a.float(), b.float()).to(torch.bfloat16),
                    rtol=1e-2,
                    atol=1e-1,
                )

    def test_batched_m_major_a_defaults_match_torch(self) -> None:
        # A batched M-major A stages through TMA, so its defaults are the
        # pipeline's tiles, not the flat base tile.
        for batch, m, n, k in ((8, 512, 64, 512), (2, 1024, 1024, 512)):
            a = torch.randn(batch, k, m, device=DEVICE, dtype=torch.bfloat16)
            b = torch.randn(batch, k, n, device=DEVICE, dtype=torch.bfloat16)
            a = a.transpose(1, 2)
            bound = _bmm.bind((a, b))
            config = bound.config_spec.default_config()
            self.assertNotEqual(config.config["pid_type"], "flat")
            bound.set_config(config)
            torch.testing.assert_close(
                bound(a, b),
                torch.bmm(a.float(), b.float()).to(torch.bfloat16),
                rtol=1e-2,
                atol=1e-1,
            )

    def test_small_grid_default_runs_the_128_row_pair(self) -> None:
        # 36 128x128 tiles: the default is the 128-row pair with the most
        # CTAs in one wave (on 148 SMs, 72 128x64 pairs: 144 CTAs).
        m, n, k = 128, 4608, 1024
        a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
        bias = torch.randn(n, device=DEVICE, dtype=torch.bfloat16)
        out = torch.empty(m, n, device=DEVICE, dtype=torch.bfloat16)
        bound = bias_mm_into.bind((a, b, bias, out))
        config = bound.config_spec.default_config()
        self.assertEqual(config.block_sizes[0], 128)
        self.assertEqual(config.config["tcgen05_cluster_m"], 2)
        sms = torch.cuda.get_device_properties(DEVICE).multi_processor_count
        self.assertLessEqual(2 * -(-n // config.block_sizes[1]), sms)
        bound.set_config(config)
        bound(a, b, bias, out)
        torch.testing.assert_close(
            out,
            (a.float() @ b.float() + bias.float()).to(out.dtype),
            rtol=1e-2,
            atol=1e-1,
        )

    def test_small_grid_column_major_output_loads_its_row_after_the_wait(
        self,
    ) -> None:
        # A column-major output reads its bias row per subtile: the small
        # grid's pair loads it after the accumulator wait (the whole-row load
        # before it did not compile beside the column-major store).
        for m, n, k in ((128, 128, 128), (256, 2048, 512), (128, 4096, 1024)):
            a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
            b = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
            bias = torch.randn(n, device=DEVICE, dtype=torch.bfloat16)
            out = torch.empty(n, m, device=DEVICE, dtype=torch.bfloat16).T
            bound = bias_mm_into.bind((a, b, bias, out))
            config = bound.config_spec.default_config()
            self.assertEqual(config.config["tcgen05_cluster_m"], 2)
            self.assertEqual(
                config.config["tcgen05_aux_load_placement"], "post_acc_wait"
            )
            bound.set_config(config)
            bound(a, b, bias, out)
            torch.testing.assert_close(
                out,
                (a.float() @ b.float() + bias.float()).to(out.dtype),
                rtol=1e-2,
                atol=1e-1,
            )

    def test_small_grid_without_a_pair_splits_k(self) -> None:
        # A 64-row M and a K tail over long K loops: the flat 64x64x64 tile
        # splits K (four slices, eight) and the last arriving CTA adds the
        # bias once; a K-major B into 65-column rows (no TensorMap) runs the
        # tile unsplit over its eight k-blocks.
        for m, n, k, split in (
            (64, 4096, 4096, 4),
            (32, 1280, 4000, 8),
            (96, 65, 512, 1),
        ):
            a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
            b = (
                torch.randn(n, k, device=DEVICE, dtype=torch.bfloat16).T
                if n % 8
                else torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
            )
            bias = torch.randn(n, device=DEVICE, dtype=torch.bfloat16)
            out = torch.empty(m, n, device=DEVICE, dtype=torch.bfloat16)
            bound = bias_mm_into.bind((a, b, bias, out))
            config = bound.config_spec.default_config()
            self.assertEqual(config.config["tcgen05_split_k"], split)
            bound.set_config(config)
            bound(a, b, bias, out)
            torch.testing.assert_close(
                out,
                (a.float() @ b.float() + bias.float()).to(out.dtype),
                rtol=1e-2,
                atol=1e-1,
            )

    def test_unaligned_rows_store_through_the_staged_tile(self) -> None:
        # Attention scores into 201-element rows: the staged SMEM tile stores
        # every tile, the 200-row / 200-column edges masked, over one k-block
        # (one AB stage) and two, matching torch.
        for k in (64, 128):
            (a, b, mask, out), padded = scaled_bmm_args(
                32, 200, 200, k, row=201, device=DEVICE
            )
            bound = scaled_bmm_into.bind((a, b, mask, out))
            config = bound.config_spec.default_config()
            self.assertIn("tcgen05_staged_tile", bound.to_code(config))
            padded.fill_(7.0)
            bound.set_config(config)
            bound(a, b, mask, out)
            expected = torch.bmm(a.float(), b.float()) * 0.125 + mask.float()
            torch.testing.assert_close(
                out, expected.to(out.dtype), rtol=1e-2, atol=1e-1
            )
            # the padding column past N stays untouched
            self.assertTrue(bool((padded[..., 200:] == 7.0).all()))

    def test_batched_defaults_walk_batches_slowest(self) -> None:
        # The batched pair defaults and the narrow persistent one-CTA tile
        # walk their batches slowest and match torch: the 256x256 direct
        # entry, a promoted 256x128 pair seed, and the narrow tile over eight
        # k-blocks, 64x256 (one M tile per batch) and 128x64 (four).
        for batch, m, n, k in (
            (64, 512, 1024, 64),
            (75, 256, 256, 64),
            (192, 64, 512, 512),
            (64, 512, 64, 512),
        ):
            a = torch.randn(batch, m, k, device=DEVICE, dtype=torch.bfloat16)
            b = torch.randn(batch, k, n, device=DEVICE, dtype=torch.bfloat16)
            bound = _bmm.bind((a, b))
            config = bound.config_spec.default_config()
            self.assertEqual(config.config["tcgen05_batch_raster"], "batch_slowest")
            bound.set_config(config)
            torch.testing.assert_close(
                bound(a, b),
                torch.bmm(a.float(), b.float()).to(torch.bfloat16),
                rtol=1e-2,
                atol=1e-1,
            )

    def test_many_small_batched_matrices_match_torch(self) -> None:
        # Window attention's batches of matrices inside one tile: one
        # zero-filled one-CTA tile each, whatever TMA can address (98-byte
        # rows of 49 columns, a K-major B, an M-major A, a K tail, rows past
        # one tile).
        for batch, m, n, k, a_m_major, b_k_major in (
            (256, 56, 56, 32, False, False),
            (256, 49, 49, 32, False, True),
            (256, 32, 49, 49, True, False),
            (256, 200, 64, 200, False, False),
        ):
            a = (
                torch.randn(batch, k, m, device=DEVICE, dtype=torch.bfloat16).transpose(
                    1, 2
                )
                if a_m_major
                else torch.randn(batch, m, k, device=DEVICE, dtype=torch.bfloat16)
            )
            b = (
                torch.randn(batch, n, k, device=DEVICE, dtype=torch.bfloat16).transpose(
                    1, 2
                )
                if b_k_major
                else torch.randn(batch, k, n, device=DEVICE, dtype=torch.bfloat16)
            )
            bound = _bmm.bind((a, b))
            config = bound.config_spec.default_config()
            self.assertEqual(config.config["tcgen05_cluster_m"], 1)
            bound.set_config(config)
            torch.testing.assert_close(
                bound(a, b),
                torch.bmm(a.float(), b.float()).to(torch.bfloat16),
                rtol=1e-2,
                atol=1e-1,
            )

    def test_unaligned_batched_output_matches_torch(self) -> None:
        # Scores stored into 1001-element rows: the one-CTA 64x128 tile's
        # SIMT stores clip each matrix's last row and column tiles.
        a = torch.randn(8, 1000, 64, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(8, 64, 1000, device=DEVICE, dtype=torch.bfloat16)
        out = torch.zeros(8, 1000, 1001, device=DEVICE, dtype=torch.bfloat16)
        view = out[:, :, :1000]
        bound = _bmm_into.bind((a, b, view))
        config = bound.config_spec.default_config()
        self.assertEqual(config.block_sizes, [1, 64, 128, 64])
        bound.set_config(config)
        bound(a, b, view)
        torch.testing.assert_close(
            view,
            torch.bmm(a.float(), b.float()).to(torch.bfloat16),
            rtol=1e-2,
            atol=1e-1,
        )
        self.assertEqual(out[:, :, 1000].abs().max().item(), 0)

    def test_residual_pair_defaults_stage_the_residual(self) -> None:
        # Whole tiles: the epilogue-issued TMA ring, four stages deep where
        # the AB ring holds the tile's four k-blocks beside it. An M edge
        # one-CTA 128-row tiles do not cover (1000 rows): the residual's TMA
        # producer warp on the persistent 256x256 pair.
        for m, n, k, mode, aux_stages in (
            (4096, 1024, 1024, "epi_tma", None),
            (4096, 1024, 256, "epi_tma", 4),
            (1000, 4096, 1024, "tma", None),
        ):
            a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
            b = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
            residual = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16)
            bound = rank_two_aux.bind((a, b, residual))
            config = bound.config_spec.default_config()
            self.assertEqual(config.config["tcgen05_cluster_m"], 2)
            self.assertEqual(config.config["tcgen05_aux_load_mode"], mode)
            self.assertEqual(config.config.get("tcgen05_aux_stages"), aux_stages)
            bound.set_config(config)
            torch.testing.assert_close(
                bound(a, b, residual),
                (a.float() @ b.float() + residual.float()).to(residual.dtype),
                rtol=1e-2,
                atol=1e-1,
            )

    def test_m_edge_default_fits_two_residual_rings(self) -> None:
        # Over an M edge the producer pair's default prices both residuals'
        # rings: eight 32-deep AB stages beside them and two C stages compile
        # (with four C stages eight overflowed: 262 KB of 232) and match torch.
        m, n, k = 1000, 4096, 1088
        a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
        r0 = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16)
        r1 = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16)
        bound = two_residual_aux.bind((a, b, r0, r1))
        config = bound.config_spec.default_config()
        self.assertEqual(config.config["tcgen05_aux_load_mode"], "tma")
        self.assertEqual(config.config["tcgen05_ab_stages"], 8)
        self.assertEqual(config.config["tcgen05_c_stages"], 2)
        bound.set_config(config)
        torch.testing.assert_close(
            bound(a, b, r0, r1),
            (a.float() @ b.float() + r0.float() + r1.float()).to(r0.dtype),
            rtol=1e-2,
            atol=1e-1,
        )

    def test_aux_producer_shallow_rings_match_torch(self) -> None:
        # The residual's TMA producer warp beside five 64-deep AB stages and
        # two C stages over an M edge (the 32-deep default's ten stages run
        # in ``test_residual_pair_defaults_stage_the_residual``).
        m, n, k = 1000, 4096, 1024
        a = torch.randn(m, k, device=DEVICE, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=DEVICE, dtype=torch.bfloat16)
        residual = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16)
        bound = rank_two_aux.bind((a, b, residual))
        config = bound.config_spec.default_config()
        config.config.update(block_sizes=[256, 256, 64], tcgen05_ab_stages=4)
        bound.config_spec.normalize(config, _fix_invalid=True)
        self.assertEqual(config.config["tcgen05_ab_stages"], 5)
        self.assertEqual(config.config["tcgen05_aux_load_mode"], "tma")
        bound.set_config(config)
        torch.testing.assert_close(
            bound(a, b, residual),
            (a.float() @ b.float() + residual.float()).to(residual.dtype),
            rtol=1e-2,
            atol=1e-1,
        )


if __name__ == "__main__":
    unittest.main()
