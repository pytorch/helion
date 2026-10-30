"""GPU tests of the flat one-CTA tcgen05 pipeline over ragged tiles.

Partial M / N output tiles and a K tail used to take the one-warp scalar
SMEM producer and store 16-bit scalars; the flat pipeline now loads every
box through TMA (zero-filled past the extents) and stores interior tiles as
vectors and fringe tiles chunk-predicated. These tests pin flat tcgen05
configs over odd extents for the plain, transposed, bias, fp16 and batched
GEMMs and check the results against torch.
"""

from __future__ import annotations

import unittest

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
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


if __name__ == "__main__":
    unittest.main()
