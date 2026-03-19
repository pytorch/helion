from __future__ import annotations

import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


# Module-level kernel shared by degradation tests (Pattern 1 / Pattern 2)
@helion.kernel()
def jagged_dense_bmm(
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    L, D = jagged.shape
    B, D, K = dense.shape
    dtype = torch.promote_types(jagged.dtype, dense.dtype)
    device = jagged.device

    jagged = jagged.view(-1)
    output = torch.empty((L, K), dtype=dtype, device=device).view(-1)
    for tile_b in hl.tile(B):
        starts = seq_offsets[tile_b]
        ends = seq_offsets[tile_b.index + 1]
        seq_len = ends - starts

        for tile_len in hl.jagged_tile(seq_len):
            jagged_indices = starts[:, None] + tile_len.index[None, :]

            for tile_k in hl.tile(0, K):
                acc = hl.zeros([tile_b, tile_len, tile_k], dtype=dtype, device=device)
                for tile_d in hl.tile(0, D):
                    jagged_data = hl.load(
                        jagged,
                        [jagged_indices[:, :, None] * D + tile_d.index[None, None, :]],
                    )
                    dense_data = dense[tile_b, tile_d, tile_k]

                    acc = acc + torch.matmul(jagged_data, dense_data)

                if bias is not None:
                    bias_data = bias[tile_b, tile_k]
                    acc = acc + bias_data.unsqueeze(1)

                hl.store(
                    output,
                    [jagged_indices[:, :, None] * K + tile_k.index[None, None, :]],
                    acc,
                )
    return output.reshape(L, K)


# Module-level persistent kernel (Pattern 3)
@helion.kernel(static_shapes=False)
def grouped_gemm_jagged_persistent(
    A_packed: torch.Tensor,
    B: torch.Tensor,
    group_offsets: torch.Tensor,
) -> torch.Tensor:
    device = A_packed.device
    if device.type == "xpu":
        num_workers = torch.xpu.get_device_properties(device.index).gpu_subslice_count
    else:
        num_workers = torch.cuda.get_device_properties(
            device.index
        ).multi_processor_count

    BLOCK_M = hl.register_block_size(32, 128)
    BLOCK_N = hl.register_block_size(32, 128)
    total_M, K = A_packed.shape
    K2, N = B.shape
    assert K == K2

    out = torch.zeros(
        total_M,
        N,
        dtype=torch.promote_types(A_packed.dtype, B.dtype),
        device=A_packed.device,
    )

    G = group_offsets.size(0) - 1

    for worker_id in hl.grid(num_workers):
        for g in hl.grid(G):
            group_start = group_offsets[g]
            group_end = group_offsets[g + 1]
            m_size = group_end - group_start

            if m_size > 0:
                num_m_tiles = (m_size + BLOCK_M - 1) // BLOCK_M
                num_n_tiles = (N + BLOCK_N - 1) // BLOCK_N
                num_group_tiles = num_m_tiles * num_n_tiles

                for local_tile in hl.grid(num_group_tiles):
                    tile_in_group = local_tile * num_workers + worker_id
                    if tile_in_group < num_group_tiles:
                        # pyrefly: ignore[unsupported-operation]
                        m_tile_idx = tile_in_group % num_m_tiles
                        n_tile_idx = tile_in_group // num_m_tiles

                        base_row = group_start + m_tile_idx * BLOCK_M
                        # pyrefly: ignore[unsupported-operation]
                        base_col = n_tile_idx * BLOCK_N

                        row_idx = base_row + hl.arange(BLOCK_M)
                        col_idx = base_col + hl.arange(BLOCK_N)

                        rows_valid = row_idx < group_end
                        cols_valid = col_idx < N

                        acc = hl.zeros([BLOCK_M, BLOCK_N], dtype=torch.float32)

                        for k_tile in hl.tile(K):
                            k_idx = k_tile.index

                            a_blk = hl.load(
                                A_packed,
                                [row_idx, k_idx],
                                extra_mask=rows_valid[  # pyrefly: ignore[bad-index]
                                    :, None
                                ],
                            )
                            b_blk = hl.load(
                                B,
                                [k_idx, col_idx],
                                extra_mask=cols_valid[
                                    None, :
                                ],  # pyrefly: ignore[bad-index]
                            )

                            acc = torch.addmm(acc, a_blk, b_blk)

                        # pyrefly: ignore[bad-index]
                        valid_2d = rows_valid[:, None] & cols_valid[None, :]
                        hl.store(
                            out,
                            [row_idx, col_idx],
                            acc.to(out.dtype),
                            extra_mask=valid_2d,
                        )

    return out


def _ref_jagged_dense_bmm(seq_offsets, jagged, dense, bias):
    L, D = jagged.shape
    B = dense.size(0)
    K = dense.size(2)
    out = torch.empty((L, K), dtype=jagged.dtype, device=jagged.device)
    for i in range(B):
        s = int(seq_offsets[i].item())
        e = int(seq_offsets[i + 1].item())
        if s < e:
            result = torch.matmul(jagged[s:e], dense[i])
            if bias is not None:
                result = result + bias[i].unsqueeze(0)
            out[s:e] = result
    return out


def _ref_grouped_gemm(A_packed, B, group_offsets):
    G = group_offsets.numel() - 1
    outs = []
    for g in range(G):
        s = int(group_offsets[g].item())
        e = int(group_offsets[g + 1].item())
        outs.append(A_packed[s:e] @ B)
    return torch.cat(outs, dim=0)


@onlyBackends(["triton"])
class TestJaggedTilePatterns(RefEagerTestDisabled, TestCase):
    def _make_jagged_bmm_data(self, D=32, K=32):
        """Create deterministic jagged BMM test data with B=3."""
        torch.manual_seed(42)
        seq_offsets = torch.tensor([0, 2, 5, 7], device=DEVICE, dtype=torch.int64)
        L = int(seq_offsets[-1].item())
        B = 3
        jagged = torch.randn(L, D, device=DEVICE, dtype=torch.float32)
        dense = torch.randn(B, D, K, device=DEVICE, dtype=torch.float32)
        bias = torch.randn(B, K, device=DEVICE, dtype=torch.float32)
        return seq_offsets, jagged, dense, bias

    def test_jagged_bmm_outer_blocksize_1(self):
        """Pattern 1 degrades to Pattern 2 when outer block_size=1.

        With tile_b=1, each program instance processes exactly one batch element,
        so the outer batch mask (mask_0) is eliminated.
        """
        seq_offsets, jagged, dense, bias = self._make_jagged_bmm_data()
        code, result = code_and_output(
            jagged_dense_bmm,
            (seq_offsets, jagged, dense, bias),
            block_sizes=[1, 32, 32, 32],
        )
        # Outer mask eliminated since block_size=1 for batch dimension
        self.assertNotIn("mask_0", code)
        # Jagged mask still present for variable-length sequences
        self.assertIn("mask_1", code)
        # Correctness
        ref = _ref_jagged_dense_bmm(seq_offsets, jagged, dense, bias)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)
        self.assertExpectedJournal(code)

    def test_jagged_bmm_batched(self):
        """Pattern 1 baseline: tile_b=4 > B=3 requires outer batch mask."""
        seq_offsets, jagged, dense, bias = self._make_jagged_bmm_data()
        code, result = code_and_output(
            jagged_dense_bmm,
            (seq_offsets, jagged, dense, bias),
            block_sizes=[4, 32, 32, 32],
        )
        # Outer mask IS present since tile_b=4 > B=3
        self.assertIn("mask_0", code)
        # Jagged mask present for variable-length sequences
        self.assertIn("tl.where", code)
        # Correctness
        ref = _ref_jagged_dense_bmm(seq_offsets, jagged, dense, bias)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)
        self.assertExpectedJournal(code)

    def test_grouped_gemm_persistent(self):
        """Pattern 3: persistent kernel with dynamic tile assignment."""
        torch.manual_seed(0)
        G = 4
        K, N = 256, 128
        dtype = torch.bfloat16
        group_A = [
            torch.randn(64 * (i + 1), K, device=DEVICE, dtype=dtype).contiguous()
            for i in range(G)
        ]
        B_shared = torch.randn(K, N, device=DEVICE, dtype=dtype).contiguous()

        # Build packed inputs
        M_sizes = [a.size(0) for a in group_A]
        offsets = [0]
        for m in M_sizes:
            offsets.append(offsets[-1] + m)
        group_offsets = torch.tensor(offsets, device=DEVICE, dtype=torch.int32)
        A_packed = torch.cat(group_A, dim=0).contiguous()

        code, result = code_and_output(
            grouped_gemm_jagged_persistent,
            (A_packed, B_shared, group_offsets),
        )
        # Persistent worker loop present
        self.assertIn("num_workers", code)
        # Boundary masks for partial tiles
        self.assertIn("extra_mask", code)
        # Correctness (bfloat16 tolerance)
        ref = _ref_grouped_gemm(A_packed, B_shared, group_offsets)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)
        self.assertExpectedJournal(code)

    def test_jagged_bmm_persistent_jagged(self):
        """persistent_jagged: flatten (B, M, K_out) tiles into one persistent loop."""
        seq_offsets, jagged, dense, bias = self._make_jagged_bmm_data()
        code, result = code_and_output(
            jagged_dense_bmm,
            (seq_offsets, jagged, dense, bias),
            block_sizes=[1, 32, 32, 32],
            pid_type="persistent_jagged",
        )
        # Persistent loop present
        self.assertIn("_TOTAL_TILES", code)
        self.assertIn("_group_tile_prefix", code)
        # No inner M/N for-loops (they're flattened into the persistent loop)
        # But K reduction loop is preserved
        # Correctness
        ref = _ref_jagged_dense_bmm(seq_offsets, jagged, dense, bias)
        torch.testing.assert_close(result, ref, rtol=1e-2, atol=1e-2)
        self.assertExpectedJournal(code)


if __name__ == "__main__":
    unittest.main()
