"""Helion FP8 rowwise grouped GEMM example."""

from __future__ import annotations

from typing import Callable, Tuple

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=False)
def fp8_gemm_rowwise_grouped(
    A_fp8: torch.Tensor,  # [total_M, K] in FP8
    B_fp8: torch.Tensor,  # [total_N, K] in FP8 (rowwise weights)
    m_sizes: torch.Tensor,  # [G] rows per group
    a_scale: torch.Tensor,  # [total_M] row dequant scales for A
    b_scale: torch.Tensor,  # [total_N] row dequant scales for B (per column of output)
    use_fast_accum: bool = True,
) -> torch.Tensor:
    """Compute grouped GEMM with rowwise FP8 dequantization."""

    total_M, K = A_fp8.shape
    total_N, K2 = B_fp8.shape
    assert K == K2

    G = m_sizes.size(0)
    assert G > 0
    assert total_N % G == 0
    N = total_N // G

    device = A_fp8.device
    num_workers = torch.cuda.get_device_properties(device).multi_processor_count  # type: ignore[arg-type]

    block_m = hl.register_block_size(64, 128)
    block_n = hl.register_block_size(64, 128)
    block_k = hl.register_block_size(64, 128)

    out = torch.zeros(
        total_M,
        N,
        dtype=torch.bfloat16,
        device=device,
    )

    for worker_id in hl.grid(num_workers):
        row_start = hl.zeros([], dtype=torch.int32)
        for g in hl.grid(G):
            m_size = m_sizes[g]
            row_end = row_start + m_size

            if m_size != 0:
                n_offset = g * N
                num_m_tiles = (m_size + block_m - 1) // block_m
                num_n_tiles = (N + block_n - 1) // block_n
                tiles_per_group = num_m_tiles * num_n_tiles

                for local_tile in hl.grid(tiles_per_group):
                    tile_linear = local_tile * num_workers + worker_id

                    if tile_linear < tiles_per_group:
                        m_tile_idx = tile_linear % num_m_tiles
                        n_tile_idx = tile_linear // num_m_tiles

                        row_idx = row_start + m_tile_idx * block_m + hl.arange(block_m)
                        col_idx = n_offset + n_tile_idx * block_n + hl.arange(block_n)

                        rows_valid = row_idx < row_end
                        cols_valid = col_idx < (n_offset + N)

                        acc = hl.zeros([block_m, block_n], dtype=torch.float32)

                        for tile_k in hl.tile(K, block_size=block_k):
                            k_idx = tile_k.index

                            a_tile = hl.load(
                                A_fp8,
                                [row_idx, k_idx],
                                extra_mask=rows_valid[:, None],
                            )
                            b_tile = hl.load(
                                B_fp8,
                                [col_idx, k_idx],
                                extra_mask=cols_valid[:, None],
                            )
                            b_tile_t = b_tile.transpose(0, 1)

                            if use_fast_accum:
                                acc = hl.dot(a_tile, b_tile_t, acc=acc)
                            else:
                                acc = acc + hl.dot(a_tile, b_tile_t)

                        a_scale_tile = hl.load(
                            a_scale,
                            [row_idx],
                            extra_mask=rows_valid,
                        )
                        b_scale_tile = hl.load(
                            b_scale,
                            [col_idx],
                            extra_mask=cols_valid,
                        )

                        acc = acc * a_scale_tile[:, None]
                        acc = acc * b_scale_tile[None, :]

                        out_cols = col_idx - n_offset
                        valid_mask = rows_valid[:, None] & cols_valid[None, :]
                        hl.store(
                            out,
                            [row_idx, out_cols],
                            acc.to(torch.bfloat16),
                            extra_mask=valid_mask,
                        )

            row_start = row_end

    return out


def _compute_group_offsets(m_sizes: torch.Tensor) -> torch.Tensor:
    """Return prefix sums [0, cumsum(m_sizes)]."""

    group_offsets = torch.empty(
        m_sizes.size(0) + 1,
        dtype=torch.int32,
        device=m_sizes.device,
    )
    group_offsets[0] = 0
    group_offsets[1:] = torch.cumsum(m_sizes, dim=0)
    return group_offsets


def _reference_grouped_fp8_matmul(
    group_A: torch.Tensor,
    group_B: torch.Tensor,
    m_sizes: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation mirroring TritonBench baseline."""

    group_size = int(m_sizes.numel())
    total_M, K = group_A.shape
    total_N = group_B.shape[0]
    assert total_N % group_size == 0
    N = total_N // group_size

    out = torch.zeros(total_M, N, dtype=torch.bfloat16, device=group_A.device)
    group_offsets = _compute_group_offsets(m_sizes)

    for g in range(group_size):
        row_start = int(group_offsets[g].item())
        row_end = int(group_offsets[g + 1].item())
        n_start = g * N
        n_end = n_start + N

        accum = (
            group_A[row_start:row_end, :].to(torch.float32)
            @ group_B[n_start:n_end, :].to(torch.float32).T
        )
        accum = accum * a_scale[row_start:row_end][:, None]
        accum = accum * b_scale[n_start:n_end][None, :]
        out[row_start:row_end, :] = accum.to(torch.bfloat16)

    return out


def fp8_gemm_rowwise_grouped_tritonbench(
    tb_op: object,
    group_A: torch.Tensor,
    group_B: torch.Tensor,
    m_sizes: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    """Adapter that matches TritonBench operator signature."""

    use_fast_accum = True if tb_op is None else getattr(tb_op, "fp8_fast_accum", True)

    return lambda: fp8_gemm_rowwise_grouped(
        group_A,
        group_B,
        m_sizes,
        a_scale,
        b_scale,
        use_fast_accum,
    )


def _quantize_rowwise(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rowwise FP8 quantization helper used by example check."""

    dtype_fp8 = torch.float8_e4m3fn
    limit = torch.finfo(dtype_fp8).max
    row_max = torch.max(torch.abs(x), dim=1, keepdim=False).values
    scale = limit / torch.clamp(row_max, min=1e-12)
    if x.dtype is torch.float16:
        scale = torch.clamp(scale, max=torch.finfo(torch.float16).max)
    xq = torch.clamp(x * scale[:, None], min=-limit, max=limit).to(dtype_fp8)
    return xq, scale.reciprocal().to(torch.float32)


def check(
    m: int,
    n: int,
    k: int,
    group_size: int,
    device: torch.device | str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate kernel against baseline for a synthetic configuration."""

    torch.manual_seed(0)

    A = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    B = torch.randn((group_size * n, k), device=device, dtype=torch.bfloat16)

    A_fp8, a_scale = _quantize_rowwise(A)
    B_fp8, b_scale = _quantize_rowwise(B)
    m_sizes = torch.full((group_size,), m // group_size, dtype=torch.int32, device=device)

    ref = _reference_grouped_fp8_matmul(A_fp8, B_fp8, m_sizes, a_scale, b_scale)
    out = fp8_gemm_rowwise_grouped_tritonbench(
        None, A_fp8, B_fp8, m_sizes, a_scale, b_scale
    )()

    torch.testing.assert_close(ref, out, atol=1e-2, rtol=5e-1)
    return ref, out


def main() -> None:
    check(m=1024, n=256, k=1024, group_size=4)


if __name__ == "__main__":
    main()
