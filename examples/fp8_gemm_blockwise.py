"""
FP8 Blockwise General Matrix Multiplication (GEMM) with Helion
================================================================
This example sketches a blockwise-quantized FP8 GEMM kernel that mirrors Triton's
``matmul_fp8_block`` logic. Inputs are assumed to be quantized with reciprocal scaling
factors per ``[block_m, block_k]`` (left operand) and ``[block_n, block_k]`` (right operand).
The implementation follows the same accumulation / rescaling pattern as the Triton kernel; it
is provided as a design reference and is **not** yet functional under the current Helion
codegen constraints.
"""

from __future__ import annotations

from typing import Callable

import torch

import helion
from helion._testing import run_example
import helion.language as hl


BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128
SPLIT_K = 2


def _cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def dequantize_blockwise(
    q: torch.Tensor,
    scale: torch.Tensor,
    block_m: int,
    block_k: int,
) -> torch.Tensor:
    """Dequantize an FP8 tensor using blockwise reciprocal scales."""
    m, k = q.shape
    out = torch.empty((m, k), dtype=torch.float32, device=q.device)
    grid_m = _cdiv(m, block_m)
    grid_k = _cdiv(k, block_k)
    for idx_m in range(grid_m):
        m_start = idx_m * block_m
        m_end = min(m_start + block_m, m)
        for idx_k in range(grid_k):
            k_start = idx_k * block_k
            k_end = min(k_start + block_k, k)
            scale_val = scale[idx_m, idx_k].to(torch.float32)
            out[m_start:m_end, k_start:k_end] = (
                q[m_start:m_end, k_start:k_end].to(torch.float32) * scale_val
            )
    return out


@helion.kernel(static_shapes=True)
def fp8_gemm_blockwise_kernel(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    scale_block_m: int,
    scale_block_n: int,
    scale_block_k: int,
) -> torch.Tensor:
    """
    Blockwise FP8 GEMM kernel in Helion.

    Args:
        x_fp8: Left operand of shape ``[M, K]`` in ``torch.float8_e4m3fn``.
        w_fp8: Right operand of shape ``[N, K]`` in ``torch.float8_e4m3fn``.
        x_scale: Reciprocal scales with shape ``[ceil(M / scale_block_m), ceil(K / scale_block_k)]``.
        w_scale: Reciprocal scales with shape ``[ceil(N / scale_block_n), ceil(K / scale_block_k)]``.
        scale_block_m / n / k: Block sizes describing how the reciprocal scales are laid out.

    Returns:
        torch.Tensor: Output tensor in torch.bfloat16 with shape [m, n].
    """
    m, k = x_fp8.shape
    n, k_w = w_fp8.shape
    assert k == k_w, "Input K dimensions must match"

    # Accumulate in fp32. Final conversion to bf16 happens once all splits are reduced.
    out_acc = torch.zeros((m, n), dtype=torch.float32, device=x_fp8.device)

    k_block = helion.next_power_of_2(_cdiv(k, SPLIT_K))
    k_multiple = max(1, scale_block_k // BLOCK_K)

    # -----------------------------------------------------------------------
    # Algorithm sketch (mirrors Triton's matmul_fp8_block_fastacc path):
    #
    # 1. Tile [M,N] by BLOCK_M x BLOCK_N and iterate over the reduction dimension
    #    in BLOCK_K chunks.  Triton additionally performs an L2-friendly swizzle
    #    (GROUP_M) which Helion currently lacks, but the logical ordering matches.
    #
    # 2. For each BLOCK_K tile, load fp8 blocks of A and B, convert them to fp32,
    #    and accumulate with `tl.dot` (here `hl.dot`) into an fp32 accumulator.
    #
    # 3. After every `k_multiple = scale_block_k / BLOCK_K` tiles, apply the block
    #    scale factors.  In Triton this happens inside the loop with `scale` and
    #    `scale_next_inv_scale`, preserving the accumulator in a consistent scale
    #    domain even when more K tiles remain.
    #
    # 4. Once all BLOCK_K tiles have been processed, write the accumulator to the
    #    output buffer.  Triton uses `tl.atomic_add` whenever SPLIT_K > 1 so partial
    #    results from every split rendezvous in C; the Helion sketch mirrors that
    #    behaviour with `hl.atomic_add`.
    # -----------------------------------------------------------------------

    # Mirror the Triton kernel's swizzled block traversal (no GROUP_M equivalent yet).
    for tile_m, tile_n, outer_k in hl.tile(
        [m, n, k], block_size=[BLOCK_M, BLOCK_N, k_block]
    ):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        scale_m = tile_m.begin // scale_block_m
        scale_n = tile_n.begin // scale_block_n
        split_begin = outer_k.begin
        split_end = outer_k.end

        for tile_k in hl.tile(k, block_size=BLOCK_K):
            if split_begin <= tile_k.begin < split_end:
                a_tile = x_fp8[tile_m, tile_k].to(torch.float32)
                b_tile = w_fp8[tile_n, tile_k].to(torch.float32)
                acc = hl.dot(a_tile, b_tile.transpose(0, 1), acc=acc)

                pid_k = tile_k.begin // BLOCK_K
                is_last_chunk_for_split = tile_k.end >= split_end
                boundary = ((pid_k + 1) % k_multiple == 0) or is_last_chunk_for_split or (tile_k.end >= k)
                if boundary:
                    scale_k = pid_k // k_multiple
                    # Triton's matmul_fp8_block keeps the accumulator in "original" FP32 units by
                    # multiplying with the current block-scale and then dividing by the next block's
                    # scale.  Conceptually, each BLOCK_K chunk produces values proportional to
                    #   acc_i = (A_i * scale_a_i) @ (B_i * scale_b_i)
                    # and when the next chunk is processed, the previous partial sum is rescaled so
                    # all contributions share the same quantization basis.  The snippet below mirrors
                    # that algebra: multiply by the present `scale`, and if another chunk follows,
                    # pre-divide by the "next" scale so the accumulator remains normalized for the
                    # upcoming iteration.  When we reach the final chunk there is no future block, so
                    # we simply multiply by the last scale to restore the dequantized magnitude.
                    a_scale = x_scale[scale_m, scale_k].to(torch.float32)
                    b_scale = w_scale[scale_n, scale_k].to(torch.float32)
                    scale = a_scale * b_scale
                    if tile_k.end < k and (scale_k + 1) < x_scale.shape[1]:
                        a_scale_next = x_scale[scale_m, scale_k + 1].to(torch.float32)
                        b_scale_next = w_scale[scale_n, scale_k + 1].to(torch.float32)
                        next_scale = a_scale_next * b_scale_next
                        acc = acc * (scale / next_scale)
                    else:
                        acc = acc * scale

        if SPLIT_K == 1 or split_begin == 0:
            out_acc[tile_m, tile_n] = acc
        else:
            hl.atomic_add(out_acc, [tile_m, tile_n], acc)

    return out_acc.to(torch.bfloat16)


def fp8_gemm_blockwise(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """Top-level API forwarding to the kernel with Triton-like block logic."""
    return fp8_gemm_blockwise_kernel(
        x_fp8,
        w_fp8,
        x_scale,
        w_scale,
        scale_block_m=BLOCK_M,
        scale_block_n=BLOCK_N,
        scale_block_k=BLOCK_K,
    )


def reference_fp8_gemm_blockwise(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    scale_block_m: int = BLOCK_M,
    scale_block_n: int = BLOCK_N,
    scale_block_k: int = BLOCK_K,
) -> torch.Tensor:
    """Reference implementation that dequantizes to FP32 and calls torch.matmul."""
    x_fp32 = dequantize_blockwise(x_fp8, x_scale, scale_block_m, scale_block_k)
    w_fp32 = dequantize_blockwise(w_fp8, w_scale, scale_block_n, scale_block_k)
    # w_fp8 is stored as [n, k]; convert to [k, n] for matmul
    w_fp32_t = w_fp32.transpose(0, 1)
    result = torch.matmul(x_fp32, w_fp32_t)
    return result.to(torch.bfloat16)


def fp8_block_quantize(
    x: torch.Tensor, block_m: int = BLOCK_M, block_k: int = BLOCK_K
) -> tuple[torch.Tensor, torch.Tensor]:
    """Utility quantizer matching TritonBench's blockwise quantization."""
    E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
    EPS = 1e-12

    m, k = x.shape
    grid_m = _cdiv(m, block_m)
    grid_k = _cdiv(k, block_k)
    padded_m = grid_m * block_m
    padded_k = grid_k * block_k

    x_padded = torch.zeros((padded_m, padded_k), dtype=x.dtype, device=x.device)
    x_padded[:m, :k] = x

    block_view = x_padded.view(grid_m, block_m, grid_k, block_k)
    block_max = block_view.abs().amax(dim=(1, 3)).clamp(min=EPS)
    scale = (E4M3_MAX_POS / block_max.to(torch.float32)).to(torch.float32)

    scaled = (
        x_padded
        * scale.repeat_interleave(block_m, dim=0).repeat_interleave(block_k, dim=1)
    )[:m, :k]
    q = scaled.to(torch.float8_e4m3fn)
    return q, 1.0 / scale


def fp8_gemm_blockwise_tritonbench(
    tb_op: object,
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
) -> Callable[[], torch.Tensor]:
    """Wrapper to plug the Helion kernel into TritonBench."""
    return lambda: fp8_gemm_blockwise(x_fp8, w_fp8, x_scale, w_scale)


def accuracy_check(
    m: int = 256,
    n: int = 512,
    k: int = 384,
    *,
    atol: float = 1e-2,
    rtol: float = 0.5,
) -> tuple[bool, torch.Tensor]:
    """Compare the Helion kernel against the reference matmul for one random shape."""
    x = torch.randn((m, k), device="cuda", dtype=torch.float32)
    w = torch.randn((n, k), device="cuda", dtype=torch.float32)
    x_fp8, x_scale = fp8_block_quantize(x)
    w_fp8, w_scale = fp8_block_quantize(w)

    helion_out = fp8_gemm_blockwise(x_fp8, w_fp8, x_scale, w_scale)
    ref_out = reference_fp8_gemm_blockwise(x_fp8, w_fp8, x_scale, w_scale)

    torch.testing.assert_close(
        helion_out.to(torch.float32),
        ref_out.to(torch.float32),
        atol=atol,
        rtol=rtol,
    )

def main() -> None:
    _ = accuracy_check()


if __name__ == "__main__":
    main()
