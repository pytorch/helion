from __future__ import annotations

import math

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)
def fp8_attention_kernel(
    q: torch.Tensor,  # [batch*heads, seq, dim]
    k: torch.Tensor,  # [batch*heads, seq, dim]
    v: torch.Tensor,  # [batch*heads, dim, seq] - pre-transposed
) -> torch.Tensor:
    batch_heads = q.size(0)
    seq_len = q.size(1)
    head_dim = q.size(2)

    # Output tensor
    out = torch.empty(
        [batch_heads, seq_len, head_dim], dtype=torch.float32, device=q.device
    )

    # Scale factor for attention
    sm_scale = 1.0 / math.sqrt(float(head_dim))
    # Triton kernel multiplies sm_scale by 1.44269504 (1/log(2)) for exp2
    sm_scale = sm_scale * 1.44269504

    # Process each batch*head in parallel
    for bh in hl.grid(batch_heads):
        # Process each query position
        for tile_m in hl.tile(seq_len):
            # Initialize for online softmax
            m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
            l_i = hl.full([tile_m], 0.0, dtype=torch.float32)
            acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)

            # Load query tile - keep in FP8
            q_tile = q[bh, tile_m, :]  # [tile_m, dim]

            # Compute attention scores for all keys
            for tile_n in hl.tile(seq_len):
                # Load key tile and transpose for Q @ K^T
                k_tile = k[bh, tile_n, :]  # [tile_n, dim] - keep in FP8
                k_tile_t = k_tile.transpose(0, 1)  # [dim, tile_n]

                # Compute Q @ K^T with FP8 inputs, result in FP32
                qk = torch.matmul(q_tile, k_tile_t).to(
                    torch.float32
                )  # [tile_m, tile_n]

                # Scale QK scores first
                qk_scaled = qk * sm_scale  # [tile_m, tile_n]

                # Compute max of scaled scores
                qk_max = torch.amax(qk_scaled, dim=-1)  # [tile_m]

                # Update global max
                m_new = torch.maximum(m_i, qk_max)

                # Shift by max for numerical stability
                qk_shifted = qk_scaled - m_new[:, None]

                # Use exp2 to match Triton kernel's implementation
                # Note: Triton kernel already multiplies sm_scale by 1.44269504
                p = torch.exp2(qk_shifted)  # [tile_m, tile_n]

                # Sum of exponentials for this block
                l_ij = torch.sum(p, dim=-1)  # [tile_m]

                # Update accumulators with correction factor
                # Correction factor for previous blocks
                alpha = torch.exp2(m_i - m_new)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]

                # Load values - V is [dim, seq]
                v_tile = v[bh, :, tile_n]  # [dim, tile_n] - keep in FP8

                # Convert p to FP8 for FP8 GEMM
                p_fp8 = p.to(v.dtype)  # Convert to same FP8 type as V

                # Accumulate attention @ V with FP8 GEMM
                v_t = v_tile.transpose(0, 1)  # [tile_n, dim]
                pv = torch.matmul(p_fp8, v_t).to(torch.float32)  # [tile_m, dim]
                acc = acc + pv

                # Update max tracker
                m_i = m_new

            # Final normalization
            acc = acc / l_i[:, None]
            out[bh, tile_m, :] = acc

    return out


def prepare_fp8_attention_inputs(
    q: torch.Tensor,  # [batch, heads, seq, dim]
    k: torch.Tensor,  # [batch, heads, seq, dim]
    v: torch.Tensor,  # [batch, heads, seq, dim]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int, int]]:
    """
    Common preprocessing for FP8 attention implementations.

    Returns:
        q_reshaped_fp8: [batch*heads, seq, dim] - in FP8 e5m2
        k_reshaped_fp8: [batch*heads, seq, dim] - in FP8 e5m2
        v_transposed_fp8: [batch*heads, dim, seq] - in FP8 e5m2
        shape: (batch, heads, seq_len, head_dim)
    """
    batch, heads, seq_len, head_dim = q.shape

    # Reshape to [batch*heads, seq, dim]
    q_reshaped = q.reshape(batch * heads, seq_len, head_dim)
    k_reshaped = k.reshape(batch * heads, seq_len, head_dim)

    # Transpose V to [batch, heads, dim, seq] then reshape
    v_transposed = v.permute(0, 1, 3, 2).reshape(batch * heads, head_dim, seq_len)

    # Convert to FP8 e5m2
    q_reshaped_fp8 = q_reshaped.to(torch.float8_e5m2)
    k_reshaped_fp8 = k_reshaped.to(torch.float8_e5m2)
    v_transposed_fp8 = v_transposed.to(torch.float8_e5m2)

    return (
        q_reshaped_fp8,
        k_reshaped_fp8,
        v_transposed_fp8,
        (batch, heads, seq_len, head_dim),
    )


def fp8_attention_tritonbench(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Wrapper for TritonBench compatibility."""
    # Common preprocessing with FP8 conversion
    q_fp8, k_fp8, v_fp8, shape = prepare_fp8_attention_inputs(q, k, v)
    batch, heads, seq_len, head_dim = shape

    # Call the fused kernel
    out_fused = fp8_attention_kernel(q_fp8, k_fp8, v_fp8)

    # Reshape back and convert to FP16
    out = out_fused.reshape(batch, heads, seq_len, head_dim)
    return out.to(torch.float16)


def fp8_attention_pytorch(
    q: torch.Tensor,  # [batch, heads, seq, dim]
    k: torch.Tensor,  # [batch, heads, seq, dim]
    v: torch.Tensor,  # [batch, heads, seq, dim]
) -> torch.Tensor:
    """
    Baseline PyTorch implementation of FP8 attention using FP8 e5m2.
    """
    # Get preprocessed inputs with FP8 conversion
    q_fp8, k_fp8, v_fp8, shape = prepare_fp8_attention_inputs(q, k, v)
    batch, heads, seq_len, head_dim = shape

    sm_scale = 1.0 / math.sqrt(float(head_dim))

    outputs = []

    for i in range(batch * heads):
        q_i = q_fp8[i]  # [seq, dim] - already FP8
        k_i = k_fp8[i]  # [seq, dim] - already FP8
        v_i = v_fp8[i]  # [dim, seq] - pre-transposed, already FP8

        # For Q @ K^T, we need K^T to be column-major
        kt_fp8 = k_i.t()  # column-major [dim, seq]

        # Q @ K^T - dequantize and use regular matmul since e5m2 not supported by _scaled_mm
        q_deq = q_i.to(torch.float32)
        kt_deq = kt_fp8.to(torch.float32)
        qk = torch.matmul(q_deq, kt_deq)

        # Compute max before scaling
        qk_max = torch.amax(qk, dim=-1, keepdim=True)

        # Scale and shift in one operation, then use exp2
        qk_scaled_shifted = qk * sm_scale - qk_max * sm_scale
        p = torch.exp2(qk_scaled_shifted * 1.44269504)

        # Normalize
        p_norm = p / p.sum(dim=-1, keepdim=True)

        # Step 2: Attention @ V using FP8
        # P is [seq, seq], V is [dim, seq]
        # We want P @ V^T = [seq, seq] @ [seq, dim] = [seq, dim]
        p_fp8 = p_norm.to(torch.float8_e5m2)  # row-major [seq, seq]

        # v_i is [dim, seq], already FP8
        vt_fp8 = v_i.t()  # column-major [seq, dim]

        # P @ V^T - dequantize and use regular matmul since e5m2 not supported by torch._scaled_mm
        p_deq = p_fp8.to(torch.float32)
        vt_deq = vt_fp8.to(torch.float32)
        out_i = torch.matmul(p_deq, vt_deq)

        outputs.append(out_i)

    # Stack and reshape back
    out_stacked = torch.stack(outputs, dim=0)  # [batch*heads, seq, dim]
    out = out_stacked.reshape(batch, heads, seq_len, head_dim)

    return out.to(torch.float16)


def check(batch: int, heads: int, seq_len: int, head_dim: int) -> None:
    torch.manual_seed(42)
    q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

    from helion._testing import run_example

    run_example(
        fp8_attention_tritonbench, fp8_attention_pytorch, (q, k, v), atol=0.1, rtol=0.1
    )


def main() -> None:
    check(1, 2, 128, 64)
    check(2, 4, 256, 64)
    check(4, 8, 512, 128)


if __name__ == "__main__":
    main()
