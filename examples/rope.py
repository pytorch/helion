"""
rope Reduction Example
================

This example demonstrates how to implement a rope reduction operation along the last dimension using Helion.
"""

# %%
# Imports
# -------
from __future__ import annotations

import torch

from typing import Tuple
import helion
from helion._testing import run_example
import helion.language as hl

# adapted from HF Transformers
def _compute_default_rope_parameters(
    device,
    rope_theta,
    head_dim,
) -> Tuple["torch.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = rope_theta
    partial_rotary_factor = 1.0
    dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, attention_factor

class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        config=None,
    ):
        super().__init__()
        self.rope_kwargs = {}
        self.rope_kwargs = {
            "rope_type": rope_type,
            "factor": scaling_factor,
            "dim": head_dim,
            "base": base,
            "max_position_embeddings": max_position_embeddings,
        }
        self.rope_type = rope_type
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings

        self.config = config
        self.rope_init_fn = _compute_default_rope_parameters
        rope_theta = 1.0 

        inv_freq, self.attention_scaling = self.rope_init_fn(device, rope_theta, head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)

# %%
# Rope Kernel
# --------
@helion.kernel()
def rope_kernel(q, k, cos, sin, pos_ids) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies RoPE operation to q and k tensors.

    Args:
        q: Input tensor of shape [B, A1, N, K]
        k: Input tensor of shape [B, A2, N, K]
        cos: Input tensor of shape [B, N, K]
        sin: Input tensor of shape [B, N, K]
        pos_ids: Input tensor of shape [B, N]
    Computes:
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    Returns:
        q_embed: Output tensor of shape [B, A1, N, K]
        k_embed: Output tensor of shape [B, A2, N, K]
    """

    b, a1, n, k = q.size()
    b, a2, n, k = k.size()
    q_out = torch.empty([b, a1, n, k], dtype=q.dtype, device=q.device)
    k_out = torch.empty([b, a2, n, k], dtype=k.dtype, device=k.device)
    last_dim_split = k // 2

    for tile_b, tile_a1, tile_n, tile_k in hl.tile([b, a1, n, last_dim_split]):
        # q first half
        # negate the sin part
        q_out[tile_b, tile_a1, tile_n, tile_k] = q[tile_b, tile_a1, tile_n, tile_k] * cos[tile_b, tile_n, tile_k] + -1 * q[tile_b, tile_a1, tile_n, tile_k + last_dim_split] * sin[tile_b, tile_n, tile_k]

        # TODO try splitting this for loop to see which is faster (potentially better memory access pattern)

        # q second half
        # no need to negate the sin part
        q_out[tile_b, tile_a1, tile_n, tile_k + last_dim_split] = q[tile_b, tile_a1, tile_n, tile_k + last_dim_split] * cos[tile_b, tile_n, tile_k + last_dim_split] + q[tile_b, tile_a1, tile_n, tile_k] * sin[tile_b, tile_n, tile_k + last_dim_split]

    for tile_b, tile_a2, tile_n, tile_k in hl.tile([b, a2, n, last_dim_split]):
        # k first half
        # negate the sin part
        k_out[tile_b, tile_a2, tile_n, tile_k] = k[tile_b, tile_a2, tile_n, tile_k] * cos[tile_b, tile_n, tile_k] + -1 * k[tile_b, tile_a2, tile_n, tile_k + last_dim_split] * sin[tile_b, tile_n, tile_k]

        # k second half
        # no need to negate the sin part
        k_out[tile_b, tile_a2, tile_n, tile_k + last_dim_split] = q[tile_b, tile_a2, tile_n, tile_k + last_dim_split] * cos[tile_b, tile_n, tile_k + last_dim_split] + q[tile_b, tile_a2, tile_n, tile_k] * sin[tile_b, tile_n, tile_k + last_dim_split]


    return q_out, k_out

def prepare_input(hidden_size, seq_length, num_q_heads, num_kv_heads, device, dtype):
    """
    Copied from tritonbench.operators.rope.
    """
    head_dim = hidden_size // num_q_heads
    rotary_emb = LlamaRotaryEmbedding(head_dim, device=device)
    q = (
        torch.randn(
            (1, seq_length, num_q_heads, head_dim),
            device=device,
            requires_grad=True,
            dtype=dtype,
        )
        .transpose(1, 2)
        .contiguous()
    )
    k = (
        torch.randn(
            (1, seq_length, num_kv_heads, head_dim),
            device=device,
            requires_grad=True,
            dtype=dtype,
        )
        .transpose(1, 2)
        .contiguous()
    )
    dq, dk = (
        torch.randn_like(q, device=device, dtype=dtype),
        torch.randn_like(k, device=device),
    )
    pos_ids = torch.arange(
        seq_length, device=device, dtype=torch.long
    ).unsqueeze(0)
    cos, sin = rotary_emb(k, pos_ids)
    return q, k, cos, sin, pos_ids

# %%
# Benchmark Wrapper
# --------------
def rope_tritonbench(hidden_size: int, seq_length: int) -> torch.Tensor:
    """
    Returns a function that benchmarks the rope kernel.

    Args:
        x: Input tensor (2D)
        y: Input tensor (2D)

    Returns:
        Rope of the tensors
    """
    num_q_heads = 32
    num_kv_heads = 8
    dtype = torch.float32
    q, k, cos, sin, pos_ids = prepare_input(hidden_size, seq_length, num_q_heads, num_kv_heads, torch.device("cuda"), dtype)
    assert q.shape == torch.Size([1, num_q_heads, 1024, 256])
    assert k.shape == torch.Size([1, num_kv_heads, 1024, 256])
    assert cos.shape == torch.Size([1, 1024, 256])
    assert sin.shape == torch.Size([1, 1024, 256])
    return lambda: rope_kernel(q, k, cos, sin, pos_ids)
    


# %%
# Verification Function
# -------------------
def check(m: int, n: int) -> None:
    """
    Verify the rope kernel implementation against PyTorch's native rope function.

    Args:
        m: First dimension of the test tensor
        n: Second dimension of the test tensor
    """
    x = torch.randn([m, n], device="cuda", dtype=torch.float32)
    kernels = {"helion": rope_kernel}
    run_example(kernels, lambda x: x.rope(-1), (x,))


# %%
# Main Function
# -----------
def main() -> None:
    """
    Main entry point that runs the rope kernel verification with different tensor sizes.

    Tests with two configurations:
    - 512x256
    - 1024x1024
    """
    check(512, 256)
    check(1024, 1024)


if __name__ == "__main__":
    main()
