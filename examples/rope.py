"""
Rotary Position Embedding Example
=================================

This example implements LLaMA-style rotary position embeddings in Helion.
"""

# %%
from __future__ import annotations

from typing import Any
from typing import Callable

import torch

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
import helion.language as hl


# %%
@helion.kernel(autotune_seed_configs=helion.Config(block_sizes=[1, 1]))
def rope_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    batch, q_heads, seq_len, head_dim = q.size()
    _, k_heads, _, _ = k.size()
    half_dim = head_dim // 2
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    for tile_b, tile_t in hl.tile([batch, seq_len]):
        cos_pair = (
            cos[tile_b, tile_t, :]
            .to(torch.float32)
            .reshape([tile_b, tile_t, 2, half_dim])  # pyrefly: ignore [no-matching-overload]
            .permute(0, 1, 3, 2)
        )
        sin_pair = (
            sin[tile_b, tile_t, :]
            .to(torch.float32)
            .reshape([tile_b, tile_t, 2, half_dim])  # pyrefly: ignore [no-matching-overload]
            .permute(0, 1, 3, 2)
        )
        cos_first, cos_second = hl.split(cos_pair)
        sin_first, sin_second = hl.split(sin_pair)

        q_pair = (
            q[tile_b, :, tile_t, :]
            .to(torch.float32)
            .reshape([tile_b, q_heads, tile_t, 2, half_dim])  # pyrefly: ignore [no-matching-overload]
            .permute(0, 1, 2, 4, 3)
        )
        q_first, q_second = hl.split(q_pair)
        q_first_out = (
            q_first * cos_first[:, None, :, :] - q_second * sin_first[:, None, :, :]
        )
        q_second_out = (
            q_second * cos_second[:, None, :, :] + q_first * sin_second[:, None, :, :]
        )
        q_out[tile_b, :, tile_t, :] = (
            hl.join(q_first_out, q_second_out)
            .permute(0, 1, 2, 4, 3)
            .reshape([tile_b, q_heads, tile_t, head_dim])  # pyrefly: ignore [no-matching-overload]
            .to(q_out.dtype)
        )

        k_pair = (
            k[tile_b, :, tile_t, :]
            .to(torch.float32)
            .reshape([tile_b, k_heads, tile_t, 2, half_dim])  # pyrefly: ignore [no-matching-overload]
            .permute(0, 1, 2, 4, 3)
        )
        k_first, k_second = hl.split(k_pair)
        k_first_out = (
            k_first * cos_first[:, None, :, :] - k_second * sin_first[:, None, :, :]
        )
        k_second_out = (
            k_second * cos_second[:, None, :, :] + k_first * sin_second[:, None, :, :]
        )
        k_out[tile_b, :, tile_t, :] = (
            hl.join(k_first_out, k_second_out)
            .permute(0, 1, 2, 4, 3)
            .reshape([tile_b, k_heads, tile_t, head_dim])  # pyrefly: ignore [no-matching-overload]
            .to(k_out.dtype)
        )

    return q_out, k_out


@helion.kernel(autotune_seed_configs=helion.Config(block_sizes=[1, 1]))
def rope_bwd(
    grad_q_out: torch.Tensor,
    grad_k_out: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradients for the RoPE inputs q and k."""
    batch, q_heads, seq_len, head_dim = grad_q_out.size()
    _, k_heads, _, _ = grad_k_out.size()
    half_dim = head_dim // 2
    grad_q = torch.empty_like(grad_q_out)
    grad_k = torch.empty_like(grad_k_out)

    for tile_b, tile_t in hl.tile([batch, seq_len]):
        cos_pair = (
            cos[tile_b, tile_t, :]
            .to(torch.float32)
            .reshape([tile_b, tile_t, 2, half_dim])  # pyrefly: ignore [no-matching-overload]
            .permute(0, 1, 3, 2)
        )
        sin_pair = (
            sin[tile_b, tile_t, :]
            .to(torch.float32)
            .reshape([tile_b, tile_t, 2, half_dim])  # pyrefly: ignore [no-matching-overload]
            .permute(0, 1, 3, 2)
        )
        cos_first, cos_second = hl.split(cos_pair)
        sin_first, sin_second = hl.split(sin_pair)

        q_grad_pair = (
            grad_q_out[tile_b, :, tile_t, :]
            .to(torch.float32)
            .reshape([tile_b, q_heads, tile_t, 2, half_dim])  # pyrefly: ignore [no-matching-overload]
            .permute(0, 1, 2, 4, 3)
        )
        q_grad_first_out, q_grad_second_out = hl.split(q_grad_pair)
        q_grad_first = (
            q_grad_first_out * cos_first[:, None, :, :]
            + q_grad_second_out * sin_second[:, None, :, :]
        )
        q_grad_second = (
            q_grad_second_out * cos_second[:, None, :, :]
            - q_grad_first_out * sin_first[:, None, :, :]
        )
        grad_q[tile_b, :, tile_t, :] = (
            hl.join(q_grad_first, q_grad_second)
            .permute(0, 1, 2, 4, 3)
            .reshape([tile_b, q_heads, tile_t, head_dim])  # pyrefly: ignore [no-matching-overload]
            .to(grad_q.dtype)
        )

        k_grad_pair = (
            grad_k_out[tile_b, :, tile_t, :]
            .to(torch.float32)
            .reshape([tile_b, k_heads, tile_t, 2, half_dim])  # pyrefly: ignore [no-matching-overload]
            .permute(0, 1, 2, 4, 3)
        )
        k_grad_first_out, k_grad_second_out = hl.split(k_grad_pair)
        k_grad_first = (
            k_grad_first_out * cos_first[:, None, :, :]
            + k_grad_second_out * sin_second[:, None, :, :]
        )
        k_grad_second = (
            k_grad_second_out * cos_second[:, None, :, :]
            - k_grad_first_out * sin_first[:, None, :, :]
        )
        grad_k[tile_b, :, tile_t, :] = (
            hl.join(k_grad_first, k_grad_second)
            .permute(0, 1, 2, 4, 3)
            .reshape([tile_b, k_heads, tile_t, head_dim])  # pyrefly: ignore [no-matching-overload]
            .to(grad_k.dtype)
        )

    return grad_q, grad_k


class RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx: Any,  # noqa: ANN401
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_out, k_out = rope_fwd(q, k, cos, sin)
        ctx.save_for_backward(cos, sin)
        return q_out, k_out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,  # noqa: ANN401
        grad_q_out: torch.Tensor,
        grad_k_out: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        cos, sin = ctx.saved_tensors
        grad_q, grad_k = rope_bwd(grad_q_out, grad_k_out, cos, sin)
        return grad_q, grad_k, None, None


def rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE with forward and backward support."""
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return RoPEFunction.apply(q, k, cos, sin)  # type: ignore[no-any-return]


def rope_tritonbench(
    tb_op: object,
    hidden_size: int,
    seq_length: int,
) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    """Wrapper for the TritonBench RoPE operator."""
    # pyrefly: ignore [missing-attribute]
    prepared_input = tb_op.prepare_input(hidden_size, seq_length)
    q, k, cos, sin = prepared_input[:4]
    return lambda: rope(q, k, cos, sin)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """PyTorch reference helper matching transformers' LLaMA RoPE convention."""
    half_dim = x.shape[-1] // 2
    return torch.cat((-x[..., half_dim:], x[..., :half_dim]), dim=-1)


def rope_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference implementation."""
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def main() -> None:
    batch, q_heads, k_heads, seq_len, head_dim = 1, 4, 2, 128, 64
    q = torch.randn(
        [batch, q_heads, seq_len, head_dim],
        device=DEVICE,
        dtype=HALF_DTYPE,
        requires_grad=True,
    )
    k = torch.randn(
        [batch, k_heads, seq_len, head_dim],
        device=DEVICE,
        dtype=HALF_DTYPE,
        requires_grad=True,
    )
    angles = torch.randn([batch, seq_len, head_dim], device=DEVICE, dtype=HALF_DTYPE)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    torch.testing.assert_close(
        rope(q, k, cos, sin),
        rope_pytorch(q, k, cos, sin),
        atol=1e-2,
        rtol=1e-2,
    )


if __name__ == "__main__":
    main()
