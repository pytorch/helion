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
from helion._testing import run_example
import helion.language as hl


# %%
@helion.kernel
def rope_fwd(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    batch, _, seq_len, head_dim = q.size()
    half_dim = head_dim // 2
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    cos = cos.expand(batch, seq_len, head_dim).unsqueeze(1)
    sin = sin.expand(batch, seq_len, head_dim).unsqueeze(1)
    # Round each product to the input dtype before adding, as in PyTorch.
    for tile_b, tile_t in hl.tile([batch, seq_len]):
        cos_first = cos[tile_b, :, tile_t, :half_dim].float()
        cos_second = cos[tile_b, :, tile_t, half_dim:].float()
        sin_first = sin[tile_b, :, tile_t, :half_dim].float()
        sin_second = sin[tile_b, :, tile_t, half_dim:].float()
        q_first = q[tile_b, :, tile_t, :half_dim].float()
        q_second = q[tile_b, :, tile_t, half_dim:].float()
        q_out[tile_b, :, tile_t, :half_dim] = (
            (q_first * cos_first).to(q.dtype) - (q_second * sin_first).to(q.dtype)
        ).to(q.dtype)
        q_out[tile_b, :, tile_t, half_dim:] = (
            (q_second * cos_second).to(q.dtype) + (q_first * sin_second).to(q.dtype)
        ).to(q.dtype)
        k_first = k[tile_b, :, tile_t, :half_dim].float()
        k_second = k[tile_b, :, tile_t, half_dim:].float()
        k_out[tile_b, :, tile_t, :half_dim] = (
            (k_first * cos_first).to(k.dtype) - (k_second * sin_first).to(k.dtype)
        ).to(k.dtype)
        k_out[tile_b, :, tile_t, half_dim:] = (
            (k_second * cos_second).to(k.dtype) + (k_first * sin_second).to(k.dtype)
        ).to(k.dtype)
    return q_out, k_out


@helion.kernel
def rope_bwd(
    grad_q_out: torch.Tensor,
    grad_k_out: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gradients for the RoPE inputs q and k."""
    batch, _, seq_len, head_dim = grad_q_out.size()
    half_dim = head_dim // 2
    grad_q = torch.empty_like(grad_q_out)
    grad_k = torch.empty_like(grad_k_out)
    cos = cos.expand(batch, seq_len, head_dim).unsqueeze(1)
    sin = sin.expand(batch, seq_len, head_dim).unsqueeze(1)
    # Round each product to the input dtype before adding, as in PyTorch.
    for tile_b, tile_t in hl.tile([batch, seq_len]):
        cos_first = cos[tile_b, :, tile_t, :half_dim].float()
        cos_second = cos[tile_b, :, tile_t, half_dim:].float()
        sin_first = sin[tile_b, :, tile_t, :half_dim].float()
        sin_second = sin[tile_b, :, tile_t, half_dim:].float()
        q_first = grad_q_out[tile_b, :, tile_t, :half_dim].float()
        q_second = grad_q_out[tile_b, :, tile_t, half_dim:].float()
        grad_q[tile_b, :, tile_t, :half_dim] = (
            (q_first * cos_first).to(grad_q_out.dtype)
            + (q_second * sin_second).to(grad_q_out.dtype)
        ).to(grad_q.dtype)
        grad_q[tile_b, :, tile_t, half_dim:] = (
            (q_second * cos_second).to(grad_q_out.dtype)
            - (q_first * sin_first).to(grad_q_out.dtype)
        ).to(grad_q.dtype)
        k_first = grad_k_out[tile_b, :, tile_t, :half_dim].float()
        k_second = grad_k_out[tile_b, :, tile_t, half_dim:].float()
        grad_k[tile_b, :, tile_t, :half_dim] = (
            (k_first * cos_first).to(grad_k_out.dtype)
            + (k_second * sin_second).to(grad_k_out.dtype)
        ).to(grad_k.dtype)
        grad_k[tile_b, :, tile_t, half_dim:] = (
            (k_second * cos_second).to(grad_k_out.dtype)
            - (k_first * sin_first).to(grad_k_out.dtype)
        ).to(grad_k.dtype)
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
    *args: Any,  # noqa: ANN401
) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    """Wrapper for the TritonBench RoPE operator.

    TritonBench changed the rope operator's input convention upstream
    (meta-pytorch/tritonbench@3f34a4e): newer checkouts yield
    ``(q, k, cos, sin, pos_ids)`` tensors directly from ``get_input_iter``,
    while older ones yield ``(hidden_size, seq_length)`` and expect each
    variant to call ``tb_op.prepare_input(...)`` itself. Support both so this
    keeps working regardless of which TritonBench revision is checked out.
    """
    if len(args) == 2:
        # Old TritonBench: args are (hidden_size, seq_length). `prepare_input`
        # (patched onto Operator in benchmarks/run.py for old checkouts) also
        # populates self.q/k/dq/dk for get_bwd_fn.
        q, k, cos, sin, pos_ids = tb_op.prepare_input(*args)  # pyrefly: ignore [missing-attribute]
    elif len(args) == 5:
        # New TritonBench: args are the (q, k, cos, sin, pos_ids) tensors.
        q, k, cos, sin, pos_ids = args
        # TritonBench's own bwd variants populate self.q/k/dq/dk via
        # _save_for_backward inside their @register_benchmark methods;
        # replicate that here so Operator.get_bwd_fn works for this variant.
        tb_op._save_for_backward(q, k)  # pyrefly: ignore [missing-attribute]
    else:
        raise ValueError(
            f"rope_tritonbench got {len(args)} positional args, expected 2 "
            "(old TritonBench: hidden_size, seq_length) or 5 (new TritonBench: "
            "q, k, cos, sin, pos_ids)"
        )
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
    run_example(
        rope,  # pyrefly: ignore [bad-argument-type]
        rope_pytorch,  # pyrefly: ignore [bad-argument-type]
        (q, k, cos, sin),
        atol=1e-2,
        rtol=1e-2,
    )


if __name__ == "__main__":
    main()
