"""Rotary position embedding for LLaMA-style Q/K tensors."""

from __future__ import annotations

from typing import Any
from typing import Callable

import torch

import helion.experimental
import helion.language as hl


def _rope_shape_key(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[int, int]:
    """Bucket RoPE configs by TritonBench's (hidden_size, seq_length)."""
    _, q_heads, seq_len, head_dim = q.size()
    return q_heads * head_dim, seq_len


@helion.experimental.aot_kernel(key=_rope_shape_key, static_shapes=True)
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


@helion.experimental.aot_kernel(key=_rope_shape_key, static_shapes=True)
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


def pretuned_rope_tritonbench(
    tb_op: object,
    hidden_size: int,
    seq_length: int,
) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    """Wrapper for TritonBench using the pretuned AOT RoPE kernels."""
    # pyrefly: ignore [missing-attribute]
    prepared_input = tb_op.prepare_input(hidden_size, seq_length)
    q, k, cos, sin = prepared_input[:4]
    return lambda: rope(q, k, cos, sin)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half_dim = x.shape[-1] // 2
    return torch.cat((-x[..., half_dim:], x[..., :half_dim]), dim=-1)


def rope_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def _make_inputs(
    hidden_size: int,
    seq_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_heads = 32
    k_heads = 8
    head_dim = hidden_size // q_heads
    q = torch.randn(
        [1, q_heads, seq_length, head_dim],
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn(
        [1, k_heads, seq_length, head_dim],
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    angles = torch.randn([1, seq_length, head_dim], device="cuda", dtype=torch.bfloat16)
    return q, k, torch.cos(angles), torch.sin(angles)


def _baselines() -> list[tuple[str, object]]:
    """Baselines main() benchmarks against.

    ``torch_compile`` is ``torch.compile`` of the torch reference -- a
    speedup-comparison baseline only (not checked for accuracy).
    """
    return [
        ("torch", rope_pytorch),
        ("torch_compile", torch.compile(rope_pytorch)),
    ]


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py)."""
    return False


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    shapes = [
        (8192, 1024),
        (8192, 2048),
        (8192, 4096),
        (8192, 8192),
        (8192, 16384),
        (512, 2048),
        (2048, 2048),
    ]
    baselines = _baselines()

    def make_calls(shape: tuple[int, int]) -> tuple:
        hidden_size, seq_length = shape
        q, k, cos, sin = _make_inputs(hidden_size, seq_length)

        def helion_call() -> tuple[torch.Tensor, torch.Tensor]:
            return rope(q, k, cos, sin)

        base_calls = [
            (name, (lambda fn=fn: fn(q, k, cos, sin))) for name, fn in baselines
        ]
        return helion_call, base_calls, f"{hidden_size:>6d}  {seq_length:>6d}"

    return run_sweep(
        shapes,
        make_calls,
        use_cudagraph=use_cudagraph(),
        verbose=verbose,
        shape_header=f"{'H':>6s}  {'T':>6s}",
    )


if __name__ == "__main__":
    main()
