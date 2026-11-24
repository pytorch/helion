"""Scaled dot-product (flash) attention for the B200 CuTe (tcgen05) backend.

Ported from ``examples/attention.py`` (the ``attention_output`` variant, which
returns only the attention output so it lines up 1:1 with
``F.scaled_dot_product_attention``).  Runs on Helion's CuTe (tcgen05) flash
backend and is pretuned for the non-causal dense attention shapes that back the
B200 CuTe attention benchmark (``benchmarks/cute/compare_attention_backends.py``);
the checked-in AOT heuristic selects the flash config per-shape instead of
online autotuning.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import helion.experimental
import helion.language as hl


def _attention_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v)


@helion.experimental.aot_kernel(backend="cute", static_shapes=True)
def attention(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    """Computes scaled dot-product attention and returns the output tensor.

    Args:
        q_in: Query tensor of shape [..., seq_len_q, head_dim]
        k_in: Key tensor of shape [..., seq_len_k, head_dim]
        v_in: Value tensor of shape [..., seq_len_k, head_dim]

    Returns:
        Output tensor of shape [..., seq_len_q, head_dim].
    """
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            # scaling Q in-loop on-demand reduces spillage, faster than keeping pre-scaled Q
            q_scaled = q * qk_scale
            k = k_view[tile_b, tile_n, :]
            qk = torch.bmm(q_scaled, k.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


def _baselines() -> list[tuple[str, object]]:
    """Baselines main() benchmarks against.

    SDPA is a single fused op, so ``torch.compile`` has nothing to fuse and is
    not a meaningful separate baseline -- torch SDPA is the only one.
    """
    return [
        ("torch", _attention_sdpa),
    ]


def use_cudagraph() -> bool:
    """Whether main() benchmarks under CUDA graphs (read by pretuned_kernels/run.py).

    Attention is timed under CUDA graphs (like ``scaled_mm``) so the numbers
    reflect kernel-only latency rather than the CuTe AOT dispatch's host
    overhead, which otherwise dominates the small-shape rows.
    """
    return True


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep

    # (z, h, seq_len, head_dim, dtype) -- non-causal dense attention shapes from
    # the CuTe attention benchmark harness (benchmarks/cute/compare_attention_backends.py).
    shapes = [
        (1, 4, 512, 64, torch.float16),
        (2, 8, 512, 64, torch.float16),
        (2, 32, 1024, 64, torch.float16),
        (2, 32, 2048, 64, torch.float16),
        (4, 32, 4096, 128, torch.bfloat16),
        (8, 32, 8192, 128, torch.bfloat16),
    ]
    baselines = _baselines()

    def make_calls(shape: tuple[int, int, int, int, torch.dtype]) -> tuple:
        z, h, seq_len, head_dim, dtype = shape
        q, k, v = (
            torch.randn([z, h, seq_len, head_dim], device="cuda", dtype=dtype)
            for _ in range(3)
        )

        def helion_call() -> torch.Tensor:
            return attention(q, k, v)

        base_calls = [(name, (lambda fn=fn: fn(q, k, v))) for name, fn in baselines]
        return (
            helion_call,
            base_calls,
            f"{z:>2d}  {h:>3d}  {seq_len:>6d}  {head_dim:>4d}",
        )

    return run_sweep(
        shapes,
        make_calls,
        use_cudagraph=use_cudagraph(),
        warmup=50,
        rep=200,
        verbose=verbose,
        shape_header=f"{'z':>2s}  {'h':>3s}  {'seq':>6s}  {'d':>4s}",
    )


if __name__ == "__main__":
    main()
