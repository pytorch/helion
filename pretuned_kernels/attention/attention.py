"""Scaled dot-product attention for the B200 CuTe flash backend.

This is the output-only dense attention kernel from ``examples/attention.py``.
It is pretuned for the non-causal shapes used by
``benchmarks/cute/compare_attention_backends.py`` and compares against a single
cuDNN SDPA baseline.
"""

from __future__ import annotations

import math

import torch

import helion
import helion.language as hl


def _attention_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Run the same cuDNN SDPA baseline as the backend comparison harness."""
    with torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.CUDNN_ATTENTION]
    ):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


@helion.aot_kernel(backend="cute", static_shapes=True)
def attention(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    """Compute dense scaled dot-product attention without an auxiliary LSE."""
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


# Non-causal dense shapes from benchmarks/cute/compare_attention_backends.py.
SHAPES = [
    (1, 4, 512, 64, torch.float16),
    (2, 8, 512, 64, torch.float16),
    (2, 32, 1024, 64, torch.float16),
    (2, 32, 2048, 64, torch.float16),
    (4, 32, 4096, 128, torch.bfloat16),
    (8, 32, 8192, 128, torch.bfloat16),
]


def _make_inputs(
    z: int,
    h: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (z, h, seq_len, head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    return q, k, v


def use_cudagraph() -> bool:
    """Benchmark kernel-only latency without CuTe AOT dispatch overhead."""
    return True


def correctness_check() -> None:
    """Check every checked-in config branch against cuDNN SDPA."""
    torch.manual_seed(0)
    shapes = (
        (1, 2, 256, 64, torch.float16),
        (1, 2, 256, 128, torch.bfloat16),
        (1, 1, 8192, 64, torch.float16),
    )
    for shape in shapes:
        args = _make_inputs(*shape)
        actual = attention(*args)
        expected = _attention_sdpa(*args)
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=2e-2)


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep  # pyrefly: ignore[missing-import]

    def make_calls(shape: tuple[int, int, int, int, torch.dtype]) -> tuple:
        z, h, seq_len, head_dim, dtype = shape
        q, k, v = _make_inputs(z, h, seq_len, head_dim, dtype)

        def helion_call() -> torch.Tensor:
            return attention(q, k, v)

        def sdpa_call() -> torch.Tensor:
            return _attention_sdpa(q, k, v)

        return (
            helion_call,
            [("sdpa", sdpa_call)],
            f"{z:>2d}  {h:>3d}  {seq_len:>6d}  {head_dim:>4d}",
        )

    return run_sweep(
        SHAPES,
        make_calls,
        use_cudagraph=use_cudagraph(),
        warmup=50,
        rep=200,
        verbose=verbose,
        shape_header=f"{'z':>2s}  {'h':>3s}  {'seq':>6s}  {'d':>4s}",
    )


if __name__ == "__main__":
    correctness_check()
    main()
