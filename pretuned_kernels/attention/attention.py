"""Scaled dot-product attention for the B200/GB300 CuTe flash backend.

These are the output-only dense and causal attention kernels from
``examples/attention.py``. They are pretuned for shapes used by
``benchmarks/cute/compare_attention_backends.py`` and compare against a single
cuDNN SDPA baseline. On GB300 the default sweep uses the long dense and causal
sequence lengths from that benchmark's eight-shape comparison.
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
    *,
    is_causal: bool = False,
) -> torch.Tensor:
    """Run the same cuDNN SDPA baseline as the backend comparison harness."""
    with torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.CUDNN_ATTENTION]
    ):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=is_causal
        )


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


@helion.aot_kernel(backend="cute", static_shapes=True)
def causal_attention(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    """Compute causal scaled dot-product attention without an auxiliary LSE."""
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
            qk = torch.where(
                tile_m.index[None, :, None] >= tile_n.index[None, None, :],
                qk,
                float("-inf"),
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
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


AttentionShape = tuple[int, int, int, int, torch.dtype, bool]


# Original B200 representative sweep.
B200_SHAPES: tuple[AttentionShape, ...] = (
    (1, 4, 512, 64, torch.float16, False),
    (2, 8, 512, 64, torch.float16, False),
    (2, 32, 1024, 64, torch.float16, False),
    (2, 32, 2048, 64, torch.float16, False),
    (4, 32, 4096, 128, torch.bfloat16, False),
    (8, 32, 8192, 128, torch.bfloat16, False),
)

# The comparison harness's dense_causal8 suite. Dense shapes select sm103
# nonpersistent two-CTA configs; causal shapes select one-CTA configs.
GB300_SHAPES: tuple[AttentionShape, ...] = (
    (2, 32, 32768, 64, torch.float16, False),
    (2, 32, 65536, 64, torch.float16, False),
    (2, 32, 131072, 64, torch.float16, False),
    (2, 32, 262144, 64, torch.float16, False),
    (2, 32, 65536, 64, torch.float16, True),
    (2, 32, 131072, 64, torch.float16, True),
    (2, 32, 262144, 64, torch.float16, True),
    (2, 32, 524288, 64, torch.float16, True),
)


def _benchmark_spec(
    compute_capability: tuple[int, int],
) -> tuple[tuple[AttentionShape, ...], int]:
    """Return the target-specific shapes and CUDA-graph repetitions."""
    if compute_capability == (10, 3):
        # These kernels take up to roughly one second, so ten samples are
        # sufficient and keep the standalone sweep practical.
        return GB300_SHAPES, 10
    return B200_SHAPES, 200


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
    checks = (
        ((1, 2, 256, 64, torch.float16, False), attention),
        ((1, 2, 256, 128, torch.bfloat16, False), attention),
        ((1, 1, 8192, 64, torch.float16, False), attention),
    )
    if torch.cuda.get_device_capability() == (10, 3):
        # Exercise all eight sm103 long-sequence selectors with a small
        # batch/head count so correctness does not require the benchmark's
        # full-size allocation.
        checks += tuple(
            ((1, 1, seq, 64, torch.float16, False), attention)
            for seq in (32768, 65536, 131072, 262144)
        )
        checks += tuple(
            ((1, 1, seq, 64, torch.float16, True), causal_attention)
            for seq in (65536, 131072, 262144, 524288)
        )
    for shape, kernel in checks:
        z, h, seq_len, head_dim, dtype, is_causal = shape
        args = _make_inputs(z, h, seq_len, head_dim, dtype)
        actual = kernel(*args)
        expected = _attention_sdpa(*args, is_causal=is_causal)
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=2e-2)


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep  # pyrefly: ignore[missing-import]

    shapes, rep = _benchmark_spec(torch.cuda.get_device_capability())

    def make_calls(shape: AttentionShape) -> tuple:
        z, h, seq_len, head_dim, dtype, is_causal = shape
        q, k, v = _make_inputs(z, h, seq_len, head_dim, dtype)
        kernel = causal_attention if is_causal else attention

        def helion_call() -> torch.Tensor:
            return kernel(q, k, v)

        def sdpa_call() -> torch.Tensor:
            return _attention_sdpa(q, k, v, is_causal=is_causal)

        return (
            helion_call,
            [("sdpa", sdpa_call)],
            f"{'causal' if is_causal else 'dense':>6s}  "
            f"{z:>2d}  {h:>3d}  {seq_len:>6d}  {head_dim:>4d}",
        )

    return run_sweep(
        shapes,
        make_calls,
        use_cudagraph=use_cudagraph(),
        warmup=50,
        rep=rep,
        verbose=verbose,
        shape_header=(f"{'kind':>6s}  {'z':>2s}  {'h':>3s}  {'seq':>6s}  {'d':>4s}"),
    )


if __name__ == "__main__":
    correctness_check()
    main()
