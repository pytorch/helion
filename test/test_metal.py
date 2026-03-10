from __future__ import annotations

import math
import unittest

import torch

import helion
import helion.language as hl

requires_mps = unittest.skipUnless(
    torch.backends.mps.is_available(),
    "MPS backend not available",
)


@helion.kernel(backend="metal")
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(backend="metal")
def softmax(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        values = x[tile_n, :]
        amax = torch.amax(values, dim=1, keepdim=True)
        exp = torch.exp(values - amax)
        sum_exp = torch.sum(exp, dim=1, keepdim=True)
        out[tile_n, :] = exp / sum_exp
    return out


@helion.kernel(backend="metal")
def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _k2, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="metal", static_shapes=True)
def fused_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Fused attention: softmax(Q @ K^T / sqrt(d)) @ V in a single kernel.

    Q[M, D], K[N, D], V[N, D] -> out[M, D].
    Uses MPP cooperative_tensor chaining: matmul -> reduce -> matmul.
    The Metal backend detects matmul + reduction ops in the body and
    emits a composed kernel via _emit_composed_body().
    """
    m_dim = q.size(0)
    n_dim = k.size(0)
    head_dim = hl.specialize(q.size(1))
    kt = k.transpose(0, 1)
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    for tile_m in hl.tile(m_dim):
        m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_m, head_dim], dtype=torch.float32)
        for tile_n in hl.tile(n_dim):
            qk = torch.mm(q[tile_m, :], kt[:, tile_n])
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            p2 = p.to(v.dtype)
            acc = torch.addmm(acc, p2, v[tile_n, :])
            m_i = m_ij
        acc = acc / l_i[:, None]
        out[tile_m, :] = acc.to(out.dtype)
    return out


@helion.kernel(
    backend="metal",
    static_shapes=True,
    config=helion.Config(block_sizes=[1, 32, 32], num_warps=4),
)
def batched_fused_attention(
    q: torch.Tensor, kt: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Batched fused attention: single dispatch for all B*H heads.

    Q[B*H, M, D], Kt[B*H, D, N], V[B*H, N, D] -> out[B*H, M, D].
    Metal backend detects 3D tensors and generates a 3D grid with
    tgid.z for head indexing.
    """
    num_heads = q.size(0)
    m_dim = q.size(1)
    head_dim = hl.specialize(q.size(2))
    n_dim = v.size(1)
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    for tile_b, tile_m in hl.tile([num_heads, m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        for tile_n in hl.tile(n_dim):
            qk = torch.bmm(q[tile_b, tile_m, :], kt[tile_b, :, tile_n])
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            p2 = p.to(v.dtype)
            acc = torch.baddbmm(acc, p2, v[tile_b, tile_n, :])
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out


class TestMetalBackend(unittest.TestCase):
    @requires_mps
    def test_vector_add(self) -> None:
        device = torch.device("mps")
        x = torch.randn(1024, device=device, dtype=torch.float32)
        y = torch.randn(1024, device=device, dtype=torch.float32)
        result = vector_add(x, y)
        expected = x + y
        torch.testing.assert_close(result, expected)

    @requires_mps
    def test_vector_add_large(self) -> None:
        device = torch.device("mps")
        x = torch.randn(100000, device=device, dtype=torch.float32)
        y = torch.randn(100000, device=device, dtype=torch.float32)
        result = vector_add(x, y)
        expected = x + y
        torch.testing.assert_close(result, expected)

    @requires_mps
    def test_softmax(self) -> None:
        device = torch.device("mps")
        x = torch.randn(128, 256, device=device, dtype=torch.float32)
        result = softmax(x)
        expected = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    @requires_mps
    def test_matmul(self) -> None:
        device = torch.device("mps")
        M, N, K = 128, 128, 128
        x = torch.randn(M, K, device=device, dtype=torch.float32)
        y = torch.randn(K, N, device=device, dtype=torch.float32)
        result = matmul(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_matmul_non_square(self) -> None:
        device = torch.device("mps")
        M, K, N = 128, 64, 256
        x = torch.randn(M, K, device=device, dtype=torch.float32)
        y = torch.randn(K, N, device=device, dtype=torch.float32)
        result = matmul(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_matmul_large_square(self) -> None:
        device = torch.device("mps")
        M, K, N = 512, 512, 512
        x = torch.randn(M, K, device=device, dtype=torch.float32)
        y = torch.randn(K, N, device=device, dtype=torch.float32)
        result = matmul(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_matmul_non_power_of_2(self) -> None:
        device = torch.device("mps")
        M, K, N = 100, 200, 150
        x = torch.randn(M, K, device=device, dtype=torch.float32)
        y = torch.randn(K, N, device=device, dtype=torch.float32)
        result = matmul(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_matmul_prime_sizes(self) -> None:
        device = torch.device("mps")
        M, K, N = 97, 113, 89
        x = torch.randn(M, K, device=device, dtype=torch.float32)
        y = torch.randn(K, N, device=device, dtype=torch.float32)
        result = matmul(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_matmul_tall_skinny(self) -> None:
        device = torch.device("mps")
        M, K, N = 1000, 64, 37
        x = torch.randn(M, K, device=device, dtype=torch.float32)
        y = torch.randn(K, N, device=device, dtype=torch.float32)
        result = matmul(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_matmul_wide_flat(self) -> None:
        device = torch.device("mps")
        M, K, N = 33, 500, 1000
        x = torch.randn(M, K, device=device, dtype=torch.float32)
        y = torch.randn(K, N, device=device, dtype=torch.float32)
        result = matmul(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_matmul_small(self) -> None:
        device = torch.device("mps")
        M, K, N = 7, 11, 13
        x = torch.randn(M, K, device=device, dtype=torch.float32)
        y = torch.randn(K, N, device=device, dtype=torch.float32)
        result = matmul(x, y)
        expected = torch.mm(x, y)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_vector_add_non_power_of_2(self) -> None:
        device = torch.device("mps")
        x = torch.randn(999, device=device, dtype=torch.float32)
        y = torch.randn(999, device=device, dtype=torch.float32)
        result = vector_add(x, y)
        expected = x + y
        torch.testing.assert_close(result, expected)

    @requires_mps
    def test_softmax_non_power_of_2(self) -> None:
        device = torch.device("mps")
        x = torch.randn(100, 200, device=device, dtype=torch.float32)
        result = softmax(x)
        expected = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    @requires_mps
    def test_softmax_autotune(self) -> None:
        """Softmax with autotuning — searches over block_sizes."""

        @helion.kernel(backend="metal", autotune_effort="quick")
        def softmax_tuned(x: torch.Tensor) -> torch.Tensor:
            n, _m = x.size()
            out = torch.empty_like(x)
            for tile_n in hl.tile(n):
                values = x[tile_n, :]
                amax = torch.amax(values, dim=1, keepdim=True)
                exp = torch.exp(values - amax)
                sum_exp = torch.sum(exp, dim=1, keepdim=True)
                out[tile_n, :] = exp / sum_exp
            return out

        device = torch.device("mps")
        x = torch.randn(128, 256, device=device, dtype=torch.float32)
        result = softmax_tuned(x)
        expected = torch.nn.functional.softmax(x, dim=1)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    @requires_mps
    def test_naive_attention(self) -> None:
        """Single-head naive attention: softmax(Q @ K^T / sqrt(d)) @ V."""
        device = torch.device("mps")
        seq_len, head_dim = 64, 32
        q = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)
        k = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)
        v = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)

        result = naive_helion_attention_single(q, k, v)

        expected = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(0).unsqueeze(0),
            k.unsqueeze(0).unsqueeze(0),
            v.unsqueeze(0).unsqueeze(0),
        )
        expected = expected.squeeze(0).squeeze(0)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_naive_attention_batched(self) -> None:
        """Batched naive attention: (batch=1, heads=2, seq=64, dim=32)."""
        device = torch.device("mps")
        batch, heads, seq_len, head_dim = 1, 2, 64, 32
        q = torch.randn(batch, heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device)

        result = naive_helion_attention_batched(q, k, v)

        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_naive_attention_non_power_of_2(self) -> None:
        """Non-power-of-2 sizes: seq_len=50, head_dim=48."""
        device = torch.device("mps")
        seq_len, head_dim = 50, 48
        q = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)
        k = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)
        v = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)

        result = naive_helion_attention_single(q, k, v)

        expected = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(0).unsqueeze(0),
            k.unsqueeze(0).unsqueeze(0),
            v.unsqueeze(0).unsqueeze(0),
        )
        expected = expected.squeeze(0).squeeze(0)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_fused_attention(self) -> None:
        """Fused single-kernel attention: softmax(Q @ K^T / sqrt(d)) @ V."""
        device = torch.device("mps")
        seq_len, head_dim = 64, 32
        q = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)
        k = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)
        v = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)

        result = fused_attention(q, k, v)

        expected = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(0).unsqueeze(0),
            k.unsqueeze(0).unsqueeze(0),
            v.unsqueeze(0).unsqueeze(0),
        )
        expected = expected.squeeze(0).squeeze(0)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)

    @requires_mps
    def test_batched_fused_attention(self) -> None:
        """Batched fused attention: single dispatch for all B*H heads."""
        device = torch.device("mps")
        batch, heads, seq_len, head_dim = 2, 2, 64, 32
        q = torch.randn(batch, heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device)

        num_heads = batch * heads
        q3 = q.reshape(num_heads, seq_len, head_dim).contiguous()
        k3 = k.reshape(num_heads, seq_len, head_dim).contiguous()
        v3 = v.reshape(num_heads, seq_len, head_dim).contiguous()
        kt3 = k3.transpose(1, 2).contiguous()

        result = batched_fused_attention(q3, kt3, v3)
        result = result.view(batch, heads, seq_len, head_dim)

        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)

    @requires_mps
    def test_fused_attention_larger(self) -> None:
        """Fused attention with larger seq_len=128 (different N for matmul)."""
        device = torch.device("mps")
        seq_len, head_dim = 128, 64
        q = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)
        k = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)
        v = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)

        result = fused_attention(q, k, v)

        expected = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(0).unsqueeze(0),
            k.unsqueeze(0).unsqueeze(0),
            v.unsqueeze(0).unsqueeze(0),
        )
        expected = expected.squeeze(0).squeeze(0)
        torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)


def naive_helion_attention_single(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Naive attention on 2D inputs: Q[M,D], K[N,D], V[N,D] -> [M,D].

    Chains Helion Metal matmul and softmax kernels:
      scores = matmul(Q, K^T)          # [M, N]
      scores = scores * (1/sqrt(d))    # elementwise scale on MPS
      weights = softmax(scores)         # row-wise softmax
      output = matmul(weights, V)       # [M, D]
    """
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = matmul(q, k.t().contiguous())
    scores = scores * scale
    weights = softmax(scores)
    return matmul(weights, v)


def naive_helion_attention_batched(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Batched naive attention: Q/K/V are [B, H, M, D] -> [B, H, M, D].

    Loops over batch and heads, dispatching 2D Helion kernels per slice.
    """
    batch, heads, seq_len, head_dim = q.shape
    out = torch.empty_like(q)
    for b in range(batch):
        for h in range(heads):
            out[b, h] = naive_helion_attention_single(q[b, h], k[b, h], v[b, h])
    return out


if __name__ == "__main__":
    unittest.main()
