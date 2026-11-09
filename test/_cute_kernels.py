"""Canonical kernels the CuTe suite redeclares: one definition per shape.

Isolation: a ``helion.Kernel`` caches its bound kernels (``bind``) and tests
mutate bound state (``bound.config_spec.tcgen05.search_enabled = ...``,
``spec.block_sizes[...]``), so a shared kernel must not be ``bind``-ed by a test
that mutates or inspects per-bind state.  Either bind in isolation
(``kernel._bind_isolated(args)`` / ``helion._testing.cute_codegen.cpu_bind``, which never
reads or publishes the cache) or take a fresh wrapper per test with
:func:`fresh`; ``kernel.reset()`` after a test is the fallback when the test
needs the cached ``bind`` itself.  The kernel names are the ones the suite
rendered before the catalogue existed, so generated code is unchanged.
"""

from __future__ import annotations

import math

import torch

import helion
import helion.language as hl


def fresh(kernel: helion.Kernel, **settings: object) -> helion.Kernel:
    """A new ``helion.kernel`` over ``kernel``'s function (same backend, shape
    specialization and source, empty bind cache); ``settings`` add or override
    kernel settings."""
    kwargs: dict[str, object] = {
        "backend": kernel.settings.backend,
        "static_shapes": kernel.settings.static_shapes,
        **settings,
    }
    return helion.kernel(kernel.fn, **kwargs)  # pyrefly: ignore [bad-argument-type]


# ---- tcgen05 GEMM family: fp16/bf16 inputs, fp32 accumulation -------------


@helion.kernel(backend="cute")
def cute_matmul_mma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@helion.kernel(backend="cute")
def cute_matmul_residual(
    x: torch.Tensor, y: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = (acc + residual[tile_m, tile_n]).to(x.dtype)
    return out


@helion.kernel(backend="cute")
def cute_matmul_bias(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = (acc + bias[tile_n]).to(x.dtype)
    return out


@helion.kernel(backend="cute")
def cute_matmul_bias_residual_gelu(
    x: torch.Tensor,
    y: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = torch.nn.functional.gelu(
            1.25 * acc + 0.5 * residual[tile_m, tile_n] + bias[tile_n],
            approximate="tanh",
        ).to(x.dtype)
    return out


@helion.kernel(backend="cute")
def cute_matmul_silu_bf16(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = torch.nn.functional.silu(acc).to(x.dtype)
    return out


@helion.kernel(backend="cute")
def cute_fp8_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.size()
    _, n = y.size()
    out = torch.empty([m, n], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
        out[tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


# ---- online softmax over a row, two passes over the columns --------------


@helion.kernel(backend="cute")
def _reduction_kernel(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
    return out


# The same two-pass softmax under the name the cluster online-pair tests render
# (``test_cute_cluster_online_pair`` binds it; the cluster FMA, reduction
# input-guard and online-cluster compatibility tests rebuild from its ``fn``).
@helion.kernel(backend="cute")
def softmax_two_pass_kernel(x: torch.Tensor) -> torch.Tensor:
    """Mirrors ``examples/softmax.py::softmax_two_pass``."""
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        for tile_n in hl.tile(n, block_size=block_size_n):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
    return out


# ---- dense attention with the log-sum-exp output ---------------------------


@helion.kernel(backend="cute", static_shapes=True)
def _attention_with_lse(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    lse = torch.empty(
        [q_view.size(0), m_dim, 1], device=q_in.device, dtype=torch.float32
    )
    qk_scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
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
        lse[tile_b, tile_m, :] = (m_i + torch.log2(l_i))[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size()), lse.reshape(q_in.size()[:-1])


# ---- default-backend GEMM used by the GPU tcgen05 tests --------------------


@helion.kernel(static_shapes=True)
def _gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.size()
    _, n = b.size()
    out = torch.empty([m, n], dtype=a.dtype, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


# ---- config factories ------------------------------------------------------
