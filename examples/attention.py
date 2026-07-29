"""
Attention Example
=================

This code implements a custom attention kernel using Helion and PyTorch for efficient computation of scaled dot-product attention,
with support for both static and dynamic input shapes.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import math
from typing import Any
from typing import Callable
from typing import cast

import torch
from torch.nn.attention.flex_attention import flex_attention

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl

# %%
# Attention Kernel Implementation
# -------------------------------


# %%
def _linear_index(index: int, shape: torch.Size) -> tuple[int, ...]:
    result = []
    for size in reversed(shape):
        result.append(index % size)
        index //= size
    return tuple(reversed(result))


def _attention_reference_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    causal: bool,
    bias: torch.Tensor | None,
    base2_lse: bool,
) -> torch.Tensor:
    q_view = q.float().reshape(-1, q.size(-2), q.size(-1))
    k_view = k.float().reshape(-1, k.size(-2), k.size(-1))
    bias_float = bias.float() if bias is not None else None
    lse = torch.empty(
        (q_view.size(0), q_view.size(1)), device=q.device, dtype=torch.float32
    )
    scale = 1.0 / math.sqrt(q.size(-1))
    query_block = 128
    key_block = 4096
    leading_shape = q.size()[:-2]
    for batch_idx in range(q_view.size(0)):
        q_batch = q_view[batch_idx]
        k_batch_t = k_view[batch_idx].transpose(-2, -1)
        bias_batch = None
        if bias_float is not None:
            q_leading_idx = _linear_index(batch_idx, leading_shape)
            bias_leading_shape = bias_float.size()[:-2]
            offset = len(leading_shape) - len(bias_leading_shape)
            bias_idx = tuple(
                0 if size == 1 else q_leading_idx[dim + offset]
                for dim, size in enumerate(bias_leading_shape)
            )
            bias_batch = bias_float[(*bias_idx, slice(None), slice(None))]
        for q_start in range(0, q_view.size(1), query_block):
            q_stop = min(q_start + query_block, q_view.size(1))
            q_block = q_batch[q_start:q_stop]
            block_lse = torch.full(
                (q_stop - q_start,), -torch.inf, device=q.device, dtype=torch.float32
            )
            for k_start in range(0, k_view.size(1), key_block):
                k_stop = min(k_start + key_block, k_view.size(1))
                scores = torch.matmul(q_block, k_batch_t[:, k_start:k_stop]) * scale
                if bias_batch is not None:
                    scores = scores + bias_batch[q_start:q_stop, k_start:k_stop]
                if causal:
                    q_idx = torch.arange(q_start, q_stop, device=q.device)
                    k_idx = torch.arange(k_start, k_stop, device=q.device)
                    scores = scores.masked_fill(
                        q_idx[:, None] < k_idx[None, :], -torch.inf
                    )
                block_lse = torch.logaddexp(block_lse, torch.logsumexp(scores, dim=-1))
            lse[batch_idx, q_start:q_stop] = block_lse
    lse = lse.reshape(q.size()[:-1])
    if base2_lse:
        lse = lse * math.log2(math.e)
    return lse


def _attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    bias: torch.Tensor | None = None,
    base2_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=bias,
        is_causal=causal,
    )
    lse = _attention_reference_lse(
        q,
        k,
        causal=causal,
        bias=bias,
        base2_lse=base2_lse,
    )
    return out, lse


def _attention_baseline(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _attention_reference(q, k, v)


def _causal_attention_baseline(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _attention_reference(q, k, v, causal=True)


def _attention_output_baseline(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def _causal_attention_output_baseline(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)


def _biased_attention_baseline(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _attention_reference(q, k, v, bias=bias, base2_lse=False)


def _biased_attention_output_baseline(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)


def _baseline_unless_tpu(fn: Callable[..., Any]) -> Callable[..., Any] | None:
    # These baselines run SDPA plus a log-sum-exp loop in PyTorch; some of those
    # ops don't execute on the TPU/XLA backend, which hard-fails autotuning.
    # Fall back to Helion's default-config baseline on TPU (the path attention
    # used before custom baselines were added in #2833).
    return None if DEVICE.type == "tpu" else fn


@helion.kernel(
    # Static shapes provides a speedup for attention
    static_shapes=True,
    autotune_baseline_fn=_baseline_unless_tpu(_attention_baseline),
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=2e-2,
)
def attention(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes scaled dot-product attention.

    Implements the attention mechanism: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

    Args:
        q_in: Query tensor of shape [..., seq_len_q, head_dim]
        k_in: Key tensor of shape [..., seq_len_k, head_dim]
        v_in: Value tensor of shape [..., seq_len_k, head_dim]

    Returns:
        Output tensor of shape [..., seq_len_q, head_dim] and base-2 LSE
        tensor of shape [..., seq_len_q]
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
    # Trailing size-1 dim sidesteps a Helion block-size inflation: a 2-D
    # `lse[B*H, S]` with a tile-indexed leading dim forces an 8-element
    # sublane alignment on block_b, and adjust_block_size_constraints
    # max-propagates that requirement to Q/K/V/out (which share the
    # block_id), inflating their block_b and blowing up scoped VMEM.
    # Tracked in pytorch/helion#2842.
    lse = torch.empty(
        [q_view.size(0), m_dim, 1], device=q_in.device, dtype=torch.float32
    )
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
            # Keep scores in fp32 to match SDPA tolerances on bf16/fp16 inputs.
            # same as hl.dot(q, k, out_dtype=torch.float32)
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


@helion.kernel(
    # Static shapes provides a speedup for attention
    static_shapes=True,
    autotune_baseline_fn=_baseline_unless_tpu(_causal_attention_baseline),
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=2e-2,
)
def causal_attention(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes causal scaled dot-product attention.
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
    lse = torch.empty([q_view.size(0), m_dim], device=q_in.device, dtype=torch.float32)
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
        lse[tile_b, tile_m] = m_i + torch.log2(l_i)
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size()), lse.view(q_in.size()[:-1])


@helion.kernel(
    static_shapes=True,
    autotune_baseline_fn=_baseline_unless_tpu(_attention_output_baseline),
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=2e-2,
)
def attention_output(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    """
    Computes scaled dot-product attention and returns only the output tensor.
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


@helion.kernel(
    static_shapes=True,
    autotune_baseline_fn=_baseline_unless_tpu(_causal_attention_output_baseline),
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=2e-2,
)
def causal_attention_output(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    """
    Computes causal scaled dot-product attention and returns only the output tensor.
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


@helion.kernel(
    static_shapes=True,
    autotune_baseline_fn=_biased_attention_baseline,
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=2e-2,
)
def biased_attention(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes scaled dot-product attention with an additive score bias.

    The bias is exact-shape only: its leading dimensions must collapse to the
    same batch-head product as ``q_in``/``k_in``/``v_in``.
    """
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    bias_view = bias.reshape([-1, m_dim, n_dim])
    out = torch.empty_like(q_view)
    lse = torch.empty([q_view.size(0), m_dim], device=q_in.device, dtype=torch.float32)
    qk_scale = 1.0 / math.sqrt(head_dim)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            q_scaled = q * qk_scale
            k = k_view[tile_b, tile_n, :]
            qk = torch.bmm(q_scaled, k.transpose(1, 2), torch.float32)
            qk = qk + bias_view[tile_b, tile_m, tile_n]
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        lse[tile_b, tile_m] = m_i + torch.log(l_i)
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size()), lse.view(q_in.size()[:-1])


@helion.kernel(
    static_shapes=True,
    autotune_baseline_fn=_biased_attention_output_baseline,
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=2e-2,
)
def biased_attention_output(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Computes scaled dot-product attention with an additive score bias and
    returns only the output tensor.

    The bias is exact-shape only: its leading dimensions must collapse to the
    same batch-head product as ``q_in``/``k_in``/``v_in``.
    """
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    bias_view = bias.reshape([-1, m_dim, n_dim])
    out = torch.empty_like(q_view)
    qk_scale = 1.0 / math.sqrt(head_dim)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            q_scaled = q * qk_scale
            k = k_view[tile_b, tile_n, :]
            qk = torch.bmm(q_scaled, k.transpose(1, 2), torch.float32)
            qk = qk + bias_view[tile_b, tile_m, tile_n]
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


# %%
# Dynamic Shape Version
# ---------------------

# %%
# pyrefly: ignore [no-matching-overload]
attention_dynamic: object = helion.kernel(
    attention.fn,
    configs=attention.configs,
    static_shapes=False,
)
"""
Dynamic shape version of the attention kernel.
This version allows for variable input shapes at runtime.
"""


# %%
# Forward + Backward Implementation
# ---------------------------------


@helion.kernel(
    config=helion.Config(block_sizes=[128], num_warps=4),
    static_shapes=True,
)
def _attention_bwd_preprocess(
    o_in: torch.Tensor,
    do_in: torch.Tensor,
) -> torch.Tensor:
    head_dim = hl.specialize(o_in.size(-1))
    o = o_in.reshape(-1, head_dim)
    do = do_in.reshape(-1, head_dim)
    total_rows = o.size(0)
    delta = torch.empty(total_rows, device=o.device, dtype=torch.float32)
    for tile in hl.tile(total_rows):
        delta[tile] = torch.sum(
            o[tile, :].to(torch.float32) * do[tile, :].to(torch.float32), dim=-1
        )
    return delta.reshape(o_in.size()[:-1])


@helion.kernel(
    config=helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=2),
    static_shapes=True,
    autotune_accuracy_check=False,
)
def attention_backward(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    o_in: torch.Tensor,
    lse_in: torch.Tensor,
    do_in: torch.Tensor,
    delta_in: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    assert o_in.size(-2) == m_dim and o_in.size(-1) == head_dim
    assert do_in.size(-2) == m_dim and do_in.size(-1) == head_dim
    q = q_in.reshape(-1, head_dim)
    k = k_in.reshape(-1, head_dim)
    v = v_in.reshape(-1, head_dim)
    do = do_in.reshape(-1, head_dim)
    lse = lse_in.reshape(-1)
    delta = delta_in.reshape(-1)
    total_n_rows = k.size(0)
    LN2: float = 0.6931471824645996
    dq = torch.zeros((q.size(0), head_dim), device=q.device, dtype=torch.float32)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    block_m = hl.register_block_size(m_dim)
    block_n = hl.register_block_size(n_dim)
    assert m_dim % block_m == 0 and n_dim % block_n == 0
    for tile_n in hl.tile(total_n_rows, block_size=block_n):
        k_j = k[tile_n, :]
        v_j = v[tile_n, :]
        dv_acc = hl.zeros([tile_n, head_dim], dtype=torch.float32)
        dk_acc = hl.zeros([tile_n, head_dim], dtype=torch.float32)
        batch_idx = tile_n.begin // n_dim
        start_m = batch_idx * m_dim
        end_m = start_m + m_dim
        for tile_m in hl.tile(start_m, end_m, block_size=block_m):
            q_i = q[tile_m, :]
            do_i = do[tile_m, :]
            m_i = lse[tile_m]
            di = delta[tile_m]
            qk_t = hl.dot(k_j, q_i.T, out_dtype=torch.float32)
            p_t = torch.exp2(qk_t - m_i[None, :])
            dp_t = hl.dot(v_j, do_i.T, out_dtype=torch.float32)
            dv_acc = hl.dot(p_t.to(v.dtype), do_i, acc=dv_acc)
            ds_t = (p_t * (dp_t - di[None, :])).to(q.dtype)
            dq_acc = hl.dot(ds_t.T, k_j, out_dtype=torch.float32)
            hl.atomic_add(dq, [tile_m, slice(None)], dq_acc * LN2)
            dk_acc = hl.dot(ds_t, q_i, acc=dk_acc)
        dv[tile_n, :] = dv_acc.to(v.dtype)
        dk[tile_n, :] = (dk_acc * sm_scale).to(k.dtype)
    return dq.reshape(q_in.size()), dk.reshape(k_in.size()), dv.reshape(v_in.size())


class AttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx: Any,  # noqa: ANN401
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        o, lse = attention(q, k, v)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = math.sqrt(1.0 / q.size(-1))
        return o

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,  # noqa: ANN401
        do: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v, o, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        k_scaled = k * (sm_scale * 1.4426950408889634)
        delta = _attention_bwd_preprocess(o, do)
        return attention_backward(
            q,
            k_scaled,
            v,
            o,
            lse,
            do.contiguous(),
            delta,
            sm_scale,
        )


def attention_fwd_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    return AttentionFunction.apply(q, k, v)  # type: ignore[no-any-return]


# %%
# Testing Function
# ----------------


# %%
def test(
    z: int,
    h: int,
    n_ctx: int,
    head_dim: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cuda",
) -> None:
    """
    Test the attention kernel implementation against PyTorch's native attention functions.

    Args:
        z: Batch size
        h: Number of attention heads
        n_ctx: Sequence length (context size)
        head_dim: Dimension of each attention head
        dtype: Data type for the tensors
        device: Device to run the test on
    """
    q, k, v = [
        torch.randn((z, h, n_ctx, head_dim), dtype=dtype, device=device)
        for _ in range(3)
    ]

    def ref_attention(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Reference manual attention implementation"""
        p = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
        p = torch.softmax(p.float(), dim=-1).to(dtype)
        return torch.matmul(p, v)

    flex_compiled = cast(
        "Callable[..., torch.Tensor]", torch.compile(flex_attention, fullgraph=True)
    )

    baselines = {
        "torch": torch.nn.functional.scaled_dot_product_attention,
        "flex": flex_compiled,
        "ref": ref_attention,
    }
    if DEVICE.type == "tpu":
        del baselines["flex"]

    run_example(lambda *args: attention(*args)[0], baselines, (q, k, v))

    q_grad, k_grad, v_grad = [
        torch.randn(
            (z, h, n_ctx, head_dim), dtype=dtype, device=device
        ).requires_grad_()
        for _ in range(3)
    ]
    run_example(
        attention_fwd_bwd,
        torch.nn.functional.scaled_dot_product_attention,
        (q_grad, k_grad, v_grad),
        kernel_name="helion_autograd",
        rtol=1e-2,
        atol=1e-1,
        bwd=True,
    )


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """
    Main entry point that runs the attention kernel test with specific parameters.
    Tests with batch size 2, 32 heads, 1024 sequence length, and 64-dimensional heads using float16.
    """
    test(2, 32, 1024, 64, HALF_DTYPE, device=DEVICE)


if __name__ == "__main__":
    main()
