"""
Qwen3.8 Decode Layers (Pallas TPU)
==================================

One decode token through a Qwen3.8 GQA attention decoder layer, written as
ordinary Helion kernels for the Pallas backend at one TP8 rank's shapes.

The layer is ``h += o_proj(attn(rms(h)))`` followed by ``h += mlp(rms(h))``.
Each stage is a separate single-root kernel:

* :func:`rms_norm_zc` - zero-centered RMSNorm (scale by ``1 + w``)
* :func:`linear` / :func:`linear_residual` - ``x @ w`` (+ residual), f32 accumulate
* :func:`gqa_decode_attention` - q/k RMSNorm, partial split-half RoPE, KV-cache
  append at ``pos``, masked online-softmax attention, sigmoid output gate
* :func:`mlp_gate_up` - ``silu(x @ w_gate) * (x @ w_up)``
"""

# %%
from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
import helion.language as hl

# %%
# Kernels
# -------


# %%
@helion.kernel(backend="pallas", static_shapes=True)
def rms_norm_zc(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: hl.constexpr = 1e-6,  # pyrefly: ignore[bad-function-definition]
) -> torch.Tensor:
    """Zero-centered RMSNorm: normalize in f32, scale by ``1 + weight``."""
    m, _ = x.size()
    out = torch.empty_like(x)
    for tm in hl.tile(m):
        v = x[tm, :].float()
        # pyrefly: ignore[unsupported-operation]
        v = v * torch.rsqrt(torch.mean(v * v, -1, keepdim=True) + eps)
        out[tm, :] = (v * (1.0 + weight[None, :].float())).to(x.dtype)
    return out


# %%
@helion.kernel(backend="pallas", static_shapes=True)
def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x [M, K] @ weight [K, N]`` with f32 accumulation."""
    m, k = x.size()
    n = weight.size(1)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tn in hl.tile(n):
        acc = hl.zeros([m, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(x[:, tk], weight[tk, tn], acc=acc)
        out[:, tn] = acc.to(x.dtype)
    return out


# %%
@helion.kernel(backend="pallas", static_shapes=True)
def linear_residual(
    x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """``residual + x @ weight`` (the product is rounded before the add)."""
    m, k = x.size()
    n = weight.size(1)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tn in hl.tile(n):
        acc = hl.zeros([m, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(x[:, tk], weight[tk, tn], acc=acc)
        out[:, tn] = residual[:, tn] + acc.to(x.dtype)
    return out


# %%
@helion.kernel(backend="pallas", static_shapes=True)
def mlp_gate_up(
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor
) -> torch.Tensor:
    """``silu(x @ w_gate) * (x @ w_up)``; both products rounded to ``x.dtype``."""
    m, k = x.size()
    n = w_gate.size(1)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tn in hl.tile(n):
        g = hl.zeros([m, tn], dtype=torch.float32)
        u = hl.zeros([m, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            xk = x[:, tk]
            g = hl.dot(xk, w_gate[tk, tn], acc=g)
            u = hl.dot(xk, w_up[tk, tn], acc=u)
        g = g.to(x.dtype).float()
        out[:, tn] = (g * torch.sigmoid(g)).to(x.dtype) * u.to(x.dtype)
    return out


# %%
@helion.kernel(backend="pallas", static_shapes=True)
def gqa_decode_attention(
    q_gate: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    q_norm_w: torch.Tensor,
    k_norm_w: torch.Tensor,
    inv_freq: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: torch.Tensor,
    eps: hl.constexpr = 1e-6,  # pyrefly: ignore[bad-function-definition]
) -> torch.Tensor:
    """One decode token of grouped-query attention with an in-place KV cache.

    Args:
        q_gate: ``[kv_heads, group, 2 * head_dim]`` query | output gate per head
        k_proj, v_proj: ``[kv_heads, head_dim]`` new key / value
        q_norm_w, k_norm_w: ``[head_dim]`` zero-centered RMSNorm weights
        inv_freq: ``[rotary_dim // 2]`` f32 RoPE inverse frequencies
        k_cache, v_cache: ``[kv_heads, context, head_dim]``, row ``pos`` is written
        pos: ``[1]`` int32 decode position

    Returns:
        ``[kv_heads, group, head_dim]`` gated attention output.
    """
    nkv, group, two_d = q_gate.shape
    d = two_d // 2
    ctx = k_cache.size(1)
    half = inv_freq.size(0)
    out = torch.empty([nkv, group, d], dtype=q_gate.dtype, device=q_gate.device)
    sm_scale = d**-0.5
    for kvh in hl.grid(nkv):
        p = pos[0]
        ang = p.to(torch.float32) * inv_freq[:]
        cos = torch.cos(ang).to(q_gate.dtype)
        sin = torch.sin(ang).to(q_gate.dtype)
        # New key: RMSNorm, RoPE on the first 2 * half channels, append to cache.
        k = k_proj[kvh, :].float()
        # pyrefly: ignore[unsupported-operation]
        k = k * torch.rsqrt(torch.mean(k * k, -1, keepdim=True) + eps)
        k = (k * (1.0 + k_norm_w[:].float())).to(q_gate.dtype)
        k1 = k[:half]
        k2 = k[half : 2 * half]
        k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin, k[2 * half :]], dim=-1)
        k_cache[kvh, p, :] = k
        v_cache[kvh, p, :] = v_proj[kvh, :]
        # Queries of this kv head.
        q = q_gate[kvh, :, :d].float()
        # pyrefly: ignore[unsupported-operation]
        q = q * torch.rsqrt(torch.mean(q * q, -1, keepdim=True) + eps)
        q = (q * (1.0 + q_norm_w[None, :].float())).to(q_gate.dtype)
        q1 = q[:, :half]
        q2 = q[:, half : 2 * half]
        c = cos[None, :]
        s = sin[None, :]
        q = torch.cat([q1 * c - q2 * s, q2 * c + q1 * s, q[:, 2 * half :]], dim=-1)
        # Online softmax over cache positions <= pos (trip count bounded by pos).
        m_i = hl.full([group], float("-inf"), dtype=torch.float32)
        l_i = hl.zeros([group], dtype=torch.float32)
        acc = hl.zeros([group, d], dtype=torch.float32)
        for tt in hl.tile(p + 1):
            scores = hl.dot(q, k_cache[kvh, tt, :].T, out_dtype=torch.float32)
            scores = torch.where(tt.index[None, :] <= p, scores * sm_scale, -1e30)
            m_new = torch.maximum(m_i, torch.amax(scores, -1))
            alpha = torch.exp(m_i - m_new)
            probs = torch.exp(scores - m_new[:, None])
            l_i = l_i * alpha + torch.sum(probs, -1)
            acc = acc * alpha[:, None] + hl.dot(
                probs.to(q_gate.dtype), v_cache[kvh, tt, :], out_dtype=torch.float32
            )
            m_i = m_new
        o = (acc / l_i[:, None]).to(q_gate.dtype)
        gate = torch.sigmoid(q_gate[kvh, :, d:].float()).to(q_gate.dtype)
        out[kvh, :, :] = o * gate
    return out


# %%
@helion.kernel(backend="pallas", static_shapes=True)
def gdn_decode_core(
    qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv_w: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_w: torch.Tensor,
    conv_state: torch.Tensor,
    rec_state: torch.Tensor,
    eps: hl.constexpr = 1e-6,  # pyrefly: ignore[bad-function-definition]
) -> torch.Tensor:
    """One decode token of the gated delta rule with in-place conv/recurrent state.

    Args:
        qkv: ``[M, conv_dim]`` projected q|k|v channels (row 0 is the token)
        z: ``[M, v_heads * v_dim]`` output gate; b, a: ``[M, v_heads]``
        conv_w: ``[conv_dim, taps]`` depthwise causal conv weights
        a_log, dt_bias: ``[v_heads]``; norm_w: ``[v_dim]``
        conv_state: ``[conv_dim, taps]`` last ``taps`` inputs, shifted in place
        rec_state: ``[v_heads, v_dim, k_dim]`` f32 recurrent state, updated in place

    Returns:
        ``[v_heads, v_dim]`` gated-RMSNorm output.
    """
    conv_dim, _ = conv_state.shape
    hv, dv, dk = rec_state.shape
    key_dim = (conv_dim - hv * dv) // 2
    hk = key_dim // dk
    out = torch.empty([hv, dv], dtype=qkv.dtype, device=qkv.device)
    for _ in hl.grid(1):
        # Shift the new channels into the conv window; depthwise conv + SiLU.
        window = conv_state[:, :]
        window = torch.cat([window[:, 1:], qkv[0, :][:, None]], dim=-1)
        conv_state[:, :] = window
        mixed = torch.sum(window.float() * conv_w[:, :].float(), -1)
        mixed = mixed * torch.sigmoid(mixed)
        # Per-head L2-normalized q/k, expanded from k_heads to v_heads.
        q = mixed[:key_dim].reshape(hk, dk)
        k = mixed[key_dim : 2 * key_dim].reshape(hk, dk)
        v = mixed[2 * key_dim :].reshape(hv, dv)
        q = q * torch.rsqrt(torch.sum(q * q, -1, keepdim=True) + 1e-6) * dk**-0.5
        k = k * torch.rsqrt(torch.sum(k * k, -1, keepdim=True) + 1e-6)
        q = q[:, None, :].expand(hk, hv // hk, dk).reshape(hv, dk)
        k = k[:, None, :].expand(hk, hv // hk, dk).reshape(hv, dk)
        beta = torch.sigmoid(b[0, :].float()).to(qkv.dtype).float()
        gate = -torch.exp(a_log[:].float()) * torch.nn.functional.softplus(
            a[0, :].float() + dt_bias[:].float()
        )
        # Delta rule on the [value, key] state.
        state = rec_state[:, :, :] * torch.exp(gate)[:, None, None]
        prediction = torch.sum(k[:, None, :] * state, -1)
        state = state + ((v - prediction) * beta[:, None])[:, :, None] * k[:, None, :]
        rec_state[:, :, :] = state
        o = torch.sum(q[:, None, :] * state, -1).to(qkv.dtype).float()
        # Gated RMSNorm: normalize in f32, scale in bf16, gate with silu(z).
        # pyrefly: ignore[unsupported-operation]
        o = o * torch.rsqrt(torch.mean(o * o, -1, keepdim=True) + eps)
        o = (norm_w[None, :] * o.to(qkv.dtype)).float()
        zf = z[0, :].reshape(hv, dv).float()
        out[:, :] = (o * (zf * torch.sigmoid(zf))).to(qkv.dtype)
    return out


# %%
# Layer
# -----


# %%
def gqa_decoder_layer(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: torch.Tensor,
) -> torch.Tensor:
    """One GQA decoder layer for one token ``hidden [1, H]`` as a chain of
    single-root kernels.

    ``w`` holds ``input_norm, q, k, v, q_norm, k_norm, inv_freq, o, post_norm,
    gate, up, down``; ``k_cache``/``v_cache`` are updated in place at ``pos``.
    Row-wise kernels run on the token padded with zero rows to the 16-row
    bf16 sublane tile, since Mosaic cannot pipeline 1-row bf16 blocks.
    """
    nkv, _, d = k_cache.shape
    rows = 16
    h = torch.nn.functional.pad(hidden, (0, 0, 0, rows - 1))
    x = rms_norm_zc(h, w["input_norm"])
    q_gate = linear(x, w["q"])[0].view(nkv, -1, 2 * d)
    k = linear(x, w["k"])[0].view(nkv, d)
    v = linear(x, w["v"])[0].view(nkv, d)
    attn = gqa_decode_attention(
        q_gate, k, v, w["q_norm"], w["k_norm"], w["inv_freq"], k_cache, v_cache, pos
    )
    attn = torch.nn.functional.pad(attn.view(1, -1), (0, 0, 0, rows - 1))
    h = linear_residual(attn, w["o"], h)
    x = rms_norm_zc(h, w["post_norm"])
    return linear_residual(mlp_gate_up(x, w["gate"], w["up"]), w["down"], h)[:1]


# %%
# Reference and verification
# --------------------------


# %%
def _rms_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    v = x.float()
    v = v * torch.rsqrt(torch.mean(v * v, -1, keepdim=True) + eps)
    return (v * (1.0 + weight.float())).to(x.dtype)


def _linear_ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (x.float() @ weight.float()).to(x.dtype)


def _rope_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = cos.size(-1)
    x1, x2, rest = x[..., :half], x[..., half : 2 * half], x[..., 2 * half :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], dim=-1)


def gqa_decoder_layer_ref(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: torch.Tensor,
) -> torch.Tensor:
    """Eager PyTorch reference for :func:`gqa_decoder_layer` (updates the caches)."""
    nkv, ctx, d = k_cache.shape
    p = int(pos[0])
    x = _rms_ref(hidden, w["input_norm"])
    q_gate = _linear_ref(x, w["q"]).view(-1, 2 * d)
    query = _rms_ref(q_gate[:, :d], w["q_norm"])
    key = _rms_ref(_linear_ref(x, w["k"]).view(nkv, d), w["k_norm"])
    value = _linear_ref(x, w["v"]).view(nkv, d)
    ang = p * w["inv_freq"]
    cos, sin = torch.cos(ang).to(hidden.dtype), torch.sin(ang).to(hidden.dtype)
    query, key = _rope_ref(query, cos, sin), _rope_ref(key, cos, sin)
    k_cache[:, p] = key
    v_cache[:, p] = value
    grouped = query.view(nkv, -1, d).float()
    scores = grouped @ k_cache.float().transpose(1, 2) * d**-0.5
    mask = torch.arange(ctx, device=hidden.device) <= p
    scores = torch.where(mask, scores, float("-inf"))
    probs = torch.softmax(scores, -1).to(hidden.dtype)
    attn = (probs.float() @ v_cache.float()).to(hidden.dtype).view(-1, d)
    attn = attn * torch.sigmoid(q_gate[:, d:].float()).to(hidden.dtype)
    hidden = hidden + _linear_ref(attn.view(1, -1), w["o"])
    x = _rms_ref(hidden, w["post_norm"])
    g = _linear_ref(x, w["gate"]).float()
    act = (g * torch.sigmoid(g)).to(hidden.dtype) * _linear_ref(x, w["up"])
    return hidden + _linear_ref(act, w["down"])


# %%
def make_gqa_layer_inputs(
    dim: int = 5120,
    heads: int = 3,
    kv_heads: int = 1,
    head_dim: int = 256,
    rotary_dim: int = 64,
    intermediate: int = 2176,
    context: int = 2048,
    position: int = 100,
    rope_theta: float = 1e7,
    device: torch.device | str = DEVICE,
) -> tuple[
    torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Random weights/state for one TP8 rank of a Qwen3.8 GQA layer."""
    bf16 = torch.bfloat16

    def weight(k: int, n: int) -> torch.Tensor:
        return (torch.randn(k, n, device=device) * k**-0.5).to(bf16)

    def norm(n: int) -> torch.Tensor:
        return (torch.randn(n, device=device) * 0.1).to(bf16)

    half = rotary_dim // 2
    exponent = torch.arange(half, dtype=torch.float32, device=device) * 2 / rotary_dim
    w = {
        "input_norm": norm(dim),
        "q": weight(dim, heads * 2 * head_dim),
        "k": weight(dim, kv_heads * head_dim),
        "v": weight(dim, kv_heads * head_dim),
        "q_norm": norm(head_dim),
        "k_norm": norm(head_dim),
        "inv_freq": 1.0 / rope_theta**exponent,
        "o": weight(heads * head_dim, dim),
        "post_norm": norm(dim),
        "gate": weight(dim, intermediate),
        "up": weight(dim, intermediate),
        "down": weight(intermediate, dim),
    }
    hidden = torch.randn(1, dim, device=device).to(bf16)
    cache_shape = (kv_heads, context, head_dim)
    k_cache = torch.randn(cache_shape, device=device).to(bf16)
    v_cache = torch.randn(cache_shape, device=device).to(bf16)
    pos = torch.tensor([position], dtype=torch.int32, device=device)
    return hidden, w, k_cache, v_cache, pos


# %%
def main() -> None:
    """Run one GQA decoder layer and compare it against the eager reference."""
    torch.manual_seed(0)
    hidden, w, k_cache, v_cache, pos = make_gqa_layer_inputs()
    k_ref, v_ref = k_cache.clone(), v_cache.clone()
    expected = gqa_decoder_layer_ref(hidden, w, k_ref, v_ref, pos)
    result = gqa_decoder_layer(hidden, w, k_cache, v_cache, pos)
    torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(k_cache, k_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v_cache, v_ref)


if __name__ == "__main__":
    main()
