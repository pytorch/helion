"""
Qwen3.8 Hybrid Decoder Stack (Pallas TPU)
=========================================

One decode token through a stack of Qwen3.8 decoder layers written as ONE
Helion kernel.  Layers repeat with period four: three Gated-DeltaNet (GDN)
layers, then one full-attention GQA layer.  Every layer is

    h += mixer(rms(h, input_norm))
    h += mlp(rms(h, post_norm))

Weights are stacked per layer type (``gdn_*[3 * p + r]``, ``attn_*[p]``) and
per layer (norms and MLP, ``[4 * p + r]``); the recurrent states and KV caches
are stacked the same way and updated in place.  The host loop over periods is
folded by the megakernel into one device loop, and ``hl.static_range`` spells
out the layer types inside a period.
"""

# %%
from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
import helion.language as hl

# %%
# Kernel
# ------

# RMSNorm epsilon.
EPS = 1e-6


# %%
@helion.kernel(backend="pallas", static_shapes=True)
def qwen38_hybrid_stack(
    hidden: torch.Tensor,  # [M, H] bf16
    input_norm: torch.Tensor,  # [L, H]
    post_norm: torch.Tensor,  # [L, H]
    w_gate: torch.Tensor,  # [L, H, I]
    w_up: torch.Tensor,  # [L, H, I]
    w_down: torch.Tensor,  # [L, I, H]
    gdn_qkv: torch.Tensor,  # [Lg, H, conv_dim]
    gdn_z: torch.Tensor,  # [Lg, H, v_heads * v_dim]
    gdn_b: torch.Tensor,  # [Lg, H, v_heads]
    gdn_a: torch.Tensor,  # [Lg, H, v_heads]
    gdn_conv: torch.Tensor,  # [Lg, conv_dim, taps]
    gdn_a_log: torch.Tensor,  # [Lg, v_heads]
    gdn_dt_bias: torch.Tensor,  # [Lg, v_heads]
    gdn_norm: torch.Tensor,  # [Lg, v_dim]
    gdn_out: torch.Tensor,  # [Lg, v_heads * v_dim, H]
    attn_q: torch.Tensor,  # [La, H, heads * 2 * d]  (query | gate per head)
    attn_k: torch.Tensor,  # [La, H, kv_heads * d]
    attn_v: torch.Tensor,  # [La, H, kv_heads * d]
    attn_q_norm: torch.Tensor,  # [La, d]
    attn_k_norm: torch.Tensor,  # [La, d]
    attn_o: torch.Tensor,  # [La, heads * d, H]
    inv_freq: torch.Tensor,  # [rotary / 2] f32
    conv_state: torch.Tensor,  # [Lg, conv_dim, taps] bf16, in place
    rec_state: torch.Tensor,  # [Lg, v_heads, v_dim, k_dim] f32, in place
    k_cache: torch.Tensor,  # [La, kv_heads, ctx, d] bf16, row pos written in place
    v_cache: torch.Tensor,  # [La, kv_heads, ctx, d]
    pos: torch.Tensor,  # [1] int32
) -> torch.Tensor:
    m, h = hidden.shape
    inter = w_gate.size(2)
    _, conv_dim, _ = conv_state.shape
    _, hv, dv, dk = rec_state.shape
    key_dim = (conv_dim - hv * dv) // 2
    hk = key_dim // dk
    periods, nkv, ctx, d = k_cache.shape
    nqg = attn_q.size(2)
    group = nqg // (2 * d) // nkv
    half = inv_freq.size(0)
    dt = hidden.dtype
    sm_scale = d**-0.5
    res = torch.empty([m, h], dtype=dt, device=hidden.device)
    x = torch.empty([m, h], dtype=dt, device=hidden.device)
    qkv = torch.empty([m, conv_dim], dtype=dt, device=hidden.device)
    zb = torch.empty([m, hv * dv], dtype=dt, device=hidden.device)
    bb = torch.empty([m, hv], dtype=dt, device=hidden.device)
    ab = torch.empty([m, hv], dtype=dt, device=hidden.device)
    core = torch.empty([m, hv * dv], dtype=dt, device=hidden.device)
    q_buf = torch.empty([m, nqg], dtype=dt, device=hidden.device)
    k_buf = torch.empty([m, nkv * d], dtype=dt, device=hidden.device)
    v_buf = torch.empty([m, nkv * d], dtype=dt, device=hidden.device)
    attn = torch.empty([m, nkv * group * d], dtype=dt, device=hidden.device)
    act = torch.empty([m, inter], dtype=dt, device=hidden.device)
    out = torch.empty_like(hidden)
    for tm in hl.tile(m):
        res[tm, :] = hidden[tm, :]
    for p in range(periods):
        for r in hl.static_range(4):
            # input RMSNorm (every layer)
            for tm in hl.tile(m):
                v = res[tm, :].float()
                v = v * torch.rsqrt(torch.mean(v * v, -1, keepdim=True) + EPS)
                w_in = input_norm[4 * p + r, :][None, :].float()
                x[tm, :] = (v * (1.0 + w_in)).to(dt)
            if r == 3:
                # full attention (GQA) layer p
                for tm, tn in hl.tile([m, nqg]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(h):
                        acc = hl.dot(x[tm, tk], attn_q[p, tk, tn], acc=acc)
                    q_buf[tm, tn] = acc.to(dt)
                for tm, tn in hl.tile([m, nkv * d]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(h):
                        acc = hl.dot(x[tm, tk], attn_k[p, tk, tn], acc=acc)
                    k_buf[tm, tn] = acc.to(dt)
                for tm, tn in hl.tile([m, nkv * d]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(h):
                        acc = hl.dot(x[tm, tk], attn_v[p, tk, tn], acc=acc)
                    v_buf[tm, tn] = acc.to(dt)
                # attention for the decode token (row 0) over all kv heads
                for _ in hl.grid(1):
                    t = pos[0]
                    ang = t.to(torch.float32) * inv_freq[:]
                    cos = torch.cos(ang).to(dt)[None, :]
                    sin = torch.sin(ang).to(dt)[None, :]
                    k = k_buf[0, :].reshape(nkv, d).float()
                    k = k * torch.rsqrt(torch.mean(k * k, -1, keepdim=True) + EPS)
                    k = (k * (1.0 + attn_k_norm[p, :][None, :].float())).to(dt)
                    k1 = k[:, :half]
                    k2 = k[:, half : 2 * half]
                    k = torch.cat(
                        [k1 * cos - k2 * sin, k2 * cos + k1 * sin, k[:, 2 * half :]],
                        dim=-1,
                    )
                    k_cache[p, :, t, :] = k
                    v_cache[p, :, t, :] = v_buf[0, :].reshape(nkv, d)
                    qg = q_buf[0, :].reshape(nkv, group, 2 * d)
                    q = qg[:, :, :d].float()
                    q = q * torch.rsqrt(torch.mean(q * q, -1, keepdim=True) + EPS)
                    q_w = attn_q_norm[p, :][None, None, :].float()
                    q = (q * (1.0 + q_w)).to(dt)
                    q1 = q[:, :, :half]
                    q2 = q[:, :, half : 2 * half]
                    c = cos[None, :, :]
                    s = sin[None, :, :]
                    q = torch.cat(
                        [q1 * c - q2 * s, q2 * c + q1 * s, q[:, :, 2 * half :]], dim=-1
                    )
                    m_i = hl.full([nkv, group], float("-inf"), dtype=torch.float32)
                    l_i = hl.zeros([nkv, group], dtype=torch.float32)
                    acc_o = hl.zeros([nkv, group, d], dtype=torch.float32)
                    for tt in hl.tile(t + 1):
                        scores = hl.dot(
                            q,
                            k_cache[p, :, tt, :].transpose(1, 2),
                            out_dtype=torch.float32,
                        )
                        scores = torch.where(
                            tt.index[None, None, :] <= t, scores * sm_scale, -1e30
                        )
                        m_new = torch.maximum(m_i, torch.amax(scores, -1))
                        alpha = torch.exp(m_i - m_new)
                        probs = torch.exp(scores - m_new[:, :, None])
                        l_i = l_i * alpha + torch.sum(probs, -1)
                        acc_o = acc_o * alpha[:, :, None] + hl.dot(
                            probs.to(dt), v_cache[p, :, tt, :], out_dtype=torch.float32
                        )
                        m_i = m_new
                    o = (acc_o / l_i[:, :, None]).to(dt)
                    gate = torch.sigmoid(qg[:, :, d:].float()).to(dt)
                    attn[0, :] = (o * gate).reshape(nkv * group * d)
                # o-proj + residual
                for tm, tn in hl.tile([m, h]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(nkv * group * d):
                        acc = hl.dot(attn[tm, tk], attn_o[p, tk, tn], acc=acc)
                    res[tm, tn] = res[tm, tn] + acc.to(dt)
            else:
                # Gated-DeltaNet layer 3 * p + r
                for tm, tn in hl.tile([m, conv_dim]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(h):
                        w_tile = gdn_qkv[3 * p + r, tk, tn]
                        acc = hl.dot(x[tm, tk], w_tile, acc=acc)
                    qkv[tm, tn] = acc.to(dt)
                for tm, tn in hl.tile([m, hv * dv]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(h):
                        w_tile = gdn_z[3 * p + r, tk, tn]
                        acc = hl.dot(x[tm, tk], w_tile, acc=acc)
                    zb[tm, tn] = acc.to(dt)
                for tm in hl.tile(m):
                    xm = x[tm, :]
                    w_b = gdn_b[3 * p + r, :, :]
                    w_a = gdn_a[3 * p + r, :, :]
                    bb[tm, :] = hl.dot(xm, w_b, out_dtype=torch.float32).to(dt)
                    ab[tm, :] = hl.dot(xm, w_a, out_dtype=torch.float32).to(dt)
                # causal conv + gated delta rule for the decode token (row 0)
                for _ in hl.grid(1):
                    window = conv_state[3 * p + r, :, :]
                    window = torch.cat([window[:, 1:], qkv[0, :][:, None]], dim=-1)
                    conv_state[3 * p + r, :, :] = window
                    taps = gdn_conv[3 * p + r, :, :].float()
                    mixed = torch.sum(window.float() * taps, -1)
                    mixed = mixed * torch.sigmoid(mixed)
                    q = mixed[:key_dim].reshape(hk, dk)
                    k = mixed[key_dim : 2 * key_dim].reshape(hk, dk)
                    v = mixed[2 * key_dim :].reshape(hv, dv)
                    q = q * torch.rsqrt(torch.sum(q * q, -1, keepdim=True) + 1e-6)
                    q = q * dk**-0.5
                    k = k * torch.rsqrt(torch.sum(k * k, -1, keepdim=True) + 1e-6)
                    q = q[:, None, :].expand(hk, hv // hk, dk).reshape(hv, dk)
                    k = k[:, None, :].expand(hk, hv // hk, dk).reshape(hv, dk)
                    beta = torch.sigmoid(bb[0, :].float()).to(dt).float()
                    a_log = gdn_a_log[3 * p + r, :].float()
                    dt_bias = gdn_dt_bias[3 * p + r, :].float()
                    decay = -torch.exp(a_log) * torch.nn.functional.softplus(
                        ab[0, :].float() + dt_bias
                    )
                    state = rec_state[3 * p + r, :, :, :]
                    state = state * torch.exp(decay)[:, None, None]
                    prediction = torch.sum(k[:, None, :] * state, -1)
                    delta = ((v - prediction) * beta[:, None])[:, :, None]
                    state = state + delta * k[:, None, :]
                    rec_state[3 * p + r, :, :, :] = state
                    o = torch.sum(q[:, None, :] * state, -1).to(dt).float()
                    o = o * torch.rsqrt(torch.mean(o * o, -1, keepdim=True) + EPS)
                    g_w = gdn_norm[3 * p + r, :][None, :]
                    o = (g_w * o.to(dt)).float()
                    zf = zb[0, :].reshape(hv, dv).float()
                    core[0, :] = (o * (zf * torch.sigmoid(zf))).to(dt).reshape(hv * dv)
                # out-proj + residual
                for tm, tn in hl.tile([m, h]):
                    acc = hl.zeros([tm, tn], dtype=torch.float32)
                    for tk in hl.tile(hv * dv):
                        w_tile = gdn_out[3 * p + r, tk, tn]
                        acc = hl.dot(core[tm, tk], w_tile, acc=acc)
                    res[tm, tn] = res[tm, tn] + acc.to(dt)
            # post RMSNorm + MLP (every layer)
            for tm in hl.tile(m):
                v = res[tm, :].float()
                v = v * torch.rsqrt(torch.mean(v * v, -1, keepdim=True) + EPS)
                w_post = post_norm[4 * p + r, :][None, :].float()
                x[tm, :] = (v * (1.0 + w_post)).to(dt)
            for tm, tn in hl.tile([m, inter]):
                g = hl.zeros([tm, tn], dtype=torch.float32)
                u = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(h):
                    xk = x[tm, tk]
                    g = hl.dot(xk, w_gate[4 * p + r, tk, tn], acc=g)
                    u = hl.dot(xk, w_up[4 * p + r, tk, tn], acc=u)
                gf = g.to(dt).float()
                act[tm, tn] = (gf * torch.sigmoid(gf)).to(dt) * u.to(dt)
            for tm, tn in hl.tile([m, h]):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(inter):
                    acc = hl.dot(act[tm, tk], w_down[4 * p + r, tk, tn], acc=acc)
                res[tm, tn] = res[tm, tn] + acc.to(dt)
    for tm in hl.tile(m):
        out[tm, :] = res[tm, :]
    return out


# %%
# Reference
# ---------


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


def _l2norm_ref(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(torch.sum(x * x, -1, keepdim=True) + 1e-6)


def _gdn_ref(
    x: torch.Tensor, w: dict[str, torch.Tensor], i: int, s: dict[str, torch.Tensor]
) -> torch.Tensor:
    """One GDN token mixer for layer ``i``; updates the conv/recurrent state."""
    _, hv, dv, dk = s["rec_state"].shape
    conv_dim = s["conv_state"].size(1)
    key_dim = (conv_dim - hv * dv) // 2
    hk = key_dim // dk
    dt = x.dtype
    qkv = _linear_ref(x, w["gdn_qkv"][i])[0]
    z = _linear_ref(x, w["gdn_z"][i])[0]
    b = _linear_ref(x, w["gdn_b"][i])[0]
    a = _linear_ref(x, w["gdn_a"][i])[0]
    window = torch.cat([s["conv_state"][i][:, 1:], qkv[:, None]], dim=-1)
    s["conv_state"][i] = window
    mixed = torch.sum(window.float() * w["gdn_conv"][i].float(), -1)
    mixed = mixed * torch.sigmoid(mixed)
    q = _l2norm_ref(mixed[:key_dim].view(hk, dk)) * dk**-0.5
    k = _l2norm_ref(mixed[key_dim : 2 * key_dim].view(hk, dk))
    v = mixed[2 * key_dim :].view(hv, dv)
    q = q.repeat_interleave(hv // hk, 0)
    k = k.repeat_interleave(hv // hk, 0)
    beta = torch.sigmoid(b.float()).to(dt).float()
    decay = -torch.exp(w["gdn_a_log"][i].float()) * torch.nn.functional.softplus(
        a.float() + w["gdn_dt_bias"][i].float()
    )
    state = s["rec_state"][i] * torch.exp(decay)[:, None, None]
    prediction = torch.einsum("hk,hvk->hv", k, state)
    state = state + ((v - prediction) * beta[:, None])[:, :, None] * k[:, None, :]
    s["rec_state"][i] = state
    o = torch.einsum("hk,hvk->hv", q, state).to(dt).float()
    o = o * torch.rsqrt(torch.mean(o * o, -1, keepdim=True) + 1e-6)
    o = (w["gdn_norm"][i] * o.to(dt)).float()
    zf = z.view(hv, dv).float()
    o = (o * (zf * torch.sigmoid(zf))).to(dt)
    return _linear_ref(o.view(1, -1), w["gdn_out"][i])


def _attention_ref(
    x: torch.Tensor,
    w: dict[str, torch.Tensor],
    i: int,
    s: dict[str, torch.Tensor],
    pos: torch.Tensor,
) -> torch.Tensor:
    """One GQA token mixer for layer ``i``; writes row ``pos`` of the caches."""
    _, nkv, ctx, d = s["k_cache"].shape
    t = int(pos[0])
    q_gate = _linear_ref(x, w["attn_q"][i]).view(-1, 2 * d)
    query = _rms_ref(q_gate[:, :d], w["attn_q_norm"][i])
    key = _rms_ref(_linear_ref(x, w["attn_k"][i]).view(nkv, d), w["attn_k_norm"][i])
    value = _linear_ref(x, w["attn_v"][i]).view(nkv, d)
    ang = t * w["inv_freq"]
    cos, sin = torch.cos(ang).to(x.dtype), torch.sin(ang).to(x.dtype)
    query, key = _rope_ref(query, cos, sin), _rope_ref(key, cos, sin)
    k_cache, v_cache = s["k_cache"][i], s["v_cache"][i]
    k_cache[:, t] = key
    v_cache[:, t] = value
    grouped = query.view(nkv, -1, d).float()
    scores = grouped @ k_cache.float().transpose(1, 2) * d**-0.5
    mask = torch.arange(ctx, device=x.device) <= t
    scores = torch.where(mask, scores, float("-inf"))
    probs = torch.softmax(scores, -1).to(x.dtype)
    attn = (probs.float() @ v_cache.float()).to(x.dtype).view(-1, d)
    attn = attn * torch.sigmoid(q_gate[:, d:].float()).to(x.dtype)
    return _linear_ref(attn.view(1, -1), w["attn_o"][i])


def qwen38_hybrid_stack_ref(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    s: dict[str, torch.Tensor],
    pos: torch.Tensor,
    period: int = 4,
) -> torch.Tensor:
    """Eager PyTorch reference for :func:`qwen38_hybrid_stack` (updates ``s``)."""
    for layer in range(w["input_norm"].size(0)):
        p, r = divmod(layer, period)
        x = _rms_ref(hidden, w["input_norm"][layer])
        if r == period - 1:
            hidden = hidden + _attention_ref(x, w, p, s, pos)
        else:
            hidden = hidden + _gdn_ref(x, w, (period - 1) * p + r, s)
        x = _rms_ref(hidden, w["post_norm"][layer])
        g = _linear_ref(x, w["w_gate"][layer]).float()
        act = (g * torch.sigmoid(g)).to(hidden.dtype) * _linear_ref(x, w["w_up"][layer])
        hidden = hidden + _linear_ref(act, w["w_down"][layer])
    return hidden


# %%
WEIGHT_NAMES = (
    "input_norm",
    "post_norm",
    "w_gate",
    "w_up",
    "w_down",
    "gdn_qkv",
    "gdn_z",
    "gdn_b",
    "gdn_a",
    "gdn_conv",
    "gdn_a_log",
    "gdn_dt_bias",
    "gdn_norm",
    "gdn_out",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_q_norm",
    "attn_k_norm",
    "attn_o",
    "inv_freq",
)
STATE_NAMES = ("conv_state", "rec_state", "k_cache", "v_cache")


def make_hybrid_stack_inputs(
    layers: int = 4,
    period: int = 4,
    dim: int = 5120,
    intermediate: int = 2176,
    k_heads: int = 2,
    k_dim: int = 128,
    v_heads: int = 6,
    v_dim: int = 128,
    conv_size: int = 4,
    heads: int = 3,
    kv_heads: int = 1,
    head_dim: int = 256,
    rotary_dim: int = 64,
    context: int = 2048,
    position: int = 100,
    rope_theta: float = 1e7,
    device: torch.device | str = DEVICE,
) -> tuple[
    torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor
]:
    """Random stacked weights/state for one TP8 rank of a Qwen3.8 hybrid stack."""
    bf16 = torch.bfloat16
    n_attn = layers // period
    n_gdn = layers - n_attn
    conv_dim = 2 * k_heads * k_dim + v_heads * v_dim

    def weight(n: int, k: int, out: int) -> torch.Tensor:
        return (torch.randn(n, k, out, device=device) * k**-0.5).to(bf16)

    def vec(n: int, k: int, scale: float = 0.1, offset: float = 0.0) -> torch.Tensor:
        return (offset + torch.randn(n, k, device=device) * scale).to(bf16)

    half = rotary_dim // 2
    exponent = torch.arange(half, dtype=torch.float32, device=device) * 2 / rotary_dim
    w = {
        "input_norm": vec(layers, dim),
        "post_norm": vec(layers, dim),
        "w_gate": weight(layers, dim, intermediate),
        "w_up": weight(layers, dim, intermediate),
        "w_down": weight(layers, intermediate, dim),
        "gdn_qkv": weight(n_gdn, dim, conv_dim),
        "gdn_z": weight(n_gdn, dim, v_heads * v_dim),
        "gdn_b": weight(n_gdn, dim, v_heads),
        "gdn_a": weight(n_gdn, dim, v_heads),
        "gdn_conv": weight(n_gdn, conv_dim, conv_size),
        "gdn_a_log": vec(n_gdn, v_heads, 0.5),
        "gdn_dt_bias": vec(n_gdn, v_heads, 0.5),
        "gdn_norm": vec(n_gdn, v_dim, 0.1, 1.0),
        "gdn_out": weight(n_gdn, v_heads * v_dim, dim),
        "attn_q": weight(n_attn, dim, heads * 2 * head_dim),
        "attn_k": weight(n_attn, dim, kv_heads * head_dim),
        "attn_v": weight(n_attn, dim, kv_heads * head_dim),
        "attn_q_norm": vec(n_attn, head_dim),
        "attn_k_norm": vec(n_attn, head_dim),
        "attn_o": weight(n_attn, heads * head_dim, dim),
        "inv_freq": 1.0 / rope_theta**exponent,
    }
    s = {
        "conv_state": torch.randn(n_gdn, conv_dim, conv_size, device=device).to(bf16),
        "rec_state": torch.randn(n_gdn, v_heads, v_dim, k_dim, device=device) * 0.1,
        "k_cache": torch.randn(n_attn, kv_heads, context, head_dim, device=device).to(
            bf16
        ),
        "v_cache": torch.randn(n_attn, kv_heads, context, head_dim, device=device).to(
            bf16
        ),
    }
    hidden = torch.randn(1, dim, device=device).to(bf16)
    pos = torch.tensor([position], dtype=torch.int32, device=device)
    return hidden, w, s, pos


def hybrid_stack_args(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    s: dict[str, torch.Tensor],
    pos: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Positional arguments of :func:`qwen38_hybrid_stack`."""
    return (
        hidden,
        *(w[n] for n in WEIGHT_NAMES),
        *(s[n] for n in STATE_NAMES),
        pos,
    )


# %%
def main() -> None:
    """Run an 8-layer hybrid stack and compare it against the eager reference."""
    torch.manual_seed(0)
    hidden, w, s, pos = make_hybrid_stack_inputs(layers=8)
    s_ref = {k: v.clone() for k, v in s.items()}
    expected = qwen38_hybrid_stack_ref(hidden, w, s_ref, pos)
    result = qwen38_hybrid_stack(*hybrid_stack_args(hidden, w, s, pos))
    # bf16 rounding differences compound over the layers.
    torch.testing.assert_close(result, expected, atol=1e-1, rtol=5e-2)
    for name in ("conv_state", "k_cache", "v_cache"):
        torch.testing.assert_close(s[name], s_ref[name], atol=5e-2, rtol=2e-2)
    torch.testing.assert_close(s["rec_state"], s_ref["rec_state"], atol=5e-3, rtol=5e-3)


if __name__ == "__main__":
    main()
