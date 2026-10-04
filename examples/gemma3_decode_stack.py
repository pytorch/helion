"""
Gemma-3 Decoder Stack (Pallas TPU)
==================================

One decode token through a stack of Gemma-3-style decoder layers written as ONE
Helion kernel.  Layers repeat with a static pattern of ``period - 1``
sliding-window (local) layers and one global layer.  Every layer is

    h += post_attn_norm(o_proj(attention(rope(qk_norm(qkv_proj(rms(h)))))))
    h += post_mlp_norm(down(gelu_tanh(gate(x)) * up(x))),  x = rms(h, pre_mlp_norm)

Local layers attend to the last ``window`` positions (applied as a mask over
the whole cache) with their own RoPE frequencies; global layers attend to every
position up to ``pos``.  Queries and keys are RMS-normalized per head before
RoPE.  All RMSNorms scale by ``1 + weight``.  The host loop over periods is
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
def gemma3_decode_stack(
    hidden: torch.Tensor,  # [M, H] bf16
    input_norm: torch.Tensor,  # [L, H]
    post_attn_norm: torch.Tensor,  # [L, H]
    pre_mlp_norm: torch.Tensor,  # [L, H]
    post_mlp_norm: torch.Tensor,  # [L, H]
    w_qkv: torch.Tensor,  # [L, H, (heads + 2 * kv_heads) * d]
    q_norm: torch.Tensor,  # [L, d]
    k_norm: torch.Tensor,  # [L, d]
    w_o: torch.Tensor,  # [L, heads * d, H]
    w_gate: torch.Tensor,  # [L, H, I]
    w_up: torch.Tensor,  # [L, H, I]
    w_down: torch.Tensor,  # [L, I, H]
    inv_freq_local: torch.Tensor,  # [d / 2] f32
    inv_freq_global: torch.Tensor,  # [d / 2] f32
    k_cache: torch.Tensor,  # [L, kv_heads, ctx, d] bf16, row pos written in place
    v_cache: torch.Tensor,  # [L, kv_heads, ctx, d]
    pos: torch.Tensor,  # [1] int32
    period: int,  # both ints are passed as hl.constexpr
    window: int,
) -> torch.Tensor:
    m, h = hidden.shape
    layers, nkv, ctx, d = k_cache.shape
    nq = w_o.size(1) // d
    group = nq // nkv
    half = d // 2
    nqkv = w_qkv.size(2)
    inter = w_gate.size(2)
    dt = hidden.dtype
    sm_scale = d**-0.5
    res = torch.empty([m, h], dtype=dt, device=hidden.device)
    x = torch.empty([m, h], dtype=dt, device=hidden.device)
    qkv = torch.empty([m, nqkv], dtype=dt, device=hidden.device)
    attn = torch.empty([m, nq * d], dtype=dt, device=hidden.device)
    branch = torch.empty([m, h], dtype=dt, device=hidden.device)
    act = torch.empty([m, inter], dtype=dt, device=hidden.device)
    out = torch.empty_like(hidden)
    for tm in hl.tile(m):
        res[tm, :] = hidden[tm, :]
    for p in range(layers // period):
        for r in hl.static_range(period):
            # pre-attention RMSNorm
            for tm in hl.tile(m):
                v = res[tm, :].float()
                v = v * torch.rsqrt(torch.mean(v * v, -1, keepdim=True) + EPS)
                w_in = input_norm[period * p + r, :][None, :].float()
                x[tm, :] = (v * (1.0 + w_in)).to(dt)
            for tm, tn in hl.tile([m, nqkv]):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(h):
                    acc = hl.dot(x[tm, tk], w_qkv[period * p + r, tk, tn], acc=acc)
                qkv[tm, tn] = acc.to(dt)
            # attention for the decode token (row 0)
            for _ in hl.grid(1):
                t = pos[0]
                if r == period - 1:
                    ang = t.to(torch.float32) * inv_freq_global[:]
                    first = t - ctx
                else:
                    ang = t.to(torch.float32) * inv_freq_local[:]
                    first = t - window
                cos = torch.cos(ang).to(dt)[None, :]
                sin = torch.sin(ang).to(dt)[None, :]
                row = qkv[0, :]
                k = row[nq * d : (nq + nkv) * d].reshape(nkv, d).float()
                k = k * torch.rsqrt(torch.mean(k * k, -1, keepdim=True) + EPS)
                k_w = k_norm[period * p + r, :][None, :].float()
                k = (k * (1.0 + k_w)).to(dt)
                k1 = k[:, :half]
                k2 = k[:, half:]
                k_cache[period * p + r, :, t, :] = torch.cat(
                    [k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1
                )
                v_cache[period * p + r, :, t, :] = row[(nq + nkv) * d :].reshape(nkv, d)
                q = row[: nq * d].reshape(nkv, group, d).float()
                q = q * torch.rsqrt(torch.mean(q * q, -1, keepdim=True) + EPS)
                q_w = q_norm[period * p + r, :][None, None, :].float()
                q = (q * (1.0 + q_w)).to(dt)
                q1 = q[:, :, :half]
                q2 = q[:, :, half:]
                c = cos[None, :, :]
                s = sin[None, :, :]
                q = torch.cat([q1 * c - q2 * s, q2 * c + q1 * s], dim=-1)
                m_i = hl.full([nkv, group], float("-inf"), dtype=torch.float32)
                l_i = hl.zeros([nkv, group], dtype=torch.float32)
                acc_o = hl.zeros([nkv, group, d], dtype=torch.float32)
                for tt in hl.tile(ctx):
                    scores = hl.dot(
                        q,
                        k_cache[period * p + r, :, tt, :].transpose(1, 2),
                        out_dtype=torch.float32,
                    )
                    idx = tt.index[None, None, :]
                    visible = (idx <= t) & (idx > first)
                    scores = torch.where(visible, scores * sm_scale, -1e30)
                    m_new = torch.maximum(m_i, torch.amax(scores, -1))
                    alpha = torch.exp(m_i - m_new)
                    probs = torch.exp(scores - m_new[:, :, None])
                    l_i = l_i * alpha + torch.sum(probs, -1)
                    acc_o = acc_o * alpha[:, :, None] + hl.dot(
                        probs.to(dt),
                        v_cache[period * p + r, :, tt, :],
                        out_dtype=torch.float32,
                    )
                    m_i = m_new
                attn[0, :] = (acc_o / l_i[:, :, None]).to(dt).reshape(nq * d)
            for tm, tn in hl.tile([m, h]):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(nq * d):
                    acc = hl.dot(attn[tm, tk], w_o[period * p + r, tk, tn], acc=acc)
                branch[tm, tn] = acc.to(dt)
            # post-attention RMSNorm + residual, then the pre-MLP RMSNorm
            for tm in hl.tile(m):
                v = branch[tm, :].float()
                v = v * torch.rsqrt(torch.mean(v * v, -1, keepdim=True) + EPS)
                w_post = post_attn_norm[period * p + r, :][None, :].float()
                hs = res[tm, :] + (v * (1.0 + w_post)).to(dt)
                res[tm, :] = hs
                v = hs.float()
                v = v * torch.rsqrt(torch.mean(v * v, -1, keepdim=True) + EPS)
                w_pre = pre_mlp_norm[period * p + r, :][None, :].float()
                x[tm, :] = (v * (1.0 + w_pre)).to(dt)
            for tm, tn in hl.tile([m, inter]):
                g = hl.zeros([tm, tn], dtype=torch.float32)
                u = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(h):
                    xk = x[tm, tk]
                    g = hl.dot(xk, w_gate[period * p + r, tk, tn], acc=g)
                    u = hl.dot(xk, w_up[period * p + r, tk, tn], acc=u)
                gelu = torch.nn.functional.gelu(g.to(dt), approximate="tanh")
                act[tm, tn] = gelu * u.to(dt)
            for tm, tn in hl.tile([m, h]):
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(inter):
                    acc = hl.dot(act[tm, tk], w_down[period * p + r, tk, tn], acc=acc)
                branch[tm, tn] = acc.to(dt)
            # post-MLP RMSNorm + residual
            for tm in hl.tile(m):
                v = branch[tm, :].float()
                v = v * torch.rsqrt(torch.mean(v * v, -1, keepdim=True) + EPS)
                w_post = post_mlp_norm[period * p + r, :][None, :].float()
                res[tm, :] = res[tm, :] + (v * (1.0 + w_post)).to(dt)
    for tm in hl.tile(m):
        out[tm, :] = res[tm, :]
    return out


# %%
# Reference
# ---------


# %%
def _rms_ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    v = x.float()
    v = v * torch.rsqrt(torch.mean(v * v, -1, keepdim=True) + EPS)
    return (v * (1.0 + weight.float())).to(x.dtype)


def _linear_ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (x.float() @ weight.float()).to(x.dtype)


def _rope_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def gemma3_layer_ref(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    s: dict[str, torch.Tensor],
    layer: int,
    pos: torch.Tensor,
    is_global: bool,
    window: int,
) -> torch.Tensor:
    """One Gemma-3 decoder layer on ``hidden`` [1, H] (updates ``s``)."""
    _, nkv, ctx, d = s["k_cache"].shape
    nq = w["w_o"].size(1) // d
    dt = hidden.dtype
    t = int(pos[0])
    inv_freq = w["inv_freq_global"] if is_global else w["inv_freq_local"]
    ang = t * inv_freq
    cos, sin = torch.cos(ang).to(dt), torch.sin(ang).to(dt)
    x = _rms_ref(hidden, w["input_norm"][layer])
    qkv = _linear_ref(x, w["w_qkv"][layer])[0]
    query = _rms_ref(qkv[: nq * d].view(nq, d), w["q_norm"][layer])
    key = _rms_ref(qkv[nq * d : (nq + nkv) * d].view(nkv, d), w["k_norm"][layer])
    query, key = _rope_ref(query, cos, sin), _rope_ref(key, cos, sin)
    k_cache, v_cache = s["k_cache"][layer], s["v_cache"][layer]
    k_cache[:, t] = key
    v_cache[:, t] = qkv[(nq + nkv) * d :].view(nkv, d)
    grouped = query.view(nkv, -1, d).float()
    scores = grouped @ k_cache.float().transpose(1, 2) * d**-0.5
    idx = torch.arange(ctx, device=hidden.device)
    mask = idx <= t
    if not is_global:
        mask &= idx > t - window
    scores = torch.where(mask, scores, float("-inf"))
    probs = torch.softmax(scores, -1).to(dt)
    attn = (probs.float() @ v_cache.float()).to(dt).view(1, -1)
    branch = _linear_ref(attn, w["w_o"][layer])
    hidden = hidden + _rms_ref(branch, w["post_attn_norm"][layer])
    x = _rms_ref(hidden, w["pre_mlp_norm"][layer])
    g = _linear_ref(x, w["w_gate"][layer])
    act = torch.nn.functional.gelu(g, approximate="tanh") * _linear_ref(
        x, w["w_up"][layer]
    )
    branch = _linear_ref(act, w["w_down"][layer])
    return hidden + _rms_ref(branch, w["post_mlp_norm"][layer])


def gemma3_decode_stack_ref(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    s: dict[str, torch.Tensor],
    pos: torch.Tensor,
    period: int,
    window: int,
) -> torch.Tensor:
    """Eager PyTorch reference for :func:`gemma3_decode_stack` (updates ``s``)."""
    for layer in range(w["input_norm"].size(0)):
        is_global = layer % period == period - 1
        hidden = gemma3_layer_ref(hidden, w, s, layer, pos, is_global, window)
    return hidden


# %%
WEIGHT_NAMES = (
    "input_norm",
    "post_attn_norm",
    "pre_mlp_norm",
    "post_mlp_norm",
    "w_qkv",
    "q_norm",
    "k_norm",
    "w_o",
    "w_gate",
    "w_up",
    "w_down",
    "inv_freq_local",
    "inv_freq_global",
)
STATE_NAMES = ("k_cache", "v_cache")


def make_gemma3_stack_inputs(
    layers: int = 6,
    dim: int = 3840,
    intermediate: int = 1920,
    heads: int = 2,
    kv_heads: int = 1,
    head_dim: int = 256,
    context: int = 2048,
    position: int = 1500,
    local_theta: float = 1e4,
    global_theta: float = 1e6,
    global_scaling: float = 8.0,
    device: torch.device | str = DEVICE,
) -> tuple[
    torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor
]:
    """Random stacked weights and caches; the defaults are one TP8 rank of
    Gemma-3-12B (16 query heads, 8 kv heads, MLP 15360), whose dims are
    already multiples of 128.  The default position lies past a 1024 window,
    so local layers mask out the oldest rows."""
    bf16 = torch.bfloat16

    def weight(k: int, out: int) -> torch.Tensor:
        return (torch.randn(layers, k, out, device=device) * k**-0.5).to(bf16)

    def vec(k: int) -> torch.Tensor:
        return (torch.randn(layers, k, device=device) * 0.1).to(bf16)

    def cache() -> torch.Tensor:
        shape = (layers, kv_heads, context, head_dim)
        return torch.randn(*shape, device=device).to(bf16)

    half = head_dim // 2
    exponent = torch.arange(half, dtype=torch.float32, device=device) * 2 / head_dim
    w = {
        "input_norm": vec(dim),
        "post_attn_norm": vec(dim),
        "pre_mlp_norm": vec(dim),
        "post_mlp_norm": vec(dim),
        "w_qkv": weight(dim, (heads + 2 * kv_heads) * head_dim),
        "q_norm": vec(head_dim),
        "k_norm": vec(head_dim),
        "w_o": weight(heads * head_dim, dim),
        "w_gate": weight(dim, intermediate),
        "w_up": weight(dim, intermediate),
        "w_down": weight(intermediate, dim),
        "inv_freq_local": 1.0 / local_theta**exponent,
        "inv_freq_global": 1.0 / global_theta**exponent / global_scaling,
    }
    s = {"k_cache": cache(), "v_cache": cache()}
    hidden = torch.randn(1, dim, device=device).to(bf16)
    pos = torch.tensor([position], dtype=torch.int32, device=device)
    return hidden, w, s, pos


def gemma3_stack_args(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    s: dict[str, torch.Tensor],
    pos: torch.Tensor,
    period: int,
    window: int,
) -> tuple[object, ...]:
    """Positional arguments of :func:`gemma3_decode_stack`."""
    return (
        hidden,
        *(w[n] for n in WEIGHT_NAMES),
        *(s[n] for n in STATE_NAMES),
        pos,
        hl.constexpr(period),
        hl.constexpr(window),
    )


# %%
def main() -> None:
    """Run one period (5 local layers, 1 global) and compare it against the
    eager reference."""
    torch.manual_seed(0)
    period, window = 6, 1024
    hidden, w, s, pos = make_gemma3_stack_inputs(layers=period)
    s_ref = {k: v.clone() for k, v in s.items()}
    expected = gemma3_decode_stack_ref(hidden, w, s_ref, pos, period, window)
    result = gemma3_decode_stack(*gemma3_stack_args(hidden, w, s, pos, period, window))
    # bf16 rounding differences compound over the layers.
    torch.testing.assert_close(result, expected, atol=1e-1, rtol=5e-2)
    for name in STATE_NAMES:
        torch.testing.assert_close(s[name], s_ref[name], atol=5e-2, rtol=2e-2)


if __name__ == "__main__":
    main()
