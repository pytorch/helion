"""
Multi-head Latent Attention Decoder Stack (Pallas TPU)
======================================================

One decode token through a stack of DeepSeek-V2-Lite-style decoder layers
written as ONE Helion kernel.  Attention is multi-head latent attention (MLA)
with absorbed up-projections; the MLP is dense.  Every layer is

    q = qproj(x);  [c_kv | k_pe] = kv_down(x),  x = rms(h, attn_norm)
    c_kv = rms(c_kv, kv_norm);  q_pe, k_pe = rope(q_pe), rope(k_pe)
    caches[pos] = c_kv, k_pe
    q_lat[h] = q_nope[h] @ w_uk[h]
    p[h] = softmax(q_lat[h] . c_kv[:pos] + q_pe[h] . k_pe[:pos])
    h += o_proj(concat_h((p[h] @ c_kv) @ w_uv[h]))
    h += down(silu(gate(x)) * up(x)),  x = rms(h, mlp_norm)

The caches hold only the 512-wide latent and the 64-wide shared RoPE key per
position, stacked per layer and updated in place at row ``pos``.  The host
loop over layers is folded by the megakernel into one device loop.
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
def mla_decode_stack(
    hidden: torch.Tensor,  # [M, H] bf16
    attn_norm: torch.Tensor,  # [L, H]
    mlp_norm: torch.Tensor,  # [L, H]
    w_q: torch.Tensor,  # [L, H, heads * (nope + rope)]  (per head: nope | rope)
    w_kv: torch.Tensor,  # [L, H, latent + rope]
    kv_norm: torch.Tensor,  # [L, latent]
    w_uk: torch.Tensor,  # [L, heads, nope, latent]
    w_uv: torch.Tensor,  # [L, heads, latent, v_dim]
    w_o: torch.Tensor,  # [L, heads * v_dim, H]
    w_gate: torch.Tensor,  # [L, H, I]
    w_up: torch.Tensor,  # [L, H, I]
    w_down: torch.Tensor,  # [L, I, H]
    inv_freq: torch.Tensor,  # [rope / 2] f32
    c_kv: torch.Tensor,  # [L, ctx, latent] bf16, row pos written in place
    k_pe: torch.Tensor,  # [L, ctx, rope] bf16, row pos written in place
    pos: torch.Tensor,  # [1] int32
) -> torch.Tensor:
    m, h = hidden.shape
    layers, ctx, latent = c_kv.shape
    rope = k_pe.size(2)
    _, heads, nope, _ = w_uk.shape
    vd = w_uv.size(3)
    half = rope // 2
    nqh = w_q.size(2)
    nkv = w_kv.size(2)
    inter = w_gate.size(2)
    dt = hidden.dtype
    sm_scale = (nope + rope) ** -0.5
    res = torch.empty([m, h], dtype=dt, device=hidden.device)
    x = torch.empty([m, h], dtype=dt, device=hidden.device)
    q_buf = torch.empty([m, nqh], dtype=dt, device=hidden.device)
    kv_buf = torch.empty([m, nkv], dtype=dt, device=hidden.device)
    attn = torch.empty([m, heads * vd], dtype=dt, device=hidden.device)
    act = torch.empty([m, inter], dtype=dt, device=hidden.device)
    out = torch.empty_like(hidden)
    for tm in hl.tile(m):
        res[tm, :] = hidden[tm, :]
    for layer in range(layers):
        for tm in hl.tile(m):
            v = res[tm, :].float()
            v = v * torch.rsqrt(torch.mean(v * v, -1, keepdim=True) + EPS)
            x[tm, :] = (v * attn_norm[layer, :][None, :].float()).to(dt)
        for tm, tn in hl.tile([m, nqh]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = hl.dot(x[tm, tk], w_q[layer, tk, tn], acc=acc)
            q_buf[tm, tn] = acc.to(dt)
        for tm, tn in hl.tile([m, nkv]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = hl.dot(x[tm, tk], w_kv[layer, tk, tn], acc=acc)
            kv_buf[tm, tn] = acc.to(dt)
        # latent attention for the decode token (row 0) over all heads
        for _ in hl.grid(1):
            t = pos[0]
            ang = t.to(torch.float32) * inv_freq[:]
            cos = torch.cos(ang).to(dt)
            sin = torch.sin(ang).to(dt)
            kv = kv_buf[0, :]
            c = kv[:latent].float()
            c = c * torch.rsqrt(torch.mean(c * c, -1, keepdim=True) + EPS)
            c_kv[layer, t, :] = (c * kv_norm[layer, :].float()).to(dt)
            kr1 = kv[latent : latent + half]
            kr2 = kv[latent + half :]
            k_pe[layer, t, :] = torch.cat(
                [kr1 * cos - kr2 * sin, kr2 * cos + kr1 * sin]
            )
            q = q_buf[0, :].reshape(heads, nope + rope)
            q_nope = q[:, :nope]
            qr1 = q[:, nope : nope + half]
            qr2 = q[:, nope + half :]
            c2 = cos[None, :]
            s2 = sin[None, :]
            q_pe = torch.cat([qr1 * c2 - qr2 * s2, qr2 * c2 + qr1 * s2], dim=-1)
            # absorb the key up-projection into the query
            q_lat = hl.dot(
                q_nope[:, None, :], w_uk[layer, :, :, :], out_dtype=torch.float32
            )
            q_lat = q_lat.to(dt).reshape(heads, latent)
            m_i = hl.full([heads], float("-inf"), dtype=torch.float32)
            l_i = hl.zeros([heads], dtype=torch.float32)
            acc_o = hl.zeros([heads, latent], dtype=torch.float32)
            for tt in hl.tile(ctx):
                lat = c_kv[layer, tt, :]
                scores = hl.dot(q_lat, lat.transpose(0, 1), out_dtype=torch.float32)
                scores = scores + hl.dot(
                    q_pe, k_pe[layer, tt, :].transpose(0, 1), out_dtype=torch.float32
                )
                scores = torch.where(tt.index[None, :] <= t, scores * sm_scale, -1e30)
                m_new = torch.maximum(m_i, torch.amax(scores, -1))
                alpha = torch.exp(m_i - m_new)
                probs = torch.exp(scores - m_new[:, None])
                l_i = l_i * alpha + torch.sum(probs, -1)
                acc_o = acc_o * alpha[:, None] + hl.dot(
                    probs.to(dt), lat, out_dtype=torch.float32
                )
                m_i = m_new
            o_lat = (acc_o / l_i[:, None]).to(dt)
            # the value up-projection, per head
            o = hl.dot(o_lat[:, None, :], w_uv[layer, :, :, :], out_dtype=torch.float32)
            attn[0, :] = o.to(dt).reshape(heads * vd)
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(heads * vd):
                acc = hl.dot(attn[tm, tk], w_o[layer, tk, tn], acc=acc)
            res[tm, tn] = res[tm, tn] + acc.to(dt)
        for tm in hl.tile(m):
            v = res[tm, :].float()
            v = v * torch.rsqrt(torch.mean(v * v, -1, keepdim=True) + EPS)
            x[tm, :] = (v * mlp_norm[layer, :][None, :].float()).to(dt)
        for tm, tn in hl.tile([m, inter]):
            g = hl.zeros([tm, tn], dtype=torch.float32)
            u = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                xk = x[tm, tk]
                g = hl.dot(xk, w_gate[layer, tk, tn], acc=g)
                u = hl.dot(xk, w_up[layer, tk, tn], acc=u)
            gf = g.to(dt).float()
            act[tm, tn] = (gf * torch.sigmoid(gf)).to(dt) * u.to(dt)
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(inter):
                acc = hl.dot(act[tm, tk], w_down[layer, tk, tn], acc=acc)
            res[tm, tn] = res[tm, tn] + acc.to(dt)
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
    return (v * weight.float()).to(x.dtype)


def _linear_ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (x.float() @ weight.float()).to(x.dtype)


def _rope_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def mla_layer_ref(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    s: dict[str, torch.Tensor],
    layer: int,
    pos: torch.Tensor,
) -> torch.Tensor:
    """One MLA decoder layer on ``hidden`` [1, H] (updates ``s``)."""
    _, ctx, latent = s["c_kv"].shape
    _, heads, nope, _ = w["w_uk"].shape
    dt = hidden.dtype
    t = int(pos[0])
    ang = t * w["inv_freq"]
    cos, sin = torch.cos(ang).to(dt), torch.sin(ang).to(dt)
    x = _rms_ref(hidden, w["attn_norm"][layer])
    q = _linear_ref(x, w["w_q"][layer]).view(heads, -1)
    kv = _linear_ref(x, w["w_kv"][layer])[0]
    s["c_kv"][layer, t] = _rms_ref(kv[:latent], w["kv_norm"][layer])
    s["k_pe"][layer, t] = _rope_ref(kv[latent:], cos, sin)
    q_pe = _rope_ref(q[:, nope:], cos, sin)
    q_lat = torch.einsum("hn,hnc->hc", q[:, :nope].float(), w["w_uk"][layer].float())
    lat = s["c_kv"][layer].float()
    scores = q_lat.to(dt).float() @ lat.T + q_pe.float() @ s["k_pe"][layer].float().T
    mask = torch.arange(ctx, device=hidden.device) <= t
    scores = torch.where(mask, scores * q.size(1) ** -0.5, float("-inf"))
    probs = torch.softmax(scores, -1).to(dt)
    o_lat = (probs.float() @ lat).to(dt)
    o = torch.einsum("hc,hcv->hv", o_lat.float(), w["w_uv"][layer].float()).to(dt)
    hidden = hidden + _linear_ref(o.view(1, -1), w["w_o"][layer])
    x = _rms_ref(hidden, w["mlp_norm"][layer])
    g = _linear_ref(x, w["w_gate"][layer]).float()
    act = (g * torch.sigmoid(g)).to(dt) * _linear_ref(x, w["w_up"][layer])
    return hidden + _linear_ref(act, w["w_down"][layer])


def mla_decode_stack_ref(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    s: dict[str, torch.Tensor],
    pos: torch.Tensor,
) -> torch.Tensor:
    """Eager PyTorch reference for :func:`mla_decode_stack` (updates ``s``)."""
    for layer in range(w["attn_norm"].size(0)):
        hidden = mla_layer_ref(hidden, w, s, layer, pos)
    return hidden


# %%
WEIGHT_NAMES = (
    "attn_norm",
    "mlp_norm",
    "w_q",
    "w_kv",
    "kv_norm",
    "w_uk",
    "w_uv",
    "w_o",
    "w_gate",
    "w_up",
    "w_down",
    "inv_freq",
)
STATE_NAMES = ("c_kv", "k_pe")


def make_mla_stack_inputs(
    layers: int = 4,
    dim: int = 2048,
    intermediate: int = 11008,
    heads: int = 16,
    nope_dim: int = 128,
    rope_dim: int = 64,
    latent_dim: int = 512,
    v_dim: int = 128,
    context: int = 2048,
    position: int = 100,
    rope_theta: float = 1e4,
    device: torch.device | str = DEVICE,
) -> tuple[
    torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor
]:
    """Random stacked weights and caches; the defaults are DeepSeek-V2-Lite's
    attention (16 heads, no query compression) on one device with a dense MLP
    in place of the experts (its dense layer's 10944, rounded up to a multiple
    of 128)."""
    bf16 = torch.bfloat16

    def weight(*shape: int) -> torch.Tensor:
        k = shape[-2]
        return (torch.randn(layers, *shape, device=device) * k**-0.5).to(bf16)

    def vec(k: int) -> torch.Tensor:
        return (1 + 0.1 * torch.randn(layers, k, device=device)).to(bf16)

    half = rope_dim // 2
    exponent = torch.arange(half, dtype=torch.float32, device=device) * 2 / rope_dim
    w = {
        "attn_norm": vec(dim),
        "mlp_norm": vec(dim),
        "w_q": weight(dim, heads * (nope_dim + rope_dim)),
        "w_kv": weight(dim, latent_dim + rope_dim),
        "kv_norm": vec(latent_dim),
        "w_uk": weight(heads, nope_dim, latent_dim),
        "w_uv": weight(heads, latent_dim, v_dim),
        "w_o": weight(heads * v_dim, dim),
        "w_gate": weight(dim, intermediate),
        "w_up": weight(dim, intermediate),
        "w_down": weight(intermediate, dim),
        "inv_freq": 1.0 / rope_theta**exponent,
    }
    s = {
        "c_kv": torch.randn(layers, context, latent_dim, device=device).to(bf16),
        "k_pe": torch.randn(layers, context, rope_dim, device=device).to(bf16),
    }
    hidden = torch.randn(1, dim, device=device).to(bf16)
    pos = torch.tensor([position], dtype=torch.int32, device=device)
    return hidden, w, s, pos


def mla_stack_args(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    s: dict[str, torch.Tensor],
    pos: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Positional arguments of :func:`mla_decode_stack`."""
    return (
        hidden,
        *(w[n] for n in WEIGHT_NAMES),
        *(s[n] for n in STATE_NAMES),
        pos,
    )


# %%
def main() -> None:
    """Run a 4-layer stack and compare it against the eager reference."""
    torch.manual_seed(0)
    hidden, w, s, pos = make_mla_stack_inputs(layers=4)
    s_ref = {k: v.clone() for k, v in s.items()}
    expected = mla_decode_stack_ref(hidden, w, s_ref, pos)
    result = mla_decode_stack(*mla_stack_args(hidden, w, s, pos))
    # bf16 rounding differences compound over the layers.
    torch.testing.assert_close(result, expected, atol=1e-1, rtol=5e-2)
    for name in STATE_NAMES:
        torch.testing.assert_close(s[name], s_ref[name], atol=5e-2, rtol=2e-2)


if __name__ == "__main__":
    main()
