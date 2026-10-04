"""
Llama Decoder Stack (Pallas TPU)
================================

One decode token through a stack of Llama-style decoder layers written as ONE
Helion kernel.  Every layer is

    h += o_proj(attention(rope(qkv_proj(rms(h, attn_norm)))))
    h += down(silu(gate(x)) * up(x)),  x = rms(h, mlp_norm)

The query, key and value projections share one fused weight, RoPE rotates the
whole head (rotate-half convention), and the KV caches are stacked per layer
and updated in place at row ``pos``.  The host loop over layers is folded by
the megakernel into one device loop.
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
EPS = 1e-5


# %%
@helion.kernel(backend="pallas", static_shapes=True)
def llama_decode_stack(
    hidden: torch.Tensor,  # [M, H] bf16
    attn_norm: torch.Tensor,  # [L, H]
    mlp_norm: torch.Tensor,  # [L, H]
    w_qkv: torch.Tensor,  # [L, H, (heads + 2 * kv_heads) * d]
    w_o: torch.Tensor,  # [L, heads * d, H]
    w_gate: torch.Tensor,  # [L, H, I]
    w_up: torch.Tensor,  # [L, H, I]
    w_down: torch.Tensor,  # [L, I, H]
    inv_freq: torch.Tensor,  # [d / 2] f32
    k_cache: torch.Tensor,  # [L, kv_heads, ctx, d] bf16, row pos written in place
    v_cache: torch.Tensor,  # [L, kv_heads, ctx, d]
    pos: torch.Tensor,  # [1] int32
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
    act = torch.empty([m, inter], dtype=dt, device=hidden.device)
    out = torch.empty_like(hidden)
    for tm in hl.tile(m):
        res[tm, :] = hidden[tm, :]
    for layer in range(layers):
        for tm in hl.tile(m):
            v = res[tm, :].float()
            v = v * torch.rsqrt(torch.mean(v * v, -1, keepdim=True) + EPS)
            x[tm, :] = (v * attn_norm[layer, :][None, :].float()).to(dt)
        for tm, tn in hl.tile([m, nqkv]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = hl.dot(x[tm, tk], w_qkv[layer, tk, tn], acc=acc)
            qkv[tm, tn] = acc.to(dt)
        # attention for the decode token (row 0)
        for _ in hl.grid(1):
            t = pos[0]
            ang = t.to(torch.float32) * inv_freq[:]
            cos = torch.cos(ang).to(dt)[None, :]
            sin = torch.sin(ang).to(dt)[None, :]
            row = qkv[0, :]
            k = row[nq * d : (nq + nkv) * d].reshape(nkv, d)
            k1 = k[:, :half]
            k2 = k[:, half:]
            k_cache[layer, :, t, :] = torch.cat(
                [k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1
            )
            v_cache[layer, :, t, :] = row[(nq + nkv) * d :].reshape(nkv, d)
            q = row[: nq * d].reshape(nkv, group, d)
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
                    q, k_cache[layer, :, tt, :].transpose(1, 2), out_dtype=torch.float32
                )
                scores = torch.where(
                    tt.index[None, None, :] <= t, scores * sm_scale, -1e30
                )
                m_new = torch.maximum(m_i, torch.amax(scores, -1))
                alpha = torch.exp(m_i - m_new)
                probs = torch.exp(scores - m_new[:, :, None])
                l_i = l_i * alpha + torch.sum(probs, -1)
                acc_o = acc_o * alpha[:, :, None] + hl.dot(
                    probs.to(dt), v_cache[layer, :, tt, :], out_dtype=torch.float32
                )
                m_i = m_new
            attn[0, :] = (acc_o / l_i[:, :, None]).to(dt).reshape(nq * d)
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(nq * d):
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
            act[tm, tn] = (g * torch.sigmoid(g) * u).to(dt)
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


def llama_decode_stack_ref(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    s: dict[str, torch.Tensor],
    pos: torch.Tensor,
) -> torch.Tensor:
    """Eager PyTorch reference for :func:`llama_decode_stack` (updates ``s``)."""
    _, nkv, ctx, d = s["k_cache"].shape
    nq = w["w_o"].size(1) // d
    t = int(pos[0])
    ang = t * w["inv_freq"]
    cos, sin = torch.cos(ang).to(hidden.dtype), torch.sin(ang).to(hidden.dtype)
    mask = torch.arange(ctx, device=hidden.device) <= t
    for layer in range(w["attn_norm"].size(0)):
        x = _rms_ref(hidden, w["attn_norm"][layer])
        qkv = _linear_ref(x, w["w_qkv"][layer])[0]
        query = _rope_ref(qkv[: nq * d].view(nq, d), cos, sin)
        key = _rope_ref(qkv[nq * d : (nq + nkv) * d].view(nkv, d), cos, sin)
        k_cache, v_cache = s["k_cache"][layer], s["v_cache"][layer]
        k_cache[:, t] = key
        v_cache[:, t] = qkv[(nq + nkv) * d :].view(nkv, d)
        grouped = query.view(nkv, -1, d).float()
        scores = grouped @ k_cache.float().transpose(1, 2) * d**-0.5
        scores = torch.where(mask, scores, float("-inf"))
        probs = torch.softmax(scores, -1).to(hidden.dtype)
        attn = (probs.float() @ v_cache.float()).to(hidden.dtype).view(1, -1)
        hidden = hidden + _linear_ref(attn, w["w_o"][layer])
        x = _rms_ref(hidden, w["mlp_norm"][layer])
        g = (x.float() @ w["w_gate"][layer].float()).float()
        u = (x.float() @ w["w_up"][layer].float()).float()
        act = (g * torch.sigmoid(g) * u).to(hidden.dtype)
        hidden = hidden + _linear_ref(act, w["w_down"][layer])
    return hidden


# %%
WEIGHT_NAMES = (
    "attn_norm",
    "mlp_norm",
    "w_qkv",
    "w_o",
    "w_gate",
    "w_up",
    "w_down",
    "inv_freq",
)
STATE_NAMES = ("k_cache", "v_cache")


def make_llama_stack_inputs(
    layers: int = 4,
    dim: int = 4096,
    intermediate: int = 1792,
    heads: int = 4,
    kv_heads: int = 1,
    head_dim: int = 128,
    context: int = 2048,
    position: int = 100,
    rope_theta: float = 5e5,
    device: torch.device | str = DEVICE,
) -> tuple[
    torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor
]:
    """Random stacked weights and caches; the defaults are one TP8 rank of
    Llama-3-8B."""
    bf16 = torch.bfloat16

    def weight(k: int, out: int) -> torch.Tensor:
        return (torch.randn(layers, k, out, device=device) * k**-0.5).to(bf16)

    def cache() -> torch.Tensor:
        shape = (layers, kv_heads, context, head_dim)
        return torch.randn(*shape, device=device).to(bf16)

    half = head_dim // 2
    exponent = torch.arange(half, dtype=torch.float32, device=device) * 2 / head_dim
    w = {
        "attn_norm": (1 + 0.1 * torch.randn(layers, dim, device=device)).to(bf16),
        "mlp_norm": (1 + 0.1 * torch.randn(layers, dim, device=device)).to(bf16),
        "w_qkv": weight(dim, (heads + 2 * kv_heads) * head_dim),
        "w_o": weight(heads * head_dim, dim),
        "w_gate": weight(dim, intermediate),
        "w_up": weight(dim, intermediate),
        "w_down": weight(intermediate, dim),
        "inv_freq": 1.0 / rope_theta**exponent,
    }
    s = {"k_cache": cache(), "v_cache": cache()}
    hidden = torch.randn(1, dim, device=device).to(bf16)
    pos = torch.tensor([position], dtype=torch.int32, device=device)
    return hidden, w, s, pos


def llama_stack_args(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    s: dict[str, torch.Tensor],
    pos: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Positional arguments of :func:`llama_decode_stack`."""
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
    hidden, w, s, pos = make_llama_stack_inputs(layers=4)
    s_ref = {k: v.clone() for k, v in s.items()}
    expected = llama_decode_stack_ref(hidden, w, s_ref, pos)
    result = llama_decode_stack(*llama_stack_args(hidden, w, s, pos))
    # bf16 rounding differences compound over the layers.
    torch.testing.assert_close(result, expected, atol=1e-1, rtol=5e-2)
    for name in STATE_NAMES:
        torch.testing.assert_close(s[name], s_ref[name], atol=5e-2, rtol=2e-2)


if __name__ == "__main__":
    main()
