"""
MoE Decoder Stack (Pallas TPU)
==============================

One decode token through a stack of Qwen3-MoE-style decoder layers written as
ONE Helion kernel.  Every layer is

    h += o_proj(attention(rope(qk_norm(qkv_proj(rms(h, attn_norm))))))
    h += moe(rms(h, mlp_norm))

The query, key and value projections share one fused weight; the queries and
keys get a per-head RMSNorm before RoPE (rotate-half over the whole head).  The
mixture of experts is a router GEMV, a softmax top-k renormalized over the
picks, and the picked SwiGLU experts, which read their ids from the top-k
output.  The residual stream stays in f32.  The KV caches are stacked per layer
and updated in place at row ``pos``; the host loop over layers is folded by the
megakernel into one device loop.
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
def moe_decode_stack(
    hidden: torch.Tensor,  # [M, H] bf16
    attn_norm: torch.Tensor,  # [L, H]
    mlp_norm: torch.Tensor,  # [L, H]
    w_qkv: torch.Tensor,  # [L, H, (heads + 2 * kv_heads) * d]
    q_norm: torch.Tensor,  # [L, d]
    k_norm: torch.Tensor,  # [L, d]
    w_o: torch.Tensor,  # [L, heads * d, H]
    w_router: torch.Tensor,  # [L, H, E]
    w_gate: torch.Tensor,  # [L, E, H, I]
    w_up: torch.Tensor,  # [L, E, H, I]
    w_down: torch.Tensor,  # [L, E, I, H]
    inv_freq: torch.Tensor,  # [d / 2] f32
    k_cache: torch.Tensor,  # [L, kv_heads, ctx, d] bf16, row pos written in place
    v_cache: torch.Tensor,  # [L, kv_heads, ctx, d]
    pos: torch.Tensor,  # [1] int32
    topk: hl.constexpr,
) -> torch.Tensor:
    m, h = hidden.shape
    layers, nkv, ctx, d = k_cache.shape
    nq = w_o.size(1) // d
    group = nq // nkv
    half = d // 2
    nqkv = w_qkv.size(2)
    n_exp = w_router.size(2)
    inter = w_gate.size(3)
    dt = hidden.dtype
    sm_scale = d**-0.5
    res = torch.empty([m, h], dtype=torch.float32, device=hidden.device)
    x = torch.empty([m, h], dtype=dt, device=hidden.device)
    qkv = torch.empty([m, nqkv], dtype=dt, device=hidden.device)
    attn = torch.empty([m, nq * d], dtype=dt, device=hidden.device)
    logits = torch.empty([m, n_exp], dtype=torch.float32, device=hidden.device)
    sel_idx = torch.empty([m, topk], dtype=torch.int32, device=hidden.device)  # pyrefly: ignore[no-matching-overload]
    sel_w = torch.empty([m, topk], dtype=torch.float32, device=hidden.device)  # pyrefly: ignore[no-matching-overload]
    act = torch.empty([m, inter], dtype=dt, device=hidden.device)
    out = torch.empty_like(hidden)
    for tm in hl.tile(m):
        res[tm, :] = hidden[tm, :].float()
    for layer in range(layers):
        # attention RMSNorm
        for tm in hl.tile(m):
            v = res[tm, :]
            v = v * torch.rsqrt(torch.mean(v * v, -1, keepdim=True) + EPS)
            x[tm, :] = (v * attn_norm[layer, :][None, :].float()).to(dt)
        for tm, tn in hl.tile([m, nqkv]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = hl.dot(x[tm, tk], w_qkv[layer, tk, tn], acc=acc)
            qkv[tm, tn] = acc.to(dt)
        # q/k RMSNorm, RoPE and attention for the decode token (row 0)
        for _ in hl.grid(1):
            t = pos[0]
            ang = t.to(torch.float32) * inv_freq[:]
            cos = torch.cos(ang).to(dt)[None, :]
            sin = torch.sin(ang).to(dt)[None, :]
            row = qkv[0, :]
            k = row[nq * d : (nq + nkv) * d].reshape(nkv, d).float()
            k = k * torch.rsqrt(torch.mean(k * k, -1, keepdim=True) + EPS)
            k = (k * k_norm[layer, :][None, :].float()).to(dt)
            k1 = k[:, :half]
            k2 = k[:, half:]
            k_cache[layer, :, t, :] = torch.cat(
                [k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1
            )
            v_cache[layer, :, t, :] = row[(nq + nkv) * d :].reshape(nkv, d)
            q = row[: nq * d].reshape(nkv, group, d).float()
            q = q * torch.rsqrt(torch.mean(q * q, -1, keepdim=True) + EPS)
            q = (q * q_norm[layer, :][None, None, :].float()).to(dt)
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
            res[tm, tn] = res[tm, tn] + acc
        # MoE RMSNorm, router and softmax top-k
        for tm in hl.tile(m):
            v = res[tm, :]
            v = v * torch.rsqrt(torch.mean(v * v, -1, keepdim=True) + EPS)
            x[tm, :] = (v * mlp_norm[layer, :][None, :].float()).to(dt)
        for tm, te in hl.tile([m, n_exp]):
            acc = hl.zeros([tm, te], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = hl.dot(x[tm, tk], w_router[layer, tk, te], acc=acc)
            logits[tm, te] = acc
        for tm in hl.tile(m):
            scores = logits[tm, :]
            lane = hl.arange(n_exp)
            top = torch.amax(scores, -1)
            total = hl.zeros([tm], dtype=torch.float32)
            for k in hl.static_range(topk):  # pyrefly: ignore[bad-argument-type]
                best = torch.amax(scores, -1)
                arg = torch.argmax(scores, -1).to(torch.int32)
                p = torch.exp(best - top)
                sel_idx[tm, k] = arg
                sel_w[tm, k] = p
                total = total + p
                scores = torch.where(lane[None, :] == arg[:, None], -1.0e30, scores)
            for k in hl.static_range(topk):  # pyrefly: ignore[bad-argument-type]
                sel_w[tm, k] = sel_w[tm, k] / total
        # the picked experts (ids of the decode token, row 0)
        for k in hl.static_range(topk):  # pyrefly: ignore[bad-argument-type]
            for tm, tn in hl.tile([m, inter]):
                e = sel_idx[0, k]
                g = hl.zeros([tm, tn], dtype=torch.float32)
                u = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(h):
                    xk = x[tm, tk]
                    g = hl.dot(xk, w_gate[layer, e, tk, tn], acc=g)
                    u = hl.dot(xk, w_up[layer, e, tk, tn], acc=u)
                act[tm, tn] = (g * torch.sigmoid(g) * u).to(dt)
            for tm, tn in hl.tile([m, h]):
                e = sel_idx[0, k]
                acc = hl.zeros([tm, tn], dtype=torch.float32)
                for tk in hl.tile(inter):
                    acc = hl.dot(act[tm, tk], w_down[layer, e, tk, tn], acc=acc)
                res[tm, tn] = res[tm, tn] + sel_w[0, k] * acc
    for tm in hl.tile(m):
        out[tm, :] = res[tm, :].to(dt)
    return out


# %%
# Reference
# ---------


# %%
def _rms_ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    v = x.float()
    v = v * torch.rsqrt(torch.mean(v * v, -1, keepdim=True) + EPS)
    return (v * weight.float()).to(torch.bfloat16)


def _linear_ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.float() @ weight.float()


def _rope_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def moe_route_ref(
    x: torch.Tensor, w_router: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expert ids and weights of row 0: softmax top-k, renormalized."""
    logits = _linear_ref(x, w_router)[0]
    vals, ids = torch.topk(logits, topk)
    return ids, torch.softmax(vals, -1)


def moe_decode_stack_ref(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    s: dict[str, torch.Tensor],
    pos: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Eager PyTorch reference for :func:`moe_decode_stack` (updates ``s``)."""
    _, nkv, ctx, d = s["k_cache"].shape
    nq = w["w_o"].size(1) // d
    dt = hidden.dtype
    t = int(pos[0])
    ang = t * w["inv_freq"]
    cos, sin = torch.cos(ang).to(dt), torch.sin(ang).to(dt)
    mask = torch.arange(ctx, device=hidden.device) <= t
    res = hidden.float()
    for layer in range(w["attn_norm"].size(0)):
        x = _rms_ref(res, w["attn_norm"][layer])
        qkv = _linear_ref(x, w["w_qkv"][layer]).to(dt)[0]
        query = _rms_ref(qkv[: nq * d].view(nq, d), w["q_norm"][layer])
        key = _rms_ref(qkv[nq * d : (nq + nkv) * d].view(nkv, d), w["k_norm"][layer])
        query, key = _rope_ref(query, cos, sin), _rope_ref(key, cos, sin)
        k_cache, v_cache = s["k_cache"][layer], s["v_cache"][layer]
        k_cache[:, t] = key
        v_cache[:, t] = qkv[(nq + nkv) * d :].view(nkv, d)
        grouped = query.view(nkv, -1, d).float()
        scores = grouped @ k_cache.float().transpose(1, 2) * d**-0.5
        scores = torch.where(mask, scores, float("-inf"))
        probs = torch.softmax(scores, -1).to(dt)
        attn = (probs.float() @ v_cache.float()).to(dt).view(1, -1)
        res = res + _linear_ref(attn, w["w_o"][layer])
        x = _rms_ref(res, w["mlp_norm"][layer])
        ids, weights = moe_route_ref(x, w["w_router"][layer], topk)
        for e, p in zip(ids.tolist(), weights, strict=True):
            g = _linear_ref(x, w["w_gate"][layer, e])
            u = _linear_ref(x, w["w_up"][layer, e])
            act = (g * torch.sigmoid(g) * u).to(dt)
            res = res + p * _linear_ref(act, w["w_down"][layer, e])
    return res.to(dt)


# %%
WEIGHT_NAMES = (
    "attn_norm",
    "mlp_norm",
    "w_qkv",
    "q_norm",
    "k_norm",
    "w_o",
    "w_router",
    "w_gate",
    "w_up",
    "w_down",
    "inv_freq",
)
STATE_NAMES = ("k_cache", "v_cache")


def make_moe_stack_inputs(
    layers: int = 4,
    dim: int = 2048,
    experts: int = 128,
    intermediate: int = 768,
    heads: int = 32,
    kv_heads: int = 4,
    head_dim: int = 128,
    context: int = 2048,
    position: int = 100,
    rope_theta: float = 1e6,
    device: torch.device | str = DEVICE,
) -> tuple[
    torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor
]:
    """Random stacked weights and caches; the defaults are Qwen3-30B-A3B-like."""
    bf16 = torch.bfloat16

    def weight(*shape: int) -> torch.Tensor:
        return (torch.randn(layers, *shape, device=device) * shape[-2] ** -0.5).to(bf16)

    def vec(k: int) -> torch.Tensor:
        return (1 + 0.1 * torch.randn(layers, k, device=device)).to(bf16)

    def cache() -> torch.Tensor:
        shape = (layers, kv_heads, context, head_dim)
        return torch.randn(*shape, device=device).to(bf16)

    half = head_dim // 2
    exponent = torch.arange(half, dtype=torch.float32, device=device) * 2 / head_dim
    w = {
        "attn_norm": vec(dim),
        "mlp_norm": vec(dim),
        "w_qkv": weight(dim, (heads + 2 * kv_heads) * head_dim),
        "q_norm": vec(head_dim),
        "k_norm": vec(head_dim),
        "w_o": weight(heads * head_dim, dim),
        "w_router": weight(dim, experts),
        "w_gate": weight(experts, dim, intermediate),
        "w_up": weight(experts, dim, intermediate),
        "w_down": weight(experts, intermediate, dim),
        "inv_freq": 1.0 / rope_theta**exponent,
    }
    s = {"k_cache": cache(), "v_cache": cache()}
    hidden = torch.randn(1, dim, device=device).to(bf16)
    pos = torch.tensor([position], dtype=torch.int32, device=device)
    return hidden, w, s, pos


def moe_stack_args(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    s: dict[str, torch.Tensor],
    pos: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor | int, ...]:
    """Positional arguments of :func:`moe_decode_stack`."""
    return (
        hidden,
        *(w[n] for n in WEIGHT_NAMES),
        *(s[n] for n in STATE_NAMES),
        pos,
        topk,
    )


# %%
def main() -> None:
    """Run a small 2-layer MoE stack and compare it against the eager
    reference."""
    torch.manual_seed(0)
    topk = 4
    hidden, w, s, pos = make_moe_stack_inputs(
        layers=2, dim=512, experts=16, intermediate=256, heads=4, kv_heads=1
    )
    s_ref = {k: v.clone() for k, v in s.items()}
    expected = moe_decode_stack_ref(hidden, w, s_ref, pos, topk)
    result = moe_decode_stack(*moe_stack_args(hidden, w, s, pos, topk))
    # bf16 rounding differences compound over the layers.
    torch.testing.assert_close(result, expected, atol=1e-1, rtol=5e-2)
    for name in STATE_NAMES:
        torch.testing.assert_close(s[name], s_ref[name], atol=5e-2, rtol=2e-2)


if __name__ == "__main__":
    main()
