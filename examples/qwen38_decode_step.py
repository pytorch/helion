"""
Qwen3.8 Decode Step (Pallas TPU)
================================

One decode token through a whole Qwen3.8 model at one TP8 rank's shapes, as
ONE Helion kernel:

    h = embedding[token]
    for each layer:  h += mixer(rms(h, input_norm));  h += mlp(rms(h, post_norm))
    top-1 of rms(h, final_norm) @ lm_head over this rank's vocab shard

The layers are those of :mod:`examples.qwen38_hybrid_stack`: three
Gated-DeltaNet layers, then one GQA layer, per period of four.  The megakernel
keeps the embedding table in HBM and copies only the token's row, streams the
LM head through its own ring right after the last layer's weights, and keeps
the running top-1 over the vocab tiles in registers.
"""

# %%
from __future__ import annotations

import torch

from examples.qwen38_hybrid_stack import STATE_NAMES
from examples.qwen38_hybrid_stack import WEIGHT_NAMES
from examples.qwen38_hybrid_stack import _rms_ref
from examples.qwen38_hybrid_stack import make_hybrid_stack_inputs
from examples.qwen38_hybrid_stack import qwen38_hybrid_stack_ref

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
def qwen38_decode_step(
    token: torch.Tensor,  # [1] int32
    embedding: torch.Tensor,  # [V, H] bf16
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
    final_norm: torch.Tensor,  # [H]
    lm_head: torch.Tensor,  # [H, VP], columns from ``vocab`` on are padding
    conv_state: torch.Tensor,  # [Lg, conv_dim, taps] bf16, in place
    rec_state: torch.Tensor,  # [Lg, v_heads, v_dim, k_dim] f32, in place
    k_cache: torch.Tensor,  # [La, kv_heads, ctx, d] bf16, row pos written in place
    v_cache: torch.Tensor,  # [La, kv_heads, ctx, d]
    pos: torch.Tensor,  # [1] int32
    vocab: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    m = token.size(0)
    h = embedding.size(1)
    inter = w_gate.size(2)
    _, conv_dim, _ = conv_state.shape
    _, hv, dv, dk = rec_state.shape
    key_dim = (conv_dim - hv * dv) // 2
    hk = key_dim // dk
    periods, nkv, ctx, d = k_cache.shape
    nqg = attn_q.size(2)
    group = nqg // (2 * d) // nkv
    half = inv_freq.size(0)
    vp = lm_head.size(1)
    dt = embedding.dtype
    sm_scale = d**-0.5
    res = torch.empty([m, h], dtype=dt, device=embedding.device)
    x = torch.empty([m, h], dtype=dt, device=embedding.device)
    qkv = torch.empty([m, conv_dim], dtype=dt, device=embedding.device)
    zb = torch.empty([m, hv * dv], dtype=dt, device=embedding.device)
    bb = torch.empty([m, hv], dtype=dt, device=embedding.device)
    ab = torch.empty([m, hv], dtype=dt, device=embedding.device)
    core = torch.empty([m, hv * dv], dtype=dt, device=embedding.device)
    q_buf = torch.empty([m, nqg], dtype=dt, device=embedding.device)
    k_buf = torch.empty([m, nkv * d], dtype=dt, device=embedding.device)
    v_buf = torch.empty([m, nkv * d], dtype=dt, device=embedding.device)
    attn = torch.empty([m, nkv * group * d], dtype=dt, device=embedding.device)
    act = torch.empty([m, inter], dtype=dt, device=embedding.device)
    top_idx = torch.empty([m], dtype=torch.int32, device=embedding.device)
    top_val = torch.empty([m], dtype=torch.float32, device=embedding.device)
    for _ in hl.grid(1):
        tok = token[0]
        res[0, :] = embedding[tok, :]
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
    # final RMSNorm, then the top-1 over this rank's vocab shard
    for tm in hl.tile(m):
        v = res[tm, :].float()
        v = v * torch.rsqrt(torch.mean(v * v, -1, keepdim=True) + EPS)
        x[tm, :] = (v * (1.0 + final_norm[None, :].float())).to(dt)
    for tm in hl.tile(m):
        best = hl.full([tm], float("-inf"), dtype=torch.float32)
        best_i = hl.zeros([tm], dtype=torch.int32)
        for tv in hl.tile(vp):
            logits = hl.dot(x[tm, :], lm_head[:, tv], out_dtype=torch.float32)
            valid = tv.index[None, :] < vocab  # pyrefly: ignore[unsupported-operation]
            logits = torch.where(valid, logits, float("-inf"))
            tile_max = torch.amax(logits, -1)
            tile_arg = torch.argmax(logits, -1).to(torch.int32) + tv.begin
            better = tile_max > best
            best_i = torch.where(better, tile_arg, best_i)
            best = torch.where(better, tile_max, best)
        top_idx[tm] = best_i
        top_val[tm] = best
    return top_idx, top_val


# %%
# Reference
# ---------


# %%
def qwen38_decode_step_logits(
    token: torch.Tensor,
    embedding: torch.Tensor,
    w: dict[str, torch.Tensor],
    s: dict[str, torch.Tensor],
    pos: torch.Tensor,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    vocab: int,
) -> torch.Tensor:
    """Eager PyTorch logits over the vocab shard for :func:`qwen38_decode_step`
    (updates ``s``)."""
    hidden = embedding[token.long()]
    hidden = qwen38_hybrid_stack_ref(hidden, w, s, pos)
    x = _rms_ref(hidden, final_norm)
    return x.float() @ lm_head[:, :vocab].float()


def make_decode_step_inputs(
    layers: int = 4,
    dim: int = 5120,
    embedding_rows: int = 248320,
    padded_vocab: int = 32000,
    vocab: int = 31040,
    token: int = 1234,
    device: torch.device | str = DEVICE,
    **stack_kwargs: object,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Random inputs for one TP8 rank: ``(token, embedding, weights, states,
    pos, final_norm, lm_head)``; the LM head holds ``padded_vocab`` columns,
    the first ``vocab`` of them real."""
    _, w, s, pos = make_hybrid_stack_inputs(
        layers=layers,
        dim=dim,
        device=device,
        **stack_kwargs,  # pyrefly: ignore[bad-argument-type]
    )
    bf16 = torch.bfloat16
    embedding = torch.randn(embedding_rows, dim, device=device).to(bf16)
    final_norm = (torch.randn(dim, device=device) * 0.1).to(bf16)
    lm_head = (torch.randn(dim, padded_vocab, device=device) * dim**-0.5).to(bf16)
    lm_head[:, vocab:] = 0
    tok = torch.tensor([token], dtype=torch.int32, device=device)
    return tok, embedding, w, s, pos, final_norm, lm_head


def decode_step_args(
    token: torch.Tensor,
    embedding: torch.Tensor,
    w: dict[str, torch.Tensor],
    s: dict[str, torch.Tensor],
    pos: torch.Tensor,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    vocab: int,
) -> tuple[object, ...]:
    """Positional arguments of :func:`qwen38_decode_step`."""
    return (
        token,
        embedding,
        *(w[n] for n in WEIGHT_NAMES),
        final_norm,
        lm_head,
        *(s[n] for n in STATE_NAMES),
        pos,
        vocab,
    )


# %%
def main() -> None:
    """Run an 8-layer decode step and compare its top-1 against the eager
    reference's logits."""
    torch.manual_seed(0)
    vocab = 31040
    inputs = make_decode_step_inputs(layers=8, vocab=vocab)
    token, embedding, w, s, pos, final_norm, lm_head = inputs
    s_ref = {k: v.clone() for k, v in s.items()}
    logits = qwen38_decode_step_logits(
        token, embedding, w, s_ref, pos, final_norm, lm_head, vocab
    )
    top_idx, top_val = qwen38_decode_step(
        *decode_step_args(token, embedding, w, s, pos, final_norm, lm_head, vocab)
    )
    # bf16 rounding differences compound over the layers: the picked logit
    # must be a largest one up to that noise.
    best = logits.amax(-1)
    picked = logits.gather(-1, top_idx.long()[:, None])[:, 0]
    torch.testing.assert_close(picked, best, atol=1e-1, rtol=5e-2)
    torch.testing.assert_close(top_val, best, atol=1e-1, rtol=5e-2)


if __name__ == "__main__":
    main()
