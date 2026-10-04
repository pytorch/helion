"""
Gemma-3 Decode Step (Pallas TPU)
================================

One decode token through a whole Gemma-3-style model at one TP8 rank's shapes,
as ONE Helion kernel:

    h = embedding[token] * sqrt(H)  (the scale rounded to bf16, as in Gemma)
    for each layer:  the Gemma-3 layer of :mod:`examples.gemma3_decode_stack`
    top-1 of softcap(rms(h, final_norm) @ lm_head) over this rank's vocab shard

with ``softcap(x) = tanh(x / c) * c``.  The layers repeat ``period - 1``
sliding-window layers and one global layer.  The megakernel keeps the
embedding table in HBM and copies only the token's row, streams the LM head
through its own ring right after the last layer's weights, and keeps the
running top-1 over the vocab tiles in registers.
"""

# %%
from __future__ import annotations

import torch

from examples.gemma3_decode_stack import STATE_NAMES
from examples.gemma3_decode_stack import WEIGHT_NAMES
from examples.gemma3_decode_stack import _rms_ref
from examples.gemma3_decode_stack import gemma3_decode_stack_ref
from examples.gemma3_decode_stack import make_gemma3_stack_inputs

import helion
from helion._testing import DEVICE
import helion.language as hl

# %%
# Kernel
# ------

# RMSNorm epsilon.
EPS = 1e-6
# Final logit soft-cap.
SOFTCAP = 30.0


# %%
@helion.kernel(backend="pallas", static_shapes=True)
def gemma3_decode_step(
    token: torch.Tensor,  # [1] int32
    embedding: torch.Tensor,  # [V, H] bf16
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
    final_norm: torch.Tensor,  # [H]
    lm_head: torch.Tensor,  # [H, VP], columns from ``vocab`` on are padding
    k_cache: torch.Tensor,  # [L, kv_heads, ctx, d] bf16, row pos written in place
    v_cache: torch.Tensor,  # [L, kv_heads, ctx, d]
    pos: torch.Tensor,  # [1] int32
    period: int,  # the three ints are passed as hl.constexpr
    window: int,
    vocab: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    m = token.size(0)
    h = embedding.size(1)
    layers, nkv, ctx, d = k_cache.shape
    nq = w_o.size(1) // d
    group = nq // nkv
    half = d // 2
    nqkv = w_qkv.size(2)
    inter = w_gate.size(2)
    vp = lm_head.size(1)
    dt = embedding.dtype
    sm_scale = d**-0.5
    normalizer = h**0.5
    res = torch.empty([m, h], dtype=dt, device=embedding.device)
    x = torch.empty([m, h], dtype=dt, device=embedding.device)
    qkv = torch.empty([m, nqkv], dtype=dt, device=embedding.device)
    attn = torch.empty([m, nq * d], dtype=dt, device=embedding.device)
    branch = torch.empty([m, h], dtype=dt, device=embedding.device)
    act = torch.empty([m, inter], dtype=dt, device=embedding.device)
    top_idx = torch.empty([m], dtype=torch.int32, device=embedding.device)
    top_val = torch.empty([m], dtype=torch.float32, device=embedding.device)
    for _ in hl.grid(1):
        tok = token[0]
        res[0, :] = embedding[tok, :] * normalizer
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
    # final RMSNorm, then the soft-capped top-1 over this rank's vocab shard
    for tm in hl.tile(m):
        v = res[tm, :].float()
        v = v * torch.rsqrt(torch.mean(v * v, -1, keepdim=True) + EPS)
        x[tm, :] = (v * (1.0 + final_norm[None, :].float())).to(dt)
    for tm in hl.tile(m):
        best = hl.full([tm], float("-inf"), dtype=torch.float32)
        best_i = hl.zeros([tm], dtype=torch.int32)
        for tv in hl.tile(vp):
            logits = hl.dot(x[tm, :], lm_head[:, tv], out_dtype=torch.float32)
            logits = torch.tanh(logits / SOFTCAP) * SOFTCAP
            valid = tv.index[None, :] < vocab
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
def gemma3_decode_step_logits(
    token: torch.Tensor,
    embedding: torch.Tensor,
    w: dict[str, torch.Tensor],
    s: dict[str, torch.Tensor],
    pos: torch.Tensor,
    final_norm: torch.Tensor,
    lm_head: torch.Tensor,
    period: int,
    window: int,
    vocab: int,
) -> torch.Tensor:
    """Eager PyTorch soft-capped logits over the vocab shard for
    :func:`gemma3_decode_step` (updates ``s``)."""
    # Gemma rounds the embedding scale to the activation dtype.
    normalizer = torch.tensor(embedding.size(1) ** 0.5, dtype=embedding.dtype)
    hidden = embedding[token.long()] * normalizer
    hidden = gemma3_decode_stack_ref(hidden, w, s, pos, period, window)
    x = _rms_ref(hidden, final_norm)
    logits = x.float() @ lm_head[:, :vocab].float()
    return torch.tanh(logits / SOFTCAP) * SOFTCAP


def make_decode_step_inputs(
    layers: int = 6,
    dim: int = 3840,
    embedding_rows: int = 262208,
    padded_vocab: int = 32768,
    vocab: int = 32768,
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
    the first ``vocab`` of them real (Gemma-3's 262144-token vocab splits into
    eight 32768-column shards)."""
    _, w, s, pos = make_gemma3_stack_inputs(
        layers=layers,
        dim=dim,
        device=device,
        **stack_kwargs,  # pyrefly: ignore[bad-argument-type]
    )
    bf16 = torch.bfloat16
    embedding = (torch.randn(embedding_rows, dim, device=device) * dim**-0.5).to(bf16)
    final_norm = (torch.randn(dim, device=device) * 0.1).to(bf16)
    # large enough logits that the soft-cap bends the top ones
    lm_head = torch.randn(dim, padded_vocab, device=device) * 8.0 * dim**-0.5
    lm_head = lm_head.to(bf16)
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
    period: int,
    window: int,
    vocab: int,
) -> tuple[object, ...]:
    """Positional arguments of :func:`gemma3_decode_step`."""
    return (
        token,
        embedding,
        *(w[n] for n in WEIGHT_NAMES),
        final_norm,
        lm_head,
        *(s[n] for n in STATE_NAMES),
        pos,
        hl.constexpr(period),
        hl.constexpr(window),
        hl.constexpr(vocab),
    )


# %%
def main() -> None:
    """Run a one-period decode step and compare its top-1 against the eager
    reference's logits."""
    torch.manual_seed(0)
    period, window, vocab = 6, 1024, 32768
    inputs = make_decode_step_inputs(layers=period, vocab=vocab)
    token, embedding, w, s, pos, final_norm, lm_head = inputs
    s_ref = {k: v.clone() for k, v in s.items()}
    logits = gemma3_decode_step_logits(
        token, embedding, w, s_ref, pos, final_norm, lm_head, period, window, vocab
    )
    top_idx, top_val = gemma3_decode_step(
        *decode_step_args(
            token, embedding, w, s, pos, final_norm, lm_head, period, window, vocab
        )
    )
    # bf16 rounding differences compound over the layers: the picked logit
    # must be a largest one up to that noise.
    best = logits.amax(-1)
    picked = logits.gather(-1, top_idx.long()[:, None])[:, 0]
    torch.testing.assert_close(picked, best, atol=1e-1, rtol=5e-2)
    torch.testing.assert_close(top_val, best, atol=1e-1, rtol=5e-2)


if __name__ == "__main__":
    main()
