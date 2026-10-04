"""
Parallel-Residual Decoder Stack (Pallas TPU)
============================================

One decode token through a stack of GPT-J-style decoder layers written as ONE
Helion kernel.  Attention and the MLP read the same normalized input and add
into the residual together:

    x = layer_norm(h)
    h = h + o_proj(attention(rope(qkv_proj(x)))) + fc_out(gelu_tanh(fc_in(x)))

LayerNorm has a weight and a bias, the MLP projections have biases, RoPE
rotates the first ``rotary`` dims of each head (rotate-half convention), and
the KV caches are stacked per layer and updated in place at row ``pos``.  One
root computes both branch outputs for a tile of the residual, so the
attention output and the MLP activation meet in one accumulator.  The host
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

# LayerNorm epsilon.
EPS = 1e-5


# %%
@helion.kernel(backend="pallas", static_shapes=True)
def parallel_residual_stack(
    hidden: torch.Tensor,  # [M, H] bf16
    ln_w: torch.Tensor,  # [L, H]
    ln_b: torch.Tensor,  # [L, H]
    w_qkv: torch.Tensor,  # [L, H, 3 * heads * d]
    w_o: torch.Tensor,  # [L, heads * d, H]
    w_in: torch.Tensor,  # [L, H, I]
    b_in: torch.Tensor,  # [L, I]
    w_out: torch.Tensor,  # [L, I, H]
    b_out: torch.Tensor,  # [L, H]
    inv_freq: torch.Tensor,  # [rotary / 2] f32
    k_cache: torch.Tensor,  # [L, heads, ctx, d] bf16, row pos written in place
    v_cache: torch.Tensor,  # [L, heads, ctx, d]
    pos: torch.Tensor,  # [1] int32
) -> torch.Tensor:
    m, h = hidden.shape
    layers, nh, ctx, d = k_cache.shape
    half = inv_freq.size(0)
    nqkv = w_qkv.size(2)
    inter = w_in.size(2)
    dt = hidden.dtype
    sm_scale = d**-0.5
    res = torch.empty([m, h], dtype=dt, device=hidden.device)
    x = torch.empty([m, h], dtype=dt, device=hidden.device)
    qkv = torch.empty([m, nqkv], dtype=dt, device=hidden.device)
    attn = torch.empty([m, nh * d], dtype=dt, device=hidden.device)
    act = torch.empty([m, inter], dtype=dt, device=hidden.device)
    out = torch.empty_like(hidden)
    for tm in hl.tile(m):
        res[tm, :] = hidden[tm, :]
    for layer in range(layers):
        # one LayerNorm feeds both branches
        for tm in hl.tile(m):
            v = res[tm, :].float()
            v = v - torch.mean(v, -1, keepdim=True)
            v = v * torch.rsqrt(torch.mean(v * v, -1, keepdim=True) + EPS)
            w_ln = ln_w[layer, :][None, :].float()
            b_ln = ln_b[layer, :][None, :].float()
            x[tm, :] = (v * w_ln + b_ln).to(dt)
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
            k = row[nh * d : 2 * nh * d].reshape(nh, d)
            k1 = k[:, :half]
            k2 = k[:, half : 2 * half]
            k_cache[layer, :, t, :] = torch.cat(
                [k1 * cos - k2 * sin, k2 * cos + k1 * sin, k[:, 2 * half :]], dim=-1
            )
            v_cache[layer, :, t, :] = row[2 * nh * d :].reshape(nh, d)
            q = row[: nh * d].reshape(nh, 1, d)
            q1 = q[:, :, :half]
            q2 = q[:, :, half : 2 * half]
            c = cos[None, :, :]
            s = sin[None, :, :]
            q = torch.cat(
                [q1 * c - q2 * s, q2 * c + q1 * s, q[:, :, 2 * half :]], dim=-1
            )
            m_i = hl.full([nh, 1], float("-inf"), dtype=torch.float32)
            l_i = hl.zeros([nh, 1], dtype=torch.float32)
            acc_o = hl.zeros([nh, 1, d], dtype=torch.float32)
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
            attn[0, :] = (acc_o / l_i[:, :, None]).to(dt).reshape(nh * d)
        # the MLP's input projection reads the same normalized input
        for tm, tn in hl.tile([m, inter]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(h):
                acc = hl.dot(x[tm, tk], w_in[layer, tk, tn], acc=acc)
            acc = acc + b_in[layer, tn][None, :].float()
            act[tm, tn] = torch.nn.functional.gelu(acc.to(dt), approximate="tanh")
        # both branches add into the residual
        for tm, tn in hl.tile([m, h]):
            acc = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(nh * d):
                acc = hl.dot(attn[tm, tk], w_o[layer, tk, tn], acc=acc)
            mlp = hl.zeros([tm, tn], dtype=torch.float32)
            for tk in hl.tile(inter):
                mlp = hl.dot(act[tm, tk], w_out[layer, tk, tn], acc=mlp)
            mlp = mlp + b_out[layer, tn][None, :].float()
            res[tm, tn] = res[tm, tn] + acc.to(dt) + mlp.to(dt)
    for tm in hl.tile(m):
        out[tm, :] = res[tm, :]
    return out


# %%
# Reference
# ---------


# %%
def _layer_norm_ref(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    v = x.float()
    v = v - torch.mean(v, -1, keepdim=True)
    v = v * torch.rsqrt(torch.mean(v * v, -1, keepdim=True) + EPS)
    return (v * weight.float() + bias.float()).to(x.dtype)


def _linear_ref(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (x.float() @ weight.float()).to(x.dtype)


def _rope_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = cos.size(-1)
    x1, x2, rest = x[..., :half], x[..., half : 2 * half], x[..., 2 * half :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest], dim=-1)


def parallel_layer_ref(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    s: dict[str, torch.Tensor],
    layer: int,
    pos: torch.Tensor,
) -> torch.Tensor:
    """One parallel-residual decoder layer on ``hidden`` [1, H] (updates ``s``)."""
    _, nh, ctx, d = s["k_cache"].shape
    dt = hidden.dtype
    t = int(pos[0])
    ang = t * w["inv_freq"]
    cos, sin = torch.cos(ang).to(dt), torch.sin(ang).to(dt)
    x = _layer_norm_ref(hidden, w["ln_w"][layer], w["ln_b"][layer])
    qkv = _linear_ref(x, w["w_qkv"][layer])[0]
    query = _rope_ref(qkv[: nh * d].view(nh, d), cos, sin)
    k_cache, v_cache = s["k_cache"][layer], s["v_cache"][layer]
    k_cache[:, t] = _rope_ref(qkv[nh * d : 2 * nh * d].view(nh, d), cos, sin)
    v_cache[:, t] = qkv[2 * nh * d :].view(nh, d)
    scores = query.float()[:, None, :] @ k_cache.float().transpose(1, 2) * d**-0.5
    mask = torch.arange(ctx, device=hidden.device) <= t
    scores = torch.where(mask, scores, float("-inf"))
    probs = torch.softmax(scores, -1).to(dt)
    attn = (probs.float() @ v_cache.float()).to(dt).view(1, -1)
    a = x.float() @ w["w_in"][layer].float() + w["b_in"][layer].float()
    act = torch.nn.functional.gelu(a.to(dt), approximate="tanh")
    mlp = act.float() @ w["w_out"][layer].float() + w["b_out"][layer].float()
    return hidden + _linear_ref(attn, w["w_o"][layer]) + mlp.to(dt)


def parallel_residual_stack_ref(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    s: dict[str, torch.Tensor],
    pos: torch.Tensor,
) -> torch.Tensor:
    """Eager PyTorch reference for :func:`parallel_residual_stack` (updates
    ``s``)."""
    for layer in range(w["ln_w"].size(0)):
        hidden = parallel_layer_ref(hidden, w, s, layer, pos)
    return hidden


# %%
WEIGHT_NAMES = (
    "ln_w",
    "ln_b",
    "w_qkv",
    "w_o",
    "w_in",
    "b_in",
    "w_out",
    "b_out",
    "inv_freq",
)
STATE_NAMES = ("k_cache", "v_cache")


def make_parallel_stack_inputs(
    layers: int = 4,
    dim: int = 4096,
    intermediate: int = 2048,
    heads: int = 2,
    head_dim: int = 256,
    rotary_dim: int = 64,
    context: int = 2048,
    position: int = 100,
    rope_theta: float = 1e4,
    device: torch.device | str = DEVICE,
) -> tuple[
    torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor
]:
    """Random stacked weights and caches; the defaults are one TP8 rank of
    GPT-J-6B (16 heads of 256, MLP 16384, rotary over 64 dims)."""
    bf16 = torch.bfloat16

    def weight(k: int, out: int) -> torch.Tensor:
        return (torch.randn(layers, k, out, device=device) * k**-0.5).to(bf16)

    def vec(k: int, scale: float, offset: float = 0.0) -> torch.Tensor:
        return (offset + scale * torch.randn(layers, k, device=device)).to(bf16)

    def cache() -> torch.Tensor:
        shape = (layers, heads, context, head_dim)
        return torch.randn(*shape, device=device).to(bf16)

    half = rotary_dim // 2
    exponent = torch.arange(half, dtype=torch.float32, device=device) * 2 / rotary_dim
    w = {
        "ln_w": vec(dim, 0.1, 1.0),
        "ln_b": vec(dim, 0.1),
        "w_qkv": weight(dim, 3 * heads * head_dim),
        "w_o": weight(heads * head_dim, dim),
        "w_in": weight(dim, intermediate),
        "b_in": vec(intermediate, 0.1),
        "w_out": weight(intermediate, dim),
        "b_out": vec(dim, 0.1),
        "inv_freq": 1.0 / rope_theta**exponent,
    }
    s = {"k_cache": cache(), "v_cache": cache()}
    hidden = torch.randn(1, dim, device=device).to(bf16)
    pos = torch.tensor([position], dtype=torch.int32, device=device)
    return hidden, w, s, pos


def parallel_stack_args(
    hidden: torch.Tensor,
    w: dict[str, torch.Tensor],
    s: dict[str, torch.Tensor],
    pos: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Positional arguments of :func:`parallel_residual_stack`."""
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
    hidden, w, s, pos = make_parallel_stack_inputs(layers=4)
    s_ref = {k: v.clone() for k, v in s.items()}
    expected = parallel_residual_stack_ref(hidden, w, s_ref, pos)
    result = parallel_residual_stack(*parallel_stack_args(hidden, w, s, pos))
    # bf16 rounding differences compound over the layers.
    torch.testing.assert_close(result, expected, atol=1e-1, rtol=5e-2)
    for name in STATE_NAMES:
        torch.testing.assert_close(s[name], s_ref[name], atol=5e-2, rtol=2e-2)


if __name__ == "__main__":
    main()
