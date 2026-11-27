"""Device helpers of the KDA verify recurrence that the Kimi-K3 (``kda_mtp_verify``)
and GLM-5.3 Flash (``glm5_3_flash_tp4.kda_tp4``) megakernels share: the causal
conv's taps and weights over the 8 token lanes, and the delta rule's per-token
operands and step."""

from __future__ import annotations

import torch

import helion.language as hl


def conv_window(
    conv_state: torch.Tensor,
    sl: torch.Tensor,
    ok: torch.Tensor,
    col: torch.Tensor,
    j: int,
) -> torch.Tensor:
    """The committed conv window's row 3 + t - j at channels ``col`` [8, C],
    the conv input of token t - j < 0 (rows t >= j are unused)."""
    tt = hl.arange(8)[:, None]
    zt = tt * 0
    w = hl.load(
        conv_state,
        [sl + zt, torch.clamp(tt + 3 - j, max=2), col[None, :]],
        extra_mask=(tt < j) & ((zt + ok) != 0),
    )
    return w.to(torch.float32)


def conv_tap(
    stg: torch.Tensor,
    task: torch.Tensor,
    kind: int,
    lcol: torch.Tensor,
    w: torch.Tensor,
    j: int,
) -> torch.Tensor:
    """Conv input at token t - j [8, C]: row t - j of the projection output this
    task stored in its ``stg`` slot ``kind`` (columns ``lcol``), or the
    committed window ``w`` where t < j."""
    tt = hl.arange(8)[:, None]
    zt = tt * 0
    p = hl.load(
        stg,
        [task + zt, zt + kind, torch.clamp(tt - j, min=0), lcol[None, :]],
        extra_mask=tt >= j,
    )
    return torch.where(tt >= j, p, w)


def conv_weights(
    conv_w: torch.Tensor, kind: int, ch: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The conv weights of channels ``ch`` broadcast over the 8 tokens [8, C]
    (the taps' shape: a [C] vector broadcast at the multiply costs a layout
    conversion through shared memory)."""
    z = hl.arange(8)[:, None] * 0
    return (
        hl.load(conv_w, [z + kind, z, ch[None, :]]),
        hl.load(conv_w, [z + kind, z + 1, ch[None, :]]),
        hl.load(conv_w, [z + kind, z + 2, ch[None, :]]),
        hl.load(conv_w, [z + kind, z + 3, ch[None, :]]),
    )


def silu_conv(
    x0: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
) -> torch.Tensor:
    """SiLU(causal conv) [8, C]: y[t] = sum_k w[k] row[t + k]."""
    y = x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3
    return y * torch.sigmoid(y)


def token_operands(
    stg: torch.Tensor,
    sdec: torch.Tensor,
    sbeta: torch.Tensor,
    task: torch.Tensor,
    d: torch.Tensor,
    t: int,
) -> tuple[torch.Tensor, ...]:
    """Token ``t``'s recurrence operands: q, k, decay [1, 128], v [16, 1], beta."""
    zd = d[None, :] * 0
    j = hl.arange(16)[:, None]
    zj = j * 0
    q = hl.load(stg, [task + zd, zd, zd + t, d[None, :]])
    k = hl.load(stg, [task + zd, zd + 1, zd + t, d[None, :]])
    e = hl.load(sdec, [task + zd, zd + t, d[None, :]])
    v = hl.load(stg, [task + zj, zj + 2, zj + t, j])
    b = hl.load(sbeta, [task, t])
    return q, k, e, v, b


def delta_step(
    st: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    e: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One delta-rule step of state rows ``st`` [16, 128] -> (state, o [16, 1])."""
    st = st * e
    r = v - torch.sum(st * k, dim=1, keepdim=True)
    st = st + (b * r) * k
    return st, torch.sum(st * q, dim=1, keepdim=True)
