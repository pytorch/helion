"""Original independent FP64 SSD oracle and per-stage rounding references."""

from __future__ import annotations

import torch


def reference_fp64(
    x: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """Independent unrounded FP64 oracle, including the two Mamba clamps."""
    repeats = x.shape[2] // b.shape[2]
    xh = x.double().permute(0, 2, 1, 3)
    bh = b.double().repeat_interleave(repeats, dim=2).permute(0, 2, 1, 3)
    ch = c.double().repeat_interleave(repeats, dim=2).permute(0, 2, 1, 3)
    dth = dt.double().permute(0, 2, 1).clamp(min=0.0)
    decay = (dth * a.double()[None, :, None]).cumsum(-1)
    weights = (ch @ bh.transpose(-1, -2)) * torch.exp(
        (decay[..., :, None] - decay[..., None, :]).clamp(max=0.0)
    )
    weights = torch.tril(weights * dth[..., None, :])
    value = weights @ xh + xh * d.double()[None, :, None, None]
    return value.permute(0, 2, 1, 3)


def tuning_reference(*inputs: torch.Tensor) -> torch.Tensor:
    return reference_fp64(*inputs).to(inputs[0].dtype)


def prefix_reference_values(
    dt: torch.Tensor, a: torch.Tensor, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure stage-reference intermediates, never used by the runtime runner."""
    batch, length, heads = dt.shape
    delta = dt.float().clamp(min=0.0)
    product = delta * a.float()[None, None, :]
    shape = (batch, length // chunk_size, chunk_size, heads)
    return delta.reshape(shape).transpose(2, 3), product.reshape(shape).cumsum(
        2
    ).transpose(2, 3)


def state_reference(
    x: torch.Tensor,
    b: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, length, heads, dim = x.shape
    groups, dstate = b.shape[2:]
    chunks = length // chunk_size
    delta, gc = prefix_reference_values(dt, a, chunk_size)
    xv = x.reshape(batch, chunks, chunk_size, heads, dim).permute(0, 1, 3, 2, 4)
    bv = b.reshape(batch, chunks, chunk_size, groups, dstate).permute(0, 1, 3, 2, 4)
    bv = bv.repeat_interleave(heads // groups, dim=2)
    scale = torch.exp((gc[..., -1:] - gc).clamp(max=0.0)) * delta
    weighted_b = (bv.float() * scale[..., None]).to(x.dtype)
    states = torch.matmul(xv.float().transpose(-1, -2), weighted_b.float())
    return (
        states,
        gc[..., -1].contiguous().clone(),
        gc.contiguous().clone(),
        delta.contiguous().clone(),
    )


def pass_reference(
    states: torch.Tensor,
    terminal: torch.Tensor,
) -> torch.Tensor:
    result = torch.empty_like(states, dtype=torch.bfloat16)
    carry = torch.zeros_like(states[:, 0], dtype=torch.float32)
    for chunk in range(states.shape[1]):
        result[:, chunk] = carry.to(result.dtype)
        decay = torch.exp(terminal[:, chunk].float())[..., None, None]
        carry = decay * carry + states[:, chunk].float()
    return result


def cb_reference(
    b: torch.Tensor,
    c: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    batch, length, groups, dstate = b.shape
    chunks = length // chunk_size
    bv = b.reshape(batch, chunks, chunk_size, groups, dstate).permute(0, 1, 3, 2, 4)
    cv = c.reshape(batch, chunks, chunk_size, groups, dstate).permute(0, 1, 3, 2, 4)
    return torch.matmul(cv.float(), bv.float().transpose(-1, -2))


def output_reference(
    x: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    prefix: torch.Tensor,
    rawdt: torch.Tensor,
    cb: torch.Tensor,
    incoming: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    batch, length, heads, dim = x.shape
    groups, dstate = c.shape[2:]
    chunks = length // chunk_size
    delta, gc = rawdt, prefix
    cv = c.reshape(batch, chunks, chunk_size, groups, dstate).permute(0, 1, 3, 2, 4)
    cv = cv.repeat_interleave(heads // groups, dim=2)
    previous = incoming.to(c.dtype).transpose(-1, -2)
    acc = torch.matmul(cv.float(), previous.float())
    acc = acc * torch.exp(gc)[..., None]
    raw_cb = cb.float().repeat_interleave(heads // groups, dim=2)
    weighted = raw_cb * torch.exp((gc[..., :, None] - gc[..., None, :]).clamp(max=0.0))
    weighted = weighted * delta[..., None, :]
    causal = torch.ones(
        (chunk_size, chunk_size), dtype=torch.bool, device=x.device
    ).tril()
    operand = torch.where(causal, weighted, 0.0).to(x.dtype)
    xv = x.reshape(batch, chunks, chunk_size, heads, dim).permute(0, 1, 3, 2, 4)
    acc = acc + torch.matmul(operand.float(), xv.float())
    residual = xv.float() * d.float()[None, None, :, None, None]
    return (acc + residual).to(x.dtype).permute(0, 1, 3, 2, 4).reshape(x.shape)


def _mamba2_ssd_recurrent_fused_reference(
    x: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor | None,
    chunk_size: int,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    seq_idx: torch.Tensor | None = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    return_final_states: bool = False,
    state_dtype: torch.dtype | None = None,
    chunk_contribution_dtype: torch.dtype = torch.float32,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """FP64 chunk recurrence with Mamba's local decay clamps."""
    batch, seqlen, nheads, headdim = x.shape
    ngroups, dstate = b.shape[2:]
    assert seqlen % chunk_size == 0
    repeats = nheads // ngroups
    xh = x.double().permute(0, 2, 1, 3)
    bh = b.double().repeat_interleave(repeats, dim=2).permute(0, 2, 1, 3)
    ch = c.double().repeat_interleave(repeats, dim=2).permute(0, 2, 1, 3)
    dth = dt.double().permute(0, 2, 1)
    if dt_bias is not None:
        dth = dth + dt_bias.double()[None, :, None]
    if dt_softplus:
        dth = torch.where(dth <= 20.0, torch.log1p(torch.exp(dth)), dth)
    dth = dth.clamp(min=dt_limit[0], max=dt_limit[1])
    out = torch.empty_like(xh)
    if initial_states is None:
        state = torch.zeros(
            (batch, nheads, headdim, dstate), dtype=torch.float64, device=x.device
        )
    else:
        state = initial_states.double().clone()
    causal = torch.ones(
        (chunk_size, chunk_size), device=x.device, dtype=torch.bool
    ).tril()
    for start in range(0, seqlen, chunk_size):
        end = start + chunk_size
        xq, bq, cq = (xh[:, :, start:end], bh[:, :, start:end], ch[:, :, start:end])
        dtq = dth[:, :, start:end]
        da = (dtq * a.double()[None, :, None]).cumsum(dim=-1)
        weights = (da[..., :, None] - da[..., None, :]).clamp(max=0.0).exp()
        weights = weights * dtq[..., None, :]
        weights = torch.where(causal, weights, 0.0)
        past_scale = da.exp()
        scale = (da[..., -1:] - da).clamp(max=0.0).exp() * dtq
        state_decay = da[..., -1].exp()
        if seq_idx is not None:
            seq = seq_idx[:, start:end]
            incoming_seq = seq_idx[:, start - 1] if start else seq[:, 0]
            weights = torch.where(
                (seq[:, :, None] == seq[:, None, :])[:, None], weights, 0.0
            )
            past_scale = torch.where(
                (seq == incoming_seq[:, None])[:, None], past_scale, 0.0
            )
            scale = torch.where((seq == seq[:, -1:])[:, None], scale, 0.0)
            state_decay = torch.where(
                (incoming_seq == seq[:, -1])[:, None], state_decay, 0.0
            )
        out[:, :, start:end] = cq @ bq.transpose(-1, -2) * weights @ xq
        out[:, :, start:end] += cq @ state.transpose(-1, -2) * past_scale[..., None]
        state = state * state_decay[..., None, None]
        state += xq.transpose(-1, -2) @ (bq * scale[..., None])
    result = out.permute(0, 2, 1, 3)
    if d is not None:
        if d.dim() == 1:
            result += x.double() * d.double()[None, None, :, None]
        else:
            result += x.double() * d.double()[None, None, :, :]
    if z is not None:
        zd = z.double()
        result *= zd * torch.sigmoid(zd)
    result = result.to(x.dtype)
    final_states = state.to(x.dtype if state_dtype is None else state_dtype)
    return (result, final_states) if return_final_states else result
