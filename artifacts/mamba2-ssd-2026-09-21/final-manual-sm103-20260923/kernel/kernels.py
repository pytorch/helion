"""Four ordinary stages with fresh FP32 prefix and processed-delta transport.

Only returned scratch buffers are written; state exports FP32 prefix/rawdt.
Full chunk128, no optional gate, initial state, resets, dt bias or softplus.
"""

from __future__ import annotations

from typing import Any

import torch

import helion
import helion.language as hl


@helion.aot_kernel(backend="cute", static_shapes=True)
def mamba_native_state(
    x: torch.Tensor,
    b: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_size = hl.specialize(chunk_size)
    batch, length, heads, dim = x.shape
    groups, dstate = b.shape[2:]
    chunks = length // chunk_size
    # Validated contiguous public inputs: split the time axis without a copy.
    # Direct chunk-local indices retain ordinary host-view semantics and avoid
    # arithmetic after a fixed-width index cast in the cooperative-copy proof.
    dt_chunks = dt.reshape(batch, chunks, chunk_size, heads)
    x_chunks = x.reshape(batch, chunks, chunk_size, heads, dim)
    b_chunks = b.reshape(batch, chunks, chunk_size, groups, dstate)
    states = torch.empty(
        (batch, chunks, heads, dim, dstate), dtype=torch.float32, device=x.device
    )
    terminal = torch.empty((batch, chunks, heads), dtype=torch.float32, device=x.device)
    prefix = torch.empty(
        (batch, chunks, heads, chunk_size), dtype=torch.float32, device=x.device
    )
    rawdt = torch.empty(
        (batch, chunks, heads, chunk_size), dtype=torch.float32, device=x.device
    )
    for bi, ci, hi, ni, pi in hl.tile(
        [batch, chunks, heads, dstate, dim],
        block_size=[1, 1, 1, None, None],
    ):
        ki = hl.arange(chunk_size)
        delta = dt_chunks[bi.begin, ci.begin, ki, hi.begin].float().clamp(min=0.0)
        decay = hl.cumsum(delta * a[hi.begin].float(), dim=0)
        last = decay[chunk_size - 1]
        xv = x_chunks[bi.begin, ci.begin, ki, hi.begin, pi]
        bv = b_chunks[bi.begin, ci.begin, ki, hi.begin // (heads // groups), ni].float()
        scale = torch.exp((last - decay).clamp(max=0.0)) * delta
        weighted_b = (bv * scale[:, None]).to(x.dtype)
        acc = hl.dot(xv.T, weighted_b)
        states[bi.begin, ci.begin, hi.begin, pi, ni] = acc
        ni_begin: Any = ni.begin
        pi_begin: Any = pi.begin
        hl.store(
            terminal,
            [bi.begin, ci.begin, hi.begin],
            last,
            extra_mask=(ni_begin == 0) & (pi_begin == 0),
        )
        hl.store(
            prefix,
            [bi.begin, ci.begin, hi.begin, ki],
            decay,
            extra_mask=(ni_begin == 0) & (pi_begin == 0),
        )
        hl.store(
            rawdt,
            [bi.begin, ci.begin, hi.begin, ki],
            delta,
            extra_mask=(ni_begin == 0) & (pi_begin == 0),
        )
    return states, terminal, prefix, rawdt


@helion.aot_kernel(backend="cute", static_shapes=True)
def mamba_native_pass(
    states: torch.Tensor,
    terminal: torch.Tensor,
    incoming: torch.Tensor,
) -> torch.Tensor:
    batch, chunks, heads, dim, dstate = states.shape
    for bi, hi, pi, ni in hl.tile(
        [batch, heads, dim, dstate], block_size=[1, 1, None, None]
    ):
        carry = hl.zeros([pi, ni], dtype=torch.float32)
        for chunk in hl.grid(chunks):
            incoming[bi.begin, chunk, hi.begin, pi, ni] = carry.to(incoming.dtype)
            decay = torch.exp(terminal[bi.begin, chunk, hi.begin].float())
            carry = decay * carry + states[bi.begin, chunk, hi.begin, pi, ni].float()
    return incoming


@helion.aot_kernel(backend="cute", static_shapes=True)
def mamba_native_cb(
    b: torch.Tensor,
    c: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    chunk_size = hl.specialize(chunk_size)
    batch, length, groups, dstate = b.shape
    chunks = length // chunk_size
    b_chunks = b.reshape(batch, chunks, chunk_size, groups, dstate)
    c_chunks = c.reshape(batch, chunks, chunk_size, groups, dstate)
    cb = torch.empty(
        (batch, chunks, groups, chunk_size, chunk_size),
        dtype=torch.float32,
        device=b.device,
    )
    for bi, ci, gi, mi, ki in hl.tile(
        [batch, chunks, groups, chunk_size, chunk_size],
        block_size=[1, 1, 1, None, None],
    ):
        ni = hl.arange(dstate)
        cv = c_chunks[bi.begin, ci.begin, mi, gi.begin, ni]
        bv = b_chunks[bi.begin, ci.begin, ki, gi.begin, ni]
        acc = hl.dot(cv, bv.T)
        # Mamba materializes the complete FP32 CB matrix; causality is applied
        # after FP32 decay/dt weighting in the output stage, before its BF16 cast.
        cb[bi.begin, ci.begin, gi.begin, mi, ki] = acc
    return cb


@helion.aot_kernel(backend="cute", static_shapes=True)
def mamba_native_output(
    x: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    prefix: torch.Tensor,
    rawdt: torch.Tensor,
    cb: torch.Tensor,
    incoming: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    chunk_size = hl.specialize(chunk_size)
    batch, length, heads, dim = x.shape
    groups, dstate = c.shape[2:]
    chunks = length // chunk_size
    x_chunks = x.reshape(batch, chunks, chunk_size, heads, dim)
    c_chunks = c.reshape(batch, chunks, chunk_size, groups, dstate)
    out = torch.empty(
        (batch, chunks, chunk_size, heads, dim), device=x.device, dtype=x.dtype
    )
    for bi, ci, hi, mi, pi in hl.tile(
        [batch, chunks, heads, chunk_size, dim],
        block_size=[1, 1, 1, None, None],
    ):
        ni = hl.arange(dstate)
        cv = c_chunks[
            bi.begin,
            ci.begin,
            mi,
            hi.begin // (heads // groups),
            ni,
        ]
        prev = incoming[bi.begin, ci.begin, hi.begin, pi, ni].to(c.dtype)
        acc = hl.dot(cv, prev.T)
        ki = hl.arange(chunk_size)
        delta = rawdt[bi.begin, ci.begin, hi.begin, ki]
        gk = prefix[bi.begin, ci.begin, hi.begin, ki]
        gm = gk[mi.index]
        acc = acc * torch.exp(gm)[:, None]
        raw_cb = cb[bi.begin, ci.begin, hi.begin // (heads // groups), mi, ki].float()
        weighted = raw_cb * torch.exp((gm[:, None] - gk[None, :]).clamp(max=0.0))
        weighted = weighted * delta[None, :]
        weighted = torch.where(mi.index[:, None] >= ki[None, :], weighted, 0.0)
        operand = weighted.to(x.dtype)
        xv = x_chunks[bi.begin, ci.begin, ki, hi.begin, pi]
        acc = acc + hl.dot(operand, xv)
        residual = (
            x_chunks[bi.begin, ci.begin, mi, hi.begin, pi].float() * d[hi.begin].float()
        )
        out[bi.begin, ci.begin, mi, hi.begin, pi] = (acc + residual).to(out.dtype)
    return out.reshape_as(x)


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
    incoming: torch.Tensor,
) -> torch.Tensor:
    result = torch.empty_like(incoming)
    carry = torch.zeros_like(states[:, 0], dtype=torch.float32)
    for chunk in range(states.shape[1]):
        result[:, chunk] = carry.to(incoming.dtype)
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


HELION_KERNELS = (
    mamba_native_state,
    mamba_native_pass,
    mamba_native_cb,
    mamba_native_output,
)
HELION_REFERENCES = (state_reference, pass_reference, cb_reference, output_reference)
