"""Output-only Mamba-2 SSD with processed dt and zero initial state.

The single-chunk entry computes the entire sequence in one call. ``mamba2_ssd``
is an explicit four-stage program for a sequence divisible by ``chunk_size``.
Inputs are contiguous token-major BF16 x/B/C/D and FP32 dt/A. Both programs
clamp dt below at zero and local decay differences above at zero, and retain
the BF16 inter-chunk state boundary. They do not implement z, dt softplus,
nonzero initial state, packed sequences, or a public final-state result.
"""

from __future__ import annotations

from typing import Any

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True, backend="cute", fast_math=False)
def mamba2_ssd_cute_scan_fused(
    x: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    batch, length, heads, dim = x.shape
    groups, state = b.shape[2:]
    out = torch.empty_like(x)
    length = hl.specialize(length)
    state = hl.specialize(state)
    for hb, row, col in hl.tile(
        [batch * heads, length, dim], block_size=[1, None, None]
    ):
        bi = hb.begin // heads
        h = hb.begin % heads
        group = h // (heads // groups)
        q = hl.arange(length)
        dts = dt[bi, q, h].float().clamp(min=0.0)
        decay = hl.cumsum(dts * a[h], dim=0)
        row_decay = decay[row]
        cb = hl.dot(c[bi, row, group, :], b[bi, q, group, :].T)
        weights = (
            cb
            * torch.exp((row_decay[:, None] - decay[None, :]).clamp(max=0.0))
            * dts[None, :]
        )
        weights = torch.where(row.index[:, None] >= q[None, :], weights, 0.0).to(
            x.dtype
        )
        value = hl.dot(weights, x[bi, q, h, col])
        value += x[bi, row, h, col].float() * d[h].float()
        out[bi, row, h, col] = value.to(x.dtype)
    return out


@helion.kernel(static_shapes=True, backend="cute", fast_math=False)
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


@helion.kernel(static_shapes=True, backend="cute", fast_math=False)
def mamba_native_pass(
    states: torch.Tensor,
    terminal: torch.Tensor,
) -> torch.Tensor:
    batch, chunks, heads, dim, dstate = states.shape
    incoming = torch.empty_like(states, dtype=torch.bfloat16)
    for bi, hi, pi, ni in hl.tile(
        [batch, heads, dim, dstate], block_size=[1, 1, None, None]
    ):
        carry = hl.zeros([pi, ni], dtype=torch.float32)
        for chunk in hl.grid(chunks):
            incoming[bi.begin, chunk, hi.begin, pi, ni] = carry.to(incoming.dtype)
            decay = torch.exp(terminal[bi.begin, chunk, hi.begin].float())
            carry = decay * carry + states[bi.begin, chunk, hi.begin, pi, ni].float()
    return incoming


@helion.kernel(static_shapes=True, backend="cute", fast_math=False)
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


@helion.kernel(static_shapes=True, backend="cute", fast_math=False)
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


def mamba2_ssd(
    x: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Run the four ordinary stages, including all scratch preparation."""
    states, terminal, prefix, rawdt = mamba_native_state(x, b, dt, a, chunk_size)
    incoming = mamba_native_pass(states, terminal)
    cb = mamba_native_cb(b, c, chunk_size)
    return mamba_native_output(x, c, d, prefix, rawdt, cb, incoming, chunk_size)


def main() -> None:
    from benchmarks.cute.compare_mamba2_ssd_backends import main as benchmark_main

    benchmark_main()


if __name__ == "__main__":
    main()
