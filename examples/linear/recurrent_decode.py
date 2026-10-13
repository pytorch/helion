"""Raw-input recurrent decode programs, with no fixed configuration dispatch.

GDN and KDA normalize q/k and evaluate unbounded softplus/sigmoid gates inside
the kernel. State uses [slot, head, value, key]. MTP has two explicit programs: gdn_mtp_fp32 keeps FP32 normalized q/k;
gdn_mtp_fp16 rounds normalized q/k through FP16. Both retain every FP32
checkpoint and use the same raw update/verify flags.
Mamba updates indexed BF16 state; Retention reads its FP32 initial state and
writes a separate FP32 final state. Outputs retain the caller's dtype/layout.
"""

from __future__ import annotations

from typing import Any

import torch

import helion
import helion.language as hl


@helion.kernel(
    backend="cute", fast_math=False, static_shapes=True, cute_structural_policy="auto"
)
def gdn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    batch, heads, key_dim = q.shape
    value_heads, value_dim = (v.size(1), v.size(2))
    group_size = value_heads // heads
    for tb, th, tv in hl.tile([batch, value_heads, value_dim], block_size=[1, 1, None]):
        ib, ih = (tb.begin, th.begin)
        iq = ih // group_size
        kk = hl.arange(key_dim)
        query = q[ib, iq, kk].float()
        key = k[ib, iq, kk].float()
        query = query * torch.rsqrt((query * query).sum() + 1e-06)
        key = key * torch.rsqrt((key * key).sum() + 1e-06)
        raw_gate = a[ib, ih].float() + dt_bias[ih].float()
        softplus = torch.where(
            raw_gate <= 20.0, torch.log1p(torch.exp(raw_gate)), raw_gate
        )
        decay = torch.exp(-torch.exp(a_log[ih].float()) * softplus)
        beta = torch.sigmoid(b[ib, ih].float())
        slot = indices[ib].long()
        current = state[slot, ih, tv, kk].float()
        current = current * decay
        residual = v[ib, ih, tv].float() - (current * key[None, :]).sum(-1)
        residual = residual * beta
        current = current + residual[:, None] * key[None, :]
        output[ib, ih, tv] = (current * query[None, :]).sum(-1) * scale
        state[slot, ih, tv, kk] = current
    return output


@helion.kernel(
    backend="cute", fast_math=False, static_shapes=True, cute_structural_policy="auto"
)
def kda_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    batch, heads, dim = q.shape
    for tb, th, tv in hl.tile([batch, heads, v.size(2)], block_size=[1, 1, None]):
        ib, ih = (tb.begin, th.begin)
        kk = hl.arange(dim)
        query = q[ib, ih, kk].float()
        key = k[ib, ih, kk].float()
        query = (query * torch.rsqrt((query * query).sum() + 1e-06)).float()
        key = (key * torch.rsqrt((key * key).sum() + 1e-06)).float()
        shifted = raw_gate[ib, ih, kk].float() + dt_bias[ih * dim + kk].float()
        softplus = torch.where(
            shifted > 20.0,
            shifted,
            torch.log1p(torch.exp(torch.clamp(shifted, max=20.0))),
        )
        decay = torch.exp(-torch.exp(a_log[ih].float()) * softplus).float()
        prepared_beta = torch.sigmoid(beta[ib, ih].float()).float()
        slot = indices[ib, 0].long()
        current = state[slot, ih, tv.index, kk].float()
        current = current * decay[None, :]
        residual = (
            v[ib, ih, tv].float() - (current * key[None, :]).sum(-1)
        ) * prepared_beta
        current = current + residual[:, None] * key[None, :]
        output[ib, ih, tv] = (current * query[None, :]).sum(-1) * scale
        state[slot, ih, tv.index, kk] = current.to(state.dtype)
    return output


@helion.kernel(
    backend="cute", fast_math=False, static_shapes=True, cute_structural_policy="auto"
)
def gdn_mtp_fp16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    checkpoints: torch.Tensor,
    scale: float,
    tokens: hl.constexpr,
    update_state: hl.constexpr,
    cache_states: hl.constexpr,
) -> torch.Tensor:
    batch, _, heads, dim = q.shape
    value_heads = v.size(2)
    group = value_heads // heads
    token_ids = tuple(range(tokens))
    for tb, th, tv in hl.tile([batch, value_heads, v.size(3)], block_size=[1, 1, None]):
        ib, ih = (tb.begin, th.begin)
        kk = hl.arange(dim)
        shifted_gates = tuple(
            a[ib, token, ih].float() + dt_bias[ih].float() for token in token_ids
        )
        softplus_gates = tuple(
            torch.where(g <= 20.0, torch.log1p(torch.exp(g)), g) for g in shifted_gates
        )
        decays = tuple(
            torch.exp(-torch.exp(a_log[ih].float()) * s) for s in softplus_gates
        )
        betas = tuple(torch.sigmoid(b[ib, token, ih].float()) for token in token_ids)
        slot = indices[ib].long()
        current = state[slot, ih, tv, kk].float()
        for it in token_ids:
            query = q[ib, it, ih // group, kk].float()
            key = k[ib, it, ih // group, kk].float()
            query = query * torch.rsqrt((query * query).sum() + 1e-06)
            key = key * torch.rsqrt((key * key).sum() + 1e-06)
            query = query.to(torch.float16).float()
            key = key.to(torch.float16).float()
            decay = decays[it]
            beta = betas[it]
            current = current * decay
            residual = v[ib, it, ih, tv].float() - (current * key[None, :]).sum(-1)
            residual = residual * beta
            current = current + residual[:, None] * key[None, :]
            output[ib, it, ih, tv] = ((current * query[None, :]).sum(-1) * scale).to(
                output.dtype
            )
            if cache_states:
                checkpoints[ib, it, ih, tv, kk] = current
        if update_state:
            state[slot, ih, tv, kk] = current
    return output


@helion.kernel(
    backend="cute", fast_math=False, static_shapes=True, cute_structural_policy="auto"
)
def mamba_decode(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    dt_bias: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    batch, heads, value_dim = x.shape
    state_dim = state.size(3)
    heads_per_group = heads // b.size(1)
    for tb, th, tv in hl.tile([batch, heads, value_dim], block_size=[1, 1, None]):
        ib, ih = (tb.id, th.id)
        nn = hl.arange(state_dim)
        slot = indices[ib].long()
        current = state[slot, ih, tv.index, nn].float()
        raw_dt = dt[ib, ih, tv].float() + dt_bias[ih, tv].float()
        delta = torch.where(raw_dt <= 20.0, torch.log1p(torch.exp(raw_dt)), raw_dt)
        a_tile = a[ih, tv.index, nn].float()
        value = x[ib, ih, tv].float()
        group = ih // heads_per_group
        b_vec = b[ib, group, nn].float()
        c_vec = c[ib, group, nn].float()
        current = current * torch.exp(delta[:, None] * a_tile)
        current = current + (delta * value)[:, None] * b_vec[None, :]
        output[ib, ih, tv] = (current * c_vec[None, :]).sum(-1) + d[
            ih, tv
        ].float() * value
        state[slot, ih, tv.index, nn] = current
    return output


@helion.kernel(
    backend="cute", fast_math=False, static_shapes=True, cute_structural_policy="auto"
)
def retention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    output: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    batch, _, heads, key_dim = q.shape
    value_dim = v.size(3)
    for tb, th, tv in hl.tile([batch, heads, value_dim], block_size=[1, 1, None]):
        ib, ih = (tb.id, th.id)
        kk = hl.arange(key_dim)
        current = initial_state[ib, ih, tv.index, kk].float()
        decay = 1.0 - torch.exp2(-5.0 - th.index.float().sum())
        key = k[ib, 0, ih, kk].float()
        query = q[ib, 0, ih, kk].float()
        current = current * decay + v[ib, 0, ih, tv].float()[:, None] * key[None, :]
        output[ib, 0, ih, tv] = (current * query[None, :]).sum(-1) * scale
        final_state[ib, ih, tv.index, kk] = current
    return output


@helion.kernel(
    backend="cute", fast_math=False, static_shapes=True, cute_structural_policy="auto"
)
def gdn_mtp_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    checkpoints: torch.Tensor,
    scale: float,
    tokens: hl.constexpr,
    update_state: hl.constexpr,
    cache_states: hl.constexpr,
) -> torch.Tensor:
    batch, _, heads, key_dim = q.shape
    value_heads, value_dim = (v.size(2), v.size(3))
    group_size = value_heads // heads
    for tb, th, tv in hl.tile([batch, value_heads, value_dim], block_size=[1, 1, None]):
        ib, ih = (tb.id, th.id)
        iq = ih // group_size
        slot = indices[ib].long()
        kk = hl.arange(key_dim)
        current = state[slot, ih, tv.index, kk].float()
        for it in hl.static_range(int(tokens)):
            query = q[ib, it, iq, kk].float()
            key = k[ib, it, iq, kk].float()
            query = query * torch.rsqrt((query * query).sum() + 1e-06)
            key = key * torch.rsqrt((key * key).sum() + 1e-06)
            raw_gate = a[ib, it, ih].float() + dt_bias[ih].float()
            softplus = torch.where(
                raw_gate <= 20.0, torch.log1p(torch.exp(raw_gate)), raw_gate
            )
            decay = torch.exp(-torch.exp(a_log[ih].float()) * softplus)
            current = current * decay
            residual = v[ib, it, ih, tv].float() - (current * key[None, :]).sum(-1)
            residual = residual * torch.sigmoid(b[ib, it, ih].float())
            current = current + residual[:, None] * key[None, :]
            output[ib, it, ih, tv] = (current * query[None, :]).sum(-1) * scale
            if cache_states:
                checkpoints[ib, it, ih, tv.index, kk] = current
        if update_state:
            state[slot, ih, tv.index, kk] = current
    return output


def kernel_for(family: str, program: str | None = None) -> helion.Kernel:
    """Choose a mathematical program by API, never by batch size/configuration."""
    if program is not None:
        assert family == "gdn-mtp"
        return {"gdn-mtp-fp32": gdn_mtp_fp32, "gdn-mtp-fp16": gdn_mtp_fp16}[program]
    return {
        "gdn-decode": gdn_decode,
        "gdn-mtp": gdn_mtp_fp32,
        "kda-decode": kda_decode,
        "mamba2-decode": mamba_decode,
        "retention-decode": retention_decode,
    }[family]


def kernel_args(family: str, raw: tuple[Any, ...]) -> tuple[Any, ...]:
    """Project the full serving ABI using views; callers retain this tuple.

    GDN T1 keeps the unused checkpoint buffer in its outer raw contract. KDA T1
    retains its accepted-token/feature flags in that contract; with one token the
    clamped accepted position is always zero. Neither projection copies inputs.
    """
    if family == "gdn-decode":
        assert len(raw) == 15
        assert raw[12] == 1 and raw[13] is True and raw[14] is False
        assert raw[0].ndim == 4 and raw[0].size(1) == 1
        assert raw[8].shape == (raw[0].size(0),)
        squeezed = (0, 1, 2, 3, 4, 9)
        assert all(raw[index].size(1) == 1 for index in squeezed)
        return tuple(
            raw[index].squeeze(1) if index in squeezed else raw[index]
            for index in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11)
        )
    if family == "kda-decode":
        assert len(raw) == 17
        assert raw[13] == 1 and raw[14] is True and raw[15] is False and raw[16] is True
        assert raw[0].ndim == 3
        assert raw[8].shape == (raw[0].size(0), 1)
        assert raw[9].shape == (raw[0].size(0),)
        return tuple(raw[index] for index in (0, 1, 3, 4, 5, 6, 2, 7, 8, 10, 11))
    return raw


def main() -> None:
    """Print usage; the benchmark CLI owns device allocation and execution."""
    print("python -m benchmarks.cute.compare_recurrent_decode_backends --help")


if __name__ == "__main__":
    main()
