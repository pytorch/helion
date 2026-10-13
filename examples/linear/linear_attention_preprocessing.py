"""Fused inference preparation for the chunked gated-delta attention stages.

The public inputs are token first. Prepared tensors use the existing engine's
flattened batch/head/chunk order; no attention scaling or Q/K normalization is
performed here. The initial state is read only and is rounded once to BF16,
matching the engine's current state initialization.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

import helion
import helion.language as hl


class GatedDeltaPrefillInputs(NamedTuple):
    """Inputs to WY, state, and output kernels, without further materialization.

    ``q``, ``k``, and ``v`` have shape ``[B * HV * N, C, D_or_DV]``;
    ``beta`` is FP32 ``[B * HV * N, C]``; ``cumulative_log_decay`` is FP32
    ``[B * HV, N, C]``; ``initial_state`` is BF16 ``[B * HV, D, DV]``.
    ``N = ceil(T / C)``. Padded tokens have zero Q/K/V/beta and zero additional
    log decay. A returned V view aliases the read-only native V input.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    beta: torch.Tensor
    cumulative_log_decay: torch.Tensor
    initial_state: torch.Tensor

    @property
    def decay_last(self) -> torch.Tensor:
        """The engine's strided ``[B * HV, N]`` chunk-end view."""
        return self.cumulative_log_decay[:, :, -1]


@helion.kernel()
def pack_gated_delta_inputs_helion(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    state_out: torch.Tensor,
    copy_v: hl.constexpr,
) -> None:
    """One pointwise launch packs Q/K/V and transposes/rounds initial state.

    A common linear work domain covers all destinations. Explicit masks keep
    shorter destinations and padded tokens out of bounds; differing D and DV
    do not require another launch. ``copy_v=False`` leaves ``v_out`` untouched.
    """
    batch, tokens, query_heads, key_dim = q.shape
    value_heads, value_dim = v.size(2), v.size(3)
    chunk_size = q_out.size(1)
    padded_tokens = q_out.size(0) // (batch * value_heads) * chunk_size
    copies = value_heads // query_heads
    q_count = q_out.numel()
    v_count = v_out.numel() if copy_v else 0
    state_count = state_out.numel()
    count = max(q_count, v_count, state_count)
    q_source = q.view(-1)
    k_source = k.view(-1)
    state_source = initial_state.view(-1)
    q_target = q_out.view(-1)
    k_target = k_out.view(-1)
    state_target = state_out.view(-1)
    if copy_v:
        v_source = v.view(-1)
        v_target = v_out.view(-1)
    else:
        v_source = v
        v_target = v_out
    for tile in hl.tile(count):
        index = tile.index
        channel = index % key_dim
        token = index // key_dim % padded_tokens
        head = index // (key_dim * padded_tokens) % value_heads
        sequence = index // (key_dim * padded_tokens * value_heads)
        valid_q = (index < q_count) & (token < tokens)
        q_offset = (
            (sequence * tokens + token) * query_heads + head // copies
        ) * key_dim + channel
        query = hl.load(q_source, [q_offset], extra_mask=valid_q)
        key = hl.load(k_source, [q_offset], extra_mask=valid_q)
        hl.store(q_target, [index], query, extra_mask=index < q_count)
        hl.store(k_target, [index], key, extra_mask=index < q_count)

        if copy_v:
            value_channel = index % value_dim
            value_token = index // value_dim % padded_tokens
            value_head = index // (value_dim * padded_tokens) % value_heads
            value_sequence = index // (value_dim * padded_tokens * value_heads)
            v_offset = (
                (value_sequence * tokens + value_token) * value_heads + value_head
            ) * value_dim + value_channel
            value = hl.load(
                v_source,
                [v_offset],
                extra_mask=(index < v_count) & (value_token < tokens),
            )
            hl.store(
                v_target,
                [index],
                value,
                extra_mask=index < v_count,
            )

        state_value = index % value_dim
        state_key = index // value_dim % key_dim
        state_head = index // (value_dim * key_dim) % value_heads
        state_sequence = index // (value_dim * key_dim * value_heads)
        state_offset = (
            (state_sequence * value_heads + state_head) * value_dim + state_value
        ) * key_dim + state_key
        state = hl.load(
            state_source,
            [state_offset],
            extra_mask=index < state_count,
        )
        hl.store(
            state_target,
            [index],
            state.to(torch.bfloat16),
            extra_mask=index < state_count,
        )


@helion.kernel()
def prepare_gated_delta_gates_helion(
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    beta_out: torch.Tensor,
    cumulative_out: torch.Tensor,
) -> None:
    """FP32 sigmoid, unbounded softplus gate, and inclusive chunk cumsum.

    The softplus uses PyTorch's default threshold 20. Clamping only the argument
    to the unselected exponential branch prevents overflow at large positive
    inputs; it does not bound the returned softplus or the resulting decay.
    """
    batch, tokens, heads = a.shape
    chunk_size = hl.specialize(beta_out.size(1))
    chunks = beta_out.size(0) // (batch * heads)
    a_source, b_source = a.view(-1), b.view(-1)
    beta_target = beta_out.view(-1)
    cumulative_target = cumulative_out.view(-1)
    for row in hl.tile(beta_out.size(0)):
        head = row.index // chunks % heads
        sequence = row.index // (chunks * heads)
        rate = -torch.exp(A_log[head].float())
        bias = dt_bias[head].float()
        prefix = hl.zeros([row], dtype=torch.float32)
        # A sequential FP32 recurrence over C, parallel over independent rows.
        # This also works where the backend cannot scan a computed expression.
        for position in hl.grid(chunk_size):
            token = row.index % chunks * chunk_size + position
            valid = (row.index < beta_out.size(0)) & (token < tokens)
            offset = (sequence * tokens + token) * heads + head
            raw = hl.load(a_source, [offset], extra_mask=valid).float()
            shifted = raw + bias
            softplus = torch.where(
                shifted > 20.0,
                shifted,
                torch.log1p(torch.exp(torch.clamp(shifted, max=20.0))),
            )
            log_decay = rate * softplus
            # Activation maps padded zero inputs to nonzero values. Mask after
            # it, as the engine pads already activated gates and beta with zero.
            log_decay = torch.where(valid, log_decay, 0.0)
            beta = torch.sigmoid(hl.load(b_source, [offset], extra_mask=valid).float())
            prefix = prefix + log_decay
            destination = row.index * chunk_size + position
            hl.store(beta_target, [destination], torch.where(valid, beta, 0.0))
            hl.store(cumulative_target, [destination], prefix)


def prepare_gated_delta_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    chunk_size: int = 64,
) -> GatedDeltaPrefillInputs:
    """Prepare native GDN inference inputs in two Helion launches.

    Q/K are BF16 ``[B,T,HQ,D]``; V is BF16 ``[B,T,HV,DV]`` with ``HV % HQ=0``.
    Raw a/b have ``[B,T,HV]``; A_log/bias are FP32 ``[HV]``; the read-only FP32
    state is ``[B,HV,DV,D]``. Q/K/V/a/b/state use dense contiguous native storage.
    Q/K head ``h // (HV/HQ)`` feeds value head h.

    Token-major V generally cannot be viewed in the engine's flattened
    batch/head/chunk order. The pack launch copies it once in that case. For
    contiguous, unpadded V with one head, or one chunk and either a single batch
    or token, a strided chunk view is returned. Returned views keep V alive.
    """
    batch, tokens, query_heads, key_dim = q.shape
    value_heads, value_dim = v.size(2), v.size(3)
    if (
        k.shape != q.shape
        or v.shape[:2] != q.shape[:2]
        or query_heads <= 0
        or value_heads % query_heads
        or tokens <= 0
        or chunk_size <= 0
    ):
        raise ValueError("incompatible native Q/K/V shapes or chunk size")
    if (
        a.shape != (batch, tokens, value_heads)
        or b.shape != a.shape
        or A_log.shape != (value_heads,)
        or dt_bias.shape != A_log.shape
        or initial_state.shape != (batch, value_heads, value_dim, key_dim)
    ):
        raise ValueError("incompatible gate or V-major state shape")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)) or any(
        t.dtype != torch.float32 for t in (A_log, dt_bias, initial_state)
    ):
        raise ValueError("expected BF16 Q/K/V and FP32 parameters/state")
    chunks = (tokens + chunk_size - 1) // chunk_size
    rows = batch * value_heads * chunks
    q_chunks = torch.empty((rows, chunk_size, key_dim), dtype=q.dtype, device=q.device)
    k_chunks = torch.empty_like(q_chunks)
    view_v = (
        tokens == chunks * chunk_size
        and v.is_contiguous()
        and (value_heads == 1 or (chunks == 1 and (batch == 1 or chunk_size == 1)))
    )
    v_chunks = (
        v.permute(0, 2, 1, 3).view(rows, chunk_size, value_dim)
        if view_v
        else torch.empty((rows, chunk_size, value_dim), dtype=v.dtype, device=v.device)
    )
    state = torch.empty(
        (batch * value_heads, key_dim, value_dim),
        dtype=torch.bfloat16,
        device=initial_state.device,
    )
    beta = torch.empty((rows, chunk_size), dtype=torch.float32, device=b.device)
    cumulative = torch.empty_like(beta)
    pack_gated_delta_inputs_helion(
        q, k, v, initial_state, q_chunks, k_chunks, v_chunks, state, not view_v
    )
    prepare_gated_delta_gates_helion(a, b, A_log, dt_bias, beta, cumulative)
    return GatedDeltaPrefillInputs(
        q_chunks,
        k_chunks,
        v_chunks,
        beta,
        cumulative.view(batch * value_heads, chunks, chunk_size),
        state,
    )


def main() -> None:
    device = "cuda"
    q = torch.randn(1, 128, 2, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, 128, 4, 64, device=device, dtype=torch.bfloat16)
    a = torch.randn(1, 128, 4, device=device, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    parameters = torch.zeros(4, device=device)
    initial = torch.zeros(1, 4, 64, 64, device=device)
    prepared = prepare_gated_delta_prefill(
        q, k, v, a, b, parameters, parameters, initial
    )
    print(
        {
            name: tuple(value.shape)
            for name, value in zip(prepared._fields, prepared, strict=True)
        }
    )


if __name__ == "__main__":
    main()
