from __future__ import annotations

import torch

import helion
import helion.language as hl

BT: int = 16
DK: int = 128
DV: int = 128
CPC: int = 4
LOG2_E: float = 1.4426950408889634
KDA_PREPARE_CONFIG = helion.Config(
    block_sizes=[],
    num_warps=4,
    num_stages=2,
    indexing="pointer",
)


def _bf16(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16).float()


def _block_inverse(lower: torch.Tensor) -> torch.Tensor:
    """The two-block FP16 inverse arithmetic used by the BT16 schedule."""

    row = hl.arange(BT)
    same_half = (row[:, None] < 8) == (row[None, :] < 8)
    lower_half = (row[:, None] >= 8) & (row[None, :] < 8)
    diagonal = torch.where(same_half, lower, 0.0)
    coupling = torch.where(lower_half, lower, 0.0)
    diagonal2 = hl.dot(
        diagonal.to(torch.float16),
        diagonal.T.to(torch.float16),
        out_dtype=torch.float32,
    )
    diagonal4 = hl.dot(
        diagonal2.to(torch.float16),
        diagonal2.T.to(torch.float16),
        out_dtype=torch.float32,
    )
    eye = (row[:, None] == row[None, :]).float()
    inverse = eye - diagonal.to(torch.float16).float()
    inverse = inverse.to(torch.float16).float() + hl.dot(
        inverse.to(torch.float16),
        diagonal2.T.to(torch.float16),
        out_dtype=torch.float32,
    )
    inverse = inverse.to(torch.float16).float() + hl.dot(
        inverse.to(torch.float16),
        diagonal4.T.to(torch.float16),
        out_dtype=torch.float32,
    )
    first = hl.dot(
        inverse.to(torch.float16),
        coupling.T.to(torch.float16),
        out_dtype=torch.float32,
    )
    lower_left = hl.dot(
        first.to(torch.float16),
        inverse.T.to(torch.float16),
        out_dtype=torch.float32,
    )
    return torch.where(lower_half, -lower_left, inverse).to(torch.bfloat16)


@helion.kernel(
    backend="cute",
    static_shapes=True,
    fast_math=True,
    config=KDA_PREPARE_CONFIG,
)
def kda_chunk_prepare(
    q: torch.Tensor,
    k: torch.Tensor,
    gate: torch.Tensor,
    beta_logits: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_chunks: torch.Tensor,
    chunk_to_seq: torch.Tensor,
    kd: torch.Tensor,
    qd: torch.Tensor,
    ak: torch.Tensor,
    aq: torch.Tensor,
    g_total: torch.Tensor,
    scale: float | None,
    gate_scale_log2: float,
) -> None:
    """Semantic carrier for the exact five-factor prepare contract."""

    tokens = q.size(1)
    heads = hl.specialize(q.size(2))
    chunks = chunk_to_seq.size(0)
    groups = (chunks + CPC - 1) // CPC
    q_rows = q.view(tokens * heads, DK)
    k_rows = k.view(tokens * heads, DK)
    gate_rows = gate.view(tokens * heads, DK)
    beta_rows = beta_logits.view(tokens * heads)
    kd_rows = kd.view(heads * chunks * BT, DK)
    qd_rows = qd.view(heads * chunks * BT, DK)
    ak_rows = ak.view(heads * chunks * BT, DK)
    aq_rows = aq.view(heads * chunks, BT * BT)
    gt_rows = g_total.view(heads * chunks, DK)

    for tile_group, tile_head in hl.tile(
        [groups, heads],
        block_size=[1, 1],
    ):
        for local_chunk in range(CPC):
            chunk = tile_group.id * CPC + local_chunk
            if chunk < chunks:
                sequence = chunk_to_seq[chunk].long()
                sequence_begin = cu_seqlens[sequence].long()
                sequence_end = cu_seqlens[sequence + 1].long()
                chunk_begin = sequence_begin + (chunk - cu_chunks[sequence]) * BT
                token_lane = hl.arange(BT)
                feature = hl.arange(DK)
                token = chunk_begin + token_lane
                valid = token < sequence_end
                input_row = token * heads + tile_head.id
                factor_row = (tile_head.id * chunks + chunk) * BT + token_lane

                q_raw = hl.load(
                    q_rows,
                    [input_row[:, None], feature[None, :]],
                    extra_mask=valid[:, None],
                ).float()
                k_raw = hl.load(
                    k_rows,
                    [input_row[:, None], feature[None, :]],
                    extra_mask=valid[:, None],
                ).float()
                gate_raw = hl.load(
                    gate_rows,
                    [input_row[:, None], feature[None, :]],
                    extra_mask=valid[:, None],
                ).float()
                dt = dt_bias[tile_head.id, feature].float()
                log_decay_scale = torch.exp2(a_log[tile_head.id].float() * LOG2_E)
                gate_increment = gate_scale_log2 * (
                    torch.tanh(log_decay_scale * (gate_raw + dt[None, :]) * 0.5) * 0.5
                    + 0.5
                )
                gate_increment = torch.where(valid[:, None], gate_increment, 0.0)
                gate_prefix = torch.cumsum(gate_increment, dim=0)
                exp_gate = torch.exp2(torch.clamp(gate_prefix, min=-126.0))
                gamma = torch.where((token_lane == BT - 1)[:, None], exp_gate, 0.0).sum(
                    0
                )

                q_ss = (q_raw * q_raw).sum(-1)
                k_ss = (k_raw * k_raw).sum(-1)
                q_inv = torch.rsqrt(torch.clamp(q_ss, min=1.0e-24))
                k_inv = torch.rsqrt(torch.clamp(k_ss, min=1.0e-24))
                q_norm = _bf16(q_raw * q_inv[:, None])
                k_norm = k_raw * k_inv[:, None]
                exp_gate_bf16 = _bf16(exp_gate)
                kd_value = _bf16(_bf16(k_norm) * exp_gate_bf16)
                ki_value = _bf16(k_norm * torch.reciprocal(exp_gate))
                qd_value = _bf16(q_norm * exp_gate_bf16)
                if scale is not None:
                    scale_bf16 = (
                        torch.full_like(q_inv, scale).to(torch.bfloat16).float()
                    )
                    qd_value = _bf16(qd_value * scale_bf16[:, None])

                beta = torch.sigmoid(
                    hl.load(beta_rows, [input_row], extra_mask=valid).float()
                )
                kk = hl.dot(
                    kd_value.to(torch.bfloat16),
                    ki_value.T.to(torch.bfloat16),
                    out_dtype=torch.float32,
                )
                qk = hl.dot(
                    qd_value.to(torch.bfloat16),
                    ki_value.T.to(torch.bfloat16),
                    out_dtype=torch.float32,
                )
                causal = token_lane[:, None] >= token_lane[None, :]
                strict = token_lane[:, None] > token_lane[None, :]
                lower = torch.where(strict, kk * beta[:, None], 0.0)
                inverse = _block_inverse(lower)
                inverse_beta = (
                    inverse.float() * beta[None, :].to(torch.bfloat16).float()
                ).to(torch.bfloat16)
                qk = torch.where(causal, qk, 0.0).to(torch.bfloat16)
                aq_value = hl.dot(
                    qk,
                    inverse_beta,
                    out_dtype=torch.float32,
                ).to(torch.bfloat16)
                kg = (ki_value * gamma[None, :].to(torch.bfloat16).float()).to(
                    torch.bfloat16
                )
                ak_value = hl.dot(
                    inverse_beta.T,
                    kg,
                    out_dtype=torch.float32,
                ).to(torch.bfloat16)

                if scale is None:
                    hl.store(
                        kd_rows,
                        [factor_row[:, None], (feature ^ 8)[None, :]],
                        kd_value.to(torch.bfloat16),
                    )
                    hl.store(
                        qd_rows,
                        [factor_row[:, None], (feature ^ 8)[None, :]],
                        qd_value.to(torch.bfloat16),
                    )
                else:
                    hl.store(
                        kd_rows,
                        [factor_row[:, None], feature[None, :]],
                        kd_value.to(torch.bfloat16),
                    )
                    hl.store(
                        qd_rows,
                        [factor_row[:, None], feature[None, :]],
                        qd_value.to(torch.bfloat16),
                    )
                ak_physical_row = (tile_head.id * chunks + chunk) * BT + (
                    token_lane ^ 8
                )
                hl.store(
                    ak_rows,
                    [ak_physical_row[:, None], feature[None, :]],
                    ak_value,
                )
                storage_col = token_lane[None, :] ^ 8
                byte_offset = 2 * (token_lane[:, None] * BT + storage_col)
                pair_index = (byte_offset ^ (((byte_offset >> 7) & 1) << 4)) // 2
                aq_rows[tile_head.id * chunks + chunk, pair_index] = aq_value
                gt_rows[tile_head.id * chunks + chunk, feature] = gamma


KDA_RECURRENCE_CONFIG = helion.Config(
    block_sizes=[64],
    num_warps=8,
    num_stages=2,
    indexing="pointer",
    pid_type="flat",
)


@helion.kernel(backend="cute", static_shapes=True, config=KDA_RECURRENCE_CONFIG)
def kda_chunk_recurrence(
    kd: torch.Tensor,
    qd: torch.Tensor,
    ak: torch.Tensor,
    aq: torch.Tensor,
    g_total: torch.Tensor,
    values: torch.Tensor,
    output: torch.Tensor,
    state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_chunks: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Semantic carrier for the exact five-factor BT16 recurrence/output path.

    ``ak`` and ``aq`` expose the physical images written by the matching prepare
    carrier.  The explicit inverse index maps below make this function's eager
    meaning the logical four-contraction recurrence instead of treating those
    images as ordinary row-major matrices.
    """

    heads = hl.specialize(values.size(2))
    value_width = hl.specialize(values.size(3))
    key_width = hl.specialize(kd.size(3))
    total_chunks = aq.size(2)
    sequences = cu_seqlens.size(0) - 1

    kd_rows = kd.view(-1, key_width)
    qd_rows = qd.view(-1, key_width)
    ak_rows = ak.view(-1, key_width)
    aq_rows = aq.view(-1, BT * BT)
    gt_rows = g_total.view(-1, key_width)
    value_rows = values.view(-1, value_width)
    output_rows = output.view(-1, value_width)
    block_v = hl.register_block_size(64, value_width)

    for tile_sequence, tile_head, tile_v in hl.tile(
        [sequences, heads, value_width], block_size=[1, 1, block_v]
    ):
        begin = cu_seqlens[tile_sequence.id].long()
        end = cu_seqlens[tile_sequence.id + 1].long()
        chunk_begin = cu_chunks[tile_sequence.id].long()
        state_value = state[
            tile_sequence.id,
            tile_head.id,
            tile_v.index,
            :,
        ].float()

        for token_tile in hl.tile(end - begin, block_size=BT):
            global_chunk = chunk_begin + token_tile.id
            token = begin + token_tile.index
            valid = token < end
            row = token * heads + tile_head.id
            factor_base = (tile_head.id * total_chunks + global_chunk) * BT
            factor_row = factor_base + token_tile.index
            feature = hl.arange(key_width)

            physical_feature = feature ^ 8
            kd_value = kd_rows[
                factor_row[:, None],
                physical_feature[None, :],
            ]
            projected = hl.dot(
                kd_value,
                state_value.T.to(kd.dtype),
                out_dtype=torch.float32,
            )
            value = hl.load(
                value_rows,
                [row, tile_v],
                extra_mask=valid[:, None],
            ).float()
            residual = _bf16(value - projected)

            qd_value = qd_rows[
                factor_row[:, None],
                physical_feature[None, :],
            ]
            result = hl.dot(
                qd_value,
                state_value.T.to(qd.dtype),
                out_dtype=torch.float32,
            )

            logical_row = token_tile.index[:, None]
            logical_col = hl.arange(BT)[None, :]
            storage_col = logical_col ^ 8
            byte_offset = 2 * (logical_row * BT + storage_col)
            pair_index = (byte_offset ^ (((byte_offset >> 7) & 1) << 4)) // 2
            aq_value = aq_rows[
                tile_head.id * total_chunks + global_chunk,
                pair_index,
            ]
            result = hl.dot(
                aq_value,
                residual.to(aq.dtype),
                acc=result,
                out_dtype=torch.float32,
            )
            hl.store(
                output_rows,
                [row, tile_v],
                result * scale,
                extra_mask=valid[:, None],
            )

            decay = gt_rows[
                tile_head.id * total_chunks + global_chunk,
                :,
            ]
            ak_value = ak_rows[
                (factor_base + (token_tile.index ^ 8))[:, None],
                feature[None, :],
            ]
            update = hl.dot(
                residual.T.to(ak.dtype),
                ak_value,
                out_dtype=torch.float32,
            )
            state_value = _bf16(state_value * decay[None, :] + update)

        state[
            tile_sequence.id,
            tile_head.id,
            tile_v.index,
            :,
        ] = state_value.to(state.dtype)

    return output, state


__all__ = [
    "BT",
    "CPC",
    "DK",
    "DV",
    "KDA_PREPARE_CONFIG",
    "KDA_RECURRENCE_CONFIG",
    "kda_chunk_prepare",
    "kda_chunk_recurrence",
]
