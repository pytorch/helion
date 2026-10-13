"""Semantic carrier for on-chip BT16 factors with an FP32 recurrent state.

Unlike the staged carrier, all chunk factors are local tensors.  This gives
backend region schedulers the complete producer/consumer graph without
changing the Helion language or dropping observable workspace stores.
"""

from __future__ import annotations

import torch

from .kda_prefill_kernels import _bf16
import helion
import helion.language as hl

BT = 16
DK = 128
LOG2_E = 1.4426950408889634


def _block_inverse(lower: torch.Tensor) -> torch.Tensor:
    """The two-block FP16 inverse arithmetic used by the BT16 schedule."""

    row = hl.arange(BT)
    same_half = (row[:, None] < 8) == (row[None, :] < 8)
    lower_half = (row[:, None] >= 8) & (row[None, :] < 8)
    diagonal = torch.where(same_half, lower, 0.0)
    coupling = torch.where(lower_half, lower, 0.0)
    diagonal2 = hl.dot(
        diagonal.to(torch.float16),
        diagonal.to(torch.float16),
        out_dtype=torch.float32,
    )
    diagonal4 = hl.dot(
        diagonal2.to(torch.float16),
        diagonal2.to(torch.float16),
        out_dtype=torch.float32,
    )
    eye = (row[:, None] == row[None, :]).float()
    inverse = eye - diagonal.to(torch.float16).float()
    inverse = inverse.to(torch.float16).float() + hl.dot(
        inverse.to(torch.float16),
        diagonal2.to(torch.float16),
        out_dtype=torch.float32,
    )
    inverse = inverse.to(torch.float16).float() + hl.dot(
        inverse.to(torch.float16),
        diagonal4.to(torch.float16),
        out_dtype=torch.float32,
    )
    first = hl.dot(
        inverse.to(torch.float16),
        coupling.to(torch.float16),
        out_dtype=torch.float32,
    )
    lower_left = hl.dot(
        first.to(torch.float16),
        inverse.to(torch.float16),
        out_dtype=torch.float32,
    )
    return torch.where(lower_half, -lower_left, inverse).to(torch.bfloat16)


@helion.kernel(backend="cute", static_shapes=True, fast_math=True)
def kda_prefill_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta_logits: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    output: torch.Tensor,
    final_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    gate_scale_log2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve split-factor rounding while retaining the FP32 state update."""
    heads = hl.specialize(q.size(2))
    value_width = hl.specialize(v.size(3))
    key_width = hl.specialize(q.size(3))
    sequences = cu_seqlens.size(0) - 1
    q_rows = q.view(-1, key_width)
    k_rows = k.view(-1, key_width)
    gate_rows = gate.view(-1, key_width)
    beta_rows = beta_logits.view(-1)
    v_rows = v.view(-1, value_width)
    out_rows = output.view(-1, value_width)
    block_v = hl.register_block_size(64, value_width)
    for seq, head, values in hl.tile(
        [sequences, heads, value_width], block_size=[1, 1, block_v]
    ):
        begin = cu_seqlens[seq.id].long()
        end = cu_seqlens[seq.id + 1].long()
        state = initial_state[seq.id, head.id, values.index, :].float()
        for chunk in hl.tile(end - begin, block_size=BT):
            feature = hl.arange(key_width)
            token_lane = hl.arange(BT)
            dt = dt_bias[head.id, feature].float()
            log_decay_scale = torch.exp2(a_log[head.id].float() * LOG2_E)
            token = begin + chunk.id * BT + token_lane
            valid = token < end
            row = token * heads + head.id
            q_raw = hl.load(
                q_rows,
                [row, feature],
                extra_mask=valid[:, None],
            ).float()
            k_raw = hl.load(
                k_rows,
                [row, feature],
                extra_mask=valid[:, None],
            ).float()
            gate_raw = hl.load(
                gate_rows,
                [row, feature],
                extra_mask=valid[:, None],
            ).float()
            increment = gate_scale_log2 * (
                torch.tanh(log_decay_scale * (gate_raw + dt[None, :]) * 0.5) * 0.5 + 0.5
            )
            increment = torch.where(valid[:, None], increment, 0.0)
            prefix = torch.cumsum(increment, dim=0)
            exp_gate = torch.exp2(torch.clamp(prefix, min=-126.0))
            gamma = torch.where((token_lane == BT - 1)[:, None], exp_gate, 0.0).sum(0)
            q_inv = torch.rsqrt(torch.clamp((q_raw * q_raw).sum(-1), min=1e-24))
            k_inv = torch.rsqrt(torch.clamp((k_raw * k_raw).sum(-1), min=1e-24))
            q_norm = _bf16(q_raw * q_inv[:, None])
            k_norm = k_raw * k_inv[:, None]
            exp_gate_bf16 = _bf16(exp_gate)
            kd = _bf16(_bf16(k_norm) * exp_gate_bf16).to(torch.bfloat16)
            ki = _bf16(k_norm * torch.reciprocal(exp_gate)).to(torch.bfloat16)
            qd = _bf16(q_norm * exp_gate_bf16).to(torch.bfloat16)
            beta = torch.sigmoid(hl.load(beta_rows, [row], extra_mask=valid).float())
            kk = hl.dot(kd, ki.T, out_dtype=torch.float32)
            qk = hl.dot(qd, ki.T, out_dtype=torch.float32)
            causal = token_lane[:, None] >= token_lane[None, :]
            strict = token_lane[:, None] > token_lane[None, :]
            inverse = _block_inverse(torch.where(strict, kk * beta[:, None], 0.0))
            inverse_beta = (
                inverse.float() * beta[None, :].to(torch.bfloat16).float()
            ).to(torch.bfloat16)
            qk = torch.where(causal, qk, 0.0).to(torch.bfloat16)
            aq = hl.dot(qk, inverse_beta, out_dtype=torch.float32).to(torch.bfloat16)
            kg = (ki.float() * gamma[None, :].to(torch.bfloat16).float()).to(
                torch.bfloat16
            )
            ak = hl.dot(inverse_beta.T, kg, out_dtype=torch.float32).to(torch.bfloat16)
            projected = hl.dot(kd, state.T.to(torch.bfloat16), out_dtype=torch.float32)
            raw_v = hl.load(v_rows, [row, values], extra_mask=valid[:, None]).float()
            residual = _bf16(raw_v - projected).to(torch.bfloat16)
            result = hl.dot(qd, state.T.to(torch.bfloat16), out_dtype=torch.float32)
            result = hl.dot(aq, residual, acc=result, out_dtype=torch.float32)
            hl.store(
                out_rows,
                [row, values],
                result * scale,
                extra_mask=valid[:, None],
            )
            update = hl.dot(residual.T, ak, out_dtype=torch.float32)
            state = state * gamma[None, :] + update
        final_state[seq.id, head.id, values.index, :] = state
    return output, final_state


@helion.kernel(backend="cute", static_shapes=True, fast_math=True)
def kda_prefill_native_math(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta_logits: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    output: torch.Tensor,
    final_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    gate_scale_log2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit beta/residual/inverse DAG for the asynchronous TMEM schedule."""
    heads = hl.specialize(q.size(2))
    value_width = hl.specialize(v.size(3))
    key_width = hl.specialize(q.size(3))
    sequences = cu_seqlens.size(0) - 1
    q_rows = q.view(-1, key_width)
    k_rows = k.view(-1, key_width)
    gate_rows = gate.view(-1, key_width)
    beta_rows = beta_logits.view(-1)
    v_rows = v.view(-1, value_width)
    out_rows = output.view(-1, value_width)
    block_v = hl.register_block_size(64, value_width)
    for seq, head, values in hl.tile(
        [sequences, heads, value_width], block_size=[1, 1, block_v]
    ):
        begin = cu_seqlens[seq.id].long()
        end = cu_seqlens[seq.id + 1].long()
        state = initial_state[seq.id, head.id, values.index, :].float()
        for chunk in hl.tile(end - begin, block_size=BT):
            feature = hl.arange(key_width)
            token_lane = hl.arange(BT)
            dt = dt_bias[head.id, feature].float()
            log_decay_scale = torch.exp2(a_log[head.id].float() * LOG2_E)
            token = begin + chunk.id * BT + token_lane
            valid = token < end
            row = token * heads + head.id
            q_raw = hl.load(
                q_rows,
                [row, feature],
                extra_mask=valid[:, None],
            ).float()
            k_raw = hl.load(
                k_rows,
                [row, feature],
                extra_mask=valid[:, None],
            ).float()
            gate_raw = hl.load(
                gate_rows,
                [row, feature],
                extra_mask=valid[:, None],
            ).float()
            increment = gate_scale_log2 * (
                torch.tanh(log_decay_scale * (gate_raw + dt[None, :]) * 0.5) * 0.5 + 0.5
            )
            increment = torch.where(valid[:, None], increment, 0.0)
            prefix = torch.cumsum(increment, dim=0)
            exp_gate = torch.exp2(prefix)
            gamma = torch.where((token_lane == BT - 1)[:, None], exp_gate, 0.0).sum(0)
            q_inv = torch.rsqrt(torch.clamp((q_raw * q_raw).sum(-1), min=1e-24))
            k_inv = torch.rsqrt(torch.clamp((k_raw * k_raw).sum(-1), min=1e-24))
            q_norm = q_raw * q_inv[:, None]
            k_norm = k_raw * k_inv[:, None]
            kd = (k_norm * exp_gate).to(torch.bfloat16)
            ki_fp32 = k_norm * torch.reciprocal(exp_gate)
            ki = ki_fp32.to(torch.bfloat16)
            qd = (q_norm * exp_gate).to(torch.bfloat16)
            beta = torch.sigmoid(hl.load(beta_rows, [row], extra_mask=valid).float())
            kk = hl.dot(kd, ki.T, out_dtype=torch.float32)
            qk = hl.dot(qd, ki.T, out_dtype=torch.float32)
            causal = token_lane[:, None] >= token_lane[None, :]
            strict = token_lane[:, None] > token_lane[None, :]
            inverse = _block_inverse(
                _bf16(torch.where(strict, kk * beta[:, None], 0.0))
            )
            qk = torch.where(causal, qk, 0.0).to(torch.bfloat16)
            kg = (ki_fp32 * gamma[None, :]).to(torch.bfloat16)
            projected = hl.dot(kd, state.T.to(torch.bfloat16), out_dtype=torch.float32)
            raw_v = hl.load(v_rows, [row, values], extra_mask=valid[:, None]).float()
            difference = _bf16(raw_v - _bf16(projected))
            rhs = (_bf16(beta)[:, None] * difference).to(torch.bfloat16)
            update_value = hl.dot(inverse, rhs, out_dtype=torch.float32).to(
                torch.bfloat16
            )
            result = hl.dot(qd, state.T.to(torch.bfloat16), out_dtype=torch.float32)
            result = hl.dot(qk, update_value, acc=result, out_dtype=torch.float32)
            hl.store(
                out_rows,
                [row, values],
                result * scale,
                extra_mask=valid[:, None],
            )
            state = hl.dot(
                update_value.T, kg, acc=state * gamma[None, :], out_dtype=torch.float32
            )
        final_state[seq.id, head.id, values.index, :] = state
    return output, final_state


def main() -> None:
    """The hillclimb harness supplies inputs, timing and numerical audits."""


if __name__ == "__main__":
    main()
