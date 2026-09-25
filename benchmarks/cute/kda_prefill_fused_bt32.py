from __future__ import annotations

import torch

import helion
import helion.language as hl

BT = 32
LOG2_E = 1.4426950408889634


def _bf16(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.bfloat16).float()


def _block_inverse32(lower: torch.Tensor) -> torch.Tensor:
    """CAKE BT32 inverse policy with explicit FP32 sums and FP16 operands."""

    row = hl.arange(BT)
    same_half = (row[:, None] // 8) == (row[None, :] // 8)
    lower_half = (
        ((row[:, None] // 16) == (row[None, :] // 16))
        & ((row[:, None] % 16) >= 8)
        & ((row[None, :] % 16) < 8)
    )
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
    # CAKE keeps these additive bases in FP32. Only MMA operands narrow.
    inverse = eye - diagonal
    inverse = inverse + hl.dot(
        inverse.to(torch.float16),
        diagonal2.to(torch.float16),
        out_dtype=torch.float32,
    )
    # Its coupling uses the degree-three polynomial snapshot, while the
    # diagonal output also receives the degree-four correction below.
    coupling_inverse = inverse.to(torch.float16)
    inverse = inverse + hl.dot(
        coupling_inverse,
        diagonal4.to(torch.float16),
        out_dtype=torch.float32,
    )
    first = -hl.dot(
        coupling_inverse,
        coupling.to(torch.float16),
        out_dtype=torch.float32,
    )
    lower_left = hl.dot(
        first.to(torch.float16),
        coupling_inverse,
        out_dtype=torch.float32,
    )
    inverse16 = torch.where(lower_half, lower_left, inverse).to(torch.bfloat16)
    outer_mask = (row[:, None] >= 16) & (row[None, :] < 16)
    outer = torch.where(outer_mask, lower, 0.0).to(torch.bfloat16)
    first32 = (-hl.dot(inverse16, outer, out_dtype=torch.float32)).to(torch.bfloat16)
    lower32 = hl.dot(first32, inverse16, out_dtype=torch.float32).to(torch.bfloat16)
    return torch.where(outer_mask, lower32, inverse16)


@helion.kernel(backend="cute", static_shapes=True, fast_math=True)
def kda_prefill_native_math_bt32(
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
    """Explicit centered BT32 policy, with FP32 authoritative state and residual."""
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
            center = gate_scale_log2 * 16.0
            center_scale = torch.exp2(hl.full([], center, dtype=torch.float32))
            exp_gate = torch.exp2(prefix - center)
            last_prefix = torch.where((token_lane == BT - 1)[:, None], prefix, 0.0).sum(
                0
            )
            gamma = torch.exp2(last_prefix)
            restore = torch.exp2(last_prefix - center)
            q_inv = torch.rsqrt((q_raw * q_raw).sum(-1) + 1e-6)
            k_inv = torch.rsqrt((k_raw * k_raw).sum(-1) + 1e-6)
            q_norm = q_raw * q_inv[:, None]
            k_norm = k_raw * k_inv[:, None]
            kd = (k_norm * exp_gate).to(torch.bfloat16)
            ki_fp32 = k_norm * torch.reciprocal(exp_gate)
            ki = ki_fp32.to(torch.bfloat16)
            qd = ((q_norm * scale) * exp_gate).to(torch.bfloat16)
            beta = torch.sigmoid(hl.load(beta_rows, [row], extra_mask=valid).float())
            kk = hl.dot(kd, ki.T, out_dtype=torch.float32)
            qk = hl.dot(qd, ki.T, out_dtype=torch.float32)
            causal = token_lane[:, None] >= token_lane[None, :]
            strict = token_lane[:, None] > token_lane[None, :]
            inverse = _block_inverse32(
                _bf16(torch.where(strict, kk * beta[:, None], 0.0))
            )
            qk = torch.where(causal, qk, 0.0).to(torch.bfloat16)
            kg = (ki.float() * restore[None, :]).to(torch.bfloat16)
            qr = (qd.float() * center_scale).to(torch.bfloat16)
            projected = (
                hl.dot(kd, state.T.to(torch.bfloat16), out_dtype=torch.float32)
                * center_scale
            )
            raw_v = hl.load(v_rows, [row, values], extra_mask=valid[:, None]).float()
            difference = raw_v - projected
            rhs = (beta[:, None] * difference).to(torch.bfloat16)
            update_value = hl.dot(inverse, rhs, out_dtype=torch.float32).to(
                torch.bfloat16
            )
            result = hl.dot(qr, state.T.to(torch.bfloat16), out_dtype=torch.float32)
            result = hl.dot(qk, update_value, acc=result, out_dtype=torch.float32)
            hl.store(
                out_rows,
                [row, values],
                result,
                extra_mask=valid[:, None],
            )
            state = hl.dot(
                update_value.T, kg, acc=state * gamma[None, :], out_dtype=torch.float32
            )
        final_state[seq.id, head.id, values.index, :] = state
    return output, final_state


def main() -> None:
    pass


if __name__ == "__main__":
    main()
