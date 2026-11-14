"""Fixed-token rank-1 kernels and CPU target shared by codegen tests."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch

import helion
from helion._testing import cpu_render_target
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Generator

_LOG2_E = 1.4426950408889634


def _fixed_rank1_body(
    first_vector: torch.Tensor,
    second_vector: torch.Tensor,
    values: torch.Tensor,
    gate_source: torch.Tensor,
    update_weight: torch.Tensor,
    log_decay_rate: torch.Tensor,
    gate_bias: torch.Tensor,
    checkpoint_pool: torch.Tensor,
    checkpoint_ids: torch.Tensor,
    accepted_counts: torch.Tensor,
    result: torch.Tensor,
    projection_scale: float,
    decay_floor: float,
    token_count: hl.constexpr,
    epsilon: hl.constexpr,
    subtract_projection: hl.constexpr,
    beta_is_logit: hl.constexpr,
    reverse_checkpoints: hl.constexpr,
    shrink_sequence_axis: hl.constexpr,
) -> torch.Tensor:
    sequences = hl.specialize(checkpoint_ids.size(0))
    heads = hl.specialize(first_vector.size(1))
    width = hl.specialize(first_vector.size(2))
    value_width = hl.specialize(values.size(2))
    hl.specialize(
        (
            first_vector.stride(),
            second_vector.stride(),
            values.stride(),
            gate_source.stride(),
            update_weight.stride(),
            log_decay_rate.stride(),
            gate_bias.stride(),
            checkpoint_pool.stride(),
            checkpoint_ids.stride(),
            accepted_counts.stride(),
            result.stride(),
        )
    )
    task_sequences = sequences - 1 if shrink_sequence_axis else sequences
    for sequence_tile, head_tile, value_tile in hl.tile(
        [task_sequences, heads, value_width], block_size=[1, 1, None]
    ):
        sequence = sequence_tile.id
        head = head_tile.id
        offsets = hl.arange(width)
        accepted_index = torch.clamp(
            accepted_counts[sequence].long() - 1,
            min=0,
            max=token_count - 1,  # pyrefly: ignore[unsupported-operation]
        )
        initial_slot = checkpoint_ids[sequence, accepted_index].long()
        recurrent = checkpoint_pool[
            initial_slot, head, value_tile.index, offsets
        ].float()

        for token_index in hl.static_range(token_count):  # pyrefly: ignore[bad-argument-type]
            token = sequence * token_count + token_index  # pyrefly: ignore[unsupported-operation]
            query = first_vector[token, head, offsets].float()
            key = second_vector[token, head, offsets].float()
            query = query * torch.rsqrt(
                (query * query).sum() + epsilon  # pyrefly: ignore[unsupported-operation]
            )
            key = key * torch.rsqrt(
                (key * key).sum() + epsilon  # pyrefly: ignore[unsupported-operation]
            )

            gate = gate_source[token, head, offsets].float()
            gate = gate + gate_bias[head * width + offsets].float()
            decay_parameter = torch.exp2(log_decay_rate[head].float() * _LOG2_E)
            decay = torch.exp2(
                decay_floor * torch.sigmoid(decay_parameter * gate) * _LOG2_E
            )
            recurrent = recurrent * decay[None, :]

            beta_value = update_weight[token, head].float()
            if beta_is_logit:
                beta_value = torch.sigmoid(beta_value)
            value = values[token, head, value_tile].float()
            prediction = (recurrent * key[None, :]).sum(-1)
            if subtract_projection:
                residual = value - prediction
            else:
                residual = value + prediction
            recurrent = recurrent + (beta_value * residual)[:, None] * key[None, :]

            projected = (recurrent * query[None, :]).sum(-1) * projection_scale
            result[token, head, value_tile] = projected.to(result.dtype)
            if reverse_checkpoints:
                checkpoint_index = token_count - 1 - token_index  # pyrefly: ignore[unsupported-operation]
            else:
                checkpoint_index = token_index
            checkpoint_slot = checkpoint_ids[sequence, checkpoint_index].long()
            checkpoint_pool[checkpoint_slot, head, value_tile.index, offsets] = (
                recurrent
            )
    return result


fixed_rank1 = helion.kernel(
    _fixed_rank1_body,
    backend="cute",
    static_shapes=False,
    fast_math=True,
)
precise_fixed_rank1 = helion.kernel(
    _fixed_rank1_body,
    backend="cute",
    static_shapes=False,
    fast_math=False,
)


def fixed_rank1_inputs(
    token_count: int,
    *,
    width: int = 128,
    state_dtype: torch.dtype = torch.bfloat16,
) -> tuple[object, ...]:
    sequences = 2
    heads = 4
    vectors = sequences * token_count
    vector_shape = (vectors, heads, width)
    return (
        torch.empty(vector_shape, dtype=torch.bfloat16),
        torch.empty(vector_shape, dtype=torch.bfloat16),
        torch.empty(vector_shape, dtype=torch.bfloat16),
        torch.empty(vector_shape, dtype=torch.bfloat16),
        torch.empty(vector_shape[:2], dtype=torch.bfloat16),
        torch.empty((heads,), dtype=torch.float32),
        torch.empty((heads * width,), dtype=torch.float32),
        torch.empty((vectors + 2, heads, width, width), dtype=state_dtype),
        torch.empty((sequences, token_count), dtype=torch.int32),
        torch.empty((sequences,), dtype=torch.int32),
        torch.empty(vector_shape, dtype=torch.bfloat16),
        width**-0.5,
        -5.0,
    )


@contextmanager
def fixed_rank1_cpu_target() -> Generator[None, None, None]:
    with (
        cpu_render_target(capability=(10, 3), fake_device_sm_count=148),
        patch("helion.language.loops._supports_warp_specialize", return_value=True),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
        patch("helion._compat._min_dot_size", return_value=(16, 16, 16)),
    ):
        yield
