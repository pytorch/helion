# ruff: noqa: ANN001, ANN202
"""Matched six-launch PDL Helion baseline for KDA decode."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pretuned_kernels.megakernels._pdl import launch_dependent
from pretuned_kernels.megakernels._pdl import signal_dependents
from pretuned_kernels.megakernels._pdl import wait_and_launch_dependents
import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


INPUT_CONFIG = {
    "block_sizes": [32, 128],
    "num_warps": 1,
    "num_stages": 4,
    "pid_type": "flat",
}
GATE_CONFIG = {
    "block_sizes": [32, 128],
    "num_warps": 1,
    "num_stages": 4,
    "pid_type": "flat",
}
CONV_CONFIG = {
    "block_sizes": [128],
    "num_warps": 1,
    "num_stages": 2,
    "pid_type": "flat",
}
RECURRENCE_CONFIGS = {
    1: {
        "block_sizes": [8],
        "num_warps": 1,
        "num_stages": 2,
        "pid_type": "flat",
    },
    2: {
        "block_sizes": [16],
        "num_warps": 1,
        "num_stages": 2,
        "pid_type": "flat",
    },
}
NORM_CONFIG = {
    "num_warps": 1,
    "num_stages": 1,
    "pid_type": "flat",
}
OUTPUT_CONFIGS = {
    1: {
        "block_sizes": [8, 128],
        "num_warps": 1,
        "num_stages": 4,
        "pid_type": "flat",
    },
    2: {
        "block_sizes": [32, 64],
        "num_warps": 1,
        "num_stages": 4,
        "pid_type": "flat",
    },
}


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def input_projection(hidden_states, input_weight):
    """Production-boundary BF16 input projection."""
    batch, hidden = hidden_states.shape
    projection_width, input_hidden = input_weight.shape
    assert input_hidden == hidden
    projected = torch.empty(
        (batch, projection_width),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    for tile_batch, tile_out in hl.tile(
        [batch, projection_width], block_size=[1, None]
    ):
        signal_dependents()
        accumulator = hl.zeros([tile_batch, tile_out], dtype=torch.float32)
        for tile_hidden in hl.tile(hidden):
            accumulator = torch.addmm(
                accumulator,
                hidden_states[tile_batch, tile_hidden],
                input_weight[tile_out, tile_hidden].T,
            )
        projected[tile_batch, tile_out] = accumulator.to(projected.dtype)
    return projected


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def gate_projection(projected, fg_weight, qkv_width, heads, head_dim):
    """Project the two low-rank KDA gate inputs."""
    batch, projection_width = projected.shape
    kinds, gate_width, gate_k = fg_weight.shape
    assert kinds == 2
    assert gate_width == heads * head_dim
    assert gate_k == head_dim
    assert projection_width == qkv_width + heads + 2 * head_dim
    gate_inputs = projected[:, qkv_width + heads :].view(batch, 2, head_dim)
    gates = torch.empty(
        (2, batch, gate_width), dtype=projected.dtype, device=projected.device
    )
    for tile_kind, tile_batch, tile_out in hl.tile(
        [2, batch, gate_width], block_size=[1, 1, None]
    ):
        wait_and_launch_dependents()
        accumulator = hl.zeros([tile_batch, tile_out], dtype=torch.float32)
        for tile_key in hl.tile(head_dim):
            accumulator = torch.addmm(
                accumulator,
                gate_inputs[tile_batch, tile_kind.id, tile_key],
                fg_weight[tile_kind.id, tile_out, tile_key].T,
            )
        gates[tile_kind.id, tile_batch, tile_out] = accumulator.to(gates.dtype)
    return gates


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def causal_conv(projected_qkv, conv_state, conv_weight, state_indices):
    """Width-four causal convolution, SiLU, and cache update."""
    batch, qkv_width = projected_qkv.shape
    slots, history, state_width = conv_state.shape
    weight_width, taps = conv_weight.shape
    assert state_width == qkv_width
    assert weight_width == qkv_width
    assert history == 3
    assert taps == 4
    assert state_indices.shape == (batch,)
    convolved = torch.empty_like(projected_qkv)
    for tile_batch, tile_channel in hl.tile([batch, qkv_width], block_size=[1, None]):
        wait_and_launch_dependents()
        state_index = state_indices[tile_batch.id].long()
        current = projected_qkv[tile_batch.id, tile_channel].float()
        value = hl.zeros([tile_channel], dtype=torch.float32)
        if state_index >= 0:
            state_0 = conv_state[state_index, 0, tile_channel].float()
            state_1 = conv_state[state_index, 1, tile_channel].float()
            state_2 = conv_state[state_index, 2, tile_channel].float()
            value = (
                state_0 * conv_weight[tile_channel, 0].float()
                + state_1 * conv_weight[tile_channel, 1].float()
                + state_2 * conv_weight[tile_channel, 2].float()
                + current * conv_weight[tile_channel, 3].float()
            )
            value = value * torch.sigmoid(value)
            conv_state[state_index, 0, tile_channel] = state_1.to(conv_state.dtype)
            conv_state[state_index, 1, tile_channel] = state_2.to(conv_state.dtype)
            conv_state[state_index, 2, tile_channel] = current.to(conv_state.dtype)
        convolved[tile_batch.id, tile_channel] = value.to(convolved.dtype)
    return convolved


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def recurrent_decode(
    convolved_qkv,
    forget_gate,
    beta,
    a_log,
    dt_bias,
    recurrent_state,
    state_indices,
    scale,
    heads,
    head_dim,
):
    """Packed one-token KDA recurrence with in-kernel Q/K L2 normalization."""
    batch, qkv_width = convolved_qkv.shape
    slots, state_heads, value_dim, key_dim = recurrent_state.shape
    assert state_heads == heads
    assert value_dim == key_dim == head_dim
    assert qkv_width == 3 * heads * head_dim
    assert forget_gate.shape == (2, batch, heads * head_dim)
    assert beta.shape == (batch, heads)
    assert a_log.shape == (heads,)
    assert dt_bias.shape == (heads * head_dim,)
    assert state_indices.shape == (batch,)
    qkv = convolved_qkv.view(batch, 3, heads, head_dim)
    gates = forget_gate.view(2, batch, heads, head_dim)
    output = torch.empty(
        (batch, heads, value_dim),
        dtype=convolved_qkv.dtype,
        device=convolved_qkv.device,
    )
    for tile_batch, tile_head, tile_value in hl.tile(
        [batch, heads, value_dim], block_size=[1, 1, None]
    ):
        wait_and_launch_dependents()
        state_index = state_indices[tile_batch.id].long()
        key_offsets = hl.arange(key_dim)
        query = qkv[tile_batch.id, 0, tile_head.id, key_offsets].float()
        key = qkv[tile_batch.id, 1, tile_head.id, key_offsets].float()
        value = qkv[tile_batch.id, 2, tile_head.id, tile_value].float()
        query = query * torch.rsqrt(torch.sum(query * query) + 1e-6) * scale
        key = key * torch.rsqrt(torch.sum(key * key) + 1e-6)
        activated_gate = (
            gates[0, tile_batch.id, tile_head.id, key_offsets].float()
            + dt_bias[tile_head.id * head_dim + key_offsets].float()
        )
        gate_exp = torch.exp(activated_gate)
        softplus = torch.where(
            activated_gate <= 20.0,
            torch.log(1.0 + gate_exp),
            activated_gate,
        )
        decay = torch.exp(-torch.exp(a_log[tile_head.id].float()) * softplus)
        beta_value = torch.sigmoid(beta[tile_batch.id, tile_head.id].float())
        result = hl.zeros([tile_value], dtype=torch.float32)
        if state_index >= 0:
            state = recurrent_state[
                state_index, tile_head.id, tile_value.index, key_offsets
            ].float()
            state = state * decay[None, :]
            residual = value - torch.sum(state * key[None, :], dim=-1)
            state = state + (residual * beta_value)[:, None] * key[None, :]
            result = torch.sum(state * query[None, :], dim=-1)
            recurrent_state[
                state_index, tile_head.id, tile_value.index, key_offsets
            ] = state.to(recurrent_state.dtype)
        output[tile_batch.id, tile_head.id, tile_value] = result.to(output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def gated_rms_norm(core_output, norm_gate, norm_weight, eps):
    """Per-head gated RMS normalization at the production boundary."""
    batch, heads, value_dim = core_output.shape
    assert norm_gate.shape == (2, batch, heads * value_dim)
    assert norm_weight.shape == (value_dim,)
    gates = norm_gate.view(2, batch, heads, value_dim)
    normalized = torch.empty_like(core_output)
    for tile_batch_head, tile_value in hl.tile(
        [batch * heads, value_dim], block_size=[1, value_dim]
    ):
        wait_and_launch_dependents()
        batch_index = tile_batch_head.id // heads
        head_index = tile_batch_head.id % heads
        values = core_output[batch_index, head_index, tile_value].float()
        inv_rms = torch.rsqrt(torch.mean(values * values) + eps)
        gate = torch.sigmoid(gates[1, batch_index, head_index, tile_value].float())
        normalized[batch_index, head_index, tile_value] = (
            values * inv_rms * norm_weight[tile_value].float() * gate
        ).to(normalized.dtype)
    return normalized


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def output_projection(normalized, output_weight):
    """Tensor-parallel-local BF16 output projection."""
    batch, heads, value_dim = normalized.shape
    output_hidden, output_k = output_weight.shape
    assert output_k == heads * value_dim
    normalized_flat = normalized.view(batch, output_k)
    output = torch.empty(
        (batch, output_hidden), dtype=normalized.dtype, device=normalized.device
    )
    for tile_batch, tile_out in hl.tile([batch, output_hidden], block_size=[1, None]):
        wait_and_launch_dependents()
        accumulator = hl.zeros([tile_batch, tile_out], dtype=torch.float32)
        for tile_input in hl.tile(output_k):
            accumulator = torch.addmm(
                accumulator,
                normalized_flat[tile_batch, tile_input],
                output_weight[tile_out, tile_input].T,
            )
        output[tile_batch, tile_out] = accumulator.to(output.dtype)
    return output


def _compile(kernel, args, config_values):
    bound = kernel.bind(args)
    values = dict(bound.config_spec.default_config())
    values.update(config_values)
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return bound.compile_config(config)


def build(
    tensors: dict[str, torch.Tensor], *, scale: float, eps: float
) -> tuple[Callable[[], torch.Tensor], tuple[object, ...], torch.Tensor]:
    """Compile and materialize the matched six-launch PDL decode graph."""
    batch = tensors["hidden_states"].shape[0]
    heads = tensors["recurrent_state"].shape[1]
    head_dim = tensors["recurrent_state"].shape[-1]
    qkv_width = 3 * heads * head_dim

    input_args = (tensors["hidden_states"], tensors["input_weight"])
    input_kernel = _compile(input_projection, input_args, INPUT_CONFIG)
    projected = input_kernel(*input_args)

    gate_args = (projected, tensors["fg_weight"], qkv_width, heads, head_dim)
    gate_kernel = _compile(gate_projection, gate_args, GATE_CONFIG)
    gates = launch_dependent(gate_kernel, *gate_args)

    conv_args = (
        projected[:, :qkv_width],
        tensors["conv_state"],
        tensors["conv_weight"],
        tensors["state_indices"],
    )
    conv_kernel = _compile(causal_conv, conv_args, CONV_CONFIG)
    convolved = launch_dependent(conv_kernel, *conv_args)

    recurrence_args = (
        convolved,
        gates,
        projected[:, qkv_width : qkv_width + heads],
        tensors["a_log"],
        tensors["dt_bias"],
        tensors["recurrent_state"],
        tensors["state_indices"],
        scale,
        heads,
        head_dim,
    )
    recurrence_kernel = _compile(
        recurrent_decode, recurrence_args, RECURRENCE_CONFIGS[batch]
    )
    core_output = launch_dependent(recurrence_kernel, *recurrence_args)

    norm_args = (core_output, gates, tensors["norm_weight"], eps)
    norm_kernel = _compile(gated_rms_norm, norm_args, NORM_CONFIG)
    normalized = launch_dependent(norm_kernel, *norm_args)

    output_args = (normalized, tensors["output_weight"])
    output_kernel = _compile(output_projection, output_args, OUTPUT_CONFIGS[batch])
    output = launch_dependent(output_kernel, *output_args)

    kernels = (
        input_kernel,
        gate_kernel,
        conv_kernel,
        recurrence_kernel,
        norm_kernel,
        output_kernel,
    )

    def launch() -> torch.Tensor:
        local_projected = input_kernel(*input_args)
        local_gates = launch_dependent(
            gate_kernel,
            local_projected,
            tensors["fg_weight"],
            qkv_width,
            heads,
            head_dim,
        )
        local_convolved = launch_dependent(
            conv_kernel,
            local_projected[:, :qkv_width],
            tensors["conv_state"],
            tensors["conv_weight"],
            tensors["state_indices"],
        )
        local_core = launch_dependent(
            recurrence_kernel,
            local_convolved,
            local_gates,
            local_projected[:, qkv_width : qkv_width + heads],
            tensors["a_log"],
            tensors["dt_bias"],
            tensors["recurrent_state"],
            tensors["state_indices"],
            scale,
            heads,
            head_dim,
        )
        local_normalized = launch_dependent(
            norm_kernel,
            local_core,
            local_gates,
            tensors["norm_weight"],
            eps,
        )
        return launch_dependent(
            output_kernel, local_normalized, tensors["output_weight"]
        )

    return launch, kernels, output
