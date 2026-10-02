# ruff: noqa: ANN001, ANN202
"""Source-matched nine-launch PDL baseline for KDA decode.

Each kernel below is one top-level root from ``kda_decode``. Tensor layouts,
rounding points, and algebra match the megakernel. Root tiles are independently
pretuned; the structural difference is persistent cross-root scheduling versus
separate PDL grids.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pretuned_kernels.megakernels._pdl import launch_dependent
from pretuned_kernels.megakernels._pdl import signal_dependents
from pretuned_kernels.megakernels._pdl import wait_and_launch_dependents
from pretuned_kernels.megakernels.kda_decode._helion_aot_kda_decode_cuda_sm100 import (
    CONFIGS as MEGAKERNEL_CONFIGS,
)
import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


_BLOCK_INDICES = {
    "gate_input": (0, 1, 13),
    "qkv": (2, 3, 14),
    "beta": (4, 15),
    "decay": (5, 6),
    "conv": (7,),
    "norm_gate": (8, 9),
    "recurrence": (10,),
    "rms": (8,),
    "output": (11, 12, 16),
}
_LOOP_ORDER_INDICES = {
    "gate_input": 0,
    "qkv": 1,
    "beta": 2,
    "decay": 3,
    "conv": 4,
    "norm_gate": 5,
    "recurrence": 6,
    # The RMS grid has only one non-unit grid dimension.
    "output": 7,
}
_ROOT_BLOCK_OVERRIDES: dict[int, dict[str, list[int]]] = {
    1: {},
    2: {
        "qkv": [2, 16, 256],
        "beta": [2, 256],
        "recurrence": [8],
    },
    4: {
        "qkv": [4, 16, 256],
        "beta": [4, 256],
        "recurrence": [8],
    },
    8: {
        "qkv": [8, 16, 256],
        "beta": [8, 256],
        "output": [8, 32, 256],
    },
    16: {
        "qkv": [16, 16, 256],
        "beta": [16, 256],
        "output": [16, 32, 256],
    },
}


def _root_config(batch: int, heads: int, root: str) -> dict[str, object]:
    """Build one standalone-root config from shared defaults and PDL tuning."""
    megakernel = MEGAKERNEL_CONFIGS[batch, heads]
    block_sizes = megakernel["block_sizes"]
    default_blocks = [block_sizes[index] for index in _BLOCK_INDICES[root]]
    config: dict[str, object] = {
        "block_sizes": _ROOT_BLOCK_OVERRIDES[batch].get(root, default_blocks),
        "num_warps": megakernel["num_warps"],
        "num_stages": megakernel["num_stages"],
        "pid_type": "flat",
        "indexing": megakernel["indexing"],
        "load_eviction_policies": megakernel["load_eviction_policies"],
    }
    loop_orders = megakernel.get("loop_orders")
    if loop_orders is not None and root in _LOOP_ORDER_INDICES:
        config["loop_orders"] = [loop_orders[_LOOP_ORDER_INDICES[root]]]
    return config


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def gate_input_projection(
    hidden_states,
    input_weight,
    projection_width_static: hl.constexpr,
    qkv_width_static: hl.constexpr,
    heads_static: hl.constexpr,
    key_dim_static: hl.constexpr,
):
    """Megakernel root 0: project the two low-rank gate inputs."""
    batch, hidden = hidden_states.shape
    projection_width, input_hidden = input_weight.shape
    assert input_hidden == hidden
    assert projection_width == projection_width_static
    assert projection_width == qkv_width_static + heads_static + 2 * key_dim_static

    gate_input_batch_block = hl.register_block_size(1, 16)
    gate_input_block = hl.register_block_size(4, 64)
    gate_input_k_block = hl.register_block_size(32, 512)
    gate_input_weight = input_weight[
        qkv_width_static + heads_static : projection_width_static
    ].view(2, key_dim_static, hidden)
    projected_gate_inputs = torch.empty(
        (batch, 2, key_dim_static),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    for tile_batch, tile_kind, tile_dim in hl.tile(
        [batch, 2, key_dim_static],
        block_size=[gate_input_batch_block, 1, gate_input_block],
    ):
        signal_dependents()
        kind = tile_kind.id
        accumulator = hl.zeros([tile_batch, tile_dim], dtype=torch.float32)
        for tile_hidden in hl.tile(hidden, block_size=gate_input_k_block):
            accumulator = torch.addmm(
                accumulator,
                hidden_states[tile_batch, tile_hidden],
                gate_input_weight[kind, tile_dim, tile_hidden].T,
            )
        projected_gate_inputs[tile_batch, kind, tile_dim] = accumulator.to(
            projected_gate_inputs.dtype
        )
    return projected_gate_inputs


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def qkv_projection(
    hidden_states,
    input_weight,
    qkv_width_static: hl.constexpr,
    heads_static: hl.constexpr,
    key_dim_static: hl.constexpr,
):
    """Megakernel root 1: project packed Q, K, and V."""
    batch, hidden = hidden_states.shape
    projection_width, input_hidden = input_weight.shape
    assert input_hidden == hidden
    assert qkv_width_static == 3 * heads_static * key_dim_static
    assert projection_width >= qkv_width_static

    projection_batch_block = hl.register_block_size(1, 16)
    projection_block = hl.register_block_size(4, 64)
    qkv_input_k_block = hl.register_block_size(32, 512)
    qkv_input_weight = input_weight[:qkv_width_static].view(
        3, heads_static, key_dim_static, hidden
    )
    projected_qkv = torch.empty(
        (batch, heads_static, 3, key_dim_static),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    for tile_batch, tile_head, tile_kind, tile_dim in hl.tile(
        [batch, heads_static, 3, key_dim_static],
        block_size=[projection_batch_block, 1, 1, projection_block],
    ):
        wait_and_launch_dependents()
        kind = tile_kind.id
        accumulator = hl.zeros([tile_batch, tile_dim], dtype=torch.float32)
        for tile_hidden in hl.tile(hidden, block_size=qkv_input_k_block):
            accumulator = torch.addmm(
                accumulator,
                hidden_states[tile_batch, tile_hidden],
                qkv_input_weight[
                    kind,
                    tile_head.id,
                    tile_dim,
                    tile_hidden,
                ].T,
            )
        projected_qkv[tile_batch, tile_head.id, kind, tile_dim] = accumulator.to(
            projected_qkv.dtype
        )
    return projected_qkv


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def beta_projection(
    hidden_states,
    input_weight,
    qkv_width_static: hl.constexpr,
    heads_static: hl.constexpr,
):
    """Megakernel root 2: project and sigmoid recurrence beta."""
    batch, hidden = hidden_states.shape
    projection_width, input_hidden = input_weight.shape
    assert input_hidden == hidden
    assert projection_width >= qkv_width_static + heads_static

    beta_batch_block = hl.register_block_size(1, 16)
    beta_input_k_block = hl.register_block_size(32, 512)
    beta_head_block = 1 << (heads_static - 1).bit_length()
    beta_input_weight = input_weight[qkv_width_static : qkv_width_static + heads_static]
    recurrence_beta = torch.empty(
        (batch * heads_static,), dtype=torch.float32, device=hidden_states.device
    )
    recurrence_beta_batched = recurrence_beta.view(batch, heads_static)

    for tile_batch, tile_head in hl.tile(
        [batch, heads_static],
        block_size=[beta_batch_block, beta_head_block],
    ):
        wait_and_launch_dependents()
        beta_accumulator = hl.zeros([tile_batch, tile_head], dtype=torch.float32)
        for beta_tile_hidden in hl.tile(hidden, block_size=beta_input_k_block):
            beta_accumulator = torch.addmm(
                beta_accumulator,
                hidden_states[tile_batch, beta_tile_hidden],
                beta_input_weight[tile_head, beta_tile_hidden].T,
            )
        recurrence_beta_batched[tile_batch, tile_head] = torch.sigmoid(
            beta_accumulator.to(hidden_states.dtype).float()
        )
    return recurrence_beta


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def prepare_decay(
    projected_gate_inputs,
    fg_weight,
    a_log,
    dt_bias,
    use_batched_gates: hl.constexpr,
    heads_static: hl.constexpr,
    key_dim_static: hl.constexpr,
):
    """Megakernel root 3: form the recurrent decay gate."""
    batch, kinds, key_dim = projected_gate_inputs.shape
    assert kinds == 2
    assert key_dim == key_dim_static
    assert fg_weight.shape == (2, heads_static * key_dim_static, key_dim_static)
    assert a_log.shape == (heads_static,)
    assert dt_bias.shape == (heads_static * key_dim_static,)

    forget_gate_batch_block = hl.register_block_size(1, 16)
    forget_gate_block = hl.register_block_size(4, 128)
    prepared_decay = torch.empty(
        (batch * heads_static, key_dim_static),
        dtype=torch.float32,
        device=projected_gate_inputs.device,
    )
    prepared_decay_batched = prepared_decay.view(batch, heads_static, key_dim_static)

    for tile_head_batch, tile_gate_dim in hl.tile(
        [heads_static * batch, key_dim_static],
        block_size=[forget_gate_batch_block, forget_gate_block],
    ):
        wait_and_launch_dependents()
        gate_dim = tile_gate_dim.index
        gate_key = hl.arange(key_dim_static)
        if use_batched_gates:
            gate_input = projected_gate_inputs[
                tile_head_batch.index
                - (tile_head_batch.id * forget_gate_batch_block // batch) * batch,
                0,
                gate_key,
            ]
            gate_weights = fg_weight[
                0,
                (tile_head_batch.id * forget_gate_batch_block // batch) * key_dim_static
                + gate_dim[:, None],
                gate_key[None, :],
            ]
            gate_accumulator = torch.addmm(
                hl.zeros([tile_head_batch, tile_gate_dim], dtype=torch.float32),
                gate_input,
                gate_weights.T,
            )
            rounded_gate = gate_accumulator.to(projected_gate_inputs.dtype)
            activated_gate = (
                rounded_gate.float()
                + dt_bias[
                    (tile_head_batch.id * forget_gate_batch_block // batch)
                    * key_dim_static
                    + gate_dim
                ].float()[None, :]
            )
            gate_exp = torch.exp(activated_gate)
            softplus = torch.where(
                activated_gate <= 20.0,
                torch.log(1.0 + gate_exp),
                activated_gate,
            )
            prepared_decay_batched[
                tile_head_batch.index
                - (tile_head_batch.id * forget_gate_batch_block // batch) * batch,
                tile_head_batch.id * forget_gate_batch_block // batch,
                tile_gate_dim,
            ] = torch.exp(
                -torch.exp(
                    a_log[tile_head_batch.id * forget_gate_batch_block // batch].float()
                )
                * softplus
            )
        else:
            batch_head_index = tile_head_batch.id
            gate_input = projected_gate_inputs[
                tile_head_batch.id // heads_static, 0, gate_key
            ].float()
            gate_weights = fg_weight[
                0,
                (tile_head_batch.id % heads_static) * key_dim_static
                + gate_dim[:, None],
                gate_key[None, :],
            ].float()
            gate_accumulator = torch.sum(gate_weights * gate_input[None, :], dim=-1)
            rounded_gate = gate_accumulator.to(projected_gate_inputs.dtype)
            activated_gate = (
                rounded_gate.float()
                + dt_bias[
                    (tile_head_batch.id % heads_static) * key_dim_static + gate_dim
                ].float()
            )
            gate_exp = torch.exp(activated_gate)
            softplus = torch.where(
                activated_gate <= 20.0,
                torch.log(1.0 + gate_exp),
                activated_gate,
            )
            prepared_decay[batch_head_index, gate_dim] = torch.exp(
                -torch.exp(a_log[tile_head_batch.id % heads_static].float()) * softplus
            )
    return prepared_decay


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def prepare_qkv(
    projected_qkv,
    conv_state,
    conv_weight,
    state_indices,
    scale: hl.constexpr,
    heads_static: hl.constexpr,
    key_dim_static: hl.constexpr,
):
    """Megakernel root 4: causal convolution and Q/K normalization."""
    batch, heads, kinds, key_dim = projected_qkv.shape
    slots, history, qkv_width = conv_state.shape
    weight_width, taps = conv_weight.shape
    assert heads == heads_static
    assert kinds == 3
    assert key_dim == key_dim_static
    assert qkv_width == weight_width == 3 * heads_static * key_dim_static
    assert history == 3
    assert taps == 4
    assert state_indices.shape == (batch,)

    conv_block = hl.register_block_size(4, 128)
    batch_heads = batch * heads_static
    prepared_qkv = torch.empty(
        (batch_heads, 3, key_dim_static),
        dtype=torch.float32,
        device=projected_qkv.device,
    )

    for tile_batch_head, tile_kind, tile_dim in hl.tile(
        [batch_heads, 3, key_dim_static], block_size=[1, 1, conv_block]
    ):
        wait_and_launch_dependents()
        batch_head_index = tile_batch_head.id
        kind = tile_kind.id
        state_index = state_indices[tile_batch_head.id // heads_static].long()
        channel = (
            kind * heads_static * key_dim_static
            + (tile_batch_head.id % heads_static) * key_dim_static
            + tile_dim.index
        )
        x = projected_qkv[
            tile_batch_head.id // heads_static,
            tile_batch_head.id % heads_static,
            kind,
            tile_dim,
        ].float()
        value = hl.zeros([tile_dim], dtype=torch.float32)
        if state_index >= 0:
            state_0 = conv_state[state_index, 0, channel].float()
            state_1 = conv_state[state_index, 1, channel].float()
            state_2 = conv_state[state_index, 2, channel].float()
            value = (
                state_0 * conv_weight[channel, 0].float()
                + state_1 * conv_weight[channel, 1].float()
                + state_2 * conv_weight[channel, 2].float()
                + x * conv_weight[channel, 3].float()
            )
            value = value * torch.sigmoid(value)
            conv_state[state_index, 0, channel] = state_1.to(conv_state.dtype)
            conv_state[state_index, 1, channel] = state_2.to(conv_state.dtype)
            conv_state[state_index, 2, channel] = x.to(conv_state.dtype)
        value = value.to(projected_qkv.dtype).float()
        if kind < 2:
            value = value * torch.rsqrt(torch.sum(value * value) + 1e-6)
            if kind == 0:
                value = value * scale
        prepared_qkv[batch_head_index, kind, tile_dim] = value
    return prepared_qkv


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def prepare_norm_gate(
    projected_gate_inputs,
    fg_weight,
    norm_weight,
    use_batched_gates: hl.constexpr,
    heads_static: hl.constexpr,
    key_dim_static: hl.constexpr,
):
    """Megakernel root 5: form the sigmoid normalization gate."""
    batch, kinds, key_dim = projected_gate_inputs.shape
    assert kinds == 2
    assert key_dim == key_dim_static
    assert fg_weight.shape == (2, heads_static * key_dim_static, key_dim_static)
    assert norm_weight.shape == (key_dim_static,)

    norm_gate_batch_block = hl.register_block_size(1, 16)
    norm_gate_block = hl.register_block_size(4, 128)
    prepared_norm_gate = torch.empty(
        (batch * heads_static, key_dim_static),
        dtype=torch.float32,
        device=projected_gate_inputs.device,
    )
    prepared_norm_gate_batched = prepared_norm_gate.view(
        batch, heads_static, key_dim_static
    )

    for tile_head_batch, tile_gate_dim in hl.tile(
        [heads_static * batch, key_dim_static],
        block_size=[norm_gate_batch_block, norm_gate_block],
    ):
        wait_and_launch_dependents()
        gate_dim = tile_gate_dim.index
        gate_key = hl.arange(key_dim_static)
        if use_batched_gates:
            gate_input = projected_gate_inputs[
                tile_head_batch.index
                - (tile_head_batch.id * norm_gate_batch_block // batch) * batch,
                1,
                gate_key,
            ]
            gate_weights = fg_weight[
                1,
                (tile_head_batch.id * norm_gate_batch_block // batch) * key_dim_static
                + gate_dim[:, None],
                gate_key[None, :],
            ]
            gate_accumulator = torch.addmm(
                hl.zeros([tile_head_batch, tile_gate_dim], dtype=torch.float32),
                gate_input,
                gate_weights.T,
            )
            rounded_gate = gate_accumulator.to(projected_gate_inputs.dtype)
            prepared_norm_gate_batched[
                tile_head_batch.index
                - (tile_head_batch.id * norm_gate_batch_block // batch) * batch,
                tile_head_batch.id * norm_gate_batch_block // batch,
                tile_gate_dim,
            ] = norm_weight[gate_dim].float()[None, :] * torch.sigmoid(
                rounded_gate.float()
            )
        else:
            batch_head_index = tile_head_batch.id
            gate_input = projected_gate_inputs[
                tile_head_batch.id // heads_static, 1, gate_key
            ].float()
            gate_weights = fg_weight[
                1,
                (tile_head_batch.id % heads_static) * key_dim_static
                + gate_dim[:, None],
                gate_key[None, :],
            ].float()
            gate_accumulator = torch.sum(gate_weights * gate_input[None, :], dim=-1)
            rounded_gate = gate_accumulator.to(projected_gate_inputs.dtype)
            prepared_norm_gate[batch_head_index, gate_dim] = norm_weight[
                gate_dim
            ].float() * torch.sigmoid(rounded_gate.float())
    return prepared_norm_gate


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def recurrent_decode(
    prepared_qkv,
    prepared_decay,
    recurrence_beta,
    recurrent_state,
    state_indices,
    heads_static: hl.constexpr,
    key_dim_static: hl.constexpr,
    value_dim_static: hl.constexpr,
):
    """Megakernel root 6: apply the one-token KDA recurrence."""
    batch_heads, kinds, key_dim = prepared_qkv.shape
    slots, heads, value_dim, state_key_dim = recurrent_state.shape
    batch = state_indices.shape[0]
    assert batch_heads == batch * heads_static
    assert kinds == 3
    assert heads == heads_static
    assert key_dim == state_key_dim == key_dim_static
    assert value_dim == value_dim_static
    assert prepared_decay.shape == (batch_heads, key_dim_static)
    assert recurrence_beta.shape == (batch_heads,)

    recurrent_block = hl.register_block_size(4, value_dim_static)
    core_output = torch.empty(
        (batch_heads, value_dim_static),
        dtype=torch.bfloat16,
        device=prepared_qkv.device,
    )

    for tile_batch_head, tile_value in hl.tile(
        [batch_heads, value_dim_static], block_size=[1, recurrent_block]
    ):
        wait_and_launch_dependents()
        batch_head_index = tile_batch_head.id
        state_index = state_indices[tile_batch_head.id // heads_static].long()
        result = hl.zeros([tile_value], dtype=torch.float32)
        for tile_key in hl.tile(key_dim_static, block_size=key_dim_static):
            key_offsets = tile_key.index
            decay = prepared_decay[batch_head_index, key_offsets]
            beta_value = recurrence_beta[batch_head_index]
            key = prepared_qkv[batch_head_index, 1, key_offsets]
            value = prepared_qkv[batch_head_index, 2, tile_value]
            query = prepared_qkv[batch_head_index, 0, key_offsets]
            if state_index >= 0:
                state = recurrent_state[
                    state_index,
                    tile_batch_head.id % heads_static,
                    tile_value.index,
                    key_offsets,
                ].float()
                state = state * decay[None, :]
                value_residual = value - torch.sum(state * key[None, :], dim=-1)
                state = state + (value_residual * beta_value)[:, None] * key[None, :]
                result = torch.sum(state * query[None, :], dim=-1)
                recurrent_state[
                    state_index,
                    tile_batch_head.id % heads_static,
                    tile_value.index,
                    key_offsets,
                ] = state.to(recurrent_state.dtype)
        core_output[batch_head_index, tile_value] = result.to(core_output.dtype)
    return core_output


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def rms_norm(
    core_output,
    prepared_norm_gate,
    eps: hl.constexpr,
    value_dim_static: hl.constexpr,
):
    """Megakernel root 7: gated per-head RMS normalization."""
    batch_heads, value_dim = core_output.shape
    assert value_dim == value_dim_static
    assert prepared_norm_gate.shape == (batch_heads, value_dim_static)

    norm_gate_batch_block = hl.register_block_size(1, 16)
    normalized_output = torch.empty_like(core_output)
    for tile_batch_head, tile_value in hl.tile(
        [batch_heads, value_dim_static],
        block_size=[norm_gate_batch_block, value_dim_static],
    ):
        wait_and_launch_dependents()
        values = core_output[tile_batch_head, tile_value].float()
        inv_rms = torch.rsqrt(
            torch.sum(values * values, dim=-1) / value_dim_static + eps
        )
        normalized_output[tile_batch_head, tile_value] = (
            values * inv_rms[:, None] * prepared_norm_gate[tile_batch_head, tile_value]
        ).to(normalized_output.dtype)
    return normalized_output


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def output_projection(
    normalized_output,
    output_weight,
    batch: hl.constexpr,
    heads_static: hl.constexpr,
    value_dim_static: hl.constexpr,
    output_splits: hl.constexpr,
):
    """Megakernel root 8: tensor-parallel-local output projection."""
    batch_heads, value_dim = normalized_output.shape
    output_hidden, output_k = output_weight.shape
    assert batch_heads == batch * heads_static
    assert value_dim == value_dim_static
    assert output_k == heads_static * value_dim_static
    assert output_splits >= 1 and output_k % output_splits == 0

    output_batch_block = hl.register_block_size(1, 16)
    output_block = hl.register_block_size(4, 64)
    output_k_block = hl.register_block_size(32, 512)
    output_chunk_static = heads_static * value_dim_static // output_splits
    normalized_output_flat = normalized_output.view(batch, output_k)
    output = torch.empty(
        (batch, output_hidden),
        dtype=normalized_output.dtype,
        device=normalized_output.device,
    )
    output_partials = torch.empty(
        (batch, output_splits, output_hidden),
        dtype=torch.float32,
        device=normalized_output.device,
    )

    for tile_batch, tile_split, tile_out in hl.tile(
        [batch, output_splits, output_hidden],
        block_size=[output_batch_block, 1, output_block],
    ):
        wait_and_launch_dependents()
        accumulator = hl.zeros([tile_batch, tile_out], dtype=torch.float32)
        for tile_local_input in hl.tile(output_chunk_static, block_size=output_k_block):
            tile_input = tile_split.id * output_chunk_static + tile_local_input.index
            accumulator = torch.addmm(
                accumulator,
                normalized_output_flat[tile_batch, tile_input],
                output_weight[tile_out, tile_input].T,
            )
        if output_splits == 1:
            output[tile_batch, tile_out] = accumulator.to(output.dtype)
        else:
            output_partials[tile_batch, tile_split, tile_out] = accumulator[:, None, :]

    if output_splits > 1:
        for tile_batch, tile_out in hl.tile(
            [batch, output_hidden], block_size=[1, output_block]
        ):
            split_offsets = hl.arange(output_splits)
            partials = output_partials[tile_batch, split_offsets, tile_out]
            output[tile_batch, tile_out] = torch.sum(partials, dim=1).to(output.dtype)
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
    """Compile and materialize the source-matched nine-launch PDL graph."""
    batch = tensors["hidden_states"].shape[0]
    heads = tensors["recurrent_state"].shape[1]
    key_dim = tensors["recurrent_state"].shape[-1]
    value_dim = tensors["recurrent_state"].shape[-2]
    qkv_width = 3 * heads * key_dim
    projection_width = tensors["input_weight"].shape[0]
    use_batched_gates = batch > 4
    output_splits = 1

    gate_input_args = (
        tensors["hidden_states"],
        tensors["input_weight"],
        projection_width,
        qkv_width,
        heads,
        key_dim,
    )
    gate_input_kernel = _compile(
        gate_input_projection,
        gate_input_args,
        _root_config(batch, heads, "gate_input"),
    )
    projected_gate_inputs = gate_input_kernel(*gate_input_args)

    qkv_args = (
        tensors["hidden_states"],
        tensors["input_weight"],
        qkv_width,
        heads,
        key_dim,
    )
    qkv_kernel = _compile(qkv_projection, qkv_args, _root_config(batch, heads, "qkv"))
    projected_qkv = launch_dependent(qkv_kernel, *qkv_args)

    beta_args = (
        tensors["hidden_states"],
        tensors["input_weight"],
        qkv_width,
        heads,
    )
    beta_kernel = _compile(
        beta_projection, beta_args, _root_config(batch, heads, "beta")
    )
    recurrence_beta = launch_dependent(beta_kernel, *beta_args)

    decay_args = (
        projected_gate_inputs,
        tensors["fg_weight"],
        tensors["a_log"],
        tensors["dt_bias"],
        use_batched_gates,
        heads,
        key_dim,
    )
    decay_kernel = _compile(
        prepare_decay, decay_args, _root_config(batch, heads, "decay")
    )
    prepared_decay = launch_dependent(decay_kernel, *decay_args)

    conv_args = (
        projected_qkv,
        tensors["conv_state"],
        tensors["conv_weight"],
        tensors["state_indices"],
        scale,
        heads,
        key_dim,
    )
    conv_kernel = _compile(prepare_qkv, conv_args, _root_config(batch, heads, "conv"))
    prepared_qkv = launch_dependent(conv_kernel, *conv_args)

    norm_gate_args = (
        projected_gate_inputs,
        tensors["fg_weight"],
        tensors["norm_weight"],
        use_batched_gates,
        heads,
        key_dim,
    )
    norm_gate_kernel = _compile(
        prepare_norm_gate,
        norm_gate_args,
        _root_config(batch, heads, "norm_gate"),
    )
    prepared_norm_gate = launch_dependent(norm_gate_kernel, *norm_gate_args)

    recurrence_args = (
        prepared_qkv,
        prepared_decay,
        recurrence_beta,
        tensors["recurrent_state"],
        tensors["state_indices"],
        heads,
        key_dim,
        value_dim,
    )
    recurrence_kernel = _compile(
        recurrent_decode,
        recurrence_args,
        _root_config(batch, heads, "recurrence"),
    )
    core_output = launch_dependent(recurrence_kernel, *recurrence_args)

    rms_args = (core_output, prepared_norm_gate, eps, value_dim)
    rms_kernel = _compile(rms_norm, rms_args, _root_config(batch, heads, "rms"))
    normalized_output = launch_dependent(rms_kernel, *rms_args)

    output_args = (
        normalized_output,
        tensors["output_weight"],
        batch,
        heads,
        value_dim,
        output_splits,
    )
    output_kernel = _compile(
        output_projection,
        output_args,
        _root_config(batch, heads, "output"),
    )
    output = launch_dependent(output_kernel, *output_args)

    kernels = (
        gate_input_kernel,
        qkv_kernel,
        beta_kernel,
        decay_kernel,
        conv_kernel,
        norm_gate_kernel,
        recurrence_kernel,
        rms_kernel,
        output_kernel,
    )

    def launch() -> torch.Tensor:
        local_gate_inputs = gate_input_kernel(*gate_input_args)
        local_qkv = launch_dependent(qkv_kernel, *qkv_args)
        local_beta = launch_dependent(beta_kernel, *beta_args)
        local_decay = launch_dependent(
            decay_kernel,
            local_gate_inputs,
            *decay_args[1:],
        )
        local_prepared_qkv = launch_dependent(
            conv_kernel,
            local_qkv,
            *conv_args[1:],
        )
        local_norm_gate = launch_dependent(
            norm_gate_kernel,
            local_gate_inputs,
            *norm_gate_args[1:],
        )
        local_core = launch_dependent(
            recurrence_kernel,
            local_prepared_qkv,
            local_decay,
            local_beta,
            *recurrence_args[3:],
        )
        local_normalized = launch_dependent(
            rms_kernel,
            local_core,
            local_norm_gate,
            *rms_args[2:],
        )
        return launch_dependent(
            output_kernel,
            local_normalized,
            *output_args[1:],
        )

    return launch, kernels, output
