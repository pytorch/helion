# ruff: noqa: ANN001, ANN202
"""Source-matched seven-launch PDL ablation for Kimi-K3 KDA decode.

Each kernel is one top-level root from :func:`kda_decode`. Tensor layouts,
rounding points, cache mutation, and algebra are identical; only the execution
model differs: seven separately launched PDL grids replace persistent dynamic
cross-root scheduling.
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
    "wide": (2, 3, 12),
    "bfa": (0, 1, 11),
    "decay": (4, 5, 13),
    "conv": (6,),
    "recurrence": (7,),
    "rms": (8,),
    "output": (9, 10, 14),
}
_LOOP_ORDER_INDICES = {
    "bfa": 0,
    "wide": 1,
    "decay": 2,
    "conv": 3,
    "recurrence": 4,
    "rms": 5,
    "output": 6,
}
_ROOT_BLOCK_OVERRIDES: dict[int, dict[str, list[int]]] = {
    batch: {} for batch in (1, 2, 4, 8, 16)
}


def _root_config(batch: int, heads: int, root: str) -> dict[str, object]:
    """Project a checked-in megakernel config onto one standalone root."""
    megakernel = MEGAKERNEL_CONFIGS[batch, heads]
    block_sizes = megakernel["block_sizes"]
    default_blocks = [block_sizes[index] for index in _BLOCK_INDICES[root]]
    config: dict[str, object] = {
        "block_sizes": _ROOT_BLOCK_OVERRIDES[batch].get(root, default_blocks),
        "num_warps": megakernel["num_warps"],
        "num_stages": megakernel["num_stages"],
        "pid_type": "flat",
        "indexing": megakernel["indexing"],
        "load_eviction_policies": "",
    }
    loop_orders = megakernel.get("loop_orders")
    if loop_orders is not None:
        config["loop_orders"] = [loop_orders[_LOOP_ORDER_INDICES[root]]]
    return config


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def wide_projection(
    hidden_states,
    input_weight,
    wide_width_static: hl.constexpr,
    heads_static: hl.constexpr,
    key_dim_static: hl.constexpr,
):
    """Project checkpoint-contiguous Q, K, V, and full-rank output gate."""
    batch, hidden = hidden_states.shape
    projection_width, input_hidden = input_weight.shape
    assert input_hidden == hidden
    assert wide_width_static == 4 * heads_static * key_dim_static
    assert projection_width >= wide_width_static

    batch_block = hl.register_block_size(1, 16)
    output_block = hl.register_block_size(4, 256)
    k_block = hl.register_block_size(32, 512)
    weight = input_weight[:wide_width_static].view(
        4, heads_static, key_dim_static, hidden
    )
    projected = torch.empty(
        (batch, heads_static, 4, key_dim_static),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    for tile_batch, tile_head, tile_kind, tile_dim in hl.tile(
        [batch, heads_static, 4, key_dim_static],
        block_size=[batch_block, 1, 1, output_block],
    ):
        signal_dependents()
        kind = tile_kind.id
        accumulator = hl.zeros([tile_batch, tile_dim], dtype=torch.float32)
        for tile_hidden in hl.tile(hidden, block_size=k_block):
            accumulator = torch.addmm(
                accumulator,
                hidden_states[tile_batch, tile_hidden],
                weight[kind, tile_head.id, tile_dim, tile_hidden].T,
            )
        projected[tile_batch, tile_head.id, kind, tile_dim] = accumulator.to(
            projected.dtype
        )
    return projected


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def bfa_projection(
    hidden_states,
    input_weight,
    wide_width_static: hl.constexpr,
    bfa_width_static: hl.constexpr,
):
    """Project the independent f_a and beta checkpoint slice."""
    batch, hidden = hidden_states.shape
    projection_width, input_hidden = input_weight.shape
    assert input_hidden == hidden
    assert projection_width >= wide_width_static + bfa_width_static

    batch_block = hl.register_block_size(1, 16)
    output_block = hl.register_block_size(4, 64)
    k_block = hl.register_block_size(32, 512)
    weight = input_weight[wide_width_static : wide_width_static + bfa_width_static]
    projected = torch.empty(
        (batch, bfa_width_static),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    for tile_batch, tile_output in hl.tile(
        [batch, bfa_width_static], block_size=[batch_block, output_block]
    ):
        wait_and_launch_dependents()
        accumulator = hl.zeros([tile_batch, tile_output], dtype=torch.float32)
        for tile_hidden in hl.tile(hidden, block_size=k_block):
            accumulator = torch.addmm(
                accumulator,
                hidden_states[tile_batch, tile_hidden],
                weight[tile_output, tile_hidden].T,
            )
        projected[tile_batch, tile_output] = accumulator.to(projected.dtype)
    return projected


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def prepare_decay(
    projected_bfa,
    f_b_weight,
    a_log,
    dt_bias,
    lower_bound: hl.constexpr,
    heads_static: hl.constexpr,
    key_dim_static: hl.constexpr,
):
    """Apply f_b and form K3's bounded per-channel recurrence decay."""
    batch, bfa_width = projected_bfa.shape
    f_b_width, f_b_input = f_b_weight.shape
    assert bfa_width >= key_dim_static + heads_static
    assert f_b_width == heads_static * key_dim_static
    assert f_b_input == key_dim_static
    assert a_log.shape == (heads_static,)
    assert dt_bias.shape == (heads_static * key_dim_static,)

    batch_heads = batch * heads_static
    batch_head_block = hl.register_block_size(1, 16)
    output_block = hl.register_block_size(4, 128)
    k_block = hl.register_block_size(32, 128)
    weight = f_b_weight.view(heads_static, key_dim_static, key_dim_static)
    decay = torch.empty(
        (batch_heads, key_dim_static),
        dtype=torch.float32,
        device=projected_bfa.device,
    )
    for tile_batch_head, tile_dim in hl.tile(
        [batch_heads, key_dim_static],
        block_size=[batch_head_block, output_block],
    ):
        wait_and_launch_dependents()
        accumulator = hl.zeros([tile_dim], dtype=torch.float32)
        for tile_input in hl.tile(key_dim_static, block_size=k_block):
            gate_input = projected_bfa[
                tile_batch_head.id // heads_static, tile_input
            ].float()
            gate_weight = weight[
                tile_batch_head.id % heads_static, tile_dim, tile_input
            ].float()
            accumulator = accumulator + torch.sum(
                gate_weight * gate_input[None, :], dim=-1
            )
        rounded_gate = accumulator.to(projected_bfa.dtype)
        raw_gate = (
            rounded_gate.float()
            + dt_bias[
                (tile_batch_head.id % heads_static) * key_dim_static + tile_dim.index
            ].float()
        )
        log_decay = lower_bound * torch.sigmoid(
            torch.exp(a_log[tile_batch_head.id % heads_static].float()) * raw_gate
        )
        decay[tile_batch_head.id, tile_dim] = torch.exp(log_decay)
    return decay


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def prepare_qkv(
    projected_wide,
    conv_weight,
    conv_state,
    state_indices,
    scale: hl.constexpr,
    heads_static: hl.constexpr,
    key_dim_static: hl.constexpr,
):
    """Update the causal-convolution cache and prepare normalized Q/K plus V."""
    batch, heads, kinds, key_dim = projected_wide.shape
    conv_kinds, conv_taps, conv_width = conv_weight.shape
    slots, history, qkv_width = conv_state.shape
    assert heads == heads_static and kinds == 4 and key_dim == key_dim_static
    assert conv_kinds == 3 and conv_taps == history + 1 and history == 3
    assert conv_width == heads_static * key_dim_static
    assert qkv_width == 3 * heads_static * key_dim_static
    assert state_indices.shape == (batch,)

    batch_heads = batch * heads_static
    conv_block = hl.register_block_size(4, 128)
    prepared = torch.empty(
        (batch_heads, 3, key_dim_static),
        dtype=torch.float32,
        device=projected_wide.device,
    )
    for tile_batch_head, tile_kind, tile_dim in hl.tile(
        [batch_heads, 3, key_dim_static], block_size=[1, 1, conv_block]
    ):
        wait_and_launch_dependents()
        batch_head_index = tile_batch_head.id
        kind = tile_kind.id
        batch_index = batch_head_index // heads_static
        head = batch_head_index % heads_static
        state_index = state_indices[batch_index].long()
        channel = (
            kind * heads_static * key_dim_static
            + head * key_dim_static
            + tile_dim.index
        )
        conv_channel = head * key_dim_static + tile_dim.index
        x = projected_wide[batch_index, head, kind, tile_dim].float()
        value = hl.zeros([tile_dim], dtype=torch.float32)
        if state_index > 0:
            state_0 = conv_state[state_index, 0, channel].float()
            state_1 = conv_state[state_index, 1, channel].float()
            state_2 = conv_state[state_index, 2, channel].float()
            value = (
                state_0 * conv_weight[kind, 0, conv_channel].float()
                + state_1 * conv_weight[kind, 1, conv_channel].float()
                + state_2 * conv_weight[kind, 2, conv_channel].float()
                + x * conv_weight[kind, 3, conv_channel].float()
            )
            value = value * torch.sigmoid(value)
            conv_state[state_index, 0, channel] = state_1.to(conv_state.dtype)
            conv_state[state_index, 1, channel] = state_2.to(conv_state.dtype)
            conv_state[state_index, 2, channel] = x.to(conv_state.dtype)
        value = value.to(projected_wide.dtype).float()
        if kind < 2:
            value = value * torch.rsqrt(torch.sum(value * value) + 1e-6)
            if kind == 0:
                value = value * scale
        prepared[batch_head_index, kind, tile_dim] = value
    return prepared


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def recurrent_decode(
    prepared_qkv,
    prepared_decay,
    projected_bfa,
    recurrent_state,
    state_indices,
    heads_static: hl.constexpr,
    key_dim_static: hl.constexpr,
    value_dim_static: hl.constexpr,
):
    """Apply the bounded KDA state update and produce one output per head."""
    batch_heads, kinds, key_dim = prepared_qkv.shape
    slots, heads, value_dim, state_key_dim = recurrent_state.shape
    batch, bfa_width = projected_bfa.shape
    assert kinds == 3 and key_dim == key_dim_static
    assert prepared_decay.shape == (batch_heads, key_dim_static)
    assert heads == heads_static
    assert batch_heads == batch * heads_static
    assert bfa_width >= key_dim_static + heads_static
    assert value_dim == value_dim_static and state_key_dim == key_dim_static
    assert state_indices.shape == (batch,)

    value_block = hl.register_block_size(4, value_dim_static)
    output = torch.empty(
        (batch_heads, value_dim_static),
        dtype=projected_bfa.dtype,
        device=projected_bfa.device,
    )
    for tile_batch_head, tile_value in hl.tile(
        [batch_heads, value_dim_static], block_size=[1, value_block]
    ):
        wait_and_launch_dependents()
        batch_head_index = tile_batch_head.id
        batch_index = batch_head_index // heads_static
        head = batch_head_index % heads_static
        state_index = state_indices[batch_index].long()
        beta = torch.sigmoid(projected_bfa[batch_index, key_dim_static + head].float())
        result = hl.zeros([tile_value], dtype=torch.float32)
        for tile_key in hl.tile(key_dim_static, block_size=key_dim_static):
            key_offsets = tile_key.index
            decay = prepared_decay[batch_head_index, key_offsets]
            key = prepared_qkv[batch_head_index, 1, key_offsets]
            value = prepared_qkv[batch_head_index, 2, tile_value]
            query = prepared_qkv[batch_head_index, 0, key_offsets]
            if state_index > 0:
                state = recurrent_state[
                    state_index, head, tile_value.index, key_offsets
                ].float()
                state = state * decay[None, :]
                value_residual = value - torch.sum(state * key[None, :], dim=-1)
                state = state + (value_residual * beta)[:, None] * key[None, :]
                result = torch.sum(state * query[None, :], dim=-1)
                recurrent_state[state_index, head, tile_value.index, key_offsets] = (
                    state.to(recurrent_state.dtype)
                )
        output[batch_head_index, tile_value] = result.to(output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def gated_rms_norm(
    core_output,
    projected_wide,
    norm_weight,
    eps: hl.constexpr,
    heads_static: hl.constexpr,
    value_dim_static: hl.constexpr,
):
    """Apply the full-rank sigmoid gate and per-head RMS normalization."""
    batch_heads, value_dim = core_output.shape
    batch, heads, kinds, gate_dim = projected_wide.shape
    assert batch_heads == batch * heads_static
    assert heads == heads_static and kinds == 4
    assert value_dim == value_dim_static and gate_dim == value_dim_static
    assert norm_weight.shape == (value_dim_static,)

    batch_head_block = hl.register_block_size(1, 16)
    normalized = torch.empty_like(core_output)
    for tile_batch_head, tile_value in hl.tile(
        [batch_heads, value_dim_static],
        block_size=[batch_head_block, value_dim_static],
    ):
        wait_and_launch_dependents()
        values = core_output[tile_batch_head, tile_value].float()
        inv_rms = torch.rsqrt(
            torch.sum(values * values, dim=-1) / value_dim_static + eps
        )
        raw_gate = projected_wide[
            tile_batch_head.id // heads_static,
            tile_batch_head.id % heads_static,
            3,
            tile_value,
        ].float()
        normalized[tile_batch_head, tile_value] = (
            values
            * inv_rms[:, None]
            * norm_weight[tile_value].float()[None, :]
            * torch.sigmoid(raw_gate)
        ).to(normalized.dtype)
    return normalized


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def output_projection(
    normalized_output,
    output_weight,
    batch: hl.constexpr,
    heads_static: hl.constexpr,
    value_dim_static: hl.constexpr,
):
    """Apply the tensor-parallel-local K3 output projection."""
    batch_heads, value_dim = normalized_output.shape
    output_hidden, output_k = output_weight.shape
    assert batch_heads == batch * heads_static
    assert value_dim == value_dim_static
    assert output_k == heads_static * value_dim_static

    batch_block = hl.register_block_size(1, 16)
    output_block = hl.register_block_size(4, 64)
    k_block = hl.register_block_size(32, 512)
    normalized_flat = normalized_output.view(batch, output_k)
    output = torch.empty(
        (batch, output_hidden),
        dtype=normalized_output.dtype,
        device=normalized_output.device,
    )
    for tile_batch, tile_output in hl.tile(
        [batch, output_hidden], block_size=[batch_block, output_block]
    ):
        wait_and_launch_dependents()
        accumulator = hl.zeros([tile_batch, tile_output], dtype=torch.float32)
        for tile_input in hl.tile(output_k, block_size=k_block):
            accumulator = torch.addmm(
                accumulator,
                normalized_flat[tile_batch, tile_input],
                output_weight[tile_output, tile_input].T,
            )
        output[tile_batch, tile_output] = accumulator.to(output.dtype)
    return output


def _compile(kernel, args, config_values):
    bound = kernel.bind(args)
    values = dict(bound.config_spec.default_config())
    values.update(config_values)
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return bound.compile_config(config)


def build(
    tensors: dict[str, torch.Tensor],
    *,
    scale: float,
    eps: float,
    lower_bound: float,
) -> tuple[Callable[[], torch.Tensor], tuple[object, ...], torch.Tensor]:
    """Compile and materialize the source-matched seven-launch PDL graph."""
    batch = tensors["hidden_states"].shape[0]
    heads = tensors["recurrent_state"].shape[1]
    key_dim = tensors["recurrent_state"].shape[-1]
    value_dim = tensors["recurrent_state"].shape[-2]
    wide_width = 4 * heads * key_dim
    bfa_width = key_dim + heads

    wide_args = (
        tensors["hidden_states"],
        tensors["input_weight"],
        wide_width,
        heads,
        key_dim,
    )
    wide_kernel = _compile(
        wide_projection, wide_args, _root_config(batch, heads, "wide")
    )
    projected_wide = wide_kernel(*wide_args)

    bfa_args = (
        tensors["hidden_states"],
        tensors["input_weight"],
        wide_width,
        bfa_width,
    )
    bfa_kernel = _compile(bfa_projection, bfa_args, _root_config(batch, heads, "bfa"))
    projected_bfa = launch_dependent(bfa_kernel, *bfa_args)

    decay_args = (
        projected_bfa,
        tensors["f_b_weight"],
        tensors["a_log"],
        tensors["dt_bias"],
        lower_bound,
        heads,
        key_dim,
    )
    decay_kernel = _compile(
        prepare_decay, decay_args, _root_config(batch, heads, "decay")
    )
    prepared_decay = launch_dependent(decay_kernel, *decay_args)

    conv_args = (
        projected_wide,
        tensors["conv_weight"],
        tensors["conv_state"],
        tensors["state_indices"],
        scale,
        heads,
        key_dim,
    )
    conv_kernel = _compile(prepare_qkv, conv_args, _root_config(batch, heads, "conv"))
    prepared_qkv = launch_dependent(conv_kernel, *conv_args)

    recurrence_args = (
        prepared_qkv,
        prepared_decay,
        projected_bfa,
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

    rms_args = (
        core_output,
        projected_wide,
        tensors["norm_weight"],
        eps,
        heads,
        value_dim,
    )
    rms_kernel = _compile(gated_rms_norm, rms_args, _root_config(batch, heads, "rms"))
    normalized = launch_dependent(rms_kernel, *rms_args)

    output_args = (
        normalized,
        tensors["output_weight"],
        batch,
        heads,
        value_dim,
    )
    output_kernel = _compile(
        output_projection, output_args, _root_config(batch, heads, "output")
    )
    output = launch_dependent(output_kernel, *output_args)

    kernels = (
        wide_kernel,
        bfa_kernel,
        decay_kernel,
        conv_kernel,
        recurrence_kernel,
        rms_kernel,
        output_kernel,
    )

    def launch() -> torch.Tensor:
        local_wide = wide_kernel(*wide_args)
        local_bfa = launch_dependent(bfa_kernel, *bfa_args)
        local_decay = launch_dependent(decay_kernel, local_bfa, *decay_args[1:])
        local_qkv = launch_dependent(conv_kernel, local_wide, *conv_args[1:])
        local_core = launch_dependent(
            recurrence_kernel,
            local_qkv,
            local_decay,
            local_bfa,
            *recurrence_args[3:],
        )
        local_normalized = launch_dependent(
            rms_kernel, local_core, local_wide, *rms_args[2:]
        )
        return launch_dependent(output_kernel, local_normalized, *output_args[1:])

    return launch, kernels, output
