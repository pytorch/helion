# pyrefly: ignore-errors
"""Kimi-Linear KDA decode megakernel, pretuned for NVIDIA B200.

The single persistent Helion kernel covers the local KDA sublayer from the
fused input projection through the output projection.  Its independent input,
gate, and QKV work is dynamically scheduled around the causal-convolution and
recurrent-state dependencies.  Mutable cache contents and ``state_indices``
remain runtime values; the checked-in heuristic supports the six physical
envelopes B1/B2/B4 x H12/H16 at hidden size 2304 and head dimension 128.

The benchmark compares against a production-boundary-matched standalone Helion
PDL pipeline and, for H12, vLLM's production fused conv1d + KDA + gated-RMSNorm
CUDA kernel surrounded by the same projections.
"""

from __future__ import annotations

from operator import itemgetter
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator


HIDDEN = 2304
HEAD_DIM = 128
POOL_SIZE = 32
EPS = 1e-5
SCALE = HEAD_DIM**-0.5
OUTPUT_SPLITS = 1
SUPPORTED_BATCHES = (1, 2, 4)
SUPPORTED_HEADS = (12, 16)
BENCHMARK_CASES = (
    ("b1_h12", 1, 12, 101),
    ("b2_h12", 2, 12, 202),
    ("b4_h12", 4, 12, 204),
)
CORRECTNESS_CASES = (
    (1, 12, 301),
    (2, 12, 302),
    (4, 12, 304),
    (1, 16, 401),
    (2, 16, 402),
    (4, 16, 404),
)
VLLM_LIBRARY_ENV = "HELION_VLLM_LIBRARY"


@helion.aot_kernel(
    static_shapes=False,
    backend="triton",
    triton_do_not_specialize=False,
    persistent_reserved_sms=88,
)
def kda_decode(
    hidden_states: torch.Tensor,
    input_weight: torch.Tensor,
    fg_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
    norm_weight: torch.Tensor,
    output_weight: torch.Tensor,
    scale: hl.constexpr,
    eps: hl.constexpr,
    output_splits: hl.constexpr,
    projection_width_static: hl.constexpr,
    qkv_width_static: hl.constexpr,
    heads_static: hl.constexpr,
    key_dim_static: hl.constexpr,
    value_dim_static: hl.constexpr,
) -> torch.Tensor:
    """Run one dense KDA decode sublayer in a persistent kernel."""
    batch, hidden = hidden_states.shape
    slots, heads, value_dim, key_dim = recurrent_state.shape
    state_slots, history, qkv_width = conv_state.shape
    projection_width, input_hidden = input_weight.shape
    output_hidden, output_k = output_weight.shape
    batch_heads = batch * heads

    assert state_slots == slots
    assert input_hidden == hidden
    assert value_dim == key_dim
    assert key_dim_static % 32 == 0
    assert qkv_width == heads * (2 * key_dim + value_dim)
    assert projection_width == qkv_width + heads + 2 * key_dim
    assert fg_weight.shape == (2, heads * key_dim, key_dim)
    assert conv_weight.shape == (qkv_width, history + 1)
    assert history == 3
    assert a_log.numel() == heads
    assert dt_bias.numel() == heads * key_dim
    assert state_indices.shape == (batch,)
    assert norm_weight.shape == (value_dim,)
    assert output_k == heads * value_dim
    assert output_splits >= 1 and output_k % output_splits == 0
    assert projection_width == projection_width_static
    assert qkv_width == qkv_width_static
    assert heads == heads_static
    assert key_dim == key_dim_static
    assert value_dim == value_dim_static

    # Specialize the fixed physical envelope and layouts, but never specialize
    # cache contents or state_indices.  Runtime slot mappings therefore reuse
    # one compiled kernel for each B/H capacity.
    hl.specialize(
        (
            hidden_states.shape,
            input_weight.shape,
            fg_weight.shape,
            conv_weight.shape,
            a_log.shape,
            dt_bias.shape,
            conv_state.shape,
            recurrent_state.shape,
            state_indices.shape,
            norm_weight.shape,
            output_weight.shape,
            hidden_states.stride(),
            input_weight.stride(),
            fg_weight.stride(),
            conv_weight.stride(),
            a_log.stride(),
            dt_bias.stride(),
            conv_state.stride(),
            recurrent_state.stride(),
            state_indices.stride(),
            norm_weight.stride(),
            output_weight.stride(),
        )
    )

    gate_input_batch_block = hl.register_block_size(1, 16)
    gate_input_block = hl.register_block_size(4, 64)
    projection_batch_block = hl.register_block_size(1, 16)
    projection_block = hl.register_block_size(4, 64)
    forget_gate_block = hl.register_block_size(4, 128)
    conv_block = hl.register_block_size(4, 128)
    norm_gate_block = hl.register_block_size(4, 128)
    recurrent_block = hl.register_block_size(4, value_dim)
    output_batch_block = hl.register_block_size(1, 16)
    output_block = hl.register_block_size(4, 64)
    gate_input_k_block = hl.register_block_size(32, 512)
    qkv_input_k_block = hl.register_block_size(32, 512)
    beta_input_k_block = hl.register_block_size(32, 512)
    output_k_block = hl.register_block_size(32, 512)
    output_chunk_static = heads_static * value_dim_static // output_splits
    beta_head_block = 1 << (heads_static - 1).bit_length()

    qkv_input_weight = input_weight[:qkv_width_static].view(
        3, heads_static, key_dim_static, hidden
    )
    beta_input_weight = input_weight[qkv_width_static : qkv_width_static + heads_static]
    gate_input_weight = input_weight[
        qkv_width_static + heads_static : projection_width_static
    ].view(2, key_dim_static, hidden)
    projected_qkv = torch.empty(
        (batch, heads_static, 3, key_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    recurrence_scalars = torch.empty(
        (batch_heads, 2), dtype=torch.float32, device=hidden_states.device
    )
    projected_gate_inputs = torch.empty(
        (batch, 2, key_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    prepared_decay = torch.empty(
        (batch_heads, key_dim), dtype=torch.float32, device=hidden_states.device
    )
    prepared_norm_gate = torch.empty(
        (batch_heads, key_dim), dtype=torch.float32, device=hidden_states.device
    )
    prepared_qkv = torch.empty(
        (batch_heads, 3, key_dim),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    core_output = torch.empty(
        (batch_heads, value_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    normalized_output = torch.empty(
        (batch_heads, value_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    normalized_output_flat = normalized_output.view(
        batch, heads_static * value_dim_static
    )
    output = torch.empty(
        (batch, output_hidden),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    output_partials = torch.empty(
        (batch, output_splits, output_hidden),
        dtype=torch.float32,
        device=hidden_states.device,
    )

    # The first three roots are independent projections.  Fine-grained
    # readiness lets downstream gate, convolution, and recurrence work begin
    # without waiting for the entire projection region.
    for tile_batch, tile_kind, tile_dim in hl.tile(
        [batch, 2, key_dim],
        block_size=[gate_input_batch_block, 1, gate_input_block],
    ):
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

    for tile_batch, tile_head, tile_kind, tile_dim in hl.tile(
        [batch, heads_static, 3, key_dim],
        block_size=[projection_batch_block, 1, 1, projection_block],
    ):
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

    for tile_batch, tile_head in hl.tile(
        [batch, heads_static], block_size=[1, beta_head_block]
    ):
        beta_accumulator = hl.zeros([tile_batch, tile_head], dtype=torch.float32)
        for beta_tile_hidden in hl.tile(hidden, block_size=beta_input_k_block):
            beta_accumulator = torch.addmm(
                beta_accumulator,
                hidden_states[tile_batch, beta_tile_hidden],
                beta_input_weight[tile_head, beta_tile_hidden].T,
            )
        flat_batch_heads = (
            tile_batch.index[:, None] * heads_static + tile_head.index[None, :]
        )
        recurrence_scalars[flat_batch_heads, 0] = torch.sigmoid(
            beta_accumulator.to(hidden_states.dtype).float()
        )
        recurrence_scalars[flat_batch_heads, 1] = torch.exp(
            a_log[tile_head.index].float()
        )[None, :]

    for tile_batch_head, tile_gate_dim in hl.tile(
        [batch_heads, key_dim_static], block_size=[1, forget_gate_block]
    ):
        batch_head_index = tile_batch_head.id
        gate_dim = tile_gate_dim.index
        gate_key = hl.arange(key_dim_static)
        gate_input = projected_gate_inputs[
            tile_batch_head.id // heads_static, 0, gate_key
        ].float()
        gate_weights = fg_weight[
            0,
            (tile_batch_head.id % heads_static) * key_dim_static + gate_dim[:, None],
            gate_key[None, :],
        ].float()
        gate_accumulator = torch.sum(gate_weights * gate_input[None, :], dim=-1)
        rounded_gate = gate_accumulator.to(hidden_states.dtype)
        activated_gate = (
            rounded_gate.float()
            + dt_bias[
                (tile_batch_head.id % heads_static) * key_dim_static + gate_dim
            ].float()
        )
        gate_exp = torch.exp(activated_gate)
        softplus = torch.where(
            activated_gate <= 20.0,
            torch.log(1.0 + gate_exp),
            activated_gate,
        )
        prepared_decay[batch_head_index, gate_dim] = torch.exp(
            -torch.exp(a_log[tile_batch_head.id % heads_static].float()) * softplus
        )

    for tile_batch_head, tile_kind, tile_dim in hl.tile(
        [batch_heads, 3, key_dim], block_size=[1, 1, conv_block]
    ):
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
        value = value.to(hidden_states.dtype).float()
        if kind < 2:
            value = value * torch.rsqrt(torch.sum(value * value) + 1e-6)
            if kind == 0:
                value = value * scale
        prepared_qkv[batch_head_index, kind, tile_dim] = value

    for tile_batch_head, tile_gate_dim in hl.tile(
        [batch_heads, key_dim_static], block_size=[1, norm_gate_block]
    ):
        batch_head_index = tile_batch_head.id
        gate_dim = tile_gate_dim.index
        gate_key = hl.arange(key_dim_static)
        gate_input = projected_gate_inputs[
            tile_batch_head.id // heads_static, 1, gate_key
        ].float()
        gate_weights = fg_weight[
            1,
            (tile_batch_head.id % heads_static) * key_dim_static + gate_dim[:, None],
            gate_key[None, :],
        ].float()
        gate_accumulator = torch.sum(gate_weights * gate_input[None, :], dim=-1)
        rounded_gate = gate_accumulator.to(hidden_states.dtype)
        prepared_norm_gate[batch_head_index, gate_dim] = norm_weight[
            gate_dim
        ].float() * torch.sigmoid(rounded_gate.float())

    for tile_batch_head, tile_value in hl.tile(
        [batch_heads, value_dim], block_size=[1, recurrent_block]
    ):
        batch_head_index = tile_batch_head.id
        state_index = state_indices[tile_batch_head.id // heads_static].long()
        result = hl.zeros([tile_value], dtype=torch.float32)
        for tile_key in hl.tile(key_dim_static, block_size=key_dim_static):
            key_offsets = tile_key.index
            decay = prepared_decay[batch_head_index, key_offsets]
            beta_value = recurrence_scalars[batch_head_index, 0]
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

    for tile_batch_head, tile_value in hl.tile(
        [batch_heads, value_dim], block_size=[1, value_dim_static]
    ):
        values = core_output[tile_batch_head.id, tile_value].float()
        inv_rms = torch.rsqrt(torch.sum(values * values) / value_dim_static + eps)
        normalized_output[tile_batch_head.id, tile_value] = (
            values * inv_rms * prepared_norm_gate[tile_batch_head.id, tile_value]
        ).to(normalized_output.dtype)

    for tile_batch, tile_split, tile_out in hl.tile(
        [batch, output_splits, output_hidden],
        block_size=[output_batch_block, 1, output_block],
    ):
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


def use_cudagraph() -> bool:
    """The timed closures replay pre-captured CUDA graphs."""
    return True


def _require_sm100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("kda_decode is pretuned only for NVIDIA SM100")


def _shape_values(heads: int) -> tuple[int, int]:
    qkv_width = 3 * heads * HEAD_DIM
    projection_width = qkv_width + heads + 2 * HEAD_DIM
    return qkv_width, projection_width


def _make_inputs(batch: int, heads: int, seed: int) -> dict[str, torch.Tensor]:
    if batch not in SUPPORTED_BATCHES or heads not in SUPPORTED_HEADS:
        raise ValueError(f"unsupported pretuned shape B={batch}, H={heads}")
    torch.manual_seed(seed)
    qkv_width, projection_width = _shape_values(heads)

    def randn(
        *shape: int,
        dtype: torch.dtype = torch.bfloat16,
        scale: float = 0.02,
    ) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", dtype=dtype) * scale

    conv_weight = randn(qkv_width, 4, dtype=torch.float32)
    norm_weight = randn(HEAD_DIM, scale=1.0)
    return {
        "hidden_states": randn(batch, HIDDEN),
        "input_weight": randn(projection_width, HIDDEN),
        "fg_weight": randn(2, heads * HEAD_DIM, HEAD_DIM),
        "conv_weight": conv_weight,
        "vllm_conv_weight": conv_weight.reshape(3, heads * HEAD_DIM, 4)
        .transpose(1, 2)
        .contiguous(),
        "a_log": randn(heads, dtype=torch.float32),
        "dt_bias": randn(heads * HEAD_DIM, dtype=torch.float32),
        "conv_state": randn(POOL_SIZE, 3, qkv_width),
        "recurrent_state": randn(
            POOL_SIZE, heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32
        ),
        "state_indices": torch.arange(batch, device="cuda", dtype=torch.int32),
        "norm_weight": norm_weight,
        "vllm_norm_weight": norm_weight.float(),
        "output_weight": randn(HIDDEN, heads * HEAD_DIM),
    }


def _clone_inputs(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cloned = dict(tensors)
    cloned["conv_state"] = tensors["conv_state"].clone()
    cloned["recurrent_state"] = tensors["recurrent_state"].clone()
    return cloned


def _kernel_args(tensors: dict[str, torch.Tensor]) -> tuple[object, ...]:
    heads = tensors["recurrent_state"].shape[1]
    qkv_width, projection_width = _shape_values(heads)
    return (
        tensors["hidden_states"],
        tensors["input_weight"],
        tensors["fg_weight"],
        tensors["conv_weight"],
        tensors["a_log"],
        tensors["dt_bias"],
        tensors["conv_state"],
        tensors["recurrent_state"],
        tensors["state_indices"],
        tensors["norm_weight"],
        tensors["output_weight"],
        SCALE,
        EPS,
        OUTPUT_SPLITS,
        projection_width,
        qkv_width,
        heads,
        HEAD_DIM,
        HEAD_DIM,
    )


def _reference(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    """Torch reference with the same BF16 stage boundaries and cache updates."""
    hidden_states = tensors["hidden_states"]
    batch = hidden_states.shape[0]
    heads = tensors["recurrent_state"].shape[1]
    qkv_width, _projection_width = _shape_values(heads)
    projected = F.linear(hidden_states, tensors["input_weight"])
    mixed_qkv = projected[:, :qkv_width]
    beta = projected[:, qkv_width : qkv_width + heads]
    gate_inputs = projected[:, qkv_width + heads :].view(batch, 2, HEAD_DIM)
    forget_gate, norm_gate = torch.bmm(
        gate_inputs.transpose(0, 1), tensors["fg_weight"].transpose(-1, -2)
    )

    indices = tensors["state_indices"].long()
    history = tensors["conv_state"][indices]
    conv_inputs = torch.cat((history, mixed_qkv[:, None, :]), dim=1).transpose(1, 2)
    convolved = torch.sum(
        conv_inputs.float() * tensors["conv_weight"][None, :, :].float(), dim=-1
    )
    convolved = F.silu(convolved).to(hidden_states.dtype)
    tensors["conv_state"][indices, 0] = history[:, 1]
    tensors["conv_state"][indices, 1] = history[:, 2]
    tensors["conv_state"][indices, 2] = mixed_qkv

    query, key, value = convolved.view(batch, 3, heads, HEAD_DIM).unbind(1)
    query = query.float()
    key = key.float()
    query = query * torch.rsqrt(torch.sum(query * query, dim=-1, keepdim=True) + 1e-6)
    key = key * torch.rsqrt(torch.sum(key * key, dim=-1, keepdim=True) + 1e-6)
    query = query * SCALE
    activated_gate = forget_gate.view(batch, heads, HEAD_DIM).float() + tensors[
        "dt_bias"
    ].view(1, heads, HEAD_DIM)
    decay = torch.exp(
        -torch.exp(tensors["a_log"].float())[None, :, None] * F.softplus(activated_gate)
    )
    state = tensors["recurrent_state"][indices]
    state = state * decay[:, :, None, :]
    residual = value.float() - torch.sum(state * key[:, :, None, :], dim=-1)
    state = (
        state
        + (residual * torch.sigmoid(beta.float())[:, :, None])[:, :, :, None]
        * key[:, :, None, :]
    )
    tensors["recurrent_state"][indices] = state
    core = torch.sum(state * query[:, :, None, :], dim=-1).to(hidden_states.dtype)
    core_float = core.float()
    normalized = (
        core_float
        * torch.rsqrt(torch.mean(core_float * core_float, dim=-1, keepdim=True) + EPS)
        * tensors["norm_weight"].float()[None, None, :]
        * torch.sigmoid(norm_gate.view(batch, heads, HEAD_DIM).float())
    ).to(hidden_states.dtype)
    return F.linear(
        normalized.reshape(batch, heads * HEAD_DIM), tensors["output_weight"]
    )


def _ensure_vllm_op() -> bool:
    if hasattr(torch.ops._C, "fused_kda_decode"):
        return True
    try:
        __import__("vllm._custom_ops")
    except (ImportError, OSError):
        library = os.environ.get(VLLM_LIBRARY_ENV)
        if not library:
            return False
        try:
            torch.ops.load_library(library)
        except OSError:
            return False
    return hasattr(torch.ops._C, "fused_kda_decode")


def has_vllm() -> bool:
    """Whether vLLM's production NVIDIA fused KDA op is available."""
    return _ensure_vllm_op()


def _make_vllm_call(
    tensors: dict[str, torch.Tensor],
) -> tuple[Callable[[], torch.Tensor], str]:
    if not _ensure_vllm_op():
        raise RuntimeError(
            "vLLM fused_kda_decode is unavailable; install vLLM or set "
            f"{VLLM_LIBRARY_ENV}"
        )
    batch = tensors["hidden_states"].shape[0]
    heads = tensors["recurrent_state"].shape[1]
    if heads != 12:
        raise ValueError("vLLM's production fused KDA kernel does not support H16")
    qkv_width, _projection_width = _shape_values(heads)
    normalized = torch.empty(
        1, batch, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )

    def launch() -> torch.Tensor:
        projected = F.linear(tensors["hidden_states"], tensors["input_weight"])
        mixed_qkv = projected[:, :qkv_width]
        beta = projected[:, qkv_width : qkv_width + heads]
        gate_inputs = projected[:, qkv_width + heads :].view(batch, 2, HEAD_DIM)
        forget_gate, norm_gate = torch.bmm(
            gate_inputs.transpose(0, 1), tensors["fg_weight"].transpose(-1, -2)
        )
        torch.ops._C.fused_kda_decode(
            mixed_qkv,
            tensors["vllm_conv_weight"],
            None,
            tensors["conv_state"].transpose(-1, -2),
            forget_gate.reshape(1, batch, heads, HEAD_DIM),
            beta.reshape(1, batch, heads),
            tensors["a_log"],
            tensors["dt_bias"],
            tensors["state_indices"],
            tensors["recurrent_state"],
            normalized,
            None,
            norm_gate.reshape(batch, heads, HEAD_DIM),
            tensors["vllm_norm_weight"],
            EPS,
        )
        return F.linear(
            normalized.reshape(batch, heads * HEAD_DIM), tensors["output_weight"]
        )

    return launch, "FUSED_KDA_DECODE"


def _make_standalone_call(
    tensors: dict[str, torch.Tensor],
) -> tuple[Callable[[], torch.Tensor], tuple[object, ...], torch.Tensor]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from pretuned_kernels.megakernels.kda_decode import _standalone

    return _standalone.build(tensors, scale=SCALE, eps=EPS)


def _make_reset(tensors: dict[str, torch.Tensor]) -> Callable[[], None]:
    initial_conv = tensors["conv_state"].clone()
    initial_recurrent = tensors["recurrent_state"].clone()

    def reset() -> None:
        tensors["conv_state"].copy_(initial_conv)
        tensors["recurrent_state"].copy_(initial_recurrent)

    return reset


def _assert_close(
    actual: torch.Tensor,
    actual_inputs: dict[str, torch.Tensor],
    expected: torch.Tensor,
    expected_inputs: dict[str, torch.Tensor],
) -> None:
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(
        actual_inputs["conv_state"], expected_inputs["conv_state"], atol=2e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_inputs["recurrent_state"],
        expected_inputs["recurrent_state"],
        atol=2e-2,
        rtol=1e-2,
    )


@torch.inference_mode()
def correctness_check() -> None:
    """Check both tuned head counts and runtime slot values."""
    _require_sm100()
    for batch, heads, seed in CORRECTNESS_CASES:
        base = _make_inputs(batch, heads, seed)
        persistent_inputs = _clone_inputs(base)
        standalone_inputs = _clone_inputs(base)
        reference_inputs = _clone_inputs(base)
        persistent = kda_decode(*_kernel_args(persistent_inputs))
        _standalone_call, _standalone_kernels, standalone = _make_standalone_call(
            standalone_inputs
        )
        reference = _reference(reference_inputs)
        torch.cuda.synchronize()
        _assert_close(persistent, persistent_inputs, reference, reference_inputs)
        _assert_close(standalone, standalone_inputs, reference, reference_inputs)

        # state_indices is runtime data, not a specialization key.
        if batch == 1:
            remapped = _clone_inputs(base)
            remapped["state_indices"] = torch.tensor(
                [POOL_SIZE - 1], device="cuda", dtype=torch.int32
            )
            remapped_reference = _clone_inputs(remapped)
            output = kda_decode(*_kernel_args(remapped))
            expected = _reference(remapped_reference)
            _assert_close(output, remapped, expected, remapped_reference)


@torch.inference_mode()
def main(verbose: bool = True) -> dict:
    """Benchmark persistent, separate Helion, and production vLLM with cold L2."""
    _require_sm100()
    if not has_vllm():
        raise RuntimeError("kda_decode performance requires vLLM's fused_kda_decode op")

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from _bench import capture_cuda_graph
    from _bench import run_sweep

    def iter_benchmark_cases() -> Iterator[tuple[object, ...]]:
        for label, batch, heads, seed in BENCHMARK_CASES:
            base = _make_inputs(batch, heads, seed)
            persistent_inputs = _clone_inputs(base)
            standalone_inputs = _clone_inputs(base)
            vllm_inputs = _clone_inputs(base)
            reference_inputs = _clone_inputs(base)
            persistent_reset = _make_reset(persistent_inputs)
            standalone_reset = _make_reset(standalone_inputs)
            vllm_reset = _make_reset(vllm_inputs)

            persistent = kda_decode(*_kernel_args(persistent_inputs))
            standalone_call, standalone_kernels, standalone = _make_standalone_call(
                standalone_inputs
            )
            vllm_call, backend = _make_vllm_call(vllm_inputs)
            vllm = vllm_call()
            reference = _reference(reference_inputs)
            torch.cuda.synchronize()
            _assert_close(persistent, persistent_inputs, reference, reference_inputs)
            _assert_close(standalone, standalone_inputs, reference, reference_inputs)
            _assert_close(vllm, vllm_inputs, reference, reference_inputs)

            persistent_graph, persistent_graph_output = capture_cuda_graph(
                lambda tensors=persistent_inputs: kda_decode(*_kernel_args(tensors)),
                persistent_reset,
            )
            standalone_graph, standalone_graph_output = capture_cuda_graph(
                standalone_call, standalone_reset
            )
            vllm_graph, vllm_graph_output = capture_cuda_graph(vllm_call, vllm_reset)
            yield (
                label,
                batch,
                heads,
                backend,
                persistent_graph,
                standalone_graph,
                vllm_graph,
                persistent_reset,
                standalone_reset,
                vllm_reset,
                (
                    base,
                    persistent_inputs,
                    standalone_inputs,
                    vllm_inputs,
                    standalone_kernels,
                    standalone_call,
                    vllm_call,
                    persistent_graph_output,
                    standalone_graph_output,
                    vllm_graph_output,
                ),
            )

    def make_calls(benchmark_case: tuple) -> tuple:
        (
            label,
            batch,
            heads,
            backend,
            persistent_graph,
            standalone_graph,
            vllm_graph,
            *_,
        ) = benchmark_case
        return (
            persistent_graph.replay,
            [
                ("standalone_helion_pdl", standalone_graph.replay),
                (f"vllm_auto ({backend})", vllm_graph.replay),
            ],
            f"{label:>7s}  {batch:>5d}  {heads:>5d}  {HIDDEN:>6d}",
        )

    benchmark_cases = iter_benchmark_cases()
    try:
        return run_sweep(
            benchmark_cases,
            make_calls,
            use_cudagraph=False,
            pre_captured_cudagraph=True,
            make_resets=itemgetter(slice(7, 10)),
            thermal_warmup_ms=10_000,
            verbose=verbose,
            shape_header=(
                f"{'case':>7s}  {'batch':>5s}  {'heads':>5s}  {'hidden':>6s}"
            ),
        )
    finally:
        benchmark_cases.close()


if __name__ == "__main__":
    main()
