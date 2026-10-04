# pyrefly: ignore-errors
"""Kimi-K3 KDA decode megakernel, pretuned for NVIDIA GB200.

The persistent kernel covers the TP-local packed input projection, f_b
projection, causal convolution, bounded KDA recurrence, gated RMSNorm, and
output projection.  The checked-in envelopes are Kimi-K3 TP8/H12 and TP16/H6
for B1/B2/B4/B8/B16. Sequence length is absent from the decode signature, and
mutable cache contents plus slot mappings remain runtime data.
"""

from __future__ import annotations

import importlib
import importlib.util
from operator import itemgetter
import os
from pathlib import Path
import sys
import types
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator


HIDDEN = 7168
HEAD_DIM = 128
POOL_SIZE = 32
LOWER_BOUND = -5.0
EPS = 1e-5
SCALE = HEAD_DIM**-0.5
SUPPORTED_BATCHES = (1, 2, 4, 8, 16)
SUPPORTED_HEADS = (12, 6)
BENCHMARK_CASES = tuple(
    (f"b{batch}_h{heads}", batch, heads, 1000 + batch * 10 + heads)
    # TP8/H12 is the production-performance gate. TP16/H6 remains in the
    # correctness matrix and is reported by the development comparison.
    for heads in (12,)
    for batch in SUPPORTED_BATCHES
)
CORRECTNESS_CASES = tuple(
    (batch, heads, 3000 + batch * 10 + heads)
    for heads in SUPPORTED_HEADS
    for batch in SUPPORTED_BATCHES
)
VLLM_LIBRARY_ENV = "HELION_VLLM_LIBRARY"
_VLLM_FALLBACK_OPS: tuple[Callable, Callable, Callable] | None = None


def _projection_width(heads: int) -> int:
    logical_width = 4 * heads * HEAD_DIM + HEAD_DIM + heads
    return (logical_width + 15) // 16 * 16


@helion.aot_kernel(
    static_shapes=False,
    backend="triton",
    triton_do_not_specialize=False,
    persistent_reserved_sms=88,
)
def kda_decode(
    hidden_states: torch.Tensor,
    input_weight: torch.Tensor,
    f_b_weight: torch.Tensor,
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
    lower_bound: hl.constexpr,
    projection_width_static: hl.constexpr,
    bfa_width_static: hl.constexpr,
    heads_static: hl.constexpr,
    key_dim_static: hl.constexpr,
    value_dim_static: hl.constexpr,
) -> torch.Tensor:
    """Run one TP-local Kimi-K3 decode layer in a persistent kernel."""
    batch, hidden = hidden_states.shape
    projection_width, input_hidden = input_weight.shape
    f_b_width, f_b_input = f_b_weight.shape
    conv_kinds, conv_taps, conv_width = conv_weight.shape
    slots, history, qkv_width = conv_state.shape
    state_slots, heads, value_dim, key_dim = recurrent_state.shape
    output_hidden, output_k = output_weight.shape
    batch_heads = batch * heads
    wide_width = 4 * heads * key_dim

    assert hidden == HIDDEN and input_hidden == hidden
    assert projection_width == projection_width_static
    assert projection_width >= wide_width + bfa_width_static
    assert projection_width - (wide_width + bfa_width_static) < 16
    assert f_b_width == heads * key_dim and f_b_input == key_dim
    assert conv_kinds == 3 and conv_taps == history + 1 and history == 3
    assert conv_width == heads * key_dim
    assert qkv_width == 3 * heads * key_dim
    assert state_slots == slots
    assert heads == heads_static
    assert key_dim == key_dim_static and value_dim == value_dim_static
    assert key_dim == HEAD_DIM and value_dim == HEAD_DIM
    assert a_log.shape == (heads,)
    assert dt_bias.shape == (heads * key_dim,)
    assert state_indices.shape == (batch,)
    assert norm_weight.shape == (value_dim,)
    assert output_hidden == hidden and output_k == heads * value_dim

    hl.specialize(
        (
            hidden_states.shape,
            input_weight.shape,
            f_b_weight.shape,
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
            f_b_weight.stride(),
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

    bfa_batch_block = hl.register_block_size(1, 16)
    bfa_output_block = hl.register_block_size(4, 64)
    wide_batch_block = hl.register_block_size(1, 16)
    wide_output_block = hl.register_block_size(4, 256)
    decay_batch_head_block = hl.register_block_size(1, 16)
    decay_output_block = hl.register_block_size(4, 128)
    conv_block = hl.register_block_size(4, 128)
    recurrent_block = hl.register_block_size(4, value_dim)
    rms_batch_head_block = hl.register_block_size(1, 16)
    output_batch_block = hl.register_block_size(1, 16)
    output_block = hl.register_block_size(4, 64)
    bfa_k_block = hl.register_block_size(32, 512)
    wide_k_block = hl.register_block_size(32, 512)
    f_b_k_block = hl.register_block_size(32, 128)
    output_k_block = hl.register_block_size(32, 512)

    wide_weight_view = input_weight[: 4 * heads_static * key_dim_static].view(
        4, heads_static, key_dim_static, hidden
    )
    bfa_weight = input_weight[
        4 * heads_static * key_dim_static : 4 * heads_static * key_dim_static
        + bfa_width_static
    ]
    f_b_weight_view = f_b_weight.view(heads_static, key_dim_static, key_dim_static)
    projected_wide = torch.empty(
        (batch, heads_static, 4, key_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    projected_bfa = torch.empty(
        (batch, bfa_width_static),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    prepared_decay = torch.empty(
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
    normalized_output = torch.empty_like(core_output)
    normalized_flat = normalized_output.view(batch, heads_static * value_dim)
    output = torch.empty(
        (batch, output_hidden),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    # f_a and beta are independent of the full-rank Q/K/V/output-gate root.
    for tile_batch, tile_output in hl.tile(
        [batch, bfa_width_static],
        block_size=[bfa_batch_block, bfa_output_block],
    ):
        accumulator = hl.zeros([tile_batch, tile_output], dtype=torch.float32)
        for tile_hidden in hl.tile(hidden, block_size=bfa_k_block):
            accumulator = torch.addmm(
                accumulator,
                hidden_states[tile_batch, tile_hidden],
                bfa_weight[tile_output, tile_hidden].T,
            )
        projected_bfa[tile_batch, tile_output] = accumulator.to(projected_bfa.dtype)

    for tile_batch, tile_head, tile_kind, tile_dim in hl.tile(
        [batch, heads_static, 4, key_dim],
        block_size=[wide_batch_block, 1, 1, wide_output_block],
    ):
        kind = tile_kind.id
        accumulator = hl.zeros([tile_batch, tile_dim], dtype=torch.float32)
        for tile_hidden in hl.tile(hidden, block_size=wide_k_block):
            accumulator = torch.addmm(
                accumulator,
                hidden_states[tile_batch, tile_hidden],
                wide_weight_view[
                    kind,
                    tile_head.id,
                    tile_dim,
                    tile_hidden,
                ].T,
            )
        projected_wide[tile_batch, tile_head.id, kind, tile_dim] = accumulator.to(
            projected_wide.dtype
        )

    # f_b preserves the model's BF16 projection boundary before the bounded
    # K3 log-decay transform.
    for tile_batch_head, tile_dim in hl.tile(
        [batch_heads, key_dim],
        block_size=[decay_batch_head_block, decay_output_block],
    ):
        accumulator = hl.zeros([tile_dim], dtype=torch.float32)
        for tile_gate_input in hl.tile(key_dim, block_size=f_b_k_block):
            gate_input = projected_bfa[
                tile_batch_head.id // heads_static, tile_gate_input
            ].float()
            gate_weight = f_b_weight_view[
                tile_batch_head.id % heads_static,
                tile_dim,
                tile_gate_input,
            ].float()
            accumulator = accumulator + torch.sum(
                gate_weight * gate_input[None, :], dim=-1
            )
        rounded_gate = accumulator.to(hidden_states.dtype)
        raw_gate = (
            rounded_gate.float()
            + dt_bias[
                (tile_batch_head.id % heads_static) * key_dim + tile_dim.index
            ].float()
        )
        log_decay = lower_bound * torch.sigmoid(
            torch.exp(a_log[tile_batch_head.id % heads_static].float()) * raw_gate
        )
        prepared_decay[tile_batch_head.id, tile_dim] = torch.exp(log_decay)

    for tile_batch_head, tile_kind, tile_dim in hl.tile(
        [batch_heads, 3, key_dim], block_size=[1, 1, conv_block]
    ):
        batch_head_index = tile_batch_head.id
        kind = tile_kind.id
        state_index = state_indices[tile_batch_head.id // heads_static].long()
        channel = (
            kind * heads_static * key_dim
            + (tile_batch_head.id % heads_static) * key_dim
            + tile_dim.index
        )
        conv_channel = (tile_batch_head.id % heads_static) * key_dim + tile_dim.index
        x = projected_wide[
            tile_batch_head.id // heads_static,
            tile_batch_head.id % heads_static,
            kind,
            tile_dim,
        ].float()
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
        value = value.to(hidden_states.dtype).float()
        if kind < 2:
            value = value * torch.rsqrt(torch.sum(value * value) + 1e-6)
            if kind == 0:
                value = value * scale
        prepared_qkv[batch_head_index, kind, tile_dim] = value

    for tile_batch_head, tile_value in hl.tile(
        [batch_heads, value_dim], block_size=[1, recurrent_block]
    ):
        batch_head_index = tile_batch_head.id
        state_index = state_indices[tile_batch_head.id // heads_static].long()
        beta = torch.sigmoid(
            projected_bfa[
                tile_batch_head.id // heads_static,
                key_dim + tile_batch_head.id % heads_static,
            ].float()
        )
        result = hl.zeros([tile_value], dtype=torch.float32)
        for tile_key in hl.tile(key_dim, block_size=key_dim):
            key_offsets = tile_key.index
            decay = prepared_decay[batch_head_index, key_offsets]
            key = prepared_qkv[batch_head_index, 1, key_offsets]
            value = prepared_qkv[batch_head_index, 2, tile_value]
            query = prepared_qkv[batch_head_index, 0, key_offsets]
            if state_index > 0:
                state = recurrent_state[
                    state_index,
                    tile_batch_head.id % heads_static,
                    tile_value.index,
                    key_offsets,
                ].float()
                state = state * decay[None, :]
                value_residual = value - torch.sum(state * key[None, :], dim=-1)
                state = state + (value_residual * beta)[:, None] * key[None, :]
                result = torch.sum(state * query[None, :], dim=-1)
                recurrent_state[
                    state_index,
                    tile_batch_head.id % heads_static,
                    tile_value.index,
                    key_offsets,
                ] = state.to(recurrent_state.dtype)
        core_output[batch_head_index, tile_value] = result.to(core_output.dtype)

    for tile_batch_head, tile_value in hl.tile(
        [batch_heads, value_dim],
        block_size=[rms_batch_head_block, value_dim],
    ):
        values = core_output[tile_batch_head, tile_value].float()
        inv_rms = torch.rsqrt(torch.sum(values * values, dim=-1) / value_dim + eps)
        raw_gate = projected_wide[
            tile_batch_head.id // heads_static,
            tile_batch_head.id % heads_static,
            3,
            tile_value,
        ].float()
        normalized_output[tile_batch_head, tile_value] = (
            values
            * inv_rms[:, None]
            * norm_weight[tile_value].float()[None, :]
            * torch.sigmoid(raw_gate)
        ).to(normalized_output.dtype)

    for tile_batch, tile_output in hl.tile(
        [batch, output_hidden],
        block_size=[output_batch_block, output_block],
    ):
        accumulator = hl.zeros([tile_batch, tile_output], dtype=torch.float32)
        for tile_input in hl.tile(output_k, block_size=output_k_block):
            accumulator = torch.addmm(
                accumulator,
                normalized_flat[tile_batch, tile_input],
                output_weight[tile_output, tile_input].T,
            )
        output[tile_batch, tile_output] = accumulator.to(output.dtype)
    return output


def use_cudagraph() -> bool:
    """The timed closures replay pre-captured CUDA graphs."""
    return True


def _require_sm100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("kda_decode is pretuned only for NVIDIA SM100")


def _make_inputs(batch: int, heads: int, seed: int) -> dict[str, torch.Tensor]:
    if batch not in SUPPORTED_BATCHES or heads not in SUPPORTED_HEADS:
        raise ValueError(f"unsupported pretuned shape B={batch}, H={heads}")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    segment = heads * HEAD_DIM
    qkv_width = 3 * segment

    def randn(
        *shape: int,
        dtype: torch.dtype = torch.bfloat16,
        scale: float = 0.02,
    ) -> torch.Tensor:
        return (
            torch.randn(
                *shape,
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            * scale
        )

    return {
        "hidden_states": randn(batch, HIDDEN),
        "input_weight": randn(_projection_width(heads), HIDDEN),
        "f_b_weight": randn(segment, HEAD_DIM),
        "conv_weight": randn(3, 4, segment, dtype=torch.float32),
        "a_log": torch.log(
            torch.empty(heads, device="cuda", dtype=torch.float32).uniform_(
                1.0, 16.0, generator=generator
            )
        ),
        "dt_bias": randn(segment, dtype=torch.float32),
        "conv_state": randn(POOL_SIZE, 3, qkv_width),
        "recurrent_state": randn(
            POOL_SIZE, heads, HEAD_DIM, HEAD_DIM, dtype=torch.float32
        ),
        "state_indices": torch.arange(1, batch + 1, device="cuda", dtype=torch.int32),
        "norm_weight": 1.0 + randn(HEAD_DIM, dtype=torch.float32),
        "output_weight": randn(HIDDEN, segment),
    }


def _clone_inputs(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cloned = dict(tensors)
    cloned["conv_state"] = tensors["conv_state"].clone()
    cloned["recurrent_state"] = tensors["recurrent_state"].clone()
    return cloned


def _kernel_args(tensors: dict[str, torch.Tensor]) -> tuple[object, ...]:
    heads = tensors["recurrent_state"].shape[1]
    return (
        tensors["hidden_states"],
        tensors["input_weight"],
        tensors["f_b_weight"],
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
        LOWER_BOUND,
        _projection_width(heads),
        HEAD_DIM + heads,
        heads,
        HEAD_DIM,
        HEAD_DIM,
    )


def _project(tensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    batch = tensors["hidden_states"].shape[0]
    heads = tensors["recurrent_state"].shape[1]
    segment = heads * HEAD_DIM
    projected = F.linear(tensors["hidden_states"], tensors["input_weight"])
    mixed_qkv = projected[:, : 3 * segment]
    output_gate = projected[:, 3 * segment : 4 * segment]
    f_a = projected[:, 4 * segment : 4 * segment + HEAD_DIM]
    beta = projected[:, 4 * segment + HEAD_DIM : 4 * segment + HEAD_DIM + heads]
    forget_gate = F.linear(f_a, tensors["f_b_weight"])
    return (
        mixed_qkv,
        output_gate.view(batch, heads, HEAD_DIM),
        forget_gate.view(1, batch, heads, HEAD_DIM),
        beta.view(1, batch, heads),
    )


def _reference(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    """Torch reference preserving K3's BF16 projection boundaries."""
    hidden_states = tensors["hidden_states"]
    batch = hidden_states.shape[0]
    heads = tensors["recurrent_state"].shape[1]
    segment = heads * HEAD_DIM
    mixed_qkv, output_gate, forget_gate, beta = _project(tensors)
    indices = tensors["state_indices"].long()
    history = tensors["conv_state"][indices]
    conv_inputs = torch.cat((history.transpose(1, 2), mixed_qkv[:, :, None]), dim=-1)
    conv_weights = tensors["conv_weight"].permute(0, 2, 1).reshape(3 * segment, 4)
    convolved = torch.sum(conv_inputs.float() * conv_weights[None, :, :], dim=-1)
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
    raw_gate = forget_gate.view(batch, heads, HEAD_DIM).float()
    raw_gate = raw_gate + tensors["dt_bias"].view(1, heads, HEAD_DIM)
    decay = torch.exp(
        LOWER_BOUND
        * torch.sigmoid(torch.exp(tensors["a_log"])[None, :, None] * raw_gate)
    )
    state = tensors["recurrent_state"][indices]
    state = state * decay[:, :, None, :]
    residual = value.float() - torch.sum(state * key[:, :, None, :], dim=-1)
    state = (
        state
        + (residual * torch.sigmoid(beta.float())[0, :, :, None])[:, :, :, None]
        * key[:, :, None, :]
    )
    tensors["recurrent_state"][indices] = state
    core = torch.sum(state * query[:, :, None, :], dim=-1).to(hidden_states.dtype)
    core_float = core.float()
    normalized = (
        core_float
        * torch.rsqrt(torch.mean(core_float * core_float, dim=-1, keepdim=True) + EPS)
        * tensors["norm_weight"][None, None, :]
        * torch.sigmoid(output_gate.float())
    ).to(hidden_states.dtype)
    return F.linear(normalized.reshape(batch, segment), tensors["output_weight"])


def _ensure_vllm_op() -> bool:
    if hasattr(torch.ops._C, "fused_kda_decode"):
        return True
    try:
        importlib.import_module("vllm._custom_ops")
    except (ImportError, OSError):
        library = os.environ.get(VLLM_LIBRARY_ENV)
        if not library:
            return False
        try:
            torch.ops.load_library(library)
        except OSError:
            return False
        sys.modules.setdefault(
            "vllm._C_stable_libtorch",
            types.ModuleType("vllm._C_stable_libtorch"),
        )
    return hasattr(torch.ops._C, "fused_kda_decode")


def _load_vllm_fallback_ops() -> tuple[Callable, Callable, Callable]:
    global _VLLM_FALLBACK_OPS
    if _VLLM_FALLBACK_OPS is not None:
        return _VLLM_FALLBACK_OPS

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.third_party.flash_linear_attention.ops.fused_norm_gate import (
        rms_norm_gated,
    )

    try:
        module = importlib.import_module(
            "vllm.models.kimi_k3.nvidia.ops.third_party.kda.fused_recurrent"
        )
    except ImportError:
        import vllm

        module_path = (
            Path(vllm.__file__).parent
            / "models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py"
        )
        spec = importlib.util.spec_from_file_location(
            "_helion_vllm_kda_fused_recurrent", module_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"cannot load vLLM KDA fallback from {module_path}"
            ) from None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    _VLLM_FALLBACK_OPS = (
        causal_conv1d_update,
        module.fused_recurrent_kda_packed_decode,
        rms_norm_gated,
    )
    return _VLLM_FALLBACK_OPS


def has_vllm() -> bool:
    """Whether both production vLLM KDA decode paths are available."""
    if not _ensure_vllm_op():
        return False
    try:
        _load_vllm_fallback_ops()
    except (ImportError, OSError):
        return False
    return True


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
    segment = heads * HEAD_DIM

    if heads == 12:
        fused_output = torch.empty(
            1,
            batch,
            heads,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )

        def launch_fused() -> torch.Tensor:
            mixed_qkv, output_gate, raw_gate, beta = _project(tensors)
            torch.ops._C.fused_kda_decode(
                mixed_qkv,
                tensors["conv_weight"],
                None,
                tensors["conv_state"].transpose(-1, -2),
                raw_gate,
                beta,
                tensors["a_log"],
                tensors["dt_bias"],
                tensors["state_indices"],
                tensors["recurrent_state"],
                fused_output,
                LOWER_BOUND,
                output_gate,
                tensors["norm_weight"],
                EPS,
            )
            return F.linear(fused_output.view(batch, segment), tensors["output_weight"])

        return launch_fused, "FUSED_KDA_DECODE"

    causal_conv1d_update, fused_recurrent_decode, rms_norm_gated = (
        _load_vllm_fallback_ops()
    )
    fallback_conv_weight = (
        tensors["conv_weight"].permute(0, 2, 1).reshape(3 * segment, 4)
    )

    def launch_fallback() -> torch.Tensor:
        mixed_qkv, output_gate, raw_gate, beta = _project(tensors)
        convolved = causal_conv1d_update(
            mixed_qkv,
            tensors["conv_state"].transpose(-1, -2),
            fallback_conv_weight,
            None,
            activation="silu",
            conv_state_indices=tensors["state_indices"],
            validate_data=False,
        )
        core, _ = fused_recurrent_decode(
            convolved,
            raw_gate,
            beta,
            tensors["a_log"],
            tensors["dt_bias"],
            LOWER_BOUND,
            tensors["recurrent_state"],
            tensors["state_indices"],
            scale=SCALE,
        )
        normalized = rms_norm_gated(
            core,
            output_gate,
            tensors["norm_weight"],
            None,
            activation="sigmoid",
            eps=EPS,
        )
        return F.linear(normalized.view(batch, segment), tensors["output_weight"])

    return launch_fallback, "TRITON_FALLBACK"


def _make_reset(tensors: dict[str, torch.Tensor]) -> Callable[[], None]:
    initial_conv = tensors["conv_state"].clone()
    initial_recurrent = tensors["recurrent_state"].clone()

    def reset() -> None:
        tensors["conv_state"].copy_(initial_conv)
        tensors["recurrent_state"].copy_(initial_recurrent)

    return reset


def _make_standalone_call(
    tensors: dict[str, torch.Tensor],
) -> tuple[Callable[[], torch.Tensor], torch.Tensor]:
    from pretuned_kernels.megakernels.kda_decode import _standalone

    launch, _kernels, output = _standalone.build(
        tensors,
        scale=SCALE,
        eps=EPS,
        lower_bound=LOWER_BOUND,
    )
    return launch, output


def _assert_close(
    actual: torch.Tensor,
    actual_inputs: dict[str, torch.Tensor],
    expected: torch.Tensor,
    expected_inputs: dict[str, torch.Tensor],
) -> None:
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(
        actual_inputs["conv_state"],
        expected_inputs["conv_state"],
        atol=2e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        actual_inputs["recurrent_state"],
        expected_inputs["recurrent_state"],
        atol=3e-2,
        rtol=2e-2,
    )


@torch.inference_mode()
def correctness_check() -> None:
    """Validate both Kimi-K3 TP envelopes against Torch and vLLM."""
    _require_sm100()
    if not has_vllm():
        raise RuntimeError("vLLM is required for the KDA decode comparison")
    for batch, heads, seed in CORRECTNESS_CASES:
        base = _make_inputs(batch, heads, seed)
        persistent_inputs = _clone_inputs(base)
        vllm_inputs = _clone_inputs(base)
        reference_inputs = _clone_inputs(base)
        persistent = kda_decode(*_kernel_args(persistent_inputs))
        vllm_call, _backend = _make_vllm_call(vllm_inputs)
        vllm_output = vllm_call()
        reference = _reference(reference_inputs)
        torch.cuda.synchronize()
        _assert_close(persistent, persistent_inputs, reference, reference_inputs)
        _assert_close(vllm_output, vllm_inputs, reference, reference_inputs)
        _assert_close(persistent, persistent_inputs, vllm_output, vllm_inputs)

        remapped = _clone_inputs(base)
        remapped["state_indices"] = torch.arange(
            POOL_SIZE - 1,
            POOL_SIZE - batch - 1,
            -1,
            device="cuda",
            dtype=torch.int32,
        )
        remapped_vllm = _clone_inputs(base)
        remapped_vllm["state_indices"] = remapped["state_indices"]
        remapped_reference = _clone_inputs(remapped)
        output = kda_decode(*_kernel_args(remapped))
        remapped_vllm_call, _backend = _make_vllm_call(remapped_vllm)
        remapped_vllm_output = remapped_vllm_call()
        expected = _reference(remapped_reference)
        torch.cuda.synchronize()
        _assert_close(output, remapped, expected, remapped_reference)
        _assert_close(
            remapped_vllm_output,
            remapped_vllm,
            expected,
            remapped_reference,
        )
        _assert_close(output, remapped, remapped_vllm_output, remapped_vllm)


@torch.inference_mode()
def main(verbose: bool = True) -> dict:
    """Benchmark Kimi-K3 TP8 against source-matched PDL and vLLM."""
    _require_sm100()
    if not has_vllm():
        raise RuntimeError("kda_decode performance requires vLLM")

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
            standalone_call, standalone = _make_standalone_call(standalone_inputs)
            vllm_call, backend = _make_vllm_call(vllm_inputs)
            vllm_output = vllm_call()
            reference = _reference(reference_inputs)
            torch.cuda.synchronize()
            _assert_close(persistent, persistent_inputs, reference, reference_inputs)
            _assert_close(standalone, standalone_inputs, reference, reference_inputs)
            _assert_close(vllm_output, vllm_inputs, reference, reference_inputs)
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
            (f"{label:>7s}  {batch:>5d}  {heads:>5d}  {HIDDEN:>6d}  {backend:<18s}"),
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
                f"{'case':>7s}  {'batch':>5s}  {'heads':>5s}  {'hidden':>6s}  "
                f"{'backend':<18s}"
            ),
        )
    finally:
        benchmark_cases.close()


if __name__ == "__main__":
    main()
