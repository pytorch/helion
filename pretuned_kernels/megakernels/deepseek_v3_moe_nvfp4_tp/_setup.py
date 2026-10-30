"""Inputs and vLLM's baseline for the DeepSeek-V3 NVFP4 MoE TP4 kernel.

The problem shape, the seeded expert weights (only the experts the routing
cases pick are filled), the routing cases, the TP4 kernel's arguments, and
vLLM's production MoE call on the same weights.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import sys

import torch

VLLM_ROOT = os.environ.get("VLLM_ROOT")
if VLLM_ROOT:
    VLLM_ROOT = str(Path(VLLM_ROOT).resolve())

ROUTED_SCALE = 2.5
ROUTING_CASES = (
    ("spread", (3, 17, 35, 50, 77, 95, 113, 127)),
    ("clustered", (0, 1, 2, 3, 4, 5, 6, 7)),
    ("edge", (128, 159, 160, 191, 192, 223, 224, 255)),
)
ROUTING_LOGITS = (8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0)


@dataclasses.dataclass(frozen=True)
class Shape:
    batch: int = 1
    hidden: int = 7168
    intermediate: int = 2048
    num_experts: int = 256
    top_k: int = 8
    num_groups: int = 8
    topk_groups: int = 4
    routed_scale: float = ROUTED_SCALE


def _ensure_vllm_importable() -> None:
    if VLLM_ROOT is not None and VLLM_ROOT not in sys.path:
        sys.path.insert(0, VLLM_ROOT)


def _router_weight(hidden: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    hidden_f32 = hidden.float()
    denominator = torch.sum(hidden_f32 * hidden_f32)
    return (target.float().T * hidden_f32 / denominator).to(torch.bfloat16)


def _set_routing_case(
    tensors: dict[str, torch.Tensor], selected_ids: tuple[int, ...]
) -> None:
    target = tensors["routing_target"]
    target.fill_(-12.0)
    bias_index = torch.arange(
        tensors["correction_bias"].numel(),
        device=target.device,
        dtype=torch.int32,
    )
    tensors["correction_bias"].copy_((((bias_index * 7) % 17) - 8).float() * 0.001)
    for expert, value in zip(selected_ids, ROUTING_LOGITS, strict=True):
        target[0, expert] = value
    tensors["router_weight"].copy_(_router_weight(tensors["hidden"], target))


def _interleave_w13(
    weight: torch.Tensor,
    scale: torch.Tensor,
    intermediate: int,
    initialized_experts: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleave gate/up rows to match the native-MMA output geometry."""
    native_weight = torch.zeros_like(weight)
    native_scale = torch.zeros_like(scale)
    for expert in initialized_experts:
        native_weight_view = native_weight[expert].view(intermediate, 2, weight.size(2))
        native_weight_view[:, 0].copy_(weight[expert, :intermediate])
        native_weight_view[:, 1].copy_(weight[expert, intermediate:])
        native_scale_view = native_scale[expert].view(intermediate, 2, scale.size(2))
        native_scale_view[:, 0].copy_(scale[expert, :intermediate])
        native_scale_view[:, 1].copy_(scale[expert, intermediate:])
    return native_weight, native_scale


def _allocate(shape: Shape, seed: int = 17) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    hidden = torch.randn(shape.batch, shape.hidden, device=device, dtype=torch.bfloat16)
    routing_target = torch.empty(
        shape.batch, shape.num_experts, device=device, dtype=torch.bfloat16
    )
    router_weight = torch.empty(
        shape.num_experts, shape.hidden, device=device, dtype=torch.bfloat16
    )
    correction_bias = torch.zeros(shape.num_experts, device=device, dtype=torch.float32)
    input_global_scale = torch.ones(1, device=device, dtype=torch.float32)
    activation_global_scale = torch.full((1,), 16.0, device=device, dtype=torch.float32)

    weight_w13 = torch.zeros(
        shape.num_experts,
        2 * shape.intermediate,
        shape.hidden // 2,
        device=device,
        dtype=torch.uint8,
    )
    scale_w13 = torch.zeros(
        shape.num_experts,
        2 * shape.intermediate,
        shape.hidden // 16,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    weight_w2 = torch.zeros(
        shape.num_experts,
        shape.hidden,
        shape.intermediate // 2,
        device=device,
        dtype=torch.uint8,
    )
    scale_w2 = torch.zeros(
        shape.num_experts,
        shape.hidden,
        shape.intermediate // 16,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    shared_weight_w13 = torch.empty(
        2 * shape.intermediate,
        shape.hidden // 2,
        device=device,
        dtype=torch.uint8,
    )
    shared_scale_w13 = torch.empty(
        2 * shape.intermediate,
        shape.hidden // 16,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    shared_weight_w2 = torch.empty(
        shape.hidden,
        shape.intermediate // 2,
        device=device,
        dtype=torch.uint8,
    )
    shared_scale_w2 = torch.empty(
        shape.hidden,
        shape.intermediate // 16,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    initialized_experts = tuple(
        sorted(
            {
                expert
                for _label, selected_ids in ROUTING_CASES
                for expert in selected_ids
            }
            | {shape.num_experts - 1}
        )
    )
    w13_rows = torch.arange(2 * shape.intermediate, device=device, dtype=torch.int32)[
        :, None
    ]
    w13_groups = torch.arange(shape.hidden // 16, device=device, dtype=torch.int32)[
        None, :
    ]
    w13_scale_pattern = torch.exp2((-9 + (w13_rows + 3 * w13_groups) % 3).float()) * (
        1.0 + ((5 * w13_rows + 7 * w13_groups) % 8).float() / 8.0
    )
    w2_rows = torch.arange(shape.hidden, device=device, dtype=torch.int32)[:, None]
    w2_groups = torch.arange(
        shape.intermediate // 16, device=device, dtype=torch.int32
    )[None, :]
    w2_scale_pattern = torch.exp2((-7 + (w2_rows + 5 * w2_groups) % 3).float()) * (
        1.0 + ((3 * w2_rows + 7 * w2_groups) % 8).float() / 8.0
    )
    for expert in initialized_experts:
        weight_w13[expert].random_(0, 256)
        scale_w13[expert].copy_(
            torch.roll(w13_scale_pattern, shifts=expert % 3, dims=1)
        )
        weight_w2[expert].random_(0, 256)
        scale_w2[expert].copy_(torch.roll(w2_scale_pattern, shifts=expert % 3, dims=1))

    shared_weight_w13.random_(0, 256)
    shared_scale_w13.copy_(torch.roll(w13_scale_pattern, shifts=1, dims=1))
    shared_weight_w2.random_(0, 256)
    shared_scale_w2.copy_(torch.roll(w2_scale_pattern, shifts=1, dims=1))

    native_w13, native_w13_scale = _interleave_w13(
        weight_w13,
        scale_w13,
        shape.intermediate,
        initialized_experts,
    )
    native_shared_w13, native_shared_w13_scale = _interleave_w13(
        shared_weight_w13[None],
        shared_scale_w13[None],
        shape.intermediate,
        (0,),
    )
    alpha1 = torch.ones(shape.num_experts, device=device, dtype=torch.float32)
    alpha2 = torch.full(
        (shape.num_experts,),
        1.0 / float(activation_global_scale.item()),
        device=device,
        dtype=torch.float32,
    )
    shared_alpha1 = torch.ones(1, device=device, dtype=torch.float32)
    shared_alpha2 = torch.full(
        (1,),
        1.0 / float(activation_global_scale.item()),
        device=device,
        dtype=torch.float32,
    )
    tensors = {
        "hidden": hidden,
        "routing_target": routing_target,
        "router_weight": router_weight,
        "correction_bias": correction_bias,
        "w13": weight_w13,
        "w13_scale": scale_w13,
        "native_w13": native_w13,
        "native_w13_scale": native_w13_scale,
        "w2": weight_w2,
        "w2_scale": scale_w2,
        "shared_w13": shared_weight_w13,
        "shared_w13_scale": shared_scale_w13,
        "native_shared_w13": native_shared_w13[0],
        "native_shared_w13_scale": native_shared_w13_scale[0],
        "shared_w2": shared_weight_w2,
        "shared_w2_scale": shared_scale_w2,
        "alpha1": alpha1,
        "alpha2": alpha2,
        "shared_alpha1": shared_alpha1,
        "shared_alpha2": shared_alpha2,
        "input_global_scale": input_global_scale,
        "activation_global_scale": activation_global_scale,
    }
    _set_routing_case(tensors, ROUTING_CASES[0][1])
    return tensors


def _kernel_args(tensors: dict[str, torch.Tensor], shape: Shape) -> tuple:
    """Kernel arguments with untiled weights (the TP4 kernel's layout)."""
    return (
        tensors["hidden"],
        tensors["router_weight"],
        tensors["correction_bias"],
        tensors["native_w13"],
        tensors["native_w13_scale"],
        tensors["w2"],
        tensors["w2_scale"],
        tensors["native_shared_w13"],
        tensors["native_shared_w13_scale"],
        tensors["shared_w2"],
        tensors["shared_w2_scale"],
        tensors["alpha1"],
        tensors["alpha2"],
        tensors["shared_alpha1"],
        tensors["shared_alpha2"],
        tensors["input_global_scale"],
        tensors["activation_global_scale"],
        shape.top_k,
        shape.num_groups,
        shape.topk_groups,
        shape.routed_scale,
    )


def _prepare_production_weights(
    tensors: dict[str, torch.Tensor], shape: Shape
) -> tuple[torch.Tensor, ...]:
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        prepare_static_weights_for_trtllm_fp4_moe,
    )

    # ModelOpt checkpoints store [gate; up]; TRT-LLM consumes [up; gate].
    weight_w13 = torch.cat(
        (
            tensors["w13"][:, shape.intermediate :],
            tensors["w13"][:, : shape.intermediate],
        ),
        dim=1,
    )
    scale_w13 = torch.cat(
        (
            tensors["w13_scale"][:, shape.intermediate :],
            tensors["w13_scale"][:, : shape.intermediate],
        ),
        dim=1,
    )
    return prepare_static_weights_for_trtllm_fp4_moe(
        weight_w13,
        tensors["w2"],
        scale_w13,
        tensors["w2_scale"],
        hidden_size=shape.hidden,
        intermediate_size=shape.intermediate,
        num_experts=shape.num_experts,
        is_gated_activation=True,
    )


def _make_vllm_linear(
    weight: torch.Tensor,
    scale: torch.Tensor,
    output_size: int,
    input_scale: float,
    alpha: float,
) -> tuple[object, torch.nn.Module]:
    from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scale.clone(), requires_grad=False)
    layer.input_size_per_partition = 2 * weight.shape[1]
    layer.output_size_per_partition = output_size
    layer.input_global_scale_inv = torch.nn.Parameter(
        torch.tensor(input_scale, device="cuda", dtype=torch.float32),
        requires_grad=False,
    )
    layer.alpha = torch.nn.Parameter(
        torch.tensor(alpha, device="cuda", dtype=torch.float32),
        requires_grad=False,
    )
    kernel = init_nvfp4_linear_kernel()
    kernel.process_weights_after_loading(layer)
    return kernel, layer


def _make_vllm_weights(tensors: dict[str, torch.Tensor], shape: Shape) -> tuple:
    """vLLM's prepared expert weights and its shared-expert linear layers."""
    _ensure_vllm_importable()
    return (
        *_prepare_production_weights(tensors, shape),
        *_make_vllm_linear(
            tensors["shared_w13"],
            tensors["shared_w13_scale"],
            2 * shape.intermediate,
            1.0,
            float(tensors["shared_alpha1"].item()),
        ),
        *_make_vllm_linear(
            tensors["shared_w2"],
            tensors["shared_w2_scale"],
            shape.hidden,
            16.0,
            float(tensors["shared_alpha2"].item()),
        ),
    )


def _make_vllm_call(
    tensors: dict[str, torch.Tensor],
    shape: Shape,
    weights: tuple | None = None,
) -> tuple[object, str, torch.Tensor]:
    """vLLM's production call on ``tensors``, with ``weights`` from
    ``_make_vllm_weights`` (prepared here when not given)."""
    _ensure_vllm_importable()
    import flashinfer
    from flashinfer import autotune as flashinfer_autotune
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        activation_to_flashinfer_int,
    )

    if weights is None:
        weights = _make_vllm_weights(tensors, shape)
    (
        weight_w13,
        scale_w13,
        weight_w2,
        scale_w2,
        shared_w13_kernel,
        shared_w13_layer,
        shared_w2_kernel,
        shared_w2_layer,
    ) = weights
    output = torch.empty(shape.batch, shape.hidden, device="cuda", dtype=torch.bfloat16)
    routing_replay = torch.empty(
        shape.batch, shape.top_k, device="cuda", dtype=torch.int16
    )
    output1_scale_scalar = tensors["alpha1"] * tensors["activation_global_scale"]

    def routed() -> torch.Tensor:
        # vLLM's GateLinear for DeepSeek-V3 (BF16 weight, no router dtype):
        # its specialized router GEMMs need FP32 logits, so it runs F.linear.
        logits = torch.nn.functional.linear(tensors["hidden"], tensors["router_weight"])
        hidden_q, hidden_scale = ops.scaled_fp4_quant(
            tensors["hidden"],
            tensors["input_global_scale"],
            is_sf_swizzled_layout=False,
        )
        result = flashinfer.fused_moe.trtllm_fp4_block_scale_moe(
            routing_logits=logits,
            routing_bias=tensors["correction_bias"],
            hidden_states=hidden_q,
            hidden_states_scale=hidden_scale,
            gemm1_weights=weight_w13,
            gemm1_weights_scale=scale_w13,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=weight_w2,
            gemm2_weights_scale=scale_w2,
            gemm2_bias=None,
            output1_scale_scalar=output1_scale_scalar,
            output1_scale_gate_scalar=tensors["alpha1"],
            output2_scale_scalar=tensors["alpha2"],
            num_experts=shape.num_experts,
            top_k=shape.top_k,
            n_group=shape.num_groups,
            topk_group=shape.topk_groups,
            intermediate_size=shape.intermediate,
            local_expert_offset=0,
            local_num_experts=shape.num_experts,
            routed_scaling_factor=shape.routed_scale,
            routing_method_type=RoutingMethodType.DeepSeekV3,
            do_finalize=True,
            activation_type=activation_to_flashinfer_int(MoEActivation.SILU),
            tune_max_num_tokens=shape.batch,
            output=output,
            routing_replay_out=routing_replay,
        )
        return result[0]

    shared_activation = torch.empty(
        shape.batch, shape.intermediate, device="cuda", dtype=torch.bfloat16
    )

    def shared() -> torch.Tensor:
        gate_up = shared_w13_kernel.apply_weights(shared_w13_layer, tensors["hidden"])
        torch.ops._C.silu_and_mul(shared_activation, gate_up)
        return shared_w2_kernel.apply_weights(shared_w2_layer, shared_activation)

    shared_stream = torch.cuda.Stream()

    def launch() -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        shared_stream.wait_stream(current_stream)
        with torch.cuda.stream(shared_stream):
            shared_output = shared()
        routed_output = routed()
        current_stream.wait_stream(shared_stream)
        return routed_output + shared_output

    # vLLM autotunes FlashInfer's MoE at startup, every token bucket up to its
    # max_num_batched_tokens; tune the bucket of this call's token count (more
    # buckets would overrun routing_replay, which holds one token).
    with flashinfer_autotune(True):
        launch()
    torch.cuda.synchronize()

    def launch_tuned() -> torch.Tensor:
        with flashinfer_autotune(False):
            return launch()

    return launch_tuned, "flashinfer_trtllm", routing_replay


def _similarity_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual_f64 = actual.double()
    expected_f64 = expected.double()
    denominator = (actual_f64.square() + expected_f64.square()).sum().clamp(min=1e-30)
    return float(1.0 - 2.0 * (actual_f64 * expected_f64).sum() / denominator)


def _dispatch_cache_signature(call: object) -> tuple[tuple[object, int], ...]:
    cache = getattr(call, "_dispatch_cache", None)
    if not cache:
        raise RuntimeError("AOT dispatch cache was not populated after launch")
    return tuple((key, id(bound)) for key, bound in cache.items())
