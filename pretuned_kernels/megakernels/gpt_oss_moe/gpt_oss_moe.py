# ruff: noqa: ANN001, ANN201, ANN202
# pyrefly: ignore-errors
"""GPT-OSS 120B batch-one MXFP4 MoE megakernel for NVIDIA B200.

The single kernel preserves the production routing, gate/up plus OAI SwiGLU,
down projection plus bias, and weighted-finalization boundaries.  The matched
standalone baseline uses the same Helion operations as four PDL-chained CUDA
launches.
"""

from __future__ import annotations

from pathlib import Path

import torch

import helion
import helion.language as hl

OUTPUT_NAMES = (
    "output",
    "topk_weights",
    "topk_ids",
    "activation",
    "expert_output",
)
ROUTING_CASES = (
    ("spread", (3, 17, 64, 101)),
    ("clustered", (0, 1, 2, 3)),
    ("edge", (7, 31, 79, 127)),
)
ROUTING_LOGITS = (4.0, 3.0, 2.0, 1.0)


def _trtllm_scale_offset(row, col, rows: int, cols: int):
    """Offset in FlashInfer's per-expert SWIZZLE_128_4_4 scale layout."""
    col_tiles = (cols + 3) // 4
    return (
        ((row >> 7) * col_tiles + (col >> 2)) * 512
        + (row & 31) * 16
        + ((row & 127) >> 5) * 4
        + (col & 3)
    )


def _e8m0_byte_to_f32(scale_byte: torch.Tensor) -> torch.Tensor:
    """Decode an unsigned E8M0 exponent byte without an SFU operation."""
    return (scale_byte.to(torch.int32) << 23).view(dtype=torch.float32)


def _split_contiguous_lanes(value, leading, width: int) -> tuple:
    """Expose a power-of-two minor dimension without tensor narrowing."""
    if width == 1:
        return (value.reshape(leading),)
    half_width = width // 2
    low, high = hl.split(value.reshape(leading, 2, half_width).permute(0, 2, 1))
    return (
        *_split_contiguous_lanes(low, leading, half_width),
        *_split_contiguous_lanes(high, leading, half_width),
    )


@helion.aot_kernel(static_shapes=True, backend="triton")
def gpt_oss_moe(
    routing_logits,
    hidden_states,
    w13,
    w13_scale_bytes,
    w13_bias,
    w2,
    w2_scale_bytes,
    w2_bias,
    output_hidden,
):
    router_input = routing_logits
    routing_tokens, routing_experts = router_input.size()
    routing_top_k = 4
    hl.specialize(routing_experts)
    topk_weights = torch.empty(
        (routing_tokens, routing_top_k),
        dtype=torch.bfloat16,
        device=router_input.device,
    )
    topk_ids = torch.empty(
        (routing_tokens, routing_top_k),
        dtype=torch.int32,
        device=router_input.device,
    )
    gemm1_hidden_states = hidden_states
    gemm1_w13 = w13
    gemm1_w13_scale_bytes = w13_scale_bytes
    gemm1_w13_bias = w13_bias
    gemm1_topk_ids = topk_ids
    gemm1_tokens, gemm1_hidden = gemm1_hidden_states.size()
    gemm1_experts, gemm1_twice_intermediate, gemm1_packed_hidden = gemm1_w13.size()
    gemm1_top_k = gemm1_topk_ids.size(1)
    gemm1_intermediate = gemm1_twice_intermediate // 2
    gemm1_scale_k = gemm1_hidden // 32
    gemm1_activation_scale_k = gemm1_intermediate // 32
    assert gemm1_tokens == 1
    assert gemm1_packed_hidden * 2 == gemm1_hidden
    hl.specialize(gemm1_experts)
    hl.specialize(gemm1_top_k)
    hl.specialize(gemm1_intermediate)
    activation = torch.empty(
        (gemm1_top_k, gemm1_intermediate),
        dtype=torch.bfloat16,
        device=gemm1_hidden_states.device,
    )
    gemm1_physical_output = activation.view(
        gemm1_top_k, gemm1_activation_scale_k, 2, 8, 2
    )
    gemm1_flat_weight = gemm1_w13.view(torch.uint8).view(-1)
    gemm1_flat_bias = gemm1_w13_bias.view(-1)
    gemm1_flat_scale = gemm1_w13_scale_bytes.view(-1)
    gemm1_block_scale_k = hl.register_block_size(1, gemm1_scale_k)
    gemm2_activation = activation
    gemm2_w2 = w2
    gemm2_w2_scale_bytes = w2_scale_bytes
    gemm2_w2_bias = w2_bias
    gemm2_topk_ids = topk_ids
    gemm2_top_k, gemm2_intermediate = gemm2_activation.size()
    gemm2_experts, gemm2_hidden, gemm2_packed_intermediate = gemm2_w2.size()
    gemm2_scale_k = gemm2_intermediate // 32
    assert gemm2_packed_intermediate * 2 == gemm2_intermediate
    hl.specialize(gemm2_experts)
    hl.specialize(gemm2_top_k)
    hl.specialize(gemm2_hidden)
    expert_output = torch.empty(
        (gemm2_top_k, gemm2_hidden),
        dtype=torch.bfloat16,
        device=gemm2_activation.device,
    )
    gemm2_physical_activation = gemm2_activation.view(
        gemm2_top_k, gemm2_scale_k, 2, 8, 2
    )
    gemm2_flat_weight = gemm2_w2.view(torch.uint8).view(-1)
    gemm2_flat_bias = gemm2_w2_bias.view(-1)
    gemm2_flat_scale = gemm2_w2_scale_bytes.view(-1)
    gemm2_block_physical_row = hl.register_block_size(8, 128)
    gemm2_block_scale_k = hl.register_block_size(1, gemm2_scale_k)
    finalize_expert_output = expert_output
    finalize_topk_weights = topk_weights
    finalize_output_hidden = output_hidden
    finalize_top_k, finalize_hidden = finalize_expert_output.size()
    finalize_output_hidden = hl.specialize(finalize_output_hidden)
    assert finalize_output_hidden <= finalize_hidden
    output = torch.empty(
        (1, finalize_output_hidden),
        dtype=finalize_expert_output.dtype,
        device=finalize_expert_output.device,
    )
    finalize_weights = finalize_topk_weights.view(finalize_top_k)
    for routing_tile_t in hl.tile(routing_tokens, block_size=1):
        routing_values = router_input[routing_tile_t, :].to(torch.float32)
        routing_expert_index = hl.arange(routing_experts)
        routing_first_value = torch.amax(routing_values, dim=-1)
        routing_first_id = torch.argmax(routing_values, dim=-1)
        routing_remaining = torch.where(
            routing_expert_index[None, :] == routing_first_id[:, None],
            float("-inf"),
            routing_values,
        )
        routing_second_value = torch.amax(routing_remaining, dim=-1)
        routing_second_id = torch.argmax(routing_remaining, dim=-1)
        routing_remaining = torch.where(
            routing_expert_index[None, :] == routing_second_id[:, None],
            float("-inf"),
            routing_remaining,
        )
        routing_third_value = torch.amax(routing_remaining, dim=-1)
        routing_third_id = torch.argmax(routing_remaining, dim=-1)
        routing_remaining = torch.where(
            routing_expert_index[None, :] == routing_third_id[:, None],
            float("-inf"),
            routing_remaining,
        )
        routing_fourth_value = torch.amax(routing_remaining, dim=-1)
        routing_fourth_id = torch.argmax(routing_remaining, dim=-1)
        routing_second_probability = torch.exp(
            routing_second_value - routing_first_value
        )
        routing_third_probability = torch.exp(routing_third_value - routing_first_value)
        routing_fourth_probability = torch.exp(
            routing_fourth_value - routing_first_value
        )
        routing_denominator = (
            1.0
            + routing_second_probability
            + routing_third_probability
            + routing_fourth_probability
        )
        routing_route_slot = hl.arange(routing_top_k)
        routing_probabilities = torch.where(
            routing_route_slot[None, :] == 0,
            (1.0 / routing_denominator)[:, None],
            torch.where(
                routing_route_slot[None, :] == 1,
                (routing_second_probability / routing_denominator)[:, None],
                torch.where(
                    routing_route_slot[None, :] == 2,
                    (routing_third_probability / routing_denominator)[:, None],
                    (routing_fourth_probability / routing_denominator)[:, None],
                ),
            ),
        )
        routing_selected_ids = torch.where(
            routing_route_slot[None, :] == 0,
            routing_first_id[:, None],
            torch.where(
                routing_route_slot[None, :] == 1,
                routing_second_id[:, None],
                torch.where(
                    routing_route_slot[None, :] == 2,
                    routing_third_id[:, None],
                    routing_fourth_id[:, None],
                ),
            ),
        )
        topk_weights[routing_tile_t, routing_route_slot] = routing_probabilities.to(
            torch.bfloat16
        )
        topk_ids[routing_tile_t, routing_route_slot] = routing_selected_ids.to(
            torch.int32
        )
    for (
        gemm1_tile_half,
        gemm1_tile_parity,
        gemm1_tile_activation_group,
        gemm1_tile_slot,
    ) in hl.tile(
        [2, 2, gemm1_activation_scale_k, gemm1_top_k], block_size=[1, 1, 1, 1]
    ):
        gemm1_slot = gemm1_tile_slot.begin
        gemm1_half = gemm1_tile_half.begin
        gemm1_parity = gemm1_tile_parity.begin
        gemm1_activation_group = gemm1_tile_activation_group.begin
        gemm1_chunk = gemm1_half * 2 + gemm1_parity
        gemm1_physical_row = (
            gemm1_activation_group * 64 + gemm1_chunk * 16 + hl.arange(16)
        )
        gemm1_expert = gemm1_topk_ids[0, gemm1_slot]
        gemm1_expert_row = gemm1_expert * gemm1_twice_intermediate + gemm1_physical_row
        gemm1_accumulator = hl.zeros([16], dtype=torch.float32)
        for gemm1_tile_scale_k in hl.tile(
            gemm1_scale_k, block_size=gemm1_block_scale_k
        ):
            gemm1_group_mask = gemm1_tile_scale_k.index < gemm1_scale_k
            gemm1_subgroup = gemm1_expert_row[:, None] * (gemm1_scale_k * 2)
            gemm1_subgroup += gemm1_tile_scale_k.index[None, :] * 2
            gemm1_valid = gemm1_group_mask[None, :]
            gemm1_weight_first = hl.load_float4_e2m1fn_x16_to_float16(
                gemm1_flat_weight, gemm1_subgroup, extra_mask=gemm1_valid
            )
            gemm1_weight_second = hl.load_float4_e2m1fn_x16_to_float16(
                gemm1_flat_weight, gemm1_subgroup + 1, extra_mask=gemm1_valid
            )
            gemm1_activation_first = hl.load_bfloat16_x16_to_float16(
                gemm1_hidden_states,
                gemm1_tile_scale_k.index * 2,
                extra_mask=gemm1_group_mask,
            )
            gemm1_activation_second = hl.load_bfloat16_x16_to_float16(
                gemm1_hidden_states,
                gemm1_tile_scale_k.index * 2 + 1,
                extra_mask=gemm1_group_mask,
            )
            gemm1_contribution = hl.zeros(
                [16, gemm1_block_scale_k], dtype=torch.float16
            )
            for gemm1_index in hl.static_range(16):
                gemm1_contribution += (
                    gemm1_weight_first[gemm1_index]
                    * gemm1_activation_first[gemm1_index][None, :]
                )
                gemm1_contribution += (
                    gemm1_weight_second[gemm1_index]
                    * gemm1_activation_second[gemm1_index][None, :]
                )
            gemm1_scale_offset = gemm1_expert * gemm1_twice_intermediate * gemm1_scale_k
            gemm1_scale_offset += _trtllm_scale_offset(
                gemm1_physical_row[:, None],
                gemm1_tile_scale_k.index[None, :],
                gemm1_twice_intermediate,
                gemm1_scale_k,
            )
            gemm1_scale = _e8m0_byte_to_f32(
                hl.load(gemm1_flat_scale, [gemm1_scale_offset])
            )
            gemm1_accumulator += torch.sum(
                gemm1_contribution.to(torch.float32) * gemm1_scale, dim=-1
            )
        gemm1_preactivation = gemm1_accumulator + gemm1_flat_bias[gemm1_expert_row]
        gemm1_pairs = gemm1_preactivation.reshape(2, 8).permute(1, 0)
        gemm1_up_value, gemm1_gate_value = hl.split(gemm1_pairs)
        gemm1_gate_value = torch.clamp(gemm1_gate_value, max=7.0)
        gemm1_up_value = torch.clamp(gemm1_up_value, min=-7.0, max=7.0)
        gemm1_activated = (
            (gemm1_up_value + 1.0)
            * gemm1_gate_value
            * torch.sigmoid(1.702 * gemm1_gate_value)
        )
        gemm1_physical_output[
            gemm1_slot, gemm1_activation_group, gemm1_half, :, gemm1_parity
        ] = gemm1_activated.to(torch.bfloat16)
    for gemm2_tile_physical_row, gemm2_tile_slot in hl.tile(
        [gemm2_hidden, gemm2_top_k], block_size=[gemm2_block_physical_row, 1]
    ):
        gemm2_slot = gemm2_tile_slot.begin
        gemm2_expert = gemm2_topk_ids[0, gemm2_slot]
        gemm2_expert_row = gemm2_expert * gemm2_hidden + gemm2_tile_physical_row.index
        gemm2_accumulator = hl.zeros([gemm2_tile_physical_row], dtype=torch.float32)
        for gemm2_tile_scale_k in hl.tile(
            gemm2_scale_k, block_size=gemm2_block_scale_k
        ):
            gemm2_group_mask = gemm2_tile_scale_k.index < gemm2_scale_k
            gemm2_subgroup = gemm2_expert_row[:, None] * (gemm2_scale_k * 2)
            gemm2_subgroup += gemm2_tile_scale_k.index[None, :] * 2
            gemm2_valid = (
                gemm2_tile_physical_row.index[:, None] < gemm2_hidden
            ) & gemm2_group_mask[None, :]
            gemm2_weight_first = hl.load_float4_e2m1fn_x16_to_float16(
                gemm2_flat_weight, gemm2_subgroup, extra_mask=gemm2_valid
            )
            gemm2_weight_second = hl.load_float4_e2m1fn_x16_to_float16(
                gemm2_flat_weight, gemm2_subgroup + 1, extra_mask=gemm2_valid
            )
            gemm2_activation_values = (
                hl.load(
                    gemm2_physical_activation,
                    [
                        gemm2_slot,
                        gemm2_tile_scale_k.index,
                        slice(None),
                        slice(None),
                        slice(None),
                    ],
                )
                .to(torch.float16)
                .reshape(gemm2_block_scale_k, 32)
            )
            gemm2_activation_lanes = _split_contiguous_lanes(
                gemm2_activation_values, gemm2_block_scale_k, 32
            )
            gemm2_contribution = hl.zeros(
                [gemm2_block_physical_row, gemm2_block_scale_k], dtype=torch.float16
            )
            for gemm2_index in hl.static_range(16):
                gemm2_contribution += (
                    gemm2_weight_first[gemm2_index]
                    * gemm2_activation_lanes[gemm2_index][None, :]
                )
                gemm2_contribution += (
                    gemm2_weight_second[gemm2_index]
                    * gemm2_activation_lanes[gemm2_index + 16][None, :]
                )
            gemm2_scale_offset = (
                gemm2_expert * gemm2_hidden * gemm2_scale_k
                + _trtllm_scale_offset(
                    gemm2_tile_physical_row.index[:, None],
                    gemm2_tile_scale_k.index[None, :],
                    gemm2_hidden,
                    gemm2_scale_k,
                )
            )
            gemm2_scale = _e8m0_byte_to_f32(
                hl.load(gemm2_flat_scale, [gemm2_scale_offset])
            )
            gemm2_accumulator += torch.sum(
                gemm2_contribution.to(torch.float32) * gemm2_scale, dim=-1
            )
        gemm2_result = gemm2_accumulator + gemm2_flat_bias[gemm2_expert_row]
        gemm2_physical_row = gemm2_tile_physical_row.begin + hl.arange(
            gemm2_block_physical_row
        )
        gemm2_lane = gemm2_physical_row & 31
        gemm2_logical_row = (
            gemm2_physical_row - gemm2_lane + (gemm2_lane & 7) * 4 + (gemm2_lane >> 3)
        )
        expert_output[gemm2_slot, gemm2_logical_row] = gemm2_result.to(torch.bfloat16)
    for finalize_tile_n in hl.tile(finalize_output_hidden):
        finalize_values = finalize_expert_output[:, finalize_tile_n].to(torch.float32)
        output[:, finalize_tile_n] = torch.sum(
            finalize_values * finalize_weights[:, None].to(torch.float32),
            dim=0,
            keepdim=True,
        )
    return (output, topk_weights, topk_ids, activation, expert_output)


def _kernel_args(tensors: dict[str, torch.Tensor], shape) -> tuple:
    return (
        tensors["logits"],
        tensors["hidden"],
        tensors["w13"],
        tensors["w13_scale"].view(torch.uint8),
        tensors["w13_bias"],
        tensors["w2"],
        tensors["w2_scale"].view(torch.uint8),
        tensors["w2_bias"],
        shape.output_hidden,
    )


def _validate(actual: tuple, expected: tuple) -> None:
    for name, left, right in zip(OUTPUT_NAMES, actual, expected, strict=True):
        if name == "topk_ids":
            torch.testing.assert_close(left, right)
        else:
            torch.testing.assert_close(left.float(), right.float(), atol=1.0, rtol=0.1)
            error = float((left.float() - right.float()).abs().max().item())
            print(f"correctness {name} max_abs={error:.6f}", flush=True)


def _assert_standalone_exact(
    persistent: tuple[torch.Tensor, ...],
    standalone: tuple[torch.Tensor, ...],
) -> None:
    for persistent_value, standalone_value in zip(
        persistent,
        standalone,
        strict=True,
    ):
        torch.testing.assert_close(
            persistent_value,
            standalone_value,
            atol=0,
            rtol=0,
        )


def _dispatch_cache_signature(call: object) -> tuple[tuple[object, int], ...]:
    """Identify the already-bound launchers used by an AOT kernel."""
    cache = getattr(call, "_dispatch_cache", None)
    if not cache:
        raise RuntimeError("AOT dispatch cache was not populated after launch")
    return tuple((key, id(bound)) for key, bound in cache.items())


def use_cudagraph() -> bool:
    """The timed closures replay pre-captured CUDA graphs."""
    return True


def has_vllm() -> bool:
    """Whether vLLM's production Blackwell GPT-OSS backend is importable."""
    try:
        from flashinfer import trtllm_fp4_block_scale_moe  # noqa: F401
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (  # noqa: F401
            Mxfp4MoeBackend,
        )
    except ImportError:
        return False
    return True


def _require_sm100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("gpt_oss_moe is pretuned only for NVIDIA SM100")


def _standalone_module():
    import sys

    repo_root = str(Path(__file__).resolve().parents[3])
    inserted = repo_root not in sys.path
    if inserted:
        sys.path.insert(0, repo_root)
    try:
        from pretuned_kernels.megakernels.gpt_oss_moe import _standalone
    finally:
        if inserted:
            sys.path.remove(repo_root)

    return _standalone


def _make_problem() -> tuple[dict[str, torch.Tensor], object]:
    _standalone = _standalone_module()

    shape = _standalone.GptOssMoeShape()
    tensors = _standalone._allocate(shape)
    return tensors, shape


def _set_routing_case(
    tensors: dict[str, torch.Tensor],
    selected_ids: tuple[int, ...],
    selected_logits: tuple[float, ...],
) -> None:
    _standalone = _standalone_module()

    _standalone.set_selected_experts(tensors, selected_ids, selected_logits)


def _reference_outputs(tensors: dict[str, torch.Tensor], shape) -> tuple:
    _standalone = _standalone_module()

    reference = _standalone._reference(tensors)
    return (
        reference[4][:, : shape.output_hidden],
        reference[0],
        reference[1],
        reference[2],
        reference[3],
    )


def _make_standalone_call(tensors: dict[str, torch.Tensor], shape) -> tuple:
    """Build the matched, independently tuned four-launch Helion graph."""
    _standalone = _standalone_module()

    return _standalone.build(tensors, shape)


def _make_vllm_call(tensors: dict[str, torch.Tensor], shape) -> tuple:
    """Build vLLM's production FlashInfer TRT-LLM GPT-OSS MoE endpoint."""
    from flashinfer import autotune as flashinfer_autotune
    from flashinfer import trtllm_fp4_block_scale_moe
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
        activation_to_flashinfer_int,
    )

    def per_expert(value: float) -> torch.Tensor:
        return torch.full((shape.experts,), value, device="cuda", dtype=torch.float32)

    gemm1_alpha = per_expert(1.702)
    gemm1_beta = per_expert(1.0)
    gemm1_clamp_limit = per_expert(7.0)
    output = torch.empty((1, shape.hidden), device="cuda", dtype=torch.bfloat16)
    activation_type = activation_to_flashinfer_int(MoEActivation.SWIGLUOAI)

    def launch() -> torch.Tensor:
        trtllm_fp4_block_scale_moe(
            routing_logits=tensors["logits"],
            routing_bias=None,
            hidden_states=tensors["hidden"],
            hidden_states_scale=None,
            gemm1_weights=tensors["w13"].view(torch.uint8),
            gemm1_weights_scale=tensors["w13_scale"],
            gemm1_bias=tensors["w13_bias"],
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            gemm1_clamp_limit=gemm1_clamp_limit,
            gemm2_weights=tensors["w2"].view(torch.uint8),
            gemm2_weights_scale=tensors["w2_scale"],
            gemm2_bias=tensors["w2_bias"],
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=shape.experts,
            top_k=shape.top_k,
            n_group=0,
            topk_group=0,
            intermediate_size=shape.intermediate,
            local_expert_offset=0,
            local_num_experts=shape.experts,
            routed_scaling_factor=None,
            routing_method_type=RoutingMethodType.RenormalizeNaive,
            do_finalize=True,
            activation_type=activation_type,
            enable_pdl=False,
            tune_max_num_tokens=1,
            output=output,
        )
        return output[:, : shape.output_hidden]

    # Production resolves the tactic once while loading the expert layer.  Keep
    # that setup outside both correctness timing and CUDA-graph capture.
    with flashinfer_autotune(True):
        launch()

    def launch_tuned() -> torch.Tensor:
        with flashinfer_autotune(False):
            return launch()

    launch_tuned()
    return launch_tuned, "flashinfer_trtllm"


@torch.inference_mode()
def correctness_check() -> None:
    """Check persistent and separate Helion against production vLLM."""
    _require_sm100()
    if not has_vllm():
        raise RuntimeError(
            "vLLM with FlashInfer is required for the GPT-OSS comparison"
        )
    tensors, shape = _make_problem()
    standalone_call, _standalone_output = _make_standalone_call(tensors, shape)
    vllm_call, _backend = _make_vllm_call(tensors, shape)
    dispatch_signature = None
    for _label, selected_ids in ROUTING_CASES:
        _set_routing_case(tensors, selected_ids, ROUTING_LOGITS)
        expected = _reference_outputs(tensors, shape)
        persistent_output = gpt_oss_moe(*_kernel_args(tensors, shape))
        current_signature = _dispatch_cache_signature(gpt_oss_moe)
        if dispatch_signature is None:
            dispatch_signature = current_signature
        elif current_signature != dispatch_signature:
            raise AssertionError("runtime MoE routing triggered a recompilation")
        standalone_output = standalone_call()
        vllm_output = vllm_call()
        torch.cuda.synchronize()
        _validate(persistent_output, expected)
        _validate(standalone_output, expected)
        _assert_standalone_exact(persistent_output, standalone_output)
        torch.testing.assert_close(
            persistent_output[0].float(),
            vllm_output.float(),
            atol=1.0,
            rtol=0.1,
        )
        torch.testing.assert_close(
            persistent_output[2],
            torch.tensor([selected_ids], device="cuda", dtype=torch.int32),
        )


@torch.inference_mode()
def main(verbose: bool = True) -> dict:
    """Benchmark persistent, separate Helion, and production vLLM with cold L2."""
    _require_sm100()
    if not has_vllm():
        raise RuntimeError(
            "vLLM with FlashInfer is required for the GPT-OSS comparison"
        )

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from _bench import capture_cuda_graph
    from _bench import run_sweep

    tensors, shape = _make_problem()
    standalone_call, _standalone_output = _make_standalone_call(tensors, shape)
    vllm_call, backend = _make_vllm_call(tensors, shape)
    kernel_args = _kernel_args(tensors, shape)
    dispatch_signature = None

    def make_calls(case: tuple[str, tuple[int, ...]]) -> tuple:
        nonlocal dispatch_signature
        label, selected_ids = case
        _set_routing_case(tensors, selected_ids, ROUTING_LOGITS)
        expected = _reference_outputs(tensors, shape)
        persistent_output = gpt_oss_moe(*kernel_args)
        current_signature = _dispatch_cache_signature(gpt_oss_moe)
        if dispatch_signature is None:
            dispatch_signature = current_signature
        elif current_signature != dispatch_signature:
            raise AssertionError("runtime MoE routing triggered a recompilation")
        standalone_output = standalone_call()
        vllm_output = vllm_call()
        torch.cuda.synchronize()
        _validate(persistent_output, expected)
        _validate(standalone_output, expected)
        _assert_standalone_exact(persistent_output, standalone_output)
        torch.testing.assert_close(
            persistent_output[0].float(),
            vllm_output.float(),
            atol=1.0,
            rtol=0.1,
        )
        torch.testing.assert_close(
            persistent_output[2],
            torch.tensor([selected_ids], device="cuda", dtype=torch.int32),
        )
        persistent_graph, _ = capture_cuda_graph(lambda: gpt_oss_moe(*kernel_args))
        standalone_graph, _ = capture_cuda_graph(standalone_call)
        vllm_graph, _ = capture_cuda_graph(vllm_call)
        return (
            persistent_graph.replay,
            [
                ("standalone_helion_pdl", standalone_graph.replay),
                (f"vllm_auto ({backend})", vllm_graph.replay),
            ],
            (
                f"{label:>10s}  {selected_ids!s:>18s}  "
                f"{shape.hidden:>6d}  {shape.intermediate:>12d}"
            ),
        )

    return run_sweep(
        ROUTING_CASES,
        make_calls,
        use_cudagraph=False,
        pre_captured_cudagraph=True,
        thermal_warmup_ms=10_000,
        verbose=verbose,
        shape_header=(
            f"{'routing':>10s}  {'expert_ids':>18s}  "
            f"{'hidden':>6s}  {'intermediate':>12s}"
        ),
    )


if __name__ == "__main__":
    main()
