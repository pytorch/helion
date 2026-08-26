# ruff: noqa: A002, ANN001, ANN201, ANN202
"""Helion reconstruction of the production Nemotron-3 Nano FP8 MoE block.

This probe targets the vLLM production configuration used for
``nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`` on B200 at TP=EP=DP=1 with
``--moe-backend=flashinfer_cutlass``.  It preserves the model dimensions,
tensor dtypes, rounding points, stream split, and the important kernel
fusion boundaries:

Main stream::

    router_gemm_fp32(hidden_states, weight, torch.float32) -> FP32[M, 128]
    # shared stream starts after the router GEMM
    topk_sigmoid(weights, ids, source_rows, logits, ...) -> None
    static_scaled_fp8_quant(out, hidden_states, a1_scale) -> None
    fused_build_expert_maps_sort_first_token(...) -> None
    expand_input_rows(...) -> None
    routed_gemm1(...) -> BF16[M * 6, 1920]
    relu2_static_fp8_quant(...) -> FP8[M * 6, 1920]
    routed_gemm2_fused_finalize(...) -> BF16[M, 2688]

Auxiliary shared-expert stream::

    static_scaled_fp8_quant -> scaled_fp8_mm -> relu_squared
      -> static_scaled_fp8_quant -> scaled_fp8_mm

The main stream then waits for the shared branch and executes one compiled
pointwise merge for ``shared + bf16(routed * 2.5)``.  FlashInfer pads the
routed intermediate dimension from the checkpoint width 1856 to 1920 for its
non-gated FP8 kernels; the padded weights are zero, as they are in vLLM.

The routed path mirrors FlashInfer's physical kernels rather than treating
``cutlass_fused_moe`` as an indivisible Python call.  In particular, ReLU^2
and the second static FP8 quantization are one kernel, while GEMM2 folds the
router-weighted finalize into its epilogue.  That epilogue uses BF16 atomic
adds in production, so the final routed output is intentionally allowed the
same run-to-run reduction-order variation.

FlashInfer's ``computeStridesTmaWarpSpecializedKernel`` writes raw CUTLASS
pointer/shape descriptors, which do not have a tensor ABI that Helion can
reproduce.  The probe preserves that dependency by making both grouped GEMMs
consume ``expert_first_token_offset`` directly.  The production
``cudaMemsetAsync`` boundary immediately before fused GEMM2 is represented by
``routed_output.zero_()``.

Only the small-token production path is represented (M <= 16): this is the
range where vLLM selects the low-latency BF16 router GEMM and where the shared
expert runs on its auxiliary CUDA stream.  The executable refuses to touch a
GPU unless ``probes.common.require_idle_visible_gpu`` confirms it is idle.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import torch

from probes.common import benchmark_interleaved
from probes.common import capture
from probes.common import require_idle_visible_gpu
from probes.common import visible_gpu_pids

import helion
import helion.language as hl

FP8_MAX = 448.0
MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
VLLM_REVISION = "b369f10d5c5dc3af29e5a2f9213a462b8faadae9"
FLASHINFER_VERSION = "0.6.16.post3"
DEFAULT_CONFIG_PATH = str(Path(__file__).with_name("nemotron3_nano_moe_b200.json"))


@dataclass(frozen=True)
class Nemotron3NanoMoEShape:
    """Production dimensions for one Nemotron-3 Nano MoE decoder block."""

    tokens: int = 1
    hidden: int = 2688
    logical_intermediate: int = 1856
    routed_intermediate: int = 1920
    shared_intermediate: int = 3712
    num_experts: int = 128
    top_k: int = 6
    routed_scaling_factor: float = 2.5

    def validate(self) -> None:
        if not 1 <= self.tokens <= 16:
            raise ValueError(
                "this probe models vLLM's small-token router path: 1 <= tokens <= 16"
            )
        expected = math.ceil(self.logical_intermediate / 128) * 128
        if self.routed_intermediate != expected:
            raise ValueError(
                "FlashInfer's non-gated FP8 path requires the routed intermediate "
                f"width {expected}, got {self.routed_intermediate}"
            )


PRODUCTION_KERNEL_GRAPH = (
    {
        "stream": "main",
        "kernel": "router_gemm_fp32",
        "signature": "(BF16[M,H], BF16[E,H], torch.float32) -> FP32[M,E]",
    },
    {
        "stream": "aux",
        "kernel": "shared static_scaled_fp8_quant",
        "signature": "(FP8[M,H], BF16[M,H], FP32[], int[]?=None) -> None",
    },
    {
        "stream": "aux",
        "kernel": "shared up scaled_fp8_mm",
        "signature": (
            "(FP8[M,H], FP8[H,S], FP32[], FP32[], torch.bfloat16, None) -> BF16[M,S]"
        ),
    },
    {
        "stream": "aux",
        "kernel": "relu_squared",
        "signature": "(BF16[M,S], BF16[M,S]) -> None",
    },
    {
        "stream": "aux",
        "kernel": "shared static_scaled_fp8_quant",
        "signature": "(FP8[M,S], BF16[M,S], FP32[], int[]?=None) -> None",
    },
    {
        "stream": "aux",
        "kernel": "shared down scaled_fp8_mm",
        "signature": (
            "(FP8[M,S], FP8[S,H], FP32[], FP32[], torch.bfloat16, None) -> BF16[M,H]"
        ),
    },
    {
        "stream": "main",
        "kernel": "topk_sigmoid",
        "signature": (
            "(FP32[M,6], I32[M,6], I32[M,6], FP32[M,128], bool, "
            "FP32[128], float, BOOL[M]?) -> None"
        ),
    },
    {
        "stream": "main",
        "kernel": "routed static_scaled_fp8_quant",
        "signature": "(FP8[M,H], BF16[M,H], FP32[], int[]?=None) -> None",
    },
    {
        "stream": "main",
        "kernel": "fused_build_expert_maps_sort_first_token",
        "signature": "(I32[M,6], I32[6M], I32[6M], I64[129]) -> None",
    },
    {
        "stream": "main",
        "kernel": "expand_input_rows",
        "signature": (
            "(FP8[M,H], FP8[6M,H], FP32[M,6], FP32[6M], I32[6M], I64[129]) -> None"
        ),
    },
    {
        "stream": "main",
        "kernel": "computeStridesTmaWarpSpecializedKernel",
        "signature": "I64[129] + tensor addresses -> CUTLASS descriptor workspace",
        "representation": "expert offsets are consumed directly by the Helion GEMMs",
    },
    {
        "stream": "main",
        "kernel": "routed_gemm1",
        "signature": (
            "(BF16[6M,1920], FP8[6M,H], FP8[128,1920,H], FP32[128], I64[129]) -> None"
        ),
    },
    {
        "stream": "main",
        "kernel": "relu2_static_fp8_quant",
        "signature": ("(FP8[6M,1920], BF16[6M,1920], FP32[], None, I64[129]) -> None"),
    },
    {
        "stream": "main",
        "kernel": "cudaMemsetAsync",
        "signature": "(BF16[M,H], 0) -> None",
    },
    {
        "stream": "main",
        "kernel": "routed_gemm2_fused_finalize",
        "signature": (
            "(BF16[M,H], FP8[6M,1920], FP8[128,H,1920], FP32[128], "
            "I64[129], I32[6M], FP32[6M]) -> None"
        ),
    },
    {
        "stream": "main",
        "kernel": "scale_routed_and_add_shared",
        "signature": "(BF16[M,H], BF16[M,H], float) -> BF16[M,H]",
    },
)

PRODUCTION_SOURCE = {
    "model": MODEL_ID,
    "vllm_revision": VLLM_REVISION,
    "flashinfer_version": FLASHINFER_VERSION,
    "moe_backend": "flashinfer_cutlass",
    "parallelism": "TP=EP=DP=1",
    "activation": "relu2_no_mul",
    "quantization": "ModelOpt static per-tensor FP8 W8A8",
    "modelopt_version": "0.29.0",
    "scope": "decode path with 1 <= M <= 16",
}


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def router_gemm_fp32(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """BF16 x BF16 router GEMM with FP32 accumulation and FP32 output."""
    num_tokens, hidden_size = hidden_states.size()
    num_experts, weight_hidden = weight.size()
    assert hidden_size == weight_hidden
    assert output_dtype == torch.float32
    output = torch.empty(
        (num_tokens, num_experts), dtype=output_dtype, device=hidden_states.device
    )
    for tile_m, tile_n in hl.tile([num_tokens, num_experts], block_size=[1, None]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(hidden_size):
            acc = torch.addmm(
                acc,
                hidden_states[tile_m, tile_k],
                weight[tile_n, tile_k].T,
            )
        output[tile_m, tile_n] = acc
    return output


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor | None,
    routed_scaling_factor: float,
    is_padding: torch.Tensor | None,
) -> None:
    """Match vLLM's mutable-output sigmoid top-k routing kernel."""
    num_tokens, num_experts = gating_output.size()
    top_k = topk_weights.size(1)
    hl.specialize(num_experts)
    top_k = hl.specialize(top_k)
    for tile_m in hl.tile(num_tokens, block_size=1):
        logits = gating_output[tile_m, :].to(torch.float32)
        scores = torch.sigmoid(logits)
        scores = torch.where(
            torch.isnan(scores) | torch.isinf(scores),
            torch.zeros_like(scores),
            scores,
        )
        selection_scores = scores
        if e_score_correction_bias is not None:
            selection_scores = selection_scores + e_score_correction_bias[None, :]
        _, selected = torch.topk(selection_scores, top_k, dim=-1, largest=True)
        selected_scores = torch.gather(scores, 1, selected)
        if renormalize:
            denominator = torch.sum(selected_scores, dim=-1, keepdim=True)
            denominator = torch.where(
                denominator > 0.0,
                denominator,
                torch.ones_like(denominator),
            )
            selected_scores = selected_scores / denominator
        selected_scores = selected_scores * routed_scaling_factor
        selected_i32 = selected.to(torch.int32)
        if is_padding is not None:
            padding = is_padding[tile_m].to(torch.bool)[:, None]
            selected_i32 = torch.where(
                padding,
                torch.full_like(selected_i32, -1),
                selected_i32,
            )
        topk_weights[tile_m, :] = selected_scores
        topk_indices[tile_m, :] = selected_i32
        slots = hl.arange(top_k)[None, :]
        token_expert_indices[tile_m, :] = (
            slots * num_tokens + tile_m.index[:, None]
        ).to(torch.int32)


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def static_scaled_fp8_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    group_shape: tuple[int, int] | None = None,
) -> None:
    """Match vLLM's static per-tensor ``scaled_fp8_quant`` CUDA kernel."""
    assert group_shape is None
    num_tokens, hidden_size = input.size()
    hl.specialize(num_tokens)
    hl.specialize(hidden_size)
    assert out.shape == input.shape
    assert out.dtype == torch.float8_e4m3fn
    for tile_m, tile_n in hl.tile([num_tokens, hidden_size]):
        inv_scale = 1.0 / hl.load(scale, [])
        values = input[tile_m, tile_n].to(torch.float32) * inv_scale
        out[tile_m, tile_n] = values.clamp(-FP8_MAX, FP8_MAX).to(out.dtype)


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def fused_build_expert_maps_sort_first_token(
    token_selected_experts: torch.Tensor,
    permuted_row_to_unpermuted_row: torch.Tensor,
    unpermuted_row_to_permuted_row: torch.Tensor,
    expert_first_token_offset: torch.Tensor,
) -> None:
    """Small-token FlashInfer fused prologue, including slot-major row IDs."""
    num_tokens, top_k = token_selected_experts.size()
    assignments = num_tokens * top_k
    num_experts = expert_first_token_offset.numel() - 1
    hl.specialize(num_tokens)
    hl.specialize(top_k)
    hl.specialize(assignments)
    hl.specialize(num_experts)
    sort_width = helion.next_power_of_2(assignments)
    flat_selected_experts = token_selected_experts.view(-1)

    for _tile_one in hl.tile(1, block_size=1):
        assignment_items = hl.arange(sort_width)
        valid = assignment_items < assignments
        expert_ids = hl.load(
            flat_selected_experts,
            [assignment_items],
            extra_mask=valid,
        )
        expert_ids = torch.where(valid, expert_ids, num_experts + 1)
        sorted_experts, sorted_items = torch.sort(expert_ids, dim=-1, descending=False)
        sorted_source_rows = (sorted_items % top_k) * num_tokens + sorted_items // top_k
        hl.store(
            permuted_row_to_unpermuted_row,
            [assignment_items],
            sorted_source_rows.to(torch.int32),
            extra_mask=valid,
        )
        hl.store(
            unpermuted_row_to_permuted_row,
            [sorted_source_rows],
            assignment_items.to(torch.int32),
            extra_mask=valid,
        )
        expert_boundaries = hl.arange(num_experts + 1)
        offsets = torch.sum(
            sorted_experts[None, :] < expert_boundaries[:, None], dim=-1
        )
        expert_first_token_offset[:] = offsets.to(torch.int64)


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def expand_input_rows(
    unpermuted_input: torch.Tensor,
    permuted_output: torch.Tensor,
    unpermuted_scales: torch.Tensor,
    permuted_scales: torch.Tensor,
    permuted_row_to_unpermuted_row: torch.Tensor,
    expert_first_token_offset: torch.Tensor,
) -> None:
    """Duplicate and expert-sort FP8 rows and their FP32 routing scales."""
    num_tokens, hidden_size = unpermuted_input.size()
    _, top_k = unpermuted_scales.size()
    assignments = num_tokens * top_k
    num_experts = expert_first_token_offset.numel() - 1
    hl.specialize(num_tokens)
    hl.specialize(top_k)
    hl.specialize(num_experts)
    flat_scales = unpermuted_scales.view(-1)
    for tile_a in hl.tile(assignments, block_size=1):
        num_valid_tokens = hl.load(expert_first_token_offset, [num_experts])
        valid_rows = tile_a.index < num_valid_tokens
        source_row = hl.load(
            permuted_row_to_unpermuted_row,
            [tile_a],
            extra_mask=valid_rows,
        )
        source_token = source_row % num_tokens
        source_slot = source_row // num_tokens
        scales = hl.load(
            flat_scales,
            [source_token * top_k + source_slot],
            extra_mask=valid_rows,
        )
        hl.store(permuted_scales, [tile_a], scales, extra_mask=valid_rows)
        for tile_n in hl.tile(hidden_size):
            values = hl.load(
                unpermuted_input,
                [source_token, tile_n],
                extra_mask=valid_rows[:, None],
            )
            hl.store(
                permuted_output,
                [tile_a, tile_n],
                values,
                extra_mask=valid_rows[:, None],
            )


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    config=helion.Config(block_sizes=[64, 64]),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def routed_gemm1(
    output: torch.Tensor,
    permuted_input: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc1_dequant_scales: torch.Tensor,
    expert_first_token_offset: torch.Tensor,
) -> None:
    """FP8 grouped GEMM1 with per-expert alpha and a BF16 output boundary."""
    assignments, hidden_size = permuted_input.size()
    num_experts, intermediate_size, weight_hidden = fc1_expert_weights.size()
    assert hidden_size == weight_hidden
    assert output.shape == (assignments, intermediate_size)
    assert expert_first_token_offset.numel() == num_experts + 1
    hl.specialize(num_experts)

    for tile_a, tile_n in hl.tile(
        [assignments, intermediate_size], block_size=[1, None]
    ):
        expert_ids = hl.arange(num_experts)
        expert_ends = expert_first_token_offset[expert_ids + 1]
        selected_experts = torch.sum(
            tile_a.index[:, None] >= expert_ends[None, :], dim=-1
        ).to(torch.int32)
        acc = hl.zeros([tile_a, 1, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(hidden_size):
            lhs = permuted_input[tile_a, tile_k].unsqueeze(1)
            rhs = fc1_expert_weights[
                selected_experts[:, None, None],
                tile_n.index[None, None, :],
                tile_k.index[None, :, None],
            ]
            acc = hl.dot(lhs, rhs, acc=acc)
        alpha = fc1_dequant_scales[selected_experts].to(torch.float32)
        output[tile_a, tile_n] = (acc.squeeze(1) * alpha[:, None]).to(output.dtype)


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def relu2_static_fp8_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    quant_scale: torch.Tensor,
    bias: torch.Tensor | None,
    expert_first_token_offset: torch.Tensor,
) -> None:
    """FlashInfer's separate BF16 ReLU^2 plus static FP8 quantization kernel."""
    rows, width = input.size()
    hl.specialize(rows)
    hl.specialize(width)
    assert out.shape == input.shape
    assert out.dtype == torch.float8_e4m3fn
    num_experts = expert_first_token_offset.numel() - 1
    assert bias is None
    hl.specialize(num_experts)
    for tile_m, tile_n in hl.tile([rows, width]):
        num_valid_tokens = hl.load(expert_first_token_offset, [num_experts])
        valid_rows = tile_m.index < num_valid_tokens
        inverse = hl.load(quant_scale, [])
        values = hl.load(
            input,
            [tile_m, tile_n],
            extra_mask=valid_rows[:, None],
        ).to(torch.float32)
        activated = torch.maximum(values, torch.zeros_like(values))
        quantized = activated * activated * inverse
        hl.store(
            out,
            [tile_m, tile_n],
            quantized.clamp(0.0, FP8_MAX).to(out.dtype),
            extra_mask=valid_rows[:, None],
        )


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    config=helion.Config(block_sizes=[64, 64]),
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
    backend="triton",
)
def routed_gemm2_fused_finalize(
    output: torch.Tensor,
    activation: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    fc2_dequant_scales: torch.Tensor,
    expert_first_token_offset: torch.Tensor,
    permuted_row_to_unpermuted_row: torch.Tensor,
    permuted_final_scales: torch.Tensor,
) -> None:
    """FP8 GEMM2 with the routing-weighted BF16 atomic finalize epilogue."""
    assignments, intermediate_size = activation.size()
    num_experts, hidden_size, weight_intermediate = fc2_expert_weights.size()
    num_tokens = output.size(0)
    assert intermediate_size == weight_intermediate
    assert output.shape == (num_tokens, hidden_size)
    assert expert_first_token_offset.numel() == num_experts + 1
    hl.specialize(num_experts)
    hl.specialize(num_tokens)

    for tile_a, tile_n in hl.tile([assignments, hidden_size], block_size=[1, None]):
        expert_ids = hl.arange(num_experts)
        expert_ends = expert_first_token_offset[expert_ids + 1]
        source_row = permuted_row_to_unpermuted_row[tile_a]
        source_token = source_row % num_tokens
        selected_experts = torch.sum(
            tile_a.index[:, None] >= expert_ends[None, :], dim=-1
        ).to(torch.int32)
        acc = hl.zeros([tile_a, 1, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(intermediate_size):
            lhs = activation[tile_a, tile_k].unsqueeze(1)
            rhs = fc2_expert_weights[
                selected_experts[:, None, None],
                tile_n.index[None, None, :],
                tile_k.index[None, :, None],
            ]
            acc = hl.dot(lhs, rhs, acc=acc)
        dequant = fc2_dequant_scales[selected_experts].to(torch.float32)
        route = permuted_final_scales[tile_a].to(torch.float32)
        contribution = (acc.squeeze(1) * dequant[:, None] * route[:, None]).to(
            output.dtype
        )
        hl.atomic_add(output, [source_token, tile_n], contribution)


@helion.kernel(
    static_shapes=True,
    autotune_effort="full",
    config=helion.Config(block_sizes=[64, 64]),
    backend="triton",
)
def fp8_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Per-tensor FP8 scaled MM used by both shared-expert projections."""
    m, k = A.size()
    weight_k, n = B.size()
    assert k == weight_k
    assert out_dtype == torch.bfloat16
    output = torch.empty((m, n), dtype=out_dtype, device=A.device)
    for tile_m, tile_n in hl.tile([m, n], block_size=[1, None]):
        scale = hl.load(scale_a, []) * hl.load(scale_b, [])
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = hl.dot(A[tile_m, tile_k], B[tile_k, tile_n], acc=acc)
        values = acc * scale
        if bias is not None:
            values = values + bias[tile_n].to(torch.float32)[None, :]
        output[tile_m, tile_n] = values.to(output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def relu_squared(out: torch.Tensor, input: torch.Tensor) -> None:
    """Match ``torch.ops._C.relu_squared(out, input)`` for BF16 tensors."""
    rows, width = input.size()
    assert out.shape == input.shape
    for tile_m, tile_n in hl.tile([rows, width]):
        values = input[tile_m, tile_n].to(torch.float32)
        relu = torch.maximum(values, torch.zeros_like(values))
        out[tile_m, tile_n] = (relu * relu).to(out.dtype)


@helion.kernel(static_shapes=True, autotune_effort="full", backend="triton")
def scale_routed_and_add_shared(
    shared_output: torch.Tensor,
    routed_output: torch.Tensor,
    routed_scaling_factor: float,
) -> torch.Tensor:
    """Compiled vLLM post-op, retaining the BF16 rounding after routed scale."""
    num_tokens, hidden_size = routed_output.size()
    output = torch.empty_like(routed_output)
    for tile_m, tile_n in hl.tile([num_tokens, hidden_size]):
        routed = (
            routed_output[tile_m, tile_n].to(torch.float32) * routed_scaling_factor
        ).to(routed_output.dtype)
        output[tile_m, tile_n] = shared_output[tile_m, tile_n] + routed
    return output


def _static_fp8_reference(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (
        (input.float() / scale.float()).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    )


def _topk_reference(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = torch.sigmoid(logits.float())
    scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
    _, ids = torch.topk(scores + correction_bias, top_k, dim=-1)
    weights = torch.gather(scores, 1, ids)
    denominator = weights.sum(dim=-1, keepdim=True)
    weights = weights / torch.where(
        denominator > 0.0, denominator, torch.ones_like(denominator)
    )
    num_tokens = logits.size(0)
    source_rows = (
        torch.arange(top_k, device=logits.device, dtype=torch.int32)[None, :]
        * num_tokens
        + torch.arange(num_tokens, device=logits.device, dtype=torch.int32)[:, None]
    )
    return weights, ids.to(torch.int32), source_rows


def _prologue_reference(
    topk_ids: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, top_k = topk_ids.shape
    assignment_items = torch.arange(
        num_tokens * top_k, device=topk_ids.device, dtype=torch.int64
    )
    expert_ids = topk_ids.flatten()
    sorted_experts, order = torch.sort(expert_ids, stable=True)
    permuted_rows = (order % top_k) * num_tokens + order // top_k
    inverse = torch.empty_like(order)
    inverse[permuted_rows] = assignment_items
    boundaries = torch.arange(
        num_experts + 1, device=topk_ids.device, dtype=sorted_experts.dtype
    )
    offsets = (sorted_experts[None, :] < boundaries[:, None]).sum(dim=-1)
    return (
        permuted_rows.to(torch.int32),
        inverse.to(torch.int32),
        offsets.to(torch.int64),
    )


def _routed_reference(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    g1_alphas: torch.Tensor,
    a2_gscale: torch.Tensor,
    g2_alphas: torch.Tensor,
    a1_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_q = _static_fp8_reference(hidden_states, a1_scale)
    num_tokens, top_k = topk_ids.shape
    intermediate = w1.size(1)
    gemm1 = torch.empty(
        (num_tokens, top_k, intermediate),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    activation_q = torch.empty_like(gemm1, dtype=torch.float8_e4m3fn)
    output = torch.zeros_like(hidden_states)
    for token in range(num_tokens):
        for slot in range(top_k):
            expert = int(topk_ids[token, slot].item())
            fc1 = (
                torch.matmul(input_q[token].float(), w1[expert].float().T)
                * g1_alphas[expert]
            )
            fc1 = fc1.to(torch.bfloat16)
            gemm1[token, slot] = fc1
            activated = torch.relu(fc1.float()).square()
            q2 = (
                (activated * a2_gscale.float())
                .clamp(0.0, FP8_MAX)
                .to(torch.float8_e4m3fn)
            )
            activation_q[token, slot] = q2
            fc2 = torch.matmul(q2.float(), w2[expert].float().T)
            contribution = (fc2 * g2_alphas[expert] * topk_weights[token, slot]).to(
                torch.bfloat16
            )
            output[token] = (output[token] + contribution).to(torch.bfloat16)
    return input_q, gemm1, activation_q, output


def _shared_reference(
    hidden_states: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    up_input_scale: torch.Tensor,
    up_weight_scale: torch.Tensor,
    down_input_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_q = _static_fp8_reference(hidden_states, up_input_scale)
    up = (
        torch.matmul(input_q.float(), up_weight.float())
        * up_input_scale.float()
        * up_weight_scale.float()
    ).to(torch.bfloat16)
    activation = torch.relu(up.float()).square().to(torch.bfloat16)
    activation_q = _static_fp8_reference(activation, down_input_scale)
    down = (
        torch.matmul(activation_q.float(), down_weight.float())
        * down_input_scale.float()
        * down_weight_scale.float()
    ).to(torch.bfloat16)
    return input_q, up, activation, activation_q, down


def _make_fp8(shape: tuple[int, ...], scale: float = 1.0) -> torch.Tensor:
    values = torch.randint(-3, 4, shape, device="cuda", dtype=torch.int8)
    return (values.to(torch.bfloat16) * scale).to(torch.float8_e4m3fn)


def allocate(shape: Nemotron3NanoMoEShape) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    device = "cuda"
    assignments = shape.tokens * shape.top_k
    hidden_states = (
        torch.randn((shape.tokens, shape.hidden), device=device, dtype=torch.bfloat16)
        * 0.2
    )
    tensors = {
        "hidden_states": hidden_states,
        "router_weight": torch.randn(
            (shape.num_experts, shape.hidden),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.02,
        "correction_bias": torch.randn(
            (shape.num_experts,), device=device, dtype=torch.float32
        )
        * 0.01,
        "routed_input_scale": torch.tensor(0.01, device=device),
        "routed_w1_scale": torch.rand(
            (shape.num_experts,), device=device, dtype=torch.float32
        )
        * 0.002
        + 0.004,
        "routed_w2_input_scale": torch.tensor(0.002, device=device),
        "routed_w2_scale": torch.rand(
            (shape.num_experts,), device=device, dtype=torch.float32
        )
        * 0.002
        + 0.004,
        "routed_w1": _make_fp8(
            (shape.num_experts, shape.routed_intermediate, shape.hidden)
        ),
        "routed_w2": _make_fp8(
            (shape.num_experts, shape.hidden, shape.routed_intermediate)
        ),
        "shared_up_input_scale": torch.tensor(0.01, device=device),
        "shared_up_weight_scale": torch.tensor(0.005, device=device),
        "shared_down_input_scale": torch.tensor(0.002, device=device),
        "shared_down_weight_scale": torch.tensor(0.005, device=device),
        "shared_up_weight": _make_fp8((shape.hidden, shape.shared_intermediate)),
        "shared_down_weight": _make_fp8((shape.shared_intermediate, shape.hidden)),
        "router_logits": torch.empty(
            (shape.tokens, shape.num_experts), device=device, dtype=torch.float32
        ),
        "topk_weights": torch.empty(
            (shape.tokens, shape.top_k), device=device, dtype=torch.float32
        ),
        "topk_ids": torch.empty(
            (shape.tokens, shape.top_k), device=device, dtype=torch.int32
        ),
        "token_expert_indices": torch.empty(
            (shape.tokens, shape.top_k), device=device, dtype=torch.int32
        ),
        "routed_input_q": torch.empty(
            (shape.tokens, shape.hidden),
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        "permuted_row_to_unpermuted_row": torch.empty(
            (assignments,), device=device, dtype=torch.int32
        ),
        "unpermuted_row_to_permuted_row": torch.empty(
            (assignments,), device=device, dtype=torch.int32
        ),
        "expert_first_token_offset": torch.empty(
            (shape.num_experts + 1,), device=device, dtype=torch.int64
        ),
        "permuted_input": torch.empty(
            (assignments, shape.hidden),
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        "permuted_scales": torch.empty(
            (assignments,), device=device, dtype=torch.float32
        ),
        "routed_gemm1": torch.empty(
            (assignments, shape.routed_intermediate),
            device=device,
            dtype=torch.bfloat16,
        ),
        "routed_activation_q": torch.empty(
            (assignments, shape.routed_intermediate),
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        "routed_output": torch.zeros_like(hidden_states),
        "shared_input_q": torch.empty(
            (shape.tokens, shape.hidden),
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        "shared_up": torch.empty(
            (shape.tokens, shape.shared_intermediate),
            device=device,
            dtype=torch.bfloat16,
        ),
        "shared_activation": torch.empty(
            (shape.tokens, shape.shared_intermediate),
            device=device,
            dtype=torch.bfloat16,
        ),
        "shared_activation_q": torch.empty(
            (shape.tokens, shape.shared_intermediate),
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        "shared_output": torch.empty_like(hidden_states),
    }
    tensors["routed_w1"][:, shape.logical_intermediate :, :] = 0
    tensors["routed_w2"][:, :, shape.logical_intermediate :] = 0
    tensors["g1_alphas"] = tensors["routed_w1_scale"] * tensors["routed_input_scale"]
    tensors["a2_gscale"] = tensors["routed_w2_input_scale"].reciprocal()
    tensors["g2_alphas"] = tensors["routed_w2_scale"] * tensors["routed_w2_input_scale"]
    return tensors


def initialize_autotune_inputs(
    shape: Nemotron3NanoMoEShape,
    tensors: dict[str, torch.Tensor],
) -> None:
    """Populate every intermediate before tuning kernels in isolation."""
    t = tensors
    router = torch.matmul(t["hidden_states"].float(), t["router_weight"].float().T)
    weights, ids, source_rows = _topk_reference(
        router, t["correction_bias"], shape.top_k
    )
    permuted_rows, inverse, offsets = _prologue_reference(ids, shape.num_experts)
    input_q, gemm1, activation_q, routed = _routed_reference(
        t["hidden_states"],
        weights,
        ids,
        t["routed_w1"],
        t["routed_w2"],
        t["g1_alphas"],
        t["a2_gscale"],
        t["g2_alphas"],
        t["routed_input_scale"],
    )
    shared_q, shared_up, shared_activation, shared_activation_q, shared_output = (
        _shared_reference(
            t["hidden_states"],
            t["shared_up_weight"],
            t["shared_down_weight"],
            t["shared_up_input_scale"],
            t["shared_up_weight_scale"],
            t["shared_down_input_scale"],
            t["shared_down_weight_scale"],
        )
    )

    t["router_logits"].copy_(router)
    t["topk_weights"].copy_(weights)
    t["topk_ids"].copy_(ids)
    t["token_expert_indices"].copy_(source_rows)
    t["routed_input_q"].copy_(input_q)
    t["permuted_row_to_unpermuted_row"].copy_(permuted_rows)
    t["unpermuted_row_to_permuted_row"].copy_(inverse)
    t["expert_first_token_offset"].copy_(offsets)
    for permuted_row in range(shape.tokens * shape.top_k):
        source_row = int(permuted_rows[permuted_row].item())
        token = source_row % shape.tokens
        slot = source_row // shape.tokens
        t["permuted_input"][permuted_row].copy_(input_q[token])
        t["permuted_scales"][permuted_row].copy_(weights[token, slot])
        t["routed_gemm1"][permuted_row].copy_(gemm1[token, slot])
        t["routed_activation_q"][permuted_row].copy_(activation_q[token, slot])
    t["routed_output"].copy_(torch.zeros_like(routed))
    t["shared_input_q"].copy_(shared_q)
    t["shared_up"].copy_(shared_up)
    t["shared_activation"].copy_(shared_activation)
    t["shared_activation_q"].copy_(shared_activation_q)
    t["shared_output"].copy_(shared_output)


def compile_default(kernel, kernel_args):
    bound = kernel.bind(kernel_args)
    if kernel.configs:
        config = helion.Config.from_dict(dict(kernel.configs[0]))
        bound.config_spec.normalize(config.config)
    else:
        config = bound.config_spec.default_config()
    return config, bound.compile_config(config)


def tune_kernel(name, kernel, kernel_args, configs, config_path, effort):
    print(f"autotune_start {name}", flush=True)
    started = time.perf_counter()
    kernel.settings.autotune_effort = effort
    bound = kernel.bind(kernel_args)
    config = bound.autotune(kernel_args, force=True)
    elapsed = time.perf_counter() - started
    configs[name] = dict(config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(configs, indent=2, sort_keys=True) + "\n")
    print(
        "autotune_result",
        json.dumps(
            {"name": name, "seconds": elapsed, "config": dict(config)},
            sort_keys=True,
        ),
        flush=True,
    )
    return config, bound.compile_config(config)


def compile_config(kernel, kernel_args, config_dict):
    bound = kernel.bind(kernel_args)
    config = helion.Config.from_dict(config_dict)
    bound.config_spec.normalize(config.config)
    return config, bound.compile_config(config)


class CompiledMoE:
    def __init__(self, shape, tensors, kernels) -> None:
        self.shape = shape
        self.tensors = tensors
        self.kernels = kernels
        self.shared_stream = torch.cuda.Stream()
        self.shared_input_ready = torch.cuda.Event()
        self.shared_output_ready = torch.cuda.Event()
        self.last_outputs: dict[str, torch.Tensor] = {}
        self.last_shared_up: torch.Tensor | None = None

    def __call__(self, overlap_shared: bool = True) -> torch.Tensor:
        t = self.tensors
        k = self.kernels
        router_logits = k["router"](
            t["hidden_states"], t["router_weight"], torch.float32
        )
        shared_output: torch.Tensor | None = None

        if overlap_shared:
            self.shared_input_ready.record(torch.cuda.current_stream())
            with torch.cuda.stream(self.shared_stream):
                self.shared_stream.wait_event(self.shared_input_ready)
                shared_output = self._run_shared()
                self.shared_output_ready.record(self.shared_stream)

        k["topk"](
            t["topk_weights"],
            t["topk_ids"],
            t["token_expert_indices"],
            router_logits,
            True,
            t["correction_bias"],
            1.0,
            None,
        )
        k["routed_quant"](
            t["routed_input_q"],
            t["hidden_states"],
            t["routed_input_scale"],
            None,
        )
        k["prologue"](
            t["topk_ids"],
            t["permuted_row_to_unpermuted_row"],
            t["unpermuted_row_to_permuted_row"],
            t["expert_first_token_offset"],
        )
        k["expand"](
            t["routed_input_q"],
            t["permuted_input"],
            t["topk_weights"],
            t["permuted_scales"],
            t["permuted_row_to_unpermuted_row"],
            t["expert_first_token_offset"],
        )
        k["routed_gemm1"](
            t["routed_gemm1"],
            t["permuted_input"],
            t["routed_w1"],
            t["g1_alphas"],
            t["expert_first_token_offset"],
        )
        k["routed_activation"](
            t["routed_activation_q"],
            t["routed_gemm1"],
            t["a2_gscale"],
            None,
            t["expert_first_token_offset"],
        )
        t["routed_output"].zero_()
        k["routed_gemm2"](
            t["routed_output"],
            t["routed_activation_q"],
            t["routed_w2"],
            t["g2_alphas"],
            t["expert_first_token_offset"],
            t["permuted_row_to_unpermuted_row"],
            t["permuted_scales"],
        )

        if overlap_shared:
            self.shared_output_ready.wait(torch.cuda.current_stream())
        else:
            shared_output = self._run_shared()
        assert shared_output is not None
        assert self.last_shared_up is not None
        output = k["merge"](
            shared_output,
            t["routed_output"],
            self.shape.routed_scaling_factor,
        )
        self.last_outputs = {
            "router_logits": router_logits,
            "shared_up": self.last_shared_up,
            "shared_output": shared_output,
            "output": output,
        }
        return output

    def _run_shared(self):
        t = self.tensors
        k = self.kernels
        k["shared_input_quant"](
            t["shared_input_q"],
            t["hidden_states"],
            t["shared_up_input_scale"],
            None,
        )
        shared_up = k["shared_up"](
            t["shared_input_q"],
            t["shared_up_weight"],
            t["shared_up_input_scale"],
            t["shared_up_weight_scale"],
            torch.bfloat16,
            None,
        )
        k["shared_activation"](t["shared_activation"], shared_up)
        k["shared_activation_quant"](
            t["shared_activation_q"],
            t["shared_activation"],
            t["shared_down_input_scale"],
            None,
        )
        shared_output = k["shared_down"](
            t["shared_activation_q"],
            t["shared_down_weight"],
            t["shared_down_input_scale"],
            t["shared_down_weight_scale"],
            torch.bfloat16,
            None,
        )
        self.last_shared_up = shared_up
        return shared_output


def build_moe(args, shape, tensors):
    config_path = Path(args.config)
    configs = json.loads(config_path.read_text()) if config_path.exists() else {}
    tune_set = set(args.tune or [])
    tune_all = "all" in tune_set
    selected_configs = {}

    def build(name, kernel, kernel_args):
        qualified = f"nemotron3_nano_moe_m{shape.tokens}_{name}"
        if tune_all or name in tune_set or qualified in tune_set:
            config, compiled = tune_kernel(
                qualified,
                kernel,
                kernel_args,
                configs,
                config_path,
                args.tune_effort,
            )
        elif qualified in configs:
            config, compiled = compile_config(kernel, kernel_args, configs[qualified])
        else:
            config, compiled = compile_default(kernel, kernel_args)
        selected_configs[qualified] = dict(config)
        return compiled

    t = tensors
    kernels = {
        "router": build(
            "router_gemm_fp32",
            router_gemm_fp32,
            (t["hidden_states"], t["router_weight"], torch.float32),
        ),
        "topk": build(
            "topk_sigmoid",
            topk_sigmoid,
            (
                t["topk_weights"],
                t["topk_ids"],
                t["token_expert_indices"],
                t["router_logits"],
                True,
                t["correction_bias"],
                1.0,
                None,
            ),
        ),
        "routed_quant": build(
            "routed_static_scaled_fp8_quant",
            static_scaled_fp8_quant,
            (
                t["routed_input_q"],
                t["hidden_states"],
                t["routed_input_scale"],
                None,
            ),
        ),
        "prologue": build(
            "fused_build_expert_maps_sort_first_token",
            fused_build_expert_maps_sort_first_token,
            (
                t["topk_ids"],
                t["permuted_row_to_unpermuted_row"],
                t["unpermuted_row_to_permuted_row"],
                t["expert_first_token_offset"],
            ),
        ),
        "expand": build(
            "expand_input_rows",
            expand_input_rows,
            (
                t["routed_input_q"],
                t["permuted_input"],
                t["topk_weights"],
                t["permuted_scales"],
                t["permuted_row_to_unpermuted_row"],
                t["expert_first_token_offset"],
            ),
        ),
        "routed_gemm1": build(
            "routed_gemm1",
            routed_gemm1,
            (
                t["routed_gemm1"],
                t["permuted_input"],
                t["routed_w1"],
                t["g1_alphas"],
                t["expert_first_token_offset"],
            ),
        ),
        "routed_activation": build(
            "relu2_static_fp8_quant",
            relu2_static_fp8_quant,
            (
                t["routed_activation_q"],
                t["routed_gemm1"],
                t["a2_gscale"],
                None,
                t["expert_first_token_offset"],
            ),
        ),
        "routed_gemm2": build(
            "routed_gemm2_fused_finalize",
            routed_gemm2_fused_finalize,
            (
                t["routed_output"],
                t["routed_activation_q"],
                t["routed_w2"],
                t["g2_alphas"],
                t["expert_first_token_offset"],
                t["permuted_row_to_unpermuted_row"],
                t["permuted_scales"],
            ),
        ),
        "shared_input_quant": build(
            "shared_input_static_scaled_fp8_quant",
            static_scaled_fp8_quant,
            (
                t["shared_input_q"],
                t["hidden_states"],
                t["shared_up_input_scale"],
                None,
            ),
        ),
        "shared_up": build(
            "shared_up_scaled_fp8_mm",
            fp8_scaled_mm,
            (
                t["shared_input_q"],
                t["shared_up_weight"],
                t["shared_up_input_scale"],
                t["shared_up_weight_scale"],
                torch.bfloat16,
                None,
            ),
        ),
        "shared_activation": build(
            "shared_relu_squared",
            relu_squared,
            (t["shared_activation"], t["shared_up"]),
        ),
        "shared_activation_quant": build(
            "shared_activation_static_scaled_fp8_quant",
            static_scaled_fp8_quant,
            (
                t["shared_activation_q"],
                t["shared_activation"],
                t["shared_down_input_scale"],
                None,
            ),
        ),
        "shared_down": build(
            "shared_down_scaled_fp8_mm",
            fp8_scaled_mm,
            (
                t["shared_activation_q"],
                t["shared_down_weight"],
                t["shared_down_input_scale"],
                t["shared_down_weight_scale"],
                torch.bfloat16,
                None,
            ),
        ),
        "merge": build(
            "scale_routed_and_add_shared",
            scale_routed_and_add_shared,
            (
                t["shared_output"],
                t["routed_output"],
                shape.routed_scaling_factor,
            ),
        ),
    }
    return CompiledMoE(shape, tensors, kernels), selected_configs


def _assert_close(name, actual, expected, *, atol, rtol):
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
    max_abs = float((actual.float() - expected.float()).abs().max().item())
    print(f"correctness {name} max_abs={max_abs:.6f}", flush=True)


def validate(shape, tensors, compiled_moe) -> None:
    t = tensors
    output = compiled_moe(overlap_shared=True)
    torch.cuda.synchronize()
    actual = compiled_moe.last_outputs
    router_ref = torch.matmul(t["hidden_states"].float(), t["router_weight"].float().T)
    weights_ref, ids_ref, source_ref = _topk_reference(
        router_ref, t["correction_bias"], shape.top_k
    )
    permuted_ref, inverse_ref, offsets_ref = _prologue_reference(
        ids_ref, shape.num_experts
    )
    input_q_ref, gemm1_ref, activation_q_ref, routed_ref = _routed_reference(
        t["hidden_states"],
        weights_ref,
        ids_ref,
        t["routed_w1"],
        t["routed_w2"],
        t["g1_alphas"],
        t["a2_gscale"],
        t["g2_alphas"],
        t["routed_input_scale"],
    )
    shared_q_ref, shared_up_ref, shared_act_ref, shared_act_q_ref, shared_ref = (
        _shared_reference(
            t["hidden_states"],
            t["shared_up_weight"],
            t["shared_down_weight"],
            t["shared_up_input_scale"],
            t["shared_up_weight_scale"],
            t["shared_down_input_scale"],
            t["shared_down_weight_scale"],
        )
    )
    scaled_routed_ref = (routed_ref.float() * shape.routed_scaling_factor).to(
        torch.bfloat16
    )
    output_ref = (shared_ref + scaled_routed_ref).to(torch.bfloat16)

    _assert_close("router", actual["router_logits"], router_ref, atol=4e-2, rtol=2e-2)
    _assert_close("topk_weights", t["topk_weights"], weights_ref, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(t["topk_ids"], ids_ref, atol=0, rtol=0)
    torch.testing.assert_close(t["token_expert_indices"], source_ref, atol=0, rtol=0)
    torch.testing.assert_close(
        t["permuted_row_to_unpermuted_row"], permuted_ref, atol=0, rtol=0
    )
    torch.testing.assert_close(
        t["unpermuted_row_to_permuted_row"], inverse_ref, atol=0, rtol=0
    )
    torch.testing.assert_close(
        t["expert_first_token_offset"], offsets_ref, atol=0, rtol=0
    )
    torch.testing.assert_close(t["routed_input_q"].float(), input_q_ref.float())

    actual_gemm1 = torch.empty_like(gemm1_ref)
    actual_activation_q = torch.empty_like(activation_q_ref)
    for permuted_row in range(shape.tokens * shape.top_k):
        source_row = int(t["permuted_row_to_unpermuted_row"][permuted_row].item())
        token = source_row % shape.tokens
        slot = source_row // shape.tokens
        actual_gemm1[token, slot] = t["routed_gemm1"][permuted_row]
        actual_activation_q[token, slot] = t["routed_activation_q"][permuted_row]
    _assert_close("routed_gemm1", actual_gemm1, gemm1_ref, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(
        actual_activation_q.float(), activation_q_ref.float(), atol=1.0, rtol=0.0
    )
    _assert_close("routed_output", t["routed_output"], routed_ref, atol=8e-2, rtol=8e-2)
    torch.testing.assert_close(t["shared_input_q"].float(), shared_q_ref.float())
    _assert_close("shared_up", actual["shared_up"], shared_up_ref, atol=3e-2, rtol=3e-2)
    _assert_close(
        "shared_relu2", t["shared_activation"], shared_act_ref, atol=3e-2, rtol=3e-2
    )
    torch.testing.assert_close(
        t["shared_activation_q"].float(),
        shared_act_q_ref.float(),
        atol=1.0,
        rtol=0.0,
    )
    _assert_close(
        "shared_output", actual["shared_output"], shared_ref, atol=8e-2, rtol=8e-2
    )
    _assert_close("moe_output", output, output_ref, atol=1.5e-1, rtol=1e-1)


def run(args) -> None:
    require_idle_visible_gpu()
    shape = Nemotron3NanoMoEShape(tokens=args.tokens)
    shape.validate()
    tensors = allocate(shape)
    if args.tune:
        initialize_autotune_inputs(shape, tensors)
    compiled_moe, selected_configs = build_moe(args, shape, tensors)
    print("SOURCE", json.dumps(PRODUCTION_SOURCE, sort_keys=True), flush=True)
    print("SHAPE", json.dumps(asdict(shape), sort_keys=True), flush=True)
    print("CONFIGS", json.dumps(selected_configs, sort_keys=True), flush=True)

    if not args.skip_validation:
        validate(shape, tensors, compiled_moe)

    if not args.benchmark:
        return

    overlapped_graph, _ = capture(lambda: compiled_moe(overlap_shared=True))
    serial_graph, _ = capture(lambda: compiled_moe(overlap_shared=False))
    pids = visible_gpu_pids()
    timings = benchmark_interleaved(
        {
            "production_overlap": overlapped_graph.replay,
            "serial_shared_expert": serial_graph.replay,
        },
        args.repeats,
        args.batch_replays,
    )
    if visible_gpu_pids() != pids:
        raise RuntimeError("GPU process set changed during benchmark")
    print("TIMINGS", json.dumps(timings, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--tune", action="append", default=[])
    parser.add_argument("--tune-effort", choices=("quick", "full"), default="full")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch-replays", type=int, default=20)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    if args.describe:
        print(
            json.dumps(
                {"source": PRODUCTION_SOURCE, "kernels": PRODUCTION_KERNEL_GRAPH},
                indent=2,
            )
        )
        return
    run(args)


if __name__ == "__main__":
    main()
