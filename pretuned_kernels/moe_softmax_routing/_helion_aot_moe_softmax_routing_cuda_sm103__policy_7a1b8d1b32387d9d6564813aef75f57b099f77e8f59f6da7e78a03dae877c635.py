"""Exact measured GB300 configuration; no online-tuning or default fallback."""

from __future__ import annotations

import torch

import helion

STRUCTURAL_POLICY = helion.CuteStructuralPolicy(
    cute_region_fission=True,
    cute_full_slice_matmul_tiling=True,
    cute_segmented_matmul_tiling=True,
    cute_flatten_nested_reductions=True,
    cute_materialize_transformed_operands=True,
)

_CONFIGS = {
    (32768, 256, 8, False, 1, 1, False, 1.0, 0.0, False): {
        "block_sizes": [32],
        "cute_lane_layouts": ["blocked", "blocked", "blocked"],
        "cute_reduction_reloads": ["auto", "auto"],
        "cute_topk_coarse_keys": True,
        "cute_topk_defer_value_gathers": False,
        "cute_topk_key_dtype": "int64",
        "cute_topk_key_encoder": "dsl",
        "cute_topk_key_recovery": "packed",
        "cute_topk_lanes_per_row": 16,
        "cute_topk_merge_schedule": "balanced",
        "cute_topk_output_vector_width": 2,
        "cute_topk_rank_mode": "signed",
        "cute_topk_rows_per_block": 8,
        "cute_topk_selection_layout": "distributed",
        "cute_topk_sort_network": "compact_pruned",
        "cute_topk_value_mode": "gather",
        "cute_topk_vector_width": 4,
        "cute_vector_widths": [1, 1, 1],
    }
}

CONFIGS = [
    helion.CuteStructuralConfig(config, STRUCTURAL_POLICY)
    for config in _CONFIGS.values()
]
_HELION_AOT_STRUCTURAL_MANIFEST = {
    "schema": "helion.aot.structural_model",
    "version": 1,
    "policy": STRUCTURAL_POLICY.to_dict(),
    "policy_id": STRUCTURAL_POLICY.identity(),
    "configs": {"moe_softmax_routing": [config.to_dict() for config in CONFIGS]},
}


def key_moe_softmax_routing(
    logits: torch.Tensor,
    bias: torch.Tensor,
    k: int,
    grouped: bool,
    groups: int,
    selected_groups: int,
    renormalize: bool,
    scale: float,
    softcap: float,
    use_bias: bool,
) -> tuple:
    if (
        tuple(logits.shape) != (32768, 256)
        or logits.dtype != torch.float32
        or logits.stride() != (256, 1)
        or tuple(bias.shape) != (256,)
        or bias.dtype != torch.float32
        or bias.stride() != (1,)
        or bias.device != logits.device
        or type(k) is not int
        or type(grouped) is not bool
        or type(groups) is not int
        or type(selected_groups) is not int
        or type(renormalize) is not bool
        or type(scale) is not float
        or type(softcap) is not float
        or type(use_bias) is not bool
    ):
        raise ValueError("No pretuned config for this MoE routing signature")
    key = (
        logits.size(0),
        logits.size(1),
        k,
        grouped,
        groups,
        selected_groups,
        renormalize,
        scale,
        softcap,
        use_bias,
    )
    if key not in _CONFIGS:
        raise ValueError(f"No pretuned MoE routing config for {key}")
    return key


def autotune_moe_softmax_routing(
    logits: torch.Tensor,
    bias: torch.Tensor,
    k: int,
    grouped: bool,
    groups: int,
    selected_groups: int,
    renormalize: bool,
    scale: float,
    softcap: float,
    use_bias: bool,
) -> helion.CuteStructuralConfig:
    key = key_moe_softmax_routing(
        logits,
        bias,
        k,
        grouped,
        groups,
        selected_groups,
        renormalize,
        scale,
        softcap,
        use_bias,
    )
    return helion.CuteStructuralConfig(_CONFIGS[key], STRUCTURAL_POLICY)
