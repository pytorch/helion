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
    (16, 8192, 1024, 1, 1): {
        "block_sizes": [],
        "cute_cluster_n": 16,
        "cute_fragment_atomic_consumer_fusion": True,
        "cute_fragment_packet_loads": True,
        "cute_fragment_published_scalars": True,
        "cute_fragment_reduction": "warp",
        "cute_fragment_register_producers": True,
        "cute_fragment_register_snapshots": True,
        "cute_fragment_skip_zero_atomics": True,
        "cute_fragment_threads": 1024,
        "cute_fragment_warp_scan": True,
        "cute_independent_reduction": False,
        "cute_lane_layouts": [
            "blocked",
            "blocked",
            "blocked",
            "blocked",
            "strided"
        ],
        "cute_min_blocks_per_mp": 4,
        "cute_reduction_reloads": [
            "auto",
            "gmem",
            "auto",
            "auto"
        ],
        "cute_replicated_reduction": False,
        "cute_vector_packet_unroll": False,
        "cute_vector_widths": [
            1,
            1,
            1,
            4,
            1
        ],
        "load_eviction_policies": [
            "",
            "l2_last",
            "l2_last"
        ],
        "num_threads": [
            0,
            8,
            512,
            1
        ]
    },
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
    "configs": {"varlen_topk": [config.to_dict() for config in CONFIGS]},
}


def key_varlen_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    next_n: int = 1,
    compress_ratio: int = 1,
) -> tuple[int, int, int, int, int]:
    if (
        tuple(logits.shape) != (16, 8192)
        or logits.dtype != torch.float32
        or logits.stride() != (8192, 1)
        or tuple(lengths.shape) != (16,)
        or lengths.dtype != torch.int32
        or lengths.stride() != (1,)
        or lengths.device != logits.device
        or type(k) is not int
        or type(next_n) is not int
        or type(compress_ratio) is not int
    ):
        raise ValueError("No pretuned config for this variable-length top-k signature")
    key = (logits.size(0), logits.size(1), k, next_n, compress_ratio)
    if key not in _CONFIGS:
        raise ValueError(f"No pretuned variable-length top-k config for {key}")
    return key


def autotune_varlen_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    next_n: int = 1,
    compress_ratio: int = 1,
) -> helion.CuteStructuralConfig:
    return helion.CuteStructuralConfig(
        _CONFIGS[key_varlen_topk(logits, lengths, k, next_n, compress_ratio)],
        STRUCTURAL_POLICY,
    )
