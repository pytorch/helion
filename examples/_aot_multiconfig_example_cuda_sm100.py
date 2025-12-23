"""
Auto-generated decision tree heuristic for kernel: row_softmax
Backend: decision_tree

Uses a hand-rolled decision tree for config selection.
"""

from __future__ import annotations

CONFIGS = [
    {
        "block_sizes": [4, 2048],
        "range_unroll_factors": [0, 1],
        "range_warp_specializes": [None, None],
        "range_num_stages": [0, 0],
        "range_multi_buffers": [None, True],
        "range_flattens": [None, None],
        "load_eviction_policies": ["", ""],
        "num_warps": 16,
        "num_stages": 2,
        "indexing": ["pointer", "pointer", "pointer"],
        "pid_type": "flat",
    },
    {
        "block_sizes": [4, 64],
        "range_unroll_factors": [0, 0],
        "range_warp_specializes": [None, False],
        "range_num_stages": [0, 0],
        "range_multi_buffers": [None, None],
        "range_flattens": [None, None],
        "load_eviction_policies": ["", ""],
        "num_warps": 4,
        "num_stages": 1,
        "indexing": ["pointer", "pointer", "pointer"],
        "pid_type": "flat",
    },
    {
        "block_sizes": [32, 128],
        "range_unroll_factors": [0, 0],
        "range_warp_specializes": [None, None],
        "range_num_stages": [0, 0],
        "range_multi_buffers": [None, None],
        "range_flattens": [None, None],
        "load_eviction_policies": ["", ""],
        "num_warps": 32,
        "num_stages": 2,
        "indexing": ["pointer", "pointer", "pointer"],
        "pid_type": "flat",
    },
    {
        "block_sizes": [4, 1024],
        "range_unroll_factors": [0, 0],
        "range_warp_specializes": [None, None],
        "range_num_stages": [0, 0],
        "range_multi_buffers": [None, True],
        "range_flattens": [None, None],
        "load_eviction_policies": ["", "first"],
        "num_warps": 32,
        "num_stages": 1,
        "indexing": ["pointer", "pointer", "pointer"],
        "pid_type": "flat",
    },
]


FEATURE_NAMES = ["arg0_dim0", "arg0_dim1", "arg0_numel", "arg0_dtype_size"]


def _predict(features: dict) -> int:
    """Predict config index using decision tree."""
    if features.get("arg0_dim0", 0) <= 1024.0:
        if features.get("arg0_dim1", 0) <= 1024.0:
            if features.get("arg0_dtype_size", 0) <= 2.0:
                return 0
            return 3
        return 0
    if features.get("arg0_dim0", 0) <= 2048.0:
        return 1
    if features.get("arg0_dim0", 0) <= 4096.0:
        return 2
    return 1


def select_config_row_softmax(features: dict) -> dict:
    """Select the optimal config for the given shape features."""
    config_idx = _predict(features)
    return CONFIGS[config_idx]


def select_config(kernel_name: str, features: dict) -> dict:
    """Generic config selector."""
    if kernel_name == "row_softmax":
        return select_config_row_softmax(features)
    raise ValueError(f"Unknown kernel: {kernel_name}")
