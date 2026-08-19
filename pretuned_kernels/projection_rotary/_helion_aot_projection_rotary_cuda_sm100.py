"""B200 config for the tcgen05 register-fragment projection rotary kernel."""

from __future__ import annotations


_CONFIG = {
    "block_sizes": [1, 128, 128, 128],
    "loop_orders": [[0, 1, 2]],
    "l2_groupings": [1],
    "indexing": ["pointer"] * 10,
    "pid_type": "persistent_interleaved",
    "tcgen05_cluster_m": 1,
    "tcgen05_cluster_n": 1,
    "tcgen05_ab_stages": 2,
    "tcgen05_acc_stages": 2,
    "tcgen05_c_stages": 2,
    "tcgen05_num_epi_warps": 4,
    "tcgen05_l2_swizzle_size": 1,
    "tcgen05_aux_load_placement": "pre_acc_wait",
    "tcgen05_strategy": "role_local_monolithic",
    "tcgen05_layout_strategy": "default",
    "tcgen05_warp_spec_mma_warps": 1,
    "tcgen05_warp_spec_ab_load_warps": 1,
    "tcgen05_warp_spec_epi_load_warps": 0,
    "tcgen05_warp_spec_scheduler_warps": 0,
    "tcgen05_warp_spec_c_input_warps": 0,
    "tcgen05_warp_spec_store_warps": 0,
    "tcgen05_warp_spec_register_decrease": 120,
    "tcgen05_warp_spec_register_increase": 256,
    "cute_vector_widths": [1, 1, 1, 1],
    "tcgen05_persistence_model": "static_persistent",
    "num_warps": 8,
    "num_stages": 2,
}


def key_projection_rotary(*args) -> int:
    """Use the sole checked-in config; static shapes remain separately compiled."""
    return 0


def autotune_projection_rotary(*args) -> dict:
    """Return the B200 config measured at M=1024, K=4096, H=32, D=128."""
    return _CONFIG
