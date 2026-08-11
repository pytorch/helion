"""Checked-in B200 heuristic for ``grouped_gemm_deepgemm``.

The single schedule is the DeepGEMM-selected configuration validated across
all eight official BF16 grouped-NT rows by
``benchmarks/cute/deepgemm_selected_path.py``.  The user AOT key supplies
``(M_total_aligned, K, G, N, K, worklist_rows)``; one config is intentionally
used for every official shape.
"""

_CONFIG = {
    "block_sizes": [256, 128, 64],
    "l2_groupings": [1],
    "loop_orders": [[0, 1, 2]],
    "num_stages": 7,
    "num_warps": 8,
    "pid_type": "persistent_interleaved",
    "tcgen05_cluster_m": 2,
    "tcgen05_cluster_n": 1,
    "tcgen05_ab_stages": 7,
    "tcgen05_acc_stages": 2,
    "tcgen05_c_stages": 2,
    "tcgen05_num_epi_warps": 4,
    "tcgen05_grouped_mode": "worklist_nm",
    "tcgen05_grouped_worklist_source_m_tile": 224,
}

# Standalone AOT compile mode looks for this explicit config collection.
CONFIGS = [_CONFIG]


def key_grouped_gemm_deepgemm(*args: int) -> tuple[int, ...]:
    """Keep each packed/grouped shape as a distinct compiled specialization."""
    return tuple(args)


def autotune_grouped_gemm_deepgemm(*args: int) -> dict:
    """Return the DeepGEMM-selected B200 grouped-GEMM config."""
    return _CONFIG
