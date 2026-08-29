"""
Auto-generated heuristic for kernel: per_token_group_fp8_quant_packed
Tuned on NVIDIA B200 (sm100).

Re-tuned for the load-hoisted kernel with full-effort AOT autotuning
(HELION_AUTOTUNE_EFFORT=full) over the 21-shape benchmark sweep: hidden_size in
{2048, 4096, 5120}, num_tokens in {4, 16, 64, 256, 1024, 2048, 8192},
group_size=128. From the union of the freshly-tuned configs and the previous
config set, each shape's config was re-benchmarked with the harness's exact
CUDA-graph + L2-clear timing (the metric the dashboard reports) and the
per-shape fastest config selected. Seven distinct configs cover the whole sweep.
"""

import functools

_KEYS = [
    (2048, 128, 4),
    (2048, 128, 16),
    (2048, 128, 64),
    (2048, 128, 256),
    (2048, 128, 1024),
    (2048, 128, 2048),
    (2048, 128, 8192),
    (4096, 128, 4),
    (4096, 128, 16),
    (4096, 128, 64),
    (4096, 128, 256),
    (4096, 128, 1024),
    (4096, 128, 2048),
    (4096, 128, 8192),
    (5120, 128, 4),
    (5120, 128, 16),
    (5120, 128, 64),
    (5120, 128, 256),
    (5120, 128, 1024),
    (5120, 128, 2048),
    (5120, 128, 8192),
]

_CONFIG_INDEXES = [
    6,
    6,
    6,
    6,
    6,
    0,
    0,
    1,
    1,
    1,
    1,
    5,
    5,
    4,
    2,
    2,
    2,
    2,
    3,
    5,
    4,
]

_INDEX_BY_KEY = {
    key: cfg_idx for key, cfg_idx in zip(_KEYS, _CONFIG_INDEXES, strict=True)
}


@functools.cache
def _pick_config_index(hidden_size: int, group_size: int, num_tokens: int) -> int:
    target = (hidden_size, group_size, num_tokens)
    exact = _INDEX_BY_KEY.get(target)
    if exact is not None:
        return exact
    # Narrow to the closest value on every dimension except the last.
    cands = _KEYS
    for i in range(len(target) - 1):
        best = min({k[i] for k in cands}, key=lambda v: abs(v - target[i]))
        cands = [k for k in cands if k[i] == best]
    # Last dimension: smallest tuned value >= the input, else the largest.
    last = target[-1]
    avail = sorted({k[-1] for k in cands})
    chosen = next((v for v in avail if v >= last), avail[-1])
    picked = next(k for k in cands if k[-1] == chosen)
    return _INDEX_BY_KEY[picked]


def key_per_token_group_fp8_quant_packed(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    a = args[0]
    hidden_size = a.shape[1]
    group_size = args[3]
    num_tokens = a.shape[0]
    return _pick_config_index(hidden_size, group_size, num_tokens)


def autotune_per_token_group_fp8_quant_packed(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {
            "block_sizes": [4],
            "loop_orders": [[1, 2, 0]],
            "l2_groupings": [16],
            "range_unroll_factors": [0],
            "range_warp_specializes": [None],
            "range_num_stages": [],
            "range_multi_buffers": [None],
            "range_flattens": [None],
            "load_eviction_policies": ["first", "last", "last", ""],
            "num_warps": 1,
            "num_stages": 7,
            "indexing": [
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [8],
            "loop_orders": [[1, 2, 0]],
            "l2_groupings": [4],
            "range_unroll_factors": [0],
            "range_warp_specializes": [None],
            "range_num_stages": [],
            "range_multi_buffers": [None],
            "range_flattens": [None],
            "load_eviction_policies": ["last", "first", "last", "last"],
            "num_warps": 8,
            "num_stages": 3,
            "indexing": [
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [1],
            "loop_orders": [[2, 1, 0]],
            "l2_groupings": [16],
            "range_unroll_factors": [0],
            "range_warp_specializes": [None],
            "range_num_stages": [],
            "range_multi_buffers": [None],
            "range_flattens": [None],
            "load_eviction_policies": ["", "last", "", "last"],
            "num_warps": 1,
            "num_stages": 4,
            "indexing": [
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [2],
            "loop_orders": [[2, 1, 0]],
            "l2_groupings": [16],
            "range_unroll_factors": [0],
            "range_warp_specializes": [None],
            "range_num_stages": [],
            "range_multi_buffers": [None],
            "range_flattens": [None],
            "load_eviction_policies": ["first", "first", "first", "last"],
            "num_warps": 1,
            "num_stages": 2,
            "indexing": [
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "pointer",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [4],
            "loop_orders": [[2, 1, 0]],
            "l2_groupings": [16],
            "range_unroll_factors": [0],
            "range_warp_specializes": [None],
            "range_num_stages": [],
            "range_multi_buffers": [None],
            "range_flattens": [None],
            "load_eviction_policies": ["", "first", "first", "first"],
            "num_warps": 1,
            "num_stages": 1,
            "indexing": [
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [4],
            "loop_orders": [[2, 1, 0]],
            "l2_groupings": [2],
            "range_unroll_factors": [0],
            "range_warp_specializes": [None],
            "range_num_stages": [],
            "range_multi_buffers": [None],
            "range_flattens": [None],
            "load_eviction_policies": ["last", "first", "last", "last"],
            "num_warps": 1,
            "num_stages": 3,
            "indexing": [
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
        {
            "block_sizes": [4],
            "loop_orders": [[2, 1, 0]],
            "l2_groupings": [1],
            "range_unroll_factors": [0],
            "range_warp_specializes": [None],
            "range_num_stages": [],
            "range_multi_buffers": [None],
            "range_flattens": [None],
            "load_eviction_policies": ["last", "last", "", "last"],
            "num_warps": 4,
            "num_stages": 4,
            "indexing": [
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "tensor_descriptor",
                "pointer",
            ],
            "atomic_indexing": [],
            "pid_type": "flat",
        },
    ]
    return _C[key_per_token_group_fp8_quant_packed(*args)]
