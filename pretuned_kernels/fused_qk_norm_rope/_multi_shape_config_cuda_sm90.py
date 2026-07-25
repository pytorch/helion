"""Single multi-shape config for kernel: fused_qk_norm_rope (sm90 / H100).

Found with ``Kernel.autotune_multi`` (PR #3119) using a geometric-mean objective
relative to each shape's default config, tuned jointly over a representative
subset of the benchmark sweep (num_tokens, q_heads, kv_heads) -- token extremes
x q-head extremes:
``[(1, 16, 8), (128, 16, 8), (2048, 16, 8), (8192, 16, 8),
   (1, 64, 8), (128, 64, 8), (2048, 64, 8), (8192, 64, 8)]``.

Unlike the per-shape ``_helion_aot_*`` heuristic (a shape->config lookup table),
this is ONE config used for every shape. Benchmark it with::

    python fused_qk_norm_rope.py --single-config

Provides:
- CONFIG: the config dict (pass to ``helion.kernel(config=...)``).
"""

from __future__ import annotations

CONFIG = {
    "block_sizes": [16],
    "loop_orders": [[1, 2, 0]],
    "l2_groupings": [64],
    "range_unroll_factors": [0],
    "range_multi_buffers": [None],
    "range_flattens": [None],
    "load_eviction_policies": [
        "",
        "last",
        "last",
        "last",
        "last",
        "last",
        "",
        "last",
    ],
    "num_warps": 8,
    "num_stages": 1,
    "indexing": [
        "tensor_descriptor",
        "tensor_descriptor",
        "tensor_descriptor",
        "tensor_descriptor",
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
        "tensor_descriptor",
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
    ],
    "pid_type": "flat",
    "atomic_indexing": [],
    "range_warp_specializes": [],
    "range_num_stages": [],
}
