"""Single multi-shape config for kernel: per_token_group_fp8_quant (sm90 / H100).

Found with ``Kernel.autotune_multi`` (PR #3119) using a geometric-mean objective
relative to each shape's default config, tuned jointly over a representative
subset of the benchmark sweep (token extremes x hidden extremes):
``[(1, 2048), (64, 2048), (1024, 2048), (8192, 2048),
   (1, 5120), (64, 5120), (1024, 5120), (8192, 5120)]``.

Unlike the per-shape ``_helion_aot_*`` heuristic (a shape->config lookup table),
this is ONE config used for every shape. On the 24-shape H100 sweep it matches
the per-shape heuristic's geomean vs torch_compile (2.72x vs 2.71x) while
collapsing 41 configs to one. Benchmark it with::

    python per_token_group_fp8_quant.py --single-config

Provides:
- CONFIG: the config dict (pass to ``helion.kernel(config=...)``).
"""
from __future__ import annotations

CONFIG = {
    "block_sizes": [8],
    "loop_orders": [[2, 1, 0]],
    "l2_groupings": [8],
    "range_unroll_factors": [0],
    "range_warp_specializes": [],
    "range_num_stages": [],
    "range_multi_buffers": [None],
    "range_flattens": [None],
    "load_eviction_policies": ["first"],
    "num_warps": 4,
    "num_stages": 2,
    "indexing": ["tensor_descriptor", "tensor_descriptor", "tensor_descriptor"],
    "atomic_indexing": [],
    "pid_type": "flat",
}
