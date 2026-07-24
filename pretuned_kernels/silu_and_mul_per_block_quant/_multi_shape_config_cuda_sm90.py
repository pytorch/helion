"""Single multi-shape config for kernel: silu_and_mul_per_block_quant (sm90 / H100).

Found with ``Kernel.autotune_multi`` (PR #3119) using a geometric-mean objective
relative to each shape's default config, tuned jointly over a representative
subset of the benchmark sweep (token extremes x intermediate-size extremes):
``[(1, 6144), (64, 6144), (1024, 6144), (4096, 6144),
   (1, 25600), (64, 25600), (1024, 25600), (4096, 25600)]``.

Unlike the per-shape ``_helion_aot_*`` heuristic (a shape->config lookup table),
this is ONE config used for every shape. Benchmark it with::

    python silu_and_mul_per_block_quant.py --single-config

Provides:
- CONFIG: the config dict (pass to ``helion.kernel(config=...)``).
"""

from __future__ import annotations

CONFIG = {
    "block_sizes": [8],
    "loop_orders": [[1, 0, 2]],
    "l2_groupings": [64],
    "range_unroll_factors": [0],
    "range_multi_buffers": [None],
    "range_flattens": [None],
    "load_eviction_policies": ["first", "first"],
    "num_warps": 4,
    "num_stages": 6,
    "indexing": ["tensor_descriptor", "pointer", "pointer", "pointer"],
    "pid_type": "flat",
    "atomic_indexing": [],
    "range_warp_specializes": [],
    "range_num_stages": [],
}
