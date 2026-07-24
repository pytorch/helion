"""Single multi-shape config for kernel: rms_norm_per_block_quant (sm90 / H100).

Found with ``Kernel.autotune_multi`` (PR #3119) using a geometric-mean objective
relative to each shape's default config, tuned jointly over a representative
subset of the benchmark sweep (hidden, group, token) -- token extremes x
hidden-size extremes:
``[(2048, 128, 1), (2048, 128, 64), (2048, 128, 1024), (2048, 128, 4096),
   (5120, 128, 1), (5120, 128, 64), (5120, 128, 1024), (5120, 128, 4096)]``.

Unlike the per-shape ``_helion_aot_*`` heuristic (a shape->config lookup table),
this is ONE config used for every shape. Benchmark it with::

    python rms_norm_per_block_quant.py --single-config

Provides:
- CONFIG: the config dict (pass to ``helion.kernel(config=...)``).
"""

from __future__ import annotations

CONFIG = {
    "block_sizes": [2048, 16],
    "loop_orders": [[1, 0]],
    "range_unroll_factors": [0, 2, 2, 4],
    "range_multi_buffers": [None, True, True, True],
    "range_flattens": [None, None, False, True],
    "static_ranges": [False],
    "load_eviction_policies": ["", "", "first", "last", "last", ""],
    "num_warps": 8,
    "num_stages": 1,
    "indexing": [
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
        "tensor_descriptor",
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
    ],
    "pid_type": "flat",
    "atomic_indexing": [],
    "range_warp_specializes": [],
    "range_num_stages": [],
}
