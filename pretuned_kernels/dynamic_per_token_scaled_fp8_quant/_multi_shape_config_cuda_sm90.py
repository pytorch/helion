"""Single multi-shape config for kernel: dynamic_per_token_scaled_fp8_quant (sm90 / H100).

Found with ``Kernel.autotune_multi`` (PR #3119) using a geometric-mean objective
relative to each shape's default config, tuned jointly over a representative
subset of the benchmark sweep (num_tokens, hidden_size) -- token extremes x
hidden-size extremes:
``[(1, 2048), (64, 2048), (1024, 2048), (4096, 2048),
   (1, 5120), (64, 5120), (1024, 5120), (4096, 5120)]``.

Unlike the per-shape ``_helion_aot_*`` heuristic (a shape->config lookup table),
this is ONE config used for every shape. Benchmark it with::

    python dynamic_per_token_scaled_fp8_quant.py --single-config

Provides:
- CONFIG: the config dict (pass to ``helion.kernel(config=...)``).
"""

from __future__ import annotations

CONFIG = {
    "block_sizes": [2048, 2048],
    "range_unroll_factors": [0, 2, 3],
    "range_multi_buffers": [None, None, None],
    "range_flattens": [None, False, None],
    "load_eviction_policies": ["", "last"],
    "num_warps": 4,
    "num_stages": 6,
    "indexing": ["pointer", "pointer", "pointer", "pointer"],
    "pid_type": "flat",
    "atomic_indexing": [],
    "range_warp_specializes": [],
    "range_num_stages": [],
}
