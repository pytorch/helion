"""Single multi-shape config for kernel: rms_norm_dynamic_per_token_quant (sm90 / H100).

Found with ``Kernel.autotune_multi`` (PR #3119) using a geometric-mean objective
relative to each shape's default config, tuned jointly over a representative
subset of the benchmark sweep (hidden_size, num_tokens) -- token extremes x
hidden-size extremes:
``[(2048, 1), (2048, 64), (2048, 512), (2048, 2048),
   (5120, 1), (5120, 64), (5120, 512), (5120, 2048)]``.

Unlike the per-shape ``_helion_aot_*`` heuristic (a shape->config lookup table),
this is ONE config used for every shape. Benchmark it with::

    python rms_norm_dynamic_per_token_quant.py --single-config

Provides:
- CONFIG: the config dict (pass to ``helion.kernel(config=...)``).
"""

from __future__ import annotations

CONFIG = {
    "block_sizes": [16, 2048, 1024],
    "range_unroll_factors": [0, 0, 2, 0],
    "range_multi_buffers": [None, False, True, None],
    "range_flattens": [None, False, False, True],
    "load_eviction_policies": ["last", "last", "first", "", "last"],
    "num_warps": 1,
    "num_stages": 1,
    "indexing": [
        "pointer",
        "pointer",
        "tensor_descriptor",
        "pointer",
        "pointer",
        "tensor_descriptor",
        "pointer",
    ],
    "pid_type": "flat",
    "atomic_indexing": [],
    "range_warp_specializes": [],
    "range_num_stages": [],
}
