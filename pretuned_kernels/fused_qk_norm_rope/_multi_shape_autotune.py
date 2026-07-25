"""Find ONE config for fused_qk_norm_rope across shapes (PR #3119).

Uses ``Kernel.autotune_multi`` to search for a single config that performs well
over a representative subset of the benchmark sweep, then prints it in the exact
form checked into ``_multi_shape_config_cuda_sm<NN>.py``. Copy the printed
``CONFIG`` there, then benchmark it against the per-shape heuristic with::

    HELION_BENCHMARK_CUDAGRAPH=1 python fused_qk_norm_rope.py --single-config

``autotune_multi`` requires the default autotuner + local best-config cache, so
this wraps the kernel body with a plain ``helion.kernel`` (the shipped kernel
uses ``aot_kernel``, whose AOTAutotuneCache ``autotune_multi`` rejects). The
body is reused from the kernel module -- not duplicated.

Run:
    HELION_BENCHMARK_CUDAGRAPH=1 python _multi_shape_autotune.py
"""

from __future__ import annotations

import os
import pprint
import sys

import torch

import helion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from itertools import starmap

from fused_qk_norm_rope import _bench_shapes
from fused_qk_norm_rope import _make_inputs
from fused_qk_norm_rope import fused_qk_norm_rope as _aot_kernel

# Plain helion.kernel over the same body (see module docstring).
fused_qk_norm_rope = helion.kernel(
    _aot_kernel.fn,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)


def _tune_shapes() -> list[tuple[int, int, int]]:
    """Representative subset (num_tokens, q_heads, kv_heads) for the joint objective.

    Spans the two performance regimes (latency-bound tiny token counts and
    bandwidth-bound large token counts) at the two q-head extremes; a config
    robust across the (16, 8) and (64, 8) head configs generalizes to the (32, 8)
    middle. Keeps the across-shape search tractable (each candidate is measured
    on every set)."""
    return [
        (1, 16, 8),
        (128, 16, 8),
        (2048, 16, 8),
        (8192, 16, 8),
        (1, 64, 8),
        (128, 64, 8),
        (2048, 64, 8),
        (8192, 64, 8),
    ]


def main() -> None:
    torch.manual_seed(0)
    tune_shapes = _tune_shapes()
    assert set(tune_shapes) <= set(_bench_shapes())

    # One representative argument set per tuning shape for the joint objective.
    arg_sets = list(starmap(_make_inputs, tune_shapes))

    print(
        f"Multi-shape autotune over {len(arg_sets)} shapes: {tune_shapes}", flush=True
    )
    config = fused_qk_norm_rope.autotune_multi(
        arg_sets,
        aggregation="geomean",
        relative_to="default",
        # Runtime numeric args (eps, head counts) + tensors => cache_tag required.
        cache_tag="fused_qk_norm_rope_multishape_v1",
    )

    major, minor = torch.cuda.get_device_capability()
    print(f"\n=== Winning config (sm{major}{minor}) ===")
    print(f"Copy into _multi_shape_config_cuda_sm{major}{minor}.py:\n")
    print("CONFIG = " + pprint.pformat(dict(config), sort_dicts=False))


if __name__ == "__main__":
    main()
