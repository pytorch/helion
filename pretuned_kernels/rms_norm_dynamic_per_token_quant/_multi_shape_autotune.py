"""Find ONE config for rms_norm_dynamic_per_token_quant across shapes (PR #3119).

Uses ``Kernel.autotune_multi`` to search for a single config that performs well
over a representative subset of the benchmark sweep, then prints it in the exact
form checked into ``_multi_shape_config_cuda_sm<NN>.py``. Copy the printed
``CONFIG`` there, then benchmark it against the per-shape heuristic with::

    HELION_BENCHMARK_CUDAGRAPH=1 python rms_norm_dynamic_per_token_quant.py --single-config

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

from rms_norm_dynamic_per_token_quant import _FP8_DTYPE
from rms_norm_dynamic_per_token_quant import _bench_shapes
from rms_norm_dynamic_per_token_quant import (
    rms_norm_dynamic_per_token_quant as _aot_kernel,
)

# Plain helion.kernel over the same body (see module docstring).
rms_norm_dynamic_per_token_quant = helion.kernel(
    _aot_kernel.fn,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)

EPSILON = 1e-6


def _tune_shapes() -> list[tuple[int, int]]:
    """Representative subset (hidden_size, num_tokens) for the joint objective.

    Spans the two performance regimes (latency-bound tiny token counts and
    bandwidth-bound large token counts) at the two hidden-size extremes; a config
    robust across 2048 and 5120 generalizes to the 4096 middle. Keeps the
    across-shape search tractable (each candidate is measured on every set)."""
    return [
        (2048, 1),
        (2048, 64),
        (2048, 512),
        (2048, 2048),
        (5120, 1),
        (5120, 64),
        (5120, 512),
        (5120, 2048),
    ]


def _make_args(hidden_size: int, num_tokens: int) -> tuple:
    x_in = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight = torch.normal(
        mean=1.0, std=1.0, size=(hidden_size,), dtype=x_in.dtype, device="cuda"
    )
    result = torch.empty_like(x_in, dtype=_FP8_DTYPE)
    scale = torch.empty((num_tokens, 1), device="cuda", dtype=torch.float32)
    return (result, x_in, weight, scale, EPSILON)


def main() -> None:
    torch.manual_seed(0)
    tune_shapes = _tune_shapes()
    assert set(tune_shapes) <= set(_bench_shapes())

    # One representative argument set per tuning shape for the joint objective.
    arg_sets = list(starmap(_make_args, tune_shapes))

    print(
        f"Multi-shape autotune over {len(arg_sets)} shapes: {tune_shapes}", flush=True
    )
    config = rms_norm_dynamic_per_token_quant.autotune_multi(
        arg_sets,
        aggregation="geomean",
        relative_to="default",
        # Runtime numeric arg (epsilon) => cache_tag required.
        cache_tag="rms_norm_dynamic_per_token_multishape_v1",
    )

    major, minor = torch.cuda.get_device_capability()
    print(f"\n=== Winning config (sm{major}{minor}) ===")
    print(f"Copy into _multi_shape_config_cuda_sm{major}{minor}.py:\n")
    print("CONFIG = " + pprint.pformat(dict(config), sort_dicts=False))


if __name__ == "__main__":
    main()
