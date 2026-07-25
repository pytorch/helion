"""Find ONE config for per_token_group_fp8_quant across shapes (PR #3119).

Uses ``Kernel.autotune_multi`` to search for a single config that performs well
over a representative subset of the benchmark sweep, then prints it in the exact
form checked into ``_multi_shape_config_cuda_sm<NN>.py``. Copy the printed
``CONFIG`` there, then benchmark it against the per-shape heuristic with::

    HELION_BENCHMARK_CUDAGRAPH=1 python per_token_group_fp8_quant.py --single-config

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

from per_token_group_fp8_quant import _bench_shapes
from per_token_group_fp8_quant import per_token_group_fp8_quant as _aot_kernel

# Plain helion.kernel over the same body (see module docstring).
per_token_group_fp8_quant = helion.kernel(
    _aot_kernel.fn,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)

GROUP_SIZE = 128
EPS = 1e-10
FP8_MIN, FP8_MAX = -448.0, 448.0
SCALE_UE8M0 = False
CONST = (GROUP_SIZE, EPS, FP8_MIN, FP8_MAX, SCALE_UE8M0)


def _tune_shapes() -> list[tuple[int, int]]:
    """Representative subset for the joint objective.

    Spans the two performance regimes (latency-bound tiny token counts and
    bandwidth-bound large token counts) at the two hidden-size extremes; a config
    robust across 2048 and 5120 generalizes to the 4096 middle. Keeps the
    across-shape search tractable (each candidate is measured on every set)."""
    return [
        (1, 2048),
        (64, 2048),
        (1024, 2048),
        (8192, 2048),
        (1, 5120),
        (64, 5120),
        (1024, 5120),
        (8192, 5120),
    ]


def _make_args(num_tokens: int, hidden_size: int) -> tuple:
    inp = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    groups = hidden_size // GROUP_SIZE
    oq = torch.empty_like(inp, dtype=torch.float8_e4m3fn)
    os_ = torch.empty((num_tokens, groups), device="cuda", dtype=torch.float32)
    return (inp, oq, os_, *CONST)


def main() -> None:
    torch.manual_seed(0)
    tune_shapes = _tune_shapes()
    assert set(tune_shapes) <= set(_bench_shapes())

    # One representative argument set per tuning shape for the joint objective.
    arg_sets = list(starmap(_make_args, tune_shapes))

    print(
        f"Multi-shape autotune over {len(arg_sets)} shapes: {tune_shapes}", flush=True
    )
    config = per_token_group_fp8_quant.autotune_multi(
        arg_sets,
        aggregation="geomean",
        relative_to="default",
        # Runtime numeric args (eps/fp8_min/fp8_max) => cache_tag required.
        cache_tag="ptgq_multishape_v1",
    )

    major, minor = torch.cuda.get_device_capability()
    print(f"\n=== Winning config (sm{major}{minor}) ===")
    print(f"Copy into _multi_shape_config_cuda_sm{major}{minor}.py:\n")
    print("CONFIG = " + pprint.pformat(dict(config), sort_dicts=False))


if __name__ == "__main__":
    main()
