"""Find ONE config for silu_and_mul_per_block_quant across shapes (PR #3119).

Uses ``Kernel.autotune_multi`` to search for a single config that performs well
over a representative subset of the benchmark sweep, then prints it in the exact
form checked into ``_multi_shape_config_cuda_sm<NN>.py``. Copy the printed
``CONFIG`` there, then benchmark it against the per-shape heuristic with::

    HELION_BENCHMARK_CUDAGRAPH=1 python silu_and_mul_per_block_quant.py --single-config

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

from silu_and_mul_per_block_quant import _bench_shapes
from silu_and_mul_per_block_quant import silu_and_mul_per_block_quant as _aot_kernel

# Plain helion.kernel over the same body (see module docstring).
silu_and_mul_per_block_quant = helion.kernel(
    _aot_kernel.fn,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)

GROUP_SIZE = 128


def _tune_shapes() -> list[tuple[int, int]]:
    """Representative subset for the joint objective.

    Spans the two performance regimes (latency-bound tiny token counts and
    bandwidth-bound large token counts) at the two intermediate-size extremes;
    a config robust across 6144 and 25600 generalizes to the 12288 middle.
    Keeps the across-shape search tractable (each candidate is measured on every
    set)."""
    return [
        (1, 6144),
        (64, 6144),
        (1024, 6144),
        (4096, 6144),
        (1, 25600),
        (64, 25600),
        (1024, 25600),
        (4096, 25600),
    ]


def _make_args(num_tokens: int, intermediate: int) -> tuple:
    x = torch.randn(num_tokens, 2 * intermediate, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(
        num_tokens, intermediate, device="cuda", dtype=torch.float8_e4m3fn
    )
    scales = torch.empty(
        num_tokens, intermediate // GROUP_SIZE, device="cuda", dtype=torch.float32
    )
    return (out, x, scales, GROUP_SIZE)


def main() -> None:
    torch.manual_seed(0)
    tune_shapes = _tune_shapes()
    assert set(tune_shapes) <= set(_bench_shapes())

    # One representative argument set per tuning shape for the joint objective.
    arg_sets = list(starmap(_make_args, tune_shapes))

    print(
        f"Multi-shape autotune over {len(arg_sets)} shapes: {tune_shapes}", flush=True
    )
    config = silu_and_mul_per_block_quant.autotune_multi(
        arg_sets,
        aggregation="geomean",
        relative_to="default",
        # Runtime numeric arg (group_size) => cache_tag required.
        cache_tag="silu_mul_pbq_multishape_v1",
    )

    major, minor = torch.cuda.get_device_capability()
    print(f"\n=== Winning config (sm{major}{minor}) ===")
    print(f"Copy into _multi_shape_config_cuda_sm{major}{minor}.py:\n")
    print("CONFIG = " + pprint.pformat(dict(config), sort_dicts=False))


if __name__ == "__main__":
    main()
