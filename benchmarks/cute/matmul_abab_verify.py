"""Interleaved ABAB verify for one matmul variant: helion-cute vs a baseline.

Runs both implementations alternately in ONE process on ONE GPU so thermal
drift hits both sides equally; reports per-round pairs and the median ratio.
This is the ground-truth measurement for variants near the goal bar, where
one-sided drift between separately-run processes is enough to flip a verdict.

Usage:
    python benchmarks/cute/matmul_abab_verify.py \
        --m 4096 --n 4096 --k 4096 --dtype bfloat16 \
        --helion-config-file <path with helion.Config(...) repr> \
        --baseline aten --rounds 5

For --baseline quack-direct, pass --quack-config '{"tile_m": 256, ...}'.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any
from typing import Callable

os.environ["HELION_BACKEND"] = "cute"

import torch

import helion


def _thermal_warmup(seconds: float) -> None:
    w = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    t0 = time.time()
    while time.time() - t0 < seconds:
        for _ in range(40):
            w = w @ w
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), required=True
    )
    parser.add_argument("--helion-config-file", required=True)
    parser.add_argument("--baseline", choices=("aten", "quack-direct"), required=True)
    parser.add_argument("--quack-config", default=None)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup-ms", type=int, default=400)
    parser.add_argument("--rep-ms", type=int, default=400)
    args = parser.parse_args()

    from triton.testing import do_bench

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    torch.manual_seed(0)
    a = torch.randn(args.m, args.k, device="cuda", dtype=dtype)
    b = torch.randn(args.k, args.n, device="cuda", dtype=dtype)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from compare_matmul_backends import IdentityEpilogue  # pyrefly: ignore
    from examples.matmul import matmul

    config_src = Path(args.helion_config_file).read_text().strip()
    helion_config = eval(config_src, {"helion": helion})
    kernel_args = (a, b, IdentityEpilogue())
    bound = matmul.bind(kernel_args)
    bound.env.config_spec.cute_tcgen05_search_enabled = True
    bound.set_config(helion_config)
    helion_fn = lambda: bound(*kernel_args)  # noqa: E731

    baseline_fn: Callable[[], object]
    if args.baseline == "aten":
        if dtype == torch.float32:
            torch.backends.cuda.matmul.allow_tf32 = True
        baseline_fn = lambda: a @ b  # noqa: E731
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "quack"))
        from quack.gemm import gemm as gemm_dispatch  # pyrefly: ignore

        qcfg: dict[str, Any] = json.loads(args.quack_config)
        out = torch.empty((1, args.m, args.n), dtype=dtype, device="cuda")
        a_d = a.unsqueeze(0)
        b_d = b.mT.unsqueeze(0)

        def baseline_fn() -> torch.Tensor:
            gemm_dispatch(
                a_d,
                b_d,
                out,
                None,
                None,
                qcfg["tile_m"],
                qcfg["tile_n"],
                qcfg["cluster_m"],
                qcfg["cluster_n"],
                pingpong=qcfg["pingpong"],
                persistent=qcfg["persistent"],
                is_dynamic_persistent=qcfg["is_dynamic_persistent"],
                max_swizzle_size=qcfg["max_swizzle_size"],
                rowvec_bias=None,
            )
            return out[0]

    for fn in (helion_fn, baseline_fn):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    _thermal_warmup(8.0)

    flops = 2.0 * args.m * args.n * args.k
    helion_ms: list[float] = []
    base_ms: list[float] = []
    for r in range(args.rounds):
        hm = do_bench(helion_fn, warmup=args.warmup_ms, rep=args.rep_ms)
        bm = do_bench(baseline_fn, warmup=args.warmup_ms, rep=args.rep_ms)
        assert isinstance(hm, float) and isinstance(bm, float)
        helion_ms.append(hm)
        base_ms.append(bm)
        print(
            f"round{r}: helion {flops / hm / 1e9:.1f} TF "
            f"baseline {flops / bm / 1e9:.1f} TF ratio {bm / hm:.4f}",
            flush=True,
        )
    hmed = statistics.median(helion_ms)
    bmed = statistics.median(base_ms)
    result = {
        "helion_median_ms": hmed,
        "baseline_median_ms": bmed,
        "helion_tflops": flops / hmed / 1e9,
        "baseline_tflops": flops / bmed / 1e9,
        "ratio": bmed / hmed,
        "paired_ratios": [bm / hm for hm, bm in zip(helion_ms, base_ms, strict=True)],
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
