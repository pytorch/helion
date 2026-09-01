"""Run one autotuning run for the autotuner study and record its artifacts.

Expected to run in a fresh subprocess with the environment already prepared by
campaign.py (CUDA_VISIBLE_DEVICES, HELION_AUTOTUNE_RANDOM_SEED,
HELION_SKIP_CACHE, HELION_AUTOTUNE_LOG, HELION_AUTOTUNER, ...).

Writes <out>/summary.json next to the per-candidate CSV written by the
autotune log sink.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--autotune-kwargs",
        default="{}",
        help="JSON kwargs forwarded to kernel.autotune (search constructor)",
    )
    args = parser.parse_args()

    from kernels import KERNEL_CASES  # pyrefly: ignore [missing-import]
    from kernels import load_kernel  # pyrefly: ignore [missing-import]
    import torch

    from helion.autotuner.metrics import AutotuneMetrics
    from helion.autotuner.metrics import register_post_autotune_hook

    case = KERNEL_CASES[args.case]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    kernel = load_kernel(case)
    kernel_args = case.make_args(device)

    collected: list[dict[str, Any]] = []

    def on_metrics(metrics: AutotuneMetrics) -> None:
        collected.append(metrics.to_dict())

    register_post_autotune_hook(on_metrics)

    autotune_kwargs = json.loads(args.autotune_kwargs)
    start = time.perf_counter()
    config = kernel.autotune(kernel_args, force=True, **autotune_kwargs)
    wall_time = time.perf_counter() - start

    # Independent post-search measurements on this GPU: the selected config and
    # the spec default config, each timed with the same interleaved harness so
    # run-to-run quality comparisons don't rely on in-search low-water marks.
    from helion.autotuner.benchmarking import interleaved_bench

    bound = kernel.bind(kernel_args)
    default_config = bound.config_spec.default_config()
    selected_fn = bound.compile_config(config)
    fns = [lambda: selected_fn(*kernel_args)]
    default_error: str | None = None
    try:
        default_fn = bound.compile_config(default_config)
        fns.append(lambda: default_fn(*kernel_args))
    except Exception as e:
        default_error = f"{type(e).__name__}: {e}"
    timings = interleaved_bench(fns, repeat=50)
    selected_perf_ms = timings[0]
    default_perf_ms = timings[1] if len(timings) > 1 else None

    summary: dict[str, Any] = {
        "case": args.case,
        "algorithm": os.environ.get("HELION_AUTOTUNER", "LFBOTreeSearch(default)"),
        "seed": os.environ.get("HELION_AUTOTUNE_RANDOM_SEED"),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "autotune_kwargs": autotune_kwargs,
        "wall_time_s": wall_time,
        "selected_config": config.config,
        "selected_perf_ms": selected_perf_ms,
        "default_perf_ms": default_perf_ms,
        "default_error": default_error,
        "default_config": default_config.config,
        "metrics": collected,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"DONE {args.case} wall={wall_time:.1f}s runs_recorded={len(collected)}")


if __name__ == "__main__":
    main()
