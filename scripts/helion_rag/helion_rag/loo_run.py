"""Run one isolated GPU cell of the leave-one-workload-out matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback

from helion_rag.loo_experiment import arm_environment


def _install_metrics_capture() -> list:
    from helion.autotuner import metrics

    captured = []
    metrics.register_post_autotune_hook(captured.append)
    return captured


def _install_retrieval_capture() -> dict:
    """Time every RAG lookup (hit or miss) and record the last tier."""
    import helion_rag.patch as patch

    stats = {"retrieval_s": 0.0, "calls": 0, "last_tier": None}
    original = patch.lookup

    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        result = original(*args, **kwargs)
        stats["retrieval_s"] += time.perf_counter() - start
        stats["calls"] += 1
        if isinstance(result, dict):
            stats["last_tier"] = result.get("tier")
        return result

    patch.lookup = wrapped
    return stats


def _neighbor_metrics(
    lookup: dict | None,
    *,
    tier: int | None,
    retrieval_s: float,
    target_kernel: str,
    target_shapes: str,
) -> dict:
    """Derive leakage/coverage metrics from a stashed Tier-1 lookup result.

    ``same_kernel_neighbor_rate`` is this query's fraction of neighbours sharing
    the target's kernel identity (aggregation with equal workload weight happens
    in the report). ``heldout_shape_leaked`` is a per-cell proxy — the exact
    held-out (kernel, shape) must never appear (the authoritative check is the
    fold leakage audit).
    """
    neighbors = (lookup.get("neighbors") if lookup else None) or []
    same_kernel = [
        neighbor
        for neighbor in neighbors
        if neighbor.get("kernel_name") == target_kernel
    ]
    return {
        "tier": tier,
        "retrieval_s": retrieval_s,
        "retrieved_kernels": [neighbor.get("kernel_name") for neighbor in neighbors],
        "retrieved_shapes": [neighbor.get("input_shapes") for neighbor in neighbors],
        "same_kernel_neighbor_rate": (
            len(same_kernel) / len(neighbors) if neighbors else None
        ),
        "heldout_shape_leaked": any(
            neighbor.get("input_shapes") == target_shapes for neighbor in same_kernel
        ),
    }


def _classify_failure(message: str) -> str:
    """Map an exception message to an explicit execution-failure status."""
    low = message.lower()
    if "out of memory" in low or "cuda oom" in low or "outofmemory" in low:
        return "oom"
    if "accuracy" in low or "allclose" in low or "tolerance" in low:
        return "accuracy_fail"
    if "invalidconfig" in low or "compile" in low or "compilation" in low:
        return "compile_fail"
    return "error"


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def run_cell(item: dict, *, fold_index_dir: Path) -> dict:
    """Execute one matrix cell in the current fresh process."""
    os.environ.update(arm_environment(item["arm"], fold_index_dir))
    os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "full")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    from helion_rag.loo_inputs import build_inputs
    from helion_rag.loo_inputs import load_kernel

    retrieval = None
    if os.environ.get("HELION_RAG_LOO_SEEDING") == "1":
        from helion_rag.patch import install

        install()
        retrieval = _install_retrieval_capture()

    import torch
    from triton.testing import do_bench

    metrics = _install_metrics_capture()
    inputs = build_inputs(item["kernel"], item["shapes"], item["dtypes"])
    kernel = load_kernel(item["kernel"])
    bound = kernel.bind(inputs)

    torch.cuda.synchronize()
    started = time.perf_counter()
    bound(*inputs)
    torch.cuda.synchronize()
    end_to_end = time.perf_counter() - started

    metric = metrics[-1] if metrics else None
    result = dict(item)
    result.update(
        {
            "ok": True,
            "status": "ok",
            "end_to_end_s": end_to_end,
            "autotune_time_s": float(metric.autotune_time) if metric else 0.0,
            "num_configs_tested": int(metric.num_configs_tested) if metric else 0,
            "num_generations": int(metric.num_generations) if metric else 0,
            "num_compile_failures": (int(metric.num_compile_failures) if metric else 0),
            "num_accuracy_failures": (
                int(metric.num_accuracy_failures) if metric else 0
            ),
            "metrics_best_perf_ms": (float(metric.best_perf_ms) if metric else None),
            "perf_ms": float(do_bench(lambda: bound(*inputs), rep=100)),
            "final_config": dict(bound._config) if bound._config is not None else None,
        }
    )
    lookup = getattr(bound, "_helion_rag_lookup", None)
    result.update(
        _neighbor_metrics(
            lookup,
            tier=retrieval["last_tier"] if retrieval else None,
            retrieval_s=retrieval["retrieval_s"] if retrieval else 0.0,
            target_kernel=item["kernel"],
            target_shapes=item["shapes"],
        )
    )
    return result


def run_oracle_cell(item: dict) -> dict:
    """Rebenchmark the target workload's five recorded-best configurations."""
    import helion
    from helion_rag.loo_inputs import build_inputs
    from helion_rag.loo_inputs import load_kernel
    import torch
    from triton.testing import do_bench

    inputs = build_inputs(item["kernel"], item["shapes"], item["dtypes"])
    kernel = load_kernel(item["kernel"])
    timings = []
    errors = []
    for config in item.get("oracle_configs", []):
        try:
            bound = kernel.bind(inputs)
            bound.set_config(helion.Config(**config))
            torch.cuda.synchronize()
            timing = float(do_bench(lambda: bound(*inputs), rep=100))
            torch.cuda.synchronize()
            timings.append({"config": config, "perf_ms": timing})
        except Exception as exc:
            # One unbenchmarkable stored config must not void the whole oracle;
            # record it and require at least one survivor below.
            errors.append(f"{type(exc).__name__}: {exc}"[:500])
    if not timings:
        raise RuntimeError("none of the top-five recorded configs could be benchmarked")
    result = dict(item)
    result.update(
        ok=True,
        oracle_perf_ms=min(timing["perf_ms"] for timing in timings),
        oracle_timings=timings,
        oracle_errors=errors,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-json", required=True)
    parser.add_argument("--fold-index-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    item = json.loads(args.item_json)
    try:
        result = (
            run_oracle_cell(item)
            if item["arm"] == "oracle"
            else run_cell(item, fold_index_dir=Path(args.fold_index_dir))
        )
    except Exception as exc:
        # Terminal boundary: every cell must write a row, so a crashed cell is
        # recorded as a classified failure rather than leaving a silent gap.
        message = f"{type(exc).__name__}: {exc}"
        result = dict(item)
        result.update(
            ok=False,
            status=_classify_failure(message),
            error=message,
            traceback=traceback.format_exc()[-4000:],
        )
    _append(Path(args.out), result)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
