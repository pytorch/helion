#!/usr/bin/env python3
"""Compare benchmark values between H100 and B200 result files.

Each input file must be valid JSON containing a list of entries with the
structure::

    {
      "benchmark": { ... },
      "model": { "name": "<model_name>" },
      "metric": { "benchmark_values": [ ... ] }
    }

The script asserts that each shared model has the same number of benchmark
samples and then ranks the indices whose H100 and B200 speedups are both larger
than 1.0 by their average speedup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ModelValues = Dict[str, List[float]]
RankedIndex = Tuple[int, float]

def load_results(path: Path) -> ModelValues:
    """Parse the results file into a mapping of model name to values list."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array of entries")

    results: ModelValues = {}
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {entry!r} in {path} is not a JSON object")

        model_info = entry.get("model")
        metric_info = entry.get("metric")
        if not isinstance(model_info, dict) or not isinstance(metric_info, dict):
            raise ValueError(f"Entry {entry!r} in {path} is missing model or metric information")

        model_name = model_info.get("name")
        values = metric_info.get("benchmark_values")
        if model_name is None or values is None:
            raise ValueError(f"Entry {entry!r} in {path} is missing required fields")
        if model_name in results:
            raise ValueError(f"Duplicate model '{model_name}' found in {path}")
        if not isinstance(values, Sequence):
            raise ValueError(f"benchmark_values for '{model_name}' must be a sequence")

        results[model_name] = list(values)

    return results

def ranked_indices_above_threshold(
    h_values: Sequence[float],
    b_values: Sequence[float],
    threshold: float = 1.0,
) -> List[RankedIndex]:
    """Return (index, average speedup) pairs sorted by descending average speedup."""
    averaged = [
        (idx, (h_val + b_val) / 2)
        for idx, (h_val, b_val) in enumerate(zip(h_values, b_values))
        if h_val > threshold and b_val > threshold
    ]
    return sorted(averaged, key=lambda item: item[1], reverse=True)

def format_ranked_indices(indices: Sequence[RankedIndex]) -> str:
    """Pretty-print ranked indices with average speedup to three decimals."""
    if not indices:
        return "[]"
    parts = [f"{idx} ({avg:.3f})" for idx, avg in indices]
    return "[" + ", ".join(parts) + "]"

def main(args: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare H100 and B200 benchmark results")
    parser.add_argument("--h100", type=Path, default=Path("h100_results.py"), help="Path to the H100 results file")
    parser.add_argument("--b200", type=Path, default=Path("b200_results.py"), help="Path to the B200 results file")
    parsed = parser.parse_args(args)

    h100_results = load_results(parsed.h100)
    b200_results = load_results(parsed.b200)

    models_h100 = set(h100_results)
    models_b200 = set(b200_results)

    missing_in_b200 = sorted(models_h100 - models_b200)
    missing_in_h100 = sorted(models_b200 - models_h100)
    if missing_in_b200:
        print(f"Models missing in B200 results: {', '.join(missing_in_b200)}")
    if missing_in_h100:
        print(f"Models missing in H100 results: {', '.join(missing_in_h100)}")

    intersection = sorted(models_h100 & models_b200)
    if not intersection:
        raise ValueError("No shared models found between the result files")

    for model in intersection:
        h_vals = h100_results[model]
        b_vals = b200_results[model]
        if len(h_vals) != len(b_vals):
            raise AssertionError(
                f"Model '{model}' has {len(h_vals)} H100 values but {len(b_vals)} B200 values"
            )
        ranked = ranked_indices_above_threshold(h_vals, b_vals, threshold=1.0)
        print(f"Model: {model}")
        print(f"  ranked shared >1 indices (avg speedup): {format_ranked_indices(ranked)}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
