"""Format benchmark results into compact feedback for LLM prompts."""

from __future__ import annotations

import collections
import json
import math
import operator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..base_search import BenchmarkResult


MAX_RESULTS_IN_PROMPT = 12
MAX_ANCHORS_IN_PROMPT = 2
MAX_CHANGED_FIELDS_PER_CONFIG = 6


def format_config_for_prompt(cfg: Config | dict[str, object]) -> str:
    """Serialize a config exactly as it should appear in prompt examples."""
    return json.dumps(dict(cfg), sort_keys=True, separators=(",", ":"))


def config_diff(
    default_config_dict: dict[str, object], cfg: Config
) -> dict[str, object]:
    """Drop unchanged defaults so feedback focuses on meaningful tunables."""
    config_dict = dict(cfg)
    diff = {
        key: value
        for key, value in config_dict.items()
        if key not in default_config_dict or value != default_config_dict[key]
    }
    if not diff:
        return {"(default)": True}
    return diff


def format_config_diff(default_config_dict: dict[str, object], cfg: Config) -> str:
    """Format only the changed fields as compact JSON."""
    return json.dumps(config_diff(default_config_dict, cfg), sort_keys=True)


def finite_results(results: list[BenchmarkResult]) -> list[tuple[Config, float]]:
    """Return successful benchmark results sorted from fastest to slowest."""
    return sorted(
        [
            (result.config, result.perf)
            for result in results
            if math.isfinite(result.perf)
        ],
        key=operator.itemgetter(1),
    )


def failed_benchmark_results(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    """Return configs that failed or produced non-finite perf values."""
    return [
        result
        for result in results
        if result.status in {"error", "timeout", "peer_compilation_fail"}
        or not math.isfinite(result.perf)
    ]


def format_results_for_llm(
    results: list[BenchmarkResult],
    default_config_dict: dict[str, object],
    *,
    limit: int = MAX_RESULTS_IN_PROMPT,
) -> str:
    """Format successful and failed benchmark results into a compact block."""
    if not results:
        return "No results yet."

    sorted_results = finite_results(results)
    failed_count = len(failed_benchmark_results(results))

    lines: list[str] = []
    for index, (cfg, perf) in enumerate(sorted_results[:limit], start=1):
        lines.append(
            f"  #{index}: {perf:.4f} ms — "
            f"{format_config_diff(default_config_dict, cfg)}"
        )
    if failed_count > 0:
        lines.append(f"  ({failed_count} configs failed to compile or had errors)")
    return "\n".join(lines)


def summarize_search_state_for_llm(
    results: list[BenchmarkResult],
    default_config_dict: dict[str, object],
) -> str:
    """Summarize the current best result, coverage, and failure counts."""
    finite = finite_results(results)
    if not finite:
        return "  No successful configs yet."

    best_cfg, best_perf = finite[0]
    lines = [
        (
            f"  Best so far: {best_perf:.4f} ms — "
            f"{format_config_diff(default_config_dict, best_cfg)}"
        )
    ]
    if len(finite) >= 2 and finite[1][1] > 0:
        gap_pct = ((finite[1][1] - best_perf) / finite[1][1]) * 100
        lines.append(f"  Margin vs runner-up: {gap_pct:.1f}%")
    failed_count = len(failed_benchmark_results(results))
    if results:
        lines.append(
            f"  Search coverage: {len(finite)} successful / "
            f"{len(results)} total configs"
        )
    if failed_count > 0:
        lines.append(f"  Failed configs so far: {failed_count}")
    return "\n".join(lines)


def summarize_failed_configs_for_llm(
    results: list[BenchmarkResult],
    default_config_dict: dict[str, object],
) -> str:
    """Show a small sample of failed configs so the next round can avoid them."""
    failed = failed_benchmark_results(results)
    if not failed:
        return "  No failed configs yet."

    counts = collections.Counter(
        "timeout" if result.status == "timeout" else "error" for result in failed
    )
    count_summary = ", ".join(
        f"{label}={counts[label]}" for label in ("error", "timeout") if counts[label]
    )
    lines = [f"  Counts: {count_summary}"]

    seen: set[str] = set()
    for result in failed:
        label = "timeout" if result.status == "timeout" else "error"
        diff_text = format_config_diff(default_config_dict, result.config)
        key = f"{label}:{diff_text}"
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  {label}: {diff_text}")
        if len(lines) >= 5:
            break
    return "\n".join(lines)


def summarize_anchor_configs_for_llm(
    results: list[BenchmarkResult],
    default_config_dict: dict[str, object],
    *,
    limit: int = MAX_ANCHORS_IN_PROMPT,
) -> str:
    """Show the strongest current configs that the next round should refine."""
    finite = finite_results(results)
    if not finite:
        return "  No successful configs yet."

    best_perf = finite[0][1]
    lines: list[str] = []
    for index, (cfg, perf) in enumerate(finite[:limit], start=1):
        if index == 1 or best_perf <= 0:
            delta = "best"
        else:
            gap_pct = ((perf - best_perf) / best_perf) * 100
            delta = f"+{gap_pct:.1f}%"
        lines.append(
            f"  Anchor {index} ({delta}): {perf:.4f} ms — "
            f"{format_config_diff(default_config_dict, cfg)}"
        )
    return "\n".join(lines)


def analyze_top_configs(
    results: list[BenchmarkResult],
    default_config_dict: dict[str, object],
) -> str:
    """Highlight which field values repeat across the best configs so far."""
    finite = finite_results(results)
    if len(finite) < 3:
        return "Not enough results for analysis yet."

    top = [
        {
            key: value
            for key, value in config_diff(default_config_dict, cfg).items()
            if key != "(default)"
        }
        for cfg, _ in finite[:5]
    ]
    all_keys = sorted({key for cfg in top for key in cfg})
    if not all_keys:
        return "Top configs are all close to the default config."

    lines: list[str] = []
    for key in all_keys:
        values = [cfg.get(key) for cfg in top if key in cfg]
        if not values:
            continue
        encoded = [
            json.dumps(value, sort_keys=True, separators=(",", ":")) for value in values
        ]
        counts = collections.Counter(encoded)
        common = counts.most_common(2)
        if len(common) == 1:
            lines.append(f"  {key}: always {common[0][0]}")
        elif common[0][1] >= 3:
            lines.append(
                f"  {key}: mostly {common[0][0]} (also {common[1][0]} x{common[1][1]})"
            )
        else:
            summary = ", ".join(f"{value} x{count}" for value, count in common)
            lines.append(f"  {key}: {summary}")
    return "\n".join(lines) or "No clear patterns yet."
