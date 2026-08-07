"""Synthetic-data tests for the four-arm analysis, trajectories, and stats."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

from helion_rag.experiment.head_to_head import ARMS
from helion_rag.stats.paired import bootstrap_geometric_mean_ci
from helion_rag.stats.paired import geometric_mean
from helion_rag.stats.paired import rank_biserial
from helion_rag.stats.paired import wilcoxon_pvalue

REPO_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "analyze_head_to_head",
    REPO_ROOT / "scripts" / "helion_rag" / "analyze_head_to_head.py",
)
assert _SPEC is not None and _SPEC.loader is not None
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)

# Per-arm synthetic latency multipliers relative to LFBO (lower = faster).
_ARM_LATENCY = {
    "lfbo": 1.0,
    "llm": 0.8,
    "hybrid_lfbo_llm": 0.7,
    "contextual_rag_llm": 0.6,
}
_KERNELS = [f"wl{i}" for i in range(6)]
_REPS = (1, 2, 3, 4, 5)


def _event(workload: str, arm: str, rep: int, incumbent: float) -> dict:
    # Two evaluations with a non-increasing incumbent and monotonic elapsed time.
    evals = [
        {
            "benchmark_status": "ok",
            "candidate_category": "initial_population",
            "candidate_source": "random",
            "elapsed_seconds": 0.5,
            "performance": incumbent * 1.2,
            "incumbent_best_perf": incumbent * 1.2,
        },
        {
            "benchmark_status": "ok",
            "candidate_category": "generation",
            "candidate_source": "random",
            "elapsed_seconds": 1.5,
            "performance": incumbent,
            "incumbent_best_perf": incumbent,
        },
    ]
    return {"run": {"workload_id": workload, "arm_id": arm}, "evaluations": evals}


def _build_campaign(tmp_path: Path) -> Path:
    campaign = tmp_path / "camp"
    results = campaign / "results"
    events = campaign / "events"
    results.mkdir(parents=True)
    events.mkdir(parents=True)
    order = 0
    for kernel_i, workload in enumerate(_KERNELS):
        for arm in ARMS:
            for rep in _REPS:
                order += 1
                latency = 10.0 * _ARM_LATENCY[arm] * (1.0 + 0.02 * kernel_i)
                readiness = 100.0 * _ARM_LATENCY[arm]
                run_id = f"{order:04d}-{workload}-{arm}-r{rep}"
                event_log = events / f"{run_id}.jsonl"
                event_log.write_text(
                    json.dumps(_event(workload, arm, rep, latency)) + "\n",
                    encoding="utf-8",
                )
                record = {
                    "workload": workload,
                    "arm": arm,
                    "repetition": rep,
                    "random_seed": 1000 + rep - 1,
                    "status": "completed",
                    "correct": True,
                    "selected_latency_ms": latency,
                    "readiness_seconds": readiness,
                    "internal_selected_performance": latency,
                    "incumbent_best_perf": latency,
                    "attempt_accounting": {"frozen_limit": 80, "attempted": 40},
                    "provider": {
                        "requests": 0 if arm == "lfbo" else 1,
                        "input_tokens": 0 if arm == "lfbo" else 100,
                        "output_tokens": 0 if arm == "lfbo" else 50,
                    },
                    "tier": 1 if arm == "contextual_rag_llm" else None,
                    "decision": "BaselineSearch",
                    "evaluation_count": 2,
                    "hybrid_stage_breakdown": (
                        {"llm_attempts": 25, "lfbo_attempts": 15}
                        if arm == "hybrid_lfbo_llm"
                        else None
                    ),
                    "event_log": f"events/{run_id}.jsonl",
                }
                (results / f"{workload}__{arm}__r{rep}.json").write_text(
                    json.dumps(record) + "\n", encoding="utf-8"
                )
    return campaign


def test_paired_helpers_are_deterministic_and_correct() -> None:
    assert geometric_mean([0.5, 1.0, 2.0]) == pytest.approx(1.0)
    a = bootstrap_geometric_mean_ci([0.8, 0.7, 0.9, 0.6], seed=7)
    b = bootstrap_geometric_mean_ci([0.8, 0.7, 0.9, 0.6], seed=7)
    assert a == b  # deterministic for a fixed seed
    assert a.low <= a.estimate <= a.high
    # All improvements (log-ratios < 0) => rank-biserial +1 (positive = numerator
    # better, matching the lifted convention) and a valid Wilcoxon p.
    logs = [math.log(x) for x in (0.8, 0.7, 0.9, 0.6, 0.75)]
    assert rank_biserial(logs) == pytest.approx(1.0)
    assert 0.0 <= wilcoxon_pvalue(logs) <= 1.0


def test_analysis_produces_all_products_and_matched_ratios(tmp_path: Path) -> None:
    campaign = _build_campaign(tmp_path)
    summary = analysis.analyze(campaign)
    assert summary["runs"] == len(_KERNELS) * len(ARMS) * len(_REPS)
    assert summary["kernels"] == len(_KERNELS)

    for name in (
        "per_run.csv",
        "per_kernel_arm.csv",
        "per_arm_summary.csv",
        "all_arm_table.csv",
        "all_arm_table.md",
        "aggregate_statistics.json",
        "aggregate_statistics.csv",
        "reliability.csv",
        "cost.csv",
        "trajectory_long.csv",
    ):
        assert (campaign / "analysis" / name).is_file(), name

    # Bounded-oracle regret: the fastest arm (contextual, 0.6x) is the oracle
    # (0% regret); LFBO (1.0x) carries the largest regret.
    kernel_rows = list(_read_csv(campaign / "analysis" / "per_kernel_arm.csv"))
    for row in kernel_rows:
        if row["workload"] == "wl0" and row["arm"] == "contextual_rag_llm":
            assert float(row["regret_pct"]) == pytest.approx(0.0, abs=1e-6)
        if row["workload"] == "wl0" and row["arm"] == "lfbo":
            # 1.0 / 0.6 - 1 = 66.7% regret vs the bounded oracle.
            assert float(row["regret_pct"]) == pytest.approx(100.0 / 1.5, abs=1e-3)
    # Per-arm rollup carries wall time, regret, configs, and tokens for each arm.
    summary_rows = {
        r["arm"]: r for r in _read_csv(campaign / "analysis" / "per_arm_summary.csv")
    }
    assert set(summary_rows) == set(ARMS)
    assert int(summary_rows["lfbo"]["input_tokens_total"]) == 0
    assert float(
        summary_rows["contextual_rag_llm"]["geomean_regret_pct"]
    ) == pytest.approx(0.0, abs=1e-6)
    # LFBO-baseline regret: lfbo is 0 by definition; contextual (0.6x lfbo) ~ -40%.
    assert float(summary_rows["lfbo"]["geomean_regret_vs_lfbo_pct"]) == pytest.approx(
        0.0, abs=1e-6
    )
    assert (
        float(summary_rows["contextual_rag_llm"]["geomean_regret_vs_lfbo_pct"]) < -30.0
    )

    aggregate = json.loads(
        (campaign / "analysis" / "aggregate_statistics.json").read_text(
            encoding="utf-8"
        )
    )
    assert aggregate["family_size"] == 12  # 6 pairwise contrasts x 2 outcomes
    # Holm-adjusted p-values are present and >= raw p for the family.
    for entry in aggregate["contrasts"]:
        assert "holm_p" in entry
    # llm-vs-lfbo latency ratio should recover the synthetic 0.8 multiplier.
    llm_latency = next(
        e
        for e in aggregate["contrasts"]
        if e["outcome"] == "selected_latency_ms"
        and e["numerator_arm"] == "llm"
        and e["denominator_arm"] == "lfbo"
    )
    assert llm_latency["geometric_mean_ratio"] == pytest.approx(0.8, abs=1e-6)
    assert llm_latency["wins"] == len(_KERNELS)  # llm faster on every kernel
    assert llm_latency["joint_success_coverage"] == 1.0


def test_trajectory_rows_monotonic_and_normalized(tmp_path: Path) -> None:
    campaign = _build_campaign(tmp_path)
    analysis.analyze(campaign)
    rows = list(_read_csv(campaign / "analysis" / "trajectory_long.csv"))
    assert rows
    # Per (workload, arm, rep): timestamps non-decreasing, incumbents non-increasing.
    by_run: dict[tuple, list[dict]] = {}
    for row in rows:
        by_run.setdefault((row["workload"], row["arm"], row["repetition"]), []).append(
            row
        )
    for series in by_run.values():
        times = [float(r["elapsed_seconds"]) for r in series]
        incumbents = [float(r["incumbent_best_perf"]) for r in series]
        assert times == sorted(times)
        assert all(a >= b for a, b in zip(incumbents, incumbents[1:]))
        # Normalized incumbent ratio is always >= 1 (oracle is the per-workload min).
        assert all(float(r["normalized_incumbent_ratio"]) >= 1.0 - 1e-9 for r in series)


def test_figures_render_from_synthetic_campaign(tmp_path: Path) -> None:
    campaign = _build_campaign(tmp_path)
    summary = analysis.analyze(campaign)
    figures = summary["figures"]
    if figures.get("_skipped"):
        pytest.skip("gnuplot not available")
    # The aggregate-effects figure renders in all three formats.
    produced = figures["aggregate_effects"]
    assert any(name.endswith(".svg") for name in produced)
    assert (campaign / "figures" / "01_aggregate_effects.svg").is_file()


def test_poster_metrics_cover_ratios_regret_and_time_to_hit(tmp_path: Path) -> None:
    campaign = _build_campaign(tmp_path)
    runs = analysis.load_runs(campaign)
    kernel_rows = analysis.per_kernel_arm(runs)
    trajectory = analysis.trajectory_rows(campaign, runs)

    ratios = analysis.poster_performance_ratios(kernel_rows)
    assert len(ratios) == len(_KERNELS) * 3
    assert {row["arm"] for row in ratios} == set(ARMS) - {"lfbo"}
    expected_ratios = {
        "llm": 0.8,
        "hybrid_lfbo_llm": 0.7,
        "contextual_rag_llm": 0.6,
    }
    assert all(
        row["ratio"] == pytest.approx(expected_ratios[row["arm"]]) for row in ratios
    )
    assert all(row["classification"] == "win" for row in ratios)

    selected = analysis.poster_workloads(kernel_rows, limit=3)
    assert len(selected) == 3
    # Returned in decreasing LFBO readiness (baseline difficulty) order.
    readiness = [
        next(
            float(row["median_readiness_seconds"])
            for row in kernel_rows
            if row["workload"] == workload and row["arm"] == "lfbo"
        )
        for workload in selected
    ]
    assert readiness == sorted(readiness, reverse=True)

    regret = analysis.suite_regret_curve(trajectory, points=9)
    assert {row["arm"] for row in regret} == set(ARMS)
    assert all(row["q25_pct"] <= row["median_pct"] <= row["q75_pct"] for row in regret)
    assert all(row["median_pct"] >= 0.0 for row in regret)

    hits = analysis.time_to_hit_curve(trajectory, threshold_pct=5.0, points=9)
    for arm in ARMS:
        arm_rows = [row for row in hits if row["arm"] == arm]
        coverage = [row["coverage_pct"] for row in arm_rows]
        assert coverage == sorted(coverage)
        assert 0.0 <= coverage[-1] <= 100.0
    final = {
        arm: [row for row in hits if row["arm"] == arm][-1]["coverage_pct"]
        for arm in ARMS
    }
    assert final["contextual_rag_llm"] == pytest.approx(100.0)


def test_poster_figure_pack_renders_all_requested_charts(tmp_path: Path) -> None:
    campaign = _build_campaign(tmp_path)
    summary = analysis.analyze(campaign)
    figures = summary["figures"]
    if figures.get("_skipped"):
        pytest.skip("gnuplot not available")
    expected = {
        "search_overhead",
        "cold_start_step",
        "per_kernel_ratio",
        "suite_regret",
        "time_to_hit",
        "small_multiples",
    }
    assert expected <= set(figures)
    for key in expected:
        assert any(name.endswith(".svg") for name in figures[key]), key
        for name in figures[key]:
            assert (campaign / "figures" / name).is_file(), name


def test_frontier_scatter_uses_all_points_and_method_level_pareto(
    tmp_path: Path,
) -> None:
    campaign = _build_campaign(tmp_path)
    runs = analysis.load_runs(campaign)
    kernel_rows = analysis.per_kernel_arm(runs)

    points = analysis.frontier_points(kernel_rows)
    assert len(points) == len(_KERNELS) * len(ARMS)
    assert {row["arm"] for row in points} == set(ARMS)
    assert all(row["latency_ms"] > 0 and row["readiness_seconds"] > 0 for row in points)

    summaries = analysis.frontier_strategy_summaries(points)
    assert len(summaries) == len(ARMS)
    assert all(len(row["ellipse"]) == 73 for row in summaries)
    assert all(x > 0 and y > 0 for row in summaries for x, y in row["ellipse"])
    # Synthetic arm multipliers improve performance and readiness together, so
    # the fastest contextual arm is the only nondominated method centroid.
    assert [row["arm"] for row in summaries if row["pareto"]] == ["contextual_rag_llm"]

    summary = analysis.analyze(campaign)
    assert "frontier_scatter" in summary["figures"]
    for name in summary["figures"]["frontier_scatter"]:
        assert (campaign / "figures" / name).is_file(), name


def _read_csv(path: Path):
    with path.open(encoding="utf-8") as handle:
        yield from csv.DictReader(handle)
