"""Synthetic-data tests for the matplotlib narrative figures."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys

import pytest

from helion_rag.experiment.head_to_head import ARMS

REPO_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "plot_narrative_figures",
    REPO_ROOT / "scripts" / "helion_rag" / "plot_narrative_figures.py",
)
assert _SPEC is not None and _SPEC.loader is not None
narrative = importlib.util.module_from_spec(_SPEC)
# Register before exec: @dataclass resolves its module via sys.modules.
sys.modules[_SPEC.name] = narrative
_SPEC.loader.exec_module(narrative)

_WORKLOADS = ("matmul-1024x1024x1024", "matmul-4096x4096x4096", "softmax-2048x2048")
_MULTIPLIER = {
    "lfbo": 1.0,
    "llm": 0.8,
    "hybrid_lfbo_llm": 0.7,
    "contextual_rag_llm": 0.6,
}


def _campaign(tmp_path: Path, *, drop: tuple[str, str] | None = None) -> Path:
    campaign = tmp_path / "camp"
    analysis = campaign / "analysis"
    analysis.mkdir(parents=True)
    with (analysis / "per_kernel_arm.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "workload",
                "arm",
                "median_selected_latency_ms",
                "median_readiness_seconds",
            ]
        )
        for workload in _WORKLOADS:
            for arm in ARMS:
                if drop == (workload, arm):
                    continue
                writer.writerow(
                    [workload, arm, 10.0 * _MULTIPLIER[arm], 100.0 * _MULTIPLIER[arm]]
                )
    return campaign


def test_short_label_and_family_track_the_workload_id() -> None:
    assert narrative.family_of("matmul_split_k-64x16384x64") == "matmul_split_k"
    assert narrative.short_label("matmul_split_k-64x16384x64") == "K=16384"
    assert narrative.short_label("matmul-4096x4096x4096") == "4096³"
    assert narrative.short_label("attention-2x8x4096x64") == "S=4096"
    assert narrative.short_label("gdn_fwd_h-b1h4s8192ds128") == "s=8192"
    # An unrecognized family degrades to the raw shape rather than raising.
    assert narrative.short_label("brand_new_kernel-7x7") == "7x7"


def test_load_kernels_drops_workloads_missing_an_arm(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path, drop=(_WORKLOADS[0], "llm"))
    kernels = narrative.load_kernels(campaign)
    assert [row.workload for row in kernels] == list(_WORKLOADS[1:])
    assert all(set(row.latency) == set(ARMS) for row in kernels)


def test_missing_analysis_csv_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="analyze_head_to_head"):
        narrative.load_kernels(tmp_path / "never-analyzed")


def test_figures_render_from_synthetic_campaign(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    campaign = _campaign(tmp_path)

    assert narrative.main(["--campaign", str(campaign)]) == 0

    figures = campaign / "figures"
    for stem in ("12_heatmap_latency_vs_llm", "13_family_walltime_vs_latency"):
        for suffix in ("png", "svg"):
            assert (figures / f"{stem}.{suffix}").is_file()
