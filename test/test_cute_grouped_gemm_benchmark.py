from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import sys
from typing import Any

import pytest


def _load_grouped_gemm_benchmark() -> Any:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "cute"
        / "compare_grouped_gemm_backends.py"
    )
    spec = importlib.util.spec_from_file_location(
        "helion_benchmarks_cute_compare_grouped_gemm_backends_test",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_saved_grouped_gemm_cases_and_shape_labels_are_available() -> None:
    benchmark = _load_grouped_gemm_benchmark()

    assert benchmark.SAVED_CASE_NAMES == (
        "default_small_parity",
        "doc_no_mn_tail",
        "doc_no_mn_g3_192_extra_full",
        "doc_original",
        "doc_mtail_g1_g3_192_extra_full",
        "doc_mtail_g1_only",
        "doc_ntail_g3_160",
    )
    assert len(benchmark.SAVED_CASES) == 7
    assert "informational regression references" in benchmark.SAVED_RESULT_NOTE

    for case_name in benchmark.SAVED_CASE_NAMES:
        case = benchmark.CASES_BY_NAME[case_name]
        payload = benchmark.case_to_dict(case)
        assert payload["label"]
        assert payload["shape_label"].startswith("G")
        assert payload["problem_sizes"]
        assert payload["shape_stats"]["total_ctas"] > 0
        assert payload["saved_helion_over_cutlass"] is not None


def test_grouped_gemm_benchmark_cli_build_is_lightweight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked_roots = {"cutlass", "helion", "torch", "triton"}
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if level == 0 and name.split(".", 1)[0] in blocked_roots:
            raise AssertionError(
                f"runtime dependency imported during CLI build: {name}"
            )
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    benchmark = _load_grouped_gemm_benchmark()
    parser = benchmark.build_arg_parser()
    args = parser.parse_args(["--list-cases", "--json"])

    assert args.list_cases
    assert benchmark.selected_cases(args) == benchmark.SAVED_CASES


def test_grouped_gemm_ratio_summary_uses_synthetic_rows() -> None:
    benchmark = _load_grouped_gemm_benchmark()

    rows = [
        {"status": "ok", "retained_over_cutlass": 0.5},
        {"status": "ok", "retained_over_cutlass": 2.0},
        {"status": "missing", "retained_over_cutlass": 100.0},
        {"status": "ok", "retained_over_cutlass": None},
    ]
    summary = benchmark.ratio_summary(rows)

    assert summary["count"] == 2
    assert summary["wins"] == 1
    assert summary["best"] == 0.5
    assert summary["worst"] == 2.0
    assert summary["geomean"] == pytest.approx(1.0)
