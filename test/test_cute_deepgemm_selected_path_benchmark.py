from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch


def _load_selected_path_benchmark() -> Any:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "cute"
        / "deepgemm_selected_path.py"
    )
    spec = importlib.util.spec_from_file_location(
        "helion_benchmarks_cute_deepgemm_selected_path_test",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_deepgemm_selected_path_defaults_to_production_selected() -> None:
    benchmark = _load_selected_path_benchmark()

    parser = benchmark.build_arg_parser()
    default_args = parser.parse_args([])

    assert benchmark.HELION_KERNEL_SOURCES == (
        benchmark.HELION_KERNEL_SOURCE_PRODUCTION_SELECTED,
    )
    assert (
        default_args.helion_kernel_source
        == benchmark.HELION_KERNEL_SOURCE_PRODUCTION_SELECTED
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--helion-kernel-source", "diagnostic"])


def test_production_selected_kernel_binding_times_timed_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = _load_selected_path_benchmark()
    a_packed = torch.empty((1, 64), dtype=torch.bfloat16)
    b_grouped = torch.empty((1, 128, 64), dtype=torch.bfloat16)
    grouped_layout = torch.zeros(1, dtype=torch.int32)
    work_tile_metadata = torch.empty((1, 4), dtype=torch.int32)
    api_result = torch.empty((1, 128), dtype=torch.bfloat16)
    api_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    route_report_call_api_counts: list[int] = []

    class FakeDeepGemmMGrouped:
        def deepgemm_m_grouped_bf16_gemm_nt_contiguous(
            self,
            actual_a_packed: torch.Tensor,
            actual_b_grouped: torch.Tensor,
            actual_grouped_layout: torch.Tensor,
        ) -> torch.Tensor:
            api_calls.append((actual_a_packed, actual_b_grouped, actual_grouped_layout))
            return api_result

    fake_deepgemm_m_grouped = FakeDeepGemmMGrouped()

    monkeypatch.setattr(
        benchmark,
        "import_helion_deepgemm_m_grouped",
        lambda: fake_deepgemm_m_grouped,
    )

    def fake_route_report(
        actual_a_packed: torch.Tensor,
        actual_b_grouped: torch.Tensor,
        actual_grouped_layout: torch.Tensor,
    ) -> SimpleNamespace:
        assert actual_a_packed is a_packed
        assert actual_b_grouped is b_grouped
        assert actual_grouped_layout is grouped_layout
        route_report_call_api_counts.append(len(api_calls))
        return SimpleNamespace(
            route="generated_selected_segment",
            layout_has_valid_prefix_tiles=False,
            use_generated_segment_kernel=True,
            use_selected_segment_kernel=True,
            selected_segment_work_tile_metadata=work_tile_metadata,
        )

    monkeypatch.setattr(
        benchmark,
        "deepgemm_m_grouped_bf16_gemm_nt_contiguous_route_report",
        fake_route_report,
    )

    binding = benchmark.production_selected_kernel_binding(
        a_packed,
        b_grouped,
        grouped_layout,
    )

    assert binding.work_tile_metadata is work_tile_metadata
    assert binding.call() is api_result
    assert api_calls == [(a_packed, b_grouped, grouped_layout)]
    assert route_report_call_api_counts == [0, 1]
    assert binding.details == {
        "helion_kernel_source": benchmark.HELION_KERNEL_SOURCE_PRODUCTION_SELECTED,
        "route": "generated_selected_segment",
        "route_assertions": {
            "layout_has_valid_prefix_tiles": False,
            "use_generated_segment_kernel": True,
            "use_selected_segment_kernel": True,
            "selected_metadata_nonempty": True,
            "route": "generated_selected_segment",
        },
        "route_retention_check": "after_each_timed_api_call",
        "route_assertions_after_last_call": {
            "layout_has_valid_prefix_tiles": False,
            "use_generated_segment_kernel": True,
            "use_selected_segment_kernel": True,
            "selected_metadata_nonempty": True,
            "route": "generated_selected_segment",
        },
        "work_tile_metadata": {
            "shape": [1, 4],
            "dtype": "torch.int32",
            "device": "cpu",
            "rows": 1,
            "values": work_tile_metadata.tolist(),
        },
        "kernel": (
            "helion._compiler.cute.deepgemm_m_grouped."
            "deepgemm_m_grouped_bf16_gemm_nt_contiguous"
        ),
        "timed_api": (
            "helion._compiler.cute.deepgemm_m_grouped."
            "deepgemm_m_grouped_bf16_gemm_nt_contiguous"
        ),
    }


def test_production_selected_kernel_binding_fails_when_timed_api_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = _load_selected_path_benchmark()
    a_packed = torch.empty((1, 64), dtype=torch.bfloat16)
    b_grouped = torch.empty((1, 128, 64), dtype=torch.bfloat16)
    grouped_layout = torch.zeros(1, dtype=torch.int32)
    work_tile_metadata = torch.empty((1, 4), dtype=torch.int32)
    api_result = torch.empty((1, 128), dtype=torch.bfloat16)
    api_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    route_report_call_api_counts: list[int] = []

    class FakeDeepGemmMGrouped:
        def deepgemm_m_grouped_bf16_gemm_nt_contiguous(
            self,
            actual_a_packed: torch.Tensor,
            actual_b_grouped: torch.Tensor,
            actual_grouped_layout: torch.Tensor,
        ) -> torch.Tensor:
            api_calls.append((actual_a_packed, actual_b_grouped, actual_grouped_layout))
            return api_result

    fake_deepgemm_m_grouped = FakeDeepGemmMGrouped()
    monkeypatch.setattr(
        benchmark,
        "import_helion_deepgemm_m_grouped",
        lambda: fake_deepgemm_m_grouped,
    )

    def fake_route_report(
        actual_a_packed: torch.Tensor,
        actual_b_grouped: torch.Tensor,
        actual_grouped_layout: torch.Tensor,
    ) -> SimpleNamespace:
        assert actual_a_packed is a_packed
        assert actual_b_grouped is b_grouped
        assert actual_grouped_layout is grouped_layout
        route_report_call_api_counts.append(len(api_calls))
        use_selected_segment_kernel = not api_calls
        route = (
            "generated_selected_segment"
            if use_selected_segment_kernel
            else "generated_segment"
        )
        return SimpleNamespace(
            route=route,
            layout_has_valid_prefix_tiles=False,
            use_generated_segment_kernel=True,
            use_selected_segment_kernel=use_selected_segment_kernel,
            selected_segment_work_tile_metadata=work_tile_metadata,
        )

    monkeypatch.setattr(
        benchmark,
        "deepgemm_m_grouped_bf16_gemm_nt_contiguous_route_report",
        fake_route_report,
    )

    binding = benchmark.production_selected_kernel_binding(
        a_packed,
        b_grouped,
        grouped_layout,
    )
    with pytest.raises(
        benchmark.BenchmarkSetupError,
        match="did not retain route generated_selected_segment after timed API call",
    ):
        binding.call()

    assert api_calls == [(a_packed, b_grouped, grouped_layout)]
    assert route_report_call_api_counts == [0, 1]


def test_production_selected_kernel_binding_reports_unselected_route_on_cpu() -> None:
    benchmark = _load_selected_path_benchmark()

    a_packed = torch.empty((224, 64), dtype=torch.bfloat16)
    b_grouped = torch.empty((1, 128, 64), dtype=torch.bfloat16)
    grouped_layout = torch.zeros(224, dtype=torch.int32)

    with pytest.raises(
        benchmark.BenchmarkSetupError,
        match="production-selected mode requires route generated_selected_segment",
    ):
        benchmark.production_selected_kernel_binding(
            a_packed,
            b_grouped,
            grouped_layout,
        )


def test_blackwell_captured_graph_launch_retention_is_unbounded() -> None:
    grouped_deepgemm = pytest.importorskip("helion._compiler.cute.grouped_deepgemm")
    retained = grouped_deepgemm._BLACKWELL_GROUPED_CAPTURED_GRAPH_LAUNCHES
    saved_retained = retained.copy()
    try:
        retained.clear()
        for _ in range(80):
            compiled: Any = object()
            buffers: Any = object()
            grouped_deepgemm._retain_blackwell_grouped_captured_launch(
                compiled,
                buffers,
            )
        assert len(retained) == 80
    finally:
        retained.clear()
        retained.update(saved_retained)
