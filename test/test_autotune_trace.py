from __future__ import annotations

import json
import logging
import math
from types import MethodType
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from helion.autotuner.base_search import BaseSearch
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.benchmark_provider import BenchmarkResult
from helion.autotuner.benchmark_provider import IsolatedBenchmarkFailure
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider
from helion.autotuner.benchmark_provider import MultiShapeBenchmarkProvider
from helion.autotuner.logger import AutotuneLogEntry
from helion.autotuner.logger import AutotuningLogger
from helion.autotuner.logger import _AutotuneTrace
from helion.autotuner.metrics import AutotuneMetrics
from helion.autotuner.metrics import KernelMetadata
from helion.runtime.config import Config
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _entry(log: AutotuningLogger, perf: float, status: str = "ok") -> AutotuneLogEntry:
    config = Config(block_sizes=[32], num_warps=4)
    config_id = log.register_config(config)
    assert config_id is not None
    return AutotuneLogEntry(1, status, perf, None, config_id, config)


def test_trace_streams_shared_clock_across_hybrid_stages(tmp_path: Path) -> None:
    base = tmp_path / "nested" / "autotune"
    path = base.with_suffix(".trace.jsonl")
    settings = Settings(autotune_log=str(base), autotune_log_details=True)
    parent = AutotuningLogger(settings)
    llm = AutotuningLogger(settings)
    lfbo = AutotuningLogger(settings)
    with (
        patch("helion.autotuner.logger.time.perf_counter", return_value=10.0) as clock,
        parent.autotune_tracing("LLMSeededLFBOTreeSearch"),
    ):
        clock.return_value = 15.0
        with llm.autotune_tracing("LLMGuidedSearch"):
            llm.record_autotune_entry(_entry(llm, math.inf, "started"))
            clock.return_value = 20.0
            llm.record_autotune_entry(_entry(llm, 2.0))
            # The successful result is available before the stage/run ends.
            assert _rows(path)[-1]["perf_ms"] == 2.0
        clock.return_value = 30.0
        with lfbo.autotune_tracing("LFBOTreeSearch"):
            lfbo.reset()
            lfbo.record_autotune_entry(_entry(lfbo, math.inf, "timeout"))
            lfbo.record_autotune_entry(_entry(lfbo, 1.0))
            lfbo.record_autotune_entry(_entry(lfbo, 1.0, "deduplicated"))
            lfbo.record_trace_entry(_entry(lfbo, 1.1), event="rebenchmark")
    rows = _rows(path)
    assert len({row["run_id"] for row in rows}) == 1
    assert [row["elapsed_s"] for row in rows] == sorted(
        row["elapsed_s"] for row in rows
    )
    trials = [row for row in rows if row["event"] == "trial"]
    assert [row["trial_index"] for row in trials] == [1, 2, 3, 4]
    assert [row["elapsed_s"] for row in trials] == [10.0, 20.0, 20.0, 20.0]
    assert [row["perf_ms"] for row in trials] == [2.0, None, 1.0, 1.0]
    assert [row["best_perf_ms"] for row in trials] == [2.0, 2.0, 1.0, 1.0]
    assert trials[0]["config"] == {"block_sizes": [32], "num_warps": 4}
    assert rows[-1]["event"] == "run_end"
    assert rows[-1]["status"] == "ok"
    assert rows[-1]["timestamp"].endswith("+00:00")
    assert not base.with_suffix(".csv").exists()

    with parent.autotune_tracing("LFBOTreeSearch"):
        parent.record_autotune_entry(_entry(parent, 3.0))
    appended = _rows(path)[len(rows) :]
    assert appended[0]["run_id"] != rows[0]["run_id"]
    assert appended[-1]["best_perf_ms"] == 3.0


def test_trace_error_closes_run_and_resets_context(tmp_path: Path) -> None:
    base = tmp_path / "autotune"
    path = base.with_suffix(".trace.jsonl")
    log = AutotuningLogger(Settings(autotune_log=str(base), autotune_log_details=True))
    with (
        pytest.raises(RuntimeError, match="compile failure"),
        log.autotune_tracing("LLMGuidedSearch"),
    ):
        log.record_autotune_entry(_entry(log, math.inf, "error"))
        raise RuntimeError("compile failure")
    assert _rows(path)[-1]["status"] == "error"
    assert log.register_config(Config()) is None
    with log.autotune_tracing("LFBOTreeSearch"):
        log.record_autotune_entry(_entry(log, 2.0))
    assert len({row["run_id"] for row in _rows(path)}) == 2


@pytest.mark.parametrize("user_filter", [False, True])
def test_trace_records_backend_and_user_filtered_configs(
    tmp_path: Path, user_filter: bool
) -> None:
    base = tmp_path / "filtered"
    configs = [Config(block_sizes=[size]) for size in (16, 32, 64)]
    settings = Settings(
        autotune_log=str(base),
        autotune_log_details=True,
        autotune_config_filter=(
            (lambda config: None if config == configs[0] else config)
            if user_filter
            else None
        ),
    )
    log = AutotuningLogger(settings)
    backend_filter = Mock(side_effect=lambda config: config != configs[1])
    search = SimpleNamespace(
        settings=settings,
        log=log,
        _autotune_metrics=AutotuneMetrics(),
        performance_unit="ms",
        _backend_config_is_viable=backend_filter,
    )
    with log.autotune_tracing("LFBOTreeSearch"):
        passing, indices = BaseSearch._apply_config_filter(search, configs)
    assert indices == ([2] if user_filter else [0, 2])
    assert passing == [configs[index] for index in indices]
    backend_inputs = configs[1:] if user_filter else configs
    assert [call.args[0] for call in backend_filter.call_args_list] == backend_inputs
    trials = [
        row
        for row in _rows(base.with_suffix(".trace.jsonl"))
        if row["event"] == "trial"
    ]
    assert [row["config"] for row in trials] == [
        config.config for index, config in enumerate(configs) if index not in indices
    ]
    assert all(row["status"] == "filtered" and row["perf_ms"] is None for row in trials)


@pytest.mark.parametrize("deduplicate", [False, True])
def test_provider_traces_measured_failed_and_aliased_configs(
    tmp_path: Path, deduplicate: bool
) -> None:
    base = tmp_path / "provider"
    path = base.with_suffix(".trace.jsonl")
    settings = Settings(
        autotune_log=str(base),
        autotune_log_details=True,
        autotune_precompile=None,
        autotune_progress_bar=False,
        autotune_log_level=logging.CRITICAL,
    )
    log = AutotuningLogger(settings)
    configs = [Config(block_sizes=[size]) for size in (16, 32, 64, 128)]
    functions = [SimpleNamespace(source_hash=source) for source in ("a", "a", "b")]
    kernel = SimpleNamespace(
        compile_config=Mock(side_effect=[RuntimeError("compile failure"), *functions]),
        format_kernel_decorator=Mock(return_value="kernel()"),
        env=SimpleNamespace(process_group_name=None),
    )
    spec = SimpleNamespace(
        backend=SimpleNamespace(
            should_deduplicate_generated_sources=lambda spec: deduplicate,
            generated_source_hash=lambda fn: fn.source_hash,
        ),
        compiler_seed_timeout_retry_repetitions=None,
    )
    with (
        patch.object(
            LocalBenchmarkProvider, "_compute_baseline", return_value=(None, (), None)
        ),
        patch.object(
            LocalBenchmarkProvider,
            "_compute_effective_tolerances",
            return_value=(0.0, 0.0),
        ),
        patch.object(LocalBenchmarkProvider, "_decide_num_jobs", return_value=1),
        patch("helion.autotuner.benchmark_provider.maybe_dump_triton_failure"),
    ):
        provider = LocalBenchmarkProvider(
            kernel, settings, spec, (), log, AutotuneMetrics()
        )

        def benchmark(config: Config, fn: object) -> float:
            if config == configs[-1]:
                provider._last_benchmark_failure_status = "timeout"
                return math.inf
            return 1.25

        with (
            log.autotune_tracing("LFBOTreeSearch"),
            patch.object(provider, "_benchmark_function", side_effect=benchmark),
        ):
            results = provider.benchmark(configs)
    trials = [row for row in _rows(path) if row["event"] == "trial"]
    assert len(trials) == 4
    by_config = {tuple(row["config"]["block_sizes"]): row for row in trials}
    for result in results:
        row = by_config[tuple(result.config["block_sizes"])]
        assert row["status"] == result.status
        assert row["perf_ms"] == (result.perf if math.isfinite(result.perf) else None)
    assert trials[0]["status"] == "error"
    assert by_config[(64,)]["status"] == ("deduplicated" if deduplicate else "ok")
    assert by_config[(128,)]["status"] == "timeout"


def test_trace_uses_existing_log_details_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "environment.log"
    monkeypatch.setenv("HELION_AUTOTUNE_LOG", str(base))
    monkeypatch.setenv("HELION_AUTOTUNE_LOG_DETAILS", "1")
    settings = Settings()
    assert settings.autotune_log == str(base)
    assert settings.autotune_log_details
    log = AutotuningLogger(settings)
    with log.autotune_tracing("LLMGuidedSearch"):
        log.record_autotune_entry(_entry(log, 1.0))
    rows = _rows(base.with_suffix(".trace.jsonl"))
    assert rows[-1]["event"] == "run_end"
    assert rows[-1]["best_perf_ms"] == 1.0


def test_rebenchmark_trace_does_not_report_retained_timing_as_measured(
    tmp_path: Path,
) -> None:
    base = tmp_path / "rebenchmark"
    path = base.with_suffix(".trace.jsonl")
    log = AutotuningLogger(Settings(autotune_log=str(base), autotune_log_details=True))
    members = [
        PopulationMember(
            fn=lambda: None,
            perfs=[1.5],
            flat_values=[],
            config=Config(block_sizes=[size]),
        )
        for size in (32, 64, 128)
    ]
    search = SimpleNamespace(
        log=log,
        performance_unit="ms",
        best_perf_so_far=1.5,
        _autotune_metrics=AutotuneMetrics(),
        _invalidate_cute_flash_rebenchmark_failures=Mock(return_value=[]),
        _refresh_benchmarked_members_after_rebenchmark=Mock(),
    )
    with log.autotune_tracing("LFBOTreeSearch"):
        PopulationBasedSearch._apply_rebenchmark_timings(
            search,
            members,
            [1.5, 1.5, 0.8],
            trace_results=[None, IsolatedBenchmarkFailure("timeout"), 0.8],
        )
    rows = [row for row in _rows(path) if row["event"] == "rebenchmark"]
    assert [row["status"] for row in rows] == ["timeout", "ok"]
    assert [row["perf_ms"] for row in rows] == [None, 0.8]
    assert members[0].perf == 1.5


@pytest.mark.parametrize("relative_to", [None, "baseline"])
def test_multi_shape_rebenchmark_traces_only_fresh_results(
    tmp_path: Path, relative_to: str | None
) -> None:
    base = tmp_path / "multi-shape"
    path = base.with_suffix(".trace.jsonl")
    log = AutotuningLogger(Settings(autotune_log=str(base), autotune_log_details=True))
    configs = [Config(block_sizes=[size]) for size in (32, 64, 128)]
    objective = 0.75 if relative_to else 12.0
    results = [
        BenchmarkResult(configs[0], lambda: None, objective, "ok", 0.1),
        BenchmarkResult(configs[1], lambda: None, math.inf, "timeout", 0.2),
        BenchmarkResult(configs[2], lambda: None, math.inf, "error", None),
    ]
    metrics = AutotuneMetrics()
    provider = object.__new__(MultiShapeBenchmarkProvider)
    provider.log = log
    provider.args = SimpleNamespace(relative_to=relative_to)
    provider._autotune_metrics = metrics
    provider.budget_exceeded_fn = Mock(return_value=False)
    provider._benchmark = Mock(return_value=results)
    provider.raw_latency = Mock(return_value=12.0)
    members = [PopulationMember(lambda: None, [20.0], [], c) for c in configs]
    search = SimpleNamespace(
        log=log,
        benchmark_provider=provider,
        performance_unit="ratio" if relative_to else "ms",
        best_perf_so_far=20.0,
        _autotune_metrics=metrics,
        _invalidate_cute_flash_rebenchmark_failures=Mock(return_value=[]),
        _refresh_benchmarked_members_after_rebenchmark=Mock(),
    )
    search._apply_rebenchmark_timings = MethodType(
        PopulationBasedSearch._apply_rebenchmark_timings, search
    )
    with log.autotune_tracing("LFBOTreeSearch"):
        PopulationBasedSearch.rebenchmark(search, members)
        provider.budget_exceeded_fn.return_value = True
        PopulationBasedSearch.rebenchmark(search, members)

    rows = [row for row in _rows(path) if row["event"] == "rebenchmark"]
    assert [row["status"] for row in rows] == ["ok", "timeout", "error"]
    assert [row["perf_ms"] for row in rows] == [12.0, None, None]
    assert [row["objective"] for row in rows] == [objective, None, None]
    assert {row["objective_unit"] for row in rows} == {"ratio" if relative_to else "ms"}
    assert [member.perf for member in members] == [objective, math.inf, math.inf]
    assert metrics.num_configs_tested == 0
    provider._benchmark.assert_called_once_with(
        configs, desc="Rebenchmarking", record_results=False, check_budget=False
    )
    assert provider.budget_exceeded_fn.call_count == 2
    provider.raw_latency.assert_called_once_with(configs[0])


@pytest.mark.parametrize(
    ("details", "log_path"),
    [
        (None, None),
        (None, "/tmp/disabled"),
        ("0", "/tmp/disabled"),
        ("1", None),
        ("1", ""),
    ],
)
def test_trace_requires_log_details_and_log_path(
    monkeypatch: pytest.MonkeyPatch, details: str | None, log_path: str | None
) -> None:
    for name, value in (
        ("HELION_AUTOTUNE_LOG_DETAILS", details),
        ("HELION_AUTOTUNE_LOG", log_path),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    settings = Settings()
    assert settings.autotune_log_details == (details == "1")
    assert settings.autotune_log == (log_path or None)
    log = AutotuningLogger(settings)
    config = Config(block_sizes=[32])
    with (
        patch("helion.autotuner.logger.Path.mkdir") as mkdir,
        patch("helion.autotuner.logger.Path.open") as open_file,
        patch("helion.autotuner.logger.canonical_config_id") as config_id,
        log.autotune_tracing("LLMGuidedSearch"),
    ):
        assert not log.trace_enabled
        assert log.register_config(config) is None
        log.record_selected_config(config)
        log.record_autotune_entry(AutotuneLogEntry(0, "ok", 1.0, None, "id", config))
        assert log.record_llm_request(messages=[]) is None
        log.record_llm_response(None, response_text="unused")
    mkdir.assert_not_called()
    open_file.assert_not_called()
    config_id.assert_not_called()


def test_non_master_rank_does_not_open_trace(tmp_path: Path) -> None:
    base = tmp_path / "rank1" / "autotune"
    log = AutotuningLogger(Settings(autotune_log=str(base), autotune_log_details=True))
    with (
        patch("helion.autotuner.logger.is_master_rank", return_value=False),
        log.autotune_tracing("LFBOTreeSearch"),
    ):
        assert not log.trace_enabled
        assert log.register_config(Config()) is None
    assert not base.parent.exists()


@pytest.mark.parametrize(
    "failing_event", ["run_start", "stage_start", "stage_end", "run_end"]
)
def test_trace_write_failure_restores_logger_and_context(
    tmp_path: Path, failing_event: str
) -> None:
    base = tmp_path / "write-failure"
    path = base.with_suffix(".trace.jsonl")
    log = AutotuningLogger(Settings(autotune_log=str(base), autotune_log_details=True))
    original_record = _AutotuneTrace.record

    def fail_write(
        trace: _AutotuneTrace, event: str, algorithm: str, **fields: object
    ) -> None:
        if event == failing_event:
            raise OSError("simulated storage failure")
        original_record(trace, event, algorithm, **fields)

    with (
        patch.object(_AutotuneTrace, "record", fail_write),
        pytest.raises(OSError, match="simulated storage failure"),
        log.autotune_tracing("LFBOTreeSearch"),
    ):
        log.record_autotune_entry(_entry(log, 1.0))
    assert not log.trace_enabled
    assert log.register_config(Config()) is None

    # Reusing the logger must open a new run, rather than inherit the failed
    # context's closed file or its historical best measurement.
    with log.autotune_tracing("LFBOTreeSearch"):
        log.record_autotune_entry(_entry(log, 2.0))
    assert _rows(path)[-1]["best_perf_ms"] == 2.0


def test_trace_records_returned_config_independently_of_historical_best(
    tmp_path: Path,
) -> None:
    base = tmp_path / "selection"
    path = base.with_suffix(".trace.jsonl")
    log = AutotuningLogger(Settings(autotune_log=str(base), autotune_log_details=True))
    returned_config = Config(block_sizes=[64])

    def autotune_with_logging(*, skip_cache: bool) -> Config:
        assert skip_cache
        log.record_autotune_entry(_entry(log, 0.5))
        return returned_config

    search = SimpleNamespace(log=log, _autotune_with_logging=autotune_with_logging)
    assert BaseSearch.autotune(search, skip_cache=True) == returned_config
    rows = _rows(path)
    selected = [row for row in rows if row["event"] == "selected"]
    assert len(selected) == 1
    assert selected[0]["config"] == returned_config.config
    assert selected[0]["best_perf_ms"] == 0.5
    assert "perf_ms" not in selected[0]
    assert rows[-1]["event"] == "run_end"


def test_late_llm_response_cannot_write_to_closed_or_next_run(tmp_path: Path) -> None:
    base = tmp_path / "late-response"
    path = base.with_suffix(".trace.jsonl")
    log = AutotuningLogger(Settings(autotune_log=str(base), autotune_log_details=True))
    with log.autotune_tracing("LLMGuidedSearch"):
        old_request = log.record_llm_request(messages=[])
    # An in-flight provider call can outlive an aborted search. It must neither
    # write to the closed file nor contaminate a later run using the same logger.
    log.record_llm_response(old_request, response="after-close")
    with log.autotune_tracing("LLMGuidedSearch"):
        log.record_llm_response(old_request, response="wrong-run")
        new_request = log.record_llm_request(messages=[])
        log.record_llm_response(new_request, response="current-run")
    rows = _rows(path)
    responses = [row for row in rows if row["event"] == "llm_response"]
    assert len(responses) == 1
    assert responses[0]["response"] == "current-run"
    assert responses[0]["run_id"] == rows[-1]["run_id"]


def test_log_without_details_still_writes_csv_without_trace(tmp_path: Path) -> None:
    base = tmp_path / "ordinary-log"
    log = AutotuningLogger(Settings(autotune_log=str(base), autotune_log_details=False))
    with log.autotune_tracing("LFBOTreeSearch"), log.autotune_logging():
        log.record_autotune_entry(_entry(log, 1.25))
    assert "1.250000" in base.with_suffix(".csv").read_text()
    assert base.with_suffix(".log").exists()
    assert not base.with_suffix(".trace.jsonl").exists()
    assert not base.with_suffix(".meta.jsonl").exists()


@pytest.mark.parametrize("restricted", [False, True])
def test_search_trace_includes_restricted_runs_without_collecting_dataset(
    tmp_path: Path, restricted: bool
) -> None:
    base = tmp_path / "search"
    search = object.__new__(BaseSearch)
    search.settings = Settings(
        autotune_log=str(base),
        autotune_log_details=True,
        autotune_log_level=logging.CRITICAL,
    )
    search.log = AutotuningLogger(search.settings)
    search.args = ()
    config = Config(block_sizes=[32], num_warps=4)
    search.kernel = Mock()
    search.kernel.kernel = SimpleNamespace(configs=[config] if restricted else [])
    search.kernel.format_kernel_decorator.return_value = "@helion.kernel(...)"
    search.kernel.get_cached_path.return_value = None
    search.benchmark_provider = SimpleNamespace(setup=Mock(), cleanup=Mock())
    search._prepare = Mock()
    search._kernel_metadata = KernelMetadata(kernel_name="test_search")
    search._autotune_metrics = AutotuneMetrics()
    search._finalize_autotune_metrics = Mock()

    def autotune() -> Config:
        search.log.record_autotune_entry(_entry(search.log, 1.25))
        return config

    search._autotune = autotune
    assert search.autotune() == config
    rows = _rows(base.with_suffix(".trace.jsonl"))
    assert [row["perf_ms"] for row in rows if row["event"] == "trial"] == [1.25]
    assert (
        next(row for row in rows if row["event"] == "selected")["config"]
        == config.config
    )
    assert base.with_suffix(".csv").exists()
    assert base.with_suffix(".meta.jsonl").exists() is not restricted
