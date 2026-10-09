from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from helion._compiler.backend import Backend
from helion.autotuner.base_cache import AutotuneCacheBase
from helion.autotuner.base_search import BaseAutotuner
from helion.autotuner.base_search import BaseSearch
import helion.autotuner.handoff as handoff
from helion.autotuner.handoff import HandoffCandidate
from helion.autotuner.handoff import HandoffPoint
from helion.autotuner.handoff import HandoffProgress
from helion.autotuner.handoff_agent import HandoffAgentResult
from helion.autotuner.handoff_bundle import HandoffBundle
from helion.autotuner.handoff_evaluation import HandoffEvaluation
import helion.autotuner.handoff_pipeline as pipeline
from helion.autotuner.logger import AutotuneLogEntry
from helion.autotuner.logger import AutotuningLogger
from helion.runtime.config import Config
from helion.runtime.settings import Settings


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    settings = Settings(
        autotune_handoff=True,
        autotune_handoff_budget_seconds=90,
        autotune_budget_seconds=30,
        autotune_log=str(tmp_path / "tuning.log"),
        autotune_log_details=True,
        autotune_log_level=0,
    )
    ordinary = Config(block_sizes=[16])
    selected = Config(block_sizes=[32])
    search = Mock(spec=BaseSearch)
    search.settings = settings
    search.log = AutotuningLogger(settings)
    search.args = ()
    search.performance_unit = "ms"
    search.kernel = SimpleNamespace(
        kernel=SimpleNamespace(name="kernel"),
        env=SimpleNamespace(process_group_name=None),
        compile_config=Mock(return_value=lambda: None),
    )
    search.autotune.return_value = ordinary
    cached = Mock(spec=AutotuneCacheBase)
    cached.autotuner = search
    cached.autotune.return_value = ordinary
    finalist = HandoffCandidate(selected, "source", (10.0, 10.0, 10.0), 10.0, 0.0)
    point = HandoffPoint(
        "time",
        selected,
        lambda: None,
        HandoffProgress(4, 2, 30.0, 10.0),
        (finalist,),
        (),
        "ms",
        None,
    )
    evaluation = HandoffEvaluation(
        True, 0.1, "ms", (), str(tmp_path / "evaluation"), "2026-10-05T00:00:00Z"
    )
    bundle = Mock(spec=HandoffBundle)
    bundle.manifest = {"baseline_evaluation": {"ok": True, "perf": 10.0, "unit": "ms"}}
    bundle.run_agent_rounds.return_value = HandoffAgentResult(
        evaluation, evaluation, (), "deadline", str(tmp_path / "rounds"), 90.0
    )
    paths: list[Path] = []

    def build(autotuner, handoff_point, directory, **kwargs):
        path = Path(directory)
        # Match build_handoff's requirement that every invocation gets a new path.
        path.mkdir(parents=True, exist_ok=False)
        paths.append(path)
        bundle.directory = path
        return bundle

    find = Mock(return_value=point)
    build_mock = Mock(side_effect=build)
    validate_available = pipeline.CLISourceAgent.validate_available
    preflight = Mock(return_value="/mock/codex")
    monkeypatch.setattr(pipeline, "find_handoff", find)
    monkeypatch.setattr(pipeline, "build_handoff", build_mock)
    monkeypatch.setattr(pipeline.CLISourceAgent, "validate_available", preflight)
    return SimpleNamespace(
        settings=settings,
        ordinary=ordinary,
        selected=selected,
        search=search,
        cached=cached,
        point=point,
        bundle=bundle,
        find=find,
        build=build_mock,
        preflight=preflight,
        validate_available=validate_available,
        paths=paths,
    )


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("skip_cache", [False, True])
def test_disabled_handoff_keeps_logging_passive(case, cached, skip_cache):
    case.settings.autotune_handoff = False
    autotuner = case.cached if cached else case.search
    assert (
        pipeline.autotune_with_handoff(autotuner, skip_cache=skip_cache)
        is case.ordinary
    )
    autotuner.autotune.assert_called_once_with(skip_cache=skip_cache)
    case.find.assert_not_called()
    case.build.assert_not_called()
    case.preflight.assert_not_called()
    case.bundle.run_agent_rounds.assert_not_called()
    assert not Path(case.settings.autotune_log).with_suffix(".handoff").exists()


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("agent", ["codex", "claude"])
def test_missing_cli_fails_before_search_or_export(case, cached, agent, monkeypatch):
    monkeypatch.setenv("HELION_HANDOFF_AGENT", agent)
    which = Mock(return_value=None)
    monkeypatch.setattr("helion.autotuner.handoff_cli.shutil.which", which)
    case.preflight.side_effect = case.validate_available
    autotuner = case.cached if cached else case.search
    with pytest.raises(FileNotFoundError, match=f"{agent.title()} CLI"):
        pipeline.autotune_with_handoff(autotuner)
    which.assert_called_once_with(agent)
    case.preflight.assert_called_once_with()
    case.find.assert_not_called()
    case.build.assert_not_called()
    case.bundle.run_agent_rounds.assert_not_called()
    case.cached.autotune.assert_not_called()
    case.search.autotune.assert_not_called()
    assert not case.paths
    assert not Path(case.settings.autotune_log).with_suffix(".trace.jsonl").exists()


@pytest.mark.parametrize("force", [False, True])
def test_backend_disabled_handoff_supports_custom_autotuners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force: bool
):
    selected = Config(block_sizes=[32])
    custom = Mock(spec=BaseAutotuner)
    custom.autotune.return_value = selected
    factory = Mock(return_value=custom)
    settings = Settings(
        autotune_handoff=False,
        autotune_effort="full",
        autotuner_fn=factory,
        autotune_log=str(tmp_path / "ordinary.log"),
        autotune_log_details=True,
        force_autotune=False,
    )
    bound = SimpleNamespace(settings=settings, kernel=SimpleNamespace(configs=[]))
    backend = SimpleNamespace(supports_precompile=lambda: True)
    automatic = Mock(side_effect=AssertionError("handoff is disabled"))
    monkeypatch.setattr(pipeline, "autotune_with_handoff", automatic)
    args = (object(),)
    assert Backend.autotune(backend, bound, args, force=force, extra=42) is selected
    factory.assert_called_once_with(bound, args, extra=42)
    custom.autotune.assert_called_once_with(skip_cache=force)
    automatic.assert_not_called()


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("skip_cache", [False, True])
def test_enabled_handoff_preserves_wrapper_and_returns_config(case, cached, skip_cache):
    autotuner = case.cached if cached else case.search
    result = pipeline.autotune_with_handoff(autotuner, skip_cache=skip_cache)
    assert result is case.selected
    case.preflight.assert_called_once_with()
    assert case.find.call_args.args[0] is autotuner
    assert case.find.call_args.kwargs == {"skip_cache": skip_cache}
    policy = case.find.call_args.args[1]
    assert policy.after_seconds == 30
    assert case.build.call_args.args[:2] == (autotuner, case.point)
    case.bundle.run_agent_rounds.assert_called_once_with(
        budget_seconds=90, log=case.search.log
    )
    # Neither the cache wrapper nor the inner search is run twice by the hook.
    case.cached.autotune.assert_not_called()
    case.search.autotune.assert_not_called()
    case.search.kernel.compile_config.assert_not_called()


def test_no_search_time_limit_waits_for_completion(case):
    case.settings.autotune_budget_seconds = None
    pipeline.autotune_with_handoff(case.search)
    policy = case.find.call_args.args[1]
    assert policy.after_seconds is None


def test_each_handoff_has_a_new_directory_under_log_path(case):
    pipeline.autotune_with_handoff(case.cached)
    pipeline.autotune_with_handoff(case.cached)
    first, second = case.paths
    assert first != second
    assert (
        first.parent
        == second.parent
        == Path(case.settings.autotune_log).with_suffix(".handoff")
    )


def test_handoff_without_log_keeps_a_persistent_artifact_directory(
    case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    case.settings.autotune_log = None
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    pipeline.autotune_with_handoff(case.search)
    assert len(case.paths) == 1
    assert case.paths[0].is_dir()
    assert case.paths[0].is_relative_to(tmp_path)


@pytest.mark.parametrize("stage", ["find", "build", "agent"])
def test_nested_automatic_calls_use_ordinary_autotune(case, stage):
    nested = Mock(spec=BaseSearch)
    nested.settings = case.settings
    nested.autotune.return_value = case.ordinary

    def call_nested(*args, **kwargs):
        assert pipeline.autotune_with_handoff(nested, skip_cache=True) is case.ordinary
        if stage == "find":
            return case.point
        if stage == "build":
            return case.bundle
        return case.bundle.run_agent_rounds.return_value

    target = {
        "find": case.find,
        "build": case.build,
        "agent": case.bundle.run_agent_rounds,
    }[stage]
    target.side_effect = call_nested
    assert pipeline.autotune_with_handoff(case.search) is case.selected
    nested.autotune.assert_called_once_with(skip_cache=True)
    assert case.find.call_count == 1
    case.preflight.assert_called_once_with()


@pytest.mark.parametrize("stage", ["find", "build", "agent"])
@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_failed_workflow_restores_automatic_guard(case, stage, error):
    target = {
        "find": case.find,
        "build": case.build,
        "agent": case.bundle.run_agent_rounds,
    }[stage]
    previous = target.side_effect
    target.side_effect = error("interrupted")
    with pytest.raises(error, match="interrupted"):
        pipeline.autotune_with_handoff(case.cached)
    target.side_effect = previous
    calls = case.find.call_count
    assert pipeline.autotune_with_handoff(case.cached) is case.selected
    assert case.find.call_count == calls + 1
    case.cached.autotune.assert_not_called()


def test_explicit_find_handoff_does_not_start_automatic_pipeline(
    case, monkeypatch: pytest.MonkeyPatch
):
    nested = Mock(spec=BaseSearch)
    nested.settings = case.settings
    nested.autotune.return_value = case.selected
    case.search.autotune.side_effect = lambda **kwargs: pipeline.autotune_with_handoff(
        nested, **kwargs
    )
    monkeypatch.setattr(
        handoff._HandoffSession, "confirm", lambda self, extra: case.point.finalists
    )
    result = handoff.find_handoff(case.search, skip_cache=True)
    assert result.config == case.selected
    nested.autotune.assert_called_once_with(skip_cache=True)
    case.find.assert_not_called()
    case.build.assert_not_called()
    case.preflight.assert_not_called()
    case.bundle.run_agent_rounds.assert_not_called()


def test_handoff_settings_environment_and_explicit_overrides(monkeypatch):
    monkeypatch.delenv("HELION_AUTOTUNE_HANDOFF", raising=False)
    monkeypatch.delenv("HELION_AUTOTUNE_HANDOFF_BUDGET_SECONDS", raising=False)
    defaults = Settings()
    assert not defaults.autotune_handoff
    assert defaults.autotune_handoff_budget_seconds == 1500
    monkeypatch.setenv("HELION_AUTOTUNE_HANDOFF", "1")
    monkeypatch.setenv("HELION_AUTOTUNE_HANDOFF_BUDGET_SECONDS", "123")
    environment = Settings()
    assert environment.autotune_handoff
    assert environment.autotune_handoff_budget_seconds == 123
    explicit = Settings(autotune_handoff=False, autotune_handoff_budget_seconds=45)
    assert not explicit.autotune_handoff
    assert explicit.autotune_handoff_budget_seconds == 45
    assert environment.copy(autotune_handoff_budget_seconds=60).autotune_handoff


def test_native_graph_measurements_do_not_replace_search_best(case):
    log = case.search.log

    def find(autotuner, policy, **kwargs):
        with log.autotune_tracing("LFBOTreeSearch"):
            log.record_autotune_entry(
                AutotuneLogEntry(1, "ok", 2.0, None, "search-config", case.selected)
            )
        return case.point

    def run_rounds(**kwargs):
        log.record_handoff_event(
            "native_round",
            perf_ms=0.1,
            objective=0.1,
            timing="cuda_graph",
            accepted=True,
        )
        return case.bundle.run_agent_rounds.return_value

    case.find.side_effect = find
    case.bundle.run_agent_rounds.side_effect = run_rounds
    assert pipeline.autotune_with_handoff(case.cached) is case.selected
    path = Path(case.settings.autotune_log).with_suffix(".trace.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len({row["run_id"] for row in rows}) == 1
    assert len([row for row in rows if row["event"] == "trial"]) == 1
    (native,) = [row for row in rows if row["event"] == "native_round"]
    assert native["algorithm"] == "native_source"
    assert native["timing"] == "cuda_graph"
    assert native["perf_ms"] == native["objective"] == 0.1
    assert native["best_perf_ms"] == native["best_objective"] == 2.0
    assert rows[-1]["event"] == "run_end"
    assert rows[-1]["best_perf_ms"] == rows[-1]["best_objective"] == 2.0


@pytest.mark.parametrize("details", [False, True])
def test_handoff_chat_preserves_full_raw_bytes_independently_of_details(
    tmp_path: Path, details: bool
):
    transcript = tmp_path / "conversation.jsonl"
    raw = (
        '{"type": "session_meta", "payload": {"id": "session-one"}}\r\n'
        '{"type":"response_item","payload":{"role":"user","text":"µs"}}\n'
        '{"type":"response_item","payload":{"role":"assistant","text":"done"}}'
    ).encode()
    transcript.write_bytes(raw)
    log = AutotuningLogger(
        Settings(
            autotune_log=str(tmp_path / "logs" / "run.log"),
            autotune_log_details=details,
        )
    )
    saved = log.record_handoff_chat(transcript, "session-one")
    assert saved == tmp_path / "logs" / "run.handoff.session-one.chat.jsonl"
    assert saved.read_bytes() == raw
    assert transcript.read_bytes() == raw


def test_handoff_chat_keeps_separate_sessions(tmp_path: Path):
    transcript = tmp_path / "conversation.jsonl"
    log = AutotuningLogger(Settings(autotune_log=str(tmp_path / "run")))
    transcript.write_bytes(b'{"session":"first"}\n')
    first = log.record_handoff_chat(transcript, "first")
    transcript.write_bytes(b'{"session":"second"}\n')
    second = log.record_handoff_chat(transcript, "second")
    assert first == tmp_path / "run.handoff.first.chat.jsonl"
    assert second == tmp_path / "run.handoff.second.chat.jsonl"
    assert first.read_bytes() == b'{"session":"first"}\n'
    assert second.read_bytes() == b'{"session":"second"}\n'


def test_handoff_chat_without_log_path_creates_no_artifact(tmp_path: Path):
    transcript = tmp_path / "conversation.jsonl"
    transcript.write_bytes(b'{"type":"response_item"}\n')
    log = AutotuningLogger(Settings(autotune_log=None, autotune_log_details=True))
    assert log.record_handoff_chat(transcript, "session-one") is None
    assert list(tmp_path.iterdir()) == [transcript]
