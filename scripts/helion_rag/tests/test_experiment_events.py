from __future__ import annotations

import dataclasses
import importlib
import json
import math
import os

import pytest

from helion.autotuner.rag.types import FallbackReason
from helion.autotuner.rag.types import LookupTier
from helion.autotuner.rag.types import Phase
from helion.autotuner.rag.types import TunerMode

from helion_rag.experiment.events import ArtifactIdentity
from helion_rag.experiment.events import AttemptAccountingRecord
from helion_rag.experiment.events import EvaluationRecord
from helion_rag.experiment.events import InstrumentationEvent
from helion_rag.experiment.events import NeighborRetrievalRecord
from helion_rag.experiment.events import OutcomeRecord
from helion_rag.experiment.events import PhaseSnapshotRecord
from helion_rag.experiment.events import PhaseTimings
from helion_rag.experiment.events import ProviderRecord
from helion_rag.experiment.events import RetrievalRecord
from helion_rag.experiment.events import RunIdentity
from helion_rag.experiment.events import event_to_canonical_json
from helion_rag.experiment.events import append_event

events_module = importlib.import_module("helion_rag.experiment.events")


def _run(**overrides) -> RunIdentity:
    fields = dict(
        workload_id="w0",
        kernel_name="matmul",
        arm_id="qwen",
        treatment_id="t-qwen",
        tuner_mode=TunerMode.LLM,
        repetition=1,
        random_seed=1234,
    )
    fields.update(overrides)
    return RunIdentity(**fields)


def _full_event() -> InstrumentationEvent:
    return InstrumentationEvent(
        run=_run(),
        artifacts=ArtifactIdentity(
            code_id="c1",
            corpus_id="corp1",
            model_id="qwen3",
            tokenizer_id="tok1",
            index_id="idx1",
            manifest_id="man1",
            runtime_id="triton-dev",
            driver_id="drv1",
            hardware_id="h100",
        ),
        retrieval=RetrievalRecord(
            lookup_tier=LookupTier.SEMANTIC,
            tier0_identity_combo="S4",
            tier0_collision_count=0,
            neighbor_provenance="run-abc",
            raw_semantic_score=0.91,
            raw_shape_score=1.5,
            combined_score=0.42,
            rank_before_rerank=3,
            rank_after_rerank=0,
            selected_configs=("cfg-a", "cfg-b"),
            neighbors=(
                NeighborRetrievalRecord(
                    provenance="run-abc",
                    config_id="cfg-a",
                    selected_config={"block_sizes": [32]},
                    raw_semantic_score=0.91,
                    raw_shape_score=1.5,
                    combined_score=0.42,
                    rank_before_rerank=3,
                    rank_after_rerank=0,
                ),
                NeighborRetrievalRecord(
                    provenance="run-def",
                    config_id="cfg-b",
                    selected_config={"block_sizes": [64]},
                    raw_semantic_score=0.9,
                    raw_shape_score=2.0,
                    combined_score=0.5,
                    rank_before_rerank=1,
                    rank_after_rerank=1,
                ),
            ),
            candidate_source="qwen",
            compatibility_status="compatible",
        ),
        provider=ProviderRecord(
            request_id="req1",
            response_id="resp1",
            cache_state="miss",
            input_tokens=100,
            cached_input_tokens=10,
            output_tokens=50,
            reasoning_tokens=25,
        ),
        outcome=OutcomeRecord(
            decision_name="ContextualSearch",
            compilation_status="ok",
            correctness_status="ok",
            benchmark_status="ok",
            timeout_status="none",
            fallback_reason=None,
            accumulated_seconds=12.5,
            censored=False,
            incumbent_best_perf=0.8,
            selected_performance=0.75,
        ),
        timings=PhaseTimings(
            lookup=0.1,
            embedding=0.2,
            decision=0.05,
            generation=1.0,
            provider=2.5,
            seeding=0.3,
            compilation=2.0,
            correctness=1.5,
            benchmarking=3.0,
            fallback=0.0,
            total=8.15,
            readiness_provider_inclusive=8.15,
            readiness_provider_exclusive=5.65,
        ),
        evaluations=(
            EvaluationRecord(
                sequence=1,
                trajectory_sequence=1,
                config_id="cfg-a",
                config_repr="Config(block_size=32)",
                candidate_source="retrieval",
                candidate_category="seed",
                compatibility_status="compatible",
                compilation_status="ok",
                compilation_seconds=0.25,
                correctness_status="ok",
                benchmark_status="ok",
                timeout_status="not_timed_out",
                performance=0.8,
                elapsed_seconds=3.5,
                incumbent_best_perf=0.8,
            ),
        ),
        phase_snapshots=(
            PhaseSnapshotRecord(
                trajectory_sequence=2,
                phase=Phase.BENCHMARKING,
                elapsed_seconds=3.5,
                phase_seconds=1.25,
                incumbent_best_perf=0.8,
            ),
        ),
    )


def _sparse_event() -> InstrumentationEvent:
    return InstrumentationEvent(
        run=_run(arm_id="cold", treatment_id="t-cold", tuner_mode=TunerMode.LFBO),
        artifacts=ArtifactIdentity(),
        retrieval=RetrievalRecord(),
        provider=ProviderRecord(),
        outcome=OutcomeRecord(fallback_reason=FallbackReason.PROVIDER_FAILURE),
        timings=PhaseTimings(),
    )


def test_canonical_json_is_deterministic():
    event = _full_event()
    assert event_to_canonical_json(event) == event_to_canonical_json(event)


def test_canonical_json_sorted_and_compact():
    raw = event_to_canonical_json(_full_event())
    text = raw.decode("utf-8")
    assert " " not in text  # compact separators, no whitespace
    parsed = json.loads(raw)
    # top-level record groups are present and keys are globally sorted.
    assert list(parsed) == sorted(parsed)
    assert set(parsed) == {
        "accounting",
        "artifacts",
        "evaluations",
        "hybrid_stage_breakdown",
        "outcome",
        "phase_snapshots",
        "provider",
        "retrieval",
        "run",
        "timings",
    }


def test_enums_serialize_by_value():
    parsed = json.loads(event_to_canonical_json(_full_event()))
    assert parsed["run"]["tuner_mode"] == "llm"
    assert parsed["retrieval"]["lookup_tier"] == 1  # LookupTier.SEMANTIC value
    sparse = json.loads(event_to_canonical_json(_sparse_event()))
    assert sparse["outcome"]["fallback_reason"] == "provider_failure"
    assert sparse["run"]["tuner_mode"] == "lfbo"
    assert sparse["retrieval"]["lookup_tier"] is None
    assert parsed["outcome"]["decision_name"] == "ContextualSearch"


def test_stable_across_dict_key_ordering():
    canonical = event_to_canonical_json(_full_event())
    reparsed = json.loads(canonical)
    # Re-dumping the parsed structure with sorted keys reproduces the bytes,
    # independent of any dict insertion order.
    reserialized = json.dumps(reparsed, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    assert reserialized == canonical


def test_round_trip_structure():
    parsed = json.loads(event_to_canonical_json(_sparse_event()))
    assert parsed["run"] == {
        "arm_id": "cold",
        "kernel_name": "matmul",
        "random_seed": 1234,
        "repetition": 1,
        "treatment_id": "t-cold",
        "tuner_mode": "lfbo",
        "workload_id": "w0",
    }
    assert parsed["artifacts"]["hardware_id"] is None
    assert parsed["retrieval"]["selected_configs"] == []  # empty tuple -> []
    assert parsed["outcome"]["censored"] is False
    assert parsed["timings"]["total"] is None


def test_selected_configs_serialize_as_list():
    parsed = json.loads(event_to_canonical_json(_full_event()))
    assert parsed["retrieval"]["selected_configs"] == ["cfg-a", "cfg-b"]
    assert [neighbor["config_id"] for neighbor in parsed["retrieval"]["neighbors"]] == [
        "cfg-a",
        "cfg-b",
    ]
    assert parsed["retrieval"]["neighbors"][1]["selected_config"] == {
        "block_sizes": [64]
    }


def test_ordered_observations_serialize_with_stable_schema():
    parsed = json.loads(event_to_canonical_json(_full_event()))

    assert parsed["evaluations"] == [
        {
            "benchmark_status": "ok",
            "candidate_category": "seed",
            "candidate_source": "retrieval",
            "compatibility_status": "compatible",
            "compilation_seconds": 0.25,
            "compilation_status": "ok",
            "config_id": "cfg-a",
            "config_repr": "Config(block_size=32)",
            "correctness_status": "ok",
            "elapsed_seconds": 3.5,
            "incumbent_best_perf": 0.8,
            "performance": 0.8,
            "sequence": 1,
            "timeout_status": "not_timed_out",
            "trajectory_sequence": 1,
        }
    ]
    assert parsed["phase_snapshots"] == [
        {
            "elapsed_seconds": 3.5,
            "incumbent_best_perf": 0.8,
            "phase": "benchmarking",
            "phase_seconds": 1.25,
            "trajectory_sequence": 2,
        }
    ]


def test_no_observations_serialize_as_empty_arrays():
    parsed = json.loads(event_to_canonical_json(_sparse_event()))

    assert parsed["evaluations"] == []
    assert parsed["phase_snapshots"] == []


def test_provider_replay_hash_identities_serialize_in_request_order():
    event = dataclasses.replace(
        _full_event(),
        provider=dataclasses.replace(
            _full_event().provider,
            replay_identities=(
                events_module.ProviderReplayIdentity(
                    ordinal=0,
                    request_hash="sha256:request-0",
                    response_hash="sha256:response-0",
                ),
                events_module.ProviderReplayIdentity(
                    ordinal=1,
                    request_hash="sha256:request-1",
                    response_hash=None,
                ),
            ),
        ),
    )

    parsed = json.loads(event_to_canonical_json(event))

    assert parsed["provider"]["replay_identities"] == [
        {
            "ordinal": 0,
            "request_hash": "sha256:request-0",
            "response_hash": "sha256:response-0",
        },
        {
            "ordinal": 1,
            "request_hash": "sha256:request-1",
            "response_hash": None,
        },
    ]


def test_nonfinite_numbers_remain_distinguishable_from_missing_values():
    event = dataclasses.replace(
        _sparse_event(),
        outcome=OutcomeRecord(
            accumulated_seconds=math.inf,
            selected_performance=math.nan,
        ),
    )

    parsed = json.loads(event_to_canonical_json(event))
    assert parsed["outcome"]["accumulated_seconds"] == "+Infinity"
    assert parsed["outcome"]["selected_performance"] == "NaN"


def test_attempt_accounting_uses_explicit_realized_counts():
    event = dataclasses.replace(
        _sparse_event(),
        accounting=AttemptAccountingRecord(
            frozen_limit=32,
            attempted=28,
            initial_population=8,
            invalid=2,
            duplicate=3,
            generation=17,
            llm_proposed=4,
            compiled=24,
            validated=22,
            benchmarked=20,
        ),
    )

    parsed = json.loads(event_to_canonical_json(event))
    assert parsed["accounting"] == {
        "attempted": 28,
        "benchmarked": 20,
        "compiled": 24,
        "duplicate": 3,
        "frozen_limit": 32,
        "generation": 17,
        "initial_population": 8,
        "invalid": 2,
        "llm_proposed": 4,
        "validated": 22,
    }


def test_attempt_accounting_is_optional():
    parsed = json.loads(event_to_canonical_json(_sparse_event()))
    assert parsed["accounting"] is None


def test_attempt_accounting_is_frozen():
    accounting = AttemptAccountingRecord(frozen_limit=4)

    with pytest.raises(dataclasses.FrozenInstanceError):
        accounting.attempted = 1


def test_append_event_uses_explicit_log_and_writes_one_canonical_line(
    tmp_path, monkeypatch
):
    path = tmp_path / "nested" / "events.jsonl"
    monkeypatch.setenv("HELION_RAG_EVENT_LOG", str(path))

    append_event(_full_event())
    append_event(_sparse_event())

    lines = path.read_bytes().splitlines()
    assert lines == [
        event_to_canonical_json(_full_event()),
        event_to_canonical_json(_sparse_event()),
    ]


def test_append_event_retries_short_writes(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("HELION_RAG_EVENT_LOG", str(path))
    real_write = os.write

    def short_write(fd, data):
        return real_write(fd, bytes(data[:7]))

    monkeypatch.setattr(os, "write", short_write)

    append_event(_full_event())

    assert path.read_bytes() == event_to_canonical_json(_full_event()) + b"\n"


def test_retrieval_event_records_typed_exact_cache_probe() -> None:
    event = dataclasses.replace(
        _sparse_event(),
        retrieval=RetrievalRecord(exact_cache_probe="miss"),
    )

    parsed = json.loads(event_to_canonical_json(event))

    assert parsed["retrieval"]["exact_cache_probe"] == "miss"
