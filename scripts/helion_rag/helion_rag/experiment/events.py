"""Uniform instrumentation event schema and canonical serializer (§4).

Section 4 mandates *one* uniform schema emitted per RAG-assisted readiness
attempt, capturing run identity, artifact identities, the retrieval record, the
provider record, the outcome, and per-phase timings. This module is a pure data
schema plus a deterministic serializer for the frozen research ledger and replay;
it performs no I/O beyond returning bytes.

The fields are grouped into nested frozen dataclasses so each §4 bullet maps to a
readable record. Optional fields are ``... | None`` and default to ``None`` (or an
empty tuple) so a partially-observed attempt is representable without inventing
values. Runtime enums are reused from :mod:`helion.autotuner.rag.types` rather
than redefined; the :class:`PhaseTimings` fields mirror the :class:`~helion.
autotuner.rag.types.Phase` members.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import enum
import fcntl
import json
import math
import os
from pathlib import Path

from helion.autotuner.rag.instrumentation import EvaluationRecord as EvaluationRecord
from helion.autotuner.rag.instrumentation import PhaseSnapshot as PhaseSnapshotRecord
from helion.autotuner.rag.types import FallbackReason
from helion.autotuner.rag.types import LookupTier
from helion.autotuner.rag.types import TunerMode


@dataclasses.dataclass(frozen=True)
class RunIdentity:
    """Workload/kernel/arm/tuner/repetition/seed/treatment identity (§4)."""

    workload_id: str
    kernel_name: str
    arm_id: str
    treatment_id: str
    tuner_mode: TunerMode
    repetition: int
    random_seed: int


@dataclasses.dataclass(frozen=True)
class ArtifactIdentity:
    """Code/corpus/model/tokenizer/index/manifest/runtime/driver/hardware IDs (§4).

    ``device_uuid``, ``kernel_environment_id``, and ``triton_id`` extend the
    identity for strict device and toolchain provenance; under a signed freeze they
    are pinned to the manifest's expected values (see ``HELION_RAG_EXPECTED_*``).
    """

    code_id: str | None = None
    corpus_id: str | None = None
    model_id: str | None = None
    tokenizer_id: str | None = None
    index_id: str | None = None
    manifest_id: str | None = None
    runtime_id: str | None = None
    driver_id: str | None = None
    hardware_id: str | None = None
    device_uuid: str | None = None
    kernel_environment_id: str | None = None
    triton_id: str | None = None


@dataclasses.dataclass(frozen=True)
class NeighborRetrievalRecord:
    """One retained neighbor with its selected compatible config and ranks."""

    provenance: str | None = None
    config_id: str | None = None
    selected_config: Mapping[str, object] | None = None
    raw_semantic_score: float | None = None
    raw_shape_score: float | None = None
    combined_score: float | None = None
    rank_before_rerank: int | None = None
    rank_after_rerank: int | None = None


@dataclasses.dataclass(frozen=True)
class RetrievalRecord:
    """Lookup tier, Tier-0 identity, neighbor scores/ranks, and candidate status (§4).

    ``tier0_identity_combo`` is the Tier-0 identity-field combination and
    ``tier0_collision_count`` its collision count. The score/rank fields describe
    the selected neighbor; ``selected_configs`` holds the chosen config
    identifiers (empty when none were selected).
    """

    lookup_tier: LookupTier | None = None
    exact_cache_probe: str | None = None
    tier0_identity_combo: str | None = None
    tier0_collision_count: int | None = None
    neighbor_provenance: str | None = None
    raw_semantic_score: float | None = None
    raw_shape_score: float | None = None
    combined_score: float | None = None
    rank_before_rerank: int | None = None
    rank_after_rerank: int | None = None
    selected_configs: tuple[str, ...] = ()
    neighbors: tuple[NeighborRetrievalRecord, ...] = ()
    candidate_source: str | None = None
    compatibility_status: str | None = None


@dataclasses.dataclass(frozen=True)
class ProviderReplayIdentity:
    """Ordered canonical request and response hashes for one provider call."""

    ordinal: int
    request_hash: str
    response_hash: str | None


@dataclasses.dataclass(frozen=True)
class ProviderRecord:
    """Provider request/response identity, cache state, and token usage (§4)."""

    request_id: str | None = None
    response_id: str | None = None
    cache_state: str | None = None
    requests: int = 0
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    replay_identities: tuple[ProviderReplayIdentity, ...] = ()


@dataclasses.dataclass(frozen=True)
class OutcomeRecord:
    """Compilation/correctness/benchmark/timeout/fallback status and timing (§4).

    ``accumulated_seconds`` is the actual accumulated end-to-end time; ``censored``
    marks it as a right-censoring boundary when the arm hit a global deadline.
    ``incumbent_best_perf`` is the incumbent best validated performance recorded at
    this evaluation / phase transition.
    """

    decision_name: str | None = None
    compilation_status: str | None = None
    correctness_status: str | None = None
    benchmark_status: str | None = None
    timeout_status: str | None = None
    fallback_reason: FallbackReason | None = None
    accumulated_seconds: float | None = None
    censored: bool = False
    incumbent_best_perf: float | None = None
    selected_performance: float | None = None
    terminal_error: str | None = None


@dataclasses.dataclass(frozen=True)
class PhaseTimings:
    """Per-phase elapsed seconds; fields mirror the ``Phase`` members (§4)."""

    lookup: float | None = None
    embedding: float | None = None
    decision: float | None = None
    generation: float | None = None
    provider: float | None = None
    seeding: float | None = None
    compilation: float | None = None
    correctness: float | None = None
    benchmarking: float | None = None
    fallback: float | None = None
    total: float | None = None
    readiness_provider_inclusive: float | None = None
    readiness_provider_exclusive: float | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class AttemptAccountingRecord:
    """Frozen budget and realized candidate/evaluation counts for one attempt."""

    frozen_limit: int
    attempted: int = 0
    initial_population: int = 0
    invalid: int = 0
    duplicate: int = 0
    generation: int = 0
    llm_proposed: int = 0
    compiled: int = 0
    validated: int = 0
    benchmarked: int = 0


@dataclasses.dataclass(frozen=True)
class HybridStageRecord:
    """Per-stage split for the composed LLM-seeded LFBO arm (null for others).

    ``llm_attempts`` / ``lfbo_attempts`` partition the run-level
    :class:`AttemptAccountingRecord` total across the two stages of the shared
    budget; ``best_perf_at_handoff_ms`` is the LLM stage's incumbent handed to
    LFBO and ``final_perf_ms`` the hybrid's global best.
    """

    candidate_attempt_limit: int | None = None
    total_attempts: int = 0
    llm_attempts: int = 0
    lfbo_attempts: int = 0
    llm_seed_time_s: float | None = None
    second_stage_time_s: float | None = None
    best_perf_at_handoff_ms: float | None = None
    final_perf_ms: float | None = None
    llm_seed_configs_tested: int = 0
    second_stage_configs_tested: int = 0
    provider_requests: int = 0
    second_stage_ran: bool = False


@dataclasses.dataclass(frozen=True)
class InstrumentationEvent:
    """The one uniform per-attempt instrumentation schema (§4)."""

    run: RunIdentity
    artifacts: ArtifactIdentity
    retrieval: RetrievalRecord
    provider: ProviderRecord
    outcome: OutcomeRecord
    timings: PhaseTimings
    accounting: AttemptAccountingRecord | None = None
    evaluations: tuple[EvaluationRecord, ...] = ()
    phase_snapshots: tuple[PhaseSnapshotRecord, ...] = ()
    hybrid_stage_breakdown: HybridStageRecord | None = None


def _jsonable(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "+Infinity" if value > 0 else "-Infinity"
    return value.value if isinstance(value, enum.Enum) else value


def _dict_factory(items: list[tuple[str, object]]) -> dict[str, object]:
    return {key: _jsonable(value) for key, value in items}


def event_to_canonical_json(event: InstrumentationEvent) -> bytes:
    """Deterministic canonical JSON for the frozen ledger and replay (§4).

    Keys are sorted and whitespace is stripped, so the bytes depend only on the
    event's values (never on dict insertion order). Enums serialize by ``.value``
    and nested dataclasses via :func:`dataclasses.asdict`.
    """
    payload = dataclasses.asdict(event, dict_factory=_dict_factory)
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def event_log_path() -> Path | None:
    """Resolve the explicit event log or the standard autotune-log location."""
    if explicit := os.environ.get("HELION_RAG_EVENT_LOG"):
        return Path(explicit)
    if log_dir := os.environ.get("HELION_RAG_AUTOTUNE_LOG_DIR"):
        return Path(log_dir) / "rag-events.jsonl"
    return None


def append_event(event: InstrumentationEvent) -> Path | None:
    """Append one canonical event line when a live log location is configured."""
    path = event_log_path()
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            payload = memoryview(event_to_canonical_json(event) + b"\n")
            while payload:
                written = os.write(fd, payload)
                if written == 0:
                    raise OSError("event log write made no progress")
                payload = payload[written:]
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    return path
