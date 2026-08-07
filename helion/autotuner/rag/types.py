"""Immutable types for the opt-in tiered RAG autotuning policy.

These types are the frozen contract between retrieval, the pure decision policy
(:mod:`helion.autotuner.rag.policy`), and the execution wrapper. Keeping them
immutable and free of any heavy imports (no faiss / torch / qwen) lets the kill
switch and the pure decision be evaluated before any retrieval dependency loads.

See ``docs/rag_autotuning_experiment.md``.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Mapping


class LookupTier(enum.IntEnum):
    """Outcome tier of a frozen RAG lookup for a workload."""

    EXACT = 0
    SEMANTIC = 1
    MISS = 2


class TunerMode(enum.Enum):
    """Which standalone tuner consumes the retrieval evidence.

    ``HYBRID`` labels the composed LLM-seeded LFBO search for canonical events;
    it never drives a RAG decision (the hybrid arm runs with retrieval disabled).
    """

    LFBO = "lfbo"
    LLM = "llm"
    HYBRID = "hybrid"


class FallbackReason(enum.Enum):
    """Stable typed reason recorded whenever the wrapper falls back to baseline.

    Each member is a distinct operational event (§3.1); they are never collapsed
    into one another and never inferred from parsing an exception message.
    """

    KILL_SWITCH = "kill_switch"
    EXACT_READ_ERROR = "exact_read_error"
    SIGNATURE_FAILURE = "signature_failure"
    MISSING_ARTIFACT = "missing_artifact"
    VERSION_MISMATCH = "version_mismatch"
    INDEX_CORRUPTION = "index_corruption"
    RETRIEVAL_FAILURE = "retrieval_failure"
    PROVIDER_FAILURE = "provider_failure"


class Phase(enum.Enum):
    """Instrumented phases of a single RAG-assisted readiness attempt (§4)."""

    LOOKUP = "lookup"
    EMBEDDING = "embedding"
    DECISION = "decision"
    GENERATION = "generation"
    PROVIDER = "provider"
    SEEDING = "seeding"
    COMPILATION = "compilation"
    CORRECTNESS = "correctness"
    BENCHMARKING = "benchmarking"
    FALLBACK = "fallback"
    TOTAL = "total"


# Stable ``BaselineSearch.reason`` strings. These are *decision* outcomes (a
# deterministic Tier-2 miss or a disabled treatment), distinct from the operational
# ``FallbackReason`` failures handled by the execution wrapper.
REASON_RAG_DISABLED = "rag_disabled"
REASON_EXACT_REUSE_NOT_PERMITTED = "exact_reuse_not_permitted"
REASON_SEMANTIC_DISABLED = "semantic_disabled"
REASON_NO_NEIGHBORS = "no_neighbors"
REASON_TIER2_MISS = "tier2_miss"


@dataclasses.dataclass(frozen=True)
class WorkloadDescriptor:
    """Everything needed to key retrieval for one kernel invocation.

    Assembled after a true existing-exact-cache miss from the live autotuner
    state: hardware + specialization identity + structural fingerprint + the
    canonical source and shape/dtype signature.
    """

    kernel_name: str
    kernel_source: str
    input_shapes: str
    dtypes: str
    hardware: str
    backend: str
    structural_fingerprint_hash: str
    specialization_key: str | None = None


@dataclasses.dataclass(frozen=True)
class RetrievedNeighbor:
    """One semantic-retrieval neighbor with provenance and its raw scores."""

    kernel_name: str | None
    input_shapes: str | None
    dtypes: str | None
    score: float
    ref: Mapping[str, object] | None = None
    config: Mapping[str, object] | None = None


@dataclasses.dataclass(frozen=True)
class RetrievalEvidence:
    """Frozen result of a lookup, consumed by the pure decision policy.

    ``exact_eligible`` is the upstream determination that a Tier-0 hit satisfies
    the strict-identity requirements (§6.2 S4-or-stricter). It is never inferred
    inside :func:`~helion.autotuner.rag.policy.decide`.
    """

    tier: LookupTier
    family: str | None = None
    exact_config: Mapping[str, object] | None = None
    exact_provenance: Mapping[str, object] | None = None
    exact_eligible: bool = False
    tier0_identity_combo: str | None = None
    tier0_collision_count: int | None = None
    neighbors: tuple[RetrievedNeighbor, ...] = ()
    artifact_identity: Mapping[str, object] | None = None


@dataclasses.dataclass(frozen=True)
class TreatmentConfig:
    """Frozen per-arm policy switches (§3, §5.2).

    ``exact_read`` / ``best_available_read`` / ``write`` are the independent
    cache-access controls (they replace the legacy coupled cache-skip switch).
    ``allow_exact_reuse`` is False for arms/evidence that can never be Tier 0
    (e.g. cross-architecture, or historical-schema shadow evidence).
    """

    treatment_id: str
    rag_enabled: bool = False
    exact_read: bool = True
    best_available_read: bool = True
    write: bool = True
    qwen_enabled: bool = False
    tuner_mode: TunerMode = TunerMode.LFBO
    allow_exact_reuse: bool = True


@dataclasses.dataclass(frozen=True)
class ExactHit:
    """Helion's real exact cache returned a config for this workload.

    ``config`` is the live ``helion.runtime.config.Config`` object from the real
    cache (not a retrieved RAG dict), kept as ``object`` so this module imports no
    heavy dependency.
    """

    config: object


@dataclasses.dataclass(frozen=True)
class ExactMiss:
    """Helion's real exact cache had no entry (a true miss; RAG may proceed)."""


@dataclasses.dataclass(frozen=True)
class ExactReadError:
    """Reading Helion's real exact cache failed (corrupt/unreadable entry).

    Recorded separately from a miss, excluded from incremental RAG-coverage
    denominators, and routed to the frozen fail-closed baseline (§2 step 3).
    """

    error: str


# Result of probing Helion's real exact cache before any RAG work.
ExactProbeResult = ExactHit | ExactMiss | ExactReadError


@dataclasses.dataclass(frozen=True)
class ExactReuse:
    """Reuse a Tier-0 config directly, skipping search."""

    config: Mapping[str, object]
    provenance: Mapping[str, object] | None = None


@dataclasses.dataclass(frozen=True)
class RetrievalSeededSearch:
    """Seed the LFBO population from Tier-1 neighbors."""

    neighbors: tuple[RetrievedNeighbor, ...]


@dataclasses.dataclass(frozen=True)
class ContextualSearch:
    """Supply Tier-1 neighbors to the LLM as structured few-shot examples."""

    neighbors: tuple[RetrievedNeighbor, ...]


@dataclasses.dataclass(frozen=True)
class BaselineSearch:
    """Delegate to the unchanged full tuner; ``reason`` is a stable string."""

    reason: str


# The closed set of decisions the pure policy may return.
Decision = ExactReuse | RetrievalSeededSearch | ContextualSearch | BaselineSearch


@dataclasses.dataclass(frozen=True)
class PhaseTimingEvent:
    """Elapsed wall time attributed to one instrumented :class:`Phase`."""

    phase: Phase
    seconds: float


@dataclasses.dataclass(frozen=True)
class ExecutionOutcome:
    """Result of executing a :class:`Decision` through the execution wrapper.

    ``accumulated_seconds`` is the actual accumulated end-to-end time; when the
    harness terminates the arm at a global deadline, ``censored`` marks the value
    as a right-censoring boundary rather than a completed observation (§4).
    """

    decision: Decision
    result: object
    validated: bool
    accumulated_seconds: float
    censored: bool = False
    fallback_reason: FallbackReason | None = None
    timings: tuple[PhaseTimingEvent, ...] = ()
