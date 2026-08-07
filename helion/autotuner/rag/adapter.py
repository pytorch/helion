"""Lazy live-autotuner adapter for the opt-in tiered RAG policy."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import hashlib
import os
import subprocess
import time
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ..._compat import extract_device
from ..._compat import get_device_name
from ...runtime.config import Config
from ..base_cache import helion_key
from ..base_cache import should_skip_cache
from ..base_search import _normalize_spec_key
from ..candidate_budget import AttemptBudget
from ..candidate_budget import AttemptCategory
from ..candidate_budget import random_replacement_draw_cap
from .execution import REASON_EXISTING_EXACT_HIT
from .execution import execute_rag_attempt
from .instrumentation import InstrumentationCollector
from .policy_config import FrozenRetrievalPolicy
from .probe import probe_exact_cache
from .seeding import build_seeded_population
from .types import BaselineSearch
from .types import ContextualSearch
from .types import Decision
from .types import ExactHit
from .types import ExactMiss
from .types import ExactProbeResult
from .types import ExactReadError
from .types import ExactReuse
from .types import ExecutionOutcome
from .types import LookupTier
from .types import Phase
from .types import PhaseTimingEvent
from .types import RetrievalEvidence
from .types import RetrievalSeededSearch
from .types import RetrievedNeighbor
from .types import TreatmentConfig
from .types import TunerMode
from .types import WorkloadDescriptor

if TYPE_CHECKING:
    from ...runtime.kernel import BoundKernel
    from ..base_cache import AutotuneCacheBase
    from ..base_search import BaseSearch
    from ..base_search import PopulationBasedSearch


_RETRIEVAL_CONTROLS = frozenset({"frozen", "random", "shuffled"})


def resolve_tuner_mode(autotuner: BaseSearch) -> TunerMode:
    """Return the supported standalone RAG tuner mode or reject configuration."""
    from ..llm_search import LLMGuidedSearch
    from ..surrogate_pattern_search import LFBOPatternSearch

    if isinstance(autotuner, LLMGuidedSearch):
        return TunerMode.LLM
    if isinstance(autotuner, LFBOPatternSearch):
        return TunerMode.LFBO
    raise ValueError(
        "unsupported RAG autotuner: expected LFBOTreeSearch/LFBOPatternSearch "
        f"or LLMGuidedSearch, got {type(autotuner).__name__}"
    )


def build_workload_descriptor(autotuner: BaseSearch) -> WorkloadDescriptor:
    """Build retrieval identity from the live bound kernel and invocation."""
    kernel = cast("BoundKernel", autotuner.kernel)
    kernel_definition = kernel.kernel
    tensors = [arg for arg in autotuner.args if isinstance(arg, torch.Tensor)]
    device = extract_device(autotuner.args)
    hardware = get_device_name(device) or (device.type if device is not None else "")
    fingerprint = autotuner.config_spec.structural_fingerprint_hash(
        advanced_controls_files=autotuner.settings.autotune_search_acf or None
    )
    specialization = kernel_definition._base_specialization_key(autotuner.args)
    return WorkloadDescriptor(
        kernel_name=kernel_definition.name,
        kernel_source=kernel_definition.kernel_source(),
        input_shapes=str([tuple(tensor.shape) for tensor in tensors]),
        dtypes=str([str(tensor.dtype) for tensor in tensors]),
        hardware=hardware,
        backend=kernel.env.backend.name,
        structural_fingerprint_hash=fingerprint,
        specialization_key=str(_normalize_spec_key(specialization)),
    )


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _required_float(value: object, field: str) -> float:
    if not isinstance(value, (int, float)):
        raise exc.RetrieverUnavailable(f"lookup field {field!r} must be numeric")
    return float(value)


def _optional_float(mapping: Mapping[str, object] | None, field: str) -> float | None:
    if mapping is None or (value := mapping.get(field)) is None:
        return None
    return _required_float(value, field)


def _optional_int(mapping: Mapping[str, object] | None, field: str) -> int | None:
    if mapping is None or (value := mapping.get(field)) is None:
        return None
    if not isinstance(value, int):
        raise exc.RetrieverUnavailable(f"lookup field {field!r} must be an integer")
    return value


def _optional_string(mapping: Mapping[str, object], field: str) -> str | None:
    value = mapping.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise exc.RetrieverUnavailable(f"lookup field {field!r} must be a string")
    return value


def _select_compatible_config(
    raw_neighbor: Mapping[str, object],
    normalize_config: Callable[[Mapping[str, object]], Config],
) -> tuple[Config, Mapping[str, object]] | None:
    top_n = raw_neighbor.get("top_n")
    if not isinstance(top_n, Sequence) or isinstance(top_n, (str, bytes)):
        return None
    for raw_entry in top_n:
        entry = _mapping(raw_entry)
        raw_config = _mapping(entry.get("config")) if entry is not None else None
        if raw_config is None:
            continue
        try:
            normalized = normalize_config(raw_config)
        except (exc.InvalidConfig, ValueError, TypeError, KeyError, AssertionError):
            continue
        assert entry is not None
        return normalized, entry
    return None


def convert_lookup_result(
    result: Mapping[str, object],
    descriptor: WorkloadDescriptor,
    *,
    normalize_config: Callable[[Mapping[str, object]], Config],
    policy: FrozenRetrievalPolicy | None = None,
) -> RetrievalEvidence:
    """Convert a live lookup dictionary into typed, shape-reranked evidence."""
    policy = policy or FrozenRetrievalPolicy.default()
    raw_tier = result.get("tier", LookupTier.MISS)
    if not isinstance(raw_tier, int):
        raise exc.RetrieverUnavailable("lookup field 'tier' must be an integer")
    tier = LookupTier(raw_tier)
    family = result.get("family")
    family_name = family if isinstance(family, str) else None
    tier0_identity_combo = _optional_string(result, "tier0_identity_combo")
    tier0_collision_count = _optional_int(result, "tier0_collision_count")
    if tier is LookupTier.EXACT:
        raw_config = _mapping(result.get("best_config"))
        # Historical artifacts do not contain the extended S4 identity required
        # for direct reuse, so they remain shadow evidence and fail closed.
        return RetrievalEvidence(
            tier=tier,
            family=family_name,
            exact_config=raw_config,
            exact_provenance=_mapping(result.get("ref")),
            exact_eligible=False,
            tier0_identity_combo=tier0_identity_combo,
            tier0_collision_count=tier0_collision_count,
            artifact_identity=_mapping(result.get("artifact_identity")),
        )
    if tier is not LookupTier.SEMANTIC:
        return RetrievalEvidence(
            tier=LookupTier.MISS,
            family=family_name,
            tier0_identity_combo=tier0_identity_combo,
            tier0_collision_count=tier0_collision_count,
            artifact_identity=_mapping(result.get("artifact_identity")),
        )

    from helion_rag.rerank import Candidate  # pyrefly: ignore [missing-import]
    from helion_rag.rerank import rerank  # pyrefly: ignore [missing-import]

    candidates: list[Candidate] = []
    raw_neighbors = result.get("neighbors")
    if isinstance(raw_neighbors, Sequence) and not isinstance(
        raw_neighbors, (str, bytes)
    ):
        for raw_value in raw_neighbors[: policy.semantic_pool_size]:
            raw_neighbor = _mapping(raw_value)
            if raw_neighbor is None:
                continue
            selected = _select_compatible_config(raw_neighbor, normalize_config)
            if selected is None:
                continue
            normalized, config_entry = selected
            payload = dict(raw_neighbor)
            payload["selected_config"] = dict(normalized.config)
            payload["selected_config_entry"] = dict(config_entry)
            candidates.append(
                Candidate(
                    semantic_score=_required_float(
                        raw_neighbor.get("score", 0.0), "score"
                    ),
                    input_shapes=str(raw_neighbor.get("input_shapes", "")),
                    dtypes=str(raw_neighbor.get("dtypes", "")),
                    payload=payload,
                )
            )

    if policy.hybrid_weight is None:
        ranked = rerank(
            descriptor.input_shapes,
            descriptor.dtypes,
            candidates,
            rule=policy.ranking_rule,
            k=policy.final_neighbors,
        )
    else:
        ranked = rerank(
            descriptor.input_shapes,
            descriptor.dtypes,
            candidates,
            rule=policy.ranking_rule,
            k=policy.final_neighbors,
            hybrid_weight=policy.hybrid_weight,
        )
    neighbors: list[RetrievedNeighbor] = []
    for scored in ranked.neighbors:
        payload = scored.candidate.payload
        entry = _mapping(payload.get("selected_config_entry")) or {}
        ref = dict(_mapping(payload.get("ref")) or {})
        ref.update(
            {
                "config_id": entry.get("config_id"),
                "raw_semantic_score": scored.semantic_score,
                "raw_shape_score": scored.shape_score,
                "combined_score": scored.combined_score,
                "rank_before_rerank": scored.rank_before,
                "rank_after_rerank": scored.rank_after,
            }
        )
        neighbors.append(
            RetrievedNeighbor(
                kernel_name=(
                    str(payload["kernel_name"])
                    if payload.get("kernel_name") is not None
                    else None
                ),
                input_shapes=(
                    str(payload["input_shapes"])
                    if payload.get("input_shapes") is not None
                    else None
                ),
                dtypes=(
                    str(payload["dtypes"])
                    if payload.get("dtypes") is not None
                    else None
                ),
                score=scored.semantic_score,
                ref=ref,
                config=_mapping(payload.get("selected_config")),
            )
        )
    return RetrievalEvidence(
        tier=LookupTier.SEMANTIC,
        family=family_name,
        neighbors=tuple(neighbors),
        tier0_identity_combo=tier0_identity_combo,
        tier0_collision_count=tier0_collision_count,
        artifact_identity=_mapping(result.get("artifact_identity")),
    )


def _normalize_live_config(
    autotuner: PopulationBasedSearch, raw_config: Mapping[str, object]
) -> Config:
    config = Config.from_dict(raw_config)
    return autotuner.config_gen.unflatten(autotuner.config_gen.flatten(config))


def _retrieval_control() -> str:
    control = os.environ.get("HELION_RAG_RETRIEVAL_CONTROL", "frozen")
    if control not in _RETRIEVAL_CONTROLS:
        raise ValueError(f"unsupported RAG retrieval control {control!r}")
    return control


def _permutation(
    count: int,
    *,
    descriptor: WorkloadDescriptor,
    seed: int,
    label: str,
) -> list[int]:
    identity = list(range(count))
    order = sorted(
        identity,
        key=lambda index: hashlib.sha256(
            (
                f"{label}\0{seed}\0{descriptor.structural_fingerprint_hash}"
                f"\0{descriptor.specialization_key}\0{index}"
            ).encode()
        ).digest(),
    )
    if count > 1 and order == identity:
        order = [*order[1:], order[0]]
    return order


def apply_retrieval_control(
    evidence: RetrievalEvidence,
    descriptor: WorkloadDescriptor,
    control: str,
    *,
    seed: int,
) -> RetrievalEvidence:
    """Apply a deterministic equal-K calibration transformation to evidence."""
    if control not in _RETRIEVAL_CONTROLS:
        raise ValueError(f"unsupported RAG retrieval control {control!r}")
    if control == "frozen" or len(evidence.neighbors) < 2:
        return evidence
    neighbors = evidence.neighbors
    if control == "random":
        order = _permutation(
            len(neighbors), descriptor=descriptor, seed=seed, label="random"
        )
        transformed = tuple(neighbors[index] for index in order)
    else:
        order = _permutation(
            len(neighbors), descriptor=descriptor, seed=seed, label="shuffled"
        )
        transformed = tuple(
            dataclasses.replace(
                neighbor,
                config=neighbors[source].config,
                ref=neighbors[source].ref,
            )
            for neighbor, source in zip(neighbors, order, strict=True)
        )
    return dataclasses.replace(evidence, neighbors=transformed)


def retrieve_evidence(
    descriptor: WorkloadDescriptor,
    autotuner: PopulationBasedSearch,
    policy: FrozenRetrievalPolicy | None = None,
) -> RetrievalEvidence:
    """Run verified live lookup and convert its dictionary to typed evidence."""
    if policy is None:
        policy = FrozenRetrievalPolicy.default()
    try:
        from helion_rag.lookup import lookup  # pyrefly: ignore [missing-import]
    except ImportError as error:
        raise exc.RetrieverUnavailable("helion_rag is not importable") from error

    result = lookup(
        descriptor.kernel_source,
        descriptor.input_shapes,
        descriptor.dtypes,
        descriptor.hardware,
        settings=autotuner.settings.to_dict(),
        k=policy.semantic_pool_size,
        propagate_artifact_errors=True,
        require_generation_pin=True,
    )
    evidence = convert_lookup_result(
        result,
        descriptor,
        normalize_config=lambda raw: _normalize_live_config(autotuner, raw),
        policy=policy,
    )
    return apply_retrieval_control(
        evidence,
        descriptor,
        _retrieval_control(),
        seed=autotuner.settings.autotune_random_seed,
    )


class _LiveDecisionRunner:
    def __init__(
        self,
        autotuner: PopulationBasedSearch,
        tuner_mode: TunerMode,
        run_search: Callable[[], Config],
        instrumentation: InstrumentationCollector,
        *,
        best_available_read: bool = False,
        retrieval_control: str = "frozen",
        policy: FrozenRetrievalPolicy | None = None,
    ) -> None:
        self.autotuner = autotuner
        self.tuner_mode = tuner_mode
        self.run_search = run_search
        self.instrumentation = instrumentation
        self.best_available_read = best_available_read
        self.retrieval_control = retrieval_control
        self._policy = policy
        self.timings: list[PhaseTimingEvent] = []

    @property
    def policy(self) -> FrozenRetrievalPolicy:
        """Retrieval policy, resolved lazily so an exact hit never builds one."""
        if self._policy is None:
            self._policy = FrozenRetrievalPolicy.default()
        return self._policy

    def _record_timing(self, phase: Phase, seconds: float) -> None:
        event = PhaseTimingEvent(phase, seconds)
        self.timings.append(event)
        self.instrumentation.record_phase_transition(phase, phase_seconds=seconds)

    def _run_lfbo(self, decision: RetrievalSeededSearch) -> Config:
        from ..surrogate_pattern_search import LFBOPatternSearch

        if not isinstance(self.autotuner, LFBOPatternSearch):
            raise ValueError("RetrievalSeededSearch requires an LFBO autotuner")
        seeding_start = time.perf_counter()
        try:
            retrieved_seeds = [
                _normalize_live_config(self.autotuner, neighbor.config)
                for neighbor in decision.neighbors
                if neighbor.config is not None
            ]
            best_available_seeds: list[Config] = []
            if self.best_available_read:
                cached = self.autotuner._find_similar_cached_configs(
                    self.autotuner.settings.autotune_best_available_max_configs
                )
                for entry in cached:
                    try:
                        config = self.autotuner.config_gen.unflatten(
                            entry.to_mutable_flat_config()
                        )
                    except (
                        exc.InvalidConfig,
                        ValueError,
                        TypeError,
                        KeyError,
                        AssertionError,
                    ):
                        continue
                    if config not in best_available_seeds:
                        best_available_seeds.append(config)
            seeds: list[Config] = []
            capped_retrieval = retrieved_seeds[: self.policy.final_neighbors]
            for index in range(max(len(capped_retrieval), len(best_available_seeds))):
                if index < len(capped_retrieval):
                    seeds.append(capped_retrieval[index])
                if index < len(best_available_seeds):
                    seeds.append(best_available_seeds[index])
            n = self.autotuner.initial_population
            draw_cap = random_replacement_draw_cap(n)

            def draw_random() -> Config:
                return self.autotuner.config_gen.unflatten(
                    self.autotuner.config_gen.random_flat()
                )

            def is_valid(config: Config) -> bool:
                try:
                    self.autotuner.config_gen.unflatten(
                        self.autotuner.config_gen.flatten(config)
                    )
                    # A retrieved seed can round-trip flatten/unflatten yet still be
                    # unencodable by the LFBO surrogate (a value outside the current
                    # spec's domain), which would crash training. Require encodability
                    # so such seeds are dropped and replaced by a random draw.
                    self.autotuner.config_gen.encode_config(
                        self.autotuner.config_gen.flatten(config)
                    )
                except (
                    exc.InvalidConfig,
                    ValueError,
                    TypeError,
                    KeyError,
                    AssertionError,
                ):
                    return False
                return True

            budget = self.autotuner.candidate_attempt_budget
            population, _ = build_seeded_population(
                seeds,
                n=n,
                num_neighbors_cap=n,
                draw_random=draw_random,
                is_valid=is_valid,
                flatten_key=lambda config: config,
                budget=budget,
                draw_cap=draw_cap,
            )

            self.autotuner.set_candidate_attempt_budget(budget)
            self.autotuner.set_fixed_initial_population_configs(
                population, attempts_already_recorded=True
            )
            retrieved = set(retrieved_seeds[: self.policy.final_neighbors])
            best_available = set(best_available_seeds)
            retrieval_source = {
                "frozen": "retrieval",
                "random": "random_neighbor",
                "shuffled": "shuffled_neighbor",
            }[self.retrieval_control]
            for config in population:
                if config in retrieved:
                    source = retrieval_source
                elif config in best_available:
                    source = "best_available"
                else:
                    source = "random_replacement"
                self.autotuner.set_candidate_source(
                    config,
                    source,
                )
        finally:
            self._record_timing(Phase.SEEDING, time.perf_counter() - seeding_start)
        best_available_read = self.autotuner.settings.autotune_best_available_read
        self.autotuner.settings.autotune_best_available_read = False
        try:
            return self.run_search()
        finally:
            self.autotuner.settings.autotune_best_available_read = best_available_read

    def _run_llm(self, decision: ContextualSearch) -> Config:
        from ..llm_search import LLMGuidedSearch

        if not isinstance(self.autotuner, LLMGuidedSearch):
            raise ValueError("ContextualSearch requires LLMGuidedSearch")
        examples = [
            {
                "kernel_name": neighbor.kernel_name,
                "input_shapes": neighbor.input_shapes,
                "dtypes": neighbor.dtypes,
                "config": dict(neighbor.config)
                if neighbor.config is not None
                else None,
            }
            for neighbor in decision.neighbors
        ]
        self.autotuner.set_retrieved_examples(examples)
        return self.run_search()

    def run(self, decision: Decision) -> Config:
        if isinstance(decision, ExactReuse):
            normalized = _normalize_live_config(self.autotuner, decision.config)
            if normalized.config != dict(decision.config):
                raise exc.InvalidConfig("Tier-0 config changed during normalization")
            return normalized
        if isinstance(decision, RetrievalSeededSearch):
            return self._run_lfbo(decision)
        if isinstance(decision, ContextualSearch):
            return self._run_llm(decision)
        raise AssertionError(f"unexpected live RAG decision {type(decision).__name__}")


def _provider_reporting_search(autotuner: BaseSearch) -> BaseSearch | None:
    """The search if it reports provider usage, else None (LFBO reports nothing).

    Dispatched on type rather than probed with hasattr: these methods live on
    classes in this same tree, so a rename should break loudly instead of
    silently zeroing the campaign's provider accounting.
    """
    from ..llm_search import LLMGuidedSearch
    from ..llm_seeded_lfbo import LLMSeededSearch

    return (
        autotuner if isinstance(autotuner, (LLMSeededSearch, LLMGuidedSearch)) else None
    )


def _provider_tokens(autotuner: BaseSearch) -> dict[str, int | None]:
    if (search := _provider_reporting_search(autotuner)) is not None:
        return search.aggregate_token_usage()
    return {
        "requests": 0,
        "input_tokens": None,
        "cached_input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
    }


def _provider_identity(autotuner: BaseSearch) -> dict[str, str | None]:
    if (search := _provider_reporting_search(autotuner)) is not None:
        return search.aggregate_provider_identity()
    return {"request_id": None, "response_id": None, "cache_state": None}


def _provider_replay_identities(
    autotuner: BaseSearch,
) -> tuple[tuple[str, str | None], ...]:
    if (search := _provider_reporting_search(autotuner)) is not None:
        return search.provider_replay_identities
    return ()


def _artifact_value(evidence: RetrievalEvidence | None, field: str) -> str | None:
    identity = evidence.artifact_identity if evidence is not None else None
    value = identity.get(field) if identity is not None else None
    return str(value) if value is not None else None


def _resolved_embedding_model(evidence: RetrievalEvidence | None) -> str:
    if model_id := _artifact_value(evidence, "model_id"):
        return model_id
    from helion_rag.config import _config  # pyrefly: ignore[missing-import]

    return _config().embed_model


def _triton_identity() -> str | None:
    """Best-effort installed Triton version for strict toolchain provenance."""
    try:
        import triton
    except ImportError:
        return None
    return getattr(triton, "__version__", None)


def _device_identity(device: torch.device | None) -> str | None:
    """Stable per-device UUID, when the accelerator exposes one."""
    if device is None or device.type != "cuda" or torch.version.cuda is None:
        return None
    uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    return str(uuid) if uuid is not None else None


def _runtime_identity(descriptor: WorkloadDescriptor | None) -> str:
    backend = descriptor.backend if descriptor is not None else "unknown"
    fields = [f"backend={backend}", f"torch={torch.__version__}"]
    if torch.version.cuda is not None:
        fields.append(f"cuda={torch.version.cuda}")
    if torch.version.hip is not None:
        fields.append(f"hip={torch.version.hip}")
    return ";".join(fields)


def resolve_driver_identity(device: torch.device | None) -> str | None:
    """Resolve the active accelerator driver without substituting toolkit versions."""
    if device is None:
        return None
    if device.type == "xpu":
        return str(torch.xpu.get_device_properties(device).driver_version)
    if device.type != "cuda" or torch.version.cuda is None:
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def _hybrid_stage_record(autotuner: BaseSearch) -> object | None:
    """Map a composed search's stage breakdown into the canonical event record."""
    from ..llm_seeded_lfbo import LLMSeededSearch

    if not isinstance(autotuner, LLMSeededSearch):
        return None
    breakdown = autotuner.hybrid_stage_breakdown
    if not isinstance(breakdown, dict):
        return None
    from helion_rag.experiment.events import (  # pyrefly: ignore[missing-import]
        HybridStageRecord,
    )

    return HybridStageRecord(
        candidate_attempt_limit=breakdown.get("candidate_attempt_limit"),
        total_attempts=breakdown.get("total_attempts", 0),
        llm_attempts=breakdown.get("llm_attempts", 0),
        lfbo_attempts=breakdown.get("lfbo_attempts", 0),
        llm_seed_time_s=breakdown.get("llm_seed_time_s"),
        second_stage_time_s=breakdown.get("second_stage_time_s"),
        best_perf_at_handoff_ms=breakdown.get("best_perf_at_handoff_ms"),
        final_perf_ms=breakdown.get("final_perf_ms"),
        llm_seed_configs_tested=breakdown.get("llm_seed_configs_tested", 0),
        second_stage_configs_tested=breakdown.get("second_stage_configs_tested", 0),
        provider_requests=breakdown.get("provider_requests", 0),
        second_stage_ran=breakdown.get("second_stage_ran", False),
    )


def emit_attempt_event(
    cache: AutotuneCacheBase,
    *,
    tuner_mode: TunerMode,
    descriptor: WorkloadDescriptor | None,
    evidence: RetrievalEvidence | None,
    outcome: ExecutionOutcome,
    instrumentation: InstrumentationCollector,
    attempt_budget: AttemptBudget,
    terminal_error: str | None = None,
    retrieval_control: str = "frozen",
    exact_cache_probe: str | None = None,
) -> None:
    """Append the canonical event for one enabled live RAG attempt."""
    from helion_rag.experiment.events import (  # pyrefly: ignore[missing-import]
        ArtifactIdentity,
    )
    from helion_rag.experiment.events import (  # pyrefly: ignore[missing-import]
        AttemptAccountingRecord,
    )
    from helion_rag.experiment.events import (  # pyrefly: ignore[missing-import]
        InstrumentationEvent,
    )
    from helion_rag.experiment.events import (  # pyrefly: ignore[missing-import]
        NeighborRetrievalRecord,
    )
    from helion_rag.experiment.events import (  # pyrefly: ignore[missing-import]
        OutcomeRecord,
    )
    from helion_rag.experiment.events import (  # pyrefly: ignore[missing-import]
        PhaseTimings,
    )
    from helion_rag.experiment.events import (  # pyrefly: ignore[missing-import]
        ProviderRecord,
    )
    from helion_rag.experiment.events import (  # pyrefly: ignore[missing-import]
        ProviderReplayIdentity,
    )
    from helion_rag.experiment.events import (  # pyrefly: ignore[missing-import]
        RetrievalRecord,
    )
    from helion_rag.experiment.events import (  # pyrefly: ignore[missing-import]
        RunIdentity,
    )
    from helion_rag.experiment.events import (  # pyrefly: ignore[missing-import]
        append_event,
    )

    first_neighbor = evidence.neighbors[0] if evidence and evidence.neighbors else None
    ref = first_neighbor.ref if first_neighbor is not None else None
    timings: dict[str, float] = {}
    for snapshot in instrumentation.phase_snapshots:
        if snapshot.phase_seconds is not None:
            phase = snapshot.phase.value
            timings[phase] = timings.get(phase, 0.0) + snapshot.phase_seconds
    if "total" not in timings:
        for timing in outcome.timings:
            if timing.phase is Phase.TOTAL:
                timings["total"] = timing.seconds
    if (total := timings.get("total")) is not None:
        provider_seconds = timings.get("provider", 0.0)
        timings["readiness_provider_inclusive"] = total
        timings["readiness_provider_exclusive"] = max(0.0, total - provider_seconds)
    tokens = _provider_tokens(cache.autotuner)
    provider_identity = _provider_identity(cache.autotuner)
    provider_replay_identities = _provider_replay_identities(cache.autotuner)
    evaluations = instrumentation.evaluations
    selected_performance: float | None = None
    if isinstance(outcome.result, Config):
        selected_repr = repr(
            _normalize_live_config(
                cast("PopulationBasedSearch", cache.autotuner), outcome.result.config
            )
        )
        selected_id = hashlib.sha256(selected_repr.encode("utf-8")).hexdigest()
        selected_performance = next(
            (
                record.performance
                for record in reversed(evaluations)
                if record.config_id == selected_id and record.performance is not None
            ),
            None,
        )
    incumbent_best_perf = instrumentation.incumbent_best_perf
    compiled_ids = {
        record.config_id
        for record in evaluations
        if record.config_id is not None and record.compilation_status == "ok"
    }
    validated_ids = {
        record.config_id
        for record in evaluations
        if record.config_id is not None and record.correctness_status == "ok"
    }
    benchmarked_ids = {
        record.config_id
        for record in evaluations
        if record.config_id is not None and record.benchmark_status == "ok"
    }
    counts = attempt_budget.by_category()
    kernel_name = (
        descriptor.kernel_name if descriptor is not None else cache.kernel.kernel.name
    )
    workload_id = (
        os.environ.get("HELION_RAG_WORKLOAD_ID") or cache._get_cache_key().stable_hash()
    )

    def _expected(name: str) -> str | None:
        return os.environ.get(name) or None

    # Under a signed freeze every identity is pinned to the manifest's expected
    # value (``HELION_RAG_EXPECTED_*``); otherwise it is derived live. Model and
    # tokenizer resolve together so the retriever config lookup is skipped whenever
    # both are pinned -- e.g. a terminal event emitted before a descriptor exists.
    model_override = _expected("HELION_RAG_EXPECTED_MODEL_ID")
    tokenizer_override = _expected("HELION_RAG_EXPECTED_TOKENIZER_ID")
    if model_override is not None and tokenizer_override is not None:
        model_id, tokenizer_id = model_override, tokenizer_override
    else:
        embedding_model = _resolved_embedding_model(evidence)
        model_id = model_override or embedding_model
        tokenizer_id = (
            tokenizer_override
            or _artifact_value(evidence, "tokenizer_id")
            or embedding_model
        )
    artifacts = ArtifactIdentity(
        code_id=_expected("HELION_RAG_EXPECTED_CODE_ID") or helion_key(),
        corpus_id=_expected("HELION_RAG_EXPECTED_CORPUS_ID")
        or _artifact_value(evidence, "corpus_id"),
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        index_id=_expected("HELION_RAG_EXPECTED_INDEX_ID")
        or _artifact_value(evidence, "index_id"),
        manifest_id=_expected("HELION_RAG_EXPECTED_MANIFEST_ID")
        or _artifact_value(evidence, "manifest_id"),
        runtime_id=_expected("HELION_RAG_EXPECTED_RUNTIME_ID")
        or _runtime_identity(descriptor),
        driver_id=_expected("HELION_RAG_EXPECTED_DRIVER_ID")
        or resolve_driver_identity(extract_device(cache.args)),
        hardware_id=_expected("HELION_RAG_EXPECTED_HARDWARE_ID")
        or (descriptor.hardware if descriptor is not None else None),
        device_uuid=_expected("HELION_RAG_EXPECTED_DEVICE_UUID")
        or _device_identity(extract_device(cache.args)),
        kernel_environment_id=workload_id,
        triton_id=_expected("HELION_RAG_EXPECTED_TRITON_ID") or _triton_identity(),
    )
    event = InstrumentationEvent(
        run=RunIdentity(
            workload_id=workload_id,
            kernel_name=kernel_name,
            arm_id=os.environ.get("HELION_RAG_ARM_ID", "live"),
            treatment_id=os.environ.get("HELION_RAG_TREATMENT_ID", "rag"),
            tuner_mode=tuner_mode,
            repetition=int(os.environ.get("HELION_RAG_REPETITION", "1")),
            random_seed=cache.autotuner.settings.autotune_random_seed,
        ),
        artifacts=artifacts,
        retrieval=RetrievalRecord(
            lookup_tier=evidence.tier if evidence is not None else None,
            exact_cache_probe=exact_cache_probe,
            tier0_identity_combo=(
                evidence.tier0_identity_combo if evidence is not None else None
            ),
            tier0_collision_count=(
                evidence.tier0_collision_count if evidence is not None else None
            ),
            neighbor_provenance=(
                str(ref.get("run_id"))
                if ref is not None and ref.get("run_id") is not None
                else None
            ),
            raw_semantic_score=_optional_float(ref, "raw_semantic_score"),
            raw_shape_score=_optional_float(ref, "raw_shape_score"),
            combined_score=_optional_float(ref, "combined_score"),
            rank_before_rerank=_optional_int(ref, "rank_before_rerank"),
            rank_after_rerank=_optional_int(ref, "rank_after_rerank"),
            selected_configs=tuple(
                str(neighbor.ref["config_id"])
                for neighbor in evidence.neighbors
                if neighbor.ref is not None
                and neighbor.ref.get("config_id") is not None
            )
            if evidence is not None
            else (),
            neighbors=tuple(
                NeighborRetrievalRecord(
                    provenance=(
                        str(neighbor.ref.get("run_id"))
                        if neighbor.ref is not None
                        and neighbor.ref.get("run_id") is not None
                        else None
                    ),
                    config_id=(
                        str(neighbor.ref.get("config_id"))
                        if neighbor.ref is not None
                        and neighbor.ref.get("config_id") is not None
                        else None
                    ),
                    selected_config=(
                        dict(neighbor.config) if neighbor.config is not None else None
                    ),
                    raw_semantic_score=_optional_float(
                        neighbor.ref, "raw_semantic_score"
                    ),
                    raw_shape_score=_optional_float(neighbor.ref, "raw_shape_score"),
                    combined_score=_optional_float(neighbor.ref, "combined_score"),
                    rank_before_rerank=_optional_int(
                        neighbor.ref, "rank_before_rerank"
                    ),
                    rank_after_rerank=_optional_int(neighbor.ref, "rank_after_rerank"),
                )
                for neighbor in evidence.neighbors
            )
            if evidence is not None
            else (),
            candidate_source=(
                {
                    "frozen": "qwen",
                    "random": "random_neighbors",
                    "shuffled": "shuffled_neighbors",
                }[retrieval_control]
                if evidence and evidence.neighbors
                else None
            ),
            compatibility_status="compatible" if first_neighbor is not None else None,
        ),
        provider=ProviderRecord(
            request_id=provider_identity["request_id"],
            response_id=provider_identity["response_id"],
            cache_state=provider_identity["cache_state"],
            requests=tokens["requests"] or 0,
            input_tokens=tokens["input_tokens"],
            cached_input_tokens=tokens["cached_input_tokens"],
            output_tokens=tokens["output_tokens"],
            reasoning_tokens=tokens["reasoning_tokens"],
            replay_identities=tuple(
                ProviderReplayIdentity(
                    ordinal=ordinal,
                    request_hash=request_hash,
                    response_hash=response_hash,
                )
                for ordinal, (request_hash, response_hash) in enumerate(
                    provider_replay_identities
                )
            ),
        ),
        outcome=OutcomeRecord(
            decision_name=type(outcome.decision).__name__,
            compilation_status=(
                "ok" if compiled_ids else "not_run" if not evaluations else "failed"
            ),
            correctness_status=(
                "ok" if validated_ids else "not_run" if not evaluations else "failed"
            ),
            benchmark_status=(
                "ok" if benchmarked_ids else "not_run" if not evaluations else "failed"
            ),
            timeout_status=(
                "timed_out"
                if any(record.timeout_status == "timed_out" for record in evaluations)
                else "not_timed_out"
            ),
            fallback_reason=outcome.fallback_reason,
            accumulated_seconds=outcome.accumulated_seconds,
            censored=outcome.censored,
            incumbent_best_perf=incumbent_best_perf,
            selected_performance=selected_performance,
            terminal_error=terminal_error,
        ),
        timings=PhaseTimings(**timings),
        accounting=(
            AttemptAccountingRecord(
                frozen_limit=attempt_budget.limit,
                attempted=attempt_budget.spent(),
                initial_population=counts[AttemptCategory.INITIAL_POPULATION],
                invalid=counts[AttemptCategory.INVALID],
                duplicate=counts[AttemptCategory.DUPLICATE],
                generation=counts[AttemptCategory.GENERATION],
                llm_proposed=counts[AttemptCategory.LLM_PROPOSED],
                compiled=len(compiled_ids),
                validated=len(validated_ids),
                benchmarked=len(benchmarked_ids),
            )
            if attempt_budget.limit is not None
            else None
        ),
        evaluations=evaluations,
        phase_snapshots=instrumentation.phase_snapshots,
        hybrid_stage_breakdown=_hybrid_stage_record(cache.autotuner),
    )
    append_event(event)


def _baseline_tuner_mode(autotuner: BaseSearch) -> TunerMode:
    """Classify a non-RAG search for canonical event labeling."""
    from ..llm_search import LLMGuidedSearch
    from ..llm_seeded_lfbo import LLMSeededSearch

    if isinstance(autotuner, LLMSeededSearch):
        return TunerMode.HYBRID
    if isinstance(autotuner, LLMGuidedSearch):
        return TunerMode.LLM
    return TunerMode.LFBO


def emit_baseline_attempt_event(cache: AutotuneCacheBase) -> Config:
    """Run the plain (non-RAG) search and emit one canonical event for it.

    Gives RAG-disabled arms (lfbo / llm / hybrid) the same schema-valid
    ``InstrumentationEvent`` the RAG path emits, from a single collector so the
    hybrid's LFBO trajectory continues its LLM stage rather than restarting. The
    caller opts in via ``HELION_AUTOTUNE_EMIT_EVENT`` so ordinary autotuning --
    and the untouched three-arm campaign -- are unaffected.
    """
    tuner_mode = _baseline_tuner_mode(cache.autotuner)
    instrumentation = InstrumentationCollector()
    cache.autotuner.set_attempt_instrumentation(instrumentation)
    started = time.perf_counter()

    def _emit(outcome: ExecutionOutcome, *, terminal_error: str | None) -> None:
        emit_attempt_event(
            cache,
            tuner_mode=tuner_mode,
            descriptor=None,
            evidence=None,
            outcome=outcome,
            instrumentation=instrumentation,
            attempt_budget=cache.autotuner.candidate_attempt_budget,
            terminal_error=terminal_error,
        )

    try:
        try:
            config = cache._run_autotune_trials()
        except Exception as error:
            elapsed = time.perf_counter() - started
            _emit(
                ExecutionOutcome(
                    decision=BaselineSearch("baseline_error"),
                    result=None,
                    validated=False,
                    accumulated_seconds=elapsed,
                    timings=(PhaseTimingEvent(Phase.TOTAL, elapsed),),
                ),
                terminal_error=type(error).__name__,
            )
            raise
        elapsed = time.perf_counter() - started
        _emit(
            ExecutionOutcome(
                decision=BaselineSearch("baseline"),
                result=config,
                validated=True,
                accumulated_seconds=elapsed,
                timings=(PhaseTimingEvent(Phase.TOTAL, elapsed),),
            ),
            terminal_error=None,
        )
        return config
    finally:
        cache.autotuner.set_attempt_instrumentation(None)


def autotune_with_rag(cache: AutotuneCacheBase, *, skip_cache: bool = False) -> Config:
    """Run one live tiered RAG attempt while preserving cache semantics."""
    settings = cache.autotuner.settings
    if settings.autotune_best_of_k != 1:
        raise ValueError("RAG autotuning does not support best-of-K values above one")
    tuner_mode = resolve_tuner_mode(cache.autotuner)
    retrieval_control = _retrieval_control()
    live_autotuner = cast("PopulationBasedSearch", cache.autotuner)
    skip_cache_env = should_skip_cache()
    skip_exact = skip_cache or skip_cache_env or not settings.autotune_exact_read
    descriptor: WorkloadDescriptor | None = None
    evidence: RetrievalEvidence | None = None
    exact_cache_probe: str | None = None
    search_started = False
    extra_timings: list[PhaseTimingEvent] = []
    instrumentation = InstrumentationCollector()
    live_autotuner.set_attempt_instrumentation(instrumentation)

    def probe() -> ExactProbeResult:
        nonlocal exact_cache_probe
        if skip_exact:
            result = ExactMiss()
        else:
            result = probe_exact_cache(cache)
            exact_cache_probe = (
                "hit"
                if isinstance(result, ExactHit)
                else "read_error"
                if isinstance(result, ExactReadError)
                else "miss"
            )
        if isinstance(result, ExactHit):
            if not isinstance(result.config, Config):
                raise TypeError("exact cache returned a non-Config value")
            cache._record_cache_hit(result.config)
        else:
            cache._record_cache_miss(
                assert_cache_hit=isinstance(result, ExactMiss) and not skip_exact
            )
        return result

    def describe() -> WorkloadDescriptor:
        nonlocal descriptor
        descriptor = build_workload_descriptor(live_autotuner)
        return descriptor

    def retrieve(workload: WorkloadDescriptor) -> RetrievalEvidence:
        nonlocal evidence
        evidence = retrieve_evidence(workload, live_autotuner)
        return evidence

    def run_search() -> Config:
        nonlocal search_started
        if not search_started:
            cache.autotuner.log("Starting autotuning process, this may take a while...")
            search_started = True
        return cache._run_autotune_trials()

    baseline_autotuner: PopulationBasedSearch | None = None

    def run_failure_fallback() -> Config:
        nonlocal baseline_autotuner
        if tuner_mode is TunerMode.LFBO:
            return run_search()
        if baseline_autotuner is None:
            from ..effort_profile import get_effort_profile
            from ..surrogate_pattern_search import LFBOTreeSearch

            profile = get_effort_profile(settings.autotune_effort)
            kwargs = LFBOTreeSearch.get_kwargs_from_profile(profile, settings)
            baseline_autotuner = LFBOTreeSearch(
                live_autotuner.kernel,
                live_autotuner.args,
                **kwargs,  # pyrefly: ignore[bad-argument-type]
            )
            baseline_autotuner.set_candidate_attempt_budget(
                live_autotuner.candidate_attempt_budget
            )
            baseline_autotuner.set_attempt_instrumentation(instrumentation)
        return baseline_autotuner.autotune()

    treatment = TreatmentConfig(
        treatment_id=os.environ.get("HELION_RAG_TREATMENT_ID", "rag"),
        rag_enabled=True,
        exact_read=settings.autotune_exact_read,
        best_available_read=settings.autotune_best_available_read,
        write=settings.autotune_cache_write,
        qwen_enabled=True,
        tuner_mode=tuner_mode,
        allow_exact_reuse=False,
    )
    runner = _LiveDecisionRunner(
        live_autotuner,
        tuner_mode,
        run_search,
        instrumentation,
        best_available_read=treatment.best_available_read,
        retrieval_control=retrieval_control,
    )

    attempt_started = time.perf_counter()

    def observe_phase(event: PhaseTimingEvent) -> None:
        instrumentation.record_phase_transition(
            event.phase, phase_seconds=event.seconds
        )

    try:
        try:
            outcome = execute_rag_attempt(
                treatment,
                probe=probe,
                describe=describe,
                retrieve=retrieve,
                runner=runner,
                baseline=run_search,
                failure_fallback=run_failure_fallback,
                phase_observer=observe_phase,
            )
        except Exception as error:
            elapsed = time.perf_counter() - attempt_started
            failed_outcome = ExecutionOutcome(
                decision=BaselineSearch("propagated_error"),
                result=None,
                validated=False,
                accumulated_seconds=elapsed,
                timings=(PhaseTimingEvent(Phase.TOTAL, elapsed),),
            )
            emit_attempt_event(
                cache,
                tuner_mode=tuner_mode,
                descriptor=descriptor,
                evidence=evidence,
                outcome=failed_outcome,
                instrumentation=instrumentation,
                attempt_budget=live_autotuner.candidate_attempt_budget,
                terminal_error=type(error).__name__,
                retrieval_control=retrieval_control,
                exact_cache_probe=exact_cache_probe,
            )
            raise
    finally:
        live_autotuner.set_attempt_instrumentation(None)
        if baseline_autotuner is not None:
            baseline_autotuner.set_attempt_instrumentation(None)
    extra_timings = [*runner.timings, *extra_timings]
    if extra_timings:
        assert outcome.timings[-1].phase is Phase.TOTAL
        outcome = dataclasses.replace(
            outcome,
            timings=(*outcome.timings[:-1], *extra_timings, outcome.timings[-1]),
        )
    if not isinstance(outcome.result, Config):
        raise TypeError(
            f"RAG autotuning returned {type(outcome.result).__name__}, expected Config"
        )
    selected = outcome.result
    existing_exact_hit = (
        isinstance(outcome.decision, BaselineSearch)
        and outcome.decision.reason == REASON_EXISTING_EXACT_HIT
    )
    if not existing_exact_hit and not skip_cache_env and settings.autotune_cache_write:
        cache.put(selected)
        from ..._utils import counters

        counters["autotune"]["cache_put"] += 1
    emit_attempt_event(
        cache,
        tuner_mode=tuner_mode,
        descriptor=descriptor,
        evidence=evidence,
        outcome=outcome,
        instrumentation=instrumentation,
        attempt_budget=live_autotuner.candidate_attempt_budget,
        retrieval_control=retrieval_control,
        exact_cache_probe=exact_cache_probe,
    )
    return selected
