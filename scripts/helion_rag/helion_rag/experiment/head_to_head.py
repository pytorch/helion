"""Frozen four-arm head-to-head autotuning harness (arms, controls, manifest).

This generalizes the three-arm :mod:`helion_rag.experiment.comparison` smoke
driver into a table-driven four-arm comparison. It does not touch the three-arm
module; the resumable overnight driver lives in ``run_head_to_head_campaign.py``.

The four arms and their frozen treatment:

======================  ===================  ===========  ==========
arm id                  HELION_AUTOTUNER     retrieval    LLM calls
======================  ===================  ===========  ==========
``lfbo``                LFBOTreeSearch       off          0
``llm``                 LLMGuidedSearch      off          1
``hybrid_lfbo_llm``     LLMSeededLFBOTree..  off          1
``contextual_rag_llm``  LLMGuidedSearch      on           1
======================  ===================  ===========  ==========

``contextual_rag_llm`` is the only retrieval-enabled arm; ``llm``, the hybrid's
LLM stage, and ``contextual_rag_llm`` share identical provider/model/round
settings so the retrieval contrast is the only difference between ``llm`` and
``contextual_rag_llm``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from helion_rag.experiment.scheduler import schedule
from helion_rag.experiment.scheduler import verify_balance
from helion_rag.experiment.workloads import WORKLOADS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from helion_rag.experiment.scheduler import ScheduledRun

# ── Stable arm identities ───────────────────────────────────────────────
ARM_LFBO = "lfbo"
ARM_LLM = "llm"
ARM_HYBRID = "hybrid_lfbo_llm"
ARM_CONTEXTUAL_RAG_LLM = "contextual_rag_llm"
ARMS = (ARM_LFBO, ARM_LLM, ARM_HYBRID, ARM_CONTEXTUAL_RAG_LLM)


@dataclasses.dataclass(frozen=True)
class ArmPolicy:
    """The frozen treatment for one arm: tuner, retrieval, and one-call policy."""

    arm_id: str
    autotuner: str
    rag_enabled: bool
    provider_requests: int


ARM_POLICIES: dict[str, ArmPolicy] = {
    ARM_LFBO: ArmPolicy(
        ARM_LFBO, "LFBOTreeSearch", rag_enabled=False, provider_requests=0
    ),
    ARM_LLM: ArmPolicy(
        ARM_LLM, "LLMGuidedSearch", rag_enabled=False, provider_requests=1
    ),
    ARM_HYBRID: ArmPolicy(
        ARM_HYBRID, "LLMSeededLFBOTreeSearch", rag_enabled=False, provider_requests=1
    ),
    ARM_CONTEXTUAL_RAG_LLM: ArmPolicy(
        ARM_CONTEXTUAL_RAG_LLM, "LLMGuidedSearch", rag_enabled=True, provider_requests=1
    ),
}

# The event's RunIdentity.tuner_mode value expected for each arm.
ARM_TUNER_MODE: dict[str, str] = {
    ARM_LFBO: "lfbo",
    ARM_LLM: "llm",
    ARM_HYBRID: "hybrid",
    ARM_CONTEXTUAL_RAG_LLM: "llm",
}

# ── Frozen controls (spec defaults) ─────────────────────────────────────
DEFAULT_REPETITIONS = 5
DEFAULT_BASE_SEED = 1000
DEFAULT_SCHEDULE_SEED = 20260722
DEFAULT_CANDIDATE_ATTEMPT_LIMIT = 80
DEFAULT_EFFORT = "quick"
DEFAULT_PROVIDER = "vertex"
DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_TIMEOUT_SECONDS = 1800.0
# quick-effort LLM knobs (from the effort profile) recorded for provenance.
LLM_ROUNDS = 1
LLM_CONFIGS_PER_ROUND = 15
LLM_INITIAL_RANDOM_CONFIGS = 10
LLM_COMPILE_TIMEOUT_S = 15


def build_run_environment(
    base: dict[str, str],
    run: ScheduledRun,
    *,
    output_dir: str | Path,
    base_seed: int = DEFAULT_BASE_SEED,
    effort: str = DEFAULT_EFFORT,
    candidate_attempt_limit: int = DEFAULT_CANDIDATE_ATTEMPT_LIMIT,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
) -> dict[str, str]:
    """Return the isolated, cold, equal-control environment for one run unit.

    Every arm shares the same frozen controls; only ``HELION_AUTOTUNER`` and
    ``HELION_RAG_ENABLED`` vary by arm. ``HELION_AUTOTUNE_EMIT_EVENT`` makes the
    RAG-disabled arms emit the same canonical event the RAG arm emits.
    """
    if candidate_attempt_limit <= 0:
        raise ValueError("candidate_attempt_limit must be positive")
    policy = ARM_POLICIES.get(run.arm)
    if policy is None:
        raise ValueError(f"unknown arm {run.arm!r}; choose from {sorted(ARM_POLICIES)}")
    output = Path(output_dir)
    run_id = f"{run.order_index:04d}-{run.workload}-{run.arm}-r{run.repetition}"
    cache_dir = output / "cache" / run_id
    env = dict(base)
    env.update(
        {
            "HELION_AUTOTUNER": policy.autotuner,
            "HELION_RAG_ENABLED": "1" if policy.rag_enabled else "0",
            "HELION_AUTOTUNE_EMIT_EVENT": "1",
            "HELION_AUTOTUNE_EXACT_READ": "0",
            "HELION_AUTOTUNE_BEST_AVAILABLE_READ": "0",
            "HELION_AUTOTUNE_CACHE_WRITE": "0",
            "HELION_AUTOTUNE_BEST_OF_K": "1",
            "HELION_AUTOTUNE_EFFORT": effort,
            "HELION_AUTOTUNE_CANDIDATE_ATTEMPT_LIMIT": str(candidate_attempt_limit),
            "HELION_AUTOTUNE_DISABLE_TRAJECTORY_EARLY_STOP": "1",
            "HELION_AUTOTUNE_RANDOM_SEED": str(base_seed + run.repetition - 1),
            # Identical LLM settings across llm / hybrid / contextual (inert for lfbo).
            "HELION_LLM_PROVIDER": provider,
            "HELION_LLM_MODEL": model,
            # Isolated caches, event log, and scratch dir for this run unit.
            "HELION_CACHE_DIR": str(cache_dir),
            "TORCHINDUCTOR_CACHE_DIR": str(cache_dir / "torchinductor"),
            "TRITON_CACHE_DIR": str(cache_dir / "triton"),
            "TMPDIR": str(output / "tmp" / run_id),
            "HELION_RAG_EVENT_LOG": str(output / "events" / f"{run_id}.jsonl"),
            "HELION_RAG_ARM_ID": run.arm,
            "HELION_RAG_TREATMENT_ID": run.arm,
            "HELION_RAG_REPETITION": str(run.repetition),
            "HELION_RAG_WORKLOAD_ID": run.workload,
        }
    )
    return env


def build_schedule(
    workloads: Sequence[str],
    *,
    schedule_seed: int = DEFAULT_SCHEDULE_SEED,
    repetitions: int = DEFAULT_REPETITIONS,
) -> list[ScheduledRun]:
    """Build one balanced Latin-square schedule over all workloads x arms x reps."""
    runs = schedule(workloads, ARMS, seed=schedule_seed, repetitions=repetitions)
    if not verify_balance(runs, workloads, ARMS, repetitions=repetitions):
        raise AssertionError("four-arm schedule is not position-balanced")
    return runs


def registered_workloads() -> list[str]:
    """Return the sorted registry of evaluation workload ids."""
    return sorted(WORKLOADS)


def build_study_manifest(
    *,
    workloads: Sequence[str],
    repetitions: int = DEFAULT_REPETITIONS,
    base_seed: int = DEFAULT_BASE_SEED,
    schedule_seed: int = DEFAULT_SCHEDULE_SEED,
    candidate_attempt_limit: int = DEFAULT_CANDIDATE_ATTEMPT_LIMIT,
    effort: str = DEFAULT_EFFORT,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    generation_id: str,
    code_id: str,
) -> dict[str, object]:
    """Build the immutable study manifest freezing every experimental control.

    The returned mapping includes a ``config_hash`` over all frozen fields so a
    resume can refuse to mix results produced under a different configuration.
    """
    workloads = list(workloads)
    manifest: dict[str, object] = {
        "manifest_version": 1,
        "arms": list(ARMS),
        "arm_policies": {
            arm: {
                "autotuner": policy.autotuner,
                "rag_enabled": policy.rag_enabled,
                "provider_requests": policy.provider_requests,
                "tuner_mode": ARM_TUNER_MODE[arm],
            }
            for arm, policy in ARM_POLICIES.items()
        },
        "workloads": workloads,
        "repetitions": repetitions,
        "base_seed": base_seed,
        "random_seeds": [base_seed + rep for rep in range(repetitions)],
        "schedule_seed": schedule_seed,
        "candidate_attempt_limit": candidate_attempt_limit,
        "effort": effort,
        "best_of_k": 1,
        "provider": provider,
        "model": model,
        "llm_rounds": LLM_ROUNDS,
        "llm_configs_per_round": LLM_CONFIGS_PER_ROUND,
        "llm_initial_random_configs": LLM_INITIAL_RANDOM_CONFIGS,
        "llm_compile_timeout_s": LLM_COMPILE_TIMEOUT_S,
        "cache_policy": {
            "exact_read": False,
            "best_available_read": False,
            "write": False,
        },
        "disable_trajectory_early_stop": True,
        "generation_id": generation_id,
        "timeout_seconds": timeout_seconds,
        "total_run_units": len(workloads) * len(ARMS) * repetitions,
        "code_id": code_id,
    }
    manifest["config_hash"] = _config_hash(manifest)
    return manifest


def _config_hash(manifest: dict[str, object]) -> str:
    """Deterministic SHA-256 over every frozen field (excluding the hash itself)."""
    payload = {key: value for key, value in manifest.items() if key != "config_hash"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def manifest_matches(manifest: dict[str, object]) -> bool:
    """Return whether a manifest's recorded ``config_hash`` matches its fields."""
    recorded = manifest.get("config_hash")
    return isinstance(recorded, str) and recorded == _config_hash(manifest)
