"""Unit tests for the four-arm head-to-head harness (arms, controls, schedule)."""

from __future__ import annotations

import pytest

from helion_rag.experiment.head_to_head import ARM_CONTEXTUAL_RAG_LLM
from helion_rag.experiment.head_to_head import ARM_HYBRID
from helion_rag.experiment.head_to_head import ARM_LFBO
from helion_rag.experiment.head_to_head import ARM_LLM
from helion_rag.experiment.head_to_head import ARMS
from helion_rag.experiment.head_to_head import build_run_environment
from helion_rag.experiment.head_to_head import build_schedule
from helion_rag.experiment.head_to_head import build_study_manifest
from helion_rag.experiment.head_to_head import manifest_matches
from helion_rag.experiment.head_to_head import registered_workloads
from helion_rag.experiment.scheduler import ScheduledRun
from helion_rag.experiment.scheduler import verify_balance

_WORKLOADS = [f"wl{i}" for i in range(15)]


def _run(arm: str, *, workload: str = "wl0", repetition: int = 1, order: int = 0):
    return ScheduledRun(
        workload=workload, arm=arm, repetition=repetition, order_index=order
    )


def _env(arm: str, **kwargs) -> dict[str, str]:
    return build_run_environment({}, _run(arm), output_dir="/tmp/h2h", **kwargs)


def test_arms_map_to_tuner_rag_and_one_call_policy() -> None:
    from helion_rag.experiment.head_to_head import ARM_POLICIES

    expected = {
        ARM_LFBO: ("LFBOTreeSearch", False, 0),
        ARM_LLM: ("LLMGuidedSearch", False, 1),
        ARM_HYBRID: ("LLMSeededLFBOTreeSearch", False, 1),
        ARM_CONTEXTUAL_RAG_LLM: ("LLMGuidedSearch", True, 1),
    }
    for arm, (autotuner, rag_enabled, provider_requests) in expected.items():
        policy = ARM_POLICIES[arm]
        assert policy.autotuner == autotuner
        assert policy.rag_enabled is rag_enabled
        assert policy.provider_requests == provider_requests
        assert _env(arm)["HELION_AUTOTUNER"] == autotuner


def test_only_contextual_arm_enables_retrieval() -> None:
    assert _env(ARM_CONTEXTUAL_RAG_LLM)["HELION_RAG_ENABLED"] == "1"
    for arm in (ARM_LFBO, ARM_LLM, ARM_HYBRID):
        assert _env(arm)["HELION_RAG_ENABLED"] == "0"


def test_all_arms_share_frozen_controls_and_emit_events() -> None:
    frozen = {
        "HELION_AUTOTUNE_EMIT_EVENT": "1",
        "HELION_AUTOTUNE_EXACT_READ": "0",
        "HELION_AUTOTUNE_BEST_AVAILABLE_READ": "0",
        "HELION_AUTOTUNE_CACHE_WRITE": "0",
        "HELION_AUTOTUNE_BEST_OF_K": "1",
        "HELION_AUTOTUNE_EFFORT": "quick",
        "HELION_AUTOTUNE_CANDIDATE_ATTEMPT_LIMIT": "80",
        "HELION_AUTOTUNE_DISABLE_TRAJECTORY_EARLY_STOP": "1",
    }
    for arm in ARMS:
        env = _env(arm)
        for key, value in frozen.items():
            assert env[key] == value, (arm, key)


def test_seeds_and_isolated_paths_are_matched_and_unique() -> None:
    # Same repetition -> identical seed across arms; per-run isolated directories.
    seen_caches: set[str] = set()
    for order, arm in enumerate(ARMS):
        env = build_run_environment(
            {}, _run(arm, repetition=3, order=order), output_dir="/tmp/h2h"
        )
        assert env["HELION_AUTOTUNE_RANDOM_SEED"] == "1002"  # base 1000 + rep 3 - 1
        assert env["HELION_CACHE_DIR"] not in seen_caches
        seen_caches.add(env["HELION_CACHE_DIR"])
        assert env["TORCHINDUCTOR_CACHE_DIR"].startswith(env["HELION_CACHE_DIR"])
        assert env["TRITON_CACHE_DIR"].startswith(env["HELION_CACHE_DIR"])
        assert arm in env["HELION_RAG_EVENT_LOG"]


def test_llm_and_contextual_share_identical_llm_settings() -> None:
    llm = _env(ARM_LLM)
    contextual = build_run_environment(
        {}, _run(ARM_CONTEXTUAL_RAG_LLM), output_dir="/tmp/h2h"
    )
    for key in (
        "HELION_AUTOTUNER",
        "HELION_LLM_PROVIDER",
        "HELION_LLM_MODEL",
        "HELION_AUTOTUNE_EFFORT",
        "HELION_AUTOTUNE_CANDIDATE_ATTEMPT_LIMIT",
        "HELION_AUTOTUNE_RANDOM_SEED",
    ):
        assert llm[key] == contextual[key], key
    # The retrieval flag is the only intended treatment difference.
    assert llm["HELION_RAG_ENABLED"] != contextual["HELION_RAG_ENABLED"]


def test_build_run_environment_rejects_unknown_arm_and_bad_limit() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        build_run_environment({}, _run("nope"), output_dir="/tmp/h2h")
    with pytest.raises(ValueError, match="positive"):
        build_run_environment(
            {}, _run(ARM_LFBO), output_dir="/tmp/h2h", candidate_attempt_limit=0
        )


def test_schedule_is_deterministic_balanced_and_full() -> None:
    runs_a = build_schedule(_WORKLOADS)
    runs_b = build_schedule(_WORKLOADS)
    assert runs_a == runs_b  # deterministic for a fixed seed
    assert len(runs_a) == len(_WORKLOADS) * len(ARMS) * 5  # 15 x 4 x 5 = 300
    assert verify_balance(runs_a, _WORKLOADS, ARMS, repetitions=5)
    # Every (workload, arm) pair appears exactly `repetitions` times.
    from collections import Counter

    pairs = Counter((run.workload, run.arm) for run in runs_a)
    assert set(pairs.values()) == {5}


def test_registry_includes_core_and_extra_shapes() -> None:
    from helion_rag.experiment.workloads.head_to_head_extra import (
        HEAD_TO_HEAD_EXTRA_IDS,
    )

    registry = set(registered_workloads())
    # The 15 original core kernels remain registered ...
    core = {
        "add-1024x1024",
        "attention-2x8x512x64",
        "matmul-1024x1024x1024",
        "rms_norm-2048x1024",
        "softmax-2048x2048",
    }
    assert core <= registry
    # ... and the 33-shape head-to-head run set is fully registered.
    assert len(HEAD_TO_HEAD_EXTRA_IDS) == 33
    assert set(HEAD_TO_HEAD_EXTRA_IDS) <= registry


def test_manifest_freezes_controls_and_detects_tampering() -> None:
    manifest = build_study_manifest(
        workloads=_WORKLOADS, generation_id="000000", code_id="deadbeef"
    )
    assert manifest["total_run_units"] == 300
    assert manifest["candidate_attempt_limit"] == 80
    assert manifest["random_seeds"] == [1000, 1001, 1002, 1003, 1004]
    assert manifest["provider"] == "vertex"
    assert manifest["model"] == "claude-opus-4-8"
    assert manifest["generation_id"] == "000000"
    assert manifest_matches(manifest)
    # Tampering with any frozen field invalidates the recorded hash.
    tampered = dict(manifest)
    tampered["candidate_attempt_limit"] = 40
    assert not manifest_matches(tampered)
