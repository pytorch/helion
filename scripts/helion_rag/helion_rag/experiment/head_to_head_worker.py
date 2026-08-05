"""GPU worker for one isolated four-arm head-to-head run over a workload.

The workload is selected by ``HELION_RAG_WORKLOAD_ID`` and the arm by
``HELION_RAG_ARM_ID``. Unlike the three-arm smoke worker, every arm here emits
exactly one canonical :class:`InstrumentationEvent` (the RAG-disabled arms via
``HELION_AUTOTUNE_EMIT_EVENT=1``), so this worker validates one event per arm and
enforces the per-arm one-call and retrieval policy.

Torch/Helion imports are deferred so the module is importable by the orchestrator
without pulling CUDA.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

from helion_rag.experiment.head_to_head import ARM_POLICIES
from helion_rag.experiment.head_to_head import ARM_TUNER_MODE
from helion_rag.experiment.workloads import DEFAULT_WORKLOAD as DEFAULT_WORKLOAD
from helion_rag.experiment.workloads import WORKLOADS as WORKLOADS


def _synchronize() -> None:
    import torch

    from helion._testing import DEVICE

    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def _seed_tensor_rng() -> None:
    """Seed workload tensor generation from the frozen arm seed."""
    import torch

    torch.manual_seed(int(os.environ["HELION_AUTOTUNE_RANDOM_SEED"]))


def _event_group(event: dict[str, object] | None, name: str) -> dict[str, object]:
    value = event.get(name) if event is not None else None
    return value if isinstance(value, dict) else {}


def _last_event(arm: str, kernel_name: str) -> dict[str, object]:
    """Read and validate the single canonical event this arm must emit."""
    policy = ARM_POLICIES[arm]
    value = os.environ.get("HELION_RAG_EVENT_LOG")
    if not value:
        raise ValueError("head-to-head run has no event log path")
    path = Path(value)
    lines = (
        [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        if path.is_file()
        else []
    )
    if len(lines) != 1:
        raise ValueError(f"expected exactly 1 canonical event, observed {len(lines)}")
    event = json.loads(lines[0])
    if not isinstance(event, dict):
        raise ValueError("canonical event must be a JSON object")

    run = _event_group(event, "run")
    expected_identity: dict[str, object] = {
        "arm_id": arm,
        "treatment_id": arm,
        "repetition": int(os.environ["HELION_RAG_REPETITION"]),
        "random_seed": int(os.environ["HELION_AUTOTUNE_RANDOM_SEED"]),
        "kernel_name": kernel_name,
        "workload_id": os.environ["HELION_RAG_WORKLOAD_ID"],
        "tuner_mode": ARM_TUNER_MODE[arm],
    }
    for field, expected_value in expected_identity.items():
        if run.get(field) != expected_value:
            raise ValueError(
                f"event {field}={run.get(field)!r}, expected {expected_value!r}"
            )

    accounting = _event_group(event, "accounting")
    frozen_limit = int(os.environ["HELION_AUTOTUNE_CANDIDATE_ATTEMPT_LIMIT"])
    if accounting.get("frozen_limit") != frozen_limit:
        raise ValueError("event candidate-attempt limit does not match the arm")
    attempted = accounting.get("attempted")
    if isinstance(attempted, int) and attempted > frozen_limit:
        raise ValueError(f"arm {arm} attempted {attempted} > limit {frozen_limit}")

    provider = _event_group(event, "provider")
    if provider.get("requests") != policy.provider_requests:
        raise ValueError(
            f"arm {arm} made {provider.get('requests')!r} provider requests, "
            f"expected {policy.provider_requests}"
        )

    retrieval = _event_group(event, "retrieval")
    if not policy.rag_enabled and retrieval.get("lookup_tier") is not None:
        raise ValueError(f"retrieval-disabled arm {arm} reported a lookup tier")
    return event


def main() -> int:
    import torch

    from helion._testing import run_example

    arm = os.environ["HELION_RAG_ARM_ID"]
    if arm not in ARM_POLICIES:
        raise ValueError(f"unknown arm {arm!r}")
    workload = WORKLOADS[os.environ["HELION_RAG_WORKLOAD_ID"]]
    _seed_tensor_rng()
    helion_fn, reference_fn, args = workload.build()
    _synchronize()
    start = time.perf_counter()
    actual = helion_fn(*args)
    _synchronize()
    readiness = time.perf_counter() - start
    # Correctness is a recorded outcome, not a smoke assertion: a mismatch yields
    # correct=False data rather than aborting the campaign.
    try:
        torch.testing.assert_close(
            actual, reference_fn(*args), rtol=workload.rtol, atol=workload.atol
        )
        correct = True
    except AssertionError:
        correct = False

    selected_latency: float | None = None
    if correct:
        # Stabilized post-search selected latency; kept separate from the event's
        # internal candidate benchmarks so trajectory and final-latency analyses
        # never mix the two measurement sources. Pass the workload's tolerances so
        # run_example's own correctness re-check matches this run's gate (its
        # defaults are stricter than some workloads, e.g. split-K's atol=1.0); a
        # failure there is an incorrect result, not a harness crash.
        try:
            timings = run_example(
                helion_fn, reference_fn, args, rtol=workload.rtol, atol=workload.atol
            )  # pyrefly: ignore[bad-argument-type]
            selected_latency = timings["helion"]
        except AssertionError:
            correct = False
            selected_latency = None

    event = _last_event(arm, workload.kernel_name)
    retrieval = _event_group(event, "retrieval")
    outcome = _event_group(event, "outcome")
    payload = {
        "readiness_seconds": readiness,
        "selected_latency_ms": selected_latency,
        "correct": correct,
        "tier": retrieval.get("lookup_tier"),
        "decision": outcome.get("decision_name"),
        "fallback": outcome.get("fallback_reason"),
        "provider": _event_group(event, "provider"),
        "attempt_accounting": _event_group(event, "accounting"),
        "incumbent_best_perf": outcome.get("incumbent_best_perf"),
        "internal_selected_performance": outcome.get("selected_performance"),
        "evaluation_count": (
            len(event["evaluations"])
            if isinstance(event.get("evaluations"), list)
            else None
        ),
        "hybrid_stage_breakdown": event.get("hybrid_stage_breakdown"),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
