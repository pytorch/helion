from __future__ import annotations

import dataclasses

import pytest

from helion.autotuner.rag import BaselineSearch
from helion.autotuner.rag import ContextualSearch
from helion.autotuner.rag import Decision
from helion.autotuner.rag import ExactReuse
from helion.autotuner.rag import LookupTier
from helion.autotuner.rag import RetrievalEvidence
from helion.autotuner.rag import RetrievalSeededSearch
from helion.autotuner.rag import RetrievedNeighbor
from helion.autotuner.rag import TreatmentConfig
from helion.autotuner.rag import TunerMode
from helion.autotuner.rag import decide
from helion.autotuner.rag.types import REASON_EXACT_REUSE_NOT_PERMITTED
from helion.autotuner.rag.types import REASON_NO_NEIGHBORS
from helion.autotuner.rag.types import REASON_RAG_DISABLED
from helion.autotuner.rag.types import REASON_SEMANTIC_DISABLED
from helion.autotuner.rag.types import REASON_TIER2_MISS

_NEIGHBOR = RetrievedNeighbor(
    kernel_name="k",
    input_shapes="[(1024, 1024)]",
    dtypes="['torch.float32']",
    score=0.97,
    config={"num_warps": 8},
)


def _treatment(**overrides: object) -> TreatmentConfig:
    base: dict[str, object] = {
        "treatment_id": "t",
        "rag_enabled": True,
        "qwen_enabled": True,
        "tuner_mode": TunerMode.LFBO,
        "allow_exact_reuse": True,
    }
    base.update(overrides)
    return TreatmentConfig(**base)  # type: ignore[arg-type]


def test_rag_disabled_always_baseline() -> None:
    ev = RetrievalEvidence(
        tier=LookupTier.EXACT, exact_config={"a": 1}, exact_eligible=True
    )
    d = decide(ev, _treatment(rag_enabled=False))
    assert d == BaselineSearch(REASON_RAG_DISABLED)


def test_tier0_eligible_reuses_config() -> None:
    ev = RetrievalEvidence(
        tier=LookupTier.EXACT,
        exact_config={"num_warps": 4},
        exact_provenance={"run_id": "r1"},
        exact_eligible=True,
    )
    d = decide(ev, _treatment())
    assert d == ExactReuse(config={"num_warps": 4}, provenance={"run_id": "r1"})


def test_tier0_not_eligible_falls_back() -> None:
    ev = RetrievalEvidence(
        tier=LookupTier.EXACT, exact_config={"a": 1}, exact_eligible=False
    )
    assert decide(ev, _treatment()) == BaselineSearch(REASON_EXACT_REUSE_NOT_PERMITTED)


def test_tier0_reuse_disallowed_falls_back() -> None:
    # e.g. cross-architecture evidence can never produce Tier 0 reuse.
    ev = RetrievalEvidence(
        tier=LookupTier.EXACT, exact_config={"a": 1}, exact_eligible=True
    )
    assert decide(ev, _treatment(allow_exact_reuse=False)) == BaselineSearch(
        REASON_EXACT_REUSE_NOT_PERMITTED
    )


def test_tier0_missing_config_falls_back() -> None:
    ev = RetrievalEvidence(
        tier=LookupTier.EXACT, exact_config=None, exact_eligible=True
    )
    assert decide(ev, _treatment()) == BaselineSearch(REASON_EXACT_REUSE_NOT_PERMITTED)


def test_tier1_lfbo_seeds_population() -> None:
    ev = RetrievalEvidence(tier=LookupTier.SEMANTIC, neighbors=(_NEIGHBOR,))
    assert decide(ev, _treatment(tuner_mode=TunerMode.LFBO)) == RetrievalSeededSearch(
        neighbors=(_NEIGHBOR,)
    )


def test_tier1_llm_supplies_context() -> None:
    ev = RetrievalEvidence(tier=LookupTier.SEMANTIC, neighbors=(_NEIGHBOR,))
    assert decide(ev, _treatment(tuner_mode=TunerMode.LLM)) == ContextualSearch(
        neighbors=(_NEIGHBOR,)
    )


def test_tier1_qwen_disabled_falls_back() -> None:
    ev = RetrievalEvidence(tier=LookupTier.SEMANTIC, neighbors=(_NEIGHBOR,))
    assert decide(ev, _treatment(qwen_enabled=False)) == BaselineSearch(
        REASON_SEMANTIC_DISABLED
    )


def test_tier1_no_neighbors_falls_back() -> None:
    ev = RetrievalEvidence(tier=LookupTier.SEMANTIC, neighbors=())
    assert decide(ev, _treatment()) == BaselineSearch(REASON_NO_NEIGHBORS)


def test_tier2_miss_falls_back() -> None:
    ev = RetrievalEvidence(tier=LookupTier.MISS)
    assert decide(ev, _treatment()) == BaselineSearch(REASON_TIER2_MISS)


@pytest.mark.parametrize("tier", list(LookupTier))
def test_decide_is_total_over_tiers(tier: LookupTier) -> None:
    ev = RetrievalEvidence(
        tier=tier, exact_config={"a": 1}, exact_eligible=True, neighbors=(_NEIGHBOR,)
    )
    assert isinstance(decide(ev, _treatment()), Decision)


def test_decide_is_pure_and_deterministic() -> None:
    ev = RetrievalEvidence(tier=LookupTier.SEMANTIC, neighbors=(_NEIGHBOR,))
    tr = _treatment()
    ev_snapshot = dataclasses.replace(ev)
    first = decide(ev, tr)
    second = decide(ev, tr)
    assert first == second
    assert ev == ev_snapshot  # inputs are frozen and untouched
