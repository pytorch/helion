"""The pure RAG decision policy.

:func:`decide` maps frozen retrieval evidence and a frozen treatment to exactly
one :data:`~helion.autotuner.rag.types.Decision`. It is a pure function: no I/O,
no clock, no mutation, no heavy imports. All validation, compilation,
benchmarking, fallback, timing, and quarantine live in the execution wrapper.

See ``docs/rag_autotuning_experiment.md``.
"""

from __future__ import annotations

from .types import REASON_EXACT_REUSE_NOT_PERMITTED
from .types import REASON_NO_NEIGHBORS
from .types import REASON_RAG_DISABLED
from .types import REASON_SEMANTIC_DISABLED
from .types import REASON_TIER2_MISS
from .types import BaselineSearch
from .types import ContextualSearch
from .types import Decision
from .types import ExactReuse
from .types import LookupTier
from .types import RetrievalEvidence
from .types import RetrievalSeededSearch
from .types import TreatmentConfig
from .types import TunerMode


def decide(evidence: RetrievalEvidence, treatment: TreatmentConfig) -> Decision:
    """Return the single closed decision for this evidence and treatment.

    A disabled treatment, an ineligible Tier-0 hit, disabled semantic retrieval,
    an empty neighbor set, and a Tier-2 miss all deterministically delegate to
    the unchanged full tuner via :class:`BaselineSearch` with a stable reason.
    """
    if not treatment.rag_enabled:
        return BaselineSearch(REASON_RAG_DISABLED)

    if evidence.tier is LookupTier.EXACT:
        if (
            treatment.allow_exact_reuse
            and evidence.exact_eligible
            and evidence.exact_config is not None
        ):
            return ExactReuse(
                config=evidence.exact_config,
                provenance=evidence.exact_provenance,
            )
        return BaselineSearch(REASON_EXACT_REUSE_NOT_PERMITTED)

    if evidence.tier is LookupTier.SEMANTIC:
        if not treatment.qwen_enabled:
            return BaselineSearch(REASON_SEMANTIC_DISABLED)
        if not evidence.neighbors:
            return BaselineSearch(REASON_NO_NEIGHBORS)
        if treatment.tuner_mode is TunerMode.LFBO:
            return RetrievalSeededSearch(neighbors=evidence.neighbors)
        return ContextualSearch(neighbors=evidence.neighbors)

    return BaselineSearch(REASON_TIER2_MISS)
