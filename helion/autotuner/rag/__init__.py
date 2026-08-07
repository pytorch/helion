"""Opt-in tiered RAG autotuning policy (experimental).

This package is off the autotuner hot path unless explicitly enabled. Importing
it pulls in no retrieval dependency (faiss / torch / qwen); those load lazily
only after the kill switch and the pure decision have run.
"""

from __future__ import annotations

from .execution import REASON_EXACT_READ_ERROR as REASON_EXACT_READ_ERROR
from .execution import REASON_EXISTING_EXACT_HIT as REASON_EXISTING_EXACT_HIT
from .execution import REASON_RETRIEVAL_FAILURE as REASON_RETRIEVAL_FAILURE
from .execution import DecisionRunner as DecisionRunner
from .execution import execute_rag_attempt as execute_rag_attempt
from .fallback import classify_fallback as classify_fallback
from .fallback import execute_with_fallback as execute_with_fallback
from .instrumentation import EvaluationRecord as EvaluationRecord
from .instrumentation import InstrumentationCollector as InstrumentationCollector
from .instrumentation import PhaseSnapshot as PhaseSnapshot
from .killswitch import rag_enabled as rag_enabled
from .killswitch import rag_enabled_env as rag_enabled_env
from .policy import decide as decide
from .probe import probe_exact_cache as probe_exact_cache
from .seeding import AttemptBudget as AttemptBudget
from .seeding import AttemptCategory as AttemptCategory
from .seeding import PopulationUnderfilled as PopulationUnderfilled
from .seeding import SeedingResult as SeedingResult
from .seeding import build_seeded_population as build_seeded_population
from .types import BaselineSearch as BaselineSearch
from .types import ContextualSearch as ContextualSearch
from .types import Decision as Decision
from .types import ExactHit as ExactHit
from .types import ExactMiss as ExactMiss
from .types import ExactProbeResult as ExactProbeResult
from .types import ExactReadError as ExactReadError
from .types import ExactReuse as ExactReuse
from .types import ExecutionOutcome as ExecutionOutcome
from .types import FallbackReason as FallbackReason
from .types import LookupTier as LookupTier
from .types import Phase as Phase
from .types import PhaseTimingEvent as PhaseTimingEvent
from .types import RetrievalEvidence as RetrievalEvidence
from .types import RetrievalSeededSearch as RetrievalSeededSearch
from .types import RetrievedNeighbor as RetrievedNeighbor
from .types import TreatmentConfig as TreatmentConfig
from .types import TunerMode as TunerMode
from .types import WorkloadDescriptor as WorkloadDescriptor
