"""Offline harness for the four-arm head-to-head autotuning study.

Balanced scheduling, the frozen arm table, the benchmarkable workload registry, and
the uniform instrumentation event schema every arm emits. Pure/offline: importing
this package pulls no CUDA and needs no live provider.
"""

from __future__ import annotations

from .events import ArtifactIdentity as ArtifactIdentity
from .events import AttemptAccountingRecord as AttemptAccountingRecord
from .events import EvaluationRecord as EvaluationRecord
from .events import InstrumentationEvent as InstrumentationEvent
from .events import OutcomeRecord as OutcomeRecord
from .events import PhaseSnapshotRecord as PhaseSnapshotRecord
from .events import PhaseTimings as PhaseTimings
from .events import ProviderRecord as ProviderRecord
from .events import ProviderReplayIdentity as ProviderReplayIdentity
from .events import RetrievalRecord as RetrievalRecord
from .events import RunIdentity as RunIdentity
from .events import event_to_canonical_json as event_to_canonical_json
from .scheduler import ScheduledRun as ScheduledRun
from .scheduler import schedule as schedule
from .scheduler import verify_balance as verify_balance
