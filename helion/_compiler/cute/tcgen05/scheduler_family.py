"""Which persistent-loop family a tcgen05 matmul plan's warp roles run.

``select_scheduler_family`` is the one decision the role-local builders in
``scheduler_pids.py`` used to spread over three dispatch ladders (the
consumer dispatch, the grouped-static dispatch and the scheduler-warp
dispatch) and four plan predicates.  The consumer families: the per-role
static scheduler (``STATIC``), the scheduler-warp mailbox consumers
(``SCHEDULER_WARP`` with the static producer, ``CLC`` with the
cluster-launch-control producer), the grouped group-search scheduler
(``GROUPED_STATIC``, specialized to a constant schedule for a few static
shapes by ``GROUPED_STATIC_SPECIALIZED``) and the N,M worklist consumers
(``GROUPED_RUNTIME_DIRECT`` reading the runtime tile table without a
pipeline, ``GROUPED_MAILBOX`` fed by the group-search scheduler warp,
``GROUPED_RUNTIME_CLC`` fed by the CLC scheduler warp's record ids).
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

from ..device_state import Tcgen05GroupedSchedulerMode
from ..tcgen05_constants import TCGEN05_GROUPED_STATIC_SPECIALIZATION_MAX_GROUPS

if TYPE_CHECKING:
    from ..device_state import CuteTcgen05MatmulPlan


class SchedulerFamily(enum.Enum):
    STATIC = "static"
    SCHEDULER_WARP = "scheduler_warp"
    CLC = "clc"
    GROUPED_STATIC = "grouped_static"
    GROUPED_STATIC_SPECIALIZED = "grouped_static_specialized"
    GROUPED_RUNTIME_DIRECT = "grouped_runtime_direct"
    GROUPED_MAILBOX = "grouped_mailbox"
    GROUPED_RUNTIME_CLC = "grouped_runtime_clc"

    @property
    def uses_worklist(self) -> bool:
        """The N,M worklist consumer (runtime table or selected mailbox)."""
        return self in (
            SchedulerFamily.GROUPED_RUNTIME_DIRECT,
            SchedulerFamily.GROUPED_MAILBOX,
            SchedulerFamily.GROUPED_RUNTIME_CLC,
        )

    @property
    def uses_runtime_table(self) -> bool:
        """The worklist consumer reads the runtime tile-record table."""
        return self in (
            SchedulerFamily.GROUPED_RUNTIME_DIRECT,
            SchedulerFamily.GROUPED_RUNTIME_CLC,
        )

    @property
    def uses_clc(self) -> bool:
        """The scheduler warp queries cluster launch control."""
        return self in (SchedulerFamily.CLC, SchedulerFamily.GROUPED_RUNTIME_CLC)


def select_scheduler_family(plan: CuteTcgen05MatmulPlan | None) -> SchedulerFamily:
    """The family of ``plan`` (``None``, the unit tests' planless builders, is
    the static family).  The order is the one the dispatch ladders had: the
    worklist pipelines first, then any scheduler warp, then the grouped
    schedulers, then the per-role static scheduler."""
    if plan is None:
        return SchedulerFamily.STATIC
    grouped = plan.grouped
    if grouped is not None:
        if grouped.scheduler_mode is Tcgen05GroupedSchedulerMode.RUNTIME_CLC:
            return SchedulerFamily.GROUPED_RUNTIME_CLC
        if (
            (plan.accumulator_view == "nm" or grouped.device_split_sizes)
            and plan.has_scheduler_warp
            and plan.uses_role_local_persistent_body
            and grouped.scheduler_mode
            is Tcgen05GroupedSchedulerMode.DEVICE_GROUP_SEARCH
        ):
            return SchedulerFamily.GROUPED_MAILBOX
    if plan.has_scheduler_warp:
        if plan.is_clc_persistent:
            return SchedulerFamily.CLC
        return SchedulerFamily.SCHEDULER_WARP
    if grouped is None:
        return SchedulerFamily.STATIC
    if grouped.scheduler_mode is Tcgen05GroupedSchedulerMode.RUNTIME_DIRECT:
        return SchedulerFamily.GROUPED_RUNTIME_DIRECT
    shapes = grouped.static_problem_shapes
    if shapes is None or len(shapes) > TCGEN05_GROUPED_STATIC_SPECIALIZATION_MAX_GROUPS:
        return SchedulerFamily.GROUPED_STATIC
    return SchedulerFamily.GROUPED_STATIC_SPECIALIZED
