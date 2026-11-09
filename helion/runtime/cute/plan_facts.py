"""Typed reads of CuTe wrapper-plan fields shared by the host emitters and
the launch-time validators.

A wrapper plan is the ``dict[str, object]`` literal the compiler renders into
``_helion_cute_wrapper_plans``; these helpers assert or re-prove the field
types the launcher relies on.  Leaf module: imports ``helion.exc`` and the
scheduler-mode enum only.
"""

from __future__ import annotations

from contextlib import suppress
from typing import cast

from ... import exc
from ..._compiler.cute.device_state import Tcgen05GroupedSchedulerMode


def plan_int_value(plan: dict[str, object], key: str) -> int:
    value = plan[key]
    assert isinstance(value, int)
    return value


def plan_str_value(plan: dict[str, object], key: str) -> str:
    value = plan[key]
    assert isinstance(value, str)
    return value


def tcgen05_plan_orientation(plan: dict[str, object]) -> str:
    orientation = plan.get("orientation", "mn")
    if orientation not in ("mn", "nm"):
        raise exc.BackendUnsupported(
            "cute", f"unsupported tcgen05 plan orientation {orientation!r}"
        )
    return cast("str", orientation)


def tcgen05_grouped_dynamic_ab_tensormap_rank(plan: dict[str, object]) -> int:
    rank = plan.get("dynamic_ab_tensormap_rank", 3)
    if (
        not isinstance(rank, int)
        or rank not in (2, 3)
        or (
            rank == 2
            and not bool(plan.get("dynamic_ab_tensormaps"))
            and not bool(plan.get("fixed_ab_tensormaps"))
            and not bool(plan.get("fixed_tensormaps"))
        )
    ):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped A/B TensorMap rank must be 2 or 3, and rank 2 "
            "requires dynamic or fixed full-allocation A/B TensorMaps",
        )
    return rank


def tcgen05_grouped_scheduler_mode(
    plan: dict[str, object],
) -> Tcgen05GroupedSchedulerMode:
    value = plan.get("scheduler_mode", Tcgen05GroupedSchedulerMode.DEVICE_GROUP_SEARCH)
    mode: Tcgen05GroupedSchedulerMode | None = None
    if isinstance(value, Tcgen05GroupedSchedulerMode):
        mode = value
    elif isinstance(value, str):
        with suppress(ValueError):
            mode = Tcgen05GroupedSchedulerMode(value)
    if mode is None:
        choices = ", ".join(
            candidate.value for candidate in Tcgen05GroupedSchedulerMode
        )
        raise exc.BackendUnsupported(
            "cute",
            f"tcgen05 grouped scheduler mode must be one of: {choices}",
        )
    uses_runtime_table = mode in (
        Tcgen05GroupedSchedulerMode.RUNTIME_DIRECT,
        Tcgen05GroupedSchedulerMode.RUNTIME_CLC,
    )
    has_runtime_table = isinstance(plan.get("runtime_tile_records_arg"), str)
    if uses_runtime_table != has_runtime_table:
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05 grouped runtime scheduler state requires matching tile-table metadata",
        )
    if uses_runtime_table and tcgen05_plan_orientation(plan) != "nm":
        raise exc.BackendUnsupported(
            "cute", "tcgen05 grouped runtime scheduling requires N,M orientation"
        )
    if mode is Tcgen05GroupedSchedulerMode.RUNTIME_CLC and not bool(
        plan.get("fixed_tensormaps")
    ):
        raise exc.BackendUnsupported(
            "cute", "grouped runtime CLC requires fixed full-allocation TensorMaps"
        )
    return mode
