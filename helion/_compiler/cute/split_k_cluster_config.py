from __future__ import annotations

from ...exc import InvalidConfig

SCHEDULE_KEY = "cute_split_k_schedule"
LEGACY = "legacy"
CLUSTER8_K4 = "cluster8_k4"
SCHEDULES = (LEGACY, CLUSTER8_K4)
FINALIZER_KEY = "cute_split_k_finalizer_warps"
FINALIZER_WARPS = (1, 4)


def normalize_cluster_schedule(
    config: dict[str, object], *, available: bool, fix_invalid: bool
) -> None:
    value = config.get(SCHEDULE_KEY, LEGACY)
    if value == LEGACY:
        config.pop(SCHEDULE_KEY, None)
        return
    if (
        value == CLUSTER8_K4
        and available
        and config.get("cute_split_k_workspace", False) is False
        and config.get("cute_collective_mma", False) is False
    ):
        return
    if fix_invalid:
        config.pop(SCHEDULE_KEY, None)
        return
    raise InvalidConfig(
        "cluster split-K requires its proved domain and excludes workspace "
        "and ordinary collective schedules"
    )


def normalize_cluster_finalizer(
    config: dict[str, object], *, available: bool, fix_invalid: bool
) -> None:
    value = config.get(FINALIZER_KEY, 1)
    if type(value) is int and value == 1:
        # Preserve the old config, generated source, and cache identity.
        config.pop(FINALIZER_KEY, None)
        return
    if (
        type(value) is int
        and value == 4
        and available
        and config.get(SCHEDULE_KEY) == CLUSTER8_K4
    ):
        return
    if fix_invalid:
        config.pop(FINALIZER_KEY, None)
        return
    raise InvalidConfig(
        "split-K finalizer warps must be 1 or 4; four warps require "
        "the proved cluster8_k4 schedule"
    )
