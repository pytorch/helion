"""Translate load policies between a phase and its owning launch bundle."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence


def project_load_policies(
    policies: Sequence[object], slots: Mapping[int, int]
) -> list[object]:
    local: list[object] = [""] * (max(slots, default=-1) + 1)
    for phase_slot, global_slot in slots.items():
        local[phase_slot] = policies[global_slot]
    return local


def scatter_load_policies(
    policies: Sequence[object], slots: Mapping[int, int], base: Sequence[object]
) -> list[object]:
    result = list(base)
    for phase_slot, global_slot in slots.items():
        result[global_slot] = policies[phase_slot]
    return result
