"""Ordinary-by-default choice for a structurally proved state-update root."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from ...exc import InvalidConfig

if TYPE_CHECKING:
    from ...autotuner.config_spec import ConfigSpec

CONFIG_KEY = "cute_state_update_schedule"
ORDINARY = "ordinary"
HORIZONTAL = "horizontal_tma"
NORMALIZATION_VERSION = 1


def normalize(
    spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    value = config.get(CONFIG_KEY, ORDINARY)
    if value == ORDINARY and type(value) is str:
        # The old ordinary dictionaries and their encoded seed prefix remain
        # identical. The added flat choice has ordinary as its final coordinate.
        config.pop(CONFIG_KEY, None)
        return
    if (
        not spec.cute_horizontal_state_available
        or type(value) is not str
        or value != HORIZONTAL
    ):
        if fix_invalid:
            config.pop(CONFIG_KEY, None)
            return
        raise InvalidConfig(
            "horizontal state schedule requires an admitted structural candidate"
        )
    if config.get("pid_type", "flat") != "flat":
        if fix_invalid:
            config["pid_type"] = "flat"
        else:
            raise InvalidConfig("horizontal state schedule requires pid_type='flat'")


def normalize_row_tile(
    spec: ConfigSpec, config: dict[str, object], *, fix_invalid: bool
) -> None:
    """Validate the schedule dependency after ordinary block-list normalization."""
    if config.get(CONFIG_KEY) != HORIZONTAL:
        return
    facts = spec.cute_horizontal_state_row_tiles
    if facts is None:
        return
    row_id, choices = facts
    if row_id not in spec.block_sizes.valid_block_ids():
        # A fixed/non-configurable row remains the original planner's concern.
        return
    index = spec.block_sizes.block_id_to_index(row_id)
    values = cast("list[int]", config["block_sizes"])
    row = values[index]
    if type(row) is int and row in choices:
        return
    if not fix_invalid:
        raise InvalidConfig(
            f"horizontal state row tile must be one of {choices!r}, got {row!r}"
        )
    values = list(values)
    values[index] = min(choices, key=lambda value: (abs(value - row), value))
    config["block_sizes"] = values


def canonicalize_inactive(spec: ConfigSpec, config: dict[str, object]) -> None:
    """Collapse ordinary SIMT knobs unused by the selected whole-root body.

    Run after ordinary validation, before any codegen receipt is constructed.
    The horizontal planner still validates geometry/ownership and never falls
    back to the ordinary body for a selected but unsupported request. Its scalar
    fallback is part of the same owned body and uses the same fixed schedule.
    """
    if not spec.cute_horizontal_state_available or config.get(CONFIG_KEY) != HORIZONTAL:
        return
    # Physical warps come from HorizontalGeometry, not ordinary lane ownership.
    config.pop("num_threads", None)
    for name, mapping in (
        ("cute_vector_widths", spec.cute_vector_widths),
        ("cute_lane_layouts", spec.cute_lane_layouts),
        ("cute_reduction_reloads", spec.cute_reduction_reloads),
    ):
        config[name] = mapping._normalize(name, (), flatten=True)
    # StateExpression emits original unqualified scalar reads; coefficient and
    # state transport use their own explicit load instructions.
    from ...autotuner.config_spec import get_valid_eviction_policies

    policies = config.get("load_eviction_policies")
    values = [policies] if isinstance(policies, str) else policies
    if (
        isinstance(values, list)
        and (
            isinstance(policies, str)
            or len(values) == spec.load_eviction_policies.length
        )
        and all(
            type(value) is str and value in get_valid_eviction_policies("cute")
            for value in values
        )
    ):
        config["load_eviction_policies"] = spec.load_eviction_policies.default()
