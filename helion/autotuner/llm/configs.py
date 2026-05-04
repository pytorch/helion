"""Normalize and validate autotune configs proposed by the LLM."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from .parsing import parse_jsonish

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..config_fragment import ConfigSpecFragment
    from ..config_spec import ConfigSpec
    from ..logger import AutotuningLogger


def is_positive_power_of_two_int(val: object) -> bool:
    """Return whether `val` is a strictly positive power-of-two integer."""
    return type(val) is int and val > 0 and (val & (val - 1)) == 0


def _expected_json_array_length(field: object) -> int | None:
    """Return the required JSON array length for fixed-length sequence fields."""
    from ..block_id_sequence import BlockIdSequence
    from ..config_fragment import ListOf

    if isinstance(field, BlockIdSequence):
        return len(field)
    if isinstance(field, ListOf):
        return field.length
    return None


def _validate_json_array_length(key: str, val: object, *, expected_len: int) -> None:
    """Validate that a JSON value is a list with the expected fixed length."""
    if not isinstance(val, list):
        raise ValueError(
            f"{key} must be a JSON array of length {expected_len}, got {val!r}"
        )
    if len(val) != expected_len:
        raise ValueError(f"{key} must have length {expected_len}, got {len(val)}")


def validate_sparse_config_shape(
    raw: dict[str, object], *, config_spec: ConfigSpec
) -> None:
    """Reject common LLM shape/type mismatches instead of silently repairing them."""
    flat_fields = config_spec._flat_fields()
    for key, val in raw.items():
        field = flat_fields.get(key)
        if key == "num_warps" and not is_positive_power_of_two_int(val):
            raise ValueError(
                f"num_warps must be a positive power-of-two integer, got {val!r}"
            )

        if (expected_len := _expected_json_array_length(field)) is not None:
            _validate_json_array_length(key, val, expected_len=expected_len)
            continue

        if field is not None and isinstance(val, list):
            raise ValueError(f"{key} must be a scalar value, got {val!r}")


def parse_response_configs(
    response: str,
    *,
    config_spec: ConfigSpec,
    default_config_dict: dict[str, object],
    log: AutotuningLogger,
) -> list[Config]:
    """Parse, validate, normalize, and deduplicate configs from an LLM response."""
    import helion

    parsed = parse_jsonish(response)
    raw_configs: object | None = None
    if isinstance(parsed, dict):
        raw_configs = parsed.get("configs")
    elif isinstance(parsed, list):
        raw_configs = parsed

    if not isinstance(raw_configs, list):
        log.warning("Failed to parse LLM response as a config list")
        return []

    configs: list[Config] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_configs):
        if not isinstance(raw, dict):
            log.debug(f"Skipping non-dict config at index {index}")
            continue
        try:
            validate_sparse_config_shape(raw, config_spec=config_spec)
            merged = dict(default_config_dict)
            merged.update(raw)
            config_spec.normalize(merged, _fix_invalid=True)
            config = helion.Config(**cast("dict[str, Any]", merged))
        except Exception as e:
            log.debug(f"Skipping invalid config {index}: {e}")
            continue

        config_key = repr(config)
        if config_key in seen:
            continue
        seen.add(config_key)
        configs.append(config)

    log(f"Parsed {len(configs)} valid configs from LLM response")
    return configs


def describe_config_space(config_spec: ConfigSpec) -> str:
    """Build a human-readable description of the tunable config space."""
    from ..block_id_sequence import BlockIdSequence
    from ..config_fragment import ListOf

    lines: list[str] = []
    for key, field in config_spec._flat_fields().items():
        if isinstance(field, BlockIdSequence):
            items: list[str] = []
            for item in field:
                try:
                    frag = item._fragment(config_spec)
                except NotImplementedError:
                    continue
                items.append(describe_fragment(frag))
            if not items:
                continue
            if len(items) == 1:
                lines.append(f"  {key}: list of 1 x {items[0]}")
            else:
                lines.append(f"  {key}: [{', '.join(items)}]")
        elif isinstance(field, ListOf):
            lines.append(
                f"  {key}: list of {field.length} x {describe_fragment(field.inner)}"
            )
        else:
            lines.append(f"  {key}: {describe_fragment(field)}")
    return "\n".join(lines)


def describe_fragment(frag: ConfigSpecFragment) -> str:
    """Generate a human-readable description of a config fragment."""
    from ..config_fragment import BooleanFragment
    from ..config_fragment import EnumFragment
    from ..config_fragment import IntegerFragment
    from ..config_fragment import PermutationFragment
    from ..config_fragment import PowerOfTwoFragment

    if isinstance(frag, PowerOfTwoFragment):
        return f"power_of_2(min={frag.low}, max={frag.high}, default={frag.default()})"
    if isinstance(frag, IntegerFragment):
        return f"integer(min={frag.low}, max={frag.high}, default={frag.default()})"
    if isinstance(frag, EnumFragment):
        choices_str = ", ".join(repr(choice) for choice in frag.choices)
        return f"enum({choices_str})"
    if isinstance(frag, BooleanFragment):
        return "boolean(default=False)"
    if isinstance(frag, PermutationFragment):
        return f"permutation(length={frag.length})"
    return f"{type(frag).__name__}()"
