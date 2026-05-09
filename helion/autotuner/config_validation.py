# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Validation helpers for sparse autotune config dictionaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config_spec import ConfigSpec


def is_positive_power_of_two_int(val: object) -> bool:
    """Return whether `val` is a strictly positive power-of-two integer."""
    return type(val) is int and val > 0 and (val & (val - 1)) == 0


def _expected_json_array_length(field: object) -> int | None:
    """Return the required JSON array length for fixed-length sequence fields."""
    from .block_id_sequence import BlockIdSequence
    from .config_fragment import ListOf

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
    """Reject sparse config shape/type mismatches instead of silently repairing them."""
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
