"""Typed argument paths for runtime metadata guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

from ..language.constexpr import ConstExpr
from .variable_origin import ArgumentOrigin
from .variable_origin import AttributeOrigin
from .variable_origin import GetItemOrigin

if TYPE_CHECKING:
    from typing import Any

    from .variable_origin import Origin


def unwrap_constexpr(value: object) -> object:
    while isinstance(value, ConstExpr):
        value = value.value
    return value


@dataclass(frozen=True)
class InputPathEntry:
    key: int | str
    attribute: bool


def argument_path(
    origin: Origin | None,
) -> tuple[ArgumentOrigin, tuple[InputPathEntry, ...]] | None:
    entries = []
    while isinstance(origin, (GetItemOrigin, AttributeOrigin)):
        entries.append(InputPathEntry(origin.key, isinstance(origin, AttributeOrigin)))
        origin = origin.value
    if isinstance(origin, ArgumentOrigin):
        return origin, tuple(reversed(entries))
    return None


def resolve_input_path(value: object, entries: tuple[InputPathEntry, ...]) -> object:
    value = unwrap_constexpr(value)
    for entry in entries:
        if entry.attribute:
            assert isinstance(entry.key, str)
            value = getattr(value, entry.key)
        else:
            value = cast("Any", value)[entry.key]
        value = unwrap_constexpr(value)
    return value
