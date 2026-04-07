"""Centralized registry for Helion codegen backends.

All backend lookup and instantiation should go through this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .backend import CuteBackend
from .backend import MetalBackend
from .backend import PallasBackend
from .backend import TileIRBackend
from .backend import TritonBackend

if TYPE_CHECKING:
    from .backend import Backend

_BUILTIN_BACKENDS: list[tuple[str, type[Backend]]] = [
    ("triton", TritonBackend),
    ("pallas", PallasBackend),
    ("cute", CuteBackend),
    ("tileir", TileIRBackend),
    ("metal", MetalBackend),
]

_REGISTRY: dict[str, type[Backend]] = {}


def register_backend(name: str, backend_class: type[Backend]) -> None:
    """Register a codegen backend by name.

    Built-in backends are registered at module load time below.

    Args:
        name: Short name used to select this backend (e.g. ``"triton"``).
        backend_class: A :class:`Backend` subclass.
    """
    _REGISTRY[name] = backend_class


def get_backend_class(name: str) -> type[Backend]:
    """Look up a registered backend class by name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backend: {name!r}. Available backends: {list_backends()}"
        )
    return _REGISTRY[name]


def list_backends() -> list[str]:
    """Return the names of all registered backends."""
    return list(_REGISTRY.keys())


# register built-in backends
for _name, _cls in _BUILTIN_BACKENDS:
    register_backend(_name, _cls)
