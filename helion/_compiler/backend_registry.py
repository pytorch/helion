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

_BUILTIN_BACKENDS: list[type[Backend]] = [
    TritonBackend,
    PallasBackend,
    CuteBackend,
    TileIRBackend,
    MetalBackend,
]

_REGISTRY: dict[str, type[Backend]] = {}


def register_compiler_backend(backend_class: type[Backend]) -> None:
    """Register a compiler backend.

    The backend's ``name`` property is used as the registry key.
    Built-in backends are registered at module load time below.

    Args:
        backend_class: A :class:`Backend` subclass.
    """
    _REGISTRY[backend_class().name] = backend_class


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


def all_reserved_launch_param_names() -> frozenset[str]:
    """Union of reserved launch param names across all registered backends.

    Reserving all names ensures kernel portability. A variable name
    that collides with any backend's launch params is avoided regardless
    of which backend is currently active.
    """
    result: set[str] = set()
    for backend_cls in _REGISTRY.values():
        result.update(backend_cls.reserved_launch_param_names())
    return frozenset(result)


# register built-in backends
for _cls in _BUILTIN_BACKENDS:
    register_compiler_backend(_cls)
