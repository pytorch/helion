"""Backend-specific dispatch decorators.

This module provides a decorator-based system for registering backend-specific
implementations of functions, similar to the @_decorators.codegen pattern.

Usage:
    @backend_dispatch
    def typed_program_id(dim: int = 0) -> str:
        '''Generate program_id expression.'''
        ...

    @backend_impl(typed_program_id, "triton")
    def _(dim: int) -> str:
        return f"tl.program_id({dim})"

    @backend_impl(typed_program_id, "pallas")
    def _(dim: int) -> str:
        return f"pl.program_id({dim})"
"""

from __future__ import annotations

import functools
from typing import Callable
from typing import TypeVar

_F = TypeVar("_F", bound=Callable)


def backend_dispatch(fn: _F) -> _F:
    """Mark a function as having backend-specific implementations.

    The decorated function becomes a dispatcher that automatically
    calls the correct backend implementation based on the current
    CompileEnvironment.backend setting.

    Supports "common" fallback (like codegen handlers).
    """
    fn._backend_impls = {}  # type: ignore[attr-defined]

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        from ..exc import BackendImplementationMissing
        from .compile_environment import CompileEnvironment

        backend = CompileEnvironment.current().backend
        impl = fn._backend_impls.get(backend) or fn._backend_impls.get("common")  # type: ignore[attr-defined]
        if impl is None:
            raise BackendImplementationMissing(backend, fn.__name__)
        return impl(*args, **kwargs)

    wrapper._backend_impls = fn._backend_impls  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def backend_impl(fn: Callable, backend: str) -> Callable[[_F], _F]:
    """Register a backend-specific implementation.

    Args:
        fn: The dispatcher function to register an implementation for.
        backend: The backend name ("triton", "pallas", or "common").

    Returns:
        A decorator that registers the implementation.
    """

    def decorator(impl: _F) -> _F:
        fn._backend_impls[backend] = impl  # type: ignore[attr-defined]
        return impl

    return decorator
