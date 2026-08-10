from __future__ import annotations

from functools import lru_cache
import importlib.metadata

from packaging.version import InvalidVersion
from packaging.version import Version

# The cute backend hard-requires the CuTe DSL 4.5.1 API generation, the
# apache-tvm-ffi package, and CUDA >= 13. ``check_cute_backend_requirements``
# (wired through ``CuteBackend.validate_environment``) enforces this up front so
# the rest of the backend can assume the modern APIs unconditionally rather than
# carry per-build compatibility shims and inline workarounds.
CUTE_MIN_CUDA_VERSION = "13"
CUTE_MIN_VERSION = Version("4.5.1")
CUTE_MATH_MIN_MAX_VERSION = Version("4.6.0")


@lru_cache(maxsize=1)
def cute_math_min_max_available() -> bool:
    """Return whether ``cute.math.min/max`` are available in this CuTe DSL."""
    return (
        Version(importlib.metadata.version("nvidia-cutlass-dsl"))
        >= CUTE_MATH_MIN_MAX_VERSION
    )


@lru_cache(maxsize=1)
def _cute_backend_requirement_error() -> str | None:
    """Return why the cute backend cannot run here, or ``None`` if it can.

    Cached because the answer is fixed for the lifetime of the process.
    """
    try:
        installed_version = Version(importlib.metadata.version("nvidia-cutlass-dsl"))
    except importlib.metadata.PackageNotFoundError:
        return (
            "the CuTe DSL is not installed "
            f"(need nvidia-cutlass-dsl >= {CUTE_MIN_VERSION})"
        )
    except InvalidVersion as e:
        return (
            "the installed CuTe DSL version is invalid "
            f"(need >= {CUTE_MIN_VERSION}): {e}"
        )
    if installed_version < CUTE_MIN_VERSION:
        return (
            "the installed CuTe DSL is too old "
            f"(need >= {CUTE_MIN_VERSION}, found {installed_version})"
        )

    try:
        import cutlass.cute as cute  # noqa: F401
    except ImportError as e:
        return (
            "the CuTe DSL is not importable "
            f"(need nvidia-cutlass-dsl >= {CUTE_MIN_VERSION}): {e}"
        )

    try:
        import tvm_ffi  # noqa: F401
    except ImportError:
        return (
            "the apache-tvm-ffi package is required by the cute backend "
            "(install it via `pip install apache-tvm-ffi`)"
        )

    from ..._compat import requires_cuda_version

    if not requires_cuda_version(CUTE_MIN_CUDA_VERSION):
        import torch

        return (
            f"the cute backend requires CUDA >= {CUTE_MIN_CUDA_VERSION}, "
            f"but torch.version.cuda is {torch.version.cuda!r}"
        )
    return None


def check_cute_backend_requirements() -> None:
    """Raise :class:`helion.exc.CuteBackendUnavailable` if the cute backend
    cannot run in the current environment (missing/old CuTe DSL, missing
    tvm-ffi, or CUDA < 13)."""
    reason = _cute_backend_requirement_error()
    if reason is not None:
        from ... import exc

        raise exc.CuteBackendUnavailable(reason)


def emit_pipeline_advance(state_expr: str, *, indent: str = "") -> str:
    """Emit code equivalent to ``<state_expr>.advance()``.

    The leading ``indent`` is applied so the caller can splice the returned
    string into an existing block without further reflowing.
    """
    return f"{indent}{state_expr}.advance()"


def emit_producer_tail_tma_umma(
    pipeline_expr: str,
    state_expr: str,
    *,
    indent: str = "",
    skip_advances: bool = False,
) -> str:
    """Emit ``<pipeline>.producer_tail(<state>)`` for a ``PipelineTmaUmma``
    (sm100 TMA->UMMA) pipeline.

    ``skip_advances`` is only for guarded invalid-output diagnostics that
    isolate AB producer-state rollover: it preserves the tail acquire but drops
    every state advance (including the ones ``producer_tail`` performs
    internally), so it emits a bare ``producer_acquire`` instead.
    """
    if skip_advances:
        return f"{indent}{pipeline_expr}.producer_acquire({state_expr})"
    return f"{indent}{pipeline_expr}.producer_tail({state_expr})"
