from __future__ import annotations

from functools import lru_cache
import inspect
import os


def _ensure_cute_dsl_libs() -> None:
    """Point ``CUTE_DSL_LIBS`` at the bundled CuTe DSL runtime library.

    The CuTe DSL needs ``libcute_dsl_runtime.so`` and auto-discovers it relative
    to a standard wheel install. In packaged/linked layouts (e.g. a buck
    link-tree) that discovery fails, so locate the ``.so`` shipped inside the
    ``nvidia_cutlass_dsl`` package and export it ourselves. No-op when the user
    already set ``CUTE_DSL_LIBS`` or the library cannot be found. Must run before
    ``import cutlass.cute`` (the discovery happens at import time).
    """
    if os.environ.get("CUTE_DSL_LIBS"):
        return
    import importlib.util

    spec = importlib.util.find_spec("nvidia_cutlass_dsl")
    for base in spec.submodule_search_locations or () if spec else ():
        candidate = os.path.join(base, "lib", "libcute_dsl_runtime.so")
        if os.path.exists(candidate):
            os.environ["CUTE_DSL_LIBS"] = candidate
            return

# The cute backend hard-requires the CuTe DSL 4.5.1 API generation, the
# apache-tvm-ffi package, and CUDA >= 13. ``check_cute_backend_requirements``
# (wired through ``CuteBackend.validate_environment``) enforces this up front so
# the rest of the backend can assume the modern APIs unconditionally rather than
# carry per-build compatibility shims and inline workarounds.
CUTE_MIN_CUDA_VERSION = "13"


@lru_cache(maxsize=1)
def _cute_backend_requirement_error() -> str | None:
    """Return why the cute backend cannot run here, or ``None`` if it can.

    Cached because the answer is fixed for the lifetime of the process. The
    CuTe DSL generation is detected by *feature*, not by ``cutlass.__version__``:
    the published wheels under-report the API generation (the 4.5.1 API has
    shipped in wheels that still self-report ``4.5.0``), so a version-string
    comparison would spuriously reject a working install.
    """
    _ensure_cute_dsl_libs()
    try:
        import cutlass.cute as cute  # noqa: F401
        from cutlass.cutlass_dsl.cutlass import if_generate
        from cutlass.pipeline import PipelineTmaUmma
        from cutlass.utils import TmemAllocator
    except ImportError as e:
        return f"the CuTe DSL is not importable (need nvidia-cutlass-dsl >= 4.5.1): {e}"

    # 4.5.1 names the TmemAllocator skip-init kwarg ``initialize_mbarrier`` (it
    # was ``dealloc_mbarrier_initialized`` in intermediate builds and absent from
    # 4.5.0.dev0). ``@dsl_user_op`` strips kwargs from the public wrapper, so
    # resolve ``__wrapped__`` first.
    init = TmemAllocator.__init__
    inner = getattr(init, "__wrapped__", init)
    try:
        params = inspect.signature(inner).parameters
    except (TypeError, ValueError):
        params = {}
    if "initialize_mbarrier" not in params:
        return (
            "the installed CuTe DSL is too old (need >= 4.5.1: "
            "TmemAllocator.initialize_mbarrier kwarg is missing)"
        )

    # 4.5.1 fixed the multi-result ``scf.if`` lowering used by nested
    # ``PipelineState.advance()`` (4.5.0.dev0 raised a DSLRuntimeError from the
    # ``OpResultList`` path) and gave ``PipelineTmaUmma.producer_tail`` peer-CTA
    # semantics (older source gated the whole tail to the leader CTA). Both are
    # detected from source.
    try:
        if "ir.OpResultList" not in inspect.getsource(if_generate):
            return (
                "the installed CuTe DSL is too old (need >= 4.5.1: "
                "if_generate is missing the OpResultList fix)"
            )
        tail_src = inspect.getsource(PipelineTmaUmma.producer_tail)
    except (OSError, TypeError) as e:
        return f"the installed CuTe DSL source cannot be inspected (need >= 4.5.1): {e}"
    leader_markers = (
        "block_idx_in_cluster",
        "cta_rank",
        "cluster_rank",
        "rank_in_cluster",
        "is_leader_cta",
    )
    if "producer_acquire(" not in tail_src or any(
        marker in tail_src for marker in leader_markers
    ):
        return (
            "the installed CuTe DSL is too old (need >= 4.5.1: "
            "PipelineTmaUmma.producer_tail lacks peer-CTA semantics)"
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
