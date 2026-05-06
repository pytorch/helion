from __future__ import annotations

from functools import lru_cache
import inspect
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

if TYPE_CHECKING:
    from collections.abc import Callable


@lru_cache(maxsize=1)
def cutedsl_has_opresultlist_fix() -> bool:
    """Detect whether ``cutlass.cutlass_dsl.cutlass.if_generate`` recognises
    ``ir.OpResultList`` containers for multi-result ``scf.if`` ops.

    The PyPI ``nvidia-cutlass-dsl==4.5.0.dev0`` wheel (uploaded 2026-04-08)
    ships an ``if_generate`` that wraps a multi-result ``OpResultList`` in a
    one-element Python list, which then breaks the result-type ``zip`` and
    raises ``DSLRuntimeError: <OpResultList> to integer conversion is not
    supported`` from the next ``Int32(...)`` call. Newer (post-PyPI) builds
    add an explicit ``isinstance(mlir_results, ir.OpResultList)`` guard that
    fixes the bug — and is the marker we look for here.

    Returns ``True`` when the fix is present, ``False`` for the buggy build.
    """
    try:
        from cutlass.cutlass_dsl.cutlass import if_generate
    except Exception:
        return True
    try:
        src = inspect.getsource(if_generate)
    except (OSError, TypeError):
        return True
    return "ir.OpResultList" in src


@lru_cache(maxsize=1)
def cutedsl_tmem_allocator_has_dealloc_init_kwarg() -> bool:
    """Detect whether ``cutlass.utils.TmemAllocator.__init__`` accepts the
    ``dealloc_mbarrier_initialized`` keyword argument.

    The PyPI ``nvidia-cutlass-dsl==4.5.0.dev0`` wheel unconditionally calls
    ``self._init_dealloc_mbarrier`` in ``__init__`` whenever ``is_two_cta``
    is True; newer builds add a ``dealloc_mbarrier_initialized`` flag that
    short-circuits the init so a second allocator (e.g. one constructed in
    the epilogue after the matmul prologue already initialized the
    barrier) doesn't double-init the mbarrier. Helion emits the kwarg in
    the epilogue path; we omit it on builds without the parameter.

    Returns ``True`` when the kwarg is supported, ``False`` otherwise.

    ``inspect.signature`` doesn't see kwargs that ``@dsl_user_op`` strips
    from the wrapper, so we resolve the underlying ``__wrapped__`` first
    and fall back to source scanning.
    """
    try:
        from cutlass.utils import TmemAllocator
    except Exception:
        return True
    init = TmemAllocator.__init__
    inner = cast("Callable[..., Any]", getattr(init, "__wrapped__", init))
    try:
        sig = inspect.signature(inner)
    except (TypeError, ValueError):
        sig = None
    if sig is not None and "dealloc_mbarrier_initialized" in sig.parameters:
        return True
    try:
        src = inspect.getsource(inner)
    except (OSError, TypeError):
        return True
    return "dealloc_mbarrier_initialized" in src


@lru_cache(maxsize=1)
def cutedsl_tma_umma_tail_has_peer_cta_semantics() -> bool:
    """Detect whether ``PipelineTmaUmma.producer_tail`` lets peer CTAs wait.

    ``cutedsl_has_opresultlist_fix`` only answers whether nested
    ``PipelineState.advance()`` can be called safely. It does not prove that
    the installed ``PipelineTmaUmma.producer_tail`` has the current upstream
    semantics where every CTA advances and calls ``producer_acquire``. Older
    source shapes that gate the whole tail to the leader CTA skip peer CTA
    empty-barrier participation, so Helion must inline the tail for them even
    if the ``advance`` bug is fixed.

    Returns ``True`` when the source looks like the current peer-CTA-safe
    implementation, ``False`` when it is leader-gated or cannot be inspected.
    """
    try:
        from cutlass.pipeline import PipelineTmaUmma
    except ImportError:
        return False
    try:
        src = inspect.getsource(PipelineTmaUmma.producer_tail)
    except (OSError, TypeError):
        return False
    # Deliberately fail closed: if upstream reshapes this function in a way
    # this text probe no longer recognizes, Helion inlines its known-good tail.
    leader_markers = (
        "block_idx_in_cluster",
        "cta_rank",
        "cluster_rank",
        "rank_in_cluster",
        "is_leader_cta",
    )
    return "producer_acquire(" in src and not any(
        marker in src for marker in leader_markers
    )


def emit_dealloc_mbarrier_initialized_kwarg() -> str:
    """Emit ``dealloc_mbarrier_initialized=True`` (with a leading comma) on
    cutedsl builds that accept it, else an empty string. Designed to be
    spliced into a ``TmemAllocator(...)`` argument list as the *last* arg
    so the comma-prefix is always safe.
    """
    if cutedsl_tmem_allocator_has_dealloc_init_kwarg():
        return ", dealloc_mbarrier_initialized=True"
    return ""


def _advance_lines(state_expr: str, indent: str) -> list[str]:
    """Inline body of a single ``state.advance()`` (buggy-cutedsl workaround)."""
    phase_update = (
        f"{indent}{state_expr}._phase = ({state_expr}._phase ^ cutlass.Int32(1)) "
        f"if {state_expr}._index == {state_expr}.stages else {state_expr}._phase"
    )
    index_update = (
        f"{indent}{state_expr}._index = cutlass.Int32(0) "
        f"if {state_expr}._index == {state_expr}.stages else {state_expr}._index"
    )
    return [
        f"{indent}{state_expr}._count = {state_expr}._count + cutlass.Int32(1)",
        f"{indent}{state_expr}._index = {state_expr}._index + cutlass.Int32(1)",
        phase_update,
        index_update,
    ]


def emit_pipeline_advance(state_expr: str, *, indent: str = "") -> str:
    """Emit code equivalent to ``<state_expr>.advance()``.

    On cutedsl builds with the OpResultList fix this returns the natural
    ``state.advance()`` call. On the buggy PyPI 4.5.0.dev0 build it inlines
    the same semantics using two single-result Python ternaries (each of
    which lowers to a single-result ``scf.if`` and avoids the broken
    multi-result path inside ``pipeline.PipelineState.advance``). The
    workaround body is wrapped in ``if True:`` so the returned string is
    always exactly one Python top-level statement — both code paths can
    therefore be passed straight to ``statement_from_string`` or spliced
    into an existing block body without breaking ``ast.parse``.

    The emitted lines all carry ``indent`` so the caller can splice the
    string into an existing block without further reflowing.
    """
    if cutedsl_has_opresultlist_fix():
        return f"{indent}{state_expr}.advance()"

    inner = indent + "    "
    body = "\n".join(_advance_lines(state_expr, inner))
    return f"{indent}if True:\n{body}"


def emit_producer_tail_tma_umma(
    pipeline_expr: str,
    state_expr: str,
    *,
    num_stages: int,
    indent: str = "",
) -> str:
    """Emit code equivalent to ``<pipeline>.producer_tail(<state>)`` for a
    ``PipelineTmaUmma`` (sm100 TMA→UMMA) pipeline.

    The current cutedsl implementation calls ``state.advance()``
    ``num_stages-1`` times and then ``producer_acquire``. On the buggy PyPI
    build that inner ``advance`` raises the OpResultList ``DSLRuntimeError``;
    some intermediate builds also had the advance fix but still gated the
    whole tail to the leader CTA. Helion's user-level ``state.advance()``
    workaround can't reach nested calls, so we inline the whole tail unless
    both the advance bug and the TMA tail ownership shape are known-good.

    Unlike ``PipelineUmmaAsync``, ``PipelineTmaUmma.producer_tail`` is
    intentionally not leader-CTA gated. Peer CTAs still need to advance
    their local producer state and wait for the empty barrier; the inner
    ``producer_acquire`` implementation gates only the full-barrier arrive
    to the leader CTA.

    ``num_stages`` is a compile-time constant from helion's tcgen05 plan
    (``ab_stage_count`` for the TMA pipeline).
    """
    if (
        cutedsl_has_opresultlist_fix()
        and cutedsl_tma_umma_tail_has_peer_cta_semantics()
    ):
        return f"{indent}{pipeline_expr}.producer_tail({state_expr})"

    body_lines: list[str] = []
    for _ in range(num_stages - 1):
        body_lines.extend(_advance_lines(state_expr, indent))
    body_lines.append(f"{indent}{pipeline_expr}.producer_acquire({state_expr})")
    return "\n".join(body_lines)


def emit_producer_tail_umma_async(
    pipeline_expr: str,
    state_expr: str,
    *,
    num_stages: int,
    indent: str = "",
) -> str:
    """Emit code equivalent to ``<pipeline>.producer_tail(<state>)`` for a
    ``PipelineUmmaAsync`` (sm100 UMMA→async-consumer) pipeline.

    The cutedsl implementation gates the drain to the leader CTA, advances
    ``num_stages - 1`` times, then calls ``producer_acquire``. We inline
    the same leader guard plus advances so each ``advance`` becomes the
    same single-result-ternary workaround used by
    :func:`emit_pipeline_advance`. ``num_stages`` is the compile-time
    ``acc_stage_count`` from helion's tcgen05 plan.
    """
    if cutedsl_has_opresultlist_fix():
        return f"{indent}{pipeline_expr}.producer_tail({state_expr})"

    inner = indent + "    "
    leader_inner = inner + "    "
    body_lines = [
        f"{indent}if True:",
        f"{inner}_pt_bidx = cute.arch.block_idx_in_cluster()",
        f"{inner}_pt_cta_rank = cute.arch.make_warp_uniform(_pt_bidx)",
        f"{inner}if _pt_cta_rank % cutlass.Int32(2) == cutlass.Int32(0):",
    ]
    for _ in range(num_stages - 1):
        body_lines.extend(
            (
                f"{leader_inner}if True:",
                *_advance_lines(state_expr, leader_inner + "    "),
            )
        )
    body_lines.append(f"{leader_inner}{pipeline_expr}.producer_acquire({state_expr})")
    return "\n".join(body_lines)
