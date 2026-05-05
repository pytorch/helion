from __future__ import annotations

from functools import lru_cache
import inspect


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
    inner = getattr(init, "__wrapped__", init)
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

    The cutedsl implementation calls ``state.advance()`` ``num_stages-1``
    times and then ``producer_acquire``. On the buggy PyPI build that
    inner ``advance`` raises the OpResultList ``DSLRuntimeError`` —
    helion's user-level ``state.advance()`` workaround can't reach those
    nested calls, so we inline the whole tail here using the same
    advance workaround for each stage hop. The leader-CTA fast path
    (``cta_rank_in_cluster % 2 == 0``) is preserved so 2-CTA matmuls
    still drain only the leader.

    ``num_stages`` is a compile-time constant from helion's tcgen05 plan
    (``ab_stage_count`` for the TMA pipeline).
    """
    if cutedsl_has_opresultlist_fix():
        return f"{indent}{pipeline_expr}.producer_tail({state_expr})"

    inner = indent + "    "
    inner2 = inner + "    "
    advance_blocks: list[str] = []
    for _ in range(num_stages - 1):
        advance_blocks.append(f"{inner}if True:")
        advance_blocks.extend(_advance_lines(state_expr, inner2))
    body_lines = [
        f"{indent}_pt_bidx = cute.arch.block_idx_in_cluster()",
        f"{indent}_pt_cta_rank = cute.arch.make_warp_uniform(_pt_bidx)",
        f"{indent}if _pt_cta_rank % cutlass.Int32(2) == cutlass.Int32(0):",
        *advance_blocks,
        f"{inner}{pipeline_expr}.producer_acquire({state_expr})",
    ]
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
