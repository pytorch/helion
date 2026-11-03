"""The legacy MmaUniversalOp and warp MMA paths of the MMA pipeline emitter.

``_emit_mma_pipeline`` lowers a matmul with one of three MMA implementations
(``plan.mma_impl``): ``tcgen05`` (the Blackwell pipeline the rest of
``cute_mma`` emits), ``warp`` (``cute.nvgpu.warp.MmaF16BF16Op`` on complete
physical warps) and ``universal`` (``MmaUniversalOp`` fp32 FMA per thread).
The statement templates of the two legacy paths live here, verbatim; each
function returns the statements the emitter appends to the device-loop
prefix, the K-loop body or the suffix at the same point as before.

Both legacy paths stay live: ``warp`` is ``_choose_mma_impl``'s auto-mode
selection for ``bn == 8`` tiles and the only MMA of the direct grouped-N
lowering (``codegen_cute_mma_direct_mm``); ``universal`` is the exact fp32
fallback. The shared scalar SMEM staging block and the role / coordinate setup
the warp path shares with ``tcgen05`` remain in ``cute_mma``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...ast_extension import statement_from_string

if TYPE_CHECKING:
    import ast

    from ..cute_mma import Tcgen05PipelinePlan


def register_acc_frag_stmt(plan: Tcgen05PipelinePlan) -> ast.stmt:
    """The register accumulator of a non-tcgen05 MMA (prefix)."""
    return statement_from_string(
        f"{plan.acc_frag} = cute.make_rmem_tensor("
        f"{plan.tiled_mma}.partition_shape_C(({plan.bm}, {plan.bn})), {plan.acc_dtype_str})"
    )


def smem_ab_alloc_stmts(
    plan: Tcgen05PipelinePlan,
    *,
    smem_a_ptr: str,
    smem_a: str,
    smem_b_ptr: str,
    smem_b: str,
) -> list[ast.stmt]:
    """Plain row-major A / B shared-memory tiles of a non-tcgen05 MMA (prefix)."""
    return [
        statement_from_string(
            f"{smem_a_ptr} = cute.arch.alloc_smem({plan.input_dtype_str}, {plan.bm * plan.bk})"
        ),
        statement_from_string(
            f"{smem_a} = cute.make_tensor("
            f"{smem_a_ptr}, cute.make_layout(({plan.bm}, {plan.bk}), stride=({plan.bk}, 1)))"
        ),
        statement_from_string(
            f"{smem_b_ptr} = cute.arch.alloc_smem({plan.input_dtype_str}, {plan.bn * plan.bk})"
        ),
        statement_from_string(
            f"{smem_b} = cute.make_tensor("
            f"{smem_b_ptr}, cute.make_layout(({plan.bn}, {plan.bk}), stride=({plan.bk}, 1)))"
        ),
    ]


def universal_acc_init_stmt(plan: Tcgen05PipelinePlan) -> ast.stmt:
    """Zero, or seed from ``acc_expr``, the universal accumulator on the first K step."""
    if plan.acc_expr is None:
        return statement_from_string(
            f"if {plan.k_offset_var} == {plan.k_loop_begin_expr}:\n"
            f"    for _mma_i in range(cute.size({plan.acc_frag})):\n"
            f"        {plan.acc_frag}[_mma_i] = {plan.acc_dtype_str}(0.0)"
        )
    return statement_from_string(
        f"if {plan.k_offset_var} == {plan.k_loop_begin_expr}:\n"
        f"    for _mma_i in range(cute.size({plan.acc_frag})):\n"
        f"        {plan.acc_frag}[_mma_i] = {plan.acc_dtype_str}({{acc}})",
        acc=plan.acc_expr,
    )


def warp_acc_init_stmt(plan: Tcgen05PipelinePlan, *, mma_active: str) -> ast.stmt:
    """Zero the warp MMA accumulator on the first K step (participating warps)."""
    return statement_from_string(
        f"if {mma_active} and {plan.k_offset_var} == {plan.k_loop_begin_expr}:\n"
        f"    for _mma_i in range(cute.size({plan.acc_frag})):\n"
        f"        {plan.acc_frag}[_mma_i] = {plan.acc_dtype_str}(0.0)"
    )


def universal_smem_load_stmts(
    plan: Tcgen05PipelinePlan, *, smem_a: str, smem_b: str
) -> list[ast.stmt]:
    """Masked global -> shared loads of the universal path's A / B K tile.

    Guards select the hardware thread that loads each row/column of the A/B
    SMEM cache. Use the *physical* thread coord (not the lane-aware local
    coord) so the same hardware threads load on every iteration of an outer
    ``for lane_<n> in range(epT):`` loop when ``elements_per_thread > 1``.
    ``n_local == 0`` only matches ``(thread_y, lane) == (0, 0)`` -- fine for
    the no-lane case but leaves sA stale on ``lane > 0`` iterations because
    the guard never fires. ``n_physical == 0`` matches ``thread_y == 0`` on
    every lane iteration so sA is re-populated for the current K tile. The
    store target is still ``sA[m_local, _k]`` so different lane iterations
    naturally write to different rows of sA / sB if the m-axis has its own
    lane var.

    Local invariant: because the K loop is nested INSIDE the lane loop in the
    current scheduler, skipping the A/B load on lane>0 iterations would reuse
    the previous K tile's sA (overwritten by the previous K iteration) --
    incorrect. Deferred hoist would require K/lane interchange. See
    cute_plan.md for the deferred-restructure paths.
    """
    return [
        statement_from_string(
            f"if {plan.n_physical} == cutlass.Int32(0):\n"
            f"    for _k in range({plan.bk}):\n"
            f"        _gk = {plan.k_offset_var} + cutlass.Int32(_k)\n"
            f"        {smem_a}[{plan.m_local}, cutlass.Int32(_k)] = ("
            f"{plan.lhs_gmem_access(plan.m_global, '_gk')} "
            f"if {plan.m_global} < cutlass.Int32({plan.m_size}) "
            f"and _gk < cutlass.Int32({plan.k_total_size}) "
            f"else {plan.input_dtype_str}(0.0))"
        ),
        statement_from_string(
            f"if {plan.m_physical} == cutlass.Int32(0):\n"
            f"    for _k in range({plan.bk}):\n"
            f"        _gk = {plan.k_offset_var} + cutlass.Int32(_k)\n"
            f"        {smem_b}[{plan.n_local}, cutlass.Int32(_k)] = ("
            f"{plan.rhs_gmem_access('_gk', plan.n_global)} "
            f"if {plan.n_global} < cutlass.Int32({plan.n_size}) "
            f"and _gk < cutlass.Int32({plan.k_total_size}) "
            f"else {plan.input_dtype_str}(0.0))"
        ),
        statement_from_string("cute.arch.sync_threads()"),
    ]


def universal_gemm_stmts(
    plan: Tcgen05PipelinePlan,
    *,
    smem_a: str,
    smem_b: str,
    rA: str,
    rB: str,
    tAsA: str,
    tBsB: str,
) -> list[ast.stmt]:
    """Shared -> register (f16 -> f32 cast) and the universal ``cute.gemm``.

    The closing ``sync_threads`` orders this iteration's SMEM reads before
    the next iteration's stores into sA / sB. The universal SMEM-load guards
    funnel the WRITES through a thin thread slice (``thread_y == 0`` for sA,
    ``thread_x == 0`` for sB) while the MmaUniversalOp register copy READS
    the staged tile with every CTA thread. Writer set != reader set, spanning
    multiple warps, so across the barrier-free K-loop back-edge a thread that
    finishes its gemm early overwrites sA / sB while a sibling in another warp
    is still reading the current K tile -- a cross-warp write-after-read
    hazard that yields nondeterministic wrong values (confirmed via
    compute-sanitizer racecheck). Warp MMA orders this boundary with a CTA
    barrier of its own; tcgen05 uses its pipeline/mbarrier protocol.
    """
    return [
        statement_from_string(f"{tAsA} = {plan.thr_mma}.partition_A({smem_a})"),
        statement_from_string(f"{tBsB} = {plan.thr_mma}.partition_B({smem_b})"),
        statement_from_string(
            f"{rA} = cute.make_fragment_like({tAsA}, {plan.acc_dtype_str})"
        ),
        statement_from_string(
            f"{rB} = cute.make_fragment_like({tBsB}, {plan.acc_dtype_str})"
        ),
        statement_from_string(
            f"for _mma_i in range(cute.size({rA})):\n"
            f"    {rA}[_mma_i] = {plan.acc_dtype_str}({tAsA}[_mma_i])"
        ),
        statement_from_string(
            f"for _mma_i in range(cute.size({rB})):\n"
            f"    {rB}[_mma_i] = {plan.acc_dtype_str}({tBsB}[_mma_i])"
        ),
        statement_from_string(
            f"cute.gemm({plan.tiled_mma}, {plan.acc_frag}, {rA}, {rB}, {plan.acc_frag})"
        ),
        statement_from_string("cute.arch.sync_threads()"),
    ]


def warp_gemm_stmts(
    plan: Tcgen05PipelinePlan,
    *,
    mma_active: str,
    smem_a: str,
    smem_b: str,
    rA: str,
    rB: str,
    tAsA: str,
    tBsB: str,
) -> list[ast.stmt]:
    """Shared -> register and the warp ``cute.gemm`` on the participating warps.

    Every participating warp must finish reading this tile before any warp
    overwrites shared operands for the next K iteration, hence the barrier.
    """
    return [
        statement_from_string(
            f"if {mma_active}:\n"
            f"    {tAsA} = {plan.thr_mma}.partition_A({smem_a})\n"
            f"    {tBsB} = {plan.thr_mma}.partition_B({smem_b})\n"
            f"    {rA} = cute.make_fragment_like({tAsA}, {plan.input_dtype_str})\n"
            f"    {rB} = cute.make_fragment_like({tBsB}, {plan.input_dtype_str})\n"
            f"    for _mma_i in range(cute.size({rA})):\n"
            f"        {rA}[_mma_i] = {tAsA}[_mma_i]\n"
            f"    for _mma_i in range(cute.size({rB})):\n"
            f"        {rB}[_mma_i] = {tBsB}[_mma_i]\n"
            f"    cute.gemm({plan.tiled_mma}, {plan.acc_frag}, {rA}, {rB}, {plan.acc_frag})"
        ),
        statement_from_string("cute.arch.sync_threads()"),
    ]


def smem_c_alloc_stmts(
    plan: Tcgen05PipelinePlan, *, smem_c_ptr: str, smem_c: str
) -> list[ast.stmt]:
    """The staged ``smem_c`` tile the legacy epilogues hand to the scalar program.

    Allocated in the device-loop prefix so all SMEM is allocated at the same
    scope level (the CuTe DSL assigns static SMEM offsets per scope); the
    tcgen05 epilogues stage through ``store_epilogue`` instead.
    """
    tile_numel = plan.bm * plan.bn
    return [
        statement_from_string(
            f"{smem_c_ptr} = cute.arch.alloc_smem({plan.acc_dtype_str}, {tile_numel}, alignment=128)"
        ),
        statement_from_string(
            f"{smem_c} = cute.make_tensor("
            f"{smem_c_ptr}, cute.make_layout(({plan.bm}, {plan.bn}), stride=({plan.bn}, 1)))"
        ),
    ]


def universal_suffix_stmts(
    plan: Tcgen05PipelinePlan, *, tCsC: str, smem_c: str
) -> list[ast.stmt]:
    """Write the universal accumulator fragment to ``smem_c`` after the K loop."""
    return [
        statement_from_string(f"{tCsC} = {plan.thr_mma}.partition_C({smem_c})"),
        statement_from_string(
            f"for _mma_i in range(cute.size({tCsC})):\n"
            f"    {tCsC}[_mma_i] = {plan.acc_frag}[_mma_i]"
        ),
    ]


def warp_suffix_stmts(
    plan: Tcgen05PipelinePlan, *, mma_active: str, tCsC: str, smem_c: str
) -> list[ast.stmt]:
    """Write the warp accumulator fragment to ``smem_c`` after the K loop."""
    return [
        statement_from_string(
            f"if {mma_active}:\n"
            f"    {tCsC} = {plan.thr_mma}.partition_C({smem_c})\n"
            f"    for _mma_i in range(cute.size({tCsC})):\n"
            f"        {tCsC}[_mma_i] = {plan.acc_frag}[_mma_i]"
        ),
        statement_from_string("cute.arch.sync_threads()"),
    ]


def result_read_stmt(
    plan: Tcgen05PipelinePlan, *, result_var: str, smem_c: str
) -> ast.stmt:
    """Each thread reads its own (m, n) element of the finished tile from ``smem_c``."""
    return statement_from_string(
        f"{result_var} = {smem_c}[{plan.m_local}, {plan.n_local}]"
    )
