"""Register-blocked SIMT FMA GEMM for exact fp32 matmuls (``cute_simt_mma``).

``dot_precision="ieee"`` (and ``"tf32x3"``) keep fp32 matmuls off the tensor
cores.  The generic lowering then computes every output element with its own
scalar K loop whose two operand loads come straight from global memory, so a
CTA re-reads each A row ``bn`` times and each B column ``bm`` times and sits
around one TFLOPS.  This lowering is the classic SIMT SGEMM structure cuBLAS
uses at the same precision (``cutlass_80_simt_sgemm_*``): per K step the CTA
stages one ``bm x bk`` A tile and one ``bk x bn`` B tile in shared memory, and
each thread keeps an ``rm x rn`` register block of the output that it updates
with rank-1 ``fma`` updates read from the staged tiles.

The scalar epilogue and the root lane loops stay in place.  The thread layout
is the launch's own ``num_threads`` choice: ``rm = bm / threads_m`` and
``rn = bn / threads_n`` are exactly the lane counts the scalar scheduler gives
those axes.  Like the collective matmul, the K-loop lowering only leaves a
``_helion_pending_simt_gemm`` marker where the scalar program binds the dot
result (``mark_simt_gemm``); the lane-loop distribution keeps a marked body
as the full nest, and ``lower_simt_gemm`` then splices the whole GEMM (shared
tiles, the staged K loop, the finished tile parked in shared memory) ahead of
the outermost lane loop and turns the marker into each lane's read of its own
element, the same ``smem_c`` hand-off the universal lowering uses.

The specialized-loop detection (``_detect_specialized_mma_loop``), the
operand-load suppression (``prepare_cute_simt_gemm_loads``) and this module
must agree: once the K loop is emitted without lanes, declining here would
leave the scalar body a single K element per step, so the emitter raises
``BackendUnsupported`` on anything it cannot lower instead of returning
``None``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ..ast_extension import expr_from_string

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from torch.fx import Node

    from ..device_function import DeviceFunction
    from ..generate_ast import GenerateAST
    from ..tile_strategy import DeviceLoopState

SIMT_MMA_KEY = "cute_simt_mma"
# Register-block row ownership: "blocked" owns rm consecutive rows (one or
# two 128-bit shared reads per K row), "strided" owns rows tid + i * threads
# (scalar reads, a warp reads consecutive words). Which wins depends on the
# thread layout, so the search decides.
SIMT_MMA_LAYOUT_KEY = "cute_simt_mma_layout"
SIMT_MMA_LAYOUTS: tuple[str, ...] = ("blocked", "strided")
MARKER = "_helion_pending_simt_gemm"
# The one-instruction PTX helpers shared with the warp-MMA owner.
RUNTIME = "_helion_mma_dag"
# Each thread holds ``rm * rn`` fp32 accumulators plus an ``rm + rn`` operand
# fragment; 64 keeps the whole block comfortably inside the register file.
MAX_REGISTER_BLOCK = 64


def simt_mma_requested(config: object) -> bool:
    getter = getattr(config, "get", None)
    return callable(getter) and getter(SIMT_MMA_KEY, False) is True


def simt_mma_layout(config: object) -> str:
    getter = getattr(config, "get", None)
    layout = (
        getter(SIMT_MMA_LAYOUT_KEY, SIMT_MMA_LAYOUTS[0]) if callable(getter) else None
    )
    return layout if layout in SIMT_MMA_LAYOUTS else SIMT_MMA_LAYOUTS[0]


def simt_gemm_root_layout_supported(
    *,
    bm: int,
    bn: int,
    root_m_threads: int,
    root_n_threads: int,
    root_m_threads_explicit: bool,
    root_n_threads_explicit: bool,
) -> bool:
    """Whether an explicit root thread layout can own a ``bm x bn`` tile.

    Automatic counts are resolved later by the scalar scheduler (always to a
    divisor of the tile); only explicit requests are checked here so the
    detection and the emitter agree on every config the search can produce.
    """
    for extent, threads, explicit in (
        (bm, root_m_threads, root_m_threads_explicit),
        (bn, root_n_threads, root_n_threads_explicit),
    ):
        if explicit and (threads <= 0 or extent % threads != 0):
            return False
    if root_m_threads_explicit and root_n_threads_explicit:
        if root_m_threads * root_n_threads > 1024:
            return False
        if (bm // root_m_threads) * (bn // root_n_threads) > MAX_REGISTER_BLOCK:
            return False
    return True


def device_loop_end_expr(device_loop: DeviceLoopState) -> str:
    """The exclusive end of a ``range(begin, end, step)`` device loop."""
    loop_iter = device_loop.for_node.iter
    assert isinstance(loop_iter, ast.Call) and loop_iter.args
    if len(loop_iter.args) == 1:
        return ast.unparse(loop_iter.args[0])
    return ast.unparse(loop_iter.args[1])


def _reject(reason: str) -> exc.BackendUnsupported:
    return exc.BackendUnsupported("cute", f"{SIMT_MMA_KEY}: {reason}")


def _thread_coord(axis: int | None) -> str:
    if axis is None:
        return "cutlass.Int32(0)"
    return f"cutlass.Int32(cute.arch.thread_idx()[{axis}])"


def _statements(source: str) -> list[ast.stmt]:
    return ast.parse(source).body


@dataclass(frozen=True)
class SimtGemmSite:
    """One admitted matmul: the scalar program's names plus the CTA layout."""

    identity: int
    bm: int
    bn: int
    bk: int
    rm: int
    rn: int
    total_threads: int
    m_size: int
    n_size: int
    m_offset_var: str
    n_offset_var: str
    k_offset_var: str
    k_end_expr: str
    tm_id: str
    tn_id: str
    linear_tid: str
    lane_vars: tuple[str, ...]
    lhs_access: Callable[[str, str], str]
    rhs_access: Callable[[str, str], str]
    lhs_k_contiguous: bool
    rhs_k_contiguous: bool
    m_local: str
    n_local: str
    smem_a: str
    smem_b: str
    smem_c: str
    s_a: str
    s_b: str
    s_c: str
    acc: str
    frag_a: str
    frag_b: str
    stage_a: str
    stage_b: str
    base_a: str
    base_b: str
    tid_var: str
    layout: str


def mark_simt_gemm(
    cg: GenerateAST,
    *,
    m_block_id: int,
    n_block_id: int,
    bm: int,
    bn: int,
    bk: int,
    m_size: int,
    n_size: int,
    m_offset_var: str,
    n_offset_var: str,
    k_offset_var: str,
    k_loop: DeviceLoopState,
    lhs_access: Callable[[str, str], str],
    rhs_access: Callable[[str, str], str],
    lhs_k_contiguous: bool,
    rhs_k_contiguous: bool,
    lhs_arg_name: str,
    rhs_arg_name: str,
    lhs_load: Node,
    rhs_load: Node,
    m_local: str,
    n_local: str,
) -> ast.AST:
    """Record the site and return the marker the scalar program binds."""
    from .cute_mma import _register_collective_handled_loads

    df = cg.device_function
    grid = cg.current_grid_state
    if grid is None:
        raise _reject("no root grid")
    strategy = grid.strategy
    m_axis = grid.block_thread_axes.get(m_block_id)
    n_axis = grid.block_thread_axes.get(n_block_id)
    launch = cg.launch_thread_axis_sizes()
    threads_m = launch.get(m_axis, 1) if m_axis is not None else 1
    threads_n = launch.get(n_axis, 1) if n_axis is not None else 1
    dims = tuple(launch.get(axis, 1) for axis in range(3))
    total_threads = math.prod(dims)
    if total_threads != threads_m * threads_n:
        raise _reject("the launch has threads outside the output tile axes")
    if bm % threads_m or bn % threads_n:
        raise _reject("thread counts must divide the output tile")
    rm = bm // threads_m
    rn = bn // threads_n
    if rm * rn > MAX_REGISTER_BLOCK:
        raise _reject(f"register block {rm}x{rn} exceeds {MAX_REGISTER_BLOCK}")
    elements_fn = getattr(strategy, "_elements_per_thread_for_block", None)
    lane_vars = getattr(strategy, "_lane_var_by_block", None)
    if not callable(elements_fn) or not isinstance(lane_vars, dict):
        raise _reject("root strategy does not expose its lane layout")
    site_lane_vars = []
    for block_id, expected in ((m_block_id, rm), (n_block_id, rn)):
        if elements_fn(block_id) != expected:
            raise _reject("lane count does not match the register block")
        if expected > 1:
            if block_id not in lane_vars:
                raise _reject("register block axis has no lane loop")
            site_lane_vars.append(lane_vars[block_id])
    linear = (
        " + ".join(
            f"{_thread_coord(axis)} * cutlass.Int32({math.prod(dims[:axis])})"
            if axis
            else _thread_coord(axis)
            for axis in range(3)
            if dims[axis] > 1
        )
        or "cutlass.Int32(0)"
    )
    sites = df.cute_state.simt_gemm_sites
    site = SimtGemmSite(
        identity=len(sites),
        bm=bm,
        bn=bn,
        bk=bk,
        rm=rm,
        rn=rn,
        total_threads=total_threads,
        m_size=m_size,
        n_size=n_size,
        m_offset_var=m_offset_var,
        n_offset_var=n_offset_var,
        k_offset_var=k_offset_var,
        k_end_expr=device_loop_end_expr(k_loop),
        tm_id=_thread_coord(m_axis),
        tn_id=_thread_coord(n_axis),
        linear_tid=linear,
        lane_vars=tuple(site_lane_vars),
        lhs_access=lhs_access,
        rhs_access=rhs_access,
        lhs_k_contiguous=lhs_k_contiguous,
        rhs_k_contiguous=rhs_k_contiguous,
        m_local=m_local,
        n_local=n_local,
        smem_a=df.new_var("simt_smem_a"),
        smem_b=df.new_var("simt_smem_b"),
        smem_c=df.new_var("simt_smem_c"),
        s_a=df.new_var("simt_sA"),
        s_b=df.new_var("simt_sB"),
        s_c=df.new_var("simt_sC"),
        acc=df.new_var("simt_acc"),
        frag_a=df.new_var("simt_frag_a"),
        frag_b=df.new_var("simt_frag_b"),
        stage_a=df.new_var("simt_stage_a"),
        stage_b=df.new_var("simt_stage_b"),
        base_a=df.new_var("simt_base_a"),
        base_b=df.new_var("simt_base_b"),
        tid_var=df.new_var("simt_tid"),
        layout=simt_mma_layout(df.config),
    )
    sites.append(site)
    if site.layout == "blocked" and (rm % 4 == 0 or rn % 4 == 0):
        # 128-bit shared reads go through the one-instruction PTX helpers.
        from .cute_mma_dag_codegen import emit_mma_dag_module_statements

        emit_mma_dag_module_statements(cg)
    _register_collective_handled_loads(df.cute_state, lhs_load, rhs_load)
    # The scalar operand loads are placeholders now; the late pass is the only
    # reader of these tensors, so keep their arguments through DCE.
    df.placeholder_args.update((lhs_arg_name, rhs_arg_name))
    cg.cute_uses_matmul = True
    return expr_from_string(f"{MARKER}({site.identity})")


def prepare_cute_simt_gemm_loads(cg: GenerateAST, graph: torch.fx.Graph) -> None:
    """Suppress the operand loads of a K loop the SIMT GEMM owns.

    Runs before the graph is lowered, like the tcgen05 suppression probe: the
    loads precede the matmul in the graph, so marking them at the matmul would
    be too late.  A K loop in ``mma_mode`` with fp32 operands and no
    tensor-core implementation is exactly the set the detection admitted.
    """
    from .cute_mma import _choose_mma_impl
    from .cute_mma import _get_mma_k_loop_info
    from .cute_mma import _register_collective_handled_loads
    from .cute_mma import analyze_cute_mma_node

    df = cg.device_function
    if not simt_mma_requested(df.config) or cg.current_grid_state is None:
        return
    from ..compile_environment import CompileEnvironment

    env = CompileEnvironment.current()
    for node in graph.nodes:
        candidate = analyze_cute_mma_node(node, graphs=cg.codegen_graphs)
        if candidate is None or candidate.requires_accumulator_seed:
            continue
        operands = candidate.operands
        if operands.lhs.source_fake.dtype != torch.float32:
            continue
        k_loop_info = _get_mma_k_loop_info(
            cg,
            env,
            operands.lhs.source_fake,
            operands.rhs.source_fake,
            fx_node=node,
            lhs_k_size=operands.lhs.matrix_cols,
            rhs_k_size=operands.rhs.matrix_rows,
        )
        if k_loop_info is None or k_loop_info[1] != operands.k_block_id:
            continue
        device_loop, _k_block_id, _k_offset, bk = k_loop_info
        if not getattr(device_loop.strategy, "mma_mode", False):
            continue
        bm = df.resolved_block_size(operands.m_block_id)
        bn = df.resolved_block_size(operands.n_block_id)
        if not isinstance(bm, int) or not isinstance(bn, int):
            continue
        if (
            _choose_mma_impl(
                torch.float32,
                bm=bm,
                bn=bn,
                bk=bk,
                config=df.config,
                input_device=operands.lhs.source_fake.device,
            )
            != "universal"
        ):
            continue
        _register_collective_handled_loads(
            df.cute_state, operands.lhs.load, operands.rhs.load
        )


def simt_owned_loop_declined(cg: GenerateAST) -> exc.BackendUnsupported | None:
    """The error for a contraction the detection admitted but the lowering declined.

    Once a K loop is emitted in ``mma_mode`` for the SIMT GEMM it has no lane
    loop, so the scalar fallback would compute one K element per step.  The
    two must agree; a disagreement fails the config loudly instead.
    """
    from ..tile_strategy import DeviceLoopState

    if not simt_mma_requested(cg.device_function.config):
        return None
    owned = any(
        isinstance(loop_state, DeviceLoopState)
        and getattr(loop_state.strategy, "mma_mode", False)
        for loops in cg.active_device_loops.values()
        for loop_state in loops
    )
    if not owned:
        return None
    return _reject(
        "the detection admitted this contraction but the lowering declined it"
    )


def _is_marker(node: ast.AST, identity: int) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == MARKER
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == identity
    )


def _marker_path(
    body: list[ast.stmt], identity: int
) -> list[tuple[list[ast.stmt], int]] | None:
    """The statement lists from ``body`` down to the marker's assignment."""
    for index, statement in enumerate(body):
        if isinstance(statement, ast.Assign) and _is_marker(statement.value, identity):
            return [(body, index)]
        if isinstance(statement, (ast.For, ast.If, ast.While)):
            for branch in (statement.body, statement.orelse):
                if path := _marker_path(branch, identity):
                    return [(body, index), *path]
    return None


def _is_loop_over(statement: ast.stmt, names: Sequence[str]) -> bool:
    return (
        isinstance(statement, ast.For)
        and isinstance(statement.target, ast.Name)
        and statement.target.id in names
    )


def _program(site: SimtGemmSite, k_iter: ast.expr) -> list[ast.stmt]:
    """The whole GEMM: allocations, the staged K loop, the parked tile.

    Shared tiles are ``(bk, extent + pad)``: the staged tile is read one K
    row at a time, and each thread owns the ``rm`` (``rn``) consecutive rows
    ``tid * rm + i`` of its register block, so a K row of the block is one or
    two 128-bit shared reads (``lds_v4_f32``) when the block is a multiple of
    four.  A K-contiguous operand is staged K-fastest (coalesced global reads:
    the consecutive threads of a warp read consecutive K words of one row)
    into rows padded by four words, so those consecutive threads write
    distinct banks and every block row stays 16-byte aligned; an M/N-
    contiguous operand is staged M/N-fastest, which is coalesced and
    conflict-free as is.

    The staging is software pipelined through registers: each thread loads
    its share of the next K step's tiles right after the barrier that
    publishes the current step, so the global latency overlaps the current
    step's FMAs; the loads past the loop's end are masked to zero like any
    partial K tile.
    """
    dtype = "cutlass.Float32"
    bm, bn, bk, rm, rn = site.bm, site.bn, site.bk, site.rm, site.rn
    total = site.total_threads
    pad = 4
    assert isinstance(k_iter, ast.Call) and k_iter.args
    k_begin = ast.unparse(k_iter.args[0]) if len(k_iter.args) > 1 else "0"

    blocked = site.layout == "blocked"

    def owned(tid: str, count: int, threads: int, index: str) -> str:
        """The tile row (column) holding element ``index`` of a thread's block.

        ``blocked`` with a block that is a multiple of four owns quads
        interleaved across the threads: quad ``q`` of thread ``t`` is rows
        ``q * 4 * threads + 4 * t + [0, 4)``, so the eight consecutive
        threads of a quarter-warp read 128 contiguous bytes of one K row
        (conflict-free 128-bit reads; consecutive 16-byte chunks at a
        32-byte stride would be 2-way conflicted). Other blocked counts own
        ``count`` consecutive rows; ``strided`` owns rows ``t + i * threads``.
        """
        if blocked and count % 4 == 0:
            return (
                f"{tid} * cutlass.Int32(4) + cutlass.Int32("
                f"({index}) // 4 * {4 * threads} + ({index}) % 4)"
            )
        if blocked:
            return f"{tid} * cutlass.Int32({count}) + cutlass.Int32({index})"
        return f"{tid} + cutlass.Int32({index} * {threads})"

    def row(index: str) -> str:
        return owned(site.tm_id, rm, bm // rm, index)

    def col(index: str) -> str:
        return owned(site.tn_id, rn, bn // rn, index)

    def fragment_reads(
        frag: str, s_tile: str, base: str, extent: int, count: int, tid: str
    ) -> str:
        """One K row of a thread's block: 128-bit reads for a blocked multiple of 4."""
        if not blocked or count % 4:
            return (
                f"    for _simt_r in cutlass.range_constexpr({count}):\n"
                f"        {frag}[_simt_r] = {s_tile}[_simt_kk, "
                f"{owned(tid, count, extent // count, '_simt_r')}]\n"
            )
        threads = extent // count
        lines = []
        for quad in range(count // 4):
            # Quad ``quad`` of this thread: row quad * 4 * threads + 4 * tid.
            offset = f"cutlass.Int32(_simt_kk * {(extent + pad) * 4} + {quad * 16 * threads})"
            lines.append(
                f"    ({frag}[{4 * quad}], {frag}[{4 * quad + 1}], {frag}[{4 * quad + 2}], "
                f"{frag}[{4 * quad + 3}]) = {RUNTIME}.lds_v4_f32("
                f"{base} + {offset} + {tid} * cutlass.Int32(16))\n"
            )
        return "".join(lines)

    def per_thread(extent: int) -> int:
        return -(-(extent * bk) // total)

    def coordinates(extent: int, k_fastest: bool) -> str:
        if k_fastest:
            return (
                f"    _simt_i = _simt_idx // cutlass.Int32({bk})\n"
                f"    _simt_k = _simt_idx % cutlass.Int32({bk})\n"
            )
        return (
            f"    _simt_i = _simt_idx % cutlass.Int32({extent})\n"
            f"    _simt_k = _simt_idx // cutlass.Int32({extent})\n"
        )

    def stage_load(
        frag: str,
        extent: int,
        size: int,
        offset_var: str,
        access: Callable[[str, str], str],
        k_fastest: bool,
        k_expr: str,
    ) -> str:
        """Each thread's share of one operand tile, from global to registers."""
        count = extent * bk
        guards = [f"_simt_gk < cutlass.Int32({site.k_end_expr})"]
        if size % extent:
            guards.append(f"_simt_g < cutlass.Int32({size})")
        if count % total:
            guards.append(f"_simt_idx < cutlass.Int32({count})")
        return (
            f"for _simt_e in cutlass.range_constexpr({per_thread(extent)}):\n"
            f"    _simt_idx = {site.tid_var} + cutlass.Int32(_simt_e * {total})\n"
            + coordinates(extent, k_fastest)
            + f"    _simt_g = {offset_var} + _simt_i\n"
            f"    _simt_gk = ({k_expr}) + _simt_k\n"
            f"    {frag}[_simt_e] = ({access('_simt_g', '_simt_gk')} "
            f"if {' and '.join(guards)} else {dtype}(0.0))\n"
        )

    def stage_store(frag: str, s_tile: str, extent: int, k_fastest: bool) -> str:
        """The registers of ``stage_load`` into the shared tile."""
        count = extent * bk
        store = f"{s_tile}[_simt_k, _simt_i] = {frag}[_simt_e]"
        if count % total:
            store = f"if _simt_idx < cutlass.Int32({count}):\n        {store}"
        return (
            f"for _simt_e in cutlass.range_constexpr({per_thread(extent)}):\n"
            f"    _simt_idx = {site.tid_var} + cutlass.Int32(_simt_e * {total})\n"
            + coordinates(extent, k_fastest)
            + f"    {store}\n"
        )

    def transposed(access: Callable[[str, str], str]) -> Callable[[str, str], str]:
        return lambda n_expr, k_expr: access(k_expr, n_expr)

    rhs_access = transposed(site.rhs_access)
    operands = (
        (
            site.stage_a,
            site.s_a,
            bm,
            site.m_size,
            site.m_offset_var,
            site.lhs_access,
            site.lhs_k_contiguous,
        ),
        (
            site.stage_b,
            site.s_b,
            bn,
            site.n_size,
            site.n_offset_var,
            rhs_access,
            site.rhs_k_contiguous,
        ),
    )
    setup = _statements(
        f"{site.smem_a} = cute.arch.alloc_smem({dtype}, {bk * (bm + pad)}, alignment=128)\n"
        f"{site.s_a} = cute.make_tensor({site.smem_a}, cute.make_layout(({bk}, {bm}), stride=({bm + pad}, 1)))\n"
        f"{site.smem_b} = cute.arch.alloc_smem({dtype}, {bk * (bn + pad)}, alignment=128)\n"
        f"{site.s_b} = cute.make_tensor({site.smem_b}, cute.make_layout(({bk}, {bn}), stride=({bn + pad}, 1)))\n"
        f"{site.smem_c} = cute.arch.alloc_smem({dtype}, {bm * bn}, alignment=128)\n"
        f"{site.s_c} = cute.make_tensor({site.smem_c}, cute.make_layout(({bm}, {bn}), stride=({bn}, 1)))\n"
        f"{site.tid_var} = {site.linear_tid}\n"
        f"{site.acc} = cute.make_rmem_tensor({rm * rn}, {dtype})\n"
        f"for _simt_i in cutlass.range_constexpr({rm * rn}):\n"
        f"    {site.acc}[_simt_i] = {dtype}(0.0)\n"
        f"{site.frag_a} = cute.make_rmem_tensor({rm}, {dtype})\n"
        f"{site.frag_b} = cute.make_rmem_tensor({rn}, {dtype})\n"
        f"{site.stage_a} = cute.make_rmem_tensor({per_thread(bm)}, {dtype})\n"
        f"{site.stage_b} = cute.make_rmem_tensor({per_thread(bn)}, {dtype})\n"
        f"{site.base_a} = cutlass.Int32({site.smem_a}.toint())\n"
        f"{site.base_b} = cutlass.Int32({site.smem_b}.toint())\n"
        + "".join(
            stage_load(frag, extent, size, offset, access, k_fastest, k_begin)
            for frag, _tile, extent, size, offset, access, k_fastest in operands
        )
    )
    k_body = _statements(
        "".join(
            stage_store(frag, tile, extent, k_fastest)
            for frag, tile, extent, _size, _offset, _access, k_fastest in operands
        )
        + "cute.arch.sync_threads()\n"
        + "".join(
            stage_load(
                frag,
                extent,
                size,
                offset,
                access,
                k_fastest,
                f"{site.k_offset_var} + cutlass.Int32({bk})",
            )
            for frag, _tile, extent, size, offset, access, k_fastest in operands
        )
        + f"for _simt_kk in cutlass.range_constexpr({bk}):\n"
        + fragment_reads(site.frag_a, site.s_a, site.base_a, bm, rm, site.tm_id)
        + fragment_reads(site.frag_b, site.s_b, site.base_b, bn, rn, site.tn_id)
        + f"    for _simt_i in cutlass.range_constexpr({rm}):\n"
        f"        for _simt_j in cutlass.range_constexpr({rn}):\n"
        f"            {site.acc}[_simt_i * {rn} + _simt_j] = cute.math.fma("
        f"{site.frag_a}[_simt_i], {site.frag_b}[_simt_j], "
        f"{site.acc}[_simt_i * {rn} + _simt_j])\n"
        "cute.arch.sync_threads()\n"
    )
    k_loop = ast.For(
        target=ast.Name(id=site.k_offset_var, ctx=ast.Store()),
        iter=k_iter,
        body=k_body,
        orelse=[],
    )
    finish = _statements(
        f"for _simt_i in cutlass.range_constexpr({rm}):\n"
        f"    for _simt_j in cutlass.range_constexpr({rn}):\n"
        f"        {site.s_c}[{row('_simt_i')}, {col('_simt_j')}] = "
        f"{site.acc}[_simt_i * {rn} + _simt_j]\n"
        "cute.arch.sync_threads()\n"
    )
    return [*setup, k_loop, *finish]


def _invariant_definitions(
    path: Sequence[tuple[list[ast.stmt], int]],
    stop: tuple[list[ast.stmt], int],
    names: set[str],
    lane_vars: Sequence[str],
) -> list[ast.stmt]:
    """Copies of the nest's lane-invariant assignments to ``names``.

    The full nest keeps lane-invariant setup (``tile_end``) inside the lane
    loops; a program spliced ahead of them needs those definitions first.
    Only plain assignments whose reads are themselves lane-invariant qualify,
    found in program order along ``path`` (the statement lists inside the
    insertion scope, whose definitions the spliced program no longer
    follows) up to ``stop``.
    """
    from ..ast_read_writes import ReadWrites

    pending = set(names)
    found: list[ast.stmt] = []
    for statements, index in path:
        limit = index if statements is not stop[0] else stop[1]
        for statement in statements[:limit]:
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                continue
            target = statement.targets[0].id
            if target not in pending:
                continue
            reads = set(ReadWrites.from_list([statement.value]).reads)
            if reads & set(lane_vars):
                raise _reject(f"{target} varies with the lane loops")
            found.append(statement)
            pending.discard(target)
        if statements is stop[0]:
            break
    return found


def lower_simt_gemm(
    body: list[ast.stmt], df: DeviceFunction, *, rename_groups: dict[str, str]
) -> list[ast.stmt]:
    """Splice every marked SIMT GEMM ahead of its lane nest."""
    from ..ast_read_writes import ReadWrites
    from ..ast_read_writes import ast_rename

    sites = df.cute_state.simt_gemm_sites
    if not sites:
        return body
    # Phi names must be canonical before the carried result is rebound.
    ast_rename(ast.Module(body=body, type_ignores=[]), rename_groups)
    for site in sites:
        path = _marker_path(body, site.identity)
        if path is None:
            raise _reject("pending GEMM marker was not a scalar assignment")
        k_loop_at: tuple[list[ast.stmt], int] | None = None
        for statements, _index in path:
            for index, statement in enumerate(statements):
                if _is_loop_over(statement, (site.k_offset_var,)):
                    k_loop_at = (statements, index)
        if k_loop_at is None:
            raise _reject("the contraction loop was not found beside its marker")
        k_parent, k_index = k_loop_at
        original = k_parent[k_index]
        assert isinstance(original, ast.For)
        insertion: tuple[list[ast.stmt], int] = k_loop_at
        for statements, index in path:
            if _is_loop_over(statements[index], site.lane_vars):
                insertion = (statements, index)
                break
        marker_parent, marker_index = path[-1]
        assignment = marker_parent[marker_index]
        assert isinstance(assignment, ast.Assign)
        assignment.value = cast(
            "ast.expr",
            expr_from_string(f"{site.s_c}[{site.m_local}, {site.n_local}]"),
        )
        program = _program(site, original.iter)
        inner = next(
            i
            for i, (statements, _index) in enumerate(path)
            if statements is insertion[0]
        )
        program[:0] = _invariant_definitions(
            path[inner + 1 :],
            k_loop_at,
            set(ReadWrites.from_list(program).reads),
            site.lane_vars,
        )
        # The scalar K loop kept only placeholder operand loads; the staged
        # loop replaces it.  A result bound inside the loop (the carried
        # accumulator) is rebound where the loop stood: the lanes' reads of
        # the parked tile follow the loop like its final iteration did.
        marker_inside_loop = any(statements is original.body for statements, _ in path)
        if marker_inside_loop:
            k_parent[k_index] = assignment
        else:
            del k_parent[k_index]
        insertion_body, insertion_index = insertion
        if (
            insertion_body is k_parent
            and insertion_index > k_index
            and not marker_inside_loop
        ):
            insertion_index -= 1
        insertion_body[insertion_index:insertion_index] = program
    if any(
        isinstance(node, ast.Name) and node.id == MARKER
        for statement in body
        for node in ast.walk(statement)
    ):
        raise _reject("unlowered SIMT GEMM marker")
    return body
