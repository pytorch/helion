"""Map a root's independent FP32 row sums to warps of an existing CTA.

The ordinary lowering still chooses the reduction chunks, addresses, masks,
arithmetic and casts. Only its serial K and output coordinates change. The
root must terminate in a single injective store, with no intersecting inputs.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ...language import _tracing_ops
from ...language import memory_ops
from ...language import tile_ops
from ...language import view_ops
from ..compile_environment import CompileEnvironment
from ..inductor_lowering import PointwiseLowering
from ..inductor_lowering import ReductionLowering
from ..reduction_strategy import LoopedReductionStrategy
from ._ast_pass_utils import assigned_name
from ._ast_pass_utils import call_path
from ._ast_pass_utils import names_read
from ._ast_pass_utils import parse_expr
from ._ast_pass_utils import same_expr
from .row_carry_terms import empty_if

if TYPE_CHECKING:
    from ..device_function import DeviceFunction
    from ..device_ir import DeviceIR
    from ..generate_ast import GenerateAST
    from ..tile_strategy import DeviceGridState

KEY = "cute_warp_row_reductions"
WARP = 32


@dataclass(frozen=True)
class RowSums:
    row_id: int
    k_id: int
    length: int
    reductions: int


def row_sum_roots(env: CompileEnvironment, ir: DeviceIR) -> dict[int, RowSums]:
    """Unrolled IR: scalar/row FP32 sums over one exact K, one row-indexed store."""
    if len(ir.root_ids) < 2 or ir.host_function is None:
        return {}
    result: dict[int, RowSums] = {}
    with ir.host_function:
        for root_id, grid in zip(ir.root_ids, ir.grid_block_ids, strict=True):
            nodes = list(ir.graphs[root_id].graph.nodes)
            reductions = [
                (n, lower)
                for n in nodes
                if isinstance(lower := n.meta.get("lowering"), ReductionLowering)
            ]
            stores = [n for n in nodes if n.target is memory_ops.store]
            if not reductions or len(stores) != 1 or stores[0] is not nodes[-2]:
                continue
            store = stores[0]
            value = store.args[2]
            indices = store.args[1]
            if not (
                isinstance(value, torch.fx.Node)
                and value is reductions[-1][0]
                and isinstance(indices, (list, tuple))
                and indices
                and isinstance(last := indices[-1], torch.fx.Node)
                and last.target is _tracing_ops._get_symnode
                and isinstance(sym := last.meta.get("val"), torch.SymInt)
                and (row := env.get_block_id(sym)) in grid
                and row is not None
                and all(
                    isinstance(i, int)
                    or (
                        isinstance(i, torch.fx.Node) and i.target is tile_ops.tile_begin
                    )
                    for i in indices[:-1]
                )
            ):
                continue
            axis = reductions[-1][1].block_index
            extent = env.block_sizes[axis].size
            if not isinstance(extent, (int, torch.SymInt)):
                continue
            k = env.static_kernel_tensor_extent(extent)
            if k is None or k < WARP or k % WARP:
                continue
            if not all(
                lower.block_index == axis
                and lower.reduction_type == "sum"
                and isinstance(v := n.meta.get("val"), torch.Tensor)
                and v.dtype is torch.float32
                and (
                    v.ndim == 0 or (v.ndim == 1 and env.get_block_id(v.shape[0]) == row)
                )
                for n, lower in reductions
            ):
                continue
            if any(
                n.op != "output"
                and not (
                    n.op == "call_function"
                    and (
                        isinstance(
                            n.meta.get("lowering"),
                            (ReductionLowering, PointwiseLowering),
                        )
                        or n.target
                        in (
                            _tracing_ops._get_symnode,
                            _tracing_ops._host_tensor,
                            _tracing_ops._mask_to,
                            tile_ops.tile_begin,
                            view_ops.subscript,
                            memory_ops.load,
                            memory_ops.store,
                        )
                    )
                )
                for n in nodes
            ):
                continue
            result[root_id] = RowSums(row, axis, k, len(reductions))
    return result


def _name(node: ast.AST, value: str) -> bool:
    return isinstance(node, ast.Name) and node.id == value


def _cast(node: ast.AST, dtype: str) -> ast.expr | None:
    return (
        node.args[0]
        if isinstance(node, ast.Call)
        and call_path(node.func) == ("cutlass", dtype)
        and len(node.args) == 1
        and not node.keywords
        else None
    )


def _constant(node: ast.AST, value: int) -> bool:
    return (
        isinstance(node, ast.Constant)
        and type(node.value) is int
        and node.value == value
    )


def _thread(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and _constant(node.slice, 0)
        and (
            isinstance(node.value, ast.Call)
            and call_path(node.value.func) == ("cute", "arch", "thread_idx")
            and not node.value.args
            and not node.value.keywords
        )
    )


def sweep_tensor_base(call: ast.Call, operation: str) -> str | None:
    if not isinstance(call.func, ast.Attribute) or call.func.attr != operation:
        return None
    ptr = call.func.value
    while isinstance(ptr, ast.BinOp) and isinstance(ptr.op, ast.Add):
        ptr = ptr.left
    return (
        ptr.value.id
        if isinstance(ptr, ast.Attribute)
        and ptr.attr == "iterator"
        and isinstance(ptr.value, ast.Name)
        else None
    )


def pure_row_expression(node: ast.AST, inputs: set[str]) -> bool:
    for a in ast.walk(node):
        if isinstance(a, (ast.NamedExpr, ast.Lambda, ast.comprehension)):
            return False
        if isinstance(a, ast.Call):
            if (
                (base := sweep_tensor_base(a, "load")) is not None
                and not a.args
                and not a.keywords
            ):
                inputs.add(base)
            elif call_path(a.func) == ("cute", "math", "div"):
                if (
                    len(a.args) != 2
                    or len(a.keywords) != 2
                    or {kw.arg for kw in a.keywords} != {"approx", "ftz"}
                    or any(
                        not isinstance(kw.value, ast.Constant)
                        or kw.value.value is not True
                        for kw in a.keywords
                    )
                ):
                    return False
            elif call_path(a.func) not in {
                ("cutlass", "Float32"),
                ("cutlass", "Float16"),
                ("cutlass", "BFloat16"),
                ("cutlass", "Int32"),
                ("cutlass", "Int64"),
                ("cute", "math", "rsqrt"),
            }:
                return False
    return True


@dataclass
class RowSweep:
    inner: ast.For
    index: ast.Assign
    finalize: ast.Call


def prove_row_sweep(
    init: ast.AST,
    outer: ast.For,
    finish: ast.AST,
    strategy: LoopedReductionStrategy,
    k: int,
    inputs: set[str],
    *,
    chunk_multiple: int = WARP,
) -> RowSweep | None:
    acc = assigned_name(init)
    chunk = strategy._loop_block_size
    if not (
        isinstance(init, ast.Assign)
        and acc
        and (zero := _cast(init.value, "Float32")) is not None
        and _constant(zero, 0)
        and isinstance(outer.target, ast.Name)
        and not outer.orelse
        and len(outer.body) == 1
        and isinstance(inner := outer.body[0], ast.For)
        and isinstance(inner.target, ast.Name)
        and not inner.orelse
        and chunk >= chunk_multiple
        and chunk % chunk_multiple == 0
        and k % chunk == 0
        and isinstance(inner.iter, ast.Call)
        and _name(inner.iter.func, "range")
        and len(inner.iter.args) == 1
        and _constant(inner.iter.args[0], chunk)
        and not inner.iter.keywords
        and isinstance(outer.iter, ast.Call)
        and _name(outer.iter.func, "range")
        and len(outer.iter.args) == 3
        and not outer.iter.keywords
        and (start := _cast(outer.iter.args[0], "Int32")) is not None
        and _constant(start, 0)
        and (end := _cast(outer.iter.args[1], "Int32")) is not None
        and _constant(end, k)
        and (step := _cast(outer.iter.args[2], "Int32")) is not None
        and isinstance(step, ast.Name)
        and step.id == strategy.block_size_var(strategy.block_index)
        and len(inner.body) >= 2
        and isinstance(index := inner.body[0], ast.Assign)
        and assigned_name(index) == strategy.index_var(strategy.block_index)
        and same_expr(
            index.value,
            parse_expr(
                f"{outer.target.id} + cutlass.Int32(0) + cutlass.Int32({inner.target.id}) * 1"
            ),
        )
        and isinstance(finish, ast.Assign)
        and assigned_name(finish)
        and isinstance(finalize := _cast(finish.value, "Float32"), ast.Call)
        and call_path(finalize.func) == ("cute", "arch", "warp_reduction_sum")
        and len(finalize.args) == 1
        and _name(finalize.args[0], acc)
        and len(finalize.keywords) == 1
        and finalize.keywords[0].arg == "threads_in_group"
        and _constant(finalize.keywords[0].value, 1)
    ):
        return None
    # All non-accumulator assignments are fresh per K coordinate. A second
    # carried recurrence, indexing by the lane-loop induction, or a side
    # effect would no longer be equivalent after walking K in strides of 32.
    names = [assigned_name(s) for s in inner.body]
    if None in names or len(names) != len(set(names)) or names[-1] != acc:
        return None
    local = {n for n in names if n is not None}
    defined = {names[0]}
    for stmt in inner.body[1:]:
        if not isinstance(stmt, ast.Assign) or not pure_row_expression(
            stmt.value, inputs
        ):
            return None
        rhs = stmt.value
        if assigned_name(stmt) == acc:
            if not (
                isinstance(rhs, ast.BinOp)
                and isinstance(rhs.op, ast.Add)
                and _name(rhs.left, acc)
            ):
                return None
            rhs = rhs.right
        reads = names_read(rhs)
        if (
            reads.intersection({inner.target.id, outer.target.id})
            or reads & local - defined
        ):
            return None
        defined.add(assigned_name(stmt))
    return RowSweep(inner, index, finalize)


@dataclass(frozen=True)
class _RowWaves:
    index: ast.Assign
    mask: ast.Assign
    base: ast.Name
    lane: str
    warps: int


def _row_waves(
    grid: DeviceGridState,
    body: list[ast.AST],
    row: int,
    rows: int,
    index: str,
    mask: str,
) -> _RowWaves | None:
    """Prove a serial row sweep owns each element of a contiguous row tile once."""
    if not grid.lane_loops or grid.deferred_vector_ops:
        return None
    needed, setup, spliced = grid._live_lane_setup(body)
    loops = [(v, size) for v, size in grid.lane_loops if v in needed]
    if len(loops) != 1 or len(setup) != 2 or spliced:
        return None
    lane, steps = loops[0]
    if steps < 2 or rows % steps or grid.lane_loop_block_ids.get(lane) != {row}:
        return None
    warps = rows // steps
    ix, pred = setup
    if not (
        isinstance(ix, ast.Assign)
        and assigned_name(ix) == index
        and isinstance(term := ix.value, ast.BinOp)
        and isinstance(term.op, ast.Add)
        and (last := _cast(term.right, "Int32")) is not None
        and _name(last, lane)
        and isinstance(offset := term.left, ast.BinOp)
        and isinstance(offset.op, ast.Add)
        and isinstance(base := offset.left, ast.Name)
        and base.id == grid.strategy.grid_origin_var(row)
        and isinstance(mul := offset.right, ast.BinOp)
        and isinstance(mul.op, ast.Mult)
        and _constant(mul.right, steps)
        and (x := _cast(mul.left, "Int32")) is not None
        and _thread(x)
        and isinstance(pred, ast.Assign)
        and assigned_name(pred) == mask
        and isinstance(pred.value, ast.BoolOp)
        and isinstance(pred.value.op, ast.And)
        and len(pred.value.values) == 2
        and isinstance(own := pred.value.values[0], ast.Compare)
        and len(own.ops) == 1
        and isinstance(own.ops[0], ast.Lt)
        and same_expr(own.left, mul.left)
        and _constant(own.comparators[0], warps)
        and isinstance(bound := pred.value.values[1], ast.Compare)
        and len(bound.ops) == 1
        and isinstance(bound.ops[0], ast.Lt)
        and _name(bound.left, index)
        and isinstance(bound.comparators[0], ast.Constant)
        and type(bound.comparators[0].value) is int
        and bound.comparators[0].value > 0
    ):
        return None
    # Grid origins and other hoisted coordinates must be ordinary arithmetic
    # on uniform values. A call (including any intrinsic) or subscript could
    # hide a second, thread-dependent coordinate or an observable load.
    if any(
        not isinstance(stmt, ast.Assign)
        or not assigned_name(stmt)
        or any(
            isinstance(n, (ast.Call, ast.Subscript, ast.Attribute))
            for n in ast.walk(stmt.value)
        )
        for stmt in [*grid.outer_prefix, *(grid.hoist_parent_statements or ())]
    ) or any(lane in names_read(stmt) for stmt in body):
        return None
    return _RowWaves(ix, pred, base, lane, warps)


def _single_row_owner(
    fn: DeviceFunction,
    grid: DeviceGridState,
    body: list[ast.AST],
    row: int,
    index: str,
) -> bool:
    """An ordinary one-row tile already gives its whole row to this program ID."""
    if not body or not isinstance(body[-1], ast.Expr) or fn.pid is None:
        return False
    shared_pid = fn.pid.shared_pid_var
    needed, setup, spliced = grid._live_lane_setup(body)
    if (
        not shared_pid
        or grid.outer_prefix
        or grid.deferred_vector_ops
        or setup
        or spliced
        or any(lane in needed for lane, _ in grid.lane_loops)
    ):
        return False
    # Permit only ordinary scalar PID decoding. In particular, even a
    # syntactically scalar row must not come from a lane, load or capture.
    uniform = {a.name for a in fn.arguments} | {shared_pid, "cutlass", "cute"}
    origin = grid.strategy.grid_origin_var(row)
    exact_pid = {shared_pid}
    units: set[str] = set()
    proved_index = False
    for stmt in grid.hoist_parent_statements or ():
        if not isinstance(stmt, ast.Assign) or not (name := assigned_name(stmt)):
            return False
        reads = names_read(stmt.value)
        if (
            name in uniform
            or not reads <= uniform
            or any(
                not isinstance(
                    n,
                    (
                        ast.Name,
                        ast.Load,
                        ast.Constant,
                        ast.BinOp,
                        ast.Add,
                        ast.Sub,
                        ast.Mult,
                        ast.FloorDiv,
                        ast.Mod,
                    ),
                )
                for n in ast.walk(stmt.value)
            )
        ):
            return False
        # The one-row grid coordinate is the local PID, possibly decoded
        # through singleton outer axes. A modulo, scale, or non-unit
        # division could cause two CTAs to elect a writer for the same row.
        value = stmt.value
        is_pid = isinstance(value, ast.Name) and value.id in exact_pid
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.FloorDiv):
            is_pid = (
                isinstance(value.left, ast.Name)
                and value.left.id in exact_pid
                and (
                    _constant(value.right, 1)
                    or isinstance(value.right, ast.Name)
                    and value.right.id in units
                )
            )
        if is_pid:
            exact_pid.add(name)
        if _constant(value, 1) or isinstance(value, ast.Name) and value.id in units:
            units.add(name)
        if name == index:
            proved_index = _name(value, origin) and origin in exact_pid
        uniform.add(name)
    if not proved_index:
        return False
    # Every value read by the transformed scalar code must be a uniform
    # argument/coordinate, a value computed in this row, or the current K.
    # A value left over from a previous K iteration or phase is not one.
    # Subscripts are limited to layout-stride metadata: tensor reads must go
    # through the ordinary load-inventory and destination-disjointness proof.
    for position, stmt in enumerate(body):
        for node in ast.walk(stmt):
            if isinstance(node, ast.Subscript) and not (
                isinstance(node.value, ast.Attribute)
                and node.value.attr == "stride"
                and isinstance(layout := node.value.value, ast.Attribute)
                and layout.attr == "layout"
                and isinstance(layout.value, ast.Name)
                and isinstance(node.slice, ast.Constant)
                and type(node.slice.value) is int
            ):
                return False
        if isinstance(stmt, ast.For):
            if not isinstance(stmt.target, ast.Name) or len(stmt.body) != 1:
                return False
            inner = stmt.body[0]
            if not isinstance(inner, ast.For) or not isinstance(inner.target, ast.Name):
                return False
            acc = assigned_name(body[position - 1]) if position else None
            local = uniform | {stmt.target.id, inner.target.id}
            for term in inner.body:
                if not isinstance(term, ast.Assign) or not (lhs := assigned_name(term)):
                    return False
                if (lhs in local and lhs != acc) or not names_read(term.value) <= local:
                    return False
                local.add(lhs)
            # No K-local name escapes the sweep; prove_row_sweep separately
            # checks the one loop-carried FP32 sum and its final shuffle.
        elif isinstance(stmt, ast.Assign) and (lhs := assigned_name(stmt)):
            if not names_read(stmt.value) <= uniform or lhs in uniform:
                return False
            uniform.add(lhs)
        elif stmt is not body[-1] or not names_read(stmt) <= uniform:
            return False
    return True


def partition_row_sums(
    cg: GenerateAST, grid: DeviceGridState, body: list[ast.AST]
) -> None:
    fn = cg.device_function
    if not fn.config.get(KEY) or (root := cg.current_root_graph_info) is None:
        return
    env = CompileEnvironment.current()
    facts = env.config_spec.cute_warp_row_reduction_roots.get(root.graph_id)
    if facts is None:
        return
    row, kid, k, count = facts
    row_index = grid.strategy.index_var(row)
    mask = grid.strategy.mask_var(row)
    reduction = fn.tile_strategy.block_id_to_strategy.get((kid,))
    sizes = env.config_spec.block_sizes
    rows = sizes.config_get(fn.config.block_sizes, row, 0)
    if not (
        isinstance(reduction, LoopedReductionStrategy)
        and reduction._thread_count == 1
        and 1 <= rows <= 32
        and (mask or rows == 1)
        and not grid.outer_suffix
        and not grid.body_finalizers
        and not grid.vec_lane_wrappers
        and not any(
            _thread(n)
            for s in (grid.hoist_parent_statements or ())
            for n in ast.walk(s)
        )
    ):
        return
    # The only thread-dependent setup is the ordinary row coordinate and
    # its row mask. Its offset and unit stride prove injective row ownership.
    single = rows == 1 and mask is None
    waves = _row_waves(grid, body, row, rows, row_index, mask) if mask else None
    if single:
        ir = cg.host_function.device_ir
        if any(
            axis != row
            and (
                type(env.block_sizes[axis].size) is not int
                or env.block_sizes[axis].size != 1
            )
            for axis in ir.grid_block_ids[ir.root_ids.index(root.graph_id)]
        ) or not _single_row_owner(fn, grid, body, row, row_index):
            return
        ix = pred = None
    elif waves is None:
        assert mask
        setup = {
            assigned_name(s): s for s in grid.outer_prefix if isinstance(s, ast.Assign)
        }
        ix, pred = setup.get(row_index), setup.get(mask)
        if not (
            not any(_thread(n) for s in grid.lane_setup_statements for n in ast.walk(s))
            and isinstance(ix, ast.Assign)
            and isinstance(ix.value, ast.BinOp)
            and isinstance(ix.value.op, ast.Add)
            and not any(_thread(n) for n in ast.walk(ix.value.left))
            and isinstance(term := ix.value.right, ast.BinOp)
            and isinstance(term.op, ast.Mult)
            and _constant(term.right, 1)
            and (x := _cast(term.left, "Int32")) is not None
            and _thread(x)
            and isinstance(pred, ast.Assign)
            and isinstance(pred.value, ast.BoolOp)
            and isinstance(pred.value.op, ast.And)
            and any(
                isinstance(v, ast.Compare)
                and len(v.ops) == 1
                and isinstance(v.ops[0], ast.Lt)
                and same_expr(v.left, term.left)
                and _constant(v.comparators[0], rows)
                for v in pred.value.values
            )
            and sum(_thread(n) for s in grid.outer_prefix for n in ast.walk(s)) == 2
        ):
            return
    else:
        ix, pred = waves.index, waves.mask
    terminal = body[-1] if body else None
    if single:
        expr = terminal
    elif (
        mask
        and isinstance(terminal, ast.If)
        and _name(terminal.test, mask)
        and not terminal.orelse
        and len(terminal.body) == 1
    ):
        expr = terminal.body[0]
    else:
        return
    if not (
        isinstance(expr, ast.Expr)
        and isinstance(store := expr.value, ast.Call)
        and (output := sweep_tensor_base(store, "store"))
        and len(store.args) == 1
        and not store.keywords
    ):
        return
    inputs: set[str] = set()
    specs: list[RowSweep] = []
    ptr = store.func
    # The emitted address must really be base + row * its stride, with no
    # dependence on row in the other dimensions. The FX index proof alone
    # does not certify the final pointer after memory lowering.
    if not (
        isinstance(ptr, ast.Attribute)
        and isinstance(addr := ptr.value, ast.BinOp)
        and isinstance(addr.op, ast.Add)
        and row_index not in names_read(addr.left)
        and isinstance(mul := addr.right, ast.BinOp)
        and isinstance(mul.op, ast.Mult)
        and (idx := _cast(mul.left, "Int32")) is not None
        and _name(idx, row_index)
        and isinstance(stride := _cast(mul.right, "Int32"), ast.Subscript)
        and call_path(stride.value) == (output, "layout", "stride")
        and isinstance(stride.slice, ast.Constant)
        and type(stride.slice.value) is int
        and fn.proven_tensor_stride_values().get((output, stride.slice.value)) == 1
        and pure_row_expression(addr, inputs)
        and not inputs
    ):
        return
    skipped: set[int] = set()
    inner_locals: set[str] = set()
    for i, stmt in enumerate(body[:-1]):
        if isinstance(stmt, ast.For):
            if (
                not 0 < i < len(body) - 2
                or (
                    sw := prove_row_sweep(
                        body[i - 1], stmt, body[i + 1], reduction, k, inputs
                    )
                )
                is None
            ):
                return
            specs.append(sw)
            skipped.update([i - 1, i, i + 1])
            inner_locals.update(
                n for s in sw.inner.body[:-1] if (n := assigned_name(s)) is not None
            )
    if len(specs) != count or len(skipped) != 3 * count:
        return
    for i, stmt in enumerate(body[:-1]):
        if i not in skipped and not (
            isinstance(stmt, ast.Assign)
            and assigned_name(stmt)
            and pure_row_expression(stmt.value, inputs)
            and not names_read(stmt.value) & inner_locals
        ):
            return
    if (
        not pure_row_expression(store.args[0], inputs)
        or names_read(store.args[0]) & inner_locals
        or any(
            frozenset((output, name)) not in fn.proven_disjoint_tensor_pairs()
            for name in inputs
        )
    ):
        return

    # Mutate only after all sweeps and the final publication validate.
    class WarpCoordinate(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            if _thread(node):
                return parse_expr("cute.arch.thread_idx()[0] // 32")
            return self.generic_visit(node)

    if single:
        actual_lane = "cute.arch.lane_idx()"
    elif waves is None:
        assert ix is not None and pred is not None
        for s in (ix, pred):
            WarpCoordinate().visit(s)
        actual_lane = "cute.arch.thread_idx()[0] % 32"
    else:
        # Warp/lane coordinates are linear across all CTA axes. The existing
        # loop still iterates exactly rows / warps times; each warp owns one
        # row per iteration and only its elected lane may publish it.
        waves.index.value = parse_expr(
            f"{waves.base.id} + cutlass.Int32(cute.arch.warp_idx()) + "
            f"cutlass.Int32({waves.lane}) * {waves.warps}"
        )
        assert isinstance(waves.mask.value, ast.BoolOp)
        waves.mask.value.values[0] = parse_expr(f"cute.arch.warp_idx() < {waves.warps}")
        actual_lane = "cute.arch.lane_idx()"
    for sw in specs:
        assert isinstance(sw.inner.iter, ast.Call)
        sw.inner.iter.args[0] = ast.Constant(reduction._loop_block_size // WARP)
        assert isinstance(sw.index.value, ast.BinOp) and isinstance(
            sw.index.value.left, ast.BinOp
        )
        assert isinstance(sw.index.value.right, ast.BinOp)
        sw.index.value.left.right = parse_expr(f"cutlass.Int32({actual_lane})")
        sw.index.value.right.right = ast.Constant(WARP)
        sw.finalize.keywords[0].value = ast.Constant(WARP)
    state = fn.cute_state.lanes
    if single:
        assert isinstance(terminal, ast.stmt)
        leader = empty_if(f"{actual_lane} == 0")
        leader.body = [terminal]
        warp = empty_if("cute.arch.warp_idx() == 0")
        warp.body = [*cast("list[ast.stmt]", body[:-1]), leader]
        body[:] = [warp]
        state.has_singleton_warp_rows = True
        return
    assert isinstance(terminal, ast.If)
    terminal.test = ast.BoolOp(
        op=ast.And(),
        values=[terminal.test, parse_expr(f"{actual_lane} == 0")],
    )
    if waves is None:
        state.warp_row_threads = max(state.warp_row_threads, WARP * rows)
    else:
        state.warp_row_wave_counts.add(waves.warps)


def validate_row_launch(fn: DeviceFunction, dims: tuple[int, int, int] | None) -> None:
    """The emitted partition and all its peers must actually share this launch."""
    state = fn.cute_state.lanes
    if state.has_singleton_warp_rows:
        launch = state.launch_block
        if (
            launch is None
            or dims != launch
            or not WARP <= prod(launch) <= 1024
            or prod(launch) % WARP
        ):
            raise exc.InvalidConfig(
                f"{KEY} needs a full physical warp for each scalar row"
            )
        if not state.warp_row_wave_counts and not state.warp_row_threads:
            return
    if state.warp_row_wave_counts:
        if (
            dims is None
            or dims != state.launch_block
            or {prod(dims)} != {WARP * count for count in state.warp_row_wave_counts}
        ):
            raise exc.InvalidConfig(
                f"{KEY} needs the exact proved number of complete physical warps"
            )
        if not state.warp_row_threads:
            return
    need = state.warp_row_threads
    if (
        not need
        or dims is None
        or dims != state.launch_block
        or dims[1:] != (1, 1)
        or dims[0] < need
        or dims[0] % WARP
    ):
        raise exc.InvalidConfig(
            f"{KEY} needs a proved serial row sum and a linear CTA with {need or WARP} threads"
        )
