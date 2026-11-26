"""Map a root's independent FP32 row sums to warps of an existing linear CTA.

The ordinary lowering still chooses the reduction chunks, addresses, masks,
arithmetic and casts. Only its serial K and output coordinates change. The
root must terminate in a single injective store, with no intersecting inputs.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

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


def _tensor(call: ast.Call, operation: str) -> str | None:
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


def _pure(node: ast.AST, inputs: set[str]) -> bool:
    for a in ast.walk(node):
        if isinstance(a, (ast.NamedExpr, ast.Lambda, ast.comprehension)):
            return False
        if isinstance(a, ast.Call):
            if (
                (base := _tensor(a, "load")) is not None
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
class _Sweep:
    inner: ast.For
    index: ast.Assign
    finalize: ast.Call


def _prove_sweep(
    init: ast.AST,
    outer: ast.For,
    finish: ast.AST,
    strategy: LoopedReductionStrategy,
    k: int,
    inputs: set[str],
) -> _Sweep | None:
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
        and chunk >= WARP
        and chunk % WARP == 0
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
        if not isinstance(stmt, ast.Assign) or not _pure(stmt.value, inputs):
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
    return _Sweep(inner, index, finalize)


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
        and mask
        and not grid.outer_suffix
        and not grid.body_finalizers
        and not grid.vec_lane_wrappers
        and not any(
            _thread(n)
            for s in (grid.hoist_parent_statements or ())
            for n in ast.walk(s)
        )
        and not any(_thread(n) for s in grid.lane_setup_statements for n in ast.walk(s))
    ):
        return
    # The only thread-dependent setup is the ordinary row coordinate and
    # its row mask. Its offset and unit stride prove injective row ownership.
    setup = {
        assigned_name(s): s for s in grid.outer_prefix if isinstance(s, ast.Assign)
    }
    ix, pred = setup.get(row_index), setup.get(mask)
    if not (
        isinstance(ix, ast.Assign)
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
    terminal = body[-1] if body else None
    if not (
        isinstance(terminal, ast.If)
        and _name(terminal.test, mask)
        and not terminal.orelse
        and len(terminal.body) == 1
        and isinstance(expr := terminal.body[0], ast.Expr)
        and isinstance(store := expr.value, ast.Call)
        and (output := _tensor(store, "store"))
        and len(store.args) == 1
        and not store.keywords
    ):
        return
    inputs: set[str] = set()
    specs: list[_Sweep] = []
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
        and _pure(addr, inputs)
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
                    sw := _prove_sweep(
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
            and _pure(stmt.value, inputs)
            and not names_read(stmt.value) & inner_locals
        ):
            return
    if (
        not _pure(store.args[0], inputs)
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

    for s in (ix, pred):
        WarpCoordinate().visit(s)
    for sw in specs:
        assert isinstance(sw.inner.iter, ast.Call)
        sw.inner.iter.args[0] = ast.Constant(reduction._loop_block_size // WARP)
        assert isinstance(sw.index.value, ast.BinOp) and isinstance(
            sw.index.value.left, ast.BinOp
        )
        assert isinstance(sw.index.value.right, ast.BinOp)
        sw.index.value.left.right = parse_expr(
            "cutlass.Int32(cute.arch.thread_idx()[0] % 32)"
        )
        sw.index.value.right.right = ast.Constant(WARP)
        sw.finalize.keywords[0].value = ast.Constant(WARP)
    terminal.test = ast.BoolOp(
        op=ast.And(),
        values=[terminal.test, parse_expr("cute.arch.thread_idx()[0] % 32 == 0")],
    )
    state = fn.cute_state.lanes
    state.warp_row_threads = max(state.warp_row_threads, WARP * rows)


def validate_row_launch(fn: DeviceFunction, dims: tuple[int, int, int] | None) -> None:
    """The emitted partition and all its peers must actually share this launch."""
    need = fn.cute_state.lanes.warp_row_threads
    if (
        not need
        or dims is None
        or dims[1:] != (1, 1)
        or dims[0] < need
        or dims[0] % WARP
    ):
        raise exc.InvalidConfig(
            f"{KEY} needs a proved serial row sum and a linear CTA with {need or WARP} threads"
        )
