"""Row values of fragment results, held in registers ("lines").

A fragment group's accumulator (``fragment_mma``) is a grid of 8x8 float
fragments in which lane ``l`` holds row ``fm`` of every fragment (``fm =
((l / 4) & 4) + ((l / 2) % 4)``), so the lanes holding one row differ only in
bits 0 and 3 of ``l``.  A value over the accumulator's rows, with no column,
is then naturally per lane: slot ``r`` of a lane holds row ``8 r + fm`` of
the tile, the same in the four lanes of the row.  This module keeps such
values, *lines*, in registers (``float x[R]``, ``R`` the grid's rows),
where the Scalar layout would need the result materialized:

- a reduction over the N block of a cooperative value (sum, max, min,
  prod) is a line: each lane folds the elements it holds of the row, then
  ``simd_shuffle_xor`` 1 and 8 fold the row's four lanes, a combine every
  lane runs alike, as the fragment element loops are unrolled;
- an element-wise op of lines, values an element recomputes over the rows
  and 0-d values is a line, computed per slot;
- a loop carry whose value in the loop is a line is one: it starts from a
  seed (a constant fill, or a value recomputed over the rows), written into
  its slots before the loop (``_frag_line_seed``), and is a line only
  whole, its values in and after the loop all lines;
- an element loop reads a line in place, at its element's slot: an
  accumulator's update (``acc * alpha[:, None]``), an epilogue (``acc /
  l_i[:, None]``), a reduction's input (``exp(s - m[:, None])``);
- a store of a line into a destination over its rows (one element per
  row) is written by the lane holding column 0 of the row.

Any other use of a line (a Scalar op, an index) drops it, and its
cooperative inputs fall back to the Scalar layout, which the fragment
lowering refuses.  A reduction over the M block (a column of the grid) is
not a line.

The planner part, :func:`plan_lines`, runs inside the MPP planner for a
fragment plan, before element staging; codegen emits ``_helion_mpp_line_*``
markers that ``mpp_matmul.elementize`` turns into line loops, and
:func:`emit_line_loop` and :func:`emit_line_reduce` emit them.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import map_arg

from ...language import _decorators
from ...language import _tracing_ops
from ...language.creation_ops import full
from ...language.memory_ops import load
from ...language.memory_ops import store
from ..ast_extension import clone_ast
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..indexing_strategy import SubscriptIndexing
from ..inductor_lowering import ReductionLowering
from . import positional_rebind
from .fragment_mma import fragment_grid
from .fragment_mma import transposed
from .mpp_matmul import HOLD
from .mpp_matmul import LINE_FILL
from .mpp_matmul import LINE_REDUCE
from .mpp_matmul import SLOT_TYPE_GLOBAL_PREFIX
from .mpp_matmul import WRITES
from .mpp_matmul import Graphs
from .mpp_matmul import MppGroup
from .mpp_matmul import MppPlan
from .mpp_matmul import _aliases
from .mpp_matmul import _carry
from .mpp_matmul import _classify_uses
from .mpp_matmul import _insert
from .mpp_matmul import _pointwise
from .mpp_matmul import block_id_of
from .mpp_matmul import current_plan
from .mpp_matmul import metal_unsupported
from .mpp_matmul import op_dims
from .mpp_matmul import stage_update_vecs
from .mpp_matmul import static_tile
from .mpp_matmul import storage_key
from .mpp_matmul import virtual_thread
from .operand_elements import tile_mask

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from torch.fx.node import Node
    from torch.multiprocessing.reductions import StorageWeakRef

    from ..aten_lowering import LoweringContext
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import CodegenState
    from ..inductor_lowering import GraphInterpreter
    from .msl_ast_walker import EmitState

#: The reductions a line takes: each has an identity and an associative
#: combine that ``simd_shuffle_xor`` folds.
_REDUCTIONS = ("sum", "max", "min", "prod")
#: Each reduction's identity and combine, in MSL.
IDENTITY = {"sum": "0.0f", "max": "-INFINITY", "min": "INFINITY", "prod": "1.0f"}
#: ``a`` and ``b`` combined, as the Metal reductions combine them
#: (``msl_reduction``): max and min propagate a NaN.
COMBINE = {
    "sum": "({a} + {b})",
    "max": "::c10::metal::max({a}, {b})",
    "min": "::c10::metal::min({a}, {b})",
    "prod": "({a} * {b})",
}
#: Views that move a line's values as they are, adding or dropping size-1
#: dims (a subscript with ``None`` is ``mpp_matmul._pointwise``).
_LINE_MOVES = (
    _tracing_ops._new_var,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.dims,
    torch.ops.aten.unsqueeze.default,
)


@_decorators.api()
def _frag_hold(value: torch.Tensor) -> torch.Tensor:
    """An element value of a fragment result read by two element-level
    consumers, computed once into registers (:func:`plan_holds`)."""
    raise AssertionError("this should never be called")


@_decorators.api()
def _frag_line_seed(value: torch.Tensor) -> torch.Tensor:
    """The seed of a loop-carried line: ``value``, a constant fill or a
    value recomputed over the rows, written into the line's slots before
    the loop."""
    raise AssertionError("this should never be called")


class _Rows:
    """The lines of one M block: the groups whose grid rows lie over it,
    what may lie over a line besides the block (their batch blocks), and
    the storage the kernel writes (``MppPlan.written``)."""

    def __init__(
        self, block: int, groups: list[MppGroup], written: set[StorageWeakRef]
    ) -> None:
        self.block = block
        self.groups = groups
        self.batch = {g.batch_block for g in groups} - {None}
        self.written = written

    def over_rows(self, value: object) -> bool:
        """Whether ``value`` is a float32 tensor over the block (and batch
        and size-1 dims), which a line can hold."""
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            return False
        return self.over_block(value.size())

    def over_block(self, shape: Sequence[object]) -> bool:
        """Whether ``shape`` lies over the block (and batch and size-1
        dims): one element per row, which a line holds."""
        dims = {block_id_of(s) for s in shape if s != 1}
        return self.block in dims and dims <= {self.block, *self.batch}

    def recomputed(self, node: Node) -> bool:
        """Whether every lane computes ``node`` at any row by itself: a 0-d
        value, or one over the rows (and batch) that an element recomputes
        and that re-runs no load of a tensor the kernel writes."""
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor) or val.ndim == 0:
            return self.groups[0].recomputable(node)
        dims = {block_id_of(s) for s in val.size() if s != 1}
        return (
            dims <= {self.block, *self.batch}
            and self.groups[0].recomputable(node)
            and not self.rereads_written(node)
        )

    def rereads_written(self, value: object, group: MppGroup | None = None) -> bool:
        """Whether re-running ``value`` (as ``group`` recomputes it, the
        first group by default) re-runs a load of a tensor the kernel
        writes.  A line loop runs where the line is used, a seed before its
        loop and a line reduction's element loop at the reduction, on the
        lane holding each row, so such a load would move across the write
        and read on another thread than the one the Scalar program writes
        the element on, which no barrier orders (as
        ``mpp_matmul._order_memory`` keeps element-loop stores from
        re-running one)."""
        group = self.groups[0] if group is None else group
        return any(
            n.target is load and storage_key(n) in self.written
            for n in group.recomputable.nodes(value)
        )


def _line_reduction(node: Node, group: MppGroup, rows: _Rows) -> bool:
    """Whether ``node`` reduces a value of ``group`` over its N block into a
    line: a sum, max, min or prod whose result lies over the rows."""
    lowering = node.meta.get("lowering")
    return (
        isinstance(lowering, ReductionLowering)
        and lowering.block_index == group.n_block
        and lowering.reduction_type in _REDUCTIONS
        and rows.over_rows(node.meta.get("val"))
    )


def _seed_ok(arg: object, rows: _Rows) -> bool:
    """Whether a loop-carried line can start from ``arg``: a fill (an
    infinity too: a running max starts from -inf), or a value an element
    recomputes over the rows."""
    if not isinstance(arg, torch.fx.Node):
        return False
    if arg.target in (full, torch.ops.aten.full.default):
        fill = arg.args[1]
        return isinstance(fill, (int, float)) and not math.isnan(fill)
    return rows.over_rows(arg.meta.get("val")) and rows.recomputed(arg)


class _Carried:
    """A loop carry that may be a line: its placeholder, the loop, its
    argument, the getitem that carries it out and the phi after the loop."""

    def __init__(
        self, placeholder: Node, loop: Node, arg: Node, out: Node, phi: Node, slot: int
    ) -> None:
        self.placeholder = placeholder
        self.loop = loop
        self.arg = arg
        self.out = out
        self.phi = phi
        self.slot = slot

    def exit(self) -> object:
        (output,) = self.placeholder.graph.find_nodes(op="output")
        return output.args[0][self.slot]  # pyrefly: ignore[bad-index]


def _carries(graphs: Graphs, rows: _Rows) -> list[_Carried]:
    """The for-loop carries over ``rows`` whose seed a line can start from."""
    out: list[_Carried] = []
    for gid, (loop, args) in graphs.owners.items():
        if not _tracing_ops.is_for_loop_target(loop.target):
            continue
        body = graphs.info(gid).graph
        for i, p in enumerate(body.find_nodes(op="placeholder")):
            arg = args[i]
            getitem = _carry(p, loop)
            if (
                getitem is None
                or not isinstance(arg, torch.fx.Node)
                or not rows.over_rows(p.meta.get("val"))
                or not _seed_ok(arg, rows)
            ):
                continue
            phis = [
                u
                for u in getitem.users
                if u.target is _tracing_ops._phi and u.args[0] is arg
            ]
            if len(phis) != 1:
                continue
            slot = getitem.args[1]
            assert isinstance(slot, int)
            out.append(_Carried(p, loop, arg, getitem, phis[0], slot))
    return out


def _element_inputs(group: MppGroup, value: Node) -> list[Node]:
    """The inputs, other than ``group``'s element values, that an element
    loop computing ``value`` (one of them) re-runs."""
    out: list[Node] = []
    seen: set[Node] = set()
    work = [value]
    while work:
        n = work.pop()
        if n in seen or n not in group.derived:
            continue
        seen.add(n)
        for a in n.all_input_nodes:
            (work if a in group.derived else out).append(a)
    return out


class _LinePlanner:
    """The fixed point of :func:`plan_lines` for one M block."""

    def __init__(self, plan: MppPlan, graphs: Graphs, rows: _Rows) -> None:
        self.plan = plan
        self.graphs = graphs
        self.rows = rows
        self.lines: set[Node] = set()
        # Nodes pruned for a use no line may have: never lines again, so
        # the fixed point only shrinks once they are found.
        self.rejected: set[Node] = set()
        self.reds: dict[Node, MppGroup] = {}
        self.stores: set[Node] = set()
        self.carried = _carries(graphs, rows)
        # The kernel's loads and writes by storage: a line store must write
        # a tensor nothing else accesses.
        self.accesses: dict[object, list[Node]] = {}
        for info in graphs.infos.values():
            for n in info.graph.nodes:
                if n.target is load or n.target in WRITES:
                    self.accesses.setdefault(storage_key(n), []).append(n)

    def run(self) -> None:
        # Optimistic: every candidate carry is a line until a use or its
        # exit says otherwise.
        for c in self.carried:
            self.lines |= {*_aliases(c.placeholder), c.out, *_aliases(c.phi)}
        while True:
            before = (len(self.lines), sum(len(g.derived) for g in self.rows.groups))
            self._classify()
            self._propagate()
            self._prune()
            after = (len(self.lines), sum(len(g.derived) for g in self.rows.groups))
            if after == before:
                break
        self._classify()

    def _classify(self) -> None:
        """Re-run each group's use classification with its lines kept, and
        find its line reductions."""
        for g in self.rows.groups:
            g.lines = set(self.lines)
            g.derived = set()
            g.stores = []
            _classify_uses(g, self.graphs, masks=True)
            g.line_reds = set()
            for v in g.cur | g.derived:
                for u in v.users:
                    if u in self.rejected or (u in self.reds and self.reds[u] is not g):
                        continue
                    if _line_reduction(
                        u, g, self.rows
                    ) and not self.rows.rereads_written(_element_inputs(g, v), g):
                        g.line_reds.add(u)
                        self.reds[u] = g
                        self.lines.add(u)

    def _input_ok(self, a: object) -> bool:
        if not isinstance(a, torch.fx.Node):
            return True
        return a in self.lines or self.rows.recomputed(a)

    def _line_op(self, node: Node) -> bool:
        return (
            (_pointwise(node) or node.target in _LINE_MOVES)
            and self.rows.over_rows(node.meta.get("val"))
            and all(map(self._input_ok, node.all_input_nodes))
        )

    def _propagate(self) -> None:
        work = list(self.lines)
        while work:
            n = work.pop()
            for u in n.users:
                if (
                    u not in self.lines
                    and u not in self.rejected
                    and u not in self.reds
                    and self._line_op(u)
                ):
                    self.lines.add(u)
                    work.append(u)

    def _line_store(self, node: Node, value: Node) -> bool:
        """Whether ``node`` stores the line ``value`` as a line store writes
        it, one element per row (:func:`emit_line_loop`): its destination
        lies over the rows, not over a block the line is broadcast across,
        and it writes a tensor nothing else accesses, at an index and mask
        every lane recomputes.  A stack tensor's store, which Metal's store
        rejects, is none."""
        group = self.rows.groups[0]
        tensor, subscript = map_arg(node.args[:2], lambda a: a.meta["val"])
        assert isinstance(subscript, (list, tuple))
        return (
            node.args[2] is value
            and isinstance(tensor, torch.Tensor)
            and self.rows.over_block(
                SubscriptIndexing.compute_shape(tensor, list(subscript))
            )
            and all(map(group.recomputable, node.args[1]))  # pyrefly: ignore
            and group.recomputable(node.args[3])
            and not self.rows.rereads_written([node.args[1], node.args[3]])
            and self.accesses.get(storage_key(node)) == [node]
        )

    def _user_ok(self, n: Node, u: Node) -> bool:
        groups = self.rows.groups
        if u in self.lines or u.target is torch.ops.aten.sym_size.int:
            return True
        if any(u in g.derived or u in g.update for g in groups):
            return True
        if u.target is store and self._line_store(u, n):
            return True
        for c in self.carried:
            if c.placeholder not in self.lines:
                continue
            if u.op == "output" and c.exit() is n and n.graph is c.placeholder.graph:
                return True
        return False

    def _prune(self) -> None:
        """Drop each line with a use no line may have, a line op whose
        input is no longer one, and a carry any part of which isn't one:
        its value in the loop, its placeholder and their copies, and its
        value after the loop.  The carry's seed and its loop's output are
        one register line, so a part dropped to the Scalar layout (an
        after-loop use the planner rejects) would read the line's array as
        one value; dropped whole, the carry's uses fall back together."""
        changed = True
        while changed:
            changed = False
            for c in self.carried:
                carry = {*_aliases(c.placeholder), c.out, *_aliases(c.phi)}
                whole = carry <= self.lines and c.exit() in self.lines
                if not whole and carry & self.lines:
                    self.lines -= carry
                    changed = True
            for n in list(self.lines):
                # Reductions, carries and their plumbing are lines by what
                # they are; a line op by its inputs, which may have gone.
                plumbing = n in self.reds or n.target in (
                    _tracing_ops._phi,
                    operator.getitem,
                )
                carry = any(c.placeholder is n for c in self.carried)
                if not (plumbing or carry or self._line_op(n)):
                    self.lines.discard(n)
                    changed = True
                    continue
                if not all(self._user_ok(n, u) for u in n.users):
                    self.lines.discard(n)
                    self.reds.pop(n, None)
                    self.rejected.add(n)
                    changed = True
        self.stores = {
            u
            for n in self.lines
            for u in n.users
            if u.target is store and self._line_store(u, n)
        }


def plan_lines(plan: MppPlan, graphs: Graphs) -> None:
    """Find the fragment plan's lines (module docstring), seed its carried
    lines, and stage each update operand that isn't one
    (``mpp_matmul.stage_update_vecs``)."""
    by_block: dict[int, list[MppGroup]] = {}
    for g in plan.by_matmul.values():
        by_block.setdefault(g.m_block, []).append(g)
    for block, groups in by_block.items():
        planner = _LinePlanner(plan, graphs, _Rows(block, groups, plan.written))
        planner.run()
        for n in planner.lines:
            plan.lines[n] = block
        for n, g in planner.reds.items():
            if n in planner.lines:
                plan.line_reds[n] = g
        plan.line_stores |= planner.stores
        for c in planner.carried:
            if c.placeholder not in planner.lines:
                continue
            seed = _insert(_frag_line_seed, c.arg, before=c.loop)
            c.loop.replace_input_with(c.arg, seed)
            c.phi.replace_input_with(c.arg, seed)
            plan.line_seeds[seed] = block
        if planner.lines:
            positional_rebind.check_positional_merges(
                sorted(planner.lines, key=lambda n: n.name),
                "a fragment row value re-run at each row would read another row",
            )
    for g in plan.by_matmul.values():
        g.line_reds = {n for n in g.line_reds if plan.line_reds.get(n) is g}
        stage_update_vecs(g, graphs)
        for vec in g.vecs:
            plan.by_vec[vec] = g


def _from_carry(group: MppGroup) -> set[Node]:
    """The group's derived values computed from its K loop's carry before
    the op writes it (``early``)."""
    out: set[Node] = set()
    work = list(group.early)
    while work:
        n = work.pop()
        for u in n.users:
            if u in group.derived and u not in out:
                out.add(u)
                work.append(u)
    return out


def plan_holds(plan: MppPlan, graphs: Graphs) -> None:
    """Compute once each element value of a fragment group that two
    element-level consumers read (element loops, line reductions, operands
    built from the group's fragments): a ``_frag_hold`` node after it holds
    it in registers, a lane's elements of the group's accumulator, as the
    source program computes it once.  Re-running it in each consumer's
    element loop would repeat its work, an exponential in attention's
    softmax.  A value computed from the K loop's carry before the op writes
    it is left alone: its consumers run where the carry holds its old
    value."""
    for g in plan.by_matmul.values():
        consumers = g.derived | g.line_reds | set(g.stores)
        early = _from_carry(g)
        for d in sorted(g.derived, key=lambda n: n.name):
            users = [u for u in d.users if u in consumers]
            if len(users) < 2 or d in early:
                continue
            hold = _insert(_frag_hold, d)
            for u in users:
                u.replace_input_with(d, hold)
            g.derived.add(hold)
            plan.holds[hold] = g


# Codegen.


@_decorators.codegen(_frag_hold, "metal")
def _(state: CodegenState) -> ast.AST:
    """``_helion_mpp_hold(value)``, which ``elementize`` turns into the loop
    that writes the value at each element (:func:`emit_hold_loop`)."""
    return state.codegen.lift(
        expr_from_string(f"{HOLD}({{value}})", value=state.ast_arg(0)),
        prefix="_held",
    )


def line_rows(cg: GenerateAST, block: int) -> int:
    """The fragment rows ``R`` a lane holds a line of ``block`` in: the grid
    rows of every fragment group over it, which must agree and lie over M
    (an accumulator transposed for a computed rhs holds N in its rows)."""
    counts: set[int] = set()
    for g in current_plan().by_matmul.values():
        if g.m_block != block:
            continue
        if transposed(g):
            raise metal_unsupported(
                "a row value of a fragment accumulator transposed for its "
                "computed rhs, whose fragment rows hold N"
            )
        m, n, k = static_tile(cg, (g.m_block, g.n_block, g.k_block))
        rows, _, _ = fragment_grid(g, *op_dims(g, m, n), k)
        counts.add(rows)
    if len(counts) != 1:
        raise metal_unsupported(
            f"fragment row values over M tiles of {sorted(counts)} fragment rows"
        )
    (rows,) = counts
    return rows


def _line_layout(cg: GenerateAST, block: int) -> str:
    """A line marker's last arguments, ``rows, block_m, binds``: the
    fragment rows a lane holds the line in (:func:`line_rows`), the M
    block's size (a block of 1 grows to an 8-row fragment, whose rows past
    the block hold no row), and the bindings that make a lane act as the
    thread holding a row in the Scalar layout."""
    (block_m,) = static_tile(cg, (block,))
    binds = ", ".join(virtual_thread(cg, ((block, "_e_m"),)))
    return f"{line_rows(cg, block)}, {block_m}, [{binds}]"


@_decorators.codegen(_frag_line_seed, "metal")
def _(state: CodegenState) -> ast.AST:
    """``_helion_mpp_line_fill(value, rows, block_m, binds)``
    (:func:`_line_layout`): the seed, re-run per row slot under the row's
    thread bindings."""
    node = state.fx_node
    assert node is not None
    block = current_plan().line_seeds[node]
    cg = state.codegen
    return cg.lift(
        expr_from_string(
            f"{LINE_FILL}({{value}}, {_line_layout(cg, block)})",
            value=state.ast_arg(0),
        ),
        prefix="_line",
    )


def codegen_line_reduce(ctx: LoweringContext, node: Node) -> ast.AST | None:
    """``_helion_mpp_line_reduce(op, value, mask, rows, block_m, binds)``
    (:func:`_line_layout`) of a planned row reduction, or None for any
    other reduction.  ``mask`` names the N tile's mask when codegen masks
    it: an element past N's end holds a value of no row (its operand loads
    clamped), which the reduction replaces with its identity."""
    from ..generate_ast import GenerateAST

    plan = current_plan()
    group = plan.line_reds.get(node)
    if group is None:
        return None
    cg = ctx.cg
    assert isinstance(cg, GenerateAST)
    lowering = node.meta["lowering"]
    assert isinstance(lowering, ReductionLowering)
    (value_node,) = node.all_input_nodes
    value = ctx.env[value_node]
    assert isinstance(value, ast.Name)
    past = tile_mask(cg, group.n_block)
    mask = "None" if past is None else past[0]
    block = plan.lines[node]
    return cg.lift(
        expr_from_string(
            f"{LINE_REDUCE}({lowering.reduction_type!r}, {value.id}, {mask}, "
            f"{_line_layout(cg, block)})"
        ),
        prefix="_line",
    )


def codegen_slot(ctx: GraphInterpreter, node: Node) -> object:
    """``node``'s own lowering when the plan keeps it in registers (a line,
    its seed or a held value): its result named, and the name's Metal
    element type, from the node's dtype, recorded in a module global
    (``SLOT_TYPE_GLOBAL_PREFIX``) that MSL emission declares its register
    array with (``EmitState.slot_types``); NotImplemented for any other
    node.  A line op's lowering may assign temporaries from its line
    inputs (``log2`` as a log times a constant), each of which would be a
    line of no FX node, so they are substituted into its result: a line op
    is one statement."""
    from ..generate_ast import GenerateAST

    plan = current_plan()
    if (
        node not in plan.lines
        and node not in plan.line_seeds
        and node not in plan.holds
    ):
        return NotImplemented
    cg = ctx.cg
    assert isinstance(cg, GenerateAST)
    statements = cg.statements_stack[-1]
    start = len(statements)
    result = node.meta["lowering"].codegen(ctx, node)
    if node not in plan.line_reds and (_pointwise(node) or node.target in _LINE_MOVES):
        reads: set[str] = set()
        for a in node.all_input_nodes:
            value = ctx.env[a]
            if (a in plan.lines or a in plan.line_seeds) and isinstance(
                value, ast.Name
            ):
                reads.add(value.id)
        result = _inline_temporaries(statements, start, result, reads)
    if isinstance(result, ast.expr) and not isinstance(
        result, (ast.Name, ast.Constant)
    ):
        # Named as ``GraphInterpreter.run_node`` names a result.
        name = cg.device_function.new_var(node.name)
        cg.add_statement(statement_from_string(f"{name} = {{result}}", result=result))
        result = create(ast.Name, id=name, ctx=ast.Load())
    if isinstance(result, ast.Name):
        val = node.meta["val"]
        assert isinstance(val, torch.Tensor)
        kind = CompileEnvironment.current().backend.dtype_str(val.dtype)
        cg.device_function.codegen.module_statements.append(
            statement_from_string(f"{SLOT_TYPE_GLOBAL_PREFIX}{result.id} = {kind!r}")
        )
    return result


def _inline_temporaries(
    statements: list[ast.AST], start: int, result: object, reads: set[str]
) -> object:
    """``result`` with each statement from ``start`` on that assigns a
    temporary from a name in ``reads`` (or from such a temporary) removed
    and its value substituted where it is read; the statement that assigns
    ``result``'s own name stays."""
    values: dict[str, ast.expr] = {}
    own = result.id if isinstance(result, ast.Name) else None

    class _Substitute(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            value = values.get(node.id)
            return node if value is None else cast("ast.expr", clone_ast(value))

    kept: list[ast.AST] = []
    for stmt in statements[start:]:
        stmt = _Substitute().visit(stmt)
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(target := stmt.targets[0], ast.Name)
            and target.id != own
            and {n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)}
            & (reads | values.keys())
        ):
            values[target.id] = stmt.value
        else:
            kept.append(stmt)
    statements[start:] = kept
    if isinstance(result, ast.AST):
        return _Substitute().visit(result)
    return result


# MSL emission (msl_ast_walker).


def line_subs(state: EmitState, slot: str) -> dict[str, str]:
    """Each line's name read at ``slot``."""
    return {name: f"{name}[{slot}]" for name in state.lines}


def element_reads(state: EmitState, root: str, loop: ast.AST) -> dict[str, str]:
    """What ``loop``, an element loop over ``root``'s elements, reads in
    place at element ``_fi``: each line its body reads, at the element's
    row slot (lines of another fragment group's rows, which may lie over
    another block, are not its own), and each value held of the root at
    the element."""
    _, cols, flipped = state.fragments[root]
    reads = {n.id for n in ast.walk(loop) if isinstance(n, ast.Name)}
    lines = {
        name: sub
        for name, sub in line_subs(state, f"((_fi >> 1) / {cols})").items()
        if name in reads
    }
    # The planner keeps a transposed accumulator, whose fragment rows hold
    # N, from reading lines (``line_rows``).
    assert not (lines and flipped), (root, sorted(lines))
    held = {name: f"{name}[_fi]" for name, r in state.held.items() if r == root}
    return lines | held


def emit_hold_loop(
    stmt: ast.For,
    parts: list[str],
    indent: int,
    state: EmitState,
    emit_body: Callable[[list[ast.stmt], list[str], int, dict[str, str]], None],
) -> None:
    """A held value's loop (``mpp_matmul.elementize``):
    ``_helion_mpp_hold_loop(root, target)``.  ``target`` is declared as a
    lane's ``2 * rows * cols`` values of its type (``EmitState.slot_types``),
    one per element it holds of the root's fragments, and each element
    written at its slot ``_fi``, the slot every element loop over the root
    reads it at."""
    from .fragment_mma import fragment_elements

    assert isinstance(stmt.iter, ast.Call)
    root_arg, target_arg = stmt.iter.args
    assert isinstance(root_arg, ast.Name)
    target = ast.literal_eval(target_arg)
    root = state.coop[root_arg.id]
    rows, cols, _ = state.fragments[root]
    pad = " " * indent
    parts.append(f"{pad}{state.slot_types[target]} {target}[{2 * rows * cols}];")
    subs = {
        **element_reads(state, root, stmt),
        **{name: "_fv" for name, rt in state.coop.items() if rt == root},
    }

    def body(ind: int) -> list[str]:
        lines: list[str] = []
        saved = state.put
        state.put = f"{target}[_fi]"
        with state.isolated_scope():
            emit_body(stmt.body, lines, ind, subs)
        state.put = saved
        return lines

    parts.extend(fragment_elements(root_arg.id, root, pad, state, body))
    state.held[target] = root
    state.declare(target)


def emit_line_loop(
    stmt: ast.For,
    parts: list[str],
    indent: int,
    state: EmitState,
    emit_body: Callable[[list[ast.stmt], list[str], int, dict[str, str]], None],
) -> None:
    """A line loop (``mpp_matmul.elementize``): ``_helion_mpp_lines(rows,
    block_m, targets, store)`` over a lane's ``rows`` row slots, unrolled,
    each slot ``_ri`` a constant and its row in ``_e_m``.  Each target, a
    line, is declared before the loop and written at the slot; every line
    the body reads is read at the slot.  A target's slots take its type
    (``EmitState.slot_types``), a line's copy (a loop carry's snapshot) its
    source's.  A store's loop writes from the lane holding column 0 of its
    row, and only a row inside the M block when the fragment rows reach
    past it (a block of 1 grown to 8, as ``fragment_mma.fragment_elements``
    skips them): a row past the block holds no row's value (a line
    reduction's identity) and its store would write the block's own row,
    or another block's."""
    pad = " " * indent
    inner = pad + "    "
    assert isinstance(stmt.iter, ast.Call)
    rows, block_m, targets, is_store = (ast.literal_eval(a) for a in stmt.iter.args)
    writer = "_fn == 0"
    if 8 * rows > block_m:
        writer += f" && _e_m < {block_m}"
    *_, last = stmt.body
    for t in targets:
        if not state.is_declared(t):
            # A target's loop ends in its assignment.
            assert isinstance(last, ast.Assign)
            if t not in state.slot_types:
                assert isinstance(last.value, ast.Name), ast.unparse(last)
                state.slot_types[t] = state.slot_types[last.value.id]
            parts.append(f"{pad}{state.slot_types[t]} {t}[{rows}];")
            state.declare(t)
        state.lines[t] = rows
    from .msl_ast_walker import _ast_expr_to_msl

    for r in range(rows):
        lines = [
            f"{pad}{{",
            f"{inner}constexpr short _ri = {r};",
            f"{inner}int _e_m = 8 * _ri + _fm;",
        ]
        subs = line_subs(state, "_ri")
        *body, last = stmt.body
        with state.isolated_scope():
            state.declare("_e_m")
            emit_body(body, lines, indent + 4, subs)
            if isinstance(last, ast.Assign):
                (target,) = last.targets
                assert isinstance(target, ast.Name) and target.id in targets
                lines.append(
                    f"{inner}{target.id}[_ri] = {_ast_expr_to_msl(last.value, subs=subs)};"
                )
            else:
                assert is_store
                store_lines: list[str] = []
                emit_body([last], store_lines, indent + 8, subs)
                lines += [f"{inner}if ({writer}) {{", *store_lines, f"{inner}}}"]
        parts += [*lines, f"{pad}}}"]


def emit_line_reduce(
    stmt: ast.For,
    parts: list[str],
    indent: int,
    state: EmitState,
    emit_body: Callable[[list[ast.stmt], list[str], int, dict[str, str]], None],
) -> None:
    """A line reduction's loop (``mpp_matmul.elementize``):
    ``_helion_mpp_line_red(root, op, target)``.  Per row slot, the lane
    folds its elements of the row, ``2 * cols`` of them, each re-run by the
    body (whose ``_helion_mpp_line_acc`` folds the element's value into the
    slot's partial), then folds the row's four lanes, which differ in lane
    bits 0 and 3, by ``simd_shuffle_xor``."""
    pad = " " * indent
    inner = pad + "    "
    assert isinstance(stmt.iter, ast.Call)
    root_arg, op_arg, target_arg = stmt.iter.args
    assert isinstance(root_arg, ast.Name)
    op, target = ast.literal_eval(op_arg), ast.literal_eval(target_arg)
    root = state.coop[root_arg.id]
    rows, cols, flipped = state.fragments[root]
    assert not flipped
    block_m, block_n = state.coop_tile[root]
    rounded = root in state.coop_rounded
    parts.append(f"{pad}float {target}[{rows}];")
    combine = COMBINE[op]
    element = f"{root_arg.id}[_fi >> 1].thread_elements()[_fi & 1]"
    for r in range(rows):
        lines = [
            f"{pad}{{",
            f"{inner}float _lr = {IDENTITY[op]};",
        ]
        saved = state.line_acc
        state.line_acc = ("_lr", combine)
        subs = {
            **line_subs(state, str(r)),
            **{n: f"{n}[_fi]" for n, rt in state.held.items() if rt == root},
            **{name: "_fv" for name, rt in state.coop.items() if rt == root},
        }
        for c in range(cols):
            for e in range(2):
                fi = 2 * (r * cols + c) + e
                lines += [
                    f"{inner}{{",
                    f"{inner}    constexpr short _fi = {fi};",
                    f"{inner}    int _e_m = 8 * {r} + _fm;",
                    f"{inner}    int _e_n = 8 * {c} + _fn + {e};",
                ]
                body: list[str] = []
                with state.isolated_scope():
                    state.declare("_e_m")
                    state.declare("_e_n")
                    emit_body(stmt.body, body, indent + 8, subs)
                if rounded:
                    lines += [
                        f"{inner}    if (_e_m < {block_m} && _e_n < {block_n}) {{",
                        f"{inner}        float _fv = {element};",
                        *("    " + b for b in body),
                        f"{inner}    }}",
                    ]
                else:
                    lines += [f"{inner}    float _fv = {element};", *body]
                lines.append(f"{inner}}}")
        state.line_acc = saved
        for xor in (1, 8):
            lines.append(
                f"{inner}_lr = {combine.format(a='_lr', b=f'simd_shuffle_xor(_lr, {xor})')};"
            )
        lines += [f"{inner}{target}[{r}] = _lr;", f"{pad}}}"]
        parts += lines
    state.lines[target] = rows
    state.declare(target)
