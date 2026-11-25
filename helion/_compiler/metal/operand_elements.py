"""Operand elements a SIMD group's lanes compute in registers.

Two lowerings build a matmul's computed operand (a dequantized weight, a
widened activation) element by element in each lane's registers instead
of staging it through threadgroup memory: the fragment lowering
(``fragment_mma``, 8x8 ``simdgroup_matrix`` fragments) and MPP's register
operands (``mpp_registers``, ``matmul2d`` on cooperative tensors at
SIMD-group scope).  Each lane re-runs the Scalar statements that compute
an operand element with the thread and lanes that hold it in the Scalar
layout bound (``mpp_matmul.elementize``'s ``rerun``), then folds the
element's index arithmetic over the lane's position (``affine_fold``).
This module holds what both share:

- which computed operands a lane can recompute at any element
  (:class:`OperandRecomputable`);
- the codegen facts their elements fold over: tile offsets
  (:func:`offset_fact`), whole tiles (:func:`tiles_in_range`) and the M or
  N tile masks whose loads past the loop's end read the tile's first
  position (:func:`tile_mask`, :func:`clamp_past_end`);
- one element's statements, folded and tabulated (:func:`element_body`),
  over the kernel's constants (:func:`kernel_constants`).
"""

from __future__ import annotations

import ast
import collections
import math
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ...language import _tracing_ops
from ...language.creation_ops import full
from ...language.matmul_ops import dot
from ...language.memory_ops import load
from ..ast_read_writes import ast_rename
from ..inductor_lowering import ReductionLowering
from .affine_fold import Affine
from .affine_fold import Fact
from .affine_fold import fold
from .free_arange import live_graphs
from .lane_fold import load_mask_other
from .lane_fold import tl_load
from .mpp_matmul import ELEMENT_RUNS
from .mpp_matmul import WRITES
from .mpp_matmul import Graphs
from .mpp_matmul import MppGroup
from .mpp_matmul import MppPlan
from .mpp_matmul import Recomputable
from .mpp_matmul import assign_target
from .mpp_matmul import assigned_names
from .mpp_matmul import block_id_of
from .mpp_matmul import call_name
from .mpp_matmul import names_read
from .mpp_matmul import storage_key
from .tabulate import tabulate

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from torch.fx.node import Node

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from ..tile_strategy import LoopDimInfo
    from .msl_ast_walker import EmitState


#: Ops a computed operand's element re-runs as they are, besides the
#: element-wise ones: a permute, an alias and a same-width bitcast keep each
#: element on the thread that holds it, and a mask fill is a select on the
#: recomputed tile masks.
_ELEMENT_MOVES = (
    torch.ops.aten.permute.default,
    torch.ops.aten.alias.default,
    torch.ops.aten.view.dtype,
    _tracing_ops._mask_to,
)


class OperandRecomputable(Recomputable):
    """Whether a computed operand's value can be computed at any element of
    its tile, over its M or N block and K, by any lane."""

    def __init__(self, group: MppGroup, graphs: Graphs, blocks: set[int]) -> None:
        super().__init__(group, graphs)
        self._operand_blocks = blocks

    def blocks(self) -> set[int]:
        return self._operand_blocks

    def _compute(self, node: Node, val: object) -> bool:
        if node.target not in _ELEMENT_MOVES:
            return super()._compute(node, val)
        return all(map(self, node.all_input_nodes)) and (
            not isinstance(val, torch.Tensor)
            or all(s == 1 or block_id_of(s) in self.blocks() for s in val.size())
        )


#: A kernel whose SIMD groups split its K loops (fragment runs' and MPP's
#: register runs', :func:`split_refusal`): each thread's SIMD group in the
#: threadgroup, and the threadgroup buffer through which the SIMD groups
#: past the first pass their accumulators to it.
SIMDGROUP = "_frag_sg"
SPLIT_BUFFER = "_frag_split"


def split_refusal(plan: MppPlan, graphs: list[GraphInfo]) -> str | None:
    """Why the SIMD groups of a threadgroup can't split the K loops of the
    plan's groups (fragment runs, or MPP's register runs), or None when they
    can. Each SIMD group runs the kernel as one SIMD group would, over its
    share of each K loop's tiles, and after the loop the first adds the
    others' accumulators to its own; the others then hold zeros and write
    nothing. So a K loop may carry nothing but accumulators its runs alone
    update, and write nothing; an accumulator starts from a constant (the
    first SIMD group's; the others start from 0); the kernel reads no tensor
    it writes, which the SIMD groups past the first would read without
    having written it; and it has no reduction, whose threadgroup combine
    would take every SIMD group's threads as its own, nor an ``hl.dot`` the
    runs don't take, whose scalar contraction reduces so too."""
    known = Graphs(graphs)
    live = live_graphs(graphs)
    if any(
        isinstance(node.meta.get("lowering"), ReductionLowering)
        for info in live
        for node in info.graph.nodes
    ):
        return "the kernel has a reduction"
    if any(
        node not in plan.by_matmul
        for info in live
        for node in info.graph.find_nodes(op="call_function", target=dot)
    ):
        return (
            "the kernel has an hl.dot the fragment runs don't take, whose "
            "contraction reduces across threads"
        )
    loops: dict[Node, list[MppGroup]] = {}
    for group in plan.by_matmul.values():
        if not group.levels:
            # One run over the whole K, or over one K tile from zeros: no K
            # loop of its own to split.
            return "a matmul over a ':' K or one K tile's product"
        if group.seed is None or group.seed.target is not full:
            return "an accumulator starts from a tile"
        if group.update or group.upd is not None:
            return "the K loop updates an accumulator other than by its matmul"
        loops.setdefault(group.levels[0].owner, []).append(group)
    for groups in loops.values():
        body = groups[0].matmul.graph
        (output,) = body.find_nodes(op="output")
        accumulators = {group.matmul for group in groups}
        carried = output.args[0]
        assert isinstance(carried, (list, tuple))
        if any(value not in accumulators for value in carried):
            return "the K loop carries a value besides its matmuls' accumulators"
        if any(
            node.target in WRITES
            for info in live
            if known.within(info.graph, body)
            for node in info.graph.nodes
        ):
            return "the K loop writes memory"
    if any(
        node.target is load and storage_key(node) in plan.written
        for info in live
        for node in info.graph.nodes
    ):
        return "the kernel reads a tensor it writes"
    return None


def emit_split_adds(
    elements: Sequence[str],
    resets: Sequence[str],
    slot: Callable[[str, int], str],
    split: int,
    parts: list[str],
    pad: str,
) -> None:
    """After a K loop ``split`` SIMD groups split, add a lane's ``elements``
    of the others' accumulators into the first's: the others write each
    element to its ``slot`` (of a SIMD group's index from 0 for the second,
    and the element's) in threadgroup memory, then run ``resets``, each
    restarting an accumulator from 0, which keeps one an enclosing loop
    carries summed once; the first adds them in SIMD-group order, so a
    launch's sum is the same every time.  The first barrier keeps the writes
    after the first SIMD group's reads of an earlier combine, the second its
    reads after the writes."""
    inner = pad + "    "
    barrier = f"{pad}threadgroup_barrier(mem_flags::mem_threadgroup);"
    parts += [
        barrier,
        f"{pad}if ({SIMDGROUP} != 0) {{",
        *(
            f"{inner}{slot(f'({SIMDGROUP} - 1)', e)} = {element};"
            for e, element in enumerate(elements)
        ),
        *(f"{inner}{reset}" for reset in resets),
        f"{pad}}}",
        barrier,
        f"{pad}if ({SIMDGROUP} == 0) {{",
        f"{inner}for (uint _s = 0; _s < {split - 1}; _s++) {{",
        *(
            f"{inner}    {element} += {slot('_s', e)};"
            for e, element in enumerate(elements)
        ),
        f"{inner}}}",
        f"{pad}}}",
    ]


def split_range(start: str, step: str, simdgroups: int) -> tuple[str, str]:
    """A split K loop's start and step: each SIMD group starts at its own
    tile and steps over the others'."""

    def paren(text: str) -> str:
        return text if text.isidentifier() or text.isdigit() else f"({text})"

    first = f"{SIMDGROUP} * {paren(step)}"
    if start != "0":
        first = f"{paren(start)} + {first}"
    return first, f"{simdgroups} * {paren(step)}"


def static_int(expr: object) -> int | None:
    if isinstance(expr, int):
        return expr
    if isinstance(expr, sympy.Integer):
        return int(expr)
    return None


def static_loop(cg: GenerateAST, block: int) -> tuple[LoopDimInfo, int, int] | None:
    """The active loop over ``block``: its info, begin and block size, or
    None when no loop over it is active or its begin isn't known at
    codegen."""
    states = cg.active_device_loops.get(block)
    info = states[-1].block_id_to_info.get(block) if states else None
    if info is None:
        return None
    begin = (
        0
        if info.begin_expr is None and info.begin_var_name is None
        else static_int(info.begin_expr)
    )
    if begin is None:
        return None
    step = cg.device_function.resolved_block_size(block)
    assert isinstance(step, int)
    return info, begin, step


def tiles_in_range(cg: GenerateAST, block: int, op: int, extent: int | None) -> bool:
    """Whether ``offset + i < extent`` for every ``i < op`` at every tile
    offset the loop over ``block`` takes: its begin, end and ``extent`` are
    known at codegen and the last tile's ``op`` positions end by ``extent``."""
    loop = static_loop(cg, block)
    if loop is None or extent is None:
        return False
    info, begin, step = loop
    last = last_tile(begin, static_int(info.end_expr), step)
    return last is not None and last + op <= extent


def last_tile(begin: int, end: int | None, step: int) -> int | None:
    """The offset of a loop's last tile from ``begin`` to ``end`` by ``step``,
    or None when its end isn't known or it runs no tile."""
    if end is None or end <= begin:
        return None
    return begin + (end - begin - 1) // step * step


def tile_mask(cg: GenerateAST, block: int) -> tuple[str, str, str] | None:
    """The mask, index and offset names of the tile over ``block``, a
    computed operand's M or N, when codegen masks the tile; None when each
    of its positions lies inside its loop's range."""
    mask = cg.mask_var(block)
    if mask is None:
        return None
    return mask, cg.index_var(block), cg.offset_var(block)


def offset_fact(cg: GenerateAST, block: int) -> tuple[str, int, int, int | None] | None:
    """``(offset, multiple, least, most)`` of the loop over ``block`` when
    its begin is known: each tile's offset is the begin plus a multiple of
    the block size, at most the last tile's when the loop's end is known at
    codegen (else None)."""
    loop = static_loop(cg, block)
    if loop is None or loop[1] < 0:
        return None
    info, begin, step = loop
    most = last_tile(begin, static_int(info.end_expr), step)
    return cg.offset_var(block), math.gcd(step, begin), begin, most


def keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def parse_expr(text: str) -> ast.expr:
    """A plain ``ast`` expression: MSL assembly works on the kernel's parsed
    source (``metal_jit``), whose nodes it deep-copies."""
    return cast("ast.Expr", ast.parse(text).body[0]).value


def src(node: ast.AST) -> str:
    return f"({ast.unparse(node)})"


def _mask_terms(mask: ast.expr) -> list[ast.expr]:
    """The terms of a load mask's ``&`` (or ``and``) chain."""
    if isinstance(mask, ast.BinOp) and isinstance(mask.op, ast.BitAnd):
        return [*_mask_terms(mask.left), *_mask_terms(mask.right)]
    if isinstance(mask, ast.BoolOp) and isinstance(mask.op, ast.And):
        return [t for v in mask.values for t in _mask_terms(v)]
    return [mask]


def _position_reads(
    binds: Mapping[str, ast.expr], position: str
) -> tuple[set[str], set[int]]:
    """The names among an element loop's ``binds`` that hold the element's
    ``position`` (``_e_m`` or ``_e_n``): its block's lane variables, and the
    components of ``tid`` along its block's axis."""
    lanes = {n for n, e in binds.items() if n != "tid" and position in names_read(e)}
    axes: set[int] = set()
    tid = binds.get("tid")
    if isinstance(tid, ast.Call):
        axes = {j for j, a in enumerate(tid.args) if position in names_read(a)}
    return lanes, axes


def _reads_any(node: ast.AST, names: set[str], axes: set[int]) -> bool:
    """Whether ``node`` reads one of ``names`` or a ``tid`` component in
    ``axes`` (any component, when it reads ``tid`` other than by one)."""
    tids = picked = 0
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id in names:
                return True
            tids += n.id == "tid"
        elif (
            isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name)
            and n.value.id == "tid"
            and isinstance(n.slice, ast.Constant)
        ):
            picked += 1
            if n.slice.value in axes:
                return True
    return bool(axes) and tids > picked


def clamp_past_end(
    stmts: list[ast.stmt],
    mask: str,
    index: str,
    offset: str,
    binds: Mapping[str, ast.expr],
    position: str,
) -> list[ast.stmt]:
    """A computed operand's element statements ``stmts`` with each load
    that its M or N tile's ``mask`` guards reading, where the mask is off,
    at the tile's first position ``offset`` instead of 0: such an element
    reaches only accumulator rows or columns past the tile's loop end (or
    past the tile, in a fragment wider than it), which the element loops
    never store, since they re-run the store's ``mask``.  Its value is
    free, so only its address must lie inside the tensor.  The load keeps
    its other mask terms: a K tail still reads 0.

    Wherever its other terms admit it, the clamped load reads the address
    the kernel reads at the tile's first position, which lies inside the
    loop's range at every tile the loop runs.  That holds only while the
    element's ``position`` reaches the address through ``index`` alone,
    read clamped, and reaches none of the other terms, which are evaluated
    at the element.  So a load keeps its mask when its address or another
    term reads a name computed from the position: from ``index`` or
    ``mask``, or from the block's lane or ``tid`` component in ``binds``.
    Such an address (``tile.index // 2``) would be left neither masked nor
    clamped, and such a term (``tile.index < x.size(0)``) can admit an
    element past the end while it excludes the tile's first position,
    whose address the kernel then never reads."""
    assigned = {assign_target(s) for s in stmts}
    if mask not in assigned or index not in assigned:
        # A mask this thread holds as it stands, not the element's.
        return stmts
    lanes, axes = _position_reads(binds, position)
    derived = {*lanes, index, mask}
    for s in stmts:
        t = assign_target(s)
        if t is None or t in (index, mask):
            continue
        assert isinstance(s, ast.Assign)
        if _reads_any(s.value, derived, axes):
            derived.add(t)
    clamped = f"_clamped_{index}"
    hits = 0

    class Clamp(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            nonlocal hits
            self.generic_visit(node)
            load = tl_load(node)
            masked = None if load is None else load_mask_other(load)
            if masked is None or _reads_any(node.args[0], derived - {index}, axes):
                return node
            terms = _mask_terms(masked[0])
            kept = [t for t in terms if not (isinstance(t, ast.Name) and t.id == mask)]
            if len(kept) == len(terms) or any(
                _reads_any(t, derived, axes) for t in kept
            ):
                return node
            hits += 1
            ptr = ast_rename(node.args[0], {index: clamped})
            if not kept:
                return ast.Call(node.func, [ptr, ast.Constant(None)], [])
            rest = kept[0]
            for term in kept[1:]:
                rest = ast.BinOp(rest, ast.BitAnd(), term)
            return ast.Call(node.func, [ptr, rest, *node.args[2:]], node.keywords)

    body = [Clamp().visit(s) for s in stmts]
    if not hits:
        return stmts
    out: list[ast.stmt] = []
    for s in body:
        out.append(s)
        if assign_target(s) == mask:
            out.append(
                ast.Assign(
                    [ast.Name(clamped, ast.Store())],
                    ast.IfExp(
                        ast.Name(mask, ast.Load()),
                        ast.Name(index, ast.Load()),
                        ast.Name(offset, ast.Load()),
                    ),
                )
            )
    for s in out:
        ast.fix_missing_locations(s)
    return out


#: An element's statements, its names' affine values and the facts they fold
#: over (:func:`element_body`).
ElementBody = tuple[list[ast.stmt], dict[str, object], dict[str, Fact]]


def element_body(
    loop: ast.For,
    decls: tuple[str, ...],
    step: int | Affine,
    i: int,
    facts: dict[str, Fact],
    tables: int,
    state: EmitState,
) -> ElementBody:
    """Element ``i`` of step ``step``'s statements (a constant, or the value
    of a rolled run's step over its chunk index), its position bound and
    folded (``affine_fold``) over the run's facts and the kernel's integer
    constants, and its decodes tabulated (``tabulate``) under the cap
    ``tables``; the affine values of its names; and the facts they fold
    over, the run's and its floor terms'."""
    env: dict[str, object] = {
        name: Affine.of(value)
        for name, value in state.constants.items()
        if isinstance(value, int)
    }
    env |= {
        "_fs": step if isinstance(step, Affine) else Affine.of(step),
        "_fi": Affine.of(i),
    }
    facts = dict(facts)
    positions = ast.parse("\n".join(decls)).body
    body = fold([*positions, *loop.body], env, facts)[len(positions) :]
    return tabulate(body, state.constants, state.tables, tables), env, facts


def kernel_constants(body: list[ast.stmt]) -> dict[str, int | float]:
    """The kernel's constants a run of register elements folds over: names
    assigned once in the whole kernel, an integer or float literal or the
    ``static_cast(int, ...)`` of an integer one that int's 32 bits hold (a
    scalar the host fixed, such as a module-level int, reaches an index as
    ``v = static_cast(int, symnode)``; a chain of casts resolves one link per
    pass).  A name assigned anywhere else too (``mpp_matmul.assigned_names``)
    is not one, whatever the control flow; a run's own statements are folded
    per element instead."""
    counts: collections.Counter[str] = collections.Counter()
    literals: dict[str, int | float] = {}
    casts: dict[str, ast.expr] = {}
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.For) and call_name(node.iter) in ELEMENT_RUNS:
            continue
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name, value = node.targets[0].id, node.value
            if (
                isinstance(value, ast.Constant)
                and isinstance(value.value, (int, float))
                and not isinstance(value.value, bool)
            ):
                literals[name] = value.value
            elif (
                isinstance(value, ast.Call)
                and call_name(value) == "static_cast"
                and len(value.args) == 2
                and isinstance(value.args[0], ast.Name)
                and value.args[0].id == "int"
            ):
                casts[name] = value.args[1]
        counts.update(assigned_names(node))
        stack.extend(ast.iter_child_nodes(node))
    constants = {name: v for name, v in literals.items() if counts[name] == 1}

    def cast_of(arg: ast.expr) -> object:
        if isinstance(arg, ast.Constant):
            return arg.value
        return constants.get(arg.id) if isinstance(arg, ast.Name) else None

    pending = {name: arg for name, arg in casts.items() if counts[name] == 1}
    while resolved := {
        name: const
        for name, arg in pending.items()
        if type(const := cast_of(arg)) is int and -(2**31) <= const < 2**31
    }:
        constants |= resolved
        pending = {name: arg for name, arg in pending.items() if name not in resolved}
    return constants
