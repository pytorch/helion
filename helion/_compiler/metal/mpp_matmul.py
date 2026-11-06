"""Cooperative MPP lowering of a K-loop matmul accumulator.

A matmul that updates its K loop's own carry, a ``torch.addmm``, a
``torch.baddbmm`` or an ``hl.dot``, keeps that carry in MPP's layout, as one
``cooperative_tensor``, across the loop.  A matmul over a ``:`` K, with an
acc or without one (``torch.mm``, ``torch.bmm``, ``hl.dot``), is one op over
the whole K, its result a cooperative value too.  An ``hl.dot`` over a K
held as per-thread parts (``factored_dims``) has no other lowering; the
kernel's other ``hl.dot`` calls reach the planner only when it can lower every
one of them (:func:`_dot_admissible`), and keep the scalar contraction
(``matmul_ops``) otherwise.  A batched matmul is one rank-2 op on
the matrices of its batch tile's one index:

- :func:`plan_mpp_matmuls` (``MetalBackend.pre_codegen``) admits each such
  matmul and raises ``BackendUnsupported`` for every form it can't lower.
  An operand is a device view when it is a load, or its transpose, of a
  tensor the kernel doesn't write, with a unit-stride dim, whose two dims
  are tiles, offset tiles, ``hl.arange`` indices or ``:`` and whose other
  dims take uniform indices (:func:`_view`); it is staged otherwise: the
  threads that hold it write it into a threadgroup tile.  Each use of the
  cooperative value that an element loop can evaluate at any element stays
  cooperative; every other use reads it through a materialization
  (``_mpp_mat``), and a seed that is not a constant fill reaches the K loop
  through ``_mpp_coop``;
- the seed's ``full`` or ``_mpp_coop`` emits ``_helion_mpp_acc(...)`` and the
  matmul emits ``_helion_mpp_run(acc, ...)``, one ``matmul2d.run`` per K tile (or
  over the whole ``:`` K) on views rebased at the tile in 64 bits or on
  staged tiles;
- :func:`elementize` (MSL assembly) moves each store of the cooperative
  value into a loop over the tensor's own elements, which re-runs the
  Scalar epilogue, index and mask statements with the thread that owns the
  element in the Scalar layout bound in place of ``tid``;
- a lane loop around a cooperative op is split at the op when only
  register and staging work sits beside it, and otherwise jammed across
  its lanes (``lane_fold.LaneJam``): Scalar work runs once per lane in body
  order, each cooperative statement once; a lane loop of staging work that
  only one thread of a row would run deals its lanes out over that row's
  threads (:class:`_Replicas`);
- an element-wise update of the K loop's carry before the op
  (``acc * alpha``) runs in place on the cooperative tensor in an element
  loop (``_mpp_upd``); an operand of it that no element can recompute, or
  such a row or column vector read by a stored epilogue (``acc / l_i``), is
  staged into a threadgroup tile (``_mpp_vec``);
- a staged operand its outermost loop doesn't change is staged once, before
  that loop (``_mpp_hoist``); a bfloat operand staged inside a loop from
  values the loop doesn't reload goes into a float32 tile, each float ->
  bfloat conversion on its way rounded in integer arithmetic
  (``_mpp_round``).

MPP chooses which thread holds each element, so the element loop's threads
diverge and differ from the Scalar layout's.  Three rules keep it exact:
every name the kernel reassigns is read through a copy taken where the
Scalar statement read it (:class:`_Copies`); a statement that exchanges
data between threads is never re-run, and its target is read by name
(:func:`_cross_thread`), which the planner makes sound by admitting such a
value only when it is 0-d; and every node the element loop re-runs meets
its operands' dims by block (``positional_rebind``).
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import math
import operator
from typing import TYPE_CHECKING
from typing import TypeGuard
from typing import cast

import torch
from torch._inductor.codegen.mps import DTYPE_TO_METAL
from torch._inductor.runtime.runtime_utils import next_power_of_2
from torch.fx.node import Node
from torch.multiprocessing.reductions import StorageWeakRef

from ... import exc
from ...language import _decorators
from ...language import _tracing_ops
from ...language import atomic_ops
from ...language import view_ops
from ...language.creation_ops import full
from ...language.matmul_ops import dot
from ...language.memory_ops import load
from ...language.memory_ops import store
from ...language.tile_ops import tile_index
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..aten_lowering import addmm_lowering
from ..aten_lowering import baddbmm_lowering
from ..aten_lowering import bmm_lowering
from ..aten_lowering import mm_lowering
from ..compile_environment import CompileEnvironment
from ..compile_environment import FixedBlockSizeSource
from ..host_function import HostFunction
from ..inductor_lowering import APIFuncLowering
from ..variable_origin import BlockSizeOrigin
from ..variable_origin import GridOrigin
from ..variable_origin import TileBeginOrigin
from ..variable_origin import TileIdOrigin
from . import positional_rebind
from .factored_dims import FactoredValue
from .factored_dims import factored_size
from .factored_dims import widened_iota
from .free_arange import live_graphs
from .lane_fold import BF16_CAST
from .lane_fold import LANE_FOLD
from .lane_fold import MAX_LANE_COPIES
from .lane_fold import LaneArrayBudget
from .lane_fold import LaneJam
from .lane_fold import const_extent
from .lane_fold import folded_lanes
from .msl_reduction import REDUCTION_NAMESPACE
from .reduction import alloc_threadgroup_buffer

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence

    from torch.fx.graph import Graph

    from ..aten_lowering import LoweringContext
    from ..device_ir import GraphInfo
    from ..device_ir import IfGraphInfo
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import CodegenState

ACC = "_helion_mpp_acc"
#: Position of ``_helion_mpp_acc``'s seed argument: the name of its staged seed tile,
#: as a string, or None (``_acc_expr``).
ACC_SEED = 13
RUN = "_helion_mpp_run"
ELEMENTS = "_helion_mpp_elements"
BIND = "_helion_mpp_bind"
STAGE = "_helion_mpp_stage"
MAT = "_helion_mpp_materialize"
#: A barrier that makes this threadgroup's earlier device writes visible.
FENCE = "_helion_mpp_fence"
#: A threadgroup barrier between two tiles that share storage (tile_sharing).
BARRIER = "_helion_mpp_barrier"
#: Name prefix of a materialization's threadgroup tile.
MAT_TILE = "_mpp_mat"
#: An in-place element-wise update of a cooperative accumulator.
UPDATE = "_helion_mpp_update"
#: The element loop's write of an updated element.
PUT = "_helion_mpp_put"
#: Name prefix of a staged per-element operand's threadgroup tile.
VEC_TILE = "_mpp_vec"
#: A float rounded to the nearest bfloat, ties to even, kept as a float.
ROUND = "helion_bf16_rne"
#: Torch's float -> bfloat rule in integer arithmetic: a NaN stays a NaN
#: (tested on the bits, which fast math keeps), and bfloat subnormals stay,
#: which Metal's own cast flushes.
ROUND_PREAMBLE = f"""inline float {ROUND}(float v) {{
    uint u = as_type<uint>(v);
    uint r = (u & 0x7FFFFFFFu) > 0x7F800000u ? (u | 0x00400000u) : u + 0x7FFFu + ((u >> 16) & 1u);
    return as_type<float>(r & 0xFFFF0000u);
}}"""
BF16_CAST_PREAMBLE = f"""inline bfloat {BF16_CAST}(float v) {{
    return as_type<bfloat>(ushort(as_type<uint>({ROUND}(v)) >> 16));
}}"""

#: Each matmul's (acc, lhs, rhs) argument positions; None for no acc.  The
#: lowering of ``aten.bmm.dtype`` drops its dtype argument and lowers it as
#: ``aten.bmm``; its node keeps its float32 result.
_MATMULS: dict[object, tuple[int | None, int, int]] = {
    torch.ops.aten.addmm.default: (0, 1, 2),
    dot: (2, 0, 1),
    torch.ops.aten.mm.default: (None, 0, 1),
    torch.ops.aten.bmm.default: (None, 0, 1),
    torch.ops.aten.bmm.dtype: (None, 0, 1),
    torch.ops.aten.baddbmm.default: (0, 1, 2),
}
_BATCHED = (
    torch.ops.aten.bmm.default,
    torch.ops.aten.bmm.dtype,
    torch.ops.aten.baddbmm.default,
)
_IOTA = torch.ops.prims.iota.default
_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_INT_DTYPES = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
_WRITES = (
    store,
    atomic_ops.atomic_add,
    atomic_ops.atomic_xchg,
    atomic_ops.atomic_and,
    atomic_ops.atomic_or,
    atomic_ops.atomic_xor,
    atomic_ops.atomic_max,
    atomic_ops.atomic_min,
    atomic_ops.atomic_cas,
)

#: One ``matmul2d`` op's hardware-validated tile range (M, N, K).
_MAX_TILE = (128, 128, 256)
#: M or N up to this when the op's scope has ``_WIDE_MIN_SIMDGROUPS`` or more.
#: Measured on an M4 Pro: a wide M or N miscomputes on one SIMD group and a
#: wide N on two, while device views, staged and transposed operands, loaded
#: seeds, f16, bf16 and f32, M, N and K tails and K 16 to 256 are exact on
#: four and eight.
_WIDE_TILE = 256
_WIDE_MIN_SIMDGROUPS = 4

_REBIND = (
    "an MPP epilogue re-run at each cooperative element would read another element"
)
_STAGE_REBIND = (
    "a staged MPP operand is written into its tile by block, so it would hold "
    "another element"
)
_MAT_REBIND = (
    "a value read from an MPP result in the Scalar layout would hold another element"
)


@_decorators.api()
def _mpp_mat(value: torch.Tensor) -> torch.Tensor:
    """A cooperative value read in the Scalar layout: the cooperative tensor
    is written to a threadgroup tile and each thread reads its elements."""
    raise AssertionError("this should never be called")


@_decorators.api()
def _mpp_coop(value: torch.Tensor) -> torch.Tensor:
    """A Scalar value that seeds a cooperative accumulator: the threads that
    hold it stage it into a threadgroup tile, which fills the tensor."""
    raise AssertionError("this should never be called")


@_decorators.api()
def _mpp_upd(carry: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """``value``, an element-wise update of the cooperative ``carry``,
    written back into the carry's own elements by an element loop."""
    raise AssertionError("this should never be called")


@_decorators.api()
def _mpp_hoist(value: torch.Tensor) -> torch.Tensor:
    """A staged operand that its loop doesn't change, computed before the
    loop: the threads that hold it stage it there once, into the tile the
    op reads in every iteration."""
    raise AssertionError("this should never be called")


@_decorators.api()
def _mpp_vec(value: torch.Tensor) -> torch.Tensor:
    """A Scalar value over some of a group's M, N and batch blocks that an
    element loop reads at any element: the threads that hold it stage it
    into a threadgroup tile, which the loop reads at the element."""
    raise AssertionError("this should never be called")


@_decorators.api()
def _mpp_round(value: torch.Tensor) -> torch.Tensor:
    """``value.to(torch.bfloat16)`` held as a float (:func:`_stage_in_float`);
    its node keeps the conversion's bfloat meta."""
    raise AssertionError("this should never be called")


class _NotAView(Exception):
    """An operand that is not a device view; staging takes it."""


def _unsupported(msg: str) -> exc.BackendUnsupported:
    return exc.BackendUnsupported("metal", msg)


@dataclasses.dataclass
class _Dim:
    block_id: int
    #: The ``hl.arange`` node of a ``base + hl.arange`` index; None for a tile.
    iota: Node | None = None
    #: The uniform base added to the iota, or to the tile's index (``base +
    #: tile.index``, with ``iota`` None); None for no base.
    base: Node | int | None = None
    #: The load's index node on this dim (the iota, or the ``base + ...``).
    index: Node | None = None
    #: The bound of a mask ``index < bound``; None for no mask.
    bound: Node | int | None = None
    #: The mask's comparison node.
    mask: Node | None = None
    #: A widened ``hl.arange`` (``factored_dims.widened_iota``): the dim has
    #: length ``factor * block`` from ``factor * tile_begin(block)``.
    factor: int = 1
    #: A ``:`` slice: origin 0, extent the size.
    full: bool = False


@dataclasses.dataclass
class _View:
    tensor: torch.Tensor
    dims: tuple[_Dim, _Dim]
    #: The tensor dims the op's two dims read.
    positions: tuple[int, int]
    #: ``(tensor dim, index)`` of every other dim, each a uniform integer.
    outer: tuple[tuple[int, Node | int], ...]
    #: ``(tensor dim, block)`` of a batched matmul's batch tile, whose one
    #: index at codegen is its offset.
    batch: tuple[int, int] | None = None


@dataclasses.dataclass
class _Operand:
    """A matmul operand: a device view, or, with no view, a value the Scalar
    layout computes, which the threads that hold it write into a threadgroup
    tile before the op (staging)."""

    dtype: torch.dtype
    view: _View | None = None
    #: The ``_mpp_hoist`` node that stages it before its loop; None to stage
    #: it at the op.
    hoist: Node | None = None
    #: The hoisted tile, set by the ``_mpp_hoist`` codegen.
    tile: str | None = None
    #: A bfloat operand staged into a float32 tile (:func:`_stage_in_float`).
    float_tile: bool = False
    #: The group from whose element loop the operand is staged
    #: (:func:`_link_element_staging`); None to stage it from the Scalar layout.
    elements: int | None = None

    @property
    def staged(self) -> bool:
        return self.view is None


@dataclasses.dataclass
class MppGroup:
    gid: int
    matmul: Node
    #: The node that emits the cooperative tensor: a constant fill or a
    #: ``_mpp_coop``; None for a matmul with no acc, whose op emits it.
    seed: Node | None
    m_block: int
    n_block: int
    k_block: int
    lhs: _Operand
    rhs: _Operand
    #: Stores of the cooperative value; each becomes an element loop.
    stores: list[Node] = dataclasses.field(default_factory=list)
    #: K's length is ``k_factor`` x its block's: a K held as per-thread parts.
    k_factor: int = 1
    #: The control-flow levels the accumulator's storage crosses, the K loop
    #: first.
    levels: list[_Level] = dataclasses.field(default_factory=list)
    #: Values the op updates in place: a reader other than the update needs a
    #: snapshot taken where the value is defined.
    old: set[Node] = dataclasses.field(default_factory=set)
    #: The current values: the op's result and each level's exit.
    cur: set[Node] = dataclasses.field(default_factory=set)
    #: ``(value, user)`` pairs that pass the storage along.
    plumbing: set[tuple[Node, Node]] = dataclasses.field(default_factory=set)
    #: Element-wise values an element loop computes from the result.
    derived: set[Node] = dataclasses.field(default_factory=set)
    #: Reads of a cooperative value in the Scalar layout, each through a
    #: materialization taken right after the value's definition.
    mats: dict[Node, Node] = dataclasses.field(default_factory=dict)
    #: The values an element loop may compute at any element.
    recomputable: _Recomputable = dataclasses.field(init=False)
    #: A batched matmul's batch block (``bmm``, ``baddbmm``): one rank-2 op per
    #: batch index, so the block must hold one index at codegen.  None for a
    #: rank-2 matmul or a batch dim of size 1.
    batch_block: int | None = None
    #: The element-wise nodes that compute the acc from the K loop's carry
    #: before the op, in graph order; an element loop runs them in place.
    update: list[Node] = dataclasses.field(default_factory=list)
    #: The ``_mpp_upd`` node that writes the update back.
    upd: Node | None = None
    #: The ``_mpp_vec`` nodes that stage the update's Scalar operands.
    vecs: list[Node] = dataclasses.field(default_factory=list)
    #: The K loop's carry and its copies, which an element loop may read
    #: before the update or the op writes them (:func:`_before_write`).
    early: set[Node] = dataclasses.field(default_factory=set)
    #: Values staged as other groups' operands from this group's element
    #: loop, each to those matmuls (:func:`_link_element_staging`).
    stage_uses: dict[Node, set[Node]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _Level:
    """A control-flow level the accumulator's storage crosses: ``owner``
    takes ``arg`` in as ``placeholder`` and gives the updated value back as
    ``exit``, which ``phi`` merges with ``arg`` after it."""

    owner: Node
    arg: Node
    placeholder: Node
    exit: Node
    phi: Node
    #: The owner's graph output slot that holds the updated value.
    slot: int


@dataclasses.dataclass
class MppPlan:
    by_matmul: dict[Node, MppGroup] = dataclasses.field(default_factory=dict)
    by_seed: dict[Node, MppGroup] = dataclasses.field(default_factory=dict)
    by_mat: dict[Node, MppGroup] = dataclasses.field(default_factory=dict)
    by_vec: dict[Node, MppGroup] = dataclasses.field(default_factory=dict)
    #: Each ``_mpp_hoist`` node's group and operand side (0 lhs, 1 rhs).
    by_hoist: dict[Node, tuple[MppGroup, int]] = dataclasses.field(default_factory=dict)
    #: Loads of a staged operand that read a tensor the kernel writes: each
    #: runs after a device-memory barrier.
    fenced: set[Node] = dataclasses.field(default_factory=set)
    #: Each op's (M, N) when either is over ``_MAX_TILE``'s, checked against
    #: the launch's SIMD groups (``check_mpp_launch``).
    wide: dict[int, tuple[int, int]] = dataclasses.field(default_factory=dict)


def _node(value: object) -> Node:
    assert isinstance(value, Node)
    return value


def _seq(value: object) -> Sequence[object]:
    assert isinstance(value, (list, tuple))
    return value


def _operands(matmul: Node) -> tuple[Node | None, Node, Node]:
    """A matmul's acc (None for none), lhs and rhs nodes."""
    acc, lhs, rhs = _MATMULS[matmul.target]
    return (
        None if acc is None or matmul.args[acc] is None else _node(matmul.args[acc]),
        _node(matmul.args[lhs]),
        _node(matmul.args[rhs]),
    )


class _Graphs:
    """Follows values across the control-flow graphs of one kernel."""

    def __init__(self, graphs: list[GraphInfo]) -> None:
        self.graph_id = {g.graph: g.graph_id for g in graphs}
        self.infos = {g.graph_id: g for g in graphs}
        #: Each while loop's condition and body graphs, to the loop.
        self.whiles: dict[object, Node] = {}
        #: Each control-flow graph's owner node and the arguments it passes in.
        self.owners: dict[object, tuple[Node, Sequence[object]]] = {}
        for g in graphs:
            for n in g.graph.nodes:
                if _tracing_ops.is_for_loop_target(n.target):
                    self.owners[n.args[0]] = (n, _seq(n.args[3]))
                elif n.target is _tracing_ops._if:
                    self.owners[n.args[1]] = (n, _seq(n.args[3]))
                    self.owners[n.args[2]] = (n, _seq(n.args[4]))
                elif n.target is _tracing_ops._while_loop:
                    self.whiles[n.args[0]] = self.whiles[n.args[1]] = n

    def info(self, gid: object) -> GraphInfo:
        """The graph a control-flow node's graph-id argument names."""
        assert isinstance(gid, int)
        return self.infos[gid]

    def if_branch(self, gid: object) -> tuple[IfGraphInfo, list[tuple[int | str, ...]]]:
        """An if branch graph's info and its ``branches_outputs``: per output
        of the ``_if``, its slot in each branch graph's output, or the name an
        else branch leaves as it was."""
        from ..device_ir import IfGraphInfo

        info = self.info(gid)
        assert isinstance(info, IfGraphInfo) and info.branches_outputs is not None
        return info, info.branches_outputs

    def owner(self, node: Node) -> Node | None:
        entry = self.owners.get(self.graph_id[node.graph])
        return entry[0] if entry is not None else None

    def within(self, graph: Graph, outer: Graph) -> bool:
        """Whether ``graph`` is ``outer`` or runs inside control flow in it."""
        while graph is not outer:
            gid = self.graph_id[graph]
            node = self.owners[gid][0] if gid in self.owners else self.whiles.get(gid)
            if node is None:
                return False
            graph = node.graph
        return True

    def outer(self, placeholder: Node) -> object:
        """The value the enclosing graph passes in for ``placeholder``."""
        _, args = self.owners[self.graph_id[placeholder.graph]]
        return args[_placeholder_index(placeholder)]

    def origin(self, node: Node) -> Node:
        """Follow ``_new_var`` and placeholders out to the defining node.

        A placeholder that its loop carries holds the passed-in value only on
        the first iteration, so the walk stops there.
        """
        while True:
            if node.target is _tracing_ops._new_var:
                node = _node(node.args[0])
                continue
            owner = self.owner(node) if node.op == "placeholder" else None
            if owner is None:
                return node
            if (
                _tracing_ops.is_for_loop_target(owner.target)
                and _carry(node, owner) is not None
            ):
                return node
            outer = self.outer(node)
            if not isinstance(outer, Node):
                return node
            node = outer


def _placeholder_index(placeholder: Node) -> int:
    return list(placeholder.graph.find_nodes(op="placeholder")).index(placeholder)


def _carry(placeholder: Node, loop: Node) -> Node | None:
    """The getitem of ``loop`` that carries ``placeholder`` into the next
    iteration, or None for a read-only input: the loop's argument for a
    carried name meets its output in a ``_phi`` after the loop."""
    arg = _seq(loop.args[3])[_placeholder_index(placeholder)]
    if not isinstance(arg, Node):
        return None
    for user in arg.users:
        if user.target is _tracing_ops._phi and user.args[0] is arg:
            out = user.args[1]
            if (
                isinstance(out, Node)
                and out.target is operator.getitem
                and out.args[0] is loop
            ):
                return out
    return None


def _block(size: object) -> int | None:
    return CompileEnvironment.current().resolve_block_id(size)


def _dim_of(size: object) -> tuple[int, int] | None:
    """A matmul dim as ``(block, factor)``: a block, or a dim of length
    ``factor * block`` held as per-thread parts."""
    block = _block(size)
    return (block, 1) if block is not None else factored_size(size)


def _uniform_int(value: object, graphs: _Graphs) -> bool:
    """Every thread holds the same integer: a Python or symbolic int, or a 0-d
    tensor of an integer dtype.  A view's origin and extent are cast to
    ``long``, which would truncate a fractional base or bound."""
    if not isinstance(value, Node):
        return isinstance(value, int) and not isinstance(value, bool)
    val = graphs.origin(value).meta.get("val")
    if isinstance(val, torch.Tensor):
        return val.ndim == 0 and val.dtype in _INT_DTYPES
    return isinstance(val, (int, torch.SymInt)) and not isinstance(val, bool)


def _scalar_index(index: object, graphs: _Graphs) -> bool:
    """Whether a subscript entry selects one element of its tensor dim, the
    same for every thread.  As ``SubscriptIndexing.compute_shape`` decides,
    an int or a SymInt that is not a block size drops its dim; it is uniform
    when every symbol in it is a loop offset (an ``hl.grid`` index, a
    ``tile.begin`` or a ``tile.id``) or a host value.  A 0-d integer tensor
    is one value for every thread."""
    if isinstance(index, int):
        return not isinstance(index, bool)
    if not isinstance(index, Node):
        return False
    val = index.meta.get("val")
    if isinstance(val, torch.SymInt):
        expr = val._sympy_()
        origins = HostFunction.current().expr_to_origin
        for sym in expr.free_symbols:
            info = origins.get(sym)
            if info is None or isinstance(info.origin, BlockSizeOrigin):
                return False
            origin = info.origin
            loop_offset = type(origin) is GridOrigin or isinstance(
                origin, (TileBeginOrigin, TileIdOrigin)
            )
            if not (loop_offset or origin.is_host()):
                return False
        return True
    return _uniform_int(index, graphs)


def _tile_base(node: Node, graphs: _Graphs) -> tuple[int, Node | int] | None:
    """``(block, base)`` of ``base + tile.index`` (an ``add`` with alpha 1 of
    a tile's index and a uniform integer)."""
    if node.target is not torch.ops.aten.add.Tensor or node.kwargs.get("alpha", 1) != 1:
        return None
    a, b = node.args
    for t, other in ((a, b), (b, a)):
        if (
            isinstance(t, Node)
            and t.target is tile_index
            and _uniform_int(other, graphs)
        ):
            block = _block(_node(t.args[0]).meta["val"])
            if block is not None:
                assert isinstance(other, (Node, int))
                return block, other
    return None


def _view_dim(index: object, size: object, graphs: _Graphs) -> _Dim:
    """A view dim from a load index: a tile, ``base + tile.index``, ``base +
    hl.arange(block)``, a widened ``hl.arange``, or a ``:`` slice (``size``
    is the dim's size in the load's value)."""
    if isinstance(index, slice):
        block = _block(size)
        if index != slice(None) or block is None:
            raise _NotAView(f"an MPP view index {index!r}")
        return _Dim(block, full=True)
    if not isinstance(index, Node):
        raise _NotAView(f"an MPP view index {index!r}")
    if isinstance(index.meta["val"], torch.SymInt):
        block = _block(index.meta["val"])
        if block is None:
            raise _NotAView("an MPP view index that is not a tile")
        return _Dim(block)
    node = graphs.origin(index)
    index_node = node
    base: Node | int | None = None
    if widened := widened_iota(node):
        block, factor = widened
        return _Dim(block, iota=node, index=node, factor=factor)
    if tb := _tile_base(node, graphs):
        return _Dim(tb[0], base=tb[1], index=node)
    if node.target is torch.ops.aten.add.Tensor:
        # aten.add.Tensor(x, y, alpha=a) is x + a * y: only alpha 1 adds a
        # unit-step index.
        a, b = node.args
        if node.kwargs.get("alpha", 1) != 1:
            a = b = None
        if isinstance(a, Node) and a.target is _IOTA:
            node, other = a, b
        elif isinstance(b, Node) and b.target is _IOTA:
            node, other = b, a
        else:
            raise _NotAView("an MPP view index that is not base + hl.arange")
        if not _uniform_int(other, graphs):
            raise _NotAView("an MPP view index base that is not a uniform integer")
        assert isinstance(other, (Node, int))
        base = other
    if node.target is not _IOTA:
        raise _NotAView("an MPP view index that is not a tile or base + hl.arange")
    if node.kwargs.get("start", 0) != 0 or node.kwargs.get("step", 1) != 1:
        raise _NotAView("an MPP view hl.arange with a start or a step")
    block = _block(node.meta["val"].size(0))
    if block is None:
        raise _NotAView("an MPP view hl.arange whose length is not a block")
    return _Dim(block, iota=node, base=base, index=index_node)


def _view(
    operand: Node,
    dims: tuple[tuple[int, int], tuple[int, int]],
    batch: int | None,
    graphs: _Graphs,
) -> _View:
    """A device view: a load, or its transpose, whose value is 2-D, or 3-D
    for a batched matmul's operand.  Its two matrix dims are indexed by
    tiles, ``base + tile.index``, ``base + hl.arange``, a widened
    ``hl.arange`` or ``:``, with an optional mask ``index < bound`` on one
    ``hl.arange`` dim; a batched operand's leading dim by the batch tile, or
    ``:`` over a dim of size 1; every other tensor dim by a uniform integer,
    which drops it."""
    permute = torch.ops.aten.permute.default
    rank = operand.meta["val"].ndim
    swap = [1, 0] if rank == 2 else [0, 2, 1]
    transposed = operand.target is permute and list(_seq(operand.args[1])) == swap
    if transposed:
        operand = _node(operand.args[0])
    if operand.target is not load:
        raise _NotAView("an MPP operand that is not a load")
    host, subscript, extra_mask = operand.args[:3]
    assert isinstance(host, Node)
    index = _seq(subscript)
    tensor = host.meta["val"]
    value = operand.meta["val"]
    assert isinstance(tensor, torch.Tensor) and isinstance(value, torch.Tensor)
    if len(index) != tensor.ndim:
        raise _NotAView("an MPP view whose subscript does not index every tensor dim")
    if tensor.dtype not in _DTYPES:
        raise _unsupported(f"an MPP operand of dtype {tensor.dtype}")
    if not all(isinstance(x, int) for x in (*tensor.shape, *tensor.stride())):
        # The descriptor's transpose flag and the view's unit-stride dim are
        # compile-time facts.
        raise _NotAView("an MPP view of a tensor with symbolic sizes or strides")
    positions: list[int] = []
    outer: list[tuple[int, Node | int]] = []
    batch_dim: tuple[int, int] | None = None
    for i, x in enumerate(index):
        if not _scalar_index(x, graphs):
            positions.append(i)
            continue
        assert isinstance(x, (Node, int))
        outer.append((i, x))
    if len(positions) != rank:
        raise _NotAView(
            "an MPP view whose load is not 2-D, or batched 3-D, after its scalar "
            "indices"
        )
    if rank == 3:
        # The value's leading dim is the batch: the batch tile's index, whose
        # block _mnk matched, or ':' over a batch of size 1 (one index, 0).
        b = positions.pop(0)
        x = index[b]
        val = x.meta.get("val") if isinstance(x, Node) else None
        if batch is not None and isinstance(val, torch.SymInt):
            batch_dim = (b, batch)
        elif batch is None and x == slice(None):
            outer.append((b, 0))
        else:
            raise _NotAView("an MPP view whose batch index is not the batch tile")
    if 1 not in (tensor.stride(positions[0]), tensor.stride(positions[1])):
        raise _NotAView("an MPP view without a unit stride")
    first = rank - 2
    value_dims = [
        _view_dim(index[p], value.size(first + j), graphs)
        for j, p in enumerate(positions)
    ]
    if extra_mask is not None:
        if rank == 3:
            raise _NotAView("a masked batched MPP view")
        _view_mask(extra_mask, value_dims, graphs)
    if transposed:
        value_dims.reverse()
        positions.reverse()
    view_dims = (value_dims[0], value_dims[1])
    if tuple((d.block_id, d.factor) for d in view_dims) != dims:
        raise _unsupported("an MPP view whose dims are not the matmul's blocks")
    return _View(
        tensor, view_dims, (positions[0], positions[1]), tuple(outer), batch_dim
    )


def _producers(node: Node, graphs: _Graphs) -> list[Node]:
    """The nodes a staged operand is computed from, out to its loads,
    placeholders and constants."""
    out: list[Node] = []
    work = [node]
    seen = {node}
    while work:
        n = graphs.origin(work.pop())
        if n.op != "call_function":
            continue
        out.append(n)
        if n.target is load:
            continue
        for inp in n.all_input_nodes:
            if inp not in seen:
                seen.add(inp)
                work.append(inp)
    return out


def _nested_loads(node: Node, graphs: _Graphs) -> list[Node]:
    """The loads in the graphs a control-flow node runs, at any depth."""
    if _tracing_ops.is_for_loop_target(node.target):
        ids = [node.args[0]]
    elif node.target is _tracing_ops._if:
        ids = [node.args[1], node.args[2]]
    elif node.target is _tracing_ops._while_loop:
        ids = [*node.args[:2], *node.args[3:4]]
    else:
        return []
    out: list[Node] = []
    for gid in ids:
        if gid is None:
            continue
        for n in graphs.info(gid).graph.nodes:
            out += [n] if n.target is load else _nested_loads(n, graphs)
    return out


def _operand(
    node: Node,
    dims: tuple[tuple[int, int], tuple[int, int]],
    batch: int | None,
    graphs: _Graphs,
) -> _Operand:
    """A device view when the operand is one, else staged.  The threads
    compute a staged operand in the Scalar layout, one element per block
    coordinate (``c`` parts on a factored K), and write it into the tile at
    those coordinates, so every node it is computed from must meet its
    operands' dims by block."""
    try:
        view = _view(node, dims, batch, graphs)
    except _NotAView:
        val = node.meta["val"]
        assert isinstance(val, torch.Tensor)
        if val.dtype not in _DTYPES:
            raise _unsupported(f"an MPP operand of dtype {val.dtype}") from None
        positional_rebind.check_positional_merges(
            _producers(node, graphs), _STAGE_REBIND
        )
        return _Operand(val.dtype)
    return _Operand(view.tensor.dtype, view)


def _view_mask(extra_mask: object, dims: Sequence[_Dim], graphs: _Graphs) -> None:
    assert isinstance(extra_mask, Node)
    mask = graphs.origin(extra_mask)
    if mask.target is not view_ops.subscript:
        raise _NotAView("an MPP view mask that is not a broadcast of a 1-D mask")
    inner, sub = _node(mask.args[0]), _seq(mask.args[1])
    positions = [i for i, s in enumerate(sub) if s is not None]
    if len(positions) != 1:
        raise _NotAView("an MPP view mask over more than one dim")
    dim = dims[positions[0]]
    cmp = graphs.origin(inner)
    if cmp.target not in (torch.ops.aten.lt.Tensor, torch.ops.aten.lt.Scalar):
        raise _NotAView("an MPP view mask that is not index < bound")
    lhs, bound = cmp.args
    # The bound becomes the view's extent from the tensor's origin, so the
    # mask must compare the absolute index the load reads, the same node.
    # A tile's own mask is its loop's end; only an hl.arange dim takes one.
    if (
        dim.iota is None
        or dim.index is None
        or not (isinstance(lhs, Node) and graphs.origin(lhs) is dim.index)
    ):
        raise _NotAView(
            "an MPP view mask that does not compare the load's own index on that dim"
        )
    if not _uniform_int(bound, graphs):
        raise _NotAView(
            "an MPP view mask index < bound with a bound that is not a uniform integer"
        )
    assert isinstance(bound, (Node, int))
    dim.bound = bound
    dim.mask = cmp


def plan_mpp_matmuls(graphs: list[GraphInfo], threaded: set[int]) -> MppPlan:
    """Admit each aten matmul (``addmm``, ``mm``, ``bmm``, ``baddbmm``), each
    ``hl.dot`` over a K held as per-thread parts, and the other ``hl.dot`` calls
    when the planner takes every one of them (:func:`_takes_dots`), of the
    kernel's live graphs as a cooperative group, or raise ``BackendUnsupported``.
    ``threaded`` holds the blocks the config's tile strategies spread over more
    than one thread.

    Each use of a group's cooperative value is decided from the FX graph: an
    element-wise use or a store that an element loop can evaluate at any
    element stays cooperative; every other use reads the value through a
    ``_mpp_mat`` node, a materialization into the Scalar layout inserted
    right after the value's definition, so it reads the version it names.  A
    seed that is not a constant fill reaches the K loop through a
    ``_mpp_coop`` node, which stages it into the cooperative tensor.

    Only the graphs the kernel runs (``live_graphs``) are scanned.
    """
    from .backend import scalar_contraction_blocks

    dots = [
        n
        for g in live_graphs(graphs)
        for n in g.graph.find_nodes(op="call_function", target=dot)
        if not _factored_dot(n)
    ]
    # Decided for the whole kernel: a cooperative group changes its staging,
    # thread checks and lane splitting, so one hl.dot the planner can't take
    # keeps every whole-K one on the scalar contraction (matmul_ops).
    dot_ks = scalar_contraction_blocks(graphs)
    take_dots = (
        bool(dots)
        and all(_dot_admissible(m, dot_ks) for m in dots)
        and _takes_dots(graphs, threaded)
    )
    return _plan_matmuls(graphs, take_dots)


def _plan_matmuls(graphs: list[GraphInfo], take_dots: bool) -> MppPlan:
    from ..device_ir import ForLoopGraphInfo

    plan = MppPlan()
    taken: set[Node] = set()
    live = live_graphs(graphs)
    known = _Graphs(graphs)
    # Collected first: planning inserts conversion nodes into these graphs.
    matmuls = [
        (g, n)
        for g in live
        for n in g.graph.nodes
        if n.op == "call_function" and n.target in _MATMULS
    ]
    for g, matmul in matmuls:
        if matmul.target is dot and not _factored_dot(matmul):
            if not take_dots:
                continue
        elif matmul.target is dot:
            if matmul.args[2] is None:
                raise _unsupported(
                    "an hl.dot over a K held as per-thread parts with no acc"
                )
            if matmul.args[3] not in (None, torch.float32):
                raise _unsupported(
                    f"an hl.dot over per-thread parts with out_dtype={matmul.args[3]}"
                )
        k_block = _full_k(matmul)
        if k_block is None and _MATMULS[matmul.target][0] is None:
            # Over a tiled K, a matmul with no acc gives each K tile's product
            # alone; its codegen rejects it, after the loads' own lowerings.
            continue
        if k_block is not None:
            group = _plan_full_k(len(plan.by_matmul), matmul, k_block, known)
        elif not isinstance(g, ForLoopGraphInfo) or len(g.block_ids) != 1:
            raise _unsupported(
                "a matmul over a tiled K outside its K loop; loop over K with "
                "hl.tile, or take K whole with a ':' slice"
            )
        else:
            group = _plan_group(len(plan.by_matmul), matmul, g.block_ids[0], known)
        # Each group's op updates its storage in place, so one value can hold
        # only one group's cooperative tensor.
        storage = group.old | group.cur
        if storage & taken:
            raise _unsupported(
                "two MPP K loops whose accumulators start from one value, such as "
                "one hl.zeros seeding both; give each loop its own"
            )
        taken |= storage
        plan.by_matmul[matmul] = group
        if group.seed is not None:
            plan.by_seed[group.seed] = group
        for vec in group.vecs:
            plan.by_vec[vec] = group
    if plan.by_matmul:
        _link_element_staging(plan, live)
        _order_memory(plan, live, known)
        _unlink_dropped(plan)
        for group in plan.by_matmul.values():
            _stage_epilogue_vecs(group, plan, known)
            _finalize(group, plan, known)
    return plan


def _link_element_staging(plan: MppPlan, live: list[GraphInfo]) -> None:
    """Stage each staged operand that is an element-wise function of another
    group's cooperative value from that group's element loop: each thread
    writes the elements MPP gives it, computed from its own cooperative
    elements, so the value needs no materialization and no Scalar staging.
    Its other inputs must be recomputable at any element of the source tile
    and both matmuls in one graph; a value the source op updates in place is
    read before the write.  An element-wise function of the source's value
    lies over the source's M and N, in their order (a permute is not
    element-wise), so the operand's blocks need no check."""
    written = {
        _storage_key(n) for gi in live for n in gi.graph.nodes if n.target in _WRITES
    }
    groups = list(plan.by_matmul.values())
    for g in groups:
        _, lhs, rhs = _operands(g.matmul)
        for operand, node in ((g.lhs, lhs), (g.rhs, rhs)):
            if not operand.staged:
                continue
            for h in groups:
                if h.matmul.graph is not g.matmul.graph:
                    continue
                chain = _element_chain(node, h, g.matmul)
                if chain is None:
                    continue
                # The element loop re-runs the chain's loads on the threads MPP
                # gives each element, which no barrier orders after another
                # thread's write.
                if any(
                    _storage_key(ld) in written
                    for c in chain
                    for a in c.all_input_nodes
                    for ld in h.recomputable.nodes(a)
                    if ld.target is load
                ):
                    continue
                h.derived |= chain
                h.stage_uses.setdefault(node, set()).add(g.matmul)
                operand.elements = h.gid
                break


def _unlink_dropped(plan: MppPlan) -> None:
    """Stage a linked operand from the Scalar layout again when pruning
    dropped its value from the source group's cooperative values (another
    use of it needs the Scalar layout)."""
    groups = {group.gid: group for group in plan.by_matmul.values()}
    for g in plan.by_matmul.values():
        _, lhs, rhs = _operands(g.matmul)
        for operand, node in ((g.lhs, lhs), (g.rhs, rhs)):
            if operand.elements is not None and not _elementwise_of(
                node, groups[operand.elements]
            ):
                operand.elements = None


def _elementwise_of(node: Node, group: MppGroup) -> bool:
    return node in group.cur | group.derived | group.early


def _element_chain(node: Node, h: MppGroup, consumer: Node) -> set[Node] | None:
    """The element-wise nodes that compute ``node`` from ``h``'s cooperative
    value, its other inputs recomputable at any element of ``h``'s tile, or
    None when it isn't such a value.  A chain that reads the value the op
    updates in place runs, with its consumer, before the op writes it."""
    chain: set[Node] = set()
    reached = reached_early = False
    work = [node]
    seen: set[Node] = set()
    while work:
        n = work.pop()
        if n in seen:
            continue
        seen.add(n)
        if n in h.cur:
            reached = True
        elif n in h.early:
            reached = reached_early = True
        elif n in h.derived:
            # Its other inputs are what _inputs_kept admitted at any element.
            chain.add(n)
            work += [a for a in n.all_input_nodes if _elementwise_of(a, h)]
        elif not h.recomputable(n):
            if not _pointwise(n):
                return None
            chain.add(n)
            work += n.all_input_nodes
    if not reached or (reached_early and not (chain | {consumer}) <= _before_write(h)):
        return None
    return chain


def _factored_dot(matmul: Node) -> bool:
    return factored_size(_node(matmul.args[0]).meta["val"].size(-1)) is not None


def _takes_dots(graphs: list[GraphInfo], threaded: set[int]) -> bool:
    """Whether the planner takes the kernel's whole-K ``hl.dot`` calls: it
    plans a copy of the graphs with them without a rejection, no operand it
    stages lies over an M or N tile that every config makes 1, which codegen
    rejects, and each operand computed from a matmul's result is staged from
    that matmul's element loop.  Such a kernel keeps the scalar contraction at
    a config whose threads codegen rejects around one of its ops
    (:func:`_foreign_threads`).  Planning rewrites the graphs it plans, so it
    runs on copies, as each codegen does (``build_codegen_graphs``)."""
    copies = [g.copy() for g in graphs]
    known = _Graphs(copies)
    derived = {
        (m, side)
        for g in live_graphs(copies)
        for m in g.graph.find_nodes(op="call_function", target=dot)
        for side in (0, 1)
        if _matmul_derived(_node(m.args[side]), known)
    }
    try:
        plan = _plan_matmuls(copies, take_dots=True)
    except exc.BackendUnsupported:
        return False
    if any(
        (operand.staged and _static_tile(block) == 1)
        # A matmul's result reaches another op only from its element loop.
        or (operand.elements is None and (group.matmul, side) in derived)
        for group in plan.by_matmul.values()
        if group.matmul.target is dot
        for side, (operand, block) in enumerate(
            ((group.lhs, group.m_block), (group.rhs, group.n_block))
        )
    ):
        return False
    return not derived or not any(
        _foreign_threads(group, threaded, known) for group in plan.by_matmul.values()
    )


def _static_tile(block: int) -> int | None:
    """The block's tile when no config changes it: a ``:`` slice's length
    rounded up to a power of two, a fixed block size, or 1 over an extent of
    1; None otherwise."""
    env = CompileEnvironment.current()
    info = env.block_sizes[block]
    if info.reduction:
        return next_power_of_2(info.size) if isinstance(info.size, int) else None
    source = info.block_size_source
    if isinstance(source, FixedBlockSizeSource):
        return source.value if isinstance(source.value, int) else None
    if isinstance(info.size, (int, torch.SymInt)) and env.known_equal(info.size, 1):
        return 1
    return None


def _dot_admissible(matmul: Node, dot_ks: set[int]) -> bool:
    """Whether an ``hl.dot`` over a whole K may go to the planner: no product
    rounded to another ``out_dtype``, each dim whose tile no config changes
    within one op, and with no acc only over a ``:`` K.  An operand computed
    from a matmul's result and the planner's own rejections are
    :func:`_takes_dots`'s."""
    lhs, rhs, acc, out_dtype = matmul.args[:4]
    # The result takes the acc's dtype, so a float32 acc can still carry a
    # product rounded to another out_dtype.
    if out_dtype not in (None, torch.float32):
        return False
    lval, rval = _node(lhs).meta["val"], _node(rhs).meta["val"]
    env = CompileEnvironment.current()
    for size, low, high, is_k in (
        (lval.size(-2), 8, _WIDE_TILE, False),
        (rval.size(-1), 8, _WIDE_TILE, False),
        (lval.size(-1), 16, _MAX_TILE[2], True),
    ):
        block = _block(size)
        if block is None or (tile := _static_tile(block)) is None:
            continue
        if env.block_sizes[block].reduction:
            if not is_k and block not in dot_ks:
                # A ':' M or N becomes a lane loop around the op wherever
                # Scalar work sits beside it, and the jam runs at most
                # MAX_LANE_COPIES copies.  The lane floor gives an hl.dot's K
                # enough threads to walk at most that many lanes
                # (backend._lane_floor).
                high = min(high, MAX_LANE_COPIES)
        elif tile == 1 and not is_k:
            # _op_dims widens it to 8; a staged operand over it is
            # _takes_dots' check.
            continue
        if not low <= tile <= high:
            return False
    return acc is not None or _full_k(matmul) is not None


def _matmul_derived(node: Node, graphs: _Graphs) -> bool:
    """Whether a matmul's result reaches ``node``: through its producers,
    each placeholder's value from the enclosing graph, each loop-carried
    placeholder's carried-back value and each for loop's output value in its
    body, up to loads and shape reads.

    Unlike :func:`_producers`, which stops at a carried placeholder (a staged
    operand's positional check needs only its first-iteration value), this
    follows every value an operand can hold.  It has no ``_if`` output edge:
    an ``if`` output that reaches an ``hl.dot`` operand fails in the
    frontend's masked-value pass (``node_masking.getitem_masked_value``)."""
    work = [node]
    seen: set[Node] = set()
    while work:
        n = work.pop()
        if n in seen:
            continue
        seen.add(n)
        if n.target in _MATMULS:
            return True
        if n.target in (load, torch.ops.aten.sym_size.int):
            continue
        if n.op == "placeholder":
            owner = graphs.owner(n)
            if owner is None:
                continue
            outer = graphs.outer(n)
            if isinstance(outer, Node):
                work.append(outer)
            if (
                _tracing_ops.is_for_loop_target(owner.target)
                and _carry(n, owner) is not None
            ):
                work.append(_carry_update(n, owner, graphs))
            continue
        flow = n.args[0] if n.target is operator.getitem and n.args else None
        if isinstance(flow, Node) and _tracing_ops.is_for_loop_target(flow.target):
            (output,) = graphs.info(flow.args[0]).graph.find_nodes(op="output")
            value = _seq(output.args[0])[cast("int", n.args[1])]
            if isinstance(value, Node):
                work.append(value)
            continue
        work += n.all_input_nodes
    return False


def _plan_group(gid: int, matmul: Node, k_loop_block: int, graphs: _Graphs) -> MppGroup:
    acc, lhs, rhs = _operands(matmul)
    assert acc is not None
    placeholder = acc
    while placeholder.target is _tracing_ops._new_var:
        placeholder = _node(placeholder.args[0])
    loop = graphs.owner(placeholder) if placeholder.op == "placeholder" else None
    carry = _carry(placeholder, loop) if loop is not None else None
    update: list[Node] = []
    if carry is None:
        chain = _update_chain(matmul, acc, graphs)
        if chain is not None:
            placeholder, update = chain
            loop = graphs.owner(placeholder)
            assert loop is not None
            carry = _carry(placeholder, loop)
    if carry is None:
        if _updates_carry(acc, graphs):
            raise _unsupported(
                "a matmul whose acc is computed from a loop carry that its "
                "result does not replace"
            )
        raise _unsupported("a matmul whose acc is not its K loop's carry")
    assert loop is not None
    (output,) = matmul.graph.find_nodes(op="output")
    slot = carry.args[1]
    assert isinstance(slot, int)
    if _seq(output.args[0])[slot] is not matmul:
        raise _unsupported("a matmul whose result is not its K loop's carry")
    levels = _levels(placeholder, loop, carry, matmul, graphs)
    m, n, kdim, batch = _mnk(matmul, k_loop_block)
    outer = levels[-1]
    seed = graphs.origin(outer.arg)
    fill = seed.args[1] if seed.target is full else None
    if not (isinstance(fill, (int, float)) and math.isfinite(fill)):
        seed = _scalar_seed(outer, m, n, batch, graphs)
    group = MppGroup(
        gid,
        matmul,
        seed,
        m,
        n,
        kdim[0],
        _operand(lhs, ((m, 1), kdim), batch, graphs),
        _operand(rhs, (kdim, (n, 1)), batch, graphs),
        k_factor=kdim[1],
        levels=levels,
        batch_block=batch,
    )
    if group.lhs.dtype != group.rhs.dtype:
        raise _unsupported("MPP operands of two dtypes")
    if update:
        upd = _insert(_mpp_upd, acc, args=(placeholder, acc))
        for user in list(acc.users):
            if user is not upd:
                user.replace_input_with(acc, upd)
        group.upd = upd
        group.update = update
        _storage(group, upd)
        # The op updates the written-back value in place, so its other
        # readers take a snapshot of it (_finalize), as of the seed's.
        group.old.add(upd)
        group.plumbing.add((placeholder, upd))
        group.recomputable = _Recomputable(group, graphs)
        _plan_update(group, graphs)
    else:
        _storage(group, acc)
        group.recomputable = _Recomputable(group, graphs)
    _classify_uses(group, graphs)
    return group


def _update_chain(
    matmul: Node, acc: Node, graphs: _Graphs
) -> tuple[Node, list[Node]] | None:
    """``(placeholder, nodes)`` when ``acc`` is computed from the carry of the
    matmul's K loop whose exit is the matmul: the carry's placeholder and,
    in graph order, every node on a path from it to ``acc``.  Each must be
    element-wise (or a ``_new_var``) and feed only the update, but ``acc``
    may have other readers.  None when ``acc`` doesn't read that carry."""
    graph = matmul.graph
    loop = graphs.owner(matmul)
    assert loop is not None
    (output,) = graph.find_nodes(op="output")
    slots = [i for i, v in enumerate(_seq(output.args[0])) if v is matmul]
    if len(slots) != 1:
        return None
    # Helion carries every loop output into the loop (a value first assigned
    # in the loop can't be read after it), so the slot has one placeholder.
    (placeholder,) = [
        p
        for p in graph.find_nodes(op="placeholder")
        if (c := _carry(p, loop)) is not None and c.args[1] == slots[0]
    ]
    cone: set[Node] = set()
    work = [acc]
    while work:
        n = work.pop()
        if n in cone:
            continue
        cone.add(n)
        work += n.all_input_nodes
    carry = set(_aliases(placeholder))
    reads = set(carry)
    chain: list[Node] = []
    for n in graph.nodes:
        # A shape read (sym_size) carries no value of the carry.
        if (
            n in cone
            and n not in carry
            and n.target is not torch.ops.aten.sym_size.int
            and reads & set(n.all_input_nodes)
        ):
            reads.add(n)
            chain.append(n)
    if acc not in reads:
        # Read only through a shape (``c[tm, tn]`` in the loop reads tm's
        # size from the carry): not a value of the carry.
        return None
    for n in chain:
        if not (n.target is _tracing_ops._new_var or _pointwise(n)):
            raise _unsupported(
                f"a matmul whose acc is computed from its K loop's carry by {n.name}, "
                "which is not element-wise (only an element-wise update of the "
                "carry before the op stays cooperative)"
            )
        if n is not acc and any(u not in reads for u in n.users):
            raise _unsupported(
                f"an update of an MPP accumulator whose intermediate value {n.name} "
                "has another reader; only the updated acc may be read"
            )
    return placeholder, chain


def _plan_update(group: MppGroup, graphs: _Graphs) -> None:
    """Admit the update's other operands: each is recomputable at any
    element, or a Scalar value over the group's blocks, which a ``_mpp_vec``
    node stages for the element loop."""
    coop = set(group.update) | set(_aliases(group.levels[0].placeholder))
    recomputable = group.recomputable
    staged: dict[Node, Node] = {}
    for node in group.update:
        for a in list(node.all_input_nodes):
            if a in coop or recomputable(a):
                continue
            vec = staged.get(a)
            if vec is None:
                val = a.meta.get("val")
                if not isinstance(val, torch.Tensor):
                    raise _unsupported(
                        f"an update of an MPP accumulator reads {a.name}, a scalar "
                        "that isn't a tensor, such as a float kernel argument"
                    )
                if not _mn_dims(group, val):
                    raise _unsupported(
                        f"an update of an MPP accumulator reads {a.name}, which is "
                        "neither recomputable at any element nor a value over "
                        "the tile's M or N blocks"
                    )
                vec = staged[a] = _stage_vec(a, node, graphs)
                group.vecs.append(vec)
            node.replace_input_with(a, vec)


def _mn_dims(group: MppGroup, val: torch.Tensor) -> set[int | None]:
    """The group's M and N blocks among the dims of ``val``, or none when a
    dim is neither 1 nor one of the group's M, N and batch blocks.  A
    ``_mpp_vec`` tile can hold a value over one or both."""
    blocks = {group.m_block, group.n_block, group.batch_block}
    blocks.discard(None)
    dims = {_block(s) for s in val.size() if s != 1}
    if not dims <= blocks:
        return set()
    return dims & {group.m_block, group.n_block}


def _stage_vec(value: Node, before: Node | None, graphs: _Graphs) -> Node:
    positional_rebind.check_positional_merges(_producers(value, graphs), _STAGE_REBIND)
    return _insert(_mpp_vec, value, before=before)


def _mnk(matmul: Node, k_block: int) -> tuple[int, int, tuple[int, int], int | None]:
    """The matmul's M and N blocks, its K as ``(block, factor)``, whose block
    must be *k_block*, and a batched matmul's batch block.

    A batched matmul (``bmm``, ``baddbmm``) is one rank-2 op per batch index:
    its batch dim is a tile whose block must hold one index at codegen
    (:func:`_check_batch`), or a dim of size 1 (batch block None).  The acc
    is checked here against the operands' blocks: ``baddbmm``'s traced
    value takes the acc's shape, so ``positional_rebind``'s accumulator
    check compares the acc with itself.
    """
    acc, lhs, rhs = _operands(matmul)
    lval, rval = lhs.meta["val"], rhs.meta["val"]
    out = matmul.meta["val"]
    batch: int | None = None
    if matmul.target in _BATCHED:
        b1, b2 = lval.size(0), rval.size(0)
        if isinstance(b1, int) and isinstance(b2, int) and b1 == b2 == 1:
            batch = None
        else:
            batch = _block(b1)
            if batch is None or _block(b2) != batch:
                raise _unsupported(
                    "a batched MPP matmul whose operands' batch dims are not one "
                    "block, or all of size 1"
                )
        lval, rval = lval[0], rval[0]
        out = out[0]
    m, n = _block(lval.size(0)), _block(rval.size(1))
    kdim, kdim2 = _dim_of(lval.size(1)), _dim_of(rval.size(0))
    if (
        m is None
        or n is None
        or kdim is None
        or kdim != kdim2
        or kdim[0] != k_block
        or len({m, n, kdim[0]}) != 3
    ):
        raise _unsupported("a matmul whose blocks are not M, N and the loop's K")
    if acc is not None and matmul.target in _BATCHED:
        aval = acc.meta["val"]
        assert isinstance(aval, torch.Tensor)
        if aval.ndim != 3 or (
            (aval.size(0) != 1 if batch is None else _block(aval.size(0)) != batch)
            or [_block(aval.size(1)), _block(aval.size(2))] != [m, n]
        ):
            raise _unsupported(
                "a batched matmul whose acc's dims are not its batch, M and N "
                "blocks in place"
            )
    if out.dtype != torch.float32:
        raise _unsupported("a matmul whose acc is not float32")
    return m, n, kdim, batch


def _full_k(matmul: Node) -> int | None:
    """The block of the matmul's K when K is a ``:`` slice: a reduction
    dimension, which no loop iterates."""
    _, lhs, _ = _operands(matmul)
    block = _block(lhs.meta["val"].size(-1))
    if block is None or not CompileEnvironment.current().block_sizes[block].reduction:
        return None
    return block


def _plan_full_k(gid: int, matmul: Node, k_block: int, graphs: _Graphs) -> MppGroup:
    """A matmul over a ``:`` K: one op over the whole K, seeded by a constant
    fill or a Scalar value of the op's own graph, or, with no acc, by zeros
    the op emits itself."""
    acc, lhs, rhs = _operands(matmul)
    m, n, kdim, batch = _mnk(matmul, k_block)
    seed = acc
    while seed is not None and seed.target is _tracing_ops._new_var:
        seed = _node(seed.args[0])
    if seed is not None and seed.op == "placeholder":
        raise _unsupported(
            "a matmul over a ':' K whose acc an enclosing loop or if passes in"
        )
    fill = seed.args[1] if seed is not None and seed.target is full else None
    if acc is not None and not (isinstance(fill, (int, float)) and math.isfinite(fill)):
        # Staging writes each thread's element at its M and N position, so a
        # seed broadcast along M or N fills the tile; a seed whose dims meet
        # M and N out of place fails the acc rule on the group's slice, and
        # its producers' merges are checked here.
        positional_rebind.check_positional_merges(
            _producers(acc, graphs), _STAGE_REBIND
        )
        seed = _insert(_mpp_coop, acc, before=matmul)
        matmul.replace_input_with(acc, seed)
        acc = seed
    group = MppGroup(
        gid,
        matmul,
        seed,
        m,
        n,
        kdim[0],
        _operand(lhs, ((m, 1), kdim), batch, graphs),
        _operand(rhs, (kdim, (n, 1)), batch, graphs),
        batch_block=batch,
    )
    if group.lhs.dtype != group.rhs.dtype:
        raise _unsupported("MPP operands of two dtypes")
    _storage(group, acc)
    group.recomputable = _Recomputable(group, graphs)
    _classify_uses(group, graphs)
    return group


def _hoist_invariant(group: MppGroup, plan: MppPlan, graphs: _Graphs) -> None:
    """Stage each staged operand that its loop doesn't change once, before
    the loop: the loop must be outermost (its own graph runs in no loop),
    and the operand computed element-wise, in the loop, from values passed
    in from before it (no carry, no load in the loop, nothing over the
    loop's own block), constants and constant fills.  A view is a load at
    the op, so it is never hoisted."""
    from ..device_ir import ForLoopGraphInfo

    loop = _enclosing_loop(group.matmul.graph, graphs)
    if (
        loop is None
        or not _tracing_ops.is_for_loop_target(loop.target)
        or _enclosing_loop(loop.graph, graphs) is not None
    ):
        return
    info = graphs.info(loop.args[0])
    assert isinstance(info, ForLoopGraphInfo)
    _, lhs, rhs = _operands(group.matmul)
    for side, (node, operand) in enumerate(((lhs, group.lhs), (rhs, group.rhs))):
        cone = _invariant_cone(node, loop, info.graph, set(info.block_ids), graphs)
        if cone is None:
            continue
        clones = _clone_before(loop, info.graph, cone, graphs)
        operand.hoist = _insert(_mpp_hoist, clones[node])
        plan.by_hoist[operand.hoist] = (group, side)


def _clone_before(
    loop: Node, body: Graph, cone: list[Node], graphs: _Graphs
) -> dict[Node, Node]:
    """Clone ``cone`` right before ``loop``: each body placeholder becomes
    the value the loop passes in, and each if's placeholder its outer
    value's clone."""
    env: dict[Node, Node] = {}
    with loop.graph.inserting_before(loop):
        for n in cone:
            if n.op == "placeholder":
                outer = _node(graphs.outer(n))
                env[n] = outer if n.graph is body else env[outer]
            else:
                env[n] = loop.graph.node_copy(n, lambda a: env[a])
    return env


def _invariant_cone(
    node: Node, loop: Node, body: Graph, blocks: set[int], graphs: _Graphs
) -> list[Node] | None:
    """The tensors ``node`` is computed from inside the loop, in dependency
    order, down to the body's placeholders, when the loop doesn't change
    it; None otherwise.  Only ``if`` graphs lie between the loop's body and
    ``node``'s graph (``_enclosing_loop``), so a placeholder outside the
    body is an if's, followed out."""
    out: list[Node] = []
    seen: set[Node] = set()

    def visit(n: Node) -> bool:
        if n in seen:
            return True
        seen.add(n)
        val = n.meta.get("val")
        # Only tensors: a loop offset or index is a symbol.
        if not isinstance(val, torch.Tensor):
            return False
        if n.op == "placeholder":
            if n.graph is body:
                ok = _carry(n, loop) is None
            else:
                ok = visit(_node(graphs.outer(n)))
        else:
            # A value over the loop's tile is masked by it in each iteration.
            ok = (
                (
                    n.target in (full, _tracing_ops._new_var, _tracing_ops._mask_to)
                    or _pointwise(n)
                )
                and not any(_block(s) in blocks for s in val.size())
                and all(visit(a) for a in n.all_input_nodes)
            )
        if ok:
            out.append(n)
        return ok

    return out if visit(node) else None


_NOT_RELOADED = (
    "a bfloat MPP operand staged inside a while loop, or over a K held as "
    "per-thread parts, from a value the loop does not reload every iteration, "
    "which Metal miscompiles; compute it from loads inside the loop, or use "
    "float16"
)


def _stage_in_float(group: MppGroup, plan: MppPlan, graphs: _Graphs) -> None:
    """Stage in float32 each bfloat operand staged inside a for loop from a
    value the loop doesn't reload every iteration (:func:`_reloaded`).
    Metal miscompiles bfloat threadgroup stores of such a value, and float
    -> bfloat conversions on its way to them: at some thread layouts part
    of the tile reads back wrong.  float16 and float32 stores stay exact.  A
    float32 tile of the operand's bfloat values, with every conversion
    rounded in integer arithmetic (:func:`_round_conversions`), stays exact
    too, and MPP multiplies it by the other operand's bfloat handle.  The
    operand of a while loop, or over a factored K, is rejected."""
    loop = _enclosing_loop(group.matmul.graph, graphs)
    if loop is None:
        return
    in_for = _tracing_ops.is_for_loop_target(loop.target)
    _, lhs, rhs = _operands(group.matmul)
    for node, operand in ((lhs, group.lhs), (rhs, group.rhs)):
        if (
            not operand.staged
            or operand.hoist is not None
            or operand.elements is not None
            or operand.dtype != torch.bfloat16
            or (in_for and _reloaded(node, graphs.info(loop.args[0]).graph, graphs))
        ):
            continue
        if not in_for or group.k_factor != 1:
            raise _unsupported(_NOT_RELOADED)
        _round_conversions(group.matmul, node, plan, graphs)
        operand.float_tile = True


_CONVERT = torch.ops.prims.convert_element_type.default
#: Nodes that pass bfloat values on without computing new ones.  The other
#: value of a ``_mask_to`` is a Python constant (its signature), emitted as a
#: bfloat literal: no conversion runs, and it is the same as a float.
_BF16_MOVES = (_tracing_ops._new_var, _tracing_ops._mask_to, view_ops.subscript)
#: Inputs that can't hold a bfloat value or a conversion: sizes, block
#: sizes, offsets, indices and scalar kernel arguments.  Following a size of
#: the acc would reach its carry.
_SCALARS = (torch.SymInt, torch.SymFloat)


def _op_name(node: Node) -> str:
    assert callable(node.target)
    return node.target.__name__


def _not_followed(source: str) -> exc.BackendUnsupported:
    return _unsupported(
        f"a bfloat MPP operand staged inside a loop from {source}, whose bfloat "
        "conversions the planner does not follow (a loop, an if or a node with "
        "several outputs), which Metal miscompiles; compute the value inside the "
        "matmul's loop"
    )


def _carry_update(placeholder: Node, loop: Node, graphs: _Graphs) -> Node:
    """The value ``loop``'s body gives back for the carried ``placeholder``."""
    slot = _node(_carry(placeholder, loop)).args[1]
    assert isinstance(slot, int)
    (output,) = graphs.info(loop.args[0]).graph.find_nodes(op="output")
    return _node(_seq(output.args[0])[slot])


def _op_value(node: Node) -> bool:
    """Whether ``node`` is a matmul or an updated acc: another MPP op's float
    value, which the plan holds by identity."""
    return node.target in _MATMULS or node.target is _mpp_upd


def _round_conversions(
    matmul: Node, node: Node, plan: MppPlan, graphs: _Graphs
) -> None:
    """Re-point ``matmul`` from its operand ``node`` to a copy of ``node``'s
    chain in which each float -> bfloat conversion is a ``_mpp_round``.

    The walk back from ``node`` stops at loads and at another op's float
    value (:func:`_op_value`), and must see every producer of the value: a
    placeholder leads to the value its for loop or if passes in, a carry to
    its seed and its update, and an input that is neither a tensor nor a
    scalar (a loop's, an if's or a multi-output node's result), or a while
    loop's placeholder, raises.  Of the bfloat values it reaches, only a
    conversion in ``matmul``'s graph and not through a carry is rounded;
    data moves pass, and every other bfloat value raises.  A float value is
    computed in float, from its bfloat inputs promoted exactly.

    The copy covers every node on a path from a rounded conversion to
    ``node``, a load included (a gather whose index the path computes),
    whose copy keeps its fence.  Another op's value is never copied: it
    keeps reading the original nodes, which stay for it."""
    graph = matmul.graph
    converts: list[Node] = []
    seen: set[Node] = set()
    work: list[tuple[Node, bool]] = [(node, False)]
    visited: set[tuple[Node, bool]] = set()
    while work:
        item = work.pop()
        if item in visited:
            continue
        visited.add(item)
        n, via_carry = item
        if n.op == "placeholder":
            owner = graphs.owner(n)
            if owner is None:
                raise _not_followed("a while loop's input")
            if (
                _tracing_ops.is_for_loop_target(owner.target)
                and _carry(n, owner) is not None
            ):
                seed = _node(graphs.outer(n))
                work += [(seed, True), (_carry_update(n, owner, graphs), True)]
            else:
                work.append((_node(graphs.outer(n)), via_carry))
            continue
        seen.add(n)
        if n.target is load or _op_value(n):
            continue
        val = n.meta["val"]
        if isinstance(val, torch.Tensor) and val.dtype == torch.bfloat16:
            if n.target is _CONVERT:
                if via_carry or n.graph is not graph:
                    raise _unsupported(
                        "a bfloat MPP operand staged inside a loop from a bfloat "
                        "conversion before the loop, in another graph of the loop "
                        "or in a carry, which Metal miscompiles; convert it inside "
                        "the loop, in the matmul's own block"
                    )
                converts.append(n)
            elif n.target not in _BF16_MOVES:
                raise _unsupported(
                    "a bfloat MPP operand staged inside a loop from a bfloat value "
                    f"computed by {_op_name(n)} that the loop does not reload, which "
                    "Metal miscompiles; compute it in float32 and convert it with "
                    ".to(torch.bfloat16)"
                )
        for a in n.all_input_nodes:
            v = a.meta.get("val")
            if isinstance(v, torch.Tensor):
                work.append((a, via_carry))
            elif not isinstance(v, _SCALARS):
                raise _not_followed(f"the output of {_op_name(a)}")
    if not converts:
        return
    # Every node on a path from a conversion to ``node`` is in ``graph``: a
    # path that left it would come back through a loop's or an if's result.
    order = {n: i for i, n in enumerate(graph.nodes)}
    path = set(converts)
    for n in sorted((m for m in seen if m.graph is graph), key=order.__getitem__):
        if not _op_value(n) and any(a in path for a in n.all_input_nodes):
            path.add(n)
    copies: dict[Node, Node] = {}
    for n in sorted(path, key=order.__getitem__):
        if n in converts:
            src = _node(n.args[0])
            copies[n] = _insert(_mpp_round, n, args=(copies.get(src, src),))
        else:
            with graph.inserting_after(n):
                copies[n] = graph.node_copy(n, lambda a: copies.get(a, a))
            if n in plan.fenced:
                plan.fenced.add(copies[n])
    matmul.replace_input_with(node, copies[node])
    for n in sorted(path, key=order.__getitem__, reverse=True):
        if not n.users:
            graph.erase_node(n)


def _enclosing_loop(graph: Graph, graphs: _Graphs) -> Node | None:
    """The innermost for or while loop *graph* runs in, or None."""
    while True:
        gid = graphs.graph_id[graph]
        if gid in graphs.whiles:
            return graphs.whiles[gid]
        if gid not in graphs.owners:
            return None
        owner = graphs.owners[gid][0]
        if _tracing_ops.is_for_loop_target(owner.target):
            return owner
        graph = owner.graph


def _reloaded(node: Node, body: Graph, graphs: _Graphs) -> bool:
    """Whether every value *node* is computed from is a load inside *body*
    or a constant fill.  A load ends its path: its index and mask decide
    where it reads, not whether the loop reloads it."""
    work, seen = [node], {node}
    while work:
        n = graphs.origin(work.pop())
        if n.op != "call_function":
            return False
        if n.target is load:
            if not graphs.within(n.graph, body):
                return False
            continue
        if n.target is full:
            value = n.args[1]
            inputs = [value] if isinstance(value, Node) else []
        else:
            inputs = []
            for i in n.all_input_nodes:
                val = i.meta.get("val")
                if isinstance(val, torch.Tensor):
                    inputs.append(i)
                elif not _launch_constant(val):
                    return False
            if not inputs:
                return False
        for i in inputs:
            if i not in seen:
                seen.add(i)
                work.append(i)
    return True


def _launch_constant(val: object) -> bool:
    """A symbolic number whose symbols are all block sizes or host values (a
    shape or a scalar kernel argument): the same in every iteration of every
    loop."""
    if not isinstance(val, (torch.SymInt, torch.SymFloat)):
        return False
    origins = HostFunction.current().expr_to_origin
    for sym in val._sympy_().free_symbols:
        info = origins.get(sym)
        if info is None or not (
            isinstance(info.origin, BlockSizeOrigin) or info.origin.is_host()
        ):
            return False
    return True


def _updates_carry(acc: Node, graphs: _Graphs) -> bool:
    """Whether ``acc`` is computed element-wise, in its own graph, from a
    value its loop carries."""
    work, seen = [acc], {acc}
    while work:
        node = work.pop()
        if node.target is _tracing_ops._new_var or _pointwise(node):
            inputs = node.all_input_nodes
        elif node.op == "placeholder":
            loop = graphs.owner(node)
            if (
                loop is not None
                and _tracing_ops.is_for_loop_target(loop.target)
                and _carry(node, loop) is not None
            ):
                return True
            inputs = []
        else:
            inputs = []
        for inp in inputs:
            if inp not in seen:
                seen.add(inp)
                work.append(inp)
    return False


def _levels(
    placeholder: Node, loop: Node, carry: Node, matmul: Node, graphs: _Graphs
) -> list[_Level]:
    """The control-flow levels from the K loop out to the seed's graph.

    ``run`` updates the cooperative tensor in place, so every level the seed
    crosses on its way in must give the updated value back at the same
    variable: a loop's carry, or an if output the else branch leaves as it
    was.  Then the storage is one variable throughout, and each level's exit
    holds its current value.
    """
    levels = [_level(loop, placeholder, carry, matmul, graphs)]
    while True:
        node = levels[-1].arg
        while node.target is _tracing_ops._new_var:
            node = _node(node.args[0])
        owner = graphs.owner(node) if node.op == "placeholder" else None
        if owner is None:
            return levels
        inner = levels[-1].phi
        if _tracing_ops.is_for_loop_target(owner.target):
            exit_ = _carry(node, owner)
        else:
            exit_ = _if_exit(owner, node, inner, graphs)
        if exit_ is None:
            raise _unsupported(
                "an MPP accumulator passed into a loop or an if that doesn't give "
                "it back as the same variable"
            )
        levels.append(_level(owner, node, exit_, inner, graphs))


def _level(
    owner: Node, placeholder: Node, exit_: Node, inner: Node, graphs: _Graphs
) -> _Level:
    arg = _node(graphs.outer(placeholder))
    slot = exit_.args[1]
    assert isinstance(slot, int)
    if _tracing_ops.is_for_loop_target(owner.target):
        out_slot = slot
    else:
        _, branches = graphs.if_branch(graphs.graph_id[placeholder.graph])
        out_slot = branches[slot][0]
        assert isinstance(out_slot, int)
    (output,) = placeholder.graph.find_nodes(op="output")
    if _seq(output.args[0])[out_slot] is not inner:
        raise _unsupported(
            "an MPP accumulator whose loop or if gives back another value than "
            "the updated one"
        )
    phis = [
        u for u in exit_.users if u.target is _tracing_ops._phi and u.args[1] is exit_
    ]
    if len(phis) != 1:
        raise _unsupported("an MPP accumulator read past its loop without a merge")
    return _Level(owner, arg, placeholder, exit_, phis[0], out_slot)


def _if_exit(
    owner: Node, placeholder: Node, inner: Node, graphs: _Graphs
) -> Node | None:
    """The getitem of ``owner``, an ``_if`` whose if branch holds
    ``placeholder``, for the output that holds ``inner``, when the else
    branch leaves that variable as it was."""
    if graphs.graph_id[placeholder.graph] != owner.args[1]:
        return None
    _, branches = graphs.if_branch(graphs.graph_id[placeholder.graph])
    (output,) = placeholder.graph.find_nodes(op="output")
    outs = _seq(output.args[0])
    for i, (if_index, else_side) in enumerate(branches):
        if (
            isinstance(else_side, str)
            and isinstance(if_index, int)
            and outs[if_index] is inner
        ):
            for u in owner.users:
                if u.target is operator.getitem and u.args[1] == i:
                    return u
    return None


def _scalar_seed(
    outer: _Level, m: int, n: int, batch: int | None, graphs: _Graphs
) -> Node:
    """Route a seed that is not a constant fill through a ``_mpp_coop`` node
    before the outermost level, which stages it into the cooperative tensor;
    the seed's other readers keep its Scalar value."""
    arg = outer.arg
    val = arg.meta["val"]
    assert isinstance(val, torch.Tensor)
    want = (m, n) if val.ndim == 2 else (batch, m, n)
    if val.ndim not in (2, 3) or tuple(_block(s) for s in val.size()) != want:
        raise _unsupported("an MPP accumulator seed whose dims are not M and N")
    positional_rebind.check_positional_merges(_producers(arg, graphs), _STAGE_REBIND)
    coop = _insert(_mpp_coop, arg, before=outer.owner)
    outer.owner.replace_input_with(arg, coop)
    outer.phi.replace_input_with(arg, coop)
    outer.arg = coop
    return coop


def _insert(
    op: Callable[..., object],
    src: Node,
    *,
    before: Node | None = None,
    args: tuple[Node, ...] | None = None,
) -> Node:
    """A call of ``op`` on ``args`` (default ``(src,)``) with ``src``'s meta,
    inserted before ``before`` or right after ``src``'s definition (after the
    graph's placeholders for one)."""
    graph = src.graph
    if before is not None:
        ctx = graph.inserting_before(before)
    elif src.op == "placeholder":
        ctx = graph.inserting_after(list(graph.find_nodes(op="placeholder"))[-1])
    else:
        ctx = graph.inserting_after(src)
    with ctx:
        node = graph.call_function(op, args or (src,), {})
    node.meta.update(src.meta)
    node.meta["lowering"] = APIFuncLowering(op)
    return node


def _aliases(node: Node) -> list[Node]:
    """``node`` and the ``_new_var`` copies of it."""
    out = [node]
    for user in node.users:
        if user.target is _tracing_ops._new_var:
            out += _aliases(user)
    return out


def _storage(group: MppGroup, acc: Node | None) -> None:
    """Sort the values that hold the cooperative storage into ``old`` (the
    seed and each level's input, which the op updates in place) and ``cur``
    (the op's result and each level's exit), and record the uses that only
    pass the storage along."""
    plumbing = group.plumbing
    if acc is not None:
        plumbing.add((acc, group.matmul))
    if group.levels:
        (output,) = group.matmul.graph.find_nodes(op="output")
        plumbing.add((group.matmul, output))
    group.cur |= set(_aliases(group.matmul))
    for i, level in enumerate(group.levels):
        group.old |= set(_aliases(level.placeholder))
        plumbing |= {(level.arg, level.owner), (level.arg, level.phi)}
        plumbing.add((level.exit, level.phi))
        group.cur |= {level.exit, *_aliases(level.phi)}
        if i + 1 < len(group.levels):
            (out,) = level.phi.graph.find_nodes(op="output")
            plumbing.add((level.phi, out))
    seed = group.levels[-1].arg if group.levels else acc
    while seed is not None and seed.target is _tracing_ops._new_var:
        seed = _node(seed.args[0])
    if seed is not None:
        group.old |= set(_aliases(seed))
    for node in group.old:
        plumbing |= {(node, u) for u in node.users if u.target is _tracing_ops._new_var}
    for node in group.cur:
        plumbing |= {(node, u) for u in node.users if u.target is _tracing_ops._new_var}


def _pointwise(node: Node) -> bool:
    target = node.target
    if target is view_ops.subscript:
        return all(x is None or x == slice(None) for x in _seq(node.args[1]))
    return isinstance(target, torch._ops.OpOverload) and (
        torch.Tag.pointwise in target.tags
        or target is torch.ops.prims.convert_element_type.default
    )


class _Recomputable:
    """Whether a value can be computed at any (m, n) of a group's tile, by
    any thread, from uniform values, the M and N indices and loads alone.

    A 0-d tensor is admitted without looking inside it: it has no block dim,
    so every thread holds the same value, and :func:`elementize` reads it by
    name.  It may hold a threadgroup reduction, whose collective stays at its
    own statement.  Every other admitted value is an iota, a fill, a load or
    an element-wise op, none of which exchanges data between threads, so
    every collective statement an element loop can reach computes a 0-d
    value.
    """

    def __init__(self, group: MppGroup, graphs: _Graphs) -> None:
        self.group = group
        self.graphs = graphs
        self.memo: dict[Node, bool] = {}

    def __call__(self, value: object) -> bool:
        if not isinstance(value, Node):
            return True
        node = self.graphs.origin(value)
        if self._updated_in_place(node):
            return False
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor) and val.ndim == 0:
            return True
        if isinstance(val, (torch.SymInt, int, float)):
            return True
        if node not in self.memo:
            self.memo[node] = False
            self.memo[node] = self._compute(node, val)
        return self.memo[node]

    def _updated_in_place(self, node: Node) -> bool:
        """Every reader of a value the op updates in place reads a snapshot in
        the Scalar layout (:func:`_finalize`), even a constant fill's, so no
        element loop may recompute it."""
        return node in self.group.old

    def _compute(self, node: Node, val: object) -> bool:
        # A batch block holds one index (_check_batch), the same at every
        # element.
        blocks = {self.group.m_block, self.group.n_block, self.group.batch_block}
        blocks.discard(None)
        if node.target is _IOTA:
            assert isinstance(val, torch.Tensor)
            ok = _block(val.size(0)) in blocks
        elif node.target is tile_index:
            # The element loop binds the thread and lanes that hold the element
            # in the Scalar layout, which give the tile's index.
            ok = _block(_node(node.args[0]).meta["val"]) in blocks
        elif node.target is full:
            ok = isinstance(node.args[1], (int, float))
        elif node.target is load:
            ok = all(map(self, _seq(node.args[1]))) and self(node.args[2])
        elif _pointwise(node):
            ok = all(map(self, node.all_input_nodes))
        else:
            ok = False
        if ok and isinstance(val, torch.Tensor):
            ok = all(s == 1 or _block(s) in blocks for s in val.size())
        return ok

    def nodes(self, value: object) -> list[Node]:
        """The admitted nodes ``value`` is computed from."""
        out: list[Node] = []
        work = [value]
        while work:
            v = work.pop()
            if isinstance(v, (list, tuple)):
                work += v
                continue
            if not isinstance(v, Node):
                continue
            node = self.graphs.origin(v)
            if not self.memo.get(node) or node in out:
                continue
            out.append(node)
            if node.target is load:
                work += [*_seq(node.args[1]), node.args[2]]
            else:
                work += node.all_input_nodes
        return out


def _classify_uses(group: MppGroup, graphs: _Graphs) -> None:
    """Find the uses an element loop evaluates on the cooperative tensor's
    own elements: element-wise ops whose other inputs are recomputable at
    any element (``derived``), and stores of the result or such an op at a
    recomputable index and mask (``stores``).  Every other use reads the
    value in the Scalar layout (:func:`_finalize`).

    The K loop's carry (``early``) counts as cooperative for the uses that
    run before the op first writes it in place: an element loop there reads
    the value from before the write.  A use after the write that reads the
    carry, through any chain, needs its snapshot.
    """
    recomputable = group.recomputable
    group.early = _carry_aliases(group)
    before_write = _before_write(group)
    work = list(group.cur) + list(group.early)
    reads_carry: set[Node] = set(group.early)
    while work:
        node = work.pop()
        for user in node.users:
            if (
                (node, user) in group.plumbing
                or user.target is torch.ops.aten.sym_size.int
                or user in group.derived
            ):
                continue
            early = any(a in reads_carry for a in user.all_input_nodes)
            if early and user not in before_write:
                continue
            if user.target is store:
                if (
                    user.args[2] is node
                    and all(map(recomputable, _seq(user.args[1])))
                    and recomputable(user.args[3])
                    and user not in group.stores
                ):
                    group.stores.append(user)
                continue
            if _pointwise(user) and _inputs_kept(
                user, group.cur | group.derived | group.early, group
            ):
                group.derived.add(user)
                work.append(user)
                if early:
                    reads_carry.add(user)


def _carry_aliases(group: MppGroup) -> set[Node]:
    """The K loop's carried placeholder and its copies: values the op updates
    in place.  Only their users in the op's own graph can run before the
    write (:func:`_before_write`)."""
    if not group.levels:
        return set()
    return set(_aliases(group.levels[0].placeholder))


def _before_write(group: MppGroup) -> set[Node]:
    """The nodes of the op's graph that run before the update or the op
    first writes the cooperative storage in place."""
    write = group.upd if group.upd is not None else group.matmul
    out: set[Node] = set()
    for n in group.matmul.graph.nodes:
        if n is write:
            break
        out.add(n)
    return out


def _inputs_kept(node: Node, coop: set[Node], group: MppGroup) -> bool:
    """Whether an element loop over the cooperative tensor can evaluate
    ``node``: each input is cooperative, recomputable at any element or a
    vector a tile can stage (:func:`_vec_input`)."""
    return all(
        a in coop or group.recomputable(a) or _vec_input(group, a)
        for a in node.all_input_nodes
    )


def _vec_input(group: MppGroup, value: Node) -> bool:
    """Whether a derived value's input that no element can recompute can be
    staged into a ``_mpp_vec`` tile: a row or a column vector of the tile
    (over one of its M and N blocks, and its batch block).  A value over both
    M and N stays in the Scalar layout: staged, it would take as much
    threadgroup memory as the materialization it avoids.  So does a value the
    op updates in place, such as a ``:`` K op's constant seed over one block:
    its readers read its materialization (:func:`_finalize`)."""
    val = value.meta.get("val")
    return (
        value not in group.old
        and isinstance(val, torch.Tensor)
        and len(_mn_dims(group, val)) == 1
    )


def _prune_derived(group: MppGroup) -> None:
    """Keep a derived value cooperative only while every input of it is
    cooperative or recomputable and every use of it is another kept one or
    an element-loop store; a store keeps its element loop only while its
    value is cooperative.  A dropped value is computed in the Scalar layout,
    so a kept value that read it would read one thread's element at every
    cooperative element."""
    while True:
        coop = group.cur | group.derived | group.early
        stores = [s for s in group.stores if s.args[2] in coop]
        keep = set(stores) | group.derived
        drop = {
            d
            for d in group.derived
            if any(
                u not in keep
                and u.target is not torch.ops.aten.sym_size.int
                and u not in group.stage_uses.get(d, ())
                for u in d.users
            )
            or not _inputs_kept(d, coop, group)
        }
        if not drop and len(stores) == len(group.stores):
            return
        group.derived -= drop
        group.stores = stores


def _storage_key(node: Node) -> StorageWeakRef:
    """The fake storage a load or a write accesses."""
    tensor = node.args[0]
    if isinstance(tensor, (list, tuple)):
        tensor = tensor[0]
    assert isinstance(tensor, Node)
    return StorageWeakRef(tensor.meta["val"].untyped_storage())


def _order_memory(plan: MppPlan, live: list[GraphInfo], graphs: _Graphs) -> None:
    """MPP gives each element to a thread of its own choosing, and no barrier
    orders device memory around an op.  So an element-loop store must write
    a tensor nothing else accesses and re-run no load of a tensor the kernel
    writes, and an operand view must not read such a tensor.  A store that
    can't keep that takes the Scalar layout, where the thread that writes an
    element is the one the Scalar program names, and a view is staged by
    the threads that hold it.  Tensors are keyed by fake storage, so views of
    one tensor fall together (as ``free_arange._check_write_once``)."""

    accesses: dict[StorageWeakRef, list[Node]] = {}
    written: set[StorageWeakRef] = set()
    for g in live:
        for n in g.graph.nodes:
            if n.target is load or n.target in _WRITES:
                accesses.setdefault(_storage_key(n), []).append(n)
                if n.target in _WRITES:
                    written.add(_storage_key(n))
    for group in plan.by_matmul.values():
        _prune_derived(group)
        recomputable = group.recomputable
        derived_loads = [
            n
            for d in group.derived
            for a in d.all_input_nodes
            for n in recomputable.nodes(a)
            if n.target is load
        ]
        kept: list[Node] = []
        for st in group.stores:
            reloads = derived_loads + [
                n
                for n in recomputable.nodes(st.args[2])
                + recomputable.nodes(st.args[1])
                + recomputable.nodes(st.args[3])
                if n.target is load
            ]
            if any(n is not st for n in accesses[_storage_key(st)]) or any(
                _storage_key(ld) in written for ld in reloads
            ):
                continue
            kept.append(st)
        group.stores = kept
        _prune_derived(group)
        # The update's element loop re-runs its recomputable operands on the
        # thread MPP gives each element, and unlike a store it can't take the
        # Scalar layout: the op reads its result.
        for ld in _update_reruns(group):
            if ld.target is load and _storage_key(ld) in written:
                raise _unsupported(
                    f"an update of an MPP accumulator reads {ld.name}, a load of a "
                    "tensor the kernel writes, which its element loop would re-run "
                    "on another thread than the one that wrote it"
                )
        _, lhs, rhs = _operands(group.matmul)
        for operand, node in ((group.lhs, lhs), (group.rhs, rhs)):
            view = operand.view
            if view is not None and (
                StorageWeakRef(view.tensor.untyped_storage()) in written
            ):
                operand.view = None
            if operand.staged:
                for ld in _operand_loads(node, group.matmul.graph, graphs):
                    if _storage_key(ld) in written:
                        _fence(plan, group, ld)


def _operand_loads(node: Node, body: Graph, graphs: _Graphs) -> list[Node]:
    """Every load in the K loop's ``body``, at any depth, that a staged operand
    reads: the loads it is computed from, those inside control flow it is
    computed from, and the loads their indices and masks are computed from.  A
    load outside the body runs in the Scalar layout before the loop, as it
    does without MPP, and the threads that stage the operand read its value
    from their own registers."""
    out: list[Node] = []
    work = [node]
    while work:
        for p in _producers(work.pop(), graphs):
            for ld in [p] if p.target is load else _nested_loads(p, graphs):
                if ld not in out and graphs.within(ld.graph, body):
                    out.append(ld)
                    work += [a for a in ld.all_input_nodes if a is not ld.args[0]]
    return out


def _fence(plan: MppPlan, group: MppGroup, ld: Node) -> None:
    """Order ``ld``, a staged operand's load of a tensor the kernel writes,
    after this threadgroup's earlier device writes.  The Scalar threads that
    stage the operand are not the ones that wrote its elements, and Metal
    orders device memory between threads only at a barrier, which every
    thread reaches in the op's own graph (:func:`_check_uniform`)."""
    if ld.graph is not group.matmul.graph:
        raise _unsupported(
            f"a staged MPP operand computed from {ld.name}, a load of a tensor the "
            "kernel writes, inside control flow in its K loop's body"
        )
    plan.fenced.add(ld)


def _finalize(group: MppGroup, plan: MppPlan, graphs: _Graphs) -> None:
    """Route every remaining use of a cooperative value through a
    materialization, and check that the code each layout runs meets dims by
    block (``positional_rebind``)."""
    # After ``_order_memory``, so an operand it demotes to staged is checked.
    _hoist_invariant(group, plan, graphs)
    _stage_in_float(group, plan, graphs)
    coop = group.cur | group.derived
    # The update's nodes read the carry where the update's element loop
    # runs them, before the op.
    keep = coop | set(group.stores) | set(group.update)
    recomputable = group.recomputable
    readers: list[tuple[Node, Node]] = []
    for node in sorted(group.old | group.cur, key=lambda n: n.name):
        for user in list(node.users):
            if (
                (node, user) in group.plumbing
                or user in keep
                or user in group.stage_uses.get(node, ())
                or user.target is torch.ops.aten.sym_size.int
                or (user.op == "output" and not _live_exit(node, user, graphs))
            ):
                continue
            readers.append((node, user))
    for node, user in readers:
        mat = group.mats.get(node)
        if mat is None:
            mat = group.mats[node] = _insert(_mpp_mat, node)
            plan.by_mat[mat] = group
        user.replace_input_with(node, mat)
    rerun = sorted(coop, key=lambda n: n.name) + list(group.stores) + group.update
    for n in [*group.derived, *group.stores]:
        for a in n.all_input_nodes:
            if a not in coop:
                rerun += recomputable.nodes(a)
    rerun += _update_reruns(group)
    _check_uniform(group.matmul, graphs)
    positional_rebind.check_positional_merges(rerun, _REBIND)
    positional_rebind.check_positional_merges(
        _forward_slice(list(group.mats.values()), graphs), _MAT_REBIND
    )


def _stage_epilogue_vecs(group: MppGroup, plan: MppPlan, graphs: _Graphs) -> None:
    """Stage each input of a kept derived value that no element can
    recompute (:func:`_vec_input`) into a ``_mpp_vec`` tile, right after the
    input's definition, so ahead of every derived reader."""
    staged: dict[Node, Node] = {}
    for d in sorted(group.derived, key=lambda n: n.name):
        for a in list(d.all_input_nodes):
            if group.recomputable(a) or not _vec_input(group, a):
                continue
            vec = staged.get(a)
            if vec is None:
                vec = staged[a] = _stage_vec(a, None, graphs)
                plan.by_vec[vec] = group
            d.replace_input_with(a, vec)


def _update_reruns(group: MppGroup) -> list[Node]:
    """The nodes the update's element loop re-runs for its recomputable
    operands; the carry, the update's own nodes and a staged vector are not
    recomputable, so they contribute none."""
    return [
        n
        for u in group.update
        for a in u.all_input_nodes
        for n in group.recomputable.nodes(a)
    ]


def _check_uniform(node: Node, graphs: _Graphs) -> None:
    """Every thread of the threadgroup reaches ``node``: no enclosing if or
    while tests a per-element predicate.  A cooperative op, and the barriers of a
    staged tile or a materialization, need all of them.  Checking the op covers
    its group's seed and materializations, which sit in the graphs of its
    levels, each the op's graph or one enclosing it."""
    while True:
        gid = graphs.graph_id[node.graph]
        if gid in graphs.owners:
            node = graphs.owners[gid][0]
            per_element = (
                node.target is _tracing_ops._if
                and graphs.if_branch(node.args[1])[0].predicate_is_tensor
            )
        elif gid in graphs.whiles:
            node = graphs.whiles[gid]
            (output,) = graphs.info(node.args[0]).graph.find_nodes(op="output")
            (test,) = _seq(output.args[0])
            val = test.meta.get("val") if isinstance(test, Node) else None
            per_element = isinstance(val, torch.Tensor) and val.ndim > 0
        else:
            return
        if per_element:
            raise _unsupported(
                "a cooperative MPP op under an if or a while on a per-element "
                "predicate; every thread of the threadgroup must reach it"
            )


def _live_exit(node: Node, output: Node, graphs: _Graphs) -> bool:
    """Whether a getitem of the graph's owner reads one of ``output``'s slots
    that hold ``node``.  An if graph outputs every name it writes, so a slot
    no getitem reads is dead."""
    owner = graphs.owner(output)
    if owner is None:
        return True
    slots = {i for i, v in enumerate(_seq(output.args[0])) if v is node}
    if owner.target is _tracing_ops._if:
        gid = graphs.graph_id[output.graph]
        _, branches = graphs.if_branch(gid)
        side = 0 if gid == owner.args[1] else 1
        slots = {i for i, pair in enumerate(branches) if pair[side] in slots}
    return any(u.target is operator.getitem and u.args[1] in slots for u in owner.users)


def _forward_slice(starts: list[Node], graphs: _Graphs) -> list[Node]:
    """Every node that reads one of ``starts``, directly or through other
    nodes, control-flow arguments and outputs included."""
    out: list[Node] = []
    seen = set(starts)
    work = list(starts)
    while work:
        node = work.pop()
        for user in node.users:
            targets = [user]
            if (
                _tracing_ops.is_for_loop_target(user.target)
                or user.target is _tracing_ops._if
            ):
                for gid, args in (
                    [(user.args[0], user.args[3])]
                    if user.target is not _tracing_ops._if
                    else [(user.args[1], user.args[3]), (user.args[2], user.args[4])]
                ):
                    placeholders = list(
                        graphs.info(gid).graph.find_nodes(op="placeholder")
                    )
                    for i, a in enumerate(_seq(args)):
                        if a is node and i < len(placeholders):
                            targets.append(placeholders[i])
            elif user.op == "output":
                owner = graphs.owner(user)
                if owner is not None:
                    targets += [u for u in owner.users if u.target is operator.getitem]
            for t in targets:
                if t not in seen:
                    seen.add(t)
                    out.append(t)
                    work.append(t)
    return out


def check_mpp_launch(plan: MppPlan, block_dims: Sequence[int]) -> None:
    """Every thread joins each MPP op, whose scope is the launch's SIMD groups."""
    if not plan.by_matmul:
        return
    threads = math.prod(block_dims)
    if threads > 1024 or threads % 32:
        raise _unsupported(
            f"a cooperative MPP matmul in a threadgroup of {threads} threads; it "
            "needs whole 32-thread SIMD groups and at most 1024 threads (change "
            "num_threads or the block sizes)"
        )
    for m, n in plan.wide.values():
        if threads // 32 < _WIDE_MIN_SIMDGROUPS:
            raise _unsupported(
                f"an MPP matmul tile {m}x{n} on {threads // 32} SIMD groups; over "
                f"{_MAX_TILE[0]} rows or columns it needs {_WIDE_MIN_SIMDGROUPS} "
                "(raise num_warps)"
            )


def _plan() -> MppPlan:
    from .backend import MetalBackend

    backend = CompileEnvironment.current().backend
    assert isinstance(backend, MetalBackend) and backend.mpp_plan is not None
    return backend.mpp_plan


def _transposed(operand: _Operand) -> bool:
    """A staged tile is written row-major."""
    view = operand.view
    return view is not None and view.tensor.stride(view.positions[1]) != 1


def _sides(group: MppGroup) -> tuple[tuple[_Operand, tuple[int, int], int], ...]:
    """The lhs and rhs, each with its blocks and the position of K in them."""
    return (
        (group.lhs, (group.m_block, group.k_block), 1),
        (group.rhs, (group.k_block, group.n_block), 0),
    )


def _tile(cg: GenerateAST, blocks: Sequence[int]) -> list[int]:
    sizes = [cg.device_function.resolved_block_size(b) for b in blocks]
    out = [b for b in sizes if isinstance(b, int)]
    if len(out) != len(sizes):
        raise _unsupported(
            "an MPP matmul tile with a block size unknown at codegen time"
        )
    return out


def _expr(node: Node | int) -> str:
    if isinstance(node, int):
        return repr(node)
    code = node.meta["codegen"]
    return code.id if isinstance(code, ast.Name) else f"({ast.unparse(code)})"


@_decorators.codegen(full, "metal")
def _(state: CodegenState) -> ast.AST:
    node = state.fx_node
    assert node is not None
    group = _plan().by_seed.get(node)
    if group is None:
        return full._codegen["common"](state)  # pyrefly: ignore[missing-attribute]
    fill = node.args[1]
    assert isinstance(fill, (int, float))
    return _acc_expr(state.codegen, group, repr(float(fill)), None)


@_decorators.codegen(_mpp_coop, "metal")
def _(state: CodegenState) -> ast.AST:
    node = state.fx_node
    assert node is not None
    group = _plan().by_seed[node]
    cg = state.codegen
    blocks = (group.m_block, group.n_block)
    rows, cols = _tile(cg, blocks)
    buf = alloc_threadgroup_buffer("float", rows * cols, prefix="_mpp_seed")
    _stage_writes(cg, buf, [state.ast_arg(0)], torch.float32, blocks, cols)
    return _acc_expr(cg, group, None, buf)


@_decorators.codegen(_mpp_mat, "metal")
def _(state: CodegenState) -> ast.AST:
    """Write the cooperative tensor to a threadgroup tile (all threads), and
    read this thread's element of it in the Scalar layout."""
    node = state.fx_node
    assert node is not None
    group = _plan().by_mat[node]
    cg = state.codegen
    blocks = (group.m_block, group.n_block)
    rows, cols = _tile(cg, blocks)
    buf = alloc_threadgroup_buffer("float", rows * cols, prefix=MAT_TILE)
    coop = state.ast_arg(0)
    assert isinstance(coop, ast.Name)
    cg.add_statement(statement_from_string(f"{MAT}({coop.id}, {buf}, {cols})"))
    return cg.lift(_tile_read(cg, buf, blocks, "0.0"), prefix="_mpp_v")


@_decorators.codegen(_mpp_upd, "metal")
def _(state: CodegenState) -> ast.AST:
    """``_helion_mpp_update(carry, value)``: :func:`elementize` runs ``value``'s
    statements per element of the carry and writes each result back."""
    carry, value = state.ast_arg(0), state.ast_arg(1)
    assert isinstance(carry, ast.Name) and isinstance(value, ast.Name)
    return state.codegen.lift(
        expr_from_string(f"{UPDATE}({carry.id}, {value.id})"), prefix="_mpp_u"
    )


@_decorators.codegen(_mpp_hoist, "metal")
def _(state: CodegenState) -> ast.AST:
    """Stage a hoisted operand into its tile, before its loop."""
    node = state.fx_node
    assert node is not None
    group, side = _plan().by_hoist[node]
    operand, blocks, k_pos = _sides(group)[side]
    tile = _tile(state.codegen, blocks)
    tile[k_pos] *= group.k_factor
    operand.tile = _stage_tile(
        state.codegen, group, operand.dtype, state.ast_arg(0), blocks, tile, k_pos
    )
    return expr_from_string("0")


@_decorators.codegen(_mpp_round, "metal")
def _(state: CodegenState) -> ast.AST:
    return state.codegen.lift(
        expr_from_string(f"{ROUND}({{value}})", value=state.ast_arg(0)),
        prefix="_rne",
    )


@_decorators.codegen(_mpp_vec, "metal")
def _(state: CodegenState) -> ast.AST:
    """Stage a Scalar value over some of the group's blocks into a
    threadgroup tile, and read it back at this thread's own position:
    re-run in an element loop, the read gives the element's value."""
    node = state.fx_node
    assert node is not None
    group = _plan().by_vec[node]
    cg = state.codegen
    backend = CompileEnvironment.current().backend
    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)
    present = {_block(s) for s in val.size() if s != 1}
    blocks = [b for b in (group.m_block, group.n_block) if b in present]
    sizes = _tile(cg, blocks)
    # A float value is staged as float32, exact for 16-bit floats, so no
    # bfloat store reaches threadgroup memory (Metal miscompiles some inside
    # a loop); any other value in its own type.
    tile_dtype = torch.float32 if val.dtype.is_floating_point else val.dtype
    buf = alloc_threadgroup_buffer(
        backend.dtype_str(tile_dtype), math.prod(sizes), prefix=VEC_TILE
    )
    _stage_writes(cg, buf, [state.ast_arg(0)], tile_dtype, blocks, sizes[-1])
    zero = "0.0" if tile_dtype.is_floating_point else "0"
    read = _tile_read(cg, buf, blocks, zero)
    if val.dtype != tile_dtype:
        read = backend.cast_ast(read, val.dtype)
    return cg.lift(read, prefix="_mpp_w")


def _acc_expr(
    cg: GenerateAST, group: MppGroup, fill: str | None, seed: str | None
) -> ast.AST:
    """``_helion_mpp_acc(...)``: the group's op and cooperative accumulator, filled
    with the constant ``fill`` or from the staged tile ``seed``."""
    tile = _tile(cg, (group.m_block, group.n_block, group.k_block))
    tile[2] *= group.k_factor
    limits = (_WIDE_TILE, _WIDE_TILE, _MAX_TILE[2])
    if any(b > limit for b, limit in zip(tile, limits, strict=True)):
        raise _unsupported(
            f"an MPP matmul tile {tile[0]}x{tile[1]}x{tile[2]}: one op covers M, "
            f"N <= {_WIDE_TILE} and K <= {_MAX_TILE[2]}"
        )
    m, n, k = tile
    block_m, block_n = m, n
    for staged, size in ((group.lhs.staged, m), (group.rhs.staged, n)):
        if size == 1 and staged:
            raise _unsupported("a staged MPP operand over a size-1 M or N")
    m, n = _op_dims(group, m, n)
    if m > _MAX_TILE[0] or n > _MAX_TILE[1]:
        _plan().wide[group.gid] = (m, n)
    # MPP's static_asserts for a SIMD-group scope (MPPTensorOpsMatMul2dImpl.h).
    if m % 8 or n % 8 or (m % 16 and n % 16) or k % 16:
        raise _unsupported(
            f"an MPP matmul tile {m}x{n}x{k}: matmul2d needs M and N multiples of "
            "8, one of them of 16, and K a multiple of 16"
        )
    binds = _virtual_thread(cg, group)
    dtype_str = CompileEnvironment.current().backend.dtype_str
    dtypes = tuple(
        dtype_str(torch.float32 if o.float_tile else o.dtype)
        for o in (group.lhs, group.rhs)
    )
    staged = [int(o.staged) for o in (group.lhs, group.rhs)]
    return expr_from_string(
        f"{ACC}({group.gid}, {m}, {n}, {k}, {dtypes!r}, "
        f"{int(_transposed(group.lhs))}, {int(_transposed(group.rhs))}, "
        f"{fill}, [{', '.join(binds)}], {staged[0]}, {staged[1]}, "
        f"{block_m}, {block_n}, {seed!r})"
    )


def _op_dims(group: MppGroup, block_m: int, block_n: int) -> tuple[int, int]:
    """The op's M and N: each block, cut to its loop's static extent rounded
    up to 8 when one tile covers the loop.

    MPP computes every row and column of its descriptor, so a block past its
    loop's extent (M = 4 under ``hl.dot``'s 16-row minimum block) multiplies
    padding.  A row of the product depends only on its own lhs row, so the
    rows the op keeps are unchanged, and the cooperative tensor has no
    element past the op's dims for a store to reach.  A group with a
    materialization keeps its blocks: its Scalar reads cover the whole tile,
    and a reduction whose mask Helion drops (the masked value of a seed's
    fill, or of an mm under fast_math) reads the rows past the extent.  A
    cut dim never grows past its block, so a staged tile, sized by the
    block, still holds every row the op reads.  MPP needs one of M and N a
    multiple of 16: a dim this rule changed takes the extra 8, and two
    unchanged blocks are left to the caller's check.
    """
    env = CompileEnvironment.current()
    dims = []
    for block, size in ((group.m_block, block_m), (group.n_block, block_n)):
        extent = env.block_sizes[block].size
        if size == 1:
            size = 8
        elif isinstance(extent, int) and not group.mats:
            # A zero-trip loop still gets an op MPP accepts.
            size = min(size, -(-max(extent, 1) // 8) * 8)
        dims.append(size)
    m, n = dims
    if m % 16 and n % 16:
        if m != block_m:
            m += 8
        elif n != block_n:
            n += 8
    return m, n


def _virtual_thread(cg: GenerateAST, group: MppGroup) -> list[str]:
    """``(name, expr)`` bindings that make a thread act, inside the element
    loop, as the thread that holds element ``(_e_m, _e_n)`` in the Scalar
    layout: ``tid`` and the M and N lane variables."""
    comps = ["0", "0", "0"]
    binds: list[str] = []
    for block, e in ((group.m_block, "_e_m"), (group.n_block, "_e_n")):
        layout = _block_layout(cg, block)
        tid_e, lane_binds = layout.binds(e)
        binds += lane_binds
        if layout.axis is not None:
            # Each loop strategy gives its blocks their own axes, free blocks
            # take axes above every loop's, and every threaded ':' slice
            # its own (TileStrategyDispatch.thread_axis_for_strategy).
            assert comps[layout.axis] == "0"
            comps[layout.axis] = tid_e
    binds.insert(0, f"('tid', uint3({', '.join(comps)}))")
    return binds


@dataclasses.dataclass
class _BlockLayout:
    """Where a block's elements live in the Scalar layout, as CuTe's
    thread-value layouts describe a tile: element ``e`` is on thread ``t``
    of ``axis`` (None when one thread holds the block) and on lane ``l`` of
    the lane loop ``lane``.  A tile's lanes are blocked (``t = e / lanes``,
    ``l = e % lanes``); a ``:`` slice's are strided (``t = e % threads``,
    ``l = e / threads``)."""

    #: This thread's index within the tile.
    local: str
    axis: int | None
    lane: str | None = None
    lanes: int = 1
    threads: int = 1
    strided: bool = False

    def binds(self, e: str) -> tuple[str, list[str]]:
        """This block's ``tid`` component and lane binding for element *e*."""
        if self.lane is None:
            return e, []
        if self.strided:
            # A slice on one thread (threads 0) has no axis, so only its
            # lane, e / 1, is bound.
            t = max(self.threads, 1)
            return f"({e}) % {t}", [f"('{self.lane}', ({e}) / {t})"]
        return f"({e}) / {self.lanes}", [f"('{self.lane}', ({e}) % {self.lanes})"]


def _block_layout(cg: GenerateAST, block: int) -> _BlockLayout:
    """The Scalar layout of an MPP op's M, N or K block: a free
    ``hl.arange``, a ``:`` slice (a persistent reduction dimension) or a
    tile loop's block."""
    from ..tile_strategy import DeviceGridState
    from ..tile_strategy import DeviceLoopState
    from ..tile_strategy import FlattenedTileStrategy
    from .backend import MetalBackend
    from .reduction_strategy import MetalPaddedTwinStrategy
    from .reduction_strategy import MetalPersistentReductionStrategy

    env = CompileEnvironment.current()
    backend = env.backend
    assert isinstance(backend, MetalBackend)
    planned = backend.free_arange_plan.get(block)
    if planned is not None:
        from .free_arange import lane_name

        axis, _threads, lanes = planned
        if lanes == 1:
            return _BlockLayout(f"tid[{axis}]", axis)
        return _BlockLayout(f"tid[{axis}]", axis, lane_name(block), lanes=lanes)
    if env.block_sizes[block].reduction:
        strategy = cg.device_function.tile_strategy.block_id_to_strategy[(block,)]
        # The planner's block checks reject a padded twin (its fake size is
        # no op's block), and a matmul's ':' dim is never rolled into a
        # reduction loop or repeated (repeated_blocks takes only kernels
        # without one).
        assert isinstance(strategy, MetalPersistentReductionStrategy)
        assert not isinstance(strategy, MetalPaddedTwinStrategy)
        assert strategy.position_lane(0) is None
        axis, threads, lane = strategy.thread_lanes()
        return _BlockLayout(
            f"({cg.index_var(block)} - {cg.offset_var(block)})",
            axis,
            lane,
            threads=threads,
            strided=True,
        )
    states = cg.active_device_loops.get(block)
    if not states:
        raise _unsupported(
            f"an MPP matmul whose block {block} is neither a tile loop, a ':' slice "
            "nor a free hl.arange"
        )
    state = states[-1]
    assert isinstance(state, (DeviceLoopState, DeviceGridState))
    if isinstance(state.strategy, FlattenedTileStrategy):
        # Its offset is each thread's own index, not the tile's start.
        raise _unsupported(
            f"an MPP matmul whose block {block} is a single root hl.tile with no "
            "device loop inside (a flattened tile)"
        )
    # Metal configs have no cute_lane_layouts, so every lane loop is blocked.
    assert not backend.supports_config_key("cute_lane_layouts")
    lanes = [
        (var, extent)
        for var, extent in state.lane_loops
        if block in state.lane_loop_block_ids.get(var, ())
    ]
    if len(lanes) > 1 or any(
        state.lane_loop_block_ids[var] != {block} for var, _ in lanes
    ):
        raise _unsupported(
            f"an MPP matmul whose block {block} spans two lane loops or "
            "shares one with another block"
        )
    lane, extent = lanes[0] if lanes else (None, 1)
    return _BlockLayout(
        f"({cg.index_var(block)} - {cg.offset_var(block)})",
        state.block_thread_axes.get(block),
        lane,
        lanes=extent,
    )


def _check_threads(cg: GenerateAST, group: MppGroup) -> None:
    """Every thread of the scope joins one ``matmul2d`` op computing one M x N
    tile, so no other block may hold threads around it: threads that differ
    in another block's index would need another tile.  The exception is a
    ``:`` slice that is none of the group's values' dims: threads that differ
    only on it hold the same element of each, staging writes from their
    thread 0 (:func:`_stage_writes`) and the element loop runs as their
    thread 0."""
    from .backend import MetalBackend

    backend = CompileEnvironment.current().backend
    assert isinstance(backend, MetalBackend)
    allowed = _thread_blocks(group)
    for states in cg.active_device_loops.values():
        for st in states:
            for block, axis in st.block_thread_axes.items():
                if block not in allowed and st.thread_axis_sizes.get(axis, 1) > 1:
                    raise _unsupported(
                        f"block {block} is spread over threads around a cooperative "
                        "MPP matmul; only its M, N and K blocks may hold threads"
                    )
    for block, (_axis, threads, _lanes) in backend.free_arange_plan.items():
        if block not in allowed and threads > 1:
            raise _unsupported(
                f"free block {block} is spread over threads around a cooperative "
                "MPP matmul; only its M, N and K blocks may hold threads"
            )


def _thread_blocks(group: MppGroup) -> set[int]:
    """The blocks that may hold threads around the group's op
    (:func:`_check_threads`)."""
    reductions = {
        b
        for b, info in enumerate(CompileEnvironment.current().block_sizes)
        if info.reduction
    }
    return {group.m_block, group.n_block, group.k_block} | (
        reductions - _group_blocks(group)
    )


def _foreign_threads(group: MppGroup, threaded: set[int], graphs: _Graphs) -> bool:
    """Whether :func:`_check_threads` rejects the group's op at the config
    whose strategies spread the blocks ``threaded`` over threads: one of them
    is a grid block or a block of a loop the op runs in, and none of
    :func:`_thread_blocks`.  It counts every grid's blocks, which can only
    reject more.  Free blocks never reach here: their planner keeps each
    matmul in a loop over its own K (``free_arange._check_matmuls``), which an
    operand over another op's M and N can't share."""
    from ..device_ir import ForLoopGraphInfo

    around = {b for ids in HostFunction.current().device_ir.grid_block_ids for b in ids}
    for info in graphs.infos.values():
        if isinstance(info, ForLoopGraphInfo) and graphs.within(
            group.matmul.graph, info.graph
        ):
            around |= set(info.block_ids)
    return bool((threaded & around) - _thread_blocks(group))


def _group_blocks(group: MppGroup) -> set[int]:
    """The blocks among the dims of the op, its operands and their inputs,
    index tiles included.  The planner admits a seed, an update's operands,
    an epilogue's and a store's only over the group's own blocks or as 0-d
    values, so threads that differ on any other block hold the same element
    of every value the group stages or reads per element."""
    nodes = {group.matmul, *(n for n in _operands(group.matmul) if n is not None)}
    blocks: set[int] = set()
    for n in nodes:
        for v in (n, *n.all_input_nodes):
            val = v.meta.get("val")
            sizes = val.size() if isinstance(val, torch.Tensor) else (val,)
            # _block maps only ints and SymInts; a float kernel argument's
            # SymFloat has no block.
            blocks |= {b for s in sizes if (b := _block(s)) is not None}
    return blocks


def _check_batch(cg: GenerateAST, group: MppGroup) -> None:
    """A batched matmul is one rank-2 op, so its batch block must hold one
    index.  A block of size 1 has one thread and no lanes in every strategy,
    so its tile's offset is that index, the same for every thread."""
    block = group.batch_block
    if block is None:
        return
    if _tile(cg, [block]) != [1]:
        raise _unsupported(
            f"a batched MPP matmul whose batch block {block} is over 1 (one op "
            "per batch index)"
        )


def _anchor(node: Node | None) -> str | None:
    """The variable a load's index or mask statement assigns, if any."""
    assert node is not None
    code = node.meta["codegen"]
    return code.id if isinstance(code, ast.Name) else None


def _view_args(
    cg: GenerateAST, view: _View, k_block: int, value: object
) -> tuple[
    list[str],
    list[tuple[int, str | None]],
    list[bool],
    list[tuple[str, int, int, str | None]],
]:
    """``tensor, (origin, extent, size, stride) per dim`` for ``_helion_mpp_run``,
    the anchors through which :class:`_Copies` reads a base or a bound,
    whether each dim is live by construction (a tile from its loop offset to
    the tensor's end, a widened slice or a ``:``), and each outer dim's
    ``(index, size, stride, anchor)``, anchored at the operand's *value*,
    whose definition is the load or reads it.  Per-thread parts
    (``FactoredValue``) have no one defining statement, so their outer
    indices carry no anchor, and :class:`_Copies` rejects a reassigned one.

    A view reads its base, bound and outer indices at the op, inside the K
    loop, while the Scalar load reads them where its index and its mask are
    computed and at the load itself; each anchor names that statement's
    target.
    """
    fn = cg.device_function
    t = view.tensor
    value_anchor = value.id if isinstance(value, ast.Name) else None
    parts = [fn.tensor_arg(t).name]
    anchors: list[tuple[int, str | None]] = []
    always_live: list[bool] = []
    for dim, i in zip(view.dims, view.positions, strict=True):
        size, stride = int(t.size(i)), int(t.stride(i))
        if dim.full:
            parts += ["0", str(size), str(size), str(stride)]
            always_live.append(True)
            continue
        if dim.factor != 1:
            always_live.append(True)
            # A widened hl.arange covers [c * begin, c * end) of K, and the
            # load's lowering (factored_dims._check_extent) proved the dim is
            # c x the loop's end: the view's extent, the dim's size, is the
            # live K.
            info = cg.active_device_loops[dim.block_id][-1].block_id_to_info[
                dim.block_id
            ]
            assert size % dim.factor == 0
            assert info.is_end_matching(size // dim.factor)
            origin = f"{dim.factor} * {cg.offset_var(dim.block_id)}"
            parts += [origin, str(size), str(size), str(stride)]
            continue
        if dim.iota is None:
            origin = cg.offset_var(dim.block_id)
            info = cg.active_device_loops[dim.block_id][-1].block_id_to_info.get(
                dim.block_id
            )
            base = None if dim.base is None else _expr(dim.base)
            if base is not None:
                origin = f"{base} + {origin}"
                anchors.append((len(parts), _anchor(dim.index)))
            # The load masks K past the loop's end (the tile's mask), while the
            # view would sum those columns, so the K extent is the loop's end
            # from the same base.  Rows and columns past M's and N's loop ends
            # are computed but never stored: the element loop re-runs the
            # store's mask.
            if dim.block_id == k_block and not (
                base is None and info is not None and info.is_end_matching(size)
            ):
                assert info is not None and info.end_var_name is not None
                extent = (
                    info.end_var_name
                    if base is None
                    else f"{base} + {info.end_var_name}"
                )
                if base is not None:
                    anchors.append((len(parts) + 1, _anchor(dim.index)))
                always_live.append(False)
            else:
                extent = str(size)
                always_live.append(base is None)
        else:
            always_live.append(False)
            origin = "0" if dim.base is None else _expr(dim.base)
            if dim.bound is None:
                extent = str(size)
            elif isinstance(dim.bound, int):
                # Clamped here: the walker prints a literal extent with no min.
                extent = str(max(0, min(dim.bound, size)))
            else:
                extent = _expr(dim.bound)
            if dim.base is not None:
                anchors.append((len(parts), _anchor(dim.index)))
            if isinstance(dim.bound, Node):
                anchors.append((len(parts) + 1, _anchor(dim.mask)))
        parts += [origin, extent, str(size), str(stride)]
    outer: list[tuple[str, int, int, str | None]] = []
    for i, x in view.outer:
        size, stride = int(t.size(i)), int(t.stride(i))
        if isinstance(x, int):
            # A negative literal counts from the end, as the Scalar load reads it.
            outer.append((str(x + size if x < 0 else x), size, stride, None))
            continue
        val = x.meta["val"]
        # A SymInt prints as the Scalar load prints its index.
        expr = fn.literal_expr(val) if isinstance(val, torch.SymInt) else _expr(x)
        outer.append((expr, size, stride, value_anchor))
    if view.batch is not None:
        i, block = view.batch
        # The batch block holds one index (_check_batch): its tile's offset.
        outer.append((cg.offset_var(block), int(t.size(i)), int(t.stride(i)), None))
    return parts, anchors, always_live, outer


def planned(node: Node | None) -> bool:
    """Whether the MPP planner lowers *node* cooperatively."""
    from .backend import MetalBackend

    backend = CompileEnvironment.current().backend
    assert isinstance(backend, MetalBackend)
    return backend.mpp_plan is not None and node in backend.mpp_plan.by_matmul


def fenced(node: Node | None) -> bool:
    """Whether *node* is a load the MPP planner orders after this
    threadgroup's earlier device writes."""
    from .backend import MetalBackend

    backend = CompileEnvironment.current().backend
    assert isinstance(backend, MetalBackend)
    return backend.mpp_plan is not None and node in backend.mpp_plan.fenced


def _in_tile(
    cg: GenerateAST, blocks: Sequence[int]
) -> tuple[list[_BlockLayout], list[str]]:
    """Each block's layout, and the checks that this thread's index within
    the tile lies inside it on the blocks where it may not: on every block
    when a wider sibling can launch surplus threads (``force_tile_mask()``),
    else only on ``:`` slices.  Under a uniform launch a tile loop's threads
    times lanes are exactly its block, so its index is always inside."""
    force = CompileEnvironment.current().backend.force_tile_mask()
    layouts = [_block_layout(cg, b) for b in blocks]
    checks = [
        f"({layout.local}) < {n}"
        for layout, n in zip(layouts, _tile(cg, blocks), strict=True)
        if force or layout.strided
    ]
    return layouts, checks


def _stage_writes(
    cg: GenerateAST,
    buf: str,
    parts: Sequence[ast.AST],
    dtype: torch.dtype,
    blocks: Sequence[int],
    cols: int,
    k_pos: int | None = None,
    k_factor: int = 1,
    owner: bool = True,
) -> None:
    """Write the elements this thread holds into the row-major tile ``buf``
    over one or two blocks at its index within the tile (part ``r`` of a
    factored K at ``c * k + r``).  A thread whose index on a block may lie
    past the block (:func:`_in_tile`) writes only inside it, and of the
    threads that hold the same element, because they differ only on another
    block's axis, the one at 0 writes it.  The write's last argument names
    those pinned axes, and the owner test is its predicate's last term.  With
    ``owner`` False there is no owner test: the write runs in another
    group's element loop, which gives each element to one thread."""
    backend = CompileEnvironment.current().backend
    layouts, pred = _in_tile(cg, blocks)
    owned = {layout.axis for layout in layouts if layout.axis is not None}
    pinned = tuple(a for a in range(3) if a not in owned)
    test = " and ".join(f"tid[{a}] == 0" for a in pinned)
    if test and owner:
        # A name assigned in the preamble, not tid[...] in the write or beside
        # it: the shared lane-loop placement takes a subscripted name for a
        # tensor access, which would make the lane nest around it inexact.
        name = cg.device_function.new_var("_mpp_owner")
        cg.device_function.preamble.append(statement_from_string(f"{name} = {test}"))
        pred.append(name)
    for r, part in enumerate(parts):
        index = [layout.local for layout in layouts]
        if k_pos is not None and k_factor != 1:
            index[k_pos] = f"{k_factor} * {index[k_pos]} + {r}"
        flat = _flat(index, cols)
        cg.add_statement(
            statement_from_string(
                f"{STAGE}({buf}, {flat}, {{value}}, {' and '.join(pred) or 'True'}, {pinned!r})",
                value=backend.cast_ast(part, dtype),
            )
        )


def _flat(index: Sequence[str], cols: int) -> str:
    """The row-major index into a tile over one or two blocks."""
    if len(index) == 1:
        return index[0]
    return f"({index[0]}) * {cols} + ({index[1]})"


def _tile_read(cg: GenerateAST, buf: str, blocks: Sequence[int], zero: str) -> ast.AST:
    """This thread's element of the row-major tile ``buf`` over ``blocks``.
    A thread whose index on a block may lie past the block (:func:`_in_tile`)
    reads ``zero`` there, so its read stays inside the tile; its value is
    masked off downstream."""
    layouts, inside = _in_tile(cg, blocks)
    locs = [layout.local for layout in layouts]
    read = f"{buf}[{_flat(locs, _tile(cg, blocks)[-1])}]"
    if not inside:
        return expr_from_string(read)
    return expr_from_string(f"({read} if {' and '.join(inside)} else {zero})")


def _stage(
    cg: GenerateAST,
    group: MppGroup,
    dtype: torch.dtype,
    value: object,
    blocks: tuple[int, int],
    tile: list[int],
    k_pos: int,
    owner: bool = True,
) -> list[str]:
    """Write a staged operand into a threadgroup tile and return its
    ``_helion_mpp_run`` arguments."""
    buf = _stage_tile(cg, group, dtype, value, blocks, tile, k_pos, owner)
    return _staged_args(cg, group, buf, tile, k_pos)


def _stage_tile(
    cg: GenerateAST,
    group: MppGroup,
    dtype: torch.dtype,
    value: object,
    blocks: tuple[int, int],
    tile: list[int],
    k_pos: int,
    owner: bool = True,
) -> str:
    """Write a staged operand into a new threadgroup tile; return its name."""
    backend = CompileEnvironment.current().backend
    rows, cols = tile
    buf = alloc_threadgroup_buffer(
        backend.dtype_str(dtype), rows * cols, prefix="_mpp_tile"
    )
    if isinstance(value, FactoredValue):
        assert (value.block_id, value.factor, value.dim) == (
            blocks[k_pos],
            group.k_factor,
            k_pos,
        )
        parts: Sequence[ast.AST] = value.parts
    elif group.k_factor != 1:
        # The planner admits a K of c * u by its size; a value of that size
        # that factored_dims doesn't build as parts (hl.full, say) has no
        # thread layout.
        raise _unsupported(
            f"a staged MPP operand over a K of {group.k_factor} * a block that "
            "is not held as per-thread parts (only a load at a widened "
            "hl.arange() or a merged torch.stack is)"
        )
    else:
        assert isinstance(value, ast.AST)
        parts = [value]
    _stage_writes(cg, buf, parts, dtype, blocks, cols, k_pos, group.k_factor, owner)
    return buf


def _staged_args(
    cg: GenerateAST, group: MppGroup, buf: str, tile: list[int], k_pos: int
) -> list[str]:
    """A staged tile's ``_helion_mpp_run`` arguments."""
    rows, cols = tile
    # Every handle of the op ends at the K tile's live length.  MPP takes K
    # from the rhs handle alone and reads the lhs up to it (measured on every
    # mix of device and threadgroup handles; MPPTensorOpsMatMul2d.h doesn't
    # say which handle), and a view's K extent is the live K (_view_args'
    # end rules).
    info = cg.active_device_loops[group.k_block][-1].block_id_to_info[group.k_block]
    extents = [str(rows), str(cols)]
    extents[k_pos] = (
        f"{group.k_factor} * ({info.end_var_name} - {cg.offset_var(group.k_block)})"
    )
    return [buf, "0", extents[0], str(rows), str(cols), "0", extents[1], str(cols), "1"]


def _codegen_matmul(
    cg: GenerateAST, node: Node, acc: object, values: tuple[object, object]
) -> ast.AST:
    group = _plan().by_matmul[node]
    _check_threads(cg, group)
    _check_batch(cg, group)
    if acc is None:
        # A matmul with no acc writes a fresh cooperative tensor: zeros, then
        # the op accumulates the product into them.
        assert group.seed is None
        acc = cg.lift(_acc_expr(cg, group, "0.0", None), prefix="_mpp_c")
    assert isinstance(acc, ast.Name)
    tiles: list[int] = []
    tile_dims: list[str] = []
    parts: list[list[str]] = []
    anchors: list[tuple[int, str | None]] = []
    outers: list[list[tuple[str, int, int, str | None]]] = []
    pos = 2
    for (operand, blocks, k_pos), value, fx_value in zip(
        _sides(group), values, _operands(node)[1:], strict=True
    ):
        tile = _tile(cg, blocks)
        tile[k_pos] *= group.k_factor
        tiles += tile
        view = operand.view
        if operand.tile is not None:
            op_parts = _staged_args(cg, group, operand.tile, tile, k_pos)
            tile_dims += ["1", "1"]
            outers.append([])
        elif operand.float_tile:
            assert isinstance(value, ast.AST)
            rounded = expr_from_string(f"{ROUND}({{value}})", value=value)
            op_parts = _stage(cg, group, torch.float32, rounded, blocks, tile, k_pos)
            tile_dims += ["1", "1"]
            outers.append([])
        elif view is None:
            if operand.elements is not None:
                _check_source(cg, group, operand.elements, fx_value, blocks, k_pos)
            op_parts = _stage(
                cg,
                group,
                operand.dtype,
                value,
                blocks,
                tile,
                k_pos,
                owner=operand.elements is None,
            )
            tile_dims += ["1", "1"]
            outers.append([])
        else:
            # Each handle is rebased at its tile in 64 bits, so every offset
            # MPP computes inside it (in int32) is below the tile's span; the
            # outer dims are folded into the base.
            strides = [view.tensor.stride(p) for p in view.positions]
            span = 1 + sum((b - 1) * s for b, s in zip(tile, strides, strict=True))
            if span >= 2**31:
                raise _unsupported(
                    f"an MPP view whose tile spans {span} elements (MPP tensor "
                    "handles index in int32)"
                )
            op_parts, op_anchors, flags, outer = _view_args(
                cg, view, group.k_block, value
            )
            anchors += [(pos + i, a) for i, a in op_anchors]
            tile_dims += [str(int(f)) for f in flags]
            outers.append(outer)
        parts.append(op_parts)
        pos += len(op_parts)
    # Anchors are names, not strings, so Helion's phi renames (applied to
    # ast.Name nodes after codegen) reach them.
    anchor_list = ", ".join(f"({p}, {a})" for p, a in anchors)
    spaces = f"[{int(group.lhs.staged)}, {int(group.rhs.staged)}]"
    outer_list = ", ".join(
        "[" + "".join(f"({e}, {s}, {st}, {a}), " for e, s, st, a in o) + "]"
        for o in outers
    )
    return expr_from_string(
        f"{RUN}({acc.id}, {group.gid}, {', '.join(parts[0])}, {', '.join(parts[1])}, "
        f"{', '.join(tile_dims)}, {', '.join(map(str, tiles))}, "
        f"anchors=[{anchor_list}], spaces={spaces}, outer=[{outer_list}])"
    )


def _check_source(
    cg: GenerateAST,
    group: MppGroup,
    source: int,
    value: Node,
    blocks: tuple[int, int],
    k_pos: int,
) -> None:
    """Check an operand staged from the source group's element loop.  That
    loop writes the elements of its op dims, which ``_op_dims`` may cut below
    the block; the consumer reads its M or N side over its own op dims, which
    must not exceed the source's."""
    (h,) = [x for x in _plan().by_matmul.values() if x.gid == source]
    assert _elementwise_of(value, h), value
    hm, hn = _op_dims(h, *_tile(cg, (h.m_block, h.n_block)))
    gm, gn = _op_dims(group, *_tile(cg, (group.m_block, group.n_block)))
    side = blocks[1 - k_pos]
    written = {h.m_block: hm, h.n_block: hn}[side]
    read = gm if k_pos == 1 else gn
    if read > written:
        raise _unsupported(
            "an MPP operand staged from another op's elements over a tile that "
            "op cuts to its loop's extent"
        )


def _codegen_aten_matmul(ctx: LoweringContext, node: Node) -> object:
    from ..generate_ast import GenerateAST

    cg = ctx.cg
    assert isinstance(cg, GenerateAST)
    acc, lhs, rhs = _operands(node)
    if not planned(node):
        # The planner plans every matmul with an acc, or raises.
        assert acc is None
        raise _unsupported(
            "a matmul with no acc over a tiled K, whose result is each K tile's "
            "product alone; accumulate it with torch.addmm(acc, a, b), or take K "
            "whole with a ':' slice"
        )
    return _codegen_matmul(
        cg, node, None if acc is None else ctx.env[acc], (ctx.env[lhs], ctx.env[rhs])
    )


for _lowering in (addmm_lowering, mm_lowering, bmm_lowering, baddbmm_lowering):
    _lowering.register_codegen("metal")(_codegen_aten_matmul)


def codegen_dot_metal(state: CodegenState) -> ast.AST:
    """``hl.dot(lhs, rhs, acc=acc)`` of a planned group."""
    from ..generate_ast import GenerateAST

    node = state.fx_node
    assert node is not None
    cg = state.codegen
    assert isinstance(cg, GenerateAST)
    lhs, rhs, acc = state.ast_args[0], state.ast_args[1], state.ast_args[2]
    return _codegen_matmul(cg, node, acc, (lhs, rhs))


# MSL assembly: element loops for stores of cooperative values.


def _node_name(node: ast.AST) -> str:
    assert isinstance(node, ast.Name)
    return node.id


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id
    return None


def names_read(node: ast.AST) -> set[str]:
    """The names a statement or expression reads; an element loop's own
    definitions are not reads."""
    if isinstance(node, ast.For) and _call_name(node.iter) == ELEMENTS:
        inner: set[str] = set()
        defs: set[str] = set()
        for s in node.body:
            inner |= names_read(s)
            if (t := _target(s)) is not None:
                defs.add(t)
            elif isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
                if _call_name(s.value) == BIND:
                    defs.add(ast.literal_eval(s.value.args[0]))
        return (inner - defs) | names_read(node.iter)
    if isinstance(node, (ast.For, ast.If)):
        out = names_read(node.iter if isinstance(node, ast.For) else node.test)
        for s in (*node.body, *node.orelse):
            out |= names_read(s)
        return out
    return {
        sub.id
        for sub in ast.walk(node)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
    }


def _target(stmt: ast.stmt) -> str | None:
    assigned = _assigned(stmt)
    return assigned[0] if assigned is not None else None


def _assigned(stmt: ast.stmt) -> tuple[str, ast.expr] | None:
    """The name and value of a single-name assignment."""
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return stmt.targets[0].id, stmt.value
    return None


def _is_store(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "store"
    )


def _is_stage(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.Expr) and _call_name(stmt.value) == STAGE


def _rename(expr: ast.expr, names: dict[str, str]) -> ast.expr:
    class Rename(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            if isinstance(node.ctx, ast.Load) and node.id in names:
                return ast.Name(names[node.id], ast.Load())
            return node

    return Rename().visit(expr)


def _anchors(call: ast.Call) -> list[tuple[int, str | None]]:
    """``_helion_mpp_run``'s ``anchors=[(argument position, name or None), ...]``."""
    (kw,) = [k for k in call.keywords if k.arg == "anchors"]
    assert isinstance(kw.value, ast.List)
    out: list[tuple[int, str | None]] = []
    for e in kw.value.elts:
        assert isinstance(e, ast.Tuple)
        pos, name = e.elts
        assert isinstance(pos, ast.Constant) and isinstance(pos.value, int)
        out.append((pos.value, name.id if isinstance(name, ast.Name) else None))
    return out


def _outer(call: ast.Call) -> list[list[list[ast.expr]]]:
    """``_helion_mpp_run``'s ``outer=[[(index, size, stride, anchor), ...] per
    operand]``, each entry's elements as a mutable list."""
    (kw,) = [k for k in call.keywords if k.arg == "outer"]
    assert isinstance(kw.value, ast.List)
    out: list[list[list[ast.expr]]] = []
    for per_operand in kw.value.elts:
        assert isinstance(per_operand, ast.List)
        out.append([])
        for e in per_operand.elts:
            assert isinstance(e, ast.Tuple)
            out[-1].append(e.elts)
    return out


def _assigns(body: list[ast.stmt]) -> list[ast.Assign]:
    return [
        n
        for n in ast.walk(ast.Module(body=body, type_ignores=[]))
        if isinstance(n, ast.Assign)
    ]


class _Copies:
    """Copies of reassigned names, taken where a Scalar statement reads them.

    Helion writes a loop carry or an if output back into its variable's name,
    so a name read later may hold a newer value.  An element loop re-runs a
    Scalar statement at the store, and an MPP view reads its base and bound
    at the op, so each statement they may evaluate reads every name the
    kernel assigns more than once through a copy taken just before it: the
    value the Scalar statement read.

    The *recomputed* names are the bound names, the cooperative names, the
    reads of a staged vector's tile and every name assigned from one of them,
    never a collective's target (a leaf); the element loop re-runs exactly
    those.
    """

    def __init__(
        self,
        body: list[ast.stmt],
        bound: set[str],
        coop: set[str],
        leaves: set[str],
        vec_reads: set[str],
    ) -> None:
        assigns = _assigns(body)
        writes: dict[str, int] = {}
        self.defs: dict[str, list[ast.Assign]] = {}
        anchors: set[str] = set()
        for a in assigns:
            for t in a.targets:
                if isinstance(t, ast.Name):
                    writes[t.id] = writes.get(t.id, 0) + 1
            if (t := _target(a)) is not None:
                self.defs.setdefault(t, []).append(a)
            if isinstance(a.value, ast.Call) and _call_name(a.value) == RUN:
                anchors |= {name for _, name in _anchors(a.value) if name is not None}
                anchors |= {
                    e[3].id
                    for o in _outer(a.value)
                    for e in o
                    if isinstance(e[3], ast.Name)
                }
        self.reassigned = {n for n, c in writes.items() if c > 1}
        self.leaves = leaves
        # A staged vector's read runs after its element loop's leading
        # barrier, so the loop re-runs it, even where its index is the same
        # for every element (a block of one row).
        self.recomputed = bound | coop | vec_reads
        changed = True
        while changed:
            changed = False
            for a in assigns:
                t = _target(a)
                if (
                    t is not None
                    and t not in self.recomputed
                    and t not in leaves
                    and names_read(a.value) & self.recomputed
                ):
                    self.recomputed.add(t)
                    changed = True
        self.chain = self._closure((self.recomputed - bound - coop) | anchors)
        self.taken: dict[tuple[str, str], str] = {}

    def _closure(self, names: set[str]) -> set[str]:
        """``names`` and every name their definitions read, stopping at a
        reassigned name (the value a copy is taken of) and at a leaf."""
        out: set[str] = set()
        work = list(names)
        while work:
            n = work.pop()
            if n in out:
                continue
            out.add(n)
            if n in self.reassigned or n in self.leaves:
                continue
            for a in self.defs.get(n, ()):
                work += names_read(a.value)
        return out

    def take(self, s: ast.stmt) -> list[ast.stmt]:
        """The copies to emit before ``s``, which is rewritten to read them."""
        t = _target(s)
        if t is None or t not in self.chain or t in self.reassigned or t in self.leaves:
            return []
        assert isinstance(s, ast.Assign)
        names = sorted(
            r
            for r in names_read(s.value)
            if r in self.reassigned and r not in self.recomputed
        )
        out: list[ast.stmt] = []
        renames: dict[str, str] = {}
        for r in names:
            name = f"_mpp_snap_{len(self.taken)}"
            out.append(
                ast.Assign([ast.Name(name, ast.Store())], ast.Name(r, ast.Load()))
            )
            renames[r] = name
            self.taken[(t, r)] = name
        if renames:
            s.value = _rename(s.value, renames)
        return out

    def rewrite_run(self, call: ast.Call) -> None:
        """Read each view base, bound and outer index through the copy taken
        where the load's index or mask is computed, and drop the anchors."""
        anchors = _anchors(call)
        call.keywords = [k for k in call.keywords if k.arg != "anchors"]

        def reread(expr: ast.expr, anchor: str | None) -> ast.expr:
            reach = self._closure({anchor}) if anchor is not None else set()
            renames: dict[str, str] = {}
            for r in names_read(expr) & self.reassigned:
                copies = {self.taken[(t, r)] for t in reach if (t, r) in self.taken}
                if len(copies) != 1:
                    raise _unsupported(
                        f"an MPP view reads {r}, which the kernel reassigns, away "
                        "from the statement that defines the load's index or mask"
                    )
                (renames[r],) = copies
            return _rename(expr, renames)

        for pos, anchor in anchors:
            call.args[pos] = reread(call.args[pos], anchor)
        for entries in _outer(call):
            for entry in entries:
                name = entry[3].id if isinstance(entry[3], ast.Name) else None
                entry[0] = reread(entry[0], name)
                del entry[3]


#: MSL Standard Library functions that exchange data between threads or
#: order them: SIMD-group and quad-group functions, and barriers.
_CROSS_THREAD_CALLS = ("simd_", "quad_", "threadgroup_barrier", "simdgroup_barrier")
#: Helion's markers for MPP ops and regions, and for lane folds and lane
#: reductions, which combine the iterations of a lane loop.
_CROSS_THREAD_MARKERS = ("_helion_mpp_", "_metal_lane_")


def _cross_thread(node: ast.AST, tg_buffers: frozenset[str]) -> bool:
    """Whether an MSL statement or expression exchanges data between threads.

    In a Metal kernel body that is a call into ``helion_red`` (every Metal
    reduction's combine goes through it, as ``metal_jit._uses_reductions``
    relies on), a read or write of a threadgroup buffer, a Helion MPP or
    lane marker, or an MSL SIMD, quad or barrier function.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and (
            sub.id == REDUCTION_NAMESPACE or sub.id in tg_buffers
        ):
            return True
        if isinstance(sub, ast.Call):
            f = sub.func
            name = (
                f.attr
                if isinstance(f, ast.Attribute)
                else f.id
                if isinstance(f, ast.Name)
                else ""
            )
            if name.startswith(_CROSS_THREAD_CALLS + _CROSS_THREAD_MARKERS):
                return True
    return False


def _collectives(body: list[ast.stmt], tg_buffers: frozenset[str]) -> set[str]:
    """Names assigned by a statement that exchanges data between threads."""
    out: set[str] = set()
    for a in _assigns(body):
        if _cross_thread(a.value, tg_buffers):
            out |= {t.id for t in a.targets if isinstance(t, ast.Name)}
    return out


def _materialized(body: list[ast.stmt], tg_buffers: frozenset[str]) -> set[str]:
    """Names that hold this thread's own elements of a materialized MPP
    result: a read of a materialization tile, and every name computed from
    one by a statement that exchanges no data between threads.  The planner
    never lets an element loop read one; a collective's target, which the
    planner admits only when it is 0-d, ends the chain."""
    tiles = {b for b in tg_buffers if b.startswith(MAT_TILE)}
    out: set[str] = set()
    assigns = _assigns(body)
    changed = True
    while changed:
        changed = False
        for a in assigns:
            t = _target(a)
            if t is None or t in out:
                continue
            reads = names_read(a.value)
            if reads & tiles or (
                reads & out and not _cross_thread(a.value, tg_buffers)
            ):
                out.add(t)
                changed = True
    return out


def elementize(
    body: list[ast.stmt],
    tensor_dtypes: Mapping[str, torch.dtype],
    tg_buffers: frozenset[str],
    tg_size: Sequence[int] | None,
) -> list[ast.stmt]:
    """Move each store of a cooperative value into a loop over the
    cooperative tensor's elements; ``tensor_dtypes`` are the kernel's tensor
    parameters' dtypes, ``tg_buffers`` its threadgroup buffer names and
    ``tg_size`` its threadgroup's threads per axis.

    The loop binds the Scalar layout's thread of each element and re-runs the
    recomputed statements the store reads, in body order.  A collective's
    target (a leaf) is read as it stands: the loop's threads diverge, so a
    collective can't run there, and the planner admits such a value only when
    it is 0-d, the same on every thread.
    """
    groups: dict[str, ast.Call] = {}  # cooperative root -> its _helion_mpp_acc call
    alias: dict[str, str] = {}  # cooperative name -> root

    def scan(stmts: list[ast.stmt]) -> None:
        for s in stmts:
            if (assigned := _assigned(s)) is not None:
                t, v = assigned
                if isinstance(v, ast.Call) and _call_name(v) == ACC:
                    groups[t] = v
                    alias[t] = t
                elif isinstance(v, ast.Call) and _call_name(v) in (RUN, UPDATE):
                    alias[t] = alias[_node_name(v.args[0])]
                elif isinstance(v, ast.Name) and v.id in alias:
                    alias[t] = alias[v.id]
            if isinstance(s, (ast.For, ast.While, ast.If)):
                scan(s.body)
                scan(s.orelse)

    scan(body)
    if not groups:
        return body
    # A staged vector's tile is written before the element loop that reads
    # it (an update's or a store's) and read only inside it, with a barrier
    # on each side, so its reads exchange nothing at the loop and are re-run
    # like any load.
    vec_bufs = frozenset(b for b in tg_buffers if b.startswith(VEC_TILE))
    tg_buffers = tg_buffers - vec_bufs
    vec_reads = {
        t
        for a in _assigns(body)
        if (t := _target(a)) and names_read(a.value) & vec_bufs
    }
    leaves = _collectives(body, tg_buffers) - set(alias)
    per_thread = _materialized(body, tg_buffers)
    bound = {_bind_name(b) for acc in groups.values() for b in _binds(acc)}
    copies = _Copies(body, bound, set(alias), leaves, vec_reads)
    derived: dict[str, str] = {}  # recomputed name -> its cooperative root

    def roots_of(names: set[str]) -> set[str]:
        return {alias[n] for n in names if n in alias} | {
            derived[n] for n in names if n in derived
        }

    def process(stmts: list[ast.stmt], visible: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        seen = list(visible)
        for s in stmts:
            t, value = _assigned(s) or (None, None)
            if isinstance(value, ast.Call) and _call_name(value) == RUN:
                copies.rewrite_run(value)
            if isinstance(value, ast.Call) and _call_name(value) == UPDATE:
                assert t is not None
                carry, new = (_node_name(a) for a in value.args)
                root = alias[carry]
                # The planner's update chain is a value path from the carry.
                assert derived.get(new) == root, (new, root)
                put = ast.Expr(
                    ast.Call(ast.Name(PUT, ast.Load()), [ast.Name(new, ast.Load())], [])
                )
                out.append(element_loop(root, put, seen))
                alias_stmt = ast.Assign(
                    [ast.Name(t, ast.Store())], ast.Name(carry, ast.Load())
                )
                ast.fix_missing_locations(alias_stmt)
                out.append(alias_stmt)
                continue
            if (
                (isinstance(value, ast.Call) and _call_name(value) in (ACC, RUN))
                or (isinstance(value, ast.Name) and value.id in alias)
                or _is_materialize(s)
            ):
                out.append(s)
                seen.append(s)
                continue
            out += copies.take(s)
            roots = (
                roots_of(names_read(s))
                if not isinstance(s, (ast.For, ast.If))
                else set()
            )
            if len(roots) > 1:
                raise _unsupported("an element-wise op on two cooperative MPP results")
            if t is not None and roots:
                (derived[t],) = roots
                seen.append(s)
                continue
            if roots:
                if not (_is_store(s) or _is_stage(s)):
                    raise _unsupported(
                        "a cooperative MPP result reaches a use that needs it in "
                        "the Scalar layout without a materialization"
                    )
                (root,) = roots
                out.append(element_loop(root, s, seen))
                continue
            if isinstance(s, ast.For):
                s.body = process(s.body, seen)
            elif isinstance(s, ast.If):
                s.body = process(s.body, seen)
                s.orelse = process(s.orelse, seen)
            out.append(s)
            seen.append(s)
        return out

    def element_loop(root: str, store_stmt: ast.stmt, seen: list[ast.stmt]) -> ast.stmt:
        binds = {_bind_name(b): b.elts[1] for b in _binds(groups[root])}
        defs = {t: s for s in seen if (t := _target(s)) is not None}
        chosen: dict[int, ast.stmt] = {}

        def need(name: str) -> None:
            if name in per_thread:
                raise _unsupported(
                    f"an element loop over a cooperative MPP result reads {name}, "
                    "this thread's own element of a materialized MPP result"
                )
            # Re-run exactly the recomputed names; any other name holds one
            # value for every element of this thread and is read as it stands.
            if name in binds or name in alias or name not in copies.recomputed:
                return
            s = defs.get(name)
            if s is None or name in copies.reassigned:
                # One definition of a reassigned name is not the value the
                # store read, and a definition outside the store's blocks
                # can't be re-run here.
                raise _unsupported(
                    f"a statement re-run per cooperative MPP element reads {name}, a "
                    "per-thread value the kernel reassigns or defines in another "
                    "block"
                )
            if id(s) not in chosen:
                chosen[id(s)] = s
                for r in names_read(s):
                    need(r)

        for r in names_read(store_stmt):
            need(r)
        last: ast.AST = store_stmt
        if isinstance(store_stmt, ast.Expr) and _call_name(store_stmt.value) == PUT:
            assert isinstance(store_stmt.value, ast.Call)
            last = store_stmt.value.args[0]
        checked: list[ast.AST] = list(chosen.values())
        # A staging write's threadgroup tile is what it exists to write.
        if not _is_stage(store_stmt):
            checked.append(last)
        for s in checked:
            if _cross_thread(s, tg_buffers):
                raise _unsupported(
                    "a statement re-run per cooperative MPP element holds a "
                    f"cross-thread operation: {ast.unparse(s)[:120]}"
                )
        order = {id(s): i for i, s in enumerate(seen)}
        loop_body: list[ast.stmt] = [
            ast.Expr(
                ast.Call(
                    ast.Name(BIND, ast.Load()), [ast.Constant(n), copy.deepcopy(e)], []
                )
            )
            for n, e in binds.items()
        ]
        loop_body += [
            copy.deepcopy(s)
            for s in sorted(chosen.values(), key=lambda s: order[id(s)])
        ]
        loop_body.append(copy.deepcopy(store_stmt))
        iter_args: list[ast.expr] = [ast.Name(root, ast.Load())]
        if any(names_read(s) & vec_bufs for s in chosen.values()):
            # Barriers on each side: after the vector's staging writes, and
            # before the next writes to its tile.
            iter_args.append(ast.Constant(True))
        return ast.For(
            target=ast.Name("_it", ast.Store()),
            iter=ast.Call(ast.Name(ELEMENTS, ast.Load()), iter_args, []),
            body=loop_body,
            orelse=[],
        )

    result = _run_lane_loops_once(
        process(body, []), set(alias), tensor_dtypes, tg_buffers, tg_size
    )
    # The planner gives a staged vector only update and kept derived nodes
    # as readers, which run in element loops, between their barriers.
    assert not _scalar_reads(result) & vec_bufs
    return result


def _scalar_reads(stmts: list[ast.stmt]) -> set[str]:
    """The names statements read outside element loops."""
    out: set[str] = set()
    for s in stmts:
        if _is_elements(s):
            continue
        if isinstance(s, (ast.For, ast.If)):
            out |= names_read(s.iter if isinstance(s, ast.For) else s.test)
            out |= _scalar_reads([*s.body, *s.orelse])
        elif (
            isinstance(s, ast.Expr)
            and isinstance(s.value, ast.Call)
            and _call_name(s.value) == STAGE
        ):
            # A staging write names its tile; it doesn't read it.
            out |= set().union(*map(names_read, s.value.args[1:]))
        else:
            out |= names_read(s)
    return out


def _binds(acc: ast.Call) -> list[ast.Tuple]:
    """``_helion_mpp_acc``'s ``[(name, expr), ...]`` virtual-thread bindings."""
    binds = acc.args[8]
    assert isinstance(binds, ast.List)
    return binds.elts  # pyrefly: ignore[bad-return]


def _bind_name(bind: ast.Tuple) -> str:
    name = bind.elts[0]
    assert isinstance(name, ast.Constant) and isinstance(name.value, str)
    return name.value


def _is_materialize(s: ast.stmt) -> bool:
    return isinstance(s, ast.Expr) and _call_name(s.value) == MAT


def _is_coop_stmt(s: ast.stmt, coop: set[str]) -> bool:
    if _is_materialize(s) or (
        isinstance(s, ast.For) and _call_name(s.iter) == ELEMENTS
    ):
        return True
    if (assigned := _assigned(s)) is not None:
        value = assigned[1]
        return _call_name(value) in (ACC, RUN) or (
            isinstance(value, ast.Name) and value.id in coop
        )
    if isinstance(s, (ast.For, ast.If)):
        return any(_is_coop_stmt(x, coop) for x in (*s.body, *s.orelse))
    return False


def _dead_code(stmts: list[ast.stmt]) -> list[ast.stmt]:
    """Drop pure assignments nothing reads: the statements an element loop
    moved leave their originals' inputs dead."""
    while True:
        live = set().union(*map(names_read, stmts))
        pruned = _prune(stmts, live)
        if pruned is None:
            return stmts
        stmts = pruned


def _prune(stmts: list[ast.stmt], live: set[str]) -> list[ast.stmt] | None:
    """``stmts`` without the assignments to names outside ``live``, or None
    when there are none."""
    out: list[ast.stmt] = []
    changed = False
    for s in stmts:
        t = _target(s)
        if t is not None and t not in live:
            assert isinstance(s, ast.Assign)
            if _call_name(s.value) not in (ACC, RUN):
                changed = True
                continue
        if isinstance(s, (ast.For, ast.While, ast.If)) and not (
            isinstance(s, ast.For) and _call_name(s.iter) == ELEMENTS
        ):
            for field in ("body", "orelse"):
                pruned = _prune(getattr(s, field), live)
                if pruned is not None:
                    setattr(s, field, pruned)
                    changed = True
        out.append(s)
    return out if changed else None


def _is_run(s: ast.stmt) -> bool:
    assigned = _assigned(s)
    return assigned is not None and _call_name(assigned[1]) == RUN


def _is_elements(s: ast.stmt) -> bool:
    return isinstance(s, ast.For) and _call_name(s.iter) == ELEMENTS


def _jam_lane_loop(loop: ast.For, coop: set[str]) -> LaneJam:
    """``lane_fold``'s jam of a lane loop around a cooperative op: each
    cooperative statement, an element loop included, runs once where it
    stands, and the Scalar work beside it runs per lane in body order."""

    def run_once(s: ast.stmt) -> set[str] | None:
        if isinstance(s, (ast.For, ast.If)) and not _is_elements(s):
            return None
        return names_read(s) if _is_coop_stmt(s, coop) else None

    jam = LaneJam(loop, run_once)
    if carried := jam.carried():
        raise _unsupported(
            f"a lane loop around an MPP op carries {min(carried)} across lanes, "
            "which a copy of it per lane can't hold"
        )
    if reading := jam.lane_reading_units():
        raise _unsupported(
            "a cooperative MPP statement reads a lane-dependent name: "
            f"{sorted(set().union(*map(names_read, reading)) & jam.dependent)}"
        )
    return jam


def _split_schedulable(loop: ast.For, coop: set[str]) -> bool:
    """Whether ``_split_lane_loop`` takes the loop: one op directly in it,
    beside register statements, staging writes, fences and cooperative
    statements only, no accumulator seeded from a staged tile, which the
    split would move ahead of the lanes that write the tile, no lane fold
    (``lane_reduction`` emits them directly in the loop), whose readers need
    every lane of the same iteration folded first, and no staged vector
    written after the op (:func:`_vec_staged_after_op`)."""
    if sum(map(_is_run, loop.body)) != 1:
        return False
    for s in loop.body:
        assigned = _assigned(s)
        value = assigned[1] if assigned is not None else None
        if isinstance(value, ast.Call) and _call_name(value) == LANE_FOLD:
            return False
        if isinstance(value, ast.Call) and _call_name(value) == ACC:
            seed = value.args[-1]
            if not (isinstance(seed, ast.Constant) and seed.value is None):
                return False
        is_stage = (
            isinstance(s, ast.Expr)
            and isinstance(s.value, ast.Call)
            and _call_name(s.value) in (STAGE, FENCE)
        )
        if not (
            _is_run(s) or is_stage or _target(s) is not None or _is_coop_stmt(s, coop)
        ):
            return False
    return not _vec_staged_after_op(loop)


def _is_vec_stage(s: ast.stmt) -> bool:
    """Whether ``s`` writes a staged vector's tile."""
    return (
        isinstance(s, ast.Expr)
        and isinstance(s.value, ast.Call)
        and _call_name(s.value) == STAGE
        and _node_name(s.value.args[0]).startswith(VEC_TILE)
    )


def _vec_staged_after_op(loop: ast.For) -> bool:
    """Whether a staged vector's write follows the op: the split would move
    it ahead of the op, before the lane-invariant statements after the op
    that it may read, where the jam keeps body order."""
    at = next(i for i, s in enumerate(loop.body) if _is_run(s))
    return any(map(_is_vec_stage, loop.body[at + 1 :]))


def _split_lane_loop(loop: ast.For, lane: str, coop: set[str]) -> list[ast.stmt]:
    """Split a lane loop at the MPP op it holds directly (its device loop's
    K tile, whose lanes compute the op's staged operands), as
    ``lane_reduction``'s ``LaneSplit`` does at its cross-lane point: the
    lane-variant statements run in one lane loop before the op, in body
    order, and the others run once, before the op if they precede it and
    after it otherwise.

    A statement is lane-variant if it reads the lane or a name an earlier
    lane-variant statement assigns.  A lane-variant name read at or before
    its assignment is carried from one lane to the next, which one register
    can't hold per lane, so it is rejected.  A lane-invariant statement then
    reads no lane-variant name, so running it once gives the value every
    lane computed.

    Only register statements, staging writes, fences and cooperative
    statements move: with no store in the body, no load can see the
    reordering (the op writes no device memory, and only the op reads a
    staged tile).  A fence reads no lane, so it runs once before the lane
    loop, ahead of every load it orders.  A lane-variant statement after the
    op moves before it, which is sound because the op reads none of its
    names.  Such a statement may read a value assigned after the op, a read
    of an in-loop materialization of the op's result, and then compute it
    stale; nothing consumes that: a store is rejected, a carry is rejected
    here, and the split never sees a lane fold.
    """
    assert _split_schedulable(loop, coop)
    (at,) = [i for i, s in enumerate(loop.body) if _is_run(s)]
    others = [s for i, s in enumerate(loop.body) if i != at]
    variant_names = {lane}
    read: set[str] = set()
    for s in others:
        reads = names_read(s)
        read |= reads
        if (t := _target(s)) is not None and reads & variant_names:
            if t in read:
                raise _unsupported(
                    f"a lane loop around an MPP op carries {t} across lanes, which "
                    "a copy of it per lane can't hold"
                )
            variant_names.add(t)
    before: list[ast.stmt] = []
    lanes: list[ast.stmt] = []
    after: list[ast.stmt] = []
    for i, s in enumerate(loop.body):
        if i == at:
            continue
        if names_read(s) & variant_names:
            # The lane loop runs after every lane-invariant statement before
            # the op, an update's element loop among them, so a staged
            # vector's writes must not vary per lane: its value is over the
            # tile's M or N blocks, not K, which it would be if it read the
            # K lane.
            assert not _is_vec_stage(s), (
                "a staged vector's tile written per lane of the op's K tile"
            )
            lanes.append(s)
        else:
            (before if i < at else after).append(s)
    if lanes:
        before.append(ast.For(loop.target, loop.iter, lanes, [], None))
    return [*before, loop.body[at], *after]


def _defines(s: ast.AST) -> set[str]:
    """The names a statement leaves assigned for the statements after it:
    every name it assigns, nested bodies included, except inside an element
    loop, whose definitions are private to it (as in :func:`names_read`)."""
    if isinstance(s, ast.For) and _call_name(s.iter) == ELEMENTS:
        return set()
    if isinstance(s, ast.Name):
        return {s.id} if isinstance(s.ctx, ast.Store) else set()
    return set().union(*map(_defines, ast.iter_child_nodes(s)))


def _loaded(s: ast.AST) -> set[str]:
    """The names the pointers of a statement's ``tl.load`` calls read."""
    return set().union(
        *(names_read(load.args[0]) for node in ast.walk(s) if (load := _tl_load(node)))
    )


def _stmts(s: ast.stmt) -> list[ast.stmt]:
    return [x for x in ast.walk(s) if isinstance(x, ast.stmt)]


def _store_tensor(s: ast.stmt) -> str:
    """The tensor a ``tl.store(tensor + offset, ...)`` writes."""
    assert isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
    ptr = s.value.args[0]
    while isinstance(ptr, ast.BinOp):
        ptr = ptr.left
    return _node_name(ptr)


def _distribute_lane_loop(
    loop: ast.For, lane: str, coop: set[str]
) -> list[ast.stmt] | None:
    """Distribute a lane loop around a cooperative statement that reads no
    lane-variant name (a ``:`` dim's lanes around the K loop that holds the
    ops): each run of lane-variant statements between two cooperative
    statements becomes a lane loop of its own, and every lane-invariant
    statement runs once, in body order.  That is the jam's order
    (:func:`_jam_lane_loop`) with each run's copies rolled back into a loop.

    A statement is lane-variant if it reads or assigns a lane-dependent
    name, and the lane-dependent names are the lane and, to a fixpoint,
    every name a lane-variant statement assigns.  So ``s = 0.0`` starting a
    per-row sum is lane-variant though it reads no lane; a device loop or an
    ``if`` that assigns a per-lane name runs per lane, so every other name
    it assigns is per lane too; and a lane fold's partial is lane-dependent,
    so what reads it stays after the fold.  A lane-invariant statement then
    reads and assigns no name a lane-variant one assigns, which is why it
    may run once, ahead of a run's loop or between runs.  None
    (the jam takes the loop) when a lane-variant statement holds a
    cooperative op, assigns a name an earlier statement read (carried across
    lanes), or a run reads a lane-variant name an earlier run assigned that
    it can't recompute: only a name whose last assignment is call-free,
    doesn't read its own target, and reads only such names that are
    lane-variant is repeated.  Its inputs still hold the values it saw: a
    lane-variant one reassigned later would be carried across lanes, and
    the planner reads a lane-invariant name that a device loop reassigns
    through an ``_mpp_snap`` copy.  Also None when a tensor the run stores
    is touched by any other statement of the run, or is loaded by the
    statement that stores it: the run's lanes go one after another, so one
    lane's access would meet another lane's store in an order the jam,
    which runs every lane's copy of a statement before the next, doesn't
    have.  Within a run, a lane-invariant store of a tensor the run doesn't
    touch moves after the run's loop, and a lane-invariant assignment moves
    ahead of it when neither the run nor those deferred stores read the
    name it assigns or store a tensor it reads."""
    variant = {lane}
    changed = True
    while changed:
        changed = False
        for s in loop.body:
            names = _defines(s)
            if names - variant and (names_read(s) | names) & variant:
                variant |= names
                changed = True
    flags = [bool((names_read(s) | _defines(s)) & variant) for s in loop.body]
    read: set[str] = set()
    for s, v in zip(loop.body, flags, strict=True):
        if v and (_is_coop_stmt(s, coop) or _defines(s) & read):
            return None
        read |= names_read(s)
    if not any(
        not v and _is_coop_stmt(s, coop) for s, v in zip(loop.body, flags, strict=True)
    ):
        return None
    pure: dict[str, ast.stmt] = {}
    out: list[ast.stmt] = []
    run: list[ast.stmt] = []
    deferred: list[ast.stmt] = []

    def repeat(names: set[str], have: set[str], prefix: list[ast.stmt]) -> bool:
        for name in sorted(names - have - {lane}):
            d = pure.get(name)
            if d is None or not repeat(names_read(d) & variant, have, prefix):
                return False
            have.add(name)
            prefix.append(copy.deepcopy(d))
        return True

    def stored(ss: list[ast.stmt]) -> set[str]:
        return {_store_tensor(x) for s in ss for x in _stmts(s) if _is_store(x)}

    def loaded(ss: list[ast.stmt]) -> set[str]:
        return set().union(*map(_loaded, ss))

    def close() -> bool:
        nonlocal run, deferred
        effects = [s for s in run if _target(s) not in pure]
        if effects:
            have = set().union(*map(_defines, run))
            prefix: list[ast.stmt] = []
            if not repeat(set().union(*map(names_read, run)) & variant, have, prefix):
                return False
            out.append(ast.For(loop.target, loop.iter, [*prefix, *run], [], None))
        out.extend(deferred)
        run, deferred = [], []
        return True

    for s, v in zip(loop.body, flags, strict=True):
        written = _defines(s)
        for name in written & pure.keys():
            del pure[name]
        if v:
            if {_store_tensor(d) for d in deferred} & names_read(s):
                return None  # a deferred store's tensor read later in the run
            if stored([s]) & (_loaded(s) | loaded(run) | stored(run)) or (
                _loaded(s) & stored(run)
            ):
                return None  # its lanes would see each other's stores
            run.append(s)
            assigned = _assigned(s)
            if (
                assigned is not None
                and assigned[0] not in names_read(assigned[1])
                and not any(isinstance(x, ast.Call) for x in ast.walk(assigned[1]))
            ):
                pure[assigned[0]] = s
        elif (
            run
            and not _is_coop_stmt(s, coop)
            and (assigned := _assigned(s)) is not None
            and assigned[0] not in set().union(*map(names_read, [*run, *deferred]))
            and not names_read(s) & stored([*run, *deferred])
        ):
            # Ahead of the run's loop and the stores deferred after it: none
            # of them reads what it assigns or stores to a tensor it reads.
            out.append(s)
        elif (
            run
            and _is_store(s)
            and _store_tensor(s) not in set().union(*map(names_read, run))
        ):
            # A store holds no load: Helion lifts each load into an assignment
            # of its own, which the rule above checks.
            deferred.append(s)
        else:
            if not close():
                return None
            out.append(s)
    if not close():
        return None
    return out


def _is_lane_loop(s: ast.stmt) -> TypeGuard[ast.For]:
    # slice_lane_*: a persistent reduction's lanes over a ':' K.
    return (
        isinstance(s, ast.For)
        and isinstance(s.target, ast.Name)
        and s.target.id.startswith(("lane_", "slice_lane_"))
        and _call_name(s.iter) == "range"
    )


def _tl_load(node: ast.AST) -> ast.Call | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
        and node.func.attr == "load"
    ):
        return node
    return None


def _load_mask_other(load: ast.Call) -> tuple[ast.expr, ast.expr] | None:
    """The mask and fill value of ``tl.load(ptr, mask, other=...)``, or None
    for an unmasked load."""
    mask = load.args[1]
    if isinstance(mask, ast.Constant) and mask.value is None:
        return None
    other = load.args[2] if len(load.args) > 2 else None
    other = next((k.value for k in load.keywords if k.arg == "other"), other)
    assert other is not None
    return mask, other


def _loads_first_all(
    stmts: list[ast.stmt],
    split: list[ast.For],
    coop: set[str],
    tensor_dtypes: Mapping[str, torch.dtype],
    tg_buffers: frozenset[str],
) -> list[ast.stmt]:
    """Rewrite each staging lane loop a split made (``split``, the loop
    objects themselves) with ``_loads_first`` where no lane loop encloses it.
    Runs after every splice and jam: a jam's copies are new statements, never
    in ``split``, and an enclosing lane loop may still be jammed or unrolled
    at launch (``unroll_lane_folds``), which would interleave one copy's load
    loop with another copy's use loop over one array."""
    taken = {n.id for s in stmts for n in ast.walk(s) if isinstance(n, ast.Name)}

    def fresh_names() -> Iterator[str]:
        n = 0
        while True:
            if (name := f"_mpp_lanes_{n}") not in taken:
                yield name
            n += 1

    names = fresh_names()

    def walk(ss: list[ast.stmt], in_lane: bool) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for s in ss:
            if not in_lane and any(s is loop for loop in split):
                assert isinstance(s, ast.For)
                out += _loads_first(s, coop, tensor_dtypes, tg_buffers, names)
                continue
            if isinstance(s, (ast.For, ast.While, ast.If)) and not _is_elements(s):
                inner = in_lane or _is_lane_loop(s)
                s.body = walk(s.body, inner)
                s.orelse = walk(s.orelse, inner)
            out.append(s)
        return out

    return walk(stmts, False)


#: Most lanes a loads-first split covers.  Metal miscompiles the split at 64
#: lanes when its K loop runs once (int8, uint8, int16 and float16 loads; 16
#: and 32 lanes are exact).
MAX_LOADS_FIRST_LANES = 32


def _loads_first(
    loop: ast.For,
    coop: set[str],
    tensor_dtypes: Mapping[str, torch.dtype],
    tg_buffers: frozenset[str],
    names: Iterator[str],
) -> list[ast.stmt]:
    """A split's staging lane loop as two lane loops over its range: the
    first issues each lane's narrow device loads into per-lane arrays, the
    second runs the body with each load replaced by its lane's element.

    In one loop, lane ``l + 1``'s load comes after lane ``l``'s staging
    writes, which the Metal compiler doesn't move it above, so each thread
    has one load in flight; CuTe's ``split_lane_loads`` splits for the same
    reason.  The loop holds register statements and staging writes only
    (``_split_schedulable``), with no device store, so a device load reads
    the same value ahead of earlier lanes' writes, and a load's pointer and
    mask read only register statements before it, which run in both loops.
    The arrays share ``lane_fold.MAX_LANE_ARRAY_BYTES`` in body order, and
    the loop is rewritten only when every load in it gets one: a load left in
    the second loop still waits behind the staging writes (two loads at 64
    lanes, one kept: 0.74x)."""
    lane = cast("ast.Name", loop.target).id
    body = loop.body
    lanes = const_extent(loop)
    if lanes > MAX_LOADS_FIRST_LANES:
        return [loop]
    budget = LaneArrayBudget(lanes)
    defs: dict[str, int] = {}
    for i, s in enumerate(body):
        if (t := _target(s)) is not None:
            assert t not in defs, t
            defs[t] = i
    moved: dict[int, str] = {}
    slice_: set[int] = set()
    decls: list[ast.stmt] = []
    for i, s in enumerate(body):
        assigned = _assigned(s)
        if assigned is None or (load := _tl_load(assigned[1])) is None:
            continue
        # Helion spells every load's pointer ``<tensor> + <offset>``.
        ptr = load.args[0]
        assert isinstance(ptr, ast.BinOp) and isinstance(ptr.left, ast.Name)
        tensor = ptr.left.id
        assert not names_read(load) & tg_buffers
        pending = sorted(names_read(load) & defs.keys())
        while pending:
            j = defs[pending.pop()]
            if j in slice_ or j in moved:
                continue
            assert j < i and not names_read(body[j]) & (coop | tg_buffers)
            slice_.add(j)
            pending += sorted(names_read(body[j]) & defs.keys())
        if budget.take(tensor_dtypes[tensor]):
            moved[i] = next(names)
            decls.append(
                statement_from_string(
                    f"{moved[i]} = _metal_lane_array("
                    f"{DTYPE_TO_METAL[tensor_dtypes[tensor]]!r}, {lanes})"
                )
            )
    loads = sum(1 for s in body for node in ast.walk(s) if _tl_load(node))
    if not moved or len(moved) < loads:
        return [loop]
    first: list[ast.stmt] = []
    second: list[ast.stmt] = []
    for i, s in enumerate(body):
        if i in slice_:
            first.append(copy.deepcopy(s))
        if i not in moved:
            second.append(s)
            continue
        array = moved[i]
        name, value = cast("tuple[str, ast.expr]", _assigned(s))
        load = cast("ast.Call", _tl_load(value))
        first += [
            s,
            statement_from_string(f"_metal_lane_put({array}, {lane}, {name})"),
        ]
        read = f"_metal_lane_get({array}, {lane})"
        masked = _load_mask_other(load)
        if masked is None:
            second.append(statement_from_string(f"{name} = {read}"))
            continue
        mask, other = masked
        # The array holds ``mask ? x : other``; the select keeps the load's
        # own MSL type.
        second.append(
            statement_from_string(
                f"{name} = {read} if {{mask}} else {{other}}",
                mask=copy.deepcopy(mask),
                other=copy.deepcopy(other),
            )
        )
    return [
        *decls,
        ast.For(copy.deepcopy(loop.target), copy.deepcopy(loop.iter), first, [], None),
        ast.For(loop.target, loop.iter, second, [], None),
    ]


class _Replicas:
    """The threads that hold the same staged elements, for dealing a staging
    lane loop's lanes out over them.

    A staged operand's dims are its group's blocks, so threads that differ
    only on another block's thread axis compute the same elements, and the
    owner test of :func:`_stage_writes` lets the one at 0 write them.  In a
    lane loop over a ``:`` dim, which holds no threads, that one thread
    walks every lane while the others wait at the op's barrier (attention's
    ``q * qk_scale`` over the head dim, idle on every N thread).
    :meth:`spread` gives each of them every n-th lane instead."""

    def __init__(
        self,
        body: list[ast.stmt],
        tg_size: Sequence[int] | None,
        tg_buffers: frozenset[str],
    ) -> None:
        self.tg_size = tg_size
        self.tg_buffers = tg_buffers
        #: Each binding of a name: the value it reads (``None``: it may vary
        #: on every axis) and the tests and loop ranges it runs under.
        self.bindings: list[tuple[str, ast.expr | None, tuple[ast.expr, ...]]] = []
        self._collect(body, ())
        self.seen = {n.id for s in body for n in ast.walk(s) if isinstance(n, ast.Name)}
        #: The thread axes each name's value may vary on, over every binding
        #: of the name.  A name the body reads but never binds is a kernel
        #: parameter, a launch constant, ``tgid``, a threadgroup buffer or a
        #: helper (``metal_jit._generate_msl`` declares nothing else), each
        #: the same on every thread of the threadgroup.  A name not in the
        #: body at all was made after it, by splicing (a lane jam's
        #: ``t_1__lane0``), and :meth:`reads_axes` lets it vary on every axis.
        self.axes: dict[str, set[int]] = {}
        changed = True
        while changed:
            changed = False
            for name, value, control in self.bindings:
                axes = {0, 1, 2} if value is None else self.value_axes(value)
                for c in control:
                    axes |= self.value_axes(c)
                if not axes <= self.axes.get(name, set()):
                    self.axes[name] = self.axes.get(name, set()) | axes
                    changed = True

    def _collect(self, stmts: list[ast.stmt], control: tuple[ast.expr, ...]) -> None:
        for s in stmts:
            if isinstance(s, ast.For) and _call_name(s.iter) == ELEMENTS:
                # The walker emits an element loop as an isolated scope
                # (``MslWalkerState.isolated_scope``): each of its bindings
                # declares a new variable that nothing after the loop sees.
                continue
            if isinstance(s, (ast.If, ast.While, ast.For)):
                header = s.iter if isinstance(s, ast.For) else s.test
                if isinstance(s, ast.For):
                    for t in ast.walk(s.target):
                        if isinstance(t, ast.Name):
                            self.bindings.append((t.id, s.iter, control))
                self._bind_every_store(header, None, control)
                self._collect([*s.body, *s.orelse], (*control, header))
                continue
            target: ast.expr | None = None
            if (assigned := _assigned(s)) is not None:
                target = cast("ast.Assign", s).targets[0]
                self.bindings.append((*assigned, control))
            elif isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name):
                target = s.target
                self.bindings.append((s.target.id, s.value, control))
            self._bind_every_store(s, target, control)

    def _bind_every_store(
        self, node: ast.AST, modelled: ast.expr | None, control: tuple[ast.expr, ...]
    ) -> None:
        """Bind every name *node* stores other than *modelled*, and the base
        name of every subscript or attribute it stores, to a value that may
        vary on every axis."""
        for t in ast.walk(node):
            if isinstance(t, (ast.Subscript, ast.Attribute)) and isinstance(
                t.ctx, ast.Store
            ):
                base = t.value
                while isinstance(base, (ast.Subscript, ast.Attribute)):
                    base = base.value
                if isinstance(base, ast.Name):
                    self.bindings.append((base.id, None, control))
            elif isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                if t is not modelled:
                    self.bindings.append((t.id, None, control))

    def value_axes(self, node: ast.expr) -> set[int]:
        """The thread axes the value of *node* may vary on: every axis for a
        value that exchanges data between threads, else :meth:`reads_axes`."""
        if _cross_thread(node, self.tg_buffers):
            return {0, 1, 2}
        return self.reads_axes(node)

    def reads_axes(self, node: ast.AST) -> set[int]:
        """The thread axes the values *node* reads may vary on: ``tid[a]``
        varies on ``a``; a whole ``tid``, and a name not in the analysed
        body, on every axis."""
        axes: set[int] = set()
        indexed: set[int] = set()
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Subscript)
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "tid"
                and isinstance(sub.slice, ast.Constant)
            ):
                axes.add(ast.literal_eval(sub.slice))
                indexed.add(id(sub.value))
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                if sub.id == "tid":
                    if id(sub) not in indexed:
                        axes |= {0, 1, 2}
                elif sub.id in self.seen:
                    axes |= self.axes.get(sub.id, set())
                else:
                    axes |= {0, 1, 2}
        return axes

    def uniform(self, node: ast.AST) -> bool:
        """Whether every thread computes the same *node*."""
        return not self.reads_axes(node)

    def owner_axes(self, name: str) -> tuple[int, ...] | None:
        """The axes *name* pins to 0 when its only binding is an owner test
        of :func:`_stage_writes`, ``tid[a] == 0 and ...``; else ``None``."""
        values = [v for n, v, _ in self.bindings if n == name]
        if len(values) != 1 or values[0] is None:
            return None
        axes: list[int] = []
        for term in _conjuncts(values[0]):
            if not (
                isinstance(term, ast.Compare)
                and len(term.ops) == 1
                and isinstance(term.ops[0], ast.Eq)
                and isinstance(term.left, ast.Subscript)
                and isinstance(term.left.value, ast.Name)
                and term.left.value.id == "tid"
                and isinstance(term.left.slice, ast.Constant)
                and isinstance(term.comparators[0], ast.Constant)
                and term.comparators[0].value == 0
            ):
                return None
            axes.append(ast.literal_eval(term.left.slice))
        return tuple(axes)

    def spread(self, loop: ast.For) -> ast.For:
        """*loop*, a lane loop every thread reaches, with its lanes dealt out
        over the threads of the one axis of more than one thread that its
        staging writes pin to 0: thread ``t`` of that axis runs lanes ``t``,
        ``t + n``, ..., so each lane still runs on exactly one thread and
        each element is still written once.  Only when the body is register
        work and such writes, all pinned to the same axes by an owner test
        that is their predicate's last term, each at an index that reads the
        lane (a flat tile index, distinct per lane), no lane reads a name an
        earlier lane assigned (no name is read before or by its assignment),
        and nothing in it but that test varies on that axis; else *loop*
        itself."""
        extent = loop.iter
        assert isinstance(extent, ast.Call)
        if self.tg_size is None or not (
            len(extent.args) == 1 and isinstance(extent.args[0], ast.Constant)
        ):
            return loop
        per_lane = {cast("ast.Name", loop.target).id}
        pinned: set[tuple[int, ...]] = set()
        read: set[str] = set()
        for s in loop.body:
            read |= names_read(s)
            if (assigned := _assigned(s)) is not None:
                if _cross_thread(assigned[1], self.tg_buffers) or assigned[0] in read:
                    return loop
                if names_read(assigned[1]) & per_lane:
                    per_lane.add(assigned[0])
                else:
                    per_lane.discard(assigned[0])
            elif isinstance(s, ast.Expr) and _call_name(s.value) == STAGE:
                call = cast("ast.Call", s.value)
                axes = ast.literal_eval(call.args[4])
                owner = _conjuncts(call.args[3])[-1]
                if not (
                    isinstance(owner, ast.Name)
                    and self.owner_axes(owner.id) == axes
                    and names_read(call.args[1]) & per_lane
                    and not any(
                        _cross_thread(a, self.tg_buffers) for a in call.args[1:4]
                    )
                ):
                    return loop
                pinned.add(axes)
            else:
                return loop
        if len(pinned) != 1:
            return loop
        axes = [a for a in pinned.pop() if self.tg_size[a] > 1]
        if len(axes) != 1:
            return loop
        (axis,) = axes
        body: list[ast.stmt] = []
        for s in loop.body:
            s = copy.deepcopy(s)
            if isinstance(s, ast.Expr) and _call_name(s.value) == STAGE:
                call = cast("ast.Call", s.value)
                *terms, _owner = _conjuncts(call.args[3])
                call.args[3] = (
                    ast.BoolOp(ast.And(), terms)
                    if len(terms) > 1
                    else terms[0]
                    if terms
                    else ast.Constant(True)
                )
                call.args[4] = ast.Tuple([], ast.Load())
            if axis in self.reads_axes(s):
                return loop
            body.append(s)
        n = self.tg_size[axis]
        return ast.For(
            loop.target,
            cast(
                "ast.expr",
                expr_from_string(f"range(tid[{axis}], {extent.args[0].value}, {n})"),
            ),
            body,
            [],
            None,
        )


def _conjuncts(test: ast.expr) -> list[ast.expr]:
    """The terms of an ``and`` chain."""
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return [t for v in test.values for t in _conjuncts(v)]
    return [test]


def _run_lane_loops_once(
    stmts: list[ast.stmt],
    coop: set[str],
    tensor_dtypes: Mapping[str, torch.dtype],
    tg_buffers: frozenset[str],
    tg_size: Sequence[int] | None,
) -> list[ast.stmt]:
    """Splice in the body of each lane loop that holds a cooperative
    statement: the op reads its whole tile from memory, so the loop's lanes
    would run it again.  Lane-invariant Scalar work beside it then runs once
    per iteration of the enclosing loop, as in Helion's semantics.  A lane
    loop holding the op directly beside lane-variant register and staging
    work only (the op's own K tile, whose lanes compute a staged operand) is
    split at the op.  A loop distributed around the op
    (``_distribute_lane_loop``) has each staging run's lanes dealt out over
    the threads that hold the same elements (``_Replicas.spread``) where
    every thread reaches it.  Any other loop that reads its lane is jammed
    across its lanes (``_jam_lane_loop``), at most ``MAX_LANE_COPIES``
    copies of a statement deep."""

    jammed: set[str] = set()
    copies_of: dict[int, int] = {}

    def copies(ss: list[ast.stmt]) -> int:
        """Most copies of one statement the jams inside ``ss`` made."""
        best = 1
        for s in ss:
            best = max(best, copies_of.get(id(s), 1))
            if isinstance(s, (ast.For, ast.If)) and not _is_elements(s):
                best = max(best, copies([*s.body, *s.orelse]))
        return best

    split: list[ast.For] = []
    replicas = _Replicas(stmts, tg_size, tg_buffers)

    def splice(ss: list[ast.stmt], uniform: bool = True) -> list[ast.stmt]:
        """*uniform*: every thread of the threadgroup runs ``ss``."""
        out: list[ast.stmt] = []
        for s in ss:
            if isinstance(s, ast.For) and _call_name(s.iter) != ELEMENTS:
                s.body = splice(s.body, uniform and replicas.uniform(s.iter))
                if _is_lane_loop(s) and any(_is_coop_stmt(x, coop) for x in s.body):
                    lane = cast("ast.Name", s.target).id
                    if not any(lane in names_read(x) for x in s.body):
                        out.extend(s.body)
                    elif _split_schedulable(s, coop):
                        parts = _split_lane_loop(s, lane, coop)
                        split.extend(p for p in parts if _is_lane_loop(p))
                        out += parts
                    elif (
                        distributed := _distribute_lane_loop(s, lane, coop)
                    ) is not None:
                        out += [
                            replicas.spread(x) if uniform and _is_lane_loop(x) else x
                            for x in distributed
                        ]
                    else:
                        if folded_lanes(s.body) - {lane}:
                            raise _unsupported(
                                "a lane loop around an MPP op holds a reduction "
                                "over another lane loop"
                            )
                        n = const_extent(s) * copies(s.body)
                        if n > MAX_LANE_COPIES:
                            raise _unsupported(
                                f"a lane loop around an MPP op would run {n} copies "
                                f"of its Scalar work per thread (max {MAX_LANE_COPIES})"
                            )
                        jam = _jam_lane_loop(s, coop)
                        body = jam.jam()
                        jammed.update(jam.dependent - {lane})
                        copies_of.update((id(x), n) for x in body)
                        out += body
                    continue
            elif isinstance(s, ast.If):
                inner = uniform and replicas.uniform(s.test)
                s.body = splice(s.body, inner)
                s.orelse = splice(s.orelse, inner)
            out.append(s)
        return out

    result = _loads_first_all(
        splice(_dead_code(stmts)), split, coop, tensor_dtypes, tg_buffers
    )
    leaked = jammed & set().union(*map(names_read, result))
    assert not leaked, f"lane-dependent names read outside their copies: {leaked}"
    return result
