"""Cooperative MPP lowering of a K-loop matmul accumulator.

A matmul that updates its K loop's own carry, a ``torch.addmm`` or an
``hl.dot`` over a K held as per-thread parts (``factored_dims``), keeps that
carry in MPP's layout, as one ``cooperative_tensor``, across the loop.  A
``torch.addmm`` over a ``:`` K is one op over the whole K, its result a
cooperative value too:

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
  order, each cooperative statement once.

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

import torch
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
from ..aten_lowering import mm_lowering
from ..compile_environment import CompileEnvironment
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
from .lane_fold import MAX_LANE_COPIES
from .lane_fold import LaneJam
from .lane_fold import const_extent
from .msl_reduction import REDUCTION_NAMESPACE
from .reduction import alloc_threadgroup_buffer

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from torch.fx.graph import Graph

    from ..aten_lowering import LoweringContext
    from ..device_ir import GraphInfo
    from ..device_ir import IfGraphInfo
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import CodegenState

ACC = "_helion_mpp_acc"
RUN = "_helion_mpp_run"
ELEMENTS = "_helion_mpp_elements"
BIND = "_helion_mpp_bind"
STAGE = "_helion_mpp_stage"
MAT = "_helion_mpp_materialize"
#: A barrier that makes this threadgroup's earlier device writes visible.
FENCE = "_helion_mpp_fence"
#: Name prefix of a materialization's threadgroup tile.
MAT_TILE = "_mpp_mat"

_ADDMM = torch.ops.aten.addmm.default
#: Each matmul's (acc, lhs, rhs) argument positions.
_MATMULS: dict[object, tuple[int, int, int]] = {_ADDMM: (0, 1, 2), dot: (2, 0, 1)}
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


@dataclasses.dataclass
class _Operand:
    """A matmul operand: a device view, or, with no view, a value the Scalar
    layout computes, which the threads that hold it write into a threadgroup
    tile before the op (staging)."""

    dtype: torch.dtype
    view: _View | None = None

    @property
    def staged(self) -> bool:
        return self.view is None


@dataclasses.dataclass
class MppGroup:
    gid: int
    matmul: Node
    seed: Node
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
    #: Loads of a staged operand that read a tensor the kernel writes: each
    #: runs after a device-memory barrier.
    fenced: set[Node] = dataclasses.field(default_factory=set)


def _node(value: object) -> Node:
    assert isinstance(value, Node)
    return value


def _seq(value: object) -> Sequence[object]:
    assert isinstance(value, (list, tuple))
    return value


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
    operand: Node, dims: tuple[tuple[int, int], tuple[int, int]], graphs: _Graphs
) -> _View:
    """A device view: a load, or its transpose, whose value is 2-D.  Its two
    value dims are indexed by tiles, ``base + tile.index``, ``base +
    hl.arange``, a widened ``hl.arange`` or ``:``, with an optional mask
    ``index < bound`` on one ``hl.arange`` dim; every other tensor dim by a
    uniform integer, which drops it."""
    permute = torch.ops.aten.permute.default
    transposed = operand.target is permute and list(_seq(operand.args[1])) == [1, 0]
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
    if len(index) != tensor.ndim or value.ndim != 2:
        raise _NotAView("an MPP view whose load is not 2-D after its scalar indices")
    if tensor.dtype not in _DTYPES:
        raise _unsupported(f"an MPP operand of dtype {tensor.dtype}")
    if not all(isinstance(x, int) for x in (*tensor.shape, *tensor.stride())):
        # The descriptor's transpose flag and the view's unit-stride dim are
        # compile-time facts.
        raise _NotAView("an MPP view of a tensor with symbolic sizes or strides")
    positions: list[int] = []
    outer: list[tuple[int, Node | int]] = []
    for i, x in enumerate(index):
        if not _scalar_index(x, graphs):
            positions.append(i)
            continue
        assert isinstance(x, (Node, int))
        outer.append((i, x))
    if len(positions) != 2:
        raise _NotAView("an MPP view whose load is not 2-D after its scalar indices")
    if 1 not in (tensor.stride(positions[0]), tensor.stride(positions[1])):
        raise _NotAView("an MPP view without a unit stride")
    value_dims = [
        _view_dim(index[p], value.size(j), graphs) for j, p in enumerate(positions)
    ]
    if extra_mask is not None:
        _view_mask(extra_mask, value_dims, graphs)
    if transposed:
        value_dims.reverse()
        positions.reverse()
    view_dims = (value_dims[0], value_dims[1])
    if tuple((d.block_id, d.factor) for d in view_dims) != dims:
        raise _unsupported("an MPP view whose dims are not the matmul's blocks")
    return _View(tensor, view_dims, (positions[0], positions[1]), tuple(outer))


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
    node: Node, dims: tuple[tuple[int, int], tuple[int, int]], graphs: _Graphs
) -> _Operand:
    """A device view when the operand is one, else staged.  The threads
    compute a staged operand in the Scalar layout, one element per block
    coordinate (``c`` parts on a factored K), and write it into the tile at
    those coordinates, so every node it is computed from must meet its
    operands' dims by block."""
    try:
        view = _view(node, dims, graphs)
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


def plan_mpp_matmuls(graphs: list[GraphInfo]) -> MppPlan:
    """Admit each ``aten.addmm``, and each ``hl.dot`` over a K held as
    per-thread parts, of the kernel's live graphs as a cooperative group, or
    raise ``BackendUnsupported``.

    Each use of a group's cooperative value is decided from the FX graph: an
    element-wise use or a store that an element loop can evaluate at any
    element stays cooperative; every other use reads the value through a
    ``_mpp_mat`` node, a materialization into the Scalar layout inserted
    right after the value's definition, so it reads the version it names.  A
    seed that is not a constant fill reaches the K loop through a
    ``_mpp_coop`` node, which stages it into the cooperative tensor.

    The graph pass leaves the K-loop graph it replaced in ``graphs``, so only
    the graphs the kernel runs are scanned.
    """
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
        if matmul.target is dot:
            # Every other hl.dot keeps the graph pass (mpp_graph_transform)
            # or the scalar contraction (matmul_ops); one over a K held as
            # per-thread parts has no other lowering.
            lhs = matmul.args[0]
            assert isinstance(lhs, Node)
            if factored_size(lhs.meta["val"].size(-1)) is None:
                continue
            if matmul.args[2] is None:
                raise _unsupported(
                    "an hl.dot over a K held as per-thread parts with no acc"
                )
            if matmul.args[3] not in (None, torch.float32):
                raise _unsupported(
                    f"an hl.dot over per-thread parts with out_dtype={matmul.args[3]}"
                )
        if (k_block := _full_k(matmul)) is not None:
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
        plan.by_seed[group.seed] = group
    if plan.by_matmul:
        _order_memory(plan, live, known)
        for group in plan.by_matmul.values():
            _finalize(group, plan, known)
    return plan


def _plan_group(gid: int, matmul: Node, k_loop_block: int, graphs: _Graphs) -> MppGroup:
    acc, lhs, rhs = (_node(matmul.args[i]) for i in _MATMULS[matmul.target])
    placeholder = acc
    while placeholder.target is _tracing_ops._new_var:
        placeholder = _node(placeholder.args[0])
    loop = graphs.owner(placeholder) if placeholder.op == "placeholder" else None
    carry = _carry(placeholder, loop) if loop is not None else None
    if carry is None:
        if _updates_carry(acc, graphs):
            raise _unsupported(
                "a matmul whose acc is an update of its K loop's carry inside the "
                "loop, such as acc * alpha before the op"
            )
        raise _unsupported("a matmul whose acc is not its K loop's carry")
    assert loop is not None
    (output,) = matmul.graph.find_nodes(op="output")
    slot = carry.args[1]
    assert isinstance(slot, int)
    if _seq(output.args[0])[slot] is not matmul:
        raise _unsupported("a matmul whose result is not its K loop's carry")
    levels = _levels(placeholder, loop, carry, matmul, graphs)
    m, n, kdim = _mnk(matmul, k_loop_block)
    outer = levels[-1]
    seed = graphs.origin(outer.arg)
    fill = seed.args[1] if seed.target is full else None
    if not (isinstance(fill, (int, float)) and math.isfinite(fill)):
        seed = _scalar_seed(outer, m, n, graphs)
    group = MppGroup(
        gid,
        matmul,
        seed,
        m,
        n,
        kdim[0],
        _operand(lhs, ((m, 1), kdim), graphs),
        _operand(rhs, (kdim, (n, 1)), graphs),
        k_factor=kdim[1],
        levels=levels,
    )
    if group.lhs.dtype != group.rhs.dtype:
        raise _unsupported("MPP operands of two dtypes")
    _check_slice_staging(group, graphs)
    _storage(group, acc)
    _classify_uses(group, graphs)
    return group


def _mnk(matmul: Node, k_block: int) -> tuple[int, int, tuple[int, int]]:
    """The matmul's M and N blocks and its K as ``(block, factor)``, whose
    block must be *k_block*."""
    lhs, rhs = (_node(matmul.args[i]) for i in _MATMULS[matmul.target][1:])
    lval, rval = lhs.meta["val"], rhs.meta["val"]
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
    if matmul.meta["val"].dtype != torch.float32:
        raise _unsupported("a matmul whose acc is not float32")
    return m, n, kdim


def _full_k(matmul: Node) -> int | None:
    """The block of the matmul's K when K is a ``:`` slice: a reduction
    dimension, which no loop iterates."""
    lhs = _node(matmul.args[_MATMULS[matmul.target][1]])
    block = _block(lhs.meta["val"].size(-1))
    if block is None or not CompileEnvironment.current().block_sizes[block].reduction:
        return None
    return block


def _plan_full_k(gid: int, matmul: Node, k_block: int, graphs: _Graphs) -> MppGroup:
    """A matmul over a ``:`` K: one op over the whole K, seeded by a constant
    fill of the op's own graph, with both operands device views."""
    acc, lhs, rhs = (_node(matmul.args[i]) for i in _MATMULS[matmul.target])
    m, n, kdim = _mnk(matmul, k_block)
    seed = acc
    while seed.target is _tracing_ops._new_var:
        seed = _node(seed.args[0])
    if seed.op == "placeholder":
        raise _unsupported(
            "a matmul over a ':' K whose acc an enclosing loop or if passes in"
        )
    fill = seed.args[1] if seed.target is full else None
    if not (isinstance(fill, (int, float)) and math.isfinite(fill)):
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
        _operand(lhs, ((m, 1), kdim), graphs),
        _operand(rhs, (kdim, (n, 1)), graphs),
    )
    if group.lhs.dtype != group.rhs.dtype:
        raise _unsupported("MPP operands of two dtypes")
    _check_slice_staging(group, graphs)
    _storage(group, acc)
    _classify_uses(group, graphs)
    return group


def _check_slice_staging(group: MppGroup, graphs: _Graphs) -> None:
    """Reject an operand the plan stages inside a loop for an op over a
    ``:`` M, N or K.  Metal miscompiles threadgroup stores inside a loop
    whose values it can prove the same in every iteration, when a lane jam
    copies them: at some thread layouts most of the tile reads back as zero.
    No rule on the FX graph can tell what the compiler proves.  A view that
    ``_order_memory`` demotes to staged later reads a tensor the kernel
    writes, which the compiler can't prove unchanged, so it stays admitted."""
    env = CompileEnvironment.current()
    if (
        (group.lhs.staged or group.rhs.staged)
        and any(
            env.block_sizes[b].reduction
            for b in (group.m_block, group.n_block, group.k_block)
        )
        and _enclosing_loop(group.matmul.graph, graphs) is not None
    ):
        raise _unsupported(
            "a staged MPP operand of a matmul over a ':' slice inside a loop, "
            "which Metal can miscompile"
        )


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


def _scalar_seed(outer: _Level, m: int, n: int, graphs: _Graphs) -> Node:
    """Route a seed that is not a constant fill through a ``_mpp_coop`` node
    before the outermost level, which stages it into the cooperative tensor;
    the seed's other readers keep its Scalar value."""
    arg = outer.arg
    val = arg.meta["val"]
    assert isinstance(val, torch.Tensor)
    if val.ndim != 2 or (_block(val.size(0)), _block(val.size(1))) != (m, n):
        raise _unsupported("an MPP accumulator seed whose dims are not M and N")
    positional_rebind.check_positional_merges(_producers(arg, graphs), _STAGE_REBIND)
    coop = _insert(_mpp_coop, arg, before=outer.owner)
    outer.owner.replace_input_with(arg, coop)
    outer.phi.replace_input_with(arg, coop)
    outer.arg = coop
    return coop


def _insert(
    op: Callable[..., object], src: Node, *, before: Node | None = None
) -> Node:
    """A call of ``op`` on ``src``, inserted before ``before`` or right after
    ``src``'s definition (after the graph's placeholders for one)."""
    graph = src.graph
    if before is not None:
        ctx = graph.inserting_before(before)
    elif src.op == "placeholder":
        ctx = graph.inserting_after(list(graph.find_nodes(op="placeholder"))[-1])
    else:
        ctx = graph.inserting_after(src)
    with ctx:
        node = graph.call_function(op, (src,), {})
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


def _storage(group: MppGroup, acc: Node) -> None:
    """Sort the values that hold the cooperative storage into ``old`` (the
    seed and each level's input, which the op updates in place) and ``cur``
    (the op's result and each level's exit), and record the uses that only
    pass the storage along."""
    plumbing = group.plumbing
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
    while seed.target is _tracing_ops._new_var:
        seed = _node(seed.args[0])
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
        blocks = {self.group.m_block, self.group.n_block}
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
    """
    recomputable = _Recomputable(group, graphs)
    work = list(group.cur)
    while work:
        node = work.pop()
        for user in node.users:
            if (
                (node, user) in group.plumbing
                or user.target is torch.ops.aten.sym_size.int
                or user in group.derived
            ):
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
                user, group.cur | group.derived, recomputable
            ):
                group.derived.add(user)
                work.append(user)
    group.recomputable = recomputable


def _inputs_kept(node: Node, coop: set[Node], recomputable: _Recomputable) -> bool:
    """Whether an element loop can evaluate every input of ``node``: each is
    cooperative or recomputable at any element."""
    return all(a in coop or recomputable(a) for a in node.all_input_nodes)


def _prune_derived(group: MppGroup) -> None:
    """Keep a derived value cooperative only while every input of it is
    cooperative or recomputable and every use of it is another kept one or
    an element-loop store; a store keeps its element loop only while its
    value is cooperative.  A dropped value is computed in the Scalar layout,
    so a kept value that read it would read one thread's element at every
    cooperative element."""
    while True:
        coop = group.cur | group.derived
        stores = [s for s in group.stores if s.args[2] in coop]
        keep = set(stores) | group.derived
        drop = {
            d
            for d in group.derived
            if any(
                u not in keep and u.target is not torch.ops.aten.sym_size.int
                for u in d.users
            )
            or not _inputs_kept(d, coop, group.recomputable)
        }
        if not drop and len(stores) == len(group.stores):
            return
        group.derived -= drop
        group.stores = stores


def _order_memory(plan: MppPlan, live: list[GraphInfo], graphs: _Graphs) -> None:
    """MPP gives each element to a thread of its own choosing, and no barrier
    orders device memory around an op.  So an element-loop store must write
    a tensor nothing else accesses and re-run no load of a tensor the kernel
    writes, and an operand view must not read such a tensor.  A store that
    can't keep that takes the Scalar layout, where the thread that writes an
    element is the one the Scalar program names, and a view is staged by
    the threads that hold it.  Tensors are keyed by fake storage, so views of
    one tensor fall together (as ``free_arange._check_write_once``)."""

    def key(node: Node) -> StorageWeakRef:
        tensor = node.args[0]
        if isinstance(tensor, (list, tuple)):
            tensor = tensor[0]
        assert isinstance(tensor, Node)
        return StorageWeakRef(tensor.meta["val"].untyped_storage())

    accesses: dict[StorageWeakRef, list[Node]] = {}
    written: set[StorageWeakRef] = set()
    for g in live:
        for n in g.graph.nodes:
            if n.target is load or n.target in _WRITES:
                accesses.setdefault(key(n), []).append(n)
                if n.target in _WRITES:
                    written.add(key(n))
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
            if any(n is not st for n in accesses[key(st)]) or any(
                key(ld) in written for ld in reloads
            ):
                continue
            kept.append(st)
        group.stores = kept
        _prune_derived(group)
        _, lhs, rhs = (
            _node(group.matmul.args[i]) for i in _MATMULS[group.matmul.target]
        )
        for operand, node in ((group.lhs, lhs), (group.rhs, rhs)):
            view = operand.view
            if view is not None and (
                StorageWeakRef(view.tensor.untyped_storage()) in written
            ):
                operand.view = None
            if operand.staged:
                for ld in _operand_loads(node, group.matmul.graph, graphs):
                    if key(ld) in written:
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
    coop = group.cur | group.derived
    keep = coop | set(group.stores)
    recomputable = group.recomputable
    readers: list[tuple[Node, Node]] = []
    for node in sorted(group.old | group.cur, key=lambda n: n.name):
        for user in list(node.users):
            if (
                (node, user) in group.plumbing
                or user in keep
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
    rerun = sorted(coop, key=lambda n: n.name) + list(group.stores)
    for n in [*group.derived, *group.stores]:
        for a in n.all_input_nodes:
            if a not in coop:
                rerun += recomputable.nodes(a)
    _check_uniform(group.matmul, graphs)
    positional_rebind.check_positional_merges(rerun, _REBIND)
    positional_rebind.check_positional_merges(
        _forward_slice(list(group.mats.values()), graphs), _MAT_REBIND
    )


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


def _plan() -> MppPlan:
    from .backend import MetalBackend

    backend = CompileEnvironment.current().backend
    assert isinstance(backend, MetalBackend) and backend.mpp_plan is not None
    return backend.mpp_plan


def _transposed(operand: _Operand) -> bool:
    """A staged tile is written row-major."""
    view = operand.view
    return view is not None and view.tensor.stride(view.positions[1]) != 1


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
    (lm, _), (ln, _) = (_local(cg, b) for b in blocks)
    # A thread past the block on a wider sibling axis has no element; its
    # read stays inside the tile, and its value is masked off downstream.
    return cg.lift(
        expr_from_string(
            f"({buf}[({lm}) * {cols} + ({ln})] if ({lm}) < {rows} and ({ln}) < {cols} "
            "else 0.0)"
        ),
        prefix="_mpp_v",
    )


def _acc_expr(
    cg: GenerateAST, group: MppGroup, fill: str | None, seed: str | None
) -> ast.AST:
    """``_helion_mpp_acc(...)``: the group's op and cooperative accumulator, filled
    with the constant ``fill`` or from the staged tile ``seed``."""
    tile = _tile(cg, (group.m_block, group.n_block, group.k_block))
    tile[2] *= group.k_factor
    if any(b > limit for b, limit in zip(tile, _MAX_TILE, strict=True)):
        raise _unsupported(
            f"an MPP matmul tile {tile[0]}x{tile[1]}x{tile[2]}: one op covers M, "
            "N <= 128 and K <= 256"
        )
    m, n, k = tile
    block_m, block_n = m, n
    # A size-1 M or N (a size-1 tensor dim resolves its block to 1) gets the
    # smallest descriptor dim MPP takes.  A view's extent on it stays 1, and
    # MPP edge-checks every handle against its extents
    # (MPPTensorOpsMatMul2d.h), so the rows past it are never read; the
    # element loop skips the elements past the block.
    for staged, size in ((group.lhs.staged, m), (group.rhs.staged, n)):
        if size == 1 and staged:
            raise _unsupported("a staged MPP operand over a size-1 M or N")
    if m == 1:
        m = 8 if n % 16 == 0 else 16
    if n == 1:
        n = 8 if m % 16 == 0 else 16
    # MPP's static_asserts for a SIMD-group scope (MPPTensorOpsMatMul2dImpl.h).
    if m % 8 or n % 8 or (m % 16 and n % 16) or k % 16:
        raise _unsupported(
            f"an MPP matmul tile {m}x{n}x{k}: matmul2d needs M and N multiples of "
            "8, one of them of 16, and K a multiple of 16"
        )
    binds = _virtual_thread(cg, group)
    dtype = CompileEnvironment.current().backend.dtype_str(group.lhs.dtype)
    staged = [int(o.staged) for o in (group.lhs, group.rhs)]
    return expr_from_string(
        f"{ACC}({group.gid}, {m}, {n}, {k}, '{dtype}', "
        f"{int(_transposed(group.lhs))}, {int(_transposed(group.rhs))}, "
        f"{fill}, [{', '.join(binds)}], {staged[0]}, {staged[1]}, "
        f"{block_m}, {block_n}, {seed!r})"
    )


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
        return _BlockLayout(f"tid[{planned[0]}]", planned[0])
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
    in another block's index would need another tile."""
    from .backend import MetalBackend

    backend = CompileEnvironment.current().backend
    assert isinstance(backend, MetalBackend)
    allowed = {group.m_block, group.n_block, group.k_block}
    for states in cg.active_device_loops.values():
        for st in states:
            for block, axis in st.block_thread_axes.items():
                if block not in allowed and st.thread_axis_sizes.get(axis, 1) > 1:
                    raise _unsupported(
                        f"block {block} is spread over threads around a cooperative "
                        "MPP matmul; only its M, N and K blocks may hold threads"
                    )
    for block, (_axis, size) in backend.free_arange_plan.items():
        if block not in allowed and size > 1:
            raise _unsupported(
                f"free block {block} is spread over threads around a cooperative "
                "MPP matmul; only its M, N and K blocks may hold threads"
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


def _local(cg: GenerateAST, block: int) -> tuple[str, int | None]:
    """The thread's index within the tile on *block*, and the thread axis
    that holds the block (None when it is on lanes only)."""
    layout = _block_layout(cg, block)
    return layout.local, layout.axis


def _stage_writes(
    cg: GenerateAST,
    buf: str,
    parts: Sequence[ast.AST],
    dtype: torch.dtype,
    blocks: tuple[int, int],
    cols: int,
    k_pos: int | None = None,
    k_factor: int = 1,
) -> None:
    """Write the elements this thread holds into the row-major tile ``buf``
    at its index within the tile (part ``r`` of a factored K at ``c * k +
    r``).  A thread whose index on a block lies past the block writes nothing
    (a wider sibling's surplus threads), and of the threads that hold the
    same element, because they differ only on another block's axis, the one
    at 0 writes it."""
    backend = CompileEnvironment.current().backend
    locals_ = [_local(cg, b) for b in blocks]
    pred = [
        f"{loc} < {b}" for (loc, _), b in zip(locals_, _tile(cg, blocks), strict=True)
    ]
    owned = {axis for _, axis in locals_ if axis is not None}
    owner = " and ".join(f"tid[{a}] == 0" for a in range(3) if a not in owned)
    if owner:
        # A name, not tid[...] in the write itself: the shared lane-loop
        # placement takes a subscripted name in a call for a tensor the call
        # may write.
        pred.append(cg.lift(expr_from_string(owner), prefix="_mpp_owner").id)
    for r, part in enumerate(parts):
        index = [loc for loc, _ in locals_]
        if k_pos is not None and k_factor != 1:
            index[k_pos] = f"{k_factor} * {index[k_pos]} + {r}"
        flat = f"({index[0]}) * {cols} + ({index[1]})"
        cg.add_statement(
            statement_from_string(
                f"{STAGE}({buf}, {flat}, {{value}}, {' and '.join(pred)})",
                value=backend.cast_ast(part, dtype),
            )
        )


def _stage(
    cg: GenerateAST,
    group: MppGroup,
    dtype: torch.dtype,
    value: object,
    blocks: tuple[int, int],
    tile: list[int],
    k_pos: int,
) -> list[str]:
    """Write a staged operand into a threadgroup tile and return its
    ``_helion_mpp_run`` arguments."""
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
    _stage_writes(cg, buf, parts, dtype, blocks, cols, k_pos, group.k_factor)
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
    assert isinstance(acc, ast.Name)
    _check_threads(cg, group)
    tiles: list[int] = []
    tile_dims: list[str] = []
    parts: list[list[str]] = []
    anchors: list[tuple[int, str | None]] = []
    outers: list[list[tuple[str, int, int, str | None]]] = []
    pos = 2
    for operand, value, blocks, k_pos in (
        (group.lhs, values[0], (group.m_block, group.k_block), 1),
        (group.rhs, values[1], (group.k_block, group.n_block), 0),
    ):
        tile = _tile(cg, blocks)
        tile[k_pos] *= group.k_factor
        tiles += tile
        view = operand.view
        if view is None:
            op_parts = _stage(cg, group, operand.dtype, value, blocks, tile, k_pos)
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


@addmm_lowering.register_codegen("metal")
def codegen_addmm_metal(ctx: LoweringContext, node: Node) -> object:
    from ..generate_ast import GenerateAST

    cg = ctx.cg
    assert isinstance(cg, GenerateAST)
    acc, lhs, rhs = (ctx.env[_node(node.args[i])] for i in _MATMULS[_ADDMM])
    return _codegen_matmul(cg, node, acc, (lhs, rhs))


@mm_lowering.register_codegen("metal")
def codegen_mm_metal(ctx: LoweringContext, node: Node) -> object:
    raise _unsupported(
        "torch.mm, a matmul with no accumulator; "
        "accumulate it over a K loop with torch.addmm(acc, a, b)"
    )


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


def _reads(node: ast.AST) -> set[str]:
    """The names a statement or expression reads; an element loop's own
    definitions are not reads."""
    if isinstance(node, ast.For) and _call_name(node.iter) == ELEMENTS:
        inner: set[str] = set()
        defs: set[str] = set()
        for s in node.body:
            inner |= _reads(s)
            if (t := _target(s)) is not None:
                defs.add(t)
            elif isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
                if _call_name(s.value) == BIND:
                    defs.add(ast.literal_eval(s.value.args[0]))
        return (inner - defs) | _reads(node.iter)
    if isinstance(node, (ast.For, ast.If)):
        out = _reads(node.iter if isinstance(node, ast.For) else node.test)
        for s in (*node.body, *node.orelse):
            out |= _reads(s)
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

    The *recomputed* names are the bound names, the cooperative names and
    every name assigned from one of them, never a collective's target (a
    leaf); the element loop re-runs exactly those.
    """

    def __init__(
        self, body: list[ast.stmt], bound: set[str], coop: set[str], leaves: set[str]
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
        self.recomputed = bound | coop
        changed = True
        while changed:
            changed = False
            for a in assigns:
                t = _target(a)
                if (
                    t is not None
                    and t not in self.recomputed
                    and t not in leaves
                    and _reads(a.value) & self.recomputed
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
                work += _reads(a.value)
        return out

    def take(self, s: ast.stmt) -> list[ast.stmt]:
        """The copies to emit before ``s``, which is rewritten to read them."""
        t = _target(s)
        if t is None or t not in self.chain or t in self.reassigned or t in self.leaves:
            return []
        assert isinstance(s, ast.Assign)
        names = sorted(
            r
            for r in _reads(s.value)
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
            for r in _reads(expr) & self.reassigned:
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
_CROSS_THREAD_MARKERS = ("_helion_mpp_", "_metal_mpp_", "_metal_lane_")


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
            reads = _reads(a.value)
            if reads & tiles or (
                reads & out and not _cross_thread(a.value, tg_buffers)
            ):
                out.add(t)
                changed = True
    return out


def elementize(body: list[ast.stmt], tg_buffers: frozenset[str]) -> list[ast.stmt]:
    """Move each store of a cooperative value into a loop over the
    cooperative tensor's elements; ``tg_buffers`` are the kernel's
    threadgroup buffer names.

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
                elif isinstance(v, ast.Call) and _call_name(v) == RUN:
                    alias[t] = alias[_node_name(v.args[0])]
                elif isinstance(v, ast.Name) and v.id in alias:
                    alias[t] = alias[v.id]
            if isinstance(s, (ast.For, ast.While, ast.If)):
                scan(s.body)
                scan(s.orelse)

    scan(body)
    if not groups:
        return body
    leaves = _collectives(body, tg_buffers) - set(alias)
    per_thread = _materialized(body, tg_buffers)
    bound = {_bind_name(b) for acc in groups.values() for b in _binds(acc)}
    copies = _Copies(body, bound, set(alias), leaves)
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
                roots_of(_reads(s)) if not isinstance(s, (ast.For, ast.If)) else set()
            )
            if len(roots) > 1:
                raise _unsupported("an element-wise op on two cooperative MPP results")
            if t is not None and roots:
                (derived[t],) = roots
                seen.append(s)
                continue
            if roots:
                if not _is_store(s):
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
                for r in _reads(s):
                    need(r)

        for r in _reads(store_stmt):
            need(r)
        for s in (*chosen.values(), store_stmt):
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
        return ast.For(
            target=ast.Name("_it", ast.Store()),
            iter=ast.Call(
                ast.Name(ELEMENTS, ast.Load()), [ast.Name(root, ast.Load())], []
            ),
            body=loop_body,
            orelse=[],
        )

    return _run_lane_loops_once(process(body, []), set(alias))


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
        live = set().union(*map(_reads, stmts))
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
        return _reads(s) if _is_coop_stmt(s, coop) else None

    jam = LaneJam(loop, run_once)
    if carried := jam.carried():
        raise _unsupported(
            f"a lane loop around an MPP op carries {min(carried)} across lanes, "
            "which a copy of it per lane can't hold"
        )
    if reading := jam.lane_reading_units():
        raise _unsupported(
            "a cooperative MPP statement reads a lane-dependent name: "
            f"{sorted(set().union(*map(_reads, reading)) & jam.dependent)}"
        )
    return jam


def _split_schedulable(loop: ast.For, coop: set[str]) -> bool:
    """Whether ``_split_lane_loop`` takes the loop: one op directly in it,
    beside register statements, staging writes, fences and cooperative
    statements only, and no accumulator seeded from a staged tile, which the
    split would move ahead of the lanes that write the tile."""
    if sum(map(_is_run, loop.body)) != 1:
        return False
    for s in loop.body:
        assigned = _assigned(s)
        value = assigned[1] if assigned is not None else None
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
    return True


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
    here, and ``lane_reduction`` rejects a lane reduction beside a
    materialization.
    """
    assert _split_schedulable(loop, coop)
    (at,) = [i for i, s in enumerate(loop.body) if _is_run(s)]
    others = [s for i, s in enumerate(loop.body) if i != at]
    variant_names = {lane}
    read: set[str] = set()
    for s in others:
        reads = _reads(s)
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
        if _reads(s) & variant_names:
            lanes.append(s)
        else:
            (before if i < at else after).append(s)
    if lanes:
        before.append(ast.For(loop.target, loop.iter, lanes, [], None))
    return [*before, loop.body[at], *after]


def _run_lane_loops_once(stmts: list[ast.stmt], coop: set[str]) -> list[ast.stmt]:
    """Splice in the body of each lane loop that holds a cooperative
    statement: the op reads its whole tile from memory, so the loop's lanes
    would run it again.  Lane-invariant Scalar work beside it then runs once
    per iteration of the enclosing loop, as in Helion's semantics.  A lane
    loop holding the op directly beside lane-variant register and staging
    work only (the op's own K tile, whose lanes compute a staged operand) is
    split at the op.  Any other loop that reads its lane is jammed across its
    lanes (``_jam_lane_loop``), at most ``MAX_LANE_COPIES`` copies of a
    statement deep."""

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

    def splice(ss: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for s in ss:
            if isinstance(s, ast.For) and _call_name(s.iter) != ELEMENTS:
                s.body = splice(s.body)
                lane = s.target.id if isinstance(s.target, ast.Name) else ""
                # slice_lane_*: a persistent reduction's lanes over a ':' K.
                if (
                    lane.startswith(("lane_", "slice_lane_"))
                    and _call_name(s.iter) == "range"
                    and any(_is_coop_stmt(x, coop) for x in s.body)
                ):
                    if not any(lane in _reads(x) for x in s.body):
                        out.extend(s.body)
                    elif _split_schedulable(s, coop):
                        out += _split_lane_loop(s, lane, coop)
                    else:
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
                s.body = splice(s.body)
                s.orelse = splice(s.orelse)
            out.append(s)
        return out

    result = splice(_dead_code(stmts))
    leaked = jammed & set().union(*map(_reads, result))
    assert not leaked, f"lane-dependent names read outside their copies: {leaked}"
    return result
