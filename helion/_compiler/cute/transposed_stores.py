"""Find the tile stores that transpose a tile another access reads.

Role: analysis.

``out_t[tile_v, tile_r] = x.permute(1, 0)`` beside ``x = hl.load(src, [tile_r,
tile_v])`` coalesces only one side: ``src`` wants consecutive threads along
``tile_v``, ``out_t`` along ``tile_r``.  ``transposed_stores`` is the
condition the seeds of ``autotune/heuristics/cute_transposed_store_seeds.py``
key on, read from ``ConfigSpec.memory_op_facts``: a rank-2 store indexed by
two tiles, one on its contiguous dim, beside another rank-2 access of the same
graph indexed by the same two tiles the other way round.  A tile stored both
ways (``out[tile_r, tile_v]`` beside ``out_t``) makes each store the other's
witness; the store a load disagrees with comes first.  ``transposed_accesses``
names those loads and the stores of the tile the other way round, with their
tensors.

``batched_layout_stores`` is the same disagreement with batch tiles around
it, a layout change such as NCHW to channels-last: ``out[tile_n, tile_s,
tile_c] = x[tile_n, tile_c, tile_s].permute(0, 2, 1)`` stores ``tile_c``
contiguously and loads ``tile_s`` contiguously.  It reads the facts from the
graph nodes (``MEMORY_OP_FACT_META``) to hand the seeds the two tensors,
whose packets they prove; ``root_tile_begins_at_multiple`` is the part of
that proof the root loop's iterator answers.

``transposed_accesses`` and ``batched_layout_stores`` serve seeds planted on
a packet proof and read only the graphs in which the lowering takes a tile's
lanes as packets (``_packet_graph_ids``): a root loop's, and a branch's that
codegen folds away.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

import torch

from ...language._tracing_ops import _if
from ..compile_environment import _to_sympy
from ..variable_origin import BlockSizeOrigin

if TYPE_CHECKING:
    from ...autotuner.config_spec import MemoryOpFact
    from ..ast_extension import ExtendedAST
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from ..type_info import TileIndexType


class TransposedStore(NamedTuple):
    """A store ``transposed_stores`` found: the block of its other dim, the
    block indexing its contiguous dim, and its element size."""

    other: int
    contiguous: int
    itemsize: int


class TransposedAccess(NamedTuple):
    """An access ``transposed_accesses`` found, a load or (``stored``) a
    store: the tensor, its contiguous dim, whether the access has an
    ``extra_mask`` and whether a subscript is a tile's index plus a scalar
    (``_shifted``; of a load, only the subscript of its contiguous dim
    counts)."""

    tensor: torch.Tensor
    lane_dim: int
    masked: bool
    shifted: bool
    stored: bool


class BatchedLayoutStore(NamedTuple):
    """A store ``batched_layout_stores`` found, with the first load that
    disagrees with it: the block the load has on its contiguous dim, the
    block on the store's own contiguous dim, the store's other blocks (its
    batch tiles, in subscript order), then the stored tensor, its contiguous
    dim, whether the store has an ``extra_mask`` and whether one of its
    subscripts is a tile's index plus a scalar (``_shifted``), and the same
    four of the load, of which only the subscript of its contiguous dim
    counts as shifted."""

    other: int
    contiguous: int
    batch: tuple[int, ...]
    tensor: torch.Tensor
    lane_dim: int
    masked: bool
    shifted: bool
    load_tensor: torch.Tensor
    load_lane_dim: int
    load_masked: bool
    load_shifted: bool


def _tile_dims(fact: MemoryOpFact) -> tuple[int, int] | None:
    """``(contiguous, other)``: the blocks of a rank-2 access indexed by two
    tiles, the first on its stride-1 dim, else ``None``."""
    blocks = fact.subscript_block_ids
    if fact.ndim != 2 or len(blocks) != 2 or blocks[0] == blocks[1]:
        return None
    first, second = blocks
    if first is None or second is None or 0 in fact.subscript_strides:
        # A broadcast dim (stride 0) reads one element for its whole tile:
        # nothing along it is laid out, let alone transposed.
        return None
    unit = [
        block
        for block, stride in zip(blocks, fact.subscript_strides, strict=True)
        if stride == 1
    ]
    if unit == [first]:
        return first, second
    if unit == [second]:
        return second, first
    return None


def transposed_stores(
    env: CompileEnvironment, device_ir: DeviceIR
) -> list[TransposedStore]:
    """The rank-2 stores indexed by two tiles that another rank-2 access of
    their graph indexes the other way round (the store's other tile on its
    contiguous dim).  Stores whose other tile a load reads contiguously come
    first, so a tile also stored untransposed (``out[tile_r, tile_v]`` beside
    ``out_t[tile_v, tile_r]``, each the other's witness) does not steer the
    seeds against the loads."""
    spec = env.config_spec
    if spec.matmul_facts or len(device_ir.phases) > 1:
        return []
    graphs: dict[int, list[tuple[MemoryOpFact, tuple[int, int] | None]]] = {}
    for fact in spec.memory_op_facts:
        graphs.setdefault(fact.graph_id, []).append((fact, _tile_dims(fact)))
    against_loads: list[TransposedStore] = []
    against_stores: list[TransposedStore] = []
    for accesses in graphs.values():
        for fact, dims in accesses:
            if fact.kind != "store" or dims is None:
                continue
            # A rank-2 access has a fake value, hence a dtype.
            assert fact.dtype is not None
            contiguous, other = dims
            witnesses = {
                candidate.kind
                for candidate, candidate_dims in accesses
                if candidate is not fact and candidate_dims == (other, contiguous)
            }
            if witnesses:
                found = TransposedStore(other, contiguous, fact.dtype.itemsize)
                (against_loads if "load" in witnesses else against_stores).append(found)
    return [*against_loads, *against_stores]


def _packet_graph_ids(device_ir: DeviceIR) -> list[int]:
    """The graphs in which the lowering takes a tile's lanes as packets: the
    root loops', and those of the branches in them that codegen folds away
    (``if tile.block_size > 32:``).  A branch is folded when its test is one
    ``DeviceFunction.evaluate_constexpr_condition`` resolves from a config:
    symbolic, every free symbol a block size.  Under a test that is a
    tensor's value or that reads a tile's begin, and in a nested loop,
    graphs of their own too, the lowering loads and stores the lanes one by
    one."""
    # The heuristics are imported while ``device_ir`` still is.
    from ..device_ir import control_flow_child_graph_ids

    host_function = device_ir.host_function
    assert host_function is not None
    origins = host_function.expr_to_origin
    found: list[int] = []
    pending = list(device_ir.root_ids)
    while pending:
        graph_id = pending.pop(0)
        found.append(graph_id)
        for node in device_ir.graphs[graph_id].graph.nodes:
            if node.op != "call_function" or node.target is not _if:
                continue
            # ``_if(test, if_graph_id, else_graph_id, ...)``.
            test = node.args[0]
            value = test.meta["val"] if isinstance(test, torch.fx.Node) else test
            if isinstance(value, torch.SymBool) and all(
                (origin := origins.get(symbol)) is not None
                and isinstance(origin.origin, BlockSizeOrigin)
                for symbol in value.node.expr.free_symbols
            ):
                pending.extend(control_flow_child_graph_ids(node))
    return found


def transposed_accesses(
    device_ir: DeviceIR, store: TransposedStore
) -> list[TransposedAccess]:
    """The rank-2 accesses that have ``store``'s other tile on their
    contiguous dim, where the lowering can take them as packets: the loads
    it transposes and the stores of the tile the other way round.  Empty
    when one of them, the store or another rank-2 access of the two tiles
    sits outside the packet graphs (``_packet_graph_ids``).  Read from the
    graph nodes (``MEMORY_OP_FACT_META``), as ``batched_layout_stores``
    reads its accesses, to hand the seeds the tensors, whose packets they
    prove."""
    # The heuristics are imported while ``device_ir`` still is.
    from ..device_ir import MEMORY_OP_FACT_META

    packet_graphs = set(_packet_graph_ids(device_ir))
    transposed = (store.other, store.contiguous)
    accesses: list[TransposedAccess] = []
    for graph_id, graph_info in enumerate(device_ir.graphs):
        for node in graph_info.graph.nodes:
            for fact in node.meta.get(MEMORY_OP_FACT_META, ()):
                dims = _tile_dims(fact)
                if dims is None or set(dims) != set(transposed):
                    continue
                if graph_id not in packet_graphs:
                    return []
                if dims != transposed:
                    continue
                stored = fact.kind == "store"
                lane_dim = fact.subscript_block_ids.index(store.other)
                accesses.append(
                    TransposedAccess(
                        # A fact with a rank is the fact of a traced tensor.
                        cast("torch.fx.Node", node.args[0]).meta["val"],
                        lane_dim,
                        # ``load(tensor, index, extra_mask, eviction_policy)``
                        # and ``store(tensor, index, value, extra_mask)``.
                        node.args[3 if stored else 2] is not None,
                        _shifted(node) if stored else _shifted(node, lane_dim),
                        stored,
                    )
                )
    return accesses


def _batched_tile_dims(fact: MemoryOpFact) -> tuple[int, tuple[int, ...]] | None:
    """``(contiguous dim, blocks)`` of an access of rank 3 or more whose every
    dim a distinct tile indexes, exactly one of them the stride-1 dim, else
    ``None``.  The subscripts are then the tensor's dims in order."""
    blocks = fact.subscript_block_ids
    if fact.ndim < 3 or len(blocks) != fact.ndim or len(set(blocks)) != fact.ndim:
        return None
    if None in blocks or 0 in fact.subscript_strides:
        # A broadcast dim (stride 0) is not laid out, as in ``_tile_dims``.
        return None
    unit = [dim for dim, stride in enumerate(fact.subscript_strides) if stride == 1]
    if len(unit) != 1:
        return None
    return unit[0], cast("tuple[int, ...]", blocks)


def _shifted(node: torch.fx.Node, position: int | None = None) -> bool:
    """Whether a subscript of the load or store ``node``, the one at
    ``position`` or any, is a tile's index plus a scalar (``out[tile_n,
    tile_s, tile_c.index + 3]``), which the access's fact reads as the tile
    (``add_tile_with_offset_metadata``)."""
    subscripts = cast("list[object]", node.args[1])
    if position is not None:
        subscripts = [subscripts[position]]
    return any(
        isinstance(subscript, torch.fx.Node) and "tile_with_offset" in subscript.meta
        for subscript in subscripts
    )


def batched_layout_stores(
    env: CompileEnvironment, device_ir: DeviceIR
) -> list[BatchedLayoutStore]:
    """The stores of rank 3 or more of the packet graphs
    (``_packet_graph_ids``), every dim indexed by a tile of its own, whose
    contiguous dim's tile is not the one a load of the same graph over the
    same tiles has on its contiguous dim: a batched layout change (NCHW to
    channels-last and back), once per tile a load has contiguous instead."""
    # The heuristics are imported while ``device_ir`` still is.
    from ..device_ir import MEMORY_OP_FACT_META

    spec = env.config_spec
    if spec.matmul_facts or len(device_ir.phases) > 1:
        return []
    found: list[BatchedLayoutStore] = []
    for graph_id in _packet_graph_ids(device_ir):
        accesses: list[
            tuple[MemoryOpFact, tuple[int, tuple[int, ...]], torch.fx.Node]
        ] = []
        for node in device_ir.graphs[graph_id].graph.nodes:
            for fact in node.meta.get(MEMORY_OP_FACT_META, ()):
                dims = _batched_tile_dims(fact)
                if dims is not None:
                    accesses.append((fact, dims, node))
        for fact, (lane_dim, blocks), node in accesses:
            if fact.kind != "store":
                continue
            others: set[int] = set()
            for load, (load_dim, load_blocks), load_node in accesses:
                other = load_blocks[load_dim]
                if (
                    load.kind != "load"
                    or set(load_blocks) != set(blocks)
                    or other == blocks[lane_dim]
                    or other in others
                ):
                    continue
                others.add(other)
                found.append(
                    BatchedLayoutStore(
                        other,
                        blocks[lane_dim],
                        tuple(b for b in blocks if b not in (other, blocks[lane_dim])),
                        # A fact with a rank is the fact of a traced tensor.
                        cast("torch.fx.Node", node.args[0]).meta["val"],
                        lane_dim,
                        # ``store(tensor, index, value, extra_mask)``.
                        node.args[3] is not None,
                        _shifted(node),
                        cast("torch.fx.Node", load_node.args[0]).meta["val"],
                        load_dim,
                        # ``load(tensor, index, extra_mask, eviction_policy)``.
                        load_node.args[2] is not None,
                        _shifted(load_node, load_dim),
                    )
                )
    return found


def root_tile_begins_at_multiple(
    env: CompileEnvironment, device_ir: DeviceIR, block_id: int, multiple: int
) -> bool:
    """Whether ``block_id`` is a tile of a root loop that begins at a known
    multiple of ``multiple``: what ``memory_ops._cute_tile_begins_at_multiple``
    answers from the grid state during codegen, read from the root loop's
    iterator for a seed.  A tile's packets start at its begin plus multiples
    of their width, so those of ``hl.tile(3, n)`` all sit off their
    alignment and the lowering stores scalars.  ``hl.tile(end)`` begins at
    zero and ``hl.tile(begin, end)`` at the traced ``begin``, which counts
    when it is zero or a static multiple; a begin read from a tensor is not
    known, and neither is one of a call that does not spell its ranges out
    (``hl.tile(*ranges)``).  The tile of a nested loop is not a root's."""
    # The heuristics are imported while ``type_info`` still is.
    from ..type_info import IterType
    from ..type_info import SequenceType

    for root in device_ir.root_nodes:
        iterator = root.iter
        assert isinstance(iterator, ast.Call)
        iter_type = cast("ExtendedAST", iterator)._type_info
        assert isinstance(iter_type, IterType)
        inner = iter_type.inner
        tiles = inner.unpack() if isinstance(inner, SequenceType) else [inner]
        block_ids = [cast("TileIndexType", tile).block_id for tile in tiles]
        if block_id not in block_ids:
            continue
        if any(isinstance(arg, ast.Starred) for arg in iterator.args):
            return False
        # ``hl.tile(begin_or_end, end_or_none=None, /, block_size=None)``.
        bounds = []
        for arg in iterator.args[:2]:
            type_info = cast("ExtendedAST", arg)._type_info
            assert type_info is not None
            bounds.append(type_info.proxy())
        if len(bounds) < 2 or bounds[1] is None:
            return True
        begins = bounds[0]
        if isinstance(begins, (list, tuple, torch.Size)):
            begins = begins[block_ids.index(block_id)]
        if not isinstance(begins, (int, torch.SymInt)):
            return False
        begin = _to_sympy(begins)
        return begin == 0 or env.known_multiple(begin, multiple)
    return False
