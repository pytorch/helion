"""One index per position of a repeated ``:`` slice block.

Equal-sized ``:`` slices share one reduction block
(``allocate_reduction_dimension`` caches by size), so with as many heads as
``half``, ``k[tile_b, :, tile_t, :half]`` has the value shape ``[b, r, t, r]``.
Triton broadcasts the block's range into both dimensions.  A Metal thread
holds one index per block, so it would read only their diagonal.

Pointwise ops and stores broadcast right-aligned, so in a kernel built only
from loads, stores, constants and pointwise ops, a block and a position
counted from the right name one coordinate in every value and access.  The
block's lowest position keeps its thread axis (or its starved lane loop).
Every other position gets the thread axis of another ``:`` slice block that
no value meets it with (``_position_hosts``), or else a full-extent lane
loop of its own (``MetalPersistentReductionStrategy.add_position_lanes``),
and loads and stores index each slice dimension with the coordinate of its
position (``memory_ops``).
"""

from __future__ import annotations

import collections
import itertools
from typing import TYPE_CHECKING

import sympy
import torch
from torch.multiprocessing.reductions import StorageWeakRef

from ... import exc
from ...language._tracing_ops import _get_symnode
from ...language._tracing_ops import _host_tensor
from ...language.creation_ops import full
from ...language.memory_ops import load
from ...language.memory_ops import store
from ..compile_environment import CompileEnvironment
from ..compile_environment import _symint_expr
from ..host_function import HostFunction
from ..indexing_strategy import SubscriptIndexing
from ..variable_origin import BlockSizeOrigin
from .reduction_strategy import MetalPersistentReductionStrategy

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..device_ir import GraphInfo
    from ..tile_dispatch import TileStrategyDispatch

_ELEMENTWISE_OPS = (load, store, full, _host_tensor, _get_symnode)


def plan_repeated_blocks(
    graphs: list[GraphInfo], tile_strategy: TileStrategyDispatch
) -> None:
    """Give each block a value or access repeats a thread axis or a lane loop
    per extra position, or raise ``BackendUnsupported`` where the lanes would be wrong.

    Load indices are read as well as values and store targets: ``x[t, :, :h]
    + hl.zeros([t, h, h])`` has a value of concrete sizes, but its load
    repeats the block.
    """
    env = CompileEnvironment.current()
    positions: dict[int, set[int]] = collections.defaultdict(set)
    repeats: dict[int, str] = {}
    for node in _nodes(graphs):
        for what, shape in _shapes(node):
            first_dim: dict[int, int] = {}
            for dim, size in enumerate(shape):
                block_id = env.get_block_id(size)
                if block_id is None:
                    continue
                positions[block_id].add(len(shape) - 1 - dim)
                if block_id in first_dim:
                    repeats.setdefault(
                        block_id,
                        f"{what} of shape {list(shape)} repeats a block in "
                        f"dimensions {first_dim[block_id]} and {dim}",
                    )
                first_dim.setdefault(block_id, dim)
    if not repeats:
        return
    # Values may have concrete sizes where the block is (``hl.zeros``), so
    # the ops touching it can't be read from shapes: check the whole kernel.
    for node in _nodes(graphs):
        if node.op == "call_function" and not _is_elementwise(node):
            raise _repeat_unsupported(
                next(iter(repeats.values())),
                f"the kernel uses {_op_name(node.target)}, and the lanes only "
                "follow loads, stores, constants and pointwise ops",
            )
    residents: dict[int, set[tuple[int, int | None]]] = {}
    for block_id, repeat in repeats.items():
        size = env.block_sizes[block_id].size
        if not isinstance(size, int):
            raise _repeat_unsupported(
                repeat, f"the block's extent {size} is not static"
            )
        strategy = tile_strategy.block_id_to_strategy.get((block_id,))
        if not isinstance(strategy, MetalPersistentReductionStrategy):
            raise _repeat_unsupported(
                repeat, f"the block is a {type(strategy).__name__}, not a `:` slice"
            )
        thread_position, *lane_positions = sorted(positions[block_id])
        _check_single_owner(graphs, block_id, thread_position)
        hosts = _position_hosts(
            graphs, tile_strategy, block_id, lane_positions, size, residents
        )
        strategy.add_position_lanes(lane_positions, size, hosts)


def _position_hosts(
    graphs: list[GraphInfo],
    tile_strategy: TileStrategyDispatch,
    block_id: int,
    lane_positions: list[int],
    extent: int,
    residents: dict[int, set[tuple[int, int | None]]],
) -> dict[int, tuple[int, int]]:
    """Position -> (thread axis, its threads) for each lane position of
    *block_id* that can take another `:` slice block's thread axis instead
    of a lane loop.

    A Metal statement runs on every thread of every axis, so a statement that
    never meets the axis's block repeats its work once per thread there; a
    position lane's statements meet only the blocks of their own values.  The
    position takes axis A, held by slice block B with at least *extent*
    threads, when no value, load or store depends (through its inputs as well
    as its shape) on both the position and on anything already on A.  Its
    index is then ``tid[A]``: each value and access holding it reads a
    thread's own coordinate, and nothing on A is read beside it.  A store
    whose value or mask depends on the position must also index it, or the
    threads of A would write one element with different values where the
    lane loop wrote it in order.
    """
    # Each grid root numbers its thread axes from 0, so B's axis in one root
    # can be the block's own axis in another.
    if len(HostFunction.current().device_ir.grid_block_ids) != 1:
        return {}
    deps = _dependencies(graphs)
    hosts: dict[int, tuple[int, int]] = {}
    for position in lane_positions:
        dim = (block_id, position)
        for strategy in tile_strategy.strategies:
            if (
                not isinstance(strategy, MetalPersistentReductionStrategy)
                or strategy.block_index == block_id
            ):
                continue
            threads = strategy._reduction_thread_count()
            axis = tile_strategy.thread_axis_for_strategy(strategy)
            if threads < extent or axis is None:
                continue
            on_axis = residents.setdefault(axis, {(strategy.block_index, None)})
            if any(
                dim in dims and any(_holds(dims, r) for r in on_axis)
                for dims in deps.values()
            ):
                continue
            if not _stores_index(graphs, deps, dim):
                continue
            on_axis.add(dim)
            hosts[position] = (axis, threads)
            break
    return hosts


def _holds(dims: set[tuple[int, int]], resident: tuple[int, int | None]) -> bool:
    """Whether *dims* holds *resident*; a None position stands for all of its
    block's positions."""
    block_id, position = resident
    if position is None:
        return any(b == block_id for b, _ in dims)
    return resident in dims


def _dependencies(graphs: list[GraphInfo]) -> dict[torch.fx.Node, set[tuple[int, int]]]:
    """Each node's (block, position) dims: those of its own value, load index
    or store target, and every dim its inputs depend on."""
    deps: dict[torch.fx.Node, set[tuple[int, int]]] = {}
    for node in _nodes(graphs):
        dims: set[tuple[int, int]] = set()
        for _, shape in _shapes(node):
            dims |= _block_positions(shape)
        for inp in node.all_input_nodes:
            dims |= deps[inp]
        deps[node] = dims
    return deps


def _stores_index(
    graphs: list[GraphInfo],
    deps: dict[torch.fx.Node, set[tuple[int, int]]],
    dim: tuple[int, int],
) -> bool:
    """Whether every store whose value or mask depends on *dim* indexes it."""
    for node in _nodes(graphs):
        if node.target is not store:
            continue
        inputs = [a for a in node.args[2:] if isinstance(a, torch.fx.Node)]
        if not any(dim in deps[a] for a in inputs):
            continue
        tensor, subscript = _access(node)
        assert isinstance(tensor, torch.Tensor)
        if dim not in _block_positions(
            SubscriptIndexing.compute_shape(tensor, subscript)
        ):
            return False
    return True


def _block_positions(shape: list[int | torch.SymInt]) -> set[tuple[int, int]]:
    """The (block, position from the right) of each block dim of *shape*."""
    env = CompileEnvironment.current()
    return {
        (block_id, len(shape) - 1 - dim)
        for dim, size in enumerate(shape)
        if (block_id := env.get_block_id(size)) is not None
    }


def slice_output_dims(subscript: list[object]) -> dict[int, tuple[int, slice]] | None:
    """Map each output dim a slice of *subscript* makes to its tensor dim and
    the slice, or None for a tensor index, whose output dims the subscript
    alone doesn't give.

    A SymInt keeps its dim only as a tile, as in
    ``SubscriptIndexing.compute_shape``.
    """
    dims: dict[int, tuple[int, slice]] = {}
    out_dim = tensor_dim = 0
    for k in subscript:
        if k is None:
            out_dim += 1
        elif isinstance(k, slice):
            dims[out_dim] = (tensor_dim, k)
            out_dim += 1
            tensor_dim += 1
        elif isinstance(k, int):
            tensor_dim += 1
        elif isinstance(k, torch.SymInt):
            symbol = _symint_expr(k)
            if isinstance(symbol, sympy.Symbol):
                origin = HostFunction.current().expr_to_origin.get(symbol)
                if origin is not None and isinstance(origin.origin, BlockSizeOrigin):
                    out_dim += 1
            tensor_dim += 1
        else:
            return None
    return dims


def slice_start_step(k: slice) -> tuple[object, object]:
    return (0 if k.start is None else k.start, 1 if k.step is None else k.step)


def _check_single_owner(
    graphs: list[GraphInfo], block_id: int, thread_position: int
) -> None:
    """Reject a written tensor that two threads, or two lane iterations of one
    thread, could reach at one element through *block_id*.

    The lanes add no barrier, and a lane loop runs a load and a store of
    iteration ``h`` before iteration ``h + 1``, where a tile statement reads
    its whole tile first.  So each element of a written tensor needs one
    owner: one thread at one lane iteration.  Accesses are grouped by fake
    storage, so host views of one tensor fall together.  In a group with a
    store, where some access holds the block, every access must:

    - go through one fake tensor, not two views of the storage;
    - hold the block at *thread_position*, and at each of its positions come
      from a slice with an int start and step;
    - index one tensor dim ``D`` at *thread_position*;
    - for every other access, cover a disjoint range of ``D`` at
      *thread_position*, or have an equal subscript.

    Equal subscripts index each element at the same coordinate of every
    block, so from one thread at one lane iteration, where the lane
    distribution keeps their program order.  Disjoint ranges of ``D`` never
    meet.  Reads from many threads are safe, so read-only tensors pass.
    """
    env = CompileEnvironment.current()
    size = env.block_sizes[block_id].size
    assert isinstance(size, int)
    groups: dict[StorageWeakRef, list[tuple[torch.Tensor, list[object]]]] = (
        collections.defaultdict(list)
    )
    written: set[StorageWeakRef] = set()
    for node in _nodes(graphs):
        if node.target not in (load, store):
            continue
        tensor, subscript = _access(node)
        if not isinstance(tensor, torch.Tensor):
            continue
        key = StorageWeakRef(tensor.untyped_storage())
        groups[key].append((tensor, subscript))
        if node.target is store:
            written.add(key)
    for key in written:
        accesses = groups[key]
        shapes = list(itertools.starmap(SubscriptIndexing.compute_shape, accesses))
        if not any(env.get_block_id(s) == block_id for s in itertools.chain(*shapes)):
            continue
        if len({id(tensor) for tensor, _ in accesses}) > 1:
            raise _owner_unsupported("it is reached through two views of its storage")
        ranges: list[tuple[int, int, int]] = []
        for (_, subscript), shape in zip(accesses, shapes, strict=True):
            mapping = _block_mapping(subscript, shape, block_id)
            if mapping is None or thread_position not in mapping:
                raise _owner_unsupported(
                    f"an access of shape {list(shape)} doesn't hold the block at "
                    f"its thread position {thread_position} through slices with "
                    "an int start and step"
                )
            dim, start, step = mapping[thread_position]
            ranges.append((dim, start, start + (size - 1) * step))
        if len({dim for dim, _, _ in ranges}) > 1:
            raise _owner_unsupported(
                f"its accesses put the block's thread position on tensor dims "
                f"{sorted({dim for dim, _, _ in ranges})}"
            )
        keys = [
            _subscript_key(subscript, shape)
            for (_, subscript), shape in zip(accesses, shapes, strict=True)
        ]
        for i, j in itertools.combinations(range(len(accesses)), 2):
            (dim, lo_i, hi_i), (_, lo_j, hi_j) = ranges[i], ranges[j]
            if (hi_i < lo_j or hi_j < lo_i) or keys[i] == keys[j]:
                continue
            raise _owner_unsupported(
                f"accesses {accesses[i][1]} and {accesses[j][1]} overlap on "
                f"tensor dim {dim} ([{lo_i}, {hi_i}] and [{lo_j}, {hi_j}]) "
                "with different subscripts"
            )


def _block_mapping(
    subscript: list[object], shape: list[int | torch.SymInt], block_id: int
) -> dict[int, tuple[int, int, int]] | None:
    """Position -> (tensor dim, start, step) of every position *block_id*
    holds in an access of index *shape*, or None if one isn't a slice with an
    int start and step."""
    env = CompileEnvironment.current()
    dims = slice_output_dims(subscript)
    if dims is None:
        return None
    mapping: dict[int, tuple[int, int, int]] = {}
    for out_dim, size in enumerate(shape):
        if env.get_block_id(size) != block_id:
            continue
        if out_dim not in dims:
            return None
        tensor_dim, k = dims[out_dim]
        start, step = slice_start_step(k)
        if not (isinstance(start, int) and isinstance(step, int)):
            return None
        mapping[len(shape) - 1 - out_dim] = (tensor_dim, start, step)
    return mapping


def _subscript_key(
    subscript: list[object], shape: list[int | torch.SymInt]
) -> tuple[object, ...]:
    """What makes two accesses of one tensor touch the same elements at the
    same coordinates: each entry, with a slice's start and step, and each
    output dim's block or size (a slice's extent)."""
    env = CompileEnvironment.current()
    entries = tuple(
        ("slice", *map(str, slice_start_step(k))) if isinstance(k, slice) else str(k)
        for k in subscript
    )
    sizes = tuple(
        ("block", block_id) if (block_id := env.get_block_id(s)) is not None else str(s)
        for s in shape
    )
    return entries, sizes


def _is_elementwise(node: torch.fx.Node) -> bool:
    return node.target in _ELEMENTWISE_OPS or (
        isinstance(node.target, torch._ops.OpOverload)
        and torch.Tag.pointwise in node.target.tags
    )


def _nodes(graphs: list[GraphInfo]) -> Iterator[torch.fx.Node]:
    for graph in graphs:
        yield from graph.graph.nodes


def _access(node: torch.fx.Node) -> tuple[object, list[object]]:
    tensor, subscript = torch.fx.map_arg(node.args[:2], lambda arg: arg.meta["val"])
    assert isinstance(subscript, (list, tuple))
    return tensor, list(subscript)


def _op_name(target: object) -> str:
    if callable(target) and not isinstance(target, torch._ops.OpOverload):
        return target.__name__
    return str(target)


def _shapes(node: torch.fx.Node) -> list[tuple[str, list[int | torch.SymInt]]]:
    value = node.meta.get("val")
    shapes: list[tuple[str, list[int | torch.SymInt]]] = []
    if isinstance(value, torch.Tensor):
        shapes.append(("a value", list(value.shape)))
    if node.target in (load, store):
        tensor, subscript = _access(node)
        if isinstance(tensor, tuple):
            # Mirrors the ops' fakes: a stack tensor is indexed through its first.
            tensor = tensor[0]
        assert isinstance(tensor, torch.Tensor)
        what = "a store target" if node.target is store else "a load index"
        shapes.append((what, SubscriptIndexing.compute_shape(tensor, subscript)))
    return shapes


def _repeat_unsupported(repeat: str, reason: str) -> exc.BackendUnsupported:
    return exc.BackendUnsupported(
        "metal",
        f"{repeat}; a Metal thread holds one index per block, so it would only "
        "cover their diagonal, and lane loops can't give the other dimensions "
        f"their own index because {reason}.  Equal-sized `:` slices share a "
        "block, as do two uses of one tile or of one block size; give one of "
        "the dimensions its own tile",
    )


def _owner_unsupported(reason: str) -> exc.BackendUnsupported:
    return exc.BackendUnsupported(
        "metal",
        "a tensor the kernel writes is accessed so that two threads, or two "
        f"lane iterations, reach one element: {reason}.  A Metal thread owns "
        "the elements at its positions of a repeated `:` slice block",
    )
