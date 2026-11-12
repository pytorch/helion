"""The ``cute_cluster_n`` split of a lane-looped tile axis across a thread-block cluster.

Role: planner (the ``cute_cluster_n`` decision of a per-thread strategy).  ``PerThreadNDTileStrategy`` asks the backend once per
strategy, before its first device loop, whether the knob applies
(``CuteBackend.apply_lane_cluster_split``); when it does, the loop's single
lane-looped axis divides its elements across ``cluster_n`` CTAs in addition
to the CTA's own threads (``LaneLoopContext.cluster_n``) and the device
function's ``cute_state.simt_cluster_n`` is set so the host-side call emits
the extra grid dim and the cluster launch shape.  ``cluster_split_races`` is
the memory check this decision shares with the rolled-reduction split
(``CuteLoopedReductionEmitter._maybe_apply_cute_rolled_cluster``).
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import torch
from torch.utils import _pytree as pytree

from ...language import _tracing_ops
from ...language.memory_ops import load
from ...language.memory_ops import store
from ...language.quantized_ops import load_bfloat16_x16_to_float16
from ...language.quantized_ops import load_float4_e2m1fn_x16_to_float16
from ..compile_environment import CompileEnvironment
from ..device_ir import ForLoopGraphInfo
from ..device_ir import WhileLoopGraphInfo
from ..device_ir import control_flow_child_graph_ids
from ..device_ir import device_loop_bounds
from ..device_ir_analysis import TENSOR_ROUTING_OPS
from ..host_function import HostFunction
from ..runtime_tensor_guards import runtime_tensors_are_proven_disjoint
from .loop_nesting import constexpr_live_graph_ids

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState
    from ..tile_strategy import PerThreadNDTileStrategy
    from ..tile_strategy import TileStrategy


def _fake(arg: object) -> object:
    return arg.meta.get("val") if isinstance(arg, torch.fx.Node) else None


def _spans(value: object, block_id: int) -> bool:
    """Whether ``value`` is a tensor with a dim of ``block_id``."""
    env = CompileEnvironment.current()
    return isinstance(value, torch.Tensor) and any(
        env.get_block_id(size) == block_id for size in value.shape
    )


def _slices_long_dim(tensor: torch.Tensor, subscript: list[object]) -> bool:
    """Whether ``subscript`` takes a slice of a dim of ``tensor`` not of size
    1, or leaves such a dim out: one a split axis can stand on."""
    indices = [index for index in subscript if index is not None]
    return any(
        not (isinstance(size, int) and size == 1)
        for size, index in itertools.zip_longest(tensor.shape, indices)
        if size is not None and (index is None or isinstance(index, slice))
    )


def _while_varies(graph: torch.fx.Graph, block_id: int) -> bool:
    """Whether the test a device ``while``'s condition ``graph`` returns
    varies along ``block_id`` (a test it cannot read counts as varying)."""
    [output] = [node for node in graph.nodes if node.op == "output"]
    leaves = pytree.tree_leaves(output.args)
    return len(leaves) != 1 or _spans(_fake(leaves[0]), block_id)


def _overlaps_itself(tensor: torch.Tensor) -> bool:
    """Whether two indices of ``tensor`` may name one element: with its dims
    of more than one element ordered by stride, one's stride does not pass
    the offsets the smaller-strided dims reach.  Symbolic sizes or strides
    count as not overlapping, as everywhere else Helion stores."""
    dims = []
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        if not isinstance(size, int) or not isinstance(stride, int):
            return False
        if size > 1:
            dims.append((abs(stride), size))
    reach = 0
    for stride, size in sorted(dims):
        if stride <= reach:
            return True
        reach += stride * (size - 1)
    return False


def _expr(value: object) -> object:
    """A traced scalar, hashable without a guard: a SymInt as its expression."""
    if isinstance(value, torch.fx.Node):
        value = _fake(value)
    return value._sympy_() if isinstance(value, torch.SymInt) else value


# A memory access: its tensor, whether it is a store, and the element mapping
# of an access the cluster rank slices (None for one every CTA replays).
_Access = tuple[torch.Tensor, bool, object]
# Loads of 16 values at each offset of a tensor (``hl.load_*_x16_to_float16``).
_PACKED_LOADS = (load_bfloat16_x16_to_float16, load_float4_e2m1fn_x16_to_float16)


def _cluster_accesses(strategy: TileStrategy, block_id: int) -> list[_Access] | None:
    """The loads and stores of the kernel's live graphs (CuTe emits only the
    taken side of a branch whose test is a constant for the config:
    ``loop_nesting.constexpr_live_graph_ids``), or None when the check cannot
    tell which memory each CTA reaches: an access of memory no tensor names
    (a stack tensor's buffers may be any tensor's storage), a store to a
    tensor whose static sizes and strides let two indices name one element
    (``_overlaps_itself``: no element mapping of it is one-to-one), a call that
    takes a host tensor other than a load, a store, a packed load
    (``_PACKED_LOADS``, a read) or an op that touches no memory
    (``device_ir_analysis.TENSOR_ROUTING_OPS``, the prefetches among them),
    a loop ``strategy`` emits inside another device loop, ``for`` or
    ``while`` (its begin and subscripts can move between that loop's
    iterations, so equal expressions need not name the same elements), or a
    store with an element mapping of its own (see below) in a device loop
    nested in such a loop: that loop's iterations run it again, each in the
    CTAs its own mask, test or subscript selects (``extra_mask=column.index
    == step * 4096`` in ``for step in hl.grid(2)`` stores in CTA 0, then in
    CTA 1), unordered against each other.  The rule for a loop inside another
    is deliberately broad: it also declines a loop whose slices stay put.
    For a loop that reduces across the cluster inside a loop of several
    trips, declining keeps the config valid, as the cluster reduce's barrier
    has a single phase and may run only once per kernel
    (``validate_cluster_reduce_placement`` rejects it there); inside a loop
    of one trip, the rule gives up a split that would be valid.

    An access is sliced when it is in a device loop ``strategy`` emits (or a
    graph nested in one) and spans ``block_id``: a load whose value has a
    dim of ``block_id``, or a store whose subscript holds the tile of
    ``block_id``, a tensor or a slice.  The slice may be of another axis,
    which every CTA stores whole: counting such a store as sliced only
    declines more, since no load of its tensor shares its element mapping,
    and the mask and branch rule below ignores the slice.  An element
    mapping is the tensor's storage, element size, offset and strides, the
    loop's begin and where each subscript entry starts (a slice's stop only
    bounds it).  A tensor in the subscript (a gather, a scatter or
    ``tile.index + 1``), or the offsets of a packed load, can reach any
    element, so that mapping matches no other; so does a store that does
    not reach along ``block_id`` (through the split tile or a tensor in its
    subscript, or with a value that has a dim of ``block_id``, through a
    slice of a dim not of size 1; its value alone does not count) whose
    value or ``extra_mask`` spans ``block_id``, or that runs under a branch
    or in a device ``while`` whose test spans it (``if tile.index == 0`` or
    ``while i + tile.index < 1``, directly or further out): only the CTAs
    whose slice the mask or the test selects run it, and the CTAs need not
    store one value (``out[row] = column.index * 0 + step`` in an inner
    device loop).
    """
    env = CompileEnvironment.current()
    host = HostFunction.current()
    fn = strategy.fn
    graphs = fn.codegen.codegen_graphs
    live = constexpr_live_graph_ids(host.device_ir, graphs, fn)
    accesses: list[_Access] = []
    parents: dict[int, torch.fx.Node] = {}
    # The bounds of the split loop around each graph nested in it.
    nested_bounds: dict[int, tuple[object, object]] = {}
    # The graphs nested in a device loop.
    in_loop: set[int] = set()
    # The graphs a branch or a while on the split axis runs, and those nested
    # in them.
    divergent: set[int] = set()
    # The graphs a device loop nested in a split loop runs, and those nested
    # in them: what they hold may run several times per split iteration.
    repeated: set[int] = set()
    pending = [*host.device_ir.root_ids]
    seen: set[int] = set()
    while pending:
        info = graphs[pending.pop()]
        if info.graph_id in seen:
            continue
        seen.add(info.graph_id)
        if (
            isinstance(info, ForLoopGraphInfo)
            and fn.tile_strategy.block_id_to_strategy.get(tuple(info.block_ids))
            is strategy
        ):
            bounds = device_loop_bounds(info, parents, block_id)
            # Rolled reductions and ``hl.tile`` loops carry their bounds.
            assert bounds is not None
            if info.graph_id in in_loop:
                # Its slices can move between the outer loop's iterations.
                return None
            repeats = False
        else:
            bounds = nested_bounds.get(info.graph_id)
            repeats = info.graph_id in repeated or (
                bounds is not None
                and isinstance(info, (ForLoopGraphInfo, WhileLoopGraphInfo))
            )
        for node in info.graph.nodes:
            branches = (
                info.graph_id in divergent
                or (
                    node.target is _tracing_ops._if
                    and _spans(_fake(node.args[0]), block_id)
                )
                or (
                    node.target is _tracing_ops._while_loop
                    and _while_varies(graphs[node.args[0]].graph, block_id)
                )
            )
            for child in control_flow_child_graph_ids(node):
                if live is not None and child not in live:
                    continue
                parents.setdefault(child, node)
                if branches:
                    divergent.add(child)
                if bounds is not None:
                    nested_bounds.setdefault(child, bounds)
                if repeats:
                    repeated.add(child)
                if (
                    isinstance(info, (ForLoopGraphInfo, WhileLoopGraphInfo))
                    or info.graph_id in in_loop
                ):
                    in_loop.add(child)
                pending.append(child)
            if node.op != "call_function":
                continue
            if node.target in _PACKED_LOADS:
                tensor = _fake(node.args[0])
                assert isinstance(tensor, torch.Tensor)
                spans = _spans(_fake(node.args[1]), block_id)
                accesses.append(
                    (tensor, False, object() if bounds is not None and spans else None)
                )
                continue
            if node.target not in (load, store):
                if node.target not in TENSOR_ROUTING_OPS and any(
                    isinstance(fake := _fake(leaf), torch.Tensor)
                    and fake in host.tensor_to_origin
                    for leaf in pytree.tree_leaves((node.args, node.kwargs))
                ):
                    return None
                continue
            tensor = _fake(node.args[0])
            if not isinstance(tensor, torch.Tensor) or (
                node.target is store and _overlaps_itself(tensor)
            ):
                return None
            value = _fake(node if node.target is load else node.args[2])
            subscript = node.args[1]
            # Whether the access reaches along the split axis: a load whose
            # value has a dim of ``block_id``, or a store whose subscript
            # holds the split tile or a tensor, or whose value has that dim
            # and whose subscript holds a slice that can stand on it (a
            # rolled reduction's ``out[row, :]``).  A store's value alone
            # does not count: ``out[row] = column.index * 0 + step`` stores
            # the one element ``out[row]`` from every CTA.
            reaches = (
                _spans(value, block_id)
                and (node.target is load or _slices_long_dim(tensor, subscript))
            ) or (
                node.target is store
                and any(
                    isinstance(fake := _fake(index), torch.Tensor)
                    or (
                        isinstance(fake, torch.SymInt)
                        and env.get_block_id(fake) == block_id
                    )
                    for index in subscript
                )
            )
            # A store that does not, whose value or mask, or the test of a
            # branch around it, varies along the split axis, runs only in the
            # CTAs whose slice the mask or the test selects, or stores each
            # CTA's own values: its copies are not identical (``out[row] =
            # 1.0`` where ``column.index == 0`` lands in one CTA of the
            # cluster, unordered against its peers' other stores).  A slice
            # in the subscript does not count: ``out[row, :]`` with a dim of
            # size 1 is the same store.
            masked = (
                node.target is store
                and not reaches
                and (
                    info.graph_id in divergent
                    or _spans(value, block_id)
                    or (len(node.args) > 3 and _spans(_fake(node.args[3]), block_id))
                )
            )
            spans = reaches or (
                node.target is store
                and any(isinstance(index, slice) for index in subscript)
            )
            key = None
            if bounds is not None and (spans or masked):
                if masked or any(
                    isinstance(_fake(index), torch.Tensor) for index in subscript
                ):
                    if repeats and node.target is store:
                        return None
                    key = object()
                else:
                    key = (
                        id(tensor.untyped_storage()),
                        tensor.element_size(),
                        *map(_expr, (tensor.storage_offset(), *tensor.stride())),
                        _expr(bounds[0]),
                        *(
                            (
                                _expr(0 if index.start is None else index.start),
                                _expr(1 if index.step is None else index.step),
                            )
                            if isinstance(index, slice)
                            else _expr(index)
                            for index in subscript
                        ),
                    )
            accesses.append((tensor, node.target is store, key))
    return accesses


def cluster_split_races(strategy: TileStrategy, block_id: int) -> bool:
    """Whether a ``cute_cluster_n`` split of ``block_id`` by ``strategy`` would
    let the cluster's CTAs race on memory the kernel writes.

    Every CTA of the cluster runs the whole kernel: only the sliced accesses
    of ``_cluster_accesses`` divide among the CTAs, and every other access is
    replayed by each CTA, unordered against its peers.  Accesses are grouped
    by the memory they may share: one storage, and the arguments the bound
    alias facts do not prove disjoint (with static shapes every argument has
    its own fake storage, even when the caller passes one tensor twice).

    A written group may be replayed only by stores, and then sliced nowhere
    (identical copies): otherwise a CTA can load ``running[t]`` after a peer
    stored ``mean * m + running[t] * (1 - m)`` and apply the update again,
    read an element before or after the peer whose slice holds it stores
    it, or store over that slice (the check does not order replayed stores
    against the slices).  Its sliced accesses must also share one element
    mapping, or a CTA's slice through one can be a peer's slice through
    another (``x[row, k:]`` beside ``x[row, :-k]``).  An atomic anywhere
    races too.
    """
    if HostFunction.current().device_ir.has_atomic_ops():
        return True
    accesses = _cluster_accesses(strategy, block_id)
    if accesses is None:
        return True
    env = CompileEnvironment.current()
    tensors = {id(tensor.untyped_storage()): tensor for tensor, _, _ in accesses}
    inputs = {id(tensor.untyped_storage()) for tensor in env.input_sources}
    parent = {storage: storage for storage in tensors}

    def group(storage: int) -> int:
        while parent[storage] != storage:
            storage = parent[storage]
        return storage

    for left, right in itertools.combinations(tensors, 2):
        if {left, right} <= inputs and not runtime_tensors_are_proven_disjoint(
            env, tensors[left], tensors[right]
        ):
            parent[group(left)] = group(right)
    groups: dict[int, list[tuple[bool, object]]] = {}
    for tensor, is_store, key in accesses:
        groups.setdefault(group(id(tensor.untyped_storage())), []).append(
            (is_store, key)
        )
    for members in groups.values():
        if not any(is_store for is_store, _ in members):
            continue
        replayed = [is_store for is_store, key in members if key is None]
        keys = {key for _, key in members if key is not None}
        if (replayed and (keys or not all(replayed))) or len(keys) > 1:
            return True
    return False


def apply_cluster_split(
    strategy: PerThreadNDTileStrategy, env: CompileEnvironment, state: CodegenState
) -> None:
    """Record the cluster split of ``strategy``'s lane-looped axis when the
    ``cute_cluster_n`` knob applies to it: a single lane-looped axis whose
    one tile covers the whole extent, no sibling axis on a thread dim, a
    warp-multiple thread count, a per-CTA slice that keeps a whole vec chunk
    per thread and no CTA race on a tensor the kernel writes
    (``cluster_split_races``).
    """
    cl = strategy.fn.config.config.get("cute_cluster_n", 1)
    if not isinstance(cl, int) or cl <= 1:
        return
    if strategy.fn.cute_state.lanes.simt_cluster_n > 1:
        # Another device loop's strategy already claimed the cluster
        # rank; splitting a second axis on the same rank would leave
        # only the diagonal rank x rank blocks covered.
        return
    if len(strategy._lane_var_by_block) != 1:
        return
    # The cluster reduce combines across the WHOLE CTA, so the grid
    # strategy must not put sibling axes on thread dims (e.g. a
    # multi-row grid tile with block_m > 1) — those rows would be
    # folded together.
    grid_axis_sizes = getattr(
        state.codegen.current_grid_state, "thread_axis_sizes", None
    )
    if isinstance(grid_axis_sizes, dict) and any(
        size > 1 for size in grid_axis_sizes.values()
    ):
        return
    [block_id] = strategy._lane_var_by_block.keys()
    vec = strategy.lane_context(block_id).vec_width
    idx = strategy.block_ids.index(block_id)
    nt = strategy._shared_thread_extents.get(block_id, strategy.num_threads[idx])
    static_bs = strategy._configured_block_size_int(strategy.block_size[idx])
    if static_bs is None or nt <= 0 or nt % 32 != 0:
        return
    if not isinstance(env.block_sizes[block_id].size, (int, torch.SymInt)):
        # A tile loop over a range read from a tensor (``hl.tile(begin, begin
        # + n)`` with ``begin = start[0]``) has no extent to slice.
        return
    numel = env.block_sizes[block_id].numel
    try:
        numel_int = int(numel)
    except (TypeError, ValueError):
        return
    # Single trip (the whole extent in one tile) so each cluster reduce
    # call site executes exactly once (fixed mbarrier phase), and the
    # per-CTA slice must keep a whole vec chunk per thread.
    if static_bs < numel_int or static_bs % (nt * cl * vec) != 0:
        return
    # All sibling axes must not use a thread axis (the cluster reduce
    # combines across the whole CTA).
    for other_id, other_bs in zip(strategy.block_ids, strategy.block_size, strict=True):
        if other_id == block_id:
            continue
        extent = strategy._static_thread_extent_for_block(other_id, other_bs)
        if extent is not None and extent != 1:
            return
    # Statements outside the cluster-split loop execute once per
    # cluster CTA — benign only for plain stores of a tensor nothing else
    # accesses.
    if cluster_split_races(strategy, block_id):
        return
    strategy.lane_context(block_id).cluster_n = cl
    strategy.fn.cute_state.lanes.simt_cluster_n = cl
