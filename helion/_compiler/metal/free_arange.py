"""``hl.arange`` over a block size that no loop owns.

A kernel may build indices as ``base + hl.arange(BLOCK)``, where
``BLOCK = hl.register_block_size(...)`` and only ``hl.grid`` loops enclose
it.  No loop strategy owns such a *free* block, so none gives it a thread
axis, and a Metal thread holds one element of each value, at one coordinate
per block.  Each free block therefore gets a launch thread axis of its own,
above every axis a loop strategy may read (:func:`plan_free_aranges`, run by
``MetalBackend.pre_codegen``), and its iota lowers to
``start + step * tid[axis]``.  The tile thread budget reserves the free
blocks' threads before the tiles take theirs
(``MetalBackend._config_with_reduction_thread_budget``).

Every statement then runs on every thread of a free axis, and a Metal thread
holds each value by block where Helion's values are positional arrays.  The
plan rejects every layout where those differ:

- a reduction, whose cross-thread combine doesn't know the free axis;
- two values meeting position by position with a free block's dim facing a
  different block (:func:`_check_positional_merges`);
- a matmul other than one accumulating a carry of its own single-thread K
  loop (:func:`_check_matmuls`), and any other carry of a lane-looped loop
  (:func:`check_lane_carries`);
- a stored tensor with another access (:func:`_check_write_once`);
- more thread axes than Metal has.

The iota length must be exactly a free block's symbol: a tile-bound, derived
or literal length is rejected by the lowering, except a tile widened by a
constant, which :mod:`factored_dims` holds as per-thread parts.
"""

from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING

import torch
from torch.fx.node import Node
from torch.fx.node import map_arg
from torch.multiprocessing.reductions import StorageWeakRef

from ... import exc
from ...language import _tracing_ops
from ...language.matmul_ops import dot
from ...language.memory_ops import load
from ...language.memory_ops import store
from ..ast_extension import expr_from_string
from ..aten_lowering import arange_default_lowering
from ..aten_lowering import iota_lowering
from ..compile_environment import CompileEnvironment
from ._constants import MAX_THREAD_AXES
from ._constants import MAX_THREADS_PER_THREADGROUP

if TYPE_CHECKING:
    import ast
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Sequence

    from ...runtime.config import Config
    from ..aten_lowering import LoweringContext
    from ..device_ir import GraphInfo
    from ..tile_dispatch import TileStrategyDispatch
    from .factored_dims import FactoredValue
    from .mpp_matmul import MppPlan

#: Each free block's ``(thread axis, threads, lanes)``, by block id: element
#: ``e`` of the block is on thread ``e // lanes``, lane ``e % lanes``.
FreeArangePlan = dict[int, tuple[int, int, int]]

_IOTA_TARGETS = (torch.ops.prims.iota.default, torch.ops.aten.arange.default)

#: The lhs argument of each matmul op.
MATMUL_LHS: dict[object, int] = {
    torch.ops.aten.addmm.default: 1,
    torch.ops.aten.baddbmm.default: 1,
    torch.ops.aten.mm.default: 0,
    torch.ops.aten.bmm.default: 0,
    dot: 0,
}


def free_arange_blocks() -> list[int]:
    """The blocks that some iota's length is exactly, that no loop owns and
    that aren't reduction dimensions, in block-id order.

    Every device-IR graph is scanned: the thread budget asks before the
    codegen graphs exist, and an iota in a rolled copy the config doesn't use
    spans the same block as in its original.
    """
    from ..device_ir import ForLoopGraphInfo
    from ..host_function import HostFunction

    env = CompileEnvironment.current()
    device_ir = HostFunction.current().device_ir
    owned = {block_id for ids in device_ir.grid_block_ids for block_id in ids}
    for graph in device_ir.graphs:
        if isinstance(graph, ForLoopGraphInfo):
            owned.update(graph.block_ids)
    free: set[int] = set()
    for graph in device_ir.graphs:
        for node in graph.graph.nodes:
            if node.op != "call_function" or node.target not in _IOTA_TARGETS:
                continue
            block_id = env.get_block_id(node.meta["val"].shape[0])
            if (
                block_id is not None
                and block_id not in owned
                and not env.block_sizes[block_id].reduction
            ):
                free.add(block_id)
    return sorted(free)


def free_block_threads(config: Config) -> dict[int, tuple[int, int]]:
    """Each free block's ``(threads, lanes)`` within the threadgroup budget
    (``threadgroup_budget``): a block's ``num_threads`` is taken first, and
    blocks left on auto (0) get their size, shrunk largest first as the shared
    planner shrinks auto tiles, until the product fits; lanes walk the rest."""
    from ..backend import largest_divisor_at_most
    from .backend import threadgroup_budget

    env = CompileEnvironment.current()
    budget = threadgroup_budget(config)
    sizes: dict[int, int] = {}
    threads: dict[int, int] = {}
    auto: list[int] = []
    used = 1
    for block_id in free_arange_blocks():
        size = sizes[block_id] = free_block_size(block_id, config)
        configured = int(
            env.config_spec.num_threads.config_get(config.num_threads, block_id, 0)
        )
        if configured > 0:
            threads[block_id] = largest_divisor_at_most(
                size, min(configured, size, max(1, budget // used))
            )
            used *= threads[block_id]
        else:
            threads[block_id] = size
            auto.append(block_id)
    while auto and math.prod(threads.values()) > budget:
        block_id = max(auto, key=lambda b: threads[b])
        if threads[block_id] <= 1:
            break
        threads[block_id] = largest_divisor_at_most(
            sizes[block_id], threads[block_id] - 1
        )
    return {b: (t, sizes[b] // t) for b, t in threads.items()}


def lane_name(block_id: int) -> str:
    """A free block's lane variable; only element loops bind it
    (:func:`check_free_lanes`)."""
    return f"_free_lane_{block_id}"


def free_block_size(block_id: int, config: Config) -> int:
    size = CompileEnvironment.current().block_sizes[block_id].from_config(config)
    # register_block_size allocates a LoopSpec block, whose from_config is an
    # int.
    assert isinstance(size, int)
    return size


def plan_free_aranges(
    graphs: list[GraphInfo], config: Config, tile_strategy: TileStrategyDispatch
) -> FreeArangePlan:
    """Give each free block a thread axis, or raise ``BackendUnsupported``
    where that layout would be wrong.  Empty for a kernel without one."""
    free = free_arange_blocks()
    if not free:
        return {}
    from ..inductor_lowering import ReductionLowering

    live = live_graphs(graphs)
    if any(
        isinstance(node.meta.get("lowering"), ReductionLowering)
        for node in _nodes(live)
    ):
        raise _unsupported(
            "the kernel has a reduction, whose cross-thread combine doesn't know "
            "the free block's thread axis"
        )
    _check_positional_merges(live, set(free))
    _check_matmuls(live, tile_strategy)
    check_lane_carries(live, tile_strategy, config)
    _check_write_once(live)
    return _assign_axes(free, config, tile_strategy)


def check_free_arange_launch(
    plan: FreeArangePlan, tid_axes: set[int], block_dims: Sequence[int]
) -> None:
    """Reject a free axis that a loop's index code also read, or a launch
    over Metal's threadgroup limit.

    Each loop is budgeted on its own, so a root tile, a free axis and an
    inner tile can overflow together; Metal would only refuse the launch when
    it builds the pipeline state.
    """
    if not plan:
        return
    for block_id, (axis, _, _) in plan.items():
        if axis in tid_axes:
            raise _unsupported(
                f"a loop's index code also reads tid[{axis}], block {block_id}'s axis"
            )
    if math.prod(block_dims) > MAX_THREADS_PER_THREADGROUP:
        raise _unsupported(
            f"a threadgroup of {list(block_dims)} threads exceeds "
            f"{MAX_THREADS_PER_THREADGROUP}; lower num_threads or block_sizes"
        )


def _unsupported(why: str) -> exc.BackendUnsupported:
    return exc.BackendUnsupported("metal", f"hl.arange() over a free block: {why}")


def _nodes(graphs: list[GraphInfo]) -> Iterator[Node]:
    for graph in graphs:
        yield from graph.graph.nodes


def _graph_children(node: Node) -> list[int]:
    """The ids of the graphs a control-flow node runs."""
    if node.op != "call_function":
        return []
    if _tracing_ops.is_for_loop_target(node.target):
        ids = node.args[:1]
    elif node.target is _tracing_ops._while_loop:
        ids = (node.args[0], node.args[1], *node.args[3:4])
    elif node.target is _tracing_ops._if:
        ids = node.args[1:3]
    else:
        return []
    return [graph_id for graph_id in ids if isinstance(graph_id, int)]


def live_graphs(graphs: list[GraphInfo]) -> list[GraphInfo]:
    """The roots and every graph they run, once each.

    Rolled reduction copies that this config doesn't use are not among them,
    and ``_apply_rolling`` points an original's id at its rolled graph, so
    graphs are told apart by their ``fx.Graph``.
    """
    from ..host_function import HostFunction

    todo = [graphs[root] for root in HostFunction.current().device_ir.root_ids]
    seen: set[int] = set()
    live: list[GraphInfo] = []
    while todo:
        graph = todo.pop()
        if id(graph.graph) in seen:
            continue
        seen.add(id(graph.graph))
        live.append(graph)
        for node in graph.graph.nodes:
            todo.extend(graphs[child] for child in _graph_children(node))
    return live


def _is_one(size: int | torch.SymInt) -> bool:
    return isinstance(size, int) and size == 1


def _check_dim(
    free: set[int],
    value_size: int | torch.SymInt,
    consumer_size: int | torch.SymInt,
    what: str,
) -> None:
    """Reject a dim of a value meeting a dim of its consumer where either
    holds a free block and the other doesn't hold the same block.  A literal
    1 binds nothing."""
    if _is_one(value_size) or _is_one(consumer_size):
        return
    env = CompileEnvironment.current()
    value_block = env.get_block_id(value_size)
    consumer_block = env.get_block_id(consumer_size)
    if (value_block in free or consumer_block in free) and (
        value_block != consumer_block
    ):
        raise _unsupported(
            f"{what} binds a dim of block {value_block} to block {consumer_block}; "
            "a Metal thread holds a value by block, not by position"
        )


def _check_aligned(
    free: set[int],
    value_shape: Sequence[int | torch.SymInt],
    consumer_shape: Sequence[int | torch.SymInt],
    what: str,
) -> None:
    """Check *value_shape* against *consumer_shape* right-aligned.

    A value with non-1 dims beyond the consumer's rank would have each
    replica along a free axis hold its own element of them.
    """
    extra = len(value_shape) - len(consumer_shape)
    if extra > 0 and not all(map(_is_one, value_shape[:extra])):
        raise _unsupported(
            f"{what} has more dims than the value it meets "
            f"({list(value_shape)} against {list(consumer_shape)})"
        )
    for value_size, consumer_size in zip(
        reversed(value_shape), reversed(consumer_shape), strict=False
    ):
        _check_dim(free, value_size, consumer_size, what)


def _broadcast_positions(
    value_shape: Sequence[int | torch.SymInt],
    result_shape: Sequence[int | torch.SymInt],
) -> list[int]:
    """The result dim that each dim of a lower-rank pointwise operand meets.

    A port of the placement ``TileStrategyDispatch.broadcast_expand_dims`` gives
    Triton: a block dim takes the first unused result dim of the same
    canonical block, a non-block dim the first unused non-block result dim of
    an equal size, and a dim that matched neither the last unused dim.  The
    expansion only inserts ``None``, so the operand's dims take the matched
    positions in their own order.

    It skips the compaction of a flattened loop's blocks into one dim: a free
    block has no strategy to be merged by, and Helion keeps a loop flattened
    only when every tensor holds all of its blocks, adjacent and in order, or
    none of them (``FlattenedTileStrategy.update_allow_flattened``).
    """
    env = CompileEnvironment.current()

    def canonical(size: int | torch.SymInt) -> int | None:
        block_id = env.resolve_block_id(size)
        return None if block_id is None else env.canonical_block_id(block_id)

    result_blocks = [canonical(size) for size in result_shape]
    matched: list[int] = []
    for size in value_shape:
        block_id = canonical(size)
        unused = [p for p in range(len(result_shape)) if p not in matched]
        if block_id is not None:
            match = next((p for p in unused if result_blocks[p] == block_id), None)
        else:
            match = next(
                (
                    p
                    for p in unused
                    if result_blocks[p] is None
                    and env.known_equal(size, result_shape[p])
                ),
                None,
            )
        matched.append(max(unused) if match is None else match)
    return sorted(matched)


def _check_indexers(free: set[int], subscript: list[object], what: str) -> None:
    """Check index tensors that one subscript broadcasts together.

    ``SubscriptIndexing.compute_shape`` right-aligns them into
    ``tensor_indexer_broadcast_shape``, so they meet position by position.
    All-1-D index tensors form a Cartesian product instead, one output dim
    each.
    """
    env = CompileEnvironment.current()
    if not env.should_broadcast_tensor_indexers(subscript):
        return
    tensors = [k for k in subscript if isinstance(k, torch.Tensor)]
    if len(tensors) > 1 and all(t.ndim == 1 for t in tensors):
        return
    joint = env.tensor_indexer_broadcast_shape(tensors)
    for tensor in tensors:
        _check_aligned(free, tensor.shape, joint, f"{what} index tensor")


def _accessed_tensor(node: Node) -> torch.Tensor:
    """The fake tensor a load or store accesses; a stack tensor through its
    first, as the ops' fakes do."""
    tensor = node.args[0]
    if isinstance(tensor, (list, tuple)):
        tensor = tensor[0]
    assert isinstance(tensor, Node)
    fake = tensor.meta["val"]
    assert isinstance(fake, torch.Tensor)
    return fake


def _check_positional_merges(live: list[GraphInfo], free: set[int]) -> None:
    """Wherever two values meet position by position, a free block's dim must
    meet the same block.

    That is: a store's value and mask against its target, a load's mask
    against its value, index tensors broadcast together, pointwise operands
    (right-aligned at the result's rank, at Triton's placement below it),
    ``where`` operands (``tl.where`` right-aligns them at every rank) and a
    carry's assigned value against its initial one.  Every other node Metal
    lowers takes at most one tensor, or is a matmul, whose accumulator is a
    carry under :func:`_check_matmuls`.
    """
    from ..indexing_strategy import SubscriptIndexing
    from ..inductor_lowering import PointwiseLowering

    for node in _nodes(live):
        val = node.meta.get("val")
        if node.target is store:
            subscript = map_arg(node.args[1], lambda arg: arg.meta["val"])
            assert isinstance(subscript, (list, tuple))
            _check_indexers(free, list(subscript), "a store's")
            shape = SubscriptIndexing.compute_shape(
                _accessed_tensor(node), list(subscript)
            )
            value = node.args[2]
            if isinstance(value, Node) and isinstance(value.meta["val"], torch.Tensor):
                _check_aligned(free, value.meta["val"].shape, shape, "a store")
            mask = node.args[3] if len(node.args) > 3 else None
            if isinstance(mask, Node):
                _check_aligned(free, mask.meta["val"].shape, shape, "a store mask")
        elif node.target is load and isinstance(val, torch.Tensor):
            subscript = map_arg(node.args[1], lambda arg: arg.meta["val"])
            assert isinstance(subscript, (list, tuple))
            _check_indexers(free, list(subscript), "a load's")
            mask = node.args[2] if len(node.args) > 2 else None
            if isinstance(mask, Node):
                _check_aligned(free, mask.meta["val"].shape, val.shape, "a load mask")
        elif isinstance(node.meta.get("lowering"), PointwiseLowering) and isinstance(
            val, torch.Tensor
        ):
            for arg in node.all_input_nodes:
                operand = arg.meta.get("val")
                if not isinstance(operand, torch.Tensor) or operand.ndim == 0:
                    continue
                what = f"{node.name}'s operand"
                if operand.ndim >= val.ndim:
                    _check_aligned(free, operand.shape, val.shape, what)
                    continue
                positions = _broadcast_positions(operand.shape, val.shape)
                for size, position in zip(operand.shape, positions, strict=True):
                    _check_dim(
                        free,
                        size,
                        val.shape[position],
                        f"{node.name}'s lower-rank operand",
                    )
        elif node.target is torch.ops.aten.where.self and isinstance(val, torch.Tensor):
            for arg in node.all_input_nodes:
                operand = arg.meta.get("val")
                if isinstance(operand, torch.Tensor):
                    _check_aligned(
                        free, operand.shape, val.shape, f"{node.name}'s operand"
                    )
        elif node.target is _tracing_ops._phi:
            # A loop's or a branch's output has the shape of the value its
            # body assigns, so ``merged`` shows what meets the carry's slot.
            initial, merged = (
                arg.meta.get("val") if isinstance(arg, Node) else arg
                for arg in node.args
            )
            if isinstance(initial, torch.Tensor) and isinstance(merged, torch.Tensor):
                _check_aligned(
                    free, merged.shape, initial.shape, f"the carry {node.name}"
                )


def _matmul_k_block(node: Node) -> int | None:
    lhs = node.args[MATMUL_LHS[node.target]]
    assert isinstance(lhs, Node)
    return CompileEnvironment.current().get_block_id(lhs.meta["val"].shape[-1])


def _is_matmul(node: Node) -> bool:
    return node.op == "call_function" and node.target in MATMUL_LHS


def _check_matmuls(live: list[GraphInfo], tile_strategy: TileStrategyDispatch) -> None:
    """Every matmul must accumulate one carry of the device loop it sits in,
    over that loop's own block, on one thread.

    Then the K tile is this thread's lanes (or one element), and folding the
    lanes into the carry is exactly the contraction: no combine crosses a
    thread, so none has to know the free axis.
    """
    from ..device_ir import ForLoopGraphInfo

    for graph in live:
        for node in graph.graph.nodes:
            if not _is_matmul(node):
                continue
            k_block = _matmul_k_block(node)
            if not (
                isinstance(graph, ForLoopGraphInfo) and graph.block_ids == [k_block]
            ):
                raise _unsupported(
                    f"{node.name}'s K dim (block {k_block}) is not the block of "
                    "the device loop it sits in"
                )
            threads = tile_strategy.thread_extent_for_block_id(graph.block_ids[0])
            if (threads or 1) > 1:
                raise _unsupported(
                    f"{node.name}'s K dim (block {k_block}) spans {threads} threads; "
                    "beside a free block Metal keeps a matmul's K on one thread"
                )
            why = _not_accumulating(live, graph, node)
            if why is not None:
                raise _unsupported(
                    f"{node.name} does not accumulate a carry of its K loop ({why})"
                )


def _accumulator(node: Node) -> object:
    if node.target is dot:
        return node.args[2] if len(node.args) > 2 else None
    if node.target in (torch.ops.aten.addmm.default, torch.ops.aten.baddbmm.default):
        return node.args[0]
    return None


def _not_accumulating(
    live: list[GraphInfo], graph: GraphInfo, node: Node
) -> str | None:
    """Why *node* isn't ``carry = matmul(carry, ...)`` of *graph*'s loop, or
    None if it is.

    The pairing of the body's output with its carry is read from the parent's
    ``_for_loop`` args and ``_phi``, not assumed from their order.
    """
    acc = _accumulator(node)
    if not (
        isinstance(acc, Node)
        and acc.target is _tracing_ops._new_var
        and isinstance(placeholder := acc.args[0], Node)
        and placeholder.op == "placeholder"
        and list(acc.users) == [node]
        # ``a[tm, tk]`` inside the loop reads ``tm``'s size from the carry.
        and all(
            user is acc or user.target is torch.ops.aten.sym_size.int
            for user in placeholder.users
        )
    ):
        return "its accumulator is not a loop input used only by it"
    output = graph.graph.find_nodes(op="output")[0]
    results = list(output.args[0])
    if list(node.users) != [output] or results.count(node) != 1:
        return "its result has a use other than the loop's output"
    position = results.index(node)
    placeholders = graph.graph.find_nodes(op="placeholder")
    for parent in live:
        for loop in parent.graph.nodes:
            if not (
                _tracing_ops.is_for_loop_target(loop.target)
                and loop.args[0] == graph.graph_id
            ):
                continue
            inputs = list(loop.args[3])
            for item in loop.users:
                if item.target is not operator.getitem or item.args[1] != position:
                    continue
                for phi in item.users:
                    if phi.target is not _tracing_ops._phi or phi.args[1] is not item:
                        continue
                    initial = phi.args[0]
                    if (
                        inputs.count(initial) == 1
                        and placeholders[inputs.index(initial)] is placeholder
                    ):
                        return None
            return "its loop output does not carry its accumulator's input"
    return "no parent loop runs it"


def check_lane_carries(
    live: list[GraphInfo],
    tile_strategy: TileStrategyDispatch,
    config: Config,
    unsupported: Callable[[str], exc.BackendUnsupported] = _unsupported,
) -> None:
    """A lane-looped device loop runs its whole body once per lane on one
    thread, so a carried register folds every lane.  Its only allowed carries
    are matmuls over its block, which :func:`_check_matmuls` showed
    accumulate there.

    A loop nested in a lane-looped one needs no rule of its own: a carry
    initialised outside the outer loop is carried by the outer loop too, and
    one initialised inside it is initialised again for each outer lane.
    """
    from ..device_ir import ForLoopGraphInfo

    for graph in live:
        if not (
            isinstance(graph, ForLoopGraphInfo)
            and any(
                _lane_looped(block_id, tile_strategy, config)
                for block_id in graph.block_ids
            )
        ):
            continue
        output = graph.graph.find_nodes(op="output")[0]
        for producer in output.all_input_nodes:
            if producer.op == "placeholder":
                continue
            if not (
                _is_matmul(producer) and graph.block_ids == [_matmul_k_block(producer)]
            ):
                raise unsupported(
                    f"the loop over block(s) {graph.block_ids} carries "
                    f"{producer.name} through a lane loop, which folds every lane "
                    "into one register"
                )


def _lane_looped(
    block_id: int, tile_strategy: TileStrategyDispatch, config: Config
) -> bool:
    """Whether fewer threads than the block's size cover it."""
    size = CompileEnvironment.current().block_sizes[block_id].from_config(config)
    threads = tile_strategy.thread_extent_for_block_id(block_id)
    return isinstance(size, int) and size > 1 and (threads or 1) < size


def _check_write_once(live: list[GraphInfo]) -> None:
    """A tensor the kernel stores to must have no other load or store.

    Every statement runs on every thread of a free axis, and a lane loop
    runs a tile statement's lanes one after another, so any element a thread
    reads may already have been stored, or not yet, by another thread or
    lane.  The replicas of a tensor's one store write identical values: its
    value holds no free block its target lacks, and nothing reduced it.
    Tensors are keyed by fake storage, so views of one tensor fall together.
    """
    accesses: dict[StorageWeakRef, list[str]] = {}
    stored: set[StorageWeakRef] = set()
    for node in _nodes(live):
        if node.target not in (load, store):
            continue
        key = StorageWeakRef(_accessed_tensor(node).untyped_storage())
        accesses.setdefault(key, []).append(node.name)
        if node.target is store:
            stored.add(key)
    for key in stored:
        if len(accesses[key]) > 1:
            raise _unsupported(
                f"a stored tensor has other accesses ({', '.join(accesses[key])}); "
                "threads of a free axis and lanes of a lane loop would race on it"
            )


def _assign_axes(
    free: list[int], config: Config, tile_strategy: TileStrategyDispatch
) -> FreeArangePlan:
    """Give the free blocks the axes above every axis a loop strategy may
    read, one each in block-id order."""
    from ..tile_strategy import PerThreadFlattenedTileStrategy

    highest = -1
    for strategy in tile_strategy.strategies:
        base = tile_strategy.thread_axis_for_strategy(strategy)
        if base is not None and strategy.thread_axes_used() > 0:
            highest = max(highest, base + strategy.thread_axes_used() - 1)
        if isinstance(strategy, PerThreadFlattenedTileStrategy):
            # Its index is ``pid * 1 + tid[axis]`` even at a thread extent of 1.
            highest = max(highest, strategy._flat_thread_axis())
    plan: FreeArangePlan = {}
    threads = free_block_threads(config)
    for axis, block_id in enumerate(free, start=highest + 1):
        if axis >= MAX_THREAD_AXES:
            raise _unsupported(
                f"no thread axis left for block {block_id} (loops claim "
                f"tid[0..{highest}])"
            )
        plan[block_id] = (axis, *threads[block_id])
    return plan


def check_free_lanes(
    plan: FreeArangePlan, mpp_plan: MppPlan, graphs: list[GraphInfo]
) -> None:
    """A free block walked in lanes needs every value that holds it to be
    MPP's: a device view, a cooperative value, or an element-loop store, the
    only code that binds the block's lane (``mpp_matmul._block_layout``).
    Scalar code evaluating such a value has no lane loop to run in."""
    laned = [block_id for block_id, (_, _, lanes) in plan.items() if lanes > 1]
    if not laned:
        return
    from .mpp_matmul import _mpp_coop
    from .mpp_matmul import matmul_operands

    owned: set[Node] = set()
    staged: set[int] = set()
    for group in mpp_plan.by_matmul.values():
        _acc, lhs, rhs = matmul_operands(group.matmul)
        for node, operand, blocks in (
            (lhs, group.lhs, (group.m_block, group.k_block)),
            (rhs, group.rhs, (group.k_block, group.n_block)),
        ):
            if operand.view is None:
                staged.update(blocks)
                continue
            owned.add(node)
            if node.target is torch.ops.aten.permute.default:
                source = node.args[0]
                assert isinstance(source, Node)
                owned.add(source)
        owned.add(group.matmul)
        owned.update(n for n in (group.seed, group.upd) if n is not None)
        owned.update(group.cur, group.derived, group.stores, group.update)
        for a, b in group.plumbing:
            owned.update((a, b))
        for level in group.levels:
            owned.update((level.arg, level.placeholder, level.exit, level.phi))
        # A ``_mpp_coop`` seed is staged from the Scalar layout, by thread.
        if group.seed is not None and group.seed.target is _mpp_coop:
            owned.discard(group.seed)
    for block_id in laned:
        why = (
            "a staged MPP operand"
            if block_id in staged
            else _scalar_reader(block_id, graphs, owned)
        )
        if why is not None:
            threads, lanes = plan[block_id][1:]
            raise _unsupported(
                f"block {block_id} on {threads} threads x {lanes} lanes, but "
                f"{why} reads it in Scalar code, which has no lane loop over a "
                "free block; give it num_threads equal to its size, or more "
                "num_warps"
            )


def _scalar_reader(
    block_id: int, graphs: list[GraphInfo], owned: set[Node]
) -> str | None:
    """A node outside *owned* whose value holds *block_id* and that Scalar
    code evaluates (named), or None.  Iotas, loads and pure element-wise
    nodes count only through their users: one used only by MPP-owned nodes,
    shape reads or loop plumbing is never evaluated."""
    from .mpp_matmul import Graphs
    from .mpp_matmul import _carry
    from .mpp_matmul import _pointwise

    env = CompileEnvironment.current()
    g = Graphs(graphs)

    def holds(node: Node) -> bool:
        val = node.meta.get("val")
        return isinstance(val, torch.Tensor) and any(
            env.resolve_block_id(s) == block_id for s in val.shape
        )

    ordered = [n for n in _nodes(live_graphs(graphs)) if holds(n) and n not in owned]
    cands = set(ordered)
    for n in ordered:
        if n.op == "placeholder":
            owner = g.owner(n)
            if owner is None or (
                _tracing_ops.is_for_loop_target(owner.target)
                and _carry(n, owner) is not None
            ):
                return n.name
        elif n.target is operator.getitem:
            if any(u.target is not torch.ops.aten.sym_size.int for u in n.users):
                return n.name
            continue
        elif not (
            n.target in _IOTA_TARGETS
            or n.target is load
            or n.target is _tracing_ops._new_var
            or _pointwise(n)
        ):
            return n.name
        for u in n.users:
            if u.target is torch.ops.aten.sym_size.int or u in owned or u in cands:
                continue
            if _tracing_ops.is_for_loop_target(u.target) or u.op == "output":
                continue
            if u.target is _tracing_ops._if and u.args[0] is not n:
                continue
            return u.name
    return None


def _codegen_iota(
    ctx: LoweringContext, node: Node, start: object, step: object
) -> ast.AST | FactoredValue:
    from ..generate_ast import GenerateAST
    from .backend import MetalBackend
    from .factored_dims import factored_iota

    env = CompileEnvironment.current()
    backend = env.backend
    assert isinstance(backend, MetalBackend)
    length = node.meta["val"].shape[0]
    block_id = env.get_block_id(length)
    planned = None if block_id is None else backend.free_arange_plan.get(block_id)
    if planned is None:
        if (factored := factored_iota(ctx, node)) is not None:
            return factored
        raise exc.BackendUnsupported(
            "metal",
            f"hl.arange() of length {length}: Metal lowers only a range over a "
            "block size that no loop owns (hl.register_block_size), or a tile "
            "widened by a constant c (start c * tile.begin, length c * block_size)",
        )
    assert block_id is not None
    axis, threads, lanes = planned
    assert isinstance(ctx.cg, GenerateAST)
    dims = ctx.cg.max_thread_block_dims
    dims[axis] = max(dims[axis], threads)
    dtype = node.kwargs.get("dtype")
    dtype_str = backend.dtype_str(
        dtype if isinstance(dtype, torch.dtype) else env.index_dtype
    )
    # Not ``backend.thread_index``: that records the axes loops read, which must
    # exclude this one.  ``tid`` and a grid index are unsigned, so both the
    # thread index and the sum are cast, keeping a negative start or step
    # signed.
    expr = backend.cast_expr(f"tid[{axis}]", dtype_str)
    if lanes > 1:
        expr = backend.cast_expr(f"{expr} * {lanes} + {lane_name(block_id)}", dtype_str)
    if step != 1:
        expr = f"{{step}} * {expr}"
    if start != 0:
        expr = f"{{start}} + {expr}"
    if step != 1 or start != 0:
        expr = backend.cast_expr(expr, dtype_str)
    return expr_from_string(expr, start=ctx.to_ast(start), step=ctx.to_ast(step))


@iota_lowering.register_codegen("metal")
def codegen_iota_metal(ctx: LoweringContext, node: Node) -> object:
    return _codegen_iota(
        ctx, node, node.kwargs.get("start", 0), node.kwargs.get("step", 1)
    )


@arange_default_lowering.register_codegen("metal")
def codegen_arange_default_metal(ctx: LoweringContext, node: Node) -> object:
    return _codegen_iota(ctx, node, 0, 1)
