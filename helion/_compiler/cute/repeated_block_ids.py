"""Detect repeated block axes that require complete coordinate ownership.

The scalar backend gives each block id one lane coordinate. Unexempted
repeated axes therefore require a complete fragment owner; this guard rejects
them if no such owner accepts the root. Explicit view coordinates, broadcasts
and the separately proved scalar/matmul paths retain their existing behavior.

A store whose subscript indexes two axes by one block is refused here too
(``_check_store_target``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language._tracing_ops import _host_tensor
from ...language.memory_ops import load
from ...language.memory_ops import store
from ..compile_environment import CompileEnvironment
from ..indexing_strategy import is_block_size_index
from ..matmul_utils import matmul_operand_indices
from .completed_matmul_sum import match_matmul_sum_path
from .cute_reshape import CUTE_DIM_LOCAL_COORD_META
from .matmul_utils import _CUTE_RHS_VIEW_PASSTHROUGH_TARGETS
from .memory_ops import _cute_expand_broadcast_dim

if TYPE_CHECKING:
    from torch.fx.node import Node

    from ..device_ir import GraphInfo
    from ..helper_function import CodegenInterface


def repeated_block_id(
    env: CompileEnvironment,
    value: torch.Tensor,
    coord_meta: list[object | None] | None = None,
) -> int | None:
    """Return a canonical block id bound to two non-broadcast axes of *value*.

    ``coord_meta`` is the per-dimension ``CUTE_DIM_LOCAL_COORD_META`` list: a
    dimension carrying explicit view coordinates (a reshape-split or permuted
    pair dim) is addressed through them rather than the block's lane, so it
    cannot collapse onto another axis that merely shares its size.
    """
    seen: set[int] = set()
    view_keys: set[tuple[object, ...]] = set()
    view_blocks: set[int] = set()
    for dim, size in enumerate(value.shape):
        # A stride-0 axis replicates one element, so it addresses no second
        # lane coordinate (see the broadcast store in memory_ops).
        if value.stride(dim) == 0 or env.size_hint(size) <= 1:
            continue
        info = coord_meta[dim] if coord_meta is not None else None
        if isinstance(info, dict) and isinstance(info.get("block_id"), int):
            # A reshape-split dim is a sub-coordinate (``div``/``mod``) of its
            # source block's lane.  Its complementary sibling from the same
            # split is fine, but a second identical sub-coordinate or a plain
            # dim bound to that source block is still one lane on two axes.
            canonical = env.canonical_block_id(info["block_id"])
            key = (canonical, str(info.get("divisor", 1)), str(info.get("modulus")))
            if key in view_keys or canonical in seen:
                return canonical
            view_keys.add(key)
            view_blocks.add(canonical)
            continue
        block_id = env.resolve_block_id(size)
        if block_id is None:
            continue
        canonical = env.canonical_block_id(block_id)
        if canonical in seen or canonical in view_blocks:
            return canonical
        seen.add(canonical)
    return None


def scalar_repeated_block_id(
    env: CompileEnvironment,
    node: Node,
) -> int | None:
    """Return a scalar-coordinate conflict after the existing exemptions.

    Host arrays, stack accesses, matmul operand loads, explicit view coordinates
    and broadcasts do not require a second scalar lane. Callers separately
    exempt baddbmm carries proved to be consumed only by their N sum.
    """
    # Host tensors are whole arrays, not per-thread tiles; their static dims
    # only coincide with a block size by value.
    if node.target is _host_tensor:
        return None
    # Stack tensors address their device-pointer table and each stacked tile
    # through their own offset lowering (``_cute_stack_tensor_offset_expr``),
    # not through one lane coordinate per block id.
    if _is_stack_tensor_access(node):
        return None
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        return None
    block_id = repeated_block_id(env, value, node.meta.get(CUTE_DIM_LOCAL_COORD_META))
    if block_id is None or _is_matmul_operand_load(node):
        return None
    return block_id


def check_repeated_block_ids(cg: CodegenInterface, node: Node) -> None:
    """Refuse repeated scalar coordinates when no complete owner handled them,
    and a store whose subscript repeats a block (``_check_store_target``)."""
    if node.target is store:
        # Stack tensors address each stacked tile through their own offset
        # lowering, not through one lane coordinate per block id.
        if not _is_stack_tensor_access(node):
            _check_store_target(cg, node)
        return
    env = CompileEnvironment.current()
    # Keep the graph proof off the common non-repeated-node path.
    block_id = scalar_repeated_block_id(env, node)
    if block_id is None:
        return
    from ..generate_ast import GenerateAST

    if isinstance(cg, GenerateAST) and node in matmul_sum_collapse_nodes(
        env, cg.codegen_graphs
    ):
        return
    value = node.meta["val"]
    raise exc.BackendUnsupported(
        "cute",
        f"tile shape {list(value.shape)} binds block id {block_id} to two axes "
        f"of one tensor ({node.target}); the SIMT lowering maps each block id "
        "to one lane",
    )


def _check_store_target(cg: CodegenInterface, node: Node) -> None:
    """Raise ``BackendUnsupported`` when a store's subscript indexes two axes
    of one tensor by one block and its value does not span both: ``out[c, r,
    c] = y[None, :, :]`` broadcasts the value along both of ``c``'s axes, and
    the per-lane store, one lane coordinate for both, writes only their
    diagonal where eager writes the whole tile.  An axis takes the block of
    its tile, or of the dims a tensor index contributes as Helion's indexing
    shapes them (``out[c, idx[c]]``, ``out[c, c.index + 1]``; tensor indices
    broadcast together give their dims once).  Not counted: full slices,
    whose sharing of a reduction dim is beyond this check, and a block of one
    element in the config being compiled, whose diagonal is the whole tile.
    Exempt: a value whose own shape repeats the block (checked on its own
    node), and the stores ``_codegen_cute_store_expand_broadcast_tile``
    lowers, whose ``expand`` value is broadcast along a tile of the block and
    whose other axis is a tensor index: a loop walks the broadcast axis."""
    index_arg = node.args[1]
    if not isinstance(index_arg, (list, tuple)):
        return
    env = CompileEnvironment.current()
    indices = [
        index.meta.get("val") if isinstance(index, torch.fx.Node) else index
        for index in index_arg
    ]
    tensors = [index for index in indices if isinstance(index, torch.Tensor)]
    together = env.should_broadcast_tensor_indexers(indices)
    axes: dict[int, list[int]] = {}
    for position, index in enumerate(indices):
        if isinstance(index, torch.SymInt) and is_block_size_index(index):
            sizes: list[int | torch.SymInt] = [index]
        elif isinstance(index, torch.Tensor) and not together:
            sizes = env.tensor_indexer_dims(index)
        elif isinstance(index, torch.Tensor) and tensors:
            # Broadcast together, the tensor indices give their shape once.
            sizes, tensors = env.tensor_indexer_broadcast_shape(tensors), []
        else:
            continue
        for size in sizes:
            resolved = (
                env.resolve_block_id(size) if isinstance(size, torch.SymInt) else None
            )
            if resolved is None:
                continue
            canonical = env.canonical_block_id(resolved)
            extent = cg.device_function.resolved_block_size(canonical)
            if not (isinstance(extent, int) and extent <= 1):
                axes.setdefault(canonical, []).append(position)
    value = node.args[2]
    if not isinstance(value, torch.fx.Node) or not isinstance(
        value.meta.get("val"), torch.Tensor
    ):
        value = None
    broadcast = _cute_expand_broadcast_dim(value) if value is not None else None
    for block_id, positions in axes.items():
        if len(positions) < 2:
            continue
        if (
            value is not None
            and repeated_block_id(
                env, value.meta["val"], value.meta.get(CUTE_DIM_LOCAL_COORD_META)
            )
            == block_id
        ):
            continue
        others = [position for position in positions if position != broadcast]
        if (
            broadcast is not None
            and len(others) == len(positions) - 1 == 1
            and isinstance(indices[broadcast], torch.SymInt)
            and isinstance(indices[others[0]], torch.Tensor)
        ):
            continue
        raise exc.BackendUnsupported(
            "cute",
            f"a store subscript indexes two axes of one tensor by block id "
            f"{block_id}; the SIMT lowering maps each block id to one lane",
        )


def _is_stack_tensor_access(node: Node) -> bool:
    """A load/store of a stack tensor, or the pointer-table load that feeds one.

    A stack tensor is ``(tensor_like, dev_ptrs)``: its loads/stores carry that
    tuple as the tensor argument, and the device-pointer table they read is
    itself a plain load consumed only by them.
    """
    if _is_stack_tensor_memory_op(node):
        return True
    return (
        node.target is load
        and bool(node.users)
        and all(_is_stack_tensor_memory_op(user) for user in node.users)
    )


def _is_stack_tensor_memory_op(node: Node) -> bool:
    return (
        node.target in (load, store)
        and bool(node.args)
        and isinstance(node.args[0], tuple)
    )


def _is_matmul_operand_load(node: Node) -> bool:
    """A plain ``load`` (through no-op views) consumed only as matmul lhs/rhs.

    Such an operand (``y[:, :]`` of a square ``y`` with K == N) puts K on a
    free axis's block, which ``cute_resolve_active_matmul_k_block_id``
    refuses, so no lowering reduces it across threads.  The matmul either
    takes a serial-K path that re-reads the operand from global memory by
    explicit coordinates (``emit_cute_serial_mm_from_load_views`` for mm, bmm
    and hl.dot; ``emit_cute_serial_scalar_mm_from_loads`` for a 2-D mm), which
    refuses an operand whose tensor the kernel may write
    (``_tensor_written_anywhere``), or it is refused; the diagonal value the
    load node itself would produce is never consumed.  A repeated id in the
    matmul *result* is still caught here.  The ``acc`` / scale operands have
    no such re-read, so only the lhs/rhs argument slots of
    ``matmul_operand_indices`` qualify.
    """
    if (
        node.target is not load
        and node.target not in _CUTE_RHS_VIEW_PASSTHROUGH_TARGETS
    ):
        return False
    operand_indices = matmul_operand_indices()
    stack = [node]
    while stack:
        current = stack.pop()
        if not current.users:
            return False
        for user in current.users:
            slots = operand_indices.get(user.target)
            if slots is not None:
                if any(
                    slot < len(user.args) and user.args[slot] is current
                    for slot in slots
                ):
                    continue
                return False
            if user.target in _CUTE_RHS_VIEW_PASSTHROUGH_TARGETS:
                stack.append(user)
                continue
            return False
    return True


def matmul_sum_collapse_nodes(
    env: CompileEnvironment, graphs: list[GraphInfo]
) -> set[Node]:
    """Every node of a proven zero-initialized baddbmm carry summed over N."""
    nodes: set[Node] = set()
    for graph_info in graphs:
        for producer in graph_info.graph.nodes:
            if (
                producer.op != "call_function"
                or producer.target is not torch.ops.aten.baddbmm.default
            ):
                continue
            value = producer.meta.get("val")
            if not isinstance(value, torch.Tensor) or value.ndim != 3:
                continue
            n_block_id = env.resolve_block_id(value.shape[-1])
            if n_block_id is None:
                continue
            path = match_matmul_sum_path(
                graphs, producer, n_block_id, env.size_hint(value.shape[-1])
            )
            if path is None:
                continue
            nodes.update(
                (
                    path.producer,
                    path.accumulator_copy,
                    path.initial,
                    path.result_item,
                    path.phi,
                )
            )
            if path.masked is not None:
                nodes.add(path.masked)
    return nodes
