"""Detect repeated block axes that require complete coordinate ownership.

The scalar backend gives each block id one lane coordinate. Unexempted
repeated axes therefore require a complete fragment owner; this guard rejects
them if no such owner accepts the root. Explicit view coordinates, broadcasts
and the separately proved scalar/matmul paths retain their existing behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language._tracing_ops import _host_tensor
from ...language.memory_ops import load
from ...language.memory_ops import store
from ..compile_environment import CompileEnvironment
from ..matmul_utils import matmul_operand_indices
from .completed_matmul_sum import match_matmul_sum_path
from .cute_reshape import CUTE_DIM_LOCAL_COORD_META
from .matmul_utils import _CUTE_RHS_VIEW_PASSTHROUGH_TARGETS

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
    """Refuse repeated scalar coordinates when no complete owner handled them."""
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
