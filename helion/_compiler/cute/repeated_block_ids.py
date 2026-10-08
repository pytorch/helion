"""Reject device tensors that bind one block id to two axes (CuTe SIMT lowering).

The CuTe scalar backend gives every block id exactly one per-thread lane
coordinate.  A tensor whose shape carries the same block id on two axes - for
example the ``[C, C]`` result of ``hl.dot(q[b, :, d], k[b, :, d].T)`` where both
full slices dedup onto one reduction dim, or ``idx[:, None] >= idx[None, :]``
over one ``hl.arange(C)`` - therefore collapses onto its diagonal and silently
produces wrong numbers.  Until the lowering can hand a second lane coordinate
to the repeated block, refuse such tensors loudly at codegen time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language._tracing_ops import _host_tensor
from ...language.memory_ops import load
from ...language.memory_ops import store
from ..compile_environment import CompileEnvironment
from .completed_matmul_sum import match_matmul_sum_path
from .cute_reshape import CUTE_DIM_LOCAL_COORD_META
from .matmul_utils import _CUTE_RHS_VIEW_PASSTHROUGH_TARGETS
from .matmul_utils import _cute_matmul_operand_indices

if TYPE_CHECKING:
    from torch.fx.node import Node

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


def check_repeated_block_ids(cg: CodegenInterface, node: Node) -> None:
    """Raise ``BackendUnsupported`` when *node*'s value repeats a block id.

    Exempt, because they never index with the repeated lane coordinate: host
    tensors (whole arrays whose static dims only coincide with a block size by
    value), stack-tensor accesses and their device-pointer table loads (which
    lower through ``_cute_stack_tensor_offset_expr``), plain loads consumed only
    as matmul lhs/rhs operands (re-read by explicit coordinates on the
    direct-load serial-K path), and the proven static-M==N baddbmm carry whose N
    axis is folded away by ``_emit_cute_matmul_n_collapse`` /
    ``completed_matmul_sum``.  Every other repeated id is a silent diagonal
    collapse.
    """
    # Host tensors are whole arrays, not per-thread tiles; their static dims
    # only coincide with a block size by value.
    if node.target is _host_tensor:
        return
    # Stack tensors address their device-pointer table and each stacked tile
    # through their own offset lowering (``_cute_stack_tensor_offset_expr``),
    # not through one lane coordinate per block id.
    if _is_stack_tensor_access(node):
        return
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        return
    env = CompileEnvironment.current()
    block_id = repeated_block_id(env, value, node.meta.get(CUTE_DIM_LOCAL_COORD_META))
    if block_id is None or _is_matmul_operand_load(node):
        return
    if node in _matmul_sum_collapse_nodes(cg):
        return
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

    Such an operand (``y[:, :]`` of a square ``y`` with K == N) is addressed
    by explicit coordinates on the direct-load serial-K path, and the scalar
    fallback's K-block resolver already refuses a contraction block shared
    with M or N; a repeated id in the matmul *result* is still caught here.
    The ``acc`` / scale operands have no such re-read, so only the lhs/rhs
    argument slots of ``_cute_matmul_operand_indices`` qualify.
    """
    if (
        node.target is not load
        and node.target not in _CUTE_RHS_VIEW_PASSTHROUGH_TARGETS
    ):
        return False
    operand_indices = _cute_matmul_operand_indices()
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


def _matmul_sum_collapse_nodes(cg: CodegenInterface) -> set[Node]:
    """Every node of a proven zero-initialized baddbmm carry summed over N."""
    from ..generate_ast import GenerateAST

    if not isinstance(cg, GenerateAST):
        return set()
    env = CompileEnvironment.current()
    nodes: set[Node] = set()
    for graph_info in cg.codegen_graphs:
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
                cg, producer, n_block_id, env.size_hint(value.shape[-1])
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
