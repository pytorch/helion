"""Recognize the two unmodified BF16 operands of a complete FP32 row dot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import view_ops
from .fragment_vector_io import PACKET_BYTES
from .fragment_vector_io import static_int
from .fragment_vector_io import static_strides
from .warp_carry_packet import uses_warp_carry

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...autotuner.config_spec import ConfigSpec
    from ..compile_environment import CompileEnvironment
    from .row_reduction_carry import RowCarry
    from .row_reduction_carry import RowLoop


ROW_DOT_WARP = 32
BF16_PACKET_LANES = PACKET_BYTES // 2
# Four cooperating lanes provide one K group; a whole K tile spans a warp
# of aligned BF16 packets.
ROW_DOT_K_GROUP = 4 * BF16_PACKET_LANES
ROW_DOT_K = ROW_DOT_WARP * BF16_PACKET_LANES
ROW_DOT_ROWS = 16


def _load_before_widen(node: object, rank: int) -> Node | None:
    if not (
        isinstance(node, Node)
        and node.target is torch.ops.prims.convert_element_type.default
        and not node.kwargs
        and len(node.args) == 2
        and node.args[1] is torch.float32
        and isinstance(load := node.args[0], Node)
        and load.target is memory_ops.load
        and not load.kwargs
        and len(load.args) == 4
        and load.args[2:] == (None, None)
        and isinstance(base := load.args[0], Node)
        and base.target is _tracing_ops._host_tensor
        and isinstance(value := load.meta.get("val"), torch.Tensor)
        and value.dtype is torch.bfloat16
        and value.ndim == rank
    ):
        return None
    return load


def row_dot_loads(plan: RowLoop) -> tuple[Node, Node] | None:
    """Return matrix and vector loads, without commuting casts or arithmetic.

    The existing RowLoop proof owns the zero FP32 carry, its only consumer
    (a trailing sum), and its full K bounds. A computed activation or
    multiply in BF16 is not the same operation as these two FP32 widens.
    """
    if len(plan.results) != 1 or plan.begin != 0 or plan.end % ROW_DOT_K:
        return None
    term = plan.results[0].term
    if term.target is _tracing_ops._mask_select:
        if (
            term.kwargs
            or len(term.args) != 3
            or term.args[1] != 0
            or term.args[2] not in ((plan.block_id,), [plan.block_id])
        ):
            return None
        product = term.args[0]
    elif plan.unmasked_rows is not None:
        product = term
    else:
        return None
    if not (
        isinstance(product, Node)
        and product.target is torch.ops.aten.mul.Tensor
        and not product.kwargs
        and len(product.args) == 2
    ):
        return None
    for mat, bcast in (product.args, product.args[::-1]):
        matrix = _load_before_widen(mat, 2)
        if (
            matrix is not None
            and isinstance(bcast, Node)
            and bcast.target is view_ops.subscript
            and not bcast.kwargs
            and len(bcast.args) == 2
            and bcast.args[1] in ([None, slice(None)], (None, slice(None)))
            and (vector := _load_before_widen(bcast.args[0], 1)) is not None
        ):
            return matrix, vector
    return None


def row_dot_structure(results: tuple[RowCarry, ...]) -> bool:
    """A matrix/vector product can have an MMA slot irrespective of its dtype."""
    if len(results) != 1:
        return False

    from .row_reduction_carry import row_value_before_casts

    product = row_value_before_casts(results[0].term)
    if isinstance(product, Node) and product.target is _tracing_ops._mask_select:
        product = product.args[0]
    product = row_value_before_casts(product)
    return (
        isinstance(product, Node)
        and product.target is torch.ops.aten.mul.Tensor
        and any(
            isinstance(operand, Node)
            and operand.target is view_ops.subscript
            and operand.args[1] in ([None, slice(None)], (None, slice(None)))
            for operand in map(row_value_before_casts, product.args)
        )
    )


def supported_row_dot(spec: ConfigSpec) -> bool:
    capability = spec.target_device_capability
    return capability is not None and capability[0] >= 8


def admitted_row_dot_loads(
    env: CompileEnvironment, plan: RowLoop
) -> tuple[Node, Node] | None:
    if (
        # MMA does not implement IEEE round-to-nearest after each FP32 sum.
        env.settings.fast_math
        and supported_row_dot(env.config_spec)
        and (loads := row_dot_loads(plan)) is not None
        and row_dot_windows(env, plan, loads)
    ):
        return loads
    return None


def row_dot_geometry(spec: ConfigSpec, config: Mapping[str, object]) -> bool:
    """At least one admitted root has simultaneous complete row and K tiles."""
    blocks = config.get("block_sizes")
    if not (
        supported_row_dot(spec)
        and uses_warp_carry(spec, config)
        and config.get("cute_fragment_vector_io") is True
        and isinstance(blocks, list)
    ):
        return False
    return any(
        spec.block_sizes.config_get(blocks, k) == ROW_DOT_K
        and isinstance(rows := spec.block_sizes.config_get(blocks, row), int)
        and rows >= ROW_DOT_ROWS
        and rows % ROW_DOT_ROWS == 0
        and all(
            recorded_axis != row or extent % rows == 0
            for recorded_axis, extent in spec.cute_fragment_row_carry_unmasked_rows
        )
        for axes in spec.cute_warp_row_dot_roots.values()
        for row, k in axes
    )


def row_dot_windows(
    env: CompileEnvironment, plan: RowLoop, loads: tuple[Node, Node]
) -> bool:
    """Each host window has a known complete K extent and uses the same K tile."""
    for load in loads:
        source, indices = load.args[:2]
        if not isinstance(source, Node) or not isinstance(indices, (list, tuple)):
            return False
        fake = source.meta.get("val")
        if (
            not isinstance(fake, torch.Tensor)
            or not fake.ndim
            or fake.dtype is not torch.bfloat16
            or static_int(fake.shape[-1]) != plan.end
            or (strides := static_strides(env, fake)) is None
            or strides[-1] != 1
            or any(stride % BF16_PACKET_LANES for stride in strides[:-1])
            or any(index is None for index in indices)
            or len(indices) != fake.ndim
            or not indices
            or not isinstance(k := indices[-1], Node)
            or k.target not in (_tracing_ops._get_symnode, torch.ops.aten.sym_size.int)
            or not isinstance(extent := k.meta.get("val"), torch.SymInt)
            or env.resolve_block_id(extent) != plan.block_id
        ):
            return False
    return True
