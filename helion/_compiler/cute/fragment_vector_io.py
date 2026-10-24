"""Static row packets for the fragment lowering's host loads and stores.

``cute_fragment_vector_io``: a host tensor window whose innermost logical
coordinate walks a unit-stride tensor dimension in whole, aligned groups is
moved as 16-byte packets (four fp32, eight 16-bit elements) instead of one
scalar per element.  Unlike ``cute_fragment_packet_loads`` (which evaluates
the scalar address recipe for every lane and compares the effective
addresses at run time), the packet here is proved at compile time:

* the tensor's strides are static, the innermost indexed dimension has
  stride 1 and every other stride is a multiple of the packet (so a packet
  starting at an aligned base stays aligned),
* the base pointer is 16-byte aligned (an input's specialized residue or a
  fresh wrapper allocation with a static storage offset),
* the innermost index is a full slice, a tile whose block size is a packet
  multiple, or an ``hl.arange`` window starting at a packet multiple, over a
  static extent that is a packet multiple,

so one address and one predicate per packet are exact for all of its lanes.
Everything else keeps the scalar element walk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch
from torch.fx import Node

from .memory_ops import tensor_has_specialized_base_alignment
from .packet_loads import PACKET_DTYPES

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from .computed_fragment import FragmentCompiler
    from .computed_fragment import HostTensor

KEY = "cute_fragment_vector_io"
PACKET_BYTES = 16


def packet_lanes(dtype: torch.dtype) -> int | None:
    if dtype not in PACKET_DTYPES:
        return None
    return PACKET_BYTES // dtype.itemsize


def host_base_aligned(env: CompileEnvironment, tensor: torch.Tensor) -> bool:
    """Whether every element offset that is a 16-byte multiple is aligned."""
    offset = tensor.storage_offset()
    if not isinstance(offset, int) or (offset * tensor.dtype.itemsize) % PACKET_BYTES:
        return False
    return tensor_has_specialized_base_alignment(
        env, tensor, PACKET_BYTES
    ) or env.tensor_storage_is_compiler_allocated(tensor)


def _static_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, torch.SymInt):
        node = value.node
        if node.has_hint() and node.expr.is_number:
            return int(sympy.Integer(node.expr))
    return None


def static_strides(tensor: torch.Tensor) -> tuple[int, ...] | None:
    strides = [_static_int(s) for s in tensor.stride()]
    if any(s is None for s in strides):
        return None
    return tuple(s for s in strides if s is not None)


def row_packet_lanes(
    compiler: FragmentCompiler,
    tensor: HostTensor,
    indices: list[object],
    values: dict[Node, object],
    shape: tuple[int, ...],
    dtype: torch.dtype,
    index_coordinates: list[tuple[int | None, ...]],
) -> int | None:
    """The packet width (in elements) when the window's innermost coordinate
    may be moved as whole aligned packets; None keeps the scalar walk."""
    from .computed_fragment import Fragment

    lanes = packet_lanes(dtype)
    if lanes is None or not shape or shape[-1] % lanes or shape[-1] < lanes:
        return None
    fake = tensor.value
    if fake.dtype != dtype or not host_base_aligned(compiler.env, fake):
        return None
    strides = static_strides(fake)
    if strides is None:
        return None
    ordinals = [ordinal for ordinal, index in enumerate(indices) if index is not None]
    if not ordinals or len(ordinals) != fake.ndim:
        return None
    last = ordinals[-1]
    inner_dim = fake.ndim - 1
    if strides[inner_dim] != 1 or any(
        stride % lanes for dim, stride in enumerate(strides) if dim != inner_dim
    ):
        return None
    # The innermost result coordinate must be the one the last index consumes,
    # and no other index may read it.
    inner_axis = len(shape) - 1
    if tuple(index_coordinates[last]) != (inner_axis,) or any(
        inner_axis in index_coordinates[ordinal]
        for ordinal in ordinals
        if ordinal != last
    ):
        return None
    size = compiler.static_extent(fake.shape[inner_dim])
    if size is None or size % lanes:
        return None
    index = indices[last]
    if isinstance(index, slice):
        return lanes if index == slice(None) and size == shape[-1] else None
    if isinstance(index, Node):
        value = values.get(index)
        proxy = index.meta.get("val")
        if isinstance(value, Fragment):
            affine = value.affine
            if (
                affine is None
                or value.shape != (shape[-1],)
                or value.logical_domain is not None
                or value.snapshot is not None
            ):
                return None
            start, step = affine
            if step != 1 or start % lanes or start < 0 or start + shape[-1] > size:
                return None
            return lanes
        if isinstance(proxy, torch.SymInt):
            block = compiler.env.resolve_block_id(proxy)
            if (
                block is None
                or block not in compiler.offsets
                or proxy._sympy_() != compiler.env.block_sizes[block].var._sympy_()
            ):
                return None
            block_size = compiler.df.resolved_block_size(block)
            if not isinstance(block_size, int) or block_size % lanes:
                return None
            return lanes if block_size == shape[-1] else None
    return None


def copy_atom(dtype: str) -> str:
    return (
        "cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), "
        f"{dtype}, num_bits_per_copy={PACKET_BYTES * 8})"
    )
