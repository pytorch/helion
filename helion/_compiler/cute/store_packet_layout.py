"""Which store packets the layout of a stored tensor leaves.

A vector store needs its address on the packet's boundary: the emitters
store a tensor's elements in packets only where its base is proven aligned
to the packet, and ``affine_vector_io`` only along a unit-stride last
dimension.  Where a tensor's layout rules the packets out, the knobs that
act on them change nothing: ``cute_store_packet_tiers`` splits no packet and
``cute_store_policy`` renames no flush.  This module reads those facts of
the bound call so that config normalization reads the knobs off there
(``search_surface._normalize_cute_lane_knobs``).  They are facts of the
call's tensors, not of the kernel's structure: the search space does not
depend on them.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language._tracing_ops import _for_loop
from ...language.memory_ops import store
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from .flat_gather_lane import base_aligned_for_seed
from .memory_ops import tensor_proven_base_alignment
from .packet_store_policy import POLICY_PACKET_BYTES
from .store_tier_blocks import tile_stores

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ...autotuner.config_spec import ConfigSpec
    from ..device_ir import DeviceIR

# The widest packet ``affine_vector_io`` stores.  Lanes wider than it store
# several (``_Vectorizer.memory_width``), so a tensor that takes it takes
# every vector width: its lane limit is the widest width.
STORE_PACKET_BYTES = 16
ANY_STORE_PACKET_LANES = 16


def _base_alignment(env: CompileEnvironment, tensor: torch.Tensor) -> int:
    """The widest of 16 and 8 bytes that ``tensor``'s base will be proven
    aligned to (``tensor_proven_base_alignment``), 0 for neither.

    The binding has not recorded the inputs' pointer residues when the
    search surface is registered, so an input's base is read as the seeds
    read it (``base_aligned_for_seed``: from the example input, whose
    residue the binding then pins in the bound kernel's cache key)."""
    proven = tensor_proven_base_alignment(
        env, tensor, (STORE_PACKET_BYTES, 8), input_aligned=base_aligned_for_seed
    )
    return 0 if proven is None else proven


def _store_packet_lanes(env: CompileEnvironment, tensor: torch.Tensor) -> int:
    """The most vector lanes ``affine_vector_io`` stores in packets along
    the last dimension of ``tensor`` (0: no packet at any width), by the
    layout facts its proofs read: the dimension's stride is the literal 1
    (lanes at any other stride, or at one the trace does not fix, are not a
    packet's elements), and the tensor's base is aligned to the packet, 16
    bytes or the 8 of four 16-bit lanes (``_base_alignment``)."""
    stride = tensor.stride()[-1]
    if type(stride) is not int or stride != 1:
        return 0
    alignment = _base_alignment(env, tensor)
    if alignment == STORE_PACKET_BYTES:
        return ANY_STORE_PACKET_LANES
    return alignment // tensor.dtype.itemsize


def store_packet_lane_limits(
    device_ir: DeviceIR, config_spec: ConfigSpec
) -> dict[int, int]:
    """For each of the ``store_tier_blocks``, the most vector lanes with
    which one of its loops still stores packets, by the layout of the
    tensors the loop stores along its tile (``_store_packet_lanes``).  A
    loop's lanes are vectorized together, so its narrowest store decides,
    and the block's widest loop stands for the block.
    ``ANY_STORE_PACKET_LANES`` where a loop's stores take the full packet,
    0 where no loop stores a packet at any width: an output view whose last
    dimension is not unit-stride, or whose base is on no 8-byte boundary."""
    env = CompileEnvironment.current()
    loops: dict[tuple[int, int], int] = {}
    for block_id, loop, _symbol, tensor in tile_stores(device_ir, config_spec):
        key = (block_id, loop.graph_id)
        loops[key] = min(
            loops.get(key, ANY_STORE_PACKET_LANES), _store_packet_lanes(env, tensor)
        )
    limits: dict[int, int] = {}
    for (block_id, _graph_id), lanes in loops.items():
        limits[block_id] = max(limits.get(block_id, 0), lanes)
    return limits


def aligned_row_multiples(
    device_ir: DeviceIR, config_spec: ConfigSpec
) -> dict[int, int]:
    """For each of the ``store_tier_blocks``, a number of elements that
    every packet its loops store starts a multiple of from its tensor's
    base, whatever the row: the gcd of the stored tensors' strides ahead of
    their last dimension, where every loop of the block begins at the
    literal 0 (a tile then starts a multiple of its lanes into the row);
    1 where a loop begins elsewhere or the trace does not fix a stride.

    Vector lanes that divide it need no alignment guard: ``affine_vector_io``
    proves every row offset (``prefix_divisible``) and the lane-zero offset,
    stores the whole packet unguarded, and has no guard to hang a tier on."""
    begins = {
        node.args[0]: node.args[1]
        for root_id in device_ir.root_ids
        for node in device_ir.graphs[root_id].graph.find_nodes(
            op="call_function", target=_for_loop
        )
    }
    multiples: dict[int, int] = {}
    for block_id, loop, _symbol, tensor in tile_stores(device_ir, config_spec):
        rows = tensor.stride()[:-1]
        if begins.get(loop.graph_id) != [0] or any(
            type(stride) is not int for stride in rows
        ):
            multiple = 1
        else:
            multiple = math.gcd(*rows)
        multiples[block_id] = math.gcd(multiples.get(block_id, 0), multiple)
    return multiples


def _stores(device_ir: DeviceIR) -> Iterator[tuple[torch.Tensor, list[object]]]:
    """``(stored tensor, index)`` of every store of the kernel into a traced
    tensor."""
    for info in device_ir.graphs:
        for write in info.graph.find_nodes(op="call_function", target=store):
            target = write.args[0]
            if not isinstance(target, torch.fx.Node):
                continue
            tensor = target.meta.get("val")
            if isinstance(tensor, torch.Tensor):
                yield tensor, cast("list[object]", write.args[1])


def _named_stores(
    device_ir: DeviceIR,
) -> Iterator[tuple[str, torch.Tensor, list[object]]]:
    """``(MemoryOpFact.tensor_name, stored tensor, index)`` of every store
    of the kernel into a traced tensor that has a host name."""
    origins = HostFunction.current().tensor_to_origin
    for tensor, index in _stores(device_ir):
        origin = origins.get(tensor)
        name = None if origin is None else origin.root_rw_name()
        if name is not None:
            yield name, tensor, index


def unaligned_store_tensors(device_ir: DeviceIR) -> frozenset[str]:
    """The names (``MemoryOpFact.tensor_name``) of the tensors the kernel
    stores into whose base no proof of the emitters puts on the boundary of
    the narrowest packet ``cute_store_policy`` applies to
    (``POLICY_PACKET_BYTES``, 16 bytes): an output view that starts off it
    in its storage.  The emitters prove a base in two ways, read here as
    the seeds read them: an aligned input and a view's offset from it that
    is a multiple of the packet (``_base_alignment``), or an input's residue
    plus the view's offset (``base_aligned_for_seed``: a view two fp32
    elements into an argument that starts eight bytes off the boundary is
    aligned).  No emitter stores a tensor that neither proves in packets of
    that size or more.  A name that also stands for an aligned tensor (two
    views of one argument) is left out, as is a tensor without a host
    name."""
    env = CompileEnvironment.current()
    aligned: dict[str, bool] = {}
    for name, tensor, _index in _named_stores(device_ir):
        aligned[name] = (
            aligned.get(name, False)
            or _base_alignment(env, tensor) >= POLICY_PACKET_BYTES
            or base_aligned_for_seed(env, tensor, POLICY_PACKET_BYTES)
        )
    return frozenset(name for name, takes in aligned.items() if not takes)


def sliced_store_dims(device_ir: DeviceIR) -> frozenset[tuple[str, int]]:
    """``(MemoryOpFact.tensor_name, position)`` of the store subscripts that
    are slices (``out[tile, :]``, ``out[tile, :half]``) in every store into
    the tensors of that name, the position counted as the fact's
    ``subscript_block_ids`` count it.  A slice's lanes are those of a block
    of its own, never an ``hl.tile`` block's.  The fact alone does not tell
    a slice from a computed index (``out[row, col.index * 1]``), which has
    no block there either but stores the lanes of the tile it reads."""
    sliced: dict[tuple[str, int], bool] = {}
    for name, tensor, index in _named_stores(device_ir):
        # The subscripts a fact counts (``memory_op_facts``).
        subscripts = [
            subscript
            for position, subscript in enumerate(index)
            if not isinstance(subscript, int) and position < tensor.ndim
        ]
        for position, subscript in enumerate(subscripts):
            key = (name, position)
            sliced[key] = sliced.get(key, True) and isinstance(subscript, slice)
    return frozenset(key for key, is_slice in sliced.items() if is_slice)


def new_axis_stores(device_ir: DeviceIR) -> bool:
    """Whether a store of the kernel has a ``None`` subscript (``out[None,
    tile_m, tile_n] = ...``), which takes a position of the index and no dim
    of the tensor.  ``memory_op_facts`` counts a store's subscripts along
    its index, so behind a ``None`` the positions of the fact
    (``subscript_block_ids``, ``subscript_strides``) are a dim off, and the
    row tile reads as the tile of the stride-1 dim: the facts of such a
    kernel do not say whose lanes a flush holds."""
    return any(
        subscript is None
        for _tensor, index in _stores(device_ir)
        for subscript in index
    )
