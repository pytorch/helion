"""Lower scalar-indexed loads from TPU VMEM-resident tensor references.

TPU VMEM is physically tiled in the two minor dimensions, and Mosaic cannot
project a runtime scalar index out of either one directly. For 32-bit dtypes
a dynamic index on the second-minor dimension is legal in the ref subscript
itself (``ref[i, :]``); the lane dimension is then selected with a static
index after the load.

``classify_vmem_scalar_load`` decides whether a load needs this lowering;
``emit_vmem_scalar_load`` generates it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from helion._compiler.ast_extension import expr_from_string

if TYPE_CHECKING:
    import ast

    from helion._compiler.inductor_lowering import CodegenState

# TODO(tcombes): packed dtypes (bf16/i16/i8/u8/fp8/bool) and dynamic lane
# indices need a pad + pltpu.roll + widen lowering; Mosaic cannot prove
# sublane alignment for a dynamic ref index on packed layouts.


@dataclass(frozen=True)
class VmemScalarLoad:
    """A runtime scalar index on the second-minor dim of a VMEM tensor."""

    lane_index: int | None  # static lane index, or None when the lane is sliced


def _is_scalar_index_pattern(pattern: object) -> bool:
    from helion._compiler.pallas.plan_tiling import ArbitraryIndexPattern
    from helion._compiler.pallas.plan_tiling import TileBeginWithOffsetPattern

    return isinstance(pattern, (ArbitraryIndexPattern, TileBeginWithOffsetPattern))


def _resident_extent(state: CodegenState, tensor: torch.Tensor, dim: int) -> int:
    """Return the extent of one dimension in the kernel's resident VMEM window."""
    from helion._compiler.compile_environment import CompileEnvironment

    dim_size = tensor.shape[dim]
    if not isinstance(dim_size, int):
        raise NotImplementedError(
            "Pallas VMEM scalar load requires a static resident extent"
        )

    tiling = state.device_function.pallas_tensor_dim_tilings[id(tensor)][dim]
    if tiling.can_tile and len(tiling.block_ids) == 1:
        block_size = (
            CompileEnvironment.current()
            .block_sizes[tiling.block_ids[0]]
            .from_config(state.config)
        )
        if not isinstance(block_size, int):
            raise NotImplementedError(
                "Pallas VMEM scalar load requires a static block size"
            )
        return min(dim_size, block_size)
    return dim_size


def _static_index(index: str, extent: int) -> int | None:
    """Normalize an integer literal, or return ``None`` for a runtime index."""
    try:
        return int(index) % extent
    except ValueError:
        return None


def classify_vmem_scalar_load(
    state: CodegenState,
    tensor: torch.Tensor,
    index_parts: list[str],
    indexing_patterns: list[object],
) -> VmemScalarLoad | None:
    """Decide whether a load needs the VMEM scalar-load lowering.

    ``None`` means an ordinary load: no runtime scalar index on the two minor
    dims, or a form Mosaic already lowers (static 32-bit extracts).
    """
    from helion._compiler.device_function import PallasMemorySpace
    from helion._compiler.pallas.plan_tiling import NonePattern

    if (
        tensor.ndim < 2
        or tensor.dtype.itemsize != 4
        or tensor.dtype == torch.bool
        or state.device_function.pallas_memory_space.get(id(tensor))
        != PallasMemorySpace.VMEM
    ):
        return None
    patterns = tuple(p for p in indexing_patterns if not isinstance(p, NonePattern))
    if len(patterns) != tensor.ndim:
        return None

    sublane, lane = tensor.ndim - 2, tensor.ndim - 1
    if not _is_scalar_index_pattern(patterns[sublane]):
        return None
    sublane_extent = _resident_extent(state, tensor, sublane)
    if _static_index(index_parts[sublane], sublane_extent) is not None:
        return None

    lane_index = None
    if _is_scalar_index_pattern(patterns[lane]):
        lane_index = _static_index(
            index_parts[lane], _resident_extent(state, tensor, lane)
        )
        if lane_index is None:
            return None

    return VmemScalarLoad(lane_index=lane_index)


# TODO(tcombes): Mosaic bug: x_ref[i, 255] with runtime i fails to compile
# when cols > 128 ("dynamic load with unaligned indices"). Fix there, then
# delete the sublane-load rewrite.
def emit_vmem_scalar_load(
    tensor: torch.Tensor,
    ref_name: str,
    index_parts: list[str],
    load: VmemScalarLoad,
) -> ast.AST:
    """Emit the ref load with the runtime sublane index in the subscript."""
    lane = tensor.ndim - 1
    parts = [*index_parts]
    if load.lane_index is not None:
        parts[lane] = ":"
    value = f"{ref_name}[{', '.join(parts)}]"
    if load.lane_index is not None:
        value = f"{value}[..., {load.lane_index}]"
    return expr_from_string(value)
