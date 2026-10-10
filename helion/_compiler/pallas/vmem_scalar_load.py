"""Lower scalar-indexed loads from TPU VMEM-resident tensor references.

TPU VMEM is physically tiled in the two minor dimensions, and Mosaic cannot
project a runtime scalar index out of either one directly. For 32-bit dtypes
a dynamic index on the second-minor dimension is legal in the ref subscript
itself (``ref[i, :]``); the lane dimension is then selected with a static
index after the load. Packed dtypes and dynamic lane indices instead load
the window, pad it to the physical tile, rotate the requested element to
index zero, widen to a 32-bit register type, and extract statically. Runtime
indices on leading dimensions also use rotation, but select whole physical
blocks without widening their dtype.

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
    from collections.abc import Sequence

    from helion._compiler.inductor_lowering import CodegenState


def normalize_negative_scalar_index(index: str, extent: str) -> str:
    """Count a negative tensor index from the logical end, before DMA/rotation."""
    return f"jnp.where(({index}) < 0, ({index}) + ({extent}), {index})"


def scalar_load_bounds(
    state: CodegenState,
    node: torch.fx.Node,
    tensor: torch.Tensor,
    subscript: Sequence[object],
    parts: list[str],
) -> tuple[list[str], str | None]:
    """Clamp masked scalar tensor indices and retain their logical bounds.

    This is shared by ordinary loads and pipeline address reconstruction.
    Integer/symbolic indices retain normal Ref indexing; tensor indices in a
    masked load follow ``hl.load``'s clamped-read, zero-outside-bounds semantics.
    """
    result = node.meta.get("val")
    if (
        not subscript
        or not isinstance(result, torch.Tensor)
        or len(node.args) < 3
        or node.args[2] is None
    ):
        return parts, None
    bounded = list(parts)
    masks = []
    # Rank-expanding None indices consume neither a tensor dim nor a Ref index.
    tensor_indices = [index for index in subscript if index is not None]
    for dim, (index, expression) in enumerate(zip(tensor_indices, parts, strict=True)):
        if isinstance(index, torch.fx.Node):
            index = index.meta.get("val")
        if (
            isinstance(index, torch.Tensor)
            and index.ndim == 0
            and index.dtype in (torch.int32, torch.int64)
        ):
            size = state.device_function.literal_expr(tensor.shape[dim])
            bounded[dim] = f"jnp.clip({expression}, 0, ({size}) - 1)"
            masks.append(f"(({expression}) >= 0) & (({expression}) < ({size}))")
    return bounded, " & ".join(f"({mask})" for mask in masks) if masks else None


@dataclass(frozen=True)
class VmemScalarLoad:
    """Scalar indices selecting elements or blocks of a resident tensor.

    Attributes:
        scalar_dims: Tensor dimensions (non-negative) that are scalar-indexed.
        extents: Size of each scalar-indexed dimension in the resident window.
            Leading dimensions are reduced only by the outer program's grid,
            not by a smaller tile in an inner device loop.
        static_indices: Normalized literal index for each scalar dim, or
            ``None`` when the index is only known at runtime.
        patterns: Per-dim indexing pattern, one entry per tensor dim.
    """

    scalar_dims: list[int]
    extents: dict[int, int]
    static_indices: dict[int, int | None]
    patterns: tuple[object, ...]

    def has_runtime_index(self, dim: int) -> bool:
        """Whether ``dim`` (may be negative) has a runtime-only scalar index."""
        dim %= len(self.patterns)
        return dim in self.static_indices and self.static_indices[dim] is None


def _is_scalar_index_pattern(pattern: object) -> bool:
    from helion._compiler.pallas.plan_tiling import ArbitraryIndexPattern
    from helion._compiler.pallas.plan_tiling import TensorIndexPattern
    from helion._compiler.pallas.plan_tiling import TileBeginWithOffsetPattern

    return isinstance(pattern, (ArbitraryIndexPattern, TileBeginWithOffsetPattern)) or (
        isinstance(pattern, TensorIndexPattern) and pattern.index_ndim == 0
    )


def _is_32bit(dtype: torch.dtype) -> bool:
    return dtype.itemsize == 4


def _resident_extent(state: CodegenState, tensor: torch.Tensor, dim: int) -> int:
    """Return the extent of one dimension in the kernel's resident VMEM window."""
    from helion._compiler.compile_environment import CompileEnvironment

    dim_size = tensor.shape[dim]
    if not isinstance(dim_size, int):
        from .tracing_ops import _resolve_dim_size

        dim_size = _resolve_dim_size(
            dim_size, CompileEnvironment.current(), state.config
        )
    if not isinstance(dim_size, int):
        raise NotImplementedError(
            "Pallas VMEM scalar load requires a static resident extent"
        )

    tensor_tilings = state.device_function.pallas_tensor_dim_tilings.get(id(tensor))
    tiling = tensor_tilings[dim] if tensor_tilings is not None else None
    if tiling is not None and tiling.can_tile and len(tiling.block_ids) == 1:
        block_id = tiling.block_ids[0]
        if dim < tensor.ndim - 2:
            grid = state.device_function.pid
            grid_block_ids = (
                {axis.block_id for axis in grid.pid_info} if grid is not None else set()
            )
            if block_id not in grid_block_ids:
                # Inner-loop indexing does not shrink the outer resident ref.
                # A leading runtime index must still select that full axis.
                return dim_size
        block_size = (
            CompileEnvironment.current().block_sizes[block_id].from_config(state.config)
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

    ``None`` means an ordinary load: no scalar index, or a form Mosaic already
    lowers directly, such as static 32-bit extracts.
    """
    from helion._compiler.device_function import PallasMemorySpace
    from helion._compiler.host_function import HostFunction
    from helion._compiler.pallas.plan_tiling import NonePattern

    memory_space = state.device_function.pallas_memory_space.get(id(tensor))
    if tensor.dtype.itemsize > 4 or memory_space not in (None, PallasMemorySpace.VMEM):
        return None
    if memory_space is None and tensor in HostFunction.current().tensor_to_origin:
        # An unplaced host allocation is not evidence of a resident value.
        return None
    patterns = tuple(p for p in indexing_patterns if not isinstance(p, NonePattern))
    if len(patterns) != tensor.ndim:
        return None
    scalar_dims = [
        d for d in range(tensor.ndim) if _is_scalar_index_pattern(patterns[d])
    ]
    if not scalar_dims:
        return None
    extents = {d: _resident_extent(state, tensor, d) for d in scalar_dims}
    static_indices = {d: _static_index(index_parts[d], extents[d]) for d in scalar_dims}
    if _is_32bit(tensor.dtype) and all(
        index is not None for index in static_indices.values()
    ):
        return None

    return VmemScalarLoad(
        scalar_dims=scalar_dims,
        extents=extents,
        static_indices=static_indices,
        patterns=patterns,
    )


# TODO(tcombes): Mosaic bug: x_ref[i, 255] with runtime i fails to compile
# when cols > 128 ("dynamic load with unaligned indices"). Fix there, then
# delete the sublane-load rewrite.
def _sublane_load_applies(tensor: torch.Tensor, load: VmemScalarLoad) -> bool:
    """32-bit dtypes can put a runtime sublane index in the ref subscript."""
    if not _is_32bit(tensor.dtype):
        return False
    # TODO(tcombes): when both indices are runtime values, compose the sublane
    # load with a lane roll instead of rolling both axes.
    return load.has_runtime_index(-2) and not load.has_runtime_index(-1)


def _sublane_load_expr(
    tensor: torch.Tensor,
    ref_name: str,
    index_parts: list[str],
    load: VmemScalarLoad,
) -> ast.AST:
    """Emit the ref load with the runtime sublane index in the subscript."""
    lane = tensor.ndim - 1
    parts = [*index_parts]
    lane_index = load.static_indices.get(lane)
    if lane_index is not None:
        parts[lane] = ":"
    value = f"{ref_name}[{', '.join(parts)}]"
    if lane_index is not None:
        value = f"{value}[..., {lane_index}]"
    return expr_from_string(value)


def _roll_load_expr(
    tensor: torch.Tensor,
    ref_name: str,
    index_parts: list[str],
    load: VmemScalarLoad,
) -> ast.AST:
    """Emit load, pad, roll to index zero, widen, and static extraction."""
    from helion._compiler.backend import PallasBackend
    from helion._compiler.compile_environment import CompileEnvironment

    window_parts = [*index_parts]
    for dim in load.scalar_dims:
        window_parts[dim] = ":"
    value = f"{ref_name}[{', '.join(window_parts)}]"

    window_dims = [
        d
        for d in range(tensor.ndim)
        if d in load.scalar_dims or not _is_scalar_index_pattern(load.patterns[d])
    ]
    expand_leading = len(window_dims) == 1
    axis_offset = int(expand_leading)
    if expand_leading:
        # Mosaic dynamic_rotate requires a physically tiled, rank >= 2 vector.
        value = f"jnp.expand_dims({value}, axis=0)"

    env = CompileEnvironment.current()
    assert isinstance(env.backend, PallasBackend)
    physical_dtype = torch.int32 if tensor.dtype == torch.bool else tensor.dtype
    sublane_tiling = env.backend.sublane_tiling(physical_dtype)
    # Predicates are physically int32 in Mosaic VMEM. Convert before layout
    # operations instead of treating predicates as packed 8-bit values.
    if tensor.dtype == torch.bool:
        value = f"lax.convert_element_type({value}, jnp.int32)"

    selectors = [":"] * (len(window_dims) + axis_offset)
    pads = [(0, 0)] * len(selectors)
    rolls: list[tuple[int, str]] = []
    if expand_leading:
        selectors[0] = "0"
    for dim in load.scalar_dims:
        axis = window_dims.index(dim) + axis_offset
        static = load.static_indices[dim]
        if static is not None or load.extents[dim] == 1:
            selectors[axis] = str(static if static is not None else 0)
            continue
        alignment = (
            1
            if dim < tensor.ndim - 2
            else 128
            if dim == tensor.ndim - 1
            else sublane_tiling
        )
        padded_extent = load.extents[dim] + (-load.extents[dim]) % alignment
        pads[axis] = (0, padded_extent - load.extents[dim])
        # Other tiled accesses can pad a host argument. A leading-axis roll
        # must use the resident Ref's actual size, not the logical tensor size.
        roll_extent = (
            f"({ref_name}).shape[{dim}]"
            if dim < tensor.ndim - 2
            else str(padded_extent)
        )
        index = index_parts[dim]
        if dim < tensor.ndim - 2:
            # Negative indices refer to logical rows, not host padding. When
            # the Ref is unpadded, modulo already implements that convention.
            index = (
                f"({index} if {roll_extent} == {load.extents[dim]} else "
                f"{normalize_negative_scalar_index(index, str(load.extents[dim]))})"
            )
        rolls.append((axis, f"-({index}) % ({roll_extent})"))
        selectors[axis] = "0"
    if expand_leading and rolls:
        pads[0] = (0, sublane_tiling - 1)

    if any(pad for _, pad in pads):
        value = f"jnp.pad({value}, {tuple(pads)!r})"
    for axis, shift in rolls:
        value = f"pltpu.roll({value}, {shift}, axis={axis})"
    # Selecting a leading block keeps whole physical sublanes and lanes. Only
    # an extract on a packed minor dimension needs a wider element dtype.
    needs_widening = (
        tensor.dtype != torch.bool
        and tensor.dtype.itemsize < 4
        and any(dim >= tensor.ndim - 2 for dim in load.scalar_dims)
    )
    if needs_widening:
        widen_dtype = "jnp.float32" if tensor.dtype.is_floating_point else "jnp.int32"
        value = f"lax.convert_element_type({value}, {widen_dtype})"

    result = f"{value}[{', '.join(selectors)}]"
    if tensor.dtype != torch.bool and not needs_widening:
        return expr_from_string(result)
    return expr_from_string(
        f"lax.convert_element_type({result}, {env.backend.dtype_str(tensor.dtype)})"
    )


def emit_vmem_scalar_load(
    tensor: torch.Tensor,
    ref_name: str,
    index_parts: list[str],
    load: VmemScalarLoad,
) -> ast.AST:
    if _sublane_load_applies(tensor, load):
        return _sublane_load_expr(tensor, ref_name, index_parts, load)
    return _roll_load_expr(tensor, ref_name, index_parts, load)
