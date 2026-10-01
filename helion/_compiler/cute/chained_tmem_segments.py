"""FP32 member readback from a caller-owned full-M128 TMEM group arena."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .chained_result_transport import load_operation

if TYPE_CHECKING:
    from .chained_execution import ChainedExecution


def validate_tmem_segment(
    full_shape: tuple[int, int],
    offset: int,
    width: int,
) -> None:
    """Require a contained dense FP32 segment of the full-M128 TMEM layout."""
    if (
        len(full_shape) != 2
        or any(type(size) is not int for size in full_shape)
        or full_shape[0] != 128
        or not 16 <= full_shape[1] <= 256
        or full_shape[1] % 16
        or type(offset) is not int
        or offset < 0
        or offset % 16
        or type(width) is not int
        or width <= 0
        or width % 16
        or offset + width > full_shape[1]
    ):
        raise ValueError(
            "TMEM member load requires a contained full-M128 physical segment"
        )


def emit_tmem_segment_views(
    prefix: str,
    source_prefix: str,
    full_shape: tuple[int, int],
    offset: int,
    width: int,
) -> list[str]:
    """Restrict accumulator and identity together, keeping group coordinates."""
    validate_tmem_segment(full_shape, offset, width)
    origin = ((0, offset), 0, 0)
    shape = ((128, width), 1, 1)
    return [
        f"{prefix}_segment = cute.composition(cute.domain_offset({origin!r}, {source_prefix}_acc), cute.make_layout({shape!r}))",
        f"{prefix}_identity = cute.composition(cute.domain_offset({origin!r}, {source_prefix}_slice.partition_C(cute.make_identity_tensor({full_shape!r}))), cute.make_layout({shape!r}))",
    ]


def emit_tmem_segment_read_views(
    prefix: str,
    source_prefix: str,
    full_shape: tuple[int, int],
    offset: int,
    width: int,
    *,
    execution: ChainedExecution | None = None,
) -> list[str]:
    """Construct one C member read while retaining its global coordinates.

    ``source_prefix_acc`` and ``source_prefix_slice`` already describe the full
    FP32 group accumulator, including its allocation base. ``prefix_values`` and
    ``prefix_coords`` can feed ordinary logical result publication: columns in
    the latter remain in ``[offset, offset + width)``, not rebased to zero.

    The caller proves the member is ready, owns the first-128-thread predicate
    for wider execution roles, and places publication under that same predicate.
    This helper issues no MMA wait, CTA/role barrier, allocation or cast. The
    No memory instruction is emitted here; it grants no arena reuse.
    """
    views = emit_tmem_segment_views(prefix, source_prefix, full_shape, offset, width)
    if execution is not None and execution.threads < 128:
        raise ValueError("TMEM loads require at least 128 execution participants")
    thread = "chain_thread" if execution is None else execution.thread
    return [
        *views,
        f"{prefix}_copy = tcgen05.make_tmem_copy(cute.make_copy_atom({load_operation((128, width))}, cutlass.Float32), {prefix}_segment)",
        f"{prefix}_thread = {prefix}_copy.get_slice({thread})",
        f"{prefix}_source = {prefix}_thread.partition_S({prefix}_segment)",
        f"{prefix}_coords = {prefix}_thread.partition_D({prefix}_identity)",
        f"{prefix}_values = cute.make_rmem_tensor({prefix}_coords.shape, cutlass.Float32)",
    ]


def emit_tmem_segment_load(
    prefix: str,
    source_prefix: str,
    full_shape: tuple[int, int],
    offset: int,
    width: int,
    *,
    execution: ChainedExecution | None = None,
) -> list[str]:
    """Original member load; its fence retires only this register readback."""
    return [
        *emit_tmem_segment_read_views(
            prefix, source_prefix, full_shape, offset, width, execution=execution
        ),
        f"cute.copy({prefix}_copy, {prefix}_source, {prefix}_values)",
        "cute.arch.fence_view_async_tmem_load()",
    ]
