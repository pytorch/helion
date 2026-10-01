"""Device-only stable ordering for independent variable-length sequences."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

SEQUENCE_ORDER_THREADS = 512


@cute.jit
def warp_sequence_order(cu_seqlens: cute.Tensor) -> cutlass.Int32:
    """Return the stable descending permutation for static rank-one offsets."""
    lane = cute.arch.lane_idx()
    index = cutlass.Int32(lane)
    length = cutlass.Int32(-1)
    if lane < cute.size(cu_seqlens) - 1:
        length = cutlass.Int32(cu_seqlens[lane + 1]) - cutlass.Int32(cu_seqlens[lane])
    for log_width in cutlass.range_constexpr(
        1, (cute.size(cu_seqlens) - 2).bit_length() + 1
    ):
        width = 1 << log_width
        for step in cutlass.range_constexpr(log_width):
            distance = 1 << (log_width - step - 1)
            peer_length = cute.arch.shuffle_sync_bfly(length, distance)
            peer_index = cute.arch.shuffle_sync_bfly(index, distance)
            peer_first = (peer_length > length) | (
                (peer_length == length) & (peer_index < index)
            )
            want_first = ((lane & width) == 0) == ((lane & distance) == 0)
            if peer_first == want_first:
                length = peer_length
                index = peer_index
    return index


@cute.jit
def select_sequence_by_length(
    cu_seqlens: cute.Tensor,
    sequence_slot: cutlass.Int32,
    thread: cutlass.Int32,
    THREADS: cutlass.Constexpr,
) -> cutlass.Int32:
    """Select a stable task permutation inside its consuming CTA.

    Every warp independently sorts small domains. Larger domains distribute
    candidates across the CTA and publish the unique matching rank. Offsets
    and lengths must fit signed Int32, as for the precomputed order.
    """
    if cutlass.const_expr(cute.size(cu_seqlens) - 1 <= 32):
        index = warp_sequence_order(cu_seqlens)
        result = cute.arch.shuffle_sync(index, sequence_slot)
    else:
        selected = cutlass.Array(
            cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4
        )
        sequences = cutlass.Int32(cute.size(cu_seqlens) - 1)
        for candidate in cutlass.range(thread, sequences, THREADS):
            length = cutlass.Int32(cu_seqlens[candidate + 1]) - cutlass.Int32(
                cu_seqlens[candidate]
            )
            rank = cutlass.Int32(0)
            for other in cutlass.range(sequences):
                other_length = cutlass.Int32(cu_seqlens[other + 1]) - cutlass.Int32(
                    cu_seqlens[other]
                )
                if (other_length > length) | (
                    (other_length == length) & (other < candidate)
                ):
                    rank += cutlass.Int32(1)
            if rank == sequence_slot:
                selected[0] = candidate
        cute.arch.barrier()
        result = selected[0]
    return result


@cute.kernel
def stable_sequence_order(cu_seqlens: cute.Tensor, order: cute.Tensor) -> None:
    """Sort by decreasing length, breaking ties by original sequence index.

    One warp sorts up to 32 sequences. Larger domains use one candidate per
    thread with global reads, so scratch and shared memory do not grow with N.
    The caller has already proved that offsets and lengths fit signed Int32.
    """
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    sequences = cutlass.Int32(cute.size(cu_seqlens) - 1)
    if cutlass.const_expr(cute.size(cu_seqlens) - 1 <= 32):
        if thread < 32:
            lane = thread % 32
            index = warp_sequence_order(cu_seqlens)
            if lane < sequences:
                order[lane] = index
    else:
        index = block * SEQUENCE_ORDER_THREADS + thread
        if index < sequences:
            length = cutlass.Int32(cu_seqlens[index + 1]) - cutlass.Int32(
                cu_seqlens[index]
            )
            rank = cutlass.Int32(0)
            for other in cutlass.range(sequences, unroll=1):
                other_length = cutlass.Int32(cu_seqlens[other + 1]) - cutlass.Int32(
                    cu_seqlens[other]
                )
                if (other_length > length) | (
                    (other_length == length) & (other < index)
                ):
                    rank += cutlass.Int32(1)
            order[rank] = index
