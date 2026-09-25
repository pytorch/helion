"""Device-only stable ordering for independent variable-length sequences."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

SEQUENCE_ORDER_THREADS = 512


@cute.kernel
def stable_sequence_order(cu_seqlens: cute.Tensor, order: cute.Tensor) -> None:
    """Sort by decreasing length, breaking ties by original sequence index.

    One warp sorts up to 32 sequences. Larger domains use one candidate per
    thread with global reads, so scratch and shared memory do not grow with N.
    The caller has already proved that offsets and lengths fit signed Int32.
    """
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    # pyrefly: ignore [bad-index, unsupported-operation]
    sequences = cutlass.Int32(cu_seqlens.shape[0] - 1)
    # pyrefly: ignore [bad-index, unsupported-operation]
    if cutlass.const_expr(cu_seqlens.shape[0] - 1 <= 32):
        if thread < 32:
            lane = thread % 32
            index = cutlass.Int32(lane)
            length = cutlass.Int32(-1)
            if lane < sequences:
                length = cutlass.Int32(cu_seqlens[lane + 1]) - cutlass.Int32(
                    cu_seqlens[lane]
                )
            for log_width in cutlass.range_constexpr(1, 6):
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
