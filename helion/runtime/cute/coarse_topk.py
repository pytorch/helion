"""Exact selection using guarded, narrower intermediate rank keys."""

from __future__ import annotations

import cutlass
from cutlass import Float32
from cutlass import Int32
from cutlass import Int64
import cutlass.cute as cute

from .topk import distributed_topk


@cute.jit
def coarse_rank_topk(
    keys: cute.Tensor,
    k: cutlass.Constexpr[int],
    lanes_per_row: cutlass.Constexpr[int],
    sort_network: cutlass.Constexpr[str],
    merge_schedule: cutlass.Constexpr[str],
    index_bits: cutlass.Constexpr[int],
    vector_width: cutlass.Constexpr[int],
    recovery: cutlass.Constexpr[str] = "direct",
    payload_only: cutlass.Constexpr[bool] = False,
) -> cute.Tensor:
    """Narrow positive-normal ranks, then refine their exact resident keys.

    Inputs are ordered Int32 ranks or the legacy Int64 rank/index encoding.
    Int32-min is padding: the compiler's canonical FP32 ranks exclude it.
    The blocked lane/register layout supplies the reversed column payload.
    Replacing low rank bits with the index can only reorder a single rank
    bucket. A unique cutoff bucket proves membership of the selected set;
    sorting its original keys restores the exact order, including ties.

    Ranks below the positive-normal range cannot beat an accepted rank and
    stay at the coarse sentinel. A positive-normal cutoff proves that at
    least k accepted ranks exist. Higher unsupported ranks and ambiguous
    cutoffs use the original sort.
    Direct recovery gathers each full rank; packed recovery gathers low-bit
    residues with the same ownership mapping. Both reconstruct identical keys.
    Payload-only consumers may omit refinement when every selected bucket
    is distinct. The low index bits then describe the exact sorted result;
    its high bits are unspecified and must never be decoded as value ranks.
    The fallback decision is uniform across the physical warp because both
    selection networks use full-warp shuffles. No producer is reevaluated.
    """
    assert recovery in ("direct", "packed")
    assert keys.element_type in (Int32, Int64)
    assert 0 < index_bits <= 23
    assert 0 < lanes_per_row <= 32
    assert lanes_per_row & (lanes_per_row - 1) == 0
    assert vector_width > 0 and vector_width & (vector_width - 1) == 0
    assert k > 0 and k & (k - 1) == 0
    size = cute.size(keys.shape)
    assert size * lanes_per_row >= k
    assert size % vector_width == 0
    rank_only = keys.element_type == Int32
    index_mask = (1 << index_bits) - 1
    padding = -9223372036854775808
    selected_size = max(1, k // lanes_per_row)
    lane = Int32(cute.arch.thread_idx()[0]) % lanes_per_row
    coarse = cute.make_rmem_tensor(size, Float32)
    bad = Int32(0)
    for i in cutlass.range_constexpr(size):
        if cutlass.const_expr(rank_only):
            rank = Int32(keys[i])
            live = rank != Int32(-2147483648)
            input_column = ((i // vector_width) * lanes_per_row + lane) * vector_width
            input_column += i % vector_width
            payload = Int32(index_mask) - input_column
        else:
            key = Int64(keys[i])
            rank = Int32(key >> index_bits)
            live = key != Int64(padding)
            payload = Int32(key & index_mask)
        coarse[i] = -Float32.inf
        if live:
            valid = (rank >= Int32(0x00800000)) & (rank < Int32(0x7F800000))
            bad += Int32(rank >= Int32(0x7F800000))
            if valid:
                bits = (rank & Int32(~index_mask)) | payload
                coarse[i] = bits.bitcast(Float32)

    selected = distributed_topk(coarse, k, lanes_per_row, sort_network, merge_schedule)
    cutoff = Int32(selected[(k - 1) // lanes_per_row].bitcast(Int32))
    cutoff = cute.arch.shuffle_sync(
        cutoff,
        offset=(k - 1) % lanes_per_row,
        mask_and_clamp=((32 - lanes_per_row) << 8) | 31,
    ) & Int32(~index_mask)
    matches = Int32(0)
    for i in cutlass.range_constexpr(size):
        if cutlass.const_expr(rank_only):
            rank = Int32(keys[i])
            live = rank != Int32(-2147483648)
        else:
            key = Int64(keys[i])
            rank = Int32(key >> index_bits)
            live = key != Int64(padding)
        matches += Int32(live & ((rank & Int32(~index_mask)) == cutoff))
    for stage in cutlass.range_constexpr(lanes_per_row.bit_length() - 1):
        matches += cute.arch.shuffle_sync_bfly(matches, offset=1 << stage)
        bad += cute.arch.shuffle_sync_bfly(bad, offset=1 << stage)
    # A sentinel cutoff can match the raw bucket of a negative rank. Check
    # the domain independently of uniqueness before recovering any payload.
    cutoff_valid = (cutoff >= Int32(0x00800000)) & (cutoff < Int32(0x7F800000))
    fallback = Int32((matches != 1) | (bad != 0) | (not cutoff_valid))
    if cutlass.const_expr(payload_only):
        collision = Int32(0)
        for i in cutlass.range_constexpr(selected_size):
            bucket = selected[i].bitcast(Int32) & Int32(~index_mask)
            previous = cute.arch.shuffle_sync(
                bucket,
                offset=(lane - 1) & (lanes_per_row - 1),
                mask_and_clamp=((32 - lanes_per_row) << 8) | 31,
            )
            if cutlass.const_expr(i > 0):
                # All lanes execute both shuffles. A lane-local choice of the
                # source slot before shuffling would read the wrong boundary.
                previous_slot = selected[i - 1].bitcast(Int32) & Int32(~index_mask)
                previous_last = cute.arch.shuffle_sync(
                    previous_slot,
                    offset=lanes_per_row - 1,
                    mask_and_clamp=((32 - lanes_per_row) << 8) | 31,
                )
                if lane == 0:
                    previous = previous_last
            position = i * lanes_per_row + lane
            # The selection helper replicates results when k < lanes.
            collision |= Int32((position > 0) & (position < k) & (bucket == previous))
        for stage in cutlass.range_constexpr(lanes_per_row.bit_length() - 1):
            collision |= cute.arch.shuffle_sync_bfly(collision, offset=1 << stage)
        fallback |= collision
    for stage in cutlass.range_constexpr(lanes_per_row.bit_length() - 1, 5):
        fallback |= cute.arch.shuffle_sync_bfly(fallback, offset=1 << stage)

    # Dominate both staged branches with the same result storage.
    result = cute.make_rmem_tensor(selected_size, Int64)
    if fallback != 0:
        if cutlass.const_expr(rank_only):
            full_keys = cute.make_rmem_tensor(size, Int64)
            for i in cutlass.range_constexpr(size):
                rank = Int32(keys[i])
                full_keys[i] = Int64(padding)
                if rank != Int32(-2147483648):
                    fallback_column = (
                        (i // vector_width) * lanes_per_row + lane
                    ) * vector_width
                    fallback_column += i % vector_width
                    full_keys[i] = (Int64(rank) << index_bits) | Int64(
                        index_mask - fallback_column
                    )
            exact = distributed_topk(
                full_keys, k, lanes_per_row, sort_network, merge_schedule
            )
        else:
            exact = distributed_topk(
                keys, k, lanes_per_row, sort_network, merge_schedule
            )
        for i in cutlass.range_constexpr(selected_size):
            result[i] = exact[i]
    elif cutlass.const_expr(payload_only):
        for i in cutlass.range_constexpr(selected_size):
            result[i] = Int64(selected[i].bitcast(Int32) & Int32(index_mask))
    else:
        # High rank bits are already present in the selected coarse key. Pack
        # only the missing low bits, so each shuffle carries several residues.
        # Signed shifts are intentional: masking after extraction recovers the
        # exact bit field even when the packed word has its sign bit set.
        residues_per_word = min(size, 1 << ((32 // index_bits).bit_length() - 1))
        packed_size = (size + residues_per_word - 1) // residues_per_word
        residues = None
        if cutlass.const_expr(recovery == "packed"):
            residues = cute.make_rmem_tensor(packed_size, Int32)
            for j in cutlass.range_constexpr(packed_size):
                word = Int32(0)
                for slot in cutlass.range_constexpr(residues_per_word):
                    register = j * residues_per_word + slot
                    if register < size:
                        if cutlass.const_expr(rank_only):
                            resident_rank = Int32(keys[register])
                        else:
                            resident_rank = Int32(Int64(keys[register]) >> index_bits)
                        residue = resident_rank & Int32(index_mask)
                        word = Int32(word) | (Int32(residue) << (slot * index_bits))
                residues[j] = word
        refined = cute.make_rmem_tensor(selected_size, Int64)
        for i in cutlass.range_constexpr(selected_size):
            bits = selected[i].bitcast(Int32)
            column = Int32(index_mask) - (bits & Int32(index_mask))
            owner = (column // vector_width) % lanes_per_row
            register = (column // (lanes_per_row * vector_width)) * vector_width
            register += column % vector_width
            if cutlass.const_expr(recovery == "packed"):
                assert residues is not None
                word = Int32(0)
                for j in cutlass.range_constexpr(packed_size):
                    peer = cute.arch.shuffle_sync(
                        residues[j],
                        offset=owner,
                        mask_and_clamp=((32 - lanes_per_row) << 8) | 31,
                    )
                    if register // residues_per_word == j:
                        word = peer
                shift = (register % residues_per_word) * index_bits
                residue = (word >> shift) & Int32(index_mask)
                rank = (bits & Int32(~index_mask)) | residue
            else:
                rank = Int32(0)
                for j in cutlass.range_constexpr(size):
                    if cutlass.const_expr(rank_only):
                        resident_rank = Int32(keys[j])
                    else:
                        resident_rank = Int32(Int64(keys[j]) >> index_bits)
                    peer = cute.arch.shuffle_sync(
                        resident_rank,
                        offset=owner,
                        mask_and_clamp=((32 - lanes_per_row) << 8) | 31,
                    )
                    if register == j:
                        rank = peer
            refined[i] = Int64(padding)
            # When k < lanes, selected contains replicas. Admit one copy only.
            if i * lanes_per_row + lane < k:
                refined[i] = (Int64(rank) << index_bits) | Int64(index_mask - column)
        exact = distributed_topk(
            refined, k, lanes_per_row, sort_network, merge_schedule
        )
        for i in cutlass.range_constexpr(selected_size):
            result[i] = exact[i]
    return result
