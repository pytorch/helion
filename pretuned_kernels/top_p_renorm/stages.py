"""Ordinary AIR stages, preserving the fourteen recorded stage configurations.

Three histogram stages each launch zeroing and processing kernels. Together
with the other eleven calls, the public pipeline makes seventeen launches.
Identical pass bodies share one kernel; pass_id remains a constexpr argument.
"""

from __future__ import annotations

from pretuned_kernels.top_p_renorm._math import add_ftz
from pretuned_kernels.top_p_renorm._math import add_rn
from pretuned_kernels.top_p_renorm._math import descending_key
from pretuned_kernels.top_p_renorm._math import from_descending_key
from pretuned_kernels.top_p_renorm._math import ftz
from pretuned_kernels.top_p_renorm._math import mul_ftz
from pretuned_kernels.top_p_renorm._math import sub_ftz
import torch

import helion
import helion.language as hl

POLICY = helion.CuteStructuralPolicy(
    cute_flatten_nested_reductions=False,
    cute_full_slice_matmul_tiling=False,
    cute_materialize_transformed_operands=False,
    cute_region_fission=False,
    cute_segmented_matmul_tiling=False,
)


@helion.aot_kernel(backend="cute", static_shapes=True, cute_structural_policy=POLICY)
def histogram(
    probs: torch.Tensor,
    prefix: torch.Tensor,
    current_count: torch.Tensor,
    compact_in: torch.Tensor,
    previous_count: torch.Tensor,
    pass_id: hl.constexpr,
    partitions: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, width = probs.shape
    capacity = ((width // 32 + 255) // 256) * 256
    hist = torch.empty((rows, 2048), dtype=torch.float32, device=probs.device)
    counts = torch.empty((rows, 2048), dtype=torch.int32, device=probs.device)
    compact = torch.empty((rows, capacity), dtype=torch.float32, device=probs.device)
    allocated = torch.empty((rows,), dtype=torch.int32, device=probs.device)
    for zr, zb in hl.tile(hist.shape):
        hist[zr, zb] = 0.0
        counts[zr, zb] = 0
        hl.store(
            allocated,
            [zr.index[:, None]],
            hl.full([zr, zb], 0, dtype=torch.int32),
            extra_mask=zb.index[None, :] == 0,
        )
    hl.barrier()
    for row, partition in hl.grid((rows, partitions)):
        local_mass = hl.zeros([2048], dtype=torch.float32)
        local_count = hl.zeros([2048], dtype=torch.int32)
        lane = hl.arange(512)
        rank = partition * 512 + lane
        original_skip = min((-row * width) % 4, width)
        if pass_id == 2:
            use_compact = previous_count[row] <= capacity
            input_len = torch.where(use_compact, previous_count[row], width)
            skip = torch.where(use_compact, 0, original_skip)
        else:
            input_len = width
            skip = original_skip
        vector_count = (input_len - skip) // 4
        for iteration in range((width + partitions * 2048 - 1) // (partitions * 2048)):
            vector_rank = rank + iteration * partitions * 512
            for item in range(4):
                column = skip + vector_rank * 4 + item
                valid = vector_rank < vector_count
                if pass_id == 2:
                    original_values = hl.load(
                        probs, [row, column], extra_mask=valid & ~use_compact
                    )
                    compact_values = hl.load(
                        compact_in, [row, column], extra_mask=valid & use_compact
                    )
                    values = torch.where(use_compact, compact_values, original_values)
                else:
                    values = hl.load(probs, [row, column], extra_mask=valid)
                key = descending_key(values)
                if pass_id == 0:
                    bucket = (key >> 21) & 2047
                    active = valid
                elif pass_id == 1:
                    bucket = (key >> 10) & 2047
                    active = valid & (((key >> 21) << 21) == prefix[row])
                else:
                    bucket = key & 1023
                    active = valid & (((key >> 10) << 10) == prefix[row])
                if pass_id == 1:
                    compact_active = active & (current_count[row] <= capacity)
                    offsets = hl.cumsum(
                        compact_active.reshape(16, 32).to(torch.int32), dim=1
                    )
                    warp_count = (
                        compact_active.reshape(16, 32)
                        .to(torch.int32)
                        .sum(dim=1, dtype=torch.int32)
                    )
                    warp = hl.arange(16)
                    warp_base = hl.atomic_add(allocated, [row + warp * 0], warp_count)
                    slots = (warp_base[:, None] + offsets - 1).reshape(512)
                    hl.store(compact, [row, slots], values, extra_mask=compact_active)
                hl.atomic_add(local_mass, [bucket], torch.where(active, values, 0.0))
                hl.atomic_add(local_count, [bucket], active.to(torch.int32))
        column = rank
        valid = rank < skip
        if pass_id == 2:
            original_values = hl.load(
                probs, [row, column], extra_mask=valid & ~use_compact
            )
            compact_values = hl.load(
                compact_in, [row, column], extra_mask=valid & use_compact
            )
            values = torch.where(use_compact, compact_values, original_values)
        else:
            values = hl.load(probs, [row, column], extra_mask=valid)
        key = descending_key(values)
        if pass_id == 0:
            bucket = (key >> 21) & 2047
            active = valid
        elif pass_id == 1:
            bucket = (key >> 10) & 2047
            active = valid & (((key >> 21) << 21) == prefix[row])
        else:
            bucket = key & 1023
            active = valid & (((key >> 10) << 10) == prefix[row])
        if pass_id == 1:
            compact_active = active & (current_count[row] <= capacity)
            offsets = hl.cumsum(compact_active.reshape(16, 32).to(torch.int32), dim=1)
            warp_count = (
                compact_active.reshape(16, 32)
                .to(torch.int32)
                .sum(dim=1, dtype=torch.int32)
            )
            warp = hl.arange(16)
            warp_base = hl.atomic_add(allocated, [row + warp * 0], warp_count)
            slots = (warp_base[:, None] + offsets - 1).reshape(512)
            hl.store(compact, [row, slots], values, extra_mask=compact_active)
        hl.atomic_add(local_mass, [bucket], torch.where(active, values, 0.0))
        hl.atomic_add(local_count, [bucket], active.to(torch.int32))
        column = skip + vector_count * 4 + rank
        valid = column < input_len
        if pass_id == 2:
            original_values = hl.load(
                probs, [row, column], extra_mask=valid & ~use_compact
            )
            compact_values = hl.load(
                compact_in, [row, column], extra_mask=valid & use_compact
            )
            values = torch.where(use_compact, compact_values, original_values)
        else:
            values = hl.load(probs, [row, column], extra_mask=valid)
        key = descending_key(values)
        if pass_id == 0:
            bucket = (key >> 21) & 2047
            active = valid
        elif pass_id == 1:
            bucket = (key >> 10) & 2047
            active = valid & (((key >> 21) << 21) == prefix[row])
        else:
            bucket = key & 1023
            active = valid & (((key >> 10) << 10) == prefix[row])
        if pass_id == 1:
            compact_active = active & (current_count[row] <= capacity)
            offsets = hl.cumsum(compact_active.reshape(16, 32).to(torch.int32), dim=1)
            warp_count = (
                compact_active.reshape(16, 32)
                .to(torch.int32)
                .sum(dim=1, dtype=torch.int32)
            )
            warp = hl.arange(16)
            warp_base = hl.atomic_add(allocated, [row + warp * 0], warp_count)
            slots = (warp_base[:, None] + offsets - 1).reshape(512)
            hl.store(compact, [row, slots], values, extra_mask=compact_active)
        hl.atomic_add(local_mass, [bucket], torch.where(active, values, 0.0))
        hl.atomic_add(local_count, [bucket], active.to(torch.int32))
        bins = hl.arange(2048)
        hl.atomic_add(hist, [row, bins], local_mass)
        hl.atomic_add(counts, [row, bins], local_count)
    return hist, counts, compact, allocated


@helion.aot_kernel(backend="cute", static_shapes=True, cute_structural_policy=POLICY)
def histogram_groups(values: torch.Tensor) -> torch.Tensor:
    """Native CG order: pair offsets16,8,4,2,1 independent of tile config."""
    rows, width = values.shape
    result = torch.empty((rows, width // 32), dtype=torch.float32, device=values.device)
    for row, group in hl.tile(result.shape):
        column = group.index[None, :] * 32
        v0 = hl.load(values, [row.index[:, None], column + 0])
        v1 = hl.load(values, [row.index[:, None], column + 1])
        v2 = hl.load(values, [row.index[:, None], column + 2])
        v3 = hl.load(values, [row.index[:, None], column + 3])
        v4 = hl.load(values, [row.index[:, None], column + 4])
        v5 = hl.load(values, [row.index[:, None], column + 5])
        v6 = hl.load(values, [row.index[:, None], column + 6])
        v7 = hl.load(values, [row.index[:, None], column + 7])
        v8 = hl.load(values, [row.index[:, None], column + 8])
        v9 = hl.load(values, [row.index[:, None], column + 9])
        v10 = hl.load(values, [row.index[:, None], column + 10])
        v11 = hl.load(values, [row.index[:, None], column + 11])
        v12 = hl.load(values, [row.index[:, None], column + 12])
        v13 = hl.load(values, [row.index[:, None], column + 13])
        v14 = hl.load(values, [row.index[:, None], column + 14])
        v15 = hl.load(values, [row.index[:, None], column + 15])
        v16 = hl.load(values, [row.index[:, None], column + 16])
        v17 = hl.load(values, [row.index[:, None], column + 17])
        v18 = hl.load(values, [row.index[:, None], column + 18])
        v19 = hl.load(values, [row.index[:, None], column + 19])
        v20 = hl.load(values, [row.index[:, None], column + 20])
        v21 = hl.load(values, [row.index[:, None], column + 21])
        v22 = hl.load(values, [row.index[:, None], column + 22])
        v23 = hl.load(values, [row.index[:, None], column + 23])
        v24 = hl.load(values, [row.index[:, None], column + 24])
        v25 = hl.load(values, [row.index[:, None], column + 25])
        v26 = hl.load(values, [row.index[:, None], column + 26])
        v27 = hl.load(values, [row.index[:, None], column + 27])
        v28 = hl.load(values, [row.index[:, None], column + 28])
        v29 = hl.load(values, [row.index[:, None], column + 29])
        v30 = hl.load(values, [row.index[:, None], column + 30])
        v31 = hl.load(values, [row.index[:, None], column + 31])
        g0_0 = add_ftz(v0, v16)
        g0_1 = add_ftz(v1, v17)
        g0_2 = add_ftz(v2, v18)
        g0_3 = add_ftz(v3, v19)
        g0_4 = add_ftz(v4, v20)
        g0_5 = add_ftz(v5, v21)
        g0_6 = add_ftz(v6, v22)
        g0_7 = add_ftz(v7, v23)
        g0_8 = add_ftz(v8, v24)
        g0_9 = add_ftz(v9, v25)
        g0_10 = add_ftz(v10, v26)
        g0_11 = add_ftz(v11, v27)
        g0_12 = add_ftz(v12, v28)
        g0_13 = add_ftz(v13, v29)
        g0_14 = add_ftz(v14, v30)
        g0_15 = add_ftz(v15, v31)
        g1_0 = add_ftz(g0_0, g0_8)
        g1_1 = add_ftz(g0_1, g0_9)
        g1_2 = add_ftz(g0_2, g0_10)
        g1_3 = add_ftz(g0_3, g0_11)
        g1_4 = add_ftz(g0_4, g0_12)
        g1_5 = add_ftz(g0_5, g0_13)
        g1_6 = add_ftz(g0_6, g0_14)
        g1_7 = add_ftz(g0_7, g0_15)
        g2_0 = add_ftz(g1_0, g1_4)
        g2_1 = add_ftz(g1_1, g1_5)
        g2_2 = add_ftz(g1_2, g1_6)
        g2_3 = add_ftz(g1_3, g1_7)
        g3_0 = add_ftz(g2_0, g2_2)
        g3_1 = add_ftz(g2_1, g2_3)
        g4_0 = add_ftz(g3_0, g3_1)
        result[row, group] = g4_0
    return result


@helion.aot_kernel(backend="cute", static_shapes=True, cute_structural_policy=POLICY)
def choose(
    hist: torch.Tensor,
    counts: torch.Tensor,
    groups: torch.Tensor,
    old_prefix: torch.Tensor,
    old_remaining: torch.Tensor,
    p: torch.Tensor | float,
    pass_id: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = hist.size(0)
    prefix = torch.empty((rows,), dtype=torch.int32, device=hist.device)
    remaining = torch.empty((rows,), dtype=torch.float32, device=hist.device)
    for row in hl.tile(rows):
        if pass_id == 0:
            h0_0 = groups[row, 0]
            h0_1 = groups[row, 1]
            h0_2 = groups[row, 2]
            h0_3 = groups[row, 3]
            h0_4 = groups[row, 4]
            h0_5 = groups[row, 5]
            h0_6 = groups[row, 6]
            h0_7 = groups[row, 7]
            h0_8 = groups[row, 8]
            h0_9 = groups[row, 9]
            h0_10 = groups[row, 10]
            h0_11 = groups[row, 11]
            h0_12 = groups[row, 12]
            h0_13 = groups[row, 13]
            h0_14 = groups[row, 14]
            h0_15 = groups[row, 15]
            h0_16 = groups[row, 16]
            h0_17 = groups[row, 17]
            h0_18 = groups[row, 18]
            h0_19 = groups[row, 19]
            h0_20 = groups[row, 20]
            h0_21 = groups[row, 21]
            h0_22 = groups[row, 22]
            h0_23 = groups[row, 23]
            h0_24 = groups[row, 24]
            h0_25 = groups[row, 25]
            h0_26 = groups[row, 26]
            h0_27 = groups[row, 27]
            h0_28 = groups[row, 28]
            h0_29 = groups[row, 29]
            h0_30 = groups[row, 30]
            h0_31 = groups[row, 31]
            h0s0_0 = add_ftz(h0_0, h0_16)
            h0s0_1 = add_ftz(h0_1, h0_17)
            h0s0_2 = add_ftz(h0_2, h0_18)
            h0s0_3 = add_ftz(h0_3, h0_19)
            h0s0_4 = add_ftz(h0_4, h0_20)
            h0s0_5 = add_ftz(h0_5, h0_21)
            h0s0_6 = add_ftz(h0_6, h0_22)
            h0s0_7 = add_ftz(h0_7, h0_23)
            h0s0_8 = add_ftz(h0_8, h0_24)
            h0s0_9 = add_ftz(h0_9, h0_25)
            h0s0_10 = add_ftz(h0_10, h0_26)
            h0s0_11 = add_ftz(h0_11, h0_27)
            h0s0_12 = add_ftz(h0_12, h0_28)
            h0s0_13 = add_ftz(h0_13, h0_29)
            h0s0_14 = add_ftz(h0_14, h0_30)
            h0s0_15 = add_ftz(h0_15, h0_31)
            h0s1_0 = add_ftz(h0s0_0, h0s0_8)
            h0s1_1 = add_ftz(h0s0_1, h0s0_9)
            h0s1_2 = add_ftz(h0s0_2, h0s0_10)
            h0s1_3 = add_ftz(h0s0_3, h0s0_11)
            h0s1_4 = add_ftz(h0s0_4, h0s0_12)
            h0s1_5 = add_ftz(h0s0_5, h0s0_13)
            h0s1_6 = add_ftz(h0s0_6, h0s0_14)
            h0s1_7 = add_ftz(h0s0_7, h0s0_15)
            h0s2_0 = add_ftz(h0s1_0, h0s1_4)
            h0s2_1 = add_ftz(h0s1_1, h0s1_5)
            h0s2_2 = add_ftz(h0s1_2, h0s1_6)
            h0s2_3 = add_ftz(h0s1_3, h0s1_7)
            h0s3_0 = add_ftz(h0s2_0, h0s2_2)
            h0s3_1 = add_ftz(h0s2_1, h0s2_3)
            h0s4_0 = add_ftz(h0s3_0, h0s3_1)
            first_half = h0s4_0
            h1_0 = groups[row, 32]
            h1_1 = groups[row, 33]
            h1_2 = groups[row, 34]
            h1_3 = groups[row, 35]
            h1_4 = groups[row, 36]
            h1_5 = groups[row, 37]
            h1_6 = groups[row, 38]
            h1_7 = groups[row, 39]
            h1_8 = groups[row, 40]
            h1_9 = groups[row, 41]
            h1_10 = groups[row, 42]
            h1_11 = groups[row, 43]
            h1_12 = groups[row, 44]
            h1_13 = groups[row, 45]
            h1_14 = groups[row, 46]
            h1_15 = groups[row, 47]
            h1_16 = groups[row, 48]
            h1_17 = groups[row, 49]
            h1_18 = groups[row, 50]
            h1_19 = groups[row, 51]
            h1_20 = groups[row, 52]
            h1_21 = groups[row, 53]
            h1_22 = groups[row, 54]
            h1_23 = groups[row, 55]
            h1_24 = groups[row, 56]
            h1_25 = groups[row, 57]
            h1_26 = groups[row, 58]
            h1_27 = groups[row, 59]
            h1_28 = groups[row, 60]
            h1_29 = groups[row, 61]
            h1_30 = groups[row, 62]
            h1_31 = groups[row, 63]
            h1s0_0 = add_ftz(h1_0, h1_16)
            h1s0_1 = add_ftz(h1_1, h1_17)
            h1s0_2 = add_ftz(h1_2, h1_18)
            h1s0_3 = add_ftz(h1_3, h1_19)
            h1s0_4 = add_ftz(h1_4, h1_20)
            h1s0_5 = add_ftz(h1_5, h1_21)
            h1s0_6 = add_ftz(h1_6, h1_22)
            h1s0_7 = add_ftz(h1_7, h1_23)
            h1s0_8 = add_ftz(h1_8, h1_24)
            h1s0_9 = add_ftz(h1_9, h1_25)
            h1s0_10 = add_ftz(h1_10, h1_26)
            h1s0_11 = add_ftz(h1_11, h1_27)
            h1s0_12 = add_ftz(h1_12, h1_28)
            h1s0_13 = add_ftz(h1_13, h1_29)
            h1s0_14 = add_ftz(h1_14, h1_30)
            h1s0_15 = add_ftz(h1_15, h1_31)
            h1s1_0 = add_ftz(h1s0_0, h1s0_8)
            h1s1_1 = add_ftz(h1s0_1, h1s0_9)
            h1s1_2 = add_ftz(h1s0_2, h1s0_10)
            h1s1_3 = add_ftz(h1s0_3, h1s0_11)
            h1s1_4 = add_ftz(h1s0_4, h1s0_12)
            h1s1_5 = add_ftz(h1s0_5, h1s0_13)
            h1s1_6 = add_ftz(h1s0_6, h1s0_14)
            h1s1_7 = add_ftz(h1s0_7, h1s0_15)
            h1s2_0 = add_ftz(h1s1_0, h1s1_4)
            h1s2_1 = add_ftz(h1s1_1, h1s1_5)
            h1s2_2 = add_ftz(h1s1_2, h1s1_6)
            h1s2_3 = add_ftz(h1s1_3, h1s1_7)
            h1s3_0 = add_ftz(h1s2_0, h1s2_2)
            h1s3_1 = add_ftz(h1s2_1, h1s2_3)
            h1s4_0 = add_ftz(h1s3_0, h1s3_1)
            second_half = h1s4_0
            if isinstance(p, torch.Tensor):
                row_p = p[row]
            else:
                row_p = hl.full([row], p, dtype=torch.float32)
            target = mul_ftz(add_ftz(first_half, second_half), row_p)
            previous_prefix = hl.full([row], 0, dtype=torch.int32)
        else:
            target = old_remaining[row]
            previous_prefix = old_prefix[row]
        previous = hl.full([row], 0.0, dtype=torch.float32)
        target_group = hl.full([row], 0, dtype=torch.int32)
        found = hl.full([row], False, dtype=torch.bool)
        for group in range(64):
            mass = groups[row, group]
            active = (ftz(mass) != 0) & ~found
            cross = ftz(add_ftz(previous, mass)) >= ftz(target)
            target_group = torch.where(active, group, target_group)
            previous = torch.where(active & ~cross, add_ftz(previous, mass), previous)
            found = found | (active & cross)
        found = hl.full([row], False, dtype=torch.bool)
        selected = hl.full([row], 0, dtype=torch.int32)
        # Native continues beyond the selected 32-bin group if rounded group
        # mass and its serial bin prefix do not cross at the same boundary.
        for bucket in range(2048):
            count = counts[row, bucket]
            mass = hist[row, bucket]
            active = (bucket >= target_group * 32) & (count != 0) & ~found
            cross = ftz(add_ftz(previous, mass)) >= ftz(target)
            selected = torch.where(active, bucket, selected)
            previous = torch.where(active & ~cross, add_ftz(previous, mass), previous)
            found = found | (active & cross)
        remaining[row] = sub_ftz(target, previous)
        if pass_id == 0:
            prefix[row] = previous_prefix | (selected << 21)
        elif pass_id == 1:
            prefix[row] = previous_prefix | (selected << 10)
        else:
            prefix[row] = previous_prefix | selected
    return prefix, remaining


@helion.aot_kernel(backend="cute", static_shapes=True, cute_structural_policy=POLICY)
def selected_count(
    counts: torch.Tensor, prefix: torch.Tensor, pass_id: hl.constexpr
) -> torch.Tensor:
    result = torch.empty((counts.size(0),), dtype=torch.int32, device=counts.device)
    flat_counts = counts.reshape(-1)
    for row in hl.tile(counts.size(0)):
        if pass_id == 0:
            bucket = (prefix[row] >> 21) & 2047
        elif pass_id == 1:
            bucket = (prefix[row] >> 10) & 2047
        else:
            bucket = prefix[row] & 1023
        result[row] = hl.load(flat_counts, [row.index * counts.size(1) + bucket])
    return result


@helion.aot_kernel(backend="cute", static_shapes=True, cute_structural_policy=POLICY)
def apply_partials(probs: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
    rows, width = probs.shape
    flat = probs.reshape(-1)
    partials = torch.empty((rows, 1024), dtype=torch.float32, device=probs.device)
    for row, tid in hl.tile(partials.shape):
        threshold = from_descending_key(prefix[row])
        total = hl.full([row, tid], 0.0, dtype=torch.float32)
        for chunk in range((width + 1023) // 1024):
            column = tid.index[None, :] + chunk * 1024
            address = row.index[:, None] * width + column
            value = hl.load(flat, [address], extra_mask=column < width)
            total = add_ftz(
                total, torch.where(ftz(value) >= ftz(threshold[:, None]), value, 0.0)
            )
        partials[row, tid] = total
    return partials


@helion.aot_kernel(backend="cute", static_shapes=True, cute_structural_policy=POLICY)
def apply_groups(values: torch.Tensor) -> torch.Tensor:
    """CUB float warp sum: adjacent pairs first (shuffle offsets1,2,4,8,16)."""
    rows, width = values.shape
    result = torch.empty((rows, width // 32), dtype=torch.float32, device=values.device)
    for row, group in hl.tile(result.shape):
        column = group.index[None, :] * 32
        v0 = hl.load(values, [row.index[:, None], column + 0])
        v1 = hl.load(values, [row.index[:, None], column + 1])
        v2 = hl.load(values, [row.index[:, None], column + 2])
        v3 = hl.load(values, [row.index[:, None], column + 3])
        v4 = hl.load(values, [row.index[:, None], column + 4])
        v5 = hl.load(values, [row.index[:, None], column + 5])
        v6 = hl.load(values, [row.index[:, None], column + 6])
        v7 = hl.load(values, [row.index[:, None], column + 7])
        v8 = hl.load(values, [row.index[:, None], column + 8])
        v9 = hl.load(values, [row.index[:, None], column + 9])
        v10 = hl.load(values, [row.index[:, None], column + 10])
        v11 = hl.load(values, [row.index[:, None], column + 11])
        v12 = hl.load(values, [row.index[:, None], column + 12])
        v13 = hl.load(values, [row.index[:, None], column + 13])
        v14 = hl.load(values, [row.index[:, None], column + 14])
        v15 = hl.load(values, [row.index[:, None], column + 15])
        v16 = hl.load(values, [row.index[:, None], column + 16])
        v17 = hl.load(values, [row.index[:, None], column + 17])
        v18 = hl.load(values, [row.index[:, None], column + 18])
        v19 = hl.load(values, [row.index[:, None], column + 19])
        v20 = hl.load(values, [row.index[:, None], column + 20])
        v21 = hl.load(values, [row.index[:, None], column + 21])
        v22 = hl.load(values, [row.index[:, None], column + 22])
        v23 = hl.load(values, [row.index[:, None], column + 23])
        v24 = hl.load(values, [row.index[:, None], column + 24])
        v25 = hl.load(values, [row.index[:, None], column + 25])
        v26 = hl.load(values, [row.index[:, None], column + 26])
        v27 = hl.load(values, [row.index[:, None], column + 27])
        v28 = hl.load(values, [row.index[:, None], column + 28])
        v29 = hl.load(values, [row.index[:, None], column + 29])
        v30 = hl.load(values, [row.index[:, None], column + 30])
        v31 = hl.load(values, [row.index[:, None], column + 31])
        s0_0 = add_rn(v0, v1)
        s0_1 = add_rn(v2, v3)
        s0_2 = add_rn(v4, v5)
        s0_3 = add_rn(v6, v7)
        s0_4 = add_rn(v8, v9)
        s0_5 = add_rn(v10, v11)
        s0_6 = add_rn(v12, v13)
        s0_7 = add_rn(v14, v15)
        s0_8 = add_rn(v16, v17)
        s0_9 = add_rn(v18, v19)
        s0_10 = add_rn(v20, v21)
        s0_11 = add_rn(v22, v23)
        s0_12 = add_rn(v24, v25)
        s0_13 = add_rn(v26, v27)
        s0_14 = add_rn(v28, v29)
        s0_15 = add_rn(v30, v31)
        s1_0 = add_rn(s0_0, s0_1)
        s1_1 = add_rn(s0_2, s0_3)
        s1_2 = add_rn(s0_4, s0_5)
        s1_3 = add_rn(s0_6, s0_7)
        s1_4 = add_rn(s0_8, s0_9)
        s1_5 = add_rn(s0_10, s0_11)
        s1_6 = add_rn(s0_12, s0_13)
        s1_7 = add_rn(s0_14, s0_15)
        s2_0 = add_rn(s1_0, s1_1)
        s2_1 = add_rn(s1_2, s1_3)
        s2_2 = add_rn(s1_4, s1_5)
        s2_3 = add_rn(s1_6, s1_7)
        s3_0 = add_rn(s2_0, s2_1)
        s3_1 = add_rn(s2_2, s2_3)
        s4_0 = add_rn(s3_0, s3_1)
        result[row, group] = s4_0
    return result


@helion.aot_kernel(backend="cute", static_shapes=True, cute_structural_policy=POLICY)
def apply_output(
    probs: torch.Tensor, prefix: torch.Tensor, mass_groups: torch.Tensor
) -> torch.Tensor:
    rows, width = probs.shape
    output = torch.empty_like(probs)
    for row, col in hl.tile(probs.shape):
        total = mass_groups[row, 0]
        for warp in range(1, 32):
            total = add_ftz(total, mass_groups[row, warp])
        # Use an approximate FP32 reciprocal with flush-to-zero and no refinement.
        reciprocal = hl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;",
            constraints="=f,f",
            args=[total],
            dtype=torch.float32,
            is_pure=True,
            pack=1,
        )
        scale = torch.where(ftz(total) > 1e-8, reciprocal, 1.0)
        threshold = from_descending_key(prefix[row])
        value = probs[row, col]
        output[row, col] = torch.where(
            ftz(value) >= ftz(threshold[:, None]), mul_ftz(value, scale[:, None]), 0.0
        )
    return output
