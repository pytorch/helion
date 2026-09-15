# ruff: noqa: ANN001
"""Root-matched three-launch Helion control for the FlashMLA megakernel."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING

import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import CompiledConfig


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def flash_mla_partial(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    task_requests: torch.Tensor,
    task_starts: torch.Tensor,
    scale: float,
    block_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one QxH by N block per compact task, with no serial N loop."""
    batch_size, query_len, num_heads, query_dim = query.shape
    _num_blocks, cache_heads, cache_block_size, cache_dim = kv_cache.shape
    assert cache_heads == 1
    assert query_dim == cache_dim
    value_dim = hl.specialize(cache_dim - 64)
    batch_size = hl.specialize(batch_size)
    query_len = hl.specialize(query_len)
    query_dim = hl.specialize(query_dim)
    cache_block_size = hl.specialize(cache_block_size)
    block_n = hl.specialize(block_n)
    heads_per_group = hl.specialize(16)
    num_head_groups = hl.specialize(num_heads // heads_per_group)
    query_rows = hl.specialize(query_len * heads_per_group)
    num_tasks = hl.specialize(task_requests.shape[0])
    num_columns = hl.specialize(block_tables.shape[1])
    cache_1d = kv_cache.view(-1)
    query_1d = query.view(-1)
    block_table_1d = block_tables.view(-1)
    partial = torch.empty(
        (
            num_tasks,
            query_len,
            num_head_groups,
            heads_per_group,
            value_dim,
        ),
        dtype=torch.float32,
        device=query.device,
    )
    partial_lse = torch.empty(
        (num_tasks, query_len, num_head_groups, heads_per_group),
        dtype=torch.float32,
        device=query.device,
    )
    qk_scale = scale * math.log2(math.e)

    for tile_task, tile_group in hl.tile(
        [num_tasks, num_head_groups], block_size=[1, 1]
    ):
        request = hl.load(task_requests, [tile_task.begin])
        task_start = hl.load(task_starts, [tile_task.begin])
        sequence_length = hl.load(seq_lens, [request])
        query_indices = hl.arange(query_len)
        causal_lengths = sequence_length - (query_len - query_indices - 1)
        head_indices = tile_group.begin * heads_per_group + hl.arange(heads_per_group)
        position = task_start + hl.arange(block_n)
        block_column = torch.clamp(position // cache_block_size, 0, num_columns - 1)
        block_offset = position % cache_block_size
        physical_block = block_table_1d[request * num_columns + block_column]
        cache_slot = physical_block * cache_block_size + block_offset
        cache_base = cache_slot[:, None] * query_dim
        query_base = (
            (request * query_len + query_indices[:, None, None]) * num_heads
            + head_indices[None, :, None]
        ) * query_dim

        # The latent and RoPE dimensions are contiguous and use the same
        # scaling, so one K128 loop preserves their FP32 accumulation order.
        # Keeping the score as one loop-carried value is also essential inside
        # a persistent dispatch loop: Triton otherwise allocates one 128-column
        # TMEM buffer for every source-level ``scores = dot(..., acc=scores)``.
        scores = hl.zeros([query_rows, block_n], dtype=torch.float32)
        for tile_k in hl.tile(query_dim, block_size=128):
            k_chunk = hl.load(
                cache_1d,
                [cache_base + tile_k.index[None, :]],
                eviction_policy="evict_first",
            )
            q_chunk = (
                hl.load(
                    query_1d,
                    [query_base + tile_k.index[None, None, :]],
                )
                * qk_scale
            ).view(query_rows, tile_k.block_size)
            scores = hl.dot(q_chunk, k_chunk.T, acc=scores)
        scores_3d = scores.view(query_len, heads_per_group, block_n)
        valid = position[None, None, :] < causal_lengths[:, None, None]
        masked_scores = torch.where(valid, scores_3d, float("-inf"))
        block_max = torch.amax(masked_scores, dim=-1)
        probabilities = torch.exp2(masked_scores - block_max[:, :, None])
        block_sum = torch.sum(probabilities, dim=-1)
        empty = block_max == float("-inf")
        safe_sum = torch.where(empty, 1.0, block_sum)
        probability_rows = probabilities.view(query_rows, block_n).to(torch.bfloat16)
        # Rebuild the small address graph at the phase boundary.  In
        # particular, do not keep an N-by-D index tensor live from QK to PV.
        pv_block_column = torch.clamp(position // cache_block_size, 0, num_columns - 1)
        pv_block_offset = position % cache_block_size
        pv_physical_block = block_table_1d[request * num_columns + pv_block_column]
        pv_cache_slot = pv_physical_block * cache_block_size + pv_block_offset
        pv_cache_base = pv_cache_slot[:, None] * query_dim
        # Preserve M64 so Blackwell can use TCGen5/TMEM, but bound each FP32 PV
        # accumulator to D128 and store it before creating the next one.  This
        # is the same D-chunk organization used by ThunderKittens and does not
        # alter the K=N128 accumulation order or precision.
        for tile_d in hl.tile(value_dim, block_size=128):
            value_index = pv_cache_base + tile_d.index[None, :]
            values = hl.load(cache_1d, [value_index], eviction_policy="evict_last")
            accumulator = hl.dot(
                probability_rows,
                values,
                out_dtype=torch.float32,
            )
            normalized = torch.where(
                empty.view(query_rows, 1),
                0.0,
                accumulator / safe_sum.view(query_rows, 1),
            ).view(query_len, heads_per_group, tile_d.block_size)
            partial[tile_task, query_indices, tile_group, :, tile_d] = normalized[
                None, :, None, :, :
            ]
        partial_lse[tile_task, query_indices, tile_group, :] = torch.where(
            empty,
            float("-inf"),
            block_max + torch.log2(safe_sum),
        )[None, :, None, :]
    return partial, partial_lse


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def flash_mla_final(
    partial: torch.Tensor,
    partial_lse: torch.Tensor,
    task_offsets: torch.Tensor,
) -> torch.Tensor:
    """Online reduction over each request's contiguous compact-task interval."""
    (
        _num_tasks,
        query_len,
        num_head_groups,
        heads_per_group,
        value_dim,
    ) = partial.shape
    batch_size = hl.specialize(task_offsets.shape[0] - 1)
    num_heads = hl.specialize(num_head_groups * heads_per_group)
    output = torch.empty(
        (batch_size, query_len, num_head_groups, heads_per_group, value_dim),
        dtype=torch.bfloat16,
        device=partial.device,
    )

    for tile_group, tile_h, tile_b, tile_q, tile_d in hl.tile(
        [num_head_groups, heads_per_group, batch_size, query_len, value_dim],
        block_size=[1, None, 1, 1, None],
    ):
        task_begin = hl.load(task_offsets, [tile_b.begin])
        task_end = hl.load(task_offsets, [tile_b.begin + 1])
        task_count = task_end - task_begin
        running_max = hl.full([tile_h.block_size], float("-inf"), torch.float32)
        running_sum = hl.zeros([tile_h.block_size], dtype=torch.float32)
        accumulator = hl.zeros(
            [tile_h.block_size, tile_d.block_size], dtype=torch.float32
        )
        for tile_task in hl.tile(task_count, block_size=1):
            task = task_begin + tile_task.begin
            lse = partial_lse[task, tile_q.begin, tile_group.begin, tile_h]
            new_max = torch.maximum(running_max, lse)
            old_weight = torch.exp2(running_max - new_max)
            new_weight = torch.exp2(lse - new_max)
            running_sum = running_sum * old_weight + new_weight
            accumulator = (
                accumulator * old_weight[:, None]
                + partial[task, tile_q.begin, tile_group.begin, tile_h, tile_d]
                * new_weight[:, None]
            )
            running_max = new_max
        output[tile_b.begin, tile_q.begin, tile_group, tile_h, tile_d] = (
            accumulator / running_sum[:, None]
        )[None, :, :]
    return output.view(batch_size, query_len, num_heads, value_dim)


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def flash_mla_radix(
    partial: torch.Tensor,
    partial_lse: torch.Tensor,
    group_starts: torch.Tensor,
    group_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge one bounded contiguous group while streaming child states."""
    (
        _num_tasks,
        query_len,
        num_head_groups,
        heads_per_group,
        value_dim,
    ) = partial.shape
    num_groups = hl.specialize(group_starts.shape[0])
    output = torch.empty(
        (
            num_groups,
            query_len,
            num_head_groups,
            heads_per_group,
            value_dim,
        ),
        dtype=torch.float32,
        device=partial.device,
    )
    output_lse = torch.empty(
        (num_groups, query_len, num_head_groups, heads_per_group),
        dtype=torch.float32,
        device=partial.device,
    )

    for tile_group, tile_q, tile_head_group, tile_h, tile_d in hl.tile(
        [
            num_groups,
            query_len,
            num_head_groups,
            heads_per_group,
            value_dim,
        ],
        block_size=[1, 1, 1, None, None],
    ):
        child_begin = hl.load(group_starts, [tile_group.begin])
        child_count = hl.load(group_counts, [tile_group.begin])
        best = hl.full([tile_h.block_size], float("-inf"), torch.float32)
        # Determine the common normalization before touching the value tensor.
        # This production-style two-pass form avoids rescaling the entire
        # HxD accumulator once per child, which dominates an online merge.
        for tile_child in hl.tile(child_count, block_size=1):
            child = child_begin + tile_child.begin
            child_lse = partial_lse[child, tile_q.begin, tile_head_group.begin, tile_h]
            best = torch.maximum(best, child_lse)

        denominator = hl.zeros([tile_h.block_size], dtype=torch.float32)
        accumulator = hl.zeros(
            [tile_h.block_size, tile_d.block_size], dtype=torch.float32
        )
        for tile_child in hl.tile(child_count, block_size=1):
            child = child_begin + tile_child.begin
            child_lse = partial_lse[child, tile_q.begin, tile_head_group.begin, tile_h]
            child_weight = torch.exp2(child_lse - best)
            denominator = denominator + child_weight
            accumulator = (
                accumulator
                + partial[
                    child,
                    tile_q.begin,
                    tile_head_group.begin,
                    tile_h,
                    tile_d,
                ]
                * child_weight[:, None]
            )

        safe_denominator = torch.where(denominator == 0, 1.0, denominator)
        output[
            tile_group.begin,
            tile_q.begin,
            tile_head_group.begin,
            tile_h,
            tile_d,
        ] = accumulator / safe_denominator[:, None]
        output_lse[
            tile_group.begin,
            tile_q.begin,
            tile_head_group.begin,
            tile_h,
        ] = torch.where(
            denominator == 0,
            float("-inf"),
            best + torch.log2(safe_denominator),
        )
    return output, output_lse


CONFIGS: dict[str, dict[str, object]] = {
    "partial": {
        "block_sizes": [],
        "loop_orders": [[0, 1]],
        "l2_groupings": [2],
        "range_unroll_factors": [0, 0, 0],
        "range_warp_specializes": [None, None, None],
        "range_num_stages": [0, 0, 0],
        "range_multi_buffers": [None, None, None],
        "range_flattens": [None, None, None],
        "static_ranges": [False, False],
        "load_eviction_policies": ["", "", "", "", "", ""],
        "num_warps": 8,
        "num_stages": 1,
        "indexing": ["pointer"] * 10,
        "pid_type": "flat",
        "atomic_indexing": [],
    },
    "radix": {
        "block_sizes": [16, 512],
        "loop_orders": [[0, 1, 2, 3, 4]],
        "l2_groupings": [8],
        "range_unroll_factors": [0, 1, 1],
        "range_warp_specializes": [None, False, False],
        "range_num_stages": [0, 1, 1],
        "range_multi_buffers": [None, True, True],
        "range_flattens": [None, True, True],
        "load_eviction_policies": ["", "", "", "", ""],
        "num_warps": 8,
        "num_stages": 2,
        "indexing": ["pointer"] * 7,
        "pid_type": "flat",
        "atomic_indexing": [],
    },
    "final": {
        "block_sizes": [1, 512],
        "loop_orders": [[2, 3, 0, 1, 4]],
        "l2_groupings": [32],
        "range_unroll_factors": [0, 8],
        "range_warp_specializes": [None, False],
        "range_num_stages": [0, 4],
        "range_multi_buffers": [None, True],
        "range_flattens": [None, True],
        "load_eviction_policies": ["", "", "", ""],
        "num_warps": 4,
        "num_stages": 8,
        "indexing": ["pointer"] * 5,
        "pid_type": "persistent_interleaved",
        "num_sm_multiplier": 16,
        "maxnreg": 128,
        "atomic_indexing": [],
    },
}


def _compile(kernel, args, config_values: dict[str, object]) -> CompiledConfig:
    bound = kernel.bind(args)
    values = copy.deepcopy(dict(bound.config_spec.default_config()))
    values.update(config_values)
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return bound.compile_config(config)


def build(
    tensors: dict[str, torch.Tensor],
    *,
    scale: float,
    block_n: int,
) -> tuple[Callable[[], torch.Tensor], tuple[CompiledConfig, ...], torch.Tensor]:
    """Compile the independently tuned partial, radix, and final launches."""
    partial_args = (
        tensors["query"],
        tensors["kv_cache"],
        tensors["block_tables"],
        tensors["seq_lens"],
        tensors["task_requests"],
        tensors["task_starts"],
        scale,
        block_n,
    )
    partial_call = _compile(flash_mla_partial, partial_args, CONFIGS["partial"])
    partial, partial_lse = partial_call(*partial_args)

    radix_args = (
        partial,
        partial_lse,
        tensors["group_starts"],
        tensors["group_counts"],
    )
    radix_call = _compile(flash_mla_radix, radix_args, CONFIGS["radix"])
    grouped, grouped_lse = radix_call(*radix_args)

    final_args = (grouped, grouped_lse, tensors["group_offsets"])
    final_call = _compile(flash_mla_final, final_args, CONFIGS["final"])

    def launch() -> torch.Tensor:
        partial_value, partial_lse_value = partial_call(*partial_args)
        grouped_value, grouped_lse_value = radix_call(
            partial_value,
            partial_lse_value,
            tensors["group_starts"],
            tensors["group_counts"],
        )
        return final_call(grouped_value, grouped_lse_value, tensors["group_offsets"])

    output = launch()
    return launch, (partial_call, radix_call, final_call), output
