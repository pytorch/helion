"""Ragged Q4/H16 FlashMLA attention megakernel, pretuned for NVIDIA B200.

The kernel preserves ThunderMLA's prepared-query/paged-cache boundary and its
BF16 operands/output with FP32 softmax, MMA accumulation, LSE, and reduction
state. It uses a fixed N128/radix-16 capacity envelope whose runtime descriptors
select the active attention topology. The benchmark compares against both the
root-matched three-launch Helion implementation and vLLM's production
FlashInfer MLA decode entry point.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import CompiledConfig


SEQUENCE_LENGTH_CASES = (
    ("canonical", (1696, 1730, 4641, 45118), 204),
    ("canonical_same_bucket", (1789, 1665, 4735, 45183), 207),
    ("skewed", (637, 2681, 16499, 51204), 206),
    ("balanced", (8191, 8196, 8401, 8573), 205),
)
SEQUENCE_LENGTHS = SEQUENCE_LENGTH_CASES[0][1]
BATCH = 4
QUERY_TOKENS = 4
NUM_HEADS = 16
KV_LORA_RANK = 512
ROPE_DIM = 64
QK_DIM = KV_LORA_RANK + ROPE_DIM
CACHE_BLOCK_SIZE = 64
BLOCK_N = 128
RADIX_FAN_IN = 16
MLA_SCALE = QK_DIM**-0.5
GROUP_CAPACITY = 38
TASK_CAPACITY = GROUP_CAPACITY * RADIX_FAN_IN
KV_BLOCK_CAPACITY = max(
    sum(math.ceil(length / CACHE_BLOCK_SIZE) for length in sequence_lengths)
    for _label, sequence_lengths, _seed in SEQUENCE_LENGTH_CASES
)
MAX_REQUEST_BLOCKS = max(
    math.ceil(max(sequence_lengths) / CACHE_BLOCK_SIZE)
    for _label, sequence_lengths, _seed in SEQUENCE_LENGTH_CASES
)
# FlashInfer requires enough page-table columns for a whole N128 scheduling
# quantum when the physical cache page is N64.
BLOCK_TABLE_CAPACITY = math.ceil(MAX_REQUEST_BLOCKS / 2) * 2


OPAQUE_PARTIAL_LOAD = """
tl.load(
    {partial}
    + {child} * 32768
    + {query} * 8192
    + {head_group} * 8192
    + {heads}[:, None] * 512
    + {values}[None, :]
)
"""

OPAQUE_PARTIAL_LSE_LOAD = """
tl.load(
    {partial_lse}
    + {child} * 64
    + {query} * 16
    + {head_group} * 16
    + {heads}
)
"""


@helion.aot_kernel(
    static_shapes=False,
    backend="triton",
    triton_do_not_specialize=True,
)
def flash_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    task_requests: torch.Tensor,
    task_starts: torch.Tensor,
    group_counts: torch.Tensor,
    group_offsets: torch.Tensor,
    scale: float,
    block_n: int,
) -> torch.Tensor:
    """Fuse partial, radix, and final roots for a runtime attention topology."""
    batch_size, query_len, num_heads, query_dim = query.shape
    _num_blocks, cache_heads, cache_block_size, cache_dim = kv_cache.shape
    batch_size = hl.specialize(batch_size)
    query_len = hl.specialize(query_len)
    num_heads = hl.specialize(num_heads)
    query_dim = hl.specialize(query_dim)
    cache_heads = hl.specialize(cache_heads)
    cache_block_size = hl.specialize(cache_block_size)
    cache_dim = hl.specialize(cache_dim)
    block_n = hl.specialize(block_n)
    value_dim = hl.specialize(cache_dim - 64)
    heads_per_group = hl.specialize(16)
    num_head_groups = hl.specialize(num_heads // heads_per_group)
    query_rows = hl.specialize(query_len * heads_per_group)
    assert query_len == QUERY_TOKENS
    assert num_heads == NUM_HEADS
    assert query_dim == QK_DIM
    assert cache_heads == 1
    assert cache_block_size == CACHE_BLOCK_SIZE
    assert cache_dim == QK_DIM
    assert value_dim == KV_LORA_RANK
    assert block_n == BLOCK_N
    task_capacity = hl.specialize(task_requests.size(0))
    group_capacity = hl.specialize(group_counts.size(0))
    group_offset_count = hl.specialize(group_offsets.size(0))
    num_columns = hl.specialize(block_tables.size(1))
    assert task_capacity == group_capacity * RADIX_FAN_IN
    assert group_offset_count == batch_size + 1
    hl.specialize(query.stride())
    hl.specialize(kv_cache.stride())
    hl.specialize(block_tables.stride())
    hl.specialize(seq_lens.stride())
    hl.specialize(task_requests.stride())
    hl.specialize(task_starts.stride())
    hl.specialize(group_counts.stride())
    hl.specialize(group_offsets.stride())
    cache_1d = kv_cache.view(-1)
    query_1d = query.view(-1)
    block_table_1d = block_tables.view(-1)
    partial = torch.empty(
        (
            task_capacity,
            query_len,
            num_head_groups,
            heads_per_group,
            value_dim,
        ),
        dtype=torch.float32,
        device=query.device,
    )
    partial_lse = torch.empty(
        (
            task_capacity,
            query_len,
            num_head_groups,
            heads_per_group,
        ),
        dtype=torch.float32,
        device=query.device,
    )
    # Publish one small, unconditional completion value per partial task.  The
    # partial payload is intentionally loaded through opaque expressions below;
    # this affine side relation is the single dependency source of truth for
    # partial -> radix readiness, including inactive envelope tickets.
    partial_ready = torch.empty(
        (task_capacity, num_head_groups),
        dtype=torch.int32,
        device=query.device,
    )
    partial_ready_groups = partial_ready.view(
        group_capacity, RADIX_FAN_IN, num_head_groups
    )
    grouped = torch.empty(
        (
            group_capacity,
            query_len,
            num_head_groups,
            heads_per_group,
            value_dim,
        ),
        dtype=torch.float32,
        device=query.device,
    )
    grouped_lse = torch.empty(
        (
            group_capacity,
            query_len,
            num_head_groups,
            heads_per_group,
        ),
        dtype=torch.float32,
        device=query.device,
    )
    grouped_ready = torch.empty(
        (group_capacity, query_len, num_head_groups),
        dtype=torch.int32,
        device=query.device,
    )
    output = torch.empty(
        (
            batch_size,
            query_len,
            num_head_groups,
            heads_per_group,
            value_dim,
        ),
        dtype=torch.bfloat16,
        device=query.device,
    )
    qk_scale = scale * math.log2(math.e)

    # A fixed global envelope keeps every producer/consumer relation affine.
    # Runtime descriptors identify real tasks and request-local tail padding.
    for tile_task, tile_group in hl.tile(
        [task_capacity, num_head_groups], block_size=[1, 1]
    ):
        request = hl.load(task_requests, [tile_task.begin])
        query_indices = hl.arange(query_len)
        if request >= 0:
            task_start = hl.load(task_starts, [tile_task.begin])
            sequence_length = hl.load(seq_lens, [request])
            causal_lengths = sequence_length - (query_len - query_indices - 1)
            head_indices = tile_group.begin * heads_per_group + hl.arange(
                heads_per_group
            )
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
            probability_rows = probabilities.view(query_rows, block_n).to(
                torch.bfloat16
            )

            pv_block_column = torch.clamp(
                position // cache_block_size, 0, num_columns - 1
            )
            pv_block_offset = position % cache_block_size
            pv_physical_block = block_table_1d[request * num_columns + pv_block_column]
            pv_cache_slot = pv_physical_block * cache_block_size + pv_block_offset
            pv_cache_base = pv_cache_slot[:, None] * query_dim
            for tile_d in hl.tile(value_dim, block_size=128):
                values = hl.load(
                    cache_1d,
                    [pv_cache_base + tile_d.index[None, :]],
                    eviction_policy="evict_last",
                )
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
                partial[
                    tile_task,
                    query_indices,
                    tile_group,
                    :,
                    tile_d,
                ] = normalized[None, :, None, :, :]
            partial_lse[
                tile_task,
                query_indices,
                tile_group,
                :,
            ] = torch.where(
                empty,
                float("-inf"),
                block_max + torch.log2(safe_sum),
            )[None, :, None, :]
        partial_ready[tile_task, tile_group] = 1

    # Every radix task consumes exactly 16 affine slots. Runtime child counts
    # mask the request-local tail, and zero marks an inactive capacity slot.
    for radix_group, tile_q, tile_group, tile_h, tile_d in hl.tile(
        [
            group_capacity,
            query_len,
            num_head_groups,
            heads_per_group,
            value_dim,
        ],
        block_size=[1, 1, 1, 16, 512],
    ):
        child_count = hl.load(group_counts, [radix_group.begin])
        best = hl.full([tile_h.block_size], float("-inf"), torch.float32)
        for child_tile in hl.tile(RADIX_FAN_IN, block_size=1):
            child_ready = partial_ready_groups[
                radix_group.begin,
                child_tile.begin,
                tile_group.begin,
            ]
            if (child_tile.begin < child_count) & (child_ready != 0):
                child_lse = hl.inline_triton(
                    OPAQUE_PARTIAL_LSE_LOAD,
                    args={
                        "partial_lse": partial_lse,
                        "child": (radix_group.begin * RADIX_FAN_IN + child_tile.begin),
                        "query": tile_q.begin,
                        "head_group": tile_group.begin,
                        "heads": tile_h.index,
                    },
                    output_like=best,
                )
                best = torch.maximum(best, child_lse)

        denominator = hl.zeros([tile_h.block_size], dtype=torch.float32)
        accumulator = hl.zeros(
            [tile_h.block_size, tile_d.block_size], dtype=torch.float32
        )
        child = torch.zeros([], dtype=torch.int32, device=query.device)
        while child < child_count:
            child_lse = hl.inline_triton(
                OPAQUE_PARTIAL_LSE_LOAD,
                args={
                    "partial_lse": partial_lse,
                    "child": radix_group.begin * RADIX_FAN_IN + child,
                    "query": tile_q.begin,
                    "head_group": tile_group.begin,
                    "heads": tile_h.index,
                },
                output_like=best,
            )
            child_weight = torch.exp2(child_lse - best)
            denominator = denominator + child_weight
            child_value = hl.inline_triton(
                OPAQUE_PARTIAL_LOAD,
                args={
                    "partial": partial,
                    "child": radix_group.begin * RADIX_FAN_IN + child,
                    "query": tile_q.begin,
                    "head_group": tile_group.begin,
                    "heads": tile_h.index,
                    "values": tile_d.index,
                },
                output_like=accumulator,
            )
            accumulator = accumulator + child_value * child_weight[:, None]
            child = child + 1
        safe_denominator = torch.where(denominator == 0, 1.0, denominator)
        grouped[
            radix_group.begin,
            tile_q.begin,
            tile_group.begin,
            tile_h,
            tile_d,
        ] = torch.where(
            (denominator == 0)[:, None],
            0.0,
            accumulator / safe_denominator[:, None],
        )
        grouped_lse[
            radix_group.begin,
            tile_q.begin,
            tile_group.begin,
            tile_h,
        ] = torch.where(
            denominator == 0,
            float("-inf"),
            best + torch.log2(safe_denominator),
        )
        grouped_ready[radix_group.begin, tile_q.begin, tile_group.begin] = 1

    # The small readiness relation remains a fixed affine scan, while opaque
    # payload loads visit only the request's runtime group interval.
    for tile_b, tile_q, tile_group, tile_h, tile_d in hl.tile(
        [batch_size, query_len, num_head_groups, heads_per_group, value_dim],
        block_size=[1, 1, 1, 1, 512],
    ):
        group_begin = hl.load(group_offsets, [tile_b.begin])
        group_end = hl.load(group_offsets, [tile_b.begin + 1])
        running_max = hl.full([tile_h.block_size], float("-inf"), torch.float32)
        running_sum = hl.zeros([tile_h.block_size], dtype=torch.float32)
        accumulator = hl.zeros(
            [tile_h.block_size, tile_d.block_size], dtype=torch.float32
        )
        ready_count = torch.zeros([], dtype=torch.int32, device=query.device)
        for ready_group in hl.tile(group_capacity, block_size=1):
            ready_count = (
                ready_count
                + grouped_ready[
                    ready_group.begin,
                    tile_q.begin,
                    tile_group.begin,
                ]
            )
        final_group = group_begin
        while (final_group < group_end) & (ready_count == group_capacity):
            state_lse = hl.inline_triton(
                OPAQUE_PARTIAL_LSE_LOAD,
                args={
                    "partial_lse": grouped_lse,
                    "child": final_group,
                    "query": tile_q.begin,
                    "head_group": tile_group.begin,
                    "heads": tile_h.index,
                },
                output_like=running_max,
            )
            new_max = torch.maximum(running_max, state_lse)
            old_weight = torch.where(
                running_max == float("-inf"),
                0.0,
                torch.exp2(running_max - new_max),
            )
            state_weight = torch.exp2(state_lse - new_max)
            running_sum = running_sum * old_weight + state_weight
            state_value = hl.inline_triton(
                OPAQUE_PARTIAL_LOAD,
                args={
                    "partial": grouped,
                    "child": final_group,
                    "query": tile_q.begin,
                    "head_group": tile_group.begin,
                    "heads": tile_h.index,
                    "values": tile_d.index,
                },
                output_like=accumulator,
            )
            accumulator = (
                accumulator * old_weight[:, None] + state_value * state_weight[:, None]
            )
            running_max = new_max
            final_group = final_group + 1
        output[
            tile_b.begin,
            tile_q.begin,
            tile_group.begin,
            tile_h,
            tile_d,
        ] = accumulator / running_sum[:, None]
    return output.view(batch_size, query_len, num_heads, value_dim)


def use_cudagraph() -> bool:
    """The timed closures replay pre-captured CUDA graphs."""
    return True


def has_vllm() -> bool:
    """Whether vLLM's production FlashInfer MLA dependency is importable."""
    try:
        from flashinfer.decode import (  # noqa: F401
            trtllm_batch_decode_with_kv_cache_mla,
        )
        from vllm.v1.attention.backends.mla.flashinfer_mla import (  # noqa: F401
            _get_workspace_buffer,
        )
    except ImportError:
        return False
    return True


def _require_sm100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("flash_mla is pretuned only for NVIDIA SM100")


def _topology_descriptors(lengths: tuple[int, ...]) -> dict[str, torch.Tensor]:
    """Build compact standalone metadata and the fixed persistent envelope."""
    compact_task_requests: list[int] = []
    compact_task_starts: list[int] = []
    task_offsets = [0]
    group_starts: list[int] = []
    compact_group_counts: list[int] = []
    group_offsets = [0]
    task_counts = tuple(math.ceil(length / BLOCK_N) for length in lengths)

    for request, task_count in enumerate(task_counts):
        task_begin = len(compact_task_requests)
        compact_task_requests.extend([request] * task_count)
        compact_task_starts.extend(task * BLOCK_N for task in range(task_count))
        task_offsets.append(len(compact_task_requests))
        for local_begin in range(0, task_count, RADIX_FAN_IN):
            group_starts.append(task_begin + local_begin)
            compact_group_counts.append(min(RADIX_FAN_IN, task_count - local_begin))
        group_offsets.append(len(compact_group_counts))

    if len(compact_group_counts) > GROUP_CAPACITY:
        raise ValueError(
            f"topology needs {len(compact_group_counts)} radix groups, "
            f"but the pretuned envelope has {GROUP_CAPACITY}"
        )

    # Each runtime group owns one full radix interval. Inactive tail and
    # envelope tickets still publish their tiny completion value, but skip the
    # attention body.
    task_requests: list[int] = []
    task_starts: list[int] = []
    for request, task_count in enumerate(task_counts):
        for local_begin in range(0, task_count, RADIX_FAN_IN):
            children = min(RADIX_FAN_IN, task_count - local_begin)
            for child in range(RADIX_FAN_IN):
                active = child < children
                task_requests.append(request if active else -1)
                task_starts.append((local_begin + child) * BLOCK_N if active else 0)
    task_requests.extend([-1] * (TASK_CAPACITY - len(task_requests)))
    task_starts.extend([0] * (TASK_CAPACITY - len(task_starts)))
    group_counts = compact_group_counts + [0] * (
        GROUP_CAPACITY - len(compact_group_counts)
    )

    def int_tensor(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    return {
        "task_requests": int_tensor(task_requests),
        "task_starts": int_tensor(task_starts),
        "group_counts": int_tensor(group_counts),
        "group_offsets": int_tensor(group_offsets),
        "compact_task_requests": int_tensor(compact_task_requests),
        "compact_task_starts": int_tensor(compact_task_starts),
        "task_offsets": int_tensor(task_offsets),
        "group_starts": int_tensor(group_starts),
        "compact_group_counts": int_tensor(compact_group_counts),
    }


def _make_inputs(
    sequence_lengths: tuple[int, ...] = SEQUENCE_LENGTHS,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Create one runtime topology inside the fixed B4 physical envelope."""
    if len(sequence_lengths) != BATCH:
        raise ValueError(f"expected {BATCH} sequence lengths, got {sequence_lengths}")
    torch.manual_seed(seed)
    block_counts = tuple(
        math.ceil(length / CACHE_BLOCK_SIZE) for length in sequence_lengths
    )
    if sum(block_counts) > KV_BLOCK_CAPACITY:
        raise ValueError("sequence lengths exceed the pretuned KV-cache capacity")
    if max(block_counts) > BLOCK_TABLE_CAPACITY:
        raise ValueError("sequence lengths exceed the pretuned page-table capacity")
    query = torch.randn(
        (BATCH, QUERY_TOKENS, NUM_HEADS, QK_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )
    kv_cache = torch.randn(
        (KV_BLOCK_CAPACITY, 1, CACHE_BLOCK_SIZE, QK_DIM),
        device="cuda",
        dtype=torch.bfloat16,
    )
    block_tables = torch.zeros(
        (BATCH, BLOCK_TABLE_CAPACITY), device="cuda", dtype=torch.int32
    )
    physical_blocks = torch.randperm(
        sum(block_counts), device="cuda", dtype=torch.int64
    )
    block_begin = 0
    for request, block_count in enumerate(block_counts):
        block_tables[request, :block_count] = physical_blocks[
            block_begin : block_begin + block_count
        ].to(torch.int32)
        block_begin += block_count
    return {
        "query": query,
        "kv_cache": kv_cache,
        "block_tables": block_tables,
        "seq_lens": torch.tensor(sequence_lengths, device="cuda", dtype=torch.int32),
        **_topology_descriptors(sequence_lengths),
    }


def _kernel_args(tensors: dict[str, torch.Tensor]) -> tuple[object, ...]:
    return (
        tensors["query"],
        tensors["kv_cache"],
        tensors["block_tables"],
        tensors["seq_lens"],
        tensors["task_requests"],
        tensors["task_starts"],
        tensors["group_counts"],
        tensors["group_offsets"],
        MLA_SCALE,
        BLOCK_N,
    )


def _make_standalone_call(
    tensors: dict[str, torch.Tensor],
) -> tuple[Callable[[], torch.Tensor], tuple[CompiledConfig, ...], torch.Tensor]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from pretuned_kernels.megakernels.flash_mla import _standalone

    standalone_tensors = {
        **tensors,
        "task_requests": tensors["compact_task_requests"],
        "task_starts": tensors["compact_task_starts"],
        "group_counts": tensors["compact_group_counts"],
    }
    return _standalone.build(standalone_tensors, scale=MLA_SCALE, block_n=BLOCK_N)


def _compiled_triton_kernels(call: object) -> tuple[object, ...]:
    """Find the compiled Triton functions owned by a Helion launch callable."""
    launchers = [call]
    for bound in getattr(call, "_dispatch_cache", {}).values():
        if (run := getattr(bound, "_run", None)) is not None:
            launchers.append(run)
    if (prepared := getattr(call, "_prepared_call", None)) is not None:
        if (run := getattr(prepared.bound, "_run", None)) is not None:
            launchers.append(run)

    kernels: list[object] = []
    for launcher in launchers:
        for value in getattr(launcher, "__globals__", {}).values():
            device_caches = getattr(value, "device_caches", None)
            if device_caches is None:
                continue
            for cache_pair in device_caches.values():
                for compiled in cache_pair[0].values():
                    if all(compiled is not previous for previous in kernels):
                        kernels.append(compiled)
    return tuple(kernels)


def _dispatch_cache_signature(call: object) -> tuple[tuple[object, int], ...]:
    """Identify the already-bound launchers used by an AOT kernel."""
    cache = getattr(call, "_dispatch_cache", None)
    if not cache:
        raise RuntimeError("AOT dispatch cache was not populated after launch")
    return tuple((key, id(bound)) for key, bound in cache.items())


def _prefer_no_cuda_cache_partition(call: object, tensor: torch.Tensor) -> None:
    """Apply the measured CUDA launch attribute without extending Helion config."""
    import importlib

    kernels = _compiled_triton_kernels(call)
    if not kernels:
        raise RuntimeError("no compiled Triton functions found for cache preference")
    cuda_driver = importlib.import_module("cuda.bindings.driver")
    preference = cuda_driver.CUfunc_cache.CU_FUNC_CACHE_PREFER_NONE
    with torch.cuda.device(tensor.device):
        for compiled in kernels:
            function = cuda_driver.CUfunction(int(compiled.function))
            error = cuda_driver.cuFuncSetCacheConfig(function, preference)[0]
            if error != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    f"CUDA cache-preference configuration failed: {error}"
                )


def _make_vllm_call(
    tensors: dict[str, torch.Tensor],
    sequence_lengths: tuple[int, ...],
) -> tuple[Callable[[], torch.Tensor], str]:
    """Build the exact production vLLM B200 FlashInfer decode call."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    from flashinfer.utils import get_device_sm_count
    from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes
    from vllm.v1.attention.backends.mla.flashinfer_mla import (
        _get_multi_ctas_kv_counter_buffer,
    )
    from vllm.v1.attention.backends.mla.flashinfer_mla import _get_workspace_buffer

    query = tensors["query"]
    block_counts = tuple(
        math.ceil(length / CACHE_BLOCK_SIZE) for length in sequence_lengths
    )
    # The persistent AOT signature uses one fixed physical capacity. FlashInfer
    # is passed the equivalent compact serving view so its own dispatch is not
    # penalized by unused page-table columns or cache blocks.
    production_kv_cache = tensors["kv_cache"][: sum(block_counts)]
    production_block_tables = tensors["block_tables"][
        :, : (max(block_counts) + 1) // 2 * 2
    ].contiguous()
    counter_bytes = get_trtllm_gen_multi_ctas_kv_counter_bytes(
        BATCH, NUM_HEADS, get_device_sm_count(query.device)
    )
    workspace = _get_workspace_buffer(return_lse=False)
    counter = _get_multi_ctas_kv_counter_buffer(counter_bytes, query.device)

    def launch() -> torch.Tensor:
        return trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=production_kv_cache,
            workspace_buffer=workspace,
            qk_nope_head_dim=KV_LORA_RANK,
            kv_lora_rank=KV_LORA_RANK,
            qk_rope_head_dim=ROPE_DIM,
            block_tables=production_block_tables,
            seq_lens=tensors["seq_lens"],
            max_seq_len=max(sequence_lengths),
            bmm1_scale=MLA_SCALE,
            bmm2_scale=1.0,
            return_lse=False,
            multi_ctas_kv_counter_buffer=counter,
        )

    return launch, "FLASHINFER_MLA"


def _assert_outputs(
    persistent: torch.Tensor,
    standalone: torch.Tensor,
    vllm: torch.Tensor,
) -> None:
    # The independently tuned launch uses eight warps for its attention root,
    # while the megakernel is faster with four.  Both retain BF16 operands and
    # FP32 dot/softmax/reduction state; the different MMA partition can round a
    # small number of final BF16 values by one ULP.
    torch.testing.assert_close(persistent, standalone, atol=1e-3, rtol=0)
    torch.testing.assert_close(persistent.float(), vllm.float(), atol=0.03, rtol=0.08)


@torch.inference_mode()
def correctness_check() -> None:
    """Validate exact scheduler isolation and production-backend agreement."""
    _require_sm100()
    if not has_vllm():
        raise RuntimeError("vLLM with FlashInfer is required for this comparison")
    dispatch_signature = None
    for _label, sequence_lengths, seed in SEQUENCE_LENGTH_CASES:
        tensors = _make_inputs(sequence_lengths, seed)
        args = _kernel_args(tensors)
        persistent = flash_mla(*args)
        current_signature = _dispatch_cache_signature(flash_mla)
        if dispatch_signature is None:
            dispatch_signature = current_signature
        elif current_signature != dispatch_signature:
            raise AssertionError("runtime MLA metadata triggered a recompilation")
        persistent_replay = flash_mla(*args)
        standalone_call, _standalone_kernels, standalone = _make_standalone_call(
            tensors
        )
        vllm_call, _backend = _make_vllm_call(tensors, sequence_lengths)
        vllm = vllm_call()
        torch.cuda.synchronize()
        torch.testing.assert_close(persistent_replay, persistent, atol=0, rtol=0)
        _assert_outputs(persistent, standalone, vllm)


@torch.inference_mode()
def main(verbose: bool = True) -> dict:
    """Benchmark persistent, root-matched Helion, and production vLLM."""
    _require_sm100()
    if not has_vllm():
        raise RuntimeError("vLLM with FlashInfer is required for this comparison")

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from _bench import capture_cuda_graph
    from _bench import run_sweep

    dispatch_signature = None

    def make_calls(case: tuple[str, tuple[int, ...], int]) -> tuple:
        nonlocal dispatch_signature
        label, sequence_lengths, seed = case
        tensors = _make_inputs(sequence_lengths, seed)
        args = _kernel_args(tensors)
        persistent_output = flash_mla(*args)
        current_signature = _dispatch_cache_signature(flash_mla)
        if dispatch_signature is None:
            dispatch_signature = current_signature
        elif current_signature != dispatch_signature:
            raise AssertionError("runtime MLA metadata triggered a recompilation")
        standalone_call, standalone_kernels, standalone_output = _make_standalone_call(
            tensors
        )
        vllm_call, backend = _make_vllm_call(tensors, sequence_lengths)
        vllm_output = vllm_call()
        torch.cuda.synchronize()

        # PREFER_NONE is materially faster for these large-shared-memory kernels.
        # Keep this launch-only setting in the benchmark rather than growing the
        # Helion configuration surface for a CUDA driver attribute.
        _prefer_no_cuda_cache_partition(flash_mla, tensors["query"])
        for standalone_kernel in standalone_kernels:
            _prefer_no_cuda_cache_partition(standalone_kernel, tensors["query"])

        _assert_outputs(persistent_output, standalone_output, vllm_output)

        persistent_graph, _ = capture_cuda_graph(lambda: flash_mla(*args))
        standalone_graph, _ = capture_cuda_graph(standalone_call)
        vllm_graph, _ = capture_cuda_graph(vllm_call)
        return (
            persistent_graph.replay,
            [
                ("standalone_helion", standalone_graph.replay),
                (f"vllm_auto ({backend})", vllm_graph.replay),
            ],
            (
                f"{label:>11s}  {BATCH:>5d}  {QUERY_TOKENS:>5d}  "
                f"{NUM_HEADS:>5d}  {sequence_lengths!s:>28s}"
            ),
        )

    return run_sweep(
        SEQUENCE_LENGTH_CASES,
        make_calls,
        use_cudagraph=False,
        pre_captured_cudagraph=True,
        thermal_warmup_ms=10_000,
        verbose=verbose,
        shape_header=(
            f"{'case':>11s}  {'batch':>5s}  {'query':>5s}  "
            f"{'heads':>5s}  {'contexts':>28s}"
        ),
    )


if __name__ == "__main__":
    main()
