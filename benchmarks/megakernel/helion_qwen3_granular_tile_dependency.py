# ruff: noqa: ANN001, ANN003, ANN202, A002
"""Qwen3 TileDependency probe with source-visible prefix and merge tiles.

This file deliberately does not rewrite a compiled tile body.  The same
Helion functions below are used as the source of the separate kernels and are
inlined, statement-for-statement, into the whole-layer graph.  Their finer
top-level loops expose the task decomposition used by the successful Triton
SM-overlap probe without teaching the compiler about RMSNorm or attention.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
from pathlib import Path
import sys
import types

import torch

import helion
from helion._compiler.program_id import ForEachProgramID
import helion.language as hl

FP8_MAX = 448.0
FP8_MIN = -448.0
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)
_ACTIVE_TILE_DEPENDENCY_SCHEDULE = helion.TileDependencySchedule()


@helion.kernel(static_shapes=True, autotune_effort="none")
def tiled_rms_norm_per_block_quant(
    result: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    scale_ub: torch.Tensor | None,
    residual: torch.Tensor | None,
    group_size: int,
    is_scale_transposed: bool,
) -> None:
    """RMS/quant expressed as source-visible producer, finalize, consumer tiles."""
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)
    hl.specialize(group_size)
    groups_per_row = scale.shape[1]
    hl.specialize(groups_per_row)
    assert group_size == 128
    assert result.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32

    rms_partials = torch.empty(
        (num_tokens, groups_per_row), dtype=torch.float32, device=input.device
    )
    unrounded_values = torch.empty_like(input, dtype=torch.float32)
    inv_rms = torch.empty((num_tokens,), dtype=torch.float32, device=input.device)

    for partial_m, partial_g, partial_n in hl.tile(
        [num_tokens, groups_per_row, group_size], block_size=[1, 1, group_size]
    ):
        partial_m_idx = partial_m.begin + hl.arange(partial_m.block_size)
        partial_group_idx = partial_g.index
        partial_n_idx = (
            partial_group_idx[:, None] * group_size + partial_n.index[None, :]
        )
        partial_m_blk = partial_m_idx[:, None, None]
        partial_n_blk = partial_n_idx[None, :, :]
        partial_values = input[partial_m_blk, partial_n_blk].to(torch.float32)
        if residual is not None:
            partial_values = partial_values + residual[partial_m_blk, partial_n_blk]
            residual[partial_m_blk, partial_n_blk] = partial_values.to(residual.dtype)
        unrounded_values[partial_m_blk, partial_n_blk] = partial_values
        rms_partials[partial_m, partial_g] = torch.sum(
            partial_values * partial_values, dim=-1
        )

    for reduce_m in hl.tile(num_tokens, block_size=1):
        square_sum = hl.zeros([reduce_m], dtype=torch.float32)
        for reduce_g in hl.tile(groups_per_row, block_size=1):
            square_sum = square_sum + torch.sum(
                rms_partials[reduce_m, reduce_g], dim=-1
            )
        inv_rms[reduce_m] = torch.rsqrt(square_sum * (1.0 / hidden_size) + epsilon)

    for quant_m, quant_g, quant_n in hl.tile(
        [num_tokens, groups_per_row, group_size], block_size=[1, 1, group_size]
    ):
        quant_m_idx = quant_m.begin + hl.arange(quant_m.block_size)
        quant_group_idx = quant_g.index
        quant_n_idx = quant_group_idx[:, None] * group_size + quant_n.index[None, :]
        quant_m_blk = quant_m_idx[:, None, None]
        quant_n_blk = quant_n_idx[None, :, :]
        quant_values = unrounded_values[quant_m_blk, quant_n_blk]
        normalized = (quant_values * inv_rms[quant_m][:, None, None]).to(
            torch.bfloat16
        ) * weight[quant_n_blk]
        quant_scale = torch.amax(torch.abs(normalized), dim=-1).to(torch.float32)
        if scale_ub is not None:
            quant_scale = quant_scale.clamp(max=hl.load(scale_ub, []))
        quant_scale = (quant_scale / FP8_MAX).clamp(min=FP8_MIN_SCALE)
        scale[quant_m, quant_g] = quant_scale
        result[quant_m_blk, quant_n_blk] = (
            (normalized / quant_scale[:, :, None])
            .clamp(FP8_MIN, FP8_MAX)
            .to(result.dtype)
        )


@helion.kernel(static_shapes=True, autotune_effort="none")
def tiled_reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Expose each disjoint K/V cache half as one source-visible tile."""
    num_tokens, num_kv_heads, head_dim = key.shape
    hl.specialize(num_kv_heads)
    hl.specialize(head_dim)
    hl.specialize(block_size)
    for tile_t, tile_h, tile_d in hl.tile(
        [num_tokens, 2 * num_kv_heads, head_dim],
        block_size=[1, 1, head_dim],
    ):
        token = tile_t.index
        combined_head = tile_h.index
        dimension = tile_d.index
        cache_half = combined_head // num_kv_heads
        cache_head = combined_head % num_kv_heads
        key_value = key[
            token[:, None, None],
            cache_head[None, :, None],
            dimension[None, None, :],
        ]
        value_value = value[
            token[:, None, None],
            cache_head[None, :, None],
            dimension[None, None, :],
        ]
        cache_value = torch.where(
            (cache_half == 0)[None, :, None], key_value, value_value
        )
        cache_dimension = (cache_half[:, None] * head_dim + dimension[None, :])[
            None, :, :
        ]
        slot = slot_mapping[token]
        block = (slot // block_size)[:, None, None]
        offset = (slot % block_size)[:, None, None]
        hl.store(
            kv_cache,
            [
                block,
                offset,
                cache_head[None, :, None],
                cache_dimension,
            ],
            cache_value,
        )


@helion.kernel(static_shapes=True, autotune_effort="none")
def tiled_merge_attention_splits(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
) -> torch.Tensor:
    """Two source-visible merge levels matching the Triton overlap probe."""
    splits, num_kv_heads, q_per_kv, head_dim = partial_out.shape
    hl.specialize(splits)
    hl.specialize(num_kv_heads)
    hl.specialize(q_per_kv)
    hl.specialize(head_dim)
    query_heads = num_kv_heads * q_per_kv
    hl.specialize(query_heads)
    merge_chunks = 16
    hl.specialize(merge_chunks)
    assert splits % merge_chunks == 0
    splits_per_chunk = splits // merge_chunks
    hl.specialize(splits_per_chunk)

    partial_out_flat = partial_out.view(splits, query_heads, head_dim)
    partial_lse_flat = partial_lse.view(splits, query_heads)
    partial_out_storage = partial_out_flat.view(-1)
    partial_lse_storage = partial_lse_flat.view(-1)
    chunk_out = torch.empty(
        (merge_chunks, query_heads, head_dim),
        dtype=torch.float32,
        device=partial_out.device,
    )
    chunk_lse = torch.empty(
        (merge_chunks, query_heads),
        dtype=torch.float32,
        device=partial_out.device,
    )
    chunk_out_storage = chunk_out.view(-1)
    chunk_lse_storage = chunk_lse.view(-1)
    output = torch.empty(
        (query_heads, head_dim),
        dtype=torch.bfloat16,
        device=partial_out.device,
    )

    for tile_chunk, chunk_head in hl.tile(
        [merge_chunks, query_heads], block_size=[1, 1]
    ):
        chunk_split_idx = (
            tile_chunk.index[:, None] * splits_per_chunk
            + hl.arange(splits_per_chunk)[None, :]
        )
        chunk_lse_offsets = (
            chunk_split_idx[:, :, None] * query_heads + chunk_head.index[None, None, :]
        )
        chunk_lse_values = partial_lse_storage[chunk_lse_offsets]
        chunk_max_lse = torch.amax(chunk_lse_values, dim=1)
        chunk_weights = torch.exp2(chunk_lse_values - chunk_max_lse[:, None, :])
        chunk_value_offsets = (
            chunk_lse_offsets[:, :, :, None] * head_dim
            + hl.arange(head_dim)[None, None, None, :]
        )
        chunk_values = partial_out_storage[chunk_value_offsets]
        chunk_denominator = torch.sum(chunk_weights, dim=1)
        chunk_merged = torch.sum(chunk_values * chunk_weights[:, :, :, None], dim=1)
        chunk_merged = chunk_merged / chunk_denominator[:, :, None]
        chunk_out[tile_chunk, chunk_head, :] = chunk_merged
        chunk_lse[tile_chunk, chunk_head] = chunk_max_lse + torch.log2(
            chunk_denominator
        )

    for final_head in hl.tile(query_heads, block_size=1):
        final_chunk_idx = hl.arange(merge_chunks)
        final_lse_offsets = (
            final_chunk_idx[:, None] * query_heads + final_head.index[None, :]
        )
        final_lse_values = chunk_lse_storage[final_lse_offsets]
        final_max_lse = torch.amax(final_lse_values, dim=0)
        final_weights = torch.exp2(final_lse_values - final_max_lse[None, :])
        final_value_offsets = (
            final_lse_offsets[:, :, None] * head_dim
            + hl.arange(head_dim)[None, None, :]
        )
        final_values = chunk_out_storage[final_value_offsets]
        final_denominator = torch.sum(final_weights, dim=0)
        final_merged = torch.sum(final_values * final_weights[:, :, None], dim=0)
        output[final_head, :] = (final_merged / final_denominator[:, None]).to(
            output.dtype
        )
    return output.view(1, num_kv_heads * q_per_kv, head_dim)


def _compile_granular_separate_kernel(kernel, kernel_args, args):
    """Compile an unchanged granular source body as its own Helion launch."""
    scheduled = helion.kernel(
        static_shapes=True,
        autotune_effort="none",
        tile_dependency_schedule=_ACTIVE_TILE_DEPENDENCY_SCHEDULE,
    )(kernel.fn)
    bound = scheduled.bind(kernel_args)
    values = dict(bound.config_spec.default_config())
    values.update(
        {
            "num_warps": 1,
            "num_stages": args.kernel_stages,
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": args.worker_multiplier,
        }
    )
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return config, bound.compile_config(config)


def _build_helion_reference(args, tensors):
    """Build a separate-launch graph from the exact sources used above."""
    from helion_qwen3_layer_baseline import FFN_CONFIGS
    from helion_qwen3_layer_baseline import block_fp8_mm
    from helion_qwen3_layer_baseline import compile_config
    from helion_qwen3_layer_baseline import fused_qk_norm_rope
    from helion_qwen3_layer_baseline import paged_gqa_decode_attention_split
    from helion_qwen3_layer_baseline import per_token_group_fp8_quant
    from helion_qwen3_layer_baseline import silu_and_mul_per_block_quant

    configs = json.loads(Path(args.config_path).read_text())
    initial_residual = tensors["residual"].clone()
    initial_kv_cache = tensors["kv_cache"].clone()
    rms_args = (
        tensors["pre_q"],
        tensors["hidden_states"],
        tensors["pre_weight"],
        tensors["pre_scale"],
        args.eps,
        None,
        tensors["residual"],
        args.group,
        False,
    )
    _, rms = _compile_granular_separate_kernel(
        tiled_rms_norm_per_block_quant, rms_args, args
    )
    rms(*rms_args)
    qkv_args = (
        tensors["pre_q"],
        tensors["pre_scale"],
        tensors["qkv_weight_q"],
        tensors["qkv_weight_scale"],
        args.group,
    )
    _, qkv_mm = compile_config(block_fp8_mm, qkv_args, configs["qkv_mm"])
    qkv = qkv_mm(*qkv_args)
    qk_args = (
        qkv,
        args.q_heads,
        args.kv_heads,
        args.kv_heads,
        args.head_dim,
        args.eps,
        tensors["q_weight"],
        tensors["k_weight"],
        tensors["cos_sin"],
        True,
        tensors["position"],
        -1,
    )
    _, qk = compile_config(fused_qk_norm_rope, qk_args, configs["qk_norm_rope"])
    qk(*qk_args)
    key_begin = args.q_heads * args.head_dim
    qkv_width = (args.q_heads + 2 * args.kv_heads) * args.head_dim
    query = qkv[:, :key_begin].view(1, args.q_heads, args.head_dim)
    key = qkv[:, key_begin : key_begin + args.kv_heads * args.head_dim].view(
        1, args.kv_heads, args.head_dim
    )
    value = qkv[:, key_begin + args.kv_heads * args.head_dim : qkv_width].view(
        1, args.kv_heads, args.head_dim
    )
    cache_args = (
        key,
        value,
        tensors["kv_cache"],
        tensors["slot_mapping"],
        args.block_size,
    )
    _, cache = _compile_granular_separate_kernel(
        tiled_reshape_and_cache_flash, cache_args, args
    )
    cache(*cache_args)
    split_args = (
        query,
        tensors["kv_cache"],
        tensors["block_table"],
        args.context,
        args.block_size,
        args.q_heads // args.kv_heads,
        args.attention_splits,
    )
    _, split_kernel = compile_config(
        paged_gqa_decode_attention_split,
        split_args,
        configs["decode_attention_split"],
    )
    partial_out, partial_lse = split_kernel(*split_args)
    merge_args = (partial_out, partial_lse)
    _, merge = _compile_granular_separate_kernel(
        tiled_merge_attention_splits, merge_args, args
    )
    attention = merge(*merge_args)
    quant_args = (
        attention.view(1, args.hidden),
        tensors["attention_q"],
        tensors["attention_scale"],
        args.group,
        1e-10,
        -448.0,
        448.0,
        False,
        False,
        False,
    )
    _, attention_quant = compile_config(
        per_token_group_fp8_quant,
        quant_args,
        configs["attention_quant"],
    )
    attention_quant(*quant_args)
    o_args = (
        tensors["attention_q"],
        tensors["attention_scale"],
        tensors["o_weight_q"],
        tensors["o_weight_scale"],
        args.group,
    )
    _, o_mm = compile_config(block_fp8_mm, o_args, configs["o_mm"])
    attention_out = o_mm(*o_args)
    post_args = (
        tensors["ffn_q"],
        attention_out,
        tensors["post_weight"],
        tensors["ffn_scale"],
        args.eps,
        None,
        tensors["residual"],
        args.group,
        False,
    )
    rms(*post_args)
    w13_args = (
        tensors["ffn_q"],
        tensors["ffn_scale"],
        tensors["w13_q"],
        tensors["w13_scale"],
        args.group,
    )
    _, w13 = compile_config(block_fp8_mm, w13_args, FFN_CONFIGS["w13"])
    gate_up = w13(*w13_args)
    silu_args = (gate_up, args.group)
    _, silu = compile_config(
        silu_and_mul_per_block_quant,
        silu_args,
        FFN_CONFIGS["silu_quant"],
    )
    activation_q, activation_scale = silu(*silu_args)
    w2_args = (
        activation_q,
        activation_scale,
        tensors["w2_q"],
        tensors["w2_scale"],
        args.group,
    )
    _, w2 = compile_config(block_fp8_mm, w2_args, FFN_CONFIGS["w2"])
    output = w2(*w2_args)

    # Compilation materializes each downstream tensor by executing its kernel.
    # Restore the two externally mutable inputs so the caller's first launch is
    # iteration one, exactly like the persistent path used for validation.
    tensors["residual"].copy_(initial_residual)
    tensors["kv_cache"].copy_(initial_kv_cache)

    def launch():
        rms(*rms_args)
        local_qkv = qkv_mm(*qkv_args)
        qk(local_qkv, *qk_args[1:])
        local_query = local_qkv[:, :key_begin].view(1, args.q_heads, args.head_dim)
        local_key = local_qkv[
            :, key_begin : key_begin + args.kv_heads * args.head_dim
        ].view(1, args.kv_heads, args.head_dim)
        local_value = local_qkv[
            :, key_begin + args.kv_heads * args.head_dim : qkv_width
        ].view(1, args.kv_heads, args.head_dim)
        cache(
            local_key,
            local_value,
            tensors["kv_cache"],
            tensors["slot_mapping"],
            args.block_size,
        )
        local_partials, local_lse = split_kernel(
            local_query,
            tensors["kv_cache"],
            tensors["block_table"],
            args.context,
            args.block_size,
            args.q_heads // args.kv_heads,
            args.attention_splits,
        )
        local_attention = merge(local_partials, local_lse)
        attention_quant(
            local_attention.view(1, args.hidden),
            tensors["attention_q"],
            tensors["attention_scale"],
            args.group,
            1e-10,
            -448.0,
            448.0,
            False,
            False,
            False,
        )
        local_attention_out = o_mm(*o_args)
        rms(
            tensors["ffn_q"],
            local_attention_out,
            tensors["post_weight"],
            tensors["ffn_scale"],
            args.eps,
            None,
            tensors["residual"],
            args.group,
            False,
        )
        local_gate = w13(*w13_args)
        local_activation_q, local_activation_scale = silu(local_gate, args.group)
        local_output = w2(
            local_activation_q,
            local_activation_scale,
            tensors["w2_q"],
            tensors["w2_scale"],
            args.group,
        )
        return (
            local_output,
            local_qkv,
            local_partials,
            local_lse,
            local_attention,
            local_attention_out,
            local_gate,
            local_activation_q,
            local_activation_scale,
        )

    return launch, {
        "output": output,
        "qkv": qkv,
        "partial_out": partial_out,
        "partial_lse": partial_lse,
        "attention": attention,
        "attention_out": attention_out,
        "gate_up": gate_up,
        "activation_q": activation_q,
        "activation_scale": activation_scale,
    }


def _build_original_helion_reference(*args: object, **kwargs: object):
    from triton_qwen3_whole_layer_persistent import build_helion_reference

    return build_helion_reference(*args, **kwargs)


def _probe_config(bound, args):
    """Map the retained one-warp probe geometry onto the granular source."""
    values = dict(bound.config_spec.default_config())
    block_size_by_id = {
        9: 8,  # QKV output tile
        12: args.qk_head_block,
        19: 4,  # four queries per attention task
        21: args.attention_context_block,
        26: 1,  # one attention quant group
        29: 8,  # O output tile
        40: 16,  # W13 output tile
        45: 8,  # W2 output tile
    }
    values["block_sizes"] = [
        block_size_by_id[spec.block_id] for spec in bound.config_spec.block_sizes
    ]
    values["loop_orders"] = [
        [0, 1, 2],
        [0, 1, 2],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2],
        [2, 1, 0],
        [0, 1],
        [0, 1, 2],
        [1, 0],
        [0, 1, 2],
        [0, 1, 2],
        [0, 1],
        [0, 1],
        [0, 1],
    ]
    values["l2_groupings"] = [1] * len(bound.config_spec.l2_groupings)

    def by_block_id(specs, choices, default):
        return [
            next(
                (
                    choices[block_id]
                    for block_id in spec.block_ids
                    if block_id in choices
                ),
                default,
            )
            for spec in specs
        ]

    projection_ranges = {10: 4, 30: 4, 41: 4, 46: 4}
    values["range_num_stages"] = by_block_id(
        bound.config_spec.range_num_stages, projection_ranges, 0
    )
    values["range_unroll_factors"] = by_block_id(
        bound.config_spec.range_unroll_factors,
        {10: 2, 30: 2, 41: 2, 46: 4},
        0,
    )
    values["range_multi_buffers"] = by_block_id(
        bound.config_spec.range_multi_buffers,
        {10: True, 21: True, 30: False, 41: True, 46: False},
        None,
    )
    values["range_flattens"] = by_block_id(
        bound.config_spec.range_flattens,
        {10: False, 21: True, 30: False, 41: False, 46: True},
        None,
    )
    values.update(
        {
            "num_warps": 1,
            "num_stages": args.kernel_stages,
            "pid_type": "persistent_blocked",
            "num_sm_multiplier": args.worker_multiplier,
        }
    )
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return config


def main() -> None:
    global _ACTIVE_TILE_DEPENDENCY_SCHEDULE

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--epoch-replicas", type=int)
    parser.add_argument("--tile-dependency-stages", type=int)
    parser.add_argument("--continuation-split", type=int)
    parser.add_argument("--producer-order", choices=("physical", "consumer_major"))
    parser.add_argument("--strict-validation", action="store_true")
    parser.add_argument("--no-waits", action="store_true")
    parser.add_argument("--skip-wait-prefix", action="append", default=[])
    parser.add_argument("--dump-accesses", action="store_true")
    parser.add_argument(
        "--reference",
        choices=("same_source", "tuned"),
        default="same_source",
    )
    args, remaining = parser.parse_known_args()

    if args.no_waits:
        ForEachProgramID._wait_for_counter = staticmethod(
            lambda **_kwargs: [ast.Pass()]
        )
    elif args.skip_wait_prefix:
        original_wait_for_counter = ForEachProgramID._wait_for_counter

        def filtered_wait_for_counter(**kwargs):
            if any(
                kwargs["prefix"].startswith(prefix) for prefix in args.skip_wait_prefix
            ):
                return [ast.Pass()]
            return original_wait_for_counter(**kwargs)

        ForEachProgramID._wait_for_counter = staticmethod(filtered_wait_for_counter)

    compatibility = types.ModuleType("triton_qwen3_sm_overlap_probe")
    compatibility.build_helion_reference = (
        _build_helion_reference
        if args.reference == "same_source"
        else _build_original_helion_reference
    )
    sys.modules.setdefault("triton_qwen3_sm_overlap_probe", compatibility)

    import helion_qwen3_tile_dependency as probe

    probe.rms_norm_per_block_quant = tiled_rms_norm_per_block_quant
    probe.reshape_and_cache_flash = tiled_reshape_and_cache_flash
    probe.merge_attention_splits = tiled_merge_attention_splits
    probe._probe_matched_config = _probe_config
    kernel, source = probe._build_composite_kernel()
    probe.GENERATED_SOURCE = source
    _ACTIVE_TILE_DEPENDENCY_SCHEDULE = helion.TileDependencySchedule(
        epoch_replicas=args.epoch_replicas,
        tile_dependency_stages=args.tile_dependency_stages,
        continuation_split=args.continuation_split,
        producer_order=args.producer_order,
    )
    probe.qwen3_layer_tile_dependency = helion.kernel(
        static_shapes=True,
        autotune_effort="none",
        tile_dependency_schedule=_ACTIVE_TILE_DEPENDENCY_SCHEDULE,
    )(kernel.fn)
    sys.argv = [sys.argv[0], *remaining]
    if args.dump_accesses:

        def dump_run(probe_args) -> None:
            tensors = probe.allocate_layer(probe_args)
            bound = probe.qwen3_layer_tile_dependency.bind(
                probe._composite_args(tensors, probe_args)
            )
            host_function = bound.host_function
            assert host_function is not None
            for root, accesses in enumerate(host_function.device_ir.tile_accesses):
                print(
                    "TILE_ACCESSES",
                    root,
                    [dataclasses.asdict(access) for access in accesses],
                    flush=True,
                )

        probe.run = dump_run
        probe.main()
        return
    if not args.strict_validation:
        probe.main()
        return

    allocations: list[dict[str, torch.Tensor]] = []
    original_allocate_layer = probe.allocate_layer
    original_assert_close = torch.testing.assert_close
    validation_names = iter(
        (
            "output",
            "qkv",
            "partial_out",
            "partial_lse",
            "attention",
            "attention_out",
            "gate_up",
            "activation_q",
            "activation_scale",
        )
    )

    def tracked_allocate_layer(namespace):
        tensors = original_allocate_layer(namespace)
        allocations.append(tensors)
        return tensors

    def report(name: str, actual: object, expected: object) -> None:
        actual_tensor = torch.as_tensor(actual)
        expected_tensor = torch.as_tensor(expected)
        if actual_tensor.numel() == expected_tensor.numel():
            actual_tensor = actual_tensor.view_as(expected_tensor)
        difference = (actual_tensor.float() - expected_tensor.float()).abs()
        print(
            "STRICT_VALIDATION",
            name,
            {
                "exact": bool(torch.equal(actual_tensor, expected_tensor)),
                "mismatches": int((actual_tensor != expected_tensor).sum().item()),
                "elements": actual_tensor.numel(),
                "max_abs": float(difference.max().item()),
                "mean_abs": float(difference.mean().item()),
            },
            flush=True,
        )

    checked_bridge_state = False

    def diagnostic_assert_close(actual, expected, **kwargs: object) -> None:
        nonlocal checked_bridge_state
        if not checked_bridge_state and len(allocations) >= 2:
            checked_bridge_state = True
            persistent_tensors, reference_tensors = allocations[:2]
            for bridge_name in (
                "pre_q",
                "pre_scale",
                "kv_cache",
                "attention_q",
                "attention_scale",
                "ffn_q",
                "ffn_scale",
                "residual",
            ):
                report(
                    bridge_name,
                    persistent_tensors[bridge_name],
                    reference_tensors[bridge_name],
                )
        report(next(validation_names, "unexpected"), actual, expected)
        original_assert_close(actual, expected, **kwargs)

    probe.allocate_layer = tracked_allocate_layer
    torch.testing.assert_close = diagnostic_assert_close
    try:
        probe.main()
    finally:
        probe.allocate_layer = original_allocate_layer
        torch.testing.assert_close = original_assert_close


if __name__ == "__main__":
    main()
