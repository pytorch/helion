"""DeepSeek-V3 B1 NVFP4 MoE with a fused TP4 all-reduce.

The model body is derived from the single-device pretuned kernel so routing,
quantization, expert arithmetic, and intermediate numerics have one source of
truth.  This module changes only two source-level choices needed by TP4:

* split the underfilled routed-W13 reduction seven ways; and
* write the local weighted sum to symmetric memory before a coarse peer pull.

Run the benchmark with four local ranks::

    NVSHMEM_DISABLE_CUDA_VMM=1 \
      python -m torch.distributed.run --standalone --nproc-per-node=4 \
      pretuned_kernels/megakernels/deepseek_v3_moe_nvfp4_tp/deepseek_v3_moe_nvfp4_tp.py
"""

from __future__ import annotations

import inspect
import linecache
import math
import sys
import types
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from pretuned_kernels.megakernels._distributed import benchmark
from pretuned_kernels.megakernels._distributed import initialize
from pretuned_kernels.megakernels.deepseek_v3_moe_nvfp4 import (
    deepseek_v3_moe_nvfp4 as source_module,
)
import torch
import torch.distributed as dist

import helion

if TYPE_CHECKING:
    from collections.abc import Callable


WORLD_SIZE = 4
W13_SPLIT_K = 7
COMMUNICATION_N = 4096
NUM_SM_MULTIPLIER = 1
SIGNAL_PAD_BYTES = 32 * 1024


_SPLIT_W13_SOURCE = """\
    for w13_tile_slot, w13_tile_output_group, w13_tile_split in hl.tile(
        [top_k, activation_groups, w13_split_k], block_size=[1, 4, 1]
    ):
        w13_slot = w13_tile_slot.begin
        w13_split = w13_tile_split.begin
        w13_expert = topk_ids[0, w13_slot]
        w13_expert_address = w13_expert.to(torch.int64)
        w13_tma_row = (
            w13_expert * (twice_intermediate // 128)
            + w13_tile_output_group.begin // 4
        )
        w13_physical_row_index = (
            w13_tile_output_group.begin * 32 + hl.arange(128)
        ).to(torch.int64)
        w13_weight_row = (
            w13_expert_address * twice_intermediate + w13_physical_row_index
        )
        w13_accumulator = hl.zeros([128, MMA_N], dtype=torch.float32)
        for w13_local_group in hl.tile(w13_groups_per_split, block_size=32):
            w13_group_begin = (
                w13_split * w13_groups_per_split + w13_local_group.begin
            )
            w13_group_index = (
                w13_split * w13_groups_per_split + w13_local_group.index
            )
            w13_packed_index = (
                w13_group_begin * 8
                + hl.arange(w13_local_group.block_size * 8)
            ).to(torch.int64)
            w13_lhs = w13_tma[w13_tma_row, :, w13_packed_index]
            w13_lhs_scale = flat_w13_scale[
                w13_weight_row[:, None], w13_group_index[None, :]
            ]
            w13_hidden_bytes = hidden_q[0, w13_packed_index]
            w13_rhs = w13_hidden_bytes[:, None].expand(
                w13_hidden_bytes.size(0), MMA_N
            )
            w13_hidden_scale = hidden_scale_bytes[0, w13_group_index]
            w13_rhs_scale = w13_hidden_scale[None, :].expand(
                MMA_N, w13_local_group.block_size
            )
            w13_accumulator = hl.dot_scaled(
                w13_lhs,
                w13_lhs_scale,
                "e2m1",
                w13_rhs,
                w13_rhs_scale,
                "e2m1",
                acc=w13_accumulator,
                out_dtype=torch.float32,
            )
        w13_partial[
            w13_slot,
            w13_tile_output_group.begin // 4,
            w13_split,
            :,
        ] = first_mma_column(w13_accumulator)
    for w13_tile_slot, w13_tile_output_group in hl.tile(
        [top_k, activation_groups], block_size=[1, 4]
    ):
        w13_slot = w13_tile_slot.begin
        w13_expert = topk_ids[0, w13_slot]
        w13_preactivation = w13_partial[
            w13_slot,
            w13_tile_output_group.begin // 4,
            :,
            hl.arange(128),
        ].sum(dim=0).reshape(64, 2)
        w13_pair = hl.arange(2)
        w13_gate = torch.sum(
            w13_preactivation * (w13_pair[None, :] == 0), dim=-1
        )
        w13_up = torch.sum(
            w13_preactivation * (w13_pair[None, :] == 1), dim=-1
        )
        w13_first_scale = alpha1[w13_expert]
        w13_gate = w13_gate * w13_first_scale
        w13_up = w13_up * w13_first_scale
        w13_activated = w13_gate * torch.sigmoid(w13_gate) * w13_up
        w13_activated_groups = w13_activated.reshape(4, 16)
        w13_global_scale = torch.sum(activation_global_scale[:])
        w13_block_scale_f32 = (
            torch.amax(torch.abs(w13_activated_groups), dim=-1)
            * w13_global_scale
            / FP4_MAX
        )
        w13_block_scale = w13_block_scale_f32.to(torch.float8_e4m3fn)
        w13_actual_scale = w13_block_scale.to(torch.float32)
        w13_divisor = torch.where(w13_actual_scale > 0, w13_actual_scale, 1.0)
        w13_scaled = (
            w13_activated_groups * w13_global_scale / w13_divisor[:, None]
        )
        w13_nibbles = fp4_nibble(w13_scaled)
        w13_low, w13_high = hl.split(w13_nibbles.reshape(4, 8, 2))
        w13_packed = w13_low | w13_high << 4
        activation_q_groups[
            w13_slot, w13_tile_output_group, :
        ] = w13_packed.reshape(4, 8).to(torch.uint8)
        activation_scale[
            w13_slot, w13_tile_output_group
        ] = w13_block_scale
"""


def _replace_once(source: str, old: str, new: str, description: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"DeepSeek source changed around {description}")
    return source.replace(old, new, 1)


def _kernel_source(*, distributed: bool) -> str:
    source = inspect.getsource(source_module.deepseek_v3_moe_nvfp4.fn)
    source = source[source.index("def deepseek_v3_moe_nvfp4(") :]
    name = (
        "deepseek_v3_moe_nvfp4_tp" if distributed else "deepseek_v3_moe_nvfp4_tp_local"
    )
    source = _replace_once(
        source,
        "def deepseek_v3_moe_nvfp4(",
        f"def {name}(",
        "function name",
    )
    extra_parameters = (
        "    w13_split_k: int,\n"
        "    symmetric_output: torch.Tensor,\n"
        "    group_name: hl.ProcessGroupName,\n"
    )
    source = _replace_once(
        source,
        "    routed_scale: float,\n):",
        "    routed_scale: float,\n" + extra_parameters + "):",
        "function parameters",
    )
    source = _replace_once(
        source,
        "    topk_groups = hl.specialize(topk_groups)\n",
        "    topk_groups = hl.specialize(topk_groups)\n"
        "    w13_split_k = hl.specialize(w13_split_k)\n",
        "W13 split specialization",
    )
    source = _replace_once(
        source,
        "    hidden_groups = packed_hidden // 8\n",
        "    hidden_groups = packed_hidden // 8\n"
        "    assert hidden_groups % w13_split_k == 0\n",
        "W13 split divisibility",
    )
    first_loop = "    for tile_row, tile_expert in hl.tile(\n"
    source = _replace_once(
        source,
        first_loop,
        "    w13_partial = torch.empty(\n"
        "        (top_k, activation_groups // 4, w13_split_k, 128),\n"
        "        dtype=torch.float32,\n"
        "        device=hidden_q.device,\n"
        "    )\n"
        "    w13_groups_per_split = hidden_groups // w13_split_k\n" + first_loop,
        "W13 partial allocation",
    )
    w13_marker = "    for w13_tile_slot, w13_tile_output_group in hl.tile(\n"
    w13_end_marker = "    for shared_activation_tile_group in hl.tile("
    if source.count(w13_marker) != 1 or source.count(w13_end_marker) != 1:
        raise RuntimeError("DeepSeek source changed around routed W13")
    w13_begin = source.index(w13_marker)
    w13_end = source.index(w13_end_marker, w13_begin)
    source = source[:w13_begin] + _SPLIT_W13_SOURCE + source[w13_end:]
    source = _replace_once(
        source,
        "for w2_tile_group in hl.tile(w2_groups, block_size=32):",
        "for w2_tile_group in hl.tile(w2_groups, block_size=16):",
        "routed W2 reduction group",
    )
    source = _replace_once(
        source,
        "for shared_w2_tile_group in hl.tile(w2_groups, block_size=64):",
        "for shared_w2_tile_group in hl.tile(w2_groups, block_size=32):",
        "shared W2 reduction group",
    )
    source = _replace_once(
        source,
        "    output = torch.empty((1, hidden), dtype=torch.bfloat16, device=w2.device)\n",
        "    symmetric_rows, symmetric_hidden = symmetric_output.size()\n"
        "    symmetric_rows = hl.specialize(symmetric_rows)\n"
        "    symmetric_hidden = hl.specialize(symmetric_hidden)\n"
        "    hl.specialize(symmetric_output.stride())\n"
        "    assert symmetric_rows == 1 and symmetric_hidden == hidden\n"
        "    assert symmetric_output.stride(0) == hidden\n"
        "    assert symmetric_output.stride(1) == 1\n"
        "    assert symmetric_output.storage_offset() == 0\n"
        "    local_output = torch.as_strided(\n"
        "        symmetric_output, (hidden,), (1,), storage_offset=0\n"
        "    )\n"
        + (
            "    output = torch.empty(\n"
            "        (1, hidden), dtype=torch.bfloat16, device=w2.device\n"
            "    )\n"
            "    remote_outputs = torch.ops.symm_mem.get_remote_tensors(\n"
            "        local_output, group_name\n"
            "    )\n"
            if distributed
            else "    output = symmetric_output\n"
        ),
        "output allocation",
    )
    source = _replace_once(
        source,
        "        output[:, final_tile_n] = (final_routed_output + final_shared_output).to(\n",
        "        local_output[final_tile_n] = (\n"
        "            final_routed_output + final_shared_output\n"
        "        ).reshape(-1).to(\n",
        "local output store",
    )
    if not distributed:
        return source
    communication = """\
    for communication_tile_n in hl.tile(hidden, block_size=COMMUNICATION_N):
        communication_total = hl.zeros(
            [communication_tile_n], dtype=torch.float32
        )
        for remote_output in remote_outputs:
            communication_total = (
                communication_total
                + remote_output[communication_tile_n].to(torch.float32)
            )
        output[0, communication_tile_n] = communication_total.to(torch.bfloat16)
"""
    return_marker = "    return (\n"
    return _replace_once(
        source,
        return_marker,
        communication + return_marker,
        "return",
    )


def _load_function(*, distributed: bool) -> Callable[..., object]:
    source = _kernel_source(distributed=distributed)
    suffix = "distributed" if distributed else "local"
    module_name = f"_helion_deepseek_v3_moe_nvfp4_tp_{suffix}"
    filename = f"<{module_name}>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    generated = types.ModuleType(module_name)
    generated.__dict__.update(source_module.__dict__)
    generated.__dict__.update(
        {
            "__name__": module_name,
            "COMMUNICATION_N": COMMUNICATION_N,
        }
    )
    sys.modules[module_name] = generated
    exec(compile(source, filename, "exec"), generated.__dict__)
    return getattr(
        generated,
        "deepseek_v3_moe_nvfp4_tp" if distributed else "deepseek_v3_moe_nvfp4_tp_local",
    )


def _config(*, distributed: bool) -> helion.Config:
    range_num_stages = [0, 4, 0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 1, 0]
    range_multi_buffers = [
        None,
        None,
        None,
        None,
        True,
        None,
        None,
        False,
        None,
        None,
        None,
        True,
        None,
        True,
        None,
    ]
    range_flattens = [
        None,
        None,
        None,
        None,
        False,
        None,
        None,
        False,
        None,
        None,
        None,
        False,
        None,
        False,
        None,
    ]
    if distributed:
        range_num_stages.append(0)
        range_multi_buffers.append(None)
        range_flattens.append(None)
    return helion.Config(
        block_sizes=[8, 512, 32, 256, 512],
        cross_loop_pipeline="dynamic",
        host_tensor_descriptors=True,
        indexing=[
            "tensor_descriptor" if index in (41, 43, 58, 60) else "pointer"
            for index in range(79 if distributed else 76)
        ],
        maxnreg=None,
        num_sm_multiplier=NUM_SM_MULTIPLIER,
        num_stages=1,
        num_warps=4,
        pid_type="persistent_blocked",
        range_flattens=range_flattens,
        range_multi_buffers=range_multi_buffers,
        range_num_stages=range_num_stages,
    )


deepseek_v3_moe_nvfp4_tp = helion.kernel(
    _load_function(distributed=True),
    config=_config(distributed=True),
    static_shapes=False,
    backend="triton",
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)

deepseek_v3_moe_nvfp4_tp_local = helion.kernel(
    _load_function(distributed=False),
    config=_config(distributed=False),
    static_shapes=False,
    backend="triton",
)


@torch.inference_mode()
def main(verbose: bool = True) -> dict[str, Any]:
    """Run TP4 routing correctness and the paired cold-L2 comparison."""
    import torch.distributed._symmetric_memory as symm_mem

    rank, local_rank, group = initialize(
        kernel_name="deepseek_v3_moe_nvfp4_tp",
        world_size=WORLD_SIZE,
        signal_pad_bytes=SIGNAL_PAD_BYTES,
    )
    shape = source_module.Shape(intermediate=2048 // WORLD_SIZE)
    tensors = source_module._allocate(shape)
    symmetric_output = symm_mem.empty(
        (shape.batch, shape.hidden),
        dtype=torch.bfloat16,
        device=torch.device("cuda", local_rank),
    )
    symm_mem.rendezvous(symmetric_output, group.group_name)
    # Model each rank as a distinct TP shard while preserving replicated input
    # and routing.  Scaling the local output factors is enough to make peer
    # mixups observable without changing the packed NVFP4 layouts.
    rank_output_scale = 1.0 + rank / 8.0
    tensors["alpha2"].mul_(rank_output_scale)
    tensors["shared_alpha2"].mul_(rank_output_scale)
    base_args = source_module._kernel_args(tensors, shape)
    local_args = (*base_args, W13_SPLIT_K, symmetric_output, group.group_name)
    raw_vllm_call, backend, routing_replay = source_module._make_vllm_call(
        tensors, shape
    )
    vllm_call = cast("Callable[[], torch.Tensor]", raw_vllm_call)
    case_metrics: dict[str, dict[str, tuple[float, float]]] = {}
    dispatch_signature = None

    for label, selected_ids in source_module.ROUTING_CASES:
        source_module._set_routing_case(tensors, selected_ids)

        def persistent() -> tuple[torch.Tensor, ...]:
            return cast(
                "tuple[torch.Tensor, ...]",
                deepseek_v3_moe_nvfp4_tp(*local_args),
            )

        def standalone() -> torch.Tensor:
            deepseek_v3_moe_nvfp4_tp_local(*local_args)
            return torch.ops.symm_mem.one_shot_all_reduce(
                symmetric_output, "sum", group.group_name
            )

        def production(call: Callable[[], torch.Tensor] = vllm_call) -> torch.Tensor:
            local = call()
            symmetric_output.copy_(local)
            return torch.ops.symm_mem.one_shot_all_reduce(
                symmetric_output, "sum", group.group_name
            )

        local = cast(
            "tuple[torch.Tensor, ...]",
            deepseek_v3_moe_nvfp4_tp_local(*local_args),
        )
        vllm_local = vllm_call()
        source_module._validate_vllm(local, vllm_local, routing_replay)
        helion_expected = local[0].clone()
        vllm_expected = vllm_local.clone()
        gathered_local = [torch.empty_like(helion_expected) for _ in range(WORLD_SIZE)]
        dist.all_gather(gathered_local, helion_expected)
        if any(
            torch.equal(left, right)
            for index, left in enumerate(gathered_local)
            for right in gathered_local[index + 1 :]
        ):
            raise AssertionError("rank-local MoE outputs must differ")
        dist.all_reduce(helion_expected)
        dist.all_reduce(vllm_expected)
        actual = persistent()
        current_signature = (
            source_module._dispatch_cache_signature(deepseek_v3_moe_nvfp4_tp),
            source_module._dispatch_cache_signature(deepseek_v3_moe_nvfp4_tp_local),
        )
        if dispatch_signature is None:
            dispatch_signature = current_signature
        elif current_signature != dispatch_signature:
            raise AssertionError("runtime routing triggered a recompilation")
        standalone_actual = standalone()
        production_actual = production()
        torch.cuda.synchronize()
        dist.barrier()
        torch.testing.assert_close(actual[0], helion_expected, rtol=2e-2, atol=6.25e-2)
        torch.testing.assert_close(
            standalone_actual, helion_expected, rtol=2e-2, atol=6.25e-2
        )
        torch.testing.assert_close(
            production_actual, vllm_expected, rtol=3e-2, atol=1.25e-1
        )
        aggregate_error = source_module._similarity_error(actual[0], production_actual)
        if not math.isfinite(aggregate_error) or aggregate_error > 1e-3:
            raise AssertionError(
                f"distributed vLLM similarity error {aggregate_error:.6g} exceeds 1e-3"
            )
        for actual_value, local_value in zip(actual[1:], local[1:], strict=True):
            torch.testing.assert_close(actual_value, local_value, rtol=0, atol=0)

        launches: dict[str, Callable[[], object]] = {
            "distributed_helion": persistent,
            "standalone_helion_one_shot": standalone,
            f"vllm_{backend}_one_shot": production,
        }

        def validate_replay(
            name: str,
            value: object,
            helion_expected: torch.Tensor = helion_expected,
            vllm_expected: torch.Tensor = vllm_expected,
        ) -> None:
            output = (
                cast("tuple[torch.Tensor, ...]", value)[0]
                if name == "distributed_helion"
                else cast("torch.Tensor", value)
            )
            expected = vllm_expected if name.startswith("vllm_") else helion_expected
            torch.testing.assert_close(output, expected, rtol=3e-2, atol=1.25e-1)

        results = benchmark(launches, validate=validate_replay)
        case_metrics[label] = results
        if verbose and rank == 0:
            print(f"routing={label} experts={selected_ids}")
            for name, (warm, cold) in results.items():
                print(f"{name:>36s}: {warm:7.2f} us warm, {cold:7.2f} us cold L2")

    speedups = []
    for results in case_metrics.values():
        persistent_cold = results["distributed_helion"][1]
        production_name = next(name for name in results if name.startswith("vllm_"))
        speedups.append(results[production_name][1] / persistent_cold)
    metrics = {
        "helion_wins": sum(speedup > 1 for speedup in speedups),
        "total": len(speedups),
        "geomean": math.prod(speedups) ** (1.0 / len(speedups)),
        "best_speedup": max(speedups),
        "results": case_metrics,
    }
    dist.destroy_process_group()
    return metrics


if __name__ == "__main__":
    main()
