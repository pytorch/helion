"""B1 TP4 projection followed by a compiler-scheduled all-reduce.

Run this benchmark with four local ranks::

    NVSHMEM_DISABLE_CUDA_VMM=1 \
      python -m torch.distributed.run --standalone --nproc-per-node=4 \
      pretuned_kernels/megakernels/distributed_tp_gemm/distributed_tp_gemm.py

The compared Helion baseline has the same split-K producer and merge roots,
then calls PyTorch symmetric memory's production one-shot all-reduce.  The
``torch.mm`` baseline uses the same collective and represents the conventional
library boundary.
"""

from __future__ import annotations

from typing import Any

from pretuned_kernels.megakernels._distributed import benchmark
from pretuned_kernels.megakernels._distributed import initialize
import torch
import torch.distributed as dist

import helion
import helion.language as hl

WORLD_SIZE = 4
M = 1
N = 7168
K_PER_RANK = 4608
SPLIT_K = 12
PRODUCER_N = 128
PRODUCER_K = 64
MERGE_N = 128
COMMUNICATION_N = 2048
NUM_SM_MULTIPLIER = 2
SIGNAL_PAD_BYTES = 32 * 1024


@helion.kernel(
    config=helion.Config(
        block_sizes=[1, PRODUCER_N, PRODUCER_K, MERGE_N, COMMUNICATION_N],
        cross_loop_pipeline="dynamic",
        num_sm_multiplier=NUM_SM_MULTIPLIER,
        num_warps=4,
        pid_type="persistent_blocked",
        range_num_stages=[0, 2, 0, 0],
    ),
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def distributed_tp_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    symmetric_output: torch.Tensor,
    split_k: int,
    group_name: hl.ProcessGroupName,
) -> torch.Tensor:
    """Compute a row-parallel projection and pull ready peer chunks."""
    split_k = hl.specialize(split_k)
    rows, reduction = x.size()
    output_features = weight.size(1)
    assert rows == 1 and reduction % split_k == 0
    reduction_per_split = reduction // split_k
    partial = torch.empty(
        (split_k, output_features), dtype=torch.float32, device=x.device
    )
    output = torch.empty((rows, output_features), dtype=x.dtype, device=x.device)
    remote_outputs = torch.ops.symm_mem.get_remote_tensors(symmetric_output, group_name)

    for tile_split, tile_n in hl.tile([split_k, output_features]):
        accumulator = hl.zeros([1, tile_n], dtype=torch.float32)
        for local_k in hl.tile(reduction_per_split):
            tile_k = tile_split.begin * reduction_per_split + local_k.index
            accumulator = torch.addmm(
                accumulator,
                x[:, tile_k],
                weight[tile_k, tile_n],
            )
        partial[tile_split.begin, tile_n] = torch.sum(accumulator, dim=0)

    for tile_n in hl.tile(output_features):
        symmetric_output[0, tile_n] = partial[:, tile_n].sum(dim=0).to(x.dtype)

    for tile_n in hl.tile(output_features):
        total = hl.zeros([tile_n], dtype=torch.float32)
        for remote_output in remote_outputs:
            total = total + remote_output[0, tile_n].to(torch.float32)
        output[0, tile_n] = total.to(x.dtype)
    return output


@helion.kernel(
    config=helion.Config(
        block_sizes=[1, PRODUCER_N, PRODUCER_K, MERGE_N],
        cross_loop_pipeline="dynamic",
        num_sm_multiplier=NUM_SM_MULTIPLIER,
        num_warps=4,
        pid_type="persistent_blocked",
        range_num_stages=[0, 2, 0],
    ),
    static_shapes=True,
)
def standalone_tp_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    split_k: int,
    group_name: hl.ProcessGroupName,
) -> torch.Tensor:
    """Matched split-K projection and merge without communication fusion."""
    split_k = hl.specialize(split_k)
    rows, reduction = x.size()
    output_features = weight.size(1)
    assert rows == 1 and reduction % split_k == 0
    reduction_per_split = reduction // split_k
    partial = torch.empty(
        (split_k, output_features), dtype=torch.float32, device=x.device
    )

    for tile_split, tile_n in hl.tile([split_k, output_features]):
        accumulator = hl.zeros([1, tile_n], dtype=torch.float32)
        for local_k in hl.tile(reduction_per_split):
            tile_k = tile_split.begin * reduction_per_split + local_k.index
            accumulator = torch.addmm(
                accumulator,
                x[:, tile_k],
                weight[tile_k, tile_n],
            )
        partial[tile_split.begin, tile_n] = torch.sum(accumulator, dim=0)

    for tile_n in hl.tile(output_features):
        output[0, tile_n] = partial[:, tile_n].sum(dim=0).to(x.dtype)
    return output


@torch.inference_mode()
def main(verbose: bool = True) -> dict[str, Any]:
    """Run the matched four-rank correctness and cold-L2 comparison."""
    import torch.distributed._symmetric_memory as symm_mem

    rank, _local_rank, group = initialize(
        kernel_name="distributed_tp_gemm",
        world_size=WORLD_SIZE,
        signal_pad_bytes=SIGNAL_PAD_BYTES,
    )
    torch.manual_seed(0)
    full_x = torch.randn(
        (M, K_PER_RANK * WORLD_SIZE), device="cuda", dtype=torch.bfloat16
    )
    full_weight = (
        torch.randn((K_PER_RANK * WORLD_SIZE, N), device="cuda", dtype=torch.bfloat16)
        / 16
    )
    x = full_x[:, rank * K_PER_RANK : (rank + 1) * K_PER_RANK].contiguous()
    weight = full_weight[rank * K_PER_RANK : (rank + 1) * K_PER_RANK].contiguous()
    symmetric_output = symm_mem.empty(
        (M, N), dtype=torch.bfloat16, device=torch.device("cuda")
    )
    symm_mem.rendezvous(symmetric_output, group.group_name)

    def persistent() -> torch.Tensor:
        return distributed_tp_gemm(
            x,
            weight,
            symmetric_output,
            SPLIT_K,
            group.group_name,
        )

    def standalone() -> torch.Tensor:
        standalone_tp_gemm(
            x,
            weight,
            symmetric_output,
            SPLIT_K,
            group.group_name,
        )
        return torch.ops.symm_mem.one_shot_all_reduce(
            symmetric_output, "sum", group.group_name
        )

    def production() -> torch.Tensor:
        torch.mm(x, weight, out=symmetric_output)
        return torch.ops.symm_mem.one_shot_all_reduce(
            symmetric_output, "sum", group.group_name
        )

    launches = {
        "distributed_helion": persistent,
        "standalone_helion_one_shot": standalone,
        "torch_mm_one_shot": production,
    }
    reference = (full_x.float() @ full_weight.float()).to(torch.bfloat16)

    def validate(_name: str, actual: torch.Tensor) -> None:
        torch.testing.assert_close(actual, reference, rtol=2e-2, atol=0.25)

    for name, launch in launches.items():
        validate(name, launch())
        torch.cuda.synchronize()
        dist.barrier()
        if verbose and rank == 0:
            print(f"correctness: {name}", flush=True)

    results = benchmark(launches, validate=validate)
    if verbose and rank == 0:
        for name, (warm, cold) in results.items():
            print(f"{name:>28s}: {warm:7.2f} us warm, {cold:7.2f} us cold L2")
    production_cold = results["torch_mm_one_shot"][1]
    persistent_cold = results["distributed_helion"][1]
    metrics = {
        "helion_wins": int(persistent_cold < production_cold),
        "total": 1,
        "geomean": production_cold / persistent_cold,
        "best_speedup": production_cold / persistent_cold,
        "results": results,
    }
    dist.destroy_process_group()
    return metrics


if __name__ == "__main__":
    main()
