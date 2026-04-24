"""
All-Gather Matrix Multiplication Example
========================================
This example demonstrates how to implement an all-gather operation followed by matrix multiplication
using Helion and PyTorch's distributed capabilities. It includes progress tracking using symmetric memory
and a Helion kernel optimized for multi-process runs.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import os

import torch
from torch._C._distributed_c10d import _SymmetricMemory
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl
from helion.runtime.triton_helpers import triton_wait_signal


RAW_CONFIGS_STR="""{

  "rank_0_mperrank_2048_n_2560_k_8192_splits_1": {
    "block_sizes": [
      128,
      256,
      128
    ],
    "loop_orders": [
      [
        1,
        0
      ]
    ],
    "l2_groupings": [
      1
    ],
    "range_unroll_factors": [
      1,
      4
    ],
    "range_warp_specializes": [],
    "range_num_stages": [
      0,
      2
    ],
    "range_multi_buffers": [
      false,
      true
    ],
    "range_flattens": [
      null,
      false
    ],
    "load_eviction_policies": [
      "",
      "",
      "first",
      ""
    ],
    "num_warps": 8,
    "num_stages": 3,
    "indexing": [
      "pointer",
      "pointer",
      "tensor_descriptor",
      "tensor_descriptor",
      "pointer"
    ],
    "atomic_indexing": [],
    "pid_type": "persistent_interleaved",
    "num_sm_multiplier": 16,
    "maxnreg": 256
  },"rank_1_mperrank_2048_n_2560_k_8192_splits_1": {
    "block_sizes": [
      128,
      256,
      128
    ],
    "loop_orders": [
      [
        1,
        0
      ]
    ],
    "l2_groupings": [
      1
    ],
    "range_unroll_factors": [
      1,
      4
    ],
    "range_warp_specializes": [],
    "range_num_stages": [
      0,
      2
    ],
    "range_multi_buffers": [
      false,
      true
    ],
    "range_flattens": [
      null,
      false
    ],
    "load_eviction_policies": [
      "",
      "",
      "first",
      ""
    ],
    "num_warps": 8,
    "num_stages": 3,
    "indexing": [
      "pointer",
      "pointer",
      "tensor_descriptor",
      "tensor_descriptor",
      "pointer"
    ],
    "atomic_indexing": [],
    "pid_type": "persistent_interleaved",
    "num_sm_multiplier": 16,
    "maxnreg": 256
  },
  
  "rank_2_mperrank_2048_n_2560_k_8192_splits_1": {
    "block_sizes": [
      128,
      256,
      128
    ],
    "loop_orders": [
      [
        1,
        0
      ]
    ],
    "l2_groupings": [
      1
    ],
    "range_unroll_factors": [
      1,
      4
    ],
    "range_warp_specializes": [],
    "range_num_stages": [
      0,
      2
    ],
    "range_multi_buffers": [
      false,
      true
    ],
    "range_flattens": [
      null,
      false
    ],
    "load_eviction_policies": [
      "",
      "",
      "first",
      ""
    ],
    "num_warps": 8,
    "num_stages": 3,
    "indexing": [
      "pointer",
      "pointer",
      "tensor_descriptor",
      "tensor_descriptor",
      "pointer"
    ],
    "atomic_indexing": [],
    "pid_type": "persistent_interleaved",
    "num_sm_multiplier": 16,
    "maxnreg": 256
  },
  "rank_3_mperrank_2048_n_2560_k_8192_splits_1": {
    "block_sizes": [
      128,
      256,
      128
    ],
    "loop_orders": [
      [
        1,
        0
      ]
    ],
    "l2_groupings": [
      1
    ],
    "range_unroll_factors": [
      1,
      4
    ],
    "range_warp_specializes": [],
    "range_num_stages": [
      0,
      2
    ],
    "range_multi_buffers": [
      false,
      true
    ],
    "range_flattens": [
      null,
      false
    ],
    "load_eviction_policies": [
      "",
      "",
      "first",
      ""
    ],
    "num_warps": 8,
    "num_stages": 3,
    "indexing": [
      "pointer",
      "pointer",
      "tensor_descriptor",
      "tensor_descriptor",
      "pointer"
    ],
    "atomic_indexing": [],
    "pid_type": "persistent_interleaved",
    "num_sm_multiplier": 16,
    "maxnreg": 256
  }
}"""
import json
RAW_CONFIGS = json.loads(RAW_CONFIGS_STR)

rank = int(os.environ.get("LOCAL_RANK", 0))

# Find the key that starts with "rank_{rank}_"
M = 2048*4
N = 2560
K = 8192
splits_per_rank = 1
M_per_rank = M // int(os.environ.get("WORLD_SIZE", 1))
target_key = f"rank_{rank}_mperrank_{M_per_rank}_n_{N}_k_{K}_splits_{splits_per_rank}"
print(f"Looking for config with key: {target_key}")
best_config_data = RAW_CONFIGS.get(target_key)
assert best_config_data is not None, f"Config for rank {rank} not found in RAW_CONFIGS"

kernel_config = helion.Config(
    block_sizes=best_config_data["block_sizes"],
    num_warps=best_config_data["num_warps"],
    num_stages=best_config_data["num_stages"],
    pid_type=best_config_data.get("pid_type", "persistent_interleaved"),
    maxnreg=best_config_data.get("maxnreg", 256),
    num_sm_multiplier=best_config_data.get("num_sm_multiplier", 4),
    indexing="block_ptr"
)
@triton.jit
def _wait_progress_at_idx(progress: tl.tensor, idx: int) -> None:
    triton_wait_signal(progress + idx, 1, 0, "acquire", "gpu", "ld", False)

@helion.kernel(
    config=kernel_config,
    static_shapes=True,
)
def helion_matmul_w_progress_fp8(
    a: torch.Tensor,  # [M, K] FP8 (full gathered)
    a_shared: torch.Tensor,  # [M//world_size, K] FP8
    scale_a: torch.Tensor,  # [M//world_size, 1] FP32
    b: torch.Tensor,  # [K, N] FP8 (may be non-contig)
    scale_b: torch.Tensor,  # [1, N] FP32
    progress: torch.Tensor,
    SPLITS_PER_RANK: int,
    RANK: int,
) -> torch.Tensor:
    """
    Performs matrix multiplication with FP8 tensors and tracks progress using Helion.
    """
    M, K = a.size()
    K2, N = b.size()
    assert K2 == K, f"size mismatch {K2} != {K}"
    out = torch.empty(
        [M, N], dtype=torch.bfloat16, device=a.device
    )  # Output buffered as BF16 for performance.
    M_per_rank = a_shared.size(0)

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)  # Initialize accumulator in FP32.
        # Once the progress is filled, we can start doing gemm
        hl.triton_kernel(
            _wait_progress_at_idx,
            args=(
                progress,
                tile_m.begin // (M_per_rank // SPLITS_PER_RANK),
            ),
            output_like=None,
        )
        # Load scales once per tile
        sa = scale_a[tile_m, :] # [tile_m, 1]
        sb = scale_b[:, tile_n]  # [1, tile_n]

        for tile_k in hl.tile(K):
            x_tile = a[tile_m, tile_k] 
            y_tile = b[tile_k, tile_n]
            acc = hl.dot(x_tile, y_tile, acc=acc) 

        # Convert result back to bfloat16
        out[tile_m, tile_n] = (acc * sa * sb).to(torch.bfloat16)

    return out


def copy_engine_all_gather_w_progress(
    output: torch.Tensor,
    inp: torch.Tensor,  # Must be symmetric tensor
    progress: torch.Tensor,
    group_name: ProcessGroup,  
    splits_per_rank: int,
    backend_stream: torch.cuda.Stream | None = None,
) -> torch.cuda.Stream:
    """
    Performs an all-gather operation with progress tracking using symmetric memory.

    - Each rank builds its full output tensor by copying data from all other ranks.
    - Data can be split into smaller chunks (splits_per_rank) for finer-grained progress.
    - The 'progress' 1D tensor signals which splits are ready (1 = ready).
    - GEMM can start operating on a split immediately once its progress flag is set,

    Example (world_size=4, splits_per_rank=2):

        Rank 0: inp0 (8 rows)
        Rank 1: inp1 (8 rows)
        Rank 2: inp2 (8 rows)
        Rank 3: inp3 (8 rows)

        Splits per rank: [A0 A1 | A2 A3], etc.

        Copy order (round-robin) for rank 0:
            Step 0: Copy all splits from rank 1 → output positions [B0 B1 | B2 B3]
            Step 1: Copy all splits from rank 2 → output positions [C0 C1 | C2 C3]
            Step 2: Copy all splits from rank 3 → output positions [D0 D1 | D2 D3]
            Step 3: Copy all splits from rank 0 → output positions [A0 A1 | A2 A3]

        After these steps, rank 0 has the full gathered tensor:
            [A0 A1 | A2 A3 | B0 B1 | B2 B3 | C0 C1 | C2 C3 | D0 D1 | D2 D3]

    Note:
    - This is a partial pipeline: GEMM starts per split as soon as it’s ready.
    - Full pipelined GEMM (all-gather + GEMM fused in Helion kernel) is future work
        (see https://github.com/pytorch/helion/pull/1532), which would eliminate extra
        copies, reduce kernel launch overhead, and maximize overlap of communication
        and computation.
    """
    backend_stream = dist._symmetric_memory._get_backend_stream(priority=-1)
    assert inp.is_contiguous(), "Input tensor 'inp' must be contiguous"
    symm_mem_group = group_name

    if symm_mem_group is None:
        raise RuntimeError("No symmetric memory group available")

    symm_mem_hdl = dist._symmetric_memory.rendezvous(inp, group=symm_mem_group)
    assert symm_mem_hdl is not None, "Failed to obtain symmetric memory handle"

    rank = symm_mem_hdl.rank
    world_size = symm_mem_hdl.world_size
    assert inp.numel() % splits_per_rank == 0, "inp.numel must be divisible by splits_per_rank"
    assert progress.numel() >= world_size * splits_per_rank, "progress size is insufficient"

    output_shape = list(inp.shape)
    output_shape[0] *= world_size
    assert list(output.shape) == output_shape, "Mismatch in output shape"
    chunks = output.chunk(world_size * splits_per_rank)

    #symm_mem_hdl.barrier()
    backend_stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(backend_stream):
        for step in range(world_size):
            src_rank = (rank + step + 1) % world_size
            for split_id in range(splits_per_rank):
                src_buf = symm_mem_hdl.get_buffer(
                    src_rank, chunks[0].shape, inp.dtype, chunks[0].numel() * split_id
                )
                chunks[src_rank * splits_per_rank + split_id].copy_(src_buf,non_blocking=True)
                # Write progress signal
                symm_mem_hdl.stream_write_value32(
                    progress,
                    offset=src_rank * splits_per_rank + split_id,
                    val=1,
                )
        #symm_mem_hdl.barrier()

    return backend_stream

def _helion_all_gather_fp8_gemm_runtime(
    a_shared: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    world_size: int,
    group_name: ProcessGroup,
    a_out: torch.Tensor | None = None,
    SPLITS_PER_RANK: int = 1, 
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Performs an all-gather on a_shared and matrix multiplication using the Helion library.
    """

    symm_mem_group = group_name
    symm_mem_hdl = dist._symmetric_memory.rendezvous(a_shared, group=group_name)

    if symm_mem_hdl is None:
        a_shared_symm = dist._symmetric_memory.empty(
            a_shared.shape,
            dtype=a_shared.dtype,
            device=a_shared.device
        )
        a_shared_symm.copy_(a_shared)
        a_shared_symm._is_symmetric_memory = True

        # Try rendezvous again with the symmetric copy
        symm_mem_hdl = dist._symmetric_memory.rendezvous(a_shared_symm, group=group_name)
        if symm_mem_hdl is None:
            raise RuntimeError("Failed to get symmetric memory handle after copy")
    else:
        a_shared_symm = a_shared  # already usable

    a_shape = list(a_shared_symm.shape)
    a_shape[0] *= symm_mem_hdl.world_size
    if a_out is None:
        a_out = torch.empty(a_shape, dtype=a_shared.dtype, device=a_shared.device)
    
    progress = torch.zeros(
        symm_mem_hdl.world_size * SPLITS_PER_RANK,
        dtype=torch.uint32,
        device=a_shared_symm.device,
    )

    backend_stream = copy_engine_all_gather_w_progress(
        a_out, a_shared_symm, progress, group_name, SPLITS_PER_RANK
    )
    
    c = helion_matmul_w_progress_fp8(
        a_out,
        a_shared_symm,
        scale_a,
        b,
        scale_b,
        progress,
        SPLITS_PER_RANK=SPLITS_PER_RANK,
        RANK=symm_mem_hdl.rank,
    )
    assert type(c) is torch.Tensor
    torch.cuda.current_stream().wait_stream(backend_stream)

    return a_out, c

def helion_ag_matmul(a_shared: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, world_size: int, dist_group: ProcessGroup) -> torch.Tensor:
    """Wrapper for helion_all_gather_matmul that returns only the matmul result."""
    a_out, c = _helion_all_gather_fp8_gemm_runtime(
            a_shared,
            b,
            scale_a,
            scale_b,
            world_size,
            dist_group.group_name,
            SPLITS_PER_RANK=1,
        )
    return c


def reference_ag_matmul(a_shared: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, world_size: int, dist_group: ProcessGroup) -> torch.Tensor:
    """Reference implementation using torch.ops.symm_mem.fused_all_gather_matmul."""
    dist_group = dist.group.WORLD
    if dist_group is None:
        raise RuntimeError("No distributed group available")
    ag_golden, mm_golden = torch.ops.symm_mem.fused_all_gather_scaled_matmul(
            a_shared,
            [b],
            scale_a,
            [scale_b],
            gather_dim=0,
            biases=[None],
            result_scales=[None],
            out_dtypes=[torch.bfloat16],
            use_fast_accum=[False],
            group_name=dist_group.group_name,
        )
    return mm_golden[0]


# %%
def test(M: int, N: int, K: int, world_size: int, device: torch.device) -> None:
    """
    Tests and benchmarks helion_all_gather_matmul against PyTorch's implementation.
    Args:
        M (int): First dimension of the matrix.
        N (int): Second dimension of the matrix.
        K (int): Third dimension of the matrix.
        world_size (int): Number of processes.
        device (torch.device): Device to run the test on.
    """
    M_per_rank = M // world_size
    FP8_DTYPE = torch.float8_e4m3fn
    torch.manual_seed(41)  # deterministic for all ranks
    a_shared = torch.rand(M_per_rank, K, device=device, dtype=torch.bfloat16) * 0.05
    a_shared = a_shared.to(FP8_DTYPE)
    b = (torch.rand(K, N, device=device, dtype=torch.bfloat16) *0.1+ 0.05).T.contiguous().T
    b = b.to(FP8_DTYPE)

    scale_a = torch.rand((M_per_rank, 1), device=device, dtype=torch.float32) * 0.05 + 0.01
    scale_b = torch.rand((1, N), device=device, dtype=torch.float32) * 0.05 + 0.01

    #adding clamping to avoid nan, inf (overflow)
    min_val=1e-3 
    max_val = 0.02 * (1024 / max(K, N))
    scale_a = scale_a.clamp(min=min_val, max=max_val)
    scale_b = scale_b.clamp(min=min_val, max=max_val)

    #allocation in advance
    a_shared_symm = dist._symmetric_memory.empty(
        a_shared.shape,
        dtype=a_shared.dtype,
        device=a_shared.device
    )
    a_shared_symm.copy_(a_shared)

    run_example(
        helion_ag_matmul,
        reference_ag_matmul,
        (a_shared_symm, b, scale_a, scale_b, world_size, dist.group.WORLD),
        rtol=1e-1,
        atol=1e-1,
    )


# %%
def main() -> None:
    """
    Main entry point that initializes the distributed environment and runs the test.
    Sets up the distributed process group, runs the test, and then cleans up.
    """
    _SymmetricMemory.signal_pad_size = 1024 * 1024 * 1024
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    test(8192, 2560, 8192, world_size, device)
    dist.destroy_process_group()


# %%
if __name__ == "__main__":
    """
    Run with:
    python -m torch.distributed.run --standalone \
    --nproc-per-node 4 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --no_python python3 examples/distributed/fp8_scaled_all_gather_matmul.py
    """
    assert DEVICE.type == "cuda", "Requires CUDA device"
    main()

""" 
- autouting the kernel in vllm per rank, and moving the config to here 
Helion:
  commit : main and also the commit :ea484a38c285318629e5b491be30f3e4955ab2be
                      (1.0.1.dev30+g49bbef0f9  /home/cloud-user/vllm/helion)

  =================================================================
  Benchmark Results
  =================================================================
  Implementation       Time (ms)    Speedup        
  -----------------------------------------------------------------
  helion               0.7250       0.78x          
  torch                0.5646       1.00x (ref)    
  =================================================================

  in https://github.com/vllm-project/vllm/pull/33933 running 
  VLLM_USE_HELION_BACKEND=1  torchrun --nproc_per_node=4   benchmarks/kernels/helion/benchmark_all_gather_gemm_fp8.py

  in vllm:
  === Final Distributed Benchmark Results ===
  rank | shape                        | baseline_ms | kernel_ms | speedup(x) | baseline_peak(MB) | kernel_peak(MB) | mem_improve(x)
  -----+------------------------------+-------------+-----------+------------+-------------------+-----------------+---------------
  ALL  | M=8192,N=2560,K=8192splits=1 | 0.914       | 0.809     | 1.130      | 414.05            | 414.02          | 1.000         
  TODO: what is the different between the 2 benchmarks

  q: can we make it faster? in terms of the kernel?
"""