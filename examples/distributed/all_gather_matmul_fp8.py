import os
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import helion
from helion._testing import DEVICE
import helion.language as hl


@helion.kernel(
    config=helion.Config(
        block_sizes=[128, 256, 64],
        num_warps=8,
        num_stages=3,
        indexing="block_ptr",
    ),
    static_shapes=True,
)
def helion_matmul_w_progress_fp8(
    a: torch.Tensor,      # FP8 input
    a_shared: torch.Tensor,  # FP8 gathered input
    b: torch.Tensor,      # FP8 input
    progress: torch.Tensor,
    SPLITS_PER_RANK: int,
    RANK: int,
) -> torch.Tensor:
    M, K = a.size()
    K2, N = b.size()
    assert K2 == K, f"size mismatch {K2} != {K}"
    out = torch.empty(
        [M, N], dtype=torch.float16, device=a.device  # Output in FP16
    )
    M_per_rank = a_shared.size(0)
    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)  # Accumulate in FP32  
        # hl.wait(
        #     progress,
        #     [
        #         tile_m.begin // (M_per_rank // SPLITS_PER_RANK),
        #     ],
        #     signal=1,
        # )
        for tile_k in hl.tile(K):
            # FP8 -> FP32 for accumulation
            acc = torch.addmm(
                acc,
                a[tile_m, tile_k].to(torch.float32),
                b[tile_k, tile_n].to(torch.float32),
            )
        out[tile_m, tile_n] = acc.to(torch.float16)  # FP16 output
    return out


def helion_all_gather_matmul_fp8(
    a_shared: torch.Tensor,
    b: torch.Tensor,
    a_out: torch.Tensor | None = None,
    progress: torch.Tensor | None = None,
    **kwargs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Combines all-gather and matrix multiplication for FP8 inputs.
    """
    configs = {
        "SPLITS_PER_RANK": kwargs.get("splits_per_rank", 1),
    }
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Prepare progress tensor
    if progress is None:
        progress = torch.zeros(
            world_size * configs["SPLITS_PER_RANK"],
            dtype=torch.uint32,
            device=a_shared.device,
        )

    # Allocate a_out for gathered results
    a_shape = list(a_shared.shape)
    a_shape[0] *= world_size
    if a_out is None:
        a_out = torch.empty(a_shape, dtype=a_shared.dtype, device=a_shared.device)

    # Perform all-gather
    dist.all_gather(list(a_out.chunk(world_size, dim=0)), a_shared) #instead of symm_mem.rendezvous 

    # Perform matrix multiplication with progress tracking
    print("Helion kernel launch")
    c = helion_matmul_w_progress_fp8(
        a_out,
        a_shared,
        b,
        progress,
        SPLITS_PER_RANK=configs["SPLITS_PER_RANK"],
        RANK=rank,
    )
    print("Helion Out kernel launch")

    return a_out, c


def test_fp8(M: int, N: int, K: int, world_size: int, device: torch.device) -> None:
    """
    Functional test for FP8 all-gather matrix multiplication.
    """
    torch.cuda.empty_cache()

    # Input tensor, shared across ranks
    a_shared = symm_mem.empty(
        M // world_size, K, dtype=torch.float8_e4m3fn, device=device
    )
    
    # Randomly initialize a_shared in FP32 and cast to FP8
    random_values = torch.randn(M // world_size, K, device=device, dtype=torch.float32)
    a_shared.copy_(random_values.to(dtype=torch.float8_e4m3fn))

    # Random matrix `b` in FP16 converted to FP8
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16).to(torch.float8_e4m3fn)

    print(f"[Rank {dist.get_rank()}] Initialized b matrix: {b.shape=}, {b.dtype=}")

    # Call matrix multiplication with all-gather
    a_out, c = helion_all_gather_matmul_fp8(a_shared, b)
    print("helion_all_gather_matmul_fp8 out ")
    # Generate FP16 reference results (golden)
    golden_a = a_out.to(torch.float32).clone()
    golden_b = b.to(torch.float32).clone()
    print("golden_a and golden_b created")

    mm_golden = golden_a @ golden_b
    print("golden_a @ golden_b ")

    # Validate against the reference
    torch.testing.assert_close(c.to(torch.float32), mm_golden, rtol=1e-1, atol=1e-1)

    print(f"[Rank {dist.get_rank()}] Matrix multiplication passed FP8 test!")


def main_fp8() -> None:
    """
    Main function for running FP8 distributed test.
    """
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    test_fp8(4096, 6656, 16384//2, world_size, device)
    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Run the FP8 All-Gather MatMul Example.
    """
    print("Running FP8 All-Gather MatMul Example")
    assert DEVICE.type == "cuda", "Requires CUDA device"
    main_fp8()