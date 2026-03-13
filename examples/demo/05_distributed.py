"""Distributed: fused all-reduce + RMS norm in one kernel.
Requires multi-GPU with NVSHMEM.
"""
import os
import torch
import helion
from helion._testing import DEVICE
import helion.language as hl
from examples.distributed.utils import symm_mem_sync


@helion.jit(config=helion.Config(block_sizes=[8], num_warps=8), static_shapes=True,
            ignore_warnings=[helion.exc.TensorOperationInWrapper])
def allreduce_rmsnorm(
    symm_mem_buffer: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    signal_pad_ptrs: torch.Tensor,
    EPS: hl.constexpr,
    RANK: hl.constexpr,
    WORLD_SIZE: hl.constexpr,
    GROUP_NAME: hl.constexpr,
) -> torch.Tensor:
    N, D = x.size()
    output = torch.empty_like(x)
    buffer_tuple = torch.ops.symm_mem.get_remote_tensors(symm_mem_buffer, GROUP_NAME)

    for tile_n in hl.tile(N):
        symm_mem_buffer[tile_n, :] = x[tile_n, :]
        hl.triton_kernel(symm_mem_sync,
                         args=(signal_pad_ptrs, None, RANK, WORLD_SIZE, True, True),
                         output_like=None)

        acc = hl.zeros([tile_n, D], dtype=torch.float32)
        for remote in buffer_tuple:
            acc = acc + remote[tile_n, :].to(torch.float32)

        variance = torch.mean(acc * acc, dim=-1, keepdim=True)
        output[tile_n, :] = (acc * torch.rsqrt(variance + EPS)  # type: ignore[unsupported-operation]
                             * weight[None, :].to(torch.float32)).to(x.dtype)

        hl.triton_kernel(symm_mem_sync,
                         args=(signal_pad_ptrs, None, RANK, WORLD_SIZE, True, False),
                         output_like=None)
    return output


if "LOCAL_RANK" in os.environ:
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem
    symm_mem.set_backend("NVSHMEM")
    rank = int(os.environ["LOCAL_RANK"])
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    assert group is not None
    symm_mem.enable_symm_mem_for_group(group.group_name)  # pyrefly: ignore [deprecated]

    N, D = 128, 4096
    x = torch.randn(N, D, device=device, dtype=torch.bfloat16)
    weight = torch.randn(D, device=device, dtype=torch.bfloat16)
    buf = symm_mem.empty(N, D, dtype=x.dtype, device=device)
    hdl = symm_mem.rendezvous(buf, group.group_name)

    result = allreduce_rmsnorm(buf, x, weight, hdl.signal_pad_ptrs_dev,
                               EPS=1e-5, RANK=hdl.rank,
                               WORLD_SIZE=hdl.world_size, GROUP_NAME=group.group_name)

    x_ref = x.clone()
    dist.all_reduce(x_ref)
    x_f = x_ref.float()
    expected = (x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + 1e-5)
                * weight.float()).to(torch.bfloat16)
    torch.testing.assert_close(result, expected, atol=0.1, rtol=0.1)
    if rank == 0:
        print("Correctness: PASSED")
    dist.destroy_process_group()
else:
    import subprocess, sys
    ngpu = torch.cuda.device_count()
    print(f"Relaunching with torchrun --nproc-per-node {ngpu}", flush=True)
    env = {**os.environ, "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME": "lo",
           "NCCL_SOCKET_IFNAME": "lo", "HELION_PRINT_OUTPUT_CODE": "1"}
    sys.exit(subprocess.run([
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc-per-node={ngpu}",
        "--rdzv-backend=c10d", "--rdzv-endpoint=localhost:0",
        "--no_python", sys.executable, __file__,
    ], env=env).returncode)
