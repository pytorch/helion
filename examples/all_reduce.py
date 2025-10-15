"""
One-Shot All-Reduce Example
===========================
This example demonstrates how to implement a one-shot pulling all-reduce operation
using Helion and PyTorch's distributed capabilities. It includes a Helion kernel
demonstrating how to do cross-device synchronization using symmetric memory signal pads
and access symmetric memory tensor resident on peer devices.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.utils.cpp_extension import load_inline

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# %%
# Work around before symm mem natively supports extract dev_ptrs as tensors: from_blob

# %%
from_blob_cpp = """
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


at::Tensor from_blob(uint64_t data_ptr, c10::IntArrayRef sizes, py::object dtype) {

    at::Tensor tensor = at::for_blob((void*)data_ptr, sizes)
             .deleter([](void *ptr) {
               ;
             })
             .options(at::device(at::kCUDA).dtype(((THPDtype*)dtype.ptr())->scalar_type))
             .make_tensor();

    return tensor;
}
"""

cpp_mod = load_inline(
    "cpp_mod", cpp_sources=from_blob_cpp, with_cuda=True, functions=["from_blob"]
)


def dev_array_to_tensor_short(
    dev_array_ptr: int, shape: tuple[int], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Convert a device array pointer to a PyTorch tensor.

    This is a workaround function that creates a PyTorch tensor from a raw device pointer
    using the C++ extension. It's used to interface with symmetric memory device pointers
    before native support is available.

    Args:
        dev_array_ptr: Raw device pointer as integer
        shape: Shape of the tensor to create
        dtype: PyTorch data type for the tensor
        device: Target device for the tensor

    Returns:
        PyTorch tensor created from the device pointer
    """
    return cpp_mod.from_blob(dev_array_ptr, shape, dtype)  # pyright: ignore[reportAttributeAccessIssue]


# %%
# One Shot All-Reduce Kernel Implementation
# -----------------------------------------


# %%
@helion.jit(
    config=helion.Config(
        block_sizes=[8192],
        num_warps=32,
    ),
    static_shapes=True,
)
def one_shot_all_reduce_kernel(
    signal_pad_addrs: torch.Tensor,
    local_signal_pad: torch.Tensor,
    a_shared_tuple: tuple[torch.Tensor, ...],
    my_rank: hl.constexpr,
) -> torch.Tensor:
    """
    Helion JIT-compiled kernel for one-shot all-reduce operation.

    This kernel implements a distributed all-reduce using symmetric memory and signal pads
    for cross-device synchronization. It performs element-wise summation across all devices
    in the distributed group using tiled computation for memory efficiency.

    Args:
        signal_pad_addrs: Tensor containing addresses of signal pads for all devices
        local_signal_pad: Local signal pad for synchronization
        a_shared_tuple: Tuple of shared tensors from all devices in the group
        my_rank: Current device's rank in the distributed group

    Returns:
        Tensor containing the all-reduced result (sum across all devices)
    """
    _, world_size = local_signal_pad.size()
    world_size = hl.specialize(world_size)
    out = torch.empty_like(a_shared_tuple[0])
    N = out.size(0)

    for tile_n in hl.tile(N):
        # Sync all devices through signal_pad to make sure
        # all previous writes to the shared tensor are visible
        ptr_tile = signal_pad_addrs[:]
        stack_signalpad = hl.stacktensor_like(local_signal_pad, ptr_tile)
        hl.signal(
            stack_signalpad,
            [tile_n.id, my_rank],
            signal=1,
            wait_for=0,
            scope="sys",
            hasPreviousMemAccess=False,
        )

        for world in hl.tile(world_size, block_size=world_size):
            hl.wait(
                local_signal_pad,
                [tile_n.id, world],
                signal=1,
                update=0,
                scope="sys",
            )

        acc = hl.zeros(
            [tile_n], dtype=a_shared_tuple[0].dtype, device=local_signal_pad.device
        )

        for a in a_shared_tuple:
            acc += a[tile_n]

        out[tile_n] = acc

        # Sync all devices through signal_pad to make sure our writes to shared
        # tensor are visible to subsequent kernels.
        hl.signal(
            stack_signalpad, [tile_n.id, my_rank], signal=1, wait_for=0, scope="sys"
        )

        for world in hl.tile(world_size, block_size=world_size):
            hl.wait(
                local_signal_pad,
                [tile_n.id, world],
                signal=1,
                update=0,
                scope="sys",
                hasSubsequentMemAccess=False,
            )
    return out


# %%
# Attract tensors from symmetric memory handler
# ---------------------------------------------


# %%
def helion_one_shot_all_reduce(a_shared: torch.Tensor) -> torch.Tensor:
    """
    Prepares symmetric memory tensors for Helion one-shot all-reduce kernel.
    Tracks shared tensors as tuple of tensors, and/or dev_ptrs tensors.

    Args:
        a_shared: Input tensor to be all-reduced across all devices

    Returns:
        Tensor containing the all-reduced result (sum across all devices)
    """
    assert dist.group.WORLD is not None

    symm_mem_hdl = symm_mem.rendezvous(a_shared, group=dist.group.WORLD)

    a_shared_tuple = tuple(
        [
            symm_mem_hdl.get_buffer(i, tuple(a_shared.shape), a_shared.dtype)
            for i in range(symm_mem_hdl.world_size)
        ]
    )

    local_signal_pad = symm_mem_hdl.get_signal_pad(
        symm_mem_hdl.rank, dtype=torch.int32
    ).view(-1, symm_mem_hdl.world_size)

    signal_pad_addrs = dev_array_to_tensor_short(
        symm_mem_hdl.signal_pad_ptrs_dev,
        (symm_mem_hdl.world_size,),
        dtype=torch.uint64,
        device=a_shared.device,
    )

    return one_shot_all_reduce_kernel(
        signal_pad_addrs,
        local_signal_pad,
        a_shared_tuple,
        my_rank=symm_mem_hdl.rank,
    )


def reference_one_shot_all_reduce(a_shared: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation using the symmetric memory one-shot primitive.
    """
    dist_group = dist.group.WORLD
    if dist_group is None:
        raise RuntimeError("No distributed group available")

    a_shared_clone = symm_mem.empty(
        a_shared.shape,
        dtype=a_shared.dtype,
        device=a_shared.device,
    )
    symm_mem.rendezvous(a_shared_clone, dist_group.group_name)
    a_shared_clone.copy_(a_shared)

    return torch.ops.symm_mem.one_shot_all_reduce(  # pyright: ignore[reportCallIssue]
        a_shared_clone, "sum", dist_group.group_name
    )


# %%
# Testing Function
# ----------------


# %%
def test(N: int, device: torch.device, dtype: torch.dtype) -> None:
    """
    Test the Helion all-reduce implementation against PyTorch's reference implementation.
    Args:
        N: Total number of elements to test (will be divided by world_size per device)
        device: CUDA device to run the test on
        dtype: Data type for the test tensors
    """
    dist_group = dist.group.WORLD
    assert dist_group is not None

    world_size = dist.get_world_size()
    a_shared = symm_mem.empty(N // world_size, dtype=dtype, device=device).normal_()

    run_example(
        helion_one_shot_all_reduce,
        reference_one_shot_all_reduce,
        (a_shared,),
        rtol=1e-1,
        atol=1e-1,
    )


def main() -> None:
    """
    Main entry point for the all-reduce example.

    Sets up the distributed environment, initializes CUDA devices, and runs the
    all-reduce test, and then clean up.
    """
    rank = int(os.environ["LOCAL_RANK"])
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    test(16384, device, torch.bfloat16)

    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Run with:
    torchrun \
    --nnodes 1 --nproc-per-node 8 \
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --no_python python3 examples/all_reduce.py
    """
    # TODO(adam-smnk): generalize to XPU
    assert DEVICE.type == "cuda", "Requires CUDA device"
    main()
