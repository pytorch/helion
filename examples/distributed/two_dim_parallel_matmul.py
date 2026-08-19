"""
2D-Parallel Matrix Multiplication Example
==========================================
Demonstrates two independent dimensions of parallelism in a single fused kernel:

* **Sequence parallelism (SP)**: each rank owns a disjoint shard of input rows.
  SP ranks never communicate with each other – they produce fully independent
  output rows.

* **Tensor parallelism (TP)**: within each SP group, every rank holds a shard of
  the K (inner/reduction) dimension.  Partial products must be summed via an
  intra-TP all-reduce implemented with symmetric-memory signal pads.

Rank layout (example: SP=2, TP=2, total 4 GPUs)::

       TP=0      TP=1
SP=0:  rank 0   rank 1   <- compute rows 0 .. M/2 of output
SP=1:  rank 2   rank 3   <- compute rows M/2 .. M of output

Each rank ``(sp, tp)`` holds:

* ``a_local``  ``[M/SP, K/TP]``  – its row-shard × K-shard of **A**
* ``b_local``  ``[K/TP,  N  ]``  – its K-shard of **B** (full N on every rank)

The kernel steps for every output tile ``[tile_m, tile_n]``:

1. Compute partial GEMM:  ``a_local @ b_local``  →  partial ``[M/SP, N]``
2. Write partial result to a symmetric-memory buffer (visible to TP peers).
3. Intra-TP barrier (release + acquire) so every peer can read the partial.
4. Sum all TP peers' partials in-kernel (fused all-reduce over the TP group).
5. Intra-TP release barrier to allow cleanup / the next tile.
"""

from __future__ import annotations

import functools
import os

import torch
from torch._C._distributed_c10d import _SymmetricMemory
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed.device_mesh import init_device_mesh

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl
from helion.runtime.dist_utils import symm_mem_sync


@helion.kernel(
    config=helion.Config(
        block_sizes=[64, 64, 32],
        num_warps=8,
        num_stages=3,
        indexing="block_ptr",
    ),
    static_shapes=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def two_dim_parallel_matmul_kernel(
    a_local: torch.Tensor,  # [M/SP, K/TP]
    b_local: torch.Tensor,  # [K/TP,   N ]
    symm_mem_buf: torch.Tensor,  # [M/SP,   N ]  symmetric-memory scratch
    signal_pad_ptrs: torch.Tensor,
    TP_RANK: hl.constexpr,
    TP_SIZE: hl.constexpr,
    GROUP_NAME: hl.ProcessGroupName,
) -> torch.Tensor:
    """
    Fused 2D-parallel (SP × TP) matmul kernel.

    Dimension 1 – Sequence Parallel (SP): different ranks own different M rows.
    No communication occurs across SP ranks; each computes its rows independently.

    Dimension 2 – Tensor Parallel (TP): ranks in the same SP group each own a
    K-shard of A and B.  After the local partial GEMM, an in-kernel all-reduce
    over the TP group produces the correct output for this rank's M rows.
    """
    M_local, K_local = a_local.size()
    N = b_local.size(1)
    out = torch.empty([M_local, N], dtype=a_local.dtype, device=a_local.device)

    # Symmetric-memory views of every TP peer's scratch buffer.
    remote_bufs = torch.ops.symm_mem.get_remote_tensors(symm_mem_buf, GROUP_NAME)

    for tile_m, tile_n in hl.tile([M_local, N]):
        # ── Dimension 2 (Tensor Parallel): partial GEMM over local K shard ──
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K_local):
            acc = torch.addmm(acc, a_local[tile_m, tile_k], b_local[tile_k, tile_n])

        # Write partial result to this rank's symmetric-memory buffer.
        symm_mem_buf[tile_m, tile_n] = acc.to(a_local.dtype)

        # Barrier: release our write so peers can see it; acquire their writes.
        hl.triton_kernel(
            symm_mem_sync,
            args=(signal_pad_ptrs, None, TP_RANK, TP_SIZE, True, True),
            output_like=None,
        )

        # ── Reduce: sum partials from all TP peers (fused intra-TP all-reduce) ──
        acc_full = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for peer_buf in remote_bufs:
            acc_full = acc_full + peer_buf[tile_m, tile_n].to(torch.float32)
        out[tile_m, tile_n] = acc_full.to(a_local.dtype)

        # Release barrier: signal that we are done reading the shared buffers.
        hl.triton_kernel(
            symm_mem_sync,
            args=(signal_pad_ptrs, None, TP_RANK, TP_SIZE, True, False),
            output_like=None,
        )

    return out


def helion_two_dim_parallel_matmul(
    a_local: torch.Tensor,
    b_local: torch.Tensor,
    tp_group: dist.ProcessGroup,
    symm_mem_buf: torch.Tensor,
) -> torch.Tensor:
    """
    Allocate symmetric memory for the TP group and invoke the 2D-parallel kernel.

    Args:
        a_local: Local A shard ``[M/SP, K/TP]``.
        b_local: Local B shard ``[K/TP,  N ]``.
        tp_group: Intra-TP process group for this rank.
    """
    group_name = tp_group.group_name  # type: ignore[missing-attribute]

    hdl = symm_mem.rendezvous(symm_mem_buf, group_name)

    return two_dim_parallel_matmul_kernel(
        a_local,
        b_local,
        symm_mem_buf,
        hdl.signal_pad_ptrs_dev,
        TP_RANK=hdl.rank,
        TP_SIZE=hdl.world_size,
        GROUP_NAME=group_name,
    )


def reference_two_dim_parallel_matmul(
    a_local: torch.Tensor,
    b_local: torch.Tensor,
    tp_group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    Reference: all-gather K shards within the TP group, then do a local matmul.
    """
    tp_size = dist.get_world_size(tp_group)

    # All-gather the K-sharded a_local across the TP group → [M/SP, K].
    a_parts = [torch.empty_like(a_local) for _ in range(tp_size)]
    dist.all_gather(a_parts, a_local, group=tp_group)
    a_full = torch.cat(a_parts, dim=1)

    # All-gather the K-sharded b_local across the TP group → [K, N].
    b_parts = [torch.empty_like(b_local) for _ in range(tp_size)]
    dist.all_gather(b_parts, b_local, group=tp_group)
    b_full = torch.cat(b_parts, dim=0)

    return torch.mm(a_full.float(), b_full.float()).to(a_local.dtype)


def test(
    M: int,
    K: int,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
    tp_group: dist.ProcessGroup,
    sp_rank: int,
    tp_size: int,
    sp_size: int,
) -> None:
    """Test the 2D-parallel kernel against the reference."""
    tp_rank = dist.get_rank(tp_group)
    torch.manual_seed(42 + sp_rank * tp_size + tp_rank)

    M_local = M // sp_size
    K_local = K // tp_size

    # Each (sp, tp) rank owns a unique [M/SP, K/TP] block of A.
    a_local = torch.randn(M_local, K_local, dtype=dtype, device=device)

    # Each TP rank owns its K-shard of B with the full N dimension.
    # Ranks in the same TP group hold different K-shards; ranks in different SP
    # groups but the same TP rank hold the *same* K-shard of B.
    torch.manual_seed(42 + tp_rank)
    b_local = torch.randn(K_local, N, dtype=dtype, device=device)

    symm_mem_buf = symm_mem.empty(
        M_local, N, dtype=a_local.dtype, device=a_local.device
    )
    symm_mem.rendezvous(symm_mem_buf, tp_group.group_name)

    _helion_two_dim_parallel_matmul = functools.partial(
        helion_two_dim_parallel_matmul, symm_mem_buf=symm_mem_buf
    )

    run_example(
        lambda a, b: _helion_two_dim_parallel_matmul(a, b, tp_group),
        lambda a, b: reference_two_dim_parallel_matmul(a, b, tp_group),
        (a_local, b_local),
        rtol=2e-1,
        atol=2e-1,
        process_group_name=tp_group.group_name,
    )


def main() -> None:
    """
    Initialize a 2-D device mesh, enable symmetric memory for the TP groups,
    and run the 2D-parallel matmul test.
    """
    _SymmetricMemory.signal_pad_size = 1024 * 1024 * 16

    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size >= 4 and world_size % 2 == 0, (
        f"Requires at least 4 GPUs arranged as SP×TP mesh (got {world_size})"
    )

    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")

    # Build a 2-D device mesh: SP (outer dim, row parallelism) × TP (inner dim,
    # K-reduction parallelism).
    tp_size = 2
    sp_size = world_size // tp_size
    mesh = init_device_mesh(
        "cuda",
        (sp_size, tp_size),
        mesh_dim_names=("sp", "tp"),
    )
    tp_group = mesh.get_group("tp")
    sp_rank = rank // tp_size

    test(
        M=1024,
        K=256,
        N=512,
        device=device,
        dtype=torch.float32,
        tp_group=tp_group,
        sp_rank=sp_rank,
        tp_size=tp_size,
        sp_size=sp_size,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Run with:
    python -m torch.distributed.run --standalone \\
    --nproc-per-node 4 \\
    --rdzv-backend c10d --rdzv-endpoint localhost:0 \\
    examples/distributed/two_dim_parallel_matmul.py
    """
    assert DEVICE.type == "cuda", "Requires CUDA device"
    main()
