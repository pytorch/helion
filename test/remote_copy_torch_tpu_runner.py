"""Isolated process-per-device runner for the TorchTPU remote-copy test."""

from __future__ import annotations

import os

# The launcher must not claim a TPU before the per-device workers are spawned.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch.multiprocessing as mp

from test.remote_copy_kernels import _PIPELINE_STEPS
from test.remote_copy_kernels import _WIDTH
from test.remote_copy_kernels import _pallas_reusable_cyclic_remote_copy
from test.remote_copy_kernels import _rank_pipeline_values


def _remote_copy_worker(rank: int, world_size: int, master_port: int) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
            "GROUP_RANK": "0",
            "LOCAL_WORLD_SIZE": str(world_size),
        }
    )

    import jax
    import torch.distributed as dist
    from torch_tpu import _loader as torch_tpu_loader  # pyrefly: ignore[missing-import]

    torch_tpu_loader.load()

    from torch_tpu._internal import pallas  # pyrefly: ignore[missing-import]

    dist.init_process_group(backend="tpu_dist")
    try:
        partition = jax.sharding.PartitionSpec
        mesh = jax.make_mesh((world_size,), ("peer",))
        input_specs = (
            partition("peer", None, None),
            partition("peer", None, None),
            partition("peer", None),
        )
        exchange = jax.shard_map(
            _pallas_reusable_cyclic_remote_copy.jax_fn,
            mesh=mesh,
            in_specs=input_specs,
            out_specs=partition("peer", None, None),
            check_vma=False,
        )

        def exchange_jax(
            src: jax.Array,
            dst: jax.Array,
            peers: jax.Array,
        ) -> jax.Array:
            return exchange(src, dst, peers)

        exchange_jax.__annotations__ = {
            "src": jax.Array,
            "dst": jax.Array,
            "peers": jax.Array,
            "return": jax.Array,
        }
        exchange_op = pallas.jax_op(
            "pallas::helion_remote_copy_test",
            exchange_jax,
            mesh=mesh,
            input_partition_specs=input_specs,
        )

        device = torch.device("tpu")
        src = torch.from_numpy(_rank_pipeline_values(rank)).to(device).unsqueeze(0)
        dst = torch.zeros_like(src)
        peers = torch.tensor(
            [[(rank + 1) % world_size, (rank - 1) % world_size]],
            dtype=torch.int32,
            device=device,
        )

        result = exchange_op(src, dst, peers)
        expected = torch.from_numpy(
            _rank_pipeline_values((rank - 1) % world_size)
        ).unsqueeze(0)
        assert result.shape == (1, _PIPELINE_STEPS, _WIDTH)
        torch.testing.assert_close(result.cpu(), expected)
    finally:
        dist.destroy_process_group()


def main() -> None:
    import portpicker  # pyrefly: ignore[missing-import]
    from torch_tpu._internal.distributed.launchers import (  # pyrefly: ignore[missing-import]
        singlehost_wrapper,
    )

    world_size = 4
    singlehost_wrapper.prepare_tpu_environment(world_size=world_size)
    master_port = portpicker.pick_unused_port()
    mp.spawn(
        _remote_copy_worker,
        args=(world_size, master_port),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
