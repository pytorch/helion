from __future__ import annotations

import contextlib
from datetime import timedelta
import os
from typing import ClassVar
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import MultiProcessTestCase

import helion
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfPallasInterpret
import helion.language as hl

_WIDTH = 128
_REMOTE_SLOT = 1


def _rank_values(rank: int, width: int = _WIDTH) -> np.ndarray:
    return rank * 1000 + np.arange(width, dtype=np.float32)


def _expected_cyclic_destination(world_size: int) -> np.ndarray:
    expected = np.full((world_size, 2, _WIDTH), -1.0, dtype=np.float32)
    for rank in range(world_size):
        expected[rank, _REMOTE_SLOT] = _rank_values((rank - 1) % world_size)
    return expected


def _cyclic_remote_copy(
    src: torch.Tensor,
    dst: torch.Tensor,
    peers: torch.Tensor,
    slots: torch.Tensor,
    signal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Push one row to the next rank and wait for the previous rank's row."""
    for _program in hl.grid(1):
        if signal is None:
            copy = hl.start_async_remote_copy(
                src,
                [0, 0],
                peers[0, 0],
                dst=dst,
                dst_index=[0, slots[0, 0]],
            )
        else:
            copy = hl.start_async_remote_copy(
                src,
                [0, 0],
                peers[0, 0],
                dst=dst,
                dst_index=[0, slots[0, 0]],
                signal=signal,
                signal_index=[0],
            )
        copy.wait()
    return dst


_pallas_cyclic_remote_copy = helion.kernel(
    backend="pallas",
    static_shapes=True,
    config=helion.Config(block_sizes=[]),
)(_cyclic_remote_copy)


_triton_cyclic_remote_copy = helion.kernel(
    backend="triton",
    static_shapes=True,
    config=helion.Config(block_sizes=[]),
)(_cyclic_remote_copy)


@unittest.skipUnless(
    torch.version.cuda is not None and torch.cuda.device_count() >= 4,
    "requires four NVIDIA CUDA devices",
)
@onlyBackends(["triton"])
class TestRemoteCopyGPU(TestCase, MultiProcessTestCase):
    """Execute the Triton lowering with four NVSHMEM-connected GPUs."""

    _nvshmem_env: ClassVar[dict[str, str]] = {
        "NVSHMEM_SYMMETRIC_SIZE": "1G",
        "NVSHMEM_DISABLE_NVLS": "1",
        "NCCL_NVLS_ENABLE": "0",
    }

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._class_stack = contextlib.ExitStack()
        cls._class_stack.enter_context(patch.dict(os.environ, cls._nvshmem_env))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._class_stack.close()
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        super().tearDown()

    @property
    def world_size(self) -> int:
        return 4

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def _init_process(self) -> None:
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            device_id=self.device,
        )
        torch.distributed.distributed_c10d._set_pg_timeout(
            timedelta(seconds=60), dist.group.WORLD
        )
        symm_mem.set_backend("NVSHMEM")

    def _cleanup_process(self) -> None:
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()

    def _make_symmetric_buffers(
        self, shape: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dst = symm_mem.empty(*shape, dtype=torch.float32, device=self.device)
        signal = symm_mem.empty(1, dtype=torch.int64, device=self.device)
        group = dist.group.WORLD
        assert group is not None
        dst_handle = symm_mem.rendezvous(dst, group=group)
        signal_handle = symm_mem.rendezvous(signal, group=group)
        self.assertEqual(dst_handle.world_size, self.world_size)
        self.assertEqual(signal_handle.world_size, self.world_size)
        return dst, signal

    def test_cyclic_remote_copy(self) -> None:
        self._init_process()
        try:
            src = torch.from_numpy(_rank_values(self.rank)).to(self.device)
            src = src.reshape(1, 1, _WIDTH)
            dst, signal = self._make_symmetric_buffers((1, 2, _WIDTH))
            peers = torch.tensor(
                [[(self.rank + 1) % self.world_size]],
                dtype=torch.int32,
                device=self.device,
            )
            slots = torch.full(
                (1, 1), _REMOTE_SLOT, dtype=torch.int32, device=self.device
            )

            expected = torch.from_numpy(
                _expected_cyclic_destination(self.world_size)[self.rank]
            ).to(self.device)
            for _invocation in range(2):
                dst.fill_(-1)
                signal.zero_()
                dist.barrier()
                result = _triton_cyclic_remote_copy(src, dst, peers, slots, signal)
                torch.cuda.synchronize()
                torch.testing.assert_close(result[0], expected)
        finally:
            self._cleanup_process()


@onlyBackends(["pallas"])
class TestRemoteCopyJaxRuntime(TestCase):
    @staticmethod
    def _run_cyclic_copy(mesh, mesh_axis, kernel_fn) -> None:
        import jax
        import jax.numpy as jnp

        world_size = mesh.devices.size
        partition = jax.sharding.PartitionSpec
        input_specs = (
            partition(mesh_axis, None, None),
            partition(mesh_axis, None, None),
            partition(mesh_axis, None),
            partition(mesh_axis, None),
        )
        src = jnp.stack(
            [jnp.asarray(_rank_values(rank))[None, :] for rank in range(world_size)]
        )
        dst = jnp.full((world_size, 2, _WIDTH), -1.0, dtype=jnp.float32)
        peers = ((jnp.arange(world_size, dtype=jnp.int32) + 1) % world_size)[:, None]
        slots = jnp.full((world_size, 1), _REMOTE_SLOT, dtype=jnp.int32)
        inputs = tuple(
            jax.device_put(value, jax.sharding.NamedSharding(mesh, spec))
            for value, spec in zip((src, dst, peers, slots), input_specs, strict=True)
        )
        copy = jax.jit(
            jax.shard_map(
                kernel_fn,
                mesh=mesh,
                in_specs=input_specs,
                out_specs=partition(mesh_axis, None, None),
                check_vma=False,
            )
        )
        expected = _expected_cyclic_destination(world_size)
        for _invocation in range(2):
            result = jax.block_until_ready(copy(*inputs))
            np.testing.assert_array_equal(
                np.asarray(result)[:, _REMOTE_SLOT], expected[:, _REMOTE_SLOT]
            )

    @skipIfPallasInterpret("remote copies require physical TPU devices")
    def test_cyclic_remote_copy(self) -> None:
        import jax

        devices = [device for device in jax.local_devices() if device.platform == "tpu"]
        if len(devices) < 2:
            self.skipTest("requires at least two TPU devices")
        mesh = jax.make_mesh((2,), ("peer",), devices=devices[:2])
        self._run_cyclic_copy(mesh, "peer", _pallas_cyclic_remote_copy.jax_fn)

    @skipIfPallasInterpret("remote copies require physical TPU devices")
    def test_multiaxis_flat_logical_peers(self) -> None:
        import jax

        devices = [device for device in jax.local_devices() if device.platform == "tpu"]
        if len(devices) < 8:
            self.skipTest("requires at least eight TPU devices")
        mesh = jax.make_mesh((2, 4), ("data", "peer"), devices=devices[:8])
        self._run_cyclic_copy(
            mesh,
            ("data", "peer"),
            _pallas_cyclic_remote_copy.jax_fn,
        )

    @skipIfPallasInterpret("remote copies require physical TPU devices")
    def test_precompiled_standalone(self) -> None:
        import sys
        import types

        args = (
            torch.zeros(1, 1, _WIDTH),
            torch.zeros(1, 2, _WIDTH),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.zeros(1, 1, dtype=torch.int32),
        )
        source = _pallas_cyclic_remote_copy.bind(args).to_code(
            options=helion.OutputCodeOptions(
                allow_helion_deps=False,
                jax_fn=True,
            )
        )
        name = "precompiled_remote_copy_test"
        module = types.ModuleType(name)
        sys.modules[name] = module
        try:
            exec(compile(source, name, "exec"), module.__dict__)

            import jax

            devices = [
                device for device in jax.local_devices() if device.platform == "tpu"
            ]
            if len(devices) < 2:
                self.skipTest("requires at least two TPU devices")
            mesh = jax.make_mesh((2,), ("peer",), devices=devices[:2])
            self._run_cyclic_copy(mesh, "peer", module._cyclic_remote_copy)
        finally:
            sys.modules.pop(name, None)
