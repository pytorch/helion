from __future__ import annotations

import contextlib
from datetime import timedelta
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import TypeVar
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import MultiProcessTestCase

import helion
import helion.language as hl

_TORCH_TPU_RUNNER_ARG = "--torch-tpu-runner"
_IS_TORCH_TPU_RUNNER = _TORCH_TPU_RUNNER_ARG in sys.argv

if TYPE_CHECKING:
    from helion._testing import TestCase
    from helion._testing import onlyBackends
    from helion._testing import skipIfPallasInterpret
elif _IS_TORCH_TPU_RUNNER:
    _Decorated = TypeVar("_Decorated")

    class TestCase:
        pass

    def _identity_test_decorator(_condition: object):
        def decorate(item: _Decorated) -> _Decorated:
            return item

        return decorate

    onlyBackends = _identity_test_decorator
    skipIfPallasInterpret = _identity_test_decorator
else:
    from helion._testing import TestCase
    from helion._testing import onlyBackends
    from helion._testing import skipIfPallasInterpret


_WIDTH = 128
_REMOTE_SLOT = 1
_PIPELINE_STEPS = 4
_GATHER_ROWS = 8


def _distributed_test_config(pallas_loop_type: str) -> helion.Config:
    if helion.Settings().backend == "pallas":
        return helion.Config(
            block_sizes=[],
            pallas_loop_type=pallas_loop_type,
        )
    return helion.Config(block_sizes=[])


def _rank_values(rank: int, width: int = _WIDTH) -> np.ndarray:
    return rank * 1000 + np.arange(width, dtype=np.float32)


def _rank_pipeline_values(rank: int) -> np.ndarray:
    steps = np.arange(_PIPELINE_STEPS, dtype=np.float32)[:, None] * 100
    columns = np.arange(_WIDTH, dtype=np.float32)[None, :]
    return rank * 1000 + steps + columns


def _rank_gather_values(rank: int) -> np.ndarray:
    values = np.arange(_GATHER_ROWS * _WIDTH, dtype=np.float32)
    return rank * 10000 + values.reshape(_GATHER_ROWS, _WIDTH)


def _expected_cyclic_destination(world_size: int) -> np.ndarray:
    expected = np.full((world_size, 2, _WIDTH), -1.0, dtype=np.float32)
    for rank in range(world_size):
        expected[rank, _REMOTE_SLOT] = _rank_values((rank - 1) % world_size)
    return expected


def _expected_pipeline_destination(world_size: int) -> np.ndarray:
    return np.stack(
        [_rank_pipeline_values((rank - 1) % world_size) for rank in range(world_size)]
    )


def _expected_all_gather(world_size: int) -> np.ndarray:
    return np.stack([_rank_gather_values(rank) for rank in range(world_size)])


def _ring_peers(rank: int, world_size: int) -> np.ndarray:
    return np.asarray(
        [[(rank + 1) % world_size, (rank - 1) % world_size]], dtype=np.int32
    )


def _ring_slots(rank: int, world_size: int) -> np.ndarray:
    return np.asarray(
        [[(rank - step) % world_size for step in range(world_size - 1)]],
        dtype=np.int32,
    )


def _empty_gather_destination(world_size: int) -> np.ndarray:
    return np.full((world_size, 1, _GATHER_ROWS, _WIDTH), -1.0, dtype=np.float32)


@helion.kernel(
    static_shapes=True,
    config=helion.Config(block_sizes=[]),
)
def _one_shot_remote_copy(
    src: torch.Tensor,
    dst: torch.Tensor,
    peers: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Issue one unconditional copy and wait for its matching incoming row."""
    for _program in hl.grid(1):
        copy = hl.make_async_remote_copy(
            src,
            [0, 0],
            peers[0, 0],
            dst=dst,
            dst_index=[0, slots[0, 0]],
        )
        copy.start()
        copy.wait()
    return dst


@helion.kernel(
    static_shapes=True,
    config=_distributed_test_config("fori_loop"),
)
def _reused_descriptor_pipeline_copy(
    src: torch.Tensor,
    dst: torch.Tensor,
    peers: torch.Tensor,
) -> torch.Tensor:
    """Reuse one static descriptor site across a pipelined sequence of rows."""
    num_steps = hl.specialize(src.size(1))
    for _program in hl.grid(1):
        hl.remote_barrier(peers[0, :])
        for step in hl.tile(num_steps, block_size=1):
            copy = hl.make_async_remote_copy(
                src,
                [0, step.begin],
                peers[0, 0],
                dst=dst,
                dst_index=[0, step.begin],
            )
            if step.begin > 0:
                copy.wait()
            copy.start()
            if step.begin == num_steps - 1:
                copy.wait()
    return dst


@helion.kernel(
    static_shapes=True,
    config=_distributed_test_config("fori_loop"),
)
def _ring_all_gather(
    local_values: torch.Tensor,
    gathered: torch.Tensor,
    peers: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Seed the local shard, then forward each newly received shard clockwise."""
    num_steps = hl.specialize(slots.size(1))
    for _program in hl.grid(1):
        gathered[slots[0, :1], 0, :, :] = local_values[:, :, :]
        hl.remote_barrier(peers[0, :])
        for step in hl.tile(num_steps, block_size=1):
            slot = slots[0, step.begin]
            copy = hl.make_async_remote_copy(
                gathered,
                [slot, 0],
                peers[0, 0],
                dst=gathered,
                dst_index=[slot, 0],
            )
            if step.begin > 0:
                copy.wait()
            copy.start()
            if step.begin == num_steps - 1:
                copy.wait()
    return gathered


@helion.kernel(
    static_shapes=True,
    config=_distributed_test_config("emit_pipeline"),
)
def _pipeline_remote_copy(
    src: torch.Tensor,
    stage: torch.Tensor,
    dst: torch.Tensor,
    peers: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    for _program in hl.grid(1):
        hl.remote_barrier(peers[0, 0])
        for tile in hl.tile(src.size(1), block_size=1):
            stage[:, :, :] = torch.sum(src[:, tile, :, :], dim=1)
            position = positions[0, tile.begin]
            copy = hl.make_async_remote_copy(
                stage,
                [0],
                peers[0, tile.begin],
                dst=dst,
                dst_index=[0, position, tile.begin],
            )
            copy.start()
            copy.wait()
    return stage, dst


@helion.kernel(
    static_shapes=True,
    config=_distributed_test_config("fori_loop"),
)
def _parent_store_nested_remote_copy(
    src: torch.Tensor,
    exchange: torch.Tensor,
    peers: torch.Tensor,
) -> torch.Tensor:
    """Initialize a resident exchange buffer before copying from a child loop."""
    num_steps = hl.specialize(src.size(1))
    for _program in hl.grid(1):
        for step in hl.tile(num_steps, block_size=1):
            exchange[0, 0, step.begin, :] = src[0, step.begin, :]
            for peer_step in hl.tile(1, block_size=1):
                copy = hl.make_async_remote_copy(
                    exchange,
                    [0, peer_step.begin, step.begin],
                    peers[0, 0],
                    dst=exchange,
                    dst_index=[0, peer_step.begin + 1, step.begin],
                )
                copy.start()
                copy.wait()
    return exchange


@helion.kernel(
    static_shapes=True,
    config=_distributed_test_config("fori_loop"),
)
def _route_forward_then_consume(
    src: torch.Tensor,
    routed: torch.Tensor,
    output: torch.Tensor,
    peers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    for _program in hl.grid(1):
        hl.remote_barrier(peers[0, 0])
        for tile in hl.tile(src.size(0), block_size=1):
            ingress = hl.make_async_remote_copy(
                src,
                [tile.begin],
                peers[0, 0],
                dst=routed,
                dst_index=[0, 0, tile.begin],
            )
            ingress.start()
            ingress.wait()
            forward = hl.make_async_remote_copy(routed, [0, 0, tile.begin], peers[0, 0])
            forward.start()
            forward.wait()
        for token in hl.tile(routed.size(3), block_size=8):
            output[:, :, :, token, :] = routed[:, :, :, token, :] + 1
    return routed, output


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

    def _make_symmetric_buffer(self, shape: tuple[int, ...]) -> torch.Tensor:
        dst = symm_mem.empty(*shape, dtype=torch.float32, device=self.device)
        group = dist.group.WORLD
        assert group is not None
        dst_handle = symm_mem.rendezvous(dst, group=group)
        self.assertEqual(dst_handle.world_size, self.world_size)
        return dst

    def test_one_shot_remote_copy(self) -> None:
        self._init_process()
        try:
            src = torch.from_numpy(_rank_values(self.rank)).to(self.device)
            src = src.reshape(1, 1, _WIDTH)
            dst = self._make_symmetric_buffer((1, 2, _WIDTH))
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
                dist.barrier()
                result = _one_shot_remote_copy(src, dst, peers, slots)
                torch.cuda.synchronize()
                torch.testing.assert_close(result[0], expected)
        finally:
            self._cleanup_process()

    def test_reused_descriptor_pipeline(self) -> None:
        self._init_process()
        try:
            src = torch.from_numpy(_rank_pipeline_values(self.rank)).to(self.device)
            src = src.reshape(1, _PIPELINE_STEPS, _WIDTH)
            dst = self._make_symmetric_buffer(src.shape)
            peers = torch.tensor(
                [
                    [
                        (self.rank + 1) % self.world_size,
                        (self.rank - 1) % self.world_size,
                    ]
                ],
                dtype=torch.int32,
                device=self.device,
            )
            expected = torch.from_numpy(
                _expected_pipeline_destination(self.world_size)[self.rank]
            ).to(self.device)
            for _invocation in range(2):
                dst.fill_(-1)
                dist.barrier()
                result = _reused_descriptor_pipeline_copy(src, dst, peers)
                torch.cuda.synchronize()
                torch.testing.assert_close(result[0], expected)
        finally:
            self._cleanup_process()

    def test_ring_all_gather(self) -> None:
        self._init_process()
        try:
            gathered = self._make_symmetric_buffer(
                (self.world_size, 1, _GATHER_ROWS, _WIDTH)
            )
            peers = torch.from_numpy(_ring_peers(self.rank, self.world_size)).to(
                self.device
            )
            slots = torch.from_numpy(_ring_slots(self.rank, self.world_size)).to(
                self.device
            )
            local_values = (
                torch.from_numpy(_rank_gather_values(self.rank))
                .to(self.device)
                .unsqueeze(0)
            )
            expected = torch.from_numpy(_expected_all_gather(self.world_size)).to(
                self.device
            )
            for _invocation in range(2):
                gathered.fill_(-1)
                dist.barrier()
                result = _ring_all_gather(local_values, gathered, peers, slots)
                torch.cuda.synchronize()
                torch.testing.assert_close(result[:, 0], expected)
        finally:
            self._cleanup_process()


@onlyBackends(["pallas"])
class TestRemoteCopyJaxRuntime(TestCase):
    @staticmethod
    def _run_one_shot_copy(mesh, mesh_axis, kernel_fn) -> None:
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

    @staticmethod
    def _run_reused_descriptor_pipeline(mesh, kernel_fn) -> None:
        import jax
        import jax.numpy as jnp

        world_size = mesh.devices.size
        partition = jax.sharding.PartitionSpec
        input_specs = (
            partition("peer", None, None),
            partition("peer", None, None),
            partition("peer", None),
        )
        src = jnp.stack(
            [jnp.asarray(_rank_pipeline_values(rank)) for rank in range(world_size)]
        )
        dst = jnp.full_like(src, -1.0)
        ranks = jnp.arange(world_size, dtype=jnp.int32)
        peers = jnp.stack(((ranks + 1) % world_size, (ranks - 1) % world_size), axis=1)
        inputs = tuple(
            jax.device_put(value, jax.sharding.NamedSharding(mesh, spec))
            for value, spec in zip((src, dst, peers), input_specs, strict=True)
        )
        copy = jax.jit(
            jax.shard_map(
                kernel_fn,
                mesh=mesh,
                in_specs=input_specs,
                out_specs=partition("peer", None, None),
                check_vma=False,
            )
        )
        expected = _expected_pipeline_destination(world_size)
        for _invocation in range(2):
            result = jax.block_until_ready(copy(*inputs))
            np.testing.assert_array_equal(np.asarray(result), expected)

    @staticmethod
    def _run_ring_all_gather(mesh) -> None:
        import jax
        import jax.numpy as jnp

        world_size = int(mesh.devices.size)
        partition = jax.sharding.PartitionSpec
        local_values_spec = partition("peer", None, None)
        gathered_spec = partition(None, "peer", None, None)
        metadata_spec = partition("peer", None)
        local_values = jnp.stack(
            [jnp.asarray(_rank_gather_values(rank)) for rank in range(world_size)]
        )
        gathered = jnp.full(
            (world_size, world_size, _GATHER_ROWS, _WIDTH),
            -1.0,
            dtype=jnp.float32,
        )
        peers = jnp.concatenate(
            [jnp.asarray(_ring_peers(rank, world_size)) for rank in range(world_size)]
        )
        slots = jnp.concatenate(
            [jnp.asarray(_ring_slots(rank, world_size)) for rank in range(world_size)]
        )
        input_specs = (
            local_values_spec,
            gathered_spec,
            metadata_spec,
            metadata_spec,
        )
        inputs = tuple(
            jax.device_put(value, jax.sharding.NamedSharding(mesh, spec))
            for value, spec in zip(
                (local_values, gathered, peers, slots), input_specs, strict=True
            )
        )
        all_gather = jax.jit(
            jax.shard_map(
                _ring_all_gather.jax_fn,
                mesh=mesh,
                in_specs=input_specs,
                out_specs=gathered_spec,
                check_vma=False,
            )
        )
        expected = np.broadcast_to(
            _expected_all_gather(world_size)[:, None, :, :],
            (world_size, world_size, _GATHER_ROWS, _WIDTH),
        )
        for _invocation in range(2):
            result = jax.block_until_ready(all_gather(*inputs))
            np.testing.assert_array_equal(np.asarray(result), expected)

    @skipIfPallasInterpret("remote copies require physical TPU devices")
    def test_one_shot_remote_copy(self) -> None:
        import jax

        devices = [device for device in jax.local_devices() if device.platform == "tpu"]
        if len(devices) < 2:
            self.skipTest("requires at least two TPU devices")
        mesh = jax.make_mesh((2,), ("peer",), devices=devices[:2])
        self._run_one_shot_copy(mesh, "peer", _one_shot_remote_copy.jax_fn)

    @skipIfPallasInterpret("remote copies require physical TPU devices")
    def test_multiaxis_flat_logical_peers(self) -> None:
        import jax

        devices = [device for device in jax.local_devices() if device.platform == "tpu"]
        if len(devices) < 8:
            self.skipTest("requires at least eight TPU devices")
        mesh = jax.make_mesh((2, 4), ("data", "peer"), devices=devices[:8])
        self._run_one_shot_copy(
            mesh,
            ("data", "peer"),
            _one_shot_remote_copy.jax_fn,
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
        source = _one_shot_remote_copy.bind(args).to_code(
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
            self._run_one_shot_copy(mesh, "peer", module._one_shot_remote_copy)
        finally:
            sys.modules.pop(name, None)

    @skipIfPallasInterpret("remote-copy pipelines require TPU VMEM lowering")
    def test_reused_descriptor_pipeline(self) -> None:
        import jax

        devices = [device for device in jax.local_devices() if device.platform == "tpu"]
        if len(devices) < 4:
            self.skipTest("requires at least four TPU devices")
        mesh = jax.make_mesh((4,), ("peer",), devices=devices[:4])
        self._run_reused_descriptor_pipeline(
            mesh, _reused_descriptor_pipeline_copy.jax_fn
        )

    @skipIfPallasInterpret("ring all-gather requires TPU remote DMA lowering")
    def test_ring_all_gather(self) -> None:
        import jax

        devices = [device for device in jax.local_devices() if device.platform == "tpu"]
        if len(devices) < 4:
            self.skipTest("requires at least four TPU devices")
        world_size = min(8, len(devices))
        mesh = jax.make_mesh((world_size,), ("peer",), devices=devices[:world_size])
        self._run_ring_all_gather(mesh)

    @skipIfPallasInterpret("remote-copy pipelines require TPU VMEM lowering")
    def test_computed_pipeline_copy(self) -> None:
        import jax
        import jax.numpy as jnp

        devices = [device for device in jax.local_devices() if device.platform == "tpu"]
        if len(devices) < 2:
            self.skipTest("requires at least two TPU devices")
        mesh = jax.make_mesh((2,), ("peer",), devices=devices[:2])
        partition = jax.sharding.PartitionSpec
        input_specs = (
            partition("peer", None, None, None),
            partition("peer", None, None),
            partition("peer", None, None, None, None),
            partition("peer", None),
            partition("peer", None),
        )
        rank_values = jnp.arange(2, dtype=jnp.float32)[:, None, None, None]
        tile_values = jnp.arange(4, dtype=jnp.float32)[None, :, None, None]
        src = jnp.broadcast_to(rank_values * 10 + tile_values, (2, 4, 16, 128))
        inputs = (
            src,
            jnp.zeros((2, 16, 128), dtype=jnp.float32),
            jnp.full((2, 2, 4, 16, 128), -7.0, dtype=jnp.float32),
            jnp.broadcast_to(1 - jnp.arange(2, dtype=jnp.int32)[:, None], (2, 4)),
            jnp.broadcast_to(jnp.arange(2, dtype=jnp.int32)[:, None], (2, 4)),
        )
        inputs = tuple(
            jax.device_put(value, jax.sharding.NamedSharding(mesh, spec))
            for value, spec in zip(inputs, input_specs, strict=True)
        )
        copy = jax.jit(
            jax.shard_map(
                _pipeline_remote_copy.jax_fn,
                mesh=mesh,
                in_specs=input_specs,
                out_specs=(
                    partition("peer", None, None),
                    partition("peer", None, None, None, None),
                ),
                check_vma=False,
            )
        )
        expected = np.full((2, 2, 4, 16, 128), -7.0, dtype=np.float32)
        src_host = np.asarray(src)
        expected[0, 1] = src_host[1]
        expected[1, 0] = src_host[0]
        for _invocation in range(2):
            _stage, result = jax.block_until_ready(copy(*inputs))
            np.testing.assert_array_equal(np.asarray(result), expected)

    @skipIfPallasInterpret("nested remote copies require TPU DMA lowering")
    def test_parent_store_nested_remote_copy(self) -> None:
        import jax
        import jax.numpy as jnp

        devices = [device for device in jax.local_devices() if device.platform == "tpu"]
        if len(devices) < 4:
            self.skipTest("requires at least four TPU devices")
        world_size = 4
        mesh = jax.make_mesh((world_size,), ("peer",), devices=devices[:world_size])
        partition = jax.sharding.PartitionSpec
        src_spec = partition("peer", None, None)
        exchange_spec = partition("peer", None, None, None)
        peer_spec = partition("peer", None)
        src = jnp.stack(
            [jnp.asarray(_rank_pipeline_values(rank)) for rank in range(world_size)]
        )
        inputs = (
            src,
            jnp.full(
                (world_size, 2, _PIPELINE_STEPS, _WIDTH),
                -1.0,
                dtype=jnp.float32,
            ),
            ((jnp.arange(world_size, dtype=jnp.int32) + 1) % world_size)[:, None],
        )
        input_specs = (src_spec, exchange_spec, peer_spec)
        inputs = tuple(
            jax.device_put(value, jax.sharding.NamedSharding(mesh, spec))
            for value, spec in zip(inputs, input_specs, strict=True)
        )
        copy = jax.jit(
            jax.shard_map(
                _parent_store_nested_remote_copy.jax_fn,
                mesh=mesh,
                in_specs=input_specs,
                out_specs=exchange_spec,
                check_vma=False,
            )
        )
        result = np.asarray(jax.block_until_ready(copy(*inputs)))
        np.testing.assert_array_equal(result[:, 0], np.asarray(src))
        np.testing.assert_array_equal(
            result[:, 1], _expected_pipeline_destination(world_size)
        )

    @skipIfPallasInterpret("remote HBM buffers require TPU DMA lowering")
    def test_route_forward_then_local_consume(self) -> None:
        import jax
        import jax.numpy as jnp

        devices = [device for device in jax.local_devices() if device.platform == "tpu"]
        if len(devices) < 2:
            self.skipTest("requires at least two TPU devices")
        mesh = jax.make_mesh((2,), ("core",), devices=devices[:2])
        partition = jax.sharding.PartitionSpec
        src_spec = partition("core", None, None)
        routed_spec = partition("core", None, None, None, None)
        peer_spec = partition("core", None)
        values = jnp.arange(8 * 16 * 128, dtype=jnp.float32).reshape(8, 16, 128)
        routed_shape = (2, 1, 4, 16, 128)
        inputs = (
            values,
            jnp.zeros(routed_shape, dtype=values.dtype),
            jnp.zeros(routed_shape, dtype=values.dtype),
            (1 - jnp.arange(2, dtype=jnp.int32))[:, None],
        )
        input_specs = (src_spec, routed_spec, routed_spec, peer_spec)
        inputs = tuple(
            jax.device_put(value, jax.sharding.NamedSharding(mesh, spec))
            for value, spec in zip(inputs, input_specs, strict=True)
        )
        route = jax.jit(
            jax.shard_map(
                _route_forward_then_consume.jax_fn,
                mesh=mesh,
                in_specs=input_specs,
                out_specs=(routed_spec, routed_spec),
                check_vma=False,
            )
        )
        routed, output = jax.block_until_ready(route(*inputs))
        expected = np.asarray(values).reshape(2, 4, 16, 128)[:, None, :, :, :]
        np.testing.assert_array_equal(np.asarray(routed), expected)
        np.testing.assert_array_equal(np.asarray(output), expected + 1)


def _remote_copy_torch_tpu_worker(rank: int, world_size: int, master_port: int) -> None:
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
    from torch_tpu import _loader as torch_tpu_loader  # pyrefly: ignore[missing-import]

    torch_tpu_loader.load()

    from torch_tpu._internal import pallas  # pyrefly: ignore[missing-import]

    dist.init_process_group(backend="tpu_dist")
    try:
        partition = jax.sharding.PartitionSpec
        mesh = jax.make_mesh((world_size,), ("peer",))

        pipeline_specs = (
            partition("peer", None, None),
            partition("peer", None, None),
            partition("peer", None),
        )
        exchange = jax.shard_map(
            _reused_descriptor_pipeline_copy.jax_fn,
            mesh=mesh,
            in_specs=pipeline_specs,
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
            input_partition_specs=pipeline_specs,
        )

        device = torch.device("tpu")
        src = torch.from_numpy(_rank_pipeline_values(rank)).to(device).unsqueeze(0)
        dst = torch.zeros_like(src)
        peers = torch.from_numpy(_ring_peers(rank, world_size)).to(device)
        result = exchange_op(src, dst, peers)
        expected = torch.from_numpy(
            _rank_pipeline_values((rank - 1) % world_size)
        ).unsqueeze(0)
        assert result.shape == (1, _PIPELINE_STEPS, _WIDTH)
        torch.testing.assert_close(result.cpu(), expected)

        local_values_spec = partition("peer", None, None)
        gathered_spec = partition(None, "peer", None, None)
        metadata_spec = partition("peer", None)
        gather_specs = (
            local_values_spec,
            gathered_spec,
            metadata_spec,
            metadata_spec,
        )
        gather = jax.shard_map(
            _ring_all_gather.jax_fn,
            mesh=mesh,
            in_specs=gather_specs,
            out_specs=gathered_spec,
            check_vma=False,
        )

        def gather_jax(
            local_values: jax.Array,
            gathered: jax.Array,
            gather_peers: jax.Array,
            slots: jax.Array,
        ) -> jax.Array:
            return gather(local_values, gathered, gather_peers, slots)

        gather_jax.__annotations__ = {
            "local_values": jax.Array,
            "gathered": jax.Array,
            "gather_peers": jax.Array,
            "slots": jax.Array,
            "return": jax.Array,
        }
        gather_op = pallas.jax_op(
            "pallas::helion_ring_all_gather_test",
            gather_jax,
            mesh=mesh,
            input_partition_specs=gather_specs,
        )
        local_values = (
            torch.from_numpy(_rank_gather_values(rank)).to(device).unsqueeze(0)
        )
        gathered = torch.from_numpy(_empty_gather_destination(world_size)).to(device)
        slots = torch.from_numpy(_ring_slots(rank, world_size)).to(device)
        result = gather_op(local_values, gathered, peers, slots)
        expected = torch.from_numpy(_expected_all_gather(world_size)).unsqueeze(1)
        assert result.shape == (world_size, 1, _GATHER_ROWS, _WIDTH)
        torch.testing.assert_close(result.cpu(), expected)
    finally:
        dist.destroy_process_group()


def _run_torch_tpu_multiprocess() -> None:
    import portpicker  # pyrefly: ignore[missing-import]
    import torch.multiprocessing as mp
    from torch_tpu._internal.distributed.launchers import (  # pyrefly: ignore[missing-import]
        singlehost_wrapper,
    )

    world_size = 4
    singlehost_wrapper.prepare_tpu_environment(world_size=world_size)
    master_port = portpicker.pick_unused_port()
    mp.spawn(
        _remote_copy_torch_tpu_worker,
        args=(world_size, master_port),
        nprocs=world_size,
        join=True,
    )


@unittest.skipUnless(
    os.environ.get("HELION_TEST_TORCH_TPU_MULTIPROCESS") == "1"
    and os.environ.get("HELION_BACKEND") == "pallas"
    and importlib.util.find_spec("torch_tpu") is not None,
    "requires a dedicated TorchTPU multiprocess test invocation",
)
class TestRemoteCopyTorchTpuRuntime(unittest.TestCase):
    def test_one_process_per_device_remote_copy(self) -> None:
        env = os.environ.copy()
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), _TORCH_TPU_RUNNER_ARG],
            check=True,
            env=env,
            timeout=240,
        )


if __name__ == "__main__" and _TORCH_TPU_RUNNER_ARG in sys.argv:
    _run_torch_tpu_multiprocess()
