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
_HBM_TILE = 8


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
    config=helion.Config(block_sizes=[]),
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
    config=helion.Config(block_sizes=[]),
)
def _remote_copy_with_unrelated_loop_input(
    src: torch.Tensor,
    dst: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    peers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep an ordinary loop input independent of remote-copy placement."""
    num_steps = hl.specialize(src.size(1))
    for _program in hl.grid(1):
        for step in hl.tile(num_steps, block_size=1):
            copy = hl.make_async_remote_copy(
                src,
                [0, step.begin],
                peers[0, 0],
                dst=dst,
                dst_index=[0, step.begin],
            )
            copy.start()
            copy.wait()
            output[0, step, :] = dst[0, step, :] + bias[0, :]
    return dst, output


@helion.kernel(
    static_shapes=True,
    config=helion.Config(block_sizes=[]),
)
def _ordered_reused_descriptor_pipeline_copy(
    src: torch.Tensor,
    dst: torch.Tensor,
    peers: torch.Tensor,
    gate: torch.Tensor,
    rank: torch.Tensor,
) -> torch.Tensor:
    """Force two completions to reach rank 0 before it consumes either."""
    num_steps = hl.specialize(src.size(1))
    for _program in hl.grid(1):
        hl.remote_barrier(peers[0, :])
        gate_copy = hl.make_async_remote_copy(
            gate,
            [0],
            peers[0, 0],
        )
        # Ranks 1-3 advance the ring while rank 0 waits. Rank 3 releases the
        # gate only after its second put into rank 0's reused completion slot.
        if rank[0] == 0:
            gate_copy.wait()
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
            if step.begin == 1:
                if rank[0] == 3:
                    gate_copy.start()
            if step.begin == num_steps - 1:
                copy.wait()
    return dst


@helion.kernel(
    static_shapes=True,
    config=helion.Config(block_sizes=[]),
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
        # Pallas scatter requires a rank-1 index; make that rank explicit so
        # Triton does not scalarize a one-element slice underneath the store.
        local_slot = slots[0, 0] + hl.arange(1)
        gathered[local_slot, 0, :, :] = local_values[:, :, :]
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
    config=helion.Config(block_sizes=[]),
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
    config=helion.Config(block_sizes=[]),
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
    config=helion.Config(block_sizes=[]),
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


@helion.kernel(
    static_shapes=False,
    config=helion.Config(block_sizes=[_HBM_TILE]),
)
def _computed_tiled_hbm_copy(
    src: torch.Tensor,
    dst: torch.Tensor,
    peers: torch.Tensor,
    ranks: torch.Tensor,
) -> torch.Tensor:
    """Send computed token tiles directly into a remote HBM output."""
    for _program in hl.grid(1):
        for tile in hl.tile(src.size(1)):
            reduced = torch.sum(src[0, tile, :, :], dim=1)
            copy = hl.make_async_remote_copy(
                reduced,
                [],
                peers[0, 0],
                dst=dst,
                dst_index=[0, ranks[0, 0], tile],
            )
            copy.start()
            copy.wait()
    return dst


@helion.kernel(
    static_shapes=False,
    config=helion.Config(block_sizes=[_HBM_TILE]),
)
def _pipelined_remote_hbm_consume(
    src: torch.Tensor,
    dst: torch.Tensor,
    output: torch.Tensor,
    peers: torch.Tensor,
    ranks: torch.Tensor,
) -> torch.Tensor:
    """Exchange tile n while consuming the completed HBM slab for tile n-1."""
    tokens = src.size(1)
    for _program in hl.grid(1):
        for tile in hl.tile(tokens):
            reduced = torch.sum(src[0, tile, :, :], dim=1)
            self_copy = hl.make_async_remote_copy(
                reduced,
                [],
                peers[0, 0],
                dst=dst,
                dst_index=[0, ranks[0, 0], tile],
            )
            peer_copy = hl.make_async_remote_copy(
                reduced,
                [],
                peers[0, 1],
                dst=dst,
                dst_index=[0, ranks[0, 0], tile],
            )
            if tile.begin > 0:
                self_copy.wait()
                peer_copy.wait()
            self_copy.start()
            peer_copy.start()

            if tile.begin > 0:
                for row in hl.tile(_HBM_TILE, block_size=1):
                    previous_row = tile.begin - _HBM_TILE + row.begin
                    previous = dst[0, :, previous_row, :]
                    output[0, previous_row, :] = torch.sum(previous, dim=0)

            if tile.begin + _HBM_TILE >= tokens:
                self_copy.wait()
                peer_copy.wait()
                current = dst[0, :, tile, :]
                output[0, tile, :] = torch.sum(current, dim=0)
    return output


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

    def _init_process(self) -> dist.Store:
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            device_id=self.device,
        )
        self.addCleanup(self._cleanup_process)
        torch.distributed.distributed_c10d._set_pg_timeout(
            timedelta(seconds=60), dist.group.WORLD
        )
        symm_mem.set_backend("NVSHMEM")
        return store

    def _cleanup_process(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.destroy_process_group()

    def _make_symmetric_buffer(self, shape: tuple[int, ...]) -> torch.Tensor:
        dst = symm_mem.empty(*shape, dtype=torch.float32, device=self.device)
        group = dist.group.WORLD
        assert group is not None
        dst_handle = symm_mem.rendezvous(dst, group=group)
        self.assertEqual(dst_handle.world_size, self.world_size)
        return dst

    def test_one_shot_remote_copy(self) -> None:
        store = self._init_process()
        src = torch.from_numpy(_rank_values(self.rank)).to(self.device)
        src = src.reshape(1, 1, _WIDTH)
        dst = self._make_symmetric_buffer((1, 2, _WIDTH))
        peers = torch.tensor(
            [[(self.rank + 1) % self.world_size]],
            dtype=torch.int32,
            device=self.device,
        )
        slots = torch.full((1, 1), _REMOTE_SLOT, dtype=torch.int32, device=self.device)
        expected = torch.from_numpy(
            _expected_cyclic_destination(self.world_size)[self.rank]
        ).to(self.device)
        for _invocation in range(2):
            dst.fill_(-1)
            dist.barrier()
            if self.rank == 0 and _invocation == 1:
                # Enter the cached launch only after rank 3 has signaled us.
                store.wait(["rank_3_completed_one_shot_copy"])
            result = _one_shot_remote_copy(src, dst, peers, slots)
            torch.cuda.synchronize()
            if self.rank == self.world_size - 1 and _invocation == 1:
                store.set("rank_3_completed_one_shot_copy", "1")
            torch.testing.assert_close(result[0], expected)

    def test_reused_descriptor_pipeline(self) -> None:
        self._init_process()
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

    def test_queued_reused_descriptor_completions(self) -> None:
        self._init_process()
        src = torch.from_numpy(_rank_pipeline_values(self.rank)).to(self.device)
        src = src.reshape(1, _PIPELINE_STEPS, _WIDTH)
        dst = self._make_symmetric_buffer(src.shape)
        gate = self._make_symmetric_buffer((1,))
        gate.fill_(self.rank)
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
        rank = torch.tensor([self.rank], dtype=torch.int32, device=self.device)
        expected = torch.from_numpy(
            _expected_pipeline_destination(self.world_size)[self.rank]
        ).to(self.device)
        for _invocation in range(2):
            dst.fill_(-1)
            dist.barrier()
            result = _ordered_reused_descriptor_pipeline_copy(
                src, dst, peers, gate, rank
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(result[0], expected)

    def test_ring_all_gather(self) -> None:
        self._init_process()
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

    def test_remote_copy_does_not_reclassify_unrelated_loop_input(self) -> None:
        self._init_process()
        src = torch.from_numpy(_rank_pipeline_values(self.rank)).to(self.device)
        src = src.reshape(1, _PIPELINE_STEPS, _WIDTH)
        dst = self._make_symmetric_buffer(src.shape)
        bias = torch.arange(_WIDTH, dtype=torch.float32, device=self.device)
        bias = bias.reshape(1, _WIDTH) * 0.25
        output = torch.full_like(src, -1)
        peers = torch.tensor(
            [[(self.rank + 1) % self.world_size]],
            dtype=torch.int32,
            device=self.device,
        )
        expected = torch.from_numpy(
            _expected_pipeline_destination(self.world_size)[self.rank]
        ).to(self.device)
        for _invocation in range(2):
            dst.fill_(-1)
            output.fill_(-1)
            dist.barrier()
            result_dst, result_output = _remote_copy_with_unrelated_loop_input(
                src, dst, bias, output, peers
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(result_dst[0], expected)
            torch.testing.assert_close(result_output[0], expected + bias)

    def test_computed_pipeline_copy(self) -> None:
        self._init_process()
        rank_value = torch.tensor(
            self.rank * 10, dtype=torch.float32, device=self.device
        )
        tile_values = torch.arange(
            _PIPELINE_STEPS, dtype=torch.float32, device=self.device
        ).reshape(1, _PIPELINE_STEPS, 1, 1)
        src = (
            (rank_value + tile_values)
            .expand(1, _PIPELINE_STEPS, 16, _WIDTH)
            .contiguous()
        )
        stage = torch.zeros(1, 16, _WIDTH, device=self.device)
        dst = self._make_symmetric_buffer(
            (1, self.world_size, _PIPELINE_STEPS, 16, _WIDTH)
        )
        peers = torch.full(
            (1, _PIPELINE_STEPS),
            (self.rank + 1) % self.world_size,
            dtype=torch.int32,
            device=self.device,
        )
        positions = torch.full(
            (1, _PIPELINE_STEPS),
            self.rank,
            dtype=torch.int32,
            device=self.device,
        )
        previous_rank = (self.rank - 1) % self.world_size
        previous_values = (previous_rank * 10 + tile_values).expand_as(src)
        expected = torch.full_like(dst, -7)
        expected[0, previous_rank] = previous_values[0]
        for _invocation in range(2):
            stage.zero_()
            dst.fill_(-7)
            dist.barrier()
            _stage, result = _pipeline_remote_copy(src, stage, dst, peers, positions)
            torch.cuda.synchronize()
            torch.testing.assert_close(result, expected)

    def test_parent_store_nested_remote_copy(self) -> None:
        self._init_process()
        src = torch.from_numpy(_rank_pipeline_values(self.rank)).to(self.device)
        src = src.reshape(1, _PIPELINE_STEPS, _WIDTH)
        exchange = self._make_symmetric_buffer((1, 2, _PIPELINE_STEPS, _WIDTH))
        peers = torch.tensor(
            [[(self.rank + 1) % self.world_size]],
            dtype=torch.int32,
            device=self.device,
        )
        expected_local = src[0]
        expected_remote = torch.from_numpy(
            _rank_pipeline_values((self.rank - 1) % self.world_size)
        ).to(self.device)
        for _invocation in range(2):
            exchange.fill_(-1)
            dist.barrier()
            result = _parent_store_nested_remote_copy(src, exchange, peers)
            torch.cuda.synchronize()
            torch.testing.assert_close(result[0, 0], expected_local)
            torch.testing.assert_close(result[0, 1], expected_remote)

    def test_route_forward_then_local_consume(self) -> None:
        self._init_process()
        rows = 4
        values_per_rank = rows * 16 * _WIDTH
        src = torch.arange(
            values_per_rank, dtype=torch.float32, device=self.device
        ).reshape(rows, 16, _WIDTH)
        src = src + self.rank * values_per_rank
        routed = self._make_symmetric_buffer((1, 1, rows, 16, _WIDTH))
        output = torch.full_like(routed, -1)
        peers = torch.tensor(
            [[(self.rank + 1) % self.world_size]],
            dtype=torch.int32,
            device=self.device,
        )
        source_rank = (self.rank - 2) % self.world_size
        expected = torch.arange(
            values_per_rank, dtype=torch.float32, device=self.device
        ).reshape(1, 1, rows, 16, _WIDTH)
        expected = expected + source_rank * values_per_rank
        for _invocation in range(2):
            routed.zero_()
            output.fill_(-1)
            dist.barrier()
            result_routed, result_output = _route_forward_then_consume(
                src, routed, output, peers
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(result_routed, expected)
            torch.testing.assert_close(result_output, expected + 1)

    def test_computed_tile_to_remote_hbm(self) -> None:
        self._init_process()
        tokens = 13
        channels = 5
        values_per_rank = tokens * channels * _WIDTH
        src = torch.arange(
            values_per_rank, dtype=torch.float32, device=self.device
        ).reshape(1, tokens, channels, _WIDTH)
        src = src + self.rank * values_per_rank
        dst = self._make_symmetric_buffer((1, self.world_size, tokens, _WIDTH))
        peers = torch.tensor(
            [[(self.rank + 1) % self.world_size]],
            dtype=torch.int32,
            device=self.device,
        )
        ranks = torch.tensor([[self.rank]], dtype=torch.int32, device=self.device)
        source_rank = (self.rank - 1) % self.world_size
        source = torch.arange(
            values_per_rank, dtype=torch.float32, device=self.device
        ).reshape(tokens, channels, _WIDTH)
        source = source + source_rank * values_per_rank
        expected = torch.full_like(dst, -7)
        expected[0, source_rank] = source.sum(dim=1)
        for _invocation in range(2):
            dst.fill_(-7)
            dist.barrier()
            result = _computed_tiled_hbm_copy(src, dst, peers, ranks)
            torch.cuda.synchronize()
            torch.testing.assert_close(result, expected)

    def test_pipeline_remote_hbm_then_consume_previous_tile(self) -> None:
        self._init_process()
        tokens = 13
        channels = 3
        values_per_rank = tokens * channels * _WIDTH
        src = torch.arange(
            values_per_rank, dtype=torch.float32, device=self.device
        ).reshape(1, tokens, channels, _WIDTH)
        src = src + self.rank * values_per_rank
        dst = self._make_symmetric_buffer((1, 2, tokens, _WIDTH))
        output = torch.full((1, tokens, _WIDTH), -11.0, device=self.device)
        partner = self.rank ^ 1
        peers = torch.tensor(
            [[self.rank, partner]], dtype=torch.int32, device=self.device
        )
        ranks = torch.tensor([[self.rank % 2]], dtype=torch.int32, device=self.device)
        own_reduced = src[0].sum(dim=1)
        partner_values = torch.arange(
            values_per_rank, dtype=torch.float32, device=self.device
        ).reshape(tokens, channels, _WIDTH)
        partner_values = partner_values + partner * values_per_rank
        expected = own_reduced + partner_values.sum(dim=1)
        for _invocation in range(2):
            dst.fill_(-7)
            output.fill_(-11)
            dist.barrier()
            result = _pipelined_remote_hbm_consume(src, dst, output, peers, ranks)
            torch.cuda.synchronize()
            torch.testing.assert_close(result[0], expected)


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
    def _run_remote_copy_with_unrelated_loop_input(mesh) -> None:
        import jax
        import jax.numpy as jnp

        world_size = mesh.devices.size
        partition = jax.sharding.PartitionSpec
        sharded_spec = partition("peer", None, None)
        peer_spec = partition("peer", None)
        replicated_spec = partition(None, None)
        src = jnp.stack(
            [jnp.asarray(_rank_pipeline_values(rank)) for rank in range(world_size)]
        )
        bias = jnp.arange(_WIDTH, dtype=jnp.float32)[None, :] * 0.25
        input_specs = (
            sharded_spec,
            sharded_spec,
            replicated_spec,
            sharded_spec,
            peer_spec,
        )
        inputs = (
            src,
            jnp.full_like(src, -1.0),
            bias,
            jnp.full_like(src, -1.0),
            ((jnp.arange(world_size, dtype=jnp.int32) + 1) % world_size)[:, None],
        )
        inputs = tuple(
            jax.device_put(value, jax.sharding.NamedSharding(mesh, spec))
            for value, spec in zip(inputs, input_specs, strict=True)
        )
        copy = jax.jit(
            jax.shard_map(
                _remote_copy_with_unrelated_loop_input.jax_fn,
                mesh=mesh,
                in_specs=input_specs,
                out_specs=(sharded_spec, sharded_spec),
                check_vma=False,
            )
        )
        expected = _expected_pipeline_destination(world_size)
        dst, output = jax.block_until_ready(copy(*inputs))
        np.testing.assert_array_equal(np.asarray(dst), expected)
        np.testing.assert_array_equal(np.asarray(output), expected + np.asarray(bias))

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
        if len(devices) < 2:
            self.skipTest("requires at least two TPU devices")
        # Keep two mesh axes to exercise flattened logical peer IDs while
        # fitting on the two-device TPU slice used by CI.
        mesh = jax.make_mesh((1, 2), ("data", "peer"), devices=devices[:2])
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
        self.addCleanup(sys.modules.pop, name, None)
        exec(compile(source, name, "exec"), module.__dict__)

        import jax

        devices = [device for device in jax.local_devices() if device.platform == "tpu"]
        if len(devices) < 2:
            self.skipTest("requires at least two TPU devices")
        mesh = jax.make_mesh((2,), ("peer",), devices=devices[:2])
        self._run_one_shot_copy(mesh, "peer", module._one_shot_remote_copy)

    @skipIfPallasInterpret("remote-copy pipelines require TPU VMEM lowering")
    def test_reused_descriptor_pipeline(self) -> None:
        import jax

        devices = [device for device in jax.local_devices() if device.platform == "tpu"]
        if len(devices) < 2:
            self.skipTest("requires at least two TPU devices")
        mesh = jax.make_mesh((2,), ("peer",), devices=devices[:2])
        self._run_reused_descriptor_pipeline(
            mesh, _reused_descriptor_pipeline_copy.jax_fn
        )

    @skipIfPallasInterpret("remote copies require physical TPU devices")
    def test_remote_copy_does_not_reclassify_unrelated_loop_input(self) -> None:
        import jax

        devices = [device for device in jax.local_devices() if device.platform == "tpu"]
        if len(devices) < 2:
            self.skipTest("requires at least two TPU devices")
        mesh = jax.make_mesh((2,), ("peer",), devices=devices[:2])
        self._run_remote_copy_with_unrelated_loop_input(mesh)

    @skipIfPallasInterpret("ring all-gather requires TPU remote DMA lowering")
    def test_ring_all_gather(self) -> None:
        import jax

        devices = [device for device in jax.local_devices() if device.platform == "tpu"]
        if len(devices) < 2:
            self.skipTest("requires at least two TPU devices")
        world_size = 2
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
        if len(devices) < 2:
            self.skipTest("requires at least two TPU devices")
        world_size = 2
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

    @skipIfPallasInterpret("remote HBM buffers require TPU DMA lowering")
    def test_computed_tile_to_remote_hbm(self) -> None:
        import types

        import jax
        import jax.numpy as jnp

        devices = [device for device in jax.local_devices() if device.platform == "tpu"]
        if len(devices) < 2:
            self.skipTest("requires at least two TPU devices")

        trace_tokens = 2 * _HBM_TILE
        args = (
            torch.zeros(1, trace_tokens, 3, _WIDTH),
            torch.zeros(1, 2, trace_tokens, _WIDTH),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.zeros(1, 1, dtype=torch.int32),
        )
        source = _computed_tiled_hbm_copy.bind(args).to_code(
            options=helion.OutputCodeOptions(
                allow_helion_deps=False,
                jax_fn=True,
            )
        )
        name = "precompiled_computed_tiled_hbm_copy_test"
        module = types.ModuleType(name)
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        exec(compile(source, name, "exec"), module.__dict__)
        mesh = jax.make_mesh((2,), ("peer",), devices=devices[:2])
        partition = jax.sharding.PartitionSpec
        tokens = 13
        runtime_channels = 5
        values = jnp.arange(
            2 * tokens * runtime_channels * _WIDTH, dtype=jnp.float32
        ).reshape(2, tokens, runtime_channels, _WIDTH)
        inputs = (
            values,
            jnp.full((2, 2, tokens, _WIDTH), -7.0, dtype=jnp.float32),
            (1 - jnp.arange(2, dtype=jnp.int32))[:, None],
            jnp.arange(2, dtype=jnp.int32)[:, None],
        )
        specs = (
            partition("peer", None, None, None),
            partition("peer", None, None, None),
            partition("peer", None),
            partition("peer", None),
        )
        inputs = tuple(
            jax.device_put(value, jax.sharding.NamedSharding(mesh, spec))
            for value, spec in zip(inputs, specs, strict=True)
        )
        copy = jax.jit(
            jax.shard_map(
                module._computed_tiled_hbm_copy,
                mesh=mesh,
                in_specs=specs,
                out_specs=specs[1],
                check_vma=False,
            )
        )
        result = np.asarray(jax.block_until_ready(copy(*inputs)))
        host_values = np.asarray(values)
        expected = np.full((2, 2, tokens, _WIDTH), -7.0, dtype=np.float32)
        expected[0, 1] = host_values[1].sum(axis=1)
        expected[1, 0] = host_values[0].sum(axis=1)
        np.testing.assert_array_equal(result, expected)

    @skipIfPallasInterpret("remote HBM buffers require TPU DMA lowering")
    def test_pipeline_remote_hbm_then_consume_previous_tile(self) -> None:
        import jax
        import jax.numpy as jnp

        devices = [device for device in jax.local_devices() if device.platform == "tpu"]
        if len(devices) < 2:
            self.skipTest("requires at least two TPU devices")

        mesh = jax.make_mesh((2,), ("peer",), devices=devices[:2])
        partition = jax.sharding.PartitionSpec
        src_spec = partition("peer", None, None, None)
        dst_spec = partition("peer", None, None, None)
        output_spec = partition("peer", None, None)
        metadata_spec = partition("peer", None)
        tokens = 13
        channels = 3
        values = jnp.arange(2 * tokens * channels * _WIDTH, dtype=jnp.float32).reshape(
            2, tokens, channels, _WIDTH
        )
        ranks = jnp.arange(2, dtype=jnp.int32)
        inputs = (
            values,
            jnp.full((2, 2, tokens, _WIDTH), -7.0, dtype=jnp.float32),
            jnp.full((2, tokens, _WIDTH), -11.0, dtype=jnp.float32),
            jnp.stack((ranks, 1 - ranks), axis=1),
            ranks[:, None],
        )
        specs = (src_spec, dst_spec, output_spec, metadata_spec, metadata_spec)
        inputs = tuple(
            jax.device_put(value, jax.sharding.NamedSharding(mesh, spec))
            for value, spec in zip(inputs, specs, strict=True)
        )
        consume = jax.jit(
            jax.shard_map(
                _pipelined_remote_hbm_consume.jax_fn,
                mesh=mesh,
                in_specs=specs,
                out_specs=output_spec,
                check_vma=False,
            )
        )
        expected = np.asarray(values).sum(axis=(0, 2))
        expected = np.broadcast_to(expected, (2, tokens, _WIDTH))
        for _invocation in range(2):
            result = np.asarray(jax.block_until_ready(consume(*inputs)))
            np.testing.assert_array_equal(result, expected)


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

    def run_checks() -> None:
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

    run_checks()
    dist.destroy_process_group()


def _run_torch_tpu_multiprocess() -> None:
    import portpicker  # pyrefly: ignore[missing-import]
    import torch.multiprocessing as mp
    from torch_tpu._internal.distributed.launchers import (  # pyrefly: ignore[missing-import]
        singlehost_wrapper,
    )

    world_size = 2
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
