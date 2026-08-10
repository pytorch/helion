from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import helion
import helion.language as hl

if TYPE_CHECKING:
    import torch

_WIDTH = 128
_REMOTE_SLOT = 1
_PIPELINE_STEPS = 4


def _rank_values(rank: int, width: int = _WIDTH) -> np.ndarray:
    return rank * 1000 + np.arange(width, dtype=np.float32)


def _rank_pipeline_values(rank: int) -> np.ndarray:
    steps = np.arange(_PIPELINE_STEPS, dtype=np.float32)[:, None] * 100
    columns = np.arange(_WIDTH, dtype=np.float32)[None, :]
    return rank * 1000 + steps + columns


def _expected_cyclic_destination(world_size: int) -> np.ndarray:
    expected = np.full((world_size, 2, _WIDTH), -1.0, dtype=np.float32)
    for rank in range(world_size):
        expected[rank, _REMOTE_SLOT] = _rank_values((rank - 1) % world_size)
    return expected


def _expected_pipeline_destination(world_size: int) -> np.ndarray:
    return np.stack(
        [_rank_pipeline_values((rank - 1) % world_size) for rank in range(world_size)]
    )


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


def _reusable_cyclic_remote_copy(
    src: torch.Tensor,
    dst: torch.Tensor,
    peers: torch.Tensor,
    signal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reuse one descriptor slot while forwarding a sequence of rows."""
    num_steps = hl.specialize(src.size(1))
    for _program in hl.grid(1):
        hl.remote_barrier(peers[0, :])
        for step in hl.tile(num_steps, block_size=1):
            if signal is None:
                copy = hl.make_async_remote_copy(
                    src,
                    [0, step.begin],
                    peers[0, 0],
                    dst=dst,
                    dst_index=[0, step.begin],
                )
            else:
                copy = hl.make_async_remote_copy(
                    src,
                    [0, step.begin],
                    peers[0, 0],
                    dst=dst,
                    dst_index=[0, step.begin],
                    signal=signal,
                    signal_index=[0],
                )
            copy.wait(step.begin > 0)
            copy.start()
            copy.wait(step.begin == num_steps - 1)
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


_pallas_reusable_cyclic_remote_copy = helion.kernel(
    backend="pallas",
    static_shapes=True,
    config=helion.Config(
        block_sizes=[],
        pallas_loop_type="fori_loop",
        pallas_collective_id=103,
    ),
)(_reusable_cyclic_remote_copy)


_triton_reusable_cyclic_remote_copy = helion.kernel(
    backend="triton",
    static_shapes=True,
    config=helion.Config(block_sizes=[]),
)(_reusable_cyclic_remote_copy)
