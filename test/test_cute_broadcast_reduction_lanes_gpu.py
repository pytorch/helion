"""Exact repetition counts and changed-input CUDA graph replay."""

from __future__ import annotations

import pytest
import torch

from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessCuteAvailable
from helion._testing.cute_broadcast_reduction import BROADCAST_PHYSICAL_CONFIGS
from helion._testing.cute_broadcast_reduction import broadcast_count


@skipIfNotCUDA()
@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize(
    ("copies", "threads", "vector_width"),
    [
        (8, None, None),
        (2, 1, 1),
        (2, 1, 2),
        (3, 1, 1),
        (8, 1, 4),
        *BROADCAST_PHYSICAL_CONFIGS,
    ],
)
def test_broadcast_reduction_counts_every_copy(
    copies: int, threads: int | None, vector_width: int | None
) -> None:
    _check_counts(copies, threads, vector_width)


@skipIfNotCUDA()
@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("reexpand", [1, 2])
def test_reexpansion_preserves_counts_and_replay(reexpand: int) -> None:
    _check_counts(8, 4, 1, reexpand)


def _check_counts(
    copies: int, threads: int | None, vector_width: int | None, reexpand: int = 0
) -> None:
    x = torch.tensor([-2, -1, 0, 1, 2, -3, 4, 0], device=DEVICE, dtype=torch.float32)
    bound = broadcast_count.bind((x, copies, reexpand))
    config = bound.config_spec.default_config()
    if threads is not None:
        config.config.update(
            num_threads=[2, threads], cute_vector_widths=[1, vector_width]
        )
    compiled = bound.compile_config(config)
    actual = compiled(x, copies, reexpand)
    torch.testing.assert_close(actual, (x > 0).long() * copies, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = compiled(x, copies, reexpand)
    x.neg_().add_(0.5)
    replay.fill_(-123)
    graph.replay()
    torch.testing.assert_close(replay, (x > 0).long() * copies, rtol=0, atol=0)
