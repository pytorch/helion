"""Explicit resource lifetimes for CUDA graph replay tests."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def cuda_graph_for_replay() -> Iterator[torch.cuda.CUDAGraph]:
    """Reset test graph resources before later captures can collect Python cycles."""
    graph = torch.cuda.CUDAGraph()
    try:
        yield graph
    finally:
        graph.reset()
