from __future__ import annotations

import functools

import torch
import triton

from .config import Config as Config
from .kernel import Kernel as Kernel
from .kernel import kernel as kernel


def _alloc_fn(size: int, alignment: int, stream: int | None) -> torch.Tensor:
    return torch.empty(size, device="cuda", dtype=torch.int8)


@functools.cache
def set_triton_allocator() -> None:
    try:
        from triton.runtime._allocation import NullAllocator
        from triton.runtime._allocation import _allocator

        if not isinstance(_allocator, NullAllocator):
            return
    except ImportError:
        pass
    triton.set_allocator(_alloc_fn)
