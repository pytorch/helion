from __future__ import annotations

import torch
from torch._C._distributed_c10d import _SymmetricMemory
import torch.distributed as dist


def max_num_blocks_for_symm_mem() -> int:
    """
    Return the max number of blocks allowed due to the restriction of
    signal pad size in symm memory.
    """
    assert dist.is_initialized()
    signal_pad_size = _SymmetricMemory.signal_pad_size
    return signal_pad_size // torch.int32.itemsize // dist.get_world_size()
