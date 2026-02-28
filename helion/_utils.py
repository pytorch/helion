from __future__ import annotations

import collections
import contextlib
from dataclasses import dataclass
import functools
import random
from typing import Generator
import contextlib
from typing import Sequence
from typing import TypeVar

import torch
import torch.distributed as dist
from torch import Tensor
import torch
from dataclasses import dataclass
import random
import os

import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from helion import exc

counters: collections.defaultdict[str, collections.Counter[str]] = (
    collections.defaultdict(collections.Counter)
)

T = TypeVar("T")


def cdiv(a: int, b: int) -> int:
    """Ceiling division: returns ceil(a / b)."""
    return (a + b - 1) // b


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


@functools.cache
def triton_is_available() -> bool:
    """Return True if triton is installed and importable."""
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


def create_shape_matching_slices(
    shape1: Sequence[int], shape2: Sequence[int]
) -> tuple[slice, ...]:
    """Create slices to match the smaller of two shapes.

    This is used for masking tensors to compatible shapes by taking the
    minimum size in each dimension.

    Args:
        shape1: First shape (can be torch.Size or any sequence of ints)
        shape2: Second shape (can be torch.Size or any sequence of ints)

    Returns:
        Tuple of slices that can be used to index a tensor
    """
    return tuple(slice(0, min(d1, d2)) for d1, d2 in zip(shape1, shape2, strict=False))


def convert_size_arg(size: object) -> object:
    """Convert a size argument that may contain RefTile objects.

    Handles:
    - Single RefTile -> int (block_size)
    - List/tuple containing RefTiles -> list with converted sizes
    - Other values -> unchanged
    """
    # Import here to avoid circular dependency
    from .language.ref_tile import RefTile

    if isinstance(size, (list, tuple)):
        return [convert_size_arg(item) for item in size]
    if isinstance(size, RefTile):
        return size._block_size
    return size


def convert_tile_indices_to_slices(index: object) -> object:
    """Convert RefTile objects in index to their corresponding slice objects.

    Args:
        index: Index that may contain RefTile objects or tuples of indices

    Returns:
        Index with RefTile objects replaced by their slice objects
    """
    # Import here to avoid circular dependency
    from .language.ref_tile import RefTile

    def _extract_slice(obj: object) -> object:
        return obj._slice if isinstance(obj, RefTile) else obj

    if isinstance(index, tuple):
        return tuple(_extract_slice(idx) for idx in index)
    return _extract_slice(index)


def is_master_rank() -> bool:
    """
    Either rank 0 in distributed workload or always return true for
    non-distributed workload.
    """
    return not dist.is_initialized() or dist.get_rank() == 0

def is_symm_mem_tensor(t: Tensor) -> bool:
    if not isinstance(t, Tensor) or not dist.is_initialized():
        return False

    # TODO: should we support group other than WORLD?
    try:
        return symm_mem.rendezvous(t, group=dist.group.WORLD.group_name) is not None
    except RuntimeError:
        # PyTorch right now throws a RuntimeError if the tensor passed
        # to rendezvious is not from symm-mem
        return False

def get_signal_pad_ptrs_dev(t: Tensor) -> int:
    hdl = symm_mem.rendezvous(t, group=dist.group.WORLD.group_name)
    return hdl.signal_pad_ptrs_dev

def check_config_consistancy(config):
    """
    Check the consistancy of configs across ranks.
    """
    if not os.getenv("CHECK_CONFIG_CONSISTANCY") == "1" or not dist.is_initialized():
        return

    all_configs = [None] * dist.get_world_size()
    dist.all_gather_object(all_configs, config)
    if dist.get_rank() == 0:
        # do the check on rank 0
        if all_configs != all_configs[:1] * len(all_configs):
            for idx, c in enumerate(all_configs):
                # print("FAIL", idx, c)
                pass
            raise exc.InconsistantConfigsAcrossRanks
        else:
            for idx, c in enumerate(all_configs):
                # print("PASS", idx, c)
                pass

def print_with_rank(*args, **kwargs):
    if dist.is_initialized():
        print(f"Rank{dist.get_rank()}: ", end="")
    print(*args, **kwargs)


@dataclass
class SeedEnsemble:
    torch_seed: int
    py_random_seed: int

    @staticmethod
    def get_seeds() -> SeedEnsemble:
        """
        There is no way to get current seed in PyTorch. We can only get
        the initial seed.

        This method instead re-initialize the seed by incrementing the
        initial seed by 1
        """
        seed = torch.initial_seed()
        return SeedEnsemble(
            seed + 1,
            seed + 1,
        )

    @staticmethod
    def set_seeds(seeds: SeedEnsemble) -> None:
        torch.manual_seed(seeds.torch_seed)
        random.seed(seeds.py_random_seed)

    @classmethod
    def update_seeds_with_rank(cls) -> None:
        seed = torch.initial_seed() + 1 + dist.get_rank()
        cls.set_seeds(SeedEnsemble(seed, seed))


@contextlib.contextmanager
def sync_seed(need_diverse_seeds_after: bool = True) -> Generator[None, None, None]:
    """
    Sync seeds across ranks.

    If need_diverse_seeds_after is True, we make sure different
    ranks have different seeds after the call. This ensures different
    rank can generate independent random tensors.
    """
    if not dist.is_initialized():
        yield
        return

    from helion._testing import sync_object

    seeds = sync_object(SeedEnsemble.get_seeds())

    try:
        SeedEnsemble.set_seeds(seeds)
        yield
    finally:
        if need_diverse_seeds_after:
            SeedEnsemble.update_seeds_with_rank()


def all_gather_object(obj: T) -> list[T]:
    if not dist.is_initialized():
        return [obj]

    object_list = [None] * dist.get_world_size()
    dist.all_gather_object(object_list, obj)
    return object_list  # pyrefly: ignore
