from __future__ import annotations

import contextlib
from dataclasses import dataclass
import os
import random
from typing import Generator
from typing import TypeVar

import torch
from torch import Tensor
from torch._C._distributed_c10d import _SymmetricMemory
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import helion
from helion import exc

T = TypeVar("T")


def all_gather_object(obj: T) -> list[T]:
    if not dist.is_initialized():
        return [obj]

    object_list = [None] * dist.get_world_size()
    dist.all_gather_object(object_list, obj)
    return object_list  # pyrefly: ignore


def max_num_blocks_for_symm_mem() -> int:
    """
    Return the max number of blocks allowed due to the restriction of
    signal pad size in symm memory.
    """
    assert dist.is_initialized()
    signal_pad_size = _SymmetricMemory.signal_pad_size
    return signal_pad_size // torch.int32.itemsize // dist.get_world_size()


def is_master_rank() -> bool:
    """
    Either return True for rank 0 in a distributed workload or
    always return true for non-distributed workload.
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def is_symm_mem_tensor(t: Tensor) -> bool:
    if not isinstance(t, Tensor) or not dist.is_initialized():
        return False

    # TODO(shunting): support group other than WORLD?
    try:
        assert dist.group.WORLD is not None
        return symm_mem.rendezvous(t, group=dist.group.WORLD.group_name) is not None
    except RuntimeError:
        # PyTorch right now throws a RuntimeError if the tensor passed
        # to rendezvious is not from symm-mem
        return False


def clone_symm_mem_tensor(t: Tensor) -> Tensor:
    """Clone a contiguous symmetric memory tensor and rendezvous the new allocation."""
    assert t.is_contiguous(), "Only support cloning contiguous symm mem tensor for now"
    new_tensor = symm_mem.empty(
        *t.shape,
        dtype=t.dtype,
        device=t.device,
    )
    new_tensor.copy_(t)
    # rendezvous so we don't count the time in benchmarking
    assert dist.group.WORLD is not None
    symm_mem.rendezvous(new_tensor, dist.group.WORLD.group_name)
    return new_tensor


def get_signal_pad_ptrs_dev(t: Tensor) -> int:
    assert dist.group.WORLD is not None
    hdl = symm_mem.rendezvous(t, group=dist.group.WORLD.group_name)
    return hdl.signal_pad_ptrs_dev


def check_config_consistancy(config: helion.Config, print_config: bool = False) -> None:
    """
    Check the consistency of configs across ranks.
    """
    if (
        os.getenv("HELION_DIST_CHECK_CONFIG_CONSISTANCY") != "1"
        or not dist.is_initialized()
    ):
        return

    all_configs = [None] * dist.get_world_size()
    dist.all_gather_object(all_configs, config)
    if dist.get_rank() == 0:
        # do the check on rank 0
        if all_configs != all_configs[:1] * len(all_configs):
            if print_config:
                for idx, c in enumerate(all_configs):
                    print("FAIL", idx, c)
            raise exc.InconsistantConfigsAcrossRanks
        if print_config:
            for idx, c in enumerate(all_configs):
                print("PASS", idx, c)


def print_with_rank(*args: object, **kwargs: object) -> None:
    if dist.is_initialized():
        print(f"Rank{dist.get_rank()}: ", end="")
    print(*args, **kwargs)  # pyrefly: ignore[no-matching-overload]


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
