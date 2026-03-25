from __future__ import annotations

import contextlib
from dataclasses import dataclass
import inspect
import os
import random
from typing import Callable
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


def _resolve_process_group(name: str) -> dist.ProcessGroup:
    for pg, pg_name in dist.distributed_c10d._world.pg_names.items():
        if pg_name == name:
            return pg
    raise ValueError(f"No process group with name {name!r}")


def all_gather_object(obj: T, process_group_name: str | None = None) -> list[T]:
    if not dist.is_initialized():
        return [obj]

    assert process_group_name is not None

    group = _resolve_process_group(process_group_name)
    object_list = [None] * dist.get_world_size(group)
    dist.all_gather_object(object_list, obj, group=group)
    return object_list  # pyrefly: ignore


def sync_object(obj: T, process_group_name: str | None = None) -> T:
    r"""
    Synchronize the number of repeations across all ranks.
    """
    if not dist.is_initialized():
        return obj

    assert process_group_name is not None
    group = _resolve_process_group(process_group_name)
    object_list = [obj]
    # use the value from group rank 0
    src = dist.get_global_rank(group, 0)
    dist.broadcast_object_list(object_list, src, group=group)
    return object_list[0]


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


def is_symm_mem_tensor(t: Tensor, process_group_name: str | None = None) -> bool:
    if not isinstance(t, Tensor) or not dist.is_initialized():
        return False

    assert process_group_name is not None
    try:
        hdl = symm_mem.rendezvous(
            t,
            group=process_group_name,  # pyrefly: ignore[bad-argument-type]
        )
        return hdl is not None
    except RuntimeError:
        # PyTorch right now throws a RuntimeError if the tensor passed
        # to rendezvious is not from symm-mem
        return False


def get_signal_pad_ptrs_dev(t: Tensor, process_group_name: str | None = None) -> int:
    assert dist.is_initialized()
    assert process_group_name is not None
    hdl = symm_mem.rendezvous(
        t,
        group=process_group_name,  # pyrefly: ignore[bad-argument-type]
    )
    return hdl.signal_pad_ptrs_dev


def check_config_consistancy(
    config: helion.Config,
    print_config: bool = False,
    process_group_name: str | None = None,
) -> None:
    """
    Check the consistency of configs across ranks.
    """
    if (
        os.getenv("HELION_DIST_CHECK_CONFIG_CONSISTANCY") != "1"
        or not dist.is_initialized()
    ):
        return

    assert process_group_name is not None
    group = _resolve_process_group(process_group_name)
    all_configs = [None] * dist.get_world_size(group)
    dist.all_gather_object(all_configs, config, group=group)
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
def sync_seed(
    need_diverse_seeds_after: bool = True, process_group_name: str | None = None
) -> Generator[None, None, None]:
    """
    Sync seeds across ranks.

    If need_diverse_seeds_after is True, we make sure different
    ranks have different seeds after the call. This ensures different
    rank can generate independent random tensors.
    """
    if not dist.is_initialized():
        yield
        return

    assert process_group_name is not None
    seeds = sync_object(SeedEnsemble.get_seeds(), process_group_name=process_group_name)

    try:
        SeedEnsemble.set_seeds(seeds)
        yield
    finally:
        if need_diverse_seeds_after:
            SeedEnsemble.update_seeds_with_rank()


def _find_process_group_name(fn: Callable, args: tuple[object, ...]) -> str | None:
    from helion._compiler.compile_environment import warning
    from helion.exc import ProcessGroupNameNotFound

    if not dist.is_initialized():
        return None

    signature = inspect.signature(fn)
    for param, arg in zip(signature.parameters.values(), args, strict=True):
        if param.annotation == "hl.ProcessGroupName":
            assert isinstance(arg, str), f"{type(arg)}"
            return arg

    warning(ProcessGroupNameNotFound)
    assert dist.group.WORLD is not None
    return dist.group.WORLD.group_name


def _clone_symm_mem_tensor(
    t: torch.Tensor, process_group_name: str | None
) -> torch.Tensor:
    assert t.is_contiguous(), "Only support cloning contiguous symm mem tensor for now"
    new_tensor = symm_mem.empty(
        *t.shape,
        dtype=t.dtype,
        device=t.device,
    )
    new_tensor.copy_(t)
    # rendezvous so we don't count the time in benchmarking
    assert process_group_name is not None
    # pyrefly: ignore[bad-argument-type]
    symm_mem.rendezvous(new_tensor, process_group_name)
    return new_tensor
