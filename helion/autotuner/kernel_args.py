from __future__ import annotations

import functools
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_unflatten

if TYPE_CHECKING:
    from collections.abc import Sequence


@functools.cache
def load_trusted_kernel_args(path: str) -> Sequence[object]:
    # Cached so re-spawning configs don't re-read the same args off disk.
    # This file is a trusted temporary artifact written by the parent
    # autotuner process. Kernel args can include user Python objects such as
    # callable epilogues, which PyTorch's weights-only loader rejects.
    return cast("Sequence[object]", torch.load(path, weights_only=False))


@functools.cache
def load_private_kernel_args(path: str, slot: int) -> Sequence[object]:
    """Return a private copy of the trusted args for finalist ``slot``.

    Every slot owns distinct tensor storage, and the copies stay alive until
    :func:`release_private_kernel_args`, so finalists timed back to back read
    different addresses: one candidate's L2-resident input cannot become a
    hit for the next one. Tensor aliasing inside the arguments is preserved.
    """
    return _clone_tensor_leaves(load_trusted_kernel_args(path))


def release_private_kernel_args() -> None:
    """Drop every finalist's private argument copies (run in the worker once
    the final selection ends, so later phases get the device memory back)."""
    load_private_kernel_args.cache_clear()
    torch.cuda.empty_cache()


def _clone_tensor_leaves(args: Sequence[object]) -> Sequence[object]:
    leaves, spec = tree_flatten(args)
    storage_clones: dict[int, torch.Tensor] = {}
    cloned: list[object] = []
    for leaf in leaves:
        if not isinstance(leaf, torch.Tensor):
            cloned.append(leaf)
            continue
        storage = leaf.untyped_storage()
        key = storage.data_ptr()
        base = storage_clones.get(key)
        if base is None:
            base = torch.empty(storage.nbytes(), dtype=torch.uint8, device=leaf.device)
            base.untyped_storage().copy_(storage)
            storage_clones[key] = base
        view = torch.empty(0, dtype=leaf.dtype, device=leaf.device)
        view.set_(
            base.untyped_storage(), leaf.storage_offset(), leaf.size(), leaf.stride()
        )
        cloned.append(view)
    return cast("Sequence[object]", tree_unflatten(cloned, spec))
