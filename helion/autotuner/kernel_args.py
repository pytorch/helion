from __future__ import annotations

import copy
import functools
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_unflatten

from helion._dist_utils import _clone_symm_mem_tensor
from helion._dist_utils import get_signal_pad_ptrs_dev
from helion._dist_utils import is_symm_mem_tensor

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


def _clone_args(
    args: Sequence[object],
    process_group_name: str | None,
    idx_to_clone: Sequence[int] | None = None,
    *,
    preserve_storage: bool = False,
) -> Sequence[object]:
    """Clone selected tensor leaves while preserving their alias topology.

    If a selected ordinary tensor shares storage with another tensor argument,
    clone that whole argument alias group.  This keeps view offsets, strides,
    mixed-dtype storage aliases, and duplicate references intact while still
    isolating the cloned group from both the caller and other candidates.
    ``preserve_storage`` also retains the full storage and offset of a lone
    contiguous view, as required when replaying a standalone kernel's layout.
    """

    clone_indices = None if idx_to_clone is None else set(idx_to_clone)

    def _should_clone(idx: int) -> bool:
        return clone_indices is None or idx in clone_indices

    args_flat, tree_spec = tree_flatten(args)
    tensor_replacements: dict[int, torch.Tensor] = {}
    signal_pad_replacements: dict[int, int] = {}
    symmetric_tensor_ids: set[int] = set()

    for i, arg in enumerate(args_flat):
        if _should_clone(i) and is_symm_mem_tensor(arg, process_group_name):
            arg_id = id(arg)
            symmetric_tensor_ids.add(arg_id)
            if arg_id not in tensor_replacements:
                new_arg = _clone_symm_mem_tensor(arg, process_group_name)
                signal_pad_replacements[
                    get_signal_pad_ptrs_dev(arg, process_group_name)
                ] = get_signal_pad_ptrs_dev(new_arg, process_group_name)
                tensor_replacements[arg_id] = new_arg

    def _storage_id(tensor: torch.Tensor) -> int | None:
        if tensor.layout is not torch.strided:
            return None
        try:
            return tensor.untyped_storage()._cdata
        except RuntimeError:
            return None

    # A partial selection must include all ordinary tensor arguments that alias
    # a selected tensor.  Otherwise an in-place candidate sees a different
    # alias relationship from the original invocation.
    selected_storage_ids: set[int] = set()
    selected_tensor_ids: set[int] = set()
    for i, arg in enumerate(args_flat):
        if (
            _should_clone(i)
            and isinstance(arg, torch.Tensor)
            and id(arg) not in symmetric_tensor_ids
        ):
            selected_tensor_ids.add(id(arg))
            storage_id = _storage_id(arg)
            if storage_id is not None:
                selected_storage_ids.add(storage_id)

    ordinary_tensors: list[torch.Tensor] = []
    seen_tensor_ids: set[int] = set()
    for arg in args_flat:
        if not isinstance(arg, torch.Tensor) or id(arg) in tensor_replacements:
            continue
        arg_id = id(arg)
        storage_id = _storage_id(arg)
        if arg_id not in selected_tensor_ids and (
            storage_id is None or storage_id not in selected_storage_ids
        ):
            continue
        if arg_id not in seen_tensor_ids:
            seen_tensor_ids.add(arg_id)
            ordinary_tensors.append(arg)

    storage_groups: dict[tuple[str, int], list[torch.Tensor]] = {}
    for tensor in ordinary_tensors:
        storage_id = _storage_id(tensor)
        key = (
            ("storage", storage_id)
            if storage_id is not None
            else ("tensor", id(tensor))
        )
        storage_groups.setdefault(key, []).append(tensor)

    for tensors in storage_groups.values():
        if not preserve_storage and len(tensors) == 1 and tensors[0].is_contiguous():
            # Retain the ordinary fast path (and its observable clone
            # semantics) when there is no cross-argument alias topology to
            # preserve.
            clones = [tensors[0].detach().clone()]
        else:
            # Deepcopy aliased detached tensors together: PyTorch memoizes their
            # storage, preserving cross-view aliases, offsets, strides, and
            # mixed dtypes. It also preserves a lone non-contiguous layout.
            clones = copy.deepcopy([tensor.detach() for tensor in tensors])
        for tensor, clone in zip(tensors, clones, strict=True):
            clone.requires_grad_(tensor.requires_grad)
            tensor_replacements[id(tensor)] = clone

    for i, arg in enumerate(args_flat):
        if isinstance(arg, torch.Tensor) and id(arg) in tensor_replacements:
            args_flat[i] = tensor_replacements[id(arg)]
            continue
        if isinstance(arg, int) and arg in signal_pad_replacements:
            args_flat[i] = signal_pad_replacements[arg]

    return tree_unflatten(args_flat, tree_spec)
