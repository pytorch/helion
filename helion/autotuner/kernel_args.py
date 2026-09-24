from __future__ import annotations

import copy
import dataclasses
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


# CuTe's packed-workspace input contract requires 256-byte base alignment.
# Matching this residue also preserves the 16-byte TMA/vector requirements;
# this is not a promise of absolute addresses or larger, future alignments.
_STORAGE_ALIGNMENT = 256
_TRUSTED_ARGS_VERSION = 1


def _storage_id(tensor: torch.Tensor) -> int | None:
    if tensor.layout is not torch.strided:
        return None
    try:
        return tensor.untyped_storage()._cdata
    except RuntimeError:
        return None


def _supports_storage_window(tensor: torch.Tensor) -> bool:
    return (
        type(tensor) in (torch.Tensor, torch.nn.Parameter)
        and tensor.layout is torch.strided
        and tensor.device.type in ("cpu", "cuda")
        and not tensor.is_quantized
    )


@dataclasses.dataclass(frozen=True)
class _SavedTensorLayout:
    leaf: int
    tensor: int
    storage: int
    nbytes: int
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool
    conjugate: bool
    negative: bool


@dataclasses.dataclass(frozen=True)
class _TrustedKernelArgs:
    version: int
    args: Sequence[object]
    layouts: tuple[_SavedTensorLayout, ...]
    residues: tuple[int, ...]


def _argument_layouts(
    args: Sequence[object],
) -> tuple[tuple[_SavedTensorLayout, ...], tuple[int, ...]]:
    flat, _ = tree_flatten(args)
    # A custom tensor sharing ordinary storage must retain its existing clone
    # protocol as a group; do not claim that we can reconstruct that subclass.
    excluded = {
        _storage_id(value)
        for value in flat
        if isinstance(value, torch.Tensor) and not _supports_storage_window(value)
    }
    tensors: dict[int, int] = {}
    storages: dict[int, int] = {}
    layouts: list[_SavedTensorLayout] = []
    residues: list[int] = []
    for index, value in enumerate(flat):
        if not isinstance(value, torch.Tensor) or not _supports_storage_window(value):
            continue
        storage = value.untyped_storage()
        if storage._cdata in excluded:
            continue
        tensor_index = tensors.setdefault(id(value), index)
        if storage._cdata not in storages:
            storages[storage._cdata] = len(storages)
            residues.append(storage.data_ptr() % _STORAGE_ALIGNMENT)
        layouts.append(
            _SavedTensorLayout(
                index,
                tensor_index,
                storages[storage._cdata],
                storage.nbytes(),
                int(value.storage_offset()),
                tuple(value.shape),
                tuple(value.stride()),
                value.dtype,
                value.device,
                value.requires_grad,
                value.is_conj(),
                value.is_neg(),
            )
        )
    return tuple(layouts), tuple(residues)


def save_trusted_kernel_args(args: Sequence[object], path: str) -> None:
    """Save arguments plus original alignment, before Torch relocates storage.

    Keep Torch's existing supported pickle set, including its rejection of
    mixed-dtype aliases. The envelope is not a general tensor serializer.
    """
    layouts, residues = _argument_layouts(args)
    torch.save(_TrustedKernelArgs(_TRUSTED_ARGS_VERSION, args, layouts, residues), path)


@functools.cache
def load_trusted_kernel_args(path: str) -> Sequence[object]:
    # Cache a pristine template, never candidate execution arguments. Each job
    # clones it before invoking a candidate, including candidates whose pure
    # reference does not mutate its arguments.
    # This file is a trusted temporary artifact written by the parent
    # autotuner process. Kernel args can include user Python objects such as
    # callable epilogues, which PyTorch's weights-only loader rejects.
    payload = torch.load(path, weights_only=False)
    if not isinstance(payload, _TrustedKernelArgs):
        # Compatibility with old temporary files/tools only: their original
        # external-storage alignment was not recorded and cannot be recovered.
        return cast("Sequence[object]", payload)
    if type(payload.version) is not int or payload.version != _TRUSTED_ARGS_VERSION:
        raise ValueError("Unsupported trusted kernel argument version")
    layouts, residues = _argument_layouts(payload.args)
    if (
        layouts != payload.layouts
        or not isinstance(payload.residues, tuple)
        or len(residues) != len(payload.residues)
        or any(
            type(residue) is not int or not 0 <= residue < _STORAGE_ALIGNMENT
            for residue in payload.residues
        )
    ):
        raise ValueError("Invalid trusted kernel argument storage metadata")
    flat, spec = tree_flatten(payload.args)
    groups: dict[int, list[torch.Tensor]] = {}
    for layout in layouts:
        if layout.leaf == layout.tensor:
            groups.setdefault(layout.storage, []).append(
                cast("torch.Tensor", flat[layout.leaf])
            )
    replacements: dict[int, torch.Tensor] = {}
    for group, tensors in groups.items():
        if residues[group] == payload.residues[group]:
            continue
        clones = _clone_storage_group(tensors, residue=payload.residues[group])
        replacements.update(
            (id(tensor), clone) for tensor, clone in zip(tensors, clones, strict=True)
        )
    for index, value in enumerate(flat):
        if isinstance(value, torch.Tensor) and id(value) in replacements:
            flat[index] = replacements[id(value)]
    return tree_unflatten(flat, spec)


def _clone_storage_group(
    tensors: list[torch.Tensor], *, residue: int | None = None
) -> list[torch.Tensor]:
    first = tensors[0]
    original = first.untyped_storage()
    if residue is None:
        residue = original.data_ptr() % _STORAGE_ALIGNMENT
    if (
        len(tensors) == 1
        and first.is_contiguous()
        and first.storage_offset() == 0
        and first.numel() * first.element_size() == original.nbytes()
        and not first.is_conj()
        and not first.is_neg()
    ):
        # Preserve the common Tensor.clone fast path and its observable calls.
        clone = first.detach().clone()
        storage = clone.untyped_storage()
        if storage.data_ptr() % _STORAGE_ALIGNMENT == residue:
            clone.requires_grad_(first.requires_grad)
            return [clone]
    else:
        # Copy raw bytes once: deepcopy materializes lazy conj/negative views,
        # which can reset offsets and split a shared-storage alias group.
        storage = original.clone()
    if storage.data_ptr() % _STORAGE_ALIGNMENT != residue:
        owner = torch.empty(
            storage.nbytes() + _STORAGE_ALIGNMENT - 1,
            dtype=torch.uint8,
            device=first.device,
        )
        shift = (residue - owner.data_ptr()) % _STORAGE_ALIGNMENT
        window = owner.untyped_storage()[shift : shift + storage.nbytes()]
        window.copy_(storage)
        # Storage slices retain the backing allocation, including its padding.
        storage = window
    result: list[torch.Tensor] = []
    for tensor in tensors:
        clone = tensor.new_empty((0,)).set_(
            storage, tensor.storage_offset(), tensor.shape, tensor.stride()
        )
        if tensor.is_conj():
            clone = clone.conj()
        if tensor.is_neg():
            clone = torch._neg_view(clone)
        clone.requires_grad_(tensor.requires_grad)
        result.append(clone)
    return result


def _argument_storage_bytes(args: Sequence[object]) -> int:
    """Conservative CUDA clone footprint, including imported-base padding."""
    flat, _ = tree_flatten(args)
    seen: set[tuple[str, int]] = set()
    total = 0
    for value in flat:
        if not isinstance(value, torch.Tensor):
            continue
        storage_id = _storage_id(value)
        key = ("tensor", id(value)) if storage_id is None else ("storage", storage_id)
        if key in seen:
            continue
        seen.add(key)
        if storage_id is None:
            total += value.numel() * value.element_size()
        else:
            storage = value.untyped_storage()
            total += storage.nbytes()
            # Ordinary CUDA allocations are aligned to 256 bytes; offset
            # views retain that base. Imported unaligned bases need a window.
            if value.device.type == "cuda" and storage.data_ptr() % _STORAGE_ALIGNMENT:
                total += _STORAGE_ALIGNMENT - 1
    return total


def _clone_args(
    args: Sequence[object],
    process_group_name: str | None,
    idx_to_clone: Sequence[int] | None = None,
) -> Sequence[object]:
    """Clone selected tensor leaves while preserving their alias topology.

    If a selected ordinary tensor shares storage with another tensor argument,
    clone that whole argument alias group.  This keeps view offsets, strides,
    mixed-dtype storage aliases, and duplicate references intact while still
    isolating the cloned group from both the caller and other candidates.
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
        if all(_supports_storage_window(tensor) for tensor in tensors):
            clones = _clone_storage_group(tensors)
        elif len(tensors) == 1 and tensors[0].is_contiguous():
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
