"""Replay handoff inputs without their original Python classes or allocations."""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import cast

import torch
from torch.utils._pytree import is_namedtuple
from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_map

from .benchmark_provider import _clone_args

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from pathlib import Path


@dataclasses.dataclass
class _NamedTupleInput:
    name: str
    fields: tuple[str, ...]
    values: tuple[object, ...]


@dataclasses.dataclass
class _SavedInputs:
    value: object
    storage_residues: tuple[int, ...]


@functools.cache
def _namedtuple_type(name: str, fields: tuple[str, ...]) -> type[tuple[object, ...]]:
    # Inputs and references need the same reconstructed pytree node type.
    factory = cast("Callable[..., type[tuple[object, ...]]]", NamedTuple)
    return factory(name, [(field, object) for field in fields])


def _storage_residues(value: object) -> tuple[int, ...]:
    # CUDA allocations guarantee 256-byte alignment; preserve pointer facts up
    # to that boundary, including DLPack views whose storage base is unaligned.
    return tuple(
        item.untyped_storage().data_ptr() % 256
        for item in tree_flatten(value)[0]
        if isinstance(item, torch.Tensor)
    )


def _restore_alignment(value: object, residues: tuple[int, ...]) -> object:
    tensors = [
        item for item in tree_flatten(value)[0] if isinstance(item, torch.Tensor)
    ]
    storages: dict[int, torch.UntypedStorage] = {}
    replacements: dict[int, torch.Tensor] = {}
    for tensor, residue in zip(tensors, residues, strict=True):
        storage = tensor.untyped_storage()
        if storage.data_ptr() % 256 == residue or id(tensor) in replacements:
            continue
        if storage._cdata not in storages:
            size = storage.nbytes()
            allocation = torch.empty(
                size + 256, dtype=torch.uint8, device=tensor.device
            )
            offset = (residue - allocation.data_ptr()) % 256
            # DLPack gives the shifted slice its own exact-sized storage, so
            # padding does not change the input's observable storage offset.
            target = torch.from_dlpack(allocation[offset : offset + size])
            source = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                storage, 0, (size,), (1,)
            )
            target.copy_(source)
            storages[storage._cdata] = target.untyped_storage()
        replacement = torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
            storages[storage._cdata],
            tensor.storage_offset(),
            tensor.size(),
            tensor.stride(),
        )
        replacement.requires_grad_(tensor.requires_grad)
        replacements[id(tensor)] = replacement
    return tree_map(lambda item: replacements.get(id(item), item), value)


def _clone_inputs(args: Sequence[object]) -> Sequence[object]:
    return cast(
        "Sequence[object]",
        _restore_alignment(
            _clone_args(args, None, preserve_storage=True), _storage_residues(args)
        ),
    )


def _save_inputs(value: object, path: str | Path) -> None:
    def encode(item: object) -> object:
        if is_namedtuple(item):
            item = cast("Any", item)
            return _NamedTupleInput(
                type(item).__name__,
                item._fields,
                tree_map(encode, tuple(item), is_leaf=is_namedtuple),
            )
        return item

    torch.save(
        _SavedInputs(
            tree_map(encode, value, is_leaf=is_namedtuple), _storage_residues(value)
        ),
        path,
    )


def _load_inputs(path: str | Path) -> object:
    saved = torch.load(path, weights_only=False)
    if not isinstance(saved, _SavedInputs):
        return saved  # Older bundles stored the argument tree directly.

    def decode(item: object) -> object:
        if isinstance(item, _NamedTupleInput):
            return _namedtuple_type(item.name, item.fields)(
                *tree_map(decode, item.values)
            )
        return item

    return _restore_alignment(tree_map(decode, saved.value), saved.storage_residues)
