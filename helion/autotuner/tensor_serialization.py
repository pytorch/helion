"""Serialize benchmark tensors without losing aliases between different dtypes."""

from __future__ import annotations

import pickle
import sys
from typing import TYPE_CHECKING
from typing import BinaryIO

import torch
from torch._tensor import _rebuild_from_type_v2
from torch._utils import _rebuild_tensor_v2
from torch._utils import _rebuild_tensor_v3

if TYPE_CHECKING:
    from pathlib import Path
    from types import NotImplementedType


@torch.no_grad()
def _rebuild_tensor_with_dtype(
    storage: torch.storage.TypedStorage,
    offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    requires_grad: bool,
    hooks: dict,
    dtype: torch.dtype,
    metadata: dict | None = None,
) -> torch.Tensor:
    # Handoff bundles persist this module path and symbol; keep it loadable.
    # Some PyTorch versions set requires_grad before set_ in the v3 rebuild.
    # Rebuilding storage is not an autograd operation.
    return _rebuild_tensor_v3(
        storage, offset, size, stride, requires_grad, hooks, dtype, metadata
    )


def _untyped_reduction(reduction: tuple) -> tuple:
    rebuild, args, *state = reduction
    if rebuild is _rebuild_tensor_v3:
        return (_rebuild_tensor_with_dtype, args, *state)
    if rebuild is _rebuild_tensor_v2:
        storage, offset, size, stride, requires_grad, hooks, *metadata = args
        return (
            _rebuild_tensor_with_dtype,
            (
                storage._untyped_storage,
                offset,
                size,
                stride,
                requires_grad,
                hooks,
                storage.dtype,
                *metadata,
            ),
            *state,
        )
    if rebuild is _rebuild_from_type_v2:
        inner_rebuild, tensor_type, inner_args, tensor_state = args
        inner_rebuild, inner_args = _untyped_reduction((inner_rebuild, inner_args))
        return (
            rebuild,
            (inner_rebuild, tensor_type, inner_args, tensor_state),
            *state,
        )
    return reduction


class Pickler(pickle.Pickler):
    """Use PyTorch's dtype-explicit tensor format for shared untyped storage.

    ``torch.save`` subclasses the supplied module's ``Pickler`` to install its
    storage persistent IDs. Keeping that mechanism preserves both tensor-object
    identity and storage aliases, including tensors inside callable arguments.
    The result loads through ordinary trusted ``torch.load``.
    """

    def __init__(self, file: BinaryIO, protocol: int = 2) -> None:
        super().__init__(file, protocol=protocol)
        self._protocol = protocol

    def reducer_override(self, obj: object, /) -> tuple | NotImplementedType:
        if isinstance(obj, torch.Tensor):
            reduction = obj.__reduce_ex__(self._protocol)
            if isinstance(reduction, tuple):
                return _untyped_reduction(reduction)
        return NotImplemented


def save_tensor_tree(value: object, path: str | Path) -> None:
    """Save trusted worker data, retaining mixed-dtype dense tensor aliases.

    Prefer normal serialization for compatibility with other Python objects and
    tensor layouts (including bare TypedStorage objects sharing a tensor's data).
    Its typed-storage format refuses, for example, INT8 and
    UINT8 views of the same cache allocation. Retry that specific failure using
    untyped storage plus each tensor's dtype, without cloning or moving tensors.
    """
    try:
        torch.save(value, path)
    except RuntimeError as error:
        if (
            "Cannot save multiple tensors or storages that view the same data "
            "as different types"
        ) not in str(error):
            raise
        torch.save(value, path, pickle_module=sys.modules[__name__])
