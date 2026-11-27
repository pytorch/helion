"""Serialize benchmark tensors: files that load, with aliases between dtypes kept."""

from __future__ import annotations

import copyreg
import pickle
import sys
import types
from typing import TYPE_CHECKING
from typing import BinaryIO

import torch
from torch._tensor import _rebuild_from_type_v2
from torch._utils import _rebuild_tensor_v2
from torch._utils import _rebuild_tensor_v3

if TYPE_CHECKING:
    from pathlib import Path


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
    # PyTorch's v3 rebuild creates the tensor with its requires_grad and then
    # calls set_ on it, which autograd refuses for a leaf that requires grad.
    # Rebuilding storage is not an autograd operation.
    return _rebuild_tensor_v3(
        storage, offset, size, stride, requires_grad, hooks, dtype, metadata
    )


def _rebuild_typed_storage(
    storage: torch.storage.TypedStorage, dtype: torch.dtype
) -> torch.storage.TypedStorage:
    # Saved files name this module path and symbol; keep it loadable.
    # ``torch.load`` hands back the saved untyped storage typed as bytes.
    return torch.storage.TypedStorage(
        wrap_storage=storage._untyped_storage, dtype=dtype, _internal=True
    )


def _loadable_reduction(reduction: tuple, *, untyped: bool) -> tuple:
    """A tensor's ``reduction``, changed so that the saved file loads.

    A v3 rebuild (PyTorch's format for the dtypes without a typed storage,
    fp8 among them) of a tensor that requires grad runs outside autograd.
    With ``untyped``, every v3 rebuild does, and the typed-storage v2 format
    becomes untyped storage plus the tensor's dtype.
    """
    rebuild, args, *state = reduction
    if rebuild is _rebuild_tensor_v3 and (untyped or args[4]):
        # (args[4]: the rebuild's requires_grad)
        return (_rebuild_tensor_with_dtype, args, *state)
    if untyped and rebuild is _rebuild_tensor_v2:
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
        inner_rebuild, inner_args = _loadable_reduction(
            (inner_rebuild, inner_args), untyped=untyped
        )
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
    A bare ``TypedStorage`` in the tree takes the same format: under its own
    persistent ID it would name a dtype for data that the tensors here record
    as untyped bytes, which ``torch.save`` refuses.
    The result loads through ordinary trusted ``torch.load``.
    """

    # Tensors and bare typed storages leave as untyped storage plus dtype.
    _untyped = True

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        persistent_id = cls.__dict__.get("persistent_id")
        if persistent_id is None or not cls._untyped:
            return

        # The pickler asks ``persistent_id`` before ``reducer_override``, and
        # the subclass of ``torch.save`` answers it for every storage.
        def untyped_storage_id(self: Pickler, obj: object) -> object:
            if isinstance(obj, torch.storage.TypedStorage):
                return None
            return persistent_id(self, obj)

        cls.persistent_id = untyped_storage_id  # pyrefly: ignore[bad-assignment]

    def __init__(self, file: BinaryIO, protocol: int = 2) -> None:
        super().__init__(file, protocol=protocol)
        self._protocol = protocol

    def reducer_override(
        self, obj: object, /
    ) -> tuple | str | types.NotImplementedType:
        if self._untyped and isinstance(obj, torch.storage.TypedStorage):
            return (_rebuild_typed_storage, (obj._untyped_storage, obj.dtype))
        if isinstance(obj, torch.Tensor):
            # This hook runs ahead of the pickler's own lookup, so repeat it:
            # a reducer registered for the type, else ``__reduce_ex__``.
            reducer = copyreg.dispatch_table.get(type(obj))
            reduction = (
                obj.__reduce_ex__(self._protocol) if reducer is None else reducer(obj)
            )
            if isinstance(reduction, tuple):
                return _loadable_reduction(reduction, untyped=self._untyped)
            return reduction
        return NotImplemented


class _NativePickler(Pickler):
    """PyTorch's own format, for every tensor that format can load.

    The one it cannot is a tensor of a v3 dtype that requires grad (an fp8
    operand of a benchmark built for autograd): ``torch.save`` writes it and
    ``torch.load`` then raises ``a leaf Variable that requires grad is being
    used in an in-place operation``, in the benchmark worker, for every
    candidate. Only that rebuild changes; every other object keeps
    PyTorch's bytes.
    """

    _untyped = False


# ``torch.save`` reads its ``pickle_module``'s name and ``Pickler``.
_NATIVE_PICKLE = types.SimpleNamespace(
    __name__=f"{__name__}.native", Pickler=_NativePickler
)


def save_tensor_tree(value: object, path: str | Path) -> None:
    """Save trusted worker data, retaining mixed-dtype dense tensor aliases.

    PyTorch's own format goes first, with a v3 tensor that requires grad
    rebuilt outside autograd so that the file loads: it is compatible with
    other Python objects and tensor layouts (including bare TypedStorage
    objects sharing a tensor's data). Its typed-storage format refuses, for
    example, INT8 and UINT8 views of the same cache allocation. Retry that
    specific failure using untyped storage plus each tensor's or storage's
    dtype, without cloning or moving tensors. Either way the file loads
    through ordinary trusted ``torch.load``.
    """
    try:
        torch.save(value, path, pickle_module=_NATIVE_PICKLE)
    except RuntimeError as error:
        if (
            "Cannot save multiple tensors or storages that view the same data "
            "as different types"
        ) not in str(error):
            raise
        torch.save(value, path, pickle_module=sys.modules[__name__])
