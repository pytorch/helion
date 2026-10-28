from __future__ import annotations

import builtins
import copy
import dataclasses
import functools
import importlib
import marshal
import sys
import types
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


@dataclasses.dataclass(frozen=True)
class PortableFunction:
    """A Python function serialized by code for the benchmark subprocess.

    Standard pickling refuses lambdas and closures (``matmul``'s ``epilogue``
    default, an ``addmm`` epilogue capturing the bias), which used to force the
    whole autotune to benchmark in-process -- and an in-process cuda-graph
    benchmark leaves the process in a state where every later CUDA graph of
    the kernel runs its kernel nodes ~0.35 us slower (see
    ``LocalBenchmarkProvider.setup``).  The worker rebuilds a real function
    from the marshalled code object (a reference would not do: a local
    function has none, and the generated host code reads
    ``epilogue.__closure__[i].cell_contents`` from a real function object),
    with the same cells in the same order, the same defaults and keyword
    defaults.  Globals are not shipped: the rebuilt function takes the
    namespace of the module named by ``__module__`` in the worker, so only
    functions whose ``__globals__`` is that module's namespace are portable
    (a ``functools.wraps`` wrapper or an ``exec``-created function is not and
    keeps the in-process fallback).  Trusted parent-to-worker data, like the
    rest of the arguments file.
    """

    module: str
    qualname: str
    name: str
    code: bytes
    defaults: tuple[object, ...] | None
    kwdefaults: dict[str, object] | None
    closure: tuple[object, ...] | None


class _NotPortable(Exception):
    pass


def _portable_value(value: object, seen: set[int]) -> object:
    if isinstance(value, types.FunctionType):
        return _portable_function(value, seen)
    return value


def _portable_function(fn: types.FunctionType, seen: set[int]) -> PortableFunction:
    if id(fn) in seen:
        raise _NotPortable("recursive closure")
    seen = {*seen, id(fn)}
    module = sys.modules.get(fn.__module__) if isinstance(fn.__module__, str) else None
    if module is None or fn.__globals__ is not vars(module):
        # The worker resolves globals through ``fn.__module__``; a function
        # whose globals are some other namespace (a ``functools.wraps``
        # wrapper carries the wrapped function's module, an ``exec``-created
        # function its own dict) would rebuild with the wrong ones.
        raise _NotPortable("globals are not the function's module namespace")
    try:
        code = marshal.dumps(fn.__code__)
    except ValueError as error:
        raise _NotPortable(str(error)) from error
    closure: tuple[object, ...] | None = None
    if fn.__closure__:
        values = []
        for cell in fn.__closure__:
            try:
                contents = cell.cell_contents
            except ValueError as error:  # an empty cell
                raise _NotPortable("empty closure cell") from error
            values.append(_portable_value(contents, seen))
        closure = tuple(values)
    defaults = (
        tuple(_portable_value(value, seen) for value in fn.__defaults__)
        if fn.__defaults__
        else None
    )
    kwdefaults = (
        {key: _portable_value(value, seen) for key, value in fn.__kwdefaults__.items()}
        if fn.__kwdefaults__
        else None
    )
    return PortableFunction(
        module=fn.__module__,
        qualname=fn.__qualname__,
        name=fn.__name__,
        code=code,
        defaults=defaults,
        kwdefaults=kwdefaults,
        closure=closure,
    )


def make_portable_args(args: Sequence[object]) -> Sequence[object] | None:
    """``args`` with every function leaf (and the functions in their closures
    and defaults) replaced by its :class:`PortableFunction`, or None when
    there is no function leaf or one cannot be serialized by code."""
    leaves, spec = tree_flatten(args)
    converted = False
    portable: list[object] = []
    for leaf in leaves:
        if isinstance(leaf, types.FunctionType):
            try:
                portable.append(_portable_function(leaf, set()))
            except _NotPortable:
                return None
            converted = True
        else:
            portable.append(leaf)
    if not converted:
        return None
    return cast("Sequence[object]", tree_unflatten(portable, spec))


def _resolve_value(value: object) -> object:
    if isinstance(value, PortableFunction):
        return _rebuild_function(value)
    return value


def _rebuild_function(portable: PortableFunction) -> types.FunctionType:
    code = marshal.loads(portable.code)
    try:
        function_globals = importlib.import_module(portable.module).__dict__
    except Exception:
        function_globals = {"__builtins__": builtins, "__name__": portable.module}
    closure = (
        tuple(types.CellType(_resolve_value(value)) for value in portable.closure)
        if portable.closure is not None
        else None
    )
    defaults = (
        tuple(_resolve_value(value) for value in portable.defaults)
        if portable.defaults is not None
        else None
    )
    fn = types.FunctionType(code, function_globals, portable.name, defaults, closure)
    fn.__qualname__ = portable.qualname
    fn.__module__ = portable.module
    if portable.kwdefaults is not None:
        fn.__kwdefaults__ = {
            key: _resolve_value(value) for key, value in portable.kwdefaults.items()
        }
    return fn


def resolve_portable_args(args: Sequence[object]) -> Sequence[object]:
    """Rebuild the functions :func:`make_portable_args` serialized by code."""
    leaves, spec = tree_flatten(args)
    if not any(isinstance(leaf, PortableFunction) for leaf in leaves):
        return args
    return cast(
        "Sequence[object]",
        tree_unflatten([_resolve_value(leaf) for leaf in leaves], spec),
    )


@functools.cache
def load_trusted_kernel_args(path: str) -> Sequence[object]:
    # Cached so re-spawning configs don't re-read the same args off disk.
    # This file is a trusted temporary artifact written by the parent
    # autotuner process. Kernel args can include user Python objects such as
    # callable epilogues, which PyTorch's weights-only loader rejects, and
    # functions the parent serialized by code (``PortableFunction``), rebuilt
    # once here so every job sees the same function objects.
    return resolve_portable_args(
        cast("Sequence[object]", torch.load(path, weights_only=False))
    )


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
    """``args`` with every tensor leaf on private storage, aliasing preserved.

    Tensors a function argument captures (an ``addmm`` epilogue's bias) or
    takes as defaults are operands of the kernel too, so a function leaf is
    rebuilt with its closure cells, defaults and keyword defaults cloned the
    same way; a function holding no tensor is kept as it is.
    """
    storage_clones: dict[int, torch.Tensor] = {}

    def clone_tensor(leaf: torch.Tensor) -> torch.Tensor:
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
        return view

    def clone_tree(tree: object) -> object:
        leaves, spec = tree_flatten(tree)
        return tree_unflatten([clone_leaf(leaf) for leaf in leaves], spec)

    def clone_leaf(leaf: object) -> object:
        if isinstance(leaf, torch.Tensor):
            return clone_tensor(leaf)
        if isinstance(leaf, types.FunctionType):
            return clone_function(leaf)
        return leaf

    def holds_clonable_leaf(values: object) -> bool:
        return any(
            isinstance(leaf, (torch.Tensor, types.FunctionType))
            for leaf in tree_flatten(values)[0]
        )

    def clone_function(fn: types.FunctionType) -> types.FunctionType:
        try:
            cells = [cell.cell_contents for cell in fn.__closure__ or ()]
        except ValueError:  # an empty cell: nothing to clone there
            return fn
        defaults = fn.__defaults__ or ()
        kwdefaults = fn.__kwdefaults__ or {}
        if not holds_clonable_leaf((cells, defaults, kwdefaults)):
            return fn
        rebuilt = types.FunctionType(
            fn.__code__,
            fn.__globals__,
            fn.__name__,
            tuple(clone_tree(value) for value in defaults) if defaults else None,
            tuple(types.CellType(clone_tree(value)) for value in cells)
            if fn.__closure__
            else None,
        )
        rebuilt.__qualname__ = fn.__qualname__
        rebuilt.__module__ = fn.__module__
        if kwdefaults:
            rebuilt.__kwdefaults__ = {
                key: clone_tree(value) for key, value in kwdefaults.items()
            }
        return rebuilt

    return cast("Sequence[object]", clone_tree(args))


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
