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
    from collections.abc import Callable
    from collections.abc import Iterator
    from collections.abc import Sequence

    from _typeshed import DataclassInstance
    from torch.utils._pytree import TreeSpec


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
    with the same cell sharing and order, defaults and keyword defaults.
    Function and private cell records preserve sharing across the whole payload;
    the parent and worker must use matching Helion builds. Globals are not
    shipped: the rebuilt function uses its defining registered module's
    namespace. An optional ``globals_module`` records that namespace when it
    differs from the public ``__module__`` metadata, as with ``functools.wraps``.
    Legacy seven-field records default to their existing module. Unregistered
    globals dictionaries keep the in-process fallback. Trusted parent-to-worker
    data, like the rest of the arguments file.
    """

    module: str
    qualname: str
    name: str
    code: bytes
    defaults: tuple[object, ...] | None
    kwdefaults: dict[str, object] | None
    closure: tuple[object, ...] | None
    globals_module: str | None = None


class _NotPortable(Exception):
    pass


@dataclasses.dataclass(frozen=True, eq=False)
class _PortableCell:
    """One closure cell in a trusted worker payload; pickle retains sharing."""

    contents: object


class _PortableTree:
    """Map callback-containing paths using one snapshot of each input node."""

    def __init__(
        self,
        args: Sequence[object],
        callback_type: type[types.FunctionType | PortableFunction],
    ) -> None:
        self.callback_type = callback_type
        self.graph = _InputCloneGraph(
            include_functions=callback_type is types.FunctionType
        )
        self.copy_ids = self.graph.copy_ids(
            args, lambda value: isinstance(value, callback_type)
        )
        self.trees: dict[int, tuple[object, object]] = {}
        self.records: dict[int, object] = {}

    def map(self, value: object, map_callback: Callable[[object], object]) -> object:
        def map_leaf(leaf: object) -> object:
            if isinstance(leaf, self.callback_type):
                return map_callback(leaf)
            if dataclasses.is_dataclass(leaf) and not isinstance(leaf, type):
                if id(leaf) in self.records:
                    return self.records[id(leaf)]
                return _copy_dataclass_fields(
                    leaf,
                    lambda field: self.map(field, map_callback),
                    self.records,
                    self.graph.record_fields(leaf),
                )
            return leaf

        return _map_input_tree(
            value, map_leaf, self.trees, self.graph.flatten, self.copy_ids
        )

    def is_recursive(self, fn: types.FunctionType) -> bool:
        # Mutable container shells can hide a callback cycle from the active
        # function stack. Check the captured graph before publishing its record;
        # cached flatten results also cover container-first argument order.
        return any(
            id(operand) in self.graph.copy_ids(operand, lambda value: value is fn)
            for operand in self.graph.function_operands(fn)
        )


def _portable_value(
    value: object,
    cells: dict[int, tuple[types.CellType, _PortableCell]],
    functions: dict[int, tuple[types.FunctionType, PortableFunction]],
    tree: _PortableTree,
) -> object:
    return tree.map(
        value,
        lambda leaf: _portable_function(
            cast("types.FunctionType", leaf), cells, functions, tree
        ),
    )


def _function_globals_module(fn: types.FunctionType) -> str | None:
    if not isinstance(fn.__module__, str):
        raise _NotPortable("function module metadata is not a module name")
    module = sys.modules.get(fn.__module__)
    if isinstance(module, types.ModuleType) and fn.__globals__ is vars(module):
        return None
    # Prefer the defining module's name, but prove the namespace by identity:
    # an exec-created dictionary can claim any __name__ without being registered.
    name = fn.__globals__.get("__name__")
    if isinstance(name, str):
        module = sys.modules.get(name)
        if isinstance(module, types.ModuleType) and fn.__globals__ is vars(module):
            return name
    for name, module in sys.modules.copy().items():
        if isinstance(module, types.ModuleType) and fn.__globals__ is vars(module):
            return name
    raise _NotPortable("globals are not a registered module namespace")


def _portable_function(
    fn: types.FunctionType,
    cells: dict[int, tuple[types.CellType, _PortableCell]],
    functions: dict[int, tuple[types.FunctionType, PortableFunction]],
    tree: _PortableTree,
) -> PortableFunction:
    if id(fn) in functions:
        return functions[id(fn)][1]
    if tree.is_recursive(fn):
        raise _NotPortable("recursive closure")
    globals_module = _function_globals_module(fn)
    try:
        code = marshal.dumps(fn.__code__)
    except ValueError as error:
        raise _NotPortable(str(error)) from error
    closure: tuple[object, ...] | None = None
    if fn.__closure__:
        values = []
        for cell in fn.__closure__:
            if id(cell) not in cells:
                try:
                    contents = cell.cell_contents
                except ValueError as error:  # an empty cell
                    raise _NotPortable("empty closure cell") from error
                cells[id(cell)] = (
                    cell,
                    _PortableCell(_portable_value(contents, cells, functions, tree)),
                )
            values.append(cells[id(cell)][1])
        closure = tuple(values)
    defaults = cast(
        "tuple[object, ...] | None",
        _portable_value(fn.__defaults__, cells, functions, tree),
    )
    kwdefaults = cast(
        "dict[str, object] | None",
        _portable_value(fn.__kwdefaults__, cells, functions, tree),
    )
    portable = PortableFunction(
        module=fn.__module__,
        qualname=fn.__qualname__,
        name=fn.__name__,
        code=code,
        defaults=defaults,
        kwdefaults=kwdefaults,
        closure=closure,
        globals_module=globals_module,
    )
    # Retain each source object as well as its replacement: identity memo keys
    # must remain valid throughout serialization of the complete argument graph.
    functions[id(fn)] = fn, portable
    return portable


def make_portable_args(args: Sequence[object]) -> Sequence[object] | None:
    """``args`` with every function leaf (and the functions in their closures
    and defaults) replaced by its :class:`PortableFunction`, or None when
    there is no function leaf or one cannot be serialized by code."""
    try:
        return _make_portable_args(args)
    except _NotPortable:
        return None


def _make_portable_args(args: Sequence[object]) -> Sequence[object] | None:
    """Return None only for no callbacks; reject unsupported callback graphs."""
    tree = _PortableTree(args, types.FunctionType)
    cells: dict[int, tuple[types.CellType, _PortableCell]] = {}
    functions: dict[int, tuple[types.FunctionType, PortableFunction]] = {}
    portable = _portable_value(args, cells, functions, tree)
    return cast("Sequence[object]", portable) if functions else None


def _resolve_value(
    value: object,
    cells: dict[int, tuple[_PortableCell, types.CellType]],
    functions: dict[int, tuple[PortableFunction, types.FunctionType]],
    tree: _PortableTree,
) -> object:
    return tree.map(
        value,
        lambda leaf: _rebuild_function(
            cast("PortableFunction", leaf), cells, functions, tree
        ),
    )


def _resolve_cell(
    value: object,
    cells: dict[int, tuple[_PortableCell, types.CellType]],
    functions: dict[int, tuple[PortableFunction, types.FunctionType]],
    tree: _PortableTree,
) -> types.CellType:
    if not isinstance(value, _PortableCell):
        # Earlier payloads stored values directly and carried no cell identity.
        return types.CellType(_resolve_value(value, cells, functions, tree))
    if id(value) not in cells:
        cell = types.CellType()
        cells[id(value)] = value, cell
        cell.cell_contents = _resolve_value(value.contents, cells, functions, tree)
    return cells[id(value)][1]


def _rebuild_function(
    portable: PortableFunction,
    cells: dict[int, tuple[_PortableCell, types.CellType]],
    functions: dict[int, tuple[PortableFunction, types.FunctionType]],
    tree: _PortableTree,
) -> types.FunctionType:
    if id(portable) in functions:
        return functions[id(portable)][1]
    code = marshal.loads(portable.code)
    try:
        function_globals = importlib.import_module(
            portable.globals_module or portable.module
        ).__dict__
    except Exception:
        function_globals = {"__builtins__": builtins, "__name__": portable.module}
    closure = (
        tuple(
            _resolve_cell(value, cells, functions, tree) for value in portable.closure
        )
        if portable.closure is not None
        else None
    )
    defaults = cast(
        "tuple[object, ...] | None",
        _resolve_value(portable.defaults, cells, functions, tree),
    )
    fn = types.FunctionType(code, function_globals, portable.name, defaults, closure)
    fn.__qualname__ = portable.qualname
    fn.__module__ = portable.module
    fn.__kwdefaults__ = cast(
        "dict[str, object] | None",
        _resolve_value(portable.kwdefaults, cells, functions, tree),
    )
    functions[id(portable)] = portable, fn
    return fn


def resolve_portable_args(args: Sequence[object]) -> Sequence[object]:
    """Rebuild the functions :func:`make_portable_args` serialized by code."""
    tree = _PortableTree(args, PortableFunction)
    cells: dict[int, tuple[_PortableCell, types.CellType]] = {}
    functions: dict[int, tuple[PortableFunction, types.FunctionType]] = {}
    return cast("Sequence[object]", _resolve_value(args, cells, functions, tree))


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


_EMPTY_CLOSURE_CELL = object()


def _function_operand_values(
    fn: types.FunctionType,
) -> tuple[list[object], tuple[object, ...], dict[str, object]]:
    """Read populated operands without losing the positions of empty cells."""
    cells = []
    for cell in fn.__closure__ or ():
        try:
            cells.append(cell.cell_contents)
        except ValueError:
            cells.append(_EMPTY_CLOSURE_CELL)
    return (
        cells,
        fn.__defaults__ if fn.__defaults__ is not None else (),
        fn.__kwdefaults__ if fn.__kwdefaults__ is not None else {},
    )


def _input_tree_leaves(
    tree: object, *, graph: _InputCloneGraph | None = None
) -> Iterator[object]:
    """Pytree leaves, expanding record fields and callback tensor operands.

    Registered pytree nodes retain their own flattening contract. Opaque records
    and functions expose the same fields, closure cells and defaults as cloning.
    """
    graph = _InputCloneGraph() if graph is None else graph
    seen_operands: set[int] = set()
    active_nodes: set[int] = set()

    def walk(value: object) -> Iterator[object]:
        if id(value) in active_nodes:
            return
        children, spec = graph.flatten(value)
        if not spec.is_leaf():
            active_nodes.add(id(value))
            for child in children:
                yield from walk(child)
            active_nodes.remove(id(value))
        elif isinstance(value, (types.FunctionType, PortableFunction)):
            # Transport preserves function identity, so shared callback graphs
            # are traversed once per function in either representation.
            if id(value) in seen_operands:
                return
            seen_operands.add(id(value))
            if isinstance(value, types.FunctionType):
                cells, defaults, kwdefaults = graph.function_operands(value)
            else:
                cells = value.closure or ()
                defaults = value.defaults or ()
                kwdefaults = value.kwdefaults or {}
            for operand in (*cells, defaults, kwdefaults):
                yield from walk(operand)
        elif isinstance(value, _PortableCell):
            # Native callbacks expose contents directly; the transport wrapper
            # must not deduplicate them. Keep the same active-path cycle guard.
            active_nodes.add(id(value))
            yield from walk(value.contents)
            active_nodes.remove(id(value))
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            if id(value) not in seen_operands:
                seen_operands.add(id(value))
                for _, field in graph.record_fields(value):
                    yield from walk(field)
        else:
            yield value

    return walk(tree)


def _copy_dataclass_fields(
    record: DataclassInstance,
    map_field: Callable[[object], object],
    memo: dict[int, object],
    fields: Sequence[tuple[str, object]],
) -> object:
    result = object.__new__(type(record))
    memo[id(record)] = result
    # These are runtime records, so retain nonfield metadata as well as
    # declared fields. Read slot descriptors directly to avoid user hooks.
    if hasattr(record, "__dict__"):
        vars(result).update(vars(record))
    for cls in type(record).__mro__:
        for descriptor in vars(cls).values():
            if isinstance(descriptor, types.MemberDescriptorType):
                try:
                    value = descriptor.__get__(record, type(record))
                except AttributeError:  # An uninitialized nonfield slot.
                    continue
                descriptor.__set__(result, value)
    for name, value in fields:
        object.__setattr__(result, name, map_field(value))
    return result


def _flatten_input_node(tree: object) -> tuple[list[object], TreeSpec]:
    """Flatten one level so repeated containers remain observable."""
    root = True

    def is_child(value: object) -> bool:
        nonlocal root
        if root:
            root = False
            return False
        return True

    return tree_flatten(tree, is_leaf=is_child)


class _InputCloneGraph:
    """Snapshot each input node once and find paths to replaced leaves.

    Keep original objects and the first registered flatten result alive. A
    tensor-free cycle remains the original graph; a cycle reaching a replaced
    tensor is copied in its entirety, without invoking equality or copy hooks.
    """

    def __init__(self, *, include_functions: bool = False) -> None:
        self.include_functions = include_functions
        self.nodes: dict[int, tuple[object, list[object], TreeSpec]] = {}
        self.records: dict[int, tuple[tuple[str, object], ...]] = {}
        self.functions: dict[
            int, tuple[list[object], tuple[object, ...], dict[str, object]]
        ] = {}
        self.cells: dict[int, tuple[types.CellType, object]] = {}

    def flatten(self, value: object) -> tuple[list[object], TreeSpec]:
        if id(value) not in self.nodes:
            children, spec = _flatten_input_node(value)
            self.nodes[id(value)] = (value, children, spec)
        _, children, spec = self.nodes[id(value)]
        return children, spec

    def record_fields(self, value: DataclassInstance) -> tuple[tuple[str, object], ...]:
        if id(value) not in self.records:
            self.records[id(value)] = tuple(
                (field.name, getattr(value, field.name))
                for field in dataclasses.fields(value)
            )
        return self.records[id(value)]

    def function_operands(
        self, fn: types.FunctionType
    ) -> tuple[list[object], tuple[object, ...], dict[str, object]]:
        if id(fn) not in self.functions:
            cells, defaults, kwdefaults = _function_operand_values(fn)
            for index, (cell, value) in enumerate(
                zip(fn.__closure__ or (), cells, strict=True)
            ):
                # A cell shared by several callbacks has one snapshot, even
                # when its current value is an interned scalar or an empty cell.
                cells[index] = self.cells.setdefault(id(cell), (cell, value))[1]
            self.functions[id(fn)] = cells, defaults, kwdefaults
        return self.functions[id(fn)]

    def copy_ids(
        self, tree: object, replace_leaf: Callable[[object], bool]
    ) -> set[int]:
        parents: dict[int, set[int]] = {}
        visited: set[int] = set()
        changed: set[int] = set()
        pending = [tree]
        while pending:
            value = pending.pop()
            key = id(value)
            if key in visited:
                continue
            visited.add(key)
            children, spec = self.flatten(value)
            if spec.is_leaf():
                if replace_leaf(value):
                    changed.add(key)
                if isinstance(value, types.FunctionType) and self.include_functions:
                    _, defaults, kwdefaults = self.function_operands(value)
                    closure = value.__closure__ or ()
                    children = [*closure, defaults, kwdefaults]
                    # Rebuilding a function gives every closure cell a private
                    # copy. Include all owners of those exact cells, including
                    # callbacks that hold no replaced tensor themselves.
                    parents.setdefault(key, set()).update(id(cell) for cell in closure)
                elif isinstance(value, types.CellType) and key in self.cells:
                    contents = self.cells[key][1]
                    children = [] if contents is _EMPTY_CLOSURE_CELL else [contents]
                elif dataclasses.is_dataclass(value) and not isinstance(value, type):
                    children = [field for _, field in self.record_fields(value)]
            for child in children:
                parents.setdefault(id(child), set()).add(key)
                pending.append(child)
        work = list(changed)
        while work:
            for parent in parents.get(work.pop(), ()):
                if parent not in changed:
                    changed.add(parent)
                    work.append(parent)
        return changed


def _map_input_tree(
    tree: object,
    map_leaf: Callable[[object], object],
    memo: dict[int, tuple[object, object]],
    flatten_node: Callable[
        [object], tuple[list[object], TreeSpec]
    ] = _flatten_input_node,
    copy_ids: set[int] | None = None,
) -> object:
    """Rebuild each pytree node once, retaining the registered node contract.

    Retain each original in the memo too: a custom flatten function can produce
    temporary children whose Python ids must not be reused during traversal.
    """
    if copy_ids is not None and id(tree) not in copy_ids:
        return tree
    if id(tree) in memo:
        return memo[id(tree)][1]
    children, spec = flatten_node(tree)
    if spec.is_leaf():
        result = map_leaf(tree)
        memo[id(tree)] = (tree, result)
        return result

    if type(tree) is list:
        items: list[object] = []
        memo[id(tree)] = (tree, items)
        items.extend(
            _map_input_tree(child, map_leaf, memo, flatten_node, copy_ids)
            for child in children
        )
        return items
    if type(tree) is dict:
        values: dict[object, object] = {}
        memo[id(tree)] = (tree, values)
        values.update(
            zip(
                tree,
                (
                    _map_input_tree(child, map_leaf, memo, flatten_node, copy_ids)
                    for child in children
                ),
                strict=True,
            )
        )
        return values
    mapped = [
        _map_input_tree(child, map_leaf, memo, flatten_node, copy_ids)
        for child in children
    ]
    # An immutable node may have been reached again through a mutable
    # child. Reuse the node completed by that traversal.
    if id(tree) in memo:
        return memo[id(tree)][1]
    result = tree_unflatten(mapped, spec)
    memo[id(tree)] = (tree, result)
    return result


def _copy_function_operands(
    fn: types.FunctionType,
    map_operand: Callable[[object], object],
    memo: dict[int, object],
    cell_copies: dict[int, types.CellType],
    operands: tuple[list[object], tuple[object, ...], dict[str, object]],
) -> types.FunctionType:
    cells, defaults, kwdefaults = operands
    closure = (
        tuple(
            cell_copies.setdefault(id(cell), types.CellType())
            for cell in fn.__closure__
        )
        if fn.__closure__ is not None
        else None
    )
    rebuilt = types.FunctionType(
        fn.__code__, fn.__globals__, fn.__name__, None, closure
    )
    # Publish the shell before mapping operands: functions can participate
    # in the same cyclic record/container graph as their captured tensors.
    memo[id(fn)] = rebuilt
    for cell, value in zip(closure or (), cells, strict=True):
        if value is not _EMPTY_CLOSURE_CELL:
            cell.cell_contents = map_operand(value)
    if fn.__defaults__ is not None:
        rebuilt.__defaults__ = cast("tuple[object, ...]", map_operand(defaults))
    rebuilt.__qualname__ = fn.__qualname__
    rebuilt.__module__ = fn.__module__
    if fn.__kwdefaults__ is not None:
        rebuilt.__kwdefaults__ = cast("dict[str, object]", map_operand(kwdefaults))
    return rebuilt


def _clone_tensor_leaves(args: Sequence[object]) -> Sequence[object]:
    """``args`` with every tensor leaf on private storage, aliasing preserved.

    Tensors a function argument captures (an ``addmm`` epilogue's bias) or
    takes as defaults are operands of the kernel too, so a function leaf is
    rebuilt with its closure cells, defaults and keyword defaults cloned the
    same way. Functions outside the copied storage and shared-cell components
    keep their identity. Unregistered dataclasses expose their stored fields
    just as they do to the frontend.
    Nonfield instance state stays shallow, and no initialization/copy hook runs.
    """
    storage_clones: dict[int, torch.Tensor] = {}
    object_clones: dict[int, object] = {}
    tree_clones: dict[int, tuple[object, object]] = {}
    cell_clones: dict[int, types.CellType] = {}
    graph = _InputCloneGraph(include_functions=True)
    copy_ids = graph.copy_ids(args, lambda value: isinstance(value, torch.Tensor))

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
        if leaf.is_conj():
            view = view.conj()
        if leaf.is_neg():
            view = torch._neg_view(view)
        return view

    def clone_tree(tree: object) -> object:
        return _map_input_tree(tree, clone_leaf, tree_clones, graph.flatten, copy_ids)

    def clone_leaf(leaf: object) -> object:
        if id(leaf) in object_clones:
            return object_clones[id(leaf)]
        if isinstance(leaf, torch.Tensor):
            result = clone_tensor(leaf)
            object_clones[id(leaf)] = result
            return result
        if isinstance(leaf, types.FunctionType):
            return _copy_function_operands(
                leaf,
                clone_tree,
                object_clones,
                cell_clones,
                graph.function_operands(leaf),
            )
        if dataclasses.is_dataclass(leaf) and not isinstance(leaf, type):
            return _copy_dataclass_fields(
                leaf, clone_tree, object_clones, graph.record_fields(leaf)
            )
        return leaf

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
    Indices remain positions in the ordinary pytree flattening: an unregistered
    dataclass or function occupies one slot, which selects its stored fields
    or populated closure/default operands. Functions outside replaced storage
    and shared-cell paths retain their identity.
    """

    clone_indices = None if idx_to_clone is None else set(idx_to_clone)

    def _should_clone(idx: int) -> bool:
        return clone_indices is None or outer_indices[idx] in clone_indices

    args_flat: list[object] = []
    outer_indices: list[int] = []
    graph = _InputCloneGraph(include_functions=True)
    next_outer_index = 0
    active_nodes: set[int] = set()

    def collect(
        value: object,
        outer_index: int | None = None,
        seen_operands: set[int] | None = None,
    ) -> None:
        nonlocal next_outer_index
        if id(value) in active_nodes:
            return
        children, spec = graph.flatten(value)
        if not spec.is_leaf():
            active_nodes.add(id(value))
            for child in children:
                collect(child, outer_index, seen_operands)
            active_nodes.remove(id(value))
            return
        if outer_index is None:
            outer_index = next_outer_index
            next_outer_index += 1
            seen_operands = set()
        assert seen_operands is not None
        if isinstance(value, types.FunctionType):
            if id(value) not in seen_operands:
                seen_operands.add(id(value))
                cells, defaults, kwdefaults = graph.function_operands(value)
                for operand in (*cells, defaults, kwdefaults):
                    collect(operand, outer_index, seen_operands)
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            if id(value) not in seen_operands:
                seen_operands.add(id(value))
                for _, field in graph.record_fields(value):
                    collect(field, outer_index, seen_operands)
        else:
            args_flat.append(value)
            outer_indices.append(outer_index)

    collect(args)
    tensor_replacements: dict[int, torch.Tensor] = {}
    signal_pad_replacements: dict[int, int] = {}
    symmetric_tensor_ids: set[int] = set()

    for i, arg in enumerate(args_flat):
        if (
            _should_clone(i)
            and isinstance(arg, torch.Tensor)
            and is_symm_mem_tensor(arg, process_group_name)
        ):
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

    object_copies: dict[int, object] = {}
    cell_copies: dict[int, types.CellType] = {}
    tree_copies: dict[int, tuple[object, object]] = {}
    copy_ids = graph.copy_ids(
        args,
        lambda value: (
            id(value) in tensor_replacements
            or (isinstance(value, int) and value in signal_pad_replacements)
        ),
    )

    def replace_leaf(arg: object) -> object:
        if isinstance(arg, torch.Tensor) and id(arg) in tensor_replacements:
            return tensor_replacements[id(arg)]
        if isinstance(arg, int) and arg in signal_pad_replacements:
            return signal_pad_replacements[arg]
        if isinstance(arg, types.FunctionType):
            if id(arg) not in object_copies:
                _copy_function_operands(
                    arg,
                    replace_tree,
                    object_copies,
                    cell_copies,
                    graph.function_operands(arg),
                )
            return object_copies[id(arg)]
        if dataclasses.is_dataclass(arg) and not isinstance(arg, type):
            if id(arg) not in object_copies:
                _copy_dataclass_fields(
                    arg, replace_tree, object_copies, graph.record_fields(arg)
                )
            return object_copies[id(arg)]
        return arg

    def replace_tree(value: object) -> object:
        return _map_input_tree(
            value, replace_leaf, tree_copies, graph.flatten, copy_ids
        )

    return cast("Sequence[object]", replace_tree(args))
