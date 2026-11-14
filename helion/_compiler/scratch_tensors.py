"""Kernel-local scratch tensors: host allocations only the kernel's own loads
and stores reach.

A kernel may allocate a workspace in its host code and use it only inside
its device loop: Inductor's looped reductions store each row to a scratch
tensor in the first sweep and read it back in the second, and ports of
Inductor kernels keep that shape::

    ws = torch.empty([rows, cols], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(rows):
        ws[tile_r, :] = ...
        ... = ws[tile_r, :]
    return out

Nothing outside the kernel can observe such a tensor.  A backend that no
longer loads from it (it forwards the stored values from registers) may drop
its stores as well, and the allocation then goes with the kernel argument
(``GenerateAST.host_dead_code_elimination``).  That goes beyond Inductor's
structure, which always writes the workspace; the results are unchanged.

The analysis is conservative and declines whatever it does not prove.  A
host name is kernel-local scratch only when

- a top-level statement of the host function binds it, once in the whole
  function, to ``torch.empty(...)``, ``torch.empty_like(...)`` or a
  tensor's ``x.new_empty(...)``, as type propagation resolved the call and
  looked up on the ``torch`` module, a tensor or a bare name, without
  ``out=`` or ``**kwargs``;
- evaluating the call's arguments runs no user code, by the types type
  propagation gave them: constants, names of numbers, tensors, dtypes,
  devices and tuples or lists of those, a tensor's ``shape``, ``device``,
  ``dtype``, ``size``, ``stride`` and ``dim``, a ``torch`` dtype, tuples
  and lists, subscripts of a tuple or list by a number (or a slice of
  numbers) and arithmetic on numbers;
- the numbers, tensors, tuples and lists those names hold are of exact
  types, which type propagation does not show (``CompileEnvironment.to_fake``
  makes any ``int`` a plain number, a subclass's instance included, and a
  tensor subclass a plain fake tensor): an argument, an item of one or a
  tensor argument's metadata (bind refuses subclasses of numbers, lists
  and tuples, and every tensor argument must be a plain ``torch.Tensor``
  or ``nn.Parameter`` when the kernel binds), a module's global of an exact
  type, or a name bound once at the top level to such an expression;
- it is not a parameter, and no tensor of another host origin shares its
  fake storage;
- every other occurrence of it subscripts it (a load, a store target or an
  in-place update) inside the body of one grid loop (grid loops are
  top-level statements, ``NestedGridLoop``): it is never returned, aliased,
  viewed, passed to a call, read through an attribute, deleted, or used by
  host code or by a second grid loop.
"""

from __future__ import annotations

import ast
import dataclasses
import types
from typing import TYPE_CHECKING
from typing import TypeGuard

import torch

from .ast_extension import ExtendedAST
from .ast_extension import LoopType
from .compile_environment import CompileEnvironment
from .device_function import TensorArg
from .variable_origin import ArgumentOrigin
from .variable_origin import AttributeOrigin
from .variable_origin import GetItemOrigin
from .variable_origin import GlobalOrigin
from .variable_origin import NameOrigin
from .variable_origin import TensorSizeOrigin
from .variable_origin import TensorStrideOrigin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .device_function import DeviceFunction
    from .host_function import HostFunction
    from .variable_origin import Origin

_TORCH_FACTORIES = (torch.empty, torch.empty_like)
_METHOD_FACTORIES = frozenset({"new_empty"})
# Tensor methods and attributes that read metadata only, admitted in a
# factory's arguments.
_METADATA_METHODS = frozenset({"size", "stride", "dim"})
_METADATA_ATTRIBUTES = frozenset({"shape", "device", "dtype"})
# The exact types an argument or global may hold for a factory to read it.
_EXACT_TENSOR_TYPES = (torch.Tensor, torch.nn.Parameter)
_EXACT_TYPES = (bool, int, float, *_EXACT_TENSOR_TYPES)


@dataclasses.dataclass(frozen=True)
class ScratchTensors:
    """The kernel-local scratch tensors of a host function."""

    names: frozenset[str]
    """Their host names."""
    storages: frozenset[int]
    """``id(fake.untyped_storage())`` of their fake tensors."""

    def holds(self, tensor: torch.Tensor) -> bool:
        return id(tensor.untyped_storage()) in self.storages


def _type_of(expr: ast.AST) -> object:
    return expr._type_info if isinstance(expr, ExtendedAST) else None


def _is_number(expr: ast.expr) -> bool:
    """Whether type propagation knows ``expr`` as a Python number: arithmetic
    on it calls no user ``__add__``."""
    from .type_info import LiteralType
    from .type_info import NumericType

    kind = _type_of(expr)
    return isinstance(kind, NumericType) or (
        isinstance(kind, LiteralType) and isinstance(kind.value, (bool, int, float))
    )


def _is_value(kind: object) -> bool:
    """Whether a value of type ``kind`` reaches a factory without running
    user code: a number, a tensor, a dtype, a device, ``None``, or a tuple or
    list of those (a shape), of an exact type (``_Provenance``).  Any other
    object may define ``__index__``, ``__len__`` or ``__iter__``, which the
    factory would call."""
    from .type_info import LiteralType
    from .type_info import NumericType
    from .type_info import SequenceType
    from .type_info import TensorType

    if isinstance(kind, (NumericType, TensorType)):
        return True
    if isinstance(kind, SequenceType):
        return all(_is_value(element) for element in kind.element_types)
    return isinstance(kind, LiteralType) and (
        kind.value is None
        or type(kind.value) in (bool, int, float, torch.dtype, torch.device)
    )


def _is_torch(expr: ast.expr) -> bool:
    """Whether ``expr`` is a name of the ``torch`` module, whose attributes a
    module ``__getattr__`` does not compute."""
    from .type_info import PythonModuleType

    kind = _type_of(expr)
    return (
        isinstance(expr, ast.Name)
        and isinstance(kind, PythonModuleType)
        and kind.value is torch
    )


@dataclasses.dataclass
class _Provenance:
    """Where the numbers, tensors, tuples and lists that names hold come
    from: a type propagation shows the same for a subclass's instance.

    A value is of an exact type when it is an argument, an item of one or a
    tensor argument's metadata (bind refuses subclasses of numbers, lists
    and tuples, and ``_kernel_local_scratch`` refuses a tensor argument of a
    subclass), a module's global of an exact type, or a name in
    ``exact_names``: bound once, at the top level, to a plain expression
    (``_is_plain``) before the statement asking.
    """

    host: HostFunction
    exact_names: set[str] = dataclasses.field(default_factory=set)

    def holds_an_exact_value(self, name: ast.Name) -> bool:
        from .type_info import LiteralType
        from .type_info import TypeInfo

        kind = _type_of(name)
        # A literal's value is the object itself, whose type ``_is_value``
        # checks.
        return (
            isinstance(kind, LiteralType)
            or name.id in self.exact_names
            or (isinstance(kind, TypeInfo) and self._exact_origin(kind.origin))
        )

    def _exact_origin(self, origin: Origin) -> bool:
        if isinstance(origin, ArgumentOrigin):
            return True
        if isinstance(origin, AttributeOrigin) and isinstance(
            origin.value, GlobalOrigin
        ):
            imported = self.host.global_imports.get(origin.value.name)
            module = imported.value if imported is not None else None
            # The module's own dictionary: no ``__getattr__`` runs.
            return isinstance(module, types.ModuleType) and (
                type(vars(module).get(origin.key)) in _EXACT_TYPES
            )
        if isinstance(origin, (GetItemOrigin, TensorSizeOrigin, TensorStrideOrigin)):
            return self._exact_origin(origin.value)
        return (
            isinstance(origin, AttributeOrigin)
            and origin.key in _METADATA_ATTRIBUTES
            and self._exact_origin(origin.value)
        )


def _is_index(expr: ast.expr, provenance: _Provenance) -> bool:
    """A subscript's index that calls no ``__index__``: a plain number, or a
    slice of them."""
    if isinstance(expr, ast.Slice):
        parts = (expr.lower, expr.upper, expr.step)
        return all(part is None or _is_index(part, provenance) for part in parts)
    return _is_number(expr) and _is_plain(expr, provenance)


def _is_plain(expr: ast.expr, provenance: _Provenance) -> bool:
    """Whether ``expr`` evaluates without running user code that deleting its
    statement would lose, by the types type propagation gave its parts:
    constants, names of values (``_is_value``) of exact types
    (``_Provenance``), tuples and lists, a tensor's ``shape``, ``device``
    and ``dtype``, a ``torch`` dtype, a tensor's ``size``, ``stride`` or
    ``dim``, subscripts of a tuple or list by a number or a slice of
    numbers, and arithmetic on numbers.  An attribute of any other object
    may be a property or a module's ``__getattr__``, and a subscript or an
    operator of one calls its methods."""
    from .type_info import LiteralType
    from .type_info import SequenceType
    from .type_info import TensorAttributeType
    from .type_info import TensorType

    if isinstance(expr, ast.Constant):
        return True
    if isinstance(expr, ast.Name):
        return _is_value(_type_of(expr)) and provenance.holds_an_exact_value(expr)
    if isinstance(expr, ast.Starred):
        return _is_plain(expr.value, provenance)
    if isinstance(expr, ast.Attribute):
        owner, kind = _type_of(expr.value), _type_of(expr)
        return (
            isinstance(owner, TensorType)
            and expr.attr in _METADATA_ATTRIBUTES
            and _is_plain(expr.value, provenance)
        ) or (
            _is_torch(expr.value)
            and isinstance(kind, LiteralType)
            and isinstance(kind.value, torch.dtype)
        )
    if isinstance(expr, ast.Subscript):
        return (
            isinstance(_type_of(expr.value), SequenceType)
            and _is_plain(expr.value, provenance)
            and _is_index(expr.slice, provenance)
        )
    if isinstance(expr, (ast.Tuple, ast.List)):
        return all(_is_plain(elt, provenance) for elt in expr.elts)
    if isinstance(expr, ast.BinOp):
        return all(
            _is_number(operand) and _is_plain(operand, provenance)
            for operand in (expr.left, expr.right)
        )
    if isinstance(expr, ast.UnaryOp):
        return _is_number(expr.operand) and _is_plain(expr.operand, provenance)
    if isinstance(expr, ast.Call) and isinstance(expr.func, ExtendedAST):
        method = expr.func._type_info
        return (
            isinstance(method, TensorAttributeType)
            and method.attr() in _METADATA_METHODS
            and _call_parts_are_plain(expr, provenance)
        )
    return False


def _call_parts_are_plain(call: ast.Call, provenance: _Provenance) -> bool:
    """Whether a call its caller resolved by type (a factory, a metadata
    method) is looked up without running user code, on the ``torch`` module,
    a tensor or as a bare name, and its arguments are ``_is_plain``."""
    from .type_info import TensorType

    func = call.func
    return (
        (
            isinstance(func, ast.Name)
            or (
                isinstance(func, ast.Attribute)
                and (
                    _is_torch(func.value)
                    or (
                        isinstance(_type_of(func.value), TensorType)
                        and _is_plain(func.value, provenance)
                    )
                )
            )
        )
        and all(_is_plain(arg, provenance) for arg in call.args)
        and all(
            keyword.arg is not None and _is_plain(keyword.value, provenance)
            for keyword in call.keywords
        )
    )


def is_allocation_call(value: ast.expr) -> TypeGuard[ast.Call]:
    """A call of ``torch.empty``, ``torch.empty_like`` or a tensor's
    ``new_empty``, as type propagation resolved it rather than as spelled,
    without ``out=`` (which would alias the result to another tensor)."""
    from .type_info import CallableType
    from .type_info import TensorAttributeType

    if not (isinstance(value, ast.Call) and isinstance(value.func, ExtendedAST)):
        return False
    func = value.func._type_info
    resolved = (
        isinstance(func, CallableType)
        and any(func.value is factory for factory in _TORCH_FACTORIES)
    ) or (isinstance(func, TensorAttributeType) and func.attr() in _METHOD_FACTORIES)
    return resolved and not any(keyword.arg == "out" for keyword in value.keywords)


def _is_factory(value: ast.expr, provenance: _Provenance) -> bool:
    """An allocation call (``is_allocation_call``) without ``**kwargs`` whose
    lookup and arguments run no user code (``_call_parts_are_plain``): an
    unread scratch tensor's whole allocation statement is deleted.
    ``cute/promote_output_axis._fresh_tensors`` proves the freshness of host
    allocations for the CuTe passes by rules of its own (more factories, no
    confinement); keep the two in step."""
    return is_allocation_call(value) and _call_parts_are_plain(value, provenance)


def _parameters(arguments: ast.arguments) -> set[str]:
    names = {
        arg.arg
        for arg in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    }
    for extra in (arguments.vararg, arguments.kwarg):
        if extra is not None:
            names.add(extra.arg)
    return names


def _bindings(tree: ast.AST) -> dict[str, int]:
    """How many times each name is bound anywhere in ``tree``: assignment
    targets, loop, ``with``, ``except`` and comprehension targets, walrus,
    ``del``, imports, definitions, ``match`` captures and
    ``global``/``nonlocal`` declarations."""
    counts: dict[str, int] = {}

    def bind(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bind(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bind(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bind((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                bind(name)
        elif (
            isinstance(node, ast.ExceptHandler)
            and node.name is not None
            or isinstance(node, (ast.MatchAs, ast.MatchStar))
            and node.name is not None
        ):
            bind(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest is not None:
            bind(node.rest)
        elif isinstance(node, ast.arg):
            bind(node.arg)
    return counts


def _subscripted_names(loop: ast.For) -> dict[str, int]:
    """Names the loop's body subscripts, with the number of such uses: the
    value of a load, store or in-place update subscript."""
    counts: dict[str, int] = {}
    for stmt in loop.body:
        for node in ast.walk(stmt):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and not isinstance(node.ctx, ast.Del)
            ):
                counts[node.value.id] = counts.get(node.value.id, 0) + 1
    return counts


def _loads(tree: ast.AST) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            counts[node.id] = counts.get(node.id, 0) + 1
    return counts


_NO_SCRATCH = ScratchTensors(frozenset(), frozenset())


def find_kernel_local_scratch(host: HostFunction) -> None:
    """Find the kernel-local scratch tensors of ``host`` as the compiler
    builds it (``KernelCompiler.compile``), while the binding's arguments are
    at hand (``_tensor_arguments_are_exact``), and from its body as written:
    code generation adds a ``device=`` keyword to a factory call that lacks
    one in place (``GenerateAST._inject_device_kwarg``), a keyword without a
    type, which ``_is_factory`` would refuse."""
    host.scratch_tensors = _kernel_local_scratch(host)


def kernel_local_scratch(host: HostFunction) -> ScratchTensors:
    """The kernel-local scratch tensors of ``host`` (see the module docstring)
    that ``find_kernel_local_scratch`` found.

    Empty for one stage of a larger host program (``partial_program``): its
    body is not the whole program, and another stage may read what it
    allocates.  Empty too for a host function the compiler did not build."""
    if host.partial_program or host.scratch_tensors is None:
        return _NO_SCRATCH
    return host.scratch_tensors


def _tensor_arguments_are_exact() -> bool:
    """Whether every tensor argument of the binding is a plain
    ``torch.Tensor`` or ``nn.Parameter``, as the example inputs show.  A
    subclass's factories, metadata and indexing run its
    ``__torch_function__`` (its ``new_empty`` may return a view of an
    input), and type propagation shows its instance as a plain fake tensor.
    A binding without the example inputs (an outer trace's fake inputs)
    finds no scratch."""
    env = CompileEnvironment.current()
    for tensor in env.input_sources:
        source = env.tensor_input_source(tensor)
        example = None if source is None else env.live_example_input(source)
        if example is None or type(example) not in _EXACT_TENSOR_TYPES:
            return False
    return True


def _kernel_local_scratch(host: HostFunction) -> ScratchTensors:
    if not _tensor_arguments_are_exact():
        return _NO_SCRATCH
    module = ast.Module(body=list(host.body), type_ignores=[])
    bindings = _bindings(module)
    parameters = _parameters(host.args)
    # One pass in program order: a factory reads names bound before it.
    provenance = _Provenance(host)
    candidates: set[str] = set()
    for stmt in host.body:
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            continue
        name = stmt.targets[0].id
        if bindings.get(name) != 1 or name in parameters:
            continue
        if _is_factory(stmt.value, provenance):
            candidates.add(name)
        elif _is_plain(stmt.value, provenance):
            provenance.exact_names.add(name)
    if not candidates:
        return _NO_SCRATCH
    # Every read of a candidate must be a subscript in the body of one
    # top-level grid loop: the reads anywhere equal its subscripts there.
    all_loads = _loads(module)
    grid_loops = [
        stmt
        for stmt in host.body
        if isinstance(stmt, ast.For)
        and isinstance(stmt, ExtendedAST)
        and stmt._loop_type is LoopType.GRID
    ]
    # Per grid loop: the names it reads, and the names its body subscripts.
    loop_uses = [(_loads(loop), _subscripted_names(loop)) for loop in grid_loops]
    names: set[str] = set()
    for name in candidates:
        users = [uses for uses in loop_uses if uses[0].get(name, 0) > 0]
        if len(users) != 1:
            continue
        ((_reads, subscripted),) = users
        if subscripted.get(name, 0) == all_loads.get(name, 0):
            names.add(name)
    # A name whose fake storage a tensor of another host origin shares (an
    # input's, say) is not a fresh allocation, however it was spelled.
    storages: dict[str, set[int]] = {}
    other_storages: set[int] = set()
    for tensor, origin in host.tensor_to_origin.items():
        storage = id(tensor.untyped_storage())
        if isinstance(origin, NameOrigin) and origin.name in names:
            storages.setdefault(origin.name, set()).add(storage)
        elif origin.is_host():
            other_storages.add(storage)
    names = {
        name
        for name in names
        if name in storages and storages[name].isdisjoint(other_storages)
    }
    if not names:
        return _NO_SCRATCH
    return ScratchTensors(
        frozenset(names),
        frozenset(storage for name in names for storage in storages[name]),
    )


def drop_unused_scratch_arguments(
    host: HostFunction,
    device_function: DeviceFunction,
    kernel_def: Sequence[ast.stmt],
    host_statements: Sequence[ast.AST],
) -> set[str]:
    """Drop the parameter, and the launcher's argument, of each kernel-local
    scratch tensor the finished kernel no longer reads.

    The kernel parameters are fixed before a backend's late passes run on the
    body (``DeviceFunction.codegen_function_def``); a pass that forwarded every
    load of a scratch tensor and dropped its stores leaves the parameter
    behind.  Returns the host names whose parameters went, for
    ``GenerateAST.host_dead_code_elimination`` to drop their allocations.
    A single-root kernel only, on a backend that opts in
    (``Backend.drops_unread_scratch_arguments``: the removal shifts every
    later positional argument of the launch); a parameter or launcher
    argument that does not appear exactly once as its plain name stays.  Only
    the tensor's own parameter goes: the size and stride parameters of a
    kernel compiled with dynamic shapes stay, with the launch's reads of the
    tensor, so its allocation stays too.
    """
    if (
        len(host.device_ir.root_ids) != 1
        or not CompileEnvironment.current().backend.drops_unread_scratch_arguments
    ):
        return set()
    definitions = [
        stmt
        for stmt in kernel_def
        if isinstance(stmt, ast.FunctionDef) and stmt.name == device_function.name
    ]
    if len(definitions) != 1:
        return set()
    (definition,) = definitions
    read = {
        node.id
        for stmt in definition.body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    unread = [
        arg
        for arg in device_function.arguments
        if isinstance(arg, TensorArg)
        and arg.name not in read
        and arg.name not in device_function.placeholder_args
    ]
    # The finished kernel reads every tensor parameter unless a late pass
    # dropped all the stores and loads of one.
    if not unread:
        return set()
    calls = [
        stmt.value
        for stmt in host_statements
        if isinstance(stmt, ExtendedAST)
        and stmt._is_kernel_call
        and isinstance(stmt, (ast.Expr, ast.Assign))
        and isinstance(stmt.value, ast.Call)
    ]
    if len(calls) != 1:
        return set()
    scratch = kernel_local_scratch(host)
    if not scratch.names:
        return set()
    (launcher,) = calls
    dropped: dict[str, str] = {}
    for arg in unread:
        host_name = arg._host_str
        if host_name is None or host_name not in scratch.names:
            continue
        params = [param for param in definition.args.args if param.arg == arg.name]
        positions = [
            position
            for position, node in enumerate(launcher.args)
            if isinstance(node, ast.Name) and node.id == host_name
        ]
        if len(params) != 1 or len(positions) != 1:
            continue
        definition.args.args.remove(params[0])
        del launcher.args[positions[0]]
        dropped[arg.name] = host_name
    device_function.drop_arguments(set(dropped))
    return set(dropped.values())
