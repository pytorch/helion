"""One host-sum site: the shared proof prelude, recognizer and guarded-call emitter.

The three host-sum tails (:mod:`host_paired_sum`, :mod:`host_single_sum`,
:mod:`host_full_sum`; driven in order by :mod:`host_sums`) prove their sites
on the typed host AST and, at generation time, guard the proven final
statements behind a runtime helper that returns ``None`` for every case it
does not cover::

    _helion_<kind> = _cute_try_<kind>(<sources>, ..., _launcher=_launcher)
    if _helion_<kind> is None:
        <original statements>
    [else: <bind the helper's result>]
    [<tail>]

This leaf module owns what the three share and imports none of them: the
knob's key and choices, the host-body prelude (:func:`prove_host_body`:
helper-shadow scan, bail-out statements, typed constexpr branch folding), the
typed ``sum(0).to(dtype)`` recognizer (:func:`sum_cast_value`), the
temporary-naming helpers and the :class:`HostSumSite` record one guarded
rewrite is emitted from.  The device kernels stay separate
(:mod:`helion.runtime.cute.paired_sum`, ``single_sum``, ``full_sum``).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch

from ..ast_extension import ExtendedAST
from ..ast_extension import statement_from_string
from ..type_info import LiteralType
from ..type_info import PythonModuleType
from ..type_info import TensorAttributeType
from ..type_info import TensorType
from ..variable_origin import ArgumentOrigin
from ..variable_origin import AttributeOrigin
from ..variable_origin import GetItemOrigin
from ..variable_origin import Origin
from ..variable_origin import TensorSizeOrigin
from ..variable_origin import TensorStrideOrigin

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..host_function import HostFunction
    from ..source_location import SourceLocation
    from ..type_info import TypeInfo

PAIRED_SUM_KEY = "cute_host_paired_sum"
PAIRED_SUM_CHOICES = ("off", "mapped", "narrow")


def node_type(node: ast.AST) -> TypeInfo | None:
    """The typed host AST's type for ``node`` (``None`` for untyped nodes)."""
    return node._type_info if isinstance(node, ExtendedAST) else None


def _constant_condition(node: ast.AST, host: HostFunction) -> bool | None:
    """The fixed truth value of an ``if`` test, or ``None`` when it can vary.

    A ``bool`` constant, or a test the type propagator folded to a boolean
    literal whose every operand is fixed by the bound kernel's specialization
    key (``_keyed_test``): constexpr arguments, ``str`` / ``dtype`` / ``device``
    arguments (keyed by value), tensor ``dtype`` / ``shape`` / ``stride``
    metadata (keyed by ``_tensor_key``; static shapes), ``torch`` constants,
    and host names bound once to such expressions (``BT, V = y_pred.shape``,
    ``n = x.size(0)``).  A literal the key does not cover is not folded: a
    tensor's ``device`` (``x.device.type == "cuda"`` is a literal at compile
    time, but ``_tensor_key`` carries no device, so another call may run the
    other branch), ``is_cuda``, a module global.  A call in the test keeps the
    branch dynamic too (dropping the test with the untaken branch would skip
    the call).
    """
    if isinstance(node, ast.Constant) and type(node.value) is bool:
        return node.value
    info = node_type(node)
    if (
        isinstance(info, LiteralType)
        and type(info.value) is bool
        and _keyed_test(node, host)
    ):
        return info.value
    return None


# Tensor metadata ``_tensor_key`` fixes (dtype; sizes and strides under static
# shapes, the only regime where they are literals), as attributes and methods.
_KEYED_TENSOR_ATTRS = frozenset({"dtype", "shape", "ndim"})
_KEYED_TENSOR_METHODS = frozenset({"size", "stride", "dim", "numel"})
# Argument types ``_specialization_key`` keys by value (ints, floats and bools
# key by type only and are symbolic on the host anyway).
_KEYED_ARGUMENT_TYPES = (str, torch.dtype, torch.device)
_BINDING_DEPTH = 8


def _keyed_test(node: ast.AST, host: HostFunction) -> bool:
    """Whether every operand of the test is fixed by the specialization key."""
    return _keyed_expression(node, host, _single_bindings(host), 0)


def _single_bindings(host: HostFunction) -> dict[str, ast.expr]:
    """Host names bound exactly once, by a plain ``name = <expr>`` assignment."""
    module = ast.Module(body=list(host.body), type_ignores=[])
    stores: dict[str, int] = {}
    values: dict[str, ast.expr] = {}
    for child in ast.walk(module):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            stores[child.id] = stores.get(child.id, 0) + 1
        elif (
            isinstance(child, ast.Assign)
            and len(child.targets) == 1
            and isinstance(child.targets[0], ast.Name)
        ):
            values[child.targets[0].id] = child.value
    arguments = {
        argument.arg
        for argument in ast.walk(host.args)
        if isinstance(argument, ast.arg)
    }
    return {
        name: value
        for name, value in values.items()
        if stores.get(name) == 1 and name not in arguments
    }


def _keyed_expression(
    node: ast.AST, host: HostFunction, bindings: dict[str, ast.expr], depth: int
) -> bool:
    if depth > _BINDING_DEPTH:
        return False
    if isinstance(node, ast.Constant):
        return True
    keyed = lambda child: _keyed_expression(child, host, bindings, depth)  # noqa: E731
    if isinstance(node, ast.Name):
        info = node_type(node)
        if not isinstance(info, LiteralType):
            return False
        if _keyed_origin(info.origin, info.value, host):
            return True
        bound = bindings.get(node.id)
        return bound is not None and _keyed_expression(bound, host, bindings, depth + 1)
    if isinstance(node, ast.Attribute):
        base = node.value
        base_info = node_type(base)
        if isinstance(base_info, TensorType):
            return node.attr in _KEYED_TENSOR_ATTRS
        if isinstance(base_info, PythonModuleType):
            # ``torch.float16``: a constant of the torch module.
            return base_info.value is torch
        if isinstance(base_info, LiteralType):
            # An attribute of a keyed value (``dev.type`` of a device argument).
            return keyed(base)
        return False
    if isinstance(node, ast.Subscript):
        return keyed(node.value) and keyed(node.slice)
    if isinstance(node, ast.Call):
        func = node.func
        return (
            isinstance(func, ast.Attribute)
            and isinstance(node_type(func.value), TensorType)
            and func.attr in _KEYED_TENSOR_METHODS
            and not node.keywords
            and all(keyed(argument) for argument in node.args)
        )
    if isinstance(node, ast.Compare):
        return keyed(node.left) and all(keyed(other) for other in node.comparators)
    if isinstance(node, ast.BoolOp):
        return all(keyed(value) for value in node.values)
    if isinstance(node, ast.UnaryOp):
        return keyed(node.operand)
    if isinstance(node, ast.BinOp):
        return keyed(node.left) and keyed(node.right)
    if isinstance(node, ast.Tuple):
        return all(keyed(element) for element in node.elts)
    return False


def _keyed_origin(origin: Origin, value: object, host: HostFunction) -> bool:
    """Whether a literal's provenance is a keyed argument or keyed tensor metadata."""
    keys: list[str] = []
    while isinstance(
        origin, (AttributeOrigin, GetItemOrigin, TensorSizeOrigin, TensorStrideOrigin)
    ):
        if isinstance(origin, AttributeOrigin):
            keys.append(origin.key)
        elif isinstance(origin, TensorSizeOrigin):
            keys.append("size")
        elif isinstance(origin, TensorStrideOrigin):
            keys.append("stride")
        origin = origin.value
    if not isinstance(origin, ArgumentOrigin):
        return False
    if origin.name in host.constexpr_args:
        return True
    argument = host.params.arguments.get(origin.name)
    if isinstance(argument, torch.Tensor):
        # ``x.shape[0]``, ``x.dtype``, ``x.size(0)``; the leading key names the
        # metadata, deeper keys index it (``shape[0]``) or read its own
        # attributes (``dtype.itemsize``).
        return bool(keys) and keys[-1] in _KEYED_TENSOR_ATTRS | _KEYED_TENSOR_METHODS
    if keys:
        return isinstance(argument, _KEYED_ARGUMENT_TYPES)
    return isinstance(value, _KEYED_ARGUMENT_TYPES)


def fold_constexpr_branches(
    body: Sequence[ast.AST], conditions: dict[SourceLocation, tuple[str, bool]]
) -> list[ast.AST]:
    """``body`` with every ``if`` in ``conditions`` replaced by its taken branch
    (recursively); the proofs walk this view and the rewrites fold the final
    generated body the same way."""
    result: list[ast.AST] = []
    for statement in body:
        if (
            isinstance(statement, ast.If)
            and isinstance(statement, ExtendedAST)
            and (condition := conditions.get(statement._location)) is not None
            and (
                ast.dump(statement.test) == condition[0]
                or (
                    isinstance(statement.test, ast.Constant)
                    and type(statement.test.value) is bool
                    and statement.test.value is condition[1]
                )
            )
        ):
            selected = statement.body if condition[1] else statement.orelse
            result.extend(fold_constexpr_branches(selected, conditions))
        else:
            result.append(statement)
    return result


@dataclass(frozen=True)
class HostBodyFacts:
    """A host function's typed body, folded through its constexpr branches.

    ``conditions`` records every typed ``if`` whose test is a constant or a
    constexpr argument, so the final generated body can be folded the same
    way.  ``defined`` is the proof's working set: it starts as the argument
    names and each kind's proof adds, in place, the names the statements it
    walks past bind (a fresh set per :func:`prove_host_body` call; the record
    itself is never rebound).
    """

    conditions: dict[SourceLocation, tuple[str, bool]]
    folded: list[ast.AST]
    defined: set[str]


def prove_host_body(
    host: HostFunction, helper_names: tuple[str, ...]
) -> HostBodyFacts | None:
    """The shared prelude of the three host-sum proofs.

    The generated helper is a module import. Any lexical binding of one of
    ``helper_names``, even an assignment after the site, can shadow that
    import, so such programs keep their original host path before any choice
    or seed is exposed. Bodies with ``del``, ``try``, ``global`` or
    ``nonlocal`` are not searched either. Only typed constexpr branches are
    folded; dynamic bodies are not searched.
    """
    nodes = list(ast.walk(ast.Module(body=host.body, type_ignores=[])))
    if host.name in helper_names or any(
        (isinstance(node, ast.Name) and node.id in helper_names)
        or (isinstance(node, ast.arg) and node.arg in helper_names)
        or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name in helper_names
        )
        or (
            isinstance(node, ast.alias)
            and (node.asname or node.name.split(".", 1)[0]) in helper_names
        )
        for node in (*nodes, *ast.walk(host.args))
    ):
        return None
    if any(
        isinstance(node, (ast.Delete, ast.Try, ast.Global, ast.Nonlocal))
        for node in nodes
    ):
        return None
    conditions = {
        node._location: (ast.dump(node.test), condition)
        for node in nodes
        if isinstance(node, ast.If)
        and isinstance(node, ExtendedAST)
        and (condition := _constant_condition(node.test, host)) is not None
    }
    return HostBodyFacts(
        conditions,
        fold_constexpr_branches(host.body, conditions),
        {argument.arg for argument in host.args.args},
    )


def assigned_names(statement: ast.AST) -> set[str]:
    """Names a top-level ``Assign`` binds (the proofs' ``defined`` tracking)."""
    if not isinstance(statement, ast.Assign):
        return set()
    return {target.id for target in statement.targets if isinstance(target, ast.Name)}


@dataclass(frozen=True)
class SumCastValue:
    """Typed, effect-free arguments to a canonical Tensor sum/cast expression."""

    source: str
    dtype_argument: str
    dtype_is_tensor: bool
    dtype: torch.dtype


def sum_cast_value(
    call: ast.AST,
    dtypes: tuple[torch.dtype, ...] = (torch.bfloat16, torch.float32),
) -> SumCastValue | None:
    """``<2-D FP32 name>.sum(0).to(<dtype>)`` with every node typed as such."""
    if not (
        isinstance(call, ast.Call)
        and not call.keywords
        and len(call.args) == 1
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "to"
        and isinstance(reduce := call.func.value, ast.Call)
        and not reduce.keywords
        and len(reduce.args) == 1
        and isinstance(reduce.args[0], ast.Constant)
        and type(reduce.args[0].value) is int
        and reduce.args[0].value == 0
        and isinstance(reduce.func, ast.Attribute)
        and reduce.func.attr == "sum"
        and isinstance(source := reduce.func.value, ast.Name)
    ):
        return None
    source_type = node_type(source)
    sum_method, cast_method = node_type(reduce.func), node_type(call.func)
    sum_type, cast_type, dtype_type = (
        node_type(reduce),
        node_type(call),
        node_type(call.args[0]),
    )
    if not (
        isinstance(source_type, TensorType)
        and source_type.fake_value.ndim == 2
        and source_type.fake_value.dtype == torch.float32
        and isinstance(sum_method, TensorAttributeType)
        and sum_method.attr() == "sum"
        and isinstance(cast_method, TensorAttributeType)
        and cast_method.attr() == "to"
        and isinstance(sum_type, TensorType)
        and sum_type.fake_value.ndim == 1
        and sum_type.fake_value.dtype == torch.float32
        and isinstance(cast_type, TensorType)
        and cast_type.fake_value.ndim == 1
        and isinstance(dtype_type, LiteralType)
        and isinstance(dtype_type.value, torch.dtype)
        and dtype_type.value in dtypes
        and cast_type.fake_value.dtype == dtype_type.value
    ):
        return None
    dtype_expression = call.args[0]
    # Pass a dtype owner, rather than reading its attribute ahead of the first
    # original sum. The runtime gate checks exact Tensor identity before reading
    # metadata; a subclass or unsupported case executes the original AST.
    if (
        isinstance(dtype_expression, ast.Attribute)
        and dtype_expression.attr == "dtype"
        and isinstance(dtype_expression.value, ast.Name)
        and isinstance(node_type(dtype_expression.value), TensorType)
    ):
        dtype_argument = dtype_expression.value.id
        dtype_is_tensor = True
    elif (
        isinstance(dtype_expression, ast.Attribute)
        and isinstance(dtype_expression.value, ast.Name)
        and isinstance(module_type := node_type(dtype_expression.value), LiteralType)
        and module_type.value is torch
        and dtype_expression.attr in ("float16", "bfloat16", "float32")
    ):
        dtype_argument = ast.unparse(dtype_expression)
        dtype_is_tensor = False
    else:
        return None
    return SumCastValue(
        source.id,
        dtype_argument,
        dtype_is_tensor,
        dtype_type.value,
    )


def bound_names(host: HostFunction, folded: Sequence[ast.AST]) -> set[str]:
    """Every name in the final folded body plus the arguments (temporaries
    must not collide with them)."""
    names = {
        node.id
        for statement in folded
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }
    names.update(argument.arg for argument in host.args.args)
    return names


def fresh_temporary(stem: str, names: set[str]) -> str:
    """``stem`` with trailing underscores until it is unbound; records it."""
    temporary = stem
    while temporary in names:
        temporary += "_"
    names.add(temporary)
    return temporary


@dataclass(frozen=True)
class HostSumSite:
    """One guarded rewrite: the helper call, the original statements it
    guards, and what runs when the helper returned a result."""

    temporary: str
    call: str
    originals: tuple[ast.AST, ...]
    orelse: tuple[ast.stmt, ...] = ()
    tail: tuple[ast.stmt, ...] = ()

    def emit(self) -> list[ast.AST]:
        return [
            statement_from_string(f"{self.temporary} = {self.call}"),
            ast.If(
                test=ast.Compare(
                    ast.Name(self.temporary, ast.Load()),
                    [ast.Is()],
                    [ast.Constant(None)],
                ),
                body=cast("list[ast.stmt]", list(self.originals)),
                orelse=list(self.orelse),
            ),
            *self.tail,
        ]
