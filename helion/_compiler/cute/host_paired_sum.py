"""Typed proof and final-host rewrite for independent sum/cast pairs."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
_PAIRED_SUM_HELPER = "_cute_try_paired_sum_cast"


def _type(node: ast.AST) -> TypeInfo | None:
    return node._type_info if isinstance(node, ExtendedAST) else None


@dataclass(frozen=True)
class SumCastValue:
    """Typed, effect-free arguments to a canonical Tensor sum/cast expression."""

    source: str
    dtype_argument: str
    dtype_is_tensor: bool
    dtype: torch.dtype


@dataclass(frozen=True)
class SumCast:
    """A typed assignment whose final generated AST must remain unchanged."""

    location: SourceLocation
    statement: str
    target: str
    source: str
    dtype_argument: str
    dtype_is_tensor: bool
    dtype: torch.dtype


def _sum_cast_value(
    cast: ast.AST,
    dtypes: tuple[torch.dtype, ...] = (torch.bfloat16, torch.float32),
) -> SumCastValue | None:
    if not (
        isinstance(cast, ast.Call)
        and not cast.keywords
        and len(cast.args) == 1
        and isinstance(cast.func, ast.Attribute)
        and cast.func.attr == "to"
        and isinstance(reduce := cast.func.value, ast.Call)
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
    source_type = _type(source)
    sum_method, cast_method = _type(reduce.func), _type(cast.func)
    sum_type, cast_type, dtype_type = _type(reduce), _type(cast), _type(cast.args[0])
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
    dtype_expression = cast.args[0]
    # Pass a dtype owner, rather than reading its attribute ahead of the first
    # original sum. The runtime gate checks exact Tensor identity before reading
    # metadata; a subclass or unsupported case executes the original AST.
    if (
        isinstance(dtype_expression, ast.Attribute)
        and dtype_expression.attr == "dtype"
        and isinstance(dtype_expression.value, ast.Name)
        and isinstance(_type(dtype_expression.value), TensorType)
    ):
        dtype_argument = dtype_expression.value.id
        dtype_is_tensor = True
    elif (
        isinstance(dtype_expression, ast.Attribute)
        and isinstance(dtype_expression.value, ast.Name)
        and isinstance(module_type := _type(dtype_expression.value), LiteralType)
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


def _sum_cast(statement: ast.AST) -> SumCast | None:
    if not (
        isinstance(statement, ast.Assign)
        and isinstance(statement, ExtendedAST)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and (value := _sum_cast_value(statement.value)) is not None
    ):
        return None
    return SumCast(
        statement._location,
        ast.dump(statement),
        statement.targets[0].id,
        value.source,
        value.dtype_argument,
        value.dtype_is_tensor,
        value.dtype,
    )


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
    info = _type(node)
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
        info = _type(node)
        if not isinstance(info, LiteralType):
            return False
        if _keyed_origin(info.origin, info.value, host):
            return True
        bound = bindings.get(node.id)
        return bound is not None and _keyed_expression(bound, host, bindings, depth + 1)
    if isinstance(node, ast.Attribute):
        base = node.value
        base_info = _type(base)
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
            and isinstance(_type(func.value), TensorType)
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


def _folded_body(
    body: Sequence[ast.AST], conditions: dict[SourceLocation, tuple[str, bool]]
) -> list[ast.AST]:
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
            result.extend(_folded_body(selected, conditions))
        else:
            result.append(statement)
    return result


@dataclass(frozen=True)
class HostSumProof:
    pairs: tuple[tuple[SumCast, SumCast], ...]
    conditions: dict[SourceLocation, tuple[str, bool]]


def prove_host_sum_pairs(host: HostFunction) -> HostSumProof:
    """Prove only straight-line host pairs, including pure constexpr branches.

    Neither device statements nor dynamic control-flow bodies are searched.
    Both inputs and dtype owners must already be bound, and both outputs must
    be fresh local names. This excludes early NameError, target destruction,
    pair dependencies and reordering across an effect or exception handler.
    """
    nodes = list(ast.walk(ast.Module(body=host.body, type_ignores=[])))
    # The generated helper is a module import. Any lexical binding of its
    # name, even an assignment after the pair, can shadow that import. Keep
    # such programs on their original host path before exposing choices/seeds.
    if host.name == _PAIRED_SUM_HELPER or any(
        (isinstance(node, ast.Name) and node.id == _PAIRED_SUM_HELPER)
        or (isinstance(node, ast.arg) and node.arg == _PAIRED_SUM_HELPER)
        or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == _PAIRED_SUM_HELPER
        )
        or (
            isinstance(node, ast.alias)
            and (node.asname or node.name.split(".", 1)[0]) == _PAIRED_SUM_HELPER
        )
        for node in (*nodes, *ast.walk(host.args))
    ):
        return HostSumProof((), {})
    if any(
        isinstance(node, (ast.Delete, ast.Try, ast.Global, ast.Nonlocal))
        for node in nodes
    ):
        return HostSumProof((), {})
    conditions = {
        node._location: (ast.dump(node.test), condition)
        for node in nodes
        if isinstance(node, ast.If)
        and isinstance(node, ExtendedAST)
        and (condition := _constant_condition(node.test, host)) is not None
    }
    body = _folded_body(host.body, conditions)
    defined = {argument.arg for argument in host.args.args}
    written = set(defined)
    pairs: list[tuple[SumCast, SumCast]] = []
    index = 0
    while index < len(body):
        statement = body[index]
        first = _sum_cast(statement)
        second = _sum_cast(body[index + 1]) if index + 1 < len(body) else None
        if first is not None and second is not None:
            arguments = {first.source, second.source}
            for site in (first, second):
                if site.dtype_is_tensor:
                    arguments.add(site.dtype_argument)
            targets = {first.target, second.target}
            if (
                len(targets) == 2
                and not targets & (written | arguments)
                and arguments <= defined
                and first.dtype == second.dtype
                and first.dtype_argument == second.dtype_argument
                and first.dtype_is_tensor == second.dtype_is_tensor
            ):
                pairs.append((first, second))
                defined.update(targets)
                written.update(targets)
                index += 2
                continue
        if isinstance(statement, ast.Assign):
            defined.update(
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name)
            )
        written.update(
            node.id
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        )
        if isinstance(statement, (ast.Return, ast.Raise)):
            break
        index += 1
    return HostSumProof(tuple(pairs), conditions)


def lower_host_sum_pairs(
    host: HostFunction, body: list[ast.AST], layout: str
) -> list[ast.AST]:
    """Confirm the typed sites survived generation before changing the host."""
    if layout == "off":
        return body
    assert layout in PAIRED_SUM_CHOICES
    proof = prove_host_sum_pairs(host)
    if not proof.pairs:
        return body
    folded = _folded_body(body, proof.conditions)
    pairs = {
        (left.location, right.location): (left, right) for left, right in proof.pairs
    }
    names = {
        node.id
        for statement in folded
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }
    names.update(argument.arg for argument in host.args.args)
    result: list[ast.AST] = []
    changed = False
    index = 0
    while index < len(folded):
        left = folded[index]
        right = folded[index + 1] if index + 1 < len(folded) else None
        pair = (
            pairs.get((left._location, right._location))
            if isinstance(left, ExtendedAST) and isinstance(right, ExtendedAST)
            else None
        )
        # Codegen can rebuild a library Name (e.g. torch) without TypeInfo.
        # Require the original typed site and complete final statement AST,
        # rather than inferring provenance again from such an untyped leaf.
        if (
            pair is not None
            and isinstance(left, ast.Assign)
            and isinstance(right, ast.Assign)
            and ast.dump(left) == pair[0].statement
            and ast.dump(right) == pair[1].statement
        ):
            first, second = pair
            temporary = "_helion_paired_sum"
            while temporary in names:
                temporary += "_"
            names.add(temporary)
            result.extend(
                (
                    statement_from_string(
                        f"{temporary} = {_PAIRED_SUM_HELPER}("
                        f"{first.source}, {second.source}, {first.dtype_argument}, "
                        f"dtype_is_tensor={first.dtype_is_tensor!r}, layout={layout!r}, "
                        "_launcher=_launcher)"
                    ),
                    ast.If(
                        test=ast.Compare(
                            ast.Name(temporary, ast.Load()),
                            [ast.Is()],
                            [ast.Constant(None)],
                        ),
                        body=[left, right],
                        orelse=[
                            statement_from_string(
                                f"{first.target}, {second.target} = {temporary}"
                            )
                        ],
                    ),
                )
            )
            changed = True
            index += 2
        else:
            result.append(folded[index])
            index += 1
    return result if changed else body
