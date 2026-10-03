"""Typed proof and final-host rewrite for independent sum/cast pairs."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..ast_extension import ExtendedAST
from ..ast_extension import statement_from_string
from ..type_info import LiteralType
from ..type_info import TensorAttributeType
from ..type_info import TensorType
from ..variable_origin import ArgumentOrigin

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
    if isinstance(node, ast.Constant) and type(node.value) is bool:
        return node.value
    if isinstance(node, ast.Name) and isinstance(info := _type(node), LiteralType):
        origin = info.origin
        if isinstance(origin, ArgumentOrigin):
            value = host.constexpr_args.get(origin.name)
            if type(value) is bool and value is info.value:
                return value
    return None


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
