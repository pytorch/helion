"""Typed proof for one terminal host FP32 sum followed by a final cast."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..ast_extension import ExtendedAST
from ..ast_extension import statement_from_string
from .host_paired_sum import PAIRED_SUM_CHOICES
from .host_paired_sum import SumCastValue
from .host_paired_sum import _constant_condition
from .host_paired_sum import _folded_body
from .host_paired_sum import _sum_cast_value

if TYPE_CHECKING:
    from ..host_function import HostFunction
    from ..source_location import SourceLocation

_SINGLE_SUM_HELPER = "_cute_try_single_sum_cast"


@dataclass(frozen=True)
class ReturnedSum:
    location: SourceLocation
    statement: str
    value: SumCastValue
    # None distinguishes a scalar Tensor return from a one-element tuple.
    prefix: tuple[str, ...] | None


@dataclass(frozen=True)
class HostSingleSumProof:
    site: ReturnedSum | None
    conditions: dict[SourceLocation, tuple[str, bool]]


def prove_host_single_sum(host: HostFunction) -> HostSingleSumProof:
    """Admit a direct return or the last expression of a tuple of bound names.

    No calls, attributes, subscripts or potentially unbound names may precede
    the sum in its return tuple. The original return remains the fallback for
    every unsupported runtime binding, including autograd and custom dispatch.
    Only typed constexpr branches are folded; dynamic bodies are not searched.
    """
    nodes = list(ast.walk(ast.Module(body=host.body, type_ignores=[])))
    # Any lexical binding can shadow the injected module import, including a
    # local store after the return. Apply this proof before exposing seeds.
    if host.name == _SINGLE_SUM_HELPER or any(
        (isinstance(node, ast.Name) and node.id == _SINGLE_SUM_HELPER)
        or (isinstance(node, ast.arg) and node.arg == _SINGLE_SUM_HELPER)
        or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == _SINGLE_SUM_HELPER
        )
        or (
            isinstance(node, ast.alias)
            and (node.asname or node.name.split(".", 1)[0]) == _SINGLE_SUM_HELPER
        )
        for node in (*nodes, *ast.walk(host.args))
    ):
        return HostSingleSumProof(None, {})
    if any(
        isinstance(node, (ast.Delete, ast.Try, ast.Global, ast.Nonlocal))
        for node in nodes
    ):
        return HostSingleSumProof(None, {})
    conditions = {
        node._location: (ast.dump(node.test), condition)
        for node in nodes
        if isinstance(node, ast.If)
        and isinstance(node, ExtendedAST)
        and (condition := _constant_condition(node.test, host)) is not None
    }
    defined = {argument.arg for argument in host.args.args}
    for statement in _folded_body(host.body, conditions):
        if isinstance(statement, ast.Return):
            if not isinstance(statement, ExtendedAST) or statement.value is None:
                break
            expression = statement.value
            prefix = None
            if isinstance(expression, ast.Tuple):
                if not expression.elts:
                    break
                prefix = tuple(
                    value.id
                    for value in expression.elts[:-1]
                    if isinstance(value, ast.Name) and value.id in defined
                )
                if len(prefix) != len(expression.elts) - 1:
                    break
                expression = expression.elts[-1]
            value = _sum_cast_value(
                expression, (torch.float16, torch.bfloat16, torch.float32)
            )
            if value is None:
                break
            arguments = {value.source}
            if value.dtype_is_tensor:
                arguments.add(value.dtype_argument)
            if arguments <= defined:
                return HostSingleSumProof(
                    ReturnedSum(
                        statement._location, ast.dump(statement), value, prefix
                    ),
                    conditions,
                )
            break
        if isinstance(statement, ast.Raise):
            break
        if isinstance(statement, ast.Assign):
            defined.update(
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name)
            )
    return HostSingleSumProof(None, conditions)


def lower_host_single_sum(
    host: HostFunction, body: list[ast.AST], layout: str
) -> list[ast.AST]:
    """Rewrite only a complete final return equal to its typed original AST."""
    if layout == "off":
        return body
    assert layout in PAIRED_SUM_CHOICES
    proof = prove_host_single_sum(host)
    if (site := proof.site) is None:
        return body
    folded = _folded_body(body, proof.conditions)
    names = {
        node.id
        for statement in folded
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }
    names.update(argument.arg for argument in host.args.args)
    temporary = "_helion_single_sum"
    while temporary in names:
        temporary += "_"
    for index, statement in enumerate(folded):
        if not (
            isinstance(statement, ast.Return)
            and isinstance(statement, ExtendedAST)
            and statement._location == site.location
            and ast.dump(statement) == site.statement
        ):
            continue
        value = site.value
        result: ast.expr = ast.Name(temporary, ast.Load())
        if site.prefix is not None:
            result = ast.Tuple(
                [*(ast.Name(name, ast.Load()) for name in site.prefix), result],
                ast.Load(),
            )
        return [
            *folded[:index],
            statement_from_string(
                f"{temporary} = {_SINGLE_SUM_HELPER}("
                f"{value.source}, {value.dtype_argument}, "
                f"dtype_is_tensor={value.dtype_is_tensor!r}, layout={layout!r}, "
                "_launcher=_launcher)"
            ),
            ast.If(
                test=ast.Compare(
                    ast.Name(temporary, ast.Load()), [ast.Is()], [ast.Constant(None)]
                ),
                body=[statement],
                orelse=[ast.Return(result)],
            ),
            *folded[index + 1 :],
        ]
    return body
