"""Typed proof for one terminal host FP32 sum followed by a final cast.

The second of the three host-sum tails (:mod:`host_sums`): a ``return`` whose
value (or whose tuple's last element) is ``<2-D FP32>.sum(0).to(<dtype>)``
becomes one ``_cute_try_single_sum_cast`` call
(:mod:`helion.runtime.cute.single_sum`) with the original return as the
fallback.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..ast_extension import ExtendedAST
from .host_sum_site import PAIRED_SUM_CHOICES
from .host_sum_site import HostSumSite
from .host_sum_site import SumCastValue
from .host_sum_site import assigned_names
from .host_sum_site import bound_names
from .host_sum_site import fold_constexpr_branches
from .host_sum_site import fresh_temporary
from .host_sum_site import prove_host_body
from .host_sum_site import sum_cast_value

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
    facts = prove_host_body(host, (_SINGLE_SUM_HELPER,))
    if facts is None:
        return HostSingleSumProof(None, {})
    defined = facts.defined
    for statement in facts.folded:
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
            value = sum_cast_value(
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
                    facts.conditions,
                )
            break
        if isinstance(statement, ast.Raise):
            break
        defined.update(assigned_names(statement))
    return HostSingleSumProof(None, facts.conditions)


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
    folded = fold_constexpr_branches(body, proof.conditions)
    temporary = fresh_temporary("_helion_single_sum", bound_names(host, folded))
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
            *HostSumSite(
                temporary,
                f"{_SINGLE_SUM_HELPER}("
                f"{value.source}, {value.dtype_argument}, "
                f"dtype_is_tensor={value.dtype_is_tensor!r}, layout={layout!r}, "
                "_launcher=_launcher)",
                (statement,),
                orelse=(ast.Return(result),),
            ).emit(),
            *folded[index + 1 :],
        ]
    return body
