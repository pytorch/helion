"""Typed proof for a host ``sum``/``mean`` of a 1-D FP32 tensor after the launch.

``loss = torch.empty([n]); <device loops>; total = torch.sum(loss)`` or
``return losses.mean()`` runs ATen's generic ``reduce_kernel`` after the
Helion kernel. On the few-microsecond shapes of the catalogue that launch is
about a microsecond more than a minimal kernel of the same geometry, so the
site is rewritten to call :func:`helion.runtime.cute.full_sum.try_full_sum`,
which computes ATen's exact addition tree in one CTA
(:mod:`helion._compiler.cute.full_sum_runtime`) and returns ``None`` for
every case it does not cover; the original statement then executes.

Admission: a top-level host statement after the last device loop (through
typed constexpr branches) that is ``name = torch.sum(x)`` / ``torch.mean(x)``
/ ``x.sum()`` / ``x.mean()`` with no other arguments, or a ``return`` whose
value is such a call or a tuple of bound names and one such call; ``x`` a
1-D contiguous FP32 tensor typed as such and the result a 0-d FP32 tensor.
Shapes, alignment, gradients, dispatch modes and the ATen build are runtime
guards in the helper. The rewrite is enabled by the ``cute_host_paired_sum``
knob (any layout), together with the paired and single sum tails.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch

from ..ast_extension import ExtendedAST
from ..ast_extension import LoopType
from ..ast_extension import statement_from_string
from ..type_info import LiteralType
from ..type_info import TensorAttributeType
from ..type_info import TensorType
from .host_paired_sum import PAIRED_SUM_CHOICES
from .host_paired_sum import _constant_condition
from .host_paired_sum import _folded_body

if TYPE_CHECKING:
    from ..host_function import HostFunction
    from ..source_location import SourceLocation

FULL_SUM_HELPER = "_cute_try_full_sum"
RETURN_RESULT = "_helion_full_sum"


@dataclass(frozen=True)
class FullSumSite:
    location: SourceLocation
    statement: str
    source: str
    mean: bool
    # ``None`` for an assignment; for a return, the names around the call in
    # the returned tuple (``()`` for a bare ``return <call>``) and the call's
    # position in that tuple.
    returned: tuple[tuple[str, ...], int] | None


@dataclass(frozen=True)
class HostFullSumProof:
    sites: tuple[FullSumSite, ...]
    conditions: dict[SourceLocation, tuple[str, bool]]


def _type(node: ast.AST) -> object:
    return node._type_info if isinstance(node, ExtendedAST) else None


def _reduction_call(value: ast.AST) -> tuple[str, bool] | None:
    """``(source, is_mean)`` for ``torch.sum(x)``, ``x.sum()`` and the means."""
    if not (
        isinstance(value, ast.Call)
        and not value.keywords
        and isinstance(value.func, ast.Attribute)
        and value.func.attr in ("sum", "mean")
    ):
        return None
    owner = value.func.value
    owner_type = _type(owner)
    if isinstance(owner_type, LiteralType) and owner_type.value is torch:
        if len(value.args) != 1 or not isinstance(value.args[0], ast.Name):
            return None
        source = value.args[0]
    elif (
        isinstance(owner, ast.Name)
        and isinstance(method := _type(value.func), TensorAttributeType)
        and method.attr() == value.func.attr
        and not value.args
    ):
        source = owner
    else:
        return None
    source_type = _type(source)
    result_type = _type(value)
    if not (
        isinstance(source_type, TensorType)
        and isinstance(result_type, TensorType)
        and source_type.fake_value.ndim == 1
        and source_type.fake_value.dtype == torch.float32
        and source_type.fake_value.stride() == (1,)
        and result_type.fake_value.ndim == 0
        and result_type.fake_value.dtype == torch.float32
    ):
        return None
    return source.id, value.func.attr == "mean"


def _shadows_helper(host: HostFunction, nodes: list[ast.AST]) -> bool:
    names = (FULL_SUM_HELPER, RETURN_RESULT)
    return host.name in names or any(
        (isinstance(node, ast.Name) and node.id in names)
        or (isinstance(node, ast.arg) and node.arg in names)
        or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name in names
        )
        or (
            isinstance(node, ast.alias)
            and (node.asname or node.name.split(".", 1)[0]) in names
        )
        for node in (*nodes, *ast.walk(host.args))
    )


def prove_host_full_sum(host: HostFunction) -> HostFullSumProof:
    """Admit sum/mean sites among the top-level statements after the device loops.

    Only typed constexpr branches are folded; dynamic bodies are not searched.
    Every site keeps its original statement as the fallback, so sources need
    no allocation or coverage proof: the helper reads the tensor the original
    call would have read, at the same point of the program.
    """
    nodes = list(ast.walk(ast.Module(body=host.body, type_ignores=[])))
    if _shadows_helper(host, nodes) or any(
        isinstance(node, (ast.Delete, ast.Try, ast.Global, ast.Nonlocal))
        for node in nodes
    ):
        return HostFullSumProof((), {})
    conditions = {
        node._location: (ast.dump(node.test), condition)
        for node in nodes
        if isinstance(node, ast.If)
        and isinstance(node, ExtendedAST)
        and (condition := _constant_condition(node.test, host)) is not None
    }
    folded = _folded_body(host.body, conditions)
    device_loops = [
        index
        for index, statement in enumerate(folded)
        if isinstance(statement, ExtendedAST) and statement._loop_type == LoopType.GRID
    ]
    if not device_loops:
        return HostFullSumProof((), conditions)
    defined = {argument.arg for argument in host.args.args}
    for statement in folded[: device_loops[-1] + 1]:
        if isinstance(statement, ast.Assign):
            defined.update(
                target.id
                for target in statement.targets
                if isinstance(target, ast.Name)
            )
    sites: list[FullSumSite] = []
    for statement in folded[device_loops[-1] + 1 :]:
        if not isinstance(statement, ExtendedAST):
            continue
        if isinstance(statement, ast.Return) and statement.value is not None:
            expression = statement.value
            returned: tuple[tuple[str, ...], int] | None = ((), 0)
            if isinstance(expression, ast.Tuple):
                calls = [
                    index
                    for index, element in enumerate(expression.elts)
                    if _reduction_call(element) is not None
                ]
                others = [
                    element
                    for index, element in enumerate(expression.elts)
                    if index not in calls
                ]
                if len(calls) != 1 or not all(
                    isinstance(other, ast.Name) and other.id in defined
                    for other in others
                ):
                    continue
                names = tuple(
                    element.id
                    for element in expression.elts
                    if isinstance(element, ast.Name)
                )
                returned = (names, calls[0])
                expression = expression.elts[calls[0]]
            found = _reduction_call(expression)
            if found is None or found[0] not in defined:
                continue
            sites.append(
                FullSumSite(
                    statement._location,
                    ast.dump(statement),
                    found[0],
                    found[1],
                    returned,
                )
            )
        elif (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(target := statement.targets[0], ast.Name)
        ):
            found = _reduction_call(statement.value)
            if found is not None and found[0] in defined:
                sites.append(
                    FullSumSite(
                        statement._location,
                        ast.dump(statement),
                        found[0],
                        found[1],
                        None,
                    )
                )
            defined.add(target.id)
    return HostFullSumProof(tuple(sites), conditions)


def lower_host_full_sum(
    host: HostFunction, body: list[ast.AST], layout: str
) -> list[ast.AST]:
    """Rewrite each proven site; its original statement stays as the fallback."""
    if layout == "off":
        return body
    assert layout in PAIRED_SUM_CHOICES
    proof = prove_host_full_sum(host)
    if not proof.sites:
        return body
    sites = {site.location: site for site in proof.sites}
    folded = _folded_body(body, proof.conditions)
    names = {
        node.id
        for statement in folded
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }
    names.update(argument.arg for argument in host.args.args)
    result: list[ast.AST] = []
    for statement in folded:
        site = (
            sites.get(statement._location)
            if isinstance(statement, ExtendedAST)
            else None
        )
        if site is None or ast.dump(statement) != site.statement:
            result.append(statement)
            continue
        if site.returned is None:
            assert isinstance(statement, ast.Assign)
            target = statement.targets[0]
            assert isinstance(target, ast.Name)
            temporary = target.id
            tail: list[ast.stmt] = []
        else:
            temporary = RETURN_RESULT
            while temporary in names:
                temporary += "_"
            names.add(temporary)
            returned_names, position = site.returned
            value: ast.expr = ast.Name(temporary, ast.Load())
            if isinstance(statement, ast.Return) and isinstance(
                statement.value, ast.Tuple
            ):
                elements: list[ast.expr] = [
                    ast.Name(name, ast.Load()) for name in returned_names
                ]
                elements.insert(position, value)
                value = ast.Tuple(elements, ast.Load())
            tail = [ast.Return(value)]
        result.extend(
            (
                statement_from_string(
                    f"{temporary} = {FULL_SUM_HELPER}({site.source}, "
                    f"mean={site.mean!r}, _launcher=_launcher)"
                ),
                ast.If(
                    test=ast.Compare(
                        ast.Name(temporary, ast.Load()),
                        [ast.Is()],
                        [ast.Constant(None)],
                    ),
                    body=[cast("ast.stmt", statement)],
                    orelse=[],
                ),
                *tail,
            )
        )
    return result
