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
A sum may also be divided by a host scalar (``torch.sum(loss) / BT``, the
``batchmean`` of a loss; ``/ (BT * V)`` for a mean): the divisor is pure
arithmetic over names, constants and tensor metadata typed as a number, and
the helper applies ATen's own form of that division (``div_true_kernel_cuda``
with a CPU scalar multiplies by the FP32 reciprocal), so the result stays
bit-identical while the separate elementwise launch disappears. Shapes,
alignment, gradients, dispatch modes and the ATen build are runtime guards
in the helper. The rewrite is enabled by the ``cute_host_paired_sum`` knob
(any layout), together with the paired and single sum tails
(:mod:`host_sums`).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch

from ..ast_extension import ExtendedAST
from ..ast_extension import LoopType
from ..type_info import LiteralType
from ..type_info import SymFloatType
from ..type_info import SymIntType
from ..type_info import TensorAttributeType
from ..type_info import TensorType
from .host_sum_site import PAIRED_SUM_CHOICES
from .host_sum_site import HostSumSite
from .host_sum_site import assigned_names
from .host_sum_site import bound_names
from .host_sum_site import fold_constexpr_branches
from .host_sum_site import fresh_temporary
from .host_sum_site import node_type
from .host_sum_site import prove_host_body

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
    # Source of the host scalar a sum is divided by, ``None`` for a bare call.
    divisor: str | None = None


@dataclass(frozen=True)
class HostFullSumProof:
    sites: tuple[FullSumSite, ...]
    conditions: dict[SourceLocation, tuple[str, bool]]


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
    owner_type = node_type(owner)
    if isinstance(owner_type, LiteralType) and owner_type.value is torch:
        if len(value.args) != 1 or not isinstance(value.args[0], ast.Name):
            return None
        source = value.args[0]
    elif (
        isinstance(owner, ast.Name)
        and isinstance(method := node_type(value.func), TensorAttributeType)
        and method.attr() == value.func.attr
        and not value.args
    ):
        source = owner
    else:
        return None
    source_type = node_type(source)
    result_type = node_type(value)
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


_METADATA_CALLS = frozenset({"size", "numel", "dim", "stride", "ndimension"})


def _host_scalar(node: ast.AST) -> bool:
    """Whether ``node`` is pure arithmetic over names, constants and tensor
    metadata whose value is a (non-bool) host number: safe to evaluate twice
    (once for the helper, once for the fallback) and a scalar ATen would wrap
    as a CPU number."""
    info = node_type(node)
    if isinstance(info, LiteralType):
        if type(info.value) not in (int, float):
            return False
    elif not isinstance(info, (SymIntType, SymFloatType)):
        return False
    for child in ast.walk(node):
        if isinstance(
            child,
            (
                ast.Name,
                ast.Constant,
                ast.BinOp,
                ast.UnaryOp,
                ast.Load,
                ast.operator,
                ast.unaryop,
                ast.Attribute,
                ast.Subscript,
            ),
        ):
            continue
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr in _METADATA_CALLS
            and isinstance(node_type(child.func.value), TensorType)
            and not child.keywords
            and all(isinstance(arg, (ast.Constant, ast.Name)) for arg in child.args)
        ):
            continue
        return False
    return True


def _scaled_reduction_call(value: ast.AST) -> tuple[str, bool, str | None] | None:
    """``(source, is_mean, divisor_source)`` for a bare reduction call or a
    ``torch.sum(x) / scalar`` quotient typed as a 0-d FP32 tensor."""
    found = _reduction_call(value)
    if found is not None:
        return (*found, None)
    if not (isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div)):
        return None
    found = _reduction_call(value.left)
    result_type = node_type(value)
    if (
        found is None
        or found[1]
        or not _host_scalar(value.right)
        or not (
            isinstance(result_type, TensorType)
            and result_type.fake_value.ndim == 0
            and result_type.fake_value.dtype == torch.float32
        )
    ):
        return None
    return found[0], False, ast.unparse(value.right)


def prove_host_full_sum(host: HostFunction) -> HostFullSumProof:
    """Admit sum/mean sites among the top-level statements after the device loops.

    Only typed constexpr branches are folded; dynamic bodies are not searched.
    Every site keeps its original statement as the fallback, so sources need
    no allocation or coverage proof: the helper reads the tensor the original
    call would have read, at the same point of the program.
    """
    facts = prove_host_body(host, (FULL_SUM_HELPER, RETURN_RESULT))
    if facts is None:
        return HostFullSumProof((), {})
    folded = facts.folded
    device_loops = [
        index
        for index, statement in enumerate(folded)
        if isinstance(statement, ExtendedAST) and statement._loop_type == LoopType.GRID
    ]
    if not device_loops:
        return HostFullSumProof((), facts.conditions)
    defined = facts.defined
    for statement in folded[: device_loops[-1] + 1]:
        defined.update(assigned_names(statement))
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
                    if _scaled_reduction_call(element) is not None
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
            found = _scaled_reduction_call(expression)
            if found is None or found[0] not in defined:
                continue
            sites.append(
                FullSumSite(
                    statement._location,
                    ast.dump(statement),
                    found[0],
                    found[1],
                    returned,
                    found[2],
                )
            )
        elif (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(target := statement.targets[0], ast.Name)
        ):
            found = _scaled_reduction_call(statement.value)
            if found is not None and found[0] in defined:
                sites.append(
                    FullSumSite(
                        statement._location,
                        ast.dump(statement),
                        found[0],
                        found[1],
                        None,
                        found[2],
                    )
                )
            defined.add(target.id)
    return HostFullSumProof(tuple(sites), facts.conditions)


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
    folded = fold_constexpr_branches(body, proof.conditions)
    names = bound_names(host, folded)
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
            tail: tuple[ast.stmt, ...] = ()
        else:
            temporary = fresh_temporary(RETURN_RESULT, names)
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
            tail = (ast.Return(value),)
        divisor = f"divisor={site.divisor}, " if site.divisor is not None else ""
        result.extend(
            HostSumSite(
                temporary,
                f"{FULL_SUM_HELPER}({site.source}, "
                f"mean={site.mean!r}, {divisor}_launcher=_launcher)",
                (cast("ast.stmt", statement),),
                tail=tail,
            ).emit()
        )
    return result
