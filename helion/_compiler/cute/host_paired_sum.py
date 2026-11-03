"""Typed proof and final-host rewrite for independent sum/cast pairs.

The first of the three host-sum tails (:mod:`host_sums`): two adjacent
``<name> = <2-D FP32>.sum(0).to(<dtype>)`` statements with the same dtype
owner become one ``_cute_try_paired_sum_cast`` call that computes both
column sums in one kernel (:mod:`helion.runtime.cute.paired_sum`).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..ast_extension import ExtendedAST
from ..ast_extension import statement_from_string
from .host_sum_site import PAIRED_SUM_CHOICES
from .host_sum_site import HostSumSite
from .host_sum_site import assigned_names
from .host_sum_site import bound_names
from .host_sum_site import fold_constexpr_branches
from .host_sum_site import fresh_temporary
from .host_sum_site import prove_host_body
from .host_sum_site import sum_cast_value

if TYPE_CHECKING:
    import torch

    from ..host_function import HostFunction
    from ..source_location import SourceLocation

_PAIRED_SUM_HELPER = "_cute_try_paired_sum_cast"


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


def _sum_cast(statement: ast.AST) -> SumCast | None:
    if not (
        isinstance(statement, ast.Assign)
        and isinstance(statement, ExtendedAST)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and (value := sum_cast_value(statement.value)) is not None
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
    facts = prove_host_body(host, (_PAIRED_SUM_HELPER,))
    if facts is None:
        return HostSumProof((), {})
    body = facts.folded
    defined = facts.defined
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
        defined.update(assigned_names(statement))
        written.update(
            node.id
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        )
        if isinstance(statement, (ast.Return, ast.Raise)):
            break
        index += 1
    return HostSumProof(tuple(pairs), facts.conditions)


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
    folded = fold_constexpr_branches(body, proof.conditions)
    pairs = {
        (left.location, right.location): (left, right) for left, right in proof.pairs
    }
    names = bound_names(host, folded)
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
            temporary = fresh_temporary("_helion_paired_sum", names)
            result.extend(
                HostSumSite(
                    temporary,
                    f"{_PAIRED_SUM_HELPER}("
                    f"{first.source}, {second.source}, {first.dtype_argument}, "
                    f"dtype_is_tensor={first.dtype_is_tensor!r}, layout={layout!r}, "
                    "_launcher=_launcher)",
                    (left, right),
                    orelse=(
                        statement_from_string(
                            f"{first.target}, {second.target} = {temporary}"
                        ),
                    ),
                ).emit()
            )
            changed = True
            index += 2
        else:
            result.append(folded[index])
            index += 1
    return result if changed else body
