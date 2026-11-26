"""Narrow the compiler's PID dispatch to roots reachable in one barrier phase."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

from ..ast_read_writes import binding_names
from ..program_id import _ROOT_DISPATCH_TEST

if TYPE_CHECKING:
    from ..ast_extension import ExtendedAST
    from ..program_id import ForEachProgramID


def _scalar_selector(node: ast.AST, *, allow_cast: bool) -> bool:
    if isinstance(node, ast.Call):
        # Only a compiler-generated integer conversion in the leading dispatch
        # is pure. Decline a call after any setup, which could mutate its callee
        # via an alias, and still reject subscript reads inside the conversion.
        fn = node.func
        return (
            allow_cast
            and isinstance(fn, ast.Attribute)
            and fn.attr in ("Int32", "Int64")
            and isinstance(fn.value, ast.Name)
            and fn.value.id == "cutlass"
            and len(node.args) == 1
            and not node.keywords
            and _scalar_selector(node.args[0], allow_cast=True)
        )
    return isinstance(
        node,
        (
            ast.Compare,
            ast.cmpop,
            ast.Name,
            ast.Load,
            ast.Constant,
            ast.BinOp,
            ast.operator,
            ast.UnaryOp,
            ast.unaryop,
        ),
    ) and all(
        _scalar_selector(child, allow_cast=allow_cast)
        for child in ast.iter_child_nodes(node)
    )


def specialize_phase_cases(
    owner: ForEachProgramID, body: list[ast.AST], first: int, stop: int
) -> list[ast.AST]:
    """An outer PID guard has selected this contiguous range of root cases.

    Specialize only the dispatch nodes we emitted, retaining the actual
    case bodies, absolute PID offsets, and unrelated surrounding setup.
    A backend which replaced a dispatch node keeps its original body.
    """

    def restrict(
        statements: list[ast.AST],
        begin: int,
        end: int,
        rebound: set[str],
        before_setup: bool,
    ) -> list[ast.AST]:
        if first <= begin and end <= stop or end - begin <= 1:
            return statements
        middle = (begin + end) // 2
        branch = owner.case_branches[middle - 1]
        for index, statement in enumerate(statements):
            if statement is not branch:
                continue
            # The phase bounds refer to the PID and counts at the dispatch
            # initializer. A sibling statement or enclosing arm can rebind
            # either before this selector executes.
            prefix = statements[:index]
            changed = rebound.union(*(binding_names(stmt) for stmt in prefix))
            inputs = {
                node.id for node in ast.walk(branch.test) if isinstance(node, ast.Name)
            }
            result = list(prefix)
            marker = getattr(branch, _ROOT_DISPATCH_TEST, None)
            # A scalar tensor read (or an opaque call) may see a write via
            # another name. Name-binding analysis only proves scalar expressions.
            pristine = before_setup and not prefix
            scalar = _scalar_selector(branch.test, allow_cast=pristine)
            if (
                marker != ast.dump(branch.test)
                or not scalar
                or not inputs.isdisjoint(changed)
            ):
                # A changed predicate, a rebound input, or an unproved load
                # can now reach either arm. Do not propagate the old proof.
                result.append(cast("ast.If", cast("ExtendedAST", branch).copy()))
            elif stop <= middle:
                result.extend(
                    restrict(list(branch.body), begin, middle, changed, pristine)
                )
            elif first >= middle:
                result.extend(
                    restrict(list(branch.orelse), middle, end, changed, pristine)
                )
            else:
                # A compiler-emitted If carries a required source location.
                # Keep the original arm identities for recursive restriction.
                narrowed = cast("ast.If", cast("ExtendedAST", branch).copy())
                # Keep the author's uniform-PID proof only for its unchanged
                # predicate. Pruning either arm never changes which CTA selects it.
                setattr(narrowed, _ROOT_DISPATCH_TEST, marker)
                narrowed.body = cast(
                    "list[ast.stmt]",
                    restrict(list(branch.body), begin, middle, changed, pristine),
                )
                narrowed.orelse = cast(
                    "list[ast.stmt]",
                    restrict(list(branch.orelse), middle, end, changed, pristine),
                )
                result.append(narrowed)
            return [*result, *statements[index + 1 :]]
        return statements

    return restrict(body, 0, len(owner.case_bodies), set(), True)
