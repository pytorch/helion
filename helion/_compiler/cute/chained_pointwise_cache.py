"""Typed immutable reads reused across a TCgen05 pointwise staging loop."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from torch.fx import Node

    from .chained_matmul import _Expression


def _key(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


@dataclass
class PointwiseReadCache:
    """Private activation state; the default never changes source emission."""

    enabled: bool
    activated: bool = False

    def vector_reads(
        self,
        expression: _Expression,
        names: tuple[str, str],
        trips: int,
    ) -> set[tuple[Node, tuple[str, ...]]]:
        """Retain immutable FP32 vectors across rows, under the original guard.

        The caller additionally requires the existing affine vector proof and
        zero physical row stride. Inspect the complete original scalar RHS too:
        equal addresses alone do not prove that a row-dependent mask is equal.
        No expression is expanded, no arithmetic is hoisted, and no allocation
        or ownership rule changes. Only the already allocated eight-value
        vector registers survive between row iterations.
        """
        if not self.enabled or trips < 2:
            return set()
        selected: set[tuple[Node, tuple[str, ...]]] = set()
        for node, coordinates, _, value in expression.loaded_inputs:
            if node.meta["val"].dtype != torch.float32 or len(coordinates) != 1:
                continue
            tree = ast.parse(expression.definitions[value], mode="eval").body
            dependencies = {
                item.id for item in ast.walk(tree) if isinstance(item, ast.Name)
            }
            if (
                names[1] in dependencies
                and names[0] not in dependencies
                and not dependencies & expression.definitions.keys()
            ):
                selected.add((node, coordinates))
        return selected if len(selected) <= 4 else set()

    def prepare(
        self,
        expression: _Expression,
        names: tuple[str, str],
        element: str,
        tag: str,
        columns: int,
        trips: int,
        column_base: str | None = None,
    ) -> list[str]:
        if not self.enabled or trips < 2:
            return []
        # This emitter only stages a pure operand into its fresh shared A/B
        # tile. No global store, scan update, collective, or carried state occurs
        # inside the loop. Its scan/cache publication barrier precedes the stage.
        # The loaded_inputs catalog excludes already-vectorized leaf registers.
        reads = [
            (
                expression.definitions[value],
                cast("Node", node.args[0]).meta["val"].dtype,
            )
            for node, _, _, value in expression.loaded_inputs
        ]
        reads.extend(
            (value, node.meta["val"].dtype)
            for (node, _), value in expression.memo.items()
            if node in expression.plan.scans
        )
        chosen: dict[str, tuple[str, torch.dtype]] = {}
        for text, dtype in reads:
            if dtype not in (torch.bfloat16, torch.float16, torch.float32):
                continue
            tree = ast.parse(text, mode="eval").body
            dependencies = {
                node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
            }
            # Check the complete RHS, including all masks and fallback accesses.
            # Reject temporary-dependent indexing instead of moving/inlining any
            # arithmetic or changing the scope of a previously computed value.
            if (
                names[1] not in dependencies
                or names[0] in dependencies
                or dependencies & expression.definitions.keys()
            ):
                continue
            chosen.setdefault(_key(tree), (text, dtype))
        # Bound additional live values independent of any kernel or shape name.
        if not chosen or len(chosen) > 4:
            return []
        self.activated = True
        lane = f"{tag}_cache_element"
        col = f"{tag}_cache_col"
        declarations, assignments = [], []
        replacements: dict[str, ast.expr] = {}

        class Column(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.AST:
                return (
                    ast.copy_location(ast.Name(id=col, ctx=node.ctx), node)
                    if node.id == names[1]
                    else node
                )

        for index, (key, (text, dtype)) in enumerate(chosen.items()):
            cache = f"{tag}_read_cache_{index}"
            dtype_name = CompileEnvironment.current().backend.dtype_str(dtype)
            declarations.append(f"{cache} = cute.make_rmem_tensor((8,), {dtype_name})")
            value = Column().visit(ast.parse(text, mode="eval").body)
            assignments.append(f"    {cache}[{lane}] = {ast.unparse(value)}")
            replacements[key] = ast.parse(f"{cache}[{element}]", mode="eval").body

        class ReplaceReads(ast.NodeTransformer):
            def visit(self, node: ast.AST) -> ast.AST:
                if (
                    isinstance(node, ast.expr)
                    and (replacement := replacements.get(_key(node))) is not None
                ):
                    return replacement
                return super().visit(node)

        expression.lines = [
            ast.unparse(ReplaceReads().visit(ast.parse(line)))
            for line in expression.lines
        ]
        return [
            *declarations,
            f"for {lane} in cutlass.range_constexpr(8):",
            f"    {col} = "
            + (f"{column_base} + " if column_base is not None else "")
            + f"chain_thread % {columns} * 8 + {lane}",
            *assignments,
        ]

    def validate(self) -> None:
        if self.enabled and not self.activated:
            raise exc.BackendUnsupported(
                "cute", "pointwise read cache requires row-invariant typed reads"
            )
