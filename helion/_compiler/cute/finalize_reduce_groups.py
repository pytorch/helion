"""Finalize the group count of shared-memory cross-warp reductions once the
launch block is known.

A reduction at the bottom of the lane index emits
``_cute_grouped_reduce_shared_two_stage(..., group_span=S, group_count=G)``
(or the register tile's ``_cute_grouped_reduce_shared_columns``; the V-loop
sink's ``_cute_grouped_reduce_shared_two_stage_fragment`` keeps the count of
the per-lane reduces it replaces) before codegen has seen every thread axis
(a sibling branch can still add a redundant axis), so
``ReductionStrategy._cute_runtime_lane_group_params`` keys the helper's
shared memory on the full runtime thread id and sizes ``G`` for the
1024-thread CTA budget.  When the launch shape is proven static the
over-provisioned count costs the kernel: the helper only takes its cheaper
serial form (one warp fold, one barrier, every thread sums the per-warp
partials) for a single group spanning the whole CTA, and the shared stage is
sized for ``G`` groups.  This pass re-renders each site with ``G`` set to the
number of groups the launched threads form, ``threads // S``, and reduces the
runtime lane expression to ``thread_idx()[0]`` when the block is
one-dimensional, so the later reduction passes see the canonical single-group
form.

The rewrite assumes the block the strategies planned is the block the
launcher emits.  Free ``hl.arange`` dims claim synthetic thread axes the
strategies' static shape does not know about (the launcher widens ``block=``
with them), so ``finalize_shared_reduce_groups_for_launch`` declines when any
claimed axis exceeds the static shape, and the launcher checks the recorded
assumption against the ``block=`` it finally emits.

The pass walks the body rather than recording emitted sites: the lane split
and the vector-loop sink clone statements through unparse-and-reparse, so
node identities do not survive to this point.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..source_location import SyntheticLocation
from .reduce_call import GROUPED_SHARED_REDUCES
from .reduce_call import ReduceCall
from .reduce_call import int_keyword
from .reduce_call import runtime_lane_expr

if TYPE_CHECKING:
    from collections.abc import Mapping

# Runtime lane dump -> the ``thread_idx()[0]`` lane of the same index type.
_X_LANE_BY_RUNTIME_DUMP = {
    ast.dump(ast.parse(runtime_lane_expr(index_type), mode="eval").body): (
        f"{index_type}(cute.arch.thread_idx()[0])"
    )
    for index_type in ("cutlass.Int32", "cutlass.Int64")
}


class _Finalize(ast.NodeTransformer):
    def __init__(self, thread_block_dims: tuple[int, int, int]) -> None:
        self.block_threads = (
            thread_block_dims[0] * thread_block_dims[1] * thread_block_dims[2]
        )
        self.one_dimensional = thread_block_dims[1] == thread_block_dims[2] == 1
        self.changed = False

    def _launched_groups(
        self, group_span: int | None, group_count: int | None
    ) -> int | None:
        """The group count the launched threads form when it is below the
        emitted ``group_count``; None when the site stays as emitted."""
        if (
            group_span is None
            or group_count is None
            or group_span <= 0
            or self.block_threads % group_span
        ):
            return None
        launched = self.block_threads // group_span
        return launched if launched < group_count else None

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not (
            isinstance(node.func, ast.Name) and node.func.id in GROUPED_SHARED_REDUCES
        ):
            return node
        reduce = ReduceCall.parse(node)
        if reduce is not None:
            launched = self._launched_groups(reduce.group_span, reduce.group_count)
            if launched is None:
                return node
            self.changed = True
            with SyntheticLocation():
                replacement = expr_from_string(
                    dataclasses.replace(reduce, group_count=launched).render()
                )
            return ast.copy_location(replacement, node)
        # The register tile's column reduce and the V-loop sink's fragment
        # combine take a list or fragment of values: not of the scalar family,
        # so their ``group_count`` keyword is resized in place (the fragment
        # with the per-lane reduces it replaces, so both pick the same form).
        launched = self._launched_groups(
            int_keyword(node, "group_span"), int_keyword(node, "group_count")
        )
        if launched is None:
            return node
        for keyword in node.keywords:
            if keyword.arg == "group_count":
                keyword.value = ast.Constant(value=launched)
        self.changed = True
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        if not (
            self.one_dimensional
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            return node
        x_lane = _X_LANE_BY_RUNTIME_DUMP.get(ast.dump(node.value))
        if x_lane is None:
            return node
        self.changed = True
        # Compiler-synthesized: carries no source comment of the current
        # location.
        with SyntheticLocation():
            replacement = statement_from_string(f"{node.targets[0].id} = {x_lane}")
        return ast.copy_location(replacement, node)


def _finalize(
    body: list[ast.stmt], thread_block_dims: tuple[int, int, int]
) -> tuple[list[ast.stmt], bool]:
    if any(dim <= 0 for dim in thread_block_dims):
        return body, False
    if not any(
        (isinstance(node, ast.Name) and node.id in GROUPED_SHARED_REDUCES)
        or (isinstance(node, ast.Attribute) and node.attr == "block_dim")
        for stmt in body
        for node in ast.walk(stmt)
    ):
        # Neither a grouped shared reduce nor the runtime lane expression
        # (which reads ``cute.arch.block_dim()``) occurs: nothing to rewrite.
        return body, False
    transformer = _Finalize(thread_block_dims)
    result = [transformer.visit(stmt) for stmt in body]
    if not transformer.changed:
        return body, False
    return [ast.fix_missing_locations(stmt) for stmt in result], True


def finalize_shared_reduce_groups_for_launch(
    body: list[ast.stmt],
    *,
    thread_block_dims: tuple[int, int, int],
    claimed_axis_sizes: Mapping[int, int],
) -> tuple[list[ast.stmt], tuple[int, int, int] | None]:
    """Rewrite over-provisioned ``group_count`` values (and the runtime lane
    expression of a one-dimensional block) for the launch shape, gated on it.

    ``thread_block_dims`` is the strategies' proven static shape;
    ``claimed_axis_sizes`` every thread extent the kernel has claimed so far
    (``GenerateAST.launch_thread_axis_sizes``), which includes the synthetic
    axes of free ``hl.arange`` dims the launcher widens ``block=`` with.  An
    axis claimed wider than the static shape means the launch is not the
    static shape: the body is returned unchanged (the conservative runtime
    lane form is correct for any block).  Otherwise returns the rewritten
    body and the shape it was sized for (None when nothing changed), which
    the launcher checks against the ``block=`` it emits.
    """
    for axis, size in claimed_axis_sizes.items():
        if axis < 0 or axis >= len(thread_block_dims) or size > thread_block_dims[axis]:
            return body, None
    result, changed = _finalize(body, thread_block_dims)
    return result, (thread_block_dims if changed else None)
