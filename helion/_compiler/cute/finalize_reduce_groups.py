"""Finalize the group count of shared-memory cross-warp reductions once the
launch block is known.

A reduction at the bottom of the lane index emits
``_cute_grouped_reduce_shared_two_stage(..., group_span=S, group_count=G)``
(or the register tile's ``_cute_grouped_reduce_shared_columns``) before
codegen has seen every thread axis (a sibling branch can still add a
redundant axis), so ``ReductionStrategy._cute_runtime_lane_group_params``
keys the helper's shared memory on the full runtime thread id and sizes ``G``
for the 1024-thread CTA budget.  When the launch shape is proven static the
over-provisioned count costs the kernel: the helper only takes its cheaper
serial form (one warp fold, one barrier, every thread sums the per-warp
partials) for a single group spanning the whole CTA, and the shared stage is
sized for ``G`` groups.  This pass rewrites ``G`` to the number of groups the
launched threads form, ``threads // S``, and reduces the runtime lane
expression to ``thread_idx()[0]`` when the block is one-dimensional, so the
later reduction passes see the canonical single-group form.

The rewrite assumes the block the strategies planned is the block the
launcher emits.  Free ``hl.arange`` dims claim synthetic thread axes the
strategies' static shape does not know about (the launcher widens ``block=``
with them), so ``finalize_shared_reduce_groups_for_launch`` declines when any
claimed axis exceeds the static shape, and the launcher checks the recorded
assumption against the ``block=`` it finally emits.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ..ast_extension import statement_from_string
from ..source_location import SyntheticLocation

if TYPE_CHECKING:
    from collections.abc import Mapping

# The shared-memory cross-warp reductions that take ``group_span`` /
# ``group_count`` over the flattened CTA thread id: the two-stage scalar
# reduce and the register tile's column reduce.
_GROUPED_SHARED_REDUCES = frozenset(
    {
        "_cute_grouped_reduce_shared_two_stage",
        "_cute_grouped_reduce_shared_columns",
    }
)


def _runtime_lane_expr(index_type: str) -> str:
    """The flattened thread id of ``_cute_runtime_lane_group_params`` for
    ``index_type`` (``cutlass.Int32`` or ``cutlass.Int64``)."""
    tid = [f"{index_type}(cute.arch.thread_idx()[{axis}])" for axis in range(3)]
    bdim = [f"{index_type}(cute.arch.block_dim()[{axis}])" for axis in range(2)]
    return (
        f"{tid[0]} + ({tid[1]}) * ({bdim[0]}) + ({tid[2]}) * ({bdim[0]}) * ({bdim[1]})"
    )


# Runtime lane dump -> the ``thread_idx()[0]`` lane of the same index type.
_X_LANE_BY_RUNTIME_DUMP = {
    ast.dump(ast.parse(_runtime_lane_expr(index_type), mode="eval").body): (
        f"{index_type}(cute.arch.thread_idx()[0])"
    )
    for index_type in ("cutlass.Int32", "cutlass.Int64")
}


def _integer_keyword(call: ast.Call, name: str) -> int | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            value = keyword.value
            if isinstance(value, ast.Constant) and type(value.value) is int:
                return value.value
            return None
    return None


class _Finalize(ast.NodeTransformer):
    def __init__(self, thread_block_dims: tuple[int, int, int]) -> None:
        self.block_threads = (
            thread_block_dims[0] * thread_block_dims[1] * thread_block_dims[2]
        )
        self.one_dimensional = thread_block_dims[1] == thread_block_dims[2] == 1
        self.changed = False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not (
            isinstance(node.func, ast.Name) and node.func.id in _GROUPED_SHARED_REDUCES
        ):
            return node
        group_span = _integer_keyword(node, "group_span")
        group_count = _integer_keyword(node, "group_count")
        if (
            group_span is None
            or group_count is None
            or group_span <= 0
            or self.block_threads % group_span
        ):
            return node
        launched = self.block_threads // group_span
        if launched < group_count:
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
    transformer = _Finalize(thread_block_dims)
    result = [transformer.visit(stmt) for stmt in body]
    if not transformer.changed:
        return body, False
    return [ast.fix_missing_locations(stmt) for stmt in result], True


def finalize_shared_reduce_groups(
    body: list[ast.stmt], *, thread_block_dims: tuple[int, int, int]
) -> list[ast.stmt]:
    """Rewrite over-provisioned ``group_count`` values (and the runtime lane
    expression of a one-dimensional block) for the launch shape
    ``thread_block_dims``, which the caller must have proven."""
    return _finalize(body, thread_block_dims)[0]


def finalize_shared_reduce_groups_for_launch(
    body: list[ast.stmt],
    *,
    thread_block_dims: tuple[int, int, int],
    claimed_axis_sizes: Mapping[int, int],
) -> tuple[list[ast.stmt], tuple[int, int, int] | None]:
    """``finalize_shared_reduce_groups`` gated on the launch shape.

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
