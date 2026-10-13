"""Canonicalize explicit operand coordinates before committing a native root."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from ... import exc
from ..ast_deepcopy import deepcopy_ast
from .chained_root_native import joined_root
from .proven_loop_bounds import simplify_proven_loop_bounds

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan


def _launch(
    cg: GenerateAST, plan: ChainedMatmulPlan
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    df = cg.device_function
    state = df.cute_state
    if (
        not joined_root(plan)
        or plan.preparation_pipeline is not None
        or state.chained_matmul_plan is not plan
        or state.simt_cluster_n != 1
        or state.direct_affine_plan is not None
        or state.collective_register_chain_block_dims is not None
        or cg.cute_wrapper_plans
        or df.config.pid_type != "flat"
        or type(plan.threads) is not int
        or not 0 < plan.threads <= 1024
        or any(
            type(size) is not int or not 0 < size < (1 << 31)
            for _axis, extent, block in plan.axes
            for size in (extent, block)
        )
    ):
        return None
    if (loop := plan.loop) is not None:
        if plan.warp_loop_carry is None:
            return None
        begin, end = loop.begin, loop.end
        if (
            type(begin) is not int
            or type(end) is not int
            or not 0 <= begin <= end < (1 << 31)
        ):
            return None
        step = df.resolved_block_size(loop.block_id)
        if type(step) is not int or step != 1:
            return None
        plan.warp_loop_carry.check(plan)
    # These are the dimensions installed by the independent whole-root
    # adapter, also for a checked scalar-step warp carry. The bounds walker
    # still derives induction facts from the actual body, never these fields.
    # The generic carrier's lane layout does not own this body.
    blocks = math.prod((extent + block - 1) // block for _, extent, block in plan.axes)
    if not 0 < blocks < (1 << 31):
        return None
    return (plan.threads, 1, 1), (blocks, 1, 1)


def _excluded_names(cg: GenerateAST) -> frozenset[str]:
    groups = cg.device_function._variable_renames
    return frozenset(groups) | frozenset(
        name for group in groups.values() for name in group
    )


@dataclass(frozen=True)
class CoordinateBoundsCanonicalization:
    """Bind the original installed template to one predicate-only rewrite."""

    device: object
    plan: ChainedMatmulPlan
    config: object
    revision: object
    launch: tuple[tuple[int, int, int], tuple[int, int, int]]
    excluded_names: frozenset[str]
    before: tuple[str, ...]
    after: tuple[str, ...]

    def check_context(self, cg: GenerateAST, plan: ChainedMatmulPlan) -> None:
        from .native_matmul_metadata import _revision
        from .native_matmul_metadata import _snapshot

        if (
            self.device is not cg.device_function
            or self.plan is not plan
            or self.config != _snapshot(cg.device_function.config.config)
            or self.revision != _revision(plan)
            or self.launch != _launch(cg, plan)
            or self.excluded_names != _excluded_names(cg)
        ):
            raise exc.BackendUnsupported(
                "cute", "coordinate bounds proof context changed"
            )

    def check(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        before: Sequence[ast.AST],
    ) -> None:
        self.check_bodies(cg, plan, before, cg.device_function.body)

    def check_bodies(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        before: Sequence[ast.AST],
        after: Sequence[ast.AST],
    ) -> None:
        self.check_context(cg, plan)
        if self.before != tuple(map(ast.dump, before)) or self.after != tuple(
            map(ast.dump, after)
        ):
            raise exc.BackendUnsupported("cute", "coordinate bounds proof body changed")


def canonicalize_coordinate_bounds(
    cg: GenerateAST, plan: ChainedMatmulPlan, body: list[ast.stmt]
) -> tuple[list[ast.stmt], CoordinateBoundsCanonicalization | None]:
    launch = _launch(cg, plan)
    if launch is None:
        return body, None
    from .native_matmul_metadata import _revision
    from .native_matmul_metadata import _snapshot

    excluded = _excluded_names(cg)
    before = tuple(map(ast.dump, body))
    # This is a new, uncommitted root body. Preserve it for the stage coverage
    # receipt; never copy a committed body to hide later mutations.
    result = simplify_proven_loop_bounds(
        deepcopy_ast(body),
        enabled=True,
        thread_block_dims=launch[0],
        block_grid_dims=launch[1],
        constexpr_values={},
        excluded_names=excluded,
    )
    after = tuple(map(ast.dump, result))
    if before == after:
        return body, None
    return result, CoordinateBoundsCanonicalization(
        cg.device_function,
        plan,
        _snapshot(cg.device_function.config.config),
        _revision(plan),
        launch,
        excluded,
        before,
        after,
    )
