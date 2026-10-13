"""Typed coordinates of the original, bounded warp operand staging loop."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from .chained_coordinates import CoordinateContext
from .chained_coordinates import CoordinateDomain
from .chained_coordinates import CoordinateText
from .chained_coordinates import IndexExpr
from .chained_coordinates import _bounded
from .chained_coordinates import _values
from .chained_coordinates import original_coordinate
from .chained_execution import ChainedExecution
from .chained_root_native import joined_root
from .chained_warp_stage import _snapshot
from .chained_warp_stage import warp_revision

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_root_warp_stage import WarpOperandLayout


class OperandCoordinates(CoordinateContext):
    """Keep bounded non-affine view pieces typed, without rewriting them."""

    def __init__(self, domain: CoordinateDomain) -> None:
        super().__init__(domain)
        # Both coordinates derive from a runtime thread/step index. Unlike a
        # row island's constexpr column, a constant replacement remains Int32.
        self.expressions[domain.column] = IndexExpr("int32", (IndexExpr("column"),))
        self.values: dict[IndexExpr, tuple[int, ...] | None] = {}

    def bounded_values(self, expression: IndexExpr) -> tuple[int, ...] | None:
        if not _bounded(expression):
            return None
        if expression not in self.values:
            points = self.domain.points()
            self.values[expression] = (
                None
                if points is None
                else _values(expression, points, cache=self._values_cache)
            )
        return self.values[expression]

    def normalize(self, expression: IndexExpr, original: str) -> str:
        if self.bounded_values(expression) is None:
            return original
        result = super().normalize(expression, original)
        # A modulo piece may be non-affine while its subsequent reshape is
        # affine. Retain the checked typed DAG, never infer it from a string.
        self.expressions.setdefault(original, expression)
        return CoordinateText(result, original_coordinate(result))

    def in_bounds(self, coordinate: str, extent: int) -> bool:
        expression = self.lookup(coordinate)
        if type(extent) is not int or extent <= 0 or expression is None:
            return False
        values = self.bounded_values(expression)
        return values is not None and all(0 <= value < extent for value in values)


@dataclass(frozen=True)
class OperandCoordinateScope:
    codegen: GenerateAST
    plan: ChainedMatmulPlan
    stage: int
    layout: WarpOperandLayout
    schedule: str
    vector: int
    steps: int
    load_range: str
    domain: CoordinateDomain
    execution: ChainedExecution
    revision: object
    config: object
    layout_facts: object

    def check(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        stage: int,
        layout: WarpOperandLayout,
    ) -> None:
        from .chained_matmul import _UnsupportedChain

        if (
            cg is not self.codegen
            or plan is not self.plan
            or stage != self.stage
            or layout is not self.layout
        ):
            raise _UnsupportedChain("foreign warp operand coordinate attempt")
        if not 0 <= self.stage < len(plan.dots):
            raise _UnsupportedChain("warp operand coordinate stage changed")
        rows, columns, reduction = plan.shapes[self.stage]
        shape = (rows, reduction) if layout.role == "a" else (columns, reduction)
        index = f"chain_{self.stage}_load"
        x, y = (
            (f"{index} // {shape[1]}", f"{index} % {shape[1]}")
            if layout.inner == 1
            else (f"{index} % {shape[0]}", f"{index} // {shape[0]}")
        )
        coords = (x, y) if layout.role == "a" else (y, x)
        logical = shape if layout.role == "a" else shape[::-1]
        if (
            not joined_root(plan)
            or plan.loop is not None
            or plan.preparation_pipeline is not None
            or self.execution != ChainedExecution(plan.threads)
            or layout.role not in ("a", "b")
            or layout.inner not in (0, 1)
            or layout.shape != shape
            or layout.node
            is not plan.dots[self.stage].args[0 if layout.role == "a" else 1]
            or layout.dtype != plan.operand_dtype(self.stage)
            or _layout_facts(layout) != self.layout_facts
            or warp_revision(plan, aliases=False) != self.revision
            or _snapshot(self.codegen.device_function.config.config) != self.config
            or self.domain
            != CoordinateDomain(*coords, (0, logical[0] - 1), (0, logical[1] - 1))
            or self.domain.points() is None
            or self.vector not in (1, 2, 4, 8)
            or self.vector
            != (1 if self.schedule == "k_major" else min(8, shape[layout.inner]))
            or self.steps != math.ceil(math.prod(shape) / (plan.threads * self.vector))
            or self.load_range != _load_range(self.schedule, self.steps)
            or not 0 < self.steps * plan.threads * self.vector < 2**31
        ):
            raise _UnsupportedChain("warp operand coordinate owner or domain changed")

    def fallback(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        stage: int,
        layout: WarpOperandLayout,
        expressions: list[str],
        value: str,
        dtype: str,
        bounds: list[str],
    ) -> list[str]:
        from .chained_matmul import _indent
        from .chained_matmul import _masked_operand

        self.check(cg, plan, stage, layout)
        prefix = f"chain_{self.stage}"
        x, y = (self.domain.row, self.domain.column)
        if self.layout.role == "b":
            x, y = y, x
        return [
            f"for {prefix}_load_step in {self.load_range}:",
            f"    for {prefix}_load_vec in cutlass.range_constexpr({self.vector}):",
            f"        {prefix}_load = chain_thread * {self.vector} + {prefix}_load_step * {self.plan.threads * self.vector} + {prefix}_load_vec",
            f"        if {prefix}_load < {math.prod(self.layout.shape)}:",
            _indent(expressions, 12),
            f"            {prefix}_{self.layout.role}[{x}, {y}] = {_masked_operand(value, dtype, bounds)}",
        ]


def operand_scope(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    stage: int,
    layout: WarpOperandLayout,
    schedule: str,
    vector: int,
    steps: int,
    load_range: str,
) -> OperandCoordinateScope | None:
    if (
        cg.device_function.config.config.get("cute_chained_operand_coordinates")
        is not True
        or not joined_root(plan)
        or plan.loop is not None
        or plan.preparation_pipeline is not None
        or any(type(size) is not int or not 0 < size <= 32 for size in layout.shape)
    ):
        return None
    index = f"chain_{stage}_load"
    x, y = (
        (f"{index} // {layout.shape[1]}", f"{index} % {layout.shape[1]}")
        if layout.inner == 1
        else (f"{index} % {layout.shape[0]}", f"{index} // {layout.shape[0]}")
    )
    coords = (x, y) if layout.role == "a" else (y, x)
    shape = layout.shape if layout.role == "a" else layout.shape[::-1]
    scope = OperandCoordinateScope(
        cg,
        plan,
        stage,
        layout,
        schedule,
        vector,
        steps,
        load_range,
        CoordinateDomain(*coords, (0, shape[0] - 1), (0, shape[1] - 1)),
        ChainedExecution(plan.threads),
        warp_revision(plan, aliases=False),
        _snapshot(cg.device_function.config.config),
        _layout_facts(layout),
    )
    scope.check(cg, plan, stage, layout)
    return scope


def _layout_facts(layout: WarpOperandLayout) -> object:
    return _snapshot(
        (
            layout.node,
            layout.role,
            layout.shape,
            layout.stride,
            layout.inner,
            layout.dtype,
        )
    )


def _load_range(schedule: str, steps: int) -> str:
    return (
        f"cutlass.range({steps}, unroll=1)"
        if schedule
        in (
            "coalesced",
            "cp_async",
            "cp_async_register",
            "cp_async_register_reuse",
            "cp_async_register_reuse_scan",
        )
        else f"cutlass.range_constexpr({steps})"
    )
