"""Per-contraction instruction selection over already prepared operands.

The row limit is a schedule choice, not a different numerical graph. Physical
operand layouts, padding, grouping, source precision and logical C publication
belong to the common stage emitter. Explicit accumulator stages retain TCgen05.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ... import exc

if TYPE_CHECKING:
    from .chained_contraction_groups import ContractionGroup
    from .chained_matmul import ChainedMatmulPlan
    from .chained_tcgen_stage import StageGeometry


def warp_mma_shape(
    geometry: StageGeometry, group: ContractionGroup | None = None
) -> tuple[int, int, int]:
    """Trim padded M to whole warp atoms; retain prepared N and original K."""
    geometries = (geometry,) if group is None else group.geometries
    rows = max(item.logical[1 if item.transpose else 0] for item in geometries)
    rows = (rows + 15) // 16 * 16
    _, columns, reduction = geometry.physical if group is None else group.physical
    return rows, columns, reduction


def select_warp_mma_stages(
    plan: ChainedMatmulPlan,
    geometries: tuple[StageGeometry, ...],
    max_rows: int,
) -> frozenset[int]:
    groups = (
        tuple((index, geometry, None) for index, geometry in enumerate(geometries))
        if plan.contraction_groups is None
        else tuple(
            (group.stages[0], group.geometries[0], group)
            for group in plan.contraction_groups
        )
    )
    return frozenset(
        stage
        for stage, geometry, group in groups
        if warp_mma_shape(geometry, group)[0] <= max_rows
        and all(
            plan.dots[index].args[2] is None
            for index in ((stage,) if group is None else group.stages)
        )
    )


def validate_warp_mma_selection(
    plan: ChainedMatmulPlan | None, max_rows: object
) -> None:
    if type(max_rows) is int and max_rows == 0:
        return
    if (
        type(max_rows) is not int
        or max_rows not in (16, 32, 64, 128)
        or plan is None
        or plan.loop is None
        or plan.strategy != "tcgen05_tmem"
        or not plan.warp_mma_stages
    ):
        raise exc.BackendUnsupported(
            "cute", "warp MMA row limit requires an eligible common TCgen05 loop stage"
        )
