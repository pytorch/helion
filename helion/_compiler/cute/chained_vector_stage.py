"""Cooperative operand vectors evaluated through the common expression lowering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ... import exc
from . import chained_matmul as chain
from .chained_vector_expression import emit_vector_expression

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_broadcast_retention import BroadcastOptions
    from .chained_broadcast_retention import BroadcastRetentionAttempt
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_pointwise_unroll import BoundedProducerUnroll
    from .chained_pointwise_unroll import PointwiseUnroll
    from .chained_tcgen_stage import StageGeometry


@dataclass
class VectorStaging:
    enabled: bool
    activated: bool = False
    group_enabled: bool = False
    group_activated: bool = False
    broadcast: BroadcastRetentionAttempt | None = None
    async_enabled: bool = False
    async_activated: bool = False

    def __post_init__(self) -> None:
        if type(self.async_enabled) is not bool:
            raise TypeError("async vector stores require an explicit bool")

    def validate(self) -> None:
        self.validate_group()
        if self.enabled and not self.activated:
            raise exc.BackendUnsupported(
                "cute", "vector staging requires a proven host operand vector"
            )

    def validate_group(self) -> None:
        if self.async_enabled and not self.async_activated:
            raise exc.BackendUnsupported(
                "cute", "async vector stores require an identity shared sink"
            )
        if self.group_enabled and not self.group_activated:
            raise exc.BackendUnsupported(
                "cute", "vector grouping requires shared coordinate-owned operands"
            )


@dataclass(frozen=True)
class VectorStageOperand:
    node: Node
    geometry: StageGeometry
    role: str
    shape: tuple[int, int]
    target: str
    offset: int = 0


def emit_vector_stage_group(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    operands: tuple[VectorStageOperand, ...],
    staging: VectorStaging,
    *,
    tag: str,
    producer_unroll: BoundedProducerUnroll | PointwiseUnroll | None = None,
    execution: ChainedExecution | None = None,
) -> list[str] | None:
    """Share original expressions inside one already synchronized fill phase.

    The caller binds disjoint destination views and retains its existing fence
    and barrier. This wrapper neither moves fills across a stage nor extends
    shared-memory lifetimes. Each output keeps its original operand-domain mask.
    """
    from .chained_vector_group import VectorGroupOutput
    from .chained_vector_group import emit_vector_group

    if (
        not staging.enabled
        or not staging.group_enabled
        or len(operands) < 2
        or any(operand.shape != operands[0].shape for operand in operands)
    ):
        return None
    outputs = tuple(
        VectorGroupOutput(
            node=operand.node,
            target=operand.target,
            coordinates=lambda row, column, operand=operand: operand.geometry.operand(
                operand.role, row, column
            )[1],
            offset=operand.offset,
            final_value=lambda value, dtype, coords, operand=operand: (
                chain._masked_operand(
                    value, dtype, chain._operand_domain(cg, operand.node, coords, plan)
                )
            ),
        )
        for operand in operands
    )
    options: BroadcastOptions = {}
    if staging.broadcast is not None:
        options["broadcast"] = staging.broadcast
    lines = emit_vector_group(
        cg,
        plan,
        boundaries,
        outputs,
        shape=operands[0].shape,
        tag=tag,
        producer_unroll=producer_unroll,
        execution=execution,
        **options,
    )
    if lines is not None:
        staging.activated = staging.group_activated = True
    return lines


def emit_vector_stage(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    operand: Node,
    geometry: StageGeometry,
    *,
    role: str,
    shape: tuple[int, int],
    offset: int,
    tag: str,
    target: str,
    producer_unroll: BoundedProducerUnroll | None = None,
    execution: ChainedExecution | None = None,
    broadcast: BroadcastRetentionAttempt | None = None,
) -> list[str] | None:
    """Vectorize operand stores and proven host leaves, retaining scalar arithmetic.

    Each producer owns eight consecutive reduction coordinates. Logical padding
    and source masks remain independent: allocation padding is zero-filled, and
    every unproven source vector uses the original scalar expression. No state
    or cache survives a producer iteration. A materialized boundary alone does
    not establish beneficial vector ownership: keep the original fill unless
    at least one host leaf has a proven vector load.
    """
    options: BroadcastOptions = {}
    if broadcast is not None:
        options["broadcast"] = broadcast
    return emit_vector_expression(
        cg,
        plan,
        boundaries,
        operand,
        shape=shape,
        coordinates=lambda row, column: geometry.operand(role, row, column)[1],
        offset=offset,
        tag=tag,
        target=target,
        final_value=lambda value, dtype, coords: chain._masked_operand(
            value, dtype, chain._operand_domain(cg, operand, coords, plan)
        ),
        producer_unroll=producer_unroll,
        execution=execution,
        **options,
    )
