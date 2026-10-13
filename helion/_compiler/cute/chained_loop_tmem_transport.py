"""Late-proven, eagerly packed TMEM operands in a common contraction loop.

Every packed image occupies a distinct slot through the entire iteration. A
source is only removed from shared materialization after its original operand
expressions have been evaluated against its exact fragment coordinates and
side inputs already available at publication. Authoritative carries are not
rounded or relocated by this component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .chained_loop_tmem_bridges import plan_loop_tmem_bridges
from .chained_tmem_transport import TmemOperandBinding

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_contraction_groups import ContractionGroup
    from .chained_execution import ChainedExecution
    from .chained_loop_tmem_slots import TmemOperandSlot
    from .chained_matmul import ChainedMatmulPlan


@dataclass(frozen=True)
class LoopTmemTransport:
    slot: TmemOperandSlot
    prefix: str
    coordinates: tuple[str, str]
    expression_lines: tuple[str, ...]
    masked_value: str
    dtype: str

    @property
    def operand(self) -> TmemOperandBinding:
        candidate = self.slot.candidate
        return TmemOperandBinding(
            candidate.destination_group,
            candidate.physical_shape,
            candidate.dtype,
            self.slot.column_offset,
        )

    def emit(self, execution: ChainedExecution) -> list[str]:
        from .chained_tmem_transport import emit_packed_tmem_fragment

        return emit_packed_tmem_fragment(
            self.prefix,
            f"chain_{self.slot.candidate.source_stage}",
            self.slot.candidate.physical_shape,
            self.dtype,
            self.coordinates,
            self.expression_lines,
            self.masked_value,
            execution=execution,
            destination=f"({execution.tmem} + {self.slot.column_offset})",
        )


def prepare_loop_tmem_transports(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    frontiers: dict[Node, str],
    groups: tuple[ContractionGroup, ...],
) -> tuple[tuple[LoopTmemTransport, ...], int]:
    """Prove expressions before reserving slots or skipping any original fill.

    Only current frontiers, lexical captures, immutable current carries and the
    source C fragment are available. In particular, do not optimistically bind
    later contractions: packing occurs immediately after this producer, even
    for a nonadjacent consumer. Each grouped member must independently preserve
    the source coordinates and the same original operand-domain mask.
    """
    from ..compile_environment import CompileEnvironment
    from . import chained_matmul as chain
    from .chained_loop_tmem_slots import plan_tmem_operand_slots

    working = max(group.physical[1] for group in groups)
    candidates = plan_loop_tmem_bridges(plan, groups)
    if not candidates:
        return (), working
    accepted = []
    expressions = []
    for candidate in candidates:
        prefix = f"chain_{candidate.source_stage}_transport"
        coords = (f"{prefix}_row", f"{prefix}_column")
        source_coords = candidate.source_geometry.result_coordinates(*coords)
        dtype = CompileEnvironment.current().backend.dtype_str(candidate.dtype)
        first = None
        domains = []
        try:
            for operand, geometry in zip(
                candidate.operands,
                candidate.destination_group.geometries,
                strict=True,
            ):
                expression = chain._Expression(cg, plan, frontiers)
                expression.coordinate_names.update(coords)
                expression.fragments[candidate.source] = (
                    source_coords,
                    f"chain_{candidate.source_stage}_values[{prefix}_index]",
                )
                operand_coords = geometry.operand("a", *coords)[1]
                value = expression.value(operand, operand_coords)
                domain = chain._operand_domain(cg, operand, operand_coords, plan)
                domains.append(domain)
                if first is None:
                    first = (
                        tuple(expression.lines),
                        chain._masked_operand(value, dtype, domain),
                    )
        except chain._UnsupportedChain:
            continue
        if first is None or any(domain != domains[0] for domain in domains[1:]):
            continue
        accepted.append(candidate)
        expressions.append((prefix, coords, *first, dtype))
    slots = plan_tmem_operand_slots(tuple(accepted), working)
    if slots is None:
        return (), working
    return (
        tuple(
            LoopTmemTransport(slot, *expression)
            for slot, expression in zip(slots.slots, expressions, strict=True)
        ),
        slots.required_columns,
    )
