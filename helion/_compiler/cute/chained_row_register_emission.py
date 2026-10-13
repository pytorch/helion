"""Original typed recipes on completed row-owned register images."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from . import chained_matmul as chain
from .chained_coordinates import CoordinateContext
from .chained_coordinates import RowCoordinateScope
from .chained_coordinates import row_scope

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_row_registers import RowRegisterEvent
    from .chained_row_registers import RowRegisterIsland
    from .chained_row_registers import RowRegisterPlan
    from .chained_scratch_layout import ScratchLayouts
    from .chained_sparse_reduction import SparseReduction


def allocate_row_outputs(plan: RowRegisterPlan, scratch: ScratchLayouts) -> list[str]:
    return [
        f"{item.name} = cute.make_tensor(cute.arch.alloc_smem(cutlass.Float32, {item.shape[0] * item.shape[1]}, alignment=128), {scratch.layout(item.name, item.shape)})"
        for item in plan.islands
    ]


def _expression(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    island: RowRegisterIsland,
    event: RowRegisterEvent,
    row: str,
    column: str,
    registers: str,
    coordinates: CoordinateContext,
    scope: RowCoordinateScope | None = None,
) -> chain._Expression:
    expression = chain._Expression(cg, plan, boundaries)
    expression.coordinate_normalization = coordinates
    expression.coordinate_names.update((row, column))
    if plan.root_vector_cache is not None:
        from .chained_root_vector_cache import bind_read_scope

        if scope is None:
            raise chain._UnsupportedChain("root vector row requires its original scope")
        bind_read_scope(expression, scope)
    images = {item.node: item for item in island.events}
    for source in event.reads:
        image = images[source]
        proof = image.reduction
        if proof is None:
            coords, index = (row, column), f"{image.offset} + ({column})"
        elif proof.axis == 0:
            coords = ("0", column) if len(proof.output_shape) == 2 else (column,)
            index = f"{image.offset} + ({column})"
        else:
            coords = (row, "0") if len(proof.output_shape) == 2 else (row,)
            index = str(image.offset)
        expression.bind_fragment_image(source, coords, f"{registers}[{index}]")
        if image.literal_columns:
            # The versioned event initialized every column of this lane's row.
            # No other row or unproved coordinate is granted an image binding.
            if proof is not None or image.size != island.shape[1]:
                raise chain._UnsupportedChain("invalid complete row panel image")
            normalized = tuple(coordinates.coordinate(c) for c in coords)
            for position in range(island.shape[1]):
                fixed = (row, str(position))
                if tuple(coordinates.coordinate(c) for c in fixed) != normalized:
                    expression.bind_fragment_image(
                        source, fixed, f"{registers}[{image.offset + position}]"
                    )
    return expression


def _zero_adds(value: str, result: str, sides: tuple[str, ...]) -> list[str]:
    return [
        f"{result} = cutlass.Float32(0) + {value}",
        *(
            f"{result} = cutlass.Float32(0) + {result}"
            if side == "left"
            else f"{result} = {result} + cutlass.Float32(0)"
            for side in sides
        ),
    ]


def _normalized_zero_adds(
    plan: ChainedMatmulPlan,
    proof: SparseReduction,
    value: str,
    result: str,
) -> list[str]:
    """Retain the first FP32 add; fold only its proved positive-zero suffix.

    The first add establishes signed-zero/FTZ behavior and is never replaced
    with the selected source. The sparse proof establishes FP32 throughout,
    positive zero on every pruned branch, and no intervening cast or effect.
    No arbitrary pointwise x+0 expression is matched by this helper.
    """
    from .chained_sparse_reduction import plan_sparse_reduction

    if plan_sparse_reduction(plan, proof.node) != proof:
        raise chain._UnsupportedChain("normalized zero suffix proof changed")
    return _zero_adds(value, result, ())


def _emit_island(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    island: RowRegisterIsland,
    execution: ChainedExecution,
    *,
    warp: int | None = None,
) -> list[str]:
    if any(node not in boundaries for node in island.external_reads):
        raise chain._UnsupportedChain("row-register source is not published")
    tag = island.name
    row, column, registers = f"{tag}_row", f"{tag}_column", f"{tag}_values"
    rows, columns = island.shape
    body = [
        f"{row} = {execution.thread}"
        if warp is None
        else f"{row} = {execution.thread} - {32 * warp}",
        f"{registers} = cute.make_rmem_tensor(({island.register_slots},), cutlass.Float32)",
    ]
    for index, event in enumerate(island.events):
        proof = event.reduction
        col = str(proof.position) if proof is not None and proof.axis == 1 else column
        scope = row_scope(plan, island, event, row, column, execution, warp)
        coordinates = CoordinateContext(scope.domain)
        expression = _expression(
            cg, plan, boundaries, island, event, row, col, registers, coordinates, scope
        )
        source = event.node if proof is None else proof.source
        value = expression.value(source, (row, col))
        if proof is not None:
            value = chain._masked_operand(
                value,
                "cutlass.Float32",
                chain._operand_domain(
                    cg,
                    source,
                    (row, col),
                    plan,
                    coordinate_normalization=coordinates,
                ),
            )
        scope.check(plan)
        if plan.root_vector_cache is not None:
            from .chained_root_vector_cache import check_read_scope

            check_read_scope(expression)
        target = f"{registers}[{event.offset} + ({col})]"
        if proof is None:
            body.extend(
                [
                    f"if {row} < {rows}:",
                    f"    for {column} in cutlass.range_constexpr({columns}):",
                    chain._indent(expression.lines, 8),
                    f"        {target} = cutlass.Float32({value})",
                ]
            )
            continue
        accumulator = f"{tag}_selected_{index}"
        if proof.axis == 0:
            owner = f"{tag}_owner_{index}"
            selected = [
                f"{owner} = cutlass.Float32(0)",
                f"if {row} == {proof.position}:",
                chain._indent(expression.lines, 4),
                f"    {owner} = cutlass.Float32({value})",
                f"{owner} = cute.arch.shuffle_sync({owner}, {proof.position}, mask=-1, mask_and_clamp=31)",
                *_normalized_zero_adds(plan, proof, owner, accumulator),
                f"{target} = {accumulator}",
            ]
            body.extend(
                [
                    f"for {column} in cutlass.range_constexpr({columns}):",
                    chain._indent(selected, 4),
                ]
            )
        else:
            selected = [
                *expression.lines,
                *_normalized_zero_adds(plan, proof, value, accumulator),
                f"{registers}[{event.offset}] = {accumulator}",
            ]
            body.extend([f"if {row} < {rows}:", chain._indent(selected, 4)])
    output = island.events[-1]
    body.extend(
        [
            f"if {row} < {rows}:",
            f"    for {column} in cutlass.range_constexpr({columns}):",
            f"        {tag}[{row}, {column}] = {registers}[{output.offset} + {column}]",
        ]
    )
    if warp is None:
        return [f"if {execution.thread} < 32:", chain._indent(body, 4), execution.sync]
    if not 0 <= warp < execution.threads // 32:
        raise chain._UnsupportedChain("row-register owner is not a complete warp")
    return [
        f"if {execution.thread} // 32 == {warp}:",
        chain._indent(body, 4),
    ]


@dataclass
class RowRegisterEmission:
    proposal: RowRegisterPlan
    completed: set[Node] = field(default_factory=set)

    def emit_collective(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        node: Node,
        execution: ChainedExecution,
    ) -> list[str] | None:
        self.proposal.check(plan)
        island = next(
            (item for item in self.proposal.islands if node in item.reductions), None
        )
        if island is None:
            return None
        if node is not island.first_collective:
            if island.output not in self.completed:
                raise chain._UnsupportedChain(
                    "row-register selection precedes publication"
                )
            return []
        if island.output in self.completed or island.output in boundaries:
            raise chain._UnsupportedChain("duplicate row-register publication")
        lines = _emit_island(cg, plan, boundaries, island, execution)
        self.proposal.check(plan)
        boundaries[island.output] = island.name
        self.completed.add(island.output)
        return lines

    def validate(self, plan: ChainedMatmulPlan) -> None:
        self.proposal.check(plan)
        if self.completed != {item.output for item in self.proposal.islands}:
            raise chain._UnsupportedChain("row-register island was not published")
