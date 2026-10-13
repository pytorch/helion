"""Adjacent full-M64 result fragments used by typed A and an FP32 seed.

This is an instruction-transport choice, not storage authority. The ordinary
C arena remains allocated and charged. Both original MMA issues, completion
waits, joins and seed/publication fences remain in place. No register value
crosses a role predicate or a lexical loop iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import torch

from .chained_contraction_groups import _physical_left
from .chained_loop_tmem_bridges import _ancestors
from .chained_loop_tmem_bridges import _coordinate_path
from .chained_loop_tmem_bridges import plan_loop_tmem_bridges
from .chained_seed_tiles import plan_seed_panels

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_contraction_groups import ContractionGroup
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_pipeline_storage import StageTransports


@dataclass(frozen=True)
class GroupedFragmentCandidate:
    source: ContractionGroup
    destination: ContractionGroup
    operand_member: int
    seed_source_member: int
    seed_member: int
    width: int
    operands: tuple[Node, ...]
    seed: Node
    paths: tuple[Node, ...]


def grouped_fragment_candidates(
    plan: ChainedMatmulPlan, groups: tuple[ContractionGroup, ...]
) -> tuple[GroupedFragmentCandidate, ...]:
    """Exclusive complete coordinate paths, after the existing loop proof.

    The existing bridge validator checks the retained typed graph and ordered
    group partition. Its singleton-M128 candidates are neither changed nor
    consumed here. Equal complete members have the proved sparse-copy order:
    each thread owns W/2 values from each member, in member order.
    """
    if (
        plan_loop_tmem_bridges(plan, groups) is None
        or plan.loop_mma_rows != 64
        or plan.warp_mma_stages
        or plan.pointwise_cache is not None
    ):
        return ()
    region = plan.region
    assert region is not None
    ancestors = {node: _ancestors(node) for node in region.nodes}
    forbidden = {
        *region.live_outs,
        *region.stores,
        *region.scans,
        *region.reductions,
        *(carry.output for carry in region.carries),
    }
    candidates = []
    occupied: set[int] = set()
    for source, destination in pairwise(groups):
        if (
            len(source.stages) != 2
            or source.stages[0] in occupied
            or destination.stages[0] in occupied
            or any(geometry.native_rows != 64 for geometry in source.geometries)
        ):
            continue
        width = source.geometries[0].physical[1]
        if (
            width not in (32, 64, 128)
            or source.physical[:2] != (64, 2 * width)
            or any(geometry.physical[1] != width for geometry in source.geometries)
            or destination.physical[::2] != (64, width)
            or any(
                geometry.native_rows != 64
                or region.contractions[index].result_dtype != torch.float32
                for group in (source, destination)
                for index, geometry in zip(group.stages, group.geometries, strict=True)
            )
        ):
            continue
        operands = tuple(
            (region.contractions[index].lhs, region.contractions[index].rhs)[
                int(geometry.transpose)
            ]
            for index, geometry in zip(
                destination.stages, destination.geometries, strict=True
            )
        )
        dtype = operands[0].meta["val"].dtype
        if dtype not in (torch.float16, torch.bfloat16) or any(
            region.contractions[index].operand_dtypes != (dtype, dtype)
            or _physical_left(plan.dots[index], geometry)
            != _physical_left(
                plan.dots[destination.stages[0]], destination.geometries[0]
            )
            for index, geometry in zip(
                destination.stages, destination.geometries, strict=True
            )
        ):
            continue
        for position in (0, 1):
            operand_member = source.stages[position]
            seed_source_member = source.stages[1 - position]
            operand_source = plan.dots[operand_member]
            seed_source = plan.dots[seed_source_member]
            seed_members = [
                index
                for index in destination.stages
                if (seed := region.contractions[index].accumulator) is not None
                and seed_source in ancestors[seed]
            ]
            if len(seed_members) != 1:
                continue
            seed_member = seed_members[0]
            seed_geometry = destination.geometries[
                destination.stages.index(seed_member)
            ]
            seed = region.contractions[seed_member].accumulator
            assert seed is not None
            if (
                seed_geometry.physical[1] != width
                or seed.meta["val"].dtype != torch.float32
                or operand_source in ancestors[seed]
            ):
                continue
            paths: set[Node] = set()
            terminals: set[tuple[Node, Node]] = set()
            valid = True
            for index, geometry, operand in zip(
                destination.stages, destination.geometries, operands, strict=True
            ):
                spec = region.contractions[index]
                other = spec.lhs if geometry.transpose else spec.rhs
                if (
                    operand_source not in ancestors[operand]
                    or seed_source in ancestors[operand]
                    or {operand_source, seed_source} & ancestors[other]
                    or (
                        spec.accumulator is not None
                        and operand_source in ancestors[spec.accumulator]
                    )
                ):
                    valid = False
                    break
                path = _coordinate_path(
                    operand,
                    geometry.operand("a", "row", "column")[1],
                    operand_source,
                    source.geometries[position].result_coordinates("row", "column"),
                    ancestors,
                )
                if path is None:
                    valid = False
                    break
                paths.update(path)
                terminals.add((operand, plan.dots[index]))
            seed_path = _coordinate_path(
                seed,
                seed_geometry.result_coordinates("row", "column"),
                seed_source,
                source.geometries[1 - position].result_coordinates("row", "column"),
                ancestors,
            )
            if not valid or seed_path is None:
                continue
            paths.update(seed_path)
            terminals.add((seed, plan.dots[seed_member]))
            if paths & forbidden or any(
                user not in paths and (node, user) not in terminals
                for node in paths
                for user in node.users
            ):
                continue
            candidates.append(
                GroupedFragmentCandidate(
                    source,
                    destination,
                    operand_member,
                    seed_source_member,
                    seed_member,
                    width,
                    operands,
                    seed,
                    tuple(node for node in region.nodes if node in paths),
                )
            )
            occupied.update((source.stages[0], destination.stages[0]))
            break
    return tuple(candidates)


@dataclass(frozen=True)
class GroupedFragmentBridge:
    candidate: GroupedFragmentCandidate
    prefix: str
    dtype: str
    operand_lines: tuple[str, ...]
    operand_value: str
    seed_lines: tuple[str, ...]
    seed_value: str

    def operand(self, execution: ChainedExecution) -> list[str]:
        """Publish original typed A into its existing, now-dead operand arena."""
        from . import chained_matmul as chain

        candidate, prefix = self.candidate, self.prefix
        source = f"chain_{candidate.source.stages[0]}"
        target = f"chain_{candidate.destination.stages[0]}_a"
        position = candidate.source.stages.index(candidate.operand_member)
        offset = position * candidate.width
        panel = plan_seed_panels(
            candidate.source.physical[:2], offset, candidate.width
        )[0]
        return [
            *panel.store_views(prefix, source, execution.thread),
            f"{prefix}_logical = cute.make_tensor({target}.iterator, cute.composition({target}.layout, cute.make_layout(((64, {candidate.width}), 1, 1))))",
            f"{prefix}_target_a = {prefix}_thread.partition_S({prefix}_logical)",
            f"{prefix}_values_a = cute.make_rmem_tensor({prefix}_target_a.shape, {self.dtype})",
            f"for {prefix}_a_index in cutlass.range_constexpr({candidate.width // 2}):",
            f"    {prefix}_a_row, {prefix}_a_column = {source}_coords[{prefix}_a_index + {offset // 2}]",
            f"    {prefix}_a_column = {prefix}_a_column - {offset}",
            chain._indent(list(self.operand_lines)),
            f"    {prefix}_values_a[{prefix}_a_index] = {self.operand_value}",
            f"cute.autovec_copy({prefix}_values_a, {prefix}_target_a)",
        ]

    def seed(self, seed: str, offset: int) -> list[str]:
        """Original FP32 seed recipe in the existing member store partition."""
        from . import chained_matmul as chain

        candidate, prefix = self.candidate, self.prefix
        return [
            f"for {prefix}_seed_index in cutlass.range_constexpr({candidate.width // 2}):",
            f"    {prefix}_seed_row, {prefix}_seed_column = {seed}_coords[{prefix}_seed_index]",
            f"    {prefix}_seed_column = {prefix}_seed_column - {offset}",
            chain._indent(list(self.seed_lines)),
            f"    {seed}_values[{prefix}_seed_index] = cutlass.Float32({self.seed_value})",
        ]


def bind_grouped_fragment_bridges(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    frontiers: dict[Node, str],
    selections: tuple[StageTransports, ...],
    execution: ChainedExecution,
) -> tuple[GroupedFragmentBridge, ...]:
    """Prove exact expressions using only already-published current inputs."""
    from ..compile_environment import CompileEnvironment
    from . import chained_matmul as chain
    from .chained_fragment_epilogue import bind_fragment_expression

    if execution.threads != 128:
        return ()
    result = []
    groups = tuple(item.group for item in selections)
    for candidate in grouped_fragment_candidates(plan, groups):
        source = selections[groups.index(candidate.source)]
        destination = selections[groups.index(candidate.destination)]
        if (
            source.tmem_output is not None
            or source.tmem_carry is not None
            or source.residency is not None
            and source.residency.source_stage == candidate.source.stages[0]
            or destination.tmem_input is not None
            or destination.tmem_output is not None
            or destination.residency is not None
            and destination.residency.destination_member == candidate.seed_member
            or destination.tmem_carry is not None
            and destination.tmem_carry.candidate.member == candidate.seed_member
        ):
            continue
        prefix = f"chain_group_bridge_{candidate.source.stages[0]}"
        dtype = CompileEnvironment.current().backend.dtype_str(
            candidate.operands[0].meta["val"].dtype
        )
        expressions = []
        try:
            for kind, source_member, targets, target_geometries in (
                (
                    "a",
                    candidate.operand_member,
                    candidate.operands,
                    candidate.destination.geometries,
                ),
                (
                    "seed",
                    candidate.seed_source_member,
                    (candidate.seed,),
                    (
                        candidate.destination.geometries[
                            candidate.destination.stages.index(candidate.seed_member)
                        ],
                    ),
                ),
            ):
                coords = (f"{prefix}_{kind}_row", f"{prefix}_{kind}_column")
                position = candidate.source.stages.index(source_member)
                source_geometry = candidate.source.geometries[position]
                fragment = f"chain_{candidate.source.stages[0]}_values[{prefix}_{kind}_index + {position * candidate.width // 2}]"
                bound = [
                    bind_fragment_expression(
                        cg,
                        plan,
                        frontiers,
                        [],
                        plan.dots[source_member],
                        target,
                        source_geometry.result_coordinates(*coords),
                        geometry.operand("a", *coords)[1]
                        if kind == "a"
                        else geometry.result_coordinates(*coords),
                        fragment,
                        dtype if kind == "a" else "cutlass.Float32",
                        operand_domain=kind == "a",
                    )
                    for target, geometry in zip(targets, target_geometries, strict=True)
                ]
                if any(item.domain != bound[0].domain for item in bound[1:]):
                    raise chain._UnsupportedChain("grouped fragment domains differ")
                first = bound[0]
                expressions.append(
                    (tuple(first.expression.lines), first.masked_value())
                )
        except chain._UnsupportedChain:
            continue
        result.append(
            GroupedFragmentBridge(
                candidate,
                prefix,
                dtype,
                *expressions[0],
                *expressions[1],
            )
        )
    return tuple(result)
