"""Bounded, separable uses of completed M64 result registers.

No allocation, MMA, join or expression is replaced here. Only original operand
fills, seed reads and exclusively consumed C publications change transport.
Registers remain inside the same ordered 128-thread role and loop iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal

import torch

from .chained_contraction_groups import _physical_left
from .chained_loop_tmem_bridges import _ancestors
from .chained_loop_tmem_bridges import _coordinate_path
from .chained_loop_tmem_bridges import plan_loop_tmem_bridges
from .chained_seed_tiles import plan_seed_panels

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_completed_members import CompletedMemberCandidate
    from .chained_contraction_groups import ContractionGroup
    from .chained_execution import ChainedExecution
    from .chained_fragment_epilogue import FragmentExpression
    from .chained_matmul import ChainedMatmulPlan
    from .chained_pipeline_storage import StageTransports


@dataclass(frozen=True)
class FragmentUse:
    source: ContractionGroup
    destination: ContractionGroup
    member: int
    kind: Literal["operand", "seed"]
    destinations: tuple[int, ...]
    targets: tuple[Node, ...]
    width: int
    paths: tuple[Node, ...]

    @property
    def offset(self) -> int:
        return self.source.offsets[self.source.stages.index(self.member)]


def _other_contraction(target: Node, source: Node, dots: set[Node]) -> bool:
    """A source is an opaque completed value, including its own ancestors."""
    pending = [target]
    seen: set[Node] = set()
    while pending:
        node = pending.pop()
        if node is source or node in seen:
            continue
        if node in dots:
            return True
        seen.add(node)
        pending.extend(node.all_input_nodes)
    return False


def fragment_uses(
    plan: ChainedMatmulPlan, groups: tuple[ContractionGroup, ...]
) -> tuple[FragmentUse, ...]:
    """Find exact A/seed edges spanning at most one intervening native group."""
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
    dots = set(plan.dots)
    forbidden = {
        *region.live_outs,
        *region.stores,
        *region.scans,
        *region.reductions,
        *(carry.output for carry in region.carries),
    }
    result = []
    occupied: set[tuple[int, str]] = set()
    for position, source in enumerate(groups):
        width = source.geometries[0].physical[1]
        if (
            len(source.stages) not in (1, 2)
            or width not in (32, 64, 128)
            or source.physical[:2] != (64, len(source.stages) * width)
            or any(
                geometry.native_rows != 64
                or geometry.physical[:2] != (64, width)
                or region.contractions[member].result_dtype != torch.float32
                for member, geometry in zip(
                    source.stages, source.geometries, strict=True
                )
            )
        ):
            continue
        for destination in groups[position + 1 : position + 3]:
            if any(
                geometry.native_rows != 64
                or geometry.physical[0] != 64
                or region.contractions[member].result_dtype != torch.float32
                for member, geometry in zip(
                    destination.stages, destination.geometries, strict=True
                )
            ):
                continue
            for member, source_geometry in zip(
                source.stages, source.geometries, strict=True
            ):
                node = plan.dots[member]
                options: list[tuple[Literal["operand", "seed"], tuple[int, ...]]] = [
                    ("operand", destination.stages),
                    *(("seed", (index,)) for index in destination.stages),
                ]
                for kind, destinations in options:
                    key = (destinations[0], kind)
                    if key in occupied:
                        continue
                    paths: set[Node] = set()
                    targets = []
                    for index in destinations:
                        geometry = destination.geometries[
                            destination.stages.index(index)
                        ]
                        spec = region.contractions[index]
                        target = (
                            (spec.lhs, spec.rhs)[int(geometry.transpose)]
                            if kind == "operand"
                            else spec.accumulator
                        )
                        if (
                            target is None
                            or node not in ancestors[target]
                            or _other_contraction(target, node, dots)
                            or (
                                kind == "operand"
                                and (
                                    destination.physical[::2] != (64, width)
                                    or target.meta["val"].dtype
                                    not in (torch.float16, torch.bfloat16)
                                    or spec.operand_dtypes
                                    != (target.meta["val"].dtype,) * 2
                                    or _physical_left(plan.dots[index], geometry)
                                    != _physical_left(
                                        plan.dots[destination.stages[0]],
                                        destination.geometries[0],
                                    )
                                )
                            )
                            or (
                                kind == "seed"
                                and (
                                    geometry.physical[1] != width
                                    or target.meta["val"].dtype != torch.float32
                                )
                            )
                        ):
                            break
                        path = _coordinate_path(
                            target,
                            geometry.operand("a", "row", "column")[1]
                            if kind == "operand"
                            else geometry.result_coordinates("row", "column"),
                            node,
                            source_geometry.result_coordinates("row", "column"),
                            ancestors,
                        )
                        if path is None or path & forbidden:
                            break
                        targets.append(target)
                        paths.update(path)
                    else:
                        result.append(
                            FragmentUse(
                                source,
                                destination,
                                member,
                                kind,
                                destinations,
                                tuple(targets),
                                width,
                                tuple(n for n in region.nodes if n in paths),
                            )
                        )
                        occupied.add(key)
    return tuple(result)


def exclusive_members(
    plan: ChainedMatmulPlan, uses: tuple[FragmentUse, ...]
) -> frozenset[int]:
    """Keep original publication for every user outside the accepted paths."""
    result = set()
    for member in {use.member for use in uses}:
        selected = tuple(use for use in uses if use.member == member)
        paths = {node for use in selected for node in use.paths}
        terminals = {
            (target, plan.dots[index])
            for use in selected
            for target, index in zip(use.targets, use.destinations, strict=True)
        }
        if all(
            user in paths or (node, user) in terminals
            for node in paths
            for user in node.users
        ):
            result.add(member)
    return frozenset(result)


@dataclass(frozen=True)
class BoundFragmentUse:
    use: FragmentUse
    prefix: str
    expressions: tuple[FragmentExpression, ...]

    def operand(self, execution: ChainedExecution) -> list[str]:
        from . import chained_matmul as chain

        use, prefix = self.use, self.prefix
        value = self.expressions[0]
        value.check()
        source = f"chain_{use.source.stages[0]}"
        target = f"chain_{use.destination.stages[0]}_a"
        panel = plan_seed_panels(use.source.physical[:2], use.offset, use.width)[0]
        return [
            *panel.store_views(prefix, source, execution.thread),
            f"{prefix}_logical = cute.make_tensor({target}.iterator, cute.composition({target}.layout, cute.make_layout(((64, {use.width}), 1, 1))))",
            f"{prefix}_target_a = {prefix}_thread.partition_S({prefix}_logical)",
            f"{prefix}_values_a = cute.make_rmem_tensor({prefix}_target_a.shape, {value.dtype})",
            f"for {prefix}_index in cutlass.range_constexpr({use.width // 2}):",
            f"    {prefix}_row, {prefix}_column = {source}_coords[{prefix}_index + {use.offset // 2}]",
            f"    {prefix}_column = {prefix}_column - {use.offset}",
            chain._indent(value.expression.lines),
            f"    {prefix}_values_a[{prefix}_index] = {value.masked_value()}",
            f"cute.autovec_copy({prefix}_values_a, {prefix}_target_a)",
        ]

    def seed(self, seed: str, offset: int) -> list[str]:
        from . import chained_matmul as chain

        value, prefix = self.expressions[0], self.prefix
        value.check()
        return [
            f"for {prefix}_index in cutlass.range_constexpr({self.use.width // 2}):",
            f"    {prefix}_row, {prefix}_column = {seed}_coords[{prefix}_index]",
            f"    {prefix}_column = {prefix}_column - {offset}",
            chain._indent(value.expression.lines),
            f"    {seed}_values[{prefix}_index] = cutlass.Float32({value.masked_value()})",
        ]


@dataclass(frozen=True)
class FragmentReuse:
    uses: tuple[BoundFragmentUse, ...]
    omitted: frozenset[int]
    groups: tuple[ContractionGroup, ...]

    def operand(self, group: ContractionGroup) -> BoundFragmentUse | None:
        return next(
            (
                item
                for item in self.uses
                if item.use.destination == group and item.use.kind == "operand"
            ),
            None,
        )

    def seed(self, member: int) -> BoundFragmentUse | None:
        return next(
            (
                item
                for item in self.uses
                if item.use.kind == "seed" and item.use.destinations == (member,)
            ),
            None,
        )

    def check(self, plan: ChainedMatmulPlan) -> None:
        from . import chained_matmul as chain

        candidates = fragment_uses(plan, self.groups)
        selected = tuple(item.use for item in self.uses)
        if (
            not selected
            or any(use not in candidates for use in selected)
            or self.omitted != exclusive_members(plan, selected)
        ):
            raise chain._UnsupportedChain("fragment reuse graph or users changed")
        for item in self.uses:
            for expression in item.expressions:
                expression.check()

    def check_terminal(self, candidate: CompletedMemberCandidate) -> None:
        """Only an independent last-group export may omit its own C request."""
        from . import chained_matmul as chain

        if (
            candidate.group != self.groups[-1]
            or candidate.ownership is None
            or candidate.member in self.omitted
            or any(
                use.use.member == candidate.member
                or use.use.source == candidate.group
                or any(node in candidate.descendants for node in use.use.paths)
                for use in self.uses
            )
        ):
            raise chain._UnsupportedChain("fragment reuse overlaps terminal export")


def bind_fragment_reuse(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    frontiers: dict[Node, str],
    selections: tuple[StageTransports, ...],
    execution: ChainedExecution,
) -> FragmentReuse | None:
    from ..compile_environment import CompileEnvironment
    from . import chained_matmul as chain
    from .chained_fragment_epilogue import bind_fragment_expression

    if execution.threads != 128:
        return None
    groups = tuple(item.group for item in selections)
    bound = []
    for use in fragment_uses(plan, groups):
        source = selections[groups.index(use.source)]
        destination = selections[groups.index(use.destination)]
        if (
            source.tmem_output is not None
            or source.tmem_carry is not None
            or source.residency is not None
            and source.residency.source_stage == use.source.stages[0]
            or destination.tmem_output is not None
            or use.kind == "operand"
            and destination.tmem_input is not None
            or use.kind == "seed"
            and (
                destination.residency is not None
                and destination.residency.destination_member in use.destinations
                or destination.tmem_carry is not None
                and destination.tmem_carry.candidate.member in use.destinations
            )
        ):
            continue
        prefix = f"chain_fragment_{use.member}_{use.kind}_{use.destinations[0]}"
        coords = (f"{prefix}_row", f"{prefix}_column")
        source_geometry = use.source.geometries[use.source.stages.index(use.member)]
        fragment = (
            f"chain_{use.source.stages[0]}_values[{prefix}_index + {use.offset // 2}]"
        )
        dtype = CompileEnvironment.current().backend.dtype_str(
            use.targets[0].meta["val"].dtype
        )
        expressions = []
        try:
            for index, target in zip(use.destinations, use.targets, strict=True):
                geometry = use.destination.geometries[
                    use.destination.stages.index(index)
                ]
                expressions.append(
                    bind_fragment_expression(
                        cg,
                        plan,
                        frontiers,
                        [],
                        plan.dots[use.member],
                        target,
                        source_geometry.result_coordinates(*coords),
                        geometry.operand("a", *coords)[1]
                        if use.kind == "operand"
                        else geometry.result_coordinates(*coords),
                        fragment,
                        dtype,
                        operand_domain=use.kind == "operand",
                    )
                )
            if any(item.domain != expressions[0].domain for item in expressions[1:]):
                raise chain._UnsupportedChain("fragment reuse domains differ")
        except chain._UnsupportedChain:
            continue
        bound.append(BoundFragmentUse(use, prefix, tuple(expressions)))
    if not bound:
        return None
    return FragmentReuse(
        tuple(bound), exclusive_members(plan, tuple(item.use for item in bound)), groups
    )
