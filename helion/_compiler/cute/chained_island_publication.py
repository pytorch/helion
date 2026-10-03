"""Disjoint full-owner publication of an original island consumer operand.

The graph candidate is not publication authority. The original island emitter
records its actual completed segment, and the original warp stage consumes that
publication exactly once. Physical allocation and final installation still
belong to the accepted preparation body.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import replace
import math
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

import torch
from torch.fx.node import map_arg

from ..compile_environment import CompileEnvironment
from .chained_matmul import _ancestors
from .chained_matmul import _Expression
from .chained_matmul import _indent
from .chained_matmul import _masked_operand
from .chained_matmul import _materialized_value
from .chained_matmul import _operand_domain
from .chained_matmul import _shape
from .chained_matmul import _UnsupportedChain
from .chained_mma_selection import warp_mma_shape
from .chained_pipeline_storage import _freeze
from .chained_preparation_reads import resolved_preparation_reads
from .chained_tcgen05 import _layout

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.fx import Node

    from ...runtime.settings import Settings
    from ..generate_ast import GenerateAST
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_operand_retention import OperandRetentionCandidate
    from .chained_operand_retention import _Revision
    from .chained_preparation_actions import AcceptedPreparation
    from .chained_preparation_actions import AcceptedPreparationAction
    from .chained_preparation_frame import PreparationStage
    from .chained_preparation_pipeline import PreparationPipeline
    from .chained_preparation_reads import CompletedPreparationFrontier
    from .chained_preparation_reads import CompletedPreparationStage
    from .chained_preparation_reads import PreparationReadFrontier
    from .chained_preparation_storage import AcceptedPreparationStorage
    from .chained_register_binding import BoundPreparationIsland
    from .chained_register_islands import RegisterImage
    from .chained_tcgen_stage import StageGeometry
    from .chained_vector_stage import VectorStageOperand
    from .chained_vector_stage import VectorStaging


def _context(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    bound: BoundPreparationIsland,
    stage: PreparationStage,
) -> tuple[object, ...]:
    """Copy primitive call facts; same-object mutation is not a new proof."""
    return (
        _freeze(cg.device_function.config.config),
        plan.dtype,
        tuple(plan.operand_dtype(i) for i in range(len(plan.dots))),
        plan.warp_mma_stages,
        bound.first_event,
        bound.stop_event,
        bound.published,
        _freeze(vars(bound.execution)),
        tuple(
            tuple(
                (issue.stage, issue.node, issue.origins, _freeze(vars(issue.geometry)))
                for issue in component.issues
            )
            for component in bound.island.components
        ),
        tuple(_freeze(vars(value)) for value in bound.island.values),
        tuple(
            (origin.node, origin.axes, origin.image.node, origin.image.tiles)
            for origin in bound.origins
        ),
        tuple(
            (item.node, item.name, item.shape, _freeze(vars(item.region)))
            for item in bound.exports
        ),
        tuple(_freeze(vars(item)) for item in bound.frame.layout.regions),
        tuple(_freeze(vars(item)) for item in bound.frame.actions),
        stage.group.stages,
        tuple(_freeze(vars(item)) for item in stage.group.geometries),
        stage.shape,
        _freeze(vars(stage.a)),
        _freeze(vars(stage.b)),
        tuple((node, tuple(node.users)) for node in bound.frame.cut.region.nodes),
    )


def _closed_users(bound: BoundPreparationIsland, operand: Node, dot: Node) -> bool:
    return island_export_cut(
        frozenset(item.node for item in bound.island.values),
        frozenset(item.node for item in bound.exports),
        operand,
        dot,
    )


def island_export_cut(
    inside: frozenset[Node],
    exports: frozenset[Node],
    operand: Node,
    dot: Node,
) -> bool:
    """Graph-only original export cut; never publication or read authority."""
    closure = _ancestors(operand)
    if not exports or not exports <= closure:
        return False
    occurrences = []
    map_arg((dot.args, dot.kwargs), lambda node: occurrences.append(node) or node)
    if occurrences.count(operand) != 1:
        return False
    pending, seen = list(exports), set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        if node is operand:
            # This genuine original SSA image is published completely. Its
            # later users are proved separately, not discarded as dead C uses.
            continue
        for user in node.users:
            if user in inside:
                # Its internal use is retained, but an external descendant
                # (for example through an already-computed cast) still reads
                # this export and must cross the same genuine published cut.
                pending.append(user)
                continue
            if user not in closure:
                return False
            pending.append(user)
    # An accumulator, nested kwarg or second operand is a distinct original use.
    others = [node for node in dot.all_input_nodes if node is not operand]
    return not any(exports & _ancestors(node) for node in others)


def _prepared_users(
    pipeline: PreparationPipeline,
    operand: Node,
    retained_input: tuple[int, OperandRetentionCandidate] | None = None,
) -> bool:
    """All published-image uses must end at an original preparation boundary."""
    from ...language import memory_ops
    from ...language import view_ops
    from .chained_collectives import classify_collective
    from .chained_preparation_cut import _known_effects

    frame = pipeline.frame
    allowed = set(frame.cut.preparation)
    stops = {buffer.node for buffer in frame.buffers if buffer.node is not None}
    pending, seen = [operand], set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        for user in node.users:
            if (
                user not in allowed
                or not _known_effects(user)
                or user.target
                in (memory_ops.load, memory_ops.store, view_ops.subscript)
                or classify_collective(user) is not None
            ):
                return False
            if user not in stops:
                pending.append(user)
    if any(leaf.node is operand for leaf in pipeline.prepared_leaves):
        return False
    if any(item.buffer.node is operand for item in pipeline.prepared_operands):
        return False
    if any(
        member.buffer.node is operand
        for binding in pipeline.prepared_groups
        for member in binding.candidate.members
    ):
        return False
    retained = pipeline.operand_retention
    return retained is None or all(
        item.node is not operand
        or retained_input is not None
        and index == retained_input[0]
        and item is retained_input[1]
        for index, item in enumerate(retained.candidates)
    )


def _retention_facts(pipeline: PreparationPipeline) -> object:
    retained = pipeline.operand_retention
    if retained is None:
        return None
    return (
        id(retained),
        _freeze(vars(retained)),
        tuple(
            (
                id(item),
                _freeze(vars(item)),
                _freeze(vars(item.group)),
                tuple(_freeze(vars(geometry)) for geometry in item.group.geometries),
                _freeze(vars(item.geometry)),
            )
            for item in retained.candidates
        ),
    )


def _retained_input(
    pipeline: PreparationPipeline,
    stage: PreparationStage,
    operand: Node,
    role: Literal["a", "b"],
    shape: tuple[int, int],
    first: int,
) -> tuple[int, OperandRetentionCandidate] | None:
    """Only the original direct alias of this complete native operand."""
    retained = pipeline.operand_retention
    if retained is None:
        return None
    owner = stage.a if role == "a" else stage.b
    related = tuple(
        (index, item)
        for index, item in enumerate(retained.candidates)
        if item.node is operand or item.owner == owner.name
    )
    if not related:
        return None
    if len(related) != 1:
        raise _UnsupportedChain("island retained operand has multiple native views")
    index, item = related[0]
    if (
        item.node is not operand
        or item.operand is not operand
        or item.view_path
        or item.group != stage.group
        or item.geometry != stage.group.geometries[0]
        or item.role != role
        or item.owner != owner.name
        or item.full_shape != shape
        or item.logical_shape != shape
        or item.logical_modes != (0, 1)
        or item.row_offset != 0
        or item.dtype != operand.meta["val"].dtype
        or item.publication_event != first + 2
        or sum(
            first < other.publication_event <= first + 2
            for other in retained.candidates
        )
        != 1
    ):
        raise _UnsupportedChain(
            "island retained operand is not a direct full-owner alias"
        )
    return index, item


@dataclass(frozen=True)
class IslandConsumerCandidate:
    cg: GenerateAST
    plan: ChainedMatmulPlan
    pipeline: PreparationPipeline
    revision: _Revision
    bound: BoundPreparationIsland
    stage: PreparationStage
    operand: Node
    role: Literal["a", "b"]
    shape: tuple[int, int]
    dtype: torch.dtype
    target: str
    workspace: str
    inputs: tuple[tuple[Node, str], ...]
    reads: tuple[str, ...]
    facts: tuple[object, ...]
    transport_facts: tuple[object, ...]
    boundary_owner: dict[Node, str]
    boundary_identity: int
    settings: Settings
    vector: VectorStaging
    vector_modes: tuple[bool, bool]
    retained_input: tuple[int, OperandRetentionCandidate] | None = None
    retention_facts: object = None
    _selection: tuple[object, ...] = ()

    def _fields(self) -> tuple[object, ...]:
        return tuple(
            value for name, value in vars(self).items() if name != "_selection"
        )

    @property
    def owner(self) -> str:
        return (self.stage.a if self.role == "a" else self.stage.b).name

    @property
    def consumer(self) -> int:
        return self.stage.group.stages[0]

    def matches(self) -> bool:
        from .chained_operand_retention import _InvalidRetention
        from .chained_operand_retention import _revision
        from .chained_preparation_actions import _transport_facts

        try:
            return (
                self._selection == self._fields()
                and self.settings.fast_math is True
                and self.vector_modes
                == (self.vector.enabled, self.vector.group_enabled)
                and id(self.boundary_owner) == self.boundary_identity
                and self.revision
                == _revision(self.plan, self.pipeline.frame, dict(self.revision.shapes))
                and self.bound.frame is self.pipeline.frame
                and self.facts == _context(self.cg, self.plan, self.bound, self.stage)
                and self.transport_facts == _transport_facts(self.pipeline)
                and self.retention_facts == _retention_facts(self.pipeline)
                and self.retained_input
                == _retained_input(
                    self.pipeline,
                    self.stage,
                    self.operand,
                    self.role,
                    self.shape,
                    self.bound.stop_event,
                )
                and _closed_users(
                    self.bound, self.operand, self.plan.dots[self.consumer]
                )
                and _prepared_users(self.pipeline, self.operand, self.retained_input)
            )
        except (_InvalidRetention, _UnsupportedChain):
            return False

    def point(
        self, coordinates: tuple[str, str], values: Mapping[Node, str]
    ) -> list[str]:
        """Lower the original operand; retain every materialized read guard."""
        expression = _Expression(self.cg, self.plan, dict(self.inputs))
        for export in self.bound.exports:
            expression.fragments[export.node] = (
                coordinates,
                _materialized_value(
                    export.name,
                    export.shape,
                    coordinates,
                    "cutlass.Float32",
                    storage_value=values[export.node],
                ),
            )
        value = expression.value(self.operand, coordinates)
        if expression.global_accesses:
            raise _UnsupportedChain("island consumer requires published shared inputs")
        dtype = CompileEnvironment.current().backend.dtype_str(self.dtype)
        domain = _operand_domain(self.cg, self.operand, coordinates, self.plan)
        predicate = " & ".join(
            f"(({coord}) < {size})"
            for coord, size in zip(coordinates, _shape(self.operand), strict=True)
        )
        row, column = coordinates
        return [
            f"if {predicate}:",
            _indent(
                [
                    *expression.lines,
                    f"{self.target}[{row}, {column}] = {_masked_operand(value, dtype, domain)}",
                ]
            ),
            "else:",
            f"    {self.target}[{row}, {column}] = {dtype}(0)",
        ]

    def publication(
        self, prefix: str, registers: Mapping[Node, str]
    ) -> tuple[list[str], list[str]]:
        """Original positive-zero complement followed by original register cells."""
        if not self.matches():
            raise _UnsupportedChain(
                "island consumer selection changed before publication"
            )
        first = self.bound.exports[0]
        coords = self.bound.coordinates(first.node, prefix)
        if any(
            self.bound.coordinates(item.node, prefix) != coords
            for item in self.bound.exports
        ):
            raise _UnsupportedChain(
                "island consumer has different fragment coordinates"
            )
        active = self.point(
            coords,
            {node: f"{name}[{prefix}_index]" for node, name in registers.items()},
        )
        index = f"{prefix}_consumer_zero"
        row, column = f"{index} // {self.shape[1]}", f"{index} % {self.shape[1]}"
        origins = next(
            item.axes for item in self.bound.origins if item.node is first.node
        )
        boxes = [
            f"(({row}) >= {origins[0][0] + i * origins[0][1]}) & (({row}) < {origins[0][0] + i * origins[0][1] + 16}) & (({column}) >= {origins[1][0] + i * origins[1][1]}) & (({column}) < {origins[1][0] + i * origins[1][1] + 16})"
            for i in range(len(self.bound.island.components))
        ]
        zero = self.point(
            (row, column),
            {item.node: "cutlass.Float32(0)" for item in self.bound.exports},
        )
        guard = " | ".join(f"({box})" for box in boxes)
        size = math.prod(self.shape)
        writes = [f"if not ({guard}):", _indent(zero)]
        if size % self.bound.execution.threads:
            writes = [f"if {index} < {size}:", _indent(writes)]
        dtype = CompileEnvironment.current().backend.dtype_str(self.dtype)
        setup = [
            f"{self.target}_ptr = {self.workspace}",
            *_layout(self.target, self.shape, 1, dtype),
            f"for {index}_step in cutlass.range({(size + self.bound.execution.threads - 1) // self.bound.execution.threads}, unroll=1):",
            f"    {index} = {self.bound.execution.thread} + {index}_step * {self.bound.execution.threads}",
            _indent(writes),
        ]
        return setup, [
            f"for {prefix}_index in cutlass.range_constexpr(8):",
            _indent(active),
        ]


def plan_island_consumer(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    pipeline: PreparationPipeline,
    revision: _Revision,
    bound: BoundPreparationIsland,
    boundaries: dict[Node, str],
    vector: VectorStaging,
    *,
    use_frontier: PreparationReadFrontier,
    published: tuple[tuple[Node, str], ...],
) -> IslandConsumerCandidate | None:
    """Conservative singleton original-coordinate fill; no layout substitution."""
    from .chained_preparation_actions import _transport_facts
    from .chained_preparation_actions import workspace_name

    frame = pipeline.frame
    first = bound.stop_event
    if (
        not bound.exports
        or first + 2 > len(frame.actions)
        or frame.actions[first].kind != "fill"
        or frame.actions[first + 1].kind != "mma"
        or frame.actions[first].stages != frame.actions[first + 1].stages
    ):
        return None
    stages = frame.actions[first].stages
    if len(stages) != 1 or stages[0] not in plan.warp_mma_stages:
        return None
    stage = next((item for item in frame.stages if item.group.stages == stages), None)
    if stage is None or stage.shape != warp_mma_shape(
        stage.group.geometries[0], stage.group
    ):
        return None
    dot, geometry = plan.dots[stages[0]], stage.group.geometries[0]
    if dot.args[2] is not None:
        return None
    candidates = []
    for role, shape in (
        ("a", (stage.shape[0], stage.shape[2])),
        ("b", (stage.shape[1], stage.shape[2])),
    ):
        index, coordinates = geometry.operand(role, "row", "column")
        operand = cast("Node", dot.args[index])
        if coordinates != ("row", "column") or not _closed_users(bound, operand, dot):
            continue
        try:
            retained_input = _retained_input(
                pipeline, stage, operand, cast("Literal['a', 'b']", role), shape, first
            )
        except _UnsupportedChain:
            continue
        if not _prepared_users(pipeline, operand, retained_input):
            continue
        if (
            _shape(operand) != shape
            or operand.meta["val"].dtype != plan.operand_dtype(stages[0])
            or operand.meta["val"].dtype not in (torch.float16, torch.bfloat16)
            or any(item.shape != shape for item in bound.exports)
            or any(
                item.node.meta["val"].dtype != torch.float32 for item in bound.exports
            )
            or any(item.node in boundaries for item in bound.exports)
        ):
            continue
        origins = tuple(
            next(item.axes for item in bound.origins if item.node is export.node)
            for export in bound.exports
        )
        if len(set(origins)) != 1:
            continue
        cells = set()
        for component in range(len(bound.island.components)):
            base = tuple(start + component * stride for start, stride in origins[0])
            block = {
                (base[0] + row, base[1] + column)
                for row in range(16)
                for column in range(16)
            }
            if cells & block or any(
                not (0 <= row < shape[0] and 0 <= column < shape[1])
                for row, column in block
            ):
                break
            cells.update(block)
        else:
            owner = stage.a if role == "a" else stage.b
            # Reject native unions and descriptor destinations. The
            # later physical binder still charges this entire standalone owner.
            if (
                owner != frame.layout.region(owner.name)
                or owner.byte_size != math.prod(shape) * 2
            ):
                continue
            try:
                # point() binds these exact exports as register fragments and
                # stops at the first shared boundary on every remaining path.
                # Ancestors behind either cut are not additional shared reads.
                reads = resolved_preparation_reads(
                    pipeline,
                    (operand,),
                    boundaries,
                    fragments=bound,
                    use_frontier=use_frontier,
                    published=published,
                )
            except ValueError:
                continue
            candidate = IslandConsumerCandidate(
                cg,
                plan,
                pipeline,
                revision,
                bound,
                stage,
                operand,
                cast("Literal['a', 'b']", role),
                shape,
                plan.operand_dtype(stages[0]),
                f"chain_{stages[0]}_{role}",
                workspace_name(owner.name),
                tuple(boundaries.items()),
                reads,
                _context(cg, plan, bound, stage),
                _transport_facts(pipeline),
                boundaries,
                id(boundaries),
                CompileEnvironment.current().settings,
                vector,
                (vector.enabled, vector.group_enabled),
                retained_input,
                _retention_facts(pipeline),
            )
            candidates.append(replace(candidate, _selection=candidate._fields()))
    return candidates[0] if len(candidates) == 1 else None


@dataclass(frozen=True)
class CompletedInputIsland:
    """Actual whole register span reading one earlier full-owner publication.

    This is not a completed ordinary stage: the original bound island remains
    the action proof. Only the emitter can capture the alias inside this span;
    the existing recorder and final body must consume these exact lines/maps.
    """

    publication: IslandConsumerPublication
    bound: BoundPreparationIsland
    outputs: tuple[tuple[Node, str], ...]
    body_first: int
    lines: tuple[str, ...]
    alias_first: int
    alias_inputs: tuple[tuple[Node, str], ...]
    alias_outputs: tuple[tuple[Node, str], ...]
    alias_lines: tuple[str, ...]
    context: tuple[object, ...]
    prefix: str
    operand_uses: tuple[tuple[int, str, RegisterImage], ...]
    input_reads: tuple[tuple[int, str, RegisterImage, str], ...]
    issue_bindings: tuple[tuple[int, str, str], ...]
    dtype: str
    _selection: tuple[object, ...] = ()

    def fields(self) -> tuple[object, ...]:
        return tuple(
            value for name, value in vars(self).items() if name != "_selection"
        )

    def current(self) -> bool:
        from .chained_register_binding import _frame_facts
        from .chained_register_binding import _image_facts
        from .chained_register_islands import register_island_matches

        item = self.publication.candidate
        expected = dict(self.bound.published)
        expected[item.operand] = self.publication.boundary_name(self.bound.stop_event)
        expected.update((export.node, export.name) for export in self.bound.exports)
        alias_expected = dict(self.alias_inputs)
        alias_expected[item.operand] = self.publication.boundary_name(
            self.bound.first_event + 2
        )
        return (
            self._selection == self.fields()
            and self.bound.island_input is self.publication
            and self.publication.matches()
            and self.bound.input_identity == id(self.publication)
            and self.bound.plan is item.plan
            and self.bound.frame is item.pipeline.frame
            and self.bound.execution == item.bound.execution
            and self.bound.frame_facts == _frame_facts(self.bound.frame)
            and self.bound.image_facts
            == _image_facts(self.bound.island, self.bound.origins)
            and register_island_matches(
                self.bound.island,
                self.bound.island.revision.region,
                self.bound.island.revision.groups,
                dict(self.bound.island.revision.shapes),
                entry_boundaries=self.bound.island.revision.entry_boundaries,
            )
            and self.bound.first_event == item.bound.stop_event
            and self.context == _context(item.cg, item.plan, self.bound, item.stage)
            and self.outputs == tuple(expected.items())
            and self.alias_inputs == self.bound.published
            and self.alias_outputs == tuple(alias_expected.items())
            and self.alias_lines
            == (
                f"{self.publication.boundary_name(self.bound.first_event + 2)} = {item.target}",
            )
            and type(self.alias_first) is int
            and 0 < self.alias_first < len(self.lines)
            and self.lines[self.alias_first - 1] == self.bound.execution.sync
            and self.lines[self.alias_first : self.alias_first + len(self.alias_lines)]
            == self.alias_lines
            and self.lines[-1] == self.bound.execution.sync
            and self.original_operations_match()
        )

    def original_operations_match(self) -> bool:
        """Join the actual ports/loads and all-role stage cut to original facts."""
        item = self.publication.candidate
        issues = self.bound.island.components[0].issues
        expected = tuple(
            (issue.stage, role, self.bound.operand_image(issue.stage, argument))
            for issue in issues
            for role, argument in zip(
                ("a", "b"), (1, 0) if issue.geometry.transpose else (0, 1), strict=True
            )
        )
        if self.operand_uses != expected or tuple(
            (stage, role, image) for stage, role, image, _ in self.input_reads
        ) != tuple(use for use in expected if use[2].node is item.operand):
            return False
        if len(self.input_reads) != 2:
            return False
        if tuple(stage for stage, _, _ in self.issue_bindings) != tuple(
            issue.stage for issue in issues
        ):
            return False
        try:
            tree = ast.parse("\n".join(self.lines))
            sync = ast.dump(ast.parse(self.bound.execution.sync).body[0])
            alias = ast.dump(ast.parse(self.alias_lines[0]).body[0])
            top = [ast.dump(node) for node in tree.body]
            all_nodes = [ast.dump(node) for node in ast.walk(tree)]
            if (
                top.count(sync) != 4
                or all_nodes.count(sync) != 4
                or top.count(alias) != 1
            ):
                return False
            cut = top.index(alias)
            active = f"{self.bound.execution.thread} < {len(self.bound.island.components) * 32}"
            issue_scopes = [
                (index, node)
                for index, node in enumerate(tree.body)
                if any(
                    isinstance(child, ast.Call)
                    and ast.unparse(child.func) == "cute.gemm"
                    for child in ast.walk(node)
                )
            ]
            actual_calls = tuple(
                ast.unparse(child)
                for node in tree.body
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and ast.unparse(child.func) == "cute.gemm"
            )
            expected_calls = tuple(
                f"cute.gemm({mma}, {dot}_acc, {dot}_a[None, None, 0], {dot}_b[None, None, 0], {dot}_acc)"
                for _, mma, dot in self.issue_bindings
            )
            if actual_calls != expected_calls:
                return False
            if (
                len(issue_scopes) != 2
                or not issue_scopes[0][0] < cut < issue_scopes[1][0]
                or any(
                    not isinstance(node, ast.If) or ast.unparse(node.test) != active
                    for _, node in issue_scopes
                )
            ):
                return False
            for ordinal, (_, _, image, target) in enumerate(self.input_reads):
                owner = self.publication.boundary_name(
                    self.bound.first_event + ordinal * 2
                )
                read = _materialized_value(
                    owner,
                    item.shape,
                    self.bound.coordinates(image, self.prefix),
                    self.dtype,
                )
                statement = ast.dump(
                    ast.parse(f"{target}[{self.prefix}_index] = {read}").body[0]
                )
                locations = [
                    index
                    for index, node in enumerate(tree.body)
                    if any(ast.dump(child) == statement for child in ast.walk(node))
                ]
                if len(locations) != 1 or not (
                    locations[0] < cut if ordinal == 0 else locations[0] > cut
                ):
                    return False
        except (SyntaxError, _UnsupportedChain):
            return False
        return True

    def matches(
        self, accepted: AcceptedPreparation, action: AcceptedPreparationAction
    ) -> bool:
        item = self.publication.candidate
        return (
            self.current()
            and accepted.pipeline is item.pipeline
            and accepted.revision == item.revision
            and action.kind == "island"
            and action.proof is self.bound
            and (action.first, action.stop)
            == (self.bound.first_event, self.bound.stop_event)
            and action.inputs == self.bound.published
            and action.outputs == self.outputs
            and item.owner in action.reads
            and item.owner not in action.omitted
            and self.body_first >= 0
            and accepted.lines[self.body_first : self.body_first + len(self.lines)]
            == self.lines
        )


@dataclass(frozen=True)
class RetainedIslandAlias:
    """The original alias postlude, captured before its emitter returns."""

    completion: CompletedPreparationStage | CompletedInputIsland
    inputs: tuple[tuple[Node, str], ...]
    outputs: tuple[tuple[Node, str], ...]
    lines: tuple[str, ...]


@dataclass
class IslandConsumerPublication:
    """One actual publication, not a replacement Node or mutable boundary."""

    candidate: IslandConsumerCandidate
    body_first: int
    lines: tuple[str, ...] = ()
    consumed: bool = False
    _published: tuple[object, ...] | None = None
    _consumption: tuple[object, ...] | None = None
    retained_alias: RetainedIslandAlias | None = None
    _retained_alias: RetainedIslandAlias | None = None
    _retained_facts: tuple[object, ...] | None = None
    island_completion: CompletedInputIsland | None = None
    _island_completion: CompletedInputIsland | None = None

    def record_island(
        self,
        completion: CompletedInputIsland,
        boundaries: dict[Node, str],
    ) -> None:
        item = self.candidate
        if (
            not self.matches()
            or self.consumed
            or self.island_completion is not None
            or self.retained_alias is not None
            or boundaries is not item.boundary_owner
            or item.retained_input is None
            or completion.publication is not self
            or completion.bound.published != self.outputs
            or completion.outputs != tuple(boundaries.items())
        ):
            raise _UnsupportedChain("island input completion changed or duplicated")
        if not completion.current():
            raise _UnsupportedChain("island input lacks whole-span completion")
        alias = RetainedIslandAlias(
            completion,
            completion.alias_inputs,
            completion.alias_outputs,
            completion.alias_lines,
        )
        self.island_completion = self._island_completion = completion
        self.retained_alias = self._retained_alias = alias
        self._retained_facts = (completion, alias.inputs, alias.outputs, alias.lines)
        self.consumed = True
        self._consumption = self._published

    def grouped_operands(
        self, operands: tuple[VectorStageOperand, ...]
    ) -> tuple[VectorStageOperand, ...]:
        """Exclude only the proved role from the ORIGINAL operand-group walk."""
        item = self.candidate
        geometry = item.stage.group.geometries[0]
        m, n, k = item.stage.shape
        if not self.matches() or self.consumed or len(operands) != 2:
            raise _UnsupportedChain("island grouped operands changed before emission")
        for operand, role, shape in zip(
            operands, ("a", "b"), ((m, k), (n, k)), strict=True
        ):
            index, _ = geometry.operand(role, "row", "column")
            if (
                operand.node is not item.plan.dots[item.consumer].args[index]
                or operand.geometry != geometry
                or operand.role != role
                or operand.shape != shape
                or operand.target != f"chain_{item.consumer}_{role}"
                or operand.offset != 0
            ):
                raise _UnsupportedChain("island grouped operand role/layout changed")
        return tuple(operand for operand in operands if operand.role != item.role)

    def boundary_name(self, event: int) -> str:
        retained = self.candidate.retained_input
        if retained is not None and event >= retained[1].publication_event:
            return f"chain_retained_operand_{retained[0]}"
        return self.candidate.owner

    def record_retained(
        self,
        completion: CompletedPreparationStage,
        before: tuple[tuple[Node, str], ...],
        boundaries: dict[Node, str],
        lines: tuple[str, ...],
    ) -> None:
        item = self.candidate
        expected = dict(before)
        expected[item.operand] = self.boundary_name(completion.stop)
        if (
            not self.matches()
            or not self.consumed
            or self.retained_alias is not None
            or item.retained_input is None
            or completion.island_input is not self
            or completion.first != item.bound.stop_event
            or not completion.original_warp_matches()
            or completion.outputs != before
            or boundaries is not item.boundary_owner
            or tuple(boundaries.items()) != tuple(expected.items())
            or lines != (f"{self.boundary_name(completion.stop)} = {item.target}",)
        ):
            raise _UnsupportedChain(
                "island retained alias lacks original stage completion"
            )
        alias = RetainedIslandAlias(
            completion, before, tuple(boundaries.items()), lines
        )
        self.retained_alias = self._retained_alias = alias
        self._retained_facts = (completion, alias.inputs, alias.outputs, alias.lines)

    def stage_outputs_match(
        self, completion: CompletedPreparationStage, action: AcceptedPreparationAction
    ) -> bool:
        if self.candidate.retained_input is None:
            return all(
                dict(action.outputs).get(node) == name
                for node, name in completion.outputs
            )
        alias = self.retained_alias
        return (
            alias is not None
            and alias is self._retained_alias
            and alias.completion is completion
            and completion.original_warp_matches()
            and alias.inputs == completion.outputs
            and alias.outputs == action.outputs
            and alias.lines
            == (f"{self.boundary_name(action.stop)} = {self.candidate.target}",)
        )

    def retained_segment_matches(
        self, action: AcceptedPreparationAction, emitted: tuple[str, ...]
    ) -> bool:
        from .chained_preparation_reads import CompletedPreparationStage

        alias = self.retained_alias
        if (
            self.candidate.retained_input is None
            or action.first != self.candidate.bound.stop_event
        ):
            return True
        if self.island_completion is not None:
            completion = self.island_completion
            return (
                completion is self._island_completion
                and completion.current()
                and action.proof is completion.bound
                and emitted == completion.lines
                and action.outputs == completion.outputs
                and alias is not None
                and alias.completion is completion
            )
        return (
            alias is not None
            and isinstance(alias.completion, CompletedPreparationStage)
            and action.proof is alias.completion
            and self.stage_outputs_match(alias.completion, action)
            and emitted == (*alias.completion.lines, *alias.lines)
        )

    @property
    def outputs(self) -> tuple[tuple[Node, str], ...]:
        return (*self.candidate.inputs, (self.candidate.operand, self.candidate.owner))

    def record(self, lines: list[str], boundaries: dict[Node, str]) -> None:
        if (
            self._published is not None
            or self.consumed
            or not self.candidate.matches()
            or boundaries is not self.candidate.boundary_owner
            or tuple(boundaries.items()) != self.candidate.inputs
        ):
            raise _UnsupportedChain("island publication changed or duplicated")
        lines.append(f"{self.candidate.owner} = {self.candidate.target}")
        self.lines = tuple(lines)
        self._published = (self.candidate, self.body_first, self.lines)
        boundaries[self.candidate.operand] = self.candidate.owner

    def matches(self) -> bool:
        return (
            self._published == (self.candidate, self.body_first, self.lines)
            and bool(self.lines)
            and self.candidate.matches()
            and self.retained_alias is self._retained_alias
            and self.island_completion is self._island_completion
            and (
                self.retained_alias is None
                and self._retained_facts is None
                or self.retained_alias is not None
                and self._retained_facts
                == (
                    self.retained_alias.completion,
                    self.retained_alias.inputs,
                    self.retained_alias.outputs,
                    self.retained_alias.lines,
                )
            )
            and (
                (self.consumed is False and self._consumption is None)
                or (self.consumed is True and self._consumption == self._published)
            )
        )

    def validate_stage(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: Mapping[Node, str],
        stage: int,
        geometry: StageGeometry,
        execution: ChainedExecution,
        vector: VectorStaging | None,
    ) -> None:
        item = self.candidate
        expected = replace(
            item.bound.execution,
            a_workspace=f"chain_preparation_workspace_{item.stage.a.name}",
            b_workspace=f"chain_preparation_workspace_{item.stage.b.name}",
        )
        if (
            not self.matches()
            or self.consumed
            or cg is not item.cg
            or plan is not item.plan
            or stage != item.consumer
            or geometry != item.stage.group.geometries[0]
            or execution != expected
            or boundaries is not item.boundary_owner
            or tuple(boundaries.items()) != self.outputs
            or vector is not item.vector
        ):
            raise _UnsupportedChain(
                "island input publication does not match this stage"
            )

    def consume(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: Mapping[Node, str],
        stage: int,
        geometry: StageGeometry,
        execution: ChainedExecution,
        vector: VectorStaging | None,
        role: str,
        shape: tuple[int, int],
        dtype: str,
    ) -> None:
        self.validate_stage(cg, plan, boundaries, stage, geometry, execution, vector)
        if (
            role != self.candidate.role
            or shape != self.candidate.shape
            or dtype
            != CompileEnvironment.current().backend.dtype_str(self.candidate.dtype)
        ):
            raise _UnsupportedChain("island input role/layout/dtype changed")
        self.consumed = True
        self._consumption = self._published

    def accepted(
        self, accepted: AcceptedPreparation, action: AcceptedPreparationAction
    ) -> bool:
        from .chained_preparation_reads import CompletedPreparationStage

        item = self.candidate
        consumers = tuple(
            other
            for other in accepted.actions
            if (
                isinstance(other.proof, CompletedPreparationStage)
                and other.proof.island_input is self
            )
            or self.island_completion is not None
            and other.proof is self.island_completion.bound
        )
        readers = tuple(
            reader for reader in accepted.island_reads if reader.publication is self
        )
        actual = tuple(
            other
            for other in accepted.actions
            if other.first >= action.stop
            and item.owner in other.reads
            and accepted.pipeline.frame.actions[other.first].kind != "ready"
        )
        return (
            self.matches()
            and self.consumed is True
            and accepted.revision == item.revision
            and accepted.pipeline is item.pipeline
            and action.proof is item.bound
            and action.inputs == item.inputs
            and action.outputs == self.outputs
            and action.kind == "island"
            and action.first == item.bound.first_event
            and action.stop == item.bound.stop_event
            and action.writes == (item.owner,)
            and accepted.lines[self.body_first : self.body_first + len(self.lines)]
            == self.lines
            and len(consumers) == 1
            and (
                self.island_completion.matches(accepted, consumers[0])
                if self.island_completion is not None
                else isinstance(consumers[0].proof, CompletedPreparationStage)
                and consumers[0].proof.matches(accepted, consumers[0])
            )
            and tuple(reader.action for reader in readers) == actual
            and bool(readers)
            and all(reader.matches(accepted) for reader in readers)
            and all(
                not ({export.name for export in item.bound.exports} & set(other.reads))
                and (
                    other.first < action.stop
                    or (
                        dict(other.inputs).get(item.operand)
                        == self.boundary_name(other.first)
                        and dict(other.outputs).get(item.operand)
                        == self.boundary_name(other.stop)
                        and all(
                            node is item.operand
                            for node, name in (*other.inputs, *other.outputs)
                            if name in (item.owner, self.boundary_name(other.stop))
                        )
                    )
                )
                for other in accepted.actions
            )
        )


@dataclass(frozen=True)
class IslandPublicationRead:
    """Actual successful existing fill/frontier segment, not an async enqueue."""

    publication: IslandConsumerPublication
    action: AcceptedPreparationAction
    body_first: int
    lines: tuple[str, ...]
    facts: tuple[object, ...]
    frontier: CompletedPreparationFrontier | None = None

    def matches(self, accepted: AcceptedPreparation) -> bool:
        from .chained_frontier_groups import FrontierGroup
        from .chained_frontier_groups import plan_frontier_group
        from .chained_preparation_reads import CompletedPreparationStage

        action, item = self.action, self.publication.candidate
        if (
            self.facts != _read_facts(action)
            or not any(action is other for other in accepted.actions)
            or item.owner not in action.reads
            or not self.lines
            or type(self.body_first) is not int
            or self.body_first < 0
            or accepted.lines[self.body_first : self.body_first + len(self.lines)]
            != self.lines
            or not self.publication.retained_segment_matches(action, self.lines)
        ):
            return False
        if isinstance(action.proof, CompletedPreparationStage):
            return (
                self.frontier is None
                and action.proof.original_warp_matches()
                and action.proof.matches(accepted, action)
            )
        completion = self.publication.island_completion
        if completion is not None and action.proof is completion.bound:
            return self.frontier is None and completion.matches(accepted, action)
        frame = accepted.pipeline.frame
        frontier = self.frontier
        if (
            frontier is None
            or not frontier.matches(
                accepted.revision.plan,
                action.inputs,
                action.outputs,
                self.lines,
                consumed=True,
            )
            or frontier.execution != accepted.execution
            or tuple(buffer.name for buffer in frontier.buffers) != action.writes
        ):
            return False
        if action.kind == "frontier":
            return (
                isinstance(action.proof, FrontierGroup)
                and action.proof == plan_frontier_group(frame, action.first)
                and action.stop == action.proof.stop_event
            )
        return (
            action.kind == "ordinary"
            and action.proof is None
            and action.stop == action.first + 1
            and frame.actions[action.first].kind == "frontier"
        )


def _read_facts(action: AcceptedPreparationAction) -> tuple[object, ...]:
    from .chained_preparation_reads import CompletedPreparationStage

    return (
        action.first,
        action.stop,
        action.kind,
        action.proof,
        action.inputs,
        action.outputs,
        action.reads,
        action.writes,
        action.omitted,
        action.scan_transfer,
        action.broadcasts,
        (
            _freeze(vars(action.proof.execution)),
            _freeze(vars(action.proof.stage)),
        )
        if isinstance(action.proof, CompletedPreparationStage)
        else (),
    )


def record_island_read(
    publication: IslandConsumerPublication,
    action: AcceptedPreparationAction,
    body_first: int | None,
    emitted: tuple[str, ...],
    *,
    frontier: CompletedPreparationFrontier | None = None,
) -> IslandPublicationRead:
    from .chained_frontier_groups import FrontierGroup
    from .chained_preparation_reads import CompletedPreparationStage

    item = publication.candidate
    if (
        not publication.matches()
        or not publication.consumed
        or body_first is None
        or not emitted
        or item.owner not in action.reads
        or dict(action.inputs).get(item.operand)
        != publication.boundary_name(action.first)
        or dict(action.outputs).get(item.operand)
        != publication.boundary_name(action.stop)
        or not publication.retained_segment_matches(action, emitted)
        or not (
            isinstance(action.proof, CompletedPreparationStage)
            and action.proof.original_warp_matches()
            and frontier is None
            or publication.island_completion is not None
            and action.proof is publication.island_completion.bound
            and publication.island_completion.current()
            and emitted == publication.island_completion.lines
            and frontier is None
            or frontier is not None
            and frontier.matches(
                item.revision.plan,
                action.inputs,
                action.outputs,
                emitted,
                consumed=True,
            )
            and (
                isinstance(action.proof, FrontierGroup)
                or action.kind == "ordinary"
                and action.proof is None
                and item.pipeline.frame.actions[action.first].kind == "frontier"
            )
        )
    ):
        raise _UnsupportedChain("island publication has an unproved reader completion")
    return IslandPublicationRead(
        publication, action, body_first, emitted, _read_facts(action), frontier
    )


def validate_island_physical(physical: AcceptedPreparationStorage) -> bool:
    """Revalidate full, disjoint native owners over actual publication spans."""
    from .chained_preparation_transfers import bind_preparation_transfer_span
    from .chained_preparation_transfers import preparation_transfer_owner

    accepted = physical.accepted
    scan = accepted.pipeline.scan_producer
    phases = (
        scan.phases
        if scan is not None
        else tuple(range(len(accepted.pipeline.frame.actions)))
    )
    for action in accepted.actions:
        publication = action.island_publication
        if publication is None:
            continue
        item = publication.candidate
        first = phases[action.first]
        stop = max(phases[index] for index in range(action.first, action.stop)) + 1
        readers = tuple(
            reader
            for reader in accepted.island_reads
            if reader.publication is publication
        )
        if not readers:
            return False
        consumer_stop = (
            max(
                phases[index]
                for reader in readers
                for index in range(reader.action.first, reader.action.stop)
            )
            + 1
        )
        transfer = bind_preparation_transfer_span(
            physical, first, stop, action.reads, (item.owner,)
        )
        owner = preparation_transfer_owner(physical, item.owner, first, consumer_stop)
        views = tuple(
            view for view in physical.views if view.original.semantic.name == item.owner
        )
        if transfer is None or owner is None or len(views) != 1:
            return False
        view = views[0]
        if (
            view.owner != item.owner
            or view.crop is not None
            or view.original.native_group is not None
            or view.original.semantic.kind != item.role
            or view.original.shape != item.shape
            or view.original.member_byte_offset != 0
            or view.byte_offset != owner.byte_offset
            or view.declared_bytes != math.prod(item.shape) * item.dtype.itemsize
            or owner.byte_size != view.declared_bytes
            or view.accesses != ((owner.byte_offset, owner.byte_size),)
        ):
            return False
        if any(export.name not in physical.omitted for export in item.bound.exports):
            return False
    return True
