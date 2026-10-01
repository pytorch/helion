"""Original accumulator relations and closed prepared body actions.

These records select neither storage nor a numerical reassociation. An explicit
accumulator edge and an already-admitted transformed seed are different proofs.
The original stage/host adapter must bind their actual owners and readiness
before BodyProgram lowers an instruction payload. Emitting that payload does
not complete an asynchronous operation or publish a logical C boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import IntEnum
from typing import TYPE_CHECKING
from typing import cast

import torch

from . import chained_matmul as chain
from .chained_completed_store import _records
from .prepared_graph_schedule import ContractionGraph
from .prepared_graph_schedule import IssuePlacement
from .prepared_graph_schedule import order_issue_placements

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_body_program import BodyProgram
    from .chained_body_program import RootActionBody
    from .chained_initialized_accumulator import InitializedAccumulator
    from .chained_root_stage import RootStageAction
    from .chunk_recurrence import CuteChunkRecurrencePlan
    from .contraction_region import ContractionRegion
    from .contraction_region import ContractionSpec
    from .prepared_tcgen_binding import PreparedProjectionHost


class ContinuationOpcode(IntEnum):
    WAIT = 0
    WAIT_TOGGLE = 1
    RELEASE = 3
    ISSUE = 4
    PREVIOUS = 5
    COMMIT = 6


@dataclass(frozen=True)
class PreparedContinuation:
    """Two original dots and their existing, explicit numerical authority."""

    region: ContractionRegion
    first: ContractionSpec
    second: ContractionSpec
    transformed: InitializedAccumulator | None = None
    graph: ContractionGraph = field(init=False, repr=False, compare=False)
    _revision: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._validate_relation()
        object.__setattr__(
            self,
            "graph",
            ContractionGraph(
                self.region, () if self.transformed is None else (self.transformed,)
            ),
        )
        object.__setattr__(self, "_revision", self._current())

    @property
    def specs(self) -> tuple[ContractionSpec, ContractionSpec]:
        return self.first, self.second

    def _validate_relation(self) -> None:
        first, second = self.specs
        if (
            first is second
            or not any(item is first for item in self.region.contractions)
            or not any(item is second for item in self.region.contractions)
            or self.region.nodes.index(first.node)
            >= self.region.nodes.index(second.node)
            or first.result_dtype is not torch.float32
            or second.result_dtype is not torch.float32
            or first.accumulator is not None
        ):
            raise chain._UnsupportedChain("invalid prepared continuation relation")
        if self.transformed is None:
            if (
                second.accumulator is not first.node
                or second.accumulator_dtype is not torch.float32
                or first.result_domain != second.result_domain
            ):
                raise chain._UnsupportedChain("missing explicit accumulator edge")
        elif (
            self.transformed.first is not first.node
            or self.transformed.second is not second.node
            or second.accumulator is not None
            or self.transformed.seed not in self.region.nodes
            or self.transformed.join not in self.region.nodes
        ):
            raise chain._UnsupportedChain("foreign transformed accumulator proof")

    def _current(self) -> object:
        return (
            id(self.region),
            id(self.region.graph),
            _records((self.specs, self.transformed)),
            tuple(
                (
                    node,
                    node.op,
                    node.target,
                    _records(node.args),
                    _records(node.kwargs),
                    _records(node.meta),
                    tuple(node.users),
                )
                for node in self.region.graph.nodes
            ),
        )

    def check(self) -> None:
        self._validate_relation()
        self.graph.check()
        if self.graph.region is not self.region or self.graph.transformed != (
            () if self.transformed is None else (self.transformed,)
        ):
            raise chain._UnsupportedChain("foreign continuation dependency graph")
        if self._current() != self._revision:
            raise chain._UnsupportedChain("prepared continuation graph changed")


def continuation_for_nodes(
    region: ContractionRegion,
    first: Node,
    second: Node,
    *,
    transformed: InitializedAccumulator | None = None,
) -> PreparedContinuation:
    """Use existing original specs, never synthesize dots or tensor domains."""
    selected = tuple(
        tuple(spec for spec in region.contractions if spec.node is node)
        for node in (first, second)
    )
    if any(len(items) != 1 for items in selected):
        raise chain._UnsupportedChain("continuation contraction is not in region")
    return PreparedContinuation(region, selected[0][0], selected[1][0], transformed)


@dataclass(frozen=True)
class ContinuationWait:
    """An already-bound event port; phase generation is owned by its adapter."""

    port: int
    toggle: bool = False
    fence_after: bool = False


@dataclass(frozen=True)
class ContinuationRelease:
    port: int


@dataclass(frozen=True)
class ContinuationCommit:
    """Original final commit when the admitted split-K spans overlap."""

    port: int


@dataclass(frozen=True)
class ContinuationIssue:
    continuation: PreparedContinuation
    ordinal: int
    port: int
    k_begin: int
    k_end: int
    initialized: bool
    commit: bool
    wait_after: bool

    def check(self) -> None:
        self.continuation.check()
        if (
            type(self.ordinal) is not int
            or self.ordinal not in (0, 1)
            or type(self.port) is not int
            or self.port < 0
            or type(self.k_begin) is not int
            or type(self.k_end) is not int
            or not 0 <= self.k_begin < self.k_end
            or type(self.initialized) is not bool
            or type(self.commit) is not bool
            or type(self.wait_after) is not bool
            or self.wait_after
            and not self.commit
            or self.initialized is not (self.ordinal == 1)
        ):
            raise chain._UnsupportedChain("invalid prepared continuation issue")


@dataclass(frozen=True)
class ContractionIssue:
    """One native interval of an actual spec under an enclosing owner.

    This validates the common instruction payload, not prefix completion or
    transformed-accumulator authority. Those remain with the original root
    continuation or segmented epoch that constructed this action.
    """

    region: ContractionRegion
    spec: ContractionSpec
    begin: int
    end: int
    atom_k: int
    initialized: bool
    commit: bool
    wait_after: bool = False

    def check(self) -> None:
        if (
            not any(spec is self.spec for spec in self.region.contractions)
            or self.spec.result_dtype is not torch.float32
            or any(
                type(value) is not int for value in (self.begin, self.end, self.atom_k)
            )
            or self.atom_k <= 0
            or not 0 <= self.begin < self.end
            or any(
                type(value) is not bool
                for value in (self.initialized, self.commit, self.wait_after)
            )
            or self.wait_after
            and not self.commit
        ):
            raise chain._UnsupportedChain("invalid bound contraction issue")
        values = tuple(node.meta.get("val") for node in (self.spec.lhs, self.spec.rhs))
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise chain._UnsupportedChain("missing bound contraction geometry")
        lhs, rhs = (chain._host_shape(cast("torch.Tensor", value)) for value in values)
        if (
            len(lhs) != 2
            or len(rhs) != 2
            or lhs[1] != rhs[0]
            or lhs[1] < self.end * self.atom_k
        ):
            raise chain._UnsupportedChain("bound contraction issue geometry changed")

    def payload(self, port: int) -> tuple[object, ...]:
        self.check()
        return (
            int(ContinuationOpcode.ISSUE),
            port,
            self.begin,
            self.end,
            self.initialized,
            self.commit,
            self.wait_after,
        )


@dataclass(frozen=True)
class ContinuationPrevious:
    """Original previous-iteration retirement, with no arbitrary predicate."""

    actions: tuple[ContinuationWait | ContinuationRelease, ...]


ContinuationAction = (
    ContinuationWait
    | ContinuationRelease
    | ContinuationCommit
    | ContinuationIssue
    | ContinuationPrevious
)


def action_facts(action: ContinuationAction) -> tuple[object, ...]:
    if isinstance(action, ContinuationIssue):
        return (
            id(action),
            id(action.continuation),
            action.ordinal,
            action.port,
            action.k_begin,
            action.k_end,
            action.initialized,
            action.commit,
            action.wait_after,
        )
    if isinstance(action, ContinuationWait):
        return id(action), action.port, action.toggle, action.fence_after
    if isinstance(action, (ContinuationRelease, ContinuationCommit)):
        return id(action), action.port
    return id(action), tuple(action_facts(item) for item in action.actions)


@dataclass
class PreparedBodyLowering:
    """Pure payload lowering under an existing body/host's actual authority.

    This has no readiness state or successful-completion token. Its owner is
    either the original RootActionBody (including its actual pending action)
    or the original matched external host. Those owners continue to validate
    physical layouts, input waits, transformed seeds, and whole-body lifetime.
    """

    codegen: GenerateAST
    owner: RootActionBody | PreparedProjectionHost
    continuation: PreparedContinuation
    program: BodyProgram
    event_count: int
    issue_ranges: tuple[tuple[int, int], ...]
    root_action: RootStageAction | None = None
    _actions: object = field(init=False, repr=False)
    _seed: object = field(init=False, repr=False)
    _context: object = field(init=False, repr=False)
    _payload: list[tuple[object, ...]] = field(
        default_factory=list, init=False, repr=False
    )
    _finished: bool = field(default=False, init=False)

    def actions(self) -> tuple[ContinuationAction, ...]:
        result = []
        for action in self.program.actions:
            if not isinstance(action, ContinuationAction):
                raise chain._UnsupportedChain("foreign prepared body action")
            result.append(action)
        return tuple(result)

    def __post_init__(self) -> None:
        self._actions = tuple(action_facts(a) for a in self.actions())
        self._seed = (
            None
            if self.root_action is None
            else self.root_action.sequence.seed_completion
        )
        self._context = self._context_facts()
        self.check()

    def _context_facts(self) -> tuple[object, ...]:
        return (
            id(self.codegen),
            id(self.owner),
            id(self.continuation),
            id(self.program),
            self.event_count,
            self.issue_ranges,
            id(self.root_action),
            id(self._seed),
        )

    def check(self) -> None:
        from .chained_body_program import RootActionBody
        from .prepared_tcgen_binding import PreparedProjectionHost

        self.continuation.check()
        if (
            self._context_facts() != self._context
            or tuple(action_facts(a) for a in self.actions()) != self._actions
            or type(self.event_count) is not int
            or self.event_count < 0
        ):
            raise chain._UnsupportedChain("prepared body actions changed")
        if isinstance(self.owner, RootActionBody):
            self.owner.check()
            action = self.root_action
            if (
                action is None
                or self.owner.codegen is not self.codegen
                or self.owner.pending is not action
                or self.continuation is not self.owner.continuation
                or action is not self.owner.actions[action.stage]
                or self.continuation.region is not self.owner.plan.region
                or self.continuation.transformed
                is not self.owner.plan.initialized_accumulator
                or action.sequence.seed_completion is not self._seed
                or action.stage == 1
                and self._seed is None
            ):
                raise chain._UnsupportedChain("foreign prepared root action")
            k = self.owner.plan.shapes[action.stage][2]
            divisor = 32 if action.k_schedule is not None else 16
            if k <= 0 or k % divisor or self.issue_ranges != ((0, k // divisor),):
                raise chain._UnsupportedChain("original root issue range changed")
        elif isinstance(self.owner, PreparedProjectionHost):
            self.owner.check()
            matched = self.owner.plan.prepared_projection
            if (
                self.root_action is not None
                or self.continuation is not self.owner.plan.prepared_continuation
                or matched is None
                or self.continuation.transformed is not None
                or self.continuation.region.graph is not matched.loop.graph
                or self.codegen.current_root_graph_info is not matched.root
            ):
                raise chain._UnsupportedChain("foreign prepared external body")
            extents = tuple(
                chain._shape(spec.lhs)[1] for spec in self.continuation.specs
            )
            if any(not isinstance(k, int) or k <= 0 or k % 16 for k in extents):
                raise chain._UnsupportedChain("unresolved original issue range")
            if self.issue_ranges != tuple((0, k // 16) for k in extents):
                raise chain._UnsupportedChain("original external issue range changed")
        else:
            raise chain._UnsupportedChain("unknown prepared body owner")

    def _encode(self, action: ContinuationAction) -> tuple[object, ...]:
        if isinstance(action, ContinuationPrevious):
            if self.root_action is not None or any(
                not isinstance(item, (ContinuationWait, ContinuationRelease))
                or isinstance(item, ContinuationWait)
                and (item.toggle or item.fence_after)
                for item in action.actions
            ):
                raise chain._UnsupportedChain("invalid previous-iteration action")
            return int(ContinuationOpcode.PREVIOUS), tuple(
                self._encode(a) for a in action.actions
            )
        if isinstance(action, ContinuationIssue):
            action.check()
            if (
                action.continuation is not self.continuation
                or not 0 <= action.port < len(self.issue_ranges)
                or (action.k_begin, action.k_end) != self.issue_ranges[action.port]
                or self.root_action is not None
                and action.ordinal != self.root_action.stage
            ):
                raise chain._UnsupportedChain("prepared issue port changed")
            atom_k = (
                32
                if self.root_action is not None
                and self.root_action.k_schedule is not None
                else 16
            )
            return ContractionIssue(
                self.continuation.region,
                self.continuation.specs[action.ordinal],
                action.k_begin,
                action.k_end,
                atom_k,
                action.initialized,
                action.commit,
                action.wait_after,
            ).payload(action.port)
        if type(action.port) is not int or not 0 <= action.port < self.event_count:
            raise chain._UnsupportedChain("prepared event port changed")
        if isinstance(action, ContinuationWait):
            if type(action.toggle) is not bool or type(action.fence_after) is not bool:
                raise chain._UnsupportedChain("prepared wait policy changed")
            return (
                int(
                    ContinuationOpcode.WAIT_TOGGLE
                    if action.toggle
                    else ContinuationOpcode.WAIT
                ),
                action.port,
                action.fence_after,
            )
        return (
            int(
                ContinuationOpcode.RELEASE
                if isinstance(action, ContinuationRelease)
                else ContinuationOpcode.COMMIT
            ),
            action.port,
            False,
        )

    def lower(self, action: ContinuationAction) -> None:
        self.check()
        if self._finished or action is not self.program.actions[len(self._payload)]:
            raise chain._UnsupportedChain(
                "prepared action lowering repeated or reordered"
            )
        self._payload.append(self._encode(action))

    def finish(self) -> list[str]:
        self.check()
        if self._finished or len(self._payload) != len(self.program.actions):
            raise chain._UnsupportedChain("incomplete prepared body lowering")
        self._finished = True
        # The typed constexpr payload is consumed directly by the original
        # host/stage. No dummy device statement or synthetic frame is emitted.
        return []

    @property
    def payload(self) -> tuple[tuple[object, ...], ...]:
        self.check()
        if not self._finished:
            raise chain._UnsupportedChain("prepared body payload is not lowered")
        expected = tuple(self._encode(a) for a in self.actions())
        if tuple(self._payload) != expected:
            raise chain._UnsupportedChain("prepared body payload changed")
        return expected


def root_issue_schedule(body: RootActionBody) -> tuple[IssuePlacement, ...]:
    """Bind complete original full/half intervals, then let graph edges order them."""
    model = body.continuation
    if model is None:
        raise chain._UnsupportedChain("unselected root issue schedule")
    bindings = []
    by_node = {spec.node: spec for spec in model.region.contractions}
    for action in reversed(body.actions):
        spec = by_node[body.plan.dots[action.stage]]
        k = body.plan.shapes[action.stage][2]
        split = action.k_schedule
        count = k // 16
        intervals = (
            ((0, count),)
            if split is None
            else (
                (0, count // 2),
                (count // 2, count),
            )
        )
        retire = split is None or split.mode == "serial64" or action.retire_each_half
        for begin, end in reversed(intervals):
            bindings.append(
                IssuePlacement(
                    ContractionIssue(
                        model.region,
                        spec,
                        begin,
                        end,
                        16,
                        begin != 0 or action.seeded_accumulator,
                        retire,
                        retire,
                    ),
                    body,
                    body.execution,
                    (0, body.plan.shapes[action.stage][1]),
                )
            )
    roles = order_issue_placements(model.graph, tuple(bindings))
    if len(roles) != 1:
        raise chain._UnsupportedChain("root issue lost original execution role")
    return roles[0]


def emit_root_continuation(
    body: RootActionBody,
    action: RootStageAction,
    prefix: str,
    *,
    final_commit: bool = False,
) -> list[str]:
    """Lower an original full/half issue at its original root-stage position."""
    from .chained_body_program import BodyProgram
    from .chained_body_program import emit_body_program

    model = body.continuation
    if model is None or body.pending is not action:
        raise chain._UnsupportedChain("unselected prepared root continuation")
    split = action.k_schedule
    scheduled = tuple(
        item.issue
        for item in root_issue_schedule(body)
        if item.issue.spec.node is body.plan.dots[action.stage]
    )
    if not scheduled or len(scheduled) != (1 if split is None else 2):
        raise chain._UnsupportedChain("root issue lost complete original intervals")
    count = scheduled[0].end - scheduled[0].begin
    if any(item.end - item.begin != count for item in scheduled):
        raise chain._UnsupportedChain("root issue changed original half geometry")
    if final_commit:
        if split is None or split.mode != "overlap64" or action.retire_each_half:
            raise chain._UnsupportedChain("foreign final overlap completion")
        actions: tuple[ContinuationAction, ...] = (
            ContinuationCommit(0),
            ContinuationWait(0),
        )
    else:
        actions = (
            ContinuationIssue(
                model,
                action.stage,
                0,
                0,
                count,
                scheduled[0].initialized,
                scheduled[0].commit,
                scheduled[0].wait_after,
            ),
        )
    lowering = PreparedBodyLowering(
        body.codegen,
        body,
        model,
        BodyProgram(actions),
        1,
        ((0, count),),
        action,
    )
    emit_body_program(
        body.codegen,
        body.plan,
        None,
        body.execution,
        None,
        None,
        None,
        prepared_body=lowering,
    )
    phase = "chain_k_half" if split is not None and not final_commit else "0"
    k_base = f"chain_k_half * {count}" if split is not None else "0"
    tmem_a = action.tmem_input is not None
    port = (
        f"({prefix}_ra, {prefix}_rb, {prefix}_acc, {prefix}_mma, "
        f"chain_bars + {action.stage}, {phase}, {tmem_a!r}, {k_base}, 16, (), 128)"
    )
    return [
        *(
            [
                f"{prefix}_rb = {prefix}_mma.make_fragment_B({prefix}_slice.partition_B({prefix}_b))"
            ]
            if split is None
            else []
        ),
        "from helion._compiler.cute.prepared_tcgen_edge import execute_prepared_continuation",
        (
            f"execute_prepared_continuation({lowering.payload!r}, ({port},), "
            f"(chain_bars + {action.stage},), ({phase},), 0, chain_warp == 0, False)"
        ),
    ]


def lower_external_continuation(
    cg: GenerateAST, plan: CuteChunkRecurrencePlan
) -> tuple[tuple[object, ...], ...]:
    """Original issuer event ports, with K extents from its original dot specs.

    Ports are, in order: prior accumulator, prior raw slot, current raw slot,
    packed state, output slot, and packed residual. Their actual pointers and
    generations stay in the existing outer issuer; no frame is synthesized.
    """
    from .chained_body_program import BodyProgram
    from .chained_body_program import emit_body_program
    from .chained_execution import ChainedExecution
    from .prepared_tcgen_binding import PreparedProjectionHost

    model = plan.prepared_continuation
    if model is None:
        raise chain._UnsupportedChain("unselected external continuation")
    first_k = chain._shape(model.first.lhs)[1] // 16
    second_k = chain._shape(model.second.lhs)[1] // 16
    program = BodyProgram(
        (
            ContinuationPrevious((ContinuationWait(0), ContinuationRelease(1))),
            ContinuationWait(2),
            ContinuationWait(3, toggle=True, fence_after=True),
            ContinuationWait(4),
            ContinuationIssue(model, 0, 0, 0, first_k, False, True, False),
            ContinuationWait(5, toggle=True, fence_after=True),
            ContinuationIssue(model, 1, 1, 0, second_k, True, True, False),
        )
    )
    lowering = PreparedBodyLowering(
        cg,
        PreparedProjectionHost(plan),
        model,
        program,
        6,
        ((0, first_k), (0, second_k)),
    )
    emit_body_program(
        cg,
        plan,
        None,
        ChainedExecution(plan.threads),
        None,
        None,
        None,
        prepared_body=lowering,
    )
    return lowering.payload
