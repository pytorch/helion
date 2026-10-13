"""Structured source effects for an original, completely matched role epoch.

The original adapter supplies scalars and native leaves. This module holds no
whole-role callback, allocation planner, expression evaluator or readiness
substitute. BodyProgram walks every original branch and induction scope.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import TYPE_CHECKING

from . import chained_matmul as chain
from .chained_completed_store import _records
from .chained_execution import ChainedExecution
from .prepared_continuation import ContinuationOpcode
from .prepared_epoch_issue import DescriptorIssue
from .prepared_epoch_memory import EpochMemoryAction
from .prepared_epoch_memory import EpochMemoryOperation
from .prepared_epoch_product import DescriptorProductBinding
from .prepared_epoch_protocol import EpochEvent
from .prepared_epoch_protocol import EpochPort
from .prepared_epoch_protocol import EpochPosition
from .prepared_epoch_protocol import EpochResourceOperation
from .prepared_epoch_protocol import EpochSynchronization
from .prepared_epoch_protocol import EpochSynchronizationAction
from .prepared_epoch_setup import EpochSetup
from .prepared_epoch_state import DescriptorStateBinding
from .prepared_epoch_state import StateArrival
from .prepared_state_body import StateEffect
from .prepared_state_body import emit_state_action
from .prepared_tcgen_binding import PreparedProjectionHost
from .prepared_tcgen_binding import _native_source

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..generate_ast import GenerateAST
    from .chained_body_program import BodyProgram
    from .chunk_recurrence import CuteChunkRecurrencePlan


class EpochPredicate(Enum):
    TMA = "warp_idx == ROLES.tma_load"
    QUERY = "warp_idx == ROLES.super_mma"
    CHAIN = "warp_idx == ROLES.tcgen05_mma"
    EPILOGUE = "warp_idx == ROLES.epilogue"
    RIGHT = "is_compute_group1_warp(warp_idx)"
    GROUP0 = "is_compute_group0_warp(warp_idx)"
    OUTPUT = "warp_idx < 4"
    OUTPUT_LEADER = "warp_idx == 0"
    TMEM_USERS = "is_tmem_user_warp(warp_idx)"
    SERVICE = "is_service_warpgroup(warp_idx)"
    ELECTED = "prims.elect_sync()"
    NONEMPTY = "num_chunks > 0"
    PREVIOUS = "chunk > 0"
    PIPELINE_FULL = "chunk >= cutlass.Int32(K2_TMA_MBAR_STAGE_COUNT - 1)"
    FULL_OUTPUT = "seqlen >= output_chunk_start + BT"


@dataclass(frozen=True)
class EpochBranch:
    predicate: EpochPredicate
    body: BodyProgram
    otherwise: BodyProgram

    @property
    def condition(self) -> str:
        if not isinstance(self.predicate, EpochPredicate):
            raise chain._UnsupportedChain("unknown original epoch predicate")
        return self.predicate.value

    def facts(self) -> object:
        return (
            id(self),
            self.predicate,
            _source_facts(self.body),
            _source_facts(self.otherwise),
        )


class EpochLoop(Enum):
    ALL = ("chunk", "cutlass.range(num_chunks, unroll=1)")
    LATER = ("chunk", "cutlass.range(1, num_chunks, 1, unroll=1)")
    TAIL = ("_tail", "cutlass.range(tma_tail, unroll=1)")
    INIT_TMA = ("stage", "cutlass.range_constexpr(K2_TMA_MBAR_STAGE_COUNT)")
    INIT_RAW = ("stage", "cutlass.range_constexpr(K2_RAW_STAGE_COUNT)")
    INIT_PAIR = ("stage", "cutlass.range_constexpr(2)")
    INIT_OUTPUT = ("stage", "cutlass.range_constexpr(K2_QSTATE_STAGE_COUNT)")


@dataclass(frozen=True)
class EpochRange:
    domain: EpochLoop
    body: BodyProgram

    @property
    def target(self) -> str:
        return self.domain.value[0]

    @property
    def iterator(self) -> str:
        return self.domain.value[1]

    def facts(self) -> object:
        if not isinstance(self.domain, EpochLoop):
            raise chain._UnsupportedChain("unknown original epoch iteration domain")
        return id(self), self.domain, _source_facts(self.body)


def _event_participant(action: EpochEvent, scope: tuple[object, ...]) -> None:
    """Original event participants and iteration domains, without new readiness."""
    p, g, pred, loop = EpochPort, EpochPosition, EpochPredicate, EpochLoop
    positive = [
        role
        for role in (pred.TMA, pred.QUERY, pred.CHAIN, pred.RIGHT, pred.GROUP0)
        if (role, True) in scope
    ]
    if len(positive) != 1:
        raise chain._UnsupportedChain("event lost unique original role")
    role = positive[0]
    participant = role.name.lower()
    if role is pred.GROUP0:
        if (pred.OUTPUT, True) in scope:
            participant = "output"
        elif (pred.OUTPUT, False) in scope:
            participant = "left"
        else:
            raise chain._UnsupportedChain("event lost original split participant")
    port, position, operation = action.port, action.position, action.operation
    if position in (g.TMA_CURSOR, g.RAW_CURSOR, g.READY_CURSOR):
        allowed = ("tma",)
    elif position in (g.QUERY_PHASE, g.UPDATE_QUERY_PHASE):
        allowed = ("query",)
    elif position in (g.LEFT_PHASE, g.RIGHT_PHASE, g.UPDATE_PHASE):
        allowed = ("chain",)
    elif port is p.RAW_READY:
        allowed = (
            ("right",) if position is g.FIRST else ("right", "left", "query", "chain")
        )
    elif port is p.RAW_FREE:
        allowed = ("query",) if position is g.PREVIOUS_QUERY else ("right",)
    elif port is p.PROJECTED:
        allowed = ("right",)
    elif port is p.LEFT_DONE:
        allowed = (
            ("chain",)
            if operation is ContinuationOpcode.COMMIT
            else ("right",)
            if position is g.FINAL
            else ("left",)
        )
    elif port is p.RIGHT_DONE:
        allowed = ("chain",) if operation is ContinuationOpcode.COMMIT else ("right",)
    elif port is p.QUERY_DONE:
        allowed = ("left", "right")
    elif port is p.QUERY_ACC:
        allowed = ("query",) if position is g.PREVIOUS_QUERY else ("output",)
    elif port is p.OUTPUT_FREE:
        allowed = ("output",) if operation is ContinuationOpcode.RELEASE else ("query",)
    elif port is p.STATE_LEFT:
        allowed = ("left",)
    elif port is p.UPDATE:
        allowed = ("left", "right")
    elif port is p.FINAL:
        allowed = (
            ("chain",)
            if operation is ContinuationOpcode.WAIT_TOGGLE
            else ("output", "right")
        )
    else:
        raise chain._UnsupportedChain("unknown original event participant")
    if participant not in allowed:
        raise chain._UnsupportedChain("event changed original participant")
    if position is g.FIRST and port is not p.FINAL:
        valid = (pred.NONEMPTY, True) in scope and not any(
            isinstance(item, EpochLoop) for item in scope
        )
    elif port is p.FINAL:
        valid = (
            not any(isinstance(item, EpochLoop) for item in scope)
            and (pred.NONEMPTY, True) not in scope
        )
    elif position is g.RAW_CURSOR:
        valid = loop.ALL in scope
    elif position in (g.TMA_CURSOR, g.READY_CURSOR):
        valid = (
            loop.TAIL in scope
            or loop.ALL in scope
            and (pred.PIPELINE_FULL, True) in scope
        )
    elif (
        position is g.CURRENT
        and port is p.UPDATE
        and operation is ContinuationOpcode.RELEASE
    ):
        valid = loop.LATER in scope or (pred.NONEMPTY, True) in scope
    elif position is g.CURRENT:
        valid = (loop.LATER if participant in ("right", "left") else loop.ALL) in scope
    elif position in (
        g.LEFT_PHASE,
        g.RIGHT_PHASE,
        g.QUERY_PHASE,
        g.UPDATE_PHASE,
        g.UPDATE_QUERY_PHASE,
    ):
        valid = loop.ALL in scope
    else:
        valid = True  # Previous/final domain checks are at the enclosing walk.
    if not valid:
        raise chain._UnsupportedChain("event changed original iteration domain")


def _synchronization_scope(
    action: EpochSynchronizationAction, scope: tuple[object, ...]
) -> None:
    pred, resource, sync = EpochPredicate, EpochResourceOperation, EpochSynchronization
    operation = action.operation
    output = all((item, True) in scope for item in (pred.GROUP0, pred.OUTPUT))
    loops = any(isinstance(item, EpochLoop) for item in scope)
    if operation in (resource.ALLOCATE, resource.PERMIT, resource.RETIRE):
        valid = (
            (pred.CHAIN, True) in scope
            and not loops
            and (pred.NONEMPTY, True) not in scope
        )
    elif operation is resource.COMPUTE_REGISTERS:
        valid = (
            (pred.RIGHT, True) in scope or (pred.GROUP0, True) in scope
        ) and not loops
    elif operation is resource.SERVICE_REGISTERS:
        valid = (pred.SERVICE, True) in scope and not loops
    elif operation in (resource.OUTPUT_SPACE, resource.OUTPUT_COMPLETE):
        valid = (
            output
            and (pred.OUTPUT_LEADER, True) in scope
            and (
                (EpochLoop.ALL in scope)
                if operation is resource.OUTPUT_SPACE
                else not loops
            )
        )
    elif operation in (sync.CTA, sync.INITIALIZED):
        valid = not scope
    elif operation is sync.TMEM_USERS:
        valid = (pred.TMEM_USERS, True) in scope and not loops
    elif operation is sync.OUTPUT:
        valid = output and (not loops or EpochLoop.ALL in scope)
    elif operation is sync.ACQUIRED_TMEM:
        valid = (
            (pred.GROUP0, True) in scope
            and (pred.OUTPUT, False) in scope
            and (pred.NONEMPTY, True) in scope
            and not loops
        ) or (
            ((pred.QUERY, True) in scope or (pred.CHAIN, True) in scope)
            and EpochLoop.ALL in scope
        )
    else:
        valid = False
    if not valid:
        raise chain._UnsupportedChain(
            "original synchronization participant/domain changed"
        )


def _source_facts(program: BodyProgram) -> object:
    facts = []
    for action in program.actions:
        if not isinstance(
            action,
            (
                EpochSetup,
                EpochMemoryAction,
                EpochBranch,
                EpochRange,
                StateEffect,
                StateArrival,
                DescriptorIssue,
                EpochEvent,
                EpochSynchronizationAction,
            ),
        ):
            raise chain._UnsupportedChain("unknown epoch source action")
        facts.append(action.facts())
    return tuple(facts)


@dataclass
class EpochSourceBody:
    codegen: GenerateAST
    plan: CuteChunkRecurrencePlan
    program: BodyProgram
    header: str
    declarations: tuple[str, ...]
    globals: tuple[str, ...]
    host: PreparedProjectionHost
    _facts: object = field(init=False, repr=False)
    _native: tuple[object, ...] = field(init=False, repr=False)
    _returned: tuple[str, ...] | None = field(default=None, init=False)
    consumed: bool = field(default=False, init=False)
    trace: list[tuple[tuple[object, ...], object]] = field(
        default_factory=list, init=False
    )

    def __post_init__(self) -> None:
        from . import chunk_recurrence_sm100 as original
        from .chunk_recurrence import validate_epoch_match

        validate_epoch_match(self.plan)
        if self.host.plan is not self.plan:
            raise chain._UnsupportedChain("foreign original epoch host")
        self._native = _native_source(original.kernel_chain_dv2, "_kernel_helper")
        self._facts = self.facts()
        self._validate_state_actions()

    def facts(self) -> object:
        from . import chunk_recurrence_sm100 as original

        return (
            id(self.codegen),
            id(self.plan),
            id(self.plan.prepared_epoch),
            self.program.facts(),
            self.header,
            self.declarations,
            self.globals,
            tuple((name, id(vars(original)[name])) for name in self.globals),
            _records(
                tuple(
                    (name, value)
                    for name, value in vars(self.plan).items()
                    if not name.startswith("prepared_")
                )
            ),
        )

    def check(self, *, completed: bool = False) -> None:
        from . import chunk_recurrence_sm100 as original
        from .chunk_recurrence import validate_epoch_match

        validate_epoch_match(self.plan)
        self.host.check()
        if (
            self.consumed is not completed
            or self.facts() != self._facts
            or _native_source(original.kernel_chain_dv2, "_kernel_helper")
            != self._native
        ):
            raise chain._UnsupportedChain("epoch source or original owner changed")
        self._validate_state_actions()

    def _validate_state_actions(self) -> None:
        """Retain actual cycle SSA identities through the enclosing source walk."""

        def flatten(program: BodyProgram) -> Iterator[object]:
            for action in program.actions:
                if isinstance(action, EpochBranch):
                    yield from flatten(action.body)
                    yield from flatten(action.otherwise)
                elif isinstance(action, EpochRange):
                    yield from flatten(action.body)
                else:
                    yield action

        cycles = {}
        products = {}
        memory = []
        issues = []
        walk = tuple(flatten(self.program))

        def position(target: object) -> int:
            indices = [i for i, action in enumerate(walk) if action is target]
            if len(indices) != 1:
                raise chain._UnsupportedChain(
                    "state cut lost its actual enclosing action"
                )
            return indices[0]

        for action in walk:
            if isinstance(action, EpochMemoryAction):
                memory.append(action)
            if isinstance(action, DescriptorIssue):
                issues.append(action)
            if not isinstance(action, (StateEffect, StateArrival)):
                continue
            binding = action.binding
            if isinstance(binding, DescriptorProductBinding):
                if (
                    not isinstance(action, StateEffect)
                    or binding.owner is not self.host
                ):
                    raise chain._UnsupportedChain("foreign result owner")
                cycle = binding.cycle
                key = cycle.kind, cycle.first
                if key not in products:
                    products[key] = (cycle, [])
                if products[key][0] is not cycle:
                    raise chain._UnsupportedChain("duplicate original result owner")
                products[key][1].append(action)
                continue
            if (
                not isinstance(binding, DescriptorStateBinding)
                or binding.owner is not self.host
            ):
                raise chain._UnsupportedChain("foreign epoch state cycle")
            cycle = binding.cycle
            key = cycle.role, cycle.first, cycle.half
            if key not in cycles:
                cycles[key] = (cycle, [])
            if cycles[key][0] is not cycle:
                raise chain._UnsupportedChain("duplicate original state value owner")
            cycles[key][1].append(action)
        if set(cycles) != {
            ("right", True, 0),
            ("right", True, 1),
            ("right", False, 1),
            ("left", False, 0),
        }:
            raise chain._UnsupportedChain("incomplete original state cycles")
        if set(products) != {
            ("residual", True),
            ("residual", False),
            ("output", False),
        }:
            raise chain._UnsupportedChain("incomplete original result effects")
        for cycle, actual in (*cycles.values(), *products.values()):
            cycle.check_actions()
            if len(actual) != len(cycle.actions) or any(
                a is not b for a, b in zip(actual, cycle.actions, strict=True)
            ):
                raise chain._UnsupportedChain(
                    "changed state def-use or pending-store order"
                )
        for cycle, actual in cycles.values():
            raw = position(cycle.coefficient_ready)
            # The first right half starts after the first left half's RAW_READY.
            # All other cycles issue their packed store before that wait.
            if cycle.first and cycle.half == 1:
                valid = raw < position(actual[0])
            else:
                valid = position(actual[2]) < raw < position(actual[3])
            if not valid or not (
                position(actual[4]) < position(cycle.arrival) < position(actual[6])
            ):
                raise chain._UnsupportedChain(
                    "state dependency crossed original event cut"
                )
        if [action.operation for action in memory] != [
            EpochMemoryOperation.INPUTS,
            EpochMemoryOperation.IMPORT_STATE,
            EpochMemoryOperation.EXPORT_STATE,
            EpochMemoryOperation.OUTPUT_FULL,
            EpochMemoryOperation.OUTPUT_TAIL,
        ] or any(action.owner is not self.host for action in memory):
            raise chain._UnsupportedChain("incomplete original boundary transports")
        output = products[("output", False)][0]
        if not (
            position(output.actions[1])
            < position(output.release)
            < position(output.space_ready)
            < position(output.actions[2])
            < position(output.output_complete)
        ):
            raise chain._UnsupportedChain(
                "result publication crossed original space cut"
            )
        if any(action.product is not output.binding for action in memory[-2:]):
            raise chain._UnsupportedChain(
                "output transport did not consume its actual publication"
            )
        epoch = self.plan.prepared_epoch
        assert epoch is not None
        expected_issues = (
            ("query", None, None),
            ("output", None, None),
            ("projection", epoch.projected.segments[0], None),
            ("projection", epoch.projected.segments[1], None),
            ("update", None, 0),
            ("update", None, 1),
        )
        if len(issues) != len(expected_issues) or any(
            action.owner is not self.host
            or action.kind != kind
            or action.segment is not segment
            or action.half != half
            for action, (kind, segment, half) in zip(
                issues, expected_issues, strict=True
            )
        ):
            raise chain._UnsupportedChain(
                "changed original contraction prefix/half order"
            )

    def step(
        self,
        action: EpochSetup
        | EpochMemoryAction
        | StateEffect
        | StateArrival
        | DescriptorIssue
        | EpochEvent
        | EpochSynchronizationAction,
        scope: tuple[object, ...],
    ) -> list[str]:
        if isinstance(action, EpochSetup):
            if action.owner is not self.host:
                raise chain._UnsupportedChain("foreign scalar setup owner")
            self.trace.append((scope, action))
            return action.emit()
        if isinstance(action, EpochMemoryAction):
            if action.owner is not self.host:
                raise chain._UnsupportedChain("foreign boundary transport owner")
            operation = action.operation
            if operation is EpochMemoryOperation.INPUTS:
                valid = (EpochPredicate.TMA, True) in scope and EpochLoop.ALL in scope
            elif operation in (
                EpochMemoryOperation.IMPORT_STATE,
                EpochMemoryOperation.EXPORT_STATE,
            ):
                valid = (
                    (EpochPredicate.RIGHT, True) in scope
                    and not any(isinstance(item, EpochLoop) for item in scope)
                    and (EpochPredicate.NONEMPTY, True) not in scope
                )
            else:
                valid = (
                    all(
                        (predicate, True) in scope
                        for predicate in (
                            EpochPredicate.GROUP0,
                            EpochPredicate.OUTPUT,
                            EpochPredicate.OUTPUT_LEADER,
                        )
                    )
                    and EpochLoop.ALL in scope
                    and (
                        EpochPredicate.FULL_OUTPUT,
                        operation is EpochMemoryOperation.OUTPUT_FULL,
                    )
                    in scope
                )
            if not valid:
                raise chain._UnsupportedChain(
                    "boundary transport changed original participant/domain"
                )
            self.trace.append((scope, action))
            return action.emit()
        if isinstance(action, (EpochEvent, EpochSynchronizationAction)):
            if action.owner is not self.host:
                raise chain._UnsupportedChain("foreign original protocol owner")
            if isinstance(action, EpochEvent):
                action.facts()
                if action.position is EpochPosition.INITIALIZE:
                    role = (
                        EpochPredicate.TMA
                        if action.port
                        in (EpochPort.TMA, EpochPort.RAW_READY, EpochPort.RAW_FREE)
                        else EpochPredicate.EPILOGUE
                        if action.port is EpochPort.OUTPUT_FREE
                        else EpochPredicate.CHAIN
                    )
                    if (role, True) not in scope or (
                        EpochPredicate.ELECTED,
                        True,
                    ) not in scope:
                        raise chain._UnsupportedChain(
                            "event initializer lost original participant"
                        )
                    period = action.port.value[1]
                    domain = {
                        6: EpochLoop.INIT_TMA,
                        8: EpochLoop.INIT_RAW,
                        2: EpochLoop.INIT_OUTPUT
                        if action.port in (EpochPort.QUERY_ACC, EpochPort.OUTPUT_FREE)
                        else EpochLoop.INIT_PAIR,
                    }.get(period)
                    if domain is not None and domain not in scope:
                        raise chain._UnsupportedChain(
                            "event initializer changed ring extent"
                        )
                elif action.position is EpochPosition.PREVIOUS_QUERY:
                    if (
                        (EpochPredicate.QUERY, True) not in scope
                        or (EpochPredicate.PREVIOUS, True) not in scope
                        or EpochLoop.ALL not in scope
                    ):
                        raise chain._UnsupportedChain(
                            "query previous-generation event moved"
                        )
                elif (
                    action.position is EpochPosition.PREVIOUS
                    and EpochLoop.LATER not in scope
                ):
                    raise chain._UnsupportedChain(
                        "state previous-generation event moved"
                    )
                elif action.position is EpochPosition.FINAL and (
                    (EpochPredicate.NONEMPTY, True) not in scope
                    or EpochLoop.LATER in scope
                    or EpochLoop.ALL in scope
                ):
                    raise chain._UnsupportedChain("terminal state event moved")
                elif (
                    action.position
                    in (
                        EpochPosition.TMA_CURSOR,
                        EpochPosition.RAW_CURSOR,
                        EpochPosition.READY_CURSOR,
                    )
                    and (EpochPredicate.TMA, True) not in scope
                ):
                    raise chain._UnsupportedChain(
                        "ring cursor moved outside original TMA role"
                    )
                if action.position is not EpochPosition.INITIALIZE:
                    _event_participant(action, scope)
            else:
                _synchronization_scope(action, scope)
            self.trace.append((scope, action))
            return action.emit()
        if isinstance(action, DescriptorIssue):
            if action.owner is not self.host:
                raise chain._UnsupportedChain("foreign epoch issuer owner")
            role = (
                EpochPredicate.QUERY
                if action.kind in ("query", "output")
                else EpochPredicate.CHAIN
            )
            if (role, True) not in scope or EpochLoop.ALL not in scope:
                raise chain._UnsupportedChain(
                    "bound issue moved out of its original role"
                )
            self.trace.append((scope, action))
            return action.emit()
        if isinstance(action, (StateEffect, StateArrival)):
            binding = action.binding
            if isinstance(binding, DescriptorProductBinding):
                if (
                    not isinstance(action, StateEffect)
                    or binding.owner is not self.host
                ):
                    raise chain._UnsupportedChain("foreign epoch result owner")
                cycle = binding.cycle
                if cycle.kind == "residual":
                    valid = (EpochPredicate.RIGHT, True) in scope and (
                        (
                            (EpochPredicate.NONEMPTY, True) in scope
                            and EpochLoop.ALL not in scope
                            and EpochLoop.LATER not in scope
                        )
                        if cycle.first
                        else EpochLoop.LATER in scope
                    )
                else:
                    valid = (
                        (EpochPredicate.GROUP0, True) in scope
                        and (EpochPredicate.OUTPUT, True) in scope
                        and EpochLoop.ALL in scope
                    )
                if not valid:
                    raise chain._UnsupportedChain(
                        "result moved outside original role/generation"
                    )
                self.trace.append((scope, action))
                return emit_state_action(action, ChainedExecution(self.plan.threads))
            if (
                not isinstance(binding, DescriptorStateBinding)
                or binding.owner is not self.host
            ):
                raise chain._UnsupportedChain("foreign epoch state owner")
            if binding.role == "right":
                valid_role = (EpochPredicate.RIGHT, True) in scope
            else:
                valid_role = (EpochPredicate.GROUP0, True) in scope and (
                    EpochPredicate.OUTPUT,
                    False,
                ) in scope
            if (
                not valid_role
                or binding.first != (EpochLoop.LATER not in scope)
                or (binding.first and (EpochPredicate.NONEMPTY, True) not in scope)
                or (
                    binding.first and any(isinstance(item, EpochLoop) for item in scope)
                )
            ):
                raise chain._UnsupportedChain(
                    "state moved out of its original generation/role"
                )
            self.trace.append((scope, action))
            return (
                action.emit()
                if isinstance(action, StateArrival)
                else emit_state_action(action, ChainedExecution(self.plan.threads))
            )
        raise chain._UnsupportedChain("unknown bound epoch action")

    def finish(self, lines: list[str]) -> None:
        self.check()
        self._validate_state_actions()
        if self._returned is not None:
            raise chain._UnsupportedChain("epoch source replay")
        # Native instruction substitutions are intentionally permitted here.
        # Full original-source inversion belongs to the CPU controls, not the
        # production authority of the original graph, owner and typed effects.
        self._returned = tuple(lines)

    def accept(self, lines: list[str]) -> None:
        self.check()
        if self._returned is None or tuple(lines) != self._returned:
            raise chain._UnsupportedChain("returned epoch walk changed")
        self.consumed = True


def lower_epoch(cg: GenerateAST, plan: CuteChunkRecurrencePlan) -> str:
    from .chained_body_program import emit_body_program
    from .chained_execution import ChainedExecution
    from .chunk_recurrence_epoch import DECLARATIONS
    from .chunk_recurrence_epoch import GLOBALS
    from .chunk_recurrence_epoch import HEADER
    from .chunk_recurrence_epoch import original_epoch_program

    epoch = plan.prepared_epoch
    if epoch is None:
        raise chain._UnsupportedChain("unselected original epoch")
    host = PreparedProjectionHost(plan)
    body = EpochSourceBody(
        cg,
        plan,
        original_epoch_program(epoch, host),
        HEADER,
        DECLARATIONS,
        GLOBALS,
        host,
    )
    lines = emit_body_program(
        cg,
        plan,
        None,
        ChainedExecution(plan.threads),
        None,
        None,
        None,
        epoch_body=body,
    )
    body.accept(lines)
    body.check(completed=True)
    if tuple(lines) != body._returned:
        raise chain._UnsupportedChain("accepted epoch walk changed")
    imports = (
        "from helion._compiler.cute import prepared_tcgen_edge\nfrom helion._compiler.cute.chunk_recurrence_sm100 import "
        + ", ".join(GLOBALS)
    )
    source = imports + "\n" + HEADER + "\n" + chain._indent([*DECLARATIONS, *lines])
    cg.module_statements.extend(ast.parse(source).body)
    return "_helion_epoch_kernel"
