"""Original full-owner descriptor ports for bound contraction actions."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Literal

from . import chained_matmul as chain
from .chained_completed_store import _records
from .prepared_continuation import ContractionIssue
from .prepared_graph_schedule import ContractionGraph
from .prepared_graph_schedule import IssuePlacement
from .prepared_graph_schedule import NativeIssueBinding
from .prepared_graph_schedule import NativeReadiness
from .prepared_graph_schedule import OperandPublication
from .prepared_graph_schedule import schedule_native_issues

if TYPE_CHECKING:
    from .prepared_epoch import ContractionSegment
    from .prepared_epoch_protocol import EpochEvent
    from .prepared_epoch_protocol import EpochSynchronizationAction
    from .prepared_epoch_setup import EpochSetup
    from .prepared_tcgen_binding import PreparedProjectionHost


@dataclass(frozen=True)
class DescriptorIssue:
    """The original spec, not a helper-name or same-shaped replacement dot.

    The owner keeps its original full TMEM/SMEM allocations. ``half`` is only
    an R result-member slice; P instead holds the exact original K segment.
    Ring waits and the two original R commits remain enclosing body actions.
    """

    owner: PreparedProjectionHost
    kind: Literal["projection", "query", "output", "update"]
    segment: ContractionSegment | None = None
    half: int | None = None
    _readiness: dict[object, NativeReadiness] | None = field(
        default=None, repr=False, compare=False
    )
    issue: ContractionIssue = field(init=False)
    binding: NativeIssueBinding = field(init=False, repr=False)
    _facts: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "issue", self._issue())
        object.__setattr__(self, "binding", self._bind())
        object.__setattr__(self, "_facts", self._current())

    def _bind(self) -> NativeIssueBinding:
        """Original operand publication/owner facts, never a role action list."""
        from . import chunk_recurrence_sm100 as original
        from .prepared_continuation import ContinuationOpcode
        from .prepared_epoch_protocol import EpochEvent
        from .prepared_epoch_protocol import EpochPort
        from .prepared_epoch_protocol import EpochPosition
        from .prepared_epoch_protocol import EpochSynchronization
        from .prepared_epoch_protocol import EpochSynchronizationAction
        from .prepared_epoch_setup import EpochSetup
        from .prepared_epoch_setup import EpochSetupKind

        role = (
            original.ROLES.super_mma
            if self.kind in ("query", "output")
            else original.ROLES.tcgen05_mma
        )
        pool = self._readiness if self._readiness is not None else {}

        def ready(
            port: EpochPort,
            position: EpochPosition,
            *,
            phase: bool = False,
            acquire: bool = False,
            address: EpochSetupKind | None = None,
            before: bool = False,
        ) -> NativeReadiness:
            expected = NativeReadiness(
                EpochEvent(
                    self.owner,
                    port,
                    position,
                    ContinuationOpcode.WAIT_TOGGLE,
                    returns_phase=phase,
                ),
                EpochSynchronizationAction(
                    self.owner, EpochSynchronization.ACQUIRED_TMEM
                )
                if acquire
                else None,
                EpochSetup(self.owner, address) if address is not None else None,
                before,
            )
            key = (role, port, position)
            actual = pool.setdefault(key, expected)
            if actual != expected:
                raise chain._UnsupportedChain(
                    "original descriptor readiness binding changed"
                )
            return actual

        raw = ready(EpochPort.RAW_READY, EpochPosition.CURRENT)
        available = complete = None
        spec = self.issue.spec
        member = (0, original.DK)
        if self.kind == "projection":
            assert self.segment is not None
            first = self.segment.begin == 0
            state = ready(
                EpochPort.STATE_LEFT if first else EpochPort.STATE_RIGHT,
                EpochPosition.LEFT_PHASE if first else EpochPosition.RIGHT_PHASE,
                phase=True,
                acquire=True,
            )
            inputs = (
                OperandPublication("rhs", spec.rhs, state),
                OperandPublication("lhs", spec.lhs, raw),
            )
        elif self.kind == "query":
            state = ready(
                EpochPort.STATE_RIGHT,
                EpochPosition.QUERY_PHASE,
                phase=True,
                acquire=True,
            )
            inputs = (
                OperandPublication("rhs", spec.rhs, state),
                OperandPublication("lhs", spec.lhs, raw),
            )
            available = ready(EpochPort.OUTPUT_FREE, EpochPosition.CURRENT)
        else:
            factor = ready(
                EpochPort.UPDATE,
                EpochPosition.UPDATE_QUERY_PHASE
                if self.kind == "output"
                else EpochPosition.UPDATE_PHASE,
                phase=True,
                acquire=True,
                address=EpochSetupKind.XPACK_COL12
                if self.kind == "output"
                else EpochSetupKind.XPACK_COL,
                before=self.kind == "update",
            )
            if self.kind == "output":
                inputs = (
                    OperandPublication("rhs", spec.rhs, factor),
                    OperandPublication("lhs", spec.lhs, raw),
                )
                available = ready(EpochPort.OUTPUT_FREE, EpochPosition.CURRENT)
            else:
                assert self.half is not None
                inputs = (
                    OperandPublication("rhs", spec.rhs, raw),
                    OperandPublication("lhs", spec.lhs, factor),
                )
                member = (
                    self.half * original.DK // 2,
                    (self.half + 1) * original.DK // 2,
                )
                complete = EpochEvent(
                    self.owner,
                    EpochPort.LEFT_DONE if self.half == 0 else EpochPort.RIGHT_DONE,
                    EpochPosition.CURRENT,
                    ContinuationOpcode.COMMIT,
                    returns_phase=False,
                )
        return NativeIssueBinding(
            IssuePlacement(self.issue, self.owner, role, member),
            inputs,
            available,
            complete,
        )

    def _issue(self) -> ContractionIssue:
        from .chunk_recurrence import validate_epoch_match

        self.owner.check()
        validate_epoch_match(self.owner.plan)
        epoch = self.owner.plan.prepared_epoch
        assert epoch is not None
        region = epoch.continuation.region
        if self.kind == "projection":
            if self.segment is None or self.half is not None:
                raise chain._UnsupportedChain("projection needs its exact K segment")
            initialized = epoch.projected.initialized(self.segment)
            return ContractionIssue(
                region,
                epoch.projected.spec,
                self.segment.begin,
                self.segment.end,
                epoch.projected.atom_k,
                initialized,
                self.segment.commit,
                self.segment.wait_after,
            )
        if self.segment is not None:
            raise chain._UnsupportedChain("foreign segmented contraction port")
        if self.kind == "update":
            if type(self.half) is not int or self.half not in (0, 1):
                raise chain._UnsupportedChain("update requires an original N half")
            return ContractionIssue(region, epoch.update, 0, 1, 16, True, False)
        if self.half is not None:
            raise chain._UnsupportedChain("N half on a non-update contraction")
        if self.kind == "query":
            return ContractionIssue(
                region, epoch.continuation.first, 0, 8, 16, False, True
            )
        if self.kind == "output":
            return ContractionIssue(
                region, epoch.continuation.second, 0, 1, 16, True, True
            )
        raise chain._UnsupportedChain("unknown original contraction port")

    def _current(self) -> object:
        from . import chunk_recurrence_sm100 as original

        return (
            id(self.owner),
            id(self.owner.plan),
            self.kind,
            id(self.segment),
            self.half,
            self.binding.facts(),
            _records((self.issue, original._TMEM_LAYOUT, original.ROLES)),
            tuple(
                (name, value)
                for name, value in vars(original).items()
                if name.startswith("TCGEN05_") and type(value) is int
            ),
        )

    def facts(self) -> object:
        current = self._issue()
        if current != self.issue or self._current() != self._facts:
            raise chain._UnsupportedChain("original contraction port changed")
        self.issue.check()
        return id(self), self._facts

    def emit(self) -> list[str]:
        from . import chunk_recurrence_sm100 as original

        self.facts()
        issue = self.issue
        state_input = original._TMEM_LAYOUT.region("state_input")
        factor = original._TMEM_LAYOUT.region("factor_input")
        primary = original._TMEM_LAYOUT.region("primary_accumulator")
        secondary = original._TMEM_LAYOUT.region("secondary_accumulator")
        auxiliary = original._TMEM_LAYOUT.region("auxiliary_accumulator")
        state = original._TMEM_LAYOUT.region("state")
        factor_column = f"{factor.column_offset} + acc_stage * {factor.columns}"
        a_column = str(state_input.column_offset)
        accumulator = f"(tmem_raw_addr, qstate_stage, {primary.column_offset}, {secondary.column_offset - primary.column_offset})"
        completion = "stateq_done_mbar.subview(kr_stage)"
        swizzle, major, base, columns = 128, 0, 0, original.BT
        leading, stride = (
            original.TCGEN05_STATE_K_B_LEADING_BYTES,
            original.TCGEN05_STATE_K_B_STRIDE_BYTES,
        )
        advance = (
            original.TCGEN05_F16_ELEM_BYTES,
            original.TCGEN05_SW128_K_PHASES_PER_SLICE,
            original.BT,
            original.TCGEN05_SW128_BYTES,
        )
        operand = "qd_smem.subview(raw_stage * TILE_ELEMS)"
        if self.kind == "projection":
            operand = "kd_stage_smem"
            accumulator = f"(tmem_raw_addr, acc_stage, {auxiliary.column_offset}, {auxiliary.columns})"
            completion = "shared_acc_ready_mbar.subview(acc_stage)"
        elif self.kind == "output":
            operand, a_column = (
                "qk_smem.subview(raw_stage * QK_REC_ELEMS)",
                factor_column,
            )
            completion = "qstate_acc_ready_mbar.subview(qstate_stage)"
            leading, stride = (
                original.TCGEN05_VALUE_PAIRWISE_B_LEADING_BYTES,
                original.TCGEN05_VALUE_PAIRWISE_B_STRIDE_BYTES,
            )
            swizzle = 32
            advance = (original.TCGEN05_F16_ELEM_BYTES, 1, original.BT, 32)
        elif self.kind == "update":
            assert self.half is not None
            operand, a_column = "w_stage_smem", factor_column
            columns, major = original.DK // 2, 1
            leading, stride = (
                original.TCGEN05_FINAL_STATE_B_LEADING_BYTES,
                original.TCGEN05_FINAL_STATE_B_STRIDE_BYTES,
            )
            base = self.half * leading
            accumulator = (
                f"(tmem_raw_addr, 0, {state.column_offset + self.half * columns}, 0)"
            )
            completion = "None"
        return [
            (
                "prepared_tcgen_edge.execute_prepared_issue("
                f"(tmem_raw_addr, {a_column}), ({operand}, {leading}, {stride}), "
                f"{accumulator}, (input_dtype, {original.DV_HALF}, {columns}), "
                f"issuer=True, completion={completion}, completion_phase=0, "
                "input_ready=None, input_phase=0, DESCRIPTOR=True, TMEM_A=True, "
                f"K_BEGIN={issue.begin}, K_END={issue.end}, K_ATOM={issue.atom_k}, "
                f"B_ADVANCE={advance!r}, INITIALIZED={issue.initialized!r}, "
                f"COMMIT={issue.commit!r}, WAIT_AFTER={issue.wait_after!r}, "
                f"B_SWIZZLE={swizzle}, B_MAJOR={major}, B_BASE_BYTES={base})"
            )
        ]


def descriptor_issue_schedules(
    host: PreparedProjectionHost,
) -> dict[
    object,
    tuple[DescriptorIssue | EpochEvent | EpochSynchronizationAction | EpochSetup, ...],
]:
    """Bind an unordered actual issue set; common graph scheduling owns order."""
    from .prepared_epoch_protocol import EpochEvent
    from .prepared_epoch_protocol import EpochSynchronizationAction
    from .prepared_epoch_setup import EpochSetup

    epoch = host.plan.prepared_epoch
    assert epoch is not None
    pool: dict[object, NativeReadiness] = {}
    operations = (
        DescriptorIssue(host, "update", half=1, _readiness=pool),
        DescriptorIssue(host, "output", _readiness=pool),
        DescriptorIssue(
            host, "projection", epoch.projected.segments[1], _readiness=pool
        ),
        DescriptorIssue(host, "query", _readiness=pool),
        DescriptorIssue(host, "update", half=0, _readiness=pool),
        DescriptorIssue(
            host, "projection", epoch.projected.segments[0], _readiness=pool
        ),
    )
    scheduled = schedule_native_issues(
        ContractionGraph(epoch.continuation.region), operations
    )
    result = {}
    for role, actions in scheduled:
        bound = []
        for action in actions:
            if not isinstance(
                action,
                (DescriptorIssue, EpochEvent, EpochSynchronizationAction, EpochSetup),
            ):
                raise chain._UnsupportedChain("foreign original native issue action")
            bound.append(action)
        result[role] = tuple(bound)
    return result
