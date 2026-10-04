"""Original ring/event and role-resource operands for the common body walk.

No action changes a ring period, owner extent or arrival contribution. These
ports name the existing allocations in the original host adapter. The same
native continuation-control leaf implements waits, arrivals and commits for
both these descriptor ports and the existing root continuation clients.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from . import chained_matmul as chain
from .prepared_continuation import ContinuationOpcode

if TYPE_CHECKING:
    from .prepared_tcgen_binding import PreparedProjectionHost


class EpochPort(Enum):
    TMA = ("tma_mbar", 6, 1)
    RAW_READY = ("raw_ready_mbar", 8, 1)
    RAW_FREE = ("raw_consumed_mbar", 8, 5)
    STATE_LEFT = ("state_input_ready_l_mbar", 1, 4)
    STATE_RIGHT = ("state_input_ready_mbar", 1, 4)
    INPUT = ("u_input_ready_mbar", 2, 4)
    UPDATE = ("update_ready_mbar", 1, 8)
    PROJECTED = ("shared_acc_ready_mbar", 2, 1)
    LEFT_DONE = ("k_restore_consumed_l_mbar", 2, 1)
    RIGHT_DONE = ("k_restore_consumed_mbar", 2, 1)
    QUERY_ACC = ("qstate_acc_ready_mbar", 2, 1)
    QUERY_DONE = ("stateq_done_mbar", 2, 1)
    OUTPUT_FREE = ("output_ready_mbar", 2, 4)
    FINAL = ("final_state_stored_mbar", 1, 8)


class EpochPosition(Enum):
    INITIALIZE = "initialize"
    FIRST = "first"
    CURRENT = "current"
    PREVIOUS = "previous"
    PREVIOUS_QUERY = "previous_query"
    FINAL = "final"
    RAW_CURSOR = "raw_cursor"
    READY_CURSOR = "ready_cursor"
    TMA_CURSOR = "tma_cursor"
    LEFT_PHASE = "si_l_phase"
    RIGHT_PHASE = "si_phase"
    QUERY_PHASE = "si_phase12"
    UPDATE_PHASE = "upd_phase"
    UPDATE_QUERY_PHASE = "upd_phase12"


@dataclass(frozen=True)
class EpochEvent:
    owner: PreparedProjectionHost
    port: EpochPort
    position: EpochPosition
    operation: ContinuationOpcode | None
    returns_phase: bool = False

    def facts(self) -> object:
        if not isinstance(self.port, EpochPort) or not isinstance(
            self.position, EpochPosition
        ):
            raise chain._UnsupportedChain("unknown original ring/event binding")
        if self.operation not in (
            None,
            ContinuationOpcode.WAIT,
            ContinuationOpcode.WAIT_TOGGLE,
            ContinuationOpcode.RELEASE,
            ContinuationOpcode.COMMIT,
        ):
            raise chain._UnsupportedChain("unknown original ring operation")
        if self.operation is not None and not isinstance(
            self.operation, ContinuationOpcode
        ):
            raise chain._UnsupportedChain("untyped original ring operation")
        if (
            type(self.returns_phase) is not bool
            or self.returns_phase
            and self.operation is not ContinuationOpcode.WAIT_TOGGLE
        ):
            raise chain._UnsupportedChain("foreign original phase result")
        self.operands()
        self._check_access()
        return (
            id(self),
            id(self.owner),
            self.port,
            self.position,
            self.operation,
            self.returns_phase,
        )

    def _check_access(self) -> None:
        """Original access modes of these allocations, not readiness inference."""
        p, g, op = EpochPort, EpochPosition, ContinuationOpcode
        if self.position is g.INITIALIZE:
            return
        if self.operation is None:
            raise chain._UnsupportedChain("missing original ring operation")
        toggle = {
            p.RAW_FREE: (g.RAW_CURSOR,),
            p.RAW_READY: (g.FIRST, g.CURRENT),
            p.STATE_LEFT: (g.FIRST, g.LEFT_PHASE),
            p.STATE_RIGHT: (g.RIGHT_PHASE, g.QUERY_PHASE),
            p.UPDATE: (g.UPDATE_PHASE, g.UPDATE_QUERY_PHASE),
            p.PROJECTED: (g.FIRST, g.CURRENT),
            p.LEFT_DONE: (g.PREVIOUS, g.FINAL),
            p.RIGHT_DONE: (g.PREVIOUS, g.FINAL),
            p.QUERY_ACC: (g.CURRENT, g.PREVIOUS_QUERY),
            p.QUERY_DONE: (g.PREVIOUS,),
            p.OUTPUT_FREE: (g.CURRENT,),
            p.FINAL: (g.FIRST,),
        }
        release = {
            p.RAW_READY: (g.READY_CURSOR,),
            p.RAW_FREE: (g.PREVIOUS, g.PREVIOUS_QUERY),
            p.UPDATE: (g.CURRENT,),
            p.OUTPUT_FREE: (g.CURRENT,),
            p.FINAL: (g.CURRENT,),
        }
        accesses = {
            op.WAIT: {p.TMA: (g.TMA_CURSOR,)},
            op.WAIT_TOGGLE: toggle,
            op.RELEASE: release,
            op.COMMIT: {p.LEFT_DONE: (g.CURRENT,), p.RIGHT_DONE: (g.CURRENT,)},
        }
        if self.position not in accesses.get(self.operation, {}).get(self.port, ()):
            raise chain._UnsupportedChain("event changed original access mode")
        phase_results = (
            g.LEFT_PHASE,
            g.RIGHT_PHASE,
            g.QUERY_PHASE,
            g.UPDATE_PHASE,
            g.UPDATE_QUERY_PHASE,
        )
        if self.returns_phase != (self.position in phase_results):
            raise chain._UnsupportedChain("event lost original phase update")

    def operands(self) -> tuple[str, str]:
        """Bind only original scalar aliases for a declared generation kind."""
        port, position = self.port, self.position
        name, period, _ = port.value
        if position is EpochPosition.INITIALIZE:
            if self.operation is not None or self.returns_phase:
                raise chain._UnsupportedChain("ring initialization became a wait")
            return (name if period == 1 else f"{name}.subview(stage)"), "0"
        if self.operation is None:
            raise chain._UnsupportedChain("missing original ring operation")
        if position is EpochPosition.FIRST:
            return (
                name if period == 1 else f"{name}.subview(0)"
            ), "cutlass.Int32(0)" if period == 1 else "0"
        if position is EpochPosition.TMA_CURSOR and port is EpochPort.TMA:
            return f"{name}.subview(wait_mbar_slot)", "tma_phase"
        if position is EpochPosition.RAW_CURSOR and port is EpochPort.RAW_FREE:
            return f"{name}.subview(raw_stage)", "raw_consumed_phase"
        if position is EpochPosition.READY_CURSOR and port is EpochPort.RAW_READY:
            return f"{name}.subview(ready_stage)", "0"
        phase_ports = {
            EpochPosition.LEFT_PHASE: EpochPort.STATE_LEFT,
            EpochPosition.RIGHT_PHASE: EpochPort.STATE_RIGHT,
            EpochPosition.QUERY_PHASE: EpochPort.STATE_RIGHT,
            EpochPosition.UPDATE_PHASE: EpochPort.UPDATE,
            EpochPosition.UPDATE_QUERY_PHASE: EpochPort.UPDATE,
        }
        if position in phase_ports and port is phase_ports[position]:
            return name, position.value
        if position is EpochPosition.CURRENT:
            if port is EpochPort.RAW_READY:
                return f"{name}.subview(raw_stage)", "chunk // K2_RAW_STAGE_COUNT % 2"
            if port is EpochPort.PROJECTED:
                return f"{name}.subview(acc_stage)", "chunk // 2 % 2"
            if port in (EpochPort.QUERY_ACC, EpochPort.OUTPUT_FREE):
                phase = (
                    "qstate_phase"
                    if port is EpochPort.QUERY_ACC
                    else "(chunk // K2_QSTATE_STAGE_COUNT + cutlass.Int32(1)) % 2"
                )
                return f"{name}.subview(qstate_stage)", phase
            if port in (EpochPort.LEFT_DONE, EpochPort.RIGHT_DONE):
                return f"{name}.subview(kr_stage)", "0"
            if port in (EpochPort.UPDATE, EpochPort.FINAL):
                return name, "0"
        if position in (EpochPosition.PREVIOUS, EpochPosition.PREVIOUS_QUERY):
            previous = "prev" if position is EpochPosition.PREVIOUS else "prev12"
            if port is EpochPort.RAW_FREE:
                return f"{name}.subview({previous} % K2_RAW_STAGE_COUNT)", "0"
            if port is EpochPort.QUERY_ACC and position is EpochPosition.PREVIOUS_QUERY:
                return (
                    f"{name}.subview(prev12 % K2_QSTATE_STAGE_COUNT)",
                    "prev12 // K2_QSTATE_STAGE_COUNT % 2",
                )
            if (
                port
                in (EpochPort.QUERY_DONE, EpochPort.LEFT_DONE, EpochPort.RIGHT_DONE)
                and position is EpochPosition.PREVIOUS
            ):
                return f"{name}.subview(prev % 2)", "prev_kr_phase"
        if position is EpochPosition.FINAL and port in (
            EpochPort.LEFT_DONE,
            EpochPort.RIGHT_DONE,
        ):
            return f"{name}.subview(last % 2)", "last_kr_phase"
        raise chain._UnsupportedChain("unsupported original event generation")

    def emit(self) -> list[str]:
        self.facts()
        pointer, phase = self.operands()
        if self.operation is None:
            return [f"prims.mbarrier_init({pointer}, {self.port.value[2]})"]
        if self.operation in (ContinuationOpcode.RELEASE, ContinuationOpcode.COMMIT):
            phase = "0"
        call = f"prepared_tcgen_edge._continuation_control({pointer}, {phase}, True, {int(self.operation)}, False, True)"
        return [f"{phase} = {call}" if self.returns_phase else call]


class EpochSynchronization(Enum):
    CTA = "cta_sync()"
    TMEM_USERS = "tmem_user_sync()"
    OUTPUT = "output_drain_sync()"
    INITIALIZED = "prims.fence_mbarrier_init()"
    ACQUIRED_TMEM = "prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)"


class EpochResourceOperation(Enum):
    ALLOCATE = "prims.tcgen05_alloc(tmem_ptr_i32, TMEM_ALLOC_COLS, group='cta_1')"
    PERMIT = "prims.tcgen05_relinquish_alloc_permit(group='cta_1')"
    RETIRE = "prims.tcgen05_dealloc(tmem_ptr, TMEM_ALLOC_COLS, group='cta_1')"
    COMPUTE_REGISTERS = (
        "prims.setmaxregister(KDA_CG1_REGS, prims.SetMaxRegisterAction.INCREASE)"
    )
    SERVICE_REGISTERS = (
        "prims.setmaxregister(KDA_SERVICE_REGS, prims.SetMaxRegisterAction.DECREASE)"
    )
    OUTPUT_SPACE = "prims.cp_async_bulk_wait_group(6, read=True)"
    OUTPUT_COMPLETE = "prims.cp_async_bulk_wait_group(0, read=True)"


@dataclass(frozen=True)
class EpochSynchronizationAction:
    owner: PreparedProjectionHost
    operation: EpochSynchronization | EpochResourceOperation

    def facts(self) -> object:
        if not isinstance(
            self.operation, (EpochSynchronization, EpochResourceOperation)
        ):
            raise chain._UnsupportedChain("unknown original role resource operation")
        return id(self), id(self.owner), self.operation, self.operation.value

    def emit(self) -> list[str]:
        self.facts()
        return [self.operation.value]
