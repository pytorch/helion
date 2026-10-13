"""Bind an initial-state scratch lease to the original recurrence/output cuts."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import torch

from . import chained_matmul as chain
from .chunk_prefill_bt32 import config
from .chunk_prefill_prepared_issue import fast_issue_layouts
from .chunk_prefill_prepared_output import FastOutputCycle
from .chunk_prefill_prepared_output import bind_fast_output
from .chunk_prefill_prepared_pairwise import bind_fast_pairwise
from .chunk_prefill_prepared_state import FastLoopStatePanel
from .chunk_prefill_prepared_state_abi import FastStateABIBinding
from .chunk_prefill_prepared_state_abi import FastStateABICycle
from .chunk_prefill_prepared_state_abi import bind_external_state_owner
from .prepared_state_copy import RawStateCopyPlan
from .startup_shared_reuse import StartupSharedReuse
from .warp_specialized_plan import MBarrierRegion
from .warp_specialized_plan import SharedBufferRegion

if TYPE_CHECKING:
    from ..device_ir import GraphInfo
    from .chunk_prefill_prepared_issue import FastRecurrence


def fast_startup_reuse(planes_per_transfer: int = 1) -> StartupSharedReuse:
    """The native factor dispatch assigns one fixed stage to each warp group."""
    return StartupSharedReuse(
        config.PIPELINE_PLAN.shared_buffer("factor_stages"),
        config.PIPELINE_PLAN.role("factor"),
        config.STAGES,
        config.STAGES - 1,
        16384 * planes_per_transfer,
    )


def validate_fast_state_copy_payload(program: object) -> None:
    """Validate the serialized physical ABI after graph-bound code generation."""
    expected = (1, 128, 128, 32, config.OUT, 64, config.DONE + 8)
    pipelined = (
        2,
        *expected[1:],
        config.QD + (config.STAGES - 1) * config.STAGE_BYTES,
        config.FACTOR_FIRST_WARP + 4 * (config.STAGES - 1),
        config.THREADS // 32,
    )
    planar = (
        3,
        128,
        128,
        32,
        config.QD + (config.STAGES - 1) * config.STAGE_BYTES,
        64,
        config.DONE + 8,
        2,
        config.FACTOR_FIRST_WARP + 4 * (config.STAGES - 1),
        config.THREADS // 32,
    )
    if (
        type(program) is not tuple
        or any(type(value) is not int for value in program)
        or program not in (expected, pipelined, planar)
    ):
        raise ValueError("invalid prepared raw state-copy program")


@dataclass
class FastStateCopyLease:
    cycle: FastStateABICycle
    recurrence: FastRecurrence
    pipelined: bool = False
    planes_per_transfer: int = 1
    publication: FastLoopStatePanel = field(init=False)
    output: FastOutputCycle = field(init=False)
    _facts: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # These constructors bind actual state consumers and the original store.
        bind_fast_output(self.recurrence)
        self.publication = FastLoopStatePanel(self.recurrence, True)
        self.publication.program()
        self.output = FastOutputCycle(self.recurrence, True)
        self.output.program()
        self._facts = self._current()

    def _current(self) -> object:
        owner = self.cycle.owner
        owner.check()
        self.recurrence.check()
        ports = self.recurrence.ports
        native, ready, complete = fast_issue_layouts()
        if (
            self.cycle.mode != 0
            or type(self.planes_per_transfer) is not int
            or self.planes_per_transfer not in (1, 2)
            or (self.pipelined and self.planes_per_transfer != 1)
            or owner.region is not self.recurrence.region
            or owner.region.numerical_policy != "centered_bt32_fp32_rhs_v2"
            or owner.region.initial_state.fake.dtype is not torch.float32
            or owner.region.final_state.fake.dtype is not torch.float32
            or (
                owner.region.chunk_size,
                owner.region.key_width,
                owner.region.value_width,
            )
            != (32, 128, 128)
            or self.cycle.cut.anchor is not owner.loop
            or self.publication.owner is not self.recurrence
            or not self.publication.publish
            or self.output.owner is not self.recurrence
            or not self.output.full
            or len(ports) != 4
            or tuple(p.descriptor for p in ports) != native
            or tuple(p.ready for p in ports) != ready
            or tuple(p.complete for p in ports) != complete
        ):
            raise chain._UnsupportedChain("changed initial-state scratch lifetime")
        producer_facts = None
        if self.pipelined or self.planes_per_transfer == 2:
            # The factor role owns these original typed inputs, independent of
            # the carried state. Its native dispatch waits before factor_loop,
            # after releasing registers, so none of its stage accesses can
            # precede initialization completion. Consumers still wait QK_FULL.
            pairwise = bind_fast_pairwise(self.recurrence)
            roots = (
                pairwise.kk.lhs,
                pairwise.kk.rhs,
                pairwise.qk.lhs,
                pairwise.qk.rhs,
                owner.region.step.gate_scan,
                owner.region.step.beta_load,
            )
            if any(
                self.recurrence.state_input in chain._ancestors(node) for node in roots
            ):
                raise chain._UnsupportedChain("state-dependent startup scratch owner")
            producer_facts = (roots, fast_startup_reuse(self.planes_per_transfer))
        # Initialization precedes the loop's state-input publication. The issuer
        # waits that event before its first issue, whose ordered chain ends in
        # FINAL_READY. Output waits FINAL_READY before reading or staging data.
        # A zero-trip sequence has no output writer and exports the initialized
        # state. The three new barriers only complete the copy implementation.
        return (
            id(owner),
            id(self.recurrence),
            self.cycle.cut.facts(),
            self.publication.packed_cut.facts(),
            self.output.release.facts(),
            self.output.store.facts(),
            tuple((p.payload(), p.ready, p.complete) for p in ports),
            config.PIPELINE_PLAN,
            producer_facts,
            self.pipelined,
            self.planes_per_transfer,
        )

    def facts(self, plan: RawStateCopyPlan) -> object:
        schedule = plan.schedule
        facts = self._current()
        output = config.PIPELINE_PLAN.shared_buffer("output_stages")
        reuse = (
            fast_startup_reuse(self.planes_per_transfer)
            if self.pipelined or self.planes_per_transfer == 2
            else None
        )
        scratch = SharedBufferRegion(
            "initial_state_copy", output.byte_offset, 16384, 0, output.live_from, 1024
        )
        if self.planes_per_transfer == 2:
            assert reuse is not None
            _, scratch = reuse.resources(config.PIPELINE_PLAN)
        if (
            facts != self._facts
            or schedule.cuts != (self.cycle.cut,)
            or len(schedule.requests) != 4
            or plan.pipeline != config.PIPELINE_PLAN
            or plan.scratch != scratch
            or plan.planes_per_transfer != self.planes_per_transfer
            or plan.barriers
            != MBarrierRegion(
                "initial_state_copy", config.DONE + 8, 4 if self.pipelined else 3, 1
            )
            or plan.startup_reuse != reuse
            or plan.tmem_column != 64
            or plan.rows != 128
        ):
            raise chain._UnsupportedChain("changed initial-state scratch lease")
        for request in schedule.requests:
            if (
                not isinstance(request.read, FastStateABIBinding)
                or request.read.cycle is not self.cycle
                or request.value is not request.read
                or request.source_view.mapping
                != ("external-fp32", "sequence", "head", "value", 128)
                or request.destination_view.mapping != ("32x32b", "warp%4", 64)
            ):
                raise chain._UnsupportedChain("foreign raw initial-state view")
        return facts


def bind_fast_state_copy(
    recurrence: FastRecurrence,
    semantic_root: GraphInfo,
    *,
    pipelined: bool = False,
    planes_per_transfer: int = 1,
) -> RawStateCopyPlan:
    owner = bind_external_state_owner(recurrence.region, semantic_root)
    cycle = FastStateABICycle(owner, 0)
    schedule = cycle.schedule()
    lease = FastStateCopyLease(cycle, recurrence, pipelined, planes_per_transfer)
    output = config.PIPELINE_PLAN.shared_buffer("output_stages")
    reuse = (
        fast_startup_reuse(planes_per_transfer)
        if pipelined or planes_per_transfer == 2
        else None
    )
    scratch = SharedBufferRegion(
        "initial_state_copy", output.byte_offset, 16384, 0, output.live_from, 1024
    )
    if planes_per_transfer == 2:
        assert reuse is not None
        _, scratch = reuse.resources(config.PIPELINE_PLAN)
    return RawStateCopyPlan(
        schedule,
        lease,
        config.PIPELINE_PLAN,
        scratch,
        MBarrierRegion("initial_state_copy", config.DONE + 8, 4 if pipelined else 3, 1),
        64,
        startup_reuse=reuse,
        planes_per_transfer=planes_per_transfer,
    )
