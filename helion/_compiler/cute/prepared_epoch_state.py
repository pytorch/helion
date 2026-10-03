"""Descriptor state ports for the original matched role epoch.

The native map is the original full Layout-F owner, not a synthetic root
accumulator. StateEffect and its read/store/completion lowering are shared with
the initialized root adapter. Ring waits and arrivals belong to the enclosing
role walk, which may interleave them with these state effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Literal

import torch

from . import chained_matmul as chain
from .chained_completed_store import _records
from .prepared_state_body import StateEffect
from .prepared_state_body import StateTransfer
from .prepared_state_body import emit_state_transfer

if TYPE_CHECKING:
    from torch.fx import Node

    from .prepared_epoch_protocol import EpochEvent
    from .prepared_state_planner import StatePublication
    from .prepared_state_planner import StateTransferPlan
    from .prepared_state_planner import StateView
    from .prepared_tcgen_binding import PreparedProjectionHost


@dataclass(frozen=True)
class DescriptorStateBinding:
    """One exact value/view of a same-generation full state half."""

    cycle: DescriptorStateCycle
    value: Literal["original", "packed", "scaled"]
    _facts: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate()
        object.__setattr__(self, "_facts", self._current())

    @property
    def owner(self) -> PreparedProjectionHost:
        return self.cycle.owner

    @property
    def role(self) -> Literal["right", "left"]:
        return self.cycle.role

    @property
    def first(self) -> bool:
        return self.cycle.first

    @property
    def half(self) -> int:
        return self.cycle.half

    @property
    def source(self) -> Node:
        epoch = self.owner.plan.prepared_epoch
        assert epoch is not None
        return epoch.projection.state

    @property
    def destination(self) -> Node:
        epoch = self.owner.plan.prepared_epoch
        assert epoch is not None
        if self.value == "packed":
            return epoch.projected.spec.rhs
        if self.value == "original":
            return self.source
        # This positional relation has already passed the original complete
        # numerical matcher; no replacement or synthetic operation is made.
        result = next(
            iter(epoch.continuation.region.graph.find_nodes(op="output"))
        ).args[0]
        assert isinstance(result, (tuple, list)) and len(result) == 1
        carried = result[0]
        assert isinstance(carried, torch.fx.Node)
        scaled = carried.args[0]
        assert isinstance(scaled, torch.fx.Node)
        return scaled

    @property
    def prefix(self) -> str:
        return f"epoch_state_{self.role}_{int(self.first)}_{self.half}"

    @property
    def offset(self) -> int:
        return self.half * 64

    @property
    def width(self) -> int:
        return 64

    def _validate(self) -> None:
        from .chunk_recurrence import validate_epoch_match

        validate_epoch_match(self.owner.plan)
        self.owner.check()
        if (
            self.role not in ("right", "left")
            or type(self.first) is not bool
            or self.role == "left"
            and (self.first or self.half != 0)
            or self.role == "right"
            and not self.first
            and self.half != 1
        ):
            raise chain._UnsupportedChain(
                "state half has a foreign original reader role"
            )
        if (
            type(self.half) is not int
            or self.half not in (0, 1)
            or self.value not in ("original", "packed", "scaled")
        ):
            raise chain._UnsupportedChain("invalid original state half binding")
        if self.owner.plan.state.fake.dtype is not torch.float32:
            raise chain._UnsupportedChain(
                "descriptor state requires original FP32 carry"
            )

    def _current(self) -> object:
        from . import chunk_recurrence_sm100 as original

        return (
            id(self.cycle),
            id(self.owner),
            id(self.owner.plan),
            self.role,
            self.first,
            self.half,
            self.value,
            self.source,
            self.destination,
            _records((original._TMEM_LAYOUT, original.ROLES)),
            original.TCGEN05_STATE_K_TMEM_ROW_BLOCKS,
            original.THREADS_PER_WARP,
            original.TCGEN05_STATE_INPUT_LOAD_COLS,
            original.TCGEN05_STATE_INPUT_PACKED_COLS,
        )

    def facts(self) -> object:
        self._validate()
        if self._current() != self._facts:
            raise chain._UnsupportedChain("original state owner or map changed")
        return id(self), self._facts

    def transfer(self) -> StateTransfer:
        self.facts()
        prefix = self.prefix
        packed = self.value == "packed"
        values = f"{prefix}_{self.value}"
        if self.value != "original":
            values += "[0:16]" if packed else "[0:32]"
        return StateTransfer(
            f"{prefix}_source",
            values,
            f"{prefix}_packed_target" if packed else f"{prefix}_source",
            "None",
            True,
            "16x256b",
            8,
            "16x128b" if packed else "16x256b",
        )

    def state_view(self, *, store: bool = False) -> StateView:
        from . import chunk_recurrence_sm100 as original
        from .prepared_state_planner import StateView

        packed = store and self.value == "packed"
        region = original._TMEM_LAYOUT.region("state_input" if packed else "state")
        return StateView(
            region,
            (id(self.owner), self.role, self.first, self.half),
            (
                original._TMEM_LAYOUT,
                original.TCGEN05_STATE_K_TMEM_ROW_BLOCKS,
                original.THREADS_PER_WARP,
                original.TCGEN05_STATE_INPUT_PACKED_COLS
                if packed
                else original.TCGEN05_STATE_INPUT_LOAD_COLS,
            ),
            self.half * (32 if packed else 64),
            32 if packed else 64,
        )

    def check_publication(self, request: StatePublication) -> None:
        if request.read_before is not None:
            raise chain._UnsupportedChain("original result has no early capture cut")

        cycle = self.cycle
        packed_cut = (
            cycle.arrival
            if cycle.first and cycle.half == 1
            else cycle.coefficient_ready
        )
        expected = (
            (packed_cut, packed_cut, cycle.arrival)
            if self.value == "packed"
            else (cycle.arrival, cycle.scaled.destination, cycle.scaled.destination)
        )
        if (
            self.value == "original"
            or request.read is not cycle.original
            or request.value is not self
            or request.complete_before is None
            or any(
                cut.owner is not self.owner or cut.anchor is not anchor
                for cut, anchor in zip(
                    (
                        request.transform_before,
                        request.store_before,
                        request.complete_before,
                    ),
                    expected,
                    strict=True,
                )
            )
        ):
            raise chain._UnsupportedChain("changed original state dependency cut")

    def views(self) -> list[str]:
        from . import chunk_recurrence_sm100 as original

        self.facts()
        prefix = self.prefix
        state = original._TMEM_LAYOUT.region("state")
        packed = original._TMEM_LAYOUT.region("state_input")
        if state.columns != 128 or packed.columns != 64:
            raise chain._UnsupportedChain("original complete state owner changed")
        # Exact original scalar address equations, including the different
        # row bases of the 16x256b read and 16x128b packed publication.
        return [
            f"{prefix}_base_col = tmem_raw_addr & 0xFFFF",
            f"{prefix}_base_row = tmem_raw_addr >> 16",
            f"{prefix}_row = ({prefix}_base_row + (warp_idx % {original.TCGEN05_STATE_K_TMEM_ROW_BLOCKS}) * {original.THREADS_PER_WARP}) << 16",
            f"{prefix}_source = cutlass.inttoptr({prefix}_row | ({prefix}_base_col + {state.column_offset} + {self.offset}), 6, cutlass.Float32)",
            f"{prefix}_packed_target = prims.make_tmem_ptr(({prefix}_base_row << 16) | ({prefix}_base_col + {packed.column_offset} + {self.half * 32}), cutlass.Int8)",
        ]


def emit_descriptor_state_action(effect: StateEffect) -> list[str]:
    binding = effect.binding
    if not isinstance(binding, DescriptorStateBinding):
        raise chain._UnsupportedChain("foreign descriptor state binding")
    binding.facts()
    prefix = binding.prefix
    if effect.kind == "read":
        if binding.value != "original":
            raise chain._UnsupportedChain("state snapshot must read original FP32")
        return [*binding.views(), *emit_state_transfer("read", binding.transfer())]
    if effect.kind == "transform":
        if binding.value == "packed":
            return [
                f"{prefix}_packed = prepared_tcgen_edge.pack_layout_f_state({prefix}_original, input_dtype)"
            ]
        if binding.value == "scaled":
            return [
                f"{prefix}_scaled = prepared_tcgen_edge.scale_layout_f_state({prefix}_original, diag_raw_stage, {binding.offset})"
            ]
        raise chain._UnsupportedChain("unknown original state transform")
    if effect.kind in ("store", "complete"):
        if binding.value == "original":
            raise chain._UnsupportedChain("untransformed state cannot replace a seed")
        return emit_state_transfer(effect.kind, binding.transfer())
    raise chain._UnsupportedChain("unknown descriptor state effect")


@dataclass(frozen=True)
class StateArrival:
    """The original four-warp packed-store contribution, after completion."""

    binding: DescriptorStateBinding

    def facts(self) -> object:
        if self.binding.value != "packed":
            raise chain._UnsupportedChain("state readiness requires packed completion")
        return id(self), self.binding.facts()

    def emit(self) -> list[str]:
        self.facts()
        event = (
            "state_input_ready_l_mbar"
            if self.binding.half == 0
            else "state_input_ready_mbar"
        )
        return [f"state_input_ready_arrive({event})"]


@dataclass(frozen=True)
class DescriptorStateCycle:
    """Original state SSA: live FP32 -> packed store, then live FP32 -> decay.

    The enclosing role walk places its original waits between the two spans.
    Keeping these spans separate is essential: the first packed store precedes
    raw readiness and remains outstanding during the FP32 decay calculation.
    """

    owner: PreparedProjectionHost
    role: Literal["right", "left"]
    first: bool
    half: int
    coefficient_ready: EpochEvent
    original: DescriptorStateBinding = field(init=False)
    packed: DescriptorStateBinding = field(init=False)
    scaled: DescriptorStateBinding = field(init=False)
    arrival: StateArrival = field(init=False)
    actions: tuple[StateEffect | StateArrival, ...] = field(init=False)
    _actions: tuple[StateEffect | StateArrival, ...] = field(init=False, repr=False)
    schedule: StateTransferPlan = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from .prepared_state_planner import StateCut
        from .prepared_state_planner import StatePublication
        from .prepared_state_planner import plan_state_transfers

        for value in ("original", "packed", "scaled"):
            object.__setattr__(self, value, DescriptorStateBinding(self, value))
        arrival = StateArrival(self.packed)
        object.__setattr__(self, "arrival", arrival)
        scope = (id(self.owner), self.role, self.first, self.half)
        raw = StateCut(self.owner, self.coefficient_ready, scope)
        ready = StateCut(self.owner, arrival, scope)
        end = StateCut(self.owner, self.scaled.destination, scope)
        already_ready = self.first and self.half == 1
        packed_cut = ready if already_ready else raw
        source = self.original.state_view()
        schedule = plan_state_transfers(
            (
                StatePublication(
                    self.original,
                    self.scaled,
                    source,
                    self.scaled.state_view(store=True),
                    ready,
                    end,
                    end,
                ),
                StatePublication(
                    self.original,
                    self.packed,
                    source,
                    self.packed.state_view(store=True),
                    packed_cut,
                    packed_cut,
                    ready,
                ),
            ),
            (ready, end) if already_ready else (raw, ready, end),
        )
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(
            self,
            "actions",
            (
                *(action for phase in schedule.phases[:-1] for action in phase),
                arrival,
                *schedule.phases[-1],
            ),
        )
        object.__setattr__(self, "_actions", self.actions)

    def check_actions(self) -> None:
        from .prepared_continuation import ContinuationOpcode
        from .prepared_epoch_protocol import EpochPort
        from .prepared_epoch_protocol import EpochPosition

        event = self.coefficient_ready
        if (
            event.owner is not self.owner
            or event.port is not EpochPort.RAW_READY
            or event.position
            is not (EpochPosition.FIRST if self.first else EpochPosition.CURRENT)
            or event.operation is not ContinuationOpcode.WAIT_TOGGLE
        ):
            raise chain._UnsupportedChain("state lost original coefficient readiness")
        self.schedule.check()
        if self.actions is not self._actions:
            raise chain._UnsupportedChain("original state action identities changed")
        planned = (
            *(action for phase in self.schedule.phases[:-1] for action in phase),
            self.arrival,
            *self.schedule.phases[-1],
        )
        if len(planned) != len(self.actions) or any(
            a is not b for a, b in zip(planned, self.actions, strict=True)
        ):
            raise chain._UnsupportedChain("original state lost planned actions")

    def pack(self) -> tuple[StateEffect | StateArrival, ...]:
        self.check_actions()
        return self.schedule.phases[0]

    def decay(self) -> tuple[StateEffect | StateArrival, ...]:
        self.check_actions()
        return self.actions[len(self.schedule.phases[0]) :]
