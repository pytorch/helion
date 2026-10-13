"""Bind original output values and lease cuts to common transfer effects."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import torch

from . import chained_matmul as chain
from .chained_completed_store import _records
from .chunk_prefill_bt32 import config
from .prepared_state_body import StateTransfer
from .prepared_state_body import state_transfer_instruction

if TYPE_CHECKING:
    from torch.fx import Node

    from .chunk_prefill_prepared_issue import FastRecurrence
    from .prepared_state_planner import StateCut
    from .prepared_state_planner import StatePublication
    from .prepared_state_planner import StateView


@dataclass(frozen=True)
class FastOutputBoundary:
    cycle: FastOutputCycle
    kind: str

    @property
    def owner(self) -> FastRecurrence:
        return self.cycle.owner

    def facts(self) -> object:
        return (
            id(self.owner),
            self.owner.region.step.output_store,
            config.PIPELINE_PLAN.role("output"),
            config.PIPELINE_PLAN.barrier("output_empty"),
            config.PIPELINE_PLAN.shared_buffer("output_stages"),
            self.cycle.full,
            self.kind,
        )


@dataclass(frozen=True)
class FastOutputBinding:
    cycle: FastOutputCycle
    half: int

    @property
    def source(self) -> Node:
        return self.cycle.owner.output_update.node

    @property
    def destination(self) -> Node:
        value = self.cycle.owner.region.step.output_store.args[2]
        assert isinstance(value, torch.fx.Node)
        return value

    @property
    def offset(self) -> int:
        return max(self.half, 0) * 16

    @property
    def width(self) -> int:
        return 16 if self.cycle.full else 32

    def facts(self) -> object:
        return (
            id(self.cycle),
            id(self.cycle.owner),
            self.source,
            self.destination,
            self.half,
            self.cycle.full,
            _records(self.cycle.owner.region.output),
        )

    def state_view(self, *, store: bool = False) -> StateView:
        from .prepared_state_planner import StateView

        if store:
            return StateView(
                self.cycle.owner.region.output,
                self.cycle.scope,
                (
                    config.PIPELINE_PLAN,
                    "sw128-stmatrix" if self.cycle.full else "masked-global",
                    self.cycle.full,
                ),
                self.offset,
                self.width,
            )
        return StateView(
            self.source,
            self.cycle.scope,
            (
                config.PIPELINE_PLAN,
                "16x256b-pair" if self.cycle.full else "32x32b",
                192,
                "warp%4",
            ),
            0,
            32,
        )

    def transfer(self) -> StateTransfer:
        return StateTransfer(
            "source",
            "values",
            "destination",
            "None",
            True,
            "16x256b" if self.cycle.full else "32x32b",
            4 if self.cycle.full else 32,
            None,
        )

    def check_publication(self, request: StatePublication) -> None:
        cycle = self.cycle
        if (
            self.half < 0
            or request.read is not cycle.read
            or request.value is not self
            or request.read_before is not cycle.release
            or request.transform_before is not cycle.store
            or request.store_before is not cycle.store
            or request.complete_before is not (cycle.store if cycle.full else None)
        ):
            raise chain._UnsupportedChain(
                "changed original output capture/publication cut"
            )


@dataclass
class FastOutputCycle:
    owner: FastRecurrence
    full: bool
    read: FastOutputBinding = field(init=False)
    release: StateCut = field(init=False)
    space: StateCut = field(init=False)
    store: StateCut = field(init=False)

    @property
    def scope(self) -> tuple[object, ...]:
        return id(self.owner), "output", self.full

    def program(self) -> tuple[tuple[object, ...], ...]:
        from .prepared_state_planner import StateCut
        from .prepared_state_planner import StatePublication
        from .prepared_state_planner import plan_state_transfers

        self.owner.check()
        for kind in ("release", "space", "store"):
            setattr(
                self,
                kind,
                StateCut(self.owner, FastOutputBoundary(self, kind), self.scope),
            )
        self.read = FastOutputBinding(self, -1)
        values = tuple(
            FastOutputBinding(self, half) for half in range(2 if self.full else 1)
        )
        schedule = plan_state_transfers(
            tuple(
                StatePublication(
                    self.read,
                    value,
                    self.read.state_view(),
                    value.state_view(store=True),
                    self.store,
                    self.store,
                    self.store if self.full else None,
                    read_before=self.release,
                )
                for value in values
            ),
            (self.release, self.space, self.store),
        )
        schedule.check()
        result: list[tuple[object, ...]] = []
        for index, phase in enumerate(schedule.phases):
            for effect in phase:
                binding = effect.binding
                assert isinstance(binding, FastOutputBinding)
                if effect.kind == "read":
                    opcode, *geometry = state_transfer_instruction(
                        "read", binding.transfer()
                    )
                    result.append((opcode, *geometry, self.full))
                elif effect.kind == "transform":
                    # A masked scalar conversion stays fused with its store;
                    # preconverting invalid lanes would change the original cut.
                    if self.full:
                        result.append((3, binding.half))
                elif effect.kind == "store":
                    result.append(
                        (1, binding.half, 128, 64, 32, 7) if self.full else (6, 32)
                    )
                else:
                    result.append((2, 8, 128, 32))
            if index == 0:
                result.append((4, 8, 128))
            elif index == 1 and self.full:
                result.append((5, 8, 128, 2, 1))
        return tuple(result)


def bind_fast_output(
    owner: FastRecurrence,
) -> tuple[tuple[tuple[object, ...], ...], ...]:
    """Keep the original store, two-slot transport, and capture-before-release."""
    owner.check()
    store = owner.region.step.output_store
    value = store.args[2]
    if (
        owner.region.numerical_policy != "centered_bt32_fp32_rhs_v2"
        or (owner.region.chunk_size, owner.region.value_width) != (32, 128)
        or not isinstance(value, torch.fx.Node)
        or value.target != torch.ops.prims.convert_element_type.default
        or value.args != (owner.output_update.node, torch.bfloat16)
        or config.PIPELINE_PLAN.role("output").registers_per_thread != 48
    ):
        raise chain._UnsupportedChain("foreign original output conversion or geometry")
    return tuple(FastOutputCycle(owner, full).program() for full in (True, False))
