"""Original residual/solved-value publications of the fast recurrence worker."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Literal

from . import chained_matmul as chain
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
class FastStateProductArrival:
    cycle: FastStateProductCycle

    @property
    def owner(self) -> FastRecurrence:
        return self.cycle.owner

    def facts(self) -> object:
        return (
            id(self.owner),
            self.cycle.kind,
            self.cycle.mask_tail,
            config.PIPELINE_PLAN.role("state"),
            config.STAGES,
            (config.V_FREE, config.U_INP_READY)
            if self.cycle.kind == "rhs"
            else (config.U2_INP_READY,),
            self.owner.inverse_product.rhs,
            self.owner.output_update.rhs,
        )


@dataclass(frozen=True)
class FastStateProductBinding:
    cycle: FastStateProductCycle
    half: int

    @property
    def source(self) -> Node:
        owner = self.cycle.owner
        return (
            owner.projection.node
            if self.cycle.kind == "rhs"
            else owner.inverse_product.node
        )

    @property
    def destination(self) -> Node:
        owner = self.cycle.owner
        return (
            owner.inverse_product.rhs
            if self.cycle.kind == "rhs"
            else owner.output_update.rhs
        )

    @property
    def offset(self) -> int:
        return max(self.half, 0) * 8

    @property
    def width(self) -> int:
        return 8 if self.cycle.kind == "rhs" else 16

    def facts(self) -> object:
        return (
            id(self.cycle),
            self.source,
            self.destination,
            self.half,
            self.cycle.mask_tail,
        )

    def state_view(self, *, store: bool = False) -> StateView:
        from .prepared_state_planner import StateView

        return StateView(
            self.destination if store else self.source,
            self.cycle.scope,
            (
                config.PIPELINE_PLAN,
                "32x32b",
                "warp%4",
                224 if store or self.cycle.kind == "rhs" else 0,
            ),
            self.offset if store else 0,
            self.width if store else 32,
        )

    def transfer(self) -> StateTransfer:
        return StateTransfer(
            "source", "values", "destination", "None", True, "32x32b", 32, "32x32b"
        )

    def check_publication(self, request: StatePublication) -> None:
        if (
            self.half < 0
            or request.read is not self.cycle.read
            or request.value is not self
            or request.transform_before is not self.cycle.ready
            or request.store_before is not self.cycle.ready
            or request.complete_before is not self.cycle.ready
            or request.read_before is not None
        ):
            raise chain._UnsupportedChain("changed original state operand publication")


@dataclass
class FastStateProductCycle:
    owner: FastRecurrence
    kind: Literal["rhs", "update"]
    mask_tail: bool
    read: FastStateProductBinding = field(init=False)
    ready: StateCut = field(init=False)

    @property
    def scope(self) -> tuple[object, ...]:
        return id(self.owner), self.kind, self.mask_tail

    def program(self) -> tuple[tuple[object, ...], ...]:
        from .prepared_state_planner import StateCut
        from .prepared_state_planner import StatePublication
        from .prepared_state_planner import plan_state_transfers

        self.owner.check()
        self.ready = StateCut(self.owner, FastStateProductArrival(self), self.scope)
        self.read = FastStateProductBinding(self, -1)
        values = tuple(
            FastStateProductBinding(self, half)
            for half in range(2 if self.kind == "rhs" else 1)
        )
        schedule = plan_state_transfers(
            tuple(
                StatePublication(
                    self.read,
                    value,
                    self.read.state_view(),
                    value.state_view(store=True),
                    self.ready,
                    self.ready,
                    self.ready,
                )
                for value in values
            ),
            (self.ready,),
        )
        schedule.check()
        result: list[tuple[object, ...]] = []
        for effect in schedule.actions:
            binding = effect.binding
            assert isinstance(binding, FastStateProductBinding)
            if effect.kind == "transform":
                result.append(
                    (3, binding.half, self.mask_tail, 128)
                    if self.kind == "rhs"
                    else (4,)
                )
            else:
                opcode, *geometry = state_transfer_instruction(
                    effect.kind, binding.transfer()
                )
                result.append((opcode, binding.offset, binding.width, *geometry))
        result.append((5, 2 if self.kind == "rhs" else 1))
        return tuple(result)


def bind_fast_state_products(
    owner: FastRecurrence,
) -> tuple[tuple[tuple[object, ...], ...], ...]:
    owner.check()
    return (
        FastStateProductCycle(owner, "rhs", False).program(),
        FastStateProductCycle(owner, "rhs", True).program(),
        FastStateProductCycle(owner, "update", False).program(),
    )
