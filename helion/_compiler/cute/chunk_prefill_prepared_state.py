"""Original whole-root loop state ports for the shared state-transfer planner."""

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
class FastLoopStateBinding:
    panel: FastLoopStatePanel
    value: Literal["original", "packed", "scaled"]

    @property
    def source(self) -> Node:
        return self.panel.owner.state_input

    @property
    def destination(self) -> Node:
        owner = self.panel.owner
        if self.value == "packed":
            return owner.packed_state_nodes[0]
        return owner.scaled_state if self.value == "scaled" else self.source

    @property
    def offset(self) -> int:
        return 0

    @property
    def width(self) -> int:
        return 32

    def facts(self) -> object:
        return (
            id(self.panel),
            id(self.panel.owner),
            self.source,
            self.destination,
            self.panel.owner.packed_state_nodes,
            self.value,
            self.panel.publish,
            config.PIPELINE_PLAN,
            config.STATE_INP_READY,
        )

    def state_view(self, *, store: bool = False) -> StateView:
        from .prepared_state_planner import StateView

        packed = store and self.value == "packed"
        # The full native owner is retained. The symbolic panel belongs to the
        # existing sequential panel loop; the planner schedules one iteration.
        return StateView(
            self.panel.owner.packed_state_nodes[0] if packed else self.source,
            self.panel.scope,
            (
                config.PIPELINE_PLAN,
                "32x32b",
                "warp%4",
                "panel",
                16 if packed else 32,
                0 if packed else 64,
            ),
            0,
            16 if packed else 32,
        )

    def transfer(self) -> StateTransfer:
        values = {
            "original": "original",
            "packed": "packed[0:16]",
            "scaled": "scaled[0:32]",
        }[self.value]
        return StateTransfer(
            "source",
            values,
            "packed_target" if self.value == "packed" else "source",
            "None",
            True,
            "32x32b",
            32,
            "32x32b",
        )

    def check_publication(self, request: StatePublication) -> None:
        if request.read_before is not None:
            raise chain._UnsupportedChain("original state has no early capture cut")

        panel = self.panel
        expected_cut = panel.packed_cut if self.value == "packed" else panel.scaled_cut
        expected_completion = (
            panel.packed_cut if self.value == "packed" and panel.publish else None
        )
        if (
            self.value == "original"
            or request.read is not panel.original
            or request.value is not self
            or request.transform_before is not expected_cut
            or request.store_before is not expected_cut
            or request.complete_before is not expected_completion
        ):
            raise chain._UnsupportedChain("changed original loop state publication cut")


@dataclass(frozen=True)
class FastLoopStateArrival:
    panel: FastLoopStatePanel

    @property
    def owner(self) -> FastRecurrence:
        return self.panel.owner

    def facts(self) -> object:
        return (
            id(self),
            id(self.owner),
            self.owner.packed_state_nodes,
            config.STATE_INP_READY,
            config.STAGES,
            config.PIPELINE_PLAN.barrier("factor_team"),
            config.PIPELINE_PLAN.role("state"),
        )


@dataclass
class FastLoopStatePanel:
    owner: FastRecurrence
    publish: bool
    original: FastLoopStateBinding = field(init=False)
    packed: FastLoopStateBinding = field(init=False)
    scaled: FastLoopStateBinding = field(init=False)
    packed_cut: StateCut = field(init=False)
    scaled_cut: StateCut = field(init=False)

    @property
    def scope(self) -> tuple[object, ...]:
        return (id(self.owner), "state", "current_panel", self.publish)

    def program(self) -> tuple[tuple[object, ...], ...]:
        from .prepared_state_planner import StateCut
        from .prepared_state_planner import StatePublication
        from .prepared_state_planner import plan_state_transfers

        self.owner.check()
        for value in ("original", "packed", "scaled"):
            setattr(self, value, FastLoopStateBinding(self, value))
        arrival = FastLoopStateArrival(self)
        self.packed_cut = StateCut(
            self.owner,
            arrival if self.publish else self.packed.destination,
            self.scope,
        )
        self.scaled_cut = StateCut(self.owner, self.scaled.destination, self.scope)
        source = self.original.state_view()
        schedule = plan_state_transfers(
            (
                StatePublication(
                    self.original,
                    self.scaled,
                    source,
                    self.scaled.state_view(store=True),
                    self.scaled_cut,
                    self.scaled_cut,
                    None,
                ),
                StatePublication(
                    self.original,
                    self.packed,
                    source,
                    self.packed.state_view(store=True),
                    self.packed_cut,
                    self.packed_cut,
                    self.packed_cut if self.publish else None,
                ),
            ),
            (self.packed_cut, self.scaled_cut),
        )
        schedule.check()
        program: list[tuple[object, ...]] = []
        for index, phase in enumerate(schedule.phases):
            for effect in phase:
                binding = effect.binding
                assert isinstance(binding, FastLoopStateBinding)
                value = ("original", "packed", "scaled").index(binding.value)
                if effect.kind == "transform":
                    program.append((3 if binding.value == "packed" else 4, value))
                else:
                    opcode, *geometry = state_transfer_instruction(
                        effect.kind, binding.transfer()
                    )
                    program.append((opcode, value, *geometry))
            if index == 0 and self.publish:
                program.append((5,))
        return tuple(program)


def bind_fast_state(
    owner: FastRecurrence,
) -> tuple[tuple[tuple[object, ...], ...], ...]:
    """Lower real loop-carry fanout into the original regular/final panel cuts."""
    from .chunk_prefill_prepared_state_products import bind_fast_state_products

    owner.check()
    region = owner.region
    if (
        region.numerical_policy != "centered_bt32_fp32_rhs_v2"
        or (region.chunk_size, region.key_width, region.value_width) != (32, 128, 128)
        or config.PIPELINE_PLAN.role("state").registers_per_thread != 144
    ):
        raise chain._UnsupportedChain("foreign original whole-root state geometry")
    return tuple(
        FastLoopStatePanel(owner, publish).program() for publish in (False, True)
    ) + bind_fast_state_products(owner)
