"""Bind exact external FP32 state transfers to the original root graph."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import operator
from typing import TYPE_CHECKING

import torch

from ...language._tracing_ops import _for_loop
from ...language._tracing_ops import _phi
from . import chained_matmul as chain
from .chained_completed_store import _records
from .fx_matcher import _load_ref
from .fx_matcher import _same_ref
from .fx_matcher import _store_ref
from .prepared_state_body import StateTransfer
from .prepared_state_body import state_transfer_instruction

if TYPE_CHECKING:
    from torch.fx import Node

    from ..device_ir import GraphInfo
    from .chunk_prefill import CuteChunkPrefillRegion
    from .prepared_state_planner import StateCut
    from .prepared_state_planner import StatePublication
    from .prepared_state_planner import StateView


@dataclass(frozen=True)
class FastStateABI:
    region: CuteChunkPrefillRegion
    root: GraphInfo
    initial: Node
    loop: Node
    final_value: Node
    final_store: Node
    facts: object

    def check(self) -> None:
        if (
            self.root.graph_id != self.region.root_graph_id
            or any(
                node.graph is not self.root.graph
                for node in (
                    self.initial,
                    self.loop,
                    self.final_value,
                    self.final_store,
                )
            )
            or not _same_ref(_load_ref(self.initial), self.region.initial_state)
            or not _same_ref(_store_ref(self.final_store), self.region.final_state)
            or _records(
                tuple(
                    (node, node.target, node.args, node.kwargs)
                    for node in self.root.graph.nodes
                )
            )
            != self.facts
        ):
            raise chain._UnsupportedChain("changed original external state ABI")


@dataclass(frozen=True)
class FastStateABIBinding:
    cycle: FastStateABICycle
    panel: int

    @property
    def source(self) -> Node:
        return (
            self.cycle.owner.final_value
            if self.cycle.mode == 1
            else self.cycle.owner.initial
        )

    @property
    def destination(self) -> Node:
        return (
            self.cycle.owner.loop
            if self.cycle.mode == 0
            else self.cycle.owner.final_store
        )

    @property
    def offset(self) -> int:
        return self.panel * 32

    @property
    def width(self) -> int:
        return 32

    def facts(self) -> object:
        self.cycle.owner.check()
        return (
            id(self.cycle),
            self.cycle.mode,
            self.panel,
            self.source,
            self.destination,
        )

    def state_view(self, *, store: bool = False) -> StateView:
        from .prepared_state_planner import StateView

        owner = self.cycle.owner
        tmem = self.cycle.mode == (0 if store else 1)
        storage = (
            owner.loop
            if tmem
            else owner.region.final_state
            if store
            else owner.region.initial_state
        )
        return StateView(
            storage,
            self.cycle.scope,
            ("32x32b", "warp%4", 64)
            if tmem
            else ("external-fp32", "sequence", "head", "value", 128),
            self.offset,
            self.width,
        )

    def transfer(self) -> StateTransfer:
        return StateTransfer(
            "source", "values", "destination", "None", True, "32x32b", 32, "32x32b"
        )

    def check_publication(self, request: StatePublication) -> None:
        if (
            request.read is not self
            or request.value is not self
            or request.transform_before is not self.cycle.cut
            or request.store_before is not self.cycle.cut
            or request.complete_before
            is not (self.cycle.cut if self.cycle.mode == 0 else None)
            or request.read_before is not None
        ):
            raise chain._UnsupportedChain(
                "changed original external state transfer cut"
            )


@dataclass
class FastStateABICycle:
    owner: FastStateABI
    mode: int
    cut: StateCut = field(init=False)

    @property
    def scope(self) -> tuple[object, ...]:
        return id(self.owner), "external-state", self.mode

    def program(self) -> tuple[tuple[object, ...], ...]:
        from .prepared_state_planner import StateCut
        from .prepared_state_planner import StatePublication
        from .prepared_state_planner import plan_state_transfers

        self.owner.check()
        self.cut = StateCut(
            self.owner,
            self.owner.loop if self.mode == 0 else self.owner.final_store,
            self.scope,
        )
        bindings = tuple(FastStateABIBinding(self, panel) for panel in range(4))
        schedule = plan_state_transfers(
            tuple(
                StatePublication(
                    binding,
                    binding,
                    binding.state_view(),
                    binding.state_view(store=True),
                    self.cut,
                    self.cut,
                    self.cut if self.mode == 0 else None,
                )
                for binding in bindings
            ),
            (self.cut,),
        )
        schedule.check()
        result: list[tuple[object, ...]] = []
        for effect in schedule.actions:
            binding = effect.binding
            assert isinstance(binding, FastStateABIBinding)
            # External ABI transfer has no conversion: every FP32 bit is kept.
            if effect.kind != "transform":
                opcode, *geometry = state_transfer_instruction(
                    effect.kind, binding.transfer()
                )
                result.append((opcode, self.mode, binding.panel, *geometry))
        return tuple(result)


def bind_fast_state_abi(
    region: CuteChunkPrefillRegion,
    semantic_root: GraphInfo,
) -> tuple[tuple[tuple[object, ...], ...], ...]:
    if region.numerical_policy != "centered_bt32_fp32_rhs_v2" or (
        region.key_width,
        region.value_width,
    ) != (128, 128):
        raise chain._UnsupportedChain("foreign external state geometry")
    return bind_external_state_abi(region, semantic_root)


def bind_external_state_abi(
    region: CuteChunkPrefillRegion,
    semantic_root: GraphInfo,
) -> tuple[tuple[tuple[object, ...], ...], ...]:
    """Original root FP32 state ports after the physical adapter's admission."""
    from torch.fx import Node

    if (region.key_width, region.value_width) != (128, 128):
        raise chain._UnsupportedChain("foreign external state geometry")
    loads = [
        node
        for node in semantic_root.graph.nodes
        if _same_ref(_load_ref(node), region.initial_state)
    ]
    stores = [
        node
        for node in semantic_root.graph.nodes
        if _same_ref(_store_ref(node), region.final_state)
    ]
    if len(loads) != 1 or len(stores) != 1:
        raise chain._UnsupportedChain("missing original external state nodes")
    initial, final_store = loads[0], stores[0]
    final_value = final_store.args[2]
    if (
        not isinstance(final_value, Node)
        or final_value.target is not _phi
        or final_value.args[0] is not initial
        or not isinstance(final_value.args[1], Node)
        or final_value.args[1].target is not operator.getitem
        or final_value.args[1].args[1] != 0
    ):
        raise chain._UnsupportedChain("changed original final/zero-trip state value")
    loop = final_value.args[1].args[0]
    if (
        not isinstance(loop, Node)
        or loop.target is not _for_loop
        or loop.args[0] != region.loop_graph_id
        or not isinstance(loop.args[3], (list, tuple))
        or not loop.args[3]
        or loop.args[3][-1] is not initial
        or region.initial_state.fake.dtype is not torch.float32
        or region.final_state.fake.dtype is not torch.float32
    ):
        raise chain._UnsupportedChain("changed original initial-state precision/owner")
    owner = FastStateABI(
        region,
        semantic_root,
        initial,
        loop,
        final_value,
        final_store,
        _records(
            tuple(
                (node, node.target, node.args, node.kwargs)
                for node in semantic_root.graph.nodes
            )
        ),
    )
    return tuple(FastStateABICycle(owner, mode).program() for mode in range(3))
