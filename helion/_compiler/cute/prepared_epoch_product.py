"""Original result/operand ports consumed by the common state-effect lowering.

The residual and output points are the original native arithmetic-only leaves.
Their accumulator and companion reads, publication and completion are separate
actions, so the original ring releases/waits remain visible to BodyProgram.
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
    from .prepared_epoch_protocol import EpochSynchronizationAction
    from .prepared_state_planner import StatePublication
    from .prepared_state_planner import StateTransferPlan
    from .prepared_state_planner import StateView
    from .prepared_tcgen_binding import PreparedProjectionHost


@dataclass(frozen=True)
class DescriptorProductBinding:
    cycle: DescriptorProductCycle
    _facts: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_facts", self._current())

    @property
    def owner(self) -> PreparedProjectionHost:
        return self.cycle.owner

    @property
    def source(self) -> Node:
        epoch = self.owner.plan.prepared_epoch
        assert epoch is not None
        return (
            epoch.projection.source
            if self.cycle.kind == "residual"
            else epoch.continuation.second.node
        )

    def _relation(self) -> tuple[Node, object]:
        from .chunk_recurrence import _ancestors
        from .chunk_recurrence import _load_ref
        from .chunk_recurrence import _output_store_contract
        from .chunk_recurrence import _store_ref

        epoch = self.owner.plan.prepared_epoch
        assert epoch is not None
        if self.cycle.kind == "residual":
            destination = epoch.projection.operand
            # The complete original matcher establishes the residual expression;
            # retain its actual V load, not a same-shaped or named substitute.
            loads = [
                node
                for node in _ancestors(destination)
                if (ref := _load_ref(node)) is not None
                and _records(ref) == _records(self.owner.plan.values)
            ]
            if len(loads) != 1:
                raise chain._UnsupportedChain("residual lost original raw-V input")
            return destination, loads[0]
        contract = _output_store_contract(
            list(epoch.continuation.region.graph.nodes), self.source
        )
        if contract is None:
            raise chain._UnsupportedChain("output lost original store contract")
        store, scale = contract
        if (
            _records(_store_ref(store)) != _records(self.owner.plan.output)
            or _records(scale) != _records(self.owner.plan.output_scale)
            or not isinstance(store.args[2], torch.fx.Node)
        ):
            raise chain._UnsupportedChain("output changed original destination/scale")
        return store.args[2], store

    @property
    def destination(self) -> Node:
        return self._relation()[0]

    @property
    def prefix(self) -> str:
        return f"epoch_{self.cycle.kind}_{int(self.cycle.first)}"

    @property
    def offset(self) -> int:
        return 0

    @property
    def width(self) -> int:
        from . import chunk_recurrence_sm100 as original

        return original.BT

    def _current(self) -> object:
        from . import chunk_recurrence_sm100 as original
        from .chunk_recurrence import validate_epoch_match

        self.owner.check()
        validate_epoch_match(self.owner.plan)
        if (
            self.cycle.kind not in ("residual", "output")
            or type(self.cycle.first) is not bool
            or self.cycle.kind == "output"
            and self.cycle.first
        ):
            raise chain._UnsupportedChain("invalid original result generation")
        return (
            id(self.cycle),
            id(self.owner),
            self.cycle.kind,
            self.cycle.first,
            self.source,
            self._relation(),
            _records((original._TMEM_LAYOUT, original.ROLES)),
            tuple(
                (name, value)
                for name, value in vars(original).items()
                if type(value) is int
            ),
        )

    def facts(self) -> object:
        if self._current() != self._facts:
            raise chain._UnsupportedChain("original result owner or point changed")
        return id(self), self._facts

    def state_view(self, *, store: bool = False) -> StateView:
        from . import chunk_recurrence_sm100 as original
        from .prepared_state_planner import StateView

        scope = (id(self.owner), self.cycle.kind, self.cycle.first)
        if store and self.cycle.kind == "output":
            # The original bound output transport owns the entire SMEM ring;
            # its exact swizzle/store map is not the accumulator's TMEM map.
            return StateView(
                self.owner.plan.output,
                scope,
                (
                    original.tcgen05_chain_store_output_smem,
                    original.o_smem_stmatrix_128b_ptr,
                    original.K2_O_SMEM_TILE_SIZE,
                    original.K2_O_SMEM_STAGE_SIZE,
                    original.O_OUT_OFFSET,
                    original.O_TMA_SWIZZLE_ELEMS,
                    original.O_TMA_SWIZZLE_GROUP_ELEMS,
                    original.O_TMA_SWIZZLE_ROW_MASK,
                    self.owner.plan.output.fake.dtype,
                ),
                0,
                original.K2_O_SMEM_STAGE_SIZE,
            )
        residual = self.cycle.kind == "residual"
        region = original._TMEM_LAYOUT.region(
            "factor_input"
            if store
            else "auxiliary_accumulator"
            if residual
            else "primary_accumulator"
        )
        stride = (
            region.columns
            if residual
            else (
                original._TMEM_LAYOUT.region("secondary_accumulator").column_offset
                - region.column_offset
            )
        )
        return StateView(
            original._TMEM_LAYOUT,
            scope,
            (
                original._TMEM_LAYOUT,
                region,
                stride,
                self.cycle.first,
                original.TCGEN05_STATE_K_TMEM_ROW_BLOCKS,
                original.THREADS_PER_WARP,
                "16x128b" if store else "16x256b",
            ),
            region.column_offset,
            region.columns,
        )

    def check_publication(self, request: StatePublication) -> None:
        from .prepared_continuation import ContinuationOpcode
        from .prepared_epoch_protocol import EpochPort
        from .prepared_epoch_protocol import EpochPosition
        from .prepared_epoch_protocol import EpochSynchronization

        residual = self.cycle.kind == "residual"
        if not residual:
            release = self.cycle.release
            space, complete = self.cycle.space_ready, self.cycle.output_complete
            if (
                release is None
                or release.owner is not self.owner
                or release.port is not EpochPort.OUTPUT_FREE
                or release.position is not EpochPosition.CURRENT
                or release.operation is not ContinuationOpcode.RELEASE
                or release.returns_phase
                or space is None
                or complete is None
                or space is complete
                or any(
                    sync.owner is not self.owner
                    or sync.operation is not EpochSynchronization.OUTPUT
                    for sync in (space, complete)
                )
            ):
                raise chain._UnsupportedChain("changed original output event cut")
        expected = (
            (self.destination, self.destination, self.destination)
            if residual
            else (self.cycle.release, self.cycle.output_complete, None)
        )
        actual = (
            request.transform_before.anchor,
            request.store_before.anchor,
            None if request.complete_before is None else request.complete_before.anchor,
        )
        if (
            request.read is not self
            or request.value is not self
            or any(
                cut.owner is not self.owner
                for cut in (request.transform_before, request.store_before)
            )
            or request.complete_before is not None
            and request.complete_before.owner is not self.owner
            or any(a is not b for a, b in zip(actual, expected, strict=True))
        ):
            raise chain._UnsupportedChain("changed original result publication cut")

    def views(self) -> list[str]:
        from . import chunk_recurrence_sm100 as original

        prefix = self.prefix
        residual = self.cycle.kind == "residual"
        owner = original._TMEM_LAYOUT.region(
            "auxiliary_accumulator" if residual else "primary_accumulator"
        )
        stride = (
            owner.columns
            if residual
            else original._TMEM_LAYOUT.region("secondary_accumulator").column_offset
            - owner.column_offset
        )
        stage = "0" if self.cycle.first else "acc_stage" if residual else "qstate_stage"
        result = [
            f"{prefix}_row = ((tmem_raw_addr >> 16) + (warp_idx % {original.TCGEN05_STATE_K_TMEM_ROW_BLOCKS}) * {original.THREADS_PER_WARP}) << 16",
            f"{prefix}_source = cutlass.inttoptr({prefix}_row | ((tmem_raw_addr & 0xFFFF) + {owner.column_offset} + {stage} * {stride}), 6, cutlass.Float32)",
        ]
        if residual:
            target = original._TMEM_LAYOUT.region("factor_input")
            result.append(
                f"{prefix}_target = prims.make_tmem_ptr(((tmem_raw_addr >> 16) << 16) | ((tmem_raw_addr & 0xFFFF) + {target.column_offset} + {stage} * {target.columns}), cutlass.Int8)"
            )
        return result

    def transfer(self, *, store: bool = False) -> StateTransfer:
        from . import chunk_recurrence_sm100 as original

        prefix = self.prefix
        companion = "None"
        if self.cycle.kind == "residual":
            raw = "0" if self.cycle.first else "raw_stage * V_TILE_ELEMS"
            companion = (
                f"(v_raw_smem.subview({raw}), (warp_idx % {original.TCGEN05_STATE_K_TMEM_ROW_BLOCKS}) * {original.ROWS_PER_WARP}, vmx_lane, "
                f"({original.BT}, {original.RAW_F16_TMA_SWIZZLE_ELEMS}, {original.RAW_F16_TMA_SWIZZLE_GROUP_ELEMS}, {original.RAW_F16_TMA_SWIZZLE_ROW_MASK}))"
            )
        return StateTransfer(
            f"{prefix}_source",
            f"{prefix}_packed[0:4]" if store else f"{prefix}_values",
            f"{prefix}_target",
            "None",
            True,
            "16x256b",
            2,
            "16x128b",
            companion,
            f"{prefix}_companion",
        )


@dataclass(frozen=True)
class DescriptorProductCycle:
    owner: PreparedProjectionHost
    kind: Literal["residual", "output"]
    first: bool = False
    release: EpochEvent | None = None
    space_ready: EpochSynchronizationAction | None = None
    output_complete: EpochSynchronizationAction | None = None
    binding: DescriptorProductBinding = field(init=False)
    actions: tuple[StateEffect, ...] = field(init=False)
    _actions: tuple[StateEffect, ...] = field(init=False, repr=False)
    schedule: StateTransferPlan = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from .prepared_continuation import ContinuationOpcode
        from .prepared_epoch_protocol import EpochEvent
        from .prepared_epoch_protocol import EpochPort
        from .prepared_epoch_protocol import EpochPosition
        from .prepared_epoch_protocol import EpochSynchronization
        from .prepared_epoch_protocol import EpochSynchronizationAction
        from .prepared_state_planner import StateCut
        from .prepared_state_planner import StatePublication
        from .prepared_state_planner import plan_state_transfers

        object.__setattr__(self, "binding", DescriptorProductBinding(self))
        scope = (id(self.owner), self.kind, self.first)
        if self.kind == "residual":
            cut = StateCut(self.owner, self.binding.destination, scope)
            cuts = (cut,)
            before, after, complete = cut, cut, cut
        else:
            if self.release is None:
                object.__setattr__(
                    self,
                    "release",
                    EpochEvent(
                        self.owner,
                        EpochPort.OUTPUT_FREE,
                        EpochPosition.CURRENT,
                        ContinuationOpcode.RELEASE,
                    ),
                )
            for name in ("space_ready", "output_complete"):
                if getattr(self, name) is None:
                    object.__setattr__(
                        self,
                        name,
                        EpochSynchronizationAction(
                            self.owner,
                            EpochSynchronization.OUTPUT,
                        ),
                    )
            cuts = tuple(
                StateCut(self.owner, anchor, scope)
                for anchor in (
                    self.release,
                    self.space_ready,
                    self.output_complete,
                )
            )
            before, after, complete = cuts[0], cuts[2], None
        binding = self.binding
        schedule = plan_state_transfers(
            (
                StatePublication(
                    binding,
                    binding,
                    binding.state_view(),
                    binding.state_view(store=True),
                    before,
                    after,
                    complete,
                ),
            ),
            cuts,
        )
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "actions", schedule.actions)
        object.__setattr__(self, "_actions", self.actions)

    def check_actions(self) -> None:
        self.schedule.check()
        if self.actions is not self._actions:
            raise chain._UnsupportedChain("original result action identities changed")
        if any(
            a is not b for a, b in zip(self.actions, self.schedule.actions, strict=True)
        ):
            raise chain._UnsupportedChain("original result lost planned actions")


def emit_product_state_action(effect: StateEffect) -> list[str]:
    binding = effect.binding
    if not isinstance(binding, DescriptorProductBinding):
        raise chain._UnsupportedChain("foreign descriptor result binding")
    binding.facts()
    prefix = binding.prefix
    residual = binding.cycle.kind == "residual"
    if effect.kind == "read":
        return [*binding.views(), *emit_state_transfer("read", binding.transfer())]
    if effect.kind == "transform":
        if residual:
            tokens = "seqlen" if binding.cycle.first else "seqlen - chunk * BT"
            return [
                f"{prefix}_packed = prepared_residual_point({prefix}_values, {prefix}_companion, vmx_lane, input_dtype, {tokens} if cutlass.const_expr(state.element_type == cutlass.Float32) else None)"
            ]
        return [
            f"{prefix}_packed = prepared_output_point({prefix}_values, SCALE, out.element_type)"
        ]
    if effect.kind == "store" and not residual:
        return [
            f"tcgen05_chain_store_output_smem(o_smem, warp_idx, lane, output_stage_base, {prefix}_packed)"
        ]
    if effect.kind in ("store", "complete") and residual:
        return emit_state_transfer(effect.kind, binding.transfer(store=True))
    raise chain._UnsupportedChain("unknown original result effect")
