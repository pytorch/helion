"""Original exclusive warp-register bridges, sequenced through shared stages."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from . import chained_matmul as chain
from . import chained_root_warp_stage as roots
from . import chained_warp_stage as shared
from .chained_execution import ChainedExecution
from .chained_warp_stage import _snapshot

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_scratch_layout import ScratchLayouts


def _scan_facts(inputs: list[chain._ScanInput]) -> object:
    return _snapshot(
        tuple(
            (
                item.source,
                item.operand,
                item.shared,
                item.extent,
                item.indices,
                item.coordinate,
                item.inverse_axis,
            )
            for item in inputs
        )
    )


def _staged_facts(inputs: list[chain._StagedInput]) -> object:
    return _snapshot(
        tuple(
            (
                item.source,
                item.operand,
                item.shared,
                item.role,
                item.shape,
                item.indices,
                item.coordinates,
                item.inverse_axes,
            )
            for item in inputs
        )
    )


def _layout_facts(layout: roots.WarpOperandLayout) -> object:
    return _snapshot(
        (
            layout.node,
            layout.role,
            layout.shape,
            layout.stride,
            layout.inner,
            layout.dtype,
        )
    )


@dataclass(frozen=True)
class WarpBridgeInput:
    fragment: shared.CompletedWarpFragment
    bridge: chain._RegisterBridge
    probes: tuple[tuple[Node, str], ...]
    _selection: object = field(repr=False)
    _emitted: bool = field(default=False, init=False, repr=False)
    _layout: object = field(default=None, init=False, repr=False)

    def facts(self) -> object:
        return _snapshot(
            (self.fragment, shared.bridge_facts(self.bridge), self.probes, self._layout)
        )

    def bind_layout(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        layout: roots.WarpOperandLayout,
        padding: int,
    ) -> None:
        """Capture the original builder's view before invoking the copy hook."""
        if (
            self._selection != self.facts()
            or self._layout is not None
            or self._emitted is not False
            or not self.fragment.matches(cg, plan, boundaries)
            or layout.role != self.bridge.role
            or layout.node is not self.bridge.operand
            or layout.shape != self.fragment.prepared.shape[:2]
            or layout.dtype != plan.operand_dtype(self.bridge.stage)
            or layout.inner not in (0, 1)
            or layout.stride
            != (
                (layout.shape[1] + padding, 1)
                if layout.inner == 1
                else (1, layout.shape[0] + padding)
            )
        ):
            raise chain._UnsupportedChain("warp bridge original layout changed")
        object.__setattr__(self, "_layout", _layout_facts(layout))
        object.__setattr__(self, "_selection", self.facts())

    def emit(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        probes: dict[Node, str],
        stage: int,
        layout: roots.WarpOperandLayout,
        bridge: chain._RegisterBridge,
    ) -> tuple[str, ...]:
        """Append the original program only at its original role/layout point."""
        if (
            self._selection != self.facts()
            or self._emitted is not False
            or self._layout != _layout_facts(layout)
            or bridge is not self.bridge
            or stage != bridge.stage
            or tuple(probes.items()) != self.probes
            or not self.fragment.matches(cg, plan, boundaries)
            or layout.role != bridge.role
            or layout.node is not bridge.operand
            or layout.shape != self.fragment.prepared.shape[:2]
            or layout.dtype != plan.operand_dtype(stage)
        ):
            raise chain._UnsupportedChain("warp bridge input changed or not completed")
        lines = bridge.render()
        object.__setattr__(self, "_emitted", True)
        return lines

    def complete(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
    ) -> None:
        if (
            self._selection != self.facts()
            or self._emitted is not True
            or not self.fragment.matches(cg, plan, boundaries)
        ):
            raise chain._UnsupportedChain("warp bridge completion changed")
        object.__setattr__(self.fragment, "_consumed", True)
        self.fragment._state.consumed = True


@dataclass
class _SequenceState:
    next_stage: int = 0
    fragment: shared.CompletedWarpFragment | None = None
    bridge_input: WarpBridgeInput | None = None
    prefix: tuple[str, ...] | None = None
    aliases: tuple[tuple[str, str], ...] = ()
    staged: object = None
    publication: shared.CompletedWarpStage | None = None


@dataclass(frozen=True)
class RootWarpBridgeSequence:
    codegen: GenerateAST
    plan: ChainedMatmulPlan
    boundaries: tuple[tuple[Node, str], ...]
    scan_inputs: list[chain._ScanInput]
    bridge: chain._RegisterBridge
    schedule: str
    padding: int
    dtype: str
    revision: object
    config: object
    scans: object
    obligation: object
    _selection: object = field(repr=False)
    _state: _SequenceState = field(repr=False)
    _body: roots._WarpBodyBinding = field(
        default_factory=roots._WarpBodyBinding, repr=False
    )
    _next_stage: int = field(default=0, init=False, repr=False)

    def facts(self) -> object:
        return _snapshot(
            (
                self.codegen,
                self.plan,
                self.boundaries,
                self.scan_inputs,
                self.bridge,
                self.schedule,
                self.padding,
                self.dtype,
                self.revision,
                self.config,
                self.scans,
                self.obligation,
                id(self._state),
                id(self._body),
            )
        )

    def check(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        staged_inputs: list[chain._StagedInput],
        stage: int,
        *,
        prepared: bool = False,
    ) -> None:
        aliases = tuple(plan.tensor_aliases.items())
        if (
            self._selection != self.facts()
            or cg is not self.codegen
            or plan is not self.plan
            or self._next_stage != stage
            or self._state.next_stage != stage
            or stage not in (0, 1)
            or tuple(boundaries.items()) != self.stage_boundaries(stage)
            or shared.warp_revision(plan, aliases=False) != self.revision
            or _snapshot(cg.device_function.config.config) != self.config
            or _scan_facts(self.scan_inputs) != self.scans
            or shared.bridge_facts(self.bridge) != self.obligation
            or (
                aliases[: len(self._state.aliases)] != self._state.aliases
                if prepared
                else aliases != self._state.aliases
            )
            or (not prepared and _staged_facts(staged_inputs) != self._state.staged)
        ):
            raise chain._UnsupportedChain("accepted warp bridge sequence changed")

    def stage_boundaries(self, stage: int) -> tuple[tuple[Node, str], ...]:
        if stage and self.bridge.keep_shared:
            return (*self.boundaries, (self.bridge.source, "chain_0_c"))
        return self.boundaries

    def emit(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        staged_inputs: list[chain._StagedInput],
        scratch: ScratchLayouts,
        prefix: list[str],
        stage: int,
    ) -> list[str]:
        self.check(cg, plan, boundaries, staged_inputs, stage)
        probes = dict(boundaries)
        bridge_input = None
        if stage == 1:
            fragment = self._state.fragment
            if (
                fragment is None
                or fragment.prefix != tuple(prefix)
                or self._state.prefix != tuple(prefix)
                or not fragment.matches(cg, plan, boundaries)
                or self._state.bridge_input is not None
            ):
                raise chain._UnsupportedChain("missing original completed warp bridge")
            # Only the discarded original probes see this unallocated symbol.
            # The actual bridge uses its original exact-coordinate FP32 fragment.
            probes[self.bridge.source] = f"chain_{stage - 1}_c"
            bridge_input = WarpBridgeInput(
                fragment, self.bridge, tuple(probes.items()), None
            )
            object.__setattr__(bridge_input, "_selection", bridge_input.facts())
        body = self._body.check(self)
        if body is not None:
            body.begin(
                self, cg, plan, boundaries, staged_inputs, scratch, prefix, stage
            )
        operands = roots.emit_original_warp_operands(
            cg,
            plan,
            boundaries,
            self.scan_inputs,
            {1: self.bridge},
            staged_inputs,
            stage,
            self.schedule,
            self.padding,
            self.dtype,
            probe_boundaries=probes,
            bridge_input=bridge_input,
        )
        self.check(cg, plan, boundaries, staged_inputs, stage, prepared=True)
        rows, columns, reduction = plan.shapes[stage]
        shapes = ((rows, reduction), (columns, reduction))
        if (
            not operands.matches()
            or len(operands.layouts) != 2
            or tuple(role for role, _ in operands.axes) != ("a", "b")
            or operands.asynchronous != bool(operands.copies)
        ):
            raise chain._UnsupportedChain("warp bridge operand emission changed")
        for index, (layout, shape) in enumerate(
            zip(operands.layouts, shapes, strict=True)
        ):
            stride = (
                (shape[1] + self.padding, 1)
                if layout.inner == 1
                else (1, shape[0] + self.padding)
            )
            if (
                layout.node is not plan.dots[stage].args[index]
                or layout.role != ("a", "b")[index]
                or layout.shape != shape
                or layout.inner not in (0, 1)
                or layout.stride != stride
                or operands.axes[index] != (layout.role, layout.inner)
                or layout.dtype != plan.operand_dtype(stage)
            ):
                raise chain._UnsupportedChain("warp bridge original owner changed")
        if stage == 1 and (
            bridge_input is None
            or bridge_input._emitted is not True
            or bridge_input._selection != bridge_input.facts()
            or not bridge_input.fragment.matches(cg, plan, boundaries)
        ):
            raise chain._UnsupportedChain(
                "original warp bridge obligation not consumed"
            )
        lines = list(operands.lines)
        execution = ChainedExecution(plan.threads)
        joined, completion = shared.join_warp_inputs(
            cg,
            plan,
            boundaries,
            execution,
            [*prefix, *lines],
            proxy_fence=False,
            asynchronous_copies=operands.copies,
        )
        lines.extend(joined)
        result: shared.WarpRegisterResult | shared.WarpRootResult
        if stage == 0:
            store = (
                shared.WarpRootResult(
                    stage,
                    rows * (columns + self.padding // 2),
                    scratch.layout(
                        f"chain_{stage}_c",
                        (rows, columns),
                        row_stride=columns + self.padding // 2,
                    ),
                )
                if self.bridge.keep_shared
                else None
            )
            result = shared.WarpRegisterResult(stage, self.bridge, shared=store)
        else:
            stride = columns + self.padding // 2
            result = shared.WarpRootResult(
                stage,
                rows * stride,
                scratch.layout(f"chain_{stage}_c", (rows, columns), row_stride=stride),
            )
        prepared = shared.prepare_warp_stage(
            completion,
            f"chain_{stage}",
            self.dtype,
            (rows, columns, reduction),
            32 * min(plan.threads // 32, 2 ** (columns.bit_length() - 4)),
            (operands.axes[0][1], operands.axes[1][1]),
            result,
        )
        # Publish final C atomically only after the complete action is checked.
        trial = dict(boundaries)
        lines.extend(
            shared.emit_prepared_warp_stage(
                cg,
                plan,
                trial,
                prepared,
                [*prefix, *lines],
            )
        )
        self.check(cg, plan, boundaries, staged_inputs, stage, prepared=True)
        if isinstance(result, shared.WarpRegisterResult):
            fragment = result.fragment
            if fragment is None or not fragment.matches(cg, plan, trial):
                raise chain._UnsupportedChain("warp stage omitted completion receipt")
            self._state.fragment = fragment
        if bridge_input is not None:
            bridge_input.complete(cg, plan, boundaries)
        self._state.bridge_input = bridge_input
        self._state.publication = completion._state.publication
        self._state.prefix = (*prefix, *lines)
        self._state.aliases = tuple(plan.tensor_aliases.items())
        self._state.staged = _staged_facts(staged_inputs)
        self._state.next_stage += 1
        object.__setattr__(self, "_next_stage", stage + 1)
        boundaries.update(trial)
        return lines

    def validate(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        staged_inputs: list[chain._StagedInput],
        prefix: list[str],
    ) -> None:
        body = self._body.check(self)
        if body is not None:
            body.validate_join(self, cg, plan, boundaries, staged_inputs, prefix)
        original = dict(self.stage_boundaries(1))
        if (
            self._selection != self.facts()
            or cg is not self.codegen
            or plan is not self.plan
            or self._next_stage != 2
            or self._state.next_stage != 2
            or self._state.fragment is None
            or self._state.bridge_input is None
            or self._state.bridge_input._emitted is not True
            or self._state.bridge_input.fragment is not self._state.fragment
            or self._state.bridge_input._selection != self._state.bridge_input.facts()
            or self.obligation != shared.bridge_facts(self.bridge)
            or not self._state.fragment.matches(cg, plan, original, consumed=True)
            or _scan_facts(self.scan_inputs) != self.scans
            or _staged_facts(staged_inputs) != self._state.staged
            or tuple(plan.tensor_aliases.items()) != self._state.aliases
            or tuple(prefix) != self._state.prefix
            or boundaries != {**original, plan.dots[1]: "chain_1_c"}
        ):
            raise chain._UnsupportedChain("incomplete original warp bridge sequence")


def root_warp_bridge_sequence(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scan_inputs: list[chain._ScanInput],
    bridges: dict[int, chain._RegisterBridge],
    schedule: str,
    padding: int,
    dtype: str,
    *,
    general_collectives: bool,
) -> RootWarpBridgeSequence | None:
    """Consume the original proven selection; never probe expressions again."""
    if (
        plan.strategy == "tcgen05_tmem"
        or plan.loop is not None
        or len(plan.dots) != 2
        or tuple(bridges) != (1,)
        or any(dot.args[2] is not None for dot in plan.dots)
        or plan.pointwise_cache is not None
        or general_collectives
        or schedule
        not in (
            "cp_async_register",
            "cp_async_register_reuse",
            "cp_async_register_reuse_scan",
        )
        or plan.operand_dtype(0) != plan.operand_dtype(1)
    ):
        return None
    bridge = bridges[1]
    if (
        bridge.stage != 1
        or bridge.source is not plan.dots[0]
        or bridge.role not in ("a", "b")
        or bridge.operand is not plan.dots[1].args[0 if bridge.role == "a" else 1]
        or _snapshot(bridge.revision)
        != _snapshot(chain._register_bridge_revision(plan))
    ):
        raise chain._UnsupportedChain("original warp bridge proof changed")
    sequence = RootWarpBridgeSequence(
        cg,
        plan,
        tuple(boundaries.items()),
        scan_inputs,
        bridge,
        schedule,
        padding,
        dtype,
        shared.warp_revision(plan, aliases=False),
        _snapshot(cg.device_function.config.config),
        _scan_facts(scan_inputs),
        shared.bridge_facts(bridge),
        None,
        _SequenceState(
            aliases=tuple(plan.tensor_aliases.items()), staged=_staged_facts([])
        ),
    )
    object.__setattr__(sequence, "_selection", sequence.facts())
    sequence._body.owner = sequence
    return sequence


@dataclass(frozen=True)
class RootWarpBridgeAction:
    """One original stage, with the sequence retaining its bridge authority."""

    sequence: RootWarpBridgeSequence
    stage: int

    def emit(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        staged_inputs: list[chain._StagedInput],
        scratch: ScratchLayouts,
        prefix: list[str],
    ) -> list[str]:
        return self.sequence.emit(
            cg, plan, boundaries, staged_inputs, scratch, prefix, self.stage
        )
