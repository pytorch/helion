"""Typed original state effects consumed by the common body interpreter.

Point arithmetic is captured by the existing expression emitter, not parsed
or reinterpreted here. The enclosing original stage still owns its final join
and completion; a state-effect program alone cannot retire that stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Literal

from . import chained_matmul as chain
from .chained_completed_store import _records

if TYPE_CHECKING:
    from torch.fx import Node

    from .chained_body_program import BodyProgram
    from .chained_body_program import RootActionBody
    from .chained_execution import ChainedExecution
    from .chained_root_stage import RootInitializedResult
    from .chained_root_stage import RootStageAction
    from .chained_seed_tiles import SeedPanel
    from .chained_tcgen_stage import StageGeometry
    from .chunk_prefill_prepared_bt16_state import BT16StateBinding
    from .chunk_prefill_prepared_output import FastOutputBinding
    from .chunk_prefill_prepared_state import FastLoopStateBinding
    from .chunk_prefill_prepared_state_abi import FastStateABIBinding
    from .chunk_prefill_prepared_state_products import FastStateProductBinding
    from .prepared_epoch_product import DescriptorProductBinding
    from .prepared_epoch_state import DescriptorStateBinding
    from .prepared_state_planner import StatePublication
    from .prepared_state_planner import StateTransferPlan
    from .prepared_state_planner import StateView


@dataclass(frozen=True)
class StatePoint:
    """The actual original expression result, not an arbitrary source block."""

    expression: chain._Expression
    coordinates: tuple[str, str]
    index: str
    value: str
    _boundary_owner: dict[Node, str] = field(init=False, repr=False)
    _boundaries: tuple[tuple[Node, str], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_boundary_owner", self.expression.boundaries)
        object.__setattr__(
            self, "_boundaries", tuple(self.expression.boundaries.items())
        )

    def facts(self) -> object:
        expression = self.expression
        # Original later stages append their completed C boundaries. They do
        # not change the expression's already-consumed input prefix; the outer
        # RootActionBody still checks the complete current publication map.
        if (
            expression.boundaries is not self._boundary_owner
            or tuple(expression.boundaries.items())[: len(self._boundaries)]
            != self._boundaries
        ):
            raise chain._UnsupportedChain("initialized point input boundaries changed")
        return (
            id(self),
            id(expression),
            id(expression.cg),
            id(expression.plan),
            self.coordinates,
            self.index,
            self.value,
            _records(
                (
                    expression.statements,
                    expression.fragments,
                    self._boundaries,
                    expression.scan_inputs,
                    expression.memo,
                    expression.definitions,
                    expression.definition_inputs,
                    expression.reads,
                    expression.loaded_inputs,
                )
            ),
        )


@dataclass(frozen=True)
class FragmentStateBinding:
    """Original full accumulator and its existing optional member panel."""

    plan: chain.ChainedMatmulPlan
    source: Node
    destination: Node
    shape: tuple[int, int]
    panel: SeedPanel | None
    prefix: str
    point: StatePoint

    @property
    def offset(self) -> int:
        return 0 if self.panel is None else self.panel.offset

    @property
    def width(self) -> int:
        return self.shape[1] if self.panel is None else self.panel.width

    def facts(self) -> object:
        return (
            id(self),
            id(self.plan),
            self.source,
            self.destination,
            self.shape,
            _records(self.panel),
            self.prefix,
            self.point.facts(),
        )

    def state_view(self, *, store: bool = False) -> StateView:
        from .chained_tmem_segments import validate_tmem_segment
        from .prepared_state_planner import StateView

        selected = self.plan.initialized_accumulator
        if (
            selected is None
            or self.source is not selected.first
            or self.destination is not selected.seed
            or self.shape != self.plan.shapes[0][:2]
            or self.point.expression.plan is not self.plan
        ):
            raise chain._UnsupportedChain("foreign initialized state value/view")
        validate_tmem_segment(self.shape, self.offset, self.width)
        # Original read/store partitions are proven equal by codegen_seed.
        # The full C owner remains the plan's actual first contraction result.
        return StateView(
            self.source,
            (id(self.plan), id(selected)),
            (self.shape, self.panel, 128, "original_m128_c_partition"),
            self.offset,
            self.width,
        )

    def check_publication(self, request: StatePublication) -> None:
        if request.read_before is not None:
            raise chain._UnsupportedChain("original state has no early capture cut")

        selected = self.plan.initialized_accumulator
        if (
            selected is None
            or request.read is not self
            or request.value is not self
            or request.complete_before is None
            or any(
                cut.owner is not self.plan
                or cut.scope != (id(self.plan), id(selected))
                or cut.anchor is not selected.join
                for cut in (
                    request.transform_before,
                    request.store_before,
                    request.complete_before,
                )
            )
        ):
            raise chain._UnsupportedChain("changed original initialized state cut")


@dataclass(frozen=True)
class StateEffect:
    kind: Literal["read", "transform", "store", "complete"]
    binding: (
        FragmentStateBinding
        | DescriptorStateBinding
        | DescriptorProductBinding
        | FastLoopStateBinding
        | FastOutputBinding
        | FastStateProductBinding
        | FastStateABIBinding
        | BT16StateBinding
    )
    schedule: StateTransferPlan | None = field(default=None, repr=False, compare=False)

    @property
    def source(self) -> Node:
        return self.binding.source

    @property
    def destination(self) -> Node:
        return self.binding.destination

    @property
    def offset(self) -> int:
        return self.binding.offset

    @property
    def width(self) -> int:
        return self.binding.width

    def facts(self) -> object:
        # The enclosing state body checks the complete plan once, rather than
        # repeating every request/owner walk for each individual effect.
        return id(self), self.kind, self.binding.facts(), id(self.schedule)


@dataclass(frozen=True)
class StateTransfer:
    """Native operands of an already-owner-checked state action.

    These names are scalar/SSA bindings supplied by the original view adapter,
    never source statements or a callback. Both descriptor and fragment clients
    use the same read/store/completion instruction lowering below.
    """

    source: str
    values: str
    destination: str
    copy: str
    descriptor: bool
    load_shape: str | None
    load_count: int
    store_shape: str | None
    companion: str = "None"
    companion_values: str = "_state_companion"


def state_transfer_instruction(
    kind: Literal["read", "store", "complete"], transfer: StateTransfer
) -> tuple[int, bool, str | None, int, str | None]:
    """Native transfer geometry shared by source and constexpr-program emission."""
    return (
        ("read", "store", "complete").index(kind),
        transfer.descriptor,
        transfer.load_shape,
        transfer.load_count,
        transfer.store_shape,
    )


def emit_state_transfer(
    kind: Literal["read", "store", "complete"], transfer: StateTransfer
) -> list[str]:
    _, descriptor, load_shape, load_count, store_shape = state_transfer_instruction(
        kind, transfer
    )
    if kind == "read":
        call = f"prepared_tcgen_edge.execute_prepared_read({transfer.source}, {('None' if descriptor else transfer.values)}, {transfer.copy}, {transfer.companion}, None, None, {descriptor!r}, {load_shape!r}, {load_count})"
        # Fragment copy mutates its already-allocated RMEM tensor. Discard its
        # redundant return rather than inventing a TMEM-derived tuple alias;
        # the original last-read storage analysis remains unchanged.
        return [
            f"{transfer.values}, {transfer.companion_values} = {call}"
            if descriptor
            else call
        ]
    if kind == "store":
        return [
            f"prepared_tcgen_edge.execute_prepared_store({transfer.values}, {transfer.destination}, {transfer.copy}, {descriptor!r}, {store_shape!r})"
        ]
    if kind == "complete":
        return [
            f"prepared_tcgen_edge.execute_prepared_store_completion({descriptor!r})"
        ]
    raise chain._UnsupportedChain("unknown prepared state transfer")


def _fragment_transfer(binding: FragmentStateBinding, *, store: bool) -> StateTransfer:
    prefix = binding.prefix
    if store:
        target = (
            "chain_seed_target" if binding.panel is None else f"{prefix}_store_target"
        )
        copy = "chain_seed_copy" if binding.panel is None else f"{prefix}_store_copy"
    else:
        target = "None"
        copy = f"{prefix}_copy"
    return StateTransfer(
        f"{prefix}_source", f"{prefix}_values", target, copy, False, None, 0, None
    )


def _emit_fragment_action(
    effect: StateEffect, execution: ChainedExecution
) -> list[str]:
    from .chained_tcgen05 import _load_result_views
    from .chained_tmem_segments import emit_tmem_segment_read_views

    binding = effect.binding
    if not isinstance(binding, FragmentStateBinding):
        raise chain._UnsupportedChain("foreign fragment state binding")
    prefix = binding.prefix
    if effect.kind == "read":
        views = (
            _load_result_views(prefix, binding.shape, execution=execution)
            if binding.panel is None
            else emit_tmem_segment_read_views(
                prefix,
                "chain_0",
                binding.shape,
                binding.offset,
                binding.width,
                execution=execution,
            )
        )
        return [
            *views,
            *emit_state_transfer("read", _fragment_transfer(binding, store=False)),
        ]
    if effect.kind == "transform":
        point = binding.point
        return [
            f"for {point.index} in cutlass.range_constexpr(cute.size({prefix}_values)):",
            f"    {point.coordinates[0]}, {point.coordinates[1]} = {prefix}_coords[{point.index}]",
            chain._indent(point.expression.lines),
            f"    {prefix}_values[{point.index}] = cutlass.Float32({point.value})",
        ]
    if effect.kind == "store":
        views = (
            [
                "chain_seed_copy = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(32)), cutlass.Float32), chain_0_acc)",
                "chain_seed_target = chain_seed_copy.get_slice(chain_thread).partition_D(chain_0_acc)",
            ]
            if binding.panel is None
            else binding.panel.store_views(
                f"{prefix}_store", "chain_0", execution.thread
            )
        )
        return [
            *views,
            *emit_state_transfer("store", _fragment_transfer(binding, store=True)),
        ]
    return emit_state_transfer("complete", _fragment_transfer(binding, store=True))


def emit_state_action(effect: StateEffect, execution: ChainedExecution) -> list[str]:
    """Shared state instruction lowering under the caller's original authority."""
    from .prepared_epoch_product import DescriptorProductBinding
    from .prepared_epoch_product import emit_product_state_action
    from .prepared_epoch_state import DescriptorStateBinding
    from .prepared_epoch_state import emit_descriptor_state_action

    if isinstance(effect.binding, FragmentStateBinding):
        return _emit_fragment_action(effect, execution)
    if isinstance(effect.binding, DescriptorStateBinding):
        return emit_descriptor_state_action(effect)
    if isinstance(effect.binding, DescriptorProductBinding):
        return emit_product_state_action(effect)
    raise chain._UnsupportedChain("unknown state representation")


@dataclass
class RootStateBody:
    owner: RootActionBody
    action: RootStageAction
    result: RootInitializedResult
    geometry: StageGeometry
    execution: ChainedExecution
    program: BodyProgram
    _cursor: int = field(default=0, init=False)
    _finished: tuple[str, ...] | None = field(default=None, init=False)

    def check(self) -> None:
        self.owner.check()
        selected = self.owner.plan.initialized_accumulator
        if (
            selected is None
            or self.owner.continuation is None
            or self.owner.continuation.transformed is not selected
            or self.owner.pending is not self.action
            or self.action.stage != 0
            or self.result is not self.action.sequence.initialized
            or self.execution != self.owner.execution
            or self.geometry != self.action.sequence.geometries[0]
            or self.geometry.transpose
            or self.geometry.native_rows != 128
            or self.program.actions != self.result.state_effects
            or any(
                action is not original
                for action, original in zip(
                    self.program.actions, self.result.state_effects, strict=True
                )
            )
        ):
            raise chain._UnsupportedChain("foreign initialized state body")
        self.owner.continuation.check()
        effects = self.result.state_effects
        if not effects or len(effects) % 3 != 1 or effects[-1].kind != "complete":
            raise chain._UnsupportedChain("incomplete initialized state effects")
        schedule = effects[0].schedule
        if schedule is None or any(
            effect.schedule is not schedule for effect in effects
        ):
            raise chain._UnsupportedChain("initialized state lost shared decision")
        schedule.check()
        if any(a is not b for a, b in zip(schedule.actions, effects, strict=True)):
            raise chain._UnsupportedChain("initialized state changed planned actions")
        offset = 0
        for start in range(0, len(effects) - 1, 3):
            read, transform, publish = effects[start : start + 3]
            if (
                (read.kind, transform.kind, publish.kind)
                != ("read", "transform", "store")
                or not isinstance(read.binding, FragmentStateBinding)
                or read.binding.plan is not self.owner.plan
                or read.binding.shape != self.owner.plan.shapes[0][:2]
                or transform.binding is not read.binding
                or publish.binding is not read.binding
                or read.binding.point.expression.cg is not self.owner.codegen
                or read.binding.point.expression.plan is not self.owner.plan
                or type(read.offset) is not int
                or type(read.width) is not int
                or read.offset != offset
                or read.width <= 0
                or any(
                    effect.source is not selected.first
                    or effect.destination is not selected.seed
                    or effect.offset != read.offset
                    or effect.width != read.width
                    for effect in (read, transform, publish)
                )
            ):
                raise chain._UnsupportedChain("changed initialized state effect order")
            offset += read.width
        if offset != self.owner.plan.shapes[0][1]:
            raise chain._UnsupportedChain("incomplete initialized state owner")
        if effects[-1].binding is not effects[-2].binding:
            raise chain._UnsupportedChain("foreign initialized state completion")

    def emit(self, effect: StateEffect) -> list[str]:
        self.check()
        if (
            self._finished is not None
            or self._cursor >= len(self.program.actions)
            or self.program.actions[self._cursor] is not effect
        ):
            raise chain._UnsupportedChain("initialized state effect replay")
        self._cursor += 1
        return emit_state_action(effect, self.execution)

    def finish(self, lines: list[str]) -> None:
        self.check()
        if self._finished is not None or self._cursor != len(self.program.actions):
            raise chain._UnsupportedChain("incomplete initialized state emission")
        self._finished = tuple(lines)

    def accept(self, lines: list[str]) -> None:
        self.check()
        if self._finished is None or tuple(lines) != self._finished:
            raise chain._UnsupportedChain("initialized state returned body changed")


def emit_initialized_state(
    owner: RootActionBody,
    action: RootStageAction,
    result: RootInitializedResult,
    geometry: StageGeometry,
    execution: ChainedExecution,
) -> list[str]:
    from .chained_body_program import BodyProgram
    from .chained_body_program import emit_body_program

    body = RootStateBody(
        owner, action, result, geometry, execution, BodyProgram(result.state_effects)
    )
    lines = emit_body_program(
        owner.codegen, owner.plan, None, execution, None, None, None, state_body=body
    )
    body.accept(lines)
    body.check()
    if tuple(lines) != body._finished:
        raise chain._UnsupportedChain("accepted initialized state body changed")
    return ["from helion._compiler.cute import prepared_tcgen_edge", *lines]
