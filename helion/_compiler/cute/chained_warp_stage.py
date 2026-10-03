"""Shared execution of already prepared, original-layout warp operands.

Preparation clients retain their layouts, copies and result allocation policy.
This executor owns the ordered MMA, active-team predicate and publication join.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from . import chained_matmul as chain

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_preparation_fragment import PreparationFragment
    from .chained_tcgen_stage import StageGeometry


def _snapshot(value: object) -> object:
    if isinstance(value, dict):
        return (dict, tuple((key, _snapshot(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_snapshot(item) for item in value))
    return (type(value), value)


def warp_revision(plan: ChainedMatmulPlan, *, aliases: bool = True) -> object:
    return _snapshot(
        (
            chain._register_bridge_revision(plan),
            plan.axes,
            plan.tensor_aliases if aliases else None,
            plan.strategy,
            plan.dtype,
            tuple(plan.operand_dtype(stage) for stage in range(len(plan.dots))),
            plan.loop,
            plan.pointwise_cache,
        )
    )


def _execution(execution: ChainedExecution) -> object:
    return _snapshot(
        (
            execution.threads,
            execution.thread,
            execution.warp,
            execution.sync,
            execution.a_workspace,
            execution.b_workspace,
            execution.tmem,
            execution.barriers,
        )
    )


@dataclass
class _CompletionState:
    consumed: bool = False
    prepared: PreparedWarpStage | None = field(default=None, repr=False)
    publication: CompletedWarpStage | None = field(default=None, repr=False)


@dataclass(frozen=True)
class WarpInputCompletion:
    """An actual input join and its complete prefix, consumed exactly once."""

    codegen: GenerateAST
    plan: ChainedMatmulPlan
    execution: ChainedExecution
    revision: object
    boundaries: tuple[tuple[Node, str], ...]
    prefix: tuple[str, ...]
    _context: object
    _config: object
    _selection: tuple[object, ...] = field(repr=False)
    _state: _CompletionState = field(repr=False)
    _consumed: bool = field(default=False, init=False, repr=False)

    def fields(self) -> tuple[object, ...]:
        return (
            self.codegen,
            self.plan,
            self.execution,
            self.revision,
            self.boundaries,
            self.prefix,
            self._context,
            self._config,
            id(self._state),
        )

    def matches(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        prefix: Sequence[str],
    ) -> bool:
        return (
            cg is self.codegen
            and plan is self.plan
            and self._state.consumed is False
            and self._consumed is False
            and self._selection == self.fields()
            and self.revision == warp_revision(plan)
            and self._context == _execution(self.execution)
            and self._config == _snapshot(cg.device_function.config.config)
            and self.boundaries == tuple(boundaries.items())
            and self.prefix == tuple(prefix)
        )


def join_warp_inputs(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    execution: ChainedExecution,
    prefix: Sequence[str],
    *,
    proxy_fence: bool,
    asynchronous_copies: tuple[tuple[str, ...], ...] = (),
) -> tuple[list[str], WarpInputCompletion]:
    """Emit the original join; copy programs, not schedule names, require a wait."""
    if type(proxy_fence) is not bool or any(not copy for copy in asynchronous_copies):
        raise chain._UnsupportedChain("invalid prepared warp input completion")
    lines = (
        ["cute.arch.cp_async_commit_group()", "cute.arch.cp_async_wait_group(0)"]
        if asynchronous_copies
        else []
    )
    if proxy_fence:
        lines.append("cute.arch.fence_view_async_shared()")
    lines.append(execution.sync)
    revision, bindings = warp_revision(plan), tuple(boundaries.items())
    complete_prefix, context = (*prefix, *lines), _execution(execution)
    config = _snapshot(cg.device_function.config.config)
    state = _CompletionState()
    selection = (
        cg,
        plan,
        execution,
        revision,
        bindings,
        complete_prefix,
        context,
        config,
        id(state),
    )
    return lines, WarpInputCompletion(
        cg,
        plan,
        execution,
        revision,
        bindings,
        complete_prefix,
        context,
        config,
        selection,
        state,
    )


@dataclass(frozen=True)
class WarpRootResult:
    stage: int
    elements: int
    layout: str


@dataclass(frozen=True)
class WarpMemberResult:
    prefix: str
    members: tuple[tuple[int, StageGeometry, int], ...]
    fragment: PreparationFragment | None = None


def bridge_facts(bridge: chain._RegisterBridge) -> object:
    return _snapshot(
        (
            bridge.role,
            bridge.lines,
            bridge.stage,
            bridge.source,
            bridge.operand,
            bridge.revision,
            bridge.epilogue_facts(),
        )
    )


@dataclass(frozen=True)
class WarpRegisterResult:
    """An existing exclusive bridge, not a newly omitted shared result."""

    stage: int
    bridge: chain._RegisterBridge
    fragment: CompletedWarpFragment | None = field(default=None, init=False)
    shared: WarpRootResult | None = None


def _result_facts(
    result: WarpRootResult | WarpMemberResult | WarpRegisterResult,
) -> object:
    if isinstance(result, WarpRootResult):
        return _snapshot(("root", result.stage, result.elements, result.layout))
    if isinstance(result, WarpRegisterResult):
        return _snapshot(
            (
                "register",
                result.stage,
                bridge_facts(result.bridge),
                None
                if result.shared is None
                else (
                    result.shared.stage,
                    result.shared.elements,
                    result.shared.layout,
                ),
            )
        )
    return _snapshot(
        (
            "members",
            result.prefix,
            tuple(
                (
                    stage,
                    geometry.logical,
                    geometry.transpose,
                    geometry.native_rows,
                    offset,
                )
                for stage, geometry, offset in result.members
            ),
            None if result.fragment is None else result.fragment.facts(),
        )
    )


@dataclass(frozen=True)
class PreparedWarpStage:
    completion: WarpInputCompletion
    prefix: str
    dtype: str
    shape: tuple[int, int, int]
    threads: int
    axes: tuple[int, int]
    result: WarpRootResult | WarpMemberResult | WarpRegisterResult
    _selection: object = field(repr=False)

    def facts(self) -> object:
        return _snapshot(
            (
                self.completion,
                id(self.completion),
                id(self.completion.execution),
                self.prefix,
                self.dtype,
                self.shape,
                self.threads,
                self.axes,
                _result_facts(self.result),
                id(self.result),
            )
        )


def prepare_warp_stage(
    completion: WarpInputCompletion,
    prefix: str,
    dtype: str,
    shape: tuple[int, int, int],
    threads: int,
    axes: tuple[int, int],
    result: WarpRootResult | WarpMemberResult | WarpRegisterResult,
) -> PreparedWarpStage:
    """Seal geometry after the client's original operand/owner validation."""
    if (
        any(type(size) is not int or size <= 0 for size in shape)
        or shape[0] % 16
        or shape[1] % 8
        or shape[2] % 16
        or type(threads) is not int
        or threads < 32
        or threads % 32
        or threads & (threads - 1)
        or threads > completion.execution.threads
        or any(type(axis) is not int or axis not in (0, 1) for axis in axes)
    ):
        raise chain._UnsupportedChain("invalid prepared warp geometry or ownership")
    stage = PreparedWarpStage(
        completion, prefix, dtype, shape, threads, axes, result, None
    )
    prepared = PreparedWarpStage(
        completion, prefix, dtype, shape, threads, axes, result, stage.facts()
    )
    if completion._state.prepared is not None or completion._state.consumed:
        raise chain._UnsupportedChain("repeated prepared warp stage")
    completion._state.prepared = prepared
    return prepared


@dataclass(frozen=True)
class CompletedWarpStage:
    """The original executor's successful publication, sealed before recording."""

    prepared: PreparedWarpStage
    prefix: tuple[str, ...]
    boundaries: tuple[tuple[Node, str], ...]
    revision: object
    _selection: object = field(repr=False)

    def facts(self) -> object:
        completion = self.prepared.completion
        return _snapshot(
            (
                id(self),
                id(self.prepared),
                self.prepared.facts(),
                completion.fields(),
                self.prefix,
                self.boundaries,
                self.revision,
            )
        )

    def matches(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        prefix: Sequence[str],
    ) -> bool:
        prepared = self.prepared
        completion = prepared.completion
        return (
            completion._state.publication is self
            and completion._state.prepared is prepared
            and completion._consumed is True
            and completion._state.consumed is True
            and self._selection == self.facts()
            and prepared._selection == prepared.facts()
            and completion._selection == completion.fields()
            and completion.codegen is cg
            and completion.plan is plan
            and completion._context == _execution(completion.execution)
            and completion._config == _snapshot(cg.device_function.config.config)
            and self.revision == warp_revision(plan)
            and self.boundaries == tuple(boundaries.items())
            and self.prefix == tuple(prefix)
        )


@dataclass(frozen=True)
class CompletedWarpFragment:
    """Actual whole-team MMA completion; valid until its one original bridge."""

    prepared: PreparedWarpStage
    prefix: tuple[str, ...]
    revision: object
    aliases: tuple[tuple[str, str], ...]
    _selection: object = field(repr=False)
    _state: _CompletionState = field(repr=False)
    _consumed: bool = field(default=False, init=False, repr=False)

    def facts(self) -> object:
        completion = self.prepared.completion
        return _snapshot(
            (
                self.prepared,
                self.prepared.facts(),
                self.prefix,
                self.revision,
                self.aliases,
                completion._selection,
                completion.fields(),
                id(self._state),
            )
        )

    def matches(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        *,
        consumed: bool = False,
    ) -> bool:
        completion, result = self.prepared.completion, self.prepared.result
        return (
            self._selection == self.facts()
            and self.prepared._selection == self.prepared.facts()
            and isinstance(result, WarpRegisterResult)
            and result.fragment is self
            and self._consumed is consumed
            and self._state.consumed is consumed
            and completion._consumed is True
            and completion._state.consumed is True
            and completion.codegen is cg
            and completion.plan is plan
            and self.revision == warp_revision(plan, aliases=False)
            and tuple(plan.tensor_aliases.items())[: len(self.aliases)] == self.aliases
            and completion._context == _execution(completion.execution)
            and completion._config == _snapshot(cg.device_function.config.config)
            and (
                completion.boundaries == tuple(boundaries.items())
                if result.shared is None
                else completion._state.publication is not None
                and completion._state.publication._selection
                == completion._state.publication.facts()
                and completion._state.publication.boundaries
                == tuple(boundaries.items())
                and completion._state.publication.prefix == self.prefix
                and tuple(boundaries.items())
                == (
                    *completion.boundaries,
                    (plan.dots[result.stage], f"chain_{result.stage}_c"),
                )
            )
            and self.prepared.threads == completion.execution.threads
        )


def emit_prepared_warp_stage(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    prepared: PreparedWarpStage,
    input_prefix: Sequence[str],
) -> list[str]:
    """One shared prepared-warp executor, retaining both original publications."""
    from .chained_tcgen_stage import _publish_result
    from .chained_warp_mma import emit_warp_mma

    completion = prepared.completion
    if (
        completion._state.prepared is not prepared
        or prepared._selection != prepared.facts()
        or not completion.matches(cg, plan, boundaries, input_prefix)
    ):
        raise chain._UnsupportedChain("prepared warp stage changed or already consumed")
    prefix, result = prepared.prefix, prepared.result
    execution = completion.execution
    rows, columns, reduction = prepared.shape
    lines: list[str] = []
    aliases: list[str] = []
    if isinstance(result, WarpRootResult):
        if (
            plan.shapes[result.stage] != prepared.shape
            or plan.dots[result.stage].args[2] is not None
        ):
            raise chain._UnsupportedChain(
                "root warp zero seed or result geometry changed"
            )
        lines.extend(
            [
                f"{prefix}_c_ptr = cute.arch.alloc_smem(cutlass.Float32, {result.elements}, alignment=128)",
                f"{prefix}_c = cute.make_tensor({prefix}_c_ptr, {result.layout})",
            ]
        )
    elif isinstance(result, WarpMemberResult):
        aliases = [
            f"{prefix}_a = cute.local_tile({result.prefix}_a, ({rows}, {reduction}), (0, 0))",
            f"{prefix}_b = {result.prefix}_b",
        ]
    else:
        bridge = result.bridge
        if (
            result.fragment is not None
            or result.stage + 1 != bridge.stage
            or plan.dots[result.stage] is not bridge.source
            or plan.shapes[result.stage] != prepared.shape
            or plan.dots[result.stage].args[2] is not None
            or prepared.threads != execution.threads
            or _snapshot(bridge.revision)
            != _snapshot(chain._register_bridge_revision(plan))
        ):
            raise chain._UnsupportedChain("unproved retained warp result")
        if (result.shared is not None) != bridge.keep_shared:
            raise chain._UnsupportedChain(
                "retained fragment changed shared publication"
            )
        if result.shared is not None:
            if result.shared.stage != result.stage:
                raise chain._UnsupportedChain("retained fragment changed shared member")
            lines.extend(
                [
                    f"{prefix}_c_ptr = cute.arch.alloc_smem(cutlass.Float32, {result.shared.elements}, alignment=128)",
                    f"{prefix}_c = cute.make_tensor({prefix}_c_ptr, {result.shared.layout})",
                ]
            )
    compute = [
        *aliases,
        *emit_warp_mma(
            prefix,
            prepared.dtype,
            prepared.shape,
            prepared.threads,
            dict(zip(("a", "b"), prepared.axes, strict=True)),
            [f"{prefix}_acc.fill(0.0)"],
            execution=execution,
        ),
    ]
    trial = dict(boundaries)
    if (
        isinstance(result, WarpRootResult)
        or isinstance(result, WarpRegisterResult)
        and result.shared is not None
    ):
        compute.extend(
            [
                f"{prefix}_sc = {prefix}_thr.partition_C({prefix}_c)",
                f"cute.autovec_copy({prefix}_acc, {prefix}_sc)",
            ]
        )
        trial[plan.dots[result.stage]] = f"{prefix}_c"
    elif isinstance(result, WarpMemberResult):
        published_members = result.members
        if result.fragment is not None:
            result.fragment.check()
            if (
                result.fragment.member not in result.members
                or prepared.threads != execution.threads
            ):
                raise chain._UnsupportedChain(
                    "grouped fragment lost complete member/team"
                )
            if not result.fragment.keep_shared:
                published_members = tuple(
                    member
                    for member in result.members
                    if member != result.fragment.member
                )
        compute.extend(
            [
                f"{result.prefix}_values = {prefix}_acc",
                f"{result.prefix}_coords = {prefix}_thr.partition_C(cute.make_identity_tensor(({rows}, {columns})))",
                *_publish_result(plan, trial, result.prefix, published_members),
            ]
        )
    if prepared.threads < execution.threads:
        lines.extend(
            [f"if {execution.thread} < {prepared.threads}:", chain._indent(compute)]
        )
    else:
        lines.extend(compute)
    lines.append(execution.sync)
    if (
        completion._state.prepared is not prepared
        or prepared._selection != prepared.facts()
        or not completion.matches(cg, plan, boundaries, input_prefix)
    ):
        raise chain._UnsupportedChain("prepared warp changed during emission")
    completion._state.consumed = True
    object.__setattr__(completion, "_consumed", True)
    boundaries.update(trial)
    if isinstance(result, WarpRegisterResult):
        fragment = CompletedWarpFragment(
            prepared,
            (*input_prefix, *lines),
            warp_revision(plan, aliases=False),
            tuple(plan.tensor_aliases.items()),
            None,
            _CompletionState(),
        )
        object.__setattr__(fragment, "_selection", fragment.facts())
        object.__setattr__(result, "fragment", fragment)
    from .native_matmul_metadata import record_native_stage

    publication = CompletedWarpStage(
        prepared,
        (*input_prefix, *lines),
        tuple(boundaries.items()),
        warp_revision(plan),
        None,
    )
    object.__setattr__(publication, "_selection", publication.facts())
    completion._state.publication = publication
    if isinstance(result, WarpMemberResult) and result.fragment is not None:
        result.fragment.bind(publication)
    stages = (
        tuple(member[0] for member in result.members)
        if isinstance(result, WarpMemberResult)
        else (result.stage,)
    )
    record_native_stage(cg, plan, tuple(plan.dots[i] for i in stages), "warp", lines)
    return lines
