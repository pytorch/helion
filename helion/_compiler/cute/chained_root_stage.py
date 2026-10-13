"""Accepted straight-line input actions for the shared contraction executor.

The root scheduler still owns allocation, scan placement, prefetch and epilogue.
This component supplies original operand producers at the executor's explicit
phase boundaries; it never supplies a replacement MMA kernel. A completed
intermediate FP32 fragment stays authoritative until its sole, proven bridge
has evaluated the original expression and published its narrowed TMEM image.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from itertools import starmap
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

import torch

from . import chained_matmul as chain
from .chained_collectives import uses_general_collectives
from .chained_contraction_groups import ContractionGroup
from .chained_tmem_transport import TmemOperandBinding

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_execution import ChainedExecution
    from .chained_k_schedule import KSchedule
    from .chained_late_rhs import LateRhsArenaPlan
    from .chained_leaf_pipeline import LeafPipeline
    from .chained_matmul import ChainedMatmulPlan
    from .chained_pointwise_cache import PointwiseReadCache
    from .chained_pointwise_inplace import PointwiseInplace
    from .chained_pointwise_unroll import PointwiseUnroll
    from .chained_root_snapshot import RootSnapshot
    from .chained_seed_tiles import SeedTiling
    from .chained_startup import StartupInput
    from .chained_tcgen_stage import StageGeometry
    from .chained_vector_stage import VectorStageOperand
    from .chained_vector_stage import VectorStaging
    from .prepared_state_body import StateEffect


def supports_root_pair(cg: GenerateAST, plan: ChainedMatmulPlan) -> bool:
    """A full-K pair with direct first inputs and one exclusive packed A edge.

    This early check is deliberately not the coordinate/bridge proof. The
    original root proof runs at its original codegen point before activation.
    """
    from .chained_plain_root import _ordinary_root_options
    from .chained_tcgen05 import _prefetch_final_b
    from .chained_tcgen05 import supported_plan

    return (
        plan.strategy == "tcgen05_tmem"
        and plan.loop is None
        and len(plan.dots) == 2
        and all(shape[0] == 128 for shape in plan.shapes)
        and all(dot.args[2] is None for dot in plan.dots)
        and plan.initialized_accumulator is None
        and plan.late_rhs_reuse is None
        and not plan.direct_output
        and plan.k_schedule is None
        and plan.pointwise_cache is None
        and not plan.warp_mma_stages
        and plan.preparation_pipeline is None
        and not uses_general_collectives(plan)
        and not plan.scan_exports
        and all(chain._direct_operand(cast("Node", x)) for x in plan.dots[0].args[:2])
        and all(not (chain._ancestors(scan) & set(plan.dots)) for scan in plan.scans)
        and not any(
            chain._ancestors(cast("Node", x)) & set(plan.scans)
            for x in plan.dots[0].args[:2]
        )
        and _prefetch_final_b(plan)
        and not cg.device_function.config.config.get(
            "cute_chained_auxiliary_cache", False
        )
        and _ordinary_root_options(cg)
        and supported_plan(plan)
    )


def supports_local_root_pair(cg: GenerateAST, plan: ChainedMatmulPlan) -> bool:
    """Weighted full-M128 inputs with original, stage-local final RHS copies.

    This is only discovery. The original four axis selections and exclusive
    packed-input proof must succeed before a shared action is accepted.
    """
    from .chained_tcgen05 import _prefetch_final_b
    from .chained_tcgen05 import supported_plan

    config = cg.device_function.config.config
    return (
        plan.strategy == "tcgen05_tmem"
        and plan.loop is None
        and len(plan.dots) == 2
        and all(shape[0] == 128 for shape in plan.shapes)
        and all(dot.args[2] is None for dot in plan.dots)
        and plan.initialized_accumulator is None
        and plan.late_rhs_reuse is None
        and not plan.direct_output
        and plan.k_schedule is None
        and plan.pointwise_cache is None
        and not plan.warp_mma_stages
        and plan.preparation_pipeline is None
        and not uses_general_collectives(plan)
        and not plan.scans
        and not plan.scan_exports
        and not _prefetch_final_b(plan)
        and type(config.get("cute_chained_pointwise_vectorize", False)) is bool
        and type(config.get("cute_chained_vector_group", False)) is bool
        and (
            not config.get("cute_chained_vector_group", False)
            or config.get("cute_chained_pointwise_vectorize", False) is True
            and plan.shapes[0][0] == plan.shapes[0][1]
        )
        and all(
            config.get(key, default) == default
            for key, default in (
                ("cute_chained_pointwise_unroll", 1),
                ("cute_chained_pointwise_read_cache", False),
                ("cute_chained_auxiliary_cache", False),
                ("cute_chained_pointwise_inplace_async", False),
                ("cute_chained_startup_transfer", "legacy"),
                ("cute_chained_leaf_pipeline", "legacy"),
                ("cute_chained_tmem_free", "legacy"),
                ("cute_chained_seed_tile_columns", 0),
                ("cute_chained_snapshot_tile_columns", 0),
                ("cute_chained_pointwise_cache_layout", "auto"),
                ("cute_chained_scratch_layout", "row_major"),
            )
        )
        and supported_plan(plan)
    )


_INDEPENDENT_OPTIONS = (
    ("cute_chained_pointwise_vectorize", False),
    ("cute_chained_pointwise_unroll", 1),
    ("cute_chained_pointwise_read_cache", False),
    ("cute_chained_auxiliary_cache", False),
    ("cute_chained_tmem_early_release", False),
    ("cute_chained_direct_output", False),
    ("cute_chained_startup_transfer", "legacy"),
    ("cute_chained_pointwise_inplace_async", False),
)


def supports_independent_root(cg: GenerateAST, plan: ChainedMatmulPlan) -> bool:
    """One original sparse M64 stage, optionally with accepted startup inputs."""
    from .chained_result_transport import is_m64_plan
    from .chained_tcgen05 import supported_plan

    config = cg.device_function.config.config
    startup = config.get("cute_chained_startup_transfer", "legacy")
    return (
        plan.strategy == "tcgen05_tmem"
        and plan.loop is None
        and is_m64_plan(plan)
        and plan.dots[0].args[2] is None
        and plan.k_schedule is None
        and plan.pointwise_cache is None
        and not plan.warp_mma_stages
        and plan.preparation_pipeline is None
        and not uses_general_collectives(plan)
        and startup in ("legacy", "tma")
        and (
            startup == "tma"
            or not config.get("cute_chained_pointwise_inplace_async", False)
        )
        and all(not (chain._ancestors(scan) & set(plan.dots)) for scan in plan.scans)
        and all(
            config.get(key, default) == default
            for key, default in (
                ("cute_chained_vector_group", False),
                ("cute_chained_leaf_pipeline", "legacy"),
                ("cute_chained_tmem_free", "legacy"),
                ("cute_chained_seed_tile_columns", 0),
                ("cute_chained_pointwise_cache_layout", "auto"),
                ("cute_chained_scratch_layout", "row_major"),
            )
        )
        and supported_plan(plan)
    )


def supports_initialized_root(cg: GenerateAST, plan: ChainedMatmulPlan) -> bool:
    """Only the full-K first issue of an already-proved initialized pair.

    A separate seeded-SMEM selection can admit its final issue. Other final
    issues retain the original root scheduler. Paired TMA additionally requires
    the original post-completion raw-leaf publication witness. This does not
    discover reassociation.
    """
    from .chained_tcgen05 import supported_plan

    config = cg.device_function.config.config
    return (
        plan.strategy == "tcgen05_tmem"
        and plan.loop is None
        and len(plan.dots) == 2
        and plan.initialized_accumulator is not None
        and plan.initialized_accumulator.first is plan.dots[0]
        and plan.initialized_accumulator.second is plan.dots[1]
        and all(shape[0] == 128 for shape in plan.shapes)
        and all(dot.args[2] is None for dot in plan.dots)
        and (plan.k_schedule is None or plan.k_schedule.stage == 1)
        and plan.pointwise_cache is None
        and not plan.warp_mma_stages
        and plan.preparation_pipeline is None
        and not uses_general_collectives(plan)
        and not plan.scan_exports
        and config.get("cute_chained_leaf_pipeline", "legacy")
        in ("legacy", "paired_tma")
        and all(
            config.get(key, default) == default
            for key, default in (
                ("cute_chained_vector_group", False),
                ("cute_chained_startup_transfer", "legacy"),
                ("cute_chained_pointwise_cache_layout", "auto"),
                ("cute_chained_scratch_layout", "row_major"),
            )
        )
        and supported_plan(plan)
    )


def _initialized_schedule(plan: ChainedMatmulPlan) -> tuple[object, ...]:
    return (
        plan.axes,
        plan.shapes,
        plan.scans,
        plan.scan_exports,
        plan.initialized_accumulator,
        plan.late_rhs_reuse,
        plan.k_schedule,
        plan.direct_output,
    )


@dataclass(frozen=True)
class RootInitializedResult:
    """Receipt of the original seed lowering, not an arbitrary result callback.

    Capture immediately after codegen_seed, before prefetch/auxiliary lowering.
    The completed FP32 C is either loaded once here or panel-loaded by the
    original seed program. That program publishes the exact initialized seed
    to TMEM before the stage's original final CTA rendezvous; C is not a new
    pointwise boundary. This record grants no stage-1 input or issue authority.
    """

    codegen: GenerateAST
    plan: ChainedMatmulPlan
    revision: tuple[object, ...]
    axes: tuple[tuple[int, int], ...]
    boundaries: tuple[tuple[Node, str], ...]
    scans: tuple[chain._ScanInput, ...]
    schedule: tuple[object, ...]
    options: object
    max_columns: int
    lines: tuple[str, ...]
    paired: RootPairedLeaf | None
    _selection: tuple[object, ...] = field(repr=False)
    state_effects: tuple[StateEffect, ...] = ()

    def matches(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        axes: tuple[tuple[int, int], ...],
        scans: tuple[chain._ScanInput, ...],
    ) -> bool:
        return (
            cg is self.codegen
            and plan is self.plan
            and supports_initialized_root(cg, plan)
            and self._selection
            == (
                self.codegen,
                self.plan,
                self.revision,
                self.axes,
                self.boundaries,
                self.scans,
                self.schedule,
                self.options,
                self.max_columns,
                self.lines,
                self.paired,
                tuple(effect.facts() for effect in self.state_effects),
            )
            and self.revision == chain._register_bridge_revision(plan)
            and self.axes == axes
            and self.boundaries == tuple(boundaries.items())
            and self.scans == scans[: len(self.scans)]
            and self.schedule == _initialized_schedule(plan)
            and self.options == _startup_value(cg.device_function.config.config)
            and type(self.max_columns) is int
            and self.max_columns
            == cg.device_function.config.config.get("cute_chained_seed_tile_columns", 0)
            and (self.paired is not None)
            == (
                cg.device_function.config.config.get(
                    "cute_chained_leaf_pipeline", "legacy"
                )
                == "paired_tma"
            )
            and (
                self.paired is None
                or self.paired.matches(cg, plan, boundaries, axes, scans)
            )
        )


def capture_initialized_result(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scans: list[chain._ScanInput],
    inner_axes: dict[tuple[int, str], int],
    seed_tiling: SeedTiling,
    seed_lines: list[str],
    *,
    paired: RootPairedLeaf | None = None,
    state_effects: tuple[StateEffect, ...] = (),
) -> RootInitializedResult:
    """Consume successful codegen_seed at its original same-attempt call site."""
    if not supports_initialized_root(cg, plan) or not seed_lines:
        raise chain._UnsupportedChain("initialized root result capability changed")
    revision = chain._register_bridge_revision(plan)
    axes = tuple((inner_axes[i, "a"], inner_axes[i, "b"]) for i in range(2))
    bindings = tuple(boundaries.items())
    sources = tuple(scans)
    schedule = _initialized_schedule(plan)
    options = _startup_value(cg.device_function.config.config)
    lines = tuple(seed_lines)
    selected = (
        cg,
        plan,
        revision,
        axes,
        bindings,
        sources,
        schedule,
        options,
        seed_tiling.max_columns,
        lines,
        paired,
        tuple(effect.facts() for effect in state_effects),
    )
    return RootInitializedResult(
        cg,
        plan,
        revision,
        axes,
        bindings,
        sources,
        schedule,
        options,
        seed_tiling.max_columns,
        lines,
        paired,
        selected,
        state_effects,
    )


@dataclass(frozen=True)
class IndependentRootInputs:
    """Axes captured at the original expression-lowering point, not re-emitted.

    Unlike a direct-address probe, a weighted operand can lower pointwise
    expressions and allocate temporary names. Its original lowering runs once;
    the action checks this immutable result and exact semantic revision before
    using it. This record does not itself assert scan or copy completion.
    """

    plan: ChainedMatmulPlan
    revision: tuple[object, ...]
    boundaries: tuple[tuple[Node, str], ...]
    axes: tuple[tuple[int, int], ...]
    options: tuple[object, ...]
    schedule: tuple[object, ...]
    _selection: tuple[object, ...] = field(repr=False)

    def matches(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        axes: tuple[tuple[int, int], ...],
    ) -> bool:
        return (
            plan is self.plan
            and self._selection
            == (
                self.plan,
                self.revision,
                self.boundaries,
                self.axes,
                self.options,
                self.schedule,
            )
            and supports_independent_root(cg, plan)
            and self.revision == chain._register_bridge_revision(plan)
            and self.boundaries == tuple(boundaries.items())
            and self.axes == axes
            and self.options == _independent_options(cg)
            and self.schedule == (plan.axes, plan.scan_exports, plan.direct_output)
        )


def _independent_options(cg: GenerateAST) -> tuple[object, ...]:
    config = cg.device_function.config.config
    return tuple(starmap(config.get, _INDEPENDENT_OPTIONS))


def independent_root_inputs(
    cg: GenerateAST, plan: ChainedMatmulPlan, boundaries: dict[Node, str]
) -> IndependentRootInputs:
    """Capture the same ordered axis-selection calls used by the root scheduler."""
    if not supports_independent_root(cg, plan):
        raise chain._UnsupportedChain("independent root input capability changed")
    axes = tuple(
        chain._operand_inner_axis(
            cg,
            plan,
            {**boundaries, **{dot: f"chain_{i}_c" for i, dot in enumerate(plan.dots)}},
            cast("Node", operand),
        )
        for operand in plan.dots[0].args[:2]
    )
    revision = chain._register_bridge_revision(plan)
    bindings = tuple(boundaries.items())
    native_axes = ((axes[0], 1 - axes[1]),)
    options = _independent_options(cg)
    schedule = (plan.axes, plan.scan_exports, plan.direct_output)
    selection = (plan, revision, bindings, native_axes, options, schedule)
    return IndependentRootInputs(
        plan, revision, bindings, native_axes, options, schedule, selection
    )


def _startup_value(value: object) -> object:
    """Snapshot descriptor containers; their identity is not proof authority."""
    if isinstance(value, dict):
        return (dict, tuple((k, _startup_value(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_startup_value(v) for v in value))
    return (type(value), value)


def _paired_leaf_value(transfer: LeafPipeline) -> tuple[object, ...]:
    return (
        transfer.leaf,
        transfer.row,
        transfer.col,
        transfer.atom,
        transfer.tensor,
        _startup_value(transfer.wrapper),
    )


@dataclass(frozen=True)
class RootPairedLeaf:
    """Original admitted raw-leaf descriptor before seed construction.

    This witnesses the initial publication after stage0 completion and the
    original retired-half stage1 producer. Descriptor fields are checked by
    value before use; finish_wrapper adds out_name only later, after the
    original epilogue has registered its host arguments.
    """

    codegen: GenerateAST
    plan: ChainedMatmulPlan
    revision: tuple[object, ...]
    axes: tuple[tuple[int, int], ...]
    boundaries: tuple[tuple[Node, str], ...]
    scans: tuple[chain._ScanInput, ...]
    schedule: tuple[object, ...]
    options: object
    transfer: LeafPipeline
    descriptor: tuple[object, ...]
    _selection: tuple[object, ...] = field(repr=False)

    def matches_finalized(self) -> bool:
        """Only the original output argument may extend the used descriptor."""
        output = cast("Node", self.plan.store.args[0])
        argument = self.codegen.device_function._tensor_args.get(output.meta["val"])
        wrapper = self.transfer.wrapper
        return (
            argument is not None
            and wrapper.get("out_name") == argument.name
            and self.descriptor
            == (
                *_paired_leaf_value(self.transfer)[:-1],
                _startup_value({k: v for k, v in wrapper.items() if k != "out_name"}),
            )
            and sum(value is wrapper for value in self.codegen.cute_wrapper_plans) == 1
        )

    def matches(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        axes: tuple[tuple[int, int], ...],
        scans: tuple[chain._ScanInput, ...],
    ) -> bool:
        return (
            self._selection
            == (
                self.codegen,
                self.plan,
                self.revision,
                self.axes,
                self.boundaries,
                self.scans,
                self.schedule,
                self.options,
                self.transfer,
                self.descriptor,
            )
            and cg is self.codegen
            and plan is self.plan
            and self.revision == chain._register_bridge_revision(plan)
            and self.axes == axes
            and self.boundaries == tuple(boundaries.items())
            and self.scans == scans[: len(self.scans)]
            and self.schedule == _initialized_schedule(plan)
            and self.options == _startup_value(cg.device_function.config.config)
            and self.descriptor == _paired_leaf_value(self.transfer)
            and sum(
                wrapper is self.transfer.wrapper for wrapper in cg.cute_wrapper_plans
            )
            == 1
            and self.transfer.wrapper["kernel_args"]
            == [self.transfer.atom, self.transfer.tensor]
        )


def capture_paired_leaf(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scans: list[chain._ScanInput],
    inner_axes: dict[tuple[int, str], int],
    transfer: LeafPipeline,
) -> RootPairedLeaf:
    """Capture only the successful original plan_leaf_pipeline result in place."""
    if (
        not supports_initialized_root(cg, plan)
        or cg.device_function.config.config.get("cute_chained_leaf_pipeline")
        != "paired_tma"
    ):
        raise chain._UnsupportedChain("paired root leaf capability changed")
    selection = (
        cg,
        plan,
        chain._register_bridge_revision(plan),
        tuple((inner_axes[i, "a"], inner_axes[i, "b"]) for i in range(2)),
        tuple(boundaries.items()),
        tuple(scans),
        _initialized_schedule(plan),
        _startup_value(cg.device_function.config.config),
        transfer,
        _paired_leaf_value(transfer),
    )
    return RootPairedLeaf(*selection, selection)


@dataclass(frozen=True)
class RootInputLayout:
    """Original stage-one SMEM owner and native operand layout, before issue."""

    operand: Node
    shape: tuple[int, int]
    inner: int
    dtype: torch.dtype
    owner_elements: int
    separate: bool


def _pair_layouts(
    plan: ChainedMatmulPlan, axes: tuple[tuple[int, int], ...]
) -> tuple[RootInputLayout, ...]:
    # Stage-one A is the existing packed TMEM bridge, not a new SMEM owner.
    return tuple(
        RootInputLayout(
            cast("Node", plan.dots[stage].args[role]),
            (plan.shapes[stage][role], plan.shapes[stage][2]),
            axes[stage][role],
            plan.operand_dtype(stage),
            max(shape[role] * shape[2] for shape in plan.shapes),
            False,
        )
        for stage, role in ((0, 0), (0, 1), (1, 1))
    )


def _pair_bridge_facts(bridge: chain._RegisterBridge) -> tuple[object, ...]:
    return (
        bridge.role,
        bridge.lines,
        bridge.stage,
        bridge.source,
        bridge.operand,
        _startup_value(bridge.revision),
        bridge.epilogue_facts(),
    )


def _pair_revision(plan: ChainedMatmulPlan) -> object:
    return _startup_value((chain._register_bridge_revision(plan), plan.axes))


def _group_operand_facts(operands: tuple[VectorStageOperand, ...]) -> object:
    return _startup_value(
        tuple(
            (
                operand.node,
                operand.node.meta["val"].dtype,
                operand.geometry.logical,
                operand.geometry.transpose,
                operand.geometry.native_rows,
                operand.geometry.physical,
                operand.role,
                operand.shape,
                operand.target,
                operand.offset,
            )
            for operand in operands
        )
    )


@dataclass(frozen=True)
class RootGroupedProduction:
    """One complete original grouped fill; not copy or MMA readiness."""

    operands: tuple[VectorStageOperand, ...]
    execution: tuple[object, ...]
    body: tuple[str, ...]

    def facts(self) -> object:
        return (_group_operand_facts(self.operands), self.execution, self.body)


@dataclass(frozen=True)
class RootPairInputs:
    """Original weighted axes/owners and exclusive bridge, not copy readiness."""

    codegen: GenerateAST
    plan: ChainedMatmulPlan
    revision: object
    boundaries: tuple[tuple[Node, str], ...]
    axes: tuple[tuple[int, int], ...]
    layouts: tuple[RootInputLayout, ...]
    bridge: chain._RegisterBridge
    options: object
    _selection: tuple[object, ...] = field(repr=False)

    def matches(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        axes: tuple[tuple[int, int], ...],
        stage: int = 0,
    ) -> bool:
        expected = dict(self.boundaries)
        if stage == 1:
            expected[self.plan.dots[0]] = "chain_0_c"
        return (
            stage in (0, 1)
            and cg is self.codegen
            and plan is self.plan
            and supports_local_root_pair(cg, plan)
            and self._selection
            == (
                self.codegen,
                self.plan,
                self.revision,
                self.boundaries,
                self.axes,
                _pair_bridge_facts(self.bridge),
                self.options,
            )
            and self.revision == _pair_revision(plan)
            and expected == boundaries
            and self.axes == axes
            and all(
                type(axis) is int and axis in (0, 1) for pair in axes for axis in pair
            )
            and self.layouts == _pair_layouts(plan, axes)
            and self.options == _startup_value(cg.device_function.config.config)
        )


def capture_root_pair_inputs(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    inner_axes: dict[tuple[int, str], int],
    bridges: dict[int, chain._RegisterBridge],
) -> RootPairInputs | None:
    """Accept existing proofs after all four original axis evaluations."""
    if not supports_local_root_pair(cg, plan):
        return None
    if (
        any(type(axis) is not int or axis not in (0, 1) for axis in inner_axes.values())
        or set(bridges) != {1}
        or bridges[1].role != "a"
        or bridges[1].stage != 1
        or bridges[1].source is not plan.dots[0]
        or bridges[1].operand is not plan.dots[1].args[0]
        or bridges[1].revision != chain._register_bridge_revision(plan)
        or inner_axes[0, "a"] != 1
        or inner_axes[0, "b"] != 1
        or inner_axes[1, "a"] != 1
        or type(inner_axes[1, "b"]) is not int
        or inner_axes[1, "b"] not in (0, 1)
        or plan.shapes[0][:2] != (plan.shapes[1][0], plan.shapes[1][2])
        or plan.dots[0].meta["val"].dtype != torch.float32
    ):
        return None
    axes = tuple((inner_axes[i, "a"], inner_axes[i, "b"]) for i in range(2))
    revision = _pair_revision(plan)
    bindings = tuple(boundaries.items())
    options = _startup_value(cg.device_function.config.config)
    selected = (
        cg,
        plan,
        revision,
        bindings,
        axes,
        _pair_bridge_facts(bridges[1]),
        options,
    )
    return RootPairInputs(
        cg,
        plan,
        revision,
        bindings,
        axes,
        _pair_layouts(plan, axes),
        bridges[1],
        options,
        selected,
    )


def _seeded_layouts(
    result: RootInitializedResult, plan: ChainedMatmulPlan, mode: str
) -> tuple[RootInputLayout, RootInputLayout]:
    m, n, k = plan.shapes[1]

    def layout(role: int, extent: int) -> RootInputLayout:
        return RootInputLayout(
            cast("Node", plan.dots[1].args[role]),
            (extent, k),
            result.axes[1][role],
            plan.operand_dtype(1),
            extent * k
            if role == 1 and mode == "upfront"
            else max(shape[role] * shape[2] for shape in plan.shapes),
            role == 1 and mode == "upfront",
        )

    return layout(0, m), layout(1, n)


@dataclass(frozen=True)
class RootSeededInputs:
    """Original initialized-stage inputs, independent of arena reuse.

    Construction is not runtime completion. Deferred lines must be enqueued
    after the seed; upfront lines are drained by stage zero; local inputs are
    constructed and drained by stage one. No packed-TMEM A is involved.
    """

    result: RootInitializedResult
    rhs: Node
    inner_axis: int
    lines: tuple[str, ...]
    arena: LateRhsArenaPlan | None
    k_schedule: KSchedule | None
    mode: Literal["deferred", "upfront", "local"]
    layouts: tuple[RootInputLayout, RootInputLayout]
    _selection: tuple[object, ...] = field(repr=False)

    def matches(self, plan: ChainedMatmulPlan) -> bool:
        from .chained_k_schedule import resolve_k_schedule
        from .chained_late_rhs import resolve_late_rhs
        from .chained_tcgen05 import _prefetch_final_b

        return (
            self.result.plan is plan
            and self._selection
            == (
                self.result,
                self.rhs,
                self.inner_axis,
                self.lines,
                self.arena,
                self.k_schedule,
                self.mode,
                self.layouts,
            )
            and self.rhs is plan.dots[1].args[1]
            and self.inner_axis == self.result.axes[1][1]
            and self.layouts == _seeded_layouts(self.result, plan, self.mode)
            and _prefetch_final_b(plan) == (self.mode != "local")
            and (
                self.mode == "deferred"
                and self.arena is not None
                and self.arena == plan.late_rhs_reuse == resolve_late_rhs(plan)
                and bool(self.lines)
                or self.mode in ("upfront", "local")
                and self.arena is None
                and plan.late_rhs_reuse is None
                and self.result.paired is None
                and self.k_schedule is None
                and bool(self.lines) == (self.mode == "upfront")
            )
            and self.k_schedule == plan.k_schedule
            and (
                self.k_schedule is None
                or self.k_schedule.mode in ("serial64", "overlap64")
                and self.result.axes[1][0] == 1
                and self.k_schedule == resolve_k_schedule(plan, self.k_schedule.mode)
            )
        )


def capture_seeded_inputs(
    result: RootInitializedResult,
    plan: ChainedMatmulPlan,
    lines: list[str],
    *,
    mode: Literal["deferred", "upfront", "local"] = "deferred",
) -> RootSeededInputs | None:
    """Capture the original construction site, without lowering it again."""
    if (
        result.plan is not plan
        or result.paired is not None
        and plan.k_schedule is None
        or mode == "deferred"
        and (plan.late_rhs_reuse is None or not lines)
        or mode != "deferred"
        and (plan.late_rhs_reuse is not None or plan.k_schedule is not None)
        or plan.k_schedule is not None
        and plan.k_schedule.mode not in ("serial64", "overlap64")
    ):
        return None
    rhs = cast("Node", plan.dots[1].args[1])
    selected = (
        result,
        rhs,
        result.axes[1][1],
        tuple(lines),
        plan.late_rhs_reuse,
        plan.k_schedule,
        mode,
        _seeded_layouts(result, plan, mode),
    )
    candidate = RootSeededInputs(*selected, selected)
    return candidate if candidate.matches(plan) else None


@dataclass(frozen=True)
class RootSeedCompletion:
    """The exact result whose seed fence and CTA join were emitted by stage0."""

    result: RootInitializedResult


@dataclass(frozen=True)
class RootQueuedRhs:
    """These original B copies were enqueued after this seed, not yet waited."""

    inputs: RootSeededInputs
    seed: RootSeedCompletion


@dataclass(frozen=True)
class RootInputCompletion:
    """The given stage's original copy wait/fence/join, not future local copies."""

    inputs: RootSeededInputs
    stage: int


def _execution_fields(execution: ChainedExecution) -> tuple[object, ...]:
    return (
        execution.threads,
        execution.thread,
        execution.warp,
        execution.sync,
        execution.a_workspace,
        execution.b_workspace,
        execution.tmem,
        execution.barriers,
    )


@dataclass(frozen=True)
class RootHalfProducer:
    """Original half producer captured during this action's A construction."""

    inputs: RootSeededInputs
    operand: Node
    lines: tuple[str, ...]
    _selection: tuple[object, ...] = field(repr=False)

    def matches(self, sequence: RootStageSequence) -> bool:
        return (
            self.inputs is sequence.seeded_inputs
            and self.operand is sequence.plan.dots[1].args[0]
            and self._selection == (self.inputs, self.operand, self.lines)
            and bool(self.lines)
        )


def _startup_facts(transfers: tuple[StartupInput, ...]) -> tuple[object, ...]:
    return tuple(
        (
            t.role,
            t.operand,
            t.leaf,
            t.coordinates,
            t.shape,
            t.inner,
            t.row,
            t.col,
            _startup_value(t.wrapper),
        )
        for t in transfers
    )


@dataclass(frozen=True)
class RootStartupInputs:
    """Receipt of the original late startup proof, captured before issue.

    The scheduler issues these exact transfers before constructing the stage
    sequence. The action still emits both original completion waits; this
    compile-time receipt never stands in for runtime transfer completion.
    """

    plan: ChainedMatmulPlan
    transfers: tuple[StartupInput, ...]
    axes: tuple[tuple[int, int], ...]
    wrappers: object
    parameters: tuple[str, ...]
    _selection: tuple[object, ...] = field(repr=False)

    def matches(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        axes: tuple[tuple[int, int], ...],
    ) -> bool:
        return (
            plan is self.plan
            and self.axes == axes
            and self._selection
            == (
                self.plan,
                self.axes,
                _startup_facts(self.transfers),
                self.wrappers,
                self.parameters,
            )
            and self.wrappers == _startup_value(cg.cute_wrapper_plans)
            and self.parameters == tuple(cg.device_function.wrapper_only_params)
        )


def capture_root_startup(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    transfers: list[StartupInput],
    axes: tuple[tuple[int, int], ...],
) -> RootStartupInputs:
    """Called immediately after the original plan_startup, without re-lowering."""
    roles = tuple(t.role for t in transfers)
    if (
        not transfers
        or roles not in (("a",), ("b",), ("a", "b"))
        or len(axes) != 1
        or any(
            t.operand is not plan.dots[0].args[0 if t.role == "a" else 1]
            or t.shape != (plan.shapes[0][0 if t.role == "a" else 1], plan.shapes[0][2])
            or t.inner != axes[0][0 if t.role == "a" else 1]
            or t.leaf not in chain._ancestors(t.operand)
            or t.leaf.meta["val"].dtype != plan.operand_dtype(0)
            or t.wrapper not in cg.cute_wrapper_plans
            for t in transfers
        )
    ):
        raise chain._UnsupportedChain("startup input selection does not match root")
    selected = tuple(transfers)
    wrappers = _startup_value(cg.cute_wrapper_plans)
    parameters = tuple(cg.device_function.wrapper_only_params)
    return RootStartupInputs(
        plan,
        selected,
        axes,
        wrappers,
        parameters,
        (plan, axes, _startup_facts(selected), wrappers, parameters),
    )


def _native_axes(
    cg: GenerateAST, plan: ChainedMatmulPlan
) -> tuple[tuple[int, int], ...]:
    """Recheck direct SMEM views without allocating expression temporaries."""
    from .chained_stage_operands import _AddressProbe

    def axis(stage: int, role: int) -> int:
        operand = cast("Node", plan.dots[stage].args[role])
        probe = _AddressProbe(
            cg, plan, operand, ("chain_coordinate_m", "chain_coordinate_n")
        )
        inner = chain._operand_inner_axis(cg, plan, {}, operand, expression=probe)
        return inner if role == 0 else 1 - inner

    return ((axis(0, 0), axis(0, 1)), (1, axis(1, 1)))


@dataclass
class RootStageSequence:
    """Codegen-local ordered readiness ledger for an accepted physical pair.

    Construction follows the original final-B prefetch and late bridge proof.
    Stage zero's full copy wait publishes that B and its completed FP32 result.
    Stage one's bridge reads every original FP32 value before overwriting it,
    either as a full fragment or through an explicit forward-panel proof.
    Its C arena starts after the maximum old C
    span. No SMEM publication/alias or lifetime is inferred from a symbol name.
    """

    plan: ChainedMatmulPlan
    facts: tuple[object, ...]
    geometries: tuple[StageGeometry, ...]
    inner_axes: tuple[tuple[int, int], ...]
    scans: tuple[chain._ScanInput, ...]
    scan_lines: tuple[str, ...]
    pointwise_unroll: PointwiseUnroll
    pointwise_cache: PointwiseReadCache
    pointwise_inplace: PointwiseInplace
    early_release: bool
    independent: IndependentRootInputs | None = None
    pair_inputs: RootPairInputs | None = None
    pair_selection: RootPairInputs | None = None
    pair_boundaries: dict[Node, str] | None = None
    pair_completed: int = 0
    pair_bridged: bool = False
    pair_staging: VectorStaging | None = None
    pair_staging_selection: tuple[object, ...] = ()
    grouped_state: tuple[bool, bool, bool] = (False, False, False)
    grouped_attempted: bool = False
    grouped_required: bool = False
    grouped_production: RootGroupedProduction | None = None
    grouped_obligation: object = None
    early_scan: bool = False
    early_cached: tuple[chain._ScanInput, ...] = ()
    input_readiness: tuple[object, ...] = ()
    startup: RootStartupInputs | None = None
    startup_issued: bool = False
    initialized: RootInitializedResult | None = None
    seeded_inputs: RootSeededInputs | None = None
    seed_completion: RootSeedCompletion | None = None
    queued_rhs: RootQueuedRhs | None = None
    input_completion: RootInputCompletion | None = None
    input_execution: ChainedExecution | None = None
    input_stage: int | None = None
    input_context: tuple[object, ...] = ()
    input_producers: tuple[str, ...] = ()
    input_committed: bool = False
    input_waited: bool = False
    half_producer: RootHalfProducer | None = None
    paired_issued: bool = False
    snapshot: RootSnapshot | None = None
    snapshot_ready: bool = False
    snapshot_consumed: bool = False
    next_stage: int = 0
    staged: list[chain._StagedInput] = field(default_factory=list)
    _rhs_completion: RootRhsCompletion | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def action(self, stage: int) -> RootStageAction:
        return RootStageAction(self, stage)

    def handles(self, stage: int) -> bool:
        """Only an accepted seeded-SMEM selection adds an initialized stage one."""
        return self.initialized is None or stage == 0 or self.seeded_inputs is not None

    def enqueue_rhs(self, lines: list[str]) -> None:
        """Called only after appending the original deferred RHS after stage0."""
        selected = self.seeded_inputs
        if selected is None:
            return
        if selected.mode != "deferred":
            if lines:
                raise chain._UnsupportedChain("unexpected deferred RHS construction")
            return
        if (
            self.next_stage != 1
            or self.seed_completion is None
            or self.seed_completion.result is not self.initialized
            or selected.result is not self.initialized
            or tuple(lines) != selected.lines
            or self.queued_rhs is not None
        ):
            raise chain._UnsupportedChain("initialized RHS enqueue order changed")
        before = root_stage_progress(self)
        self.queued_rhs = RootQueuedRhs(selected, self.seed_completion)
        self._rhs_completion = RootRhsCompletion(
            self, self.queued_rhs, before, root_stage_progress(self)
        )


def root_stage_state(sequence: RootStageSequence) -> object:
    """Deep codegen-local transition facts, never another readiness policy."""
    return _startup_value(
        (
            sequence.next_stage,
            sequence.pair_completed,
            sequence.pair_bridged,
            sequence.pair_boundaries,
            sequence.input_stage,
            sequence.input_context,
            sequence.input_producers,
            sequence.input_committed,
            sequence.input_waited,
            sequence.input_completion,
            (
                _execution_fields(sequence.input_execution)
                if sequence.input_execution is not None
                else None
            ),
            sequence.snapshot_ready,
            sequence.snapshot_consumed,
            sequence.grouped_state,
            sequence.grouped_attempted,
            sequence.grouped_required,
            sequence.grouped_production.facts()
            if sequence.grouped_production is not None
            else None,
            sequence.grouped_obligation,
            tuple(vars(x) for x in sequence.staged),
            sequence.seed_completion
            if not isinstance(sequence.seed_completion, RootSeedCompletion)
            else (
                id(sequence.seed_completion),
                sequence.seed_completion,
                id(sequence.seed_completion.result),
                sequence.seed_completion.result,
            ),
            sequence.queued_rhs
            if not isinstance(sequence.queued_rhs, RootQueuedRhs)
            else (
                id(sequence.queued_rhs),
                sequence.queued_rhs,
                id(sequence.queued_rhs.inputs),
                sequence.queued_rhs.inputs,
                id(sequence.queued_rhs.seed),
                sequence.queued_rhs.seed,
            ),
            sequence.half_producer
            if not isinstance(sequence.half_producer, RootHalfProducer)
            else (
                id(sequence.half_producer),
                sequence.half_producer,
                sequence.half_producer.inputs,
                sequence.half_producer.operand,
                sequence.half_producer.lines,
                sequence.half_producer._selection,
            ),
            sequence.paired_issued,
        )
    )


def root_stage_progress(sequence: RootStageSequence) -> object:
    return _startup_value(
        (root_stage_state(sequence), tuple(sequence.plan.tensor_aliases.items()))
    )


@dataclass(frozen=True)
class RootRhsCompletion:
    """The original enqueue transition, with no copy-completion authority."""

    sequence: RootStageSequence
    queued: RootQueuedRhs
    before: object
    progress: object

    def matches(self, sequence: RootStageSequence) -> bool:
        return (
            self.sequence is sequence
            and sequence._rhs_completion is self
            and sequence.queued_rhs is self.queued
            and self.queued.inputs is sequence.seeded_inputs
            and self.queued.seed is sequence.seed_completion
            and self.progress == root_stage_progress(sequence)
        )


@dataclass(frozen=True)
class RootStageCompletion:
    """Original successful transition; graph/allocation authority stays external."""

    sequence: RootStageSequence
    stage: int
    progress: object

    def matches(self, action: RootStageAction) -> bool:
        return (
            action._completion is self
            and action.sequence is self.sequence
            and action.stage == self.stage
            and self.progress == root_stage_progress(self.sequence)
        )


@dataclass(frozen=True)
class RootStageAction:
    sequence: RootStageSequence
    stage: int
    _completion: RootStageCompletion | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def result(self) -> Literal["seed_only", "logical_fragment"]:
        return (
            "seed_only"
            if self.stage == 0 and self.sequence.initialized is not None
            else "logical_fragment"
        )

    def _check_grouped_state(self) -> None:
        sequence = self.sequence
        pair = sequence.pair_selection
        requested = (
            pair is not None
            and pair.codegen.device_function.config.config.get(
                "cute_chained_vector_group", False
            )
            is True
        )
        staging = sequence.pair_staging
        if not requested:
            valid = (
                staging is None
                and not sequence.pair_staging_selection
                and not sequence.grouped_attempted
                and not sequence.grouped_required
                and sequence.grouped_production is None
                and sequence.grouped_obligation is None
            )
        else:
            valid = (
                staging is not None
                and sequence.pair_staging_selection
                == (
                    staging,
                    sequence.pointwise_unroll,
                    _startup_value(
                        (staging.enabled, staging.group_enabled, staging.broadcast)
                    ),
                    sequence.pointwise_unroll.factor,
                )
                and staging.enabled is True
                and staging.group_enabled is True
                and staging.broadcast is None
                and type(sequence.grouped_attempted) is bool
                and type(sequence.grouped_required) is bool
                and all(type(value) is bool for value in sequence.grouped_state)
                and sequence.grouped_state
                == (
                    staging.activated,
                    staging.group_activated,
                    sequence.pointwise_unroll.activated,
                )
            )
            receipt = sequence.grouped_production
            if sequence.grouped_required:
                valid = (
                    valid
                    and sequence.grouped_attempted
                    and receipt is not None
                    and sequence.grouped_obligation == (receipt, receipt.facts())
                    and receipt.execution == sequence.input_context
                    and _group_operand_facts(receipt.operands)
                    == _group_operand_facts(self._group_operands())
                )
            else:
                valid = (
                    valid and receipt is None and sequence.grouped_obligation is None
                )
        if not valid:
            raise chain._UnsupportedChain("root grouped producer authority changed")

    def _group_operands(self) -> tuple[VectorStageOperand, ...]:
        from .chained_vector_stage import VectorStageOperand

        sequence = self.sequence
        m, _, k = sequence.plan.shapes[0]
        return tuple(
            VectorStageOperand(
                cast("Node", node),
                sequence.geometries[0],
                role,
                (m, k),
                f"chain_0_{role}",
            )
            for role, node in zip(
                ("a", "b"), sequence.plan.dots[0].args[:2], strict=True
            )
        )

    @property
    def grouped_operands(self) -> bool:
        return self.stage == 0 and self.sequence.pair_staging is not None

    def grouped_operand_lines(
        self,
        cg: GenerateAST,
        boundaries: dict[Node, str],
        operands: tuple[VectorStageOperand, ...],
    ) -> list[str] | None:
        """Use the ordinary group helper once, at the original producer point."""
        from .chained_pointwise_unroll import PointwiseUnroll
        from .chained_vector_stage import VectorStaging
        from .chained_vector_stage import emit_vector_stage_group

        self._check_input_context()
        self._check_pair_arguments(cg, boundaries)
        sequence = self.sequence
        staging = sequence.pair_staging
        operand_facts = _group_operand_facts(operands)
        if (
            self.stage != 0
            or staging is None
            or sequence.grouped_attempted
            or sequence.input_producers
            or sequence.input_committed
            or sequence.input_waited
            or operand_facts != _group_operand_facts(self._group_operands())
        ):
            raise chain._UnsupportedChain(
                "root grouped producer order or operands changed"
            )
        trial = VectorStaging(
            staging.enabled,
            staging.activated,
            staging.group_enabled,
            staging.group_activated,
        )
        unroll = PointwiseUnroll(
            sequence.pointwise_unroll.factor, sequence.pointwise_unroll.activated
        )
        lines = emit_vector_stage_group(
            cg,
            sequence.plan,
            boundaries,
            operands,
            trial,
            tag="chain_0_vector_group",
            producer_unroll=unroll,
            execution=sequence.input_execution,
        )
        # Failed or stale probes never publish activation or producer roles.
        self._check_input_context()
        self._check_pair_arguments(cg, boundaries)
        if (
            _group_operand_facts(operands) != operand_facts
            or _group_operand_facts(self._group_operands()) != operand_facts
        ):
            raise chain._UnsupportedChain("root grouped producer operands changed")
        if lines is not None and (
            not lines
            or trial.activated is not True
            or trial.group_activated is not True
        ):
            raise chain._UnsupportedChain("root grouped producer did not complete")
        sequence.grouped_attempted = True
        if lines is None:
            return None
        receipt = RootGroupedProduction(operands, sequence.input_context, tuple(lines))
        sequence.grouped_obligation = (receipt, receipt.facts())
        sequence.grouped_production = receipt
        sequence.grouped_required = True
        staging.activated, staging.group_activated = (
            trial.activated,
            trial.group_activated,
        )
        sequence.pointwise_unroll.activated = unroll.activated
        sequence.grouped_state = (
            staging.activated,
            staging.group_activated,
            unroll.activated,
        )
        sequence.input_producers = ("a", "b")
        return lines

    def check_grouped_body(
        self, cg: GenerateAST, boundaries: dict[Node, str], lines: list[str]
    ) -> None:
        """The common stage installs only the completed, witnessed fill body."""
        self._check_input_context()
        self._check_pair_arguments(cg, boundaries)
        receipt = self.sequence.grouped_production
        if receipt is None or tuple(lines) != receipt.body:
            raise chain._UnsupportedChain("root grouped producer body changed")

    def _local_input_protocol(self) -> bool:
        selected = self.sequence.seeded_inputs
        return (
            self.sequence.pair_selection is not None
            or bool(self.sequence.input_context)
            or (selected is not None and selected.mode != "deferred")
        )

    def _check_input_context(self) -> None:
        self._check_grouped_state()
        sequence = self.sequence
        pair = sequence.pair_selection
        selected = (
            sequence.pair_inputs is pair
            and sequence.pair_boundaries is not None
            and pair.matches(
                pair.codegen,
                sequence.plan,
                sequence.pair_boundaries,
                sequence.inner_axes,
                self.stage,
            )
            if pair is not None
            else sequence.seeded_inputs is not None
            and sequence.seeded_inputs.matches(sequence.plan)
        )
        if (
            self.stage != sequence.input_stage
            or self.stage != sequence.next_stage
            or sequence.input_execution is None
            or sequence.input_context != _execution_fields(sequence.input_execution)
            or not selected
            or sequence.facts != chain._register_bridge_revision(sequence.plan)
        ):
            raise chain._UnsupportedChain("initialized input context changed")

    def _check_pair_arguments(
        self, cg: GenerateAST, boundaries: dict[Node, str]
    ) -> None:
        pair = self.sequence.pair_selection
        if pair is not None and not pair.matches(
            cg, self.sequence.plan, boundaries, self.sequence.inner_axes, self.stage
        ):
            raise chain._UnsupportedChain("root pair input arguments changed")

    @property
    def major_modes(self) -> tuple[str, str]:
        a, b = self.sequence.inner_axes[self.stage]
        return (
            "K" if (self.stage and not self.seeded_accumulator) or a else "MN",
            "K" if b else "MN",
        )

    @property
    def seeded_accumulator(self) -> bool:
        return self.stage == 1 and self.sequence.seeded_inputs is not None

    @property
    def k_schedule(self) -> KSchedule | None:
        selected = self.sequence.seeded_inputs
        return selected.k_schedule if self.stage == 1 and selected is not None else None

    def half_lines(self) -> tuple[str, ...]:
        producer = self.sequence.half_producer
        if (
            producer is None
            or not producer.matches(self.sequence)
            or not producer.inputs.matches(self.sequence.plan)
        ):
            raise chain._UnsupportedChain("initialized half producer changed")
        return producer.lines

    @property
    def retire_each_half(self) -> bool:
        """Paired raw panels reuse A, even for an overlap64 K schedule."""
        selected = self.sequence.seeded_inputs
        return (
            self.stage == 1
            and selected is not None
            and selected.result.paired is not None
        )

    def post_issue_lines(self) -> list[str]:
        """Original raw publication after MMA retirement, before FP32 seed loads."""
        sequence = self.sequence
        result = sequence.initialized
        if result is None or result.paired is None:
            return []
        if self.stage == 1:
            if (
                sequence.next_stage != 1
                or sequence.paired_issued is not True
                or sequence.seeded_inputs is None
                or sequence.seeded_inputs.result is not result
                or not sequence.seeded_inputs.matches(sequence.plan)
                or sequence.seed_completion is None
                or sequence.seed_completion.result is not result
                or sequence.queued_rhs is None
                or sequence.queued_rhs.inputs is not sequence.seeded_inputs
                or sequence.queued_rhs.seed is not sequence.seed_completion
                or not result.matches(
                    result.codegen,
                    sequence.plan,
                    dict(result.boundaries),
                    sequence.inner_axes,
                    sequence.scans,
                )
            ):
                raise chain._UnsupportedChain("paired half completion changed")
            self.half_lines()
            # The original half producer already issues/waits its raw panels;
            # the shared K issuer retires both halves. Never issue panel0 again.
            return []
        if (
            self.stage != 0
            or sequence.next_stage != 0
            or sequence.paired_issued is not False
            or sequence.seed_completion is not None
            or sequence.queued_rhs is not None
            or sequence.seeded_inputs is not None
            and (
                sequence.seeded_inputs.result is not result
                or not sequence.seeded_inputs.matches(sequence.plan)
            )
            or not result.matches(
                result.codegen,
                sequence.plan,
                dict(result.boundaries),
                sequence.inner_axes,
                sequence.scans,
            )
        ):
            raise chain._UnsupportedChain("paired initial publication changed")
        sequence.paired_issued = True
        return [
            "cute.arch.fence_view_async_shared()",
            "cute.arch.sync_threads()",
            *result.paired.transfer.issue("0", "0"),
        ]

    @property
    def tmem_input(self) -> TmemOperandBinding | None:
        if self.stage == 0 or self.seeded_accumulator:
            return None
        sequence = self.sequence
        geometry = sequence.geometries[self.stage]
        m, _, k = geometry.physical
        return TmemOperandBinding(
            ContractionGroup((self.stage,), (geometry,)),
            (m, k),
            sequence.plan.operand_dtype(self.stage),
            0,
        )

    @property
    def accumulator(self) -> str:
        offset = (
            max(n for _, n, _ in self.sequence.plan.shapes)
            if self.stage and not self.seeded_accumulator
            else 0
        )
        return f"chain_tptr + {offset}"

    def validate(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        stage: int,
        geometry: StageGeometry,
        execution: ChainedExecution,
        tmem_input: TmemOperandBinding | None,
        accumulator: str | None,
        pending_allocation: bool | None,
        terminal_fragment: bool,
        boundaries: dict[Node, str],
    ) -> None:
        sequence = self.sequence
        self._check_grouped_state()
        if sequence.initialized is not None:
            inputs_valid = (
                (stage == 0 or self.seeded_accumulator)
                and type(sequence.early_release) is bool
                and sequence.early_release
                == cg.device_function.config.config.get(
                    "cute_chained_tmem_early_release", False
                )
                and sequence.independent is None
                and sequence.startup is None
                and sequence.startup_issued is False
                and type(sequence.paired_issued) is bool
                and sequence.paired_issued
                == (stage == 1 and sequence.initialized.paired is not None)
                and sequence.initialized.matches(
                    cg, plan, boundaries, sequence.inner_axes, sequence.scans
                )
                and sequence.input_readiness
                == (
                    sequence.scans,
                    sequence.scan_lines,
                    sequence.early_scan,
                    sequence.early_cached,
                )
                and sequence.early_scan
                == any(
                    chain._ancestors(cast("Node", operand)) & set(plan.scans)
                    for operand in plan.dots[0].args[:2]
                )
                and sequence.pointwise_unroll.factor
                == cg.device_function.config.config.get(
                    "cute_chained_pointwise_unroll", 1
                )
                and sequence.pointwise_cache.enabled
                == cg.device_function.config.config.get(
                    "cute_chained_pointwise_read_cache", False
                )
                and sequence.pointwise_inplace.enabled
                == cg.device_function.config.config.get(
                    "cute_chained_pointwise_inplace_async", False
                )
                and (
                    stage == 0
                    and sequence.seed_completion is None
                    and sequence.queued_rhs is None
                    or stage == 1
                    and sequence.seeded_inputs is not None
                    and sequence.seeded_inputs.result is sequence.initialized
                    and sequence.seeded_inputs.matches(plan)
                    and sequence.seed_completion is not None
                    and sequence.seed_completion.result is sequence.initialized
                    and (
                        sequence.seeded_inputs.mode == "deferred"
                        and sequence.queued_rhs is not None
                        and sequence.queued_rhs.inputs is sequence.seeded_inputs
                        and sequence.queued_rhs.seed is sequence.seed_completion
                        or sequence.seeded_inputs.mode != "deferred"
                        and sequence.queued_rhs is None
                        and sequence.input_completion
                        == RootInputCompletion(sequence.seeded_inputs, 0)
                    )
                    and sequence.half_producer is None
                )
            )
        elif sequence.pair_selection is not None or sequence.pair_inputs is not None:
            pair = sequence.pair_selection
            inputs_valid = (
                pair is not None
                and sequence.pair_inputs is pair
                and pair.matches(cg, plan, boundaries, sequence.inner_axes, stage)
                and sequence.independent is None
                and sequence.startup is None
                and sequence.startup_issued is False
                and sequence.seeded_inputs is None
                and sequence.seed_completion is None
                and sequence.queued_rhs is None
                and sequence.snapshot is None
                and not sequence.scans
                and not sequence.scan_lines
                and sequence.early_scan is False
                and not sequence.early_cached
                and sequence.pointwise_unroll.factor == 1
                and sequence.pointwise_cache.enabled is False
                and sequence.pointwise_inplace.enabled is False
                and type(sequence.pair_completed) is int
                and sequence.pair_completed == stage
                and sequence.pair_bridged is False
                and type(sequence.early_release) is bool
                and sequence.early_release
                == cg.device_function.config.config.get(
                    "cute_chained_tmem_early_release", False
                )
            )
        elif sequence.independent is not None:
            inputs_valid = (
                sequence.independent.matches(cg, plan, boundaries, sequence.inner_axes)
                and sequence.geometries[0].native_rows == 64
                and sequence.early_scan
                == any(
                    chain._ancestors(cast("Node", operand)) & set(plan.scans)
                    for operand in plan.dots[0].args[:2]
                )
                and sequence.input_readiness
                == (
                    sequence.scans,
                    sequence.scan_lines,
                    sequence.early_scan,
                    sequence.early_cached,
                )
                and sequence.pointwise_unroll.factor
                == cg.device_function.config.config.get(
                    "cute_chained_pointwise_unroll", 1
                )
                and sequence.pointwise_cache.enabled
                == cg.device_function.config.config.get(
                    "cute_chained_pointwise_read_cache", False
                )
                and sequence.pointwise_inplace.enabled
                == cg.device_function.config.config.get(
                    "cute_chained_pointwise_inplace_async", False
                )
                and type(sequence.startup_issued) is bool
                and sequence.startup_issued == (sequence.startup is not None)
                and (sequence.startup is not None)
                == (
                    cg.device_function.config.config.get(
                        "cute_chained_startup_transfer", "legacy"
                    )
                    == "tma"
                )
                and (
                    sequence.startup is None
                    or sequence.startup.matches(cg, plan, sequence.inner_axes)
                )
            )
        else:
            inputs_valid = (
                supports_root_pair(cg, plan)
                and sequence.startup is None
                and sequence.startup_issued is False
                and sequence.inner_axes == _native_axes(cg, plan)
                and (
                    sequence.snapshot is None
                    or sequence.snapshot.matches(cg, plan, boundaries, sequence.scans)
                    and sequence.snapshot_ready is (stage == 1)
                    and sequence.snapshot_consumed is False
                )
            )
        if (
            plan is not sequence.plan
            or sequence.facts != chain._register_bridge_revision(plan)
            or not inputs_valid
            or stage != self.stage
            or stage != sequence.next_stage
            or not 0 <= stage < len(sequence.geometries)
            or geometry != sequence.geometries[stage]
            or geometry.physical != plan.shapes[stage]
            or geometry.transpose
            or plan.threads != 128
            or execution.threads != 128
            or execution.thread != "chain_thread"
            or execution.warp != "chain_warp"
            or execution.barriers != "chain_bars"
            or execution.tmem != "chain_tptr"
            or execution.sync != "cute.arch.sync_threads()"
            or self._local_input_protocol()
            and (
                execution.a_workspace != "chain_a_workspace"
                or execution.b_workspace != "chain_b_workspace"
            )
            or tmem_input != self.tmem_input
            or accumulator != self.accumulator
            or (
                type(pending_allocation) is not bool
                or pending_allocation != sequence.early_release
                if stage == 0
                else pending_allocation is not None
            )
            or type(terminal_fragment) is not bool
            or terminal_fragment != (stage == len(sequence.geometries) - 1)
        ):
            raise chain._UnsupportedChain(
                "root stage input provenance or readiness changed"
            )
        if self._local_input_protocol():
            if sequence.input_stage != (None if stage == 0 else 0):
                raise chain._UnsupportedChain("initialized input action repeated")
            if (
                sequence.input_execution is not None
                and sequence.input_context
                != _execution_fields(sequence.input_execution)
            ):
                raise chain._UnsupportedChain("initialized input context changed")
            if stage == 0 and sequence.input_completion is not None:
                raise chain._UnsupportedChain("initialized inputs already completed")
            sequence.input_execution = execution
            sequence.input_stage = stage
            sequence.input_context = _execution_fields(execution)
            sequence.input_producers = ()
            sequence.input_committed = False
            sequence.input_waited = False
            if sequence.pair_selection is not None:
                sequence.pair_boundaries = boundaries

    def operand_lines(
        self, cg: GenerateAST, boundaries: dict[Node, str], role: str
    ) -> list[str]:
        sequence = self.sequence
        if self._local_input_protocol():
            self._check_input_context()
            self._check_pair_arguments(cg, boundaries)
            if (
                role not in ("a", "b")
                or sequence.input_producers != (() if role == "a" else ("a",))
                or sequence.input_committed
                or sequence.input_waited
            ):
                raise chain._UnsupportedChain(
                    "initialized input producer order changed"
                )
        # The second A is packed after the copy wait. Its B was prefetched and
        # completed by stage zero; reissuing either producer changes the schedule.
        if (
            self.stage
            and not self.seeded_accumulator
            and not (sequence.pair_selection is not None and role == "b")
        ):
            if role == "b":
                plan = self.sequence.plan
                _, n, k = plan.shapes[self.stage]
                cached = chain._stage_input(
                    cg,
                    plan,
                    cast("Node", plan.dots[self.stage].args[1]),
                    f"chain_{self.stage}_b",
                    "b",
                    (n, k),
                )
                if cached is not None:
                    self.sequence.staged.append(cached)
            if self._local_input_protocol():
                sequence.input_producers += (role,)
            return []
        from ..compile_environment import CompileEnvironment
        from .chained_tcgen05 import _finish_startup_operand
        from .chained_tcgen05 import _layout
        from .chained_tcgen05 import _stage

        plan = sequence.plan
        m, n, k = plan.shapes[self.stage]
        shape = (m if role == "a" else n, k)
        inner = sequence.inner_axes[self.stage][0 if role == "a" else 1]
        dtype = CompileEnvironment.current().backend.dtype_str(
            plan.operand_dtype(self.stage)
        )
        prefix = f"chain_{self.stage}_{role}"
        if self.seeded_accumulator:
            lines: list[str] = []
            if role == "a" or (
                sequence.seeded_inputs is not None
                and sequence.seeded_inputs.mode == "local"
            ):
                producer = _stage(
                    cg,
                    plan,
                    boundaries,
                    list(sequence.scans),
                    self.stage,
                    role,
                    inner,
                    dtype,
                    sequence.pointwise_unroll,
                    sequence.pointwise_cache,
                    sequence.pointwise_inplace,
                    k_half="chain_k_half" if self.k_schedule is not None else None,
                    leaf_pipeline=sequence.initialized.paired.transfer
                    if sequence.initialized is not None
                    and sequence.initialized.paired is not None
                    else None,
                )
                if self.k_schedule is not None:
                    assert sequence.seeded_inputs is not None
                    operand = cast("Node", plan.dots[1].args[0])
                    selection = (sequence.seeded_inputs, operand, tuple(producer))
                    sequence.half_producer = RootHalfProducer(*selection, selection)
                lines = [
                    f"{prefix}_ptr = chain_{role}_workspace",
                    *_layout(prefix, shape, inner, dtype),
                    *([] if self.k_schedule is not None else producer),
                ]
            cached = chain._stage_input(
                cg,
                plan,
                cast("Node", plan.dots[self.stage].args[0 if role == "a" else 1]),
                prefix,
                role,
                shape,
            )
            if cached is not None:
                sequence.staged.append(cached)
            if self._local_input_protocol():
                sequence.input_producers += (role,)
            return lines
        transfer = (
            next((t for t in sequence.startup.transfers if t.role == role), None)
            if sequence.startup is not None
            else None
        )
        lines = (
            _finish_startup_operand(
                cg,
                plan,
                transfer,
                boundaries,
                list(sequence.scans if sequence.early_scan else sequence.early_cached),
                self.stage,
                inner,
                dtype,
                sequence.pointwise_unroll,
                sequence.pointwise_cache,
                sequence.pointwise_inplace,
            )
            if transfer is not None
            else [
                f"{prefix}_ptr = chain_{role}_workspace",
                *_layout(prefix, shape, inner, dtype),
                *_stage(
                    cg,
                    plan,
                    boundaries,
                    list(
                        sequence.scans if sequence.early_scan else sequence.early_cached
                    )
                    if sequence.independent is not None
                    or sequence.initialized is not None
                    else [],
                    self.stage,
                    role,
                    inner,
                    dtype,
                    sequence.pointwise_unroll,
                    sequence.pointwise_cache,
                    sequence.pointwise_inplace,
                ),
            ]
        )
        if sequence.independent is not None or (
            sequence.pair_selection is not None and self.stage == 1
        ):
            cached = chain._stage_input(
                cg,
                plan,
                cast("Node", plan.dots[self.stage].args[0 if role == "a" else 1]),
                prefix,
                role,
                shape,
            )
            if cached is not None:
                sequence.staged.append(cached)
        if self._local_input_protocol():
            sequence.input_producers += (role,)
            if sequence.pair_staging is not None:
                # A rejected group still uses the original single producers;
                # their original unroll activation is legitimate, not grouping.
                sequence.grouped_state = (
                    sequence.pair_staging.activated,
                    sequence.pair_staging.group_activated,
                    sequence.pointwise_unroll.activated,
                )
        return lines

    def after_commit(self) -> list[str]:
        if self._local_input_protocol():
            self._check_input_context()
            if (
                self.sequence.input_producers != ("a", "b")
                or self.sequence.input_committed
                or self.sequence.input_waited
            ):
                raise chain._UnsupportedChain("initialized input commit order changed")
            self.sequence.input_committed = True
        return (
            list(self.sequence.scan_lines)
            if self.stage == 0 and not self.sequence.early_scan
            else []
        ) + (
            [
                "cute.arch.mbarrier_wait(chain_start_bar, 0)",
                "cute.arch.sync_threads()",
            ]
            if self.stage == 0 and self.sequence.startup is not None
            else []
        )

    def after_wait(self, cg: GenerateAST, boundaries: dict[Node, str]) -> list[str]:
        if self._local_input_protocol():
            self._check_input_context()
            self._check_pair_arguments(cg, boundaries)
            sequence = self.sequence
            if not sequence.input_committed or sequence.input_waited:
                raise chain._UnsupportedChain("initialized input copy wait changed")
            if sequence.pair_selection is None:
                assert sequence.seeded_inputs is not None
                sequence.input_completion = RootInputCompletion(
                    sequence.seeded_inputs, self.stage
                )
                sequence.input_waited = True
            elif self.stage == 0:
                sequence.input_waited = True
        if self.stage == 0 or self.seeded_accumulator:
            return []
        from ..compile_environment import CompileEnvironment
        from .chained_tcgen05 import _bridge

        plan = self.sequence.plan
        snapshot = self.sequence.snapshot
        if self.sequence.pair_selection is not None and (
            self.sequence.pair_completed != 1 or self.sequence.pair_bridged
        ):
            raise chain._UnsupportedChain("root pair bridge completion changed")
        if snapshot is not None and (
            self.sequence.snapshot_ready is not True
            or self.sequence.snapshot_consumed is not False
            or not snapshot.matches(cg, plan, boundaries, self.sequence.scans)
        ):
            raise chain._UnsupportedChain("root snapshot read before completion")
        lines = _bridge(
            cg,
            plan,
            boundaries,
            list(self.sequence.scans),
            self.stage,
            CompileEnvironment.current().backend.dtype_str(
                plan.operand_dtype(self.stage)
            ),
            **({"snapshot": snapshot} if snapshot is not None else {}),
        )
        if snapshot is not None:
            self.sequence.snapshot_consumed = True
        if self.sequence.pair_selection is not None:
            self.sequence.input_waited = True
            self.sequence.pair_bridged = True
        return lines

    def retains_completed_result(
        self, cg: GenerateAST, boundaries: dict[Node, str]
    ) -> bool:
        """Only called after the shared issuer's actual completion wait."""
        snapshot = self.sequence.snapshot
        if snapshot is None or self.stage != 0:
            return False
        if (
            self.sequence.next_stage != 0
            or self.sequence.snapshot_ready is not False
            or self.sequence.snapshot_consumed is not False
            or not snapshot.matches(
                cg, self.sequence.plan, boundaries, self.sequence.scans
            )
        ):
            raise chain._UnsupportedChain("root snapshot completion changed")
        return True

    def complete(self) -> None:
        if self.stage != self.sequence.next_stage:
            raise chain._UnsupportedChain("root stage completed out of order")
        if self._completion is not None:
            raise chain._UnsupportedChain("root stage transition already completed")
        if self._local_input_protocol():
            self._check_input_context()
            if not self.sequence.input_waited or (
                self.sequence.pair_completed != self.stage
                or self.sequence.pair_bridged is not (self.stage == 1)
                if self.sequence.pair_selection is not None
                else self.sequence.seeded_inputs is None
                or self.sequence.input_completion
                != RootInputCompletion(self.sequence.seeded_inputs, self.stage)
            ):
                raise chain._UnsupportedChain("initialized input completion missing")
            if self.sequence.pair_selection is not None:
                self.sequence.pair_completed += 1
        if self.sequence.snapshot is not None:
            if self.stage == 0:
                if self.sequence.snapshot_ready or self.sequence.snapshot_consumed:
                    raise chain._UnsupportedChain("root snapshot already published")
                self.sequence.snapshot_ready = True
            elif (
                not self.sequence.snapshot_ready or not self.sequence.snapshot_consumed
            ):
                raise chain._UnsupportedChain("root snapshot was not consumed")
        if self.stage == 0 and self.sequence.initialized is not None:
            if self.sequence.paired_issued is not (
                self.sequence.initialized.paired is not None
            ):
                raise chain._UnsupportedChain("paired initial publication missing")
            self.sequence.seed_completion = RootSeedCompletion(
                self.sequence.initialized
            )
        self.sequence.next_stage += 1
        object.__setattr__(
            self,
            "_completion",
            RootStageCompletion(
                self.sequence, self.stage, root_stage_progress(self.sequence)
            ),
        )


def plan_root_stage_sequence(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    bridges: dict[int, chain._RegisterBridge],
    inner_axes: dict[tuple[int, str], int],
    scans: list[chain._ScanInput],
    scan_lines: list[str],
    *,
    prefetched: bool,
    pointwise_unroll: PointwiseUnroll,
    pointwise_cache: PointwiseReadCache,
    pointwise_inplace: PointwiseInplace,
    independent: IndependentRootInputs | None = None,
    pair_inputs: RootPairInputs | None = None,
    pair_staging: VectorStaging | None = None,
    boundaries: dict[Node, str] | None = None,
    early_scan: bool = False,
    early_cached: tuple[chain._ScanInput, ...] = (),
    startup: RootStartupInputs | None = None,
    startup_issued: bool = False,
    initialized: RootInitializedResult | None = None,
    seeded_inputs: RootSeededInputs | None = None,
    snapshot_tile_columns: int = 0,
) -> RootStageSequence | None:
    """Consume the original late coordinate/exclusive-use proof in place.

    Rejection keeps the same already-prepared root scheduler state; it does not
    restart codegen and consume fresh temporary names. The bridge dictionary
    must be the result of the original `_register_bridges` call for this plan.
    """
    from .chained_tcgen_stage import StageGeometry

    if type(snapshot_tile_columns) is not int or snapshot_tile_columns not in (0, 32):
        raise chain._UnsupportedChain("root snapshot tile columns must be 0 or 32")
    if snapshot_tile_columns and (initialized is not None or independent is not None):
        raise chain._UnsupportedChain(
            "root snapshot requires an exclusive packed bridge"
        )
    if pair_inputs is not None:
        axes = tuple((inner_axes[i, "a"], inner_axes[i, "b"]) for i in range(2))
        if (
            boundaries is None
            or not pair_inputs.matches(cg, plan, boundaries, axes)
            or set(bridges) != {1}
            or bridges[1] is not pair_inputs.bridge
            or prefetched
            or independent is not None
            or initialized is not None
            or seeded_inputs is not None
            or startup is not None
            or startup_issued is not False
            or snapshot_tile_columns
            or scans
            or scan_lines
            or early_scan
            or early_cached
            or (pair_staging is not None)
            != (
                cg.device_function.config.config.get("cute_chained_vector_group", False)
                is True
            )
        ):
            raise chain._UnsupportedChain("accepted local root pair inputs changed")
        return RootStageSequence(
            plan,
            chain._register_bridge_revision(plan),
            tuple(StageGeometry(shape, False) for shape in plan.shapes),
            axes,
            (),
            (),
            pointwise_unroll,
            pointwise_cache,
            pointwise_inplace,
            cast(
                "bool",
                cg.device_function.config.config.get(
                    "cute_chained_tmem_early_release", False
                ),
            ),
            pair_inputs=pair_inputs,
            pair_selection=pair_inputs,
            pair_staging=pair_staging,
            pair_staging_selection=(
                pair_staging,
                pointwise_unroll,
                _startup_value(
                    (
                        pair_staging.enabled,
                        pair_staging.group_enabled,
                        pair_staging.broadcast,
                    )
                ),
                pointwise_unroll.factor,
            )
            if pair_staging is not None
            else (),
            grouped_state=(
                pair_staging.activated,
                pair_staging.group_activated,
                pointwise_unroll.activated,
            )
            if pair_staging is not None
            else (False, False, False),
        )
    if pair_staging is not None:
        raise chain._UnsupportedChain(
            "grouped root producer requires accepted pair inputs"
        )
    if initialized is not None:
        axes = tuple((inner_axes[i, "a"], inner_axes[i, "b"]) for i in range(2))
        if (
            boundaries is None
            or not initialized.matches(cg, plan, boundaries, axes, tuple(scans))
            or bridges
            or independent is not None
            or startup is not None
            or startup_issued is not False
            or seeded_inputs is not None
            and (
                prefetched != (seeded_inputs.mode != "local")
                or seeded_inputs.result is not initialized
                or not seeded_inputs.matches(plan)
            )
        ):
            return None
        return RootStageSequence(
            plan,
            initialized.revision,
            tuple(StageGeometry(shape, False) for shape in plan.shapes),
            axes,
            tuple(scans),
            tuple(scan_lines),
            pointwise_unroll,
            pointwise_cache,
            pointwise_inplace,
            cast(
                "bool",
                cg.device_function.config.config.get(
                    "cute_chained_tmem_early_release", False
                ),
            ),
            early_scan=early_scan,
            early_cached=early_cached,
            input_readiness=(tuple(scans), tuple(scan_lines), early_scan, early_cached),
            initialized=initialized,
            seeded_inputs=seeded_inputs,
        )

    if independent is not None:
        axes = ((inner_axes[0, "a"], inner_axes[0, "b"]),)
        if (
            boundaries is None
            or not independent.matches(cg, plan, boundaries, axes)
            or bridges
            or prefetched
            or type(startup_issued) is not bool
            or startup_issued != (startup is not None)
            or (startup is not None)
            != (
                cg.device_function.config.config.get(
                    "cute_chained_startup_transfer", "legacy"
                )
                == "tma"
            )
            or (startup is not None and not startup.matches(cg, plan, axes))
            or type(early_scan) is not bool
            or early_scan
            != any(
                chain._ancestors(cast("Node", operand)) & set(plan.scans)
                for operand in plan.dots[0].args[:2]
            )
        ):
            return None
        return RootStageSequence(
            plan,
            independent.revision,
            (StageGeometry(plan.shapes[0], False, native_rows=64),),
            axes,
            tuple(scans),
            tuple(scan_lines),
            pointwise_unroll,
            pointwise_cache,
            pointwise_inplace,
            cast(
                "bool",
                cg.device_function.config.config.get(
                    "cute_chained_tmem_early_release", False
                ),
            ),
            independent=independent,
            early_scan=early_scan,
            early_cached=early_cached,
            input_readiness=(tuple(scans), tuple(scan_lines), early_scan, early_cached),
            startup=startup,
            startup_issued=startup_issued,
        )
    if (
        not supports_root_pair(cg, plan)
        or startup is not None
        or startup_issued is not False
        or not prefetched
        or set(bridges) != {1}
        or bridges[1].role != "a"
        or bridges[1].stage != 1
        or bridges[1].source is not plan.dots[0]
        or bridges[1].operand is not plan.dots[1].args[0]
        or bridges[1].revision != chain._register_bridge_revision(plan)
        or inner_axes[0, "a"] != 1
        or inner_axes[0, "b"] != 1
        or plan.shapes[0][:2] != (plan.shapes[1][0], plan.shapes[1][2])
        or plan.dots[0].meta["val"].dtype != torch.float32
    ):
        return None
    try:
        axes = _native_axes(cg, plan)
    except chain._UnsupportedChain:
        return None
    if axes != ((inner_axes[0, "a"], inner_axes[0, "b"]), (1, inner_axes[1, "b"])):
        return None
    snapshot = None
    if snapshot_tile_columns:
        from .chained_root_snapshot import plan_root_snapshot

        if boundaries is None:
            raise chain._UnsupportedChain(
                "root snapshot is missing original boundaries"
            )
        snapshot = plan_root_snapshot(cg, plan, bridges[1], boundaries, tuple(scans))
    return RootStageSequence(
        plan,
        chain._register_bridge_revision(plan),
        tuple(StageGeometry(shape, False) for shape in plan.shapes),
        axes,
        tuple(scans),
        tuple(scan_lines),
        pointwise_unroll,
        pointwise_cache,
        pointwise_inplace,
        cast(
            "bool",
            cg.device_function.config.config.get(
                "cute_chained_tmem_early_release", False
            ),
        ),
        snapshot=snapshot,
    )
