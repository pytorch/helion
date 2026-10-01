"""Original root warp operand preparation, independent of MMA execution."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import math
from typing import TYPE_CHECKING
from typing import cast

from . import chained_matmul as chain
from . import chained_warp_stage as shared
from .chained_execution import ChainedExecution
from .chained_warp_stage import _snapshot
from .chained_warp_stage import warp_revision

if TYPE_CHECKING:
    import torch
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_body_program import RootWarpBody
    from .chained_matmul import ChainedMatmulPlan
    from .chained_scratch_layout import ScratchLayouts
    from .chained_warp_bridge import WarpBridgeInput


@dataclass
class _WarpBodyBinding:
    """The selected owner survives copies; body admission is one-way."""

    owner: object = None
    body: RootWarpBody | None = None
    accepted: bool = False

    def check(self, owner: object) -> RootWarpBody | None:
        if self.owner is not owner or (self.accepted and self.body is None):
            raise chain._UnsupportedChain("root warp body owner changed")
        return self.body


@dataclass(frozen=True)
class RootWarpStageAction:
    """Singleton zero-seeded root, before any original operand is evaluated."""

    codegen: GenerateAST
    plan: ChainedMatmulPlan
    boundaries: tuple[tuple[Node, str], ...]
    schedule: str
    padding: int
    dtype: str
    revision: object
    config: object
    aliases: tuple[tuple[str, str], ...]
    _selection: object = field(repr=False)
    _body: _WarpBodyBinding = field(default_factory=_WarpBodyBinding, repr=False)
    _consumed: bool = field(default=False, init=False, repr=False)
    _completion: shared.WarpInputCompletion | None = field(
        default=None, init=False, repr=False
    )
    _staged: object = field(default=None, init=False, repr=False)

    def facts(self) -> object:
        return _snapshot(
            (
                self.codegen,
                self.plan,
                self.boundaries,
                self.schedule,
                self.padding,
                self.dtype,
                self.revision,
                self.config,
                self.aliases,
                id(self._body),
            )
        )

    def matches(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        *,
        prepared: bool = False,
    ) -> bool:
        return (
            self._consumed is False
            and self._completion is None
            and self._selection == self.facts()
            and cg is self.codegen
            and plan is self.plan
            and tuple(boundaries.items()) == self.boundaries
            and warp_revision(plan, aliases=False) == self.revision
            and (
                tuple(plan.tensor_aliases.items())[: len(self.aliases)] == self.aliases
                if prepared
                else tuple(plan.tensor_aliases.items()) == self.aliases
            )
            and _snapshot(cg.device_function.config.config) == self.config
        )

    def emit(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        staged_inputs: list[chain._StagedInput],
        scratch: ScratchLayouts,
        prefix: list[str],
    ) -> list[str]:
        """Original builder once, actual copy completion, then shared executor."""
        if not self.matches(cg, plan, boundaries):
            raise chain._UnsupportedChain("root warp action changed before preparation")
        body = self._body.check(self)
        if body is not None:
            body.begin(self, cg, plan, boundaries, staged_inputs, scratch, prefix, 0)
        operands = emit_original_warp_operands(
            cg,
            plan,
            boundaries,
            [],
            {},
            staged_inputs,
            0,
            self.schedule,
            self.padding,
            self.dtype,
        )
        if not self.matches(cg, plan, boundaries, prepared=True):
            raise chain._UnsupportedChain("root warp action changed during preparation")
        rows, columns, reduction = plan.shapes[0]
        expected_shapes = ((rows, reduction), (columns, reduction))
        if (
            not operands.matches()
            or tuple(role for role, _ in operands.axes) != ("a", "b")
            or len(operands.layouts) != 2
            or operands.asynchronous != bool(operands.copies)
        ):
            raise chain._UnsupportedChain("root warp operand selection changed")
        for index, (layout, shape) in enumerate(
            zip(operands.layouts, expected_shapes, strict=True)
        ):
            stride = (
                (shape[1] + self.padding, 1)
                if layout.inner == 1
                else (1, shape[0] + self.padding)
            )
            if (
                layout.node is not plan.dots[0].args[index]
                or layout.role != ("a", "b")[index]
                or layout.shape != shape
                or layout.inner not in (0, 1)
                or layout.stride != stride
                or operands.axes[index] != (layout.role, layout.inner)
                or layout.dtype != plan.operand_dtype(0)
            ):
                raise chain._UnsupportedChain("root warp original owner changed")
        execution = ChainedExecution(plan.threads)
        lines = list(operands.lines)
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
        output_stride = columns + self.padding // 2
        result = shared.WarpRootResult(
            0,
            rows * output_stride,
            scratch.layout("chain_0_c", (rows, columns), row_stride=output_stride),
        )
        prepared = shared.prepare_warp_stage(
            completion,
            "chain_0",
            self.dtype,
            (rows, columns, reduction),
            32 * min(plan.threads // 32, 2 ** (columns.bit_length() - 4)),
            (operands.axes[0][1], operands.axes[1][1]),
            result,
        )
        lines.extend(
            shared.emit_prepared_warp_stage(
                cg,
                plan,
                boundaries,
                prepared,
                [*prefix, *lines],
            )
        )
        object.__setattr__(self, "_consumed", True)
        object.__setattr__(self, "_completion", completion)
        from .chained_warp_bridge import _staged_facts

        object.__setattr__(self, "_staged", _staged_facts(staged_inputs))
        return lines


def root_warp_stage_action(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    schedule: str,
    padding: int,
    dtype: str,
    *,
    general_collectives: bool,
) -> RootWarpStageAction | None:
    """No expression probing or new schedule admission; other policies stay old."""
    if (
        plan.strategy == "tcgen05_tmem"
        or plan.loop is not None
        or len(plan.dots) != 1
        or plan.dots[0].args[2] is not None
        or plan.scans
        or general_collectives
        or plan.pointwise_cache is not None
        or schedule not in ("cp_async", "cp_async_register_reuse")
    ):
        return None
    action = RootWarpStageAction(
        cg,
        plan,
        tuple(boundaries.items()),
        schedule,
        padding,
        dtype,
        warp_revision(plan, aliases=False),
        _snapshot(cg.device_function.config.config),
        tuple(plan.tensor_aliases.items()),
        None,
    )
    selected = RootWarpStageAction(
        cg,
        plan,
        action.boundaries,
        schedule,
        padding,
        dtype,
        action.revision,
        action.config,
        action.aliases,
        action.facts(),
        action._body,
    )
    selected._body.owner = selected
    return selected


@dataclass(frozen=True)
class WarpOperandLayout:
    node: Node
    role: str
    shape: tuple[int, int]
    stride: tuple[int, int]
    inner: int
    dtype: torch.dtype


@dataclass(frozen=True)
class WarpOperandEmission:
    lines: tuple[str, ...]
    axes: tuple[tuple[str, int], ...]
    asynchronous: bool
    copies: tuple[tuple[str, ...], ...]
    layouts: tuple[WarpOperandLayout, ...]
    _selection: object = field(repr=False)

    def facts(self) -> object:
        return _snapshot(
            (
                self.lines,
                self.axes,
                self.asynchronous,
                self.copies,
                tuple(
                    (
                        item.node,
                        item.role,
                        item.shape,
                        item.stride,
                        item.inner,
                        item.dtype,
                    )
                    for item in self.layouts
                ),
            )
        )

    def matches(self) -> bool:
        return self._selection == self.facts()


def emit_original_warp_operands(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scan_inputs: list[chain._ScanInput],
    bridges: dict[int, chain._RegisterBridge],
    staged_inputs: list[chain._StagedInput],
    stage: int,
    schedule: str,
    padding: int,
    dtype: str,
    *,
    probe_boundaries: dict[Node, str] | None = None,
    bridge_input: WarpBridgeInput | None = None,
) -> WarpOperandEmission:
    """The old root builder, called once by migrated and unported root stages."""
    node = plan.dots[stage]
    rows, columns, reduction = plan.shapes[stage]
    prefix = f"chain_{stage}"
    lines: list[str] = []
    inner_axes: dict[str, int] = {}
    async_stage = False
    copy_programs: list[tuple[str, ...]] = []
    layouts: list[WarpOperandLayout] = []
    probes = boundaries if probe_boundaries is None else probe_boundaries
    for role, shape in (("a", (rows, reduction)), ("b", (columns, reduction))):
        operand = cast("Node", node.args[0 if role == "a" else 1])
        inner = chain._operand_inner_axis(cg, plan, probes, operand)
        if role == "b":
            inner = 1 - inner
        if schedule in ("k_major", "k_major_padded"):
            inner = 1
        inner_axes[role] = inner
        if inner == 1:
            x, y = f"{prefix}_load // {shape[1]}", f"{prefix}_load % {shape[1]}"
            stride = (shape[1] + padding, 1)
        else:
            x, y = f"{prefix}_load % {shape[0]}", f"{prefix}_load // {shape[0]}"
            stride = (1, shape[0] + padding)
        layouts.append(
            WarpOperandLayout(
                operand, role, shape, stride, inner, plan.operand_dtype(stage)
            )
        )
        coords = (x, y) if role == "a" else (y, x)
        expression = chain._Expression(cg, plan, probes)
        expression.scan_inputs = scan_inputs
        expression.coordinate_names.add(f"{prefix}_load")
        value = expression.value(operand, coords)
        domain = chain._operand_domain(cg, operand, coords, plan)
        vector = 1 if schedule == "k_major" else min(8, shape[inner])
        steps = (math.prod(shape) + plan.threads * vector - 1) // (
            plan.threads * vector
        )
        load_range = (
            f"cutlass.range({steps}, unroll=1)"
            if schedule
            in (
                "coalesced",
                "cp_async",
                "cp_async_register",
                "cp_async_register_reuse",
                "cp_async_register_reuse_scan",
            )
            else f"cutlass.range_constexpr({steps})"
        )
        lines.extend(
            [
                (
                    f"{prefix}_{role}_ptr = chain_{role}_workspace"
                    if plan.operand_dtype(stage) == plan.dtype
                    else f"{prefix}_{role}_ptr = cute.recast_ptr(chain_{role}_workspace, dtype={dtype})"
                ),
                f"{prefix}_{role} = cute.make_tensor({prefix}_{role}_ptr, cute.make_layout({shape!r}, stride={stride!r}))",
            ]
        )
        if (
            schedule in ("cp_async_register_reuse", "cp_async_register_reuse_scan")
            and stage == len(plan.dots) - 1
        ):
            staged = chain._stage_input(
                cg, plan, operand, f"{prefix}_{role}", role, shape
            )
            if staged is not None:
                staged_inputs.append(staged)
        bridge = bridges.get(stage)
        if bridge is not None and bridge.role == role:
            if bridge_input is not None:
                bridge_input.bind_layout(cg, plan, boundaries, layouts[-1], padding)
            lines.extend(
                bridge.render()
                if bridge_input is None
                else bridge_input.emit(
                    cg, plan, boundaries, probes, stage, layouts[-1], bridge
                )
            )
            continue
        fallback = [
            f"for {prefix}_load_step in {load_range}:",
            f"    for {prefix}_load_vec in cutlass.range_constexpr({vector}):",
            f"        {prefix}_load = chain_thread * {vector} + {prefix}_load_step * {plan.threads * vector} + {prefix}_load_vec",
            f"        if {prefix}_load < {math.prod(shape)}:",
            chain._indent(expression.lines, 12),
            f"            {prefix}_{role}[{x}, {y}] = {chain._masked_operand(value, dtype, domain)}",
        ]
        async_lines = (
            chain._async_copy(
                cg,
                plan,
                operand,
                prefix,
                role,
                shape,
                stride,
                inner,
                dtype,
                fallback,
            )
            if schedule
            in (
                "cp_async",
                "cp_async_register",
                "cp_async_register_reuse",
                "cp_async_register_reuse_scan",
            )
            else None
        )
        async_stage |= async_lines is not None
        if async_lines is not None:
            copy_programs.append(tuple(async_lines))
        lines.extend(fallback if async_lines is None else async_lines)
    emission = WarpOperandEmission(
        tuple(lines),
        tuple(inner_axes.items()),
        async_stage,
        tuple(copy_programs),
        tuple(layouts),
        None,
    )
    return WarpOperandEmission(
        emission.lines,
        emission.axes,
        emission.asynchronous,
        emission.copies,
        emission.layouts,
        emission.facts(),
    )
