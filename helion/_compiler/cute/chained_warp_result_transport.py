"""Explicit live C storage and original terminal stores for warp roots.

Packing owns no arithmetic. A terminal proposal owns no register image until
the original stage has completed with the same plan, coordinates and fragment.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language import _tracing_ops
from ...language import memory_ops
from . import chained_matmul as chain
from .chained_register_islands import _freeze
from .chained_root_exports import _classify_store
from .chained_store_expression import _node_facts
from .chained_store_expression import lower_store_point
from .chained_store_expression import materialized_store_read
from .chained_workspace import plan_contraction_workspace
from .physical_use_frontier import PhysicalUseFrontier

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_matmul import _ScanInput
    from .chained_matmul import _StagedInput
    from .chained_root_exports import RootExport
    from .chained_warp_register_transport import WarpRegisterPlan
    from .warp_specialized_plan import SharedMemoryLayoutPlan


KEY = "cute_chained_warp_result_transport"
MODES = ("legacy", "packed", "packed_direct", "direct", "packed_register")
SCHEDULES = (
    "k_major",
    "k_major_padded",
    "coalesced",
    "coalesced_unrolled",
    "cp_async",
)


def _plan_facts(plan: ChainedMatmulPlan) -> tuple[object, ...]:
    assert plan.region is not None
    return (
        plan.root_graph_id,
        plan.dots,
        plan.store,
        plan.axes,
        plan.shapes,
        plan.threads,
        plan.strategy,
        plan.loop,
        plan.contraction_groups,
        plan.pointwise_cache,
        plan.scan_exports,
        plan.root_exports,
        _node_facts(tuple(plan.dots[0].graph.nodes)),
    )


def complete_warp_tiles(
    plan: ChainedMatmulPlan, *, terminal_only: bool = False
) -> bool:
    """Full logical C coverage under the original (1, W, 1) warp layout."""
    return plan.threads in (32, 64, 128, 256) and all(
        type(m) is int
        and type(n) is int
        and type(k) is int
        and min(m, n, k) > 0
        and m % 16 == 0
        and n % (8 * (plan.threads // 32)) == 0
        and k % 16 == 0
        and 2 ** (n.bit_length() - 4) >= plan.threads // 32
        for m, n, k in (plan.shapes[-1:] if terminal_only else plan.shapes)
    )


@dataclass(frozen=True)
class WarpTerminalStore:
    source: Node
    output: RootExport
    facts: tuple[object, ...]


def classify_terminal_store(plan: ChainedMatmulPlan) -> WarpTerminalStore | None:
    """Complete fresh ownership and a single same-coordinate pointwise sink."""
    if plan.region is None or not plan.dots:
        return None
    source = plan.dots[-1]
    store = plan.store
    axes = {axis: (extent, block) for axis, extent, block in plan.axes}
    output = _classify_store(
        store, set(axes), resolved_shape=chain._shape(cast("Node", store.args[2]))
    )
    if output is None or output.shape != plan.shapes[-1][:2]:
        return None
    target = cast("Node", store.args[0])
    if any(
        axis is not None and (axes[axis][1] != 1 or size != axes[axis][0])
        for size, axis in zip(target.meta["val"].shape, output.axes, strict=True)
    ):
        return None
    target_storage = target.meta["val"].untyped_storage()
    for node in plan.region.nodes:
        if node is store or node.target not in (memory_ops.load, memory_ops.store):
            continue
        tensor = cast("Node", node.args[0]).meta.get("val")
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.untyped_storage() is target_storage
        ):
            return None
    positions = {node: index for index, node in enumerate(plan.region.nodes)}
    if any(
        positions[node] > positions[source]
        for node in (*plan.region.scans, *plan.region.reductions)
    ):
        return None
    value = cast("Node", store.args[2])
    frontier = PhysicalUseFrontier(tuple(source.graph.nodes))
    if not frontier.exclusive_use(source, value, store):
        return None
    pending, seen = list(source.users), {source}
    while pending:
        node = pending.pop()
        if node is store or node in seen:
            continue
        seen.add(node)
        if chain._shape(node) != output.shape or (
            chain._pointwise_inputs(node) is None
            and node.target
            not in (
                _tracing_ops._new_var,
                _tracing_ops._mask_to,
                torch.ops.aten.where.self,
            )
        ):
            return None
        pending.extend(node.users)
    frontier.check()
    return WarpTerminalStore(source, output, _plan_facts(plan))


@dataclass(frozen=True)
class WarpResultTransport:
    mode: str
    workspace: SharedMemoryLayoutPlan | None
    terminal: WarpTerminalStore | None
    facts: tuple[object, ...]
    registers: WarpRegisterPlan | None = None

    def check(self, plan: ChainedMatmulPlan) -> None:
        if (
            plan.warp_result_transport is not self
            or plan.warp_workspace != self.workspace
            or self.mode not in MODES[1:]
            or self.facts != _plan_facts(plan)
            or (self.mode in ("packed_direct", "direct")) != (self.terminal is not None)
            or (self.mode == "direct") != (self.workspace is None)
            or (self.mode == "packed_register") != (self.registers is not None)
            or self.terminal is not None
            and self.terminal != classify_terminal_store(plan)
        ):
            raise chain._UnsupportedChain("warp result transport proof changed")
        if self.registers is not None:
            self.registers.check(plan)
            if self.workspace != self.registers.workspace:
                raise chain._UnsupportedChain("warp register workspace changed")


def plan_warp_result_transport(
    plan: ChainedMatmulPlan, mode: str, schedule: str
) -> WarpResultTransport | None:
    if (
        mode not in MODES[1:]
        or schedule not in SCHEDULES
        or plan.strategy != "warp"
        or plan.loop is not None
        or len(plan.dots) < 2
        or plan.contraction_groups is not None
        or plan.k_schedule is not None
        or plan.preparation_pipeline is not None
        or plan.root_typed_operands is not None
        or plan.initialized_accumulator is not None
        or plan.late_rhs_reuse is not None
        or plan.direct_output
        or not complete_warp_tiles(plan, terminal_only=mode == "direct")
    ):
        return None
    direct = mode in ("packed_direct", "direct")
    terminal = classify_terminal_store(plan) if direct else None
    if direct and terminal is None:
        return None
    registers = None
    if mode == "packed_register":
        from .chained_warp_register_transport import plan_warp_register_transport

        if schedule != "cp_async":
            return None
        registers = plan_warp_register_transport(plan)
        if registers is None:
            return None
    workspace = (
        None
        if mode == "direct"
        else plan_contraction_workspace(
            plan,
            row_padding=0 if schedule == "k_major" else 4,
            include_final=terminal is None,
        )
    )
    if registers is not None:
        workspace = registers.workspace
    return WarpResultTransport(mode, workspace, terminal, _plan_facts(plan), registers)


@dataclass
class TerminalStoreCompletion:
    """One exact original inline warp stage, consumed once by its store."""

    cg: GenerateAST
    plan: ChainedMatmulPlan
    transport: WarpResultTransport
    prefix: str
    stage_lines: tuple[str, ...]
    emitted: tuple[str, ...] | None = field(default=None, init=False)
    _selection: tuple[object, ...] = field(init=False, repr=False)
    _publication: tuple[str, ...] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._selection = self._fields()

    def _fields(self) -> tuple[object, ...]:
        return (
            id(self.cg),
            id(self.plan),
            id(self.transport),
            self.prefix,
            self.stage_lines,
            _freeze(self.cg.device_function.config.config),
        )

    def _check(self) -> None:
        self.transport.check(self.plan)
        if self._selection != self._fields() or self.emitted != self._publication:
            raise chain._UnsupportedChain("completed warp terminal selection changed")

    def emit(
        self,
        boundaries: dict[Node, str],
        staged_inputs: list[_StagedInput],
        scan_inputs: list[_ScanInput],
    ) -> tuple[str, ...]:
        self._check()
        terminal = self.transport.terminal
        if (
            terminal is None
            or self.emitted is not None
            or not self.stage_lines
            or self.prefix != f"chain_{len(self.plan.dots) - 1}"
            or terminal.source in boundaries
            or self.cg.device_function.config.config.get(KEY) != self.transport.mode
            or self.cg.device_function.config.num_warps * 32 != self.plan.threads
        ):
            raise chain._UnsupportedChain("invalid completed warp terminal store")
        prefix = f"{self.prefix}_terminal"
        coordinates = (f"{prefix}_row", f"{prefix}_col")
        logical = {**boundaries, terminal.source: f"{self.prefix}_c"}
        read = materialized_store_read(
            self.plan,
            logical,
            terminal.source,
            coordinates,
            f"{self.prefix}_acc[{prefix}_index]",
        )
        point = lower_store_point(
            self.cg,
            self.plan,
            terminal.output.store,
            logical,
            staged_inputs,
            scan_inputs,
            coordinates=coordinates,
            coordinate_names=coordinates,
            completed_read=read,
        )
        if any(
            node is terminal.source and coords != coordinates
            for node, coords in point.reads
        ):
            raise chain._UnsupportedChain("cross-coordinate warp terminal store")
        shape = terminal.output.shape
        lines = (
            f"{prefix}_coords = {self.prefix}_thr.partition_C(cute.make_identity_tensor({shape!r}))",
            f"for {prefix}_index in cutlass.range_constexpr(cute.size({self.prefix}_acc)):",
            f"    {coordinates[0]}, {coordinates[1]} = {prefix}_coords[{prefix}_index]",
            chain._indent(point.lines),
            f"    if {' and '.join(point.bounds)}:",
            f"        {point.target}[{', '.join(point.indices)}] = {point.value}",
        )
        self.emitted = self._publication = lines
        return lines

    def validate(self, lines: list[str]) -> None:
        self._check()
        if self.emitted is None:
            raise chain._UnsupportedChain("missing warp terminal publication")
        body = "\n".join(lines)
        completed = "\n".join(
            (
                *self.stage_lines,
                "cute.arch.sync_threads()",
                *self.emitted,
                "cute.arch.sync_threads()",
            )
        )
        if body.count(completed) != 1 or body.count("\n".join(self.emitted)) != 1:
            raise chain._UnsupportedChain("completed warp terminal body changed")
