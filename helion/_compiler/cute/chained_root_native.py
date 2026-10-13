"""Bound native instruction choices inside the original joined root schedule.

This capability owns neither a new DAG nor an independent root. Selected
contractions keep their full original FP32 shared publication; all other
contractions and collective events retain their existing order and ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from .chained_execution import ChainedExecution

if TYPE_CHECKING:
    from ..device_ir import GraphInfo
    from ..device_ir import RootGraphInfo
    from .chained_matmul import ChainedMatmulPlan
    from .chained_tcgen_stage import StageGeometry


KEY = "cute_chained_root_native"
STRATEGY = "mixed_root"


def _raw_leaf(node: Node) -> Node | None:
    from . import chained_matmul as chain

    while node.target in chain._VIEWS:
        if not node.args or not isinstance(node.args[0], Node):
            return None
        node = node.args[0]
    if (
        node.target is not chain.memory_ops.load
        or len(node.args) < 2
        or not isinstance(node.args[0], Node)
        or node.args[0].target is not chain._tracing_ops._host_tensor
    ):
        return None
    return node


def _scope(plan: ChainedMatmulPlan) -> tuple[object, ...]:
    from .chained_root_vector_cache import _tensor_facts
    from .chained_store_expression import _node_facts

    region = plan.region
    if region is None:
        return ()
    nodes = tuple(region.graph.nodes)
    return (
        plan.root_graph_id,
        id(region),
        id(region.graph),
        region.graph_id,
        region.nodes,
        region.stores,
        region.live_outs,
        plan.dots,
        plan.store,
        plan.axes,
        plan.shapes,
        plan.dtype,
        plan.threads,
        plan.scans,
        plan.scan_exports,
        plan.root_exports,
        _node_facts(nodes),
        _tensor_facts(nodes),
    )


def _ordinary_storage(plan: ChainedMatmulPlan) -> bool:
    return (
        plan.region is not None
        and plan.loop is None
        and plan.threads == 128
        and plan.initialized_accumulator is None
        and plan.late_rhs_reuse is None
        and plan.k_schedule is None
        and plan.contraction_groups is None
        and not plan.warp_mma_stages
        and plan.preparation_pipeline is None
        and plan.pointwise_cache is None
        and plan.loop_workspace is None
        and plan.warp_workspace is None
        and plan.warp_result_transport is None
        and plan.root_workspace is None
        and plan.root_typed_operands is None
        and not plan.direct_output
    )


@dataclass(frozen=True)
class RootNativeStage:
    index: int
    node: Node
    geometry: StageGeometry
    operands: tuple[Node, Node]
    leaves: tuple[Node, Node]
    dtype: torch.dtype


def _stages(plan: ChainedMatmulPlan) -> tuple[RootNativeStage, ...]:
    from .chained_tcgen_stage import stage_geometry

    result = []
    for index, (node, shape) in enumerate(zip(plan.dots, plan.shapes, strict=True)):
        if shape[0] not in (64, 128) or node.args[2] is not None:
            continue
        geometry = stage_geometry(shape, native_rows=shape[0])
        if geometry is None or geometry.transpose or geometry.physical != shape:
            continue
        operands = cast("tuple[Node, Node]", node.args[:2])
        leaves = tuple(_raw_leaf(operand) for operand in operands)
        if any(leaf is None for leaf in leaves):
            continue
        dtype = plan.operand_dtype(index)
        if dtype not in (torch.float16, torch.bfloat16):
            continue
        result.append(
            RootNativeStage(
                index,
                node,
                geometry,
                operands,
                cast("tuple[Node, Node]", leaves),
                dtype,
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class RootNativeSchedule:
    root: RootGraphInfo
    phase: int
    stages: tuple[RootNativeStage, ...]
    execution: ChainedExecution
    scope: tuple[object, ...]
    aliases: dict[str, str]
    alias_prefix: tuple[tuple[str, str], ...]
    original_workspace_elements: tuple[int, int]
    tmem_columns: int
    barrier_count: int

    def check(self, plan: ChainedMatmulPlan) -> None:
        from .chained_matmul import _UnsupportedChain

        if (
            type(self.phase) is not int
            or type(self.root.phase_index) is not int
            or any(type(item.index) is not int for item in self.stages)
            or plan.root_native is not self
            or plan.strategy != STRATEGY
            or not _ordinary_storage(plan)
            or plan.region is None
            or self.root.graph is not plan.region.graph
            or self.root.graph_id != plan.root_graph_id
            or self.root.phase_index != self.phase
            or self.execution != ChainedExecution(plan.threads)
            or _scope(plan) != self.scope
            or _stages(plan) != self.stages
            or not self.stages
            or plan.tensor_aliases is not self.aliases
            or any(self.aliases.get(name) != value for name, value in self.alias_prefix)
            or self.barrier_count != len(plan.dots)
            or self.tmem_columns
            != max(item.geometry.physical[1] for item in self.stages)
            or self.original_workspace_elements
            != (
                max(m * k + 8 * max(m, k) for m, _, k in plan.shapes),
                max(n * k + 8 * max(n, k) for _, n, k in plan.shapes),
            )
            or any(
                m * k > self.original_workspace_elements[0]
                or n * k > self.original_workspace_elements[1]
                for m, n, k in (item.geometry.physical for item in self.stages)
            )
        ):
            raise _UnsupportedChain("joined root native schedule changed")

    def selected(self, plan: ChainedMatmulPlan, stage: int) -> RootNativeStage | None:
        self.check(plan)
        return next((item for item in self.stages if item.index == stage), None)

    @property
    def shared_metadata_bytes(self) -> int:
        return (8 * self.barrier_count + 127) // 128 * 128 + 128


def plan_root_native(
    plan: ChainedMatmulPlan, root: GraphInfo
) -> RootNativeSchedule | None:
    """Raw-source discovery only; concrete copy and completion proofs follow."""
    from ..device_ir import RootGraphInfo

    if (
        not isinstance(root, RootGraphInfo)
        or plan.region is None
        or root.graph is not plan.region.graph
        or root.graph_id != plan.root_graph_id
        or plan.strategy != "warp"
        or plan.root_native is not None
        or not _ordinary_storage(plan)
    ):
        return None
    stages = _stages(plan)
    if not stages:
        return None
    workspace = (
        max(m * k + 8 * max(m, k) for m, _, k in plan.shapes),
        max(n * k + 8 * max(n, k) for _, n, k in plan.shapes),
    )
    if any(
        m * k > workspace[0] or n * k > workspace[1]
        for m, n, k in (item.geometry.physical for item in stages)
    ):
        return None
    return RootNativeSchedule(
        root,
        root.phase_index,
        stages,
        ChainedExecution(plan.threads),
        _scope(plan),
        plan.tensor_aliases,
        tuple(plan.tensor_aliases.items()),
        workspace,
        max(item.geometry.physical[1] for item in stages),
        len(plan.dots),
    )


def joined_root(plan: ChainedMatmulPlan) -> bool:
    """Only the original root or its explicitly bound mixed schedule."""
    if plan.strategy == "warp":
        return plan.root_native is None
    if plan.strategy != STRATEGY or not isinstance(
        plan.root_native, RootNativeSchedule
    ):
        return False
    plan.root_native.check(plan)
    return True


def native_facts(plan: ChainedMatmulPlan) -> object:
    """Snapshot instruction/physical choices in existing joint-plan witnesses."""
    native = plan.root_native
    if native is None:
        return None
    return (
        id(native),
        native.root,
        native.phase,
        tuple(
            (
                item.index,
                item.node,
                item.geometry.logical,
                item.geometry.transpose,
                item.geometry.native_rows,
                item.operands,
                item.leaves,
                item.dtype,
            )
            for item in native.stages
        ),
        native.execution,
        native.scope,
        id(native.aliases),
        native.alias_prefix,
        native.original_workspace_elements,
        native.tmem_columns,
        native.barrier_count,
    )
