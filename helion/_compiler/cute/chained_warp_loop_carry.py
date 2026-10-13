"""A lexical FP32 carry owned by one unchanged warp-C register partition.

The first implementation admits one contraction and one carry. Incoming stores
retain their original expressions and execute after operand preparation but
before the in-place seed. No coefficient formula or loop extent is recognized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain
from .chained_loop_tmem_bridges import _ancestors
from .chained_loop_tmem_bridges import _coordinate_path
from .chained_preparation_cut import _shape_input
from .chained_store_expression import _node_facts
from .chained_store_expression import lower_store_point
from .chained_store_expression import materialized_store_read

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..generate_ast import GenerateAST
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .contraction_region import ContractionCarry


def _revision(plan: ChainedMatmulPlan) -> tuple[object, ...]:
    assert plan.loop is not None
    return (
        chain._register_bridge_revision(plan),
        _node_facts(tuple(plan.loop.root.graph.nodes)),
        plan.loop,
        plan.strategy,
        tuple(
            (node, node.meta["val"].untyped_storage()._cdata)
            for graph in (plan.loop.root.graph, plan.loop.body.graph)
            for node in graph.nodes
            if node.target is _tracing_ops._host_tensor
            and isinstance(node.meta.get("val"), torch.Tensor)
        ),
    )


def carry_topology(
    shape: tuple[int, int], threads: int, row_warps: int = 1
) -> tuple[int, int, int] | None:
    """Cover complete native C atoms with all configured warp participants."""
    rows, columns = shape
    if (
        threads not in (32, 64, 128, 256)
        or rows <= 0
        or columns <= 0
        or rows % 16
        or columns % 8
    ):
        return None
    warps = threads // 32
    if (
        row_warps not in (1, 2, 4, 8)
        or row_warps > warps
        or warps % row_warps
        or rows % (16 * row_warps)
        or columns % (8 * (warps // row_warps))
    ):
        return None
    return row_warps, warps // row_warps, 1


def _value_ancestors(node: Node) -> set[Node]:
    """A typed size query does not read the elements of its tensor input."""
    result: set[Node] = set()
    pending = [node]
    while pending:
        current = pending.pop()
        if current not in result:
            result.add(current)
            if current.target is not torch.ops.aten.sym_size.int:
                pending.extend(current.all_input_nodes)
    return result


@dataclass(frozen=True)
class WarpLoopCarryPlan:
    carry: ContractionCarry
    seed: Node
    shape: tuple[int, int]
    topology: tuple[int, int, int]
    stores: tuple[Node, ...]
    final_sources: tuple[tuple[Node, Node], ...]
    revision: tuple[object, ...]

    def check(self, plan: ChainedMatmulPlan) -> None:
        if (
            self.revision != _revision(plan)
            or self.topology
            != carry_topology(self.shape, plan.threads, self.topology[0])
            or plan.dots != (self.carry.output,)
            or plan.shapes[0][:2] != self.shape
        ):
            raise chain._UnsupportedChain("warp loop carry ownership changed")


def plan_warp_loop_carry(
    plan: ChainedMatmulPlan, row_warps: int = 1
) -> WarpLoopCarryPlan | None:
    loop = plan.loop
    if (
        loop is None
        or plan.strategy != "warp"
        or len(plan.dots) != 1
        or len(loop.region.carries) != 1
        or loop.region.scans
        or loop.region.reductions
        or plan.pointwise_cache is not None
        or plan.preparation_pipeline is not None
        or plan.root_typed_operands is not None
        or plan.contraction_groups is not None
        or plan.k_schedule is not None
        or plan.initialized_accumulator is not None
        or plan.late_rhs_reuse is not None
        or plan.root_exports is not None
        or plan.scan_exports
        or plan.direct_output
        or any(
            node.target is torch.ops.aten.sym_size.int and not _shape_input(node)
            for node in loop.region.nodes
        )
    ):
        return None
    carry = loop.region.carries[0]
    output = plan.dots[0]
    seed = output.args[2]
    shape = cast("tuple[int, int]", plan.shapes[0][:2])
    topology = carry_topology(shape, plan.threads, row_warps)
    if (
        topology is None
        or carry.output is not output
        or carry.input.meta["val"].dtype != torch.float32
        or output.meta["val"].dtype != torch.float32
        or chain._shape(carry.input) != shape
        or not isinstance(seed, Node)
        or seed.meta["val"].dtype != torch.float32
        or plan.shapes[0][2] % 16
        or any(
            carry.input in _value_ancestors(cast("Node", x)) for x in output.args[:2]
        )
    ):
        return None
    ancestors = {node: _value_ancestors(node) for node in loop.region.nodes}
    coordinates = ("row", "column")
    seed_path = _coordinate_path(seed, coordinates, carry.input, coordinates, ancestors)
    if seed_path is None:
        return None
    allowed = {*seed_path, output}
    stores = loop.region.stores
    if not stores:
        return None
    for store in stores:
        value = cast("Node", store.args[2])
        path = _coordinate_path(value, coordinates, carry.input, coordinates, ancestors)
        if path is None or output in ancestors[value]:
            return None
        # Addresses and predicates cannot depend on the register state.
        if any(
            carry.input in ancestors[node]
            for node in store.all_input_nodes
            if node is not value
        ):
            return None
        allowed.update((*path, store))
    pending, seen = [carry.input], set()
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        if node not in allowed:
            return None
        if node is output or node in stores:
            continue
        pending.extend(
            user
            for user in node.users
            if user.target is not torch.ops.aten.sym_size.int
        )
    # Moving an incoming store before the seed may not overwrite any of the
    # seed's host reads. Original typed storage identities remain authoritative.
    for store in stores:
        target = cast("Node", store.args[0]).meta["val"]
        for node in ancestors[seed]:
            if node.target is memory_ops.load:
                tensor = cast("Node", node.args[0]).meta["val"]
                if target.untyped_storage()._cdata == tensor.untyped_storage()._cdata:
                    return None
    root_ancestors = {node: _ancestors(node) for node in loop.root.graph.nodes}
    final_sources = []
    for store in loop.final_stores:
        value = cast("Node", store.args[2])
        source = next(
            (
                node
                for node, index in reversed(loop.final_values)
                if index == carry.input_index
                and _coordinate_path(
                    value, coordinates, node, coordinates, root_ancestors
                )
                is not None
            ),
            None,
        )
        if source is None:
            return None
        final_sources.append((store, source))
    return WarpLoopCarryPlan(
        carry, seed, shape, topology, stores, tuple(final_sources), _revision(plan)
    )


class WarpLoopCarryEmission:
    """Actual initial fill, seed, issue and stores share one register owner."""

    def __init__(self, cg: GenerateAST, plan: ChainedMatmulPlan) -> None:
        assert plan.warp_loop_carry is not None
        self.cg = cg
        self.plan = plan
        self.selected = plan.warp_loop_carry
        self.selected.check(plan)
        self.initialized = False
        self.seed: tuple[str, ...] | None = None
        self.issue: tuple[str, ...] | None = None
        self.finalized = False

    @property
    def topology(self) -> tuple[int, int, int]:
        return self.selected.topology

    def initialize(self) -> list[str]:
        self.selected.check(self.plan)
        if self.initialized:
            raise chain._UnsupportedChain("warp carry initialized twice")
        loop = self.plan.loop
        assert loop is not None
        dtype = CompileEnvironment.current().backend.dtype_str(
            self.plan.operand_dtype(0)
        )
        shape = self.selected.shape
        index = "chain_carry_init_index"
        coordinates = ("chain_carry_init_row", "chain_carry_init_column")
        expression = chain._Expression(self.cg, self.plan, {})
        expression.coordinate_names.update(coordinates)
        value = expression.value(
            loop.inputs[self.selected.carry.input_index], coordinates
        )
        self.initialized = True
        return [
            f"chain_0_mma = cute.make_tiled_mma(cute.make_mma_atom(cute.nvgpu.warp.MmaF16BF16Op({dtype}, cutlass.Float32, (16, 8, 16))), atom_layout_mnk={self.topology!r})",
            "chain_0_thr = chain_0_mma.get_slice(chain_thread)",
            f"chain_0_acc = cute.make_rmem_tensor(chain_0_mma.partition_shape_C({shape!r}), cutlass.Float32)",
            f"chain_carry_coords = chain_0_thr.partition_C(cute.make_identity_tensor({shape!r}))",
            f"for {index} in cutlass.range_constexpr(cute.size(chain_0_acc)):",
            f"    {', '.join(coordinates)} = chain_carry_coords[{index}]",
            chain._indent(expression.lines),
            f"    chain_0_acc[{index}] = cutlass.Float32({value})",
        ]

    def _store(self, store: Node, source: Node, prefix: str) -> tuple[str, ...]:
        loop = self.plan.loop
        assert loop is not None
        coordinates = (f"{prefix}_row", f"{prefix}_column")
        index = f"{prefix}_index"
        boundaries = loop.boundaries()
        read = materialized_store_read(
            self.plan, boundaries, source, coordinates, f"chain_0_acc[{index}]"
        )
        point = lower_store_point(
            self.cg,
            self.plan,
            store,
            boundaries,
            [],
            [],
            coordinates=coordinates,
            coordinate_names=coordinates,
            completed_read=read,
        )
        if any(
            node is not source or coords != coordinates for node, coords in point.reads
        ):
            raise chain._UnsupportedChain(
                "incoming store changed warp carry coordinates"
            )
        fallback = (
            f"for {index} in cutlass.range_constexpr(cute.size(chain_0_acc)):",
            f"    {', '.join(coordinates)} = chain_carry_coords[{index}]",
            chain._indent(point.lines),
            f"    if {' and '.join(point.bounds)}:",
            f"        {point.target}[{', '.join(point.indices)}] = {point.value}",
        )
        from .chained_warp_global_store import render_warp_global_store

        return (
            render_warp_global_store(
                self.cg,
                self.plan,
                store,
                point,
                accumulator="chain_0_acc",
                thread_slice="chain_0_thr",
                shape=self.selected.shape,
                prefix=prefix,
                index=index,
                fallback=fallback,
            )
            or fallback
        )

    def seed_lines(self) -> list[str]:
        self.selected.check(self.plan)
        if not self.initialized or self.seed is not None:
            raise chain._UnsupportedChain("warp carry seed lacks its initial owner")
        incoming = [
            line
            for index, store in enumerate(self.selected.stores)
            for line in self._store(
                store, self.selected.carry.input, f"chain_entering_{index}"
            )
        ]
        index = "chain_carry_seed_index"
        coordinates = ("chain_carry_seed_row", "chain_carry_seed_column")
        expression = chain._Expression(self.cg, self.plan, {})
        expression.coordinate_names.update(coordinates)
        expression.fragments[self.selected.carry.input] = (
            coordinates,
            f"chain_0_acc[{index}]",
        )
        value = expression.value(self.selected.seed, coordinates)
        if any(
            node is self.selected.carry.input and coords != coordinates
            for node, coords in expression.memo
        ):
            raise chain._UnsupportedChain("cross-coordinate warp carry seed")
        lines = [
            *incoming,
            f"for {index} in cutlass.range_constexpr(cute.size(chain_0_acc)):",
            f"    {', '.join(coordinates)} = chain_carry_coords[{index}]",
            chain._indent(expression.lines),
            f"    chain_0_acc[{index}] = cutlass.Float32({value})",
        ]
        self.seed = tuple(lines)
        return lines

    def check_issue(
        self,
        prefix: str,
        dtype: str,
        shape: tuple[int, int, int],
        threads: int,
        seed: Sequence[str],
        execution: ChainedExecution,
    ) -> None:
        self.selected.check(self.plan)
        if (
            not self.initialized
            or self.issue is not None
            or prefix != "chain_0"
            or shape != self.plan.shapes[0]
            or threads != self.plan.threads
            or execution.threads != threads
            or execution.thread != "chain_thread"
            or tuple(seed) != self.seed
            or dtype
            != CompileEnvironment.current().backend.dtype_str(
                self.plan.operand_dtype(0)
            )
        ):
            raise chain._UnsupportedChain("warp carry issue changed its owner")

    def record_issue(self, lines: Sequence[str]) -> None:
        if self.issue is not None or self.seed is None:
            raise chain._UnsupportedChain("warp carry issue repeated")
        self.issue = tuple(lines)

    def final_stores(self) -> list[str]:
        self.selected.check(self.plan)
        if self.issue is None or self.finalized:
            raise chain._UnsupportedChain("warp carry final export lacks completion")
        self.finalized = True
        return [
            line
            for index, (store, source) in enumerate(self.selected.final_sources)
            for line in self._store(store, source, f"chain_final_carry_{index}")
        ]
