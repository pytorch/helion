"""Whole-warp scheduling of independent, already-published row islands.

The graph proof grants movement, not readiness. Each code-generation attempt
records the original joined materializations in graph order. Preallocated names
never count as completion. Original C lifetimes and all allocations are retained.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

from ...language import memory_ops
from .chained_execution import ChainedExecution
from .chained_root_native import joined_root
from .chained_row_registers import _pointwise
from .chained_row_registers import _record

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_root_native_emission import RootNativeCompletion
    from .chained_row_packing import RowPackingProof
    from .chained_row_registers import RowRegisterIsland
    from .chained_row_registers import RowRegisterPlan


def _storage(plan: ChainedMatmulPlan) -> tuple[object, ...]:
    # tensor_aliases is the emitter's incrementally populated argument-name
    # map, not physical alias evidence. Raw storage is read-only throughout the
    # proved interval; joined materializations are separately bound by receipt.
    return (
        plan.scan_exports,
        plan.root_exports,
        plan.warp_workspace,
        plan.warp_result_transport,
        plan.loop_workspace,
        plan.root_workspace,
        plan.warp_mma_stages,
    )


@dataclass(frozen=True)
class RowIslandReads:
    island: RowRegisterIsland
    materialized: tuple[Node, ...]
    raw_loads: tuple[Node, ...]
    closure: tuple[Node, ...]


@dataclass(frozen=True)
class RowBatch:
    event: Node
    islands: tuple[RowIslandReads, ...]
    crossed: tuple[Node, ...]
    # Reference programs may execute sequentially on repeated legal warps.
    # Their order is distinct from the packed physical subgroup assignment.
    original_warps: tuple[int, ...] | None = None


@dataclass(frozen=True)
class RowParallelPlan:
    rows: RowRegisterPlan
    batches: tuple[RowBatch, ...]
    events: tuple[Node, ...]
    storage: object
    selection: object

    def check(self, plan: ChainedMatmulPlan) -> None:
        from .chained_matmul import _UnsupportedChain

        self.rows.check(plan)
        if (
            plan.row_registers is not self.rows
            or self.storage != _record(_storage(plan))
            or self.selection != _record((self.batches, self.events))
        ):
            raise _UnsupportedChain("parallel row graph, movement or storage changed")


def plan_row_parallel(plan: ChainedMatmulPlan) -> RowParallelPlan | None:
    """Batch closed islands at existing events; no scalar recipe is rewritten.

    The ordinary root already proves fresh external output storage. Here the
    complete prefix is also checked for effects/unknown calls, including raw
    image initialization outside reduction-influenced island membership. Raw
    read aliases are harmless because no write can be crossed. Every shared
    source keeps its original lifetime through the original later consumer.
    """
    from .chained_collectives import collective_nodes
    from .chained_matmul import _supported

    rows = plan.row_registers
    region = plan.region
    if (
        rows is None
        or region is None
        or len(rows.islands) < 2
        or plan.threads < 64
        or plan.threads % 32
        or not joined_root(plan)
        or plan.loop is not None
        or plan.preparation_pipeline is not None
        or plan.contraction_groups is not None
        or plan.root_typed_operands is not None
        or plan.pointwise_cache is not None
        or plan.initialized_accumulator is not None
        or plan.late_rhs_reuse is not None
        or plan.k_schedule is not None
        or plan.warp_mma_stages
    ):
        return None
    rows.check(plan)
    nodes = region.nodes
    positions = {node: index for index, node in enumerate(nodes)}
    last = max(positions[item.output] for item in rows.islands)

    def pure(node: Node) -> bool:
        return node.op == "placeholder" or (
            node.op == "call_function"
            and node.target is not memory_ops.store
            and not node.is_impure()
            and (_supported(node) or _pointwise(node))
        )

    if any(not pure(node) for node in nodes[: last + 1]):
        return None
    physical = {*plan.dots, *collective_nodes(plan)}
    island_nodes = {node for island in rows.islands for node in island.members}
    proofs: list[RowIslandReads] = []
    for island in rows.islands:
        pending = [island.output]
        closure: set[Node] = set()
        published: set[Node] = set()
        raw: set[Node] = set()
        while pending:
            node = pending.pop()
            if node in closure:
                continue
            closure.add(node)
            if node in physical and node not in island.members:
                published.add(node)
                continue
            if node in island_nodes and node not in island.members:
                return None
            if not pure(node):
                return None
            if node.target is memory_ops.load:
                raw.add(node)
            pending.extend(node.all_input_nodes)
        if published != set(island.external_reads):
            # Unsupported collectives or another island must not be silently
            # expanded as a scalar input at an earlier graph position.
            return None
        proofs.append(
            RowIslandReads(
                island,
                tuple(node for node in nodes if node in published),
                tuple(node for node in nodes if node in raw),
                tuple(node for node in nodes if node in closure),
            )
        )
    pending_proofs = list(proofs)
    batches: list[RowBatch] = []
    while pending_proofs:
        anchor = pending_proofs[0]
        event = anchor.island.first_collective
        start = positions[event]
        ready = [
            item
            for item in pending_proofs
            if all(positions[node] < start for node in item.materialized)
        ]
        warp_count = plan.threads // 32
        limit = warp_count
        if plan.row_frontier:
            if not plan.row_subwarp_packing:
                return None
            width = anchor.island.shape[0]
            if width not in (2, 4, 8, 16):
                return None
            # Keep a contiguous equal-shape prefix and complete physical warps.
            # The existing graph proof below owns every operation crossed.
            same_shape = 0
            for item in ready[: plan.threads // width]:
                if item.island.shape != anchor.island.shape:
                    break
                same_shape += 1
            group_multiple = 32 // width
            limit = same_shape // group_multiple * group_multiple
        ready = ready[:limit]
        if not ready or ready[0] is not anchor:
            return None
        end = max(positions[item.island.first_collective] for item in ready)
        # All crossed contractions retain their complete original C allocation
        # or original workspace region and read/publish joins. Row outputs are
        # separate allocations; no crossed operation can clobber one.
        crossed = nodes[start:end]
        if any(not pure(node) for node in crossed):
            return None
        original_warps = (
            tuple(index % warp_count for index in range(len(ready)))
            if len(ready) > warp_count
            else None
        )
        batches.append(RowBatch(event, tuple(ready), tuple(crossed), original_warps))
        selected = {item.island.output for item in ready}
        pending_proofs = [
            item for item in pending_proofs if item.island.output not in selected
        ]
    if not any(len(batch.islands) > 1 for batch in batches):
        return None
    if plan.row_frontier and not any(
        batch.original_warps is not None for batch in batches
    ):
        return None
    events = tuple(node for node in nodes if node in physical)
    return RowParallelPlan(
        rows,
        tuple(batches),
        events,
        _record(_storage(plan)),
        _record((tuple(batches), events)),
    )


@dataclass(frozen=True)
class RowInputCompletion:
    owner: object
    node: Node
    binding: str
    event: int
    execution: ChainedExecution
    phase: tuple[str, ...]


@dataclass(frozen=True)
class RowBatchCompletion:
    owner: object
    batch: RowBatch
    inputs: tuple[RowInputCompletion, ...]
    execution: ChainedExecution
    phase: tuple[str, ...]
    packing: RowPackingProof | None = None


@dataclass
class RowParallelEmission:
    parallel: RowParallelPlan
    owner: object = field(default_factory=object)
    cursor: int = 0
    packed_batches: int = 0
    completed: dict[Node, RowInputCompletion] = field(default_factory=dict)
    batches: list[RowBatchCompletion] = field(default_factory=list)

    @property
    def proposal(self) -> RowRegisterPlan:
        return self.parallel.rows

    def _visit(
        self, plan: ChainedMatmulPlan, node: Node, execution: ChainedExecution
    ) -> None:
        from .chained_matmul import _UnsupportedChain

        self.parallel.check(plan)
        if (
            execution != ChainedExecution(plan.threads)
            or self.cursor >= len(self.parallel.events)
            or self.parallel.events[self.cursor] is not node
        ):
            raise _UnsupportedChain("parallel row event or execution order changed")
        self.cursor += 1

    def complete_original(
        self,
        plan: ChainedMatmulPlan,
        node: Node,
        binding: str,
        phase: tuple[str, ...],
        execution: ChainedExecution,
        *,
        native_completion: RootNativeCompletion | None = None,
    ) -> None:
        """Called only after the original producer's shared publication join."""
        from .chained_matmul import _UnsupportedChain

        self.parallel.check(plan)
        selected = (
            plan.root_native.selected(plan, plan.dots.index(node))
            if plan.root_native is not None and node in plan.dots
            else None
        )
        if (selected is None) != (native_completion is None):
            raise _UnsupportedChain("parallel row input lacks native completion")
        if native_completion is not None:
            native_completion.check(plan, node, binding, phase)
        if node in plan.dots:
            self._visit(plan, node, execution)
        if (
            not self.cursor
            or self.parallel.events[self.cursor - 1] is not node
            or execution != ChainedExecution(plan.threads)
            or node in self.completed
            or not phase
            or phase[-1] != execution.sync
        ):
            raise _UnsupportedChain(
                "parallel row input lacks its original joined phase"
            )
        self.completed[node] = RowInputCompletion(
            self.owner, node, binding, self.cursor - 1, execution, phase
        )

    def emit_collective(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        boundaries: dict[Node, str],
        node: Node,
        execution: ChainedExecution,
    ) -> list[str] | None:
        from .chained_matmul import _UnsupportedChain
        from .chained_row_register_emission import _emit_island

        self._visit(plan, node, execution)
        island = next(
            (item for item in self.proposal.islands if node in item.reductions), None
        )
        if island is None:
            return None
        batch = next(
            (item for item in self.parallel.batches if item.event is node), None
        )
        if batch is None:
            if island.output not in self.completed:
                raise _UnsupportedChain("parallel row selection precedes batch join")
            return []
        inputs: list[RowInputCompletion] = []
        for item in batch.islands:
            if item.island.output in boundaries or item.island.output in self.completed:
                raise _UnsupportedChain("duplicate parallel row publication")
            for source in item.materialized:
                proof = self.completed.get(source)
                if (
                    proof is None
                    or proof.owner is not self.owner
                    or proof.node is not source
                    or proof.event >= self.cursor - 1
                    or proof.execution != execution
                    or boundaries.get(source) != proof.binding
                    or not proof.phase
                    or proof.phase[-1] != execution.sync
                ):
                    raise _UnsupportedChain("parallel row input is not completed")
                inputs.append(proof)
        # Do not mutate bindings or readiness if any typed recipe fails.
        original_warps = batch.original_warps or tuple(range(len(batch.islands)))
        if (
            len(original_warps) != len(batch.islands)
            or any(not 0 <= warp < execution.threads // 32 for warp in original_warps)
            or (batch.original_warps is not None and not plan.row_frontier)
        ):
            raise _UnsupportedChain("packed row reference warp ownership changed")
        programs = [
            _emit_island(cg, plan, boundaries, item.island, execution, warp=warp)
            for warp, item in zip(original_warps, batch.islands, strict=True)
        ]
        lines = [line for program in programs for line in program]
        packing = None
        if plan.row_subwarp_packing:
            from .chained_coordinate_bounds import _launch
            from .chained_row_packing import pack_emitted_batch

            launch = _launch(cg, plan)
            if launch is None or launch[0] != (execution.threads, 1, 1):
                raise _UnsupportedChain("packed row lacks its actual root launch")
            receipts: list[RowPackingProof] = []
            packed = pack_emitted_batch(
                [item.island for item in batch.islands],
                programs,
                execution,
                cg.device_function.new_var,
                receipts=receipts,
                original_warps=original_warps,
            )
            if packed is None and batch.original_warps is not None:
                raise _UnsupportedChain("enlarged row frontier could not be packed")
            if packed is not None:
                if len(receipts) != 1:
                    raise _UnsupportedChain("packed row lacks its scalar receipt")
                packing = receipts[0]
                lines = packed.splitlines()
                self.packed_batches += 1
        lines.append(execution.sync)
        self.parallel.check(plan)
        receipt = RowBatchCompletion(
            self.owner, batch, tuple(inputs), execution, tuple(lines), packing
        )
        self.batches.append(receipt)
        for item in batch.islands:
            output = item.island.output
            boundaries[output] = item.island.name
            self.completed[output] = RowInputCompletion(
                self.owner,
                output,
                item.island.name,
                self.cursor - 1,
                execution,
                receipt.phase,
            )
        return lines

    def validate(self, plan: ChainedMatmulPlan) -> None:
        from .chained_matmul import _UnsupportedChain

        self.parallel.check(plan)
        for receipt in self.batches:
            if receipt.packing is not None:
                receipt.packing.check(
                    tuple(item.island for item in receipt.batch.islands),
                    receipt.execution,
                    receipt.phase,
                )
        if self.packed_batches != sum(
            item.packing is not None for item in self.batches
        ):
            raise _UnsupportedChain("packed row completion count changed")
        if (
            (plan.row_subwarp_packing and not self.packed_batches)
            or self.cursor != len(self.parallel.events)
            or tuple(item.batch for item in self.batches) != self.parallel.batches
            or any(
                item.owner is not self.owner
                or item.execution != ChainedExecution(plan.threads)
                or not item.phase
                or item.phase[-1] != item.execution.sync
                or any(
                    source.owner is not self.owner
                    or source.execution != item.execution
                    or self.completed.get(source.node) is not source
                    or not source.phase
                    or source.phase[-1] != item.execution.sync
                    for source in item.inputs
                )
                for item in self.batches
            )
            or any(item.output not in self.completed for item in self.proposal.islands)
        ):
            raise _UnsupportedChain("parallel row publication sequence is incomplete")
