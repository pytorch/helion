"""Late-proven FP32 carry ownership with a separate typed operand snapshot.

The final group's whole C arena is persistent. Only its carry member is state;
other members are ordinary outputs. No narrowed image substitutes for FP32 C.
Initial and final shared copies preserve the existing loop/export interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .chained_execution import ChainedExecution
from .chained_tmem_drains import FP32DrainPanels
from .chained_tmem_drains import FP32SharedTarget
from .chained_tmem_drains import ScalarPublication
from .chained_tmem_drains import emit_streamed_fp32_drain
from .chained_tmem_drains import plan_fp32_drain
from .chained_tmem_transport import TmemOperandBinding
from .warp_specialized_plan import VALID_TMEM_COLUMNS
from .warp_specialized_plan import TensorMemoryRegionRequest
from .warp_specialized_plan import allocate_tmem_regions
from .warp_specialized_plan import packed_input_tmem_columns

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_contraction_groups import ContractionGroup
    from .chained_loop_tmem_carry import LoopTmemCarryCandidate
    from .chained_matmul import ChainedMatmulPlan
    from .chained_preparation_cut import PreparationCut
    from .chained_tmem_accumulator import TmemAccumulatorResidency


@dataclass(frozen=True)
class LoopTmemCarryTransport:
    candidate: LoopTmemCarryCandidate
    arena_offset: int
    snapshot_offset: int
    required_columns: int
    dtype: str
    snapshot_lines: tuple[str, ...]
    snapshot_value: str
    accumulator_lines: tuple[str, ...]
    accumulator_value: str
    snapshot_proof: tuple[object, ...] | None = None
    drain_panels: FP32DrainPanels | None = None
    _drain_facts: tuple[object, ...] | None = None
    snapshot_seed: tuple[tuple[str, ...], str] | None = None

    @property
    def prefix(self) -> str:
        return f"chain_resident_carry_{self.candidate.carry_index}"

    @property
    def seed(self) -> str:
        return (
            f"chain_{self.candidate.final_group.stages[0]}_seed_{self.candidate.member}"
        )

    @property
    def arena(self) -> str:
        return f"(chain_tptr + {self.arena_offset})"

    def operand(self, group: ContractionGroup) -> TmemOperandBinding | None:
        if not any(use.group == group for use in self.candidate.snapshot_users):
            return None
        return TmemOperandBinding(
            group,
            self.candidate.snapshot_shape,
            self.candidate.snapshot_dtype,
            self.snapshot_offset,
        )

    def accumulator(self, stage: int) -> str | None:
        residency = self.candidate.residency
        return (
            self.arena
            if stage == self.candidate.final_group.stages[0]
            or residency is not None
            and stage == residency.source_stage
            else None
        )

    def view(self) -> list[str]:
        prefix = self.prefix
        shape = self.candidate.geometry.physical[:2]
        offset = self.arena_offset + self.candidate.member_offset
        return [
            f"{prefix}_mma = chain_sm100.make_trivial_tiled_mma({self.dtype}, {self.dtype}, cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K, cutlass.Float32, tcgen05.CtaGroup.ONE, {shape!r}, tcgen05.OperandSource.SMEM)",
            f"{prefix}_layout = {prefix}_mma.make_fragment_C({prefix}_mma.partition_shape_C({shape!r})).layout",
            f"{prefix}_acc = cute.make_tensor(chain_tptr + {offset}, {prefix}_layout)",
            f"{prefix}_slice = {prefix}_mma.get_slice(0)",
        ]

    def shared_transfer(self, *, upload: bool) -> list[str]:
        """First CTA warpgroup copies the original FP32 view once at each end."""
        from .chained_matmul import _indent
        from .chained_tcgen05 import _load_result

        prefix = self.prefix
        shape = self.candidate.geometry.physical[:2]
        row, column = f"{prefix}_row", f"{prefix}_column"
        logical = self.candidate.geometry.result_coordinates(row, column)
        shared = f"chain_loop_carry_{self.candidate.carry_index}"
        if upload:
            body = [
                f"{prefix}_copy = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition({min(32, shape[1] & -shape[1])})), cutlass.Float32), {prefix}_acc)",
                f"{prefix}_thread = {prefix}_copy.get_slice(chain_thread)",
                f"{prefix}_target = {prefix}_thread.partition_D({prefix}_acc)",
                f"{prefix}_identity = {prefix}_slice.partition_C(cute.make_identity_tensor({shape!r}))",
                f"{prefix}_coords = {prefix}_thread.partition_S({prefix}_identity)",
                f"{prefix}_values = cute.make_rmem_tensor({prefix}_coords.shape, cutlass.Float32)",
                f"for {prefix}_index in cutlass.range_constexpr(cute.size({prefix}_values)):",
                f"    {row}, {column} = {prefix}_coords[{prefix}_index]",
                f"    {prefix}_values[{prefix}_index] = {shared}[{', '.join(logical)}]",
                f"cute.copy({prefix}_copy, {prefix}_values, {prefix}_target)",
                "cute.arch.fence_view_async_tmem_store()",
            ]
        else:
            publication = ScalarPublication(
                f"{prefix}_index",
                row,
                column,
                (
                    FP32SharedTarget(
                        shared,
                        self.candidate.geometry.logical[:2],
                        self.candidate.geometry.transpose,
                    ),
                ),
            )
            if self.drain_panels is not None or self._drain_facts is not None:
                if (
                    self.drain_panels is None
                    or self._drain_facts != self._current_drain_facts()
                ):
                    raise ValueError("FP32 carry drain selection changed")
                body = emit_streamed_fp32_drain(
                    self.drain_panels,
                    f"{prefix}_drain",
                    prefix,
                    publication,
                    execution=ChainedExecution(128),
                )
            else:
                body = [
                    *_load_result(prefix, shape),
                    *publication.lines(f"{prefix}_values", f"{prefix}_coords"),
                ]
        return [
            "if chain_thread < 128:",
            _indent([*body, "chain_tmem_barrier.arrive_and_wait()"]),
            "cute.arch.sync_threads()",
        ]

    def _current_drain_facts(self) -> tuple[object, ...]:
        return (
            None if self.drain_panels is None else self.drain_panels.shape,
            self.candidate.geometry.physical,
            self.candidate.geometry.logical,
            self.candidate.geometry.transpose,
            self.arena_offset,
            self.candidate.member_offset,
            self.required_columns,
            self.candidate.carry_index,
        )

    def snapshot(
        self, execution: ChainedExecution, *, tile_columns: int = 0
    ) -> list[str]:
        from ... import exc
        from .chained_matmul import _indent
        from .chained_tcgen05 import _load_result
        from .chained_tmem_snapshots import PackedSnapshotPanels
        from .chained_tmem_snapshots import emit_streamed_snapshot
        from .chained_tmem_transport import emit_packed_tmem_fragment

        if type(tile_columns) is not int or tile_columns not in (0, 32):
            raise exc.BackendUnsupported(
                "cute", "snapshot tile columns must be 0 or 32"
            )
        if self.snapshot_seed is not None and tile_columns != 32:
            raise exc.BackendUnsupported("cute", "snapshot seed requires N32 panels")
        local = ChainedExecution(
            128,
            thread=execution.thread,
            warp=execution.warp,
            sync="chain_tmem_barrier.arrive_and_wait()",
        )
        prefix = f"{self.prefix}_snapshot"
        if tile_columns:
            if self.snapshot_proof is None or self.snapshot_proof != _snapshot_facts(
                self.candidate, self.dtype, self.snapshot_lines, self.snapshot_value
            ):
                raise exc.BackendUnsupported(
                    "cute", "streamed carry snapshot original expression proof changed"
                )
            if execution.threads < 128 or execution.threads % 128:
                raise exc.BackendUnsupported(
                    "cute", "streamed carry snapshot requires whole 128-thread teams"
                )
            try:
                panels = PackedSnapshotPanels(
                    self.candidate.snapshot_shape,
                    self.arena_offset + self.candidate.member_offset,
                    self.snapshot_offset,
                    self.required_columns,
                )
                if self.candidate.geometry.physical[:2] != panels.shape:
                    raise ValueError("snapshot no longer matches its original C view")
                body = emit_streamed_snapshot(
                    panels,
                    prefix,
                    self.prefix,
                    self.dtype,
                    (f"{prefix}_row", f"{prefix}_column"),
                    self.snapshot_lines,
                    self.snapshot_value,
                    execution=local,
                    source_update=self.snapshot_seed,
                )
            except ValueError as error:
                raise exc.BackendUnsupported("cute", str(error)) from error
            return [
                *(
                    [f"if {execution.thread} < 128:", _indent(body)]
                    if execution.threads > 128
                    else body
                ),
                execution.sync,
            ]
        body = [
            *_load_result(self.prefix, self.candidate.snapshot_shape, execution=local),
            *emit_packed_tmem_fragment(
                prefix,
                self.prefix,
                self.candidate.snapshot_shape,
                self.dtype,
                (f"{prefix}_row", f"{prefix}_column"),
                self.snapshot_lines,
                self.snapshot_value,
                execution=local,
                destination=f"(chain_tptr + {self.snapshot_offset})",
            ),
        ]
        return [
            *(
                [f"if {execution.thread} < 128:", _indent(body)]
                if execution.threads > 128
                else body
            ),
            execution.sync,
        ]

    def seed_values(
        self, execution: ChainedExecution, *, width: int | None = None
    ) -> list[str]:
        """Load exactly the seed segment before its original FP32 expression.

        The caller has established the existing seed segment, identity, copy,
        coordinates and RMEM values. Full-M128 load/store partitions must agree;
        dedicated CuTe layout and GPU tests establish that transport contract.
        """
        from .chained_matmul import _indent
        from .chained_result_transport import load_operation

        seed = self.seed
        shape = self.candidate.geometry.physical[:2]
        if width is not None:
            shape = (shape[0], width)
        return [
            f"{seed}_load = tcgen05.make_tmem_copy(cute.make_copy_atom({load_operation(shape)}, cutlass.Float32), {seed}_segment)",
            f"{seed}_reader = {seed}_load.get_slice({execution.thread})",
            f"{seed}_source = {seed}_reader.partition_S({seed}_segment)",
            f"cute.copy({seed}_load, {seed}_source, {seed}_values)",
            "cute.arch.fence_view_async_tmem_load()",
            f"for {seed}_index in cutlass.range_constexpr(cute.size({seed}_values)):",
            f"    {seed}_row, {seed}_column = {seed}_coords[{seed}_index]",
            _indent(self.accumulator_lines),
            f"    {seed}_values[{seed}_index] = cutlass.Float32({self.accumulator_value})",
            "chain_tmem_barrier.arrive_and_wait()",
        ]


def _snapshot_facts(
    candidate: LoopTmemCarryCandidate,
    dtype: str,
    lines: tuple[str, ...],
    value: str,
) -> tuple[object, ...]:
    """Retain the exact existing late fragment/domain proof, without re-lowering.

    The carry planner only admits canonical source cast paths. This token is
    not a new expression authority; it prevents later streaming from using a
    changed original graph, typed boundary, or pre-rendered expression.
    """
    from .chained_pipeline_storage import _freeze

    return (
        candidate,
        tuple(candidate.region.graph.nodes),
        tuple(
            (
                node,
                node.op,
                node.target,
                _freeze(node.args),
                _freeze(node.kwargs),
                (
                    (
                        metadata.dtype,
                        _freeze(metadata.shape),
                        _freeze(metadata.stride()),
                    )
                    if isinstance(metadata := node.meta.get("val"), torch.Tensor)
                    else _freeze(metadata)
                ),
                node.meta.get("lowering"),
                tuple(node.users),
            )
            for node in candidate.snapshot_nodes
        ),
        dtype,
        lines,
        value,
    )


def prepare_loop_tmem_carry(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    frontiers: dict[Node, str],
    groups: tuple[ContractionGroup, ...],
    residency: TmemAccumulatorResidency | None,
    occupied_columns: int,
    *,
    drain_tile_columns: int = 0,
    snapshot_cut: PreparationCut | None = None,
) -> LoopTmemCarryTransport | None:
    from ..compile_environment import CompileEnvironment
    from . import chained_matmul as chain
    from .chained_loop_tmem_carry import plan_loop_tmem_carry

    if type(drain_tile_columns) is not int or drain_tile_columns not in (0, 32):
        raise chain._UnsupportedChain("FP32 drain tile columns must be 0 or 32")
    candidate = plan_loop_tmem_carry(plan, groups, residency)
    if candidate is None:
        return None
    dtype = CompileEnvironment.current().backend.dtype_str(candidate.snapshot_dtype)
    prefix = f"chain_resident_carry_{candidate.carry_index}"
    snapshot = f"{prefix}_snapshot"
    coords = (f"{snapshot}_row", f"{snapshot}_column")
    source_coords = candidate.geometry.result_coordinates(*coords)
    first = None
    domains = []
    try:
        for use in candidate.snapshot_users:
            for operand, geometry in zip(
                use.operands, use.group.geometries, strict=True
            ):
                expression = chain._Expression(cg, plan, frontiers)
                expression.coordinate_names.update(coords)
                expression.fragments[candidate.input] = (
                    source_coords,
                    f"{prefix}_values[{snapshot}_index]",
                )
                operand_coords = geometry.operand("a", *coords)[1]
                value = expression.value(operand, operand_coords)
                domain = chain._operand_domain(cg, operand, operand_coords, plan)
                domains.append(domain)
                if first is None:
                    first = (
                        tuple(expression.lines),
                        chain._masked_operand(value, dtype, domain),
                    )
        seed = f"chain_{candidate.final_group.stages[0]}_seed_{candidate.member}"
        seed_coords = candidate.geometry.result_coordinates(
            f"{seed}_row", f"({seed}_column - {candidate.member_offset})"
        )
        expression = chain._Expression(cg, plan, frontiers)
        expression.coordinate_names.update((f"{seed}_row", f"{seed}_column"))
        expression.fragments[candidate.input] = (
            seed_coords,
            f"{seed}_values[{seed}_index]",
        )
        accumulator = expression.value(candidate.accumulator, seed_coords)
    except chain._UnsupportedChain:
        return None
    if first is None or any(domain != domains[0] for domain in domains[1:]):
        return None
    snapshot_seed = None
    if snapshot_cut is not None and snapshot_seed_available(candidate, snapshot_cut):
        update = chain._Expression(cg, plan, frontiers)
        update.coordinate_names.update(coords)
        update.fragments[candidate.input] = (
            source_coords,
            f"{prefix}_values[{snapshot}_index]",
        )
        try:
            value = update.value(candidate.accumulator, source_coords)
        except chain._UnsupportedChain:
            pass
        else:
            snapshot_seed = tuple(update.lines), value
    arena = (occupied_columns + 31) // 32 * 32
    snapshot_offset = (arena + candidate.arena_columns + 31) // 32 * 32
    snapshot_columns = packed_input_tmem_columns(candidate.snapshot_shape[1])
    if snapshot_offset + snapshot_columns > VALID_TMEM_COLUMNS[-1]:
        return None
    requests = [TensorMemoryRegionRequest("occupied", occupied_columns)]
    if arena != occupied_columns:
        requests.append(
            TensorMemoryRegionRequest("arena_padding", arena - occupied_columns)
        )
    requests.append(TensorMemoryRegionRequest("carry_arena", candidate.arena_columns))
    if snapshot_offset != arena + candidate.arena_columns:
        requests.append(
            TensorMemoryRegionRequest(
                "snapshot_padding", snapshot_offset - arena - candidate.arena_columns
            )
        )
    requests.append(TensorMemoryRegionRequest("carry_snapshot", snapshot_columns))
    layout = allocate_tmem_regions(tuple(requests))
    result = LoopTmemCarryTransport(
        candidate,
        layout.region("carry_arena").column_offset,
        layout.region("carry_snapshot").column_offset,
        layout.required_columns,
        dtype,
        *first,
        tuple(expression.lines),
        accumulator,
        _snapshot_facts(candidate, dtype, *first),
        plan_fp32_drain(candidate.geometry.physical[:2], drain_tile_columns),
        snapshot_seed=snapshot_seed,
    )
    if result.drain_panels is not None:
        from dataclasses import replace

        result = replace(result, _drain_facts=result._current_drain_facts())
    return result


def snapshot_seed_available(
    candidate: LoopTmemCarryCandidate, cut: PreparationCut
) -> bool:
    """Side inputs must cross this READY cut or be its invariant shared ports.

    The carry proof already excludes other carries and recurrence products on
    its coordinate-preserving path. This check binds early evaluation to the
    actual frame; a renderable arbitrary boundary map alone is insufficient.
    Original image nodes remain valid with a proved raw-widening replacement.
    """
    if (
        cut.region.graph is not candidate.region.graph
        or cut.region.nodes != candidate.region.nodes
        or cut.carries != candidate.region.carries
    ):
        return False
    path = set(candidate.accumulator_nodes)
    side_inputs = {
        source for node in path for source in node.all_input_nodes if source not in path
    }
    return side_inputs <= {image.node for image in cut.images} | set(cut.shared_inputs)
