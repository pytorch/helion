"""Render dependency-scheduled SparseCore programs through emit_pipeline."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .access import AccessKind
from .sc_base import SC_LANES
from .sc_base import SC_SHARED_ITEM_CHUNK
from .sc_base import _reject
from .sc_base import shared_acc_items
from .sparsecore_access import AtomicAddAccess
from .sparsecore_access import CachedLoadAccess
from .sparsecore_access import DirectLoadAccess
from .sparsecore_access import DirectStoreAccess
from .sparsecore_access import IndirectLoadAccess
from .sparsecore_access import IndirectStoreAccess
from .sparsecore_access import SparseCoreAccess
from .sparsecore_compute import SparseCoreCompute
from .sparsecore_program import SchedulePhase
from .sparsecore_program import ScheduleStage
from .sparsecore_program import SparseCoreProgram
from .sparsecore_program import TaskKind
from .sparsecore_program import load_buffer_shape

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST


@dataclass
class CodegenResources:
    buffers: dict[torch.fx.Node, str]
    output_buffers: dict[torch.fx.Node, str]
    atomic_buffers: dict[torch.fx.Node, str]
    atomic_zero_buffers: dict[torch.fx.Node, str]
    semaphore: str
    semaphore_index: dict[torch.fx.Node, int]


def allocate_resources(
    program: SparseCoreProgram, codegen: GenerateAST
) -> CodegenResources:
    device_fn = codegen.device_function
    assert program.schedule is not None
    depth = program.schedule.depth
    items = program.geometry.items_per_subcore
    buffers: dict[torch.fx.Node, str] = {}
    output_buffers: dict[torch.fx.Node, str] = {}
    atomic_buffers: dict[torch.fx.Node, str] = {}
    atomic_zero_buffers: dict[torch.fx.Node, str] = {}
    semaphore_index: dict[torch.fx.Node, int] = {}

    for access in program.loads:
        node = access.site.node
        if isinstance(access, CachedLoadAccess):
            buffers[node] = device_fn.register_scratch(
                load_buffer_shape(program, access),
                access.layout.storage_dtype,
                "sc_cached",
            )
            continue
        if isinstance(access, IndirectLoadAccess) or node not in program.index_nodes:
            semaphore_index[node] = len(semaphore_index)
        buffers[node] = device_fn.register_scratch(
            (depth, *load_buffer_shape(program, access)),
            access.layout.storage_dtype,
            "sc_value",
        )

    for access in program.stores:
        value_size = access.layout.value_size
        shape = (items, SC_LANES if value_size == 1 else value_size)
        output_buffers[access.site.node] = device_fn.register_scratch(
            shape, access.layout.storage_dtype, "sc_output"
        )
        if isinstance(access, AtomicAddAccess):
            shared_items = shared_acc_items(int(access.site.tensor.shape[0]))
            atomic_buffers[access.site.node] = device_fn.register_scratch(
                (shared_items, value_size), torch.float32, "sc_acc", "vmem_shared"
            )
            atomic_zero_buffers[access.site.node] = device_fn.register_scratch(
                (SC_LANES, value_size), torch.float32, "sc_zero"
            )

    semaphore = device_fn.register_dma_semaphore(
        "sc_dma", (depth, max(len(semaphore_index), 1))
    )
    return CodegenResources(
        buffers,
        output_buffers,
        atomic_buffers,
        atomic_zero_buffers,
        semaphore,
        semaphore_index,
    )


def _tensor_name(codegen: GenerateAST, tensor: torch.Tensor) -> str:
    return codegen.device_function.tensor_arg(tensor).name


def _direct_copy_lines(
    program: SparseCoreProgram,
    access: DirectLoadAccess,
    resources: CodegenResources,
    codegen: GenerateAST,
    method: str | None,
) -> list[str]:
    stream = access.stream
    assert stream is not None
    source = _tensor_name(codegen, access.site.tensor)
    destination = resources.buffers[access.site.node]
    items = program.geometry.items_per_subcore
    if access.site.node in program.index_nodes:
        if method is not None:
            raise AssertionError("SparseCore index streams must be synchronous")
        count = items * stream.elements_per_item
        base = stream.group * program.geometry.padded_items * stream.elements_per_item
        offset = (
            f"{base} + _sc_o0 * {stream.elements_per_item}"
            if base
            else (
                "_sc_o0"
                if stream.elements_per_item == 1
                else f"_sc_o0 * {stream.elements_per_item}"
            )
        )
        return [
            (
                f"        pltpu.sync_copy({source}.at[pl.ds({offset}, {count})], "
                f"{destination}.at[_sc_q, pl.ds(0, {count})])"
            )
        ]
    base = stream.group * program.geometry.padded_items
    offset = f"{base} + _sc_o0" if base else "_sc_o0"
    if method is not None:
        semaphore_index = resources.semaphore_index[access.site.node]
        return [
            (
                f"        pltpu.make_async_copy({source}.at[pl.ds({offset}, {items})], "
                f"{destination}.at[_sc_q], {resources.semaphore}.at["
                f"_sc_q, {semaphore_index}]).{method}()"
            )
        ]
    return [
        (
            f"        pltpu.sync_copy({source}.at[pl.ds({offset}, {items})], "
            f"{destination}.at[_sc_q])"
        )
    ]


def _index_access(
    program: SparseCoreProgram,
    access: IndirectLoadAccess | IndirectStoreAccess | AtomicAddAccess,
) -> SparseCoreAccess:
    index_access = program.access_by_node.get(access.index_node)
    if index_access is None or index_access.site.kind is not AccessKind.LOAD:
        raise NotImplementedError(
            "SparseCore indirect index must be produced by a lowered load"
        )
    return index_access


def _offset_lines(
    program: SparseCoreProgram,
    access: IndirectLoadAccess,
    index_access: SparseCoreAccess,
    resources: CodegenResources,
) -> list[str]:
    offset = access.index_offset
    if not offset:
        return []
    if isinstance(index_access, IndirectLoadAccess):
        raise NotImplementedError(
            "SparseCore static offsets on dependent indices are not implemented"
        )
    index_buffer = resources.buffers[index_access.site.node]
    stream = index_access.stream
    if stream is None:
        raise AssertionError("SparseCore index has no stream")
    count = program.geometry.items_per_subcore * stream.elements_per_item
    if count % SC_LANES:
        _reject(
            "layout",
            "SparseCore offset index buffer has "
            f"{count} values; it must contain complete {SC_LANES}-lane chunks",
            node=access.site.node,
        )
    return [
        f"        for _sc_ic in range({count // SC_LANES}):",
        f"            _sc_is = pl.ds(_sc_ic * {SC_LANES}, {SC_LANES})",
        (
            f"            {index_buffer}[_sc_q, _sc_is] = "
            f"{index_buffer}[_sc_q, _sc_is] + {offset}"
        ),
    ]


def _indirect_copy_lines(
    program: SparseCoreProgram,
    access: IndirectLoadAccess,
    resources: CodegenResources,
    codegen: GenerateAST,
    method: str,
) -> list[str]:
    index_access = _index_access(program, access)
    index_buffer = resources.buffers[index_access.site.node]
    destination = resources.buffers[access.site.node]
    table = _tensor_name(codegen, access.site.tensor)
    semaphore = resources.semaphore
    semaphore_index = resources.semaphore_index[access.site.node]
    prefix = (
        _offset_lines(program, access, index_access, resources)
        if method == "start"
        else []
    )
    if not isinstance(index_access, IndirectLoadAccess):
        expression = (
            f"pltpu.make_async_copy({table}.at[{index_buffer}.at[_sc_q]], "
            f"{destination}.at[_sc_q], "
            f"{semaphore}.at[_sc_q, {semaphore_index}]).{method}()"
        )
        return [*prefix, f"        {expression}"]

    parent_entries = index_access.stream.elements_per_item if index_access.stream else 1
    index_size = index_access.layout.value_size // parent_entries
    index_items = program.geometry.items_per_subcore * parent_entries
    return [
        *prefix,
        f"        for _sc_h in range({index_items}):",
        (
            f"            pltpu.make_async_copy({table}.at["
            f"{index_buffer}.at[_sc_q, _sc_h]], "
            f"{destination}.at[_sc_q, pl.ds(_sc_h * {index_size}, {index_size})], "
            f"{semaphore}.at[_sc_q, {semaphore_index}]).{method}()"
        ),
    ]


def _cached_lines(
    access: CachedLoadAccess,
    resources: CodegenResources,
    codegen: GenerateAST,
) -> list[str]:
    source = _tensor_name(codegen, access.site.tensor)
    destination = resources.buffers[access.site.node]
    return [f"        pltpu.sync_copy({source}, {destination})"]


def _atomic_initialize_lines(
    access: AtomicAddAccess, resources: CodegenResources
) -> list[str]:
    accumulator = resources.atomic_buffers[access.site.node]
    zero = resources.atomic_zero_buffers[access.site.node]
    shared_items = shared_acc_items(int(access.site.tensor.shape[0]))
    chunks_per_subcore = shared_items // SC_SHARED_ITEM_CHUNK
    lines = [
        "        _sc_sub = lax.axis_index('_sc_sub_axis')",
        f"        for _sc_zr in range({SC_LANES}):",
    ]
    for start in range(0, access.layout.value_size, SC_LANES):
        lines.append(
            f"            {zero}[_sc_zr, pl.ds({start}, {SC_LANES})] = "
            f"jnp.zeros(({SC_LANES},), jnp.float32)"
        )
    lines.extend(
        (
            f"        for _sc_zj in range({chunks_per_subcore}):",
            (
                f"            pltpu.sync_copy({zero}, {accumulator}.at[pl.ds("
                f"(_sc_sub * {chunks_per_subcore} + _sc_zj) * {SC_LANES}, "
                f"{SC_LANES})])"
            ),
            "        plsc.subcore_barrier()",
        )
    )
    return lines


def _atomic_finalize_lines(
    access: AtomicAddAccess,
    resources: CodegenResources,
    codegen: GenerateAST,
) -> list[str]:
    accumulator = resources.atomic_buffers[access.site.node]
    staging = resources.atomic_zero_buffers[access.site.node]
    destination = _tensor_name(codegen, access.site.tensor)
    shared_items = shared_acc_items(int(access.site.tensor.shape[0]))
    chunks_per_subcore = shared_items // SC_SHARED_ITEM_CHUNK
    return [
        "        plsc.subcore_barrier()",
        "        _sc_sub = lax.axis_index('_sc_sub_axis')",
        "        _sc_cid = lax.axis_index('_sc_core_axis')",
        f"        for _sc_ej in range({chunks_per_subcore}):",
        (
            f"            pltpu.sync_copy({accumulator}.at[pl.ds("
            f"(_sc_sub * {chunks_per_subcore} + _sc_ej) * {SC_LANES}, "
            f"{SC_LANES})], {staging})"
        ),
        (
            f"            pltpu.sync_copy({staging}, {destination}.at["
            f"_sc_cid, pl.ds((_sc_sub * {chunks_per_subcore} + _sc_ej) * "
            f"{SC_LANES}, {SC_LANES})])"
        ),
    ]


def _store_lines(
    program: SparseCoreProgram,
    access: SparseCoreAccess,
    resources: CodegenResources,
    codegen: GenerateAST,
) -> list[str]:
    source = resources.output_buffers[access.site.node]
    destination = _tensor_name(codegen, access.site.tensor)
    items = program.geometry.items_per_subcore
    if isinstance(access, DirectStoreAccess):
        return [
            (
                f"        pltpu.sync_copy({source}, "
                f"{destination}.at[pl.ds(_sc_o0, {items})])"
            )
        ]
    if isinstance(access, AtomicAddAccess):
        index_access = _index_access(program, access)
        if isinstance(index_access, IndirectLoadAccess):
            raise NotImplementedError(
                "SparseCore dependent indices for atomic stores are not implemented"
            )
        index_buffer = resources.buffers[index_access.site.node]
        accumulator = resources.atomic_buffers[access.site.node]
        if items == SC_LANES:
            return [
                (
                    f"        pltpu.sync_copy({source}, {accumulator}.at["
                    f"{index_buffer}.at[_sc_q, pl.ds(0, {SC_LANES})]], add=True)"
                )
            ]
        return [
            f"        for _sc_ac in range({items // SC_LANES}):",
            f"            _sc_as = pl.ds(_sc_ac * {SC_LANES}, {SC_LANES})",
            (
                f"            pltpu.sync_copy({source}.at[_sc_as], "
                f"{accumulator}.at[{index_buffer}.at[_sc_q, _sc_as]], add=True)"
            ),
        ]
    if not isinstance(access, IndirectStoreAccess):
        raise AssertionError(f"unexpected SparseCore store {type(access).__name__}")
    index_access = _index_access(program, access)
    if isinstance(index_access, IndirectLoadAccess):
        raise NotImplementedError(
            "SparseCore dependent indices for indirect stores are not implemented"
        )
    index_buffer = resources.buffers[index_access.site.node]
    return [
        f"        pltpu.sync_copy({source}, {destination}.at[{index_buffer}.at[_sc_q]])"
    ]


def _stage_guard(stage: ScheduleStage, program: SparseCoreProgram) -> str:
    if stage.phase is SchedulePhase.INITIALIZE:
        return "_sc_blk == 0"
    if stage.phase is SchedulePhase.FINALIZE:
        assert program.schedule is not None
        return f"_sc_blk == {program.geometry.blocks + program.schedule.depth - 2}"
    conditions = []
    if stage.lag:
        conditions.append(f"_sc_blk >= {stage.lag}")
    block = f"_sc_blk - {stage.lag}" if stage.lag else "_sc_blk"
    conditions.append(f"{block} < {program.geometry.blocks}")
    return " & ".join(f"({condition})" for condition in conditions)


def render_sparsecore_program(
    program: SparseCoreProgram, codegen: GenerateAST
) -> list[str]:
    assert program.schedule is not None
    schedule = program.schedule
    resources = allocate_resources(program, codegen)
    grid = (
        program.geometry.blocks + schedule.depth - 1,
        program.geometry.subcores,
    )
    lines = [
        (
            "@functools.partial(pltpu.emit_pipeline, "
            f"grid=({grid[0]}, {grid[1]}), "
            'core_axis_name=("_sc_core_axis", "_sc_sub_axis"), '
            "dimension_semantics=(pltpu.ARBITRARY, pltpu.PARALLEL))"
        ),
        "def _sc_pipeline():",
        "    _sc_blk = pl.program_id(0)",
        "    _sc_core = pl.program_id(1)",
        "",
    ]
    compute = SparseCoreCompute(
        codegen, program, resources.buffers, resources.output_buffers
    )
    for index, stage in enumerate(schedule.stages):
        lines.extend(
            (
                f"    @pl.when({_stage_guard(stage, program)})",
                f"    def _sc_stage{index}():",
            )
        )
        if stage.phase is SchedulePhase.STEADY:
            block = f"_sc_blk - {stage.lag}" if stage.lag else "_sc_blk"
            lines.extend(
                (
                    f"        _sc_t = {block}",
                    f"        _sc_q = lax.rem(_sc_t, {schedule.depth})",
                    (
                        "        _sc_o0 = (_sc_t * "
                        f"{program.geometry.subcores} + _sc_core) * "
                        f"{program.geometry.items_per_subcore}"
                    ),
                )
            )

        stores = []
        for task in stage.tasks:
            access = program.access_by_node.get(task.node)
            if task.kind is TaskKind.ONCE_TRANSFER:
                assert isinstance(access, CachedLoadAccess)
                lines.extend(_cached_lines(access, resources, codegen))
            elif task.kind is TaskKind.INITIALIZE:
                assert isinstance(access, AtomicAddAccess)
                lines.extend(_atomic_initialize_lines(access, resources))
            elif task.kind is TaskKind.SYNC_TRANSFER:
                assert isinstance(access, DirectLoadAccess)
                lines.extend(
                    _direct_copy_lines(program, access, resources, codegen, None)
                )
            elif task.kind in (TaskKind.ASYNC_START, TaskKind.ASYNC_WAIT):
                method = "start" if task.kind is TaskKind.ASYNC_START else "wait"
                if isinstance(access, IndirectLoadAccess):
                    lines.extend(
                        _indirect_copy_lines(
                            program, access, resources, codegen, method
                        )
                    )
                elif isinstance(access, DirectLoadAccess):
                    lines.extend(
                        _direct_copy_lines(program, access, resources, codegen, method)
                    )
                else:
                    raise AssertionError(
                        f"unexpected async SparseCore access {type(access).__name__}"
                    )
            elif task.kind is TaskKind.STORE:
                assert access is not None
                stores.append(access)
            elif task.kind is TaskKind.FINALIZE:
                assert isinstance(access, AtomicAddAccess)
                lines.extend(_atomic_finalize_lines(access, resources, codegen))
        if stores:
            lines.extend(
                (
                    f"        @plsc.parallel_loop(0, {program.geometry.items_per_subcore})",
                    "        def _sc_item_loop(_sc_item):",
                    *compute.emit_body(
                        " " * 12,
                        store_nodes={access.site.node for access in stores},
                    ),
                )
            )
            for access in stores:
                lines.extend(_store_lines(program, access, resources, codegen))
        lines.append("")
    lines.append("_sc_pipeline()")
    return lines


def codegen_sparsecore_program(
    program: SparseCoreProgram, codegen: GenerateAST
) -> None:
    from ..ast_extension import convert
    from ..device_function import PallasMemorySpace

    for access in program.accesses:
        codegen.device_function.pallas_memory_space[id(access.site.tensor)] = (
            PallasMemorySpace.HBM
        )
    module = ast.parse("\n".join(render_sparsecore_program(program, codegen)))
    for statement in module.body:
        codegen.device_function.body.append(convert(statement))  # type: ignore[arg-type]
