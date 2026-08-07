"""Render dependency-scheduled SparseCore programs through emit_pipeline."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .memory_access import MemoryAccessKind
from .sc_base import SC_LANES
from .sc_base import _reject
from .sparsecore_compute import SparseCoreCompute
from .sparsecore_plan import DirectLoadPlan
from .sparsecore_plan import DirectStorePlan
from .sparsecore_plan import IndirectLoadPlan
from .sparsecore_plan import IndirectStorePlan
from .sparsecore_plan import SparseCoreMemoryPlan
from .sparsecore_program import ScheduleStage
from .sparsecore_program import SparseCoreProgram
from .sparsecore_program import TaskKind
from .sparsecore_program import load_buffer_shape

if TYPE_CHECKING:
    import torch

    from ..generate_ast import GenerateAST


@dataclass
class CodegenResources:
    buffers: dict[torch.fx.Node, str]
    output_buffers: dict[torch.fx.Node, str]
    semaphore: str
    semaphore_index: dict[torch.fx.Node, int]


def allocate_resources(
    program: SparseCoreProgram, codegen: GenerateAST
) -> CodegenResources:
    device_fn = codegen.device_function
    assert program.schedule is not None
    depth = program.schedule.depth
    items = program.items_per_subcore
    buffers: dict[torch.fx.Node, str] = {}
    output_buffers: dict[torch.fx.Node, str] = {}
    semaphore_index: dict[torch.fx.Node, int] = {}

    for plan in program.loads:
        node = plan.access.node
        if isinstance(plan, IndirectLoadPlan) or node not in program.index_nodes:
            semaphore_index[node] = len(semaphore_index)
        buffers[node] = device_fn.register_scratch(
            (depth, *load_buffer_shape(program, plan)),
            plan.layout.storage_dtype,
            "sc_value",
        )

    for plan in program.stores:
        value_size = plan.layout.value_size
        shape = (items, SC_LANES if value_size == 1 else value_size)
        output_buffers[plan.access.node] = device_fn.register_scratch(
            shape, plan.layout.storage_dtype, "sc_output"
        )

    semaphore = device_fn.register_dma_semaphore(
        "sc_dma", (depth, max(len(semaphore_index), 1))
    )
    return CodegenResources(
        buffers,
        output_buffers,
        semaphore,
        semaphore_index,
    )


def _tensor_name(codegen: GenerateAST, tensor: torch.Tensor) -> str:
    return codegen.device_function.tensor_arg(tensor).name


def _direct_copy_lines(
    program: SparseCoreProgram,
    plan: DirectLoadPlan,
    resources: CodegenResources,
    codegen: GenerateAST,
    method: str | None,
) -> list[str]:
    stream = plan.stream
    assert stream is not None
    source = _tensor_name(codegen, plan.access.tensor)
    destination = resources.buffers[plan.access.node]
    items = program.items_per_subcore
    if plan.access.node in program.index_nodes:
        if method is not None:
            raise AssertionError("SparseCore index streams must be synchronous")
        count = items * stream.elements_per_item
        base = stream.group * program.padded_items * stream.elements_per_item
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
    base = stream.group * program.padded_items
    offset = f"{base} + _sc_o0" if base else "_sc_o0"
    if method is not None:
        semaphore_index = resources.semaphore_index[plan.access.node]
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


def _index_plan(
    program: SparseCoreProgram,
    plan: IndirectLoadPlan | IndirectStorePlan,
) -> SparseCoreMemoryPlan:
    index_plan = program.plan_by_node.get(plan.index_node)
    if index_plan is None or index_plan.access.kind is not MemoryAccessKind.LOAD:
        raise NotImplementedError(
            "SparseCore indirect index must be produced by a lowered load"
        )
    return index_plan


def _offset_lines(
    program: SparseCoreProgram,
    plan: IndirectLoadPlan,
    index_plan: SparseCoreMemoryPlan,
    resources: CodegenResources,
) -> list[str]:
    offset = plan.index_offset
    if not offset:
        return []
    if isinstance(index_plan, IndirectLoadPlan):
        raise NotImplementedError(
            "SparseCore static offsets on dependent indices are not implemented"
        )
    index_buffer = resources.buffers[index_plan.access.node]
    stream = index_plan.stream
    if stream is None:
        raise AssertionError("SparseCore index has no stream")
    count = program.items_per_subcore * stream.elements_per_item
    if count % SC_LANES:
        _reject(
            "layout",
            "SparseCore offset index buffer has "
            f"{count} values; it must contain complete {SC_LANES}-lane chunks",
            node=plan.access.node,
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
    plan: IndirectLoadPlan,
    resources: CodegenResources,
    codegen: GenerateAST,
    method: str,
) -> list[str]:
    index_plan = _index_plan(program, plan)
    index_buffer = resources.buffers[index_plan.access.node]
    destination = resources.buffers[plan.access.node]
    table = _tensor_name(codegen, plan.access.tensor)
    semaphore = resources.semaphore
    semaphore_index = resources.semaphore_index[plan.access.node]
    prefix = (
        _offset_lines(program, plan, index_plan, resources) if method == "start" else []
    )
    if not isinstance(index_plan, IndirectLoadPlan):
        expression = (
            f"pltpu.make_async_copy({table}.at[{index_buffer}.at[_sc_q]], "
            f"{destination}.at[_sc_q], "
            f"{semaphore}.at[_sc_q, {semaphore_index}]).{method}()"
        )
        return [*prefix, f"        {expression}"]

    parent_entries = index_plan.stream.elements_per_item if index_plan.stream else 1
    index_size = index_plan.layout.value_size // parent_entries
    index_items = program.items_per_subcore * parent_entries
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


def _store_lines(
    program: SparseCoreProgram,
    plan: SparseCoreMemoryPlan,
    resources: CodegenResources,
    codegen: GenerateAST,
) -> list[str]:
    source = resources.output_buffers[plan.access.node]
    destination = _tensor_name(codegen, plan.access.tensor)
    items = program.items_per_subcore
    if isinstance(plan, DirectStorePlan):
        return [
            (
                f"        pltpu.sync_copy({source}, "
                f"{destination}.at[pl.ds(_sc_o0, {items})])"
            )
        ]
    if not isinstance(plan, IndirectStorePlan):
        raise AssertionError(f"unexpected SparseCore store {type(plan).__name__}")
    index_plan = _index_plan(program, plan)
    if isinstance(index_plan, IndirectLoadPlan):
        raise NotImplementedError(
            "SparseCore dependent indices for indirect stores are not implemented"
        )
    index_buffer = resources.buffers[index_plan.access.node]
    return [
        f"        pltpu.sync_copy({source}, {destination}.at[{index_buffer}.at[_sc_q]])"
    ]


def _stage_guard(stage: ScheduleStage, program: SparseCoreProgram) -> str:
    conditions = []
    if stage.lag:
        conditions.append(f"_sc_blk >= {stage.lag}")
    block = f"_sc_blk - {stage.lag}" if stage.lag else "_sc_blk"
    conditions.append(f"{block} < {program.blocks}")
    return " & ".join(f"({condition})" for condition in conditions)


def render_sparsecore_program(
    program: SparseCoreProgram, codegen: GenerateAST
) -> list[str]:
    assert program.schedule is not None
    schedule = program.schedule
    resources = allocate_resources(program, codegen)
    grid = (
        program.blocks + schedule.depth - 1,
        program.total_subcores,
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
        block = f"_sc_blk - {stage.lag}" if stage.lag else "_sc_blk"
        lines.extend(
            (
                f"        _sc_t = {block}",
                f"        _sc_q = lax.rem(_sc_t, {schedule.depth})",
                (
                    "        _sc_o0 = (_sc_t * "
                    f"{program.total_subcores} + _sc_core) * "
                    f"{program.items_per_subcore}"
                ),
            )
        )

        stores = []
        for task in stage.tasks:
            plan = program.plan_by_node.get(task.node)
            if task.kind is TaskKind.SYNC_TRANSFER:
                assert isinstance(plan, DirectLoadPlan)
                lines.extend(
                    _direct_copy_lines(program, plan, resources, codegen, None)
                )
            elif task.kind in (TaskKind.ASYNC_START, TaskKind.ASYNC_WAIT):
                method = "start" if task.kind is TaskKind.ASYNC_START else "wait"
                if isinstance(plan, IndirectLoadPlan):
                    lines.extend(
                        _indirect_copy_lines(program, plan, resources, codegen, method)
                    )
                elif isinstance(plan, DirectLoadPlan):
                    lines.extend(
                        _direct_copy_lines(program, plan, resources, codegen, method)
                    )
                else:
                    raise AssertionError(
                        f"unexpected async SparseCore plan {type(plan).__name__}"
                    )
            elif task.kind is TaskKind.STORE:
                assert plan is not None
                stores.append(plan)
        if stores:
            lines.extend(
                (
                    f"        @plsc.parallel_loop(0, {program.items_per_subcore})",
                    "        def _sc_item_loop(_sc_item):",
                    *compute.emit_body(
                        " " * 12,
                        store_nodes={plan.access.node for plan in stores},
                    ),
                )
            )
            for plan in stores:
                lines.extend(_store_lines(program, plan, resources, codegen))
        lines.append("")
    lines.append("_sc_pipeline()")
    return lines


def codegen_sparsecore_program(
    program: SparseCoreProgram, codegen: GenerateAST
) -> None:
    from ..ast_extension import convert
    from ..device_function import PallasMemorySpace

    for plan in program.memory_plans:
        codegen.device_function.pallas_memory_space[id(plan.access.tensor)] = (
            PallasMemorySpace.HBM
        )
    module = ast.parse("\n".join(render_sparsecore_program(program, codegen)))
    for statement in module.body:
        codegen.device_function.body.append(convert(statement))  # type: ignore[arg-type]
