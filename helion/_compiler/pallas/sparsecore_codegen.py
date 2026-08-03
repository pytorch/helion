"""Render dependency-scheduled SparseCore programs through emit_pipeline."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

from .sparsecore_compute import SparseCoreCompute
from .sparsecore_plan import DirectLoadPlan
from .sparsecore_plan import DirectStorePlan
from .sparsecore_plan import IndirectLoadPlan
from .sparsecore_plan import IndirectStorePlan
from .sparsecore_program import ScheduleStage
from .sparsecore_program import SparseCoreProgram
from .sparsecore_program import TaskKind
from .sparsecore_program import load_buffer_shape

if TYPE_CHECKING:
    import torch

    from ..generate_ast import GenerateAST


@dataclass(frozen=True)
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
        output_buffers[plan.access.node] = device_fn.register_scratch(
            plan.layout.storage_shape, plan.layout.storage_dtype, "sc_output"
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
    transfer = plan.transfer
    source = _tensor_name(codegen, plan.access.tensor)
    destination = resources.buffers[plan.access.node]
    items = program.items_per_subcore
    if plan.access.node in program.index_nodes:
        count = items * transfer.elements_per_item
        base = transfer.prefix_index * program.padded_items * transfer.elements_per_item
        offset = (
            f"{base} + _sc_item_offset * {transfer.elements_per_item}"
            if base
            else (
                "_sc_item_offset"
                if transfer.elements_per_item == 1
                else f"_sc_item_offset * {transfer.elements_per_item}"
            )
        )
        return [
            (
                f"        pltpu.sync_copy({source}.at[pl.ds({offset}, {count})], "
                f"{destination}.at[_sc_ring_slot, pl.ds(0, {count})])"
            )
        ]
    base = transfer.prefix_index * program.padded_items
    offset = f"{base} + _sc_item_offset" if base else "_sc_item_offset"
    if method is not None:
        semaphore_index = resources.semaphore_index[plan.access.node]
        return [
            (
                f"        pltpu.make_async_copy({source}.at[pl.ds({offset}, {items})], "
                f"{destination}.at[_sc_ring_slot], {resources.semaphore}.at["
                f"_sc_ring_slot, {semaphore_index}]).{method}()"
            )
        ]
    return [
        (
            f"        pltpu.sync_copy({source}.at[pl.ds({offset}, {items})], "
            f"{destination}.at[_sc_ring_slot])"
        )
    ]


def _index_plan(
    program: SparseCoreProgram,
    plan: IndirectLoadPlan | IndirectStorePlan,
) -> DirectLoadPlan | IndirectLoadPlan:
    return cast(
        "DirectLoadPlan | IndirectLoadPlan", program.plan_by_node[plan.index_node]
    )


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
    if not isinstance(index_plan, IndirectLoadPlan):
        expression = (
            f"pltpu.make_async_copy({table}.at[{index_buffer}.at[_sc_ring_slot]], "
            f"{destination}.at[_sc_ring_slot], "
            f"{semaphore}.at[_sc_ring_slot, {semaphore_index}]).{method}()"
        )
        return [f"        {expression}"]

    parent_entries = index_plan.transfer.elements_per_item
    index_size = index_plan.layout.elements_per_item // parent_entries
    index_items = program.items_per_subcore * parent_entries
    return [
        f"        for _sc_gather_item in range({index_items}):",
        (
            f"            pltpu.make_async_copy({table}.at["
            f"{index_buffer}.at[_sc_ring_slot, _sc_gather_item]], "
            f"{destination}.at[_sc_ring_slot, "
            f"pl.ds(_sc_gather_item * {index_size}, {index_size})], "
            f"{semaphore}.at[_sc_ring_slot, {semaphore_index}]).{method}()"
        ),
    ]


def _store_lines(
    program: SparseCoreProgram,
    plan: DirectStorePlan | IndirectStorePlan,
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
                f"{destination}.at[pl.ds(_sc_item_offset, {items})])"
            )
        ]
    index_plan = _index_plan(program, plan)
    index_buffer = resources.buffers[index_plan.access.node]
    return [
        (
            f"        pltpu.sync_copy({source}, "
            f"{destination}.at[{index_buffer}.at[_sc_ring_slot]])"
        )
    ]


def _stage_guard(stage: ScheduleStage, program: SparseCoreProgram) -> str:
    conditions = []
    if stage.lag:
        conditions.append(f"_sc_pipeline_step >= {stage.lag}")
    block = f"_sc_pipeline_step - {stage.lag}" if stage.lag else "_sc_pipeline_step"
    conditions.append(f"{block} < {program.item_blocks}")
    return " & ".join(f"({condition})" for condition in conditions)


def render_sparsecore_program(
    program: SparseCoreProgram, codegen: GenerateAST
) -> list[str]:
    assert program.schedule is not None
    schedule = program.schedule
    resources = allocate_resources(program, codegen)
    grid = (
        program.item_blocks + schedule.depth - 1,
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
        "    _sc_pipeline_step = pl.program_id(0)",
        "    _sc_subcore_id = pl.program_id(1)",
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
        block = f"_sc_pipeline_step - {stage.lag}" if stage.lag else "_sc_pipeline_step"
        lines.extend(
            (
                f"        _sc_item_block = {block}",
                (f"        _sc_ring_slot = lax.rem(_sc_item_block, {schedule.depth})"),
                (
                    "        _sc_item_offset = (_sc_item_block * "
                    f"{program.total_subcores} + _sc_subcore_id) * "
                    f"{program.items_per_subcore}"
                ),
            )
        )

        stores: list[DirectStorePlan | IndirectStorePlan] = []
        for task in stage.tasks:
            plan = program.plan_by_node.get(task.node)
            if task.kind is TaskKind.SYNC_TRANSFER:
                lines.extend(
                    _direct_copy_lines(
                        program,
                        cast("DirectLoadPlan", plan),
                        resources,
                        codegen,
                        None,
                    )
                )
            elif task.kind in (TaskKind.ASYNC_START, TaskKind.ASYNC_WAIT):
                method = "start" if task.kind is TaskKind.ASYNC_START else "wait"
                load_plan = cast("DirectLoadPlan | IndirectLoadPlan", plan)
                if isinstance(load_plan, IndirectLoadPlan):
                    lines.extend(
                        _indirect_copy_lines(
                            program, load_plan, resources, codegen, method
                        )
                    )
                else:
                    lines.extend(
                        _direct_copy_lines(
                            program, load_plan, resources, codegen, method
                        )
                    )
            elif task.kind is TaskKind.STORE:
                stores.append(cast("DirectStorePlan | IndirectStorePlan", plan))
        if stores:
            lines.extend(
                (
                    f"        @plsc.parallel_loop(0, {program.items_per_subcore})",
                    "        def _sc_item_loop(_sc_local_item):",
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
