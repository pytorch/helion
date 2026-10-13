"""Emit original typed expressions inside a late-proved warp-owned region.

Graph support, final shared lifetimes and role ownership belong to the binder.
This emitter retains every typed SSA version and original contraction orientation;
only raw fragment transport is selected here. No complete workload is encoded.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from ..compile_environment import CompileEnvironment
from .chained_matmul import _Expression
from .chained_matmul import _indent
from .chained_matmul import _materialized_value
from .chained_matmul import _operand_domain
from .chained_matmul import _UnsupportedChain
from .chained_register_fragments import plan_warp_fragment_map
from .chained_register_transport import emit_fragment_transport

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_island_publication import IslandConsumerPublication
    from .chained_matmul import ChainedMatmulPlan
    from .chained_register_binding import BoundPreparationIsland
    from .chained_register_fragments import FragmentRole
    from .chained_register_islands import RegisterImage


def emit_register_island(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    bound: BoundPreparationIsland,
    boundaries: dict[Node, str],
    *,
    publication: IslandConsumerPublication | None = None,
    body_first: int | None = None,
) -> list[str] | None:
    """Publish exports only after complete emission succeeds; otherwise decline.

    Caller replaces exactly ``[first_event, stop_event)`` of the bound frame.
    Entry-producing actions must already have completed their role barriers.
    Every execution participant reaches both emitted publication barriers, even
    when only a subset of the role's complete warps owns register components.
    """
    execution, island = bound.execution, bound.island
    island_input = bound.island_input
    if not bound.matches(plan, bound.frame, execution, boundaries):
        return None
    if publication is not None and any(
        value.additional_images for value in island.values
    ):
        return None
    if island_input is not None and (
        publication is not None or island_input.consumed or body_first is None
    ):
        return None
    backend = CompileEnvironment.current().backend
    issues = {issue.node: issue for issue in island.components[0].issues}
    values = {value.node: value for value in island.values}
    # Verify the precise origin of each operand image, not merely equal shapes.
    try:
        for issue in issues.values():
            bound.operand_image(issue.stage, 0)
            bound.operand_image(issue.stage, 1)
            value = values[issue.node]
            if value.additional_images or value.tiles != tuple(
                (index, current.origins[:2])
                for index, component in enumerate(island.components)
                for current in component.issues
                if current.stage == issue.stage
            ):
                return None
    except _UnsupportedChain:
        return None
    prefix = cg.device_function.new_var("chain_register_island")
    types = tuple(
        dict.fromkeys(plan.operand_dtype(issue.stage) for issue in issues.values())
    )
    mmas = {dtype: f"{prefix}_mma_{index}" for index, dtype in enumerate(types)}
    body = [
        f"{prefix}_lane = {execution.thread} % 32",
        f"{prefix}_warp = {execution.thread} // 32",
    ]
    for dtype, mma in mmas.items():
        body.extend(
            [
                f"{mma} = cute.make_tiled_mma(cute.make_mma_atom(cute.nvgpu.warp.MmaF16BF16Op({backend.dtype_str(dtype)}, cutlass.Float32, (16, 8, 16))), atom_layout_mnk=(1, 1, 1))",
                f"{mma}_thread = {mma}.get_slice({prefix}_lane)",
                f"{mma}_a_coords = {mma}_thread.partition_A(cute.make_identity_tensor((16, 16)))",
                f"{mma}_b_coords = {mma}_thread.partition_B(cute.make_identity_tensor((16, 16)))",
            ]
        )
    body.append(
        f"{prefix}_coords = {next(iter(mmas.values()))}_thread.partition_C(cute.make_identity_tensor((16, 16)))"
    )
    allocations = body if island_input is not None else []
    if island_input is not None:
        body = []
    segments: list[str] = []
    local_boundaries = dict(boundaries)
    alias_first = -1
    alias_inputs: tuple[tuple[Node, str], ...] = ()
    alias_outputs: tuple[tuple[Node, str], ...] = ()
    alias_lines: tuple[str, ...] = ()
    operand_uses: list[tuple[int, str, RegisterImage]] = []
    input_reads: list[tuple[int, str, RegisterImage, str]] = []
    issue_bindings: list[tuple[int, str, str]] = []

    def flush_active() -> None:
        if body:
            segments.extend(
                [
                    f"if {execution.thread} < {len(island.components) * 32}:",
                    _indent(body),
                ]
            )
            body.clear()

    fragments: list[tuple[Node, tuple[str, ...], str]] = []
    registers: dict[RegisterImage, str] = {}
    try:
        for index, (value, image) in enumerate(
            (item, image) for item in island.values for image in item.images
        ):
            node = value.node
            name = f"{prefix}_value_{index}"
            coordinates = bound.coordinates(
                image if value.additional_images else node, prefix
            )
            if _operand_domain(cg, node, coordinates, plan):
                return None
            if node in issues:
                issue = issues[node]
                dtype = plan.operand_dtype(issue.stage)
                mma = mmas[dtype]
                dot = f"{prefix}_dot_{index}"
                issue_bindings.append((issue.stage, mma, dot))
                (allocations if island_input is not None else body).extend(
                    [
                        f"{dot}_a = {mma}.make_fragment_A({mma}_a_coords.shape)",
                        f"{dot}_b = {mma}.make_fragment_B({mma}_b_coords.shape)",
                        f"{dot}_acc = cute.make_rmem_tensor({mma}.partition_shape_C((16, 16)), cutlass.Float32)",
                    ]
                )
                body.append(f"{dot}_acc.fill(0.0)")
                lhs, rhs = (
                    bound.operand_image(issue.stage, 0),
                    bound.operand_image(issue.stage, 1),
                )
                operands = (rhs, lhs) if issue.geometry.transpose else (lhs, rhs)
                roles: tuple[FragmentRole, FragmentRole] = ("a", "b")
                for role, operand in zip(roles, operands, strict=True):
                    mapping = plan_warp_fragment_map(
                        role, dtype, transpose=issue.geometry.transpose
                    )
                    if (
                        operand not in registers
                        and island_input is not None
                        and operand.node is island_input.candidate.operand
                    ):
                        entry = f"{dot}_{role}_entry"
                        allocations.append(
                            f"{entry} = cute.make_rmem_tensor((8,), {backend.dtype_str(island_input.candidate.dtype)})"
                        )
                        expression = _Expression(cg, plan, local_boundaries)
                        result = expression.value(
                            operand.node, bound.coordinates(operand, prefix)
                        )
                        if (
                            expression.global_accesses
                            or expression.lines
                            or result
                            != _materialized_value(
                                local_boundaries[operand.node],
                                island_input.candidate.shape,
                                bound.coordinates(operand, prefix),
                                backend.dtype_str(island_input.candidate.dtype),
                            )
                        ):
                            return None
                        input_reads.append((issue.stage, role, operand, entry))
                        body.extend(
                            [
                                f"for {prefix}_index in cutlass.range_constexpr(8):",
                                _indent(
                                    [
                                        *expression.lines,
                                        f"{entry}[{prefix}_index] = {result}",
                                    ]
                                ),
                            ]
                        )
                        registers[operand] = entry
                    if mapping is None or operand not in registers:
                        return None
                    transport = emit_fragment_transport(
                        mapping,
                        registers[operand],
                        f"{dot}_{role}",
                        prefix=f"{dot}_{role}_copy",
                        lane=f"{prefix}_lane",
                    )
                    if transport is None:
                        return None
                    body.extend(transport)
                    operand_uses.append((issue.stage, role, operand))
                if island_input is not None:
                    flush_active()
                    segments.extend(
                        ["cute.arch.fence_view_async_shared()", execution.sync]
                    )
                body.append(
                    f"cute.gemm({mma}, {dot}_acc, {dot}_a[None, None, 0], {dot}_b[None, None, 0], {dot}_acc)"
                )
                if issue.geometry.transpose:
                    mapping = plan_warp_fragment_map("c", torch.float32, transpose=True)
                    assert mapping is not None
                    transport = emit_fragment_transport(
                        mapping,
                        f"{dot}_acc",
                        name,
                        prefix=f"{dot}_c_copy",
                        lane=f"{prefix}_lane",
                    )
                    if transport is None:
                        return None
                    (allocations if island_input is not None else body).append(
                        f"{name} = cute.make_rmem_tensor((8,), cutlass.Float32)"
                    )
                    body.extend(transport)
                else:
                    # Native C already has canonical ownership; preserve its
                    # FP32 value rather than introducing a needless conversion.
                    (allocations if island_input is not None else body).append(
                        f"{name} = {dot}_acc"
                    )
                if (
                    island_input is not None
                    and issue.stage == island_input.candidate.consumer
                ):
                    flush_active()
                    segments.append(execution.sync)
                    retained = island_input.candidate.retained_input
                    assert retained is not None
                    alias_inputs = tuple(local_boundaries.items())
                    alias_lines = tuple(
                        retained[1].alias_lines(
                            island_input.candidate.target,
                            island_input.boundary_name(bound.first_event + 2),
                        )
                    )
                    alias_first = len(segments)
                    segments.extend(alias_lines)
                    local_boundaries[island_input.candidate.operand] = (
                        island_input.boundary_name(bound.first_event + 2)
                    )
                    alias_outputs = tuple(local_boundaries.items())
            else:
                expression = _Expression(cg, plan, dict(local_boundaries))
                for source, coords, fragment in fragments:
                    if source is not node:
                        expression.bind_fragment_image(source, coords, fragment)
                result = expression.value(node, coordinates)
                if expression.global_accesses:
                    return None
                (allocations if island_input is not None else body).append(
                    f"{name} = cute.make_rmem_tensor((8,), {backend.dtype_str(value.dtype)})"
                )
                body.extend(
                    [
                        f"for {prefix}_index in cutlass.range_constexpr(8):",
                        _indent(
                            [*expression.lines, f"{name}[{prefix}_index] = {result}"]
                        ),
                    ]
                )
            registers[image] = name
            fragments.append((node, coordinates, f"{name}[{prefix}_index]"))
    except _UnsupportedChain:
        return None
    if publication is not None:
        lines, stores = publication.candidate.publication(
            prefix, {node: registers[value.images[0]] for node, value in values.items()}
        )
        body.extend(stores)
    else:
        for export in bound.exports:
            row, column = bound.coordinates(export.node, prefix)
            body.extend(
                [
                    f"for {prefix}_index in cutlass.range_constexpr(8):",
                    f"    {export.name}[{row}, {column}] = {registers[values[export.node].images[0]]}[{prefix}_index]",
                ]
            )
        lines = []
    shapes = tuple(dict.fromkeys(export.shape for export in bound.exports))
    for number, shape in enumerate(shapes if publication is None else ()):
        size = shape[0] * shape[1]
        zero = f"{prefix}_zero_{number}"
        writes = [
            f"{export.name}[{zero} // {shape[1]}, {zero} % {shape[1]}] = cutlass.Float32(0)"
            for export in bound.exports
            if export.shape == shape
        ]
        if size % execution.threads:
            writes = [f"if {zero} < {size}:", _indent(writes)]
        lines.extend(
            [
                f"for {zero}_step in cutlass.range({(size + execution.threads - 1) // execution.threads}, unroll=1):",
                f"    {zero} = {execution.thread} + {zero}_step * {execution.threads}",
                _indent(writes),
            ]
        )
    if island_input is None:
        lines.extend(
            [
                execution.sync,
                f"if {execution.thread} < {len(island.components) * 32}:",
                _indent(body),
                execution.sync,
            ]
        )
    else:
        # Storage and fragment objects dominate every active-warp branch.
        # The original full-role stage join and retained alias stay between
        # the two issue spans, not inside a warp predicate or after the island.
        flush_active()
        segments.append(execution.sync)
        alias_first += len(lines) + len(allocations)
        lines.extend([*allocations, *segments])
    if publication is None:
        boundaries.update((export.node, export.name) for export in bound.exports)
    else:
        # This is the real original half operand and its complete native view,
        # published only after the unchanged full-role join. No C boundary is
        # fabricated for an omitted export.
        publication.record(lines, boundaries)
    if island_input is not None:
        from .chained_island_publication import CompletedInputIsland
        from .chained_island_publication import _context

        assert body_first is not None
        boundaries[island_input.candidate.operand] = local_boundaries[
            island_input.candidate.operand
        ]
        completion = CompletedInputIsland(
            publication=island_input,
            bound=bound,
            outputs=tuple(boundaries.items()),
            body_first=body_first,
            lines=tuple(lines),
            alias_first=alias_first,
            alias_inputs=alias_inputs,
            alias_outputs=alias_outputs,
            alias_lines=alias_lines,
            context=_context(cg, plan, bound, island_input.candidate.stage),
            prefix=prefix,
            operand_uses=tuple(operand_uses),
            input_reads=tuple(input_reads),
            issue_bindings=tuple(issue_bindings),
            dtype=backend.dtype_str(island_input.candidate.dtype),
        )
        completion = replace(completion, _selection=completion.fields())
        island_input.record_island(completion, boundaries)
    from .native_matmul_metadata import record_native_stage

    record_native_stage(cg, plan, tuple(issues), "register_island", lines)
    return lines
