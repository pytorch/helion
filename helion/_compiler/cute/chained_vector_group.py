"""Coordinate-owned sharing between simultaneous vector expression outputs."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language.memory_ops import _cute_scalar_pointer_expr
from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain
from .chained_execution import ChainedExecution
from .chained_vector_expression import _substitute
from .chained_vector_leaf import emit_vector_leaf
from .chained_vector_leaf import prove_vector_leaf
from .chained_vector_native import emit_native_inputs
from .chained_vector_ownership import plan_vector_ownership

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_broadcast_retention import BroadcastRetentionAttempt
    from .chained_matmul import ChainedMatmulPlan
    from .chained_native_stores import NativeStMatrixStore
    from .chained_pointwise_unroll import BoundedProducerUnroll
    from .chained_pointwise_unroll import PointwiseUnroll
    from .chained_vector_native import NativeReadInputs
    from .chained_vector_ownership import VectorOwnership


@dataclass(frozen=True)
class VectorGroupOutput:
    """An existing target and its exact source-coordinate mapping.

    Targets and their layouts remain caller-owned. Distinct target names must
    denote disjoint destination storage, and every materialized input must
    remain alive until the whole group finishes. The caller supplies exactly
    the final cast/mask callback used by its ordinary materialization path.
    ``vector_store=False`` is required for layouts that permute within-vector
    values without a separately proven compatible vector-copy partition.
    """

    node: Node
    target: str
    coordinates: Callable[[str, str], tuple[str, ...]]
    offset: int = 0
    final_value: Callable[[str, str, tuple[str, ...]], str] | None = None
    vector_store: bool = True
    native_store: NativeStMatrixStore | None = None
    ordinal: int | None = None


def _conjunction(terms: Sequence[str]) -> frozenset[str]:
    """Compare exact pure predicate atoms, without arithmetic simplification."""
    result: set[str] = set()

    def collect(node: ast.AST) -> None:
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            for value in node.values:
                collect(value)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            collect(node.left)
            collect(node.right)
        else:
            result.add(ast.dump(node))

    for term in terms:
        collect(ast.parse(term, mode="eval").body)
    return frozenset(result)


def emit_vector_group(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    outputs: Sequence[VectorGroupOutput],
    *,
    shape: tuple[int, int],
    tag: str,
    producer_unroll: PointwiseUnroll | BoundedProducerUnroll | None = None,
    execution: ChainedExecution | None = None,
    ownership: VectorOwnership | None = None,
    native_inputs: NativeReadInputs | None = None,
    broadcast: BroadcastRetentionAttempt | None = None,
) -> list[str] | None:
    """Attempt a group without installing any partial producer statements."""
    try:
        return _emit_vector_group(
            cg,
            plan,
            boundaries,
            outputs,
            shape=shape,
            tag=tag,
            producer_unroll=producer_unroll,
            execution=execution,
            ownership=ownership,
            native_inputs=native_inputs,
            broadcast=broadcast,
        )
    except chain._UnsupportedChain:
        return None


def emit_materialized_group(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    outputs: Sequence[VectorGroupOutput],
    *,
    shape: tuple[int, int],
    tag: str,
    producer_unroll: PointwiseUnroll | BoundedProducerUnroll | None = None,
    execution: ChainedExecution | None = None,
    ownership: VectorOwnership | None = None,
    native_inputs: NativeReadInputs | None = None,
    broadcast: BroadcastRetentionAttempt | None = None,
) -> list[str] | None:
    """Publish boundary-only expressions with the existing eight-value owners.

    This is an ownership schedule, not vector-load or CSE admission. Every
    output must read an existing materialized boundary and no residual host
    input. Original expression lowering, logical bounds, final casts/masks and
    target stores are unchanged. The caller proves that all source boundaries
    remain live and disjoint from every destination through publication, and
    that distinct target names denote disjoint storage. Equal target names are
    checked here, but physical aliases require the caller's frame proof.

    No vector/CSE activation is recorded. Unsupported attempts return no
    statements and do not activate producer unroll.
    """
    try:
        return _emit_vector_group(
            cg,
            plan,
            boundaries,
            outputs,
            shape=shape,
            tag=tag,
            producer_unroll=producer_unroll,
            execution=execution,
            materialized=True,
            ownership=ownership,
            native_inputs=native_inputs,
            broadcast=broadcast,
        )
    except chain._UnsupportedChain:
        return None


def _emit_vector_group(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    outputs: Sequence[VectorGroupOutput],
    *,
    shape: tuple[int, int],
    tag: str,
    producer_unroll: PointwiseUnroll | BoundedProducerUnroll | None,
    execution: ChainedExecution | None,
    materialized: bool = False,
    ownership: VectorOwnership | None = None,
    native_inputs: NativeReadInputs | None = None,
    broadcast: BroadcastRetentionAttempt | None = None,
) -> list[str] | None:
    """Share original DAG values at identical per-element coordinates.

    All outputs have the same physical producer shape and exact logical extent
    predicate/operand domain. Each keeps its own original logical coordinates,
    dtype, final callback and target copy or scalar publication. Memoization is
    only by the original FX node and its exact propagated coordinates: neither
    algebraic CSE nor cross-coordinate sharing is performed.

    Meaningful shared original-node work or a shared host vector is required.
    Materialized inputs may remove every host read without removing the
    coordinate-owned pointwise CSE; their reads remain ordinary scalar reads.
    Without a host vector, residual scalar host loads retain the old fallback.
    Host vector bounds, masking, alignment, no-wrap guards and scalar fallbacks
    are delegated unchanged to the existing leaf emitter. Unsupported groups
    return None before activating unroll or returning any emitted statements.
    As in the existing single-output probe, temporary names/tensor arguments
    may be registered while probing. No resources, barriers, or boundaries are
    created; the caller must not introduce this attempt on its default path.
    """
    if len(outputs) < 2 or any(
        item.node.meta["val"].dtype
        not in (torch.bfloat16, torch.float16, torch.float32)
        or not materialized
        and item.node in boundaries
        or item.node.graph is not plan.dots[0].graph
        or plan.loop is not None
        and item.node in plan.loop.boundaries()
        or type(item.offset) is not int
        or item.offset < 0
        for item in outputs
    ):
        return None
    ordinals = tuple(
        index if item.ordinal is None else item.ordinal
        for index, item in enumerate(outputs)
    )
    if any(type(index) is not int or index < 0 for index in ordinals) or len(
        set(ordinals)
    ) != len(ordinals):
        return None
    if (
        type(shape) is not tuple
        or len(shape) != 2
        or any(type(extent) is not int or extent <= 0 for extent in shape)
    ):
        return None
    execution = execution or ChainedExecution(plan.threads)
    height, width = shape
    ownership = ownership or plan_vector_ownership(shape, execution.threads)
    if (
        ownership is None
        or not ownership.matches(shape, execution.threads)
        or ownership.thread_order != "row_major"
    ):
        return None
    if any(
        item.native_store is not None
        and (
            item.vector_store
            or item.offset != 0
            or not item.native_store.matches()
            or item.native_store.ownership != ownership
            or item.native_store.dtype != item.node.meta["val"].dtype
        )
        for item in outputs
    ):
        return None
    for index, item in enumerate(outputs):
        for other in outputs[index + 1 :]:
            if item.target == other.target and max(item.offset, other.offset) < min(
                item.offset + height, other.offset + height
            ):
                return None
    rows, columns = ownership.thread_rows, ownership.thread_columns
    row, base, element = f"{tag}_row", f"{tag}_base", f"{tag}_element"
    coordinates = [item.coordinates(row, f"({base} + {element})") for item in outputs]
    if any(
        len(coords) != len(chain._shape(item.node))
        for item, coords in zip(outputs, coordinates, strict=True)
    ):
        return None
    predicates = [
        [
            f"(({coord}) < {extent})"
            for coord, extent in zip(coords, chain._shape(item.node), strict=True)
        ]
        for item, coords in zip(outputs, coordinates, strict=True)
    ]
    if any(
        _conjunction(predicate) != _conjunction(predicates[0])
        for predicate in predicates[1:]
    ):
        return None
    domains = [
        chain._operand_domain(cg, item.node, coords, plan)
        for item, coords in zip(outputs, coordinates, strict=True)
    ]
    if any(_conjunction(domain) != _conjunction(domains[0]) for domain in domains[1:]):
        return None
    probes = []
    shared: Counter[tuple[Node, tuple[str, ...]]] = Counter()
    for item, coords in zip(outputs, coordinates, strict=True):
        probe = chain._Expression(cg, plan, boundaries)
        probe.coordinate_names.update((row, base, element))
        probe.value(item.node, coords)
        probes.append(probe)
        shared.update(
            key
            for key in probe.memo
            if key[0] not in probe.boundaries and key[0] not in probe.fragments
        )
    if materialized:
        # Inspect actual coordinate-scoped reads, not merely the presence of a
        # boundary somewhere in the original expression's ancestry. Aliasing
        # views with different names remain the caller's lifetime obligation.
        targets = {item.target for item in outputs}
        for probe in probes:
            reads = {
                probe.boundaries[node]
                for node, _ in probe.memo
                if node in probe.boundaries and node not in probe.fragments
            }
            if probe.loaded_inputs or not reads or reads & targets:
                return None
    meaningful = {
        key
        for key, count in shared.items()
        if count > 1 and chain._pointwise_inputs(key[0])
    }
    loads: list[str] = []
    replacements: dict[tuple[Node, tuple[str, ...]], str] = {}
    for probe in () if materialized else probes:
        uniform = {row, base, *probe.origins.values()}
        if plan.loop is not None:
            uniform.update(plan.loop.captures().values())
        for leaf, leaf_coords, indices, _loaded in probe.loaded_inputs:
            key = leaf, leaf_coords
            if key in replacements or any(
                chain._names(ast.parse(coord, mode="eval")) & probe.definitions.keys()
                for coord in leaf_coords
            ):
                continue
            source = cast("Node", leaf.args[0])
            tensor = source.meta["val"]
            proof = prove_vector_leaf(
                indices,
                probe.definitions,
                element=element,
                uniform_names=uniform,
                shape=chain._host_shape(tensor),
                strides=tuple(tensor.stride()),
                dtype=tensor.dtype,
                mask=probe._load_mask(leaf, leaf_coords),
            )
            if proof is None:
                continue
            tensor_name = probe.tensor_name(source)

            def scalar_for_element(
                value: str,
                leaf: Node = leaf,
                leaf_coords: tuple[str, ...] = leaf_coords,
            ) -> tuple[list[str], str]:
                expression = chain._Expression(cg, plan, boundaries)
                expression.coordinate_names.update((row, base, value))
                result = expression.value(
                    leaf,
                    tuple(_substitute(coord, element, value) for coord in leaf_coords),
                )
                return expression.lines, result

            emission = emit_vector_leaf(
                proof,
                tensor=tensor_name,
                prefix=f"{tag}_leaf_{len(replacements)}",
                pointer_for_indices=lambda ix, tensor_name=tensor_name: (
                    _cute_scalar_pointer_expr(tensor_name, list(ix))
                ),
                scalar_for_element=scalar_for_element,
            )
            loads.extend(emission.lines)
            replacements[key] = f"{emission.values}[{element}]"
    if (
        not materialized
        and not meaningful
        and not any(shared[key] > 1 for key in replacements)
    ):
        return None
    if not replacements and any(probe.loaded_inputs for probe in probes):
        return None
    native = (
        emit_native_inputs(
            plan,
            boundaries,
            probes,
            native_inputs,
            ownership,
            tag,
            row,
            base,
            element,
            execution,
        )
        if native_inputs is not None
        else None
    )
    if native is not None:
        replacements.update(native.replacements)
        loads = [*native.loads, *loads]
    retained = (
        broadcast.emit(
            cg,
            plan,
            boundaries,
            probes,
            ownership,
            tag=tag,
            row=row,
            base=base,
            element=element,
            execution=execution,
        )
        if broadcast is not None
        else None
    )
    if retained is not None:
        replacements.update(retained.replacements)
    expression = chain._Expression(cg, plan, boundaries)
    expression.coordinate_names.update((row, base, element))
    expression.memo.update(replacements)
    setup, values, zeros, copies = [], [], [], []
    for index, (item, coords) in enumerate(zip(outputs, coordinates, strict=True)):
        value = expression.value(item.node, coords)
        dtype = CompileEnvironment.current().backend.dtype_str(
            item.node.meta["val"].dtype
        )
        value = (
            f"{dtype}({value})"
            if item.final_value is None
            else item.final_value(value, dtype, coords)
        )
        prefix = f"{tag}_output_{ordinals[index]}"
        if item.native_store is not None:
            emission = item.native_store.emit(
                item.target, prefix, execution.thread, f"{tag}_step"
            )
            setup.extend(emission.setup)
            target = f"{emission.values}[{element}]"
            copies.append(emission.copy)
        elif item.vector_store:
            destination = (
                item.target
                if item.offset == 0
                else f"cute.domain_offset(({item.offset}, 0), {item.target})"
            )
            setup.extend(
                [
                    f"{prefix}_copy = cute.make_tiled_copy_tv(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), {dtype}, num_bits_per_copy=128), cute.make_layout(({rows}, {columns}), stride=({columns}, 1)), cute.make_layout((1, 8)))",
                    f"{prefix}_thread = {prefix}_copy.get_slice({execution.thread})",
                    f"{prefix}_target = {prefix}_thread.partition_D({destination})",
                    f"{prefix}_values = cute.make_rmem_tensor({prefix}_target[None, 0, 0].shape, {dtype})",
                ]
            )
            target = f"{prefix}_values[{element}]"
            copies.append(
                f"cute.copy({prefix}_copy, {prefix}_values, {prefix}_target[{ownership.copy_indices(f'{tag}_step')}])"
            )
        else:
            target = f"{item.target}[{row} + {item.offset}, {base} + {element}]"
        values.append(f"{target} = {value}")
        zeros.append(f"{target} = {dtype}(0)")
    trips = ownership.trips
    unroll = 1 if producer_unroll is None else producer_unroll.loop_factor(trips)
    result = [
        *(native.setup if native is not None else ()),
        *setup,
        *(retained.before_steps if retained is not None else ()),
        f"for {tag}_step in cutlass.range({trips}, unroll={unroll}):",
        f"    {row} = {ownership.row_expression(execution.thread, f'{tag}_step')}",
        f"    {base} = {ownership.base_expression(execution.thread, f'{tag}_step')}",
        f"    if {row} < {height}:",
        chain._indent(loads, 8),
        *(
            [chain._indent(retained.per_step, 8)]
            if retained is not None and retained.per_step
            else []
        ),
        f"        for {element} in cutlass.range_constexpr(8):",
        f"            if {' & '.join(predicates[0])}:",
        chain._indent([*expression.lines, *values], 16),
        "            else:",
        chain._indent(zeros, 16),
        *([chain._indent(copies, 8)] if copies else []),
    ]
    if native is not None and native_inputs is not None:
        native_inputs.emitted = native.sources
        native_inputs.activated = True
    if retained is not None and broadcast is not None:
        broadcast.complete(retained, boundaries, execution, result)
    return result
