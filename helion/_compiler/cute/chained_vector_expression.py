"""Coordinate-aware vector materialization of original graph expressions."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language.memory_ops import _cute_scalar_pointer_expr
from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain
from .chained_execution import ChainedExecution
from .chained_vector_leaf import emit_vector_leaf
from .chained_vector_leaf import prove_vector_leaf
from .chained_vector_native import emit_native_inputs
from .chained_vector_ownership import plan_vector_ownership

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_broadcast_retention import BroadcastRetentionAttempt
    from .chained_matmul import ChainedMatmulPlan
    from .chained_pointwise_unroll import BoundedProducerUnroll
    from .chained_vector_leaf import DenseVectorSink
    from .chained_vector_native import NativeReadInputs
    from .chained_vector_ownership import VectorOwnership
    from .chained_vector_stage import VectorStaging


def _substitute(text: str, name: str, replacement: str) -> str:
    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            return ast.parse(replacement, mode="eval").body if node.id == name else node

    return ast.unparse(Substitute().visit(ast.parse(text, mode="eval").body))


def _preserves_raw_host_vector(
    probe: chain._Expression,
    node: Node,
    coordinates: tuple[str, ...],
    *,
    row: str,
    base: str,
    element: str,
    raw_boundaries: Mapping[Node, str] | None,
) -> bool:
    """Retain an original host-vector route after exact raw materialization.

    The caller supplies only published, live, repeated raw-leaf images from
    its admitted frame. This proof grants no storage, lifetime or shared-vector
    load authority: the ordinary bound expression still reads each element.
    Re-probe only to establish the host admission that materialization removed.
    """
    if not raw_boundaries:
        return False
    used = {key for key in probe.memo if key[0] in raw_boundaries}
    if not used or any(
        leaf.target is not chain.memory_ops.load
        or leaf.graph is not node.graph
        or probe.boundaries.get(leaf) != raw_boundaries[leaf]
        or coords != (row, f"({base} + {element})")
        for leaf, coords in used
    ):
        return False
    original = chain._Expression(
        probe.cg,
        probe.plan,
        {
            leaf: name
            for leaf, name in probe.boundaries.items()
            if leaf not in raw_boundaries
        },
    )
    original.coordinate_names.update((row, base, element))
    try:
        original.value(node, coordinates)
        uniform = {row, base, *original.origins.values()}
        if probe.plan.loop is not None:
            uniform.update(probe.plan.loop.captures().values())
        for leaf, leaf_coords, indices, _loaded in original.loaded_inputs:
            if (leaf, leaf_coords) not in used or any(
                chain._names(ast.parse(coord, mode="eval"))
                & original.definitions.keys()
                for coord in leaf_coords
            ):
                continue
            source = cast("Node", leaf.args[0])
            if source.target is not chain._tracing_ops._host_tensor:
                continue
            tensor = source.meta["val"]
            if (
                prove_vector_leaf(
                    indices,
                    original.definitions,
                    element=element,
                    uniform_names=uniform,
                    shape=chain._host_shape(tensor),
                    strides=tuple(tensor.stride()),
                    dtype=tensor.dtype,
                    mask=original._load_mask(leaf, leaf_coords),
                )
                is not None
            ):
                return True
    except chain._UnsupportedChain:
        return False
    return False


def emit_vector_expression(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    node: Node,
    *,
    shape: tuple[int, int],
    coordinates: Callable[[str, str], tuple[str, ...]],
    offset: int,
    tag: str,
    target: str,
    final_value: Callable[[str, str, tuple[str, ...]], str] | None = None,
    vector_store: bool = True,
    producer_unroll: BoundedProducerUnroll | None = None,
    execution: ChainedExecution | None = None,
    raw_boundaries: Mapping[Node, str] | None = None,
    ownership: VectorOwnership | None = None,
    native_inputs: NativeReadInputs | None = None,
    broadcast: BroadcastRetentionAttempt | None = None,
    shared_sink: DenseVectorSink | None = None,
    async_staging: VectorStaging | None = None,
) -> list[str] | None:
    """Materialize eight columns per producer without rewriting their arithmetic.

    ``shape`` describes the physical row/column destination. ``coordinates``
    maps its row and column expressions to the original node's logical axes;
    allocation padding outside that node is zero-filled. ``offset`` is a row
    offset into the destination view, matching cooperative operand staging.

    The default final value is only the original expression cast to its declared
    dtype. In particular, masked host loads remain zero *inputs* to pointwise
    operations: an exp/sigmoid of a masked zero must not become a zero result.
    An operand-stage caller can explicitly supply ``final_value(value, dtype,
    logical_coordinates)`` to apply its independently proven final domain mask.
    The callback runs after original expression emission, before unroll planning.

    Only BF16, FP16 and FP32 values with at least one proven host vector are
    admitted, including an original vector preserved by ``raw_boundaries``.
    Arbitrary materialized boundaries do not enable this path. All leaf
    bounds, stride, alignment, mask, and no-wrap guards and scalar fallbacks are
    delegated unchanged to the existing leaf emitter. No barrier or publication
    protocol is implied; the caller owns the target layout and synchronization.
    ``vector_store=False`` keeps the same host vectors but publishes through
    explicit logical scalar coordinates. Use it when the destination layout
    permutes elements within a vector (e.g. low-column-bit XOR scratch); a
    legal dynamic copy partition alone does not prove its value ordering.
    """
    execution = execution or ChainedExecution(plan.threads)
    height, width = shape
    ownership = ownership or plan_vector_ownership(shape, execution.threads)
    if (
        ownership is None
        or not ownership.matches(shape, execution.threads)
        or ownership.thread_order != "row_major"
    ):
        return None
    rows, columns = ownership.thread_rows, ownership.thread_columns
    if node.meta["val"].dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ):
        return None
    row, base, element = f"{tag}_row", f"{tag}_base", f"{tag}_element"
    coords = coordinates(row, f"({base} + {element})")
    probe = chain._Expression(cg, plan, boundaries)
    probe.coordinate_names.update((row, base, element))
    probe.value(node, coords)
    dtype = CompileEnvironment.current().backend.dtype_str(node.meta["val"].dtype)
    identity_sink = (
        shared_sink is not None
        and async_staging is not None
        and async_staging.async_enabled
        and vector_store
        and final_value is None
        and offset == 0
        and shared_sink.target == target
        and shared_sink.shape == shape == chain._shape(node)
        and shared_sink.dtype == dtype
        and coords == (row, f"({base} + {element})")
        and node.target is chain.memory_ops.load
        and len(probe.loaded_inputs) == 1
        and probe.loaded_inputs[0][:2] == (node, coords)
    )
    uniform = {row, base, *probe.origins.values()}
    if plan.loop is not None:
        uniform.update(plan.loop.captures().values())
    loads: list[str] = []
    replacements: dict[tuple[Node, tuple[str, ...]], str] = {}
    asynchronous: str | None = None
    for leaf, leaf_coords, indices, _loaded in probe.loaded_inputs:
        # A gathered internal view can put probe-local bindings in the leaf's
        # coordinates. Fresh expression instances do not own those bindings,
        # and their memo keys would differ. Keep that access scalar until its
        # coordinate identity is represented independently of emitted names.
        if any(
            chain._names(ast.parse(coord, mode="eval")) & probe.definitions.keys()
            for coord in leaf_coords
        ):
            continue
        source = cast("Node", leaf.args[0])
        tensor = source.meta["val"]
        mask = probe._load_mask(leaf, leaf_coords)
        proof = prove_vector_leaf(
            indices,
            probe.definitions,
            element=element,
            uniform_names=uniform,
            shape=chain._host_shape(tensor),
            strides=tuple(tensor.stride()),
            dtype=tensor.dtype,
            mask=mask,
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
            shared_pointer=shared_sink.pointer(row, base)
            if identity_sink and shared_sink is not None and proof.dtype == dtype
            else None,
        )
        if identity_sink and proof.dtype == dtype:
            asynchronous = emission.vectorized
        loads.extend(emission.lines)
        replacements[leaf, leaf_coords] = f"{emission.values}[{element}]"
    if not replacements and not _preserves_raw_host_vector(
        probe,
        node,
        coords,
        row=row,
        base=base,
        element=element,
        raw_boundaries=raw_boundaries,
    ):
        return None
    native = (
        emit_native_inputs(
            plan,
            boundaries,
            (probe,),
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
            (probe,),
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
    value = expression.value(node, coords)
    value = (
        f"{dtype}({value})"
        if final_value is None
        else final_value(value, dtype, coords)
    )
    predicate = " & ".join(
        f"(({coord}) < {extent})"
        for coord, extent in zip(coords, chain._shape(node), strict=True)
    )
    destination = (
        target if offset == 0 else f"cute.domain_offset(({offset}, 0), {target})"
    )
    trips = ownership.trips
    unroll = 1 if producer_unroll is None else producer_unroll.loop_factor(trips)
    setup = [
        f"{tag}_copy = cute.make_tiled_copy_tv(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), {dtype}, num_bits_per_copy=128), cute.make_layout(({rows}, {columns}), stride=({columns}, 1)), cute.make_layout((1, 8)))",
        f"{tag}_thread = {tag}_copy.get_slice({execution.thread})",
        f"{tag}_target = {tag}_thread.partition_D({destination})",
        f"{tag}_values = cute.make_rmem_tensor({tag}_target[None, 0, 0].shape, {dtype})",
    ]
    output = (
        f"{tag}_values[{element}]"
        if vector_store
        else f"{target}[{row} + {offset}, {base} + {element}]"
    )
    publication = [
        f"        for {element} in cutlass.range_constexpr(8):",
        f"            if {predicate}:",
        chain._indent(expression.lines, 16),
        f"                {output} = {value}",
        "            else:",
        f"                {output} = {dtype}(0)",
        *(
            [
                f"        cute.copy({tag}_copy, {tag}_values, {tag}_target[{ownership.copy_indices(f'{tag}_step')}])"
            ]
            if vector_store
            else []
        ),
    ]
    result = [
        *(native.setup if native is not None else ()),
        *(setup if vector_store else []),
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
        *(
            [
                f"        if not {asynchronous}:",
                chain._indent(publication, 4),
            ]
            if asynchronous is not None
            else publication
        ),
    ]
    if asynchronous is not None:
        assert async_staging is not None
        async_staging.async_activated = True
        result.extend(
            ["cute.arch.cp_async_commit_group()", "cute.arch.cp_async_wait_group(0)"]
        )
    if native is not None and native_inputs is not None:
        native_inputs.activated = True
    if retained is not None and broadcast is not None:
        broadcast.complete(retained, boundaries, execution, result)
    return result
