"""Complete raw rectangles in already admitted prepared async-copy images.

Only a uniform source guard changes. The original masked packet body remains
its fallback, and the caller retains all allocation and completion authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ...language.memory_ops import _cute_scalar_pointer_expr
from . import chained_matmul as chain
from .chained_completed_store import _records
from .chained_rectangular_leaf import prove_rectangular_leaf

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_execution import ChainedExecution
    from .chained_frontier_layout import PacketXorVectorSink
    from .chained_matmul import ChainedMatmulPlan
    from .chained_native_vector_sink import NativeVectorSink
    from .chained_rectangular_leaf import RectangularLeafPlan
    from .chained_vector_leaf import DenseVectorSink
    from .chained_vector_ownership import VectorOwnership


def _input_facts(node: Node) -> object:
    pending = [node]
    seen = set()
    result = []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        result.append(
            (
                current,
                id(current.graph),
                current.op,
                current.target,
                current.args,
                current.kwargs,
                current.meta,
            )
        )
        pending.extend(current.all_input_nodes)
    return result


@dataclass(frozen=True)
class RectangularFrontier:
    node: Node
    source: Node
    proof: RectangularLeafPlan
    sink: DenseVectorSink | NativeVectorSink | PacketXorVectorSink
    ownership: VectorOwnership
    execution: ChainedExecution
    tensor: str
    prefix: str
    fallback: tuple[str, ...]
    _selection: object = field(repr=False)

    def facts(self) -> object:
        return _records(
            (
                _input_facts(self.node),
                self.source,
                self.source.op,
                self.source.target,
                self.source.args,
                self.source.kwargs,
                self.source.meta,
                self.proof,
                self.sink,
                self.ownership,
                self.execution,
                self.tensor,
                self.prefix,
                self.fallback,
            )
        )

    def check(self) -> None:
        if self._selection != self.facts():
            raise chain._UnsupportedChain("rectangular frontier evidence changed")

    def lines(self) -> list[str]:
        self.check()
        proof, prefix, tensor = self.proof, self.prefix, self.tensor
        height, width = proof.tile_shape
        pointer = _cute_scalar_pointer_expr(tensor, list(proof.origin_indices))
        byte_span = proof.element_bytes * ((height - 1) * proof.pitch + width) - 1
        guards = [proof.guard]
        for axis, (size, stride) in enumerate(
            zip(proof.shape, proof.strides, strict=True)
        ):
            guards.extend(
                (
                    f"{tensor}.shape[{axis}] == {size}",
                    f"{tensor}.layout.stride[{axis}] == {stride}",
                )
            )
        rows, columns = self.ownership.thread_rows, self.ownership.thread_columns
        # Pass the original whole tensor to partition_D. Its native pointer
        # swizzle or layout-owned XOR remains intact, including dynamic bases.
        return [
            f"{prefix}_copied = cutlass.Boolean(False)",
            f"if {' & '.join(f'({guard})' for guard in guards)}:",
            f"    {prefix}_pointer = {pointer}",
            f"    {prefix}_address = cutlass.Int64({prefix}_pointer.toint())",
            f"    if ({prefix}_address >= 0) & ({prefix}_address <= {2**63 - 1 - byte_span}) & ({prefix}_address % 16 == 0) & (cutlass.Int64({self.sink.target}.iterator.toint()) % 16 == 0):",
            f"        {prefix}_source = cute.make_tensor({prefix}_pointer.align(16), cute.make_layout(({height}, {width}), stride=({proof.pitch}, 1)))",
            f"        {prefix}_copy = cute.make_tiled_copy_tv(cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), {proof.dtype}, num_bits_per_copy=128), cute.make_layout(({rows}, {columns}), stride=({columns}, 1)), cute.make_layout((1, 8)))",
            f"        {prefix}_thread = {prefix}_copy.get_slice({self.execution.thread})",
            f"        cute.copy({prefix}_copy, {prefix}_thread.partition_S({prefix}_source), {prefix}_thread.partition_D({self.sink.target}))",
            f"        {prefix}_copied = cutlass.Boolean(True)",
            f"if not {prefix}_copied:",
            chain._indent(self.fallback),
        ]


def bind_rectangular_frontier(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    node: Node,
    boundaries: Mapping[Node, str],
    execution: ChainedExecution,
    ownership: VectorOwnership,
    sink: DenseVectorSink | NativeVectorSink | PacketXorVectorSink,
    tag: str,
    fallback: list[str],
) -> RectangularFrontier | None:
    """Bind an original identity load, never a computed or transposed image."""
    if (
        node.target is not chain.memory_ops.load
        or node.meta["val"].dtype not in (torch.float16, torch.bfloat16)
        or node.meta["val"].ndim != 2
        or not ownership.matches(sink.shape, execution.threads)
        or ownership.changed
        or sink.shape != chain._shape(node)
        or ownership.shape[0] % ownership.thread_rows
        or execution.threads & (execution.threads - 1)
        or cast("Node", node.args[0]).target is not chain._tracing_ops._host_tensor
        or any(item is not None for item in node.args[3:4])
    ):
        return None
    prefix = f"{tag}_rectangle"
    coordinates = (f"{prefix}_row", f"{prefix}_column")
    expression = chain._Expression(cg, plan, dict(boundaries))
    expression.coordinate_names.update(coordinates)
    try:
        expression.value(node, coordinates)
        if len(expression.loaded_inputs) != 1:
            return None
        leaf, leaf_coords, indices, _value = expression.loaded_inputs[0]
        if leaf is not node or leaf_coords != coordinates:
            return None
        mask = expression._load_mask(node, coordinates)
        predicates = (*chain._operand_domain(cg, node, coordinates, plan),)
        if mask is not None:
            predicates = (mask, *predicates)
    except chain._UnsupportedChain:
        return None
    uniform = set(expression.origins.values())
    uniform.update(
        argument.name
        for scalar, argument in cg.device_function._expr_args.items()
        if sympy.ask(sympy.Q.integer(scalar)) is True
    )
    if plan.loop is not None:
        uniform.update(
            name
            for captured, name in plan.loop.captures().items()
            if isinstance(captured.meta.get("val"), (int, torch.SymInt))
            or isinstance(captured.meta.get("val"), torch.Tensor)
            and captured.meta["val"].ndim == 0
            and captured.meta["val"].dtype in (torch.int32, torch.int64)
        )
    source = cast("Node", node.args[0])
    fake = source.meta["val"]
    proof = prove_rectangular_leaf(
        indices,
        expression.definitions,
        row=coordinates[0],
        column=coordinates[1],
        uniform_names=uniform,
        tile_shape=sink.shape,
        shape=chain._host_shape(fake),
        strides=tuple(fake.stride()),
        dtype=fake.dtype,
        mask=(
            " & ".join(f"({predicate})" for predicate in predicates)
            if predicates
            else None
        ),
    )
    if proof is None or proof.dtype != sink.dtype or proof.pitch % 8:
        return None
    result = RectangularFrontier(
        node,
        source,
        proof,
        sink,
        ownership,
        execution,
        expression.tensor_name(source),
        prefix,
        tuple(fallback),
        None,
    )
    object.__setattr__(result, "_selection", result.facts())
    result.check()
    return result
