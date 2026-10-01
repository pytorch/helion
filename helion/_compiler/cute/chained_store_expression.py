"""Original scalar store lowering, independent of its iteration transport.

This helper does not authorize a register read, a changed iteration order, or
omission of a shared image. A completed-member caller must separately establish
completion, ownership and lifetime before using a materialized read replacement.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain
from .chained_broadcast_expressions import _lowering_facts
from .chained_register_islands import _freeze
from .contraction_region import _domain

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_matmul import _ScanInput
    from .chained_matmul import _StagedInput


def _node_facts(nodes: tuple[Node, ...]) -> tuple[object, ...]:
    """Independent original graph/typed-lowering facts, without an active codegen."""
    return tuple(
        (
            node,
            node.op,
            node.target,
            _freeze(node.args),
            _freeze(node.kwargs),
            _lowering_facts(node),
            tuple(node.users),
            (value.dtype, _domain(value), _freeze(value.stride()))
            if isinstance(value := node.meta.get("val"), torch.Tensor)
            else _freeze(value),
        )
        for node in nodes
    )


@dataclass(frozen=True)
class MaterializedStoreRead:
    """Typed expression substitution only; no native/completion permission."""

    node: Node
    tensor: str
    shape: tuple[int, ...]
    dtype: str
    coordinates: tuple[str, ...]
    storage_value: str
    _plan: ChainedMatmulPlan
    _facts: tuple[object, ...]
    _selection: tuple[object, ...]

    def _fields(self) -> tuple[object, ...]:
        return (
            self.node,
            self.tensor,
            self.shape,
            self.dtype,
            self.coordinates,
            self.storage_value,
            self._plan,
            self._facts,
        )

    def matches(self, plan: ChainedMatmulPlan, boundaries: Mapping[Node, str]) -> bool:
        return (
            self._selection == self._fields()
            and plan is self._plan
            and boundaries.get(self.node) == self.tensor
            and chain._shape(self.node) == self.shape
            and self._facts == _node_facts((self.node,))
        )


def materialized_store_read(
    plan: ChainedMatmulPlan,
    boundaries: Mapping[Node, str],
    node: Node,
    coordinates: tuple[str, ...],
    storage_value: str,
) -> MaterializedStoreRead:
    """Preserve the original typed boundary, including its guarded zero arm."""
    shape = chain._shape(node)
    if node not in boundaries or len(coordinates) != len(shape):
        raise chain._UnsupportedChain("store replacement lacks its original boundary")
    fields = (
        node,
        boundaries[node],
        shape,
        CompileEnvironment.current().backend.dtype_str(node.meta["val"].dtype),
        coordinates,
        storage_value,
        plan,
        _node_facts((node,)),
    )
    return MaterializedStoreRead(*fields, fields)


@dataclass(frozen=True)
class UnboundStoreTarget:
    """Original output identity, without registering a device argument early."""

    store: Node
    output: Node
    facts: tuple[object, ...]

    def bind(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        store: Node,
        boundaries: dict[Node, str],
    ) -> str:
        if (
            store is not self.store
            or store.args[0] is not self.output
            or self.facts != _node_facts((store, self.output))
        ):
            raise chain._UnsupportedChain("completed store target binding changed")
        return chain._Expression(cg, plan, boundaries).tensor_name(self.output)


@dataclass(frozen=True)
class StorePoint:
    """One original store point, before any execution/iteration is chosen."""

    shape: tuple[int, int]
    coordinates: tuple[str, str]
    lines: tuple[str, ...]
    target: str | UnboundStoreTarget
    indices: tuple[str, ...]
    bounds: tuple[str, ...]
    value: str
    reads: tuple[tuple[Node, tuple[str, ...]], ...]
    host_reads: tuple[Node, ...]


def lower_store_point(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    store: Node,
    boundaries: dict[Node, str],
    staged_inputs: list[_StagedInput],
    scan_inputs: list[_ScanInput],
    *,
    coordinates: tuple[str, str],
    coordinate_names: tuple[str, ...],
    completed_read: MaterializedStoreRead | None = None,
    defer_target: bool = False,
) -> StorePoint:
    """The original `_emit_store` scalar body, with no loop or added masks.

    Until its existing caller delegates here, CPU source-equality tests enforce
    exact parity. No statement parsing/hoisting or generated-kernel rewriting is
    performed; `_Expression` and `_store_domain` retain the original semantics.
    """
    shape = chain._shape(cast("Node", store.args[2]))
    if len(shape) != 2:
        raise chain._UnsupportedChain("contraction live-out rank")
    expression = chain._Expression(cg, plan, boundaries)
    expression.staged_inputs = staged_inputs
    expression.scan_inputs = scan_inputs
    expression.coordinate_names.update(coordinate_names)
    if completed_read is not None:
        if not completed_read.matches(plan, boundaries):
            raise chain._UnsupportedChain("changed materialized store read")
        expression.fragments[completed_read.node] = (
            completed_read.coordinates,
            chain._materialized_value(
                completed_read.tensor,
                completed_read.shape,
                completed_read.coordinates,
                completed_read.dtype,
                storage_value=completed_read.storage_value,
            ),
        )
    value = expression.value(cast("Node", store.args[2]), coordinates)
    indices = expression.indices(store, coordinates)
    indices = [
        expression.bind(index)
        if any(
            isinstance(node, ast.IfExp) for node in ast.walk(expr_from_string(index))
        )
        else index
        for index in indices
    ]
    output = cast("Node", store.args[0])
    fake = output.meta["val"]
    name = (
        UnboundStoreTarget(store, output, _node_facts((store, output)))
        if defer_target
        else expression.tensor_name(output)
    )
    bounds = [
        f"0 <= ({i}) < {s}"
        for i, s in zip(indices, chain._host_shape(fake), strict=True)
    ]
    bounds.extend(chain._store_domain(cg, store, coordinates, plan))
    if len(store.args) > 3 and isinstance(store.args[3], Node):
        mask = store.args[3]
        mask_shape = chain._shape(mask)
        offset = len(coordinates) - len(mask_shape)
        mask_coords = tuple(
            "0" if size == 1 else coordinates[i + offset]
            for i, size in enumerate(mask_shape)
        )
        bounds.append(expression.value(mask, mask_coords))
    return StorePoint(
        cast("tuple[int, int]", shape),
        coordinates,
        tuple(expression.lines),
        name,
        tuple(indices),
        tuple(bounds),
        value,
        tuple(key for key in expression.memo if key[0] in boundaries),
        tuple(item[0] for item in expression.loaded_inputs),
    )
