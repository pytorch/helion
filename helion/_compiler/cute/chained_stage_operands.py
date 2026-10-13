"""Accepted full-tile operand transfers for the shared contraction stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain
from .chained_tcgen05 import direct_stage

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_tcgen_stage import StageGeometry


class _AddressProbe(chain._DomainExpression):
    """Pure direct-address interpretation; no tensor args or generated locals.

    Restrict tensor-valued indices to the existing view/iota/tile vocabulary.
    Arbitrary pointwise lowering could register codegen temporaries and is left
    to the original root path, as are dynamic host scalar arguments.
    """

    def __init__(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        node: Node,
        coordinates: tuple[str, ...],
    ) -> None:
        super().__init__(cg, plan, node, coordinates)
        leaf = node
        while leaf.target in chain._VIEWS:
            leaf = cast("Node", leaf.args[0])
        self.leaf = leaf

    def bind(self, expression: str) -> str:
        return f"({expression})"

    def tensor_name(self, node: Node) -> str:
        return "chain_probe_tensor"

    def scalar(self, arg: object) -> str:
        if isinstance(arg, Node):
            if isinstance(arg.meta.get("val"), torch.Tensor):
                raise chain._UnsupportedChain("computed scalar index in direct probe")
            if arg.target in (_tracing_ops._get_symnode, torch.ops.aten.sym_size.int):
                value = arg.meta.get("val")
                axis = CompileEnvironment.current().resolve_block_id(value)
                if axis is not None:
                    return str(self.block_size(axis))
                if type(value) is int:
                    return str(value)
                raise chain._UnsupportedChain("dynamic scalar index in direct probe")
        return super().scalar(arg)

    def _load(self, node: Node, coordinates: tuple[str, ...]) -> str:
        if node is not self.leaf:
            raise chain._UnsupportedChain("indirect address in direct probe")
        source = cast("Node", node.args[0])
        indices = self.indices(node, coordinates)
        self.accesses.append((indices, tuple(source.meta["val"].stride())))
        self.global_accesses.append((source, indices))
        return "0"

    def value(self, node: Node, coordinates: tuple[str, ...]) -> str:
        if node.target not in (
            *chain._VIEWS,
            memory_ops.load,
            torch.ops.prims.iota.default,
            chain.tile_index,
        ):
            raise chain._UnsupportedChain("unsupported direct address probe")
        return super().value(node, coordinates)


def _preflight(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    node: Node,
    geometry: StageGeometry,
    role: str,
) -> bool:
    coordinates = ("chain_coordinate_m", "chain_coordinate_n")
    probe = _AddressProbe(cg, plan, node, coordinates)
    inner = chain._operand_inner_axis(cg, plan, {}, node, expression=probe)
    if (inner if role == "a" else 1 - inner) != 1:
        return False
    m, n, k = geometry.physical
    shape = (m if role == "a" else n, k)
    # The original copy proof evaluates the full-tile endpoint's domain.
    last = tuple(str(size - 1) for size in shape)
    logical_last = last if role == "a" else last[::-1]
    domain = _AddressProbe(cg, plan, node, logical_last)
    domain.value(node, logical_last)
    copy_coordinates = ("chain_copy_row", "chain_copy_col")
    copy_probe = _AddressProbe(
        cg, plan, node, copy_coordinates if role == "a" else copy_coordinates[::-1]
    )
    return (
        chain._async_copy(
            cg,
            plan,
            node,
            "chain_probe",
            role,
            shape,
            (k, 1),
            1,
            CompileEnvironment.current().backend.dtype_str(node.meta["val"].dtype),
            [],
            expression=copy_probe,
            domain=chain._domain_bounds(domain),
        )
        is not None
    )


@dataclass(frozen=True)
class DirectStageOperands:
    """Two proved K-major copies and their unchanged masked scalar alternatives.

    This is a codegen-local result, consumed before the graph is changed. It
    grants operand readiness only after commit/wait and the shared-proxy fence;
    it grants neither resource allocation nor result/accumulator lifetime.
    """

    stage: int
    nodes: tuple[Node, Node]
    shape: tuple[int, int, int]
    dtype: torch.dtype
    a: tuple[str, ...]
    b: tuple[str, ...]


def plan_direct_stage_operands(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    stage: int,
    geometry: StageGeometry,
) -> DirectStageOperands | None:
    """Admit exactly the existing direct full-K asynchronous transfer policy."""
    if (
        plan.threads != 128
        or geometry.logical != plan.shapes[stage]
        or geometry.transpose
        or geometry.physical != geometry.logical
    ):
        return None
    nodes = cast("tuple[Node, Node]", tuple(plan.dots[stage].args[:2]))
    if not all(chain._direct_operand(node) for node in nodes):
        return None
    try:
        if not all(
            _preflight(cg, plan, node, geometry, role)
            for node, role in zip(nodes, ("a", "b"), strict=True)
        ):
            return None
    except chain._UnsupportedChain:
        return None
    inner_a = chain._operand_inner_axis(cg, plan, {}, nodes[0])
    inner_b = 1 - chain._operand_inner_axis(cg, plan, {}, nodes[1])
    if inner_a != 1 or inner_b != 1:
        return None
    dtype = plan.operand_dtype(stage)
    dtype_name = CompileEnvironment.current().backend.dtype_str(dtype)
    a = direct_stage(cg, plan, stage, "a", dtype_name)
    b = direct_stage(cg, plan, stage, "b", dtype_name)
    if a is None or b is None:
        raise chain._UnsupportedChain("direct transfer proof changed during emission")
    return DirectStageOperands(
        stage, nodes, geometry.physical, dtype, tuple(a), tuple(b)
    )
