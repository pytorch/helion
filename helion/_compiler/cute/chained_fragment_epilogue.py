"""Original pointwise lowering over a bound completed-result image.

Expression construction reserves the original SSA names but grants no native
ownership. Callers retain their actual completion/layout/publication checks.
Both warp and TMEM adapters use this expression, not a second pointwise renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from ..inductor_lowering import PointwiseCodegenPolicy
    from . import chained_matmul as chain
    from .chained_matmul import ChainedMatmulPlan
    from .prepared_graph_schedule import ContractionGraph


@dataclass(frozen=True)
class FragmentExpression:
    expression: chain._Expression
    source: Node
    target: Node
    coordinates: tuple[str, ...]
    target_coordinates: tuple[str, ...]
    fragment: str
    dtype: str
    value: str
    domain: tuple[str, ...]
    _selection: object = field(repr=False)

    def facts(self) -> object:
        # Enclosing stage/graph revisions remain the original authority. These
        # facts protect the actual expression and exact per-thread image binding.
        expression = self.expression
        return (
            expression.cg,
            expression.context,
            expression.pointwise_policy,
            self.source,
            self.target,
            self.coordinates,
            self.target_coordinates,
            self.fragment,
            self.dtype,
            self.value,
            self.domain,
            tuple(expression.fragments.items()),
            tuple(expression.pointwise_views.items()),
            tuple(expression.memo.items()),
            tuple(expression.lines),
        )

    def check(self) -> None:
        from . import chained_matmul as chain

        if (
            self._selection != self.facts()
            or self.expression.fragments.get(self.source)
            != (self.coordinates, self.fragment)
            or self.expression.memo.get((self.source, self.coordinates))
            != self.fragment
            or any(
                node is self.source and coordinates != self.coordinates
                for node, coordinates in self.expression.memo
            )
        ):
            raise chain._UnsupportedChain("fragment epilogue expression changed")

    def masked_value(self) -> str:
        from . import chained_matmul as chain

        self.check()
        return chain._masked_operand(self.value, self.dtype, list(self.domain))


def bind_fragment_expression(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scans: list[chain._ScanInput],
    source: Node,
    target: Node,
    coordinates: tuple[str, ...],
    target_coordinates: tuple[str, ...],
    fragment: str,
    dtype: str,
    *,
    operand_domain: bool = True,
) -> FragmentExpression:
    """Evaluate the original Node, retaining all casts, masks and SSA order."""
    from . import chained_matmul as chain

    if source not in plan.dots or source.graph is not target.graph:
        raise chain._UnsupportedChain("foreign fragment epilogue relation")
    expression = chain._Expression(cg, plan, boundaries)
    expression.scan_inputs = scans
    expression.coordinate_names.update(coordinates)
    expression.fragments[source] = (coordinates, fragment)
    value = expression.value(target, target_coordinates)
    domain = (
        tuple(chain._operand_domain(cg, target, target_coordinates, plan))
        if operand_domain
        else ()
    )
    result = FragmentExpression(
        expression,
        source,
        target,
        coordinates,
        target_coordinates,
        fragment,
        dtype,
        value,
        domain,
        None,
    )
    object.__setattr__(result, "_selection", result.facts())
    result.check()
    return result


def bind_register_fragment_expression(
    cg: GenerateAST,
    graph: ContractionGraph,
    source: Node,
    target: Node,
    coordinates: tuple[str, ...],
    target_coordinates: tuple[str, ...],
    fragment: str,
    dtype: str,
    *,
    inputs: dict[Node, tuple[tuple[str, ...], str]] | None = None,
    policy: PointwiseCodegenPolicy | None = None,
) -> FragmentExpression:
    """Original pointwise lowering with every native register input explicit.

    This view grants no completion/publication authority. The physical caller
    owns each input image and the output store cut. No whole-root plan, global
    load, materialization or synthetic graph is constructed for the expression.
    """
    from . import chained_matmul as chain

    graph.check()
    nodes = graph.region.nodes
    bindings = dict(inputs or {})
    if (
        source in bindings
        or source not in nodes
        or target not in nodes
        or any(node not in nodes for node in bindings)
    ):
        raise chain._UnsupportedChain("foreign native fragment expression input")
    bindings[source] = (coordinates, fragment)
    expression = chain._Expression(cg, None, {})
    expression.pointwise_policy = policy
    expression.fragments.update(bindings)
    expression.coordinate_names.update(
        name for coords, value in bindings.values() for name in coords
    )
    value = (
        expression.scalar(target)
        if not target_coordinates
        and not isinstance(target.meta.get("val"), torch.Tensor)
        else expression.value(target, target_coordinates)
    )
    result = FragmentExpression(
        expression,
        source,
        target,
        coordinates,
        target_coordinates,
        fragment,
        dtype,
        value,
        (),
        None,
    )
    object.__setattr__(result, "_selection", result.facts())
    result.check()
    graph.check()
    return result


@dataclass(frozen=True)
class WarpOperandEpilogue:
    """Original typed shared-operand publication, not completion authority."""

    value: FragmentExpression
    prefix: str
    previous: str
    shape: tuple[int, int]
    destination: str

    def facts(self) -> object:
        return (
            self.value.facts(),
            self.prefix,
            self.previous,
            self.shape,
            self.destination,
        )

    def render(self) -> tuple[str, ...]:
        from . import chained_matmul as chain

        value = self.value
        value.check()
        prefix, previous = self.prefix, self.previous
        return (
            f"{prefix}_coords = {previous}_thr.partition_C(cute.make_identity_tensor({self.shape!r}))",
            f"{prefix}_values = cute.make_rmem_tensor({previous}_acc.shape, {value.dtype})",
            f"for {prefix}_index in cutlass.range_constexpr(cute.size({prefix}_values)):",
            f"    {value.coordinates[0]}, {value.coordinates[1]} = {prefix}_coords[{prefix}_index]",
            chain._indent(value.expression.lines),
            f"    {prefix}_values[{prefix}_index] = {value.masked_value()}",
            f"cute.autovec_copy({prefix}_values, {previous}_thr.partition_C({self.destination}))",
        )
