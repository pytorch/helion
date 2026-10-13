"""A pure root vector with independent storage and complete-CTA publication.

This admission is separate from stage-local pointwise residency. A cache budget
alone grants no read: emission must publish the original typed vector and bind
every redirected coordinate to an existing row/operand or export domain.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from dataclasses import fields
from fractions import Fraction
from operator import itemgetter
from typing import TYPE_CHECKING
from typing import cast
from typing_extensions import Self

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ..device_ir import RootGraphInfo
from ..host_function import HostFunction
from .chained_pointwise_residency import _cost
from .chained_pointwise_residency import _pointwise
from .chained_pointwise_residency import _producers
from .chained_root_native import joined_root
from .chained_root_native import native_facts
from .chained_store_expression import _node_facts

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from .chained_coordinates import RowCoordinateScope
    from .chained_matmul import ChainedMatmulPlan
    from .chained_matmul import _Expression
    from .chained_operand_coordinates import OperandCoordinateScope
    from .chained_root_exports import RootExport
    from .chained_root_exports import RootExports
    from .chained_root_native import RootNativeStage
    from .chained_root_native_emission import RootNativeEmission
    from .chained_scratch_layout import ScratchLayouts
    from .contraction_region import ContractionRegion


def _host_facts() -> tuple[str, ...]:
    return tuple(ast.dump(statement) for statement in HostFunction.current().body)


def _tensor_facts(nodes: tuple[Node, ...]) -> tuple[object, ...]:
    return tuple(
        (
            node,
            id(value),
            id(value.untyped_storage()),
            value.dtype,
            tuple(map(str, value.shape)),
            tuple(map(str, value.stride())),
            str(value.storage_offset()),
        )
        for node in nodes
        if isinstance(value := node.meta.get("val"), torch.Tensor)
    )


def _pure_vectors(
    region: ContractionRegion,
    exports: RootExports | None,
    dots: tuple[Node, ...],
) -> tuple[tuple[Node, tuple[Node, ...]], ...]:
    from .chained_matmul import _ancestors
    from .chained_matmul import _supported

    if exports is None or region.carries:
        return ()
    barriers = {*dots, *region.scans, *region.reductions}
    position = {node: index for index, node in enumerate(region.nodes)}
    result = []
    for export in exports.stores:
        node = cast("Node", export.store.args[2])
        value = node.meta.get("val")
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 1
            or value.dtype != torch.float32
            or not _pointwise(node)
        ):
            continue
        closure = _ancestors(node)
        loads = [item for item in closure if item.target is memory_ops.load]
        if (
            closure & barriers
            or not loads
            or any(not _supported(item) for item in closure)
            or any(
                not isinstance(item.args[0], Node)
                or item.args[0].target is not _tracing_ops._host_tensor
                for item in loads
            )
            or any(item.target is torch.ops.aten.view.dtype for item in closure)
            or not any(_cost(item) > 1 for item in closure)
        ):
            continue
        result.append((node, tuple(sorted(closure, key=position.__getitem__))))
    return tuple(result)


def has_root_vector_candidate(graphs: Sequence[GraphInfo]) -> bool:
    """Discover an exported pure vector; final readers and storage remain gates."""
    from .chained_matmul import _classify_chained_graph

    graph = _classify_chained_graph(graphs)
    return (
        graph is not None
        and graph.loop is None
        and isinstance(graph.root, RootGraphInfo)
        and bool(_pure_vectors(graph.region, graph.root_exports, graph.dots))
    )


def _scope(plan: ChainedMatmulPlan) -> tuple[object, ...]:
    return (
        plan.root_graph_id,
        plan.region,
        plan.dots,
        plan.store,
        plan.axes,
        plan.shapes,
        plan.dtype,
        plan.threads,
        plan.strategy,
        native_facts(plan),
        plan.root_exports,
        plan.scan_exports,
    )


def _ordinary(plan: ChainedMatmulPlan) -> bool:
    return (
        plan.region is not None
        and joined_root(plan)
        and plan.loop is None
        and plan.pointwise_cache is None
        and plan.preparation_pipeline is None
        and plan.contraction_groups is None
        and plan.initialized_accumulator is None
        and plan.late_rhs_reuse is None
        and plan.root_typed_operands is None
        and plan.k_schedule is None
        and not plan.warp_mma_stages
        and plan.loop_workspace is None
        and plan.warp_workspace is None
        and plan.warp_result_transport is None
        and plan.root_workspace is None
        and not plan.direct_output
    )


@dataclass(frozen=True)
class RootVectorCache:
    node: Node
    name: str
    extent: int
    allocated_bytes: int
    closure: tuple[Node, ...]
    root: RootGraphInfo
    phase: int
    nodes: tuple[Node, ...]
    graph_facts: tuple[object, ...]
    tensor_facts: tuple[object, ...]
    # Keep identity witnesses alive; never infer runtime nonaliasing from these.
    tensor_owners: tuple[torch.Tensor, ...]
    storage_owners: tuple[object, ...]
    host_facts: tuple[str, ...]
    scope: tuple[object, ...]
    aliases: dict[str, str]
    alias_items: tuple[tuple[str, str], ...]
    selection: tuple[object, ...]

    def fields(self) -> tuple[object, ...]:
        return (
            self.node,
            self.name,
            self.extent,
            self.allocated_bytes,
            self.closure,
            self.root,
            self.phase,
            self.nodes,
            self.graph_facts,
            self.tensor_facts,
            tuple(map(id, self.tensor_owners)),
            tuple(map(id, self.storage_owners)),
            self.host_facts,
            self.scope,
            id(self.aliases),
            self.alias_items,
        )

    def check(self, plan: ChainedMatmulPlan) -> None:
        from .chained_matmul import _shape
        from .chained_matmul import _UnsupportedChain
        from .chained_root_exports import valid_root_exports

        if (
            not _ordinary(plan)
            or plan.region is None
            or plan.root_vector_cache is not self
            or self.fields() != self.selection
            or self.root.graph is not plan.region.graph
            or self.root.phase_index != self.phase
            or self.root.graph_id != plan.root_graph_id
            or tuple(self.root.graph.nodes) != self.nodes
            or _scope(plan) != self.scope
            or _node_facts(self.nodes) != self.graph_facts
            or _tensor_facts(self.nodes) != self.tensor_facts
            or _host_facts() != self.host_facts
            or plan.tensor_aliases is not self.aliases
            or any(self.aliases.get(key) != value for key, value in self.alias_items)
            or _shape(self.node) != (self.extent,)
            or self.allocated_bytes != (self.extent * 4 + 127) // 128 * 128
            or not valid_root_exports(plan)
            or (self.node, self.closure)
            not in _pure_vectors(plan.region, plan.root_exports, plan.dots)
        ):
            raise _UnsupportedChain(
                "root vector graph, owner, phase or storage changed"
            )


def plan_root_vector(
    plan: ChainedMatmulPlan, root: RootGraphInfo, budget: int
) -> RootVectorCache | None:
    from .chained_matmul import _ancestors
    from .chained_matmul import _shape
    from .chained_root_exports import valid_root_exports

    if (
        not _ordinary(plan)
        or plan.region is None
        or budget <= 0
        or not valid_root_exports(plan)
    ):
        return None
    choices = []
    boundaries = {*plan.dots, *plan.scans, *plan.region.reductions}
    producers = tuple(_ancestors(value) for _, value in _producers(plan))
    for node, closure in _pure_vectors(plan.region, plan.root_exports, plan.dots):
        extent = _shape(node)[0]
        if type(extent) is not int or not 0 < extent <= 2**31 - plan.threads:
            continue
        allocated = (extent * 4 + 127) // 128 * 128
        uses = sum(node in ancestors for ancestors in producers)
        if allocated > budget or uses < 2 or set(closure) & boundaries:
            continue
        saved = (uses - 1) * extent * sum(map(_cost, closure))
        choices.append(
            (Fraction(saved, allocated), saved, node, closure, extent, allocated)
        )
    if not choices:
        return None
    _, _, node, closure, extent, allocated = max(choices, key=itemgetter(slice(2)))
    nodes = tuple(root.graph.nodes)
    tensors = tuple(
        value
        for item in nodes
        if isinstance(value := item.meta.get("val"), torch.Tensor)
    )
    fields = (
        node,
        "chain_root_vector_0",
        extent,
        allocated,
        closure,
        root,
        root.phase_index,
        nodes,
        _node_facts(nodes),
        _tensor_facts(nodes),
        tensors,
        tuple(value.untyped_storage() for value in tensors),
        _host_facts(),
        _scope(plan),
        plan.tensor_aliases,
        tuple(plan.tensor_aliases.items()),
    )
    cache = RootVectorCache(*fields, ())
    object.__setattr__(cache, "selection", cache.fields())
    return cache


@dataclass(frozen=True)
class VectorExportScope:
    plan: ChainedMatmulPlan
    ordinal: int
    export: RootExport
    prefix: str
    extent: int

    def check(self, plan: ChainedMatmulPlan) -> None:
        from .chained_matmul import _UnsupportedChain
        from .chained_root_exports import valid_root_exports

        if (
            plan is not self.plan
            or plan.root_exports is None
            or not 0 <= self.ordinal < len(plan.root_exports.stores)
            or plan.root_exports.stores[self.ordinal] is not self.export
            or self.prefix != f"chain_root_export_{self.ordinal}"
            or self.export.shape != (self.extent,)
            or self.extent <= 0
            or not valid_root_exports(plan)
        ):
            raise _UnsupportedChain("root vector export scope changed")

    def wrap(self, body: list[str]) -> list[str]:
        from .chained_matmul import _indent

        self.check(self.plan)
        return [
            f"for {self.prefix}_step in cutlass.range_constexpr({(self.extent + self.plan.threads - 1) // self.plan.threads}):",
            f"    {self.prefix} = chain_thread + {self.prefix}_step * {self.plan.threads}",
            f"    if {self.prefix} < {self.extent}:",
            _indent(body, 8),
        ]


_PUBLICATION_KEY = object()


class RootVectorPublication:
    def __init__(
        self,
        cg: GenerateAST,
        plan: ChainedMatmulPlan,
        lines: list[str],
        cache: RootVectorCache,
        program: tuple[str, ...],
        key: object,
    ) -> None:
        from .chained_execution import ChainedExecution
        from .chained_matmul import _UnsupportedChain
        from .chained_warp_stage import _snapshot

        if key is not _PUBLICATION_KEY:
            raise _UnsupportedChain(
                "root vector publication requires its emitted producer"
            )
        self.codegen = cg
        self.plan = plan
        self.cache = cache
        self.execution = ChainedExecution(plan.threads)
        self.lines = lines
        self.prefix = tuple(lines)
        self.program = program
        self.aliases = plan.tensor_aliases
        self.alias_items = tuple(self.aliases.items())
        self.config = _snapshot(cg.device_function.config.config)
        self.selection = self.fields()
        self.reads: list[RootReadScope] = []

    def fields(self) -> tuple[object, ...]:
        return (
            id(self.codegen),
            id(self.plan),
            id(self.cache),
            self.execution,
            id(self.lines),
            self.prefix,
            self.program,
            id(self.aliases),
            self.alias_items,
            self.config,
        )

    def check(self) -> None:
        from .chained_matmul import _UnsupportedChain
        from .chained_warp_stage import _snapshot

        self.cache.check(self.plan)
        if (
            self.fields() != self.selection
            or self.codegen.current_root_graph_info is not self.cache.root
            or self.plan.tensor_aliases is not self.aliases
            or any(self.aliases.get(k) != v for k, v in self.alias_items)
            or len(self.program) != 6
            or self.prefix[-len(self.program) :] != self.program
            or _snapshot(self.codegen.device_function.config.config) != self.config
            or tuple(self.lines[: len(self.prefix)]) != self.prefix
            or not self.prefix
            or self.prefix[-1] != self.execution.sync
        ):
            raise _UnsupportedChain("root vector publication prefix or team changed")

    def finish(self, boundaries: dict[Node, str]) -> None:
        from .chained_matmul import _UnsupportedChain

        self.check()
        binding = boundaries.get(self.cache.node)
        if (
            not isinstance(binding, PublishedRootVector)
            or binding.publication is not self
            or not self.reads
        ):
            raise _UnsupportedChain("root vector publication was lost or unused")
        for read in self.reads:
            read.check()


class PublishedRootVector(str):
    """A name with a private, already-emitted root publication authority."""

    __slots__ = ("publication", "_owner")

    publication: RootVectorPublication
    _owner: RootVectorPublication

    def __new__(cls, publication: RootVectorPublication) -> Self:
        result = super().__new__(cls, publication.cache.name)
        result.publication = publication
        result._owner = publication
        return result


@dataclass(frozen=True)
class NativeFragmentScope:
    """Original TMEM load coordinates, before its retained publication join."""

    body: RootNativeEmission
    stage: RootNativeStage
    lines: list[str]
    prefix: tuple[str, ...]
    coordinates: tuple[str, str]
    index: str
    fragment: str
    load_program: tuple[str, ...]

    def check(self, plan: ChainedMatmulPlan) -> None:
        from .chained_matmul import _UnsupportedChain
        from .chained_tcgen05 import _load_result

        self.body.check()
        prefix = f"chain_{self.stage.index}"
        shape = self.stage.geometry.physical[:2]
        if (
            plan is not self.body.plan
            or self.body.native.selected(plan, self.stage.index) is not self.stage
            or self.stage.geometry.transpose
            or self.stage.geometry.logical[:2] != shape
            or self.coordinates != (f"{prefix}_typed_row", f"{prefix}_typed_column")
            or self.index != f"{prefix}_typed_index"
            or self.fragment != f"{prefix}_values[{self.index}]"
            or self.load_program
            != tuple(_load_result(prefix, shape, execution=self.body.execution))
            or self.prefix[-len(self.load_program) :] != self.load_program
            or tuple(self.lines[: len(self.prefix)]) != self.prefix
        ):
            raise _UnsupportedChain("native fragment coordinate scope changed")

    def check_fragment(
        self,
        plan: ChainedMatmulPlan,
        source: Node,
        coordinates: tuple[str, ...],
        fragment: str,
    ) -> None:
        from .chained_matmul import _UnsupportedChain

        self.check(plan)
        if (
            source is not self.stage.node
            or coordinates != self.coordinates
            or fragment != self.fragment
        ):
            raise _UnsupportedChain("native fragment source binding changed")

    def wrap(self, body: list[str]) -> list[str]:
        from .chained_matmul import _indent

        self.check(self.body.plan)
        prefix = f"chain_{self.stage.index}"
        row, column = self.coordinates
        rows, columns = self.stage.geometry.logical[:2]
        return [
            f"for {self.index} in cutlass.range_constexpr(cute.size({prefix}_values)):",
            f"    {row}, {column} = {prefix}_coords[{self.index}]",
            f"    if 0 <= {row} < {rows} and 0 <= {column} < {columns}:",
            _indent(body, 8),
        ]


class RootReadScope:
    def __init__(
        self,
        expression: _Expression,
        origin: RowCoordinateScope
        | OperandCoordinateScope
        | VectorExportScope
        | NativeFragmentScope,
    ) -> None:

        self.expression = expression
        self.expression_identity = id(expression)
        self.origin = origin
        self.origin_identity = id(origin)
        self.origin_facts = tuple(
            (item.name, getattr(origin, item.name)) for item in fields(origin)
        )
        self.context = expression.coordinate_normalization
        self.domain = None if self.context is None else self.context.domain
        self.context_seed = self._initial_context()
        self.reads: list[tuple[str, str, object, object]] = []
        self.check()

    def _initial_context(self) -> tuple[tuple[str, object], ...]:
        from .chained_coordinates import CoordinateContext
        from .chained_coordinates import RowCoordinateScope
        from .chained_matmul import _UnsupportedChain
        from .chained_operand_coordinates import OperandCoordinates
        from .chained_operand_coordinates import OperandCoordinateScope

        if isinstance(self.origin, VectorExportScope):
            if self.context is not None:
                raise _UnsupportedChain("export scope must retain its original loop")
            return ()
        if isinstance(self.origin, NativeFragmentScope):
            if self.context is not None:
                raise _UnsupportedChain("fragment scope must retain its original loop")
            return ()
        if isinstance(self.origin, RowCoordinateScope):
            expected = CoordinateContext(self.origin.domain)
        elif isinstance(self.origin, OperandCoordinateScope):
            expected = OperandCoordinates(self.origin.domain)
        else:
            raise _UnsupportedChain("root vector read has no native coordinate scope")
        if (
            type(self.context) is not type(expected)
            or self.context.domain is not self.origin.domain
            or self.context.expressions != expected.expressions
            or self.context.proofs
        ):
            raise _UnsupportedChain(
                "root vector coordinate context was not created by its scope"
            )
        return tuple(expected.expressions.items())

    def _coordinate_matches(self, text: str, expected: object) -> bool:
        from .chained_coordinates import prove

        assert self.context is not None
        current = self.context.lookup(text)
        if current == expected:
            return True
        # Original normalization can replace even a seed row with its pivot
        # constant. Accept only that recomputed native proof, in this domain.
        from .chained_coordinates import IndexExpr

        if not isinstance(expected, IndexExpr):
            return False
        proof = prove(expected, self.context.domain)
        return (
            proof is not None
            and self.context.proofs.get(expected) == proof
            and current == proof.replacement
        )

    def check(self) -> None:
        from .chained_coordinates import RowCoordinateScope
        from .chained_matmul import _UnsupportedChain
        from .chained_operand_coordinates import OperandCoordinateScope

        expression = self.expression
        origin = self.origin
        if (
            id(expression) != self.expression_identity
            or id(origin) != self.origin_identity
            or expression.root_vector_scope is not None
            and expression.root_vector_scope is not self
            or tuple((item.name, getattr(origin, item.name)) for item in fields(origin))
            != self.origin_facts
            or expression.coordinate_normalization is not self.context
            or self.context is not None
            and (
                self.context.domain is not self.domain
                or any(not self._coordinate_matches(k, v) for k, v in self.context_seed)
            )
        ):
            raise _UnsupportedChain(
                "root vector read context or originating scope changed"
            )
        if isinstance(origin, RowCoordinateScope):
            origin.check(expression.plan)
            if self.context is None or self.domain != origin.domain:
                raise _UnsupportedChain("root vector row domain is foreign")
        elif isinstance(origin, OperandCoordinateScope):
            origin.check(expression.cg, expression.plan, origin.stage, origin.layout)
            if self.context is None or self.domain != origin.domain:
                raise _UnsupportedChain("root vector operand domain is foreign")
        elif isinstance(origin, VectorExportScope):
            origin.check(expression.plan)
            if self.context is not None:
                raise _UnsupportedChain("root vector export context is foreign")
        elif isinstance(origin, NativeFragmentScope):
            origin.check(expression.plan)
            if (
                self.context is not None
                or expression.cg is not origin.body.cg
                or expression.fragments.get(origin.stage.node)
                != (origin.coordinates, origin.fragment)
            ):
                raise _UnsupportedChain("root vector fragment context is foreign")
        else:
            raise _UnsupportedChain("root vector read lacks an owned emission scope")
        for coordinate, original, term, original_term in self.reads:
            if self.context is not None and (
                not self._coordinate_matches(coordinate, term)
                or not self._coordinate_matches(original, original_term)
            ):
                raise _UnsupportedChain("root vector consumed coordinate proof changed")

    def read(
        self, binding: PublishedRootVector, node: Node, coordinates: tuple[str, ...]
    ) -> None:
        from .chained_coordinates import _values
        from .chained_coordinates import original_coordinate
        from .chained_matmul import _UnsupportedChain

        self.check()
        publication = binding.publication
        publication.check()
        cache = publication.cache
        if (
            binding.publication is not binding._owner
            or publication.codegen is not self.expression.cg
            or publication.plan is not self.expression.plan
            or node is not cache.node
            or str(binding) != cache.name
            or self.expression.boundaries.get(node) is not binding
            or len(coordinates) != 1
        ):
            raise _UnsupportedChain("root vector read has a foreign owner")
        coordinate = coordinates[0]
        original = original_coordinate(coordinate)
        if isinstance(self.origin, VectorExportScope):
            if (
                self.origin.export.store.args[2] is not node
                or coordinate != self.origin.prefix
                or original != self.origin.prefix
                or self.origin.extent != cache.extent
            ):
                raise _UnsupportedChain("root vector export coordinate is foreign")
        elif isinstance(self.origin, NativeFragmentScope):
            if (
                coordinate != original
                or coordinate not in self.origin.coordinates
                or self.origin.stage.geometry.logical[
                    self.origin.coordinates.index(coordinate)
                ]
                > cache.extent
            ):
                raise _UnsupportedChain("root vector fragment coordinate is unbounded")
        else:
            assert self.context is not None
            term = self.context.lookup(coordinate)
            original_term = self.context.lookup(original)
            points = self.context.domain.points()
            values = None if term is None or points is None else _values(term, points)
            if (
                values is None
                or points is None
                or not all(0 <= value < cache.extent for value in values)
                or original_term is None
                or _values(original_term, points) != values
            ):
                raise _UnsupportedChain(
                    "root vector read is not proved inside its allocation"
                )
            self.reads.append((str(coordinate), original, term, original_term))
        if not any(item is self for item in publication.reads):
            publication.reads.append(self)


def bind_read_scope(
    expression: _Expression,
    origin: RowCoordinateScope
    | OperandCoordinateScope
    | VectorExportScope
    | NativeFragmentScope,
) -> None:
    if expression.plan.root_vector_cache is not None:
        expression.root_vector_scope = RootReadScope(expression, origin)


def check_read_scope(expression: _Expression) -> None:
    if expression.root_vector_scope is not None:
        expression.root_vector_scope.check()


def check_read(
    expression: _Expression, node: Node, coordinates: tuple[str, ...]
) -> None:
    from .chained_matmul import _UnsupportedChain

    binding = expression.boundaries[node]
    if isinstance(binding, PublishedRootVector):
        if expression.root_vector_scope is None:
            raise _UnsupportedChain("root vector read precedes its owned scope")
        expression.root_vector_scope.read(binding, node, coordinates)


def original_probe_boundaries(boundaries: dict[Node, str]) -> dict[Node, str]:
    return {
        node: name
        for node, name in boundaries.items()
        if not isinstance(name, PublishedRootVector)
    }


def executable_boundaries(
    probes: dict[Node, str], boundaries: dict[Node, str]
) -> dict[Node, str]:
    return {
        **probes,
        **{
            node: name
            for node, name in boundaries.items()
            if isinstance(name, PublishedRootVector)
        },
    }


def publish_root_vector(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    lines: list[str],
    boundaries: dict[Node, str],
    scratch: ScratchLayouts,
) -> RootVectorPublication | None:
    from .chained_execution import ChainedExecution
    from .chained_matmul import _Expression
    from .chained_matmul import _indent
    from .chained_matmul import _UnsupportedChain

    cache = plan.root_vector_cache
    if cache is None:
        return None
    cache.check(plan)
    if cache.node in boundaries or any(node in boundaries for node in cache.closure):
        raise _UnsupportedChain("root vector producer depends on a materialized value")
    execution = ChainedExecution(plan.threads)
    index = f"{cache.name}_index"
    expression = _Expression(cg, plan, boundaries)
    expression.coordinate_names.add(index)
    value = expression.value(cache.node, (index,))
    if scratch.read_buffers is not None:
        scratch.read_buffers = scratch.read_buffers | frozenset((cache.name,))
    layout = scratch.layout(cache.name, (cache.extent,), torch.float32)
    program = (
        f"{cache.name} = cute.make_tensor(cute.arch.alloc_smem(cutlass.Float32, {cache.extent}, alignment=128), {layout})",
        f"for {cache.name}_step in cutlass.range_constexpr({(cache.extent + plan.threads - 1) // plan.threads}):",
        f"    {index} = {execution.thread} + {cache.name}_step * {plan.threads}",
        f"    if {index} < {cache.extent}:",
        _indent([*expression.lines, f"{cache.name}[{index}] = {value}"], 8),
        execution.sync,
    )
    lines.extend(program)
    publication = RootVectorPublication(
        cg, plan, lines, cache, program, _PUBLICATION_KEY
    )
    publication.check()
    boundaries[cache.node] = PublishedRootVector(publication)
    return publication
