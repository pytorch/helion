"""Graph-owned candidates for independent, one-warp contraction components.

This is mathematical support analysis under the caller's explicit fast-math
policy, NOT an executable proof. Removing zero products can change signed zero,
NaN and infinity behavior. Original arithmetic nodes, casts and surviving K
order are retained; cancellation and approximate-zero inference are never used.

Candidates authorize no emission, register allocation or shared-storage reuse.
A late binder must revalidate the revision, every original expression and
operand domain at the actual owned coordinates (initially empty intermediate
domains), physical fragment maps, entry completion and all export lifetimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.fx import Node
from torch.fx.node import map_arg

from ...language import _tracing_ops
from ...language import view_ops
from .chained_preparation_cut import _shape_input
from .contraction_region import _contraction
from .contraction_region import _domain

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .chained_contraction_groups import ContractionGroup
    from .chained_tcgen_stage import StageGeometry
    from .contraction_region import ContractionRegion


@dataclass(frozen=True)
class RegisterIssue:
    stage: int
    node: Node
    geometry: StageGeometry
    # Logical origins, before the original group's physical transpose/offset.
    origins: tuple[int, int, int]


@dataclass(frozen=True)
class RegisterComponent:
    issues: tuple[RegisterIssue, ...]


@dataclass(frozen=True)
class RegisterImage:
    """One original Node at an exact, complete component-origin relation."""

    node: Node
    tiles: tuple[tuple[int, tuple[int, int]], ...]


@dataclass(frozen=True)
class RegisterValue:
    """One original typed SSA version; event positions are original FX order."""

    node: Node
    dtype: torch.dtype
    shape: tuple[int, ...]
    defined_at: int
    last_use: int
    consumers: tuple[Node, ...]
    # Bit j in each logical row means column j MAY be nonzero. These facts
    # require this candidate's unchanged revision and fast-math authority.
    support_rows: tuple[int, ...]
    # Component index and logical 16x16 origin. Distinct versions never alias.
    tiles: tuple[tuple[int, tuple[int, int]], ...]
    additional_images: tuple[RegisterImage, ...] = ()

    @property
    def images(self) -> tuple[RegisterImage, ...]:
        return (RegisterImage(self.node, self.tiles), *self.additional_images)


@dataclass(frozen=True)
class RegisterIslandRevision:
    region: ContractionRegion
    groups: tuple[ContractionGroup, ...]
    shapes: tuple[tuple[Node, tuple[int, ...]], ...]
    entry_boundaries: frozenset[Node]
    facts: tuple[object, ...]
    multi_image: bool = False


@dataclass(frozen=True)
class RegisterIsland:
    revision: RegisterIslandRevision
    groups: tuple[ContractionGroup, ...]
    components: tuple[RegisterComponent, ...]
    values: tuple[RegisterValue, ...]
    entries: tuple[Node, ...]
    # Pure FX expressions interleaved with the span but used only outside it;
    # keep their original downstream evaluation and shared C dependencies.
    deferred_nodes: tuple[Node, ...]
    # Original C boundaries, including transitive pointwise escapes. Exporting
    # them preserves existing downstream computations rather than relocating
    # arbitrary pointwise snapshots to a newly invented shared buffer.
    exports: tuple[Node, ...]
    domain_nodes: tuple[Node, ...]


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return tuple((_freeze(k), _freeze(v)) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_freeze(item) for item in value))
    if isinstance(value, slice):
        return (slice, value.start, value.stop, value.step)
    if type(value) is float:
        return (float, value.hex())
    return value


def _facts(region: ContractionRegion) -> tuple[object, ...]:
    return tuple(
        (
            node,
            node.op,
            node.target,
            _freeze(node.args),
            _freeze(node.kwargs),
            (value.dtype, _domain(value))
            if isinstance(value := node.meta.get("val"), torch.Tensor)
            else _freeze(value),
            tuple(node.users),
        )
        for node in region.graph.nodes
    )


def register_island_matches(
    island: RegisterIsland,
    region: ContractionRegion,
    groups: tuple[ContractionGroup, ...],
    shapes: Mapping[Node, tuple[int, ...]],
    *,
    entry_boundaries: frozenset[Node],
) -> bool:
    """Revision equality only; this does not replace any late execution proof."""
    revision = island.revision
    return (
        revision.region is region
        and revision.groups == groups
        and revision.entry_boundaries == entry_boundaries
        and all(shapes.get(node) == shape for node, shape in revision.shapes)
        and revision.facts == _facts(region)
    )


_ATEN = torch.ops.aten
_CAST = torch.ops.prims.convert_element_type.default
_ARITHMETIC = {
    _ATEN.add.Tensor,
    _ATEN.sub.Tensor,
    _ATEN.mul.Tensor,
    _ATEN.neg.default,
    _CAST,
    _ATEN.where.self,
    _tracing_ops._mask_to,
    _tracing_ops._new_var,
}
_EXACT = {
    _ATEN.scalar_tensor.default,
    _ATEN.div.Tensor_mode,
    _ATEN.remainder.Scalar,
    _ATEN.eq.Tensor,
    _ATEN.eq.Scalar,
    _ATEN.ne.Tensor,
    _ATEN.ge.Tensor,
    _ATEN.gt.Tensor,
    _ATEN.ge.Scalar,
    _ATEN.lt.Scalar,
    _ATEN.bitwise_and.Tensor,
    _ATEN.bitwise_or.Tensor,
    _CAST,
}
_VIEWS = {_ATEN.unsqueeze.default, _ATEN.permute.default, view_ops.subscript}
_UNKNOWN = object()


class _Support:
    def __init__(
        self, region: ContractionRegion, shapes: Mapping[Node, tuple[int, ...]]
    ) -> None:
        self.shapes = shapes
        self.dots = {spec.node: spec for spec in region.contractions}
        self.constants: dict[Node, object] = {}
        self.supports: dict[Node, torch.Tensor] = {}

    def exact(self, node: object) -> object:
        if not isinstance(node, Node):
            return _UNKNOWN if isinstance(node, torch.Tensor) else node
        if node in self.constants:
            return self.constants[node]
        result: object = _UNKNOWN
        if node.target is torch.ops.prims.iota.default:
            length = node.args[0]
            start, step = node.kwargs.get("start"), node.kwargs.get("step")
            dtype = node.kwargs.get("dtype")
            if (
                all(type(x) is int for x in (length, start, step))
                and cast("int", length) > 0
                and cast("int", step) != 0
                and dtype in (torch.int32, torch.int64)
            ):
                result = torch.arange(
                    cast("int", length), dtype=cast("torch.dtype", dtype)
                )
                result = result * cast("int", step) + cast("int", start)
        elif node.target in _EXACT | _VIEWS:
            if any(
                isinstance(value, torch.Tensor)
                for value in (*node.args, *node.kwargs.values())
            ):
                self.constants[node] = _UNKNOWN
                return _UNKNOWN
            if all(self.exact(item) is not _UNKNOWN for item in node.all_input_nodes):
                args: Any = map_arg(
                    node.args, lambda source: cast("Any", self.exact(source))
                )
                kwargs: Any = map_arg(
                    node.kwargs, lambda source: cast("Any", self.exact(source))
                )
                if node.target is _ATEN.scalar_tensor.default:
                    kwargs = {**kwargs, "device": "cpu"}
                if node.target in (_ATEN.div.Tensor_mode, _ATEN.remainder.Scalar) and (
                    not isinstance(args[0], torch.Tensor)
                    or args[0].dtype not in (torch.int32, torch.int64)
                    or type(args[1]) is not int
                    or args[1] == 0
                ):
                    self.constants[node] = _UNKNOWN
                    return _UNKNOWN
                if (
                    node.target in _VIEWS
                    and _source_coordinate(node, (0,) * len(self.shapes[node])) is None
                ):
                    result = _UNKNOWN
                elif node.target is view_ops.subscript:
                    selectors = args[1]
                    if all(x is None or x == slice(None) for x in selectors):
                        result = args[0][tuple(selectors)]
                else:
                    result = cast("Any", node.target)(*args, **kwargs)
        self.constants[node] = result
        return result

    def __call__(self, node: object) -> torch.Tensor:
        if not isinstance(node, Node):
            return (
                torch.as_tensor(node) != 0
                if type(node) in (int, float, bool)
                else torch.tensor(True)
            )
        if node in self.supports:
            return self.supports[node]
        # Scalar captures need not have tensor metadata. Their unknown value
        # has full scalar support; a runtime origin is never treated as iota.
        shape = self.shapes.get(node, ())
        exact = self.exact(node)
        result = torch.tensor(True)
        if exact is not _UNKNOWN:
            result = torch.as_tensor(exact) != 0
        elif node.target is _ATEN.where.self:
            predicate = self.exact(node.args[0])
            left, right = self(node.args[1]), self(node.args[2])
            result = (
                left | right
                if predicate is _UNKNOWN
                else torch.where(cast("torch.Tensor", predicate), left, right)
            )
        elif node.target in (_ATEN.add.Tensor, _ATEN.sub.Tensor):
            result = self(node.args[0]) | self(node.args[1])
        elif node.target is _ATEN.mul.Tensor:
            result = self(node.args[0]) & self(node.args[1])
        elif node.target in (_ATEN.neg.default, _CAST, _tracing_ops._new_var):
            result = self(node.args[0])
        elif node.target is _tracing_ops._mask_to:
            # Unknown domain can select the original fill. Only zero fill
            # cannot increase mathematical support; retain the original node.
            result = self(node.args[0]) | self(node.args[1])
        elif node.target is _ATEN.permute.default:
            result = self(node.args[0]).permute(cast("tuple[int, ...]", node.args[1]))
        elif node.target is _ATEN.unsqueeze.default:
            result = self(node.args[0]).unsqueeze(cast("int", node.args[1]))
        elif node.target is view_ops.subscript:
            selectors = cast("tuple[slice | None, ...]", node.args[1])
            if all(x is None or x == slice(None) for x in selectors):
                result = self(node.args[0])[tuple(selectors)]
        elif node in self.dots:
            spec = self.dots[node]
            # Existential products, not floating-point evaluation or inference
            # from a runtime value. Integer counts cannot cancel.
            result = self(spec.lhs).int() @ self(spec.rhs).int() != 0
            if spec.accumulator is not None:
                result |= self(spec.accumulator)
        result = result.broadcast_to(shape)
        self.supports[node] = result
        return result

    def triples(self, stage: int) -> tuple[tuple[int, int, int], ...]:
        spec = tuple(self.dots.values())[stage]
        left, right = self(spec.lhs), self(spec.rhs)
        m, k = left.shape
        _, n = right.shape
        triples = []
        for row, column, inner in product(
            range(0, m, 16), range(0, n, 16), range(0, k, 16)
        ):
            a = left[row : row + 16, inner : inner + 16]
            b = right[inner : inner + 16, column : column + 16]
            if bool((a[:, :, None] & b[None, :, :]).any()):
                triples.append((row, column, inner))
        # One interval per participating mode: no merged component can require
        # two K atoms or cross-warp exchange. Origins need not be equal.
        if any(
            len({triple[axis] for triple in triples}) != len(triples)
            for axis in range(3)
        ):
            return ()
        return tuple(triples)


def _coordinates(
    node: Node, coordinate: tuple[int, ...], shapes: Mapping[Node, tuple[int, ...]]
) -> tuple[int, ...]:
    shape = shapes[node]
    return tuple(
        0 if size == 1 else index
        for size, index in zip(
            shape, coordinate[len(coordinate) - len(shape) :], strict=True
        )
    )


def _source_coordinate(
    node: Node, coordinate: tuple[int, ...]
) -> tuple[int, ...] | None:
    if node.target is _ATEN.permute.default:
        permutation = node.args[1]
        if not isinstance(permutation, (tuple, list)) or any(
            type(axis) is not int for axis in permutation
        ):
            return None
        permutation = cast("tuple[int, ...]", permutation)
        if sorted(permutation) != list(range(len(coordinate))):
            return None
        return tuple(
            coordinate[permutation.index(axis)] for axis in range(len(coordinate))
        )
    if node.target is _ATEN.unsqueeze.default:
        axis = node.args[1]
        if type(axis) is not int or not -len(coordinate) <= axis < len(coordinate):
            return None
        axis %= len(coordinate)
        return coordinate[:axis] + coordinate[axis + 1 :]
    if node.target is view_ops.subscript:
        selectors = node.args[1]
        if (
            isinstance(selectors, (tuple, list))
            and len(selectors) == len(coordinate)
            and all(x is None or x == slice(None) for x in selectors)
        ):
            return tuple(
                index
                for index, selector in zip(coordinate, selectors, strict=True)
                if selector is not None
            )
    return None


def _candidate(
    revision: RegisterIslandRevision,
    groups: tuple[ContractionGroup, ...],
    support: _Support,
    triples: Mapping[int, tuple[tuple[int, int, int], ...]],
) -> RegisterIsland | None:
    region = revision.region
    shapes = support.shapes
    stages = tuple(stage for group in groups for stage in group.stages)
    selected = {region.contractions[stage].node: stage for stage in stages}
    positions = {node: index for index, node in enumerate(region.nodes)}
    start, end = positions[next(iter(selected))], positions[next(reversed(selected))]
    if any(
        start < positions[node] < end
        for node in (*region.stores, *region.scans, *region.reductions)
    ):
        return None
    boundaries = revision.entry_boundaries | (support.dots.keys() - selected.keys())
    internal: set[Node] = set()
    entries: set[Node] = set()

    def collect(node: Node) -> bool:
        if node in selected:
            return True
        if node in boundaries:
            entries.add(node)
            return positions[node] < start
        if node in internal:
            return True
        if support.exact(node) is not _UNKNOWN:
            return True
        if node.target not in _ARITHMETIC | _VIEWS or any(
            isinstance(x, Node) for x in node.kwargs.values()
        ):
            return False
        if node.kwargs and not (
            node.target in (_ATEN.add.Tensor, _ATEN.sub.Tensor)
            and set(node.kwargs) <= {"alpha"}
            and type(node.kwargs["alpha"]) in (int, float)
        ):
            return False
        if (
            isinstance(value := node.meta.get("val"), torch.Tensor)
            and value.dtype.is_floating_point
            and len(shapes[node]) != 2
        ):
            return False
        if (
            node.target in _VIEWS
            and _source_coordinate(node, (0,) * len(shapes[node])) is None
        ):
            return False
        internal.add(node)
        return all(collect(source) for source in node.all_input_nodes)

    if not all(collect(source) for node in selected for source in node.all_input_nodes):
        return None
    # Shape-only metadata uses are the sole allowed exception. Other effects
    # inside the ordered span cannot be moved across the replacement.
    deferred = tuple(
        node
        for node in region.nodes
        if start < positions[node] < end
        and node not in internal | selected.keys()
        and node.target in _ARITHMETIC | _VIEWS
    )
    if any(
        start < positions[node] < end
        and node not in internal | selected.keys() | set(deferred)
        and node.op not in ("placeholder", "output")
        and support.exact(node) is _UNKNOWN
        and not _shape_input(node)
        for node in region.nodes
    ):
        return None
    keys = tuple((stage, triple) for stage in stages for triple in triples[stage])
    parent = {key: key for key in keys}

    def root(key: tuple[int, tuple[int, int, int]]) -> tuple[int, tuple[int, int, int]]:
        while parent[key] != key:
            key = parent[key]
        return key

    def join(
        a: tuple[int, tuple[int, int, int]], b: tuple[int, tuple[int, int, int]]
    ) -> None:
        parent[root(b)] = root(a)

    output_owners = {
        (region.contractions[stage].node, row, column): (stage, triple)
        for stage, triple in keys
        for row, column in [(triple[0], triple[1])]
    }
    visited: dict[
        tuple[Node, tuple[int, ...]], frozenset[tuple[int, tuple[int, int, int]]]
    ] = {}
    cells: dict[tuple[Node, int], set[tuple[int, int]]] = {}
    recorded: set[tuple[Node, tuple[int, ...], int]] = set()

    def dependencies(
        node: Node, coordinate: tuple[int, ...]
    ) -> frozenset[tuple[int, tuple[int, int, int]]]:
        coordinate = _coordinates(node, coordinate, shapes)
        key = node, coordinate
        if key in visited:
            return visited[key]
        result: set[tuple[int, tuple[int, int, int]]] = set()
        if (
            not bool(support(node)[coordinate])
            or node in boundaries
            or support.exact(node) is not _UNKNOWN
        ):
            pass
        elif node in selected:
            owner = output_owners.get(
                (node, coordinate[0] // 16 * 16, coordinate[1] // 16 * 16)
            )
            if owner is not None:
                result.add(owner)
        else:
            sources = node.all_input_nodes
            if (
                node.target is _ATEN.where.self
                and support.exact(node.args[0]) is not _UNKNOWN
            ):
                condition = cast(
                    "torch.Tensor", support.exact(node.args[0])
                ).broadcast_to(shapes[node])
                branch = node.args[1 if bool(condition[coordinate]) else 2]
                sources = [branch] if isinstance(branch, Node) else []
            mapped = (
                _source_coordinate(node, coordinate)
                if node.target in _VIEWS
                else coordinate
            )
            assert mapped is not None
            for source in sources:
                result.update(dependencies(source, mapped))
        visited[key] = frozenset(result)
        return visited[key]

    for stage, triple in keys:
        spec = region.contractions[stage]
        row, column, inner = triple
        for node, origin in ((spec.lhs, (row, inner)), (spec.rhs, (inner, column))):
            for i, j in product(range(16), repeat=2):
                for owner in dependencies(node, (origin[0] + i, origin[1] + j)):
                    join((stage, triple), owner)
    components: dict[
        tuple[int, tuple[int, int, int]], list[tuple[int, tuple[int, int, int]]]
    ] = {}
    for key in keys:
        components.setdefault(root(key), []).append(key)
    if any(
        len(items) < 2 or len({stage for stage, _ in items}) != len(items)
        for items in components.values()
    ):
        return None
    geometries = {
        stage: geometry
        for group in groups
        for stage, geometry in zip(group.stages, group.geometries, strict=True)
    }
    ordered = tuple(tuple(items) for items in components.values())

    def record(node: Node, coordinate: tuple[int, ...], component: int) -> bool:
        coordinate = _coordinates(node, coordinate, shapes)
        if node in boundaries or support.exact(node) is not _UNKNOWN:
            return True
        if (node, coordinate, component) in recorded:
            return True
        recorded.add((node, coordinate, component))
        if len(coordinate) == 2:
            cells.setdefault((node, component), set()).add(
                (coordinate[0] // 16 * 16, coordinate[1] // 16 * 16)
            )
        if node in selected:
            return True
        mapped = (
            _source_coordinate(node, coordinate)
            if node.target in _VIEWS
            else coordinate
        )
        assert mapped is not None
        return all(
            record(source, mapped, component)
            for source in node.all_input_nodes
            if support.exact(source) is _UNKNOWN
        )

    for component, items in enumerate(ordered):
        for stage, (row, column, inner) in items:
            spec = region.contractions[stage]
            for node, origin in (
                (spec.node, (row, column)),
                (spec.lhs, (row, inner)),
                (spec.rhs, (inner, column)),
            ):
                for i, j in product(range(16), repeat=2):
                    record(node, (origin[0] + i, origin[1] + j), component)
    multiple = any(len(origins) != 1 for origins in cells.values())
    if multiple and not revision.multi_image:
        return None
    images: dict[Node, list[RegisterImage]] = {}
    if multiple:
        # Pair components by the same original stage/operand use, never by
        # sorting independent origin sets. All images remain warp-local.
        if any(
            tuple(stage for stage, _ in items)
            != tuple(stage for stage, _ in ordered[0])
            for items in ordered
        ) or any(
            node in selected and len(origins) != 1
            for (node, _), origins in cells.items()
        ):
            return None

        def image_use(
            node: Node, components: tuple[tuple[tuple[int, ...], ...], ...]
        ) -> bool:
            if node in boundaries or support.exact(node) is not _UNKNOWN:
                return True
            coordinates = tuple(
                tuple(_coordinates(node, point, shapes) for point in component)
                for component in components
            )
            if len(shapes[node]) == 2:
                origins = tuple(
                    {(point[0] // 16 * 16, point[1] // 16 * 16) for point in component}
                    for component in coordinates
                )
                if any(len(items) != 1 for items in origins):
                    return False
                image = RegisterImage(
                    node,
                    tuple((i, next(iter(items))) for i, items in enumerate(origins)),
                )
                current = images.setdefault(node, [])
                if image in current:
                    return True
                current.append(image)
            if node in selected:
                return True
            mapped = tuple(
                tuple(
                    _source_coordinate(node, point) if node.target in _VIEWS else point
                    for point in component
                )
                for component in coordinates
            )
            assert all(point is not None for component in mapped for point in component)
            return all(
                image_use(
                    source, cast("tuple[tuple[tuple[int, ...], ...], ...]", mapped)
                )
                for source in node.all_input_nodes
                if support.exact(source) is _UNKNOWN
            )

        for ordinal, (stage, _) in enumerate(ordered[0]):
            spec = region.contractions[stage]
            for node, axes in (
                (spec.node, (0, 1)),
                (spec.lhs, (0, 2)),
                (spec.rhs, (2, 1)),
            ):
                points = tuple(
                    tuple(
                        (items[ordinal][1][axes[0]] + i, items[ordinal][1][axes[1]] + j)
                        for i, j in product(range(16), repeat=2)
                    )
                    for items in ordered
                )
                if not image_use(node, points):
                    return None
        if any(
            {dict(image.tiles)[component] for image in images.get(node, ())} != origins
            for (node, component), origins in cells.items()
        ):
            return None
    body = internal | selected.keys()

    def escapes(node: Node, seen: set[Node]) -> bool:
        if node in seen:
            return False
        seen.add(node)
        return any(
            user not in selected
            and not _shape_input(user)
            and (user not in internal or escapes(user, seen))
            for user in node.users
        )

    exports = tuple(node for node in selected if escapes(node, set()))
    values = []
    for node in region.nodes:
        value = node.meta.get("val")
        if (
            node not in body
            or not isinstance(value, torch.Tensor)
            or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            continue
        users = tuple(user for user in node.users if not _shape_input(user))
        values.append(
            RegisterValue(
                node,
                value.dtype,
                shapes[node],
                positions[node],
                max((positions[user] for user in users), default=positions[node]),
                users,
                tuple(
                    sum(
                        1 << column for column, supported in enumerate(row) if supported
                    )
                    for row in support(node).tolist()
                ),
                images[node][0].tiles
                if multiple
                else tuple(
                    (component, next(iter(origins)))
                    for (current, component), origins in cells.items()
                    if current is node
                ),
                tuple(images[node][1:]) if multiple else (),
            )
        )
    return RegisterIsland(
        revision,
        groups,
        tuple(
            RegisterComponent(
                tuple(
                    RegisterIssue(
                        stage,
                        region.contractions[stage].node,
                        geometries[stage],
                        origins,
                    )
                    for stage, origins in items
                )
            )
            for items in ordered
        ),
        tuple(values),
        tuple(node for node in region.nodes if node in entries),
        deferred,
        exports,
        tuple(value.node for value in values),
    )


def plan_register_islands(
    region: ContractionRegion,
    groups: tuple[ContractionGroup, ...],
    shapes: Mapping[Node, tuple[int, ...]],
    *,
    fast_math: bool,
    entry_boundaries: frozenset[Node] = frozenset(),
    multi_image: bool = False,
) -> tuple[RegisterIsland, ...]:
    """Find bounded candidates; false policy is an immediate unchanged fallback.

    Groups must describe the region's complete original contraction order.
    Entry boundaries are already published *values*, not masks or permission
    to elide their original expressions. Their support is still graph-derived.
    """
    if type(fast_math) is not bool:
        raise TypeError("register islands require an explicit bool fast_math policy")
    if type(multi_image) is not bool:
        raise TypeError("register image selection must be bool")
    if not fast_math:
        return ()
    nodes = set(region.nodes)
    if (
        tuple(region.graph.nodes) != region.nodes
        or any(node.graph is not region.graph for node in nodes)
        or not entry_boundaries <= nodes
        or any(_contraction(spec.node) != spec for spec in region.contractions)
        or tuple(stage for group in groups for stage in group.stages)
        != tuple(range(len(region.contractions)))
        or any(
            len(group.stages) != len(group.geometries) or not group.stages
            for group in groups
        )
    ):
        return ()
    seen: set[Node] = set()
    for node in region.nodes:
        if any(source not in seen for source in node.all_input_nodes):
            return ()
        seen.add(node)
    resolved = tuple(
        (node, shapes.get(node))
        for node in region.nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    )
    if any(
        type(shape) is not tuple
        # A zero-trip loop can capture empty host inputs/outputs even though
        # its physical contraction tiles are nonempty. Record those semantic
        # shapes; positive per-dot geometry is checked separately below.
        or any(type(size) is not int or size < 0 for size in shape)
        for _, shape in resolved
    ):
        return ()
    resolved = cast("tuple[tuple[Node, tuple[int, ...]], ...]", resolved)
    if any(
        len(shape) != len(_domain(node.meta["val"]))
        or any(
            type(size) is int and size != resolved_size
            for size, resolved_size in zip(
                _domain(node.meta["val"]), shape, strict=True
            )
        )
        for node, shape in resolved
    ):
        return ()
    revision = RegisterIslandRevision(
        region,
        groups,
        resolved,
        entry_boundaries,
        _facts(region),
        multi_image,
    )
    with unset_fake_temporarily():
        support = _Support(region, shapes)
        triples = {}
        for stage, spec in enumerate(region.contractions):
            lhs, rhs, result = shapes[spec.lhs], shapes[spec.rhs], shapes[spec.node]
            if (
                len(lhs) == len(rhs) == len(result) == 2
                and lhs[1] == rhs[0]
                and result == (lhs[0], rhs[1])
                and all(size > 0 and size % 16 == 0 for size in (*lhs, *rhs))
                and spec.operand_dtypes[0] == spec.operand_dtypes[1]
                and spec.operand_dtypes[0] in (torch.float16, torch.bfloat16)
                and spec.result_dtype is torch.float32
                and spec.requested_out_dtype in (None, torch.float32)
                and spec.accumulator is None
            ):
                triples[stage] = support.triples(stage)
        runs: list[list[ContractionGroup]] = [[]]
        for group in groups:
            if all(
                triples.get(stage)
                and geometry.logical
                == (
                    shapes[region.contractions[stage].lhs][0],
                    shapes[region.contractions[stage].rhs][1],
                    shapes[region.contractions[stage].lhs][1],
                )
                for stage, geometry in zip(group.stages, group.geometries, strict=True)
            ):
                runs[-1].append(group)
            elif runs[-1]:
                runs.append([])
        result = []
        for run in runs:
            start = 0
            while start + 1 < len(run):
                for stop in range(len(run), start + 1, -1):
                    candidate = _candidate(
                        revision, tuple(run[start:stop]), support, triples
                    )
                    if candidate is not None:
                        result.append(candidate)
                        start = stop
                        break
                else:
                    start += 1
        return tuple(result)
