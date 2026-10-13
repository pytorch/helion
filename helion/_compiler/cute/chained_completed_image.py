"""Proofs for partially published, completed typed matrix compositions.

This module does not authorize native issue or replace an existing publication.
It records exact static view/cast provenance, disjoint publication regions and
all evaluated physical operand addresses. A caller must bind the original
publication joins and operand domains before consuming these records.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import view_ops
from .chained_root_native import joined_root
from .chained_root_native import native_facts
from .chained_split_join import supports as split_join_supports
from .chained_store_expression import _node_facts

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Mapping

    from .chained_matmul import ChainedMatmulPlan


@dataclass(frozen=True)
class CompletedSource:
    node: Node
    shape: tuple[int, int]
    # Row islands publish before this stage; contraction outputs after stage-1.
    ready_before_stage: int
    kind: str


@dataclass(frozen=True)
class ImageCell:
    """One original FP32 value and its unchanged conversion path; None is +0."""

    source: Node | None
    index: int
    casts: tuple[torch.dtype, ...] = ()


@dataclass(frozen=True)
class _TensorImage:
    shape: tuple[int, ...]
    dtype: torch.dtype
    cells: tuple[ImageCell, ...]

    def at(self, coordinates: tuple[int, ...]) -> ImageCell:
        if len(coordinates) != len(self.shape) or any(
            not 0 <= coordinate < extent
            for coordinate, extent in zip(coordinates, self.shape, strict=True)
        ):
            raise ValueError("completed image coordinate outside its logical source")
        offset = sum(
            coordinate * math.prod(self.shape[axis + 1 :])
            for axis, coordinate in enumerate(coordinates)
        )
        return self.cells[offset]


class _NotImage(Exception):
    pass


def _shape(node: Node) -> tuple[int, ...]:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor) or any(
        type(size) is not int or size <= 0 for size in value.shape
    ):
        raise _NotImage("image needs exact positive static tensor extents")
    return tuple(value.shape)


def _coordinates(shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    return product(*(range(extent) for extent in shape))


class ImageProvenance:
    """One read-only graph analysis; no cached result survives graph mutation."""

    def __init__(
        self, sources: tuple[CompletedSource, ...], *, maximum_elements: int
    ) -> None:
        if type(maximum_elements) is not int or maximum_elements <= 0:
            raise ValueError("invalid completed image proof budget")
        self.sources = {source.node: source for source in sources}
        if len(self.sources) != len(sources):
            raise ValueError("duplicate completed image source")
        self.maximum_elements = maximum_elements
        self.memo: dict[Node, _TensorImage | None] = {}
        self.rejections: dict[Node, str] = {}

    def image(self, node: Node) -> _TensorImage | None:
        if node not in self.memo:
            try:
                self.memo[node] = self._image(node)
            except _NotImage as error:
                self.memo[node] = None
                self.rejections[node] = str(error)
        return self.memo[node]

    def _input(self, value: object) -> _TensorImage:
        if not isinstance(value, Node):
            raise _NotImage("non-tensor image input")
        result = self.image(value)
        if result is None:
            raise _NotImage("image depends on an unproved operation")
        return result

    def _image(self, node: Node) -> _TensorImage:
        shape = _shape(node)
        size = math.prod(shape)
        if size > self.maximum_elements:
            raise _NotImage("completed image proof budget exceeded")
        dtype = node.meta["val"].dtype
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise _NotImage("unsupported completed image dtype")
        if node in self.sources:
            source = self.sources[node]
            if dtype != torch.float32 or shape != source.shape:
                raise _NotImage("completed source type or shape changed")
            return _TensorImage(
                shape, dtype, tuple(ImageCell(node, index) for index in range(size))
            )
        if node.op != "call_function":
            raise _NotImage("image is not a completed value composition")
        target = node.target
        if target is torch.ops.aten.full.default:
            if not split_join_supports(node):
                raise _NotImage("unproved constant image")
            fill = node.args[1]
            if (
                type(fill) not in (int, float)
                or fill != 0
                or (isinstance(fill, float) and math.copysign(1.0, fill) < 0)
            ):
                raise _NotImage("only original positive-zero regions are admitted")
            return _TensorImage(shape, dtype, (ImageCell(None, 0),) * size)
        if target is operator.getitem:
            if not split_join_supports(node):
                raise _NotImage("unproved split selection")
            split = cast("Node", node.args[0])
            source = self._input(split.args[0])
            index = cast("int", node.args[1])
            return _TensorImage(
                shape,
                dtype,
                tuple(source.at((*coords, index)) for coords in _coordinates(shape)),
            )
        if target is view_ops.join:
            if not split_join_supports(node):
                raise _NotImage("unproved static join")
            left, right = (self._input(arg) for arg in node.args)
            return _TensorImage(
                shape,
                dtype,
                tuple(
                    cell
                    for pair in zip(left.cells, right.cells, strict=True)
                    for cell in pair
                ),
            )
        if (
            node.kwargs
            and not (
                target is torch.ops.aten.clone.default
                and set(node.kwargs) == {"memory_format"}
                and node.kwargs["memory_format"]
                in (torch.contiguous_format, torch.preserve_format)
            )
        ) or not node.args:
            raise _NotImage("unsupported image keyword arguments")
        source = self._input(node.args[0])
        if target is torch.ops.prims.convert_element_type.default:
            if (
                len(node.args) != 2
                or node.args[1] != dtype
                or source.shape != shape
                or source.dtype != torch.float32
                or dtype not in (torch.float16, torch.bfloat16)
                or any(cell.casts for cell in source.cells)
            ):
                raise _NotImage("image requires the original single FP32-to-half cast")
            return _TensorImage(
                shape,
                dtype,
                tuple(
                    ImageCell(cell.source, cell.index, (dtype,))
                    for cell in source.cells
                ),
            )
        if source.dtype != dtype:
            raise _NotImage("view changed dtype")
        if target in (_tracing_ops._new_var, torch.ops.aten.clone.default):
            if len(node.args) != 1 or shape != source.shape:
                raise _NotImage("unproved identity image")
            return _TensorImage(shape, dtype, source.cells)
        if target in (
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten._unsafe_view.default,
        ):
            if (
                len(node.args) != 2
                or size != len(source.cells)
                or not isinstance(node.args[1], (tuple, list))
                or tuple(node.args[1]) != shape
            ):
                raise _NotImage("unproved complete reshape")
            return _TensorImage(shape, dtype, source.cells)
        if target is torch.ops.aten.permute.default:
            if len(node.args) != 2 or not isinstance(node.args[1], (tuple, list)):
                raise _NotImage("unproved permutation")
            order = tuple(node.args[1])
        elif target is torch.ops.aten.t.default:
            if len(node.args) != 1 or len(shape) != 2:
                raise _NotImage("unproved matrix transpose")
            order = (1, 0)
        elif target is torch.ops.aten.transpose.int:
            if len(node.args) != 3 or any(
                type(axis) is not int for axis in node.args[1:]
            ):
                raise _NotImage("unproved transpose axes")
            order_list = list(range(len(shape)))
            left, right = cast("tuple[int, int]", node.args[1:])
            if not -len(shape) <= left < len(shape) or not -len(shape) <= right < len(
                shape
            ):
                raise _NotImage("transpose axes outside source rank")
            order_list[left], order_list[right] = order_list[right], order_list[left]
            order = tuple(order_list)
        else:
            raise _NotImage(
                "arithmetic, masks, loads and arbitrary indexing stay on their original path"
            )
        if len(order) != len(shape) or any(type(axis) is not int for axis in order):
            raise _NotImage("unproved permutation rank")
        integer_order = cast("tuple[int, ...]", order)
        if any(not -len(shape) <= axis < len(shape) for axis in integer_order):
            raise _NotImage("permutation axis outside source rank")
        order = tuple(axis % len(shape) for axis in integer_order)
        if sorted(order) != list(range(len(shape))) or shape != tuple(
            source.shape[axis] for axis in order
        ):
            raise _NotImage("permutation shape mismatch")
        return _TensorImage(
            shape,
            dtype,
            tuple(
                source.at(
                    tuple(coords[order.index(axis)] for axis in range(len(shape)))
                )
                for coords in _coordinates(shape)
            ),
        )


@dataclass(frozen=True)
class ImagePublication:
    source: CompletedSource
    image_origin: tuple[int, int]
    source_origin: tuple[int, int]
    extent: tuple[int, int]

    def locations(self, image_columns: int) -> tuple[tuple[int, int], ...]:
        """Destination/source offsets; only logical rows, never native padding."""
        return tuple(
            (
                (self.image_origin[0] + row) * image_columns
                + self.image_origin[1]
                + column,
                (self.source_origin[0] + row) * self.source.shape[1]
                + self.source_origin[1]
                + column,
            )
            for row, column in _coordinates(self.extent)
        )


@dataclass(frozen=True)
class CompletedOperandView:
    stage: int
    operand_index: int
    operand: Node
    physical_shape: tuple[int, int]
    origin: tuple[int, int]
    period: tuple[int, int]
    transpose: bool
    addresses: tuple[int, ...]

    def layout(self, image_columns: int) -> str:
        """Native logical layout only; issue additionally needs late domain proof."""
        strides = (1, image_columns) if self.transpose else (image_columns, 1)
        modes = tuple(
            str(extent) if extent == period else repr((period, extent // period))
            for extent, period in zip(self.physical_shape, self.period, strict=True)
        )
        steps = tuple(
            str(stride) if extent == period else repr((stride, 0))
            for extent, period, stride in zip(
                self.physical_shape, self.period, strides, strict=True
            )
        )
        return f"cute.make_layout(({', '.join(modes)}), stride=({', '.join(steps)}))"


@dataclass(frozen=True)
class CompletedImage:
    node: Node
    shape: tuple[int, int]
    dtype: torch.dtype
    cells: tuple[ImageCell, ...]
    publications: tuple[ImagePublication, ...]
    operands: tuple[CompletedOperandView, ...]
    facts: tuple[object, ...]
    plan: ChainedMatmulPlan
    plan_facts: tuple[object, ...]
    _selection: tuple[object, ...]

    @property
    def allocated_bytes(self) -> int:
        return 2 * math.prod(self.shape)

    def selection(self) -> tuple[object, ...]:
        return (
            self.node,
            self.shape,
            self.dtype,
            self.cells,
            self.publications,
            self.operands,
            self.facts,
            self.plan,
            self.plan_facts,
        )

    def check(self, plan: ChainedMatmulPlan) -> None:
        if plan.row_registers is not None:
            plan.row_registers.check(plan)
        if plan.row_parallel is not None:
            plan.row_parallel.check(plan)
        if (
            plan is not self.plan
            or plan.region is None
            or self._selection != self.selection()
            or self.facts != _node_facts(tuple(plan.region.graph.nodes))
            or self.plan_facts != _plan_facts(plan)
        ):
            raise ValueError("completed image selection or graph changed")

    def summary(self) -> dict[str, object]:
        return {
            "node": self.node.name,
            "shape": self.shape,
            "dtype": str(self.dtype),
            "added_shared_bytes": self.allocated_bytes,
            "zeros": sum(cell.source is None for cell in self.cells),
            "publications": [
                {
                    "node": item.source.node.name,
                    "kind": item.source.kind,
                    "ready_before_stage": item.source.ready_before_stage,
                    "source_shape": item.source.shape,
                    "image_origin": item.image_origin,
                    "source_origin": item.source_origin,
                    "extent": item.extent,
                }
                for item in self.publications
            ],
            "operands": [
                {
                    "stage": item.stage,
                    "index": item.operand_index,
                    "operand": item.operand.name,
                    "shape": item.physical_shape,
                    "origin": item.origin,
                    "period": item.period,
                    "transpose": item.transpose,
                    "layout": item.layout(self.shape[1]),
                    "evaluated_addresses": len(item.addresses),
                }
                for item in self.operands
            ],
        }


def _publications(
    image: _TensorImage, sources: Mapping[Node, CompletedSource]
) -> tuple[ImagePublication, ...] | None:
    columns = image.shape[1]
    result = []
    for node in dict.fromkeys(
        cell.source for cell in image.cells if cell.source is not None
    ):
        assert node is not None
        positions = [
            index for index, cell in enumerate(image.cells) if cell.source is node
        ]
        rows = [index // columns for index in positions]
        cols = [index % columns for index in positions]
        origin = (min(rows), min(cols))
        extent = (max(rows) - origin[0] + 1, max(cols) - origin[1] + 1)
        if math.prod(extent) != len(positions):
            return None
        source = sources[node]
        first = image.cells[origin[0] * columns + origin[1]].index
        source_origin = divmod(first, source.shape[1])
        if any(
            start + count > bound
            for start, count, bound in zip(
                source_origin, extent, source.shape, strict=True
            )
        ):
            return None
        publication = ImagePublication(source, origin, source_origin, extent)
        if any(
            image.cells[dst] != ImageCell(node, src, (image.dtype,))
            for dst, src in publication.locations(columns)
        ):
            return None
        result.append(publication)
    return tuple(result)


def _view(
    image: _TensorImage, operand: _TensorImage, node: Node, stage: int, index: int
) -> CompletedOperandView | None:
    if len(operand.shape) != 2 or operand.dtype != image.dtype:
        return None
    shape = cast(
        "tuple[int, int]", operand.shape if index == 0 else operand.shape[::-1]
    )
    cells = (
        operand.cells
        if index == 0
        else tuple(operand.at((column, row)) for row, column in _coordinates(shape))
    )
    anchor = next((i for i, cell in enumerate(cells) if cell.source is not None), None)
    if anchor is None:
        return None
    positions = [i for i, cell in enumerate(image.cells) if cell == cells[anchor]]
    if len(positions) != 1:
        return None
    anchor_row, anchor_column = divmod(anchor, shape[1])
    image_row, image_column = divmod(positions[0], image.shape[1])
    periods = tuple(
        tuple(n for n in range(1, extent + 1) if extent % n == 0) for extent in shape
    )
    for period, transpose in product(product(*periods), (False, True)):
        row, column = anchor_row % period[0], anchor_column % period[1]
        if transpose:
            row, column = column, row
        origin = image_row - row, image_column - column
        height, width = period[::-1] if transpose else period
        if (
            min(origin) < 0
            or origin[0] + height > image.shape[0]
            or origin[1] + width > image.shape[1]
        ):
            continue
        addresses = []
        for offset, (row, column) in enumerate(_coordinates(shape)):
            row, column = row % period[0], column % period[1]
            if transpose:
                row, column = column, row
            address = (origin[0] + row) * image.shape[1] + origin[1] + column
            if image.cells[address] != cells[offset]:
                break
            addresses.append(address)
        else:
            return CompletedOperandView(
                stage,
                index,
                node,
                shape,
                origin,
                cast("tuple[int, int]", period),
                transpose,
                tuple(addresses),
            )
    return None


def _plan_facts(plan: ChainedMatmulPlan) -> tuple[object, ...]:
    return (
        plan.root_graph_id,
        plan.dots,
        plan.store,
        plan.axes,
        plan.shapes,
        plan.dtype,
        plan.threads,
        plan.strategy,
        native_facts(plan),
        plan.loop,
        plan.contraction_groups,
        plan.row_registers,
        plan.row_parallel,
        plan.pointwise_cache,
        plan.root_vector_cache,
        plan.warp_workspace,
        plan.warp_result_transport,
        plan.root_typed_operands,
        plan.k_schedule,
        plan.preparation_pipeline,
        tuple(plan.tensor_aliases.items()),
    )


def discover_completed_images(
    plan: ChainedMatmulPlan, *, capacity_bytes: int = 128 * 1024
) -> tuple[CompletedImage, ...]:
    """Discover disjoint, single-cast matrix compositions in ordinary warp roots.

    The finite proof budget is an analysis/storage limit, not shape dispatch.
    It is not sufficient workspace accounting or native descriptor admission.
    """
    if type(capacity_bytes) is not int or capacity_bytes <= 0:
        raise ValueError("invalid completed image capacity")
    if plan.region is None or plan.loop is not None or not joined_root(plan):
        return ()
    try:
        sources = tuple(
            CompletedSource(
                node, cast("tuple[int, int]", _shape(node)), stage + 1, "contraction"
            )
            for stage, node in enumerate(plan.dots)
        )
    except _NotImage:
        return ()
    rows = plan.row_registers
    if plan.row_parallel is not None:
        plan.row_parallel.check(plan)
    if rows is not None:
        rows.check(plan)
        sources += tuple(
            CompletedSource(item.output, item.shape, item.stage, "row")
            for item in rows.islands
        )
    analysis = ImageProvenance(sources, maximum_elements=capacity_bytes // 2)
    graph_nodes = tuple(plan.region.graph.nodes)
    candidates = []
    for node in graph_nodes:
        # Members may already carry their original FP32-to-half conversion
        # before the complete join/view. Candidate discovery must include that
        # typed composition; ImageProvenance still proves every original cast
        # and rejects arithmetic, repeated conversions and unknown operations.
        value = node.meta.get("val")
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or value.dtype not in (torch.float16, torch.bfloat16)
        ):
            continue
        image = analysis.image(node)
        if (
            image is None
            or len(image.shape) != 2
            or image.dtype not in (torch.float16, torch.bfloat16)
        ):
            continue
        publications = _publications(image, analysis.sources)
        if publications is None or len(publications) < 2:
            continue
        uses = []
        for stage, dot in enumerate(plan.dots):
            for index, operand_node in enumerate(dot.args[:2]):
                if not isinstance(operand_node, Node):
                    continue
                operand = analysis.image(operand_node)
                if operand is None:
                    continue
                view = _view(image, operand, operand_node, stage, index)
                if view is not None and all(
                    analysis.sources[cast("Node", cell.source)].ready_before_stage
                    <= stage
                    for address in view.addresses
                    if (cell := image.cells[address]).source is not None
                ):
                    uses.append(view)
        if uses:
            candidates.append((node, image, publications, tuple(uses)))
    # A larger exact composition subsumes the same operand uses; it is chosen
    # only as a deterministic proof proposal, not a performance preference.
    candidates.sort(key=lambda item: -len(item[1].cells))
    selected = []
    covered = set()
    facts = _node_facts(graph_nodes)
    for node, image, publications, uses in candidates:
        uses = tuple(
            use for use in uses if (use.stage, use.operand_index) not in covered
        )
        if (
            not uses
            or sum(item.allocated_bytes for item in selected) + len(image.cells) * 2
            > capacity_bytes
        ):
            continue
        fields = (
            node,
            cast("tuple[int, int]", image.shape),
            image.dtype,
            image.cells,
            publications,
            uses,
            facts,
            plan,
            _plan_facts(plan),
        )
        selected.append(CompletedImage(*fields, fields))
        covered.update((use.stage, use.operand_index) for use in uses)
    return tuple(selected)
