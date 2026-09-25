"""Uniquely owned FP32 exports from a resident contraction's existing scan."""

from __future__ import annotations

import dataclasses
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import scan_ops
from ...language import tile_ops
from ...language import view_ops
from ...language.matmul_ops import dot
from ..compile_environment import CompileEnvironment
from .fragment_epilogue import _has_fresh_output_allocation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_matmul import _ScanInput


@dataclasses.dataclass(frozen=True)
class ScanScalarExport:
    store: Node
    scan: Node
    index: int
    extent: int
    retained_axes: tuple[int, ...]
    masked_axes: tuple[int, ...]
    mask_nodes: frozenset[Node]


@dataclasses.dataclass(frozen=True)
class ScanVectorExport:
    store: Node
    scan: Node
    value: Node
    extent: int
    retained_axes: tuple[int, ...]
    masked_axes: tuple[int, ...]
    mask_nodes: frozenset[Node]
    vector_axis: int


ScanExport = ScanScalarExport | ScanVectorExport


@dataclasses.dataclass(frozen=True)
class ScanExportStores:
    primary: Node
    exports: tuple[ScanExport, ...]

    @property
    def permitted_nodes(self) -> frozenset[Node]:
        return frozenset(
            node
            for export in self.exports
            for node in (*export.mask_nodes, export.store)
        )


def _ancestors(node: Node) -> set[Node]:
    seen: set[Node] = set()
    pending = [node]
    while pending:
        current = pending.pop()
        if current not in seen:
            seen.add(current)
            pending.extend(current.all_input_nodes)
    return seen


def requests_scan_export(nodes: Sequence[Node]) -> bool:
    """Recognize the family before validating masks, dtypes or fresh targets.

    A malformed export must not fall through to ordinary scalar-store lowering.
    Unrelated multioutput roots, and every legacy single-output root, are left
    alone. In particular this is not a generic multioutput rejection rule.
    """
    stores = [node for node in nodes if node.target is memory_ops.store]
    if len(stores) < 2 or not any(node.target is dot for node in nodes):
        return False
    scan_leaves = {
        leaf
        for scan in nodes
        if scan.target is scan_ops._associative_scan
        for leaf in _ancestors(cast("Node", scan.args[1]))
        if leaf.target is memory_ops.load
    }
    return any(
        isinstance(value := store.args[2], Node)
        and isinstance(value.meta.get("val"), torch.Tensor)
        and not any(node.target is dot for node in _ancestors(value))
        and (
            any(node.target is scan_ops._associative_scan for node in _ancestors(value))
            or (value.meta["val"].ndim == 1 and bool(_ancestors(value) & scan_leaves))
        )
        for store in stores
    )


def _strip_identity(value: object) -> Node | None:
    while (
        isinstance(value, Node)
        and value.target is _tracing_ops._new_var
        and len(value.args) == 1
        and not value.kwargs
    ):
        value = value.args[0]
    return value if isinstance(value, Node) else None


def _block_id(node: object) -> int | None:
    if not isinstance(node, Node) or node.target is not _tracing_ops._get_symnode:
        return None
    env = CompileEnvironment.current()
    axis = env.resolve_block_id(node.meta.get("val"))
    return None if axis is None else env.canonical_block_id(axis)


def _begin_axis(node: object) -> int | None:
    if (
        not isinstance(node, Node)
        or node.target is not tile_ops.tile_begin
        or len(node.args) != 1
        or node.kwargs
    ):
        return None
    return _block_id(node.args[0])


def _first_tile_mask(mask: object) -> tuple[tuple[int, ...], frozenset[Node]] | None:
    axes: list[int] = []
    nodes: set[Node] = set()

    def visit(value: object) -> bool:
        if value is None or value is True:
            return True
        if not isinstance(value, Node) or value.kwargs or len(value.args) != 2:
            return False
        if value.target is operator.and_:
            nodes.add(value)
            return all(visit(arg) for arg in value.args)
        if value.target is not operator.eq:
            return False
        left, right = value.args
        if type(left) is int and left == 0:
            left, right = right, left
        axis = _begin_axis(left)
        if axis is None or type(right) is not int or right != 0 or axis in axes:
            return False
        nodes.add(value)
        axes.append(axis)
        return True

    return (tuple(axes), frozenset(nodes)) if visit(mask) else None


def _full_iota(index: object, extent: int) -> bool:
    if not isinstance(index, Node) or index.target is not torch.ops.prims.iota.default:
        return False
    if len(index.args) != 1 or type(index.args[0]) is not int:
        return False
    start, step = index.kwargs.get("start", 0), index.kwargs.get("step", 1)
    return (
        index.args[0] == extent
        and type(start) is int
        and start == 0
        and type(step) is int
        and step == 1
    )


def _classify_side(primary: Node, side: Node) -> ScanExport | None:
    from .chained_matmul import _additive_scan

    if len(side.args) != 4 or side.kwargs:
        return None
    value = _strip_identity(side.args[2])
    if (
        value is None
        or not isinstance(value.meta.get("val"), torch.Tensor)
        or value.meta["val"].dtype is not torch.float32
    ):
        return None
    primary_nodes = _ancestors(cast("Node", primary.args[2]))
    index: int | None = None
    if value.meta["val"].ndim == 0:
        if (
            value.target is not view_ops.subscript
            or len(value.args) != 2
            or value.kwargs
            or not isinstance(value.args[1], (tuple, list))
            or len(value.args[1]) != 1
            or type(value.args[1][0]) is not int
        ):
            return None
        index = cast("int", value.args[1][0])
        scan = _strip_identity(value.args[0])
    elif value.meta["val"].ndim == 1:
        if value.target is scan_ops._associative_scan:
            scan = value
        elif _scan_input_leaf(value) is not None:
            # A single-coordinate cast/clamp chain must itself already feed
            # the scan. Re-emit that exact typed expression, not a reconstruction.
            scan = next(
                (
                    node
                    for node in sorted(primary_nodes, key=lambda item: item.name)
                    if node.target is scan_ops._associative_scan
                    and value in _ancestors(cast("Node", node.args[1]))
                ),
                None,
            )
        else:
            return None
    else:
        return None
    if (
        scan is None
        or scan.target is not scan_ops._associative_scan
        or not _additive_scan(scan)
        or scan not in primary_nodes
    ):
        return None
    extent = scan.meta["val"].shape[0]
    if type(extent) is not int or extent <= 0:
        return None
    if index is not None and not 0 <= index < extent:
        return None
    if index is None and tuple(value.meta["val"].shape) != (extent,):
        return None
    mask = _first_tile_mask(side.args[3])
    if mask is None:
        return None
    masked_axes, mask_nodes = mask
    if not isinstance(side.args[1], (list, tuple)):
        return None
    retained: list[int] = []
    vector_axis: int | None = None
    for dim, subscript in enumerate(side.args[1]):
        if index is None and _full_iota(subscript, extent):
            if vector_axis is not None:
                return None
            vector_axis = dim
            continue
        axis = _begin_axis(subscript)
        if axis is None or axis in retained:
            return None
        retained.append(axis)
    if index is None:
        if vector_axis is None:
            return None
        return ScanVectorExport(
            side,
            scan,
            value,
            extent,
            tuple(retained),
            masked_axes,
            mask_nodes,
            vector_axis,
        )
    return ScanScalarExport(
        side, scan, index, extent, tuple(retained), masked_axes, mask_nodes
    )


def _scan_input_leaf(value: Node) -> Node | None:
    """Recognize one raw leaf through FP32-only casts and literal clamps.

    No arithmetic, coordinate views, scalar/tensor coefficients, narrowing,
    reductions or secondary memory leaf enter this export family.
    """
    node = _strip_identity(value)
    if node is None:
        return None
    tensor = node.meta.get("val")
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1:
        return None
    if node.target is memory_ops.load:
        return (
            node
            if tensor.dtype in (torch.float32, torch.bfloat16, torch.float16)
            else None
        )
    if tensor.dtype is not torch.float32 or node.kwargs:
        return None
    if node.target is torch.ops.prims.convert_element_type.default:
        if len(node.args) != 2 or node.args[1] is not torch.float32:
            return None
    elif node.target in (
        torch.ops.aten.clamp_min.default,
        torch.ops.aten.clamp_max.default,
        torch.ops.aten.clamp.default,
    ):
        expected = 3 if node.target is torch.ops.aten.clamp.default else 2
        if len(node.args) != expected or not all(
            bound is None
            or (
                type(bound) in (int, float)
                and math.isfinite(cast("int | float", bound))
            )
            for bound in node.args[1:]
        ):
            return None
    else:
        return None
    source = _strip_identity(node.args[0])
    if source is None or source.meta["val"].shape != tensor.shape:
        return None
    return _scan_input_leaf(source)


def classify_scan_exports(nodes: Sequence[Node]) -> ScanExportStores | None:
    """Classify complete projected stores and pairwise fresh FP32 outputs."""
    stores = [node for node in nodes if node.target is memory_ops.store]
    dots = {node for node in nodes if node.target is dot}
    if len(stores) < 2 or not dots:
        return None
    primaries = [
        store
        for store in stores
        if isinstance(store.args[2], Node) and dots <= _ancestors(store.args[2])
    ]
    if len(primaries) != 1:
        return None
    primary = primaries[0]
    exports: list[ScanExport] = []
    for side in stores:
        if side is primary:
            continue
        export = _classify_side(primary, side)
        if export is None:
            return None
        exports.append(export)
    allowed_masks = frozenset(node for export in exports for node in export.mask_nodes)
    if any(set(node.users) - allowed_masks - set(stores) for node in allowed_masks):
        return None
    targets = [store.args[0] for store in stores]
    if any(
        not isinstance(target, Node)
        or target.target is not _tracing_ops._host_tensor
        or not _has_fresh_output_allocation(target)
        or target.meta["val"].dtype is not torch.float32
        for target in targets
    ):
        return None
    targets = cast("list[Node]", targets)
    if any(
        first is second
        or first.args[0] == second.args[0]
        or first.meta["val"].untyped_storage() is second.meta["val"].untyped_storage()
        for i, first in enumerate(targets)
        for second in targets[i + 1 :]
    ):
        return None
    for export in exports:
        tensor = cast("Node", export.store.args[0]).meta["val"]
        if not tensor.is_contiguous() or tensor.ndim != len(export.retained_axes) + int(
            isinstance(export, ScanVectorExport)
        ):
            return None
    for node in nodes:
        if node.target is memory_ops.load:
            source = node.args[0]
            if isinstance(source, Node) and isinstance(
                source.meta.get("val"), torch.Tensor
            ):
                if any(
                    source is target
                    or source.meta["val"].untyped_storage()
                    is target.meta["val"].untyped_storage()
                    for target in targets
                ):
                    return None
    return ScanExportStores(primary, tuple(exports))


def _index_range(
    index: object, retained: set[int], axes: dict[int, tuple[int, int]]
) -> tuple[int, int] | None:
    if type(index) is int:
        return index, index
    axis = _begin_axis(index)
    if axis is not None and axis in retained:
        return 0, axes[axis][0] - 1
    if isinstance(index, Node) and index.target is torch.ops.prims.iota.default:
        count = index.args[0]
        start, step = index.kwargs.get("start", 0), index.kwargs.get("step", 1)
        if type(count) is int and count > 0 and start == 0 and step == 1:
            return 0, count - 1
    return None


def valid_scan_exports(plan: ChainedMatmulPlan) -> bool:
    """Prove projection, complete coverage and the existing scan's load domain."""
    from .chained_matmul import _shape

    if not plan.scan_exports:
        return True
    if plan.strategy != "tcgen05_tmem":
        return False
    axes = {axis: (extent, block) for axis, extent, block in plan.axes}
    for export in plan.scan_exports:
        retained = set(export.retained_axes)
        omitted = set(axes) - retained
        target = cast("Node", export.store.args[0]).meta["val"]
        shape = [axes[axis][0] for axis in export.retained_axes if axis in axes]
        if isinstance(export, ScanVectorExport):
            shape.insert(export.vector_axis, export.extent)
        if (
            not retained <= set(axes)
            or set(export.masked_axes) != omitted
            or any(axes[axis][1] != 1 for axis in retained)
            or tuple(target.shape) != tuple(shape)
            or _shape(export.scan) != (export.extent,)
        ):
            return False

        for node in _ancestors(cast("Node", export.scan.args[1])):
            if _block_id(node) in omitted:
                return False
            if node.target is not memory_ops.load:
                continue
            source, indices = node.args[:2]
            if (
                not isinstance(source, Node)
                or source.target is not _tracing_ops._host_tensor
                or not isinstance(indices, (list, tuple))
                or len(indices) != source.meta["val"].ndim
                or any(arg is not None for arg in node.args[2:])
                or node.kwargs
            ):
                return False
            for index, size in zip(indices, source.meta["val"].shape, strict=True):
                bounds = _index_range(index, retained, axes)
                if (
                    bounds is None
                    or type(size) is not int
                    or not 0 <= bounds[0] <= bounds[1] < size
                ):
                    return False
    return True


def codegen_scan_exports(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    scan_inputs: list[_ScanInput],
) -> list[str]:
    """Store after the matrix epilogue; no extra allocation or collective."""
    from .chained_matmul import _Expression
    from .chained_matmul import _indent

    lines: list[str] = []
    for ordinal, export in enumerate(plan.scan_exports):
        expression = _Expression(cg, plan, boundaries)
        expression.scan_inputs = scan_inputs
        target = expression.tensor_name(cast("Node", export.store.args[0]))
        coordinate = f"chain_export_{ordinal}_index"
        expression.coordinate_names.add(coordinate)
        vector = isinstance(export, ScanVectorExport)
        indices = expression.indices(export.store, (coordinate,) if vector else ())
        index_dtype = CompileEnvironment.current().backend.dtype_str(
            CompileEnvironment.current().index_dtype
        )
        offset = " + ".join(
            f"{index_dtype}({index}) * {index_dtype}({target}.layout.stride[{axis}])"
            for axis, index in enumerate(indices)
        )
        pointer = f"{target}.iterator" + (f" + {offset}" if offset else "")
        owner = " and ".join(
            [
                *(f"chain_origin_{axis} == 0" for axis in export.masked_axes),
                f"{coordinate} < {export.extent}" if vector else "chain_thread == 0",
            ]
        )
        if isinstance(export, ScanVectorExport):
            value = expression.value(export.value, (coordinate,))
            body = [
                f"{coordinate} = chain_thread + chain_export_{ordinal}_step * {plan.threads}",
                f"if {owner}:",
                _indent(
                    [*expression.lines, f"({pointer}).store(cutlass.Float32({value}))"]
                ),
            ]
            lines.extend(
                [
                    f"for chain_export_{ordinal}_step in cutlass.range_constexpr({(export.extent + plan.threads - 1) // plan.threads}):",
                    _indent(body),
                ]
            )
            continue
        lines.extend(
            [
                f"if {owner}:",
                f"    ({pointer}).store(cutlass.Float32({boundaries[export.scan]}[{export.index}]))",
            ]
        )
    return lines
