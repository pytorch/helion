"""Complete fresh output tiles owned by every root CTA axis.

Unlike projected scan exports, these stores omit no grid axis. Their original
typed values, masked input loads and coordinate domains remain ordinary chained
expressions; this proof only establishes unique complete output ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import scan_ops
from ...language import tile_ops
from ...language.matmul_ops import dot
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from .chained_scan_export import _ancestors
from .chained_scan_export import _block_id
from .chained_scan_export import _full_iota
from .fragment_epilogue import _has_fresh_output_allocation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_matmul import _ScanInput
    from .chained_matmul import _StagedInput


@dataclass(frozen=True)
class RootExport:
    store: Node
    # One entry per target dimension: a retained CTA axis or a full value axis.
    axes: tuple[int | None, ...]
    shape: tuple[int, ...]


@dataclass(frozen=True)
class RootExports:
    primary: Node
    stores: tuple[RootExport, ...]


def _classify_store(
    store: Node, axes: set[int], *, resolved_shape: tuple[int, ...] | None = None
) -> RootExport | None:
    if len(store.args) != 4 or store.args[3] is not None or store.kwargs:
        return None
    target, indices, value = store.args[:3]
    if (
        not isinstance(target, Node)
        or target.target is not _tracing_ops._host_tensor
        or not _has_fresh_output_allocation(target)
        or not isinstance(value, Node)
        or not isinstance(value.meta.get("val"), torch.Tensor)
        or not isinstance(indices, (tuple, list))
    ):
        return None
    tensor, result = target.meta["val"], value.meta["val"]
    shape = tuple(result.shape) if resolved_shape is None else resolved_shape
    if resolved_shape is not None:
        from .chained_matmul import _shape

        # Only an exact current codegen shape may resolve symbolic full slices.
        # The original discovery caller keeps its static-shape-only admission.
        if resolved_shape != _shape(value):
            return None
    if (
        result.ndim not in (1, 2)
        or tensor.dtype != result.dtype
        or tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or not tensor.is_contiguous()
        or len(indices) != tensor.ndim
        or len(shape) != result.ndim
        or any(type(size) is not int or size <= 0 for size in shape)
    ):
        return None
    retained: list[int | None] = []
    full = 0
    for selector, size in zip(indices, tensor.shape, strict=True):
        axis = (
            _block_id(selector.args[0])
            if isinstance(selector, Node)
            and selector.target in (tile_ops.tile_begin, tile_ops.tile_id)
            and len(selector.args) == 1
            and not selector.kwargs
            else None
        )
        if axis is not None:
            if axis not in axes or axis in retained:
                return None
            retained.append(axis)
        elif (
            full < result.ndim
            and type(size) is int
            and size == shape[full]
            and (selector == slice(None) or _full_iota(selector, size))
        ):
            retained.append(None)
            full += 1
        else:
            return None
    if set(retained) - {None} != axes or full != result.ndim:
        return None
    return RootExport(store, tuple(retained), shape)


def classify_root_exports(nodes: Sequence[Node]) -> RootExports | None:
    ir = HostFunction.current().device_ir
    if len(ir.grid_block_ids) != 1:
        return None
    env = CompileEnvironment.current()
    axes = {env.canonical_block_id(axis) for axis in ir.grid_block_ids[0]}
    stores = [node for node in nodes if node.target is memory_ops.store]
    dots = {node for node in nodes if node.target is dot}
    if len(stores) < 2 or not dots:
        return None
    primaries = [
        store
        for store in stores
        if isinstance(store.args[2], Node) and dots <= _ancestors(store.args[2])
    ]
    separate_contraction_outputs = not primaries
    if separate_contraction_outputs:
        # Every contraction must reach a fresh matrix export. Independent
        # typed exports need not share one homogeneous aggregate value.
        matrix_outputs = [
            store
            for store in stores
            if isinstance(store.args[2], Node)
            and isinstance(store.args[2].meta.get("val"), torch.Tensor)
            and store.args[2].meta["val"].ndim == 2
            and dots & _ancestors(store.args[2])
        ]
        if len(matrix_outputs) < 2:
            return None
        primary_ancestors = set().union(
            *(_ancestors(cast("Node", store.args[2])) for store in matrix_outputs)
        )
        if not dots <= primary_ancestors:
            return None
        primary = matrix_outputs[0]
    else:
        if len(primaries) != 1:
            return None
        primary = primaries[0]
        primary_value = cast("Node", primary.args[2])
        if primary_value.meta["val"].ndim != 2:
            return None
        primary_ancestors = _ancestors(primary_value)
    exports: list[RootExport] = []
    targets: list[Node] = []
    for store in stores:
        export = _classify_store(store, axes)
        if export is None:
            return None
        if store is not primary:
            ancestors = _ancestors(cast("Node", store.args[2]))
            if (
                dots & ancestors
                and (not separate_contraction_outputs or len(export.shape) != 2)
            ) or any(
                node.target is scan_ops._associative_scan
                and node not in primary_ancestors
                for node in ancestors
            ):
                return None
        exports.append(export)
        targets.append(cast("Node", store.args[0]))
    storage = [target.meta["val"].untyped_storage() for target in targets]
    if len({id(item) for item in storage}) != len(storage):
        return None
    if any(
        isinstance(source := node.args[0], Node)
        and isinstance(source.meta.get("val"), torch.Tensor)
        and any(source.meta["val"].untyped_storage() is item for item in storage)
        for node in nodes
        if node.target is memory_ops.load
    ):
        return None
    return RootExports(primary, tuple(exports))


def valid_root_exports(plan: ChainedMatmulPlan) -> bool:
    if plan.root_exports is None:
        return True
    if plan.loop is not None or plan.region is None:
        return False
    # Recheck the complete graph; eligibility never relies on a stale receipt.
    if classify_root_exports(plan.region.nodes) != plan.root_exports:
        return False
    axes = {axis: (extent, block) for axis, extent, block in plan.axes}
    for export in plan.root_exports.stores:
        tensor = cast("Node", export.store.args[0]).meta["val"]
        if {axis for axis in export.axes if axis is not None} != set(axes):
            return False
        for size, axis in zip(tensor.shape, export.axes, strict=True):
            if axis is not None and (axes[axis][1] != 1 or size != axes[axis][0]):
                return False
    return True


def codegen_root_exports(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    boundaries: dict[Node, str],
    staged_inputs: list[_StagedInput],
    scan_inputs: list[_ScanInput],
) -> list[str]:
    from .chained_matmul import _emit_store
    from .chained_matmul import _Expression
    from .chained_matmul import _indent
    from .chained_matmul import _store_domain

    if plan.root_exports is None:
        return []
    lines: list[str] = []
    for ordinal, export in enumerate(plan.root_exports.stores):
        if export.store is plan.store:
            continue
        prefix = f"chain_root_export_{ordinal}"
        if len(export.shape) == 2:
            lines.extend(
                _emit_store(
                    cg,
                    plan,
                    export.store,
                    boundaries,
                    staged_inputs,
                    scan_inputs,
                    prefix,
                )
            )
            continue
        expression = _Expression(cg, plan, boundaries)
        expression.staged_inputs = staged_inputs
        expression.scan_inputs = scan_inputs
        expression.coordinate_names.add(prefix)
        coordinates = (prefix,)
        vector_scope = None
        if plan.root_vector_cache is not None:
            from .chained_root_vector_cache import VectorExportScope
            from .chained_root_vector_cache import bind_read_scope

            vector_scope = VectorExportScope(
                plan, ordinal, export, prefix, export.shape[0]
            )
            bind_read_scope(expression, vector_scope)
        value = expression.value(cast("Node", export.store.args[2]), coordinates)
        indices = expression.indices(export.store, coordinates)
        target = expression.tensor_name(cast("Node", export.store.args[0]))
        bounds = _store_domain(cg, export.store, coordinates, plan)
        body = [*expression.lines, f"{target}[{', '.join(indices)}] = {value}"]
        if bounds:
            body = [f"if {' and '.join(bounds)}:", _indent(body)]
        extent = export.shape[0]
        if vector_scope is not None:
            from .chained_root_vector_cache import check_read_scope

            check_read_scope(expression)
            lines.extend(vector_scope.wrap(body))
            continue
        lines.extend(
            [
                f"for {prefix}_step in cutlass.range_constexpr({(extent + plan.threads - 1) // plan.threads}):",
                f"    {prefix} = chain_thread + {prefix}_step * {plan.threads}",
                f"    if {prefix} < {extent}:",
                _indent(body, 8),
            ]
        )
    return lines
