"""Retile a private grid-row axis of a shared, row-separable carry phase."""

from __future__ import annotations

import ast
import operator
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ... import exc
from ...language import _tracing_ops
from ...language import creation_ops
from ...language import memory_ops
from ...language import tile_ops
from ...runtime.config import Config
from ..ast_extension import ExtendedAST
from ..compile_environment import CompileEnvironment
from ..compile_environment import LoopSpecBlockSizeSource
from ..source_location import SourceLocation
from .autotune.config_keys import COMPACT_ROWS_KEY
from .fragment_vector_io import static_int
from .fragment_vector_io import static_strides
from .register_loads import host_storages_may_alias
from .warp_carry_packet import uses_warp_carry

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...autotuner.config_spec import ConfigSpec
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from ..host_function import HostFunction


def _tile_memory_indices(
    roots: list[ast.For], axis: int
) -> dict[tuple[object, SourceLocation, int], set[int]]:
    """Which memory subscripts denoted a tile before tracing replaced it by its size?"""
    from ..type_info import CallableType
    from ..type_info import TileIndexType

    result: dict[tuple[object, SourceLocation, int], set[int]] = {}
    for root in roots:
        for site in ast.walk(root):
            index: ast.expr
            if isinstance(site, ast.Subscript) and isinstance(site.ctx, ast.Load):
                op, index = memory_ops.load, site.slice
            elif (
                isinstance(site, ast.Assign)
                and len(site.targets) == 1
                and isinstance(target := site.targets[0], ast.Subscript)
            ):
                op, index = memory_ops.store, target.slice
            elif (
                isinstance(site, ast.Call)
                and isinstance(site.func, ExtendedAST)
                and isinstance(callee := site.func._type_info, CallableType)
                and callee.value in (memory_ops.load, memory_ops.store)
                and len(site.args) >= 2
            ):
                op, index = callee.value, site.args[1]
            else:
                continue
            if not isinstance(site, ExtendedAST) or not site._location.lineno:
                continue
            items = index.elts if isinstance(index, (ast.Tuple, ast.List)) else [index]
            positions = {
                i
                for i, item in enumerate(items)
                if isinstance(item, ExtendedAST)
                and isinstance(info := item._type_info, TileIndexType)
                and info.block_id == axis
            }
            # Rewrites can copy source locations. Only facts shared by every
            # matching site may justify an otherwise ambiguous FX index.
            key = (op, site._location, len(items))
            result.setdefault(key, positions).intersection_update(positions)
    return result


def complete_row_extent(
    env: CompileEnvironment,
    ir: DeviceIR,
    root: int,
    region: list[GraphInfo],
    axis: int,
    *,
    allow_unbound: bool = False,
) -> int | None:
    """Only a private, leading, tunable grid axis with row-local tensor uses."""
    info = env.block_sizes[axis]
    extent = static_int(info.size)
    if (
        extent is None
        or extent <= 0
        or extent % 3
        or axis in ir.noncanonical_task_origin_block_ids
        or not isinstance(info.block_size_source, LoopSpecBlockSizeSource)
        or [
            rid
            for rid, grid in zip(ir.root_ids, ir.grid_block_ids, strict=True)
            if axis in grid
        ]
        != [root]
    ):
        return None

    def row_dims(v: object) -> list[int]:
        return (
            [
                i
                for i, d in enumerate(v.shape)
                if isinstance(d, torch.SymInt) and env.resolve_block_id(d) == axis
            ]
            if isinstance(v, torch.Tensor)
            else []
        )

    def touches(v: object) -> bool:
        if isinstance(v, (tuple, list)):
            return any(touches(x) for x in v)
        return (
            bool(row_dims(v))
            or isinstance(v, torch.SymInt)
            and env.resolve_block_id(v) == axis
        )

    def first_tile(node: object) -> bool:
        return (
            isinstance(node, Node)
            and node.target is operator.eq
            and (
                any(
                    isinstance(arg, Node) and arg.target is tile_ops.tile_begin
                    for arg in node.args
                )
                and any(isinstance(arg, int) and arg == 0 for arg in node.args)
            )
        )

    scalar_publications: set[Node] = set()

    def first_tile_publication(begin: Node) -> bool:
        # One block still publishes a scalar exactly once. A row-shaped
        # capture, a live branch result or a nonempty else can expose how
        # many rows that block owns (or how many other blocks execute).
        # Heuristic registration imports this module before the public Config
        # class, which device_ir itself imports, is available.
        from ..device_ir import codegen_node_order
        from ..device_ir import control_flow_child_graph_ids

        for predicate in begin.users:
            if predicate.target is not operator.eq or predicate.args not in (
                (begin, 0),
                (0, begin),
            ):
                return False
            for branch in predicate.users:
                if (
                    branch.target is not _tracing_ops._if
                    or branch.args[0] is not predicate
                    or branch.users
                ):
                    return False
                children = control_flow_child_graph_ids(branch)
                if len(children) != 2 or any(
                    n.op == "call_function" for n in ir.graphs[children[1]].graph.nodes
                ):
                    return False
                scalar_publications.update(
                    n
                    for _, n in codegen_node_order(
                        ir.graphs,
                        children[:1],
                        repeat_conditions=False,
                        unreached=False,
                    )
                    if n.target is memory_ops.store
                )
                if any(
                    touches(v.meta.get("val"))
                    for _, n in codegen_node_order(
                        ir.graphs, children, repeat_conditions=False, unreached=False
                    )
                    for v in (n, *n.all_input_nodes)
                ):
                    return False
        return True

    tile_indices = _tile_memory_indices(ir.root_nodes, axis)

    def private_width(width: Node) -> bool:
        # A block-size symbol doubles as a tile coordinate in memory index
        # lists and as a dimension of fragment allocations. If it reaches
        # scalar arithmetic or a stored value, the physical width is visible.
        for user in width.users:
            if user.target in (tile_ops.tile_index, tile_ops.tile_begin):
                if user.args != (width,):
                    return False
            elif user.target is creation_ops.full:
                if (
                    not isinstance(user.args[0], (tuple, list))
                    or width not in user.args[0]
                    or width in user.args[1:]
                ):
                    return False
            elif user.target in (memory_ops.load, memory_ops.store):
                indices = user.args[1]
                if not isinstance(indices, (tuple, list)) or width not in indices:
                    return False
                # x[tile] and x[tile.block_size] are identical in the FX IR.
                # Only the former denotes a coordinate that may be retiled.
                location = user.meta.get("location")
                if not isinstance(location, SourceLocation):
                    return False
                proved = tile_indices.get((user.target, location, len(indices)), set())
                if any(
                    item is width and i not in proved for i, item in enumerate(indices)
                ) or width in (user.args[0], *user.args[2:]):
                    return False
            else:
                return False
        return True

    owned = {graph.graph_id for graph in region}
    if any(
        touches(node.meta.get("val"))
        for graph in ir.graphs
        if graph.graph_id not in owned
        for node in graph.graph.nodes
    ):
        return None
    width_symbol = info.symbol()
    for graph in region:
        for n in graph.graph.nodes:
            value = n.meta.get("val")
            if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                expr = value.node.expr
                if width_symbol in expr.free_symbols:
                    if expr != width_symbol or not private_width(n):
                        return None
                elif any(env.get_block_id(s) == axis for s in expr.free_symbols):
                    # A nested graph may have captured an offset or a derived
                    # predicate without a direct FX edge to its defining node.
                    if n.target is not tile_ops.tile_begin and not first_tile(n):
                        return None
            if n.op != "call_function":
                continue
            values = [x.meta.get("val") for x in (n, *n.all_input_nodes)]
            if not any(touches(v) for v in values):
                continue
            if any(any(i != 0 for i in row_dims(v)) for v in values):
                return None
            if n.target is tile_ops.tile_begin:
                if not first_tile_publication(n):
                    return None
                continue
            if first_tile(n) or n.target is _tracing_ops._if and first_tile(n.args[0]):
                continue
            if n.target is torch.ops.aten.sum.dim_IntList:
                # A horizontal reduction cannot consume another tile's rows.
                source = n.all_input_nodes[0].meta.get("val")
                if (
                    not isinstance(source, torch.Tensor)
                    or source.ndim != 2
                    or n.args[1] not in ((-1,), [-1], (1,), [1])
                    or row_dims(values[0]) != [0]
                ):
                    return None
                continue
            if (
                n.target
                in (
                    operator.getitem,
                    _tracing_ops._for_loop,
                    _tracing_ops._get_symnode,
                    _tracing_ops._mask_select,
                    _tracing_ops._mask_to,
                    _tracing_ops._new_var,
                    _tracing_ops._phi,
                    creation_ops.full,
                    memory_ops.load,
                    memory_ops.store,
                    tile_ops.tile_index,
                    torch.ops.aten.sym_size.int,
                    torch.ops.prims.convert_element_type.default,
                )
                or isinstance(n.target, torch._ops.OpOverload)
                and torch.Tag.pointwise in n.target.tags
            ):
                continue
            return None
    # A private stripe can change which CTA owns a pair of adjacent rows.
    # Prove that every row write indexes its own tile directly, and decline
    # multiple writes to the same or overlapping storage. The scalar-only
    # first-tile branch above defines the sole non-row publication exception.
    outputs: list[torch.Tensor] = []
    for graph in region:
        for store in graph.graph.nodes:
            if store.target is not memory_ops.store:
                continue
            base, indices = store.args[:2]
            if not isinstance(base, Node) or not isinstance(indices, (list, tuple)):
                return None
            tensor = base.meta.get("val")
            if not isinstance(tensor, torch.Tensor):
                return None
            if any(
                host_storages_may_alias(
                    env, tensor, output, allow_unbound=allow_unbound
                )
                for output in outputs
            ):
                return None
            outputs.append(tensor)
            if store in scalar_publications:
                continue
            location = store.meta.get("location")
            if not isinstance(location, SourceLocation) or not tile_indices.get(
                (memory_ops.store, location, len(indices))
            ):
                return None
            # Distinct logical row coordinates must also be physically
            # disjoint under the final guarded layout (expanded rows are not).
            strides = static_strides(env, tensor)
            sizes = [static_int(d) for d in tensor.shape]
            if strides is None or any(d is None for d in sizes):
                return None
            span = 1
            for stride, size in sorted(
                zip(strides, sizes, strict=True), key=operator.itemgetter(0)
            ):
                assert size is not None
                if size > 1:
                    if stride < span:
                        return None
                    span += (size - 1) * stride
    return extent


def stripe_tiles(spec: ConfigSpec, config: Mapping[str, object]) -> dict[int, int]:
    """Complete 3/4 stripes only; never create tails or redistribute a thread axis."""
    blocks = config.get("block_sizes")
    # Normalization drops all-AUTO threads and installs the pipeline default
    # after backend validation. Read the same values before and after it.
    threads = config.get("num_threads", [])
    pipeline = config.get("cross_loop_pipeline")
    if "cross_loop_pipeline" not in config and spec.cross_loop_pipeline is not None:
        pipeline = spec.cross_loop_pipeline.default()
    if (
        not uses_warp_carry(spec, config)
        or pipeline != "barrier"
        or config.get("pid_type") != "persistent_interleaved"
        or not isinstance(blocks, list)
        or not isinstance(threads, list)
    ):
        return {}
    physical: dict[int, int] = {}
    for axis, extent in spec.cute_fragment_compact_row_extents.items():
        item = spec.block_sizes.block_id_lookup(axis)
        tile = spec.block_sizes.config_get(blocks, axis)
        if (
            not isinstance(tile, int)
            or tile < 64
            or tile % 64
            or item.structural_pin is not None
            or item.bounded_by_block_id is not None
            # AUTO can map the physical row tile onto threads (e.g. 64
            # lanes become 48 after retile). Require a proven serial row.
            or spec.num_threads.config_get(threads, axis, 0) != 1
        ):
            continue
        smaller = 3 * (tile // 4)
        if smaller >= item.min_size and extent % tile == 0 and extent % smaller == 0:
            physical[axis] = smaller
    return physical


def generate_striped_rows(
    func: HostFunction, config: Config, emit_repro_caller: bool
) -> ast.Module:
    from ..generate_ast import _generate_ast

    env = CompileEnvironment.current()
    spec = env.config_spec
    stripes = stripe_tiles(spec, config)
    if not stripes:
        raise exc.InvalidConfig(
            "CuTe compact row stripes require a complete independent warp carry tile"
        )
    # Search may describe disjoint potential arguments. Emission owns the
    # actual storage/stride guards: never reuse enrollment as a layout proof.
    from .row_reduction_carry import phase_region

    ir = func.device_ir
    for root, axes in zip(ir.root_ids, ir.grid_block_ids, strict=True):
        for axis in set(axes) & stripes.keys():
            if (
                complete_row_extent(env, ir, root, phase_region(ir.graphs, root), axis)
                is None
            ):
                raise exc.InvalidConfig(
                    "compact row writes must have disjoint stripe ownership"
                )
    # The public search coordinates remain powers of two. The physical copy
    # goes through the usual grid and fragment emitters so offsets, masks,
    # allocations and the launches of all later phases agree on the stripe.
    blocks = list(config.block_sizes)
    for axis, tile in stripes.items():
        blocks[spec.block_sizes.block_id_to_index(axis)] = tile
    options = dict(config.config)
    options.pop(COMPACT_ROWS_KEY)
    options["block_sizes"] = blocks
    # Emit one physical pass. Let its regeneration signals reach the outer
    # retry loop, which normalizes the public config and reapplies the stripe.
    return _generate_ast(func, Config.from_dict(options), emit_repro_caller)
