"""Generic thread-local tcgen05 fragment epilogues.

The planner owns a pure, single-store slice of the live FX graph.  It admits
operators individually and interprets their logical coordinates; it never
recognizes an expression or workload pattern.  The execution envelope is
deliberately narrow: every accumulator value needed by an output must be in
the same thread-local T2R residency subtile as the other values for that
output.
"""

from __future__ import annotations

import ast
import dataclasses
from fractions import Fraction
import functools
import math
import operator
import types
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
from typing import cast

import sympy
import torch
from torch._inductor.virtualized import V
from torch.fx.node import Node
from torch.fx.node import map_arg

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import tile_index
from ...language import view_ops
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..indexing_strategy import exact_tile_block_ids
from .cute_fx_walk import build_inner_outputs_index_from_graphs
from .cute_fx_walk import reach_matmul_anchors
from .cute_fx_walk import walk_carrier_to_tcgen05_matmul
from .cute_reshape import _get_tile_shape

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ...runtime.config import Config
    from ..device_ir import GraphInfo
    from ..inductor_lowering import CodegenState

    _IndexEvaluator = Callable[[dict[str, int]], int]


_T = TypeVar("_T")

_TCGEN05_FRAGMENT_SOURCE_BM = 128
_TCGEN05_FRAGMENT_SOURCE_BNS = (64, 128)
_TCGEN05_FRAGMENT_DTYPES = (torch.float16, torch.bfloat16)


class _UnsupportedFragment(Exception):
    pass


def _tcgen05_fragment_dtype_supported(dtype: torch.dtype) -> bool:
    return dtype in _TCGEN05_FRAGMENT_DTYPES


def _tcgen05_fragment_source_layout_supported(
    *, bm: int, bn: int, input_dtype: torch.dtype
) -> bool:
    """Whether the source tile has an ownership layout the oracle can query."""
    return (
        bm == _TCGEN05_FRAGMENT_SOURCE_BM
        and bn in _TCGEN05_FRAGMENT_SOURCE_BNS
        and _tcgen05_fragment_dtype_supported(input_dtype)
    )


@dataclasses.dataclass(frozen=True)
class _FragmentSlot:
    thread: int
    subtile: int
    register: int


@dataclasses.dataclass(frozen=True)
class _FragmentOwnership:
    source_shape: tuple[int, int]
    destination_shape: tuple[int, int]
    source_slots: tuple[_FragmentSlot, ...]
    destination_slots: tuple[_FragmentSlot, ...]
    source_subtile_count: int
    destination_subtile_count: int
    source_register_count: int
    destination_register_count: int
    thread_count: int


@dataclasses.dataclass(frozen=True)
class _BoundaryRegisterMap:
    boundary: Node
    index: _Index
    registers: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _SourceSubtileProgram:
    source_subtile: int
    destination_registers: tuple[int, ...]
    boundary_registers: tuple[_BoundaryRegisterMap, ...]


@dataclasses.dataclass(frozen=True)
class _DestinationSubtileProgram:
    groups: tuple[_SourceSubtileProgram, ...]


@dataclasses.dataclass(frozen=True)
class Tcgen05FragmentEpiloguePlan:
    anchor: Node
    store_node: Node
    value_node: Node
    boundary_nodes: frozenset[Node]
    owned_nodes: frozenset[Node]
    source_shape: tuple[int, ...]
    destination_shape: tuple[int, ...]
    store_tile_sizes: tuple[int | torch.SymInt, ...]
    programs: tuple[_DestinationSubtileProgram, ...]
    source_register_count: int
    destination_register_count: int

    @property
    def changes_shape(self) -> bool:
        return self.source_shape != self.destination_shape

    @property
    def streaming_program(self) -> _SourceSubtileProgram | None:
        """One source fragment produces one equal-sized destination fragment."""
        if self.changes_shape or not self.programs:
            return None
        first = self.programs[0]
        if len(first.groups) != 1:
            return None
        program = first.groups[0]
        if program.source_subtile != 0 or program.destination_registers != tuple(
            range(self.destination_register_count)
        ):
            return None
        signature = (
            program.destination_registers,
            tuple(
                (mapping.boundary, mapping.index, mapping.registers)
                for mapping in program.boundary_registers
            ),
        )
        for destination_subtile, current in enumerate(self.programs):
            if len(current.groups) != 1:
                return None
            group = current.groups[0]
            if group.source_subtile != destination_subtile:
                return None
            current_signature = (
                group.destination_registers,
                tuple(
                    (mapping.boundary, mapping.index, mapping.registers)
                    for mapping in group.boundary_registers
                ),
            )
            if current_signature != signature:
                return None
        return program


@functools.lru_cache(maxsize=16)
def _query_tcgen05_fragment_ownership(
    *,
    bm: int,
    bn: int,
    bk: int,
    destination_bm: int,
    destination_bn: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> _FragmentOwnership:
    """Query the exact CuTe T2R layout without compiling or launching a kernel."""
    if (
        not _tcgen05_fragment_source_layout_supported(
            bm=bm, bn=bn, input_dtype=input_dtype
        )
        or destination_bm != bm
        or destination_bn <= 0
        or destination_bn % 32
        or destination_bn > bn
        or not _tcgen05_fragment_dtype_supported(output_dtype)
    ):
        raise _UnsupportedFragment

    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.nvgpu import tcgen05
    from cutlass.utils import blackwell_helpers
    from cutlass.utils.gemm import sm100
    from cutlass.utils.layout import LayoutEnum

    from ._mlir_compat import ir

    input_type = cutlass.BFloat16 if input_dtype is torch.bfloat16 else cutlass.Float16
    output_type = (
        cutlass.BFloat16 if output_dtype is torch.bfloat16 else cutlass.Float16
    )

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            tiled_mma = blackwell_helpers.make_trivial_tiled_mma(
                input_type,
                input_type,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.MN,
                cutlass.Float32,
                tcgen05.CtaGroup.ONE,
                (bm, bn),
                tcgen05.OperandSource.SMEM,
            )
            accumulator = tiled_mma.make_fragment_C(
                cute.append(tiled_mma.partition_shape_C((bm, bn)), 1)
            )
            tmem_ptr = cute.make_ptr(
                cutlass.Float32,
                0,
                cute.AddressSpace.tmem,
                assumed_align=16,
            )
            tacc = sm100.transform_partitioned_tensor_layout(
                cute.make_tensor(tmem_ptr, accumulator.layout)
            )
            epilogue_tile = blackwell_helpers.compute_epilogue_tile_shape(
                (bm, bn),
                False,
                LayoutEnum.ROW_MAJOR,
                output_type,
                layout_c=LayoutEnum.ROW_MAJOR,
                elem_ty_c=output_type,
            )
            descriptor = types.SimpleNamespace(
                cta_tile_shape_mnk=(bm, bn, bk),
                c_layout=LayoutEnum.ROW_MAJOR,
                c_dtype=output_type,
                acc_dtype=cutlass.Float32,
            )
            copy_atom = blackwell_helpers.get_tmem_load_op(
                descriptor.cta_tile_shape_mnk,
                descriptor.c_layout,
                descriptor.c_dtype,
                descriptor.acc_dtype,
                epilogue_tile,
                False,
            )
            tacc_epilogue = cute.flat_divide(tacc, epilogue_tile)
            tiled_copy = tcgen05.make_tmem_copy(
                copy_atom, tacc_epilogue[(None, None, 0, 0, 0)]
            )
            thread_count = int(
                cute.size(tiled_copy.layout_dst_tv_tiled.shape, mode=[0])
            )

            def query(
                shape: tuple[int, int],
            ) -> tuple[tuple[_FragmentSlot, ...], int, int]:
                coordinates = cute.make_identity_tensor(shape)
                divided = cute.flat_divide(coordinates, epilogue_tile)
                fragments = []
                for thread in range(thread_count):
                    partitioned = tiled_copy.get_slice(thread).partition_D(divided)
                    grouped = cast(
                        "Any",
                        cute.group_modes(partitioned, 3, cute.rank(partitioned)),
                    )
                    fragments.append(
                        tuple(
                            grouped[(None, None, None, subtile)]
                            for subtile in range(
                                int(cute.size(grouped.shape, mode=[3]))
                            )
                        )
                    )
                if not fragments or not fragments[0]:
                    raise _UnsupportedFragment
                representative = fragments[0][0]
                register_count = int(cute.size(representative))
                representative_layout = str(representative.layout)
                representative_base = tuple(map(int, representative.iterator))
                offsets = tuple(
                    tuple(
                        int(coord) - base
                        for coord, base in zip(
                            representative[register],
                            representative_base,
                            strict=True,
                        )
                    )
                    for register in range(register_count)
                )
                subtile_count = len(fragments[0])
                slots: list[_FragmentSlot | None] = [None] * math.prod(shape)
                for thread, thread_fragments in enumerate(fragments):
                    if len(thread_fragments) != subtile_count:
                        raise _UnsupportedFragment
                    for subtile, fragment in enumerate(thread_fragments):
                        if (
                            str(fragment.layout) != representative_layout
                            or int(cute.size(fragment)) != register_count
                        ):
                            raise _UnsupportedFragment
                        base = tuple(map(int, fragment.iterator))
                        for register, offset in enumerate(offsets):
                            coordinate = tuple(
                                component + delta
                                for component, delta in zip(base, offset, strict=True)
                            )
                            if (
                                len(coordinate) != 2
                                or not 0 <= coordinate[0] < shape[0]
                                or not 0 <= coordinate[1] < shape[1]
                            ):
                                raise _UnsupportedFragment
                            flat = coordinate[0] * shape[1] + coordinate[1]
                            if slots[flat] is not None:
                                raise _UnsupportedFragment
                            slots[flat] = _FragmentSlot(thread, subtile, register)
                if any(slot is None for slot in slots):
                    raise _UnsupportedFragment
                return (
                    tuple(cast("_FragmentSlot", slot) for slot in slots),
                    subtile_count,
                    register_count,
                )

            source_slots, source_subtiles, source_registers = query((bm, bn))
            if (destination_bm, destination_bn) == (bm, bn):
                destination_slots = source_slots
                destination_subtiles = source_subtiles
                destination_registers = source_registers
            else:
                (
                    destination_slots,
                    destination_subtiles,
                    destination_registers,
                ) = query((destination_bm, destination_bn))

    return _FragmentOwnership(
        (bm, bn),
        (destination_bm, destination_bn),
        source_slots,
        destination_slots,
        source_subtiles,
        destination_subtiles,
        source_registers,
        destination_registers,
        thread_count,
    )


@dataclasses.dataclass(frozen=True)
class _Region:
    store: Node
    value: Node
    owned: frozenset[Node]
    boundaries: frozenset[Node]


_RESHAPE_TARGETS = {
    torch.ops.aten.reshape.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.view.default,
}
_PERMUTE_TARGETS = {
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.t.default,
}
_SHAPE_TARGETS = {
    *_RESHAPE_TARGETS,
    *_PERMUTE_TARGETS,
    torch.ops.aten.expand.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.dim,
    view_ops.subscript,
    view_ops.split,
    view_ops.join,
}


def _pointwise_inputs(node: Node) -> tuple[Node, ...] | None:
    from ..inductor_lowering import PointwiseLowering

    lowering = node.meta.get("lowering")
    if not isinstance(lowering, PointwiseLowering):
        return None
    inputs: list[Node] = []

    def visit(value: Node) -> Node:
        if isinstance(value.meta.get("val"), torch.Tensor):
            inputs.append(value)
        return value

    map_arg((node.args, {**node.kwargs, "_extra_deps": None}), visit)
    if not inputs or len(inputs) != len(lowering.input_names):
        return None
    return tuple(inputs)


def _is_split_getitem(node: Node) -> bool:
    base = node.args[0] if node.args else None
    index = node.args[1] if len(node.args) > 1 else None
    return (
        node.op == "call_function"
        and node.target is operator.getitem
        and isinstance(base, Node)
        and base.target is view_ops.split
        and index in (0, 1)
    )


def _is_host_tensor(node: Node) -> bool:
    return node.op == "call_function" and node.target is _tracing_ops._host_tensor


def _shape_only_subscript(node: Node) -> bool:
    source = node.args[0] if node.args else None
    indices = node.args[1] if len(node.args) > 1 else None
    source_value = source.meta.get("val") if isinstance(source, Node) else None
    output_value = node.meta.get("val")
    return (
        isinstance(source_value, torch.Tensor)
        and isinstance(output_value, torch.Tensor)
        and isinstance(indices, (list, tuple))
        and all(index is None or index == slice(None) for index in indices)
        and sum(index == slice(None) for index in indices) == source_value.ndim
        and len(indices) == output_value.ndim
    )


def _tile_index_block_ids(node: Node) -> set[int]:
    """Return tile-index provenance, rejecting data-derived indices."""
    if node.target is tile_index:
        size_node = node.args[0] if node.args else None
        size = size_node.meta.get("val") if isinstance(size_node, Node) else size_node
        block_id = (
            CompileEnvironment.current().get_block_id(size)
            if isinstance(size, (int, torch.SymInt, sympy.Basic))
            else None
        )
        if block_id is not None:
            return {block_id}
    source = node.args[0] if node.args else None
    shape_target = node.target in _RESHAPE_TARGETS | _PERMUTE_TARGETS | {
        torch.ops.aten.expand.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.dim,
        view_ops.subscript,
    }
    if isinstance(source, Node) and (
        shape_target or (node.target is memory_ops.load and not _is_host_tensor(source))
    ):
        if node.target is not view_ops.subscript or _shape_only_subscript(node):
            return _tile_index_block_ids(source)
    if _is_split_getitem(node):
        split = node.args[0]
        assert isinstance(split, Node)
        split_source = split.args[0]
        if isinstance(split_source, Node):
            return _tile_index_block_ids(split_source)
    if (pointwise_inputs := _pointwise_inputs(node)) is not None:
        # Index arithmetic remains rooted exclusively in tile coordinates.
        # The committed renderer evaluates the owned arithmetic independently
        # for each logical lane; this step only rejects host/data provenance.
        return set().union(*map(_tile_index_block_ids, pointwise_inputs))
    raise _UnsupportedFragment


def _tile_index_roots(node: Node) -> set[Node]:
    """Return the tile-index terminals of a shape/address expression."""
    if node.target is tile_index:
        return {node}
    source = node.args[0] if node.args else None
    shape_target = node.target in _RESHAPE_TARGETS | _PERMUTE_TARGETS | {
        torch.ops.aten.expand.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.dim,
        view_ops.subscript,
    }
    if isinstance(source, Node) and (
        shape_target or (node.target is memory_ops.load and not _is_host_tensor(source))
    ):
        if node.target is not view_ops.subscript or _shape_only_subscript(node):
            return _tile_index_roots(source)
    if _is_split_getitem(node):
        split = node.args[0]
        assert isinstance(split, Node)
        split_source = split.args[0]
        if isinstance(split_source, Node):
            return _tile_index_roots(split_source)
    if (pointwise_inputs := _pointwise_inputs(node)) is not None:
        return set().union(*map(_tile_index_roots, pointwise_inputs))
    raise _UnsupportedFragment


def _tile_index_uses_arithmetic(node: Node) -> bool:
    """Whether tile-derived addressing needs an explicit bounds predicate."""
    if _pointwise_inputs(node) is not None:
        return True
    source = node.args[0] if node.args else None
    shape_target = node.target in _RESHAPE_TARGETS | _PERMUTE_TARGETS | {
        torch.ops.aten.expand.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.dim,
        view_ops.subscript,
    }
    if isinstance(source, Node) and (
        shape_target or (node.target is memory_ops.load and not _is_host_tensor(source))
    ):
        if node.target is not view_ops.subscript or _shape_only_subscript(node):
            return _tile_index_uses_arithmetic(source)
    if _is_split_getitem(node):
        split = node.args[0]
        assert isinstance(split, Node)
        split_source = split.args[0]
        return isinstance(split_source, Node) and _tile_index_uses_arithmetic(
            split_source
        )
    return False


def _fragment_store_block_ids(subscripts: Sequence[object]) -> tuple[int, ...] | None:
    """Recover tile provenance through shape and integer-index arithmetic."""
    env = CompileEnvironment.current()
    result: list[int] = []
    try:
        for subscript in subscripts:
            if isinstance(subscript, Node) and isinstance(
                subscript.meta.get("val"), torch.Tensor
            ):
                block_ids = _tile_index_block_ids(subscript)
                if len(block_ids) != 1:
                    return None
                (block_id,) = block_ids
            elif isinstance(subscript, Node) and isinstance(
                subscript.meta.get("val"), torch.SymInt
            ):
                block_id = env.get_block_id(subscript.meta["val"])
                if block_id is None:
                    return None
            elif isinstance(subscript, torch.SymInt):
                block_id = env.get_block_id(subscript)
                if block_id is None:
                    return None
            else:
                return None
            result.append(block_id)
    except _UnsupportedFragment:
        return None
    return tuple(result)


def _fragment_store_tile_sizes(
    subscripts: Sequence[object],
) -> tuple[int | torch.SymInt, ...]:
    result: list[int | torch.SymInt] = []
    for subscript in subscripts:
        if not isinstance(subscript, Node) or not isinstance(
            subscript.meta.get("val"), torch.Tensor
        ):
            raise _UnsupportedFragment
        roots = _tile_index_roots(subscript)
        if len(roots) != 1:
            raise _UnsupportedFragment
        (root,) = roots
        size_node = root.args[0] if root.args else None
        size = size_node.meta.get("val") if isinstance(size_node, Node) else size_node
        if not isinstance(size, (int, torch.SymInt)):
            raise _UnsupportedFragment
        result.append(size)
    return tuple(result)


def _extent_matches_block(extent: int | torch.SymInt, block_id: int) -> bool:
    env = CompileEnvironment.current()
    size = env.block_sizes[env.canonical_block_id(block_id)].size
    return isinstance(size, (int, torch.SymInt)) and env.known_equal(extent, size)


def _validate_host_load(node: Node, output_node: Node) -> None:
    source = node.args[0] if node.args else None
    indices = node.args[1] if len(node.args) > 1 else None
    source_value = source.meta.get("val") if isinstance(source, Node) else None
    output_value = node.meta.get("val")
    if not (
        isinstance(source, Node)
        and _is_host_tensor(source)
        and source is not output_node
        and isinstance(source_value, torch.Tensor)
        and isinstance(output_value, torch.Tensor)
        and isinstance(indices, (list, tuple))
        and len(indices) == source_value.ndim
        and (len(node.args) <= 2 or node.args[2] is None)
        and source_value.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        raise _UnsupportedFragment
    tensor_indices = [
        index
        for index in indices
        if isinstance(index, Node) and isinstance(index.meta.get("val"), torch.Tensor)
    ]
    env = CompileEnvironment.current()
    if tensor_indices and len(tensor_indices) != len(indices):
        raise _UnsupportedFragment
    used: set[int] = set()
    output_dim = 0
    for tensor_dim, index in enumerate(indices):
        block_ids: set[int] = set()
        if isinstance(index, Node) and isinstance(index.meta.get("val"), torch.Tensor):
            block_ids = _tile_index_block_ids(index)
        elif isinstance(index, Node) and isinstance(
            index.meta.get("val"), torch.SymInt
        ):
            if (block_id := env.get_block_id(index.meta["val"])) is not None:
                block_ids = {block_id}
        elif index == slice(None) and output_dim < output_value.ndim:
            if (
                block_id := env.resolve_block_id(output_value.shape[output_dim])
            ) is not None:
                block_ids = {block_id}
        elif not (
            isinstance(index, int)
            and isinstance(source_value.shape[tensor_dim], int)
            and 0 <= index < source_value.shape[tensor_dim]
        ):
            raise _UnsupportedFragment
        if block_ids:
            if len(block_ids) != 1:
                raise _UnsupportedFragment
            (block_id,) = block_ids
            canonical = env.canonical_block_id(block_id)
            expected = (
                None
                if tensor_indices
                else env.resolve_block_id(output_value.shape[output_dim])
            )
            if (
                canonical in used
                or (
                    expected is not None
                    and canonical != env.canonical_block_id(expected)
                )
                or not _extent_matches_block(source_value.shape[tensor_dim], block_id)
            ):
                raise _UnsupportedFragment
            used.add(canonical)
            output_dim += 1
    if not tensor_indices and output_dim != output_value.ndim:
        raise _UnsupportedFragment


def _node_inputs(node: Node, output_node: Node) -> tuple[Node, ...] | None:
    if node.op != "call_function":
        return None
    if (pointwise := _pointwise_inputs(node)) is not None:
        return pointwise
    if node.target is tile_index:
        return ()
    if node.target is memory_ops.load:
        source = node.args[0] if node.args else None
        if not isinstance(source, Node) or (
            len(node.args) > 2 and node.args[2] is not None
        ):
            return None
        if not _is_host_tensor(source):
            return (source,) if _shape_only_subscript(node) else None
        _validate_host_load(node, output_node)
        indices = node.args[1] if len(node.args) > 1 else ()
        if not isinstance(indices, (list, tuple)):
            raise _UnsupportedFragment
        return tuple(
            index
            for index in indices
            if isinstance(index, Node)
            and isinstance(index.meta.get("val"), torch.Tensor)
        )
    if node.target is view_ops.subscript and not _shape_only_subscript(node):
        return None
    if node.target is view_ops.join:
        return tuple(arg for arg in node.args[:2] if isinstance(arg, Node))
    if node.target not in _SHAPE_TARGETS and not _is_split_getitem(node):
        return None
    source = node.args[0] if node.args else None
    return (source,) if isinstance(source, Node) else ()


def _extract_region(
    graphs: Sequence[GraphInfo],
    anchor: Node,
    expected_output_block_ids: tuple[int, ...],
) -> _Region:
    inner_outputs = build_inner_outputs_index_from_graphs(graphs)
    reachable = [
        (store, value)
        for graph_info in graphs
        for store in graph_info.graph.nodes
        if store.target is memory_ops.store
        and len(store.args) > 2
        and isinstance((value := store.args[2]), Node)
        if anchor
        in reach_matmul_anchors(
            value,
            target_fx_nodes={anchor},
            inner_outputs_by_graph_id=inner_outputs,
        )
    ]
    if len(reachable) != 1:
        raise _UnsupportedFragment
    store, value = reachable[0]
    output_node = store.args[0] if store.args else None
    output = output_node.meta.get("val") if isinstance(output_node, Node) else None
    subscripts = store.args[1] if len(store.args) > 1 else None
    mask = store.args[3] if len(store.args) > 3 else None
    if (
        not isinstance(output_node, Node)
        or not isinstance(output, torch.Tensor)
        or output.ndim != 3
        or output.dtype not in (torch.float16, torch.bfloat16)
        or not isinstance(subscripts, (list, tuple))
        or (
            exact_tile_block_ids(CompileEnvironment.current(), subscripts)
            or _fragment_store_block_ids(subscripts)
        )
        != expected_output_block_ids
        or mask is not None
    ):
        raise _UnsupportedFragment

    owned: set[Node] = set()
    boundaries: set[Node] = set()
    visiting: set[Node] = set()

    def visit(node: Node) -> None:
        if node in owned or node in boundaries:
            return
        if walk_carrier_to_tcgen05_matmul(node, {anchor}, inner_outputs) is anchor:
            boundaries.add(node)
            return
        if node.graph is not store.graph or node in visiting:
            raise _UnsupportedFragment(
                f"unsupported region node {node.name}: {node.target}"
            )
        dependencies = _node_inputs(node, output_node)
        if dependencies is None:
            raise _UnsupportedFragment(
                f"unsupported region node {node.name}: {node.target}"
            )
        visiting.add(node)
        for dependency in dependencies:
            visit(dependency)
        visiting.remove(node)
        owned.add(node)

    visit(value)
    for subscript in subscripts:
        if isinstance(subscript, Node) and isinstance(
            subscript.meta.get("val"), torch.Tensor
        ):
            visit(subscript)
    if not boundaries:
        raise _UnsupportedFragment
    allowed_users = {*owned, store}
    if any(
        user not in allowed_users
        for node in (*owned, *boundaries)
        for user in node.users
    ):
        raise _UnsupportedFragment

    body_graph_ids = [
        graph_info.graph_id for graph_info in graphs if graph_info.graph is anchor.graph
    ]
    if len(body_graph_ids) != 1:
        raise _UnsupportedFragment
    (body_graph_id,) = body_graph_ids
    loop_nodes = [
        node
        for node in store.graph.nodes
        if node.op == "call_function"
        and _tracing_ops.is_for_loop_target(node.target)
        and node.args
        and node.args[0] == body_graph_id
    ]
    if len(loop_nodes) != 1:
        raise _UnsupportedFragment
    (loop_node,) = loop_nodes
    order = {node: index for index, node in enumerate(store.graph.nodes)}
    if not all(order[loop_node] < order[node] < order[store] for node in owned):
        raise _UnsupportedFragment
    for node in store.graph.nodes:
        if not (order[loop_node] < order[node] < order[store]) or node in owned:
            continue
        is_loop_result = (
            node.target is operator.getitem and node.args and node.args[0] is loop_node
        )
        is_shape_scalar = isinstance(node.meta.get("val"), (int, torch.SymInt))
        if (
            node.op == "call_function"
            and node not in boundaries
            and not is_loop_result
            and not is_shape_scalar
            and not _is_host_tensor(node)
        ):
            raise _UnsupportedFragment(f"intervening node {node.name}: {node.target}")

    return _Region(store, value, frozenset(owned), frozenset(boundaries))


def analyze_tcgen05_fragment_epilogue_candidate(
    graphs: Sequence[GraphInfo],
    anchor: Node,
    *,
    expected_output_block_ids: tuple[int, ...],
) -> bool:
    """Side-effect-free preflight sharing the fragment planner's extraction."""
    try:
        _extract_region(graphs, anchor, expected_output_block_ids)
    except _UnsupportedFragment:
        return False
    return True


def _interpret_index(
    value: sympy.Expr,
    variables: Mapping[str, _T],
    *,
    constant: Callable[[int], _T],
    add: Callable[[tuple[_T, ...]], _T],
    multiply: Callable[[tuple[_T, ...]], _T],
    floor_divide: Callable[[_T, _T], _T],
    modulo: Callable[[_T, _T], _T],
) -> _T:
    """Interpret the complete logical-index grammar into one target domain."""

    def visit(current: sympy.Expr) -> _T:
        if isinstance(current, sympy.Integer):
            return constant(int(current))
        if isinstance(current, sympy.Symbol):
            return variables[str(current)]
        if current.is_Add:
            return add(tuple(visit(cast("sympy.Expr", arg)) for arg in current.args))
        if current.is_Mul:
            return multiply(
                tuple(visit(cast("sympy.Expr", arg)) for arg in current.args)
            )
        if current.func is sympy.floor:
            numerator, denominator = cast(
                "sympy.Expr", current.args[0]
            ).as_numer_denom()
            return floor_divide(visit(numerator), visit(denominator))
        if current.func is sympy.Mod:
            return modulo(
                visit(cast("sympy.Expr", current.args[0])),
                visit(cast("sympy.Expr", current.args[1])),
            )
        raise _UnsupportedFragment(f"unsupported logical index {current}")

    return visit(value)


@dataclasses.dataclass(frozen=True)
class _Index:
    semantic: sympy.Expr
    bounds: tuple[tuple[sympy.Symbol, int, int], ...] = ()

    @staticmethod
    def constant(value: int) -> _Index:
        return _Index(sympy.Integer(value))

    @staticmethod
    def variable(name: str, upper: int) -> _Index:
        symbol = sympy.Symbol(name, integer=True, nonnegative=True)
        return _Index(symbol, ((symbol, 0, upper),))

    def compile(self) -> _IndexEvaluator:
        """Compile this index once for repeated exhaustive evaluation."""

        def variable(name: str) -> _IndexEvaluator:
            return operator.itemgetter(name)

        def constant(value: int) -> _IndexEvaluator:
            return lambda _values: value

        def add(parts: tuple[_IndexEvaluator, ...]) -> _IndexEvaluator:
            return lambda values: sum(part(values) for part in parts)

        def multiply(parts: tuple[_IndexEvaluator, ...]) -> _IndexEvaluator:
            return lambda values: math.prod(part(values) for part in parts)

        def floor_divide(
            numerator: _IndexEvaluator, denominator: _IndexEvaluator
        ) -> _IndexEvaluator:
            return lambda values: numerator(values) // denominator(values)

        def modulo(
            numerator: _IndexEvaluator, denominator: _IndexEvaluator
        ) -> _IndexEvaluator:
            return lambda values: numerator(values) % denominator(values)

        variables = {
            str(symbol): variable(str(symbol)) for symbol, _lower, _upper in self.bounds
        }
        return _interpret_index(
            self.semantic,
            variables,
            constant=constant,
            add=add,
            multiply=multiply,
            floor_divide=floor_divide,
            modulo=modulo,
        )

    def render(self, variables: dict[str, str]) -> str:
        return _interpret_index(
            self.semantic,
            variables,
            constant=lambda value: f"cutlass.Int32({value})",
            add=lambda parts: "(" + " + ".join(parts) + ")",
            multiply=lambda parts: "(" + " * ".join(parts) + ")",
            floor_divide=lambda numerator, denominator: (
                f"(({numerator}) // ({denominator}))"
            ),
            modulo=lambda numerator, denominator: f"(({numerator}) % ({denominator}))",
        )


def _merge_bounds(*indices: _Index) -> tuple[tuple[sympy.Symbol, int, int], ...]:
    merged: dict[sympy.Symbol, tuple[int, int]] = {}
    for index in indices:
        for symbol, lower, upper in index.bounds:
            if merged.setdefault(symbol, (lower, upper)) != (lower, upper):
                raise _UnsupportedFragment
    return tuple((symbol, *merged[symbol]) for symbol in sorted(merged, key=str))


def _semantic_interval(
    value: sympy.Expr, bounds: dict[sympy.Symbol, tuple[int, int]]
) -> tuple[Fraction, Fraction] | None:
    if value.is_Rational:
        rational = Fraction(str(value))
        return rational, rational
    if isinstance(value, sympy.Symbol):
        interval = bounds.get(value)
        if interval is None:
            return None
        return Fraction(interval[0]), Fraction(interval[1])
    if value.is_Add:
        parts = [
            _semantic_interval(cast("sympy.Expr", arg), bounds) for arg in value.args
        ]
        if any(part is None for part in parts):
            return None
        lower = Fraction(0)
        upper = Fraction(0)
        for part in parts:
            if part is not None:
                lower += part[0]
                upper += part[1]
        return lower, upper
    if value.is_Mul:
        result = (Fraction(1), Fraction(1))
        for arg in value.args:
            part = _semantic_interval(cast("sympy.Expr", arg), bounds)
            if part is None:
                return None
            products = tuple(left * right for left in result for right in part)
            result = min(products), max(products)
        return result
    if value.func is sympy.floor:
        part = _semantic_interval(cast("sympy.Expr", value.args[0]), bounds)
        if part is None:
            return None
        return Fraction(math.floor(part[0])), Fraction(math.floor(part[1]))
    if value.func is sympy.Mod:
        part = _semantic_interval(cast("sympy.Expr", value.args[0]), bounds)
        modulus = value.args[1]
        if not isinstance(modulus, sympy.Integer):
            return None
        modulus_value = int(modulus)
        if part is not None and 0 <= part[0] <= part[1] < modulus_value:
            return part
        if modulus_value > 0:
            return Fraction(0), Fraction(modulus_value - 1)
    return None


@functools.lru_cache(maxsize=512)
def _simplify_semantic(
    value: sympy.Expr, bounds_tuple: tuple[tuple[sympy.Symbol, int, int], ...]
) -> sympy.Expr:
    bounds = {symbol: (lower, upper) for symbol, lower, upper in bounds_tuple}

    def visit(current: sympy.Expr) -> sympy.Expr:
        if current.args:
            current = cast(
                "sympy.Expr",
                sympy.simplify(
                    current.func(
                        *(visit(cast("sympy.Expr", arg)) for arg in current.args)
                    )
                ),
            )
        if current.func is sympy.floor:
            interval = _semantic_interval(cast("sympy.Expr", current.args[0]), bounds)
            if interval is not None and math.floor(interval[0]) == math.floor(
                interval[1]
            ):
                return sympy.Integer(math.floor(interval[0]))
        if current.func is sympy.Mod:
            interval = _semantic_interval(cast("sympy.Expr", current.args[0]), bounds)
            modulus = current.args[1]
            if (
                interval is not None
                and isinstance(modulus, sympy.Integer)
                and 0 <= interval[0] <= interval[1] < int(modulus)
            ):
                return cast("sympy.Expr", current.args[0])
        return current

    previous = value
    for _ in range(4):
        current = visit(previous)
        if current == previous:
            return current
        previous = current
    return previous


def _binary(op: str, left: _Index | int, right: _Index | int) -> _Index:
    left = left if isinstance(left, _Index) else _Index.constant(left)
    right = right if isinstance(right, _Index) else _Index.constant(right)
    bounds = _merge_bounds(left, right)
    if op == "add":
        semantic = sympy.Add(left.semantic, right.semantic)
    elif op == "mul":
        semantic = sympy.Mul(left.semantic, right.semantic)
    elif op == "floordiv":
        quotient = sympy.Mul(left.semantic, sympy.Pow(right.semantic, -1))
        semantic = cast("sympy.Expr", sympy.floor(quotient))
    elif op == "mod":
        semantic = sympy.Mod(left.semantic, right.semantic)
    else:
        raise AssertionError(op)
    return _Index(_simplify_semantic(semantic, bounds), bounds)


def _add(left: _Index | int, right: _Index | int) -> _Index:
    return _binary("add", left, right)


def _mul(left: _Index | int, right: _Index | int) -> _Index:
    return _binary("mul", left, right)


def _floordiv(left: _Index | int, right: _Index | int) -> _Index:
    return _binary("floordiv", left, right)


def _mod(left: _Index | int, right: _Index | int) -> _Index:
    return _binary("mod", left, right)


def _constant_index(index: _Index) -> int | None:
    return int(index.semantic) if isinstance(index.semantic, sympy.Integer) else None


def _coords_from_flat(flat: _Index, shape: Sequence[int]) -> list[_Index]:
    coords: list[_Index] = []
    stride = 1
    for extent in reversed(shape):
        coords.append(_mod(_floordiv(flat, stride), extent))
        stride *= extent
    return list(reversed(coords))


def _flat_from_coords(coords: Sequence[_Index], shape: Sequence[int]) -> _Index:
    result = _Index.constant(0)
    for coord, extent in zip(coords, shape, strict=True):
        result = _add(_mul(result, extent), coord)
    return result


def _broadcast_flat(
    flat: _Index, output_shape: Sequence[int], source_shape: Sequence[int]
) -> _Index:
    if len(source_shape) > len(output_shape):
        raise _UnsupportedFragment
    output_coords = _coords_from_flat(flat, output_shape)
    offset = len(output_shape) - len(source_shape)
    source_coords: list[_Index] = []
    for dim, source_extent in enumerate(source_shape):
        output_extent = output_shape[offset + dim]
        if source_extent == 1:
            source_coords.append(_Index.constant(0))
        elif source_extent == output_extent:
            source_coords.append(output_coords[offset + dim])
        else:
            raise _UnsupportedFragment
    return _flat_from_coords(source_coords, source_shape)


def _shape(node: Node, config: Config) -> list[int]:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        raise _UnsupportedFragment
    return _get_tile_shape(value, CompileEnvironment.current(), config)


def _permutation(node: Node, ndim: int) -> list[int]:
    if node.target is torch.ops.aten.permute.default:
        dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dims")
        if not isinstance(dims, (list, tuple)):
            raise _UnsupportedFragment
        result: list[int] = []
        for dim in dims:
            if not isinstance(dim, int):
                raise _UnsupportedFragment
            result.append(dim % ndim)
        return result
    if node.target is torch.ops.aten.transpose.int:
        dim0 = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim0")
        dim1 = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim1")
        if not isinstance(dim0, int) or not isinstance(dim1, int):
            raise _UnsupportedFragment
        result = list(range(ndim))
        result[dim0 % ndim], result[dim1 % ndim] = (
            result[dim1 % ndim],
            result[dim0 % ndim],
        )
        return result
    if node.target is torch.ops.aten.t.default and ndim == 2:
        return [1, 0]
    raise _UnsupportedFragment


def _resolve_shape(
    node: Node,
    flat: _Index,
    *,
    config: Config,
    evaluate: Callable[[Node, _Index, int | None], object],
    choose: Callable[[_Index, object, object], object],
    projection: int | None = None,
) -> object:
    if node.target in _RESHAPE_TARGETS:
        source = node.args[0]
        if not isinstance(source, Node):
            raise _UnsupportedFragment
        return evaluate(source, flat, None)
    if node.target in _PERMUTE_TARGETS:
        source = node.args[0]
        if not isinstance(source, Node):
            raise _UnsupportedFragment
        output_coords = _coords_from_flat(flat, _shape(node, config))
        perm = _permutation(node, len(output_coords))
        source_coords = [output_coords[perm.index(dim)] for dim in range(len(perm))]
        return evaluate(
            source, _flat_from_coords(source_coords, _shape(source, config)), None
        )
    if node.target is torch.ops.aten.expand.default:
        source = node.args[0]
        if not isinstance(source, Node):
            raise _UnsupportedFragment
        return evaluate(
            source,
            _broadcast_flat(flat, _shape(node, config), _shape(source, config)),
            None,
        )
    if node.target is torch.ops.aten.unsqueeze.default:
        source = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        if not isinstance(source, Node) or not isinstance(dim, int):
            raise _UnsupportedFragment
        coords = _coords_from_flat(flat, _shape(node, config))
        dim %= len(coords)
        return evaluate(
            source,
            _flat_from_coords(coords[:dim] + coords[dim + 1 :], _shape(source, config)),
            None,
        )
    if node.target is torch.ops.aten.squeeze.dim:
        source = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        if not isinstance(source, Node) or not isinstance(dim, int):
            raise _UnsupportedFragment
        source_shape = _shape(source, config)
        dim %= len(source_shape)
        coords = _coords_from_flat(flat, _shape(node, config))
        if source_shape[dim] == 1:
            coords.insert(dim, _Index.constant(0))
        return evaluate(source, _flat_from_coords(coords, source_shape), None)
    source_arg = node.args[0] if node.args else None
    if node.target is view_ops.subscript or (
        node.target is memory_ops.load
        and isinstance(source_arg, Node)
        and not _is_host_tensor(source_arg)
    ):
        source = source_arg
        indices = node.args[1] if len(node.args) > 1 else None
        if not isinstance(source, Node) or not isinstance(indices, (list, tuple)):
            raise _UnsupportedFragment
        output_coords = iter(_coords_from_flat(flat, _shape(node, config)))
        source_coords: list[_Index] = []
        for index in indices:
            coord = next(output_coords)
            if index is None:
                continue
            if index != slice(None):
                raise _UnsupportedFragment
            source_coords.append(coord)
        return evaluate(
            source, _flat_from_coords(source_coords, _shape(source, config)), None
        )
    if _is_split_getitem(node):
        split = node.args[0]
        projection_arg = node.args[1]
        assert isinstance(split, Node) and isinstance(projection_arg, int)
        source = split.args[0]
        if not isinstance(source, Node):
            raise _UnsupportedFragment
        return evaluate(source, _add(_mul(flat, 2), projection_arg), None)
    if node.target is view_ops.split:
        if projection not in (0, 1):
            raise _UnsupportedFragment
        source = node.args[0]
        if not isinstance(source, Node):
            raise _UnsupportedFragment
        return evaluate(source, _add(_mul(flat, 2), projection), None)
    if node.target is view_ops.join:
        left = node.args[0] if node.args else None
        right = node.args[1] if len(node.args) > 1 else None
        if not isinstance(left, Node) or not isinstance(right, Node):
            raise _UnsupportedFragment
        coords = _coords_from_flat(flat, _shape(node, config))
        selector = coords[-1]
        source_flat = _flat_from_coords(coords[:-1], _shape(left, config))
        return choose(
            selector,
            evaluate(left, source_flat, None),
            evaluate(right, source_flat, None),
        )
    return evaluate(node, flat, projection)


def _logical_index_value(
    node: Node,
    flat: _Index,
    *,
    config: Config,
    projection: int | None = None,
    tile_origins: Mapping[int, _Index] | None = None,
    memo: dict[tuple[Node, _Index, int | None], _Index] | None = None,
) -> _Index:
    """Interpret a tile-derived integer tensor at one logical coordinate."""
    if memo is None:
        memo = {}
    key = (node, flat, projection)
    if key in memo:
        return memo[key]

    def evaluate(
        current: Node, current_flat: _Index, current_projection: int | None
    ) -> object:
        if (
            current is not node
            or current_flat != flat
            or current_projection != projection
        ):
            return _logical_index_value(
                current,
                current_flat,
                config=config,
                projection=current_projection,
                tile_origins=tile_origins,
                memo=memo,
            )
        if current.target is tile_index:
            shape = _shape(current, config)
            if len(shape) != 1:
                raise _UnsupportedFragment
            coordinate = _coords_from_flat(current_flat, shape)[0]
            if tile_origins is None:
                return coordinate
            size_node = current.args[0] if current.args else None
            size = (
                size_node.meta.get("val") if isinstance(size_node, Node) else size_node
            )
            if not isinstance(size, (int, torch.SymInt, sympy.Basic)):
                raise _UnsupportedFragment
            block_id = CompileEnvironment.current().get_block_id(size)
            if block_id is None:
                raise _UnsupportedFragment
            origin = tile_origins.get(
                CompileEnvironment.current().canonical_block_id(block_id)
            )
            if origin is None:
                raise _UnsupportedFragment
            return _add(origin, coordinate)

        inputs = _pointwise_inputs(current)
        if inputs is None:
            raise _UnsupportedFragment
        output_shape = _shape(current, config)

        def argument(value: object) -> _Index:
            if isinstance(value, Node) and isinstance(
                value.meta.get("val"), torch.Tensor
            ):
                return _logical_index_value(
                    value,
                    _broadcast_flat(current_flat, output_shape, _shape(value, config)),
                    config=config,
                    tile_origins=tile_origins,
                    memo=memo,
                )
            if isinstance(value, (int, bool)):
                return _Index.constant(int(value))
            raise _UnsupportedFragment

        target = current.target
        args = current.args
        if target in {
            operator.add,
            torch.ops.aten.add.Scalar,
            torch.ops.aten.add.Tensor,
        }:
            if current.kwargs.get("alpha", 1) != 1 or len(args) < 2:
                raise _UnsupportedFragment
            return _add(argument(args[0]), argument(args[1]))
        if target in {
            operator.sub,
            torch.ops.aten.sub.Scalar,
            torch.ops.aten.sub.Tensor,
        }:
            if current.kwargs.get("alpha", 1) != 1 or len(args) < 2:
                raise _UnsupportedFragment
            return _add(argument(args[0]), _mul(argument(args[1]), -1))
        if target in {
            operator.mul,
            torch.ops.aten.mul.Scalar,
            torch.ops.aten.mul.Tensor,
        }:
            if len(args) < 2:
                raise _UnsupportedFragment
            return _mul(argument(args[0]), argument(args[1]))
        if target in {
            operator.floordiv,
            torch.ops.aten.floor_divide.default,
        }:
            if len(args) < 2:
                raise _UnsupportedFragment
            return _floordiv(argument(args[0]), argument(args[1]))
        if target is torch.ops.aten.div.Tensor_mode:
            rounding_mode = (
                args[2] if len(args) > 2 else current.kwargs.get("rounding_mode")
            )
            if len(args) < 2 or rounding_mode != "floor":
                raise _UnsupportedFragment
            return _floordiv(argument(args[0]), argument(args[1]))
        if target in {
            operator.mod,
            torch.ops.aten.remainder.Scalar,
            torch.ops.aten.remainder.Tensor,
        }:
            if len(args) < 2:
                raise _UnsupportedFragment
            return _mod(argument(args[0]), argument(args[1]))
        if target in {operator.neg, torch.ops.aten.neg.default}:
            if not args:
                raise _UnsupportedFragment
            return _mul(argument(args[0]), -1)
        raise _UnsupportedFragment

    def choose(selector: _Index, left: object, right: object) -> object:
        constant = _constant_index(selector)
        if (
            constant not in (0, 1)
            or not isinstance(left, _Index)
            or not isinstance(right, _Index)
        ):
            raise _UnsupportedFragment
        return left if constant == 0 else right

    result = _resolve_shape(
        node,
        flat,
        config=config,
        evaluate=evaluate,
        choose=choose,
        projection=projection,
    )
    if not isinstance(result, _Index):
        raise _UnsupportedFragment
    memo[key] = result
    return result


def _store_indices_are_tile_identity(
    region: _Region,
    source_shape: Sequence[int],
    output_shape: Sequence[int],
    *,
    source_global_shape: Sequence[int | torch.SymInt],
    expected_output_block_ids: Sequence[int],
    config: Config,
) -> bool:
    """Prove derived indices enumerate every translated destination tile."""
    subscripts = region.store.args[1] if len(region.store.args) > 1 else None
    if not isinstance(subscripts, (list, tuple)) or len(subscripts) != len(
        output_shape
    ):
        return False
    try:
        if (
            len(source_shape) != len(output_shape)
            or len(output_shape) != len(expected_output_block_ids)
            or len(source_global_shape) != len(output_shape)
        ):
            raise _UnsupportedFragment
        output_node = region.store.args[0] if region.store.args else None
        output_value = (
            output_node.meta.get("val") if isinstance(output_node, Node) else None
        )
        if not isinstance(output_value, torch.Tensor) or any(
            not isinstance(extent, int) for extent in output_value.shape
        ):
            raise _UnsupportedFragment
        tile_counts: list[int] = []
        for source_global, destination_global, source_extent, destination_extent in zip(
            source_global_shape,
            output_value.shape,
            source_shape,
            output_shape,
            strict=True,
        ):
            source_global_extent = CompileEnvironment.current().size_hint(source_global)
            if (
                source_extent % destination_extent
                or source_global_extent <= 0
                or source_global_extent % source_extent
                or destination_global <= 0
                or destination_global % destination_extent
                or source_global_extent * destination_extent
                != destination_global * source_extent
            ):
                raise _UnsupportedFragment
            tile_counts.append(source_global_extent // source_extent)
        if exact_tile_block_ids(CompileEnvironment.current(), subscripts) is not None:
            # Exact source tile indices need no expression enumeration, but a
            # compact output still needs a translated origin (for example
            # source N tile i -> destination N/2 tile i). Global tile coverage
            # above is mandatory for both paths.
            return tuple(source_shape) == tuple(output_shape)
        row = _Index.variable("row", output_shape[-2] - 1)
        column = _Index.variable("column", output_shape[-1] - 1)
        output_coords = [_Index.constant(0), row, column]
        output_flat = _flat_from_coords(output_coords, output_shape)
        evaluators: list[tuple[_IndexEvaluator, str, int]] = []
        for dim, (
            subscript,
            expected,
            source_extent,
            destination_extent,
            block_id,
            tile_count,
        ) in enumerate(
            zip(
                subscripts,
                output_coords,
                source_shape,
                output_shape,
                expected_output_block_ids,
                tile_counts,
                strict=True,
            )
        ):
            if not isinstance(subscript, Node) or not isinstance(
                subscript.meta.get("val"), torch.Tensor
            ):
                raise _UnsupportedFragment
            tile_name = f"tile_{dim}"
            tile = _Index.variable(tile_name, tile_count - 1)
            source_origin = _mul(tile, source_extent)
            value = _logical_index_value(
                subscript,
                _broadcast_flat(output_flat, output_shape, _shape(subscript, config)),
                config=config,
                tile_origins={
                    CompileEnvironment.current().canonical_block_id(
                        block_id
                    ): source_origin
                },
            )
            destination_origin = _mul(tile, destination_extent)
            difference = _add(
                value,
                _mul(_add(destination_origin, expected), -1),
            )
            if _constant_index(difference) == 0:
                continue
            # Keep the fallback exact and bounded. Valid affine/floor-divide
            # compact stores simplify above; an exotic expression which does
            # not simplify is accepted only after enumerating its complete
            # finite runtime domain, never from a few sampled tile origins.
            if tile_count * output_shape[-2] * output_shape[-1] > 1_000_000:
                raise _UnsupportedFragment
            evaluators.append((difference.compile(), tile_name, tile_count))
        variables = {"row": 0, "column": 0}
        for evaluator, tile_name, tile_count in evaluators:
            for tile in range(tile_count):
                variables[tile_name] = tile
                for m in range(output_shape[-2]):
                    variables["row"] = m
                    for n in range(output_shape[-1]):
                        variables["column"] = n
                        if evaluator(variables) != 0:
                            return False
    except (_UnsupportedFragment, IndexError, StopIteration, ValueError):
        return False
    return True


def _collect_demands(
    node: Node,
    flat: _Index,
    *,
    region: _Region,
    config: Config,
    projection: int | None = None,
    memo: dict[tuple[Node, _Index, int | None], frozenset[tuple[Node, _Index]]]
    | None = None,
) -> frozenset[tuple[Node, _Index]]:
    if memo is None:
        memo = {}
    key = (node, flat, projection)
    if key in memo:
        return memo[key]

    def evaluate(
        current: Node, current_flat: _Index, current_projection: int | None
    ) -> object:
        if (
            current is not node
            or current_flat != flat
            or current_projection != projection
        ):
            return _collect_demands(
                current,
                current_flat,
                region=region,
                config=config,
                projection=current_projection,
                memo=memo,
            )
        if current in region.boundaries:
            return frozenset({(current, current_flat)})
        if current.target is tile_index:
            return frozenset()
        source = current.args[0] if current.args else None
        if (
            current.target is memory_ops.load
            and isinstance(source, Node)
            and _is_host_tensor(source)
        ):
            output_shape = _shape(current, config)
            demands: set[tuple[Node, _Index]] = set()
            indices = current.args[1] if len(current.args) > 1 else None
            if not isinstance(indices, (list, tuple)):
                raise _UnsupportedFragment
            for index in indices:
                if isinstance(index, Node) and isinstance(
                    index.meta.get("val"), torch.Tensor
                ):
                    index_flat = _broadcast_flat(
                        current_flat, output_shape, _shape(index, config)
                    )
                    demands.update(
                        _collect_demands(
                            index, index_flat, region=region, config=config, memo=memo
                        )
                    )
            return frozenset(demands)
        inputs = _pointwise_inputs(current)
        if inputs is not None:
            demands: set[tuple[Node, _Index]] = set()
            output_shape = _shape(current, config)
            for input_node in inputs:
                demands.update(
                    _collect_demands(
                        input_node,
                        _broadcast_flat(
                            current_flat, output_shape, _shape(input_node, config)
                        ),
                        region=region,
                        config=config,
                        memo=memo,
                    )
                )
            return frozenset(demands)
        raise _UnsupportedFragment

    def choose(selector: _Index, left: object, right: object) -> object:
        if not isinstance(left, frozenset) or not isinstance(right, frozenset):
            raise _UnsupportedFragment
        return left | right

    result = _resolve_shape(
        node,
        flat,
        config=config,
        evaluate=evaluate,
        choose=choose,
        projection=projection,
    )
    if not isinstance(result, frozenset):
        raise _UnsupportedFragment
    memo[key] = result
    return result


def analyze_tcgen05_fragment_epilogue_plan(
    graphs: Sequence[GraphInfo],
    anchor: Node,
    *,
    expected_output_block_ids: tuple[int, ...],
    config: Config,
    bm: int,
    bn: int,
    bk: int,
    input_dtype: torch.dtype,
    source_global_shape: tuple[int | torch.SymInt, ...],
) -> Tcgen05FragmentEpiloguePlan | None:
    """Finalize exact thread-local fragment ownership for one configuration."""
    try:
        region = _extract_region(graphs, anchor, expected_output_block_ids)
        output_shape = _shape(region.value, config)
        if len(output_shape) != 3 or output_shape[0] != 1:
            raise _UnsupportedFragment
        source_shapes = {
            tuple(_shape(boundary, config)) for boundary in region.boundaries
        }
        if len(source_shapes) != 1:
            raise _UnsupportedFragment
        (source_shape_list,) = source_shapes
        source_shape = tuple(source_shape_list)
        destination_shape = tuple(output_shape)
        if (
            source_shape != (1, bm, bn)
            or destination_shape[-2] != bm
            or destination_shape[-1] not in (bn, bn // 2)
            or not _store_indices_are_tile_identity(
                region,
                source_shape,
                output_shape,
                source_global_shape=source_global_shape,
                expected_output_block_ids=expected_output_block_ids,
                config=config,
            )
        ):
            raise _UnsupportedFragment
        for boundary in region.boundaries:
            value = boundary.meta.get("val")
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype is not torch.float32
                or tuple(_shape(boundary, config)) != source_shape
            ):
                raise _UnsupportedFragment
        output_node = region.store.args[0] if region.store.args else None
        output_value = (
            output_node.meta.get("val") if isinstance(output_node, Node) else None
        )
        if not isinstance(output_value, torch.Tensor):
            raise _UnsupportedFragment
        store_tile_sizes: tuple[int | torch.SymInt, ...] = ()
        if destination_shape != source_shape and (
            not isinstance(output_value.shape[-2], int)
            or not isinstance(output_value.shape[-1], int)
            or output_value.shape[-2] % destination_shape[-2]
            or output_value.shape[-1] % destination_shape[-1]
        ):
            raise _UnsupportedFragment
        if destination_shape != source_shape:
            store_tile_sizes = _fragment_store_tile_sizes(
                cast("Sequence[object]", region.store.args[1])
            )
            if len(store_tile_sizes) != len(destination_shape):
                raise _UnsupportedFragment
        ownership = _query_tcgen05_fragment_ownership(
            bm=bm,
            bn=bn,
            bk=bk,
            destination_bm=destination_shape[-2],
            destination_bn=destination_shape[-1],
            input_dtype=input_dtype,
            output_dtype=output_value.dtype,
        )
        # The compact schedule reuses one register tensor for each source and
        # destination residency subtile. Distinct subtile *counts* are fine,
        # but a layout with different per-subtile register counts needs two
        # separately shaped buffers and is outside this first implementation.
        if ownership.source_register_count != ownership.destination_register_count:
            raise _UnsupportedFragment

        row = _Index.variable("row", destination_shape[-2] - 1)
        column = _Index.variable("column", destination_shape[-1] - 1)
        output_flat = _flat_from_coords(
            [_Index.constant(0), row, column], destination_shape
        )
        demands = tuple(
            sorted(
                _collect_demands(
                    region.value,
                    output_flat,
                    region=region,
                    config=config,
                ),
                key=lambda item: (item[0].name, str(item[1].semantic)),
            )
        )
        if not demands:
            raise _UnsupportedFragment
        demand_evaluators = tuple(index.compile() for _, index in demands)
        thread_programs: list[dict[tuple[int, int], tuple[int, tuple[int, ...]]]] = [
            {} for _ in range(ownership.thread_count)
        ]
        variables = {"row": 0, "column": 0}
        for m in range(destination_shape[-2]):
            variables["row"] = m
            for n in range(destination_shape[-1]):
                variables["column"] = n
                destination = ownership.destination_slots[m * destination_shape[-1] + n]
                source_slots: list[_FragmentSlot] = []
                for evaluate in demand_evaluators:
                    source_flat = evaluate(variables)
                    if not 0 <= source_flat < math.prod(source_shape):
                        raise _UnsupportedFragment
                    leading, matrix_flat = divmod(source_flat, bm * bn)
                    if leading != 0:
                        raise _UnsupportedFragment
                    source_slots.append(ownership.source_slots[matrix_flat])
                if any(slot.thread != destination.thread for slot in source_slots):
                    raise _UnsupportedFragment
                source_subtiles = {slot.subtile for slot in source_slots}
                if len(source_subtiles) != 1:
                    raise _UnsupportedFragment
                (source_subtile,) = source_subtiles
                key = (destination.subtile, destination.register)
                value = (source_subtile, tuple(slot.register for slot in source_slots))
                if key in thread_programs[destination.thread]:
                    raise _UnsupportedFragment
                thread_programs[destination.thread][key] = value

        reference = thread_programs[0]
        if not reference or any(program != reference for program in thread_programs):
            raise _UnsupportedFragment
        programs: list[_DestinationSubtileProgram] = []
        for destination_subtile in range(ownership.destination_subtile_count):
            grouped: dict[int, list[int]] = {}
            for destination_register in range(ownership.destination_register_count):
                source_subtile, _ = reference[
                    (destination_subtile, destination_register)
                ]
                grouped.setdefault(source_subtile, []).append(destination_register)
            groups: list[_SourceSubtileProgram] = []
            for source_subtile, destination_registers in sorted(grouped.items()):
                boundary_registers = tuple(
                    _BoundaryRegisterMap(
                        boundary,
                        index,
                        tuple(
                            reference[(destination_subtile, destination_register)][1][
                                demand_index
                            ]
                            for destination_register in destination_registers
                        ),
                    )
                    for demand_index, (boundary, index) in enumerate(demands)
                )
                groups.append(
                    _SourceSubtileProgram(
                        source_subtile,
                        tuple(destination_registers),
                        boundary_registers,
                    )
                )
            programs.append(_DestinationSubtileProgram(tuple(groups)))

        plan = Tcgen05FragmentEpiloguePlan(
            anchor=anchor,
            store_node=region.store,
            value_node=region.value,
            boundary_nodes=region.boundaries,
            owned_nodes=region.owned,
            source_shape=source_shape,
            destination_shape=destination_shape,
            store_tile_sizes=store_tile_sizes,
            programs=tuple(programs),
            source_register_count=ownership.source_register_count,
            destination_register_count=ownership.destination_register_count,
        )
        if not plan.changes_shape and plan.streaming_program is None:
            raise _UnsupportedFragment
        return plan
    except (_UnsupportedFragment, IndexError, StopIteration, ValueError):
        return None


def _render_statements(statements: Sequence[ast.AST], indent: str) -> str:
    return "".join(
        f"{indent}{line}\n"
        for statement in statements
        for line in ast.unparse(ast.fix_missing_locations(statement)).splitlines()
    )


def _register_index_expression(registers: Sequence[int], position: str) -> str:
    """Compress a uniform register map into a device-index expression."""
    if not registers:
        raise RuntimeError("empty committed register map")
    if len(registers) == 1:
        return str(registers[0])
    for period in range(1, len(registers)):
        constant = registers[0]
        within = registers[1] - constant if period > 1 else 0
        across = registers[period] - constant
        if all(
            register
            == across * (index // period) + within * (index % period) + constant
            for index, register in enumerate(registers)
        ):
            terms = []
            if across:
                terms.append(f"cutlass.Int32({across}) * ({position} // {period})")
            if within:
                terms.append(f"cutlass.Int32({within}) * ({position} % {period})")
            if constant or not terms:
                terms.append(f"cutlass.Int32({constant})")
            return "(" + " + ".join(terms) + ")"
    expression = f"cutlass.Int32({registers[-1]})"
    for index in range(len(registers) - 2, -1, -1):
        expression = (
            f"cutlass.Int32({registers[index]}) if {position} == {index} "
            f"else ({expression})"
        )
    return f"({expression})"


class _Evaluator:
    def __init__(
        self,
        state: CodegenState,
        plan: Tcgen05FragmentEpiloguePlan,
        *,
        carrier: str,
        destination_position: str,
        destination_coordinate: str,
        output_flat: _Index,
        boundary_registers: tuple[_BoundaryRegisterMap, ...],
        indent: str,
        terminals: dict[tuple[str, Node, _Index], ast.AST],
    ) -> None:
        self.state = state
        self.plan = plan
        self.carrier = carrier
        self.destination_position = destination_position
        self.output_flat = output_flat
        self.boundary_registers = boundary_registers
        self.indent = indent
        self.terminals = terminals
        self.memo: dict[tuple[Node, _Index, int | None], ast.AST] = {}
        self.lines: list[str] = []
        tile_shape = plan.destination_shape
        self.variables = {
            "row": (
                f"cutlass.Int32({destination_coordinate}[0]) % "
                f"cutlass.Int32({tile_shape[-2]})"
            ),
            "column": (
                f"cutlass.Int32({destination_coordinate}[1]) % "
                f"cutlass.Int32({tile_shape[-1]})"
            ),
        }

    def _bind(self, prefix: str, value: ast.AST) -> ast.AST:
        name = self.state.device_function.new_var(prefix)
        self.lines.append(
            _render_statements(
                [statement_from_string(f"{name} = {{value}}", value=value)], self.indent
            )
        )
        return expr_from_string(name)

    def _boundary(self, node: Node, flat: _Index) -> ast.AST:
        key = ("boundary", node, flat)
        if key in self.terminals:
            return self.terminals[key]
        mapping = next(
            (
                mapping
                for mapping in self.boundary_registers
                if mapping.boundary is node and mapping.index == flat
            ),
            None,
        )
        if mapping is None:
            raise RuntimeError(
                f"missing committed register mapping for {node.name}: {flat.semantic}"
            )
        source_register = _register_index_expression(
            mapping.registers, self.destination_position
        )
        expression = f"{self.carrier}[{source_register}]"
        result = self._bind("tcgen05_epi_acc", expr_from_string(expression))
        self.terminals[key] = result
        return result

    def _tile_index(self, node: Node, flat: _Index) -> ast.AST:
        from ...language.memory_ops import _cute_tile_begin_expr

        key = ("index", node, flat)
        if key in self.terminals:
            return self.terminals[key]
        size_node = node.args[0] if node.args else None
        size = size_node.meta.get("val") if isinstance(size_node, Node) else size_node
        coord = _coords_from_flat(
            flat, _shape(node, self.state.device_function.config)
        )[0]
        result = self._bind(
            "tcgen05_epi_index",
            expr_from_string(
                f"cutlass.Int32({_cute_tile_begin_expr(self.state, size)}) + "
                f"cutlass.Int32({coord.render(self.variables)})"
            ),
        )
        self.terminals[key] = result
        return result

    def _host_load(self, node: Node, flat: _Index) -> ast.AST:
        from ...language.memory_ops import _cute_scalar_load_expr
        from ...language.memory_ops import _cute_tensor_dim_size_expr
        from ...language.memory_ops import _cute_tile_begin_expr

        key = ("load", node, flat)
        if key in self.terminals:
            return self.terminals[key]
        source = node.args[0]
        indices = node.args[1]
        assert isinstance(source, Node) and isinstance(indices, (list, tuple))
        tensor = source.meta["val"]
        assert isinstance(tensor, torch.Tensor)
        tensor_name = self.state.device_function.tensor_arg(tensor).name
        self.state.device_function.placeholder_args.add(tensor_name)
        output_shape = _shape(node, self.state.device_function.config)
        output_coords = _coords_from_flat(flat, output_shape)
        tensor_indices = [
            index
            for index in indices
            if isinstance(index, Node)
            and isinstance(index.meta.get("val"), torch.Tensor)
        ]
        index_exprs: list[str] = []
        bounds: list[str] = []
        env = CompileEnvironment.current()
        if tensor_indices:
            index_dtype = env.index_type()
            for tensor_dim, index in enumerate(tensor_indices):
                index_flat = _broadcast_flat(
                    flat, output_shape, _shape(index, self.state.device_function.config)
                )
                index_expr = ast.unparse(self.evaluate(index, index_flat))
                index_exprs.append(index_expr)
                if _tile_index_uses_arithmetic(index):
                    dim_size = _cute_tensor_dim_size_expr(
                        self.state, tensor, tensor_dim
                    )
                    bounds.append(
                        f"({index_dtype}({index_expr}) >= {index_dtype}(0) and "
                        f"{index_dtype}({index_expr}) < {index_dtype}({dim_size}))"
                    )
        else:
            output_dim = 0
            for index in indices:
                if isinstance(index, Node) and isinstance(
                    index.meta.get("val"), torch.SymInt
                ):
                    extent = index.meta["val"]
                elif index == slice(None):
                    extent = node.meta["val"].shape[output_dim]
                else:
                    assert isinstance(index, int)
                    index_exprs.append(str(index))
                    continue
                index_exprs.append(
                    f"cutlass.Int32({_cute_tile_begin_expr(self.state, extent)}) + "
                    f"cutlass.Int32({output_coords[output_dim].render(self.variables)})"
                )
                output_dim += 1
        load_expr = _cute_scalar_load_expr(tensor_name, index_exprs, tensor.dtype)
        if bounds:
            dtype = env.backend.dtype_str(tensor.dtype)
            load_expr = f"({load_expr} if {' and '.join(bounds)} else {dtype}(0))"
        result = self._bind(
            "tcgen05_epi_load",
            expr_from_string(load_expr),
        )
        self.terminals[key] = result
        return result

    def _pointwise(self, node: Node, flat: _Index, inputs: tuple[Node, ...]) -> ast.AST:
        from ..aten_lowering import LoweringContext
        from ..inductor_lowering import PointwiseLowering

        lowering = node.meta.get("lowering")
        if not isinstance(lowering, PointwiseLowering):
            raise RuntimeError(f"non-pointwise fragment node {node.name}")
        output_shape = _shape(node, self.state.device_function.config)
        input_values = [
            self.evaluate(
                input_node,
                _broadcast_flat(
                    flat,
                    output_shape,
                    _shape(input_node, self.state.device_function.config),
                ),
            )
            for input_node in inputs
        ]
        ctx = LoweringContext.__new__(LoweringContext)
        ctx.cg = self.state.codegen
        ctx.env = self.state.env
        statements: list[ast.AST] = []
        with (
            self.state.codegen.set_statements(statements),
            V.set_current_node(node),
        ):
            result = lowering.codegen_from_input_asts(ctx, node, input_values)
        if not isinstance(result, ast.AST):
            raise RuntimeError(f"invalid pointwise fragment result for {node.name}")
        self.lines.append(_render_statements(statements, self.indent))
        return self._bind("tcgen05_epi_value", result)

    def evaluate(
        self, node: Node, flat: _Index, projection: int | None = None
    ) -> ast.AST:
        key = (node, flat, projection)
        if key in self.memo:
            return self.memo[key]

        def leaf(
            current: Node, current_flat: _Index, current_projection: int | None
        ) -> object:
            if (
                current is not node
                or current_flat != flat
                or current_projection != projection
            ):
                return self.evaluate(current, current_flat, current_projection)
            if current in self.plan.boundary_nodes:
                return self._boundary(current, current_flat)
            if current.target is tile_index:
                return self._tile_index(current, current_flat)
            source = current.args[0] if current.args else None
            if (
                current.target is memory_ops.load
                and isinstance(source, Node)
                and _is_host_tensor(source)
            ):
                return self._host_load(current, current_flat)
            inputs = _pointwise_inputs(current)
            if inputs is not None:
                return self._pointwise(current, current_flat, inputs)
            raise RuntimeError(f"unsupported committed fragment node {current.name}")

        def choose(selector: _Index, left: object, right: object) -> object:
            if not isinstance(left, ast.AST) or not isinstance(right, ast.AST):
                raise RuntimeError("invalid fragment selection")
            if (constant := _constant_index(selector)) is not None:
                return left if constant == 0 else right
            return self._bind(
                "tcgen05_epi_select",
                expr_from_string(
                    f"({{left}} if ({selector.render(self.variables)}) == "
                    "cutlass.Int32(0) else {right})",
                    left=left,
                    right=right,
                ),
            )

        result = _resolve_shape(
            node,
            flat,
            config=self.state.device_function.config,
            evaluate=leaf,
            choose=choose,
            projection=projection,
        )
        if not isinstance(result, ast.AST):
            raise RuntimeError(f"invalid committed fragment result for {node.name}")
        self.memo[key] = result
        return result


def _render_tcgen05_fragment_group(
    state: CodegenState,
    plan: Tcgen05FragmentEpiloguePlan,
    program: _SourceSubtileProgram,
    *,
    carrier_name: str,
    coordinate_name: str,
    target_dtype: str,
    indent: str,
    output_name: str | None = None,
) -> tuple[str, str]:
    output_shape = plan.destination_shape
    row = _Index.variable("row", output_shape[-2] - 1)
    column = _Index.variable("column", output_shape[-1] - 1)
    output_flat = _flat_from_coords([_Index.constant(0), row, column], output_shape)
    df = state.device_function
    output = output_name or df.new_var("tcgen05_epi_output")
    destination_position = df.new_var("tcgen05_epi_position")
    destination_index = df.new_var("tcgen05_epi_register")
    destination_coordinate = df.new_var("tcgen05_epi_coord")
    body_indent = indent + "    "
    terminals: dict[tuple[str, Node, _Index], ast.AST] = {}
    evaluator = _Evaluator(
        state,
        plan,
        carrier=carrier_name,
        destination_position=destination_position,
        destination_coordinate=destination_coordinate,
        output_flat=output_flat,
        boundary_registers=program.boundary_registers,
        indent=body_indent,
        terminals=terminals,
    )
    value = evaluator.evaluate(plan.value_node, output_flat)
    destination_register = _register_index_expression(
        program.destination_registers, destination_position
    )
    output_setup = (
        f"{indent}{output} = cute.make_rmem_tensor("
        f"cute.make_layout({carrier_name}.shape), {target_dtype})\n"
        if output_name is None
        else ""
    )
    prelude = (
        f"{indent}assert cute.size({carrier_name}.shape) == "
        f"{plan.source_register_count}, "
        '"tcgen05 source fragment must be complete"\n'
        f"{output_setup}"
        f"{indent}for {destination_position} in "
        f"range({len(program.destination_registers)}):\n"
        f"{body_indent}{destination_index} = "
        f"{destination_register}\n"
        f"{body_indent}{destination_coordinate} = "
        f"{coordinate_name}[{destination_index}]\n"
        f"{''.join(evaluator.lines)}"
        f"{body_indent}{output}[{destination_index}] = "
        f"{target_dtype}({ast.unparse(value)})\n"
    )
    return prelude, output


def render_tcgen05_fragment_epilogue(
    state: CodegenState,
    plan: Tcgen05FragmentEpiloguePlan,
    *,
    carrier_name: str,
    coordinate_name: str,
    target_dtype: str,
    indent: str,
) -> tuple[str, str]:
    """Render a uniform same-shape thread-local register program."""
    program = plan.streaming_program
    if program is None:
        raise RuntimeError("shape-changing fragment requires the scheduled renderer")
    prelude, output = _render_tcgen05_fragment_group(
        state,
        plan,
        program,
        carrier_name=carrier_name,
        coordinate_name=coordinate_name,
        target_dtype=target_dtype,
        indent=indent,
    )
    return prelude, f"{output}.load()"


def render_tcgen05_fragment_epilogue_group(
    state: CodegenState,
    plan: Tcgen05FragmentEpiloguePlan,
    program: _SourceSubtileProgram,
    *,
    carrier_name: str,
    destination_name: str,
    coordinate_name: str,
    target_dtype: str,
    indent: str,
) -> str:
    """Fill one destination fragment from its resident source subtile."""
    prelude, _ = _render_tcgen05_fragment_group(
        state,
        plan,
        program,
        carrier_name=carrier_name,
        coordinate_name=coordinate_name,
        target_dtype=target_dtype,
        indent=indent,
        output_name=destination_name,
    )
    return prelude
