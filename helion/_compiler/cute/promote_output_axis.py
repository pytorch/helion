"""Promote a partitioned device-loop axis into the CuTe launch grid.

The remaining device-loop axes retain their original order. In particular, a
gathered row axis stays serial: this transformation does not assume that its
indices are unique. Every write, and every load from a written allocation, must
use either the promoted tile as the exact final subscript of one contiguous
tensor view, or a proved disjoint column residue in flattened storage. The
implementation only writes fresh host allocations, so input
aliasing cannot invalidate that partition proof on a later bound-kernel call.
"""

from __future__ import annotations

import ast
import dataclasses
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...autotuner.config_spec import L2GroupingSpec
from ...autotuner.config_spec import LoopOrderSpec
from ...language import _tracing_ops
from ...language import creation_ops
from ...language import memory_ops
from ...language import tile_ops
from ...language import view_ops
from ...language.constexpr import specialize
from ...language.tunable_ops import register_block_size
from ..ast_extension import ExtendedAST
from ..ast_extension import LoopType
from ..ast_extension import create
from ..compile_environment import CompileEnvironment
from ..device_ir import ForLoopGraphInfo
from ..device_ir import RootGraphInfo
from ..device_ir import control_flow_parent_entries
from ..indexing_strategy import subscript_tile_info
from ..inductor_lowering import codegen_call_with_graph
from ..tile_dependency import TaskAxis
from ..type_info import CallableType
from ..type_info import GridIndexType
from ..type_info import IterType
from ..type_info import SequenceType
from ..type_info import TensorAttributeType
from ..type_info import TensorType
from ..type_info import TileIndexType
from .flat_index_partition import is_flat_column_partition

if TYPE_CHECKING:
    from ...autotuner.block_id_sequence import BlockIdSequence
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from ..host_function import HostFunction
    from ..inductor_lowering import CodegenState
    from ..type_info import TypeInfo


_FRESH_FACTORIES = frozenset(
    (
        torch.empty,
        torch.empty_like,
        torch.zeros,
        torch.zeros_like,
        torch.ones,
        torch.ones_like,
        torch.full,
        torch.full_like,
    )
)
_READ_ONLY_HOST_CALLS = _FRESH_FACTORIES | frozenset(
    (
        torch.as_strided,
        torch.detach,
        torch.flatten,
        torch.permute,
        torch.promote_types,
        torch.reshape,
        torch.squeeze,
        torch.t,
        torch.transpose,
        torch.unsqueeze,
        len,
        slice,
    )
)
_READ_ONLY_TENSOR_METHODS = frozenset(
    {
        "as_strided",
        "detach",
        "dim",
        "element_size",
        "flatten",
        "is_contiguous",
        "ndimension",
        "numel",
        "permute",
        "reshape",
        "size",
        "squeeze",
        "stride",
        "t",
        "transpose",
        "unsqueeze",
        "view",
    }
)
_PURE_CALLS = frozenset(
    (
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        _tracing_ops._new_var,
        _tracing_ops._phi,
        _tracing_ops._for_loop,
        _tracing_ops._for_loop_step,
        _tracing_ops._if,
        _tracing_ops._mask_to,
        creation_ops.full,
        view_ops.subscript,
        tile_ops.tile_index,
        tile_ops.tile_begin,
        tile_ops.tile_end,
        tile_ops.tile_id,
        operator.getitem,
        operator.add,
        operator.sub,
        operator.mul,
        operator.floordiv,
        operator.mod,
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    )
)


def _empty_output(graph: torch.fx.Graph) -> bool:
    output = next(iter(graph.find_nodes(op="output")))
    return output.args[0] is None or output.args[0] == [] or output.args[0] == ()


def _shape_metadata(value: object) -> bool:
    if type(value) in (int, bool, torch.SymInt):
        return True
    if type(value) in (tuple, list, torch.Size):
        return all(
            _shape_metadata(item)
            for item in cast("tuple[object, ...] | list[object]", value)
        )
    return False


def _read_only_host_dsl(call: ast.Call, callee: CallableType) -> bool:
    # These exact DSL APIs specialize integer metadata or register a tile size;
    # they do not mutate tensor storage. Opaque containers and tensor arguments
    # are outside this proof. Nested argument calls are checked independently.
    return any(
        callee.value is function for function in (specialize, register_block_size)
    ) and all(
        isinstance(argument, ExtendedAST)
        and (metadata_type := argument._type_info) is not None
        and _shape_metadata(metadata_type.proxy())
        for argument in (*call.args, *(keyword.value for keyword in call.keywords))
    )


def _fresh_tensors(host: HostFunction) -> set[torch.Tensor]:
    """Prove that factory storage survives all host-side operations.

    Host tracing can mutate the same FakeTensor retained by an allocation's
    type metadata. After ``out.set_(input)``, even that allocation node then
    names input storage. Unknown calls or host writes therefore invalidate the
    entire proof; a factory's current storage alone cannot establish freshness.
    Check statements between and after grids too: the retained metadata is
    shared with those later operations, and packet lowering uses this proof
    for every device graph. Only verified GRID bodies execute on the device;
    their iterator arguments still run on the host. Known non-mutating views
    retain the existing storage/partition checks.
    """
    result: set[torch.Tensor] = set()
    storages: set[int] = set()
    for statement in host.body:
        if isinstance(statement, ast.For):
            if (
                not isinstance(statement, ExtendedAST)
                or statement._loop_type is not LoopType.GRID
                or not isinstance(statement.iter, ast.Call)
            ):
                return set()
            # Type propagation proves this outer call is a device-loop API.
            # Its children may still contain arbitrary host calls.
            expressions = list(ast.walk(statement.iter))[1:]
        elif isinstance(
            statement, (ast.Assign, ast.AnnAssign, ast.Expr, ast.Assert, ast.Return)
        ):
            expressions = list(ast.walk(statement))
        else:
            return set()
        if any(
            isinstance(node, (ast.Attribute, ast.Subscript))
            and isinstance(node.ctx, (ast.Store, ast.Del))
            or isinstance(node, (ast.NamedExpr, ast.Lambda))
            for node in expressions
        ):
            return set()
        for call in expressions:
            if not isinstance(call, ast.Call):
                continue
            if not isinstance(call.func, ExtendedAST) or any(
                keyword.arg in (None, "out") for keyword in call.keywords
            ):
                return set()
            callee = call.func._type_info
            if not (
                isinstance(callee, CallableType)
                and (
                    any(callee.value is function for function in _READ_ONLY_HOST_CALLS)
                    or _read_only_host_dsl(call, callee)
                )
                or isinstance(callee, TensorAttributeType)
                and callee.attr() in _READ_ONLY_TENSOR_METHODS
            ):
                return set()
            if (
                isinstance(call, ExtendedAST)
                and isinstance(callee, CallableType)
                and any(callee.value is factory for factory in _FRESH_FACTORIES)
                and isinstance(value_type := call._type_info, TensorType)
            ):
                storages.add(value_type.proxy().untyped_storage()._cdata)
        for expression in expressions:
            if (
                isinstance(expression, ExtendedAST)
                and isinstance(value_type := expression._type_info, TensorType)
                and (tensor := value_type.proxy()).untyped_storage()._cdata in storages
            ):
                result.add(tensor)
    return result


def _ordinary_operation(node: torch.fx.Node) -> bool:
    if node.op in ("placeholder", "output"):
        return True
    if node.op != "call_function":
        return False
    target = node.target
    if target in _PURE_CALLS or target in (memory_ops.load, memory_ops.store):
        return True
    return (
        isinstance(target, torch._ops.OpOverload)
        and not target._schema.is_mutable
        and torch.Tag.nondeterministic_seeded not in target.tags
        and not node.is_impure()
    )


def _access_tensor(node: torch.fx.Node) -> torch.Tensor | None:
    source = node.args[0]
    if isinstance(source, torch.fx.Node):
        value = source.meta.get("val")
        if isinstance(value, torch.Tensor):
            return value
    return None


def _is_exact_partition(
    node: torch.fx.Node,
    tensor: torch.Tensor,
    block_id: int,
    env: CompileEnvironment,
) -> bool:
    indices = node.args[1]
    if not isinstance(indices, (list, tuple)) or len(indices) != tensor.ndim:
        return False
    if tensor.ndim == 1 and env.known_equal(tensor.stride(0), 1):
        if is_flat_column_partition(indices[0], block_id, env):
            return True
    info = subscript_tile_info(env, indices[-1])
    if (
        info is None
        or info.block_id != block_id
        or not env.known_equal(info.offset, 0)
        or info.block_size is not None
    ):
        return False
    extent = env.block_sizes[block_id].size
    if not isinstance(extent, (int, torch.SymInt)) or not env.known_equal(
        tensor.shape[-1], extent
    ):
        return False
    # Row-major contiguous layout gives each final-axis coordinate a unique
    # residue modulo the row width, even when another index is an arbitrary
    # gather. Reject overlapping/strided views rather than infer injectivity.
    stride: int | torch.SymInt = 1
    for size, actual_stride in reversed(
        tuple(zip(tensor.shape, tensor.stride(), strict=True))
    ):
        if not env.known_equal(actual_stride, stride):
            return False
        stride = stride * size
    return True


def _partitioned_memory(
    device_ir: DeviceIR,
    fresh: set[torch.Tensor],
    block_id: int,
    env: CompileEnvironment,
) -> bool:
    accesses: list[tuple[torch.fx.Node, torch.Tensor]] = []
    written: dict[int, torch.Tensor] = {}
    for graph in device_ir.graphs:
        for node in graph.graph.nodes:
            if not _ordinary_operation(node):
                return False
            if node.target not in (memory_ops.load, memory_ops.store):
                continue
            tensor = _access_tensor(node)
            if tensor is None:
                return False
            accesses.append((node, tensor))
            if node.target is memory_ops.store:
                if tensor not in fresh or not _is_exact_partition(
                    node, tensor, block_id, env
                ):
                    return False
                written[tensor.untyped_storage()._cdata] = tensor
    if not written:
        return False
    for node, tensor in accesses:
        stored_view = written.get(tensor.untyped_storage()._cdata)
        if stored_view is not None and (
            tensor is not stored_view
            or not _is_exact_partition(node, tensor, block_id, env)
        ):
            return False
    return True


def _indices(loop: ast.For) -> list[TileIndexType | GridIndexType] | None:
    if not isinstance(loop.iter, ExtendedAST) or not isinstance(
        info := loop.iter._type_info, IterType
    ):
        return None
    values = (
        list(info.inner.unpack())
        if isinstance(info.inner, SequenceType)
        else [info.inner]
    )
    if not all(isinstance(value, (TileIndexType, GridIndexType)) for value in values):
        return None
    return cast("list[TileIndexType | GridIndexType]", values)


def _canonical_iterator(loop: ast.For) -> list[ast.expr] | None:
    if not isinstance(loop.iter, ast.Call) or loop.iter.keywords:
        return None
    if (
        len(loop.iter.args) == 2
        and isinstance(loop.iter.args[0], ast.Constant)
        and type(loop.iter.args[0].value) is int
        and loop.iter.args[0].value == 0
    ):
        return [loop.iter.args[1]]
    if len(loop.iter.args) != 1:
        return None
    arg = loop.iter.args[0]
    return list(arg.elts) if isinstance(arg, (ast.List, ast.Tuple)) else [arg]


def _typed_sequence(
    values: list[ast.expr],
    *,
    target: bool = False,
    types: list[TypeInfo] | None = None,
) -> ast.expr:
    infos = types or [cast("ExtendedAST", value)._type_info for value in values]
    assert all(info is not None for info in infos)
    origin = cast("TypeInfo", infos[0]).origin
    return create(
        ast.Tuple if target else ast.List,
        elts=values,
        ctx=ast.Store() if target else ast.Load(),
        _type_info=SequenceType(origin, tuple(cast("list[TypeInfo]", infos))),
    )


def _rewrite_ast(root: ast.For, loop: ast.For) -> None:
    root_indices = _indices(root)
    loop_indices = _indices(loop)
    root_ends = _canonical_iterator(root)
    loop_ends = _canonical_iterator(loop)
    assert root_indices and loop_indices and root_ends and loop_ends
    assert isinstance(root.iter, ExtendedAST)
    assert isinstance(root.iter, ast.Call)
    assert isinstance(loop.iter, ExtendedAST)
    assert isinstance(loop.iter, ast.Call)
    loop_targets = (
        list(loop.target.elts)
        if isinstance(loop.target, (ast.List, ast.Tuple))
        else [loop.target]
    )
    root_targets = (
        list(root.target.elts)
        if isinstance(root.target, (ast.List, ast.Tuple))
        else [root.target]
    )
    with root.iter:
        root.iter.args = [_typed_sequence([*root_ends, loop_ends[-1]])]
        origin = cast("IterType", root.iter._type_info).origin
        root.iter._type_info = IterType(
            origin, SequenceType(origin, tuple([*root_indices, loop_indices[-1]]))
        )
        root.target = _typed_sequence(
            [*root_targets, loop_targets[-1]],
            target=True,
            types=[*root_indices, loop_indices[-1]],
        )
    if len(loop_indices) == 1:
        for parent in ast.walk(root):
            for _name, field in ast.iter_fields(parent):
                if isinstance(field, list):
                    for position, child in enumerate(field):
                        if child is loop:
                            field[position : position + 1] = loop.body
                            return
        raise AssertionError("promoted loop is not under its root")
    with loop.iter:
        loop.iter.args = [_typed_sequence(loop_ends[:-1])]
        origin = cast("IterType", loop.iter._type_info).origin
        loop.iter._type_info = IterType(
            origin, SequenceType(origin, tuple(loop_indices[:-1]))
        )
        loop.target = _typed_sequence(
            loop_targets[:-1], target=True, types=list(loop_indices[:-1])
        )


class _PromotedGridAxisGraphInfo(ForLoopGraphInfo):
    """A fully promoted, output-free loop whose capture scope stays in place."""

    @property
    def name(self) -> str:
        return f"promoted_grid_axis_{self.graph_id}"

    def codegen(self, state: CodegenState) -> list[object]:
        args = state.ast_args[3]
        assert isinstance(args, list)
        return codegen_call_with_graph(state.codegen, self.graph, args)


def _rewrite_config(
    env: CompileEnvironment, root_ids: list[int], loop_ids: list[int]
) -> None:
    spec = env.config_spec
    promoted = loop_ids[-1]
    new_root = [*root_ids, promoted]
    remaining_loop = loop_ids[:-1]
    for sequence in (spec.loop_orders, spec.flatten_loops, spec.l2_groupings):
        for block_id in (*root_ids, *loop_ids):
            sequence.disable_block_id(block_id)
    spec.loop_orders.append(LoopOrderSpec(new_root))
    spec.l2_groupings.append(L2GroupingSpec(new_root))
    if len(remaining_loop) > 1:
        spec.loop_orders.append(LoopOrderSpec(remaining_loop))
    # Range controls are attached to the root's persistent loop as a group,
    # whereas each serial device-loop axis has a separate range slot.
    range_sequences: tuple[BlockIdSequence, ...] = (
        spec.range_unroll_factors,
        spec.range_warp_specialize,
        spec.range_num_stages,
        spec.range_multi_buffers,
        spec.range_flattens,
        spec.static_ranges,
    )
    for sequence in range_sequences:
        sequence.disable_block_id(promoted)
        for index, item in enumerate(sequence):
            if root_ids[0] in item.block_ids:
                item.block_ids = new_root.copy()
                sequence[index] = item
                break
    spec.cute_reflatten_candidates = [
        candidate
        for candidate in spec.cute_reflatten_candidates
        if not set(candidate.block_ids).intersection((*root_ids, *loop_ids))
    ]
    spec.grid_block_ids.append(promoted)


def promote_partitioned_output_axis(
    host: HostFunction, device_ir: DeviceIR, root_nodes: list[ast.For]
) -> bool:
    """Expose one independent trailing tile axis without changing memory order.

    Limit promotion to scalar grid coordinates (including tile axes whose
    maximum block size is one) and a canonical device loop. Other tiled roots
    retain their positional loop-order and flattening configuration schema.
    An ancestor may branch, but may not carry a register value out of the loop
    being distributed. Full-extent tile sizes retain one partition, giving the
    existing autotuner a serial-equivalent choice without a second enable flag.
    """
    if len(root_nodes) != 1:
        return False
    env = CompileEnvironment.current()
    root = root_nodes[0]
    root_indices = _indices(root)
    block_specs = {
        block_id: spec
        for spec in env.config_spec.block_sizes
        for block_id in spec.block_ids
    }
    if (
        not root_indices
        or not all(
            isinstance(index, GridIndexType)
            or (
                (spec := block_specs.get(index.block_id)) is not None
                and spec.max_size == 1
            )
            for index in root_indices
        )
        or _canonical_iterator(root) is None
    ):
        return False
    parents = control_flow_parent_entries(device_ir.graphs)
    fresh = _fresh_tensors(host)
    for loop in ast.walk(root):
        if not isinstance(loop, ast.For) or not isinstance(loop, ExtendedAST):
            continue
        if loop._loop_type is not LoopType.DEVICE:
            continue
        indices = _indices(loop)
        ends = _canonical_iterator(loop)
        if (
            indices is None
            or not indices
            or not all(isinstance(index, TileIndexType) for index in indices)
            or ends is None
            or len(ends) != len(indices)
        ):
            continue
        block_ids = [index.block_id for index in indices]
        promoted = block_ids[-1]
        extent = env.block_sizes[promoted].size
        # A launch-grid extent must be host-known. Reject scalar loads and
        # jagged bounds even if the particular binding gives them a size hint.
        if not isinstance(extent, (int, torch.SymInt)) or promoted in (
            device_ir.noncanonical_task_origin_block_ids
        ):
            continue
        end_type = cast("ExtendedAST", ends[-1])._type_info
        if end_type is None or not end_type.origin.is_host():
            continue
        graph = next(
            (
                graph
                for graph in device_ir.graphs
                if type(graph) is ForLoopGraphInfo and graph.block_ids == block_ids
            ),
            None,
        )
        if graph is None:
            continue
        if any(
            isinstance(other, ForLoopGraphInfo)
            and other is not graph
            and promoted in other.block_ids
            for other in device_ir.graphs
        ):
            continue
        ancestor: GraphInfo = graph
        while ancestor.graph_id in parents and _empty_output(ancestor.graph):
            parent, _ = parents[ancestor.graph_id]
            ancestor = next(
                item for item in device_ir.graphs if item.graph is parent.graph
            )
        if not isinstance(ancestor, RootGraphInfo):
            continue
        if not _partitioned_memory(device_ir, fresh, promoted, env):
            continue
        parent, _ = parents[graph.graph_id]
        if parent.target is not _tracing_ops._for_loop:
            continue
        begins, ends = parent.args[1:3]
        assert isinstance(begins, (list, tuple))
        assert isinstance(ends, (list, tuple))
        # All checks precede mutations: rejected kernels retain identical IR,
        # AST, block IDs, and positional autotuner configuration schemas.
        _rewrite_ast(root, loop)
        root_ids = device_ir.grid_block_ids[0]
        _rewrite_config(env, root_ids, block_ids)
        graph.block_ids = block_ids[:-1]
        if not graph.block_ids:
            device_ir.graphs[graph.graph_id] = _PromotedGridAxisGraphInfo(
                graph_id=graph.graph_id,
                graph=graph.graph,
                node_args=graph.node_args,
                block_ids=[],
                host_loop_reads=graph.host_loop_reads,
                host_loop_writes=graph.host_loop_writes,
            )
        parent.args = (
            parent.args[0],
            begins[:-1],
            ends[:-1],
            *parent.args[3:],
        )
        root_ids.append(promoted)
        family = device_ir.task_families[0]
        device_ir.task_families[0] = dataclasses.replace(
            family,
            axes=(
                *family.axes,
                TaskAxis(promoted, env.block_sizes[promoted].numel, True),
            ),
        )
        return True
    return False
