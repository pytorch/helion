"""Prove a zero-seeded grouped contraction with flat A/D addresses.

The result is an IR proof, not runtime admission. Dimensions, strides, pointer
alignment, storage spans, RNA numerical mode and complete resources still need
the explicit schedule/host guards. Unknown programs retain ordinary codegen.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import TYPE_CHECKING
from typing import TypeGuard

import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import creation_ops
from ...language import memory_ops
from ...language import tile_ops
from ...language import view_ops
from ..device_ir import ForLoopGraphInfo
from ..device_ir import NodeArgsGraphInfo
from ..device_ir import RootGraphInfo
from ..indexing_strategy import subscript_tile_info
from .promote_output_axis import _PromotedGridAxisGraphInfo
from .tcgen05_flat_grouped_host import prove_flat_grouped_host

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from ..host_function import HostFunction


@dataclass(frozen=True)
class FlatGroupedIRProof:
    root_graph_id: int
    group_axis: int
    row_axis: int
    column_axis: int
    reduction_axis: int
    offset_bits: int
    offsets_argument: str
    a_argument: str
    b_argument: str
    bias_argument: str
    a_flat_expression: str
    output_flat_expression: str
    offsets: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    bias: torch.Tensor
    output: torch.Tensor
    contraction_node: Node
    matched_calls: int


def _call(value: object, target: object, count: int) -> TypeGuard[Node]:
    return (
        isinstance(value, Node)
        and value.op == "call_function"
        and value.target is target
        and not value.kwargs
        and len(value.args) == count
    )


def _tensor(value: object) -> torch.Tensor | None:
    if isinstance(value, Node):
        result = value.meta.get("val")
        if isinstance(result, torch.Tensor):
            return result
    return None


def _output(info: GraphInfo) -> object:
    nodes = tuple(info.graph.find_nodes(op="output"))
    return nodes[0].args[0] if len(nodes) == 1 else object()


def _loop_stop(loop: Node) -> object:
    # _Matcher.child has already established this loop-call schema.
    ends = loop.args[2]
    assert isinstance(ends, (tuple, list)) and len(ends) == 1
    return ends[0]


class _Matcher:
    def __init__(self, env: CompileEnvironment, ir: DeviceIR) -> None:
        self.env = env
        self.ir = ir
        self.infos = {info.graph: info for info in ir.graphs}
        self.matched: set[Node] = set()

    def strip(self, value: object) -> object:
        seen: set[Node] = set()
        while isinstance(value, Node) and value not in seen:
            seen.add(value)
            if _call(value, _tracing_ops._new_var, 1):
                self.matched.add(value)
                value = value.args[0]
            elif value.op == "placeholder":
                info = self.infos[value.graph]
                if not isinstance(info, NodeArgsGraphInfo):
                    return None
                placeholders = tuple(info.graph.find_nodes(op="placeholder"))
                if len(placeholders) != len(info.node_args):
                    return None
                value = info.node_args[placeholders.index(value)]
            else:
                return value
        return None

    def unbroadcast(self, value: object) -> object:
        value = self.strip(value)
        while _call(value, view_ops.subscript, 2):
            assert isinstance(value, Node)
            source, indices = value.args
            tensor = _tensor(source)
            if tensor is None or not isinstance(indices, (tuple, list)):
                return None
            if (
                any(
                    item is not None
                    and not (
                        isinstance(item, slice)
                        and item.start is item.stop is item.step is None
                    )
                    for item in indices
                )
                or sum(item is not None for item in indices) != tensor.ndim
            ):
                return None
            self.matched.add(value)
            value = self.strip(source)
        return value

    def axis(self, value: object) -> int | None:
        value = self.unbroadcast(value)
        if _call(value, tile_ops.tile_index, 1):
            assert isinstance(value, Node)
            self.matched.add(value)
            value = value.args[0]
        tile = subscript_tile_info(self.env, value)
        if (
            tile is not None
            and tile.block_size is None
            and self.env.known_equal(tile.offset, 0)
        ):
            return self.env.canonical_block_id(tile.block_id)
        return None

    def scalar_equal(self, value: object, expected: object) -> bool:
        if isinstance(value, Node):
            if not _call(value, _tracing_ops._get_symnode, 1):
                return False
            self.matched.add(value)
            value = value.meta.get("val")
        return (
            isinstance(value, (int, torch.SymInt))
            and isinstance(expected, (int, torch.SymInt))
            and self.env.known_equal(value, expected)
        )

    def extent_equal(self, axis: int, expected: int | torch.SymInt) -> bool:
        extent = self.env.block_sizes[axis].size
        return isinstance(extent, (int, torch.SymInt)) and self.env.known_equal(
            extent, expected
        )

    def host_load(self, value: object) -> tuple[Node, torch.Tensor, str, object] | None:
        if not _call(value, memory_ops.load, 4):
            return None
        assert isinstance(value, Node)
        source, indices, mask, other = value.args
        if (
            mask is not None
            or other is not None
            or not _call(source, _tracing_ops._host_tensor, 1)
        ):
            return None
        assert isinstance(source, Node)
        tensor = _tensor(source)
        name = source.args[0]
        if tensor is None or not isinstance(name, str) or not name.isidentifier():
            return None
        self.matched.update((value, source))
        return value, tensor, name, indices

    def direct_load(
        self, value: object, axes: tuple[int, ...]
    ) -> tuple[torch.Tensor, str] | None:
        load = self.host_load(value)
        if load is None:
            return None
        node, tensor, name, indices = load
        if not isinstance(indices, (tuple, list)) or len(indices) != len(axes):
            return None
        if tensor.ndim != len(axes) or tuple(self.axis(x) for x in indices) != axes:
            return None
        result = _tensor(node)
        if result is None or result.dtype is not tensor.dtype:
            return None
        return tensor, name

    def integer_call(
        self, value: object, target: object, dtype: torch.dtype
    ) -> Node | None:
        value = self.unbroadcast(value)
        tensor = _tensor(value)
        if not _call(value, target, 2) or tensor is None or tensor.dtype is not dtype:
            return None
        assert isinstance(value, Node)
        self.matched.add(value)
        return value

    def row(
        self, value: object, start: Node, row_axis: int, dtype: torch.dtype
    ) -> Node | None:
        node = self.integer_call(value, torch.ops.aten.add.Tensor, dtype)
        if node is None:
            return None
        for first, second in (node.args, tuple(reversed(node.args))):
            if self.unbroadcast(first) is start and self.axis(second) == row_axis:
                return node
        return None

    def flat_index(
        self,
        value: object,
        row: Node,
        width: object,
        inner_axis: int,
        dtype: torch.dtype,
    ) -> bool:
        add = self.integer_call(value, torch.ops.aten.add.Tensor, dtype)
        if add is None:
            return False
        for scaled, inner in (add.args, tuple(reversed(add.args))):
            mul = self.integer_call(scaled, torch.ops.aten.mul.Tensor, dtype)
            if mul is None or self.axis(inner) != inner_axis:
                continue
            for source_row, stride in (mul.args, tuple(reversed(mul.args))):
                if self.unbroadcast(source_row) is row and self.scalar_equal(
                    stride, width
                ):
                    return True
        return False

    def child(
        self, info: GraphInfo, *, promoted: bool = False
    ) -> tuple[Node, ForLoopGraphInfo] | None:
        loops = tuple(
            info.graph.find_nodes(op="call_function", target=_tracing_ops._for_loop)
        )
        if len(loops) != 1 or not _call(loops[0], _tracing_ops._for_loop, 4):
            return None
        loop = loops[0]
        graph_id, begins, ends, values = loop.args
        if (
            type(graph_id) is not int
            or not 0 <= graph_id < len(self.ir.graphs)
            or begins != ([] if promoted else [0])
            or not isinstance(ends, (tuple, list))
            or len(ends) != (0 if promoted else 1)
            or not isinstance(values, (tuple, list))
        ):
            return None
        child = self.ir.graphs[graph_id]
        if (
            not isinstance(child, ForLoopGraphInfo)
            or len(child.block_ids) != (0 if promoted else 1)
            or child.node_args != list(values)
            or isinstance(child, _PromotedGridAxisGraphInfo) != promoted
        ):
            return None
        self.matched.add(loop)
        return loop, child


def _argument(host: HostFunction, name: str, tensor: torch.Tensor) -> bool:
    return host.params.arguments.get(name) is tensor


def prove_flat_grouped_rna(
    env: CompileEnvironment, ir: DeviceIR
) -> FlatGroupedIRProof | None:
    """Prove the complete four-region typed program without reading tensor data."""
    host = ir.host_function
    if (
        env.backend_name != "cute"
        or env.device.type != "cuda"
        or host is None
        or env._is_distributed
        or env.cute_fission_plan is not None
        or len(ir.root_ids) != 1
        or len(ir.graphs) != 4
        or len(ir.grid_block_ids) != 1
        or len(ir.grid_block_ids[0]) not in (1, 2)
        or env.config_spec.target_device_capability is None
        or env.config_spec.target_device_capability[0] != 10
    ):
        return None
    root = ir.graphs[ir.root_ids[0]]
    if not isinstance(root, RootGraphInfo) or _output(root) is not None:
        return None
    match = _Matcher(env, ir)
    row_child = match.child(root)
    if row_child is None:
        return None
    row_loop, row_info = row_child
    promoted_column = len(ir.grid_block_ids[0]) == 2
    n_child = match.child(row_info, promoted=promoted_column)
    if n_child is None:
        return None
    n_loop, n_info = n_child
    k_child = match.child(n_info)
    if k_child is None:
        return None
    k_loop, k_info = k_child
    g, m, n, k = (
        env.canonical_block_id(axis)
        for axis in (
            ir.grid_block_ids[0][0],
            row_info.block_ids[0],
            ir.grid_block_ids[0][1] if promoted_column else n_info.block_ids[0],
            k_info.block_ids[0],
        )
    )
    if len({g, m, n, k}) != 4 or env.jagged_tile_parent_ids.get(m) != [g]:
        return None
    if _output(row_info) not in ([], ()) or _output(n_info) not in ([], ()):
        return None
    calls = [
        node
        for info in ir.graphs
        for node in info.graph.nodes
        if node.op == "call_function"
    ]
    contractions = [
        node for node in calls if node.target is torch.ops.aten.baddbmm.default
    ]
    stores = [node for node in calls if node.target is memory_ops.store]
    if len(contractions) != 1 or len(stores) != 1:
        return None
    mm, store = contractions[0], stores[0]
    if (
        not _call(mm, torch.ops.aten.baddbmm.default, 3)
        or mm.graph is not k_info.graph
        or _output(k_info) != [mm]
    ):
        return None
    if (
        not _call(store, memory_ops.store, 4)
        or store.graph is not n_info.graph
        or store.args[3] is not None
        or store.users
    ):
        return None
    match.matched.update((mm, store))
    rhs = match.direct_load(mm.args[2], (g, k, n))
    lhs = match.host_load(mm.args[1])
    if rhs is None or lhs is None:
        return None
    b, b_name = rhs
    a_load, a, a_name, a_indices = lhs
    if (
        a.ndim != 1
        or a.dtype is not torch.float32
        or b.dtype is not torch.float32
        or not isinstance(a_indices, (tuple, list))
        or len(a_indices) != 1
        or not _argument(host, b_name, b)
        or not match.extent_equal(g, b.shape[0])
        or not match.scalar_equal(_loop_stop(k_loop), b.shape[1])
    ):
        return None
    if promoted_column:
        if not match.extent_equal(n, b.shape[2]):
            return None
    elif not match.scalar_equal(_loop_stop(n_loop), b.shape[2]):
        return None
    row_arguments = row_info.node_args
    if len(row_arguments) != 2:
        return None
    extent, starts = row_arguments
    start_load = match.direct_load(starts, (g,))
    if start_load is None or not isinstance(starts, Node):
        return None
    offsets, offsets_name = start_load
    dtype = offsets.dtype
    if dtype not in (torch.int32, torch.int64) or not _argument(
        host, offsets_name, offsets
    ):
        return None
    extent_node = match.integer_call(extent, torch.ops.aten.sub.Tensor, dtype)
    if extent_node is None or match.strip(extent_node.args[1]) is not starts:
        return None
    end_load = match.host_load(extent_node.args[0])
    if end_load is None or end_load[1] is not offsets:
        return None
    end_indices = end_load[3]
    if not isinstance(end_indices, (list, tuple)) or len(end_indices) != 1:
        return None
    next_group = match.integer_call(
        end_indices[0], torch.ops.aten.add.Tensor, torch.int32
    )
    if next_group is None or not any(
        match.axis(axis) == g and type(one) is int and one == 1
        for axis, one in (next_group.args, tuple(reversed(next_group.args)))
    ):
        return None
    # The jagged extent is exactly this source-width difference, reduced only
    # to choose a tile loop bound. Per-group masks remain the registered parent.
    stop = _loop_stop(row_loop)
    if not _call(stop, torch.ops.aten.amax.default, 1):
        return None
    assert isinstance(stop, Node)
    masked = stop.args[0]
    bits = 32 if dtype is torch.int32 else 64
    if not _call(masked, _tracing_ops._mask_to, 2):
        return None
    assert isinstance(masked, Node)
    view = masked.args[0]
    if (
        masked.args[1] != -(1 << (bits - 1))
        or not _call(view, torch.ops.aten.view.default, 2)
        or view.args[0] is not extent
        or view.args[1] != [-1]
    ):
        return None
    match.matched.update((stop, masked, view))
    n_arguments = n_info.node_args
    if len(n_arguments) != 1:
        return None
    row = match.row(n_arguments[0], starts, m, dtype)
    if row is None or not match.flat_index(a_indices[0], row, b.shape[1], k, dtype):
        return None
    k_arguments = k_info.node_args
    if len(k_arguments) != 2 or match.strip(k_arguments[0]) is not row:
        return None
    local_seed = mm.args[0]
    while _call(local_seed, _tracing_ops._new_var, 1):
        match.matched.add(local_seed)
        local_seed = local_seed.args[0]
    k_parameters = tuple(k_info.graph.find_nodes(op="placeholder"))
    if len(k_parameters) != 2 or local_seed is not k_parameters[1]:
        return None
    zero = k_arguments[1]
    if not _call(zero, creation_ops.full, 4):
        return None
    assert isinstance(zero, Node)
    shape, value, zero_dtype, device = zero.args
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 3
        or tuple(match.axis(axis) for axis in shape) != (g, m, n)
        or type(value) not in (int, float)
        or value != 0
        or math.copysign(1, value) < 0
        or zero_dtype is not torch.float32
        or device != env.device
        or match.strip(mm.args[0]) is not zero
    ):
        return None
    match.matched.add(zero)
    phis = tuple(n_info.graph.find_nodes(op="call_function", target=_tracing_ops._phi))
    if (
        len(phis) != 1
        or not _call(phis[0], _tracing_ops._phi, 2)
        or phis[0].args[0] is not zero
    ):
        return None
    phi = phis[0]
    loop_result = phi.args[1]
    if not _call(loop_result, operator.getitem, 2) or loop_result.args != (k_loop, 0):
        return None
    match.matched.update((phi, loop_result))
    value = store.args[2]
    if (
        not _call(value, torch.ops.aten.add.Tensor, 2)
        or (value_tensor := _tensor(value)) is None
        or value_tensor.dtype is not torch.float32
    ):
        return None
    assert isinstance(value, Node)
    if value.args[0] is not phi or not _call(
        value.args[1], torch.ops.aten.unsqueeze.default, 2
    ):
        return None
    bias_expand = value.args[1]
    if bias_expand.args[1] != 1:
        return None
    bias_load = match.direct_load(bias_expand.args[0], (g, n))
    if bias_load is None:
        return None
    bias, bias_name = bias_load
    if bias.dtype is not torch.float32 or not _argument(host, bias_name, bias):
        return None
    match.matched.update((value, bias_expand))
    out_node, out_indices = store.args[:2]
    output = _tensor(out_node)
    if (
        not _call(out_node, _tracing_ops._host_tensor, 1)
        or output is None
        or output.ndim != 1
        or output.dtype is not torch.float32
        or not isinstance(out_indices, (list, tuple))
        or len(out_indices) != 1
        or not match.flat_index(out_indices[0], row, b.shape[2], n, dtype)
        or set(out_node.users) != {store}
    ):
        return None
    assert isinstance(out_node, Node)
    output_name = out_node.args[0]
    if not isinstance(output_name, str) or not output_name.isidentifier():
        return None
    match.matched.add(out_node)
    a_argument = prove_flat_grouped_host(host, a_name, output_name)
    if a_argument is None:
        return None
    original_a = host.params.arguments[a_argument]
    assert isinstance(original_a, torch.Tensor)
    if not env.known_equal(output.numel(), original_a.shape[0] * b.shape[2]):
        return None
    # No extra memory effect, branch, cast, numerical operation or recurrence
    # may disappear merely because the selected contraction matched.
    metadata_calls = {
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        torch.ops.aten.sym_size.int,
    }
    if any(
        node not in match.matched and node.target not in metadata_calls
        for node in calls
    ):
        return None
    if any(
        (tensor := _tensor(node)) is None or tensor.dtype is not torch.float32
        for node in (mm, a_load, mm.args[2], phi, value)
    ):
        return None
    return FlatGroupedIRProof(
        root.graph_id,
        g,
        m,
        n,
        k,
        bits,
        offsets_name,
        a_argument,
        b_name,
        bias_name,
        a_name,
        output_name,
        offsets,
        original_a,
        b,
        bias,
        output,
        mm,
        len(match.matched),
    )
