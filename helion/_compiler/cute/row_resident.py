"""Typed admission for a compact row-resident pair of contractions.

This is a schedule for an already proved materialized-fission program. It
does not recognize Python names or examples. The owning typed host and every
node of both device regions are checked again before replacing their calls.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import TYPE_CHECKING

import sympy
import torch

from ...language import _tracing_ops
from ...language import creation_ops
from ...language import matmul_ops
from ...language import memory_ops
from ..device_ir import ForLoopGraphInfo
from ..device_ir import RootGraphInfo
from ..indexing_strategy import exact_tile_block_ids
from .cute_epilogue import Tcgen05UnaryEpilogueChain
from .cute_epilogue import analyze_tcgen05_unary_epilogue_chain
from .cute_fx_walk import build_inner_outputs_index_from_graphs
from .epilogue_fanout import _fresh_returned_tensors
from .memory_ops import _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
from .memory_ops import tensor_has_specialized_tma_alignment
from .promote_output_axis import _access_tensor

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..host_function import HostFunction


ROW_RESIDENT_KEY = "cute_materialized_schedule"
ROW_RESIDENT_MODES = ("off", "warp_rows2")


@dataclass(frozen=True)
class RowStore:
    name: str
    tensor: torch.Tensor
    chain: Tcgen05UnaryEpilogueChain


@dataclass(frozen=True)
class RowContraction:
    row_axis: int
    column_axis: int
    reduction_axis: int
    lhs_name: str
    rhs_name: str
    lhs: torch.Tensor
    rhs: torch.Tensor
    stores: tuple[RowStore, ...]


@dataclass(frozen=True)
class RowResidentPlan:
    stages: tuple[RowContraction, RowContraction]
    arguments: tuple[str, ...]
    rows: int
    # Physical N/K tile extents equal the complete static tensor axes. Only M
    # may have a tail; no accumulator is rounded at a newly introduced K cut.
    columns: tuple[int, int]
    reductions: tuple[int, int]

    @property
    def shared_bytes(self) -> int:
        cache = 2 * self.columns[0] * 2
        operand_a = 2 * max(self.reductions) * 2
        operand_b = (
            max(n * k for n, k in zip(self.columns, self.reductions, strict=True)) * 2
        )
        # The cache is followed by independently 1024-aligned operand arenas.
        return ((cache + 1023) // 1024 + (operand_a + 1023) // 1024) * 1024 + operand_b


def _call(node: object, target: object, count: int) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is target
        and len(node.args) == count
        and not node.kwargs
    )


def _value(node: object) -> torch.Tensor | None:
    if isinstance(node, torch.fx.Node):
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            return value
    return None


def _literal_int(value: object) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, torch.SymInt):
        expression = value._sympy_()
        if isinstance(expression, sympy.Integer):
            return int(expression)
    return None


def _matrix(tensor: torch.Tensor, dtype: torch.dtype = torch.float16) -> bool:
    if tensor.ndim != 2 or tensor.dtype is not dtype:
        return False
    # Explicit shape specialization can retain SymInt containers whose actual
    # expressions are already literals. No size hint or symbolic expression is
    # accepted here, and the original tensor and its metadata remain unchanged.
    rows, columns = (_literal_int(value) for value in tensor.shape)
    row_stride, column_stride = (_literal_int(value) for value in tensor.stride())
    return (
        rows is not None
        and columns is not None
        and row_stride is not None
        and column_stride is not None
        and rows > 0
        and columns > 0
        and (row_stride, column_stride) == (columns, 1)
        # All compact copy and output coordinates are signed Int32. Include
        # the one padded row even though its pointers are never evaluated.
        and (rows + 1) * columns < 2**31
    )


def _axis_size(env: CompileEnvironment, axis: int, size: int | torch.SymInt) -> bool:
    extent = env.block_sizes[axis].size
    return isinstance(extent, (int, torch.SymInt)) and env.known_equal(extent, size)


def _direct_load(
    env: CompileEnvironment,
    node: object,
    axes: tuple[int, int],
    dtype: torch.dtype = torch.float16,
) -> tuple[str, torch.Tensor] | None:
    if not _call(node, memory_ops.load, 4):
        return None
    assert isinstance(node, torch.fx.Node)
    source, indices, mask, other = node.args
    if (
        mask is not None
        or other is not None
        or not _call(source, _tracing_ops._host_tensor, 1)
        or not isinstance(indices, (tuple, list))
        or exact_tile_block_ids(env, indices) != axes
    ):
        return None
    assert isinstance(source, torch.fx.Node)
    tensor = _value(source)
    loaded = _value(node)
    name = source.args[0]
    if (
        tensor is None
        or not _matrix(tensor, dtype)
        or loaded is None
        or loaded.dtype is not tensor.dtype
        or not isinstance(name, str)
        or not name.isidentifier()
        or any(
            not _axis_size(env, axis, size)
            for axis, size in zip(axes, tensor.shape, strict=True)
        )
    ):
        return None
    return name, tensor


def _stage(
    env: CompileEnvironment,
    host: HostFunction,
    index: int,
    fresh: frozenset[torch.Tensor],
    dtype: torch.dtype = torch.float16,
) -> RowContraction | None:
    ir = host.device_ir
    root = ir.graphs[ir.root_ids[index]]
    axes = ir.grid_block_ids[index]
    if not isinstance(root, RootGraphInfo) or len(axes) != 2:
        return None
    loops = [
        node
        for node in root.graph.nodes
        if _tracing_ops.is_for_loop_target(node.target)
    ]
    if len(loops) != 1 or not _call(loops[0], _tracing_ops._for_loop, 4):
        return None
    loop = loops[0]
    graph_id, starts, stops, carries = loop.args
    if (
        type(graph_id) is not int
        or not 0 <= graph_id < len(ir.graphs)
        or starts != [0]
        or not isinstance(stops, (tuple, list))
        or len(stops) != 1
        or not isinstance(carries, (tuple, list))
        or len(carries) != 1
        or not _call(carries[0], creation_ops.full, 4)
    ):
        return None
    stop = stops[0]
    if _call(stop, _tracing_ops._get_symnode, 1):
        assert isinstance(stop, torch.fx.Node)
        if not isinstance(stop.args[0], str):
            return None
        # This canonical tracing op is emitted from this exact typed value.
        # _axis_size below must also prove it is the complete reduction axis;
        # a coincidental value on any other FX operation is never accepted.
        stop = stop.meta.get("val")
    if type(stop) is not int and not isinstance(stop, torch.SymInt):
        return None
    seed = carries[0]
    assert isinstance(seed, torch.fx.Node)
    shape, zero, seed_dtype, device = seed.args
    if (
        not isinstance(shape, (tuple, list))
        or exact_tile_block_ids(env, shape) != tuple(axes)
        or type(zero) not in (int, float)
        or zero != 0
        or type(zero) is float
        and math.copysign(1.0, zero) < 0
        or seed_dtype is not torch.float32
        or device is not None
    ):
        return None
    child = ir.graphs[graph_id]
    if (
        not isinstance(child, ForLoopGraphInfo)
        or len(child.block_ids) != 1
        or child.node_args != list(carries)
    ):
        return None
    k_axis = child.block_ids[0]
    if k_axis in axes or not _axis_size(env, k_axis, stop):
        return None
    placeholders = list(child.graph.find_nodes(op="placeholder"))
    mmas = [
        node
        for node in child.graph.nodes
        if node.op == "call_function"
        and node.target in (torch.ops.aten.addmm.default, matmul_ops.dot)
    ]
    outputs = list(child.graph.find_nodes(op="output"))
    if len(placeholders) != 1 or len(mmas) != 1 or len(outputs) != 1:
        return None
    mma = mmas[0]
    if outputs[0].args != ([mma],):
        return None
    if _call(mma, torch.ops.aten.addmm.default, 3):
        acc, lhs, rhs = mma.args
    elif _call(mma, matmul_ops.dot, 4) and mma.args[3] is None:
        lhs, rhs, acc, _ = mma.args
    else:
        return None
    if _call(acc, _tracing_ops._new_var, 1):
        assert isinstance(acc, torch.fx.Node)
        acc = acc.args[0]
    value = _value(mma)
    if acc is not placeholders[0] or value is None or value.dtype is not torch.float32:
        return None
    a = _direct_load(env, lhs, (axes[0], k_axis), dtype)
    b = _direct_load(env, rhs, (k_axis, axes[1]), dtype)
    if a is None or b is None:
        return None
    # The loop may contain only this contraction, its two direct loads, and
    # their typed metadata/accumulator plumbing. In particular no other carry,
    # pointwise operand recipe, memory effect or nested loop can be discarded.
    allowed_child = {
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        _tracing_ops._new_var,
        torch.ops.aten.sym_size.int,
    }
    if any(
        node not in (mma, lhs, rhs, outputs[0], placeholders[0])
        and not (node.op == "call_function" and node.target in allowed_child)
        for node in child.graph.nodes
    ):
        return None
    allowed_root = {
        _tracing_ops._get_symnode,
        _tracing_ops._host_tensor,
        _tracing_ops._phi,
        _tracing_ops._new_var,
        _tracing_ops._for_loop,
        creation_ops.full,
        memory_ops.load,
        memory_ops.store,
        operator.getitem,
        torch.ops.aten.sym_size.int,
        torch.ops.prims.convert_element_type.default,
        torch.ops.aten.relu.default,
        torch.ops.aten.sigmoid.default,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.neg.default,
    }
    if any(
        node.op != "output"
        and not (node.op == "call_function" and node.target in allowed_root)
        for node in root.graph.nodes
    ):
        return None
    inner_outputs = build_inner_outputs_index_from_graphs(ir.graphs)
    stores: list[RowStore] = []
    for node in root.graph.nodes:
        if node.target is memory_ops.load:
            read = _direct_load(env, node, (axes[0], axes[1]), dtype)
            if (
                read is None
                or read[0] not in host.params.arguments
                or read[1] is not host.params.arguments[read[0]]
            ):
                return None
        if node.target is not memory_ops.store:
            continue
        if not _call(node, memory_ops.store, 4) or node.args[3] is not None:
            return None
        tensor = _access_tensor(node)
        source, indices, stored, _ = node.args
        if (
            tensor is None
            or tensor not in fresh
            or not _matrix(tensor, dtype)
            or not _call(source, _tracing_ops._host_tensor, 1)
            or not isinstance(indices, (tuple, list))
            or exact_tile_block_ids(env, indices) != tuple(axes)
            or not isinstance(stored, torch.fx.Node)
            or tuple(tensor.shape) != (a[1].shape[0], b[1].shape[1])
        ):
            return None
        analyzed = analyze_tcgen05_unary_epilogue_chain(
            None,
            stored,
            output_global_shape=tuple(tensor.shape),
            target_fx_nodes={mma},
            inner_outputs_by_graph_id=inner_outputs,
        )
        if analyzed is None or analyzed[1] is not mma:
            return None
        chain = analyzed[0]
        for aux in chain.auxiliary_tensor_loads:
            if (
                aux.broadcast_axis is not None
                or _direct_load(env, aux.load_node, (axes[0], axes[1]), dtype) is None
            ):
                return None
        assert isinstance(source, torch.fx.Node) and isinstance(source.args[0], str)
        stores.append(RowStore(source.args[0], tensor, chain))
    if not stores or len({store.name for store in stores}) != len(stores):
        return None
    return RowContraction(
        axes[0], axes[1], k_axis, a[0], b[0], a[1], b[1], tuple(stores)
    )


def prove_row_resident(
    env: CompileEnvironment, host: HostFunction, *, binding: bool = False
) -> RowResidentPlan | None:
    fission = env.cute_fission_plan
    ir = host.device_ir
    if (
        env.backend_name != "cute"
        or fission is None
        or fission.region_count != 2
        or fission.pointwise_region_indices
        or len(ir.root_ids) != 2
        or len(ir.graphs) != 4
        or env._is_distributed
        or env.settings.fast_math
        or env.config_spec.target_device_capability is None
        or env.config_spec.target_device_capability[0] < 8
    ):
        return None
    fresh = _fresh_returned_tensors(host)
    stages = tuple(_stage(env, host, i, fresh) for i in range(2))
    first, second = stages
    if first is None or second is None or len(first.stores) != 1:
        return None
    materialized = first.stores[0]
    if (
        second.lhs is not materialized.tensor
        or second.lhs_name != materialized.name
        or first.lhs.shape[0] != second.lhs.shape[0]
        or first.stores[0].chain.auxiliary_tensor_loads
    ):
        return None
    stores = (*first.stores, *second.stores)
    if len(stores) != len(fresh) or {store.tensor for store in stores} != set(fresh):
        return None
    written = {store.name for store in stores}
    inputs = host.params.arguments
    if any(
        name not in inputs
        or not isinstance(inputs[name], torch.Tensor)
        or tensor is not inputs[name]
        or not binding
        and not tensor_has_specialized_tma_alignment(env, tensor)
        for name, tensor in (
            (first.lhs_name, first.lhs),
            (first.rhs_name, first.rhs),
            (second.rhs_name, second.rhs),
        )
    ):
        return None
    if binding:
        # Kernel.bind installs the already registered input-classifier snapshot
        # after constructing this ConfigSpec. Search discovery may inspect its
        # live constructor arguments; final codegen above *only* accepts the
        # immutable snapshot checked by public dispatch. A late changed/missing
        # fact therefore rejects the candidate instead of trusting this sample.
        for name in (first.lhs_name, first.rhs_name, second.rhs_name):
            value = env.runtime_arg_values_by_name.get(name)
            source = env.tensor_input_source(inputs[name])
            specialization = env.runtime_input_specializations.get(
                _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
            )
            if (
                not isinstance(value, torch.Tensor)
                or value.data_ptr() % 16
                or source is None
                or specialization is None
                or source not in specialization.sources
            ):
                return None
    for stage in (first, second):
        for store in stage.stores:
            for aux in store.chain.auxiliary_tensor_loads:
                read = _direct_load(
                    env, aux.load_node, (stage.row_axis, stage.column_axis)
                )
                assert read is not None
                if (
                    read[0] in written
                    or read[0] not in inputs
                    or read[1] is not inputs[read[0]]
                ):
                    return None
    # Initial compact-LdMatrix recipe: full static N and K, 8 column warps,
    # FP16 inputs/outputs, FP32 accumulators. This is a layout/resource domain,
    # not a shape identity. M may be any positive representable extent.
    columns = (int(first.rhs.shape[1]), int(second.rhs.shape[1]))
    reductions = (int(first.rhs.shape[0]), int(second.rhs.shape[0]))
    if any(n not in (64, 128, 256) for n in columns) or any(
        k not in (32, 64, 128, 256) for k in reductions
    ):
        return None
    arguments = tuple(inputs) + tuple(store.name for store in stores)
    if len(set(arguments)) != len(arguments) or any(
        not isinstance(value, torch.Tensor) for value in inputs.values()
    ):
        return None
    plan = RowResidentPlan(
        (first, second), arguments, int(first.lhs.shape[0]), columns, reductions
    )
    return plan if plan.shared_bytes <= 48 * 1024 else None
