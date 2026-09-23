"""Prove private FP32 split reductions before replacing their atomic output.

This envelope contains one direct matrix contraction per disjoint K partition.
The destination is a fresh zero tensor and has no readers. An optional FP32
bias addition belongs only to partition zero. Unsupported programs retain the
ordinary atomic lowering, including its existing FP32 output promotion.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ...language import _tracing_ops
from ...language import atomic_ops
from ...language import creation_ops
from ...language import memory_ops
from ...language import tile_ops
from ..ast_extension import ExtendedAST
from ..ast_extension import LoopType
from ..compile_environment import ConfigValueExpression
from ..compile_environment import FixedBlockSizeSource
from ..device_ir import ElseGraphInfo
from ..device_ir import ForLoopGraphInfo
from ..device_ir import IfGraphInfo
from ..indexing_strategy import subscript_tile_info
from ..type_info import CallableType
from .atomic_output_promotions import fresh_half_atomic_outputs
from .pipeline_smem import Tcgen05PipelineSmemFacts
from .pipeline_smem import max_pipeline_ab_stages
from .split_k_workspace_config import STAGES_KEY
from .split_k_workspace_config import WORKSPACE_CONFIG_KEYS as WORKSPACE_CONFIG_KEYS
from .split_k_workspace_config import WORKSPACE_KEY as WORKSPACE_KEY

if TYPE_CHECKING:
    from torch.fx import Node

    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from ..host_function import HostFunction


@dataclass(frozen=True)
class SplitKBias:
    tensor: torch.Tensor
    host_expression: str
    # A physical bias coordinate is either the output row (0) or column (1).
    dimensions: tuple[int, ...]
    strides: tuple[int, ...]


@dataclass(frozen=True)
class SplitKWorkspaceProof:
    root_graph_id: int
    lhs: torch.Tensor
    rhs: torch.Tensor
    output: torch.Tensor
    lhs_expression: str
    rhs_expression: str
    output_name: str
    output_dtype: torch.dtype
    bias: SplitKBias | None
    m: int
    n: int
    k: int
    lhs_strides: tuple[int, int]
    rhs_strides: tuple[int, int]
    m_block_id: int
    n_block_id: int
    k_block_id: int
    partition_block_id: int
    chunk: ConfigValueExpression | int

    def chunk_size(self, config: Config) -> int:
        return (
            self.chunk.evaluate(config)
            if isinstance(self.chunk, ConfigValueExpression)
            else self.chunk
        )


@dataclass(frozen=True)
class SplitKWorkspaceSchedule:
    """Logical partitions are independent of persistent launch-grid extents."""

    partitions: int
    chunk: int
    bm: int
    bn: int
    bk: int
    stages: int


def _tensor(node: object) -> torch.Tensor | None:
    if isinstance(node, torch.fx.Node):
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            return value
    return None


def _is_call(node: object, target: object) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is target
        and not node.kwargs
    )


def _static(env: CompileEnvironment, value: object) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, torch.SymInt):
        expression = env.specialize_expr(env.shape_env.replace(value._sympy_()))
        if isinstance(expression, sympy.Integer):
            return int(expression)
    return None


def _shape(env: CompileEnvironment, tensor: torch.Tensor) -> tuple[int, ...] | None:
    sizes = tuple(_static(env, size) for size in tensor.shape)
    return (
        cast("tuple[int, ...]", sizes)
        if all(size is not None and size > 0 for size in sizes)
        else None
    )


def _strides(env: CompileEnvironment, tensor: torch.Tensor) -> tuple[int, ...] | None:
    strides = tuple(_static(env, stride) for stride in tensor.stride())
    return (
        cast("tuple[int, ...]", strides)
        if all(stride is not None and stride >= 0 for stride in strides)
        else None
    )


def _exact_axis(env: CompileEnvironment, index: object) -> int | None:
    tile = subscript_tile_info(env, index)
    if (
        tile is None
        or tile.block_size is not None
        or not env.known_equal(tile.offset, 0)
    ):
        return None
    return tile.block_id


def _direct_load(
    node: object, env: CompileEnvironment
) -> tuple[torch.Tensor, str, tuple[int, ...]] | None:
    if (
        not _is_call(node, memory_ops.load)
        or not isinstance(node, torch.fx.Node)
        or len(node.args) != 4
        or node.args[2:] != (None, None)
        or not _is_call(node.args[0], _tracing_ops._host_tensor)
    ):
        return None
    source = node.args[0]
    assert isinstance(source, torch.fx.Node)
    tensor = _tensor(source)
    indices = node.args[1]
    if (
        tensor is None
        or len(source.args) != 1
        or not isinstance(source.args[0], str)
        or not isinstance(indices, (list, tuple))
        or len(indices) != tensor.ndim
    ):
        return None
    axes = tuple(_exact_axis(env, index) for index in indices)
    if any(axis is None for axis in axes):
        return None
    return tensor, source.args[0], cast("tuple[int, ...]", axes)


def _output(info: GraphInfo) -> object:
    outputs = tuple(info.graph.find_nodes(op="output"))
    return outputs[0].args[0] if len(outputs) == 1 else object()


def _strip_new_vars(node: object, matched: set[Node]) -> object:
    while _is_call(node, _tracing_ops._new_var):
        assert isinstance(node, torch.fx.Node)
        if len(node.args) != 1:
            return None
        matched.add(node)
        node = node.args[0]
    return node


def _getitem(node: object, source: Node, index: int, matched: set[Node]) -> bool:
    if _is_call(node, operator.getitem) and isinstance(node, torch.fx.Node):
        if node.args == (source, index):
            matched.add(node)
            return True
    return False


def _fresh_zero_output(host: HostFunction, name: str, env: CompileEnvironment) -> bool:
    if name not in env.cute_half_atomic_output_promotions and name not in (
        fresh_half_atomic_outputs(host.body, {name: torch.float32})
    ):
        return False
    assignments = [
        node
        for node in ast.walk(ast.Module(body=host.body, type_ignores=[]))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(assignments) != 1 or assignments[0] not in host.body:
        return False
    module = ast.Module(body=host.body, type_ignores=[])
    parents = {
        id(child): parent
        for parent in ast.walk(module)
        for child in ast.iter_child_nodes(parent)
    }
    for node in ast.walk(module):
        if (
            not isinstance(node, ast.Name)
            or node.id != name
            or not isinstance(node.ctx, ast.Load)
        ):
            continue
        parent = parents[id(node)]
        if isinstance(parent, ast.Return):
            continue
        if (
            isinstance(parent, ast.Attribute)
            and parent.attr == "to"
            and isinstance(call := parents[id(parent)], ast.Call)
            and isinstance(parents[id(call)], ast.Return)
            and name in env.cute_half_atomic_output_promotions
        ):
            continue
        if (
            isinstance(parent, ast.Call)
            and parent.args
            and parent.args[0] is node
            and isinstance(parent.func, ExtendedAST)
            and isinstance(parent.func._type_info, CallableType)
            and parent.func._type_info.value is atomic_ops.atomic_add
        ):
            continue
        # The output allocation's dtype/lifetime changes. Metadata consumers,
        # host callbacks, aliases and additional returned views must retain the
        # original path even if the promotion proof allowed a metadata read.
        return False
    allocation = assignments[0].value
    return (
        isinstance(allocation, ast.Call)
        and isinstance(allocation.func, ExtendedAST)
        and isinstance(allocation.func._type_info, CallableType)
        and allocation.func._type_info.value in (torch.zeros, torch.zeros_like)
        and not any(kw.arg in (None, "out") for kw in allocation.keywords)
    )


def _first_partition_bias(
    value: Node,
    loop_value: Node,
    begin: Node,
    root: GraphInfo,
    ir: DeviceIR,
    env: CompileEnvironment,
    m_axis: int,
    n_axis: int,
    m: int,
    n: int,
    matched: set[Node],
    regions: set[int],
) -> tuple[bool, SplitKBias | None]:
    """Match a branch whose only output is the partition-zero accumulator."""
    if value is loop_value:
        return True, None
    if not _is_call(value, _tracing_ops._phi) or len(value.args) != 2:
        return False, None
    false, true = value.args
    if not _is_call(true, operator.getitem) or not isinstance(true, torch.fx.Node):
        return False, None
    branch = true.args[0]
    if (
        not _is_call(branch, _tracing_ops._if)
        or not isinstance(branch, torch.fx.Node)
        or branch.graph is not root.graph
        or len(branch.args) != 5
        or not _getitem(true, branch, 0, matched)
        or not _getitem(false, branch, 1, matched)
    ):
        return False, None
    predicate, then_id, else_id, then_args, else_args = branch.args
    if (
        not _is_call(predicate, operator.eq)
        or not isinstance(predicate, torch.fx.Node)
        or predicate.args not in ((begin, 0), (0, begin))
        or type(then_id) is not int
        or type(else_id) is not int
        or not isinstance(then_args, (tuple, list))
        or not isinstance(else_args, (tuple, list))
        or list(then_args) != [loop_value]
        or list(else_args) != []
    ):
        return False, None
    then = ir.graphs[then_id]
    otherwise = ir.graphs[else_id]
    if (
        not isinstance(then, IfGraphInfo)
        or not isinstance(otherwise, ElseGraphInfo)
        or then.predicate_is_tensor
        or then.else_branch != else_id
        or then.node_args != [loop_value]
        or otherwise.node_args
        or _output(otherwise) not in ([], ())
    ):
        return False, None
    placeholders = tuple(then.graph.find_nodes(op="placeholder"))
    returned = _output(then)
    if (
        len(placeholders) != 1
        or not isinstance(returned, (list, tuple))
        or len(returned) != 1
    ):
        return False, None
    result = _strip_new_vars(returned[0], matched)
    matched.update((value, branch, predicate))
    regions.update((then_id, else_id))
    if result is placeholders[0]:
        return True, None
    if (
        not _is_call(result, torch.ops.aten.add.Tensor)
        or not isinstance(result, torch.fx.Node)
        or len(result.args) != 2
        or _tensor(result) is None
        or cast("torch.Tensor", _tensor(result)).dtype is not torch.float32
    ):
        return False, None
    left, right = result.args
    if _strip_new_vars(left, matched) is not placeholders[0]:
        return False, None
    load = _direct_load(right, env)
    if load is None:
        return False, None
    tensor, expression, axes = load
    shape, strides = _shape(env, tensor), _strides(env, tensor)
    if (
        tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not 1 <= tensor.ndim <= 2
        or len(set(axes)) != len(axes)
        or any(axis not in (m_axis, n_axis) for axis in axes)
        or shape != tuple(m if axis == m_axis else n for axis in axes)
        or strides is None
    ):
        return False, None
    assert shape is not None
    if (
        sum((size - 1) * stride for size, stride in zip(shape, strides, strict=True))
        >= 1 << 31
    ):
        return False, None
    assert isinstance(right, torch.fx.Node)
    matched.update((result, right))
    return True, SplitKBias(
        tensor, expression, tuple(0 if axis == m_axis else 1 for axis in axes), strides
    )


def analyze_split_k_reduction(
    env: CompileEnvironment, ir: DeviceIR
) -> SplitKWorkspaceProof | None:
    """Use typed graph identities, never source names or sample shape hints."""
    host = ir.host_function
    if (
        env.backend_name != "cute"
        or host is None
        or env._is_distributed
        or env.cute_fission_plan is not None
        or len(ir.root_ids) != 1
        or len(ir.grid_block_ids[0]) != 3
        or env.config_spec.target_device_capability is None
        or env.config_spec.target_device_capability[0] != 10
    ):
        return None
    if (
        sum(
            isinstance(statement, ast.For)
            and isinstance(statement, ExtendedAST)
            and statement._loop_type is LoopType.GRID
            for statement in host.body
        )
        != 1
    ):
        return None
    root = ir.graphs[ir.root_ids[0]]
    calls = [
        node
        for info in ir.graphs
        for node in info.graph.nodes
        if node.op == "call_function"
    ]
    atomics = [node for node in calls if node.target is atomic_ops.atomic_add]
    contractions = [
        node for node in calls if node.target is torch.ops.aten.addmm.default
    ]
    loops = [node for node in root.graph.nodes if node.target is _tracing_ops._for_loop]
    if (
        len(atomics) != 1
        or len(contractions) != 1
        or len(loops) != 1
        or _output(root) is not None
    ):
        return None
    atomic, mm, loop = atomics[0], contractions[0], loops[0]
    if (
        atomic.graph is not root.graph
        or atomic.users
        or atomic.kwargs
        or len(atomic.args) != 4
        or atomic.args[3] != "relaxed"
        or not _is_call(atomic.args[0], _tracing_ops._host_tensor)
        or not isinstance(atomic.args[1], (tuple, list))
        or len(atomic.args[1]) != 2
        or _tensor(atomic.args[2]) is None
        or cast("torch.Tensor", _tensor(atomic.args[2])).dtype is not torch.float32
        or loop.kwargs
        or len(loop.args) != 4
        or type(loop.args[0]) is not int
        or mm.kwargs
        or len(mm.args) != 3
    ):
        return None
    output_node = cast("Node", atomic.args[0])
    output = _tensor(output_node)
    output_name = output_node.args[0]
    if (
        output is None
        or output.ndim != 2
        or output.dtype is not torch.float32
        or not isinstance(output_name, str)
        or not output_name.isidentifier()
        or not _fresh_zero_output(host, output_name, env)
        or set(output_node.users) != {atomic}
    ):
        return None
    m_axis, n_axis = (_exact_axis(env, index) for index in atomic.args[1])
    if m_axis is None or n_axis is None or m_axis == n_axis:
        return None
    extra = set(ir.grid_block_ids[0]) - {m_axis, n_axis}
    if len(extra) != 1:
        return None
    (partition_axis,) = extra
    inner = ir.graphs[loop.args[0]]
    if (
        not isinstance(inner, ForLoopGraphInfo)
        or len(inner.block_ids) != 1
        or mm.graph is not inner.graph
    ):
        return None
    (k_axis,) = inner.block_ids
    if k_axis in ir.grid_block_ids[0]:
        return None
    lhs, rhs = _direct_load(mm.args[1], env), _direct_load(mm.args[2], env)
    if lhs is None or rhs is None:
        return None
    a, a_expression, a_axes = lhs
    b, b_expression, b_axes = rhs
    a_shape, b_shape, out_shape = _shape(env, a), _shape(env, b), _shape(env, output)
    a_strides, b_strides, out_strides = (
        _strides(env, a),
        _strides(env, b),
        _strides(env, output),
    )
    if (
        a.ndim != 2
        or b.ndim != 2
        or a.dtype not in (torch.float16, torch.bfloat16)
        or b.dtype is not a.dtype
        or a_axes != (m_axis, k_axis)
        or b_axes != (k_axis, n_axis)
        or a_shape is None
        or b_shape is None
        or out_shape is None
        or a_strides is None
        or b_strides is None
        or a_strides[1] != 1
        or b_strides[1] != 1
        or a_strides[0] < a_shape[1]
        or b_strides[0] < b_shape[1]
        or any(
            stride != 1 and stride * a.element_size() % 16
            for stride in (*a_strides, *b_strides)
        )
    ):
        return None
    m, k = a_shape
    k2, n = b_shape
    if (
        k != k2
        or out_shape != (m, n)
        or out_strides != (n, 1)
        or env.specialize_expr(env.block_sizes[m_axis].numel) != m
        or env.specialize_expr(env.block_sizes[n_axis].numel) != n
        or env.specialize_expr(env.block_sizes[partition_axis].numel) != k
        or max(m, n, k, m * n) >= 1 << 31
    ):
        return None
    start, stop, carried = loop.args[1:]
    if any(
        not isinstance(values, (list, tuple)) or len(values) != 1
        for values in (start, stop, carried)
    ):
        return None
    begin, end, initial = start[0], stop[0], carried[0]
    if (
        not _is_call(begin, tile_ops.tile_begin)
        or not _is_call(end, tile_ops.tile_end)
        or not isinstance(begin, torch.fx.Node)
        or not isinstance(end, torch.fx.Node)
        or len(begin.args) != 1
        or end.args != begin.args
        or _exact_axis(env, begin.args[0]) != partition_axis
        or not _is_call(initial, creation_ops.full)
        or not isinstance(initial, torch.fx.Node)
        or len(initial.args) != 4
        or type(initial.args[1]) not in (int, float)
        or initial.args[1] != 0
        or math.copysign(1, initial.args[1]) != 1
        or initial.args[2:] != (torch.float32, None)
        or inner.node_args != [initial]
    ):
        return None
    if not isinstance(initial.args[0], (list, tuple)) or tuple(
        _exact_axis(env, x) for x in initial.args[0]
    ) != (m_axis, n_axis):
        return None
    placeholders = tuple(inner.graph.find_nodes(op="placeholder"))
    matched: set[Node] = {
        atomic,
        mm,
        loop,
        begin,
        end,
        initial,
        cast("Node", mm.args[1]),
        cast("Node", mm.args[2]),
    }
    if (
        len(placeholders) != 1
        or _strip_new_vars(mm.args[0], matched) is not placeholders[0]
        or _output(inner) != [mm]
    ):
        return None
    loop_values = [
        node
        for node in root.graph.nodes
        if _is_call(node, _tracing_ops._phi)
        and len(node.args) == 2
        and node.args[0] is initial
        and _getitem(node.args[1], loop, 0, matched)
    ]
    if len(loop_values) != 1:
        return None
    loop_value = loop_values[0]
    matched.add(loop_value)
    regions = {root.graph_id, inner.graph_id}
    valid, bias = _first_partition_bias(
        cast("Node", atomic.args[2]),
        loop_value,
        begin,
        root,
        ir,
        env,
        m_axis,
        n_axis,
        m,
        n,
        matched,
        regions,
    )
    if not valid or regions != set(range(len(ir.graphs))):
        return None
    metadata_calls = {
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        torch.ops.aten.sym_size.int,
    }
    if any(node not in matched and node.target not in metadata_calls for node in calls):
        return None
    # A second host binding or device read of the destination is not allowed,
    # even if it escaped FakeTensor alias identity during type propagation.
    if any(
        node is not output_node
        and node.target is _tracing_ops._host_tensor
        and node.args == (output_name,)
        for node in calls
    ):
        return None
    source = env.block_sizes[partition_axis].block_size_source
    if not isinstance(source, FixedBlockSizeSource):
        return None
    chunk: ConfigValueExpression | int | None = _static(env, source.value)
    if chunk is None and isinstance(source.value, torch.SymInt):
        expression = source.value._sympy_()
        if isinstance(expression, sympy.Expr):
            chunk = env.config_value_expressions.get(expression)
    if chunk is None:
        return None
    returns = [stmt for stmt in host.body if isinstance(stmt, ast.Return)]
    output_dtype = env.cute_half_atomic_output_promotions.get(
        output_name, torch.float32
    )
    expected_return = (
        output_name
        if output_dtype is torch.float32
        else f"{output_name}.to(torch.float16)"
    )
    if (
        len(returns) != 1
        or returns[0].value is None
        or ast.unparse(returns[0].value) != expected_return
    ):
        return None
    return SplitKWorkspaceProof(
        root.graph_id,
        a,
        b,
        output,
        a_expression,
        b_expression,
        output_name,
        output_dtype,
        bias,
        m,
        n,
        k,
        cast("tuple[int, int]", a_strides),
        cast("tuple[int, int]", b_strides),
        m_axis,
        n_axis,
        k_axis,
        partition_axis,
        chunk,
    )


def analyze_split_k_workspace(
    env: CompileEnvironment, ir: DeviceIR
) -> SplitKWorkspaceProof | None:
    """Keep the original workspace domain while sharing the semantic proof."""
    proof = analyze_split_k_reduction(env, ir)
    if proof is None or proof.m % 128 or proof.n % 64 or proof.k % 64:
        return None
    return proof


def workspace_schedule(
    proof: SplitKWorkspaceProof,
    env: CompileEnvironment,
    config: Config,
    *,
    capacity_bytes: int,
) -> SplitKWorkspaceSchedule | None:
    """A full batch view cannot include padding or cross a partition boundary."""
    chunk = proof.chunk_size(config)
    blocks = tuple(
        env.config_spec.block_sizes.config_get(config.block_sizes, axis)
        for axis in (proof.m_block_id, proof.n_block_id, proof.k_block_id)
    )
    if any(type(block) is not int for block in blocks) or type(chunk) is not int:
        return None
    bm, bn, bk = cast("tuple[int, int, int]", blocks)
    stages = config.get(STAGES_KEY, 2)
    if (
        bm not in (128, 256)
        or bn not in (64, 128, 256)
        or bk not in (64, 128, 256)
        or chunk <= 0
        or proof.k % chunk
        or proof.m % bm
        or proof.n % bn
        or chunk % bk
        or type(stages) is not int
        or not 1 <= stages <= 16
    ):
        return None
    partitions = proof.k // chunk
    if partitions * proof.m * proof.n >= 1 << 31:
        return None
    facts = Tcgen05PipelineSmemFacts(2, 4, capacity_bytes)
    maximum = max_pipeline_ab_stages(
        facts, bm=bm, bn=bn, bk=bk, c_stages=2, acc_stages=2
    )
    if stages > maximum:
        return None
    return SplitKWorkspaceSchedule(partitions, chunk, bm, bn, bk, stages)
