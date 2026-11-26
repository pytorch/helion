"""Prove private FP32 split reductions before replacing their atomic output.

This envelope contains one direct matrix contraction per disjoint K partition.
The destination is a fresh zero tensor and has no readers. An optional FP32
bias addition belongs only to partition zero. Unsupported programs retain the
ordinary atomic lowering, including its existing FP32 output promotion.

The same producer/reducer schedule also proves the plain GEMM store
(``out[tile_m, tile_n] = acc`` after a whole-K loop, optionally through a
bias addition and the output cast): there the kernel has no partition axis,
so ``cute_split_k_partitions`` names the partition count of the schedule.
"""

from __future__ import annotations

import ast
import dataclasses
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
from .analysis.match_common import node_tensor
from .atomic_output_promotions import fresh_half_atomic_outputs
from .memory_ops import _bound_vec_alignment_signature
from .pipeline_smem import Tcgen05PipelineSmemFacts
from .pipeline_smem import max_pipeline_ab_stages
from .split_k_workspace_config import ATOMIC_EPILOGUE_KEY
from .split_k_workspace_config import PARTIALS_ACCUMULATOR
from .split_k_workspace_config import PARTIALS_KEY
from .split_k_workspace_config import PARTIALS_OUTPUT
from .split_k_workspace_config import PARTITION_CHOICES
from .split_k_workspace_config import PARTITIONS_KEY
from .split_k_workspace_config import STAGES_KEY
from .split_k_workspace_config import WORKSPACE_CONFIG_KEYS as WORKSPACE_CONFIG_KEYS
from .split_k_workspace_config import WORKSPACE_KEY as WORKSPACE_KEY
from .state import cute_env_state

if TYPE_CHECKING:
    from torch.fx import Node

    from ...autotuner.block_id_sequence import BlockIdSequence
    from ...autotuner.config_spec import BlockSizeSpec
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from ..host_function import HostFunction


_TORCH_DTYPE_NAMES = {
    torch.float16: "torch.float16",
    torch.bfloat16: "torch.bfloat16",
    torch.float32: "torch.float32",
}
_CUTLASS_DTYPE_NAMES = {
    torch.float16: "cutlass.Float16",
    torch.bfloat16: "cutlass.BFloat16",
    torch.float32: "cutlass.Float32",
}


def torch_dtype_name(dtype: torch.dtype) -> str:
    """``torch.<dtype>`` spelling of a proved split-K output or operand dtype."""
    return _TORCH_DTYPE_NAMES[dtype]


def cutlass_dtype_name(dtype: torch.dtype) -> str:
    """``cutlass.<Dtype>`` spelling of a proved split-K output or operand dtype."""
    return _CUTLASS_DTYPE_NAMES[dtype]


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
    # -1 for the plain store pattern, which has no partition axis.
    partition_block_id: int
    chunk: ConfigValueExpression | int
    # The plain GEMM store pattern: the reducer writes the whole output (the
    # host keeps its ``torch.empty`` allocation) and no atomic is replaced.
    store: bool = False
    # The plain store's output is a kernel argument (``out0`` of a
    # helion.inductor kernel) instead of a fresh allocation: the host has no
    # allocation to keep and its returns stay as written.
    argument_output: bool = False

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
    # The producer reduces into the zero-filled output with packed atomic
    # adds (no fp32 partials, no reducer launch); see
    # ``atomic_epilogue_admissible``.
    atomic_epilogue: bool = False
    # The partitions do not tile K exactly (``partitions * chunk > k``): the
    # producer reads the operands as they are and offsets each partition's K
    # coordinates by ``partition * chunk``, so the TMA loads zero-fill the
    # last partition past K (``plain_store_chunk``).
    k_offset: bool = False
    # The dtype the producer stores the partials in and the reducer loads:
    # fp32, or the 16-bit output dtype (``cute_split_k_partials="output"``,
    # ``partials_admissible``).
    partial_dtype: torch.dtype = torch.float32


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
    tensor = node_tensor(source)
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
    promotions = cute_env_state(env).half_atomic_output_promotions
    if name not in promotions and name not in (
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
            and name in promotions
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
        or node_tensor(result) is None
        or cast("torch.Tensor", node_tensor(result)).dtype is not torch.float32
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
    env_state = cute_env_state(env)
    if (
        env.backend_name != "cute"
        or host is None
        or env._is_distributed
        or env_state.fission_plan is not None
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
        or node_tensor(atomic.args[2]) is None
        or cast("torch.Tensor", node_tensor(atomic.args[2])).dtype is not torch.float32
        or loop.kwargs
        or len(loop.args) != 4
        or type(loop.args[0]) is not int
        or mm.kwargs
        or len(mm.args) != 3
    ):
        return None
    output_node = cast("Node", atomic.args[0])
    output = node_tensor(output_node)
    output_name = output_node.args[0]
    # The destination is either FP32 (a user FP32 output, or a float16 output
    # the atomic lowering promoted to FP32 storage) or a half output the
    # atomic lowering leaves in its requested dtype (bfloat16 always, float16
    # with ``cute_promote_half_atomic_outputs`` off): the reducer casts the
    # FP32 partial sum to it once, where the atomics rounded every partition.
    if (
        output is None
        or output.ndim != 2
        or output.dtype not in (torch.float32, torch.bfloat16, torch.float16)
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
    output_dtype = env_state.half_atomic_output_promotions.get(
        output_name, output.dtype
    )
    expected_return = (
        output_name
        if output_dtype is output.dtype
        else f"{output_name}.to({torch_dtype_name(output_dtype)})"
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


_METADATA_ATTRIBUTES = frozenset(
    (
        "dtype",
        "device",
        "shape",
        "size",
        "stride",
        "ndim",
        "dim",
        "numel",
        "element_size",
    )
)


def _fresh_store_output(host: HostFunction, name: str) -> bool:
    """``name`` is bound once to a fresh ``torch`` factory allocation and is
    only ever stored to by the device loop, read for metadata, or returned."""
    module = ast.Module(body=host.body, type_ignores=[])
    assignments = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(assignments) != 1 or assignments[0] not in host.body:
        return False
    allocation = assignments[0].value
    if not (
        isinstance(allocation, ast.Call)
        and isinstance(allocation.func, ExtendedAST)
        and isinstance(allocation.func._type_info, CallableType)
        and allocation.func._type_info.value
        in (
            torch.empty,
            torch.empty_like,
            torch.zeros,
            torch.zeros_like,
            torch.ones,
            torch.ones_like,
            torch.full,
            torch.full_like,
        )
        and not any(kw.arg in (None, "out") for kw in allocation.keywords)
    ):
        return False
    return _only_stored_by_device_loop(host, name)


def _argument_store_output(host: HostFunction, name: str) -> bool:
    """``name`` is a parameter of the kernel that the body never rebinds, only
    ever stored to by the device loop, read for metadata, or returned: the
    ``out0`` a helion.inductor kernel writes,

        def kernel(in0, in1, out0):
            for tile_y, tile_x in hl.tile([768, 768]):
                acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
                for tile_r in hl.tile(8192):
                    acc = torch.addmm(acc, in0[tile_y, tile_r], in1[tile_r, tile_x])
                out0[tile_y, tile_x] = acc.to(torch.bfloat16)

    The device loop overwrites every element of the argument, as the
    schedule's reducer does after the producer has read every input, so the
    caller sees the same tensor either way."""
    arguments = host.args
    if name not in {
        argument.arg
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
    }:
        return False
    module = ast.Module(body=host.body, type_ignores=[])
    if any(
        isinstance(node, ast.Name)
        and node.id == name
        and not isinstance(node.ctx, ast.Load)
        for node in ast.walk(module)
    ):
        return False
    return _only_stored_by_device_loop(host, name)


def vector_store_alignment(
    env: CompileEnvironment, output: torch.Tensor
) -> bool | None:
    """Whether an argument output's bound base is aligned to the four elements
    the reducer stores at a time (``_reducer_source``: a 64-bit store of four
    bf16 values, 128-bit for fp32); None before the binding records the
    output's alignment (heuristics register ahead of it, so the proof admits
    the output there and declines it when code generation proves the schedule
    again with the bound facts).

    A fresh allocation is always aligned; a caller's tensor need not be:
    ``torch.empty(256 * 256 + 1, dtype=torch.bfloat16)[1:].view(256, 256)`` is
    row-major with its base 2 bytes past an 8-byte boundary, and the reducer's

        cute.copy(store_atom, converted, cute.make_tensor(out.iterator + output_begin, ...))

    does not compile for it (GB300, a 256 x 8192 x 256 product into that
    out0, without this check; the launcher types ``out`` with its actual
    alignment):

        'cute.copy' op D ptr alignment (16 bits) does not meet requirement
        (64 bits) of atom '!cute_nvgpu.atom.universal_copy<bf16, 64 b>'

    so the schedule declines it."""
    signature = _bound_vec_alignment_signature(env, env.tensor_input_source(output))
    if signature is None:
        return None
    width = 4 * output.element_size()
    return signature[0] % width == 0 and env.input_residues_hold_modulo(width)


def _only_stored_by_device_loop(host: HostFunction, name: str) -> bool:
    """Every read of ``name`` in the host body is the device loop's
    ``name[tile_m, tile_n] = ...``, a metadata attribute or a return."""
    module = ast.Module(body=host.body, type_ignores=[])
    parents = {
        id(child): parent
        for parent in ast.walk(module)
        for child in ast.iter_child_nodes(parent)
    }
    device_nodes = {
        id(node)
        for statement in host.body
        if isinstance(statement, ast.For)
        and isinstance(statement, ExtendedAST)
        and statement._loop_type is LoopType.GRID
        for node in ast.walk(statement)
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
            isinstance(parent, ast.Subscript)
            and isinstance(parent.ctx, ast.Store)
            and id(parent) in device_nodes
        ):
            # The device loop's ``out[tile_m, tile_n] = ...`` (the graph proof
            # pins it to the one tcgen05 store); a host-side element write
            # would make the allocation observable before the reducer runs.
            continue
        if isinstance(parent, ast.Attribute) and parent.attr in _METADATA_ATTRIBUTES:
            continue
        return False
    return True


def analyze_split_k_store_reduction(
    env: CompileEnvironment, ir: DeviceIR
) -> SplitKWorkspaceProof | None:
    """The plain GEMM store: ``acc = zeros; for k: acc = addmm(acc, a, b);
    out[tile_m, tile_n] = (acc [+ bias[tile_n]]) [.to(out.dtype)]`` over the
    whole K extent, ``out`` a fresh private allocation returned as is, or a
    kernel argument that only the device loop writes
    (``_argument_store_output``). None under ``split_k_reduction="none"``:
    the schedule splits a K loop the kernel runs whole, where a kernel's own
    atomic split (``analyze_split_k_reduction``) only changes how its
    partials meet."""
    host = ir.host_function
    if (
        env.backend_name != "cute"
        or env.settings.split_k_reduction == "none"
        or host is None
        or env._is_distributed
        or cute_env_state(env).fission_plan is not None
        or len(ir.root_ids) != 1
        or len(ir.grid_block_ids[0]) != 2
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
    stores = [node for node in calls if node.target is memory_ops.store]
    contractions = [
        node for node in calls if node.target is torch.ops.aten.addmm.default
    ]
    loops = [node for node in root.graph.nodes if node.target is _tracing_ops._for_loop]
    if (
        len(stores) != 1
        or len(contractions) != 1
        or len(loops) != 1
        or _output(root) is not None
    ):
        return None
    store, mm, loop = stores[0], contractions[0], loops[0]
    if (
        store.graph is not root.graph
        or store.users
        or store.kwargs
        or len(store.args) != 4
        or store.args[3] is not None
        or not _is_call(store.args[0], _tracing_ops._host_tensor)
        or not isinstance(store.args[1], (tuple, list))
        or len(store.args[1]) != 2
        or loop.kwargs
        or len(loop.args) != 4
        or type(loop.args[0]) is not int
        or mm.kwargs
        or len(mm.args) != 3
    ):
        return None
    output_node = cast("Node", store.args[0])
    output = node_tensor(output_node)
    output_name = output_node.args[0]
    if (
        output is None
        or output.ndim != 2
        or output.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not isinstance(output_name, str)
        or not output_name.isidentifier()
        or set(output_node.users) != {store}
    ):
        return None
    argument_output = _argument_store_output(host, output_name)
    if not argument_output and not _fresh_store_output(host, output_name):
        return None
    if argument_output and vector_store_alignment(env, output) is False:
        return None
    m_axis, n_axis = (_exact_axis(env, index) for index in store.args[1])
    if (
        m_axis is None
        or n_axis is None
        or m_axis == n_axis
        or set(ir.grid_block_ids[0]) != {m_axis, n_axis}
    ):
        return None
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
    # Each operand is K-major or MN-major, its leading stride 16-byte
    # aligned: the producer's batched views keep the operand's strides (an
    # M-major A [m, k] with strides (1, m) is viewed as [P, m, chunk] with
    # strides (chunk * m, 1, m)), and the tcgen05 GEMM takes either major
    # (``OperandMajorMode.MN`` for an M-major A or an N-major B).
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
        or not (
            (a_strides[1] == 1 and a_strides[0] >= a_shape[1])
            or (a_strides[0] == 1 and a_strides[1] >= a_shape[0])
        )
        or not (
            (b_strides[1] == 1 and b_strides[0] >= b_shape[1])
            or (b_strides[0] == 1 and b_strides[1] >= b_shape[0])
        )
        or any(
            stride != 1 and stride * a.element_size() % 16
            for stride in (*a_strides, *b_strides)
        )
    ):
        return None
    m, k = a_shape
    k2, n = b_shape
    # A column-major output is the row-major output of the swapped product
    # (``_swap_output_major``).
    output_column_major = out_strides == (1, m) and out_strides != (n, 1)
    if (
        k != k2
        or out_shape != (m, n)
        or (out_strides != (n, 1) and not output_column_major)
        or env.specialize_expr(env.block_sizes[m_axis].numel) != m
        or env.specialize_expr(env.block_sizes[n_axis].numel) != n
        or env.specialize_expr(env.block_sizes[k_axis].numel) != k
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
    # The loop walks the whole K extent (the schedule partitions it).
    if (
        _static(env, begin) != 0
        or _static(env, end) != k
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
        store,
        mm,
        loop,
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
    # The stored value: the accumulator, through an optional row-vector or
    # matrix bias addition and the optional cast to the output dtype.
    value = _strip_new_vars(store.args[2], matched)
    if (
        _is_call(value, torch.ops.prims.convert_element_type.default)
        and isinstance(value, torch.fx.Node)
        and len(value.args) == 2
        and value.args[1] is output.dtype
    ):
        matched.add(value)
        value = _strip_new_vars(value.args[0], matched)
    elif output.dtype is not torch.float32:
        return None
    bias: SplitKBias | None = None
    if (
        _is_call(value, torch.ops.aten.add.Tensor)
        and isinstance(value, torch.fx.Node)
        and len(value.args) == 2
        and node_tensor(value) is not None
        and cast("torch.Tensor", node_tensor(value)).dtype is torch.float32
        and _strip_new_vars(value.args[0], matched) is loop_value
    ):
        load = _direct_load(value.args[1], env)
        if load is None:
            return None
        tensor, expression, axes = load
        shape, strides = _shape(env, tensor), _strides(env, tensor)
        if (
            tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or not 1 <= tensor.ndim <= 2
            or len(set(axes)) != len(axes)
            or any(axis not in (m_axis, n_axis) for axis in axes)
            or shape is None
            or shape != tuple(m if axis == m_axis else n for axis in axes)
            or strides is None
            or sum(
                (size - 1) * stride for size, stride in zip(shape, strides, strict=True)
            )
            >= 1 << 31
        ):
            return None
        matched.update((value, cast("Node", value.args[1])))
        bias = SplitKBias(
            tensor,
            expression,
            tuple(0 if axis == m_axis else 1 for axis in axes),
            strides,
        )
        value = loop_value
    if value is not loop_value:
        return None
    if {root.graph_id, inner.graph_id} != set(range(len(ir.graphs))):
        return None
    metadata_calls = {
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        torch.ops.aten.sym_size.int,
    }
    if any(node not in matched and node.target not in metadata_calls for node in calls):
        return None
    if any(
        node is not output_node
        and node.target is _tracing_ops._host_tensor
        and node.args == (output_name,)
        for node in calls
    ):
        return None
    returns = [stmt for stmt in host.body if isinstance(stmt, ast.Return)]
    if argument_output:
        # ``return``, ``return None`` or the argument itself.
        if any(
            stmt.value is not None
            and not (isinstance(stmt.value, ast.Constant) and stmt.value.value is None)
            and ast.unparse(stmt.value) != output_name
            for stmt in returns
        ):
            return None
    elif (
        len(returns) != 1
        or returns[0].value is None
        or ast.unparse(returns[0].value) != output_name
    ):
        return None
    proof = SplitKWorkspaceProof(
        root.graph_id,
        a,
        b,
        output,
        a_expression,
        b_expression,
        output_name,
        output.dtype,
        bias,
        m,
        n,
        k,
        cast("tuple[int, int]", a_strides),
        cast("tuple[int, int]", b_strides),
        m_axis,
        n_axis,
        k_axis,
        -1,
        ConfigValueExpression(
            "cdiv", (k, ConfigValueExpression("config", (PARTITIONS_KEY,)))
        ),
        store=True,
        argument_output=argument_output,
    )
    return _swap_output_major(proof) if output_column_major else proof


def _swap_output_major(proof: SplitKWorkspaceProof) -> SplitKWorkspaceProof:
    """The proof of a plain store into a column-major ``[m, n]`` output, as
    the swapped product ``out.T = b.T @ a.T`` into the row-major ``[n, m]``
    view of the same memory, whose partials the reducer sums in that memory's
    order. helion.inductor's transposed product (``lowering/mm.py``) is such
    a kernel: bf16 256 x 512 x 197952 with b stored [K, N] runs

        def kernel(in0, in1, out0):  # in0 = b.T, in1 = a.T, out0 = out.T
            for tile_x, tile_y in hl.tile([512, 256]):
                acc = hl.zeros([tile_x, tile_y], dtype=torch.float32)
                for tile_r in hl.tile(197952):
                    acc = torch.addmm(acc, in0[tile_x, tile_r], in1[tile_r, tile_y])
                out0[tile_x, tile_y] = acc.to(torch.bfloat16)

    with out0 of strides (1, 512). The schedule's producer then multiplies
    the partitions of in1.T = a by in0.T = b into row-major partials of the
    [256, 512] product, which the reducer adds element by element into
    out0's storage (out0[i, j] at i + 512 * j is the partials' element
    [j, i]); 32 partitions of 256x256 tiles (k-offset partitions, the last
    one running past K into the loads' zero fill) render

        _helion_split_k_partials = torch.empty((32, 256, 512), dtype=torch.float32, ...)
        _helion_split_k_a = in1.as_strided((1, 256, 197952), (6208, 197952, 1))
        _helion_split_k_b = in0.as_strided((1, 197952, 512), (3178496, 512, 1))
        _helion_split_k_native._native_partition_gemm(
            _helion_split_k_a, _helion_split_k_b, _helion_split_k_partials, ...)
        _launcher(_helion_split_k_reduce._reduce_partitions, (128,),
                  _helion_split_k_partials, out0, ...)

    Kept as written, the producer would store [P, 512, 256] row-major
    partials whose element order is not out0's, and the reducer's flat
    vector loop (``_reducer_source``) would write them transposed. The
    swapped proof reads its tile from the kernel's block sizes through the
    swapped block ids (the producer's rows are the kernel's columns), and a
    bias indexed by the output's row or column indexes the swapped
    product's column or row."""
    bias = proof.bias
    return dataclasses.replace(
        proof,
        lhs=proof.rhs,
        rhs=proof.lhs,
        lhs_expression=proof.rhs_expression,
        rhs_expression=proof.lhs_expression,
        bias=None
        if bias is None
        else dataclasses.replace(
            bias, dimensions=tuple(1 - dimension for dimension in bias.dimensions)
        ),
        m=proof.n,
        n=proof.m,
        lhs_strides=(proof.rhs_strides[1], proof.rhs_strides[0]),
        rhs_strides=(proof.lhs_strides[1], proof.lhs_strides[0]),
        m_block_id=proof.n_block_id,
        n_block_id=proof.m_block_id,
    )


# Each partition of a plain-store split keeps at least this many whole
# 64-wide K tiles: shorter partitions spend their pipeline on fill and
# drain, and 128^3 .. 512^3 GEMMs (whose small grids pass the SM gate)
# would otherwise take 48 workspace seeds into their initial populations.
PLAIN_STORE_MIN_PARTITION_K_TILES = 8
# A grid of 256x256 CTA pairs at least half the SMs wide splits K only
# where its last wave leaves pairs idle (the waves fill less than this) and
# every partition stays this many 64-wide K tiles deep, so the partials'
# traffic and the reducer launch stay small beside the GEMM. GB300 (76 SM
# pairs), bf16 with b stored [K, N], one CUDA graph over cold inputs, of
# stock Inductor (cuBLAS), plain kernel (full-effort pick) -> split
# (256x256 tiles): 16384x768x50272 (192 tiles: 84% of three waves) 0.988 ->
# 1.048 in three partitions (95% of eight); 4096x2048x50264 (128 tiles:
# 84%) 0.991 -> 1.003 in four (96%); 4096x1536x128104 (96 tiles: 63%)
# 0.968 -> 1.068 in three (95%). Over a K of 160 tiles the 160 tiles of
# bf16 4096x2560x10240 (70% of three waves) keep the plain kernel, whose
# 256x192 pairs fill 98% of three.
PLAIN_STORE_SPLIT_WAVE_FILL = 0.9
PLAIN_STORE_DEEP_PARTITION_K_TILES = 128


def wave_fill(units: int, num_sm: int) -> float:
    """The share of ``ceil(units / pairs)`` waves of ``num_sm // 2`` CTA pairs
    that ``units`` equal work units keep busy: 96 tiles fill 63% of two waves
    of 76 pairs, 288 (three partitions of them) 95% of four."""
    pairs = max(1, num_sm // 2)
    return units / (-(-units // pairs) * pairs)


def plain_store_partition_choices(
    proof: SplitKWorkspaceProof, num_sm: int | None
) -> tuple[int, ...]:
    """``cute_split_k_partitions`` values of a plain store proof: 1 (the plain
    kernel) and every search choice whose partitions keep at least
    ``PLAIN_STORE_MIN_PARTITION_K_TILES`` 64-wide K tiles, for GEMMs whose
    256x256 CTA pairs leave at least half the SMs idle (a fuller grid only
    pays the partial traffic, and the schedule's seeds would displace the
    random half of the initial population), and those keeping
    ``PLAIN_STORE_DEEP_PARTITION_K_TILES`` for a fuller grid whose waves its
    pairs fill less than ``PLAIN_STORE_SPLIT_WAVE_FILL``. Empty without an SM
    count (hosts without a device) or without an admissible split."""
    if not proof.store or not num_sm:
        return ()
    tiles = -(-proof.m // 256) * -(-proof.n // 256)
    min_tiles = PLAIN_STORE_MIN_PARTITION_K_TILES
    if tiles * 2 >= num_sm // 2:
        if wave_fill(tiles, num_sm) >= PLAIN_STORE_SPLIT_WAVE_FILL:
            return ()
        min_tiles = PLAIN_STORE_DEEP_PARTITION_K_TILES
    splits = tuple(
        split
        for split in PARTITION_CHOICES
        if split > 1
        and (chunk := plain_store_chunk(proof.k, split, 64)) is not None
        and chunk // 64 >= min_tiles
    )
    return (1, *splits) if splits else ()


def analyze_split_k_workspace(
    env: CompileEnvironment, ir: DeviceIR
) -> SplitKWorkspaceProof | None:
    """Keep the original workspace domain while sharing the semantic proof."""
    proof = analyze_split_k_reduction(env, ir)
    if proof is None:
        proof = analyze_split_k_store_reduction(env, ir)
        if proof is not None and not plain_store_partition_choices(
            proof, env.config_spec.num_sm
        ):
            return None
    if (
        proof is None
        or proof.m % 128
        or proof.n % 64
        or (proof.k % 64 and not proof.store)
    ):
        return None
    return proof


# Output dtypes with a packed relaxed atomic add the producer's epilogue can
# issue from its staged tile (``red.global.add.noftz.{f16x2,bf16x2}`` and
# ``red.global.add.v2.f32``).
ATOMIC_EPILOGUE_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32))


def atomic_epilogue_admissible(proof: SplitKWorkspaceProof) -> bool:
    """Whether the proved reduction can take ``cute_split_k_atomic_epilogue``:
    an atomic split-K add (the plain store has no zero-filled destination to
    add into) whose output is kept in a dtype with a packed atomic add (a
    promoted half output keeps its fp32 storage and host cast, which the
    reducer form serves)."""
    return (
        not proof.store
        and proof.output.dtype is proof.output_dtype
        and proof.output_dtype in ATOMIC_EPILOGUE_DTYPES
    )


@dataclass(frozen=True)
class AtomicEpilogueGeometry:
    """How the producer's epilogue warps walk one staged epilogue subtile
    (``tile_m`` x ``tile_n`` fp32 values in shared memory) as packed atomic
    adds of ``width`` output elements (8 bytes) per lane: ``chunks_per_row``
    lanes cover a row, a warp instruction covers ``32 // chunks_per_row``
    rows, and the four epilogue warps take ``iterations`` instructions each.

    The subtiles of a CTA's ``rows`` x ``cta_n`` tile are enumerated M
    fastest (``m_subtiles`` along M); a subtile's columns are ``run``-long
    contiguous runs of the CTA tile placed ``run_stride`` apart (CUTLASS's
    ``(tile_n // warp_n, warp_n):(1, cta_n // warp_n)`` epilogue tiler), the
    N subtile index selecting the run offset.
    """

    rows: int
    tile_m: int
    tile_n: int
    m_subtiles: int
    width: int
    chunks_per_row: int
    iterations: int
    run: int
    run_stride: int

    def subtile_expressions(self, subtile: str) -> tuple[str, str]:
        """The (M, N) subtile indices of flat subtile index ``subtile``."""
        if self.m_subtiles == 1:
            return "0", subtile
        return f"{subtile} % {self.m_subtiles}", f"{subtile} // {self.m_subtiles}"

    def row_expression(self, subtile_m: str, row: str) -> str:
        """The CTA tile row of subtile row ``row``."""
        if self.m_subtiles == 1:
            return row
        return f"{subtile_m} * {self.tile_m} + {row}"

    def column_expression(self, subtile_n: str, column: str) -> str:
        """The CTA tile column of subtile ``subtile_n``'s column ``column``."""
        if self.run == self.tile_n:
            return f"{subtile_n} * {self.tile_n} + {column}"
        return (
            f"{column} // {self.run} * {self.run_stride} + {subtile_n} * {self.run}"
            f" + {column} % {self.run}"
        )

    @property
    def lane_column_fixed(self) -> bool:
        """A lane's column within the subtile is the same in every iteration
        (its row advances): the column-indexed bias is loaded once."""
        return 32 % self.chunks_per_row == 0


# Bytes of output one lane reduces per packed atomic instruction (the
# ``red.global.add.v2.f32`` / ``.v2.f16x2`` forms; one 128-byte row of a
# 64-column half tile per 16 lanes).
ATOMIC_EPILOGUE_LANE_BYTES = 8


def atomic_epilogue_geometry(
    *,
    rows: int,
    cta_m: int,
    cta_n: int,
    use_2cta_instrs: bool,
    elem_width_c: int | None,
    d_is_m_major: bool,
    c_is_m_major: bool,
    itemsize: int,
) -> AtomicEpilogueGeometry | None:
    """The epilogue walk of a producer CTA owning ``rows`` rows of the pair
    tile whose staged epilogue subtile is CUTLASS's
    ``compute_epilogue_tile_shape`` of the lowering's own arguments (the
    ``(cta_m, cta_n)`` shape, two-CTA flag and source-C description it
    emitted); ``None`` when the packed width does not tile a run or the
    chunks do not spread evenly over 128 epilogue threads."""
    # Lazy import: ``cutlass`` is only loaded on the cute backend.
    from cutlass.utils.blackwell_helpers import compute_epilogue_tile_size

    tile_m, tile_n = compute_epilogue_tile_size(
        cta_m,
        cta_n,
        use_2cta_instrs,
        32,
        elem_width_c,
        d_is_m_major=d_is_m_major,
        c_is_m_major=c_is_m_major,
    )
    # CUTLASS ``compute_epilogue_tile_shape``: the warps form a (2, 2) grid
    # for a 64-row two-CTA tile and a (4, 1) grid otherwise; the N tiler is
    # ``(tile_n // warp_n, warp_n)`` striding by ``cta_n // warp_n``. The
    # generated kernel re-checks this mapping against the tiler it builds
    # while it is traced (``_split_k_atomic_runtime.check_epilogue_tiler``),
    # so a CUTLASS change fails the compile instead of misaddressing.
    warp_n = 2 if cta_m == 64 and use_2cta_instrs else 1
    run = tile_n // warp_n
    width = ATOMIC_EPILOGUE_LANE_BYTES // itemsize
    if width < 1 or run % width or rows % tile_m or cta_n % tile_n:
        return None
    chunks_per_row = tile_n // width
    chunks = tile_m * chunks_per_row
    if chunks % 128:
        return None
    return AtomicEpilogueGeometry(
        rows,
        tile_m,
        tile_n,
        rows // tile_m,
        width,
        chunks_per_row,
        chunks // 128,
        run,
        cta_n // warp_n,
    )


def partials_admissible(proof: SplitKWorkspaceProof, env: CompileEnvironment) -> bool:
    """Whether a proved plain GEMM store may keep its partials in its 16-bit
    output dtype (``cute_split_k_partials="output"``): torch lets cuBLAS
    reduce bf16 / fp16 split-K partials in the output type
    (``allow_{bf16,fp16}_reduced_precision_reduction``, which helion.inductor
    passes as ``Settings.split_k_reduction="output"``); the sum of the
    partials stays fp32 in the reducer. bf16 256 x 512 x 197952 in 16
    partitions: 8 MiB of fp32 partials, 4 MiB of bf16 ones."""
    return (
        proof.store
        and env.settings.split_k_reduction == "output"
        and proof.output_dtype in (torch.float16, torch.bfloat16)
    )


def workspace_tile_max_stages(
    *,
    m: int,
    n: int,
    k: int,
    chunk: int,
    tile: tuple[int, int, int],
    capacity_bytes: int,
    k_offset: bool = False,
) -> int | None:
    """The deepest AB ring the partition producer fits for a ``(bm, bn, bk)``
    tile of an ``m x n x k`` GEMM cut into ``chunk``-deep partitions, ``None``
    when the producer does not take the tile: a full batch view cannot
    include padding or cross a partition boundary; with ``k_offset`` the
    last partition may run past K (the producer's loads zero-fill it)."""
    bm, bn, bk = tile
    if (
        bm not in (128, 256)
        or bn not in (64, 128, 256)
        or bk not in (64, 128, 256)
        or chunk <= 0
        or (k % chunk if not k_offset else chunk >= k)
        or m % bm
        or n % bn
        or chunk % bk
        or -(-k // chunk) * m * n >= 1 << 31
    ):
        return None
    facts = Tcgen05PipelineSmemFacts(2, 4, capacity_bytes)
    maximum = max_pipeline_ab_stages(
        facts, bm=bm, bn=bn, bk=bk, c_stages=2, acc_stages=2
    )
    return min(maximum, 16) if maximum > 0 else None


def workspace_schedule(
    proof: SplitKWorkspaceProof,
    env: CompileEnvironment,
    config: Config,
    *,
    capacity_bytes: int,
) -> SplitKWorkspaceSchedule | None:
    """The schedule a config asks for, ``None`` when the producer does not
    take its tile or ring (``workspace_tile_max_stages``)."""
    atomic_epilogue = config.get(ATOMIC_EPILOGUE_KEY, False)
    if type(atomic_epilogue) is not bool or (
        atomic_epilogue and not atomic_epilogue_admissible(proof)
    ):
        return None
    chunk = proof.chunk_size(config)
    blocks = tuple(
        env.config_spec.block_sizes.config_get(config.block_sizes, axis)
        for axis in (proof.m_block_id, proof.n_block_id, proof.k_block_id)
    )
    if any(type(block) is not int for block in blocks) or type(chunk) is not int:
        return None
    bm, bn, bk = cast("tuple[int, int, int]", blocks)
    partitions = proof.k // chunk if chunk > 0 else 0
    k_offset = False
    if proof.store:
        partitions = config.get(PARTITIONS_KEY, 1)
        if type(partitions) is not int or partitions < 2:
            return None
        chunk = plain_store_chunk(proof.k, partitions, bk)
        if chunk is None:
            return None
        k_offset = partitions * chunk != proof.k
    stages = config.get(STAGES_KEY, 2)
    maximum = workspace_tile_max_stages(
        m=proof.m,
        n=proof.n,
        k=proof.k,
        chunk=chunk,
        tile=(bm, bn, bk),
        capacity_bytes=capacity_bytes,
        k_offset=k_offset,
    )
    if maximum is None or type(stages) is not int or not 1 <= stages <= maximum:
        return None
    partials = config.get(PARTIALS_KEY, PARTIALS_ACCUMULATOR)
    if partials == PARTIALS_OUTPUT and partials_admissible(proof, env):
        partial_dtype = proof.output_dtype
    elif partials == PARTIALS_ACCUMULATOR:
        partial_dtype = torch.float32
    else:
        return None
    return SplitKWorkspaceSchedule(
        partitions,
        chunk,
        bm,
        bn,
        bk,
        stages,
        atomic_epilogue=atomic_epilogue,
        k_offset=k_offset,
        partial_dtype=partial_dtype,
    )


def plain_store_chunk(k: int, partitions: int, bk: int) -> int | None:
    """The K depth of each of ``partitions`` partitions of a plain store's
    ``k``, in whole ``bk`` tiles: ``k / partitions`` where that is whole
    tiles (the producer's batch views cut K exactly), else that quotient
    rounded up to whole tiles, the last partition running past K into the
    zero fill of the producer's loads. bf16 2048 x 768 x 20008 in three
    partitions of 64-deep tiles: 6720 + 6720 + 6568 (the last 152 columns
    of the third partition zero). ``None`` when the rounding leaves the last
    partition empty."""
    per_partition = -(-k // partitions)
    chunk = -(-per_partition // bk) * bk
    if (partitions - 1) * chunk >= k:
        return None
    return chunk


@dataclass(frozen=True)
class WorkspaceGeometry:
    """What ``workspace_schedule`` reads of a proof, kept on the config spec
    (``cute_split_k_workspace_geometry``) so the search's repair
    (``split_k_workspace_config.repair_workspace_geometry``) can tell a
    sampled or mutated schedule the producer cannot run from one it can."""

    m: int
    n: int
    k: int
    block_ids: tuple[int, int, int]
    chunk: ConfigValueExpression | int
    capacity_bytes: int
    # A plain store proof: its partitions run ``plain_store_chunk`` deep.
    store: bool = False

    @classmethod
    def of(cls, proof: SplitKWorkspaceProof, capacity_bytes: int) -> WorkspaceGeometry:
        return cls(
            proof.m,
            proof.n,
            proof.k,
            (proof.m_block_id, proof.n_block_id, proof.k_block_id),
            proof.chunk,
            capacity_bytes,
            proof.store,
        )

    def max_stages(
        self, block_sizes: BlockIdSequence[BlockSizeSpec], config: dict[str, object]
    ) -> int | None:
        """``workspace_tile_max_stages`` of the tile ``config``'s block sizes
        name (read through the spec's ``block_sizes``) and of its chunk."""
        values = config.get("block_sizes")
        if not isinstance(values, list):
            return None
        tile = tuple(block_sizes.config_get(values, axis) for axis in self.block_ids)
        chunk = (
            self.chunk.evaluate(cast("Config", config))
            if isinstance(self.chunk, ConfigValueExpression)
            else self.chunk
        )
        if type(chunk) is not int or any(type(block) is not int for block in tile):
            return None
        k_offset = False
        if self.store:
            partitions = config.get(PARTITIONS_KEY, 1)
            if type(partitions) is not int or partitions < 2:
                return None
            aligned = plain_store_chunk(self.k, partitions, cast("int", tile[2]))
            if aligned is None:
                return None
            chunk, k_offset = aligned, partitions * aligned != self.k
        return workspace_tile_max_stages(
            m=self.m,
            n=self.n,
            k=self.k,
            chunk=chunk,
            tile=cast("tuple[int, int, int]", tile),
            capacity_bytes=self.capacity_bytes,
            k_offset=k_offset,
        )
