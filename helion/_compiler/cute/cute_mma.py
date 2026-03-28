"""CuTe MMA (tensor core) codegen for matmul operations.

Generates cute.gemm calls using MmaUniversalOp for warp-level MMA.
Follows the reduction strategy pattern: initialization in outer_prefix,
per-K-tile MMA in the loop body, fragment→scalar conversion in outer_suffix.

The MMA always accumulates in float32 for precision.  Input data (float16
or bfloat16) is cast to float32 during the register load.  After the
K-loop the fragment is written to shared memory via partition_C and each
thread reads back its own scalar element, re-entering the normal
scalar-per-thread model so epilogue ops (bias, activation, cast) work.

Features:
- Works through both aten lowering (addmm/mm) and hl.dot API paths
- Shared memory staging for A and B operands with sync_threads
- Multi-warp tiling via atom_layout_mnk for larger tile sizes
- Masking for non-divisible tile boundaries
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import operator
import os
from typing import TYPE_CHECKING

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.node import Node

from ... import exc
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..dtype_utils import cast_ast
from ..matmul_utils import _needs_f32_accumulator
from ..tile_strategy import DeviceLoopState
from .mma_support import get_cute_mma_support

if TYPE_CHECKING:
    from ..aten_lowering import LoweringContext
    from ..compile_environment import CompileEnvironment
    from ..device_function import DeviceFunction
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import CodegenState


_TRACE_THROUGH_TARGETS = {
    torch.ops.prims.convert_element_type.default,
    # NOTE: permute is NOT included because the MMA pipeline reads
    # raw tensor data — tracing through permute would bypass the
    # data shuffle.  Permuted operands fall back to scalar codegen.
}


@dataclass(frozen=True)
class _Tcgen05LayoutPlan:
    acc_stage_count: str
    ab_stage_count: str
    exec_thread_count: str
    consumer_thread_count: str
    exec_active: str
    smem_a_layout: str
    smem_b_layout: str
    smem_desc_view_layout: str
    c_layout: str
    smem_c_layout: str
    epi_tile: str
    tmem_load_atom: str
    acc_tmem_cols: str
    tmem_holding_buf: str
    tmem_alloc_barrier: str
    tmem_allocator: str
    acc_tmem_ptr: str
    acc_pipeline_barriers: str
    acc_pipeline_producer_group: str
    acc_pipeline_consumer_group: str
    acc_pipeline: str
    acc_producer_state: str
    acc_consumer_state: str
    epilogue_rest_mode: str


def _iter_node_inputs(arg: object) -> list[Node]:
    nodes: list[Node] = []
    if isinstance(arg, Node):
        nodes.append(arg)
    elif isinstance(arg, (list, tuple)):
        for item in arg:
            nodes.extend(_iter_node_inputs(item))
    elif isinstance(arg, dict):
        for item in arg.values():
            nodes.extend(_iter_node_inputs(item))
    return nodes


def _collect_node_dependencies(node: Node) -> set[Node]:
    required: set[Node] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current in required:
            continue
        required.add(current)
        for arg in current.args:
            stack.extend(_iter_node_inputs(arg))
        for arg in current.kwargs.values():
            stack.extend(_iter_node_inputs(arg))
    return required


def _mma_loop_is_exclusive(node: Node) -> bool:
    """Require the loop body to contain only the candidate MMA dataflow."""
    required = _collect_node_dependencies(node)
    for graph_node in node.graph.nodes:
        if graph_node in required or graph_node.op in {
            "placeholder",
            "output",
            "get_attr",
        }:
            continue
        if graph_node.op == "call_function":
            return False
    return True


def _trace_to_load_tensor(node: Node) -> tuple[str, torch.Tensor] | None:
    """Trace through casts/permutes to find the underlying load tensor.

    Only traces through data-preserving ops (type casts, permute).
    Does NOT trace through arithmetic (add, mul, etc.) because the MMA
    pipeline reads raw tensor data and those ops would be skipped.
    """
    from ...language import memory_ops

    cur = node
    while cur.op == "call_function" and cur.target is not memory_ops.load:
        if cur.target not in _TRACE_THROUGH_TARGETS:
            return None
        input_nodes = [a for a in cur.args if isinstance(a, Node)]
        if len(input_nodes) != 1:
            return None
        cur = input_nodes[0]

    if cur.op != "call_function" or cur.target is not memory_ops.load:
        return None
    tensor_node = cur.args[0]
    if not isinstance(tensor_node, Node):
        return None
    fake = tensor_node.meta.get("val")
    if not isinstance(fake, torch.Tensor):
        return None
    return tensor_node.name, fake


def _has_mma_operands(lhs_node: Node, rhs_node: Node) -> bool:
    """Check if lhs/rhs come from loads with MMA-compatible dtypes."""
    lhs_info = _trace_to_load_tensor(lhs_node)
    rhs_info = _trace_to_load_tensor(rhs_node)
    if lhs_info is None or rhs_info is None:
        return False
    _, lhs_fake = lhs_info
    _, rhs_fake = rhs_info
    supported = {torch.float16, torch.bfloat16, torch.float32}
    return (
        lhs_fake.dtype in supported
        and rhs_fake.dtype in supported
        and lhs_fake.dtype == rhs_fake.dtype
        and lhs_fake.ndim == 2
        and rhs_fake.ndim == 2
    )


def is_mma_compatible_aten(node: Node, with_acc: bool) -> bool:
    """Check if an aten addmm/mm node can use MMA."""
    args = node.args
    if with_acc:
        if len(args) < 3:
            return False
        acc_node = args[0]
        lhs_node, rhs_node = args[1], args[2]
        if isinstance(acc_node, Node):
            acc_val = acc_node.meta.get("val")
            if isinstance(acc_val, torch.Tensor) and acc_val.ndim != 2:
                return False
    else:
        if len(args) < 2:
            return False
        lhs_node, rhs_node = args[0], args[1]
    if not isinstance(lhs_node, Node) or not isinstance(rhs_node, Node):
        return False
    return _has_mma_operands(lhs_node, rhs_node)


def is_mma_compatible_dot(node: Node) -> bool:
    """Check if an hl.dot FX node can use MMA."""
    # dot args: (lhs, rhs, acc_or_None, out_dtype_or_None)
    if len(node.args) < 2:
        return False
    acc_node = node.args[2] if len(node.args) > 2 else None
    lhs_node, rhs_node = node.args[0], node.args[1]
    if not isinstance(lhs_node, Node) or not isinstance(rhs_node, Node):
        return False
    if isinstance(acc_node, Node):
        acc_val = acc_node.meta.get("val")
        if isinstance(acc_val, torch.Tensor) and acc_val.ndim != 2:
            return False
    return _has_mma_operands(lhs_node, rhs_node)


def can_codegen_cute_mma_dot(node: Node) -> bool:
    """Return True when hl.dot both supports MMA and matches MMA dtype semantics."""
    if not is_mma_compatible_dot(node):
        return False
    if not _mma_result_can_be_deferred(node) or not _mma_loop_is_exclusive(node):
        return False

    lhs_node = node.args[0]
    rhs_node = node.args[1]
    assert isinstance(lhs_node, Node) and isinstance(rhs_node, Node)

    lhs_val = lhs_node.meta.get("val")
    rhs_val = rhs_node.meta.get("val")
    if not isinstance(lhs_val, torch.Tensor) or not isinstance(rhs_val, torch.Tensor):
        return False

    if not _needs_f32_accumulator(lhs_val.dtype, rhs_val.dtype):
        return True

    acc_dtype: torch.dtype | None = None
    if len(node.args) > 2 and isinstance(node.args[2], Node):
        acc_val = node.args[2].meta.get("val")
        if isinstance(acc_val, torch.Tensor):
            acc_dtype = acc_val.dtype

    out_dtype = node.args[3] if len(node.args) > 3 else None
    if out_dtype is not None and not isinstance(out_dtype, torch.dtype):
        return False

    return out_dtype in (None, torch.float32) and acc_dtype in (
        None,
        torch.float32,
    )


def can_codegen_cute_mma_aten(node: Node, with_acc: bool) -> bool:
    return (
        is_mma_compatible_aten(node, with_acc)
        and _mma_result_can_be_deferred(node)
        and _mma_loop_is_exclusive(node)
    )


def _graph_signature(graph: torch.fx.Graph) -> tuple[tuple[str, str], ...]:
    signature: list[tuple[str, str]] = []
    for node in graph.nodes:
        target = node.op
        if node.op == "call_function":
            target = getattr(node.target, "__name__", str(node.target))
        signature.append((node.op, target))
    return tuple(signature)


def _graph_tensor_output_count(graph: torch.fx.Graph) -> int:
    output_nodes = list(graph.find_nodes(op="output"))
    if not output_nodes:
        return 0
    (output_node,) = output_nodes
    outputs: set[Node] = set()
    for node in _iter_node_inputs(output_node.args):
        value = node.meta.get("val")
        if isinstance(value, torch.Tensor):
            outputs.add(node)
    return len(outputs)


def _trace_acc_init_node(node: Node) -> Node | None:
    from ...language import _tracing_ops
    from ..device_ir import NodeArgsGraphInfo
    from ..host_function import HostFunction

    current = node
    seen: set[Node] = set()
    while current not in seen:
        seen.add(current)
        if current.op == "placeholder":
            current_placeholders = list(current.graph.find_nodes(op="placeholder"))
            current_signature = _graph_signature(current.graph)
            for graph_info in HostFunction.current().device_ir.graphs:
                if current.graph is graph_info.graph and isinstance(
                    graph_info, NodeArgsGraphInfo
                ):
                    if _graph_tensor_output_count(current.graph) > 1:
                        return current
                    current = graph_info.placeholder_to_outer_arg(current)
                    break
                if not isinstance(graph_info, NodeArgsGraphInfo):
                    continue
                if _graph_signature(graph_info.graph) != current_signature:
                    continue
                if _graph_tensor_output_count(graph_info.graph) > 1:
                    return current
                for placeholder, outer_node in zip(
                    current_placeholders,
                    graph_info.node_args,
                    strict=True,
                ):
                    if placeholder is current:
                        current = outer_node
                        break
                else:
                    continue
                break
            else:
                return current
            continue
        if current.op != "call_function":
            return current
        if current.target is _tracing_ops._new_var:
            (arg,) = current.args
            if not isinstance(arg, Node):
                return None
            current = arg
            continue
        if current.target is _tracing_ops._phi:
            lhs = current.args[0]
            if not isinstance(lhs, Node):
                return None
            current = lhs
            continue
        return current
    return None


def _is_zero_init_acc_node(node: Node) -> bool:
    from ...language import creation_ops

    init_node = _trace_acc_init_node(node)
    if init_node is None or init_node.op != "call_function":
        return False
    if init_node.target is creation_ops.full:
        value = init_node.args[1]
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value == 0
        )
    return False


def _local_mma_coord_expr(
    cg: GenerateAST,
    block_id: int,
) -> str:
    """Return the current block-local coordinate for an MMA output axis."""
    block_thread_axes: dict[int, int] = {}
    if cg.current_grid_state is not None:
        block_thread_axes = cg.current_grid_state.block_thread_axes
    thread_axis = block_thread_axes.get(block_id)
    if thread_axis is None:
        return "cutlass.Int32(0)"

    coord = f"cutlass.Int32(cute.arch.thread_idx()[{thread_axis}])"
    if cg.current_grid_state is None:
        return coord

    strategy = cg.current_grid_state.strategy
    lane_vars = getattr(strategy, "_lane_var_by_block", None)
    if not isinstance(lane_vars, dict) or block_id not in lane_vars:
        return coord

    elements_per_thread_fn = getattr(strategy, "_elements_per_thread_for_block", None)
    if not callable(elements_per_thread_fn):
        return coord
    elements_per_thread = elements_per_thread_fn(block_id)
    lane_var = lane_vars[block_id]
    if elements_per_thread == 1:
        return f"{coord} + cutlass.Int32({lane_var})"
    return f"{coord} * cutlass.Int32({elements_per_thread}) + cutlass.Int32({lane_var})"


def _get_mma_k_loop_info(
    cg: GenerateAST,
    env: CompileEnvironment,
    lhs_fake: torch.Tensor,
    rhs_fake: torch.Tensor,
    fx_node: Node | None = None,
) -> tuple[DeviceLoopState, int, str, int] | None:
    """Return the active reduction loop for the operands' shared K dimension."""
    if fx_node is not None:
        from ..device_ir import ForLoopGraphInfo

        graph_k_block_ids = [
            graph_info.block_ids
            for graph_info in cg.codegen_graphs
            if isinstance(graph_info, ForLoopGraphInfo)
            and graph_info.graph is fx_node.graph
        ]
        if len(graph_k_block_ids) == 1:
            active_graph_block_ids = [
                block_id
                for block_id in graph_k_block_ids[0]
                if any(
                    isinstance(loop_state, DeviceLoopState)
                    for loop_state in cg.active_device_loops.get(block_id, ())
                )
            ]
            if len(active_graph_block_ids) == 1:
                k_block_id = active_graph_block_ids[0]
                loops = cg.active_device_loops.get(k_block_id)
                assert loops is not None
                device_loop = next(
                    (
                        loop_state
                        for loop_state in reversed(loops)
                        if isinstance(loop_state, DeviceLoopState)
                    ),
                    None,
                )
                if device_loop is not None:
                    block_size = env.block_sizes[k_block_id].from_config(
                        cg.device_function.config
                    )
                    if isinstance(block_size, int):
                        return (
                            device_loop,
                            k_block_id,
                            device_loop.strategy.offset_var(k_block_id),
                            block_size,
                        )

    lhs_k_block_id = env.resolve_block_id(lhs_fake.shape[1])
    rhs_k_block_id = env.resolve_block_id(rhs_fake.shape[0])
    candidate_block_ids: set[int] = set()
    if (
        lhs_k_block_id is not None
        and rhs_k_block_id is not None
        and lhs_k_block_id == rhs_k_block_id
    ):
        candidate_block_ids.add(lhs_k_block_id)
    else:
        for block_id, loops in cg.active_device_loops.items():
            if not any(isinstance(loop_state, DeviceLoopState) for loop_state in loops):
                continue
            size = env.block_sizes[block_id].size
            if not isinstance(size, int | torch.SymInt):
                continue
            if env.known_equal(size, lhs_fake.shape[1]) and env.known_equal(
                size, rhs_fake.shape[0]
            ):
                candidate_block_ids.add(block_id)

    if len(candidate_block_ids) != 1:
        return None

    (k_block_id,) = tuple(candidate_block_ids)
    loops = cg.active_device_loops.get(k_block_id)
    assert loops is not None

    device_loop = next(
        (
            loop_state
            for loop_state in reversed(loops)
            if isinstance(loop_state, DeviceLoopState)
        ),
        None,
    )
    if device_loop is None:
        return None

    block_size = env.block_sizes[k_block_id].from_config(cg.device_function.config)
    if not isinstance(block_size, int):
        return None

    return (
        device_loop,
        k_block_id,
        device_loop.strategy.offset_var(k_block_id),
        block_size,
    )


def _device_loop_begin_expr(device_loop: DeviceLoopState) -> str:
    loop_iter = device_loop.for_node.iter
    if not isinstance(loop_iter, ast.Call) or not loop_iter.args:
        return "cutlass.Int32(0)"
    if len(loop_iter.args) == 1:
        return "cutlass.Int32(0)"
    return ast.unparse(loop_iter.args[0])


def _has_active_lane_loops(cg: GenerateAST) -> bool:
    grid_state = cg.current_grid_state
    if grid_state is not None and grid_state.has_lane_loops():
        return True
    seen: set[int] = set()
    for loops in cg.active_device_loops.values():
        for loop_state in loops:
            key = id(loop_state)
            if key in seen:
                continue
            seen.add(key)
            strategy = getattr(loop_state, "strategy", None)
            lane_vars = getattr(strategy, "_lane_var_by_block", None)
            if lane_vars:
                return True
    return False


def _mma_result_can_be_deferred(node: Node) -> bool:
    """Return True when the node value is only consumed after the K loop finishes."""
    return all(user.op == "output" for user in node.users)


def _emit_mma_pipeline(
    cg: GenerateAST,
    lhs_node: Node,
    rhs_node: Node,
    acc_expr: ast.AST | None = None,
    fx_node: Node | None = None,
) -> ast.AST | None:
    """Core MMA codegen shared by both aten and hl.dot paths.

    Emits outer_prefix (MMA setup + acc init), loop body (smem staging +
    gemm), and outer_suffix (fragment → per-thread scalar via smem).

    Returns a per-thread scalar expression, or None on failure.
    """
    from ..compile_environment import CompileEnvironment

    lhs_info = _trace_to_load_tensor(lhs_node)
    rhs_info = _trace_to_load_tensor(rhs_node)
    if lhs_info is None or rhs_info is None:
        return None
    _, lhs_fake = lhs_info
    _, rhs_fake = rhs_info
    if lhs_fake.ndim != 2 or rhs_fake.ndim != 2:
        return None

    df = cg.device_function
    lhs_arg_name = df.tensor_arg(lhs_fake).name
    rhs_arg_name = df.tensor_arg(rhs_fake).name

    input_dtype = lhs_fake.dtype
    _dtype_map = {
        torch.float16: "cutlass.Float16",
        torch.bfloat16: "cutlass.BFloat16",
        torch.float32: "cutlass.Float32",
    }
    input_dtype_str = _dtype_map[input_dtype]
    acc_dtype_str = "cutlass.Float32"

    k_total_size = int(lhs_fake.shape[1])

    env = CompileEnvironment.current()

    k_loop_info = _get_mma_k_loop_info(cg, env, lhs_fake, rhs_fake, fx_node=fx_node)
    if k_loop_info is None:
        return None
    device_loop, _, k_offset_var, bk = k_loop_info
    k_loop_begin_expr = _device_loop_begin_expr(device_loop)

    # Get M, N offsets and block sizes from grid state
    m_offset_var: str | None = None
    n_offset_var: str | None = None
    m_block_id: int | None = None
    n_block_id: int | None = None
    bm: int | None = None
    bn: int | None = None
    grid_state = cg.current_grid_state
    if grid_state is not None:
        for bid in grid_state.block_ids:
            offset = grid_state.strategy.offset_var(bid)
            bs_info = env.block_sizes[bid]
            size = bs_info.size
            bs = bs_info.from_config(df.config)
            if isinstance(size, (int, torch.SymInt)):
                if m_offset_var is None and env.known_equal(size, lhs_fake.shape[0]):
                    m_offset_var = offset
                    m_block_id = bid
                    bm = int(bs) if isinstance(bs, int) else None
                elif n_offset_var is None and env.known_equal(size, rhs_fake.shape[1]):
                    n_offset_var = offset
                    n_block_id = bid
                    bn = int(bs) if isinstance(bs, int) else None

    if (
        bm is None
        or bn is None
        or m_offset_var is None
        or n_offset_var is None
        or m_block_id is None
        or n_block_id is None
    ):
        return None

    m_index_var = cg.index_var(m_block_id)
    n_index_var = cg.index_var(n_block_id)
    # Use thread_idx directly for local indices within the tile.
    # indices_0 - offset_0 SHOULD equal thread_idx[0], but the CuTe DSL
    # compiler may not simplify the subtraction, leading to illegal memory
    # accesses when partition shapes depend on dynamic values.
    assert grid_state is not None
    m_local = _local_mma_coord_expr(cg, m_block_id)
    n_local = _local_mma_coord_expr(cg, n_block_id)
    m_global = f"cutlass.Int32({m_index_var})"
    n_global = f"cutlass.Int32({n_index_var})"

    mma_impl = _choose_mma_impl(input_dtype, bm=bm, bn=bn, bk=bk)
    zero_acc_expr = acc_expr is not None and _is_zero_acc_expr(acc_expr)
    if acc_expr is not None and mma_impl != "universal" and not zero_acc_expr:
        mma_impl = "universal"
    if mma_impl != "universal" and zero_acc_expr:
        acc_expr = None

    # Variable names
    tiled_mma = df.new_var("tiled_mma")
    thr_mma = df.new_var("thr_mma")
    acc_frag = df.new_var("acc_frag")
    acc_frag_base = df.new_var("acc_frag_base")
    tcgen05_plan = _new_tcgen05_layout_plan(df) if mma_impl == "tcgen05" else None

    # === outer_prefix: MMA setup + shared memory alloc + accumulator init ===
    prefix = device_loop.outer_prefix

    mma_thread_linear: str | None = None
    mma_active: str | None = None
    mma_phys_n = _mma_active_n_threads(mma_impl)
    if mma_impl == "universal":
        prefix.extend(
            _make_tiled_mma_setup(
                mma_impl,
                tiled_mma,
                thr_mma,
                f"{m_local} + ({n_local}) * cutlass.Int32({bm})",
                input_dtype_str,
                acc_dtype_str,
                bm,
                bn,
            )
        )
    else:
        mma_thread_linear = df.new_var("mma_tidx")
        mma_active = df.new_var("mma_active")
        prefix.append(
            statement_from_string(
                f"{mma_thread_linear} = {m_local} + ({n_local}) * cutlass.Int32({bm})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{mma_active} = ({n_local}) < cutlass.Int32({mma_phys_n})"
            )
        )
        prefix.extend(
            _make_tiled_mma_setup(
                mma_impl,
                tiled_mma,
                thr_mma,
                mma_thread_linear,
                input_dtype_str,
                acc_dtype_str,
                bm,
                bn,
            )
        )
    if mma_impl == "tcgen05":
        assert tcgen05_plan is not None
        prefix.extend(
            _make_tcgen05_layout_plan_setup(
                tcgen05_plan,
                tiled_mma,
                bm=bm,
                bn=bn,
                bk=bk,
                input_dtype_str=input_dtype_str,
                acc_dtype_str=acc_dtype_str,
            )
        )
        prefix.append(
            statement_from_string(
                f"{acc_frag_base} = {tiled_mma}.make_fragment_C("
                f"cute.append({tiled_mma}.partition_shape_C(({bm}, {bn})), "
                "1))"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.exec_active} = "
                f"{mma_thread_linear} < {tcgen05_plan.exec_thread_count}"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_tmem_cols} = cutlass.utils.get_num_tmem_alloc_cols("
                f"{acc_frag_base}, arch='sm_100')"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.tmem_holding_buf} = cute.arch.alloc_smem(cutlass.Int32, 1)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.tmem_alloc_barrier} = cutlass.pipeline.NamedBarrier("
                f"barrier_id=1, num_threads={_tcgen05_tmem_barrier_thread_count(bm, bn)})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.tmem_allocator} = cutlass.utils.TmemAllocator("
                f"{tcgen05_plan.tmem_holding_buf}, "
                f"barrier_for_retrieve={tcgen05_plan.tmem_alloc_barrier}, "
                "allocator_warp_id=0, is_two_cta=False)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.tmem_allocator}.allocate({tcgen05_plan.acc_tmem_cols})"
            )
        )
        prefix.append(
            statement_from_string(f"{tcgen05_plan.tmem_allocator}.wait_for_alloc()")
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_tmem_ptr} = "
                f"{tcgen05_plan.tmem_allocator}.retrieve_ptr({acc_dtype_str})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_pipeline_barriers} = cute.arch.alloc_smem(cutlass.Int64, 2)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_pipeline_producer_group} = "
                f"cutlass.pipeline.CooperativeGroup("
                f"cutlass.pipeline.Agent.Thread, {tcgen05_plan.consumer_thread_count})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_pipeline_consumer_group} = "
                f"cutlass.pipeline.CooperativeGroup("
                f"cutlass.pipeline.Agent.Thread, {tcgen05_plan.consumer_thread_count})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_pipeline} = cutlass.pipeline.PipelineUmmaAsync.create("
                "num_stages=1, "
                f"producer_group={tcgen05_plan.acc_pipeline_producer_group}, "
                f"consumer_group={tcgen05_plan.acc_pipeline_consumer_group}, "
                f"barrier_storage={tcgen05_plan.acc_pipeline_barriers})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_producer_state} = cutlass.pipeline.make_pipeline_state("
                "cutlass.pipeline.PipelineUserType.Producer, 1)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_consumer_state} = cutlass.pipeline.make_pipeline_state("
                "cutlass.pipeline.PipelineUserType.Consumer, 1)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{acc_frag_base} = cute.make_tensor("
                f"{tcgen05_plan.acc_tmem_ptr}, {acc_frag_base}.layout)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{acc_frag} = {acc_frag_base}[(None, 0, 0, {tcgen05_plan.acc_producer_state}.index)]"
            )
        )
        prefix.append(
            statement_from_string(
                f"if {mma_active}:\n"
                f"    {tcgen05_plan.acc_pipeline}.producer_acquire({tcgen05_plan.acc_producer_state})"
            )
        )
    else:
        prefix.append(
            statement_from_string(
                f"{acc_frag} = cute.make_fragment("
                f"{tiled_mma}.partition_shape_C(({bm}, {bn})), {acc_dtype_str})"
            )
        )
    # Allocate shared memory for A and B tiles (reused across K iterations)
    # Keep these allocations in the device-loop prefix. Lane-loop MMA relies on
    # per-iteration shared-memory state; hoisting them outside the lane loops
    # regresses the existing lane-loop coverage.
    smem_a_ptr = df.new_var("smem_a")
    smem_b_ptr = df.new_var("smem_b")
    smem_a = df.new_var("sA")
    smem_b = df.new_var("sB")
    smem_a_mma = df.new_var("sA_mma")
    smem_b_mma = df.new_var("sB_mma")
    smem_a_mma_next = df.new_var("sA_mma_next")
    smem_b_mma_next = df.new_var("sB_mma_next")
    smem_a_desc = df.new_var("sA_desc")
    smem_b_desc = df.new_var("sB_desc")
    mma_stage = df.new_var("mma_stage")
    mma_next_stage = df.new_var("mma_next_stage")
    if mma_impl == "tcgen05":
        assert tcgen05_plan is not None
        prefix.append(
            statement_from_string(
                f"{smem_a_ptr} = cute.arch.alloc_smem("
                f"{input_dtype_str}, cute.cosize({tcgen05_plan.smem_a_layout}.outer), alignment=128)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{smem_a} = cute.make_tensor("
                f"cute.recast_ptr({smem_a_ptr}, {tcgen05_plan.smem_a_layout}.inner, dtype={input_dtype_str}), "
                f"{tcgen05_plan.smem_a_layout}.outer)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{smem_b_ptr} = cute.arch.alloc_smem("
                f"{input_dtype_str}, cute.cosize({tcgen05_plan.smem_b_layout}.outer), alignment=128)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{smem_b} = cute.make_tensor("
                f"cute.recast_ptr({smem_b_ptr}, {tcgen05_plan.smem_b_layout}.inner, dtype={input_dtype_str}), "
                f"{tcgen05_plan.smem_b_layout}.outer)"
            )
        )
    else:
        prefix.append(
            statement_from_string(
                f"{smem_a_ptr} = cute.arch.alloc_smem({input_dtype_str}, {bm * bk})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{smem_a} = cute.make_tensor("
                f"{smem_a_ptr}, cute.make_layout(({bm}, {bk}), stride=({bk}, 1)))"
            )
        )
        prefix.append(
            statement_from_string(
                f"{smem_b_ptr} = cute.arch.alloc_smem({input_dtype_str}, {bn * bk})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{smem_b} = cute.make_tensor("
                f"{smem_b_ptr}, cute.make_layout(({bn}, {bk}), stride=({bk}, 1)))"
            )
        )
    # === loop body: global → smem → register → gemm ===
    rA = df.new_var("rA")
    rB = df.new_var("rB")
    tAsA = df.new_var("tAsA")
    tBsB = df.new_var("tBsB")

    # --- Global → Shared memory with masking ---
    # Each thread loads elements into shared memory using scalar indexing
    # with bounds checking for non-divisible tile boundaries.
    m_size = int(lhs_fake.shape[0])
    n_size = int(rhs_fake.shape[1])

    if acc_expr is None and mma_impl == "universal":
        cg.add_statement(
            statement_from_string(
                f"if {k_offset_var} == {k_loop_begin_expr}:\n"
                f"    for _mma_i in range(cute.size({acc_frag})):\n"
                f"        {acc_frag}[_mma_i] = {acc_dtype_str}(0.0)"
            )
        )
    elif acc_expr is not None and mma_impl == "universal":
        cg.add_statement(
            statement_from_string(
                f"if {k_offset_var} == {k_loop_begin_expr}:\n"
                f"    for _mma_i in range(cute.size({acc_frag})):\n"
                f"        {acc_frag}[_mma_i] = {acc_dtype_str}({{acc}})",
                acc=acc_expr,
            )
        )
    elif acc_expr is None:
        assert mma_active is not None
        if mma_impl == "warp":
            cg.add_statement(
                statement_from_string(
                    f"if {mma_active} and {k_offset_var} == {k_loop_begin_expr}:\n"
                    f"    for _mma_i in range(cute.size({acc_frag})):\n"
                    f"        {acc_frag}[_mma_i] = {acc_dtype_str}(0.0)"
                )
            )
    else:
        raise AssertionError("non-universal MMA with acc_expr should fall back")
    if mma_impl == "universal":
        cg.add_statement(
            statement_from_string(
                f"if {n_local} == cutlass.Int32(0):\n"
                f"    for _k in range({bk}):\n"
                f"        _gk = {k_offset_var} + cutlass.Int32(_k)\n"
                f"        {smem_a}[{m_local}, cutlass.Int32(_k)] = ("
                f"{lhs_arg_name}[{m_global}, _gk] "
                f"if {m_global} < cutlass.Int32({m_size}) "
                f"and _gk < cutlass.Int32({k_total_size}) "
                f"else {input_dtype_str}(0.0))"
            )
        )
        cg.add_statement(
            statement_from_string(
                f"if {m_local} == cutlass.Int32(0):\n"
                f"    for _k in range({bk}):\n"
                f"        _gk = {k_offset_var} + cutlass.Int32(_k)\n"
                f"        {smem_b}[{n_local}, cutlass.Int32(_k)] = ("
                f"{rhs_arg_name}[_gk, {n_global}] "
                f"if {n_global} < cutlass.Int32({n_size}) "
                f"and _gk < cutlass.Int32({k_total_size}) "
                f"else {input_dtype_str}(0.0))"
            )
        )
    else:
        active_threads = bm * mma_phys_n
        assert mma_active is not None and mma_thread_linear is not None
        if mma_impl == "tcgen05":
            assert tcgen05_plan is not None
            cg.add_statement(
                statement_from_string(
                    f"{mma_stage} = "
                    f"({k_offset_var} // cutlass.Int32({bk})) % {tcgen05_plan.ab_stage_count}"
                )
            )
            cg.add_statement(
                statement_from_string(
                    f"{mma_next_stage} = "
                    f"({mma_stage} + cutlass.Int32(1)) % {tcgen05_plan.ab_stage_count}"
                )
            )
            cg.add_statement(
                statement_from_string(
                    f"{smem_a_mma} = {smem_a}[(None, 0, 0, {mma_stage})]"
                )
            )
            cg.add_statement(
                statement_from_string(
                    f"{smem_b_mma} = {smem_b}[(None, 0, 0, {mma_stage})]"
                )
            )
            cg.add_statement(
                statement_from_string(
                    f"{smem_a_mma_next} = {smem_a}[(None, 0, 0, {mma_next_stage})]"
                )
            )
            cg.add_statement(
                statement_from_string(
                    f"{smem_b_mma_next} = {smem_b}[(None, 0, 0, {mma_next_stage})]"
                )
            )
        smem_a_store = f"{smem_a}[_row, _col]"
        smem_b_store = f"{smem_b}[_row, _col]"
        if mma_impl == "tcgen05":
            smem_a_store = f"{smem_a_mma}[((_row, _col),)]"
            smem_b_store = f"{smem_b_mma}[((_row, _col),)]"
        cg.add_statement(
            statement_from_string(
                f"if {mma_active}:\n"
                f"    for _load_i in range(({bm * bk} + {active_threads} - 1) // {active_threads}):\n"
                f"        _flat = {mma_thread_linear} + cutlass.Int32(_load_i) * cutlass.Int32({active_threads})\n"
                f"        if _flat < cutlass.Int32({bm * bk}):\n"
                f"            _row = _flat // cutlass.Int32({bk})\n"
                f"            _col = _flat % cutlass.Int32({bk})\n"
                f"            _gm = {m_offset_var} + _row\n"
                f"            _gk = {k_offset_var} + _col\n"
                f"            {smem_a_store} = ("
                f"{lhs_arg_name}[_gm, _gk] "
                f"if _gm < cutlass.Int32({m_size}) "
                f"and _gk < cutlass.Int32({k_total_size}) "
                f"else {input_dtype_str}(0.0))"
            )
        )
        cg.add_statement(
            statement_from_string(
                f"if {mma_active}:\n"
                f"    for _load_i in range(({bn * bk} + {active_threads} - 1) // {active_threads}):\n"
                f"        _flat = {mma_thread_linear} + cutlass.Int32(_load_i) * cutlass.Int32({active_threads})\n"
                f"        if _flat < cutlass.Int32({bn * bk}):\n"
                f"            _row = _flat // cutlass.Int32({bk})\n"
                f"            _col = _flat % cutlass.Int32({bk})\n"
                f"            _gn = {n_offset_var} + _row\n"
                f"            _gk = {k_offset_var} + _col\n"
                f"            {smem_b_store} = ("
                f"{rhs_arg_name}[_gk, _gn] "
                f"if _gn < cutlass.Int32({n_size}) "
                f"and _gk < cutlass.Int32({k_total_size}) "
                f"else {input_dtype_str}(0.0))"
            )
        )

    cg.add_statement(statement_from_string("cute.arch.sync_threads()"))

    # --- Shared → Register with f16→f32 cast ---
    if mma_impl == "universal":
        cg.add_statement(
            statement_from_string(f"{tAsA} = {thr_mma}.partition_A({smem_a})")
        )
        cg.add_statement(
            statement_from_string(f"{tBsB} = {thr_mma}.partition_B({smem_b})")
        )
        cg.add_statement(
            statement_from_string(
                f"{rA} = cute.make_fragment_like({tAsA}, {acc_dtype_str})"
            )
        )
        cg.add_statement(
            statement_from_string(
                f"{rB} = cute.make_fragment_like({tBsB}, {acc_dtype_str})"
            )
        )
        cg.add_statement(
            statement_from_string(
                f"for _mma_i in range(cute.size({rA})):\n"
                f"    {rA}[_mma_i] = {acc_dtype_str}({tAsA}[_mma_i])"
            )
        )
        cg.add_statement(
            statement_from_string(
                f"for _mma_i in range(cute.size({rB})):\n"
                f"    {rB}[_mma_i] = {acc_dtype_str}({tBsB}[_mma_i])"
            )
        )
        cg.add_statement(
            statement_from_string(
                f"cute.gemm({tiled_mma}, {acc_frag}, {rA}, {rB}, {acc_frag})"
            )
        )
    else:
        assert mma_active is not None
        if mma_impl == "warp":
            cg.add_statement(
                statement_from_string(
                    f"if {mma_active}:\n"
                    f"    {tAsA} = {thr_mma}.partition_A({smem_a})\n"
                    f"    {tBsB} = {thr_mma}.partition_B({smem_b})\n"
                    f"    {rA} = cute.make_fragment_like({tAsA}, {input_dtype_str})\n"
                    f"    {rB} = cute.make_fragment_like({tBsB}, {input_dtype_str})\n"
                    f"    for _mma_i in range(cute.size({rA})):\n"
                    f"        {rA}[_mma_i] = {tAsA}[_mma_i]\n"
                    f"    for _mma_i in range(cute.size({rB})):\n"
                    f"        {rB}[_mma_i] = {tBsB}[_mma_i]\n"
                    f"    cute.gemm({tiled_mma}, {acc_frag}, {rA}, {rB}, {acc_frag})"
                )
            )
        else:
            assert tcgen05_plan is not None
            cg.add_statement(
                statement_from_string("cute.arch.fence_view_async_shared()")
            )
            cg.add_statement(
                statement_from_string(
                    f"{smem_a_desc} = cute.make_tensor("
                    f"cute.nvgpu.tcgen05.make_umma_smem_desc("
                    f"{smem_a_mma}.iterator, {smem_a_mma}.layout, 'k', "
                    f"next_src={smem_a_mma_next}.iterator"
                    "), "
                    f"{tcgen05_plan.smem_desc_view_layout})"
                )
            )
            cg.add_statement(
                statement_from_string(
                    f"{smem_b_desc} = cute.make_tensor("
                    f"cute.nvgpu.tcgen05.make_umma_smem_desc("
                    f"{smem_b_mma}.iterator, {smem_b_mma}.layout, 'k', "
                    f"next_src={smem_b_mma_next}.iterator"
                    "), "
                    f"{tcgen05_plan.smem_desc_view_layout})"
                )
            )
            cg.add_statement(
                statement_from_string(
                    f"{tiled_mma}.set("
                    f"cute.nvgpu.tcgen05.Field.ACCUMULATE, "
                    f"{k_offset_var} != {k_loop_begin_expr})"
                )
            )
            cg.add_statement(
                statement_from_string(
                    f"if {tcgen05_plan.exec_active}:\n"
                    f"    cute.mma_atom_call("
                    f"{tiled_mma}, {acc_frag}, {smem_a_desc}, {smem_b_desc}, {acc_frag})"
                )
            )
            cg.add_statement(statement_from_string("cute.arch.sync_threads()"))

    # === outer_suffix: convert fragment → per-thread scalar ===
    # Allocate smem_c in outer_prefix so all smem is allocated at the same
    # scope level (CuTe DSL assigns static smem offsets per scope).
    smem_c_ptr = df.new_var("smem_c")
    smem_c = df.new_var("smem_c_t")
    tCsC = df.new_var("tCsC")
    tcgen05_tacc = df.new_var("tcgen05_tacc")
    tcgen05_tacc_epi = df.new_var("tcgen05_tacc_epi")
    tcgen05_tmem_ref = df.new_var("tcgen05_tmem_ref")
    tcgen05_tiled_copy_t2r = df.new_var("tcgen05_tiled_copy_t2r")
    tcgen05_thr_copy_t2r = df.new_var("tcgen05_thr_copy_t2r")
    tcgen05_ttr_tacc_base = df.new_var("tcgen05_ttr_tacc_base")
    tcgen05_ttr_tacc_stage = df.new_var("tcgen05_ttr_tacc_stage")
    tcgen05_ttr_tacc = df.new_var("tcgen05_ttr_tacc")
    tcgen05_ttr_tacc_mn = df.new_var("tcgen05_ttr_tacc_mn")
    tcgen05_ttr_racc = df.new_var("tcgen05_ttr_racc")
    tcgen05_tcgc = df.new_var("tcgen05_tcgc")
    tcgen05_tcgc_planned = df.new_var("tcgen05_tcgc_planned")
    tcgen05_tcgc_epi = df.new_var("tcgen05_tcgc_epi")
    tcgen05_ttr_gc = df.new_var("tcgen05_ttr_gC")
    tcgen05_ttr_gc_grouped = df.new_var("tcgen05_ttr_gC_grouped")
    tcgen05_ttr_rc = df.new_var("tcgen05_ttr_rc")
    tcgen05_mcld = df.new_var("tcgen05_mcld")
    tcgen05_num_bits = df.new_var("tcgen05_num_bits")
    tcgen05_simt_atom = df.new_var("tcgen05_simt_atom")
    tcgen05_acc_vec = df.new_var("tcgen05_acc_vec")
    tcgen05_subtile_count = df.new_var("tcgen05_subtile_count")
    result_var = df.new_var("mma_result")

    tile_numel = bm * bn
    prefix.append(
        statement_from_string(
            f"{smem_c_ptr} = cute.arch.alloc_smem({acc_dtype_str}, {tile_numel}, alignment=128)"
        )
    )
    prefix.append(
        statement_from_string(
            f"{smem_c} = cute.make_tensor("
            f"{smem_c_ptr}, cute.make_layout(({bm}, {bn}), stride=({bn}, 1)))"
        )
    )
    suffix = device_loop.outer_suffix
    if mma_impl == "universal":
        suffix.append(
            statement_from_string(f"{tCsC} = {thr_mma}.partition_C({smem_c})")
        )
        suffix.append(
            statement_from_string(
                f"for _mma_i in range(cute.size({tCsC})):\n"
                f"    {tCsC}[_mma_i] = {acc_frag}[_mma_i]"
            )
        )
    else:
        assert mma_active is not None
        if mma_impl == "warp":
            suffix.append(
                statement_from_string(
                    f"if {mma_active}:\n"
                    f"    {tCsC} = {thr_mma}.partition_C({smem_c})\n"
                    f"    for _mma_i in range(cute.size({tCsC})):\n"
                    f"        {tCsC}[_mma_i] = {acc_frag}[_mma_i]"
                )
            )
        else:
            assert tcgen05_plan is not None
            suffix.append(
                statement_from_string(
                    f"if {mma_active}:\n"
                    f"    {tcgen05_plan.acc_pipeline}.producer_commit({tcgen05_plan.acc_producer_state})"
                )
            )
            suffix.append(
                statement_from_string(f"{tcgen05_plan.acc_producer_state}.advance()")
            )
            suffix.append(statement_from_string("cute.arch.sync_threads()"))
            suffix.append(
                statement_from_string(
                    f"{tcgen05_tcgc} = "
                    "cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
                    f"{thr_mma}.partition_C({smem_c}))"
                )
            )
            suffix.append(
                statement_from_string(
                    f"{tcgen05_tcgc_planned} = {_tcgen05_epilogue_dest_expr(tcgen05_plan, tcgen05_tcgc)}"
                )
            )
            suffix.append(
                statement_from_string(
                    f"{tcgen05_tacc} = "
                    "cutlass.utils.gemm.sm100.transform_partitioned_tensor_layout("
                    f"{acc_frag_base})"
                )
            )
            suffix.append(
                statement_from_string(
                    f"{tcgen05_tacc_epi} = cute.flat_divide({tcgen05_tacc}, {tcgen05_plan.epi_tile})"
                )
            )
            suffix.append(
                statement_from_string(
                    f"{tcgen05_tmem_ref} = {tcgen05_tacc_epi}[(None, None, 0, 0, 0)]"
                )
            )
            suffix.append(
                statement_from_string(
                    f"{tcgen05_tiled_copy_t2r} = "
                    f"cute.nvgpu.tcgen05.make_tmem_copy({tcgen05_plan.tmem_load_atom}, {tcgen05_tmem_ref})"
                )
            )
            suffix.append(
                statement_from_string(
                    f"{tcgen05_thr_copy_t2r} = {tcgen05_tiled_copy_t2r}.get_slice({mma_thread_linear})"
                )
            )
            suffix.append(
                statement_from_string(
                    f"{tcgen05_ttr_tacc_base} = {tcgen05_thr_copy_t2r}.partition_S({tcgen05_tacc_epi})"
                )
            )
            suffix.append(
                statement_from_string(
                    f"{tcgen05_tcgc_epi} = cute.flat_divide({tcgen05_tcgc_planned}, {tcgen05_plan.epi_tile})"
                )
            )
            suffix.append(
                statement_from_string(
                    f"{tcgen05_ttr_gc} = {tcgen05_thr_copy_t2r}.partition_D({tcgen05_tcgc_epi})"
                )
            )
            suffix.append(
                statement_from_string(
                    f"{tcgen05_ttr_racc} = cute.make_rmem_tensor("
                    f"{tcgen05_ttr_gc}[(None, None, None, 0, 0, 0, 0, 0)].shape, {acc_dtype_str})"
                )
            )
            suffix.append(
                statement_from_string(
                    f"{tcgen05_ttr_rc} = "
                    f"cute.make_rmem_tensor({tcgen05_ttr_racc}.shape, {acc_dtype_str})"
                )
            )
            suffix.append(
                statement_from_string(
                    f"{tcgen05_ttr_tacc_stage} = "
                    f"{tcgen05_ttr_tacc_base}[(None, None, None, None, None, {tcgen05_plan.acc_consumer_state}.index)]"
                )
            )
            suffix.append(
                statement_from_string(
                    f"if {mma_active}:\n"
                    f"    {tcgen05_plan.acc_pipeline}.consumer_wait({tcgen05_plan.acc_consumer_state})"
                )
            )
            suffix.append(
                statement_from_string(
                    f"{tcgen05_ttr_tacc} = cute.group_modes("
                    f"{tcgen05_ttr_tacc_stage}, 3, cute.rank({tcgen05_ttr_tacc_stage}))"
                )
            )
            suffix.append(
                statement_from_string(
                    f"{tcgen05_ttr_gc_grouped} = cute.group_modes("
                    f"{tcgen05_ttr_gc}, 3, cute.rank({tcgen05_ttr_gc}))"
                )
            )
            suffix.append(
                statement_from_string(
                    f"{tcgen05_mcld} = cute.max_common_layout("
                    f"{tcgen05_ttr_rc}.layout, "
                    f"{tcgen05_ttr_gc_grouped}[(None, None, None, 0)].layout)"
                )
            )
            suffix.append(
                statement_from_string(
                    f"{tcgen05_num_bits} = min("
                    f"{tcgen05_ttr_gc_grouped}.iterator.alignment * 8, "
                    f"cute.size({tcgen05_mcld}) * {acc_dtype_str}.width, 256)"
                )
            )
            suffix.append(
                statement_from_string(
                    f"{tcgen05_simt_atom} = cute.make_copy_atom("
                    f"cute.nvgpu.CopyUniversalOp(), {acc_dtype_str}, "
                    f"num_bits_per_copy={tcgen05_num_bits}, "
                    f"l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE)"
                )
            )
            suffix.append(
                statement_from_string(
                    f"{tcgen05_subtile_count} = cute.size({tcgen05_ttr_tacc}.shape, mode=[3])"
                )
            )
            suffix.append(
                statement_from_string(
                    f"for _tcgen05_subtile in range({tcgen05_subtile_count}):\n"
                    f"    if {mma_active}:\n"
                    f"        {tcgen05_ttr_tacc_mn} = {tcgen05_ttr_tacc}[(None, None, None, cutlass.Int32(_tcgen05_subtile))]\n"
                    f"        cute.copy({tcgen05_tiled_copy_t2r}, {tcgen05_ttr_tacc_mn}, {tcgen05_ttr_racc})\n"
                    f"        {tcgen05_acc_vec} = {tcgen05_ttr_racc}.load()\n"
                    f"        {tcgen05_ttr_rc}.store({tcgen05_acc_vec})\n"
                    f"        cute.copy({tcgen05_simt_atom}, {tcgen05_ttr_rc}, {tcgen05_ttr_gc_grouped}[(None, None, None, cutlass.Int32(_tcgen05_subtile))])\n"
                    f"    cute.arch.sync_threads()"
                )
            )
            suffix.append(
                statement_from_string(
                    f"if {mma_active}:\n"
                    f"    with cute.arch.elect_one():\n"
                    f"        {tcgen05_plan.acc_pipeline}.consumer_release({tcgen05_plan.acc_consumer_state})"
                )
            )
            suffix.append(
                statement_from_string(f"{tcgen05_plan.acc_consumer_state}.advance()")
            )
    suffix.append(statement_from_string("cute.arch.sync_threads()"))
    if mma_impl == "tcgen05":
        assert tcgen05_plan is not None
        suffix.append(
            statement_from_string(
                f"if {mma_active}:\n"
                f"    {tcgen05_plan.acc_pipeline}.producer_tail({tcgen05_plan.acc_producer_state})"
            )
        )
        suffix.append(statement_from_string("cute.arch.sync_threads()"))
        suffix.append(
            statement_from_string(
                f"{tcgen05_plan.tmem_allocator}.relinquish_alloc_permit()"
            )
        )
        suffix.append(
            statement_from_string(
                f"{tcgen05_plan.tmem_allocator}.free("
                f"{tcgen05_plan.acc_tmem_ptr}, {tcgen05_plan.acc_tmem_cols})"
            )
        )

    # Each thread reads its own (m, n) element from shared memory
    suffix.append(
        statement_from_string(f"{result_var} = {smem_c}[{m_local}, {n_local}]")
    )

    return expr_from_string(result_var)


def _mma_active_n_threads(mma_impl: str) -> int:
    if mma_impl in ("warp", "tcgen05"):
        return 2
    return 0


def _tcgen05_pipeline_arrive_count(bm: int) -> int:
    # PipelineUmmaAsync counts elected arrivals, not every participating lane.
    return (bm * _mma_active_n_threads("tcgen05")) // 32


_tcgen05_tmem_barrier_thread_count = operator.mul


def _mma_impl_matches_problem_shape(
    mma_impl: str,
    input_dtype: torch.dtype,
    *,
    bm: int,
    bn: int,
    bk: int,
) -> bool:
    if mma_impl == "universal":
        return True
    if input_dtype not in (torch.float16, torch.bfloat16) or bk != 16 or bn != 8:
        return False
    if mma_impl == "warp":
        return bm == 16
    if mma_impl == "tcgen05":
        return bm in (64, 128)
    return False


def _is_zero_acc_expr(acc_expr: ast.AST) -> bool:
    if isinstance(acc_expr, ast.Constant):
        return acc_expr.value in (0, 0.0)
    if isinstance(acc_expr, ast.Call):
        if len(acc_expr.args) != 1 or acc_expr.keywords:
            return False
        if not _is_zero_acc_expr(acc_expr.args[0]):
            return False
        if isinstance(acc_expr.func, ast.Attribute):
            return acc_expr.func.attr in {"Float16", "Float32", "BFloat16"}
        if isinstance(acc_expr.func, ast.Name):
            return acc_expr.func.id in {"float", "int"}
    return False


def _choose_mma_impl(
    input_dtype: torch.dtype,
    *,
    bm: int,
    bn: int,
    bk: int,
) -> str:
    env_choice = os.environ.get("HELION_CUTE_MMA_IMPL", "auto").strip().lower()
    support = get_cute_mma_support()
    if env_choice != "auto":
        if env_choice not in support.supported_impls:
            raise exc.BackendUnsupported(
                "cute",
                (
                    f"Requested HELION_CUTE_MMA_IMPL={env_choice!r} is not supported "
                    f"on this machine. Supported: {support.supported_impls}"
                ),
            )
        if _mma_impl_matches_problem_shape(
            env_choice,
            input_dtype,
            bm=bm,
            bn=bn,
            bk=bk,
        ):
            return env_choice
        return "universal"
    if _mma_impl_matches_problem_shape("tcgen05", input_dtype, bm=bm, bn=bn, bk=bk):
        if support.tcgen05_f16bf16:
            return "tcgen05"
    if _mma_impl_matches_problem_shape("warp", input_dtype, bm=bm, bn=bn, bk=bk):
        if support.warp_f16bf16:
            return "warp"
    return "universal"


def _make_tiled_mma_setup(
    mma_impl: str,
    tiled_mma: str,
    thr_mma: str,
    mma_thread_linear: str,
    input_dtype_str: str,
    acc_dtype_str: str,
    bm: int,
    bn: int,
) -> list[ast.AST]:
    if mma_impl == "warp":
        tiled_mma_expr = (
            "cute.make_tiled_mma("
            "cute.make_mma_atom("
            f"cute.nvgpu.warp.MmaF16BF16Op({input_dtype_str}, {acc_dtype_str}, (16, 8, 16))"
            "), atom_layout_mnk=(1, 1, 1))"
        )
    elif mma_impl == "tcgen05":
        tiled_mma_expr = _tcgen05_tiled_mma_expr(input_dtype_str, acc_dtype_str, bm, bn)
    else:
        assert mma_thread_linear
        return [
            statement_from_string(
                f"{tiled_mma} = cute.make_tiled_mma("
                f"cute.nvgpu.MmaUniversalOp(abacc_dtype={acc_dtype_str}), "
                f"atom_layout_mnk=({bm}, {bn}, 1))"
            ),
            statement_from_string(
                f"{thr_mma} = {tiled_mma}.get_slice({mma_thread_linear})"
            ),
        ]
    return [
        statement_from_string(f"{tiled_mma} = {tiled_mma_expr}"),
        statement_from_string(
            f"{thr_mma} = {tiled_mma}.get_slice({mma_thread_linear})"
        ),
    ]


def _tcgen05_tiled_mma_expr(
    input_dtype_str: str,
    acc_dtype_str: str,
    bm: int,
    bn: int,
) -> str:
    return (
        "cutlass.utils.blackwell_helpers.make_trivial_tiled_mma("
        f"{input_dtype_str}, "
        "cute.nvgpu.tcgen05.OperandMajorMode.K, "
        "cute.nvgpu.tcgen05.OperandMajorMode.K, "
        f"{acc_dtype_str}, "
        "cute.nvgpu.tcgen05.CtaGroup.ONE, "
        f"({bm}, {bn}), "
        "cute.nvgpu.tcgen05.OperandSource.SMEM)"
    )


def _new_tcgen05_layout_plan(df: DeviceFunction) -> _Tcgen05LayoutPlan:
    return _Tcgen05LayoutPlan(
        acc_stage_count=df.new_var("tcgen05_acc_stage_count"),
        ab_stage_count=df.new_var("tcgen05_ab_stage_count"),
        exec_thread_count=df.new_var("tcgen05_exec_thread_count"),
        consumer_thread_count=df.new_var("tcgen05_pipeline_arrive_count"),
        exec_active=df.new_var("tcgen05_exec_active"),
        smem_a_layout=df.new_var("sA_layout"),
        smem_b_layout=df.new_var("sB_layout"),
        smem_desc_view_layout=df.new_var("smem_desc_view_layout"),
        c_layout=df.new_var("tcgen05_c_layout"),
        smem_c_layout=df.new_var("sC_layout"),
        epi_tile=df.new_var("tcgen05_epi_tile"),
        tmem_load_atom=df.new_var("tcgen05_tmem_load_atom"),
        acc_tmem_cols=df.new_var("tcgen05_acc_tmem_cols"),
        tmem_holding_buf=df.new_var("tcgen05_tmem_holding_buf"),
        tmem_alloc_barrier=df.new_var("tcgen05_tmem_alloc_barrier"),
        tmem_allocator=df.new_var("tcgen05_tmem_allocator"),
        acc_tmem_ptr=df.new_var("tcgen05_acc_tmem_ptr"),
        acc_pipeline_barriers=df.new_var("tcgen05_acc_pipeline_barriers"),
        acc_pipeline_producer_group=df.new_var("tcgen05_acc_pipeline_producer_group"),
        acc_pipeline_consumer_group=df.new_var("tcgen05_acc_pipeline_consumer_group"),
        acc_pipeline=df.new_var("tcgen05_acc_pipeline"),
        acc_producer_state=df.new_var("tcgen05_acc_producer_state"),
        acc_consumer_state=df.new_var("tcgen05_acc_consumer_state"),
        epilogue_rest_mode=df.new_var("tcgen05_epilogue_rest_mode"),
    )


def _make_tcgen05_layout_plan_setup(
    plan: _Tcgen05LayoutPlan,
    tiled_mma: str,
    *,
    bm: int,
    bn: int,
    bk: int,
    input_dtype_str: str,
    acc_dtype_str: str,
) -> list[ast.AST]:
    return [
        statement_from_string(f"{plan.ab_stage_count} = cutlass.Int32(2)"),
        statement_from_string(f"{plan.exec_thread_count} = cutlass.Int32(32)"),
        statement_from_string(
            f"{plan.consumer_thread_count} = "
            f"cutlass.Int32({_tcgen05_pipeline_arrive_count(bm)})"
        ),
        statement_from_string(
            f"{plan.smem_a_layout} = {_tcgen05_smem_layout_expr(tiled_mma, bm, bn, bk, input_dtype_str, operand='a')}"
        ),
        statement_from_string(
            f"{plan.smem_b_layout} = {_tcgen05_smem_layout_expr(tiled_mma, bm, bn, bk, input_dtype_str, operand='b')}"
        ),
        statement_from_string(
            f"{plan.smem_desc_view_layout} = cute.make_layout(1, stride=0)"
        ),
        statement_from_string(
            f"{plan.c_layout} = cutlass.utils.layout.LayoutEnum.ROW_MAJOR"
        ),
        statement_from_string(
            f"{plan.epi_tile} = cutlass.utils.blackwell_helpers.compute_epilogue_tile_shape("
            f"({bm}, {bn}), False, {plan.c_layout}, "
            f"{acc_dtype_str})"
        ),
        statement_from_string(
            f"{plan.smem_c_layout} = cutlass.utils.blackwell_helpers.make_smem_layout_epi("
            f"{acc_dtype_str}, {plan.c_layout}, {plan.epi_tile}, 1)"
        ),
        statement_from_string(
            f"{plan.tmem_load_atom} = cutlass.utils.blackwell_helpers.get_tmem_load_op("
            f"({bm}, {bn}, {bk}), {plan.c_layout}, "
            f"{acc_dtype_str}, {acc_dtype_str}, {plan.epi_tile}, False)"
        ),
        statement_from_string(
            f"{plan.epilogue_rest_mode} = cute.make_layout(1, stride=0)"
        ),
    ]


def _tcgen05_epilogue_dest_expr(plan: _Tcgen05LayoutPlan, tensor: str) -> str:
    planned_layout = tensor + ".layout"
    for _ in range(3):
        planned_layout = f"cute.append({planned_layout}, {plan.epilogue_rest_mode})"
    return f"cute.make_tensor({tensor}.iterator, {planned_layout})"


def _tcgen05_smem_layout_expr(
    tiled_mma: str,
    bm: int,
    bn: int,
    bk: int,
    dtype_str: str,
    *,
    operand: str,
) -> str:
    if operand == "a":
        return (
            "cutlass.utils.blackwell_helpers.make_smem_layout_a("
            f"{tiled_mma}, ({bm}, {bn}, {bk}), {dtype_str}, 2)"
        )
    assert operand == "b"
    return (
        "cutlass.utils.blackwell_helpers.make_smem_layout_b("
        f"{tiled_mma}, ({bm}, {bn}, {bk}), {dtype_str}, 2)"
    )


# ---- Aten lowering entry point (addmm/mm/bmm/baddbmm) ----


def codegen_cute_mma(
    ctx: LoweringContext,
    node: Node,
    with_acc: bool,
) -> ast.AST | None:
    """Generate MMA code for an aten addmm/mm node.  Returns None to fall back."""
    from ..generate_ast import GenerateAST

    if not isinstance(ctx.cg, GenerateAST):
        return None
    if _has_active_lane_loops(ctx.cg):
        return None
    if (
        ctx.cg.current_grid_state is None
        or len(ctx.cg.current_grid_state.block_ids) != 2
    ):
        return None
    if not can_codegen_cute_mma_aten(node, with_acc):
        return None

    if with_acc:
        acc_node = node.args[0]
        assert isinstance(acc_node, Node)
        acc_expr = (
            None if _is_zero_init_acc_node(acc_node) else ctx.to_ast(ctx.env[acc_node])
        )
        lhs_node, rhs_node = node.args[1], node.args[2]
    else:
        acc_expr = None
        lhs_node, rhs_node = node.args[0], node.args[1]
    assert isinstance(lhs_node, Node) and isinstance(rhs_node, Node)

    return _emit_mma_pipeline(
        ctx.cg,
        lhs_node,
        rhs_node,
        acc_expr=acc_expr,
        fx_node=node,
    )


# ---- hl.dot entry point ----


def codegen_cute_mma_dot(state: CodegenState) -> object | None:
    """Generate MMA code for an hl.dot node.  Returns None to fall back."""
    from ..generate_ast import GenerateAST

    if not isinstance(state.codegen, GenerateAST):
        return None
    if _has_active_lane_loops(state.codegen):
        return None
    if (
        state.codegen.current_grid_state is None
        or len(state.codegen.current_grid_state.block_ids) != 2
    ):
        return None
    if state.fx_node is None:
        return None
    if not can_codegen_cute_mma_dot(state.fx_node):
        return None

    lhs_node = state.fx_node.args[0]
    rhs_node = state.fx_node.args[1]
    acc_expr = None
    if len(state.fx_node.args) > 2:
        acc_node = state.fx_node.args[2]
        if isinstance(acc_node, Node) and _is_zero_init_acc_node(acc_node):
            acc_expr = None
        else:
            acc_ast = state.ast_arg(2)
            if not (isinstance(acc_ast, ast.Constant) and acc_ast.value is None):
                acc_expr = acc_ast
    assert isinstance(lhs_node, Node) and isinstance(rhs_node, Node)

    result = _emit_mma_pipeline(
        state.codegen,
        lhs_node,
        rhs_node,
        acc_expr=acc_expr,
        fx_node=state.fx_node,
    )
    if result is None:
        return None

    acc_proxy = state.proxy_args[2] if len(state.proxy_args) > 2 else None
    if isinstance(acc_proxy, FakeTensor) and acc_proxy.dtype != torch.float32:
        return cast_ast(result, acc_proxy.dtype)

    out_dtype_proxy = state.proxy_args[3] if len(state.proxy_args) > 3 else None
    if isinstance(out_dtype_proxy, torch.dtype) and out_dtype_proxy != torch.float32:
        return cast_ast(result, out_dtype_proxy)

    return result
