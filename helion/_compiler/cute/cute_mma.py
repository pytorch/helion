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
import os
import textwrap
from typing import TYPE_CHECKING
from typing import Protocol
from typing import cast

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.node import Node

from ... import exc
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..device_function import CuteTcgen05MatmulPlan
from ..device_function import CuteTcgen05StoreValue
from ..dtype_utils import cast_ast
from ..matmul_utils import _needs_f32_accumulator
from ..tile_strategy import DeviceLoopState
from .cutedsl_compat import emit_pipeline_advance
from .layout import MatmulExecutionKind
from .layout import MatmulExecutionPlan
from .matmul_utils import analyze_direct_grouped_n_loads
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

# Register reallocation budget for tcgen05 warp-specialized kernels.
# Producer warps (TMA loads, scheduler) only do address arithmetic and
# barrier ops, so they can give back registers; consumer warps (MMA
# exec, epilogue) need the extra budget for register-resident
# accumulators and TMEM↔RMEM staging. Values match Quack's sm100
# reference (`gemm_sm100.py`).
_TCGEN05_PRODUCER_REGS = 120
_TCGEN05_CONSUMER_REGS = 256

# Named-barrier ids reserved by Helion's tcgen05 codegen. Kept as module
# constants so the codegen sites read symbolically instead of hardcoding
# magic numbers, and so the next free id is obvious if a third role-local
# barrier is added.
_TCGEN05_TMEM_ALLOC_BARRIER_ID = 1
_TCGEN05_EPILOG_SYNC_BARRIER_ID = 2


@dataclass(frozen=True)
class _Tcgen05LayoutPlan:
    """Generated CuTe variable names for the tcgen05 layout setup.

    Pure name container: every field is the textual identifier of a value
    materialized in the kernel prefix. Compile-time integer constants
    (stage counts, arrive counts, barrier ids) are not stored here; they
    live as Python ints alongside ``CuteTcgen05MatmulPlan`` and get
    inlined at the codegen call site.
    """

    exec_active: str
    smem_a_layout: str
    smem_b_layout: str
    c_layout: str
    epi_tile: str
    tmem_load_atom: str
    acc_tmem_cols: str
    tmem_holding_buf: str
    tmem_dealloc_mbar_ptr: str
    tmem_alloc_barrier: str
    tmem_allocator: str
    acc_pipeline_barriers: str
    acc_pipeline_producer_group: str
    acc_pipeline_consumer_group: str
    acc_pipeline: str
    acc_producer_state: str
    acc_consumer_state: str
    epilogue_rest_mode: str


class _ConfigLike(Protocol):
    def get(self, key: str, default: object = ...) -> object: ...


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


def _trace_to_load(node: Node) -> Node | None:
    """Trace through casts/permutes to the underlying load node."""
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
    return cur


def _trace_to_load_tensor(node: Node) -> tuple[Node, str, torch.Tensor] | None:
    """Trace through casts/permutes to find the underlying load tensor.

    Only traces through data-preserving ops (type casts, permute).
    Does NOT trace through arithmetic (add, mul, etc.) because the MMA
    pipeline reads raw tensor data and those ops would be skipped.
    """
    load_node = _trace_to_load(node)
    if load_node is None:
        return None
    tensor_node = load_node.args[0]
    if not isinstance(tensor_node, Node):
        return None
    fake = tensor_node.meta.get("val")
    if not isinstance(fake, torch.Tensor):
        return None
    return load_node, tensor_node.name, fake


def _has_mma_operands(lhs_node: Node, rhs_node: Node) -> bool:
    """Check if lhs/rhs come from loads with MMA-compatible dtypes."""
    lhs_info = _trace_to_load_tensor(lhs_node)
    rhs_info = _trace_to_load_tensor(rhs_node)
    if lhs_info is None or rhs_info is None:
        return False
    lhs_load, _, lhs_fake = lhs_info
    rhs_load, _, rhs_fake = rhs_info
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


def _physical_mma_coord_expr(
    cg: GenerateAST,
    block_id: int,
) -> str:
    """Return the physical thread coordinate for an MMA output axis."""
    grid_state = cg.current_grid_state
    if grid_state is None:
        return "cutlass.Int32(0)"
    thread_axis = grid_state.block_thread_axes.get(block_id)
    if thread_axis is None:
        return "cutlass.Int32(0)"
    return f"cutlass.Int32(cute.arch.thread_idx()[{thread_axis}])"


def _local_mma_coord_expr(
    cg: GenerateAST,
    block_id: int,
) -> str:
    """Return the current block-local coordinate for an MMA output axis.

    Same as ``_physical_mma_coord_expr`` plus a lane offset when the grid
    strategy has registered a per-block lane var (the lane-loop fast path
    serializes `elements_per_thread` consecutive elements per physical
    thread).
    """
    coord = _physical_mma_coord_expr(cg, block_id)
    grid_state = cg.current_grid_state
    if grid_state is None or grid_state.block_thread_axes.get(block_id) is None:
        return coord

    strategy = grid_state.strategy
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


def _grid_thread_extent(cg: GenerateAST, block_id: int) -> int:
    grid_state = cg.current_grid_state
    if grid_state is None:
        return 1
    thread_axis = grid_state.block_thread_axes.get(block_id)
    if thread_axis is None:
        return 1
    return grid_state.thread_axis_sizes.get(thread_axis, 1)


def _grid_cta_thread_count(cg: GenerateAST) -> int:
    grid_state = cg.current_grid_state
    if grid_state is None:
        return 1
    cta_threads = 1
    for size in grid_state.thread_axis_sizes.values():
        cta_threads *= size
    return cta_threads


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


def _has_non_root_lane_loops(
    cg: GenerateAST, *, allowed_loop_states: tuple[DeviceLoopState, ...] = ()
) -> bool:
    seen: set[int] = set()
    allowed_ids = {id(loop_state) for loop_state in allowed_loop_states}
    for loops in cg.active_device_loops.values():
        for loop_state in loops:
            key = id(loop_state)
            if key in seen:
                continue
            seen.add(key)
            if loop_state is cg.current_grid_state or key in allowed_ids:
                continue
            strategy = getattr(loop_state, "strategy", None)
            lane_vars = getattr(strategy, "_lane_var_by_block", None)
            if lane_vars:
                return True
    return False


def prepare_cute_collective_lane_loop_suppression(
    cg: GenerateAST, graph: torch.fx.Graph
) -> None:
    from ..compile_environment import CompileEnvironment

    grid_state = cg.current_grid_state
    if grid_state is None:
        return

    env = CompileEnvironment.current()
    if env.backend_name != "cute":
        return

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target is torch.ops.aten.addmm.default:
            with_acc = True
            lhs_node = node.args[1]
            rhs_node = node.args[2]
            if not can_codegen_cute_mma_aten(node, with_acc):
                continue
        elif node.target is torch.ops.aten.mm.default:
            with_acc = False
            lhs_node = node.args[0]
            rhs_node = node.args[1]
            if not can_codegen_cute_mma_aten(node, with_acc):
                continue
        elif can_codegen_cute_mma_dot(node):
            lhs_node = node.args[0]
            rhs_node = node.args[1]
        else:
            continue

        if not isinstance(lhs_node, Node) or not isinstance(rhs_node, Node):
            continue

        lhs_info = _trace_to_load_tensor(lhs_node)
        rhs_info = _trace_to_load_tensor(rhs_node)
        if lhs_info is None or rhs_info is None:
            continue
        lhs_load, _, lhs_fake = lhs_info
        rhs_load, _, rhs_fake = rhs_info
        if lhs_fake.ndim != 2 or rhs_fake.ndim != 2:
            continue

        if not (
            isinstance(lhs_fake.shape[0], int)
            and isinstance(rhs_fake.shape[1], int)
            and isinstance(lhs_fake.shape[1], int)
        ):
            continue
        m = lhs_fake.shape[0]
        n = rhs_fake.shape[1]
        k = lhs_fake.shape[1]
        bm = bn = bk = None
        candidate_block_ids = [*grid_state.block_ids]
        if (
            k_loop_info := _get_mma_k_loop_info(
                cg, env, lhs_fake, rhs_fake, fx_node=node
            )
        ) is not None:
            _, k_block_id, _, k_block_size = k_loop_info
            candidate_block_ids.append(k_block_id)
            bk = int(k_block_size)
        for bid in dict.fromkeys(candidate_block_ids):
            size = env.block_sizes[bid].size
            bs = env.block_sizes[bid].from_config(cg.device_function.config)
            if not isinstance(bs, int):
                continue
            if isinstance(size, (int, torch.SymInt)):
                if bm is None and env.known_equal(size, lhs_fake.shape[0]):
                    bm = int(bs)
                elif bn is None and env.known_equal(size, rhs_fake.shape[1]):
                    bn = int(bs)
                elif bk is None and env.known_equal(size, lhs_fake.shape[1]):
                    bk = int(bs)
        if bm is None or bn is None or bk is None:
            continue
        if (
            _choose_mma_impl(
                lhs_fake.dtype, bm=bm, bn=bn, bk=bk, config=cg.device_function.config
            )
            != "tcgen05"
        ):
            continue
        if m % bm != 0 or n % bn != 0 or k % bk != 0:
            continue
        if (
            len(lhs_load.users) != 1
            or len(rhs_load.users) != 1
            or next(iter(lhs_load.users)) is not node
            or next(iter(rhs_load.users)) is not node
        ):
            continue

        cg.device_function.register_cute_collective_handled_load(lhs_load.name)
        cg.device_function.register_cute_collective_handled_load(rhs_load.name)
        if grid_state.has_lane_loops():
            cg.device_function.suppress_cute_root_lane_loops = True


def _mma_result_can_be_deferred(node: Node) -> bool:
    """Return True when the node value is only consumed after the K loop finishes."""
    return all(user.op == "output" for user in node.users)


@dataclass(frozen=True)
class _PerKiterTmaArgs:
    """Variable names + flags threaded into the per-K-iter TMA builders.

    All ``str`` fields name a Python identifier in the generated code.
    Only valid when the tcgen05 TMA path is active, so every name is
    guaranteed bound at the call site.
    """

    tma_pipeline: str
    tma_producer_state: str
    tma_consumer_state: str
    tma_producer_try_token: str
    tma_consumer_try_token: str
    tma_barrier_ptr: str
    tma_full_tile: str
    tma_next_full_tile: str
    tma_warp: str
    tma_atom_a: str
    tma_atom_b: str
    tma_gA: str
    tma_gB: str
    tma_sA: str
    tma_sB: str
    tma_k_tile: str
    tma_a_mcast_mask: str
    tma_b_mcast_mask: str
    ab_stage_count: int
    cluster_m: int
    is_two_cta: bool
    use_tma_a: bool
    use_tma_b: bool
    exec_active: str
    scalar_load_a: ast.stmt
    scalar_load_b: ast.stmt


def _kloop_tma_copy_a_src(args: _PerKiterTmaArgs, *, k_offset: str) -> str:
    """Per-K-iter TMA copy source for A; ``""`` when A is not TMA-loaded.

    A only multicasts in 2-CTA mode (asymmetric vs. B, which also
    multicasts across cluster CTAs).
    """
    if not args.use_tma_a:
        return ""
    mcast = f", mcast_mask={args.tma_a_mcast_mask}" if args.is_two_cta else ""
    return (
        f"    cute.copy({args.tma_atom_a}, "
        f"{args.tma_gA}[None, {k_offset}], "
        f"{args.tma_sA}[None, {args.tma_producer_state}.index], "
        f"tma_bar_ptr={args.tma_barrier_ptr}{mcast})\n"
    )


def _kloop_tma_copy_b_src(args: _PerKiterTmaArgs, *, k_offset: str) -> str:
    """Per-K-iter TMA copy source for B; ``""`` when B is not TMA-loaded.

    B multicasts on ``cluster_m > 1`` or 2-CTA; A only on 2-CTA, so the
    two helpers are not folded together.
    """
    if not args.use_tma_b:
        return ""
    mcast = (
        f", mcast_mask={args.tma_b_mcast_mask}"
        if args.cluster_m > 1 or args.is_two_cta
        else ""
    )
    return (
        f"    cute.copy({args.tma_atom_b}, "
        f"{args.tma_gB}[None, {k_offset}], "
        f"{args.tma_sB}[None, {args.tma_producer_state}.index], "
        f"tma_bar_ptr={args.tma_barrier_ptr}{mcast})\n"
    )


def _build_kloop_pipeline_producer_if(args: _PerKiterTmaArgs) -> ast.stmt:
    """Per-K-iter TMA producer ``if`` for the pipelined branch.

    The pipelined branch is only entered when both A and B are TMA-
    loaded (``tcgen05_use_tma_pipeline = use_tma_a and use_tma_b``), so
    both ``cute.copy`` emissions must be present; assert that invariant
    rather than silently dropping a side.
    """
    assert args.use_tma_a and args.use_tma_b, (
        "pipelined branch requires both A and B to be TMA-loaded"
    )
    k_offset = f"{args.tma_k_tile} + cutlass.Int32({args.ab_stage_count})"
    src = (
        f"if {args.tma_full_tile} and {args.tma_warp} and {args.tma_next_full_tile}:\n"
        f"    {args.tma_producer_try_token} = "
        f"{args.tma_pipeline}.producer_try_acquire({args.tma_producer_state})\n"
        f"    {args.tma_pipeline}.producer_acquire("
        f"{args.tma_producer_state}, {args.tma_producer_try_token})\n"
        f"    {args.tma_barrier_ptr} = "
        f"{args.tma_pipeline}.producer_get_barrier({args.tma_producer_state})\n"
        + _kloop_tma_copy_a_src(args, k_offset=k_offset)
        + _kloop_tma_copy_b_src(args, k_offset=k_offset)
        + f"    {args.tma_pipeline}.producer_commit({args.tma_producer_state})\n"
        + emit_pipeline_advance(args.tma_producer_state, indent="    ")
    )
    return statement_from_string(src)


def _build_kloop_pipeline_consumer_if(args: _PerKiterTmaArgs) -> ast.stmt:
    """Per-K-iter TMA consumer / scalar-fallback ``if`` for the pipelined branch."""
    scalar_load_a_src = textwrap.indent(ast.unparse(args.scalar_load_a), "    ")
    scalar_load_b_src = textwrap.indent(ast.unparse(args.scalar_load_b), "    ")
    src = (
        f"if {args.tma_full_tile}:\n"
        f"    if {args.exec_active}:\n"
        f"        {args.tma_consumer_try_token} = "
        f"{args.tma_pipeline}.consumer_try_wait({args.tma_consumer_state})\n"
        f"        {args.tma_pipeline}.consumer_wait("
        f"{args.tma_consumer_state}, {args.tma_consumer_try_token})\n"
        "else:\n"
        f"{scalar_load_a_src}\n"
        f"{scalar_load_b_src}\n"
        "    cute.arch.sync_threads()"
    )
    return statement_from_string(src)


def _build_kloop_pipeline_release_if(args: _PerKiterTmaArgs) -> ast.stmt:
    """Per-K-iter consumer release ``if`` for the pipelined branch.

    Producer-state advance lives in the producer block (one per
    commit), so it is intentionally absent here.
    """
    src = (
        f"if {args.tma_full_tile}:\n"
        f"    if {args.exec_active}:\n"
        f"        {args.tma_pipeline}.consumer_release({args.tma_consumer_state})\n"
        + emit_pipeline_advance(args.tma_consumer_state, indent="        ")
        + "\n"
        "else:\n"
        "    cute.arch.sync_threads()"
    )
    return statement_from_string(src)


def _build_kloop_non_pipeline_producer_if(args: _PerKiterTmaArgs) -> ast.stmt:
    """Per-K-iter TMA producer ``if`` for the non-pipelined branch.

    Single AB stage alive at a time: no try-token, no stage-count
    offset on the cute.copy, and no ``advance`` here (the release block
    advances both producer and consumer state).
    """
    src = (
        f"if {args.tma_full_tile} and {args.tma_warp}:\n"
        f"    {args.tma_pipeline}.producer_acquire({args.tma_producer_state})\n"
        f"    {args.tma_barrier_ptr} = "
        f"{args.tma_pipeline}.producer_get_barrier({args.tma_producer_state})\n"
        + _kloop_tma_copy_a_src(args, k_offset=args.tma_k_tile)
        + _kloop_tma_copy_b_src(args, k_offset=args.tma_k_tile)
        + f"    {args.tma_pipeline}.producer_commit({args.tma_producer_state})"
    )
    return statement_from_string(src)


def _build_kloop_non_pipeline_consumer_if(args: _PerKiterTmaArgs) -> ast.stmt:
    """Per-K-iter consumer / scalar-fallback ``if`` for the non-pipelined branch.

    Interleaves scalar fallback loads for any operand NOT TMA-loaded
    into the full-tile branch (e.g. A-TMA + B-scalar still loads B
    here on full tiles).
    """
    scalar_load_a_src = textwrap.indent(ast.unparse(args.scalar_load_a), "    ")
    scalar_load_b_src = textwrap.indent(ast.unparse(args.scalar_load_b), "    ")
    scalar_load_a_tma_src = (
        textwrap.indent(ast.unparse(args.scalar_load_a), "    ") + "\n"
        if not args.use_tma_a
        else ""
    )
    scalar_load_b_tma_src = (
        textwrap.indent(ast.unparse(args.scalar_load_b), "    ") + "\n"
        if not args.use_tma_b
        else ""
    )
    src = (
        f"if {args.tma_full_tile}:\n"
        f"{scalar_load_a_tma_src}"
        f"{scalar_load_b_tma_src}"
        f"    if {args.exec_active}:\n"
        "        cute.arch.sync_warp()\n"
        f"        {args.tma_pipeline}.consumer_wait("
        f"{args.tma_consumer_state}, {args.tma_consumer_try_token})\n"
        "    cute.arch.sync_threads()\n"
        "else:\n"
        f"{scalar_load_a_src}\n"
        f"{scalar_load_b_src}\n"
        "    cute.arch.sync_threads()"
    )
    return statement_from_string(src)


def _build_kloop_non_pipeline_release_if(args: _PerKiterTmaArgs) -> ast.stmt:
    """Per-K-iter consumer release ``if`` for the non-pipelined branch.

    CTA-wide ``sync_threads()`` runs first so every warp sees the
    consumer wait completed; single-stage means BOTH producer and
    consumer state must advance here.
    """
    src = (
        f"if {args.tma_full_tile}:\n"
        "    cute.arch.sync_threads()\n"
        f"    if {args.exec_active}:\n"
        "        cute.arch.sync_warp()\n"
        f"        {args.tma_pipeline}.consumer_release({args.tma_consumer_state})\n"
        + emit_pipeline_advance(args.tma_producer_state, indent="    ")
        + "\n"
        + emit_pipeline_advance(args.tma_consumer_state, indent="    ")
        + "\n"
        "else:\n"
        "    cute.arch.sync_threads()"
    )
    return statement_from_string(src)


@dataclass(frozen=True)
class _InitialPrefetchTmaArgs:
    """Variable names threaded into the initial-prefetch TMA builder.

    The initial prefetch warms stages ``0..ab_stage_count-1`` of the AB
    pipeline at the start of each tile. Only valid on the tcgen05 TMA
    path, which requires both A and B to be TMA-loaded, so both
    ``cute.copy`` emissions are always present.

    All ``str`` fields name a Python identifier or expression in the
    generated code; every name is guaranteed bound at the call site.
    """

    tma_pipeline: str
    tma_producer_state: str
    tma_barrier_ptr: str
    tma_warp: str
    tma_atom_a: str
    tma_atom_b: str
    tma_gA: str
    tma_gB: str
    tma_sA: str
    tma_sB: str
    tma_a_mcast_mask: str
    tma_b_mcast_mask: str
    cluster_m: int
    is_two_cta: bool


def _initial_prefetch_copy_a_src(
    args: _InitialPrefetchTmaArgs, *, k_offset: str
) -> str:
    """Initial-prefetch TMA copy source for A.

    A only multicasts in 2-CTA mode (asymmetric vs. B, which also
    multicasts across cluster CTAs); matches the asymmetry pinned by
    ``test_mcast_mask_asymmetry_between_a_and_b`` for the per-K-iter
    builders.
    """
    mcast = f", mcast_mask={args.tma_a_mcast_mask}" if args.is_two_cta else ""
    return (
        f"    cute.copy({args.tma_atom_a}, "
        f"{args.tma_gA}[None, {k_offset}], "
        f"{args.tma_sA}[None, {args.tma_producer_state}.index], "
        f"tma_bar_ptr={args.tma_barrier_ptr}{mcast})\n"
    )


def _initial_prefetch_copy_b_src(
    args: _InitialPrefetchTmaArgs, *, k_offset: str
) -> str:
    """Initial-prefetch TMA copy source for B.

    B multicasts on ``cluster_m > 1`` or 2-CTA; A only on 2-CTA, so the
    two helpers are not folded together.
    """
    mcast = (
        f", mcast_mask={args.tma_b_mcast_mask}"
        if args.cluster_m > 1 or args.is_two_cta
        else ""
    )
    return (
        f"    cute.copy({args.tma_atom_b}, "
        f"{args.tma_gB}[None, {k_offset}], "
        f"{args.tma_sB}[None, {args.tma_producer_state}.index], "
        f"tma_bar_ptr={args.tma_barrier_ptr}{mcast})\n"
    )


def _build_initial_prefetch_if(
    args: _InitialPrefetchTmaArgs,
    *,
    full_tile_gates: list[str],
    k_offset: str,
) -> ast.stmt:
    """Initial-prefetch ``if`` block for stage ``k_offset``.

    The predicate is ``<full_tile_gates joined with ' and '> and
    {args.tma_warp}``: stage-0 callers pass
    ``[tma_initial_full_tile]``; stage-(N-1) callers (only when
    ``ab_stage_count > 1``) extend with ``tma_initial_next_full_tile``.
    The body performs ``producer_acquire / get_barrier / copy A /
    copy B / producer_commit / advance``. Caller passes a literal
    ``cutlass.Int32(stage_idx)`` for ``k_offset``.
    """
    predicate = " and ".join([*full_tile_gates, args.tma_warp])
    src = (
        f"if {predicate}:\n"
        f"    {args.tma_pipeline}.producer_acquire({args.tma_producer_state})\n"
        f"    {args.tma_barrier_ptr} = "
        f"{args.tma_pipeline}.producer_get_barrier({args.tma_producer_state})\n"
        + _initial_prefetch_copy_a_src(args, k_offset=k_offset)
        + _initial_prefetch_copy_b_src(args, k_offset=k_offset)
        + f"    {args.tma_pipeline}.producer_commit({args.tma_producer_state})\n"
        + emit_pipeline_advance(args.tma_producer_state, indent="    ")
    )
    return statement_from_string(src)


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
    lhs_load, _, lhs_fake = lhs_info
    rhs_load, _, rhs_fake = rhs_info
    if lhs_fake.ndim != 2 or rhs_fake.ndim != 2:
        return None

    df = cg.device_function
    lhs_arg = df.tensor_arg(lhs_fake)
    rhs_arg = df.tensor_arg(rhs_fake)
    lhs_arg_name = lhs_arg.name
    rhs_arg_name = rhs_arg.name

    input_dtype = lhs_fake.dtype
    _dtype_map = {
        torch.float16: "cutlass.Float16",
        torch.bfloat16: "cutlass.BFloat16",
        torch.float32: "cutlass.Float32",
    }
    input_dtype_str = _dtype_map[input_dtype]
    acc_dtype_str = "cutlass.Float32"
    tcgen05_use_tma = (
        input_dtype in (torch.float16, torch.bfloat16)
        and lhs_fake.is_contiguous()
        and rhs_fake.is_contiguous()
    )
    tcgen05_use_tma_a = tcgen05_use_tma
    tcgen05_use_tma_b = tcgen05_use_tma
    tcgen05_use_tma = tcgen05_use_tma_a or tcgen05_use_tma_b
    tcgen05_use_tma_pipeline = tcgen05_use_tma_a and tcgen05_use_tma_b

    k_total_size = int(lhs_fake.shape[1])

    env = CompileEnvironment.current()

    k_loop_info = _get_mma_k_loop_info(cg, env, lhs_fake, rhs_fake, fx_node=fx_node)
    if k_loop_info is None:
        return None
    device_loop, _, k_offset_var, bk = k_loop_info
    if _has_non_root_lane_loops(cg):
        return None
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
        if len(grid_state.block_ids) == 2:
            m_block_id, n_block_id = grid_state.block_ids
            m_offset_var = grid_state.strategy.offset_var(m_block_id)
            n_offset_var = grid_state.strategy.offset_var(n_block_id)
            m_bs = env.block_sizes[m_block_id].from_config(df.config)
            n_bs = env.block_sizes[n_block_id].from_config(df.config)
            bm = int(m_bs) if isinstance(m_bs, int) else None
            bn = int(n_bs) if isinstance(n_bs, int) else None
        else:
            for bid in grid_state.block_ids:
                offset = grid_state.strategy.offset_var(bid)
                bs_info = env.block_sizes[bid]
                size = bs_info.size
                bs = bs_info.from_config(df.config)
                if isinstance(size, (int, torch.SymInt)):
                    if m_offset_var is None and env.known_equal(
                        size, lhs_fake.shape[0]
                    ):
                        m_offset_var = offset
                        m_block_id = bid
                        bm = int(bs) if isinstance(bs, int) else None
                    elif n_offset_var is None and env.known_equal(
                        size, rhs_fake.shape[1]
                    ):
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
    # All tcgen05 widths share the direct TMEM->register->GMEM SIMT epilogue
    # emitted by `_codegen_cute_store_tcgen05_tile` in
    # `helion/language/memory_ops.py`. It uses
    # `cutlass.utils.gemm.sm100.epilogue_tmem_copy_and_partition` for the
    # TMEM->reg copy and a SIMT `CopyUniversalOp` for the reg->GMEM store on
    # the four epi warps, avoiding the SMEM round-trip used by the previous
    # staged path.

    m_index_var = cg.index_var(m_block_id)
    n_index_var = cg.index_var(n_block_id)
    # Use thread_idx directly for local indices within the tile.
    # indices_0 - offset_0 SHOULD equal thread_idx[0], but the CuTe DSL
    # compiler may not simplify the subtraction, leading to illegal memory
    # accesses when partition shapes depend on dynamic values.
    assert grid_state is not None
    m_local = _local_mma_coord_expr(cg, m_block_id)
    n_local = _local_mma_coord_expr(cg, n_block_id)
    n_physical = _physical_mma_coord_expr(cg, n_block_id)
    m_global = f"cutlass.Int32({m_index_var})"
    n_global = f"cutlass.Int32({n_index_var})"
    m_size = int(lhs_fake.shape[0])
    n_size = int(rhs_fake.shape[1])

    mma_impl = _choose_mma_impl(input_dtype, bm=bm, bn=bn, bk=bk, config=df.config)
    zero_acc_expr = acc_expr is not None and _is_zero_acc_expr(acc_expr)
    if (
        not zero_acc_expr
        and acc_expr is not None
        and fx_node is not None
        and fx_node.target is torch.ops.aten.addmm.default
    ):
        acc_node = fx_node.args[0] if fx_node.args else None
        if isinstance(acc_node, Node) and _is_zero_init_acc_node(acc_node):
            zero_acc_expr = True
    if acc_expr is not None and mma_impl != "universal" and not zero_acc_expr:
        mma_impl = "universal"
    if mma_impl != "universal" and zero_acc_expr:
        acc_expr = None
    tcgen05_collective_handles_operand_loads = (
        mma_impl == "tcgen05"
        and fx_node is not None
        and cg.current_grid_state is not None
        and m_size % bm == 0
        and n_size % bn == 0
        and k_total_size % bk == 0
        and len(lhs_load.users) == 1
        and len(rhs_load.users) == 1
        and next(iter(lhs_load.users)) is fx_node
        and next(iter(rhs_load.users)) is fx_node
    )
    if tcgen05_collective_handles_operand_loads:
        df.register_cute_collective_handled_load(lhs_load.name)
        df.register_cute_collective_handled_load(rhs_load.name)
        grid_state = cg.current_grid_state
        assert grid_state is not None
        if grid_state.has_lane_loops():
            df.suppress_cute_root_lane_loops = True

    # Variable names
    tiled_mma = df.new_var("tiled_mma")
    thr_mma = df.new_var("thr_mma")
    acc_frag = df.new_var("acc_frag")
    acc_frag_base = df.new_var("acc_frag_base")
    tcgen05_exec_acc_frag_base = df.new_var("tcgen05_exec_acc_frag_base")
    tcgen05_exec_acc_tmem_ptr = df.new_var("tcgen05_exec_acc_tmem_ptr")
    tcgen05_epi_acc_tmem_ptr = df.new_var("tcgen05_epi_acc_tmem_ptr")
    tcgen05_epi_acc_frag_base = df.new_var("tcgen05_epi_acc_frag_base")
    tcgen05_plan = _new_tcgen05_layout_plan(df) if mma_impl == "tcgen05" else None
    tcgen05_cluster_layout_vmnk = df.new_var("tcgen05_cluster_layout_vmnk")

    # === outer_prefix: MMA setup + shared memory alloc + accumulator init ===
    prefix = device_loop.outer_prefix
    suffix = device_loop.outer_suffix
    # Statements appended to ``prefix`` that reference per-tile coordinates
    # (m_offset_var, n_offset_var, advancing pipeline state). When the
    # persistent kernel splits the device-loop prefix, these stay inside the
    # work-tile loop while everything else hoists out. See
    # ``DeviceFunction.register_cute_tcgen05_per_tile_stmts`` and
    # ``ProgramID._split_tcgen05_invariant_setup``.
    per_tile_stmts: list[ast.AST] = []
    # Statements that conceptually belong to the TMA-load warp's role
    # block (see ``Tcgen05PersistentProgramIDs._collect_tcgen05_role_blocks``).
    # Two kinds of statements get tagged:
    # - The initial TMA prefetch ``producer_acquire`` / ``cute.copy`` /
    #   ``producer_commit`` cycle at the start of each tile. These are
    #   top-level statements added via ``_emit_per_tile(..., tma_load=True)``.
    # - The per-K-iter producer block (``if {tma_full_tile} and {tma_warp}
    #   [and {tma_next_full_tile}]: producer_acquire / cute.copy /
    #   producer_commit``). These are emitted INSIDE the K-loop body via
    #   ``cg.add_statement`` and added to the list directly. The role
    #   partitioner recurses one level into top-level ``for`` / ``while``
    #   loops to find these tagged children and wraps them with
    #   ``if {tma_warp_predicate}: ...`` -- structural prep for the
    #   role-local-while lift in ``cute_plan.md`` step 3b.
    tma_load_role_stmts: list[ast.AST] = []

    def _emit_per_tile(text: str, *, tma_load: bool = False) -> ast.stmt:
        """Append a per-tile statement to ``prefix`` and tag it for the
        persistent-loop splitter. Returns the AST node so callers can
        chain (e.g. when constructing ``If`` bodies). When ``tma_load``
        is true the statement is ALSO tagged for the role-block
        partitioner so it lands in the TMA-load warp's role block when
        the persistent kernel emits warp-role-gated bodies.
        """
        stmt = statement_from_string(text)
        prefix.append(stmt)
        per_tile_stmts.append(stmt)
        if tma_load:
            tma_load_role_stmts.append(stmt)
        return stmt

    mma_participant_linear: str | None = None
    mma_copy_linear: str | None = None
    mma_active: str | None = None
    tma_warp: str | None = None
    warp_idx: str | None = None
    lane_idx: str | None = None
    epi_active: str | None = None
    epi_tidx: str | None = None
    mma_phys_n = _mma_active_n_threads(mma_impl)
    mma_physical_m_threads = _grid_thread_extent(cg, m_block_id)
    tcgen05_cta_thread_count = _grid_cta_thread_count(cg)
    tcgen05_cluster_m = _tcgen05_cluster_m(df.config)
    # cluster_m == 2 currently requires CTA-local M tiles >= 128. Smaller
    # M tiles (e.g. bm=64) miscompile with cluster_m=2, so transparently
    # demote to a single-CTA cluster. Also demote when we cannot statically
    # confirm the launch grid has enough M-CTAs to satisfy the cluster
    # shape -- CUDA rejects clusters with fewer CTAs than the cluster size,
    # so symbolic M dimensions take the safe path too.
    #
    # The matmul autotune search also narrows to cluster_m=1 because
    # runtime CUDA "unspecified launch failure" reproduces on the
    # 1-CTA clustered path even when the demotion below would not
    # trigger. The narrowing is the user-facing safety net; the
    # demotion below is the codegen-side fallback that catches cases
    # the narrowing missed (e.g. explicit Configs from ``set_config``
    # with cluster_m=2 + bm=64).
    if tcgen05_cluster_m > 1:
        m_total = lhs_fake.shape[0]
        if (
            bm < 128
            or not isinstance(m_total, int)
            or m_total // bm < tcgen05_cluster_m
        ):
            tcgen05_cluster_m = 1
    assert tcgen05_cluster_m == 1 or bm >= 128
    tcgen05_is_two_cta = _tcgen05_use_2cta_instrs(bm=bm, cluster_m=tcgen05_cluster_m)
    if mma_impl == "tcgen05" and tcgen05_cluster_m > 1:
        df.cute_cluster_shape = (tcgen05_cluster_m, 1, 1)
    tcgen05_acc_stage_count_value = _tcgen05_config_int(
        df.config, "tcgen05_acc_stages", _tcgen05_acc_stage_count(bn)
    )
    tcgen05_ab_stage_count_value = _tcgen05_config_int(
        df.config, "tcgen05_ab_stages", _tcgen05_ab_stage_count(df.config.num_stages)
    )
    tcgen05_c_stage_count_value = _tcgen05_config_int(
        df.config, "tcgen05_c_stages", _tcgen05_c_stage_count(bn)
    )
    tcgen05_epi_warp_count_value = _tcgen05_epi_warp_count(
        df.config, cta_thread_count=tcgen05_cta_thread_count
    )
    tcgen05_tmem_barrier_thread_count_value = _tcgen05_tmem_barrier_thread_count(
        tcgen05_epi_warp_count_value
    )
    tcgen05_matmul_plan: CuteTcgen05MatmulPlan | None = None
    if mma_impl == "tcgen05":
        tcgen05_matmul_plan = CuteTcgen05MatmulPlan(
            bm=bm,
            bn=bn,
            bk=bk,
            cluster_m=tcgen05_cluster_m,
            is_two_cta=tcgen05_is_two_cta,
            cta_thread_count=tcgen05_cta_thread_count,
            physical_m_threads=mma_physical_m_threads,
            acc_stage_count=tcgen05_acc_stage_count_value,
            ab_stage_count=tcgen05_ab_stage_count_value,
            c_stage_count=tcgen05_c_stage_count_value,
            epi_warp_count=tcgen05_epi_warp_count_value,
        )
        candidate_block_shape = tcgen05_matmul_plan.block_shape
        df.register_cute_tcgen05_matmul_plan(tcgen05_matmul_plan)
        if (
            candidate_block_shape[0]
            * candidate_block_shape[1]
            * candidate_block_shape[2]
            > 1024
        ):
            raise exc.BackendUnsupported(
                "cute",
                f"tcgen05 launch block shape {candidate_block_shape} exceeds 1024 threads",
            )
        df.cute_block_shape = candidate_block_shape
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
                tcgen05_cluster_m=tcgen05_cluster_m,
            )
        )
    else:
        mma_participant_linear = df.new_var("mma_tidx")
        mma_slice_linear = df.new_var("mma_slice_tidx")
        mma_copy_linear = df.new_var("mma_copy_tidx")
        mma_active = df.new_var("mma_active")
        tma_warp = df.new_var("tcgen05_tma_warp")
        warp_idx = df.new_var("tcgen05_warp_idx")
        lane_idx = df.new_var("tcgen05_lane_idx")
        epi_active = df.new_var("tcgen05_epi_active")
        epi_tidx = df.new_var("tcgen05_epi_tidx")
        prefix.append(
            statement_from_string(
                f"{warp_idx} = cute.arch.make_warp_uniform(cute.arch.warp_idx())"
            )
        )
        prefix.append(statement_from_string(f"{lane_idx} = cute.arch.lane_idx()"))
        prefix.append(
            statement_from_string(
                f"{mma_participant_linear} = "
                f"{_physical_mma_coord_expr(cg, m_block_id)} + "
                f"({n_physical}) * cutlass.Int32({mma_physical_m_threads})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{mma_copy_linear} = "
                + (
                    mma_participant_linear
                    if tcgen05_collective_handles_operand_loads
                    else f"{m_local} + ({n_local}) * cutlass.Int32({bm})"
                )
            )
        )
        prefix.append(
            statement_from_string(
                f"{mma_active} = ({n_physical}) < cutlass.Int32({mma_phys_n})"
            )
        )
        if mma_impl == "tcgen05":
            assert tcgen05_plan is not None
            assert tcgen05_matmul_plan is not None
            # The current lowering has a single A/B load warp at
            # `tma_warp_id`, so `tma_warp` doubles as the A/B-load-active
            # predicate. When role-local persistent loops land and split
            # those roles, this is the place to add a separate
            # `tcgen05_ab_load_active` predicate.
            prefix.append(
                statement_from_string(
                    f"{tma_warp} = {warp_idx} == cutlass.Int32({tcgen05_matmul_plan.tma_warp_id})"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{tcgen05_plan.exec_active} = "
                    f"{warp_idx} == cutlass.Int32({tcgen05_matmul_plan.exec_warp_id})"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{epi_active} = "
                    f"{warp_idx} < cutlass.Int32({tcgen05_matmul_plan.epi_warp_count})"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{epi_tidx} = {lane_idx} + {warp_idx} * cutlass.Int32(32)"
                    f" if {epi_active} else cutlass.Int32(0)"
                )
            )
            # Register reallocation: consumer warps (exec MMA + epilogue
            # warps) request a larger per-thread register budget; the
            # producer warps (TMA load, A/B load) drop to the producer
            # budget. The "not consumer" form is just a more compact
            # spelling of "tma_warp or ab_load_active"; both are
            # equivalent now that the launched CTA has no idle padding
            # warps. Matches Quack's sm100 split. The setmaxregister
            # calls are warp-uniform and must precede the first pipeline
            # op of each role; placing them with the role-gate invariants
            # keeps them out of the per-tile work loop.
            consumer_predicate = f"{tcgen05_plan.exec_active} or {epi_active}"
            prefix.append(
                statement_from_string(
                    f"if not ({consumer_predicate}):\n"
                    f"    cute.arch.setmaxregister_decrease("
                    f"cutlass.Int32({_TCGEN05_PRODUCER_REGS}))"
                )
            )
            prefix.append(
                statement_from_string(
                    f"if {consumer_predicate}:\n"
                    f"    cute.arch.setmaxregister_increase("
                    f"cutlass.Int32({_TCGEN05_CONSUMER_REGS}))"
                )
            )
            prefix.append(
                statement_from_string(
                    # tcgen05 tiled_mma slicing is CTA-scoped, not per-thread.
                    # Quack/CUTLASS use the CTA's MMA tile coordinate here.
                    # On Helion's 1-CTA path that is always 0, but clustered
                    # widened kernels need the cluster-local CTA rank so each
                    # CTA takes the right MMA slice.
                    f"{mma_slice_linear} = "
                    + (
                        "cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) "
                        f"% cutlass.Int32({tcgen05_cluster_m})"
                        if tcgen05_cluster_m > 1
                        else "cutlass.Int32(0)"
                    )
                )
            )
            prefix.extend(
                _make_tiled_mma_setup(
                    mma_impl,
                    tiled_mma,
                    thr_mma,
                    mma_slice_linear,
                    input_dtype_str,
                    acc_dtype_str,
                    bm,
                    bn,
                    tcgen05_cluster_m=tcgen05_cluster_m,
                )
            )
        else:
            prefix.append(
                statement_from_string(f"{tma_warp} = {warp_idx} == cutlass.Int32(0)")
            )
            prefix.extend(
                _make_tiled_mma_setup(
                    mma_impl,
                    tiled_mma,
                    thr_mma,
                    mma_participant_linear,
                    input_dtype_str,
                    acc_dtype_str,
                    bm,
                    bn,
                    tcgen05_cluster_m=tcgen05_cluster_m,
                )
            )
    if mma_impl == "tcgen05":
        assert tcgen05_plan is not None
        prefix.append(
            statement_from_string(
                f"{tcgen05_cluster_layout_vmnk} = cute.tiled_divide("
                f"cute.make_layout(({tcgen05_cluster_m}, 1, 1)), "
                f"({tiled_mma}.thr_id.shape,))"
            )
        )
        prefix.extend(
            _make_tcgen05_layout_plan_setup(
                tcgen05_plan,
                tiled_mma,
                bm=bm,
                bn=bn,
                bk=bk,
                ab_stage_count=tcgen05_ab_stage_count_value,
                is_two_cta=tcgen05_is_two_cta,
                input_dtype_str=input_dtype_str,
                acc_dtype_str=acc_dtype_str,
            )
        )
        prefix.append(
            statement_from_string(
                f"{acc_frag_base} = {tiled_mma}.make_fragment_C("
                f"cute.append({tiled_mma}.partition_shape_C(({bm}, {bn})), "
                f"{tcgen05_acc_stage_count_value}))"
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
                f"{tcgen05_plan.tmem_dealloc_mbar_ptr} = cute.arch.alloc_smem(cutlass.Int64, 1)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.tmem_alloc_barrier} = cutlass.pipeline.NamedBarrier("
                f"barrier_id={_TCGEN05_TMEM_ALLOC_BARRIER_ID}, "
                f"num_threads={tcgen05_tmem_barrier_thread_count_value})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.tmem_allocator} = cutlass.utils.TmemAllocator("
                f"{tcgen05_plan.tmem_holding_buf}, "
                f"barrier_for_retrieve={tcgen05_plan.tmem_alloc_barrier}, "
                f"allocator_warp_id=0, is_two_cta={tcgen05_is_two_cta!s}, "
                f"two_cta_tmem_dealloc_mbar_ptr={tcgen05_plan.tmem_dealloc_mbar_ptr})"
            )
        )
        if tcgen05_cluster_m > 1:
            prefix.append(
                statement_from_string(
                    "cutlass.pipeline.pipeline_init_arrive("
                    f"cluster_shape_mn={tcgen05_cluster_layout_vmnk}, "
                    "is_relaxed=True)"
                )
            )
            prefix.append(
                statement_from_string(
                    "cutlass.pipeline.pipeline_init_wait("
                    f"cluster_shape_mn={tcgen05_cluster_layout_vmnk})"
                )
            )
        prefix.append(
            statement_from_string(
                f"if {epi_active}:\n"
                f"    {tcgen05_plan.tmem_allocator}.allocate({tcgen05_plan.acc_tmem_cols})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_exec_acc_tmem_ptr} = cute.make_ptr("
                f"{acc_dtype_str}, 0, cute.AddressSpace.tmem, assumed_align=16)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_epi_acc_tmem_ptr} = cute.make_ptr("
                f"{acc_dtype_str}, 0, cute.AddressSpace.tmem, assumed_align=16)"
            )
        )
        # ``acc_frag`` is reassigned per-tile below to a stage-indexed
        # slice; an extra ``acc_frag = acc_frag_base`` here would land
        # in the hoisted setup with a different CuTe type and break the
        # persistent ``while`` ("acc_frag is structured different after
        # this while").
        prefix.append(
            statement_from_string(
                f"if {tcgen05_plan.exec_active}:\n"
                f"    {tcgen05_plan.tmem_allocator}.wait_for_alloc()\n"
                f"    {tcgen05_exec_acc_tmem_ptr} = "
                f"{tcgen05_plan.tmem_allocator}.retrieve_ptr({acc_dtype_str})"
            )
        )
        prefix.append(
            statement_from_string(
                f"if {epi_active}:\n"
                f"    {tcgen05_plan.tmem_allocator}.wait_for_alloc()\n"
                f"    {tcgen05_epi_acc_tmem_ptr} = "
                f"{tcgen05_plan.tmem_allocator}.retrieve_ptr({acc_dtype_str})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_pipeline_barriers} = cute.arch.alloc_smem("
                f"cutlass.Int64, cutlass.Int32({tcgen05_acc_stage_count_value * 2}))"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_pipeline_producer_group} = "
                "cutlass.pipeline.CooperativeGroup("
                "cutlass.pipeline.Agent.Thread)"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_pipeline_consumer_group} = "
                f"cutlass.pipeline.CooperativeGroup("
                f"cutlass.pipeline.Agent.Thread, cutlass.Int32({tcgen05_epi_warp_count_value}))"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_pipeline} = cutlass.pipeline.PipelineUmmaAsync.create("
                f"num_stages={tcgen05_acc_stage_count_value}, "
                f"producer_group={tcgen05_plan.acc_pipeline_producer_group}, "
                f"consumer_group={tcgen05_plan.acc_pipeline_consumer_group}, "
                f"barrier_storage={tcgen05_plan.acc_pipeline_barriers}, "
                f"cta_layout_vmnk={tcgen05_cluster_layout_vmnk})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_producer_state} = cutlass.pipeline.make_pipeline_state("
                f"cutlass.pipeline.PipelineUserType.Producer, {tcgen05_acc_stage_count_value})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_plan.acc_consumer_state} = cutlass.pipeline.make_pipeline_state("
                f"cutlass.pipeline.PipelineUserType.Consumer, {tcgen05_acc_stage_count_value})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_exec_acc_frag_base} = cute.make_tensor("
                f"{tcgen05_exec_acc_tmem_ptr}, {acc_frag_base}.layout)"
            )
        )
        # ``acc_frag`` indexes ``tcgen05_exec_acc_frag_base`` by the current
        # ``acc_producer_state.index`` stage. The K-loop suffix advances
        # that producer state once per UMMA fence, so under the persistent
        # path each tile sees a different index. Mark per-tile so the
        # alias is recomputed inside the work-tile loop.
        _emit_per_tile(
            f"{acc_frag} = "
            f"{tcgen05_exec_acc_frag_base}[None, None, None, "
            f"{tcgen05_plan.acc_producer_state}.index]"
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_epi_acc_frag_base} = cute.make_tensor("
                f"{tcgen05_epi_acc_tmem_ptr}, {acc_frag_base}.layout)"
            )
        )
        # Initial producer_acquire for stage 0 of the acc pipeline. The
        # ``acc_producer_state`` advances once per UMMA fence inside the
        # K-loop, so per tile we want to start by acquiring whatever stage
        # the persistent loop currently points at. Tag as per-tile so this
        # acquire stays in the work-tile body when the persistent loop
        # splitter runs.
        _emit_per_tile(
            f"if {tcgen05_plan.exec_active}:\n"
            f"    {tcgen05_plan.acc_pipeline}.producer_acquire("
            f"{tcgen05_plan.acc_producer_state})"
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
    tma_smem_a_layout = df.new_var("sA_tma_layout")
    tma_smem_b_layout = df.new_var("sB_tma_layout")
    tma_thr_mma = df.new_var("tma_thr_mma")
    gmem_a_tma = df.new_var("gA_tma")
    gmem_b_tma = df.new_var("gB_tma")
    gmem_a_tma_part = df.new_var("gA_tma_part")
    gmem_b_tma_part = df.new_var("gB_tma_part")
    tma_atom_a = df.new_var("tma_atom_a")
    tma_atom_b = df.new_var("tma_atom_b")
    tma_tensor_a = df.new_var("tma_tensor_a")
    tma_tensor_b = df.new_var("tma_tensor_b")
    tma_cta_layout = df.new_var("tma_cta_layout")
    tma_gA = df.new_var("tma_gA")
    tma_sA = df.new_var("tma_sA")
    tma_gB = df.new_var("tma_gB")
    tma_sB = df.new_var("tma_sB")
    tma_initial_full_tile = df.new_var("tcgen05_tma_initial_full_tile")
    tma_initial_next_full_tile = df.new_var("tcgen05_tma_initial_next_full_tile")
    tma_full_tile = df.new_var("tcgen05_tma_full_tile")
    tma_next_full_tile = df.new_var("tcgen05_tma_next_full_tile")
    tma_k_tile = df.new_var("tcgen05_tma_k_tile")
    tma_barrier_ptr = df.new_var("tcgen05_tma_barrier")
    tma_producer_try_token = df.new_var("tcgen05_ab_producer_try_token")
    tma_consumer_try_token = df.new_var("tcgen05_ab_consumer_try_token")
    tma_cta_rank_in_cluster = df.new_var("tcgen05_cta_rank_in_cluster")
    tma_block_in_cluster_coord_vmnk = df.new_var("tcgen05_block_in_cluster_coord_vmnk")
    tma_a_mcast_mask = df.new_var("tcgen05_a_mcast_mask")
    tma_b_mcast_mask = df.new_var("tcgen05_b_mcast_mask")
    tma_pipeline_mbars = df.new_var("tcgen05_ab_pipeline_mbars")
    tma_pipeline_producer_group = df.new_var("tcgen05_ab_pipeline_producer_group")
    tma_pipeline_consumer_group = df.new_var("tcgen05_ab_pipeline_consumer_group")
    tma_pipeline = df.new_var("tcgen05_ab_pipeline")
    tma_producer_state = df.new_var("tcgen05_ab_producer_state")
    tma_consumer_state = df.new_var("tcgen05_ab_consumer_state")
    tcgen05_frag_a = df.new_var("tcgen05_tCrA")
    tcgen05_frag_b = df.new_var("tcgen05_tCrB")
    mma_stage = df.new_var("mma_stage")
    if mma_impl == "tcgen05":
        assert tcgen05_plan is not None
        if tcgen05_use_tma:
            df.wrapper_only_params.extend(
                [tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b]
            )
            cg.cute_wrapper_plans.append(
                {
                    "kind": "tcgen05_ab_tma",
                    "lhs_name": lhs_arg_name,
                    "rhs_name": rhs_arg_name,
                    "bm": bm,
                    "bn": bn,
                    "bk": bk,
                    "cluster_m": tcgen05_cluster_m,
                    "cluster_n": 1,
                    "ab_stage_count": tcgen05_ab_stage_count_value,
                    "input_dtype": input_dtype_str,
                    "acc_dtype": acc_dtype_str,
                    "kernel_args": [tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b],
                }
            )
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
        prefix.append(
            statement_from_string(
                f"{tcgen05_frag_a} = {tiled_mma}.make_fragment_A({smem_a})"
            )
        )
        prefix.append(
            statement_from_string(
                f"{tcgen05_frag_b} = {tiled_mma}.make_fragment_B({smem_b})"
            )
        )
        if tcgen05_use_tma:
            prefix.append(
                statement_from_string(
                    f"{tma_smem_a_layout} = cute.slice_({tcgen05_plan.smem_a_layout}, (None, None, None, 0))"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{tma_smem_b_layout} = cute.slice_({tcgen05_plan.smem_b_layout}, (None, None, None, 0))"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{tma_thr_mma} = {tiled_mma}.get_slice(cutlass.Int32(0))"
                )
            )
            # gA, gB depend on per-tile (m_offset_var, n_offset_var). Their
            # downstream partitions and tma_partition outputs all inherit
            # that per-tile dependency, so all of these stay inside the
            # work-tile body when the persistent loop splitter runs.
            _emit_per_tile(
                f"{gmem_a_tma} = cute.local_tile("
                f"{tma_tensor_a}, ({bm}, {bk}), "
                f"({m_offset_var} // cutlass.Int32({bm}), None))"
            )
            _emit_per_tile(
                f"{gmem_b_tma} = cute.local_tile("
                f"{tma_tensor_b}, ({bn}, {bk}), "
                f"({n_offset_var} // cutlass.Int32({bn}), None))"
            )
            _emit_per_tile(
                f"{gmem_a_tma_part} = {tma_thr_mma}.partition_A({gmem_a_tma})"
            )
            _emit_per_tile(
                f"{gmem_b_tma_part} = {tma_thr_mma}.partition_B({gmem_b_tma})"
            )
            prefix.append(
                statement_from_string(f"{tma_cta_layout} = cute.make_layout(1)")
            )
            if tcgen05_cluster_m > 1 or tcgen05_is_two_cta:
                prefix.append(
                    statement_from_string(
                        f"{tma_cta_rank_in_cluster} = cute.arch.make_warp_uniform("
                        "cute.arch.block_idx_in_cluster())"
                    )
                )
                prefix.append(
                    statement_from_string(
                        f"{tma_block_in_cluster_coord_vmnk} = "
                        f"{tcgen05_cluster_layout_vmnk}.get_flat_coord({tma_cta_rank_in_cluster})"
                    )
                )
                if tcgen05_is_two_cta:
                    prefix.append(
                        statement_from_string(
                            f"{tma_a_mcast_mask} = cute.nvgpu.cpasync.create_tma_multicast_mask("
                            f"{tcgen05_cluster_layout_vmnk}, {tma_block_in_cluster_coord_vmnk}, "
                            "mcast_mode=2)"
                        )
                    )
                if tcgen05_cluster_m > 1 or tcgen05_is_two_cta:
                    prefix.append(
                        statement_from_string(
                            f"{tma_b_mcast_mask} = cute.nvgpu.cpasync.create_tma_multicast_mask("
                            f"{tcgen05_cluster_layout_vmnk}, {tma_block_in_cluster_coord_vmnk}, "
                            "mcast_mode=1)"
                        )
                    )
            # tma_partition consumes the per-tile gA_part / gB_part, so the
            # resulting (tma_sA, tma_gA) / (tma_sB, tma_gB) are also per-tile.
            _emit_per_tile(
                f"{tma_sA}, {tma_gA} = cute.nvgpu.cpasync.tma_partition("
                f"{tma_atom_a}, 0, {tma_cta_layout}, "
                f"cute.group_modes({smem_a}, 0, cute.rank({smem_a}) - 1), "
                f"cute.group_modes({gmem_a_tma_part}, 0, "
                f"cute.rank({gmem_a_tma_part}) - 1))"
            )
            _emit_per_tile(
                f"{tma_sB}, {tma_gB} = cute.nvgpu.cpasync.tma_partition("
                f"{tma_atom_b}, 0, {tma_cta_layout}, "
                f"cute.group_modes({smem_b}, 0, cute.rank({smem_b}) - 1), "
                f"cute.group_modes({gmem_b_tma_part}, 0, "
                f"cute.rank({gmem_b_tma_part}) - 1))"
            )
            prefix.append(
                statement_from_string(
                    f"{tma_pipeline_mbars} = cute.arch.alloc_smem("
                    f"cutlass.Int64, cutlass.Int32({tcgen05_ab_stage_count_value}))"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{tma_pipeline_producer_group} = cutlass.pipeline.CooperativeGroup("
                    "cutlass.pipeline.Agent.Thread, 1)"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{tma_pipeline_consumer_group} = cutlass.pipeline.CooperativeGroup("
                    "cutlass.pipeline.Agent.Thread, 1)"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{tma_pipeline} = cutlass.pipeline.PipelineTmaUmma.create("
                    f"num_stages={tcgen05_ab_stage_count_value}, "
                    f"producer_group={tma_pipeline_producer_group}, "
                    f"consumer_group={tma_pipeline_consumer_group}, "
                    "tx_count="
                    f"{'cute.size_in_bytes(' + input_dtype_str + ', ' + tma_smem_a_layout + ')' if tcgen05_use_tma_a else '0'} + "
                    f"{'cute.size_in_bytes(' + input_dtype_str + ', ' + tma_smem_b_layout + ')' if tcgen05_use_tma_b else '0'}, "
                    f"barrier_storage={tma_pipeline_mbars}, "
                    f"cta_layout_vmnk={tcgen05_cluster_layout_vmnk})"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{tma_producer_state} = cutlass.pipeline.make_pipeline_state("
                    f"cutlass.pipeline.PipelineUserType.Producer, {tcgen05_ab_stage_count_value})"
                )
            )
            prefix.append(
                statement_from_string(
                    f"{tma_consumer_state} = cutlass.pipeline.make_pipeline_state("
                    f"cutlass.pipeline.PipelineUserType.Consumer, {tcgen05_ab_stage_count_value})"
                )
            )
            if tcgen05_use_tma_pipeline:
                # Initial TMA prefetch warms stages 0..ab_stage_count-1 of the
                # AB pipeline at the START of each tile. Both the boolean
                # full-tile predicates and the TMA copies reference per-tile
                # gA/gB tensors and m_offset/n_offset, so they must stay in
                # the work-tile body.
                #
                # The full-tile predicate booleans (``tma_initial_full_tile``,
                # ``tma_initial_next_full_tile``) are read by every warp later
                # for cross-role gating, so they live in the shared role
                # block. The producer_acquire/cute.copy/producer_commit
                # cycles below are exclusively TMA-load work (they are
                # already gated by ``if {tma_warp}:`` inline) and therefore
                # also tagged with ``tma_load=True`` so the role
                # partitioner can route them into the TMA-load block.
                assert tma_warp is not None
                prefetch_args = _InitialPrefetchTmaArgs(
                    tma_pipeline=tma_pipeline,
                    tma_producer_state=tma_producer_state,
                    tma_barrier_ptr=tma_barrier_ptr,
                    tma_warp=tma_warp,
                    tma_atom_a=tma_atom_a,
                    tma_atom_b=tma_atom_b,
                    tma_gA=tma_gA,
                    tma_gB=tma_gB,
                    tma_sA=tma_sA,
                    tma_sB=tma_sB,
                    tma_a_mcast_mask=tma_a_mcast_mask,
                    tma_b_mcast_mask=tma_b_mcast_mask,
                    cluster_m=tcgen05_cluster_m,
                    is_two_cta=tcgen05_is_two_cta,
                )
                _emit_per_tile(
                    f"{tma_initial_full_tile} = "
                    f"{m_offset_var} + cutlass.Int32({bm}) <= cutlass.Int32({m_size}) "
                    f"and {n_offset_var} + cutlass.Int32({bn}) <= cutlass.Int32({n_size}) "
                    f"and cutlass.Int32({bk}) <= cutlass.Int32({k_total_size})"
                )
                stage0_prefetch = _build_initial_prefetch_if(
                    prefetch_args,
                    full_tile_gates=[tma_initial_full_tile],
                    k_offset="cutlass.Int32(0)",
                )
                prefix.append(stage0_prefetch)
                per_tile_stmts.append(stage0_prefetch)
                tma_load_role_stmts.append(stage0_prefetch)
                if tcgen05_ab_stage_count_value > 1:
                    _emit_per_tile(
                        f"{tma_initial_next_full_tile} = "
                        f"{m_offset_var} + cutlass.Int32({bm}) <= cutlass.Int32({m_size}) "
                        f"and {n_offset_var} + cutlass.Int32({bn}) <= cutlass.Int32({n_size}) "
                        f"and cutlass.Int32({bk * tcgen05_ab_stage_count_value}) <= cutlass.Int32({k_total_size})"
                    )
                    stage_n_prefetch = _build_initial_prefetch_if(
                        prefetch_args,
                        full_tile_gates=[
                            tma_initial_full_tile,
                            tma_initial_next_full_tile,
                        ],
                        k_offset=f"cutlass.Int32({tcgen05_ab_stage_count_value - 1})",
                    )
                    prefix.append(stage_n_prefetch)
                    per_tile_stmts.append(stage_n_prefetch)
                    tma_load_role_stmts.append(stage_n_prefetch)
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
    # Built once below in the tcgen05+TMA branch; reused by the
    # release block emitted later in the same branch.
    tma_kloop_args: _PerKiterTmaArgs | None = None

    # --- Global → Shared memory with masking ---
    # Each thread loads elements into shared memory using scalar indexing
    # with bounds checking for non-divisible tile boundaries.
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
        cg.add_statement(statement_from_string("cute.arch.sync_threads()"))
    else:
        active_threads = bm * mma_phys_n
        assert (
            mma_active is not None
            and mma_participant_linear is not None
            and mma_copy_linear is not None
        )
        load_thread_count = active_threads
        load_guard = mma_active
        if mma_impl == "tcgen05":
            assert tcgen05_plan is not None
            # The smem cache for A/B is laid out as (..., ab_stage_count); we
            # index into the current stage every K-loop iteration.
            #
            # When the kernel uses the TMA pipeline, ``tma_consumer_state.index``
            # is the canonical stage index: it advances exactly once per K-loop
            # iteration via ``consumer_release`` + ``advance``, and (critically
            # for the persistent path) it carries its value across virtual
            # tiles. Computing ``mma_stage`` from ``k_offset // bk`` resets to
            # zero at each tile while the pipeline state stays where it was at
            # the end of the prior tile -- the two diverge across persistent
            # tile boundaries. Use the pipeline state directly so the K-loop
            # always reads the stage the consumer just unblocked.
            #
            # For the non-TMA tcgen05 path there is no pipeline state to track
            # and ``ab_stage_count`` is always 1, so the modular form is a
            # constant zero anyway.
            if tcgen05_use_tma:
                cg.add_statement(
                    statement_from_string(f"{mma_stage} = {tma_consumer_state}.index")
                )
            else:
                cg.add_statement(
                    statement_from_string(
                        f"{mma_stage} = "
                        f"({k_offset_var} // cutlass.Int32({bk})) "
                        f"% cutlass.Int32({tcgen05_ab_stage_count_value})"
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
            if tcgen05_use_tma:
                cg.add_statement(
                    statement_from_string(
                        f"{tma_k_tile} = {k_offset_var} // cutlass.Int32({bk})"
                    )
                )
                cg.add_statement(
                    statement_from_string(
                        f"{tma_full_tile} = "
                        f"{m_offset_var} + cutlass.Int32({bm}) <= cutlass.Int32({m_size}) "
                        f"and {n_offset_var} + cutlass.Int32({bn}) <= cutlass.Int32({n_size}) "
                        f"and {k_offset_var} + cutlass.Int32({bk}) <= cutlass.Int32({k_total_size})"
                    )
                )
        smem_a_store = f"{smem_a}[_row, _col]"
        smem_b_store = f"{smem_b}[_row, _col]"
        if mma_impl == "tcgen05":
            smem_a_store = f"{smem_a_mma}[((_row, _col),)]"
            smem_b_store = f"{smem_b_mma}[((_row, _col),)]"
        scalar_load_a = statement_from_string(
            f"if {load_guard}:\n"
            f"    for _load_i in range(({bm * bk} + {load_thread_count} - 1) // {load_thread_count}):\n"
            f"        _flat = {mma_copy_linear} + cutlass.Int32(_load_i) * cutlass.Int32({load_thread_count})\n"
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
        scalar_load_b = statement_from_string(
            f"if {load_guard}:\n"
            f"    for _load_i in range(({bn * bk} + {load_thread_count} - 1) // {load_thread_count}):\n"
            f"        _flat = {mma_copy_linear} + cutlass.Int32(_load_i) * cutlass.Int32({load_thread_count})\n"
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
        if mma_impl == "tcgen05" and tcgen05_use_tma:
            assert tcgen05_plan is not None
            assert tma_warp is not None
            tma_kloop_args = _PerKiterTmaArgs(
                tma_pipeline=tma_pipeline,
                tma_producer_state=tma_producer_state,
                tma_consumer_state=tma_consumer_state,
                tma_producer_try_token=tma_producer_try_token,
                tma_consumer_try_token=tma_consumer_try_token,
                tma_barrier_ptr=tma_barrier_ptr,
                tma_full_tile=tma_full_tile,
                tma_next_full_tile=tma_next_full_tile,
                tma_warp=tma_warp,
                tma_atom_a=tma_atom_a,
                tma_atom_b=tma_atom_b,
                tma_gA=tma_gA,
                tma_gB=tma_gB,
                tma_sA=tma_sA,
                tma_sB=tma_sB,
                tma_k_tile=tma_k_tile,
                tma_a_mcast_mask=tma_a_mcast_mask,
                tma_b_mcast_mask=tma_b_mcast_mask,
                ab_stage_count=tcgen05_ab_stage_count_value,
                cluster_m=tcgen05_cluster_m,
                is_two_cta=tcgen05_is_two_cta,
                use_tma_a=tcgen05_use_tma_a,
                use_tma_b=tcgen05_use_tma_b,
                exec_active=tcgen05_plan.exec_active,
                scalar_load_a=scalar_load_a,
                scalar_load_b=scalar_load_b,
            )
            if tcgen05_use_tma_pipeline:
                cg.add_statement(
                    statement_from_string(
                        f"{tma_next_full_tile} = "
                        f"{m_offset_var} + cutlass.Int32({bm}) <= cutlass.Int32({m_size}) "
                        f"and {n_offset_var} + cutlass.Int32({bn}) <= cutlass.Int32({n_size}) "
                        f"and {k_offset_var} + cutlass.Int32({bk * (tcgen05_ab_stage_count_value + 1)}) <= cutlass.Int32({k_total_size})"
                    )
                )
                cg.add_statement(
                    statement_from_string(
                        f"{tma_producer_try_token} = cutlass.Boolean(0)"
                    )
                )
                cg.add_statement(
                    statement_from_string(
                        f"{tma_consumer_try_token} = cutlass.Boolean(0)"
                    )
                )
                # Producer is tagged for the role partitioner so it gets
                # wrapped in ``if {tma_warp_predicate}: ...`` when the
                # K-loop is split per-role.
                pipeline_producer_stmt = _build_kloop_pipeline_producer_if(
                    tma_kloop_args
                )
                cg.add_statement(pipeline_producer_stmt)
                tma_load_role_stmts.append(pipeline_producer_stmt)
                cg.add_statement(_build_kloop_pipeline_consumer_if(tma_kloop_args))
            else:
                non_pipeline_producer_stmt = _build_kloop_non_pipeline_producer_if(
                    tma_kloop_args
                )
                cg.add_statement(non_pipeline_producer_stmt)
                tma_load_role_stmts.append(non_pipeline_producer_stmt)
                cg.add_statement(_build_kloop_non_pipeline_consumer_if(tma_kloop_args))
        else:
            cg.add_statement(scalar_load_a)
            cg.add_statement(scalar_load_b)
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
                f"cute.gemm({tiled_mma}, {acc_frag}, [{rA}], [{rB}], {acc_frag})"
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
                    f"    cute.gemm({tiled_mma}, {acc_frag}, [{rA}], [{rB}], {acc_frag})"
                )
            )
        else:
            assert tcgen05_plan is not None
            cg.add_statement(
                statement_from_string(
                    f"if {tcgen05_plan.exec_active}:\n    cute.arch.fence_view_async_shared()"
                )
            )
            cg.add_statement(
                statement_from_string(
                    f"if {tcgen05_plan.exec_active}:\n"
                    f"    for _tcgen05_kblk_idx in range(cute.size({tcgen05_frag_a}, mode=[2])):\n"
                    f"        {tiled_mma}.set(\n"
                    f"            cute.nvgpu.tcgen05.Field.ACCUMULATE,\n"
                    f"            {k_offset_var} != {k_loop_begin_expr} or cutlass.Int32(_tcgen05_kblk_idx) != cutlass.Int32(0),\n"
                    "        )\n"
                    f"        cute.gemm(\n"
                    f"            {tiled_mma},\n"
                    f"            {acc_frag},\n"
                    f"            [{tcgen05_frag_a}[None, None, cutlass.Int32(_tcgen05_kblk_idx), {mma_stage}]],\n"
                    f"            [{tcgen05_frag_b}[None, None, cutlass.Int32(_tcgen05_kblk_idx), {mma_stage}]],\n"
                    f"            {acc_frag},\n"
                    "        )"
                )
            )
            if tcgen05_use_tma:
                assert tma_kloop_args is not None
                if tcgen05_use_tma_pipeline:
                    cg.add_statement(_build_kloop_pipeline_release_if(tma_kloop_args))
                else:
                    cg.add_statement(
                        _build_kloop_non_pipeline_release_if(tma_kloop_args)
                    )
            else:
                cg.add_statement(statement_from_string("cute.arch.sync_threads()"))

    # === outer_suffix: convert fragment → per-thread scalar ===
    # Allocate smem_c in outer_prefix so all smem is allocated at the same
    # scope level (CuTe DSL assigns static smem offsets per scope). Only the
    # `universal` and `warp` MMA paths still need the staged smem_c buffer;
    # tcgen05 now uses the direct TMEM->reg->GMEM SIMT epilogue from
    # `_codegen_cute_store_tcgen05_tile` and skips the smem_c allocation.
    smem_c_ptr = df.new_var("smem_c")
    smem_c = df.new_var("smem_c_t")
    tCsC = df.new_var("tCsC")
    result_var = df.new_var("mma_result")

    tile_numel = bm * bn
    if mma_impl != "tcgen05":
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
            suffix.append(statement_from_string("cute.arch.sync_threads()"))
        else:
            assert tcgen05_plan is not None
            assert epi_active is not None
            assert epi_tidx is not None
            # The K-loop suffix's `acc_pipeline.producer_commit` +
            # `acc_producer_state.advance()` must run ONCE PER OUTPUT TILE.
            # In the persistent path, the splitter walks top-level
            # statements and only marks them per-tile if they read or
            # write a name that's already per-tile. These suffix
            # statements only mutate ``acc_producer_state`` via a method
            # call (no AST-visible write) and reference no per-tile
            # name directly, so without explicit tagging they get hoisted
            # out of the work-tile loop -- which means the SECOND tile
            # never commits its accumulator and the consumer-side
            # ``consumer_wait`` deadlocks (or the data is silently wrong
            # if no deadlock fires). Tag them per-tile via
            # ``_emit_per_tile_suffix`` so they stay inside the work-tile
            # loop.
            suffix_stmt = statement_from_string(
                f"if {tcgen05_plan.exec_active}:\n"
                f"    {tcgen05_plan.acc_pipeline}.producer_commit({tcgen05_plan.acc_producer_state})"
            )
            suffix.append(suffix_stmt)
            per_tile_stmts.append(suffix_stmt)
            advance_stmt = statement_from_string(
                emit_pipeline_advance(tcgen05_plan.acc_producer_state)
            )
            suffix.append(advance_stmt)
            per_tile_stmts.append(advance_stmt)
            # The full TMEM->reg->GMEM epilogue + allocator teardown for
            # tcgen05 is emitted by `_codegen_cute_store_tcgen05_tile` when
            # the kernel actually stores `out[tile_m, tile_n] = result`. That
            # path covers all bn widths now (the previously separate
            # staged-via-smem_c flow has been removed; the dead code lived
            # here).
            sync_stmt = statement_from_string("cute.arch.sync_threads()")
            suffix.append(sync_stmt)
            per_tile_stmts.append(sync_stmt)

    if mma_impl == "tcgen05":
        assert tcgen05_plan is not None
        assert epi_tidx is not None
        assert epi_active is not None
        assert tma_warp is not None
        df.register_cute_tcgen05_store_value(
            result_var,
            CuteTcgen05StoreValue(
                bm=bm,
                bn=bn,
                bk=bk,
                thr_mma=thr_mma,
                epi_warp_count=tcgen05_epi_warp_count_value,
                epi_acc_frag_base=tcgen05_epi_acc_frag_base,
                epi_tidx=epi_tidx,
                epi_active=epi_active,
                exec_active=tcgen05_plan.exec_active,
                epi_tile=tcgen05_plan.epi_tile,
                c_stage_count=tcgen05_c_stage_count_value,
                epilog_sync_barrier_id=_TCGEN05_EPILOG_SYNC_BARRIER_ID,
                tmem_load_atom=tcgen05_plan.tmem_load_atom,
                epilogue_rest_mode=tcgen05_plan.epilogue_rest_mode,
                acc_pipeline=tcgen05_plan.acc_pipeline,
                acc_producer_state=tcgen05_plan.acc_producer_state,
                acc_consumer_state=tcgen05_plan.acc_consumer_state,
                tmem_alloc_barrier=tcgen05_plan.tmem_alloc_barrier,
                tmem_allocator=tcgen05_plan.tmem_allocator,
                tmem_holding_buf=tcgen05_plan.tmem_holding_buf,
                tmem_dealloc_mbar_ptr=tcgen05_plan.tmem_dealloc_mbar_ptr,
                epi_acc_tmem_ptr=tcgen05_epi_acc_tmem_ptr,
                acc_tmem_cols=tcgen05_plan.acc_tmem_cols,
                tma_warp=tma_warp,
                tma_pipeline=tma_pipeline,
                tma_producer_state=tma_producer_state,
                is_two_cta=tcgen05_is_two_cta,
                use_tma=tcgen05_use_tma,
                ab_stage_count=tcgen05_ab_stage_count_value,
                acc_stage_count=tcgen05_acc_stage_count_value,
            ),
        )
    else:
        # Each thread reads its own (m, n) element from shared memory.
        suffix.append(
            statement_from_string(f"{result_var} = {smem_c}[{m_local}, {n_local}]")
        )

    # Register per-tile statements with the persistent-loop splitter so
    # everything else hoists out of the work-tile loop. The splitter also
    # auto-detects PID-decomposition statements via ``virtual_pid_var``
    # name lookup, so callers don't need to plumb registration through
    # ``_decompose_virtual_pid``. No-op when the kernel uses a
    # non-persistent ``pid_type`` (the splitter is only invoked from
    # ``_setup_tcgen05_persistent_kernel``).
    if per_tile_stmts:
        df.register_cute_tcgen05_per_tile_stmts(per_tile_stmts)
    # Register TMA-load role-block statements with the persistent role
    # partitioner (see ``Tcgen05PersistentProgramIDs._collect_tcgen05_role_blocks``).
    # Two registration shapes land here:
    # - Top-level prefix statements (the initial TMA prefetch IFs) --
    #   these are ALSO registered as per-tile via ``_emit_per_tile``,
    #   which is what keeps them inside the work-tile body so the
    #   partitioner can see them at top level.
    # - Nested statements emitted inside the K-loop body via
    #   ``cg.add_statement(...)`` -- these are NOT per-tile-registered;
    #   the K-loop itself rides into the work-tile body via per-tile
    #   name propagation, and the partitioner recurses one level into
    #   it to wrap these tagged children.
    # The partitioner asserts at run time that every registered tag was
    # visited, so a misregistered top-level stmt fails loudly rather
    # than silently dropping its role gate.
    if tma_load_role_stmts:
        df.register_cute_tcgen05_tma_load_role_stmts(tma_load_role_stmts)

    return expr_from_string(result_var)


def _mma_active_n_threads(mma_impl: str) -> int:
    if mma_impl in ("warp", "tcgen05"):
        return 2
    return 0


def _tcgen05_root_m_threads(bm: int, bn: int) -> int:
    # Wide tcgen05 tiles use a compact physical M thread map plus root-lane
    # serialization. Narrow N=8 tiles keep the original full-M thread family.
    if bn <= 8:
        return bm
    return min(32, bm)


def _tcgen05_tmem_barrier_thread_count(epi_warp_count: int) -> int:
    return 32 * (epi_warp_count + 1)


def _tcgen05_c_stage_count(bn: int) -> int:
    # Match the SM100 GEMM helper: narrow epilogues use a deeper TMA-store
    # ring buffer, while wider-N tiles fall back to two stages.
    return 4 if bn <= 16 else 2


def _tcgen05_ab_stage_count(num_stages: int) -> int:
    return max(1, min(int(num_stages), 2))


def _tcgen05_acc_stage_count(bn: int) -> int:
    # Match Quack/CUTLASS SM100 staging for the current non-blockscaled path:
    # keep two accumulator stages for all currently supported N<=256 tiles.
    return 2 if bn <= 256 else 1


def _tcgen05_config_int(config: object, key: str, default: int) -> int:
    value = cast("_ConfigLike", config).get(key, default)
    if not isinstance(value, int):
        return default
    return value


def _tcgen05_cluster_m(config: object) -> int:
    return max(1, min(2, _tcgen05_config_int(config, "tcgen05_cluster_m", 1)))


def _tcgen05_use_2cta_instrs(*, bm: int, cluster_m: int) -> bool:
    # Match Quack/CUTLASS SM100: clustered kernels are not automatically the
    # tcgen05 "CTA pair" instruction family. The special 2-CTA instructions only
    # apply to the 256-wide M tiler. Our current legal Helion family still uses
    # CTA-local M=128 tiles, even when the cluster shape is (2, 1, 1).
    return cluster_m == 2 and bm == 256


def _tcgen05_epi_warp_count(config: object, *, cta_thread_count: int) -> int:
    """Pick the epilogue warp count for a tcgen05 matmul kernel.

    Returns at most ``cta_thread_count // 32`` warps, capped by the
    ``tcgen05_num_epi_warps`` autotune knob (default 4). The other roles
    (one MMA exec warp + one A/B load warp) are added on top of this in
    ``CuteTcgen05MatmulPlan.role_warp_count``.
    """
    cta_warp_count = max(1, cta_thread_count // 32)
    return min(
        cta_warp_count,
        max(1, _tcgen05_config_int(config, "tcgen05_num_epi_warps", 4)),
    )


def _mma_impl_matches_problem_shape(
    mma_impl: str,
    input_dtype: torch.dtype,
    *,
    bm: int,
    bn: int,
    bk: int,
    tcgen05_cluster_m: int = 1,
) -> bool:
    if mma_impl == "universal":
        return True
    if (
        input_dtype not in (torch.float16, torch.bfloat16)
        or bn < 8
        or bn > 256
        or bn % 8 != 0
    ):
        return False
    if mma_impl == "warp":
        # Warp MMA atom is fixed-K (16 elements per BF16/FP16 instruction).
        return bk == 16 and bm >= 16 and bm % 16 == 0 and bn == 8
    if mma_impl == "tcgen05":
        # tcgen05 mma instruction K is 16 elements for BF16/FP16, but the
        # tile's K can be any positive multiple of that (the inner cute.gemm
        # loop just runs more instructions per K iteration). Larger tile_k
        # roughly halves the per-K-iter overhead per doubling. Capped at 256
        # to keep AB SMEM staging budget sane.
        if bk < 16 or bk > 256 or bk % 16 != 0:
            return False
        if bm in (64, 128):
            return True
        return bm == 256 and tcgen05_cluster_m == 2
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


def _is_zero_acc_node(node: Node | None) -> bool:
    if node is None:
        return False
    if node.op != "call_function":
        return False

    if node.target in _TRACE_THROUGH_TARGETS:
        src = node.args[0] if node.args else None
        return isinstance(src, Node) and _is_zero_acc_node(src)

    target_name = getattr(node.target, "__name__", "")
    if target_name in {"clone", "detach"}:
        src = node.args[0] if node.args else None
        return isinstance(src, Node) and _is_zero_acc_node(src)
    if target_name == "zeros":
        return True
    if target_name != "full":
        return False

    value = None
    if len(node.args) > 1:
        value = node.args[1]
    elif "value" in node.kwargs:
        value = node.kwargs["value"]
    return value in (0, 0.0)


def _choose_mma_impl(
    input_dtype: torch.dtype,
    *,
    bm: int,
    bn: int,
    bk: int,
    config: object | None = None,
) -> str:
    tcgen05_cluster_m = 1
    if config is not None:
        tcgen05_cluster_m = _tcgen05_cluster_m(config)
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
            tcgen05_cluster_m=tcgen05_cluster_m,
        ):
            return env_choice
        return "universal"
    if _mma_impl_matches_problem_shape(
        "tcgen05",
        input_dtype,
        bm=bm,
        bn=bn,
        bk=bk,
        tcgen05_cluster_m=tcgen05_cluster_m,
    ):
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
    *,
    tcgen05_cluster_m: int = 1,
) -> list[ast.AST]:
    if mma_impl == "warp":
        tiled_mma_expr = (
            "cute.make_tiled_mma("
            "cute.make_mma_atom("
            f"cute.nvgpu.warp.MmaF16BF16Op({input_dtype_str}, {acc_dtype_str}, (16, 8, 16))"
            f"), atom_layout_mnk=({bm // 16}, 1, 1))"
        )
    elif mma_impl == "tcgen05":
        tiled_mma_expr = _tcgen05_tiled_mma_expr(
            input_dtype_str,
            acc_dtype_str,
            bm,
            bn,
            tcgen05_cluster_m=tcgen05_cluster_m,
        )
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
    *,
    tcgen05_cluster_m: int = 1,
) -> str:
    cta_group_expr = "cute.nvgpu.tcgen05.CtaGroup.ONE"
    if _tcgen05_use_2cta_instrs(bm=bm, cluster_m=tcgen05_cluster_m):
        cta_group_expr = "cute.nvgpu.tcgen05.CtaGroup.TWO"
    return (
        "cutlass.utils.blackwell_helpers.make_trivial_tiled_mma("
        f"{input_dtype_str}, "
        "cute.nvgpu.tcgen05.OperandMajorMode.K, "
        "cute.nvgpu.tcgen05.OperandMajorMode.MN, "
        f"{acc_dtype_str}, "
        f"{cta_group_expr}, "
        f"({bm}, {bn}), "
        "cute.nvgpu.tcgen05.OperandSource.SMEM)"
    )


def _new_tcgen05_layout_plan(df: DeviceFunction) -> _Tcgen05LayoutPlan:
    return _Tcgen05LayoutPlan(
        exec_active=df.new_var("tcgen05_exec_active"),
        smem_a_layout=df.new_var("sA_layout"),
        smem_b_layout=df.new_var("sB_layout"),
        c_layout=df.new_var("tcgen05_c_layout"),
        epi_tile=df.new_var("tcgen05_epi_tile"),
        tmem_load_atom=df.new_var("tcgen05_tmem_load_atom"),
        acc_tmem_cols=df.new_var("tcgen05_acc_tmem_cols"),
        tmem_holding_buf=df.new_var("tcgen05_tmem_holding_buf"),
        tmem_dealloc_mbar_ptr=df.new_var("tcgen05_tmem_dealloc_mbar_ptr"),
        tmem_alloc_barrier=df.new_var("tcgen05_tmem_alloc_barrier"),
        tmem_allocator=df.new_var("tcgen05_tmem_allocator"),
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
    ab_stage_count: int,
    is_two_cta: bool,
    input_dtype_str: str,
    acc_dtype_str: str,
) -> list[ast.AST]:
    return [
        statement_from_string(
            f"{plan.smem_a_layout} = "
            f"{_tcgen05_smem_layout_expr(tiled_mma, bm, bn, bk, input_dtype_str, ab_stage_count, operand='a')}"
        ),
        statement_from_string(
            f"{plan.smem_b_layout} = "
            f"{_tcgen05_smem_layout_expr(tiled_mma, bm, bn, bk, input_dtype_str, ab_stage_count, operand='b')}"
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
            f"{plan.tmem_load_atom} = cutlass.utils.blackwell_helpers.get_tmem_load_op("
            f"({bm}, {bn}, {bk}), {plan.c_layout}, "
            f"{acc_dtype_str}, {acc_dtype_str}, {plan.epi_tile}, {is_two_cta!s})"
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
    num_stages: int,
    *,
    operand: str,
) -> str:
    if operand == "a":
        return (
            "cutlass.utils.blackwell_helpers.make_smem_layout_a("
            f"{tiled_mma}, ({bm}, {bn}, {bk}), {dtype_str}, {num_stages})"
        )
    assert operand == "b"
    return (
        "cutlass.utils.blackwell_helpers.make_smem_layout_b("
        f"{tiled_mma}, ({bm}, {bn}, {bk}), {dtype_str}, {num_stages})"
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
    if ctx.cg.current_grid_state is None:
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


def codegen_cute_mma_direct_mm(
    ctx: LoweringContext,
    node: Node,
    *,
    serial_k_extent: int | None,
) -> ast.AST | None:
    from ..generate_ast import GenerateAST

    if not isinstance(ctx.cg, GenerateAST):
        return None
    plan = getattr(ctx, "cute_matmul_plan", None)
    if not isinstance(plan, MatmulExecutionPlan):
        return None
    if plan.kind is not MatmulExecutionKind.DIRECT_GROUPED_N:
        return None
    if serial_k_extent is None or serial_k_extent <= 0:
        return None
    if node.target is not torch.ops.aten.mm.default:
        return None

    lhs_node = node.args[0]
    rhs_node = node.args[1]
    if not isinstance(lhs_node, Node) or not isinstance(rhs_node, Node):
        return None
    lhs_info = _trace_to_load_tensor(lhs_node)
    rhs_info = _trace_to_load_tensor(rhs_node)
    if lhs_info is None or rhs_info is None:
        return None
    lhs_load, _, lhs_fake = lhs_info
    rhs_load, _, rhs_fake = rhs_info
    lhs_val = lhs_node.meta.get("val")
    rhs_val = rhs_node.meta.get("val")
    if (
        lhs_fake.ndim != 2
        or rhs_fake.ndim != 2
        or not isinstance(lhs_val, torch.Tensor)
        or not isinstance(rhs_val, torch.Tensor)
        or lhs_val.ndim != 2
        or rhs_val.ndim != 2
    ):
        return None
    if lhs_fake.dtype not in (torch.float16, torch.bfloat16):
        return None
    load_plan = analyze_direct_grouped_n_loads(
        lhs_load,
        rhs_load,
        k_extent=serial_k_extent,
        n_extent=int(rhs_val.shape[1]),
    )
    if load_plan is None:
        return None

    mma_impl = _choose_mma_impl(
        lhs_fake.dtype,
        bm=plan.bm,
        bn=plan.bn,
        bk=plan.bk,
        config=ctx.cg.device_function.config,
    )
    # The grouped-N direct path only emits warp MMA. Auto-selection prefers
    # tcgen05 for plan.bm in (64, 128), but tcgen05 isn't implemented here, so
    # transparently fall back to warp on tcgen05-capable machines as long as
    # the user didn't explicitly request a different implementation.
    if (
        mma_impl == "tcgen05"
        and os.environ.get("HELION_CUTE_MMA_IMPL", "auto").strip().lower() == "auto"
        and _mma_impl_matches_problem_shape(
            "warp", lhs_fake.dtype, bm=plan.bm, bn=plan.bn, bk=plan.bk
        )
        and get_cute_mma_support().warp_f16bf16
    ):
        mma_impl = "warp"
    if mma_impl != "warp":
        return None

    cg = ctx.cg
    grid_state = cg.current_grid_state
    if grid_state is None:
        return None
    prefix = grid_state.outer_prefix
    scalar_axis = grid_state.block_thread_axes.get(plan.scalar_block_id)
    if scalar_axis is None:
        return None
    scalar_strategy = cg.device_function.tile_strategy.block_id_to_strategy.get(
        (plan.scalar_block_id,)
    )
    lane_var = getattr(scalar_strategy, "_synthetic_cute_lane_var", None)
    if plan.lane_extent > 1 and not isinstance(lane_var, str):
        return None

    m_index_var = grid_state.strategy.index_var(plan.m_block_id)
    m_local = _local_mma_coord_expr(cg, plan.m_block_id)
    m_tile_origin = f"cutlass.Int32({m_index_var}) - ({m_local})"
    scalar_thread = f"cutlass.Int32(cute.arch.thread_idx()[{scalar_axis}])"
    lane_group_base = (
        "cutlass.Int32(0)"
        if not isinstance(lane_var, str)
        else f"cutlass.Int32({lane_var}) * cutlass.Int32({plan.groups_per_lane})"
    )
    tile_group = f"({scalar_thread}) // cutlass.Int32({plan.bn})"
    tile_n_local = f"({scalar_thread}) % cutlass.Int32({plan.bn})"
    mma_active = f"({tile_n_local}) < cutlass.Int32({_mma_active_n_threads(mma_impl)})"
    mma_thread_linear = f"{m_local} + ({tile_n_local}) * cutlass.Int32({plan.bm})"
    m_size = int(lhs_fake.shape[0])
    n_size = int(rhs_val.shape[1])
    k_size = serial_k_extent

    df = cg.device_function
    input_dtype_str = (
        "cutlass.Float16" if lhs_fake.dtype is torch.float16 else "cutlass.BFloat16"
    )
    acc_dtype_str = "cutlass.Float32"
    lhs_arg_name = df.tensor_arg(lhs_fake).name
    rhs_arg_name = df.tensor_arg(rhs_fake).name

    tiled_mma = df.new_var("direct_tiled_mma")
    thr_mma = df.new_var("direct_thr_mma")
    acc_frag = df.new_var("direct_acc_frag")
    smem_a_ptr = df.new_var("direct_smem_a")
    smem_a = df.new_var("direct_sA")
    smem_b_ptr = df.new_var("direct_smem_b")
    smem_b = df.new_var("direct_sB")
    smem_c_ptr = df.new_var("direct_smem_c")
    smem_c = df.new_var("direct_sC")
    tAsA = df.new_var("direct_tAsA")
    tBsB = df.new_var("direct_tBsB")
    tCsC = df.new_var("direct_tCsC")
    rA = df.new_var("direct_rA")
    rB = df.new_var("direct_rB")
    k_offset_var = df.new_var("direct_k_offset")
    result_var = df.new_var("direct_mma_result")

    for stmt in _make_tiled_mma_setup(
        mma_impl,
        tiled_mma,
        thr_mma,
        mma_thread_linear,
        input_dtype_str,
        acc_dtype_str,
        plan.bm,
        plan.bn,
    ):
        prefix.append(stmt)
    prefix.append(
        statement_from_string(
            f"{acc_frag} = cute.make_fragment("
            f"{tiled_mma}.partition_shape_C(({plan.bm}, {plan.bn})), {acc_dtype_str})"
        )
    )
    prefix.append(
        statement_from_string(
            f"{smem_a_ptr} = cute.arch.alloc_smem({input_dtype_str}, {plan.bm * plan.bk})"
        )
    )
    prefix.append(
        statement_from_string(
            f"{smem_a} = cute.make_tensor("
            f"{smem_a_ptr}, cute.make_layout(({plan.bm}, {plan.bk}), stride=({plan.bk}, 1)))"
        )
    )
    prefix.append(
        statement_from_string(
            f"{smem_b_ptr} = cute.arch.alloc_smem({input_dtype_str}, {plan.bn * plan.bk})"
        )
    )
    prefix.append(
        statement_from_string(
            f"{smem_b} = cute.make_tensor("
            f"{smem_b_ptr}, "
            f"cute.make_layout(({plan.bn}, {plan.bk}), stride=({plan.bk}, 1)))"
        )
    )
    prefix.append(
        statement_from_string(
            f"{smem_c_ptr} = cute.arch.alloc_smem({acc_dtype_str}, {plan.bm * plan.bn}, alignment=128)"
        )
    )
    prefix.append(
        statement_from_string(
            f"{smem_c} = cute.make_tensor("
            f"{smem_c_ptr}, "
            f"cute.make_layout(({plan.bm}, {plan.bn}), stride=({plan.bn}, 1)))"
        )
    )
    cg.add_statement(statement_from_string(f"{result_var} = {acc_dtype_str}(0.0)"))
    cg.add_statement(
        statement_from_string(
            f"if {mma_active}:\n"
            f"    for _mma_i in range(cute.size({acc_frag})):\n"
            f"        {acc_frag}[_mma_i] = {acc_dtype_str}(0.0)"
        )
    )
    cg.add_statement(
        statement_from_string(
            f"for {k_offset_var} in range(0, {k_size}, {plan.bk}):\n"
            f"    if {mma_active} and ({tile_group}) == cutlass.Int32(0):\n"
            f"        for _load_i in range(({plan.bm * plan.bk} + {plan.bm * 2} - 1) // {plan.bm * 2}):\n"
            f"            _flat = {mma_thread_linear} + cutlass.Int32(_load_i) * cutlass.Int32({plan.bm * 2})\n"
            f"            if _flat < cutlass.Int32({plan.bm * plan.bk}):\n"
            f"                _row = _flat // cutlass.Int32({plan.bk})\n"
            f"                _col = _flat % cutlass.Int32({plan.bk})\n"
            f"                _gm = {m_tile_origin} + _row\n"
            f"                _gk = cutlass.Int32({load_plan.lhs_k_offset}) + cutlass.Int32({k_offset_var}) + _col\n"
            f"                {smem_a}[_row, _col] = ("
            f"{lhs_arg_name}[_gm, _gk] "
            f"if _gm < cutlass.Int32({m_size}) and _gk < cutlass.Int32({load_plan.lhs_k_offset + k_size}) "
            f"else {input_dtype_str}(0.0))\n"
            f"    cute.arch.sync_threads()\n"
            f"    for _n_group in range({plan.groups_per_lane}):\n"
            f"        if {mma_active} and ({tile_group}) == cutlass.Int32(_n_group):\n"
            f"            for _load_i in range(({plan.bn * plan.bk} + {plan.bm * 2} - 1) // {plan.bm * 2}):\n"
            f"                _flat = {mma_thread_linear} + cutlass.Int32(_load_i) * cutlass.Int32({plan.bm * 2})\n"
            f"                if _flat < cutlass.Int32({plan.bn * plan.bk}):\n"
            f"                    _row = _flat // cutlass.Int32({plan.bk})\n"
            f"                    _col = _flat % cutlass.Int32({plan.bk})\n"
            f"                    _gn = cutlass.Int32({load_plan.rhs_n_offset}) + ({lane_group_base} + cutlass.Int32(_n_group)) * cutlass.Int32({plan.bn}) + _row\n"
            f"                    _gk = cutlass.Int32({load_plan.rhs_k_offset}) + cutlass.Int32({k_offset_var}) + _col\n"
            f"                    {smem_b}[_row, _col] = ("
            f"{rhs_arg_name}[_gk, _gn] "
            f"if _gn < cutlass.Int32({load_plan.rhs_n_offset + n_size}) and _gk < cutlass.Int32({load_plan.rhs_k_offset + k_size}) "
            f"else {input_dtype_str}(0.0))\n"
            f"        cute.arch.sync_threads()\n"
            f"        if {mma_active} and ({tile_group}) == cutlass.Int32(_n_group):\n"
            f"            {tAsA} = {thr_mma}.partition_A({smem_a})\n"
            f"            {tBsB} = {thr_mma}.partition_B({smem_b})\n"
            f"            {rA} = cute.make_fragment_like({tAsA}, {input_dtype_str})\n"
            f"            {rB} = cute.make_fragment_like({tBsB}, {input_dtype_str})\n"
            f"            for _mma_i in range(cute.size({rA})):\n"
            f"                {rA}[_mma_i] = {tAsA}[_mma_i]\n"
            f"            for _mma_i in range(cute.size({rB})):\n"
            f"                {rB}[_mma_i] = {tBsB}[_mma_i]\n"
            f"            cute.gemm({tiled_mma}, {acc_frag}, [{rA}], [{rB}], {acc_frag})\n"
            f"        cute.arch.sync_threads()"
        )
    )
    cg.add_statement(
        statement_from_string(
            f"for _n_group in range({plan.groups_per_lane}):\n"
            f"    if {mma_active} and ({tile_group}) == cutlass.Int32(_n_group):\n"
            f"        {tCsC} = {thr_mma}.partition_C({smem_c})\n"
            f"        for _mma_i in range(cute.size({tCsC})):\n"
            f"            {tCsC}[_mma_i] = {acc_frag}[_mma_i]\n"
            f"    cute.arch.sync_threads()\n"
            f"    if ({tile_group}) == cutlass.Int32(_n_group):\n"
            f"        {result_var} = {smem_c}[{m_local}, {tile_n_local}]\n"
            f"    cute.arch.sync_threads()"
        )
    )
    return expr_from_string(result_var)


# ---- hl.dot entry point ----


def codegen_cute_mma_dot(state: CodegenState) -> object | None:
    """Generate MMA code for an hl.dot node.  Returns None to fall back."""
    from ..generate_ast import GenerateAST

    if not isinstance(state.codegen, GenerateAST):
        return None
    if state.codegen.current_grid_state is None:
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
