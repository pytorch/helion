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
from typing import TYPE_CHECKING

import torch
from torch.fx.node import Node

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..matmul_utils import _needs_f32_accumulator
from ..tile_strategy import DeviceLoopState

if TYPE_CHECKING:
    from ..aten_lowering import LoweringContext
    from ..compile_environment import CompileEnvironment
    from ..generate_ast import GenerateAST
    from ..inductor_lowering import CodegenState


_TRACE_THROUGH_TARGETS = {
    torch.ops.prims.convert_element_type.default,
    torch.ops.aten._to_copy.default,
    # NOTE: permute is NOT included because the MMA pipeline reads
    # raw tensor data — tracing through permute would bypass the
    # data shuffle.  Permuted operands fall back to scalar codegen.
}


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

    # --- Multi-warp tiling ---
    # Compute atom_layout_mnk based on tile size and available threads.
    # MmaUniversalOp atom shape is (1,1,1) so atom_layout directly
    # controls thread distribution.
    atom_layout = f"({bm}, {bn}, 1)"

    # Variable names
    tiled_mma = df.new_var("tiled_mma")
    thr_mma = df.new_var("thr_mma")
    acc_frag = df.new_var("acc_frag")

    # === outer_prefix: MMA setup + shared memory alloc + accumulator init ===
    prefix = device_loop.outer_prefix

    prefix.append(
        statement_from_string(
            f"{tiled_mma} = cute.make_tiled_mma("
            f"cute.nvgpu.MmaUniversalOp(abacc_dtype={acc_dtype_str}), "
            f"atom_layout_mnk={atom_layout})"
        )
    )
    prefix.append(
        statement_from_string(
            f"{thr_mma} = {tiled_mma}.get_slice({m_local} + ({n_local}) * cutlass.Int32({bm}))"
        )
    )
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

    prefix.append(
        statement_from_string(
            f"{smem_a_ptr} = cute.arch.alloc_smem({input_dtype_str}, {bm * bk})"
        )
    )
    prefix.append(
        statement_from_string(
            f"{smem_a} = cute.make_tensor({smem_a_ptr}, "
            f"cute.make_layout(({bm}, {bk}), stride=({bk}, 1)))"
        )
    )
    prefix.append(
        statement_from_string(
            f"{smem_b_ptr} = cute.arch.alloc_smem({input_dtype_str}, {bn * bk})"
        )
    )
    prefix.append(
        statement_from_string(
            f"{smem_b} = cute.make_tensor({smem_b_ptr}, "
            f"cute.make_layout(({bn}, {bk}), stride=({bk}, 1)))"
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

    if acc_expr is None:
        cg.add_statement(
            statement_from_string(
                f"if {k_offset_var} == cutlass.Int32(0):\n"
                f"    for _mma_i in range(cute.size({acc_frag})):\n"
                f"        {acc_frag}[_mma_i] = {acc_dtype_str}(0.0)"
            )
        )
    else:
        cg.add_statement(
            statement_from_string(
                f"if {k_offset_var} == cutlass.Int32(0):\n"
                f"    for _mma_i in range(cute.size({acc_frag})):\n"
                f"        {acc_frag}[_mma_i] = {acc_dtype_str}({{acc}})",
                acc=acc_expr,
            )
        )

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

    # --- Shared → Register with f16→f32 cast ---
    cg.add_statement(statement_from_string(f"{tAsA} = {thr_mma}.partition_A({smem_a})"))
    cg.add_statement(statement_from_string(f"{tBsB} = {thr_mma}.partition_B({smem_b})"))
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

    # Execute MMA (in-place accumulation)
    cg.add_statement(
        statement_from_string(
            f"cute.gemm({tiled_mma}, {acc_frag}, {rA}, {rB}, {acc_frag})"
        )
    )

    # === outer_suffix: convert fragment → per-thread scalar ===
    # Allocate smem_c in outer_prefix so all smem is allocated at the same
    # scope level (CuTe DSL assigns static smem offsets per scope).
    smem_c_ptr = df.new_var("smem_c")
    smem_c = df.new_var("smem_c_t")
    tCsC = df.new_var("tCsC")
    result_var = df.new_var("mma_result")

    tile_numel = bm * bn
    prefix.append(
        statement_from_string(
            f"{smem_c_ptr} = cute.arch.alloc_smem({acc_dtype_str}, {tile_numel})"
        )
    )
    prefix.append(
        statement_from_string(
            f"{smem_c} = cute.make_tensor("
            f"{smem_c_ptr}, cute.make_layout(({bm}, {bn}), stride=({bn}, 1)))"
        )
    )

    suffix = device_loop.outer_suffix
    suffix.append(statement_from_string(f"{tCsC} = {thr_mma}.partition_C({smem_c})"))
    suffix.append(
        statement_from_string(
            f"for _mma_i in range(cute.size({tCsC})):\n"
            f"    {tCsC}[_mma_i] = {acc_frag}[_mma_i]"
        )
    )
    suffix.append(statement_from_string("cute.arch.sync_threads()"))

    # Each thread reads its own (m, n) element from shared memory
    suffix.append(
        statement_from_string(f"{result_var} = {smem_c}[{m_local}, {n_local}]")
    )

    return expr_from_string(result_var)


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
    if not can_codegen_cute_mma_aten(node, with_acc):
        return None

    if with_acc:
        acc_node = node.args[0]
        assert isinstance(acc_node, Node)
        acc_expr = ctx.to_ast(ctx.env[acc_node])
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
    if state.fx_node is None:
        return None
    if not can_codegen_cute_mma_dot(state.fx_node):
        return None

    lhs_node = state.fx_node.args[0]
    rhs_node = state.fx_node.args[1]
    acc_expr = None
    if len(state.fx_node.args) > 2:
        acc_ast = state.ast_arg(2)
        if not (isinstance(acc_ast, ast.Constant) and acc_ast.value is None):
            acc_expr = acc_ast
    assert isinstance(lhs_node, Node) and isinstance(rhs_node, Node)

    return _emit_mma_pipeline(
        state.codegen,
        lhs_node,
        rhs_node,
        acc_expr=acc_expr,
        fx_node=state.fx_node,
    )
