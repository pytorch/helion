"""Let idle warps fetch immutable weight lines while warp zero reduces one row."""

from __future__ import annotations

import ast
from math import prod
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...language import _tracing_ops
from ...language import memory_ops
from ...runtime.config import Config
from ..compile_environment import CompileEnvironment
from ._ast_pass_utils import assigned_name
from ._ast_pass_utils import parse_expr
from ._ast_pass_utils import phase_barrier_call
from ._ast_pass_utils import same_expr
from .analysis.match_common import node_tensor
from .autotune.config_keys import PHASE_PARTICIPATION_KEY
from .autotune.config_keys import SPARE_ROW_PREFETCH_KEY
from .cooperative_row_normalization import _dispatch
from .early_indexed_input import has_known_effects
from .fragment_vector_io import static_int
from .local_atomic import _reachable_graphs
from .loop_nesting import boundary_only_grid_blocks
from .register_loads import host_load_is_readonly
from .thread_block_projection import literal_launch_block
from .thread_budget import tile_loop_thread_count

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...autotuner.config_spec import ConfigSpec
    from ..device_function import DeviceFunction
    from ..device_ir import DeviceIR

_WARP = 32
_LINE_BF16 = 128 // 2


def matrix_for_row(
    env: CompileEnvironment, ir: DeviceIR, root: int, *, allow_unbound: bool
) -> torch.Tensor | None:
    """Prove a complete immutable BF16 [row, :] read, including the row domain."""
    spec = env.config_spec
    if spec.target_device_capability is None or spec.target_device_capability[0] != 10:
        return None
    ids = spec.cute_warp_row_reduction_roots.get(root)
    if not ids or ir.host_function is None or ir.root_ids[0] != root:
        return None
    row, _, length, _ = ids
    rows = env.block_sizes[row].size
    if type(rows) is not int or rows < 1 or length % _LINE_BF16:
        return None
    if any(
        axis != row
        and (
            type(env.block_sizes[axis].size) is not int
            or env.block_sizes[axis].size != 1
        )
        for axis in ir.grid_block_ids[ir.root_ids.index(root)]
    ):
        return None
    found = []
    with ir.host_function:
        for n in ir.graphs[root].graph.nodes:
            if not (
                n.target is memory_ops.load
                and isinstance(n.args[1], (list, tuple))
                and len(n.args[1]) == 2
                and n.args[1][1] == slice(None)
                and isinstance(index := n.args[1][0], torch.fx.Node)
                and index.target is _tracing_ops._get_symnode
                and isinstance(symbol := index.meta.get("val"), torch.SymInt)
                and env.get_block_id(symbol) == row
                and isinstance(arg := n.args[0], torch.fx.Node)
                and arg.target is _tracing_ops._host_tensor
                and (tensor := node_tensor(arg)) is not None
                and env.tensor_input_source(tensor) is not None
                and tensor.dtype == torch.bfloat16
                and tensor.ndim == 2
                and env.static_kernel_tensor_extent(tensor.shape[0]) == rows
                and env.static_kernel_tensor_extent(tensor.shape[1]) == length
                and env.traced_strides_hold(tensor)
                and env.input_residues_hold_modulo(16)
                and (pitch := static_int(tensor.stride(0))) is not None
                and pitch >= length
                and pitch % 8 == 0
                and static_int(tensor.stride(1)) == 1
                # Match the ordinary signed Int32 address without wraparound.
                and (rows - 1) * pitch + length - 1 < 2**31
            ):
                continue
            # Alignment belongs to the installed CuTe tensor. The generated
            # constexpr guards the hint; it must not change the shared search
            # schema for different base addresses of the same workload.
            found.append((n, tensor))
        if len(found) != 1:
            return None
        load, tensor = found[0]
        graphs = _reachable_graphs(ir.graphs)
        if any(not has_known_effects(n) for g in graphs for n in g.graph.nodes):
            return None
        if not host_load_is_readonly(load, env, graphs, allow_unbound=allow_unbound):
            return None
        return tensor


def logical_geometry(spec: ConfigSpec, cfg: Mapping[str, object]) -> bool:
    root = spec.cute_spare_row_prefetch_root
    if root is None:
        return False
    row, k, width, _ = spec.cute_warp_row_reduction_roots[root]
    blocks, threads, chunks = (
        cfg.get("block_sizes"),
        cfg.get("num_threads"),
        cfg.get("reduction_loops"),
    )
    return (
        not cfg.get(PHASE_PARTICIPATION_KEY)
        and cfg.get("cute_warp_row_reductions") is True
        and cfg.get("pid_type") == "persistent_interleaved"
        and cfg.get("cross_loop_pipeline") == "barrier"
        and not cfg.get("cute_pdl")
        and cfg.get("cute_cluster_n", 1) == 1
        and isinstance(blocks, list)
        and isinstance(threads, list)
        and isinstance(chunks, list)
        and spec.block_sizes.config_get(blocks, row, 0) == 1
        and spec.num_threads.config_get(threads, row, 0) == 1
        and spec.num_threads.config_get(threads, k, 0) == 1
        and type(chunk := spec.reduction_loops.config_get(chunks, k)) is int
        and chunk >= _WARP
        and chunk % _WARP == 0
        and width % chunk == 0
    )


def has_spare_lanes(threads: int, width: int) -> bool:
    return (
        width > 0
        and width % _LINE_BF16 == 0
        and (threads % _WARP == 0 and _WARP + width // _LINE_BF16 <= threads <= 1024)
    )


def add_seed(
    env: CompileEnvironment, ir: DeviceIR, seeds: list[Config]
) -> list[Config]:
    spec = env.config_spec
    root = spec.cute_spare_row_prefetch_root
    if root is None or ir.host_function is None:
        return seeds
    width = spec.cute_warp_row_reduction_roots[root][2]
    with ir.host_function:
        inactive = boundary_only_grid_blocks(env, ir, ir.graphs)
        for parent in reversed(seeds):
            if logical_geometry(spec, parent.config) and has_spare_lanes(
                tile_loop_thread_count(
                    env, ir, ir.graphs, parent, inactive_block_ids=inactive
                ),
                width,
            ):
                return [
                    *seeds,
                    Config.from_dict(parent.config | {SPARE_ROW_PREFETCH_KEY: True}),
                ]
    return seeds


def mark_proved_warp(
    fn: DeviceFunction, ir: DeviceIR, root: int, owner: ast.If, index: str, k_index: str
) -> None:
    """Anchor only the elected singleton warp, after the row-store and sweep proofs."""
    env = CompileEnvironment.current()
    if env.config_spec.cute_spare_row_prefetch_root != root:
        return
    tensor = matrix_for_row(env, ir, root, allow_unbound=False)
    if tensor is None:
        return
    name = fn.tensor_arg(tensor).name
    # The working warp must in fact issue that exact ordinary [row,K] load.
    ptr = parse_expr(
        f"({name}.iterator + cutlass.Int32({index}) * cutlass.Int32({name}.layout.stride[0]) "
        f"+ cutlass.Int32({k_index}) * cutlass.Int32({name}.layout.stride[1])).load()"
    )
    if not any(isinstance(n, ast.Call) and same_expr(n, ptr) for n in ast.walk(owner)):
        return
    fn.cute_state.lanes.spare_row_witness = (name, index, ptr)


def _first_row_index(prefix: list[ast.stmt], index: str, virtual: str) -> bool:
    """Verify the final row still equals the dispatched local PID; no input reads."""
    pid = {virtual}
    unit: set[str] = set()
    zero: set[str] = set()
    names = {virtual}
    for st in prefix:
        if (
            not isinstance(st, ast.Assign)
            or not (name := assigned_name(st))
            or name in names
        ):
            return False
        names.add(name)
        v = st.value
        if isinstance(v, ast.Name) and v.id in pid:
            pid.add(name)
        elif isinstance(v, ast.Constant) and type(v.value) is int and v.value in (0, 1):
            (unit if v.value == 1 else zero).add(name)
        elif isinstance(v, ast.Name) and (v.id in unit or v.id in zero):
            (unit if v.id in unit else zero).add(name)
        elif (
            isinstance(v, ast.BinOp)
            and isinstance(v.left, ast.Name)
            and v.left.id in pid
            and (
                isinstance(v.right, ast.Name)
                and v.right.id in unit
                or isinstance(v.right, ast.Constant)
                and type(v.right.value) is int
                and v.right.value == 1
            )
        ):
            if isinstance(v.op, ast.FloorDiv):
                pid.add(name)
            elif isinstance(v.op, ast.Mod):
                zero.add(name)
            else:
                return False
        else:
            return False
    return index in pid


def install(fn: DeviceFunction, definition: ast.FunctionDef) -> None:
    spec = CompileEnvironment.current().config_spec
    root = spec.cute_spare_row_prefetch_root
    dims = literal_launch_block(fn.codegen.host_statements, fn.name)
    if root is None or not dims or fn.cute_state.lanes.launch_block != dims:
        raise exc.InvalidConfig("spare-warp prefetch requires a literal physical CTA")
    row, _, width, _ = spec.cute_warp_row_reduction_roots[root]
    rows = CompileEnvironment.current().block_sizes[row].size
    if type(rows) is not int or not has_spare_lanes(prod(dims), width):
        raise exc.InvalidConfig(
            "spare-warp prefetch requires idle lanes for the complete row"
        )
    witness = fn.cute_state.lanes.spare_row_witness
    if witness is None:
        raise exc.InvalidConfig(
            "spare-warp prefetch requires the proved singleton warp and matrix load"
        )
    name, index, ptr = witness
    anchors = []
    joins = [
        call.func.id
        for n in ast.walk(definition)
        if isinstance(n, ast.stmt)
        and (call := phase_barrier_call(n)) is not None
        and isinstance(call.func, ast.Name)
    ]
    if joins and all(join == "_cute_grid_barrier" for join in joins):
        for loop in definition.body:
            if (
                not isinstance(loop, ast.For)
                or (dispatch := _dispatch(loop, rows)) is None
            ):
                continue
            body, virtual, base = dispatch
            if base != 0 or not body or not isinstance(owner := body[-1], ast.If):
                continue
            if (
                same_expr(owner.test, parse_expr("cute.arch.warp_idx() == 0"))
                and all(isinstance(n, ast.Pass) for n in owner.orelse)
                and _first_row_index(body[:-1], index, virtual)
                and any(
                    isinstance(n, ast.Call) and same_expr(n, ptr)
                    for n in ast.walk(owner)
                )
            ):
                anchors.append(body)
    if len(anchors) != 1:
        raise exc.InvalidConfig(
            "spare-warp prefetch requires an exact first row dispatch and live warp-zero load"
        )
    (body,) = anchors
    position, matrix = len(body) - 1, name
    tid, col, addr = (
        fn.new_var("row_prefetch_" + suffix, dce=False)
        for suffix in ("tid", "col", "ptr")
    )
    # Use the launcher dimensions, matching the hardware warp-index linearization.
    # A separate folded constant would make ptxas compute this same index twice.
    physical = "cutlass.Int32(cute.arch.thread_idx()[0])"
    if dims[1:] != (1, 1):
        yz = "cutlass.Int32(cute.arch.thread_idx()[1])"
        if dims[2] != 1:
            yz += " + cutlass.Int32(cute.arch.block_dim()[1]) * cutlass.Int32(cute.arch.thread_idx()[2])"
        physical += f" + cutlass.Int32(cute.arch.block_dim()[0]) * ({yz})"
    # The SDK sees the actual address on every rebind. On weak pointers the
    # hint disappears and the original warp executes its ordinary scalar loads.
    hint = ast.parse(
        f"if cutlass.const_expr({matrix}.iterator.alignment >= 16 and {matrix}.layout.stride[1] == 1):\n"
        f"    {tid} = {physical}\n"
        f"    if {tid} >= {_WARP} and {tid} < {_WARP + width // _LINE_BF16}:\n"
        f"        {col} = ({tid} - {_WARP}) * {_LINE_BF16}\n"
        f"        {addr} = {matrix}.iterator + cutlass.Int64({index}) * cutlass.Int64({matrix}.layout.stride[0]) + {col}\n"
        f"        cute.arch.prefetch({addr}.llvm_ptr, cache_level='L2')"
    ).body
    body[position:position] = hint
