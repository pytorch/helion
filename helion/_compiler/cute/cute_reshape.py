"""CuTe codegen for tile reshape/permute operations.

Each thread (or lane iteration) holds one element of a tile, the one at its
coordinates. Those coordinates are per block id: each tile dimension's block_id
is looked up and mapped to its thread axis or lane loop via
active_device_loops, NOT derived from the element's position in the tile.

Permuting the dimensions of a tile therefore never moves an element between
threads (the thread holding ``x[i, j]`` holds ``x.T[j, i]``); only a reshape
that changes which block ids own the elements shuffles them through shared
memory.
"""

from __future__ import annotations

import ast
import contextlib
from typing import TYPE_CHECKING
from typing import Callable
from typing import cast

import sympy
import torch
from torch.fx.node import Node
from torch.fx.node import map_arg
from torch.utils._sympy.functions import FloorDiv

from ... import exc
from ...language._tracing_ops import _for_loop
from ...language._tracing_ops import _for_loop_step
from ...language.inline_asm_ops import inline_asm_elementwise
from ...language.matmul_ops import dot as hl_dot
from ...language.memory_ops import _cute_resolve_active_slice_block_id
from ...language.reduce_ops import _reduce
from ...language.scan_ops import _associative_scan
from ...language.view_ops import join
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..compile_environment import _symint_expr
from ..host_function import HostFunction
from ..indexing_strategy import _get_tile_with_offset_info
from ..indexing_strategy import compute_slice_size
from ..tile_strategy import DeviceLoopState
from ..variable_origin import BlockSizeOrigin
from .cute_fx_walk import build_inner_outputs_index_from_graphs
from .cute_fx_walk import reach_matmul_anchors
from .indexing import CuteShapeChainView
from .indexing import is_cute_shape_chain_target

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Sequence

    from ..aten_lowering import LoweringContext
    from ..compile_environment import Config
    from ..generate_ast import GenerateAST
    from ..helper_function import CodegenInterface
    from ..inductor_lowering import CodegenState
    from ..tile_strategy import TileStrategy

    # Resolves a non-shape-chain leaf at a row-major flat index of its tile.
    LeafResolver = Callable[[LoweringContext, Node, str], "ast.AST | None"]

CUTE_DIM_LOCAL_COORD_META = "cute_dim_local_coords"


def _env_arg(ctx: LoweringContext, node: Node) -> object:
    return ctx.env[node]


def _shape_chain_only_users(node: Node) -> bool:
    if not node.users:
        return False
    return all(
        user.op == "call_function" and is_cute_shape_chain_target(user.target)
        for user in node.users
    )


def _get_tile_shape(
    fake_tensor: torch.Tensor,
    env: CompileEnvironment,
    config: Config,
) -> list[int]:
    """Map a FakeTensor's symbolic dimensions to concrete tile (block) sizes."""
    shape: list[int] = []
    for dim_size in fake_tensor.shape:
        if (extent := _resolve_tile_extent(dim_size, env, config)) is not None:
            shape.append(extent)
            continue
        with contextlib.suppress(Exception):
            shape.append(int(dim_size))
            continue
        shape.append(env.size_hint(dim_size))
    return shape


def _resolve_tile_extent(
    size: int | torch.SymInt,
    env: CompileEnvironment,
    config: Config,
) -> int | None:
    """Evaluate a tile extent from the selected config without SymInt guards.

    Runtime symbols are deliberately left unresolved. Their size hints are not
    evidence for a static iteration count or a thread-to-element assignment.
    """
    if isinstance(size, int):
        return size
    block_id = env.get_block_id(size)
    if block_id is not None:
        value = env.block_sizes[block_id].from_config(config)
        return value if isinstance(value, int) else None
    expr = size.node._expr
    if not isinstance(expr, sympy.Expr):
        return None
    replacements: dict[sympy.Symbol, sympy.Integer] = {}
    for symbol in expr.free_symbols:
        if not isinstance(symbol, sympy.Symbol):
            return None
        block_id = env.get_block_id(symbol)
        if block_id is None:
            return None
        value = env.block_sizes[env.canonical_block_id(block_id)].from_config(config)
        if not isinstance(value, int):
            return None
        replacements[symbol] = sympy.Integer(value)
    resolved = expr.xreplace(replacements)
    return int(resolved) if isinstance(resolved, sympy.Integer) else None


def _resolve_dim_block_id(
    cg: GenerateAST,
    fake_tensor: torch.Tensor,
    dim: int,
) -> int | None:
    """Return the block_id active for a tile dimension, if any.

    Falls back to searching ``env.block_sizes`` for matches with the same
    extent when ``env.get_block_id`` cannot resolve a static int dim directly.
    """
    env = CompileEnvironment.current()
    dim_size = fake_tensor.shape[dim]
    block_id = env.get_block_id(dim_size)
    if block_id is not None:
        return block_id
    if not isinstance(dim_size, (int, torch.SymInt)):
        return None
    grid_state = cg.current_grid_state
    grid_axes = grid_state.block_thread_axes if grid_state is not None else {}
    candidates: list[int] = []
    for info in env.block_sizes:
        if not isinstance(info.size, (int, torch.SymInt)):
            continue
        if not env.known_equal(info.size, dim_size):
            continue
        bid = info.block_id
        if cg.active_device_loops.get(bid) or bid in grid_axes:
            candidates.append(bid)
    if len(candidates) == 1:
        return candidates[0]
    return None


def _strategy_aliases_index_and_offset(strategy: object, block_id: int) -> bool:
    """Return True when ``strategy`` produces an index var aliased to its offset.

    ``PerThreadFlattenedTileStrategy`` (and ``FlattenedTileStrategy``) emit
    ``indices_X = offsets_X`` when collapsing a single block id, so the
    difference ``indices_X - offsets_X`` is always zero. We must derive the
    per-thread coordinate from ``thread_idx`` instead.
    """
    from ..tile_strategy import FlattenedTileStrategy

    if not isinstance(strategy, FlattenedTileStrategy):
        return False
    if len(strategy.block_ids) != 1:
        return False
    return strategy.block_ids[0] == block_id


def _unowned_dim_coord(
    cg: GenerateAST,
    fake_tensor: torch.Tensor,
    dim: int,
    strict: bool,
) -> str:
    """Coordinate for a dim that no thread axis or lane loop distributes.

    Shape chains that merely relabel a per-thread scalar never consume this
    value, so a zero placeholder is harmless there. ``strict`` callers
    (``hl.split``/``hl.join``) select data with it and must fail loudly
    rather than silently read the wrong element.
    """
    size = fake_tensor.shape[dim]
    if strict:
        env = CompileEnvironment.current()
        extent = _resolve_tile_extent(size, env, cg.device_function.config)
        if extent != 1:
            raise exc.BackendUnsupported(
                "cute",
                f"tile dimension {dim} (size {size}) has no thread or lane owner",
            )
    return "cutlass.Int32(0)"


def _get_dim_local_coord(
    cg: GenerateAST,
    fake_tensor: torch.Tensor,
    dim: int,
    *,
    strict: bool = False,
) -> str:
    """Get the current local coordinate expression for a tile dimension.

    Uses the current block-local index when the dimension is active, which
    preserves lane-loop coordinates as well as CUDA thread coordinates.
    """
    block_id = _resolve_dim_block_id(cg, fake_tensor, dim)
    if block_id is None:
        size = fake_tensor.shape[dim]
        if isinstance(size, torch.SymInt) and isinstance(size.node._expr, FloorDiv):
            base, divisor = size.node._expr.args
            env = CompileEnvironment.current()
            block_id = env.get_block_id(base)
            if (
                block_id is not None
                and isinstance(divisor, sympy.Integer)
                and divisor > 0
            ):
                extent = cg.device_function.resolved_block_size(block_id)
                coord = _get_block_local_coord(cg, block_id)
                if (
                    isinstance(extent, int)
                    and extent > 0
                    and extent % int(divisor) == 0
                    and coord is not None
                ):
                    return f"({coord}) // cutlass.Int32({int(divisor)})"
        return _unowned_dim_coord(cg, fake_tensor, dim, strict)
    coord = _get_block_local_coord(cg, block_id)
    if coord is None:
        return _unowned_dim_coord(cg, fake_tensor, dim, strict)
    return coord


def _get_node_dim_local_coord(
    cg: GenerateAST,
    node: Node,
    fake_tensor: torch.Tensor,
    dim: int,
    *,
    strict: bool = False,
) -> str:
    """Get a local coordinate, honoring explicit metadata on shape-chain nodes."""
    coord_meta = node.meta.get(CUTE_DIM_LOCAL_COORD_META)
    if isinstance(coord_meta, (list, tuple)) and dim < len(coord_meta):
        info = coord_meta[dim]
        if isinstance(info, dict):
            coord = _subtile_coord_expr(cg, info)
            if coord is not None:
                return coord
    return _get_dim_local_coord(cg, fake_tensor, dim, strict=strict)


def _subtile_coord_expr(cg: GenerateAST, info: dict[object, object]) -> str | None:
    block_id = info.get("block_id")
    if not isinstance(block_id, int):
        return None
    local_coord = _get_block_local_coord(cg, block_id)
    if local_coord is None:
        return None

    env = CompileEnvironment.current()
    expr = local_coord
    divisor = info.get("divisor", 1)
    if not (isinstance(divisor, (int, torch.SymInt)) and env.known_equal(divisor, 1)):
        divisor_expr = cg.device_function.literal_expr(divisor)
        expr = f"({expr}) // cutlass.Int32({divisor_expr})"

    modulus = info.get("modulus")
    if modulus is not None:
        modulus_expr = cg.device_function.literal_expr(modulus)
        expr = f"({expr}) % cutlass.Int32({modulus_expr})"
    return expr


def _get_block_local_coord(cg: GenerateAST, block_id: int) -> str | None:
    loops = cg.active_device_loops.get(block_id)
    if loops:
        loop_state = loops[-1]
        if _strategy_aliases_index_and_offset(loop_state.strategy, block_id):
            thread_axis = loop_state.block_thread_axes.get(block_id)
            if thread_axis is not None:
                return _grid_local_coord_expr(cg, block_id, thread_axis)
        try:
            offset_var = cg.offset_var(block_id)
        except NotImplementedError:
            thread_axis = loop_state.block_thread_axes.get(block_id)
            if thread_axis is not None:
                return _grid_local_coord_expr(cg, block_id, thread_axis)
            return f"({cg.index_var(block_id)})"
        return f"(({cg.index_var(block_id)}) - ({offset_var}))"

    if cg.current_grid_state is not None:
        thread_axis = cg.current_grid_state.block_thread_axes.get(block_id)
        if thread_axis is not None:
            return _grid_local_coord_expr(cg, block_id, thread_axis)

    return None


def _per_thread_nd_tile_offset(strategy: TileStrategy, block_id: int) -> str | None:
    """Return the uniform tile base for the given index's owning strategy.

    Unlike flattened strategies, PerThreadND keeps the tile offset separate
    from its lane-dependent index, for every blocked/strided vector layout.
    Callers must pass the strategy that owns the index being reconstructed.
    """
    from ..tile_strategy import PerThreadNDTileStrategy

    if isinstance(strategy, PerThreadNDTileStrategy) and block_id in strategy.block_ids:
        return strategy.offset_var(block_id)
    return None


def _grid_local_coord_expr(
    cg: GenerateAST,
    block_id: int,
    thread_axis: int,
) -> str:
    """Return the current grid-local coordinate, including lane-loop offsets."""
    from ..tile_strategy import NDTileStrategy

    loops = cg.active_device_loops.get(block_id)
    grid_state = cg.current_grid_state
    strategy = (
        loops[-1].strategy
        if loops
        else (grid_state.strategy if grid_state is not None else None)
    )
    if isinstance(strategy, NDTileStrategy) and block_id in strategy.block_ids:
        # The emitted index already includes the selected blocked/strided lane
        # layout, vector inner lane, and any CTA slice. Reconstructing these
        # from thread_idx and an outer lane counter loses part of that mapping.
        return f"({strategy.index_var(block_id)}) - ({strategy.offset_var(block_id)})"

    coord = f"cutlass.Int32(cute.arch.thread_idx()[{thread_axis}])"
    if cg.current_grid_state is None:
        return coord

    strategy = cg.current_grid_state.strategy
    if (tile_offset := _per_thread_nd_tile_offset(strategy, block_id)) is not None:
        # The emitted index already includes the selected blocked/strided
        # layout and any vector-lane partition. Reconstructing tid * EPT +
        # lane here silently changes those layouts (including tile.begin).
        return f"(({strategy.index_var(block_id)}) - ({tile_offset}))"
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


# Per-thread pointwise / cast ops whose output element on a given thread is a
# pure function of that same thread's input element(s). A transpose-like shape op
# feeding such an op only relabels the logical layout; the scalar each thread
# holds is unchanged, so the shape op can stay folded into the load index and we
# can look *through* these ops to find the ultimate consumer.
_LAYOUT_PRESERVING_POINTWISE_TARGETS = frozenset(
    {
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add.Scalar,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.sub.Scalar,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.div.Scalar,
        torch.ops.aten.neg.default,
        torch.ops.aten.exp.default,
        torch.ops.aten.exp2.default,
        torch.ops.aten._to_copy.default,
        torch.ops.prims.convert_element_type.default,
    }
)

# Shape ops whose per-thread scalar equals their input's: adding a unit dim or
# broadcasting one leaves each thread's element in place, so a transpose feeding
# them (e.g. the rank-broadcast matmul's ``permute -> unsqueeze -> expand ->
# bmm`` chain) is judged by the ultimate consumer, exactly like the pointwise
# set above.
_LAYOUT_PRESERVING_SHAPE_TARGETS = frozenset(
    {
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.expand.default,
    }
)

# Transpose-like shape ops: a chain of them feeding a matmul relabels the
# operand layout as a whole (``k[tile, :, :].transpose(0, 1).transpose(-2, -1)``
# in the HSTU examples).
_PERMUTE_TARGETS = frozenset(
    {
        torch.ops.aten.permute.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.t.default,
    }
)

# Device loops whose arguments (``args[3]``) become the placeholders of the
# loop body graph (``args[0]``).
_LOOP_TARGETS = frozenset({_for_loop, _for_loop_step})


def _loop_body_placeholders_for(value: Node, loop: Node) -> list[Node]:
    """The body placeholders ``value`` binds to as an argument of ``loop``.

    Matched through the loop node's own argument list, not the body's recorded
    ``node_args``: the graph under codegen may be a copy whose nodes are not
    the traced ones.
    """
    graph_id = loop.args[0]
    args = loop.args[3]
    assert isinstance(graph_id, int) and isinstance(args, (list, tuple))
    placeholders = (
        HostFunction.current()
        .device_ir.graphs[graph_id]
        .graph.find_nodes(op="placeholder")
    )
    return [placeholders[index] for index, arg in enumerate(args) if arg is value]


def _shape_op_needs_materialization(node: Node) -> bool:
    """Return True when non-store consumers need values, not just metadata."""
    from ...language import memory_ops
    from ...language._tracing_ops import _new_var
    from ...language.matmul_ops import dot_scaled as hl_dot_scaled

    matmul_targets = {
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.baddbmm.default,
        hl_dot,
        hl_dot_scaled,
    }
    reduction_names = ("sum", "amax", "amin", "prod", "mean")

    def _feeds_only_matmuls(value: Node, visited: set[Node]) -> bool:
        """Whether every value flowing out of ``value`` ends in a matmul.

        Follows further transposes, ``_new_var`` copies and the placeholders
        of the device loop bodies ``value`` is passed to (a transpose hoisted
        out of the KV loop reaches its bmm that way); a shape query needs no
        values.
        """
        if value in visited:
            return True
        visited.add(value)
        if not value.users:
            return False
        for downstream in value.users:
            if downstream.op != "call_function":
                return False
            target = downstream.target
            if target in matmul_targets or target is torch.ops.aten.sym_size.int:
                continue
            if target in _PERMUTE_TARGETS or target is _new_var:
                if not _feeds_only_matmuls(downstream, visited):
                    return False
                continue
            if target in _LOOP_TARGETS:
                placeholders = _loop_body_placeholders_for(value, downstream)
                if not placeholders or not all(
                    _feeds_only_matmuls(placeholder, visited)
                    for placeholder in placeholders
                ):
                    return False
                continue
            return False
        return True

    def _consumer_needs_materialization(
        value: Node, user: Node, visited: set[Node]
    ) -> bool:
        if user.op != "call_function":
            return True
        if user.target is memory_ops.store:
            return False
        # CuTe matmul fallbacks consume one scalar per thread/lane. In that mode a
        # transpose-like shape op only changes the logical operand layout, not the
        # scalar value held by the current thread.
        if user.target in matmul_targets:
            return False
        # The same holds when the value reaches its matmuls through further
        # transposes, ``_new_var`` copies or a device loop argument: the
        # matmul reads one scalar per thread (or re-reads the operand loads
        # in the synthetic-lane K fold), so the chain relabels the layout as
        # a whole.  Any other consumer on the way reads this op's value, so
        # the op lowers to a concrete per-thread scalar.
        if user.target in _PERMUTE_TARGETS or user.target is _new_var:
            return not _feeds_only_matmuls(user, visited)
        if user.target in _LOOP_TARGETS:
            placeholders = _loop_body_placeholders_for(value, user)
            return not placeholders or not all(
                _feeds_only_matmuls(placeholder, visited)
                for placeholder in placeholders
            )
        target_name = str(user.target)
        if any(name in target_name for name in reduction_names):
            return False
        # Layout-preserving per-thread pointwise/cast/shape ops keep each
        # thread's element in place, so recurse to find the ultimate consumer.
        if (
            user.target in _LAYOUT_PRESERVING_POINTWISE_TARGETS
            or user.target in _LAYOUT_PRESERVING_SHAPE_TARGETS
        ):
            if user in visited:
                return False
            visited.add(user)
            # A pointwise op with no downstream users is not provably safe to
            # fold (we cannot see where its value flows), so stay conservative
            # and materialize, matching the pre-recursion behavior.
            if not user.users:
                return True
            return any(
                _consumer_needs_materialization(user, downstream, visited)
                for downstream in user.users
            )
        return True

    visited: set[Node] = set()
    return any(
        _consumer_needs_materialization(node, user, visited) for user in node.users
    )


def _flat_index_from_coords(
    coords: list[str],
    shape: list[int],
) -> str:
    """Convert ND coordinate expressions to a flat row-major index."""
    ndim = len(shape)
    if ndim == 1:
        return coords[0]
    parts: list[str] = []
    for i in range(ndim):
        stride = 1
        for j in range(i + 1, ndim):
            stride *= shape[j]
        if stride == 1:
            parts.append(f"({coords[i]})")
        else:
            parts.append(f"({coords[i]}) * cutlass.Int32({stride})")
    return " + ".join(parts)


def _coords_from_flat_index(
    flat_index: str,
    shape: list[int],
) -> list[str]:
    """Convert a row-major flat index to ND coordinates."""
    coords: list[str] = []
    ndim = len(shape)
    for i in range(ndim):
        size = shape[i]
        if size == 1:
            coords.append("cutlass.Int32(0)")
            continue
        stride = 1
        for j in range(i + 1, ndim):
            stride *= shape[j]
        if stride == 1:
            coords.append(f"({flat_index}) % cutlass.Int32({size})")
        else:
            coords.append(
                f"(({flat_index}) // cutlass.Int32({stride})) % cutlass.Int32({size})"
            )
    return coords


def _inverse_permute_coords(coords: list[str], perm: list[int]) -> list[str]:
    return [coords[perm.index(i)] for i in range(len(perm))]


def _expand_source_coords(
    output_coords: list[str],
    *,
    source_shape: list[int],
    output_shape: list[int],
) -> list[str] | None:
    rank_delta = len(output_shape) - len(source_shape)
    if rank_delta < 0:
        return None
    source_coords: list[str] = []
    for i, source_extent in enumerate(source_shape):
        output_dim = i + rank_delta
        output_extent = output_shape[output_dim]
        if source_extent == 1:
            source_coords.append("cutlass.Int32(0)")
        elif source_extent == output_extent:
            source_coords.append(output_coords[output_dim])
        else:
            return None
    return source_coords


def _stack_choice_expr(
    inputs: list[ast.AST],
    *,
    selector: str,
) -> ast.AST:
    selected = inputs[-1]
    for i in range(len(inputs) - 2, -1, -1):
        selected = expr_from_string(
            "({then_expr}) if ({selector}) == cutlass.Int32({i}) else ({else_expr})",
            then_expr=inputs[i],
            selector=expr_from_string(selector),
            i=ast.Constant(value=i),
            else_expr=selected,
        )
    return selected


def _resolve_shape_chain_expr(
    ctx: LoweringContext,
    node: Node,
    flat_index: str,
    leaf_resolver: LeafResolver | None = None,
) -> ast.AST | None:
    env = CompileEnvironment.current()
    df = ctx.cg.device_function
    config = df.config
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        resolved = ctx.env.get(node)
        return resolved if isinstance(resolved, ast.AST) else None

    if node.target in (
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.view.default,
    ):
        source = node.args[0]
        if not isinstance(source, Node):
            return None
        return _resolve_shape_chain_expr(ctx, source, flat_index, leaf_resolver)

    def _recurse(source: Node, source_coords: list[str]) -> ast.AST | None:
        source_val = source.meta.get("val")
        if not isinstance(source_val, torch.Tensor):
            return None
        source_flat = _flat_index_from_coords(
            source_coords, _get_tile_shape(source_val, env, config)
        )
        return _resolve_shape_chain_expr(ctx, source, source_flat, leaf_resolver)

    def _coords() -> list[str]:
        return _coords_from_flat_index(flat_index, _get_tile_shape(value, env, config))

    if node.target is torch.ops.aten.permute.default:
        source = node.args[0]
        dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dims")
        if not isinstance(source, Node) or not isinstance(dims, (list, tuple)):
            return None
        perm = [dim for dim in dims if isinstance(dim, int)]
        if len(perm) != len(dims):
            return None
        return _recurse(source, _inverse_permute_coords(_coords(), perm))

    if node.target is torch.ops.aten.expand.default:
        source = node.args[0]
        if not isinstance(source, Node):
            return None
        source_val = source.meta.get("val")
        if not isinstance(source_val, torch.Tensor):
            return None
        output_shape = _get_tile_shape(value, env, config)
        source_shape = _get_tile_shape(source_val, env, config)
        output_coords = _coords_from_flat_index(flat_index, output_shape)
        source_coords = _expand_source_coords(
            output_coords,
            source_shape=source_shape,
            output_shape=output_shape,
        )
        if source_coords is None:
            return None
        source_flat = _flat_index_from_coords(source_coords, source_shape)
        return _resolve_shape_chain_expr(ctx, source, source_flat, leaf_resolver)

    if node.target is torch.ops.aten.transpose.int:
        source = node.args[0]
        dim0 = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim0")
        dim1 = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim1")
        if (
            not isinstance(source, Node)
            or not isinstance(dim0, int)
            or not isinstance(dim1, int)
        ):
            return None
        ndim = len(value.shape)
        dim0 %= ndim
        dim1 %= ndim
        perm = list(range(ndim))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return _recurse(source, _inverse_permute_coords(_coords(), perm))

    if node.target is torch.ops.aten.t.default:
        source = node.args[0]
        if not isinstance(source, Node) or len(value.shape) != 2:
            return None
        coords = _coords()
        return _recurse(source, [coords[1], coords[0]])

    if node.target is torch.ops.aten.unsqueeze.default:
        source = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        if not isinstance(source, Node) or not isinstance(dim, int):
            return None
        dim %= len(value.shape)
        coords = _coords()
        return _recurse(source, coords[:dim] + coords[dim + 1 :])

    if node.target is torch.ops.aten.squeeze.dim:
        source = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        if not isinstance(source, Node) or not isinstance(dim, int):
            return None
        source_val = source.meta.get("val")
        if not isinstance(source_val, torch.Tensor):
            return None
        dim %= source_val.ndim
        coords = _coords()
        if int(source_val.shape[dim]) != 1:
            source_coords = coords
        else:
            source_coords = [*coords[:dim], "cutlass.Int32(0)", *coords[dim:]]
        return _recurse(source, source_coords)

    if node.target is torch.ops.aten.stack.default:
        tensors = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        if not isinstance(tensors, (list, tuple)) or not isinstance(dim, int):
            return None
        if not all(isinstance(tensor, Node) for tensor in tensors):
            return None
        dim %= len(value.shape)
        coords = _coords()
        selector = coords[dim]
        input_exprs: list[ast.AST] = []
        for tensor in tensors:
            assert isinstance(tensor, Node)
            input_val = tensor.meta.get("val")
            if not isinstance(input_val, torch.Tensor):
                return None
            input_coords = coords[:dim] + coords[dim + 1 :]
            input_flat = _flat_index_from_coords(
                input_coords,
                _get_tile_shape(input_val, env, config),
            )
            input_expr = _resolve_shape_chain_expr(
                ctx, tensor, input_flat, leaf_resolver
            )
            if input_expr is None:
                return None
            input_exprs.append(input_expr)
        return _stack_choice_expr(
            input_exprs,
            selector=selector,
        )

    resolved = ctx.env.get(node)
    if isinstance(resolved, CuteShapeChainView):
        return _resolve_shape_chain_expr(ctx, resolved.node, flat_index, leaf_resolver)
    if leaf_resolver is not None:
        return leaf_resolver(ctx, node, flat_index)
    return resolved if isinstance(resolved, ast.AST) else None


def _current_flat_index_for_value(ctx: LoweringContext, value: torch.Tensor) -> str:
    cg = cast("GenerateAST", ctx.cg)
    env = CompileEnvironment.current()
    config = cg.device_function.config
    shape = _get_tile_shape(value, env, config)
    coords = [_get_dim_local_coord(cg, value, i) for i in range(len(shape))]
    return _flat_index_from_coords(coords, shape)


def resolve_cute_shape_chain_value(
    ctx: LoweringContext,
    node: Node,
) -> ast.AST | None:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        resolved = ctx.env.get(node)
        return resolved if isinstance(resolved, ast.AST) else None
    return _resolve_shape_chain_expr(
        ctx, node, _current_flat_index_for_value(ctx, value)
    )


def resolve_cute_shape_chain_value_at(
    state: CodegenState,
    node: Node,
    flat_index: str,
    leaf_resolver: LeafResolver | None = None,
) -> ast.AST | None:
    """Resolve a shape-chain value (reshape/stack/...) at an explicit flat index.

    The store path holds a ``CodegenState`` rather than the ``GraphInterpreter``
    ``LoweringContext`` that the regular value-lowering path uses, but the shape
    chain resolver only needs the ``GenerateAST`` codegen object and the
    fx-node -> argument map, both of which ``CodegenState`` already carries.
    """
    from ..aten_lowering import LoweringContext

    ctx = LoweringContext.__new__(LoweringContext)
    ctx.cg = state.codegen
    ctx.env = state.env
    return _resolve_shape_chain_expr(ctx, node, flat_index, leaf_resolver)


def codegen_cute_virtual_clone(
    ctx: LoweringContext, node: Node
) -> CuteShapeChainView | None:
    """Keep a logical tile copy virtual until its final shape is known.

    A contiguous clone introduced by reshape copies the same logical elements.
    Its already-loaded SSA leaves are immutable, so deferring their selection
    preserves the copy without assigning coordinates to temporary split dims.
    Ordinary materialized clones keep the existing pointwise lowering.
    """
    source = node.args[0]
    if not isinstance(source, Node):
        return None
    value = ctx.env[source]
    if not isinstance(value, CuteShapeChainView):
        return None
    if (
        len(node.args) != 1
        or set(node.kwargs) - {"memory_format"}
        or node.kwargs.get("memory_format")
        not in (None, torch.contiguous_format, torch.preserve_format)
        or not _shape_chain_only_users(node)
    ):
        raise exc.BackendUnsupported(
            "cute", "virtual shape-chain clone requires shape-only consumers"
        )
    return value


def codegen_cute_reshape(ctx: LoweringContext, node: Node) -> object:
    """Codegen for view/reshape on CuTe tiles."""
    from ..generate_ast import GenerateAST
    from .indexing import CutePackedTerms
    from .packed_matmul import virtual_packed_terms

    assert isinstance(ctx.cg, GenerateAST)
    packed_terms = virtual_packed_terms(node)
    if packed_terms is not None:
        values = tuple(ctx.env[term] for term in packed_terms)
        if all(isinstance(value, ast.AST) for value in values):
            return CutePackedTerms(cast("tuple[ast.AST, ...]", values))
    # pyrefly: ignore [bad-argument-type]
    tensor = map_arg(node.args[0], lambda arg: _env_arg(ctx, arg))
    shape_chain = tensor if isinstance(tensor, CuteShapeChainView) else None

    # pyrefly: ignore [missing-attribute]
    input_val = node.args[0].meta["val"]
    output_val = node.meta["val"]
    assert isinstance(input_val, torch.Tensor) and isinstance(output_val, torch.Tensor)

    env = CompileEnvironment.current()
    df = ctx.cg.device_function
    config = df.config

    input_shape = _get_tile_shape(input_val, env, config)
    output_shape = _get_tile_shape(output_val, env, config)

    if input_shape == output_shape and isinstance(tensor, ast.AST):
        return tensor

    source_node = shape_chain.node if shape_chain is not None else node.args[0]
    if shape_chain is not None and _shape_chain_only_users(node):
        return CuteShapeChainView(node)
    if isinstance(source_node, Node):
        output_coords = [
            _get_dim_local_coord(ctx.cg, output_val, i)
            for i in range(len(output_shape))
        ]
        output_flat = _flat_index_from_coords(output_coords, output_shape)
        fused_expr = _resolve_shape_chain_expr(ctx, source_node, output_flat)
        if fused_expr is not None:
            return fused_expr

    if (
        ctx.cg.current_grid_state is not None
        and ctx.cg.current_grid_state.has_lane_loops()
        and not _shape_op_needs_materialization(node)
        and isinstance(tensor, ast.AST)
    ):
        return tensor

    # Adding/removing unit dimensions is a no-op
    input_non_unit = [s for s in input_shape if s != 1]
    output_non_unit = [s for s in output_shape if s != 1]
    if input_non_unit == output_non_unit and isinstance(tensor, ast.AST):
        return tensor

    input_numel = 1
    for s in input_shape:
        input_numel *= s

    if input_numel == 1:
        assert isinstance(tensor, ast.AST)
        return tensor

    dtype_str = env.backend.dtype_str(input_val.dtype)

    smem_ptr = df.new_var("reshape_smem_ptr")
    smem = df.new_var("reshape_smem")
    input_name = df.new_var("reshape_input")
    result = df.new_var("reshaped")

    # Get per-dimension thread coordinates using block_id → thread axis mapping
    src_coords = [
        _get_dim_local_coord(ctx.cg, input_val, i) for i in range(len(input_shape))
    ]
    src_flat = _flat_index_from_coords(src_coords, input_shape)
    output_coords = [
        _get_dim_local_coord(ctx.cg, output_val, i) for i in range(len(output_shape))
    ]
    output_flat = _flat_index_from_coords(output_coords, output_shape)

    ctx.cg.add_statement(
        statement_from_string(
            f"{smem_ptr} = cute.arch.alloc_smem({dtype_str}, {input_numel})"
        )
    )
    ctx.cg.add_statement(
        statement_from_string(
            f"{smem} = cute.make_tensor({smem_ptr}, ({input_numel},))"
        )
    )

    if shape_chain is not None:
        tensor = _resolve_shape_chain_expr(ctx, shape_chain.node, src_flat)
    if not isinstance(tensor, ast.AST):
        raise TypeError(f"Expected AST for CuTe reshape input, got {type(tensor)}")
    ctx.cg.add_statement(statement_from_string(f"{input_name} = {{_inp}}", _inp=tensor))
    ctx.cg.add_statement(statement_from_string(f"{smem}[{src_flat}] = {input_name}"))
    ctx.cg.add_statement(statement_from_string("cute.arch.sync_threads()"))
    ctx.cg.add_statement(statement_from_string(f"{result} = {smem}[{output_flat}]"))

    return expr_from_string(result)


def codegen_cute_permute(ctx: LoweringContext, node: Node) -> object:
    """Codegen for permute/transpose on CuTe tiles: a relabel of each thread's element.

    Helion's tiles are positional arrays, but the SIMT lowering binds every
    block id to one coordinate per thread (and per lane iteration when the
    block runs as a lane loop) and addresses loads, stores and reductions by
    those coordinates.  While every consumer binds each dim of the permuted
    tile to the same block id (``out[tile_n, tile_m] = x[tile_m, tile_n].T``),
    permuting never moves an element between threads: the thread holding
    ``x[i, j]`` is the one that holds ``x.T[j, i]``.  Shape-only consumers keep
    the permute virtual (``CuteShapeChainView``); any other consumer gets the
    thread's own scalar, selected from a virtual chain at this thread's
    coordinates.  A consumer that re-binds a dim to another block id
    (``out[tile_m, tile_n] = x[tile_m, tile_n].T`` with equal block sizes) is
    handled where it happens, see ``rebound_block_dims``.
    """
    from ..generate_ast import GenerateAST

    assert isinstance(ctx.cg, GenerateAST)
    # pyrefly: ignore [bad-argument-type]
    tensor, dims = map_arg(node.args, lambda arg: _env_arg(ctx, arg))
    shape_chain = tensor if isinstance(tensor, CuteShapeChainView) else None

    # pyrefly: ignore [missing-attribute]
    input_val = node.args[0].meta["val"]
    assert isinstance(input_val, torch.Tensor)

    # pyrefly: ignore [not-iterable]
    perm = [*dims]
    assert len(perm) == len(input_val.shape)

    if isinstance(tensor, ast.AST):
        # A concrete per-thread scalar is the permuted tile's element at this
        # thread's coordinates already.
        return tensor
    if shape_chain is None:
        raise TypeError(f"Expected AST for CuTe permute input, got {type(tensor)}")
    if _shape_chain_only_users(node):
        return CuteShapeChainView(node)
    output_val = node.meta["val"]
    if isinstance(output_val, torch.Tensor):
        # This thread's coordinates in the permuted shape; the chain resolver
        # maps them back through the permute (and the virtual chain below it)
        # to the same thread's source coordinates.
        output_shape = _get_tile_shape(
            output_val, CompileEnvironment.current(), ctx.cg.device_function.config
        )
        output_coords = [
            _get_dim_local_coord(ctx.cg, output_val, i, strict=True)
            for i in range(len(output_shape))
        ]
        output_flat = _flat_index_from_coords(output_coords, output_shape)
        fused_expr = _resolve_shape_chain_expr(ctx, node, output_flat)
        if fused_expr is not None:
            return fused_expr
    raise TypeError(f"Unresolved CuTe permute of a virtual shape chain: {node}")


def rebound_block_dims(
    env: CompileEnvironment,
    config: Config,
    value: torch.Tensor,
    consumer_sizes: Sequence[int | torch.SymInt],
    consumer_block_ids: Sequence[int | None],
    *,
    lower_rank_by_block_id: bool,
) -> list[tuple[int, int, int]]:
    """Dims of ``value`` that a consumer binds to a different block id.

    ``consumer_sizes`` / ``consumer_block_ids`` describe the consumer's dims
    (the dims of a pointwise op's result, the tiles of a store subscript).
    Helion's tiles are positional arrays: an operand of the consumer's rank
    is right-aligned (``PointwiseLowering._check_block_broadcast_compatibility``)
    and a store or atomic value is right-aligned whatever its rank
    (``tl.store`` / ``tl.atomic_*`` receive it unexpanded), so the consumer
    reads the element at the same *position*.  Only a lower-rank pointwise
    operand is placed by ``TileDispatch.broadcast_expand_dims``
    (``lower_rank_by_block_id``): every source dim is matched to a result dim,
    a tile dim to the one with its canonical block id, a non-tile dim to an
    unused non-tile dim of equal size or else the last unused dim, and the
    source dims then occupy the matched positions *in order* (the expansion
    only inserts ``None``, it never permutes), so ``b[tile1, tile0]`` meets a
    ``[tile0, tile1, tile2]`` result positionally while ``b[tile1]`` is its
    middle-dim broadcast.  The SIMT lowering keeps each thread's element by
    block-id coordinates, so a dim bound to another block id must be
    exchanged between threads (or the program refused).  Returns
    ``(value_dim, value_block_id, consumer_block_id)`` for each such dim.
    Dims that are not block ids (static extents; a reduction dim is a block
    id here, cute pads factory tensors so every slice gets one) or that
    broadcast (stride 0, or a block whose extent under ``config`` is 1, on
    either side) bind nothing.  A dim bound to a block id already on another
    axis of the same tensor is rejected earlier (``check_repeated_block_ids``).
    """
    consumer_canonical = [
        None if block_id is None else env.canonical_block_id(block_id)
        for block_id in consumer_block_ids
    ]
    offset = len(consumer_block_ids) - value.ndim
    positions = [dim + offset for dim in range(value.ndim)]
    if lower_rank_by_block_id and 0 < value.ndim < len(consumer_block_ids):
        matched: list[int] = []
        used: set[int] = set()
        for size in value.shape:
            block_id = env.get_block_id(size)
            canonical = None if block_id is None else env.canonical_block_id(block_id)
            match = None
            if canonical is not None:
                match = next(
                    (
                        position
                        for position, candidate in enumerate(consumer_canonical)
                        if position not in used and candidate == canonical
                    ),
                    None,
                )
            else:
                match = next(
                    (
                        position
                        for position, candidate in enumerate(consumer_canonical)
                        if position not in used
                        and candidate is None
                        and env.known_equal(size, consumer_sizes[position])
                    ),
                    None,
                )
            if match is None:
                match = max(
                    position
                    for position in range(len(consumer_block_ids))
                    if position not in used
                )
            matched.append(match)
            used.add(match)
        positions = sorted(matched)
    rebound: list[tuple[int, int, int]] = []
    for dim, size in enumerate(value.shape):
        position = positions[dim]
        if position < 0 or value.stride(dim) == 0:
            continue
        block_id = env.get_block_id(size)
        if block_id is None or _resolve_tile_extent(size, env, config) == 1:
            continue
        consumer_block_id = consumer_block_ids[position]
        if (
            consumer_block_id is None
            or block_extent(env, config, consumer_block_id) == 1
        ):
            continue
        if env.canonical_block_id(block_id) != consumer_canonical[position]:
            rebound.append((dim, block_id, consumer_block_id))
    return rebound


def block_extent(env: CompileEnvironment, config: Config, block_id: int) -> int | None:
    """``block_id``'s extent under ``config`` (``None`` while it is symbolic)."""
    extent = env.block_sizes[env.canonical_block_id(block_id)].from_config(config)
    return extent if isinstance(extent, int) else None


def tensor_dim_block_ids(
    env: CompileEnvironment, sizes: Sequence[int | torch.SymInt]
) -> list[int | None]:
    """The block id each of ``sizes`` names, ``None`` for non-block extents."""
    return [
        env.get_block_id(size) if isinstance(size, torch.SymInt) else None
        for size in sizes
    ]


def _subscript_slot_dims(
    state: CodegenState, tensor: torch.Tensor, subscript: Sequence[object]
) -> tuple[list[int | torch.SymInt], list[int | None]]:
    """The dims of ``tensor[subscript]`` with the block id each is addressed
    by, walked the way ``_cute_index_exprs`` addresses them.

    A tile binds its block id; a slice binds the block the index expressions
    resolve for it (``_cute_resolve_active_slice_block_id`` with the raw slice
    size, the block ids used so far, each resolved slice block then used), a
    size-1 dim and a scalar index bind nothing, a tensor indexer binds its
    own dims' block ids.
    """
    env = CompileEnvironment.current()
    used_block_ids = {
        block_id
        for idx in subscript
        if isinstance(idx, torch.SymInt)
        if (block_id := env.get_block_id(idx)) is not None
    }
    sizes: list[int | torch.SymInt] = []
    block_ids: list[int | None] = []
    tensor_indexers = [idx for idx in subscript if isinstance(idx, torch.Tensor)]
    should_broadcast = env.should_broadcast_tensor_indexers([*subscript])
    tensor_dim = 0
    for position, idx in enumerate(subscript):
        if idx is None:
            sizes.append(1)
            block_ids.append(None)
            continue
        unit_dim = tensor_dim < tensor.ndim and env.known_equal(
            tensor.shape[tensor_dim], 1
        )
        tile_info = _get_tile_with_offset_info(idx, state.fx_node, position)
        if tile_info is not None and tile_info.block_size is not None:
            # ``_cute_index_exprs`` takes its unit-dim "0" branch first and
            # does not use the block then.
            if not unit_dim:
                used_block_ids.add(tile_info.block_id)
            sizes.append(tile_info.resolved_block_size_var(env))
            block_ids.append(None if unit_dim else tile_info.block_id)
            tensor_dim += 1
        elif isinstance(idx, torch.SymInt):
            # As ``compute_shape``: only a block-size symbol (a tile) keeps a
            # dim; ``tile.begin`` and other scalar symbols index it away.
            symbol = _symint_expr(idx)
            origin = (
                HostFunction.current().expr_to_origin.get(symbol)
                if isinstance(symbol, sympy.Symbol)
                else None
            )
            if origin is not None and isinstance(origin.origin, BlockSizeOrigin):
                sizes.append(idx)
                block_ids.append(None if unit_dim else origin.origin.block_id)
            tensor_dim += 1
        elif isinstance(idx, int):
            tensor_dim += 1
        elif isinstance(idx, torch.Tensor):
            if not should_broadcast:
                indexer_sizes = [*env.tensor_indexer_dims(idx)]
            elif idx is tensor_indexers[0]:
                indexer_sizes = [*env.tensor_indexer_broadcast_shape(tensor_indexers)]
            else:
                indexer_sizes = []
            sizes.extend(indexer_sizes)
            block_ids.extend(tensor_dim_block_ids(env, indexer_sizes))
            tensor_dim += 1
        elif isinstance(idx, slice):
            dim_size = tensor.shape[tensor_dim]
            size = dim_size if idx == slice(None) else compute_slice_size(idx, dim_size)
            block_id = None
            if not env.known_equal(size, 1):
                block_id = _cute_resolve_active_slice_block_id(
                    state, size, used_block_ids
                )
                if block_id is not None:
                    used_block_ids.add(block_id)
            sizes.append(size)
            block_ids.append(block_id)
            tensor_dim += 1
        else:
            raise exc.InvalidIndexingType(idx)
    return sizes, block_ids


def subscript_rebound_block_dims(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: Sequence[object],
    value: torch.Tensor,
    *,
    leading_sizes: Sequence[int | torch.SymInt] = (),
) -> list[tuple[int, int, int]]:
    """``rebound_block_dims`` of ``value`` written through ``tensor[subscript]``
    (``leading_sizes``: the dims a stack tensor's pointer table puts in front).

    The value is right-aligned whatever its rank, as ``tl.store`` and
    ``tl.atomic_*`` receive it, against the blocks the index expressions
    address each dim with (``_subscript_slot_dims``), so an exchanged value is
    read at the slice's block coordinate (the load's reduction block for
    ``out[tile_m, :] = x[:, tile_m]``, ``tile_n`` for the second slice of
    ``out[tile_m, :, :] = x[tile_m, :, tile_n]``).
    """
    env = CompileEnvironment.current()
    slot_sizes, slot_block_ids = _subscript_slot_dims(state, tensor, subscript)
    leading = [*leading_sizes]
    return rebound_block_dims(
        env,
        state.device_function.config,
        value,
        [*leading, *slot_sizes],
        [*tensor_dim_block_ids(env, leading), *slot_block_ids],
        lower_rank_by_block_id=False,
    )


def describe_rebound_block_dims(rebound: Sequence[tuple[int, int, int]]) -> str:
    return ", ".join(
        f"dim {dim} (block id {value_block_id}) bound to block id {consumer_block_id}"
        for dim, value_block_id, consumer_block_id in rebound
    )


# Custom lowerings that combine several tile operands per thread and so need
# ``check_pointwise_rebound_block_ids`` like a PointwiseLowering: where, stack,
# hl.join, hl.inline_asm_elementwise, the tuple forms of hl.reduce /
# hl.associative_scan (whose combine function pairs the inputs' elements) and
# the matmuls with an accumulator, which the SIMT fallback adds per thread.
REBOUND_CHECK_TARGETS = frozenset(
    {
        torch.ops.aten.where.self,
        torch.ops.aten.stack.default,
        join,
        inline_asm_elementwise,
        _reduce,
        _associative_scan,
        torch.ops.aten.addmm.default,
        torch.ops.aten.baddbmm.default,
        hl_dot,
        torch.ops.aten.gather.default,
    }
)

_MATMUL_WITH_ACCUMULATOR = frozenset(
    {torch.ops.aten.addmm.default, torch.ops.aten.baddbmm.default, hl_dot}
)


def _rebound_operands(node: Node) -> list[Node]:
    """The operands of ``node`` whose elements are combined per thread.

    A matmul's lhs / rhs have their own layout logic; only its accumulator
    (``addmm(acc, ...)``, ``baddbmm(acc, ...)``, ``hl.dot(..., acc=acc)``) is
    added element-wise to the product.
    """
    if node.target in _MATMUL_WITH_ACCUMULATOR:
        if node.target is hl_dot:
            acc = node.args[2] if len(node.args) > 2 else node.kwargs.get("acc")
        else:
            acc = node.args[0] if node.args else None
        return [acc] if isinstance(acc, Node) else []
    if node.target is torch.ops.aten.gather.default:
        # The index is read per thread and selects along ``dim`` of the input
        # the thread addresses by its own block coordinates.
        index = node.args[2] if len(node.args) > 2 else node.kwargs.get("index")
        return [index] if isinstance(index, Node) else []
    return list(node.all_input_nodes)


def _rebound_consumer_dims(
    node: Node,
) -> tuple[list[int | torch.SymInt], bool] | None:
    """The dims an operand of ``node`` is compared against, and whether a
    lower-rank operand is placed by block id.

    Only a PointwiseLowering expands a lower-rank operand by block id
    (``TileDispatch.broadcast_expand_dims``).  ``tl.where``, the inline asm,
    ``hl.join``, the combine functions of a tuple reduce / scan and the
    matmul accumulator receive their operands unexpanded and so right-align
    them: a stack's or join's operands meet the result without the stacked
    (last) dim, a tuple reduce or scan's inputs meet each other, a
    tuple-result inline asm's operands meet its (broadcast) outputs.  A
    single-input reduce or scan has nothing to re-bind against.
    """
    value = node.meta.get("val")
    target = node.target
    if target is torch.ops.aten.stack.default or target is join:
        if not isinstance(value, torch.Tensor):
            return None
        sizes: list[int | torch.SymInt] = [*value.shape]
        if target is join:
            dim = value.ndim - 1
        else:
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
            assert isinstance(dim, int)
            dim %= value.ndim
        del sizes[dim]
        return sizes, False
    if target is _reduce or target is _associative_scan:
        is_tuple_input = (
            node.args[4] if len(node.args) > 4 else node.kwargs.get("is_tuple_input")
        )
        if not is_tuple_input:
            return None
        first = next(
            (
                operand.meta["val"]
                for operand in node.all_input_nodes
                if isinstance(operand.meta.get("val"), torch.Tensor)
            ),
            None,
        )
        return None if first is None else ([*first.shape], False)
    if target is inline_asm_elementwise:
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        return ([*value.shape], False) if isinstance(value, torch.Tensor) else None
    if target is torch.ops.aten.gather.default:
        source = node.args[0] if node.args else None
        source_val = source.meta.get("val") if isinstance(source, Node) else None
        if not isinstance(source_val, torch.Tensor):
            return None
        return [*source_val.shape], False
    if not isinstance(value, torch.Tensor):
        return None
    if target is torch.ops.aten.where.self or target in _MATMUL_WITH_ACCUMULATOR:
        return [*value.shape], False
    return [*value.shape], True


def check_pointwise_rebound_block_ids(
    cg: CodegenInterface, node: Node, *, defer_tcgen05_epilogues: bool = True
) -> None:
    """Refuse an element-wise op that binds an operand dim to another block id
    than the dims it is combined with (``t + t.T`` with equal block sizes,
    ``torch.where(c, t, t.T)``, ``hl.join(t, t.T)``, an inline asm or tuple
    reduce over ``t`` and ``t.T``, ``torch.gather(t, 1, idx.T)``, or a
    lower-rank pointwise operand whose tile dims are in another order than
    the result's).

    The op would need the operand element held by another thread; the SIMT
    lowering combines each thread's own scalars.  Unequal extents are the
    ``ShapeMismatch`` the Triton backend raises.  An op on a tcgen05 matmul's
    epilogue chain is checked from the store instead
    (``run_deferred_rebound_checks``), after the epilogue classifier has had
    its say: its diagnostic names the supported epilogue forms.
    """
    from ..generate_ast import GenerateAST

    consumer = _rebound_consumer_dims(node)
    if consumer is None or not consumer[0]:
        return
    consumer_sizes, lower_rank_by_block_id = consumer
    df = cg.device_function
    env = CompileEnvironment.current()
    config = df.config
    result_block_ids = tensor_dim_block_ids(env, consumer_sizes)
    if node.target is torch.ops.aten.gather.default:
        # Along the gather dim the index has the result's extent and its
        # values select the position; only the other dims must align.
        gather_dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        assert isinstance(gather_dim, int)
        result_block_ids[gather_dim % len(result_block_ids)] = None
    for operand in _rebound_operands(node):
        operand_val = operand.meta.get("val")
        if not isinstance(operand_val, torch.Tensor):
            continue
        rebound = rebound_block_dims(
            env,
            config,
            operand_val,
            consumer_sizes,
            result_block_ids,
            lower_rank_by_block_id=lower_rank_by_block_id,
        )
        if not rebound:
            continue
        cute_state = df.cute_state
        if (
            defer_tcgen05_epilogues
            and isinstance(cg, GenerateAST)
            and cute_state.matmul_fx_nodes
        ):
            if cute_state.rebound_inner_outputs_index is None:
                cute_state.rebound_inner_outputs_index = (
                    build_inner_outputs_index_from_graphs(cg.codegen_graphs)
                )
            if reach_matmul_anchors(
                node,
                target_fx_nodes=cute_state.matmul_fx_nodes,
                inner_outputs_by_graph_id=cute_state.rebound_inner_outputs_index,
            ):
                cute_state.deferred_rebound_pointwise_nodes.append(node)
                return
        for dim, _operand_block_id, result_block_id in rebound:
            operand_extent = _resolve_tile_extent(operand_val.shape[dim], env, config)
            result_extent = block_extent(env, config, result_block_id)
            if operand_extent != result_extent:
                raise exc.ShapeMismatch(
                    f"operand {operand.name} dim {dim} of extent {operand_extent}",
                    f"{node.target} result tile of extent {result_extent}",
                )
        raise exc.BackendUnsupported(
            "cute",
            f"{node.target} reads operand {operand.name} (shape "
            f"{list(operand_val.shape)}) at the position of another "
            f"block's lane: {describe_rebound_block_dims(rebound)}; the "
            "SIMT lowering combines each thread's own elements",
        )


def run_deferred_rebound_checks(
    cg: CodegenInterface, nodes: Collection[Node] | None = None
) -> None:
    """Check the pointwise ops deferred from a tcgen05 epilogue chain.

    Called by the store lowering once the epilogue classifier has accepted
    or passed on the chain (a rejected chain raises its own diagnostic
    first), for the deferred ops among ``nodes`` (the chain's ancestors) so
    that another chain's store still gets its classifier's diagnostic, and
    once more for everything at the end of the root codegen for chains that
    no store consumed.
    """
    cute_state = cg.device_function.cute_state
    pending = cute_state.deferred_rebound_pointwise_nodes
    selected = [node for node in pending if nodes is None or node in nodes]
    cute_state.deferred_rebound_pointwise_nodes = [
        node for node in pending if node not in selected
    ]
    for node in selected:
        check_pointwise_rebound_block_ids(cg, node, defer_tcgen05_epilogues=False)


def cute_lane_loops_active(cg: GenerateAST) -> bool:
    """Whether the statements being emitted run inside a lane loop."""
    grid_state = cg.current_grid_state
    if grid_state is not None and grid_state.has_lane_loops():
        return True
    return any(
        isinstance(state, DeviceLoopState) and bool(state.lane_loops)
        for states in cg.active_device_loops.values()
        for state in states
    )


def check_memory_mask_rebound(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: Sequence[object],
    mask_val: object,
    *,
    what: str,
    leading_sizes: Sequence[int | torch.SymInt] = (),
) -> None:
    """Refuse a load or store ``extra_mask`` bound to another block id than
    the subscript's dims: Triton ANDs it positionally into the index masks,
    the per-thread lowering would test another lane's element."""
    if not isinstance(mask_val, torch.Tensor) or mask_val.ndim == 0:
        return
    rebound = subscript_rebound_block_dims(
        state, tensor, subscript, mask_val, leading_sizes=leading_sizes
    )
    if rebound:
        raise exc.BackendUnsupported(
            "cute",
            f"{what} mask of {list(mask_val.shape)} re-binds "
            f"{describe_rebound_block_dims(rebound)}; the mask must be held by "
            "the thread that owns the addressed lane",
        )


def store_rebound_dims(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: Sequence[object],
    *,
    leading_sizes: Sequence[int | torch.SymInt] = (),
) -> list[tuple[int, int, int]]:
    """The dims of the stored value that ``tensor[subscript]`` binds to another
    block id (``subscript_rebound_block_dims``), with unequal extents the
    ``ShapeMismatch`` the Triton backend raises; a store mask bound that way
    is refused (a mask is not exchanged)."""
    env = CompileEnvironment.current()
    config = state.device_function.config
    check_memory_mask_rebound(
        state,
        tensor,
        subscript,
        state.proxy_arg(3),
        what="store",
        leading_sizes=leading_sizes,
    )
    value_val = state.proxy_arg(2)
    if not isinstance(value_val, torch.Tensor) or value_val.ndim == 0:
        return []
    rebound = subscript_rebound_block_dims(
        state, tensor, subscript, value_val, leading_sizes=leading_sizes
    )
    if not rebound:
        return []
    value_shape = _get_tile_shape(value_val, env, config)
    for dim, _value_block_id, slot_block_id in rebound:
        slot_extent = block_extent(env, config, slot_block_id)
        if value_shape[dim] != slot_extent:
            raise exc.ShapeMismatch(
                f"stored tile dim {dim} of extent {value_shape[dim]}",
                f"subscript tile of extent {slot_extent}",
            )
    return rebound


def tcgen05_rebound_store_error(
    state: CodegenState, rebound: Sequence[tuple[int, int, int]]
) -> exc.BackendUnsupported:
    """A re-binding store of a tcgen05 matmul epilogue: the accumulator tile
    is not one element per thread, so there is nothing to exchange.  Raised
    by the store lowering where a tcgen05 store path would otherwise accept
    the chain (its own classifier's diagnostic wins where it rejects)."""
    value_val = state.proxy_arg(2)
    assert isinstance(value_val, torch.Tensor)
    return exc.BackendUnsupported(
        "cute",
        f"store of {list(value_val.shape)} re-binds "
        f"{describe_rebound_block_dims(rebound)} of a tcgen05 matmul epilogue; "
        "the accumulator tile is not held one element per thread, so it cannot "
        "be exchanged between threads (store the accumulator in its own layout)",
    )


def codegen_cute_store_rebound_value(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: Sequence[object],
    value: ast.AST,
    rebound: Sequence[tuple[int, int, int]],
) -> ast.AST:
    """Exchange ``value`` between threads when ``tensor[subscript]`` re-binds
    its dims (``rebound``, from ``store_rebound_dims``), so that each thread
    stores the element at its own position.

    ``out[tile_m, tile_n] = x[tile_m, tile_n].T`` with equal block sizes
    stores, at position ``(i, j)`` of the slot, the value's element ``(i, j)``:
    ``x[j, i]`` of the tile, held by the thread with the swapped coordinates.
    Every thread stages its element at its position in the value's shape (its
    coordinates by the value's block ids) and, after a barrier, reads the
    element at its position in the slot (its coordinates by the subscript's
    block ids).  Both accesses are predicated on the coordinates lying inside
    the tile: a launch widened for a sibling root loop has surplus threads
    whose coordinates exceed the extent (their masks keep them off the
    store, but not off the buffer).  The whole tile must be resident across
    the thread block at one barrier, so the exchange is refused inside lane
    loops.
    """
    from ..generate_ast import GenerateAST

    value_val = state.proxy_arg(2)
    assert isinstance(value_val, torch.Tensor) and rebound
    cg = state.codegen
    assert isinstance(cg, GenerateAST)
    env = CompileEnvironment.current()
    df = state.device_function
    description = describe_rebound_block_dims(rebound)
    value_shape = _get_tile_shape(value_val, env, df.config)
    if cute_lane_loops_active(cg):
        raise exc.BackendUnsupported(
            "cute",
            f"store of {list(value_val.shape)} re-binds {description} inside a "
            "lane loop; the exchange needs the whole thread block at one barrier",
        )

    own_coords = [
        _get_dim_local_coord(cg, value_val, dim, strict=True)
        for dim in range(value_val.ndim)
    ]
    slot_coords = list(own_coords)
    for dim, _value_block_id, slot_block_id in rebound:
        coord = _get_block_local_coord(cg, slot_block_id)
        if coord is None:
            raise exc.BackendUnsupported(
                "cute",
                f"store re-binds {description} to a block without a thread coordinate",
            )
        slot_coords[dim] = coord
    numel = 1
    for extent in value_shape:
        numel *= extent
    dtype_str = env.backend.dtype_str(value_val.dtype)
    smem_ptr = df.new_var("rebind_smem_ptr")
    smem = df.new_var("rebind_smem")
    staged = df.new_var("rebind_staged")
    reads = df.new_var("rebind_reads")
    result = df.new_var("rebound")
    cg.add_statement(
        statement_from_string(
            f"{smem_ptr} = cute.arch.alloc_smem({dtype_str}, {numel})"
        )
    )
    cg.add_statement(
        statement_from_string(f"{smem} = cute.make_tensor({smem_ptr}, ({numel},))")
    )
    cg.add_statement(
        statement_from_string(
            f"{staged} = {_in_tile_predicate(own_coords, value_shape)}"
        )
    )
    cg.add_statement(
        statement_from_string(
            f"{reads} = {_in_tile_predicate(slot_coords, value_shape)}"
        )
    )
    cg.add_statement(
        statement_from_string(
            f"if {staged}:\n"
            f"    {smem}[{_flat_index_from_coords(own_coords, value_shape)}] = {{value}}",
            value=value,
        )
    )
    cg.add_statement(statement_from_string("cute.arch.sync_threads()"))
    cg.add_statement(
        statement_from_string(
            f"{result} = {smem}[{_flat_index_from_coords(slot_coords, value_shape)}]"
            f" if {reads} else {dtype_str}(0)"
        )
    )
    # Unconditional: an enclosing device or persistent loop runs the store
    # again and must not overwrite the buffer before every read of it.
    cg.add_statement(statement_from_string("cute.arch.sync_threads()"))
    return expr_from_string(result)


def _in_tile_predicate(coords: Sequence[str], shape: Sequence[int]) -> str:
    """``coords`` all below their extents (surplus threads of a widened launch
    sit at or beyond the extent)."""
    return " and ".join(
        f"({coord}) < cutlass.Int32({extent})"
        for coord, extent in zip(coords, shape, strict=True)
    )
