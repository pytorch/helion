"""General-purpose CuTe emitter helpers for memory access sites.

Index-expression, address, access-region and mask builders shared by the CuTe
load/store/atomic/scan lowerings: active block-id resolution, rolled-reduction
replicas, ``_cute_index_exprs`` / ``_cute_combined_mask`` / ``_cute_access_regions``
and the scalar pointer / load expression forms.  Moved out of
``helion/language/memory_ops.py``; no config reads happen here.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

import sympy
import torch

from .... import exc
from ...ast_extension import expr_from_string
from ...ast_extension import statement_from_string
from ...ast_read_writes import HELION_ACCESS_REGIONS_ATTR
from ...compile_environment import CompileEnvironment
from ...compile_environment import _symint_expr
from ...host_function import HostFunction
from ...indexing_strategy import TileWithOffsetInfo
from ...indexing_strategy import _get_tile_with_offset_info
from ...utils import compute_slice_size
from ...variable_origin import BlockSizeOrigin
from ...variable_origin import GridOrigin
from ...variable_origin import TileBeginOrigin
from ...variable_origin import TileCountOrigin
from ...variable_origin import TileEndOrigin
from ...variable_origin import TileIdOrigin
from ..access_regions import access_address
from ..access_regions import block_size_symbol
from ..access_regions import tile_begin_symbol
from ..cache_policy_loads import CachePolicy
from ..iota_utils import cute_arange_index_source
from ..state import cute_codegen_state
from ..tile_ops import cute_masked_block_end

if TYPE_CHECKING:
    from ...host_function import SymbolOrigin
    from ...inductor_lowering import CodegenState
    from ...tile_strategy import DeviceLoopOrGridState
    from ...tile_strategy import LoopDimInfo


def _maybe_get_symbol_origin(idx: object) -> SymbolOrigin | None:
    if not isinstance(idx, torch.SymInt):
        return None
    expr = _symint_expr(idx)
    if expr is None:
        return None
    return HostFunction.current().expr_to_origin.get(expr)


def _matching_block_ids(env: CompileEnvironment, size: object) -> list[int]:
    """Find all block_ids that match the given dimension size."""
    candidates: list[int] = []
    if isinstance(size, (int, torch.SymInt)):
        if (direct := env.get_block_id(size)) is not None:
            candidates.append(direct)
    if not isinstance(size, (int, torch.SymInt)):
        return candidates
    for info in env.block_sizes:
        if not isinstance(info.size, (int, torch.SymInt)):
            continue
        if not env.known_equal(info.size, size):
            continue
        if info.block_id not in candidates:
            candidates.append(info.block_id)
    return candidates


def _cute_remap_block_id(state: CodegenState, block_id: int) -> int:
    """Apply the active matmul-operand block-id remap, if any.

    Used while re-materializing a matmul operand load so its contraction
    dimension is indexed by the active contraction block instead of the
    loop-invariant block it was originally lowered with.  Returns *block_id*
    unchanged when no remap is active.
    """
    remap = state.device_function.cute_state.tcgen05.matmul_operand_block_remap
    if not remap:
        return block_id
    return remap.get(block_id, block_id)


def _cute_index_override(state: CodegenState, block_id: int) -> str | None:
    """Return a raw index-expression override for *block_id*, if active.

    Applied after ``_cute_remap_block_id``.  When set (only while
    re-materializing the rhs of a static-MN-collapse baddbmm), the operand's
    free (N) axis is indexed by this serial-loop variable instead of the shared
    M thread index, and masking for that axis is suppressed.
    """
    override = state.device_function.cute_state.tcgen05.matmul_operand_index_override
    if not override:
        return None
    return override.get(_cute_remap_block_id(state, block_id))


def _cute_rolled_replica(state: CodegenState, block_id: int) -> str | None:
    """The index variable of the chunk lane loop standing in for a rolled
    reduction block outside the reduction's own loop on the current grid
    (``_cute_make_rolled_replica``), or None."""
    grid_state = state.codegen.current_grid_state
    if grid_state is None or not grid_state.rolled_reduction_replicas:
        return None
    return grid_state.rolled_reduction_replicas.get(
        _cute_remap_block_id(state, block_id)
    )


def _cute_make_rolled_replica(
    state: CodegenState, size: object, used_block_ids: set[int]
) -> int | None:
    """Give a full-slice access over a rolled reduction's extent, outside that
    reduction's loop, its own lane loop over the block.

    The rolled strategy walks its block in chunks of its thread count inside
    ``for roffset``: the block's index exists only there, and a statement
    after the loop spanning the whole block (the zero store of a dQ row next
    to the row's ``sum(o * do)`` at head dim 128, where the two share the
    block of that extent) had no coordinate for it.  The replica registers,
    once per block and grid, a synthetic lane loop with the reduction's
    thread geometry -- ``index = thread_idx()[axis] + lane * threads`` over
    ``numel / threads`` lanes -- and the lane distribution places only the
    statements reading it inside, as it does for any other full-slice block.
    Only a block whose extent the threads divide qualifies: the thread count
    is a power of two, so that is every power-of-two extent, and the device
    IR attaches a store over any other extent to a replay of the rolled loop
    (which walks the tail chunk with the loop's own bound), so no such store
    reaches here and the replica needs no mask.  Returns the block id it
    stands for, or None when no unused rolled block of that static extent
    qualifies at the grid level (the caller then raises as before).
    """
    from ...reduction_strategy import LoopedReductionStrategy

    env = CompileEnvironment.current()
    grid_state = state.codegen.current_grid_state
    if grid_state is None:
        return None
    fn = state.device_function
    for block_id in _matching_block_ids(env, size):
        if block_id in used_block_ids:
            continue
        if _cute_rolled_replica(state, block_id) is not None:
            return block_id
        strategy = fn.tile_strategy.block_id_to_strategy.get_any(block_id)
        if (
            not isinstance(strategy, LoopedReductionStrategy)
            or strategy._thread_count <= 0
            or strategy._cute_reduction_vec_width != 1
            or not strategy._claims_thread_axis()
        ):
            continue
        numel = env.block_sizes[block_id].numel
        if isinstance(numel, sympy.Integer):
            numel = int(numel)
        if not isinstance(numel, int) or numel <= 0:
            continue
        threads = strategy._thread_count
        if numel % threads:
            continue
        axis = strategy._get_thread_axis()
        lane_var = fn.new_var(f"synthetic_lane_{block_id}")
        index_var = fn.new_var(f"rindex_lane_{block_id}")
        grid_state.add_lane_loop(block_id, lane_var, numel // threads)
        grid_state.lane_setup_statements.append(
            statement_from_string(
                f"{index_var} = cutlass.Int32(cute.arch.thread_idx()[{axis}])"
                f" + cutlass.Int32({lane_var}) * {threads}"
            )
        )
        grid_state.rolled_reduction_replicas[block_id] = index_var
        return block_id
    return None


def _cute_active_index_var(state: CodegenState, block_id: int) -> str | None:
    if (override := _cute_index_override(state, block_id)) is not None:
        return override
    block_id = _cute_remap_block_id(state, block_id)
    loops = state.codegen.active_device_loops.get(block_id)
    if loops:
        return loops[-1].strategy.index_var(block_id)
    grid_state = state.codegen.current_grid_state
    if grid_state is not None and block_id in grid_state.block_ids:
        return grid_state.strategy.index_var(block_id)
    if (replica := _cute_rolled_replica(state, block_id)) is not None:
        return replica
    return None


def _cute_resolve_active_slice_block_id(
    state: CodegenState,
    size: object,
    used_block_ids: set[int],
    *,
    tile_block_ids: frozenset[int] | set[int] = frozenset(),
) -> int | None:
    """Resolve the same slice axis for pointer arithmetic and atomic ownership.

    A block no other position of the subscript indexes comes first.  Failing
    that, a block another SLICE position already took may be returned: two
    slices of one tensor whose extents coincide share the reduction block of
    that extent (a matmul operand's K and N at 128), and the consumer handles
    the pair (the matmul's operand index override).  A block a TILE position
    took (``tile_block_ids``) never is: the tile's block coordinates the
    thread's row, so a slice bound to it would run along the diagonal.  With
    the batch extent equal to the head dim (``static_shapes``) and the
    slice's own block's index not active at the store (a rolled reduction's
    lives only inside its loop; a persistent one's lanes closed before the
    store), the batch tile's block was the one active candidate and was
    returned here: the store took the batch tile's index as its column and
    wrote one element per row, silently.  With no candidate the caller
    falls back (a chunk lane loop for a rolled reduction's block) or rejects
    the site ("indexing dimension is not active in this scope").
    """
    env = CompileEnvironment.current()
    active_candidates = [
        block_id
        for block_id in _matching_block_ids(env, size)
        if block_id not in tile_block_ids
        and _cute_active_index_var(state, block_id) is not None
    ]
    active_unused_candidates = [
        block_id for block_id in active_candidates if block_id not in used_block_ids
    ]
    if len(active_unused_candidates) == 1:
        return active_unused_candidates[0]
    if len(active_candidates) == 1:
        return active_candidates[0]
    if len(active_unused_candidates) > 1:
        reduction_unused = [
            block_id
            for block_id in active_unused_candidates
            if env.block_sizes[block_id].reduction
        ]
        if len(reduction_unused) == 1:
            return reduction_unused[0]
    if len(active_candidates) > 1:
        reduction_active = [
            block_id
            for block_id in active_candidates
            if env.block_sizes[block_id].reduction
        ]
        if len(reduction_active) == 1:
            return reduction_active[0]
    return None


def _cute_active_mask_var(state: CodegenState, block_id: int) -> str | None:
    if _cute_index_override(state, block_id) is not None:
        return None
    block_id = _cute_remap_block_id(state, block_id)
    loops = state.codegen.active_device_loops.get(block_id)
    if loops:
        return loops[-1].strategy.mask_var(block_id)
    return None


def _cute_active_local_coord(
    state: CodegenState, block_id: int, *, honor_override: bool = False
) -> str | None:
    """The per-lane coordinate of ``block_id`` along its thread axis, or None
    when no active loop or grid maps the block onto a thread axis.

    ``honor_override`` is the mask builder's variant: an operand index
    override (``_cute_index_override``) replaces the block's index by a serial
    loop variable, which has no lane coordinate, so the mask builder reports
    None for it; the address builder keeps the thread coordinate and lets the
    override win in ``_cute_active_index_var``.  The two variants are kept
    apart on purpose: unifying them changes the lifted ``tile_begin`` of an
    overridden operand.
    """
    from ..cute_reshape import _grid_local_coord_expr

    if honor_override and _cute_index_override(state, block_id) is not None:
        return None
    block_id = _cute_remap_block_id(state, block_id)
    loops = state.codegen.active_device_loops.get(block_id)
    if loops:
        thread_axis = loops[-1].block_thread_axes.get(block_id)
        if thread_axis is not None:
            return _grid_local_coord_expr(state.codegen, block_id, thread_axis)
    grid_state = state.codegen.current_grid_state
    if grid_state is not None:
        thread_axis = grid_state.block_thread_axes.get(block_id)
        if thread_axis is not None:
            return _grid_local_coord_expr(state.codegen, block_id, thread_axis)
    return None


def _cute_tile_begin_var_or_lifted(
    state: CodegenState, block_id: int, *, fallback: str, honor_override: bool
) -> str:
    """The uniform tile base of ``block_id``: the active loop's
    ``tile_begin_var`` (not ``offset_var``: on the per-thread flattened
    strategy the offset is already the per-element index), else the lifted
    ``(index) - (lane coordinate)`` named ``tile_begin_N``, else the index
    itself, else ``fallback``.

    Lift-order contract: the ``tile_begin_N`` names are numbered by
    ``GenerateAST.lift`` in the order the address and mask builders reach the
    dimensions, so a builder must resolve the dimensions in subscript order
    and the address builder must run before the mask builder of the same
    access (the render corpus pins the numbering).
    """
    block_id = _cute_remap_block_id(state, block_id)
    loops = state.codegen.active_device_loops.get(block_id)
    if loops:
        return state.codegen.tile_begin_var(block_id)
    global_index = _cute_active_index_var(state, block_id)
    local_coord = _cute_active_local_coord(
        state, block_id, honor_override=honor_override
    )
    if global_index is not None and local_coord is not None:
        return state.codegen.lift(
            expr_from_string(f"({global_index}) - ({local_coord})"),
            dce=True,
            prefix="tile_begin",
        ).id
    if global_index is not None:
        return global_index
    return fallback


def _cute_unique_graph_block_id(state: CodegenState) -> int | None:
    # ``device_ir`` imports this package's emitters while it initializes, so
    # the graph-info class is imported here, not at module level.
    from ...device_ir import ForLoopGraphInfo

    fx_node = state.fx_node
    if fx_node is None:
        return None
    graph_block_ids = [
        graph_info.block_ids
        for graph_info in state.codegen.codegen_graphs
        if graph_info.graph is fx_node.graph
        and isinstance(graph_info, ForLoopGraphInfo)
    ]
    if len(graph_block_ids) != 1 or len(graph_block_ids[0]) != 1:
        return None
    (block_id,) = graph_block_ids[0]
    return block_id


def _maybe_codegen_cute_packed_affine_lhs_load(
    state: CodegenState,
    tensor: torch.Tensor,
    subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
) -> object | None:
    from ....language.matmul_ops import dot
    from ..indexing import CutePackedAffineLoad
    from ..indexing import match_cute_affine_range_iota
    from ..indexing import match_cute_stack_reshape_rhs

    fx_node = state.fx_node
    if (
        fx_node is None
        or len(fx_node.users) != 1
        or len(subscript) not in (2, 3)
        or len(fx_node.args) < 2
    ):
        return None

    fx_subscript = fx_node.args[1]
    if not isinstance(fx_subscript, (list, tuple)) or len(fx_subscript) != len(
        subscript
    ):
        return None
    range_node = fx_subscript[-1]
    if not isinstance(range_node, torch.fx.Node):
        return None
    affine_range = match_cute_affine_range_iota(range_node)
    if affine_range is None:
        return None

    user = next(iter(fx_node.users))
    if user.op != "call_function" or user.target not in {
        dot,
        torch.ops.aten.bmm.default,
        torch.ops.aten.baddbmm.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
    }:
        return None

    rhs_index = (
        2
        if user.target in (torch.ops.aten.addmm.default, torch.ops.aten.baddbmm.default)
        else 1
    )
    rhs_arg = user.args[rhs_index]
    if not isinstance(rhs_arg, torch.fx.Node):
        return None
    packed_rhs = match_cute_stack_reshape_rhs(rhs_arg)
    if packed_rhs is None:
        return None
    _, factor = packed_rhs
    if factor != affine_range.factor:
        return None

    packed_block_id = _cute_unique_graph_block_id(state)
    if packed_block_id is None:
        return None
    packed_index = _cute_active_index_var(state, packed_block_id)
    if packed_index is None:
        return None

    leading_subscript = [*subscript[:-1]]
    row_index_exprs = _cute_index_exprs(
        state,
        leading_subscript,
        tensor=tensor,
        inactive_slice_expr="None",
        inactive_singleton_slice_expr="0",
    )
    if len(row_index_exprs) != len(leading_subscript):
        return None

    tensor_name = state.device_function.tensor_arg(tensor).name
    mask_terms: list[str] = []
    row_mask = _cute_combined_mask(state, leading_subscript, extra_mask, tensor=tensor)
    if row_mask is not None:
        mask_terms.append(row_mask)
    if packed_mask := _cute_active_mask_var(state, packed_block_id):
        mask_terms.append(f"({packed_mask})")
    mask_expr = " and ".join(mask_terms) if mask_terms else None
    zero = CompileEnvironment.current().backend.dtype_str(tensor.dtype)
    terms: list[ast.AST] = []
    for offset in range(factor):
        index_expr = ", ".join(
            [
                *row_index_exprs,
                f"cutlass.Int32({factor}) * ({packed_index}) + cutlass.Int32({offset})",
            ]
        )
        term = expr_from_string(f"{tensor_name}[{index_expr}]")
        if mask_expr is not None:
            term = expr_from_string(
                f"({{value}} if {mask_expr} else {zero}(0))",
                value=term,
            )
        terms.append(term)
    return CutePackedAffineLoad(tuple(terms))


def _cute_wrap_negative_index(idx: object, dim_size: int | torch.SymInt) -> object:
    """``idx`` with a negative constant counted back from ``dim_size`` (``x[-1]``)."""
    if isinstance(idx, int) and not isinstance(idx, bool) and idx < 0:
        return idx + dim_size
    return idx


@dataclasses.dataclass(frozen=True)
class CuteDimResolution:
    """How one subscript position addresses one tensor dimension.

    ``kind`` names the branch of the resolver that produced ``index_expr``:
    ``unit`` (a size-1 tensor dim, addressed by ``0``), ``tile`` (a tile
    with an offset, ``begin + offset + lane``), ``symint`` (a block index,
    a tile attribute such as ``tile.begin`` / ``tile.id``, or a symbolic
    scalar), ``int``, ``tensor`` (a gather index, lifted to ``index_N``),
    ``slice`` (a full slice bound to an active block), ``replica`` (a full
    slice over a rolled reduction's extent, served by a chunk lane loop),
    ``singleton_slice`` / ``inactive_slice`` (a full slice over a size-1 /
    inactive dim, the caller's substitute expression), ``partial_slice``
    and ``partial_singleton`` (``start + step * lane`` forms).  ``block_id``
    is the block the position is bound to, when there is one.
    """

    pos: int
    tensor_dim: int
    kind: str
    block_id: int | None
    index_expr: str


@dataclasses.dataclass(frozen=True)
class CuteSubscriptResolution:
    """The per-dimension records of one resolved subscript, in subscript
    order (``None`` positions, which add a unit axis, have no record).

    ``_cute_index_exprs`` is its projection onto the index expressions.
    The mask builder (``_cute_combined_mask``) and the region builder
    (``_cute_access_regions``) are deliberately *not* projections of these
    records: a full slice is bound to a block by
    ``_cute_resolve_active_slice_block_id`` here, by "the first matching
    block that is not yet seen and has a mask variable" in the mask builder
    and to no block at all in the region builder, and
    ``_cute_resolved_load_mask`` bridges the first two by handing the mask
    builder a different subscript.  Lift order: the resolver runs the
    positions in subscript order and lifts ``tile_begin_N`` / ``index_N``
    names as it reaches them; the mask builder of the same access runs
    after it (``AccessSite``), so the numbering in a render is fixed.
    """

    dims: tuple[CuteDimResolution, ...]

    @property
    def index_exprs(self) -> list[str]:
        return [dim.index_expr for dim in self.dims]


def _cute_index_exprs(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...] | None = None,
    tensor: torch.Tensor | None = None,
    *,
    inactive_slice_expr: str | None = None,
    inactive_singleton_slice_expr: str | None = None,
    for_store: bool = False,
) -> list[str]:
    """The index expression of every tensor dimension ``subscript`` addresses
    (``_cute_resolve_subscript`` projected onto its expressions)."""
    return _cute_resolve_subscript(
        state,
        subscript,
        ast_subscript,
        tensor,
        inactive_slice_expr=inactive_slice_expr,
        inactive_singleton_slice_expr=inactive_singleton_slice_expr,
        for_store=for_store,
    ).index_exprs


def _cute_resolve_subscript(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    ast_subscript: list[object] | tuple[object, ...] | None = None,
    tensor: torch.Tensor | None = None,
    *,
    inactive_slice_expr: str | None = None,
    inactive_singleton_slice_expr: str | None = None,
    for_store: bool = False,
) -> CuteSubscriptResolution:
    """Resolve ``subscript`` against ``tensor`` into per-dimension records.

    ``inactive_slice_expr`` / ``inactive_singleton_slice_expr`` are what a
    full slice over an inactive / size-1 dimension reads (a load passes
    ``"None"`` / ``"0"`` for the ``T[...]`` subscript form; without them
    such a dimension is rejected); ``for_store`` skips the broadcast
    resolution of dynamic sizes (a valid store's dim covers its tile).
    """
    env = CompileEnvironment.current()

    def symint_index_expr(idx: torch.SymInt, used_block_ids: set[int]) -> str:
        expr = _symint_expr(idx)
        if expr is not None:
            origin_info = HostFunction.current().expr_to_origin.get(expr)
            if origin_info is not None and isinstance(origin_info.origin, GridOrigin):
                if type(origin_info.origin) is not GridOrigin:
                    block_id = origin_info.origin.block_id
                    loop_info = active_loop_info(block_id)
                    begin_var = tile_begin_expr(block_id, loop_info)
                    block_size_var = (
                        state.device_function.block_size_var(block_id) or "1"
                    )
                    if isinstance(origin_info.origin, TileBeginOrigin):
                        return begin_var
                    if isinstance(origin_info.origin, TileEndOrigin):
                        if loop_info is not None and loop_info.end_var_name is not None:
                            return env.backend.minimum_expr(
                                f"({begin_var}) + ({block_size_var})",
                                loop_info.end_var_name,
                            )
                        return f"({begin_var}) + ({block_size_var})"
                    if isinstance(origin_info.origin, TileCountOrigin):
                        # The tiles of the whole iteration space, counted
                        # from its first index (``tile_count``'s common
                        # codegen), not from this program's own tile.
                        iteration_begin = (
                            loop_info.begin_var_name
                            if loop_info is not None
                            and loop_info.begin_var_name is not None
                            else "0"
                        )
                        end_var = (
                            loop_info.end_var_name
                            if loop_info is not None
                            and loop_info.end_var_name is not None
                            else f"({iteration_begin}) + ({block_size_var})"
                        )
                        extent = f"({end_var}) - ({iteration_begin})"
                        return env.backend.cdiv_expr(
                            extent, block_size_var, is_device=True
                        )
                    if isinstance(origin_info.origin, TileIdOrigin):
                        if block_size_var == "1":
                            return begin_var
                        return f"({begin_var}) // ({block_size_var})"
                    return state.sympy_expr(expr)
        block_id = env.get_block_id(idx)
        if block_id is not None:
            used_block_ids.add(block_id)
            return index_var_for_block_id(block_id, idx)
        if expr is not None:
            return state.sympy_expr(expr)
        raise exc.BackendUnsupported("cute", f"unlowerable symbolic index: {idx}")

    def active_loop_info(block_id: int) -> LoopDimInfo | None:
        block_id = _cute_remap_block_id(state, block_id)
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            return loops[-1].block_id_to_info.get(block_id)
        grid_state = state.codegen.current_grid_state
        if grid_state is not None:
            return grid_state.block_id_to_info.get(block_id)
        return None

    def tile_begin_expr(block_id: int, loop_info: LoopDimInfo | None) -> str:
        begin_var = "0"
        if loop_info is not None and loop_info.begin_var_name is not None:
            begin_var = loop_info.begin_var_name
        return _cute_tile_begin_var_or_lifted(
            state, block_id, fallback=begin_var, honor_override=False
        )

    def active_index_var(block_id: int) -> str | None:
        return _cute_active_index_var(state, block_id)

    def active_local_coord(block_id: int) -> str | None:
        return _cute_active_local_coord(state, block_id)

    def index_var_for_block_id(block_id: int, size: object) -> str:
        if (idx_var := active_index_var(block_id)) is not None:
            return idx_var

        raise exc.BackendUnsupported(
            "cute",
            (
                "indexing dimension is not active in this scope "
                f"(block_id={block_id}, size={size})"
            ),
        )

    def local_coord_for_block_id(block_id: int, begin_var: str) -> str | None:
        if (local_coord := active_local_coord(block_id)) is not None:
            return local_coord
        if (idx_var := active_index_var(block_id)) is not None:
            return f"({idx_var}) - ({begin_var})"
        return None

    def tile_with_offset_index_expr(tile_info: TileWithOffsetInfo) -> str:
        block_id = tile_info.block_id
        begin_var = tile_begin_expr(block_id, active_loop_info(block_id))
        local_coord = local_coord_for_block_id(block_id, begin_var)
        if local_coord is None:
            raise exc.BackendUnsupported(
                "cute",
                (
                    "indexing dimension is not active in this scope "
                    f"(block_id={block_id})"
                ),
            )
        offset_expr = state.device_function.literal_expr(tile_info.offset)
        return f"({begin_var}) + cutlass.Int32({offset_expr}) + ({local_coord})"

    used_block_ids = {
        block_id
        for idx in subscript
        if isinstance(idx, torch.SymInt)
        if (block_id := env.get_block_id(idx)) is not None
    }
    # The blocks the tile positions take: never a slice's (see
    # ``_cute_resolve_active_slice_block_id``).
    tile_block_ids = set(used_block_ids)
    dims: list[CuteDimResolution] = []
    tensor_dim = 0

    def resolved(kind: str, block_id: int | None, index_expr: str) -> None:
        dims.append(CuteDimResolution(pos, tensor_dim, kind, block_id, index_expr))

    for pos, idx in enumerate(subscript):
        ast_idx = None
        if ast_subscript is not None:
            ast_idx = ast_subscript[pos]
        if idx is None:
            continue
        tile_info = _get_tile_with_offset_info(
            idx, getattr(state, "fx_node", None), pos
        )
        # A store target never needs to broadcast (a valid store's dim covers
        # the tile's extent), so its dynamic sizes are not compared with 1.
        if tensor is not None and tensor_dim < tensor.ndim and not for_store:
            if tile_info is not None:
                env.resolve_broadcast_dim(
                    tensor.shape[tensor_dim], tile_info.block_id, tile_info.offset
                )
            elif (
                isinstance(idx, torch.SymInt)
                and (block_id := env.get_block_id(idx)) is not None
            ):
                env.resolve_broadcast_dim(tensor.shape[tensor_dim], block_id)
        if (
            tensor is not None
            and tensor_dim < tensor.ndim
            and env.known_equal(tensor.shape[tensor_dim], 1)
            and not (isinstance(idx, slice) and idx == slice(None))
        ):
            resolved("unit", None, "0")
            tensor_dim += 1
            continue
        if tile_info is not None and tile_info.block_size is not None:
            used_block_ids.add(tile_info.block_id)
            tile_block_ids.add(tile_info.block_id)
            resolved("tile", tile_info.block_id, tile_with_offset_index_expr(tile_info))
            tensor_dim += 1
            continue
        if isinstance(idx, torch.SymInt):
            resolved(
                "symint", env.get_block_id(idx), symint_index_expr(idx, used_block_ids)
            )
            tensor_dim += 1
        elif isinstance(idx, int):
            if idx >= 0:
                resolved("int", None, str(idx))
            elif tensor is None:
                raise exc.BackendUnsupported("cute", "negative index without tensor")
            else:
                wrapped = _cute_wrap_negative_index(idx, tensor.shape[tensor_dim])
                resolved("int", None, state.device_function.literal_expr(wrapped))
            tensor_dim += 1
        elif isinstance(idx, torch.Tensor):
            from ..indexing import CuteAffineRangeIndex

            if isinstance(ast_idx, CuteAffineRangeIndex):
                raise exc.BackendUnsupported(
                    "cute",
                    "affine hl.arange() indexing is only supported in CuTe packed-matmul load fusion",
                )
            if not isinstance(ast_idx, ast.AST):
                raise exc.BackendUnsupported(
                    "cute", f"tensor index without AST at position {pos}"
                )
            lifted = state.codegen.lift(ast_idx, dce=True, prefix="index")
            index_dtype = env.backend.dtype_str(env.index_dtype)
            resolved("tensor", None, f"{index_dtype}({lifted.id})")
            tensor_dim += 1
        elif isinstance(idx, slice) and idx == slice(None):
            if tensor is None:
                raise exc.BackendUnsupported("cute", "slice indexing without tensor")
            dim_size = tensor.shape[tensor_dim]
            block_id = _cute_resolve_active_slice_block_id(
                state, dim_size, used_block_ids, tile_block_ids=tile_block_ids
            )
            if block_id is not None:
                idx_var = active_index_var(block_id)
                assert idx_var is not None
                used_block_ids.add(block_id)
                resolved("slice", block_id, idx_var)
                tensor_dim += 1
                continue
            if inactive_singleton_slice_expr is not None and env.known_equal(
                dim_size, 1
            ):
                resolved("singleton_slice", None, inactive_singleton_slice_expr)
                tensor_dim += 1
                continue
            # The slice's block is a rolled reduction's and its loop is not
            # open here: stand a chunk lane loop in for it.
            block_id = _cute_make_rolled_replica(state, dim_size, used_block_ids)
            if block_id is not None:
                idx_var = active_index_var(block_id)
                assert idx_var is not None
                used_block_ids.add(block_id)
                resolved("replica", block_id, idx_var)
                tensor_dim += 1
                continue
            if inactive_slice_expr is None:
                raise exc.BackendUnsupported(
                    "cute",
                    (
                        "indexing dimension is not active in this scope "
                        f"(tensor_dim={pos}, size={dim_size})"
                    ),
                )
            resolved("inactive_slice", None, inactive_slice_expr)
            tensor_dim += 1
        elif isinstance(idx, slice) and (
            idx.step is None or (isinstance(idx.step, int) and idx.step > 0)
        ):
            # Partial slice (e.g. :16, 16:, 5:20, or 1:16:3): element ``j`` of
            # the slice is ``start + step * j``.
            if tensor is None:
                raise exc.BackendUnsupported(
                    "cute", "partial slice indexing without tensor"
                )
            dim_size = tensor.shape[tensor_dim]
            slice_size = compute_slice_size(idx, dim_size)
            start = idx.start if idx.start is not None else 0
            step = idx.step if idx.step is not None else 1
            block_id = _cute_resolve_active_slice_block_id(
                state, slice_size, used_block_ids, tile_block_ids=tile_block_ids
            )
            if block_id is not None:
                idx_var = active_index_var(block_id)
                assert idx_var is not None
                used_block_ids.add(block_id)
                if step != 1:
                    idx_var = f"cutlass.Int32({step}) * {idx_var}"
                if start != 0:
                    start_expr = state.device_function.literal_expr(start)
                    idx_var = f"({start_expr} + {idx_var})"
                resolved("partial_slice", block_id, idx_var)
                tensor_dim += 1
                continue
            if inactive_singleton_slice_expr is not None and env.known_equal(
                slice_size, 1
            ):
                start_expr = state.device_function.literal_expr(start)
                resolved(
                    "partial_singleton",
                    None,
                    f"({start_expr} + {inactive_singleton_slice_expr})",
                )
                tensor_dim += 1
                continue
            raise exc.BackendUnsupported(
                "cute",
                (
                    "partial slice dimension is not active in this scope "
                    f"(tensor_dim={pos}, size={slice_size})"
                ),
            )
        elif isinstance(idx, slice):
            raise exc.BackendUnsupported(
                "cute", f"slices of step {idx.step} are not supported"
            )
        else:
            raise exc.BackendUnsupported("cute", f"index type: {type(idx)}")
    return CuteSubscriptResolution(tuple(dims))


_CuteAccessRegions = tuple[tuple[sympy.Expr, sympy.Expr] | None, ...]


@dataclasses.dataclass
class AccessSite:
    """The facts of one load or store site the lowering computes once and
    every emitter of the site reads: the tensor's device argument name, the
    per-dimension index expressions (``_cute_index_exprs``), the access
    regions (``_cute_access_regions``) and the mask (``_cute_combined_mask``
    or its load projection).

    Lift-order contract: the address builder lifts ``tile_begin_N`` /
    ``index_N`` names and the mask builder ``mask_N`` / ``index_mask_N``
    names through ``GenerateAST.lift``, numbered in call order, so a site is
    always built as ``tensor_name`` -> ``index_exprs`` -> ``regions`` ->
    ``mask_expr`` (``tensor_arg`` and the size / stride lookups register
    kernel arguments in call order as well).  The render corpus pins the
    numbering.  ``index_exprs`` and ``mask_expr`` are mutable: the
    special-purpose store rewrites (hl.split coordinates, topk lanes) adjust
    them before the general-purpose emitter runs.
    """

    tensor_name: str
    index_exprs: list[str]
    regions: _CuteAccessRegions | None
    mask_expr: str | None


def _cute_access_regions(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    tensor: torch.Tensor,
) -> _CuteAccessRegions | None:
    """The elements a subscript of ``tensor`` covers, per tensor dimension.

    The ``[begin, end)`` bounds of the tile program's access along each
    dimension, over every thread and lane: a tile's ``[begin, begin + block)``
    (its begin one symbol per block id, which a ``tile.begin`` index reads
    too), a slice's ``[start, start + size)``, a scalar's ``[k, k + 1)``; None
    for a dimension indexed by a tensor or by a tile bound computed at run
    time (``tile.end``).  The lane-loop distribution's barrier analysis
    (``cute/lane_loops/barriers.py``) needs no barrier between accesses
    whose regions are apart along some dimension.  None when the subscript
    does not resolve against the tensor's dimensions.
    """
    env = CompileEnvironment.current()

    def begin_symbol(block_id: int) -> sympy.Expr:
        # The symbol names the loop instance iterating the block: two loops
        # over one block id have unrelated begins (``tile_begin_symbol``).
        block_id = _cute_remap_block_id(state, block_id)
        loops = state.codegen.active_device_loops.get(block_id)
        return tile_begin_symbol(block_id, loops[-1] if loops else None)

    def block_size(block_id: int) -> sympy.Expr:
        block_id = _cute_remap_block_id(state, block_id)
        return block_size_symbol(state.device_function.block_size_var(block_id))

    def substitute(expr: sympy.Expr) -> sympy.Expr | None:
        """``expr`` over the region symbols; None when a tile bound computed at run time is left."""
        substitutions: dict[sympy.Basic, sympy.Expr] = {}
        for symbol in expr.free_symbols:
            origin_info = HostFunction.current().expr_to_origin.get(symbol)
            if origin_info is None:
                continue
            if isinstance(origin_info.origin, TileBeginOrigin):
                substitutions[symbol] = begin_symbol(origin_info.origin.block_id)
            elif isinstance(origin_info.origin, BlockSizeOrigin):
                substitutions[symbol] = block_size(origin_info.origin.block_id)
            elif isinstance(origin_info.origin, GridOrigin):
                return None
        return expr.xreplace(substitutions)

    def scalar(value: object) -> sympy.Expr | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return sympy.Integer(value)
        if isinstance(value, sympy.Expr):
            return value
        if not isinstance(value, torch.SymInt):
            return None
        expr = _symint_expr(value)
        if expr is None:
            return None
        return substitute(expr)

    def tile_end_interval(expr: sympy.Expr) -> tuple[sympy.Expr, sympy.Expr] | None:
        """The bounds of a scalar index ``tile.end + k``.

        A tile's end (``min(begin + block, end)``) lies in ``(begin, begin +
        block]``, so the index lies in ``[begin + 1 + k, begin + block + k]``:
        an interval, not a point, that the loop's later iterations leave
        behind.  Only an index moving one for one with the end qualifies.
        """
        ends = [
            (symbol, origin_info.origin.block_id)
            for symbol in expr.free_symbols
            if (origin_info := HostFunction.current().expr_to_origin.get(symbol))
            is not None
            and isinstance(origin_info.origin, TileEndOrigin)
        ]
        if len(ends) != 1:
            return None
        ((end, block_id),) = ends
        if sympy.diff(expr, end) != 1:
            return None
        begin = begin_symbol(block_id)
        low = substitute(expr.xreplace({end: sympy.Add(begin, 1)}))
        high = substitute(expr.xreplace({end: sympy.Add(begin, block_size(block_id))}))
        if low is None or high is None:
            return None
        return low, sympy.Add(high, 1)

    regions: list[tuple[sympy.Expr, sympy.Expr] | None] = []
    tensor_dim = 0
    for pos, idx in enumerate(subscript):
        if idx is None:
            continue
        if tensor_dim >= tensor.ndim:
            return None
        dim_size = tensor.shape[tensor_dim]
        tensor_dim += 1
        if env.known_equal(dim_size, 1) and not (
            isinstance(idx, slice) and idx == slice(None)
        ):
            regions.append((sympy.Integer(0), sympy.Integer(1)))
            continue
        idx = _cute_wrap_negative_index(idx, dim_size)
        tile_info = _get_tile_with_offset_info(
            idx, getattr(state, "fx_node", None), pos
        )
        if tile_info is not None and tile_info.block_size is not None:
            offset = scalar(tile_info.offset)
            size = scalar(tile_info.block_size)
            if offset is None or size is None:
                regions.append(None)
            else:
                begin = sympy.Add(begin_symbol(tile_info.block_id), offset)
                regions.append((begin, sympy.Add(begin, size)))
            continue
        if isinstance(idx, torch.SymInt):
            expr = _symint_expr(idx)
            origin_info = (
                None
                if expr is None
                else HostFunction.current().expr_to_origin.get(expr)
            )
            if expr is not None and (interval := tile_end_interval(expr)) is not None:
                regions.append(interval)
            elif (
                origin_info is not None
                and isinstance(origin_info.origin, GridOrigin)
                and type(origin_info.origin) is not GridOrigin
            ):
                if isinstance(origin_info.origin, TileBeginOrigin):
                    begin = begin_symbol(origin_info.origin.block_id)
                    regions.append((begin, sympy.Add(begin, 1)))
                else:
                    regions.append(None)
            elif (block_id := env.get_block_id(idx)) is not None:
                begin = begin_symbol(block_id)
                regions.append((begin, sympy.Add(begin, block_size(block_id))))
            else:
                value = scalar(idx)
                regions.append(None if value is None else (value, sympy.Add(value, 1)))
        elif isinstance(idx, int) and not isinstance(idx, bool):
            regions.append((sympy.Integer(idx), sympy.Integer(idx + 1)))
        elif isinstance(idx, slice) and (idx.step is None or idx.step == 1):
            start = scalar(idx.start if idx.start is not None else 0)
            size = scalar(compute_slice_size(idx, dim_size))
            regions.append(
                None
                if start is None or size is None
                else (start, sympy.Add(start, size))
            )
        else:
            regions.append(None)
    if tensor_dim != tensor.ndim:
        return None
    return tuple(regions)


def _cute_tag_access_regions(
    node: ast.AST, tensor_name: str, regions: _CuteAccessRegions | None
) -> None:
    """Record ``regions`` on the load and store calls of ``node`` addressing ``tensor_name``.

    The address is the receiver of a ``.load()`` / ``.store()`` call or the
    first argument of any other call, ``cute.arch.load`` included
    (``access_address``); a call whose address only holds the tensor inside
    a nested call (a cast around a load) is not an access.
    """
    if regions is None:
        return
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        address = access_address(call)
        if address is None:
            continue
        pending: list[ast.AST] = [address]
        while pending:
            child = pending.pop()
            if isinstance(child, ast.Call):
                continue
            if (
                isinstance(child, ast.Attribute)
                and child.attr == "iterator"
                and isinstance(child.value, ast.Name)
                and child.value.id == tensor_name
            ):
                setattr(call, HELION_ACCESS_REGIONS_ATTR, regions)
                break
            pending.extend(ast.iter_child_nodes(child))


def _cute_index_tuple(index_exprs: list[str]) -> str:
    if len(index_exprs) == 1:
        return f"({index_exprs[0]},)"
    return f"({', '.join(index_exprs)})"


def _cute_scalar_pointer_expr(tensor_name: str, index_exprs: list[str]) -> str:
    if not index_exprs:
        # Rank-0 tensor (``scale[()]``): the base pointer is the element.
        return f"({tensor_name}.iterator)"
    env = CompileEnvironment.current()
    index_dtype = env.index_type()
    offset = " + ".join(
        f"({index_dtype}({index}) * {index_dtype}({tensor_name}.layout.stride[{dim}]))"
        for dim, index in enumerate(index_exprs)
    )
    return f"({tensor_name}.iterator + {offset})"


def _cute_scalar_load_expr(
    tensor_name: str,
    index_exprs: list[str],
    dtype: torch.dtype,
    *,
    eviction_suffix: str = "",
) -> str:
    if "None" in index_exprs:
        return f"{tensor_name}[{', '.join(index_exprs)}]"
    if dtype in (torch.float4_e2m1fn_x2, torch.float8_e4m3fn):
        return (
            f"cute.arch.load({_cute_scalar_pointer_expr(tensor_name, index_exprs)}, "
            "cutlass.Uint8)"
        )
    # The L2 policy helpers only exist in vector form: their marker renders
    # no keyword.
    kwargs = CachePolicy.from_suffix(eviction_suffix).load_kwargs
    if kwargs and dtype.itemsize >= 4 and dtype is not torch.bool:
        # ``(ptr).load()`` has no cache-hint kwargs; route hinted scalar
        # sites through ``cute.arch.load`` instead.  Sub-32-bit dtypes stay
        # on the plain form: their scalar load lowers to ``nvvm.load.ext``,
        # which rejects cache modifiers ("Unsupported FP type for
        # ExtLoadOp"); their hints apply on the vectorized forms instead.
        from ...compile_environment import CompileEnvironment

        dtype_str = CompileEnvironment.current().backend.dtype_str(dtype)
        return (
            f"cute.arch.load({_cute_scalar_pointer_expr(tensor_name, index_exprs)}, "
            f"{dtype_str}{kwargs})"
        )
    loaded = f"{_cute_scalar_pointer_expr(tensor_name, index_exprs)}.load()"
    if dtype in (torch.uint8, torch.uint32):
        # CuTe pointer offsets reconstruct signless MLIR integers as signed
        # Numeric types. Restore the logical unsigned dtype before a masked
        # zero joins the load or a later conversion extends its high bit.
        from ...compile_environment import CompileEnvironment

        dtype_str = CompileEnvironment.current().backend.dtype_str(dtype)
        return f"{dtype_str}({loaded})"
    return loaded


def _cute_is_tile_scalar(
    env: CompileEnvironment, idx: torch.SymInt, block_id: int
) -> bool:
    """Whether ``idx``, mapped to ``block_id``, is a tile attribute or grid index.

    ``CompileEnvironment.get_block_id`` answers for the block's own size
    symbol, the tile, and for the symbols derived from it (``tile.begin``,
    ``tile.id``, a grid index), which are scalars.
    """
    return _symint_expr(idx) != _symint_expr(env.block_sizes[block_id].var)


def _cute_tile_id_thread_extent_is_complete(
    state: CodegenState, owner: DeviceLoopOrGridState, block_id: int
) -> bool:
    """A tile-level bound cannot discard a live physical/serial-lane mask."""
    from ...tile_strategy import DeviceLoopState
    from ...tile_strategy import PerThreadNDTileStrategy

    # A serial device loop may still emit its store inside each element-lane
    # iteration. Keep the original mask until that store's placement is proved:
    # an empty tail lane must not overwrite a preceding nonempty lane's result.
    if isinstance(owner, DeviceLoopState) and block_id in owner.lane_loop_blocks:
        return False
    axis = owner.block_thread_axes.get(block_id)
    if axis is None:
        return True
    extent = owner.thread_axis_sizes.get(axis)
    if extent is None:
        return False
    if isinstance(owner.strategy, PerThreadNDTileStrategy):
        extent = owner.strategy.thread_extent_for_masking(block_id, extent)
    # The dispatch includes ordinary sibling branches/roots. A wider sibling
    # therefore keeps the original element mask and its existing arbitration.
    planned = state.device_function.tile_strategy.thread_block_dims()
    return 0 <= axis < len(planned) and 0 < planned[axis] <= extent


def _cute_combined_mask(
    state: CodegenState,
    subscript: list[object] | tuple[object, ...],
    extra_mask: ast.AST | None,
    tensor: torch.Tensor | None = None,
    *,
    include_tensor_index_masks: bool = True,
    for_store: bool = False,
) -> str | None:
    # Destination stores may need an element-owner mask for arbitration.
    # A source/reloaded tile scalar must remain uniform across those lanes.
    env = CompileEnvironment.current()
    terms: list[str] = []

    def mask_var_for_block_id(block_id: int) -> str | None:
        return _cute_active_mask_var(state, block_id)

    def active_index_var(block_id: int) -> str | None:
        return _cute_active_index_var(state, block_id)

    def active_local_coord(block_id: int) -> str | None:
        return _cute_active_local_coord(state, block_id, honor_override=True)

    def tile_begin_expr(block_id: int) -> str:
        return _cute_tile_begin_var_or_lifted(
            state, block_id, fallback="0", honor_override=True
        )

    def tile_with_offset_mask_terms(
        tile_info: TileWithOffsetInfo,
        tensor_dim: int,
    ) -> list[str]:
        block_id = tile_info.block_id
        local_coord = active_local_coord(block_id)
        begin_var = tile_begin_expr(block_id)
        if local_coord is None:
            if (idx_var := active_index_var(block_id)) is None:
                raise exc.BackendUnsupported(
                    "cute",
                    (
                        "indexing dimension is not active in this scope "
                        f"(block_id={block_id})"
                    ),
                )
            local_coord = f"({idx_var}) - ({begin_var})"

        offset_expr = state.device_function.literal_expr(tile_info.offset)
        global_coord = f"({begin_var}) + cutlass.Int32({offset_expr}) + ({local_coord})"
        tile_terms = []
        if tile_info.block_size is not None:
            block_size_expr = state.device_function.literal_expr(tile_info.block_size)
            tile_terms.append(f"({local_coord}) < cutlass.Int32({block_size_expr})")
            if mask_var_for_block_id(block_id) is not None and (
                end := cute_masked_block_end(
                    state.codegen,
                    _cute_remap_block_id(state, block_id),
                    "an epilogue-subtile store",
                )
            ):
                # An epilogue-subtile piece of the tile: mask it like the
                # tile, by its lanes' place in the loop (not shifted by the
                # store's own offset) against the loop's end, which the
                # target may exceed
                loop_coord = global_coord
                if tile_info.mask_offset is not None:
                    mask_offset = state.device_function.literal_expr(
                        tile_info.mask_offset
                    )
                    loop_coord = (
                        f"({begin_var}) + cutlass.Int32({mask_offset}) + "
                        f"({local_coord})"
                    )
                tile_terms.append(f"({loop_coord}) < {end}")
        if tensor is not None and tensor_dim < tensor.ndim:
            dim_size = _cute_tensor_dim_size_expr(state, tensor, tensor_dim)
            tile_terms.append(f"({global_coord}) < {dim_size}")
        return tile_terms

    def tensor_index_bounds_term(pos: int, tensor_dim: int) -> str | None:
        if tensor is None or tensor_dim >= tensor.ndim:
            return None
        ast_args = state.ast_args
        if not (
            isinstance(ast_args, list)
            and len(ast_args) > 1
            and isinstance(ast_args[1], (list, tuple))
            and len(ast_args[1]) == len(subscript)
            and isinstance(ast_args[1][pos], ast.AST)
        ):
            return None
        index_var = state.codegen.lift(
            ast_args[1][pos],
            dce=True,
            prefix="index_mask",
        ).id
        index_dtype = env.backend.dtype_str(env.index_dtype)
        dim_size = _cute_tensor_dim_size_expr(state, tensor, tensor_dim)
        return f"{index_dtype}({index_var}) < {dim_size}"

    if extra_mask is not None:
        terms.append(state.codegen.lift(extra_mask, dce=True, prefix="mask").id)

    seen: set[int] = set()
    tensor_dim = 0
    for pos, idx in enumerate(subscript):
        block_id: int | None = None
        if idx is None:
            continue
        if (
            tile_info := _get_tile_with_offset_info(
                idx, getattr(state, "fx_node", None), pos
            )
        ) is not None and tile_info.block_size is not None:
            seen.add(tile_info.block_id)
            for term in tile_with_offset_mask_terms(tile_info, tensor_dim):
                if term not in terms:
                    terms.append(term)
            tensor_dim += 1
            continue
        if isinstance(idx, torch.SymInt):
            block_id = env.get_block_id(idx)
            origin = _maybe_get_symbol_origin(idx)
            if (
                for_store
                and origin is not None
                and isinstance(origin.origin, TileIdOrigin)
                and block_id is not None
                and mask_var_for_block_id(block_id) is not None
            ):
                # A tile ID denotes one scalar per nonempty tile, not the
                # current element of that tile. Its element mask can depend
                # on both a thread coordinate and a serial lane loop, neither
                # of which belongs to this address. Keep the tile's absolute
                # end and destination bounds instead of borrowing that mask.
                from ...tile_strategy import DeviceGridState

                remapped = _cute_remap_block_id(state, block_id)
                owners = state.codegen.active_device_loops.get(remapped)
                owner = owners[-1] if owners else state.codegen.current_grid_state
                info = (
                    owner.block_id_to_info.get(remapped) if owner is not None else None
                )
                if info is None or owner is None:
                    raise exc.BackendUnsupported("cute", "tile ID has no active owner")
                if not _cute_tile_id_thread_extent_is_complete(state, owner, remapped):
                    original_mask = mask_var_for_block_id(block_id)
                    assert original_mask is not None
                    terms.append(original_mask)
                    seen.add(block_id)
                    tensor_dim += 1
                    continue
                if isinstance(owner, DeviceGridState):
                    end = info.grid_end_expr
                    end_expr = (
                        state.device_function.literal_expr(end)
                        if end is not None
                        else None
                    )
                else:
                    end_expr = info.end_var_name
                    if end_expr is None and info.end_expr is not None:
                        end_expr = state.device_function.literal_expr(info.end_expr)
                if end_expr is None:
                    raise exc.BackendUnsupported("cute", "tile ID has no logical end")
                begin = tile_begin_expr(block_id)
                terms.append(f"({begin}) < ({end_expr})")
                if tensor is not None and tensor_dim < tensor.ndim:
                    size = _cute_tensor_dim_size_expr(state, tensor, tensor_dim)
                    tile_size = state.device_function.block_size_var(remapped) or "1"
                    terms.extend(
                        (
                            f"(({begin}) // ({tile_size})) >= 0",
                            f"(({begin}) // ({tile_size})) < ({size})",
                        )
                    )
                seen.add(block_id)
                tensor_dim += 1
                continue
            if block_id is not None and _cute_is_tile_scalar(env, idx, block_id):
                # A tile attribute (``tile.begin``, ``tile.id``) or a grid
                # index is one address for the whole tile, in range whenever
                # the tile is, and drops the dimension: no lane mask applies
                # to it, as in the Triton lowering.  The axis's mask here
                # would gate a value to its default in lanes past the tile's
                # end, which a store of a tensor lacking the axis (guarded by
                # fewer masks) then writes.
                block_id = None
        elif isinstance(idx, slice) and idx == slice(None) and tensor is not None:
            for bid in _matching_block_ids(env, tensor.shape[tensor_dim]):
                if bid not in seen and mask_var_for_block_id(bid) is not None:
                    block_id = bid
                    break
        elif isinstance(idx, slice) and idx != slice(None) and tensor is not None:
            slice_size = compute_slice_size(idx, tensor.shape[tensor_dim])
            for bid in _matching_block_ids(env, slice_size):
                if bid not in seen and mask_var_for_block_id(bid) is not None:
                    block_id = bid
                    break
        elif isinstance(idx, torch.Tensor):
            added_tensor_index_mask = False
            # A range lowered onto the reduction axis allocated for its length
            # (``_cute_iota_expr``) holds that axis' coordinate, and the axis
            # may launch more lanes than the range has: its mask bounds the
            # range's position (``hl.arange(1, 6)`` on 8 lanes), where the
            # address bound below only checks the index value against the
            # tensor (lanes 5..7 read columns 6..8 into a reduction).
            fx_node = state.fx_node
            fx_index = fx_node.args[1] if fx_node is not None else None
            iota = (
                cute_arange_index_source(fx_index[pos])
                if isinstance(fx_index, (list, tuple))
                and len(fx_index) == len(subscript)
                else None
            )
            length = iota.meta["val"].size(0) if iota is not None else None
            if (
                iota is not None
                and (block_id := env.resolve_block_id(length)) is not None
                and env.block_sizes[block_id].reduction
                and (mask_var := mask_var_for_block_id(block_id)) is not None
                and mask_var not in terms
            ):
                terms.append(mask_var)
            # The same for a range on the reduction axis of its padded size
            # (``cute_padded_range_reduction_block``), which has no mask.  It
            # is spelled as the address bound below, which is the same term
            # for ``hl.arange(n)`` over a row of ``n``.
            bound_block = (
                cute_codegen_state(state.codegen).padded_range_blocks.get(iota)
                if iota is not None
                else None
            )
            if bound_block is not None:
                index_var = active_index_var(bound_block)
                if index_var is None:
                    # A sum of this load skips its own mask, since the zeros
                    # the load fills in are its identity
                    # (``_reads_padded_range_load``): without the term the
                    # lanes past the range would add data.
                    raise exc.BackendUnsupported(
                        "cute",
                        "an access over a padded hl.arange outside the lanes "
                        "of its reduction dim",
                    )
                term = (
                    f"{env.backend.dtype_str(env.index_dtype)}({index_var}) < {length}"
                )
                if term not in terms:
                    terms.append(term)
            # A free ``hl.arange`` mapped onto a synthetic thread axis carries no
            # block id, so the loops below add no bound for it. Emit its lane
            # bound explicitly to mask the out-of-bounds lanes a wider sibling
            # branch can introduce on a shared axis.
            arange_bound = _cute_synthetic_arange_lane_bound(
                state, pos, tensor, tensor_dim
            )
            if arange_bound is not None and arange_bound not in terms:
                terms.append(arange_bound)
            # A reduction/grid dim mapped onto a thread axis can be *widened*
            # beyond its own extent by a mutually-exclusive sibling branch that
            # reuses the same axis (the launch block is sized to the widest
            # user). When this access addresses such a dim per-lane, bound the
            # lane to its own dim size so the surplus lanes a wider sibling adds
            # do not load/store out of bounds. ``active_local_coord`` is the
            # per-lane in-axis coordinate; ``< dim_size`` is a no-op when the
            # axis already matches this dim.
            if tensor is not None and tensor_dim < tensor.ndim:
                for bid in _matching_block_ids(env, tensor.shape[tensor_dim]):
                    local_coord = active_local_coord(bid)
                    if local_coord is not None:
                        lane_bound = (
                            f"({local_coord}) < "
                            f"{_cute_tensor_dim_size_expr(state, tensor, tensor_dim)}"
                        )
                        if lane_bound not in terms:
                            terms.append(lane_bound)
                        break
            if not include_tensor_index_masks:
                for dim_size in idx.shape:
                    # Address bounds do not replace an index's symbolic tile
                    # domain. Equal extents may belong to unrelated axes.
                    direct = env.get_block_id(dim_size)
                    if direct is not None:
                        direct_mask = mask_var_for_block_id(direct)
                        if direct_mask is not None and direct_mask not in terms:
                            terms.append(direct_mask)
                    for bid in _matching_block_ids(env, dim_size):
                        if bid in seen or not env.is_jagged_tile(bid):
                            continue
                        mask_var = mask_var_for_block_id(bid)
                        if mask_var is not None:
                            added_tensor_index_mask = True
                            seen.add(bid)
                            if mask_var not in terms:
                                terms.append(mask_var)
                            break
                if (
                    not added_tensor_index_mask
                    and (bound := tensor_index_bounds_term(pos, tensor_dim)) is not None
                ):
                    if bound not in terms:
                        terms.append(bound)
                tensor_dim += 1
                continue
            for dim_size in idx.shape:
                for bid in _matching_block_ids(env, dim_size):
                    if bid in seen:
                        continue
                    mask_var = mask_var_for_block_id(bid)
                    if mask_var is not None:
                        added_tensor_index_mask = True
                        seen.add(bid)
                        if mask_var not in terms:
                            terms.append(mask_var)
                        break
                else:
                    continue
            if (
                not added_tensor_index_mask
                and (bound := tensor_index_bounds_term(pos, tensor_dim)) is not None
            ):
                if bound not in terms:
                    terms.append(bound)
            tensor_dim += 1
            continue
        else:
            tensor_dim += 1
            continue
        if block_id is None or block_id in seen:
            tensor_dim += 1
            continue
        seen.add(block_id)
        if (mask_var := mask_var_for_block_id(block_id)) is not None:
            if mask_var not in terms:
                terms.append(mask_var)
        tensor_dim += 1

    if not terms:
        return None
    return " and ".join(f"({term})" for term in terms)


def _cute_synthetic_arange_lane_bound(
    state: CodegenState,
    subscript_pos: int,
    tensor: torch.Tensor | None,
    tensor_dim: int,
) -> str | None:
    """Bounds mask for a free ``hl.arange`` index mapped onto a synthetic CUDA
    thread axis (CuTe backend).

    The launch block is sized to the *widest* arange across all (mutually
    exclusive) grid branches that share a thread axis. A narrower arange in
    another branch -- e.g. ``hl.arange(0, 32)`` sharing a 64-wide axis with a
    sibling's ``hl.arange(0, 64)`` -- therefore addresses lanes beyond its own
    extent. Those extra lanes carry a coordinate ``thread_idx()[axis] >= size``
    and, without this mask, perform out-of-bounds loads/stores. Returns the term
    ``(thread_idx()[axis]) < length`` (a no-op when the block matches the arange
    exactly), or ``None`` when this index is not such a synthetic arange.

    The synthetic-axis coordinate is the arange's *position* ``0..length-1``
    regardless of ``start``/``step`` (the ``start + step *`` wrapping is applied
    separately), so the surplus lanes a wider sibling adds are exactly those with
    position ``>= length``. Bounding to the arange's own ``length`` is therefore
    correct for canonical and non-canonical (non-zero start / non-unit step)
    arange dims alike.
    """
    if tensor is None or tensor_dim >= tensor.ndim:
        return None
    cg = state.codegen
    # A free arange maps onto a synthetic thread axis either directly
    # (``synthetic_arange_axes`` key -> axis, coord ``thread_idx()[axis]``)
    # or, when it overflows the thread budget, onto a lane loop whose coordinate
    # is cached in ``synthetic_arange_lane_exprs``.
    codegen_state = cute_codegen_state(cg)
    axes = codegen_state.synthetic_arange_axes
    lane_exprs = codegen_state.synthetic_arange_lane_exprs
    if not axes and not lane_exprs:
        return None
    fx_node = getattr(state, "fx_node", None)
    if fx_node is None or len(fx_node.args) < 2:
        return None
    subscript_arg = fx_node.args[1]
    if not isinstance(subscript_arg, (list, tuple)) or subscript_pos >= len(
        subscript_arg
    ):
        return None
    idx_node = subscript_arg[subscript_pos]
    if not isinstance(idx_node, torch.fx.Node):
        return None
    from ..iota_utils import cute_free_arange_indexed_dim_key
    from ..iota_utils import cute_free_arange_lanes

    # ``key`` is ``(lane class, length)`` (``_cute_free_arange_axis_expr``) or,
    # outside the modeled ops, ``(dim_key, length, start, step)``. The bound
    # masks lanes beyond this arange's own extent, which is its ``length`` --
    # independent of ``start``/``step`` -- so match on the class or ``dim_key``.
    root = cute_free_arange_lanes(cg).root(idx_node)
    if root is not None:

        def _match(key: object) -> bool:
            return isinstance(key, tuple) and len(key) == 2 and key[0] == root

    else:
        dim_key = cute_free_arange_indexed_dim_key(idx_node, cg)
        if dim_key is None:
            return None

        def _match(key: object) -> bool:
            return isinstance(key, tuple) and len(key) == 4 and key[0] == dim_key

    coord = None
    length: object = None
    for key, axis in axes.items():
        if _match(key):
            coord = f"cutlass.Int32(cute.arch.thread_idx()[{axis}])"
            length = key[1]
            break
    if coord is None:
        for key, expr in lane_exprs.items():
            if _match(key):
                coord = expr
                length = key[1]
                break
    if coord is None:
        return None
    return f"({coord}) < {length}"


def _cute_tensor_dim_size_expr(
    state: CodegenState, tensor: torch.Tensor, dim: int
) -> str:
    return state.device_function.tensor_size(tensor, dim).name


def _cute_tile_begin_expr(state: CodegenState, idx: object) -> str:
    env = CompileEnvironment.current()

    def active_index_var(block_id: int) -> str | None:
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            return loops[-1].strategy.index_var(block_id)
        grid_state = state.codegen.current_grid_state
        if grid_state is not None and block_id in grid_state.block_ids:
            return grid_state.strategy.index_var(block_id)
        return None

    def active_local_coord(block_id: int) -> str | None:
        from ..cute_reshape import _grid_local_coord_expr

        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            thread_axis = loops[-1].block_thread_axes.get(block_id)
            if thread_axis is not None:
                return _grid_local_coord_expr(state.codegen, block_id, thread_axis)
        grid_state = state.codegen.current_grid_state
        if grid_state is not None:
            thread_axis = grid_state.block_thread_axes.get(block_id)
            if thread_axis is not None:
                return _grid_local_coord_expr(state.codegen, block_id, thread_axis)
        return None

    def tile_begin_from_block_id(block_id: int) -> str:
        loops = state.codegen.active_device_loops.get(block_id)
        if loops:
            return state.codegen.offset_var(block_id)
        global_index = active_index_var(block_id)
        local_coord = active_local_coord(block_id)
        if global_index is not None and local_coord is not None:
            return state.codegen.lift(
                expr_from_string(f"({global_index}) - ({local_coord})"),
                dce=True,
                prefix="tile_begin",
            ).id
        if global_index is not None:
            return global_index
        return "0"

    if isinstance(idx, int):
        return str(idx)
    if not isinstance(idx, torch.SymInt):
        raise exc.BackendUnsupported("cute", f"tile base index type: {type(idx)}")

    expr = _symint_expr(idx)
    if expr is not None:
        origin_info = HostFunction.current().expr_to_origin.get(expr)
        if origin_info is not None and isinstance(origin_info.origin, TileBeginOrigin):
            return tile_begin_from_block_id(origin_info.origin.block_id)
    block_id = env.get_block_id(idx)
    if block_id is not None:
        return tile_begin_from_block_id(block_id)
    if expr is not None:
        return state.sympy_expr(expr)
    raise exc.BackendUnsupported("cute", f"unlowerable tile base index: {idx}")
