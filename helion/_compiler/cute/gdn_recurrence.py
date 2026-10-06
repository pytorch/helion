"""Whole-root tcgen05 lowering for the gated-delta-rule chunked state recurrence.

The matcher recognizes the two-contraction chunk recurrence written by
``examples/gdn_fwd_h.py`` purely from its FX dataflow (no names).  Per chunk of
``chunk`` tokens the root carries a ``[dhead, block_v]`` fp32 state and does::

    h[chunk] = bf16(state)
    x = w_chunk @ bf16(state)  # dot 1
    bv = bf16((u_chunk - x) * where(valid, exp(g_last - g_chunk), 0)[:, None])
    state = state * exp(g_last) + k_chunk ^ T @ bv  # dot 2

The replacement keeps ``state^T`` resident in TMEM so it is the M side of both
tcgen05 contractions (a tile narrower than 128 rows is replicated across the
TMEM quadrants so every SM sub-partition shares the per-chunk epilogue), the
epilogue warps issue both MMAs themselves, the ``w``/``k``/``u`` chunk tiles
stream through a TMA-fed shared-memory ring, and the gating and the fp32
rescale are exact (``exp`` is the ordinary ``exp2(x * log2 e)`` lowering,
never approximate).

Admission is fail-closed: SM100, static shapes, contiguous CUDA tensors, bf16
``k``/``w``/``u``/``h`` with fp32 ``g``, ``dhead`` in {16, 32, 64}, a
power-of-two ``chunk`` in ``[16, 256]``, ``dstate`` a multiple of 8 (16-byte
TMA strides), a ``dstate`` tile in {16, 32, 64, 128} kept in a 128-row tcgen05
``M`` tile (rows past ``dstate`` are masked), and a TMEM budget of 512
columns; see :mod:`.gdn_recurrence_geometry`.  Any other shape or any graph
mutation keeps the ordinary SIMT lowering.
"""

from __future__ import annotations

import ast
from collections import Counter
import dataclasses
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch

from ... import exc
from ...autotuner.config_spec import CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY
from ...autotuner.config_spec import CUTE_GDN_RECURRENCE_MMA_M_KEY
from ...autotuner.config_spec import CUTE_GDN_RECURRENCE_STAGES_KEY
from ...autotuner.config_spec import CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY
from ...language.matmul_ops import dot
from ..compile_environment import CompileEnvironment
from ..compile_environment import FixedBlockSizeSource
from .fx_matcher import _canonical_root_axis_ids
from .fx_matcher import _is_call
from .fx_matcher import _linear_offsets_fit_i32
from .fx_matcher import _load_ref
from .fx_matcher import _memory_indices
from .fx_matcher import _same_ref
from .fx_matcher import _store_ref
from .fx_matcher import _TensorRef
from .fx_matcher import _xyz_grid_fits
from .gdn_recurrence_geometry import GDN_ADMITTED_BLOCK_V
from .gdn_recurrence_geometry import GDN_MMA_M
from .gdn_recurrence_geometry import GDN_RECURRENCE_DEVICE_ABI
from .gdn_recurrence_geometry import GDN_RECURRENCE_KIND
from .gdn_recurrence_geometry import gdn_candidate_block_sizes
from .gdn_recurrence_geometry import gdn_cta_warps
from .gdn_recurrence_geometry import gdn_epilogue_warp_choices
from .gdn_recurrence_geometry import gdn_mma_m_choices
from .gdn_recurrence_geometry import gdn_shape_admitted
from .gdn_recurrence_geometry import gdn_smem_bytes
from .gdn_recurrence_geometry import gdn_stage_choices
from .gdn_recurrence_geometry import gdn_state_replicas
from .gdn_recurrence_geometry import gdn_tmem_layout
from .gdn_recurrence_geometry import gdn_token_group_choices

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..device_ir import GraphInfo
    from ..generate_ast import GenerateAST
    from ..tile_dispatch import TileStrategyDispatch


@dataclasses.dataclass(frozen=True)
class GdnRecurrenceGeometry:
    """Static problem geometry of one matched gdn recurrence root."""

    batch: int
    heads: int
    seqlen: int
    chunk: int
    dhead: int
    dstate: int
    num_chunks: int
    value_block_id: int


@dataclasses.dataclass(frozen=True)
class _GdnRecurrenceMatch:
    root_graph_id: int
    geometry: GdnRecurrenceGeometry
    k: _TensorRef
    w: _TensorRef
    u: _TensorRef
    g: _TensorRef
    h: _TensorRef
    grid_block_ids: tuple[int, int, int]


@dataclasses.dataclass(frozen=True)
class CuteGdnRecurrencePlan:
    """Compile-time contract between the matched root and the SM100 schedule."""

    root_graph_id: int
    geometry: GdnRecurrenceGeometry
    k: _TensorRef
    w: _TensorRef
    u: _TensorRef
    g: _TensorRef
    h: _TensorRef
    block_v: int
    epilogue_warps: int
    stages: int
    token_groups: int
    mma_m: int
    threads: int
    tmem_cols: int
    smem_bytes: int


def _call_counts(nodes: Sequence[torch.fx.Node]) -> Counter[object]:
    return Counter(node.target for node in nodes if node.op == "call_function")


def _single(nodes: Sequence[torch.fx.Node], target: object) -> torch.fx.Node | None:
    matches = [node for node in nodes if _is_call(node, target)]
    return matches[0] if len(matches) == 1 else None


def _static_int(value: object) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, torch.SymInt):
        expr = value._sympy_()
        if isinstance(expr, sympy.Integer):
            return int(expr)
        return None
    if isinstance(value, torch.fx.Node):
        return _static_int(value.meta.get("val"))
    return None


def _static_shape(fake: torch.Tensor) -> tuple[int, ...] | None:
    dims = tuple(_static_int(dim) for dim in fake.shape)
    if any(dim is None or dim <= 0 for dim in dims):
        return None
    return cast("tuple[int, ...]", dims)


def _binary_args(
    node: object, targets: tuple[object, ...]
) -> tuple[object, object] | None:
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target not in targets
        or len(node.args) != 2
    ):
        return None
    return node.args[0], node.args[1]


def _is_convert(node: object, source: object, dtype: torch.dtype) -> bool:
    return bool(
        isinstance(node, torch.fx.Node)
        and _is_call(node, torch.ops.prims.convert_element_type.default)
        and len(node.args) == 2
        and node.args[0] is source
        and node.args[1] is dtype
    )


def _is_full_slice(value: object) -> bool:
    return isinstance(value, slice) and value == slice(None)


def _is_plain_load(node: object) -> bool:
    return bool(
        isinstance(node, torch.fx.Node)
        and _load_ref(node) is not None
        and len(node.args) == 4
        and node.args[2] is None
        and node.args[3] is None
    )


def _indices_are(node: torch.fx.Node, expected: Sequence[object]) -> bool:
    indices = _memory_indices(node)
    if indices is None or len(indices) != len(expected):
        return False
    for actual, wanted in zip(indices, expected, strict=True):
        if wanted == "full":
            if not _is_full_slice(actual):
                return False
        elif actual is not wanted:
            return False
    return True


def _is_tile_begin_symbol(symbol: sympy.Symbol, block_id: int) -> bool:
    from ..host_function import HostFunction
    from ..variable_origin import TileBeginOrigin

    origin_info = HostFunction.current().expr_to_origin.get(symbol)
    return (
        origin_info is not None
        and isinstance(origin_info.origin, TileBeginOrigin)
        and origin_info.origin.block_id == block_id
    )


def _is_tile_id_coordinate(node: object) -> bool:
    """``node`` carries a root ``tile.id`` scalar (``TileIdOrigin``).

    Every grid-derived scalar of a tile (``tile.end``, ``tile.begin`` ...)
    resolves to the same block id, but only the id addresses the row the
    schedule reads.
    """

    from ..host_function import HostFunction
    from ..variable_origin import TileIdOrigin

    value = cast("torch.fx.Node", node).meta.get("val")
    if not isinstance(value, torch.SymInt):
        return False
    expr = value._sympy_()
    if not isinstance(expr, sympy.Symbol):
        return False
    origin_info = HostFunction.current().expr_to_origin.get(expr)
    return origin_info is not None and isinstance(origin_info.origin, TileIdOrigin)


def _match_gdn_recurrence_graphs(
    graphs: Sequence[GraphInfo],
) -> _GdnRecurrenceMatch | None:
    """Recognize the exact gated-delta-rule chunk recurrence dataflow."""

    from ...language import memory_ops
    from ...language import view_ops
    from ...language._tracing_ops import _for_loop
    from ...language._tracing_ops import _get_symnode
    from ...language._tracing_ops import _host_tensor
    from ...language._tracing_ops import _new_var
    from ...language._tracing_ops import _phi
    from ...language.creation_ops import full
    from ...language.tile_ops import tile_id
    from ...language.tile_ops import tile_index
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import RootGraphInfo
    from ..host_function import HostFunction

    convert = torch.ops.prims.convert_element_type.default
    roots = [graph for graph in graphs if isinstance(graph, RootGraphInfo)]
    loops = [graph for graph in graphs if isinstance(graph, ForLoopGraphInfo)]
    if len(graphs) != 2 or len(roots) != 1 or len(loops) != 1:
        return None
    root, loop = roots[0], loops[0]
    if len(loop.block_ids) != 1:
        return None
    chunk_block_id = loop.block_ids[0]

    root_nodes = list(root.graph.nodes)
    inner = list(loop.graph.nodes)
    expected_root: Counter[object] = Counter(
        {_get_symnode: 1, full: 1, _for_loop: 1, operator.getitem: 1, _phi: 1}
    )
    expected_inner: Counter[object] = Counter(
        {
            _new_var: 1,
            convert: 4,
            _get_symnode: 4,
            tile_id: 1,
            _host_tensor: 5,
            torch.ops.aten.sym_size.int: 1,
            memory_ops.store: 1,
            memory_ops.load: 5,
            dot: 2,
            torch.ops.aten.sub.Tensor: 2,
            tile_index: 1,
            torch.ops.aten.lt.Scalar: 1,
            operator.sub: 1,
            torch.ops.aten.exp.default: 2,
            torch.ops.aten.scalar_tensor.default: 1,
            torch.ops.aten.where.self: 1,
            view_ops.subscript: 1,
            torch.ops.aten.mul.Tensor: 2,
            torch.ops.aten.permute.default: 1,
        }
    )
    if _call_counts(root_nodes) != expected_root:
        return None
    if _call_counts(inner) != expected_inner:
        return None
    placeholders = [node for node in inner if node.op == "placeholder"]
    if len(placeholders) != 1:
        return None
    state_in = placeholders[0]

    # Root: zero-initialized [dhead, block_v] fp32 carrier, one chunk loop over
    # [0, seqlen), nothing else.
    full_node = _single(root_nodes, full)
    loop_call = _single(root_nodes, _for_loop)
    getitem = _single(root_nodes, operator.getitem)
    phi = _single(root_nodes, _phi)
    root_outputs = list(root.graph.find_nodes(op="output"))
    if (
        full_node is None
        or loop_call is None
        or getitem is None
        or phi is None
        or len(root_outputs) != 1
        or root_outputs[0].args != (None,)
    ):
        return None
    if len(full_node.args) not in (3, 4) or (
        len(full_node.args) == 4 and full_node.args[3] is not None
    ):
        return None
    shape = full_node.args[0]
    fill = full_node.args[1]
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 2
        or type(fill) not in (int, float)
        or fill != 0
        or full_node.args[2] is not torch.float32
    ):
        return None
    dhead = _static_int(shape[0])
    value_size_node = shape[1]
    if dhead is None or not _is_call(value_size_node, _get_symnode):
        return None
    env = CompileEnvironment.current()
    value_block_id = env.resolve_block_id(
        cast("torch.fx.Node", value_size_node).meta.get("val")
    )
    if value_block_id is None:
        return None
    if (
        len(loop_call.args) != 4
        or loop_call.args[0] != loop.graph_id
        or loop_call.args[1] != [0]
        or not isinstance(loop_call.args[2], (list, tuple))
        or len(loop_call.args[2]) != 1
        or not isinstance(loop_call.args[3], (list, tuple))
        or len(loop_call.args[3]) != 1
        or loop_call.args[3][0] is not full_node
    ):
        return None
    seqlen = _static_int(loop_call.args[2][0])
    if seqlen is None or seqlen <= 0:
        return None
    if getitem.args != (loop_call, 0) or phi.args != (full_node, getitem):
        return None
    chunk_info = env.block_sizes[chunk_block_id]
    chunk_source = chunk_info.block_size_source
    chunk = (
        _static_int(chunk_source.value)
        if isinstance(chunk_source, FixedBlockSizeSource)
        else None
    )
    if chunk is None or chunk <= 0 or _static_int(chunk_info.size) != seqlen:
        return None

    # Loop body: two contractions with the carried state as the B side of the
    # first and the accumulator of the second.
    state_phi = _single(inner, _new_var)
    if state_phi is None or state_phi.args != (state_in,):
        return None
    dots = [node for node in inner if _is_call(node, dot)]
    projection, update = dots
    if (
        len(projection.args) != 4
        or projection.args[2] is not None
        or projection.args[3] is not torch.float32
        or not _is_convert(projection.args[1], state_phi, torch.bfloat16)
    ):
        return None
    w_load = projection.args[0]
    if not _is_plain_load(w_load):
        return None
    assert isinstance(w_load, torch.fx.Node)

    if len(update.args) != 4 or update.args[3] is not None:
        return None
    permute = update.args[0]
    if (
        not _is_call(permute, torch.ops.aten.permute.default)
        or len(cast("torch.fx.Node", permute).args) != 2
        or list(cast("torch.fx.Node", permute).args[1]) != [1, 0]  # type: ignore[arg-type]
    ):
        return None
    k_load = cast("torch.fx.Node", permute).args[0]
    if not _is_plain_load(k_load):
        return None
    assert isinstance(k_load, torch.fx.Node)
    update_bf16 = update.args[1]
    if (
        not isinstance(update_bf16, torch.fx.Node)
        or not _is_call(update_bf16, convert)
        or len(update_bf16.args) != 2
        or update_bf16.args[1] is not torch.bfloat16
    ):
        return None
    gated_args = _binary_args(update_bf16.args[0], (torch.ops.aten.mul.Tensor,))
    if gated_args is None:
        return None
    residual_args = _binary_args(gated_args[0], (torch.ops.aten.sub.Tensor,))
    if residual_args is None or residual_args[1] is not projection:
        return None
    u_f32 = residual_args[0]
    if (
        not isinstance(u_f32, torch.fx.Node)
        or not _is_call(u_f32, convert)
        or len(u_f32.args) != 2
        or u_f32.args[1] is not torch.float32
    ):
        return None
    u_load = u_f32.args[0]
    if not _is_plain_load(u_load):
        return None
    assert isinstance(u_load, torch.fx.Node)

    # Gate: where(chunk rows < seqlen, exp(g_last - g_chunk), 0)[:, None].
    gate_bcast = gated_args[1]
    if (
        not _is_call(gate_bcast, view_ops.subscript)
        or len(cast("torch.fx.Node", gate_bcast).args) != 2
        or list(cast("torch.fx.Node", gate_bcast).args[1])  # type: ignore[arg-type]
        != [slice(None), None]
    ):
        return None
    where = cast("torch.fx.Node", gate_bcast).args[0]
    if (
        not _is_call(where, torch.ops.aten.where.self)
        or len(cast("torch.fx.Node", where).args) != 3
    ):
        return None
    mask, gate_exp, zero = cast("torch.fx.Node", where).args
    mask_args = _binary_args(mask, (torch.ops.aten.lt.Scalar,))
    if mask_args is None or _static_int(mask_args[1]) != seqlen:
        return None
    tile_index_node = mask_args[0]
    if (
        not _is_call(tile_index_node, tile_index)
        or len(cast("torch.fx.Node", tile_index_node).args) != 1
    ):
        return None
    chunk_sym = cast("torch.fx.Node", tile_index_node).args[0]
    if (
        not _is_call(chunk_sym, _get_symnode)
        or env.resolve_block_id(cast("torch.fx.Node", chunk_sym).meta.get("val"))
        != chunk_block_id
    ):
        return None
    if (
        not _is_call(zero, torch.ops.aten.scalar_tensor.default)
        or len(cast("torch.fx.Node", zero).args) != 1
        or cast("torch.fx.Node", zero).args[0] != 0
        or cast("torch.fx.Node", zero).kwargs.get("dtype") is not torch.float32
    ):
        return None
    if (
        not _is_call(gate_exp, torch.ops.aten.exp.default)
        or len(cast("torch.fx.Node", gate_exp).args) != 1
    ):
        return None
    gate_args = _binary_args(
        cast("torch.fx.Node", gate_exp).args[0], (torch.ops.aten.sub.Tensor,)
    )
    if gate_args is None:
        return None
    g_last_load, g_load = gate_args
    if not _is_plain_load(g_last_load) or not _is_plain_load(g_load):
        return None
    assert isinstance(g_last_load, torch.fx.Node)
    assert isinstance(g_load, torch.fx.Node)

    # Rescale: acc = state * exp(g_last) with the same g_last load.
    rescale_args = _binary_args(update.args[2], (torch.ops.aten.mul.Tensor,))
    if rescale_args is None or rescale_args[0] is not state_phi:
        return None
    rescale_exp = rescale_args[1]
    if not _is_call(rescale_exp, torch.ops.aten.exp.default) or cast(
        "torch.fx.Node", rescale_exp
    ).args != (g_last_load,):
        return None

    # Coordinates: every load shares the batch/head coordinates and the chunk
    # tile; u and the h store share the dstate coordinate of the carrier.
    w_indices = _memory_indices(w_load)
    if w_indices is None or len(w_indices) != 4:
        return None
    batch_coord, w_chunk, head_coord, w_feature = w_indices
    if (
        w_chunk is not chunk_sym
        or not _is_full_slice(w_feature)
        or not _is_call(batch_coord, _get_symnode)
        or not _is_call(head_coord, _get_symnode)
        or batch_coord is head_coord
        or not _is_tile_id_coordinate(batch_coord)
        or not _is_tile_id_coordinate(head_coord)
    ):
        return None
    value_coord = _single(inner, torch.ops.aten.sym_size.int)
    if value_coord is None or value_coord.args != (state_in, 1):
        return None
    if not _indices_are(k_load, (batch_coord, chunk_sym, head_coord, "full")):
        return None
    if not _indices_are(u_load, (batch_coord, chunk_sym, head_coord, value_coord)):
        return None
    if not _indices_are(g_load, (batch_coord, chunk_sym, head_coord)):
        return None
    g_last_indices = _memory_indices(g_last_load)
    if (
        g_last_indices is None
        or len(g_last_indices) != 3
        or g_last_indices[0] is not batch_coord
        or g_last_indices[2] is not head_coord
    ):
        return None
    last_args = _binary_args(g_last_indices[1], (operator.sub,))
    if last_args is None or last_args[1] != 1:
        return None
    last_sym = last_args[0]
    if not _is_call(last_sym, _get_symnode):
        return None
    last_val = cast("torch.fx.Node", last_sym).meta.get("val")
    if not isinstance(last_val, torch.SymInt):
        return None
    last_expr = last_val._sympy_()
    symbols = tuple(last_expr.free_symbols)
    begin_symbol = symbols[0] if len(symbols) == 1 else None
    if not isinstance(begin_symbol, sympy.Symbol) or not _is_tile_begin_symbol(
        begin_symbol, chunk_block_id
    ):
        return None
    if last_expr != sympy.Min(
        sympy.Integer(seqlen), sympy.Add(begin_symbol, sympy.Integer(chunk))
    ):
        return None

    # State store at the top of every chunk and the carried update.
    store = _single(inner, memory_ops.store)
    if (
        store is None
        or len(store.args) != 4
        or store.args[3] is not None
        or not _is_convert(store.args[2], state_phi, torch.bfloat16)
    ):
        return None
    store_indices = _memory_indices(store)
    if store_indices is None or len(store_indices) != 5:
        return None
    chunk_id = store_indices[1]
    if (
        not _is_call(chunk_id, tile_id)
        or cast("torch.fx.Node", chunk_id).args != (chunk_sym,)
        or not _indices_are(
            store, (batch_coord, chunk_id, head_coord, "full", value_coord)
        )
    ):
        return None
    loop_outputs = list(loop.graph.find_nodes(op="output"))
    if len(loop_outputs) != 1:
        return None
    carried = loop_outputs[0].args[0] if loop_outputs[0].args else None
    if not isinstance(carried, (list, tuple)) or len(carried) != 1:
        return None
    if carried[0] is not update:
        return None

    # Tensor contract.
    k = _load_ref(k_load)
    w = _load_ref(w_load)
    u = _load_ref(u_load)
    g = _load_ref(g_load)
    g_last = _load_ref(g_last_load)
    h = _store_ref(store)
    if any(ref is None for ref in (k, w, u, g, g_last, h)) or not _same_ref(g, g_last):
        return None
    assert k is not None
    assert w is not None
    assert u is not None
    assert g is not None
    assert h is not None
    refs = (k, w, u, g, h)
    if any(ref.fake.device.type != "cuda" for ref in refs):
        return None
    if any(not ref.fake.is_contiguous() for ref in refs):
        return None
    if any(ref.fake.dtype is not torch.bfloat16 for ref in (k, w, u, h)):
        return None
    if g.fake.dtype is not torch.float32:
        return None
    shapes = tuple(_static_shape(ref.fake) for ref in refs)
    if any(shape is None for shape in shapes):
        return None
    k_shape, w_shape, u_shape, g_shape, h_shape = cast(
        "tuple[tuple[int, ...], ...]", shapes
    )
    if len(k_shape) != 4 or k_shape != w_shape:
        return None
    batch, seqlen_dim, heads, dhead_dim = k_shape
    if seqlen_dim != seqlen or dhead_dim != dhead:
        return None
    if len(u_shape) != 4 or u_shape[:3] != (batch, seqlen, heads):
        return None
    dstate = u_shape[3]
    if g_shape != (batch, seqlen, heads):
        return None
    num_chunks = -(-seqlen // chunk)
    if h_shape != (batch, num_chunks, heads, dhead, dstate):
        return None
    if _static_int(state_in.meta["val"].shape[0]) != dhead:
        return None

    device_ir = HostFunction.current().device_ir
    grid_ids = _canonical_root_axis_ids(
        device_ir,
        root_phase_index=root.phase_index,
        coordinates=(
            cast("torch.fx.Node", batch_coord),
            cast("torch.fx.Node", head_coord),
            value_coord,
        ),
        expected_extents=(batch, heads, dstate),
    )
    if grid_ids is None or len(grid_ids) != 3 or grid_ids[2] != value_block_id:
        return None
    if chunk_block_id in grid_ids:
        return None
    return _GdnRecurrenceMatch(
        root_graph_id=root.graph_id,
        geometry=GdnRecurrenceGeometry(
            batch=batch,
            heads=heads,
            seqlen=seqlen,
            chunk=chunk,
            dhead=dhead,
            dstate=dstate,
            num_chunks=num_chunks,
            value_block_id=value_block_id,
        ),
        k=k,
        w=w,
        u=u,
        g=g,
        h=h,
        grid_block_ids=cast("tuple[int, int, int]", tuple(grid_ids)),
    )


def _geometry_admitted(geometry: GdnRecurrenceGeometry, block_v: int) -> bool:
    return gdn_shape_admitted(
        dhead=geometry.dhead,
        chunk=geometry.chunk,
        dstate=geometry.dstate,
        block_v=block_v,
    )


def detect_gdn_recurrence_search_geometry(
    graphs: Sequence[GraphInfo],
) -> GdnRecurrenceGeometry | None:
    """Recognize the recurrence before config generation (any dstate tile)."""

    match = _match_gdn_recurrence_graphs(graphs)
    if match is None:
        return None
    geometry = match.geometry
    if not gdn_admitted_block_sizes(geometry):
        return None
    return geometry


def gdn_admitted_block_sizes(geometry: GdnRecurrenceGeometry) -> tuple[int, ...]:
    """Every dstate tile the planner lowers for this geometry, preferred first.

    The candidates (tiles no wider than dstate) lead.  The remaining admitted
    tiles follow: the ``block_sizes`` search reaches the next power of two
    above a non-power-of-two dstate, and the schedule treats such a tile like
    a partial last tile (TMA zero fill plus a bounded ``h`` store).
    """

    candidates = gdn_candidate_block_sizes(geometry.dstate)
    wider = tuple(
        block_v for block_v in GDN_ADMITTED_BLOCK_V if block_v not in candidates
    )
    return tuple(
        block_v
        for block_v in (*candidates, *wider)
        if _geometry_admitted(geometry, block_v)
    )


def gdn_seed_block_sizes(geometry: GdnRecurrenceGeometry) -> tuple[int, ...]:
    """The preferred dstate tile and its half, when the schedule admits them.

    Wider admitted tiles stay reachable through the ``block_sizes`` search
    instead of being seeded.
    """

    return tuple(
        block_v
        for block_v in gdn_candidate_block_sizes(geometry.dstate)
        if _geometry_admitted(geometry, block_v)
    )[:2]


def gdn_recurrence_mma_m_choices_by_tile(
    geometry: GdnRecurrenceGeometry,
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Legal tcgen05 ``M`` values for every (dstate tile, epilogue warps) pair
    the planner lowers, preferred first; ``ConfigSpec.normalize`` re-validates
    the ``M`` of any pair the search selects."""

    return {
        (block_v, warps): gdn_mma_m_choices(
            geometry.chunk, geometry.dhead, block_v, warps
        )
        for block_v in gdn_admitted_block_sizes(geometry)
        for warps in gdn_epilogue_warp_choices(geometry.chunk, geometry.dhead)
    }


def gdn_recurrence_stage_choices_by_tile(
    geometry: GdnRecurrenceGeometry,
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Legal TMA ring depths for every (dstate tile, tcgen05 ``M``) pair the
    planner lowers, preferred first (the replicated update image of the
    wider ``M`` takes shared memory from the ring).

    Keyed so ``ConfigSpec.normalize`` can re-validate the depth of any pair
    the search selects.
    """

    by_tile = gdn_recurrence_mma_m_choices_by_tile(geometry)
    return {
        (block_v, mma_m): gdn_stage_choices(
            geometry.chunk, geometry.dhead, block_v, mma_m
        )
        for block_v in gdn_admitted_block_sizes(geometry)
        for mma_m in dict.fromkeys(
            mma_m
            for (tile, _warps), choices in by_tile.items()
            if tile == block_v
            for mma_m in choices
        )
    }


def gdn_recurrence_token_group_choices_by_tile(
    geometry: GdnRecurrenceGeometry,
) -> dict[tuple[int, int, int], tuple[int, ...]]:
    """Legal token groups for every (dstate tile, epilogue warps, tcgen05
    ``M``) triple the planner lowers, preferred first; ``ConfigSpec.normalize``
    re-validates the group count of any triple the search selects."""

    return {
        (block_v, warps, mma_m): gdn_token_group_choices(
            geometry.chunk, block_v, warps, mma_m
        )
        for (block_v, warps), mma_m_choices in gdn_recurrence_mma_m_choices_by_tile(
            geometry
        ).items()
        for mma_m in mma_m_choices
    }


def gdn_recurrence_warp_choices(
    geometry: GdnRecurrenceGeometry, block_v: int
) -> tuple[int, ...]:
    """Epilogue warp choices for ``block_v``, preferred first.

    A tile of at most 32 rows (replicated four times by the full MMA) spreads
    one column slice over all four SM sub-partitions, so it prefers four
    epilogue warps; wider tiles prefer eight (two slices per quadrant hide
    the TMEM latencies).
    """

    preferred_warps = 4 if gdn_state_replicas(block_v, GDN_MMA_M) == 4 else 8
    warp_choices = gdn_epilogue_warp_choices(geometry.chunk, geometry.dhead)
    return tuple(sorted(warp_choices, key=lambda warps: warps != preferred_warps))


def _plan_gdn_recurrence(
    graphs: Sequence[GraphInfo],
    _tile_strategy: TileStrategyDispatch,
) -> CuteGdnRecurrencePlan | None:
    from ..device_function import DeviceFunction

    df = DeviceFunction.current()
    if df.config.get("pid_type", "flat") != "flat":
        return None
    match = _match_gdn_recurrence_graphs(graphs)
    if match is None:
        return None
    env = CompileEnvironment.current()
    target = env.config_spec.target_device_capability
    if target is None or target[0] != 10:
        return None
    geometry = match.geometry
    resolved = [df.resolved_block_size(block_id) for block_id in match.grid_block_ids]
    if resolved[0] != 1 or resolved[1] != 1:
        return None
    block_v = resolved[2]
    if type(block_v) is not int or block_v not in gdn_admitted_block_sizes(geometry):
        return None
    tmem = gdn_tmem_layout(geometry.dhead, geometry.chunk)
    if tmem is None:
        return None
    warp_choices = gdn_recurrence_warp_choices(geometry, block_v)
    epilogue_warps = df.config.get(
        CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY, warp_choices[0]
    )
    if type(epilogue_warps) is not int or epilogue_warps not in warp_choices:
        raise exc.BackendUnsupported(
            "cute",
            f"{CUTE_GDN_RECURRENCE_EPILOGUE_WARPS_KEY}={epilogue_warps!r} cannot "
            f"slice the gdn recurrence TMEM columns (legal: {warp_choices})",
        )
    # A config that leaves the MMA height out keeps the full 128-row tile
    # (the search domain's default); the seeds pin the tile's preferred M.
    mma_m_choices = gdn_mma_m_choices(
        geometry.chunk, geometry.dhead, block_v, epilogue_warps
    )
    mma_m = df.config.get(CUTE_GDN_RECURRENCE_MMA_M_KEY, GDN_MMA_M)
    if type(mma_m) is not int or mma_m not in mma_m_choices:
        raise exc.BackendUnsupported(
            "cute",
            f"{CUTE_GDN_RECURRENCE_MMA_M_KEY}={mma_m!r} cannot hold the gdn "
            f"recurrence dstate tile over these warps (legal: {mma_m_choices})",
        )
    stage_choices = gdn_stage_choices(geometry.chunk, geometry.dhead, block_v, mma_m)
    stages = df.config.get(CUTE_GDN_RECURRENCE_STAGES_KEY, stage_choices[0])
    if type(stages) is not int or stages not in stage_choices:
        raise exc.BackendUnsupported(
            "cute",
            f"{CUTE_GDN_RECURRENCE_STAGES_KEY}={stages!r} does not fit the "
            f"gdn recurrence shared-memory budget (legal: {stage_choices})",
        )
    group_choices = gdn_token_group_choices(
        geometry.chunk, block_v, epilogue_warps, mma_m
    )
    token_groups = df.config.get(CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY, group_choices[0])
    if type(token_groups) is not int or token_groups not in group_choices:
        raise exc.BackendUnsupported(
            "cute",
            f"{CUTE_GDN_RECURRENCE_TOKEN_GROUPS_KEY}={token_groups!r} cannot split "
            f"the gdn recurrence chunk over these warps (legal: {group_choices})",
        )
    grid = (-(-geometry.dstate // block_v), geometry.heads, geometry.batch)
    if not _xyz_grid_fits(grid):
        return None
    refs = (match.k, match.w, match.u, match.g, match.h)
    if not _linear_offsets_fit_i32(
        grid[0] * grid[1] * grid[2],
        tuple(ref.fake.numel() for ref in refs),
    ):
        return None
    return CuteGdnRecurrencePlan(
        root_graph_id=match.root_graph_id,
        geometry=geometry,
        k=match.k,
        w=match.w,
        u=match.u,
        g=match.g,
        h=match.h,
        block_v=block_v,
        epilogue_warps=epilogue_warps,
        stages=stages,
        token_groups=token_groups,
        mma_m=mma_m,
        threads=32 * gdn_cta_warps(block_v, epilogue_warps, mma_m),
        tmem_cols=tmem.alloc_cols,
        smem_bytes=gdn_smem_bytes(
            geometry.chunk, geometry.dhead, block_v, stages, mma_m
        ),
    )


def plan_gdn_recurrence(
    graphs: Sequence[GraphInfo], tile_strategy: TileStrategyDispatch
) -> None:
    from ..device_function import DeviceFunction

    DeviceFunction.current().cute_state.gdn_recurrence_plan = _plan_gdn_recurrence(
        graphs, tile_strategy
    )


def _tensor_arg(cg: GenerateAST, ref: _TensorRef) -> str:
    return cg.device_function.tensor_arg(ref.fake, prefer_name=ref.name).name


def _emit_module_imports(cg: GenerateAST) -> None:
    helper_name = cg.device_function.new_var("_helion_gdn_recurrence_host")
    source = (
        "from helion._compiler.cute.gdn_recurrence_sm100 "
        f"import host_gdn_recurrence as {helper_name}"
    )
    cg.module_statements.extend(ast.parse(source).body)


def codegen_gdn_recurrence(cg: GenerateAST) -> bool:
    """Replace the matched root with the SM100 schedule's wrapper plan."""

    df = cg.device_function
    plan = df.cute_state.gdn_recurrence_plan
    root = cg.current_root_graph_info
    if plan is None or root is None or root.graph_id != plan.root_graph_id:
        return False
    tmem = gdn_tmem_layout(plan.geometry.dhead, plan.geometry.chunk)
    if tmem is None or tmem.alloc_cols != plan.tmem_cols:
        return False
    names = tuple(
        _tensor_arg(cg, ref) for ref in (plan.k, plan.w, plan.u, plan.g, plan.h)
    )
    k, w, u, g, h = names
    _emit_module_imports(cg)
    geometry = plan.geometry
    cg.cute_wrapper_plans.append(
        {
            "kind": GDN_RECURRENCE_KIND,
            "k_name": k,
            "w_name": w,
            "u_name": u,
            "g_name": g,
            "h_name": h,
            "batch": geometry.batch,
            "heads": geometry.heads,
            "seqlen": geometry.seqlen,
            "chunk": geometry.chunk,
            "dhead": geometry.dhead,
            "dstate": geometry.dstate,
            "num_chunks": geometry.num_chunks,
            "block_v": plan.block_v,
            "epilogue_warps": plan.epilogue_warps,
            "stages": plan.stages,
            "token_groups": plan.token_groups,
            "mma_m": plan.mma_m,
            "threads": plan.threads,
            "tmem_cols": plan.tmem_cols,
            "smem_bytes": plan.smem_bytes,
            "state_col": tmem.state_col,
            "state_image_col": tmem.state_image_col,
            "acc_col": tmem.acc_col,
            "update_image_col": tmem.update_image_col,
            "device_abi": GDN_RECURRENCE_DEVICE_ABI,
        }
    )
    df.placeholder_args.update(names)
    df.preamble = []
    df.body = [ast.Pass()]
    cg.cute_uses_matmul = True
    return True


__all__ = [
    "CuteGdnRecurrencePlan",
    "GdnRecurrenceGeometry",
    "codegen_gdn_recurrence",
    "detect_gdn_recurrence_search_geometry",
    "gdn_admitted_block_sizes",
    "gdn_recurrence_mma_m_choices_by_tile",
    "gdn_recurrence_stage_choices_by_tile",
    "gdn_recurrence_token_group_choices_by_tile",
    "gdn_recurrence_warp_choices",
    "gdn_seed_block_sizes",
    "plan_gdn_recurrence",
]
