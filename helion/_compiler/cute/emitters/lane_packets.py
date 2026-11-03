"""General-purpose CuTe emitter for lane packets.

A lane loop that the strategy split into an outer loop and a constexpr V-loop
reads and writes its tensors one vector transaction per lane group:
``LanePacketEmitter`` splices the packet around the V-loop of a
``LaneLoopContext`` -- a hoisted ``cute.arch.load(base_ptr, <carrier> x V)``
above it, whose lanes the body extracts with ``_cute_unroll_vec_extract``, or
a compile-time list the body appends to and one ``_cute_store_u16_vec`` /
``_cute_store_u32_vec`` flush after it.  A ``GuardBuilder`` decides what
predicates a whole packet and where a masked-off thread points instead: the
tile lane's guard (``TilePacketGuard``: the uniform mask terms and the
per-thread bound unless the tile provably covers the extent) and the rolled
reduction's (``cute/memory_ops._cute_register_unroll_vec_hoist``: the non-lane
mask and the chunk-in-bounds test) differ, the splice does not.

The three protocol entry points (``_cute_register_tile_unroll_vec_hoist``,
``_cute_register_tile_unroll_vec_store``,
``_cute_register_reduction_unroll_vec_store``) and the dtype tables are kept
under their historical names for the memory lowering and the tests.
"""

from __future__ import annotations

import dataclasses
import functools
import operator
from typing import TYPE_CHECKING
from typing import Protocol

import sympy
import torch

from ...ast_extension import statement_from_string
from ...compile_environment import CompileEnvironment
from ..cache_policy_loads import CachePolicy
from ..cache_policy_loads import hint_wide_packets
from ..packet_access import render_packet_load
from .memory_access import _cute_scalar_pointer_expr

if TYPE_CHECKING:
    import ast

    from ...inductor_lowering import CodegenState
    from ...lane_loop_context import LaneLoopContext
    from ...tile_strategy import PerThreadFlattenedTileStrategy
    from ...tile_strategy import PerThreadNDTileStrategy
    from ..signed_bitfield import PackedStoreValue
    from ..signed_bitfield import SignedByteSite

    CuteLaneTileStrategy = PerThreadNDTileStrategy | PerThreadFlattenedTileStrategy


# Maximum bytes per vector load/store transaction (LDG.128/STG.128).
_CUTE_VECTOR_MAX_BYTES = 16
# A lane of the sunk flat-gather path (``cute_vloop_sink``) may span up to
# four consecutive 16-byte packets, loaded under one predicate: a wider lane
# puts more of a row in flight per thread without a second sweep over the
# rows for the next chunk of columns.
_FLAT_GATHER_SINK_MAX_BYTES = 64

# Dtype -> (cutlass scalar type name, max vector width) of the explicit
# ``cute.arch.load(ptr, ir.VectorType.get([V], elem.mlir_type))`` packet
# loads (the retired ``vec`` reduction mode folded them with a
# ``_cute_pre_vec_fold`` helper that no longer exists).
_CUTE_VECTOR_DTYPES: dict[torch.dtype, tuple[str, int]] = {
    torch.float32: ("cutlass.Float32", _CUTE_VECTOR_MAX_BYTES // 4),
    torch.float16: ("cutlass.Float16", _CUTE_VECTOR_MAX_BYTES // 2),
    torch.bfloat16: ("cutlass.BFloat16", _CUTE_VECTOR_MAX_BYTES // 2),
}

# ``unroll`` mode loads inputs as same-width INTEGER vectors (Uint16 for
# bf16/fp16/int16, Uint32 for fp32/int32) and bitcasts each extracted lane
# back to the original dtype.  This avoids the CuTe DSL crash that fires when
# subscripting a bf16/fp16 vector value, and for fp32 sidesteps the retired
# explicit-vec mode (see ``_cute_vec_kernel_mode``).  Signed 16/32-bit
# integer inputs (an int16 weight cast to bf16 inside the kernel) take the
# same carrier packets; the lane bitcast is exact and the program's own cast
# runs on the extracted scalar unchanged.  Maps the tensor dtype to the
# cutlass scalar type of the extracted lane.
_CUTE_VECTOR_UNROLL_DTYPES: dict[torch.dtype, str] = {
    torch.float16: "cutlass.Float16",
    torch.bfloat16: "cutlass.BFloat16",
    torch.float32: "cutlass.Float32",
    torch.int16: "cutlass.Int16",
    torch.int32: "cutlass.Int32",
}

# Integer carrier type for an unroll-mode vector load of a given dtype.
_CUTE_VECTOR_UNROLL_CARRIER: dict[torch.dtype, str] = {
    torch.float16: "cutlass.Uint16",
    torch.bfloat16: "cutlass.Uint16",
    torch.float32: "cutlass.Uint32",
    torch.int16: "cutlass.Uint16",
    torch.int32: "cutlass.Uint32",
}

# One-byte FP8 and signed INT8 inputs use ``unroll`` mode. Rather than a
# ``VectorType([V], Uint8)`` load (which ICEs at V=8 in the CuTe DSL and
# emits two LDG.32s for V=4), an fp8 vec chunk is loaded as a SINGLE packed
# integer (``Uint32`` for V=4, ``Uint64`` for V=8) — one LDG.32 / LDG.64 —
# and each lane byte is extracted with a shift+mask. FP8 keeps its Uint8
# payload for downstream decoding; INT8 bitcasts back before signed arithmetic.
_CUTE_VECTOR_UNROLL_BYTE_DTYPES: frozenset[torch.dtype] = frozenset(
    {torch.float8_e4m3fn, torch.int8}
)

# Packed-integer cutlass type per total byte width of an fp8 vec chunk.
_CUTE_BYTE_PACK_TYPE: dict[int, str] = {
    1: "cutlass.Uint8",
    2: "cutlass.Uint16",
    4: "cutlass.Uint32",
    8: "cutlass.Uint64",
}


def _cute_is_byte_packed(dtype: torch.dtype) -> bool:
    return dtype in _CUTE_VECTOR_UNROLL_BYTE_DTYPES


def _cute_is_unroll_dtype(dtype: torch.dtype) -> bool:
    """True for dtypes that use ``unroll`` mode: bf16/fp16 (Uint16 vector +
    bitcast) and fp8 (packed-integer load + shift extract)."""
    return (
        dtype in _CUTE_VECTOR_UNROLL_DTYPES or dtype in _CUTE_VECTOR_UNROLL_BYTE_DTYPES
    )


def _cute_unroll_vec_elem_type(dtype: torch.dtype, vec_width: int = 1) -> str:
    """Cutlass load type for an ``unroll``-mode hoisted vec load.

    fp8 loads ``vec_width`` bytes as a single packed integer; bf16/fp16/fp32
    load a same-width integer vector element (the ``VectorType`` width is
    applied by callers).
    """
    if _cute_is_byte_packed(dtype):
        pack = _CUTE_BYTE_PACK_TYPE.get(vec_width)
        assert pack is not None, f"unsupported fp8 vec_width {vec_width}"
        return pack
    return _CUTE_VECTOR_UNROLL_CARRIER[dtype]


def _cute_unroll_vec_load_expr(
    ptr_expr: str,
    dtype: torch.dtype,
    vec_width: int,
    eviction_suffix: str = "",
    *,
    wide_packets: bool = False,
) -> str:
    """Build the ``cute.arch.load(...)`` RHS for an unroll-mode hoist.

    An L2-policy hint (``CachePolicy.is_l2_marker``) is not a
    ``cute.arch.load`` kwarg (it has no L2 policy support): a 16-, 8- or
    4-byte hoist load (a 32-byte one under ``wide_packets``, see
    ``hint_wide_packets``) goes through the inline-PTX
    ``createpolicy.fractional`` helper of its width; byte-packed dtypes and
    2-byte scalars silently drop the hint.
    """
    policy = CachePolicy.from_suffix(eviction_suffix)
    if _cute_is_byte_packed(dtype):
        pack = _cute_unroll_vec_elem_type(dtype, vec_width)
        return f"cute.arch.load({ptr_expr}, {pack}{policy.load_kwargs})"
    return render_packet_load(
        ptr_expr,
        vec_width,
        _cute_unroll_vec_elem_type(dtype),
        helper=policy.vector_helper(
            vec_width * dtype.itemsize, wide_packets=wide_packets
        ),
        kwargs=policy.load_kwargs,
    )


def _cute_unroll_vec_extract(hoist_var: str, idx: str, dtype: torch.dtype) -> str:
    """Per-lane extract expr from a hoisted ``unroll``-mode vec load.

    fp8: ``hoist_var`` is a packed integer (Uint32/Uint64); byte ``idx`` is
    extracted with a shift+mask and returned as a ``Uint8`` (decoded
    downstream). INT8 bytes bitcast back to Int8 before arithmetic.
    bf16/fp16: ``hoist_var`` is a ``Uint16`` vector; lane ``idx``
    is bitcast back to the original dtype.
    """
    if dtype in _CUTE_VECTOR_UNROLL_BYTE_DTYPES:
        value = f"cutlass.Uint8(({hoist_var} >> (8 * ({idx}))) & 0xFF)"
        return f"{value}.bitcast(cutlass.Int8)" if dtype is torch.int8 else value
    elem_dtype = _CUTE_VECTOR_UNROLL_DTYPES[dtype]
    carrier = _CUTE_VECTOR_UNROLL_CARRIER[dtype]
    return f"{carrier}({hoist_var}[{idx}]).bitcast({elem_dtype})"


@dataclasses.dataclass(eq=False)
class CuteTileVecStoreSite:
    """A per-lane store collected into ``list_var`` and flushed after the V-loop.

    ``body_stmt`` stands in for the store inside the constexpr V-loop body once
    the vector protocol is in place: the list append, or the packet binding of
    a packed signed-byte store.  ``scalar_stmt`` is the per-lane scalar store
    it replaced (a grid keeps it in the root body until the wrap) and the one
    to restore when the flush cannot legally run after a later access of the
    tensor (see ``demote_reordered_tile_vec_stores`` in
    ``_compiler/cute/memory_ops.py``); a rolled reduction's site, which is
    never restored, records none.  A vector ``atomic_add`` flushed as one
    ``red.global.add`` after the V-loop registers the same record
    (``_compiler/cute/atomic_ops.py``).
    """

    list_var: str
    tensor_name: str
    body_stmt: ast.stmt
    scalar_stmt: ast.stmt | None
    init_stmt: ast.stmt | None
    flush_stmt: ast.stmt


@dataclasses.dataclass(frozen=True)
class PacketGuard:
    """What predicates a whole packet, and the pointer masked-off threads use.

    ``predicate`` is None when every thread's packet is provably in bounds.
    Otherwise a thread failing it reads ``anchor_ptr_expr`` -- an address
    inside the tensor whose bytes the per-lane mask gate discards -- so the
    vector load never reads past the allocation.
    """

    predicate: str | None
    anchor_ptr_expr: str

    def pointer(self, base_ptr_expr: str) -> str:
        if self.predicate is None:
            return base_ptr_expr
        return f"({base_ptr_expr} if {self.predicate} else {self.anchor_ptr_expr})"

    def build(self) -> PacketGuard:
        return self


class GuardBuilder(Protocol):
    """Builds a packet's ``PacketGuard`` when the hoist is first emitted."""

    def build(self) -> PacketGuard: ...


class LanePacketEmitter:
    """Splices one vector transaction per lane group around a V-loop.

    ``hoist_load`` places a packet load above the context's V-loop once per
    ``(tensor, base pointer, cache tag)`` and ``flush_store`` collects a
    per-lane store into a compile-time list flushed after it, keeping the
    flushes in source order; both read and record their state in the
    ``LaneLoopContext``.
    """

    def __init__(self, context: LaneLoopContext) -> None:
        self.context = context

    def base_pointer(
        self,
        tensor_name: str,
        index_exprs: list[str],
        lane_pos: int,
        lane_base_expr: str | None = None,
    ) -> str:
        """The packet's pointer: the per-thread lane base at the lane axis.

        A flattened multi-dim tile addresses the chunk by its flat lane base
        (the ctx gate guarantees the tensor is contiguous and covers the
        whole iteration space, so the V-wide chunk is memory-contiguous even
        when it straddles a row boundary).  Otherwise the lane-axis index
        (the stride-1 dim) is swapped for the per-lane base -- or for
        ``lane_base_expr`` (``lane_base - n1`` for ``tile.index - n1``) -- so
        the vector access points at the start of the V-wide chunk this thread
        owns.
        """
        base_index_var = self.context.base_index_var
        assert base_index_var is not None
        if self.context.flat_multi:
            index_dtype = CompileEnvironment.current().index_type()
            return f"({tensor_name}.iterator + {index_dtype}({base_index_var}))"
        base_exprs = list(index_exprs)
        base_exprs[lane_pos] = (
            base_index_var if lane_base_expr is None else lane_base_expr
        )
        return _cute_scalar_pointer_expr(tensor_name, base_exprs)

    def hoist_load(
        self,
        state: CodegenState,
        *,
        tensor_name: str,
        dtype: torch.dtype,
        base_ptr_expr: str,
        vec_width: int,
        eviction_suffix: str,
        cache_tag: str | None,
        guard: GuardBuilder,
        var_prefix: str,
        wide_packets: bool = False,
    ) -> tuple[str, ast.stmt | None]:
        """Place the packet load of ``(tensor_name, base_ptr_expr, cache_tag)``
        above the V-loop, or reuse the one already placed.

        Returns the packet variable and the hoist statement (None when the
        packet was already placed).  The guard is built only for a new packet.
        """
        context = self.context
        lane_body = context.lane_body
        assert lane_body is not None
        cache_key = (tensor_name, base_ptr_expr, cache_tag)
        cached = context.vec_loads.get(cache_key)
        if cached is not None:
            return cached[0], None
        hoist_var = state.device_function.new_var(
            f"{var_prefix}{len(context.vec_loads)}", dce=False
        )
        context.vec_loads[cache_key] = (hoist_var, dtype)
        pointer = guard.build().pointer(base_ptr_expr)
        load_expr = _cute_unroll_vec_load_expr(
            pointer, dtype, vec_width, eviction_suffix, wide_packets=wide_packets
        )
        hoist_stmt = statement_from_string(f"{hoist_var} = {load_expr}")
        # Insert the hoist just BEFORE the constexpr V-loop.
        lane_body.insert(context.vloop_position(), hoist_stmt)
        return hoist_var, hoist_stmt

    def flush_store(
        self,
        state: CodegenState,
        *,
        tensor_name: str,
        base_ptr_expr: str,
        value_expr: str,
        mask_expr: str | None,
        dtype: torch.dtype,
        var_prefix: str,
        scalar_stmt: ast.stmt | None,
        packed_values: PackedStoreValue | None = None,
    ) -> ast.stmt:
        """Collect a per-lane store and flush it as one vector after the V-loop.

        Replaces the per-lane scalar ``(ptr).store(value)`` inside the
        constexpr V-loop with an append onto a compile-time Python list, and
        splices ONE ``_cute_store_u16_vec(base_ptr, vals)`` (bf16/fp16;
        ``_cute_store_u32_vec`` for fp32) flush AFTER the V-loop, emitting an
        ST.64/ST.128 instead of V scalar stores.  ``mask_expr`` predicates the
        whole flush, so the caller passes only terms uniform across the V
        lanes.  With ``packed_values`` the flush converts a whole signed byte
        packet at once, so the site appends nothing: it binds the packet under
        the flush operand's name, which keeps the store's place in the body
        for the lane-loop distribution (``PackedStoreValue``).  Returns the
        per-lane site statement (the append, or the packet binding).
        """
        context = self.context
        lane_body = context.lane_body
        assert lane_body is not None
        site_index = len(context.vec_stores)
        list_var = state.device_function.new_var(f"{var_prefix}{site_index}", dce=False)
        init_stmt: ast.stmt | None = None
        if packed_values is None:
            init_stmt = statement_from_string(f"{list_var} = []")
            lane_body.insert(context.vloop_position(), init_stmt)
        carrier = _CUTE_VECTOR_UNROLL_CARRIER[dtype]
        flush_helper = (
            "_cute_store_u32_vec" if dtype.itemsize == 4 else "_cute_store_u16_vec"
        )
        flush_values = (
            list_var if packed_values is None else packed_values.flush_operand(list_var)
        )
        flush_expr = f"{flush_helper}({base_ptr_expr}, {flush_values})"
        if mask_expr is not None:
            flush_stmt = statement_from_string(f"if {mask_expr}:\n    {flush_expr}")
        else:
            flush_stmt = statement_from_string(flush_expr)
        # Insert after the V-loop AND after any earlier sites' flushes so the
        # emitted store order matches the source order.
        lane_body.insert(context.vloop_position() + 1 + site_index, flush_stmt)
        if packed_values is not None:
            body_stmt = statement_from_string(f"{list_var} = {packed_values.carrier}")
        else:
            body_stmt = statement_from_string(
                f"{list_var}.append(({value_expr}).bitcast({carrier}))"
            )
        context.add_vec_store(
            CuteTileVecStoreSite(
                list_var,
                tensor_name,
                body_stmt,
                scalar_stmt,
                init_stmt,
                flush_stmt,
            )
        )
        return body_stmt


@dataclasses.dataclass
class TilePacketGuard:
    """The guard of a tile lane's packet.

    Guards the LDG against per-thread OOB: on the very last grid block + tail
    outer-tile iter, a thread whose vec base equals ``numel`` would otherwise
    read past the end of the underlying allocation (the next row doesn't
    exist for the last grid block).  Masked-off threads read an "anchor
    pointer" inside the tensor instead (the per-thread base of the FIRST
    outer-tile iter, which is the ``base_ptr_expr`` with the outer-lane index
    folded to 0); the fetched bytes are then ignored downstream by the
    per-lane mask gate that wraps the bitcast result.  ``uniform_mask``
    carries the mask terms other than the lane mask (an outer tile mask, the
    bound of a gathered row coordinate) the caller proved uniform across the
    V lanes and available above the V-loop; they decide the whole packet and,
    since such a coordinate can itself be out of range, send masked threads
    to the tensor's first element.
    """

    state: CodegenState
    strategy: CuteLaneTileStrategy
    context: LaneLoopContext
    tensor_name: str
    index_exprs: list[str]
    lane_pos: int
    base_ptr_expr: str
    uniform_mask: str | None

    def build(self) -> PacketGuard:
        from ...tile_strategy import DeviceGridState

        state = self.state
        strategy = self.strategy
        context = self.context
        block_id = context.block_id
        base_index_var = context.base_index_var
        assert base_index_var is not None
        env_local = CompileEnvironment.current()
        if context.flat_multi:
            # Bounds are in FLAT elements over the whole iteration space.
            numel = functools.reduce(  # pyrefly: ignore
                operator.mul,
                [env_local.block_sizes[bid].numel for bid in strategy.block_ids],
            )
        else:
            numel = env_local.block_sizes[block_id].numel
        numel_expr = state.sympy_expr(numel)
        block_pos = strategy.block_ids.index(block_id)
        bs_obj = strategy.block_size
        if isinstance(bs_obj, (list, tuple)):
            bs_obj = bs_obj[block_pos]
        static_bs = strategy._configured_block_size_int(bs_obj)
        mask_vars = getattr(strategy, "mask_vars", None)
        if isinstance(mask_vars, dict):
            # PerThreadNDTileStrategy: per-block mask registry.
            mask_elided = block_id in mask_vars and mask_vars[block_id] is None
        else:
            # FlattenedTileStrategy keeps a single (elision-aware) mask var.
            mask_elided = strategy.mask_var(block_id) is None
        try:
            numel_int = int(numel)
        except (TypeError, ValueError):
            numel_int = None
        # GRID tiles have no software-pipelined prefetch past the end, so a
        # mask-free extent that divides evenly into blocks means every
        # per-thread vec base is provably in-bounds.  (The pipelining pass'
        # pipeline_inner_loads only matches an outer ``range`` loop wrapping
        # an inner lane For, which a grid lane loop can never be — if that
        # ever changes, this elision must learn about it.)
        loops_for_block = state.codegen.active_device_loops.get(block_id)
        is_grid_state = bool(loops_for_block) and isinstance(
            loops_for_block[-1], DeviceGridState
        )
        bound_expr = numel_expr
        if not context.flat_multi and loops_for_block:
            # The lane base is a coordinate: a tile that starts past zero
            # (``hl.tile(begin, end)``) ends at ``begin + numel``.  A
            # flattened multi-dim tile always starts at zero.
            info = loops_for_block[-1].block_id_to_info[block_id]
            if info.begin_expr is not None:
                bound_expr = state.sympy_expr(sympy.Add(info.begin_expr, numel))
            elif info.begin_var_name is not None:
                bound_expr = f"({info.begin_var_name}) + ({numel_expr})"
        guard_terms: list[str] = []
        if self.uniform_mask is not None:
            # Outer-row / gathered-coordinate bounds the caller proved uniform
            # across the V lanes decide the whole packet.
            guard_terms.append(f"({self.uniform_mask})")
        if not (
            mask_elided
            and isinstance(static_bs, int)
            and numel_int is not None
            and (
                static_bs >= numel_int or (is_grid_state and numel_int % static_bs == 0)
            )
        ):
            guard_terms.append(f"{base_index_var} < {bound_expr}")
        if not guard_terms:
            # Single-trip tile loop whose block provably covers the extent
            # (the strategy elided the bounds mask): every per-thread vec
            # base is in-bounds and there is no next-iteration prefetch to
            # clamp (the software-pipelining pass, which prefetches one
            # tile PAST the loop end, needs the guard on multi-trip
            # loops).  Dropping the pointer select saves ~14 registers per
            # thread on the resident-row softmax family.  Same for exact
            # grid tilings (numel % block == 0).
            return PacketGuard(None, "")
        # Build the "anchor" pointer: same index_exprs but with the
        # inner reduction-axis index forced to 0.  This is the
        # ``tile_offset == 0, lane_var == 0, vec_lane_var == 0`` base
        # for the very first outer-tile iter, which is always
        # in-bounds for any grid block.
        if context.flat_multi:
            index_dtype_local = CompileEnvironment.current().index_type()
            anchor_ptr_expr = f"({self.tensor_name}.iterator + {index_dtype_local}(0))"
        else:
            anchor_exprs = list(self.index_exprs)
            anchor_exprs[self.lane_pos] = "0"
            if self.uniform_mask is not None:
                # A masked outer row or gathered coordinate can itself be
                # out of range, so the anchor cannot keep it: point at the
                # tensor's first element instead (in bounds for any
                # non-empty tensor); the per-lane mask gate discards the
                # fetched bytes.
                anchor_exprs = ["0"] * len(anchor_exprs)
            anchor_ptr_expr = _cute_scalar_pointer_expr(self.tensor_name, anchor_exprs)
        return PacketGuard(" and ".join(guard_terms), anchor_ptr_expr)


def _cute_register_tile_unroll_vec_store(
    state: CodegenState,
    context: LaneLoopContext,
    tensor_name: str,
    index_exprs: list[str],
    value_expr: str,
    mask_expr: str | None,
    dtype: torch.dtype = torch.float16,
    *,
    scalar_stmt: ast.stmt,
    lane_axis_pos: int | None = None,
    packed_values: PackedStoreValue | None = None,
) -> ast.stmt | None:
    """Vector-store counterpart of ``_cute_register_tile_unroll_vec_hoist``.

    Caller must guarantee the per-element mask is uniform across the V
    lanes (``numel % V == 0``, no extra_mask); the flush reuses the scalar
    mask expression, whose per-element vars hold the last unrolled lane's
    (uniform) value after the V-loop.  ``scalar_stmt`` is the per-lane scalar
    store this site replaces; the site is recorded in ``context.vec_stores``
    so a later access of the tensor in the same lane loop can restore it.
    Returns the per-lane site statement, or None when the lane context isn't
    available.
    """
    if (
        context.base_index_var is None
        or context.lane_body is None
        or context.vloop is None
    ):
        return None
    emitter = LanePacketEmitter(context)
    lane_pos = (
        lane_axis_pos if lane_axis_pos is not None else context.lane_axis(index_exprs)
    )
    return emitter.flush_store(
        state,
        tensor_name=tensor_name,
        base_ptr_expr=emitter.base_pointer(tensor_name, index_exprs, lane_pos),
        value_expr=value_expr,
        mask_expr=mask_expr,
        dtype=dtype,
        var_prefix=f"_tile_store_vals_{context.block_id}_",
        scalar_stmt=scalar_stmt,
        packed_values=packed_values,
    )


def _cute_register_reduction_unroll_vec_store(
    state: CodegenState,
    context: LaneLoopContext,
    tensor_name: str,
    index_exprs: list[str],
    value_expr: str,
    mask_expr: str | None,
    dtype: torch.dtype = torch.float16,
) -> ast.stmt | None:
    """Reduction-lane variant of ``_cute_register_tile_unroll_vec_store``.

    Consume sweeps of rolled reductions (layernorm / rmsnorm epilogues)
    store one fp16/bf16 element per constexpr V-loop iter; this collects
    the V per-lane values into a compile-time list and flushes them with a
    single ``_cute_store_u16_vec`` after the V-loop (ST.64/ST.128 instead
    of V scalar 2-byte stores).

    ``mask_expr`` predicates the whole flush, so the caller passes only terms
    that are uniform across the V lanes: an outer row mask, the chunk-level
    bounds mask of a roll whose extent is a multiple of V, or the whole-chunk
    predicate of a roll whose straddling tail chunk is stored per element.
    """
    if (
        context.base_index_var is None
        or context.lane_body is None
        or context.vloop is None
    ):
        return None
    emitter = LanePacketEmitter(context)
    # The inner reduction-axis index_expr is the last entry (same layout as
    # ``_cute_register_unroll_vec_hoist``).
    return emitter.flush_store(
        state,
        tensor_name=tensor_name,
        base_ptr_expr=emitter.base_pointer(
            tensor_name, index_exprs, len(index_exprs) - 1
        ),
        value_expr=value_expr,
        mask_expr=mask_expr,
        dtype=dtype,
        var_prefix="_reduction_store_vals_",
        scalar_stmt=None,
    )


def _cute_register_tile_unroll_vec_hoist(
    state: CodegenState,
    context: LaneLoopContext,
    strategy: CuteLaneTileStrategy,
    tensor: torch.Tensor,
    tensor_name: str,
    index_exprs: list[str],
    vec_width: int,
    eviction_suffix: str = "",
    *,
    lane_axis_pos: int | None = None,
    mask_expr: str | None = None,
    signed_byte_site: SignedByteSite | None = None,
    uniform_mask: str | None = None,
    lane_base_expr: str | None = None,
) -> str:
    """Tile-loop variant of ``_cute_register_unroll_vec_hoist`` for the
    per-thread tile strategies' lane loops.

    Splices a single ``cute.arch.load(base_ptr, <elem>x V)`` into the
    outer-lane body (above the constexpr V-loop) and returns the
    per-element extract expression so the existing scalar pipeline keeps
    working.  bf16/fp16 load as ``Uint16`` and bitcast; fp8 loads ``vec_width``
    bytes as one packed integer that the matmul fallback decodes downstream.

    ``uniform_mask`` carries the mask terms other than the lane mask (an
    outer tile mask, the bound of a gathered row coordinate, an
    ``extra_mask`` proven uniform across the packet) that the caller proved
    uniform across the V lanes and available above the V-loop
    (``TilePacketGuard``).  ``lane_base_expr`` replaces the block's lane base
    as the lane-axis coordinate of the packet (``lane_base - n1`` for
    ``tile.index - n1``).
    """
    assert context.base_index_var is not None
    assert context.lane_body is not None
    vec_lane_var = context.vec_lane_var
    assert vec_lane_var is not None
    emitter = LanePacketEmitter(context)
    if context.flat_multi:
        lane_pos = -1
    else:
        # The position is the last entry for a row-major lhs, or the recorded
        # position for a K-major rhs.
        lane_pos = (
            lane_axis_pos
            if lane_axis_pos is not None
            else context.lane_axis(index_exprs)
        )
    base_ptr_expr = emitter.base_pointer(
        tensor_name, index_exprs, lane_pos, lane_base_expr
    )
    hoist_var, hoist_stmt = emitter.hoist_load(
        state,
        tensor_name=tensor_name,
        dtype=tensor.dtype,
        base_ptr_expr=base_ptr_expr,
        vec_width=vec_width,
        eviction_suffix=eviction_suffix,
        cache_tag=uniform_mask,
        guard=TilePacketGuard(
            state,
            strategy,
            context,
            tensor_name,
            index_exprs,
            lane_pos,
            base_ptr_expr,
            uniform_mask,
        ),
        var_prefix=f"_tile_unroll_vec_{context.block_id}_",
        wide_packets=hint_wide_packets(state.config.cute_wide_packet_hints),
    )
    if (
        tensor.dtype is torch.int8
        and state.config.config.get("cute_signed_bitfield_bf16") is True
    ):
        from ..signed_bitfield import record_signed_byte_packet

        record_signed_byte_packet(
            state,
            context,
            tensor,
            vec_width,
            index_exprs,
            hoist_var,
            hoist_stmt,
            mask_expr,
            signed_byte_site,
        )
    return _cute_unroll_vec_extract(hoist_var, vec_lane_var, tensor.dtype)
