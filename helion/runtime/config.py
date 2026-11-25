from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import struct
from typing import Hashable
from typing import Literal
from typing import TypeVar
from typing import cast
import uuid


@dataclass(frozen=True, eq=False)
class _ConfigObjectIdentity:
    value: object

    def __hash__(self) -> int:
        return id(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ConfigObjectIdentity) and self.value is other.value


def _config_value_key(value: object) -> Hashable:
    """In-memory identity for values that may affect generated code.

    Preserve order in mappings and bits in floating-point values. Hashable
    opaque objects retain their equality contract: this includes complex
    numbers, frozensets, and numeric types that are not Python float subclasses.
    Unhashable list and dict subclasses use identity plus their contents:
    sharing less is safer than
    dropping their custom equality or invoking arbitrary copy hooks. As with
    other opaque tunables, mutation of custom attributes or external state is
    outside the cache contract.
    """
    kind = type(value)
    if kind is float:
        return kind, struct.pack("!d", cast("float", value))
    if isinstance(value, float):
        return kind, struct.pack("!d", value), value
    if kind in (list, tuple):
        return kind, tuple(
            _config_value_key(item)
            for item in cast("list[object] | tuple[object, ...]", value)
        )
    if kind in (dict, OrderedDict):
        return kind, tuple(
            (_config_value_key(key), _config_value_key(item))
            for key, item in cast("dict[object, object]", value).items()
        )
    if kind.__hash__ is None:
        if isinstance(value, list):
            return (
                kind,
                _ConfigObjectIdentity(value),
                tuple(_config_value_key(item) for item in value),
            )
        if isinstance(value, dict):
            return (
                kind,
                _ConfigObjectIdentity(value),
                tuple(
                    (_config_value_key(key), _config_value_key(item))
                    for key, item in value.items()
                ),
            )
    return kind, cast("Hashable", value)


def _config_key(config: Config) -> Hashable:
    """Ignore option order; preserve code-relevant identity within each value."""
    return frozenset(
        (name, _config_value_key(value)) for name, value in config.config.items()
    )


_ConfigValue = TypeVar("_ConfigValue")


def _copy_config_structure(value: _ConfigValue) -> _ConfigValue:
    """Copy known config containers while preserving opaque leaf objects."""
    if type(value) is OrderedDict:
        return cast(
            "_ConfigValue",
            OrderedDict(
                (key, _copy_config_structure(item)) for key, item in value.items()
            ),
        )
    if type(value) is dict:
        return cast(
            "_ConfigValue",
            {key: _copy_config_structure(item) for key, item in value.items()},
        )
    if type(value) is list:
        return cast("_ConfigValue", [_copy_config_structure(item) for item in value])
    if type(value) is tuple:
        return cast(
            "_ConfigValue", tuple(_copy_config_structure(item) for item in value)
        )
    return value


IndexingLiteral = Literal["pointer", "tensor_descriptor", "block_ptr"]
PidTypeLiteral = Literal[
    "flat",
    "xyz",
    "persistent_blocked",
    "persistent_interleaved",
]
CrossLoopPipelineLiteral = Literal["barrier", "static", "dynamic"]
EvictionPolicyLiteral = Literal["", "first", "last"]
LoadCacheModifierLiteral = Literal["", ".cg"]
StoreCacheModifierLiteral = Literal["", ".cs", ".wt"]
CuteAsyncLoadCacheLiteral = Literal["cg", "ca"]
CuteAsyncStorePolicyLiteral = Literal["default", "l2_evict_last"]
CuteStorePolicyLiteral = Literal["default", "streaming", "l2_evict_last"]
CuteCompilerPipelineLiteral = Literal["default", "opt2"]
CuteExactDivFormLiteral = Literal["redo", "plain"]
CuteAffineScanScheduleLiteral = Literal[
    "ordinary",
    "direct_m16n8_v1",
    "direct_m16n16_v1",
]
CuteHostPairedSumLiteral = Literal["off", "mapped", "narrow"]
CutePdlReleaseLiteral = Literal[
    "auto", "entry", "none", "exit", "last_wave", "loader_tail"
]
NumSmMultiplierLiteral = int
MaxnregLiteral = int | None


class Config(Mapping[str, object]):
    config: dict[str, object]

    def __init__(
        self,
        *,
        # Core properties
        block_sizes: list[int] | None = None,
        num_threads: list[int] | int | None = None,
        loop_orders: list[list[int]] | None = None,
        flatten_loops: list[bool] | None = None,
        l2_groupings: list[int] | None = None,
        reduction_loops: list[int | None] | None = None,
        range_unroll_factors: list[int] | None = None,
        range_warp_specializes: list[bool | None] | None = None,
        range_num_stages: list[int] | None = None,
        range_multi_buffers: list[bool | None] | None = None,
        range_flattens: list[bool | None] | None = None,
        static_ranges: list[bool] | None = None,
        pallas_load_buffer_count: list[int] | None = None,
        pallas_worklist_buffer_count: list[int] | None = None,
        pallas_worklist_bucket_rows: int | None = None,
        load_eviction_policies: (
            EvictionPolicyLiteral | list[EvictionPolicyLiteral] | None
        ) = None,
        load_cache_modifiers: list[LoadCacheModifierLiteral] | None = None,
        store_cache_modifiers: list[StoreCacheModifierLiteral] | None = None,
        cute_store_policy: CuteStorePolicyLiteral | None = None,
        cute_compiler_pipeline: CuteCompilerPipelineLiteral | None = None,
        cute_exact_div_form: CuteExactDivFormLiteral | None = None,
        cute_async_load_stages: int | None = None,
        cute_async_load_lookahead: int | None = None,
        cute_async_load_group_rows: int | None = None,
        cute_async_load_cache: CuteAsyncLoadCacheLiteral | None = None,
        cute_async_store_policy: CuteAsyncStorePolicyLiteral | None = None,
        cute_bf16x2_recurrence: bool | None = None,
        cute_signed_bitfield_bf16: bool | None = None,
        cute_proven_bounds: bool | None = None,
        cute_affine_scan_schedule: CuteAffineScanScheduleLiteral | None = None,
        cute_rng_packet: bool | None = None,
        cute_independent_reduction: bool | None = None,
        cute_replicated_reduction: bool | None = None,
        cute_vector_packet_unroll: bool | None = None,
        cute_vloop_sink: bool | None = None,
        cute_grid_lane_jam: bool | None = None,
        cute_scratch_forwarding: bool | None = None,
        cute_pair_lane_math: bool | None = None,
        cute_lane_unroll: int | None = None,
        cute_unpadded_rows: bool | None = None,
        cute_bool_byte_packets: bool | None = None,
        cute_uniform_lane_compares: bool | None = None,
        cute_tail_peel: bool | None = None,
        cute_guarded_lanes: bool | None = None,
        cute_fmnmx_extrema: bool | None = None,
        cute_predicated_element_loads: bool | None = None,
        cute_store_packet_tiers: bool | None = None,
        cute_defer_tile_reduce: bool | None = None,
        cute_accumulate_stores: bool | None = None,
        cute_staged_store: bool | None = None,
        cute_wide_packet_hints: bool | None = None,
        cute_scalar_load_hints: bool | None = None,
        cute_narrow_load_hints: bool | None = None,
        cute_pdl: bool | None = None,
        cute_pdl_capture_overlap: bool | None = None,
        cute_pdl_release: CutePdlReleaseLiteral | None = None,
        cute_pdl_spread: bool | None = None,
        cute_warp_aggregated_atomics: bool | None = None,
        cute_privatize_atomics: bool | None = None,
        cute_packet_prefetch: int | None = None,
        cute_reduction_pipeline_depth: int | None = None,
        cute_host_paired_sum: CuteHostPairedSumLiteral | None = None,
        num_warps: int | None = None,
        num_stages: int | None = None,
        pid_type: PidTypeLiteral | None = None,
        cross_loop_pipeline: CrossLoopPipelineLiteral | None = None,
        num_sm_multiplier: NumSmMultiplierLiteral | None = None,
        maxnreg: MaxnregLiteral = None,
        host_tensor_descriptors: bool | None = None,
        indexing: IndexingLiteral | list[IndexingLiteral] | None = None,
        atomic_indexing: IndexingLiteral | list[IndexingLiteral] | None = None,
        prefetch_indexing: IndexingLiteral | list[IndexingLiteral] | None = None,
        advanced_controls_file: str | None = None,
        epilogue_subtile: int | None = None,
        xcd_remap: bool | None = None,
        reduction_fusion: bool | None = None,
        # For user-defined properties
        **kwargs: object,
    ) -> None:
        """
        Initialize a Config object.

        Args:
            block_sizes: Controls tile sizes for hl.tile invocations.
            num_threads: Target thread count per axis (backend-specific).
            loop_orders: Permutes iteration order of tiles.
            l2_groupings: Reorders program IDs for L2 cache locality.
            reduction_loops: Configures reduction loop behavior.
            range_unroll_factors: Loop unroll factors for tl.range calls.
            range_warp_specializes: Warp specialization for tl.range calls.
            range_num_stages: Number of stages for tl.range calls.
            range_multi_buffers: Controls disallow_acc_multi_buffer for tl.range calls.
            range_flattens: Controls flatten parameter for tl.range calls.
            static_ranges: Whether to use tl.static_range instead tl.range.
            pallas_load_buffer_count: Pallas-only load buffer count (1 or 2) for
                each input tensor. Tensors without an existing DMA route use the
                ordinary path.
            pallas_worklist_buffer_count: Pallas-only pipeline buffer count (2 to
                8) for each input tensor of a compact-worklist kernel
                (``pallas_worklist_grouping`` 1 or 2). Inputs the worklist
                pipeline does not prefetch (e.g. SMEM offsets) ignore it.
            pallas_worklist_bucket_rows: Pallas-only row bucket of a
                compact-worklist kernel whose compact tile holds only static
                inner loops (``pallas_worklist_grouping`` 1). A positive value
                smaller than the compact block runs each work item's body on
                its live rows rounded up to a multiple of it instead of the
                whole block.
                0 (the default) runs the whole block.
            load_eviction_policies: Eviction policies for load operations. A single
                value applies to every load; a list specifies one value per load.
                Valid values are "", "first", and "last". CuTe also supports
                "nc_l2_first" for proven read-only 16-byte or 32-byte packets.
            load_cache_modifiers: Cache modifiers for load operations ("", ".cg").
            store_cache_modifiers: Cache modifiers for store operations ("", ".cs", ".wt").
            cute_store_policy: Cache policy for proven full aligned 16-byte CuTe
                global store packets ("default", "streaming" = evict-first
                ``st.global.cs`` for outputs written once, or "l2_evict_last").
                The packets ``cute_staged_store`` flushes are not covered:
                they keep the default policy.
            cute_compiler_pipeline: CuTe compiler optimization pipeline. "default"
                leaves SDK options unchanged; "opt2" requests optimization level 2.
                Explicit SDK compiler environment overrides still take precedence.
            cute_async_load_stages: Shared-memory ring stages for eligible CuTe
                in-place 16-byte state loads. Zero disables the transformation.
            cute_async_load_lookahead: Async-copy groups kept ahead of compute.
            cute_async_load_group_rows: Per-thread row iterations in each group.
            cute_exact_div_form: Under ``Settings.cute_exact_division``
                (IEEE fp32 division, off by default), how CuTe lowers an fp32
                division by a loop-invariant divisor (a row sum, a scalar
                argument); both forms give the IEEE quotient bit for bit.
                ``"redo"`` (default) speculates the Markstein sequence over
                the loop and re-runs the loop with ``div.rn`` on a hard case
                (a loop the redo cannot take keeps its plain division),
                ``"plain"`` keeps one ``div.rn`` per element.
            cute_async_load_cache: PTX cp.async cache policy ("cg" or "ca").
            cute_async_store_policy: Cache policy for the matching in-place
                16-byte state store ("default" or "l2_evict_last").
            cute_bf16x2_recurrence: Pack a structurally proven BF16 rank-one
                recurrence into native BF16x2 operations.
            cute_signed_bitfield_bf16: Convert proved signed byte fields to BF16
                using the existing vector load/store packets. Disabled by default.
            cute_proven_bounds: Remove CuTe index guards only when exact launch
                dimensions and cache-specialized tensor sizes prove them true,
                and divide the Int32 indices those facts prove nonnegative
                (``//``, ``%``) in unsigned arithmetic.
            cute_affine_scan_schedule: Physical schedule for a compatible affine
                scan. ``"ordinary"`` disables the direct lowering;
                ``"direct_m16n8_v1"`` and ``"direct_m16n16_v1"`` select the
                measured direct schedule profiles.
            cute_independent_reduction: Sum each vector lane independently in
                FP32 before combining lanes, preserving the element expression.
                This changes summation order. Disabled by default.
            cute_replicated_reduction: Replicate single-use whole-CTA sums in
                each warp using one shared-memory barrier. Disabled by default.
            cute_vector_packet_unroll: Fully unroll small straight-line vector
                packet loops. Disabled by default to retain compact code.
            cute_vloop_sink: Sink a grid constexpr vector loop into the serial
                reduction nest it wraps (column sums): one vector load per row,
                per-lane register accumulators, one grouped cross-thread combine
                per row tile. A gathered coordinate that the masks of its tiles
                keep inside its dim carries no bound of its own, so the sink can
                load the gather in packets; every lane reads the same element as
                before. Disabled by default. Kernels where nothing can be sunk
                generate exactly the knob-off code.
            cute_grid_lane_jam: Interleave the lanes of a grid lane loop into
                the device loops it wraps: the lanes' copies of each
                statement run side by side, their loads of consecutive
                elements become one vector load and their grouped reduces
                one fragment reduce. The results are bitwise unchanged,
                except that the lanes' adds into a ``cute_privatize_atomics``
                copy are summed per cell before one shared-memory add (that
                knob leaves their order unspecified). Disabled by default; a
                lane loop the pass cannot prove independent across lanes
                keeps the knob-off code.
            cute_scratch_forwarding: Serve a later sweep's reload of what an
                earlier sweep stored into a kernel-local scratch tensor (a
                host ``torch.empty`` only the kernel's loads and stores
                reach) from the earlier sweep's registers, and drop the
                stores, the kernel argument and the allocation of a scratch
                tensor nothing reads any more.  The results are bitwise
                unchanged: the forwarded value passes through an opaque
                register move, as the reload did.  It saves the workspace's
                traffic but keeps its values live across the sweeps.
                Disabled by default.
            cute_pair_lane_math: Run the exps, exact divisions and fp32
                products, sums, differences and fmas of adjacent vector
                lanes together as Blackwell's packed fp32 instructions
                (sm_100): fewer instructions per element, at the cost of
                the registers the second lane holds early. The results are
                bitwise unchanged. It also folds the maxima and minima of
                bf16 and f16 vector lanes as packed half2 words. Disabled by
                default; offered to the autotuner for kernels with fp32 exps,
                sigmoids, divisions or arithmetic, or bf16/f16 max or min
                reductions, without ``fast_math`` or a matmul.
            cute_lane_unroll: Unroll factor (1, 2, 4, 8, 16) for per-thread
                lane loops: every unrolled lane's loads, the scalar loads of
                its vector loop included, issue before the first of them
                computes, and a vector loop's scalar loads issue before the
                stores it makes after them even when the lane loop has a
                single trip or stays rolled. It takes the row lane loop of a
                sunk vector nest (with cute_vloop_sink), else the lane loops
                of tile axes the config gives fewer threads than elements, of
                looped reductions whose chunk spans more than one vector per
                thread and of persistent reduction rows their threads do not
                cover; kernels where no loop takes it generate exactly the
                knob-off code. A persistent row whose explicit thread count,
                at most a warp, walks its static extent in whole lanes keeps
                that layout under the knob beside a tile axis of several rows
                per CTA (the rows in registers), in a kernel none of whose
                memory accesses can need an order across threads. That test
                is conservative: a tensor the kernel writes must be touched
                by that one store only or by atomics alone, and must share
                no buffer with another tensor the kernel accesses (a row
                written in place rolls). Without the knob, with more threads,
                under cute_vloop_sink or in any other kernel, such a row is
                rolled in chunks of the threads the tile leaves. With the loads
                apart from the arithmetic, LLVM may contract into one FMA a
                float multiply and add that the knob-off code rounds
                separately (a product feeding an add
                whose other operand is a masked load stays in that load's
                branch there), so float results can round differently, and so
                can values computed from them (a comparison, an integer cast,
                an argmax). Integer and boolean values that do not depend on
                such arithmetic, masks drawn from Philox words included, do
                not change.
            cute_unpadded_rows: Run a persistent reduction row whose static
                length is not a power of two unpadded and unmasked when the
                config gives it a whole number of warps that one vector per
                thread covers exactly (768 = 96 threads x 8): the multi-warp
                vector row, combined across warps through shared memory,
                instead of one warp walking the padded row in masked scalar
                lanes. Normalized away where no row has such a layout; a row
                an ``hl.reduce`` reduces keeps its padded slots, which fold
                the reduce's ``other``. A kept row that the code generator
                then runs on another row's threads (a full slice beside
                another) or as a resident reduction refuses the config. Off
                by default.
                The sums run in another order (a fold of each thread's vector,
                then across lanes and warps, rather than across 32 scalar lanes
                and one warp), and without the masked loads' branches LLVM may
                contract float multiply-adds differently, so results can differ
                from the padded row's by more than an ulp.
            cute_bool_byte_packets: Load a bool row that a looped reduction,
                or a persistent row of vectors that has no ``reduction_loops``
                slot, reads along its reduction dim as byte packets: one
                integer of 2, 4 or 8 bytes per lane, each element a byte
                compared against zero, instead of one byte load per element.
                The packets cost fewer instructions and load requests; where a
                later sweep reloads the row from L2, the faster first sweep
                can leave less of it there. Results are bitwise unchanged. Off
                by default; offered to the autotuner for kernels whose looped
                reduction, or persistent row without a ``reduction_loops``
                slot, reads a bool row.
            cute_uniform_lane_compares: Compare a vector's lane base, once per
                vector, instead of each lane where a lane index meets a
                bound that is a multiple of the vector width (a chunk's tile
                mask, the bounds of a de-linearized ``x[r // hw, r % hw]``):
                no lane reaches the next multiple.  Both multiples are
                proven on the generated code (constants, constexpr extents,
                block sizes); a bound only known at run time, as under
                dynamic shapes, keeps the per-lane compare.  The knob also
                gives a Python remainder by a positive constant a known
                factor, so quotients and remainders of one (``(e % 64) //
                32``, ``(e % 64) % 32``) take the vector base's as well;
                those are exact integer identities, behind the knob only so
                that the code with it off is the code without it.  Integer
                and boolean results, masks included, do not change.
                Its only numeric effect is FMA contraction where per-lane
                selects kept products unfused: with the masks one value per
                vector, LLVM may fuse into one FMA a float multiply and add
                that the knob-off code rounds separately, and those
                elements can differ by about one ulp.  Off by default;
                offered to the autotuner on kernels with CuTe lane loops.
            cute_tail_peel: Run a device loop's complete tiles apart from its
                partial last tile (a static extent that is not a multiple of
                the tile, such as an odd vocabulary) so that their lanes load
                8- or 16-byte packets instead of masked scalars; the last tile
                keeps its masks.  Each value is computed as before, from the same
                elements in the same order; like ``cute_uniform_lane_compares``
                its only possible numeric effect is FMA contraction where
                per-lane selects kept products unfused.  In a nest that
                ``cute_vloop_sink`` interchanged, whose lanes load predicated
                packets already, the complete tiles load them unpredicated
                (under the same cache hints) and drop the selects of every
                mask that the loop's bounds, the launch shape and the
                launch's block grid imply (the grid where it is exact: static,
                or over input sizes the bound kernel is specialized on), the
                kernel's own index masks included; tiles that divide the
                extent are unmasked without a split.  Off by default;
                offered to the autotuner on kernels with an ``hl.tile`` loop
                over a static extent that loads a float tensor along its tile
                and no matmul, and elsewhere rejected (an explicit False is
                dropped).
            cute_guarded_lanes: Run the loads and the arithmetic of a tile's
                lanes under the guard of the stores they feed, so a thread
                that the tile masks turn off (a partial last tile, a block
                with more threads than the extent has elements) skips them
                instead of loading from an in-bounds anchor and computing a
                value no store writes.  Selects and load predicates that the
                stores' guard implies are dropped; a packet whose mask the
                guard does not imply (a padding ``extra_mask``) loads
                predicated and reads as zeros where masked off, instead of
                through one select per lane (an fp16 packet never loads
                predicated: the assembler widens the wrong half of a
                predicated packet's word for an fp16 lane without a select,
                so it keeps its plain load and the selects the guard does
                not imply); and a gathered coordinate that
                the masks of its tiles keep inside its dim carries no bound
                of its own.  Every stored value is computed as before from
                the same elements; like ``cute_uniform_lane_compares`` its
                only possible numeric effects are FMA contraction where
                per-lane selects kept products unfused, and which NaN (its
                sign and payload bits) a NaN result is: without the selects
                an fp32 chain over widened half-precision lanes can run in
                half-precision instructions.  Off by default;
                offered to the autotuner on pointwise kernels (no reduction,
                matmul or atomic) with a load through a computed or gathered
                coordinate or with an ``extra_mask``, and elsewhere rejected
                (an explicit False is dropped).
            cute_fmnmx_extrema: Lower an fp32 ``torch.maximum`` /
                ``torch.minimum`` (and the ``relu`` and ``clamp`` beside it)
                to the single NaN-propagating instruction the extremum
                reductions use (FMNMX.NaN) instead of two compares and a
                select per element; over widened bf16 / fp16 lanes the
                assembler then folds the maxima into packed half-precision
                instructions.  NaN results are NaNs in the same places;
                which NaN (its sign and payload bits) is not part of the
                result and can differ.  The sign of a zero
                result can differ: the instruction orders ``-0.0`` below
                ``+0.0`` (IEEE 754-2019 maximum / minimum), where the
                compare-and-select form returns one of its operands by
                position, so ``maximum(-0.0, +0.0)`` is ``+0.0`` in either
                operand order.  Off by default; offered to the autotuner on
                pointwise kernels (no reduction, matmul or atomic) with an
                fp32 ``torch.maximum`` or ``torch.minimum`` of two tensors,
                and elsewhere rejected (an explicit False is dropped).
            cute_predicated_element_loads: Load a single element under a
                mask (a gathered or computed coordinate with an
                ``extra_mask``) with one predicated instruction instead of a
                load inside a conditional, which compiles to a branch around
                the load.  Where the load's only reader is a select under the
                same mask against a constant the element type holds exactly
                (``torch.where(mask, hl.load(x, [...], extra_mask=mask),
                -inf)``, also through widening casts), the load leaves that
                constant where the mask is false and the select is dropped.
                Every stored value is computed as before from the same
                elements; without the selects an fp32 chain over widened
                half-precision elements can run in half-precision
                instructions, so which NaN (its sign and payload bits) a NaN
                result is can differ, as under ``cute_guarded_lanes``.  A
                masked load whose address is not plain arithmetic, or whose
                element is not a 2-, 4- or 8-byte float or signed integer
                (a byte, a bool, an unsigned integer), keeps its
                conditional.  Off by default; offered to the autotuner on
                pointwise kernels (no reduction, matmul or atomic) with an
                ``hl.load`` under an ``extra_mask`` whose last subscript is
                a scaled or gathered coordinate (not a tile or a tile
                shifted by a constant, nor a row offset plus a tile's index
                into a tensor that vector lanes load in packets), and
                elsewhere rejected (an explicit False is dropped).
            cute_store_packet_tiers: Store a vector lane loop's elements in
                narrower packets where a row fails the full packet's
                alignment check at run time (an output whose row stride is
                not a multiple of the packet, such as ``[rows, 30522]``
                bf16), instead of one element at a time with scalar reloads:
                sixteen-byte packets where the row is aligned to them, then
                packets of half the elements, each tier under its own guard,
                down to two elements (four bytes of a 16-bit dtype, eight of
                a 32-bit one), and the scalar loop last.  The stored values
                and their order are unchanged.  Off by default; offered to the
                autotuner on kernels with a one-axis ``hl.tile`` loop inside
                the kernel's outer ``hl.tile`` loop that stores along its
                tile into the last dimension of a float16, bfloat16,
                float32, int32 or uint32 tensor of two or more dimensions
                and no matmul, and elsewhere rejected (an explicit False is
                dropped): the outer loop's own tiles do not open it, though
                their stores take the narrower packets too once it is on.
                Read as False in a config that gives no such loop vector
                lanes of four or more elements (two elements are already
                the narrowest packet); where the stored tensor takes no
                packet of those lanes (a view that is not unit-stride along
                that dimension, or whose base is off the packet's
                boundary); and where its row strides align every row to
                them, so that no check can fail.
            cute_defer_tile_reduce: Combine a carried tile sum across the
                threads once, after its ``hl.tile`` loop, instead of once per
                tile: in ``for tile in hl.tile(n): acc = acc + x[tile].sum(0)``
                each thread keeps the partial sum of its own elements through
                the whole loop, and the warp shuffles (or the shared-memory
                round) that combine the threads run once after it.  This
                changes the summation order: per-thread partial sums accumulate
                across the tiles before the single cross-thread combine, the
                order TorchInductor's looped split reductions use, where the
                knob-off code adds each tile's combined sum to the carried
                value.  The result differs by the rounding of another
                summation order, for which no bound independent of the length
                holds: each order's rounding error is at most ``n - 1`` unit
                roundoffs times the sum of the addends' magnitudes for ``n``
                addends (to first order), so the two results can differ by
                up to twice that, more the more elements are summed and by
                far more than the last bits where the addends cancel.  Only
                fp32 and fp64 sums added to a carried value of their own
                dtype are deferred; integer and half-precision sums stay per
                tile.  Off by default; offered to the autotuner on kernels
                with an ``hl.tile`` loop whose reductions over its tile are
                all such sums, each read only by its carried add, the carried
                value read nowhere else in the loop, and no matmul, and
                elsewhere rejected (an explicit False is dropped).  Where the
                config puts one thread on the tile (nothing to combine),
                gives each thread one element of the tile (as many threads as
                the tile is long) or folds the tile's lanes as vector lanes,
                the knob leaves the code as it is.
            cute_accumulate_stores: Run the stores of a reduction's loop
                that read no reduction result (``out[tile_r, tile_x] = y``
                beside ``acc = acc + y.sum(0)``) in the pass over a thread's
                elements that accumulates the reductions, with the values they
                store, instead of in the pass after the reductions combine,
                which computes those values a second time.  Each element is
                then computed once and stored in the order the loop gives it,
                but a store between one element's loads and the next one's
                holds the next element's loads behind it (the compiler cannot
                prove that they read other memory), unless the element loop's
                loads are issued ahead of it as vector packets.  It therefore
                takes effect only with ``cute_vloop_sink`` and a vector width
                of two or more: a searched config without them normalizes it
                off, and an explicit request is an error.  The stored
                values are bitwise the same; a sum can differ in its last
                bits, since a product the knob-off code reads once (in the
                accumulate pass) contracts into the accumulating add as an
                FMA, and one the store also reads stays rounded.  Off by
                default; offered to the autotuner on
                kernels with a store in the body of an ``hl.tile`` loop that
                also reduces, whose value reads no reduction of that body, and
                no matmul, and elsewhere rejected (an explicit False is
                dropped); a reduction over a whole rdim keeps its consume
                pass's values in registers with ``cute_reduction_reloads``
                instead.  A store into a tensor that another access of the
                loop may touch stays in the later pass.
            cute_staged_store: Write a pointwise kernel's output through a
                shared-memory image of each block's tile: the threads store
                their elements into the image, meet at one barrier, and
                flush it as 16-byte packets, each warp writing 32 consecutive
                packets, so that every 32-byte sector of the output is sent
                to L2 once however the threads' own elements lie in it.  For
                a kernel whose loads are contiguous along one tile and whose
                stores along another (an unfold or sliding window, ``out[n,
                k] = x[n + c * k]``) this is what lets the loads be packets.
                The values and the elements written are those of the
                knob-off code; the flushed packets take the default cache
                policy whatever ``cute_store_policy`` says, so on a staged
                kernel that option changes nothing.  Applied only where the
                generated kernel is proven to write one contiguous,
                16-byte-aligned span of a single output per block, each
                element once, into a tensor no load can alias, at static
                shapes; a block with a partial tile keeps its direct stores,
                and where the proof fails the code is the knob-off code.
                One barrier and ``tile elements x element size`` bytes of
                shared memory per block (at most 48 KiB).  Off by default;
                offered to the autotuner on kernels with static shapes that
                gather a flattened tensor along one tile and store along
                another (bf16, fp16, fp32, int16 or int32 tensors), without
                a matmul, and elsewhere rejected (an explicit False is
                dropped).
            cute_wide_packet_hints: Carry the ``l2_last`` / ``l1_l2_first`` /
                ``l1_l2_last`` load eviction policies onto 32-byte CuTe packets
                (``cute_vector_widths`` 8 for fp32, 16 for bf16/fp16) through
                one 256-bit cache-hinted load; SM100+ with CUDA 13 or newer,
                elsewhere inert. Off by default: a hinted 32-byte packet then
                loads without the hint, as it always has. The autotuner sets
                it on every config it tries that names one of those policies.
            cute_scalar_load_hints: Carry ``load_eviction_policies`` onto the
                CuTe loads of single 4- and 8-byte elements (a gather, a lane
                on a strided dim) in full: the element loads of a tensor take
                the first ``l2_last`` / ``l1_l2_first`` / ``l1_l2_last`` its
                load slots name, and a hinted element load under a mask
                becomes one predicated load. Off by default: an L2 policy
                then reaches packets only and a masked hinted element keeps
                its conditional, as they always have. The autotuner sets it
                on every config it tries that names a policy an element load
                takes (any but ``nc_l2_first``).
            cute_narrow_load_hints: Carry ``load_eviction_policies`` onto the
                CuTe loads of single 2-byte elements (bf16, fp16, int16): the
                element loads of such a tensor take the first policy its load
                slots name (``first`` / ``last`` / ``streaming`` / ``l2_last``
                / ``l1_l2_first`` / ``l1_l2_last``), a masked one as one
                predicated load. A tensor one of whose loads spells out its
                own policy (``hl.load(..., eviction_policy=...)``) takes none.
                Off by default: those loads then go without a hint, as they
                always have. The autotuner neither searches nor sets it: the
                hinted 2-byte load costs instructions of its own and has
                measured slower than the plain one.
            cute_pdl: Launch the kernel as a programmatic dependent of the
                previous kernel in the stream: its launch and prologue overlap
                that kernel's tail and it waits (``griddepcontrol.wait``) ahead
                of its first global memory access; when its own host zero fills
                launch as dependent fill kernels, ahead of its first access to a
                filled tensor instead (the fill kernel releases only after its
                own wait, so every earlier kernel has completed), which lets a
                split-K kernel's operand loads overlap the fill of its atomic
                output. Off by default. Kernels that
                already take part in dependent launch (native plans launched
                with it, bodies with their own wait or release, the
                materialized-fission producer) and pre-sm_90 targets generate
                exactly the knob-off code.
            cute_pdl_capture_overlap: During CUDA graph capture, permit a proved
                paired attention kernel to read inputs before its PDL dependency
                completes when none of its known pending ancestors can write
                those storages. It still waits before every global write.
                Requires ``cute_pdl=True`` and automatic, entry or loader-tail
                release.
                Unknown predecessors, aliases, unsupported wrappers and eager
                calls retain the ordinary early wait. Default autotuning and
                compiler seeds leave it off. An explicit Config, user seed, or
                autotune_config_overrides may opt an eligible kernel in, with
                ``cute_pdl=True`` and ``cute_pdl_spread=False``.
                Eager benchmark timings cannot demonstrate this capture-only
                overlap benefit; a selected configuration is not evidence of a
                speedup for the application's captured graph.
                With this policy enabled, captured topology and parameter
                addresses must remain immutable; tensor contents may change
                between replays. Recapture after rewiring a graph or replacing
                its pointer arguments. Applications that edit captured graphs
                must leave this policy off, including when supplying stored
                configurations or user seeds that previously enabled it.
            cute_pdl_release: Where a ``cute_pdl`` kernel releases its own
                dependents (``griddepcontrol.launch_dependents``), which is what
                lets the next kernel of a sequence launch early too. ``entry``
                releases right after the wait, so the dependents' CTAs become
                resident while this kernel runs; ``exit`` releases after the
                body's last statement; ``none`` leaves the implicit trigger at
                grid completion, after which a dependent starts later than a
                plain stream edge would have launched it; ``last_wave`` releases
                at entry in the CTAs of the launch's last wave only (the CTAs
                whose dispatch-order index lies within the resident capacity of
                the grid's end; the earlier CTAs count as released when they
                exit), so the dependents launch as the last wave starts and
                fill the slots its tail frees, which is plain ``entry`` for a
                grid that fits in one wave and keeps the middle waves of a
                larger grid undisturbed. ``auto`` (the default) picks ``entry``
                while the launch grid is known to fit in one wave of resident
                CTAs (the SM count times the CTAs an SM holds, from the launch
                block or ``cute_min_blocks_per_mp``) and ``last_wave`` past it
                or when the grid is sized at runtime. ``loader_tail`` requires
                proved paired capture overlap: the loader releases with four
                logical pages left to issue, or after a shorter/empty loop.
                Its waits and ancestor-storage proof are unchanged. This choice
                also requires explicit opt-in; default search leaves placement
                at ``auto``. Every placement reads the same data: a
                dependent's own wait blocks until this grid has completed and
                flushed. On the tcgen05 GEMM kernels, whose lowering waits in
                its warp roles and releases after the last MMA, ``entry`` and
                ``last_wave`` add the release right after each of those waits
                and ``exit`` / ``none`` keep the late release alone. Inert when
                ``cute_pdl`` is off.
            cute_pdl_spread: Pad a ``cute_pdl`` kernel's shared memory so that
                at most ``ceil(CTAs / SMs)`` of its CTAs fit on one SM, its
                density on an idle device. A programmatic dependent launches
                while its primary runs and the hardware dispatches its CTAs
                onto whatever SMs have room; when the primary's CTAs leave none
                beside them, a small dependent's CTAs all land on the few SMs
                the primary left idle and run packed there once it completes.
                With the pad the CTAs beyond the idle SMs wait for the
                primary's to retire and take their SMs one each. Off by
                default; inert when ``cute_pdl`` is off, when the kernel's own
                allocation already holds its CTAs to that density, when the
                grid is sized at runtime, and on the families that own their
                dependent launch.
            cute_warp_aggregated_atomics: Combine relaxed Int32 zero/one ticket
                reservations for equal addresses within each active warp. Each
                lane retains its own valid fetch-add return value. Off by default.
            cute_privatize_atomics: Accumulate the relaxed ``hl.atomic_add``
                updates of a small fp32 or int32 tensor (16 KiB in all) that the
                kernel touches only through such adds in a shared-memory copy
                per CTA, and add each changed cell to the tensor once after the
                CTA's last tile. This changes the fp32 summation order. Off by
                default; offered only for tensors of static shape and rank at
                least 1, proven not to alias the kernel's other tensors, whose
                views share one dtype, in kernels without a matmul,
                ``hl.barrier()``, an atomic that is not relaxed, asm not
                declared pure, an access through a stack tensor
                (``hl.stacktensor_like``) or a remote copy or barrier.
            cute_packet_prefetch: Prefetch 2, 4, or 8 independent vector packets
                in a proved complete tile; 0 (the default) keeps the original order.
                Requires cute_proven_bounds and storage-disjointness proof.
            cute_reduction_pipeline_depth: Shared-memory ring depth (2 or 4) for
                the proved CuTe pipelined resident-reduction schedule. Defaults
                to 2. Depth 4 prefetches three row groups ahead and requires
                enough shared memory for all four slots and reduction scratch.
            cute_host_paired_sum: Fuse a proved host FP32 sum(0)/cast pair or
                terminal sum/cast. ``"off"`` (the default) retains Torch calls;
                ``"mapped"`` and ``"narrow"`` change independent-column
                ownership while preserving the audited FP32 sum tree. Unknown
                Torch implementations or unsupported bindings use the original
                operations. Available only for typed, independent host pairs
                or a terminal sum/cast with an effect-free return prefix.
            num_warps: Number of warps per block.
            num_stages: Number of stages for software pipelining.
            pid_type: Program ID type strategy ("flat", "xyz", "persistent_blocked", "persistent_interleaved").
            cross_loop_pipeline: Execution strategy for kernels with
                compiler-inferred cross-loop dependencies. ``"barrier"`` uses
                grid synchronization. ``"static"`` and ``"dynamic"`` (Triton
                only) execute the same compiler-derived dependency schedule
                with fixed worker ownership or one-shot packet dispatch,
                respectively (under the gluon ring, dynamic workers claim the
                tasks of a data-dependent task count past a fixed first wave).
                Unsupported kernels reject this field.
            num_sm_multiplier: Positive integer multiplier for the number of SMs
                in persistent kernels. The autotuner searches powers of two, but
                explicit configurations may use intermediate values.
                Controls multi-occupancy by launching N * num_sms thread blocks instead of just num_sms.
            maxnreg: None or a positive integer maximum number of registers per
                thread (1--256). The autotuner searches None, 32, 64, 128, and
                256, but explicit configurations may use intermediate values.
                Lower values allow higher occupancy but may hurt performance.
                Used with persistent kernels to ensure multi-occupancy can be
                achieved.
            host_tensor_descriptors: Construct tensor descriptors once in the host
                launcher instead of independently in every device program. This
                only affects operations using tensor-descriptor indexing.
            indexing: Indexing strategy for load and store operations. Can be:
                - A single strategy string (all loads/stores use this strategy):
                  indexing="block_ptr"  # backward compatible
                - A list of strategies (one per load/store operation, must specify all):
                  indexing=["pointer", "block_ptr", "tensor_descriptor"]
                - Empty/omitted (all loads/stores default to "pointer")
                Valid strategies: "pointer", "tensor_descriptor", "block_ptr"
                ("block_ptr" needs Triton < 3.9; newer Triton lowers it as "pointer")
            atomic_indexing: Indexing strategy for atomic operations (e.g., hl.atomic_add).
                Same format as ``indexing`` (a single string or a list per atomic op).
                Defaults to "pointer" when omitted.
            prefetch_indexing: Form of each ``hl.prefetch`` on NVIDIA targets of the
                Triton and Gluon backends. Same format as ``atomic_indexing`` (a single
                string or a list per prefetch). "pointer" (the default) issues bulk or
                per-element L2 prefetches; "tensor_descriptor" issues TMA box prefetches
                from a host-built tensor map where a box can describe the subscript.
            advanced_controls_file: Path to a PTXAS control file applied during compilation, or empty string for none.
            epilogue_subtile: Split factor for the epilogue (post-matmul pointwise + store) along
                the N dimension. None = disabled (default), valid values are 2 or 4.
            xcd_remap: AMD CDNA only. Remap program IDs into contiguous per-XCD regions to
                improve L2 locality on multi-XCD GPUs (MI300/MI350). Supported for pid_type
                "flat", "persistent_blocked", and "persistent_interleaved"; composes with
                ``l2_groupings``.
            reduction_fusion: When multiple independent reductions in a looped
                (rolled) reduction read the same operand, fold them into one loop
                with a single shared load instead of re-reading the operand per
                reduction. Trades global-memory bandwidth for register pressure,
                so it is autotuned. Defaults to False.
            **kwargs: Additional user-defined configuration parameters.
        """
        self.config = {}
        core_props = {
            "block_sizes": block_sizes,
            "num_threads": num_threads,
            "loop_orders": loop_orders,
            "flatten_loops": flatten_loops,
            "l2_groupings": l2_groupings,
            "reduction_loops": reduction_loops,
            "range_unroll_factors": range_unroll_factors,
            "range_warp_specializes": range_warp_specializes,
            "range_num_stages": range_num_stages,
            "range_multi_buffers": range_multi_buffers,
            "range_flattens": range_flattens,
            "static_ranges": static_ranges,
            "pallas_load_buffer_count": pallas_load_buffer_count,
            "pallas_worklist_buffer_count": pallas_worklist_buffer_count,
            "pallas_worklist_bucket_rows": pallas_worklist_bucket_rows,
            "load_eviction_policies": load_eviction_policies,
            "load_cache_modifiers": load_cache_modifiers,
            "store_cache_modifiers": store_cache_modifiers,
            "cute_store_policy": cute_store_policy,
            "cute_compiler_pipeline": cute_compiler_pipeline,
            "cute_exact_div_form": cute_exact_div_form,
            "cute_async_load_stages": cute_async_load_stages,
            "cute_async_load_lookahead": cute_async_load_lookahead,
            "cute_async_load_group_rows": cute_async_load_group_rows,
            "cute_async_load_cache": cute_async_load_cache,
            "cute_async_store_policy": cute_async_store_policy,
            "cute_bf16x2_recurrence": cute_bf16x2_recurrence,
            "cute_signed_bitfield_bf16": cute_signed_bitfield_bf16,
            "cute_proven_bounds": cute_proven_bounds,
            "cute_affine_scan_schedule": cute_affine_scan_schedule,
            "cute_rng_packet": cute_rng_packet,
            "cute_independent_reduction": cute_independent_reduction,
            "cute_replicated_reduction": cute_replicated_reduction,
            "cute_vector_packet_unroll": cute_vector_packet_unroll,
            "cute_vloop_sink": cute_vloop_sink,
            "cute_grid_lane_jam": cute_grid_lane_jam,
            "cute_scratch_forwarding": cute_scratch_forwarding,
            "cute_pair_lane_math": cute_pair_lane_math,
            "cute_lane_unroll": cute_lane_unroll,
            "cute_unpadded_rows": cute_unpadded_rows,
            "cute_bool_byte_packets": cute_bool_byte_packets,
            "cute_uniform_lane_compares": cute_uniform_lane_compares,
            "cute_tail_peel": cute_tail_peel,
            "cute_guarded_lanes": cute_guarded_lanes,
            "cute_fmnmx_extrema": cute_fmnmx_extrema,
            "cute_predicated_element_loads": cute_predicated_element_loads,
            "cute_store_packet_tiers": cute_store_packet_tiers,
            "cute_defer_tile_reduce": cute_defer_tile_reduce,
            "cute_accumulate_stores": cute_accumulate_stores,
            "cute_staged_store": cute_staged_store,
            "cute_wide_packet_hints": cute_wide_packet_hints,
            "cute_scalar_load_hints": cute_scalar_load_hints,
            "cute_narrow_load_hints": cute_narrow_load_hints,
            "cute_pdl": cute_pdl,
            "cute_pdl_capture_overlap": cute_pdl_capture_overlap,
            "cute_pdl_release": cute_pdl_release,
            "cute_pdl_spread": cute_pdl_spread,
            "cute_warp_aggregated_atomics": cute_warp_aggregated_atomics,
            "cute_privatize_atomics": cute_privatize_atomics,
            "cute_packet_prefetch": cute_packet_prefetch,
            "cute_reduction_pipeline_depth": cute_reduction_pipeline_depth,
            "cute_host_paired_sum": cute_host_paired_sum,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "indexing": indexing,
            "atomic_indexing": atomic_indexing,
            "prefetch_indexing": prefetch_indexing,
            "pid_type": pid_type,
            "cross_loop_pipeline": cross_loop_pipeline,
            "num_sm_multiplier": num_sm_multiplier,
            "maxnreg": maxnreg,
            "host_tensor_descriptors": host_tensor_descriptors,
            "advanced_controls_file": advanced_controls_file,
            "epilogue_subtile": epilogue_subtile,
            "xcd_remap": xcd_remap,
            "reduction_fusion": reduction_fusion,
        }
        for key, value in core_props.items():
            if value is not None:
                self.config[key] = value
        self.config.update(kwargs)

    def __getitem__(self, key: str) -> object:
        return self.config[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.config)

    def __len__(self) -> int:
        return len(self.config)

    def __repr__(self) -> str:
        return f"helion.{self.__str__()}"

    def __str__(self) -> str:
        args = [f"{key}={value!r}" for key, value in sorted(self.config.items())]
        return f"Config({', '.join(args)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Config):
            return NotImplemented
        return self.config == other.config

    def __hash__(self) -> int:
        return hash(frozenset([(k, _to_hashable(v)) for k, v in self.config.items()]))

    def __getstate__(self) -> dict[str, object]:
        return dict(self.config)

    def __setstate__(self, state: dict[str, object]) -> None:
        self.config = dict(state)

    def to_json(self) -> str:
        """Convert the config to a JSON string."""
        return json.dumps(self.config, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, object]) -> Config:
        """Create a Config from a plain dictionary."""
        obj = Config()
        obj.config = dict(config_dict)
        return obj

    @classmethod
    def from_json(cls, json_str: str) -> Config:
        """Create a Config object from a JSON string."""
        config_dict = json.loads(json_str)
        return cls(**config_dict)  # Changed to use dictionary unpacking

    def minimize(self, config_spec: object) -> Config:
        """
        Return a new Config with values matching effective defaults removed.

        Matching defaults are removed only when normalization preserves the
        selected config. Dependent or unprovable forms retain their fields.

        Args:
            config_spec: The ConfigSpec that defines the defaults for this kernel.

        Returns:
            A new Config with default values removed.
        """
        from ..autotuner.config_fragment import _plain_builtin_fragment_key
        from ..autotuner.config_spec import ConfigSpec
        from ..exc import InvalidConfig

        assert isinstance(config_spec, ConfigSpec)
        default_config = config_spec.default_config()

        # Arbitrary default protocols may have side effects. Keep explicit
        # custom values while minimizing the independent schedule fields.
        custom_keys = {
            key
            for key, fragment in config_spec.user_defined_tunables.items()
            if _plain_builtin_fragment_key(fragment) is None
        }
        if not custom_keys.issubset(self.config):
            # Proving an omitted custom value would call its default again.
            return Config.from_dict(self.config)
        required_keys = {"block_sizes"} | custom_keys

        minimal: dict[str, object] = {}
        for key, value in self.config.items():
            # Keep value if it differs from the default or is required
            default_value = default_config.config.get(key)
            if (
                _config_value_key(value) != _config_value_key(default_value)
                or key in required_keys
            ):
                minimal[key] = value

        candidate = Config.from_dict(minimal)
        try:
            expected = config_spec.normalized_config(self)
            normalized = config_spec.normalized_config(candidate)
            if _config_key(normalized) == _config_key(expected):
                return candidate

            # Effective search defaults may differ from normalization's omitted
            # defaults (for example after shrinking a small kernel's warp count).
            # Restore those fields rather than change the selected configuration.
            for key, value in self.config.items():
                if key not in minimal and (
                    (key in expected) != (key in normalized)
                    or _config_value_key(expected.config.get(key))
                    != _config_value_key(normalized.config.get(key))
                ):
                    minimal[key] = value
            candidate = Config.from_dict(minimal)
            if _config_key(config_spec.normalized_config(candidate)) == _config_key(
                expected
            ):
                return candidate
        except InvalidConfig:
            # Minimization is optional; leave rejected or dependent forms for
            # the usual compilation validation instead of guessing defaults.
            pass
        return Config.from_dict(self.config)

    def save(self, path: str | Path) -> None:
        """Save the config to a JSON file."""
        # Write to temp dir and rename to make the operation atomic
        # in case we are in a multithreaded environment
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        tmp = Path(path).parent / f"tmp.{uuid.uuid4()!s}"
        tmp.write_text(self.to_json())
        os.rename(str(tmp), str(path))

    @classmethod
    def load(cls, path: str | Path) -> Config:
        """Load a config from a JSON file."""
        return cls.from_json(Path(path).read_text())

    @property
    def block_sizes(self) -> list[int]:
        return cast("list[int]", self.config["block_sizes"])

    @property
    def loop_orders(self) -> list[list[int]]:
        return cast("list[list[int]]", self.config.get("loop_orders", []))

    @property
    def num_threads(self) -> list[int]:
        value = self.config.get("num_threads", [])
        if isinstance(value, int):
            return [value]
        return cast("list[int]", value)

    @property
    def flatten_loops(self) -> list[bool]:
        return cast("list[bool]", self.config.get("flatten_loops", []))

    @property
    def reduction_loops(self) -> list[int | None]:
        return cast("list[int | None]", self.config.get("reduction_loops", []))

    @property
    def num_warps(self) -> int:
        from ..autotuner.config_spec import DEFAULT_NUM_WARPS

        return cast("int", self.config.get("num_warps", DEFAULT_NUM_WARPS))

    @property
    def num_stages(self) -> int:
        from ..autotuner.config_spec import DEFAULT_NUM_STAGES

        return cast("int", self.config.get("num_stages", DEFAULT_NUM_STAGES))

    @property
    def l2_groupings(self) -> list[int]:
        return cast("list[int]", self.config.get("l2_groupings", []))

    @property
    def pid_type(self) -> PidTypeLiteral:
        return cast("PidTypeLiteral", self.config.get("pid_type", "flat"))

    @property
    def cross_loop_pipeline(self) -> CrossLoopPipelineLiteral:
        return cast(
            "CrossLoopPipelineLiteral",
            self.config.get("cross_loop_pipeline", "barrier"),
        )

    @property
    def xcd_remap(self) -> bool:
        return cast("bool", self.config.get("xcd_remap", False))

    @property
    def num_sm_multiplier(self) -> int:
        from ..autotuner.config_spec import DEFAULT_NUM_SM_MULTIPLIER

        return cast(
            "int", self.config.get("num_sm_multiplier", DEFAULT_NUM_SM_MULTIPLIER)
        )

    @property
    def maxnreg(self) -> int | None:
        from ..autotuner.config_spec import DEFAULT_MAXNREG

        return cast("int | None", self.config.get("maxnreg", DEFAULT_MAXNREG))

    @property
    def host_tensor_descriptors(self) -> bool:
        return cast("bool", self.config.get("host_tensor_descriptors", False))

    @property
    def reduction_fusion(self) -> bool:
        return cast("bool", self.config.get("reduction_fusion", False))

    @property
    def range_unroll_factors(self) -> list[int]:
        return cast("list[int]", self.config.get("range_unroll_factors", []))

    @property
    def advanced_controls_file(self) -> str:
        return cast("str", self.config.get("advanced_controls_file", ""))

    @property
    def range_warp_specializes(self) -> list[bool | None]:
        return cast("list[bool | None]", self.config.get("range_warp_specializes", []))

    @property
    def range_num_stages(self) -> list[int]:
        return cast("list[int]", self.config.get("range_num_stages", []))

    @property
    def range_multi_buffers(self) -> list[bool | None]:
        return cast("list[bool | None]", self.config.get("range_multi_buffers", []))

    @property
    def range_flattens(self) -> list[bool | None]:
        return cast("list[bool | None]", self.config.get("range_flattens", []))

    @property
    def static_ranges(self) -> list[bool]:
        return cast("list[bool]", self.config.get("static_ranges", []))

    @property
    def pallas_load_buffer_count(self) -> list[int]:
        return cast("list[int]", self.config.get("pallas_load_buffer_count", []))

    @property
    def pallas_worklist_buffer_count(self) -> list[int]:
        return cast("list[int]", self.config.get("pallas_worklist_buffer_count", []))

    @property
    def pallas_worklist_bucket_rows(self) -> int:
        return cast("int", self.config.get("pallas_worklist_bucket_rows", 0))

    @property
    def load_eviction_policies(
        self,
    ) -> EvictionPolicyLiteral | list[EvictionPolicyLiteral]:
        return cast(
            "EvictionPolicyLiteral | list[EvictionPolicyLiteral]",
            self.config.get("load_eviction_policies", []),
        )

    @property
    def load_cache_modifiers(self) -> list[LoadCacheModifierLiteral]:
        return cast(
            "list[LoadCacheModifierLiteral]",
            self.config.get("load_cache_modifiers", []),
        )

    @property
    def store_cache_modifiers(self) -> list[StoreCacheModifierLiteral]:
        return cast(
            "list[StoreCacheModifierLiteral]",
            self.config.get("store_cache_modifiers", []),
        )

    @property
    def cute_async_load_stages(self) -> int:
        return cast("int", self.config.get("cute_async_load_stages", 0))

    @property
    def cute_async_load_lookahead(self) -> int:
        return cast("int", self.config.get("cute_async_load_lookahead", 4))

    @property
    def cute_async_load_group_rows(self) -> int:
        return cast("int", self.config.get("cute_async_load_group_rows", 2))

    @property
    def cute_async_load_cache(self) -> CuteAsyncLoadCacheLiteral:
        return cast(
            "CuteAsyncLoadCacheLiteral",
            self.config.get("cute_async_load_cache", "cg"),
        )

    @property
    def cute_async_store_policy(self) -> CuteAsyncStorePolicyLiteral:
        return cast(
            "CuteAsyncStorePolicyLiteral",
            self.config.get("cute_async_store_policy", "default"),
        )

    @property
    def cute_store_policy(self) -> CuteStorePolicyLiteral:
        return cast(
            "CuteStorePolicyLiteral",
            self.config.get("cute_store_policy", "default"),
        )

    @property
    def cute_compiler_pipeline(self) -> CuteCompilerPipelineLiteral:
        return cast(
            "CuteCompilerPipelineLiteral",
            self.config.get("cute_compiler_pipeline", "default"),
        )

    @property
    def cute_wide_packet_hints(self) -> bool:
        return self.config.get("cute_wide_packet_hints") is True

    @property
    def cute_scalar_load_hints(self) -> bool:
        return self.config.get("cute_scalar_load_hints") is True

    @property
    def cute_narrow_load_hints(self) -> bool:
        return self.config.get("cute_narrow_load_hints") is True

    @property
    def cute_exact_div_form(self) -> CuteExactDivFormLiteral:
        return cast(
            "CuteExactDivFormLiteral",
            self.config.get("cute_exact_div_form", "redo"),
        )

    @property
    def cute_bf16x2_recurrence(self) -> bool:
        return cast("bool", self.config.get("cute_bf16x2_recurrence", False))

    @property
    def cute_proven_bounds(self) -> bool:
        return cast("bool", self.config.get("cute_proven_bounds", False))

    @property
    def cute_affine_scan_schedule(self) -> CuteAffineScanScheduleLiteral:
        return cast(
            "CuteAffineScanScheduleLiteral",
            self.config.get("cute_affine_scan_schedule", "ordinary"),
        )

    @property
    def cute_packet_prefetch(self) -> int:
        return cast("int", self.config.get("cute_packet_prefetch", 0))

    @property
    def cute_reduction_pipeline_depth(self) -> int:
        return cast("int", self.config.get("cute_reduction_pipeline_depth", 2))

    @property
    def indexing(self) -> IndexingLiteral | list[IndexingLiteral]:
        return cast(
            "IndexingLiteral | list[IndexingLiteral]", self.config.get("indexing", [])
        )

    @property
    def atomic_indexing(self) -> IndexingLiteral | list[IndexingLiteral]:
        return cast(
            "IndexingLiteral | list[IndexingLiteral]",
            self.config.get("atomic_indexing", []),
        )

    @property
    def prefetch_indexing(self) -> IndexingLiteral | list[IndexingLiteral]:
        return cast(
            "IndexingLiteral | list[IndexingLiteral]",
            self.config.get("prefetch_indexing", []),
        )

    @property
    def epilogue_subtile(self) -> int | None:
        return cast("int | None", self.config.get("epilogue_subtile", None))


def _to_hashable(x: object) -> object:
    if isinstance(x, list):
        return tuple([_to_hashable(i) for i in x])
    if isinstance(x, dict):
        return tuple(sorted([(k, _to_hashable(v)) for k, v in x.items()]))
    return x
