"""Device-code emitter of the ``cute_mma_dag`` warp mode (see cute_mma_dag.py).

The body is rendered as source text from the :class:`MmaDagPlan` (as
``cute_warp_mma_gemm.py`` does) and parsed into the root tile loop's
statements; the launch keeps the generic flat grid (one CTA per root tile)
and owns the block shape through ``cute_state.owned_root_block_dims``.

Program of one CTA (``32 * warps`` threads; every fragment slot has its own
warp grid of 16x8 ``mma.sync`` atoms; ``S`` operand ring slots)::

    fragment registers <- zeros | per-lane loads of the seed tiles
    cp.async the epilogue's operand tiles (one group; they land during the loop)
    cp.async the source tiles of the first S - 1 loop iterations (one group each)
    for i in loop:
        [S == 1] cp.async this iteration's tiles, commit
        synchronous operand passes (unaligned sources: pair loads, pointwise
            DAG, st.shared into the single-slot swizzled tile)
        snapshots of the carries (staging tile, or pair stores to global)
        accumulators that depend on nothing computed in the iteration:
            zeros, the carry, a rescaled carry no earlier step reads
        exchange tiles of the carry at loop entry (bf16 pairs at the fragment
            coordinates, double-buffered by iteration parity)
        cp.async wait (S - 2 younger groups may stay in flight)
        in-place operand passes: each thread rewrites its own landed packets
            through the pointwise DAG (ld.shared.v4, Inductor lowering, pack,
            st.shared.v4)
        CTA barrier
        [S > 1] cp.async the tiles of iteration i + S - 1 into the slot read
            one iteration ago, commit (empty past the end)
        staging tile -> 16-byte global stores; then the body in program order:
            dot: exchange tiles of fragments computed in this iteration + CTA
                barrier, the remaining accumulator rescale, K / 16 steps of
                ldmatrix(.trans) / register repack / movmatrix + mma.sync
            derived fragment: pointwise DAG per register over the input
                fragments, 2-D tiles loaded as pairs at the coordinates
            store of a dot / derived value: pair stores at its program point
        [S == 1] CTA barrier before the refill
    epilogue: wait for the epilogue tiles, their passes, fragment -> shared
        exchange tiles (bf16 pairs at the fragment coordinates), CTA barrier,
        the epilogue dots (operands from shared tiles, exchange tiles or the
        identity repack of a same-layout fragment), stores of the results

Pointwise values (operand prologues, masks, rescales, store values) go
through Inductor's ``PointwiseLowering.codegen_from_input_asts`` (and the
``where`` / ``scalar_tensor`` aten lowerings) per element, so ``exp`` /
``exp2`` / casts / ``where`` are the expressions the SIMT path emits; 16-bit
values are re-typed to their DSL dtype before the ops run and packed with
``cvt.rn`` pairs, so every rounding step matches.  Fragment elements are
placed by coordinates, so a ``[C, C]`` tile over one reduction dim and an
``hl.arange`` mask need no lane bookkeeping.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import itertools
import math
import textwrap
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language import memory_ops
from ..compile_environment import CompileEnvironment
from .cute_mma_dag import AXIS_LEAD
from .cute_mma_dag import AXIS_LOOP
from .cute_mma_dag import AXIS_RDIM
from .cute_mma_dag import MMA_DAG_HINTED_CP_ASYNC_CHECK
from .cute_mma_dag import MMA_DAG_LOADS_TMA
from .cute_mma_dag import MMA_DAG_TCGEN05
from .cute_mma_dag import MMA_DAG_WARP
from .cute_mma_dag import REGISTER_ROUTES
from .cute_mma_dag import ROUTE_CONVERT
from .cute_mma_dag import ROUTE_DIRECT
from .cute_mma_dag import ROUTE_EXCHANGE
from .cute_mma_dag import ROUTE_IDENTITY
from .cute_mma_dag import ROUTE_INPLACE
from .cute_mma_dag import ROUTE_MOVMATRIX
from .cute_mma_dag import ROUTE_SHUFFLE
from .cute_mma_dag import ROUTE_SYNC
from .cute_mma_dag import Access
from .cute_mma_dag import Axis
from .cute_mma_dag import Derived
from .cute_mma_dag import Dot
from .cute_mma_dag import Loop
from .cute_mma_dag import MmaDagPlan
from .cute_mma_dag import Operand
from .cute_mma_dag import Reduce
from .cute_mma_dag import Scan
from .cute_mma_dag import Step
from .cute_mma_dag import Store
from .cute_mma_dag import VecOp
from .cute_mma_dag import VecStore
from .cute_mma_dag import _decline
from .cute_mma_dag import _is_call
from .cute_mma_dag import _storage
from .cute_mma_dag import async_tile_reason
from .cute_mma_dag import frag_word_blocks
from .cute_mma_dag_emitter_base import _RUNTIME
from .cute_mma_dag_emitter_base import _TC
from .cute_mma_dag_emitter_base import _TMA_SWIZZLE
from .cute_mma_dag_emitter_base import Coords
from .cute_mma_dag_emitter_base import MmaDagEmitterBase
from .cute_mma_dag_emitter_base import _Evaluator
from .cute_mma_dag_emitter_base import _float_literal
from .cute_mma_dag_emitter_base import _frag_dtype
from .cute_mma_dag_emitter_base import _Hoist
from .cute_mma_dag_emitter_base import _Lane
from .cute_mma_dag_emitter_base import _Layout
from .cute_mma_dag_emitter_base import _Packet
from .cute_mma_dag_emitter_base import _reduce_combine
from .cute_mma_dag_emitter_base import _ring_alias
from .cute_mma_dag_emitter_base import _RingAlias
from .cute_mma_dag_emitter_base import _shape
from .cute_mma_dag_emitter_base import _staging_swizzle
from .cute_mma_dag_emitter_base import _Tile
from .cute_mma_dag_emitter_base import _TmaMap
from .cute_mma_dag_emitter_base import _transposed_swizzle
from .cute_mma_dag_emitter_base import tma_wrapper_plan
from .cute_warp_mma_gemm import _swizzled_chunk
from .state import cute_codegen_state

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from torch.fx.node import Node

    from ..device_function import DeviceFunction
    from ..generate_ast import GenerateAST
    from .cute_mma_dag import MmaDagProgram
    from .cute_mma_dag_tcgen05 import TmaSpec

# The runtime helpers' module imports (``_RUNTIME``; ``_TC`` is the tcgen05
# runtime the warp mode's TMA ring borrows).
_PREAMBLE = f"import helion._compiler.cute._mma_dag_runtime as {_RUNTIME}\n"
_TC_PREAMBLE = f"import helion._compiler.cute._mma_dag_tcgen05_runtime as {_TC}\n"
# The launcher's wrapper plan that names the program's tensor arguments so
# a launch can refuse arguments that alias (see ``mma_dag_alias_plan``).
ALIAS_KIND = "helion_mma_dag_alias"


def emit_mma_dag_module_statements(cg: GenerateAST) -> None:
    """Emit the once-per-module import of the runtime helpers."""
    cg.ensure_module_statements(_PREAMBLE)


def codegen_mma_dag(cg: GenerateAST) -> bool:
    """Emit the warp-MMA program for the planned root (False: not this root)."""
    df = cg.device_function
    plan = df.cute_state.root_plans.mma_dag_plan
    root = cg.current_root_graph_info
    if plan is None or root is None or root.graph_id != plan.program.root_graph_id:
        return False
    # This render's exact fp32 widenings (``MmaDagEmitterBase.retyped``): fresh per
    # kernel, the names recur from one render of a bound kernel to the next.
    df.cute_state.root_plans.mma_dag_exact_f32 = {}
    if plan.attention is not None:
        from .paged_attention_codegen import codegen_paged_attention

        codegen_paged_attention(cg, plan.attention)
        return True
    if plan.compute == MMA_DAG_TCGEN05:
        from .cute_mma_dag_tcgen05 import codegen_mma_dag_tcgen05

        codegen_mma_dag_tcgen05(cg, plan)
        return True
    emit_mma_dag_module_statements(cg)
    wrapper_plans = cute_codegen_state(cg).wrapper_plans
    emitter = WarpMmaDagEmitter(cg, plan)
    source = emitter.render()
    if emitter.tma_maps:
        cg.ensure_module_statements(_TC_PREAMBLE)
    for statement in ast.parse(textwrap.dedent(source)).body:
        cg.add_statement(statement)
    if emitter.tma_maps:
        # The host builds one tensor map per entry over the tensor views and
        # passes them as grid constants after the ordinary arguments.
        wrapper_plans.append(
            tma_wrapper_plan(
                df,
                emitter.tma_maps,
                lambda access: df.tensor_arg(access.fake, prefer_name=access.name).name,
            )
        )
    wrapper_plans.append(mma_dag_alias_plan(df, plan.program))
    df.cute_state.lanes.owned_root_block_dims = (plan.launch_threads, 1, 1)
    return True


def mma_dag_alias_plan(df: DeviceFunction, program: MmaDagProgram) -> dict[str, object]:
    """The launcher's alias guard: the program's tensor arguments (kernel
    argument names, resolved to launch positions by the host codegen) and
    which of them it stores to.

    The planner admits a program whose loads and stores touch distinct
    storages, and the emitter hoists side loads and stages tiles ahead of
    the stores on that proof.  It is a proof over the fake tensors the
    kernel is compiled against, which have one storage per argument: a
    caller passing one tensor (or two views of one buffer) under two names
    is only visible at launch, where the guard refuses it.
    """
    names: list[str] = []
    stored: list[bool] = []
    for access in program.accesses:
        name = df.tensor_arg(access.fake, prefer_name=access.name).name
        is_store = _is_call(access.node, memory_ops.store)
        if name in names:
            index = names.index(name)
            stored[index] = stored[index] or is_store
            continue
        names.append(name)
        stored.append(is_store)
    return {"kind": ALIAS_KIND, "tensor_names": names, "stored": stored}


class WarpMmaDagEmitter(MmaDagEmitterBase):
    """The warp-mode renderer: mma.sync fragments over ldmatrix tiles, the
    cp.async / TMA operand rings and the per-iteration schedule described in
    the module docstring, on the shared text builders, evaluator and
    addressing of :class:`MmaDagEmitterBase`."""

    def __init__(self, cg: GenerateAST, plan: MmaDagPlan) -> None:
        self.cg = cg
        self.df: DeviceFunction = cg.device_function
        self.env = CompileEnvironment.current()
        self.plan = plan
        self.program = plan.program
        self.lines: list[str] = []
        self.indent = ""
        self.accesses: dict[Node, Access] = {
            access.node: access for access in self.program.accesses
        }
        self.base_vars: dict[Node, str] = {}
        self.iter_vars: dict[Node, str] = {}
        # L2 cache policy ("evict_last" / "evict_first") -> the register
        # holding its policy word (see ``render``).
        self.policy_vars: dict[str, str] = {}
        # Uniform tensor loads and composed integer loop bounds, keeping
        # Int64 width for loads feeding a bound (defined with the root bases).
        self.scalar_vars: dict[Node, str] = {}
        # Global stores deferred past the next fragment loads, each under
        # its row guard (None: unguarded).
        self.deferred: list[tuple[str | None, str]] = []
        self._defined_tiles: set[str] = set()
        # Lane terms of the shuffle route (see ``emit_shuffle_words``).
        self.quad_terms: dict[str, tuple[str, str]] = {}
        # Packet terms by chunks per row (see ``own_packets``).
        self.packet_names: dict[int, tuple[str, str, str]] = {}
        # The loops being emitted, outermost first (see ``hoist_level``), how
        # many loops enclosed each name when it was bound, and the storages
        # the program stores to (never side-loaded ahead of a loop: a store
        # through a view or another name of the tensor is a store to it).
        self.hoists: list[_Hoist] = []
        # Loop-variant per-thread side loads of an innermost loop that are
        # loaded one iteration ahead: loop index -> (node, coords) -> the
        # register holding the current iteration's value
        # (``prefetch_level``); ``refilled`` tracks the sites reloaded for
        # the next iteration so far in the iteration being emitted.
        self.prefetch_sites: dict[int, dict[tuple[Node, Coords], str]] = {}
        self.refilled: set[tuple[Node, Coords]] = set()
        # ``MMA_DAG_AUX_RING``: per loop, the side-load sites staged through
        # the per-stage shared slots (site -> slot index), the issue anchors
        # (lines inserted once the body is rendered and every site is known:
        # position, indent, loop values, stage) and whether allocation closed
        # (a site met after the in-loop refill could not be refilled).
        self.aux_sites: dict[int, dict[tuple[Node, Coords], int]] = {}
        self.aux_anchors: dict[int, list[tuple[_Hoist, dict[int, str], str]]] = {}
        self.aux_closed: set[int] = set()
        # The ``for`` lines of the innermost loops being emitted (the pre-loop
        # wait of a side tile read ahead of the iteration barrier goes right
        # ahead of the ``for``, once the body is rendered): kept current by
        # ``insert_line`` like the hoist anchors, since the body's hoisted
        # lines land ahead of the loop's prologue, above the ``for``.
        self.loop_entries: list[_Hoist] = []
        # The current iteration of an aux-ring loop waited for its stage /
        # passed the barrier that publishes the stage's vectors.
        self.aux_waited = False
        self.aux_synced = False
        # Per aux-ring loop being iterated, the side tile slot of the current
        # iteration (``AuxRing.shared``; bound at the iteration top), and the
        # loops whose side tile is read ahead of the iteration barrier (the
        # first iteration then needs the pre-loop wait and barrier).
        self.aux_shared_stage: dict[int, str] = {}
        self.aux_shared_early: set[int] = set()
        # The TMA ring (``MMA_DAG_LOADS_TMA`` in the warp mode): (tile name,
        # load node) -> (its tensor map, its loop) -- a tile shared by two
        # loops' operands is fetched by each loop's own access --, the maps,
        # the bytes one stage of a loop's TMA tiles expects, and the loops'
        # wait parity.
        self.tma_tiles: dict[tuple[str, Node], tuple[_TmaMap, int]] = {}
        self.tma_maps: list[_TmaMap] = []
        self.tma_tx: dict[int, int] = {}
        self.tma_parity: dict[int, str] = {}
        # The snapshots one thread stores by bulk tensor copies
        # (``MMA_DAG_STORE_TMA``): store node -> its tensor map and slabs
        # (``plan_tma_stores``; the maps ride ``tma_maps``).
        self.store_tma: dict[Node, TmaSpec] = {}
        self.name_depth: dict[str, int] = {}
        self.stored_storages = {
            _storage(store.access)
            for store in (*self.program.stores, *self.program.vec_stores)
        }
        # Element pairs of the passes, derived values, exchanges and stores
        # render their fp32 mul / add / sub as packed f32x2 ops.
        self.pairs = True
        self.packable_memo: dict[Node, bool] = {}
        # Lines redirected ahead of a loop (see ``redirected``), the nest of
        # blocks being emitted (ids, outermost first) and the lane bases
        # bound so far by (iteration base, lane terms, block path).
        self.redirect: _Hoist | None = None
        self.scope_path: list[int] = []
        self.scope_count = 0
        self.lane_bases: dict[tuple[object, ...], str] = {}
        self.fold_addresses = True
        # Loops that issue their nested loop's ring prologue for the next
        # iteration during this one's epilogue (see ``early_prologues``).
        self.early_prologues = plan.early_prologues()
        self.early_issued: set[int] = set()
        self.evaluator = _Evaluator(self)
        program = self.program
        self.sync = "cute.arch.barrier()" if plan.warps > 1 else "cute.arch.sync_warp()"
        # One warp grid per distinct fragment shape (the plan's solver picks
        # M-first or N-first per shape).
        self.layouts: dict[tuple[int, int], _Layout] = {}
        self.slot_layouts: list[_Layout] = []
        for slot in range(len(program.slots)):
            shape = plan.slot_shape(slot)
            if shape not in self.layouts:
                self.layouts[shape] = _Layout(
                    len(self.layouts), *shape, plan.warps, plan.slot_layout(slot)
                )
            self.slot_layouts.append(self.layouts[shape])
        # Shared memory: the plan's layout (ring tiles, synchronous /
        # epilogue tiles, exchange tiles shared by key, the derived values'
        # ring tiles, then the carry staging buffers); ``q`` feeds two dots
        # of fwd_o through one tile, ``dA`` is read straight and transposed
        # in dqk; an exchange tile of the carry at loop entry is written
        # before the iteration barrier and so double-buffered by parity.
        operand_specs, derived_specs, staging_specs, self.smem_bytes, scratch = (
            plan.smem_layout()
        )
        self.smem_bytes += plan.aux_bytes
        self.scratch: _Tile | None = (
            _Tile(scratch.name, scratch.offset, scratch.rows, scratch.cols, 4, 1)
            if scratch is not None
            else None
        )
        # Vector slot -> the warp grid (and lane coordinates) its family
        # follows (None: a scalar).
        self.vec_layouts: list[_Layout | None] = [
            self.slot_layouts[program.vec_frag_slots[slot][0]]
            if program.vec_frag_slots[slot]
            else None
            for slot in range(len(program.vec_slots))
        ]
        self.shared_tiles: dict[tuple[object, ...], _Tile] = {}
        # Operands whose raw source tile overlays the loops' frames, and the
        # root exchange tiles that do.
        self.overlaid_keys = {
            key for key, spec in operand_specs.items() if spec.raw_overlays
        }
        self.overlaid_tiles = {
            key for key, spec in operand_specs.items() if spec.overlays
        }
        for key, spec in operand_specs.items():
            tile = _Tile(
                spec.name,
                spec.offset,
                spec.rows,
                spec.cols,
                spec.elem_bytes,
                spec.slots,
            )
            if spec.raw_elem_bytes:
                raw_rows, raw_cols = (
                    (spec.cols, spec.rows)
                    if spec.raw_swapped
                    else (spec.rows, spec.cols)
                )
                tile.raw = _Tile(
                    f"{spec.name}r",
                    spec.raw_offset,
                    raw_rows,
                    raw_cols,
                    spec.raw_elem_bytes,
                    spec.raw_slots,
                )
                tile.raw.is_raw = True
            self.shared_tiles[key] = tile
        self.tiles: dict[tuple[Node, str], _Tile] = {}
        self.tile_buffers: dict[tuple[Node, str], str] = {}
        for dot in program.dots:
            for operand in (dot.a, dot.b):
                route = plan.route(dot, operand)
                if route in REGISTER_ROUTES:
                    continue
                if route == ROUTE_EXCHANGE:
                    buffer = (
                        self.parity_var(dot.loop)
                        if plan.exchange_copies(dot, operand) == 2
                        else "0"
                    )
                elif program.is_ring(dot, operand):
                    buffer = self.stage_var(dot.loop)
                else:
                    # (A pass chain's tile among them: one copy, written
                    # behind the iteration barrier like an exchange tile.)
                    buffer = "0"
                self.tile_buffers[(dot.node, operand.role)] = buffer
                self.tiles[(dot.node, operand.role)] = self.shared_tiles[
                    operand.chain_key
                ]
        # Emission state: the loop values of the enclosing loops, the
        # regions whose staged tiles are issued, the tiles whose passes ran,
        # whether a cp.async group may still be in flight, and the fragments
        # defined so far (an exchange tile is written once its producer is).
        self.values: dict[int, str] = {}
        # The raw source tiles' ring slot of each loop's current iteration.
        self.raw_stages: dict[int, str] = {}
        self.issued: set[int] = set()
        self.settled: set[tuple[object, ...]] = set()
        self.async_pending = False
        # Younger cp.async groups committed since the staged group (None:
        # none pending, or unknown: a commit inside a loop body).
        self.staged_group_age: int | None = None
        # The root loop whose ring prologue went out ahead of the root
        # dot's wait (``root_early_candidate``).
        self.root_early_loop: int | None = None
        self.defined: set[Node] = set()
        self.derived_tiles: dict[Node, _Tile] = {
            node: _Tile(
                spec.name,
                spec.offset,
                spec.rows,
                spec.cols,
                spec.elem_bytes,
                spec.slots,
            )
            for node, spec in derived_specs.items()
        }
        self.staging: dict[Node, _Tile] = {
            node: _Tile(
                spec.name, spec.offset, spec.rows, spec.cols, spec.elem_bytes, 2
            )
            for node, spec in staging_specs.items()
        }
        # The fp32 scalars and vectors the aux rings copy once per CTA into
        # their side tiles and the lanes read by coordinate
        # (``AuxRing.shared``), by node.
        self.shared_aux_nodes: set[Node] = {
            node for ring in plan.aux_rings.values() for node in ring.shared
        }
        # Vector loads served from a ring slot (see ``_ring_aliases``) and
        # the loops whose current slot is landed and published (between the
        # iteration barrier behind the cp.async wait and the iteration's end).
        self.ring_ready: set[int] = set()
        self.ring_aliases = self._ring_aliases()

    def _ring_aliases(self) -> dict[Node, _RingAlias]:
        """The loop's vector loads that read a row or a column of a tile
        the same loop stages through its cp.async ring, as the ring slot
        they alias: ``gc[idx, i_t, C - 1, tile_d]`` beside the staged
        ``gc[idx, i_t, :, tile_d]`` (the chunk's last decay row, read by
        the carry rescale and the key's decay pass).  The slot holds the
        same bytes after the iteration's wait and barrier, so the lanes
        read them from shared memory (``_load``: ``lds`` at the slot's
        swizzled address) instead of re-fetching the row from global
        memory on the chain -- the same values.

        Fail closed: the staged tile must be a plain copy of its source (a
        derived value's tile or a direct-route operand tile, not a tile a
        pass rewrites in place, not a transposed fp32 tile), 4-byte
        elements, the two accesses the same tensor at the same strides and
        uniform subscripts, the vector's one tile dim the tile's own axis
        and its other tile dim a constant inside a ``:`` rdim, no windows,
        offsets or ragged axes.
        """
        program, plan = self.program, self.plan
        stored = {store.access.node for store in program.stores}
        stored.update(store.access.node for store in program.vec_stores)
        tiles: list[tuple[int, _Tile, Access, int]] = []
        for loop in program.loops:
            if not loop.innermost:
                continue
            for dot, operand in plan.ring_operands(loop.index):
                tile = self.tiles[(dot.node, operand.role)]
                if (
                    plan.route(dot, operand) != ROUTE_DIRECT
                    or operand.transposed_tile
                    or tile.raw is not None
                    or operand.load is None
                ):
                    continue
                tiles.append((loop.index, tile, operand.load, operand.rows_pos))
            for _derived, access in plan.ring_derived(loop.index):
                tiles.append((loop.index, self.derived_tiles[access.node], access, 0))
        result: dict[Node, _RingAlias] = {}
        for access in program.accesses:
            if access.node in stored or access.node in result:
                continue
            if access.node in self.shared_aux_nodes:
                # Staged in its loop's aux ring's side tile (``cute_mma_dag_
                # aux='ring'``): read from there; the two routes never share
                # a load.
                continue
            if len(access.tile_dims) != 1 or access.dtype.itemsize != 4:
                continue
            if _storage(access) in self.stored_storages:
                # A tensor the program stores to: the slot's copy could be
                # older than the global bytes a load would see.
                continue
            for loop_index, tile, load, rows_pos in tiles:
                alias = _ring_alias(plan, access, loop_index, tile, load, rows_pos)
                if alias is not None:
                    result[access.node] = alias
                    break
        return result

    def reg(self, slot: int, am: int, an: int, j: int) -> str:
        return f"md_c{slot}_{am}_{an}_{j}"

    @staticmethod
    def mma_kind(dtype: torch.dtype) -> str:
        """The ``mma.sync`` operand kind: ``bf16`` / ``f16`` (m16n8k16) or
        ``tf32`` (m16n8k8, fp32 registers truncated to tf32)."""
        if dtype == torch.float32:
            return "tf32"
        return "bf16" if dtype == torch.bfloat16 else "f16"

    @staticmethod
    def k_step(dtype: torch.dtype) -> int:
        """Elements per MMA K step (16 for 16-bit operands, 8 for tf32): 32
        bytes of each operand row."""
        return 32 // dtype.itemsize

    def parity_var(self, loop: int) -> str:
        return "md_parity" if loop == 0 else f"md_parity{loop}"

    def row_exists(self, access: Access, tile_position: int, coord: str) -> str | None:
        """The Boolean saying tile-local ``coord`` along fragment dim
        ``tile_position`` of ``access`` is a row that exists: inside a
        ragged root axis' extent and inside the dim's window (None: every
        row does)."""
        axis = access.tile_dims[tile_position]
        terms = []
        if self.plan.ragged(axis):
            terms.append(self.in_bounds(axis, coord))
        window = self.window_guard(access, access.tile_positions[tile_position], coord)
        if window is not None:
            terms.append(window)
        return " and ".join(terms) if terms else None

    def bounds_pred(self, access: Access, coords: Coords) -> str | None:
        """The Int32 predicate of a per-lane load of ``access`` at ``coords``
        bound at the current site (None: unpredicated)."""
        bounds = self.bounds(access, coords)
        if bounds is None:
            return None
        return self.bind("md_ok", self.pred_int(bounds))

    def has_aux_shared(self, loop: int) -> bool:
        """Whether ``loop``'s aux ring has a side tile."""
        ring = self.plan.aux_rings.get(loop)
        return ring is not None and bool(ring.shared)

    def aux_shared_slot(self, loop: Loop, value: str) -> str:
        """The side tile slot of ``loop``'s iteration ``value``: the
        iteration's ordinal modulo the ``stages + 1`` slots."""
        ring = self.plan.aux_rings[loop.index]
        begin = self.loop_bounds(loop)[0]
        offset = f"({value})" if begin == "0" else f"(({value}) - {begin})"
        return f"{offset} % {ring.shared_slots}"

    def emit_aux_shared(
        self, loop: Loop, values: dict[int, str], slot: str | None = None
    ) -> None:
        """``cp.async`` the side tile of ``loop``'s iteration at ``values``
        into its slot (``slot``, or the iteration's own): the scalar by
        thread 0, element ``t`` of each vector by thread ``t`` round robin
        over the CTA, 4 bytes each (no commit: the tile rides the group
        being issued)."""
        ring = self.plan.aux_rings[loop.index]
        threads = ring.threads
        if slot is None:
            slot = self.aux_shared_slot(loop, values[loop.index])
        slot = self.aux_shared_base(loop.index, slot)
        for node, (elements, offset) in ring.shared.items():
            access = self.accesses[node]
            source = self.iteration_base_expr(access, values)
            target = f"{slot} + {offset}" if offset else slot
            if elements == 1:
                with self.nest("if md_tidx == 0:"):
                    self.emit(f"{_RUNTIME}.cp_async_4({target}, {source})")
                continue
            stride = access.tile_dim_strides[0] * access.dtype.itemsize
            for first in range(0, elements, threads):
                idx = f"(md_tidx + {first})" if first else "md_tidx"
                with (
                    self.nest(f"if {idx} < {elements}:")
                    if elements - first < threads
                    else contextlib.nullcontext()
                ):
                    self.emit(
                        f"{_RUNTIME}.cp_async_4({target} + {idx} * 4,"
                        f" {source} + cutlass.Int64({idx}) * {stride})"
                    )

    def emit_aux_shared_ahead(
        self, loop: Loop, outer: dict[int, str], ahead: int
    ) -> None:
        """The side tile of iteration ``begin + ahead`` of ``loop`` (at the
        enclosing loops' values ``outer``; slot ``ahead`` of the fresh
        ring), guarded against the trip count where it is not static."""
        if not self.has_aux_shared(loop.index):
            return
        begin, end, trip_static = self.loop_bounds(loop)
        value = f"({begin} + {ahead})" if ahead else begin
        values = {**outer, loop.index: value}
        slot = str(ahead % self.plan.aux_rings[loop.index].shared_slots)
        if trip_static is not None:
            if ahead < trip_static:
                self.emit_aux_shared(loop, values, slot)
            return
        with self.nest(f"if {value} < {end}:"):
            self.emit_aux_shared(loop, values, slot)

    def aux_anchor(self, loop: int, values: dict[int, str], stage: str) -> None:
        """Record the current position as the issue point of the aux copies of
        the iteration at ``values`` into ring ``stage`` (filled in once the
        loop body is rendered)."""
        if loop not in self.plan.aux_rings:
            return
        anchor = _Hoist(len(self.lines), self.indent, len(self.values), loop)
        self.aux_anchors.setdefault(loop, []).append((anchor, dict(values), stage))

    def aux_issue_lines(
        self, loop: int, values: dict[int, str], stage: str
    ) -> list[str]:
        """``cp.async`` of each aux site's word (a 16-bit element copies the
        aligned word holding it) and of each run of the iteration at
        ``values`` into the thread's slots at ``stage`` (the side tile of
        an iteration rides one group ahead: ``emit_aux_shared``)."""
        lines: list[str] = []
        for (node, coords), slot in self.aux_sites.get(loop, {}).items():
            access = self.accesses[node]
            source = self.plain_address(
                access, self.iteration_base_expr(access, values), coords
            )
            if access.dtype.itemsize == 2:
                word = self.df.new_var("md_aw")
                lines.append(f"{word} = {source}")
                source = f"{word} - (cutlass.Int32({word}) & 3)"
            lines.append(
                f"{_RUNTIME}.cp_async_4({self.aux_slot_address(loop, slot, stage)}, {source})"
            )
        ring = self.plan.aux_rings.get(loop)
        if ring is None:
            return lines
        threads = ring.threads
        for node, (_elems, _run_bytes, offset) in ring.runs.items():
            # The vector's block extent, a 16-byte chunk per thread.
            access = self.accesses[node]
            units = self.vector_units(loop, node)
            source = self.df.new_var("md_ar")
            lines.append(f"{source} = {self.iteration_base_expr(access, values)}")
            target = f"{self.aux_stage_base(loop, stage)} + {ring.words * threads * 4 + offset}"
            for first in range(0, units, threads):
                chunk = f"(md_tidx + {first})" if first else "md_tidx"
                copy = (
                    f"{_RUNTIME}.cp_async_16({target} + {chunk} * 16,"
                    f" {source} + cutlass.Int64({chunk}) * 16)"
                )
                if units - first < threads:
                    lines.extend((f"if {chunk} < {units}:", f"    {copy}"))
                else:
                    lines.append(copy)
        return lines

    def vector_units(self, loop: int, node: Node) -> int:
        """16-byte chunks of the per-stage vector of run-form access ``node``
        (the next run's offset, or the vectors' end, less its own)."""
        ring = self.plan.aux_rings[loop]
        _elems, _run_bytes, offset = ring.runs[node]
        end = min(
            [o for _e, _r, o in ring.runs.values() if o > offset] + [ring.vector_bytes]
        )
        return (end - offset) // 16

    def fill_aux_anchors(self, loop: int) -> None:
        """Insert the aux copies at every recorded anchor of ``loop`` (the
        anchors stay registered meanwhile: each insertion moves the later
        ones)."""
        anchors = self.aux_anchors.get(loop, [])
        for anchor, values, stage in list(anchors):
            for line in self.aux_issue_lines(loop, values, stage):
                self.insert_line(anchor, line)
        self.aux_anchors.pop(loop, None)

    def emit_prefetch_refills(self, loop: Loop, var: str, end: str) -> None:
        """Reload the prefetch sites of ``loop`` not yet refilled in this
        iteration with the next iteration's values (guarded by the trip
        count)."""
        pending = [
            (key, register)
            for key, register in self.prefetch_sites.get(loop.index, {}).items()
            if key not in self.refilled
        ]
        if not pending:
            return
        values = {**self.values, loop.index: f"({var} + 1)"}
        with self.nest(f"if {var} + 1 < {end}:"):
            for (node, coords), register in pending:
                access = self.accesses[node]
                base = self.iteration_base_expr(access, values)
                address = self.plain_address(access, base, coords)
                self.emit(
                    f"{register} = {self.scalar_load(address, access.dtype, readonly=self.readonly(access))}"
                )
                self.refilled.add((node, coords))

    # -- fragment coordinates -----------------------------------------------------

    def frag_coords(
        self, node: Node, layout: _Layout, am: int, an: int, j: int, swap: bool = False
    ) -> Coords:
        """Positional coordinates of ``node`` for the register ``(am, an, j)``
        of a fragment with ``layout``; ``swap`` puts the fragment's row on
        the node's second tile dim (a transposed view of the fragment)."""
        shape = _shape(node)
        row, col = layout.row(am, j // 2), layout.col(an, j & 1)
        positions = [
            i
            for i, size in enumerate(shape)
            if not (isinstance(size, int) and size == 1)
        ]
        positions = [i for i in positions if not self._is_lead_dim(node, i)]
        if len(positions) != 2:
            raise _decline(f"{node.name} is not a two-dim fragment value")
        coords = ["0"] * len(shape)
        first, second = (col, row) if swap else (row, col)
        coords[positions[0]] = first
        coords[positions[1]] = second
        return tuple(coords)

    # -- shared tiles ---------------------------------------------------------------

    def own_packets(self, tile: _Tile, body: Callable[[_Packet], None]) -> None:
        """Call ``body(packet)`` for every packet this thread owns of the
        tile (round-robin over the CTA, as the cp.async issue).

        When the CTA's threads cover whole rows, round ``rep`` adds a
        constant ``rep * threads // cpr`` rows to the thread's own row and
        keeps its chunk: the addresses become one lane term plus an
        immediate (and the swizzle one term) instead of a register per
        round, which ptxas otherwise keeps live across the loop."""
        packets, cpr = tile.packets, tile.cpr
        threads = self.plan.threads
        whole_rows = threads % cpr == 0
        rows_per_round = threads // cpr if whole_rows else 0
        for rep in range(-(-packets // threads)):
            idx = f"({rep * threads} + md_tidx)" if rep else "md_tidx"
            guard = packets < (rep + 1) * threads
            nest = self.nest(f"if {idx} < {packets}:") if guard else None
            if nest is not None:
                nest.__enter__()
            self.evaluator.push()
            if whole_rows:
                pr, pch, pz = self.packet_terms(cpr)
                row_off = rep * rows_per_round
                prow = f"({row_off} + {pr})" if row_off else pr
                if rows_per_round % 8 == 0:
                    srow, swz = pr, pz
                else:
                    srow, swz = prow, _swizzled_chunk(prow, pch, cpr)
                packet = _Packet(idx, prow, pr, row_off, srow, pch, swz)
            else:
                prow = self.bind("md_prow", f"{idx} // {cpr}")
                pch = self.bind("md_pch", f"{idx} % {cpr}")
                swz = _swizzled_chunk(prow, pch, cpr)
                packet = _Packet(idx, prow, prow, 0, prow, pch, swz)
            self.evaluator.packet = packet.pch
            body(packet)
            self.evaluator.packet = None
            self.evaluator.pop()
            if nest is not None:
                nest.__exit__(None, None, None)

    def packet_terms(self, cpr: int) -> tuple[str, str, str]:
        """The thread's own packet row, chunk and swizzled chunk of a tile
        with ``cpr`` chunks per row (bound once in the preamble)."""
        return self.packet_names[cpr]

    def plan_tma(self) -> None:
        """Pick the ring tiles one tensor map fetches (``MMA_DAG_LOADS_TMA``):
        an innermost loop's operand or derived tile (not a converted
        operand's raw tile) of at most 256 rows and 1, 2, 4 or 8 chunks per
        row (the box is one swizzle atom wide, so the emitter's XOR swizzle
        is TMA's), a source ``async_tile_reason`` admits, no windowed dim
        (a window's rows at and past its runtime limit load as zero: the
        packets zero-fill them row by row, a box lands whole;
        ``tma_tile_source``), and ring slots at the swizzle pattern's
        period.  Other tiles keep their packets."""
        plan = self.plan
        if plan.compute != MMA_DAG_WARP or plan.loads != MMA_DAG_LOADS_TMA:
            return
        from .cute_mma_dag_tcgen05 import _tma_dims

        shared: dict[tuple[object, ...], _TmaMap] = {}
        for loop in self.program.loops:
            if not loop.innermost:
                continue
            for tile, load, rows_pos in self.ring_items(loop.index):
                if load.node in plan.load_hints:
                    # A hinted tile keeps its cp.async packets: the L2 policy
                    # rides them (``MMA_DAG_L2_HINTS``), the bulk tensor copy
                    # carries none.
                    continue
                if (
                    tile.is_raw
                    or len(load.tile_dims) != 2
                    or load.windows
                    or tile.rows > 256
                    or tile.cpr not in (1, 2, 4, 8)
                    or tile.slots != plan.stages
                    or async_tile_reason(self.env, load, 1 - rows_pos) is not None
                ):
                    continue
                period = max(128, 8 * tile.row_bytes if tile.cpr > 1 else 16)
                if tile.offset % period or tile.bytes % period:
                    continue
                dims = _tma_dims(load)
                tile_dims = [load.subscript_index(p) for p in load.tile_positions]
                if dims[0] != tile_dims[1 - rows_pos]:
                    continue
                rows_dim = tile_dims[rows_pos]
                box = tuple(
                    tile.cols if position == 0 else tile.rows if dim == rows_dim else 1
                    for position, dim in enumerate(dims)
                )
                if len(dims) > 5 or not all(1 <= b <= 256 for b in box):
                    continue
                swizzle = _TMA_SWIZZLE[tile.row_bytes]
                key = (load.fake, dims, box, swizzle)
                tmap = shared.get(key)
                if tmap is None:
                    tmap = shared[key] = _TmaMap(
                        len(self.tma_maps),
                        f"md_tma{len(self.tma_maps)}",
                        load,
                        dims,
                        box,
                        swizzle,
                    )
                    self.tma_maps.append(tmap)
                self.tma_tiles[(tile.name, load.node)] = (tmap, loop.index)
                self.tma_tx[loop.index] = self.tma_tx.get(loop.index, 0) + tile.bytes

    def plan_tma_stores(self) -> None:
        """Pick the staged snapshots one thread stores by bulk tensor copies
        (``MMA_DAG_STORE_TMA``): a 16-bit snapshot (the fp32 staging swizzle
        is no TMA pattern) of at most 256 rows, rows of one swizzle atom
        (16, 32, 64 or 128 bytes) or of whole 128-byte slabs
        (``emit_stage_in`` lays those side by side), at a staging offset
        the swizzle pattern's period divides, with no window (a box lands
        whole rows; the packets drop the rows at and past the limit) and
        no ragged row axis (the packets drop the rows the CTA does not own;
        a box clips at the tensor's bounds only), of a storage no other
        access touches (a bulk store completes at the kernel's end: the
        in-loop waits retire its staging reads, not its global writes),
        within the tensor map rank and count limits.  Other snapshots keep
        their packets."""
        plan = self.plan
        if not plan.tma_stores:
            return
        from .cute_mma_dag_tcgen05 import ATOM_MAX_BYTES
        from .cute_mma_dag_tcgen05 import TMA_MAX_MAPS
        from .cute_mma_dag_tcgen05 import TMA_MAX_RANK
        from .cute_mma_dag_tcgen05 import TmaSpec
        from .cute_mma_dag_tcgen05 import _tma_dims

        for store in plan.staged_stores:
            if not self.program.loops[store.loop].innermost:
                # (An outer loop's snapshot is a direct store: ``emit_outer_
                # iteration``.)
                continue
            access = store.access
            tile = self.staging[access.node]
            es = access.dtype.itemsize
            atom = min(tile.row_bytes, ATOM_MAX_BYTES)
            # (``plan_tma``: the swizzle pattern's period, 128 B at least.)
            period = max(128, 8 * atom if tile.cpr > 1 else 16)
            storage = _storage(access)
            if (
                es != 2
                or access.windows
                or plan.ragged(access.tile_dims[0])
                or tile.rows > 256
                or (tile.cpr not in (1, 2, 4, 8) and tile.row_bytes % ATOM_MAX_BYTES)
                or tile.offset % period
                or tile.bytes % period
                or sum(1 for a in self.program.accesses if _storage(a) == storage) > 1
                or len(self.tma_maps) >= TMA_MAX_MAPS
            ):
                continue
            dims = _tma_dims(access)
            tile_dims = [access.subscript_index(p) for p in access.tile_positions]
            if len(dims) > TMA_MAX_RANK or dims[0] != tile_dims[1]:
                continue
            slab_elems = atom // es
            box = tuple(
                slab_elems if position == 0 else tile.rows if dim == tile_dims[0] else 1
                for position, dim in enumerate(dims)
            )
            spec = TmaSpec(
                index=len(self.tma_maps),
                param=f"md_tma{len(self.tma_maps)}",
                access=access,
                dims=dims,
                box=box,
                swizzle=_TMA_SWIZZLE[atom],
                slab_dim=0,
                slab_elems=slab_elems,
                slabs=tile.row_bytes // atom,
                slab_bytes=tile.rows * atom,
                offset=tile.offset,
            )
            self.tma_maps.append(spec)
            self.store_tma[access.node] = spec

    def tma_coords(
        self,
        tmap: _TmaMap,
        access: Access,
        values: dict[int, str],
        col_offset: int = 0,
    ) -> list[str]:
        """The block-uniform element coordinates of ``access``'s tile at the
        loop values ``values``, per tensor dim in TMA order (``tmap`` may be
        shared with another access of the tensor: same dims and box);
        ``col_offset`` elements along the contiguous dim (a store's slab)."""
        coords = []
        for position, dim in enumerate(tmap.dims):
            subscript = access.subscripts[dim]
            if subscript.kind == "scalar":
                assert subscript.expr is not None
                value = self.uniform(subscript.expr, dict(values))
            else:
                axis = subscript.axis
                assert axis is not None
                if axis.kind == AXIS_RDIM:
                    value = "0"
                elif axis.kind == AXIS_LOOP:
                    offset = self.axis_offset(axis, values[axis.loop])
                    value = f"({offset}) * {self.plan.size(axis)}"
                else:
                    value = self.axis_base(axis)
                if subscript.offset is not None:
                    value = f"{value} + {self.uniform(subscript.offset, {})}"
            if position == 0 and col_offset:
                value = f"{value} + {col_offset}"
            coords.append(f"cutlass.Int32({value})")
        return coords

    def tma_count(self, loop: int) -> str:
        """The running iteration count of TMA ring loop ``loop`` (across its
        re-entries: slot and barrier phase follow it)."""
        return f"md_cnt{loop}"

    def ring_items(self, loop: int) -> list[tuple[_Tile, Access, int]]:
        """The ring tiles of loop ``loop`` in issue order: the operands'
        source tiles (raw tiles of converted operands) and the derived
        values' tiles, as ``(tile, load, position of the tile's rows among
        the load's dims)``."""
        items: list[tuple[_Tile, Access, int]] = []
        for dot, operand in self.plan.ring_operands(loop):
            tile = self.tiles[(dot.node, operand.role)]
            assert operand.load is not None
            items.append((tile.raw or tile, operand.load, operand.rows_pos))
        for _derived, access in self.plan.ring_derived(loop):
            items.append((self.derived_tiles[access.node], access, 0))
        return items

    def emit_issue(
        self,
        items: list[tuple[_Tile, Access, int]],
        values: dict[int, str],
        stage: str,
        raw_stage: str = "0",
    ) -> None:
        """cp.async the tiles of the iteration at ``values`` into ring slot
        ``stage`` (raw source tiles into ``raw_stage``; no commit); the TMA
        ring's tiles go out as one bulk tensor copy each from thread 0, which
        first fences the slot's generic-proxy reads and pass writes and
        expects the stage's bytes on the slot's barrier."""
        tma = [item for item in items if (item[0].name, item[1].node) in self.tma_tiles]
        for tile, load, rows_pos in items:
            if (tile.name, load.node) in self.tma_tiles:
                continue
            self.emit_issue_tile(
                tile, load, rows_pos, values, raw_stage if tile.is_raw else stage
            )
        if not tma:
            return
        loop = self.tma_tiles[(tma[0][0].name, tma[0][1].node)][1]
        barrier = f"md_full{loop} + ({stage})"
        with self.nest("if md_tidx == 0:"):
            self.emit('cute.arch.fence_proxy(kind="async.shared", space="cta")')
            self.emit(
                f"{_TC}.prims.mbarrier_arrive_expect_tx({barrier}, {self.tma_tx[loop]})"
            )
            for tile, load, _rows_pos in tma:
                tmap, _loop = self.tma_tiles[(tile.name, load.node)]
                coords = ", ".join(self.tma_coords(tmap, load, values))
                self.emit(
                    f"{_TC}.prims.cp_async_bulk_tensor_shared_cta_global("
                    f"md_smem + ({tile.offset} + ({stage}) * {tile.bytes}), "
                    f"{tmap.param}.get_ptr(), [{coords}], {barrier})"
                )

    def emit_issue_operand(self, dot: Dot, operand: Operand) -> None:
        """cp.async a staged operand tile (single slot) at the current loop
        values."""
        load = operand.load
        assert load is not None
        tile = self.tiles[(dot.node, operand.role)]
        self.emit_issue_tile(tile.raw or tile, load, operand.rows_pos, self.values, "0")

    def policy_var(self, hint: str) -> str:
        """The register holding the L2 cache-policy word of ``hint``
        ("evict_last" / "evict_first"): bound once, in the prologue for the
        tiles ``render`` knows cp.async stages, at the first packet
        otherwise."""
        name = self.policy_vars.get(hint)
        if name is None:
            name = self.policy_vars[hint] = self.bind(
                f"md_pol_{hint}", f"{_RUNTIME}.l2_policy_{hint}()"
            )
        return name

    def emit_issue_tile(
        self,
        tile: _Tile,
        load: Access,
        rows_pos: int,
        values: dict[int, str],
        stage: str,
    ) -> None:
        row_stride = load.tile_dim_strides[rows_pos] * load.dtype.itemsize
        smem = tile.base(stage)
        gbase = self.iteration_base(load, values)
        row_bytes = tile.row_bytes
        # The rows of a ragged axis past the extent, and of a window past
        # its limit, are zero-filled (the packet reads nothing: ``src-size``
        # 0).

        # The packets carry the L2 cache policy of the load's
        # ``load_eviction_policies`` slot (``MmaDagPlan.load_hints``; the
        # policy word is created once in the prologue, ``policy_var``) as
        # ``cp.async.cg.shared.global.L2::cache_hint``.  A pure cache hint:
        # the bits are the plain packets'.  The CuTe DSL's embedded
        # assembler (cutlass-dsl 4.7.0: ptxas 13.3.27) has dropped the
        # uniform-register operand fields of that instruction whenever it
        # keeps a uniform term of the shared address in a uniform register:
        # the SASS reads ``[R42+UR0+0xc00], desc[UR1]`` -- registers no
        # instruction of the kernel writes -- where ptxas 13.4.59 encodes
        # the same PTX as ``[R42+UR10+0xc00], desc[UR11]`` with both
        # registers written just before, and the kernel dies with
        # CUDA_ERROR_ILLEGAL_INSTRUCTION (the kda WY transform's warp
        # configs did, in the autotuner's initial population).  Which
        # packets get that encoding is the assembler's register allocation,
        # not anything the render controls, so a kernel with hinted packets
        # declares ``MMA_DAG_HINTED_CP_ASYNC_CHECK`` and the launcher
        # verifies the SASS of every LDGSTS before the kernel may run
        # (``helion.runtime.cute.sass_check``: an LDGSTS naming a uniform
        # register the kernel never writes, and -- conservatively, until
        # the DSL's assembler is validated on it -- the split-address form
        # itself): a refused kernel never launches, and the autotuner drops
        # its config.
        hint = self.plan.load_hints.get(load.node)
        policy = f", {self.policy_var(hint)}" if hint is not None else ""
        suffix = "_l2" if hint is not None else ""
        if hint is not None:
            self.declare_sass_check()

        def body(pk: _Packet) -> None:
            gofs = f" + {pk.row_off * row_stride}" if pk.row_off else ""
            dst = f"{smem} + {pk.offset(row_bytes)} + {pk.swz} * 16"
            src = f"{gbase} + cutlass.Int64({pk.pr}) * {row_stride} + cutlass.Int64({pk.pch} * 16){gofs}"
            guard = self.row_exists(load, rows_pos, pk.prow)
            if guard is not None:
                src_bytes = self.bind(
                    "md_zf",
                    f"cutlass.Int32(16) if {guard} else cutlass.Int32(0)",
                )
                self.emit(
                    f"{_RUNTIME}.cp_async_16{suffix}_zfill({dst}, {src}, {src_bytes}{policy})"
                )
            else:
                self.emit(f"{_RUNTIME}.cp_async_16{suffix}({dst}, {src}{policy})")

        self.own_packets(tile, body)

    def declare_sass_check(self) -> None:
        """Record on the device function that the kernel's SASS must pass the
        hinted ``cp.async`` encoding check before it runs."""
        state = self.cg.device_function.cute_state
        if MMA_DAG_HINTED_CP_ASYNC_CHECK not in state.lanes.sass_checks:
            state.lanes.sass_checks = (
                *state.lanes.sass_checks,
                MMA_DAG_HINTED_CP_ASYNC_CHECK,
            )

    def emit_bound_tail_zero(self, index: int, stage: str) -> None:
        """A varying-bound loop's partial trip: the operand the pass masks
        (``MmaDagPlan.bound_masks``) is zeroed past the per-CTA bound by
        its pass; the dot's other operand tiles are zeroed there in shared
        memory, a packet at a time by the thread owning it (the tiles'
        rows past the bound hold the tensor's next rows, which the program
        never reads: an Inf or NaN there would otherwise reach the
        accumulator through ``0 * x``).  After this thread's copies landed
        (its own packets; a TMA tile landed whole) and after the passes --
        ahead of them the stores delay the passes' shared reads (chunk_scan
        s0: 1.78 -> 1.91 us) -- before the iteration barrier."""
        plan = self.plan
        loop = self.program.loops[index]
        if loop.tile_end is None:
            return
        end = self.end_var(loop)
        for dot, operand in plan.ring_pairs(
            index, (ROUTE_DIRECT, ROUTE_INPLACE, ROUTE_CONVERT)
        ):
            role = plan.bound_masks.get(dot.node)
            load = operand.load
            if role is None or role == operand.role or load is None:
                continue
            if dot.k not in load.tile_dims:
                continue
            tile = self.tiles[(dot.node, operand.role)]
            source = tile.raw or tile
            smem = source.base(
                self.raw_stages.get(index, "0") if source.is_raw else stage
            )
            base = self.axis_base(dot.k)
            k_rows = load.tile_dims.index(dot.k) == operand.rows_pos
            elems = 16 // load.dtype.itemsize
            row_bytes = source.row_bytes

            self.own_packets(
                source,
                functools.partial(
                    self.emit_tail_zero_packet,
                    smem,
                    row_bytes,
                    base,
                    k_rows,
                    elems,
                    end,
                ),
            )

    def emit_tail_zero_packet(
        self,
        smem: str,
        row_bytes: int,
        base: str,
        k_rows: bool,
        elems: int,
        end: str,
        pk: _Packet,
    ) -> None:
        """One packet of ``emit_bound_tail_zero``: zeroed when its first K
        element (the packet's row, or its chunk's first element) is at or
        past the bound."""
        k = f"({base} + {pk.prow})" if k_rows else f"({base} + {pk.pch} * {elems})"
        with self.nest(f"if {k} >= {end}:"):
            self.emit(
                f"{_RUNTIME}.sts_v4_b32({smem} + {pk.offset(row_bytes)} + {pk.swz} * 16, "
                "cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(0))"
            )

    def emit_inplace_pass(self, dot: Dot, operand: Operand, stage: str) -> None:
        """Rewrite the thread's own packets of the staged source tile through
        the operand's pointwise DAG (after this thread's cp.async wait; no
        barrier is needed before the CTA-wide one that precedes ldmatrix)."""
        load = operand.load
        assert load is not None
        tile = self.tiles[(dot.node, operand.role)]
        smem = tile.base(stage)
        unpack = "unpack_f16x2" if load.dtype == torch.float16 else "unpack_bf16x2"
        self.hoist_scalars(operand.value)
        row_bytes = tile.row_bytes

        def body(pk: _Packet) -> None:
            address = self.bind(
                "md_pa", f"{smem} + {pk.offset(row_bytes)} + {pk.swz} * 16"
            )
            words = [self.df.new_var("md_w") for _ in range(4)]
            self.emit(f"{', '.join(words)} = {_RUNTIME}.lds_v4_b32({address})")
            for q, word in enumerate(words):
                lo, hi = self.df.new_var("md_lo"), self.df.new_var("md_hi")
                self.emit(f"{lo}, {hi} = {_RUNTIME}.{unpack}({word})")
                self.set_tile_leaf(load, 2 * q, lo)
                self.set_tile_leaf(load, 2 * q + 1, hi)
            packed = []
            for q in range(4):
                lanes = tuple(
                    _Lane(
                        self.tile_coords(operand, pk.prow, f"({pk.pch} * 8 + {e})"),
                        e,
                        None,
                    )
                    for e in (2 * q, 2 * q + 1)
                )
                values = self.evaluator.unrounded_pair(operand.value, lanes)
                self.evaluator.element = None
                values = [
                    self.masked(dot, operand, lane.coords, value)
                    for lane, value in zip(lanes, values, strict=True)
                ]
                packed.append(self.pack(values[0], values[1], operand.dtype))
            self.emit(f"{_RUNTIME}.sts_v4_b32({address}, {', '.join(packed)})")

        self.own_packets(tile, body)

    def emit_byte_convert_pass(self, dot: Dot, operand: Operand, stage: str) -> None:
        """Expand a 16-byte source packet into two packed 16-bit packets."""
        load = operand.load
        assert load is not None and load.dtype in (torch.int8, torch.uint8)
        assert operand.dtype in (torch.float16, torch.bfloat16)
        assert not operand.transposed_tile
        tile = self.tiles[(dot.node, operand.role)]
        raw = tile.raw
        assert raw is not None
        smem = tile.base(stage)
        raw_smem = raw.base(self.raw_stages.get(dot.loop, "0"))
        self.hoist_scalars(operand.value)

        def body(pk: _Packet) -> None:
            words = [self.df.new_var("md_w") for _ in range(4)]
            self.emit(
                f"{', '.join(words)} = {_RUNTIME}.lds_v4_b32({raw_smem} + {pk.offset(raw.row_bytes)} + {pk.swz} * 16)"
            )
            self.set_byte_packet(load, words)
            packed = []
            for q in range(8):
                lanes = tuple(
                    _Lane(
                        self.tile_coords(operand, pk.prow, f"({pk.pch} * 16 + {e})"),
                        e,
                        None,
                    )
                    for e in (2 * q, 2 * q + 1)
                )
                values = self.evaluator.unrounded_pair(operand.value, lanes)
                self.evaluator.element = None
                values = [
                    self.masked(dot, operand, lane.coords, value)
                    for lane, value in zip(lanes, values, strict=True)
                ]
                packed.append(self.pack(values[0], values[1], operand.dtype))
            for half in range(2):
                dst = _swizzled_chunk(pk.srow, f"({pk.pch} * 2 + {half})", tile.cpr)
                self.emit(
                    f"{_RUNTIME}.sts_v4_b32({smem} + {pk.offset(tile.row_bytes)} + {dst} * 16, "
                    f"{', '.join(packed[4 * half : 4 * half + 4])})"
                )

        self.own_packets(raw, body)

    def emit_convert_pass(self, dot: Dot, operand: Operand, stage: str) -> None:
        """Rewrite the thread's own packets of the raw source tile through
        the operand's pointwise DAG into the operand tile of the other width:
        an fp32 source's four elements per packet -> one 8-byte pair of
        16-bit pairs; a 16-bit source's eight elements -> two 16-byte fp32
        packets (a tf32 operand)."""
        load = operand.load
        assert load is not None
        if load.dtype in (torch.int8, torch.uint8):
            self.emit_byte_convert_pass(dot, operand, stage)
            return
        tile = self.tiles[(dot.node, operand.role)]
        raw = tile.raw
        assert raw is not None
        smem = tile.base(stage)
        raw_smem = raw.base(self.raw_stages.get(dot.loop, "0"))
        self.hoist_scalars(operand.value)
        cpr, row_bytes = tile.cpr, tile.row_bytes
        if load.dtype != torch.float32:
            unpack = "unpack_f16x2" if load.dtype == torch.float16 else "unpack_bf16x2"

            def widen(pk: _Packet) -> None:
                prow, pch = pk.prow, pk.pch
                words = [self.df.new_var("md_w") for _ in range(4)]
                self.emit(
                    f"{', '.join(words)} = {_RUNTIME}.lds_v4_b32({raw_smem} + {pk.offset(raw.row_bytes)} + {pk.swz} * 16)"
                )
                for q, word in enumerate(words):
                    lo, hi = self.df.new_var("md_lo"), self.df.new_var("md_hi")
                    self.emit(f"{lo}, {hi} = {_RUNTIME}.{unpack}({word})")
                    self.set_tile_leaf(load, 2 * q, lo)
                    self.set_tile_leaf(load, 2 * q + 1, hi)
                values = []
                for e in range(0, 8, 2):
                    lanes = tuple(
                        _Lane(
                            self.tile_coords(operand, prow, f"({pch} * 8 + {e + bit})"),
                            e + bit,
                            None,
                        )
                        for bit in range(2)
                    )
                    values.extend(
                        self.masked(dot, operand, lane.coords, value)
                        for lane, value in zip(
                            lanes,
                            self.evaluator.pair(operand.value, lanes),
                            strict=True,
                        )
                    )
                self.evaluator.element = None
                if operand.transposed_tile:
                    # Element (prow, 8p + e) of the source lands at row 8p + e,
                    # column prow of the transposed fp32 tile.  The thread's
                    # own row is below the rows per round, which divide 4, so
                    # the round offset moves ``prow // 4`` by whole chunks.
                    col4 = (
                        f"({pk.pr} // 4 + {pk.row_off // 4})"
                        if pk.row_off
                        else f"({pk.pr} // 4)"
                    )
                    for e, value in enumerate(values):
                        row = f"({pch} * 8 + {e})"
                        dst = _transposed_swizzle(row, col4, cpr)
                        self.emit(
                            f"{_RUNTIME}.sts_f32({smem} + {row} * {row_bytes} + {dst} * 16 + {pk.srow} % 4 * 4, {value})"
                        )
                    return
                # Raw packet p of a row holds columns 8p..8p+7: fp32 chunks
                # 2p and 2p + 1.
                for half in range(2):
                    dst = _swizzled_chunk(pk.srow, f"({pch} * 2 + {half})", cpr)
                    self.emit(
                        f"{_RUNTIME}.sts_v4_f32({smem} + {pk.offset(row_bytes)} + {dst} * 16, "
                        f"{', '.join(values[4 * half : 4 * half + 4])})"
                    )

            self.own_packets(raw, widen)
            return

        def body(pk: _Packet) -> None:
            prow, pch = pk.prow, pk.pch
            values = [self.df.new_var("md_f") for _ in range(4)]
            self.emit(
                f"{', '.join(values)} = {_RUNTIME}.lds_v4_f32({raw_smem} + {pk.offset(raw.row_bytes)} + {pk.swz} * 16)"
            )
            for e, value in enumerate(values):
                self.evaluator.tile_leaves[(load.node, e)] = value
            packed = []
            for q in range(2):
                lanes = tuple(
                    _Lane(
                        self.tile_coords(operand, prow, f"({pch} * 4 + {e})"), e, None
                    )
                    for e in (2 * q, 2 * q + 1)
                )
                pair = self.evaluator.unrounded_pair(operand.value, lanes)
                self.evaluator.element = None
                pair = tuple(
                    self.masked(dot, operand, lane.coords, value)
                    for lane, value in zip(lanes, pair, strict=True)
                )
                packed.append(self.pack(pair[0], pair[1], operand.dtype))
            # Raw packet p of a row holds columns 4p..4p+3: 16-bit chunk p // 2,
            # byte offset 8 * (p % 2) within it.
            dst = _swizzled_chunk(pk.srow, f"({pch} // 2)", cpr)
            self.emit(
                f"{_RUNTIME}.sts_v2_b32({smem} + {pk.offset(row_bytes)} + {dst} * 16 + ({pch} % 2) * 8, "
                f"{', '.join(packed)})"
            )

        self.own_packets(raw, body)

    def emit_sync_pass(self, dot: Dot, operand: Operand, stage: str) -> None:
        """Cooperative pass for an unaligned source: every thread evaluates
        element pairs along the contiguous dim straight from global memory
        and packs them into the swizzled shared tile."""
        load = operand.load
        assert load is not None
        tile = self.tiles[(dot.node, operand.role)]
        if operand.transposed_tile:
            raise _decline("a transposed tf32 operand tile needs an aligned source")
        smem = tile.base(stage)
        pairs = tile.rows * tile.cols // 2
        cpp = tile.cols // 2
        threads = self.plan.threads
        paired = self.pairs_aligned(load, operand.cols_pos)
        load_es = load.dtype.itemsize
        row_stride = load.tile_dim_strides[operand.rows_pos] * load_es
        self.hoist_scalars(operand.value)
        for rep in range(-(-pairs // threads)):
            idx = f"({rep * threads} + md_tidx)" if rep else "md_tidx"
            guard = pairs < (rep + 1) * threads
            nest = self.nest(f"if {idx} < {pairs}:") if guard else None
            if nest is not None:
                nest.__enter__()
            self.evaluator.push()
            prow = self.bind("md_pr", f"{idx} // {cpp}")
            pcol = self.bind("md_pc", f"{idx} % {cpp}")
            address = (
                f"{self.iter_vars[load.node]} + cutlass.Int64({prow}) * {row_stride} "
                f"+ cutlass.Int64({pcol}) * {2 * load_es}"
            )
            guard = self.row_exists(load, operand.rows_pos, prow)
            pred = (
                self.bind("md_ok", self.pred_int(guard)) if guard is not None else None
            )
            lo, hi = self.pair_load(
                address, load.dtype, paired, pred, readonly=self.readonly(load)
            )
            self.set_tile_leaf(load, 0, lo)
            self.set_tile_leaf(load, 1, hi)
            values = []
            for half in range(2):
                self.evaluator.element = half
                col = f"(2 * {pcol} + {half})" if half else f"(2 * {pcol})"
                coords = self.tile_coords(operand, prow, col)
                values.append(
                    self.masked(
                        dot,
                        operand,
                        coords,
                        self.evaluator.unrounded(operand.value, coords),
                    )
                )
            self.evaluator.element = None
            es = operand.dtype.itemsize
            per_chunk = 16 // (2 * es)
            # Pair (2p, 2p + 1) sits in 16-byte chunk p // per_chunk at byte
            # 2 * es * (p % per_chunk).
            swz = _swizzled_chunk(prow, f"({pcol} // {per_chunk})", tile.cpr)
            address = f"{smem} + {prow} * {tile.row_bytes} + {swz} * 16 + ({pcol} % {per_chunk}) * {2 * es}"
            if es == 4:
                self.emit(f"{_RUNTIME}.sts_v2_f32({address}, {values[0]}, {values[1]})")
            else:
                self.emit(
                    f"{_RUNTIME}.sts_b32({address}, {self.pack(values[0], values[1], operand.dtype)})"
                )
            self.evaluator.pop()
            if nest is not None:
                nest.__exit__(None, None, None)

    def pass_grid(self, operand: Operand) -> tuple[int, int]:
        """``(rows, cols)`` of a pass chain's source tiles (the chain's
        fragment shape: every source is loaded at those coordinates)."""
        root = self.program.chain_root(operand)
        assert root is not None
        return self.plan.slot_shape(self.program.frag_slots[root])

    def emit_derived_pass(self, dot: Dot, operand: Operand, stage: str) -> None:
        """Write a pass chain's operand tile from the chain's staged source
        tiles (``cute_mma_dag_derived='pass'``): every thread takes
        eight-element groups of the source rows round-robin over the CTA
        (one 16-byte packet of a 16-bit source, two of an fp32 one), loads
        each source's group from its ring slot, evaluates the operand value
        through the chain's pointwise DAG in fp32 and stores the eight
        results into the swizzled operand tile -- packed and rounded once
        for a 16-bit operand, as two fp32 packets for a tf32 one (scattered
        per element into a transposed tf32 B tile), the way the convert
        pass widens a single source."""
        program = self.program
        root = program.chain_root(operand)
        assert root is not None
        tile = self.tiles[(dot.node, operand.role)]
        smem = tile.base(stage)
        rows, cols = self.pass_grid(operand)
        sources = []
        for access in program.chain_loads(root):
            source = self.derived_tiles[access.node]
            assert (source.rows, source.cols) == (rows, cols)
            sources.append((access, source, source.base(self.stage_var(dot.loop))))
        inline = frozenset(step.node for step in program.pass_chains[root])
        saved = self.evaluator.inline
        self.evaluator.inline = inline
        self.hoist_scalars(operand.value)
        # The groups follow a 16-bit tile's packet grid (eight elements each).
        grid = _Tile("", 0, rows, cols, 2, 1)
        es = operand.dtype.itemsize
        cpr, row_bytes = tile.cpr, tile.row_bytes

        def body(pk: _Packet) -> None:
            prow, pg = pk.prow, pk.pch
            for access, source, base in sources:
                if access.dtype == torch.float32:
                    # Group g of an fp32 row: its chunks 2g and 2g + 1.
                    for half in range(2):
                        swz = _swizzled_chunk(
                            pk.srow, f"({pg} * 2 + {half})", source.cpr
                        )
                        names = [self.df.new_var("md_f") for _ in range(4)]
                        self.emit(
                            f"{', '.join(names)} = {_RUNTIME}.lds_v4_f32("
                            f"{base} + {pk.offset(source.row_bytes)} + {swz} * 16)"
                        )
                        for e, name in enumerate(names):
                            self.set_tile_leaf(access, 4 * half + e, name)
                    continue
                unpack = (
                    "unpack_f16x2" if access.dtype == torch.float16 else "unpack_bf16x2"
                )
                words = [self.df.new_var("md_w") for _ in range(4)]
                self.emit(
                    f"{', '.join(words)} = {_RUNTIME}.lds_v4_b32("
                    f"{base} + {pk.offset(source.row_bytes)} + {pk.swz} * 16)"
                )
                for q, word in enumerate(words):
                    lo, hi = self.df.new_var("md_lo"), self.df.new_var("md_hi")
                    self.emit(f"{lo}, {hi} = {_RUNTIME}.{unpack}({word})")
                    self.set_tile_leaf(access, 2 * q, lo)
                    self.set_tile_leaf(access, 2 * q + 1, hi)
            values = []
            for e in range(0, 8, 2):
                lanes = tuple(
                    _Lane(
                        self.tile_coords(operand, prow, f"({pg} * 8 + {e + bit})"),
                        e + bit,
                        None,
                    )
                    for bit in range(2)
                )
                if es == 4:
                    values.extend(self.evaluator.pair(operand.value, lanes))
                else:
                    values.extend(self.evaluator.unrounded_pair(operand.value, lanes))
            self.evaluator.element = None
            if es != 4:
                packed = [
                    self.pack(values[2 * q], values[2 * q + 1], operand.dtype)
                    for q in range(4)
                ]
                self.emit(
                    f"{_RUNTIME}.sts_v4_b32({smem} + {pk.offset(row_bytes)} + {pk.swz} * 16, "
                    f"{', '.join(packed)})"
                )
                return
            if operand.transposed_tile:
                # Element (prow, 8g + e) of the source lands at row 8g + e,
                # column prow of the transposed fp32 tile (see the convert
                # pass: the rows per round divide 4).
                col4 = (
                    f"({pk.pr} // 4 + {pk.row_off // 4})"
                    if pk.row_off
                    else f"({pk.pr} // 4)"
                )
                for e, value in enumerate(values):
                    row = f"({pg} * 8 + {e})"
                    dst = _transposed_swizzle(row, col4, cpr)
                    self.emit(
                        f"{_RUNTIME}.sts_f32({smem} + {row} * {row_bytes} + {dst} * 16 + {pk.srow} % 4 * 4, {value})"
                    )
                return
            for half in range(2):
                dst = _swizzled_chunk(pk.srow, f"({pg} * 2 + {half})", cpr)
                self.emit(
                    f"{_RUNTIME}.sts_v4_f32({smem} + {pk.offset(row_bytes)} + {dst} * 16, "
                    f"{', '.join(values[4 * half : 4 * half + 4])})"
                )

        self.own_packets(grid, body)
        self.evaluator.inline = saved

    def emit_exchange(self, dot: Dot, operand: Operand) -> None:
        """Write the operand value (a pointwise DAG over a fragment) into its
        exchange tile at the fragment's own coordinates, as packed pairs."""
        assert operand.frag is not None
        tile = self.shared_tiles[operand.chain_key]
        if operand.chain_key in self.overlaid_tiles:
            # The tile overlays the loops' frames: every lane finished the
            # last loop's reads of them first.
            self.emit(self.sync)
        base = tile.base(self.tile_buffers[(dot.node, operand.role)])
        slot = self.program.frag_slots[operand.frag]
        layout = self.slot_layouts[slot]
        es = operand.dtype.itemsize
        self.hoist_scalars(operand.value)
        for am in range(layout.atoms_m):
            for an in range(layout.atoms_n):
                for half in range(2):
                    lanes = tuple(
                        _Lane(
                            self.frag_coords(
                                operand.value,
                                layout,
                                am,
                                an,
                                2 * half + bit,
                                operand.swap,
                            ),
                            None,
                            (am, an, 2 * half + bit),
                        )
                        for bit in range(2)
                    )
                    values = self.evaluator.unrounded_pair(operand.value, lanes)
                    self.evaluator.reg = None
                    row, col = layout.row(am, half), layout.col(an)
                    if operand.transposed_tile:
                        # Fragment element (row, col + bit) -> tile (col + bit, row).
                        for bit in range(2):
                            trow = f"({col} + {bit})" if bit else col
                            swz = _transposed_swizzle(trow, f"(({row}) // 4)", tile.cpr)
                            self.emit(
                                f"{_RUNTIME}.sts_f32({base} + {trow} * {tile.row_bytes} + {swz} * 16 + ({row}) % 4 * 4, {values[bit]})"
                            )
                        continue
                    chunk = f"(({col}) * {es} // 16)"
                    swz = _swizzled_chunk(row, chunk, tile.cpr)
                    address = f"{base} + {row} * {tile.row_bytes} + {swz} * 16 + ({col}) * {es} % 16"
                    if es == 4:
                        self.emit(
                            f"{_RUNTIME}.sts_v2_f32({address}, {values[0]}, {values[1]})"
                        )
                    else:
                        self.emit(
                            f"{_RUNTIME}.sts_b32({address}, {self.pack(values[0], values[1], operand.dtype)})"
                        )

    # -- carry snapshots (loop stores) ---------------------------------------

    def emit_stage_in(self, store: Store, parity: str) -> None:
        """Write the fragment pairs into the staging buffer of ``parity``."""
        tile = self.staging[store.access.node]
        slot = self.program.frag_slots[store.frag]
        layout = self.slot_layouts[slot]
        es = store.access.dtype.itemsize
        base = tile.base(parity)
        pack = "pack_f16x2" if store.access.dtype == torch.float16 else "pack_bf16x2"
        # A TMA-stored tile wider than one 128-byte atom is laid out as the
        # box layout of its slabs side by side: a slab's rows at a 128-byte
        # pitch, the atom's swizzle within (``plan_tma_stores``).
        slabs = tile.cpr > 8 and store.access.node in self.store_tma
        for am in range(layout.atoms_m):
            for an in range(layout.atoms_n):
                for half in range(2):
                    row, col = layout.row(am, half), layout.col(an)
                    chunk = f"(({col}) * {es} // 16)"
                    if slabs:
                        swz = _swizzled_chunk(row, f"(({chunk}) % 8)", 8)
                        address = (
                            f"{base} + (({chunk}) // 8) * {tile.rows * 128} + {row} * 128"
                            f" + {swz} * 16 + ({col}) * {es} % 16"
                        )
                    else:
                        swz = _staging_swizzle(row, chunk, tile.cpr, es)
                        address = f"{base} + {row} * {tile.row_bytes} + {swz} * 16 + ({col}) * {es} % 16"
                    lo = self.reg(slot, am, an, 2 * half)
                    hi = self.reg(slot, am, an, 2 * half + 1)
                    if store.access.dtype == torch.float32:
                        self.emit(f"{_RUNTIME}.sts_v2_f32({address}, {lo}, {hi})")
                    else:
                        self.emit(
                            f"{_RUNTIME}.sts_b32({address}, {_RUNTIME}.{pack}({lo}, {hi}))"
                        )

    def emit_stage_out(self, store: Store, parity: str) -> None:
        """Copy the staging buffer of ``parity`` to global memory as 16-byte
        packets, every thread its round-robin share (after the CTA barrier
        that follows its writes); the global stores are deferred past the
        first fragment loads so the shared-memory latency is not exposed."""
        tile = self.staging[store.access.node]
        access = store.access
        es = access.dtype.itemsize
        row_stride = access.tile_dim_strides[0] * es
        gbase = self.iteration_base(access, self.values)
        base = tile.base(parity)
        threads = self.plan.threads
        # The staging rows are the store's first tile dim (``staged_store_
        # reason``); a ragged axis or a window there drops the packets past
        # the extent / the limit.
        for rep in range(-(-tile.packets // threads)):
            idx = f"({rep * threads} + md_tidx)" if rep else "md_tidx"
            guard = tile.packets < (rep + 1) * threads
            nest = self.nest(f"if {idx} < {tile.packets}:") if guard else None
            if nest is not None:
                nest.__enter__()
            prow = self.bind("md_orow", f"{idx} // {tile.cpr}")
            pch = self.bind("md_och", f"{idx} % {tile.cpr}")
            swz = _staging_swizzle(prow, pch, tile.cpr, es)
            words = [self.df.new_var("md_w") for _ in range(4)]
            self.emit(
                f"{', '.join(words)} = {_RUNTIME}.lds_v4_b32({base} + {prow} * {tile.row_bytes} + {swz} * 16)"
            )
            line = (
                f"{_RUNTIME}.stg_v4_b32({gbase} + cutlass.Int64({prow}) * {row_stride} "
                f"+ cutlass.Int64({pch} * 16), {', '.join(words)})"
            )
            row_guard = self.row_exists(access, 0, prow)
            if nest is not None:
                self.emit_guarded(row_guard, line)
                nest.__exit__(None, None, None)
            else:
                self.deferred.append((row_guard, line))

    def emit_tma_stores(self, stores: Sequence[Store], parity: str) -> None:
        """The store thread: the staging buffers of ``parity`` of ``stores``
        to global memory, one bulk tensor copy per 128-byte slab, committed
        as one bulk group (its staging reads are retired ahead of the next
        iteration's barrier and at the loop's exit; the global writes
        complete at the kernel's end)."""
        with self.nest("if md_tidx == 0:"):
            for store in stores:
                spec = self.store_tma[store.access.node]
                tile = self.staging[store.access.node]
                for slab in range(spec.slabs):
                    coords = ", ".join(
                        self.tma_coords(
                            spec, store.access, self.values, slab * spec.slab_elems
                        )
                    )
                    src = f"{spec.offset + slab * spec.slab_bytes} + ({parity}) * {tile.bytes}"
                    self.emit(
                        f"{_TC}.prims.cp_async_bulk_tensor_global_shared_cta("
                        f"{spec.param}.get_ptr(), md_smem + ({src}), [{coords}])"
                    )
            self.emit(f"{_TC}.prims.cp_async_bulk_commit_group()")

    def emit_guarded(self, guard: str | None, line: str) -> None:
        if guard is None:
            self.emit(line)
            return
        with self.nest(f"if {guard}:"):
            self.emit(line)

    def flush_deferred(self) -> None:
        for guard, line in self.deferred:
            self.emit_guarded(guard, line)
        self.deferred = []

    def emit_store(self, store: Store) -> None:
        """Pair stores of a fragment-derived value at the store's coordinates."""
        access = store.access
        slot = self.program.frag_slots[store.frag]
        layout = self.slot_layouts[slot]
        positions = access.tile_positions
        paired = self.pairs_aligned(access, 1) and not store.swap
        self.hoist_scalars(store.value)
        self.evaluator.enter_frag(slot)
        for am in range(layout.atoms_m):
            for an in range(layout.atoms_n):
                for half in range(2):
                    lanes = tuple(
                        _Lane(
                            self.frag_coords(
                                store.value, layout, am, an, 2 * half + bit, store.swap
                            ),
                            None,
                            (am, an, 2 * half + bit),
                        )
                        for bit in range(2)
                    )
                    # Paired 16-bit stores pack (and so round) the pair.
                    if paired and access.dtype != torch.float32:
                        values = list(self.evaluator.unrounded_pair(store.value, lanes))
                    else:
                        values = list(self.evaluator.pair(store.value, lanes))
                    self.evaluator.reg = None
                    row, col = layout.row(am, half), layout.col(an)
                    # The store's positional coordinates of the fragment element.
                    coords = ["0"] * len(access.dims)
                    first, second = (col, row) if store.swap else (row, col)
                    coords[positions[0]] = first
                    coords[positions[1]] = second
                    # Elements past a ragged axis' extent are not stored.
                    if paired:
                        address = self.address(access, tuple(coords))
                        guard = self.bounds(access, tuple(coords))
                        with (
                            self.nest(f"if {guard}:")
                            if guard is not None
                            else contextlib.nullcontext()
                        ):
                            self.pair_store(
                                address, values[0], values[1], access.dtype, True
                            )
                    else:
                        for bit in range(2):
                            element = list(coords)
                            element[positions[1 if not store.swap else 0]] = (
                                f"({col} + {bit})" if bit else col
                            )
                            address = self.address(access, tuple(element))
                            scalar = {
                                torch.float32: "stg_f32",
                                torch.float16: "stg_f16",
                                torch.bfloat16: "stg_bf16",
                            }[access.dtype]
                            self.emit_guarded(
                                self.bounds(access, tuple(element)),
                                f"{_RUNTIME}.{scalar}({address}, {values[bit]})",
                            )
        self.evaluator.leave_frag()

    def emit_carry_init(self, loop: Loop) -> None:
        for carry in loop.carries:
            layout = self.slot_layouts[carry.slot]
            if carry.init_value is not None and carry.init is None:
                # A pointwise value over a fragment of the enclosing region,
                # evaluated per register into the carry's registers.
                self.hoist_scalars(carry.init_value)
                self.evaluator.enter_frag(carry.slot)
                updates = []
                for am, an, j in layout.registers():
                    self.evaluator.reg = (am, an, j)
                    coords = self.frag_coords(carry.init_value, layout, am, an, j)
                    updates.append(
                        (
                            self.reg(carry.slot, am, an, j),
                            self.evaluator.evaluate(carry.init_value, coords),
                        )
                    )
                self.evaluator.leave_frag()
                for name, value in updates:
                    self.emit(f"{name} = {value}")
                continue
            if carry.init_value is not None:
                # A pointwise value over a 2-D load (``A = -(beta * Akk[b])``):
                # the tile's pairs loaded at the fragment coordinates, the
                # chain evaluated per register into the carry's registers.
                init = carry.init
                assert init is not None
                self.hoist_scalars(carry.init_value)
                self.evaluator.enter_frag(carry.slot)
                paired = self.pairs_aligned(init, 1)
                positions = init.tile_positions
                updates = []
                for am in range(layout.atoms_m):
                    for an in range(layout.atoms_n):
                        for half in range(2):
                            coords = ["0"] * len(init.dims)
                            coords[positions[0]] = layout.row(am, half)
                            coords[positions[1]] = layout.col(an)
                            address = self.address(init, tuple(coords))
                            lo, hi = self.pair_load(
                                address,
                                init.dtype,
                                paired,
                                self.bounds_pred(init, tuple(coords)),
                            )
                            self.set_tile_leaf(init, 0, lo)
                            self.set_tile_leaf(init, 1, hi)
                            lanes = tuple(
                                _Lane(
                                    self.frag_coords(
                                        carry.init_value, layout, am, an, 2 * half + bit
                                    ),
                                    bit,
                                    (am, an, 2 * half + bit),
                                )
                                for bit in range(2)
                            )
                            for bit, value in enumerate(
                                self.evaluator.pair(carry.init_value, lanes)
                            ):
                                updates.append(
                                    (
                                        self.reg(carry.slot, am, an, 2 * half + bit),
                                        value,
                                    )
                                )
                            self.evaluator.reg = None
                            self.evaluator.element = None
                self.evaluator.leave_frag()
                for name, value in updates:
                    self.emit(f"{name} = {value}")
                continue
            if carry.init_frag is not None:
                # An accumulator of the enclosing region carried on: its own
                # registers, or a copy when something else still reads them.
                source = self.program.frag_slots[carry.init_frag]
                if source != carry.slot:
                    for am, an, j in layout.registers():
                        self.emit(
                            f"{self.reg(carry.slot, am, an, j)} = {self.reg(source, am, an, j)}"
                        )
                continue
            if carry.init is None:
                for am in range(layout.atoms_m):
                    for an in range(layout.atoms_n):
                        self.emit(
                            "; ".join(
                                f"{self.reg(carry.slot, am, an, j)} = cutlass.Float32(0.0)"
                                for j in range(4)
                            )
                        )
                continue
            init = carry.init
            paired = self.pairs_aligned(init, 1)
            positions = [
                i for i, axis in enumerate(init.dims) if axis.kind != AXIS_LEAD
            ]
            for am in range(layout.atoms_m):
                for an in range(layout.atoms_n):
                    for half in range(2):
                        coords = ["0"] * len(init.dims)
                        coords[positions[0]] = layout.row(am, half)
                        coords[positions[1]] = layout.col(an)
                        address = self.address(init, tuple(coords))
                        lo, hi = self.pair_load(
                            address,
                            init.dtype,
                            paired,
                            self.bounds_pred(init, tuple(coords)),
                            readonly=self.readonly(init),
                        )
                        self.emit(f"{self.reg(carry.slot, am, an, 2 * half)} = {lo}")
                        self.emit(
                            f"{self.reg(carry.slot, am, an, 2 * half + 1)} = {hi}"
                        )

    def emit_acc(self, dot: Dot) -> None:
        """Materialize the dot's accumulator registers: zeros, the fragment
        leaf in place, or a pointwise DAG over the leaf evaluated per register."""
        layout = self.slot_layouts[dot.slot]
        if dot.acc is None:
            for am, an, j in layout.registers():
                self.emit(f"{self.reg(dot.slot, am, an, j)} = cutlass.Float32(0.0)")
            return
        if dot.acc_frag is not None and self.program.resolve(dot.acc) is dot.acc_frag:
            if dot.slot != dot.acc_slot:
                for am, an, j in layout.registers():
                    self.emit(
                        f"{self.reg(dot.slot, am, an, j)} = {self.reg(cast('int', dot.acc_slot), am, an, j)}"
                    )
            return
        self.hoist_scalars(dot.acc)
        # (The accumulator's slot shares the dot's layout: same shape.)
        self.evaluator.enter_frag(dot.slot)
        updates = []
        for am, an, j in layout.registers():
            self.evaluator.reg = (am, an, j)
            coords = self.frag_coords(dot.acc, layout, am, an, j)
            updates.append(
                (
                    self.reg(dot.slot, am, an, j),
                    self.evaluator.evaluate(dot.acc, coords),
                )
            )
        self.evaluator.leave_frag()
        for name, value in updates:
            self.emit(f"{name} = {value}")

    def emit_frag_words(
        self, dot: Dot, operand: Operand
    ) -> dict[tuple[int, int], list[str]]:
        """The operand words of every (atom, K step) from the producer
        fragment's registers (see ``frag_register_route``).

        The producer's C layout holds, per 8x8 block ``(am, half)`` x ``an``,
        row ``g`` / columns ``2c, 2c + 1`` in each lane.  An ``[M, K]`` A
        operand's four words are those blocks packed (rows ``g`` / ``g + 8``,
        K blocks ``2 st`` / ``2 st + 1``); an ``[N, K]`` B operand's two words
        are the packed blocks (n-block ``an``, K block ``2 st`` / ``2 st +
        1``); a ``[K, N]`` B operand needs each block transposed
        (``movmatrix``): K block ``kb`` is the producer's rows ``(kb // 2, kb
        % 2)``.
        """
        assert operand.frag is not None
        slot = self.program.frag_slots[operand.frag]
        layout = self.slot_layouts[slot]
        self.hoist_scalars(operand.value)
        values: dict[tuple[int, int, int], str] = {}
        for am, an, j in layout.registers():
            self.evaluator.reg = (am, an, j)
            coords = self.frag_coords(operand.value, layout, am, an, j, operand.swap)
            values[(am, an, j)] = self.evaluator.unrounded(operand.value, coords)
        self.evaluator.reg = None
        route = self.plan.route(dot, operand)

        def block(am: int, an: int, half: int) -> str:
            packed = self.pack(
                values[(am, an, 2 * half)],
                values[(am, an, 2 * half + 1)],
                operand.dtype,
            )
            if route == ROUTE_MOVMATRIX:
                return f"{_RUNTIME}.movmatrix_trans({packed})"
            return packed

        words: dict[tuple[int, int], list[str]] = {}
        k = self.plan.size(dot.k)
        consumer = self.slot_layouts[dot.slot]
        atoms = consumer.atoms_m if operand.role == "a" else consumer.atoms_n
        prefix = "md_aw" if operand.role == "a" else "md_bw"
        for atom in range(atoms):
            for st in range(k // 16):
                words[(atom, st)] = [
                    self.bind(prefix, block(*blk))
                    for blk in frag_word_blocks(operand.role, route, atom, st)
                ]
        return words

    def shuffle_words(
        self, dot: Dot, operand: Operand, atom: int, st: int
    ) -> list[str]:
        """The fp32 operand words of consumer atom ``atom`` at K step ``st``
        (eight K values) picked from the producer fragment's registers in
        other lanes of the same warp (``ROUTE_SHUFFLE``, see
        ``frag_register_route`` for the warp grids that allow it).

        The producer's C layout holds, per 8x8 block, row ``g`` / columns
        ``2c, 2c + 1`` in lane ``(g, c)``.  An A word ``(row g, k c)`` or an
        ``[N, K]`` B word ``(n g, k c)`` sits in lane ``(g, c // 2)`` at
        register parity ``c % 2``; a ``[K, N]`` B word ``(k c, n g)`` in lane
        ``(c, g // 2)`` at parity ``g % 2``.  ``k + 4`` moves the source
        column by two lanes (A, ``[N, K]``) or the source row by sixteen
        (``[K, N]``).
        """
        assert operand.frag is not None
        slot = self.program.frag_slots[operand.frag]
        layout = self.slot_layouts[slot]
        transposed = operand.role == "b" and not operand.logical_swap
        lane, flag = self.quad_terms["t" if transposed else "q"]

        def bits(am: int, an: int, j: int) -> str:
            self.evaluator.reg = (am, an, j)
            coords = self.frag_coords(operand.value, layout, am, an, j, operand.swap)
            value = self.evaluator.unrounded(operand.value, coords)
            self.evaluator.reg = None
            return f"{_RUNTIME}.f32_bits({value})"

        def pick(src: str, lo: str, hi: str) -> str:
            return f"{_RUNTIME}.shfl_pick_b32({src}, {flag}, {lo}, {hi})"

        far = f"({lane} + 16)" if transposed else f"({lane} + 2)"
        if operand.role == "a":
            # Rows g / g + 8 of atom row ``atom``, producer n-atom ``st``.
            return [
                pick(lane, bits(atom, st, 0), bits(atom, st, 1)),
                pick(lane, bits(atom, st, 2), bits(atom, st, 3)),
                pick(far, bits(atom, st, 0), bits(atom, st, 1)),
                pick(far, bits(atom, st, 2), bits(atom, st, 3)),
            ]
        if transposed:
            # K rows 8 st .. 8 st + 7 of the producer: block (st // 2, st % 2)
            # of its n-atom ``atom``.
            am, half = st // 2, st % 2
            lo, hi = bits(am, atom, 2 * half), bits(am, atom, 2 * half + 1)
        else:
            # N rows of atom column ``atom``: block (atom // 2, atom % 2) of
            # the producer's n-atom ``st``.
            am, half = atom // 2, atom % 2
            lo, hi = bits(am, st, 2 * half), bits(am, st, 2 * half + 1)
        return [pick(lane, lo, hi), pick(far, lo, hi)]

    def hoisted_shuffle_words(
        self, dot: Dot, operand: Operand
    ) -> dict[tuple[int, int], list[str]] | None:
        """The shuffle-routed words of every (atom, K step) bound before the
        MMAs when the producer is the dot's own accumulator (the MMAs rewrite
        its registers step by step); None when they can be picked per step."""
        assert operand.frag is not None
        if self.program.frag_slots[operand.frag] != dot.slot:
            return None
        layout = self.slot_layouts[dot.slot]
        atoms = layout.atoms_m if operand.role == "a" else layout.atoms_n
        prefix = "md_aw" if operand.role == "a" else "md_bw"
        steps = self.plan.size(dot.k) // self.k_step(operand.dtype)
        return {
            (atom, st): [
                self.bind(prefix, word)
                for word in self.shuffle_words(dot, operand, atom, st)
            ]
            for atom in range(atoms)
            for st in range(steps)
        }

    def emit_dot(self, dot: Dot, stage: str) -> None:
        """K / 16 steps of ldmatrix + mma.sync accumulating into the slot."""
        plan = self.plan
        layout = self.slot_layouts[dot.slot]
        k = plan.size(dot.k)
        dtype = dot.a.dtype
        tf32 = dtype == torch.float32
        steps = k // self.k_step(dtype)
        mma = f"{_RUNTIME}.mma_{self.mma_kind(dtype)}"
        packed_routes = (ROUTE_IDENTITY, ROUTE_MOVMATRIX)
        a_words = (
            self.emit_frag_words(dot, dot.a)
            if plan.route(dot, dot.a) in packed_routes
            else None
        )
        b_words = (
            self.emit_frag_words(dot, dot.b)
            if plan.route(dot, dot.b) in packed_routes
            else None
        )
        a_shuffle = plan.route(dot, dot.a) == ROUTE_SHUFFLE
        b_shuffle = plan.route(dot, dot.b) == ROUTE_SHUFFLE
        a_hoisted = b_hoisted = None
        if a_shuffle:
            self.hoist_scalars(dot.a.value)
            a_hoisted = self.hoisted_shuffle_words(dot, dot.a)
        if b_shuffle:
            self.hoist_scalars(dot.b.value)
            b_hoisted = self.hoisted_shuffle_words(dot, dot.b)
        a_tile = self.tiles.get((dot.node, "a"))
        b_tile = self.tiles.get((dot.node, "b"))
        # Per-lane ldmatrix row / chunk terms (see ``ldmatrix_lane_address``);
        # the warp's M / N offset follows the slot's warp grid.
        wmi, wni = f"md_L{layout.index}_wmi", f"md_L{layout.index}_wni"
        wtm, wtn = layout.wtm, layout.wtn
        arow = achunk = a_base = brow = bchunk = b_base = acol = bcol = ""
        # tf32 operands whose K runs along the tile rows cannot use
        # ``ldmatrix.trans`` (it transposes 16-bit elements): each lane loads
        # its fp32 values (rows ``c`` / ``c + 4`` of the K step at its M / N
        # index ``g``).
        a_lanes = a_tile is not None and tf32 and dot.a.trans
        b_lanes = b_tile is not None and tf32 and dot.b.trans
        if a_tile is not None:
            if a_lanes:
                acol = self.bind("md_acol4", f"({wmi} * {wtm}) // 4 + md_g // 4")
            elif dot.a.trans:
                arow = self.bind("md_arow", "md_l8 + 8 * (md_lq // 2)")
                achunk = self.bind("md_achunk", f"({wmi} * {wtm}) // 8 + md_lq % 2")
            else:
                arow = self.bind("md_arow", f"{wmi} * {wtm} + md_l8 + 8 * (md_lq % 2)")
                achunk = self.bind("md_achunk", "md_lq // 2")
            a_base = a_tile.base(self.tile_buffers[(dot.node, "a")])
        if b_tile is not None:
            if b_lanes:
                bcol = self.bind("md_bcol4", f"({wni} * {wtn}) // 4 + md_g // 4")
            elif dot.b.trans:
                brow = self.bind("md_brow", "md_l8 + 8 * (md_lq % 2)")
                bchunk = self.bind("md_bchunk", f"({wni} * {wtn}) // 8 + md_lq // 2")
            else:
                brow = self.bind("md_brow", f"{wni} * {wtn} + md_l8 + 8 * (md_lq // 2)")
                bchunk = self.bind("md_bchunk", "md_lq % 2")
            b_base = b_tile.base(self.tile_buffers[(dot.node, "b")])
        if a_lanes or b_lanes:
            self.emit("md_gofs = md_g % 4 * 4")
        for st in range(steps):
            for am in range(layout.atoms_m):
                if a_words is not None:
                    a0, a1, a2, a3 = a_words[(am, st)]
                    self.emit(
                        f"md_a_{am}_0, md_a_{am}_1, md_a_{am}_2, md_a_{am}_3 = {a0}, {a1}, {a2}, {a3}"
                    )
                    continue
                if a_shuffle:
                    words = (
                        a_hoisted[(am, st)]
                        if a_hoisted is not None
                        else self.shuffle_words(dot, dot.a, am, st)
                    )
                    self.emit(
                        f"md_a_{am}_0, md_a_{am}_1, md_a_{am}_2, md_a_{am}_3 = "
                        f"{', '.join(words)}"
                    )
                    continue
                assert a_tile is not None
                if a_lanes:
                    # (row g, k c), (row g + 8, k c), (row g, k c + 4), (row
                    # g + 8, k c + 4): rows k of the tile, columns m.
                    words = []
                    for k_half in range(2):
                        for m_half in range(2):
                            row = f"({8 * st + 4 * k_half} + md_c)"
                            chunk = f"({acol} + {4 * am + 2 * m_half})"
                            swz = _swizzled_chunk(row, chunk, a_tile.cpr)
                            words.append(
                                f"{_RUNTIME}.lds_b32({a_base} + {row} * {a_tile.row_bytes} + {swz} * 16 + md_gofs)"
                            )
                    self.emit(
                        f"md_a_{am}_0, md_a_{am}_1, md_a_{am}_2, md_a_{am}_3 = {', '.join(words)}"
                    )
                    continue
                if dot.a.trans:
                    row = f"({16 * st} + {arow})"
                    chunk = f"({achunk} + {2 * am})" if am else achunk
                    op = "ldmatrix_x4_trans"
                else:
                    row = f"({arow} + {16 * am})" if am else arow
                    chunk = f"({2 * st} + {achunk})"
                    op = "ldmatrix_x4"
                swz = _swizzled_chunk(row, chunk, a_tile.cpr)
                self.emit(
                    f"md_a_{am}_0, md_a_{am}_1, md_a_{am}_2, md_a_{am}_3 = {_RUNTIME}.{op}("
                    f"{a_base} + {row} * {a_tile.row_bytes} + {swz} * 16)"
                )
            if b_words is not None:
                for an in range(layout.atoms_n):
                    b0, b1 = b_words[(an, st)]
                    self.emit(f"md_b_{an}_0, md_b_{an}_1 = {b0}, {b1}")
            if b_shuffle:
                for an in range(layout.atoms_n):
                    words = (
                        b_hoisted[(an, st)]
                        if b_hoisted is not None
                        else self.shuffle_words(dot, dot.b, an, st)
                    )
                    self.emit(f"md_b_{an}_0, md_b_{an}_1 = {', '.join(words)}")
            if b_lanes:
                assert b_tile is not None
                for an in range(layout.atoms_n):
                    # (k c, n g) and (k c + 4, n g): rows k, columns n.
                    words = []
                    for k_half in range(2):
                        row = f"({8 * st + 4 * k_half} + md_c)"
                        chunk = f"({bcol} + {2 * an})"
                        swz = _swizzled_chunk(row, chunk, b_tile.cpr)
                        words.append(
                            f"{_RUNTIME}.lds_b32({b_base} + {row} * {b_tile.row_bytes} + {swz} * 16 + md_gofs)"
                        )
                    self.emit(f"md_b_{an}_0, md_b_{an}_1 = {', '.join(words)}")
            an = 0
            while (
                b_words is None
                and not b_shuffle
                and not b_lanes
                and an < layout.atoms_n
            ):
                assert b_tile is not None
                pair = an + 1 < layout.atoms_n
                if dot.b.trans:
                    row = f"({16 * st} + {brow})"
                    if pair:
                        chunk = f"({bchunk} + {an})" if an else bchunk
                    else:
                        chunk = f"(({wni} * {wtn}) // 8 + {an})"
                    op = "ldmatrix_x4_trans" if pair else "ldmatrix_x2_trans"
                else:
                    if pair:
                        row = f"({brow} + {8 * an})" if an else brow
                    else:
                        row = f"({wni} * {wtn} + {8 * an} + md_l8)"
                    chunk = f"({2 * st} + {bchunk})"
                    op = "ldmatrix_x4" if pair else "ldmatrix_x2"
                if dot.b.transposed_tile:
                    swz = _transposed_swizzle(row, chunk, b_tile.cpr)
                else:
                    swz = _swizzled_chunk(row, chunk, b_tile.cpr)
                address = f"{b_base} + {row} * {b_tile.row_bytes} + {swz} * 16"
                if pair:
                    self.emit(
                        f"md_b_{an}_0, md_b_{an}_1, md_b_{an + 1}_0, md_b_{an + 1}_1 = {_RUNTIME}.{op}({address})"
                    )
                else:
                    self.emit(f"md_b_{an}_0, md_b_{an}_1 = {_RUNTIME}.{op}({address})")
                an += 2 if pair else 1
            if st == 0:
                self.flush_deferred()
            for am in range(layout.atoms_m):
                for an in range(layout.atoms_n):
                    acc = ", ".join(self.reg(dot.slot, am, an, j) for j in range(4))
                    self.emit(
                        f"{acc} = {mma}(md_a_{am}_0, md_a_{am}_1, md_a_{am}_2, md_a_{am}_3, "
                        f"md_b_{an}_0, md_b_{an}_1, {acc})"
                    )

    # -- the body -----------------------------------------------------------------

    def render(self) -> str:
        self.plan_tma()
        self.plan_tma_stores()
        for dot in self.program.dots:
            role = self.plan.bound_masks.get(dot.node)
            if role is not None:
                operand = dot.a if role == "a" else dot.b
                if self.plan.route(dot, operand) not in (
                    ROUTE_INPLACE,
                    ROUTE_CONVERT,
                    ROUTE_SYNC,
                ):
                    raise _decline(
                        "the operand masking the partial last trip of a"
                        " varying-bound loop takes no pass route"
                    )
        self.emit("md_tidx, _, _ = cute.arch.thread_idx()")
        self.name_depth["md_tidx"] = 0
        self.define("md_warp", "cute.arch.make_warp_uniform(cute.arch.warp_idx())")
        self.define("md_lane", "md_tidx % 32")
        self.define("md_g", "md_lane // 4")
        self.define("md_c", "md_lane % 4")
        self.define("md_l8", "md_lane % 8")
        self.define("md_lq", "md_lane // 8")
        cprs = {tile.cpr for tile in self.tiles.values()}
        cprs |= {tile.raw.cpr for tile in self.tiles.values() if tile.raw is not None}
        cprs |= {tile.cpr for tile in self.derived_tiles.values()}
        # The derived passes walk eight-element groups of their source rows.
        cprs |= {
            self.pass_grid(operand)[1] // 8
            for _dot, operand in self.plan.derived_pairs()
        }
        for cpr in sorted(cprs):
            if self.plan.threads % cpr:
                continue
            pr, pch, pz = f"md_pr{cpr}", f"md_pc{cpr}", f"md_pz{cpr}"
            self.define(pr, f"md_tidx // {cpr}")
            self.define(pch, f"md_tidx % {cpr}")
            self.define(pz, _swizzled_chunk(pr, pch, cpr))
            self.packet_names[cpr] = (pr, pch, pz)
        for dot in self.program.dots:
            for operand in (dot.a, dot.b):
                if self.plan.route(dot, operand) != ROUTE_SHUFFLE:
                    continue
                if operand.role == "b" and not operand.logical_swap:
                    # (k c, n g) sits in lane (c, g // 2), register g % 2.
                    if "t" not in self.quad_terms:
                        self.define("md_tsl", "md_c * 4 + md_g // 2")
                        self.define("md_todd", "md_g % 2")
                        self.quad_terms["t"] = ("md_tsl", "md_todd")
                elif "q" not in self.quad_terms:
                    # (row g, k c) sits in lane (g, c // 2), register c % 2.
                    self.define("md_qsl", "md_g * 4 + md_c // 2")
                    self.define("md_qodd", "md_c % 2")
                    self.quad_terms["q"] = ("md_qsl", "md_qodd")
        for layout in self.layouts.values():
            # The lane's tile-local fragment coordinates in this warp grid:
            # rows g / g + 8 of each atom row, columns 2c / 2c + 1 of each
            # atom column.
            self.define(f"md_L{layout.index}_wmi", f"md_warp // {layout.wn}")
            self.define(f"md_L{layout.index}_wni", f"md_warp % {layout.wn}")
            self.define(layout.rowl, f"md_L{layout.index}_wmi * {layout.wtm} + md_g")
            self.define(
                layout.coll, f"md_L{layout.index}_wni * {layout.wtn} + 2 * md_c"
            )
        self.define_axis_bases()
        if self.smem_bytes:
            # The TMA swizzle is address-based: its pattern's period (1024 B
            # at most) must divide every TMA slot's and staging buffer's
            # address.
            alignment = 1024 if self.tma_tx or self.store_tma else 128
            self.emit(
                f"md_smem = cute.arch.alloc_smem(cutlass.Uint8, {self.smem_bytes}, alignment={alignment})"
            )
            self.emit("md_s0 = cutlass.Int32(md_smem.toint())")
            for loop in sorted(self.tma_tx):
                self.emit(
                    f"md_full{loop} = cute.arch.alloc_smem(cutlass.Int64, {self.plan.stages}, alignment=8)"
                )
                self.define(self.tma_count(loop), "cutlass.Int32(0)")
            if self.tma_tx:
                with self.nest("if md_tidx == 0:"):
                    for loop in sorted(self.tma_tx):
                        for stage in range(self.plan.stages):
                            self.emit(
                                f"{_TC}.prims.mbarrier_init(md_full{loop} + {stage}, 1)"
                            )
                    self.emit(f"{_TC}.prims.fence_mbarrier_init()")
                    self.emit('cute.arch.fence_proxy(kind="async.shared", space="cta")')
                self.emit(self.sync)
            for tile in (
                *self.tiles.values(),
                *self.derived_tiles.values(),
                *self.staging.values(),
                *([self.scratch] if self.scratch is not None else []),
            ):
                for each in (tile, tile.raw):
                    if each is not None and each.name not in self._defined_tiles:
                        self._defined_tiles.add(each.name)
                        self.emit(f"{each.name} = md_s0 + {each.offset}")
            for index, ring in self.plan.aux_rings.items():
                self.emit(f"{self.aux_base(index)} = md_s0 + {ring.offset}")
        self.define_bases()
        # The L2 cache-policy words of the hinted tiles cp.async stages (the
        # asynchronously routed operands and the derived values' ring
        # tiles), one per policy: ``createpolicy`` once, every packet of the
        # tile names it (``emit_issue_tile``).
        used: set[str] = set()
        for dot in self.program.dots:
            for operand in (dot.a, dot.b):
                if operand.load is not None and self.plan.route(dot, operand) in (
                    ROUTE_DIRECT,
                    ROUTE_INPLACE,
                    ROUTE_CONVERT,
                ):
                    used.add(self.plan.load_hints.get(operand.load.node, ""))
        used.update(
            self.plan.load_hints.get(access.node, "")
            for _derived, access in self.plan.ring_derived()
        )
        for hint in sorted(used - {""}):
            self.policy_var(hint)
        self.define_iteration_bases(-1)
        self.emit_region(self.program.body, -1, set())
        return "\n".join(self.lines) + "\n"

    # -- regions -----------------------------------------------------------------

    def root_early_candidate(self, steps: Sequence[Step]) -> Loop | None:
        """The root's first loop when its ring prologue can go out ahead of
        the root steps before it: an innermost ring loop at more than one
        stage behind at least one root step, with no root tile overlaying
        the loops' frames (a raw source tile consumed ahead of the loops or
        an exchange tile read within one root segment: the prologue would
        rewrite it under the root's reads).  A root loop's bounds are
        per-CTA values of the preamble."""
        loops = [step for step in steps if isinstance(step, Loop)]
        if not loops or not steps or isinstance(steps[0], Loop):
            return None
        loop = loops[0]
        if (
            not loop.innermost
            or self.plan.stages < 2
            or not self.ring_items(loop.index)
            or self.overlaid_keys
            or self.overlaid_tiles
        ):
            return None
        return loop

    def emit_region(
        self, steps: Sequence[Step], region: int, written: set[tuple[object, ...]]
    ) -> None:
        """Emit the steps of region ``region`` (-1: root) in program order.

        The region's staged tiles (loads invariant in the loops nested here
        but varying with this region) are issued once, after the first
        loop's carry init or before the first dot; a *settle* (cp.async
        wait, per-element passes, exchange tiles of fragments defined
        earlier, CTA barrier) precedes the first step of each segment that
        reads them.  ``written`` tracks the exchange tiles valid here.
        """
        last_loop = max(
            (index for index, step in enumerate(steps) if isinstance(step, Loop)),
            default=-1,
        )
        early = self.root_early_candidate(steps) if region < 0 else None
        for position, step in enumerate(steps):
            if isinstance(step, Loop):
                self.emit_loop(step, region, written)
                # A loop whose frame a later sibling reuses must finish its
                # reads before the sibling's prologue rewrites the slots.
                if any(isinstance(later, Loop) for later in steps[position + 1 :]):
                    self.emit(self.sync)
                continue
            self.issue_staged(region)
            if early is not None and self.root_early_loop is None:
                # The first loop's ring prologue goes out behind the root's
                # staged group and ahead of the root dot's wait: the ring
                # tiles' latency overlaps the staged tiles' instead of
                # following the root MMAs (the settle below waits for all
                # but these younger groups).
                self.root_early_loop = early.index
                self.emit_ring_prologue(early, self.values)
            segment = list(
                itertools.takewhile(
                    lambda later: not isinstance(later, Loop), steps[position:]
                )
            )
            self.settle(region, segment, written)
            if position > last_loop >= 0:
                self.emit_early_prologue(region)
            self.emit_step(step, written)

    def emit_step(self, step: Step, written: set[tuple[object, ...]]) -> None:
        """A dot (exchange tiles of fragments computed since the last
        barrier right before it), derived value or store at its program
        point."""
        plan = self.plan
        if isinstance(step, Dot):
            pending = [
                operand
                for operand in (step.a, step.b)
                if plan.route(step, operand) == ROUTE_EXCHANGE
                and operand.chain_key not in written
            ]
            for operand in pending:
                written.add(operand.chain_key)
                self.emit_exchange(step, operand)
            if pending:
                self.emit(self.sync)
            self.emit_acc(step)
            self.emit_dot(step, "0")
            self.defined.add(step.node)
        elif isinstance(step, Derived):
            if step.node in plan.absorbed:
                return
            self.emit_derived(step)
            self.defined.add(step.node)
        elif isinstance(step, Store):
            self.emit_store(step)
        elif isinstance(step, Reduce):
            self.emit_reduce(step)
        elif isinstance(step, Scan):
            self.emit_scan(step)
            self.defined.add(step.node)
        elif isinstance(step, VecOp):
            self.emit_vec_op(step)
        elif isinstance(step, VecStore):
            self.emit_vec_store(step)
        else:
            raise AssertionError(step)

    def issue_staged(self, region: int) -> None:
        """cp.async the staged tiles of ``region`` (one group) if not yet
        done in this execution of the region."""
        if region in self.issued:
            return
        self.issued.add(region)
        staged = self.plan.staged_operands(
            region, (ROUTE_DIRECT, ROUTE_INPLACE, ROUTE_CONVERT)
        )
        # The epilogue values' loop-invariant tiles ride the root's group:
        # ahead of the ring prologues, they complete under the first ring
        # wait and the epilogue reads them from shared memory.
        fixed = self.plan.fixed_derived() if region < 0 else ()
        if not staged and not fixed:
            return
        for dot, operand in staged:
            self.emit_issue_operand(dot, operand)
        for _derived, access in fixed:
            self.emit_issue_tile(
                self.derived_tiles[access.node], access, 0, self.values, "0"
            )
        self.commit_group()
        self.staged_group_age = 0
        self.async_pending = True

    def commit_group(self) -> None:
        """Commit the cp.async group issued so far.  The staged group's age
        (younger groups behind it) is tracked through root-scope commits; a
        commit inside a loop body runs a dynamic number of times, so the age
        is unknown from there on (the settle then drains every group)."""
        self.emit("cute.arch.cp_async_commit_group()")
        if self.staged_group_age is not None:
            self.staged_group_age = (
                self.staged_group_age + 1 if not self.values else None
            )

    def consumers_of(self, steps: Sequence[Step]) -> list[Dot]:
        """The dots ``steps`` run: the dots among them, and every dot inside
        their loops."""
        result: list[Dot] = []
        for step in steps:
            if isinstance(step, Dot):
                result.append(step)
            elif isinstance(step, Loop):
                result.extend(
                    dot
                    for dot in self.program.dots
                    if self.inside(dot.loop, step.index)
                )
        return result

    def inside(self, region: int, loop: int) -> bool:
        """Whether region ``region`` is loop ``loop`` or nested in it."""
        return loop in self.enclosing(region)

    def exchange_ready(
        self, dot: Dot, operand: Operand, written: set[tuple[object, ...]]
    ) -> bool:
        """Whether the exchange tile of a fragment operand can be written
        now: the producer is defined, the tile not yet written, and it is
        not the dot's own loop's carry (written per iteration)."""
        assert operand.frag is not None
        if (
            self.plan.route(dot, operand) != ROUTE_EXCHANGE
            or operand.chain_key in written
            or self.program.early_exchange(dot, operand)
        ):
            return False
        return self.program.resolve(operand.frag) in self.defined

    def settle(
        self, region: int, steps: Sequence[Step], written: set[tuple[object, ...]]
    ) -> None:
        """Before a segment of steps (a loop, or the run of dots and stores
        up to the next loop): complete the cp.async groups in flight, run
        the per-element passes of the staged tiles the segment reads, write
        the exchange tiles its dots read of fragments defined so far, then a
        CTA barrier when anything happened."""
        plan = self.plan
        consumers = self.consumers_of(steps)
        # The fixed derived tiles the segment's own derived steps read
        # (issued with the root's staged tiles; read behind the drain and
        # the barrier like them).
        fixed = [
            access
            for step in steps
            if isinstance(step, Derived)
            for _derived, access in plan.fixed_derived()
            if access in step.loads and ("fixed", access.node) not in self.settled
        ]
        if not consumers and not fixed:
            return
        keys = {
            operand.chain_key
            for dot in consumers
            for operand in (dot.a, dot.b)
            if operand.load is not None
        }
        pending = [
            (dot, operand)
            for frame in [*self.enclosing(region), -1]
            for dot, operand in plan.staged_operands(frame)
            if operand.chain_key not in self.settled and operand.chain_key in keys
        ]
        exchanges = plan.unique(
            (dot, operand)
            for dot in consumers
            for operand in (dot.a, dot.b)
            if operand.kind == "frag" and self.exchange_ready(dot, operand, written)
        )
        if not pending and not exchanges and not fixed:
            return
        needs_barrier = False
        if (pending or fixed) and self.async_pending:
            # The staged group is complete once every group but the younger
            # ones (a root loop's ring prologue issued ahead of this wait)
            # is; groups complete in commit order.
            depth = self.staged_group_age or 0
            self.emit(f"cute.arch.cp_async_wait_group({depth})")
            if depth == 0:
                self.async_pending = False
            self.staged_group_age = None
            needs_barrier = True
        for access in fixed:
            self.settled.add(("fixed", access.node))
            needs_barrier = True
        passes = {
            ROUTE_SYNC: self.emit_sync_pass,
            ROUTE_INPLACE: self.emit_inplace_pass,
            ROUTE_CONVERT: self.emit_convert_pass,
        }
        for _dot, operand in pending:
            self.settled.add(operand.chain_key)
        for route, emit_pass in passes.items():
            for dot, operand in pending:
                if plan.route(dot, operand) != route:
                    continue
                emit_pass(dot, operand, "0")
                needs_barrier |= route == ROUTE_SYNC
        # (The exchange tiles written here never overlay a raw source tile
        # the passes read: the overlays are laid out apart, and loop-frame
        # exchange tiles are written inside their loops.)
        for dot, operand in exchanges:
            written.add(operand.chain_key)
            self.emit_exchange(dot, operand)
            needs_barrier = True
        if needs_barrier or pending:
            self.emit(self.sync)

    def bound_mask_end(self, dot: Dot, operand: Operand) -> str | None:
        """The loop bound ``operand``'s pass masks the dot's K positions
        against (``MmaDagPlan.bound_masks``), None for every other operand."""
        if self.plan.bound_masks.get(dot.node) != operand.role:
            return None
        return self.end_var(self.program.loops[dot.k.loop])

    def masked(self, dot: Dot, operand: Operand, coords: Coords, value: str) -> str:
        """``value`` (the operand element at ``coords`` of the operand value)
        zeroed where its K position is at or past the loop bound."""
        end = self.bound_mask_end(dot, operand)
        if end is None:
            return value
        position = operand.dims.index(dot.k)
        k = f"({self.axis_base(dot.k)} + {coords[position]})"
        # Both branches of the DSL's conditional expression must share a type.
        return CompileEnvironment.current().backend.where_expr(
            f"{k} < {end}", f"cutlass.Float32({value})", "cutlass.Float32(0.0)"
        )

    def ring_ordinal(
        self, loop: Loop, ring: list[tuple[_Tile, Access, int]]
    ) -> str | None:
        """A bounded ordinal only where a ring slot consumes the wide offset."""
        stages = self.plan.stages
        if not ring or stages <= 1 or not self.wide_loop_bounds(loop):
            return None
        if loop.index in self.tma_tx and not (
            stages > 2 and any(tile.is_raw for tile, _load, _pos in ring)
        ):
            # TMA already has its own bounded phase count. Only its shorter
            # raw ring needs a separate ordinal relative to this loop entry.
            return None
        return f"md_ring{loop.index}"

    def emit_loop(
        self, loop: Loop, region: int, written: set[tuple[object, ...]]
    ) -> None:
        """Carry init, the enclosing region's staged tiles, a settle for
        whatever the body reads of the enclosing regions, then the loop."""
        self.emit_carry_init(loop)
        self.emit_vec_init(loop)
        for carry in loop.carries:
            self.defined.add(carry.placeholder)
        self.issue_staged(region)
        self.settle(region, [loop], written)
        begin, end, trip_static = self.loop_bounds(loop)
        var = self.loop_index_var(loop.index)
        body_written = set(written)
        # The loops' frames overlay the raw source tiles the root frame's
        # early passes consumed: those passes ran (behind their barrier).
        if not self.overlaid_keys <= self.settled:
            raise _decline("an overlaid raw source tile is live at a loop")
        stale = written & self.overlaid_tiles
        if stale:
            # They overlay root exchange tiles read before this loop too:
            # every lane's reads first, and the tiles are not valid past the
            # prologue.
            self.emit(self.sync)
            written -= stale
            body_written -= stale
        self.hoists.append(
            _Hoist(
                len(self.lines),
                self.indent,
                len(self.values),
                loop.index,
                tuple(self.scope_path),
            )
        )
        if loop.innermost:
            ring = self.ring_items(loop.index)
            ordinal = self.ring_ordinal(loop, ring)
            if ordinal is not None:
                self.emit(f"{ordinal} = cutlass.Int32(0)")
            if ring and self.plan.stages > 1:
                if (
                    loop.index in self.early_prologues.values()
                    or loop.index == self.root_early_loop
                ):
                    # The enclosing loop issued this prologue ahead of its
                    # iteration (peeled before its ``for`` / in the previous
                    # iteration's epilogue).
                    self.async_pending = True
                else:
                    self.emit_ring_prologue(loop, self.values)
            elif ring and self.has_aux_shared(loop.index):
                # One stage: the first iteration's side tile goes out ahead
                # of the loop in a group of its own.
                self.emit_aux_shared_ahead(loop, self.values, 0)
                self.commit_group()
                self.async_pending = True
            # The ``for`` line: the body's loop-invariant side loads and
            # redirected values are hoisted ahead of the prologue while it
            # renders, so ``insert_line`` keeps the entry current.
            entry = _Hoist(len(self.lines), self.indent, len(self.values), loop.index)
            self.loop_entries.append(entry)
            with self.nest(f"for {var} in cutlass.range({begin}, {end}):"):
                self.values[loop.index] = var
                self.emit_iteration(loop, begin, end, ring, body_written)
                if ordinal is not None:
                    self.advance_ring_counter(
                        ordinal,
                        math.lcm(self.plan.stages, self.plan.stages - 1),
                    )
                if loop.index in self.tma_tx:
                    # Int32 TMA phase count; empty loops consume no slots,
                    # and the potentially Int64 induction is never narrowed.
                    count = self.tma_count(loop.index)
                    if self.wide_loop_bounds(loop):
                        self.advance_ring_counter(count, 2 * self.plan.stages)
                    else:
                        # Preserve the existing static/Int32 loop instruction path.
                        self.emit(f"{count} = {count} + 1")
                del self.values[loop.index]
            self.aux_shared_stage.pop(loop.index, None)
            assert self.loop_entries.pop() is entry
            assert self.lines[entry.position].lstrip().startswith(f"for {var} in ")
            if any(
                store.access.node in self.store_tma
                for store in self.plan.staged_stores
                if store.loop == loop.index
            ):
                # The last iteration's bulk copies still read their staging
                # buffer: retired ahead of a barrier, so whatever writes the
                # loop's frame next (a re-entry's snapshot, a sibling loop's
                # prologue, a root exchange tile) comes after.
                with self.nest("if md_tidx == 0:"):
                    self.emit(f"{_TC}.prims.cp_async_bulk_wait_group(0, read=True)")
                self.emit(self.sync)
            if ring and loop.index in self.aux_shared_early:
                # The side tile is read ahead of the iteration barrier (the
                # carry rescale's scalar at the top, the pass's gate row):
                # the first iteration's tile rode the first group, so it is
                # landed, and published to every lane, before the loop
                # (iteration ``i``'s wait and barrier do the same for the
                # side tile of ``i + 1``).  Read behind the barrier alone,
                # the tile needs nothing ahead of the loop: the first
                # iteration's wait and barrier publish the tiles of
                # iterations 0 and 1 together, and the group's operand
                # packets keep landing under the iteration's early work.
                for line in (
                    f"cute.arch.cp_async_wait_group({max(self.plan.stages - 2, 0)})",
                    self.sync,
                ):
                    self.insert_line(entry, line)
        else:
            nested = self.early_prologues.get(loop.index)
            if nested is not None and trip_static != 0:
                # The nested ring's prologue for this loop's first iteration
                # (each iteration's epilogue issues the next one's).  A loop
                # of no iterations issues nothing; one whose trip count is
                # dynamic guards it (the commits sit inside the guard and no
                # wait precedes the loop, so a false guard is safe).
                inner = self.program.loops[nested]
                values = {**self.values, loop.index: begin}
                if trip_static is None:
                    with self.nest(f"if {begin} < {end}:"):
                        self.emit_ring_prologue(inner, values)
                else:
                    self.emit_ring_prologue(inner, values)
            with self.nest(f"for {var} in cutlass.range({begin}, {end}):"):
                self.values[loop.index] = var
                self.emit_outer_iteration(loop, body_written)
                del self.values[loop.index]
        self.hoists.pop()
        self.fill_aux_anchors(loop.index)
        self.forget_iteration_bases(loop.index)
        # Values memoized inside the loop are not in scope after it.
        self.evaluator = _Evaluator(self)
        for carry in loop.carries:
            self.defined.add(carry.phi)

    def emit_ring_prologue(self, loop: Loop, outer: dict[int, str]) -> None:
        """Prologue of ``loop``'s operand ring at the enclosing loops' values
        ``outer``: the first ``stages - 1`` iterations' tiles, one commit
        group each (an iteration past the end commits an empty group so the
        wait depth stays uniform); iteration ``ahead`` takes ring and raw
        slot ``ahead`` (below the raw ring's depth of ``stages - 1``).  The
        aux ring's side tiles ride one group ahead of their operand tiles:
        group ``ahead`` carries iteration ``ahead + 1``'s (and the first
        group iteration ``begin``'s, waited for before the loop)."""
        ring = self.ring_items(loop.index)
        begin, end, trip_static = self.loop_bounds(loop)
        stages = self.plan.stages
        tma = loop.index in self.tma_tx
        for ahead in range(stages - 1):
            value = f"({begin} + {ahead})" if ahead else begin
            values = {**outer, loop.index: value}
            # A TMA ring's slots follow the loop's running count (a nested
            # ring re-entered by its outer loop keeps its barrier phases).
            slot = (
                f"(({self.tma_count(loop.index)} + {ahead}) % {stages})"
                if tma
                else str(ahead)
            )
            if trip_static is not None and ahead < trip_static:
                self.emit_issue(ring, values, slot, str(ahead))
                self.aux_anchor(loop.index, values, slot)
            elif trip_static is None:
                with self.nest(f"if {value} < {end}:"):
                    self.emit_issue(ring, values, slot, str(ahead))
                    self.aux_anchor(loop.index, values, slot)
            if ahead == 0:
                self.emit_aux_shared_ahead(loop, outer, 0)
            self.emit_aux_shared_ahead(loop, outer, ahead + 1)
            self.commit_group()
        self.async_pending = True

    def emit_early_prologue(self, region: int) -> None:
        """In the epilogue of region ``region`` (a loop): the nested ring's
        prologue for the next iteration, behind a barrier (every lane is
        past the nested loop's reads of the frame the prologue rewrites),
        so its loads land while the epilogue runs."""
        nested = self.early_prologues.get(region)
        if nested is None or region in self.early_issued:
            return
        self.early_issued.add(region)
        loop = self.program.loops[region]
        _begin, end, _trip = self.loop_bounds(loop)
        var = self.loop_index_var(region)
        if not self.lines or self.lines[-1].strip() != self.sync:
            self.emit(self.sync)
        with self.nest(f"if {var} + 1 < {end}:"):
            self.emit_ring_prologue(
                self.program.loops[nested], {**self.values, region: f"({var} + 1)"}
            )
        # The nested loop's body is rendered (its aux sites and slots are
        # known), so this prologue's aux copies go in now: the loop's own
        # fill ran at the end of its ``emit_loop``, before this point.
        self.fill_aux_anchors(nested)

    def enter_iteration(self, loop: Loop) -> None:
        """The per-iteration preamble: the tile axes' bases and the bases of
        the accesses varying with this loop."""
        var = self.loop_index_var(loop.index)
        for axis in loop.axes:
            offset = self.axis_offset(axis, var)
            self.emit(f"{self.axis_base(axis)} = {self.axis_product(axis, offset)}")
        self.forget_iteration_bases(loop.index)

    def emit_outer_iteration(
        self, loop: Loop, written: set[tuple[object, ...]]
    ) -> None:
        """An iteration of a loop with nested loops: its own tiles are
        issued once per iteration into single slots (they land during the
        nested loops), its carry snapshots stream out, the body runs in
        program order and a trailing barrier keeps the next iteration's
        rewrites -- its own tiles, the nested loops' ring prologues and
        exchange tiles -- behind every lane's reads of this one."""
        program = self.program
        self.enter_iteration(loop)
        self.define_iteration_bases(loop.index)
        self.evaluator = _Evaluator(self)
        self.issued.discard(loop.index)
        self.early_issued.discard(loop.index)
        for store in program.stores:
            if store.loop == loop.index and store.source == "carry_in":
                self.emit_store(store)
        self.emit_region(loop.steps, loop.index, written)
        self.emit_carry_copies(loop)
        self.emit(self.sync)

    def emit_iteration(
        self,
        loop: Loop,
        begin: str,
        end: str,
        ring: list[tuple[_Tile, Access, int]],
        written: set[tuple[object, ...]],
    ) -> None:
        plan, program = self.plan, self.program
        index = loop.index
        var = self.loop_index_var(index)
        stages = plan.stages
        stage = self.stage_var(index)
        parity = self.parity_var(index)
        steps = loop.steps
        self.enter_iteration(loop)
        self.refilled = set()
        self.aux_waited = False
        self.aux_synced = False
        self.raw_stages[index] = "0"
        offset = self.ring_ordinal(loop, ring) or (
            f"{var}" if begin == "0" else f"({var} - {begin})"
        )
        if index in self.tma_tx:
            # Slot and barrier phase of this iteration from the loop's
            # running count (see ``emit_ring_prologue``).
            count = self.tma_count(index)
            self.emit(f"{stage} = {count} % {stages}")
            self.tma_parity[index] = self.bind("md_par", f"({count} // {stages}) & 1")
        elif ring and stages > 1:
            self.emit(f"{stage} = {offset} % {stages}")
        else:
            self.emit(f"{stage} = 0")
        if ring and stages > 2 and any(tile.is_raw for tile, _load, _pos in ring):
            # The raw source tiles ride a ring one slot shorter (the
            # refill past the barrier takes this iteration's slot over).
            self.raw_stages[index] = self.bind("md_rstage", f"{offset} % {stages - 1}")
        staged_stores = [store for store in plan.staged_stores if store.loop == index]
        if staged_stores or parity in self.tile_buffers.values():
            self.emit(f"{parity} = {var} % 2")
        if ring and self.has_aux_shared(index):
            self.aux_shared_stage[index] = self.bind(
                "md_sstage", self.aux_shared_slot(loop, var)
            )
        self.define_iteration_bases(index)
        self.evaluator = _Evaluator(self)
        if ring and stages == 1:
            self.emit_issue(ring, self.values, "0")
            self.aux_anchor(index, self.values, "0")
            if self.has_aux_shared(index):
                # The next iteration's side tile, landed with this group and
                # published by the barrier below.
                with self.nest(f"if {var} + 1 < {end}:"):
                    self.emit_aux_shared(loop, {**self.values, index: f"({var} + 1)"})
            self.commit_group()
            self.async_pending = True
        # Work that needs no shared data overlaps the copies in flight.
        for dot, operand in plan.ring_pairs(index, (ROUTE_SYNC,)):
            self.emit_sync_pass(dot, operand, stage)
        for store in program.stores:
            if store.loop == index and store.source == "carry_in":
                if store in plan.staged_stores:
                    self.emit_stage_in(store, parity)
                else:
                    self.emit_store(store)
        # Accumulators that depend on nothing computed in the iteration
        # (zeros, the carry, a rescaled carry no earlier step reads).
        hoisted = {dot.node for dot in self.hoistable_accs(loop)}
        for step in steps:
            if isinstance(step, Dot) and step.node in hoisted:
                self.emit_acc(step)
        # Exchange tiles of the carry at loop entry are written here, so the
        # iteration barrier below orders them before their ldmatrix reads.
        early = [
            (dot, operand)
            for dot, operand in plan.unique(
                (
                    (dot, operand)
                    for dot in program.dots
                    if dot.loop == index
                    for operand in (dot.a, dot.b)
                ),
                (ROUTE_EXCHANGE,),
            )
            if program.early_exchange(dot, operand)
            # (A chain reading a row of this iteration's ring slot waits for
            # the slot: its exchange runs in the body, behind its own barrier.)
            and not any(a.node in self.ring_aliases for a in operand.aux_loads)
        ]
        for dot, operand in early:
            self.emit_exchange(dot, operand)
        if ring:
            # The groups committed so far are the prologue's ``stages - 1``
            # plus one per earlier iteration, so this iteration's group is
            # followed by ``stages - 2`` younger ones (the iterations ahead;
            # the refill issued below comes after the wait): wait for all but
            # those and this iteration's tiles are in shared memory.
            if not self.aux_waited:
                self.emit(f"cute.arch.cp_async_wait_group({max(stages - 2, 0)})")
                self.aux_waited = True
            if index in self.tma_tx:
                # This iteration's TMA tiles landed (every thread waits; the
                # barrier after the passes orders the ldmatrix reads).
                self.emit(
                    f"{_TC}.mbar_wait(md_full{index} + ({stage}), {self.tma_parity[index]})"
                )
            aux = plan.aux_rings.get(index)
            if aux is not None and aux.runs:
                # The aux ring's vectors were copied by the first threads:
                # publish them to every lane's pass (outside the packet
                # guards the pass bodies may sit under).
                self.emit(self.sync)
                self.aux_synced = True
        if index in self.tma_tx and plan.ring_pairs(
            index, (ROUTE_INPLACE, ROUTE_CONVERT)
        ):
            # The passes' generic writes of a slot a bulk tensor copy refills
            # ``stages - 1`` iterations on: every writing thread fences the
            # async proxy before the barrier that precedes the refill (the
            # documented pattern; the issuing thread's own fence after the
            # barrier covers only its own accesses).  The fence sits at the
            # top of the next iteration, past its waits, where the previous
            # pass's stores are long complete: right behind the stores it
            # cost chunk_scan s1 3 % (3.96 -> 4.09 us).
            self.emit('cute.arch.fence_proxy(kind="async.shared", space="cta")')
        for dot, operand in plan.ring_pairs(index, (ROUTE_INPLACE,)):
            self.emit_inplace_pass(dot, operand, stage)
        for dot, operand in plan.ring_pairs(index, (ROUTE_CONVERT,)):
            self.emit_convert_pass(dot, operand, stage)
        self.emit_bound_tail_zero(index, stage)
        tma_stores = [s for s in staged_stores if s.access.node in self.store_tma]
        if tma_stores:
            # The snapshots' generic-proxy writes, fenced by every writing
            # thread ahead of the barrier the store thread's bulk copies
            # follow; the store thread first retires the previous
            # iteration's copies -- their reads of the other buffer, which
            # the next iteration's snapshot rewrites past this barrier.
            self.emit('cute.arch.fence_proxy(kind="async.shared", space="cta")')
            with self.nest("if md_tidx == 0:"):
                self.emit(f"{_TC}.prims.cp_async_bulk_wait_group(0, read=True)")
        self.emit(self.sync)
        # Every lane's aux copies of this stage are landed and published
        # (the next iteration's side tile with them).
        self.aux_synced = True
        # The next iteration's side loads of the passes and the carry
        # rescale land in their registers while the MMAs run (issued after
        # the MMAs instead, fwd_h measured 2.78 / 4.23 us against 2.62 /
        # 3.76 at s1 / s2 shapes).
        self.emit_prefetch_refills(loop, var, end)
        if ring:
            # This iteration's slot is landed and published: the vector
            # loads aliasing its rows read it (``_ring_aliases``).
            self.ring_ready.add(index)
        if ring and stages > 1:
            # Past the barrier every lane finished the previous iteration's
            # reads of the slot refilled here (``stages - 1`` iterations ahead
            # is the slot read one iteration ago); an iteration past the end
            # commits an empty group so the wait depth stays uniform.
            ahead = stages - 1
            with self.nest(f"if {var} + {ahead} < {end}:"):
                # Iteration ``var + ahead``'s raw slot is this iteration's
                # (``ahead`` is the raw ring's depth), consumed above.
                self.emit_issue(
                    ring,
                    {**self.values, index: f"({var} + {ahead})"},
                    f"({stage} + {ahead}) % {stages}",
                    self.raw_stages[index],
                )
                # The refill's aux copies are emitted here, so a site met
                # later in the body could not be refilled: allocation closes.
                for line in self.aux_issue_lines(
                    index,
                    {**self.values, index: f"({var} + {ahead})"},
                    f"({stage} + {ahead}) % {stages}",
                ):
                    self.emit(line)
                self.aux_closed.add(index)
            if self.has_aux_shared(index):
                # The side tile one iteration further: waited for at
                # iteration ``var + ahead``, published by its barrier, read
                # at ``var + ahead + 1`` (its slot was last read at ``var +
                # ahead - stages``, two barriers ago).
                with self.nest(f"if {var} + {ahead + 1} < {end}:"):
                    self.emit_aux_shared(
                        loop, {**self.values, index: f"({var} + {ahead + 1})"}
                    )
            self.commit_group()
        derived_passes = plan.derived_pairs(index)
        if derived_passes:
            # The derived passes run with the refill in flight (they read
            # the derived tiles of this iteration's slot and write their own
            # operand slot, which the refill never touches) and publish
            # behind a barrier of their own -- where the exchange route
            # computed the fragment and wrote its tile.
            for dot, operand in derived_passes:
                self.emit_derived_pass(dot, operand, stage)
            self.emit(self.sync)
        # This iteration's snapshots (written into the staging buffer before
        # the barrier) stream out: the bulk tensor copies from the store
        # thread, the others as 16-byte packets from every thread; the next
        # iteration writes the other buffer.
        if tma_stores:
            self.emit_tma_stores(tma_stores, parity)
        for store in staged_stores:
            if store.access.node not in self.store_tma:
                self.emit_stage_out(store, parity)
        # The body in program order: a dot's accumulator (unless hoisted) and
        # the exchange tiles of fragments computed in this iteration (behind
        # their own barrier) right before it, derived values register by
        # register, stores at their program point.
        written.update(operand.chain_key for _d, operand in early)
        absorbed = plan.absorbed
        for step in steps:
            if isinstance(step, Dot):
                pending = [
                    operand
                    for operand in (step.a, step.b)
                    if plan.route(step, operand) == ROUTE_EXCHANGE
                    and operand.chain_key not in written
                ]
                for operand in pending:
                    written.add(operand.chain_key)
                    self.emit_exchange(step, operand)
                if pending:
                    self.emit(self.sync)
                if step.node not in hoisted:
                    self.emit_acc(step)
                self.emit_dot(step, stage)
                self.defined.add(step.node)
            elif isinstance(step, Derived):
                if step.node in absorbed:
                    # Its value went into the pass-routed operand tile.
                    continue
                self.emit_derived(step)
                self.defined.add(step.node)
            elif isinstance(step, Store):
                if step.source == "body":
                    self.emit_store(step)
            elif isinstance(step, Reduce):
                self.emit_reduce(step)
            elif isinstance(step, Scan):
                self.emit_scan(step)
                self.defined.add(step.node)
            elif isinstance(step, VecOp):
                self.emit_vec_op(step)
            elif isinstance(step, VecStore):
                self.emit_vec_store(step)
            else:
                raise AssertionError(step)
        # The first dot flushed the snapshot stores behind its fragment loads.
        assert not self.deferred
        # Sites first met after the passes (derived values, epilogue-side
        # reads of the body) refill here.
        self.emit_prefetch_refills(loop, var, end)
        self.emit_carry_copies(loop)
        self.ring_ready.discard(index)
        if not (ring and stages > 1):
            # Every lane finished reading the slot the next iteration refills.
            self.emit(self.sync)

    def emit_carry_copies(self, loop: Loop) -> None:
        """Carry outputs computed in another slot (a no-acc dot over the
        carry itself) move into the carry's registers for the next
        iteration."""
        program = self.program
        fragment_copies: list[tuple[str, str]] = []
        for carry in loop.carries:
            out = program.resolve(carry.output)
            if out is carry.placeholder:
                continue
            source = program.frag_slots[out]
            if source == carry.slot:
                continue
            layout = self.slot_layouts[carry.slot]
            for am, an, j in layout.registers():
                value = self.bind("md_cnext", self.reg(source, am, an, j))
                fragment_copies.append((self.reg(carry.slot, am, an, j), value))
        for target, value in fragment_copies:
            self.emit(f"{target} = {value}")

        # Read all outputs before overwriting any carry: vector results may
        # exchange slots, and the assignments must happen simultaneously.
        vector_copies: list[tuple[str, str]] = []
        for carry in loop.vec_carries:
            if carry.identity:
                continue
            source = program.vec_leaves[program.resolve(carry.output)]
            if source != carry.slot:
                for atom, half in self.vec_regs(carry.slot):
                    value = self.bind("md_vnext", self.vreg(source, atom, half))
                    vector_copies.append((self.vreg(carry.slot, atom, half), value))
        for target, value in vector_copies:
            self.emit(f"{target} = {value}")

    def hoistable_accs(self, loop: Loop) -> list[Dot]:
        """The loop dots whose accumulator may be materialized at the
        iteration top: zeros, or a value of the carry at loop entry whose
        slot no earlier step reads (a carry used as an operand of an earlier
        dot must be read before its rescale overwrites the registers)."""
        program = self.program
        result = []
        steps = loop.steps
        entry_vectors = {carry.placeholder for carry in loop.vec_carries}
        for index, step in enumerate(steps):
            if not isinstance(step, Dot):
                continue
            if step.acc is None:
                result.append(step)
                continue
            if step.acc_frag not in loop.carry_placeholders:
                continue
            if any(
                program.frag_regions.get(vec, -1) == loop.index
                and vec not in entry_vectors
                for vec in step.acc_vecs
            ):
                # A vector computed this iteration (e.g. an online softmax
                # rescale) is unavailable at loop entry, even when its slot
                # reuses a carry's registers.
                continue
            if any(
                access.node in self.ring_aliases
                for access in program.acc_loads.get(step.node, ())
            ):
                # The rescale reads a row of this iteration's ring slot
                # (``_ring_aliases``): it runs behind the iteration barrier,
                # at the dot.
                continue
            slot = step.slot
            reads_slot = False
            for earlier in steps[:index]:
                if isinstance(earlier, Dot):
                    leaves = [o.frag for o in (earlier.a, earlier.b)] + [
                        earlier.acc_frag
                    ]
                elif isinstance(earlier, Derived):
                    leaves = list(earlier.frags)
                elif isinstance(earlier, Store):
                    if earlier.source == "carry_in":
                        # Snapshots are hoisted above every accumulator.
                        continue
                    leaves = [earlier.frag]
                elif isinstance(earlier, Reduce):
                    leaves = [earlier.frag] if earlier.frag is not None else []
                elif isinstance(earlier, Scan):
                    leaves = [earlier.frag]
                elif isinstance(earlier, (VecOp, VecStore)):
                    continue
                else:
                    raise AssertionError(earlier)
                if any(
                    leaf is not None
                    and program.frag_slots[program.resolve(leaf)] == slot
                    for leaf in leaves
                ):
                    reads_slot = True
                    break
            if not reads_slot:
                result.append(step)
        return result

    def vec_coords(self, node: Node, slot: int, atom: int, half: int) -> Coords:
        """Positional coordinates of vector value ``node`` for register
        ``(atom, half)``: the vector coordinate on its one tile dim, zero
        elsewhere (a scalar has none)."""
        shape = _shape(node)
        coords = ["0"] * len(shape)
        if self.vec_layouts[slot] is None:
            return tuple(coords)
        positions = [
            i
            for i, size in enumerate(shape)
            if not (isinstance(size, int) and size == 1)
            and not self._is_lead_dim(node, i)
        ]
        if len(positions) != 1:
            raise _decline(f"{node.name} is not a vector value")
        coords[positions[0]] = self.vec_index(slot, atom, half)
        return tuple(coords)

    def emit_vec_init(self, loop: Loop) -> None:
        """Fill the loop's vector carries, or copy an earlier loop's vector
        carried on into a different slot."""
        for carry in loop.vec_carries:
            if carry.identity:
                continue
            if carry.init_vec is not None:
                source = self.program.vec_leaves[carry.init_vec]
                if source != carry.slot:
                    for atom, half in self.vec_regs(carry.slot):
                        self.emit(
                            f"{self.vreg(carry.slot, atom, half)} = {self.vreg(source, atom, half)}"
                        )
                continue
            for atom, half in self.vec_regs(carry.slot):
                self.emit(
                    f"{self.vreg(carry.slot, atom, half)} = "
                    f"cutlass.Float32({_float_literal(carry.init_fill)})"
                )

    def emit_reduce(self, step: Reduce) -> None:
        """Reductions of a pointwise value over a fragment along its columns (row
        vector) or rows (column vector), or of a vector into a scalar.

        Each lane reduces its own registers, the shuffles combine the lanes of the
        warp sharing the result (the quad, xor 1 / 2, for rows; the octet,
        xor 4 / 8 / 16, for columns) and the warps along the other split
        meet in the scratch tile (one vector per warp, combined in a fixed
        order: launches are bit-identical)."""
        program = self.program
        if step.vec is not None:
            self.emit_scalar_reduce(step)
            return
        assert step.frag is not None
        frag_slot = program.frag_slots[step.frag]
        layout = self.slot_layouts[frag_slot]
        rows = step.orient == "rows"
        self.hoist_scalars(step.value)
        self.evaluator.enter_frag(frag_slot)
        values: dict[tuple[int, int, int], str] = {}
        for am, an, j in layout.registers():
            self.evaluator.reg = (am, an, j)
            coords = self.frag_coords(step.value, layout, am, an, j)
            values[(am, an, j)] = self.evaluator.evaluate(step.value, coords)
        self.evaluator.leave_frag()
        partials: dict[tuple[int, int], str] = {}
        for atom, half in self.vec_regs(step.slot):
            if rows:
                terms = [
                    values[(atom, an, 2 * half + bit)]
                    for an in range(layout.atoms_n)
                    for bit in range(2)
                ]
            else:
                terms = [
                    values[(am, atom, 2 * row_half + half)]
                    for am in range(layout.atoms_m)
                    for row_half in range(2)
                ]
            total = terms[0]
            for term in terms[1:]:
                total = self.bind("md_s", _reduce_combine(step.kind, total, term))
            shuffle = f"{'lane_group' if rows else 'octet'}_{step.kind}"
            args = f"{total}, 4" if rows else total
            partials[(atom, half)] = self.bind("md_s", f"{_RUNTIME}.{shuffle}({args})")
        other = layout.wn if rows else layout.wm
        if other > 1:
            base = self.scratch_base()
            extent = self.plan.size(cast("Axis", program.vec_slots[step.slot].axis))
            warp = f"md_L{layout.index}_{'wni' if rows else 'wmi'}"
            with self.nest(f"if {'md_c' if rows else 'md_g'} == 0:"):
                for (atom, half), partial in partials.items():
                    index = self.vec_index(step.slot, atom, half)
                    self.emit(
                        f"{_RUNTIME}.sts_f32({base} + ({warp} * {extent} + {index}) * 4, {partial})"
                    )
            self.emit(self.sync)
            for atom, half in partials:
                index = self.vec_index(step.slot, atom, half)
                total = None
                for column in range(other):
                    value = self.bind(
                        "md_s",
                        f"{_RUNTIME}.lds_f32({base} + ({column * extent} + {index}) * 4)",
                    )
                    total = (
                        value
                        if total is None
                        else self.bind("md_s", _reduce_combine(step.kind, total, value))
                    )
                assert total is not None
                partials[(atom, half)] = total
            # The scratch is reused by the next vector operation: every lane
            # read.
            self.emit(self.sync)
        for (atom, half), partial in partials.items():
            self.emit(f"{self.vreg(step.slot, atom, half)} = {partial}")

    def emit_scalar_reduce(self, step: Reduce) -> None:
        """A vector reduced into a scalar: each lane combines its registers, the
        lanes of the warp holding distinct elements (the octet for a row
        vector, the quad for a column vector) combine theirs, and the warps
        along the vector's split meet in the scratch tile."""
        program = self.program
        assert step.vec is not None
        source = program.vec_leaves[step.vec]
        layout = self.vec_layouts[source]
        assert layout is not None
        rows = program.vec_slots[source].orient == "rows"
        # Preserve the frontend's padding mask on the reduction input. The
        # source registers can hold non-identity values outside a ragged tile.
        regs = []
        for atom, half in self.vec_regs(source):
            self.evaluator.vec_reg = (atom, half)
            coords = self.vec_coords(step.value, source, atom, half)
            regs.append(self.evaluator.evaluate(step.value, coords))
        self.evaluator.vec_reg = None
        total = regs[0]
        for term in regs[1:]:
            total = self.bind("md_s", _reduce_combine(step.kind, total, term))
        shuffle = f"{'octet' if rows else 'lane_group'}_{step.kind}"
        args = total if rows else f"{total}, 4"
        total = self.bind("md_s", f"{_RUNTIME}.{shuffle}({args})")
        split = layout.wm if rows else layout.wn
        if split > 1:
            base = self.scratch_base()
            warp = f"md_L{layout.index}_{'wmi' if rows else 'wni'}"
            other = f"md_L{layout.index}_{'wni' if rows else 'wmi'}"
            with self.nest(f"if md_lane == 0 and {other} == 0:"):
                self.emit(f"{_RUNTIME}.sts_f32({base} + {warp} * 4, {total})")
            self.emit(self.sync)
            total = None
            for index in range(split):
                value = self.bind("md_s", f"{_RUNTIME}.lds_f32({base} + {index * 4})")
                total = (
                    value
                    if total is None
                    else self.bind("md_s", _reduce_combine(step.kind, total, value))
                )
            self.emit(self.sync)
        self.emit(f"{self.vreg(step.slot, 0, 0)} = {total}")

    def emit_scan(self, step: Scan) -> None:
        """An inclusive prefix sum of a pointwise value over a fragment down
        (``reverse``: up) every column, into the scan's slot.

        In the mma.sync C layout a column's rows sit in the octet of lanes
        sharing ``c`` (row ``g`` of each half of each m atom), so per
        column register: the octet scans its eight rows with three
        ``shfl.up`` / ``shfl.down`` adds (the lanes past the edge keep
        their value), the halves and m atoms chain in row order with the
        segment total broadcast from the edge lane, and the warps along M
        (``wm > 1``) exchange their column totals through the scratch tile,
        each adding the totals of the warps before (after) it in a fixed
        order.  The input registers are all read before the slot is
        written, so the scan may take over its input's slot."""
        layout = self.slot_layouts[step.slot]
        if step.orient != "rows":
            raise _decline("a scan along the fragment columns")
        reverse = step.reverse
        self.hoist_scalars(step.value)
        # The value is evaluated over the input fragment's registers (the
        # scan's slot has the same shape); the vectors it reads beside the
        # fragment resolve at each register's row / column (``frag_vec``).
        self.evaluator.enter_frag(self.program.frag_slots[step.frag])
        values: dict[tuple[int, int, int], str] = {}
        for am, an, j in layout.registers():
            self.evaluator.reg = (am, an, j)
            coords = self.frag_coords(step.value, layout, am, an, j)
            values[(am, an, j)] = self.evaluator.evaluate(step.value, coords)
        self.evaluator.leave_frag()
        # Row segments of the warp tile in scan order: (m atom, half).
        segments = [(am, half) for am in range(layout.atoms_m) for half in range(2)]
        if reverse:
            segments.reverse()
        shuffle = f"{_RUNTIME}.{'shfl_down_add_f32' if reverse else 'shfl_up_add_f32'}"
        # The lane holding a segment's last scanned row: g == 7 (reverse:
        # g == 0) of the column's quad.
        edge = "md_c" if reverse else "28 + md_c"
        results: dict[tuple[int, int, int], str] = {}
        totals: dict[tuple[int, int], str] = {}
        for an in range(layout.atoms_n):
            for bit in range(2):
                carry: str | None = None
                for am, half in segments:
                    value = values[(am, an, 2 * half + bit)]
                    for offset in (1, 2, 4):
                        value = self.bind("md_sc", f"{shuffle}({value}, {4 * offset})")
                    if carry is not None:
                        value = self.bind("md_sc", f"{value} + {carry}")
                    results[(am, an, 2 * half + bit)] = value
                    carry = self.bind(
                        "md_sc",
                        f"cute.arch.shuffle_sync({value}, {edge}, mask=0xffffffff, mask_and_clamp=31)",
                    )
                assert carry is not None
                totals[(an, bit)] = carry
        if layout.wm > 1:
            # Every warp's column totals, then the sum of the warps before
            # (after) this one in a fixed order.
            base = self.scratch_base()
            cols = layout.cols
            warp = f"md_L{layout.index}_wmi"
            with self.nest("if md_g == 0:"):
                for (an, bit), total in totals.items():
                    self.emit(
                        f"{_RUNTIME}.sts_f32({base} + ({warp} * {cols} + {layout.col(an, bit)}) * 4, {total})"
                    )
            self.emit(self.sync)
            for an in range(layout.atoms_n):
                for bit in range(2):
                    prefix = self.bind("md_sc", "cutlass.Float32(0.0)")
                    for other in range(layout.wm):
                        condition = (
                            f"{warp} < {other}" if reverse else f"{warp} > {other}"
                        )
                        with self.nest(f"if {condition}:"):
                            self.emit(
                                f"{prefix} = {prefix} + {_RUNTIME}.lds_f32({base} + ({other * cols} + {layout.col(an, bit)}) * 4)"
                            )
                    for am in range(layout.atoms_m):
                        for half in range(2):
                            key = (am, an, 2 * half + bit)
                            results[key] = self.bind(
                                "md_sc", f"{results[key]} + {prefix}"
                            )
            # The scratch is reused by the next vector operation.
            self.emit(self.sync)
        for (am, an, j), value in results.items():
            self.emit(f"{self.reg(step.slot, am, an, j)} = {value}")

    def emit_relayout(self, leaf: Node, target: int) -> str:
        """Bring vector ``leaf`` (a row vector) into the layout of column
        vector slot ``target`` or vice versa through the scratch tile: the
        owning lanes write each element, every lane reads its registers of
        the target layout.  Returns the temporaries' prefix."""
        program = self.program
        source = program.vec_leaves[leaf]
        base = self.scratch_base()
        prefix = self.df.new_var("md_rt")
        with self.nest(f"if {self.vec_owner(source)}:"):
            for atom, half in self.vec_regs(source):
                index = self.vec_index(source, atom, half)
                self.emit(
                    f"{_RUNTIME}.sts_f32({base} + ({index}) * 4, {self.vreg(source, atom, half)})"
                )
        self.emit(self.sync)
        for atom, half in self.vec_regs(target):
            index = self.vec_index(target, atom, half)
            self.emit(
                f"{prefix}_{atom}_{half} = {_RUNTIME}.lds_f32({base} + ({index}) * 4)"
            )
        self.emit(self.sync)
        return prefix

    def emit_vec_op(self, step: VecOp) -> None:
        """A pointwise vector value per vector register; inputs of the other
        orientation are relaid out through the scratch tile first."""
        program = self.program
        overrides: dict[Node, str] = {}
        target = program.vec_slots[step.slot]
        for leaf in step.vecs:
            slot = program.vec_leaves[leaf]
            source = program.vec_slots[slot]
            if (
                source.axis is not None
                and target.axis is not None
                and source.orient != target.orient
            ):
                overrides[leaf] = self.emit_relayout(leaf, step.slot)
        self.hoist_scalars(step.node)
        updates = []
        self.evaluator.vec_overrides = overrides
        for atom, half in self.vec_regs(step.slot):
            self.evaluator.vec_reg = (atom, half)
            coords = self.vec_coords(step.node, step.slot, atom, half)
            updates.append(
                (
                    self.vreg(step.slot, atom, half),
                    self.evaluator.define(step.node, coords),
                )
            )
        self.evaluator.vec_reg = None
        self.evaluator.vec_overrides = {}
        for name, value in updates:
            self.emit(f"{name} = {value}")

    def emit_vec_store(self, step: VecStore) -> None:
        """Store a vector value: the lanes holding each element once write
        theirs (thread 0 for a scalar)."""
        slot = self.program.vec_leaves[step.vec]
        access = step.access
        positions = access.tile_positions
        self.hoist_scalars(step.value)
        with self.nest(f"if {self.vec_owner(slot)}:"):
            for atom, half in self.vec_regs(slot):
                self.evaluator.vec_reg = (atom, half)
                coords = self.vec_coords(step.value, slot, atom, half)
                value = self.evaluator.evaluate(step.value, coords)
                self.evaluator.vec_reg = None
                element = ["0"] * len(access.dims)
                if positions:
                    element[positions[0]] = self.vec_index(slot, atom, half)
                address = self.address(access, tuple(element))
                self.emit_guarded(
                    self.bounds(access, tuple(element)),
                    f"{_RUNTIME}.stg_f32({address}, {value})",
                )

    def emit_derived(self, step: Derived) -> None:
        """Materialize a derived fragment: evaluate its pointwise DAG per
        register over the input fragments' registers, with the 2-D tile
        inputs loaded as pairs at the fragment coordinates."""
        layout = self.slot_layouts[step.slot]
        self.hoist_scalars(step.node)
        self.evaluator.enter_frag(step.slot)
        # A 16-bit-typed value: its rounded elements widened (exactly) into
        # the fp32 registers; the readers re-type them (``_frag_dtype``).
        widen = _frag_dtype(step.node) is not None
        updates = []
        for am in range(layout.atoms_m):
            for an in range(layout.atoms_n):
                for half in range(2):
                    row, col = layout.row(am, half), layout.col(an)
                    for access in step.loads:
                        tile = self.derived_tiles.get(access.node)
                        if tile is not None:
                            # The pair (col, col + 1) sits in 16-byte chunk
                            # col * es // 16 of the staged row (swizzled).
                            es = access.dtype.itemsize
                            chunk = f"(({col}) * {es} // 16)"
                            swz = _swizzled_chunk(row, chunk, tile.cpr)
                            lo, hi = self.df.new_var("md_lo"), self.df.new_var("md_hi")
                            address = (
                                f"{tile.base(self.stage_var(step.loop))} + {row} * {tile.row_bytes}"
                                f" + {swz} * 16 + ({col}) * {es} % 16"
                            )
                            if access.dtype == torch.float32:
                                self.emit(
                                    f"{lo}, {hi} = {_RUNTIME}.lds_v2_f32({address})"
                                )
                            else:
                                unpack = (
                                    "unpack_f16x2"
                                    if access.dtype == torch.float16
                                    else "unpack_bf16x2"
                                )
                                self.emit(
                                    f"{lo}, {hi} = {_RUNTIME}.{unpack}({_RUNTIME}.lds_b32({address}))"
                                )
                        else:
                            positions = [
                                i
                                for i, axis in enumerate(access.dims)
                                if axis.kind != AXIS_LEAD
                            ]
                            coords = ["0"] * len(access.dims)
                            coords[positions[0]] = row
                            coords[positions[1]] = col
                            address = self.address(access, tuple(coords))
                            paired = self.pairs_aligned(access, 1)
                            lo, hi = self.pair_load(
                                address,
                                access.dtype,
                                paired,
                                self.bounds_pred(access, tuple(coords)),
                                readonly=self.readonly(access),
                            )
                        self.set_tile_leaf(access, 0, lo)
                        self.set_tile_leaf(access, 1, hi)
                    lanes = tuple(
                        _Lane(
                            self.frag_coords(step.node, layout, am, an, 2 * half + bit),
                            bit,
                            (am, an, 2 * half + bit),
                        )
                        for bit in range(2)
                    )
                    for bit, value in enumerate(
                        self.evaluator.define_pair(step.node, lanes)
                    ):
                        if widen:
                            value = f"cutlass.Float32({value})"
                        updates.append(
                            (self.reg(step.slot, am, an, 2 * half + bit), value)
                        )
                    self.evaluator.reg = None
                    self.evaluator.element = None
        self.evaluator.leave_frag()
        for name, value in updates:
            self.emit(f"{name} = {value}")
