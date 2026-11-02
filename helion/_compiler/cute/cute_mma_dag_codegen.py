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
import dataclasses
import functools
import itertools
import math
import re
import textwrap
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast

import sympy
import torch

from ...language import memory_ops
from ..compile_environment import CompileEnvironment
from .cute_mma_dag import AXIS_LEAD
from .cute_mma_dag import AXIS_LOOP
from .cute_mma_dag import AXIS_RDIM
from .cute_mma_dag import AXIS_ROOT
from .cute_mma_dag import BOUND_ID
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
from .cute_mma_dag import _is_pointwise
from .cute_mma_dag import _node_inputs
from .cute_mma_dag import _storage
from .cute_mma_dag import _tile_index_position
from .cute_mma_dag import async_tile_reason
from .cute_mma_dag import frag_word_blocks
from .cute_mma_dag import loop_bound_terms
from .cute_mma_dag import loop_trip_range
from .cute_mma_dag import rounded_dot_dtype
from .cute_warp_mma_gemm import _swizzled_chunk
from .cute_warp_mma_gemm import warp_mma_warp_layout

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Iterator
    from collections.abc import Sequence

    from torch.fx.node import Node

    from ..device_function import DeviceFunction
    from ..generate_ast import GenerateAST
    from .cute_mma_dag import MmaDagProgram

_RUNTIME = "_helion_mma_dag"
# The tcgen05 runtime (mbarriers, bulk tensor copies) the warp mode's TMA
# ring borrows, and its module import.
_TC = "_helion_mma_dag_tc"
_TC_PREAMBLE = f"import helion._compiler.cute._mma_dag_tcgen05_runtime as {_TC}\n"
# TMA swizzle of a tile's row width (the emitter's XOR swizzle equals it).
_TMA_SWIZZLE = {16: "none", 32: "s32b", 64: "s64b", 128: "s128b"}
# A name in a coordinate expression (not an attribute).
_IDENTIFIER = r"(?<![\w.])[A-Za-z_]\w*"
# Registers a thread spends on side loads prefetched one iteration ahead
# (``prefetch_level``): a pass over a tile with a decay per element has one
# site per packet element, far more than the serial chain's few scalars.
MMA_DAG_MAX_PREFETCH = 16
# Module roots an address expression may name (``cutlass.Int64(...)``).
_MODULES = frozenset({"cutlass", "cute"})
_PREAMBLE = f"import helion._compiler.cute._mma_dag_runtime as {_RUNTIME}\n"
# Side-load dtypes loaded widened to fp32 (``ldg_*_as_f32``): the raw fp32
# value stands for the element, the DSL-typed one for the FX value.
_WIDENED: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16)
# The launcher's wrapper plan that names the program's tensor arguments so
# a launch can refuse arguments that alias (see ``mma_dag_alias_plan``).
ALIAS_KIND = "helion_mma_dag_alias"
Coords = tuple[str, ...]


def emit_mma_dag_module_statements(cg: GenerateAST) -> None:
    """Emit the once-per-module import of the runtime helpers."""
    if getattr(cg, "_helion_mma_dag_module_emitted", False):
        return
    cg._helion_mma_dag_module_emitted = True  # type: ignore[attr-defined]
    for stmt in ast.parse(_PREAMBLE).body:
        cg.module_statements.append(stmt)


def codegen_mma_dag(cg: GenerateAST) -> bool:
    """Emit the warp-MMA program for the planned root (False: not this root)."""
    df = cg.device_function
    plan = df.cute_state.mma_dag_plan
    root = cg.current_root_graph_info
    if plan is None or root is None or root.graph_id != plan.program.root_graph_id:
        return False
    # This render's exact fp32 widenings (``_Emitter.retyped``): fresh per
    # kernel, the names recur from one render of a bound kernel to the next.
    df.cute_state.mma_dag_exact_f32 = {}
    if plan.compute == MMA_DAG_TCGEN05:
        from .cute_mma_dag_tcgen05 import codegen_mma_dag_tcgen05

        codegen_mma_dag_tcgen05(cg, plan)
        return True
    emit_mma_dag_module_statements(cg)
    emitter = _Emitter(cg, plan)
    source = emitter.render()
    if emitter.tma_maps and not getattr(cg, "_helion_mma_dag_tc_module_emitted", False):
        cg._helion_mma_dag_tc_module_emitted = True  # type: ignore[attr-defined]
        for stmt in ast.parse(_TC_PREAMBLE).body:
            cg.module_statements.append(stmt)
    for statement in ast.parse(textwrap.dedent(source)).body:
        cg.add_statement(statement)
    if emitter.tma_maps:
        # The host builds one tensor map per entry over the tensor views and
        # passes them as grid constants after the ordinary arguments.
        wrapper: dict[str, object] = {
            "kind": "helion_mma_dag_tma",
            "tma_count": len(emitter.tma_maps),
        }
        for tmap in emitter.tma_maps:
            access = tmap.access
            name = df.tensor_arg(access.fake, prefer_name=access.name).name
            wrapper[f"tma{tmap.index}_name"] = name
            wrapper[f"tma{tmap.index}_param"] = tmap.param
            wrapper[f"tma{tmap.index}_dims"] = list(tmap.dims)
            wrapper[f"tma{tmap.index}_box"] = list(tmap.box)
            wrapper[f"tma{tmap.index}_swizzle"] = tmap.swizzle
            wrapper[f"tma{tmap.index}_elem"] = access.dtype.itemsize
            df.wrapper_only_params.append(tmap.param)
            df.wrapper_only_param_annotations[tmap.param] = (
                f"cutlass.GridConstant[{_TC}.cuda.TensorMap]"
            )
        cg.cute_wrapper_plans.append(wrapper)
    cg.cute_wrapper_plans.append(mma_dag_alias_plan(df, plan.program))
    df.cute_state.owned_root_block_dims = (plan.launch_threads, 1, 1)
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


class _Layout:
    """The warp grid of one fragment shape and the lane's coordinates in it."""

    def __init__(
        self,
        index: int,
        rows: int,
        cols: int,
        warps: int,
        layout: tuple[int, int] | None = None,
    ) -> None:
        if layout is None:
            layout = warp_mma_warp_layout(rows, cols, warps)
        if layout is None:
            raise _decline(f"{warps} warps cannot tile a {rows}x{cols} fragment")
        self.index = index
        self.rows, self.cols = rows, cols
        self.wm, self.wn = layout
        self.wtm, self.wtn = rows // self.wm, cols // self.wn
        self.atoms_m, self.atoms_n = self.wtm // 16, self.wtn // 8

    @property
    def rowl(self) -> str:
        return f"md_L{self.index}_rowl"

    @property
    def coll(self) -> str:
        return f"md_L{self.index}_coll"

    def row(self, am: int, half: int) -> str:
        offset = 16 * am + 8 * half
        return f"({self.rowl} + {offset})" if offset else self.rowl

    def col(self, an: int, bit: int = 0) -> str:
        offset = 8 * an + bit
        return f"({self.coll} + {offset})" if offset else self.coll

    def registers(self) -> list[tuple[int, int, int]]:
        return [
            (am, an, j)
            for am in range(self.atoms_m)
            for an in range(self.atoms_n)
            for j in range(4)
        ]


class _Evaluator:
    """Render FX values element by element at positional coordinates.

    ``coords`` holds one tile-local coordinate expression per dim of the
    value (``"0"`` for size-1 and leading dims); fragment leaves resolve to
    the register of the current ``reg`` (set by the per-register loops);
    2-D operand loads are supplied by the caller as already-loaded leaves
    keyed by the element's position in the loaded pair / packet.
    """

    def __init__(self, emitter: _Emitter) -> None:
        self.emitter = emitter
        self.program = emitter.program
        self.scopes: list[dict[object, str]] = [{}]
        self.tile_leaves: dict[tuple[Node, int], str] = {}
        # The exact fp32 values of 16-bit tile leaves (before re-typing).
        self.raw_leaves: dict[tuple[Node, int], str] = {}
        self.reg: tuple[int, int, int] | None = None
        self.element: int | None = None
        # The chunk of the source-tile packet a pass is rewriting (its
        # ``element`` indexes the packet's elements along the contiguous dim).
        self.packet: str | None = None
        # The vector register (atom, half) vector leaves resolve to; leaves
        # relaid out for the current op resolve to their temporaries.
        self.vec_reg: tuple[int, int] | None = None
        self.vec_overrides: dict[Node, str] = {}
        # The fragment slot whose registers ``reg`` indexes (set by the
        # per-register steps: stores, derived values, reductions, rescales,
        # seeds), and the vector leaves relaid out into that slot's register
        # positions for the current step (see ``_Emitter.frag_vec``).
        self.frag_slot: int | None = None
        self.frag_vecs: dict[tuple[Node, int], dict[tuple[object, ...], str]] = {}
        # Derived fragments evaluated through their pointwise DAG instead of
        # their registers: the chain a pass-routed operand absorbs (its
        # steps never materialize; the pass reads their tile leaves).
        self.inline: frozenset[Node] = frozenset()
        # A 16-bit ``convert_element_type`` whose rounding a consumer that
        # rounds itself (a pack) performs: while ``unrounded_scope`` holds,
        # the nodes above it (views) are evaluated as their unrounded fp32
        # values under separate memo keys, and the convert itself as its
        # input (see ``unrounded_pair``).
        self.skip_round: Node | None = None
        self.unrounded_scope = False

    def enter_frag(self, slot: int) -> None:
        """Begin a step evaluating values register by register over slot
        ``slot``'s layout."""
        self.frag_slot = slot
        self.frag_vecs = {}

    def leave_frag(self) -> None:
        self.reg = None
        self.element = None
        self.frag_slot = None
        self.frag_vecs = {}

    def push(self) -> None:
        self.scopes.append({})

    def pop(self) -> None:
        self.scopes.pop()

    def _lookup(self, key: object) -> str | None:
        for scope in reversed(self.scopes):
            if key in scope:
                return scope[key]
        return None

    def _key(self, node: Node, coords: Coords) -> tuple[object, Coords]:
        """The memo key of ``node`` at ``coords``: the nodes above a convert
        whose rounding the consumer performs (``unrounded_scope``) hold
        unrounded values and are keyed apart from their rounded ones."""
        if self.unrounded_scope:
            return (("unrounded", node), coords)
        return (node, coords)

    def evaluate(self, node: Node, coords: Coords) -> str:
        from ...language import _tracing_ops
        from ...language import creation_ops
        from ...language import memory_ops
        from ...language import view_ops

        em = self.emitter
        if node.op == "call_function" and node.target is _tracing_ops._new_var:
            return self.evaluate(cast("Node", node.args[0]), coords)
        if node in self.program.aliases:
            return self.evaluate(self.program.aliases[node], coords)
        if self.skip_round is not None and node is self.skip_round:
            # The rounding of this convert belongs to the consumer's pack:
            # its input's fp32 value, memoized under the ordinary keys.
            saved = self.unrounded_scope
            self.unrounded_scope = False
            try:
                return self.evaluate(cast("Node", node.args[0]), coords)
            finally:
                self.unrounded_scope = saved
        if node.op == "call_function" and node.target is _tracing_ops._mask_to:
            # An identity on whole tiles; over a ragged root tile the
            # elements past the extent take the fill (the frontend inserts
            # it ahead of the reductions and dots the padded rows would
            # otherwise feed).
            source = cast("Node", node.args[0])
            bounds = em.value_bounds(node, coords)
            if bounds is None:
                return self.evaluate(source, coords)
            key = self._key(node, coords)
            cached = self._lookup(key)
            if cached is None:
                inner = self.evaluate(source, coords)
                fill = em.constant(node.args[1], node.meta["val"].dtype)
                cached = em.bind("md_v", f"{inner} if {bounds} else {fill}")
                self.scopes[-1][key] = cached
            return cached
        if node in self.program.frag_slots and node not in self.inline:
            assert self.reg is not None
            register = em.reg(self.program.frag_slots[node], *self.reg)
            rounded = _frag_dtype(node)
            if rounded is None:
                return register
            # A 16-bit ``aten.mm`` (its fp32 product rounded once, as the
            # SIMT lowering rounds it) or a 16-bit-typed derived value (its
            # rounded elements widened into the registers): the DSL value of
            # the FX dtype.
            key = ("rounded", node, coords, register)
            cached = self._lookup(key)
            if cached is None:
                cached = em.bind("md_v", em.typed(register, rounded))
                self.scopes[-1][key] = cached
            return cached
        if node in self.program.vec_leaves:
            slot = self.program.vec_leaves[node]
            if self.program.vec_slots[slot].axis is None:
                return em.vreg(slot, 0, 0)
            if self.vec_reg is None:
                # Read beside a fragment value (``dgc_pos + is_last *
                # dgc_last[:, None, :]``): the vector's register at the
                # fragment element's row / column.
                return em.frag_vec(node, coords)
            if node in self.vec_overrides:
                return f"{self.vec_overrides[node]}_{self.vec_reg[0]}_{self.vec_reg[1]}"
            return em.vreg(slot, *self.vec_reg)
        # Memoized per element position: values that do not depend on the
        # packet element (a K-vector decay of the packet's row) are shared
        # across the packet; the loaded tile leaves are keyed by element.
        key = self._key(node, coords)
        cached = self._lookup(key)
        if cached is not None:
            return cached
        widened = self.widened_leaf(node, coords)
        rounded = self.rounded_leaf(node, coords) if widened is None else None
        if widened is not None:
            result = widened
        elif rounded is not None:
            result = rounded
        elif node in self.program.tile_index_loads:
            axis = self.program.tile_index_loads[node]
            coord = coords[cast("int", _tile_index_position(node))]
            result = em.bind("md_v", f"cutlass.Int32({em.axis_base(axis)} + {coord})")
        elif node.op == "call_function" and node.target is memory_ops.load:
            # (A uniform scalar read from a tensor -- a window's limit --
            # was loaded once with the bases.)
            result = em.scalar_vars.get(node) or self._load(node, coords)
        elif node.op == "call_function" and node.target is _tracing_ops._get_symnode:
            val = node.meta["val"]
            result = em.df.sympy_expr(val._sympy_())
        elif node.op == "call_function" and node.target is view_ops.subscript:
            index = cast("list[object]", node.args[1])
            base_coords = tuple(
                coord
                for coord, item in zip(coords, index, strict=True)
                if item is not None
            )
            result = self.evaluate(cast("Node", node.args[0]), base_coords)
        elif node.op == "call_function" and node.target in (
            torch.ops.aten.permute.default,
            torch.ops.aten.transpose.int,
        ):
            result = self.evaluate(
                cast("Node", node.args[0]), _permute_in(node, coords)
            )
        elif node.op == "call_function" and node.target is torch.ops.prims.iota.default:
            result = em.bind("md_v", f"cutlass.Int32({coords[0]})")
        elif node.op == "call_function" and node.target is torch.ops.aten.full.default:
            # ``torch.zeros_like(v)``: the fill at every coordinate (its
            # shape inputs are no values); the consumer casts it.
            from torch._inductor.codegen.simd import constant_repr

            fill = cast("float | int", node.args[1])
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor) and val.dtype.is_floating_point:
                fill = float(fill)
            result = constant_repr(fill)
        elif node.op == "call_function" and node.target is creation_ops.full:
            # ``hl.zeros`` / ``hl.full`` beside a fragment: the constant
            # (``hl.full(..., float("-inf"))`` under a ``where`` spelled as
            # a call: ``repr(-inf)`` is a bare name in the DSL module).
            fill = cast("float", node.args[1])
            result = em.bind(
                "md_v",
                em.typed(
                    f"cutlass.Float32({_float_literal(float(fill))})",
                    node.meta["val"].dtype,
                ),
            )
        elif (
            node.op == "call_function" and node.target is torch.ops.aten.expand.default
        ):
            # ``broadcast_to``: the input's coordinates under broadcasting.
            source = cast("Node", node.args[0])
            result = self.evaluate(
                source, _broadcast(coords, _shape(node), _shape(source))
            )
        else:
            result = self._lowered(node, coords)
            if node in self.inline:
                rounded = _frag_dtype(node)
                if rounded is not None:
                    # A 16-bit-typed step of an inlined chain rounds where
                    # the register path widened it into the fp32 register
                    # and re-typed it on the read (the DSL's 16-bit product
                    # is not rounded by itself).
                    result = em.bind(
                        "md_v", em.typed(f"cutlass.Float32({result})", rounded)
                    )
        self.scopes[-1][key] = result
        return result

    def define(self, node: Node, coords: Coords) -> str:
        """Render the *definition* of a derived fragment at ``coords`` (the
        node is a fragment leaf, so ``evaluate`` would name its register)."""
        return self._lowered(node, coords)

    # -- element pairs ---------------------------------------------------------

    def lane(self, node: Node, lane: _Lane) -> str:
        """``evaluate`` at one lane of a pair (its coordinates, packet element
        and register)."""
        self.element, self.reg = lane.element, lane.reg
        return self.evaluate(node, lane.coords)

    def pair(self, node: Node, lanes: tuple[_Lane, ...]) -> tuple[str, str]:
        """Render ``node`` at two adjacent element positions.

        An fp32 ``mul`` / ``add`` / ``sub`` over two tile values whose
        scalar lowering could not fuse into an FMA (``_packable``) renders
        as one packed f32x2 op over both lanes (each lane rounded as the
        scalar op would be: the same bits in half the instructions); a value
        uniform over the pair is rendered once; everything else lane by lane
        through ``evaluate``, which memoizes per (node, coords) so the two
        paths share their inputs."""
        from ...language import view_ops

        lo, hi = lanes
        if lo.coords == hi.coords and lo.element == hi.element and lo.reg == hi.reg:
            value = self.lane(node, lo)
            return value, value
        resolved = self.program.resolve(node)
        if self.skip_round is not None and resolved is self.skip_round:
            # The consumer's pack rounds: the convert's input, as a pair.
            saved = self.unrounded_scope
            self.unrounded_scope = False
            try:
                return self.pair(cast("Node", resolved.args[0]), lanes)
            finally:
                self.unrounded_scope = saved
        if (
            not self.emitter.pairs
            or (resolved in self.program.frag_slots and resolved not in self.inline)
            or resolved in self.program.vec_leaves
            or not _has_packable(resolved, self.emitter.packable_memo)
        ):
            return self.lane(node, lo), self.lane(node, hi)
        keys = (self._key(resolved, lo.coords), self._key(resolved, hi.coords))
        cached = (self._lookup(keys[0]), self._lookup(keys[1]))
        if cached[0] is not None and cached[1] is not None:
            return cast("tuple[str, str]", cached)
        # Views: the input at the lanes' mapped coordinates.
        if resolved.op == "call_function" and resolved.target in (
            torch.ops.aten.permute.default,
            torch.ops.aten.transpose.int,
        ):
            names = self.pair(
                cast("Node", resolved.args[0]),
                (
                    lo._replace(coords=_permute_in(resolved, lo.coords)),
                    hi._replace(coords=_permute_in(resolved, hi.coords)),
                ),
            )
        elif resolved.op == "call_function" and resolved.target is view_ops.subscript:
            index = cast("list[object]", resolved.args[1])
            names = self.pair(
                cast("Node", resolved.args[0]),
                (
                    lo._replace(coords=_subscript_in(lo.coords, index)),
                    hi._replace(coords=_subscript_in(hi.coords, index)),
                ),
            )
        elif (
            resolved.op == "call_function"
            and resolved.target is torch.ops.aten.expand.default
        ):
            source = cast("Node", resolved.args[0])
            shape, in_shape = _shape(resolved), _shape(source)
            names = self.pair(
                source,
                (
                    lo._replace(coords=_broadcast(lo.coords, shape, in_shape)),
                    hi._replace(coords=_broadcast(hi.coords, shape, in_shape)),
                ),
            )
        elif _packable(resolved):
            names = self._packed(resolved, lanes)
        else:
            # A pointwise op over a packable subtree: its tensor inputs
            # first, as pairs (the lanes then find them memoized).
            shape = _shape(resolved)
            for arg in _node_inputs(resolved):
                arg_shape = _shape(arg)
                self.pair(
                    arg,
                    (
                        lo._replace(coords=_broadcast(lo.coords, shape, arg_shape)),
                        hi._replace(coords=_broadcast(hi.coords, shape, arg_shape)),
                    ),
                )
            return self.lane(node, lo), self.lane(node, hi)
        self.scopes[-1][keys[0]], self.scopes[-1][keys[1]] = names
        return names

    def rounding_root(self, node: Node) -> Node | None:
        """The 16-bit ``convert_element_type`` that is ``node`` or sits
        right under ``node``'s views (permute / transpose / subscript /
        expand -- ``k_i.transpose(-2, -1)`` of ``k_i.to(bf16)``), whose
        rounding a consumer that rounds itself may perform instead; None
        when a pointwise op or anything else sits between."""
        from ...language import view_ops

        source = self.program.resolve(node)
        while source.op == "call_function":
            if (
                source.target is torch.ops.prims.convert_element_type.default
                and source.args[1] in (torch.bfloat16, torch.float16)
            ):
                return source
            if (
                source.target
                in (
                    torch.ops.aten.permute.default,
                    torch.ops.aten.transpose.int,
                    torch.ops.aten.expand.default,
                )
                or source.target is view_ops.subscript
            ):
                source = self.program.resolve(cast("Node", source.args[0]))
                continue
            return None
        return None

    @contextlib.contextmanager
    def unrounding(self, root: Node) -> Iterator[None]:
        """Evaluate with the rounding of convert ``root`` left to the caller."""
        saved = (self.skip_round, self.unrounded_scope)
        self.skip_round, self.unrounded_scope = root, True
        try:
            yield
        finally:
            self.skip_round, self.unrounded_scope = saved

    def unrounded_pair(self, node: Node, lanes: tuple[_Lane, ...]) -> tuple[str, str]:
        """``unrounded`` over a pair (see ``pair``)."""
        root = self.rounding_root(node)
        if root is not None:
            if root is self.program.resolve(node):
                return self.pair(cast("Node", root.args[0]), lanes)
            # The convert under views: the views map the lanes' coordinates,
            # the convert yields its input (one rounding, the pack's, instead
            # of a scalar ``cvt`` to 16 bits widened back and rounded again).
            with self.unrounding(root):
                return self.pair(node, lanes)
        source = self.program.resolve(node)
        if _unrounded_select(self.program, source) is not None:
            values = []
            for lane in lanes:
                self.element, self.reg = lane.element, lane.reg
                values.append(self.unrounded(node, lane.coords))
            return values[0], values[1]
        return self.pair(node, lanes)

    def define_pair(self, node: Node, lanes: tuple[_Lane, ...]) -> tuple[str, str]:
        """``define`` over a pair: a packable op renders packed over its
        inputs' pairs, anything else lane by lane."""
        if self.emitter.pairs and _packable(node):
            return self._packed(node, lanes)
        lo, hi = lanes
        self.element, self.reg = lo.element, lo.reg
        first = self._lowered(node, lo.coords)
        self.element, self.reg = hi.element, hi.reg
        return first, self._lowered(node, hi.coords)

    def _packed(self, node: Node, lanes: tuple[_Lane, ...]) -> tuple[str, str]:
        """One packed f32x2 op for ``node`` (a packable op) over both lanes."""
        lo, hi = lanes
        shape = _shape(node)
        operands = []
        for arg in node.args[:2]:
            arg_node = cast("Node", arg)
            if not isinstance(arg_node.meta.get("val"), torch.Tensor):
                # A uniform scalar (a float argument): the same fp32 value
                # on both lanes.
                value = f"cutlass.Float32({self.lane(arg_node, lo)})"
                operands.append((value, value))
                continue
            arg_shape = _shape(arg_node)
            operands.append(
                self.pair(
                    arg_node,
                    (
                        lo._replace(coords=_broadcast(lo.coords, shape, arg_shape)),
                        hi._replace(coords=_broadcast(hi.coords, shape, arg_shape)),
                    ),
                )
            )
        (a0, a1), (b0, b1) = operands
        op = _PACKED_OPS[node.target]
        return self.emitter.bind_pair(
            "md_v", f"cute.arch.{op}_packed_f32x2(({a0}, {a1}), ({b0}, {b1}))"
        )

    def unrounded(self, node: Node, coords: Coords) -> str:
        """The fp32 value whose round-to-nearest 16-bit conversion is
        ``node``'s value, for a consumer that rounds itself (``cvt.rn.*x2``
        of a pack): a ``convert_element_type(x, 16-bit)`` root yields ``x``
        (one rounding instead of a scalar ``cvt`` followed by the pack's), a
        ``where(mask, rounded, c)`` over such a value and an exactly
        representable constant selects the unrounded value (the mask's
        ``where`` over 16-bit operands cost three scalar converts per
        element), anything else its own value."""
        root = self.rounding_root(node)
        if root is not None:
            if root is self.program.resolve(node):
                return self.evaluate(cast("Node", root.args[0]), coords)
            with self.unrounding(root):
                return self.evaluate(node, coords)
        source = self.program.resolve(node)
        selected = _unrounded_select(self.program, source)
        if selected is not None:
            mask, value, constant = selected
            return self.emitter.bind(
                "md_v",
                CompileEnvironment.current().backend.where_expr(
                    self.evaluate(
                        mask, _broadcast(coords, _shape(source), _shape(mask))
                    ),
                    f"cutlass.Float32({self.unrounded(value, _broadcast(coords, _shape(source), _shape(value)))})",
                    f"cutlass.Float32({constant!r})",
                ),
            )
        return self.evaluate(node, coords)

    def rounded_leaf(self, node: Node, coords: Coords) -> str | None:
        """``node`` is ``convert_element_type(frag, float32)`` of a fragment
        leaf read rounded to bf16 (a 16-bit ``aten.mm``'s fp32 product, or a
        bf16-typed derived value widened into its fp32 register): the
        register rounded and widened in one step (``round_bf16_f32``: an
        FMA-pipe ``cvt.rn.bf16x2.f32`` and a mask) instead of the DSL's
        scalar ``cvt`` to bf16 on the conversion unit and the widening cast
        behind it -- the same bits.  (fp16 widens by a conversion either
        way and keeps the casts.)"""
        if (
            node.op != "call_function"
            or node.target is not torch.ops.prims.convert_element_type.default
            or node.args[1] is not torch.float32
            or self.reg is None
        ):
            return None
        source = self.program.resolve(cast("Node", node.args[0]))
        if source not in self.program.frag_slots or source in self.inline:
            return None
        if _frag_dtype(source) is not torch.bfloat16:
            return None
        em = self.emitter
        register = em.reg(self.program.frag_slots[source], *self.reg)
        return em.bind("md_v", f"{_RUNTIME}.round_bf16_f32({register})")

    def widened_leaf(self, node: Node, coords: Coords) -> str | None:
        """``node`` is ``convert_element_type(load, float32)`` of a 16-bit
        load whose fp32 value is already at hand (the unpacked packet half or
        the widening global load): that value, exact, instead of the DSL's
        16-bit re-typing and widening casts."""
        if (
            node.op != "call_function"
            or node.target is not torch.ops.prims.convert_element_type.default
            or node.args[1] is not torch.float32
        ):
            return None
        source = self.program.resolve(cast("Node", node.args[0]))
        access = self.emitter.accesses.get(source)
        if access is None or access.dtype not in _WIDENED:
            return None
        if len(access.tile_dims) == 2:
            assert self.element is not None
            raw = self.raw_leaves.get((source, self.element))
            if raw is None:
                return None
            # (A cast keeps its source's shape: the coordinates are the
            # source's.)
            return self.tile_leaf(source, coords, raw, "cutlass.Float32")
        key = (source, coords)
        raw = self._lookup(("raw", key))
        if raw is None:
            # Evaluating the load binds both the raw and the typed value.
            self.evaluate(source, coords)
            raw = self._lookup(("raw", key))
        return raw

    def tile_leaf(self, node: Node, coords: Coords, leaf: str, dtype: str) -> str:
        """The staged tile element ``leaf`` of load ``node`` at ``coords``:
        zero when the element's row lies at or past its window's limit (a
        TMA box lands the window's whole rows; the warp emitter's packets
        zero-fill them, so there the guard folds to the same value)."""
        em = self.emitter
        access = em.accesses[node]
        if not access.windows or not em.masked_tile_windows:
            return leaf
        guard = em.bounds(access, coords)
        assert guard is not None
        # (Both branches typed alike: an unpacked 16-bit half is an fp32
        # value, exact in its own type again.)
        return em.bind("md_v", f"{dtype}({leaf}) if {guard} else {dtype}(0)")

    def _load(self, node: Node, coords: Coords) -> str:
        em = self.emitter
        access = em.accesses[node]
        if len(access.tile_dims) == 2:
            assert self.element is not None
            leaf = self.tile_leaves[(node, self.element)]
            return self.tile_leaf(
                node, coords, leaf, em.env.backend.dtype_str(access.dtype)
            )
        alias = em.ring_aliases.get(node)
        if alias is not None and alias.loop in em.ring_ready:
            # A row / column of a tile this iteration's ring slot holds (see
            # ``_ring_aliases``): the element from shared memory at the
            # slot's swizzled address, the same bytes the global load reads.
            tile = alias.tile
            coord = coords[access.tile_positions[0]]
            row, col = (
                (str(alias.fixed), coord)
                if alias.fixed_row
                else (coord, str(alias.fixed))
            )
            es = access.dtype.itemsize
            chunk = f"(({col}) * {es} // 16)"
            swz = _swizzled_chunk(row, chunk, tile.cpr)
            address = (
                f"{tile.base(em.stage_var(alias.loop))} + ({row}) * {tile.row_bytes}"
                f" + {swz} * 16 + (({col}) * {es}) % 16"
            )
            op = "lds_f32" if access.dtype == torch.float32 else "lds_b32"
            return em.bind("md_v", f"{_RUNTIME}.{op}({address})")
        address = em.address(access, coords)
        # An element past a ragged axis' extent loads as zero (the
        # frontend's masked-value model), under a predicate.
        bounds = em.bounds(access, coords)
        hoist = em.hoist_level(access, coords)
        if hoist is not None:
            # A loop-invariant per-thread load goes out once, ahead of the
            # loop (in flight under its prologue), and the iterations read
            # the registers.
            key = (node, coords)
            if key not in hoist.loads:
                pred = None
                if bounds is not None:
                    pred = em.df.new_var("md_ok")
                    em.hoist(hoist, f"{pred} = {em.pred_int(bounds)}")
                value = em.df.new_var("md_sl")
                em.hoist(
                    hoist,
                    f"{value} = {em.scalar_load(address, access.dtype, pred, readonly=em.readonly(access))}",
                )
                typed = value
                if access.dtype in _WIDENED:
                    typed = em.df.new_var("md_sl")
                    em.hoist(hoist, f"{typed} = {em.typed(value, access.dtype)}")
                    em.df.cute_state.mma_dag_exact_f32[typed] = value
                hoist.loads[key] = (value, typed)
            value, typed = hoist.loads[key]
            if access.dtype in _WIDENED:
                self.scopes[-1][("raw", key)] = value
            return typed
        # An element past a ragged axis' extent loads as zero under a
        # predicate; the aux ring and the register prefetch stage only the
        # unpredicated loads.
        pred = None if bounds is None else em.bind("md_ok", em.pred_int(bounds))
        shared = em.aux_shared_level(access) if pred is None else None
        if shared is not None:
            # An fp32 scalar / vector the aux ring copies once per CTA into
            # the loop's side tile (``AuxRing.shared``: the pass's decay
            # scalar and gate row, the carry rescale's, a derived value's
            # row): this iteration's slot holds its block extent, landed and
            # published by the previous iteration's wait and barrier (the
            # tile rides one group ahead of the operand tiles), so it is
            # read at the lane's own coordinate anywhere in the iteration.
            if not em.aux_synced:
                em.aux_shared_early.add(shared)
            return em.bind("md_v", em.aux_shared_read(shared, access, coords))
        aux = em.aux_level(access, coords) if pred is None else None
        if aux is not None:
            value = None
            ring = em.plan.aux_rings[aux.loop]
            if (
                node in ring.runs
                and em.aux_synced
                and self.packet is not None
                and self.element is not None
                and self.element < ring.runs[node][0]
            ):
                # The run of the vector this packet reads rode the operand
                # ring into this thread's shared slots (a ring depth ahead):
                # one 16-byte read per 16 bytes, shared by the packet's lanes.
                run_key = ("aux_run", node, self.packet, self.element)
                value = self._lookup(run_key)
                if value is None:
                    em.aux_wait(aux.loop)
                    values = em.aux_run_values(aux.loop, access, self.packet)
                    for e, name in enumerate(values):
                        self.scopes[-1][("aux_run", node, self.packet, e)] = name
                    value = values[self.element]
            if value is None:
                slot = em.aux_slot(aux, access, coords)
                if slot is not None:
                    em.aux_wait(aux.loop)
                    value = em.bind("md_v", em.aux_read(aux.loop, slot, access, coords))
            if value is not None:
                if access.dtype not in _WIDENED:
                    return value
                self.scopes[-1][("raw", (node, coords))] = value
                return em.retyped(value, access.dtype)
        prefetch = em.prefetch_level(access, coords) if pred is None else None
        if prefetch is not None:
            # A loop-variant per-thread load of the innermost loop: its
            # register was loaded one iteration ahead (the first iteration's
            # ahead of the loop); snapshot it, the iteration refills it.
            register = em.prefetch_register(prefetch, access, coords)
            if register is not None:
                value = em.bind("md_v", register)
                if access.dtype not in _WIDENED:
                    return value
                self.scopes[-1][("raw", (node, coords))] = value
                return em.retyped(value, access.dtype)
        value = em.bind(
            "md_v",
            em.scalar_load(address, access.dtype, pred, readonly=em.readonly(access)),
        )
        if access.dtype not in _WIDENED:
            return value
        self.scopes[-1][("raw", (node, coords))] = value
        return em.retyped(value, access.dtype)

    def _lowered(self, node: Node, coords: Coords) -> str:
        from torch._inductor.virtualized import V
        from torch.fx.node import map_arg

        from ..ast_extension import expr_from_string
        from ..aten_lowering import AtenLowering
        from ..aten_lowering import LoweringContext
        from ..inductor_lowering import PointwiseLowering

        lowering = node.meta.get("lowering")
        em = self.emitter
        shape = _shape(node)
        env: dict[Node, object] = {}

        def visit(value: Node) -> Node:
            if value not in env:
                env[value] = expr_from_string(
                    self.evaluate(value, _broadcast(coords, shape, _shape(value)))
                )
            return value

        map_arg((node.args, {**node.kwargs, "_extra_deps": None}), visit)
        ctx = LoweringContext.__new__(LoweringContext)
        ctx.cg = em.cg
        ctx.env = env  # pyrefly: ignore[bad-assignment]
        statements: list[ast.AST] = []
        with em.cg.set_statements(statements), V.set_current_node(node):
            if isinstance(lowering, PointwiseLowering):
                result = lowering.codegen_from_input_asts(
                    ctx, node, lowering.input_asts(ctx, node)
                )
            elif isinstance(lowering, AtenLowering):
                result = lowering.codegen(ctx, node)
            else:
                raise _decline(f"{node.name} is not a pointwise value")
        if not isinstance(result, ast.AST):
            raise _decline(f"invalid pointwise result for {node.name}")
        for statement in statements:
            for line in ast.unparse(ast.fix_missing_locations(statement)).splitlines():
                em.emit(line)
        return em.bind("md_v", ast.unparse(result))


def _load_args(address: str, pred: str | None, readonly: bool) -> str:
    """The argument list of a per-lane global load: the address, the Int32
    predicate of a ``*_pred`` form, or ``nc=True`` for the read-only
    (``ld.global.nc``) form of an unpredicated load."""
    if pred is not None:
        return f"{address}, {pred}"
    return f"{address}, nc=True" if readonly else address


def _float_literal(value: float) -> str:
    """Python source for a float constant: ``repr`` for a finite value;
    ``inf``, ``-inf`` and ``nan`` repr to bare names (undefined in the
    generated module), so they are spelled as ``float(...)`` calls."""
    if math.isinf(value):
        return "float('inf')" if value > 0 else "float('-inf')"
    if math.isnan(value):
        return "float('nan')"
    return repr(value)


def _frag_dtype(node: Node) -> torch.dtype | None:
    """The 16-bit dtype a fragment leaf's FX value has (a rounded ``aten.mm``
    or a 16-bit-typed derived value): its fp32 register is re-typed on every
    read.  None for an fp32 fragment."""
    rounded = rounded_dot_dtype(node)
    if rounded is not None:
        return rounded
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor) and val.dtype in (
        torch.bfloat16,
        torch.float16,
    ):
        return val.dtype
    return None


class _Lane(NamedTuple):
    """One element position of a pair: tile-local coordinates, the packet
    element (a 2-D tile leaf) and the fragment register it reads."""

    coords: Coords
    element: int | None
    reg: tuple[int, int, int] | None


_PACKED_OPS: dict[object, str] = {
    torch.ops.aten.mul.Tensor: "mul",
    torch.ops.aten.add.Tensor: "add",
    torch.ops.aten.sub.Tensor: "sub",
}
# Ops the compiler fuses an fp32 product into (an FMA rounds once): a
# product one of them reads, or a sum over a product, stays scalar so the
# packed op's per-lane rounding matches the scalar lowering's.
_ADDERS = (
    torch.ops.aten.add.Tensor,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.sum.dim_IntList,
    torch.ops.aten.sum.default,
    torch.ops.aten.rsub.Scalar,
    torch.ops.aten.addcmul.default,
)


class _RingAlias(NamedTuple):
    """A vector load served from a ring tile's slot (see
    ``_Emitter._ring_aliases``): the tile, its loop, whether the vector's
    constant index is the tile's row (else its column) and that index."""

    tile: _Tile
    loop: int
    fixed_row: bool
    fixed: int


def _ring_alias(
    plan: MmaDagPlan,
    access: Access,
    loop: int,
    tile: _Tile,
    load: Access,
    rows_pos: int,
) -> _RingAlias | None:
    """``access`` (a vector load) as a row or column of the ring tile
    ``tile`` staged for ``load`` in loop ``loop``, or None (see
    ``_Emitter._ring_aliases`` for the conditions)."""
    if (
        access.name != load.name
        or access.strides != load.strides
        or access.dtype != load.dtype
        or len(access.subscripts) != len(load.subscripts)
        or len(load.tile_dims) != 2
        or plan.program.level(access) != loop
    ):
        return None
    fixed: tuple[int, int] | None = None
    shared: int | None = None
    tile_position = 0
    for own, staged in zip(access.subscripts, load.subscripts, strict=True):
        if own.window is not None or staged.window is not None:
            return None
        if staged.kind == "axis":
            axis = staged.axis
            assert axis is not None
            if staged.offset is not None or own.offset is not None:
                return None
            if axis.kind == AXIS_LEAD:
                if own.kind != "axis" or own.axis is not axis:
                    return None
                continue
            position = tile_position
            tile_position += 1
            if own.kind == "axis":
                if own.axis is not axis or shared is not None or plan.ragged(axis):
                    return None
                shared = position
                continue
            if (
                own.loop_dependent
                or axis.kind != AXIS_RDIM
                or not isinstance(own.expr, sympy.Integer)
                or fixed is not None
            ):
                return None
            value = int(own.expr)
            if not 0 <= value < axis.extent or value >= plan.size(axis):
                return None
            fixed = (position, value)
            continue
        if (
            own.kind != "scalar"
            or own.expr != staged.expr
            or own.loops != staged.loops
            or own.loop_dependent != staged.loop_dependent
        ):
            return None
    if fixed is None or shared is None:
        return None
    position, value = fixed
    rows, cols = (tile.rows, tile.cols)
    if position == rows_pos:
        if value >= rows:
            return None
        return _RingAlias(tile, loop, True, value)
    if value >= cols:
        return None
    return _RingAlias(tile, loop, False, value)


def _packable(node: Node) -> bool:
    """Whether ``node`` is an fp32 ``mul`` / ``add`` / ``sub`` of two tile
    values that renders bit-for-bit as a packed f32x2 op: both inputs are
    fp32 tensors (no scalar, no alpha), and the scalar lowering had no FMA
    to form around it -- a product feeding no adder, a sum / difference of
    no product."""
    from ...language import _tracing_ops

    if node.op != "call_function" or node.target not in _PACKED_OPS:
        return False
    if node.kwargs or len(node.args) != 2:
        return False
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor) or val.dtype != torch.float32:
        return False
    tensors = 0
    for arg in node.args:
        if not isinstance(arg, torch.fx.Node):
            return False
        arg_val = arg.meta.get("val")
        if isinstance(arg_val, torch.Tensor):
            if arg_val.dtype != torch.float32:
                return False
            tensors += 1
        elif not isinstance(arg_val, (float, torch.SymFloat)):
            return False
    if not tensors:
        return False

    def through(current: Node) -> Node:
        while _is_wrapper(current, _tracing_ops) or _is_view(current):
            current = cast("Node", current.args[0])
        return current

    if node.target is torch.ops.aten.mul.Tensor:
        pending = list(node.users)
        while pending:
            user = pending.pop()
            if _is_wrapper(user, _tracing_ops) or _is_view(user):
                pending.extend(user.users)
            elif user.op == "call_function" and user.target in _ADDERS:
                return False
        return True
    return not any(
        through(cast("Node", arg)).target is torch.ops.aten.mul.Tensor
        for arg in node.args
    )


def _has_packable(node: Node, memo: dict[Node, bool]) -> bool:
    """Whether a packable op sits under ``node`` (through views and
    pointwise ops; stops at loads, fragments and anything else)."""
    if node in memo:
        return False if memo[node] is None else memo[node]
    memo[node] = None  # type: ignore[assignment]  # (cycle guard)
    result = _packable(node)
    if not result and node.op == "call_function":
        from ...language import _tracing_ops
        from ...language import memory_ops

        if node.target not in (memory_ops.load, _tracing_ops._get_symnode) and (
            _is_pointwise(node)
            or node.target
            in (
                torch.ops.aten.permute.default,
                torch.ops.aten.transpose.int,
                torch.ops.aten.expand.default,
                torch.ops.prims.convert_element_type.default,
            )
            or _is_wrapper(node, _tracing_ops)
        ):
            result = any(_has_packable(arg, memo) for arg in _node_inputs(node))
    memo[node] = result
    return result


_COORD_FORMS = (
    re.compile(r"^(md_\w+)$"),
    re.compile(r"^(md_\w+) \+ (\d+)$"),
    re.compile(r"^(\d+) \+ (md_\w+)$"),
)


def _split_coord(coord: str) -> tuple[str | None, int]:
    """A tile-local coordinate as ``(lane term, immediate)``: ``md_x``,
    ``(md_x + 8)`` and ``(16 + md_x)`` split, a bare integer is all
    immediate, anything else is one opaque lane term."""
    text = coord.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    if text.isdigit():
        return None, int(text)
    for form in _COORD_FORMS:
        match = form.match(text)
        if match is None:
            continue
        groups = match.groups()
        if len(groups) == 1:
            return groups[0], 0
        if groups[0].isdigit():
            return groups[1], int(groups[0])
        return groups[0], int(groups[1])
    return coord, 0


def _subscript_in(coords: Coords, index: list[object]) -> Coords:
    """Coordinates of the base of a ``subscript`` view given the view's."""
    return tuple(
        coord for coord, item in zip(coords, index, strict=True) if item is not None
    )


_VIEWS = (
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.expand.default,
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.dim,
)


def _is_view(node: Node) -> bool:
    """A view of its first input (the compiler fuses a product into an add
    through it just the same)."""
    from ...language import view_ops

    return node.op == "call_function" and (
        node.target in _VIEWS or node.target is view_ops.subscript
    )


def _is_wrapper(node: Node, tracing_ops: object) -> bool:
    return node.op == "call_function" and node.target in (
        tracing_ops._new_var,  # pyrefly: ignore[missing-attribute]
        tracing_ops._mask_to,  # pyrefly: ignore[missing-attribute]
    )


def _shape(node: Node) -> tuple[object, ...]:
    val = node.meta.get("val")
    if isinstance(val, torch.Tensor):
        return tuple(val.shape)
    return ()


def _broadcast(
    coords: Coords, out_shape: tuple[object, ...], in_shape: tuple[object, ...]
) -> Coords:
    """The input's coordinates under right-aligned broadcasting."""
    offset = len(out_shape) - len(in_shape)
    return tuple(
        "0" if (isinstance(size, int) and size == 1) else coords[offset + i]
        for i, size in enumerate(in_shape)
    )


def _permute_in(node: Node, coords: Coords) -> Coords:
    """Coordinates of the input of a permute given the output's."""
    ndim = len(coords)
    if node.target is torch.ops.aten.permute.default:
        perm = [int(p) % ndim for p in cast("list[int]", node.args[1])]
    else:
        a, b = (
            int(cast("int", node.args[1])) % ndim,
            int(cast("int", node.args[2])) % ndim,
        )
        perm = list(range(ndim))
        perm[a], perm[b] = perm[b], perm[a]
    result = [""] * ndim
    for out_pos, in_pos in enumerate(perm):
        result[in_pos] = coords[out_pos]
    return tuple(result)


class _Packet:
    """One 16-byte packet a thread owns in a tile's round-robin over the CTA.

    ``prow`` / ``pch`` are the packet's row and chunk; ``pr`` the row's
    lane-dependent part and ``row_off`` its round offset (a Python int, so
    addresses fold it into an immediate), ``srow`` the row the swizzle may
    use (``pr`` when every round keeps the row residue) and ``swz`` the
    swizzled chunk of the owning tile."""

    def __init__(
        self, idx: str, prow: str, pr: str, row_off: int, srow: str, pch: str, swz: str
    ) -> None:
        self.idx, self.prow, self.pr, self.row_off = idx, prow, pr, row_off
        self.srow, self.pch, self.swz = srow, pch, swz

    def offset(self, row_bytes: int) -> str:
        """Byte offset of the packet's row at ``row_bytes`` per row."""
        if self.row_off:
            return f"{self.pr} * {row_bytes} + {self.row_off * row_bytes}"
        return f"{self.pr} * {row_bytes}"


class _Hoist:
    """Where a loop's loop-invariant side loads go: the lines right ahead of
    its ``for`` (``position`` into the emitter's lines, at ``indent``),
    ``depth`` enclosing loops, and the loads hoisted there so far by
    ``(node, coords)``: their fp32 value and the value typed as loaded."""

    def __init__(
        self,
        position: int,
        indent: str,
        depth: int,
        loop: int,
        scope_path: tuple[int, ...] = (),
    ) -> None:
        self.position, self.indent, self.depth, self.loop = (
            position,
            indent,
            depth,
            loop,
        )
        self.body_indent = indent + "    "
        self.loads: dict[tuple[Node, Coords], tuple[str, str]] = {}
        # The block the loop sits in (values hoisted ahead of it are visible
        # to every block inside the loop).
        self.scope_path = scope_path


@dataclasses.dataclass(frozen=True)
class _TmaMap:
    """One host-built tensor map of the warp mode's TMA ring: the kernel
    parameter, the tensor dims in TMA order and the box (same order)."""

    index: int
    param: str
    access: Access
    dims: tuple[int, ...]
    box: tuple[int, ...]
    swizzle: str


class _Tile:
    """A shared operand tile: where it lives and how it is shaped."""

    def __init__(
        self,
        name: str,
        offset: int,
        rows: int,
        cols: int,
        elem_bytes: int,
        slots: int,
    ) -> None:
        self.name = name
        self.offset = offset
        self.rows, self.cols = rows, cols
        self.row_bytes = cols * elem_bytes
        self.cpr = self.row_bytes // 16
        self.bytes = rows * self.row_bytes
        self.slots = slots
        # The raw source tile of a converted operand (``mma_dag_raw_slots``
        # ring slots, addressed by the loop's raw stage).
        self.raw: _Tile | None = None
        self.is_raw = False

    @property
    def packets(self) -> int:
        return self.rows * self.cpr

    def base(self, stage: str) -> str:
        if self.slots == 1 or stage == "0":
            return self.name
        return f"({self.name} + ({stage}) * {self.bytes})"


class _Emitter:
    # Whether a windowed tile's rows at and past its limit hold memory the
    # reads must mask (the tcgen05 renderers' TMA boxes land the window's
    # whole rows); the warp emitter's packets and per-lane loads zero-fill
    # them as they are staged.
    masked_tile_windows = False

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
        # Loop bounds read from tensors (``program.scalar_loads``) -> the
        # Int32 holding the value (defined with the root iteration bases).
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

    # -- rendering primitives ---------------------------------------------

    def emit(self, line: str) -> None:
        if self.redirect is not None:
            self.hoist(self.redirect, line)
            return
        self.lines.append(f"{self.indent}{line}")

    def bind(self, prefix: str, expr: str) -> str:
        name = self.df.new_var(prefix)
        self.define(name, expr)
        return name

    def define(self, name: str, expr: str) -> None:
        """``name = expr`` at the current loop depth (the hoist's when the
        line is redirected ahead of a loop)."""
        self.emit(f"{name} = {expr}")
        self.name_depth[name] = (
            self.redirect.depth if self.redirect is not None else len(self.values)
        )

    def bind_pair(self, prefix: str, expr: str) -> tuple[str, str]:
        """``a, b = expr`` (a packed f32x2 op's two lanes)."""
        first, second = self.df.new_var(prefix), self.df.new_var(prefix)
        self.emit(f"{first}, {second} = {expr}")
        self.name_depth[first] = self.name_depth[second] = len(self.values)
        return first, second

    def hoist_level(self, access: Access, coords: Coords) -> _Hoist | None:
        """The outermost loop being emitted that a per-thread scalar load of
        ``access`` at ``coords`` may be issued ahead of: a loop nested in
        the one the access varies with (any, for a loop-invariant access)
        whose ``for`` every name of the address -- the iteration base and
        the coordinates -- was bound ahead of, at a site in the loop body
        proper (not a guarded nest), of a tensor no store of the program
        writes (a loop may read-modify-write it).  None keeps the load at
        its site."""
        if not self.hoists or self.indent != self.hoists[-1].body_indent:
            return None
        if _storage(access) in self.stored_storages:
            return None
        names = {
            name
            for text in (*coords, self.iter_vars[access.node])
            for name in re.findall(_IDENTIFIER, text)
        }
        if any(name not in self.name_depth for name in names):
            return None
        level = self.program.level(access)
        stack = [hoist.loop for hoist in self.hoists]
        if level >= 0 and level not in stack:
            return None
        start = stack.index(level) + 1 if level >= 0 else 0
        for hoist in self.hoists[start:]:
            if all(self.name_depth[name] <= hoist.depth for name in names):
                return hoist
        return None

    def hoist(self, hoist: _Hoist, line: str) -> None:
        """Insert ``line`` ahead of ``hoist``'s loop."""
        self.insert_line(hoist, line)

    def insert_line(self, anchor: _Hoist, line: str) -> None:
        """Insert ``line`` at ``anchor`` (a loop's hoist point, an aux issue
        anchor or a loop's ``for`` line), keeping every later anchor's
        position current."""
        position = anchor.position
        self.lines.insert(position, f"{anchor.indent}{line}")
        anchors = [
            *self.hoists,
            *(a for v in self.aux_anchors.values() for a, _, _ in v),
            *self.loop_entries,
        ]
        for later in anchors:
            if later is not anchor and later.position >= position:
                later.position += 1
        anchor.position += 1

    def nest(self, header: str) -> _Nest:
        return _Nest(self, header)

    def reg(self, slot: int, am: int, an: int, j: int) -> str:
        return f"md_c{slot}_{am}_{an}_{j}"

    def vreg(self, slot: int, atom: int, half: int) -> str:
        """Vector register of slot ``slot`` for the lane's row ``g`` (``half``
        0) / ``g + 8`` (1) of atom row ``atom`` (a row vector), its column
        ``2c`` / ``2c + 1`` of atom column ``atom`` (a column vector), or the
        one register of a scalar."""
        return f"md_r{slot}_{atom}_{half}"

    def typed(self, value: str, dtype: torch.dtype) -> str:
        """``value`` (an fp32 scalar) re-typed to the DSL type of ``dtype``."""
        if dtype == torch.float32:
            return value
        return f"{self.env.backend.dtype_str(dtype)}({value})"

    def retyped(self, value: str, dtype: torch.dtype) -> str:
        """``typed`` bound to a name, with ``value`` (exactly the 16-bit
        value in fp32) on record as its widening: the pointwise lowering's
        ``to_dtype(name, float32)`` reads ``value`` instead of casting the
        name back (a ``cvt`` pair per element otherwise, the 16-bit one on
        the conversion pipe)."""
        if dtype == torch.float32:
            return value
        name = self.bind("md_v", self.typed(value, dtype))
        self.df.cute_state.mma_dag_exact_f32[name] = value
        return name

    @staticmethod
    def zero(dtype: torch.dtype) -> str:
        """A register's initial value for a side load of ``dtype``."""
        return "cutlass.Int32(0)" if dtype == torch.int32 else "cutlass.Float32(0.0)"

    def readonly(self, access: Access) -> bool:
        """Whether no store of the program writes the tensor ``access`` reads:
        its loads may take the read-only (``ld.global.nc``) path, which is
        coherent across launches and 3-4 % faster on the side loads of the
        state recurrences (fwd_h at s1 shapes: 2.69 against 2.79 us)."""
        return _storage(access) not in self.stored_storages

    def scalar_load(
        self,
        address: str,
        dtype: torch.dtype,
        pred: str | None = None,
        *,
        readonly: bool = False,
    ) -> str:
        """One element at ``address`` as an fp32 / int32 scalar; under an
        Int32 ``pred`` (see ``pred_int``) a zero where it is 0, else the
        read-only (``ld.global.nc``) form when ``readonly`` (the predicated
        loads have no ``.nc`` form)."""
        suffix = "" if pred is None else "_pred"
        args = _load_args(address, pred, readonly)
        if dtype == torch.float32:
            return f"{_RUNTIME}.ldg_f32{suffix}({args})"
        if dtype == torch.int32:
            return f"{_RUNTIME}.ldg_s32{suffix}({args})"
        if dtype == torch.float16:
            return f"{_RUNTIME}.ldg_f16_as_f32{suffix}({args})"
        assert dtype == torch.bfloat16, dtype
        return f"{_RUNTIME}.ldg_bf16_as_f32{suffix}({args})"

    def pair_load(
        self,
        address: str,
        dtype: torch.dtype,
        paired: bool,
        pred: str | None = None,
        *,
        readonly: bool = False,
    ) -> tuple[str, str]:
        """Two adjacent elements at ``address`` as fp32 scalars (zeros
        under a false Int32 ``pred``)."""
        lo, hi = self.df.new_var("md_lo"), self.df.new_var("md_hi")
        suffix = "" if pred is None else "_pred"
        args = _load_args(address, pred, readonly)
        if not paired:
            self.emit(
                f"{lo} = {self.scalar_load(address, dtype, pred, readonly=readonly)}"
            )
            self.emit(
                f"{hi} = {self.scalar_load(f'{address} + {dtype.itemsize}', dtype, pred, readonly=readonly)}"
            )
        elif dtype == torch.float32:
            self.emit(f"{lo}, {hi} = {_RUNTIME}.ldg_v2_f32{suffix}({args})")
        else:
            unpack = "unpack_f16x2" if dtype == torch.float16 else "unpack_bf16x2"
            self.emit(
                f"{lo}, {hi} = {_RUNTIME}.{unpack}({_RUNTIME}.ldg_b32{suffix}({args}))"
            )
        return lo, hi

    def constant(self, value: object, dtype: torch.dtype) -> str:
        """A scalar constant of the DSL type of ``dtype`` (a ``_mask_to``
        fill)."""
        if dtype == torch.int32:
            return f"cutlass.Int32({int(cast('int', value))})"
        if dtype == torch.bool:
            return f"cutlass.Boolean({bool(value)})"
        if not dtype.is_floating_point:
            raise _decline(f"no {dtype} fill for a masked value")
        return self.typed(
            f"cutlass.Float32({_float_literal(float(cast('float', value)))})", dtype
        )

    def pair_store(
        self, address: str, lo: str, hi: str, dtype: torch.dtype, paired: bool
    ) -> None:
        if not paired:
            scalar = {
                torch.float32: "stg_f32",
                torch.float16: "stg_f16",
                torch.bfloat16: "stg_bf16",
            }[dtype]
            self.emit(f"{_RUNTIME}.{scalar}({address}, {lo})")
            self.emit(f"{_RUNTIME}.{scalar}({address} + {dtype.itemsize}, {hi})")
        elif dtype == torch.float32:
            self.emit(f"{_RUNTIME}.stg_v2_f32({address}, {lo}, {hi})")
        else:
            pack = "pack_f16x2" if dtype == torch.float16 else "pack_bf16x2"
            self.emit(f"{_RUNTIME}.stg_b32({address}, {_RUNTIME}.{pack}({lo}, {hi}))")

    def pack(self, lo: str, hi: str, dtype: torch.dtype | None = None) -> str:
        """Two fp32 values rounded and packed into one 16-bit pair (of the
        program's operand dtype when none is given)."""
        if dtype is None:
            dtype = self.program.dtype
        pack = "pack_f16x2" if dtype == torch.float16 else "pack_bf16x2"
        return f"{_RUNTIME}.{pack}(cutlass.Float32({lo}), cutlass.Float32({hi}))"

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

    def set_tile_leaf(self, load: Access, element: int, raw: str) -> None:
        """Bind element ``element`` of the loaded tile: ``raw`` is its exact
        fp32 value (an unpacked packet half or a widening load), the typed
        leaf the DSL value of the tile's dtype."""
        key = (load.node, element)
        self.evaluator.raw_leaves[key] = raw
        self.evaluator.tile_leaves[key] = self.retyped(raw, load.dtype)

    def pairs_aligned(self, access: Access, cols_pos: int) -> bool:
        """Whether adjacent elements along tile dim ``cols_pos`` may move as
        one aligned pair (contiguous columns, pair-aligned strides and base)."""
        from .memory_ops import cute_tensor_base_is_aligned

        es = access.dtype.itemsize
        if access.tile_dim_strides[cols_pos] != 1:
            return False
        if not cute_tensor_base_is_aligned(self.env, access.fake, 2 * es):
            return False
        return all(
            (stride * es) % (2 * es) == 0 for stride in access.strides if stride != 1
        )

    # -- loops ----------------------------------------------------------------

    def loop_index_var(self, loop: int) -> str:
        return "md_i" if loop == 0 else f"md_i{loop}"

    def stage_var(self, loop: int) -> str:
        return "md_stage" if loop == 0 else f"md_stage{loop}"

    def parity_var(self, loop: int) -> str:
        return "md_parity" if loop == 0 else f"md_parity{loop}"

    def enclosing(self, region: int) -> list[int]:
        """Region ``region`` and the loops enclosing it, outermost first."""
        return self.program.enclosing_loops(region)[::-1]

    # -- index expressions ---------------------------------------------------

    def uniform(self, expr: sympy.Expr, values: dict[int, str] | str) -> str:
        """Render a uniform scalar subscript with the grid loops' indices as
        ``values`` (loop index -> value expression; a string is the single
        loop's value)."""
        if isinstance(values, str):
            values = {0: values}
        symbols = {
            loop.symbol: index
            for index, loop in enumerate(self.program.loops)
            if loop.symbol is not None
        }
        replacements: dict[sympy.Symbol, sympy.Symbol] = {}
        for symbol in expr.free_symbols:
            symbol = cast("sympy.Symbol", symbol)
            if symbol in symbols:
                replacements[symbol] = sympy.Symbol(
                    values[symbols[symbol]], integer=True
                )
            else:
                replacements[symbol] = sympy.Symbol(
                    self.df.sympy_expr(symbol), integer=True
                )
        return self.env.backend.sympy_printer_expr(expr.xreplace(replacements))

    def bound(self, value: object) -> str:
        if isinstance(value, torch.fx.Node):
            if value in self.scalar_vars:
                return self.scalar_vars[value]
            val = value.meta["val"]
            if isinstance(val, int):
                return str(val)
            return self.df.sympy_expr(val._sympy_())
        return str(value)

    def define_scalar_load(self, access: Access) -> None:
        """A uniform scalar read from a tensor (a loop bound, a window's
        offset or limit) as an Int32: one load at the access's base (its
        only address: a 0-d access has no tile dims).  Defined with the
        bases, ahead of the accesses that window on it and of every loop."""
        if access.node in self.scalar_vars:
            return
        base = self.base_vars[access.node]
        # An int64 table value is narrowed to Int32: the value is a row
        # index or a row count of a tensor dim (a static extent, below
        # 2**31), so the narrowing is exact for every in-range value; a
        # value past 2**31 is not representable and is not guarded here.
        load = (
            f"{_RUNTIME}.ldg_s32({base})"
            if access.dtype == torch.int32
            else f"cutlass.Int32({_RUNTIME}.ldg_s64({base}))"
        )
        self.scalar_vars[access.node] = self.bind("md_scalar", load)

    def window_guard(self, access: Access, position: int, coord: str) -> str | None:
        """The DSL Boolean saying tile-local ``coord`` of value dim
        ``position`` of ``access`` is a row of its window: below the limit
        and inside the tensor dim (None: the dim is not windowed)."""
        window = access.window_at(position)
        if window is None:
            return None
        limit = self.scalar_vars[window.limit.node]
        offset = self.scalar_vars[window.offset.node]
        return f"({coord} < {limit} and {offset} + {coord} < {window.extent})"

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

    def axis_base(self, axis: Axis) -> str:
        """The CTA's first index along ``axis`` (loop axes: per iteration)."""
        return f"md_a{axis.index}"

    def is_split_axis(self, axis: Axis) -> bool:
        """Whether a thread-block cluster splits ``axis`` (a full-slice axis
        whose per-CTA slice starts at the cluster rank's offset)."""
        return self.plan.cluster > 1 and axis.index == self.plan.split_axis

    def define_axis_bases(self) -> None:
        for axis in self.program.axes:
            if self.is_split_axis(axis):
                # The trailing grid dim indexes the cluster's CTAs: this
                # CTA's slice of the split axis.
                self.emit(
                    f"{self.axis_base(axis)} = cutlass.Int32(cute.arch.block_idx()[1]) "
                    f"* {self.plan.size(axis)}"
                )
                continue
            if axis.kind in (AXIS_LOOP, AXIS_RDIM):
                continue
            self.define(
                self.axis_base(axis),
                f"cutlass.Int32({self.cg.tile_begin_var(axis.block_id)})",
            )
        self.define_trip_counts()

    # -- ragged root tiles ---------------------------------------------------
    #
    # A root tile may overshoot its axis extent (``mma_dag_config_reason``
    # admits it for the axes ``ragged_root_axes`` names): the rows past the
    # extent exist in the fragments and the shared tiles but not in memory.
    # The operand staging zero-fills them, the side loads read zero under a
    # predicate, ``_mask_to`` takes its fill there and the stores are
    # guarded -- all keyed by the element's tile-local coordinate along the
    # axis against the CTA's first index.

    def in_bounds(self, axis: Axis, coord: str) -> str:
        """The DSL Boolean saying tile-local ``coord`` of ragged ``axis`` is
        a row the CTA owns: below the block when the fragment is padded
        past it, inside the extent when the block does not divide it."""
        block = self.plan.block(axis)
        terms = []
        if block < self.plan.size(axis):
            terms.append(f"{coord} < {block}")
        if axis.extent % block:
            terms.append(f"{self.axis_base(axis)} + {coord} < {axis.extent}")
        assert terms, "in_bounds of an axis that is not ragged"
        return f"({' and '.join(terms)})"

    def bounds(self, access: Access, coords: Coords) -> str | None:
        """The Boolean saying the element of ``access`` at ``coords`` exists
        (None: every element does -- no dim of the access is ragged or
        windowed)."""
        terms = [
            self.in_bounds(axis, coords[position])
            for position, axis in enumerate(access.dims)
            if axis.kind != AXIS_LEAD and self.plan.ragged(axis)
        ]
        for position, _window in access.windows:
            terms.append(
                cast("str", self.window_guard(access, position, coords[position]))
            )
        return " and ".join(terms) if terms else None

    def axis_of_dim(self, node: Node, position: int) -> Axis | None:
        """The program axis behind dim ``position`` of ``node`` (None for a
        size-1 dim or a dim no axis tiles)."""
        size = _shape(node)[position]
        block_id = self.env.resolve_block_id(size)
        if block_id is None:
            return None
        block_id = self.env.canonical_block_id(block_id)
        return next(
            (axis for axis in self.program.axes if axis.block_id == block_id), None
        )

    def value_bounds(self, node: Node, coords: Coords) -> str | None:
        """The Boolean saying the element of value ``node`` at ``coords``
        lies inside every ragged axis among its dims (None: no dim is)."""
        terms = []
        for position in range(len(_shape(node))):
            axis = self.axis_of_dim(node, position)
            if axis is not None and self.plan.ragged(axis):
                terms.append(self.in_bounds(axis, coords[position]))
        return " and ".join(terms) if terms else None

    @staticmethod
    def pred_int(bounds: str) -> str:
        """``bounds`` (a DSL Boolean) as the Int32 operand of the predicated
        runtime loads (``ldg_*_pred``)."""
        return f"cutlass.Int32(1) if {bounds} else cutlass.Int32(0)"

    def bounds_pred(self, access: Access, coords: Coords) -> str | None:
        """The Int32 predicate of a per-lane load of ``access`` at ``coords``
        bound at the current site (None: unpredicated)."""
        bounds = self.bounds(access, coords)
        if bounds is None:
            return None
        return self.bind("md_ok", self.pred_int(bounds))

    def define_bases(self) -> None:
        """Per-access byte bases: the tensor pointer plus the loop-invariant
        scalar subscripts and the root / leading tile offsets (Int64).  The
        uniform scalar loads come first (their bases are their addresses),
        so the accesses windowing on them can add the offsets they hold."""
        scalar_nodes = {access.node for access in self.program.scalar_loads}
        ordered = sorted(
            self.program.accesses, key=lambda access: access.node not in scalar_nodes
        )
        for access in ordered:
            if access.node in self.base_vars:
                continue
            name = self.df.tensor_arg(access.fake, prefer_name=access.name).name
            self.df.placeholder_args.add(name)
            es = access.dtype.itemsize
            terms = [f"{name}.iterator.toint()"]
            for subscript, stride in zip(
                access.subscripts, access.strides, strict=True
            ):
                scale = stride * es
                if subscript.kind == "axis":
                    axis = subscript.axis
                    assert axis is not None
                    if subscript.offset is not None:
                        terms.append(
                            f"cutlass.Int64({self.uniform(subscript.offset, {})}) * {scale}"
                        )
                    if subscript.window is not None:
                        terms.append(
                            f"cutlass.Int64({self.scalar_vars[subscript.window.offset.node]}) * {scale}"
                        )
                    if axis.kind == AXIS_LOOP or (
                        axis.kind == AXIS_RDIM and not self.is_split_axis(axis)
                    ):
                        continue
                    terms.append(f"cutlass.Int64({self.axis_base(axis)}) * {scale}")
                elif not subscript.loop_dependent:
                    assert subscript.expr is not None
                    terms.append(
                        f"cutlass.Int64({self.uniform(subscript.expr, {})}) * {scale}"
                    )
            self.base_vars[access.node] = self.bind("md_base", " + ".join(terms))
            if access.node in scalar_nodes:
                self.define_scalar_load(access)

    def iteration_base(self, access: Access, values: dict[int, str] | str) -> str:
        """The access base for the iterations whose loop indices are
        ``values`` (loop index -> value; a string is the single loop's
        value); the enclosing loops' current values are cached per access."""
        if isinstance(values, str):
            values = {0: values}
        if values == self.values and access.node in self.iter_vars:
            return self.iter_vars[access.node]
        expr = self.iteration_base_expr(access, values)
        if expr == self.base_vars[access.node]:
            return expr
        return self.bind("md_p", expr)

    def iteration_base_expr(self, access: Access, values: dict[int, str]) -> str:
        """``iteration_base`` as an expression (nothing bound)."""
        terms = [self.base_vars[access.node]]
        es = access.dtype.itemsize
        for subscript, stride in zip(access.subscripts, access.strides, strict=True):
            scale = stride * es
            if subscript.kind == "scalar" and subscript.loop_dependent:
                assert subscript.expr is not None
                terms.append(
                    f"cutlass.Int64({self.uniform(subscript.expr, values)}) * {scale}"
                )
            elif subscript.kind == "axis":
                axis = subscript.axis
                assert axis is not None
                if axis.kind == AXIS_LOOP:
                    step = self.plan.size(axis)
                    offset = self.axis_offset(axis, values[axis.loop])
                    terms.append(f"cutlass.Int64({offset}) * {step * scale}")
        return " + ".join(terms)

    def plain_address(self, access: Access, base: str, coords: Coords) -> str:
        """Byte address of the element at ``coords`` over ``base`` with every
        term spelled out (no lane base bound)."""
        terms = [base]
        es = access.dtype.itemsize
        for position, axis in enumerate(access.dims):
            if axis.kind == AXIS_LEAD:
                continue
            stride = access.dim_strides[position] * es
            coord = coords[position]
            if coord == "0":
                continue
            terms.append(
                f"cutlass.Int64({coord}) * {stride}"
                if stride != 1
                else f"cutlass.Int64({coord})"
            )
        return " + ".join(terms)

    def prefetch_level(self, access: Access, coords: Coords) -> _Hoist | None:
        """The innermost loop being emitted whose per-thread side load of
        ``access`` at ``coords`` is loaded one iteration ahead (a dependent
        global load would otherwise sit on the serial chain of a state
        recurrence): the access varies with that loop alone, the site sits
        in the loop body proper (not a guarded nest), the coordinates' names
        were bound ahead of the loop (a thread's fragment row, a packet row)
        and no store of the program writes the tensor."""
        if self.plan.compute == MMA_DAG_TCGEN05:
            # The tcgen05 emitter prefetches its side loads itself.
            return None
        if not self.hoists or self.indent != self.hoists[-1].body_indent:
            return None
        hoist = self.hoists[-1]
        if hoist.loop < 0 or self.program.level(access) != hoist.loop:
            return None
        loop = self.program.loops[hoist.loop]
        if not loop.innermost or len(loop.axes) > 1:
            return None
        if self.fewest_trips(loop) < 2:
            # Nothing to run ahead of: the refill never executes, and the
            # first iteration's loads hoisted ahead of the ring issue only
            # delay the tiles (chunk_scan s2: 18.3 against 17.2 us).
            return None
        if _storage(access) in self.stored_storages:
            return None
        if access.node in self.ring_aliases:
            # Served from this iteration's ring slot behind the barrier
            # (``_ring_aliases``): not loaded ahead as well.
            return None
        names = {name for text in coords for name in re.findall(_IDENTIFIER, text)}
        if any(
            name not in self.name_depth or self.name_depth[name] > hoist.depth
            for name in names
        ):
            return None
        return hoist

    def fewest_trips(self, loop: Loop) -> int:
        """The smallest trip count any CTA runs of ``loop`` (a dynamic grid
        bound counts as many)."""
        if loop.axes:
            if loop.tile_end is not None:
                (axis,) = loop.axes
                lo, _hi, _aligned = loop_trip_range(
                    loop.tile_end, self.program.axes, self.plan.sizes, axis.index
                )
                return lo
            trips = 1
            for axis in loop.axes:
                trips *= axis.extent // self.plan.size(axis)
            return trips
        trip = loop.static_trip
        return trip if trip is not None else 2

    def prefetch_register(
        self, hoist: _Hoist, access: Access, coords: Coords
    ) -> str | None:
        """The register holding the current iteration's value of the side
        load (the first iteration's load goes ahead of the loop); None once
        the loop's prefetch registers are spent."""
        key = (access.node, coords)
        sites = self.prefetch_sites.setdefault(hoist.loop, {})
        register = sites.get(key)
        if register is None:
            if len(sites) >= MMA_DAG_MAX_PREFETCH:
                return None
            register = self.df.new_var("md_px")
            loop = self.program.loops[hoist.loop]
            begin, _end, _trip = self.loop_bounds(loop)
            base = self.iteration_base_expr(access, {**self.values, hoist.loop: begin})
            address = self.plain_address(access, base, coords)
            self.hoist(
                hoist,
                f"{register} = {self.scalar_load(address, access.dtype, readonly=self.readonly(access))}",
            )
            sites[key] = register
        return register

    def aux_base(self, loop: int) -> str:
        return f"md_aux{loop}"

    def aux_level(self, access: Access, coords: Coords) -> _Hoist | None:
        """The innermost ring loop being emitted whose per-thread side load of
        ``access`` at ``coords`` rides the loop's aux ring (``MMA_DAG_AUX_
        RING``): the prefetch conditions without the trip-count one, the
        loop has an aux region and its slot allocation is still open."""
        if not self.hoists or self.indent != self.hoists[-1].body_indent:
            return None
        hoist = self.hoists[-1]
        if (
            hoist.loop < 0
            or hoist.loop not in self.plan.aux_rings
            or hoist.loop in self.aux_closed
            or self.program.level(access) != hoist.loop
            or access.node in self.shared_aux_nodes
        ):
            return None
        if access.dtype.itemsize not in (2, 4):
            return None
        names = {name for text in coords for name in re.findall(_IDENTIFIER, text)}
        if any(
            name not in self.name_depth or self.name_depth[name] > hoist.depth
            for name in names
        ):
            return None
        return hoist

    def aux_slot(self, hoist: _Hoist, access: Access, coords: Coords) -> int | None:
        """The word slot of the side load (None once the loop's words are
        spent)."""
        key = (access.node, coords)
        sites = self.aux_sites.setdefault(hoist.loop, {})
        slot = sites.get(key)
        if slot is None:
            if len(sites) >= self.plan.aux_rings[hoist.loop].words:
                return None
            slot = sites[key] = len(sites)
        return slot

    def aux_stage_base(self, loop: int, stage: str) -> str:
        ring = self.plan.aux_rings[loop]
        stride = f" + ({stage}) * {ring.stage_bytes}" if stage != "0" else ""
        return f"{self.aux_base(loop)}{stride}"

    def aux_slot_address(self, loop: int, slot: int, stage: str) -> str:
        """Shared address of this thread's word slot ``slot`` at ring
        ``stage``: ``[stage][slot][thread]`` words."""
        threads = self.plan.threads
        return (
            f"{self.aux_stage_base(loop, stage)} + {slot * threads * 4} + md_tidx * 4"
        )

    def aux_run_address(
        self, loop: int, access: Access, packet: str, stage: str
    ) -> str:
        """Shared address of the run of vector ``access`` the packet at chunk
        ``packet`` reads at ring ``stage``."""
        ring = self.plan.aux_rings[loop]
        _elems, run_bytes, offset = ring.runs[access.node]
        return (
            f"{self.aux_stage_base(loop, stage)} + {ring.words * ring.threads * 4 + offset}"
            f" + cutlass.Int32({packet}) * {run_bytes}"
        )

    def aux_read(self, loop: int, slot: int, access: Access, coords: Coords) -> str:
        """The side-load value of the current iteration from its word slot
        (a 16-bit element sits in the half of the word its address selects)."""
        address = self.aux_slot_address(loop, slot, self.stage_var(loop))
        dtype = access.dtype
        if dtype == torch.float32:
            return f"{_RUNTIME}.lds_f32({address})"
        if dtype == torch.int32:
            return f"{_RUNTIME}.lds_s32({address})"
        element = self.plain_address(
            access, self.iteration_base_expr(access, self.values), coords
        )
        # (The low address bits: a signed remainder of the truncated address
        # is negative when bit 31 is set.)
        half = f"(cutlass.Int32({element}) & 3)"
        op = "lds_f16_as_f32" if dtype == torch.float16 else "lds_bf16_as_f32"
        return f"{_RUNTIME}.{op}({address} + {half})"

    def aux_shared_level(self, access: Access) -> int | None:
        """The innermost ring loop being iterated whose aux ring copies
        scalar / vector ``access`` once per CTA into its side tile
        (``AuxRing.shared``; None: none does): read by coordinate at any
        site of the iteration (the slot landed and was published an
        iteration ahead)."""
        if not self.hoists:
            return None
        loop = self.hoists[-1].loop
        ring = self.plan.aux_rings.get(loop)
        if (
            ring is None
            or access.node not in ring.shared
            or self.program.level(access) != loop
            or loop not in self.aux_shared_stage
        ):
            return None
        return loop

    def has_aux_shared(self, loop: int) -> bool:
        """Whether ``loop``'s aux ring has a side tile."""
        ring = self.plan.aux_rings.get(loop)
        return ring is not None and bool(ring.shared)

    def aux_shared_base(self, loop: int, slot: str) -> str:
        """Shared address of side tile slot ``slot`` of ``loop``'s aux ring
        (the slots follow the ring's ``stages`` stages of per-thread words
        and runs)."""
        ring = self.plan.aux_rings[loop]
        base = self.aux_base(loop)
        region = ring.shared_base - ring.offset
        if region:
            base = f"{base} + {region}"
        return f"{base} + {slot} * {ring.shared_bytes}"

    def aux_shared_slot(self, loop: Loop, value: str) -> str:
        """The side tile slot of ``loop``'s iteration ``value``: the
        iteration's ordinal modulo the ``stages + 1`` slots."""
        ring = self.plan.aux_rings[loop.index]
        begin = self.loop_bounds(loop)[0]
        offset = f"({value})" if begin == "0" else f"(({value}) - {begin})"
        return f"{offset} % {ring.shared_slots}"

    def aux_shared_read(self, loop: int, access: Access, coords: Coords) -> str:
        """The current iteration's element of side tile entry ``access`` at
        ``coords`` (its one tile coordinate; a scalar has none) from the
        iteration's slot."""
        ring = self.plan.aux_rings[loop]
        _elements, offset = ring.shared[access.node]
        address = self.aux_shared_base(loop, self.aux_shared_stage[loop])
        if offset:
            address = f"{address} + {offset}"
        if access.tile_dims:
            coord = coords[access.tile_positions[0]]
            if coord != "0":
                address = f"{address} + ({coord}) * 4"
        return f"{_RUNTIME}.lds_f32({address})"

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

    def aux_run_values(self, loop: int, access: Access, packet: str) -> list[str]:
        """The fp32 values of the run of ``access`` the packet at chunk
        ``packet`` reads (one 16-byte shared read per 16 bytes), in element
        order."""
        address = self.aux_run_address(loop, access, packet, self.stage_var(loop))
        elems, run_bytes, _offset = self.plan.aux_rings[loop].runs[access.node]
        values: list[str] = []
        for unit in range(run_bytes // 16):
            at = f"{address} + {unit * 16}" if unit else address
            if access.dtype == torch.float32:
                names = [self.df.new_var("md_v") for _ in range(4)]
                self.emit(f"{', '.join(names)} = {_RUNTIME}.lds_v4_f32({at})")
                values.extend(names)
            else:
                words = [self.df.new_var("md_w") for _ in range(4)]
                self.emit(f"{', '.join(words)} = {_RUNTIME}.lds_v4_b32({at})")
                unpack = (
                    "unpack_f16x2" if access.dtype == torch.float16 else "unpack_bf16x2"
                )
                for word in words:
                    lo, hi = self.df.new_var("md_v"), self.df.new_var("md_v")
                    self.emit(f"{lo}, {hi} = {_RUNTIME}.{unpack}({word})")
                    values.extend((lo, hi))
        assert len(values) == elems
        return values

    def aux_wait(self, loop: int) -> None:
        """This iteration's stage of the aux ring is complete for the
        thread's own word copies (read before the operand passes); the
        CTA's vector copies are published by the barrier ahead of the passes
        (``emit_iteration``)."""
        if loop in self.plan.aux_rings and not self.aux_waited:
            self.emit(f"cute.arch.cp_async_wait_group({max(self.plan.stages - 2, 0)})")
            self.aux_waited = True

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

    def define_iteration_bases(
        self, region: int = -1, *, in_loop: bool | None = None
    ) -> None:
        """Per-access bases of the accesses varying with region ``region``
        (-1: the loop-invariant ones), at the enclosing loops' current
        values.  ``in_loop`` is the single-loop form (every access inside
        the loop, the loop-invariant ones outside) of the tcgen05 emitter."""
        program = self.program
        if in_loop is not None:
            self.iter_vars = {}
            self.values = {0: "md_i"} if in_loop else {}
            for access in program.accesses:
                if not in_loop and program.level(access) >= 0:
                    continue
                self.iter_vars[access.node] = self.iteration_base(access, self.values)
            return
        for access in program.accesses:
            if program.level(access) != region or access.node in self.iter_vars:
                continue
            self.iter_vars[access.node] = self.iteration_base(access, self.values)

    def loop_dependent(self, access: Access) -> bool:
        """Whether the access varies with some device loop."""
        return self.program.level(access) >= 0

    def forget_iteration_bases(self, region: int) -> None:
        """Drop the bases of the accesses varying with loop ``region`` (the
        next iteration defines them anew)."""
        program = self.program
        for access in program.accesses:
            if program.level(access) == region:
                self.iter_vars.pop(access.node, None)

    def address(self, access: Access, coords: Coords) -> str:
        """Byte address of the element at tile-local positional ``coords``:
        the lane base (the access's iteration base plus the lane terms of
        the coordinates, bound once per scope and hoisted ahead of the loops
        it is invariant in) plus the coordinates' immediates -- so the
        registers of one fragment or the rounds of one packet share one
        64-bit base and differ by a constant the address folds.  (A renderer
        that does not fold addresses spells every term out.)"""
        base = self.iter_vars[access.node]
        es = access.dtype.itemsize
        if not self.fold_addresses:
            terms = [base]
            for position, axis in enumerate(access.dims):
                if axis.kind == AXIS_LEAD:
                    continue
                stride = access.dim_strides[position] * es
                coord = coords[position]
                if coord == "0":
                    continue
                terms.append(
                    f"cutlass.Int64({coord}) * {stride}"
                    if stride != 1
                    else f"cutlass.Int64({coord})"
                )
            return " + ".join(terms)
        lane_terms: list[str] = []
        immediate = 0
        for position, axis in enumerate(access.dims):
            if axis.kind == AXIS_LEAD:
                continue
            stride = access.dim_strides[position] * es
            coord = coords[position]
            if coord == "0":
                continue
            lane, offset = _split_coord(coord)
            immediate += offset * stride
            if lane is not None:
                lane_terms.append(
                    f"cutlass.Int64({lane}) * {stride}"
                    if stride != 1
                    else f"cutlass.Int64({lane})"
                )
        if not lane_terms:
            return f"{base} + {immediate}" if immediate else base
        lane_base = self.lane_base(base, tuple(lane_terms))
        return f"{lane_base} + {immediate}" if immediate else lane_base

    def lane_base(self, base: str, lane_terms: tuple[str, ...]) -> str:
        """``base + lane terms`` bound once: reused by every address of the
        same scope (or an enclosing one) built from them, hoisted ahead of
        the loops it is invariant in."""
        key = (base, lane_terms)
        for depth in range(len(self.scope_path), -1, -1):
            found = self.lane_bases.get((*key, tuple(self.scope_path[:depth])))
            if found is not None:
                return found
        expr = " + ".join([base, *lane_terms])
        names = set(re.findall(_IDENTIFIER, expr))
        hoist = self.value_hoist(names)
        if hoist is not None:
            with self.redirected(hoist):
                name = self.bind("md_lb", expr)
            path = hoist.scope_path
        else:
            name = self.bind("md_lb", expr)
            path = tuple(self.scope_path)
        self.lane_bases[(*key, path)] = name
        return name

    def value_hoist(self, names: set[str]) -> _Hoist | None:
        """The outermost loop being emitted that a value of ``names`` (all
        bound ahead of its ``for``) may be computed ahead of, at a site in
        the loop body proper; None keeps it at its site."""
        if not self.hoists or self.indent != self.hoists[-1].body_indent:
            return None
        names = names - _MODULES
        if any(name not in self.name_depth for name in names):
            return None
        for hoist in self.hoists:
            if all(self.name_depth[name] <= hoist.depth for name in names):
                return hoist
        return None

    @contextlib.contextmanager
    def redirected(self, hoist: _Hoist) -> Iterator[None]:
        """Lines emitted inside go ahead of ``hoist``'s loop."""
        previous = self.redirect
        self.redirect = hoist
        try:
            yield
        finally:
            self.redirect = previous

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

    def _is_lead_dim(self, node: Node, position: int) -> bool:
        size = _shape(node)[position]
        block_id = self.env.resolve_block_id(size)
        if block_id is None:
            return False
        block_id = self.env.canonical_block_id(block_id)
        return any(
            axis.block_id == block_id and axis.kind == AXIS_LEAD
            for axis in self.program.axes
        )

    def tile_coords(self, operand: Operand, prow: str, col: str) -> Coords:
        """Positional coordinates of the operand VALUE for the source tile
        element at ``(prow, col)`` (rows / cols of the shared tile)."""
        source = operand.load.node if operand.load is not None else operand.frag
        assert source is not None
        value_shape = _shape(operand.value)
        source_shape = _shape(source)
        # The source's two tile dims in positional order and which is rows.
        positions = [
            i
            for i, size in enumerate(source_shape)
            if not (isinstance(size, int) and size == 1)
            and not self._is_lead_dim(source, i)
        ]
        assert len(positions) == 2
        # The shared tile's rows are the source's non-contiguous tile dim.
        rows_first = operand.load is None or operand.load.stride_at(positions[0]) != 1
        native = [prow, col] if rows_first else [col, prow]
        if operand.swap:
            native.reverse()
        coords = ["0"] * len(value_shape)
        value_positions = [
            i
            for i, size in enumerate(value_shape)
            if not (isinstance(size, int) and size == 1)
            and not self._is_lead_dim(operand.value, i)
        ]
        assert len(value_positions) == 2
        coords[value_positions[0]] = native[0]
        coords[value_positions[1]] = native[1]
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

    def tma_ring_coords(
        self, tmap: _TmaMap, access: Access, values: dict[int, str]
    ) -> list[str]:
        """The block-uniform element coordinates of ``access``'s tile at the
        loop values ``values``, per tensor dim in TMA order (``tmap`` may be
        shared with another access of the tensor: same dims and box)."""
        coords = []
        for dim in tmap.dims:
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
                coords = ", ".join(self.tma_ring_coords(tmap, load, values))
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
        if MMA_DAG_HINTED_CP_ASYNC_CHECK not in state.sass_checks:
            state.sass_checks = (*state.sass_checks, MMA_DAG_HINTED_CP_ASYNC_CHECK)

    def hoist_scalars(self, value: Node) -> None:
        """Evaluate the 0-d (uniform) values under ``value`` once."""
        for node in _scalar_nodes(self.program, value, self.evaluator.inline):
            self.evaluator.evaluate(node, ())

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

    def emit_convert_pass(self, dot: Dot, operand: Operand, stage: str) -> None:
        """Rewrite the thread's own packets of the raw source tile through
        the operand's pointwise DAG into the operand tile of the other width:
        an fp32 source's four elements per packet -> one 8-byte pair of
        16-bit pairs; a 16-bit source's eight elements -> two 16-byte fp32
        packets (a tf32 operand)."""
        load = operand.load
        assert load is not None
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
        for am in range(layout.atoms_m):
            for an in range(layout.atoms_n):
                for half in range(2):
                    row, col = layout.row(am, half), layout.col(an)
                    chunk = f"(({col}) * {es} // 16)"
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

    # -- fragments ----------------------------------------------------------------

    def frag_vec(self, leaf: Node, coords: Coords) -> str:
        """The register of vector ``leaf`` at the fragment element the
        evaluator is at (``evaluator.reg`` of ``evaluator.frag_slot``): the
        vector's own register when its family follows this fragment's warp
        grid and it varies along the dim its orientation names (a row
        vector along the rows: register ``(am, j // 2)``; a column vector
        along the columns: ``(an, j & 1)``), else a copy relaid through the
        scratch tile at the element's coordinate, once per step."""
        evaluator = self.evaluator
        assert evaluator.reg is not None and evaluator.frag_slot is not None
        program = self.program
        frag_slot = evaluator.frag_slot
        vslot = program.vec_leaves[leaf]
        vec = program.vec_slots[vslot]
        rows, cols = program.slots[frag_slot]
        assert vec.axis is not None
        # (Over one axis the dim is positional: the matcher recorded the
        # broadcast the vector was read with.)
        along_rows = (
            program.vec_positions[leaf] == 0 if rows is cols else vec.axis is rows
        )
        am, an, j = evaluator.reg
        # Warp grids are interned per fragment shape (``self.layouts``), so
        # identity here is shape equality: every fragment of the vector's
        # family shape shares the one ``_Layout`` object.
        if (
            self.vec_layouts[vslot] is self.slot_layouts[frag_slot]
            and (vec.orient == "rows") == along_rows
        ):
            if along_rows:
                return self.vreg(vslot, am, j // 2)
            return self.vreg(vslot, an, j & 1)
        key = (leaf, frag_slot)
        names = evaluator.frag_vecs.get(key)
        if names is None:
            names = self.relayout_to_frag(leaf, frag_slot, along_rows)
            evaluator.frag_vecs[key] = names
        return names[evaluator.reg]

    def relayout_to_frag(
        self, leaf: Node, frag_slot: int, along_rows: bool
    ) -> dict[tuple[object, ...], str]:
        """Bring vector ``leaf`` to the register positions of fragment slot
        ``frag_slot`` through the scratch tile: the owning lanes write each
        element, every lane reads the elements of its registers' rows
        (``along_rows``) or columns.  Returns register -> temporary."""
        program = self.program
        vslot = program.vec_leaves[leaf]
        base = self.scratch_base()
        layout = self.slot_layouts[frag_slot]
        prefix = self.df.new_var("md_fv")
        with self.nest(f"if {self.vec_owner(vslot)}:"):
            for atom, half in self.vec_regs(vslot):
                index = self.vec_index(vslot, atom, half)
                self.emit(
                    f"{_RUNTIME}.sts_f32({base} + ({index}) * 4, {self.vreg(vslot, atom, half)})"
                )
        self.emit(self.sync)
        names: dict[tuple[object, ...], str] = {}
        by_index: dict[str, str] = {}
        for am, an, j in layout.registers():
            index = layout.row(am, j // 2) if along_rows else layout.col(an, j & 1)
            name = by_index.get(index)
            if name is None:
                name = f"{prefix}_{len(by_index)}"
                self.emit(f"{name} = {_RUNTIME}.lds_f32({base} + ({index}) * 4)")
                by_index[index] = name
            names[(am, an, j)] = name
        # The scratch is reused by the next vector operation: every lane read.
        self.emit(self.sync)
        return names

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
            # at most) must divide every TMA slot's address.
            alignment = 1024 if self.tma_tx else 128
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

    # -- loops ----------------------------------------------------------------

    def loop_bounds(self, loop: Loop | None = None) -> tuple[str, str, int | None]:
        """``(begin, end, static trip count)`` of the emitted loop: a grid
        loop's own bounds, or the tile loop's iteration count (the single
        loop when omitted)."""
        if loop is None:
            loop = self.program.loops[0]
        if not loop.axes:
            return self.bound(loop.begin), self.bound(loop.end), loop.static_trip
        if loop.tile_end is not None:
            (axis,) = loop.axes
            lo, hi, _aligned = loop_trip_range(
                loop.tile_end, self.program.axes, self.plan.sizes, axis.index
            )
            if lo == hi:
                return "0", str(lo), lo
            return "0", self.trip_var(loop), None
        trip = 1
        for axis in loop.axes:
            trip *= axis.extent // self.plan.size(axis)
        return "0", str(trip), trip

    def trip_var(self, loop: Loop) -> str:
        """The per-CTA trip count of a varying-bound tile loop."""
        return f"md_trips{loop.index}"

    def end_var(self, loop: Loop) -> str:
        """The per-CTA bound of a varying-bound tile loop."""
        return f"md_end{loop.index}"

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

    def render_bound(self, end: sympy.Expr) -> str:
        """A varying loop bound (``Loop.tile_end``) from the CTA's root tile
        bases: ``constant + sum(coefficient * coordinate)``."""
        constant, terms = loop_bound_terms(end, self.plan.sizes)
        parts = [str(constant)]
        for (kind, index), coefficient in terms.items():
            axis = self.program.axes[index]
            if kind == BOUND_ID and axis.kind == AXIS_ROOT:
                coordinate = f"({self.axis_base(axis)} // {self.plan.size(axis)})"
            else:
                coordinate = self.axis_base(axis)
            parts.append(f"{coefficient} * {coordinate}")
        return " + ".join(parts)

    def define_trip_counts(self) -> None:
        """The bounds of the varying-bound tile loops (after the root axis
        bases) and the trip counts of those whose CTAs run different numbers
        of trips."""
        for loop in self.program.loops:
            if loop.tile_end is None:
                continue
            (axis,) = loop.axes
            size = self.plan.size(axis)
            self.emit(
                f"{self.end_var(loop)} = cutlass.Int32({self.render_bound(loop.tile_end)})"
            )
            lo, hi, _aligned = loop_trip_range(
                loop.tile_end, self.program.axes, self.plan.sizes, axis.index
            )
            if lo != hi:
                self.emit(
                    f"{self.trip_var(loop)} = cutlass.Int32(({self.end_var(loop)} + {size - 1}) // {size})"
                )

    def axis_offset(self, axis: Axis, value: str) -> str:
        """The tile index along loop axis ``axis`` at the loop's flat
        iteration ``value`` (a multi-dim tile loop walks its axes in row-major
        order: the last axis fastest)."""
        loop = self.program.loops[axis.loop]
        if len(loop.axes) == 1:
            return value
        trips = [a.extent // self.plan.size(a) for a in loop.axes]
        position = loop.axes.index(axis)
        inner = 1
        for trip in trips[position + 1 :]:
            inner *= trip
        expr = value if inner == 1 else f"({value}) // {inner}"
        if position > 0:
            expr = f"({expr}) % {trips[position]}"
        return expr

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
                del self.values[loop.index]
            self.aux_shared_stage.pop(loop.index, None)
            assert self.loop_entries.pop() is entry
            assert self.lines[entry.position].lstrip().startswith(f"for {var} in ")
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
            if loop.index in self.tma_tx:
                count = self.tma_count(loop.index)
                self.emit(f"{count} = {count} + ({end} - {begin})")
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
            self.emit(f"{self.axis_base(axis)} = {offset} * {self.plan.size(axis)}")
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
        offset = f"{var}" if begin == "0" else f"({var} - {begin})"
        if index in self.tma_tx:
            # Slot and barrier phase of this iteration from the loop's
            # running count (see ``emit_ring_prologue``).
            count = f"({self.tma_count(index)} + {offset})"
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
        # the barrier) stream out as 16-byte packets; the next iteration
        # writes the other buffer.
        for store in staged_stores:
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
        for carry in loop.carries:
            out = program.resolve(carry.output)
            if out is carry.placeholder:
                continue
            source = program.frag_slots[out]
            if source == carry.slot:
                continue
            layout = self.slot_layouts[carry.slot]
            for am, an, j in layout.registers():
                self.emit(
                    f"{self.reg(carry.slot, am, an, j)} = {self.reg(source, am, an, j)}"
                )

    def hoistable_accs(self, loop: Loop) -> list[Dot]:
        """The loop dots whose accumulator may be materialized at the
        iteration top: zeros, or a value of the carry at loop entry whose
        slot no earlier step reads (a carry used as an operand of an earlier
        dot must be read before its rescale overwrites the registers)."""
        program = self.program
        result = []
        steps = loop.steps
        for index, step in enumerate(steps):
            if not isinstance(step, Dot):
                continue
            if step.acc is None:
                result.append(step)
                continue
            if step.acc_frag not in loop.carry_placeholders:
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

    # -- vectors ----------------------------------------------------------------

    def vec_regs(self, slot: int) -> list[tuple[int, int]]:
        """The ``(atom, half)`` registers of a vector slot: a row vector's
        rows ``g`` / ``g + 8`` per atom row, a column vector's columns ``2c``
        / ``2c + 1`` per atom column, a scalar's one register."""
        layout = self.vec_layouts[slot]
        if layout is None:
            return [(0, 0)]
        orient = self.program.vec_slots[slot].orient
        atoms = layout.atoms_m if orient == "rows" else layout.atoms_n
        return [(atom, half) for atom in range(atoms) for half in range(2)]

    def vec_index(self, slot: int, atom: int, half: int) -> str:
        """The vector coordinate (row or column) of register ``(atom, half)``."""
        layout = self.vec_layouts[slot]
        assert layout is not None
        if self.program.vec_slots[slot].orient == "rows":
            return layout.row(atom, half)
        return layout.col(atom, half)

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

    def vec_owner(self, slot: int) -> str:
        """The lane predicate selecting each vector element once: row
        vectors are replicated over the quad (``c``) and the warp columns,
        column vectors over ``g`` and the warp rows, scalars everywhere."""
        layout = self.vec_layouts[slot]
        if layout is None:
            return "md_tidx == 0"
        if self.program.vec_slots[slot].orient == "rows":
            guard = "md_c == 0"
            if layout.wn > 1:
                guard += f" and md_L{layout.index}_wni == 0"
            return guard
        guard = "md_g == 0"
        if layout.wm > 1:
            guard += f" and md_L{layout.index}_wmi == 0"
        return guard

    def scratch_base(self) -> str:
        assert self.scratch is not None, "vector scratch missing from the layout"
        return self.scratch.name

    def emit_vec_init(self, loop: Loop) -> None:
        """Zero the loop's vector carries, or copy an earlier loop's vector
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
                self.emit(f"{self.vreg(carry.slot, atom, half)} = cutlass.Float32(0.0)")

    def emit_reduce(self, step: Reduce) -> None:
        """Sums of a pointwise value over a fragment along its columns (row
        vector) or rows (column vector), or of a vector into a scalar.

        Each lane adds its own registers, the shuffles add the lanes of the
        warp sharing the result (the quad, xor 1 / 2, for rows; the octet,
        xor 4 / 8 / 16, for columns) and the warps along the other split
        meet in the scratch tile (one vector per warp, summed in a fixed
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
                total = self.bind("md_s", f"{total} + {term}")
            shuffle = "lane_group_sum" if rows else "octet_sum"
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
                        else self.bind("md_s", f"{total} + {value}")
                    )
                assert total is not None
                partials[(atom, half)] = total
            # The scratch is reused by the next vector operation: every lane
            # read.
            self.emit(self.sync)
        for (atom, half), partial in partials.items():
            self.emit(f"{self.vreg(step.slot, atom, half)} = {partial}")

    def emit_scalar_reduce(self, step: Reduce) -> None:
        """A vector summed into a scalar: each lane adds its registers, the
        lanes of the warp holding distinct elements (the octet for a row
        vector, the quad for a column vector) add theirs, and the warps
        along the vector's split meet in the scratch tile."""
        program = self.program
        assert step.vec is not None
        source = program.vec_leaves[step.vec]
        layout = self.vec_layouts[source]
        assert layout is not None
        rows = program.vec_slots[source].orient == "rows"
        regs = [self.vreg(source, atom, half) for atom, half in self.vec_regs(source)]
        total = regs[0]
        for term in regs[1:]:
            total = self.bind("md_s", f"{total} + {term}")
        shuffle = "octet_sum" if rows else "lane_group_sum"
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
                    value if total is None else self.bind("md_s", f"{total} + {value}")
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


class _Nest:
    """Indented block: ``with emitter.nest("if x:"):``."""

    def __init__(self, emitter: _Emitter, header: str) -> None:
        self.emitter = emitter
        self.header = header

    def __enter__(self) -> None:
        self.emitter.emit(self.header)
        self.emitter.indent += "    "
        self.emitter.scope_count += 1
        self.emitter.scope_path.append(self.emitter.scope_count)

    def __exit__(self, *exc_info: object) -> None:
        self.emitter.indent = self.emitter.indent[:-4]
        self.emitter.scope_path.pop()


def _transposed_swizzle(row: str, chunk: str, chunks_per_row: int) -> str:
    """Swizzled 16-byte chunk of a transposed fp32 operand tile.

    Its rows are written one element at a time by lanes whose rows differ by
    multiples of 8 (the eight elements of a source packet, a fragment's
    atom columns) at one column: with the operand tiles' row swizzle alone
    they share a bank group, so the chunk is also XOR-ed by ``row // 8``.
    The eight consecutive rows of one ldmatrix start at a multiple of 8 and
    keep their eight distinct groups.
    """
    if chunks_per_row >= 8:
        return f"(({chunk}) ^ (({row}) % 8) ^ ((({row}) // 8) % 8))"
    return _swizzled_chunk(row, chunk, chunks_per_row)


def _staging_swizzle(row: str, chunk: str, chunks_per_row: int, elem_bytes: int) -> str:
    """Swizzled 16-byte chunk of the carry staging tile.

    fp32 fragment pairs are 8-byte lane writes: four lanes cover 32 bytes of
    one row, so XOR-ing the chunk by ``2 * (row % 8)`` keeps the sixteen
    lanes of each half-warp on distinct bank pairs; 16-bit pairs (16 bytes per
    four lanes) use the operand tiles' row swizzle.
    """
    if elem_bytes == 4 and chunks_per_row >= 2:
        return f"(({chunk}) ^ ((2 * ({row})) % {min(8, chunks_per_row)}))"
    return _swizzled_chunk(row, chunk, chunks_per_row)


def _unrounded_select(program: object, source: Node) -> tuple[Node, Node, float] | None:
    """``where(mask, convert_element_type(x, 16-bit), c)`` (or with the
    branches swapped) where ``c`` is a constant exactly representable in the
    16-bit type: ``(mask, rounded value, c)``.  Its pack may round once:
    ``round(where(m, x, c)) == where(m, round(x), c)``."""
    if source.op != "call_function" or source.target is not torch.ops.aten.where.self:
        return None
    mask, a, b = source.args[0], source.args[1], source.args[2]
    if not isinstance(mask, torch.fx.Node):
        return None
    resolve = cast("Callable[[Node], Node]", program.resolve)  # pyrefly: ignore[missing-attribute]

    def rounded(value: object) -> bool:
        if not isinstance(value, torch.fx.Node):
            return False
        value = resolve(value)
        return (
            value.op == "call_function"
            and value.target is torch.ops.prims.convert_element_type.default
            and value.args[1] in (torch.bfloat16, torch.float16)
        )

    def constant(value: object) -> float | None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value) if float(value) in (0.0, 1.0, -1.0) else None
        if not isinstance(value, torch.fx.Node):
            return None
        value = resolve(value)
        if value.op == "call_function" and value.target in (
            torch.ops.aten.full.default,
            torch.ops.aten.scalar_tensor.default,
        ):
            fill = (
                value.args[1]
                if value.target is torch.ops.aten.full.default
                else value.args[0]
            )
            if isinstance(fill, (int, float)) and float(fill) in (0.0, 1.0, -1.0):
                return float(fill)
        return None

    if rounded(a) and constant(b) is not None:
        return mask, cast("Node", a), cast("float", constant(b))
    return None


def _scalar_nodes(
    program: object, value: Node, inline: Collection[Node] = ()
) -> list[Node]:
    """The 0-d (uniform) loads and pointwise values under ``value``, in
    evaluation order (a load's tensor and subscript inputs are not values);
    the walk descends into the fragment leaves in ``inline`` (a pass
    chain's derived values, evaluated through their DAG)."""
    from ...language import _tracing_ops
    from ...language import memory_ops

    order: list[Node] = []
    seen: set[Node] = set()
    frag_slots = cast("dict[Node, int]", program.frag_slots)  # pyrefly: ignore[missing-attribute]
    vec_leaves = cast("dict[Node, int]", program.vec_leaves)  # pyrefly: ignore[missing-attribute]
    aliases = cast("dict[Node, Node]", program.aliases)  # pyrefly: ignore[missing-attribute]

    def visit(node: Node) -> None:
        if node in aliases:
            visit(aliases[node])
            return
        if (
            node in seen
            or node.op != "call_function"
            or (node in frag_slots and node not in inline)
            or node in vec_leaves
        ):
            return
        seen.add(node)
        if node.target is _tracing_ops._get_symnode:
            return
        if (
            node.target is not memory_ops.load
            and node.target is not torch.ops.aten.full.default
            and _tile_index_position(node) is None
        ):
            # A ``tile.index`` value and a constant fill are leaves (their
            # shape inputs are no values); a load's inputs are not values
            # either.
            for arg in node.all_input_nodes:
                visit(arg)
        if not _shape(node):
            order.append(node)

    visit(value)
    return order
