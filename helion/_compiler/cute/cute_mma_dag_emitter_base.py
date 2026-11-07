"""The shared layer of the ``cute_mma_dag`` emitters (warp and tcgen05 modes).

``MmaDagEmitterBase`` holds what the warp emitter (``cute_mma_dag_codegen.py``)
and the tensor-memory emitters (``cute_mma_dag_tcgen05.py``,
``cute_mma_dag_tcgen05_regions.py``) run alike: the text builders (``emit`` /
``bind`` / ``define`` / ``hoist`` / ``nest``), the pointwise evaluator
(``_Evaluator``: Inductor lowerings per element over fragment coordinates),
addressing (``address`` / ``plain_address`` / ``iteration_base`` / ``bounds``),
the per-lane side loads and their register / shared-memory prefetch rings,
the trip counts of the loops, and the vector-fragment layouts.  Its contents
are the methods both tcgen05 emitters reach (measured over the suite's
renders), moved verbatim from the warp emitter; the siblings set up the
attributes declared here in their own constructors.  The MMA, exchange,
pass and store schedules live in the siblings.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import math
import re
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
from .cute_mma_dag import MMA_DAG_TCGEN05
from .cute_mma_dag import Access
from .cute_mma_dag import Axis
from .cute_mma_dag import Loop
from .cute_mma_dag import MmaDagPlan
from .cute_mma_dag import Operand
from .cute_mma_dag import _decline
from .cute_mma_dag import _is_pointwise
from .cute_mma_dag import _node_inputs
from .cute_mma_dag import _storage
from .cute_mma_dag import _tile_index_position
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
    from torch.multiprocessing.reductions import StorageWeakRef

    from ..device_function import DeviceFunction
    from ..generate_ast import GenerateAST
    from .cute_mma_dag import MmaDagProgram

_RUNTIME = "_helion_mma_dag"
# The tcgen05 runtime (mbarriers, bulk tensor copies) the warp mode's TMA
# ring borrows.
_TC = "_helion_mma_dag_tc"
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
# Side-load dtypes loaded widened to fp32 (``ldg_*_as_f32``): the raw fp32
# value stands for the element, the DSL-typed one for the FX value.
_WIDENED: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16)
Coords = tuple[str, ...]


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

    def __init__(self, emitter: MmaDagEmitterBase) -> None:
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
        # positions for the current step (see ``MmaDagEmitterBase.frag_vec``).
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
        elif node in self.program.aux_indices:
            result = em.uniform(self.program.aux_indices[node], em.values)
        elif node in self.program.uniform_values and isinstance(
            node.meta.get("val"), (int, torch.SymInt)
        ):
            val = node.meta["val"]
            if isinstance(val, torch.SymInt):
                expr = val._sympy_()
                assert isinstance(expr, sympy.Expr)
                result = em.df.sympy_expr(expr)
            else:
                result = str(val)
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

            fill = cast("float | int | Node", node.args[1])
            val = node.meta.get("val")
            if (
                node in self.program.uniform_values
                and isinstance(val, torch.Tensor)
                and not val.dtype.is_floating_point
            ):
                # Integer full_like truncates its fill before downstream
                # arithmetic, even when the fill argument is a float. Only
                # uniform bounds use this path: narrow epilogue fills retain
                # their existing literal lowering below.
                result = (
                    em.typed(self.evaluate(fill, ()), val.dtype)
                    if isinstance(fill, torch.fx.Node)
                    else em.constant(fill, val.dtype)
                )
            else:
                if isinstance(val, torch.Tensor) and val.dtype.is_floating_point:
                    fill = float(cast("float | int", fill))
                result = constant_repr(cast("float | int", fill))
        elif node.op == "call_function" and node.target is creation_ops.full:
            # Preserve integer fills exactly and evaluate a scalar tensor fill
            # at its own width before converting it to the result's dtype.
            fill = node.args[1]
            dtype = node.meta["val"].dtype
            value = (
                em.typed(self.evaluate(fill, ()), dtype)
                if isinstance(fill, torch.fx.Node)
                else em.constant(fill, dtype)
            )
            result = em.bind("md_v", value)
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
        if len(access.tile_dims) == 2 and not access.grouped:
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
        if len(access.tile_dims) == 2 and not access.grouped:
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
        if access.grouped:
            # Grouped scales are read at the current logical coordinate.
            # They do not share the affine vector ring/prefetch protocol.
            pred = None if bounds is None else em.bind("md_ok", em.pred_int(bounds))
            value = em.bind(
                "md_v",
                em.scalar_load(
                    address, access.dtype, pred, readonly=em.readonly(access)
                ),
            )
            if access.dtype in _WIDENED:
                self.scopes[-1][("raw", (node, coords))] = value
                return em.retyped(value, access.dtype)
            return em.bind("md_v", em.typed(value, access.dtype))
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
                    em.df.cute_state.root_plans.mma_dag_exact_f32[typed] = value
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

    def _lower_pointwise(
        self, node: Node, coords: Coords
    ) -> tuple[ast.AST, list[ast.AST]]:
        """Inductor's lowering of *node* at *coords*: the result expression and
        the statements it needs, not yet emitted."""
        from torch._inductor.virtualized import V
        from torch.fx.node import map_arg

        from ..ast_extension import expr_from_string
        from ..aten_lowering import AtenLowering
        from ..inductor_lowering import PointwiseLowering
        from .scalar_lowering import scalar_lowering_context

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
        ctx = scalar_lowering_context(em.cg, env)
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
        return result, statements

    def _emit_lowered(self, result: ast.AST, statements: list[ast.AST]) -> str:
        em = self.emitter
        for statement in statements:
            for line in ast.unparse(ast.fix_missing_locations(statement)).splitlines():
                em.emit(line)
        return em.bind("md_v", ast.unparse(result))

    def _lowered(self, node: Node, coords: Coords) -> str:
        result, statements = self._lower_pointwise(node, coords)
        return self._emit_lowered(result, statements)


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


def _reduce_combine(kind: str, lhs: str, rhs: str) -> str:
    """One fp32 reduction step, preserving ``amax``'s NaN propagation."""
    if kind == "sum":
        return f"{lhs} + {rhs}"
    assert kind == "max", kind
    return f"cute.math.max({lhs}, {rhs}, propagate_nan=True)"


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
    ``WarpMmaDagEmitter._ring_aliases``): the tile, its loop, whether the vector's
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
    ``WarpMmaDagEmitter._ring_aliases`` for the conditions)."""
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


TMA_KIND = "helion_mma_dag_tma"


def tma_wrapper_plan(
    df: DeviceFunction,
    specs: Sequence[_TmaMap],
    tensor_name: Callable[[Access], str],
) -> dict[str, object]:
    """The ``helion_mma_dag_tma`` wrapper plan of ``specs`` (the warp mode's
    ``_TmaMap`` or the tcgen05 mode's ``TmaSpec``), declaring each tensor-map
    parameter as a wrapper-only grid constant of the kernel.

    The dict is ``repr``'d into the rendered module, so its insertion order
    -- ``kind``, ``tma_count``, then per map ``name`` / ``param`` / ``dims`` /
    ``box`` / ``swizzle`` / ``elem`` -- is part of the generated text.
    """
    wrapper: dict[str, object] = {"kind": TMA_KIND, "tma_count": len(specs)}
    for spec in specs:
        wrapper[f"tma{spec.index}_name"] = tensor_name(spec.access)
        wrapper[f"tma{spec.index}_param"] = spec.param
        wrapper[f"tma{spec.index}_dims"] = list(spec.dims)
        wrapper[f"tma{spec.index}_box"] = list(spec.box)
        wrapper[f"tma{spec.index}_swizzle"] = spec.swizzle
        wrapper[f"tma{spec.index}_elem"] = spec.access.dtype.itemsize
        df.wrapper_only_params.append(spec.param)
        df.wrapper_only_param_annotations[spec.param] = (
            f"cutlass.GridConstant[{_TC}.cuda.TensorMap]"
        )
    return wrapper


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


class MmaDagEmitterBase:
    # Whether a windowed tile's rows at and past its limit hold memory the
    # reads must mask (the tcgen05 renderers' TMA boxes land the window's
    # whole rows); the warp emitter's packets and per-lane loads zero-fill
    # them as they are staged.
    masked_tile_windows = False

    # The state the subclasses' constructors set up (the warp emitter builds
    # its mma.sync warp grids and tile map on top; the tcgen05 emitters set
    # the shared state by hand and keep their own fragment / barrier state).
    cg: GenerateAST
    df: DeviceFunction
    env: CompileEnvironment
    plan: MmaDagPlan
    program: MmaDagProgram
    evaluator: _Evaluator
    lines: list[str]
    indent: str
    sync: str
    pairs: bool
    fold_addresses: bool
    accesses: dict[Node, Access]
    base_vars: dict[Node, str]
    iter_vars: dict[Node, str]
    scalar_vars: dict[Node, str]
    values: dict[int, str]
    name_depth: dict[str, int]
    scope_path: list[int]
    scope_count: int
    lane_bases: dict[tuple[object, ...], str]
    redirect: _Hoist | None
    hoists: list[_Hoist]
    loop_entries: list[_Hoist]
    prefetch_sites: dict[int, dict[tuple[Node, Coords], str]]
    aux_sites: dict[int, dict[tuple[Node, Coords], int]]
    aux_anchors: dict[int, list[tuple[_Hoist, dict[int, str], str]]]
    aux_closed: set[int]
    aux_shared_stage: dict[int, str]
    aux_shared_early: set[int]
    aux_waited: bool
    aux_synced: bool
    shared_aux_nodes: set[Node]
    ring_aliases: dict[Node, _RingAlias]
    ring_ready: set[int]
    packable_memo: dict[Node, bool]
    stored_storages: set[StorageWeakRef]
    scratch: _Tile | None
    slot_layouts: list[_Layout]
    vec_layouts: list[_Layout | None]

    def reg(self, slot: int, am: int, an: int, j: int, /) -> str:
        """The register holding element ``(am, an, j)`` of fragment slot
        ``slot``: the warp emitter's mma.sync accumulator registers; the
        tcgen05 emitters' tensor-memory load chunks (``reg(slot, chunk, j)``)."""
        raise NotImplementedError

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

    def vreg(self, slot: int, atom: int, half: int) -> str:
        """Vector register of slot ``slot`` for the lane's row ``g`` (``half``
        0) / ``g + 8`` (1) of atom row ``atom`` (a row vector), its column
        ``2c`` / ``2c + 1`` of atom column ``atom`` (a column vector), or the
        one register of a scalar."""
        return f"md_r{slot}_{atom}_{half}"

    def typed(self, value: str, dtype: torch.dtype) -> str:
        """``value`` re-typed to the DSL scalar type of ``dtype``."""
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
        self.df.cute_state.root_plans.mma_dag_exact_f32[name] = value
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
        """One element widened to fp32 or int32, zero under a false predicate.

        Floating and int32 loads use the read-only form when ``readonly``
        and unpredicated. Byte loads always use their predicated integer row.
        """
        if dtype in (torch.int8, torch.uint8):
            op = "ldg_s8_pred" if dtype == torch.int8 else "ldg_u8_pred"
            return f"{_RUNTIME}.{op}({address}, {pred or 'cutlass.Int32(1)'})"
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
        """Two adjacent elements widened to fp32 or int32 scalars (zeros
        under a false Int32 ``pred``)."""
        lo, hi = self.df.new_var("md_lo"), self.df.new_var("md_hi")
        suffix = "" if pred is None else "_pred"
        args = _load_args(address, pred, readonly)
        # A byte pair must read exactly two bytes, including at an odd base.
        if not paired or dtype in (torch.int8, torch.uint8):
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
        if dtype in (torch.int8, torch.uint8, torch.int32, torch.int64):
            return f"{self.env.backend.dtype_str(dtype)}({int(cast('int', value))})"
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

    def set_tile_leaf(self, load: Access, element: int, raw: str) -> None:
        """Bind a loaded tile element: ``raw`` is an exact widened fp32
        half or a sign/zero-extended byte, and the leaf retains its DSL type."""
        key = (load.node, element)
        if load.dtype in (torch.int8, torch.uint8):
            # The raw byte has been sign/zero extended to an integer, not fp32.
            # Preserve the original byte type before evaluating the recipe.
            self.evaluator.tile_leaves[key] = self.bind(
                "md_byte", self.typed(raw, load.dtype)
            )
        else:
            self.evaluator.raw_leaves[key] = raw
            self.evaluator.tile_leaves[key] = self.retyped(raw, load.dtype)

    def set_byte_packet(self, load: Access, words: list[str]) -> None:
        """Unpack four words as sixteen typed signed or unsigned byte leaves."""
        assert load.dtype in (torch.int8, torch.uint8)
        for q, word in enumerate(words):
            for byte in range(4):
                value = (
                    f"(cutlass.Int32({word} << {24 - 8 * byte}) >> 24)"
                    if load.dtype == torch.int8
                    else f"(({word} >> {8 * byte}) & 255)"
                )
                self.set_tile_leaf(load, 4 * q + byte, value)

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

    def enclosing(self, region: int) -> list[int]:
        """Region ``region`` and the loops enclosing it, outermost first."""
        return self.program.enclosing_loops(region)[::-1]

    # -- index expressions ---------------------------------------------------

    def uniform(self, expr: sympy.Expr, values: dict[int, str] | str) -> str:
        """Render a uniform scalar subscript with the grid loops' indices as
        ``values`` (loop index -> value expression; a string is the single
        loop's value). Table lookups are rendered at those same indices,
        including future-iteration indices supplied by the prefetcher."""
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
            if symbol in self.program.index_loads:
                access = self.program.index_loads[symbol]
                address = self.iteration_base_expr(access, values)
                suffix = "s32" if access.dtype == torch.int32 else "s64"
                replacements[symbol] = sympy.Symbol(
                    f"{_RUNTIME}.ldg_{suffix}({address})", integer=True
                )
            elif symbol in symbols:
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
            value = self.program.resolve(value)
            if value in self.scalar_vars:
                return self.scalar_vars[value]
            val = value.meta["val"]
            if isinstance(val, int):
                return str(val)
            return f"({self.df.sympy_expr(val._sympy_())})"
        return str(value)

    def define_scalar_load(self, access: Access) -> None:
        """A uniform scalar read from a tensor (a loop bound, a window's
        offset or limit): one load at the access's base (its
        only address: a 0-d access has no tile dims).  Defined with the
        bases, ahead of the accesses that window on it and of every loop."""
        if access.node in self.scalar_vars:
            return
        base = self.base_vars[access.node]
        if access.dtype == torch.int32:
            load = f"{_RUNTIME}.ldg_s32({base})"
        elif access.node in self.program.uniform_values:
            # Bounds retain their integer width through arithmetic and induction.
            load = f"{_RUNTIME}.ldg_s64({base})"
        else:
            # The existing window protocol uses Int32 row indices into a static
            # tensor extent. General uniform bounds do not share that restriction.
            # Out-of-range window values beyond Int32 are not guarded here.
            load = f"cutlass.Int32({_RUNTIME}.ldg_s64({base}))"
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

    def define_bases(self) -> None:
        """Per-access byte bases: the tensor pointer plus the loop-invariant
        scalar subscripts and the root / leading tile offsets (Int64).  The
        scalar and table-load address dependencies come before consumers,
        so table-indexed bounds and window offsets use defined bases."""
        scalar_nodes = {access.node for access in self.program.scalar_loads}
        ordered: list[Access] = []
        seen: set[Node] = set()

        def visit(access: Access) -> None:
            if access.node in seen:
                return
            seen.add(access.node)
            # A table index can itself read another table. Define every
            # address dependency first, including those of uniform bounds.
            for subscript in access.subscripts:
                if subscript.window is not None:
                    visit(subscript.window.offset)
                for expr in (subscript.expr, subscript.offset):
                    if expr is not None:
                        for symbol in sorted(expr.free_symbols, key=str):
                            assert isinstance(symbol, sympy.Symbol)
                            dependency = self.program.index_loads.get(symbol)
                            if dependency is not None:
                                visit(dependency)
            ordered.append(access)

        for access in sorted(
            self.program.accesses, key=lambda access: access.node not in scalar_nodes
        ):
            visit(access)
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
                    if subscript.divisor is not None:
                        # Divide the complete global coordinate at the load;
                        # adding an affine tile base here would divide twice.
                        continue
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
        for loop in self.program.loops:
            for value in (loop.begin, loop.end):
                if isinstance(value, torch.fx.Node):
                    value = self.program.resolve(value)
                # Symbolic endpoints may also feed tensor bounds; they
                # remain symbolic and have no tensor dtype to materialize.
                if (
                    isinstance(value, torch.fx.Node)
                    and isinstance(value.meta["val"], torch.Tensor)
                    and value in self.program.uniform_values
                    and value not in self.scalar_vars
                ):
                    result = self.evaluator.evaluate(value, ())
                    dtype = self.env.backend.dtype_str(value.meta["val"].dtype)
                    self.scalar_vars[value] = self.bind(
                        "md_bound", f"{dtype}({result})"
                    )

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
                if subscript.divisor is not None:
                    continue
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
            subscript = access.subscripts[access.subscript_index(position)]
            if subscript.divisor is not None:
                global_base = (
                    "0"
                    if axis.kind == AXIS_RDIM and not self.is_split_axis(axis)
                    else self.axis_base(axis)
                )
                coord = f"(({global_base} + ({coord})) // {subscript.divisor})"
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
        if access.grouped:
            return self.plain_address(access, base, coords)
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

    def hoist_scalars(self, value: Node) -> None:
        """Evaluate the 0-d (uniform) values under ``value`` once."""
        for node in _scalar_nodes(self.program, value, self.evaluator.inline):
            self.evaluator.evaluate(node, ())

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

    # -- loops ----------------------------------------------------------------

    def wide_loop_bounds(self, loop: Loop | None = None) -> bool:
        """Whether a grid loop retains an Int64 tensor endpoint."""
        if loop is None:
            loop = self.program.loops[0]
        if loop.axes:
            return False
        for value in (loop.begin, loop.end):
            if isinstance(value, torch.fx.Node):
                val = self.program.resolve(value).meta.get("val")
                if isinstance(val, torch.Tensor) and val.dtype == torch.int64:
                    return True
        return False

    def advance_ring_counter(self, name: str, period: int) -> None:
        """Advance a small Int32 slot/phase ordinal without growing indefinitely.

        Only ring state uses this ordinal; memory indices retain the original
        loop induction type. ``period`` is a multiple of every derived ring's
        slot count and, when needed, its two barrier phases.
        """
        self.emit(f"{name} = {name} + 1")
        self.emit(f"{name} = cutlass.Int32(0) if {name} == {period} else {name}")

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


class _Nest:
    """Indented block: ``with emitter.nest("if x:"):``."""

    def __init__(self, emitter: MmaDagEmitterBase, header: str) -> None:
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

    order: list[Node] = []
    seen: set[Node] = set()
    frag_slots = cast("dict[Node, int]", program.frag_slots)  # pyrefly: ignore[missing-attribute]
    vec_leaves = cast("dict[Node, int]", program.vec_leaves)  # pyrefly: ignore[missing-attribute]
    aliases = cast("dict[Node, Node]", program.aliases)  # pyrefly: ignore[missing-attribute]
    aux_indices = cast("dict[Node, sympy.Expr]", program.aux_indices)  # pyrefly: ignore[missing-attribute]

    def visit(node: Node) -> None:
        if node in aliases:
            visit(aliases[node])
            return
        if (
            node in seen
            or node.op != "call_function"
            or (node in frag_slots and node not in inline)
            or node in vec_leaves
            or node in aux_indices
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
