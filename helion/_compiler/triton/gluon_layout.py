# ruff: noqa: ANN001, ANN202, ANN401
"""Triton-style layout assignment for Gluon kernels written like Triton.

Gluon resolves ``AutoLayout`` values in MLIR but rejects conflicts; Triton seeds
layouts at memory ops, propagates them forward and rematerializes cheap slices
per consumer.  This semantic does the same while tracing:

* a tensor-of-pointers access with an element per thread is an anchor: Triton's
  coalesced layout, from a port of its axis analysis;
* concrete layouts propagate; where two meet, the smaller operand converts;
* an auto value adopts its one consumer's layout; later consumers get a replay
  of its recipe or a convert to auto, so Gluon's resolver sees no conflict.
"""

from __future__ import annotations

import ast
import collections
import contextlib
import dataclasses
import functools
import itertools
import math
from typing import Any

from torch.utils._pytree import tree_leaves
from torch.utils._pytree import tree_map
from triton._utils import apply_with_path
from triton._utils import get_iterable_path
from triton.compiler import code_generator
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._layouts import AutoLayout
from triton.experimental.gluon.language._layouts import SliceLayout
from triton.experimental.gluon.language._semantic import GluonSemantic
import triton.language as tl

_MAX_DIV = 1 << 62


def _pow2_div(v: int) -> int:
    return _MAX_DIV if v == 0 else min(abs(v) & -abs(v), _MAX_DIV)


def _gcd(*xs: int) -> int:
    return math.gcd(*xs) if xs else 1


def _mul_div(a: int, b: int) -> int:
    return min(a * b, _MAX_DIV)


@dataclasses.dataclass(frozen=True)
class _AI:
    """Triton's AxisInfo for one value (rank 1 for scalars)."""

    contig: tuple[int, ...]
    div: tuple[int, ...]
    const: tuple[int, ...]
    value: int | None = None

    @staticmethod
    def unknown(shape: list[int]) -> _AI:
        r = max(len(shape), 1)
        return _AI((1,) * r, (1,) * r, (1,) * r)

    @staticmethod
    def constant(v: int, shape: list[int]) -> _AI:
        r = max(len(shape), 1)
        return _AI((1,) * r, (_pow2_div(v),) * r, tuple(shape) if shape else (1,), v)

    def splat(self, shape: list[int]) -> _AI:
        return _AI(
            (1,) * len(shape), (self.div[0],) * len(shape), tuple(shape), self.value
        )

    def unit_div(self) -> int:
        """A new unit dimension's divisibility (Triton's expand_dims)."""
        pairs = zip(self.contig, self.div, strict=True)
        return _gcd(*[1 if c > 1 else d for c, d in pairs])

    def lead(self, n: int) -> _AI:
        """``n`` leading unit dimensions (Triton's broadcast of a lower rank)."""
        d = self.unit_div()
        return _AI(
            (1,) * n + self.contig,
            (d,) * n + self.div,
            (1,) * n + self.const,
            self.value,
        )

    def broadcast(self, src: list[int], dst: list[int]) -> _AI:
        return _AI(
            tuple(1 if s == 1 else c for s, c in zip(src, self.contig, strict=True)),
            self.div,
            tuple(
                d if s == 1 else k for s, d, k in zip(src, dst, self.const, strict=True)
            ),
            self.value,
        )


# Undo log of metadata changes while a loop body is traced dry (its ops are erased).
_JOURNAL: list | None = None
# Memory writes traced so far (stores, atomics, shared stores, side-effecting asm; in
# any function, a loop's dry run too): a value reading memory replays only before the next.
_WRITES = 0


def _wrote() -> None:
    global _WRITES
    _WRITES += 1


@dataclasses.dataclass
class _Meta:
    # (method, args, kwargs, output index) of a replayable value
    recipe: tuple | None = None
    ai: _AI | None = None  # integer and pointer values
    consumed: bool = False  # an auto value already has its one consumer
    writes: int | None = None  # a replayable value reading memory: _WRITES then

    def consume(self) -> None:
        if _JOURNAL is not None:
            _JOURNAL.append(self)
        self.consumed = True


def _is_block(x: Any) -> bool:
    return isinstance(x, gl.tensor) and x.type.is_block()  # pyrefly: ignore [missing-attribute]


@gluon.constexpr_function
def _shared(ty: Any) -> bool:
    """Whether a value of type ``ty`` is a shared-memory descriptor."""
    return isinstance(ty, gl.shared_memory_descriptor_type)


def _is_auto(x: Any) -> bool:
    return _is_block(x) and isinstance(x.type.layout, AutoLayout)


def _shape(x: Any) -> list[int]:
    return [int(gl._unwrap_if_constexpr(s)) for s in x.shape] if _is_block(x) else []  # pyrefly: ignore [bad-argument-type]


# Ops whose concrete results can be re-emitted in another layout: same-layout ops and creations.
_REMAT_SAME = {
    "add", "sub", "mul", "truediv", "floordiv", "fdiv", "mod", "minus", "plus", "invert",
    "not_", "and_", "or_", "xor_", "shl", "lshr", "ashr", "equal", "not_equal",
    "less_than", "less_equal", "greater_than", "greater_equal", "logical_and",
    "logical_or", "minimum", "maximum", "clamp", "where", "cast", "bitcast",
    "broadcast_impl_shape",
}  # fmt: skip
_REMAT_LEAF = {"arange", "full", "splat"}
_REMAT = _REMAT_SAME | _REMAT_LEAF | {"expand_dims"}
# Values re-created in any layout: creations, and reads of memory nothing rewrites
# while they live (``slot_load``: Triton folds a convert into the local_load).
_REBUILT = _REMAT_LEAF | {"shared_load"}
# Ops whose auto results can be replayed from their recipe.
_REPLAYABLE = _REMAT | {"reshape", "split", "join", "permute", "reduction", "load"}
_WRITE_OPS = {"store", "shared_store", "atomic_cas", "atomic_xchg"} | {
    f"atomic_{op}" for op in ("add", "and", "max", "min", "or", "xor")
}

# Recipe walks' caps: ops a replay re-emits, rematerialization depth.
_REPLAY_OPS, _REMAT_DEPTH = 1024, 32


@dataclasses.dataclass
class _Loop:
    """A ``for`` / ``while`` being traced: the insertion point ahead of it, the ids
    of the values visible there, and the values emitted there by key (``_ahead``:
    until the dry run that finds its carries ends, ``open``; the loop op is
    inserted there next) with how many of each the body's trace took."""

    ip: tuple[Any, Any]
    invariant: set[int]
    open: bool = True
    ahead: dict[object, list[Any]] = dataclasses.field(default_factory=dict)
    taken: collections.Counter[object] = dataclasses.field(
        default_factory=collections.Counter
    )


_STATE: dict[int, _Meta] = {}
# (semantic, argument values) of the calls whose callees are being traced.
_CALLERS: list[tuple[Any, list[Any]]] = []
_POPULARITY: collections.Counter[Any] = collections.Counter()


class HelionGluonSemantic(GluonSemantic):
    def __init__(self, builder: Any, generator: Any = None) -> None:
        super().__init__(builder)
        self.generator = generator
        self.meta = _STATE
        self.depth = 0
        self.popularity = _POPULARITY  # values produced per concrete layout
        self.workers: list[list[Any]] = []  # warp_specialize partitions' arguments

    # ---- metadata ------------------------------------------------------------

    def _m(self, t: Any) -> _Meta:
        return self.meta.setdefault(t.handle.id(), _Meta())

    def _warps(self) -> int:
        return int(self.num_warps(self.generator))

    def _ai(self, x: Any, shape: list[int] | None = None) -> _AI:
        """Axis info of ``x`` brought to ``shape`` (splat/broadcast)."""
        if isinstance(x, gl.constexpr):
            x = x.value
        if isinstance(x, bool) or not isinstance(x, (int, gl.tensor)):
            return _AI.unknown(shape or [])
        if isinstance(x, int):
            return _AI.constant(x, shape or [])
        ai = self.meta.get(x.handle.id(), _Meta()).ai
        if ai is None:
            ai = _AI.unknown(_shape(x))
        if shape is not None and _shape(x) != shape:
            if not _is_block(x):
                ai = ai.splat(shape)
            else:
                n = len(shape) - len(_shape(x))
                ai = ai.lead(n).broadcast([1] * n + _shape(x), shape)
        return ai

    def _axis_info(self, name: str, args: tuple, out: Any) -> _AI | None:
        if not (out.dtype.is_int() or out.dtype.is_ptr()):
            return None
        shape = _shape(out)
        if name in ("make_scalar", "scalar_constant") and isinstance(
            gl._unwrap_if_constexpr(args[0]), int
        ):
            return _AI.constant(int(gl._unwrap_if_constexpr(args[0])), [])  # pyrefly: ignore [bad-argument-type]
        if name == "arange":
            start, end = (int(gl._unwrap_if_constexpr(a)) for a in args[:2])  # pyrefly: ignore [bad-argument-type]
            return _AI((end - start,), (_pow2_div(start),), (1,))
        if name in ("full", "splat"):
            v = args[1] if name == "full" else args[0]
            return self._ai(v, shape)
        if (
            name == "cast"
            and isinstance(args[0], gl.tensor)
            and (args[0].dtype.is_int() or args[0].dtype.is_ptr())
        ):
            return self._ai(args[0], shape)
        if name == "broadcast_impl_shape":
            return self._ai(args[0], shape)
        if name == "expand_dims":
            src = self._ai(args[0])
            axis = int(gl._unwrap_if_constexpr(args[1])) % len(shape)  # pyrefly: ignore [bad-argument-type]
            unit = src.unit_div()
            ins = lambda t, v: (*t[:axis], v, *t[axis:])  # noqa: E731
            return _AI(
                ins(src.contig, 1), ins(src.div, unit), ins(src.const, 1), src.value
            )
        if name == "reshape" and _is_block(args[0]):
            # Unit dimensions in or out only: the others keep theirs.
            src, kept = (
                self._ai(args[0]),
                [i for i, n in enumerate(_shape(args[0])) if n > 1],
            )
            if [_shape(args[0])[i] for i in kept] != [n for n in shape if n > 1]:
                return None
            where = iter(kept)
            dims = [next(where) if n > 1 else None for n in shape]
            pick = lambda t, v: tuple(v if d is None else t[d] for d in dims)  # noqa: E731
            return _AI(
                pick(src.contig, 1),
                pick(src.div, src.unit_div()),
                pick(src.const, 1),
                src.value,
            )
        if name in ("add", "sub", "mul"):
            lhs, rhs = self._ai(args[0], shape), self._ai(args[1], shape)
            elem = 1
            if out.dtype.is_ptr():
                ptr_ty = out.type.element_ty if _is_block(out) else out.dtype
                elem = max(1, ptr_ty.element_ty.primitive_bitwidth // 8)
            return _binary_ai(name, lhs, rhs, elem)
        if name in _COMBINE:
            lhs, rhs = self._ai(args[0], shape), self._ai(args[1], shape)
            return _combine_ai(name, lhs, rhs, shape)
        return None

    def _note(self, name: str, args: tuple, kwargs: dict, out: Any) -> None:
        ins = [x for x in tree_leaves((args, kwargs)) if _is_block(x)]
        outs = out if isinstance(out, (list, tuple)) else (out,)
        in_ids = {x.handle.id() for x in ins}
        for idx, o in enumerate(outs):
            if (
                not isinstance(o, gl.tensor)
                or o.handle.id() in self.meta
                or o.handle.id() in in_ids
            ):
                continue  # known, or an operand returned unchanged (a no-op cast)
            m = self.meta[o.handle.id()] = _Meta(
                ai=self._axis_info(name, args, o) if len(outs) == 1 else None
            )
            if not _is_auto(o):
                if _is_block(o):
                    self.popularity[o.type.layout] += 1  # pyrefly: ignore [missing-attribute]
                    if name in _REMAT:
                        m.recipe = (name, args, kwargs, idx)
                continue
            auto_ins = [x for x in ins if _is_auto(x)]
            replayable = name in _REPLAYABLE and all(
                self._m(x).recipe is not None for x in auto_ins
            )
            if replayable:
                m.recipe = (name, args, kwargs, idx)
                reads = [self._m(x).writes for x in auto_ins]
                m.writes = (
                    _WRITES
                    if name == "load"
                    else min((w for w in reads if w is not None), default=None)
                )

    # ---- adoption, replay, conversion -----------------------------------------

    def _call(self, name: str, args: Any, kwargs: Any, idx: int = 0) -> Any:
        """Re-emit op ``name`` (output ``idx``) as the one consumer of its operands."""
        # Consumed auto operands replay in turn.
        args, kwargs = self._consume((args, kwargs))
        self.depth += 1
        try:
            out = getattr(self, name)(*args, **kwargs)
        finally:
            self.depth -= 1
        return out[idx] if isinstance(out, (list, tuple)) else out

    def _replay(self, t: Any) -> Any:
        """A fresh copy of auto value ``t`` (and of its auto operands)."""
        return self._call(*self._m(t).recipe)  # pyrefly: ignore [not-iterable]

    def _cheap(self, x: Any) -> bool:
        """Whether replaying auto ``x`` re-emits at most ``_REPLAY_OPS`` ops."""
        budget, stack = _REPLAY_OPS, [x]
        while stack and budget >= 0:
            budget -= 1
            recipe = self._m(stack.pop()).recipe
            stack += [
                a for a in tree_leaves(recipe[1:3] if recipe else ()) if _is_auto(a)
            ]
        return budget >= 0

    def _consume(self, obj: Any) -> Any:
        """Give each auto value in ``obj`` this op as its one consumer."""

        def one(x: Any) -> Any:
            if not _is_auto(x):
                return x
            if self._m(x).consumed:
                x = self._again(x)
            self._m(x).consume()
            return x

        return tree_map(one, obj)

    def _again(self, x: Any) -> Any:
        """Another consumer's copy of consumed auto ``x``: its replay, else (a replay
        re-reads memory: not behind a write since, as a store between y = x and
        x *= 2 would give y = 2x) a convert keeping its axis info.  Emitted ahead
        of the outermost loop ``x`` is invariant in (Triton's LICM); the body's
        trace takes its dry run's copies back."""
        m = self._m(x)

        def copy() -> Any:
            if m.recipe is not None and m.writes in (None, _WRITES):
                return self._replay(x)
            y = super(HelionGluonSemantic, self).convert_layout(x, AutoLayout())
            self._m(y).ai = self._ai(x)
            return y

        loops = [lp for lp in self.generator._hl_loops if x.handle.id() in lp.invariant]
        y = self._ahead(loops[0], x.handle.id(), copy) if loops else None
        return copy() if y is None else y

    def _ahead(self, loop: _Loop, key: object, make: Any, shared: bool = False) -> Any:
        """``make()`` emitted ahead of ``loop`` by its dry run; the body's trace
        takes the dry run's values back in the same order (``shared``: every use
        the first), None past them."""
        made = loop.ahead.setdefault(key, [])
        if loop.open and not (shared and made):
            # Ahead of ``loop``, only the loops outside it are being traced: the
            # copies ``make`` takes are theirs (the body's trace takes ``y`` alone).
            loops = self.generator._hl_loops
            outer = loops[: [lp is loop for lp in loops].index(True)]
            new = len(self.meta)
            with _at(self.generator, loop.ip, outer):
                y = make()
            # The dry run's metadata is forgotten with its ops: ``y``'s and its
            # operands' (``y`` replays) are kept with it.
            self._m(y)
            keys = {y.handle.id(), *itertools.islice(self.meta, new, None)}
            made.append((y, {k: dataclasses.replace(self.meta[k]) for k in keys}))
            return y
        loop.taken[key] += not shared
        if (index := loop.taken[key] - 1 if not shared else 0) >= len(made):
            return None
        y, metas = made[index]
        self.meta.update({k: dataclasses.replace(m) for k, m in metas.items()})
        self._m(y).consumed = False
        return y

    def _adopt(self, t: Any, layout: Any) -> Any:
        """Give auto value ``t`` (consumed by the current op) the layout of that op."""
        return super().set_auto_layout(t, layout)

    def convert_layout(
        self, value: Any, layout: Any, assert_trivial: bool = False
    ) -> Any:
        return self._read(self._converting(value, layout, assert_trivial), layout)

    def convert_layouts(self, pairs: list[tuple[Any, Any]]) -> list[Any]:
        """Each (value, layout)'s ``convert_layout`` (a shared value's read), every
        shared copy stored before any is read: Triton's order, so no copy's store
        waits on another's reads, nor reuses its memory."""
        outs = [v if _shared(v.type) else self._converting(v, lay) for v, lay in pairs]
        return [self._read(o, lay) for o, (_, lay) in zip(outs, pairs, strict=True)]

    def _read(self, x: Any, layout: Any) -> Any:
        return self.shared_load(x, layout) if _shared(x.type) else x

    def _converting(self, value: Any, layout: Any, assert_trivial: bool = False) -> Any:
        """An auto value (consumed by this op) adopts ``layout`` where Gluon can infer it back
        (blocked and slice layouts, creations), else takes the default layout and converts: so
        Triton-style helpers (the translator's) convert auto values freely."""
        if _is_auto(value) and not isinstance(layout, AutoLayout):
            recipe = self._m(value).recipe
            if isinstance(layout, (gl.BlockedLayout, SliceLayout)) or (
                recipe and recipe[0] in _REBUILT
            ):
                return self._adopt(value, layout)
            value = self._adopt(value, default_layout(_shape(value), self._warps()))
        if value.type.layout == layout and not assert_trivial:
            return value
        recipe = self.meta.get(value.handle.id(), _Meta()).recipe
        if recipe is not None and recipe[0] in _REBUILT and not assert_trivial:
            # A creation is re-created in ``layout`` (Triton folds the convert).
            return self.remat(value, layout)
        key = (value.handle.id(), layout)
        for loop in self.generator._hl_loops:
            if key[0] not in loop.invariant or assert_trivial:
                continue
            # Loop invariant (Triton's LICM): converted once ahead of the
            # outermost loop it does not vary in (a shared copy read per use).
            hoisted = self._ahead(loop, key, lambda: self._copy(value, layout), True)
            if hoisted is not None:
                return hoisted
        if assert_trivial:
            return super().convert_layout(value, layout, assert_trivial)
        return self._copy(value, layout)

    def slot_load(self, memory: Any, layout: Any) -> Any:
        """``memory`` read in ``layout``; nothing rewrites it while the value lives,
        so a use in another layout re-reads it instead of converting."""
        value = self.shared_load(memory, layout)
        self._m(value).recipe = ("shared_load", (memory, layout), {}, 0)
        return value

    def loop_buffer(self, element_ty: Any, shape: Any, layout: Any) -> Any:
        """Shared memory allocated ahead of the innermost loop being traced: the
        dry run allocates, the body takes them in the same order."""
        shape = [gl._unwrap_if_constexpr(s) for s in shape]
        allocate = functools.partial(
            super().allocate_shared, element_ty, shape, layout, None
        )
        loops = self.generator._hl_loops
        return self._ahead(loops[-1], "buffer", allocate) if loops else allocate()

    def _copy(self, value: Any, layout: Any) -> Any:
        """Gluon's convert, but (Triton's ReduceDataDuplication) one into a dot
        operand across warps is a shared copy (to read), swizzled for its reads: a
        private copy, so no traced write (loads replay past it)."""
        shape = _shape(value)
        src, dst = (
            self.to_linear_layout(x, shape).value for x in (value.type.layout, layout)
        )
        if not isinstance(layout, gl.DotOperandLayout) or (
            src.warp_bases == dst.warp_bases
        ):
            return super().convert_layout(value, layout)
        smem = _operand_smem(layout, shape, _memory_order(src), value.dtype)
        return super().allocate_shared(value.dtype, shape, smem, value)

    def settle(self, x: Any, layout: Any) -> Any:
        """``x`` in ``layout`` if it is auto; a concrete value keeps its own."""
        return self.convert_layout(x, layout) if _is_auto(x) else x

    def _remat_ok(self, x: Any, numel: int, depth: int = 0) -> bool:
        """Whether concrete ``x`` can be re-emitted in another layout converting only smaller values."""
        if not _is_block(x) or _is_auto(x):
            return True
        # A concrete value's recipe is a rematerializable op's (``_note``).
        recipe = self.meta.get(x.handle.id(), _Meta()).recipe
        if recipe is not None and depth < _REMAT_DEPTH:
            return recipe[0] in _REBUILT or all(
                self._remat_ok(a, numel, depth + 1) for a in tree_leaves(recipe[1:3])
            )
        return x.numel < numel

    def _fold_splats(self, name: str, args: tuple, kwargs: dict) -> Any:
        """As Triton's reorder-broadcast pass: an elementwise or shape op whose tensor
        operands are all splats is the splat of the op on their scalars (or None)."""
        blocks = [x for x in tree_leaves((args, kwargs)) if _is_block(x)]
        recipes = [self._m(x).recipe for x in blocks]
        if not blocks or any(r is None or r[0] != "splat" for r in recipes):
            return None
        source = {x.handle.id(): r[1][0] for x, r in zip(blocks, recipes, strict=True)}  # pyrefly: ignore [unsupported-operation]
        x, *rest = args
        shape, layout = _shape(x), AutoLayout()
        if name == "expand_dims":
            shape.insert(rest[0] % (len(shape) + 1), 1)
        elif name == "permute":
            shape = [shape[d] for d in rest[0]]
        elif name in ("reshape", "broadcast_impl_shape"):
            shape = [int(gl._unwrap_if_constexpr(s)) for s in rest[0]]  # pyrefly: ignore [bad-argument-type]
            if name == "broadcast_impl_shape":  # keeps its operand's layout
                if shape == _shape(x):
                    return x
                layout = x.type.layout
        else:
            shape = _broadcast_shape(*map(_shape, blocks))
            layout = next(
                (
                    b.type.layout
                    for b in blocks
                    if _shape(b) == shape and not _is_auto(b)
                ),
                layout,
            )
            args, kwargs = tree_map(
                lambda a: source[a.handle.id()] if _is_block(a) else a, (args, kwargs)
            )
            return self.splat(getattr(self, name)(*args, **kwargs), shape, layout)
        return self.splat(source[x.handle.id()], shape, layout)

    def remat(self, x: Any, layout: Any) -> Any:
        """``x`` in ``layout``: its cheap slice re-emitted there (Triton's backward rematerialization),
        converting only the values it cannot rebuild."""
        if not _is_block(x):
            return x
        if _is_auto(x):
            return self._adopt(self._consume(x), layout)
        if x.type.layout == layout:
            return x
        if not self._remat_ok(x, x.numel):
            return super().convert_layout(x, layout)
        name, args, kwargs, idx = self.meta[x.handle.id()].recipe  # pyrefly: ignore [not-iterable]
        if name in _REBUILT:
            # Its arguments but the layout, then ``layout``.
            keep = {"full": 3, "shared_load": 1}.get(name, 2)
            out = self._call(name, (*args[:keep], layout), {})
        elif name == "expand_dims":
            axis = int(gl._unwrap_if_constexpr(args[1])) % len(_shape(x))  # pyrefly: ignore [bad-argument-type]
            out = self._call(
                name, (self.remat(args[0], SliceLayout(axis, layout)), axis), {}
            )
        else:
            args, kwargs = tree_map(
                lambda a: self.remat(a, layout) if _is_block(a) else a, (args, kwargs)
            )
            out = self._call(name, args, kwargs, idx)
        return out[idx] if isinstance(out, (list, tuple)) else out

    def unify(self, args: Any) -> Any:
        """One layout for the concrete tensor operands of an elementwise op: the one that
        cannot be rebuilt, else the biggest (Triton hoists converts above broadcasts),
        else the layout more values have."""
        blocks = [x for x in args if _is_block(x) and not _is_auto(x)]
        if len({x.type.layout for x in blocks}) <= 1:
            return args
        lead = max(
            blocks,
            key=lambda x: (
                not self._remat_ok(x, x.numel),
                x.numel,
                self.popularity[x.type.layout],
            ),
        )
        return [
            self.remat(x, lead.type.layout) if any(x is y for y in blocks) else x
            for x in args
        ]

    def binary_op_sanitize_overflow_impl(
        self, lhs: Any, rhs: Any, binary_op: Any
    ) -> None:
        # Else the check's assert is dropped: no dead (layout-less) check.
        if self.builder.options.debug:
            super().binary_op_sanitize_overflow_impl(lhs, rhs, binary_op)

    def device_assert(self, cond: Any, msg: str, mask: Any) -> Any:
        return super().device_assert(*self.seed_default([cond])[:1], msg, mask)  # pyrefly: ignore [bad-argument-count]

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        return super().where(*self.unify([condition, x, y]))

    def seed_default(self, args: Any) -> Any:
        """Triton's default layout for a side-effecting op whose tensor operands are all auto."""
        tensors = [x for x in args if _is_block(x)]
        if not tensors or not all(_is_auto(x) for x in tensors):
            return args
        args = self._consume(list(args))
        return [
            self._adopt(x, default_layout(_shape(x), self._warps()))
            if _is_block(x)
            else x
            for x in args
        ]

    def _join(self, args: list[Any]) -> list[Any]:
        """``unify``, then the auto tensor operands adopt that layout."""
        args = self.unify(args)
        lay = next(
            (x.type.layout for x in args if _is_block(x) and not _is_auto(x)), None
        )
        return [
            self._adopt(x, lay) if lay is not None and _is_auto(x) else x for x in args
        ]

    def _coalesced(self, ptr: Any, max_bits: int = 128) -> gl.BlockedLayout:
        """Triton's coalesced encoding for a tensor-of-pointers access."""
        shape = _shape(ptr)
        ai = self._ai(ptr)
        order = sorted(range(len(shape)), key=lambda d: (-ai.contig[d], -d))
        bits = ptr.type.element_ty.element_ty.primitive_bitwidth
        nbytes = max(bits // 8, 1)
        align = min(
            max(ai.div[order[0]] // nbytes, 1), ai.contig[order[0]], shape[order[0]]
        )
        return _vector(shape, order, bits, align, self._warps(), max_bits)  # pyrefly: ignore [bad-return]

    # ---- overrides --------------------------------------------------------------

    # Triton-language builtins (``tl.arange`` in a ``@triton.jit`` helper) pass no layout.
    def arange(self, start: Any, end: Any, layout: Any = None) -> Any:
        return super().arange(start, end, layout)

    def full(self, shape: Any, value: Any, dtype: Any, layout: Any = None) -> Any:
        return super().full(shape, value, dtype, layout)

    def splat(self, value: Any, shape: Any, layout: Any = None) -> Any:
        return super().splat(
            value, shape, layout if layout is not None else AutoLayout()
        )

    def unsplat(self, value: Any) -> Any:
        # No consumer gives a scalar's operand a layout: Triton's default.
        lay = default_layout(_shape(value), self._warps())
        return super().unsplat(self.settle(value, lay))

    def broadcast_impl_shape(self, input: Any, shape: Any) -> Any:  # noqa: A002
        # As Triton: a scalar splats.
        if not _is_block(input):
            return self.splat(input, shape)
        return super().broadcast_impl_shape(input, shape)

    def broadcast_impl_value(self, lhs: Any, rhs: Any) -> Any:
        # As Triton: a lower-rank operand gains leading unit dimensions first.
        if _is_block(lhs) and _is_block(rhs):
            while len(_shape(lhs)) < len(_shape(rhs)):
                lhs = self.expand_dims(lhs, 0)
            while len(_shape(rhs)) < len(_shape(lhs)):
                rhs = self.expand_dims(rhs, 0)
        return super().broadcast_impl_value(*self._join([lhs, rhs]))

    def to_auto(self, x: Any) -> Any:
        """Release concrete ``x`` to its consumer's layout (a replayable convert)."""
        out = super().convert_layout(x, AutoLayout())
        self.meta[out.handle.id()] = _Meta(
            recipe=("to_auto", (x,), {}, 0), ai=self._ai(x)
        )
        return out

    def expand_dims(self, input: Any, axis: int) -> Any:  # noqa: A002
        if not _is_block(input):  # As Triton: a scalar splats to [1].
            return self.splat(input, [1])
        lay = input.type.layout
        if not _is_auto(input):
            if not (
                isinstance(lay, SliceLayout)
                and lay.dim == axis % (len(input.shape) + 1)
            ):
                input = self.to_auto(input)  # noqa: A001
        return super().expand_dims(input, axis)

    def multiple_of(self, x: Any, values: Any) -> Any:
        return self._hint(super().multiple_of(x, values), div=values)

    def max_contiguous(self, x: Any, values: Any) -> Any:
        return self._hint(super().max_contiguous(x, values), contig=values)

    def max_constancy(self, x: Any, values: Any) -> Any:
        return self._hint(super().max_constancy(x, values), const=values)

    def _hint(self, x: Any, **values: Any) -> Any:
        """As Triton's axis analysis, a hint overrides the inferred dimensions."""
        fields = {k: tuple(map(int, v)) for k, v in values.items()}
        self._m(x).ai = dataclasses.replace(self._ai(x), **fields)  # pyrefly: ignore [bad-argument-type]
        return x

    def warp_specialize(self, functions_and_args: Any, *args: Any) -> Any:  # pyrefly: ignore [bad-override]
        # Each worker traces from its region's block arguments: its callee seeds
        # them from these (``call_JitFunction``).
        self.workers = [list(a) for _, a in functions_and_args[1:]]
        return super().warp_specialize(functions_and_args, *args)

    def gather(self, src: Any, index: Any, axis: int) -> Any:
        # As TritonToTritonGPU: auto operands take the default layout.
        src, index = (
            self.settle(x, default_layout(_shape(x), self._warps()))
            for x in (src, index)
        )
        return super().gather(src, index, axis)

    def split(self, a: Any) -> Any:
        # As TritonToTritonGPU: the split dim in-thread (Gluon cannot infer split siblings from auto).
        if _is_auto(a):
            a = self._adopt(a, _split_layout(_shape(a), self._warps()))
        elif _is_block(a) and not self._split_in_thread(a):
            a = self.remat(a, _split_layout(_shape(a), self._warps()))
        return super().split(a)

    def _split_in_thread(self, a: Any) -> bool:
        # ... and no broadcast (zero) basis: Triton infers a split of such a linear
        # layout one way when building and another when verifying.
        reg, rest = self._bases(a)
        rank = len(_shape(a))
        return [0] * (rank - 1) + [1] in reg and [0] * rank not in reg + rest

    def _bases(self, x: Any) -> tuple[list[list[int]], list[list[int]]]:
        """``x``'s register bases and its lane and warp bases."""
        ll = gl._unwrap_if_constexpr(self.to_linear_layout(x.type.layout, _shape(x)))
        reg = [list(b) for b in ll.reg_bases]  # pyrefly: ignore [missing-attribute]
        return reg, [list(b) for b in (*ll.lane_bases, *ll.warp_bases)]  # pyrefly: ignore [missing-attribute]

    def reduction(self, inputs: Any, axis: int, region_builder_fn: Any) -> Any:
        inputs = self._join(list(inputs))
        if inputs and _is_auto(inputs[0]):
            # Gluon infers a reduce operand only from a matching slice layout: seed it here (Triton's default).
            lay = default_layout(_shape(inputs[0]), self._warps())
            inputs = [self._adopt(x, lay) for x in inputs]
        return super().reduction(inputs, axis, region_builder_fn)

    def associative_scan(
        self, inputs: Any, axis: int, region_builder_fn: Any, reverse: bool
    ) -> Any:
        # Triton lowers scans of blocked layouts only (its layout passes never propagate others into one),
        # all operands in one.
        blocked = [
            x.type.layout
            for x in inputs
            if not _is_auto(x) and isinstance(x.type.layout, gl.BlockedLayout)
        ]
        lay = (
            blocked[0] if blocked else default_layout(_shape(inputs[0]), self._warps())
        )
        inputs = [
            x
            if not _is_auto(x) and x.type.layout == lay
            else self.convert_layout(self._consume(x), lay)
            for x in inputs
        ]
        return super().associative_scan(inputs, axis, region_builder_fn, reverse)

    def _anchors(self, ptr: Any) -> bool:
        """Whether a pointer tensor is big enough to anchor its access's layout: a
        value per thread or more."""
        return ptr.numel >= self._warps() * 32

    def _recoalesce(self, ptr: Any, max_bits: int = 128) -> Any:
        """A big concrete pointer tensor in a layout other than its coalesced one, rebuilt there."""
        if _is_block(ptr) and not _is_auto(ptr) and self._anchors(ptr):
            lay = self._coalesced(ptr, max_bits)
            if lay != ptr.type.layout and self._remat_ok(ptr, ptr.numel):
                return self.remat(ptr, lay)
        return ptr

    def load(self, ptr: Any, mask: Any, other: Any, *args: Any) -> Any:  # pyrefly: ignore [bad-override]
        # Big loads anchor; small ones adopt their consumer's layout unless they cannot be replayed (volatile).
        if _is_auto(ptr) and all(
            x is None or not _is_block(x) or _is_auto(x) for x in (mask, other)
        ):
            if self._anchors(ptr) or args[-1]:
                ptr = self._adopt(ptr, self._coalesced(ptr))
        elif (new := self._recoalesce(ptr)) is not ptr:
            ptr = new
            mask, other = (self.remat(x, ptr.type.layout) for x in (mask, other))
        return super().load(ptr, mask, other, *args)

    def _store_like(
        self, ptr: Any, val: Any, mask: Any, max_bits: int = 128
    ) -> tuple[Any, Any, Any]:
        if not _is_auto(ptr):
            if (new := self._recoalesce(ptr, max_bits)) is not ptr:
                lay = new.type.layout
                val = (
                    val
                    if not _is_block(val) or _is_auto(val)
                    else super().convert_layout(val, lay)
                )
                return new, val, self.remat(mask, lay)
            return ptr, val, mask
        anchored = [x for x in (val, mask) if _is_block(x) and not _is_auto(x)]
        if not anchored or self._anchors(ptr):
            # An anchor (as Triton's coalesce): the store stays coalesced, its operands convert.
            ptr = self._adopt(ptr, self._coalesced(ptr, max_bits))
            lay = ptr.type.layout
            val, mask = (
                super(HelionGluonSemantic, self).convert_layout(x, lay)
                if _is_block(x) and not _is_auto(x) and x.type.layout != lay
                else x
                for x in (val, mask)
            )
        return ptr, val, mask

    def allocate_shared(
        self, element_ty: Any, shape: Any, layout: Any, value: Any
    ) -> Any:
        # Gluon infers no layout from shared memory: a stored auto value takes the
        # tile's coalesced one (so the translator's descriptor store takes any value).
        if value is not None and _is_auto(value):
            lay = vector_layout(_shape(value), value.dtype, self._warps())
            value = self._adopt(value, gl._unwrap_if_constexpr(lay))
        if value is not None:
            _wrote()
        return super().allocate_shared(element_ty, shape, layout, value)

    def store(self, ptr: Any, val: Any, mask: Any, *args: Any) -> Any:  # pyrefly: ignore [bad-override]
        ptr, val, mask = self._store_like(ptr, val, mask)
        return super().store(ptr, val, mask, *args)

    def atom_red_typechecking_impl(self, ptr: Any, val: Any, mask: Any, op: str) -> Any:
        if _is_block(ptr):
            # Triton's atomic width: one integer or f64 element a thread, else a vector.
            ty = ptr.type.scalar.element_ty
            wide = ty.is_floating() and ty.primitive_bitwidth < 64
            bits = 128 if wide else ty.primitive_bitwidth
            ptr, val, mask = self._store_like(ptr, val, mask, max_bits=bits)
            held = next(
                (x for x in (ptr, val) if _is_block(x) and not _is_auto(x)), None
            )
            if held is not None and [0] * len(_shape(ptr)) in self._bases(held)[0]:
                # A register held twice would issue its atomic twice: coalesce it.
                lay = self._coalesced(ptr, bits)
                ptr, val, mask = (
                    self.remat(x, lay) if _is_block(x) else x for x in (ptr, val, mask)
                )
        return super().atom_red_typechecking_impl(ptr, val, mask, op)


def _binary_ai(name: str, lhs: _AI, rhs: _AI, elem: int) -> _AI:
    contig, div, const = [], [], []
    for d in range(len(lhs.contig)):
        lc, rc, lk, rk = lhs.contig[d], rhs.contig[d], lhs.const[d], rhs.const[d]
        ld, rd = lhs.div[d], rhs.div[d]
        const.append(_gcd(lk, rk))
        if name == "mul":
            contig.append(max(lc if rhs.value == 1 else 1, rc if lhs.value == 1 else 1))
            ld = 1 if lc > 1 and rhs.value != 1 else ld
            rd = 1 if rc > 1 and lhs.value != 1 else rd
            div.append(_mul_div(ld, rd))
            continue
        c = _gcd(lc, rk) if name == "sub" else max(_gcd(lk, rc), _gcd(lc, rk))
        rd = _mul_div(rd, elem)
        if lc > 1 and rc > 1:
            dv = (
                _gcd(ld, rd)
                if name == "sub" and lc == rc
                else (2 if ld % 2 == 0 and rd % 2 == 0 else 1)
            )
        elif lc > 1 or rc > 1:
            dv = _gcd(ld, rd, _mul_div(c, elem))
        else:
            dv = _gcd(ld, rd)
        contig.append(c)
        div.append(dv)
    value = None
    if lhs.value is not None and rhs.value is not None:
        value = {
            "add": lhs.value + rhs.value,
            "sub": lhs.value - rhs.value,
            "mul": lhs.value * rhs.value,
        }[name]
    return _AI(tuple(contig), tuple(div), tuple(const), value)


_COMBINE = {
    "equal", "not_equal", "less_than", "less_equal", "greater_than", "greater_equal",
    "and_", "or_", "xor_", "minimum", "maximum",
}  # fmt: skip


def _combine_ai(name: str, lhs: _AI, rhs: _AI, shape: list[int]) -> _AI:
    """Triton's AxisInfo of a comparison (a run against a bound both divide is
    constant), a logical op, or a min / max (the join)."""
    join = name in ("minimum", "maximum")
    contig, div, const = [], [], []
    for d in range(len(lhs.contig)):
        lc, rc, ld, rd = lhs.contig[d], rhs.contig[d], lhs.div[d], rhs.div[d]
        k = _gcd(lhs.const[d], rhs.const[d])
        whole = shape[d] if shape else 1
        if name in ("greater_than", "less_equal") and lhs.const[d] == whole:
            k = max(k, _gcd(rc, ld, rd))
        elif name in ("less_than", "greater_equal") and rhs.const[d] == whole:
            k = max(k, _gcd(lc, ld, rd))
        const.append(k)
        contig.append(_gcd(lc, rc) if join else 1)
        div.append((_gcd(ld, rd) if lc == rc else _gcd(ld, rd, lc, rc)) if join else 1)
    value = lhs.value if join and lhs.value == rhs.value else None
    return _AI(tuple(contig), tuple(div), tuple(const), value)


@gluon.constexpr_function
def _blocked(
    shape: list[int], spt: list[int], order: list[int], num_warps: int
) -> gl.BlockedLayout:
    """``BlockedEncodingAttr::get``: lanes and warps along ``order`` from the contiguous dim."""
    rank = len(shape)
    tpw, wpc = [1] * rank, [1] * rank
    lanes, threads, warps, prev_l, prev_w = 32, num_warps * 32, num_warps, 1, 1
    for i in order[:-1]:
        per_cta = min(max(threads, 1), max(1, shape[i] // spt[i]))
        tpw[i] = min(max(per_cta, 1), lanes)
        wpc[i] = min(max(per_cta // tpw[i], 1), warps)
        warps //= wpc[i]
        lanes //= tpw[i]
        threads //= per_cta
        prev_l *= tpw[i]
        prev_w *= wpc[i]
    tpw[order[-1]] = 32 // prev_l
    wpc[order[-1]] = num_warps // prev_w
    return gl.BlockedLayout(spt, tpw, wpc, order)


@gluon.constexpr_function
def _vector(
    shape: list[int],
    order: list[int],
    bits: int,
    align: int,
    num_warps: int,
    max_bits: int = 128,
) -> gl.BlockedLayout:
    """Triton's coalesced encoding: up to ``max_bits`` per thread along ``order[0]``,
    of which ``align`` elements are contiguous."""
    per = min(
        align, max(max_bits // bits, 1), max(math.prod(shape) // (num_warps * 32), 1)
    )
    spt = [1] * len(shape)
    spt[order[0]] = per
    return _blocked(shape, spt, order, num_warps)  # pyrefly: ignore [bad-return]


@gluon.constexpr_function
def vector_layout(shape: Any, dtype: Any, num_warps: Any) -> gl.BlockedLayout:
    """A contiguous shared-memory tile's register layout (Triton's coalesced one)."""
    dims = [int(d) for d in shape]
    order = list(reversed(range(len(dims))))
    return _vector(dims, order, dtype.primitive_bitwidth, dims[-1], int(num_warps))  # pyrefly: ignore [bad-return]


def _memory_order(ll: Any) -> list[int]:
    """Triton's ``getOrderForMemory``: dimensions in register order, the lanes'
    where their fastest dimension holds one element a thread."""

    def order(bases: list[list[int]]) -> list[int]:
        dims = [b.index(max(b)) for b in bases if any(b)]
        return [*dict.fromkeys([*dims, *reversed(range(len(ll.shape)))])]

    regs, lanes = order(ll.reg_bases), order(ll.lane_bases)
    if regs != lanes and not any(b[lanes[0]] for b in ll.reg_bases):
        return lanes
    return regs


def _operand_smem(
    layout: Any, shape: list[int], order: list[int], dtype: Any
) -> gl.SwizzledSharedLayout:
    """Triton's swizzle of a dot operand's shared copy: mma.sync's ldmatrix reads
    conflict free (vectors of ``4 * k_width`` along K); an FMA operand's none."""
    if not isinstance(layout.parent, gl.NVMMADistributedLayout):
        return gl.SwizzledSharedLayout(1, 1, 1, order)
    bits = dtype.primitive_bitwidth
    per_phase = max(1024 // (bits * shape[order[0]]), 1)
    vec, stride = 4 * layout.k_width, 4 if bits == 64 else 8
    if order[0] != len(shape) - 1 - layout.operand_index:
        vec, stride = stride, vec
    phases = max(min(stride, 1024 // (vec * bits)), 1)
    return gl.SwizzledSharedLayout(vec, per_phase, max(phases // per_phase, 1), order)


def _broadcast_shape(*shapes: list[int]) -> list[int]:
    n = max(map(len, shapes))
    return [max(([1] * (n - len(s)) + s)[i] for s in shapes) for i in range(n)]


@gluon.constexpr_function
def default_layout(shape: list[int], num_warps: int) -> gl.BlockedLayout:
    """ConvertTritonToTritonGPU's default blocked layout."""
    dims = [int(d) for d in shape]
    return _blocked(dims, [1] * len(dims), list(reversed(range(len(dims)))), num_warps)  # pyrefly: ignore [bad-return]


def _split_layout(shape: list[int], num_warps: int) -> gl.BlockedLayout:
    """A split operand layout: trailing dims in-thread as far as elements per thread allow, so the
    reshape/split trees Helion emits (tree reductions, lane packing) stay in-thread."""
    budget = max(math.prod(shape) // (num_warps * 32), 2)
    spt = [1] * len(shape)
    for d in reversed(range(len(shape))):
        spt[d] = max(1, min(shape[d], budget))
        budget //= spt[d]
        if budget <= 1:
            break
    return _blocked(shape, spt, list(reversed(range(len(shape)))), num_warps)  # pyrefly: ignore [bad-return]


def _wrap(name: str) -> None:
    orig = getattr(HelionGluonSemantic, name)
    record = name in _REPLAYABLE or name in _CONSTANTS
    fold = name in _REMAT_SAME or name in ("expand_dims", "reshape", "permute")

    def wrapped(self: HelionGluonSemantic, *args: Any, **kwargs: Any) -> Any:
        if fold and (out := self._fold_splats(name, args, kwargs)) is not None:
            return out
        outer = self.depth == 0
        if outer:
            args, kwargs = self._consume((args, kwargs))
        self.depth += 1
        try:
            out = orig(self, *args, **kwargs)
        finally:
            self.depth -= 1
        if name in _WRITE_OPS:
            _wrote()
        if outer:  # an op returning an operand unchanged did not use it
            outs = {
                o.handle.id()
                for o in (out if isinstance(out, (list, tuple)) else (out,))
                if _is_block(o)
            }
            for x in tree_leaves((args, kwargs)):
                if _is_auto(x) and x.handle.id() in outs:
                    self._m(x).consumed = False
        if record:
            self._note(name, args, kwargs, out)
            if outer and _is_auto(out) and not self._cheap(out):
                # A long auto chain (a hash) would replay per use: Triton's default.
                out = self._adopt(out, default_layout(_shape(out), self._warps()))
        return out

    setattr(HelionGluonSemantic, name, wrapped)


# Every op taking tensors consumes them (at the outermost semantic call);
# constants get axis info.
_CONSTANTS = ("make_scalar", "scalar_constant")
for _name in sorted(
    _REPLAYABLE
    | _WRITE_OPS
    | set(_CONSTANTS)
    | {
        "to_tensor",
        "convert_layout",
        "convert_layouts",
        "unsplat",
        "associative_scan",
        "histogram",
        "gather",
        "device_print",
        "device_assert",
        "fp4_to_fp",
        "map_elementwise",
        "allocate_shared",
        "dot",
        "dot_scaled",
    }
):
    _wrap(_name)


# ---- code generator hooks -----------------------------------------------------


def install(enabled: Any) -> None:
    """Use ``HelionGluonSemantic`` for Gluon kernels where ``enabled(generator)``."""
    init = code_generator.CodeGenerator.__init__
    find_carries = code_generator.CodeGenerator._find_carries
    verify = code_generator.CodeGenerator._verify_loop_carried_variable
    then_else = code_generator.CodeGenerator.visit_then_else_blocks
    loops = {
        name: getattr(code_generator.CodeGenerator, name)
        for name in ("visit_For", "visit_While")
    }
    compound = code_generator.CodeGenerator.visit_compound_statement

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)
        if self.is_gluon and enabled(self):
            if self.caller_context is None and kwargs.get("is_kernel"):
                _STATE.clear()
                _POPULARITY.clear()
                _note_arguments(self.prototype)
            elif _CALLERS:
                _note_arguments(self.prototype, _CALLERS[-1])
            self.semantic = HelionGluonSemantic(self.builder, self)
        # Loops being traced, the ``if`` and the loop carries being traced.
        self._hl_loops, self._hl_if, self._hl_carry = [], None, None

    def _verify_loop_carried_variable(
        self: Any, name: str, loop_val: Any, live_val: Any
    ) -> None:
        if (
            isinstance(self.semantic, HelionGluonSemantic)
            and not _is_auto(loop_val)
            and _relayout(loop_val, live_val)
        ):
            self._hl_carry[name] = loop_val
            return
        verify(self, name, loop_val, live_val)

    def visit_then_else_blocks(self: Any, node: Any, liveins: dict, *rest: Any) -> Any:
        if not isinstance(self.semantic, HelionGluonSemantic):
            return then_else(self, node, liveins, *rest)
        # The insertion point is still the parent's: the ``if`` goes there after.
        outer = self._hl_if
        self._hl_if = (node, liveins, self._get_insertion_point_and_loc())
        try:
            return then_else(self, node, liveins, *rest)
        finally:
            self._hl_if = outer

    def visit_compound_statement(self: Any, stmts: Any) -> None:
        compound(self, stmts)
        if not isinstance(self.semantic, HelionGluonSemantic) or self._hl_if is None:
            return
        node, liveins, parent = self._hl_if
        if stmts is node.body or stmts is node.orelse:
            for name, live in liveins.items():
                if not _relayout(value := self.lscope.get(name), live):
                    continue
                if _is_auto(live) and not _is_auto(value):
                    # An auto live-in takes the branch's layout ahead of the
                    # ``if``, for the other branch and the yield types too.
                    with _at(self, parent):
                        liveins[name] = self.semantic._adopt(live, value.type.layout)
                else:
                    # A live-in redefined in another layout yields in its own.
                    layout = live.type.layout
                    self.set_value(name, self.semantic.convert_layout(value, layout))

    def visit_loop(visit: Any) -> Any:
        def visit_node(self: Any, node: Any) -> Any:
            if not isinstance(self.semantic, HelionGluonSemantic):
                return visit(self, node)
            if isinstance(node, ast.For) and (loops := _unrolled(self, node)):
                for loop in loops:
                    self.visit(loop)
                return None
            invariant = {v.handle.id() for v in self.lscope.values() if _is_block(v)}
            self._hl_loops.append(_Loop(self._get_insertion_point_and_loc(), invariant))
            try:
                return visit(self, node)
            finally:
                self._hl_loops.pop()

        return visit_node

    def _find_carries(
        self: Any,
        node: Any,
        liveins: dict,
        ignore: set[str] = set(),  # noqa: B006
    ) -> Any:
        sem = self.semantic
        if not isinstance(sem, HelionGluonSemantic):
            return find_carries(self, node, liveins, ignore)
        ip, loc = self._get_insertion_point_and_loc()
        if isinstance(node, ast.For) and node.target.id in self.lscope:  # pyrefly: ignore [missing-attribute]
            _note_induction_var(self, node)
        global _JOURNAL
        outer_carry = self._hl_carry
        carry = self._hl_carry = {}
        keys, outer, _JOURNAL = set(_STATE), _JOURNAL, []
        try:
            names, handles, tys = find_carries(self, node, liveins, ignore)
        finally:
            # The loop op goes at the insertion point next: no hoisting there after.
            self._hl_loops[-1].open = False
            # The dry run's ops are erased: forget their values (their addresses get reused) and uses.
            for k in set(_STATE) - keys:
                del _STATE[k]
            for m in _JOURNAL:
                m.consumed = False
            _JOURNAL = outer
            self._hl_carry = outer_carry
        if not carry:
            return names, handles, tys
        # A carried value initialized auto and anchored in the body starts in the body's layout.
        # The dry run's block is erased, so the builder's insertion point dangles: adopt at the
        # loop's own insertion point and stay there (the caller restores that one anyway).
        self._set_insertion_point_and_loc(ip, loc)
        for name, loop_val in carry.items():
            live, layout = liveins[name], loop_val.type.layout
            liveins[name] = (
                sem._adopt(live, layout)
                if _is_auto(live)
                else sem.convert_layout(live, layout)
            )
        self.lscope = liveins.copy()
        handles, tys = [], []
        for name in names:
            handles.extend(code_generator.flatten_values_to_ir([liveins[name]]))
            tys.append(liveins[name].type)
        return names, handles, tys

    call_jit = code_generator.CodeGenerator.call_JitFunction

    def call_JitFunction(
        self: Any, fn: Any, args: Any, kwargs: Any, *rest: Any, **kw: Any
    ) -> Any:
        sem = self.semantic
        if not isinstance(sem, HelionGluonSemantic):
            return call_jit(self, fn, args, kwargs, *rest, **kw)
        args, kwargs = sem._consume((list(args), dict(kwargs)))
        # A callee's scalar arguments take the caller's axis info (Triton's
        # analysis propagates it into callees): a worker's, its partition args'.
        source = (
            sem.workers.pop(0) if kw.get("caller_context") and sem.workers else args
        )
        bound = fn.signature.bind(*source, **kwargs)
        bound.apply_defaults()
        _CALLERS.append((sem, [bound.arguments[n] for n in fn.arg_names]))
        try:
            return call_jit(self, fn, args, kwargs, *rest, **kw)
        finally:
            _CALLERS.pop()

    code_generator.CodeGenerator.call_JitFunction = call_JitFunction
    # A side-effecting inline asm whose tensor operands are all auto (``bar.warp.sync`` on an
    # arange, in Helion's runtime helpers too) gets Triton's default layout.
    inline_asm = tl.core.inline_asm_elementwise

    @tl.core.builtin
    def inline_asm_elementwise(
        asm, constraints, args, dtype, is_pure, pack, _semantic=None
    ):
        if isinstance(_semantic, HelionGluonSemantic):
            if not tl.core._unwrap_if_constexpr(is_pure):
                args = _semantic.seed_default(args)
                _wrote()
            args = _semantic.unify(list(args))
        return inline_asm(
            asm, constraints, args, dtype, is_pure, pack, _semantic=_semantic
        )

    # (A gluon kernel's ``tl`` resolves the name from ``gl``.)
    for module in (tl, tl.core, gl, gl._core):
        module.inline_asm_elementwise = inline_asm_elementwise  # pyrefly: ignore [missing-attribute]
    code_generator.CodeGenerator.__init__ = __init__
    code_generator.CodeGenerator._verify_loop_carried_variable = (
        _verify_loop_carried_variable
    )
    code_generator.CodeGenerator._find_carries = _find_carries
    code_generator.CodeGenerator.visit_then_else_blocks = visit_then_else_blocks  # pyrefly: ignore [bad-assignment]
    code_generator.CodeGenerator.visit_compound_statement = visit_compound_statement
    for name, visit in loops.items():
        setattr(code_generator.CodeGenerator, name, visit_loop(visit))


def _note_arguments(
    prototype: Any, caller: tuple[Any, list[Any]] | None = None
) -> None:
    """Axis info of a function's scalar arguments as Triton's analysis seeds it: a
    kernel's specialized divisibility (``tt.divisibility``, else 1), a callee's
    from its caller's values."""
    deserialize = prototype.deserialize

    def note(fn: Any) -> Any:
        values = deserialize(fn)

        def one(path: tuple[int, ...], _: Any) -> None:
            value = get_iterable_path(values, path)
            if isinstance(value, gl.tensor) and not _is_block(value):
                if caller is not None:
                    sem, args = caller
                    _STATE[value.handle.id()] = _Meta(
                        ai=sem._ai(get_iterable_path(args, path))
                    )
                    return
                specs = dict(prototype.attrs.get(path, []))
                div = specs.get("tt.divisibility", 1)
                _STATE[value.handle.id()] = _Meta(ai=_AI((1,), (div,), (1,)))

        apply_with_path(prototype.arg_types, one)
        return values

    prototype.deserialize = note


@contextlib.contextmanager
def _at(gen: Any, ip: tuple[Any, Any], loops: list[_Loop] | None = None):
    """Emit at insertion point ``ip`` (ahead of a loop or branch; inside ``loops``,
    if given), then back."""
    here, traced = gen._get_insertion_point_and_loc(), gen._hl_loops
    gen._set_insertion_point_and_loc(*ip)
    gen._hl_loops = traced if loops is None else loops
    yield
    gen._set_insertion_point_and_loc(*here)
    gen._hl_loops = traced


def _relayout(value: Any, live: Any) -> bool:
    """Whether a value redefining live-in ``live`` in a loop or branch differs from
    it only in layout."""
    return (
        _is_block(value)
        and _is_block(live)
        and value.type.layout != live.type.layout
        and _shape(value) == _shape(live)
        and value.dtype == live.dtype
    )


def range_bounds(args: list[Any], zero: object, one: object) -> list[Any]:
    """``range(*args)``'s [start, stop, step]."""
    return ([zero] * (len(args) == 1) + [*args, one])[:3]


def _unrolled(gen: Any, node: ast.For) -> list[ast.stmt] | None:
    """A ``range(..., loop_unroll_factor=U)`` loop as Triton's loop unroll pass makes
    it (Gluon's compile runs none): the body ``U`` times a step, then the remainder."""
    it = node.iter
    if not isinstance(it, ast.Call):
        return None
    kws = [k for k in it.keywords if k.arg != "loop_unroll_factor"]
    if len(kws) == len(it.keywords):
        return None
    factor = gen.visit(next(k for k in it.keywords if k not in kws).value)
    if (factor := int(gl._unwrap_if_constexpr(factor))) <= 1:  # pyrefly: ignore [bad-argument-type]
        return None
    bounds = range_bounds(
        [gen.visit(a) for a in it.args], gl.constexpr(0), gl.constexpr(1)
    )
    if not isinstance(bounds[2], gl.constexpr) or bounds[2].value <= 0:
        return None  # (the trip count below assumes a positive step)
    lb, ub, st, mid, u = (
        f"_hl_{id(node):x}_{k}" for k in ("lb", "ub", "st", "mid", "u")
    )
    for name, value in zip((lb, ub, st), bounds, strict=True):
        gen.set_value(name, value)
    if all(isinstance(v, gl.constexpr) for v in bounds):
        lo, hi, step = (v.value for v in bounds)
        gen.set_value(
            mid, gl.constexpr(lo + max(hi - lo, 0) // (step * factor) * (step * factor))
        )
    else:  # as MLIR's unroll; ``//`` and ``%`` truncate: an empty range keeps mid >= ub
        trips = gen.visit(
            ast.parse(f"({ub} - {lb} + {st} - 1) // {st}", mode="eval").body
        )
        gen.set_value(mid, trips)
        expr = f"{lb} + ({mid} - {mid} % {factor}) * {st}"
        gen.set_value(mid, gen.visit(ast.parse(expr, mode="eval").body))

    def loop(target: ast.expr, args: str, body: list[ast.stmt]) -> ast.stmt:
        call = ast.Call(it.func, ast.parse(f"f({args})", mode="eval").body.args, kws)  # pyrefly: ignore [missing-attribute]
        return ast.For(target, call, body, [])

    body = [
        stmt
        for k in range(factor)
        for stmt in [
            ast.parse(f"{node.target.id} = {u} + {k} * {st}").body[0],  # pyrefly: ignore [missing-attribute]
            *node.body,
        ]
    ]
    return [
        ast.fix_missing_locations(ast.copy_location(stmt, node))
        for stmt in (
            loop(ast.Name(u, ast.Store()), f"{lb}, {mid}, {st} * {factor}", body),
            loop(node.target, f"{mid}, {ub}, {st}", node.body),
        )
    ]


def _note_induction_var(gen: Any, node: ast.For) -> None:
    """Divisibility of a ``for`` induction variable: gcd(lower bound, step), as Triton's analysis."""
    call = node.iter
    if not isinstance(call, ast.Call):
        return
    lb, _, step = range_bounds([gen.visit(a) for a in call.args], 0, 1)
    sem = gen.semantic
    d = _gcd(sem._ai(lb).div[0], sem._ai(step).div[0])
    iv = gen.lscope[node.target.id]
    sem.meta[iv.handle.id()] = _Meta(ai=_AI((1,), (d,), (1,)))
