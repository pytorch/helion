"""Python AST → MSL C++ statement-level translation functions.

Provides pure functions for converting Python AST nodes to MSL C++ text.
Used by ``metal_kernel._generate_msl`` to translate the body of a
``@metal_jit`` decorated function into MSL source code.

MetalOverrides (reused from Inductor's MPS codegen) emits MSL expression strings;
``msl_expr`` parses them into Python AST (``metal::precise::sin`` →
``metal.precise.sin``, ``static_cast<T>(x)`` → ``static_cast(T, x)``,
``a && b`` → ``a and b``, ...) and this module prints that AST back as MSL.

Handles:
  - Statement-level Python→C++ translation (assignments, if/for, etc.)
  - ``tl.load(ptr + offset, mask, other=0)`` → ``(mask ? *(ptr+offset) : (T)(0))``
    (``select_masked_loads`` first turns a read-only buffer's into
    ``_helion_masked(ptr[mask ? offset : 0], mask, 0)``)
  - ``tl.store(ptr + offset, val, mask)`` → ``if (mask) { *(ptr+offset) = val; }``
  - ``static_cast(T, x)`` → ``static_cast<T>(x)`` (and the other ``TEMPLATE_CALLS``)
  - C++ namespace restoration: ``metal.precise.sin`` → ``metal::precise::sin``
  - Generic function call fallback (main handler for MetalOverrides expressions)
"""

from __future__ import annotations

import ast
from collections import Counter
import contextlib
import copy
import dataclasses
from typing import Iterator

from ... import exc
from .fragment_mma import EarlyLoad
from .fragment_mma import FragmentElement
from .fragment_mma import emit_fragment_acc
from .fragment_mma import emit_fragment_run
from .fragment_mma import fragment_elements
from .fragment_mma import is_fragment_acc
from .fragment_mma import prefetch_loads
from .lane_fold import LANE_PUT
from .lane_fold import load_mask_other
from .lane_fold import tl_load
from .mpp_matmul import ACC_SEED
from .mpp_matmul import FRAG_RUN
from .msl_expr import GLOBAL_SCOPE
from .msl_expr import TEMPLATE_CALLS
from .msl_reduction import REDUCTION_NAMESPACE


@dataclasses.dataclass
class EmitState:
    """Mutable state passed through ``_emit_stmts`` calls.

    Tracks declared variable names (to avoid duplicate ``auto`` declarations).

    Declarations are tracked per C++ block scope.  A rolled reduction emits
    several sibling ``for`` loops that assign the same Helion variable (e.g.
    ``rindex_1``); each needs its own ``auto`` because the previous loop's
    declaration went out of scope at its closing brace.
    """

    declared: set[str] = dataclasses.field(default_factory=set)
    _enclosing: list[set[str]] = dataclasses.field(default_factory=list)
    #: Each MPP cooperative tensor's name, or a reference alias's, to its root.
    coop: dict[str, str] = dataclasses.field(default_factory=dict)
    #: Each cooperative root's MPP group id and its lhs and rhs element types.
    coop_group: dict[str, tuple[int, tuple[str, str]]] = dataclasses.field(
        default_factory=dict
    )
    #: Each cooperative root's M and N blocks.
    coop_tile: dict[str, tuple[int, int]] = dataclasses.field(default_factory=dict)
    #: The cooperative roots whose descriptor rounds a size-1 dim up.
    coop_rounded: set[str] = dataclasses.field(default_factory=set)
    #: Each fragment group's accumulator root to its fragment grid ``(rows,
    #: cols, transposed)`` (``fragment_mma``).
    fragments: dict[str, tuple[int, int, int]] = dataclasses.field(default_factory=dict)
    #: The kernel's names that are constants, which a fragment run folds its
    #: elements' index arithmetic over and tabulates their decodes with
    #: (``fragment_mma``'s ``kernel_constants``).
    constants: dict[str, int | float] = dataclasses.field(default_factory=dict)
    #: Each ``constant`` table (``tabulate``) by its element type and
    #: entries, to its name; the kernel's source declares them.
    tables: dict[tuple[str, tuple[str, ...]], str] = dataclasses.field(
        default_factory=dict
    )
    #: Each fragment run's K offset, the loads its K loop issues a tile
    #: early, to the registers holding them, and the elements built to find
    #: them (``fragment_mma``'s ``prefetch_loads``), by the run's id.
    prefetch: dict[
        int,
        tuple[
            str,
            dict[EarlyLoad, str],
            dict[tuple[int, int, int], FragmentElement],
        ],
    ] = dataclasses.field(default_factory=dict)
    #: How many prefetch registers the kernel has named, which keeps their
    #: names unique.
    early_count: int = 0
    #: The element an element loop's ``_helion_mpp_put`` writes.
    put: str = "*_it"

    def is_declared(self, name: str) -> bool:
        return name in self.declared or any(name in s for s in self._enclosing)

    def declare(self, name: str) -> None:
        self.declared.add(name)

    @contextlib.contextmanager
    def isolated_scope(self) -> Iterator[None]:
        """Enter a C++ block whose every assignment declares a new variable,
        shadowing any outer one (an MPP element loop's re-run statements)."""
        saved = (self.declared, self._enclosing)
        self.declared, self._enclosing = set(), []
        try:
            yield
        finally:
            self.declared, self._enclosing = saved

    @contextlib.contextmanager
    def block_scope(self) -> Iterator[None]:
        """Enter a nested C++ block; declarations made inside do not escape."""
        self._enclosing.append(self.declared)
        self.declared = set()
        try:
            yield
        finally:
            self.declared = self._enclosing.pop()


# ---------------------------------------------------------------------------
# Statement emitter (handles Assign, Expr, If, For, and tl.store calls)
# ---------------------------------------------------------------------------


def _is_tl_store_call(node: ast.AST) -> bool:
    """Return True if *node* is a ``tl.store(...)`` call."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
        and node.func.attr == "store"
    )


def _emit_tl_store(
    node: ast.Call,
    parts: list[str],
    indent: int,
    subs: dict[str, str] | None = None,
) -> None:
    """Emit MSL for ``tl.store(ptr + offset, val, mask)``.

    Dereferences the pointer expression: ``*(ptr + offset) = val``.
    All three positional args (ptr, val, mask) are always present —
    PointerIndexingStrategy emits them positionally.
    """
    pad = " " * indent
    access = _ptr_access_expr(node.args[0], subs=subs)
    val_expr = _ast_expr_to_msl(node.args[1], subs=subs)
    mask = None
    if len(node.args) >= 3:
        mask_node = node.args[2]
        # None mask = no masking
        if not (isinstance(mask_node, ast.Constant) and mask_node.value is None):
            mask = _ast_expr_to_msl(mask_node, subs=subs)
    if mask is not None:
        parts.extend(
            (
                f"{pad}if ({mask}) {{",
                f"{pad}    {access} = {val_expr};",
                f"{pad}}}",
            )
        )
    else:
        parts.append(f"{pad}{access} = {val_expr};")


def _emit_stmts(
    stmts: list[ast.stmt],
    parts: list[str],
    indent: int,
    state: EmitState,
    subs: dict[str, str] | None = None,
) -> None:
    """Emit a list of AST statements as MSL lines into *parts*.

    *subs* is an optional substitution dict applied when converting
    expressions to MSL (e.g. ``{"acc": "(*_it)"}`` inside an MPP element
    loop).  It's set by the ``_helion_mpp_elements`` branch of
    :func:`_emit_for` and propagated downward through nested control flow.

    This function does pure text emission.
    """
    pad = " " * indent
    for stmt in stmts:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            value = stmt.value
            if not isinstance(target, ast.Name):
                raise exc.BackendUnsupported(
                    "metal",
                    f"assignment target type: {type(target).__name__}",
                )
            if isinstance(value, ast.Call) and _is_call_to(value, "_helion_mpp_acc"):
                _emit_coop_acc(target.id, value, parts, pad, state)
                continue
            if isinstance(value, ast.Call) and _is_call_to(value, "_helion_mpp_run"):
                _emit_coop_run(target.id, value, parts, pad, state)
                continue
            if isinstance(value, ast.Name) and value.id in state.coop:
                root = state.coop[value.id]
                if state.coop.get(target.id) != root:
                    parts.append(f"{pad}thread auto& {target.id} = {value.id};")
                    state.coop[target.id] = root
                    state.declare(target.id)
                continue
            if isinstance(value, ast.Call) and _is_call_to(value, "_metal_lane_array"):
                element, extent = map(ast.literal_eval, value.args)
                parts.append(f"{pad}{element} {target.id}[{extent}];")
                state.declare(target.id)
                continue
            val_msl = _ast_expr_to_msl(value, subs=subs)
            if state.is_declared(target.id):
                parts.append(f"{pad}{target.id} = {val_msl};")
            else:
                parts.append(f"{pad}auto {target.id} = {val_msl};")
                state.declare(target.id)
        elif isinstance(stmt, ast.Expr):
            call = stmt.value
            # tl.store(ptr + offset, val, mask) → if (mask) { *(ptr+offset) = val; }
            if _is_tl_store_call(call):
                assert isinstance(call, ast.Call)
                _emit_tl_store(call, parts, indent=indent, subs=subs)
            elif _is_call_to(call, LANE_PUT):
                assert isinstance(call, ast.Call)
                array, lane, value = (_ast_expr_to_msl(a, subs=subs) for a in call.args)
                parts.append(f"{pad}{array}[{lane}] = {value};")
            elif _is_call_to(call, "_helion_mpp_stage"):
                assert isinstance(call, ast.Call)
                buf, index, value, pred = (
                    _ast_expr_to_msl(a, subs=subs) for a in call.args[:4]
                )
                parts.append(f"{pad}if ({pred}) {{ {buf}[{index}] = {value}; }}")
            elif _is_call_to(call, "_helion_mpp_put"):
                # An element loop's write of the updated element (mpp_matmul's
                # _helion_mpp_update).
                assert isinstance(call, ast.Call)
                parts.append(
                    f"{pad}{state.put} = {_ast_expr_to_msl(call.args[0], subs=subs)};"
                )
            elif _is_call_to(call, "_helion_mpp_fence"):
                parts.append(
                    f"{pad}threadgroup_barrier(mem_flags::mem_device | "
                    "mem_flags::mem_threadgroup);"
                )
            elif _is_call_to(call, "_helion_mpp_barrier"):
                parts.append(f"{pad}threadgroup_barrier(mem_flags::mem_threadgroup);")
            elif _is_call_to(call, "_helion_mpp_materialize"):
                assert isinstance(call, ast.Call)
                _emit_coop_materialize(call, parts, pad, state)
            elif _is_call_to(call, "_helion_mpp_bind"):
                assert isinstance(call, ast.Call)
                name = ast.literal_eval(call.args[0])
                parts.append(f"{pad}auto {name} = {_ast_expr_to_msl(call.args[1])};")
                state.declare(name)
            else:
                parts.append(f"{pad}{_ast_expr_to_msl(call, subs=subs)};")
        elif isinstance(stmt, ast.If):
            test_msl = _ast_expr_to_msl(stmt.test, subs=subs)
            parts.append(f"{pad}if ({test_msl}) {{")
            with state.block_scope():
                _emit_stmts(stmt.body, parts, indent=indent + 4, state=state, subs=subs)
            if stmt.orelse:
                parts.append(f"{pad}}} else {{")
                with state.block_scope():
                    _emit_stmts(
                        stmt.orelse, parts, indent=indent + 4, state=state, subs=subs
                    )
            parts.append(f"{pad}}}")
        elif isinstance(stmt, ast.For):
            _emit_for(stmt, parts, indent=indent, state=state, subs=subs)
        elif isinstance(stmt, ast.AugAssign):
            target_msl = _ast_expr_to_msl(stmt.target, subs=subs)
            val_msl = _ast_expr_to_msl(stmt.value, subs=subs)
            op_map: dict[type[ast.operator], str] = {
                ast.Add: "+=",
                ast.Sub: "-=",
                ast.Mult: "*=",
                ast.Div: "/=",
            }
            op_str = op_map.get(type(stmt.op))
            if op_str is None:
                raise exc.BackendUnsupported(
                    "metal", f"augmented assign op: {type(stmt.op).__name__}"
                )
            parts.append(f"{pad}{target_msl} {op_str} {val_msl};")
        elif isinstance(stmt, ast.Pass):
            pass
        else:
            raise exc.BackendUnsupported(
                "metal",
                f"AST statement type: {type(stmt).__name__}",
            )


def _emit_coop_acc(
    name: str, call: ast.Call, parts: list[str], pad: str, state: EmitState
) -> None:
    """Declare an MPP group's ``matmul2d`` op and its cooperative accumulator,
    filled with the seed value (``mpp_matmul``'s ``_helion_mpp_acc``).

    The op's scope is every SIMD group the launch dispatches, built from the
    launch's own constants; ``check_mpp_launch`` rejects launches that are not
    whole SIMD groups.  A fragment group's accumulator is
    ``fragment_mma``'s.
    """
    if is_fragment_acc(call):
        emit_fragment_acc(name, call, parts, pad, state)
        return
    gid, tm, tn, tk, dtypes, lhs_t, rhs_t, fill = (
        ast.literal_eval(a) for a in call.args[:8]
    )
    lhs_staged, rhs_staged, block_m, block_n = (
        ast.literal_eval(a) for a in call.args[9:ACC_SEED]
    )
    seed = ast.literal_eval(call.args[ACC_SEED])
    if (block_m, block_n) != (tm, tn):
        state.coop_rounded.add(name)
    state.coop_tile[name] = (block_m, block_n)
    lhs_extents = f"{tm}, {tk}" if lhs_t else f"{tk}, {tm}"
    rhs_extents = f"{tk}, {tn}" if rhs_t else f"{tn}, {tk}"
    scope = "(_TG_SIZE_X * _TG_SIZE_Y * _TG_SIZE_Z) / 32"
    flags = f"{str(bool(lhs_t)).lower()}, {str(bool(rhs_t)).lower()}"
    desc = f"_mpp_desc_{gid}"
    lhs_space = "threadgroup" if lhs_staged else "device"
    rhs_space = "threadgroup" if rhs_staged else "device"
    lhs_dtype, rhs_dtype = dtypes
    lhs = f"tensor<{lhs_space} {lhs_dtype}, extents<int32_t, {lhs_extents}>, tensor_inline>"
    rhs = f"tensor<{rhs_space} {rhs_dtype}, extents<int32_t, {rhs_extents}>, tensor_inline>"
    parts.extend(
        [
            (
                f"{pad}constexpr auto {desc} = matmul2d_descriptor({tm}, {tn}, {tk}, "
                f"{flags}, false, matmul2d_descriptor::mode::multiply_accumulate);"
            ),
            f"{pad}matmul2d<{desc}, execution_simdgroups<{scope}>> _mpp_op_{gid};",
            (
                f"{pad}auto {name} = _mpp_op_{gid}.get_destination_cooperative_tensor<"
                f"{lhs}, {rhs}, float>();"
            ),
        ]
    )
    if seed is None:
        parts.append(
            f"{pad}for (auto _it = {name}.begin(); _it != {name}.end(); _it++) "
            f"*_it = {float(fill)!r}f;"
        )
    else:
        # The staged seed (``mpp_matmul``'s ``_mpp_coop``): every thread's
        # writes are visible before any element reads them, and every element
        # has read the tile before any thread writes it again, as the next
        # tile of an enclosing loop does.  Elements past a rounded block are
        # never read, so they keep no value.
        parts.extend(
            [
                f"{pad}threadgroup_barrier(mem_flags::mem_threadgroup);",
                *_coop_elements(name, pad, block_m, block_n),
                f"{pad}    *_it = {seed}[_e_m * {block_n} + _e_n];",
                f"{pad}}}",
                f"{pad}threadgroup_barrier(mem_flags::mem_threadgroup);",
            ]
        )
    state.declare(name)
    state.coop[name] = name
    state.coop_group[name] = (gid, dtypes)


def _coop_elements(coop: str, pad: str, block_m: int, block_n: int) -> list[str]:
    """The head of a loop over the elements of ``coop`` that lie inside its
    ``block_m`` x ``block_n`` block, with each element's row and column in
    ``_e_m`` and ``_e_n``; the caller closes it."""
    return [
        f"{pad}for (auto _it = {coop}.begin(); _it != {coop}.end(); _it++) {{",
        f"{pad}    if (!{coop}.is_valid_element(_it)) continue;",
        f"{pad}    auto _md = {coop}.get_multidimensional_index(_it);",
        f"{pad}    int _e_n = (int)_md[0];",
        f"{pad}    int _e_m = (int)_md[1];",
        f"{pad}    if (_e_m >= {block_m} || _e_n >= {block_n}) continue;",
    ]


def _emit_coop_materialize(
    call: ast.Call, parts: list[str], pad: str, state: EmitState
) -> None:
    """Write a cooperative tensor to the row-major threadgroup tile its
    Scalar readers read (``mpp_matmul``'s ``_mpp_mat``).  The barriers order
    the writes after every read of the tile's previous contents and before
    every read of these."""
    coop, buf = (_ast_expr_to_msl(a) for a in call.args[:2])
    cols = ast.literal_eval(call.args[2])
    root = state.coop[coop]
    block_m, block_n = state.coop_tile[root]
    assert block_n == cols
    parts.extend(
        [
            f"{pad}threadgroup_barrier(mem_flags::mem_threadgroup);",
            *_coop_elements(coop, pad, block_m, block_n),
            f"{pad}    {buf}[_e_m * {cols} + _e_n] = *_it;",
            f"{pad}}}",
            f"{pad}threadgroup_barrier(mem_flags::mem_threadgroup);",
        ]
    )


def _coop_view(
    args: list[ast.expr],
    dtype: str,
    var: str,
    tile_dims: tuple[bool, bool],
    blocks: tuple[int, int],
    outer: list[ast.expr],
) -> tuple[list[str], list[str], str]:
    """A device view of one tile, from ``tensor, (origin, extent, size,
    stride)`` per dim, and ``(index, size, stride)`` per outer dim, each
    folded into the base pointer in 64 bits and live when ``0 <= index <
    size``.

    Returns the 64-bit declarations, the terms that test the view is live,
    and the handle.  MPP tensor handles index in int32, so the view's origin
    is folded into the base pointer in 64 bits and the handle's extents are
    each dim's live length from there, capped at the tile: ``min(e - o, B)``,
    in ``[1, B]`` once the live test passed.  MPP reads rows ``[0, B)`` of the
    handle and zero-fills past its extents.

    An iota dim (``tile_dims[d]`` false) holds its origin and extent in
    ``long``, the extent clamped to the size, and is live when
    ``origin < extent``.  A tile dim's origin is its loop offset and its
    extent the size, so it is live by construction (offset < loop end <=
    size).
    """
    ptr = _ast_expr_to_msl(args[0])
    decls: list[str] = []
    live: list[str] = []
    origins: list[str] = []
    extents: list[str] = []
    strides: list[str] = []
    for d in range(2):
        o, e, size, stride = (_ast_expr_to_msl(a) for a in args[1 + 4 * d : 5 + 4 * d])
        strides.append(stride)
        if tile_dims[d]:
            origins.append(f"(long)({o})")
            extents.append(
                f"(int)metal::min((long)({size}) - (long)({o}), (long){blocks[d]})"
            )
            continue
        ov, ev = f"{var}_o{d}", f"{var}_e{d}"
        # mpp_matmul clamps a literal extent beside a literal size.
        if e == size or (e.lstrip("-").isdigit() and size.isdigit()):
            e = f"(long)({e})"
        else:
            e = f"metal::min((long)({e}), (long)({size}))"
        decls += [f"long {ov} = (long)({o});", f"long {ev} = {e};"]
        live.append(f"{ov} < {ev}")
        origins.append(ov)
        extents.append(f"(int)metal::min({ev} - {ov}, (long){blocks[d]})")
    terms = [f"{origins[0]} * {strides[0]}", f"{origins[1]} * {strides[1]}"]
    for j, entry in enumerate(outer):
        assert isinstance(entry, ast.Tuple)
        idx, size, stride = (_ast_expr_to_msl(a) for a in entry.elts)
        xv = f"{var}_x{j}"
        decls.append(f"long {xv} = (long)({idx});")
        live.append(f"0 <= {xv} && {xv} < (long)({size})")
        terms.append(f"{xv} * {stride}")
    base = f"{ptr} + ({' + '.join(terms)})"
    # Handle dims are innermost first.
    if strides[1] == "1":
        dims, handle_strides = f"{extents[1]}, {extents[0]}", f"1, {strides[0]}"
    else:
        assert strides[0] == "1", "an MPP view without a unit stride"
        dims, handle_strides = f"{extents[0]}, {extents[1]}", f"1, {strides[1]}"
    handle = (
        f"auto {var} = tensor<device {dtype}, dextents<int32_t, 2>, tensor_inline>("
        f"{base}, dextents<int32_t, 2>({dims}), "
        f"metal::array<int32_t, 2>({{{handle_strides}}}));"
    )
    return decls, live, handle


def _staged_view(
    args: list[ast.expr], dtype: str, var: str, blocks: tuple[int, int]
) -> str:
    """The handle of a staged tile (``mpp_matmul._stage``), written row-major,
    from ``tile, (0, extent, rows, cols), (0, extent, cols, 1)``: each extent
    is a live length, capped at the tile, since a K extent runs from the
    tile's offset to the loop's end."""

    def extent(e: ast.expr, block: int) -> str:
        msl = _ast_expr_to_msl(e)
        return (
            msl
            if msl == str(block)
            else f"(int)metal::min((long)({msl}), (long){block})"
        )

    rows, cols = blocks
    return (
        f"auto {var} = tensor<threadgroup {dtype}, dextents<int32_t, 2>, tensor_inline>("
        f"{_ast_expr_to_msl(args[0])}, dextents<int32_t, 2>({extent(args[6], cols)}, "
        f"{extent(args[2], rows)}), metal::array<int32_t, 2>({{1, {cols}}}));"
    )


def _emit_coop_run(
    name: str, call: ast.Call, parts: list[str], pad: str, state: EmitState
) -> None:
    """One K tile, or a whole ``:`` K, of an MPP group: ``op.run(lhs, rhs,
    acc)`` on device views or staged tiles (``mpp_matmul``'s ``_helion_mpp_run``)."""
    src = call.args[0]
    assert isinstance(src, ast.Name) and src.id in state.coop
    root = state.coop[src.id]
    gid, dtypes = state.coop_group[root]
    # _helion_mpp_run's arguments: acc, gid, the lhs and rhs views (nine each), four
    # tile-dim flags, four block sizes.
    tile_dims = [bool(ast.literal_eval(a)) for a in call.args[20:24]]
    blocks = [int(ast.literal_eval(a)) for a in call.args[24:28]]
    (spaces,) = [ast.literal_eval(k.value) for k in call.keywords if k.arg == "spaces"]
    (outers,) = [k.value for k in call.keywords if k.arg == "outer"]
    assert isinstance(outers, ast.List)
    handles: list[tuple[list[str], list[str], str]] = []
    for i, var in enumerate(("_mpp_a", "_mpp_b")):
        args = call.args[2 + 9 * i : 11 + 9 * i]
        dims = (blocks[2 * i], blocks[2 * i + 1])
        dtype = dtypes[i]
        if spaces[i]:
            handles.append(([], [], _staged_view(args, dtype, var, dims)))
        else:
            tile = (tile_dims[2 * i], tile_dims[2 * i + 1])
            outer = outers.elts[i]
            assert isinstance(outer, ast.List)
            handles.append(_coop_view(args, dtype, var, tile, dims, outer.elts))
    (a_decls, a_live, a_handle), (b_decls, b_live, b_handle) = handles
    inner = pad + "    "
    body = [a_handle, b_handle, f"_mpp_op_{gid}.run(_mpp_a, _mpp_b, {src.id});"]
    staged = any(spaces)
    if staged:
        # Every thread's staging writes are visible to every SIMD group's run.
        parts.append(f"{pad}threadgroup_barrier(mem_flags::mem_threadgroup);")
    parts.append(f"{pad}{{")
    parts.extend(inner + line for line in a_decls + b_decls)
    live = a_live + b_live
    if live:
        # An empty view adds nothing to the accumulator, and MPP reads real
        # rows for one at 64-row descriptors.  Every term is a tile offset, a
        # uniform base, bound or outer index, so every thread of the scope
        # takes the same branch.
        parts.append(f"{inner}if ({' && '.join(live)}) {{")
        parts.extend(f"{inner}    {line}" for line in body)
        parts.append(f"{inner}}}")
    else:
        parts.extend(inner + line for line in body)
    parts.append(f"{pad}}}")
    if staged:
        # Every run has read the staged tiles before the next K tile's writes.
        parts.append(f"{pad}threadgroup_barrier(mem_flags::mem_threadgroup);")
    if state.coop.get(name) != root:
        parts.append(f"{pad}thread auto& {name} = {src.id};")
        state.coop[name] = root
        state.declare(name)


def _emit_for(
    stmt: ast.For,
    parts: list[str],
    indent: int,
    state: EmitState,
    subs: dict[str, str] | None = None,
) -> None:
    """Emit MSL for a for-loop.  Supports two iterator shapes:

    - ``range(...)`` / ``tl.range(...)`` → C-style ``for (int v = ..; ...)``
    - ``_helion_mpp_elements(coop, ...)``
      → MPP cooperative_tensor iteration ``for (auto _it = coop.begin();
      _it != coop.end(); _it++)`` with the tensor's names substituted by
      ``(*_it)`` in the body, or a fragment group's unrolled loop over the
      elements each lane holds (``fragment_mma.fragment_elements``)
    - ``_helion_mpp_fragments(acc, steps, facts, tables, k_step)`` → a
      fragment group's op over one K tile (``fragment_mma.emit_fragment_run``)
    """
    pad = " " * indent
    it = stmt.iter
    if not isinstance(it, ast.Call):
        raise exc.BackendUnsupported("metal", f"for loop iter: {ast.unparse(it)}")

    func = it.func

    # --- A fragment group's op over one K tile (fragment_mma.fragment_run) ---
    if isinstance(func, ast.Name) and func.id == FRAG_RUN:
        emit_fragment_run(
            stmt,
            parts,
            indent,
            state,
            lambda body, out, ind, st: _emit_stmts(body, out, indent=ind, state=st),
        )
        return

    # --- MPP element loop over a cooperative tensor (mpp_matmul.elementize) ---
    if isinstance(func, ast.Name) and func.id == "_helion_mpp_elements":
        assert isinstance(it.args[0], ast.Name)
        coop = it.args[0].id
        root = state.coop[coop]
        # A fragment element is read once, as ``_fv``, and written through
        # ``state.put``.
        element = "_fv" if root in state.fragments else "(*_it)"
        element_subs = {
            **(subs or {}),
            **{name: element for name, r in state.coop.items() if r == root},
        }
        fenced = len(it.args) > 1 and ast.literal_eval(it.args[1])
        if fenced:
            # The loop reads a staged vector's tile: after its staging writes.
            parts.append(f"{pad}threadgroup_barrier(mem_flags::mem_threadgroup);")

        def body(ind: int) -> list[str]:
            lines: list[str] = []
            with state.isolated_scope():
                _emit_stmts(
                    stmt.body, lines, indent=ind, state=state, subs=element_subs
                )
            return lines

        if root in state.fragments:
            parts.extend(fragment_elements(coop, root, pad, state, body))
        else:
            # A cooperative tensor's multidimensional index is (column, row).
            parts.extend(
                (
                    f"{pad}for (auto _it = {coop}.begin(); _it != {coop}.end(); _it++) {{",
                    f"{pad}    if (!{coop}.is_valid_element(_it)) continue;",
                    f"{pad}    auto _md = {coop}.get_multidimensional_index(_it);",
                    f"{pad}    int _e_n = (int)_md[0];",
                    f"{pad}    int _e_m = (int)_md[1];",
                )
            )
            if root in state.coop_rounded:
                block_m, block_n = state.coop_tile[root]
                parts.append(
                    f"{pad}    if (_e_m >= {block_m} || _e_n >= {block_n}) continue;"
                )
            parts.extend(body(indent + 4))
            parts.append(f"{pad}}}")
        if fenced:
            # Every element has read the tile before its next writes.
            parts.append(f"{pad}threadgroup_barrier(mem_flags::mem_threadgroup);")
        return

    # --- Regular range / tl.range ---
    assert isinstance(stmt.target, ast.Name)
    loop_var = stmt.target.id

    start = "0"
    end = "0"
    step = "1"

    it = stmt.iter
    # range(end) / range(start, end) / range(start, end, step)
    if not isinstance(it, ast.Call):
        raise exc.BackendUnsupported("metal", f"for loop iter: {ast.unparse(it)}")
    func = it.func
    is_range = isinstance(func, ast.Name) and func.id == "range"
    is_tl_range = (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "tl"
        and func.attr == "range"
    )
    if not (is_range or is_tl_range):
        raise exc.BackendUnsupported("metal", f"for loop iter: {ast.unparse(it)}")
    args = it.args
    if len(args) == 1:
        end = _ast_expr_to_msl(args[0], subs=subs)
    elif len(args) == 2:
        start = _ast_expr_to_msl(args[0], subs=subs)
        end = _ast_expr_to_msl(args[1], subs=subs)
    elif len(args) >= 3:
        start = _ast_expr_to_msl(args[0], subs=subs)
        end = _ast_expr_to_msl(args[1], subs=subs)
        step = _ast_expr_to_msl(args[2], subs=subs)

    before, top = prefetch_loads(stmt.body, loop_var, end, pad, state)
    parts += before
    parts.append(
        f"{pad}for (int {loop_var} = {start}; {loop_var} < {end}; {loop_var} += {step}) {{"
    )
    parts += top
    with state.block_scope():
        state.declare(loop_var)
        _emit_stmts(stmt.body, parts, indent=indent + 4, state=state, subs=subs)
    parts.append(f"{pad}}}")


def _ptr_access_expr(ptr_node: ast.AST, subs: dict[str, str] | None = None) -> str:
    """Convert a pointer expression to an MSL memory access.

    Recognizes ``buf + offset`` and emits ``buf[offset]``
    instead of ``*(buf + offset)``.
    """
    if isinstance(ptr_node, ast.BinOp) and isinstance(ptr_node.op, ast.Add):
        base = ptr_node.left
        if isinstance(base, ast.Name):
            offset = _ast_expr_to_msl(ptr_node.right, subs=subs)
            return f"{base.id}[{offset}]"
    return f"*({_ast_expr_to_msl(ptr_node, subs=subs)})"


def _is_call_to(node: ast.AST, name: str) -> bool:
    """Return True if *node* is a ``name(...)`` call."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    )


# ---------------------------------------------------------------------------
# AST-to-MSL expression converter (module-level pure functions)
# ---------------------------------------------------------------------------


def _ast_expr_to_msl(
    node: ast.AST,
    *,
    subs: dict[str, str] | None = None,
) -> str:
    """Recursively convert an AST expression node to MSL C++ string.

    *subs*: optional substitution dict mapping variable names to replacement
    strings.  Used by the epilogue loop to replace MMA result variables with
    ``(*_it)``.  Propagated through all recursive calls.
    """
    if isinstance(node, ast.Name):
        if subs and node.id in subs:
            return subs[node.id]
        return node.id

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        if isinstance(node.value, float):
            return repr(node.value)
        if isinstance(node.value, int):
            return str(node.value)
        if node.value is None:
            return "None"  # used as sentinel (e.g. mask=None)
        raise exc.BackendUnsupported(
            "metal", f"constant type: {type(node.value).__name__}"
        )

    if isinstance(node, ast.UnaryOp):
        operand = _ast_expr_to_msl(node.operand, subs=subs)
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
        if isinstance(node.op, ast.Not):
            return f"(!{operand})"
        if isinstance(node.op, ast.Invert):
            return f"(~{operand})"
        if isinstance(node.op, ast.UAdd):
            return f"(+{operand})"
        raise exc.BackendUnsupported("metal", f"unary op: {type(node.op).__name__}")

    if isinstance(node, ast.BinOp):
        left = _ast_expr_to_msl(node.left, subs=subs)
        right = _ast_expr_to_msl(node.right, subs=subs)
        op = node.op
        if isinstance(op, ast.Mult):
            if isinstance(node.right, ast.Constant) and node.right.value == 1:
                return left
            if isinstance(node.left, ast.Constant) and node.left.value == 1:
                return right
        if isinstance(op, ast.FloorDiv):
            # FloorDiv comes from Helion's index arithmetic (sympy FloorDiv),
            # not from MetalOverrides (which emits c10.metal.floor_divide as
            # a function call that the generic handler processes).
            return f"c10::metal::floor_divide({left}, {right})"
        if isinstance(op, ast.Add):
            op_str = "+"
        elif isinstance(op, ast.Sub):
            op_str = "-"
        elif isinstance(op, ast.Mult):
            op_str = "*"
        elif isinstance(op, ast.Div):
            op_str = "/"
        elif isinstance(op, ast.Mod):
            op_str = "%"
        elif isinstance(op, ast.BitAnd):
            op_str = "&"
        elif isinstance(op, ast.BitOr):
            op_str = "|"
        elif isinstance(op, ast.BitXor):
            op_str = "^"
        elif isinstance(op, ast.LShift):
            op_str = "<<"
        elif isinstance(op, ast.RShift):
            op_str = ">>"
        else:
            raise exc.BackendUnsupported("metal", f"binary op: {type(op).__name__}")
        return f"({left} {op_str} {right})"

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            op_str = " && "
        elif isinstance(node.op, ast.Or):
            op_str = " || "
        else:
            raise exc.BackendUnsupported("metal", f"bool op: {type(node.op).__name__}")
        parts = [_ast_expr_to_msl(v, subs=subs) for v in node.values]
        return f"({op_str.join(parts)})"

    if isinstance(node, ast.Compare):
        left = _ast_expr_to_msl(node.left, subs=subs)
        parts = [left]
        for op, comp in zip(node.ops, node.comparators, strict=True):
            if isinstance(op, ast.Eq):
                parts.append("==")
            elif isinstance(op, ast.NotEq):
                parts.append("!=")
            elif isinstance(op, ast.Lt):
                parts.append("<")
            elif isinstance(op, ast.LtE):
                parts.append("<=")
            elif isinstance(op, ast.Gt):
                parts.append(">")
            elif isinstance(op, ast.GtE):
                parts.append(">=")
            else:
                raise exc.BackendUnsupported(
                    "metal", f"comparison op: {type(op).__name__}"
                )
            parts.append(_ast_expr_to_msl(comp, subs=subs))
        return f"({' '.join(parts)})"

    if isinstance(node, ast.IfExp):
        test = _ast_expr_to_msl(node.test, subs=subs)
        body = _ast_expr_to_msl(node.body, subs=subs)
        orelse = _ast_expr_to_msl(node.orelse, subs=subs)
        return f"({test} ? {body} : {orelse})"

    if isinstance(node, ast.Call):
        return _ast_call_to_msl(node, subs=subs)

    if isinstance(node, ast.Subscript):
        return _ast_subscript_to_msl(node, subs=subs)

    if isinstance(node, ast.Attribute):
        # C++ namespace access: metal.precise.sin → metal::precise::sin, and
        # _msl_global.metal.precise.atan2 → ::metal::precise::atan2 (see
        # msl_expr).
        if isinstance(node.value, ast.Name) and node.value.id == GLOBAL_SCOPE:
            return f"::{node.attr}"
        value = _ast_expr_to_msl(node.value, subs=subs)
        sep = "::" if _is_cpp_namespace_root(node) else "."
        return f"{value}{sep}{node.attr}"

    raise exc.BackendUnsupported("metal", f"AST expression type: {type(node).__name__}")


def _ast_call_to_msl(node: ast.AST, subs: dict[str, str] | None = None) -> str:
    """Convert an AST Call node to MSL with optional variable substitutions.

    Handles tl.load specifically; everything else falls through to a
    generic ``func(args, ...)`` emit (the path used by MetalOverrides
    expressions like ``c10.metal.max`` and ``metal.precise.sin``).
    """
    assert isinstance(node, ast.Call)
    func = node.func

    if isinstance(func, ast.Name) and func.id == "_metal_lane_get":
        array, lane = (_ast_expr_to_msl(a, subs=subs) for a in node.args)
        return f"{array}[{lane}]"

    # Inductor's constant_repr spells non-finite floats as float("inf") /
    # float("-inf") / float("nan"); MSL has literals for those.  Reduction
    # identities for max/min are the main source.
    if (
        isinstance(func, ast.Name)
        and func.id == "float"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        text = node.args[0].value.strip().lower()
        literal = _NON_FINITE_FLOATS.get(text)
        if literal is None:
            raise exc.BackendUnsupported("metal", f"float constant: {text!r}")
        return literal

    # tl.load(ptr + offset, mask, other=0) → (mask ? *(ptr+offset) : (other))
    if (load := tl_load(node)) is not None:
        access = _ptr_access_expr(load.args[0], subs=subs)
        masked = load_mask_other(load)
        if masked is None:
            return access
        mask, other = masked
        if other is None:
            raise exc.BackendUnsupported(
                "metal", "tl.load with mask requires 'other' argument"
            )
        mask_msl = _ast_expr_to_msl(mask, subs=subs)
        return f"({mask_msl} ? {access} : ({_ast_expr_to_msl(other, subs=subs)}))"

    # A C++ cast or MSL template function: static_cast(T, x) → static_cast<T>(x)
    if _template_name(func) in TEMPLATE_CALLS:
        callee = _ast_expr_to_msl(func, subs=subs)
        template_arg, *args = node.args
        args_msl = [_ast_expr_to_msl(a, subs=subs) for a in args]
        return f"{callee}<{_template_arg_to_msl(template_arg, subs)}>({', '.join(args_msl)})"

    # Generic function call (main path for MetalOverrides expressions)
    func_msl = _ast_expr_to_msl(func, subs=subs)
    args_msl = [_ast_expr_to_msl(a, subs=subs) for a in node.args]
    return f"{func_msl}({', '.join(args_msl)})"


def _template_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _template_arg_to_msl(node: ast.AST, subs: dict[str, str] | None) -> str:
    """A template argument: ``float``, ``metal::half``, ``decltype(a + b)``."""
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == GLOBAL_SCOPE:
            return f"::{node.attr}"
        return f"{_template_arg_to_msl(node.value, subs)}::{node.attr}"
    if _is_call_to(node, "decltype"):
        assert isinstance(node, ast.Call)
        (operand,) = node.args
        inner = _ast_expr_to_msl(operand, subs=subs)
        # decltype((*_it)) is a reference type; the element value type is
        # decltype(+(*_it)).
        return "decltype(+(*_it))" if inner == "(*_it)" else f"decltype({inner})"
    return _ast_expr_to_msl(node, subs=subs)


#: ``m ? v : other`` as a call, so ``v`` (a load at a safe index) is
#: evaluated unconditionally; the result has the conditional's own type.
MASKED_LOAD = "_helion_masked"
MASKED_LOAD_PREAMBLE = f"""template <typename T, typename O>
inline auto {MASKED_LOAD}(T v, bool m, O other) {{
    return m ? v : other;
}}"""


def select_masked_loads(
    stmts: list[ast.stmt], candidates: frozenset[str]
) -> list[ast.stmt]:
    """Rewrite each masked ``tl.load(buf + offset, mask, other)`` of a buffer
    in *candidates* to ``_helion_masked(tl.load(buf + (offset if mask else
    0)), mask, other)``: load unconditionally, at element 0 when masked off,
    and select afterwards.  A load under the condition is a branch per load,
    which keeps Metal from issuing loads together.

    *candidates* are the buffers with an element 0 that no other parameter
    shares storage with.  A candidate's loads still stay conditional where
    the kernel uses it other than through ``tl.load`` (a store, an atomic, an
    MPP operand), since another thread may write element 0 while a
    masked-off lane reads it."""
    loaded = Counter(
        load.args[0].left.id
        for stmt in stmts
        for node in ast.walk(stmt)
        if (load := tl_load(node)) is not None
        and isinstance(load.args[0], ast.BinOp)
        and isinstance(load.args[0].left, ast.Name)
    )
    named = Counter(
        node.id
        for stmt in stmts
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name)
    )
    read_only = {name for name in candidates if loaded[name] == named[name]}

    class Select(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if (load := tl_load(node)) is None or (
                masked := load_mask_other(load)
            ) is None:
                return node
            mask, other = masked
            ptr = load.args[0]
            if not (
                other is not None
                and isinstance(ptr, ast.BinOp)
                and isinstance(ptr.op, ast.Add)
                and isinstance(ptr.left, ast.Name)
                and ptr.left.id in read_only
            ):
                return node
            safe = ast.BinOp(
                left=ptr.left,
                op=ast.Add(),
                right=ast.IfExp(
                    test=copy.deepcopy(mask), body=ptr.right, orelse=ast.Constant(0)
                ),
            )
            return ast.Call(
                func=ast.Name(id=MASKED_LOAD, ctx=ast.Load()),
                args=[
                    ast.Call(func=node.func, args=[safe], keywords=[]),
                    mask,
                    other,
                ],
                keywords=[],
            )

    return [Select().visit(stmt) for stmt in stmts]


#: What Inductor's ``constant_repr`` emits for non-finite float constants.
_NON_FINITE_FLOATS = {
    "inf": "INFINITY",
    "-inf": "(-INFINITY)",
    "nan": "NAN",
}

_CPP_NAMESPACE_ROOTS = frozenset({"metal", "c10", REDUCTION_NAMESPACE, GLOBAL_SCOPE})


def _is_cpp_namespace_root(node: ast.Attribute) -> bool:
    """Return True if *node* is an attribute chain rooted at a C++ namespace.

    ``msl_expr`` parses C++ namespace syntax (``metal::precise::sin``) into
    Python dot notation (``metal.precise.sin``).  This detects those chains so
    the walker can emit ``::`` instead of ``.``.
    """
    while isinstance(node.value, ast.Attribute):
        node = node.value
    return isinstance(node.value, ast.Name) and node.value.id in _CPP_NAMESPACE_ROOTS


def _is_broadcast_subscript(index: ast.expr) -> bool:
    """Return True for a pure reshape/broadcast index such as ``[:, None]``."""
    elts = index.elts if isinstance(index, ast.Tuple) else [index]
    if not elts:
        return False
    return all(
        (
            isinstance(elt, ast.Slice)
            and elt.lower is None
            and elt.upper is None
            and elt.step is None
        )
        or (isinstance(elt, ast.Constant) and elt.value is None)
        for elt in elts
    )


def _ast_subscript_to_msl(node: ast.AST, subs: dict[str, str] | None = None) -> str:
    """Convert an AST Subscript node to MSL.

    Only simple index subscripts (e.g. ``tgid[0]``) are supported, plus
    tile-shape broadcasts (``x[:, None]``), which Helion emits when a reduced
    value is broadcast back over the reduction axis.  Metal values are
    per-thread scalars, so those are a no-op.
    """
    assert isinstance(node, ast.Subscript)
    if _is_broadcast_subscript(node.slice):
        return _ast_expr_to_msl(node.value, subs=subs)
    buf_name = _ast_expr_to_msl(node.value, subs=subs)
    idx = _ast_expr_to_msl(node.slice, subs=subs)
    return f"{buf_name}[{idx}]"
