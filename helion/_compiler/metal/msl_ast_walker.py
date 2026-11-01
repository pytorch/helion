"""Python AST → MSL C++ statement-level translation functions.

Provides pure functions for converting Python AST nodes to MSL C++ text.
Used by ``metal_kernel._generate_msl`` to translate the body of a
``@metal_jit`` decorated function into MSL source code.

MetalOverrides (reused from Inductor's MPS codegen) emits MSL expression strings
using ``c10::metal::`` and ``metal::precise::`` namespaces.  The ``::`` namespace
separator is replaced with ``.`` before Python AST parsing and converted back
to ``::`` by this module when emitting C++.

Handles:
  - Statement-level Python→C++ translation (assignments, if/for, etc.)
  - ``tl.load(ptr + offset, mask, other=0)`` → ``(mask ? *(ptr+offset) : (T)(0))``
  - ``tl.store(ptr + offset, val, mask)`` → ``if (mask) { *(ptr+offset) = val; }``
  - ``static_cast<T>(x)`` parsed as Compare (from MetalOverrides.to_dtype)
  - C++ namespace restoration: ``metal.precise.sin`` → ``metal::precise::sin``
  - Generic function call fallback (main handler for MetalOverrides expressions)
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
from typing import Iterator

from ... import exc
from .mpp_graph_codegen import MPPSetupParams
from .msl_reduction import REDUCTION_NAMESPACE


@dataclasses.dataclass
class EmitState:
    """Mutable state passed through ``_emit_stmts`` calls.

    Tracks declared variable names (to avoid duplicate ``auto`` declarations)
    and MPP matmul setup parameters (keyed by setup variable name).

    Declarations are tracked per C++ block scope.  A rolled reduction emits
    several sibling ``for`` loops that assign the same Helion variable (e.g.
    ``rindex_1``); each needs its own ``auto`` because the previous loop's
    declaration went out of scope at its closing brace.

    MPPGraph lowering emits explicit ``_coop_iter`` loops. The walker derives
    the MMA-result substitution from the setup marker's ``fx_name``.
    """

    declared: set[str] = dataclasses.field(default_factory=set)
    mpp_setups: dict[str, MPPSetupParams] = dataclasses.field(default_factory=dict)
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
    expressions to MSL (e.g. ``{"acc": "(*_it)"}`` inside the coop
    iteration loop).  It's set by the ``_coop_iter`` branch of
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
            # Check for _metal_mpp_setup assignment
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "_metal_mpp_setup"
            ):
                params = _extract_mpp_setup_params(value)
                state.mpp_setups[target.id] = params
                _emit_mpp_setup(target.id, params, parts, indent=indent)
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
            elif _is_call_to(call, "_metal_lane_put"):
                assert isinstance(call, ast.Call)
                array, lane, value = (_ast_expr_to_msl(a, subs=subs) for a in call.args)
                parts.append(f"{pad}{array}[{lane}] = {value};")
            elif _is_call_to(call, "_helion_mpp_stage"):
                assert isinstance(call, ast.Call)
                buf, index, value, pred = (_ast_expr_to_msl(a) for a in call.args[:4])
                parts.append(f"{pad}if ({pred}) {{ {buf}[{index}] = {value}; }}")
            elif _is_call_to(call, "_helion_mpp_put"):
                # An element loop's write of the updated element (mpp_matmul's
                # _helion_mpp_update).
                assert isinstance(call, ast.Call)
                parts.append(
                    f"{pad}*_it = {_ast_expr_to_msl(call.args[0], subs=subs)};"
                )
            elif _is_call_to(call, "_helion_mpp_fence"):
                parts.append(
                    f"{pad}threadgroup_barrier(mem_flags::mem_device | "
                    "mem_flags::mem_threadgroup);"
                )
            elif _is_call_to(call, "_helion_mpp_materialize"):
                assert isinstance(call, ast.Call)
                _emit_coop_materialize(call, parts, pad, state)
            elif _is_call_to(call, "_helion_mpp_bind"):
                assert isinstance(call, ast.Call)
                name = ast.literal_eval(call.args[0])
                parts.append(f"{pad}auto {name} = {_ast_expr_to_msl(call.args[1])};")
                state.declare(name)
            elif _is_call_to(call, "_metal_mpp_k_step"):
                assert isinstance(call, ast.Call)
                setup_arg = call.args[0]
                assert isinstance(setup_arg, ast.Name)
                k_offset_expr = _ast_expr_to_msl(call.args[1])
                setup_params = state.mpp_setups[setup_arg.id]
                _emit_mpp_k_step(
                    setup_arg.id,
                    k_offset_expr,
                    parts,
                    indent=indent,
                    lhs_transposed=setup_params.lhs_transposed,
                    rhs_transposed=setup_params.rhs_transposed,
                )
            elif _is_call_to(call, "_metal_mpp_coop_store"):
                # Three positional args: setup_var (Name), out_name (str),
                # out_dtype (str).
                assert isinstance(call, ast.Call)
                setup_arg, out_name_node, out_dtype_node = call.args
                assert isinstance(setup_arg, ast.Name)
                _emit_mpp_coop_store(
                    setup_arg.id,
                    _ast_str_value(out_name_node),
                    _ast_str_value(out_dtype_node),
                    parts,
                    indent,
                )
            elif _is_call_to(call, "_metal_mpp_scalar_store"):
                # Same args as _metal_mpp_coop_store; mixed acc/out path.
                assert isinstance(call, ast.Call)
                setup_arg, out_name_node, out_dtype_node = call.args
                assert isinstance(setup_arg, ast.Name)
                _emit_mpp_scalar_store(
                    setup_arg.id,
                    _ast_str_value(out_name_node),
                    _ast_str_value(out_dtype_node),
                    parts,
                    indent,
                )
            elif _is_call_to(call, "_metal_mpp_threadgroup_barrier"):
                parts.append(f"{pad}threadgroup_barrier(mem_flags::mem_device);")
            elif _is_call_to(call, "_coop_writeback"):
                # The argument is a plain local variable name — subs do NOT
                # apply (we want the local value, not a recursive *_it lookup).
                assert isinstance(call, ast.Call)
                writeback_expr = _ast_expr_to_msl(call.args[0])
                parts.append(f"{pad}*_it = {writeback_expr};")
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
    whole SIMD groups.
    """
    gid, tm, tn, tk, dtypes, lhs_t, rhs_t, fill = (
        ast.literal_eval(a) for a in call.args[:8]
    )
    lhs_staged, rhs_staged, block_m, block_n, seed = (
        ast.literal_eval(a) for a in call.args[9:14]
    )
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
        if e == size or e.lstrip("-").isdigit():
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
    """Emit MSL for a for-loop.  Supports three iterator shapes:

    - ``range(...)`` / ``tl.range(...)`` → C-style ``for (int v = ..; ...)``
    - ``_coop_iter(setup_var)``
      → MPP cooperative_tensor iteration ``for (auto _it = _coop.begin();
      _it != _coop.end(); _it++)`` with the plan's substitution dict
      activated for the body.
    """
    pad = " " * indent
    it = stmt.iter
    if not isinstance(it, ast.Call):
        raise exc.BackendUnsupported("metal", f"for loop iter: {ast.unparse(it)}")

    func = it.func

    # --- MPP element loop over a cooperative tensor (mpp_matmul.elementize) ---
    if isinstance(func, ast.Name) and func.id == "_helion_mpp_elements":
        assert isinstance(it.args[0], ast.Name)
        coop = it.args[0].id
        root = state.coop[coop]
        element_subs = {
            **(subs or {}),
            **{name: "(*_it)" for name, r in state.coop.items() if r == root},
        }
        fenced = len(it.args) > 1 and ast.literal_eval(it.args[1])
        if fenced:
            # The loop reads a staged vector's tile: after its staging writes.
            parts.append(f"{pad}threadgroup_barrier(mem_flags::mem_threadgroup);")
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
        with state.isolated_scope():
            _emit_stmts(
                stmt.body, parts, indent=indent + 4, state=state, subs=element_subs
            )
        parts.append(f"{pad}}}")
        if fenced:
            # Every element has read the tile before its next writes.
            parts.append(f"{pad}threadgroup_barrier(mem_flags::mem_threadgroup);")
        return

    # --- MPP cooperative iterator (synthesized by the pass) ---
    if isinstance(func, ast.Name) and func.id == "_coop_iter":
        assert len(it.args) == 1 and isinstance(it.args[0], ast.Name), (
            "_coop_iter takes exactly one setup-var argument"
        )
        setup_name = it.args[0].id
        setup = state.mpp_setups[setup_name]
        coop_subs = {setup.fx_name: "(*_it)"} if setup.fx_name else {}
        coop = _scoped_mpp_name(setup_name, "_coop")
        parts.extend(
            (
                f"{pad}// Epilogue: element-wise ops on cooperative_tensor",
                f"{pad}for (auto _it = {coop}.begin(); _it != {coop}.end(); _it++) {{",
            )
        )
        with state.block_scope():
            _emit_stmts(
                stmt.body, parts, indent=indent + 4, state=state, subs=coop_subs
            )
        parts.append(f"{pad}}}")
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

    parts.append(
        f"{pad}for (int {loop_var} = {start}; {loop_var} < {end}; {loop_var} += {step}) {{"
    )
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


# ---------------------------------------------------------------------------
# MPP matmul2d emission helpers
# ---------------------------------------------------------------------------


def _is_call_to(node: ast.AST, name: str) -> bool:
    """Return True if *node* is a ``name(...)`` call."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    )


def _scoped_mpp_name(setup_name: str, suffix: str) -> str:
    """Return an MSL identifier scoped by the setup variable name."""
    return f"{setup_name}{suffix}"


def _ast_str_value(node: ast.AST) -> str:
    """Extract string constant from AST node."""
    assert isinstance(node, ast.Constant) and isinstance(node.value, str)
    return node.value


def _ast_int_value(node: ast.AST) -> int:
    """Extract integer constant from AST node."""
    assert isinstance(node, ast.Constant) and isinstance(node.value, int)
    return node.value


def _ast_name_value(node: ast.AST) -> str:
    """Extract a bare variable name from AST node."""
    assert isinstance(node, ast.Name)
    return node.id


def _extract_mpp_setup_params(node: ast.Call) -> MPPSetupParams:
    """Extract parameters from ``_metal_mpp_setup(...)`` call.

    Positional args (in order):
      0: lhs tensor name,  1: rhs tensor name,
      2-4: M, N, K,        5-7: TILE_M, TILE_N, TILE_K,
      8: NUM_SG,           9: in_dtype,       10: acc_dtype,
      11: FX node name of the MMA op (or ""),
      12: lhs transpose flag (0/1),  13: rhs transpose flag (0/1),
      14: lhs storage row width,  15: rhs storage row width,
      16: M tile-offset variable,  17: N tile-offset variable (bare names).

    Output tensor name and dtype are deliberately carried by the explicit
    ``_metal_mpp_coop_store(setup, out_name, out_dtype)`` marker instead of
    this setup marker.  The setup creates the operand tensors, operator, and
    cooperative accumulator before the epilogue runs; the final output view is
    determined by the root ``memory_ops.store`` that the MPPGraph rewrite
    consumes.  Keeping the store target on the store marker makes that
    ownership explicit and avoids assuming at setup time that every MPPGraph
    will store to the same destination shape/dtype path.
    """
    args = node.args
    assert len(args) == 18, (
        f"_metal_mpp_setup expects 18 positional args, got {len(args)}"
    )
    return MPPSetupParams(
        lhs=_ast_str_value(args[0]),
        rhs=_ast_str_value(args[1]),
        M=_ast_int_value(args[2]),
        N=_ast_int_value(args[3]),
        K=_ast_int_value(args[4]),
        TILE_M=_ast_int_value(args[5]),
        TILE_N=_ast_int_value(args[6]),
        TILE_K=_ast_int_value(args[7]),
        NUM_SG=_ast_int_value(args[8]),
        in_dtype=_ast_str_value(args[9]),
        acc_dtype=_ast_str_value(args[10]),
        fx_name=_ast_str_value(args[11]) or None,
        lhs_transposed=bool(_ast_int_value(args[12])),
        rhs_transposed=bool(_ast_int_value(args[13])),
        lhs_storage_row_width=_ast_int_value(args[14]),
        rhs_storage_row_width=_ast_int_value(args[15]),
        m_offset=_ast_name_value(args[16]),
        n_offset=_ast_name_value(args[17]),
    )


def _emit_mpp_setup(
    setup_name: str,
    params: MPPSetupParams,
    parts: list[str],
    indent: int,
) -> None:
    """Emit MPP matmul2d setup MSL.

    Declares the input tensor handles (``_lhs`` / ``_rhs``), the
    ``matmul2d_descriptor`` and operator, the per-axis tile indices, the
    operand slices, and the cooperative_tensor accumulator.  The output
    tensor handle (``_C``) is declared by :func:`_emit_mpp_coop_store`
    because its name and dtype are sourced from the trailing
    ``tl.store(out_ptr, ...)`` rather than the setup.

    Handles cover each operand's storage: transposed operands swap the
    handle extents and set the descriptor transpose flag, while row-padded
    operands widen the handle's physical row width.
    """
    pad = " " * indent
    M_var = _scoped_mpp_name(setup_name, "_M")
    N_var = _scoped_mpp_name(setup_name, "_N")
    K_var = _scoped_mpp_name(setup_name, "_K")
    TILE_M_var = _scoped_mpp_name(setup_name, "_TILE_M")
    TILE_N_var = _scoped_mpp_name(setup_name, "_TILE_N")
    TILE_K_var = _scoped_mpp_name(setup_name, "_TILE_K")
    NUM_SG_var = _scoped_mpp_name(setup_name, "_NUM_SG")
    lhs_var = _scoped_mpp_name(setup_name, "_lhs")
    rhs_var = _scoped_mpp_name(setup_name, "_rhs")
    desc_var = _scoped_mpp_name(setup_name, "_desc")
    op_var = _scoped_mpp_name(setup_name, "_op")
    ty_var = _scoped_mpp_name(setup_name, "_ty")
    tx_var = _scoped_mpp_name(setup_name, "_tx")
    lhs_slice_var = _scoped_mpp_name(setup_name, "_lhs_slice")
    rhs_slice_var = _scoped_mpp_name(setup_name, "_rhs_slice")
    coop_var = _scoped_mpp_name(setup_name, "_coop")

    needs_k_loop = params.TILE_K < params.K
    mm_mode = "multiply_accumulate" if needs_k_loop else "multiply"
    lhs_transposed = "true" if params.lhs_transposed else "false"
    rhs_transposed = "true" if params.rhs_transposed else "false"
    if params.lhs_transposed:
        lhs_ext1 = params.K
        lhs_slice_shape = f"{params.TILE_M}, {params.TILE_K}"
        lhs_slice_offset = f"{ty_var} * {TILE_M_var}, 0"
    else:
        lhs_ext1 = params.M
        lhs_slice_shape = f"{params.TILE_K}, {params.TILE_M}"
        lhs_slice_offset = f"0, {ty_var} * {TILE_M_var}"
    if params.rhs_transposed:
        rhs_ext1 = params.N
        rhs_slice_shape = f"{params.TILE_K}, {params.TILE_N}"
        rhs_slice_offset = f"0, {tx_var} * {TILE_N_var}"
    else:
        rhs_ext1 = params.K
        rhs_slice_shape = f"{params.TILE_N}, {params.TILE_K}"
        rhs_slice_offset = f"{tx_var} * {TILE_N_var}, 0"

    parts.extend(
        [
            f"{pad}// MPP matmul2d setup",
            f"{pad}constexpr int {M_var} = {params.M};",
            f"{pad}constexpr int {N_var} = {params.N};",
            f"{pad}constexpr int {K_var} = {params.K};",
            f"{pad}constexpr int {TILE_M_var} = {params.TILE_M};",
            f"{pad}constexpr int {TILE_N_var} = {params.TILE_N};",
            f"{pad}constexpr int {TILE_K_var} = {params.TILE_K};",
            f"{pad}constexpr int {NUM_SG_var} = {params.NUM_SG};",
            "",
            f"{pad}auto {lhs_var} = tensor<device {params.in_dtype}, dextents<int32_t, 2>, tensor_inline>(",
            f"{pad}    {params.lhs}, dextents<int32_t, 2>({params.lhs_storage_row_width}, {lhs_ext1}));",
            f"{pad}auto {rhs_var} = tensor<device {params.in_dtype}, dextents<int32_t, 2>, tensor_inline>(",
            f"{pad}    {params.rhs}, dextents<int32_t, 2>({params.rhs_storage_row_width}, {rhs_ext1}));",
            "",
            f"{pad}constexpr auto {desc_var} = matmul2d_descriptor(",
            f"{pad}    {TILE_M_var}, {TILE_N_var}, {TILE_K_var},",
            f"{pad}    {lhs_transposed}, {rhs_transposed}, false, matmul2d_descriptor::mode::{mm_mode});",
            f"{pad}matmul2d<{desc_var}, execution_simdgroups<{NUM_SG_var}>> {op_var};",
            "",
            # Tile indices from the branch-local tile offsets.  Each top-level
            # loop computes its own offsets from its rebased program ids, so
            # unlike the raw launch id these are exact in every root grid.
            f"{pad}uint {ty_var} = ({params.m_offset} / {TILE_M_var});",
            f"{pad}uint {tx_var} = ({params.n_offset} / {TILE_N_var});",
            "",
            # The setup slices only define the cooperative_tensor type.  Keep
            # their extents static so MPP can allocate the cooperative tile
            # without falling back to deferred dynamic storage.
            f"{pad}auto {lhs_slice_var} = {lhs_var}.slice<{lhs_slice_shape}>(",
            f"{pad}    {lhs_slice_offset});",
            f"{pad}auto {rhs_slice_var} = {rhs_var}.slice<{rhs_slice_shape}>(",
            f"{pad}    {rhs_slice_offset});",
            "",
            f"{pad}auto {coop_var} = {op_var}.get_destination_cooperative_tensor<",
            f"{pad}    decltype({lhs_slice_var}), decltype({rhs_slice_var}), {params.acc_dtype}>();",
        ]
    )
    if needs_k_loop:
        # multiply_accumulate mode (used when needs_k_loop) accumulates
        # onto _coop's existing values, so zero-init is required.
        parts.extend(
            [
                f"{pad}for (auto _it = {coop_var}.begin(); _it != {coop_var}.end(); _it++)",
                f"{pad}    *_it = ({params.acc_dtype})(0);",
            ]
        )


def _emit_mpp_k_step(
    setup_name: str,
    k_offset_expr: str,
    parts: list[str],
    indent: int,
    *,
    lhs_transposed: bool,
    rhs_transposed: bool,
) -> None:
    """Emit MPP K-tile step MSL.

    Slice offsets mirror the setup slices: a transposed operand stores K
    along the handle's second axis, so the K offset and tile offset swap.
    """
    pad = " " * indent
    lhs_var = _scoped_mpp_name(setup_name, "_lhs")
    rhs_var = _scoped_mpp_name(setup_name, "_rhs")
    lhs_k_var = _scoped_mpp_name(setup_name, "_lhs_k")
    rhs_k_var = _scoped_mpp_name(setup_name, "_rhs_k")
    TILE_M_var = _scoped_mpp_name(setup_name, "_TILE_M")
    TILE_N_var = _scoped_mpp_name(setup_name, "_TILE_N")
    ty_var = _scoped_mpp_name(setup_name, "_ty")
    tx_var = _scoped_mpp_name(setup_name, "_tx")
    op_var = _scoped_mpp_name(setup_name, "_op")
    coop_var = _scoped_mpp_name(setup_name, "_coop")
    if lhs_transposed:
        lhs_k_slice = f"{ty_var} * {TILE_M_var}, {k_offset_expr}"
    else:
        lhs_k_slice = f"{k_offset_expr}, {ty_var} * {TILE_M_var}"
    if rhs_transposed:
        rhs_k_slice = f"{k_offset_expr}, {tx_var} * {TILE_N_var}"
    else:
        rhs_k_slice = f"{tx_var} * {TILE_N_var}, {k_offset_expr}"
    parts.extend(
        [
            f"{pad}auto {lhs_k_var} = {lhs_var}.slice({lhs_k_slice});",
            f"{pad}auto {rhs_k_var} = {rhs_var}.slice({rhs_k_slice});",
            f"{pad}{op_var}.run({lhs_k_var}, {rhs_k_var}, {coop_var});",
        ]
    )


def _emit_mpp_coop_store(
    setup_name: str,
    out_name: str,
    out_dtype: str,
    parts: list[str],
    indent: int,
) -> None:
    """Emit the cooperative_tensor → device memory store.

    Declares the output tensor handle (``_C``) inline using *out_name* /
    *out_dtype* from the explicit MPPGraph store marker and emits
    ``_coop.store(_Cs)``.

    The accumulator dtype is set in :func:`_emit_mpp_setup`.  The
    cooperative store requires the accumulator and output dtypes to match;
    mixed pairs instead use :func:`_emit_mpp_scalar_store`.
    """
    pad = " " * indent
    C_var = _scoped_mpp_name(setup_name, "_C")
    Cs_var = _scoped_mpp_name(setup_name, "_Cs")
    M_var = _scoped_mpp_name(setup_name, "_M")
    N_var = _scoped_mpp_name(setup_name, "_N")
    TILE_M_var = _scoped_mpp_name(setup_name, "_TILE_M")
    TILE_N_var = _scoped_mpp_name(setup_name, "_TILE_N")
    ty_var = _scoped_mpp_name(setup_name, "_ty")
    tx_var = _scoped_mpp_name(setup_name, "_tx")
    coop_var = _scoped_mpp_name(setup_name, "_coop")
    parts.extend(
        [
            f"{pad}auto {C_var} = tensor<device {out_dtype}, dextents<int32_t, 2>, tensor_inline>(",
            f"{pad}    {out_name}, dextents<int32_t, 2>({N_var}, {M_var}));",
            f"{pad}auto {Cs_var} = {C_var}.slice({tx_var} * {TILE_N_var}, {ty_var} * {TILE_M_var});",
            f"{pad}{coop_var}.store({Cs_var});",
        ]
    )


def _emit_mpp_scalar_store(
    setup_name: str,
    out_name: str,
    out_dtype: str,
    parts: list[str],
    indent: int,
) -> None:
    """Emit a mixed-dtype cooperative → device memory store.

    Declares the same ``_C`` / ``_Cs`` output view as
    :func:`_emit_mpp_coop_store`, then converts and stores each accumulator
    element individually.  The cooperative ``store`` requires matching
    dtypes, so mixed accumulator/output pairs (e.g. fp32 accumulate with
    fp16 output) cannot use it.

    Each thread visits exactly the accumulator elements it owns.  Elements are
    written only when ``is_valid_element`` holds and the tile-local
    coordinates are inside the matrix bounds, so partial tiles never write
    out of bounds (unlike ``_coop.store``, the raw ``operator[]`` does no
    destination-side masking).  Coordinates come from
    ``get_multidimensional_index`` into the tile-local ``_Cs`` view; the
    multidimensional index order matches the ``_Cs`` dim order, so the
    indices transfer directly.
    """
    pad = " " * indent
    C_var = _scoped_mpp_name(setup_name, "_C")
    Cs_var = _scoped_mpp_name(setup_name, "_Cs")
    M_var = _scoped_mpp_name(setup_name, "_M")
    N_var = _scoped_mpp_name(setup_name, "_N")
    TILE_M_var = _scoped_mpp_name(setup_name, "_TILE_M")
    TILE_N_var = _scoped_mpp_name(setup_name, "_TILE_N")
    ty_var = _scoped_mpp_name(setup_name, "_ty")
    tx_var = _scoped_mpp_name(setup_name, "_tx")
    coop_var = _scoped_mpp_name(setup_name, "_coop")
    it_var = _scoped_mpp_name(setup_name, "_s_it")
    md_var = _scoped_mpp_name(setup_name, "_s_md")
    idx_var = _scoped_mpp_name(setup_name, "_s_idx")
    n_base_var = _scoped_mpp_name(setup_name, "_s_n_base")
    m_base_var = _scoped_mpp_name(setup_name, "_s_m_base")
    parts.extend(
        [
            f"{pad}// MPP mixed-dtype store: convert accumulator to {out_dtype}.",
            f"{pad}auto {C_var} = tensor<device {out_dtype}, dextents<int32_t, 2>, tensor_inline>(",
            f"{pad}    {out_name}, dextents<int32_t, 2>({N_var}, {M_var}));",
            f"{pad}auto {Cs_var} = {C_var}.slice({tx_var} * {TILE_N_var}, {ty_var} * {TILE_M_var});",
            f"{pad}int {n_base_var} = (int)({tx_var} * {TILE_N_var});",
            f"{pad}int {m_base_var} = (int)({ty_var} * {TILE_M_var});",
            f"{pad}for (auto {it_var} = {coop_var}.begin(); {it_var} != {coop_var}.end(); ++{it_var}) {{",
            f"{pad}    if ({coop_var}.is_valid_element({it_var})) {{",
            f"{pad}        auto {md_var} = {coop_var}.get_multidimensional_index({it_var});",
            f"{pad}        metal::array<int, 2> {idx_var} = {{(int){md_var}[0], (int){md_var}[1]}};",
            # Tail elements of partial tiles address outside the matrix;
            # operator[] performs no bounds checking, so test explicitly.
            f"{pad}        if ({idx_var}[0] + {n_base_var} < {N_var} && {idx_var}[1] + {m_base_var} < {M_var}) {{",
            f"{pad}            {Cs_var}[{idx_var}] = static_cast<{out_dtype}>(*{it_var});",
            f"{pad}        }}",
            f"{pad}    }}",
            f"{pad}}}",
        ]
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
        # Detect static_cast<dtype>(expr) parsed as Compare by Python.
        # The type can be a simple Name (e.g. float) or a Call like
        # decltype(a+b) from Inductor's MetalOverrides.
        if (
            isinstance(node.left, ast.Name)
            and node.left.id == "static_cast"
            and len(node.ops) == 2
            and isinstance(node.ops[0], ast.Lt)
            and isinstance(node.ops[1], ast.Gt)
        ):
            type_node = node.comparators[0]
            if isinstance(type_node, ast.Name):
                metal_type = type_node.id
            elif isinstance(type_node, ast.Call) and isinstance(
                type_node.func, ast.Name
            ):
                fn = type_node.func.id
                # decltype(expr) → decltype(expr)
                args_msl = [_ast_expr_to_msl(a, subs=subs) for a in type_node.args]
                metal_type = f"{fn}({', '.join(args_msl)})"
                if metal_type == "decltype((*_it))":
                    metal_type = "decltype(+(*_it))"
            else:
                metal_type = _ast_expr_to_msl(type_node, subs=subs)
            inner = _ast_expr_to_msl(node.comparators[1], subs=subs)
            return f"static_cast<{metal_type}>({inner})"

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
        # C++ namespace access: metal.precise.sin → metal::precise::sin
        # The :: was replaced with . before Python AST parsing; restore it here.
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
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "tl"
        and func.attr == "load"
    ):
        access = _ptr_access_expr(node.args[0], subs=subs)
        if len(node.args) >= 2:
            mask_node = node.args[1]
            if isinstance(mask_node, ast.Constant) and mask_node.value is None:
                return access
            mask = _ast_expr_to_msl(mask_node, subs=subs)
            other = None
            if len(node.args) >= 3:
                other = _ast_expr_to_msl(node.args[2], subs=subs)
            else:
                for kw in node.keywords:
                    if kw.arg == "other":
                        other = _ast_expr_to_msl(kw.value, subs=subs)
                        break
            if other is None:
                raise exc.BackendUnsupported(
                    "metal", "tl.load with mask requires 'other' argument"
                )
            return f"({mask} ? {access} : ({other}))"
        return access

    # Generic function call (main path for MetalOverrides expressions)
    func_msl = _ast_expr_to_msl(func, subs=subs)
    args_msl = [_ast_expr_to_msl(a, subs=subs) for a in node.args]
    return f"{func_msl}({', '.join(args_msl)})"


#: What Inductor's ``constant_repr`` emits for non-finite float constants.
_NON_FINITE_FLOATS = {
    "inf": "INFINITY",
    "-inf": "(-INFINITY)",
    "nan": "NAN",
}

_CPP_NAMESPACE_ROOTS = frozenset({"metal", "c10", REDUCTION_NAMESPACE})


def _is_cpp_namespace_root(node: ast.Attribute) -> bool:
    """Return True if *node* is an attribute chain rooted at a C++ namespace.

    C++ namespace syntax (``metal::precise::sin``) is converted to Python
    dot notation (``metal.precise.sin``) before AST parsing.  This detects
    those chains so the walker can emit ``::`` instead of ``.``.
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
