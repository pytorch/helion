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
import dataclasses

from ... import exc


@dataclasses.dataclass
class EmitState:
    """Mutable state passed through ``_emit_stmts`` calls.

    Tracks declared variable names (to avoid duplicate ``auto`` declarations)
    and MPP matmul setup parameters (keyed by setup variable name).

    MPPGraph lowering emits explicit ``_coop_iter`` loops. The walker derives
    the MMA-result substitution from the setup marker's ``fx_name``.
    """

    declared: set[str] = dataclasses.field(default_factory=set)
    mpp_setups: dict[str, dict[str, object]] = dataclasses.field(default_factory=dict)


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
) -> None:
    """Emit MSL for ``tl.store(ptr + offset, val, mask)``.

    Dereferences the pointer expression: ``*(ptr + offset) = val``.
    All three positional args (ptr, val, mask) are always present —
    PointerIndexingStrategy emits them positionally.
    """
    pad = " " * indent
    access = _ptr_access_expr(node.args[0])
    val_expr = _ast_expr_to_msl(node.args[1])
    mask = None
    if len(node.args) >= 3:
        mask_node = node.args[2]
        # None mask = no masking
        if not (isinstance(mask_node, ast.Constant) and mask_node.value is None):
            mask = _ast_expr_to_msl(mask_node)
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

    MPPGraph lowering emits explicit markers and cooperative epilogue loops
    before this point; this function does pure text emission.
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
            # Check for _metal_mpp_setup assignment
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "_metal_mpp_setup"
            ):
                params = _extract_mpp_setup_params(value)
                state.mpp_setups[target.id] = params
                _emit_mpp_setup(target.id, params, parts, indent=indent)
                state.declared.add(target.id)
                continue
            val_msl = _ast_expr_to_msl(value, subs=subs)
            if target.id in state.declared:
                parts.append(f"{pad}{target.id} = {val_msl};")
            else:
                parts.append(f"{pad}auto {target.id} = {val_msl};")
                state.declared.add(target.id)
        elif isinstance(stmt, ast.Expr):
            call = stmt.value
            # tl.store(ptr + offset, val, mask) → if (mask) { *(ptr+offset) = val; }
            if _is_tl_store_call(call):
                assert isinstance(call, ast.Call)
                _emit_tl_store(call, parts, indent=indent)
            elif _is_call_to(call, "_metal_mpp_k_step"):
                assert isinstance(call, ast.Call)
                setup_arg = call.args[0]
                assert isinstance(setup_arg, ast.Name)
                k_offset_expr = _ast_expr_to_msl(call.args[1])
                _emit_mpp_k_step(setup_arg.id, k_offset_expr, parts, indent=indent)
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
            _emit_stmts(stmt.body, parts, indent=indent + 4, state=state, subs=subs)
            if stmt.orelse:
                parts.append(f"{pad}}} else {{")
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
        else:
            raise exc.BackendUnsupported(
                "metal",
                f"AST statement type: {type(stmt).__name__}",
            )


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

    # --- MPP cooperative iterator (synthesized by the pass) ---
    func = it.func
    if isinstance(func, ast.Name) and func.id == "_coop_iter":
        assert len(it.args) == 1 and isinstance(it.args[0], ast.Name), (
            "_coop_iter takes exactly one setup-var argument"
        )
        setup_name = it.args[0].id
        setup = state.mpp_setups[setup_name]
        fx_name = setup.get("fx_name")
        coop_subs = {str(fx_name): "(*_it)"} if fx_name else {}
        coop = _mpp_symbol(setup_name, "_coop")
        parts.extend(
            (
                f"{pad}// Epilogue: element-wise ops on cooperative_tensor",
                f"{pad}for (auto _it = {coop}.begin(); _it != {coop}.end(); _it++) {{",
            )
        )
        _emit_stmts(stmt.body, parts, indent=indent + 4, state=state, subs=coop_subs)
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
    state.declared.add(loop_var)
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


def _mpp_symbol(setup_name: str, suffix: str) -> str:
    """Return a setup-scoped MPP MSL identifier."""
    return f"{setup_name}{suffix}"


def _ast_str_value(node: ast.AST) -> str:
    """Extract string constant from AST node."""
    assert isinstance(node, ast.Constant) and isinstance(node.value, str)
    return node.value


def _ast_int_value(node: ast.AST) -> int:
    """Extract integer constant from AST node."""
    assert isinstance(node, ast.Constant) and isinstance(node.value, int)
    return node.value


def _extract_mpp_setup_params(node: ast.Call) -> dict[str, object]:
    """Extract parameters from ``_metal_mpp_setup(...)`` call.

    Positional args (in order):
      0: lhs tensor name,  1: rhs tensor name,
      2-4: M, N, K,        5-7: TILE_M, TILE_N, TILE_K,
      8: NUM_SG,           9: in_dtype,       10: acc_dtype,
      11: bias tensor name (or ""),  12: bias metal dtype (or ""),
      13: FX node name of the MMA op (or "").

    Output tensor name and dtype are *not* in the setup. MPPGraph lowering
    passes them through the explicit
    ``_metal_mpp_coop_store(setup, out_name, out_dtype)`` marker.
    """
    args = node.args
    assert len(args) == 14, (
        f"_metal_mpp_setup expects 14 positional args, got {len(args)}"
    )
    return {
        "lhs": _ast_str_value(args[0]),
        "rhs": _ast_str_value(args[1]),
        "M": _ast_int_value(args[2]),
        "N": _ast_int_value(args[3]),
        "K": _ast_int_value(args[4]),
        "TILE_M": _ast_int_value(args[5]),
        "TILE_N": _ast_int_value(args[6]),
        "TILE_K": _ast_int_value(args[7]),
        "NUM_SG": _ast_int_value(args[8]),
        "in_dtype": _ast_str_value(args[9]),
        "acc_dtype": _ast_str_value(args[10]),
        "bias": _ast_str_value(args[11]) or None,
        "bias_dtype": _ast_str_value(args[12]) or None,
        "fx_name": _ast_str_value(args[13]) or None,
    }


def _emit_mpp_setup(
    setup_name: str,
    params: dict[str, object],
    parts: list[str],
    indent: int,
) -> None:
    """Emit MPP matmul2d setup MSL.

    Declares the input tensor handles (``_A`` / ``_B``), the
    ``matmul2d_descriptor`` and operator, the threadgroup-id → tile
    decomposition, the operand slices, and the cooperative_tensor
    accumulator.  The output tensor handle (``_C``) is declared by
    :func:`_emit_mpp_coop_store` because its name and dtype are sourced
    from the trailing ``tl.store(out_ptr, ...)`` rather than the setup.
    """
    pad = " " * indent
    lhs = params["lhs"]
    rhs = params["rhs"]
    bias = params.get("bias")
    M = params["M"]
    N = params["N"]
    K = params["K"]
    TILE_M = params["TILE_M"]
    TILE_N = params["TILE_N"]
    TILE_K = params["TILE_K"]
    NUM_SG = params["NUM_SG"]
    in_dtype = params["in_dtype"]
    acc_dtype = params["acc_dtype"]
    M_var = _mpp_symbol(setup_name, "_M")
    N_var = _mpp_symbol(setup_name, "_N")
    K_var = _mpp_symbol(setup_name, "_K")
    TILE_M_var = _mpp_symbol(setup_name, "_TILE_M")
    TILE_N_var = _mpp_symbol(setup_name, "_TILE_N")
    TILE_K_var = _mpp_symbol(setup_name, "_TILE_K")
    NUM_SG_var = _mpp_symbol(setup_name, "_NUM_SG")
    A_var = _mpp_symbol(setup_name, "_A")
    B_var = _mpp_symbol(setup_name, "_B")
    D_var = _mpp_symbol(setup_name, "_D")
    Ds_var = _mpp_symbol(setup_name, "_Ds")
    desc_var = _mpp_symbol(setup_name, "_desc")
    op_var = _mpp_symbol(setup_name, "_op")
    gm_var = _mpp_symbol(setup_name, "_gm")
    flat_id_var = _mpp_symbol(setup_name, "_flat_id")
    ty_var = _mpp_symbol(setup_name, "_ty")
    tx_var = _mpp_symbol(setup_name, "_tx")
    As_var = _mpp_symbol(setup_name, "_As")
    Bs_var = _mpp_symbol(setup_name, "_Bs")
    coop_var = _mpp_symbol(setup_name, "_coop")

    needs_k_loop = int(TILE_K) < int(K)  # type: ignore[arg-type]
    # When bias is provided, always use multiply_accumulate so
    # _op.run accumulates onto the pre-loaded bias values.
    mm_mode = "multiply_accumulate" if needs_k_loop or bias else "multiply"

    parts.extend(
        [
            f"{pad}// MPP matmul2d setup",
            f"{pad}constexpr int {M_var} = {M};",
            f"{pad}constexpr int {N_var} = {N};",
            f"{pad}constexpr int {K_var} = {K};",
            f"{pad}constexpr int {TILE_M_var} = {TILE_M};",
            f"{pad}constexpr int {TILE_N_var} = {TILE_N};",
            f"{pad}constexpr int {TILE_K_var} = {TILE_K};",
            f"{pad}constexpr int {NUM_SG_var} = {NUM_SG};",
            "",
            f"{pad}auto {A_var} = tensor<device {in_dtype}, dextents<int32_t, 2>, tensor_inline>(",
            f"{pad}    {lhs}, dextents<int32_t, 2>({K_var}, {M_var}));",
            f"{pad}auto {B_var} = tensor<device {in_dtype}, dextents<int32_t, 2>, tensor_inline>(",
            f"{pad}    {rhs}, dextents<int32_t, 2>({N_var}, {K_var}));",
            "",
            f"{pad}constexpr auto {desc_var} = matmul2d_descriptor(",
            f"{pad}    {TILE_M_var}, {TILE_N_var}, {TILE_K_var},",
            f"{pad}    false, false, false, matmul2d_descriptor::mode::{mm_mode});",
            f"{pad}matmul2d<{desc_var}, execution_simdgroups<{NUM_SG_var}>> {op_var};",
            "",
            f"{pad}// Decompose flat threadgroup ID into 2D tile indices",
            f"{pad}uint {gm_var} = ({M_var} + {TILE_M_var} - 1) / {TILE_M_var};",
            f"{pad}uint {flat_id_var} = tgid[0];",
            f"{pad}uint {ty_var} = {flat_id_var} % {gm_var};",
            f"{pad}uint {tx_var} = {flat_id_var} / {gm_var};",
            "",
            f"{pad}auto {As_var} = {A_var}.slice(0, {ty_var} * {TILE_M_var});",
            f"{pad}auto {Bs_var} = {B_var}.slice({tx_var} * {TILE_N_var}, 0);",
            "",
            f"{pad}auto {coop_var} = {op_var}.get_destination_cooperative_tensor<",
            f"{pad}    decltype({As_var}), decltype({Bs_var}), {acc_dtype}>();",
        ]
    )
    if bias:
        # For addmm (C = A*B + bias): load the bias tensor into the
        # cooperative_tensor BEFORE the K-loop.  Since _op.run uses
        # multiply_accumulate mode (C += A*B), the bias values serve as
        # the initial accumulator and the final result is bias + A*B.
        bias_dtype = params["bias_dtype"]
        assert bias_dtype, "bias_dtype must be populated when bias is provided"
        parts.extend(
            [
                f"{pad}auto {D_var} = tensor<device {bias_dtype}, dextents<int32_t, 2>, tensor_inline>(",
                f"{pad}    {bias}, dextents<int32_t, 2>({N_var}, {M_var}));",
                f"{pad}auto {Ds_var} = {D_var}.slice({tx_var} * {TILE_N_var}, {ty_var} * {TILE_M_var});",
                f"{pad}{coop_var}.load({Ds_var});",
            ]
        )
    elif needs_k_loop:
        # multiply_accumulate mode (used when needs_k_loop) accumulates
        # onto _coop's existing values, so zero-init is required.
        parts.extend(
            [
                f"{pad}for (auto _it = {coop_var}.begin(); _it != {coop_var}.end(); _it++)",
                f"{pad}    *_it = ({acc_dtype})(0);",
            ]
        )


def _emit_mpp_k_step(
    setup_name: str,
    k_offset_expr: str,
    parts: list[str],
    indent: int,
) -> None:
    """Emit MPP K-tile step MSL."""
    pad = " " * indent
    A_var = _mpp_symbol(setup_name, "_A")
    B_var = _mpp_symbol(setup_name, "_B")
    Ak_var = _mpp_symbol(setup_name, "_Ak")
    Bk_var = _mpp_symbol(setup_name, "_Bk")
    TILE_M_var = _mpp_symbol(setup_name, "_TILE_M")
    TILE_N_var = _mpp_symbol(setup_name, "_TILE_N")
    ty_var = _mpp_symbol(setup_name, "_ty")
    tx_var = _mpp_symbol(setup_name, "_tx")
    op_var = _mpp_symbol(setup_name, "_op")
    coop_var = _mpp_symbol(setup_name, "_coop")
    parts.extend(
        [
            f"{pad}auto {Ak_var} = {A_var}.slice({k_offset_expr}, {ty_var} * {TILE_M_var});",
            f"{pad}auto {Bk_var} = {B_var}.slice({tx_var} * {TILE_N_var}, {k_offset_expr});",
            f"{pad}{op_var}.run({Ak_var}, {Bk_var}, {coop_var});",
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

    The accumulator dtype is set in :func:`_emit_mpp_setup`; MPP handles the
    cooperative_tensor-to-output conversion during ``store`` for supported
    dtype combinations.
    """
    pad = " " * indent
    C_var = _mpp_symbol(setup_name, "_C")
    Cs_var = _mpp_symbol(setup_name, "_Cs")
    M_var = _mpp_symbol(setup_name, "_M")
    N_var = _mpp_symbol(setup_name, "_N")
    TILE_M_var = _mpp_symbol(setup_name, "_TILE_M")
    TILE_N_var = _mpp_symbol(setup_name, "_TILE_N")
    ty_var = _mpp_symbol(setup_name, "_ty")
    tx_var = _mpp_symbol(setup_name, "_tx")
    coop_var = _mpp_symbol(setup_name, "_coop")
    parts.extend(
        [
            f"{pad}auto {C_var} = tensor<device {out_dtype}, dextents<int32_t, 2>, tensor_inline>(",
            f"{pad}    {out_name}, dextents<int32_t, 2>({N_var}, {M_var}));",
            f"{pad}auto {Cs_var} = {C_var}.slice({tx_var} * {TILE_N_var}, {ty_var} * {TILE_M_var});",
            f"{pad}{coop_var}.store({Cs_var});",
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


_CPP_NAMESPACE_ROOTS = frozenset({"metal", "c10"})


def _is_cpp_namespace_root(node: ast.Attribute) -> bool:
    """Return True if *node* is an attribute chain rooted at a C++ namespace.

    C++ namespace syntax (``metal::precise::sin``) is converted to Python
    dot notation (``metal.precise.sin``) before AST parsing.  This detects
    those chains so the walker can emit ``::`` instead of ``.``.
    """
    while isinstance(node.value, ast.Attribute):
        node = node.value
    return isinstance(node.value, ast.Name) and node.value.id in _CPP_NAMESPACE_ROOTS


def _ast_subscript_to_msl(node: ast.AST, subs: dict[str, str] | None = None) -> str:
    """Convert an AST Subscript node to MSL.

    Only simple index subscripts (e.g. ``tgid[0]``) are supported.
    """
    assert isinstance(node, ast.Subscript)
    buf_name = _ast_expr_to_msl(node.value, subs=subs)
    idx = _ast_expr_to_msl(node.slice, subs=subs)
    return f"{buf_name}[{idx}]"
