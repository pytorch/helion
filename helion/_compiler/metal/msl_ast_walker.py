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

from ... import exc

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
    ptr_expr = _ast_expr_to_msl(node.args[0])
    deref = f"*({ptr_expr})"
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
                f"{pad}    {deref} = {val_expr};",
                f"{pad}}}",
            )
        )
    else:
        parts.append(f"{pad}{deref} = {val_expr};")


def _emit_stmts(
    stmts: list[ast.stmt],
    parts: list[str],
    indent: int,
    declared: set[str],
) -> None:
    """Emit a list of AST statements as MSL lines into *parts*.

    *declared* tracks variable names that have already been declared in the
    current or an enclosing scope so that reassignments do not repeat ``auto``.
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
            val_msl = _ast_expr_to_msl(value)
            if target.id in declared:
                parts.append(f"{pad}{target.id} = {val_msl};")
            else:
                parts.append(f"{pad}auto {target.id} = {val_msl};")
                declared.add(target.id)
        elif isinstance(stmt, ast.Expr):
            call = stmt.value
            # tl.store(ptr + offset, val, mask) → if (mask) { *(ptr+offset) = val; }
            if _is_tl_store_call(call):
                assert isinstance(call, ast.Call)
                _emit_tl_store(call, parts, indent=indent)
            else:
                parts.append(f"{pad}{_ast_expr_to_msl(call)};")
        elif isinstance(stmt, ast.If):
            test_msl = _ast_expr_to_msl(stmt.test)
            parts.append(f"{pad}if ({test_msl}) {{")
            _emit_stmts(stmt.body, parts, indent=indent + 4, declared=declared)
            if stmt.orelse:
                parts.append(f"{pad}}} else {{")
                _emit_stmts(stmt.orelse, parts, indent=indent + 4, declared=declared)
            parts.append(f"{pad}}}")
        elif isinstance(stmt, ast.For):
            _emit_for(stmt, parts, indent=indent, declared=declared)
        else:
            raise exc.BackendUnsupported(
                "metal",
                f"AST statement type: {type(stmt).__name__}",
            )


def _emit_for(
    stmt: ast.For,
    parts: list[str],
    indent: int,
    declared: set[str],
) -> None:
    """Emit MSL for a ``for var in range(...)`` or ``for var in tl.range(...)`` loop.

    Both ``range`` and ``tl.range`` accept 1-3 positional args
    (end), (start, end), or (start, end, step) and produce a
    C-style ``for (int var = start; var < end; var += step)`` loop.
    """
    pad = " " * indent
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
        end = _ast_expr_to_msl(args[0])
    elif len(args) == 2:
        start = _ast_expr_to_msl(args[0])
        end = _ast_expr_to_msl(args[1])
    elif len(args) >= 3:
        start = _ast_expr_to_msl(args[0])
        end = _ast_expr_to_msl(args[1])
        step = _ast_expr_to_msl(args[2])

    parts.append(
        f"{pad}for (int {loop_var} = {start}; {loop_var} < {end}; {loop_var} += {step}) {{"
    )
    declared.add(loop_var)
    _emit_stmts(stmt.body, parts, indent=indent + 4, declared=declared)
    parts.append(f"{pad}}}")


# ---------------------------------------------------------------------------
# AST-to-MSL expression converter (module-level pure functions)
# ---------------------------------------------------------------------------


def _ast_expr_to_msl(node: ast.AST) -> str:
    """Recursively convert an AST expression node to MSL C++ string."""
    if isinstance(node, ast.Name):
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
        operand = _ast_expr_to_msl(node.operand)
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
        if isinstance(node.op, ast.Not):
            return f"(!{operand})"
        if isinstance(node.op, ast.Invert):
            return f"(~{operand})"
        raise exc.BackendUnsupported("metal", f"unary op: {type(node.op).__name__}")

    if isinstance(node, ast.BinOp):
        left = _ast_expr_to_msl(node.left)
        right = _ast_expr_to_msl(node.right)
        op = node.op
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
        parts = [_ast_expr_to_msl(v) for v in node.values]
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
                # decltype(expr) → decltype(expr)
                fn = type_node.func.id
                args_msl = [_ast_expr_to_msl(a) for a in type_node.args]
                metal_type = f"{fn}({', '.join(args_msl)})"
            else:
                metal_type = _ast_expr_to_msl(type_node)
            inner = _ast_expr_to_msl(node.comparators[1])
            return f"static_cast<{metal_type}>({inner})"

        left = _ast_expr_to_msl(node.left)
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
            parts.append(_ast_expr_to_msl(comp))
        return f"({' '.join(parts)})"

    if isinstance(node, ast.IfExp):
        test = _ast_expr_to_msl(node.test)
        body = _ast_expr_to_msl(node.body)
        orelse = _ast_expr_to_msl(node.orelse)
        return f"({test} ? {body} : {orelse})"

    if isinstance(node, ast.Call):
        return _ast_call_to_msl(node)

    if isinstance(node, ast.Subscript):
        return _ast_subscript_to_msl(node)

    if isinstance(node, ast.Attribute):
        value = _ast_expr_to_msl(node.value)
        # C++ namespace access: metal.precise.sin → metal::precise::sin
        # The :: was replaced with . before Python AST parsing; restore it here.
        sep = "::" if _is_cpp_namespace_root(node) else "."
        return f"{value}{sep}{node.attr}"

    raise exc.BackendUnsupported("metal", f"AST expression type: {type(node).__name__}")


def _ast_call_to_msl(node: ast.AST) -> str:
    """Convert an AST Call node to MSL.

    Handles:
    - ``tl.load`` — from PointerIndexingStrategy
    - Generic function call fallback — main handler for MetalOverrides expressions
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
        ptr_expr = _ast_expr_to_msl(node.args[0])
        deref = f"*({ptr_expr})"
        if len(node.args) >= 2:
            # Check if mask is None (no masking)
            mask_node = node.args[1]
            if isinstance(mask_node, ast.Constant) and mask_node.value is None:
                return deref
            mask = _ast_expr_to_msl(mask_node)
            other = None
            if len(node.args) >= 3:
                other = _ast_expr_to_msl(node.args[2])
            else:
                for kw in node.keywords:
                    if kw.arg == "other":
                        other = _ast_expr_to_msl(kw.value)
                        break
            if other is None:
                raise exc.BackendUnsupported(
                    "metal", "tl.load with mask requires 'other' argument"
                )
            return f"({mask} ? {deref} : ({other}))"
        return deref

    # Generic function call (main path for MetalOverrides expressions)
    func_msl = _ast_expr_to_msl(func)
    args_msl = [_ast_expr_to_msl(a) for a in node.args]
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


def _ast_subscript_to_msl(node: ast.AST) -> str:
    """Convert an AST Subscript node to MSL.

    Only simple index subscripts (e.g. ``tgid[0]``) are supported.
    """
    assert isinstance(node, ast.Subscript)
    buf_name = _ast_expr_to_msl(node.value)
    idx = _ast_expr_to_msl(node.slice)
    return f"{buf_name}[{idx}]"
