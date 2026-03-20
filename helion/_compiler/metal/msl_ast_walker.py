"""MSL AST walker and AST-to-MSL C++ converter.

Converts the Python AST body (produced by the standard Helion codegen pipeline
with TritonOverrides) into MSL C++ source code.

The walker handles Triton-flavored AST patterns:
  - ``tl.cast(x, tl.float32)`` → ``static_cast<float>(x)``
  - ``tl.where(m, t, f)`` → ``select(f, t, m)``
  - ``tl.full([], val, dtype)`` → ``((metal_type)(val))``
  - ``tl.reshape(x, shape)`` → ``x``  (no-op for per-thread dispatch)
  - ``libdevice.func(x)`` / ``tl_math.func(x)`` → ``func(x)``
  - ``x.to(dtype)`` → ``x``  (strip dtype casts)
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ... import exc

if TYPE_CHECKING:
    from ..device_function import DeviceFunction


# Mapping from Triton tl.* dtype attribute names to Metal type strings
_TL_DTYPE_TO_METAL: dict[str, str] = {
    "float16": "half",
    "bfloat16": "bfloat",
    "float32": "float",
    "float64": "double",
    "int8": "char",
    "int16": "short",
    "int32": "int",
    "int64": "long",
    "uint8": "uchar",
}


class MslAstWalker:
    """AST-walking MSL emitter.

    Walks the body AST statement-by-statement, emitting MSL from composable
    building blocks.  Each kernel type adds a ``_generate_*`` method; the
    ``generate()`` entry point dispatches based on kernel classification.

    Currently supports elementwise kernels only.  Reduction and matmul
    generators can be added as new ``_generate_*`` methods.
    """

    def __init__(
        self,
        device_fn: DeviceFunction,
        body_stmts: list[ast.stmt],
    ) -> None:
        from ..device_function import TensorArg

        self.device_fn = device_fn
        self.body_stmts = body_stmts

        self.tensor_args = [
            a for a in device_fn.sorted_args() if isinstance(a, TensorArg)
        ]
        self.arg_map: dict[str, object] = {a.name: a for a in self.tensor_args}

    def generate(self) -> str:
        """Generate complete MSL source for the kernel."""
        return self._generate_elementwise()

    def _generate_elementwise(self) -> str:
        """Generate a simple 1D per-thread elementwise kernel from body AST."""
        from ..device_function import SymbolArgument
        from ..device_function import TensorArg

        msl_parts: list[str] = [
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
        ]

        params: list[str] = []
        scalar_preamble: list[str] = []
        buf_idx = 0
        for arg in self.device_fn.sorted_args():
            if isinstance(arg, TensorArg):
                from ..backend import MetalBackend

                dtype = arg.fake_value.dtype
                if dtype not in MetalBackend._DTYPE_TO_METAL:
                    raise exc.BackendUnsupported("metal", f"tensor dtype {dtype}")
                metal_dtype = MetalBackend._DTYPE_TO_METAL[dtype]
                params.append(f"device {metal_dtype}* {arg.name} [[buffer({buf_idx})]]")
                buf_idx += 1
            elif isinstance(arg, SymbolArgument):
                buf_param = f"_buf_{arg.name}"
                params.append(f"device const float* {buf_param} [[buffer({buf_idx})]]")
                buf_idx += 1
                scalar_preamble.append(f"    float {arg.name} = {buf_param}[0];")
        params.append("uint _gid [[thread_position_in_grid]]")

        sig = ", ".join(params)
        msl_parts.append(f"kernel void {self.device_fn.name}({sig}) {{")
        msl_parts.extend(scalar_preamble)

        for stmt in self.body_stmts:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                value = stmt.value
                if isinstance(target, ast.Name):
                    val_msl = _ast_expr_to_msl(value)
                    msl_parts.append(f"    auto {target.id} = {val_msl};")
                elif isinstance(target, ast.Subscript):
                    target_msl = _ast_expr_to_msl(target)
                    val_msl = _ast_expr_to_msl(value)
                    msl_parts.append(f"    {target_msl} = {val_msl};")
                else:
                    msl_parts.append(
                        f"    {_ast_expr_to_msl(target)} = {_ast_expr_to_msl(value)};"
                    )
            elif isinstance(stmt, ast.Expr):
                msl_parts.append(f"    {_ast_expr_to_msl(stmt.value)};")
            elif isinstance(stmt, ast.If):
                test_msl = _ast_expr_to_msl(stmt.test)
                msl_parts.append(f"    if ({test_msl}) {{")
                for body_stmt in stmt.body:
                    if (
                        isinstance(body_stmt, ast.Assign)
                        and len(body_stmt.targets) == 1
                    ):
                        target = body_stmt.targets[0]
                        val_msl = _ast_expr_to_msl(body_stmt.value)
                        if isinstance(target, ast.Subscript):
                            target_msl = _ast_expr_to_msl(target)
                            msl_parts.append(f"        {target_msl} = {val_msl};")
                        elif isinstance(target, ast.Name):
                            msl_parts.append(f"        auto {target.id} = {val_msl};")
                        else:
                            msl_parts.append(
                                f"        {_ast_expr_to_msl(target)} = {val_msl};"
                            )
                    elif isinstance(body_stmt, ast.Expr):
                        msl_parts.append(
                            f"        {_ast_expr_to_msl(body_stmt.value)};"
                        )
                    else:
                        raise exc.BackendUnsupported(
                            "metal",
                            f"AST statement type inside if: {type(body_stmt).__name__}",
                        )
                msl_parts.append("    }")
            else:
                raise exc.BackendUnsupported(
                    "metal",
                    f"AST statement type: {type(stmt).__name__}",
                )

        msl_parts.append("}")
        return "\n".join(msl_parts)


# ---------------------------------------------------------------------------
# AST-to-MSL expression converter (module-level pure functions)
# ---------------------------------------------------------------------------


def _ast_expr_to_msl(node: ast.AST) -> str:
    """Recursively convert an AST expression node to MSL C++ string."""
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Constant):
        if isinstance(node.value, float):
            return repr(node.value)
        return str(node.value)

    if isinstance(node, ast.UnaryOp):
        operand = _ast_expr_to_msl(node.operand)
        if isinstance(node.op, ast.USub):
            return f"(-{operand})"
        if isinstance(node.op, ast.Not):
            return f"(!{operand})"
        if isinstance(node.op, ast.Invert):
            return f"(~{operand})"
        return operand

    if isinstance(node, ast.BinOp):
        left = _ast_expr_to_msl(node.left)
        right = _ast_expr_to_msl(node.right)
        if isinstance(node.op, ast.FloorDiv):
            return f"({left} / {right})"
        if isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and node.right.value == 0.5:
                return f"sqrt({left})"
            return f"pow({left}, {right})"
        op = node.op
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
            op_str = "+"
        return f"({left} {op_str} {right})"

    if isinstance(node, ast.BoolOp):
        op_str = " && " if isinstance(node.op, ast.And) else " || "
        parts = [_ast_expr_to_msl(v) for v in node.values]
        return f"({op_str.join(parts)})"

    if isinstance(node, ast.Compare):
        # Detect static_cast<dtype>(expr) parsed as Compare by Python
        if (
            isinstance(node.left, ast.Name)
            and node.left.id == "static_cast"
            and len(node.ops) == 2
            and isinstance(node.ops[0], ast.Lt)
            and isinstance(node.ops[1], ast.Gt)
            and isinstance(node.comparators[0], ast.Name)
        ):
            metal_type = node.comparators[0].id
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
                parts.append("==")
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
        return f"{value}.{node.attr}"

    if isinstance(node, ast.Tuple):
        parts = [_ast_expr_to_msl(e) for e in node.elts]
        return ", ".join(parts)

    return ast.unparse(node)


def _ast_call_to_msl(node: ast.AST) -> str:
    """Convert an AST Call node to MSL."""
    assert isinstance(node, ast.Call)
    func = node.func

    # libdevice.func(x) / tl_math.func(x) → func(x)
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id in ("libdevice", "tl_math")
    ):
        args_msl = [_ast_expr_to_msl(a) for a in node.args]
        if func.attr == "exp" and len(args_msl) == 1:
            return f"exp2({args_msl[0]} * 1.4426950408889634f)"
        return f"{func.attr}({', '.join(args_msl)})"

    # triton_helpers.maximum/minimum(a, b) → max/min(a, b)
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "triton_helpers"
        and func.attr in ("maximum", "minimum")
    ):
        args_msl = [_ast_expr_to_msl(a) for a in node.args]
        msl_fn = "max" if func.attr == "maximum" else "min"
        return f"{msl_fn}({', '.join(args_msl)})"

    # tl.func(x) → MSL equivalent (sqrt_rn→sqrt, sigmoid→inline, etc.)
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "tl"
        and func.attr not in ("cast", "where", "reshape", "full")
    ):
        args_msl = [_ast_expr_to_msl(a) for a in node.args]
        attr = func.attr
        # Strip Triton rounding-mode suffixes (sqrt_rn → sqrt, div_rn → div)
        attr = attr.removesuffix("_rn")
        if attr == "sigmoid" and len(args_msl) == 1:
            return f"(1.0f / (1.0f + exp(-({args_msl[0]}))))"
        return f"{attr}({', '.join(args_msl)})"

    # tl.cast(x, tl.float32) → static_cast<float>(x)
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "tl"
        and func.attr == "cast"
        and len(node.args) == 2
    ):
        x_msl = _ast_expr_to_msl(node.args[0])
        dtype_node = node.args[1]
        if isinstance(dtype_node, ast.Attribute) and isinstance(
            dtype_node.value, ast.Name
        ):
            tl_dtype = dtype_node.attr
            metal_type = _TL_DTYPE_TO_METAL.get(tl_dtype, "float")
        else:
            metal_type = "float"
        return f"static_cast<{metal_type}>({x_msl})"

    # tl.where(mask, true, false) → select(false, true, mask)
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "tl"
        and func.attr == "where"
        and len(node.args) == 3
    ):
        mask = _ast_expr_to_msl(node.args[0])
        true_val = _ast_expr_to_msl(node.args[1])
        false_val = _ast_expr_to_msl(node.args[2])
        return f"select({false_val}, {true_val}, {mask})"

    # tl.reshape(x, shape) → x
    # Safe for 1D per-thread elementwise dispatch where every value is
    # scalar.  Will need a real implementation for multi-element tiles.
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "tl"
        and func.attr == "reshape"
        and len(node.args) >= 1
    ):
        return _ast_expr_to_msl(node.args[0])

    # tl.full([], val, dtype) → ((metal_type)(val))
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "tl"
        and func.attr == "full"
        and len(node.args) >= 2
    ):
        shape_arg = node.args[0]
        if isinstance(shape_arg, ast.List) and len(shape_arg.elts) == 0:
            val_msl = _ast_expr_to_msl(node.args[1])
            if len(node.args) >= 3:
                dtype_node = node.args[2]
                if isinstance(dtype_node, ast.Attribute) and isinstance(
                    dtype_node.value, ast.Name
                ):
                    metal_type = _TL_DTYPE_TO_METAL.get(dtype_node.attr, "float")
                else:
                    metal_type = "float"
            else:
                metal_type = "float"
            return f"(({metal_type})({val_msl}))"

    # select(f, t, m) → select(f, t, m)
    if isinstance(func, ast.Name) and func.id == "select":
        args_msl = [_ast_expr_to_msl(a) for a in node.args]
        return f"select({', '.join(args_msl)})"

    # x.to(dtype) → x
    if isinstance(func, ast.Attribute) and func.attr == "to":
        return _ast_expr_to_msl(func.value)

    # Generic function call
    func_msl = _ast_expr_to_msl(func)
    args_msl = [_ast_expr_to_msl(a) for a in node.args]
    return f"{func_msl}({', '.join(args_msl)})"


def _ast_subscript_to_msl(node: ast.AST) -> str:
    """Convert an AST Subscript node to MSL."""
    assert isinstance(node, ast.Subscript)
    buf_name = _ast_expr_to_msl(node.value)
    sl = node.slice

    if isinstance(sl, ast.Tuple):
        parts = [_ast_expr_to_msl(e) for e in sl.elts]
        return f"{buf_name}[{', '.join(parts)}]"

    if isinstance(sl, ast.Slice):
        return f"{buf_name}[:]"

    idx = _ast_expr_to_msl(sl)
    return f"{buf_name}[{idx}]"
