"""Preserve FP32 multiplication rounding in moved scalar operands.

Moving a producer to a cooperative copy loop or a separate kernel changes its
reuse and can enable ptxas to contract a previously separate multiply and add.
Before a narrowing cast, that small difference can change a low-precision ULP of the
matmul operand. In particular, ``a * p - a * p`` must remain zero when the
source multiplies were separately rounded.

The caller uses this only without fast math. Explicit FMA calls remain fused;
integer, low-precision, FP64, and unproven arithmetic remain unchanged.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Literal

from ..ast_extension import expr_from_string
from .scalar_recipe import _MATH_CALLS
from .scalar_recipe import _clone
from .scalar_recipe import _path

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

_Kind = Literal["fp32", "literal"] | None
_FLOAT_MATH_CALLS = _MATH_CALLS - {"isfinite", "isinf", "isnan"}

# A materialized pointwise producer has the same rounding contract as a
# rematerialized collective operand. Its FX dtype facts can prove products
# whose input is a direct FP32 load, without an explicit generated cast.
FP32_MULTIPLY_ROUNDING_META_KEY = "cute_preserve_fp32_multiply_rounding"
FP32_MULTIPLY_ROUNDING_EXPR = (
    "_cute_inline_asm_elementwise("
    "(cutlass.Float32({a}), cutlass.Float32({b})), "
    "asm='mul.rn.f32 $0, $1, $2;', constraints='=f,f,f', "
    "dtype=cutlass.Float32, is_pure=True)"
)


def is_rounded_fp32_multiply(value: ast.expr) -> bool:
    """Recognize only the existing, explicitly rounded scalar operation."""
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "_cute_inline_asm_elementwise"
        and len(value.args) == 1
        and isinstance(value.args[0], ast.Tuple)
        and len(value.args[0].elts) == 2
        and len(value.keywords) == 4
    ):
        return False
    return {keyword.arg: ast.unparse(keyword.value) for keyword in value.keywords} == {
        "asm": repr("mul.rn.f32 $0, $1, $2;"),
        "constraints": repr("=f,f,f"),
        "dtype": "cutlass.Float32",
        "is_pure": "True",
    }


def _combined_kind(kinds: Sequence[_Kind]) -> _Kind:
    if not kinds or any(kind is None for kind in kinds):
        return None
    return "fp32" if "fp32" in kinds else "literal"


def _kind(node: ast.expr, names: Mapping[str, _Kind]) -> _Kind:
    if isinstance(node, ast.Constant):
        # Python integers can become Int64 (including through constant
        # arithmetic), which can promote a CuTe Float32 expression to Float64.
        return "literal" if isinstance(node.value, float) else None
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _kind(node.operand, names)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        return _combined_kind([_kind(node.left, names), _kind(node.right, names)])
    if isinstance(node, ast.IfExp):
        left, right = _kind(node.body, names), _kind(node.orelse, names)
        return left if left == right else None
    if not isinstance(node, ast.Call):
        return None
    path = _path(node.func)
    if path == ("cutlass", "Float32"):
        return "fp32"
    if (
        isinstance(node.func, ast.Attribute)
        and node.func.attr in {"to", "bitcast"}
        and len(node.args) == 1
        and not node.keywords
        and _path(node.args[0]) == ("cutlass", "Float32")
    ):
        return "fp32"
    if path == ("_cute_inline_asm_elementwise",) and any(
        keyword.arg == "dtype" and _path(keyword.value) == ("cutlass", "Float32")
        for keyword in node.keywords
    ):
        return "fp32"
    if (
        path is not None
        and len(path) == 3
        and path[:2] == ("cute", "math")
        and path[2] in _FLOAT_MATH_CALLS
    ) or path in {
        ("operator", "add"),
        ("operator", "sub"),
        ("operator", "mul"),
        ("operator", "truediv"),
    }:
        kinds = [_kind(arg, names) for arg in node.args]
        # Calls on Python literals need not have CuTe FP32 return semantics.
        return "fp32" if _combined_kind(kinds) == "fp32" else None
    return None


def _rounded_multiply(left: ast.expr, right: ast.expr) -> ast.expr:
    result = expr_from_string(
        FP32_MULTIPLY_ROUNDING_EXPR,
        a=left,
        b=right,
    )
    assert isinstance(result, ast.expr)
    return result


class _PreserveRounding(ast.NodeTransformer):
    def __init__(self) -> None:
        self.names: dict[str, _Kind] = {}

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        rounded = isinstance(node.op, ast.Mult) and _kind(node, self.names) == "fp32"
        transformed = self.generic_visit(node)
        assert isinstance(transformed, ast.BinOp)
        return (
            _rounded_multiply(transformed.left, transformed.right)
            if rounded
            else transformed
        )

    def visit_Call(self, node: ast.Call) -> ast.expr:
        rounded = (
            _path(node.func) == ("operator", "mul")
            and len(node.args) == 2
            and not node.keywords
            and _kind(node, self.names) == "fp32"
        )
        transformed = self.generic_visit(node)
        assert isinstance(transformed, ast.Call)
        return (
            _rounded_multiply(transformed.args[0], transformed.args[1])
            if rounded
            else transformed
        )


def preserve_fp32_multiply_rounding(
    statements: Sequence[ast.Assign], value: ast.expr
) -> tuple[list[ast.Assign], ast.expr]:
    """Copy a sequential recipe, protecting only proven FP32 products.

    Reaching definitions are processed in order; unknown rebindings remove
    prior type facts. Boundary names deliberately start without a dtype proof.
    """
    rewrite = _PreserveRounding()
    result: list[ast.Assign] = []
    for statement in statements:
        assert len(statement.targets) == 1
        assert isinstance(statement.targets[0], ast.Name)
        kind = _kind(statement.value, rewrite.names)
        copied = _clone(statement)
        copied.value = rewrite.visit(copied.value)
        result.append(copied)
        rewrite.names[statement.targets[0].id] = kind
    return result, rewrite.visit(_clone(value))
