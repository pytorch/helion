"""Invert the exact signed-fold uint32-to-FP32 uniform comparison.

For ``0 <= m < 2**31``, the existing conversion is
``float32(float32(m) * float32.from_bits(0x2fffffff))``. For nonzero m this
is the predecessor of ``float32(m) * 2**-31``. Its inverse rounding-cell
endpoint therefore gives an exact integer comparison, including ties to
even. This changes only the comparison consumer; observable uniform values
and the arithmetic producing the integer word remain untouched.
"""

from __future__ import annotations

import ast
import struct
from typing import TYPE_CHECKING
from typing import cast

import torch

from ..ast_extension import expr_from_string
from ..ast_read_writes import ReadWrites
from ..aten_lowering import Lowering
from ..aten_lowering import LoweringContext

if TYPE_CHECKING:
    from ..device_function import DeviceFunction

_CONVERT = torch.ops.prims.convert_element_type.default
_ADD = (torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar)
_SUB = (torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar)
_AND = (torch.ops.aten.bitwise_and.Scalar, torch.ops.aten.bitwise_and.Tensor)
_MUL = (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar)
_GT = (torch.ops.aten.gt.Scalar, torch.ops.aten.gt.Tensor)
_LT = (torch.ops.aten.lt.Scalar, torch.ops.aten.lt.Tensor)


def _typed(node: object, dtype: torch.dtype) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    value = node.meta.get("val")
    return isinstance(value, torch.Tensor) and value.dtype is dtype


def _args(
    node: object, targets: tuple[object, ...], dtype: torch.dtype, count: int = 2
) -> tuple[object, ...] | None:
    if (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target in targets
        and len(node.args) == count
        and not node.kwargs
        and _typed(node, dtype)
    ):
        return node.args
    return None


def _constant_rhs(
    node: object, targets: tuple[object, ...], constant: int
) -> torch.fx.Node | None:
    args = _args(node, targets, torch.int64)
    if (
        args is not None
        and type(args[1]) is int
        and args[1] == constant
        and _typed(args[0], torch.int64)
    ):
        return cast("torch.fx.Node", args[0])
    return None


def match_uniform_float_gt(node: torch.fx.Node) -> torch.fx.Node | None:
    """Return the integer word for a proved FP32 uniform > scalar comparison."""
    comparison = _args(node, _GT, torch.bool)
    if comparison is None:
        return None
    uniform, threshold = comparison
    if not (
        type(threshold) is float
        or (
            isinstance(threshold, torch.fx.Node)
            and isinstance(threshold.meta.get("val"), torch.SymFloat)
        )
    ):
        return None
    product = _args(uniform, _MUL, torch.float32)
    if product is None:
        return None
    converted, scale = product
    # Only constants rounding to the proved FP32 scale are interchangeable.
    # The range check also avoids a Python float-packing overflow.
    if not (
        type(scale) is float
        and 0.0 < scale < 1.0
        and struct.pack("<f", scale) == b"\xff\xff\xff\x2f"
    ):
        return None
    conversion = _args(converted, (_CONVERT,), torch.float32)
    if conversion is None or conversion[1] is not torch.float32:
        return None
    magnitude = _args(conversion[0], (torch.ops.aten.where.self,), torch.int64, 3)
    if magnitude is None:
        return None
    condition, negative, signed = magnitude
    sign_test = _args(condition, _LT, torch.bool)
    if not (
        sign_test is not None
        and sign_test[0] is signed
        and type(sign_test[1]) is int
        and sign_test[1] == 0
    ):
        return None
    negation = _constant_rhs(negative, _SUB, 1)
    if _args(negation, (torch.ops.aten.neg.default,), torch.int64, 1) != (signed,):
        return None
    masked = _constant_rhs(signed, _SUB, 1 << 31)
    shifted = _constant_rhs(masked, _AND, (1 << 32) - 1)
    return _constant_rhs(shifted, _ADD, 1 << 31)


class UniformComparisonLowering(Lowering):
    """Preserve a typed proof until complete device argument scopes are known."""

    def codegen(self, ctx: LoweringContext, node: torch.fx.Node) -> ast.AST:
        left, right = (ctx.to_ast(arg) for arg in node.args)
        word = match_uniform_float_gt(node)
        function = ctx.cg.device_function
        # Reduction-combine helper bodies use a separate codegen interface
        # and do not pass through DeviceFunction's completed-body rewrite.
        # Collective operand rematerialization cannot yet follow the tuple
        # results of a hoisted cutoff helper. Keep its original scalar recipe
        # until those results have explicit operand-boundary provenance.
        if (
            word is None
            or ctx.cg is not function.codegen
            or function.config.get("cute_collective_mma", False)
        ):
            return expr_from_string(
                "operator.gt({left}, {right})", left=left, right=right
            )
        state = function.cute_state
        if state.uniform_comparison_marker is None:
            state.uniform_comparison_marker = function.new_var("_helion_uniform_gt")
        # GraphInterpreter retains intermediate values. The extra word read
        # also preserves its AST liveness until the marker is resolved.
        return expr_from_string(
            f"{state.uniform_comparison_marker}({{word}}, {{left}}, {{right}})",
            word=ctx.to_ast(word),
            left=left,
            right=right,
        )


# This definition is emitted into the generated module, so its exact source
# participates in the normal compilation/cache key. Branch-selected shifts
# have counts in [0, 23]. PTX's unsigned shifts clamp oversize inactive counts,
# even when the backend hoists them above the structured selections.
_CUTOFF_TEMPLATE = """
@cute.jit
def {name}(probability):
    bits = cutlass.Float32(probability).bitcast(cutlass.Uint32)
    absolute = bits & cutlass.Uint32(0x7fffffff)
    cutoff = cutlass.Int32(0x7fffffff)
    if absolute > cutlass.Uint32(0x7f800000):
        pass
    elif (bits & cutlass.Uint32(0x80000000)) != cutlass.Uint32(0) and absolute != cutlass.Uint32(0):
        cutoff = cutlass.Int32(-1)
    elif absolute < cutlass.Uint32(0x2fffffff):
        cutoff = cutlass.Int32(0)
    elif absolute < cutlass.Uint32(0x3f7fffff):
        successor = absolute + cutlass.Uint32(1)
        exponent = cutlass.Int32(successor >> cutlass.Uint32(23)) - cutlass.Int32(96)
        significand = (successor & cutlass.Uint32(0x7fffff)) | cutlass.Uint32(0x800000)
        if exponent <= cutlass.Int32(23):
            cutoff = cutlass.Int32(significand >> cutlass.Uint32(cutlass.Int32(23) - exponent))
        else:
            cutoff = cutlass.Int32((significand << cutlass.Uint32(exponent - cutlass.Int32(23))) + (cutlass.Uint32(1) << cutlass.Uint32(exponent - cutlass.Int32(24))) - (significand & cutlass.Uint32(1)))
    first = cutlass.Uint32(cutlass.Uint32(cutoff) + cutlass.Uint32(1))
    span = cutlass.Uint32(cutlass.Uint32(0) - cutlass.Uint32(cutlass.Uint32(2) * first))
    all_keep = cutoff < cutlass.Int32(0)
    return first, span, all_keep
"""


def cutoff_function(name: str) -> ast.FunctionDef:
    return cast(
        "ast.FunctionDef", ast.parse(_CUTOFF_TEMPLATE.format(name=name)).body[0]
    )


def lower_uniform_comparisons(
    body: list[ast.stmt],
    function: DeviceFunction,
    *,
    float_scalar_names: set[str],
) -> list[ast.stmt]:
    """Hoist exact cutoffs for immutable FP32 arguments before all lane loops.

    Unproved scopes retain the original comparison. In particular, a scalar
    argument assigned inside the device function is not loop invariant.
    """
    marker = function.cute_state.uniform_comparison_marker
    if marker is None:
        return body
    renames = {key: values[0] for key, values in function._variable_renames.items()}

    def canonical(name: str) -> str:
        return renames.get(name, name)

    module = ast.Module(body=body, type_ignores=[])
    writes = set(ReadWrites.from_ast(module).writes)
    # ReadWrites intentionally skips induction targets; those writes also
    # invalidate a hoisted argument, including a renamed loop-carried alias.
    writes.update(
        node.id
        for node in ast.walk(module)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    immutable = {canonical(name) for name in float_scalar_names} - {
        canonical(name) for name in writes
    }
    preamble: list[ast.stmt] = []
    cutoffs: dict[str, tuple[str, str, str]] = {}
    helper: str | None = None

    class Rewrite(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            nonlocal helper
            self.generic_visit(node)
            if not isinstance(node.func, ast.Name) or node.func.id != marker:
                return node
            assert len(node.args) == 3 and not node.keywords
            word, uniform, probability = node.args
            if not (
                isinstance(probability, ast.Name)
                and canonical(probability.id) in immutable
                or isinstance(probability, ast.Constant)
                and type(probability.value) is float
            ):
                return expr_from_string(
                    "operator.gt({uniform}, {probability})",
                    uniform=uniform,
                    probability=probability,
                )
            key = ast.dump(probability)
            if key not in cutoffs:
                if helper is None:
                    helper = function.new_var(f"{function.name}_uniform_cutoff")
                    function.codegen.module_statements.append(cutoff_function(helper))
                names = tuple(
                    function.new_var(f"_helion_uniform_{suffix}")
                    for suffix in ("first", "span", "all")
                )
                cutoffs[key] = cast("tuple[str, str, str]", names)
                preamble.extend(
                    ast.parse(
                        f"{', '.join(names)} = {helper}({ast.unparse(probability)})"
                    ).body
                )
            first, span, all_keep = cutoffs[key]
            return expr_from_string(
                f"{all_keep} or (cutlass.Uint32(cutlass.Uint32({{word}}) - {first}) < {span})",
                word=word,
            )

    Rewrite().visit(module)
    return [*preamble, *module.body]
