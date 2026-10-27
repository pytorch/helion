"""Preserve FP32 multiplication rounding in moved scalar operands.

Moving a producer to a cooperative copy loop or a separate kernel changes its
reuse and can enable ptxas to contract a previously separate multiply and add.
Before a narrowing cast, that small difference can change a low-precision ULP of the
matmul operand. In particular, ``a * p - a * p`` must remain zero when the
source multiplies were separately rounded.

The caller uses this only without fast math. Explicit FMA calls remain fused;
integer, low-precision, FP64, and unproven arithmetic remain unchanged.

``mark_narrowed_fp32_multiplies`` applies the same contract to the FP32
products an explicit cast rounds to a lower-precision float before more
arithmetic, independently of where the product is emitted.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

import torch

from ..ast_extension import expr_from_string
from .indexing import is_cute_shape_chain_target
from .scalar_recipe import _MATH_CALLS
from .scalar_recipe import ROUNDED_FP32_MULTIPLY_ASM
from .scalar_recipe import ROUNDED_FP32_MULTIPLY_CONSTRAINTS
from .scalar_recipe import ROUNDED_MULTIPLY_FORMS
from .scalar_recipe import _clone
from .scalar_recipe import _path

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from torch.fx import Node

_Kind = Literal["fp32", "literal"] | None
_FLOAT_MATH_CALLS = _MATH_CALLS - {"isfinite", "isinf", "isnan"}

# A materialized pointwise producer has the same rounding contract as a
# rematerialized collective operand. Its FX dtype facts can prove products
# whose input is a direct FP32 load, without an explicit generated cast.
FP32_MULTIPLY_ROUNDING_META_KEY = "cute_preserve_fp32_multiply_rounding"
# A flagged product whose operands are upcasts of one 16-bit float dtype and
# whose every use is a cast back to that dtype is rewritten into a multiply of
# the 16-bit operands (``narrow_exact_products``); the lowering emits it as an
# explicitly rounded ``mul.rn.bf16`` / ``mul.rn.f16`` so ptxas cannot contract
# it either.
NARROW_MULTIPLY_ROUNDING_META_KEY = "cute_narrow_multiply_rounding"
# ``mul{.rnd}.bf16`` entered PTX with ISA 7.8 for sm_90; ``mul{.rnd}.f16`` is
# available from sm_53 (``narrow_exact_products``).
NARROW_BF16_MIN_CAPABILITY = (9, 0)
# The spellings are shared with ``scalar_recipe.ROUNDED_MULTIPLY_FORMS`` so the
# recipe purity check admits them (``scalar_recipe.is_rounded_multiply``).
NARROW_MULTIPLY_ROUNDING_EXPR = {
    dtype: (
        "_cute_inline_asm_elementwise("
        f"(cutlass.{name}({{a}}), cutlass.{name}({{b}})), "
        f"asm={ROUNDED_MULTIPLY_FORMS[name][0]!r}, "
        f"constraints={ROUNDED_MULTIPLY_FORMS[name][1]!r}, "
        f"dtype=cutlass.{name}, is_pure=True)"
    )
    for dtype, name in ((torch.bfloat16, "BFloat16"), (torch.float16, "Float16"))
}
FP32_MULTIPLY_ROUNDING_EXPR = (
    "_cute_inline_asm_elementwise("
    "(cutlass.Float32({a}), cutlass.Float32({b})), "
    f"asm={ROUNDED_FP32_MULTIPLY_ASM!r}, "
    f"constraints={ROUNDED_FP32_MULTIPLY_CONSTRAINTS!r}, "
    "dtype=cutlass.Float32, is_pure=True)"
)


_PRODUCTS = frozenset({torch.ops.aten.mul.Tensor, torch.ops.aten.mul_.Tensor})
_SQUARE = torch.ops.aten.pow.Tensor_Scalar
_CONVERT = torch.ops.prims.convert_element_type.default
_WHERE = torch.ops.aten.where.self
# Shape and sign changes LLVM sees through; ``hl.subscript`` (``x[:, None]``)
# joins them in ``_forwarded``.
_VALUE_VIEWS = frozenset(
    {
        torch.ops.aten.transpose.int,
        torch.ops.aten.t.default,
        torch.ops.aten.neg.default,
    }
)


def mark_narrowed_fp32_multiplies(graph: torch.fx.Graph) -> None:
    """Flag FP32 products that an explicit cast rounds before more arithmetic.

    ``(a.float() * b.float()).to(half) - (c.float() * d.float()).to(half)``
    asks for each product rounded to the half type before the subtraction.
    LLVM narrows ``fptrunc(fmul(fpext a, fpext b))`` to an exact half multiply
    and emits it without a rounding modifier, so ptxas may contract that
    ``mul.f16`` with the following ``sub.f16`` into one half FMA, dropping the
    product's rounding.  Whether it does depends on unrelated codegen details
    (a predicated load feeding the multiply blocks the narrowing).  A
    ``mul.rn.f32`` can be neither narrowed nor contracted, so the cast rounds
    the product exactly as written.

    The hazard needs three things, and only a product with all three is
    flagged:

    * a product LLVM can narrow: ``aten.mul.Tensor``, its in-place form or
      ``x ** 2`` (rendered ``x * x``) whose operands are all lower-precision
      floats, upcasts of them or Python scalars.  A product with a genuinely
      FP32 operand (a load, an earlier FP32 result, an accumulator) is never
      narrowed, so it keeps its plain multiply;
    * a cast of the product, directly or through views, ``where`` or a
      negation, to a float narrower than FP32;
    * a consumer of the cast value, again through views, ``where``,
      negations, multiplies by ``1.0``/``-1.0`` (LLVM folds those away) or
      an upcast, that can take the product into an FMA.  A store, a matmul
      lhs/rhs operand and another multiply consume the rounded value as is
      and keep the plain multiply (and its packed half lowering); every
      other consumer counts.

    A tensor filled with one Python scalar (``hl.full``, ``torch.full_like``)
    is a splat constant LLVM narrows like the scalar itself.  Known gap: a
    scalar operand counts as narrow even when the half type cannot represent
    it (LLVM would not narrow; the only cost is a ``mul.rn.f32``).
    """
    from ...language import memory_ops
    from ..device_ir_analysis import matmul_operand_positions

    operand_positions = matmul_operand_positions()
    flagged: list[Node] = []
    for node in graph.nodes:
        if not _is_product(node):
            continue
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor) or value.dtype is not torch.float32:
            continue
        if not all(_is_narrow(operand) for operand in _product_operands(node)):
            continue
        if any(
            _can_fuse(cast, operand_positions, memory_ops.store)
            for cast in _narrowing_casts(node)
        ):
            node.meta[FP32_MULTIPLY_ROUNDING_META_KEY] = True
            flagged.append(node)
    narrow_exact_products(graph, flagged)


def _upcast_source(value: object) -> Node | None:
    """The 16-bit float node ``value`` is the FP32 upcast of, else None."""
    if not (
        isinstance(value, torch.fx.Node)
        and value.op == "call_function"
        and value.target is _CONVERT
        and len(value.args) == 2
        and value.args[1] is torch.float32
        and isinstance(value.args[0], torch.fx.Node)
    ):
        return None
    source = value.args[0]
    tensor = source.meta.get("val")
    if not isinstance(tensor, torch.Tensor) or tensor.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        return None
    return source


def _product_factors(product: Node) -> tuple[object, object] | None:
    """The two factors of a flagged product, a repeated factor restored.

    ``x ** 2`` is ``pow(x, 2)``; ``inductor_lowering.strip_unused_inputs``
    masks a repeated operand with None after the lowering (``x * x`` arrives
    as ``mul(x, None)``), and the masked operand is the node's unique tensor
    input (``flash_row_epilogue._operands``).
    """
    if product.target is _SQUARE:
        return product.args[0], product.args[0]
    if product.target is not torch.ops.aten.mul.Tensor or len(product.args) != 2:
        return None
    left, right = product.args
    if right is None and isinstance(left, torch.fx.Node):
        return left, left
    if left is None and isinstance(right, torch.fx.Node):
        return right, right
    return left, right


def narrow_exact_products(graph: torch.fx.Graph, products: Sequence[Node]) -> None:
    """Multiply 16-bit operands directly where that is bit-identical.

    For ``(a.float() * b.float()).to(a.dtype)`` with ``a`` and ``b`` of one
    16-bit float dtype, the FP32 product is exact (two 8- or 11-bit
    significands fit in 24 bits), so its single rounding to that dtype is the
    rounding of a ``mul.rn`` in the dtype itself.  Each product among
    ``products`` (already flagged ``FP32_MULTIPLY_ROUNDING_META_KEY``) whose
    operands are both such upcasts of the same dtype and whose every use is
    a direct cast back to it is replaced: the casts become one 16-bit
    ``aten.mul`` of the sources, marked ``NARROW_MULTIPLY_ROUNDING_META_KEY``
    for the lowering's explicitly rounded ``mul.rn``, and the dead FP32
    product and upcasts are erased.  A self-product (``f * f``, ``f ** 2``)
    is the same multiply with one source twice.  The SASS of a rope step then holds one
    packed-width multiply per element instead of two upcasts, an FP32
    multiply and a convert (nvfp4-free pointwise kernels are issue-bound at
    low occupancy: examples/rope.py s2 29.8 vs 31.4 us).  Under fast math
    the FP32 product is a plain multiply the compiler may narrow itself, so
    nothing is rewritten.  ``mul.rn.bf16`` is an sm_90+ PTX form (PTX ISA
    7.8), so a bfloat16 product is rewritten only for a known target of at
    least that capability and otherwise keeps the FP32 form, which is legal
    on every CuTe target; ``mul.rn.f16`` needs only sm_53.
    """
    from ..compile_environment import CompileEnvironment
    from ..inductor_lowering import prepare_added_node_lowering

    env = CompileEnvironment.current()
    if env.settings.fast_math:
        return
    capability = env.config_spec.target_device_capability
    narrowable = {torch.float16}
    if capability is not None and capability >= NARROW_BF16_MIN_CAPABILITY:
        narrowable.add(torch.bfloat16)
    for product in products:
        factors = _product_factors(product)
        if factors is None:
            continue
        sources = [_upcast_source(factor) for factor in factors]
        if any(source is None for source in sources):
            continue
        left, right = cast("list[Node]", sources)
        dtype = cast("torch.Tensor", left.meta["val"]).dtype
        if dtype not in narrowable:
            continue
        if cast("torch.Tensor", right.meta["val"]).dtype is not dtype:
            continue
        casts = list(product.users)
        if not casts or not all(
            user.op == "call_function"
            and user.target is _CONVERT
            and len(user.args) == 2
            and user.args[0] is product
            and user.args[1] is dtype
            for user in casts
        ):
            continue
        with graph.inserting_after(product):
            narrow = graph.call_function(torch.ops.aten.mul.Tensor, (left, right))
        # The cast's value (shape and 16-bit dtype) and source position; its
        # lowering-derived keys (``lowering``, ``orig_node``, ``output_index``)
        # and any value cached on the convert belong to the erased node.
        for key in ("val", "location", "stack_trace"):
            if key in casts[0].meta:
                narrow.meta[key] = casts[0].meta[key]
        narrow.meta[NARROW_MULTIPLY_ROUNDING_META_KEY] = True
        # The graph's lowerings are already prepared: give the new node its own.
        prepare_added_node_lowering(graph, narrow)
        for user in casts:
            user.replace_all_uses_with(narrow)
            graph.erase_node(user)
        upcasts = dict.fromkeys(
            factor for factor in factors if isinstance(factor, torch.fx.Node)
        )
        graph.erase_node(product)
        for upcast in upcasts:
            if not upcast.users:
                graph.erase_node(upcast)


def _is_product(node: Node) -> bool:
    if node.op != "call_function":
        return False
    if node.target in _PRODUCTS:
        return True
    if node.target is not _SQUARE or len(node.args) != 2:
        return False
    exponent = node.args[1]
    if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
        return False
    return exponent == 2


def _product_operands(node: Node) -> tuple[object, ...]:
    return node.args[:1] if node.target is _SQUARE else node.args[:2]


def _below_fp32(dtype: object) -> bool:
    return (
        isinstance(dtype, torch.dtype)
        and dtype.is_floating_point
        and dtype.itemsize < 4
    )


def _is_unit_scalar(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and abs(value) == 1
    )


def _is_scalar_fill(node: Node) -> bool:
    """Whether ``node`` is a tensor filled with one Python scalar."""
    from ...language import creation_ops

    if node.op != "call_function":
        return False
    if node.target is torch.ops.aten.scalar_tensor.default:
        fill = node.args[0] if node.args else None
    elif node.target in (
        creation_ops.full,
        torch.ops.aten.full.default,
        torch.ops.aten.full_like.default,
    ):
        fill = node.args[1] if len(node.args) > 1 else None
    else:
        return False
    return isinstance(fill, (int, float)) and not isinstance(fill, bool)


def _forwarded(node: Node) -> tuple[object, ...] | None:
    """The operands ``node`` passes on unchanged but for shape or sign."""
    from ...language import view_ops

    if node.op != "call_function":
        return None
    if node.target is _WHERE:
        return node.args[1:3]
    if node.target is torch.ops.aten.mul.Tensor and len(node.args) == 2:
        # ``x * 1.0`` and ``x * -1.0`` fold to ``x`` and ``-x``.
        left, right = node.args
        if _is_unit_scalar(right):
            return (left,)
        if _is_unit_scalar(left):
            return (right,)
    if (
        node.target is view_ops.subscript
        or node.target in _VALUE_VIEWS
        or is_cute_shape_chain_target(node.target)
    ):
        return node.args[:1]
    return None


def _passes(node: Node, value: Node) -> bool:
    forwarded = _forwarded(node)
    return forwarded is not None and any(source is value for source in forwarded)


def _is_narrow(value: object) -> bool:
    """Whether LLVM sees ``value`` as a float narrower than FP32 or a scalar."""
    if not isinstance(value, torch.fx.Node):
        return True
    tensor = value.meta.get("val")
    if not isinstance(tensor, torch.Tensor):
        return True
    if _below_fp32(tensor.dtype):
        return True
    if tensor.dtype is not torch.float32:
        return False
    if _is_scalar_fill(value):
        return True
    if value.op == "call_function" and value.target is _CONVERT:
        return _is_narrow(value.args[0])
    forwarded = _forwarded(value)
    return forwarded is not None and all(_is_narrow(source) for source in forwarded)


def _narrowing_casts(product: Node) -> list[Node]:
    """Casts of ``product`` below FP32, reached through pass-through nodes."""
    casts: list[Node] = []
    seen: set[Node] = set()
    stack = [product]
    while stack:
        value = stack.pop()
        for user in value.users:
            if user.op != "call_function":
                continue
            if user.target is _CONVERT:
                if user.args[0] is value and _below_fp32(user.args[1]):
                    casts.append(user)
            elif _passes(user, value) and user not in seen:
                seen.add(user)
                stack.append(user)
    return casts


def _can_fuse(
    cast: Node, operand_positions: Mapping[object, tuple[int, int]], store: object
) -> bool:
    """Whether the rounded value reaches a consumer able to fuse the product."""
    seen: set[Node] = set()
    stack = [cast]
    while stack:
        value = stack.pop()
        for user in value.users:
            if user.target is store:
                continue
            positions = operand_positions.get(user.target)
            if positions is not None and any(
                len(user.args) > position and user.args[position] is value
                for position in positions
            ):
                continue
            upcast = (
                user.op == "call_function"
                and user.target is _CONVERT
                and user.args[0] is value
                and user.args[1] is torch.float32
            )
            if upcast or _passes(user, value):
                if user not in seen:
                    seen.add(user)
                    stack.append(user)
            elif not _is_product(user):
                return True
    return False


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
