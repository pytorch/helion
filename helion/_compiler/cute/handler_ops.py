"""General-purpose emitter: CuTe forms for the shared Inductor op handler.

``GenerateASTFromInductor`` asks the backend's :class:`HandlerOpOverrides`
before its shared lowering for the few ops whose CuTe form needs the handler's
lifting facilities: the fp8 and exact-fp32 widening casts, the conversions of
integers and bools to floats (``integer_to_float_expr``), ``rsqrt`` under the
fast-math flag, nan-propagating ``max``/``min``, ``neg``/``abs`` without the
DSL's signedness quirks, and narrow-integer shifts truncated like torch.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch
from torch._inductor.virtualized import V

from ...language import _tracing_ops
from ..ast_extension import expr_from_string
from ..backend import HandlerOpOverrides
from ..compile_environment import CompileEnvironment
from ..device_function import DeviceFunction
from ..host_function import HostFunction
from ..inductor_lowering import _unpack_opsvalue
from .math_templates import nan_extremum_expr
from .state import cute_codegen_state

if TYPE_CHECKING:
    from torch.fx import Node

    from ..device_ir import DeviceIR
    from ..inductor_lowering import GenerateASTFromInductor


# The pointwise ops that lower to an extremum of two computed values (a
# ``relu`` or a ``clamp`` against constants is one instruction either way).
_TRACED_EXTREMA = frozenset(
    {torch.ops.aten.maximum.default, torch.ops.aten.minimum.default}
)
# The pointwise ops that take one extremum per element, and which: the links
# a chain of extrema is made of (``_in_extremum_chain``).  ``torch.clamp``
# traces as a ``clamp_min`` under a ``clamp_max``.
_EXTREMUM_KINDS: dict[object, str] = {
    torch.ops.aten.maximum.default: "max",
    torch.ops.aten.clamp_min.default: "max",
    torch.ops.aten.clamp_min.Tensor: "max",
    torch.ops.aten.relu.default: "max",
    torch.ops.aten.minimum.default: "min",
    torch.ops.aten.clamp_max.default: "min",
    torch.ops.aten.clamp_max.Tensor: "min",
}


class CuteHandlerOpOverrides(HandlerOpOverrides):
    def to_dtype(
        self,
        handler: GenerateASTFromInductor,
        x: object,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None,
    ) -> str | None:
        if src_dtype is torch.float8_e4m3fn and dtype is torch.float32:
            cast_expr = expr_from_string(
                "_cute_fp8e4m3fn_to_float32({x})",
                x=handler._to_ast(x),
            )
            return handler._lift(cast_expr)
        x_ast = handler._to_ast(x)
        device_function = handler.cg.device_function
        converted = integer_to_float_expr(x_ast, dtype, src_dtype, device_function)
        if converted is not None:
            return handler._lift(converted)
        if dtype is torch.float32:
            # A 16-bit value the mma_dag emitters hold exactly in fp32 (an
            # unpacked tile element, a widening side load) widens to that
            # value, not through a cast pair.  The record lives on the
            # device function's cute state; a lowering driven by a stand-in
            # device function (the flex score-mod lowering's tests) has none.
            if isinstance(device_function, DeviceFunction) and isinstance(
                x_ast, ast.Name
            ):
                exact = device_function.cute_state.root_plans.mma_dag_exact_f32.get(
                    x_ast.id
                )
                if exact is not None:
                    return handler._lift(expr_from_string(exact))
        return None

    def div_rn(
        self, handler: GenerateASTFromInductor, a: object, b: object
    ) -> str | None:
        # ``/`` compiles to ``div.full.f32`` (the launcher's ``-prec-div=0``)
        # unless ``Settings.cute_exact_division`` asks for the IEEE division.
        # ``div.rn.f32`` as an NVVM intrinsic keeps its rounding under either
        # option, and no pass rewrites it: under the setting too, where
        # ``fast_math`` turns a ``/`` by a loop-invariant divisor into a
        # multiply by its reciprocal (41 * RN(1 / 41) truncates to 0).  It
        # takes fp32 scalars, so 16-bit and integer operands are widened
        # first; the quotient stays fp32, as the Triton backend's does, for
        # the lowering to round where eager does and for the store to cast.
        return handler._lift(
            expr_from_string(
                "cute.math.div({a}, {b}, rounding=cute.math.RoundingMode.NEAREST_EVEN)",
                a=handler._create_cast_expr(a, torch.float32),
                b=handler._create_cast_expr(b, torch.float32),
            )
        )

    def rsqrt(self, handler: GenerateASTFromInductor, x: object) -> str | None:
        suffix = (
            ", fastmath=True" if CompileEnvironment.current().settings.fast_math else ""
        )
        return handler._lift(
            expr_from_string(f"cute.math.rsqrt({{x}}{suffix})", x=handler._to_ast(x))
        )

    def integer_shift(
        self, handler: GenerateASTFromInductor, name: str, x: object, y: object
    ) -> str | None:
        dtype = handler._expected_tensor_dtype()
        if dtype is None or dtype not in (
            torch.int8,
            torch.uint8,
            torch.int16,
            torch.uint16,
        ):
            return None
        # CuTe promotes a narrow integer combined with a Python shift count
        # to Int32. Torch truncates each tensor operation to its result dtype,
        # before any subsequent shift or conversion.
        result = handler._parse_op(getattr(handler.parent_handler, name)(x, y))
        return handler._lift(handler._create_cast_expr(result, dtype))

    def neg(self, handler: GenerateASTFromInductor, x: object) -> str | None:
        return handler._lift(expr_from_string("-({x})", x=handler._to_ast(x)))

    def abs(self, handler: GenerateASTFromInductor, x: object) -> str | None:
        dtype = handler._expected_tensor_dtype()
        if dtype is not None and not dtype.is_signed:
            # The DSL's abs reads an unsigned value as signed.
            return _unpack_opsvalue(x)
        return handler._lift(expr_from_string("abs({x})", x=handler._to_ast(x)))

    def maximum(
        self, handler: GenerateASTFromInductor, a: object, b: object
    ) -> str | None:
        return _nan_extremum(handler, "max", a, b)

    def minimum(
        self, handler: GenerateASTFromInductor, a: object, b: object
    ) -> str | None:
        return _nan_extremum(handler, "min", a, b)


def _in_extremum_chain(kind: str) -> bool:
    """Whether the ``max`` / ``min`` (``kind``) the current FX node lowers to
    is a link of a chain of extrema of that kind.

    ptxas (CUDA 13.3) fuses two signed Int32 or Int16 maxima (or minima) of
    three values into one three-operand instruction and loses the negation
    of one operand, so the links of such a chain compare unsigned keys
    (``math_templates.nan_extremum_expr``).  A link is the node of a maximum
    (``torch.maximum``, a lower ``clamp`` bound, a ``relu``) that takes the
    result of another maximum as an operand, or whose result another maximum
    takes; the same for minima (``torch.minimum``, an upper ``clamp``
    bound).  That is every node of the chain, however long, whether or not
    the source names the value between two of them.  A node that
    accumulates into a value its loop body takes in and hands back
    (``acc = torch.maximum(acc, x)`` per trip) is a link too: its trips
    chain.

    Every other extremum keeps the plain form, which the matmul lowerings
    read back from the index statements of their kernels: a single extremum,
    a maximum of a minimum (``torch.clamp`` with both bounds is one) and a
    minimum of a maximum were right at every layout measured.
    """
    node = V.current_node
    if node is None or _EXTREMUM_KINDS.get(node.target) != kind:
        return False

    def link(other: object) -> bool:
        return _EXTREMUM_KINDS.get(getattr(other, "target", None)) == kind

    if any(link(arg) for arg in node.args) or any(link(user) for user in node.users):
        return True
    return any(
        getattr(arg, "target", None) is _tracing_ops._new_var for arg in node.args
    ) and any(user.op == "output" for user in node.users)


def _nan_extremum(
    handler: GenerateASTFromInductor, kind: str, a: object, b: object
) -> str | None:
    """``torch.maximum`` / ``torch.minimum`` of two values of the op's dtype.

    Under ``cute_fmnmx_extrema`` an fp32 extremum is the one-instruction
    NaN-propagating form (``math_templates.nan_extremum_expr``, as the
    reductions use): besides saving two compares and a select per element,
    it is the form ptxas folds into packed half-precision extrema when both
    operands are widened bf16 / fp16 lanes.  It orders ``-0.0`` below
    ``+0.0`` (IEEE 754-2019 maximum / minimum), where the compare-and-select
    form returns one of its operands by position.

    The extrema of one expression (``torch.maximum(torch.maximum(a, b), c)``)
    form a chain as a reduction's do: a link of one gets the op's CuTe dtype,
    with which the helper takes an Int32 or Int16 extremum over unsigned
    keys (``_in_extremum_chain``).
    """
    dtype = handler._expected_tensor_dtype()
    chain_dtype = None
    if dtype is not None:
        a = handler._create_cast_expr(a, dtype)
        b = handler._create_cast_expr(b, dtype)
        if _in_extremum_chain(kind):
            chain_dtype = CompileEnvironment.current().backend.dtype_str(dtype)
    float32 = (
        dtype is torch.float32
        and handler.cg.device_function.config.config.get("cute_fmnmx_extrema") is True
    )
    return handler._lift(
        expr_from_string(
            nan_extremum_expr(
                kind, "{a}", "{b}", float32=float32, chain_dtype=chain_dtype
            ),
            a=handler._to_ast(a),
            b=handler._to_ast(b),
        )
    )


def is_fp32_extremum(node: Node) -> bool:
    """Whether ``node`` is a traced fp32 ``torch.maximum`` / ``minimum``."""
    return (
        node.op == "call_function"
        and node.target in _TRACED_EXTREMA
        and isinstance(value := node.meta.get("val"), torch.Tensor)
        and value.dtype is torch.float32
    )


def has_fp32_extrema(device_ir: DeviceIR) -> bool:
    """Whether the kernel traces an fp32 ``torch.maximum`` / ``minimum``:
    what offers ``cute_fmnmx_extrema`` to the autotuner.  Structure only, no
    extents."""
    return any(
        is_fp32_extremum(node) for info in device_ir.graphs for node in info.graph.nodes
    )


def integer_to_float_expr(
    x: ast.AST,
    dtype: torch.dtype,
    src_dtype: torch.dtype | None,
    device_function: object,
) -> ast.AST | None:
    """``x``, an integer or a bool, converted to the float ``dtype`` with 0.0
    added where it becomes a float, or None for another conversion, or where
    the kernel (``device_function``) needs no such add.

    The DSL converts an integer or a bool to a float with one
    ``arith.sitofp`` / ``uitofp``, and MLIR folds into it what follows: float
    casts, and identities such as ``x * 1.0`` or ``x - 0.0``.  A conversion so
    folded into an 8-bit float fails in the NVVM backend (a bool, an Int32)
    or, for an 8-bit integer, lowers as a reinterpretation of its bits.
    Adding 0.0 where the integer becomes a float stops the fold, whatever
    copies, casts or identities follow it (exact: an integer's float is never
    -0.0).  An 8-bit float target takes the add in a Float32; another float
    target takes it only in a kernel that holds 8-bit floats
    (``_holds_8bit_floats``), leaving every other kernel's code as it was.
    The lowerings that convert an integer to a float go through it: the
    pointwise one and the computed fragment's (``computed_fragment``: its
    ``to_dtype``, fills and iota), the re-reads of ``hl.split`` and of a
    reshaped store's chain (``view_ops._split_leaf_expr``,
    ``memory_ops._cute_reread_leaf``), ``hl.full`` (``creation_ops``),
    ``torch.full`` and ``torch.scalar_tensor`` (``aten_lowering``) of an
    integer value, and a store of an integer scalar to a float tensor."""
    if (
        src_dtype is None
        or src_dtype.is_floating_point
        or src_dtype.is_complex
        or not dtype.is_floating_point
    ):
        return None
    if dtype.itemsize == 1:
        via = torch.float32
    else:
        if not (
            isinstance(device_function, DeviceFunction)
            and _holds_8bit_floats(device_function)
        ):
            return None
        via = dtype
    backend = CompileEnvironment.current().backend
    value = expr_from_string(
        "{x} + {zero}",
        x=backend.cast_ast(x, via),
        zero=backend.cast_ast(expr_from_string("0.0"), via),
    )
    return value if via is dtype else backend.cast_ast(value, dtype)


def scalar_value_dtype(value: object) -> torch.dtype | None:
    """The dtype of ``value``, a traced fill or store value (a tensor's own,
    ``bool`` for a SymBool, ``int64`` for a SymInt), or None for a constant,
    which the DSL converts when it traces the kernel."""
    if isinstance(value, torch.Tensor):
        return value.dtype
    if isinstance(value, torch.SymBool):
        return torch.bool
    if isinstance(value, torch.SymInt):
        return torch.int64
    return None


def _holds_8bit_floats(device_function: DeviceFunction) -> bool:
    """Whether a value of the kernel's device code, or a tensor it touches, is
    a float of one byte: only then can an integer's float reach an 8-bit float
    conversion (a store into an fp8 tensor converts its value first).  The
    packed ``float4_e2m1fn_x2`` and the ``float8_e8m0fnu`` scales count too,
    so the conversions of a kernel holding them take the (exact) add as
    well."""
    state = cute_codegen_state(device_function.codegen)
    if state.holds_8bit_floats is None:
        state.holds_8bit_floats = any(
            isinstance(val := node.meta.get("val"), torch.Tensor)
            and val.dtype.is_floating_point
            and val.dtype.itemsize == 1
            for info in HostFunction.current().device_ir.graphs
            for node in info.graph.nodes
        )
    return state.holds_8bit_floats
