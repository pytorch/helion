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
from ...language.view_ops import subscript
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
# The ops that give their operand's elements by their kind: a plain copy of
# a variable (``kept = inner`` is a ``_new_var``); a subscript (this backend
# admits one of a computed value for ``None`` and full slices only); the
# ATen views that ``aten_lowering`` registers (the three ``squeeze``
# overloads share one lowering; ``view.dtype``, which reads its operand's
# bits as another dtype, is none); and the two ATen copies.  Their lowering
# computes nothing: the plain copy emits ``kept = v``, the others no
# statement at all.
_SAME_ELEMENTS = frozenset(
    {
        _tracing_ops._new_var,
        subscript,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.lift_fresh_copy.default,
    }
)
# The ops that give an integer operand's elements when their other operand is
# this literal: value identities.  Their lowering emits a statement, which the
# optimizer removes (``x ** 1`` emits none), so ptxas sees a chain through
# them.  ``div.Tensor_mode`` is the floor or the truncating division: the
# true one gives a float.
_IDENTITY_LITERALS: dict[object, int] = {
    torch.ops.aten.add.Tensor: 0,
    torch.ops.aten.sub.Tensor: 0,
    torch.ops.aten.mul.Tensor: 1,
    torch.ops.aten.div.Tensor_mode: 1,
    torch.ops.aten.pow.Tensor_Scalar: 1,
    torch.ops.aten.bitwise_or.Scalar: 0,
    torch.ops.aten.bitwise_xor.Scalar: 0,
    torch.ops.aten.bitwise_and.Scalar: -1,
    torch.ops.aten.__lshift__.Scalar: 0,
    torch.ops.aten.__rshift__.Scalar: 0,
}
# The two ops that copy their operand for one literal argument: a ``repeat``
# of one along every dim of its operand, and a ``flip`` of no dim (which
# reaches the device graph as ``prims.rev``).  No statement is emitted.
_REPEAT = torch.ops.aten.repeat.default
_FLIP = torch.ops.prims.rev.default
# The roundings, each the identity of an integer (no statement is emitted).
_ROUNDINGS = frozenset(
    {
        torch.ops.aten.round.default,
        torch.ops.aten.floor.default,
        torch.ops.aten.trunc.default,
        torch.ops.aten.ceil.default,
    }
)
# The two ops of which two in a row give their operand's elements: a negation
# of a negation, and a conversion to an integer dtype at least as wide and
# back.
_NEGATION = torch.ops.aten.neg.default
_CONVERSION = torch.ops.prims.convert_element_type.default


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


def extremum_result_kind(device_function: DeviceFunction, name: str) -> str | None:
    """``max`` / ``min`` when the variable ``name`` holds the result of such
    an elementwise extremum and of nothing else, else None.

    ``_nan_extremum`` records the variable of each extremum it emits
    (``CuteDeviceFunctionState.extremum_results``), and the codegen of a
    plain copy (``tracing_ops``) the copy of such a variable.  A loop or a
    branch that assigns a variable merges its names
    (``DeviceFunction.variable_aliases``), so every name merged with this
    one has to hold a result of the same kind."""
    results = device_function.cute_state.extremum_results
    kind = results.get(name)
    if kind is None:
        return None
    aliases = device_function.variable_aliases(name)
    return kind if all(results.get(alias) == kind for alias in aliases) else None


def _reads_extremum(
    handler: GenerateASTFromInductor, operand: ast.AST, kind: str
) -> bool:
    """Whether ``operand`` is a variable that holds the result of a ``kind``
    extremum (``extremum_result_kind``)."""
    return (
        isinstance(operand, ast.Name)
        and extremum_result_kind(handler.cg.device_function, operand.id) == kind
    )


def _integer_dtype(value: object) -> torch.dtype | None:
    """The dtype of the integer tensor that the FX node ``value`` computes,
    else None."""
    if not isinstance(value, torch.fx.Node):
        return None
    traced = value.meta.get("val")
    if not isinstance(traced, torch.Tensor):
        return None
    dtype = traced.dtype
    if dtype.is_floating_point or dtype.is_complex or dtype is torch.bool:
        return None
    return dtype


def _same_elements_operand(value: torch.fx.Node) -> torch.fx.Node | None:
    """The operand whose elements the node ``value`` holds unchanged, else
    None.

    ``value`` is a copy or a view of that operand (``_SAME_ELEMENTS``; a
    ``repeat`` of one along every dim of the operand; a ``flip`` of no dim),
    or a value identity of an integer of its dtype: an operation with the
    literal that leaves its other operand as it is (``_IDENTITY_LITERALS``:
    ``x + 0``, ``x * 1``, ``x // 1``, ``x | 0``, ``x << 0``, ...), a rounding,
    ``torch.where`` with one value for both branches, the negation of a
    negation, or the conversion back from one to an integer dtype at least
    as wide; copies, views and other identities may lie between the two
    negations or conversions.  The identities are named by their operation
    and a literal alone.  A computation that leaves the value as it is by
    the values or the range of its operands is none of them:
    ``x + (y - y)``, a ``where`` whose condition always holds, a clamp at
    the dtype's bound, a conversion to float64 and back.
    """
    if value.op != "call_function":
        return None
    target, args = value.target, value.args
    if target in _SAME_ELEMENTS:
        return args[0] if isinstance(args[0], torch.fx.Node) else None
    if target in (_REPEAT, _FLIP):
        source, literal = args[0], args[1] if len(args) > 1 else None
        if not isinstance(source, torch.fx.Node) or not isinstance(literal, list):
            return None
        if target is _FLIP:
            return source if not literal else None
        traced = source.meta.get("val")
        once_per_dim = (
            isinstance(traced, torch.Tensor)
            and len(literal) == traced.ndim
            and all(type(count) is int and count == 1 for count in literal)
        )
        return source if once_per_dim else None
    dtype = _integer_dtype(value)
    if dtype is None:
        return None
    operand: object = None
    if target in _IDENTITY_LITERALS:
        literal = args[1] if len(args) > 1 else None
        if type(literal) is int and literal == _IDENTITY_LITERALS[target]:
            operand = args[0]
    elif target in _ROUNDINGS:
        operand = args[0]
    elif target is torch.ops.aten.where.self:
        if args[1] is args[2]:
            operand = args[1]
    elif target in (_NEGATION, _CONVERSION):
        # The first of the two, through the nodes that hold its result
        # unchanged.
        first = args[0]
        while isinstance(first, torch.fx.Node) and first.target is not target:
            first = _same_elements_operand(first)
        if isinstance(first, torch.fx.Node) and first.target is target:
            between = _integer_dtype(first)
            if between is not None and between.itemsize >= dtype.itemsize:
                operand = first.args[0]
    if isinstance(operand, torch.fx.Node) and _integer_dtype(operand) == dtype:
        return operand
    return None


def _in_extremum_chain(kind: str) -> bool:
    """Whether the ``max`` / ``min`` (``kind``) the current FX node lowers to
    is a link of a chain of extrema of that kind in the FX graph.

    ptxas (CUDA 13.3) fuses two signed Int32 or Int16 maxima (or minima) of
    three values into one three-operand instruction and loses the negation
    of one operand, so the links of such a chain compare unsigned keys
    (``math_templates.nan_extremum_expr``).  A link is the node of a maximum
    (``torch.maximum``, a lower ``clamp`` bound, a ``relu``) that takes the
    result of another maximum as an operand, or whose result another maximum
    takes; the same for minima (``torch.minimum``, an upper ``clamp``
    bound).  That is every node of the chain, however long, whether or not
    the source names the value between two of them, and also when only
    nodes that hold that value's elements unchanged lie between them
    (``_same_elements_operand``): copies and views (``kept = inner``,
    ``[:, None]``, ``expand``, ``permute``, ``clone``, ...) and value
    identities (``inner + 0``, ``-(-inner)``, ...).

    A node that accumulates into a value its loop body takes in and hands
    back (``acc = torch.maximum(acc, x)`` per trip) is a link too: its trips
    chain.  The loop hands the value back in a statement after its body,
    which ``_nan_extremum`` cannot see, so this is read off the graph: an
    operand that is a plain copy (the body's graph takes a value in as one),
    and a result that the graph hands back, each through nodes that hold
    the same elements.  A copy of a value computed in the body counts as
    well: counting only the values the body takes in would take the keys
    from ``moved = inner + (y - y); kept = moved; last =
    torch.maximum(kept, c)``, which needs them and is no chain by the rule
    above.

    ``_nan_extremum`` adds the links that only the generated code shows: an
    extremum that reads the variable another one of its kind assigned.

    Every other extremum keeps the plain form, which the matmul lowerings
    read back from the index statements of their kernels: a single extremum,
    a maximum of a minimum (``torch.clamp`` with both bounds is one) and a
    minimum of a maximum were right at every layout measured.
    """
    node = V.current_node
    if node is None or _EXTREMUM_KINDS.get(node.target) != kind:
        return False

    def link(other: object) -> bool:
        return (
            isinstance(other, torch.fx.Node)
            and _EXTREMUM_KINDS.get(other.target) == kind
        )

    # The value whose elements each operand holds, and whether a plain copy
    # lies on the way to it.
    takes_copy = False
    operand_links = False
    for value in node.args:
        while isinstance(value, torch.fx.Node):
            takes_copy |= value.target is _tracing_ops._new_var
            operand = _same_elements_operand(value)
            if operand is None:
                break
            value = operand
        operand_links |= link(value)
    # The users of the result and of the nodes that hold its elements.
    users = []
    pending = [node]
    while pending:
        holder = pending.pop()
        for user in holder.users:
            if _same_elements_operand(user) is holder:
                pending.append(user)
                continue
            users.append(user)
            # A negation or a conversion may be the first of two that give
            # the elements back: the second is a user of the first, or of
            # a node that holds the first one's result unchanged.
            firsts = [user]
            while firsts:
                first = firsts.pop()
                for second in first.users:
                    held = _same_elements_operand(second)
                    if held is holder:
                        pending.append(second)
                    elif held is first:
                        firsts.append(second)
    if operand_links or any(link(user) for user in users):
        return True
    return takes_copy and any(user.op == "output" for user in users)


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
    keys.  A link is a node that chains in the FX graph
    (``_in_extremum_chain``), or an extremum one of whose operands is a
    variable that holds the result of an extremum of its kind
    (``_reads_extremum``): whatever lies between the two in the graph
    emitted no statement (a select on a dim of size one, a narrow over the
    whole extent, any op whose lowering returns its operand) or a plain copy
    (``kept = inner``), so the generated code is a chain, of which this
    later link is keyed; keys on one of two extrema keep them apart.  A
    value identity that emits a statement (``inner + 0``) is a link in the
    graph.  Any other statement between two extrema ends the chain here.
    Where the optimizer removes it all the same (``inner + (y - y)``) ptxas
    fuses the two: not covered.
    """
    operands = (handler._to_ast(a), handler._to_ast(b))
    dtype = handler._expected_tensor_dtype()
    chain_dtype = None
    if dtype is not None:
        if _in_extremum_chain(kind) or any(
            _reads_extremum(handler, operand, kind) for operand in operands
        ):
            chain_dtype = CompileEnvironment.current().backend.dtype_str(dtype)
        operands = tuple(handler._cast_ast(operand, dtype) for operand in operands)
    float32 = (
        dtype is torch.float32
        and handler.cg.device_function.config.config.get("cute_fmnmx_extrema") is True
    )
    result = handler._lift(
        expr_from_string(
            nan_extremum_expr(
                kind, "{a}", "{b}", float32=float32, chain_dtype=chain_dtype
            ),
            a=operands[0],
            b=operands[1],
        )
    )
    handler.cg.device_function.cute_state.extremum_results[result] = kind
    return result


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
