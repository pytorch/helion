"""General-purpose emitter: CuTe forms for the shared Inductor op handler.

``GenerateASTFromInductor`` asks the backend's :class:`HandlerOpOverrides`
before its shared lowering for the few ops whose CuTe form needs the handler's
lifting facilities: the fp8 and exact-fp32 widening casts, ``rsqrt`` under the
fast-math flag, nan-propagating ``max``/``min``, ``neg``/``abs`` without the
DSL's signedness quirks, and narrow-integer shifts truncated like torch.
Bodies moved verbatim from ``helion/_compiler/inductor_lowering.py``.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ..ast_extension import expr_from_string
from ..backend import HandlerOpOverrides
from ..compile_environment import CompileEnvironment
from ..device_function import DeviceFunction
from ..inductor_lowering import _unpack_opsvalue

if TYPE_CHECKING:
    from ..inductor_lowering import GenerateASTFromInductor


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
        if dtype is torch.float32:
            # A 16-bit value the mma_dag emitters hold exactly in fp32 (an
            # unpacked tile element, a widening side load) widens to that
            # value, not through a cast pair.  The record lives on the
            # device function's cute state; a lowering driven by a stand-in
            # device function (the flex score-mod lowering's tests) has none.
            device_function = handler.cg.device_function
            x_ast = handler._to_ast(x)
            if isinstance(device_function, DeviceFunction) and isinstance(
                x_ast, ast.Name
            ):
                exact = device_function.cute_state.root_plans.mma_dag_exact_f32.get(
                    x_ast.id
                )
                if exact is not None:
                    return handler._lift(expr_from_string(exact))
        return None

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
        dtype = handler._expected_tensor_dtype()
        if dtype is not None:
            a = handler._create_cast_expr(a, dtype)
            b = handler._create_cast_expr(b, dtype)
        return handler._lift(
            expr_from_string(
                "cute.math.max({a}, {b}, propagate_nan=True)",
                a=handler._to_ast(a),
                b=handler._to_ast(b),
            )
        )

    def minimum(
        self, handler: GenerateASTFromInductor, a: object, b: object
    ) -> str | None:
        dtype = handler._expected_tensor_dtype()
        if dtype is not None:
            a = handler._create_cast_expr(a, dtype)
            b = handler._create_cast_expr(b, dtype)
        return handler._lift(
            expr_from_string(
                "cute.math.min({a}, {b}, propagate_nan=True)",
                a=handler._to_ast(a),
                b=handler._to_ast(b),
            )
        )
