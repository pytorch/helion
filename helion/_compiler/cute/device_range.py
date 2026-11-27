"""Validate the original operands of a nested CuTe Int32 range."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.utils._sympy.value_ranges import ValueRanges

from ... import exc
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    import ast
    from collections.abc import Iterable

    from ..device_function import DeviceFunction


def needs_wide_increment(begin: object, end: object, step: object) -> bool:
    """CuTe walks descending ranges as ascending ranges from end to begin."""
    if not isinstance(step, int):
        return True
    if abs(step) <= 1:
        return False
    lower, upper = (begin, end) if step > 0 else (end, begin)
    stride = abs(step)
    if not isinstance(upper, int):
        return True
    if isinstance(lower, int):
        return lower + len(range(lower, upper, stride)) * stride > 2**31 - 1
    return upper + stride - 1 > 2**31 - 1


def int32_range_guard(
    operands: Iterable[tuple[object, str, bool]], device_function: DeviceFunction
) -> ast.AST | None:
    """Check each (original value, device expression, is step) before narrowing."""
    env = CompileEnvironment.current()
    hi = 2**31 - 1
    checks = []
    message = "nested CuTe tile range bounds and the magnitude of the step must fit signed Int32; the step must be nonzero"
    for bound, value, step in operands:
        # CuTe normalizes a descending loop by negating its step in Int32.
        lo = -hi if step else -hi - 1
        if isinstance(bound, torch.SymInt) and isinstance(
            bound.node.expr, sympy.Integer
        ):
            bound = int(bound.node.expr)
        if isinstance(bound, int):
            if not lo <= bound <= hi or (step and not bound):
                raise exc.BackendUnsupported("cute", message)
            continue
        if isinstance(bound, torch.Tensor) and bound.dtype.is_floating_point:
            # CuTe converts float bounds toward zero. Both exclusive limits
            # are exact in Float64, and NaN fails this ordered comparison.
            v = f"cutlass.Float64({value})"
            valid = f"{v} > {lo - 1} and {v} < {hi + 1}"
            if step:
                valid += f" and ({v} <= -1 or {v} >= 1)"
            checks.append(f"not ({valid})")
            continue
        if step:
            checks.append(f"(({value}) == 0)")
        if isinstance(bound, torch.SymInt):
            symbol = cast("sympy.Expr", bound.node.expr)
            # Use the actual argument dimension even when host code renamed
            # its size. A dimension from a wrapper allocation is not proof.
            if (
                env.index_dtype == torch.int32
                and not env.wide_empty_grid
                and symbol in device_function.host_grid_size_exprs
            ):
                continue
            if env.shape_env.bound_sympy(symbol).issubset(ValueRanges(lo, hi)):
                continue
        if isinstance(bound, torch.Tensor):
            dtype = bound.dtype
            if dtype == torch.bool:
                continue
            if dtype.is_complex:
                raise exc.BackendUnsupported("cute", "complex nested tile range")
            limits = torch.iinfo(dtype)
            if lo <= limits.min and limits.max <= hi:
                continue
            if dtype == torch.uint64:
                # CuTe loads uint64 memory through a signed Int64 carrier.
                value = f"cutlass.Uint64({value})"
            if limits.min >= lo:
                checks.append(f"(({value}) > {hi})")
                continue
        checks.append(f"(({value}) < ({lo}) or ({value}) > {hi})")
    if not checks:
        return None
    error = env.backend.device_error_stmt(message)
    return statement_from_string(f"if {' or '.join(checks)}:\n    {error}")
