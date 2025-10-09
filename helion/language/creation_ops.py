from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._inductor.utils import triton_type

import sympy

from .._compiler.ast_extension import expr_from_string
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.device_function import DeviceFunction
from .._compiler.host_function import NoCurrentFunction
from ..exc import NotInsideKernel
from . import _decorators
from .ref_tile import RefTile

if TYPE_CHECKING:
    import ast

    from .._compiler.inductor_lowering import CodegenState

__all__ = ["arange", "full", "zeros"]


def _normalize_arange_bounds(
    start: int | torch.SymInt,
    end: int | torch.SymInt | None,
    step: int | torch.SymInt,
) -> tuple[int | torch.SymInt, int | torch.SymInt, int | torch.SymInt]:
    if end is None:
        return 0, start, step
    return start, end, step


def _to_sympy(value: int | torch.SymInt) -> sympy.Expr:
    if isinstance(value, torch.SymInt):
        return value._sympy_()
    if isinstance(value, int):
        return sympy.Integer(value)
    raise TypeError(f"Unsupported arange bound of type {type(value)!r}")


def _size_hint(value: int | torch.SymInt, env: CompileEnvironment) -> int:
    if isinstance(value, torch.SymInt):
        return env.size_hint(value)
    if isinstance(value, int):
        return value
    raise TypeError(f"Unsupported arange bound of type {type(value)!r}")


def _extract_tile_symbol(
    env: CompileEnvironment, bound: int | torch.SymInt
) -> tuple[sympy.Expr, str] | None:
    if not isinstance(bound, torch.SymInt):
        return None
    expr = bound._sympy_()
    for key, value in env._symint_cache.items():
        if isinstance(key, tuple) and key:
            tag = key[0]
            if tag in {"tile_begin", "tile_end"}:
                symbol = key[1]
                if not isinstance(symbol, sympy.Expr):
                    continue
                if value is bound or expr.has(symbol):
                    return symbol, tag
    return None


def _compute_length_info(
    env: CompileEnvironment,
    start: int | torch.SymInt,
    end: int | torch.SymInt,
    step: int | torch.SymInt,
) -> tuple[torch.SymInt, int] | None:
    step_hint = abs(_size_hint(step, env))
    if step_hint == 0:
        raise ValueError("arange() step argument must not be zero")

    # Symbolic span between end and start
    span_sym = sympy.simplify(_to_sympy(end) - _to_sympy(start))
    span_hint = _size_hint(end - start, env)

    block_id = env.resolve_block_id(span_sym)
    tile_block_id: int | None = None
    span_abs_hint = abs(span_hint)
    if block_id is not None:
        block_info = env.block_sizes[block_id]
        if isinstance(block_info.size, (int, torch.SymInt)):
            span_abs_hint = abs(env.size_hint(block_info.size)) or step_hint
    else:
        # Try to reduce overly-large hints by clamping to the tile block size when
        # ``start``/``end`` originate from a tile iterator. This ensures we don't
        # over-estimate the compile-time extent for values like tile.end - tile.begin
        # which are bounded by the block size but lack precise symbolic information.
        tile_data_start = _extract_tile_symbol(env, start)
        tile_data_end = _extract_tile_symbol(env, end)
        tile_symbol = None
        saw_tile_end = False
        for data in (tile_data_start, tile_data_end):
            if data is None:
                continue
            symbol, tag = data
            tile_symbol = symbol
            if tag == "tile_end":
                saw_tile_end = True
        tile_block_id = None
        block_hint: int | None = None
        if tile_symbol is not None:
            tile_block_id = env.get_block_id(tile_symbol)
            if tile_block_id is not None:
                tile_block = env.block_sizes[tile_block_id]
                try:
                    device_config = DeviceFunction.current().config
                except NoCurrentFunction:
                    device_config = getattr(env, "_default_config_cache", None)
                    if device_config is None:
                        device_config = env.config_spec.default_config()
                        env._default_config_cache = device_config  # type: ignore[attr-defined]
                block_value = tile_block.from_config(device_config)
                if isinstance(block_value, torch.SymInt):
                    block_hint = abs(env.size_hint(block_value))
                elif isinstance(block_value, int):
                    block_hint = abs(block_value)
                else:
                    block_hint = abs(env.size_hint(tile_block.var))
    if span_abs_hint == 8192 and env.block_sizes:
        last = env.block_sizes[-1]
        if isinstance(last.size, (int, torch.SymInt)):
            span_abs_hint = abs(env.size_hint(last.size)) or step_hint
    if span_abs_hint == 0:
        span_abs_hint = step_hint

    length_hint = (span_abs_hint + step_hint - 1) // step_hint

    step_is_int = isinstance(step_hint, int)
    use_tile_block = (
        tile_block_id is not None
        and block_hint is not None
        and block_hint > 0
        and (
            saw_tile_end
            or (
                step_is_int
                and step_hint not in (0, 1)
                and length_hint <= block_hint
            )
        )
    )

    if use_tile_block:
        length_hint = min(length_hint, block_hint)

    if (span_hint <= 0 and step_hint > 0) or length_hint == 0:
        return None

    length_symint = env.cached_create_unbacked_symint(
        ("hl.arange.length", span_sym, step_hint), hint=max(length_hint, 1)
    )
    if use_tile_block and tile_block_id is not None:
        env.arange_tile_block_map[length_symint._sympy_()] = tile_block_id
    return length_symint, length_hint


def _is_constant(value: object, target: int) -> bool:
    if isinstance(value, int):
        return value == target
    if isinstance(value, torch.SymInt):
        return CompileEnvironment.current().known_equal(value, target)
    return False


def _normalize_ast_args(
    state: "CodegenState",
) -> tuple["ast.AST", "ast.AST", "ast.AST"]:
    start_ast = state.ast_arg(0)
    end_ast = state.ast_arg(1)
    step_ast = state.ast_arg(2)
    if state.proxy_arg(1) is None:
        # Only stop provided, make start explicit zero and reuse original start as stop
        stop_ast = start_ast
        start_ast = expr_from_string("0")
    else:
        stop_ast = end_ast
    return start_ast, stop_ast, step_ast


def zeros(
    shape: list[object],
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Return a device-tensor filled with zeros.

    Equivalent to ``hl.full(shape, 0.0 if dtype.is_floating_point else 0, dtype=dtype)``.

    Note:
        Only use within ``hl.tile()`` loops for creating local tensors.
        For output tensor creation, use ``torch.zeros()`` with proper device placement.

    Args:
        shape: A list of sizes (or tile indices which are implicitly converted to sizes)
        dtype: Data type of the tensor (default: torch.float32)

    Returns:
        torch.Tensor: A device tensor of the given shape and dtype filled with zeros

    Examples:

        .. code-block:: python

            @helion.kernel
            def process_kernel(input: torch.Tensor) -> torch.Tensor:
                result = torch.empty_like(input)

                for tile in hl.tile(input.size(0)):
                    buffer = hl.zeros([tile], dtype=input.dtype)  # Local buffer
                    buffer += input[tile]  # Add input values to buffer
                    result[tile] = buffer

                return result

    See Also:
        - :func:`~helion.language.full`: For filling with arbitrary values
        - :func:`~helion.language.arange`: For creating sequences
    """
    return full(
        shape, 0.0 if dtype.is_floating_point else 0, dtype=dtype, device=device
    )


@_decorators.api(tiles_as_sizes=True)
def full(
    shape: list[object],
    value: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Create a device-tensor filled with a specified value.

    Note:
        Only use within ``hl.tile()`` loops for creating local tensors.
        For output tensor creation, use ``torch.full()`` with proper device placement.

    Args:
        shape: A list of sizes (or tile indices which are implicitly converted to sizes)
        value: The value to fill the tensor with
        dtype: The data type of the tensor (default: torch.float32)

    Returns:
        torch.Tensor: A device tensor of the given shape and dtype filled with value

    Examples:
        .. code-block:: python

            @helion.kernel
            def process_kernel(input: torch.Tensor) -> torch.Tensor:
                result = torch.empty_like(input)

                for tile in hl.tile(input.size(0)):
                    # Create local buffer filled with initial value
                    buffer = hl.full([tile], 0.0, dtype=input.dtype)
                    buffer += input[tile]  # Add input values to buffer
                    result[tile] = buffer

                return result

    See Also:
        - :func:`~helion.language.zeros`: For filling with zeros
        - :func:`~helion.language.arange`: For creating sequences
    """
    raise NotInsideKernel


@_decorators.register_fake(full)
def _full_fake(
    shape: list[int | torch.SymInt],
    value: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(shape, (list, tuple)):
        raise TypeError(f"Expected list[SymInt], got {type(shape).__name__}")
    env = CompileEnvironment.current()
    env.add_kernel_tensor_size(shape)
    return torch.empty(
        [*shape],
        dtype=dtype,
        device=env.device if device is None else device,
    )


@_decorators.codegen(full)
def _full_codegen(state: CodegenState) -> ast.AST:
    fake_value = state.fake_value
    assert isinstance(fake_value, torch.Tensor)
    shape_str = state.device_function.tile_strategy.shape_str(fake_value.size())
    type_str = triton_type(fake_value.dtype)

    # Check if the value is static (literal) or dynamic (node)
    proxy_value = state.proxy_arg(1)
    if isinstance(proxy_value, (int, float, bool)):
        # For static values, use literal_expr to preserve special representations like float('-inf')
        value_str = state.device_function.literal_expr(proxy_value)
        return expr_from_string(f"tl.full({shape_str}, {value_str}, {type_str})")
    # For dynamic values, use ast_arg to get the proper AST representation
    value_ast = state.ast_arg(1)
    return expr_from_string(
        f"tl.full({shape_str}, {{value}}, {type_str})", value=value_ast
    )


@_decorators.get_masked_value(full)
def _(
    node: torch.fx.Node,
) -> float | bool | None:
    value = node.args[1]
    if isinstance(value, (int, float, bool)):
        return value
    # Return None for dynamic values (like tensor elements)
    return None


@_decorators.ref(full)
def _(
    shape: list[int | RefTile],
    value: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    processed_shape = []
    for s in shape:
        if isinstance(s, RefTile):
            processed_shape.append(s.end - s.begin)
        else:
            processed_shape.append(s)
    env = CompileEnvironment.current()
    return torch.full(
        processed_shape,
        value,
        dtype=dtype,
        device=env.device if device is None else device,
    )


@_decorators.api()
def arange(
    start: int | torch.SymInt,
    end: int | torch.SymInt | None = None,
    step: int | torch.SymInt = 1,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Helion equivalent of :func:`torch.arange` that automatically uses the active kernel
    configuration for dtype and device defaults.

    Args:
        start: Range start when ``end`` is provided, otherwise the exclusive stop value.
        end: Optional exclusive stop value. When omitted, ``start`` is treated as ``stop``.
        step: Step between consecutive elements (defaults to 1).
        dtype: Optional dtype override (defaults to the kernel index dtype).
        device: Optional device override (defaults to the kernel device).

    Returns:
        torch.Tensor: 1D tensor containing the requested integer sequence.
    """
    env = CompileEnvironment.current()
    start_norm, end_norm, step_norm = _normalize_arange_bounds(start, end, step)
    if dtype is None:
        dtype = env.settings.index_dtype
    return torch.arange(
        start_norm,
        end_norm,
        step_norm,
        dtype=dtype,
        device=env.device if device is None else device,
    )


@_decorators.register_fake(arange)
def _(
    start: int | torch.SymInt,
    end: int | torch.SymInt | None = None,
    step: int | torch.SymInt = 1,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    env = CompileEnvironment.current()
    start_norm, end_norm, step_norm = _normalize_arange_bounds(start, end, step)
    if dtype is None:
        dtype = env.settings.index_dtype
    device = env.device if device is None else device
    length_info = _compute_length_info(env, start_norm, end_norm, step_norm)
    if length_info is None:
        return torch.empty((0,), dtype=dtype, device=device)
    length_symint, length_hint = length_info
    tile_block_id = None
    if isinstance(length_symint, torch.SymInt):
        tile_block_id = env.arange_tile_block_map.pop(
            length_symint._sympy_(), None
        )

    if tile_block_id is not None:
        tile_block = env.block_sizes[tile_block_id]
        return torch.empty((tile_block.var,), dtype=dtype, device=device)

    rdim = env.allocate_reduction_dimension(length_symint)
    from .._compiler.host_function import HostFunction, SymbolOrigin
    from .._compiler.variable_origin import BlockSizeOrigin

    HostFunction.current().expr_to_origin[length_symint._sympy_()] = SymbolOrigin(
        BlockSizeOrigin(rdim.block_id)
    )
    rdim.mark_alternate_size(length_hint)
    return torch.empty((rdim.var,), dtype=dtype, device=device)


@_decorators.codegen(arange)
def _(state: "CodegenState") -> "ast.AST":
    env = CompileEnvironment.current()
    fake_value = state.fake_value
    assert isinstance(fake_value, torch.Tensor)

    block_id = env.resolve_block_id(fake_value.size(0))
    if block_id is not None:
        block_size_var = state.codegen.device_function.block_size_var(block_id)
        assert block_size_var is not None
        length_ast = expr_from_string(block_size_var)
    else:
        length_ast = expr_from_string(
            repr(env.size_hint(fake_value.size(0)))
        )

    start_ast, _, step_ast = _normalize_ast_args(state)
    start_val, _, step_val = _normalize_arange_bounds(
        state.proxy_arg(0), state.proxy_arg(1), state.proxy_arg(2)
    )

    expr = "tl.arange(0, {length})"
    kwargs: dict[str, object] = {"length": length_ast}

    if not _is_constant(step_val, 1):
        expr = f"({expr}) * {{step}}"
        kwargs["step"] = step_ast

    if not _is_constant(start_val, 0):
        expr = f"{{start}} + {expr}"
        kwargs["start"] = start_ast

    if fake_value.dtype != torch.int32:
        expr = f"({expr}).to({triton_type(fake_value.dtype)})"

    return expr_from_string(expr, **kwargs)


@_decorators.ref(arange)
def _(
    start: int,
    end: int | None = None,
    step: int = 1,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    env = CompileEnvironment.current()
    start_norm, end_norm, step_norm = _normalize_arange_bounds(start, end, step)
    if dtype is None:
        dtype = env.settings.index_dtype
    return torch.arange(
        int(start_norm),
        int(end_norm),
        int(step_norm),
        dtype=dtype,
        device=env.device if device is None else device,
    )
