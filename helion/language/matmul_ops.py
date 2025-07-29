from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch
from torch._inductor.utils import triton_type
from torch._subclasses.fake_tensor import FakeTensor

from .. import exc
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState


@_decorators.api(is_device_only=True)
def dot(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    acc: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Performs a matrix multiplication of tensors with support for multiple dtypes.

    This operation performs matrix multiplication with inputs of various dtypes including
    float16, bfloat16, float32, int8, and FP8 formats (e4m3fn, e5m2). The computation is
    performed with appropriate precision based on the input dtypes.

    Args:
        mat1: First matrix (2D or 3D tensor of torch.float16, torch.bfloat16, torch.float32, torch.int8, torch.float8_e4m3fn, or torch.float8_e5m2)
        mat2: Second matrix (2D or 3D tensor of torch.float16, torch.bfloat16, torch.float32, torch.int8, torch.float8_e4m3fn, or torch.float8_e5m2)
        acc: The accumulator tensor (2D or 3D tensor of torch.float16, torch.float32, or torch.int32).
             If not None, the result is added to this tensor.
             If None, a new tensor is created with appropriate dtype based on inputs.

    Returns:
        Result of matrix multiplication. If acc is provided, returns acc + (mat1 @ mat2).
        Otherwise returns (mat1 @ mat2) with promoted dtype.

    Example:
        >>> # FP8 example
        >>> a = torch.randn(32, 64, device="cuda").to(torch.float8_e4m3fn)
        >>> b = torch.randn(64, 128, device="cuda").to(torch.float8_e4m3fn)
        >>> c = torch.zeros(32, 128, device="cuda", dtype=torch.float32)
        >>> result = hl.dot(a, b, acc=c)  # result is c + (a @ b)

        >>> # Float16 example
        >>> a = torch.randn(32, 64, device="cuda", dtype=torch.float16)
        >>> b = torch.randn(64, 128, device="cuda", dtype=torch.float16)
        >>> result = hl.dot(a, b)  # result dtype will be torch.float16

        >>> # Int8 example
        >>> a = torch.randint(-128, 127, (32, 64), device="cuda", dtype=torch.int8)
        >>> b = torch.randint(-128, 127, (64, 128), device="cuda", dtype=torch.int8)
        >>> acc = torch.zeros(32, 128, device="cuda", dtype=torch.int32)
        >>> result = hl.dot(a, b, acc=acc)  # int8 x int8 -> int32
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(dot)
def _(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    acc: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # Define supported dtypes
    supported_dtypes = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.int8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )

    # Validate input types
    if mat1.dtype not in supported_dtypes:
        raise TypeError(
            f"hl.dot: mat1 must be one of {[str(d) for d in supported_dtypes]}, got {mat1.dtype}"
        )
    if mat2.dtype not in supported_dtypes:
        raise TypeError(
            f"hl.dot: mat2 must be one of {[str(d) for d in supported_dtypes]}, got {mat2.dtype}"
        )

    # Validate shapes for matrix multiplication
    if mat1.ndim not in (2, 3):
        raise ValueError(f"hl.dot: mat1 must be 2D or 3D tensor, got {mat1.ndim}D")
    if mat2.ndim not in (2, 3):
        raise ValueError(f"hl.dot: mat2 must be 2D or 3D tensor, got {mat2.ndim}D")

    # Check matrix multiplication compatibility
    if mat1.shape[-1] != mat2.shape[-2]:
        raise ValueError(
            f"hl.dot: incompatible matrix dimensions for multiplication: "
            f"{mat1.shape} @ {mat2.shape}"
        )

    # Validate accumulator if provided
    if acc is not None:
        # Allow int32 accumulator for int8 inputs
        valid_acc_dtypes = (torch.float16, torch.float32, torch.int32)
        if acc.dtype not in valid_acc_dtypes:
            raise TypeError(
                f"hl.dot: acc must be one of {[str(d) for d in valid_acc_dtypes]}, got {acc.dtype}"
            )

        # Check int8 inputs require int32 accumulator
        if mat1.dtype == torch.int8 or mat2.dtype == torch.int8:
            if acc.dtype != torch.int32:
                raise TypeError(
                    f"hl.dot: int8 inputs require int32 accumulator, got {acc.dtype}"
                )

        # Check accumulator shape compatibility
        expected_shape = list(mat1.shape)
        expected_shape[-1] = mat2.shape[-1]

        if acc.ndim not in (2, 3):
            raise ValueError(f"hl.dot: acc must be 2D or 3D tensor, got {acc.ndim}D")

        if list(acc.shape) != expected_shape:
            raise ValueError(
                f"hl.dot: acc shape {list(acc.shape)} incompatible with result shape {expected_shape}"
            )

    return (mat1, mat2, acc)


def _compute_out_dtype(
    mat1_dtype: torch.dtype,
    mat2_dtype: torch.dtype,
    acc_dtype: torch.dtype | None = None,
) -> torch.dtype:
    """Compute the output dtype for dot operation."""
    if acc_dtype is not None:
        # If accumulator is provided, use its dtype
        return acc_dtype

    # When no accumulator is specified:
    # For int8 inputs, default to int32
    if mat1_dtype == torch.int8 or mat2_dtype == torch.int8:
        return torch.int32
    # For all other inputs (including FP8), default to float32
    return torch.float32


@_decorators.register_fake(dot)
def _(
    mat1: torch.Tensor, mat2: torch.Tensor, acc: torch.Tensor | None = None
) -> torch.Tensor:
    # Matrix multiplication shape computation
    result_shape = list(mat1.shape)
    result_shape[-1] = mat2.shape[-1]

    if acc is not None:
        return acc.new_empty(result_shape)

    # Determine output dtype using the helper function
    out_dtype = _compute_out_dtype(mat1.dtype, mat2.dtype)
    return torch.empty(result_shape, dtype=out_dtype, device=mat1.device)


@_decorators.codegen(dot)
def _(state: CodegenState) -> object:
    # Import here to avoid circular imports
    from .._compiler.ast_extension import expr_from_string
    from .._compiler.compile_environment import CompileEnvironment

    # Get the AST representations of our arguments
    lhs_ast = state.ast_arg(0)
    rhs_ast = state.ast_arg(1)
    acc_ast = state.ast_arg(2)

    # Get the dtypes of the inputs from proxy args
    lhs_proxy = state.proxy_args[0]
    assert isinstance(lhs_proxy, FakeTensor), "lhs_proxy must be a FakeTensor"
    rhs_proxy = state.proxy_args[1]
    assert isinstance(rhs_proxy, FakeTensor), "rhs_proxy must be a FakeTensor"
    acc_proxy = state.proxy_args[2] if len(state.proxy_args) > 2 else None

    # Access dtype - proxy_args can be FakeTensor objects
    lhs_dtype = None
    rhs_dtype = None
    acc_dtype = None

    # For FakeTensor objects, dtype is directly accessible
    lhs_dtype = lhs_proxy.dtype
    rhs_dtype = rhs_proxy.dtype

    # Get accumulator dtype if available
    if acc_proxy is not None:
        assert isinstance(acc_proxy, FakeTensor), "acc_proxy must be a FakeTensor"
        acc_dtype = acc_proxy.dtype

    # Check if accumulator is None
    is_acc_none = isinstance(acc_ast, ast.Constant) and acc_ast.value is None

    # Determine output dtype using the helper function
    out_dtype = _compute_out_dtype(
        lhs_dtype, rhs_dtype, None if is_acc_none else acc_dtype
    )

    return expr_from_string(
        f"tl.dot(lhs, rhs, acc=acc, input_precision='{CompileEnvironment.current().settings.dot_precision}', out_dtype={triton_type(out_dtype)})",
        lhs=lhs_ast,
        rhs=rhs_ast,
        acc=acc_ast,
    )


@_decorators.ref(dot)
def _(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    acc: torch.Tensor | None = None,
) -> torch.Tensor:
    out_dtype = _compute_out_dtype(
        mat1.dtype, mat2.dtype, None if acc is None else acc.dtype
    )

    is_fp8 = mat1.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) or mat2.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if is_fp8:
        # Use torch._scaled_mm for FP8 operations
        # Ensure column-major for second operand as required by torch._scaled_mm
        mat2_t = mat2.T.contiguous().T
        scale_a = torch.tensor(1.0, device=mat1.device)
        scale_b = torch.tensor(1.0, device=mat2.device)

        result = torch._scaled_mm(
            mat1,
            mat2_t,
            scale_a,
            scale_b,
            use_fast_accum=False,
            out_dtype=out_dtype,
        )
    else:
        # For non-FP8 tensors, use regular matmul
        result = torch.mm(mat1, mat2, out_dtype=out_dtype)

    if acc is not None:
        return acc + result
    return result
