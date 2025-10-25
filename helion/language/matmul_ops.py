from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch
from torch._subclasses.fake_tensor import FakeTensor

from .. import exc
from .._compat import min_dot_size
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.compile_environment import format_shape
from .._compiler.matmul_utils import _compute_out_dtype
from .._compiler.matmul_utils import emit_tl_dot_with_padding
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState


@_decorators.api(is_device_only=True)
def dot(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    acc: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
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
        out_dtype: Optional dtype that controls the output type of the multiplication prior
            to any accumulation. This maps directly to the Triton ``tl.dot`` ``out_dtype``
            argument and overrides the default promotion rules when provided.

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
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.dtype | None]:
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

    # Check batch dimension compatibility (broadcastable or matching) if any input is 3D
    if mat1.ndim == 3 or mat2.ndim == 3:
        from itertools import zip_longest

        batch_shape_1 = mat1.shape[:-2] if mat1.ndim > 2 else ()
        batch_shape_2 = mat2.shape[:-2] if mat2.ndim > 2 else ()

        for lhs_dim, rhs_dim in zip_longest(
            reversed(batch_shape_1), reversed(batch_shape_2), fillvalue=1
        ):
            # Allow broadcasting with 1
            if str(lhs_dim) == "1" or str(rhs_dim) == "1":
                continue
            # Check if dimensions match
            if str(lhs_dim) != str(rhs_dim):
                raise exc.DotBatchDimensionMismatch(
                    lhs=format_shape(batch_shape_1),
                    rhs=format_shape(batch_shape_2),
                )

    if out_dtype is not None and not isinstance(out_dtype, torch.dtype):
        raise TypeError(
            f"hl.dot: out_dtype must be a torch.dtype or None, got {type(out_dtype)}"
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

    # Apply min-dot-size constraints so autotuner won't pick invalid block_size
    enforce_dot_requirements(mat1, mat2)

    return (mat1, mat2, acc, out_dtype)


def enforce_dot_requirements(lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    """Update config-spec min sizes for M, N, K of a dot/matmul.

    This ensures the autotuner does not select block sizes below the hardware
    minimums for the current device and dtypes.
    """

    # Last two dims are used for matmul
    lshape = lhs.size()
    rshape = rhs.size()
    m, k = lshape[-2], lshape[-1]
    k2, n = rshape[-2], rshape[-1]
    assert k == k2, f"Mismatched K dimensions for dot: {k} vs {k2}"

    a, b, c = min_dot_size(lhs.device, lhs.dtype, rhs.dtype)
    env = CompileEnvironment.current()
    for shape, min_size in ((m, a), (n, b), (k, c)):
        block_idx = env.get_block_id(shape)
        if block_idx is not None:
            env.block_sizes[block_idx].update_min_block(min_size, allow_flattened=True)


@_decorators.register_fake(dot)
def _(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    acc: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    # Matrix multiplication shape computation
    result_shape = list(mat1.shape)
    result_shape[-1] = mat2.shape[-1]

    if acc is not None:
        return acc.new_empty(result_shape)

    # Determine output dtype using the helper function
    resolved_out_dtype = out_dtype or _compute_out_dtype(mat1.dtype, mat2.dtype)
    return torch.empty(result_shape, dtype=resolved_out_dtype, device=mat1.device)


@_decorators.codegen(dot)
def _(state: CodegenState) -> object:
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
    out_dtype_proxy = state.proxy_args[3] if len(state.proxy_args) > 3 else None

    lhs_dtype = lhs_proxy.dtype
    rhs_dtype = rhs_proxy.dtype
    acc_dtype: torch.dtype | None = None
    if acc_proxy is not None:
        assert isinstance(acc_proxy, FakeTensor), "acc_proxy must be a FakeTensor"
        acc_dtype = acc_proxy.dtype

    out_dtype: torch.dtype | None = None
    if out_dtype_proxy is not None:
        assert isinstance(out_dtype_proxy, torch.dtype), (
            "out_dtype must be a torch.dtype"
        )
        out_dtype = out_dtype_proxy

    # Check if accumulator is None
    is_acc_none = isinstance(acc_ast, ast.Constant) and acc_ast.value is None

    lhs_shape: list[int | torch.SymInt] = list(lhs_proxy.shape)
    rhs_shape: list[int | torch.SymInt] = list(rhs_proxy.shape)
    acc_shape: list[int | torch.SymInt] | None = (
        list(acc_proxy.shape) if acc_proxy is not None else None
    )
    acc_arg = None if is_acc_none else acc_ast
    acc_dtype_arg = acc_dtype if not is_acc_none else None

    # Perform dot with optional padding
    return emit_tl_dot_with_padding(
        lhs_ast,
        rhs_ast,
        acc_arg,
        lhs_dtype,
        rhs_dtype,
        acc_dtype=acc_dtype_arg,
        lhs_shape=lhs_shape,
        rhs_shape=rhs_shape,
        acc_shape=acc_shape,
        out_dtype=out_dtype,
    )


@_decorators.ref(dot)
def _(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    acc: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    resolved_out_dtype = out_dtype or _compute_out_dtype(
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
            out_dtype=resolved_out_dtype,
        )
    else:
        # For non-FP8 tensors, use regular matmul
        result = torch.mm(mat1, mat2, out_dtype=resolved_out_dtype)

    if acc is not None:
        return acc + result
    return result
