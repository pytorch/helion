from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch
from torch._subclasses.fake_tensor import FakeTensor

from .. import exc
from .._compat import min_dot_size
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.dtype_utils import cast_ast
from .._compiler.dtype_utils import emit_tl_dot
from .._compiler.dtype_utils import promote_and_cast_pair
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

    # Apply min-dot-size constraints so autotuner won't pick invalid block_size
    enforce_dot_requirements(mat1, mat2)

    return (mat1, mat2, acc)


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

    lhs_dtype = lhs_proxy.dtype
    rhs_dtype = rhs_proxy.dtype
    acc_dtype: torch.dtype | None = None
    if acc_proxy is not None:
        assert isinstance(acc_proxy, FakeTensor), "acc_proxy must be a FakeTensor"
        acc_dtype = acc_proxy.dtype

    # Check if accumulator is None
    is_acc_none = isinstance(acc_ast, ast.Constant) and acc_ast.value is None

    # Harmonize operand dtypes using promotion
    lhs_casted, rhs_casted, common = promote_and_cast_pair(
        lhs_ast, rhs_ast, lhs_dtype, rhs_dtype
    )
    prec = CompileEnvironment.current().settings.dot_precision

    # Check if padding is needed for small dimensions
    from .._compiler.ast_extension import expr_from_string as _expr

    min_m, min_n, min_k = min_dot_size(lhs_proxy.device, lhs_dtype, rhs_dtype)
    m, k, n = lhs_proxy.shape[-2], lhs_proxy.shape[-1], rhs_proxy.shape[-1]
    m_val = int(m) if isinstance(m, int) else None
    n_val = int(n) if isinstance(n, int) else None
    k_val = int(k) if isinstance(k, int) else None

    needs_pad = (
        (m_val and m_val < min_m)
        or (n_val and n_val < min_n)
        or (k_val and k_val < min_k)
    )

    # Helper to create dimension masks
    def make_mask(dim: int, val: int, pos: str) -> ast.AST:
        offs = _expr(f"tl.arange(0, {dim})")
        return _expr(f"{{o}}{pos} < {val}", o=offs)

    # Helper to pad tensor
    def pad_tensor(tensor: ast.AST, shape: str, mask: ast.AST) -> ast.AST:
        zeros = _expr(f"tl.zeros([{shape}], dtype={{t}}.dtype)", t=tensor)
        return _expr(
            f"tl.where({{m}}, tl.broadcast_to({{t}}, [{shape}]), {{z}})",
            m=mask,
            t=tensor,
            z=zeros,
        )

    # Initialize padding dimensions (will be set if padding is needed)
    pad_m = min_m
    pad_n = min_n
    pad_k = min_k

    if needs_pad:
        # Apply padding logic for small matrices
        pad_m = min_m if m_val and m_val < min_m else (m_val or min_m)
        pad_n = min_n if n_val and n_val < min_n else (n_val or min_n)
        pad_k = min_k if k_val and k_val < min_k else (k_val or min_k)

        # Pad LHS [m,k]
        lhs_padded = lhs_casted
        if (m_val and m_val < min_m) or (k_val and k_val < min_k):
            masks = []
            if m_val and m_val < min_m:
                masks.append(make_mask(pad_m, m_val, "[:, None]"))
            if k_val and k_val < min_k:
                masks.append(make_mask(pad_k, k_val, "[None, :]"))
            mask = (
                masks[0]
                if len(masks) == 1
                else _expr(
                    " & ".join(f"{{m{i}}}" for i in range(len(masks))),
                    **{f"m{i}": m for i, m in enumerate(masks)},
                )
            )
            lhs_padded = pad_tensor(lhs_casted, f"{pad_m}, {pad_k}", mask)

        # Pad RHS [k,n]
        rhs_padded = rhs_casted
        if (k_val and k_val < min_k) or (n_val and n_val < min_n):
            masks = []
            if k_val and k_val < min_k:
                masks.append(make_mask(pad_k, k_val, "[:, None]"))
            if n_val and n_val < min_n:
                masks.append(make_mask(pad_n, n_val, "[None, :]"))
            mask = (
                masks[0]
                if len(masks) == 1
                else _expr(
                    " & ".join(f"{{m{i}}}" for i in range(len(masks))),
                    **{f"m{i}": m for i, m in enumerate(masks)},
                )
            )
            rhs_padded = pad_tensor(rhs_casted, f"{pad_k}, {pad_n}", mask)

        # Pad accumulator [m,n] if not None
        acc_padded = None
        if not is_acc_none and ((m_val and m_val < min_m) or (n_val and n_val < min_n)):
            masks = []
            if m_val and m_val < min_m:
                masks.append(make_mask(pad_m, m_val, "[:, None]"))
            if n_val and n_val < min_n:
                masks.append(make_mask(pad_n, n_val, "[None, :]"))
            mask = (
                masks[0]
                if len(masks) == 1
                else _expr(
                    " & ".join(f"{{m{i}}}" for i in range(len(masks))),
                    **{f"m{i}": m for i, m in enumerate(masks)},
                )
            )
            acc_padded = pad_tensor(acc_ast, f"{pad_m}, {pad_n}", mask)
        elif not is_acc_none:
            acc_padded = acc_ast

        # Use padded tensors for tl.dot execution
        lhs_to_use = lhs_padded
        rhs_to_use = rhs_padded
        acc_to_use = acc_padded
    else:
        # Use original casted tensors for tl.dot execution
        lhs_to_use = lhs_casted
        rhs_to_use = rhs_casted
        acc_to_use = acc_ast

    # Common tl.dot execution logic for both padded and standard paths
    if is_acc_none:
        out_dtype = _compute_out_dtype(lhs_dtype, rhs_dtype)
        result = emit_tl_dot(
            lhs_to_use, rhs_to_use, input_precision=prec, out_dtype=out_dtype
        )
    else:
        assert acc_dtype is not None
        if acc_dtype == common:
            # Triton requires out_dtype=fp16 to fuse acc when compute is fp16
            if common == torch.float16:
                result = emit_tl_dot(
                    lhs_to_use,
                    rhs_to_use,
                    input_precision=prec,
                    acc=acc_to_use,
                    out_dtype=torch.float16,
                )
            else:
                result = emit_tl_dot(
                    lhs_to_use, rhs_to_use, input_precision=prec, acc=acc_to_use
                )
        else:
            # Compute in input-promoted dtype, add to acc separately
            mm = emit_tl_dot(lhs_to_use, rhs_to_use, input_precision=prec)
            mm_cast = cast_ast(mm, acc_dtype)
            assert acc_to_use is not None, (
                "acc_to_use must not be None when accumulator is used"
            )
            result = _expr("{acc} + {mm}", acc=acc_to_use, mm=mm_cast)

    # Extract valid region when dimensions were padded
    if needs_pad and ((m_val and m_val < min_m) or (n_val and n_val < min_n)):
        # Create masks for the valid region
        masks = []
        if m_val and m_val < min_m:
            masks.append(make_mask(pad_m, m_val, "[:, None]"))
        if n_val and n_val < min_n:
            masks.append(make_mask(pad_n, n_val, "[None, :]"))
        mask = (
            masks[0]
            if len(masks) == 1
            else _expr(
                " & ".join(f"{{m{i}}}" for i in range(len(masks))),
                **{f"m{i}": m for i, m in enumerate(masks)},
            )
        )
        # Extract only the valid region from the padded result
        zeros = _expr("tl.zeros_like({r})", r=result)
        result = _expr("tl.where({m}, {r}, {z})", m=mask, r=result, z=zeros)

    return result


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
