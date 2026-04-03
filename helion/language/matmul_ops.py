from __future__ import annotations

import ast
from itertools import zip_longest
from typing import TYPE_CHECKING

import torch
from torch._subclasses.fake_tensor import FakeTensor

from .. import exc
from .._compat import min_dot_size
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.compile_environment import format_shape
from .._compiler.cute.indexing import CutePackedAffineLoad
from .._compiler.cute.indexing import CutePackedTerms
from .._compiler.cute.matmul_fallback import _emit_cute_matmul
from .._compiler.cute.matmul_utils import cute_lower_rhs_for_matmul
from .._compiler.cute.matmul_utils import cute_outer_accumulates_result
from .._compiler.cute.matmul_utils import cute_outer_accumulator_dtype
from .._compiler.cute.matmul_utils import cute_outer_accumulator_out_dtype
from .._compiler.cute.matmul_utils import cute_resolve_active_block_id
from .._compiler.cute.matmul_utils import cute_resolve_active_matmul_k_block_id
from .._compiler.cute.matmul_utils import cute_static_k_invariant_extent
from .._compiler.matmul_utils import _compute_out_dtype
from .._compiler.matmul_utils import _emit_pallas_matmul
from .._compiler.matmul_utils import _emit_tl_dot_scaled
from .._compiler.matmul_utils import _needs_f32_accumulator
from .._compiler.matmul_utils import emit_tl_dot_with_padding
from . import _decorators

if TYPE_CHECKING:
    from .._compiler.inductor_lowering import CodegenState


def _cute_dot_outer_accumulates_result(fx_node: object, *, is_acc_none: bool) -> bool:
    if not isinstance(fx_node, torch.fx.Node):
        fx_node = getattr(fx_node, "fx_node", fx_node)
    if not isinstance(fx_node, torch.fx.Node):
        fx_node = None
    return cute_outer_accumulates_result(fx_node, is_acc_none=is_acc_none)


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


def _cute_mma_matches_dot_semantics(
    lhs_dtype: torch.dtype,
    rhs_dtype: torch.dtype,
    acc_dtype: torch.dtype | None,
    out_dtype: torch.dtype | None,
) -> bool:
    """Return True when fixed-f32 MMA accumulation matches hl.dot semantics."""
    if not _needs_f32_accumulator(lhs_dtype, rhs_dtype):
        return True
    return out_dtype in (None, torch.float32) and acc_dtype in (None, torch.float32)


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
    """Update config-spec min/max sizes for a dot/matmul.

    This ensures the autotuner does not select block sizes below the hardware
    minimums for the current device and dtypes, and constrains the batch
    dimension block size to 1 for 3D operands since Triton does not support
    3D dot operations.
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

    # Triton only supports 2D dot operations.  When the operands are 3D
    # (batched matmul), constrain the batch dimension block size to 1 so
    # the codegen can squeeze it away before emitting tl.dot.
    # Pallas uses jnp.matmul which handles batched matmul natively.
    if len(lshape) == 3 and env.backend_name != "pallas":
        for batch_dim in (lshape[0], rshape[0]):
            block_idx = env.get_block_id(batch_dim)
            if block_idx is not None:
                env.block_sizes[block_idx].update_max_block(1)


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


@_decorators.codegen(dot, "triton")
def _(state: CodegenState) -> object:
    # Get the AST representations of our arguments
    lhs_ast = state.ast_arg(0)
    rhs_ast = state.ast_arg(1)
    acc_ast = state.ast_arg(2)
    assert isinstance(lhs_ast, (ast.AST, CutePackedAffineLoad))
    assert isinstance(rhs_ast, ast.AST)

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


@_decorators.codegen(dot, "cute")
def _(state: CodegenState) -> object:
    lhs_proxy = state.proxy_args[0]
    assert isinstance(lhs_proxy, FakeTensor)
    rhs_proxy = state.proxy_args[1]
    assert isinstance(rhs_proxy, FakeTensor)
    acc_proxy = state.proxy_args[2] if len(state.proxy_args) > 2 else None
    out_dtype_proxy = state.proxy_args[3] if len(state.proxy_args) > 3 else None

    lhs_ast = state.ast_args[0]
    if isinstance(lhs_ast, int | float | bool | None):
        lhs_ast = ast.Constant(value=lhs_ast)
    rhs_ast = state.ast_arg(1)
    acc_ast = state.ast_arg(2)
    assert isinstance(lhs_ast, (ast.AST, CutePackedAffineLoad))

    is_acc_none = isinstance(acc_ast, ast.Constant) and acc_ast.value is None

    acc_dtype: torch.dtype | None = None
    if not is_acc_none:
        assert isinstance(acc_proxy, FakeTensor)
        acc_dtype = acc_proxy.dtype
        if lhs_proxy.dtype == torch.float32 and rhs_proxy.dtype == torch.float32:
            if acc_dtype == torch.float16:
                raise exc.BackendUnsupported(
                    "cute",
                    "hl.dot(float32, float32, acc=float16) is not supported on CuTe; use a float32 accumulator or cast after the dot",
                )

    out_dtype: torch.dtype | None = None
    if out_dtype_proxy is not None:
        assert isinstance(out_dtype_proxy, torch.dtype)
        out_dtype = out_dtype_proxy

    # Try MMA path first for configurations whose dtype semantics match fp32 MMA.
    if _cute_mma_matches_dot_semantics(
        lhs_proxy.dtype, rhs_proxy.dtype, acc_dtype, out_dtype
    ):
        from .._compiler.cute.cute_mma import codegen_cute_mma_dot

        result = codegen_cute_mma_dot(state)
        if result is not None:
            return result

    resolved_out_dtype = out_dtype or _compute_out_dtype(
        lhs_proxy.dtype,
        rhs_proxy.dtype,
        acc_dtype,
    )
    outer_acc_dtype = cute_outer_accumulator_dtype(
        state.fx_node,
        is_acc_none=is_acc_none,
    )
    effective_out_dtype = cute_outer_accumulator_out_dtype(
        resolved_out_dtype,
        outer_acc_dtype,
    )
    k_block_id = cute_resolve_active_matmul_k_block_id(
        state.codegen,
        lhs_proxy.shape[-1],
        rhs_proxy.shape[-2],
        rhs_proxy.shape[-1],
    )
    packed_rhs = None
    if (
        k_block_id is None
        and state.fx_node is not None
        and len(state.fx_node.args) >= 2
        and isinstance(rhs_node := state.fx_node.args[1], torch.fx.Node)
    ):
        rhs_ast, packed_rhs = cute_lower_rhs_for_matmul(
            state.env,
            lhs_ast,
            rhs_node,
            rhs_ast,
        )
    if k_block_id is None and packed_rhs is not None:
        packed_nodes, _ = packed_rhs
        packed_node = packed_nodes[0]
        k_block_id = cute_resolve_active_block_id(
            state.codegen, packed_node.meta["val"].shape[0]
        )
    assert isinstance(rhs_ast, (ast.AST, CutePackedTerms))
    static_k_extent = None
    if k_block_id is None and state.fx_node is not None:
        lhs_node = state.fx_node.args[0] if len(state.fx_node.args) > 0 else None
        rhs_node = state.fx_node.args[1] if len(state.fx_node.args) > 1 else None
        if isinstance(lhs_node, torch.fx.Node) and isinstance(rhs_node, torch.fx.Node):
            static_k_extent = cute_static_k_invariant_extent(lhs_node, rhs_node)
    env = CompileEnvironment.current()
    size_hint = getattr(env, "size_hint", None)

    def hinted(size: int | torch.SymInt) -> int:
        if callable(size_hint):
            hinted_size = size_hint(size)
            assert isinstance(hinted_size, int)
            return hinted_size
        return int(size)

    k_is_one = hinted(lhs_proxy.shape[-1]) == 1 and hinted(rhs_proxy.shape[-2]) == 1
    if static_k_extent is None and k_block_id is None and not k_is_one:
        raise exc.BackendUnsupported(
            "cute",
            "CuTe scalar matmul fallback requires an active K tile or a K-invariant static shortcut",
        )
    return _emit_cute_matmul(
        state.codegen,
        lhs_ast,
        rhs_ast,
        accumulate_in_lane_loop=not cute_outer_accumulates_result(
            state.fx_node,
            is_acc_none=is_acc_none,
        ),
        k_block_id=k_block_id,
        static_k_extent=static_k_extent,
        acc=None if is_acc_none else acc_ast,
        out_dtype=effective_out_dtype,
        acc_dtype=acc_dtype,
        lhs_dtype=lhs_proxy.dtype,
        rhs_dtype=rhs_proxy.dtype,
    )


@_decorators.codegen(dot, "pallas")
def _(state: CodegenState) -> object:
    lhs_ast = state.ast_arg(0)
    rhs_ast = state.ast_arg(1)
    acc_ast = state.ast_arg(2)

    lhs_proxy = state.proxy_args[0]
    assert isinstance(lhs_proxy, FakeTensor)
    rhs_proxy = state.proxy_args[1]
    assert isinstance(rhs_proxy, FakeTensor)
    acc_proxy = state.proxy_args[2] if len(state.proxy_args) > 2 else None
    out_dtype_proxy = state.proxy_args[3] if len(state.proxy_args) > 3 else None

    lhs_dtype = lhs_proxy.dtype
    rhs_dtype = rhs_proxy.dtype
    need_f32_acc = _needs_f32_accumulator(lhs_dtype, rhs_dtype)

    # Determine the accumulator AST (None if acc argument is None)
    is_acc_none = isinstance(acc_ast, ast.Constant) and acc_ast.value is None
    acc = None if is_acc_none else acc_ast

    # Determine desired output dtype
    out_dtype: torch.dtype | None = None
    if out_dtype_proxy is not None:
        assert isinstance(out_dtype_proxy, torch.dtype)
        out_dtype = out_dtype_proxy
    elif acc_proxy is not None and isinstance(acc_proxy, FakeTensor):
        out_dtype = acc_proxy.dtype

    return _emit_pallas_matmul(
        lhs_ast,
        rhs_ast,
        acc=acc,
        need_f32_acc=need_f32_acc,
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


VALID_SCALED_FORMATS = frozenset({"e2m1", "e4m3", "e5m2", "bf16", "fp16"})


@_decorators.api(is_device_only=True)
def dot_scaled(
    mat1: torch.Tensor,
    mat1_scale: torch.Tensor,
    mat1_format: str,
    mat2: torch.Tensor,
    mat2_scale: torch.Tensor,
    mat2_format: str,
    acc: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Performs a block-scaled matrix multiplication using Triton's tl.dot_scaled.

    This operation performs matrix multiplication with block-scaled inputs in formats
    such as FP4 (e2m1), FP8 (e4m3, e5m2), BF16, and FP16. Each input tensor has an
    associated scale factor tensor and format string.

    Args:
        mat1: First matrix (2D tensor of packed data)
        mat1_scale: Scale factors for mat1 (2D tensor)
        mat1_format: Format string for mat1 (one of "e2m1", "e4m3", "e5m2", "bf16", "fp16")
        mat2: Second matrix (2D tensor of packed data)
        mat2_scale: Scale factors for mat2 (2D tensor)
        mat2_format: Format string for mat2 (one of "e2m1", "e4m3", "e5m2", "bf16", "fp16")
        acc: Optional accumulator tensor (2D, float32 or float16)
        out_dtype: Optional output dtype for the multiplication

    Returns:
        Result of block-scaled matrix multiplication.
    """
    raise exc.NotInsideKernel


@_decorators.prepare_args(dot_scaled)
def _(
    mat1: torch.Tensor,
    mat1_scale: torch.Tensor,
    mat1_format: str,
    mat2: torch.Tensor,
    mat2_scale: torch.Tensor,
    mat2_format: str,
    acc: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    str,
    torch.Tensor,
    torch.Tensor,
    str,
    torch.Tensor | None,
    torch.dtype | None,
]:
    if mat1_format not in VALID_SCALED_FORMATS:
        raise ValueError(
            f"hl.dot_scaled: mat1_format must be one of {sorted(VALID_SCALED_FORMATS)}, "
            f"got '{mat1_format}'"
        )
    if mat2_format not in VALID_SCALED_FORMATS:
        raise ValueError(
            f"hl.dot_scaled: mat2_format must be one of {sorted(VALID_SCALED_FORMATS)}, "
            f"got '{mat2_format}'"
        )

    if mat1.ndim != 2:
        raise ValueError(f"hl.dot_scaled: mat1 must be a 2D tensor, got {mat1.ndim}D")
    if mat2.ndim != 2:
        raise ValueError(f"hl.dot_scaled: mat2 must be a 2D tensor, got {mat2.ndim}D")

    if mat1_scale.ndim != 2:
        raise ValueError(
            f"hl.dot_scaled: mat1_scale must be a 2D tensor, got {mat1_scale.ndim}D"
        )
    if mat2_scale.ndim != 2:
        raise ValueError(
            f"hl.dot_scaled: mat2_scale must be a 2D tensor, got {mat2_scale.ndim}D"
        )

    if acc is not None:
        expected_shape = [mat1.shape[0], mat2.shape[-1]]
        if acc.ndim != 2:
            raise ValueError(f"hl.dot_scaled: acc must be a 2D tensor, got {acc.ndim}D")
        if list(acc.shape) != expected_shape:
            raise ValueError(
                f"hl.dot_scaled: acc shape {list(acc.shape)} incompatible with "
                f"result shape {expected_shape}"
            )
        valid_acc_dtypes = (torch.float16, torch.float32)
        if acc.dtype not in valid_acc_dtypes:
            raise TypeError(
                f"hl.dot_scaled: acc must be one of {[str(d) for d in valid_acc_dtypes]}, "
                f"got {acc.dtype}"
            )

    if out_dtype is not None and not isinstance(out_dtype, torch.dtype):
        raise TypeError(
            f"hl.dot_scaled: out_dtype must be a torch.dtype or None, got {type(out_dtype)}"
        )

    # Enforce minimum block sizes so autotuner picks valid configs.
    enforce_dot_requirements(mat1, mat2)
    # K must be >= 32 because scale tensors have shape [dim, K // 32].
    env = CompileEnvironment.current()
    k_dim = mat1.shape[-1]
    k_block_idx = env.get_block_id(k_dim)
    if k_block_idx is not None:
        env.block_sizes[k_block_idx].update_min_block(32, allow_flattened=True)

    return (
        mat1,
        mat1_scale,
        mat1_format,
        mat2,
        mat2_scale,
        mat2_format,
        acc,
        out_dtype,
    )


@_decorators.register_fake(dot_scaled)
def _(
    mat1: torch.Tensor,
    mat1_scale: torch.Tensor,
    mat1_format: str,
    mat2: torch.Tensor,
    mat2_scale: torch.Tensor,
    mat2_format: str,
    acc: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    result_shape = [mat1.shape[0], mat2.shape[-1]]
    if acc is not None:
        return acc.new_empty(result_shape)
    resolved_dtype = out_dtype or torch.float32
    return torch.empty(result_shape, dtype=resolved_dtype, device=mat1.device)


@_decorators.codegen(dot_scaled, "triton")
def _(state: CodegenState) -> object:
    lhs_ast = state.ast_arg(0)  # mat1
    lhs_scale_ast = state.ast_arg(1)  # mat1_scale
    lhs_format = state.proxy_args[2]  # "e2m1" etc (string, not AST)
    assert isinstance(lhs_format, str), "lhs_format must be a string"
    rhs_ast = state.ast_arg(3)  # mat2
    rhs_scale_ast = state.ast_arg(4)  # mat2_scale
    rhs_format = state.proxy_args[5]  # "e2m1" etc (string, not AST)
    assert isinstance(rhs_format, str), "rhs_format must be a string"
    acc_ast = state.ast_arg(6)  # acc
    out_dtype_proxy = state.proxy_args[7] if len(state.proxy_args) > 7 else None

    out_dtype: torch.dtype | None = None
    if out_dtype_proxy is not None:
        assert isinstance(out_dtype_proxy, torch.dtype), (
            "out_dtype must be a torch.dtype"
        )
        out_dtype = out_dtype_proxy

    is_acc_none = isinstance(acc_ast, ast.Constant) and acc_ast.value is None
    return _emit_tl_dot_scaled(
        lhs_ast,
        lhs_scale_ast,
        lhs_format,
        rhs_ast,
        rhs_scale_ast,
        rhs_format,
        acc=None if is_acc_none else acc_ast,
        out_dtype=out_dtype,
    )


@_decorators.ref(dot_scaled)
def _(
    mat1: torch.Tensor,
    mat1_scale: torch.Tensor,
    mat1_format: str,
    mat2: torch.Tensor,
    mat2_scale: torch.Tensor,
    mat2_format: str,
    acc: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    def _dequant(data: torch.Tensor, scale: torch.Tensor, fmt: str) -> torch.Tensor:
        data_f32 = data.to(torch.float32)
        # Scale is in e8m0 format (uint8): value = 2^(byte - 127)
        # e.g. byte=127 means 2^0=1.0, byte=0 means 2^(-127), byte=254 means 2^127
        scale_f32 = torch.pow(2.0, scale.to(torch.float32) - 127.0)
        k_data = data_f32.shape[-1]
        k_scale = scale_f32.shape[-1]
        if k_scale < k_data:
            repeat_factor = k_data // k_scale
            scale_f32 = scale_f32.repeat_interleave(repeat_factor, dim=-1)
        return data_f32 * scale_f32

    mat1_dequant = _dequant(mat1, mat1_scale, mat1_format)
    mat2_dequant = _dequant(mat2, mat2_scale, mat2_format)

    result = torch.mm(mat1_dequant, mat2_dequant)
    resolved_dtype = out_dtype or torch.float32
    result = result.to(resolved_dtype)

    if acc is not None:
        return acc + result
    return result
