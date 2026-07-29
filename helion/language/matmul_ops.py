from __future__ import annotations

import dataclasses
from itertools import zip_longest

import torch

from .. import exc
from .._compat import min_dot_size
from .._compiler.compile_environment import CompileEnvironment
from .._compiler.compile_environment import _to_sympy
from .._compiler.compile_environment import format_shape
from .._compiler.compile_environment import shape_env_var_hints
from .._compiler.cute.matmul_utils import cute_outer_accumulates_result
from .._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_M
from .._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_BLOCK_N
from .._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_EDGE_K_TAIL_MIN_DIM
from .._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
from .._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N
from .._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
from .._compiler.matmul_utils import _compute_out_dtype
from ..autotuner.config_spec import MatmulFact
from . import _decorators


def _static_dim_value(env: CompileEnvironment, size: int | torch.SymInt) -> int | None:
    if isinstance(size, int):
        return size
    expr = _to_sympy(size)
    expr = env.specialize_expr(env.shape_env.replace(expr))
    if expr.free_symbols:
        if not env.settings.static_shapes:
            return None
        expr = expr.xreplace(shape_env_var_hints(env.shape_env))
    if expr.free_symbols:
        return None
    return int(expr)


def _dot_result_shape(
    mat1: torch.Tensor, mat2: torch.Tensor
) -> list[int | torch.SymInt]:
    """Return the broadcasted ``[..., M, N]`` result shape for ``hl.dot``."""
    batch_shape = torch.broadcast_shapes(mat1.shape[:-2], mat2.shape[:-2])
    return [*batch_shape, mat1.shape[-2], mat2.shape[-1]]


def _cute_dot_outer_accumulates_result(fx_node: object, *, is_acc_none: bool) -> bool:
    if not isinstance(fx_node, torch.fx.Node):
        fx_node = getattr(fx_node, "fx_node", fx_node)
    if not isinstance(fx_node, torch.fx.Node):
        fx_node = None
    return cute_outer_accumulates_result(fx_node, is_acc_none=is_acc_none)


def _cuda_num_sms_or_zero(device: torch.device) -> int:
    """Return the device SM count, or 0 on devices ``get_num_sm`` does not support.

    Used by the cluster_m=2 small-shape wave-quantization gate in
    ``enforce_dot_requirements`` (cute_plan.md §7.6.3.2). The 0 fallback
    keeps cluster_m=2 search live for configuration round-trip tests
    that bind on CPU or other unsupported device types.
    """
    if device.type != "cuda":
        return 0
    # Local import: ``helion.runtime`` is in the import chain that loads
    # this module, so a top-level import would be circular.
    from ..runtime import get_num_sm

    try:
        return get_num_sm(device)
    except (AssertionError, NotImplementedError):
        return 0


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
        expected_shape = _dot_result_shape(mat1, mat2)

        if acc.ndim not in (2, 3):
            raise ValueError(f"hl.dot: acc must be 2D or 3D tensor, got {acc.ndim}D")

        if list(acc.shape) != expected_shape:
            raise ValueError(
                f"hl.dot: acc shape {list(acc.shape)} incompatible with result shape {expected_shape}"
            )

    # Apply min-dot-size constraints so autotuning cannot select an invalid tile.
    # Leading-passthrough tcgen05 search is intentionally not inferred from rank:
    # the DeviceIR post-pass opens it only after the same structural analyzer
    # used by codegen accepts the operands.
    enforce_dot_requirements(mat1, mat2)

    return (mat1, mat2, acc, out_dtype)


def _dot_dimensions(
    lhs: torch.Tensor, rhs: torch.Tensor
) -> tuple[int | torch.SymInt, int | torch.SymInt, int | torch.SymInt]:
    """Return the M, N, and K problem extents for a dot."""
    m, k = lhs.shape[-2:]
    k2, n = rhs.shape[-2:]
    assert k == k2, f"Mismatched K dimensions for dot: {k} vs {k2}"
    return m, n, k


def _static_problem_extent(
    env: CompileEnvironment, size: int | torch.SymInt
) -> int | None:
    block_idx = env.get_block_id(size)
    if block_idx is not None:
        block_size = env.block_sizes[block_idx].size
        if isinstance(block_size, (int, torch.SymInt)):
            return _static_dim_value(env, block_size)
    return _static_dim_value(env, size)


def enforce_dot_requirements(lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    """Record a matmul fact and apply backend-independent dot constraints."""
    m, n, k = _dot_dimensions(lhs, rhs)

    from ..autotuner.config_spec import SMALL_DIM_BLOCK_SIZE_OVERSHOOT

    env = CompileEnvironment.current()
    a, b, c = min_dot_size(lhs.device, lhs.dtype, rhs.dtype)
    # M and N are the output tile dims; K is the contraction loop.
    for shape, min_size, is_output_dim in (
        (m, a, True),
        (n, b, True),
        (k, c, False),
    ):
        block_idx = env.get_block_id(shape)
        if block_idx is not None:
            # On Pallas, clamp min to the tensor dimension so we don't
            # force blocks larger than the tensor (Pallas BlockSpecs can't
            # handle that, unlike Triton which masks out-of-bounds accesses).
            # The dot-level padding in matmul_utils.py will pad the smaller
            # tile up to min_dot_size at codegen time.
            if env.backend_name == "pallas":
                try:
                    bspec = env.config_spec.block_sizes.block_id_lookup(block_idx)
                    min_size = min(min_size, bspec.size_hint)
                except KeyError:
                    pass
            env.block_sizes[block_idx].update_min_block(min_size, allow_flattened=True)
            # Let the autotuner try output block sizes larger than a small matmul
            # dimension: masked rows/cols can map to a more efficient MMA tile.
            if is_output_dim and env.backend_name == "triton":
                try:
                    spec = env.config_spec.block_sizes.block_id_lookup(block_idx)
                except KeyError:
                    pass
                else:
                    spec.allow_overshoot(SMALL_DIM_BLOCK_SIZE_OVERSHOOT)

    env.config_spec.matmul_facts.append(
        MatmulFact(
            lhs_ndim=lhs.ndim,
            rhs_ndim=rhs.ndim,
            m_block_id=env.get_block_id(m),
            n_block_id=env.get_block_id(n),
            k_block_id=env.get_block_id(k),
            static_m=_static_problem_extent(env, m),
            static_n=_static_problem_extent(env, n),
            static_k=_static_problem_extent(env, k),
            lhs_dtype=lhs.dtype,
            rhs_dtype=rhs.dtype,
        )
    )

    # Triton only supports 2D dot operations. Constrain each leading axis from
    # a rank-3 operand to one element so codegen can squeeze it before tl.dot.
    # Pallas uses jnp.dot_general and handles leading axes natively.
    if (lhs.ndim == 3 or rhs.ndim == 3) and env.backend_name != "pallas":
        leading_dims = []
        if lhs.ndim == 3:
            leading_dims.append(lhs.shape[0])
        if rhs.ndim == 3:
            leading_dims.append(rhs.shape[0])
        for leading_dim in leading_dims:
            block_idx = env.get_block_id(leading_dim)
            if block_idx is not None:
                env.block_sizes[block_idx].update_max_block(1)


@dataclasses.dataclass(frozen=True)
class CuteTcgen05SearchPlan:
    """Side-effect-free tcgen05 search limits for one MMA candidate."""

    m: int | torch.SymInt
    n: int | torch.SymInt
    k: int | torch.SymInt
    static_m: int
    static_n: int
    static_k: int
    leading_work_multiplier: int
    is_fp8: bool
    mma_k: int
    max_tcgen05_m: int
    max_tcgen05_n: int
    max_search_m: int
    max_search_n: int
    max_search_k: int
    min_search_m: int


def plan_cute_tcgen05_search(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    has_leading_passthrough: bool,
) -> CuteTcgen05SearchPlan | None:
    """Return search limits without mutating the shared ``ConfigSpec``."""
    m, n, k = _dot_dimensions(lhs, rhs)
    env = CompileEnvironment.current()
    static_m = _static_problem_extent(env, m)
    static_n = _static_problem_extent(env, n)
    static_k = _static_problem_extent(env, k)
    is_fp8 = lhs.dtype == torch.float8_e4m3fn
    mma_k = 32 if is_fp8 else 16
    if (
        env.backend_name != "cute"
        or lhs.dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
        or rhs.dtype != lhs.dtype
        or static_m is None
        or static_n is None
        or static_k is None
        or static_m < 64
        or static_n < 8
        or static_k < mma_k
    ):
        return None

    from .._compiler.cute.mma_support import get_cute_mma_support

    support = get_cute_mma_support()
    if not (support.tcgen05_f8 if is_fp8 else support.tcgen05_f16bf16):
        return None

    def pow2_floor_at_least(value: int, minimum: int) -> int:
        return 1 << (max(minimum, value).bit_length() - 1)

    max_tcgen05_n = min(256, pow2_floor_at_least(static_n, 8))
    max_tcgen05_m = 256 if max_tcgen05_n >= 128 and static_m >= 256 else 128
    max_search_m = min(max_tcgen05_m, pow2_floor_at_least(static_m, 64))
    max_search_n = max_tcgen05_n
    max_search_k = min(128, pow2_floor_at_least(static_k, mma_k))
    min_search_m = 128 if max_tcgen05_m >= 256 else 64
    leading_work_multiplier = 1
    if has_leading_passthrough:
        leading_operand = lhs if lhs.ndim == 3 else rhs
        static_leading = _static_problem_extent(env, leading_operand.shape[0])
        if static_leading is not None:
            leading_work_multiplier = max(static_leading, 1)
        if static_m % min_search_m != 0:
            min_search_m = 64
        max_search_m = min(max_search_m, static_m & -static_m)
        max_search_n = min(max_search_n, static_n & -static_n)
        max_search_k = min(max_search_k, static_k & -static_k)
        if max_search_m < min_search_m or max_search_n < 8 or max_search_k < mma_k:
            return None
    if static_m % max_search_m != 0 and static_n % max_search_n != 0:
        max_search_m = min(max_search_m, 128)
    return CuteTcgen05SearchPlan(
        m=m,
        n=n,
        k=k,
        static_m=static_m,
        static_n=static_n,
        static_k=static_k,
        leading_work_multiplier=leading_work_multiplier,
        is_fp8=is_fp8,
        mma_k=mma_k,
        max_tcgen05_m=max_tcgen05_m,
        max_tcgen05_n=max_tcgen05_n,
        max_search_m=max_search_m,
        max_search_n=max_search_n,
        max_search_k=max_search_k,
        min_search_m=min_search_m,
    )


def enable_cute_tcgen05_search(
    lhs: torch.Tensor,
    *,
    plan: CuteTcgen05SearchPlan,
    m_block_id: int,
    n_block_id: int,
    k_block_id: int,
    input_dtype: torch.dtype,
    has_leading_passthrough: bool,
    explicit_epi_tile_compatible: bool,
) -> None:
    """Apply one preflighted tcgen05 search plan to the shared config."""
    env = CompileEnvironment.current()
    spec = env.config_spec
    spec.register_cute_tcgen05_mma_analysis(
        m_block_id=m_block_id,
        n_block_id=n_block_id,
        k_block_id=k_block_id,
        input_dtype=input_dtype,
        has_leading_passthrough=has_leading_passthrough,
        explicit_epi_tile_compatible=explicit_epi_tile_compatible,
    )
    static_m = plan.static_m
    static_n = plan.static_n
    static_k = plan.static_k
    max_search_m = plan.max_search_m
    max_search_n = plan.max_search_n
    max_search_k = plan.max_search_k
    spec.cute_tcgen05_search_enabled = True
    allow_full_tile_persistent_pid_types = (
        static_m % max_search_m == 0
        and static_n % max_search_n == 0
        and static_k % max_search_k == 0
    )
    max_cluster_m2_search_k = TCGEN05_TWO_CTA_MAX_K_TILES * max_search_k
    allow_full_tile_cluster_m2_search = (
        allow_full_tile_persistent_pid_types
        and max_search_m >= TCGEN05_TWO_CTA_BLOCK_M
        and max_search_n >= TCGEN05_TWO_CTA_BLOCK_N
        and static_k <= max_cluster_m2_search_k
    )
    allow_edge_cluster_m2_search = (
        not has_leading_passthrough
        and not allow_full_tile_persistent_pid_types
        and plan.max_tcgen05_m >= TCGEN05_TWO_CTA_BLOCK_M
        and plan.max_tcgen05_n >= TCGEN05_TWO_CTA_BLOCK_N
        and static_m >= TCGEN05_TWO_CTA_EDGE_K_TAIL_MIN_DIM
        and static_n >= TCGEN05_TWO_CTA_EDGE_K_TAIL_MIN_DIM
        and static_k >= TCGEN05_TWO_CTA_EDGE_K_TAIL_MIN_DIM
        and static_k <= max_cluster_m2_search_k
        and static_m % TCGEN05_TWO_CTA_BLOCK_M != 0
        and static_n % TCGEN05_TWO_CTA_BLOCK_N != 0
        and static_k % max_search_k != 0
    )
    allow_fp8_small_grid_cluster_m2_search = (
        not has_leading_passthrough
        and plan.is_fp8
        and allow_full_tile_persistent_pid_types
        and max_search_m >= TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
        and max_search_n >= TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N
        and static_k <= max_cluster_m2_search_k
    )
    allow_cluster_m2_search = (
        allow_full_tile_cluster_m2_search
        or allow_edge_cluster_m2_search
        or allow_fp8_small_grid_cluster_m2_search
    )
    if allow_cluster_m2_search:
        num_sms = _cuda_num_sms_or_zero(lhs.device)
        if num_sms > 0:
            if allow_fp8_small_grid_cluster_m2_search:
                cluster_m = TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_M
                cluster_n = TCGEN05_TWO_CTA_FP8_SMALL_GRID_BLOCK_N
            else:
                cluster_m = TCGEN05_TWO_CTA_BLOCK_M
                cluster_n = TCGEN05_TWO_CTA_BLOCK_N
            work_clusters = (
                plan.leading_work_multiplier
                * (static_m // cluster_m)
                * (static_n // cluster_n)
            )
            if work_clusters < num_sms // 4:
                allow_cluster_m2_search = False
                allow_fp8_small_grid_cluster_m2_search = False
    spec.narrow_tcgen05_autotune_to_validated_configs(
        allow_persistent_pid_types=allow_full_tile_persistent_pid_types,
        allow_cluster_m2_search=allow_cluster_m2_search,
        cluster_m2_static_k=static_k if allow_cluster_m2_search else None,
        allow_cluster_m2_edge_k_tail_family=allow_edge_cluster_m2_search,
        allow_cluster_m2_fp8_small_grid=allow_fp8_small_grid_cluster_m2_search,
        ab_stages_three_dtype_bytes=lhs.dtype.itemsize,
        ab_stages_three_device=lhs.device,
    )
    for axis_name, shape, max_size in (
        ("m", plan.m, max_search_m),
        ("n", plan.n, max_search_n),
        ("k", plan.k, max_search_k),
    ):
        block_idx = env.get_block_id(shape)
        if block_idx is None:
            continue
        if axis_name == "k":
            min_size = plan.mma_k
        elif axis_name == "m":
            min_size = plan.min_search_m
        else:
            min_size = 8
        env.block_sizes[block_idx].update_min_block(min_size, allow_flattened=True)
        env.block_sizes[block_idx].update_max_block(max_size)


@_decorators.register_fake(dot)
def _(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    acc: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    result_shape = _dot_result_shape(mat1, mat2)

    if acc is not None:
        return acc.new_empty(result_shape)

    # Determine output dtype using the helper function
    resolved_out_dtype = out_dtype or _compute_out_dtype(mat1.dtype, mat2.dtype)
    return torch.empty(result_shape, dtype=resolved_out_dtype, device=mat1.device)


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
        if mat1.ndim == 3 or mat2.ndim == 3:
            mat1_batched = mat1 if mat1.ndim == 3 else mat1.unsqueeze(0)
            mat2_batched = mat2 if mat2.ndim == 3 else mat2.unsqueeze(0)
            batch = max(mat1_batched.shape[0], mat2_batched.shape[0])
            result = torch.bmm(
                mat1_batched.expand(batch, -1, -1),
                mat2_batched.expand(batch, -1, -1),
                out_dtype=resolved_out_dtype,
            )
        else:
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
