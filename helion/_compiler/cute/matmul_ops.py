"""CuTe-backend codegen for ops defined in ``helion.language.matmul_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "cute")``
registrations; ``matmul_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch
from torch._subclasses.fake_tensor import FakeTensor

from ... import exc
from ...language import _decorators
from ...language.matmul_ops import _static_dim_value
from ...language.matmul_ops import dot
from ...language.matmul_ops import dot_scaled
from ..ast_extension import expr_from_string
from ..compile_environment import CompileEnvironment
from ..matmul_utils import _compute_out_dtype
from .indexing import CutePackedAffineLoad
from .indexing import CutePackedTerms
from .matmul_fallback import _cast_operand_to_f32
from .matmul_fallback import _cute_operand_is_computed_fp8
from .matmul_fallback import _emit_cute_matmul
from .matmul_utils import _cute_active_index_var
from .matmul_utils import _cute_active_mask_var
from .matmul_utils import cute_f32_mma_matches_dot
from .matmul_utils import cute_has_synthetic_lane_k
from .matmul_utils import cute_lower_rhs_for_matmul
from .matmul_utils import cute_outer_accumulates_result
from .matmul_utils import cute_outer_accumulator_dtype
from .matmul_utils import cute_outer_accumulator_out_dtype
from .matmul_utils import cute_resolve_active_block_id
from .matmul_utils import cute_resolve_active_matmul_k_block_id
from .matmul_utils import cute_static_k_invariant_extent
from .matmul_utils import emit_cute_serial_mm_from_load_views
from .strategies import is_pure_matmul_role_lifecycle_config
from .tcgen05_constants import TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


def _requested_pure_matmul_role_lifecycle(state: CodegenState) -> bool:
    return is_pure_matmul_role_lifecycle_config(state.device_function.config)


def _requested_tcgen05_flat_role_coordinates(state: CodegenState) -> bool:
    return bool(
        state.device_function.config.get(
            TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY, False
        )
    )


@_decorators.codegen(dot, "cute")
def _(state: CodegenState) -> object:
    from .collective_matmul import mark_collective_dot

    lhs_proxy = state.proxy_args[0]
    assert isinstance(lhs_proxy, FakeTensor)
    rhs_proxy = state.proxy_args[1]
    assert isinstance(rhs_proxy, FakeTensor)
    acc_proxy = state.proxy_args[2] if len(state.proxy_args) > 2 else None
    out_dtype_proxy = state.proxy_args[3] if len(state.proxy_args) > 3 else None

    lhs_ast = state.ast_args[0]
    if isinstance(lhs_ast, int | float | bool | None):
        lhs_ast = ast.Constant(value=lhs_ast)
    rhs_ast = state.ast_args[1]
    if isinstance(rhs_ast, int | float | bool | None):
        rhs_ast = ast.Constant(value=rhs_ast)
    acc_ast = state.ast_arg(2)
    assert isinstance(lhs_ast, (ast.AST, CutePackedAffineLoad, CutePackedTerms))
    assert isinstance(rhs_ast, (ast.AST, CutePackedTerms))

    is_acc_none = isinstance(acc_ast, ast.Constant) and acc_ast.value is None

    acc_dtype: torch.dtype | None = None
    if not is_acc_none:
        assert isinstance(acc_proxy, FakeTensor)
        acc_dtype = acc_proxy.dtype

    out_dtype: torch.dtype | None = None
    if out_dtype_proxy is not None:
        assert isinstance(out_dtype_proxy, torch.dtype)
        out_dtype = out_dtype_proxy

    # Try MMA path first for configurations whose dtype semantics match fp32 MMA.
    if cute_f32_mma_matches_dot(lhs_proxy.dtype, rhs_proxy.dtype, acc_dtype, out_dtype):
        from .cute_mma import codegen_cute_mma_dot

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
        lhs_m_size=lhs_proxy.shape[-2],
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
    if (
        not is_acc_none
        and cute_f32_mma_matches_dot(
            lhs_proxy.dtype, rhs_proxy.dtype, acc_dtype, out_dtype
        )
        and (
            collective := mark_collective_dot(
                state, k_block_id=k_block_id, lhs=lhs_ast, rhs=rhs_ast, acc=acc_ast
            )
        )
        is not None
    ):
        return collective
    static_k_extent = None
    if k_block_id is None and state.fx_node is not None:
        lhs_node = state.fx_node.args[0] if len(state.fx_node.args) > 0 else None
        rhs_node = state.fx_node.args[1] if len(state.fx_node.args) > 1 else None
        if isinstance(lhs_node, torch.fx.Node) and isinstance(rhs_node, torch.fx.Node):
            static_k_extent = cute_static_k_invariant_extent(lhs_node, rhs_node)
    env = CompileEnvironment.current()
    static_lhs_k = _static_dim_value(env, lhs_proxy.shape[-1])
    static_rhs_k = _static_dim_value(env, rhs_proxy.shape[-2])
    k_is_one = static_lhs_k == 1 and static_rhs_k == 1
    serial_result = None
    if static_k_extent is None and k_block_id is None and not k_is_one:
        if (
            state.fx_node is not None
            and isinstance(lhs_node := state.fx_node.args[0], torch.fx.Node)
            and isinstance(rhs_node := state.fx_node.args[1], torch.fx.Node)
        ):
            serial_result = emit_cute_serial_mm_from_load_views(
                state.codegen,
                state.env,
                state.fx_node,
                lhs_node,
                rhs_node,
                acc=None if is_acc_none else acc_ast,
                acc_dtype=acc_dtype,
                out_dtype=effective_out_dtype,
            )
        if serial_result is None:
            raise exc.BackendUnsupported(
                "cute",
                "CuTe scalar matmul fallback requires an active K tile or a K-invariant static shortcut",
            )
    if _requested_pure_matmul_role_lifecycle(state):
        raise exc.BackendUnsupported(
            "cute",
            "tcgen05_strategy='pure_matmul_role_lifecycle' requires hl.dot "
            "to lower through the tcgen05 K-loop path",
        )
    if _requested_tcgen05_flat_role_coordinates(state):
        raise exc.BackendUnsupported(
            "cute",
            f"{TCGEN05_FLAT_ROLE_COORDINATES_CONFIG_KEY}=True requires "
            "hl.dot to lower through the tcgen05 K-loop path",
        )
    if serial_result is not None:
        return serial_result
    if cute_has_synthetic_lane_k(state.codegen, k_block_id):
        # The wrapping lane loop also repeats accumulator initialization and
        # users of this dot. A reduction over live threads would therefore
        # expose a partial dot on every iteration, not the complete K sum.
        # Native/collective lowering above owns the whole K axis; the scalar
        # fallback is only valid when that axis is fully mapped to threads.
        raise exc.BackendUnsupported(
            "cute",
            "CuTe hl.dot scalar fallback cannot reduce a K axis split across synthetic lanes",
        )
    dot_lhs_node = (
        state.fx_node.args[0]
        if state.fx_node is not None and len(state.fx_node.args) > 0
        else None
    )
    dot_rhs_node = (
        state.fx_node.args[1]
        if state.fx_node is not None and len(state.fx_node.args) > 1
        else None
    )
    dot_acc_node = (
        state.fx_node.args[2]
        if state.fx_node is not None and len(state.fx_node.args) > 2
        else None
    )
    return _emit_cute_matmul(
        state.codegen,
        lhs_ast,
        rhs_ast,
        accumulate_in_lane_loop=not cute_outer_accumulates_result(
            state.fx_node, is_acc_none=is_acc_none
        ),
        k_block_id=k_block_id,
        static_k_extent=static_k_extent,
        acc=None if is_acc_none else acc_ast,
        out_dtype=effective_out_dtype,
        acc_dtype=acc_dtype,
        lhs_dtype=lhs_proxy.dtype,
        rhs_dtype=rhs_proxy.dtype,
        lhs_node=dot_lhs_node,
        rhs_node=dot_rhs_node,
        acc_node=None if is_acc_none else dot_acc_node,
        fx_node=state.fx_node,
    )


# Formats the scaled fallback decodes per element: e8m0-scaled fp8 (raw bytes
# through the bit-exact decode helper) and half-precision floats.  Packed e2m1
# needs a K-split nibble unpack and e5m2 a decode helper the backend lacks.
_CUTE_DOT_SCALED_FORMATS = {
    "e4m3": (torch.float8_e4m3fn, torch.uint8),
    "fp16": (torch.float16,),
    "bf16": (torch.bfloat16,),
}


def _cute_dot_scaled_operand(
    state: CodegenState,
    *,
    position: int,
    k_block_id: int,
    k_dim: int,
) -> ast.AST:
    """The dequantized float32 element of ``hl.dot_scaled`` operand ``position``
    (0: mat1 ``[M, K]``, 3: mat2 ``[K, N]``) at this thread's coordinates.

    The scale (``[M, K // group]`` / ``[N, K // group]`` e8m0 bytes, the factor
    ``2 ** (byte - 127)``) is read at the thread's own row and K coordinates
    divided by the group.  ``expose_dot_scaled_scales`` hands it the scale
    tensor in place of the scale tile when both are full K slices indexed by
    the operand's own row tile; anything else is refused.
    """
    from ...language._tracing_ops import _host_tensor

    env = CompileEnvironment.current()
    fx_node = state.fx_node
    assert fx_node is not None
    data_node = fx_node.args[position]
    scale_node = fx_node.args[position + 1]
    fmt = fx_node.args[position + 2]
    data = state.proxy_arg(position)
    scale = state.proxy_arg(position + 1)
    assert isinstance(data, torch.Tensor) and isinstance(scale, torch.Tensor)
    if fmt not in _CUTE_DOT_SCALED_FORMATS:
        raise exc.BackendUnsupported("cute", f"hl.dot_scaled with {fmt} operands")
    if data.dtype not in _CUTE_DOT_SCALED_FORMATS[fmt]:
        raise exc.BackendUnsupported(
            "cute", f"hl.dot_scaled {fmt} operand of dtype {data.dtype}"
        )
    if scale.dtype is not torch.uint8:
        raise exc.BackendUnsupported("cute", "hl.dot_scaled scales other than e8m0")
    if not (
        isinstance(scale_node, torch.fx.Node) and scale_node.target is _host_tensor
    ):
        raise exc.BackendUnsupported(
            "cute",
            "hl.dot_scaled needs its operands and their scales loaded as full K "
            "slices of the same row tile",
        )
    groups = _static_dim_value(env, scale.shape[1])
    if groups is None or groups <= 0 or k_dim % groups != 0:
        raise exc.BackendUnsupported("cute", "hl.dot_scaled scale groups along K")
    row_axis = 0 if position == 0 else 1
    row_block_id = cute_resolve_active_block_id(state.codegen, data.shape[row_axis])
    row_index = (
        None
        if row_block_id is None
        else _cute_active_index_var(state.codegen, row_block_id)
    )
    k_index = _cute_active_index_var(state.codegen, k_block_id)
    if row_block_id is None or row_index is None or k_index is None:
        raise exc.BackendUnsupported(
            "cute", "hl.dot_scaled needs active row and K coordinates"
        )
    scale_name = state.device_function.tensor_arg(scale).name
    scale_load = (
        f"cute.arch.load({scale_name}.iterator"
        f" + cutlass.Int32({row_index}) * cutlass.Int32({scale_name}.layout.stride[0])"
        f" + cutlass.Int32(({k_index}) // {k_dim // groups})"
        f" * cutlass.Int32({scale_name}.layout.stride[1]), cutlass.Uint8)"
    )
    # Lanes past the row tile or past K (padded to a power of two) hold a
    # zero operand element; their scale must stay finite, since a byte past
    # the scale tensor's last group (255 decodes to 2**128, infinite in fp32)
    # would make the product 0 * inf = NaN.
    masks = [
        mask
        for block_id in (row_block_id, k_block_id)
        if (mask := _cute_active_mask_var(state.codegen, block_id)) is not None
    ]
    if masks:
        scale_load = f"({scale_load} if {' and '.join(masks)} else cutlass.Uint8(127))"
    factor = expr_from_string(
        f"cute.math.exp2(cutlass.Float32({scale_load}) - cutlass.Float32(127.0),"
        " fastmath=False)"
    )
    data_ast = state.ast_args[position]
    if not isinstance(data_ast, ast.AST):
        raise exc.BackendUnsupported("cute", "hl.dot_scaled packed operand")
    decoded = _cast_operand_to_f32(
        data_ast,
        torch.float8_e4m3fn if fmt == "e4m3" else data.dtype,
        is_computed_fp8=_cute_operand_is_computed_fp8(data_node),
    )
    return expr_from_string("{x} * {scale}", x=decoded, scale=factor)


@_decorators.codegen(dot_scaled, "cute")
def _(state: CodegenState) -> object:
    """Lower ``hl.dot_scaled`` correct-first through the scalar matmul fallback.

    Each thread dequantizes its own elements of both operands
    (``_cute_dot_scaled_operand``) and the float32 products reduce over K like
    ``hl.dot``'s; the tcgen05 block-scaled MMA is not used.
    """
    fx_node = state.fx_node
    assert fx_node is not None
    mat1 = state.proxy_arg(0)
    mat2 = state.proxy_arg(3)
    acc_proxy = state.proxy_arg(6) if len(state.proxy_args) > 6 else None
    out_dtype_proxy = state.proxy_arg(7) if len(state.proxy_args) > 7 else None
    assert isinstance(mat1, torch.Tensor) and isinstance(mat2, torch.Tensor)
    acc_ast = state.ast_arg(6) if len(state.ast_args) > 6 else None
    is_acc_none = acc_ast is None or (
        isinstance(acc_ast, ast.Constant) and acc_ast.value is None
    )
    acc_dtype: torch.dtype | None = None
    if not is_acc_none:
        assert isinstance(acc_proxy, torch.Tensor)
        acc_dtype = acc_proxy.dtype
    out_dtype = out_dtype_proxy if isinstance(out_dtype_proxy, torch.dtype) else None

    env = CompileEnvironment.current()
    k_block_id = cute_resolve_active_matmul_k_block_id(
        state.codegen,
        mat1.shape[-1],
        mat2.shape[-2],
        mat2.shape[-1],
        lhs_m_size=mat1.shape[-2],
    )
    # The K block's own extent: the operand's last dim is the reduction dim's
    # power-of-two size (128 for K = 96), which would mis-group the scales.
    k_size = None if k_block_id is None else env.block_sizes[k_block_id].size
    k_dim = (
        _static_dim_value(env, k_size)
        if isinstance(k_size, (int, torch.SymInt))
        else None
    )
    if k_block_id is None or k_dim is None:
        raise exc.BackendUnsupported(
            "cute", "CuTe hl.dot_scaled requires an active K axis of static extent"
        )
    if cute_has_synthetic_lane_k(state.codegen, k_block_id):
        raise exc.BackendUnsupported(
            "cute",
            "CuTe hl.dot_scaled cannot reduce a K axis split across synthetic lanes",
        )
    lhs = _cute_dot_scaled_operand(
        state, position=0, k_block_id=k_block_id, k_dim=k_dim
    )
    rhs = _cute_dot_scaled_operand(
        state, position=3, k_block_id=k_block_id, k_dim=k_dim
    )
    resolved_out_dtype = out_dtype or torch.float32
    effective_out_dtype = cute_outer_accumulator_out_dtype(
        resolved_out_dtype,
        cute_outer_accumulator_dtype(fx_node, is_acc_none=is_acc_none),
    )
    return _emit_cute_matmul(
        state.codegen,
        lhs,
        rhs,
        accumulate_in_lane_loop=not cute_outer_accumulates_result(
            fx_node, is_acc_none=is_acc_none
        ),
        k_block_id=k_block_id,
        acc=None if is_acc_none else acc_ast,
        out_dtype=effective_out_dtype,
        acc_dtype=acc_dtype,
        lhs_dtype=torch.float32,
        rhs_dtype=torch.float32,
        lhs_node=fx_node.args[0],
        rhs_node=fx_node.args[3],
        acc_node=None if is_acc_none else fx_node.args[6],
        fx_node=fx_node,
    )
