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
from ..compile_environment import CompileEnvironment
from ..matmul_utils import _compute_out_dtype
from ..matmul_utils import _needs_f32_accumulator
from .indexing import CutePackedAffineLoad
from .indexing import CutePackedTerms
from .matmul_fallback import _emit_cute_matmul
from .matmul_utils import cute_lower_rhs_for_matmul
from .matmul_utils import cute_outer_accumulates_result
from .matmul_utils import cute_outer_accumulator_dtype
from .matmul_utils import cute_outer_accumulator_out_dtype
from .matmul_utils import cute_resolve_active_block_id
from .matmul_utils import cute_resolve_active_matmul_k_block_id
from .matmul_utils import cute_static_k_invariant_extent
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
    static_lhs_k = _static_dim_value(env, lhs_proxy.shape[-1])
    static_rhs_k = _static_dim_value(env, rhs_proxy.shape[-2])
    k_is_one = static_lhs_k == 1 and static_rhs_k == 1
    if static_k_extent is None and k_block_id is None and not k_is_one:
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
        lhs_node=dot_lhs_node,
        rhs_node=dot_rhs_node,
        acc_node=None if is_acc_none else dot_acc_node,
    )
