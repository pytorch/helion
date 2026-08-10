"""Triton-backend codegen for ops defined in ``helion.language.matmul_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "triton")``
registrations; ``matmul_ops`` imports it at the bottom so registration keeps
the same eager timing as before.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch
from torch._subclasses.fake_tensor import FakeTensor

from ...language import _decorators
from ...language.matmul_ops import dot
from ...language.matmul_ops import dot_scaled
from ..cute.indexing import CutePackedAffineLoad
from ..matmul_utils import _emit_tl_dot_scaled
from ..matmul_utils import emit_tl_dot_with_padding

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


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
