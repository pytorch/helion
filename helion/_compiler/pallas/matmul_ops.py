"""Pallas-backend codegen for ops defined in ``helion.language.matmul_ops``.

Backend-specific codegen bodies live here (not in the backend-neutral language
module).  Importing this module runs the ``@_decorators.codegen(op, "pallas")``
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
from ..matmul_utils import _emit_pallas_matmul
from ..matmul_utils import _needs_f32_accumulator

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


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
        lhs_ndim=lhs_proxy.ndim,
    )
