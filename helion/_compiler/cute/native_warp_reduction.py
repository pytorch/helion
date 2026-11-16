"""Select exact native warp reductions (``redux.sync``) under
``cute_warp_fp32_redux``.

The grouped helper API takes ``fp32_redux=True`` and picks the native fp32
extrema itself.  The direct warp reductions a row emits
(``CuteBackend.reduction_expr``) are rewritten, by the last kernel pass, into
helpers that pick the native form when the operand and group allow it (fp32
extrema on SM100, a 32-bit integer OR over a full warp) and the shuffle
reduction otherwise.
Every rewrite gives the same bits: extrema and OR do not depend on the
combine order, and the native fp32 extrema propagate NaN like
``_cute_nan_max`` / ``_cute_nan_min``.

Despite its name, the knob also takes a row's ``any`` natively: the integer
OR is a ``redux.sync`` on every architecture since SM80, while the fp32
extrema need the SM100 family (on Hopper the knob changes only the OR).
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from ..inductor_lowering import ReductionLowering
from ._ast_pass_utils import call_path_str
from .reduce_call import FP32_REDUX_REDUCES

if TYPE_CHECKING:
    from ..device_ir import DeviceIR

# The dtypes whose extrema a row reduces as fp32 values.
_FP32_ACCUMULATED = (torch.float32, torch.bfloat16, torch.float16)


def has_native_redux_reduction(
    device_ir: DeviceIR, capability: tuple[int, int] | None, *, with_any: bool = True
) -> bool:
    """Whether some reduction of the kernel is one ``cute_warp_fp32_redux``
    takes natively on a device of ``capability``: an ``any`` (a 32-bit OR
    over a full warp, on every target) or, on the SM100 family, a max or a
    min of fp32, bf16 or f16 values, which the reduction accumulates in fp32
    (the fp32 ``redux.sync`` of ``reduce_helpers._native_fp32_extremum``;
    elsewhere, and for integer or fp64 values, the knob leaves them the
    shuffle reduction).  ``with_any=False`` counts the extrema only, all the
    grouped and cluster reduce helpers take natively (an ``any`` keeps their
    shuffle reduction)."""
    extrema = (
        {"max", "min"} if capability is not None and capability[0] == 10 else set()
    )

    def native(node: torch.fx.Node) -> bool:
        lowering = node.meta.get("lowering")
        if not isinstance(lowering, ReductionLowering):
            return False
        if lowering.reduction_type == "any":
            return with_any
        value = node.meta.get("val")
        return (
            lowering.reduction_type in extrema
            and isinstance(value, torch.Tensor)
            and value.dtype in _FP32_ACCUMULATED
        )

    return any(native(node) for info in device_ir.graphs for node in info.graph.nodes)


def _direct_redux_helper(call: ast.Call) -> str | None:
    """The native helper replacing a direct warp reduction, or None."""
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    path = call_path_str(call.func)
    if (
        path == "cute.arch.warp_reduction_max"
        and len(call.args) == 1
        and set(keywords) == {"op", "threads_in_group"}
        and isinstance(op := keywords["op"], ast.Name)
        and op.id == "_cute_nan_max"
    ):
        return "_cute_warp_reduce_max_redux"
    if (
        path != "cute.arch.warp_reduction"
        or len(call.args) != 2
        or set(keywords) != {"threads_in_group"}
    ):
        return None
    combine = call.args[1]
    if isinstance(combine, ast.Name) and combine.id == "_cute_nan_min":
        return "_cute_warp_reduce_min_redux"
    if (
        isinstance(combine, ast.Lambda)
        and [arg.arg for arg in combine.args.args] == ["a", "b"]
        and isinstance(body := combine.body, ast.BinOp)
        and isinstance(body.op, ast.BitOr)
        and isinstance(body.left, ast.Name)
        and isinstance(body.right, ast.Name)
        and (body.left.id, body.right.id) == ("a", "b")
    ):
        return "_cute_warp_reduce_or_redux"
    return None


class _DirectRedux(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        helper = _direct_redux_helper(node)
        if helper is None:
            return node
        threads = next(k for k in node.keywords if k.arg == "threads_in_group")
        return ast.copy_location(
            ast.Call(
                func=ast.Name(id=helper, ctx=ast.Load()),
                args=[node.args[0]],
                keywords=[threads],
            ),
            node,
        )


def select_native_warp_reductions(body: list[ast.stmt]) -> list[ast.stmt]:
    """Rewrite the direct warp reductions into their native helpers.  Runs
    last, so no other pass needs to recognize the helpers."""
    return [_DirectRedux().visit(statement) for statement in body]


def enable_fp32_warp_redux(body: list[ast.stmt]) -> list[ast.stmt]:
    for statement in body:
        for node in ast.walk(statement):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in FP32_REDUX_REDUCES
            ):
                continue
            if not any(keyword.arg == "fp32_redux" for keyword in node.keywords):
                node.keywords.append(
                    ast.keyword(arg="fp32_redux", value=ast.Constant(True))
                )
    return body
