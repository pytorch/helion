"""Backend-neutral helpers for associative-scan combine graphs."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import torch

from .. import exc

if TYPE_CHECKING:
    from collections.abc import Callable

SCAN_ARITHMETIC_OPS: dict[object, str] = {
    operator.add: "+",
    torch.add: "+",
    torch.ops.aten.add.Tensor: "+",
    torch.ops.aten.add.Scalar: "+",
    operator.sub: "-",
    torch.sub: "-",
    torch.ops.aten.sub.Tensor: "-",
    torch.ops.aten.sub.Scalar: "-",
    operator.mul: "*",
    torch.mul: "*",
    torch.ops.aten.mul.Tensor: "*",
    torch.ops.aten.mul.Scalar: "*",
}
SCAN_PYTHON_COMPARISON_OPS: dict[object, str] = {
    operator.eq: "==",
    operator.ne: "!=",
    operator.lt: "<",
    operator.gt: ">",
    operator.le: "<=",
    operator.ge: ">=",
}
SCAN_ATEN_COMPARISON_OPS: dict[object, str] = {
    torch.ops.aten.eq.Tensor: "==",
    torch.ops.aten.eq.Scalar: "==",
    torch.ops.aten.ne.Tensor: "!=",
    torch.ops.aten.ne.Scalar: "!=",
    torch.ops.aten.lt.Tensor: "<",
    torch.ops.aten.lt.Scalar: "<",
    torch.ops.aten.gt.Tensor: ">",
    torch.ops.aten.gt.Scalar: ">",
    torch.ops.aten.le.Tensor: "<=",
    torch.ops.aten.le.Scalar: "<=",
    torch.ops.aten.ge.Tensor: ">=",
    torch.ops.aten.ge.Scalar: ">=",
}
SCAN_MIN_OPS = (torch.minimum, torch.ops.aten.minimum.default)
SCAN_MAX_OPS = (torch.maximum, torch.ops.aten.maximum.default)
SCAN_WHERE_OPS = (torch.where, torch.ops.aten.where.self)
SCAN_CAST_OP = torch.ops.prims.convert_element_type.default


def scan_combine_arg(
    node: torch.fx.Node, index: int, kwarg: str, backend_name: str
) -> object:
    if len(node.args) > index:
        return node.args[index]
    if kwarg in node.kwargs:
        return node.kwargs[kwarg]
    raise exc.BackendUnsupported(
        backend_name, f"associative_scan combine missing argument {kwarg!r}"
    )


def scan_combine_check_extra_args(
    node: torch.fx.Node, max_args: int, backend_name: str
) -> None:
    if len(node.args) > max_args:
        raise exc.BackendUnsupported(
            backend_name, f"associative_scan combine args {node.args!r}"
        )


def scan_combine_check_kwargs(
    node: torch.fx.Node, backend_name: str, allowed: tuple[str, ...] = ()
) -> None:
    unexpected = set(node.kwargs) - set(allowed)
    if unexpected:
        raise exc.BackendUnsupported(
            backend_name,
            f"associative_scan combine kwargs {sorted(unexpected)!r}",
        )


def scan_combine_binary_expression(
    node: torch.fx.Node,
    op: str,
    lhs: str,
    rhs: str,
    operand: Callable[[object], str],
    backend_name: str,
) -> str:
    alpha: object = 1
    max_args = 2
    if op in {"+", "-"}:
        if len(node.args) > 2:
            alpha = node.args[2]
            max_args = 3
        elif "alpha" in node.kwargs:
            alpha = node.kwargs["alpha"]
        scan_combine_check_kwargs(node, backend_name, ("input", "other", "alpha"))
    else:
        scan_combine_check_kwargs(node, backend_name, ("input", "other"))
    scan_combine_check_extra_args(node, max_args, backend_name)
    if not isinstance(alpha, (int, float)) or alpha != 1:
        rhs = f"({rhs}) * ({operand(alpha)})"
    return f"(({lhs}) {op} ({rhs}))"
