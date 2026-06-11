"""HelionTritonOverrides -- subclass of Inductor's TritonOverrides."""

from __future__ import annotations

import torch
from torch._inductor.codegen.triton import TritonOverrides
from torch._inductor.virtualized import V

from ...language._decorators import is_api_func

# Tile API functions that always produce non-negative integer values.
_NON_NEGATIVE_TILE_FUNCS = frozenset(
    {
        "tile_index",
        "tile_begin",
        "tile_end",
        "tile_id",
        "tile_count",
        "tile_block_size",
    }
)


def _is_provably_non_negative(
    node: torch.fx.Node,
    _visited: frozenset[torch.fx.Node] = frozenset(),
) -> bool:
    """Check if an FX node produces a provably non-negative integer value.

    Recognized non-negative sources:
    - Tile API calls (``tile_index``, ``tile_begin``, etc.): always >= 0
    - ``prims.iota``: produces [0, 1, 2, ...]
    - ``aten.div.Tensor_mode`` (floordiv) with non-negative dividend AND
      positive divisor: result >= 0
    - ``aten.remainder`` with non-negative dividend AND positive divisor:
      result >= 0

    Uses a visited set to guard against cycles in the FX graph.
    """
    if not isinstance(node, torch.fx.Node) or node.op != "call_function":
        return False
    if node in _visited:
        return False
    visited = _visited | {node}

    target = node.target
    name = getattr(target, "__name__", "")

    # Tile API functions: offset + arange(0, BLOCK), always >= 0
    if is_api_func(target) and name in _NON_NEGATIVE_TILE_FUNCS:
        return True

    # prims.iota (tl.arange): [0, 1, 2, ...]
    if target is torch.ops.prims.iota.default:
        return True

    # floordiv / remainder: non-negative when dividend >= 0 AND divisor > 0
    if target in (torch.ops.aten.div.Tensor_mode, torch.ops.aten.remainder.Scalar):
        if len(node.args) >= 2:
            dividend, divisor = node.args[0], node.args[1]
            if (
                isinstance(dividend, torch.fx.Node)
                and _is_provably_non_negative(dividend, visited)
                and _is_provably_positive_divisor(divisor)
            ):
                return True

    return False


def _is_provably_positive_divisor(node: object) -> bool:
    """Check if a divisor operand is provably positive.

    Recognizes:
    - FX nodes whose ``meta["val"]`` is a ``SymInt`` (kernel size parameters
      like ``M_cols``, ``KH * KW`` -- these represent dimensions and are
      always positive)
    - Positive integer constants
    """
    if isinstance(node, torch.fx.Node):
        val = node.meta.get("val")
        return isinstance(val, torch.SymInt)
    if isinstance(node, int):
        return node > 0
    return False


class HelionTritonOverrides(TritonOverrides):
    """Helion Triton op overrides.

    Inherits all expression generation from Inductor's TritonOverrides.
    """

    @staticmethod
    def _can_simplify_div() -> bool:
        """Check if the current floordiv/remainder can use simple ``//``/``%``.

        Returns True when the dividend is provably non-negative (derived from
        tile indices or prior non-negative arithmetic) AND the divisor is
        provably positive (a SymInt size parameter or positive constant).

        When both conditions hold, Triton's truncation ``//`` equals Python's
        floor ``//``, and Triton's ``%`` equals Python's ``%``.
        """
        node = V.current_node
        if node is None or len(node.args) < 2:
            return False
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor) or val.dtype.is_floating_point:
            return False
        dividend, divisor = node.args[0], node.args[1]
        return (
            isinstance(dividend, torch.fx.Node)
            and _is_provably_non_negative(dividend)
            and _is_provably_positive_divisor(divisor)
        )

    @staticmethod
    def floordiv(a: str, b: str) -> str:
        """Simplified floor division for non-negative integer operands.

        Triton's ``//`` is truncation division, which equals floor division
        when both operands are non-negative.  Emits ``a // b`` directly
        instead of Inductor's defensive sequence that handles negative
        operands and division-by-zero.
        """
        if HelionTritonOverrides._can_simplify_div():
            return f"{a} // {b}"
        return TritonOverrides.floordiv(a, b)

    @staticmethod
    def remainder(a: str, b: str) -> str:
        """Simplified remainder for non-negative integer operands.

        For non-negative operands, Triton's ``%`` matches Python's ``%``.
        Emits ``a % b`` directly instead of Inductor's defensive sequence
        that handles negative operands.
        """
        if HelionTritonOverrides._can_simplify_div():
            return f"{a} % {b}"
        return TritonOverrides.remainder(a, b)
