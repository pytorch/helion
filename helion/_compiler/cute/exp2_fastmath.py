# pyrefly: ignore-errors
"""``ex2.approx.ftz`` policy for the cute backend.

The default exp2 lowering guards the denormal output range: each call
costs an extra FSETP plus two predicated FMULs per element and lengthens
live ranges (the input must survive until the fixup).  ``ex2.approx.ftz``
skips the fixup; exp results below the normal f32 range flush to zero.
The two knobs here:

- ``mark_ftz_safe_exp_nodes``: device-IR analysis marking exp sites where
  flushing is PROVABLY output-identical (the ``sum(exp(x - amax(x)))``
  cross-entropy / logsumexp / softmax-denominator pattern); the cute op
  overrides emit ``fastmath=True`` for marked sites unconditionally.
  Measured +16% on cross-entropy fwd 32768x8192 fp16 on B200 (the exp
  fixup was the entire gap to quack's handwritten kernel).

- ``apply_exp2_fastmath``: applies ``fastmath=True`` to EVERY exp2 call
  when the ``fast_math`` SETTING is enabled.  This one changes numerics,
  so it is deliberately a setting (``helion.Settings.fast_math`` /
  ``HELION_FAST_MATH=1``) and not an autotuner config knob: tuned configs
  must never change numerics.  For exp-of-shifted-value reductions the
  flushed elements contribute nothing to the sum and round to zero in a
  16-bit output anyway — measured +1.7% on register-resident softmax rows
  on B200.
"""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.fx import Node

FTZ_SAFE_EXP_META_KEY = "ftz_safe_exp"


def _is_neg_inf_mask_of(node: object, source: Node) -> bool:
    """True when ``node`` is the ``_mask_to(source, -inf)`` wrapper Helion
    inserts on reduction inputs (out-of-tile lanes become -inf so amax
    ignores them).  The direction matters: an amax over the MASKED input
    still bounds every valid lane of the raw input (masked lanes only go
    DOWN to -inf), so exp(source - amax(mask(source))) keeps a valid lane
    at exp(0) = 1.  The reverse (amax over the raw input while exp sees
    the masked one) is NOT accepted: raw padding lanes could exceed every
    valid lane and push the whole sum into the flushable range.
    """
    from ...language._tracing_ops import _mask_to

    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and node.target is _mask_to
        and len(node.args) >= 2
        and node.args[0] is source
        and isinstance(node.args[1], float)
        and math.isinf(node.args[1])
        and node.args[1] < 0
    )


def _broadcast_view_source(node: Node) -> Node | None:
    """Return the source of a pure broadcast view (``subscript`` with only
    full slices and Nones, or ``unsqueeze``), else None."""
    from ...language import view_ops

    if node.op != "call_function":
        return None
    if node.target is torch.ops.aten.unsqueeze.default and isinstance(
        node.args[0], torch.fx.Node
    ):
        return node.args[0]
    if node.target is view_ops.subscript and isinstance(node.args[0], torch.fx.Node):
        indices = node.args[1]
        if not isinstance(indices, (list, tuple)):
            return None
        for index in indices:
            if index is None:
                continue
            if (
                isinstance(index, slice)
                and index.start is None
                and index.stop is None
                and index.step is None
            ):
                continue
            return None
        return node.args[0]
    return None


def _normalized_reduce_dims(node: Node, dims: object) -> tuple[int, ...] | None:
    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor) or not isinstance(dims, (list, tuple)):
        return None
    if not all(isinstance(d, int) for d in dims):
        return None
    return tuple(sorted(d % val.ndim for d in dims))


def _users_are_sum_over_dims(node: Node, dims: tuple[int, ...]) -> bool:
    """Every (transitive, through ``_mask_to(., 0)``) user must be a
    ``sum`` over exactly ``dims`` — the exp values may not escape."""
    from ...language._tracing_ops import _mask_to

    for user in node.users:
        if (
            user.op == "call_function"
            and user.target is _mask_to
            and len(user.args) >= 2
            and user.args[0] is node
            and user.args[1] == 0
        ):
            if not _users_are_sum_over_dims(user, dims):
                return False
            continue
        if (
            user.op == "call_function"
            and user.target is torch.ops.aten.sum.dim_IntList
            and len(user.args) >= 2
            and user.args[0] is node
            and _normalized_reduce_dims(node, user.args[1]) == dims
        ):
            continue
        return False
    return True


def mark_ftz_safe_exp_nodes(graph: torch.fx.Graph) -> None:
    """Mark ``exp``/``exp2`` nodes whose ``ex2.approx.ftz`` lowering is
    provably output-identical, so codegen can skip the denormal fixup.

    The pattern is ``sum(exp(x - amax(x, dims)))`` over the same ``x``:
    the shifted argument is <= 0 with at least one element equal to 0
    among the lanes the amax covered, so the fp32 sum is >= 1, and exp
    outputs below 2^-126 — the only range where ``ex2.approx.ftz``
    differs from the guarded lowering — can never reach the sum's ulp
    (their total mass is < N * 2^-126 << 2^-24).  The exp values must
    not escape into anything but the sum.  This matches the numerics
    triton (libdevice) and handwritten CuTe kernels (quack) already
    have on this pattern, without touching exp sites the proof does
    not cover (those stay behind the ``fast_math`` setting).

    Runs on the pre-roll device IR graphs; the reduction roller's
    ``node_copy`` carries ``node.meta`` into the rolled sweep subgraphs,
    where the cute backend's op overrides consume the mark.
    """
    for node in graph.nodes:
        if node.op != "call_function" or node.target not in (
            torch.ops.aten.exp.default,
            torch.ops.aten.exp2.default,
        ):
            continue
        # The >= 1-sum ulp argument is written for fp32 accumulation; a
        # pure 16-bit pipeline flushes below 2^-14 where the error mass
        # CAN reach the sum's ulp, so only fp32 exp sites are marked.
        node_val = node.meta.get("val")
        if (
            not isinstance(node_val, torch.Tensor)
            or node_val.dtype is not torch.float32
        ):
            continue
        sub_node = node.args[0] if node.args else None
        if (
            not isinstance(sub_node, torch.fx.Node)
            or sub_node.op != "call_function"
            or sub_node.target is not torch.ops.aten.sub.Tensor
            or len(sub_node.args) < 2
            or not isinstance(sub_node.args[0], torch.fx.Node)
            or not isinstance(sub_node.args[1], torch.fx.Node)
        ):
            continue
        x, max_view = sub_node.args[0], sub_node.args[1]
        # amax with keepdim feeds sub directly; without keepdim it goes
        # through a broadcast view (subscript-with-None / unsqueeze).
        max_node = _broadcast_view_source(max_view) or max_view
        if (
            max_node.op != "call_function"
            or max_node.target is not torch.ops.aten.amax.default
            or len(max_node.args) < 2
            or not isinstance(max_node.args[0], torch.fx.Node)
        ):
            continue
        max_input = max_node.args[0]
        if max_input is not x and not _is_neg_inf_mask_of(max_input, x):
            continue
        dims = _normalized_reduce_dims(max_node.args[0], max_node.args[1])
        if dims is None:
            continue
        # The max must broadcast back along exactly the reduced dims.
        x_val = x.meta.get("val")
        view_val = max_view.meta.get("val")
        if not isinstance(x_val, torch.Tensor) or not isinstance(
            view_val, torch.Tensor
        ):
            continue
        if view_val.ndim != x_val.ndim:
            continue
        expected = [1 if d in dims else x_val.shape[d] for d in range(x_val.ndim)]
        if list(view_val.shape) != expected:
            continue
        if not _users_are_sum_over_dims(node, dims):
            continue
        node.meta[FTZ_SAFE_EXP_META_KEY] = True


def apply_exp2_fastmath(body: list[ast.stmt]) -> list[ast.stmt]:
    for top in body:
        for node in ast.walk(top):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "exp2"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "math"
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "cute"
            ):
                continue
            if any(kw.arg == "fastmath" for kw in node.keywords):
                continue
            node.keywords.append(
                ast.keyword(arg="fastmath", value=ast.Constant(value=True))
            )
            ast.fix_missing_locations(node)
    return body
