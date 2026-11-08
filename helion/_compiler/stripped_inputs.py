"""The inverse of the Inductor lowering's ``strip_unused_inputs``.

Kept apart from ``inductor_lowering`` (which imports backend modules) so the
backends' FX walkers can import it.
"""

from __future__ import annotations

import torch


def restore_stripped_operands(node: torch.fx.Node) -> tuple[object, ...] | None:
    """The arguments of the lowered pointwise ``node`` as the user wrote them.

    ``inductor_lowering.strip_unused_inputs`` keeps the first use of each
    input and masks the later ones with ``None`` (``x * x`` reaches the
    device IR as ``mul(x, None)``); a masked operand is the node's unique
    tensor input. ``None`` when a masked operand is ambiguous (no or several
    tensor inputs).
    """
    args = tuple(node.args)
    if not any(arg is None for arg in args):
        return args
    inputs = {arg for arg in args if isinstance(arg, torch.fx.Node)}
    if len(inputs) != 1:
        return None
    (operand,) = inputs
    return tuple(operand if arg is None else arg for arg in args)
