"""Argument-aware effects for Helion device graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.fx import Node


def is_impure_node(node: Node, impure_random: bool = True) -> bool:
    """Preserve device effects during DCE and graph scheduling.

    FX recognizes function-level effects, but inline assembly declares its
    effects per call. Helion API tracing normalizes arguments to positional
    form before creating a device-graph node.
    """
    # Autodiff imports this helper before the language and runtime finish
    # importing; resolve the API target only when analyzing a completed graph.
    from ..language.inline_asm_ops import inline_asm_elementwise

    if node.op == "call_function":
        if node.target is inline_asm_elementwise:
            return node.args[4] is not True
        # These ATen checks have no returned tensor. FX marks the assertion
        # overload with a message as effectful, but not either of these.
        if node.target in (
            torch.ops.aten._assert_async.default,
            torch.ops.aten._validate_compressed_sparse_indices.default,
        ):
            return True
    return node.is_impure(impure_random=impure_random)
