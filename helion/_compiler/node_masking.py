from __future__ import annotations

import functools

import torch.fx
from torch.fx.experimental import proxy_tensor

from helion.language._tracing_ops import _mask_to


def mask_node_inputs(
    node: torch.fx.Node,
    other: float | bool = 0,
) -> None:
    """Inplace update the node's args and kwargs to apply masking if needed."""
    apply = functools.partial(apply_masking, other=other, base_node=node)
    node.args = torch.fx.map_arg(node.args, apply)
    node.kwargs = torch.fx.map_arg(node.kwargs, apply)


def apply_masking(
    node: torch.fx.Node,
    *,
    base_node: torch.fx.Node,
    other: float | bool = 0,
) -> torch.fx.Node:
    """Analyze the node and apply masking if needed."""
    current_mask = cached_masked_value(node)
    if current_mask == other:
        return node  # already masked, no need to change it
    for user in node.users:
        if user.op == "call_function" and user.target == _mask_to:
            if user.args[1] == other:
                assert user.args[0] is node
                return user  # reuse existing mask_to node
    from helion._compiler.inductor_lowering import APIFuncLowering

    # If we reach here, we need to create a new mask_to node
    with node.graph.inserting_before(base_node):
        new_node = node.graph.call_function(_mask_to, (node, other), {})
    new_node.meta.update(base_node.meta)
    with proxy_tensor.disable_proxy_modes_tracing():
        new_node.meta["val"] = node.meta["val"].clone()
    # pyre-ignore[6]
    new_node.meta["lowering"] = APIFuncLowering(_mask_to)
    new_node.meta["masked_value"] = other
    return new_node


def cached_masked_value(
    node: torch.fx.Node,
) -> float | bool | None:
    """Determine the current masked value for the node."""
    if "masked_value" in node.meta:
        return node.meta["masked_value"]
    if node.op != "call_function":
        return None
    node.meta["masked_value"] = result = node.meta["lowering"].get_masked_value(node)
    return result
