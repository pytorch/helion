"""Recognize the integer decoder before or after the local native fusion.

The fused form is an exact instruction/NaN-fixup contract, not a marker trusted
after arbitrary graph rewrites. Addressing and transport remain caller proofs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ....language.inline_asm_ops import inline_asm_elementwise
from ..native_s2f6_contract import native_s2f6_asm_matches
from ..native_s2f6_contract import native_s2f6_broadcast_matches
from .match_common import node_tensor
from .native_s2f6_match import NativeS2F6Decode
from .native_s2f6_match import match_native_s2f6_decode
from .native_s2f6_match import native_s2f6_same_shape
from .native_s2f6_match import unwrap_native_s2f6_value

if TYPE_CHECKING:
    from collections.abc import Callable


def match_native_s2f6_operand(
    value: torch.fx.Node,
    *,
    resolve: Callable[[torch.fx.Node], torch.fx.Node] | None = None,
) -> NativeS2F6Decode | None:
    """Retain the exact BF16 contract across the early local fusion phase."""
    if proof := match_native_s2f6_decode(value, resolve=resolve):
        raw, scale, result = (
            node_tensor(proof.raw),
            node_tensor(proof.scale),
            node_tensor(value),
        )
        if raw is not None and scale is not None and result is not None:
            if native_s2f6_broadcast_matches(raw, scale, result):
                return proof
        return None
    nodes: set[torch.fx.Node] = set()

    def unwrap(node: object) -> object:
        return unwrap_native_s2f6_value(node, nodes, resolve, require_same_shape=True)

    def call(node: object, target: object, count: int) -> bool:
        return (
            isinstance(node, torch.fx.Node)
            and node.op == "call_function"
            and node.target is target
            and len(node.args) == count
            and not node.kwargs
        )

    result = unwrap(value)
    if not call(result, torch.ops.aten.view.dtype, 2):
        return None
    assert isinstance(result, torch.fx.Node)
    result_tensor = node_tensor(result)
    if (
        result.args[1] is not torch.bfloat16
        or result_tensor is None
        or result_tensor.dtype != torch.bfloat16
    ):
        return None
    asm = unwrap(result.args[0])
    if not call(asm, inline_asm_elementwise, 6):
        return None
    assert isinstance(asm, torch.fx.Node)
    text, constraints, operands, dtype, pure, pack = asm.args
    asm_tensor = node_tensor(asm)
    if (
        not isinstance(operands, (tuple, list))
        or not native_s2f6_asm_matches(
            text, constraints, dtype, pure, pack, len(operands)
        )
        or asm_tensor is None
        or asm_tensor.dtype != torch.int16
    ):
        return None
    casts: list[torch.fx.Node] = []
    sources: list[torch.fx.Node] = []
    for operand, expected in zip(operands, (torch.int8, torch.uint8), strict=True):
        cast = unwrap(operand)
        if not call(cast, torch.ops.prims.convert_element_type.default, 2):
            return None
        assert isinstance(cast, torch.fx.Node)
        source = unwrap(cast.args[0])
        cast_tensor, source_tensor = node_tensor(cast), node_tensor(source)
        if (
            cast.args[1] is not torch.int32
            or cast_tensor is None
            or cast_tensor.dtype != torch.int32
            or source_tensor is None
            or source_tensor.dtype != expected
            or not native_s2f6_same_shape(cast_tensor, source_tensor)
        ):
            return None
        assert isinstance(source, torch.fx.Node)
        casts.append(cast)
        sources.append(source)
    raw_tensor, scale_tensor = (node_tensor(node) for node in sources)
    assert raw_tensor is not None and scale_tensor is not None
    if not native_s2f6_broadcast_matches(
        raw_tensor, scale_tensor, result_tensor
    ) or not native_s2f6_same_shape(asm_tensor, result_tensor):
        return None
    return NativeS2F6Decode(
        sources[0], sources[1], casts[0], casts[1], value, frozenset(nodes)
    )
