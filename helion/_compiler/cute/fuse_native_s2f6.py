"""Fuse a proved integer S2F6 decode without changing its BF16 bit contract."""

from __future__ import annotations

import torch

from ...language.inline_asm_ops import inline_asm_elementwise
from ..fx_side_effects import is_impure_node
from .analysis.match_common import node_tensor
from .analysis.native_s2f6_match import match_native_s2f6_decode
from .native_s2f6_contract import NATIVE_S2F6_ASM as _ASM
from .native_s2f6_contract import NATIVE_S2F6_TARGETS
from .native_s2f6_contract import native_s2f6_broadcast_matches as _broadcast_matches
from .selection.ordered_selection import is_ordered_selection

_CONVERT = torch.ops.prims.convert_element_type.default


def fuse_native_s2f6(graph: torch.fx.Graph, capability: tuple[int, int] | None) -> int:
    """Replace local proved decodes on SM100/SM103, retaining other targets.

    Only the low packed lane is used. Selecting the authored canonical NaN
    explicitly is necessary: the native instruction's NaN payload is not the
    source expression's bit contract. Existing int32 casts preserve the signed
    raw byte and unsigned scale; the instruction consumes their low bytes.
    """
    if capability not in NATIVE_S2F6_TARGETS:
        return 0
    # Ordered selection currently infers fragment shapes without traversing
    # inline assembly. Preserve its original producer until that owner can
    # prove the replacement's complete shape, even through intervening views.
    selection_inputs: set[torch.fx.Node] = set()
    pending = [n for n in graph.nodes if is_ordered_selection(n)]
    while pending:
        current = pending.pop()
        if current in selection_inputs:
            continue
        selection_inputs.add(current)
        pending.extend(current.all_input_nodes)
    changed = 0
    for node in list(graph.nodes):
        if node.target is not torch.ops.aten.view.dtype or node in selection_inputs:
            continue
        proof = match_native_s2f6_decode(node)
        if proof is None:
            continue
        raw, scale, value = (
            node_tensor(proof.raw),
            node_tensor(proof.scale),
            node_tensor(node),
        )
        assert raw is not None and scale is not None and value is not None
        if not _broadcast_matches(raw, scale, value):
            continue
        narrowed = node.args[0]
        # Keep observable aliases of the original int16 temporary intact. The
        # remaining arithmetic may have live users; it is not rewritten.
        if (
            not isinstance(narrowed, torch.fx.Node)
            or narrowed.target is not _CONVERT
            or set(narrowed.users) != {node}
        ):
            continue
        operands = [proof.raw_int32, proof.scale_int32]
        with graph.inserting_before(node):
            replacement = graph.call_function(
                inline_asm_elementwise,
                (_ASM, "=h,r,r", operands, torch.int16, True, 1),
            )
            replacement.meta = dict(narrowed.meta)
        node.args = (replacement, torch.bfloat16)
        changed += 1
    if changed:
        graph.eliminate_dead_code(is_impure_node=is_impure_node)
        graph.lint()
    return changed
