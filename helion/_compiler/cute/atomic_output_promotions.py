"""Prove private host storage before promoting half atomic outputs to FP32."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    import torch


def fresh_half_atomic_outputs(
    body: Sequence[ast.stmt], promotions: Mapping[str, torch.dtype]
) -> dict[str, torch.dtype]:
    """Require a fresh allocation whose host references cannot escape as aliases."""
    factories = {
        f"torch.{name}"
        for name in (
            "empty",
            "empty_like",
            "zeros",
            "zeros_like",
            "ones",
            "ones_like",
            "full",
            "full_like",
        )
    }
    module = ast.Module(body=list(body), type_ignores=[])
    parents = {
        id(child): parent
        for parent in ast.walk(module)
        for child in ast.iter_child_nodes(parent)
    }
    bindings: dict[str, list[ast.Name]] = {name: [] for name in promotions}
    escaped: set[str] = set()
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in promotions
        ):
            parent = parents[id(node)]
            ancestor = parent
            captured = False
            while ancestor is not module:
                if isinstance(
                    ancestor,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef),
                ):
                    captured = True
                    break
                ancestor = parents[id(ancestor)]
            if captured:
                escaped.add(node.id)
                continue
            if isinstance(parent, ast.Attribute) and parent.attr in {
                "dtype",
                "device",
                "shape",
                "ndim",
            }:
                continue
            if isinstance(parent, ast.Attribute) and parent.attr in {
                "size",
                "stride",
                "numel",
                "element_size",
            }:
                call = parents[id(parent)]
                # A bound method retains its Tensor receiver in __self__.
                # Only an immediate metadata call prevents that receiver from
                # escaping under another name.
                if isinstance(call, ast.Call) and call.func is parent:
                    continue
            if (
                isinstance(parent, ast.Call)
                and ast.unparse(parent.func)
                in {"hl.atomic_add", "helion.language.atomic_add"}
                and parent.args
                and parent.args[0] is node
            ):
                continue
            # Returning the destination after execution does not make its
            # contents available to the device under another host binding.
            while isinstance(parent, (ast.Tuple, ast.List)):
                parent = parents[id(parent)]
            if not isinstance(parent, ast.Return):
                escaped.add(node.id)
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id in bindings
        ):
            bindings[node.id].append(node)
    result = {}
    for assignment in ast.walk(module):
        if (
            not isinstance(assignment, ast.Assign)
            or len(assignment.targets) != 1
            or not isinstance(assignment.targets[0], ast.Name)
        ):
            continue
        target = assignment.targets[0]
        if (
            bindings.get(target.id) == [target]
            and target.id not in escaped
            and isinstance(assignment.value, ast.Call)
            and ast.unparse(assignment.value.func) in factories
            and all(
                keyword.arg not in (None, "out")
                for keyword in assignment.value.keywords
            )
        ):
            result[target.id] = promotions[target.id]
    return result
