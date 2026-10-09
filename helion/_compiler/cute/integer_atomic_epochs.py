"""Publish private integer update epochs once after a uniform scalar loop."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import torch

from ...language import _tracing_ops
from ...language import atomic_ops
from ..device_ir import ForLoopGraphInfo
from .local_atomic import atomic_target_origins

if TYPE_CHECKING:
    from ..device_ir import GraphInfo


def integer_atomic_loop_graphs(graphs: list[GraphInfo]) -> frozenset[int]:
    """Structural candidates only; storage/physical effects are checked later."""
    origins = atomic_target_origins(graphs)
    result = set()
    for info in graphs:
        if not isinstance(info, ForLoopGraphInfo):
            continue
        updates = [n for n in info.graph.nodes if n.target is atomic_ops.atomic_add]
        if updates and all(
            n in origins
            and origins[n].target is not _tracing_ops._host_tensor
            and isinstance(origins[n].meta.get("val"), torch.Tensor)
            and origins[n].meta["val"].dtype == torch.int32
            and not n.users
            and n.args[3] == "relaxed"
            for n in updates
        ):
            result.add(info.graph_id)
    return frozenset(result)


def can_defer_integer_epoch(
    body: list[ast.AST], buffers: dict[str, torch.dtype], targets: set[str]
) -> bool:
    """Prove the emitted body has no staging WAR or observable update order.

    The caller retains the existing initialized/non-escaping local-allocation
    proof and uniform loop schedule. Only unused relaxed Int32 atomic writes
    may mutate memory here. All other shared buffers are read-only for the
    entire loop; target reads and escaped shared pointers are forbidden. There
    are no internal barriers, collectives, nested control transfers or opaque
    calls. The removed terminal barrier is emitted unconditionally immediately
    after the loop, before any consumer or storage reuse, including zero trips.

    Integer addition preserves the final modular value under arbitrary worker
    interleavings. Floating-point, returned, global and ordered atomics are not
    covered. Int64 atomic support is intentionally not introduced by this pass.
    """
    if not targets or any(buffers.get(name) != torch.int32 for name in targets):
        return False
    tree = ast.Module(body=cast("list[ast.stmt]", body), type_ignores=[])
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    allowed_buffer_names: set[ast.AST] = set()
    atomics: set[ast.AST] = set()
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Call)
            or ast.unparse(node.func) != "cute.arch.atomic_add"
        ):
            continue
        if not isinstance(parents.get(node), ast.Expr) or len(node.args) != 2:
            return False
        if {
            k.arg: ast.literal_eval(k.value)
            for k in node.keywords
            if isinstance(k.value, ast.Constant)
        } != {"sem": "relaxed", "scope": "cta"} or len(node.keywords) != 2:
            return False
        pointer = node.args[0]
        if not (
            isinstance(pointer, ast.Attribute)
            and pointer.attr == "llvm_ptr"
            and isinstance(pointer.value, ast.BinOp)
            and isinstance(pointer.value.op, ast.Add)
            and isinstance(pointer.value.left, ast.Attribute)
            and pointer.value.left.attr == "iterator"
            and isinstance(pointer.value.left.value, ast.Name)
            and pointer.value.left.value.id in targets
        ):
            return False
        allowed_buffer_names.add(pointer.value.left.value)
        atomics.add(node)
    if not atomics:
        return False
    casts = {
        f"cutlass.{name}"
        for name in (
            "Boolean",
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "Uint8",
            "Uint16",
            "Uint32",
            "Uint64",
            "Float16",
            "BFloat16",
            "Float32",
            "Float64",
        )
    }
    pure = {"range", "abs", "min", "max", "_cute_python_mod"} | {
        f"operator.{name}"
        for name in (
            "lt",
            "le",
            "eq",
            "ne",
            "gt",
            "ge",
            "and_",
            "or_",
            "xor",
            "neg",
            "add",
            "sub",
            "mul",
            "floordiv",
            "mod",
            "lshift",
            "rshift",
        )
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and not isinstance(
            node, (ast.Assign, ast.AugAssign, ast.For, ast.If, ast.Pass, ast.Expr)
        ):
            return False
        if isinstance(node, ast.Assign) and any(
            not isinstance(t, ast.Name) for t in node.targets
        ):
            return False
        if isinstance(node, ast.AugAssign) and not isinstance(node.target, ast.Name):
            return False
        if isinstance(node, ast.For) and (
            node.orelse or not isinstance(node.target, ast.Name)
        ):
            return False
        if isinstance(node, ast.Expr) and node.value not in atomics:
            return False
        if (
            isinstance(node, ast.Name)
            and node.id in buffers
            and node not in allowed_buffer_names
        ):
            parent = parents.get(node)
            if not (
                isinstance(parent, ast.Subscript)
                and parent.value is node
                and isinstance(parent.ctx, ast.Load)
                and node.id not in targets
            ):
                return False
        if isinstance(node, ast.Call) and node not in atomics:
            name = ast.unparse(node.func)
            if name in casts | pure:
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "bitcast"
                and len(node.args) == 1
                and not node.keywords
                and ast.unparse(node.args[0]) in casts
            ):
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
                and not node.args
                and not node.keywords
            ):
                address = node.func.value
                if (
                    isinstance(address, ast.BinOp)
                    and isinstance(address.op, ast.Add)
                    and isinstance(address.left, ast.Attribute)
                    and address.left.attr == "iterator"
                    and isinstance(address.left.value, ast.Name)
                    and address.left.value.id not in buffers
                ):
                    continue
            return False
    return True
