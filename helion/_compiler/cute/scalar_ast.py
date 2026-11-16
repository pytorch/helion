"""Common scalar facts for emitted shared-storage schedule proofs."""

from __future__ import annotations

import ast

SCALAR_CAST_CALLS = frozenset(
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
)

# The intersection shared by the arena, integer-epoch and private-scalar
# proofs. Each caller keeps its own additional capabilities and effects.
SCALAR_CALLS = frozenset({"min", "max", "_cute_python_mod"}) | frozenset(
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
)


def _scalar_index(node: ast.AST) -> bool:
    # Fragment emitters flatten coordinates into one integer expression.
    # Tuples (including nested ones), slices, None and Ellipsis instead select
    # views; never accept them as evidence of a completed scalar access.
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Constant):
        return type(node.value) is int
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)) and _scalar_index(
            node.operand
        )
    if isinstance(node, ast.BinOp):
        return (
            isinstance(
                node.op,
                (
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.FloorDiv,
                    ast.Mod,
                    ast.LShift,
                    ast.RShift,
                    ast.BitAnd,
                    ast.BitOr,
                    ast.BitXor,
                ),
            )
            and _scalar_index(node.left)
            and _scalar_index(node.right)
        )
    if isinstance(node, ast.Subscript):
        # A bounded gather may load its integer coordinate from another flat
        # buffer. Every nested access obeys this same scalar contract, and
        # callers still check that buffer's complete name-use/escape proof.
        return (
            isinstance(node.value, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and _scalar_index(node.slice)
        )
    if isinstance(node, ast.Call):
        return (
            ast.unparse(node.func) in SCALAR_CALLS | SCALAR_CAST_CALLS | {"int", "abs"}
            and not node.keywords
            and all(_scalar_index(arg) for arg in node.args)
        )
    return False


def is_flat_scalar_access(parent: ast.AST | None, name: ast.Name) -> bool:
    """The only nonescaping access form emitted for flat fragment buffers."""
    return (
        isinstance(parent, ast.Subscript)
        and parent.value is name
        and isinstance(parent.ctx, (ast.Load, ast.Store))
        and _scalar_index(parent.slice)
    )
