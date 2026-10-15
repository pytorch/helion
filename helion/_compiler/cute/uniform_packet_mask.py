"""Uniform packet bounds in a proved register-fragment loading sweep."""

from __future__ import annotations

import ast
from collections import Counter
import math
from typing import cast

from . import cluster_online_pair as pair
from . import rolled_cluster_online_pair as rolled
from .memory_ops import _cute_lane_uniform_mask
from .memory_ops import _cute_tile_unroll_uniform_definitions
from .memory_ops import _CuteLaneIndexFacts
from .memory_ops import _CuteVloopScope

_INTEGER_CASTS = {
    "cutlass.Int32",
    "cutlass.Int64",
    "cutlass.Uint32",
    "cutlass.Uint64",
}


def integer_alignment(node: ast.expr, known: dict[str, int]) -> int | None:
    """The expression's integer multiple; zero denotes the integer zero.

    Integer casts retain only already-proved integer arithmetic. A float
    expression truncated after multiplication is not assumed aligned.
    """
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return abs(node.value)
    if isinstance(node, ast.Name):
        return known.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return integer_alignment(node.operand, known)
    if isinstance(node, ast.Call) and not node.keywords and len(node.args) == 1:
        if ast.unparse(node.func) in _INTEGER_CASTS:
            return integer_alignment(node.args[0], known)
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func)
        in (
            "cute.arch.thread_idx",
            "cute.arch.block_idx",
            "cute.arch.block_dim",
        )
    ):
        return 1
    if isinstance(node, ast.BinOp):
        left = integer_alignment(node.left, known)
        right = integer_alignment(node.right, known)
        if left is None or right is None:
            return None
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return math.gcd(left, right)
        if isinstance(node.op, ast.Mult):
            return left * right
    return None


def loop_alignment(loop: ast.For, known: dict[str, int]) -> int | None:
    iterator = loop.iter
    if not (
        isinstance(iterator, ast.Call)
        and ast.unparse(iterator.func) in ("range", "cutlass.range_constexpr")
        and not iterator.keywords
    ):
        return None
    if len(iterator.args) == 1:
        return 1
    if len(iterator.args) in (2, 3):
        start = integer_alignment(iterator.args[0], known)
        step = (
            integer_alignment(iterator.args[2], known) if len(iterator.args) == 3 else 1
        )
        if start is not None and step is not None:
            return math.gcd(start, step)
    return None


class _AssumeTrue(ast.NodeTransformer):
    def __init__(self, mask: str) -> None:
        self.mask = mask

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.mask and isinstance(node.ctx, ast.Load):
            return ast.Constant(True)
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.And):
            if any(
                isinstance(value, ast.Constant) and value.value is False
                for value in node.values
            ):
                return ast.Constant(False)
            node.values = [
                value
                for value in node.values
                if not (isinstance(value, ast.Constant) and value.value is True)
            ]
            return (
                node.values[0]
                if len(node.values) == 1
                else node
                if node.values
                else ast.Constant(True)
            )
        return node

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and type(node.test.value) is bool:
            return node.body if node.test.value else node.orelse
        return node


def uniform_packet_mask(
    prefix: list[ast.stmt],
    loop: ast.For,
    width: int,
    row: str,
    known: dict[str, int],
) -> list[ast.stmt] | None:
    """Branch once for a packet's reduction mask, preserving false -inf slots.

    The caller proved a pure max-load sweep into ``row``. We additionally
    prove ``index = aligned_base + element`` and use the ordinary CuTe
    predicate proof to show the last store's mask holds one value for all
    elements. Dynamic or unaligned bounds and per-element predicates retain
    the original loading sweep.
    """
    assert isinstance(loop.target, ast.Name)
    if not loop.body or not isinstance(loop.body[-1], ast.Assign):
        return None
    store = loop.body[-1]
    if not (
        len(store.targets) == 1
        and isinstance(store.targets[0], ast.Subscript)
        and isinstance(store.targets[0].value, ast.Name)
        and store.targets[0].value.id == row
    ):
        return None
    domain, _value = rolled._domain(store.value, -math.inf)
    if not isinstance(domain, ast.Name):
        return None
    masks = [
        statement
        for statement in loop.body[:-1]
        if pair._is_name_assign(statement)
        and pair._assigned_name(statement) == domain.id
    ]
    if len(masks) != 1:
        return None
    mask = masks[0]
    indices = [
        statement
        for statement in loop.body[:-1]
        if pair._is_name_assign(statement)
        and isinstance(statement.value, ast.BinOp)
        and isinstance(statement.value.op, ast.Add)
        and isinstance(statement.value.left, ast.Name)
        and ast.unparse(statement.value.right) == f"cutlass.Int32({loop.target.id})"
    ]
    if len(indices) != 1:
        return None
    index = indices[0]
    base = cast("ast.Name", cast("ast.BinOp", index.value).left).id
    alignments = dict(known)
    for statement in prefix:
        alignment = (
            integer_alignment(statement.value, alignments)
            if pair._is_name_assign(statement)
            else None
        )
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                alignments.pop(node.id, None)
        if pair._is_name_assign(statement):
            name = pair._assigned_name(statement)
            if alignment is not None:
                alignments[name] = alignment
    if base not in alignments or alignments[base] % width:
        return None
    writes = Counter(
        node.id
        for statement in [*prefix, *loop.body]
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    constants = [
        statement
        for statement in prefix
        if pair._is_name_assign(statement)
        and isinstance(statement.value, ast.Constant)
        and type(statement.value.value) is int
        and writes[pair._assigned_name(statement)] == 1
    ]
    scope = _CuteVloopScope(
        [*constants, *loop.body],
        frozenset((pair._assigned_name(index), loop.target.id)),
        [],
        [loop],
        loop,
        _CuteLaneIndexFacts(
            pair._assigned_name(index), base, loop.target.id, width, True
        ),
    )
    expression = _cute_lane_uniform_mask(scope, ast.unparse(mask.value))
    if expression is None:
        return None
    proof = _cute_tile_unroll_uniform_definitions(
        scope, [pair._assigned_name(index)], 0, expression
    )
    if proof is None or any(statement is mask for statement in proof.moved):
        return None
    moved = [
        cast("ast.stmt", statement)
        for statement in proof.moved
        if statement in loop.body
    ]
    true_body = [
        cast(
            "ast.stmt",
            _AssumeTrue(domain.id).visit(pair._stmt(ast.unparse(statement))),
        )
        for statement in loop.body
        if statement is not mask and statement not in moved
    ]
    true_loop = pair._stmt(ast.unparse(loop))
    assert isinstance(true_loop, ast.For)
    true_loop.body = true_body
    false_loop = pair._stmt(ast.unparse(loop))
    assert isinstance(false_loop, ast.For)
    false_loop.body = [
        pair._stmt(f"{ast.unparse(store.targets[0])} = cutlass.Float32(float('-inf'))")
    ]
    branch = pair._stmt(f"if {domain.id}:\n    pass\nelse:\n    pass")
    assert isinstance(branch, ast.If)
    branch.body = [true_loop]
    branch.orelse = [false_loop]
    return [
        *moved,
        pair._stmt(f"{domain.id} = {expression}"),
        branch,
    ]
