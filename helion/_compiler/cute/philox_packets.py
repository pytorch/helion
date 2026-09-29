"""Share one Philox evaluation across four proved consecutive logical lanes."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import TypeGuard
from typing import cast

from .philox_stream import PACKET_ASM
from .philox_stream import PACKET_CONSTRAINTS
from .philox_stream import SCALAR_ASM
from .philox_stream import SCALAR_CONSTRAINTS
from .scalar_recipe import _clone
from .scalar_recipe import _read_names

if TYPE_CHECKING:
    from collections.abc import Callable

_CASTS = {f"cutlass.{name}" for name in ("Int32", "Int64", "Uint32", "Uint64")}


def _integer(node: ast.expr, known: set[str]) -> bool:
    if isinstance(node, ast.Constant):
        return type(node.value) is int
    if isinstance(node, ast.Name):
        return node.id in known
    if isinstance(node, ast.Call):
        return (
            ast.unparse(node.func) in _CASTS
            and len(node.args) == 1
            and not node.keywords
            and _integer(node.args[0], known)
        )
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        return (
            type(node.slice.value) is int
            and 0 <= node.slice.value < 3
            and isinstance(node.value, ast.Call)
            and not node.value.args
            and not node.value.keywords
            and ast.unparse(node.value.func)
            in {"cute.arch.thread_idx", "cute.arch.block_idx"}
        )
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor)
    ):
        return _integer(node.left, known) and _integer(node.right, known)
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.UAdd, ast.USub, ast.Invert)
    ):
        return _integer(node.operand, known)
    return False


def _rng_call(node: ast.AST) -> TypeGuard[ast.Call]:
    if (
        not isinstance(node, ast.Call)
        or ast.unparse(node.func) != "_cute_inline_asm_elementwise"
    ):
        return False
    keywords = {keyword.arg: keyword.value for keyword in node.keywords}
    return (
        len(node.args) == 1
        and isinstance(node.args[0], ast.Tuple)
        and len(node.args[0].elts) == 2
        and set(keywords) == {"asm", "constraints", "dtype", "is_pure"}
        and ast.unparse(keywords["dtype"]) == "cutlass.Float32"
        and all(
            isinstance(constant := keywords[key], ast.Constant)
            and constant.value == value
            for key, value in (
                ("asm", SCALAR_ASM),
                ("constraints", SCALAR_CONSTRAINTS),
                ("is_pure", True),
            )
        )
    )


class _Substitute(ast.NodeTransformer):
    def __init__(self, values: dict[str, ast.expr]) -> None:
        self.values = values

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.values:
            return _clone(self.values[node.id])
        return node


class _ReplaceCall(ast.NodeTransformer):
    def __init__(self, expression: ast.expr) -> None:
        self.expression = expression

    def visit_Call(self, node: ast.Call) -> ast.AST:
        return _clone(self.expression) if _rng_call(node) else self.generic_visit(node)


def _aligned_packet_adjacency(expression: ast.expr, lane: str, known: set[str]) -> bool:
    """Prove adjacency conditional on the existing four-aligned base guard.

    Accept only integer casts around one invariant-plus-lane addition. Casts
    and integer promotion preserve the low two bits. Every signed/unsigned
    32/64-bit boundary is four-aligned, so a four-aligned packet cannot cross
    any such boundary internally. Width-eight loops use two separate guards.

    Do not move arithmetic through a cast: adding two *after* a widened
    Int32 wrap can align the final base while concealing an interior wrap.
    The typed zero/one identity below only recognizes the same explicit
    integer value on both sides; it never rewrites the emitted expression.
    """

    def integer_cast(node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) in _CASTS
            and len(node.args) == 1
            and not node.keywords
        )

    def scaled_cast(node: ast.expr, value: int) -> ast.expr | None:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            for constant, operand in ((node.left, node.right), (node.right, node.left)):
                if (
                    isinstance(constant, ast.Constant)
                    and type(constant.value) is int
                    and constant.value == value
                    and integer_cast(operand)
                ):
                    return operand
        return None

    # Multiplication by an int literal zero/one preserves an explicitly
    # typed Int32/Int64/Uint32/Uint64 operand's dtype under CuTe promotion.
    node = expression
    identity = scaled_cast(node, 1)
    if identity is not None:
        node = identity
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        for zero, one in ((node.left, node.right), (node.right, node.left)):
            left, right = scaled_cast(zero, 0), scaled_cast(one, 1)
            if (
                left is not None
                and right is not None
                and ast.dump(left) == ast.dump(right)
            ):
                node = left
                break
    while integer_cast(node):
        assert isinstance(node, ast.Call)
        node = node.args[0]
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
        return False
    for invariant, varying in ((node.left, node.right), (node.right, node.left)):
        while integer_cast(varying):
            assert isinstance(varying, ast.Call)
            varying = varying.args[0]
        if (
            isinstance(varying, ast.Name)
            and varying.id == lane
            and lane not in _read_names(invariant)
            and _integer(invariant, known)
        ):
            return True
    return False


def lower_philox_packets(
    body: list[ast.stmt],
    *,
    seed_names: set[str],
    integer_names: set[str],
    new_name: Callable[[str], str],
    vectorize_packet: Callable[[ast.For, ast.Subscript, frozenset[str]], list[ast.stmt]]
    | None = None,
) -> list[ast.stmt]:
    """Leave the scalar new stream unchanged whenever a packet proof declines.

    Only immutable compiler-owned seed tensors may move their read. Integer
    replay excludes loads, divisions, shifts and conversions from unknown
    scalar types. A narrow cast/add proof can remove redundant equalities;
    other expressions retain runtime checks for narrow-integer wrap behavior.
    The optional vectorizer receives a freshly constructed register-tuple read
    and the dominating integer-kind facts. It must prove memory alignment,
    aliasing, address and mask legality independently; declining it retains
    this same packet and scalar I/O.
    """

    def attempt(loop: ast.For, known: set[str]) -> list[ast.stmt] | None:
        if (
            not isinstance(loop.target, ast.Name)
            or loop.orelse
            or not isinstance(loop.iter, ast.Call)
            or ast.unparse(loop.iter.func) != "cutlass.range_constexpr"
            or loop.iter.keywords
            or len(loop.iter.args) != 1
            or not isinstance(loop.iter.args[0], ast.Constant)
            or type(loop.iter.args[0].value) is not int
            or loop.iter.args[0].value not in (4, 8)
        ):
            return None
        calls = [node for node in ast.walk(loop) if _rng_call(node)]
        if len(calls) != 1:
            return None
        call = calls[0]
        seed, offset = cast("ast.Tuple", call.args[0]).elts
        if not (
            isinstance(seed, ast.Subscript)
            and isinstance(seed.value, ast.Name)
            and seed.value.id in seed_names
            and isinstance(seed.slice, ast.Constant)
            and seed.slice.value == 0
        ):
            return None
        lane = loop.target.id
        definitions: dict[str, ast.expr] = {}
        available = known | {lane}
        found = False
        for statement in loop.body:
            if (
                not isinstance(statement, ast.Assign)
                or len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
            ):
                break
            if statement.value is call:
                found = True
                break
            name = statement.targets[0].id
            expression = _Substitute(definitions).visit(_clone(statement.value))
            if _integer(expression, available):
                definitions[name] = expression
            else:
                definitions.pop(name, None)
            # An earlier body write is not an invariant external input.
            available.discard(name)
        if not found:
            return None
        expanded = _Substitute(definitions).visit(_clone(offset))
        if not _integer(expanded, known | {lane}):
            return None
        # An invariant used in replay cannot be changed anywhere in the loop.
        writes = {
            node.id
            for node in ast.walk(loop)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        if (_read_names(expanded) - {lane}) & writes:
            return None
        aligned_adjacency = _aligned_packet_adjacency(expanded, lane, known)
        width = loop.iter.args[0].value
        offsets = [
            _Substitute({lane: ast.Constant(index)}).visit(_clone(expanded))
            for index in range(width)
        ]
        preamble: list[ast.stmt] = []
        values: list[ast.expr] = []
        guards: list[ast.expr] = []
        for start in range(0, width, 4):
            base = offsets[start]
            guards.append(
                ast.Compare(
                    ast.BinOp(_clone(base), ast.BitAnd(), ast.Constant(3)),
                    [ast.Eq()],
                    [ast.Constant(0)],
                )
            )
            for delta in () if aligned_adjacency else range(1, 4):
                guards.append(
                    ast.Compare(
                        _clone(offsets[start + delta]),
                        [ast.Eq()],
                        [ast.BinOp(_clone(base), ast.Add(), ast.Constant(delta))],
                    )
                )
            name = new_name("philox_packet")
            packet = _clone(call)
            packet.args[0] = ast.Tuple([_clone(seed), _clone(base)], ast.Load())
            for keyword in packet.keywords:
                if keyword.arg == "asm":
                    keyword.value = ast.Constant(PACKET_ASM)
                elif keyword.arg == "constraints":
                    keyword.value = ast.Constant(PACKET_CONSTRAINTS)
                elif keyword.arg == "dtype":
                    keyword.value = ast.Tuple(
                        [
                            ast.parse("cutlass.Float32", mode="eval").body
                            for _ in range(4)
                        ],
                        ast.Load(),
                    )
            preamble.append(ast.Assign([ast.Name(name, ast.Store())], packet))
            values.extend(
                ast.Subscript(
                    ast.Name(name, ast.Load()), ast.Constant(index), ast.Load()
                )
                for index in range(4)
            )
        name = new_name("philox_lanes")
        preamble.append(
            ast.Assign([ast.Name(name, ast.Store())], ast.Tuple(values, ast.Load()))
        )
        register_read = ast.Subscript(
            ast.Name(name, ast.Load()), ast.Name(lane, ast.Load()), ast.Load()
        )
        fast = _ReplaceCall(register_read).visit(_clone(loop))
        assert isinstance(fast, ast.For)
        # new_name cannot alias an original binding. The tuple contains only
        # scalar RNG results, is defined immediately before this loop and is
        # indexed by its bounded constexpr induction variable. Grant exactly
        # that Load expression, never arbitrary named tensor subscripts.
        fast_body = (
            vectorize_packet(fast, register_read, frozenset(known))
            if vectorize_packet is not None
            else [fast]
        )
        return [
            ast.If(
                guards[0] if len(guards) == 1 else ast.BoolOp(ast.And(), guards),
                [*preamble, *fast_body],
                [_clone(loop)],
            )
        ]

    def walk(statements: list[ast.stmt], known: set[str]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        known = set(known)
        for statement in statements:
            if isinstance(statement, ast.For):
                replacement = attempt(statement, known)
                if replacement is not None:
                    result.extend(replacement)
                    known.difference_update(
                        node.id
                        for node in ast.walk(statement)
                        if isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Store)
                    )
                    continue
                nested = set(known)
                if (
                    isinstance(statement.target, ast.Name)
                    and isinstance(statement.iter, ast.Call)
                    and ast.unparse(statement.iter.func)
                    in {"range", "cutlass.range", "cutlass.range_constexpr"}
                ):
                    nested.add(statement.target.id)
                statement.body = walk(statement.body, nested)
            elif isinstance(statement, ast.If):
                statement.body = walk(statement.body, known)
                statement.orelse = walk(statement.orelse, known)
            if not isinstance(statement, ast.Assign):
                # Control-flow and other writes invalidate earlier scalar-kind facts.
                known.difference_update(
                    node.id
                    for node in ast.walk(statement)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                )
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        if _integer(statement.value, known):
                            known.add(target.id)
                        else:
                            known.discard(target.id)
            result.append(statement)
        return result

    return walk(body, integer_names)
