"""Lower an exact two-limb unsigned multiply to native 32-bit instructions.

The portable base-2**16 carry sequence computes the high and low words of
``(a & 0xffffffff) * (b & 0xffffffff)`` using int64 tensors. Recognize that
arithmetic, independently of its caller, and retain its int64 output contract
while performing the multiplication with ``mul.hi.u32`` / ``mul.lo.u32``.

All limbs are explicitly masked to 16 bits and every intermediate is int64.
Their products and carry sums fit in 33 bits, so the identity is exact even
when the original operands are negative or contain bits above bit 31. The
two outputs are matched independently because either one may already be dead.
"""

from __future__ import annotations

import dataclasses
from typing import cast

import torch

from ...language.inline_asm_ops import inline_asm_elementwise

_Value = int | torch.fx.Node
_MASK16 = (1 << 16) - 1
_MASK32 = (1 << 32) - 1
_ADD = torch.ops.aten.add.Tensor
_MUL = torch.ops.aten.mul.Tensor
_AND = (torch.ops.aten.bitwise_and.Scalar, torch.ops.aten.bitwise_and.Tensor)
_OR = (torch.ops.aten.bitwise_or.Scalar, torch.ops.aten.bitwise_or.Tensor)
_SHR = (
    torch.ops.aten.__rshift__.Scalar,
    torch.ops.aten.bitwise_right_shift.Tensor_Scalar,
)
_SHL = (
    torch.ops.aten.__lshift__.Scalar,
    torch.ops.aten.bitwise_left_shift.Tensor_Scalar,
)


def _is_int64(node: object) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    value = node.meta.get("val")
    return isinstance(value, torch.Tensor) and value.dtype is torch.int64


def _binary_args(
    node: object, targets: tuple[object, ...]
) -> tuple[_Value, _Value] | None:
    if (
        not isinstance(node, torch.fx.Node)
        or node.op != "call_function"
        or node.target not in targets
        or len(node.args) != 2
        or node.kwargs
        or not _is_int64(node)
    ):
        return None
    lhs, rhs = node.args
    if not all(
        type(arg) is int or isinstance(arg, torch.fx.Node) for arg in (lhs, rhs)
    ):
        return None
    return cast("tuple[_Value, _Value]", (lhs, rhs))


def _commutative_args(
    node: object, targets: tuple[object, ...]
) -> tuple[tuple[_Value, _Value], ...]:
    args = _binary_args(node, targets)
    return () if args is None else (args, (args[1], args[0]))


def _masked(node: object, mask: int) -> _Value | None:
    for value, constant in _commutative_args(node, _AND):
        if constant == mask:
            return value
    return None


def _shifted(node: object, targets: tuple[object, ...]) -> _Value | None:
    args = _binary_args(node, targets)
    return args[0] if args is not None and args[1] == 16 else None


def _operand_from_limbs(low: _Value, high: _Value) -> _Value | None:
    if type(low) is int and type(high) is int:
        if 0 <= low <= _MASK16 and 0 <= high <= _MASK16:
            return low | (high << 16)
        return None
    original = _masked(low, _MASK16)
    upper = _masked(high, _MASK16)
    if (
        original is not None
        and _is_int64(original)
        and _shifted(upper, _SHR) is original
    ):
        return original
    return None


@dataclasses.dataclass(frozen=True)
class _LimbProduct:
    a: _Value
    b: _Value
    a1: _Value
    b1: _Value
    t0: torch.fx.Node
    t1: torch.fx.Node
    t2: torch.fx.Node


def _match_carries(t2: object) -> _LimbProduct | None:
    # t0 = a0*b0; t1 = a1*b0 + (t0 >> 16);
    # t2 = a0*b1 + (t1 & 0xffff).
    for cross, low_t1 in _commutative_args(t2, (_ADD,)):
        t1 = _masked(low_t1, _MASK16)
        for product, carry in _commutative_args(t1, (_ADD,)):
            t0 = _shifted(carry, _SHR)
            for a0, b0 in _commutative_args(t0, (_MUL,)):
                for a1, other_b0 in _commutative_args(product, (_MUL,)):
                    if other_b0 != b0:
                        continue
                    for other_a0, b1 in _commutative_args(cross, (_MUL,)):
                        if other_a0 != a0:
                            continue
                        a = _operand_from_limbs(a0, a1)
                        b = _operand_from_limbs(b0, b1)
                        if a is not None and b is not None:
                            return _LimbProduct(
                                a,
                                b,
                                a1,
                                b1,
                                cast("torch.fx.Node", t0),
                                cast("torch.fx.Node", t1),
                                cast("torch.fx.Node", t2),
                            )
    return None


def _match_low(node: torch.fx.Node) -> _LimbProduct | None:
    for upper, lower in _commutative_args(_masked(node, _MASK32), _OR):
        t2 = _masked(_shifted(upper, _SHL), _MASK16)
        product = _match_carries(t2)
        if product is not None and _masked(lower, _MASK16) is product.t0:
            return product
    return None


def _three_addends(node: object) -> tuple[_Value, ...] | None:
    pending = [node]
    terms: list[_Value] = []
    while pending:
        current = pending.pop()
        args = _binary_args(current, (_ADD,))
        if args is not None:
            pending.extend(args)
        elif isinstance(current, torch.fx.Node) or type(current) is int:
            terms.append(current)
        else:
            return None
        if len(pending) + len(terms) > 3:
            return None
    return tuple(terms) if len(terms) == 3 else None


def _match_high(node: torch.fx.Node) -> _LimbProduct | None:
    terms = _three_addends(_masked(node, _MASK32))
    if terms is None:
        return None
    for index, term in enumerate(terms):
        product = _match_carries(_shifted(term, _SHR))
        if product is None:
            continue
        others = terms[:index] + terms[index + 1 :]
        for carry, upper_product in (others, (others[1], others[0])):
            if _shifted(carry, _SHR) is not product.t1:
                continue
            if (product.a1, product.b1) in _commutative_args(upper_product, (_MUL,)):
                return product
    return None


def _emit_native_product(
    graph: torch.fx.Graph,
    node: torch.fx.Node,
    product: _LimbProduct,
    word: str,
) -> None:
    args: list[torch.fx.Node] = []
    instructions = [".reg .b32 product;"]
    operands: list[str] = []
    for index, operand in enumerate((product.a, product.b)):
        if isinstance(operand, int):
            operands.append(str(operand))
        else:
            args.append(operand)
            name = f"operand{index}"
            instructions.extend(
                [
                    f".reg .b32 {name};",
                    f"cvt.u32.u64 {name}, ${len(args)};",
                ]
            )
            operands.append(name)
    instructions.extend(
        [
            f"mul.{word}.u32 product, {operands[0]}, {operands[1]};",
            "cvt.u64.u32 $0, product;",
        ]
    )
    asm = "{\n" + "\n".join(instructions) + "\n}"
    with graph.inserting_before(node):
        replacement = graph.call_function(
            inline_asm_elementwise,
            (asm, "=l" + ",l" * len(args), args, torch.int64, True, 1),
        )
        replacement.meta = dict(node.meta)
    node.replace_all_uses_with(replacement)


def fuse_u32_multiply(graph: torch.fx.Graph) -> int:
    """Replace proven int64 limb multiplication results, returning their count."""
    changed = 0
    for node in list(graph.nodes):
        if (product := _match_low(node)) is not None:
            _emit_native_product(graph, node, product, "lo")
        elif (product := _match_high(node)) is not None:
            _emit_native_product(graph, node, product, "hi")
        else:
            continue
        changed += 1
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
    return changed
