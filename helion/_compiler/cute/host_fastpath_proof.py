"""Small nonnegative integer/congruence proof for host-guarded copy branches.

Only complete true conditions are erased. Unknown math/control flow retains the
original branch. No floating-point expression or memory operation is evaluated.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from math import gcd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

LIMIT = 2**31 - 1


@dataclass(frozen=True)
class Integer:
    low: int
    high: int
    multiple: int


@dataclass(frozen=True)
class Pointer:
    tensor: str
    offset: Integer


@dataclass(frozen=True)
class Tensor:
    name: str


@dataclass(frozen=True)
class RegisterTensor:
    size: int


def integer(low: int, high: int, multiple: int = 1) -> Integer | None:
    if 0 <= low <= high <= LIMIT:
        return Integer(low, high, abs(low) if low == high else multiple)
    return None


def path(node: ast.AST) -> tuple[str, ...] | None:
    result = []
    while isinstance(node, ast.Attribute):
        result.append(node.attr)
        node = node.value
    return tuple([node.id, *reversed(result)]) if isinstance(node, ast.Name) else None


def assigned(node: ast.AST) -> set[str]:
    result = {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
    }
    result.update(
        n.asname or n.name.split(".")[0]
        for n in ast.walk(node)
        if isinstance(n, ast.alias)
    )
    return result


@dataclass
class Proof:
    tensors: dict[str, tuple[tuple[int, ...], tuple[int, ...], str, int]]
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    removed: int = 0

    def value(
        self,
        node: ast.AST,
        values: Mapping[str, Integer | Pointer | Tensor | RegisterTensor],
    ) -> Integer | Pointer | Tensor | RegisterTensor | None:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return integer(node.value, node.value)
        if isinstance(node, ast.Name):
            return Tensor(node.id) if node.id in self.tensors else values.get(node.id)
        if isinstance(node, ast.Attribute) and node.attr == "iterator":
            tensor = self.value(node.value, values)
            if isinstance(tensor, Tensor):
                return Pointer(tensor.name, Integer(0, 0, 0))
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and type(node.slice.value) is int
        ):
            axis = node.slice.value
            p = path(node.value)
            if p and len(p) == 3 and p[1] == "layout" and p[2] in ("shape", "stride"):
                tensor = self.value(ast.Name(id=p[0], ctx=ast.Load()), values)
                if isinstance(tensor, Tensor):
                    data = (
                        self.tensors[tensor.name][0]
                        if p[2] == "shape"
                        else self.tensors[tensor.name][1]
                    )
                    return (
                        integer(data[axis], data[axis])
                        if 0 <= axis < len(data)
                        else None
                    )
            if (
                isinstance(node.value, ast.Call)
                and not node.value.args
                and not node.value.keywords
                and 0 <= axis < 3
            ):
                p = path(node.value.func)
                if p in (("cute", "arch", "thread_idx"), ("cute", "arch", "block_idx")):
                    assert p is not None
                    bound = self.block if p[-1] == "thread_idx" else self.grid
                    return integer(0, bound[axis] - 1)
        if isinstance(node, ast.Call):
            if (
                path(node.func) == ("cute", "make_rmem_tensor")
                and len(node.args) == 2
                and not node.keywords
                and isinstance(node.args[0], ast.Tuple)
                and len(node.args[0].elts) == 1
                and path(node.args[1])
                in {
                    ("cutlass", "Float32"),
                    ("cutlass", "Float16"),
                    ("cutlass", "BFloat16"),
                }
            ):
                size = self.value(node.args[0].elts[0], values)
                if isinstance(size, Integer) and size.low == size.high > 0:
                    return RegisterTensor(size.low)
            if (
                path(node.func)
                in {
                    ("cutlass", "Int32"),
                    ("cutlass", "Int64"),
                    ("cutlass", "Uint32"),
                    ("cutlass", "Uint64"),
                }
                and len(node.args) == 1
                and not node.keywords
            ):
                result = self.value(node.args[0], values)
                return result if isinstance(result, Integer) else None
            return None
        if not isinstance(node, ast.BinOp):
            return None
        # Never model an address as a bounded integer. Only modular alignment.
        if (
            isinstance(node.op, ast.Mod)
            and isinstance(node.left, ast.Call)
            and isinstance(node.left.func, ast.Attribute)
            and node.left.func.attr == "toint"
            and not node.left.args
            and not node.left.keywords
        ):
            pointer = self.value(node.left.func.value, values)
            divisor = self.value(node.right, values)
            if (
                isinstance(pointer, Pointer)
                and isinstance(divisor, Integer)
                and divisor.low == divisor.high > 0
            ):
                element = self.tensors[pointer.tensor][3]
                if (
                    16 % divisor.low == 0
                    and pointer.offset.multiple * element % divisor.low == 0
                ):
                    return Integer(0, 0, 0)
            return None
        left, right = self.value(node.left, values), self.value(node.right, values)
        if isinstance(node.op, ast.Add):
            if isinstance(left, Pointer) and isinstance(right, Integer):
                offset = integer(
                    left.offset.low + right.low,
                    left.offset.high + right.high,
                    gcd(left.offset.multiple, right.multiple),
                )
                return Pointer(left.tensor, offset) if offset is not None else None
            if isinstance(right, Pointer) and isinstance(left, Integer):
                return self.value(
                    ast.BinOp(left=node.right, op=ast.Add(), right=node.left), values
                )
        if not isinstance(left, Integer) or not isinstance(right, Integer):
            return None
        if isinstance(node.op, ast.Add):
            return integer(
                left.low + right.low,
                left.high + right.high,
                gcd(left.multiple, right.multiple),
            )
        if isinstance(node.op, ast.Sub):
            return integer(
                left.low - right.high,
                left.high - right.low,
                gcd(left.multiple, right.multiple),
            )
        if isinstance(node.op, ast.Mult):
            return integer(
                left.low * right.low,
                left.high * right.high,
                left.multiple * right.multiple,
            )
        if right.low == right.high > 0:
            if isinstance(node.op, ast.FloorDiv):
                multiple = (
                    left.multiple // right.low if left.multiple % right.low == 0 else 1
                )
                return integer(left.low // right.low, left.high // right.low, multiple)
            if isinstance(node.op, ast.Mod):
                if left.low == left.high:
                    remainder = left.low % right.low
                    return integer(remainder, remainder)
                return integer(
                    0, min(left.high, right.low - 1), gcd(left.multiple, right.low)
                )
        return None

    def true(
        self,
        node: ast.AST,
        values: Mapping[str, Integer | Pointer | Tensor | RegisterTensor],
    ) -> bool:
        if isinstance(node, ast.Constant):
            return node.value is True
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            return all(self.true(n, values) for n in node.values)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            return self.true(node.left, values) and self.true(node.right, values)
        if not isinstance(node, ast.Compare):
            return False
        parts = [node.left, *node.comparators]
        for a, op, b in zip(parts, node.ops, parts[1:], strict=False):
            left, right = self.value(a, values), self.value(b, values)
            if not isinstance(left, Integer) or not isinstance(right, Integer):
                return False
            if isinstance(op, ast.Eq):
                ok = left.low == left.high == right.low == right.high
            elif isinstance(op, ast.Lt):
                ok = left.high < right.low
            elif isinstance(op, ast.LtE):
                ok = left.high <= right.low
            elif isinstance(op, ast.Gt):
                ok = left.low > right.high
            elif isinstance(op, ast.GtE):
                ok = left.low >= right.high
            else:
                return False
            if not ok:
                return False
        return bool(node.ops)

    def interval(
        self,
        node: ast.AST,
        values: Mapping[str, Integer | Pointer | Tensor | RegisterTensor],
    ) -> Integer | None:
        if (
            not isinstance(node, ast.Call)
            or path(node.func)
            not in {("range",), ("cutlass", "range"), ("cutlass", "range_constexpr")}
            or not 1 <= len(node.args) <= 3
        ):
            return None
        if any(
            k.arg != "unroll"
            or not isinstance(k.value, ast.Constant)
            or type(k.value.value) is not int
            for k in node.keywords
        ):
            return None
        args = [self.value(arg, values) for arg in node.args]
        if not all(isinstance(v, Integer) and v.low == v.high for v in args):
            return None
        exact = [v.low for v in args if isinstance(v, Integer)]
        start, stop, step = (
            (0, exact[0], 1)
            if len(exact) == 1
            else (exact[0], exact[1], exact[2] if len(exact) == 3 else 1)
        )
        if stop <= start or step <= 0:
            return None
        last = start + (stop - start - 1) // step * step
        return integer(start, last, gcd(start, step))

    def block_body(
        self,
        body: list[ast.stmt],
        values: dict[str, Integer | Pointer | Tensor | RegisterTensor],
    ) -> list[ast.stmt]:
        result = []
        for stmt in body:
            assignment_inputs = values
            if isinstance(stmt, ast.Assign):
                # A simple target may read its old value on the RHS, but nested
                # assignments must never supply stale facts to that evaluation.
                assignment_inputs = dict(values)
                expressions = [stmt.value]
                expressions.extend(
                    target
                    for target in stmt.targets
                    if not isinstance(target, ast.Name)
                )
                for expression in expressions:
                    for name in assigned(expression):
                        assignment_inputs.pop(name, None)
                # Invalidate all writes, including nested RHS/target writes,
                # before either the ordinary or local-register-store handler.
                for name in assigned(stmt):
                    values.pop(name, None)
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                name = stmt.targets[0].id
                val = self.value(stmt.value, assignment_inputs)
                if val is not None:
                    values[name] = val
            elif (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Subscript)
                and isinstance(stmt.targets[0].value, ast.Name)
            ):
                # A bounded data write to a positively identified local register
                # tensor does not rebind integer/pointer locals or alter launch
                # tensor metadata. Never model values read back from this data.
                target = stmt.targets[0]
                tensor = self.value(target.value, values)
                index = self.value(target.slice, values)
                if not (
                    isinstance(tensor, RegisterTensor)
                    and isinstance(index, Integer)
                    and 0 <= index.low <= index.high < tensor.size
                    and not assigned(stmt.value)
                ):
                    values.clear()
            elif isinstance(stmt, ast.If):
                # The test executes before either branch. Even a conditionally
                # evaluated nested write invalidates its old proof fact.
                for name in assigned(stmt.test):
                    values.pop(name, None)
                # Only vector-copy branches, never scalar causal/FP conditionals.
                guard = any(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "toint"
                    for n in ast.walk(stmt.test)
                )
                copy = any(
                    isinstance(n, ast.Call)
                    and path(n.func) in {("cute", "copy"), ("cute", "autovec_copy")}
                    for n in ast.walk(ast.Module(body=stmt.body, type_ignores=[]))
                )
                if stmt.orelse and guard and copy and self.true(stmt.test, values):
                    self.removed += 1
                    result.extend(self.block_body(stmt.body, values))
                    continue
                stmt.body = self.block_body(stmt.body, dict(values))
                stmt.orelse = self.block_body(stmt.orelse, dict(values))
                for name in assigned(stmt):
                    values.pop(name, None)
            elif isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
                loop_values = {
                    k: v for k, v in values.items() if k not in assigned(stmt)
                }
                bound = self.interval(stmt.iter, values)
                if bound is not None:
                    loop_values[stmt.target.id] = bound
                stmt.body = self.block_body(stmt.body, loop_values)
                stmt.orelse = self.block_body(stmt.orelse, {})
                for name in assigned(stmt):
                    values.pop(name, None)
            elif isinstance(stmt, ast.Expr):
                for name in assigned(stmt):
                    values.pop(name, None)
            elif isinstance(stmt, ast.Pass):
                pass
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                for name in stmt.names:
                    if name.name == "*":
                        values.clear()
                    else:
                        values.pop(name.asname or name.name.split(".")[0], None)
            else:
                # Unknown control flow can assign names or have nonlocal effects.
                values.clear()
            result.append(stmt)
        return result
