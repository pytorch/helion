"""Guarded per-thread vector loads with exact, vector-uniform outer indices.

Unlike a whole-tile affine copy, this proof only describes consecutive inner
elements. A row may contain captures and fixed-width arithmetic: those source
operations are retained verbatim, not simplified with unbounded integers.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence
    from collections.abc import Set as AbstractSet


_DTYPES = {
    torch.bfloat16: ("cutlass.BFloat16", 2),
    torch.float16: ("cutlass.Float16", 2),
    torch.float32: ("cutlass.Float32", 4),
}
_INTEGER_CASTS = {"cutlass.Int32", "cutlass.Int64"}
_COMPARISONS = {
    "operator.eq",
    "operator.ne",
    "operator.lt",
    "operator.le",
    "operator.gt",
    "operator.ge",
}


class _NotProven(Exception):
    pass


@dataclass(frozen=True)
class VectorLeafPlan:
    """Source expressions and guards; no FX or frontend representation."""

    indices: tuple[str, ...]
    element: str
    axis: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: str
    element_bytes: int
    mask: str | None
    width: int = 8


@dataclass(frozen=True)
class VectorLeafEmission:
    lines: tuple[str, ...]
    values: str
    vectorized: str


@dataclass(frozen=True)
class DenseVectorSink:
    """A dense shared view established by the original allocation binder.

    This describes destination coordinates only; it grants no source-load,
    alias/lifetime, or synchronization authority. The per-vector pointer still
    receives a runtime alignment guard before the async copy.
    """

    target: str
    shape: tuple[int, int]
    dtype: str

    def pointer(self, row: str, column: str) -> str:
        return f"{self.target}.iterator + ({row}) * {self.shape[1]} + ({column})"


def _expand(
    text: str,
    definitions: Mapping[str, str],
    element: str,
    uniform_names: AbstractSet[str],
    active: frozenset[str] = frozenset(),
) -> ast.expr:
    """Inline definitions before trusting a name's vector-uniform annotation."""

    def visit(node: ast.expr) -> ast.expr:
        if isinstance(node, ast.Constant) and type(node.value) in (int, bool):
            return node
        if isinstance(node, ast.Name):
            if node.id in definitions:
                if node.id in active:
                    raise _NotProven
                return _expand(
                    definitions[node.id],
                    definitions,
                    element,
                    uniform_names,
                    active | {node.id},
                )
            if node.id == element or node.id in uniform_names:
                return node
            raise _NotProven
        if isinstance(node, ast.BinOp) and isinstance(
            node.op,
            (
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.FloorDiv,
                ast.Mod,
                ast.BitAnd,
                ast.BitOr,
                ast.BitXor,
                ast.LShift,
                ast.RShift,
            ),
        ):
            return ast.BinOp(visit(node.left), node.op, visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub, ast.Invert, ast.Not)
        ):
            return ast.UnaryOp(node.op, visit(node.operand))
        if isinstance(node, ast.BoolOp):
            return ast.BoolOp(node.op, [visit(value) for value in node.values])
        if isinstance(node, ast.Compare):
            return ast.Compare(
                visit(node.left), node.ops, [visit(value) for value in node.comparators]
            )
        if isinstance(node, ast.IfExp):
            return ast.IfExp(visit(node.test), visit(node.body), visit(node.orelse))
        if isinstance(node, ast.Call) and not node.keywords:
            function = ast.unparse(node.func)
            count = 1 if function in _INTEGER_CASTS else 2
            if function in _INTEGER_CASTS | _COMPARISONS and len(node.args) == count:
                return ast.Call(node.func, [visit(value) for value in node.args], [])
        # In particular, a load/subscript is not a uniform scalar just because
        # it has no explicit element name. The caller must capture it first.
        raise _NotProven

    return visit(ast.parse(text, mode="eval").body)


def _varies(node: ast.AST, element: str) -> bool:
    return any(
        isinstance(value, ast.Name) and value.id == element for value in ast.walk(node)
    )


def _unit_step(node: ast.expr, element: str) -> bool:
    """No reassociation/cancellation, and no varying casts or divisions."""
    if isinstance(node, ast.Name):
        return node.id == element
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _unit_step(node.operand, element)
    if isinstance(node, ast.BinOp):
        left, right = _varies(node.left, element), _varies(node.right, element)
        if isinstance(node.op, (ast.Add, ast.Sub)) and left and not right:
            return _unit_step(node.left, element)
        if isinstance(node.op, ast.Add) and right and not left:
            return _unit_step(node.right, element)
        if isinstance(node.op, ast.Mult):
            if isinstance(node.right, ast.Constant) and node.right.value == 1:
                return _unit_step(node.left, element)
            if isinstance(node.left, ast.Constant) and node.left.value == 1:
                return _unit_step(node.right, element)
    return False


def prove_vector_leaf(
    indices: Sequence[str],
    definitions: Mapping[str, str],
    *,
    element: str,
    uniform_names: AbstractSet[str],
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    dtype: torch.dtype,
    mask: str | None = None,
) -> VectorLeafPlan | None:
    """Prove a unit-step inner vector, retaining exact outer expressions.

    ``uniform_names`` is an explicit caller obligation: each named scalar must
    be constant across this thread's eight elements. Definitions take precedence
    over this set and are recursively checked. Unknown/cyclic dependencies fail
    closed. Uniform expressions may retain fixed-width arithmetic, but varying
    casts, divisions, cancellation, and masks are deliberately unsupported.

    The result is conditional on emitted per-vector bounds, stride, alignment,
    and no-wrap guards; it does not authorize an unconditional vector load.
    """
    if (
        dtype not in _DTYPES
        or not indices
        or len(indices) != len(shape)
        or len(indices) != len(strides)
        or any(extent < 0 for extent in shape)
    ):
        return None
    try:
        expanded = tuple(
            _expand(index, definitions, element, uniform_names) for index in indices
        )
        predicate = (
            _expand(mask, definitions, element, uniform_names)
            if mask is not None
            else None
        )
    except _NotProven:
        return None
    varying = [axis for axis, value in enumerate(expanded) if _varies(value, element)]
    if (
        len(varying) != 1
        or strides[varying[0]] != 1
        or not _unit_step(expanded[varying[0]], element)
        or (predicate is not None and _varies(predicate, element))
    ):
        return None
    dtype_name, element_bytes = _DTYPES[dtype]
    return VectorLeafPlan(
        tuple(ast.unparse(value) for value in expanded),
        element,
        varying[0],
        shape,
        strides,
        dtype_name,
        element_bytes,
        ast.unparse(predicate) if predicate is not None else None,
    )


def _at(text: str, element: str, value: int) -> str:
    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            return ast.Constant(value) if node.id == element else node

    return ast.unparse(Substitute().visit(ast.parse(text, mode="eval").body))


def emit_vector_leaf(
    plan: VectorLeafPlan,
    *,
    tensor: str,
    prefix: str,
    pointer_for_indices: Callable[[tuple[str, ...]], str],
    scalar_for_element: Callable[[str], tuple[Sequence[str], str]],
    shared_pointer: str | None = None,
) -> VectorLeafEmission:
    """Emit one independent thread's vector load and exact scalar fallback.

    ``pointer_for_indices`` must use the original scalar addressing expression
    (including its integer casts) with the supplied logical indices, not a new
    widened offset formula. It must be ordinary stride-linear tensor addressing.
    ``scalar_for_element`` supplies the original masked scalar load and any local
    statements needed to evaluate it; the returned value retains its zero/default
    semantics. Neither callback may have side effects besides the scalar load.

    Guard arithmetic alone is widened. Every original source index/address is
    evaluated exactly at element zero and seven, so fixed-width wrapping cannot
    turn a noncontiguous range into a vector copy. All failed guards use the
    scalar callback. FP32 has eight values but two 128-bit transactions.

    A proven same-dtype dense shared sink may supply shared_pointer.
    Only the fast copy changes destination; fallback still produces the exact
    original register values. The caller must exclusively publish that fallback
    and commit/wait all async copies before its original publication barrier.
    """
    values, vectorized = f"{prefix}_values", f"{prefix}_vectorized"
    first = tuple(f"{prefix}_index_{axis}" for axis in range(len(plan.indices)))
    last = list(first)
    last[plan.axis] = f"{prefix}_last"
    lines = [
        f"{values} = cute.make_rmem_tensor(({plan.width},), {plan.dtype})",
        f"{vectorized} = cutlass.Boolean(False)",
        *(
            f"{name} = {_at(index, plan.element, 0)}"
            for name, index in zip(first, plan.indices, strict=True)
        ),
        f"{last[plan.axis]} = {_at(plan.indices[plan.axis], plan.element, plan.width - 1)}",
    ]
    bounds = []
    for axis, (name, extent, stride) in enumerate(
        zip(first, plan.shape, plan.strides, strict=True)
    ):
        bounds.extend(
            (
                f"cutlass.Int64({name}) >= 0",
                f"cutlass.Int64({name}) < {extent}",
                f"{tensor}.shape[{axis}] == {extent}",
                f"{tensor}.layout.stride[{axis}] == {stride}",
            )
        )
    base, end = first[plan.axis], last[plan.axis]
    bounds.extend(
        (
            f"cutlass.Int64({base}) <= {plan.shape[plan.axis] - plan.width}",
            f"cutlass.Int64({end}) == cutlass.Int64({base}) + {plan.width - 1}",
        )
    )
    if plan.mask is not None:
        bounds.append(plan.mask)
    lines.append("if " + " & ".join(f"({bound})" for bound in bounds) + ":")
    byte_delta = (plan.width - 1) * plan.element_bytes
    copy_shape = (plan.width,)
    if shared_pointer is not None and plan.width * plan.element_bytes > 16:
        # CopyG2S consumes one atom-sized V mode, then iterates Rest.
        # Eight FP32 values require two distinct 128-bit transactions.
        atom_values = 16 // plan.element_bytes
        assert plan.width % atom_values == 0
        copy_shape = (atom_values, plan.width // atom_values)
    fast = [
        f"{prefix}_pointer = {pointer_for_indices(first)}",
        f"{prefix}_last_pointer = {pointer_for_indices(tuple(last))}",
        f"{prefix}_address = cutlass.Int64({prefix}_pointer.toint())",
        f"{prefix}_last_address = cutlass.Int64({prefix}_last_pointer.toint())",
        *(
            [f"{prefix}_shared_pointer = {shared_pointer}"]
            if shared_pointer is not None
            else []
        ),
        f"if ({prefix}_address >= 0) & ({prefix}_address <= {2**63 - 1 - byte_delta}) & ({prefix}_last_address == {prefix}_address + {byte_delta}) & ({prefix}_address % 16 == 0)"
        + (
            f" & (cutlass.Int64({prefix}_shared_pointer.toint()) % 16 == 0)"
            if shared_pointer is not None
            else ""
        )
        + ":",
        f"    {prefix}_source = cute.make_tensor({prefix}_pointer.align(16), cute.make_layout({copy_shape}))",
        *(
            [
                f"    {prefix}_sink = cute.make_tensor({prefix}_shared_pointer.align(16), cute.make_layout({copy_shape}))"
            ]
            if shared_pointer is not None
            else []
        ),
        f"    {prefix}_copy = cute.make_copy_atom("
        + (
            "cute.nvgpu.cpasync.CopyG2SOp()"
            if shared_pointer is not None
            else "cute.nvgpu.CopyUniversalOp()"
        )
        + f", {plan.dtype}, num_bits_per_copy=128)",
        f"    cute.copy({prefix}_copy, {prefix}_source, "
        + (f"{prefix}_sink" if shared_pointer is not None else values)
        + ")",
        f"    {vectorized} = cutlass.Boolean(True)",
    ]
    lines.extend("    " + line for line in fast)
    element = f"{prefix}_element"
    scalar_lines, scalar = scalar_for_element(element)
    lines.extend(
        (
            f"if not {vectorized}:",
            f"    for {element} in cutlass.range_constexpr({plan.width}):",
            *("        " + line for line in scalar_lines),
            f"        {values}[{element}] = {plan.dtype}({scalar})",
        )
    )
    return VectorLeafEmission(tuple(lines), values, vectorized)
