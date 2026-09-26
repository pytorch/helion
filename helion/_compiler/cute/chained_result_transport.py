"""Sparse M64 resident result geometry and explicitly selected output transport."""

from __future__ import annotations

import ast
import dataclasses
from itertools import starmap
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy

from ..compile_environment import CompileEnvironment
from . import chained_matmul as chain

if TYPE_CHECKING:
    from torch.fx import Node

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan


M64_WIDTHS = (32, 64, 96, 128, 256)


def is_m64_plan(plan: ChainedMatmulPlan) -> bool:
    """Only one independent, full M64 dot uses the sparse datapath layout."""
    return (
        len(plan.dots) == len(plan.shapes) == 1
        and plan.shapes[0][0] == 64
        and plan.shapes[0][1] in M64_WIDTHS
        and plan.shapes[0][2] > 0
        and plan.shapes[0][2] % 16 == 0
        and plan.initialized_accumulator is None
        and plan.late_rhs_reuse is None
    )


def load_operation(shape: tuple[int, int]) -> str:
    """M64 keeps N physical columns; unused datapaths are not packed together."""
    m, n = shape
    if m == 128:
        return "tcgen05.Ld32x32bOp(tcgen05.Repetition(32))"
    if m != 64 or n not in M64_WIDTHS:
        raise ValueError("unsupported resident result geometry")
    repetition = min(8, (n & -n) // 8)
    return f"tcgen05.Ld16x256bOp(tcgen05.Repetition({repetition}))"


@dataclasses.dataclass(frozen=True)
class DirectLayout:
    """A static, injective whole-grid map; runtime checks retain exact strides."""

    offset: sympy.Expr
    row_stride: int
    strides: tuple[int, ...]


def _affine_bounds(
    value: sympy.Expr, domains: dict[sympy.Symbol, tuple[int, int]]
) -> tuple[int, int]:
    zero = dict.fromkeys(domains, 0)
    constant = value.subs(zero)
    if not isinstance(constant, sympy.Integer):
        raise chain._UnsupportedChain("non-affine direct output address")
    remainder = sympy.Add(value, -constant)
    lo = hi = int(constant)
    for symbol, (lower, upper) in domains.items():
        coefficient = sympy.diff(value, symbol)
        if not isinstance(coefficient, sympy.Integer):
            raise chain._UnsupportedChain("non-affine direct output address")
        remainder -= coefficient * symbol
        ends = (int(coefficient) * lower, int(coefficient) * upper)
        lo += min(ends)
        hi += max(ends)
    if sympy.expand(remainder) != 0:
        raise chain._UnsupportedChain("unproved direct output address dependency")
    return lo, hi


def _index(
    text: str,
    definitions: dict[str, str],
    symbols: dict[str, sympy.Symbol],
    domains: dict[sympy.Symbol, tuple[int, int]],
    bits: int,
) -> sympy.Expr:
    """Prove every intermediate integer operation before symbolic cancellation.

    The common evaluator owns coordinate construction. This only accepts its
    affine integer subset, including a fixed-width cast when it is lossless.
    """

    def visit(
        node: ast.AST, active: frozenset[str] = frozenset()
    ) -> tuple[sympy.Expr, int]:
        width = bits
        if isinstance(node, ast.Name) and node.id in definitions:
            if node.id in active:
                raise chain._UnsupportedChain("cyclic direct output index")
            return visit(
                ast.parse(definitions[node.id], mode="eval").body,
                active | {node.id},
            )
        if isinstance(node, ast.Constant) and type(node.value) is int:
            value = sympy.Integer(node.value)
        elif isinstance(node, ast.Name) and node.id in symbols:
            value = symbols[node.id]
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand, width = visit(node.operand, active)
            value = -operand
        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult)
        ):
            left, left_width = visit(node.left, active)
            right, right_width = visit(node.right, active)
            # Conservatively keep the narrowest operand type for arithmetic;
            # never use symbolic cancellation to widen an explicit Int32.
            width = min(left_width, right_width)
            value = (
                sympy.Add(left, right)
                if isinstance(node.op, ast.Add)
                else sympy.Add(left, sympy.Mul(-1, right))
                if isinstance(node.op, ast.Sub)
                else sympy.Mul(left, right)
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cutlass"
            and node.func.attr in ("Int32", "Int64")
            and len(node.args) == 1
            and not node.keywords
        ):
            value, _ = visit(node.args[0], active)
            width = int(node.func.attr[3:])
        else:
            raise chain._UnsupportedChain("unproved direct output index operation")
        lo, hi = _affine_bounds(value, domains)
        if lo < -(2 ** (width - 1)) or hi >= 2 ** (width - 1):
            raise chain._UnsupportedChain("direct output index may overflow")
        return value, width

    return visit(ast.parse(text, mode="eval").body)[0]


def prove_layout(
    indices: tuple[sympy.Expr, ...],
    extents: tuple[int, ...],
    strides: tuple[int, ...],
    row: sympy.Symbol,
    col: sympy.Symbol,
    shape: tuple[int, int],
    origins: tuple[tuple[sympy.Symbol, int, int], ...],
    storage_elements: int,
    bits: int,
) -> DirectLayout:
    """Prove all local and CTA output spans are bounded and mutually disjoint."""
    if not len(indices) == len(extents) == len(strides) or any(
        type(value) is not int or value <= 0 for value in (*extents, *strides)
    ):
        raise chain._UnsupportedChain("direct output requires static positive strides")
    domains = {row: (0, shape[0] - 1), col: (0, shape[1] - 1)}
    domains.update((symbol, (0, extent - block)) for symbol, extent, block in origins)
    for value, extent in zip(indices, extents, strict=True):
        lo, hi = _affine_bounds(value, domains)
        if lo < 0 or hi >= extent:
            raise chain._UnsupportedChain("direct output exceeds its logical extent")
    physical = sympy.Add(*starmap(sympy.Mul, zip(indices, strides, strict=True)))
    lo, hi = _affine_bounds(physical, domains)
    if lo < 0 or hi >= storage_elements or hi >= 2 ** (bits - 1):
        raise chain._UnsupportedChain("direct output exceeds its storage span")
    row_stride, col_stride = sympy.diff(physical, row), sympy.diff(physical, col)
    if (
        col_stride != 1
        or not isinstance(row_stride, sympy.Integer)
        or row_stride < shape[1]
    ):
        raise chain._UnsupportedChain("direct output requires an injective row layout")
    dimensions = [(shape[0], int(row_stride)), (shape[1], 1)]
    for symbol, extent, block in origins:
        coefficient = sympy.diff(physical, symbol)
        if not isinstance(coefficient, sympy.Integer):
            raise chain._UnsupportedChain("unproved direct output CTA stride")
        dimensions.append((extent // block, int(coefficient) * block))
    # Sufficient mixed-radix span proof, including every varying CTA origin.
    # An omitted/overlapping grid dimension cannot become duplicate writers.
    span = 1
    for count, step in sorted(
        ((count, step) for count, step in dimensions if count > 1),
        key=operator.itemgetter(1),
    ):
        if step < span:
            raise chain._UnsupportedChain("overlapping direct output CTA or lane spans")
        span += (count - 1) * step
    return DirectLayout(
        cast("sympy.Expr", physical.subs({row: 0, col: 0})), int(row_stride), strides
    )


def direct_store(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    expression: chain._Expression,
    coords: tuple[str, str],
    prefix: str,
) -> list[str]:
    """Store the final typed epilogue registers, using their actual TMEM slots."""
    if not is_m64_plan(plan):
        raise chain._UnsupportedChain("direct output requires one M64 dot")
    output = cast("Node", plan.store.args[0])
    fake = output.meta["val"]
    if not chain._has_fresh_output_allocation(output) or fake.storage_offset() != 0:
        raise chain._UnsupportedChain(
            "direct output requires fresh zero-offset storage"
        )
    shape = plan.shapes[0][:2]
    symbols = {
        name: sympy.Symbol(name, integer=True)
        for name in (*expression.origins.values(), *coords)
    }
    row, col = (symbols[name] for name in coords)
    origins = tuple(
        (symbols[expression.origins[axis]], extent, block)
        for axis, extent, block in plan.axes
    )
    domains = {row: (0, shape[0] - 1), col: (0, shape[1] - 1)}
    domains.update((symbol, (0, extent - block)) for symbol, extent, block in origins)
    bits = CompileEnvironment.current().index_dtype.itemsize * 8
    indices = expression.indices(plan.store, coords)
    layout = prove_layout(
        tuple(
            _index(index, expression.definitions, symbols, domains, bits)
            for index in indices
        ),
        tuple(fake.shape),
        tuple(fake.stride()),
        row,
        col,
        shape,
        origins,
        fake.untyped_storage().nbytes() // fake.element_size(),
        bits,
    )
    name = expression.tensor_name(output)
    guards = [
        f"{name}.layout.stride[{axis}] == {stride}"
        for axis, stride in enumerate(layout.strides)
    ]
    guards.extend(
        [
            "chain_direct_pointer.toint() % 16 == 0",
            f"{layout.row_stride * fake.element_size()} % 16 == 0",
        ]
    )
    fallback = chain._Expression(cg, plan, {})
    fallback_indices = fallback.indices(plan.store, coords)
    bounds = [
        f"0 <= ({index}) < {extent}"
        for index, extent in zip(fallback_indices, fake.shape, strict=True)
    ]
    return [
        f"chain_direct_pointer = {name}.iterator + ({chain._copy_code(layout.offset)})",
        f"if {' & '.join(f'({guard})' for guard in guards)}:",
        f"    chain_direct_tile = cute.make_tensor(chain_direct_pointer.align(16), cute.make_layout({shape!r}, stride=({layout.row_stride}, 1)))",
        f"    chain_direct_target = {prefix}_thread.partition_D({prefix}_slice.partition_C(chain_direct_tile))",
        "    cute.autovec_copy(chain_epi_values, chain_direct_target)",
        "else:",
        "    for chain_direct_index in cutlass.range_constexpr(cute.size(chain_epi_values)):",
        f"        {coords[0]}, {coords[1]} = {prefix}_coords[chain_direct_index]",
        chain._indent(fallback.lines, 8),
        f"        if {' & '.join(f'({bound})' for bound in bounds)}:",
        f"            {name}[{', '.join(fallback_indices)}] = chain_epi_values[chain_direct_index]",
    ]
