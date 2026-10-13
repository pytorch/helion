"""Optional ownership selection for the original independent scalar fills.

This changes only which thread evaluates a logical element. Address/cast/mask
expressions and publication stay with the original operand emitter. Cost facts
never authorize an effect or an otherwise unsupported coordinate expression.
"""

from __future__ import annotations

import ast
from collections import defaultdict
import copy
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..generate_ast import GenerateAST
    from .chained_matmul import ChainedMatmulPlan
    from .chained_matmul import _Expression
    from .chained_root_warp_stage import WarpOperandLayout

IndexValues = tuple[dict[str, int], tuple[int, ...]]


class _Decline(Exception):
    pass


@dataclass(frozen=True)
class Read:
    owner: str
    indices: tuple[str, ...]
    strides: tuple[int, ...]
    itemsize: int
    shared: bool


class _Coordinates:
    """Keep uniform terms separate; never substitute a loaded index with zero."""

    def __init__(
        self, size: int, index: str, origins: Sequence[str], definitions: dict[str, str]
    ) -> None:
        self.size, self.index = size, index
        self.origins, self.definitions = set(origins), definitions
        self.memo: dict[str, IndexValues] = {}
        self.active: set[str] = set()

    def parse(self, text: str) -> IndexValues:
        return self.visit(ast.parse(text, mode="eval").body)

    def visit(self, node: ast.AST) -> IndexValues:
        zeros = (0,) * self.size
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return {}, (node.value,) * self.size
        if isinstance(node, ast.Name):
            name = node.id
            if name == self.index:
                return {}, tuple(range(self.size))
            if name in self.origins:
                return {name: 1}, zeros
            if name in self.memo:
                return self.memo[name]
            if name in self.definitions and name not in self.active:
                self.active.add(name)
                value = self.parse(self.definitions[name])
                self.active.remove(name)
                self.memo[name] = value
                return value
            raise _Decline
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            base, values = self.visit(node.operand)
            return {k: -v for k, v in base.items()}, tuple(-v for v in values)
        if (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "cutlass.Int32"
            and len(node.args) == 1
            and not node.keywords
        ):
            base, values = self.visit(node.args[0])
            if not all(-(2**31) <= x < 2**31 for x in values):
                raise _Decline
            # Uniform wrap changes a common alignment phase, which the cost
            # model enumerates. No emitted fixed-width arithmetic is rewritten.
            return base, values
        if isinstance(node, ast.BinOp):
            a, av = self.visit(node.left)
            b, bv = self.visit(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                sign = -1 if isinstance(node.op, ast.Sub) else 1
                base = {
                    k: a.get(k, 0) + sign * b.get(k, 0) for k in a.keys() | b.keys()
                }
                return {k: v for k, v in base.items() if v}, tuple(
                    x + sign * y for x, y in zip(av, bv, strict=True)
                )
            if isinstance(node.op, ast.Mult):
                if not a and len(set(av)) == 1:
                    factor, base, values = av[0], b, bv
                elif not b and len(set(bv)) == 1:
                    factor, base, values = bv[0], a, av
                else:
                    raise _Decline
                return {k: v * factor for k, v in base.items()}, tuple(
                    x * factor for x in values
                )
            if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
                if (
                    b
                    or len(set(bv)) != 1
                    or bv[0] <= 0
                    or any(x % bv[0] for x in a.values())
                ):
                    raise _Decline
                divisor = bv[0]
                if isinstance(node.op, ast.Mod):
                    return {}, tuple(x % divisor for x in av)
                return {k: v // divisor for k, v in a.items()}, tuple(
                    x // divisor for x in av
                )
        raise _Decline

    def predicate(self, node: ast.AST) -> None:
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            return
        if isinstance(node, ast.Name) and node.id in self.definitions:
            self.predicate(ast.parse(self.definitions[node.id], mode="eval").body)
            return
        if isinstance(node, ast.Compare):
            for operand in (node.left, *node.comparators):
                if isinstance(operand, (ast.Compare, ast.BoolOp)):
                    self.predicate(operand)
                else:
                    self.visit(operand)
            return
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                self.predicate(value)
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)):
            self.predicate(node.left)
            self.predicate(node.right)
            return
        if (
            isinstance(node, ast.Call)
            and ast.unparse(node.func)
            in (
                "operator.lt",
                "operator.le",
                "operator.gt",
                "operator.ge",
                "operator.eq",
                "operator.ne",
            )
            and len(node.args) == 2
            and not node.keywords
        ):
            for arg in node.args:
                self.visit(arg)
            return
        raise _Decline


def _effect_check(
    lines: Sequence[str],
    value: str,
    bounds: Sequence[str],
    reads: Sequence[Read],
    index: str,
    origins: Sequence[str],
    *,
    uniform_definitions: bool = False,
) -> dict[str, str]:
    known = {index, *origins}
    global_owners = {r.owner for r in reads if not r.shared}
    shared_owners = {r.owner for r in reads if r.shared}
    definitions: dict[str, str] = {}
    model = _Coordinates(1, index, origins, definitions)

    def expression(node: ast.AST) -> None:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool):
            return
        if isinstance(node, ast.Name) and node.id in known:
            return
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
        ):
            expression(node.left)
            expression(node.right)
            return
        if (
            uniform_definitions
            and isinstance(node, ast.BinOp)
            and isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.RShift, ast.LShift))
            and _uniform_definition(node, model.origins)
        ):
            expression(node.left)
            expression(node.right)
            return
        if isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.USub, ast.UAdd, ast.Not)
        ):
            expression(node.operand)
            return
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            model.predicate(node)
            return
        if isinstance(node, ast.IfExp):
            model.predicate(node.test)
            expression(node.body)
            expression(node.orelse)
            return
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in shared_owners
        ):
            for coordinate in (
                node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
            ):
                model.visit(coordinate)
            return
        if isinstance(node, ast.Call) and not node.keywords:
            name = ast.unparse(node.func)
            if (
                name
                in (
                    "cutlass.Int32",
                    "cutlass.Float32",
                    "cutlass.BFloat16",
                    "cutlass.Float16",
                )
                and len(node.args) == 1
            ):
                expression(node.args[0])
                return
            if (
                name
                in (
                    "operator.lt",
                    "operator.le",
                    "operator.gt",
                    "operator.ge",
                    "operator.eq",
                    "operator.ne",
                )
                and len(node.args) == 2
            ):
                model.predicate(node)
                return
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
                and not node.args
            ):
                pointer(node.func.value)
                return
        raise _Decline

    def pointer(node: ast.AST) -> None:
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "iterator"
            and isinstance(node.value, ast.Name)
            and node.value.id in global_owners
        ):
            return
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            pointer(node.left)

            # Only original integer address arithmetic and declared strides.
            class Strides(ast.NodeTransformer):
                def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
                    text = ast.unparse(node.value)
                    if (
                        any(text == f"{owner}.layout.stride" for owner in global_owners)
                        and isinstance(node.slice, ast.Constant)
                        and type(node.slice.value) is int
                    ):
                        return ast.Constant(1)
                    raise _Decline

            model.visit(Strides().visit(copy.deepcopy(node.right)))
            return
        raise _Decline

    for line in lines:
        statements = ast.parse(line).body
        if len(statements) != 1 or not isinstance(statements[0], ast.Assign):
            raise _Decline
        statement = statements[0]
        if len(statement.targets) != 1 or not isinstance(
            statement.targets[0], ast.Name
        ):
            raise _Decline
        name = statement.targets[0].id
        if name in known or name in global_owners | shared_owners:
            raise _Decline
        # Validate before insertion: only earlier immutable definitions may
        # reach this RHS. Together with the no-rebind check this makes the
        # predicate expansion graph acyclic (including self/mutual references).
        expression(statement.value)
        known.add(name)
        definitions[name] = ast.unparse(statement.value)
        if uniform_definitions and _uniform_definition(statement.value, model.origins):
            model.origins.add(name)
    expression(ast.parse(value, mode="eval").body)
    for bound in bounds:
        model.predicate(ast.parse(bound, mode="eval").body)
    return definitions


def _uniform_definition(node: ast.AST, origins: set[str]) -> bool:
    # Called only after _effect_check has validated the expression's effects.
    # A pure expression of CTA origins remains uniform even when a modulo or
    # signed-division adjustment cannot be represented as an affine origin.
    return not any(isinstance(part, ast.Subscript) for part in ast.walk(node)) and {
        part.id for part in ast.walk(node) if isinstance(part, ast.Name)
    } <= origins | {"cutlass", "operator"}


def _demand(
    offsets: Sequence[int],
    itemsize: int,
    shared: bool,
    *,
    store: bool = False,
    phase: int = 0,
) -> float:
    if shared:
        banks: dict[int, set[int]] = defaultdict(set)
        for offset in offsets:
            for byte in range(itemsize):
                address = offset * itemsize + byte
                banks[(address // 4) % 32].add(address if store else address // 4)
        # Store byte counts are divided by item size; comparisons are exact for
        # this scalar width. Read broadcasts of the same 32-bit word coalesce.
        return max(map(len, banks.values()), default=0) / (itemsize if store else 1)
    return len(
        {
            (offset * itemsize + byte + phase) // 32
            for offset in offsets
            for byte in range(itemsize)
        }
    )


def choose_striped(
    *,
    lines: Sequence[str],
    value: str,
    bounds: Sequence[str],
    reads: Sequence[Read],
    index: str,
    origins: Sequence[str],
    shape: tuple[int, int],
    stride: tuple[int, int],
    threads: int,
    vector: int,
) -> bool:
    """Return False on every unknown effect/layout/address or uncertain cost."""
    if (
        len(shape) != 2
        or len(stride) != 2
        or any(type(x) is not int or x <= 0 for x in (*shape, *stride, threads))
    ):
        return False
    size = math.prod(shape)
    if (
        threads % 32
        or vector not in (2, 4, 8)
        or size % (threads * vector)
        or not reads
        or any(
            len(r.indices) != len(r.strides)
            or r.itemsize not in (2, 4)
            or any(type(s) is not int or s <= 0 for s in r.strides)
            for r in reads
        )
    ):
        return False
    try:
        definitions = _effect_check(lines, value, bounds, reads, index, origins)
        model = _Coordinates(size, index, origins, definitions)
        addresses = []
        for read in reads:
            components = [model.parse(text)[1] for text in read.indices]
            offsets = tuple(
                sum(
                    component[i] * step
                    for component, step in zip(components, read.strides, strict=True)
                )
                for i in range(size)
            )
            addresses.append((read, offsets))
        # Destination uses exactly the original physical layout, without any
        # fragment-layout or shared-allocation transformation.
        inner = 1 if stride[1] == 1 else 0
        destination = tuple(
            (i // shape[1] * stride[0] + i % shape[1] * stride[1])
            if inner == 1
            else (i % shape[0] * stride[0] + i // shape[0] * stride[1])
            for i in range(size)
        )
        if len(set(destination)) != size:
            return False
        ideal_gain = False
        for step in range(size // (threads * vector)):
            for element in range(vector):
                for warp in range(threads // 32):
                    team = range(warp * 32, (warp + 1) * 32)
                    blocked = [
                        t * vector + step * threads * vector + element for t in team
                    ]
                    striped = [
                        t + element * threads + step * threads * vector for t in team
                    ]
                    if _demand(
                        [destination[i] for i in striped], 2, True, store=True
                    ) > _demand([destination[i] for i in blocked], 2, True, store=True):
                        return False
                    for read, offsets in addresses:
                        a, b = (
                            [offsets[i] for i in blocked],
                            [offsets[i] for i in striped],
                        )
                        phases = (0,) if read.shared else range(0, 32, read.itemsize)
                        for phase in phases:
                            old = _demand(a, read.itemsize, read.shared, phase=phase)
                            new = _demand(b, read.itemsize, read.shared, phase=phase)
                            if new > old:
                                return False
                            ideal = (
                                1
                                if read.shared
                                else math.ceil(len(set(b)) * read.itemsize / 32)
                            )
                            ideal_gain |= old > new and new == ideal
        return ideal_gain
    except _Decline:
        return False


def select_fill(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    expression: _Expression,
    layout: WarpOperandLayout,
    stage: int,
    padding: int,
    value: str,
    domain: Sequence[str],
    vector: int,
) -> bool:
    """Conservative bridge from existing typed storage owners to the selector."""
    from .chained_loop import loop_storage_is_proven
    from .chained_matmul import _shape

    config = cg.device_function.config.config
    if (
        config.get("cute_chained_fill_ownership", "blocked") != "auto"
        or plan.strategy != "warp"
        or plan.loop is None
        or not loop_storage_is_proven(plan.loop)
        or config.get("cute_chained_scratch_layout", "row_major") != "row_major"
        or plan.preparation_pipeline is not None
        or plan.pointwise_cache is not None
        or plan.loop_workspace is not None
        or plan.warp_workspace is not None
        or plan.row_registers is not None
        or plan.row_parallel is not None
        or expression.scan_inputs
        or expression.staged_inputs
        or expression.fragment_images
        or expression.coordinate_normalization is not None
        or layout.dtype not in (torch.float16, torch.bfloat16)
    ):
        return False
    known: dict[str, tuple[tuple[int, ...], tuple[int, ...], int]] = {}
    for node, name in expression.boundaries.items():
        if node in plan.loop.boundaries() and node.meta["val"].dtype is torch.float32:
            shape = _shape(node)
            if len(shape) == 2:
                known[name] = (shape, (shape[1], 1), node.meta["val"].element_size())
        elif node in plan.dots[:stage]:
            shape = _shape(node)
            known[name] = (shape, (shape[1] + padding // 2, 1), 4)
    reads = []
    for source, indices in expression.global_accesses:
        fake = source.meta["val"]
        reads.append(
            Read(
                expression.tensor_name(source),
                tuple(indices),
                tuple(fake.stride()),
                fake.element_size(),
                False,
            )
        )
    for text in (*expression.lines, value):
        for node in ast.walk(ast.parse(text)):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id in known
            ):
                shape, stride, size = known[node.value.id]
                coordinates = (
                    node.slice.elts
                    if isinstance(node.slice, ast.Tuple)
                    else (node.slice,)
                )
                if len(coordinates) != len(shape):
                    return False
                reads.append(
                    Read(
                        node.value.id,
                        tuple(map(ast.unparse, coordinates)),
                        stride,
                        size,
                        True,
                    )
                )
    return choose_striped(
        lines=expression.lines,
        value=value,
        bounds=domain,
        reads=reads,
        index=f"chain_{stage}_load",
        origins=tuple(expression.origins.values()),
        shape=layout.shape,
        stride=layout.stride,
        threads=plan.threads,
        vector=vector,
    )


def choose_carry_first_axis(
    *,
    lines: Sequence[str],
    value: str,
    reads: Sequence[Read],
    index: str,
    origins: Sequence[str],
    shape: tuple[int, int],
    threads: int,
    scratch_mode: str,
) -> bool:
    """Compare two complete scalar-fill permutations, using the original reads."""
    from .chained_scratch_layout import xor_swizzle

    if (
        len(shape) != 2
        or any(type(n) is not int or n < 2 for n in shape)
        or type(threads) is not int
        or threads < 32
        or threads % 32
        or math.prod(shape) % threads
        or math.prod(shape) > 65536
        or scratch_mode not in ("row_major", "xor")
        or not reads
        or any(
            read.shared
            or len(read.indices) != len(read.strides)
            or read.itemsize not in (2, 4)
            or any(type(s) is not int or s <= 0 for s in read.strides)
            for read in reads
        )
    ):
        return False
    size = math.prod(shape)
    rows, columns = shape
    try:
        definitions = _effect_check(
            lines, value, (), reads, index, origins, uniform_definitions=True
        )
        uniform = set(origins)
        for name, text in definitions.items():
            if _uniform_definition(ast.parse(text, mode="eval").body, uniform):
                uniform.add(name)
        model = _Coordinates(size, index, tuple(uniform), definitions)
        addresses = []
        contiguous = False
        for read in reads:
            components = [model.parse(text)[1] for text in read.indices]
            offsets = tuple(
                sum(
                    component[i] * step
                    for component, step in zip(components, read.strides, strict=True)
                )
                for i in range(size)
            )
            row_step = offsets[columns] - offsets[0]
            column_step = offsets[1] - offsets[0]
            # Views and pointwise expressions are already traversed by
            # _Expression. Decline a gather/non-affine physical read, rather
            # than infer its behavior from a few sampled coordinates.
            if (
                row_step < 0
                or column_step < 0
                or any(
                    offsets[i]
                    != offsets[0]
                    + (i // columns) * row_step
                    + (i % columns) * column_step
                    for i in range(size)
                )
            ):
                return False
            contiguous |= row_step == 1 and column_step > 1
            addresses.append((read, offsets))
        if not contiguous:
            return False
        permutation = tuple((i % rows) * columns + i // rows for i in range(size))
        swizzle = xor_swizzle(shape) if scratch_mode == "xor" else None
        destination = tuple(range(size))
        if swizzle is not None:
            bits, base, shift = swizzle
            mask = ((1 << bits) - 1) << base
            destination = tuple(i ^ ((i >> shift) & mask) for i in destination)
        if len(set(destination)) != size:
            return False
        gain = False
        for start in range(0, size, 32):
            old = range(start, start + 32)
            new = permutation[start : start + 32]
            if _demand([destination[i] for i in new], 4, True, store=True) > _demand(
                [destination[i] for i in old], 4, True, store=True
            ):
                return False
            for read, offsets in addresses:
                a, b = [offsets[i] for i in old], [offsets[i] for i in new]
                for phase in range(0, 32, read.itemsize):
                    before = _demand(a, read.itemsize, False, phase=phase)
                    after = _demand(b, read.itemsize, False, phase=phase)
                    if after > before:
                        return False
                    gain |= after < before
        return gain
    except _Decline:
        return False


def select_carry_order(
    cg: GenerateAST,
    plan: ChainedMatmulPlan,
    expression: _Expression,
    shape: tuple[int, int],
    dtype: torch.dtype,
    scratch_mode: str,
    index: str,
    value: str,
) -> tuple[list[str], str, tuple[str, str]] | None:
    """Rebind only original logical coordinates; preserve the single FX visit."""
    from .chained_loop import loop_storage_is_proven

    if (
        cg.device_function.config.config.get("cute_chained_fill_ownership", "blocked")
        != "auto"
        or plan.loop is None
        or not loop_storage_is_proven(plan.loop)
        or dtype is not torch.float32
        or len(expression.accesses) != len(expression.global_accesses)
        or expression.scan_inputs
        or expression.staged_inputs
        or expression.fragment_images
        or expression.coordinate_normalization is not None
    ):
        return None
    reads = [
        Read(
            expression.tensor_name(source),
            tuple(indices),
            tuple(source.meta["val"].stride()),
            source.meta["val"].element_size(),
            False,
        )
        for source, indices in expression.global_accesses
    ]
    if not choose_carry_first_axis(
        lines=expression.lines,
        value=value,
        reads=reads,
        index=index,
        origins=tuple(expression.origins.values()),
        shape=shape,
        threads=plan.threads,
        scratch_mode=scratch_mode,
    ):
        return None
    old_coords = (f"{index} // {shape[1]}", f"{index} % {shape[1]}")
    coords = (f"{index} % {shape[0]}", f"{index} // {shape[0]}")
    substitutions = {
        ast.dump(ast.parse(old, mode="eval").body): ast.parse(new, mode="eval").body
        for old, new in zip(old_coords, coords, strict=True)
    }

    class Rebind(ast.NodeTransformer):
        def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
            replacement = substitutions.get(ast.dump(node))
            if replacement is not None:
                return ast.copy_location(copy.deepcopy(replacement), node)
            return self.generic_visit(node)

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id == index:
                # Every use must derive from an original logical coordinate;
                # a naked flat-index value needs a separate semantic proof.
                raise _Decline
            return node

    def rewrite(text: str) -> str:
        return ast.unparse(Rebind().visit(ast.parse(text)))

    try:
        return [rewrite(line) for line in expression.lines], rewrite(value), coords
    except _Decline:
        return None
