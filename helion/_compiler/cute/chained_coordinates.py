"""Bounded integer composition inside an already owned row event.

Only compiler-created coordinates enter this sidecar. Unknown strings, tensor
indices and casts keep their original spelling; emitted source is never parsed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import TYPE_CHECKING
from typing_extensions import Self

if TYPE_CHECKING:
    from .chained_execution import ChainedExecution
    from .chained_matmul import ChainedMatmulPlan
    from .chained_row_registers import RowRegisterEvent
    from .chained_row_registers import RowRegisterIsland
    from .chained_row_registers import RowRegisterPlan


MAX_NODES = 128
MAX_POINTS = 1024
INT32_MAX = 2**31 - 1


class CoordinateText(str):
    """Rendered spelling plus the original coordinate's domain identity.

    Value/image memoization uses canonical equality. Domain extent decisions
    retain their original key, so a simplified derived index cannot become an
    originally registered full-vector coordinate.
    """

    __slots__ = ("original",)

    original: str

    def __new__(cls, text: str, original: str) -> Self:
        result = super().__new__(cls, text)
        result.original = original
        return result


def original_coordinate(text: str) -> str:
    return text.original if isinstance(text, CoordinateText) else text


@dataclass(frozen=True)
class IndexExpr:
    kind: str
    args: tuple[IndexExpr, ...] = ()
    value: int = 0

    @property
    def runtime(self) -> bool:
        return self.kind in ("row", "int32") or any(a.runtime for a in self.args)


def constant(value: int) -> IndexExpr:
    return IndexExpr("constant", value=value)


def binary(kind: str, left: IndexExpr, right: IndexExpr) -> IndexExpr:
    return IndexExpr(kind, (left, right))


def _bounded(expression: IndexExpr) -> bool:
    pending = [expression]
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if len(seen) > MAX_NODES:
            return False
        pending.extend(node.args)
    return True


@dataclass(frozen=True)
class CoordinateDomain:
    row: str
    column: str
    rows: tuple[int, int]
    columns: tuple[int, int]

    def points(self) -> tuple[tuple[int, int], ...] | None:
        lo, hi = self.rows
        left, right = self.columns
        if not (0 <= lo <= hi < 32 and 0 <= left <= right < 32):
            return None
        if (hi - lo + 1) * (right - left + 1) > MAX_POINTS:
            return None
        return tuple(product(range(lo, hi + 1), range(left, right + 1)))


@dataclass(frozen=True)
class CoordinateProof:
    original: IndexExpr
    replacement: IndexExpr
    domain: CoordinateDomain
    values: tuple[int, ...]


class _ValueCache:
    """Bounded reuse of immutable subexpressions in one coordinate context."""

    def __init__(self) -> None:
        self.points: tuple[tuple[int, int], ...] = ()
        # Retain each expression with its identity key, including after its
        # parent leaves the caller's scope. Never change structural proof keys.
        self.entries: dict[int, tuple[IndexExpr, tuple[int, ...]]] = {}

    def for_points(self, points: tuple[tuple[int, int], ...]) -> None:
        if points != self.points:
            self.entries.clear()
            self.points = points

    def remember(self, node: IndexExpr, values: tuple[int, ...]) -> None:
        if len(self.entries) >= MAX_NODES:
            self.entries.clear()
        self.entries[id(node)] = (node, values)


def _values(
    expression: IndexExpr,
    points: tuple[tuple[int, int], ...],
    *,
    cache: _ValueCache | None = None,
) -> tuple[int, ...] | None:
    if not _bounded(expression):
        return None
    if cache is not None:
        cache.for_points(points)
    # The root retains the complete immutable DAG during this evaluation.
    # Identity keys avoid recursively hashing every subtree at every visit.
    memo: dict[int, tuple[int, ...]] = {}

    def visit(node: IndexExpr) -> tuple[int, ...] | None:
        key = id(node)
        if key in memo:
            return memo[key]
        if cache is not None and key in cache.entries:
            owner, values = cache.entries[key]
            assert owner is node
            return values
        if len(memo) >= MAX_NODES:
            return None
        if node.kind == "constant":
            result = (node.value,) * len(points)
        elif node.kind == "row":
            result = tuple(row for row, _ in points)
        elif node.kind == "column":
            result = tuple(column for _, column in points)
        elif node.kind == "int32" and len(node.args) == 1:
            result = visit(node.args[0])
            if result is None:
                return None
        elif node.kind in ("add", "sub", "mul", "div", "mod") and len(node.args) == 2:
            left, right = (visit(arg) for arg in node.args)
            if left is None or right is None:
                return None
            if node.kind in ("div", "mod") and (
                node.args[1].kind != "constant" or node.args[1].value <= 0
            ):
                return None
            result = tuple(
                a + b
                if node.kind == "add"
                else a - b
                if node.kind == "sub"
                else a * b
                if node.kind == "mul"
                else a // b
                if node.kind == "div"
                else a % b
                for a, b in zip(left, right, strict=True)
            )
        else:
            return None
        # Bound every original intermediate. Keeping all literals and constexpr
        # intermediates in Int32 also avoids an implicit wider-type promotion.
        if any(value < 0 or value > INT32_MAX for value in result):
            return None
        memo[key] = result
        if cache is not None:
            cache.remember(node, result)
        return result

    return visit(expression)


def prove(
    expression: IndexExpr,
    domain: CoordinateDomain,
    *,
    _cache: _ValueCache | None = None,
) -> CoordinateProof | None:
    points = domain.points()
    if points is None:
        return None
    values = _values(expression, points, cache=_cache)
    if values is None:
        return None
    r, c = points[0]
    table = dict(zip(points, values, strict=True))
    dr = table[r + 1, c] - values[0] if r < domain.rows[1] else 0
    dc = table[r, c + 1] - values[0] if c < domain.columns[1] else 0
    offset = values[0] - dr * r - dc * c
    if min(dr, dc, offset) < 0:
        return None
    terms = [constant(offset)] if offset else []
    for coefficient, kind in ((dr, "row"), (dc, "column")):
        if coefficient:
            leaf = IndexExpr(kind)
            terms.append(
                leaf if coefficient == 1 else binary("mul", leaf, constant(coefficient))
            )
    replacement = terms[0] if terms else constant(0)
    for term in terms[1:]:
        replacement = binary("add", replacement, term)
    # A runtime coordinate can become constant/constexpr under a dominating
    # branch. Keep its original Int32 type in downstream CuTe expressions.
    if expression.runtime and not replacement.runtime:
        replacement = IndexExpr("int32", (replacement,))
    if _values(replacement, points, cache=_cache) != values:
        return None
    return CoordinateProof(expression, replacement, domain, values)


def render(node: IndexExpr, domain: CoordinateDomain) -> str:
    if node.kind == "constant":
        return str(node.value)
    if node.kind in ("row", "column"):
        return domain.row if node.kind == "row" else domain.column
    if node.kind == "int32":
        return f"cutlass.Int32({render(node.args[0], domain)})"
    op = {"add": "+", "sub": "-", "mul": "*", "div": "//", "mod": "%"}[node.kind]
    return f"({render(node.args[0], domain)} {op} {render(node.args[1], domain)})"


class CoordinateContext:
    def __init__(self, domain: CoordinateDomain) -> None:
        self.domain = domain
        self.expressions = {
            domain.row: IndexExpr("row"),
            domain.column: IndexExpr("column"),
        }
        self.proofs: dict[IndexExpr, CoordinateProof | None] = {}
        self._values_cache = _ValueCache()

    def in_bounds(self, coordinate: str, extent: int) -> bool:
        """Old row contexts preserve their original logical-domain predicates."""
        return False

    def lookup(self, text: str) -> IndexExpr | None:
        if text in self.expressions:
            return self.expressions[text]
        # These are literal positional indices supplied by the typed view
        # emitters (singleton axes, split members), never evaluated source.
        return constant(int(text)) if text.isdecimal() else None

    def normalize(self, expression: IndexExpr, original: str) -> str:
        if not _bounded(expression):
            return original
        if expression not in self.proofs:
            self.proofs[expression] = prove(
                expression, self.domain, _cache=self._values_cache
            )
        proof = self.proofs[expression]
        if proof is None:
            return original
        self.expressions[original] = expression
        replacement = render(proof.replacement, self.domain)
        if len(replacement) >= len(original):
            return CoordinateText(original, original)
        self.expressions[replacement] = proof.replacement
        return CoordinateText(replacement, original)

    def coordinate(self, text: str) -> str:
        value = self.lookup(text)
        return (
            text if value is None else self.normalize(value, original_coordinate(text))
        )

    def reshape(
        self, coordinates: tuple[str, ...], old: tuple[int, ...], new: tuple[int, ...]
    ) -> tuple[str, ...] | None:
        inputs = [self.lookup(text) for text in coordinates]
        if any(value is None for value in inputs):
            return None
        terms = [
            binary("mul", value, constant(math.prod(new[i + 1 :])))
            for i, value in enumerate(inputs)
            if value is not None
        ]
        flat = terms[0] if terms else constant(0)
        for term in terms[1:]:
            flat = binary("add", flat, term)
        text = (
            " + ".join(
                f"({original_coordinate(c)}) * {math.prod(new[i + 1 :])}"
                for i, c in enumerate(coordinates)
            )
            or "0"
        )
        return tuple(
            self.normalize(
                binary(
                    "mod",
                    binary("div", flat, constant(math.prod(old[i + 1 :]))),
                    constant(size),
                ),
                f"(({text}) // {math.prod(old[i + 1 :])}) % {size}",
            )
            for i, size in enumerate(old)
        )

    def iota(self, coordinate: str, start: object, step: object, original: str) -> str:
        value = self.lookup(coordinate)
        if value is None or type(start) is not int or type(step) is not int:
            return original
        return self.normalize(
            binary("add", constant(start), binary("mul", value, constant(step))),
            original,
        )


@dataclass(frozen=True)
class RowCoordinateScope:
    proposal: RowRegisterPlan
    island: RowRegisterIsland
    event: RowRegisterEvent
    domain: CoordinateDomain
    execution: ChainedExecution
    warp: int | None

    def check(self, plan: ChainedMatmulPlan) -> None:
        from .chained_execution import ChainedExecution
        from .chained_matmul import _UnsupportedChain

        self.proposal.check(plan)
        if (
            plan.row_registers is not self.proposal
            or self.island not in self.proposal.islands
            or self.event not in self.island.events
        ):
            raise _UnsupportedChain("foreign row coordinate scope")
        proof = self.event.reduction
        rows = (
            (proof.position, proof.position)
            if proof is not None and proof.axis == 0
            else (0, self.island.shape[0] - 1)
        )
        columns = (
            (proof.position, proof.position)
            if proof is not None and proof.axis == 1
            else (0, self.island.shape[1] - 1)
        )
        if (
            self.execution != ChainedExecution(plan.threads)
            or (self.warp is not None and not 0 <= self.warp < plan.threads // 32)
            or self.domain
            != CoordinateDomain(
                f"{self.island.name}_row", f"{self.island.name}_column", rows, columns
            )
            or self.domain.points() is None
        ):
            raise _UnsupportedChain("row coordinate branch or owner changed")


def row_scope(
    plan: ChainedMatmulPlan,
    island: RowRegisterIsland,
    event: RowRegisterEvent,
    row: str,
    column: str,
    execution: ChainedExecution,
    warp: int | None,
) -> RowCoordinateScope:
    from .chained_execution import ChainedExecution
    from .chained_matmul import _UnsupportedChain

    if (
        plan.row_registers is None
        or execution != ChainedExecution(plan.threads)
        or (warp is not None and not 0 <= warp < plan.threads // 32)
    ):
        raise _UnsupportedChain("row coordinate scope has no original warp owner")
    proof = event.reduction
    # These are exactly the branches emitted around this event, not the full
    # lane range: selected-owner row==pivot or active-row row<logical_rows.
    rows = (
        (proof.position, proof.position)
        if proof is not None and proof.axis == 0
        else (0, island.shape[0] - 1)
    )
    columns = (
        (proof.position, proof.position)
        if proof is not None and proof.axis == 1
        else (0, island.shape[1] - 1)
    )
    result = RowCoordinateScope(
        plan.row_registers,
        island,
        event,
        CoordinateDomain(row, column, rows, columns),
        execution,
        warp,
    )
    result.check(plan)
    return result
