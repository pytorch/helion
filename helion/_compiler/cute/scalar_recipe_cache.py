"""Cache invariant scalar load results for a small, static copy-slot loop.

The caller proves that the named read-only tensors cannot change during the
enclosing reduction. Both prefill and consumption use compile-time slot
indices, avoiding dynamic indexing of a register-memory array.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

from .scalar_recipe import ScalarRecipe
from .scalar_recipe import _clone
from .scalar_recipe import _memory_reads
from .scalar_recipe import _read_names
from .scalar_recipe import build_recipe

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence

_CACHE_DTYPES = frozenset(
    f"cutlass.{name}"
    for name in (
        "Boolean",
        "Int32",
        "Int64",
        "Uint32",
        "Uint64",
        "Float16",
        "BFloat16",
        "Float32",
        "Float64",
    )
)


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _path(value: ast.expr) -> str | None:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute) and (base := _path(value.value)) is not None:
        return f"{base}.{value.attr}"
    return None


def _tuple(names: Collection[str]) -> ast.Tuple:
    return ast.Tuple(
        elts=[ast.Name(id=name, ctx=ast.Load()) for name in names], ctx=ast.Load()
    )


class _Dependencies:
    def __init__(
        self, definitions: Mapping[str, ast.expr], tensor_dtypes: Mapping[str, str]
    ) -> None:
        self.definitions = definitions
        self.tensor_dtypes = tensor_dtypes
        self._leaves: dict[ast.AST, frozenset[str]] = {}
        self._load_sources: dict[ast.AST, frozenset[str] | None] = {}
        self._dtypes: dict[ast.expr, str | None] = {}

    def resolve(self, value: ast.expr) -> ast.expr:
        while isinstance(value, ast.Name) and value.id in self.definitions:
            value = self.definitions[value.id]
        return value

    def leaves(self, value: ast.AST) -> frozenset[str]:
        if value in self._leaves:
            return self._leaves[value]
        result: set[str] = set()
        for name in _read_names(value):
            if name in self.definitions:
                result.update(self.leaves(self.definitions[name]))
            else:
                result.add(name)
        self._leaves[value] = frozenset(result)
        return self._leaves[value]

    def load_sources(self, value: ast.AST) -> frozenset[str] | None:
        if value not in self._load_sources:
            self._load_sources[value] = self._find_load_sources(value)
        return self._load_sources[value]

    def _find_load_sources(self, value: ast.AST) -> frozenset[str] | None:
        result: set[str] = set()
        for node in ast.walk(value):
            if isinstance(node, ast.Subscript) and _memory_reads(node):
                # Scalar recipes also accept tensor indexing. Only ordinary
                # pointer loads have a proven source and dtype here.
                return None
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
            ):
                source = self.pointer_source(node.func.value)
                if source is None:
                    return None
                result.add(source)
        for name in _read_names(value):
            if name in self.definitions:
                sources = self.load_sources(self.definitions[name])
                if sources is None:
                    return None
                result.update(sources)
        return frozenset(result)

    def pointer_source(self, value: ast.expr) -> str | None:
        roots: list[str] = []
        for node in ast.walk(value):
            if isinstance(node, ast.Attribute) and node.attr == "iterator":
                root = self.resolve(node.value)
                if not isinstance(root, ast.Name) or root.id not in self.tensor_dtypes:
                    return None
                roots.append(root.id)
        if not roots and isinstance(value, ast.Name) and value.id in self.definitions:
            return self.pointer_source(self.definitions[value.id])
        return roots[0] if len(roots) == 1 else None

    def dtype(self, value: ast.expr) -> str | None:
        if value not in self._dtypes:
            self._dtypes[value] = self._infer_dtype(value)
        return self._dtypes[value]

    def _infer_dtype(self, value: ast.expr) -> str | None:
        value = self.resolve(value)
        if isinstance(value, ast.Call):
            path = _path(value.func)
            if path in _CACHE_DTYPES and len(value.args) == 1 and not value.keywords:
                return path
            if isinstance(value.func, ast.Attribute) and value.func.attr == "load":
                source = self.pointer_source(value.func.value)
                return self.tensor_dtypes.get(source) if source is not None else None
            if path in {
                "operator.lt",
                "operator.le",
                "operator.gt",
                "operator.ge",
                "operator.eq",
                "operator.ne",
            }:
                return "cutlass.Boolean"
        if isinstance(value, ast.Compare):
            return "cutlass.Boolean"
        if isinstance(value, ast.BoolOp):
            return (
                "cutlass.Boolean"
                if all(self.dtype(part) == "cutlass.Boolean" for part in value.values)
                else None
            )
        if isinstance(value, ast.IfExp):
            left, right = self.dtype(value.body), self.dtype(value.orelse)
            return left if left is not None and left == right else None
        if isinstance(value, ast.UnaryOp):
            if isinstance(value.op, ast.Not):
                return "cutlass.Boolean"
            dtype = self.dtype(value.operand)
            return dtype if dtype != "cutlass.Boolean" else None
        if isinstance(value, ast.BinOp) and isinstance(
            value.op, (ast.Add, ast.Sub, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor)
        ):
            left, right = self.dtype(value.left), self.dtype(value.right)
            integers = {
                "cutlass.Int32",
                "cutlass.Int64",
                "cutlass.Uint32",
                "cutlass.Uint64",
            }
            if left != right:
                return None
            if (
                left in integers
                or left == "cutlass.Boolean"
                and isinstance(value.op, (ast.BitAnd, ast.BitOr, ast.BitXor))
            ):
                return left
        return None


@dataclasses.dataclass(frozen=True)
class ScalarCacheEmission:
    """Insert ``prologue`` before K and replace the slot loop's prefix/iterator.

    ``slot_iterator`` is required: consuming the returned arrays through a
    dynamic slot loop would reintroduce register-array addressing and spills.
    The original slot-coordinate prelude, tile guard, and copy body remain.
    """

    prologue: list[ast.stmt]
    prefix: list[ast.Assign]
    slot_iterator: ast.expr
    cached_names: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class ScalarRecipeCache:
    slot_name: str
    slot_count: int
    _values: tuple[tuple[str, str], ...]
    _value_recipe: ScalarRecipe
    _guard_recipe: ScalarRecipe
    _prefix: tuple[ast.Assign, ...]

    def emit(
        self,
        fresh_name: Callable[[str], str],
        *,
        execution_guard: ast.expr,
    ) -> ScalarCacheEmission:
        """Emit cache prefill only when the original reduction executes.

        ``execution_guard`` is the caller-proven nonempty-reduction predicate,
        using values available before K. Prefill also preserves the original
        copy-slot guard before evaluating any cached load recipe.
        """
        prefill_slot = fresh_name("prefill_slot")
        caches = {name: fresh_name(f"{name}_cache") for name, _dtype in self._values}
        allocations: list[ast.stmt] = []
        for name, dtype in self._values:
            allocations.extend(
                ast.parse(
                    f"{caches[name]} = cute.make_rmem_tensor(({self.slot_count},), {dtype})\n{caches[name]}.fill(0)"
                ).body
            )
        replacements = {self.slot_name: ast.Name(id=prefill_slot, ctx=ast.Load())}
        guards, condition = self._guard_recipe.emit(replacements, fresh_name)
        setup, values = self._value_recipe.emit(replacements, fresh_name)
        assert isinstance(values, ast.Tuple)
        prefill_body: list[ast.stmt] = [*setup]
        for (name, _dtype), value in zip(self._values, values.elts, strict=True):
            prefill_body.append(
                ast.Assign(
                    targets=[
                        ast.Subscript(
                            value=ast.Name(id=caches[name], ctx=ast.Load()),
                            slice=ast.Name(id=prefill_slot, ctx=ast.Load()),
                            ctx=ast.Store(),
                        )
                    ],
                    value=value,
                )
            )
        prefill = ast.For(
            target=ast.Name(id=prefill_slot, ctx=ast.Store()),
            iter=_expr(f"cutlass.range_constexpr({self.slot_count})"),
            body=[*guards, ast.If(test=condition, body=prefill_body, orelse=[])],
            orelse=[],
        )
        allocations.append(
            ast.If(test=_clone(execution_guard), body=[prefill], orelse=[])
        )
        prefix: list[ast.Assign] = []
        for statement in self._prefix:
            statement = _clone(statement)
            target = statement.targets[0]
            assert isinstance(target, ast.Name)
            if target.id in caches:
                statement.value = ast.Subscript(
                    value=ast.Name(id=caches[target.id], ctx=ast.Load()),
                    slice=ast.Name(id=self.slot_name, ctx=ast.Load()),
                    ctx=ast.Load(),
                )
            prefix.append(ast.fix_missing_locations(statement))
        return ScalarCacheEmission(
            [ast.fix_missing_locations(stmt) for stmt in allocations],
            prefix,
            _expr(f"cutlass.range_constexpr({self.slot_count})"),
            tuple(caches),
        )


def plan_scalar_recipe_cache(
    prefix: Sequence[ast.Assign],
    copy_body: Sequence[ast.stmt],
    *,
    slot_name: str,
    slot_count: int,
    slot_prelude: Sequence[ast.Assign],
    slot_guard: ast.expr,
    boundary_names: Collection[str],
    varying_names: Collection[str],
    tensor_dtypes: Mapping[str, str],
    readonly_tensors: Collection[str],
    max_slots: int = 8,
    max_cached_values: int = 16,
) -> ScalarRecipeCache | None:
    """Cache live, typed, memory-derived outputs invariant across a reduction.

    Boundaries must be available before the reduction and remain unchanged;
    ``varying_names`` must include every changed binding. The caller proves
    ``readonly_tensors`` cannot change anywhere across the cached interval,
    including aliases and concurrent work. Slot prelude/guard must be invariant
    and pure, and are replayed before every prefetched slot's memory loads.

    The default eligibility bounds are eight static slots and sixteen cached
    scalar values per thread. Callers can raise ``max_slots`` to sixteen for
    larger copy tiles while retaining the independent scalar-value budget.
    """
    if not 0 < slot_count <= max_slots:
        return None
    statements = [*slot_prelude, *prefix]
    definitions: dict[str, ast.expr] = {}
    for statement in statements:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
        ):
            return None
        name = statement.targets[0].id
        if (
            name in definitions
            or name in boundary_names
            or name in varying_names
            or name == slot_name
        ):
            return None
        definitions[name] = statement.value
    boundaries = set(boundary_names) | {slot_name}
    if (
        build_recipe(_tuple(definitions), statements, boundaries | set(varying_names))
        is None
    ):
        return None
    guard = build_recipe(
        slot_guard, slot_prelude, boundaries, mutable_names=varying_names
    )
    if guard is None:
        return None
    deps = _Dependencies(definitions, tensor_dtypes)
    if deps.leaves(slot_guard) & set(varying_names):
        return None
    if deps.load_sources(slot_guard) != set():
        # Slot guards describe the copy layout; metadata load predicates stay
        # inside the original scalar value recipe instead.
        return None
    prefix_names = {
        statement.targets[0].id
        for statement in prefix
        if isinstance(statement.targets[0], ast.Name)
    }
    reads = set().union(*(_read_names(statement) for statement in copy_body))
    live = reads & prefix_names
    cached: dict[str, str] = {}
    visited: set[str] = set()

    def select(name: str) -> None:
        if name in visited or name not in prefix_names:
            return
        visited.add(name)
        sources = deps.load_sources(definitions[name])
        dtype = deps.dtype(definitions[name])
        invariant = not deps.leaves(definitions[name]) & set(varying_names)
        if (
            invariant
            and sources
            and sources <= set(readonly_tensors)
            and dtype in _CACHE_DTYPES
        ):
            assert dtype is not None
            cached[name] = dtype
            return
        for dependency in sorted(_read_names(definitions[name])):
            select(dependency)

    for name in definitions:
        if name in live:
            select(name)
    cached = {name: cached[name] for name in definitions if name in cached}
    if not cached or len(cached) * slot_count > max_cached_values:
        return None
    values = build_recipe(
        _tuple(cached), statements, boundaries, mutable_names=varying_names
    )
    if values is None:
        return None

    needed: set[str] = set()

    def visit(name: str) -> None:
        if name in needed or name not in definitions:
            return
        needed.add(name)
        if name not in cached:
            for dependency in _read_names(definitions[name]):
                visit(dependency)

    for name in live:
        visit(name)
    retained = tuple(
        _clone(statement)
        for statement in prefix
        if isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id in needed
    )
    return ScalarRecipeCache(
        slot_name, slot_count, tuple(cached.items()), values, guard, retained
    )
