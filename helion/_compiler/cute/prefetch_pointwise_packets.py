"""Prefetch independent vector packets within a finite scalar tile loop.

The caller supplies the existing full-tile proof and storage disjointness facts.
Only unconditional vector loads from storage disjoint from every loop store
move. Address bindings must be independent of all preceding loop iterations.
The scalar computation and output-store order remain unchanged.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

from .persistent_branch_vec import _DTYPE_INFO
from .persistent_branch_vec import _binding_write_roots
from .persistent_branch_vec import _range_extent
from .persistent_branch_vec import _single_tensor_iterator_root
from .pure_lane_packets import _PacketExpression
from .scalar_recipe import _GLOBALS
from .scalar_recipe import _clone
from .scalar_recipe import _NotRecipe
from .scalar_recipe import _read_names

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Set as AbstractSet

_STORES = frozenset(info[2] for info in _DTYPE_INFO.values())
_GLOBALS_USED = _GLOBALS | _STORES | {"ir", "range", "_cute_inline_asm_elementwise"}


class _Expression(_PacketExpression):
    def __init__(self, registers: Collection[str]) -> None:
        super().__init__()
        self.registers = registers

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in self.registers:
            if not isinstance(node.ctx, ast.Load):
                raise _NotRecipe
            self.visit(node.slice)
            return
        super().visit_Subscript(node)


def _pure(value: ast.expr, registers: Collection[str] = ()) -> bool:
    visitor = _Expression(registers)
    try:
        visitor.visit(value)
    except _NotRecipe:
        return False
    return not visitor.reads_memory


def _vector_load(value: ast.expr) -> str | None:
    if not (
        isinstance(value, ast.Call)
        and ast.unparse(value.func) == "cute.arch.load"
        and len(value.args) == 2
        and not value.keywords
        and _pure(value.args[0])
    ):
        return None
    vector = value.args[1]
    if not (
        isinstance(vector, ast.Call)
        and ast.unparse(vector.func) == "ir.VectorType.get"
        and len(vector.args) == 2
        and not vector.keywords
        and isinstance(vector.args[0], ast.List)
        and len(vector.args[0].elts) == 1
        and isinstance(vector.args[0].elts[0], ast.Constant)
        and type(vector.args[0].elts[0].value) is int
        and vector.args[0].elts[0].value in (2, 4, 8)
        and ast.unparse(vector.args[1])
        in {"cutlass.Uint16.mlir_type", "cutlass.Uint32.mlir_type"}
    ):
        return None
    return _single_tensor_iterator_root(value.args[0])


def _literal_range_extent(loop: ast.For) -> int | None:
    call = loop.iter
    if not (
        isinstance(call, ast.Call)
        and ast.unparse(call.func) in {"range", "cutlass.range_constexpr"}
        and len(call.args) == 1
        and not call.keywords
        and isinstance(call.args[0], ast.Constant)
        and type(call.args[0].value) is int
    ):
        return None
    return call.args[0].value


def _compute_stores(
    body: list[ast.stmt], registers: set[str], lists: set[str]
) -> set[str] | None:
    """Reject unclassified effects and memory reads in the compute phase."""
    stores: set[str] = set()
    for statement in body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            target = statement.targets[0].id
            value = statement.value
            if isinstance(value, ast.Name) and value.id in registers:
                # Do not permit an untracked mutable container alias.
                return None
            if isinstance(value, ast.List) and not value.elts:
                lists.add(target)
                registers.add(target)
            elif not _pure(value, registers):
                return None
            else:
                registers.discard(target)
                lists.discard(target)
        elif isinstance(statement, ast.For):
            if (
                statement.orelse
                or _range_extent(statement) is None
                or not isinstance(statement.target, ast.Name)
            ):
                return None
            registers.discard(statement.target.id)
            lists.discard(statement.target.id)
            nested = _compute_stores(statement.body, registers, lists)
            if nested is None:
                return None
            stores.update(nested)
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id in lists
                and call.func.attr == "append"
                and len(call.args) == 1
                and not call.keywords
                and _pure(call.args[0], registers)
            ):
                continue
            if not (
                isinstance(call.func, ast.Name)
                and call.func.id in _STORES
                and len(call.args) == 2
                and not call.keywords
                and _pure(call.args[0])
                and isinstance(call.args[1], ast.Name)
                and call.args[1].id in lists
            ):
                return None
            root = _single_tensor_iterator_root(call.args[0])
            if root is None:
                return None
            stores.add(root)
        elif not isinstance(statement, ast.Pass):
            return None
    return stores


class _Prefetch:
    def __init__(
        self,
        body: list[ast.stmt],
        batch_size: int,
        disjoint: AbstractSet[frozenset[str]],
        reserved_names: Collection[str] = (),
    ) -> None:
        self.batch_size = batch_size
        self.disjoint = disjoint
        self.names = set(reserved_names) | {
            node.id
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name)
        }

    def fresh(self, base: str) -> str:
        name = base
        while name in self.names:
            name += "_"
        self.names.add(name)
        return name

    def loop(self, original: ast.For) -> list[ast.stmt] | None:
        extent = _literal_range_extent(original)
        if (
            extent is None
            or extent < self.batch_size
            or extent % self.batch_size
            or original.orelse
            or not isinstance(original.target, ast.Name)
            or _binding_write_roots(original) & _GLOBALS_USED
        ):
            return None
        lane = original.target.id
        writes = _binding_write_roots(original)
        prefix: list[ast.Assign] = []
        loads: dict[str, str] = {}
        loaded_roots: set[str] = set()
        defined = {lane}
        for statement in original.body:
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                break
            target = statement.targets[0].id
            root = _vector_load(statement.value)
            if root is None and not _pure(statement.value):
                break
            if target in defined or (_read_names(statement.value) - defined) & writes:
                return None
            prefix.append(statement)
            defined.add(target)
            if root is not None:
                loads[target] = self.fresh("_helion_prefetched_vectors")
                loaded_roots.add(root)
        if not loads:
            return None
        rest = original.body[len(prefix) :]
        stores = _compute_stores(rest, set(loads), set())
        if stores is None or not stores:
            return None
        if any(
            frozenset((source, target)) not in self.disjoint
            for source in loaded_roots
            for target in stores
        ):
            return None

        packet = lane if extent == self.batch_size else self.fresh("_helion_packet")
        batch = self.fresh("_helion_packet_batch")
        index = ast.Name(id=packet, ctx=ast.Load())
        assignment = (
            []
            if extent == self.batch_size
            else ast.parse(f"{lane} = {batch} * {self.batch_size} + {packet}").body
        )
        initialize = [
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast.List(elts=[], ctx=ast.Load()),
            )
            for name in loads.values()
        ]
        preload: list[ast.stmt] = [_clone(statement) for statement in prefix]
        preload.extend(
            ast.parse(f"{container}.append({value})").body[0]
            for value, container in loads.items()
        )
        compute: list[ast.stmt] = []
        for statement in prefix:
            cloned = _clone(statement)
            name = cast("ast.Name", statement.targets[0]).id
            if name in loads:
                cloned.value = ast.Subscript(
                    value=ast.Name(id=loads[name], ctx=ast.Load()),
                    slice=_clone(index),
                    ctx=ast.Load(),
                )
            compute.append(cloned)
        compute.extend(_clone(statement) for statement in rest)

        def loop(body: list[ast.stmt]) -> ast.For:
            return ast.For(
                target=ast.Name(id=packet, ctx=ast.Store()),
                iter=ast.parse(
                    f"cutlass.range_constexpr({self.batch_size})", mode="eval"
                ).body,
                body=[*(_clone(statement) for statement in assignment), *body],
                orelse=[],
            )

        result: list[ast.stmt] = [*initialize, loop(preload), loop(compute)]
        if extent == self.batch_size:
            return result
        return [
            ast.For(
                target=ast.Name(id=batch, ctx=ast.Store()),
                iter=ast.parse(f"range({extent // self.batch_size})", mode="eval").body,
                body=result,
                orelse=[],
            )
        ]

    def body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for statement in body:
            if isinstance(statement, ast.For):
                replacement = self.loop(statement)
                if replacement is not None:
                    result.extend(replacement)
                    continue
            if isinstance(statement, (ast.If, ast.For)):
                statement = _clone(statement)
                statement.body = self.body(statement.body)
                statement.orelse = self.body(statement.orelse)
            result.append(statement)
        return result


def prefetch_pointwise_packets(
    body: list[ast.stmt],
    *,
    batch_size: int,
    proven_disjoint_tensor_pairs: AbstractSet[frozenset[str]],
    reserved_names: Collection[str] = (),
) -> list[ast.stmt]:
    """Transform only the caller's proved complete-tile branch."""
    if batch_size not in (2, 4, 8):
        return body
    if _binding_write_roots(ast.Module(body=body, type_ignores=[])) & _GLOBALS_USED:
        return body
    transformed = _Prefetch(
        body, batch_size, proven_disjoint_tensor_pairs, reserved_names
    ).body(body)
    return ast.fix_missing_locations(ast.Module(body=transformed, type_ignores=[])).body
