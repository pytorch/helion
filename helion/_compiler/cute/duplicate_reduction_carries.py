"""Remove identical max carries in a proven single-iteration logical loop.

Two independent reductions may become equal after specializing a logical tile
loop to its sole iteration. For example, a reduction restricted to the first
tile then has exactly the same contribution as an unrestricted reduction. This
proof compares the typed contribution, iteration domain, collective, and carry;
it never uses a variable prefix or a kernel/shape identity.

Run after lane reduction hoisting and before sibling vector loops acquire
register caches. Unknown effects, predicates, casts, recurrences, and observable
intermediate values keep the original reductions.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass

from .hoist_warp_reduce import _match_carry_update

_FLOAT_TYPES = frozenset({"Float16", "BFloat16", "Float32", "Float64"})
_SCALAR_TYPES = _FLOAT_TYPES | frozenset(
    {
        "Boolean",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Uint8",
        "Uint16",
        "Uint32",
        "Uint64",
    }
)


def _assignment(stmt: ast.AST) -> tuple[str, ast.expr] | None:
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return stmt.targets[0].id, stmt.value
    return None


def _lane_count(loop: ast.For) -> int | None:
    iterator = loop.iter
    if (
        loop.orelse
        or not isinstance(loop.target, ast.Name)
        or not isinstance(iterator, ast.Call)
        or ast.unparse(iterator.func) not in {"range", "cutlass.range_constexpr"}
        or len(iterator.args) != 1
        or iterator.keywords
        or not isinstance(iterator.args[0], ast.Constant)
        or type(iterator.args[0].value) is not int
        or iterator.args[0].value < 1
        or any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id == loop.target.id
            for statement in loop.body
            for node in ast.walk(statement)
        )
    ):
        return None
    return iterator.args[0].value


def _one_zero_iteration(
    loop: ast.For, constants: dict[str, int], writes: Counter[str]
) -> bool:
    iterator = loop.iter
    if (
        loop.orelse
        or not isinstance(loop.target, ast.Name)
        or not isinstance(iterator, ast.Call)
        or ast.unparse(iterator.func) != "range"
        or len(iterator.args) != 3
        or iterator.keywords
        or any(
            isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id == loop.target.id
            for statement in loop.body
            for node in ast.walk(statement)
        )
    ):
        return False
    values = []
    for expr in iterator.args:
        width = 64
        if _scalar_cast(expr) in {"Int32", "Int64"}:
            width = 32 if _scalar_cast(expr) == "Int32" else 64
            assert isinstance(expr, ast.Call)
            expr = expr.args[0]
        if isinstance(expr, ast.Name) and not writes[expr.id]:
            value = constants.get(expr.id)
        elif isinstance(expr, ast.Constant):
            value = expr.value
        else:
            return False
        if type(value) is not int or not -(1 << (width - 1)) <= value < 1 << (
            width - 1
        ):
            return False
        values.append(value)
    return values[0] == 0 and 0 < values[1] <= values[2]


def _scalar_cast(expr: ast.AST) -> str | None:
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and isinstance(expr.func.value, ast.Name)
        and expr.func.value.id == "cutlass"
        and expr.func.attr in _SCALAR_TYPES
        and len(expr.args) == 1
        and not expr.keywords
    ):
        return expr.func.attr
    return None


def _dtype(expr: ast.AST) -> str | None:
    if dtype := _scalar_cast(expr):
        return dtype
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and expr.func.attr == "bitcast"
        and _scalar_cast(expr.func.value) is not None
        and len(expr.args) == 1
        and not expr.keywords
        and isinstance(expr.args[0], ast.Attribute)
        and isinstance(expr.args[0].value, ast.Name)
        and expr.args[0].value.id == "cutlass"
        and expr.args[0].attr in _SCALAR_TYPES
    ):
        return expr.args[0].attr
    return None


def _pure(expr: ast.AST, vectors: set[str]) -> bool:
    if isinstance(expr, (ast.Name, ast.Constant)):
        return True
    if isinstance(expr, ast.Subscript):
        if (
            isinstance(expr.value, ast.Call)
            and ast.unparse(expr.value.func)
            in ("cute.arch.thread_idx", "cute.arch.block_idx")
            and not expr.value.args
            and not expr.value.keywords
            and isinstance(expr.slice, ast.Constant)
            and type(expr.slice.value) is int
            and 0 <= expr.slice.value < 3
        ):
            return True
        return (
            isinstance(expr.value, ast.Name)
            and expr.value.id in vectors
            and _pure(expr.slice, vectors)
        )
    if isinstance(expr, ast.BinOp):
        return (
            isinstance(
                expr.op, (ast.Add, ast.Sub, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor)
            )
            and _pure(expr.left, vectors)
            and _pure(expr.right, vectors)
        )
    if isinstance(expr, ast.UnaryOp):
        return _pure(expr.operand, vectors)
    if isinstance(expr, ast.Compare):
        return _pure(expr.left, vectors) and all(
            _pure(value, vectors) for value in expr.comparators
        )
    if isinstance(expr, ast.IfExp):
        return all(
            _pure(value, vectors) for value in (expr.test, expr.body, expr.orelse)
        )
    if _scalar_cast(expr) is not None:
        assert isinstance(expr, ast.Call)
        return _pure(expr.args[0], vectors)
    if isinstance(expr, ast.Call):
        name = ast.unparse(expr.func)
        if name == "float":
            return (
                len(expr.args) == 1
                and not expr.keywords
                and isinstance(expr.args[0], ast.Constant)
                and expr.args[0].value in ("-inf", "inf", "nan")
            )
        if name in ("cute.arch.thread_idx", "cute.arch.block_idx"):
            return not expr.args and not expr.keywords
        if name in ("cute.arch.fmax", "cute.math.max"):
            return (
                len(expr.args) == 2
                and all(_pure(arg, vectors) for arg in expr.args)
                and len(expr.keywords) <= (1 if name == "cute.math.max" else 0)
                and all(
                    kw.arg == "propagate_nan"
                    and isinstance(kw.value, ast.Constant)
                    and type(kw.value.value) is bool
                    for kw in expr.keywords
                )
            )
        if _dtype(expr) is not None:
            assert isinstance(expr.func, ast.Attribute)
            return _pure(expr.func.value, vectors)
    return False


def _raw_vector_load(expr: ast.AST) -> bool:
    """Only the generated immutable vector load, with effect-free arguments."""
    if (
        not isinstance(expr, ast.Call)
        or ast.unparse(expr.func) != "cute.arch.load"
        or len(expr.args) != 2
        or any(
            kw.arg not in {"cop", "level1_eviction_priority"}
            or not isinstance(kw.value, ast.Constant)
            or not isinstance(kw.value.value, str)
            for kw in expr.keywords
        )
    ):
        return False
    vector_type = expr.args[1]
    if (
        not isinstance(vector_type, ast.Call)
        or ast.unparse(vector_type.func) != "ir.VectorType.get"
        or len(vector_type.args) != 2
        or vector_type.keywords
        or not isinstance(vector_type.args[0], ast.List)
        or len(vector_type.args[0].elts) != 1
        or not isinstance(vector_type.args[0].elts[0], ast.Constant)
        or type(vector_type.args[0].elts[0].value) is not int
        or vector_type.args[0].elts[0].value < 1
        or not isinstance(vector_type.args[1], ast.Attribute)
        or vector_type.args[1].attr != "mlir_type"
        or ast.unparse(vector_type.args[1].value)
        not in {f"cutlass.{dtype}" for dtype in _SCALAR_TYPES}
    ):
        return False
    for node in ast.walk(expr.args[0]):
        if isinstance(node, ast.Call):
            if _scalar_cast(node) is not None:
                continue
            if (
                ast.unparse(node.func)
                in {"cute.arch.thread_idx", "cute.arch.block_idx"}
                and not node.args
                and not node.keywords
            ):
                continue
            return False
        if isinstance(node, (ast.NamedExpr, ast.Lambda, ast.comprehension)):
            return False
    return True


def _effect_free_loop(
    loop: ast.For,
    vectors: set[str],
    caches: set[str],
    collective_stmts: set[ast.stmt],
) -> bool:
    for statement in loop.body:
        if statement in collective_stmts:
            continue
        if isinstance(statement, ast.For):
            if _lane_count(statement) is None or not _effect_free_loop(
                statement, vectors, caches, collective_stmts
            ):
                return False
            continue
        assigned = _assignment(statement)
        if assigned is not None:
            name, value = assigned
            if name in vectors and _raw_vector_load(value):
                continue
            if _pure(value, vectors):
                continue
            return False
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Subscript)
            and isinstance(statement.targets[0].value, ast.Name)
            and statement.targets[0].value.id in caches
            and _pure(statement.targets[0].slice, vectors)
            and _pure(statement.value, vectors)
        ):
            continue
        return False
    return True


def _copy(expr: ast.expr) -> ast.expr:
    return ast.parse(ast.unparse(expr), mode="eval").body


class _Recipe:
    def __init__(
        self,
        definitions: dict[str, ast.expr],
        constants: dict[str, int],
        aliases: dict[str, str],
        written: set[str],
        vectors: set[str],
    ) -> None:
        self.definitions = definitions
        self.constants = constants
        self.aliases = aliases
        self.written = written
        self.vectors = vectors

    def expand(self, expr: ast.expr) -> ast.expr | None:
        if isinstance(expr, ast.Name):
            if expr.id in self.aliases:
                return ast.Name(id=self.aliases[expr.id], ctx=ast.Load())
            if expr.id in self.constants:
                return ast.Constant(value=self.constants[expr.id])
            definition = self.definitions.get(expr.id)
            if definition is not None:
                # Definitions are snapshots made at their assignment, not
                # aliases re-evaluated after a later scalar overwrite.
                return _copy(definition)
            if expr.id in self.written and expr.id not in self.vectors:
                return None
            return _copy(expr)
        if isinstance(expr, ast.Constant):
            return _copy(expr)
        if isinstance(expr, ast.Compare):
            if len(expr.ops) != 1 or len(expr.comparators) != 1:
                return None
            left = self.expand(expr.left)
            right = self.expand(expr.comparators[0])
            if not (
                isinstance(left, ast.Constant)
                and type(left.value) is int
                and isinstance(right, ast.Constant)
                and type(right.value) is int
                and isinstance(expr.ops[0], (ast.Eq, ast.NotEq))
            ):
                return None
            equal = left.value == right.value
            return ast.Constant(
                value=equal if isinstance(expr.ops[0], ast.Eq) else not equal
            )
        if isinstance(expr, ast.IfExp):
            test = self.expand(expr.test)
            if not isinstance(test, ast.Constant) or type(test.value) is not bool:
                return None
            return self.expand(expr.body if test.value else expr.orelse)
        if _scalar_cast(expr) is not None:
            assert isinstance(expr, ast.Call)
            value = self.expand(expr.args[0])
            if value is None:
                return None
            dtype = _scalar_cast(expr)
            if (
                dtype == "Boolean"
                and isinstance(value, ast.Constant)
                and type(value.value) is bool
            ):
                return value
            if dtype in _FLOAT_TYPES and _dtype(value) == dtype:
                return value
            result = _copy(expr)
            assert isinstance(result, ast.Call)
            result.args = [value]
            return result
        if isinstance(expr, ast.Subscript):
            if (
                not isinstance(expr.value, ast.Name)
                or expr.value.id not in self.vectors
            ):
                return None
            index = self.expand(expr.slice)
            if index is None:
                return None
            return ast.Subscript(value=_copy(expr.value), slice=index, ctx=ast.Load())
        if isinstance(expr, ast.Call) and _dtype(expr) is not None:
            assert isinstance(expr.func, ast.Attribute)
            value = self.expand(expr.func.value)
            if value is None:
                return None
            result = _copy(expr)
            assert isinstance(result, ast.Call) and isinstance(
                result.func, ast.Attribute
            )
            result.func.value = value
            return result
        if (
            isinstance(expr, ast.Call)
            and ast.unparse(expr.func) == "float"
            and _pure(expr, self.vectors)
        ):
            return _copy(expr)
        return None


@dataclass
class _Fold:
    init: ast.Assign
    update: ast.Assign
    domain: tuple[str, ...]
    contribution: str


def _fold(
    loop: ast.For,
    accumulator: str,
    initial: ast.Assign,
    constants: dict[str, int],
    writes: Counter[str],
    vectors: set[str],
    iteration_prefix: str,
) -> _Fold | None:
    if writes[accumulator] != 2:
        return None
    found: list[_Fold] = []

    def walk(
        body: list[ast.stmt],
        definitions: dict[str, ast.expr],
        loops: list[ast.For],
        available_vectors: set[str],
    ) -> bool:
        local = dict(definitions)
        available_vectors = set(available_vectors)
        # A name written by this repeated body is loop-carried until a
        # preceding assignment in the current iteration proves its value.
        # An initializer from the parent describes only iteration zero.
        for statement in body:
            for node in ast.walk(statement):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    local.pop(node.id, None)
                    available_vectors.discard(node.id)
        aliases = {
            inner.target.id: f"{iteration_prefix}_{index}"
            for index, inner in enumerate(loops)
            if isinstance(inner.target, ast.Name)
        }
        for stmt in body:
            assigned = _assignment(stmt)
            if assigned is not None:
                name, value = assigned
                recipe = _Recipe(
                    local, constants, aliases, set(writes), available_vectors
                )
                if name == accumulator and stmt is not initial:
                    if not (
                        isinstance(stmt, ast.Assign)
                        and isinstance(value, ast.Call)
                        and ast.unparse(value.func) == "cute.arch.fmax"
                        and len(value.args) == 2
                        and not value.keywords
                        and isinstance(value.args[0], ast.Name)
                        and value.args[0].id == accumulator
                        and loops
                    ):
                        return False
                    domain = []
                    for inner in loops:
                        count = _lane_count(inner)
                        if count is None:
                            return False
                        domain.append(ast.unparse(inner.iter))
                    contribution = recipe.expand(value.args[1])
                    if contribution is None:
                        return False
                    found.append(
                        _Fold(
                            initial,
                            stmt,
                            tuple(domain),
                            ast.dump(contribution),
                        )
                    )
                if name in vectors and _raw_vector_load(value):
                    available_vectors.add(name)
                snapshot = (
                    recipe.expand(value) if _pure(value, available_vectors) else None
                )
                if snapshot is None:
                    local.pop(name, None)
                else:
                    local[name] = snapshot
            elif isinstance(stmt, ast.For):
                if (
                    _lane_count(stmt) is None
                    or not isinstance(stmt.target, ast.Name)
                    or stmt.target.id in aliases
                    or stmt.target.id in constants
                    or not walk(stmt.body, local, [*loops, stmt], available_vectors)
                ):
                    return False
                # Values defined in a repeated child body are not definite
                # recipes for the parent's next statement. In particular a
                # last-iteration vector cannot stand for every earlier lane.
                for node in ast.walk(stmt):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        local.pop(node.id, None)
                        available_vectors.discard(node.id)
            elif any(
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Store)
                and node.id == accumulator
                for node in ast.walk(stmt)
            ):
                return False
        return True

    return found[0] if walk(loop.body, {}, [], set()) and len(found) == 1 else None


@dataclass
class _Carry:
    accumulator: str
    carry: str
    nodes: list[ast.stmt]
    signature: str
    call: ast.Call


def _carry(
    loop: ast.For,
    index: int,
    renames: dict[str, str],
    reads: Counter[str],
    writes: Counter[str],
) -> _Carry | None:
    assigned = _assignment(loop.body[index])
    if assigned is None:
        return None
    result, value = assigned
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "_cute_grouped_reduce_cluster"
        and len(value.args) == 6
        and isinstance(value.args[0], ast.Name)
        and isinstance(value.args[1], ast.Constant)
        and value.args[1].value == "max"
        and all(isinstance(arg, ast.Name) for arg in value.args[4:])
        and {kw.arg for kw in value.keywords} == {"group_span", "cluster_n"}
        and len(value.keywords) == 2
        and all(
            isinstance(kw.value, ast.Constant)
            and type(kw.value.value) is int
            and kw.value.value > 1
            for kw in value.keywords
        )
        and reads[result] == 1
        and writes[result] == 1
    ):
        return None
    chain = result
    definitions: dict[str, ast.expr] = {
        result: ast.Name(id="_reduction_result", ctx=ast.Load())
    }
    nodes = [loop.body[index]]
    for stmt in loop.body[index + 1 :]:
        assigned = _assignment(stmt)
        if assigned is None:
            return None
        target, expr = assigned
        matched = _match_carry_update(stmt, chain, "max")
        if matched is not None:
            carry, source = matched
            canon = renames.get(carry, carry)
            seen: set[str] = set()
            while renames.get(source, source) != canon:
                if source in seen or source not in definitions:
                    return None
                seen.add(source)
                rhs = definitions[source]
                if not isinstance(rhs, ast.Name):
                    return None
                source = rhs.id
            if any(reads[name] != 1 for name in definitions if name != source):
                return None

            # Compare every cast and max keyword, using placeholders only for
            # the independently initialized carry and the collective result.
            class Replace(ast.NodeTransformer):
                def visit_Name(self, node: ast.Name, carry: str = canon) -> ast.AST:
                    if node.id == result:
                        return ast.Name(id="_reduction_result", ctx=ast.Load())
                    if renames.get(node.id, node.id) == carry:
                        return ast.Name(id="_carry", ctx=ast.Load())
                    if node.id in definitions:
                        return self.visit(_copy(definitions[node.id]))
                    return node

            normalized = Replace().visit(_copy(expr))
            nodes.append(stmt)
            return _Carry(value.args[0].id, canon, nodes, ast.dump(normalized), value)
        if not (isinstance(expr, ast.Name) or _scalar_cast(expr) in _FLOAT_TYPES):
            return None
        if target in definitions or writes[target] != 1:
            return None
        # A result chain consists solely of scalar casts. The other temporary
        # assignments must be plain aliases of the carry's incoming value.
        inner = expr
        while _scalar_cast(inner) in _FLOAT_TYPES:
            assert isinstance(inner, ast.Call)
            inner = inner.args[0]
        if not isinstance(inner, ast.Name):
            return None
        if inner.id == chain:
            chain = target
        elif not isinstance(expr, ast.Name):
            return None
        definitions[target] = expr
        nodes.append(stmt)
    return None


def eliminate_duplicate_cluster_maxima(
    body: list[ast.stmt],
    constexpr_values: dict[str, int],
    rename_groups: dict[str, str],
) -> list[ast.stmt]:
    """Fold equal typed max carries only within an exact one-trip loop."""
    # Validate before mutating. Reuse the original nodes so backend metadata
    # on lane loops, vector wrappers, and expressions survives a successful
    # rewrite. A failed proof returns the untouched input.
    work = list(body)
    if any(
        isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Lambda,
                ast.comprehension,
            ),
        )
        for statement in work
        for node in ast.walk(statement)
    ):
        return body
    reads = Counter(
        node.id
        for stmt in work
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    )
    writes = Counter(
        node.id
        for stmt in work
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    )
    if any(writes[name] for name in ("cute", "cutlass", "ir", "float", "range")):
        return body
    iteration_prefix = "_iteration"
    while any(
        name.startswith(iteration_prefix) for name in reads.keys() | writes.keys()
    ):
        iteration_prefix += "_"
    vectors = {
        name
        for stmt in work
        for node in ast.walk(stmt)
        if (assigned := _assignment(node)) is not None
        for name, value in [assigned]
        if _raw_vector_load(value) and writes[name] == 1
    }
    # A vector value is immutable only while no element write can alter it.
    vectors -= {
        node.value.id
        for stmt in work
        for node in ast.walk(stmt)
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, (ast.Store, ast.Del))
        and isinstance(node.value, ast.Name)
    }
    caches = {
        name
        for stmt in work
        if (assigned := _assignment(stmt)) is not None
        for name, value in [assigned]
        if isinstance(value, ast.Call)
        and ast.unparse(value.func) == "cute.make_rmem_tensor"
        and writes[name] == 1
    }
    for top_index, loop in enumerate(work):
        if not isinstance(loop, ast.For) or not _one_zero_iteration(
            loop, constexpr_values, writes
        ):
            continue
        assert isinstance(loop.target, ast.Name)
        constants = {
            name: value for name, value in constexpr_values.items() if not writes[name]
        }
        constants[loop.target.id] = 0
        carries = [
            candidate
            for index in range(len(loop.body))
            if (candidate := _carry(loop, index, rename_groups, reads, writes))
            is not None
        ]
        if len(carries) != 2:
            continue
        first, second = carries
        if first.signature != second.signature or first.carry == second.carry:
            continue
        # Future backend renames are mutable aliases. The carry chains model
        # theirs explicitly; an alias elsewhere in the recipe needs a more
        # general dataflow proof and therefore keeps both collectives.
        if any(
            isinstance(node, ast.Name)
            and rename_groups.get(node.id, node.id) != node.id
            and rename_groups[node.id] not in {first.carry, second.carry}
            for node in ast.walk(loop)
        ):
            continue
        if not _effect_free_loop(
            loop, vectors, caches, {first.nodes[0], second.nodes[0]}
        ):
            continue
        initializers = {}
        for stmt in work[:top_index]:
            assigned = _assignment(stmt)
            if assigned is not None:
                name, value = assigned
                canonical = rename_groups.get(name, name)
                if canonical in {first.carry, second.carry}:
                    initializers[canonical] = stmt
        if set(initializers) != {first.carry, second.carry}:
            continue
        first_init = _assignment(initializers[first.carry])
        second_init = _assignment(initializers[second.carry])
        assert first_init is not None and second_init is not None
        if (
            ast.dump(first_init[1]) != ast.dump(second_init[1])
            or ast.unparse(first_init[1]) != "cutlass.Float32(float('-inf'))"
        ):
            continue
        # The carried states have exactly their initializer and one update;
        # no other branch, recurrence, or store can change either state.
        if any(
            sum(
                count
                for name, count in writes.items()
                if rename_groups.get(name, name) == carry.carry
            )
            != 2
            for carry in carries
        ):
            continue
        fold_sites = []
        for carry in carries:
            acc_init = next(
                (
                    stmt
                    for stmt in loop.body
                    if (assigned := _assignment(stmt)) is not None
                    and assigned[0] == carry.accumulator
                ),
                None,
            )
            if (
                not isinstance(acc_init, ast.Assign)
                or ast.unparse(acc_init.value) != "cutlass.Float32(float('-inf'))"
            ):
                break
            folded = _fold(
                loop,
                carry.accumulator,
                acc_init,
                constants,
                writes,
                vectors,
                iteration_prefix,
            )
            if folded is None or reads[carry.accumulator] != 2:
                break
            fold_sites.append(folded)
        if len(fold_sites) != 2 or (
            fold_sites[0].domain,
            fold_sites[0].contribution,
        ) != (fold_sites[1].domain, fold_sites[1].contribution):
            continue
        # Collective geometry and identity are equal; lane argument aliases
        # are permitted only for identical preceding scalar definitions.
        lane_args = []
        for carry in carries:
            arg = carry.call.args[3]
            definitions = {
                name: value
                for stmt in loop.body[: loop.body.index(carry.nodes[0])]
                if (assigned := _assignment(stmt)) is not None
                for name, value in [assigned]
            }
            if isinstance(arg, ast.Name) and arg.id in definitions:
                if writes[arg.id] != 1:
                    break
                arg = definitions[arg.id]
            if not _pure(arg, set()) or any(
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and writes[node.id]
                for node in ast.walk(arg)
            ):
                break
            lane_args.append(ast.dump(arg))
        if (
            len(lane_args) != 2
            or lane_args[0] != lane_args[1]
            or ast.dump(ast.Tuple(elts=first.call.args[1:3], ctx=ast.Load()))
            != ast.dump(ast.Tuple(elts=second.call.args[1:3], ctx=ast.Load()))
            or [ast.dump(kw) for kw in first.call.keywords]
            != [ast.dump(kw) for kw in second.call.keywords]
        ):
            continue
        second_nodes = set(second.nodes) | {
            fold_sites[1].init,
            fold_sites[1].update,
            initializers[second.carry],
        }
        # Every use of the second carry before the loop is its initializer;
        # inside the loop, its only use is the closed carry-update chain.
        if any(
            rename_groups.get(node.id, node.id) == second.carry
            for stmt in work[:top_index]
            if stmt is not initializers[second.carry]
            for node in ast.walk(stmt)
            if isinstance(node, ast.Name)
        ):
            continue
        closed = {id(node) for stmt in second.nodes for node in ast.walk(stmt)}
        if any(
            rename_groups.get(node.id, node.id) == second.carry
            and id(node) not in closed
            for node in ast.walk(loop)
            if isinstance(node, ast.Name)
        ):
            continue
        buf, barrier = (
            arg.id for arg in second.call.args[4:] if isinstance(arg, ast.Name)
        )
        private_preamble = []
        for stmt in work[:top_index]:
            assigned = _assignment(stmt)
            if assigned is not None and assigned[0] in {buf, barrier}:
                if (
                    not isinstance(assigned[1], ast.Call)
                    or ast.unparse(assigned[1].func) != "cute.arch.alloc_smem"
                    or len(assigned[1].args) != 2
                    or assigned[1].keywords
                    or ast.unparse(assigned[1].args[0])
                    != ("cutlass.Float32" if assigned[0] == buf else "cutlass.Int64")
                    or not isinstance(assigned[1].args[1], ast.Constant)
                    or type(assigned[1].args[1].value) is not int
                    or assigned[1].args[1].value < 1
                    or writes[assigned[0]] != 1
                ):
                    break
                private_preamble.append(stmt)
            elif isinstance(stmt, ast.If) and any(
                isinstance(node, ast.Name) and node.id == barrier
                for node in ast.walk(stmt)
            ):
                if (
                    len(stmt.body) != 1
                    or any(not isinstance(node, ast.Pass) for node in stmt.orelse)
                    or not _pure(stmt.test, vectors)
                    or not isinstance(stmt.body[0], ast.Expr)
                    or not isinstance(stmt.body[0].value, ast.Call)
                    or ast.unparse(stmt.body[0].value.func) != "cute.arch.mbarrier_init"
                    or len(stmt.body[0].value.args) != 2
                    or stmt.body[0].value.keywords
                    or not isinstance(stmt.body[0].value.args[0], ast.Name)
                    or stmt.body[0].value.args[0].id != barrier
                    or not isinstance(stmt.body[0].value.args[1], ast.Constant)
                    or type(stmt.body[0].value.args[1].value) is not int
                    or stmt.body[0].value.args[1].value != 1
                ):
                    break
                private_preamble.append(stmt)
        else:
            if len(private_preamble) != 3:
                continue
            allowed = {
                id(node)
                for stmt in [*private_preamble, second.nodes[0]]
                for node in ast.walk(stmt)
            }
            if any(
                node.id in {buf, barrier} and id(node) not in allowed
                for stmt in work
                for node in ast.walk(stmt)
                if isinstance(node, ast.Name)
            ):
                continue
            second_nodes.update(private_preamble)

            class Rewrite(ast.NodeTransformer):
                def visit(
                    self, node: ast.AST, removed: set[ast.stmt] = second_nodes
                ) -> ast.AST | None:
                    if node in removed:
                        return None
                    return super().visit(node)

                def visit_For(self, node: ast.For) -> ast.AST:
                    self.generic_visit(node)
                    if not node.body:
                        node.body = [ast.Pass()]
                    return node

            # Only consumers after this exact logical loop receive the alias.
            for stmt in work[top_index + 1 :]:
                for node in ast.walk(stmt):
                    if (
                        isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load)
                        and rename_groups.get(node.id, node.id) == second.carry
                    ):
                        node.id = first.carry
            rewritten = Rewrite().visit(ast.Module(body=work, type_ignores=[]))
            assert isinstance(rewritten, ast.Module)
            # Discard only dead, pure scalar recipes. Preserve memory reads,
            # stores, cache mutations, and every unmodelled expression.
            changed = True
            while changed:
                changed = False
                live = Counter(
                    rename_groups.get(node.id, node.id)
                    for node in ast.walk(rewritten)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                )

                class Prune(ast.NodeTransformer):
                    def visit_Assign(
                        self, node: ast.Assign, read_counts: Counter[str] = live
                    ) -> ast.AST | None:
                        nonlocal changed
                        assigned = _assignment(node)
                        if (
                            assigned is not None
                            and not read_counts[
                                rename_groups.get(assigned[0], assigned[0])
                            ]
                            and _pure(assigned[1], vectors)
                        ):
                            changed = True
                            return None
                        return node

                    def visit_For(self, node: ast.For) -> ast.AST:
                        self.generic_visit(node)
                        if not node.body:
                            node.body = [ast.Pass()]
                        return node

                Prune().visit(loop)
            ast.fix_missing_locations(rewritten)
            return rewritten.body
    return body
