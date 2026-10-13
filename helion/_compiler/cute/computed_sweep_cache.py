"""Retain complete typed scalar results between identical reduction sweeps.

This is a separate opt-in policy from raw load caching. Value identity uses a
bounded, versioned DAG; address identity and memory movement retain the raw
fuser's binding, alias and per-lane snapshot proofs.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING

from ..ast_extension import statement_from_string
from ..ast_read_writes import ReadWrites
from .fuse_two_pass_loads import _PERSISTENT_BRANCH_VEC_LOAD
from .fuse_two_pass_loads import _PERSISTENT_BRANCH_VEC_STORE
from .fuse_two_pass_loads import _canonical_load_text
from .fuse_two_pass_loads import _dtype_from_default
from .fuse_two_pass_loads import _is_store_call
from .fuse_two_pass_loads import _LexicalMemoryOrder
from .fuse_two_pass_loads import _load_kind
from .fuse_two_pass_loads import _range_bounds
from .fuse_two_pass_loads import _scalar_bindings
from .fuse_two_pass_loads import _tensor_arg_roots
from .fuse_two_pass_loads import _trip_count_for
from .fuse_two_pass_loads import _unmasked_load_tensor_dtype
from .fuse_two_pass_loads import _unwrap_persistent_branch_vec_load

if TYPE_CHECKING:
    from .fuse_two_pass_loads import _CuteFuseTwoPassLoads

MAX_CACHE_ELEMENTS = 64
_MAX_VALUE_NODES = 128
_MAX_DEFINITIONS = 96
_FP32 = "cutlass.Float32"
_CASTS = frozenset(
    (_FP32, "cutlass.Float16", "cutlass.BFloat16", "cutlass.Int32", "cutlass.Int64")
)
_COMPARES = frozenset(
    (
        "operator.le",
        "operator.lt",
        "operator.ge",
        "operator.gt",
        "operator.eq",
        "operator.ne",
    )
)
_INDEX_CALLS = frozenset(
    (
        "cute.arch.thread_idx",
        "cute.arch.block_idx",
        "cute.arch.lane_idx",
        "cute.arch.warp_idx",
    )
)
_MATH_CALLS = frozenset(("cute.math.exp2", "cute.math.log"))


@dataclass(frozen=True)
class _Load:
    key: str
    statement: ast.Assign
    roots: frozenset[str]


@dataclass(frozen=True)
class _Value:
    key: int
    dtype: str
    loads: frozenset[_Load]
    definitions: frozenset[ast.Assign]
    nodes: frozenset[int]


def _target(statement: ast.stmt) -> str | None:
    if (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    ):
        return statement.targets[0].id
    return None


def _safe_expression(node: ast.AST, tensors: set[str]) -> bool:
    """No unknown calls or mutable local-array reads in a crossed region."""
    for child in ast.walk(node):
        if isinstance(child, (ast.Lambda, ast.NamedExpr, ast.Await, ast.Yield)):
            return False
        if isinstance(child, ast.Subscript):
            metadata = child.value
            tensor_metadata = False
            if isinstance(metadata, ast.Attribute) and metadata.attr in (
                "stride",
                "shape",
            ):
                tensor_metadata = True
                metadata = metadata.value
                if isinstance(metadata, ast.Attribute) and metadata.attr == "layout":
                    metadata = metadata.value
            # Only immutable tensor metadata or an index intrinsic's tuple.
            if not (
                tensor_metadata
                and isinstance(metadata, ast.Name)
                and metadata.id in tensors
                or isinstance(child.value, ast.Call)
                and ast.unparse(child.value.func) in _INDEX_CALLS
            ):
                return False
        if not isinstance(child, ast.Call):
            continue
        name = ast.unparse(child.func)
        if name in _CASTS | _COMPARES | _INDEX_CALLS | _MATH_CALLS:
            continue
        if _load_kind(child) is not None:
            if _tensor_arg_roots(child, tensors) is not None:
                continue
        return False
    return True


def _safe_region(statements: list[ast.stmt], tensors: set[str]) -> bool:
    for statement in statements:
        if _target(statement) is not None:
            assert isinstance(statement, ast.Assign)
            value = statement.value
            # These complete an original reduction, but never enter a cached
            # expression. Unknown calls, atomics and synchronization decline.
            if isinstance(value, ast.Call) and ast.unparse(value.func) == _FP32:
                if len(value.args) == 1 and isinstance(value.args[0], ast.Call):
                    reduction = value.args[0]
                    if ast.unparse(reduction.func) == "cute.arch.warp_reduction_sum":
                        if (
                            not value.keywords
                            and all(
                                keyword.arg is not None
                                for keyword in reduction.keywords
                            )
                            and all(
                                _safe_expression(arg, tensors)
                                for arg in (
                                    *reduction.args,
                                    *(keyword.value for keyword in reduction.keywords),
                                )
                            )
                        ):
                            continue
            if not _safe_expression(value, tensors):
                return False
        elif isinstance(statement, ast.If):
            if not _safe_expression(statement.test, tensors) or not _safe_region(
                [*statement.body, *statement.orelse], tensors
            ):
                return False
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if (
                isinstance(call.func, ast.Name)
                and call.func.id == _PERSISTENT_BRANCH_VEC_STORE
                and len(call.args) == 6
                and not call.keywords
                and all(
                    isinstance(arg, ast.Constant) and type(arg.value) is int
                    for arg in call.args[:2]
                )
                and isinstance(call.args[2], ast.Constant)
                and call.args[2].value in _CASTS
                and _tensor_arg_roots(call.args[3], tensors) is not None
            ):
                # This compiler marker still denotes one scalar store. Its
                # eventual vector transaction retains its own complete-mask,
                # alignment and alias proofs in persistent_branch_vec.
                if all(_safe_expression(arg, tensors) for arg in call.args[3:]):
                    continue
                return False
            if not (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "store"
                and len(call.args) == 1
                and not call.keywords
                and _is_store_call(call)
                and _tensor_arg_roots(call, tensors) is not None
            ):
                return False
            # Check pointer and value computations, excluding just the store.
            if not all(_safe_expression(arg, tensors) for arg in call.args):
                return False
            if isinstance(call.func, ast.Attribute) and not _safe_expression(
                call.func.value, tensors
            ):
                return False
        elif not isinstance(statement, ast.Pass):
            return False
    return True


class _Values:
    def __init__(self, owner: _CuteFuseTwoPassLoads) -> None:
        self.owner = owner
        self.keys: dict[tuple[object, ...], int] = {}

    def make(
        self,
        operation: tuple[object, ...],
        dtype: str,
        children: tuple[_Value, ...] = (),
        load: _Load | None = None,
    ) -> _Value | None:
        key = (dtype, *operation, *(child.key for child in children))
        number = self.keys.setdefault(key, len(self.keys))
        nodes = frozenset((number,)).union(*(child.nodes for child in children))
        definitions = frozenset().union(*(child.definitions for child in children))
        if len(nodes) > _MAX_VALUE_NODES or len(definitions) > _MAX_DEFINITIONS:
            return None
        loads = frozenset((load,)) if load is not None else frozenset()
        return _Value(
            number,
            dtype,
            loads.union(*(child.loads for child in children)),
            definitions,
            nodes,
        )

    def expression(
        self,
        node: ast.expr,
        statement: ast.Assign,
        values: dict[str, _Value],
        bindings: dict[str, ast.expr],
    ) -> _Value | None:
        scalar = _unwrap_persistent_branch_vec_load(node)
        assert isinstance(scalar, ast.expr)
        if scalar is not node:
            assert isinstance(node, ast.Call)
            if (
                node.keywords
                or not all(
                    isinstance(arg, ast.Constant) and type(arg.value) is int
                    for arg in node.args[:2]
                )
                or not isinstance(node.args[2], ast.Constant)
                or node.args[2].value not in _CASTS
                or not isinstance(node.args[3], ast.Constant)
                or not isinstance(node.args[3].value, str)
            ):
                return None
            # Match the original scalar value, while keeping the statement
            # and marker identity for memory-order and snapshot proofs.
            node = scalar
        if isinstance(node, ast.Name):
            return values.get(node.id)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool):
            return self.make(
                ("literal", type(node.value).__name__, repr(node.value)),
                type(node.value).__name__,
            )
        kind = _load_kind(node, self.owner._tensor_dtypes)
        if kind in ("masked", "unmasked"):
            raw = node.body if isinstance(node, ast.IfExp) else node
            dtype = _unmasked_load_tensor_dtype(raw, self.owner._tensor_dtypes)
            if (
                dtype is None
                or dtype not in _CASTS
                or _load_kind(raw, self.owner._tensor_dtypes) != "unmasked"
                or not _safe_expression(node, set(self.owner._tensor_dtypes))
            ):
                return None
            assert isinstance(raw, ast.Call)
            if (
                ast.unparse(raw.func) == "cute.arch.load"
                and len(raw.args) != 2
                or ast.unparse(raw.func) != "cute.arch.load"
                and raw.args
            ):
                return None
            if any(
                keyword.arg
                not in ("cop", "level1_eviction_priority", "level2_eviction_priority")
                or not isinstance(keyword.value, ast.Constant)
                or not isinstance(keyword.value.value, str)
                for keyword in raw.keywords
            ):
                return None
            if isinstance(node, ast.IfExp) and (
                _dtype_from_default(node.orelse) != dtype
                or not isinstance(node.orelse, ast.Call)
                or len(node.orelse.args) != 1
                or node.orelse.keywords
            ):
                return None
            roots = _tensor_arg_roots(node, set(self.owner._tensor_dtypes))
            if roots is None or len(roots) != 1:
                return None
            key = _canonical_load_text(node, {}, bindings)
            return self.make(("load", key), dtype, load=_Load(key, statement, roots))
        children: list[ast.expr]
        operation: tuple[object, ...]
        forced: str | None = None
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
            children = [node.left, node.right]
            operation = (type(node.op).__name__,)
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            children = [node.operand]
            operation = (type(node.op).__name__,)
        elif isinstance(node, ast.IfExp):
            children = [node.test, node.body, node.orelse]
            operation = ("select",)
        elif (
            isinstance(node, ast.Compare)
            and len(node.ops) == len(node.comparators) == 1
            and isinstance(
                node.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
            )
        ):
            children = [node.left, node.comparators[0]]
            operation, forced = (type(node.ops[0]).__name__,), "bool"
        elif isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name in _CASTS and len(node.args) == 1 and not node.keywords:
                children, operation, forced = node.args, (name,), name
            elif name in _MATH_CALLS and len(node.args) == 1:
                children, operation, forced = node.args, (name,), _FP32
                if node.keywords:
                    if (
                        len(node.keywords) != 1
                        or node.keywords[0].arg != "fastmath"
                        or not isinstance(node.keywords[0].value, ast.Constant)
                        or type(node.keywords[0].value.value) is not bool
                    ):
                        return None
                    # Reuse only the same explicit arithmetic policy. Keeping
                    # absence distinct also preserves the old expression keys.
                    operation = (name, "fastmath", node.keywords[0].value.value)
            elif name in _COMPARES and len(node.args) == 2 and not node.keywords:
                children, operation, forced = node.args, (name,), "bool"
            else:
                return None
        else:
            return None
        operands = [
            self.expression(child, statement, values, bindings) for child in children
        ]
        if any(value is None for value in operands):
            return None
        args = tuple(value for value in operands if value is not None)
        types = {value.dtype for value in args}
        if forced is not None:
            if operation[0] in _MATH_CALLS and args[0].dtype != _FP32:
                return None
            dtype = forced
        elif operation == ("select",):
            if args[0].dtype != "bool" or args[1].dtype != args[2].dtype:
                return None
            dtype = args[1].dtype
        elif types <= {_FP32, "float", "int"} and _FP32 in types:
            dtype = _FP32
        else:
            # No implicit half/int/FP64 arithmetic inference.
            return None
        return self.make(operation, dtype, args)

    def step(
        self,
        statement: ast.stmt,
        values: dict[str, _Value],
        bindings: dict[str, ast.expr],
    ) -> _Value | None:
        name = _target(statement)
        value = None
        if name is not None:
            assert isinstance(statement, ast.Assign)
            value = self.expression(statement.value, statement, values, bindings)
        for written in _scalar_bindings(statement):
            values.pop(written, None)
        if value is not None and name is not None:
            assert isinstance(statement, ast.Assign)
            if len(value.definitions | {statement}) > _MAX_DEFINITIONS:
                return None
            value = _Value(
                value.key,
                value.dtype,
                value.loads,
                value.definitions | {statement},
                value.nodes,
            )
            values[name] = value
        return value


def _loop_values(
    dag: _Values,
    body: list[ast.stmt],
    loop: ast.For,
    canonical: ast.For,
    inherited: dict[str, ast.expr],
) -> list[tuple[ast.Assign, _Value]]:
    owner = dag.owner
    values: dict[str, _Value] = {}
    bindings = dict(inherited)
    for statement in body[: body.index(loop)]:
        dag.step(statement, values, bindings)
        bindings = owner._pure_definitions_before([statement], 1, bindings)
    bindings = owner._container_definitions(
        loop, loop.body, bindings, canonical, canonical.body
    )
    if bindings is None:
        return []
    for name in _scalar_bindings(loop):
        values.pop(name, None)
    results = []
    for statement in loop.body:
        value = dag.step(statement, values, bindings)
        if isinstance(statement, ast.Assign) and value is not None:
            results.append((statement, value))
        bindings = owner._pure_definitions_before([statement], 1, bindings)
    return results


def _valid_memory(
    owner: _CuteFuseTwoPassLoads,
    memory: _LexicalMemoryOrder,
    first: ast.For,
    second: ast.For,
    producer: _Value,
    consumer: _Value,
) -> bool:
    for load in consumer.loads:
        matches = [leaf for leaf in producer.loads if leaf.key == load.key]
        if not matches:
            return False
        # A retained outer immutable leaf may be the very same occurrence.
        if not any(
            leaf.statement is load.statement
            or not memory.has_aliasing_store_between(
                leaf.statement, load.statement, load.roots
            )
            for leaf in matches
        ):
            return False
        if memory.has_aliasing_store_in(first, load.roots):
            return False
        if memory.has_aliasing_store_in(second, load.roots):
            if (
                load.statement not in second.body
                or not owner._consumer_store_preserves_snapshot(
                    second,
                    second.body,
                    second.body.index(load.statement),
                    load.statement,
                    load.roots,
                )
            ):
                return False
    return True


def _prune_dead(
    body: list[ast.stmt], removable: frozenset[ast.Assign], live_after: set[str]
) -> None:
    live = set(live_after)
    kept = []
    for statement in reversed(body):
        name = _target(statement)
        if statement in removable and name is not None and name not in live:
            continue
        reads = ReadWrites.from_ast(statement)
        if name is not None:
            live.discard(name)
        live.update(reads.reads)
        kept.append(statement)
    body[:] = reversed(kept)


def live_before(statements: list[ast.stmt], live_after: set[str]) -> set[str]:
    """Conservative scalar liveness, including zero-trip loops and branch exits."""
    live = set(live_after)
    for statement in reversed(statements):
        name = _target(statement)
        if name is not None:
            live.discard(name)
            live.update(ReadWrites.from_ast(statement).reads)
        elif isinstance(statement, ast.If):
            live = live_before(statement.body, live) | live_before(
                statement.orelse, live
            )
            live.update(ReadWrites.from_ast(statement.test).reads)
        elif isinstance(statement, ast.For):
            # The body may not execute. Its incoming values also survive a
            # back edge, even if a definition in a nested sweep is removed.
            inside = live_before(statement.body, live)
            inside.difference_update(_scalar_bindings(statement.target))
            live.update(inside)
            live.update(ReadWrites.from_ast(statement.iter).reads)
            live.update(live_before(statement.orelse, live))
        else:
            live.update(ReadWrites.from_ast(statement).reads)
    return live


def _hoist_immutable_address_loads(
    owner: _CuteFuseTwoPassLoads, body: list[ast.stmt], loop: ast.For
) -> None:
    """Expose invariant gather addresses to the existing vector marker lowering.

    Move only unconditional scalar loads from kernel-wide immutable storage.
    Their arguments must already be invariant at loop entry, their target must
    have one definition, and nothing may read its incoming value. The positive
    static trip count makes the final binding identical to the original loop.
    """
    assert isinstance(loop.target, ast.Name)
    tensors = set(owner._tensor_dtypes)
    if not _safe_region(loop.body, tensors):
        return
    required: set[str] = set()
    for statement in loop.body:
        for node in ast.walk(statement):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and len(node.args) == 6
                and not node.keywords
            ):
                continue
            if node.func.id == _PERSISTENT_BRANCH_VEC_LOAD:
                expressions = node.args[4:]
            elif node.func.id == _PERSISTENT_BRANCH_VEC_STORE:
                expressions = [node.args[3], node.args[5]]
            else:
                continue
            for expression in expressions:
                required.update(ReadWrites.from_ast(expression).reads)
    counts = Counter(name for stmt in loop.body for name in _scalar_bindings(stmt))
    writes = set(counts) | {loop.target.id}
    earlier_reads = set(ReadWrites.from_ast(loop.iter).reads)
    hoisted: list[ast.stmt] = []
    retained: list[ast.stmt] = []
    for statement in loop.body:
        name = _target(statement)
        if name is not None:
            assert isinstance(statement, ast.Assign)
            roots = _tensor_arg_roots(statement.value, tensors)
            if (
                name in required
                and counts[name] == 1
                and name not in earlier_reads
                and _load_kind(statement.value, owner._tensor_dtypes) == "unmasked"
                and roots is not None
                and roots <= owner._readonly_tensors
                and not (set(ReadWrites.from_ast(statement.value).reads) & writes)
                and _Values(owner).expression(statement.value, statement, {}, {})
                is not None
            ):
                hoisted.append(statement)
                continue
        earlier_reads.update(ReadWrites.from_ast(statement).reads)
        retained.append(statement)
    if hoisted:
        index = body.index(loop)
        body[index:index] = hoisted
        loop.body = retained


def cache_computed_sweeps(
    owner: _CuteFuseTwoPassLoads,
    body: list[ast.stmt],
    inherited: dict[str, ast.expr],
    escaping: set[str],
) -> None:
    """Select at most one maximal complete FP32 result per scalar sweep pair."""
    tensors = set(owner._tensor_dtypes)
    loops = [
        statement
        for statement in body
        if isinstance(statement, ast.For) and not statement.orelse
    ]
    for loop in loops:
        block = owner._reduction_block_id(loop)
        mode = os.environ.get(
            "HELION_FUSER_MODE",
            owner._reload_modes.get(block, "auto") if block is not None else "auto",
        )
        if mode != "computed":
            continue
        bounds = _range_bounds(loop.iter)
        if bounds is not None and isinstance(loop.target, ast.Name):
            trip = _trip_count_for(*bounds, owner._constexpr_values)
            if trip is not None and 1 <= trip <= MAX_CACHE_ELEMENTS:
                _hoist_immutable_address_loads(owner, body, loop)
    used: set[ast.For] = set()
    for first_index, first in enumerate(loops):
        block = owner._reduction_block_id(first)
        mode = os.environ.get(
            "HELION_FUSER_MODE",
            owner._reload_modes.get(block, "auto") if block is not None else "auto",
        )
        if mode != "computed" or first in used:
            continue
        bounds = _range_bounds(first.iter)
        if bounds is None or not isinstance(first.target, ast.Name):
            continue
        trip = _trip_count_for(*bounds, owner._constexpr_values)
        if (
            trip is None
            or not 1 <= trip <= MAX_CACHE_ELEMENTS - owner._computed_cache_elements
        ):
            continue
        # Keep initial scope to scalar loops; vector/nested protocols retain
        # their original code until their own complete result proof exists.
        if not _safe_region(first.body, tensors):
            continue
        for second in loops[first_index + 1 :]:
            if (
                second in used
                or ast.dump(first.target) != ast.dump(second.target)
                or ast.dump(first.iter) != ast.dump(second.iter)
            ):
                continue
            middle = body[body.index(first) + 1 : body.index(second)]
            if not _safe_region([*middle, *second.body], tensors):
                continue
            first_scope = owner._pure_definitions_before(
                body, body.index(first), inherited
            )
            second_scope = owner._pure_definitions_before(
                body, body.index(second), inherited
            )
            if _canonical_load_text(
                first.iter, {}, first_scope
            ) != _canonical_load_text(second.iter, {}, second_scope):
                continue
            dag = _Values(owner)
            first_values = _loop_values(dag, body, first, first, inherited)
            second_values = _loop_values(dag, body, second, first, inherited)
            candidates = []
            for consumer_statement, consumer in second_values:
                if (
                    consumer.dtype != _FP32
                    or not consumer.loads
                    or len(consumer.nodes) < 3
                ):
                    continue
                for producer_statement, producer in first_values:
                    if producer.key == consumer.key:
                        candidates.append(
                            (producer_statement, producer, consumer_statement, consumer)
                        )
            candidates.sort(key=lambda item: len(item[3].nodes), reverse=True)
            memory = _LexicalMemoryOrder(
                body, tensors, owner._proven_disjoint_tensor_pairs
            )
            for (
                producer_statement,
                producer,
                consumer_statement,
                consumer,
            ) in candidates:
                if not _valid_memory(owner, memory, first, second, producer, consumer):
                    continue
                cache = owner._new_cache_name().replace(
                    "_fuse_cache_", "_computed_cache_"
                )
                while cache in owner._reserved_names:
                    cache = owner._new_cache_name().replace(
                        "_fuse_cache_", "_computed_cache_"
                    )
                owner._reserved_names.add(cache)
                start, _, step = bounds
                slot = f"({first.target.id} - ({ast.unparse(start)})) // ({ast.unparse(step)})"
                owner._root_declarations.append(
                    statement_from_string(
                        f"{cache} = cute.make_rmem_tensor({trip}, cutlass.Float32)"
                    )
                )
                name = _target(producer_statement)
                assert name is not None
                first.body.insert(
                    first.body.index(producer_statement) + 1,
                    statement_from_string(f"{cache}[{slot}] = {name}"),
                )
                consumer_statement.value = ast.parse(
                    f"{cache}[{slot}]", mode="eval"
                ).body
                # Remove only the closed pure slice and preserve all remaining
                # uses, including address/control consumers and escaping names.
                tail = body[body.index(second) + 1 :]
                live_after = live_before(tail, escaping)
                _prune_dead(
                    second.body, consumer.definitions - {consumer_statement}, live_after
                )
                owner._computed_cache_elements += trip
                used.update((first, second))
                break
            if first in used:
                break
