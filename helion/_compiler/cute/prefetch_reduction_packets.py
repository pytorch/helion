"""Move bounded read-only consumer packets into a matching one-trip sweep.

Only raw register-cache plumbing changes. The existing scalar arithmetic and
collectives have already been lowered, and are never reconstructed here.
"""

from __future__ import annotations

import ast
from collections import Counter
from itertools import pairwise
import os
from typing import cast

from ..ast_extension import clone_ast
from ..ast_extension import statement_from_string
from .cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES
from .fuse_two_pass_loads import _CuteFuseTwoPassLoads
from .fuse_two_pass_loads import _LexicalMemoryOrder
from .fuse_two_pass_loads import _range_bounds
from .fuse_two_pass_loads import _rewrite_vec_extract
from .fuse_two_pass_loads import _static_int
from .fuse_two_pass_loads import _store_tensor_roots
from .fuse_two_pass_loads import _tensor_arg_roots
from .fuse_two_pass_loads import _tensor_roots_may_alias
from .fuse_two_pass_loads import _trip_count_for

# A conservative *cache* policy, in 32-bit register slots, not an occupancy
# guarantee. Include all explicit retained fragments, not just this input.
_CACHE_SLOTS = 128
_CASTS = {
    f"cutlass.{n}"
    for n in (
        "Int32",
        "Int64",
        "Uint16",
        "Uint32",
        "Float16",
        "BFloat16",
        "Float32",
        "Float64",
    )
}
_STORES = {"_cute_store_u16_vec", "_cute_store_u32_vec"}
_COLLECTIVES = {
    "cute.arch.sync_threads",
    "cute.arch.warp_reduction_sum",
    "_cute_grouped_reduce_shared_two_stage",
    "_cute_grouped_reduce_shared",
}
_PURE = _CASTS | {
    "range",
    "cutlass.range_constexpr",
    "cute.arch.thread_idx",
    "cute.arch.block_idx",
    "ir.VectorType.get",
    "cute.math.rsqrt",
    "cute.math.sqrt",
    "cute.math.exp",
    "cute.math.exp2",
    "cute.math.log",
    "cute.math.log2",
    "cute.math.abs",
    "cute.math.min",
    "cute.math.max",
    "cute.arch.fmin",
    "cute.arch.fmax",
    "cute.math.fma",
}


def _path(node: ast.AST) -> str:
    return ast.unparse(node)


def _names(node: ast.AST) -> set[str]:
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _assigned(node: ast.AST) -> set[str]:
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
    }


def _assignment(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    ):
        return node.targets[0].id
    return None


def _index_expression(node: ast.AST, tensors: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return isinstance(node.ctx, ast.Load)
    if isinstance(node, ast.Constant):
        return type(node.value) in (bool, int)
    if isinstance(node, ast.Call):
        return (
            _path(node.func) in {"cutlass.Int32", "cutlass.Int64"}
            and len(node.args) == 1
            and not node.keywords
            and _index_expression(node.args[0], tensors)
        ) or (
            _path(node.func) in {"cute.arch.thread_idx", "cute.arch.block_idx"}
            and not node.args
            and not node.keywords
        )
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id in tensors
            and node.attr == "iterator"
        )
    if isinstance(node, ast.Subscript):
        metadata = (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "stride"
            and isinstance(node.value.value, ast.Attribute)
            and node.value.value.attr == "layout"
            and isinstance(node.value.value.value, ast.Name)
            and node.value.value.value.id in tensors
        )
        coordinate = (
            isinstance(node.value, ast.Call)
            and _path(node.value.func)
            in {"cute.arch.thread_idx", "cute.arch.block_idx"}
            and not node.value.args
            and not node.value.keywords
        )
        # No data-dependent pointer loads or mutable array lookups. Only
        # launch coordinates and immutable tensor-layout metadata move.
        return (
            (metadata or coordinate)
            and isinstance(node.slice, ast.Constant)
            and type(node.slice.value) is int
            and node.slice.value >= 0
        )
    if isinstance(node, ast.BinOp):
        return (
            isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod))
            and _index_expression(node.left, tensors)
            and _index_expression(node.right, tensors)
        )
    if isinstance(node, ast.IfExp):
        return all(
            _index_expression(x, tensors) for x in (node.test, node.body, node.orelse)
        )
    if isinstance(node, ast.Compare):
        return all(
            isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq))
            for op in node.ops
        ) and all(_index_expression(x, tensors) for x in (node.left, *node.comparators))
    if isinstance(node, ast.BoolOp):
        return isinstance(node.op, (ast.And, ast.Or)) and all(
            _index_expression(x, tensors) for x in node.values
        )
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.Not, ast.UAdd, ast.USub)) and _index_expression(
            node.operand, tensors
        )
    return False


def _containers(
    body: list[ast.stmt], constants: dict[str, int]
) -> tuple[set[str], set[str], int] | None:
    """Classify direct private allocations; unknown retained storage declines."""
    module = ast.Module(body=body, type_ignores=[])
    counts = Counter(
        n.id
        for n in ast.walk(module)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    )
    private: set[str] = set()
    pointers: set[str] = set()
    lists: set[str] = set()
    slots = 0
    for statement in ast.walk(module):
        name = _assignment(statement)
        if name is None or counts[name] != 1:
            continue
        value = cast("ast.Assign", statement).value
        if isinstance(value, ast.List) and not value.elts:
            lists.add(name)
        if not isinstance(value, ast.Call):
            continue
        path = _path(value.func)
        if path == "cute.arch.alloc_smem":
            pointers.add(name)
        elif (
            path == "cute.make_tensor"
            and value.args
            and isinstance(value.args[0], ast.Name)
            and value.args[0].id in pointers
        ):
            private.add(name)
        elif path == "cute.make_rmem_tensor":
            if statement not in body or len(value.args) != 2 or value.keywords:
                return None
            size = _static_int(value.args[0], constants)
            dtype = _path(value.args[1])
            if size is None or size <= 0 or dtype not in _CASTS:
                return None
            slots += size * (2 if dtype in {"cutlass.Int64", "cutlass.Float64"} else 1)
            private.add(name)
    # A malformed/rebound allocation must not evade the budget/classification.
    allocs = [
        n
        for n in ast.walk(module)
        if isinstance(n, ast.Call) and _path(n.func) == "cute.make_rmem_tensor"
    ]
    if len(allocs) != sum(
        1
        for s in body
        if _assignment(s) in private
        and isinstance(cast("ast.Assign", s).value, ast.Call)
        and _path(cast("ast.Call", cast("ast.Assign", s).value).func)
        == "cute.make_rmem_tensor"
    ):
        return None
    return private, lists, slots


def _effects_safe(
    region: list[ast.stmt],
    private: set[str],
    lists: set[str],
    tensors: set[str],
    source: frozenset[str],
    disjoint: set[frozenset[str]],
) -> bool:
    for statement in region:
        for node in ast.walk(statement):
            if isinstance(node, ast.stmt) and not isinstance(
                node, (ast.Assign, ast.Expr, ast.For, ast.If, ast.Pass)
            ):
                return False
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
                if not isinstance(node.value, ast.Name):
                    return False
                name = node.value.id
                if name not in private and (
                    name not in tensors
                    or _tensor_roots_may_alias(source, frozenset({name}), disjoint)
                ):
                    return False
            if isinstance(node, ast.Attribute) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                return False
            if not isinstance(node, ast.Call):
                continue
            path = _path(node.func)
            if path in _STORES or (
                isinstance(node.func, ast.Attribute) and node.func.attr == "store"
            ):
                roots = _store_tensor_roots(node, tensors)
                if _tensor_roots_may_alias(source, roots, disjoint):
                    return False
            elif (
                path
                in _PURE
                | _COLLECTIVES
                | _CUTE_CACHE_LOAD_HELPER_NAMES
                | {
                    "cute.arch.load",
                    "cute.make_rmem_tensor",
                    "cute.arch.alloc_smem",
                    "cute.make_tensor",
                }
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "bitcast"
                and isinstance(node.func.value, ast.Call)
                and _path(node.func.value.func) in _CASTS
                and len(node.args) == 1
                and not node.keywords
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in lists
                and len(node.args) == 1
                and not node.keywords
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
                and not node.args
                and not node.keywords
                and _tensor_arg_roots(node.func.value, tensors) is not None
            ):
                pass
            else:
                return False
    return True


def _packet(
    statement: ast.stmt, tensors: set[str]
) -> tuple[str, ast.Call, int, str, frozenset[str]] | None:
    name = _assignment(statement)
    if name is None:
        return None
    call = cast("ast.Assign", statement).value
    if (
        not isinstance(call, ast.Call)
        or _path(call.func) not in {"cute.arch.load", *_CUTE_CACHE_LOAD_HELPER_NAMES}
        or len(call.args) != 2
        or call.keywords
    ):
        return None
    vector = call.args[1]
    if (
        not isinstance(vector, ast.Call)
        or _path(vector.func) != "ir.VectorType.get"
        or len(vector.args) != 2
        or vector.keywords
    ):
        return None
    shape, element = vector.args
    if (
        not isinstance(shape, ast.List)
        or len(shape.elts) != 1
        or not isinstance(shape.elts[0], ast.Constant)
        or type(shape.elts[0].value) is not int
    ):
        return None
    width = shape.elts[0].value
    if not isinstance(element, ast.Attribute) or element.attr != "mlir_type":
        return None
    dtype = _path(element.value)
    if (
        width not in (2, 4, 8)
        or dtype not in {"cutlass.Uint16", "cutlass.Uint32"}
        or width * (2 if dtype == "cutlass.Uint16" else 4) not in (8, 16)
    ):
        return None
    roots = _tensor_arg_roots(call.args[0], tensors)
    if roots is None or len(roots) != 1 or not _index_expression(call.args[0], tensors):
        return None
    return name, call, width, dtype, roots


def _domain(
    loop: ast.For, constants: dict[str, int]
) -> tuple[ast.expr, ast.expr, tuple[str, str] | None] | None:
    bounds = _range_bounds(loop.iter)
    if bounds is None or not isinstance(loop.target, ast.Name) or loop.orelse:
        return None
    values = [_static_int(x, constants) for x in bounds]
    if any(x is None for x in values):
        return None
    start, stop, step = cast("list[int]", values)
    if (
        not (
            0 <= start < stop <= 2**31 - 1
            and 0 < step <= 2**31 - 1
            and start + step <= 2**31 - 1
        )
        or _trip_count_for(*bounds, constants) != 1
    ):
        return None
    children = [s for s in loop.body if isinstance(s, ast.For)]
    lane = None
    if children:
        if (
            len(loop.body) != 1
            or len(children) != 1
            or children[0].orelse
            or not isinstance(children[0].target, ast.Name)
        ):
            return None
        lane_bounds = _range_bounds(children[0].iter)
        if lane_bounds is None:
            return None
        lane_values = [_static_int(x, constants) for x in lane_bounds]
        if (
            lane_values[0] != 0
            or lane_values[2] != 1
            or lane_values[1] is None
            or not 1 <= lane_values[1] <= 32
        ):
            return None
        lane = children[0].target.id, _path(children[0].iter)
    return bounds[0], bounds[2], lane


class _Rename(ast.NodeTransformer):
    def __init__(self, names: dict[str, str]) -> None:
        self.names = names

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.names:
            node.id = self.names[node.id]
        return node


def prefetch_reduction_packets(
    body: list[ast.stmt],
    *,
    constexpr_values: dict[str, int],
    thread_block_dims: tuple[int, int, int],
    tensor_dtypes: dict[str, str],
    reload_modes: dict[int, str],
    argument_names: set[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
) -> list[ast.stmt]:
    """Prefetch at most one consumer packet in an explicit register-prefetch mode."""
    if (
        "register_prefetch" not in reload_modes.values()
        or os.environ.get("HELION_FUSER_MODE") is not None
    ):
        return body
    module = ast.Module(body=body, type_ignores=[])
    occupied = _names(module) | _assigned(module)
    tensors = set(tensor_dtypes)
    reserved = {
        "cute",
        "cutlass",
        "ir",
        "range",
        *_CUTE_CACHE_LOAD_HELPER_NAMES,
        *_STORES,
        *_COLLECTIVES,
    }
    if argument_names & reserved or _assigned(module) & (
        reserved | tensors | set(constexpr_values)
    ):
        return body
    classified = _containers(body, constexpr_values)
    if classified is None:
        return body
    private, lists, used_slots = classified
    helper = _CuteFuseTwoPassLoads(
        constexpr_values,
        thread_block_dims,
        tensor_dtypes,
        reload_modes,
        proven_disjoint_tensor_pairs,
    )
    loops = [(i, s) for i, s in enumerate(body) if isinstance(s, ast.For)]
    for (first_index, first), (last_index, last) in pairwise(loops):
        block_id = helper._reduction_block_id(first)
        if (
            block_id is None
            or _path(first.target) != _path(last.target)
            or _path(first.iter) != _path(last.iter)
            or reload_modes.get(block_id) != "register_prefetch"
        ):
            continue
        domain = _domain(first, constexpr_values)
        other = _domain(last, constexpr_values)
        if domain is None or other is None or domain[2] != other[2]:
            continue
        first_ctx = helper._resolve_load_container(first, domain[0], domain[1], 1)
        last_ctx = helper._resolve_load_container(last, other[0], other[1], 1)
        if first_ctx is None or last_ctx is None or first_ctx[2] != last_ctx[2]:
            continue
        producer, producer_index, slots = first_ctx
        consumer, consumer_index, _ = last_ctx
        anchor = next((s for s in producer if _packet(s, tensors) is not None), None)
        if anchor is None:
            continue
        region = body[first_index : last_index + 1]
        region_ast = ast.Module(body=region, type_ignores=[])
        available = (
            argument_names
            | set(constexpr_values)
            | {"cute", "cutlass", "ir"}
            | set(helper._pure_definitions_before(body, first_index))
        )
        induction = {cast("ast.Name", first.target).id} | (
            {domain[2][0]} if domain[2] else set()
        )
        induction_targets = {id(first.target), id(last.target)}
        if domain[2] is not None:
            induction_targets.update(
                id(s.target)
                for loop in (first, last)
                for s in loop.body
                if isinstance(s, ast.For)
            )
        if any(
            isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Store)
            and n.id in induction
            and id(n) not in induction_targets
            for n in ast.walk(region_ast)
        ):
            continue
        for index, statement in enumerate(consumer):
            packet = _packet(statement, tensors)
            if packet is None:
                continue
            name, call, width, dtype, roots = packet
            if used_slots + slots * width > _CACHE_SLOTS or not _effects_safe(
                region, private, lists, tensors, roots, proven_disjoint_tensor_pairs
            ):
                continue
            order = _LexicalMemoryOrder(body, tensors, proven_disjoint_tensor_pairs)
            if order.has_aliasing_store_between(
                cast("ast.Assign", anchor), cast("ast.Assign", statement), roots
            ):
                continue
            prefix = consumer[:index]
            defs = {_assignment(s): s for s in prefix if _assignment(s) is not None}
            counts = Counter(
                n.id
                for s in prefix
                for n in ast.walk(s)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            )
            needed = _names(call.args[0])
            selected: set[str] = set()
            valid = True
            while needed:
                current = needed.pop()
                if current in selected or current in induction:
                    continue
                if current in defs:
                    definition = cast("ast.Assign", defs[current])
                    if counts[current] != 1 or not _index_expression(
                        definition.value, tensors
                    ):
                        valid = False
                        break
                    selected.add(current)
                    needed.update(_names(definition.value))
                elif current not in available or current in _assigned(region_ast):
                    valid = False
                    break
            if not valid:
                continue
            # Snapshot dependencies must not change after their definition.
            if any(
                _names(cast("ast.Assign", s).value)
                & set().union(*(_assigned(t) for t in prefix[j + 1 :]))
                for j, s in enumerate(prefix)
                if _assignment(s) in selected
            ):
                continue
            allowed = {id(cast("ast.Assign", statement).targets[0])}
            for node in ast.walk(
                ast.Module(body=consumer[index + 1 :], type_ignores=[])
            ):
                if (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.ctx, ast.Load)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == name
                ):
                    allowed.add(id(node.value))
            if any(
                id(n) not in allowed
                for n in ast.walk(module)
                if isinstance(n, ast.Name) and n.id == name
            ):
                continue
            serial = 0
            while f"_consume_prefetch_{serial}" in occupied or any(
                f"_consume_prefetch_{serial}_{n}" in occupied for n in [name, *selected]
            ):
                serial += 1
            cache = f"_consume_prefetch_{serial}"
            renames = {n: f"{cache}_{n}" for n in [name, *selected]}
            moved = [
                cast("ast.stmt", _Rename(renames).visit(cast("ast.stmt", clone_ast(s))))
                for s in prefix
                if _assignment(s) in selected
            ]
            moved.append(
                cast(
                    "ast.stmt",
                    _Rename(renames).visit(cast("ast.stmt", clone_ast(statement))),
                )
            )
            moved.extend(
                statement_from_string(
                    f"{cache}[({producer_index}) * {width} + {i}] = {renames[name]}[{i}]"
                )
                for i in range(width)
            )
            producer[producer.index(anchor) + 1 : producer.index(anchor) + 1] = moved
            consumer.remove(statement)
            _rewrite_vec_extract(last, name, cache, consumer_index, width)
            declaration = statement_from_string(
                f"{cache} = cute.make_rmem_tensor({slots * width}, {dtype})"
            )
            return [declaration, *body]
    return body
