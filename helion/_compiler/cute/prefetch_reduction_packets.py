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
from ..ast_read_writes import ReadWrites
from ._ast_pass_utils import _canonical_name
from ._ast_pass_utils import assigned_name
from ._ast_pass_utils import names_read
from ._ast_pass_utils import names_stored_or_deleted
from ._ast_pass_utils import rename_names
from ._ast_pass_utils import tensor_roots_may_alias
from .cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES
from .fuse_two_pass_loads import _CuteFuseTwoPassLoads
from .fuse_two_pass_loads import _LexicalMemoryOrder
from .fuse_two_pass_loads import _range_bounds
from .fuse_two_pass_loads import _rewrite_vec_extract
from .fuse_two_pass_loads import _static_int
from .fuse_two_pass_loads import _store_tensor_roots
from .fuse_two_pass_loads import _tensor_arg_roots
from .fuse_two_pass_loads import _trip_count_for
from .packet_access import match_packet_access
from .proven_loop_bounds import _Context
from .proven_loop_bounds import _Proof
from .reduce_call import ReduceScope

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
    ReduceScope.SHARED_TWO_STAGE.value,
    ReduceScope.CLUSTER.value,
}
_PREAMBLE_SYNC = {
    "cute.arch.griddepcontrol_wait",
    "cute.arch.griddepcontrol_launch_dependents",
    "cute.arch.mbarrier_init",
    "cute.arch.mbarrier_init_fence",
    "cute.arch.cluster_arrive_relaxed",
    "cute.arch.cluster_wait",
}
_PURE = _CASTS | {
    "range",
    "cutlass.range_constexpr",
    "cute.arch.thread_idx",
    "cute.arch.block_idx",
    "cute.arch.block_dim",
    "cute.arch.grid_dim",
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
        name = assigned_name(statement)
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
        if assigned_name(s) in private
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
    *,
    allow_preamble_sync: bool = False,
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
                    or tensor_roots_may_alias(source, frozenset({name}), disjoint)
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
                if tensor_roots_may_alias(source, roots, disjoint):
                    return False
            elif (
                path
                in _PURE
                | _COLLECTIVES
                | _CUTE_CACHE_LOAD_HELPER_NAMES
                | (_PREAMBLE_SYNC if allow_preamble_sync else set())
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
    name = assigned_name(statement)
    if name is None:
        return None
    call = cast("ast.Assign", statement).value
    access = match_packet_access(call)
    if (
        access is None
        or access.kind not in ("packet", "hinted_packet")
        or access.keywords
        or access.width is None
        or access.carrier is None
    ):
        return None
    assert isinstance(call, ast.Call)
    width, dtype = access.width, access.carrier
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


def _bound_before(statements: list[ast.stmt], stop: int) -> set[str]:
    """The names a scalar assignment binds before ``stop``, other than from a
    load or a store (or a ``_cute_`` load, store or atomic helper), that no
    later statement there writes.  The region a prefetch reads them in
    checks its own stores apart, so a name bound by ``n = n + 1``, by ``c =
    offs[0]`` or by ``c = k * 2`` before ``k = 1`` is as available as any:
    a prefetch needs a bound name, not the definition that still spells its
    value which ``fuse_two_pass_loads`` inlines into a key."""
    bound: set[str] = set()
    for statement in statements[:stop]:
        bound -= set(ReadWrites.from_ast(statement).writes)
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and not any(
                isinstance(node, ast.Call)
                and (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("load", "store")
                    or isinstance(node.func, ast.Name)
                    and node.func.id.startswith(
                        ("_cute_load_", "_cute_store_", "_cute_atomic_")
                    )
                )
                for node in ast.walk(statement.value)
            )
        ):
            bound.add(statement.targets[0].id)
    return bound


def _index_context(
    statements: list[ast.stmt],
    stop: int,
    constants: dict[str, int],
    proof: _Proof,
    merged: set[str],
) -> _Context:
    """Read-only integer value snapshots before the producer sweep.

    The loop-bound prover supplies architectural coordinate ranges and rejects
    signed-index overflow. Analyze assignments in source order so a copied
    scalar keeps its old value after the original name is rebound. Do not run
    its AST rewriter: the global tail masks must remain unchanged. A name in
    ``merged`` gets no snapshot: the final rename merges it with another name,
    whose writes rebind it too (``stop_next = stop + 64`` rebinds ``stop``).
    """
    context = _Context()
    for name, integer in constants.items():
        value = proof.value(ast.Constant(value=integer), context)
        if value is not None:
            context.values[name] = value
    for statement in statements[:stop]:
        target = assigned_name(statement)
        value = (
            proof.value(cast("ast.Assign", statement).value, context)
            if target is not None
            else None
        )
        for name in names_stored_or_deleted(statement):
            context.values.pop(name, None)
        if target is not None and value is not None and target not in merged:
            context.values[target] = value
    return context


def _domain(
    loop: ast.For,
    constants: dict[str, int],
    *,
    proof: _Proof | None = None,
    context: _Context | None = None,
) -> tuple[ast.expr, ast.expr, tuple[str, str] | None] | None:
    bounds = _range_bounds(loop.iter)
    if bounds is None or not isinstance(loop.target, ast.Name) or loop.orelse:
        return None
    values = [_static_int(x, constants) for x in bounds]
    if all(value is not None for value in values):
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
    else:
        if proof is None or context is None:
            return None
        # The same rank-relative span/cache-index proof used by the load fuser,
        # with immutable value snapshots and nonnegative Int32 intervals. A
        # dynamic base is legal only when end-start is one exact constant step;
        # a size hint or a runtime length cannot prove the single cache slot.
        if not all(_index_expression(bound, set()) for bound in bounds):
            return None
        evaluated = [proof.value(bound, context) for bound in bounds]
        if any(value is None for value in evaluated):
            return None
        start_value, stop_value, step_value = evaluated
        assert start_value is not None and stop_value is not None
        assert step_value is not None
        span = stop_value.form.add(start_value.form, -1)
        if (
            span is None
            or span.terms
            or span.constant <= 0
            or step_value.low != step_value.high
            or span.constant != step_value.low
            or start_value.high + step_value.low > 2**31 - 1
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


def prefetch_reduction_packets(
    body: list[ast.stmt],
    *,
    constexpr_values: dict[str, int],
    thread_block_dims: tuple[int, int, int],
    tensor_dtypes: dict[str, str],
    reload_modes: dict[int, str],
    argument_names: set[str],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    rename_groups: dict[str, str] | None = None,
) -> list[ast.stmt]:
    """Prefetch at most one consumer packet in an explicit register-prefetch mode.

    ``rename_groups`` maps each name to the name the final rename gives it:
    the checks compare names through it, so ``start_next = start + 64``
    writes ``start``.
    """
    if (
        "register_prefetch" not in reload_modes.values()
        or os.environ.get("HELION_FUSER_MODE") is not None
    ):
        return body
    module = ast.Module(body=body, type_ignores=[])
    occupied = names_read(module) | names_stored_or_deleted(module)
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
    groups = rename_groups or {}

    def canonical(names: set[str]) -> set[str]:
        return {_canonical_name(name, groups) for name in names}

    # The final rename makes a store of a name it merges into one of these a
    # store of that name: ``stop_next = stop + 64`` rebinds the constexpr
    # ``stop``.
    if argument_names & reserved or canonical(
        names_stored_or_deleted(module)
    ) & canonical(reserved | tensors | set(constexpr_values)):
        return body
    classified = _containers(body, constexpr_values)
    if classified is None:
        return body
    private, lists, used_slots = classified

    # Names the final rename merges with another name of the body. The move
    # chooses and renames the statements it copies, and the packet's reads it
    # rewrites, by spelling: a packet held under such a name or whose address
    # depends on one is not moved, and such a name has no value for the
    # single-trip proof.
    spellings: dict[str, set[str]] = {}
    for node in ast.walk(module):
        if isinstance(node, ast.Name):
            spellings.setdefault(_canonical_name(node.id, groups), set()).add(node.id)
    merged = {name for names in spellings.values() if len(names) > 1 for name in names}
    # The rename may rebind a private buffer or list to a tensor under another
    # name of its group (``cache_next = weight`` makes ``cache`` the tensor),
    # so a merged one is not private.
    private -= merged
    lists -= merged
    helper = _CuteFuseTwoPassLoads(
        constexpr_values,
        thread_block_dims,
        tensor_dtypes,
        reload_modes,
        proven_disjoint_tensor_pairs,
        rename_groups=groups,
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
        proof = _Proof(thread_block_dims)
        context = _index_context(body, first_index, constexpr_values, proof, merged)
        domain = _domain(first, constexpr_values, proof=proof, context=context)
        other = _domain(last, constexpr_values, proof=proof, context=context)
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
        first_bounds = _range_bounds(first.iter)
        assert first_bounds is not None
        bound_names = set().union(*(names_read(bound) for bound in first_bounds))
        if bound_names & names_stored_or_deleted(region_ast):
            # Identical syntax must retain identical values on both sweeps.
            # Otherwise the prefetched packet would belong to another slice.
            continue
        available = (
            argument_names
            | set(constexpr_values)
            | {"cute", "cutlass", "ir"}
            | _bound_before(body, first_index)
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
        canonical_induction = canonical(induction)
        if any(
            isinstance(n, ast.Name)
            and isinstance(n.ctx, ast.Store)
            and _canonical_name(n.id, groups) in canonical_induction
            and id(n) not in induction_targets
            for n in ast.walk(region_ast)
        ):
            continue
        for index, statement in enumerate(consumer):
            packet = _packet(statement, tensors)
            if packet is None:
                continue
            name, call, width, dtype, roots = packet
            if name in merged:
                continue
            if used_slots + slots * width > _CACHE_SLOTS or not _effects_safe(
                region, private, lists, tensors, roots, proven_disjoint_tensor_pairs
            ):
                continue
            if not _effects_safe(
                body,
                private,
                lists,
                tensors,
                roots,
                proven_disjoint_tensor_pairs,
                allow_preamble_sync=True,
            ):
                # A prefix write by another CTA thread may become visible only
                # at the crossed collective. The consumer source must be read-
                # only across the entire kernel, not just between these loops.
                # Unknown calls/stores and shadow aliases remain conservative.
                continue
            order = _LexicalMemoryOrder(body, tensors, proven_disjoint_tensor_pairs)
            if order.has_aliasing_store_between(
                cast("ast.Assign", anchor), cast("ast.Assign", statement), roots
            ):
                continue
            prefix = consumer[:index]
            defs = {assigned_name(s): s for s in prefix if assigned_name(s) is not None}
            counts = Counter(
                n.id
                for s in prefix
                for n in ast.walk(s)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            )
            needed = names_read(call.args[0])
            selected: set[str] = set()
            valid = True
            while needed:
                current = needed.pop()
                if current in merged:
                    valid = False
                    break
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
                    needed.update(names_read(definition.value))
                elif current not in available or current in names_stored_or_deleted(
                    region_ast
                ):
                    valid = False
                    break
            if not valid:
                continue
            # Snapshot dependencies must not change after their definition.
            if any(
                names_read(cast("ast.Assign", s).value)
                & set().union(*(names_stored_or_deleted(t) for t in prefix[j + 1 :]))
                for j, s in enumerate(prefix)
                if assigned_name(s) in selected
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
                rename_names(cast("ast.stmt", clone_ast(s)), renames)
                for s in prefix
                if assigned_name(s) in selected
            ]
            moved.append(
                cast(
                    "ast.stmt",
                    rename_names(cast("ast.stmt", clone_ast(statement)), renames),
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
