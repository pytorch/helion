"""One owner for a pure scalar row epilogue after a whole-CTA collective."""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING
from typing import cast

from ..ast_read_writes import ReadWrites
from ..compile_environment import CompileEnvironment
from ..device_function import TensorArg
from . import cluster_online_pair as pair
from .fuse_two_pass_loads import _is_store_call
from .fuse_two_pass_loads import _store_tensor_roots
from .fuse_two_pass_loads import _tensor_arg_roots

if TYPE_CHECKING:
    from ..device_function import DeviceFunction


_MODULE_NAMES = {
    "cutlass",
    "cute",
    "ir",
    "torch",
    "hl",
    "helion",
    "operator",
    "math",
    "float",
    "int",
    "bool",
    "range",
}
_COLLECTIVES = {
    "_cute_grouped_reduce_cluster",
    "_cute_grouped_reduce_cluster_online_pair",
    "_cute_grouped_reduce_cluster_warp_online_pair",
    "_cute_grouped_reduce_shared_two_stage",
    "_cute_grouped_reduce_shared_tree",
    "_cute_grouped_reduce_warp",
}


def _targets(statement: ast.stmt) -> set[str]:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return set()
    target = statement.targets[0]
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Tuple) and all(
        isinstance(item, ast.Name) for item in target.elts
    ):
        return {cast("ast.Name", item).id for item in target.elts}
    return set()


def _whole_collective(statement: ast.stmt, threads: int) -> ast.Call | None:
    if not _targets(statement):
        return None
    assert isinstance(statement, ast.Assign)
    value = statement.value
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in _COLLECTIVES
        and pair._int_kw(value, "group_span") == threads
        and pair._int_kw(value, "pre") in (None, 1)
        and pair._int_kw(value, "group_count") in (None, 1)
    ):
        return None
    return value


def _uniform(expr: ast.AST, names: set[str], cluster: int) -> bool:
    if set(ReadWrites.from_ast(expr).reads) - names - _MODULE_NAMES:
        return False
    parents = {
        id(child): node
        for node in ast.walk(expr)
        for child in ast.iter_child_nodes(node)
    }
    for node in ast.walk(expr):
        if not isinstance(node, ast.Call):
            continue
        function = ast.unparse(node.func)
        if function in {
            "cute.arch.thread_idx",
            "cute.arch.warp_idx",
            "cute.arch.lane_idx",
            "cute.arch.block_idx_in_cluster",
        }:
            return False
        if cluster > 1 and function == "cute.arch.block_idx":
            parent = parents.get(id(node))
            if not (
                isinstance(parent, ast.Subscript)
                and isinstance(parent.slice, ast.Constant)
                and parent.slice.value in (0, 2)
            ):
                return False
    return True


def _scalar_suffix(
    statements: list[ast.stmt], uniform: set[str], cluster: int
) -> list[ast.Call] | None:
    from ..tile_strategy import _is_proven_relocatable_assignment
    from ..tile_strategy import _validated_idempotent_store

    stores: list[ast.Call] = []
    for statement in statements:
        if pair._is_name_assign(statement) and _is_proven_relocatable_assignment(
            statement, allow_load=True
        ):
            assert isinstance(statement, ast.Assign)
            if not _uniform(statement.value, uniform, cluster):
                return None
            uniform.update(_targets(statement))
        elif isinstance(statement, ast.Expr):
            store = _validated_idempotent_store(statement)
            if store is None or not _uniform(store, uniform, cluster):
                return None
            stores.append(store)
        elif not isinstance(statement, ast.Pass):
            return None
    return stores


def lower_scalar_epilogue(
    body: list[ast.stmt],
    fn: DeviceFunction,
    constexpr_values: dict[str, int] | None,
    thread_block_dims: tuple[int, int, int],
    disjoint_pairs: set[frozenset[str]],
) -> list[ast.stmt]:
    from ..autotuner_heuristics.common import is_canonical_row_reduction
    from ..tile_strategy import _is_proven_relocatable_assignment

    env = CompileEnvironment.current()
    if not is_canonical_row_reduction(env) or fn.config.block_sizes != [1]:
        return body
    threads = math.prod(thread_block_dims)
    collectives = [
        (index, call)
        for index, statement in enumerate(body)
        if (call := _whole_collective(statement, threads)) is not None
    ]
    if not collectives:
        return body
    last, call = collectives[-1]
    if last + 1 == len(body):
        return body
    cluster = fn.cute_state.simt_cluster_n
    uniform = {argument.name for argument in fn.arguments} | set(constexpr_values or {})
    for statement in body[: last + 1]:
        if _whole_collective(statement, threads) is not None:
            uniform.update(_targets(statement))
        elif pair._is_name_assign(statement) and _is_proven_relocatable_assignment(
            statement, allow_load=True
        ):
            assert isinstance(statement, ast.Assign)
            if _uniform(statement.value, uniform, cluster):
                uniform.update(_targets(statement))
            else:
                uniform.difference_update(_targets(statement))
        else:
            uniform.difference_update(
                node.id
                for node in ast.walk(statement)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            )
    suffix = body[last + 1 :]
    stores = _scalar_suffix(suffix, uniform.copy(), cluster)
    if not stores:
        return body
    # A moved read cannot cross a write to its storage. The compiler proves
    # fresh outputs disjoint from inputs and preserves input alias guards.
    prefix = body[: last + 1]
    if any(
        isinstance(node, ast.Call) and _is_store_call(node)
        for statement in prefix
        for node in ast.walk(statement)
    ):
        return body
    tensor_names = {
        argument.name for argument in fn.arguments if isinstance(argument, TensorArg)
    }
    destinations = [_store_tensor_roots(store, tensor_names) for store in stores]
    if any(roots is None for roots in destinations):
        return body
    store_roots = {
        root for roots in destinations if roots is not None for root in roots
    }
    needed = set(ReadWrites.from_ast(ast.Module(body=suffix, type_ignores=[])).reads)
    moved: list[ast.stmt] = []
    moved_ids: set[int] = set()
    # Keeping uniform inputs before the collective overlaps their load latency
    # with the reduction, at the cost of replicated loads and longer lifetimes.
    scan = (
        range(0)
        if fn.config.get("cute_scalar_epilogue_keep_inputs", False)
        else range(last - 1, -1, -1)
    )
    for index in scan:
        statement = body[index]
        if not (
            pair._is_name_assign(statement)
            and _targets(statement) & needed
            and _is_proven_relocatable_assignment(statement, allow_load=True)
        ):
            continue
        assert isinstance(statement, ast.Assign)
        target = pair._assigned_name(statement)
        other_reads = {
            name
            for later in body[index + 1 : last + 1]
            if id(later) not in moved_ids
            for name in ReadWrites.from_ast(later).reads
        }
        if target in other_reads:
            continue
        crossed_writes = {
            name
            for later in body[index + 1 : last + 1]
            if id(later) not in moved_ids
            for name in ReadWrites.from_ast(later).writes
        }
        if (
            {target} | set(ReadWrites.from_ast(statement.value).reads)
        ) & crossed_writes:
            continue
        roots = _tensor_arg_roots(statement.value, tensor_names)
        if roots is None:
            if any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
                for node in ast.walk(statement.value)
            ):
                continue
            roots = frozenset()
        if any(
            read_root == store_root
            or frozenset((read_root, store_root)) not in disjoint_pairs
            for read_root in roots
            for store_root in store_roots
        ):
            continue
        needed.update(ReadWrites.from_ast(statement.value).reads)
        moved.insert(0, statement)
        moved_ids.add(id(statement))
    call_name = cast("ast.Name", call.func).id
    if call_name == "_cute_grouped_reduce_cluster":
        lane = call.args[3]
    elif call_name in {
        "_cute_grouped_reduce_cluster_online_pair",
        "_cute_grouped_reduce_cluster_warp_online_pair",
    }:
        lane = call.args[2]
    else:
        lane = call.args[3]
    condition = f"{ast.unparse(lane)} == 0"
    if cluster > 1:
        condition += " and cute.arch.block_idx_in_cluster() == 0"
    guard = pair._stmt(f"if {condition}:\n    pass")
    assert isinstance(guard, ast.If)
    guard.body = [*moved, *suffix]
    # The whole-CTA, single-row proof above moves every final-result consumer
    # into lane0/rank0. The helper may now keep its communication unconditional
    # while folding its local receive mirror only in that owner's warp.
    if (
        fn.config.get("cute_scalar_epilogue_owner_fold", False)
        and ast.unparse(call.func) == "_cute_grouped_reduce_cluster_warp_online_pair"
    ):
        call.keywords.append(ast.keyword(arg="owner_only", value=ast.Constant(True)))
    return [statement for statement in prefix if id(statement) not in moved_ids] + [
        guard
    ]
