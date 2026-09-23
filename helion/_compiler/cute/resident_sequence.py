"""Ordered reductions over one logical device tile.

The ordinary scalar lowering updates an online carry for every synthetic lane.
This optional lowering instead finishes each reduction over the whole tile,
then evaluates its dependent pointwise work and the original carry updates.
Only compiler-owned device-loop axes can create a region.  The late proof
rejects effects, unmodelled control flow, and lane-dependent live-outs.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import cast

from ... import exc
from .. import tile_strategy
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ..ast_read_writes import ReadWrites
from .resident_reductions import _pointer_parts
from .scalar_recipe import _MATH_CALLS
from .scalar_recipe import _NUMERIC_TYPES
from .scalar_recipe import _OPERATOR_CALLS
from .scalar_recipe import _clone as _clone_ast
from .scalar_recipe import build_recipe

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    import torch

    from ..inductor_lowering import CodegenState
    from ..reduction_strategy import BlockReductionStrategy
    from .resident_sequence_cache import SequenceCache


SEQUENCE_KEY = "cute_reduction_sequence"
_MARKER = "_helion_sequence_reduce"
_VECTOR_LOADS = frozenset(
    {
        "cute.arch.load",
        "_cute_load_l2_evict_last",
        "_cute_load_l1_l2_evict_first",
        "_cute_load_l1_l2_evict_last",
        "_cute_load_l1_l2_evict_first_8b",
        "_cute_load_l1_l2_evict_last_8b",
    }
)
_GLOBALS = frozenset(
    {
        "cutlass",
        "cute",
        "operator",
        "ir",
        "range",
        "float",
        "int",
        "_cute_load_l2_evict_last",
        "_cute_load_l1_l2_evict_first",
        "_cute_load_l1_l2_evict_last",
        "_cute_load_l1_l2_evict_first_8b",
        "_cute_load_l1_l2_evict_last_8b",
        _MARKER,
    }
)
_MAX_ELEMENTS_PER_THREAD = 128
_MAX_FRAGMENT_BYTES = 1024
_MAX_REDUCTIONS = 16
_MAX_STATEMENTS = 512


@dataclasses.dataclass(frozen=True)
class SequenceRegion:
    """Proven axis/launch facts recorded while its DeviceLoopState is active."""

    loop: ast.For
    lane_var: str
    lane_extent: int
    vector_var: str | None
    vector_width: int
    reduction_axis: int
    threads: int
    thread_count: int
    linear_thread: str

    @property
    def elements_per_thread(self) -> int:
        return self.lane_extent * self.vector_width

    @property
    def element_index(self) -> str:
        if self.vector_var is None:
            return self.lane_var
        return f"{self.lane_var} * {self.vector_width} + {self.vector_var}"


def sequence_reduction_expr(
    strategy: BlockReductionStrategy,
    state: CodegenState,
    input_name: str,
    reduction_type: str,
    fake_output: torch.Tensor,
    default: float | bool,
) -> str | None:
    """Emit a distinct marker, never a heuristic match on emitted warp calls."""
    from torch._inductor.codegen.simd import constant_repr
    from torch._prims_common import get_computation_dtype

    from ..compile_environment import CompileEnvironment
    from ..tile_strategy import DeviceLoopState

    env = CompileEnvironment.current()
    fn = state.device_function
    if (
        env.backend.name != "cute"
        or fn.config.config.get(SEQUENCE_KEY, "scalar") == "scalar"
        or strategy.block_index not in env.config_spec.cute_sequence_reduction_blocks
    ):
        return None
    candidates = {
        id(loop): loop
        for loops in state.codegen.active_device_loops.values()
        for loop in loops
        if isinstance(loop, DeviceLoopState)
        and strategy.block_index in loop.lane_loop_blocks
    }
    if len(candidates) != 1:
        raise exc.BackendUnsupported(
            "cute", "sequence reduction needs one device lane axis"
        )
    loop = next(iter(candidates.values()))
    if len(loop.block_ids) != 1 or reduction_type not in {"sum", "max", "min", "prod"}:
        raise exc.BackendUnsupported("cute", "unsupported sequence reduction region")
    lane_vars = [
        node
        for node in ast.walk(loop.for_node)
        if isinstance(node, ast.For)
        and isinstance(getattr(node, HELION_LANE_LOOP_VAR_ATTR, None), str)
    ]
    if len(lane_vars) != 1:
        raise exc.BackendUnsupported(
            "cute", "sequence reduction needs a single lane loop"
        )
    lane_loop = lane_vars[0]
    assert isinstance(lane_loop.target, ast.Name)
    lane_extent = _literal_loop_extent(lane_loop)
    nested = [
        node
        for node in ast.walk(lane_loop)
        if isinstance(node, ast.For) and node is not lane_loop
    ]
    vector_loop = nested[0] if len(nested) == 1 else None
    if len(nested) > 1 or (
        vector_loop is not None
        and (
            not isinstance(vector_loop.target, ast.Name)
            or not isinstance(vector_loop.iter, ast.Call)
            or _qualified(vector_loop.iter.func) != "cutlass.range_constexpr"
        )
    ):
        raise exc.BackendUnsupported(
            "cute", "sequence reduction has an unknown nested loop"
        )
    vector_width = _literal_loop_extent(vector_loop) if vector_loop is not None else 1
    if (
        lane_extent is None
        or vector_width is None
        or lane_extent * vector_width > _MAX_ELEMENTS_PER_THREAD
    ):
        raise exc.BackendUnsupported(
            "cute", "sequence reduction exceeds the resident lane bound"
        )
    axes, sizes = strategy._active_thread_layout()
    axis = axes.get(strategy.block_index)
    threads = sizes.get(axis, 0) if axis is not None else 0
    pre = 1
    total = 1
    for dimension, size in sizes.items():
        total *= size
        if axis is not None and dimension < axis:
            pre *= size
    linear = env.backend.thread_linear_index_expr(sizes)
    launch = fn.tile_strategy.thread_block_dims()
    if (
        axis is None
        or pre != 1
        or threads < 32
        or threads > 1024
        or threads & (threads - 1)
        or total > 1024
        or total % threads
        or not linear
        or strategy._lane_reduce_cluster_n() != 1
        or any(size != sizes.get(dimension, 1) for dimension, size in enumerate(launch))
    ):
        raise exc.BackendUnsupported(
            "cute", "sequence reduction needs consecutive CTA-local groups"
        )
    region = SequenceRegion(
        loop=loop.for_node,
        lane_var=lane_loop.target.id,
        lane_extent=lane_extent,
        vector_var=cast("ast.Name", vector_loop.target).id
        if vector_loop is not None
        else None,
        vector_width=vector_width,
        reduction_axis=axis,
        threads=threads,
        thread_count=total,
        linear_thread=linear,
    )
    fn.cute_state.resident_sequence_regions[id(loop.for_node)] = region
    identity = env.backend.cast_expr(
        constant_repr(default),
        env.backend.dtype_str(get_computation_dtype(fake_output.dtype)),
    )
    return tile_strategy._lane_reduce_marker_expr(
        input_name, reduction_type, identity, min(threads, 32)
    ).replace(tile_strategy._HELION_LANE_REDUCE_MARKER, _MARKER, 1)


class _Decline(Exception):
    pass


def _qualified(node: ast.AST) -> str | None:
    return tile_strategy._qualified_name(node)


def _literal_loop_extent(loop: ast.For) -> int | None:
    iterator = loop.iter
    if (
        isinstance(iterator, ast.Call)
        and _qualified(iterator.func) in {"range", "cutlass.range_constexpr"}
        and len(iterator.args) == 1
        and not iterator.keywords
        and isinstance(iterator.args[0], ast.Constant)
        and type(iterator.args[0].value) is int
        and iterator.args[0].value > 0
    ):
        return iterator.args[0].value
    return None


def _clone(node: ast.AST) -> ast.stmt:
    result = _clone_ast(node)
    for original, copied in zip(ast.walk(node), ast.walk(result), strict=True):
        for name, value in vars(original).items():
            if name not in original._fields:
                setattr(copied, name, value)
    return cast("ast.stmt", result)


def _canonical(node: ast.AST, renames: Mapping[str, str]) -> ast.AST:
    class Names(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.Name:
            node.id = renames.get(node.id, node.id)
            return node

    return Names().visit(_clone(node))


def _live_in(statements: list[ast.AST], live_out: set[str]) -> set[str]:
    """Scalar liveness with zero-trip loops and branch definition dominance."""
    live = set(live_out)
    for statement in reversed(statements):
        name = _assignment(statement)
        if name is not None:
            live.discard(name)
            live.update(_reads(statement))
        elif isinstance(statement, ast.If):
            live = (
                _live_in(list(statement.body), live)
                | _live_in(list(statement.orelse), live)
                | _reads(statement.test)
            )
        elif isinstance(statement, ast.For) and isinstance(statement.target, ast.Name):
            carried = set(live)
            while True:
                needed = _live_in(list(statement.body), carried) - {statement.target.id}
                updated = carried | needed
                if updated == carried:
                    break
                carried = updated
            live |= carried | _reads(statement.iter)
            live = _live_in(list(statement.orelse), live)
        else:
            live.update(_reads(statement))
    return live


def _reads(node: ast.AST) -> set[str]:
    return set(ReadWrites.from_ast(node).reads)


def _assignment(node: ast.AST) -> str | None:
    return tile_strategy._plain_assignment_name(node)


def _marker(node: ast.AST) -> tile_strategy._LaneReduceMarker | None:
    if not isinstance(node, ast.Assign):
        return None
    calls = [
        item
        for item in ast.walk(node.value)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == _MARKER
    ]
    if not calls:
        return None
    if len(calls) != 1:
        raise _Decline
    copied = _clone(node)
    for item in ast.walk(copied):
        if (
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id == _MARKER
        ):
            item.func.id = tile_strategy._HELION_LANE_REDUCE_MARKER
    result = tile_strategy._is_lane_reduce_marker_assign(copied)
    if result is None:
        raise _Decline
    return result


def _pointer_root(expression: ast.expr) -> str:
    if isinstance(expression, ast.IfExp):
        left = _pointer_root(expression.body)
        if left != _pointer_root(expression.orelse):
            raise _Decline
        return left
    parts = _pointer_parts(expression)
    if parts is None:
        raise _Decline
    return parts[0]


def _memory_roots(body: list[ast.AST]) -> tuple[set[str], set[str]]:
    """Fail closed on unmodelled memory effects throughout this device function."""
    reads: set[str] = set()
    writes: set[str] = set()
    # Only the compiler's local empty-list collectors may be mutated by
    # append. An arbitrary receiver could hide a write to an operand.
    list_definitions: dict[str, list[ast.expr]] = {}
    for statement in body:
        for node in ast.walk(statement):
            name = _assignment(node)
            if name is not None and isinstance(node, ast.Assign):
                list_definitions.setdefault(name, []).append(node.value)
    lists = {
        name
        for name, definitions in list_definitions.items()
        if all(isinstance(value, ast.List) and not value.elts for value in definitions)
    }
    packets = {
        name
        for name, definitions in list_definitions.items()
        if all(
            isinstance(value, ast.Call) and _valid_vector_load(value)
            for value in definitions
        )
    }
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Subscript):
                if isinstance(node.ctx, ast.Store):
                    raise _Decline
                if isinstance(node.value, ast.Name) and node.value.id in packets:
                    continue
                if isinstance(node.value, ast.Attribute) and node.value.attr in {
                    "stride",
                    "shape",
                }:
                    continue
                if isinstance(node.value, ast.Call) and _qualified(node.value.func) in {
                    "cute.arch.thread_idx",
                    "cute.arch.block_idx",
                    "cute.arch.block_dim",
                    "cute.arch.grid_dim",
                }:
                    continue
                raise _Decline
            if not isinstance(node, ast.Call):
                continue
            path = _qualified(node.func)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"load", "store"}
                and path not in _VECTOR_LOADS
            ):
                root = _pointer_root(node.func.value)
                (reads if node.func.attr == "load" else writes).add(root)
            elif path in _VECTOR_LOADS:
                if not _valid_vector_load(node):
                    raise _Decline
                reads.add(_pointer_root(node.args[0]))
            elif path in {
                "_cute_store_u16_vec",
                "_cute_store_u32_vec",
                "_cute_store_u64_vec",
            }:
                writes.add(_pointer_root(node.args[0]))
            elif (
                path
                in {
                    _MARKER,
                    "range",
                    "cutlass.range_constexpr",
                    "float",
                    "int",
                    "ir.VectorType.get",
                }
                or path is not None
                and (
                    path.startswith("cutlass.")
                    and path[8:] in _NUMERIC_TYPES
                    or path.startswith("cute.math.")
                    and path[10:] in _MATH_CALLS
                    or path.startswith("operator.")
                    and path[9:] in _OPERATOR_CALLS
                    or path
                    in {
                        "cute.arch.thread_idx",
                        "cute.arch.block_idx",
                        "cute.arch.block_dim",
                        "cute.arch.grid_dim",
                    }
                )
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "bitcast"
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in lists
            ):
                continue
            else:
                raise _Decline
    return reads, writes


def _valid_vector_load(node: ast.Call) -> bool:
    return (
        _qualified(node.func) in _VECTOR_LOADS
        and len(node.args) == 2
        and isinstance(node.args[1], ast.Call)
        and _qualified(node.args[1].func) == "ir.VectorType.get"
        and all(
            keyword.arg
            in {"cop", "level1_eviction_priority", "level2_eviction_priority"}
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
            for keyword in node.keywords
        )
    )


def _vector_loads(statements: list[ast.stmt]) -> set[str]:
    result: set[str] = set()
    for statement in statements:
        name = _assignment(statement)
        if (
            name is not None
            and isinstance(statement, ast.Assign)
            and isinstance(statement.value, ast.Call)
            and _qualified(statement.value.func) in _VECTOR_LOADS
        ):
            result.add(name)
    return result


def _pure_expression(expression: ast.expr, vectors: set[str]) -> bool:
    """Validate vector leaves separately, then use the scalar recipe grammar."""

    class VectorLeaves(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            if isinstance(node.value, ast.Name) and node.value.id in vectors:
                return ast.copy_location(ast.Constant(value=0), node)
            return self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> ast.AST:
            if _qualified(node.func) in _VECTOR_LOADS:
                if not _valid_vector_load(node):
                    raise _Decline
                # The compiler emitted this vector type; only the pointer can
                # read memory or depend on a scalar coordinate.
                _pointer_root(node.args[0])
                if build_recipe(node.args[0], [], _reads(node.args[0])) is None:
                    raise _Decline
                return ast.copy_location(ast.Constant(value=0), node)
            return self.generic_visit(node)

    copied = cast("ast.Expr", _clone(ast.Expr(value=expression))).value
    value = cast("ast.expr", VectorLeaves().visit(copied))
    return build_recipe(value, [], _reads(value)) is not None


def _pure_statement(statement: ast.stmt, vectors: set[str]) -> bool:
    if isinstance(statement, ast.Assign) and _assignment(statement) is not None:
        return _marker(statement) is not None or _pure_expression(
            statement.value, vectors
        )
    return False


def _dtype(expression: ast.expr, tensor_dtypes: Mapping[str, str]) -> str | None:
    """Only stash scalar leaves with an exact emitted dtype, never guess."""
    if isinstance(expression, ast.Call):
        path = _qualified(expression.func)
        if (
            path is not None
            and path.startswith("cutlass.")
            and path[8:] in _NUMERIC_TYPES
            and len(expression.args) == 1
        ):
            return path
        if (
            isinstance(expression.func, ast.Attribute)
            and expression.func.attr == "bitcast"
            and len(expression.args) == 1
        ):
            path = _qualified(expression.args[0])
            return (
                path
                if path is not None
                and path.startswith("cutlass.")
                and path[8:] in _NUMERIC_TYPES
                else None
            )
        if (
            isinstance(expression.func, ast.Attribute)
            and expression.func.attr == "load"
        ):
            return tensor_dtypes.get(_pointer_root(expression.func.value))
    if isinstance(expression, ast.IfExp):
        left, right = (
            _dtype(expression.body, tensor_dtypes),
            _dtype(expression.orelse, tensor_dtypes),
        )
        return left if left is not None and left == right else None
    return None


def _is_memory_leaf(expression: ast.expr, vectors: set[str]) -> bool:
    return any(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            or _qualified(node.func) in _VECTOR_LOADS
        )
        or isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id in vectors
        for node in ast.walk(expression)
    )


def _axis_dependency(expression: ast.AST, axis: int) -> bool:
    for node in ast.walk(expression):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Call)
            and _qualified(node.value.func) == "cute.arch.thread_idx"
        ):
            if not isinstance(node.slice, ast.Constant) or node.slice.value == axis:
                return True
        if isinstance(node, ast.Call) and _qualified(node.func) in {
            "cute.arch.lane_idx",
            "cute.arch.warp_idx",
        }:
            return True
    return False


def _uniform(
    expression: ast.expr, prefix: list[ast.stmt], boundaries: set[str]
) -> bool:
    recipe = build_recipe(expression, prefix, boundaries)
    if recipe is None:
        return False
    nodes = [
        *(prefix[index] for index in recipe.source_statement_indices),
        ast.Expr(value=expression),
    ]
    return not any(
        isinstance(node, ast.Call)
        and _qualified(node.func)
        in {"cute.arch.thread_idx", "cute.arch.lane_idx", "cute.arch.warp_idx"}
        for statement in nodes
        for node in ast.walk(statement)
    )


def _combine(kind: str, left: str, right: str) -> str:
    if kind in {"max", "min"}:
        return f"cute.math.{kind}({left}, {right}, propagate_nan=True)"
    return f"({left}) {'+' if kind == 'sum' else '*'} ({right})"


def _finalize(
    marker: tile_strategy._LaneReduceMarker,
    accumulator: str,
    region: SequenceRegion,
    new_var: Callable[..., str],
) -> list[ast.stmt]:
    if marker.reduction_type == "sum":
        partial = f"cute.arch.warp_reduction_sum({accumulator}, threads_in_group=32)"
    else:
        operation = _combine(marker.reduction_type, "a", "b")
        partial = f"cute.arch.warp_reduction({accumulator}, lambda a, b: {operation}, threads_in_group=32)"
    if region.threads == 32:
        return [
            statement_from_string(
                f"{marker.result_var} = {marker.finalize_expr(partial)}"
            )
        ]
    dtype = tile_strategy._dtype_ctor_from_identity(marker.identity_expr)
    assert dtype is not None
    warps = region.threads // 32
    pointer, buffer, tid, value, index = (
        new_var("sequence_" + name)
        for name in ("smem_ptr", "smem", "tid", "group_value", "warp")
    )
    return [
        statement_from_string(
            f"{pointer} = cute.arch.alloc_smem({dtype}, {region.thread_count // 32})"
        ),
        statement_from_string(
            f"{buffer} = cute.make_tensor({pointer}, ({region.thread_count // 32},))"
        ),
        statement_from_string(f"{tid} = {region.linear_thread}"),
        statement_from_string(f"{accumulator} = {partial}"),
        statement_from_string(
            f"if {tid} % 32 == 0:\n    {buffer}[{tid} // 32] = {accumulator}"
        ),
        statement_from_string("cute.arch.sync_threads()"),
        statement_from_string(f"{value} = {marker.identity_expr}"),
        statement_from_string(
            f"for {index} in cutlass.range_constexpr({warps}):\n    {value} = {_combine(marker.reduction_type, value, f'{buffer}[({tid} // {region.threads}) * {warps} + {index}]')}"
        ),
        # The same static allocation is reused on the next logical tile.
        statement_from_string("cute.arch.sync_threads()"),
        statement_from_string(f"{marker.result_var} = {marker.finalize_expr(value)}"),
    ]


@dataclasses.dataclass
class _Step:
    statement: ast.stmt
    name: str
    outer: bool
    reads: set[str]
    varying: bool
    level: int
    marker: tile_strategy._LaneReduceMarker | None


@dataclasses.dataclass
class _LoweredSequence:
    prefix: list[ast.stmt]
    body: list[ast.stmt]
    caches: list[SequenceCache]
    uniform_carries: frozenset[str]


def _lower_region(
    region: SequenceRegion,
    *,
    prefix: list[ast.stmt],
    boundaries: set[str],
    tensor_dtypes: Mapping[str, str],
    rename_groups: Mapping[str, str],
    resident: bool,
    new_var: Callable[..., str],
    live_out: set[str],
    known_carries: Mapping[int, tuple[SequenceRegion, frozenset[str]]],
) -> _LoweredSequence:
    from .resident_sequence_cache import cache_for_value

    body = region.loop.body
    if len(body) != 1 or not isinstance(body[0], ast.For):
        raise _Decline
    lane_loop = body[0]
    if (
        not isinstance(lane_loop.target, ast.Name)
        or lane_loop.target.id != region.lane_var
        or _literal_loop_extent(lane_loop) != region.lane_extent
        or lane_loop.orelse
    ):
        raise _Decline
    outer: list[ast.stmt] = []
    inner = lane_loop.body
    if region.vector_var is not None:
        if not inner or not isinstance(inner[-1], ast.For):
            raise _Decline
        vector_loop = inner[-1]
        if (
            not isinstance(vector_loop.target, ast.Name)
            or vector_loop.target.id != region.vector_var
            or _literal_loop_extent(vector_loop) != region.vector_width
            or vector_loop.orelse
        ):
            raise _Decline
        outer, inner = inner[:-1], vector_loop.body
    vectors = _vector_loads(outer)
    statements = [*outer, *inner]
    if len(statements) > _MAX_STATEMENTS or not all(
        _pure_statement(statement, vectors) for statement in statements
    ):
        raise _Decline
    assigned = [_assignment(statement) for statement in statements]
    if None in assigned or len(set(assigned)) != len(assigned):
        raise _Decline
    varying = {region.lane_var}
    available = set(boundaries) | set(_GLOBALS)
    assert isinstance(region.loop.target, ast.Name)
    available.add(region.loop.target.id)
    if region.vector_var is not None:
        varying.add(region.vector_var)
    # Coordinate aliases defined outside the lane wrapper remain varying.
    proof_prefix = [
        cast("ast.stmt", _canonical(statement, rename_groups)) for statement in prefix
    ]
    for original, statement in zip(prefix, proof_prefix, strict=True):
        if name := _assignment(statement):
            if _reads(statement) - available:
                available.discard(name)
            else:
                available.add(name)
            if _reads(statement) & varying or _axis_dependency(
                statement, region.reduction_axis
            ):
                varying.add(name)
            else:
                varying.discard(name)
        else:
            written = set(ReadWrites.from_ast(statement).writes)
            if known := known_carries.get(id(original)):
                producer, names = known
                if (
                    producer.reduction_axis == region.reduction_axis
                    and producer.threads == region.threads
                    and producer.thread_count == region.thread_count
                    and producer.linear_thread == region.linear_thread
                ):
                    # The earlier validated region preserves these initialized
                    # carries even when it has zero trips. Their values are
                    # uniform only within the same reduction group; they must
                    # not become boundary facts for CTA-wide control flow.
                    written = written - (names & available - varying)
            available.difference_update(written)
    initialized_uniform = (
        available
        - varying
        - {
            region.loop.target.id,
            region.lane_var,
            region.vector_var,
        }
    )
    steps: list[_Step] = []
    by_name: dict[str, int] = {}
    marker_levels: dict[str, int] = {}
    for ordinal, statement in enumerate(statements):
        name = cast("str", assigned[ordinal])
        marker = _marker(statement)
        reads = _reads(statement)
        external = reads - set(by_name) - {region.lane_var, region.vector_var}
        if any(rename_groups.get(name, name) not in available for name in external):
            raise _Decline
        if name in reads or any(value in assigned[ordinal + 1 :] for value in reads):
            raise _Decline
        level = max(
            (
                steps[by_name[value]].level
                + int(steps[by_name[value]].marker is not None)
                for value in reads
                if value in by_name
            ),
            default=0,
        )
        is_varying = marker is None and bool(
            reads & varying or _axis_dependency(statement, region.reduction_axis)
        )
        if is_varying:
            varying.add(name)
        else:
            varying.discard(name)
        if marker is not None:
            marker_levels[name] = level
        steps.append(
            _Step(
                statement, name, ordinal < len(outer), reads, is_varying, level, marker
            )
        )
        by_name[name] = ordinal
        available.add(name)
    if not 1 <= len(marker_levels) <= _MAX_REDUCTIONS:
        raise _Decline
    # Only row-uniform SSA outputs may become loop-carried state.  This also
    # rejects an unrelated accumulator which the lane split would duplicate.
    all_reads = set().union(*(step.reads for step in steps))
    for step in steps:
        canonical = rename_groups.get(step.name, step.name)
        if step.varying and (
            canonical in live_out or canonical != step.name and canonical in all_reads
        ):
            raise _Decline
    # A marker cannot depend on a value whose recipe crosses a mutable alias
    # back into this same logical tile.
    locals_set = set(by_name)
    for step in steps:
        for name in (
            step.reads - locals_set - _GLOBALS - {region.lane_var, region.vector_var}
        ):
            canonical = rename_groups.get(name, name)
            if canonical in varying:
                raise _Decline

    candidates: dict[str, str] = {}
    if resident:
        for step in steps:
            if (
                step.varying
                and not step.outer
                and isinstance(step.statement, ast.Assign)
                and _is_memory_leaf(step.statement.value, vectors)
            ):
                dtype = _dtype(step.statement.value, tensor_dtypes)
                if dtype is not None:
                    candidates[step.name] = dtype
    caches: dict[str, str] = {}
    cached: set[str] = set()
    result: list[ast.stmt] = []
    setup: list[ast.stmt] = []
    exports: list[SequenceCache] = []
    fragment_bytes = 0
    initialized: str | None = None
    origin: str | None = None
    stages = max(marker_levels.values()) + 1
    carried_outputs = {rename_groups.get(name, name) for name in by_name}
    external_replacements: dict[str, str] = {}
    for name in sorted(
        all_reads - set(by_name) - _GLOBALS - {region.lane_var, region.vector_var}
    ):
        if rename_groups.get(name, name) in carried_outputs:
            snapshot = new_var("sequence_carry_input")
            external_replacements[name] = snapshot
            result.append(statement_from_string(f"{snapshot} = {name}"))

    def replay(statement: ast.stmt) -> ast.stmt:
        class Snapshot(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name) -> ast.Name:
                if isinstance(node.ctx, ast.Load):
                    node.id = external_replacements.get(node.id, node.id)
                return node

        return cast("ast.stmt", Snapshot().visit(_clone(statement)))

    def closure(roots: set[str], level: int) -> set[int]:
        selected: set[int] = set()
        pending = list(roots)
        while pending:
            name = pending.pop()
            index = by_name.get(name)
            if index is None or index in selected:
                continue
            step = steps[index]
            if not step.varying or step.marker is not None:
                continue
            if step.level > level:
                raise _Decline
            selected.add(index)
            if name not in cached:
                pending.extend(step.reads)
        return selected

    needed_later: dict[str, bool] = {}
    for name in candidates:
        visits = 0
        for level in range(stages):
            roots = set().union(
                *(
                    _reads(
                        expr_from_string(
                            cast(
                                "tile_strategy._LaneReduceMarker", step.marker
                            ).input_name
                        )
                    )
                    for step in steps
                    if step.marker is not None and step.level == level
                )
            )
            visits += int(by_name[name] in closure(roots, level))
        needed_later[name] = visits > 1
    for name, dtype in candidates.items():
        if not needed_later[name]:
            continue
        bits = (
            64
            if dtype.endswith(("64",))
            else 16
            if dtype.endswith(("16",))
            else 8
            if dtype.endswith(("8",))
            else 32
        )
        fragment_bytes += region.elements_per_thread * bits // 8
        if fragment_bytes > _MAX_FRAGMENT_BYTES:
            raise _Decline
        buffer = new_var("sequence_values")
        caches[name] = buffer
        setup.append(
            statement_from_string(
                f"{buffer} = cute.make_rmem_tensor(({region.elements_per_thread},), {dtype})"
            )
        )
    if caches:
        initialized, origin = (
            new_var("sequence_initialized"),
            new_var("sequence_origin"),
        )
        iterator = cast("ast.Call", region.loop.iter)
        setup.extend(
            [
                statement_from_string(f"{initialized} = cutlass.Boolean(False)"),
                statement_from_string(f"{origin} = {ast.unparse(iterator.args[0])}"),
            ]
        )
        for name, buffer in caches.items():
            step = steps[by_name[name]]
            assert isinstance(step.statement, ast.Assign)
            cache = cache_for_value(
                step.statement.value,
                [*prefix, *statements[: by_name[name]]],
                region,
                buffer,
                initialized,
                origin,
                rename_groups,
            )
            if cache is not None:
                exports.append(cache)

    def wrap(outer_body: list[ast.stmt], inner_body: list[ast.stmt]) -> ast.stmt:
        if region.vector_var is not None:
            vector = statement_from_string(
                f"for {region.vector_var} in cutlass.range_constexpr({region.vector_width}):\n    pass"
            )
            assert isinstance(vector, ast.For)
            vector.body = inner_body
            contents = [*outer_body, vector]
        else:
            contents = inner_body
        loop = statement_from_string(
            f"for {region.lane_var} in cutlass.range_constexpr({region.lane_extent}):\n    pass"
        )
        assert isinstance(loop, ast.For)
        loop.body = contents
        return loop

    for level in range(stages + 1):
        result.extend(
            replay(step.statement)
            for step in steps
            if step.marker is None and not step.varying and step.level == level
        )
        current = [
            step for step in steps if step.marker is not None and step.level == level
        ]
        if not current:
            continue
        roots = set().union(
            *(
                _reads(
                    expr_from_string(
                        cast("tile_strategy._LaneReduceMarker", step.marker).input_name
                    )
                )
                for step in current
            )
        )
        selected = closure(roots, level)
        accumulators = {step.name: new_var("sequence_acc") for step in current}
        for step in current:
            marker = cast("tile_strategy._LaneReduceMarker", step.marker)
            result.append(
                statement_from_string(
                    f"{accumulators[step.name]} = {marker.identity_expr}"
                )
            )
        outer_body: list[ast.stmt] = []
        inner_body: list[ast.stmt] = []
        for index in sorted(selected):
            step = steps[index]
            destination = outer_body if step.outer else inner_body
            buffer = caches.get(step.name)
            if step.name in cached:
                destination.append(
                    statement_from_string(
                        f"{step.name} = {buffer}[{region.element_index}]"
                    )
                )
            else:
                destination.append(replay(step.statement))
                if buffer is not None:
                    destination.append(
                        statement_from_string(
                            f"{buffer}[{region.element_index}] = {step.name}"
                        )
                    )
                    cached.add(step.name)
        for step in current:
            marker = cast("tile_strategy._LaneReduceMarker", step.marker)
            dtype = tile_strategy._dtype_ctor_from_identity(marker.identity_expr)
            if dtype is None:
                raise _Decline
            accumulator = accumulators[step.name]
            value = f"{dtype}({marker.input_name})"
            inner_body.append(
                replay(
                    statement_from_string(
                        f"{accumulator} = {_combine(marker.reduction_type, accumulator, value)}"
                    )
                )
            )
        result.append(wrap(outer_body, inner_body))
        for step in current:
            result.extend(
                _finalize(
                    cast("tile_strategy._LaneReduceMarker", step.marker),
                    accumulators[step.name],
                    region,
                    new_var,
                )
            )
    if caches:
        assert isinstance(region.loop.target, ast.Name)
        assert initialized is not None and origin is not None
        result.extend(
            [
                statement_from_string(f"{initialized} = cutlass.Boolean(True)"),
                statement_from_string(f"{origin} = {region.loop.target.id}"),
            ]
        )
    uniform_carries = initialized_uniform & {
        rename_groups.get(step.name, step.name) for step in steps if not step.varying
    } - {rename_groups.get(step.name, step.name) for step in steps if step.varying}
    return _LoweredSequence(setup, result, exports, frozenset(uniform_carries))


def materialize_resident_sequences(
    body: list[ast.AST],
    *,
    regions: Mapping[int, SequenceRegion],
    tensor_dtypes: Mapping[str, str],
    rename_groups: Mapping[str, str],
    disjoint_pairs: set[frozenset[str]],
    boundary_names: set[str],
    resident: bool,
    new_var: Callable[..., str],
) -> list[ast.AST]:
    """Apply all admitted regions transactionally, or decline this config."""
    from .resident_sequence_cache import reuse_sequence_cache

    try:
        if not regions:
            raise _Decline
        reads, writes = _memory_roots(body)
        if any(
            frozenset((read, write)) not in disjoint_pairs
            for read in reads
            for write in writes
        ):
            raise _Decline
        encountered: set[int] = set()
        known_carries: dict[int, tuple[SequenceRegion, frozenset[str]]] = {}

        def visit(
            statements: list[ast.AST],
            prefix: list[ast.stmt],
            boundaries: set[str],
            live_out: set[str],
        ) -> list[ast.AST]:
            result: list[ast.AST] = []
            caches: list[SequenceCache] = []
            dominating = list(prefix)
            live_after: dict[int, set[str]] = {}
            live = set(live_out)
            for statement in reversed(statements):
                live_after[id(statement)] = set(live)
                live = _live_in([_canonical(statement, rename_groups)], live)
            for statement in statements:
                region = regions.get(id(statement))
                if region is not None:
                    iterator = region.loop.iter
                    proof_prefix = [
                        cast("ast.stmt", _canonical(item, rename_groups))
                        for item in dominating
                    ]
                    if (
                        not isinstance(iterator, ast.Call)
                        or _qualified(iterator.func) != "range"
                        or not all(
                            _uniform(
                                cast("ast.expr", _canonical(arg, rename_groups)),
                                proof_prefix,
                                boundaries,
                            )
                            for arg in iterator.args
                        )
                    ):
                        raise _Decline
                    body_live = (
                        _live_in(
                            [
                                _canonical(item, rename_groups)
                                for item in region.loop.body
                            ],
                            live_after[id(statement)],
                        )
                        | live_after[id(statement)]
                    )
                    lowered = _lower_region(
                        region,
                        prefix=dominating,
                        boundaries=boundaries,
                        tensor_dtypes=tensor_dtypes,
                        rename_groups=rename_groups,
                        resident=resident,
                        new_var=new_var,
                        live_out=body_live,
                        known_carries=known_carries,
                    )
                    replacement = _clone(statement)
                    assert isinstance(replacement, ast.For)
                    replacement.body = lowered.body
                    result.extend([*lowered.prefix, replacement])
                    caches.extend(lowered.caches)
                    encountered.add(id(statement))
                    known_carries[id(statement)] = region, lowered.uniform_carries
                elif isinstance(statement, ast.For):
                    if (
                        not isinstance(statement.target, ast.Name)
                        or not isinstance(statement.iter, ast.Call)
                        or _qualified(statement.iter.func)
                        not in {"range", "cutlass.range_constexpr"}
                    ):
                        raise _Decline
                    nested_regions = any(
                        id(node) in regions for node in ast.walk(statement)
                    )
                    reused = (
                        reuse_sequence_cache(
                            statement, caches, dominating, rename_groups
                        )
                        if not nested_regions and caches
                        else None
                    )
                    if reused is not None:
                        result.append(reused)
                        dominating.append(statement)
                        continue
                    proof_prefix = [
                        cast("ast.stmt", _canonical(item, rename_groups))
                        for item in dominating
                    ]
                    if nested_regions and not all(
                        _uniform(
                            cast("ast.expr", _canonical(arg, rename_groups)),
                            proof_prefix,
                            boundaries,
                        )
                        for arg in statement.iter.args
                    ):
                        raise _Decline
                    replacement = _clone(statement)
                    assert isinstance(replacement, ast.For)
                    body_live = (
                        _live_in(
                            [
                                _canonical(item, rename_groups)
                                for item in statement.body
                            ],
                            live_after[id(statement)],
                        )
                        | live_after[id(statement)]
                    )
                    replacement.body = cast(
                        "list[ast.stmt]",
                        visit(
                            list(statement.body),
                            dominating,
                            boundaries | {statement.target.id},
                            body_live,
                        ),
                    )
                    result.append(replacement)
                elif isinstance(statement, ast.If):
                    proof_prefix = [
                        cast("ast.stmt", _canonical(item, rename_groups))
                        for item in dominating
                    ]
                    if any(
                        id(node) in regions for node in ast.walk(statement)
                    ) and not _uniform(
                        cast("ast.expr", _canonical(statement.test, rename_groups)),
                        proof_prefix,
                        boundaries,
                    ):
                        raise _Decline
                    replacement = _clone(statement)
                    assert isinstance(replacement, ast.If)
                    replacement.body = cast(
                        "list[ast.stmt]",
                        visit(
                            list(statement.body),
                            dominating,
                            boundaries,
                            live_after[id(statement)],
                        ),
                    )
                    replacement.orelse = cast(
                        "list[ast.stmt]",
                        visit(
                            list(statement.orelse),
                            dominating,
                            boundaries,
                            live_after[id(statement)],
                        ),
                    )
                    result.append(replacement)
                else:
                    result.append(_clone(statement))
                if isinstance(statement, ast.stmt):
                    dominating.append(statement)
            return result

        result = visit(body, [], boundary_names, set())
        if encountered != set(regions):
            raise _Decline
        return result
    except _Decline as error:
        raise exc.BackendUnsupported(
            "cute", "resident dependent reduction sequence proof declined"
        ) from error
