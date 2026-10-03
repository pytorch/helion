"""Resident feature carries for rectangular serial-loop reductions."""

from __future__ import annotations

import ast
import dataclasses
import math
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch._subclasses import FakeTensor

from ... import exc
from .. import tile_strategy
from ..ast_extension import statement_from_string
from ..ast_read_writes import ReadWrites
from ..ast_read_writes import ast_rename
from .affine_vector_io import _i32_offset
from .interchanged_store_dce import _known_statements
from .interchanged_store_dce import _known_subscript
from .interchanged_store_dce import _live_in
from .interchanged_store_dce import _single_store
from .persistent_branch_vec import _binding_write_roots
from .persistent_branch_vec import _freeze_definition
from .persistent_branch_vec import _memory_load_calls
from .persistent_branch_vec import _plain_scalar_load_pointer
from .persistent_branch_vec import _plain_scalar_store_pointer
from .persistent_branch_vec import _pointer_integer_affine_form
from .persistent_branch_vec import _scalar_pointer_calls_are_known
from .scalar_recipe import build_recipe

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    from ..device_function import DeviceFunction


@dataclasses.dataclass(frozen=True)
class ResidentReductionLayout:
    block_id: int
    feature_extent: int
    threads: int
    lane_extent: int
    vector_width: int
    index_name: str


def supports_resident_threads(fn: DeviceFunction, block_id: int) -> bool:
    """Admit whole-CTA sums only for an unbranched serial row-loop tree."""
    from ..compile_environment import CompileEnvironment
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import RootGraphInfo
    from ..reduction_strategy import ReductionStrategy

    env = CompileEnvironment.current()
    if (
        fn.config.config.get("cute_reduction_schedule", "scalar") == "scalar"
        or block_id not in env.config_spec.cute_resident_reduction_blocks
        or fn.config.config.get("cute_cluster_n", 1) != 1
    ):
        return False
    if any(
        not isinstance(info, (RootGraphInfo, ForLoopGraphInfo))
        for info in fn.codegen.codegen_graphs
    ):
        return False
    for strategy in fn.tile_strategy.strategies:
        if isinstance(strategy, ReductionStrategy):
            if strategy._reduction_thread_count() > 1:
                return False
        elif any(size > 1 for size in strategy.thread_block_sizes()):
            return False
    return True


def proven_resident_tensor_strides(fn: DeviceFunction) -> dict[tuple[str, int], int]:
    """Combine guarded input strides with static fresh-allocation strides."""
    from ..compile_environment import CompileEnvironment
    from ..device_function import TensorArg

    env = CompileEnvironment.current()
    result = fn.proven_tensor_stride_values()
    constants = _guarded_metadata_constants(fn)
    input_storages = {id(tensor.untyped_storage()) for tensor in env.input_sources}
    for argument in fn.arguments:
        if not isinstance(argument, TensorArg):
            continue
        tensor = argument.fake_value
        if id(tensor.untyped_storage()) in input_storages:
            continue
        for dimension, stride in enumerate(tensor.stride()):
            value = _resolved_integer(stride, constants)
            if value is not None:
                result[argument.name, dimension] = value
    return result


def _guarded_metadata_constants(fn: DeviceFunction) -> dict[sympy.Expr, sympy.Integer]:
    from ..compile_environment import CompileEnvironment
    from ..device_function import TensorArg

    env = CompileEnvironment.current()
    if "input_tensor_metadata" not in env.compiler_fact_specialization_facts:
        return {}
    values: dict[sympy.Expr, set[int]] = {}
    for argument in fn.arguments:
        if not isinstance(argument, TensorArg):
            continue
        fake = argument.fake_value
        runtime = env.runtime_value_for_tensor(fake)
        if (
            env.tensor_input_source(fake) is None
            or not isinstance(runtime, torch.Tensor)
            or isinstance(runtime, FakeTensor)
        ):
            continue
        for symbolic, actual in zip(
            (*fake.shape, *fake.stride()),
            (*runtime.shape, *runtime.stride()),
            strict=True,
        ):
            if isinstance(symbolic, torch.SymInt):
                expression = env.shape_env.replace(symbolic._sympy_())
                values.setdefault(expression, set()).add(int(actual))
    return {
        expression: sympy.Integer(next(iter(results)))
        for expression, results in values.items()
        if len(results) == 1
    }


def _resolved_integer(
    value: int | torch.SymInt, constants: Mapping[sympy.Expr, sympy.Integer]
) -> int | None:
    from ..compile_environment import CompileEnvironment

    if isinstance(value, int):
        return value
    env = CompileEnvironment.current()
    expression = env.specialize_expr(
        env.shape_env.replace(value._sympy_()).xreplace(constants)
    )
    return int(expression) if isinstance(expression, sympy.Integer) else None


def proven_resident_tensor_alignments(fn: DeviceFunction) -> dict[str, int]:
    from ..compile_environment import CompileEnvironment
    from ..device_function import TensorArg
    from .memory_ops import tensor_has_specialized_base_alignment

    env = CompileEnvironment.current()
    constants = _guarded_metadata_constants(fn)
    input_storages = {id(tensor.untyped_storage()) for tensor in env.input_sources}
    result = {}
    for argument in fn.arguments:
        if not isinstance(argument, TensorArg):
            continue
        tensor = argument.fake_value
        if id(tensor.untyped_storage()) not in input_storages:
            offset = _resolved_integer(tensor.storage_offset(), constants)
            if offset is not None:
                result[argument.name] = math.gcd(16, offset * tensor.element_size())
            continue
        for alignment in (16, 8, 4, 2):
            if tensor_has_specialized_base_alignment(env, tensor, alignment):
                result[argument.name] = alignment
                break
    return result


def feature_index_expression(layout: ResidentReductionLayout, lane: str) -> str:
    """Give each thread V contiguous features in each T*V-wide chunk."""
    tid = "cutlass.Int32(cute.arch.thread_idx()[0])"
    if layout.vector_width == 1:
        return f"{tid} + cutlass.Int32({lane}) * {layout.threads}"
    width = layout.vector_width
    return (
        f"({tid} + cutlass.Int32({lane} // {width}) * {layout.threads})"
        f" * {width} + cutlass.Int32({lane} % {width})"
    )


def _clone(statement: ast.AST) -> ast.stmt:
    return statement_from_string(ast.unparse(statement))


def _reads(node: ast.AST) -> set[str]:
    return set(ReadWrites.from_ast(node).reads)


def _assignment(statement: ast.AST) -> str | None:
    return tile_strategy._plain_assignment_name(statement)


def _dtype(
    expression: ast.expr,
    names: Mapping[str, str],
    tensor_dtypes: Mapping[str, str],
) -> str | None:
    if isinstance(expression, ast.Constant):
        if type(expression.value) is bool:
            return "bool"
        if type(expression.value) is int:
            return "int"
        if type(expression.value) is float:
            return "float"
        return None
    if isinstance(expression, ast.Name):
        return names.get(expression.id)
    if isinstance(expression, (ast.Compare, ast.BoolOp)):
        return "cutlass.Boolean"
    if isinstance(expression, ast.UnaryOp):
        return (
            "cutlass.Boolean"
            if isinstance(expression.op, ast.Not)
            else _dtype(expression.operand, names, tensor_dtypes)
        )
    if isinstance(expression, ast.IfExp):
        left = _dtype(expression.body, names, tensor_dtypes)
        right = _dtype(expression.orelse, names, tensor_dtypes)
        return left if left is not None and left == right else None
    if isinstance(expression, ast.BinOp):
        left = _dtype(expression.left, names, tensor_dtypes)
        right = _dtype(expression.right, names, tensor_dtypes)
        if left is None or right is None:
            return None
        if left == right:
            return left
        if left.startswith("cutlass.") and right in {"int", "float"}:
            return left
        if right.startswith("cutlass.") and left in {"int", "float"}:
            return right
        return None
    if isinstance(expression, ast.Call):
        name = tile_strategy._qualified_name(expression.func)
        if (
            name
            in {
                "cutlass.Boolean",
                "cutlass.Int32",
                "cutlass.Int64",
                "cutlass.Float16",
                "cutlass.BFloat16",
                "cutlass.Float32",
                "cutlass.Float64",
            }
            and len(expression.args) == 1
        ):
            return name
        pointer = _plain_scalar_load_pointer(expression)
        if pointer is not None:
            parts = _pointer_parts(pointer)
            return tensor_dtypes.get(parts[0]) if parts is not None else None
        if name is not None and name.startswith("cute.math."):
            dtypes = {
                _dtype(argument, names, tensor_dtypes) for argument in expression.args
            }
            return (
                "cutlass.Float32"
                if "cutlass.Float32" in dtypes
                and dtypes <= {"cutlass.Float32", "int", "float"}
                else None
            )
    return None


def _pointer_parts(pointer: ast.expr) -> tuple[str, ast.expr] | None:
    if (
        isinstance(pointer, ast.Attribute)
        and pointer.attr == "iterator"
        and isinstance(pointer.value, ast.Name)
    ):
        return pointer.value.id, ast.Constant(value=0)
    if isinstance(pointer, ast.BinOp) and isinstance(pointer.op, (ast.Add, ast.Sub)):
        left = _pointer_parts(pointer.left)
        if left is not None and not any(
            isinstance(node, ast.Attribute) and node.attr == "iterator"
            for node in ast.walk(pointer.right)
        ):
            return left[0], ast.BinOp(left=left[1], op=pointer.op, right=pointer.right)
    return None


def _definitions(
    statements: list[ast.stmt],
    boundaries: set[str],
    initial: Mapping[str, ast.expr] | None = None,
) -> dict[str, ast.expr] | None:
    definitions = dict(initial or {})
    for statement in statements:
        name = _assignment(statement)
        if name is None or name in boundaries:
            continue
        assert isinstance(statement, ast.Assign)
        value = _freeze_definition(statement.value, definitions)
        if value is None:
            return None
        definitions[name] = value
    return definitions


def _slice(statements: list[ast.stmt], needed: set[str]) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    needed = set(needed)
    for statement in reversed(statements):
        name = _assignment(statement)
        if name is not None and name in needed:
            result.append(statement)
            needed.remove(name)
            needed.update(_reads(statement))
    return list(reversed(result))


def _floating_literal(expression: ast.expr) -> bool:
    if isinstance(expression, ast.UnaryOp) and isinstance(
        expression.op, (ast.UAdd, ast.USub)
    ):
        expression = expression.operand
    return isinstance(expression, ast.Constant) and type(expression.value) is float


def _tainted(statements: list[ast.stmt], initial: set[str]) -> set[str]:
    result = set(initial)
    for statement in statements:
        name = _assignment(statement)
        if name is not None:
            if tile_strategy._is_lane_reduce_marker_assign(statement) is not None:
                continue
            if _reads(statement) & result:
                result.add(name)
            else:
                result.discard(name)
    return result


def _known_region(statements: list[ast.stmt]) -> bool:
    if not _known_statements(statements):
        return False
    for statement in statements:
        for node in ast.walk(statement):
            if isinstance(node, ast.Subscript) and not (
                isinstance(node.slice, ast.Constant)
                and type(node.slice.value) is int
                and node.slice.value >= 0
            ):
                return False
        if tile_strategy._is_lane_reduce_marker_assign(statement) is not None:
            continue
        store = _single_store(statement)
        values: list[ast.expr] = []
        if isinstance(statement, ast.Assign):
            values.append(statement.value)
        elif store is not None:
            pointer = _plain_scalar_store_pointer(store)
            assert pointer is not None
            values.extend([pointer, *store.args])
            values.extend(
                node.test for node in ast.walk(statement) if isinstance(node, ast.If)
            )
        if any(build_recipe(value, [], _reads(value)) is None for value in values):
            return False
    return True


def _row_count_bound(
    iterator: ast.expr,
    definitions: dict[str, ast.expr],
    mutable_names: set[str],
    uniform_names: set[str],
) -> tuple[int, bool] | None:
    if not (
        isinstance(iterator, ast.Call)
        and tile_strategy._qualified_name(iterator.func) == "range"
        and not iterator.keywords
        and 1 <= len(iterator.args) <= 3
    ):
        return None
    arguments = [_freeze_definition(arg, definitions) for arg in iterator.args]
    if any(arg is None for arg in arguments):
        return None
    args = cast("list[ast.expr]", arguments)
    if len(args) == 3:
        step = _pointer_integer_affine_form(args[2], "", {}, set(), "")
        if step != (0, {}, 1):
            return None
    start, end = (ast.Constant(value=0), args[0]) if len(args) == 1 else args[:2]
    if any(
        not (
            isinstance(value, ast.Call)
            and tile_strategy._qualified_name(value.func) == "cutlass.Int32"
            and len(value.args) == 1
            and not value.keywords
            or isinstance(value, ast.Constant)
            and type(value.value) is int
            and -(2**31) <= value.value < 2**31
        )
        for value in (start, end)
    ):
        return None
    if (_reads(start) | _reads(end)) & mutable_names:
        return None
    for value in (start, end):
        if (
            not _scalar_pointer_calls_are_known(value)
            or _reads(value) - (uniform_names | {"cutlass", "cute"})
            or any(
                isinstance(node, ast.Call)
                and tile_strategy._qualified_name(node.func)
                in {"cute.arch.thread_idx", "cute.arch.warp_idx"}
                or isinstance(node, ast.Subscript)
                and not _known_subscript(node)
                for node in ast.walk(value)
            )
        ):
            return None
    candidates = [(end, True)]
    unwrapped = end
    while (
        isinstance(unwrapped, ast.Call)
        and tile_strategy._qualified_name(unwrapped.func) == "cutlass.Int32"
        and len(unwrapped.args) == 1
        and (
            _i32_offset(unwrapped.args[0])
            or isinstance(unwrapped.args[0], ast.IfExp)
            and _i32_offset(unwrapped.args[0].body)
            and _i32_offset(unwrapped.args[0].orelse)
        )
    ):
        unwrapped = unwrapped.args[0]
    if (
        isinstance(unwrapped, ast.IfExp)
        and isinstance(unwrapped.test, ast.Compare)
        and len(unwrapped.test.ops) == len(unwrapped.test.comparators) == 1
        and isinstance(unwrapped.test.ops[0], (ast.Lt, ast.LtE))
        and ast.dump(unwrapped.test.left) == ast.dump(unwrapped.body)
        and ast.dump(unwrapped.test.comparators[0]) == ast.dump(unwrapped.orelse)
        and _i32_offset(unwrapped.body)
        and _i32_offset(unwrapped.orelse)
    ):
        # min(start + coarse_extent, runtime_end) has a bounded trip count,
        # but its last group may be partial. Do not narrow Int64 operands.
        candidates.extend([(unwrapped.body, False), (unwrapped.orelse, False)])
    for candidate, exact in candidates:
        difference = _pointer_integer_affine_form(
            ast.BinOp(left=candidate, op=ast.Sub(), right=start), "", {}, set(), ""
        )
        # Equal symbolic bases and a nonnegative constant difference bound
        # the trip count even when final Int32 bounds wrap to an empty range.
        if (
            difference is not None
            and difference[:2] == (0, {})
            and 0 <= difference[2] < 2**31
        ):
            return difference[2], exact
    return None


@dataclasses.dataclass
class _VectorAccess:
    tensor: str
    offset: ast.expr
    dtype: str
    write: bool
    feature_stride: int


def _vector_accesses(
    statements: list[ast.stmt],
    definitions: dict[str, ast.expr],
    layout: ResidentReductionLayout,
    row_name: str,
    tensor_dtypes: Mapping[str, str],
    tensor_strides: Mapping[tuple[str, int], int],
    alignments: Mapping[str, int],
    uniform_names: set[str],
    *,
    include_broadcast: bool = False,
) -> dict[str, _VectorAccess]:
    result = {}
    element_bytes = {"cutlass.Float32": 4, "cutlass.Float16": 2, "cutlass.BFloat16": 2}
    for statement in statements:
        write = False
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
            call = statement.value
            pointer = _plain_scalar_load_pointer(call)
        elif isinstance(statement, ast.Expr):
            call = _single_store(statement)
            pointer = _plain_scalar_store_pointer(call) if call is not None else None
            write = True
        else:
            continue
        if pointer is None or call is None:
            continue
        expanded = _freeze_definition(pointer, definitions)
        parts = _pointer_parts(expanded) if expanded is not None else None
        if parts is None:
            continue
        tensor, offset = parts
        dtype = tensor_dtypes.get(tensor)
        if dtype not in element_bytes:
            continue
        assert dtype is not None
        if not _i32_offset(
            offset, frozenset({layout.index_name, row_name})
        ) or not _scalar_pointer_calls_are_known(offset):
            continue
        if _reads(offset) - (
            uniform_names
            | set(tensor_dtypes)
            | {"cutlass", "cute", layout.index_name, row_name}
        ):
            continue
        if any(
            isinstance(node, ast.Call)
            and tile_strategy._qualified_name(node.func)
            in {"cute.arch.thread_idx", "cute.arch.lane_idx", "cute.arch.warp_idx"}
            for node in ast.walk(offset)
        ):
            continue
        form = _pointer_integer_affine_form(
            offset, layout.index_name, tensor_strides, set(), ""
        )
        if form is None or form[0] not in ((0, 1) if include_broadcast else (1,)):
            continue
        if form[0] == 0 and layout.index_name in _reads(offset):
            continue
        width = layout.vector_width if form[0] else 1
        if alignments.get(tensor, 1) >= min(16, width * element_bytes[dtype]):
            result[ast.dump(call)] = _VectorAccess(
                tensor, offset, dtype, write, form[0]
            )
    return result


@dataclasses.dataclass
class _Plan:
    layout: ResidentReductionLayout
    lane_name: str
    row_loop: ast.For
    prefix: list[ast.stmt]
    suffix: list[ast.stmt]
    row_setup: list[ast.stmt]
    producer: list[ast.stmt]
    consumer: list[ast.stmt]
    invariant_producer: list[ast.stmt]
    invariant_consumer: list[ast.stmt]
    state_names: list[str]
    carry_names: list[str]
    stash_names: list[str]
    row_stash_names: list[str]
    replay_prefix: list[ast.stmt]
    markers: list[tile_strategy._LaneReduceMarker]
    vector_accesses: dict[str, _VectorAccess]
    group_rows: int
    replay_row: list[ast.stmt]
    staged_loads: dict[str, _VectorAccess]
    partial_rows: bool
    pipeline_depth: int
    terminal_product: str | None


def _plan(
    loop: ast.For,
    layout: ResidentReductionLayout,
    tensor_dtypes: Mapping[str, str],
    tensor_strides: Mapping[tuple[str, int], int],
    disjoint_pairs: set[frozenset[str]],
    enclosing_definitions: Mapping[str, ast.expr],
    uniform_names: set[str],
    alignments: Mapping[str, int],
    group_rows: int,
    pipelined: bool,
    pipeline_depth: int,
    shared_memory_budget: int,
    row_schedule: str,
    pack_output: bool,
) -> _Plan | None:
    if (
        not isinstance(loop.target, ast.Name)
        or loop.orelse
        or tile_strategy._static_lane_loop_extent(loop) != layout.lane_extent
        or layout.feature_extent != layout.threads * layout.lane_extent
        or layout.feature_extent <= 0
        or layout.feature_extent & (layout.feature_extent - 1)
        or layout.vector_width not in (1, 2, 4, 8)
        or layout.lane_extent % layout.vector_width
        or group_rows not in (1, 2, 3, 4)
        or type(pipeline_depth) is not int
        or pipeline_depth not in (2, 4)
        or (pipeline_depth != 2 and not pipelined)
        or row_schedule not in ("batched", "serial", "serial_deferred")
        or (
            row_schedule == "serial_deferred"
            and (not pipelined or pipeline_depth != 2 or layout.threads <= 32)
        )
    ):
        return None
    lane_name = loop.target.id
    serial = [
        (index, statement)
        for index, statement in enumerate(loop.body)
        if isinstance(statement, ast.For)
    ]
    if len(serial) != 1:
        return None
    row_index, row_loop = serial[0]
    if row_loop.orelse or not isinstance(row_loop.target, ast.Name):
        return None
    row_name = row_loop.target.id
    if any(
        isinstance(node, (ast.For, ast.While))
        for stmt in row_loop.body
        for node in ast.walk(stmt)
    ):
        return None
    prefix, suffix = loop.body[:row_index], loop.body[row_index + 1 :]
    row_body = row_loop.body
    if not all(_known_region(part) for part in (prefix, row_body, suffix)):
        return None
    if any(_single_store(statement) is not None for statement in prefix):
        return None
    prefix_names = {name for stmt in prefix if (name := _assignment(stmt)) is not None}
    row_names = {name for stmt in row_body if (name := _assignment(stmt)) is not None}
    suffix_names = {name for stmt in suffix if (name := _assignment(stmt)) is not None}
    # Reaching-definition analysis below is deliberately straight-line.
    if any(
        len(names) != sum(_assignment(stmt) is not None for stmt in statements)
        for names, statements in (
            (prefix_names, prefix),
            (row_names, row_body),
            (suffix_names, suffix),
        )
    ):
        return None
    if suffix_names & (prefix_names | row_names):
        return None
    mutable_names = _binding_write_roots(loop)
    if mutable_names & set(tensor_dtypes):
        return None
    if (
        layout.index_name in row_names
        or lane_name in row_names
        or row_name in prefix_names
    ):
        return None
    carries = prefix_names & row_names
    if not carries:
        return None
    if _live_in(row_body) & (row_names - carries):
        return None
    if _live_in(suffix) & ((row_names - carries) | {row_name}):
        return None
    index_definition = next(
        (stmt for stmt in prefix if _assignment(stmt) == layout.index_name), None
    )
    expected_index = ast.parse(
        feature_index_expression(layout, lane_name), mode="eval"
    ).body
    if not isinstance(index_definition, ast.Assign) or ast.dump(
        index_definition.value
    ) != ast.dump(expected_index):
        return None
    markers = [
        marker
        for statement in row_body
        if (marker := tile_strategy._is_lane_reduce_marker_assign(statement))
        is not None
    ]
    if not markers or any(
        marker.reduction_type != "sum"
        or marker.identity_expr != "cutlass.Float32(0)"
        or marker.wrap_template != "cutlass.Float32(__HELION_FINALIZED__)"
        or marker.threads_in_group != layout.threads
        or marker.group_pre != 1
        or marker.group_span not in (0, layout.threads)
        or marker.group_count != 1
        or marker.group_cluster_n != 1
        for marker in markers
    ):
        return None
    marker_names = {marker.result_var for marker in markers}
    dependent = _tainted(row_body, marker_names)
    # Never reorder an online reduction or a reduction feeding the row carry.
    if dependent & carries or any(marker.input_name in dependent for marker in markers):
        return None
    if any(
        tile_strategy._is_lane_reduce_marker_assign(statement) is not None
        for statement in [*prefix, *suffix]
    ):
        return None
    # The row-range proof below requires final Int32 bounds. Its induction
    # variable therefore remains Int32 when rematerializing row coordinates.
    types: dict[str, str] = {row_name: "cutlass.Int32"}
    prefix_types: dict[str, str] = {}
    for statement in [*prefix, *row_body]:
        name = _assignment(statement)
        if name is None:
            continue
        assert isinstance(statement, ast.Assign)
        dtype = _dtype(statement.value, types, tensor_dtypes)
        if dtype is not None:
            types[name] = dtype
        else:
            types.pop(name, None)
        if statement in prefix and dtype is not None:
            prefix_types[name] = dtype
    live_prefix = prefix_names & (_live_in(row_body) | _live_in(suffix))
    state = {
        name for name in live_prefix if prefix_types.get(name) == "cutlass.Float32"
    }
    if not carries <= state or any(
        types.get(name) != "cutlass.Float32" for name in carries
    ):
        return None
    if any(types.get(marker.input_name) != "cutlass.Float32" for marker in markers):
        return None
    # Integer index setup is rematerialized. Other persistent values need an
    # explicit supported fragment dtype rather than an implicit narrowing.
    if any(
        prefix_types.get(name)
        not in {"int", "bool", "cutlass.Int32", "cutlass.Int64", "cutlass.Boolean"}
        for name in live_prefix - state
    ):
        return None
    replay = _slice(
        [statement for statement in prefix if _assignment(statement) not in state],
        _live_in(row_body) | _live_in(suffix),
    )
    if any(
        _memory_load_calls(statement) or _reads(statement) & state
        for statement in replay
    ):
        return None
    row_setup = _slice(prefix, _reads(row_loop.iter))
    if any(
        _reads(statement) & ({lane_name, layout.index_name} | state)
        or _memory_load_calls(statement)
        for statement in row_setup
    ):
        return None
    all_statements = [*prefix, *row_body, *suffix]
    writes = [
        store
        for statement in all_statements
        if (store := _single_store(statement)) is not None
    ]
    row_stores = [
        store
        for statement in row_body
        if (store := _single_store(statement)) is not None
    ]
    if len(row_stores) != 1 or not suffix:
        return None
    definitions = _definitions(
        [*prefix, *row_body, *suffix],
        {layout.index_name, row_name, *state, *marker_names},
        enclosing_definitions,
    )
    if definitions is None:
        return None
    row_domain = _row_count_bound(
        row_loop.iter, definitions, mutable_names, uniform_names | set(tensor_dtypes)
    )
    if row_domain is None:
        return None
    row_count, exact_row_count = row_domain
    read_roots: set[str] = set()
    write_roots: list[str] = []
    for statement in all_statements:
        for load in _memory_load_calls(statement):
            pointer = _plain_scalar_load_pointer(load)
            if pointer is None:
                return None
            expanded = _freeze_definition(pointer, definitions)
            parts = _pointer_parts(expanded) if expanded is not None else None
            if parts is None or parts[0] not in tensor_dtypes:
                return None
            read_roots.add(parts[0])
    for store in writes:
        pointer = _plain_scalar_store_pointer(store)
        assert pointer is not None
        expanded = _freeze_definition(pointer, definitions)
        parts = _pointer_parts(expanded) if expanded is not None else None
        if parts is None or not _i32_offset(
            parts[1], frozenset({layout.index_name, row_name})
        ):
            return None
        if _reads(parts[1]) & (state | marker_names | {lane_name}):
            return None
        if _reads(parts[1]) - (
            uniform_names
            | set(tensor_dtypes)
            | {"cutlass", "cute", layout.index_name, row_name}
        ):
            return None
        if any(
            isinstance(node, ast.Call)
            and tile_strategy._qualified_name(node.func)
            in {"cute.arch.thread_idx", "cute.arch.lane_idx", "cute.arch.warp_idx"}
            for node in ast.walk(parts[1])
        ):
            return None
        feature_form = _pointer_integer_affine_form(
            parts[1], layout.index_name, tensor_strides, set(), ""
        )
        if feature_form is None or feature_form[0] != 1:
            return None
        if store is row_stores[0]:
            row_form = _pointer_integer_affine_form(
                parts[1], row_name, tensor_strides, set(), ""
            )
            if (
                row_form is None
                or row_form[0] < layout.feature_extent
                or row_form[0] % layout.feature_extent
            ):
                return None
            # Offsets are Int32. Distinct rows remain distinct modulo 2**32
            # only up to the exact period of the row stride.
            period = (1 << 32) // math.gcd(row_form[0], 1 << 32)
            if row_count > period:
                return None
        elif row_name in _reads(parts[1]):
            return None
        write_roots.append(parts[0])
    if len(write_roots) != len(set(write_roots)) or any(
        left == right or frozenset((left, right)) not in disjoint_pairs
        for index, left in enumerate(write_roots)
        for right in [*read_roots, *write_roots[index + 1 :]]
    ):
        return None
    producer: list[ast.stmt] = []
    consumer: list[ast.stmt] = []
    taint = set(marker_names)
    for statement in row_body:
        name = _assignment(statement)
        if name in marker_names:
            continue
        if _single_store(statement) is not None or _reads(statement) & taint:
            consumer.append(statement)
            if name is not None:
                taint.add(name)
        else:
            producer.append(statement)
    # An epilogue reading a carry directly could observe its old or updated
    # value depending on the source position. SSA aliases retain that version
    # explicitly; direct ambiguous carrier reads use the existing lowering.
    if _live_in(consumer) & carries:
        return None
    varying = _tainted(
        [*replay, *producer, *consumer], {lane_name, layout.index_name, *state}
    )
    producer_names = {
        name for stmt in producer if (name := _assignment(stmt)) is not None
    }
    live = _live_in(consumer) & producer_names
    stash = live & varying
    row_stash = {
        name for name in live - varying if types.get(name) == "cutlass.Float32"
    }
    if any(types.get(name) != "cutlass.Float32" for name in stash):
        return None
    invariant_producer = [
        statement for statement in producer if _assignment(statement) not in varying
    ]
    invariant_consumer = [
        statement
        for statement in consumer
        if _assignment(statement) is not None and _assignment(statement) not in varying
    ]
    producer = [
        statement for statement in producer if statement not in invariant_producer
    ]
    consumer = [
        statement for statement in consumer if statement not in invariant_consumer
    ]
    replay_row = _slice(
        [stmt for stmt in invariant_producer if _assignment(stmt) not in row_stash],
        _live_in([*invariant_consumer, *consumer]),
    )
    if group_rows > 1 and any(
        _memory_load_calls(stmt)
        or (
            types.get(_assignment(stmt) or "")
            not in {"int", "bool", "cutlass.Int32", "cutlass.Int64", "cutlass.Boolean"}
            # CUDA decomposition can replace division by a constant with a
            # reciprocal literal. Re-emitting that same literal is exact and
            # preserves the original consumer arithmetic, including its type.
            and not (isinstance(stmt, ast.Assign) and _floating_literal(stmt.value))
        )
        for stmt in replay_row
    ):
        return None
    terminal_product = (
        _terminal_output_product(consumer, types, tensor_dtypes)
        if pack_output
        else None
    )
    if pack_output and (
        layout.vector_width not in (2, 4, 8) or terminal_product is None
    ):
        return None
    staged_loads = {}
    if pipelined:
        candidates = _vector_accesses(
            row_body,
            definitions,
            layout,
            row_name,
            tensor_dtypes,
            tensor_strides,
            alignments,
            uniform_names,
            include_broadcast=True,
        )
        for key, access in candidates.items():
            row_form = _pointer_integer_affine_form(
                access.offset, row_name, tensor_strides, set(), ""
            )
            if (
                not access.write
                and row_form is not None
                and row_form[0] != 0
                and alignments.get(access.tensor, 1)
                >= (16 if access.feature_stride else 4)
            ):
                staged_loads[key] = access
        if not any(access.feature_stride for access in staged_loads.values()):
            return None
        shared_bytes = sum(
            pipeline_depth
            * group_rows
            * (layout.feature_extent if access.feature_stride else 1)
            * (4 if access.dtype == "cutlass.Float32" else 2)
            for access in staged_loads.values()
        )
        # Each allocation is aligned to 16 bytes; fused reduction scratch
        # needs one FP32 scalar per participating warp and per row/sum.
        shared_bytes += 16 * len(staged_loads) + (
            group_rows * len(markers) * max(1, layout.threads // 32) * 4
        )
        if shared_bytes > shared_memory_budget:
            return None
    return _Plan(
        layout,
        lane_name,
        row_loop,
        prefix,
        suffix,
        row_setup,
        producer,
        consumer,
        invariant_producer,
        invariant_consumer,
        sorted(state),
        sorted(carries),
        sorted(stash),
        sorted(row_stash),
        replay,
        markers,
        _vector_accesses(
            all_statements,
            definitions,
            layout,
            row_name,
            tensor_dtypes,
            tensor_strides,
            alignments,
            uniform_names,
        ),
        group_rows,
        replay_row,
        staged_loads,
        not exact_row_count or row_count % group_rows != 0,
        pipeline_depth,
        terminal_product,
    )


def _loop(lane: str, extent: int, body: list[ast.stmt]) -> ast.For:
    statement = statement_from_string(
        f"for {lane} in cutlass.range_constexpr({extent}):\n    pass"
    )
    assert isinstance(statement, ast.For)
    statement.body = body
    return statement


def _terminal_output_product(
    consumer: list[ast.stmt],
    types: Mapping[str, str],
    tensor_dtypes: Mapping[str, str],
) -> str | None:
    """Prove a terminal FP32 product whose only remaining users are casts/store.

    Packing must not reassociate an earlier multiply/add, reduce across lanes,
    or change a carried value. The resident proof already establishes disjoint
    output ownership; here the straight-line suffix has no other effects.
    """
    if not consumer or not isinstance(consumer[-1], ast.Expr):
        return None
    store = _single_store(consumer[-1])
    if store is None or len(store.args) != 1:
        return None
    value = store.args[0]
    allowed_casts = {"cutlass.Float16", "cutlass.BFloat16", "cutlass.Float32"}

    def strip_casts(node: ast.expr) -> ast.expr:
        while (
            isinstance(node, ast.Call)
            and tile_strategy._qualified_name(node.func) in allowed_casts
            and len(node.args) == 1
            and not node.keywords
        ):
            node = node.args[0]
        return node

    value = strip_casts(value)
    for statement in reversed(consumer[:-1]):
        name = _assignment(statement)
        if name is None or not isinstance(value, ast.Name) or value.id != name:
            return None
        assert isinstance(statement, ast.Assign)
        expression = statement.value
        if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Mult):
            if types.get(name) != "cutlass.Float32" or any(
                _dtype(operand, types, tensor_dtypes) != "cutlass.Float32"
                for operand in (expression.left, expression.right)
            ):
                return None
            return name
        stripped = strip_casts(expression)
        if stripped is expression:
            return None
        value = stripped
    return None


def _packed_terminal_body(
    body: list[ast.stmt],
    component: str,
    width: int,
    product_name: str | None,
    new_var: Callable[[str], str],
    index_name: str,
) -> list[ast.stmt]:
    """Pair only the proven terminal product; keep each preceding scalar DAG."""
    assert product_name is not None and width in (2, 4, 8)
    # Vector I/O has already replaced feature addresses with the chunk base.
    # Do not keep a renamed, otherwise dead scalar coordinate in each lane.
    if not any(
        index_name in _reads(statement)
        for statement in body
        if _assignment(statement) != index_name
    ):
        body = [statement for statement in body if _assignment(statement) != index_name]
    matches = [
        index
        for index, statement in enumerate(body)
        if _assignment(statement) == product_name
    ]
    assert len(matches) == 1
    split = matches[0]
    product = body[split]
    assert isinstance(product, ast.Assign) and isinstance(product.value, ast.BinOp)
    assert isinstance(product.value.op, ast.Mult)
    names = {name for statement in body if (name := _assignment(statement)) is not None}
    result: list[ast.stmt] = []
    for first in range(0, width, 2):
        mappings = [
            {name: new_var(f"packed_output_{lane}_{name}") for name in sorted(names)}
            for lane in (first, first + 1)
        ]

        def lane_copy(
            statement: ast.AST,
            lane: int,
            first: int = first,
            mappings: list[dict[str, str]] = mappings,
        ) -> ast.stmt:
            copied = _clone(statement)
            ast_rename(copied, mappings[lane])

            class Component(ast.NodeTransformer):
                def visit_Name(self, node: ast.Name) -> ast.expr:
                    return ast.Constant(first + lane) if node.id == component else node

            return cast("ast.stmt", Component().visit(copied))

        for lane in (0, 1):
            result.extend(lane_copy(statement, lane) for statement in body[:split])
        products = [lane_copy(product, lane) for lane in (0, 1)]
        assert all(
            isinstance(statement, ast.Assign) and isinstance(statement.value, ast.BinOp)
            for statement in products
        )
        expressions = [
            cast("ast.BinOp", cast("ast.Assign", statement).value)
            for statement in products
        ]
        result.append(
            statement_from_string(
                f"{mappings[0][product_name]}, {mappings[1][product_name]} = cute.arch.mul_packed_f32x2("
                f"({ast.unparse(expressions[0].left)}, {ast.unparse(expressions[1].left)}), "
                f"({ast.unparse(expressions[0].right)}, {ast.unparse(expressions[1].right)}), rnd='rn', ftz=False)"
            )
        )
        for lane in (0, 1):
            result.extend(lane_copy(statement, lane) for statement in body[split + 1 :])
    return result


def _feature_loop(
    plan: _Plan,
    body: list[ast.stmt],
    new_var: Callable[[str], str],
    *,
    pack_output: bool = False,
) -> ast.For:
    width = plan.layout.vector_width
    if width == 1:
        return _loop(plan.lane_name, plan.layout.lane_extent, body)
    chunk, component, base = (
        new_var(name) for name in ("resident_chunk", "resident_lane", "resident_base")
    )
    before: list[ast.stmt] = []
    after: list[ast.stmt] = []
    rewritten: list[ast.stmt] = []
    for statement in body:
        call = (
            statement.value
            if isinstance(statement, (ast.Assign, ast.Expr))
            and isinstance(statement.value, ast.Call)
            else None
        )
        access = plan.vector_accesses.get(ast.dump(call)) if call else None
        if access is None:
            rewritten.append(statement)
            continue
        fragment = new_var("resident_copy")
        expression = _freeze_definition(
            access.offset,
            {plan.layout.index_name: ast.Name(id=base, ctx=ast.Load())},
        )
        assert expression is not None
        offset = ast.unparse(expression)
        if access.write:
            assert isinstance(statement, ast.Expr) and isinstance(call, ast.Call)
            before.append(
                statement_from_string(
                    f"{fragment} = cute.make_rmem_tensor({width}, {access.dtype})"
                )
            )
            rewritten.append(
                statement_from_string(
                    f"{fragment}[{component}] = {access.dtype}({ast.unparse(call.args[0])})"
                )
            )
            after.append(
                statement_from_string(
                    f"_cute_resident_store_vector({access.tensor}, {offset}, {fragment}, {width})"
                )
            )
        else:
            assert isinstance(statement, ast.Assign)
            before.append(
                statement_from_string(
                    f"{fragment} = _cute_resident_load_vector({access.tensor}, {offset}, {width})"
                )
            )
            assignment = _clone(statement)
            assert isinstance(assignment, ast.Assign)
            assignment.value = ast.parse(f"{fragment}[{component}]", mode="eval").body
            rewritten.append(assignment)
    rewritten.insert(
        0, statement_from_string(f"{plan.lane_name} = {chunk} * {width} + {component}")
    )
    base_expression = feature_index_expression(plan.layout, f"({chunk} * {width})")
    return _loop(
        chunk,
        plan.layout.lane_extent // width,
        [
            statement_from_string(f"{base} = {base_expression}"),
            *before,
            *(
                _packed_terminal_body(
                    rewritten,
                    component,
                    width,
                    plan.terminal_product,
                    new_var,
                    plan.layout.index_name,
                )
                if pack_output
                else [_loop(component, width, rewritten)]
            ),
            *after,
        ],
    )


def _stage_group(
    plan: _Plan,
    shared: Mapping[str, str],
    first_row: str,
    slot: str,
    end: str,
    new_var: Callable[[str], str],
) -> list[ast.stmt]:
    assert isinstance(plan.row_loop.target, ast.Name)
    row_name = plan.row_loop.target.id
    tid = "cutlass.Int32(cute.arch.thread_idx()[0])"
    result: list[ast.stmt] = []
    for key, access in plan.staged_loads.items():
        relative = new_var("resident_stage_row")
        span = plan.layout.feature_extent if access.feature_stride else 1
        width = min(span, 4 if access.dtype == "cutlass.Float32" else 8)
        column = new_var("resident_stage_column") if access.feature_stride else "0"
        offset = _freeze_definition(
            access.offset,
            {
                row_name: ast.parse(
                    f"cutlass.Int32(({first_row}) + cutlass.Int64({relative}))",
                    mode="eval",
                ).body,
                plan.layout.index_name: ast.parse(column, mode="eval").body,
            },
        )
        assert offset is not None
        copy = statement_from_string(
            f"_cute_resident_copy_async({access.tensor}, {shared[key]}, "
            f"{ast.unparse(offset)}, cutlass.Int32((({slot}) * {plan.group_rows} + "
            f"{relative}) * {span}) + {column}, {width})"
        )
        if access.feature_stride:
            chunk = new_var("resident_stage_chunk")
            conditional = statement_from_string(f"if {column} < {span}:\n    pass")
            assert isinstance(conditional, ast.If)
            conditional.body = [copy]
            row_body: list[ast.stmt] = [
                _loop(
                    chunk,
                    (span + plan.layout.threads * width - 1)
                    // (plan.layout.threads * width),
                    [
                        statement_from_string(
                            f"{column} = ({tid} + cutlass.Int32({chunk}) * "
                            f"{plan.layout.threads}) * {width}"
                        ),
                        conditional,
                    ],
                )
            ]
        else:
            conditional = statement_from_string(f"if {tid} == 0:\n    pass")
            assert isinstance(conditional, ast.If)
            conditional.body = [copy]
            row_body = [conditional]
        if plan.partial_rows:
            active = statement_from_string(
                f"if ({first_row}) + cutlass.Int64({relative}) < ({end}):\n    pass"
            )
            assert isinstance(active, ast.If)
            active.body = row_body
            row_body = [active]
        result.append(_loop(relative, plan.group_rows, row_body))
    return result


def _shared_loads(
    plan: _Plan,
    statements: list[ast.stmt],
    shared: Mapping[str, str],
    slot: str,
    relative: str,
) -> list[ast.stmt]:
    result = []
    for statement in statements:
        statement = _clone(statement)
        key = (
            ast.dump(statement.value)
            if isinstance(statement, ast.Assign)
            and isinstance(statement.value, ast.Call)
            else ""
        )
        access = plan.staged_loads.get(key)
        if access is not None:
            assert isinstance(statement, ast.Assign)
            span = plan.layout.feature_extent if access.feature_stride else 1
            column = plan.layout.index_name if access.feature_stride else "0"
            offset = ast.parse(
                f"cutlass.Int32(({slot} * {plan.group_rows} + {relative}) * {span})"
                f" + cutlass.Int32({column})",
                mode="eval",
            ).body
            statement.value = ast.parse(
                f"({shared[key]}.iterator + {ast.unparse(offset)}).load()", mode="eval"
            ).body
            if access.feature_stride:
                plan.vector_accesses[ast.dump(statement.value)] = _VectorAccess(
                    shared[key], offset, access.dtype, False, 1
                )
        result.append(statement)
    return result


def _local_tree_update(
    lane: str, extent: int, value: str, fragment: str, result: str
) -> list[ast.stmt]:
    """Combine adjacent complete subtrees using logarithmic temporary storage.

    Every branch is resolved while unrolling the complete feature fragment.
    Slot k retains its left subtree until the corresponding right subtree is
    complete. Only the last feature writes the final sum. This reassociates
    the local FP32 sum; cross-row carries and the CTA reduction are unchanged.
    """
    # Preserve the positive-zero sum identity, including an all-negative-zero
    # fragment and the one-element case. Reassociation alone need not do so.
    body: list[ast.stmt] = [
        statement_from_string(f"{result} = cutlass.Float32(0) + {value}")
    ]
    for level in reversed(range(extent.bit_length() - 1)):
        half = 1 << level
        branch = statement_from_string(
            f"if cutlass.const_expr(({lane} % {2 * half}) < {half}):\n"
            f"    {fragment}[{level}] = {value}\n"
            f"else:\n    pass"
        )
        assert isinstance(branch, ast.If)
        branch.orelse = [
            statement_from_string(f"{value} = {fragment}[{level}] + {value}"),
            *body,
        ]
        body = [branch]
    return body


def _emit(
    plan: _Plan,
    new_var: Callable[[str], str],
    *,
    local_tree: bool = False,
    row_schedule: str = "batched",
    pack_output: bool = False,
) -> list[ast.stmt]:
    lane = plan.lane_name
    extent = plan.layout.lane_extent
    rows = plan.group_rows
    serial = row_schedule != "batched"
    deferred = row_schedule == "serial_deferred"
    stored_rows = 1 if serial else rows
    depth = plan.pipeline_depth
    marker_count = len(plan.markers)
    group, relative = new_var("resident_row_group"), new_var("resident_row")
    assert isinstance(plan.row_loop.target, ast.Name)
    row_name = plan.row_loop.target.id
    row_index = lane if serial else f"{relative} * {extent} + {lane}"
    row_storage_index = "0" if serial else relative
    shared = {key: new_var("resident_shared") for key in plan.staged_loads}
    slot = new_var("resident_slot")
    state = {name: new_var("resident_state") for name in plan.state_names}
    stash = {name: new_var("resident_value") for name in plan.stash_names}
    row_stash = {name: new_var("resident_row_value") for name in plan.row_stash_names}
    partials, sums = new_var("resident_partials"), new_var("resident_sums")
    declarations = [
        statement_from_string(
            f"{fragment} = cute.make_rmem_tensor({size}, cutlass.Float32)"
        )
        for fragment, size in [
            *((name, extent) for name in state.values()),
            *((name, stored_rows * extent) for name in stash.values()),
            *((name, stored_rows) for name in row_stash.values()),
            (partials, stored_rows * marker_count),
        ]
    ]
    scratch = new_var("resident_reduce_shared") if serial else None
    if scratch is not None:
        declarations.append(
            statement_from_string(
                f"{scratch} = cutlass.utils.SmemAllocator().allocate_tensor(cutlass.Float32, cute.make_layout({rows * marker_count * max(1, plan.layout.threads // 32)}))"
            )
        )
    for key, access in plan.staged_loads.items():
        span = plan.layout.feature_extent if access.feature_stride else 1
        size = depth * rows * span
        declarations.append(
            statement_from_string(
                f"{shared[key]} = cute.make_tensor(cute.arch.alloc_smem("
                f"{access.dtype}, {size}, alignment=16), cute.make_layout({size}))"
            )
        )
    init = [*map(_clone, plan.prefix)]
    init.extend(
        statement_from_string(f"{fragment}[{lane}] = {name}")
        for name, fragment in state.items()
    )
    row_assignment = statement_from_string(
        f"{row_name} = cutlass.Int32({group} + cutlass.Int64({relative}))"
    )
    producer_row: list[ast.stmt] = [
        _clone(row_assignment),
        *_shared_loads(plan, plan.invariant_producer, shared, slot, relative),
    ]
    producer_row.extend(
        statement_from_string(f"{fragment}[{row_storage_index}] = {name}")
        for name, fragment in row_stash.items()
    )
    accumulators = {
        marker.result_var: new_var("resident_sum") for marker in plan.markers
    }
    tree_fragments = (
        {marker.result_var: new_var("resident_tree") for marker in plan.markers}
        if local_tree and extent > 1
        else {}
    )
    declarations.extend(
        statement_from_string(
            f"{fragment} = cute.make_rmem_tensor({extent.bit_length() - 1}, cutlass.Float32)"
        )
        for fragment in tree_fragments.values()
    )
    producer_row.extend(
        statement_from_string(f"{name} = cutlass.Float32(0)")
        for name in accumulators.values()
    )
    producer = [
        statement_from_string(f"{name} = {fragment}[{lane}]")
        for name, fragment in state.items()
    ]
    producer.extend(map(_clone, plan.replay_prefix))
    producer.extend(_shared_loads(plan, plan.producer, shared, slot, relative))
    producer.extend(
        statement_from_string(f"{state[name]}[{lane}] = {name}")
        for name in plan.carry_names
    )
    producer.extend(
        statement_from_string(f"{fragment}[{row_index}] = {name}")
        for name, fragment in stash.items()
    )
    for marker in plan.markers:
        accumulator = accumulators[marker.result_var]
        if local_tree:
            value = new_var("resident_tree_value")
            producer.append(statement_from_string(f"{value} = {marker.input_name}"))
            producer.extend(
                _local_tree_update(
                    lane,
                    extent,
                    value,
                    tree_fragments.get(marker.result_var, ""),
                    accumulator,
                )
            )
        else:
            producer.append(
                statement_from_string(
                    f"{accumulator} = {accumulator} + {marker.input_name}"
                )
            )
    producer_row.append(_feature_loop(plan, producer, new_var))
    producer_row.extend(
        statement_from_string(
            f"{partials}[{index if serial else f'{relative} * {marker_count} + {index}'}] = {accumulators[marker.result_var]}"
        )
        for index, marker in enumerate(plan.markers)
    )
    consumer_row: list[ast.stmt] = [_clone(row_assignment)]
    consumer_row.extend(
        statement_from_string(f"{name} = {fragment}[{row_storage_index}]")
        for name, fragment in row_stash.items()
    )
    consumer_row.extend(map(_clone, plan.replay_row))
    consumer_row.extend(
        statement_from_string(
            f"{marker.result_var} = cutlass.Float32({sums}[{index if serial else f'{relative} * {marker_count} + {index}'}])"
        )
        for index, marker in enumerate(plan.markers)
    )
    consumer_row.extend(map(_clone, plan.invariant_consumer))
    consumer = [
        statement_from_string(f"{name} = {fragment}[{lane}]")
        for name, fragment in state.items()
        if name not in plan.carry_names
    ]
    consumer.extend(map(_clone, plan.replay_prefix))
    consumer.extend(
        statement_from_string(f"{name} = {fragment}[{row_index}]")
        for name, fragment in stash.items()
    )
    consumer.extend(map(_clone, plan.consumer))
    consumer_row.append(_feature_loop(plan, consumer, new_var, pack_output=pack_output))
    row_loop = _clone(plan.row_loop)
    assert isinstance(row_loop, ast.For) and isinstance(row_loop.iter, ast.Call)
    row_loop.target = ast.Name(id=group, ctx=ast.Store())
    bounds = row_loop.iter.args[:2]
    if len(bounds) == 1:
        bounds.insert(0, ast.Constant(value=0))
    # Evaluate the original final Int32 bounds before widening the scheduler.
    # A grouped increment or a speculative next-row guard can cross INT32_MAX,
    # even though every original row is representable. Only active rows narrow
    # back to the original type before evaluating their unchanged arithmetic.
    bounds = [
        ast.parse(f"cutlass.Int64({ast.unparse(bound)})", mode="eval").body
        for bound in bounds
    ]
    row_loop.iter.args = [*bounds, ast.Constant(value=rows)]
    begin, end = (ast.unparse(bound) for bound in bounds)
    partial_init = []
    if plan.partial_rows and not serial:
        for_row_guard = f"if {group} + cutlass.Int64({relative}) < ({end}):\n    pass"
        active_producer = statement_from_string(for_row_guard)
        active_consumer = statement_from_string(for_row_guard)
        assert isinstance(active_producer, ast.If) and isinstance(
            active_consumer, ast.If
        )
        active_producer.body = producer_row
        active_consumer.body = consumer_row
        producer_row = [active_producer]
        consumer_row = [active_consumer]
        index = new_var("resident_partial_index")
        partial_init = [
            _loop(
                index,
                rows * marker_count,
                [statement_from_string(f"{partials}[{index}] = cutlass.Float32(0)")],
            )
        ]
    row_loop.body = [
        *partial_init,
        _loop(relative, rows, producer_row),
        statement_from_string(
            f"{sums} = _cute_resident_sums({partials}, {plan.layout.threads})"
        ),
        _loop(relative, rows, consumer_row),
    ]
    if serial:
        serial_body: list[ast.stmt] = [
            *producer_row,
            statement_from_string(
                f"{sums} = _cute_resident_sums_disjoint({partials}, {plan.layout.threads}, {scratch}, {relative})"
            ),
            *([] if deferred else [statement_from_string("cute.arch.sync_threads()")]),
            *consumer_row,
        ]
        if plan.partial_rows:
            active_row = statement_from_string(
                f"if {group} + cutlass.Int64({relative}) < ({end}):\n    pass"
            )
            assert isinstance(active_row, ast.If)
            active_row.body = serial_body
            serial_body = [active_row]
        row_loop.body = [_loop(relative, rows, serial_body)]
    prologue: list[ast.stmt] = []
    if shared:
        # Keep the depth-two AST and variable allocation order unchanged.
        # Deeper rings prefill D-1 guarded groups; all threads commit a group,
        # including threads with no copy, so each wait has the same age.
        initial_row, initial_slot = begin, "0"
        initial = new_var("resident_prefetch_initial") if depth > 2 else None
        if initial is not None:
            initial_row = new_var("resident_prefetch_row")
            initial_slot = new_var("resident_prefetch_slot")
        initial_stage = statement_from_string(
            f"if ({initial_row}) < ({end}):\n    pass"
        )
        assert isinstance(initial_stage, ast.If)
        initial_stage.body = [
            *_stage_group(plan, shared, initial_row, initial_slot, end, new_var),
            statement_from_string("cute.arch.cp_async_commit_group()"),
        ]
        prologue = [statement_from_string(f"{slot} = cutlass.Int32(0)"), initial_stage]
        if initial is not None:
            prologue[-1] = _loop(
                initial,
                depth - 1,
                [
                    statement_from_string(
                        f"{initial_row} = ({begin}) + cutlass.Int64({initial}) * {rows}"
                    ),
                    statement_from_string(f"{initial_slot} = cutlass.Int32({initial})"),
                    initial_stage,
                ],
            )
        lookahead = (depth - 1) * rows
        next_slot = (
            f"1 - {slot}"
            if depth == 2
            else f"({slot} + cutlass.Int32({depth - 1})) % cutlass.Int32({depth})"
        )
        next_stage = statement_from_string(
            f"if ({group} + {lookahead}) < ({end}):\n    pass\nelse:\n    pass"
        )
        assert isinstance(next_stage, ast.If)
        next_stage.body = [
            *_stage_group(
                plan, shared, f"{group} + {lookahead}", next_slot, end, new_var
            ),
            statement_from_string("cute.arch.cp_async_commit_group()"),
            statement_from_string(f"cute.arch.cp_async_wait_group({depth - 1})"),
        ]
        next_stage.orelse = [statement_from_string("cute.arch.cp_async_wait_group(0)")]
        row_loop.body = [
            next_stage,
            statement_from_string("cute.arch.sync_threads()"),
            *row_loop.body,
            *([] if deferred else [statement_from_string("cute.arch.sync_threads()")]),
            statement_from_string(
                f"{slot} = cutlass.Int32(1) - {slot}"
                if depth == 2
                else f"{slot} = ({slot} + cutlass.Int32(1)) % cutlass.Int32({depth})"
            ),
        ]
    suffix = [
        statement_from_string(f"{name} = {fragment}[{lane}]")
        for name, fragment in state.items()
    ]
    suffix.extend(map(_clone, plan.replay_prefix))
    suffix.extend(map(_clone, plan.suffix))
    return [
        *declarations,
        _feature_loop(plan, init, new_var),
        *map(_clone, plan.row_setup),
        *prologue,
        row_loop,
        _feature_loop(plan, suffix, new_var),
    ]


def materialize_resident_reductions(
    body: list[ast.AST],
    *,
    layouts: Mapping[str, ResidentReductionLayout],
    tensor_dtypes: Mapping[str, str],
    tensor_strides: Mapping[tuple[str, int], int],
    disjoint_pairs: set[frozenset[str]],
    rename_groups: dict[str, str],
    new_var: Callable[[str], str],
    constexpr_values: Mapping[str, int],
    uniform_names: set[str],
    tensor_alignments: Mapping[str, int] | None = None,
    group_rows: int = 1,
    pipelined: bool = False,
    pipeline_depth: int = 2,
    local_tree: bool = False,
    row_schedule: str = "batched",
    pack_output: bool = False,
    shared_memory_budget: int = 0,
    require_proof: bool = False,
) -> list[ast.AST]:
    result: list[ast.AST] = []
    materialized: set[str] = set()
    for index, statement in enumerate(body):
        layout = (
            layouts.get(statement.target.id)
            if isinstance(statement, ast.For) and isinstance(statement.target, ast.Name)
            else None
        )
        if layout is None:
            result.append(statement)
            continue
        cloned = _clone(statement)
        ast_rename(cloned, rename_groups)
        assert isinstance(cloned, ast.For)
        prelude = [_clone(node) for node in body[:index]]
        for node in prelude:
            ast_rename(node, rename_groups)
        names = [_assignment(node) for node in prelude if isinstance(node, ast.Assign)]
        if (
            not _known_region(prelude)
            or len(names) != len(set(names))
            or set(names) & set(tensor_dtypes)
        ):
            result.append(statement)
            continue
        definitions = _definitions(
            prelude,
            set(),
            {
                name: ast.Constant(value=value)
                for name, value in constexpr_values.items()
            },
        )
        if definitions is None:
            result.append(statement)
            continue
        plan = _plan(
            cloned,
            layout,
            tensor_dtypes,
            tensor_strides,
            disjoint_pairs,
            definitions,
            uniform_names,
            tensor_alignments or {},
            group_rows,
            pipelined,
            pipeline_depth,
            shared_memory_budget,
            row_schedule,
            pack_output,
        )
        escapes = {
            rename_groups.get(name, name)
            for later in body[index + 1 :]
            for name in _reads(later)
        } & _binding_write_roots(cloned)
        if plan is None or escapes:
            result.append(statement)
        else:
            result.extend(
                _emit(
                    plan,
                    new_var,
                    local_tree=local_tree,
                    row_schedule=row_schedule,
                    pack_output=pack_output,
                )
            )
            materialized.add(plan.lane_name)
    if require_proof and materialized != layouts.keys():
        raise exc.BackendUnsupported(
            "cute",
            "resident reduction could not prove rectangular independent FP32 carries",
        )
    return result
