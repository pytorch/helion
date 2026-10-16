"""Prove an exact contiguous input tile before replacing a row's scalar loader."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

from . import cluster_online_pair as pair
from .pipeline_state_loads import _indices_by_stride
from .pipeline_state_loads import _pointer_info
from .pipeline_state_loads import _vector_type_info
from .rolled_cluster_online_pair import _domain
from .rolled_cluster_online_pair import _literal
from .rolled_cluster_online_pair import _under_domain

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..device_function import DeviceFunction


_INPUT_CARRIERS = {
    "cutlass.BFloat16": ("cutlass.Uint16", 2),
    "cutlass.Float16": ("cutlass.Uint16", 2),
    "cutlass.Float32": ("cutlass.Uint32", 4),
}


@dataclass(frozen=True)
class TensorCopyInput:
    dtype: str
    shape: tuple[int, int]
    strides: tuple[int, int]
    base_alignment: int


@dataclass(frozen=True)
class TensorCopyContext:
    inputs: Mapping[str, TensorCopyInput]
    threads: tuple[int, int, int]


@dataclass(frozen=True)
class TensorCopyPlan:
    source: str
    row: ast.expr
    column_tile: ast.expr
    columns: int
    threads: int
    vector_width: int

    def assignment(self, target: str) -> ast.stmt:
        return pair._stmt(
            f"{target} = _cute_complete_row_tensor_copy({self.source}, "
            f"cutlass.Int32({ast.unparse(self.row)}), "
            f"cutlass.Int32({ast.unparse(self.column_tile)}), "
            f"columns={self.columns}, threads={self.threads}, "
            f"vector_width={self.vector_width})"
        )


def context_from_device_function(
    fn: DeviceFunction,
    threads: tuple[int, int, int],
    dtypes: Mapping[str, str],
) -> TensorCopyContext:
    from ..compile_environment import CompileEnvironment
    from ..device_function import TensorArg
    from ..device_function import TensorDescriptorArg
    from .memory_ops import runtime_tensor_has_specialized_alignment

    env = CompileEnvironment.current()
    strides = fn.proven_tensor_stride_values()
    inputs = {}
    if env.settings.static_shapes:
        for arg in fn.arguments:
            if not isinstance(arg, TensorArg) or isinstance(arg, TensorDescriptorArg):
                continue
            shape = arg.fake_value.shape
            if len(shape) != 2 or not all(type(size) is int for size in shape):
                continue
            dtype = dtypes.get(arg.name)
            if dtype not in _INPUT_CARRIERS:
                continue
            if (arg.name, 0) not in strides or (arg.name, 1) not in strides:
                continue
            alignment = next(
                (
                    size
                    for size in (32, 16, 8, 4, 2)
                    if runtime_tensor_has_specialized_alignment(
                        env, arg.fake_value, size
                    )
                ),
                0,
            )
            inputs[arg.name] = TensorCopyInput(
                dtype,
                (shape[0], shape[1]),
                (strides[arg.name, 0], strides[arg.name, 1]),
                alignment,
            )
    return TensorCopyContext(inputs, threads)


def _integer_cast(expr: ast.expr) -> ast.expr:
    while (
        isinstance(expr, ast.Call)
        and ast.unparse(expr.func) in ("cutlass.Int32", "cutlass.Int64")
        and len(expr.args) == 1
        and not expr.keywords
    ):
        expr = expr.args[0]
    return expr


def _coordinate(expr: ast.expr) -> str | None:
    expr = _integer_cast(expr)
    if (
        isinstance(expr, ast.Subscript)
        and isinstance(expr.value, ast.Call)
        and not expr.value.args
        and not expr.value.keywords
        and isinstance(expr.slice, ast.Constant)
        and type(expr.slice.value) is int
        and expr.slice.value in (0, 1, 2)
    ):
        path = ast.unparse(expr.value.func)
        if path == "cute.arch.thread_idx":
            return f"thread{expr.slice.value}"
        if path == "cute.arch.block_idx":
            return f"block{expr.slice.value}"
    return None


def _linear(
    expr: ast.expr,
    definitions: Mapping[str, ast.expr],
    active: frozenset[str] = frozenset(),
) -> dict[str, int] | None:
    expr = _integer_cast(expr)
    if isinstance(expr, ast.Constant) and type(expr.value) is int:
        return {"constant": expr.value}
    if coordinate := _coordinate(expr):
        return {coordinate: 1}
    if isinstance(expr, ast.Name):
        if expr.id.startswith("_tensor_copy_counter_"):
            return {expr.id: 1}
        if expr.id in active or expr.id not in definitions:
            return None
        return _linear(definitions[expr.id], definitions, active | {expr.id})
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
        value = _linear(expr.operand, definitions, active)
        return (
            {key: -coefficient for key, coefficient in value.items()}
            if value is not None
            else None
        )
    if not isinstance(expr, ast.BinOp):
        return None
    left = _linear(expr.left, definitions, active)
    right = _linear(expr.right, definitions, active)
    if left is None or right is None:
        return None
    if isinstance(expr.op, (ast.Add, ast.Sub)):
        result = dict(left)
        for key, value in right.items():
            result[key] = result.get(key, 0) + (
                -value if isinstance(expr.op, ast.Sub) else value
            )
        return {key: value for key, value in result.items() if value}
    if isinstance(expr.op, ast.Mult):
        if set(left) <= {"constant"}:
            return {
                key: value * left.get("constant", 0)
                for key, value in right.items()
                if value * left.get("constant", 0)
            }
        if set(right) <= {"constant"}:
            return {
                key: value * right.get("constant", 0)
                for key, value in left.items()
                if value * right.get("constant", 0)
            }
    return None


def _same_linear(
    first: ast.expr, second: ast.expr, definitions: Mapping[str, ast.expr]
) -> bool:
    value = _linear(first, definitions)
    return value is not None and value == _linear(second, definitions)


def _original_vector_element(
    expr: ast.expr, dtype: str, position: int
) -> ast.Call | None:
    storage = _INPUT_CARRIERS.get(dtype)
    if storage is None:
        return None
    carrier, _element_size = storage
    widening = False
    while (
        isinstance(expr, ast.Call)
        and ast.unparse(expr.func) == "cutlass.Float32"
        and len(expr.args) == 1
        and not expr.keywords
    ):
        widening = True
        expr = expr.args[0]
    # Half inputs must widen once. FP32 inputs must retain the exact original
    # Uint32 identity bitcast; numeric integer conversions are not equivalent.
    if (not widening and dtype != "cutlass.Float32") or not (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and expr.func.attr == "bitcast"
        and len(expr.args) == 1
        and ast.unparse(expr.args[0]) == dtype
        and not expr.keywords
    ):
        return None
    raw = expr.func.value
    if not (
        isinstance(raw, ast.Call)
        and ast.unparse(raw.func) == carrier
        and len(raw.args) == 1
        and not raw.keywords
        and isinstance(raw.args[0], ast.Subscript)
        and isinstance(raw.args[0].slice, ast.Constant)
        and raw.args[0].slice.value == position
    ):
        return None
    vector = raw.args[0].value
    return vector if isinstance(vector, ast.Call) else None


def prove_tensor_copy(
    body: list[ast.stmt],
    site: pair._Site,
    sum_site: pair._Site,
    fold: ast.Assign,
    caches: set[str],
    chain: list[tuple[ast.For, ast.stmt]],
    values: list[str],
    trips: list[int],
    constants: Mapping[str, int],
    context: TensorCopyContext,
) -> TensorCopyPlan | None:
    """Admit original Half widening or FP32 identity loads over a complete slice."""
    from ..tile_strategy import _is_proven_relocatable_assignment
    from .fuse_two_pass_loads import _range_bounds

    threads = site.group_span
    if threads is None or context.threads != (threads, 1, 1):
        return None
    if len(trips) != len(chain) - 1 or not values:
        return None
    writes = Counter(
        node.id
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    maximum_acc = site.call.args[0]
    sum_acc = sum_site.call.args[0]
    if not isinstance(maximum_acc, ast.Name) or not isinstance(sum_acc, ast.Name):
        return None
    initial = [
        statement.value
        for statement in body[: site.top_idx]
        if pair._is_name_assign(statement)
        and pair._assigned_name(statement) == maximum_acc.id
    ]
    if (
        writes[maximum_acc.id] != 2
        or writes[sum_acc.id] != 2
        or len(initial) != 1
        or _literal(initial[0]) != -float("inf")
    ):
        return None
    names = {
        node.id
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }
    if any(name.startswith(("_loop", "_tensor_copy_counter_")) for name in names):
        return None
    definitions: dict[str, ast.expr] = {
        name: ast.Constant(value=value)
        for name, value in constants.items()
        if type(value) is int and not writes[name]
    }
    for statement in body[: site.top_idx]:
        if (
            pair._is_name_assign(statement)
            and writes[pair._assigned_name(statement)] == 1
            and _is_proven_relocatable_assignment(statement, allow_load=False)
        ):
            definitions[pair._assigned_name(statement)] = statement.value
    for level, (loop, _) in enumerate(chain):
        if not isinstance(loop.target, ast.Name) or loop.orelse:
            return None
        bounds = _range_bounds(loop.iter)
        if bounds is None:
            return None
        begin, _, step = bounds
        trip = 1 if level == 0 else trips[level - 1]
        counter = (
            ast.Constant(value=0)
            if trip == 1
            else ast.Name(id=f"_tensor_copy_counter_{level}", ctx=ast.Load())
        )
        definitions[f"_loop{level}_"] = ast.BinOp(
            left=begin,
            op=ast.Add(),
            right=ast.BinOp(
                left=counter,
                op=ast.Mult(),
                right=step,
            ),
        )
        definitions[loop.target.id] = definitions[f"_loop{level}_"]
    vector_call = None
    domain = None
    source = None
    row = None
    column = None
    for position, text in enumerate(values):
        predicate, value = _domain(ast.parse(text, mode="eval").body, -float("inf"))
        if predicate is None:
            return None
        value = ast.parse(_under_domain(value, predicate), mode="eval").body
        # The source dtype is proved through the vector's exact input pointer.
        for name, metadata in context.inputs.items():
            candidate = _original_vector_element(value, metadata.dtype, position)
            if candidate is None or ast.unparse(candidate.func) != "cute.arch.load":
                continue
            if len(candidate.args) != 2 or candidate.keywords:
                continue
            vector_type = _vector_type_info(candidate.args[1])
            if vector_type != (len(values), _INPUT_CARRIERS[metadata.dtype][0]):
                continue
            pointer = candidate.args[0]
            if isinstance(pointer, ast.IfExp):
                if not _same_linear_mask(pointer.test, predicate, definitions):
                    continue
                pointer = pointer.body
            access = _pointer_info(pointer)
            if access is None or access.tensor_name != name:
                continue
            indices = _indices_by_stride(access.offset, name, 2)
            if indices is None:
                continue
            if vector_call is None:
                vector_call, domain, source = candidate, predicate, name
                row, column = indices[0], indices[1]
            elif (
                ast.unparse(candidate) != ast.unparse(vector_call)
                or source != name
                or not _same_linear(indices[0], cast("ast.expr", row), definitions)
                or not _same_linear(indices[1], cast("ast.expr", column), definitions)
                or not _same_linear_mask(
                    predicate, cast("ast.expr", domain), definitions
                )
            ):
                return None
            break
        else:
            return None
    if (
        vector_call is None
        or source is None
        or row is None
        or column is None
        or domain is None
    ):
        return None
    metadata = context.inputs[source]
    carrier, element_size = _INPUT_CARRIERS[metadata.dtype]
    vector_width = len(values)
    if vector_width not in (2, 4, 8):
        return None
    if (
        metadata.strides[1] != 1
        or metadata.strides[0] <= 0
        or metadata.strides[0] % vector_width
        or metadata.shape[0] <= 0
        or metadata.shape[1] <= 0
        or metadata.shape[1] % vector_width
        or metadata.base_alignment < vector_width * element_size
        or metadata.base_alignment % (vector_width * element_size)
        or (metadata.shape[0] - 1) * metadata.strides[0] + metadata.shape[1]
        >= (1 << 31)
    ):
        return None
    row_form = _linear(row, definitions)
    column_form = _linear(column, definitions)
    if (
        row_form is None
        or column_form is None
        or set(row_form) - {"block0", "constant"}
    ):
        return None
    if row_form != {"block0": 1}:
        return None
    if not (
        isinstance(domain, ast.Compare)
        and len(domain.ops) == len(domain.comparators) == 1
        and isinstance(domain.ops[0], ast.Lt)
        and _same_linear(domain.left, column, definitions)
        and _linear(domain.comparators[0], definitions)
        == {"constant": metadata.shape[1]}
    ):
        return None
    elements = 1
    for trip in trips:
        if trip <= 0:
            return None
        elements *= trip
    columns = elements * threads
    if elements % vector_width or column_form.get("thread0") != vector_width:
        return None
    allocations = {
        pair._assigned_name(statement): statement.value
        for statement in body
        if pair._is_name_assign(statement) and pair._assigned_name(statement) in caches
    }
    if set(allocations) != caches:
        return None
    for allocation in allocations.values():
        if not (
            isinstance(allocation, ast.Call)
            and ast.unparse(allocation.func) == "cute.make_rmem_tensor"
            and len(allocation.args) == 2
            and not allocation.keywords
            and isinstance(allocation.args[0], ast.Constant)
            and allocation.args[0].value == elements
            and ast.unparse(allocation.args[1]) == carrier
        ):
            return None
    expected = {"thread0": vector_width}
    stride = threads * vector_width
    for level in reversed(range(1, len(chain) - 1)):
        trip = trips[level - 1]
        if trip > 1:
            expected[f"_tensor_copy_counter_{level}"] = stride
        stride *= trip
    remainder = {
        key: value
        for key, value in column_form.items()
        if key not in expected and value
    }
    if any(column_form.get(key) != value for key, value in expected.items()):
        return None
    if set(remainder) - {"block1", "constant"}:
        return None
    if (
        remainder.get("block1", 0) != columns
        or remainder.get("constant", 0) < 0
        or remainder.get("constant", 0) % columns
    ):
        return None
    # No observable statement may disappear when the complete loader replaces
    # the old loop. Stores to its private register cache remain covered by the
    # existing MAX/SUM provenance proof; all other statements must be pure.
    for statement in site.for_node.body[: site.stmt_idx]:
        for node in ast.walk(statement):
            if isinstance(
                node, (ast.AnnAssign, ast.AugAssign, ast.If, ast.While, ast.Expr)
            ):
                return None
            if (
                isinstance(node, ast.Assign)
                and node is not fold
                and not (
                    pair._is_name_assign(node)
                    and _is_proven_relocatable_assignment(node, allow_load=True)
                )
            ):
                target = node.targets[0] if len(node.targets) == 1 else None
                if not (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in caches
                ):
                    return None
    private = {
        node.id
        for statement in site.for_node.body[: site.stmt_idx]
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    if any(
        set(pair._loads(statement)) & private
        for statement in (
            *body[site.top_idx + 1 : sum_site.top_idx],
            *sum_site.for_node.body[sum_site.stmt_idx + 1 :],
            *body[sum_site.top_idx + 1 :],
        )
    ):
        return None
    column_tile = ast.parse("cute.arch.block_idx()[1]", mode="eval").body
    bias = remainder.get("constant", 0) // columns
    if bias:
        column_tile = ast.BinOp(
            left=column_tile, op=ast.Add(), right=ast.Constant(value=bias)
        )
    row_expression = ast.parse("cute.arch.block_idx()[0]", mode="eval").body
    row_bias = row_form.get("constant", 0)
    if row_bias:
        row_expression = ast.BinOp(
            left=row_expression, op=ast.Add(), right=ast.Constant(value=row_bias)
        )
    return TensorCopyPlan(
        source, row_expression, column_tile, columns, threads, vector_width
    )


def _same_linear_mask(
    first: ast.expr, second: ast.expr, definitions: Mapping[str, ast.expr]
) -> bool:
    return (
        isinstance(first, ast.Compare)
        and isinstance(second, ast.Compare)
        and len(first.ops)
        == len(first.comparators)
        == len(second.ops)
        == len(second.comparators)
        == 1
        and isinstance(first.ops[0], ast.Lt)
        and isinstance(second.ops[0], ast.Lt)
        and _same_linear(first.left, second.left, definitions)
        and _same_linear(first.comparators[0], second.comparators[0], definitions)
    )
