"""Pack a narrow triple BF16 reduction into BF16x2 grouped accumulation.

This opt-in pass targets the common statistics pattern
``sum(x*x), sum(y*y), sum(x*y)`` after CuTe vector-load lowering.  It batches
all V4 loads for one logical reduction group, consumes them as aligned packed
BF16x2 words, and flushes the three packed accumulators to FP32 once per group.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import os

PACKED_BF16X2_CONFIG_KEY = "cute_packed_bf16x2_reduction"
PACKED_BF16X2_THREADS_CONFIG_KEY = "cute_packed_bf16x2_threads_per_row"
PACKED_BF16X2_WARP0_CONFIG_KEY = "cute_packed_bf16x2_warp0_epilogue"


@dataclass
class _PackedPattern:
    lane_loop: ast.For
    lane_var: str
    lane_reps: int
    vector_width: int
    lane_base_name: str
    lane_base_rhs: ast.expr
    tensor_names: tuple[str, str]
    offset_exprs: tuple[ast.expr, ast.expr]
    mask_name: str
    accumulator_names: tuple[str, str, str]


_COUNTER = 0


def _assign(stmt: ast.stmt) -> tuple[str, ast.expr] | None:
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    target = stmt.targets[0]
    if not isinstance(target, ast.Name):
        return None
    return target.id, stmt.value


def _int_constant(node: ast.expr) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "cutlass"
        and node.func.attr in {"Int32", "Int64"}
        and len(node.args) == 1
    ):
        return _int_constant(node.args[0])
    return None


def _range_reps(loop: ast.For) -> int | None:
    call = loop.iter
    if not (isinstance(call, ast.Call) and len(call.args) == 1):
        return None
    is_range = isinstance(call.func, ast.Name) and call.func.id == "range"
    is_constexpr_range = (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "range_constexpr"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "cutlass"
    )
    if not (is_range or is_constexpr_range):
        return None
    return _int_constant(call.args[0])


def _is_vector_load(value: ast.expr) -> bool:
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "load"
        and isinstance(value.func.value, ast.Attribute)
        and value.func.value.attr == "arch"
        and isinstance(value.func.value.value, ast.Name)
        and value.func.value.value.id == "cute"
        and len(value.args) == 2
    )


def _flatten_add(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [*_flatten_add(node.left), *_flatten_add(node.right)]
    return [node]


def _pointer_parts(pointer: ast.expr) -> tuple[str, ast.expr] | None:
    terms = _flatten_add(pointer)
    iterator_index = next(
        (
            index
            for index, term in enumerate(terms)
            if isinstance(term, ast.Attribute)
            and term.attr == "iterator"
            and isinstance(term.value, ast.Name)
        ),
        None,
    )
    if iterator_index is None:
        return None
    iterator = terms.pop(iterator_index)
    assert isinstance(iterator, ast.Attribute)
    assert isinstance(iterator.value, ast.Name)
    if not terms:
        return None
    offset = terms[0]
    for term in terms[1:]:
        offset = ast.BinOp(left=offset, op=ast.Add(), right=term)
    return iterator.value.id, offset


def _float32_cast_of(value: ast.expr, name: str) -> bool:
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "Float32"
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "cutlass"
        and len(value.args) == 1
        and isinstance(value.args[0], ast.Name)
        and value.args[0].id == name
    )


def _is_bfloat16_masked_load(value: ast.expr) -> bool:
    if not (
        isinstance(value, ast.IfExp)
        and isinstance(value.test, ast.Name)
        and isinstance(value.body, ast.Call)
        and isinstance(value.body.func, ast.Attribute)
        and value.body.func.attr == "bitcast"
        and len(value.body.args) == 1
    ):
        return False
    dtype = value.body.args[0]
    return bool(
        isinstance(dtype, ast.Attribute)
        and dtype.attr == "BFloat16"
        and isinstance(dtype.value, ast.Name)
        and dtype.value.id == "cutlass"
    )


def _match_scalar_body(
    body: list[ast.stmt],
) -> tuple[str, tuple[str, str, str]] | None:
    assigns = [_assign(stmt) for stmt in body]
    if any(item is None for item in assigns):
        return None
    pairs = [item for item in assigns if item is not None]

    masked_loads = [
        (name, value) for name, value in pairs if _is_bfloat16_masked_load(value)
    ]
    if len(masked_loads) != 2:
        return None
    first_masked_load = masked_loads[0][1]
    second_masked_load = masked_loads[1][1]
    assert isinstance(first_masked_load, ast.IfExp)
    assert isinstance(first_masked_load.test, ast.Name)
    assert isinstance(second_masked_load, ast.IfExp)
    assert isinstance(second_masked_load.test, ast.Name)
    if first_masked_load.test.id != second_masked_load.test.id:
        return None
    mask_name = first_masked_load.test.id

    converted: list[str] = []
    for load_name, _ in masked_loads:
        matches = [name for name, value in pairs if _float32_cast_of(value, load_name)]
        if len(matches) != 1:
            return None
        converted.append(matches[0])

    products: dict[str, tuple[str, str]] = {}
    for name, value in pairs:
        if (
            isinstance(value, ast.BinOp)
            and isinstance(value.op, ast.Mult)
            and isinstance(value.left, ast.Name)
            and isinstance(value.right, ast.Name)
            and value.left.id in converted
            and value.right.id in converted
        ):
            products[name] = (value.left.id, value.right.id)

    accumulators: dict[tuple[str, str], str] = {}
    for name, value in pairs:
        if not (
            isinstance(value, ast.BinOp)
            and isinstance(value.op, ast.Add)
            and isinstance(value.left, ast.Name)
            and value.left.id == name
            and isinstance(value.right, ast.Call)
            and len(value.right.args) == 1
            and isinstance(value.right.args[0], ast.Name)
        ):
            continue
        product = products.get(value.right.args[0].id)
        if product is not None:
            accumulators[product] = name

    first, second = converted
    square_first = accumulators.get((first, first))
    square_second = accumulators.get((second, second))
    cross = accumulators.get((first, second)) or accumulators.get((second, first))
    if square_first is None or square_second is None or cross is None:
        return None
    return mask_name, (square_first, square_second, cross)


def _match_outer(loop: ast.For) -> _PackedPattern | None:
    if len(loop.body) != 1 or not isinstance(loop.body[0], ast.For):
        return None
    lane_loop = loop.body[0]
    if not isinstance(lane_loop.target, ast.Name):
        return None
    lane_reps = _range_reps(lane_loop)
    if lane_reps is None or lane_reps < 4:
        return None
    if len(lane_loop.body) != 4:
        return None
    lane_base = _assign(lane_loop.body[0])
    first_load = _assign(lane_loop.body[1])
    second_load = _assign(lane_loop.body[2])
    vector_loop = lane_loop.body[3]
    if lane_base is None or first_load is None or second_load is None:
        return None
    if not (
        _is_vector_load(first_load[1])
        and _is_vector_load(second_load[1])
        and isinstance(vector_loop, ast.For)
    ):
        return None
    assert isinstance(first_load[1], ast.Call)
    assert isinstance(second_load[1], ast.Call)
    first_pointer = _pointer_parts(first_load[1].args[0])
    second_pointer = _pointer_parts(second_load[1].args[0])
    vector_reps = _range_reps(vector_loop)
    if vector_reps != 4:
        return None
    if first_pointer is None or second_pointer is None:
        return None
    scalar_match = _match_scalar_body(vector_loop.body)
    if scalar_match is None:
        return None
    mask_name, accumulators = scalar_match
    return _PackedPattern(
        lane_loop=lane_loop,
        lane_var=lane_loop.target.id,
        lane_reps=lane_reps,
        vector_width=vector_reps,
        lane_base_name=lane_base[0],
        lane_base_rhs=lane_base[1],
        tensor_names=(first_pointer[0], second_pointer[0]),
        offset_exprs=(first_pointer[1], second_pointer[1]),
        mask_name=mask_name,
        accumulator_names=accumulators,
    )


def _statements(source: str) -> list[ast.stmt]:
    return ast.parse(source).body


def _widen_offset(node: ast.expr) -> ast.expr:
    class WidenInt32(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            visited = self.generic_visit(node)
            if (
                isinstance(visited, ast.Call)
                and isinstance(visited.func, ast.Attribute)
                and visited.func.attr == "Int32"
                and isinstance(visited.func.value, ast.Name)
                and visited.func.value.id == "cutlass"
            ):
                visited.func.attr = "Int64"
            return visited

    copied = ast.parse(ast.unparse(node), mode="eval").body
    widened = WidenInt32().visit(copied)
    assert isinstance(widened, ast.expr)
    return widened


def _rewrite_outer(loop: ast.For, pattern: _PackedPattern) -> list[ast.stmt]:
    global _COUNTER
    suffix = _COUNTER
    _COUNTER += 1
    branch_cache = f"_packed_branch_{suffix}"
    stream_cache = f"_packed_stream_{suffix}"
    packed_lane = f"_packed_lane_{suffix}"
    branch_vec = f"_packed_branch_vec_{suffix}"
    stream_vec = f"_packed_stream_vec_{suffix}"
    branch_addr = f"_packed_branch_addr_{suffix}"
    stream_addr = f"_packed_stream_addr_{suffix}"
    branch_ptr = f"_packed_branch_ptr_{suffix}"
    stream_ptr = f"_packed_stream_ptr_{suffix}"
    branch_pair = f"_packed_branch_pair_{suffix}"
    stream_pair = f"_packed_stream_pair_{suffix}"
    branch_acc = f"_packed_branch_acc_{suffix}"
    stream_acc = f"_packed_stream_acc_{suffix}"
    cross_acc = f"_packed_cross_acc_{suffix}"

    lane = pattern.lane_var
    lane_base = pattern.lane_base_name
    lane_rhs = ast.unparse(pattern.lane_base_rhs)
    branch_offset = ast.unparse(_widen_offset(pattern.offset_exprs[0]))
    stream_offset = ast.unparse(_widen_offset(pattern.offset_exprs[1]))
    branch_tensor, stream_tensor = pattern.tensor_names
    acc_branch, acc_stream, acc_cross = pattern.accumulator_names
    words_per_lane = pattern.vector_width // 2
    pair_count = pattern.lane_reps * words_per_lane

    loop.body = _statements(
        f"""
for {lane} in cutlass.range_constexpr({pattern.lane_reps}):
    {lane_base} = {lane_rhs}
    {branch_addr} = {branch_tensor}.iterator.toint() + ({branch_offset}) * 2
    {branch_ptr} = cute.make_ptr(cutlass.Uint32, {branch_addr}, cute.AddressSpace.gmem, assumed_align=8)
    {branch_vec} = cute.arch.load({branch_ptr}, ir.VectorType.get([{words_per_lane}], cutlass.Uint32.mlir_type))
    for {packed_lane} in cutlass.range_constexpr({words_per_lane}):
        {branch_cache}[{lane} * {words_per_lane} + {packed_lane}] = {branch_vec}[{packed_lane}]
for {lane} in cutlass.range_constexpr({pattern.lane_reps}):
    {lane_base} = {lane_rhs}
    {stream_addr} = {stream_tensor}.iterator.toint() + ({stream_offset}) * 2
    {stream_ptr} = cute.make_ptr(cutlass.Uint32, {stream_addr}, cute.AddressSpace.gmem, assumed_align=8)
    {stream_vec} = cute.arch.load({stream_ptr}, ir.VectorType.get([{words_per_lane}], cutlass.Uint32.mlir_type))
    for {packed_lane} in cutlass.range_constexpr({words_per_lane}):
        {stream_cache}[{lane} * {words_per_lane} + {packed_lane}] = {stream_vec}[{packed_lane}]
{branch_acc} = cutlass.Uint32(0)
{stream_acc} = cutlass.Uint32(0)
{cross_acc} = cutlass.Uint32(0)
for {packed_lane} in cutlass.range_constexpr({pair_count}):
    {branch_pair} = cutlass.Uint32({branch_cache}[{packed_lane}]) if {pattern.mask_name} else cutlass.Uint32(0)
    {stream_pair} = cutlass.Uint32({stream_cache}[{packed_lane}]) if {pattern.mask_name} else cutlass.Uint32(0)
    {branch_acc}, {stream_acc}, {cross_acc} = _cute_bf16x2_accumulate3({branch_pair}, {stream_pair}, {branch_acc}, {stream_acc}, {cross_acc})
{acc_branch} = {acc_branch} + _cute_bf16x2_sum_to_f32({branch_acc})
{acc_stream} = {acc_stream} + _cute_bf16x2_sum_to_f32({stream_acc})
{acc_cross} = {acc_cross} + _cute_bf16x2_sum_to_f32({cross_acc})
"""
    )
    caches = _statements(
        f"""
{branch_cache} = cute.make_rmem_tensor({pair_count}, cutlass.Uint32)
{stream_cache} = cute.make_rmem_tensor({pair_count}, cutlass.Uint32)
"""
    )
    return [*caches, loop]


def _transform_body(body: list[ast.stmt]) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.For, ast.If, ast.With)):
            stmt.body = _transform_body(stmt.body)
            if isinstance(stmt, (ast.For, ast.If)):
                stmt.orelse = _transform_body(stmt.orelse)
        if isinstance(stmt, ast.For):
            pattern = _match_outer(stmt)
            if pattern is not None:
                result.extend(_rewrite_outer(stmt, pattern))
                continue
        result.append(stmt)
    return result


def pack_bf16x2_reductions(
    body: list[ast.stmt], *, enabled: bool = False
) -> list[ast.stmt]:
    """Apply the opt-in packed triple-reduction transform."""
    if not enabled and not os.environ.get("HELION_CUTE_PACKED_BF16X2_REDUCTION"):
        return body
    global _COUNTER
    _COUNTER = 0
    return _transform_body(body)
