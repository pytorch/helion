"""Literal device AST ownership/order interpreter with an independent sum tree.

The reference tree transcribes the fingerprinted ATen thread_reduce_impl and
block_y_reduce. Tests also mutate the emitted AST to make each obligation fail.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from helion._compiler.cute import paired_sum_runtime
from helion._compiler.cute import single_sum_runtime

if TYPE_CHECKING:
    from helion.runtime.cute.paired_sum import SumPlan


@dataclass(frozen=True, slots=True)
class Value:
    op: str
    args: tuple[object, ...] = ()

    def __add__(self, other: Value) -> Value:
        assert isinstance(other, Value)
        if self == ZERO and other == ZERO:
            return ZERO
        return Value("add_rn_f32", (self, other))


ZERO = Value("positive_zero")


def float32(value: object) -> Value:
    if isinstance(value, Value):
        return value
    assert value == 0
    return ZERO


def cast_bf16(value: Value) -> Value:
    return Value("cast_bfloat16", (value,))


def cast_fp16(value: Value) -> Value:
    return Value("cast_float16", (value,))


class Register:
    def __init__(self, shape: int | tuple[int, ...], dtype: object) -> None:
        self.shape = (shape,) if isinstance(shape, int) else shape
        self.element_type = dtype
        self.values = {}

    def _key(self, key: int | tuple[int, ...]) -> tuple[int, ...]:
        key = (key,) if isinstance(key, int) else key
        assert len(key) == len(self.shape)
        assert all(
            0 <= index < size for index, size in zip(key, self.shape, strict=True)
        )
        return key

    def __getitem__(self, key: int | tuple[int, ...]) -> Value:
        return self.values[self._key(key)]

    def __setitem__(self, key: int | tuple[int, ...], value: Value) -> None:
        assert isinstance(value, Value)
        self.values[self._key(key)] = value

    def fill(self, value: Value) -> None:
        import itertools

        for key in itertools.product(*(range(size) for size in self.shape)):
            self.values[key] = value


class Global:
    def __init__(
        self,
        kind: str,
        stream: int,
        rows: int,
        columns: int,
        stride: int,
        output_type: object = cast_bf16,
    ) -> None:
        self.kind = kind
        self.stream = stream
        self.rows = rows
        self.columns = columns
        self.layout = SimpleNamespace(stride=(stride, 1))
        self.element_type = float32 if kind == "input" else output_type
        self.loads = Counter()
        self.stores = {}

    def load(self, offset: int) -> Value:
        assert self.kind == "input"
        row, column = divmod(offset, self.layout.stride[0])
        assert 0 <= row < self.rows and 0 <= column < self.columns, (row, column)
        self.loads[row, column] += 1
        return Value("input", (self.stream, row, column))

    def store(self, offset: int, value: Value) -> None:
        assert self.kind == "output" and 0 <= offset < self.columns
        assert offset not in self.stores, (self.stream, offset)
        self.stores[offset] = value


@dataclass
class Shared:
    size: int
    values: dict[int, Value] = field(default_factory=dict)
    writes: Counter = field(default_factory=Counter)

    element_type = float32

    def load(self, offset: int) -> Value:
        assert 0 <= offset < self.size
        assert offset in self.values, ("shared read before publication", offset)
        return self.values[offset]

    def store(self, offset: int, value: Value) -> None:
        assert 0 <= offset < self.size
        self.values[offset] = value
        self.writes[offset] += 1


def load_vector(tensor: object, offset: int, width: int) -> Register:
    assert offset % width == 0
    value = Register(width, tensor.element_type)
    for lane in range(width):
        value[lane] = tensor.load(offset + lane)
    return value


def store_vector(tensor: object, offset: int, fragment: Register, width: int) -> None:
    assert offset % width == 0
    for lane in range(width):
        tensor.store(offset + lane, fragment[lane])


def literal_kernel_ast(single: bool = False) -> ast.FunctionDef:
    module = single_sum_runtime if single else paired_sum_runtime
    name = "single_sum_cast_kernel" if single else "paired_sum_cast_kernel"
    tree = ast.parse(Path(module.__file__).read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


class InterpreterTransform(ast.NodeTransformer):
    def visit_Expr(self, node: ast.Expr) -> object:
        if (
            isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "cute.arch.sync_threads"
        ):
            return ast.Expr(value=ast.Yield(value=ast.Constant("cta_barrier")))
        return self.generic_visit(node)


def run_literal(
    plan: SumPlan,
    *,
    stride: int | None = None,
    mutation: str | None = None,
    return_expressions: bool = False,
    single: bool = False,
    output_dtype: str = "bfloat16",
) -> dict[str, object]:
    output_type = {
        "bfloat16": cast_bf16,
        "float16": cast_fp16,
        "float32": float32,
    }[output_dtype]
    streams = 1 if single else 2
    stride = plan.columns if stride is None else stride
    if plan.columns == 0:
        assert plan.grid[0] == 0
        return {"plan": asdict(plan), "loads": 0, "stores": 0, "barriers": 0}
    inputs = [
        Global("input", stream, plan.rows, plan.columns, stride)
        for stream in range(streams)
    ]
    outputs = [
        Global("output", stream, 1, plan.columns, plan.columns, output_type)
        for stream in range(streams)
    ]
    node = literal_kernel_ast(single)
    node.decorator_list = []
    for arg in node.args.args:
        arg.annotation = None
    # These are explicit negative controls against the literal emitted code.
    text = ast.unparse(node)
    if mutation == "missing_barrier":
        text = text.replace("cute.arch.sync_threads()", "pass")
    elif mutation == "ascending_tree":
        text = text.replace("row_lanes >> level + 1", "1 << level")
    elif mutation == "missing_row_guard":
        text = text.replace("row < rows and column < columns", "column < columns")
    elif mutation == "missing_column_guard":
        text = text.replace("row < rows and column < columns", "row < rows")
    elif mutation == "cast_partial":
        text = text.replace(
            "+ values_left[lane]", "+ cutlass.BFloat16(values_left[lane])"
        )
    elif mutation == "reuse_left_output":
        text = text.replace(
            "right_out, column, final_right", "left_out, column, final_right"
        )
    elif mutation == "wrong_source":
        text = text.replace(
            "_cute_resident_load_vector(right,", "_cute_resident_load_vector(left,"
        )
    elif mutation is not None:
        raise AssertionError(mutation)
    if mutation:
        assert text != ast.unparse(node), mutation
    node = ast.parse(text).body[0]
    node = InterpreterTransform().visit(node)
    # Force a generator also for plans with no barrier or the negative control.
    node.body.append(
        ast.If(
            test=ast.Constant(False),
            body=[ast.Expr(value=ast.Yield(value=ast.Constant(None)))],
            orelse=[],
        )
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    context = {"tid": (0, 0, 0), "bid": (0, 0, 0), "shared": None}

    def allocate(dtype: object, size: int, alignment: int) -> Shared:
        assert dtype is float32 and alignment == 16
        if context["shared"] is None:
            context["shared"] = Shared(size)
        assert context["shared"].size == size
        return context["shared"]

    cute = SimpleNamespace(
        arch=SimpleNamespace(
            block_idx=lambda: context["bid"],
            thread_idx=lambda: context["tid"],
            alloc_smem=allocate,
        ),
        make_rmem_tensor=Register,
        make_tensor=lambda pointer, layout: pointer,
        make_layout=lambda shape: shape,
        ceil_div=lambda a, b: (a + b - 1) // b,
    )
    cutlass = SimpleNamespace(
        Int32=int,
        Float32=float32,
        BFloat16=cast_bf16,
        Constexpr=object,
        range_constexpr=range,
        const_expr=bool,
    )
    scope = {
        "cute": cute,
        "cutlass": cutlass,
        "_cute_resident_load_vector": load_vector,
        "_cute_resident_store_vector": store_vector,
    }
    exec(compile(module, "<literal paired tail interpreter>", "exec"), scope)
    kernel = scope[node.name]
    barriers = 0
    shared_bytes = 0
    for block in range(plan.grid[0]):
        context["bid"] = (block, 0, 0)
        context["shared"] = None
        threads = [
            (
                (tx, ty, 0),
                kernel(
                    *inputs,
                    *outputs,
                    plan.rows,
                    plan.columns,
                    plan.row_lanes,
                    plan.column_threads,
                    plan.vector,
                ),
            )
            for ty in range(plan.row_lanes)
            for tx in range(plan.column_threads)
        ]
        while threads:
            live = []
            for tid, thread in threads:
                context["tid"] = tid
                try:
                    assert next(thread) == "cta_barrier"
                    live.append((tid, thread))
                except StopIteration:
                    pass
            assert len(live) in (0, len(threads)), "divergent CTA barrier"
            if live:
                barriers += 1
            threads = live
        if context["shared"] is not None:
            shared_bytes = max(shared_bytes, context["shared"].size * 4)
    for stream, tensor in enumerate(inputs):
        assert len(tensor.loads) == plan.rows * plan.columns
        assert all(value == 1 for value in tensor.loads.values())
        out = outputs[stream]
        assert set(out.stores) == set(range(plan.columns))
        for column, value in out.stores.items():
            assert value == output_type(
                reference_tree(plan.rows, plan.row_lanes, stream, column)
            ), (stream, column, mutation)
    result = {
        "plan": asdict(plan),
        "stride": stride,
        "loads": sum(sum(t.loads.values()) for t in inputs),
        "stores": sum(len(t.stores) for t in outputs),
        "barriers": barriers,
        "shared_bytes": shared_bytes,
        "reference_tree_equal_per_output": True,
        "single_final_cast": True,
        "each_input_loaded_once": True,
        "every_output_written_once": True,
    }
    if return_expressions:
        result["expressions"] = [tensor.stores for tensor in outputs]
    return result


def reference_tree(rows: int, row_lanes: int, stream: int, column: int) -> Value:
    """Independent Reduce.cuh thread_reduce_impl + block_y_reduce transcription."""
    thread_values = []
    for ty in range(row_lanes):
        idx = ty
        stride = row_lanes
        values = [ZERO] * 4
        while idx + 3 * stride < rows:
            for slot in range(4):
                values[slot] = values[slot] + Value(
                    "input", (stream, idx + slot * stride, column)
                )
            idx += 4 * stride
        for slot in range(4):
            if idx >= rows:
                break
            values[slot] = values[slot] + Value("input", (stream, idx, column))
            idx += stride
        result = values[0]
        for slot in range(1, 4):
            result = result + values[slot]
        thread_values.append(result)
    offset = row_lanes // 2
    while offset:
        for ty in range(offset):
            thread_values[ty] = thread_values[ty] + thread_values[ty + offset]
        offset //= 2
    return thread_values[0]
