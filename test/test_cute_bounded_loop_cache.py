from __future__ import annotations

import ast
import struct
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from helion._compiler.cute.affine_vector_io import _Vectorizer
from helion._compiler.cute.bounded_loop_cache import attach_variant
from helion._compiler.cute.bounded_loop_cache import cache_single_tile_loads
from helion._compiler.cute.bounded_loop_cache import private_fragment_accesses
from helion._compiler.cute.bounded_loop_cache import specialize_single_tile_sweeps

SOURCE = """
def kernel(x, out, n, row):
    acc = cutlass.Float32(0)
    for tile in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(BLOCK)):
        for lane in range(2):
            base = tile + cutlass.Int32(lane) * 4
            for vec_lane_0 in cutlass.range_constexpr(4):
                column = base + cutlass.Int32(vec_lane_0)
                valid = column < n
                value = (x.iterator + cutlass.Int32(row) * cutlass.Int32(x.layout.stride[0]) + cutlass.Int32(column) * cutlass.Int32(x.layout.stride[1])).load() if valid else cutlass.Float32(0)
                acc = cutlass.Float32(acc + value)
    for tile in range(cutlass.Int32(0), cutlass.Int32(n), cutlass.Int32(BLOCK)):
        for other_lane in range(2):
            other_base = tile + cutlass.Int32(other_lane) * 4
            for vec_lane_1 in cutlass.range_constexpr(4):
                other_column = other_base + cutlass.Int32(vec_lane_1)
                other_valid = other_column < n
                other_value = (x.iterator + cutlass.Int32(row) * cutlass.Int32(x.layout.stride[0]) + cutlass.Int32(other_column) * cutlass.Int32(x.layout.stride[1])).load() if other_valid else cutlass.Float32(0)
                product = cutlass.Float32(other_value * acc)
                if other_valid:
                    (out.iterator + cutlass.Int32(row) * cutlass.Int32(out.layout.stride[0]) + cutlass.Int32(other_column) * cutlass.Int32(out.layout.stride[1])).store(product)
"""


def _prepare(source: str = SOURCE, *, disjoint: bool = True):
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    specialized = specialize_single_tile_sweeps(
        function.body,
        integer_arguments={"n", "row"},
        constexpr_values={"BLOCK": 8},
        launch_block=(1, 1, 1),
    )
    if specialized is None:
        return function, None, None
    cached = cache_single_tile_loads(
        specialized.body,
        specialized.plan,
        argument_names={"x", "out", "n", "row"},
        constexpr_values={"BLOCK": 8},
        tensor_dtypes={"x": "cutlass.Float32", "out": "cutlass.Float32"},
        proven_disjoint_tensor_pairs={frozenset({"x", "out"})} if disjoint else set(),
        rename_groups={},
    )
    return function, specialized, cached


def _i32(value):
    return np.int32((int(value) + (1 << 31)) % (1 << 32) - (1 << 31))


def _cuda_range(*values):
    dtype = np.int64 if any(isinstance(value, np.int64) for value in values) else _i32
    return [dtype(value) for value in range(*(int(value) for value in values))]


class _Fragment:
    def __init__(self, size, dtype):
        self.size = size
        self.dtype = dtype
        self.values = {}

    def __getitem__(self, index):
        assert 0 <= index < self.size
        assert int(index) in self.values, "uninitialized cache read"
        return self.values[int(index)]

    def __setitem__(self, index, value):
        assert 0 <= index < self.size
        self.values[int(index)] = self.dtype(value)


class _Pointer:
    def __init__(self, owner, offset=0):
        self.owner = owner
        self.offset = int(offset)

    def __add__(self, value):
        return _Pointer(self.owner, self.offset + int(value))

    def load(self):
        assert self.offset in self.owner.values, "invalid global read"
        self.owner.reads += 1
        return self.owner.values[self.offset]

    def store(self, value):
        assert self.offset in self.owner.values, "invalid global write"
        self.owner.writes += 1
        self.owner.values[self.offset] = np.float32(value)


class _Tensor:
    def __init__(self, values, stride):
        self.values = dict(values)
        self.layout = SimpleNamespace(stride=stride)
        self.iterator = _Pointer(self)
        self.reads = 0
        self.writes = 0


def _execute(function, n, row, stride, values):
    with np.errstate(over="ignore"):
        offsets = [
            int(_i32(row) * _i32(stride[0])) + int(_i32(i) * _i32(stride[1]))
            for i in range(max(0, n))
        ]
    x = _Tensor(zip(offsets, values, strict=True), stride)
    out = _Tensor(zip(offsets, [np.float32(-17)] * len(offsets), strict=True), stride)
    namespace: dict[str, Any] = {
        "BLOCK": 8,
        "range": _cuda_range,
        "cutlass": SimpleNamespace(
            Int32=_i32, Int64=np.int64, Float32=np.float32, range_constexpr=range
        ),
        "cute": SimpleNamespace(make_rmem_tensor=_Fragment),
    }
    code = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(code, "<bounded-cache-test>", "exec"), namespace)
    with np.errstate(all="ignore"):
        namespace[function.name](x, out, n, row)
    return (
        b"".join(np.float32(out.values[index]).tobytes() for index in offsets),
        x.reads,
        out.writes,
    )


@pytest.mark.parametrize("n", [-1, 0, 1, 3, 7, 8, 9, 17])
@pytest.mark.parametrize(
    "stride,row", [((32, 1), 0), ((37, 2), 1), ((1, 5), 2), ((8, 1), 268435456)]
)
@pytest.mark.parametrize("nonfinite", [False, True])
def test_guarded_variant_preserves_exact_values_accesses_and_typed_offsets(
    n, stride, row, nonfinite
):
    original, specialized, cached = _prepare()
    assert specialized is not None and cached is not None
    assert cached.replaced_loads == 1
    assert private_fragment_accesses(cached.body, cached.fragments)
    fast = ast.FunctionDef(
        name="kernel", args=original.args, body=cached.body, decorator_list=[]
    )
    pattern = [np.float32(1.25), np.float32(-0.0), np.float32(-0.125), np.float32(3)]
    if nonfinite:
        pattern = [
            np.float32(np.nan),
            np.float32(np.inf),
            np.float32(-np.inf),
            np.float32(-0.0),
        ]
    values = [pattern[index % len(pattern)] for index in range(max(0, n))]
    expected = _execute(original, n, row, stride, values)
    selected = fast if 0 < n <= specialized.plan.extent.upper else original
    actual = _execute(selected, n, row, stride, values)
    assert actual[0] == expected[0]
    assert actual[2] == expected[2] == max(0, n)
    assert actual[1] == (max(0, n) if 0 < n <= 8 else expected[1])


@pytest.mark.parametrize(
    "old,new",
    [
        ("cutlass.Int32(0), cutlass.Int32(n)", "cutlass.Int32(1), cutlass.Int32(n)"),
        ("cutlass.Int32(n)", "cutlass.Int32(n + 1)"),
        ("cutlass.Int32(n)", "cutlass.Int16(n)"),
        ("cutlass.Int32(BLOCK)", "cutlass.Int32(-8)"),
        ("acc = cutlass.Float32(0)", "n = n - 1\n    acc = cutlass.Float32(0)"),
        ("acc = cutlass.Float32(0)", "cutlass = x\n    acc = cutlass.Float32(0)"),
    ],
)
def test_unproved_bounds_keep_original(old, new):
    original = SOURCE.replace(old, new)
    function, specialized, cached = _prepare(original)
    assert specialized is None and cached is None
    assert ast.dump(function) == ast.dump(ast.parse(original).body[0])


@pytest.mark.parametrize(
    "replacement",
    [
        "other_base + cutlass.Int32(vec_lane_1) + 1",
        "other_base + cutlass.Int32(vec_lane_1) - 1",
    ],
)
def test_different_address_does_not_reuse(replacement):
    _, specialized, cached = _prepare(
        SOURCE.replace("other_base + cutlass.Int32(vec_lane_1)", replacement)
    )
    assert specialized is not None and cached is None


@pytest.mark.parametrize(
    "addition",
    [
        "unknown_effect()",
        "cute.arch.sync_threads()",
        "cute.arch.atomic_add(out.iterator.llvm_ptr, val=1, sem='release')",
        "(x.iterator + cutlass.Int32(0)).store(cutlass.Float32(9))",
        "x[0] = cutlass.Float32(9)",
        "alias = x\n    alias[0] = cutlass.Float32(9)",
    ],
)
def test_unknown_and_aliasing_effects_decline(addition):
    source = SOURCE.replace("    for tile", f"    {addition}\n    for tile", 1)
    _, specialized, cached = _prepare(source)
    assert cached is None


def test_argument_names_are_not_a_disjointness_proof():
    _, specialized, cached = _prepare(disjoint=False)
    assert specialized is not None and cached is None


def test_conditional_definition_is_not_a_dominating_value():
    source = SOURCE.replace("    acc =", "    shift = cutlass.Int32(0)\n    acc =", 1)
    source = source.replace(
        "other_base = tile",
        "if row > 0:\n                shift = cutlass.Int32(1)\n            other_base = shift + tile",
    )
    _, specialized, cached = _prepare(source)
    assert specialized is not None and cached is None


@pytest.mark.parametrize(
    "mutation", ["missing", "escape", "wrong_size", "late", "wrong_index"]
)
def test_private_memory_proof_does_not_allow_arbitrary_subscripts(mutation):
    _, _, cached = _prepare()
    assert cached is not None
    body = ast.parse(ast.unparse(ast.Module(body=cached.body, type_ignores=[]))).body
    name = cached.fragments[0].name
    if mutation == "missing":
        body.pop(0)
    elif mutation == "late":
        body.append(body.pop(0))
    elif mutation == "escape":
        body.append(ast.parse(f"escaped = {name}").body[0])
    elif mutation == "wrong_size":
        body[0] = ast.parse(f"{name} = cute.make_rmem_tensor(1, cutlass.Float32)").body[
            0
        ]
    else:
        subscript = next(
            node
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == name
        )
        subscript.slice = ast.Constant(9)
    assert not private_fragment_accesses(body, cached.fragments)


def test_host_selector_preserves_fallback_and_exact_launch_abi():
    original, specialized, cached = _prepare()
    assert specialized is not None and cached is not None
    header = "BLOCK = 8\n"
    wrapper = "\ndef wrapper(x, out, n, row, _launcher):\n    _launcher(kernel, (1,), x, out, n, row, block=(1, 1, 1))\n"
    control = header + ast.unparse(original) + wrapper
    fast_function = ast.FunctionDef(
        name="kernel", args=original.args, body=cached.body, decorator_list=[]
    )
    fast = header + ast.unparse(ast.fix_missing_locations(fast_function)) + wrapper
    merged = attach_variant(control, fast, specialized.plan, kernel_name="kernel")
    assert merged is not None
    module = ast.parse(merged)
    fallback = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "kernel"
    )
    assert ast.dump(fallback) == ast.dump(original)
    launch = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_launcher"
    )
    assert isinstance(launch.args[0], ast.IfExp)
    assert ast.unparse(launch.args[0].test) == "0 < n <= 8"
    assert (
        attach_variant(
            control.replace("block=(1, 1, 1)", "block=(2, 1, 1)"),
            fast,
            specialized.plan,
            kernel_name="kernel",
        )
        is None
    )


def test_expansion_budget_declines_without_omitting_dependencies():
    prefix = "    hidden0 = row\n"
    for index in range(1, 14):
        prefix += f"    hidden{index} = hidden{index - 1} * hidden{index - 1}\n"
    source = SOURCE.replace("    acc =", prefix + "    acc =", 1).replace(
        "cutlass.Int32(row)", "cutlass.Int32(hidden13)"
    )
    _, specialized, cached = _prepare(source)
    assert specialized is not None and cached is None


class _F32(float):
    def __new__(cls, value):
        return float.__new__(cls, np.float32(value))

    def bitcast(self, dtype):
        assert dtype is _U32
        return _U32(struct.unpack("I", struct.pack("f", self))[0])


class _U32(int):
    mlir_type = "i32"

    def bitcast(self, dtype):
        assert dtype is _F32
        return _F32(struct.unpack("f", struct.pack("I", self))[0])


def _execute_vector_body(original, body, n, *, stride=1):
    stats = {"vector_loads": 0, "vector_stores": 0}
    offsets = [index * stride for index in range(n)]
    data = [np.float32((index - 3) * 0.125) for index in range(n)]
    source = _Tensor(zip(offsets, data, strict=True), (32, stride))
    output = _Tensor(zip(offsets, [-17.0] * n, strict=True), (32, stride))

    def vector_load(pointer, vector_type):
        stats["vector_loads"] += 1
        assert pointer.offset % 4 == 0
        width = vector_type[0][0]
        return [_F32((pointer + index).load()).bitcast(_U32) for index in range(width)]

    def vector_store(pointer, values):
        stats["vector_stores"] += 1
        assert pointer.offset % 4 == 0
        for index, value in enumerate(values):
            (pointer + index).store(value.bitcast(_F32))

    namespace: dict[str, Any] = {
        "BLOCK": 8,
        "range": _cuda_range,
        "cutlass": SimpleNamespace(
            Int32=_i32,
            Int64=np.int64,
            Float32=_F32,
            Uint32=_U32,
            Boolean=bool,
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(
            make_rmem_tensor=_Fragment, arch=SimpleNamespace(load=vector_load)
        ),
        "ir": SimpleNamespace(
            VectorType=SimpleNamespace(get=lambda shape, dtype: (shape, dtype))
        ),
        "_cute_store_u32_vec": vector_store,
    }
    function = ast.FunctionDef(
        name="kernel", args=original.args, body=body, decorator_list=[]
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, "<bounded-vector-test>", "exec"), namespace)
    namespace["kernel"](source, output, n, 0)
    return b"".join(
        np.float32(output.values[offset]).tobytes() for offset in offsets
    ), stats


@pytest.mark.parametrize("n", [1, 3, 4, 5, 7, 8])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("aligned", [False, True])
def test_owned_cache_composes_with_vector_packets_and_scalar_tails(n, stride, aligned):
    original, _, cached = _prepare()
    assert cached is not None
    vectorizer = _Vectorizer(
        cached.body,
        strides={
            (name, axis): value
            for name in ("x", "out")
            for axis, value in enumerate((32, stride))
        },
        alignments={"x": 16 if aligned else 1, "out": 16},
        dtypes={"x": "cutlass.Float32", "out": "cutlass.Float32"},
        disjoint={frozenset({"x", "out"})},
        constexpr={"BLOCK": 8},
        private_accesses=private_fragment_accesses(cached.body, cached.fragments),
    )
    body = vectorizer.body(cached.body)
    expected, _ = _execute_vector_body(original, original.body, n, stride=stride)
    actual, stats = _execute_vector_body(original, body, n, stride=stride)
    assert actual == expected
    assert stats["vector_loads"] == (n // 4 if stride == 1 and aligned else 0)
    assert stats["vector_stores"] == (n // 4 if stride == 1 else 0)


def test_private_cache_does_not_make_unknown_memory_accesses_vectorizable():
    original, _, cached = _prepare()
    assert cached is not None
    body = ast.parse(ast.unparse(ast.Module(body=cached.body, type_ignores=[]))).body
    loop = next(
        node
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.For) and ast.unparse(node.target) == "vec_lane_0"
    )
    loop.body.append(ast.parse("opaque = out[0]").body[0])
    vectorizer = _Vectorizer(
        body,
        strides={("x", 0): 32, ("x", 1): 1, ("out", 0): 32, ("out", 1): 1},
        alignments={"x": 16, "out": 16},
        dtypes={"x": "cutlass.Float32", "out": "cutlass.Float32"},
        disjoint={frozenset({"x", "out"})},
        constexpr={"BLOCK": 8},
        private_accesses=private_fragment_accesses(body, cached.fragments),
    )
    rewritten = ast.unparse(
        ast.fix_missing_locations(
            ast.Module(body=vectorizer.body(body), type_ignores=[])
        )
    )
    assert "opaque = out[0]" in rewritten
    assert "cute.arch.load(" not in rewritten


@pytest.mark.parametrize("integer_type", ["Int32", "Int64"])
def test_runtime_lane_counter_preserves_original_integer_width(integer_type):
    source = SOURCE.replace("range(2)", f"range(cutlass.{integer_type}(2))")
    _, specialized, cached = _prepare(source)
    assert specialized is not None and cached is not None
    code = ast.unparse(ast.Module(body=cached.body, type_ignores=[]))
    assert f"lane = cutlass.{integer_type}(_helion_cache_lane)" in code
    assert f"other_lane = cutlass.{integer_type}(_helion_cache_lane_)" in code


def test_different_original_lane_widths_do_not_share_a_cache():
    source = SOURCE.replace(
        "other_lane in range(2)", "other_lane in range(cutlass.Int64(2))"
    )
    _, specialized, cached = _prepare(source)
    assert specialized is not None and cached is None


def test_int64_lane_multiplication_does_not_acquire_int32_wraparound():
    source = (
        SOURCE.replace("range(2)", "range(cutlass.Int64(3))")
        .replace("cutlass.range_constexpr(4)", "cutlass.range_constexpr(1)")
        .replace("tile + cutlass.Int32(lane) * 4", "lane * cutlass.Int32(2147483647)")
        .replace(
            "tile + cutlass.Int32(other_lane) * 4",
            "other_lane * cutlass.Int32(2147483647)",
        )
    )
    original, specialized, cached = _prepare(source)
    assert specialized is not None and cached is not None
    fast = ast.FunctionDef(
        name="kernel", args=original.args, body=cached.body, decorator_list=[]
    )
    values = [np.float32(value + 1) for value in range(8)]
    expected = _execute(original, 8, 0, (8, 1), values)
    actual = _execute(fast, 8, 0, (8, 1), values)
    assert actual[0] == expected[0]
    assert actual[1:] == (1, 1)
    # The formerly tempting Int32 restoration would wrap 2 * 2147483647 to
    # -2, changing an inactive positive coordinate into an invalid global read.
    wrong = ast.parse(
        ast.unparse(fast).replace(
            "cutlass.Int64(_helion_cache_lane", "cutlass.Int32(_helion_cache_lane"
        )
    ).body[0]
    with pytest.raises(AssertionError, match="invalid global read"):
        _execute(wrong, 8, 0, (8, 1), values)


@pytest.mark.parametrize(
    "control_prefix,fast_prefix,wrapper_argument",
    [
        ("import first as hidden\n", "import second as hidden\n", "n"),
        ("from first import *\n", "from first import *\n", "n"),
        ("", "", "resolve_n(n)"),
    ],
)
def test_host_dispatch_declines_import_collision_or_repeated_argument_evaluation(
    control_prefix, fast_prefix, wrapper_argument
):
    original, specialized, cached = _prepare()
    assert specialized is not None and cached is not None
    wrapper = f"\ndef wrapper(x, out, n, row, _launcher):\n    _launcher(kernel, (1,), x, out, {wrapper_argument}, row, block=(1, 1, 1))\n"
    fast = ast.FunctionDef(
        name="kernel", args=original.args, body=cached.body, decorator_list=[]
    )
    assert (
        attach_variant(
            control_prefix + ast.unparse(original) + wrapper,
            fast_prefix + ast.unparse(ast.fix_missing_locations(fast)) + wrapper,
            specialized.plan,
            kernel_name="kernel",
        )
        is None
    )
