"""CPU value and memory oracles for collective store repartitioning."""

from __future__ import annotations

import ast
from collections import Counter
import dataclasses
import itertools
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.collective_vector_epilogue import _load_pointer_roots
from helion._compiler.cute.collective_vector_epilogue import (
    vectorize_collective_store_epilogue,
)
from helion._compiler.cute.collective_vector_epilogue import (
    vectorize_collective_store_epilogues,
)
from helion._compiler.cute.contiguous_copy import CopyTensorFacts
from helion._compiler.device_function import TensorArg

if TYPE_CHECKING:
    from helion._compiler.cute.collective_matmul import CollectiveMmaSite
    from helion._compiler.device_function import DeviceFunction

_SOURCE = """
for lane in range(16):
    mi = mo + cutlass.Int32(cute.arch.thread_idx()[0]) * 16 + cutlass.Int32(lane)
    valid = mi < rows and ni < columns
    acc = shared[mi - mo, ni - no]
    bias_value = (bias.iterator + ni).load() if valid else cutlass.Float32(0)
    value = cutlass.Float16(acc + bias_value)
    if valid:
        (out.iterator + cutlass.Int32(mi) * cutlass.Int32(out.layout.stride[0]) + cutlass.Int32(ni)).store(value)
"""


def _plan(
    source: str = _SOURCE,
    *,
    dtype: str = "cutlass.Float16",
    after: str = "",
    dominating: str = "",
    disjoint: bool = True,
    output_specialized: bool = False,
    unroll: bool = False,
) -> ast.stmt | None:
    loop = cast("ast.For", ast.parse(source).body[0])
    original = ast.dump(loop)
    counter = itertools.count()
    output = CopyTensorFacts(dtype, (64, 1), 16)
    tensors = {"bias": CopyTensorFacts("cutlass.Float32", (1,), 16)}
    if output_specialized:
        tensors["out"] = output
    result = vectorize_collective_store_epilogue(
        loop,
        m_index="mi",
        n_index="ni",
        m_offset="mo",
        n_offset="no",
        block_m=32,
        block_n=64,
        thread_index="tid",
        thread_dimensions=(2, 64, 1),
        shared_results=frozenset({"shared"}),
        tensors=tensors,
        output_layouts={"out": output},
        tensor_names=frozenset({"out", "bias", "aux"}),
        disjoint_pairs={frozenset({"out", "bias"})} if disjoint else set(),
        boundary_names={"out", "bias", "aux", "rows", "columns", "saved"},
        dominating=ast.parse(dominating).body,
        live_after=ast.parse(after).body,
        unroll=unroll,
        fresh_name=lambda hint: f"v{next(counter)}_{hint}",
    )
    assert ast.dump(loop) == original
    return result


def _bf16(value: float) -> np.float32:
    rounded = np.asarray(value, dtype=np.float32)
    bits = int(rounded.view(np.uint32))
    if bits & 0x7F800000 != 0x7F800000:
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
    return np.float32(
        np.asarray(bits & 0xFFFF0000, dtype=np.uint32).view(np.float32)[()]
    )


_CASTS = {
    "cutlass.Float16": (np.float16, 2),
    "cutlass.BFloat16": (_bf16, 2),
    "cutlass.Float32": (np.float32, 4),
}


@dataclasses.dataclass
class _Pointer:
    tensor: _Tensor
    offset: int = 0

    def __add__(self, value: int) -> _Pointer:
        return _Pointer(self.tensor, self.offset + int(value))

    def toint(self) -> int:
        return self.tensor.base + self.offset * self.tensor.itemsize

    def load(self) -> np.float32:
        assert 0 <= self.offset < self.tensor.values.size, (
            "out-of-bounds read",
            self.tensor.name,
            self.offset,
        )
        self.tensor.loads.append(self.offset)
        return np.float32(self.tensor.values[self.offset])

    def store(self, value: float) -> None:
        assert 0 <= self.offset < self.tensor.values.size, (
            "out-of-bounds write",
            self.tensor.name,
            self.offset,
        )
        self.tensor.writes.append(self.offset)
        self.tensor.values[self.offset] = self.tensor.element_type(value)


class _Tensor:
    def __init__(
        self,
        name: str,
        values: np.ndarray,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        *,
        dtype: str,
        base: int,
    ) -> None:
        self.name = name
        self.values = values.copy()
        self.layout = SimpleNamespace(shape=shape, stride=strides)
        self.element_type, self.itemsize = _CASTS[dtype]
        self.base = base
        self.iterator = _Pointer(self)
        self.loads: list[int] = []
        self.writes: list[int] = []

    def __getitem__(self, index: int) -> np.float32:
        return (self.iterator + index).load()

    def __setitem__(self, index: int, value: float) -> None:
        (self.iterator + index).store(value)


def _run(
    statement: ast.stmt,
    *,
    rows: int,
    columns: int,
    stride: int,
    base_offset: int,
    dtype: str,
    after: str = "",
    before: str = "",
    reverse_threads: bool = False,
    inner_stride: int = 1,
) -> tuple[np.ndarray, Counter[int], set[int], Counter[str], list[dict[str, object]]]:
    output = _Tensor(
        "out",
        np.full(rows * stride, -777.0, dtype=np.float32),
        (rows, columns),
        (stride, inner_stride),
        dtype=dtype,
        base=0x100000 + base_offset,
    )
    bias = _Tensor(
        "bias",
        np.arange(columns, dtype=np.float32) * np.float32(0.013),
        (columns,),
        (1,),
        dtype="cutlass.Float32",
        base=0x200000,
    )
    counts: Counter[str] = Counter()
    tensors = [output, bias]

    def make_layout(shape, stride=None):
        assert len(shape) == 1
        return SimpleNamespace(shape=shape, stride=(1,) if stride is None else stride)

    def make_ptr(element_type, address, space, *, assumed_align):
        assert space == "gmem" and assumed_align == 16 and address % 16 == 0
        for tensor in tensors:
            if (
                tensor.base
                <= address
                < tensor.base + tensor.values.size * tensor.itemsize
            ):
                assert element_type is tensor.element_type
                assert (address - tensor.base) % tensor.itemsize == 0
                return _Pointer(tensor, (address - tensor.base) // tensor.itemsize)
        raise AssertionError(("invalid vector address", address))

    def make_registers(layout, element_type):
        dtype_name = next(
            name for name, (fn, size) in _CASTS.items() if fn is element_type
        )
        return _Tensor(
            "registers",
            np.full(layout.shape[0], np.nan, dtype=np.float32),
            layout.shape,
            layout.stride,
            dtype=dtype_name,
            base=0,
        )

    def copy(source, destination):
        assert source.layout.shape == destination.layout.shape
        width = source.layout.shape[0]
        counts[
            "vector_store" if destination.iterator.tensor is output else "vector_load"
        ] += 1
        for index in range(width):
            (destination.iterator + index).store((source.iterator + index).load())

    thread = [0, 0, 0]
    namespace = {
        "out": output,
        "bias": bias,
        "rows": rows,
        "columns": columns,
        "saved": np.float32(3),
        "cutlass": SimpleNamespace(
            Int32=lambda value: (int(value) + 2**31) % 2**32 - 2**31,
            Int64=lambda value: (int(value) + 2**63) % 2**64 - 2**63,
            Float16=np.float16,
            BFloat16=_bf16,
            Float32=np.float32,
            range=lambda *args, **kwargs: range(*args),
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(thread_idx=lambda: thread),
            make_layout=make_layout,
            make_ptr=make_ptr,
            make_tensor=lambda pointer, layout: SimpleNamespace(
                iterator=pointer, layout=layout
            ),
            make_rmem_tensor=make_registers,
            autovec_copy=copy,
            AddressSpace=SimpleNamespace(gmem="gmem"),
        ),
    }
    program = [*ast.parse(before).body, statement, *ast.parse(after).body]
    code = compile(
        ast.fix_missing_locations(ast.Module(body=program, type_ignores=[])),
        "<vector-epilogue-oracle>",
        "exec",
    )
    liveouts: list[dict[str, object]] = []
    for mo in range(0, rows, 32):
        for no in range(0, columns, 64):
            shared = np.arange(32 * 64, dtype=np.float32).reshape(32, 64) * np.float32(
                0.019
            )
            for tid in reversed(range(128)) if reverse_threads else range(128):
                thread[0], thread[1] = tid % 2, tid // 2
                local = dict(
                    namespace, mo=mo, no=no, shared=shared, tid=tid, ni=no + thread[1]
                )
                exec(code, local)
                liveouts.append(
                    {key: local[key] for key in ("observed",) if key in local}
                )
    return output.values, Counter(output.writes), set(bias.loads), counts, liveouts


@pytest.mark.parametrize("dtype", list(_CASTS))
@pytest.mark.parametrize("unroll", [False, True])
@pytest.mark.parametrize(
    "rows,columns,stride,base_offset",
    [(32, 64, 64, 0), (31, 63, 64, 0), (1, 7, 64, 0), (33, 71, 79, 0), (32, 64, 64, 2)],
)
def test_store_values_addresses_masks_and_runtime_fallback(
    dtype: str, unroll: bool, rows: int, columns: int, stride: int, base_offset: int
) -> None:
    source = _SOURCE.replace("cutlass.Float16", dtype)
    planned = _plan(source, dtype=dtype, unroll=unroll)
    assert planned is not None
    arguments = {
        "rows": rows,
        "columns": columns,
        "stride": stride,
        "base_offset": base_offset,
        "dtype": dtype,
    }
    expected, writes, reads, _, _ = _run(ast.parse(source).body[0], **arguments)
    actual, actual_writes, actual_reads, counts, _ = _run(planned, **arguments)
    np.testing.assert_array_equal(actual, expected)
    assert writes == actual_writes and len(writes) == rows * columns
    assert set(writes.values()) == {1}
    assert reads == actual_reads
    width = 4 if dtype == "cutlass.Float32" else 8
    if stride == 64 and base_offset == 0 and columns >= width:
        assert counts["vector_store"] > 0
    else:
        assert counts["vector_store"] == 0


def test_inline_store_value_remains_under_original_mask() -> None:
    source = _SOURCE.replace(
        "    bias_value = (bias.iterator + ni).load() if valid else cutlass.Float32(0)\n"
        "    value = cutlass.Float16(acc + bias_value)\n",
        "",
    ).replace(
        ".store(value)", ".store(cutlass.Float16(acc + (bias.iterator + ni).load()))"
    )
    arguments = {
        "rows": 31,
        "columns": 7,
        "stride": 64,
        "base_offset": 0,
        "dtype": "cutlass.Float16",
    }
    expected, writes, reads, _, _ = _run(ast.parse(source).body[0], **arguments)
    planned = _plan(source)
    if planned is not None:
        actual, actual_writes, actual_reads, _, _ = _run(planned, **arguments)
        np.testing.assert_array_equal(actual, expected)
        assert actual_writes == writes and actual_reads == reads


def test_loop_target_liveout_keeps_original_binding() -> None:
    before = "lane = -1"
    after = "observed = lane"
    planned = _plan(after=after)
    arguments = {
        "rows": 32,
        "columns": 64,
        "stride": 64,
        "base_offset": 0,
        "dtype": "cutlass.Float16",
        "before": before,
        "after": after,
    }
    expected, _, _, _, expected_liveouts = _run(ast.parse(_SOURCE).body[0], **arguments)
    if planned is not None:
        actual, _, _, _, actual_liveouts = _run(planned, **arguments)
        np.testing.assert_array_equal(actual, expected)
        assert actual_liveouts == expected_liveouts


def _rewrite_program(program: list[ast.stmt], loop: ast.For) -> None:
    counter = itertools.count()
    environment = SimpleNamespace(
        backend=SimpleNamespace(
            dtype_str=lambda dtype: (
                "cutlass.Float16" if dtype == torch.float16 else "cutlass.Float32"
            )
        ),
        size_hint=int,
    )
    function = SimpleNamespace(
        arguments=[
            TensorArg("out", torch.empty((32, 64), dtype=torch.float16), "out"),
            TensorArg("bias", torch.empty((64,), dtype=torch.float32), "bias"),
        ],
        config={"cute_collective_epilogue": "vector"},
        new_var=lambda hint: f"r{next(counter)}_{hint}",
    )
    site = SimpleNamespace(
        grid_row_lane="lane",
        m_axis=0,
        n_axis=1,
        m_index="mi",
        n_index="ni",
        m_offset="mo",
        n_offset="no",
        bm=32,
        bn=64,
    )
    with (
        patch.object(CompileEnvironment, "current", return_value=environment),
        patch(
            "helion._compiler.cute.collective_matmul._copy_tensor_facts",
            return_value={"bias": CopyTensorFacts("cutlass.Float32", (1,), 16)},
        ),
    ):
        vectorize_collective_store_epilogues(
            program,
            cast("DeviceFunction", function),
            [(cast("CollectiveMmaSite", site), loop, "tid", "shared")],
            boundary_names={"out", "bias", "rows", "columns", "saved"},
            disjoint_pairs={frozenset({"out", "bias"})},
        )


def test_enclosing_loop_preserves_value_used_on_next_iteration() -> None:
    source = _SOURCE.replace(
        "    value = cutlass.Float16(acc + bias_value)",
        "    carry = cutlass.Float16(acc + bias_value)",
    ).replace(".store(value)", ".store(carry)")
    outer = cast(
        "ast.For",
        ast.parse(
            "for repeat in range(2):\n"
            "    observed = carry\n"
            + "\n".join("    " + line for line in source.strip().splitlines())
        ).body[0],
    )
    original = ast.parse(ast.unparse(outer)).body[0]
    loop = cast("ast.For", outer.body[-1])
    program: list[ast.stmt] = [outer]
    _rewrite_program(program, loop)
    arguments = {
        "rows": 32,
        "columns": 64,
        "stride": 64,
        "base_offset": 0,
        "dtype": "cutlass.Float16",
        "before": "carry = cutlass.Float16(-1)",
    }
    expected, writes, reads, _, before_liveouts = _run(original, **arguments)
    actual, actual_writes, actual_reads, _, after_liveouts = _run(
        program[0], **arguments
    )
    np.testing.assert_array_equal(actual, expected)
    assert writes == actual_writes and reads == actual_reads
    assert before_liveouts == after_liveouts


@pytest.mark.parametrize("kind", ["condition", "for_else"])
def test_enclosing_control_observes_original_epilogue_binding(kind: str) -> None:
    source = _SOURCE.replace(
        "    value = cutlass.Float16(acc + bias_value)",
        "    carry = cutlass.Float16(-(acc + bias_value + 1))",
    ).replace(".store(value)", ".store(carry)")
    if kind == "condition":
        text = "for repeat in range(2):\n    if carry > 0:\n" + "\n".join(
            "        " + line for line in source.strip().splitlines()
        )
    else:
        text = (
            "for repeat in range(2):\n"
            + "\n".join("    " + line for line in source.strip().splitlines())
            + "\nelse:\n    observed = carry"
        )
    program = ast.parse(text).body
    original = ast.parse(text).body[0]
    loop = next(
        node
        for node in ast.walk(program[0])
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "lane"
    )
    _rewrite_program(program, loop)
    arguments = {
        "rows": 32,
        "columns": 64,
        "stride": 64,
        "base_offset": 0,
        "dtype": "cutlass.Float16",
        "before": "carry = cutlass.Float16(1)",
    }
    expected, writes, reads, _, before_liveouts = _run(original, **arguments)
    actual, actual_writes, actual_reads, _, after_liveouts = _run(
        program[0], **arguments
    )
    np.testing.assert_array_equal(actual, expected)
    assert writes == actual_writes and reads == actual_reads
    assert before_liveouts == after_liveouts


def test_stateless_enclosing_loop_still_vectorizes() -> None:
    text = "for repeat in range(2):\n" + "\n".join(
        "    " + line for line in _SOURCE.strip().splitlines()
    )
    program = ast.parse(text).body
    original = ast.parse(text).body[0]
    _rewrite_program(program, cast("ast.For", cast("ast.For", program[0]).body[0]))
    assert "epilogue_vector_slot" in ast.unparse(
        ast.Module(body=program, type_ignores=[])
    )
    arguments = {
        "rows": 32,
        "columns": 64,
        "stride": 64,
        "base_offset": 0,
        "dtype": "cutlass.Float16",
    }
    expected, writes, reads, _, _ = _run(original, **arguments)
    actual, actual_writes, actual_reads, counts, _ = _run(program[0], **arguments)
    np.testing.assert_array_equal(actual, expected)
    assert writes == actual_writes and reads == actual_reads
    assert set(writes.values()) == {2} and counts["vector_store"] > 0


@pytest.mark.parametrize(
    "old,new",
    [
        ("    acc = shared", "    unused = unknown_effect()\n    acc = shared"),
        ("    acc = shared", "    cute.arch.sync_threads()\n    acc = shared"),
        (
            "    acc = shared",
            "    (aux.iterator + ni).store(cutlass.Float32(1))\n    acc = shared",
        ),
        (
            "    acc = shared",
            "    cute.arch.atomic_add((aux.iterator + ni).llvm_ptr, val=1, sem='relaxed')\n    acc = shared",
        ),
        ("    value =", "    mi = mi + 1\n    value ="),
        ("    value =", "    ni = ni + 1\n    value ="),
        ("    value =", "    mo = 1\n    value ="),
        ("    value =", "    out = bias\n    value ="),
        ("    value =", "    cutlass = saved\n    value ="),
        ("acc + bias_value", "acc + cutlass.Float32(cute.arch.thread_idx()[0])"),
        ("if valid:", "if valid and cute.arch.thread_idx()[0] < 1:"),
        ("if valid:", "if valid and ni % 3 != 0:"),
    ],
)
def test_effects_coordinates_rebinding_and_mask_holes_decline(
    old: str, new: str
) -> None:
    assert _plan(_SOURCE.replace(old, new)) is None


@pytest.mark.parametrize(
    "source,dominating",
    [
        (_SOURCE, ""),
        (_SOURCE.replace("(bias.iterator + ni).load()", "bias[ni]"), ""),
        (
            _SOURCE.replace("if valid:", "if valid and captured > 0:")
            .replace(
                "    bias_value = (bias.iterator + ni).load() if valid else cutlass.Float32(0)\n",
                "",
            )
            .replace("acc + bias_value", "acc"),
            "captured = (bias.iterator + 0).load()",
        ),
    ],
)
def test_replayed_value_and_mask_reads_require_disjoint_storage(
    source: str, dominating: str
) -> None:
    assert _plan(source, dominating=dominating, disjoint=False) is None
    assert _plan(source, dominating=dominating, disjoint=True) is not None


@pytest.mark.parametrize(
    "source,dominating",
    [
        (_SOURCE.replace("(bias.iterator + ni).load()", "alias[ni]"), "alias = bias"),
        (
            _SOURCE.replace(
                "(bias.iterator + ni).load()", "(out.iterator + ni).load()"
            ),
            "",
        ),
        (
            _SOURCE.replace(
                "if valid:", "if valid and (out.iterator + ni).load() > 0:"
            ),
            "",
        ),
    ],
)
def test_unclassified_alias_and_output_reads_decline(
    source: str, dominating: str
) -> None:
    assert _plan(source, dominating=dominating) is None


@pytest.mark.parametrize(
    "dominating",
    [
        "acc = cutlass.Float32(1)\ncaptured = acc",
        "acc = cutlass.Float32(1)\nfirst = acc\ncaptured = first",
        "mi = 1\ncaptured = cutlass.Float32(mi)",
    ],
)
def test_rebound_dominating_capture_declines(dominating: str) -> None:
    source = _SOURCE.replace("acc + bias_value", "acc + captured")
    assert _plan(source, dominating=dominating) is None


@pytest.mark.parametrize("name", ["acc", "valid", "value", "mi"])
def test_existing_scalar_liveouts_decline(name: str) -> None:
    assert _plan(after=f"observed = {name}") is None


@pytest.mark.parametrize("unroll", [False, True])
@pytest.mark.parametrize(
    "predicate",
    [
        "mi < rows and 2 <= ni and ni < 6",
        "False",
        "cutlass.Float32(2.5) if mi < rows else cutlass.Float32(-0.0)",
        "cutlass.Int32(cute.arch.thread_idx()[0]) < 2 and mi < rows and cutlass.Int64(cute.arch.thread_idx()[1]) < 64 and ni < columns",
    ],
)
def test_interval_numeric_and_full_thread_masks_preserve_values(
    predicate: str, unroll: bool
) -> None:
    source = _SOURCE.replace("mi < rows and ni < columns", predicate).replace(
        "acc + bias_value", "acc + bias_value + valid"
    )
    planned = _plan(source, unroll=unroll)
    assert planned is not None
    arguments = {
        "rows": 3,
        "columns": 64,
        "stride": 64,
        "base_offset": 0,
        "dtype": "cutlass.Float16",
    }
    expected, writes, reads, _, _ = _run(ast.parse(source).body[0], **arguments)
    actual, actual_writes, actual_reads, _, _ = _run(planned, **arguments)
    np.testing.assert_array_equal(actual, expected)
    assert actual_writes == writes and actual_reads == reads


def test_conditional_pointer_alias_preserves_ordered_same_thread_reads() -> None:
    # Only the first physical M thread writes: its original row-lane loop
    # performs the recurrence in order, independently for every column. A
    # repartition moves dependent rows to different threads. Use the reverse
    # thread order as an allowed interleaving, with the same order on both sides.
    source = _SOURCE.replace("cutlass.Float16", "cutlass.Float32").replace(
        "    bias_value = (bias.iterator + ni).load() if valid else cutlass.Float32(0)",
        "    prior = mi - 1 if mi > 0 else 0\n"
        "    bias_value = ((bias.iterator + ni) if saved < 0 else "
        "(alias_pointer + prior * 64 + ni)).load() if valid else cutlass.Float32(0)",
    )
    dominating = "alias_pointer = out.iterator"
    arguments = {
        "rows": 16,
        "columns": 64,
        "stride": 64,
        "base_offset": 0,
        "dtype": "cutlass.Float32",
        "before": dominating,
    }
    original = ast.parse(source).body[0]
    expected, writes, reads, _, _ = _run(original, **arguments, reverse_threads=True)
    forward, _, _, _, _ = _run(original, **arguments)
    np.testing.assert_array_equal(expected, forward)
    planned = _plan(source, dtype="cutlass.Float32", dominating=dominating)
    if planned is not None:
        actual, actual_writes, actual_reads, _, _ = _run(
            planned, **arguments, reverse_threads=True
        )
        np.testing.assert_array_equal(actual, expected)
        assert actual_writes == writes and actual_reads == reads


@pytest.mark.parametrize("dtype", list(_CASTS))
@pytest.mark.parametrize("unroll", [False, True])
def test_runtime_guard_checks_inner_stride(dtype: str, unroll: bool) -> None:
    source = _SOURCE.replace("cutlass.Float16", dtype).replace(
        "+ cutlass.Int32(ni)).store",
        "+ cutlass.Int32(ni) * cutlass.Int32(out.layout.stride[1])).store",
    )
    planned = _plan(source, dtype=dtype, unroll=unroll)
    assert planned is not None
    arguments = {
        "rows": 3,
        "columns": 31,
        "stride": 64,
        "inner_stride": 2,
        "base_offset": 0,
        "dtype": dtype,
    }
    expected, writes, reads, _, _ = _run(ast.parse(source).body[0], **arguments)
    actual, actual_writes, actual_reads, counts, _ = _run(planned, **arguments)
    np.testing.assert_array_equal(actual, expected)
    assert writes == actual_writes and reads == actual_reads
    assert counts["vector_store"] == 0


@pytest.mark.parametrize(
    "index",
    [
        "cutlass.Int32(ni)",
        "cutlass.Int32(ni) if ni % 2 == 0 else 0",
        "cutlass.Int32(cutlass.Float32(ni) * 0.125)",
    ],
)
def test_direct_loads_with_proved_integer_temporaries_remain_eligible(
    index: str,
) -> None:
    source = _SOURCE.replace(
        "    acc = shared", f"    column = {index}\n    acc = shared"
    ).replace("(bias.iterator + ni).load()", "(bias.iterator + column).load()")
    planned = _plan(source, output_specialized=True)
    assert isinstance(planned, ast.For)
    arguments = {
        "rows": 32,
        "columns": 64,
        "stride": 64,
        "base_offset": 0,
        "dtype": "cutlass.Float16",
    }
    expected, writes, reads, _, _ = _run(ast.parse(source).body[0], **arguments)
    actual, actual_writes, actual_reads, counts, _ = _run(planned, **arguments)
    np.testing.assert_array_equal(actual, expected)
    assert writes == actual_writes and reads == actual_reads
    assert counts["vector_store"] > 0


@pytest.mark.parametrize("target", ["bias", "out"])
def test_dominating_tensor_rebind_invalidates_storage_facts(target: str) -> None:
    other = "out" if target == "bias" else "bias"
    assert _plan(dominating=f"{target} = {other}") is None


def test_pointer_origin_type_facts_do_not_expand_integer_alias_dags() -> None:
    source = ["q0 = cutlass.Int32(ni)"]
    source.extend(f"q{i} = q{i - 1} + q{i - 1}" for i in range(1, 800))
    source.append("value = (bias.iterator + q799).load()")
    assert _load_pointer_roots(
        ast.parse("\n".join(source)).body,
        tensor_names=frozenset({"bias"}),
        integer_names={"ni"},
    ) == {"bias"}


def test_pointer_origin_discards_rebound_integer_type() -> None:
    source = """
index = cutlass.Int32(ni)
index = saved
value = (bias.iterator + index).load()
"""
    assert (
        _load_pointer_roots(
            ast.parse(source).body,
            tensor_names=frozenset({"bias"}),
            integer_names={"ni"},
        )
        is None
    )


@pytest.mark.parametrize(
    "kind", ["prior_sibling", "inner_iterator", "inner_body", "stateless"]
)
def test_nested_for_else_preserves_enclosing_backedge(kind: str) -> None:
    prefixes = {
        "prior_sibling": (
            "for repeat in range(2):\n"
            "    observed = carry\n"
            "    for child in range(1):\n"
            "        pass\n"
            "    else:\n"
        ),
        "inner_iterator": (
            "for repeat in range(2):\n"
            "    for child in range(1 if carry < 0 else 2):\n"
            "        observed = child\n"
            "    else:\n"
        ),
        "inner_body": (
            "for repeat in range(2):\n"
            "    for child in range(1):\n"
            "        observed = carry\n"
            "    else:\n"
        ),
        "stateless": (
            "for repeat in range(2):\n"
            "    for child in range(1):\n"
            "        pass\n"
            "    else:\n"
        ),
    }
    source = _SOURCE
    if kind != "stateless":
        source = source.replace(
            "    value = cutlass.Float16(acc + bias_value)",
            "    carry = cutlass.Float16(acc + bias_value)",
        ).replace(".store(value)", ".store(carry)")
    text = prefixes[kind] + "\n".join(
        "        " + line for line in source.strip().splitlines()
    )
    original = ast.parse(text).body[0]
    program = ast.parse(text).body
    loop = next(
        node
        for node in ast.walk(program[0])
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "lane"
    )
    _rewrite_program(program, loop)
    arguments = {
        "rows": 32,
        "columns": 64,
        "stride": 64,
        "base_offset": 0,
        "dtype": "cutlass.Float16",
        "before": "carry = cutlass.Float16(-1)",
    }
    expected, writes, reads, _, before_liveouts = _run(original, **arguments)
    actual, actual_writes, actual_reads, counts, after_liveouts = _run(
        program[0], **arguments
    )
    np.testing.assert_array_equal(actual, expected)
    assert writes == actual_writes and reads == actual_reads
    assert before_liveouts == after_liveouts
    if kind == "stateless":
        assert counts["vector_store"] > 0


@pytest.mark.parametrize("scope", ["dominating", "enclosing"])
def test_for_targets_preserve_global_constructor_bindings(scope: str) -> None:
    # The scalar path has no cute calls after expressing its physical lane
    # through tid. Vectorization introduces cute.make_rmem_tensor, so it may
    # not reuse an overwritten global constructor binding.
    source = _SOURCE.replace("cute.arch.thread_idx()[0]", "(tid % 2)")
    assert _plan(source) is not None
    if scope == "dominating":
        prefix = "for cute in range(1):\n    pass\n"
        original = ast.parse(prefix + source).body
        planned = _plan(source, dominating=prefix)
        rewritten = [
            *ast.parse(prefix).body,
            planned if planned is not None else ast.parse(source).body[0],
        ]
    else:
        text = "for cute in range(1):\n" + "\n".join(
            "    " + line for line in source.strip().splitlines()
        )
        original = ast.parse(text).body
        rewritten = ast.parse(text).body
        loop = cast("ast.For", cast("ast.For", rewritten[0]).body[0])
        _rewrite_program(rewritten, loop)
    # A one-trip wrapper executes a statement sequence with one shared scope.
    before = ast.For(
        target=ast.Name("wrapper", ast.Store()),
        iter=ast.parse("range(1)", mode="eval").body,
        body=original,
        orelse=[],
    )
    after = ast.For(
        target=ast.Name("wrapper", ast.Store()),
        iter=ast.parse("range(1)", mode="eval").body,
        body=rewritten,
        orelse=[],
    )
    arguments = {
        "rows": 32,
        "columns": 64,
        "stride": 64,
        "base_offset": 0,
        "dtype": "cutlass.Float16",
    }
    expected, writes, reads, _, _ = _run(before, **arguments)
    actual, actual_writes, actual_reads, _, _ = _run(after, **arguments)
    np.testing.assert_array_equal(actual, expected)
    assert writes == actual_writes and reads == actual_reads
