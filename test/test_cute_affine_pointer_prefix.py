from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test.test_cute_affine_vector_io import _SOURCE
from test.test_cute_affine_vector_io import _execute
from test.test_cute_affine_vector_io import _rewrite

import helion
from helion._compiler.runtime_tensor_guards import (
    _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY,
)
from helion._testing import import_example
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_codegen import tcgen05_target
from helion.autotuner.benchmarking import _make_cudagraph_replay
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only

add_2d = import_example("aot_compile_example").add_2d


def _prefix_source(prefix: str) -> str:
    source = _SOURCE.replace(
        "address = cutlass.Int32(start) + indices_1", "address = indices_1"
    )
    for name in ("src", "out"):
        source = source.replace(f"{name}.iterator + ", f"{name}.iterator + {prefix} + ")
    return source


@pytest.mark.parametrize("length", [0, 1, 3, 4, 5, 15, 16, 17, 32])
@pytest.mark.parametrize("row", range(3))
@pytest.mark.parametrize(
    "prefix",
    [
        "cutlass.Int32(start) * cutlass.Int32(16)",
        "cutlass.Int32(start) * cutlass.Int32(8) + cutlass.Int32(start) * cutlass.Int32(8)",
    ],
)
def test_row_and_batch_prefixes_preserve_values_and_tails(
    length: int, row: int, prefix: str
) -> None:
    source = _prefix_source(prefix)
    transformed, changed = _rewrite(source)
    assert changed == 1
    expected, before = _execute(ast.parse(source), row, length)
    actual, after = _execute(transformed, row, length)
    assert actual == expected
    assert before["scalar_loads"] == before["scalar_stores"] == length
    assert after["vector_loads"] == after["vector_stores"] == length // 4
    assert after["scalar_loads"] == after["scalar_stores"] == length % 4


@pytest.mark.parametrize(
    "prefix",
    [
        "cutlass.Int32(start * 4)",
        "cutlass.Int64(start) * cutlass.Int64(16)",
        "cutlass.Int32(vec_lane_1) * cutlass.Int32(16)",
        "cutlass.Int32(2)",
    ],
)
def test_unproved_or_lane_varying_prefix_keeps_original_ast(prefix: str) -> None:
    source = _prefix_source(prefix)
    transformed, changed = _rewrite(source)
    assert changed == 0
    assert ast.dump(transformed, include_attributes=False) == ast.dump(
        ast.parse(source), include_attributes=False
    )


def test_float_before_prefix_cast_is_not_assumed_divisible() -> None:
    source = _prefix_source("cutlass.Int32(start * 4)")
    transformed, changed = _rewrite(source)
    expected, _before = _execute(ast.parse(source), 1.25, 16)
    actual, after = _execute(transformed, 1.25, 16)
    assert changed == 0
    assert actual == expected
    assert after["vector_loads"] == 0


@pytest.mark.parametrize("start", [0, 1, 2, 3, 4, 8])
def test_dynamic_prefix_alignment_uses_scalar_fallback(start: int) -> None:
    source = _prefix_source("cutlass.Int32(start)")
    transformed, changed = _rewrite(source)
    expected, _before = _execute(ast.parse(source), start, 16)
    actual, after = _execute(transformed, start, 16)
    assert changed == 1
    assert actual == expected
    assert after["vector_loads"] == (4 if start % 4 == 0 else 0)


def test_dynamic_prefix_guard_follows_original_masks() -> None:
    source = _prefix_source("cutlass.Int32(8 // cutlass.Int32(start))").replace(
        "indices_1 < cutlass.Int32(length)",
        "indices_1 < cutlass.Int32(length) and start != 0",
    )
    transformed, changed = _rewrite(source)
    expected, _before = _execute(ast.parse(source), 0, 16)
    actual, after = _execute(transformed, 0, 16)
    assert changed == 1
    assert actual == expected
    assert not any(after.values())


def test_inactive_prefix_float_conversion_is_not_speculated() -> None:
    source = _prefix_source("cutlass.Int32(start) * cutlass.Int32(16)").replace(
        "indices_1 < cutlass.Int32(length)",
        "indices_1 < cutlass.Int32(length) and start == start",
    )
    transformed, changed = _rewrite(source)
    expected, _before = _execute(ast.parse(source), float("nan"), 16)
    actual, after = _execute(transformed, float("nan"), 16)
    assert changed == 1
    assert actual == expected
    assert not any(after.values())


@pytest.mark.parametrize("prefix", ["cutlass.Int32(4)", "cutlass.Int32(2147483644)"])
def test_prefix_pointer_additions_are_not_reassociated_to_int32(prefix: str) -> None:
    source = _prefix_source(prefix).replace(
        "address = indices_1", f"address = indices_1 - {prefix}"
    )
    transformed, changed = _rewrite(source)
    expected, _before = _execute(ast.parse(source), 0, 16)
    actual, after = _execute(transformed, 0, 16)
    assert changed == 1
    assert actual == expected
    assert after["vector_loads"] == 4
    for node in ast.walk(transformed):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "load"
            and node.args
        ):
            pointer = node.args[0]
            assert isinstance(pointer, ast.BinOp)
            assert isinstance(pointer.left, ast.BinOp)
            assert isinstance(pointer.left.left, ast.Attribute)
            assert pointer.left.left.attr == "iterator"


@pytest.mark.parametrize("shape", [(256, 256), (1024, 1024), (4096, 4096)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@skipUnlessBackends(["cute"])
def test_original_dynamic_aot_add_uses_vector_memory(
    shape: tuple[int, int], dtype: torch.dtype
) -> None:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            tcgen05_target(),
            mock_cuda_unavailable(),
            patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        ):
            kernel = helion.kernel(
                add_2d.fn, backend="cute", static_shapes=False, autotune_effort="none"
            )
            bound = kernel._bind_isolated(
                (
                    torch.empty(shape, dtype=dtype),
                    torch.empty(shape, dtype=dtype),
                )
            )
            width = 4 if dtype is torch.float32 else 8
            code = bound.to_code(
                helion.Config(
                    block_sizes=[1, 4096],
                    num_threads=[0, 256],
                    cute_vector_widths=[1, width],
                    cute_lane_layouts=["blocked", "blocked"],
                    cute_cluster_n=1,
                )
            )
            assert "cute.arch.load(" in code
            assert f"ir.VectorType.get([{width}]" in code
            assert "_cute_store_u" in code
    finally:
        torch.set_num_threads(previous)


def _add_into_2d(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for row, column in hl.tile(x.shape):
        out[row, column] = x[row, column] + y[row, column]
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@skipUnlessBackends(["cute"])
def test_external_output_registers_alias_facts_for_vector_memory(
    dtype: torch.dtype,
) -> None:
    """A store into an ``out`` argument proves disjointness only through the
    cache-specialized alias matrix, which grid kernels register on demand."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            tcgen05_target(),
            mock_cuda_unavailable(),
            patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        ):
            kernel = helion.kernel(
                _add_into_2d,
                backend="cute",
                static_shapes=False,
                autotune_effort="none",
            )
            x = torch.empty((256, 256), dtype=dtype)
            # Keep every argument alive: the disjointness proof replays their
            # current storage spans while generating code.
            args = (x, torch.empty_like(x), torch.empty_like(x))
            bound = kernel._bind_isolated(args)
            assert _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY in (
                bound.env.runtime_input_specializations
            )
            width = 4 if dtype is torch.float32 else 8
            code = bound.to_code(
                helion.Config(
                    block_sizes=[1, 4096],
                    num_threads=[0, 256],
                    cute_vector_widths=[1, width],
                    cute_lane_layouts=["blocked", "blocked"],
                    cute_cluster_n=1,
                )
            )
            assert f"ir.VectorType.get([{width}]" in code
            assert "_cute_store_u" in code
    finally:
        torch.set_num_threads(previous)


# Generated pointer-prefix memory I/O, including guarded and aliased layouts.
#
# This file is kept outside the frozen compiler and can be promoted unchanged to
# the production test suite after the component's GPU validation.


CUDA_DEVICE = "cuda"


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
LAYOUTS = (
    "aligned",
    "padded",
    "odd_tail",
    "misaligned_rows",
    "unaligned_base",
    "inner_stride_2",
    "alias_x",
    "alias_y",
    "aligned_return",
)
SCALAR_LAYOUTS = {"unaligned_base", "inner_stride_2", "alias_x", "alias_y"}
ORIGINAL_SHAPES = ((256, 256), (1024, 1024), (4096, 4096))


def add_into_2d(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for row, column in hl.tile(x.shape):
        out[row, column] = x[row, column] + y[row, column]
    return out


def add_into_3d(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    for batch, row, column in hl.tile(x.shape):
        out[batch, row, column] = x[batch, row, column] + y[batch, row, column]
    return out


def pinned_config(rank: int, dtype: torch.dtype) -> helion.Config:
    width = 4 if dtype is torch.float32 else 8
    return helion.Config(
        block_sizes=[*([1] * (rank - 1)), 4096],
        num_threads=[*([0] * (rank - 1)), 256],
        cute_vector_widths=[*([1] * (rank - 1)), width],
        cute_lane_layouts=["blocked"] * rank,
        cute_cluster_n=1,
    )


def make_kernel(rank: int, dtype: torch.dtype) -> helion.Kernel:
    return helion.kernel(
        add_into_2d if rank == 2 else add_into_3d,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        configs=[pinned_config(rank, dtype)],
    )


@dataclass
class Buffers:
    arguments: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    backings: list[torch.Tensor]
    output_owner: int

    def mutate(self, seed: int) -> None:
        for index, storage in enumerate(self.backings):
            values = torch.arange(storage.numel(), device=storage.device)
            storage.copy_(((values * 17 + seed * 29 + index * 31) % 251 - 125) / 16)
        for index, tensor in enumerate(self.arguments[:2]):
            values = torch.arange(tensor.numel(), device=tensor.device)
            tensor.copy_(
                (((values * (31 + index * 6) + seed * 47) % 1031 - 515) / 13)
                .to(tensor.dtype)
                .reshape(tensor.shape)
            )
        # Include signed zero and cancellation without relying on NaN payloads.
        left, right, _out = self.arguments
        prefix = (0,) * (left.ndim - 1)
        left[(*prefix, 0)] = -0.0
        right[(*prefix, 0)] = -0.0
        left[(*prefix, 1)] = 1.0
        right[(*prefix, 1)] = -1.0

    def expected_backings(self) -> list[torch.Tensor]:
        result = [storage.clone() for storage in self.backings]
        left, right, out = self.arguments
        destination = result[self.output_owner].as_strided(
            out.shape, out.stride(), out.storage_offset()
        )
        destination.copy_(left + right)
        return result


def make_buffers(rank: int, dtype: torch.dtype, layout: str, device: str) -> Buffers:
    columns = (
        259
        if layout in {"odd_tail", "misaligned_rows", "unaligned_base", "inner_stride_2"}
        else 256
    )
    shape = (5, columns) if rank == 2 else (3, 5, columns)
    offset = 1 if layout == "unaligned_base" else 0
    inner_stride = 2 if layout == "inner_stride_2" else 1
    backings = []
    arguments = []
    for index in range(3):
        if layout in {"aligned", "aligned_return", "alias_x", "alias_y"}:
            row_stride = columns
        elif layout == "misaligned_rows":
            row_stride = columns + 2 + index * 8
        else:
            row_stride = ((columns * inner_stride + 7) // 8 + 2 + index) * 8
        strides = (row_stride, inner_stride)
        if rank == 3:
            batch_stride = 5 * row_stride + (
                0
                if layout in {"aligned", "aligned_return", "alias_x", "alias_y"}
                else 16 * (index + 1)
            )
            strides = (batch_stride, *strides)
        size = (
            offset
            + sum(
                (dim - 1) * stride for dim, stride in zip(shape, strides, strict=True)
            )
            + 1
        )
        storage = torch.empty(size + 17, dtype=dtype, device=device)
        backings.append(storage)
        arguments.append(storage.as_strided(shape, strides, offset))
    owner = 2
    if layout in {"alias_x", "alias_y"}:
        owner = 0 if layout == "alias_x" else 1
        arguments[2] = arguments[owner]
        backings.pop()
    result = Buffers((arguments[0], arguments[1], arguments[2]), backings, owner)
    result.mutate(0)
    return result


def memory_contract(source: str, dtype: torch.dtype, layout: str) -> dict[str, object]:
    width = 4 if dtype is torch.float32 else 8
    wide_load = f"ir.VectorType.get([{width}]" in source
    wide_store = "_cute_store_u" in source
    expected_wide = layout not in SCALAR_LAYOUTS
    assert wide_load is expected_wide, (layout, width, "load", source)
    assert wide_store is expected_wide, (layout, width, "store", source)
    if layout in {"odd_tail", "misaligned_rows"}:
        assert ").load()" in source and ").store(" in source
    if layout == "misaligned_rows":
        assert f"% {width} == 0" in source
    return {
        "layout": layout,
        "wide_load": wide_load,
        "wide_store": wide_store,
        "width": width,
        "scalar_required": not expected_wide,
    }


def tensor_metadata(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "dtype": str(tensor.dtype),
        "base_alignment": next(
            alignment
            for alignment in (16, 8, 4, 2, 1)
            if tensor.data_ptr() % alignment == 0
        ),
    }


def _check_call(function: Callable[[], object], buffers: Buffers) -> None:
    expected = buffers.expected_backings()
    actual = function()
    assert isinstance(actual, torch.Tensor)
    assert actual.data_ptr() == buffers.arguments[2].data_ptr()
    for observed, reference in zip(buffers.backings, expected, strict=True):
        # Includes every padding byte, guard byte, and unmodified input byte.
        assert torch.equal(observed.view(torch.uint8), reference.view(torch.uint8))


def _check_source() -> None:
    expected = os.environ.get("HELION_POINTER_PREFIX_SOURCE_ROOT")
    if expected is not None:
        assert (
            Path(helion.__file__).resolve()
            == (Path(expected) / "helion/__init__.py").resolve()
        )


def _save_case(
    name: str,
    bound: BoundKernel,
    config: helion.Config,
    source: str,
    record: dict[str, object],
) -> None:
    destination = os.environ.get("HELION_POINTER_PREFIX_ARTIFACT_DIR")
    if destination is None:
        return
    directory = Path(destination) / name
    directory.mkdir(parents=True, exist_ok=False)
    actual_path = bound.get_cached_path(config)
    assert actual_path is not None
    actual_source = Path(actual_path).read_text()
    assert actual_source == source
    (directory / "generated.py").write_text(actual_source)
    record.update(
        actual_module_path=actual_path,
        generated_sha256=hashlib.sha256(actual_source.encode()).hexdigest(),
        config=bound._normalized_config_copy(config).config,
        status="pass",
    )
    (directory / "result.json").write_text(json.dumps(record, indent=2) + "\n")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype_name", DTYPES)
@pytest.mark.parametrize("rank", [2, 3], ids=["2d", "3d"])
def test_pointer_prefix_add_into_cuda(rank: int, dtype_name: str) -> None:
    _check_source()
    dtype = DTYPES[dtype_name]
    kernel = make_kernel(rank, dtype)
    config = pinned_config(rank, dtype)
    retained: list[Buffers] = []
    for ordinal, layout in enumerate(LAYOUTS):
        buffers = make_buffers(rank, dtype, layout, "cuda")
        # Keep the original specialization's input metadata available for
        # repeated source inspection when the aligned layout is revisited.
        retained.append(buffers)
        bound = kernel.bind(buffers.arguments)
        source = bound.to_code(config)
        contract = memory_contract(source, dtype, layout)
        bound.set_config(config)

        def run(buffers: Buffers = buffers) -> torch.Tensor:
            return kernel(*buffers.arguments)

        _check_call(run, buffers)
        replay = _make_cudagraph_replay(run)
        for iteration in range(3):
            buffers.mutate(ordinal * 10 + iteration + 1)
            _check_call(replay, buffers)
        fresh = make_buffers(rank, dtype, layout, "cuda")
        fresh.mutate(71 + ordinal)
        assert fresh.arguments[0].data_ptr() != buffers.arguments[0].data_ptr()
        _check_call(lambda fresh=fresh: kernel(*fresh.arguments), fresh)
        _save_case(
            f"{rank}d-{dtype_name}-{layout}",
            bound,
            config,
            source,
            {
                **contract,
                "arguments": [tensor_metadata(tensor) for tensor in buffers.arguments],
                "eager_exact": True,
                "mutated_graph_replays_exact": 3,
                "fresh_pointer_exact": True,
                "all_backing_bytes_exact": True,
                "static_shapes": False,
            },
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_original_aot_add_dynamic_cuda() -> None:
    _check_source()
    dtype = torch.bfloat16
    kernel = helion.kernel(
        add_2d.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        configs=[pinned_config(2, dtype)],
    )
    config = pinned_config(2, dtype)
    retained = []
    for shape in ORIGINAL_SHAPES:
        arguments = tuple(
            torch.empty(shape, device=CUDA_DEVICE, dtype=dtype) for _ in range(2)
        )
        retained.append(arguments)
        for tensor in arguments:
            tensor.normal_()
        bound = kernel.bind(arguments)
        source = bound.to_code(config)
        contract = memory_contract(source, dtype, "aligned")
        bound.set_config(config)

        def run(arguments: tuple[torch.Tensor, ...] = arguments) -> torch.Tensor:
            return kernel(*arguments)

        assert torch.equal(
            run().view(torch.uint8), (arguments[0] + arguments[1]).view(torch.uint8)
        )
        replay = _make_cudagraph_replay(run)
        for _iteration in range(3):
            for tensor in arguments:
                tensor.normal_()
            expected = arguments[0] + arguments[1]
            assert torch.equal(replay().view(torch.uint8), expected.view(torch.uint8))
        fresh = tuple(tensor.clone() for tensor in arguments)
        assert torch.equal(
            kernel(*fresh).view(torch.uint8), (fresh[0] + fresh[1]).view(torch.uint8)
        )
        _save_case(
            f"original-aot-add-{shape[0]}-bf16",
            bound,
            config,
            source,
            {
                **contract,
                "arguments": [tensor_metadata(tensor) for tensor in arguments],
                "eager_exact": True,
                "mutated_graph_replays_exact": 3,
                "fresh_pointer_exact": True,
                "static_shapes": False,
            },
        )
