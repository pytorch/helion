from __future__ import annotations

import ast
import inspect
import textwrap
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute.flatten_nested_reductions import flatten_nested_reductions
from helion._testing import import_example
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_structural_kernels import _joint_sum

jagged_layer_norm_kernel = import_example("jagged_layer_norm").jagged_layer_norm_kernel
reference_jagged_layer_norm_pytorch = import_example(
    "jagged_layer_norm"
).reference_jagged_layer_norm_pytorch

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]

if TYPE_CHECKING:
    from pathlib import Path
    from types import FunctionType


@pytest.mark.parametrize("columns", [32, 69, 128, 512])
@pytest.mark.parametrize("static_shapes", [False, True])
@pytest.mark.parametrize("function", [jagged_layer_norm_kernel.fn, _joint_sum])
def test_joint_reductions_compile_as_flat_jagged_passes(
    columns: int, static_shapes: bool, function: FunctionType
) -> None:
    kernel = helion.kernel(
        function,
        backend="cute",
        static_shapes=static_shapes,
        cute_flatten_nested_reductions=True,
        autotune_effort="none",
    )
    with mock_cuda_unavailable():
        bound = cpu_bind(
            kernel, (torch.empty((731, columns)), torch.empty(18, dtype=torch.int64))
        )
    host = bound.host_function
    assert host is not None
    source = ast.unparse(ast.Module(body=host.body, type_ignores=[]))
    expected_axes = 4 if function is jagged_layer_norm_kernel.fn else 3
    assert len(bound.config_spec.block_sizes) == expected_axes
    assert "_helion_flat_reduction" in source
    config = helion.Config(
        block_sizes=[1, *([1024] * (expected_axes - 1))],
        num_threads=[1, *([128] * (expected_axes - 1))],
        cute_vector_widths=[1, *([4] * (expected_axes - 1))],
        cute_lane_layouts=["strided"] * expected_axes,
    )
    with patch(
        "helion._compiler.cute.thread_budget.shared_memory_budget_bytes",
        return_value=232448,
    ):
        generated = bound.to_code(config)
    assert " // " not in generated and " % 69" not in generated


class _Decision(Exception):
    def __init__(self, changed: int) -> None:
        self.changed = changed


@pytest.mark.parametrize("shift", [0, 1])
def test_host_loop_storage_rebinding_preserves_original_loops(
    tmp_path: Path, shift: int
) -> None:
    source = textwrap.dedent(inspect.getsource(_joint_sum))
    source = source.replace(
        "    for row in hl.tile(batches):",
        "    for unused in range(1):\n"
        f"        dest.set_(source[{shift}:])\n"
        "    for row in hl.tile(batches):",
    )
    path = tmp_path / "host_storage_rebinding.py"
    path.write_text(source)
    namespace = dict(_joint_sum.__globals__)
    exec(compile(source, str(path), "exec"), namespace)

    bodies = []
    for enabled in (False, True):
        kernel = helion.kernel(
            namespace["_joint_sum"],
            backend="cute",
            cute_flatten_nested_reductions=enabled,
            autotune_effort="none",
        )
        with mock_cuda_unavailable():
            bound = cpu_bind(
                kernel,
                (torch.empty((731, 69)), torch.empty(18, dtype=torch.int64)),
            )
        host = bound.host_function
        assert host is not None
        bodies.append(ast.dump(ast.Module(body=host.body, type_ignores=[])))
        assert len(bound.config_spec.block_sizes) == 5
    assert bodies[0] == bodies[1]


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("torch.empty_like(x)", "x"),
        (
            "total = hl.zeros([row], dtype=torch.float32)",
            "total = -hl.zeros([row], dtype=torch.float32)",
        ),
        (
            "total = hl.zeros([row], dtype=torch.float32)",
            "total = hl.zeros([row], dtype=torch.float32) + 1",
        ),
        (
            "partial = hl.zeros([row, column], dtype=torch.float32)",
            "partial = hl.full([row, column], 1, dtype=torch.float32)",
        ),
        ("dtype=torch.float32", "dtype=torch.float64"),
        (
            "(value * 1.25 + 2).sum(dim=1)",
            "(value * 1.25 + 2).sum(dim=1, dtype=torch.float64)",
        ),
        ("(value * 1.25 + 2).sum(dim=1)", "(value + partial[:, None, :]).sum(dim=1)"),
        ("(value * 1.25 + 2).sum(dim=1)", "(value + total[:, None, None]).sum(dim=1)"),
        (
            "value = hl.load(source, [index])",
            "value = hl.load(source, [index])\n                hl.store(source, [index], value)",
        ),
        (
            "value = hl.load(source, [index])",
            "value = custom_operation(hl.load(source, [index]))",
        ),
        ("scale = total /", "escaped = partial\n        scale = total /"),
        ("hl.store(dest, [index],", "hl.store(dest, [index % columns],"),
        ("value = hl.load(source, [index])", "value = hl.load(dest, [index])"),
        ("scale = total /", "offsets[row] = 0\n        scale = total /"),
    ],
)
def test_rewrite_declines_unproved_reduction_and_memory_contracts(
    tmp_path: Path, old: str, new: str
) -> None:
    source = textwrap.dedent(inspect.getsource(_joint_sum))
    assert old in source
    source = source.replace(old, new)
    path = tmp_path / "joint_variant.py"
    path.write_text(source)
    namespace = dict(_joint_sum.__globals__)
    exec(compile(source, str(path), "exec"), namespace)

    def decide(host):
        raise _Decision(flatten_nested_reductions(host))

    kernel = helion.kernel(
        namespace["_joint_sum"], backend="cute", cute_flatten_nested_reductions=True
    )
    with (
        mock_cuda_unavailable(),
        patch(
            "helion._compiler.cute.flatten_nested_reductions.flatten_nested_reductions",
            decide,
        ),
        pytest.raises(_Decision) as result,
    ):
        cpu_bind(kernel, (torch.empty((731, 69)), torch.empty(18, dtype=torch.int64)))
    assert result.value.changed == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_original_reduction_precision_is_required(dtype: torch.dtype) -> None:
    def decide(host):
        raise _Decision(flatten_nested_reductions(host))

    kernel = helion.kernel(
        jagged_layer_norm_kernel.fn, backend="cute", cute_flatten_nested_reductions=True
    )
    with (
        mock_cuda_unavailable(),
        patch(
            "helion._compiler.cute.flatten_nested_reductions.flatten_nested_reductions",
            decide,
        ),
        pytest.raises(_Decision) as result,
    ):
        cpu_bind(
            kernel,
            (torch.empty((731, 69), dtype=dtype), torch.empty(18, dtype=torch.int64)),
        )
    assert result.value.changed == 0


CUDA_DEVICE = "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("columns", [4, 69, 128])
@pytest.mark.parametrize("rows_per_tile", [1, 2])
@pytest.mark.parametrize("unaligned", [False, True])
def test_flattened_reductions_preserve_tails_empty_rows_and_graph_updates(
    columns: int, rows_per_tile: int, unaligned: bool
) -> None:
    lengths = torch.tensor([0, 1, 2, 9, 257, 0, 513, 7, 65], dtype=torch.int64)
    rows = int(lengths.sum())
    shift = int(unaligned)
    storage = torch.randn(rows * columns + shift, device=CUDA_DEVICE)
    values = storage[shift:].view(rows, columns)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), lengths.cumsum(0)]).cuda()
    kernel = helion.kernel(
        jagged_layer_norm_kernel.fn,
        backend="cute",
        cute_flatten_nested_reductions=True,
        static_shapes=False,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated((values, offsets, 1e-6))
    assert len(bound.config_spec.block_sizes) == 4
    config = helion.Config(
        block_sizes=[rows_per_tile, 512, 512, 512],
        num_threads=[rows_per_tile, 128, 128, 128],
        cute_vector_widths=[1, 4, 4, 4],
        cute_lane_layouts=["strided"] * 4,
    )
    generated = bound.to_code(config)
    if not unaligned:
        assert "_helion_affine_load" in generated
    bound.set_config(config)
    args = (values, offsets, 1e-6)
    for _ in range(2):
        actual = bound(*args)
        expected = reference_jagged_layer_norm_pytorch(*args)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = bound(*args)
    for seed in range(3):
        values.copy_(torch.randn_like(values) * (seed + 1))
        changed_lengths = lengths.roll(seed + 1)
        offsets.copy_(
            torch.cat(
                [torch.zeros(1, dtype=torch.int64), changed_lengths.cumsum(0)]
            ).cuda()
        )
        graph.replay()
        torch.cuda.synchronize()
        expected = reference_jagged_layer_norm_pytorch(*args)
        torch.testing.assert_close(captured, expected, atol=1e-5, rtol=1e-4)
