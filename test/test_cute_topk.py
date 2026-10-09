"""Top-k lowering, selection networks, tuning, and correctness tests."""

from __future__ import annotations

import ast
import copy
import dataclasses
import gc
import itertools
import json
import operator
import os
from pathlib import Path
import random
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_native_compile
from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target
from test.cute_register_model import execute_register_plan_events
from test.cute_register_model import gather_registers

import helion
from helion import exc
from helion._compiler.autotuner_heuristics import get_heuristics
from helion._compiler.autotuner_heuristics.cute import CuteTopKHeuristic
from helion._compiler.backend import CuteBackend
from helion._compiler.backend import TritonBackend
from helion._compiler.cute import selection_coarse
from helion._compiler.cute.memory_ops import _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
from helion._compiler.cute.row_fragment import RowFragment
from helion._compiler.cute.row_fragment import RowFragmentLayout
from helion._compiler.cute.selection_coarse import trace_coarse_rank_selection
from helion._compiler.cute.selection_network import selection_network
from helion._compiler.cute.selection_network import trace_selection_network
from helion._compiler.cute.topk import _match_direct_topk_root
from helion._compiler.cute.topk import match_topk_root
from helion._compiler.cute.topk import topk_tensors_are_proven_disjoint
from helion._testing import DEVICE
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessCuteAvailable
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import ConfigSpec
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator


from helion._compiler.cute.sorting_networks import COMPACT_SORT_LAYERS
from helion.runtime import default_cute_launcher

# Basic top-k tests.


_TOPK_STRUCTURAL_POLICY = helion.CuteStructuralPolicy(
    cute_region_fission=True,
    cute_full_slice_matmul_tiling=True,
    cute_segmented_matmul_tiling=True,
    cute_flatten_nested_reductions=True,
    cute_materialize_transformed_operands=True,
)


def _selected_topk(
    x: torch.Tensor,
    k: int,
    softmax: hl.constexpr = False,  # pyrefly: ignore[bad-function-definition]
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = x.size(0)
    k = hl.specialize(k)
    values = torch.empty((rows, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((rows, k), dtype=torch.int32, device=x.device)
    for row in hl.tile(rows):
        vals, idx = torch.topk(x[row, :], k, dim=-1, largest=True, sorted=True)
        if softmax:
            vals = torch.softmax(vals.to(torch.float32), dim=-1).to(x.dtype)
        values[row, :] = vals
        indices[row, :] = idx
    return values, indices


def _compact_register_plans(source: str) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    if not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "helion.runtime.cute.register_tensor"
        and any(alias.name == "_cute_execute_register_plan" for alias in node.names)
        for node in tree.body
    ):
        return []
    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    return [
        json.loads(constants[node.args[0].id])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_cute_execute_register_plan"
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in constants
    ]


def _has_register_selection(source: str) -> bool:
    return bool(_compact_register_plans(source)) or any(
        isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "cute.make_rmem_tensor"
        and any(
            isinstance(target, ast.Name) and "register_output" in target.id
            for target in node.targets
        )
        for node in ast.walk(ast.parse(source))
    )


def _has_register_communication(source: str) -> bool:
    if any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func).startswith("cute.arch.shuffle_sync")
        for node in ast.walk(ast.parse(source))
    ):
        return True
    for plan in _compact_register_plans(source):
        for node in plan["nodes"]:
            if node.get("axis") != 0 or not node.get("live_registers"):
                continue
            mapping = plan["maps"][node["map"]]
            if any(
                mapping["rows"][owner][register] != group
                for group, owner in enumerate(mapping["owners"])
                for register in node["live_registers"]
            ):
                return True
    return False


def _assert_register_selection(source: str) -> None:
    assert _has_register_selection(source)
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "helion.runtime.cute.topk"
        and any(
            alias.name in ("local_topk", "distributed_topk") for alias in node.names
        )
        for node in ast.walk(ast.parse(source))
    )


def test_topk_collection_without_cute(tmp_path: Path) -> None:
    """A Triton-only installation keeps portable tests and skips CuTe tests."""
    script = textwrap.dedent(
        """
        import sys
        sys.modules["cutlass"] = None
        sys.modules["cutlass.cute"] = None

        import pytest

        class Results:
            def __init__(self):
                self.nodes = []
                self.reports = []

            def pytest_collection_modifyitems(self, items):
                self.nodes = [item.nodeid for item in items]

            def pytest_runtest_logreport(self, report):
                if report.when == "call" or report.skipped:
                    self.reports.append((report.outcome, str(report.longrepr)))

        collected = Results()
        assert pytest.main([
            "test/test_misc.py::TestMisc::test_torch_topk_in_kernel",
            "test/test_misc.py::TestMisc::test_torch_topk_smallest",
            "--collect-only", "-q"
        ], plugins=[collected]) == 0
        assert {node.rsplit("::", 1)[-1] for node in collected.nodes} == {
            "test_torch_topk_in_kernel", "test_torch_topk_smallest"
        }

        executed = Results()
        assert pytest.main([
            "test/test_cute_topk.py::test_topk_numeric_float_key_exactness_boundary",
            "test/test_cute_topk.py::test_compact_network_bounds_and_operation_count",
            "-q"
        ], plugins=[executed]) == 0
        assert [outcome for outcome, _ in executed.reports] == ["passed", "skipped"]
        assert "CuTe DSL" in executed.reports[1][1]
        """
    )
    probe = tmp_path / "without_cute.py"
    probe.write_text(script)
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HELION_BACKEND": "triton"}
    env["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [sys.executable, str(probe)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_topk(
    x: torch.Tensor, k: int, largest: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    rows = x.size(0)
    values = torch.empty((rows, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((rows, k), dtype=torch.int64, device=x.device)
    for row in hl.tile(rows):
        vals, idx = torch.topk(x[row, :], k, dim=-1, largest=largest, sorted=True)
        values[row, :] = vals
        indices[row, :] = idx
    return values, indices


@skipIfRefEager("requires compiler IR and explicit configurations")
@pytest.mark.parametrize("backend", ["cute", "triton"])
@pytest.mark.parametrize("operation", ["topk", "sort"])
def test_ordering_axis_keeps_complete_reduction_input(backend, operation):
    if backend == "cute":
        pytest.importorskip("cutlass.cute")
    function = _row_topk.fn if operation == "topk" else _row_network_sort.fn
    inputs = (
        (torch.ones(3, 8192), 64, True)
        if operation == "topk"
        else (torch.ones(3, 8192), True)
    )
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
    ):
        kernel = helion.kernel(
            function, backend=backend, static_shapes=True, autotune_effort="none"
        )
        bound = _cpu_bind(kernel, inputs)
        assert not bound.config_spec.reduction_loops.valid_block_ids()
        source = bound.to_code(bound.config_spec.default_config())
    if backend == "triton":
        tree = ast.parse(source)
        assert not any(isinstance(node, ast.For) for node in ast.walk(tree))
        assert "8192" in source


@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def _topk_with_independent_sum(x: torch.Tensor, y: torch.Tensor):
    values = torch.empty((x.size(0), 8), device=x.device, dtype=x.dtype)
    indices = torch.empty((x.size(0), 8), device=x.device, dtype=torch.int64)
    sums = torch.empty((x.size(0),), device=x.device, dtype=x.dtype)
    for row in hl.tile(x.size(0)):
        selected, order = torch.topk(x[row, :], 8, dim=-1)
        values[row, :] = selected
        indices[row, :] = order
        sums[row] = y[row, :].sum(-1)
    return values, indices, sums


@skipIfRefEager("requires compiler IR and explicit configurations")
def test_ordering_does_not_disable_independent_reduction_rolling():
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
    ):
        bound = _cpu_bind(
            _topk_with_independent_sum, (torch.ones(3, 64), torch.ones(3, 8192))
        )
        blocks = bound.config_spec.reduction_loops.valid_block_ids()
        assert blocks
        with bound.env:
            assert all(
                bound.env.block_sizes[block].size_hint() == 8192 for block in blocks
            )
        config = bound.config_spec.default_config()
        config.config["reduction_loops"] = [4096] * len(blocks)
        source = bound.to_code(config)
    assert any(
        isinstance(node, ast.For) and "8192" in ast.unparse(node.iter)
        for node in ast.walk(ast.parse(source))
    )


@pytest.mark.parametrize("operation", ["topk", "sort"])
@pytest.mark.parametrize("dim", [-2, -1, 0, 1])
def test_ordering_axis_guard_uses_consumed_dimension(operation, dim):
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.roll_reduction import ReductionRoller

    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    source.meta["val"] = torch.empty(5, 9)
    target = (
        torch.ops.aten.topk.default
        if operation == "topk"
        else torch.ops.aten.sort.default
    )
    node = graph.call_function(
        target, (source, 3, dim) if operation == "topk" else (source, dim)
    )
    node.meta.update(
        val=target(source.meta["val"], 3, dim)
        if operation == "topk"
        else target(source.meta["val"], dim),
        lowering=None,
    )
    axis = dim % 2
    env = SimpleNamespace(get_block_id=lambda size: {5: 0, 9: 1}.get(size))
    roller = ReductionRoller(None, SimpleNamespace(block_id=axis), {})
    with (
        patch.object(CompileEnvironment, "current", return_value=env),
        pytest.raises(
            NotImplementedError, match="selection axes require complete input"
        ),
    ):
        roller.should_go_in_inner_graph(node)


def _check_topk(
    x: torch.Tensor,
    original: torch.Tensor,
    output: tuple[torch.Tensor, torch.Tensor],
    k: int,
    largest: bool,
) -> None:
    values, indices = output
    expected = torch.topk(original, k, dim=-1, largest=largest, sorted=True).values
    assert values.shape == indices.shape == (x.size(0), k)
    assert values.dtype == x.dtype
    assert indices.dtype == torch.int64
    assert values.device == indices.device == x.device
    torch.testing.assert_close(values, expected, rtol=0, atol=0, equal_nan=True)
    assert bool(((indices >= 0) & (indices < x.size(1))).all())
    torch.testing.assert_close(
        values, original.gather(1, indices), rtol=0, atol=0, equal_nan=True
    )
    ordered_indices = indices.sort(dim=-1).values
    assert bool((ordered_indices[:, 1:] != ordered_indices[:, :-1]).all())
    # Compare bit patterns to preserve NaN payloads and signed zero in the input.
    assert torch.equal(x.view(torch.int16), original.view(torch.int16))


def _inputs(width: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator().manual_seed(2026)
    x = torch.randn((17, width), generator=generator, dtype=dtype)
    x[0].zero_()
    x[1] = (torch.arange(width) % 5 - 2).to(dtype)
    x[2] = -x[2].abs() - 1
    x[3].fill_(float("inf"))
    x[4].fill_(float("-inf"))
    x[5, 0::3] = float("nan")
    x[5, 1::3] = float("inf")
    x[5, 2::3] = float("-inf")
    x[6].fill_(float("nan"))
    x[7].zero_()
    x[7, 1::2] = -0.0
    # The final row is outside the two complete eight-row blocks.
    x[-1] = torch.arange(width - 1, -1, -1).to(dtype)
    return x.to(DEVICE)


@skipUnlessCuteAvailable("requires CuTe DSL")
@onlyBackends(["cute"])
@pytest.mark.parametrize(
    "dtype,width,k,largest,lanes",
    [
        (torch.bfloat16, 16, 1, True, 1),
        (torch.float16, 16, 3, False, 4),
        (torch.bfloat16, 65, 3, True, 4),
        (torch.float16, 65, 32, False, 16),
        (torch.bfloat16, 1024, 32, True, 16),
        (torch.float16, 1024, 32, False, 32),
        (torch.bfloat16, 65, 1, False, 1),
        (torch.float16, 65, 3, True, 32),
        (torch.bfloat16, 16, 3, False, 16),
        (torch.float16, 1024, 1, True, 4),
    ],
)
def test_direct_topk(
    dtype: torch.dtype, width: int, k: int, largest: bool, lanes: int
) -> None:
    x = _inputs(width, dtype)
    original = x.clone()
    code, output = code_and_output(
        _row_topk,
        (x, k, largest),
        block_sizes=[8],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=8,
        cute_topk_vector_width=8,
    )
    _assert_register_selection(code)
    assert "sort_rank" not in code
    _check_topk(x, original, output, k, largest)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _topk_and_copy(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = x.size(0)
    values = torch.empty((rows, 3), dtype=x.dtype, device=x.device)
    indices = torch.empty((rows, 3), dtype=torch.int64, device=x.device)
    copied = torch.empty_like(x)
    for row in hl.tile(rows):
        vals, idx = torch.topk(x[row, :], 3, dim=-1, largest=True, sorted=True)
        values[row, :] = vals
        indices[row, :] = idx
        copied[row, :] = x[row, :]
    return values, indices, copied


@skipUnlessCuteAvailable("requires CuTe DSL")
@onlyBackends(["cute"])
def test_direct_topk_preserves_other_stores() -> None:
    """Register selection composes with an independent full-width output."""
    x = torch.arange(48, dtype=torch.bfloat16, device=DEVICE).reshape(3, 16)
    original = x.clone()
    code, (values, indices, copied) = code_and_output(
        _topk_and_copy, (x,), block_sizes=[8]
    )
    _assert_register_selection(code)
    _check_topk(x, original, (values, indices), 3, True)
    torch.testing.assert_close(copied, original, rtol=0, atol=0)


# Cached launcher alignment transitions.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _alignment_relaunch_topk(
    x: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    rows = x.size(0)
    values = torch.empty((rows, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((rows, k), dtype=torch.int64, device=x.device)
    for row in hl.tile(rows):
        vals, idx = torch.topk(x[row, :], k, dim=-1, sorted=True)
        values[row, :] = vals
        indices[row, :] = idx
    return values, indices


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_same_topk_launcher_aligned_shifted_aligned() -> None:
    rows, width, k = 17, 64, 3
    storage = torch.randn(rows * width + 1, dtype=torch.bfloat16, device=DEVICE)
    bits = storage.view(torch.int16)
    bits[:8] = torch.tensor(
        [0x7FC1, -46, -32768, 0, -128, 0x7F80, 0x3F80, -16512],
        dtype=torch.int16,
        device=DEVICE,
    )
    original_storage = storage.clone()
    aligned = storage[:-1].view(rows, width)
    shifted = storage[1:].view(rows, width)
    assert aligned.shape == shifted.shape and aligned.stride() == shifted.stride()
    assert aligned.data_ptr() % 16 == 0 and shifted.data_ptr() % 16 == 2
    bound = _alignment_relaunch_topk.bind((aligned, k))
    config = bound.config_spec.default_config()
    config.config.update(
        cute_topk_lanes_per_row=16, cute_topk_rows_per_block=8, cute_topk_vector_width=8
    )
    # Reuse this exact generated host function to exercise launcher caches even
    # if Kernel.bind independently specializes input alignment.
    compiled = bound.compile_config(config)
    for tensor in (aligned, shifted, aligned):
        original = tensor.clone()
        values, indices = compiled(tensor, k)
        expected = torch.topk(original, k, dim=-1, sorted=True).values
        torch.testing.assert_close(values, expected, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(
            values.view(torch.int16), original.gather(1, indices).view(torch.int16)
        )
        assert torch.equal(
            storage.view(torch.int16), original_storage.view(torch.int16)
        )


# Geometry top-k tests.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _extra_row_topk(
    x: torch.Tensor, k: int, largest: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    rows = x.size(0)
    values = torch.empty((rows, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((rows, k), dtype=torch.int64, device=x.device)
    for row in hl.tile(rows):
        vals, idx = torch.topk(x[row, :], k, dim=-1, largest=bool(largest), sorted=True)
        values[row, :] = vals
        indices[row, :] = idx
    return values, indices


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _extra_out_topk(
    x: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    k: int,
    largest: hl.constexpr,
) -> None:
    k = hl.specialize(k)
    for row in hl.tile(x.size(0)):
        vals, idx = torch.topk(x[row, :], k, dim=-1, largest=bool(largest), sorted=True)
        values[row, :] = vals
        indices[row, :] = idx


def _layout_input(
    rows: int, width: int, dtype: torch.dtype, padding: int, offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    stride = width + padding
    generator = torch.Generator().manual_seed(20260924)
    storage = torch.randn(offset + rows * stride, generator=generator, dtype=dtype)
    cpu_view = torch.as_strided(storage, (rows, width), (stride, 1), offset)
    # Include distinct positive/negative NaN payloads and both zero signs.
    words = (
        (0x7FC1, 0xFFD2, 0x8000, 0x0000, 0xFF80, 0x7F80, 0x3F80, 0xBF80)
        if dtype == torch.bfloat16
        else (0x7E01, 0xFE22, 0x8000, 0x0000, 0xFC00, 0x7C00, 0x3C00, 0xBC00)
    )
    bits = torch.tensor(
        [word if word < 0x8000 else word - 0x10000 for word in words],
        dtype=torch.int16,
    )
    cpu_view[0] = bits.view(dtype).repeat((width + 7) // 8)[:width]
    if rows > 1:
        cpu_view[1].zero_()
        cpu_view[1, 1::2] = -0.0
    storage = storage.to("cuda")
    return torch.as_strided(storage, (rows, width), (stride, 1), offset), storage


def _assert_topk_output(
    original: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    k: int,
    largest: bool = True,
    *,
    index_dtype: torch.dtype = torch.int64,
) -> None:
    assert values.shape == indices.shape == (original.size(0), k)
    assert values.dtype == original.dtype and indices.dtype == index_dtype
    assert bool(((indices >= 0) & (indices < original.size(1))).all())
    expected_values = torch.topk(
        original, k, dim=-1, largest=largest, sorted=True
    ).values
    torch.testing.assert_close(values, expected_values, rtol=0, atol=0, equal_nan=True)
    gathered = original.gather(1, indices.to(torch.int64))
    assert torch.equal(values.view(torch.int16), gathered.view(torch.int16))
    sorted_indices = indices.sort(dim=-1).values
    assert bool((sorted_indices[:, 1:] != sorted_indices[:, :-1]).all())


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "rows,width,k,dtype,lanes,block_rows,vector,padding,offset,output_vector",
    [
        pytest.param(3, 1, 1, torch.bfloat16, 1, 1, 1, 0, 0, 1, id="singleton-min-cta"),
        pytest.param(
            17, 65, 65, torch.float16, 8, 4, 2, 4, 1, 1, id="all-elements-odd-stride"
        ),
        pytest.param(
            35, 65, 3, torch.bfloat16, 32, 32, 4, 2, 1, 1, id="max-cta-odd-stride"
        ),
        pytest.param(9, 33, 17, torch.float16, 2, 2, 4, 0, 1, 1, id="vector4-offset"),
        pytest.param(
            7, 31, 1, torch.bfloat16, 8, 1, 1, 5, 0, 1, id="unit-vector-row-stride"
        ),
        pytest.param(
            17, 64, 3, torch.bfloat16, 16, 8, 2, 0, 0, 1, id="vector2-aligned"
        ),
        pytest.param(
            129, 64, 32, torch.bfloat16, 1, 128, 8, 0, 0, 1, id="one-lane-128-rows-tail"
        ),
        pytest.param(
            65, 128, 32, torch.float16, 2, 64, 8, 0, 0, 1, id="two-lanes-64-rows-tail"
        ),
        pytest.param(5, 64, 32, torch.bfloat16, 4, 4, 8, 0, 0, 2, id="output2-bf16"),
        pytest.param(5, 64, 6, torch.float16, 4, 4, 8, 0, 0, 2, id="output2-fp16-tail"),
        pytest.param(5, 64, 32, torch.bfloat16, 4, 4, 8, 0, 0, 4, id="output4-bf16"),
        pytest.param(
            5, 64, 12, torch.float16, 4, 4, 8, 0, 0, 4, id="output4-fp16-tail"
        ),
        pytest.param(5, 64, 32, torch.bfloat16, 4, 4, 8, 0, 0, 8, id="output8-bf16"),
        pytest.param(
            5, 64, 24, torch.float16, 4, 4, 8, 0, 0, 8, id="output8-fp16-tail"
        ),
        pytest.param(
            5, 64, 17, torch.float16, 4, 4, 8, 2, 1, 8, id="output8-odd-k-offset"
        ),
    ],
)
def test_direct_topk_layout_and_geometry(
    rows: int,
    width: int,
    k: int,
    dtype: torch.dtype,
    lanes: int,
    block_rows: int,
    vector: int,
    padding: int,
    offset: int,
    output_vector: int,
) -> None:
    x, storage = _layout_input(rows, width, dtype, padding, offset)
    original = x.clone()
    original_storage = storage.clone()
    code, (values, indices) = code_and_output(
        _extra_row_topk,
        (x, k, True),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=block_rows,
        cute_topk_vector_width=vector,
        cute_topk_output_vector_width=output_vector,
    )
    _assert_register_selection(code)
    assert ("cute.autovec_copy" in code) == (
        output_vector > 1 and k % output_vector == 0
    )
    _assert_topk_output(original, values, indices, k)
    assert torch.equal(storage.view(torch.int16), original_storage.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("value_mode", ["gather", "decode"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "selection_layout,key_dtype,k,lanes",
    [
        ("replicated", "int32", 32, 4),
        ("distributed", "int32", 32, 4),
        ("distributed", "float32_bits", 6, 2),
        ("distributed", "float32_bits", 8, 32),
    ],
)
def test_topk_output_alignment_changes_on_same_bound_kernel(
    dtype: torch.dtype,
    value_mode: str,
    index_dtype: torch.dtype,
    selection_layout: str,
    key_dtype: str,
    k: int,
    lanes: int,
) -> None:
    rows, width = 5, 64
    x, storage = _layout_input(rows, width, dtype, 0, 0)
    original_storage = storage.clone()
    value_storage = torch.empty(rows * k + 3, dtype=dtype, device=DEVICE)
    index_storage = torch.empty(rows * k + 3, dtype=index_dtype, device=DEVICE)
    values = value_storage[: rows * k].view(rows, k)
    indices = index_storage[: rows * k].view(rows, k)
    bound = _extra_out_topk._bind_isolated((x, values, indices, k, True))
    bound.set_config(
        helion.Config(
            block_sizes=[1],
            cute_topk_lanes_per_row=lanes,
            cute_topk_rows_per_block=4,
            cute_topk_vector_width=8,
            cute_topk_output_vector_width=8,
            cute_topk_value_mode=value_mode,
            cute_topk_key_dtype=key_dtype,
            cute_topk_rank_mode="ordinal",
            cute_topk_selection_layout=selection_layout,
        )
    )
    for value_offset, index_offset in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (0, 0)):
        value_storage.fill_(7)
        index_storage.fill_(-7)
        values = value_storage[value_offset : value_offset + rows * k].view(rows, k)
        indices = index_storage[index_offset : index_offset + rows * k].view(rows, k)
        bound(x, values, indices, k, True)
        _assert_topk_output(x, values, indices, k, index_dtype=index_dtype)
        assert bool((value_storage[:value_offset] == 7).all())
        assert bool((value_storage[value_offset + rows * k :] == 7).all())
        assert bool((index_storage[:index_offset] == -7).all())
        assert bool((index_storage[index_offset + rows * k :] == -7).all())
        assert torch.equal(
            storage.view(torch.int16), original_storage.view(torch.int16)
        )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "width,k,lanes,output_vector,key_dtype,rank_mode,value_mode,layout,padding,offset",
    [
        (1, 1, 1, 1, "int32", "signed", "gather", "replicated", 0, 0),
        (64, 32, 4, 2, "float32_bits", "ordinal", "decode", "replicated", 0, 0),
        (65, 65, 8, 8, "float32", "signed", "decode", "distributed", 2, 1),
        (128, 24, 8, 8, "float32_native", "ordinal", "decode", "distributed", 2, 1),
        (128, 8, 32, 8, "float32_native", "signed", "gather", "distributed", 0, 0),
        (64, 6, 2, 2, "int32", "ordinal", "decode", "distributed", 0, 0),
        (128, 32, 8, 4, "float32_native", "ordinal", "decode", "replicated", 0, 0),
    ],
)
def test_topk_index_output_dtype_preserves_selected_bits(
    index_dtype: torch.dtype,
    dtype: torch.dtype,
    largest: bool,
    width: int,
    k: int,
    lanes: int,
    output_vector: int,
    key_dtype: str,
    rank_mode: str,
    value_mode: str,
    layout: str,
    padding: int,
    offset: int,
) -> None:
    rows = 5
    x, storage = _layout_input(rows, width, dtype, padding, offset)
    original_storage = storage.clone()
    values = torch.full((rows, k), 7, dtype=dtype, device=DEVICE)
    indices = torch.full((rows, k), -7, dtype=index_dtype, device=DEVICE)
    code, _output = code_and_output(
        _extra_out_topk,
        (x, values, indices, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=output_vector,
        cute_topk_value_mode=value_mode,
        cute_topk_key_dtype=key_dtype,
        cute_topk_rank_mode=rank_mode,
        cute_topk_selection_layout=layout,
        cute_topk_sort_network="compact_pruned",
    )
    # Singleton top-k simplifies to a value copy and index zero.
    if width > 1:
        _assert_register_selection(code)
    _assert_topk_output(x, values, indices, k, largest, index_dtype=index_dtype)
    assert torch.equal(storage.view(torch.int16), original_storage.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_topk_narrow_indices_preserve_wide_address_math(
    index_dtype: torch.dtype,
) -> None:
    x, storage = _layout_input(1, 8, torch.bfloat16, 0, 0)
    x = torch.as_strided(x, (1, 8), (2**35, 1))
    original_storage = storage.clone()
    values = torch.empty((1, 8), dtype=x.dtype, device=DEVICE)
    indices = torch.empty((1, 8), dtype=index_dtype, device=DEVICE)
    code, _output = code_and_output(
        _extra_out_topk,
        (x, values, indices, 8, True),
        block_sizes=[1],
        cute_topk_lanes_per_row=1,
        cute_topk_rows_per_block=1,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=8,
    )
    assert "topk_row * cutlass.Int64(34359738368)" in code
    _assert_topk_output(x, values, indices, 8, index_dtype=index_dtype)
    assert torch.equal(storage.view(torch.int16), original_storage.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("rank_mode", ["signed", "ordinal"])
@pytest.mark.parametrize(
    "width,k,output_vector,padding,offset",
    [(64, 64, 8, 0, 0), (65, 65, 1, 2, 1), (64, 17, 4, 2, 1)],
)
def test_topk_decode_preserves_selected_bits(
    dtype: torch.dtype,
    largest: bool,
    rank_mode: str,
    width: int,
    k: int,
    output_vector: int,
    padding: int,
    offset: int,
) -> None:
    x, storage = _layout_input(5, width, dtype, padding, offset)
    original = x.clone()
    original_storage = storage.clone()
    code, (values, indices) = code_and_output(
        _extra_row_topk,
        (x, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=4,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=output_vector,
        cute_topk_value_mode="decode",
        cute_topk_rank_mode=rank_mode,
    )
    assert "topk_value_decodable" in code
    _assert_topk_output(original, values, indices, k, largest)
    assert torch.equal(storage.view(torch.int16), original_storage.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("rank_mode", ["signed", "ordinal"])
@pytest.mark.parametrize(
    "key_dtype,width,k,output_vector,padding,offset,value_mode",
    [
        ("float32", 512, 32, 8, 0, 0, "decode"),
        ("float32", 65, 65, 1, 2, 1, "gather"),
        ("float32", 1024, 32, 4, 0, 1, "decode"),
        ("float32_bits", 1024, 32, 8, 0, 0, "decode"),
        ("float32_bits", 65, 65, 1, 2, 1, "gather"),
        ("float32_bits", 1, 1, 1, 0, 0, "decode"),
    ],
)
def test_topk_float_keys_preserve_selected_bits(
    dtype: torch.dtype,
    largest: bool,
    rank_mode: str,
    key_dtype: str,
    width: int,
    k: int,
    output_vector: int,
    padding: int,
    offset: int,
    value_mode: str,
) -> None:
    x, storage = _layout_input(5, width, dtype, padding, offset)
    original = x.clone()
    original_storage = storage.clone()
    code, (values, indices) = code_and_output(
        _extra_row_topk,
        (x, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=4,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=output_vector,
        cute_topk_value_mode=value_mode,
        cute_topk_key_dtype=key_dtype,
        cute_topk_rank_mode=rank_mode,
    )
    _assert_register_selection(code)
    _assert_topk_output(original, values, indices, k, largest)
    assert torch.equal(storage.view(torch.int16), original_storage.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "width,k,lanes,block_rows,padding,offset,key_dtype,rank_mode,value_mode",
    [
        (1, 1, 1, 1, 0, 0, "int32", "ordinal", "decode"),
        (65, 33, 1, 8, 2, 1, "int32", "ordinal", "decode"),
        (65, 6, 2, 8, 2, 1, "int32", "ordinal", "decode"),
        (128, 32, 2, 64, 0, 0, "float32_bits", "ordinal", "decode"),
        (128, 32, 4, 32, 0, 0, "float32", "signed", "decode"),
        (128, 64, 8, 16, 0, 0, "int32", "ordinal", "gather"),
        (128, 24, 8, 16, 2, 1, "float32_bits", "ordinal", "decode"),
        (128, 12, 8, 16, 0, 0, "int32", "signed", "decode"),
        (256, 128, 16, 8, 0, 0, "float32_bits", "ordinal", "decode"),
        (128, 64, 32, 4, 0, 0, "float32", "signed", "decode"),
        (65, 65, 4, 32, 2, 1, "float32", "signed", "gather"),
        (64, 7, 8, 4, 0, 0, "int32", "ordinal", "decode"),
        (128, 17, 16, 4, 0, 0, "float32_bits", "ordinal", "decode"),
        (65, 31, 32, 1, 2, 1, "float32", "signed", "decode"),
        (64, 3, 32, 1, 0, 0, "int32", "ordinal", "gather"),
    ],
)
def test_topk_distributed_selection(
    dtype: torch.dtype,
    largest: bool,
    width: int,
    k: int,
    lanes: int,
    block_rows: int,
    padding: int,
    offset: int,
    key_dtype: str,
    rank_mode: str,
    value_mode: str,
) -> None:
    x, storage = _layout_input(block_rows + 1, width, dtype, padding, offset)
    original = x.clone()
    original_storage = storage.clone()
    code, (values, indices) = code_and_output(
        _extra_row_topk,
        (x, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=block_rows,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=8,
        cute_topk_value_mode=value_mode,
        cute_topk_key_dtype=key_dtype,
        cute_topk_rank_mode=rank_mode,
        cute_topk_selection_layout="distributed",
    )
    _assert_register_selection(code)
    vector = min(8, lanes, max(1, (1 << (k - 1).bit_length()) // lanes))
    assert ("cute.autovec_copy" in code) == (vector > 1 and k % vector == 0)
    _assert_topk_output(original, values, indices, k, largest)
    assert torch.equal(storage.view(torch.int16), original_storage.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "width,k,lanes,input_vector,key_dtype,rank_mode,value_mode",
    [
        (8, 8, 8, 1, "int32", "ordinal", "decode"),
        (13, 9, 16, 1, "float32_bits", "signed", "decode"),
        (64, 32, 32, 2, "float32", "ordinal", "gather"),
        (128, 32, 16, 8, "int32", "ordinal", "decode"),
        (128, 32, 32, 4, "float32_bits", "ordinal", "decode"),
        (65, 33, 32, 4, "float32", "signed", "decode"),
    ],
)
def test_topk_distributed_growing_selection(
    dtype: torch.dtype,
    largest: bool,
    width: int,
    k: int,
    lanes: int,
    input_vector: int,
    key_dtype: str,
    rank_mode: str,
    value_mode: str,
) -> None:
    # Misaligned, strided rows and a partial CTA exercise the smaller fragment
    # with both real values and padding through the growing subgroup stages.
    x, storage = _layout_input(5, width, dtype, 2, 1)
    original = storage.clone()
    code, (values, indices) = code_and_output(
        _extra_row_topk,
        (x, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=input_vector,
        cute_topk_output_vector_width=8,
        cute_topk_key_dtype=key_dtype,
        cute_topk_rank_mode=rank_mode,
        cute_topk_value_mode=value_mode,
        cute_topk_selection_layout="distributed",
    )
    _assert_register_selection(code)
    _assert_topk_output(x, values, indices, k, largest)
    assert torch.equal(storage.view(torch.int16), original.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "width,k,lanes,input_vector,key_dtype,network",
    [
        (256, 8, 16, 8, "int32", "batcher"),
        (256, 8, 32, 8, "float32", "compact"),
        (512, 8, 32, 8, "float32_bits", "compact_pruned"),
        (128, 3, 16, 8, "int32", "compact_pruned"),
        (33, 1, 32, 4, "float32", "batcher"),
        (3, 3, 32, 1, "float32_bits", "compact"),
    ],
)
def test_topk_distributed_more_lanes_than_outputs(
    dtype: torch.dtype,
    largest: bool,
    width: int,
    k: int,
    lanes: int,
    input_vector: int,
    key_dtype: str,
    network: str,
) -> None:
    x, storage = _layout_input(5, width, dtype, 2, 1)
    original = storage.clone()
    code, (values, indices) = code_and_output(
        _extra_row_topk,
        (x, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=input_vector,
        cute_topk_output_vector_width=8,
        cute_topk_key_dtype=key_dtype,
        cute_topk_rank_mode="ordinal",
        cute_topk_value_mode="decode",
        cute_topk_selection_layout="distributed",
        cute_topk_sort_network=network,
    )
    _assert_register_selection(code)
    assert "cute.autovec_copy" not in code
    assert f"topk_output_col < cutlass.Int32({k})" in code
    _assert_topk_output(x, values, indices, k, largest)
    assert torch.equal(storage.view(torch.int16), original.view(torch.int16))


@pytest.fixture
def topk_spec(monkeypatch: pytest.MonkeyPatch) -> ConfigSpec:
    # Keep configuration validation independent of CUDA device discovery.
    monkeypatch.setattr("helion.autotuner.config_spec.get_num_xcd", lambda device: 1)
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(10, 0), num_sm=1)
    spec.enable_cute_topk_search()
    return spec


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("network", ["compact", "compact_pruned"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "width,k,lanes,layout,key_dtype,rank_mode,value_mode",
    [
        (64, 32, 1, "replicated", "float32_bits", "ordinal", "decode"),
        (128, 32, 2, "replicated", "float32", "signed", "decode"),
        (128, 32, 16, "distributed", "int32", "ordinal", "decode"),
        (65, 33, 2, "replicated", "int32", "signed", "gather"),
        (512, 8, 2, "replicated", "float32_bits", "ordinal", "decode"),
    ],
)
def test_topk_compact_selection_networks(
    network: str,
    dtype: torch.dtype,
    largest: bool,
    width: int,
    k: int,
    lanes: int,
    layout: str,
    key_dtype: str,
    rank_mode: str,
    value_mode: str,
) -> None:
    x, storage = _layout_input(5, width, dtype, 2, 1)
    original = storage.clone()
    with patch(
        "helion._compiler.cute.selection_network.trace_selection_network",
        wraps=trace_selection_network,
    ) as trace:
        code, (values, indices) = code_and_output(
            _extra_row_topk,
            (x, k, largest),
            block_sizes=[1],
            cute_topk_lanes_per_row=lanes,
            cute_topk_rows_per_block=4,
            cute_topk_vector_width=8,
            cute_topk_output_vector_width=4,
            cute_topk_key_dtype=key_dtype,
            cute_topk_rank_mode=rank_mode,
            cute_topk_value_mode=value_mode,
            cute_topk_selection_layout=layout,
            cute_topk_sort_network=network,
        )
    assert trace.call_args_list
    assert all(call.args[4] == network for call in trace.call_args_list)
    _assert_register_selection(code)
    _assert_topk_output(x, values, indices, k, largest)
    assert torch.equal(storage.view(torch.int16), original.view(torch.int16))


@pytest.mark.parametrize(
    "key,value",
    [
        ("cute_topk_lanes_per_row", True),
        ("cute_topk_rows_per_block", False),
        ("cute_topk_vector_width", True),
        ("cute_topk_lanes_per_row", 0),
        ("cute_topk_rows_per_block", 3),
        ("cute_topk_vector_width", 16),
        ("cute_topk_vector_width", 8.0),
        ("cute_topk_output_vector_width", True),
        ("cute_topk_output_vector_width", 16),
        ("cute_topk_value_mode", True),
        ("cute_topk_value_mode", 1),
        ("cute_topk_value_mode", "unknown"),
        ("cute_topk_key_dtype", True),
        ("cute_topk_key_dtype", 1),
        ("cute_topk_key_dtype", "unknown"),
        ("cute_topk_rank_mode", True),
        ("cute_topk_rank_mode", 1),
        ("cute_topk_rank_mode", "unknown"),
        ("cute_topk_selection_layout", True),
        ("cute_topk_selection_layout", 1),
        ("cute_topk_selection_layout", "unknown"),
        ("cute_topk_sort_network", True),
        ("cute_topk_sort_network", 1),
        ("cute_topk_sort_network", "unknown"),
    ],
)
def test_topk_config_rejects_noninteger_or_unsupported_choices(
    topk_spec: ConfigSpec, key: str, value: object
) -> None:
    config = helion.Config.from_dict({key: value})
    with pytest.raises(exc.InvalidConfig, match="must be one of"):
        topk_spec.normalize(config)


def test_topk_config_is_scoped_to_matched_cute_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("helion.autotuner.config_spec.get_num_xcd", lambda device: 1)
    cute_spec = ConfigSpec(
        backend=CuteBackend(), target_device_capability=(10, 0), num_sm=1
    )
    with pytest.raises(exc.InvalidConfig, match="compatible top-k root"):
        cute_spec.normalize({"cute_topk_lanes_per_row": 16})
    triton_spec = ConfigSpec(backend=TritonBackend(), num_sm=1)
    with pytest.raises(exc.InvalidConfig, match="Unsupported config keys"):
        triton_spec.normalize({"cute_topk_lanes_per_row": 16})


@pytest.mark.parametrize("lanes,rows", [(2, 4), (1, 128), (2, 64), (8, 128), (16, 64)])
def test_topk_geometry_survives_flat_config_roundtrip(
    topk_spec: ConfigSpec, lanes: int, rows: int
) -> None:
    config = helion.Config(
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=rows,
        cute_topk_vector_width=1,
    )
    topk_spec.normalize(config)
    generation = ConfigGeneration(topk_spec)
    assert generation.unflatten(generation.flatten(config)) == config


@pytest.mark.parametrize("vector", [1, 2, 4, 8])
def test_topk_output_vector_survives_flat_config_roundtrip(
    topk_spec: ConfigSpec, vector: int
) -> None:
    config = helion.Config(cute_topk_output_vector_width=vector)
    topk_spec.normalize(config)
    generation = ConfigGeneration(topk_spec)
    assert generation.unflatten(generation.flatten(config)) == config


@pytest.mark.parametrize("mode", ["gather", "decode"])
def test_topk_value_mode_survives_flat_config_roundtrip(
    topk_spec: ConfigSpec, mode: str
) -> None:
    assert topk_spec.default_config()["cute_topk_value_mode"] == "gather"
    config = helion.Config(cute_topk_value_mode=mode)
    topk_spec.normalize(config)
    generation = ConfigGeneration(topk_spec)
    assert generation.unflatten(generation.flatten(config)) == config


def test_topk_invalid_value_mode_normalizes_to_gather(topk_spec: ConfigSpec) -> None:
    config = helion.Config(cute_topk_value_mode=False)
    topk_spec.normalize(config, _fix_invalid=True)
    assert config["cute_topk_value_mode"] == "gather"


@pytest.mark.parametrize("key_dtype", ["int32", "float32", "float32_bits"])
def test_topk_key_dtype_survives_flat_config_roundtrip(
    topk_spec: ConfigSpec, key_dtype: str
) -> None:
    assert topk_spec.default_config()["cute_topk_key_dtype"] == "int32"
    config = helion.Config(cute_topk_key_dtype=key_dtype)
    topk_spec.normalize(config)
    generation = ConfigGeneration(topk_spec)
    assert generation.unflatten(generation.flatten(config)) == config


def test_topk_invalid_key_dtype_normalizes_to_int32(topk_spec: ConfigSpec) -> None:
    config = helion.Config(cute_topk_key_dtype=False)
    topk_spec.normalize(config, _fix_invalid=True)
    assert config["cute_topk_key_dtype"] == "int32"


@pytest.mark.parametrize("rank_mode", ["signed", "ordinal"])
def test_topk_rank_mode_survives_flat_config_roundtrip(
    topk_spec: ConfigSpec, rank_mode: str
) -> None:
    assert topk_spec.default_config()["cute_topk_rank_mode"] == "signed"
    config = helion.Config(cute_topk_rank_mode=rank_mode)
    topk_spec.normalize(config)
    generation = ConfigGeneration(topk_spec)
    assert generation.unflatten(generation.flatten(config)) == config


def test_topk_invalid_rank_mode_normalizes_to_signed(topk_spec: ConfigSpec) -> None:
    config = helion.Config(cute_topk_rank_mode=False)
    topk_spec.normalize(config, _fix_invalid=True)
    assert config["cute_topk_rank_mode"] == "signed"


@pytest.mark.parametrize("layout", ["replicated", "distributed"])
def test_topk_selection_layout_survives_flat_config_roundtrip(
    topk_spec: ConfigSpec, layout: str
) -> None:
    assert topk_spec.default_config()["cute_topk_selection_layout"] == "replicated"
    config = helion.Config(cute_topk_selection_layout=layout)
    topk_spec.normalize(config)
    generation = ConfigGeneration(topk_spec)
    assert generation.unflatten(generation.flatten(config)) == config


def test_topk_invalid_selection_layout_normalizes_to_replicated(
    topk_spec: ConfigSpec,
) -> None:
    config = helion.Config(cute_topk_selection_layout=False)
    topk_spec.normalize(config, _fix_invalid=True)
    assert config["cute_topk_selection_layout"] == "replicated"


@pytest.mark.parametrize("network", ["batcher", "compact", "compact_pruned"])
def test_topk_sort_network_config_roundtrip(
    topk_spec: ConfigSpec, network: str
) -> None:
    assert topk_spec.default_config()["cute_topk_sort_network"] == "batcher"
    config = helion.Config(cute_topk_sort_network=network)
    topk_spec.normalize(config)
    generation = ConfigGeneration(topk_spec)
    assert generation.unflatten(generation.flatten(config)) == config
    assert helion.Config.from_json(config.to_json()) == config


def test_topk_invalid_sort_network_normalizes_to_batcher(topk_spec: ConfigSpec) -> None:
    config = helion.Config(cute_topk_sort_network=False)
    topk_spec.normalize(config, _fix_invalid=True)
    assert config["cute_topk_sort_network"] == "batcher"


@pytest.mark.parametrize("lanes,rows", [(16, 128), (32, 64), (32, 128)])
def test_topk_config_limits_threads_per_block(
    topk_spec: ConfigSpec, lanes: int, rows: int
) -> None:
    config = helion.Config(
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=rows,
    )
    with pytest.raises(exc.InvalidConfig, match="must not exceed 1024 threads"):
        topk_spec.normalize(config)
    topk_spec.normalize(config, _fix_invalid=True)
    assert config["cute_topk_lanes_per_row"] == lanes
    assert config["cute_topk_rows_per_block"] == 1024 // lanes
    topk_spec.normalize(config)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("selection_layout", ["replicated", "distributed"])
@pytest.mark.parametrize("sort_network", ["batcher", "compact_pruned"])
@pytest.mark.parametrize(
    "width,k,lanes,output_vector,padding,offset,value_mode",
    [
        (64, 64, 4, 8, 0, 0, "decode"),
        (65, 65, 4, 1, 2, 1, "gather"),
        (128, 32, 16, 4, 0, 0, "decode"),
        (128, 8, 32, 4, 0, 0, "decode"),
    ],
)
def test_topk_native_float_preserves_all_bits(
    dtype: torch.dtype,
    largest: bool,
    selection_layout: str,
    sort_network: str,
    width: int,
    lanes: int,
    k: int,
    output_vector: int,
    padding: int,
    offset: int,
    value_mode: str,
) -> None:
    rows = (65536 + width - 1) // width + 2
    x, storage = _layout_input(rows, width, dtype, padding, offset)
    words = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    data = words.view(dtype).repeat(2)[: (rows - 2) * width].reshape(rows - 2, width)
    x[2:].copy_(data)
    original = x.clone()
    code, (values, indices) = code_and_output(
        _extra_row_topk,
        (x, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=8,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=output_vector,
        cute_topk_value_mode=value_mode,
        cute_topk_key_dtype="float32_native",
        cute_topk_selection_layout=selection_layout,
        cute_topk_sort_network=sort_network,
    )
    assert "topk_native_key" in code
    _assert_topk_output(original, values, indices, k, largest)
    assert torch.equal(x.view(torch.int16), original.view(torch.int16))


# Safety top-k tests.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _allocating_topk(
    x: torch.Tensor,
    k: int,
    largest: hl.constexpr,
    index_dtype: torch.dtype = torch.int64,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    values = torch.empty((x.size(0), k), dtype=x.dtype, device=x.device)
    indices = torch.empty((x.size(0), k), dtype=index_dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        vals, idx = torch.topk(x[row, :], k, dim=-1, largest=bool(largest), sorted=True)
        values[row, :] = vals
        indices[row, :] = idx
    return values, indices


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _transformed_indices_topk(
    x: torch.Tensor,
    k: int,
    largest: hl.constexpr,
    index_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    values = torch.empty((x.size(0), k), dtype=x.dtype, device=x.device)
    indices = torch.empty((x.size(0), k), dtype=index_dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        vals, idx = torch.topk(x[row, :], k, dim=-1, largest=bool(largest), sorted=True)
        values[row, :] = vals
        indices[row, :] = idx + 1
    return values, indices


def _code(
    rows: int,
    cols: int,
    stride: int,
    k: int,
    output_vector: int = 1,
    *,
    dtype: torch.dtype = torch.bfloat16,
    largest: bool = True,
    value_mode: str = "gather",
    key_dtype: str = "int32",
    rank_mode: str = "signed",
    selection_layout: str = "replicated",
    lanes: int = 16,
    input_vector: int = 8,
    sort_network: str = "batcher",
    key_encoder: str = "dsl",
    defer_value_gathers: bool = False,
    merge_schedule: str = "sequential",
    index_dtype: torch.dtype = torch.int64,
    kernel: helion.Kernel[Any] = _allocating_topk,
) -> str:
    with (
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 0)),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 0),
        ),
        patch("helion.language.loops.use_tileir_tunables", return_value=False),
        patch("helion.language.loops._supports_warp_specialize", return_value=True),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
        patch("helion._compat._min_dot_size", return_value=(16, 16, 16)),
        patch("helion._compat._is_hip", return_value=False),
    ):
        with FakeTensorMode():
            x = torch.empty_strided(
                (rows, cols), (stride, 1), dtype=dtype, device=torch.device("cpu")
            )
        bound = kernel._bind_isolated((x, k, largest, index_dtype))
        config = bound.config_spec.default_config()
        config.config.update(
            cute_topk_lanes_per_row=lanes,
            cute_topk_rows_per_block=8,
            cute_topk_vector_width=input_vector,
            cute_topk_output_vector_width=output_vector,
            cute_topk_value_mode=value_mode,
            cute_topk_key_dtype=key_dtype,
            cute_topk_rank_mode=rank_mode,
            cute_topk_selection_layout=selection_layout,
            cute_topk_sort_network=sort_network,
            cute_topk_key_encoder=key_encoder,
            cute_topk_defer_value_gathers=defer_value_gathers,
            cute_topk_merge_schedule=merge_schedule,
        )
        return bound.to_triton_code(config)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("vector", [1, 2, 4, 8])
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
def test_topk_index_output_dtype_codegen(
    index_dtype: torch.dtype, vector: int, layout: str
) -> None:
    code = _code(
        5,
        128,
        128,
        64,
        vector,
        index_dtype=index_dtype,
        selection_layout=layout,
        lanes=8,
    )
    dtype = "cutlass.Int32" if index_dtype == torch.int32 else "cutlass.Int64"
    element_bytes = 4 if index_dtype == torch.int32 else 8
    assert f"indices[topk_row, topk_output_col] = {dtype}(topk_selected_index)" in code
    if vector > 1:
        assert f"topk_index_fragment = cute.make_rmem_tensor({vector}, {dtype})" in code
        assert (
            f"indices.iterator.alignment >= {min(16, element_bytes * vector)}" in code
        )
    assert "topk_row = cutlass.Int32(" in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rows,stride", [(1, 2**35), (268435457, 0)])
def test_topk_output_dtype_does_not_narrow_addresses(
    index_dtype: torch.dtype, rows: int, stride: int
) -> None:
    code = _code(rows, 8, stride, 8, 2, index_dtype=index_dtype)
    assert "cutlass.Int64(cute.arch.block_idx()[0])" in code
    assert f"topk_row * cutlass.Int64({stride})" in code
    assert "topk_output_offset = topk_row * cutlass.Int64(8)" in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("index_dtype", [torch.int16, torch.float32])
def test_topk_fragment_supports_other_index_store_dtypes(
    index_dtype: torch.dtype,
) -> None:
    code = _code(5, 64, 64, 32, index_dtype=index_dtype)
    assert "row_selected_indices" in code
    assert "sort_rank" not in code


@skipUnlessCuteAvailable("requires CuTe DSL")
def test_topk_fragment_supports_arithmetic_before_index_narrowing() -> None:
    code = _code(
        5, 64, 64, 32, index_dtype=torch.int32, kernel=_transformed_indices_topk
    )
    assert "row_selected_indices" in code
    assert "sort_rank" not in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "index_dtype,transform",
    [(torch.int16, False), (torch.float32, False), (torch.int32, True)],
)
def test_topk_composition_index_store_casts(
    index_dtype: torch.dtype, transform: bool
) -> None:
    x = torch.arange(64, device=DEVICE, dtype=torch.bfloat16).repeat(5, 1)
    kernel = _transformed_indices_topk if transform else _allocating_topk
    code, (values, indices) = code_and_output(
        kernel, (x, 32, True, index_dtype), **_composition_config(8, "distributed")
    )
    _assert_composed_register_selection(code, "distributed")
    expected = torch.topk(x, 32, dim=-1)
    torch.testing.assert_close(values, expected.values, rtol=0, atol=0)
    torch.testing.assert_close(
        indices, (expected.indices + int(transform)).to(index_dtype), rtol=0, atol=0
    )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize(
    "rows,cols,stride,k,wide",
    [
        (17, 8, 8, 3, False),
        (1, 8, 2**35, 3, True),
        (65536, 8, 32768, 3, False),
        (65537, 8, 32768, 3, True),
        (268435457, 8, 0, 8, True),
    ],
)
def test_topk_wide_address_codegen(
    rows: int, cols: int, stride: int, k: int, wide: bool
) -> None:
    code = _code(rows, cols, stride, k)
    dtype = "cutlass.Int64" if wide else "cutlass.Int32"
    assert f"{dtype}(cute.arch.block_idx()[0])" in code
    assert f"* {dtype}({stride})" in code
    _assert_register_selection(code)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize(
    "rows,cols,stride,k,vector,wide",
    [
        (17, 64, 64, 32, 2, False),
        (17, 64, 64, 32, 4, False),
        (17, 64, 64, 32, 8, False),
        (1, 8, 2**35, 8, 8, True),
        (268435457, 8, 0, 8, 8, True),
    ],
)
def test_topk_output_vector_address_codegen(
    rows: int, cols: int, stride: int, k: int, vector: int, wide: bool
) -> None:
    code = _code(rows, cols, stride, k, vector)
    dtype = "cutlass.Int64" if wide else "cutlass.Int32"
    assert f"topk_row * {dtype}({k}) + {dtype}(topk_output_col)" in code
    assert f"values.iterator.alignment >= {2 * vector}" in code
    assert "indices.iterator.alignment >= 16" in code
    assert code.count("cute.autovec_copy") == 2
    # Preserve the dynamic offset's divisibility through pointer addition.
    # The address arithmetic must retain its selected width before the hint.
    offset = f"topk_row * {dtype}({k}) + {dtype}(topk_output_col)"
    assumption = f"cute.assume(topk_output_offset, divby={vector})"
    assert code.index(offset) < code.index(assumption)
    assert code.index(assumption) < code.index("values.iterator + topk_output_offset")
    assert code.index(assumption) < code.index("indices.iterator + topk_output_offset")
    assert k % vector == 0
    assert f"topk_lane * cutlass.Int32({vector})" in code


@skipUnlessCuteAvailable("requires CuTe DSL")
def test_topk_odd_output_stride_uses_scalar_stores() -> None:
    code = _code(17, 64, 64, 17, 8)
    _assert_register_selection(code)
    assert "cute.autovec_copy" not in code
    assert "cute.assume(topk_output_offset" not in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("rank_mode", ["signed", "ordinal"])
@pytest.mark.parametrize("key_encoder", ["dsl", "asm"])
def test_topk_decode_all_16bit_inputs(
    dtype: torch.dtype, largest: bool, rank_mode: str, key_encoder: str
) -> None:
    code = _code(
        1,
        64,
        64,
        32,
        dtype=dtype,
        largest=largest,
        value_mode="decode",
        rank_mode=rank_mode,
        key_encoder=key_encoder,
    )
    tree = ast.parse(code)
    encoder = None
    if key_encoder == "asm":
        packed_assignment = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "topk_packed"
        )
        assert isinstance(packed_assignment.value, ast.Call)
        encoder_call = packed_assignment.value
        assert ast.unparse(encoder_call.func).startswith("_cute_encode_ordered_topk_")
        assert [ast.literal_eval(arg) for arg in encoder_call.args[2:]] == [
            6,
            largest,
            rank_mode,
            0x7F80 if dtype == torch.bfloat16 else 0x7C00,
        ]
    else:
        encode_body = next(
            node.body
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and any(
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "topk_magnitude"
                for stmt in node.body
            )
        )
        encode_start = next(
            i
            for i, stmt in enumerate(encode_body)
            if isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "topk_magnitude"
        )
        encode_end = next(
            i
            for i, stmt in enumerate(encode_body)
            if isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "topk_packed"
        )
        encoder = compile(
            ast.Module(
                body=encode_body[encode_start : encode_end + 1], type_ignores=[]
            ),
            "<topk-encode>",
            "exec",
        )
    names = {
        "topk_value_rank",
        "topk_value_sign",
        "topk_value_magnitude",
        "topk_value_bits",
        "topk_value_decodable",
    }
    assignments: list[ast.stmt] = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in names
    ]
    assert len(assignments) == (3 if rank_mode == "ordinal" else 5) + (not largest)
    decoder = compile(
        ast.Module(body=assignments, type_ignores=[]), "<topk-decode>", "exec"
    )
    infinity = 0x7F80 if dtype == torch.bfloat16 else 0x7C00
    namespace = {"cutlass": SimpleNamespace(Int32=int)}
    ranks = []
    for word in range(1 << 16):
        magnitude = word & 0x7FFF
        rank = magnitude if word < 32768 else -magnitude - (rank_mode == "ordinal")
        if magnitude > infinity:
            rank = 32767
        if not largest:
            rank = -rank
        packed = (rank << 6) | (63 - (word & 63))
        if encoder is not None:
            encoded = {
                "topk_bits": word if word < 32768 else word - 65536,
                "topk_col": word & 63,
            }
            exec(encoder, namespace, encoded)
            assert encoded["topk_packed"] == packed
        assert packed & 63 == 63 - (word & 63)
        ranks.append(packed >> 6)
        local = {"topk_local": [packed], "topk_j": 0}
        exec(decoder, namespace, local)
        decodable = magnitude <= infinity and (rank_mode == "ordinal" or magnitude != 0)
        assert local["topk_value_decodable"] == decodable
        if decodable:
            # Uint16 truncates the recovered signed representation.
            bits = local["topk_value_bits"]
            assert isinstance(bits, int)
            assert bits & 65535 == word
    # These bounds justify both Float32 key guards for either ranking mode.
    assert min(ranks) >= -32767 and max(ranks) <= 32767
    words = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = words.view(dtype).float()
    finite_or_inf = ~torch.isnan(values)
    order = torch.tensor(ranks)[finite_or_inf].argsort(descending=True)
    ordered_values = values[finite_or_inf][order]
    assert bool(
        (
            ordered_values[:-1] >= ordered_values[1:]
            if largest
            else ordered_values[:-1] <= ordered_values[1:]
        ).all()
    )
    if rank_mode == "ordinal":
        assert ranks[0] == 0
        assert ranks[32768] == (-1 if largest else 1)


@skipUnlessCuteAvailable("requires CuTe DSL")
def test_topk_decode_keeps_wide_vector_output_addresses() -> None:
    code = _code(268435457, 8, 0, 8, 8, value_mode="decode")
    assert "topk_row * cutlass.Int64(8) + cutlass.Int64(topk_output_col)" in code
    assert code.count("topk_value_decodable =") == 2
    assert "values.iterator.alignment >= 16" in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize(
    "cols,key_dtype,selected_dtype",
    [
        (512, "int32", "Int32"),
        (512, "float32", "Float32"),
        (513, "float32", "Int32"),
        (1024, "float32", "Int32"),
        (1, "float32_bits", "Float32"),
        (1024, "float32_bits", "Float32"),
        (16384, "float32_bits", "Float32"),
        (16385, "float32_bits", "Int32"),
        (32768, "float32_bits", "Int32"),
    ],
)
def test_topk_float_key_codegen_precision_guard(
    cols: int, key_dtype: str, selected_dtype: str
) -> None:
    code = _code(
        5, cols, cols, min(cols, 32), 8, key_dtype=key_dtype, value_mode="decode"
    )
    tree = ast.parse(code)
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "topk_keys"
    )
    assert ast.unparse(assignment.value).endswith(f"cutlass.{selected_dtype})")
    if selected_dtype == "Int32":
        assert "topk_keys.fill(cutlass.Int32(-2147483648))" in code
        assert "topk_keys[topk_i] = topk_packed" in code
    elif key_dtype == "float32":
        assert "topk_keys.fill(cutlass.Float32(-2147483648))" in code
        assert "topk_keys[topk_i] = cutlass.Float32(topk_packed)" in code
        assert "cutlass.Int32(topk_local[topk_j])" in code
        assert "cutlass.Int32(topk_output_keys[topk_j * 8 + topk_v])" in code
    else:
        assert "topk_keys.fill(cutlass.Float32(0.0))" in code
        assert (
            "(topk_packed + cutlass.Int32(1073741824)).bitcast(cutlass.Float32)" in code
        )
        assert (
            "topk_local[topk_j].bitcast(cutlass.Int32) - cutlass.Int32(1073741824)"
            in code
        )


def test_topk_numeric_float_key_exactness_boundary() -> None:
    # Every integer in [-2**24, 2**24] has an exact Float32 representation.
    # Check both rank signs and payload endpoints at the widest enabled key.
    ranks = torch.arange(-32767, 32768, dtype=torch.int32)
    payloads = torch.tensor([0, 1, 510, 511], dtype=torch.int32)
    keys = ((ranks[:, None] << 9) | payloads).flatten()
    floats = keys.to(torch.float32)
    assert bool((keys.abs() <= 2**24).all())
    assert torch.equal(floats.to(torch.int32), keys)
    assert bool((floats[1:] > floats[:-1]).all())
    padding = torch.tensor(-2147483648, dtype=torch.int32)
    assert padding.float().to(torch.int32) == padding
    assert bool((floats > padding.float()).all())
    # At ten payload bits a representable rank can lose its tie-breaking bit.
    too_wide = torch.tensor((32767 << 10) | 1, dtype=torch.int32)
    assert too_wide.float().to(torch.int32) != too_wide


def test_topk_biased_float_key_exactness_boundary() -> None:
    # All valid biased keys are positive normal finite floats. Their IEEE
    # representation therefore preserves integer order without conversions.
    ranks = torch.arange(-32767, 32768, dtype=torch.int32)
    payloads = torch.tensor([0, 1, 16382, 16383], dtype=torch.int32)
    keys = ((ranks[:, None] << 14) | payloads).flatten()
    floats = (keys + 0x40000000).view(torch.float32)
    assert bool(torch.isfinite(floats).all())
    assert bool((floats >= torch.finfo(torch.float32).tiny).all())
    assert bool((floats[1:] > floats[:-1]).all())
    assert bool((floats > 0.0).all())  # Padding cannot displace any real key.
    assert torch.equal(floats.view(torch.int32) - 0x40000000, keys)
    # One more index bit admits NaN encodings and must use the Int32 fallback.
    too_wide = torch.tensor(((32767 << 15) | 32767) + 0x40000000, dtype=torch.int32)
    assert bool(torch.isnan(too_wide.view(torch.float32)))


def _simulate_distributed_topk(inputs: list[list[int]], k: int) -> list[int]:
    """Simulate cyclic subgroup exchanges, independently of CuTe execution."""
    lanes = len(inputs)
    selected = [sorted(values, reverse=True)[:k] for values in inputs]
    for stage in range(1, lanes.bit_length()):
        half = 1 << (stage - 1)
        group = 2 * half
        previous_size = len(selected[0])
        grow = previous_size * group <= k
        merge_group = group
        if previous_size == 1 and group > k:
            merged = [
                [max(selected[lane][0], selected[lane ^ (group - 1)][0])]
                for lane in range(lanes)
            ]
            halves = [
                [
                    merged[lane ^ (k - 1)] if lane & half else merged[lane]
                    for lane in range(lanes)
                ]
            ]
            merge_group = k
        elif grow and previous_size == 1:
            halves = [
                [
                    selected[lane ^ (half - 1)] if lane & half else selected[lane]
                    for lane in range(lanes)
                ]
            ]
        else:
            halves = []
            for compare in (max, min) if grow else (max,):
                merged = []
                for lane in range(lanes):
                    peer_lane = lane ^ (group - 1)
                    fragment = []
                    for index in range(previous_size // 2):
                        own_index = (
                            previous_size - 2 - 2 * index if lane & half else 2 * index
                        )
                        peer_index = (
                            previous_size - 1 - 2 * index
                            if peer_lane & half
                            else 2 * index + 1
                        )
                        fragment.append(
                            compare(
                                selected[lane][own_index],
                                selected[peer_lane][peer_index],
                            )
                        )
                    merged.append(fragment)
                # Reverse only the upper half to establish cyclic rank ownership.
                halves.append(
                    [
                        merged[lane ^ (half - 1)] if lane & half else merged[lane]
                        for lane in range(lanes)
                    ]
                )
        for merged in halves:
            distance = len(merged[0]) * merge_group // 2
            while distance:
                previous = [fragment[:] for fragment in merged]
                for lane in range(lanes):
                    for index in range(len(merged[0])):
                        rank = index * merge_group + lane % merge_group
                        peer = (
                            previous[lane][index ^ (distance // merge_group)]
                            if distance >= merge_group
                            else previous[lane ^ distance][index]
                        )
                        compare = min if rank & distance else max
                        merged[lane][index] = compare(previous[lane][index], peer)
                distance //= 2
        selected = [
            list(itertools.chain.from_iterable(part[lane] for part in halves))
            for lane in range(lanes)
        ]
    return [selected[rank % lanes][rank // lanes] for rank in range(k)]


@pytest.mark.parametrize("k,lanes", [(2, 2), (4, 4), (8, 4)])
def test_distributed_topk_zero_one_network(k: int, lanes: int) -> None:
    # After local sorting, every binary fragment is characterized by its
    # number of ones; this exhausts all binary inputs without permutations.
    for counts in itertools.product(range(k + 1), repeat=lanes):
        inputs = [[1] * count + [0] * (k - count) for count in counts]
        expected = sorted(itertools.chain.from_iterable(inputs), reverse=True)[:k]
        assert _simulate_distributed_topk(inputs, k) == expected


@pytest.mark.parametrize("lanes", [1, 2, 4, 8, 16, 32])
def test_distributed_topk_random_keys_and_padding(lanes: int) -> None:
    generator = random.Random(20260925)
    for k in (1, 2, 4, 8, 16, 32, 64, 128):
        if k < lanes:
            continue
        for _trial in range(16):
            inputs = [
                [
                    generator.choice(
                        (-2147483648, 0, 1, generator.randrange(-100000, 100000))
                    )
                    for _index in range(2 * k)
                ]
                for _lane in range(lanes)
            ]
            expected = sorted(itertools.chain.from_iterable(inputs), reverse=True)[:k]
            assert _simulate_distributed_topk(inputs, k) == expected


@pytest.mark.parametrize(
    "size,lanes,k", [(1, 8, 8), (2, 4, 8), (2, 8, 8), (4, 4, 8), (4, 4, 16)]
)
def test_distributed_topk_growing_zero_one_network(
    size: int, lanes: int, k: int
) -> None:
    for counts in itertools.product(range(size + 1), repeat=lanes):
        inputs = [[1] * count + [0] * (size - count) for count in counts]
        expected = sorted(itertools.chain.from_iterable(inputs), reverse=True)[:k]
        assert _simulate_distributed_topk(inputs, k) == expected


@pytest.mark.parametrize("lanes", [2, 4, 8, 16, 32])
def test_distributed_topk_growing_random_and_padding(lanes: int) -> None:
    generator = random.Random(20260926)
    for size in (1, 2, 4, 8, 16, 32, 64):
        for k in (2, 4, 8, 16, 32, 64, 128):
            if not lanes <= k <= size * lanes or size >= k:
                continue
            for _trial in range(16):
                inputs = [
                    [
                        generator.choice(
                            (-2147483648, -1, 0, 1, generator.randrange(-10000, 10000))
                        )
                        for _index in range(size)
                    ]
                    for _lane in range(lanes)
                ]
                expected = sorted(itertools.chain.from_iterable(inputs), reverse=True)[
                    :k
                ]
                assert _simulate_distributed_topk(inputs, k) == expected


@pytest.mark.parametrize("size,lanes,k", [(1, 8, 1), (1, 8, 2), (2, 8, 4), (4, 4, 2)])
def test_distributed_topk_wide_zero_one_network(size: int, lanes: int, k: int) -> None:
    for counts in itertools.product(range(size + 1), repeat=lanes):
        inputs = [[1] * count + [0] * (size - count) for count in counts]
        expected = sorted(itertools.chain.from_iterable(inputs), reverse=True)[:k]
        assert _simulate_distributed_topk(inputs, k) == expected


@pytest.mark.parametrize("lanes", [2, 4, 8, 16, 32])
def test_distributed_topk_wider_than_k_random(lanes: int) -> None:
    generator = random.Random(20260928)
    for k in (1, 2, 4, 8, 16):
        if k >= lanes:
            continue
        for size in (1, 2, 4, 8, 16, 32, 64):
            for _trial in range(32):
                inputs = [
                    [
                        generator.choice(
                            (-2147483648, -1, 0, 1, generator.randrange(-10000, 10000))
                        )
                        for _index in range(size)
                    ]
                    for _lane in range(lanes)
                ]
                expected = sorted(itertools.chain.from_iterable(inputs), reverse=True)[
                    :k
                ]
                assert _simulate_distributed_topk(inputs, k) == expected


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize(
    "n,k,lanes,vector,fragment",
    [
        (128, 32, 16, 8, 8),
        (128, 32, 32, 8, 8),
        (128, 32, 32, 4, 4),
        (8, 8, 8, 1, 1),
        (13, 9, 16, 1, 1),
        (64, 32, 32, 2, 2),
    ],
)
def test_distributed_fragment_does_not_pad_every_lane_to_k(
    n: int, k: int, lanes: int, vector: int, fragment: int
) -> None:
    code = _code(
        9, n, n, k, selection_layout="distributed", lanes=lanes, input_vector=vector
    )
    _assert_register_selection(code)
    assert f"topk_keys = cute.make_rmem_tensor({fragment}, cutlass.Int32)" in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("network", ["batcher", "compact", "compact_pruned"])
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
def test_topk_sort_network_reaches_codegen(network: str, layout: str) -> None:
    with patch(
        "helion._compiler.cute.selection_network.trace_selection_network",
        wraps=trace_selection_network,
    ) as trace:
        code = _code(
            9, 128, 128, 32, selection_layout=layout, lanes=2, sort_network=network
        )
    assert trace.call_count == 1
    groups, _, k, _, selected_network, schedule, selected_layout = trace.call_args.args
    assert (groups, k, selected_network, schedule, selected_layout) == (
        2,
        32,
        network,
        "sequential",
        layout,
    )
    _assert_register_selection(code)
    assert _has_register_communication(code)


@skipUnlessCuteAvailable("requires CuTe DSL")
def test_topk_cache_hash_includes_compact_tables() -> None:
    from helion._compiler.cute import selection_network

    selection_network._compact_sort_layers.cache_clear()
    trace_selection_network.cache_clear()
    before = _code(5, 32, 32, 8, sort_network="compact_pruned", lanes=1)
    try:
        with patch.dict(COMPACT_SORT_LAYERS, {32: COMPACT_SORT_LAYERS[32][1:]}):
            selection_network._compact_sort_layers.cache_clear()
            trace_selection_network.cache_clear()
            after = _code(5, 32, 32, 8, sort_network="compact_pruned", lanes=1)
    finally:
        selection_network._compact_sort_layers.cache_clear()
        trace_selection_network.cache_clear()
    assert before != after
    assert _compact_register_plans(before) != _compact_register_plans(after)
    _assert_register_selection(before)
    _assert_register_selection(after)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize(
    "k,lanes",
    [(3, 8), (3, 4), (6, 2), (33, 1), (32, 16), (1, 32), (8, 32)],
)
@pytest.mark.parametrize("key_dtype", ["int32", "float32", "float32_bits"])
def test_topk_distributed_codegen_guard(k: int, lanes: int, key_dtype: str) -> None:
    code = _code(
        5,
        64,
        64,
        k,
        8,
        lanes=lanes,
        key_dtype=key_dtype,
        selection_layout="distributed",
        value_mode="decode",
        rank_mode="ordinal",
    )
    _assert_register_selection(code)
    assert "topk_selected_key =" in code
    assert "topk_local = cute.make_rmem_tensor" not in code
    vector = min(8, lanes, max(1, (1 << (k - 1).bit_length()) // lanes))
    assert ("cute.autovec_copy" in code) == (vector > 1 and k % vector == 0)
    assert f"topk_j * {min(lanes, 1 << (k - 1).bit_length())}" in code
    if k != 1 << (k - 1).bit_length() or k < lanes:
        assert f"topk_output_col < cutlass.Int32({k})" in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("rows,stride", [(1, 2**35), (268435457, 0)])
def test_topk_distributed_keeps_wide_addresses(rows: int, stride: int) -> None:
    code = _code(
        rows,
        8,
        stride,
        8,
        8,
        selection_layout="distributed",
        lanes=4,
        value_mode="decode",
        rank_mode="ordinal",
    )
    _assert_register_selection(code)
    assert "cutlass.Int64(cute.arch.block_idx()[0])" in code
    assert f"* cutlass.Int64({stride})" in code
    assert "values[topk_row, topk_output_col]" in code
    assert "indices[topk_row, topk_output_col]" in code
    assert "topk_row * cutlass.Int64(8) + cutlass.Int64(topk_output_col)" in code
    assert "cute.assume(topk_output_offset, divby=2)" in code


@pytest.mark.parametrize("lanes", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("requested_vector", [1, 2, 4, 8])
def test_distributed_output_transpose_rank_mapping(
    lanes: int, requested_vector: int
) -> None:
    # Label every register with its original global rank. This verifies the
    # permutation for any key values, including duplicates and padding.
    for local_size in (1, 2, 4, 8, 16, 32, 64):
        vector = min(requested_vector, lanes, local_size)
        ranks = [
            [index * lanes + lane for index in range(local_size)]
            for lane in range(lanes)
        ]
        for block in range(local_size // vector):
            for stage in range(vector.bit_length() - 1):
                bit = 1 << stage
                previous = [registers[:] for registers in ranks]
                for lane in range(lanes):
                    for register in range(vector):
                        if register & bit:
                            continue
                        low = block * vector + register
                        high = low + bit
                        ranks[lane][low] = (
                            previous[lane ^ bit][high]
                            if lane & bit
                            else previous[lane][low]
                        )
                        ranks[lane][high] = (
                            previous[lane][high]
                            if lane & bit
                            else previous[lane ^ bit][low]
                        )
        for lane in range(lanes):
            output_lane = (lane % vector) * (lanes // vector) + lane // vector
            expected = [
                block * lanes * vector + output_lane * vector + element
                for block in range(local_size // vector)
                for element in range(vector)
            ]
            assert ranks[lane] == expected


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("key_dtype", ["int32", "float32", "float32_bits"])
@pytest.mark.parametrize(
    "k,lanes,requested,effective",
    [
        (32, 1, 8, 8),
        (32, 32, 8, 1),
        (32, 2, 8, 8),
        (32, 4, 8, 8),
        (64, 8, 8, 8),
        (24, 8, 8, 4),
        (12, 8, 8, 2),
        (6, 2, 8, 2),
        (3, 2, 8, 1),
        (32, 4, 1, 1),
        (64, 8, 2, 2),
        (64, 8, 4, 4),
    ],
)
def test_distributed_vector_codegen(
    k: int, lanes: int, requested: int, effective: int, key_dtype: str
) -> None:
    code = _code(
        9,
        128,
        130,
        k,
        requested,
        lanes=lanes,
        selection_layout="distributed",
        key_dtype=key_dtype,
        value_mode="decode",
        rank_mode="ordinal",
    )
    assert ("_cute_cyclic_to_vector" in code) == (effective > 1)
    assert ("cute.autovec_copy" in code) == (effective > 1)
    if effective == 1:
        return
    assert f"values.iterator.alignment >= {2 * effective}" in code
    assert "indices.iterator.alignment >= 16" in code
    assert f"cute.assume(topk_output_offset, divby={effective})" in code
    output_lane = "topk_lane" if effective > lanes else "topk_output_lane"
    assert f"{output_lane} * cutlass.Int32({effective})" in code
    if k != 1 << (k - 1).bit_length():
        assert f"topk_output_col < cutlass.Int32({k})" in code
    # Scalar ABI fallback must retain the original cyclic fragment, while
    # Float32 keys are only converted back after the transpose.
    assert "topk_selected[topk_j]" in code
    vector_key = f"topk_output_keys[topk_j * {effective} + topk_v]"
    if key_dtype == "float32":
        assert f"cutlass.Int32({vector_key})" in code
    elif key_dtype == "float32_bits":
        assert f"{vector_key}.bitcast(cutlass.Int32)" in code
    else:
        assert f"topk_selected_key = {vector_key}" in code


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _out_topk(x: torch.Tensor, values: torch.Tensor, indices: torch.Tensor) -> None:
    for row in hl.tile(x.size(0)):
        vals, idx = torch.topk(x[row, :], x.size(1), dim=-1, sorted=True)
        values[row, :] = vals
        indices[row, :] = idx


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("alias_kind", ["separate", "view", "dlpack"])
@pytest.mark.parametrize("alias_target", ["values", "indices"])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_matcher_uses_runtime_span_alias_proof(
    alias_kind: str, alias_target: str, index_dtype: torch.dtype
) -> None:
    x = torch.arange(32, dtype=torch.bfloat16).reshape(2, 16)
    indices = torch.empty_like(x, dtype=index_dtype)
    if alias_target == "indices" and alias_kind != "separate":
        x = indices.view(torch.bfloat16).flatten()[: x.numel()].view_as(x)
    values = x.clone()
    if alias_target == "values" and alias_kind != "separate":
        values = x.view_as(x) if alias_kind == "view" else torch.from_dlpack(x)
    elif alias_target == "indices" and alias_kind == "dlpack":
        indices = torch.from_dlpack(indices)
    aliased = values if alias_target == "values" else indices
    if alias_kind == "dlpack":
        assert x.data_ptr() == aliased.data_ptr()
        assert x.untyped_storage()._cdata != aliased.untyped_storage()._cdata
        with FakeTensorMode() as mode:
            fx, fa = mode.from_tensor(x), mode.from_tensor(aliased)
            assert fx.untyped_storage()._cdata != fa.untyped_storage()._cdata
    with (
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 0)),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 0),
        ),
        patch("helion.language.loops.use_tileir_tunables", return_value=False),
        patch("helion.language.loops._supports_warp_specialize", return_value=True),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
        patch("helion._compat._min_dot_size", return_value=(16, 16, 16)),
        patch("helion._compat._is_hip", return_value=False),
    ):
        bound = _out_topk._bind_isolated((x, values, indices))
        code = bound.to_triton_code(bound.config_spec.default_config())
        assert _has_register_selection(code) == (alias_kind == "separate")


@pytest.mark.parametrize(
    "dtype,width", [(torch.bfloat16, 16384), (torch.float16, 8192)]
)
def test_native_float_key_range_and_roundtrip(dtype: torch.dtype, width: int) -> None:
    words = torch.arange(65536, dtype=torch.int32)
    original = words.to(torch.int16).view(dtype)
    magnitude = words & 32767
    infinity = 0x7F80 if dtype == torch.bfloat16 else 0x7C00
    bits = original.float().view(torch.int32)
    bits = (
        bits | 16384
        if dtype == torch.bfloat16
        else torch.where(magnitude == 0, bits | 32768, bits)
    )
    bits = torch.where(
        magnitude == infinity, 0x7F7F8000 | ((words & 32768) << 16), bits
    )
    bits = torch.where(magnitude > infinity, 0x7F7FC000, bits)
    indices = words % width
    encoded_index = torch.where(bits < 0, indices, width - 1 - indices)
    keys_bits = bits | encoded_index
    keys = keys_bits.view(torch.float32)
    assert bool(torch.isfinite(keys).all())
    assert not bool((keys == 0).any())
    recovered_indices = (keys_bits ^ ((keys_bits >> 31) ^ -1)) & (width - 1)
    assert torch.equal(recovered_indices, indices)
    clean = keys_bits & ~(width - 1)
    recovered_values = clean.view(torch.float32).to(dtype)
    finite = magnitude < infinity
    assert torch.equal(
        recovered_values[finite].view(torch.int16), original[finite].view(torch.int16)
    )
    order = keys.argsort()
    values = original.float()[order]
    nonnan = ~values.isnan()
    assert bool((values[nonnan][:-1] <= values[nonnan][1:]).all())
    assert bool(values[-int((~nonnan).sum()) :].isnan().all())


@pytest.mark.parametrize("index_bits", range(15))
@pytest.mark.parametrize("largest", [False, True])
def test_native_bfloat_bias_exhaustive(index_bits: int, largest: bool) -> None:
    # All payload widths cover every N in [1, 16384]. Checking both endpoints
    # bounds every intervening index because the payload is monotone and lies
    # strictly below the reserved bit. Also check a middle index explicitly.
    mask = (1 << index_bits) - 1
    words = torch.arange(65536, dtype=torch.int32)
    original = words.to(torch.int16).view(torch.bfloat16)
    magnitude = words & 32767
    finite = magnitude < 0x7F80
    bits = original.float().view(torch.int32) | 0x4000
    bits = torch.where(magnitude == 0x7F80, 0x7F7F8000 | ((words & 32768) << 16), bits)
    bits = torch.where(magnitude > 0x7F80, 0x7F7FC000, bits)
    indices = torch.tensor([0, mask // 2, mask], dtype=torch.int32)[:, None]
    payload = torch.where(bits[None, :] < 0, indices, mask - indices)
    key_bits = bits[None, :] | payload
    keys = key_bits.view(torch.float32)
    if not largest:
        keys = -keys
    assert bool(torch.isfinite(keys).all())
    assert not bool((keys == 0).any())

    # Recover indices and bits by the emitted decoder, including undoing the
    # smallest-first sign reversal before interpreting the native payload.
    output_bits = (keys if largest else -keys).view(torch.int32)
    recovered_indices = (output_bits ^ ((output_bits >> 31) ^ -1)) & mask
    assert torch.equal(recovered_indices, indices.expand_as(recovered_indices))
    clean = output_bits & ~mask
    decoded = clean.view(torch.float32).to(torch.bfloat16)
    assert torch.equal(
        decoded[:, finite].view(torch.int16),
        original[finite].view(torch.int16)[None, :].expand_as(decoded[:, finite]),
    )
    # The existing guard must gather exactly infinities and NaNs, preserving
    # their original sign/payload. Quarter-ULP finite keys must stay below it.
    decodable = (output_bits & 0x7FFF8000) < 0x7F7F8000
    assert torch.equal(decodable, finite[None, :].expand_as(decodable))
    assert bool(((output_bits[:, finite] & 0x7FFFFFFF) <= 0x7F7F7FFF).all())
    gathered = original.view(torch.int16)[None, :].expand_as(output_bits)
    selected = torch.where(decodable, decoded.view(torch.int16), gathered)
    assert torch.equal(selected, gathered)

    # An integer ordinal oracle independently orders original BF16 words;
    # signed zeros can be ordered either way and NaNs share the largest rank.
    signed = words.to(torch.int16).to(torch.int32)
    ordinal = signed ^ ((signed >> 31) & 32767)
    ordinal = torch.where(magnitude > 0x7F80, 32767, ordinal)
    if not largest:
        ordinal = -ordinal
    order = ordinal.argsort()
    distinct = ordinal[order][1:] != ordinal[order][:-1]
    minimum = keys.amin(dim=0)[order]
    maximum = keys.amax(dim=0)[order]
    # Even opposite index endpoints cannot exchange differently ranked values.
    assert bool((maximum[:-1][distinct] < minimum[1:][distinct]).all())


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
def test_native_float_bias_codegen(dtype: torch.dtype, largest: bool) -> None:
    code = _code(
        3,
        128,
        128,
        32,
        4,
        dtype=dtype,
        largest=largest,
        key_dtype="float32_native",
        value_mode="decode",
    )
    input_type = "cutlass.BFloat16" if dtype == torch.bfloat16 else "cutlass.Float16"
    assert f"cutlass.Uint16(topk_bits).bitcast({input_type})" in code
    assert ("if topk_magnitude == 0:" in code) == (dtype == torch.float16)
    expected_bias = 16384 if dtype == torch.bfloat16 else 32768
    assert f"topk_native_bits | cutlass.Int32({expected_bias})" in code
    assert ("topk_native_key = -topk_native_key" in code) == (not largest)


@pytest.fixture
def selection_codegen_stub() -> Iterator[None]:
    """Isolate key/epilogue guards from the size of the sorting program."""

    def emit(
        cg: Any,
        *,
        dtype: torch.dtype,
        k: int,
        groups: int,
        lane: str,
        mode: str,
        **kwargs: Any,
    ) -> RowFragment:
        registers = k if mode == "replicated" else max(1, k // groups)
        name = cg.device_function.new_var("guard_selected")
        dtype_name = {
            torch.int32: "cutlass.Int32",
            torch.int64: "cutlass.Int64",
            torch.float32: "cutlass.Float32",
        }[dtype]
        cg.add_statement(
            ast.parse(
                f"{name} = cute.make_rmem_tensor({registers}, {dtype_name})"
            ).body[0]
        )
        return RowFragment(
            name, dtype, registers * groups, RowFragmentLayout(groups, 1, lane)
        )

    with patch(
        "helion._compiler.cute.topk_codegen.emit_selection_network", side_effect=emit
    ):
        yield


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("selection_codegen_stub")
@pytest.mark.parametrize(
    "dtype,width,native",
    [
        (torch.bfloat16, 16384, True),
        (torch.bfloat16, 16385, False),
        (torch.float16, 8192, True),
        (torch.float16, 8193, False),
    ],
)
def test_native_float_width_guard(dtype: torch.dtype, width: int, native: bool) -> None:
    code = _code(
        3,
        width,
        width,
        32,
        4,
        dtype=dtype,
        key_dtype="float32_native",
        value_mode="decode",
    )
    assert ("topk_native_key" in code) == native


# Networks top-k tests.


def _evaluate(values: np.ndarray, size: int, k: int, network: str) -> np.ndarray:
    assert values.shape[1] == size
    # Execute the current tensor decomposition, with one independent result
    # per input row. NumPy's full sort remains the independent oracle.
    return selection_network(
        torch.from_numpy(values), k, network, groups_per_result=1
    ).numpy()


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("network", ["batcher", "compact", "compact_pruned"])
def test_selection_network_exhaustive_binary(size: int, network: str) -> None:
    values = (
        (np.arange(1 << size, dtype=np.uint32)[:, None] >> np.arange(size)) & 1
    ).astype(np.uint8)
    expected = np.sort(values, axis=1)[:, ::-1]
    # Zero-one verification proves data-independent comparator networks for
    # every ordered input domain. Check every supported prefix length too.
    for exponent in range(size.bit_length()):
        k = 1 << exponent
        np.testing.assert_array_equal(
            _evaluate(values, size, k, network), expected[:, :k]
        )


@pytest.mark.parametrize("size", [32, 64, 128, 256])
@pytest.mark.parametrize("network", ["batcher", "compact", "compact_pruned"])
@pytest.mark.parametrize("floating", [False, True])
def test_selection_network_random_duplicates_and_padding(
    size: int, network: str, floating: bool
) -> None:
    generator = np.random.default_rng(20260926)
    values = generator.integers(-(1 << 22), 1 << 22, size=(128, size), dtype=np.int32)
    values[0] = 0
    values[1] = np.iinfo(np.int32).min
    values[2] = np.arange(size) % 4
    for row in range(3, len(values)):
        values[row, row % size :] = np.iinfo(np.int32).min
        generator.shuffle(values[row])
    if floating:
        values = values.astype(np.float32)
    expected = np.sort(values, axis=1)[:, ::-1]
    for exponent in range(size.bit_length()):
        k = 1 << exponent
        np.testing.assert_array_equal(
            _evaluate(values, size, k, network), expected[:, :k]
        )


@skipUnlessCuteAvailable("requires CuTe DSL")
def test_compact_network_bounds_and_operation_count() -> None:
    from helion._compiler.cute.selection_network import _compact_sort_layers

    for size, layers in COMPACT_SORT_LAYERS.items():
        for layer in layers:
            wires = [wire for pair in layer for wire in pair]
            assert len(wires) == len(set(wires))
            assert all(0 <= left < right < size for left, right in layer)
    assert sum(map(len, _compact_sort_layers(32, 32))) == 185
    assert sum(map(len, _compact_sort_layers(64, 64))) == 521
    merge_operations = 32 * 6
    assert 4 * sum(map(len, _compact_sort_layers(32, 32))) + merge_operations == 932
    assert (
        sum(
            high + low
            for layer in _compact_sort_layers(64, 32)
            for _, _, high, low in layer
        )
        == 870
    )


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
def test_network_dispatch_and_power_two_fallback(size: int) -> None:
    generator = np.random.default_rng(619)
    values = generator.permutation(size).astype(np.int32)[None, :]
    expected = np.sort(values, axis=1)[:, ::-1]
    for network in ("batcher", "compact", "compact_pruned"):
        for k in sorted({1, max(1, size // 2), size}):
            np.testing.assert_array_equal(
                _evaluate(values, size, k, network), expected[:, :k]
            )


@pytest.mark.parametrize("size,k", [(0, 1), (3, 1), (8, 0), (8, 3), (8, 16)])
def test_selection_network_rejects_invalid_sizes(size: int, k: int) -> None:
    with pytest.raises(AssertionError):
        selection_network(
            torch.empty((1, size), dtype=torch.int32), k, "compact_pruned"
        )


# Ordered asm top-k tests.


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rank_mode", ["signed", "ordinal"])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("index_bits", range(16))
def test_ordered_key_asm_exhaustive(
    dtype: torch.dtype, rank_mode: str, largest: bool, index_bits: int
) -> None:
    words = torch.arange(65536, dtype=torch.int32)
    signed = words.to(torch.int16).to(torch.int32)
    magnitude = signed & 32767
    sign = signed >> 31
    mask = (1 << index_bits) - 1
    infinity = 0x7F80 if dtype == torch.bfloat16 else 0x7C00
    # PTX arithmetic with signed32 wrapping and the same narrow input.
    rank = (
        signed ^ (sign & 32767) if rank_mode == "ordinal" else (magnitude ^ sign) - sign
    )
    rank = torch.where(magnitude > infinity, 32767, rank)
    if not largest:
        rank = -rank
    columns = torch.tensor([0, mask // 2, mask], dtype=torch.int32)[:, None]
    packed = (rank[None, :] << index_bits) | (mask - columns)

    # Independent sign/magnitude oracle: ordinal negative values include -1.
    expected_rank = torch.where(
        words < 32768, magnitude, -magnitude - int(rank_mode == "ordinal")
    )
    expected_rank = torch.where(magnitude > infinity, 32767, expected_rank)
    if not largest:
        expected_rank = -expected_rank
    expected = (expected_rank[None, :] << index_bits) | (mask - columns)
    assert torch.equal(packed, expected)
    assert torch.equal(packed >> index_bits, rank[None, :].expand_as(packed))
    assert torch.equal(mask - (packed & mask), columns.expand_as(packed))
    assert bool((packed > torch.iinfo(torch.int32).min).all())
    assert int(rank.min()) >= -32767 and int(rank.max()) <= 32767
    # The existing Float32 conversions remain exact inside their guards.
    if index_bits <= 9:
        assert torch.equal(packed.float().to(torch.int32), packed)
    if index_bits <= 14:
        biased = packed + 0x40000000
        assert bool(torch.isfinite(biased.view(torch.float32)).all())
        assert bool((biased.view(torch.float32) > 0).all())
        assert torch.equal(
            biased.view(torch.float32).view(torch.int32) - 0x40000000, packed
        )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "width,key_dtype,ordered",
    [
        (128, "int32", True),
        (512, "float32", True),
        (1024, "float32", True),
        (16384, "float32_bits", True),
        (32768, "float32_bits", True),
        (128, "float32_native", False),
        (32768, "float32_native", True),
    ],
)
@pytest.mark.parametrize("key_encoder", ["dsl", "asm"])
def test_ordered_key_asm_codegen_scope(
    dtype: torch.dtype, width: int, key_dtype: str, ordered: bool, key_encoder: str
) -> None:
    code = _code(
        3,
        width,
        width + 1,
        8,
        dtype=dtype,
        key_dtype=key_dtype,
        key_encoder=key_encoder,
    )
    ordered = ordered and key_encoder == "asm"
    assert ("_cute_encode_ordered_topk_" in code) == ordered
    if ordered:
        assert "topk_bits = topk_input_bits[topk_row, topk_col]" in code
        tree = ast.parse(code)
        assert not any(
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "topk_sign"
            for node in ast.walk(tree)
        )


# Packed rare top-k tests.


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("index_bits", range(16))
@pytest.mark.parametrize("largest", [False, True])
def test_packed_rare_exhaustive(
    dtype: torch.dtype, index_bits: int, largest: bool
) -> None:
    words = ((torch.arange(65536, dtype=torch.int32) * 32771) & 65535).reshape(-1, 16)
    magnitude = words & 32767
    infinity = 0x7F80 if dtype == torch.bfloat16 else 0x7C00
    original_rank = torch.where(words < 32768, magnitude, -1 - magnitude)
    original_rank = torch.where(magnitude > infinity, 32767, original_rank)
    mask = (1 << index_bits) - 1
    for column in (0, mask // 2, mask):
        ranked = original_rank if largest else -original_rank
        packed = (ranked << index_bits) | (mask - column)
        for key_dtype in ("int32", "float32", "float32_bits"):
            if key_dtype == "float32" and index_bits <= 9:
                selected = packed.float().to(torch.int32)
            elif key_dtype == "float32_bits" and index_bits <= 14:
                selected = (packed + 0x40000000).view(torch.float32).view(
                    torch.int32
                ) - 0x40000000
            else:
                selected = packed
            rank = selected >> index_bits
            if not largest:
                rank = -rank
            assert torch.equal(rank, original_rank)
            branch = rank.amax(dim=-1, keepdim=True) == 32767
            exceptional = magnitude > infinity
            assert torch.equal(branch, exceptional.any(dim=-1, keepdim=True))
            ordered_rank = selected.sort(dim=-1, descending=True).values >> index_bits
            if not largest:
                ordered_rank = -ordered_rank
            endpoint = ordered_rank[:, :1] if largest else ordered_rank[:, -1:]
            assert torch.equal(endpoint, rank.amax(dim=-1, keepdim=True))
            assert torch.equal(endpoint == 32767, branch)
            direct = (rank ^ ((rank >> 31) & 32767)).to(torch.int16)
            repaired = torch.where(
                branch & (rank == 32767), words.to(torch.int16), direct
            )
            assert torch.equal(repaired, words.to(torch.int16))
            assert bool((mask - (selected & mask) == column).all())


@pytest.mark.parametrize(
    "key,default,valid,invalid",
    [
        ("cute_topk_key_encoder", "dsl", "asm", True),
        ("cute_topk_key_encoder", "dsl", "dsl", "unknown"),
        ("cute_topk_defer_value_gathers", False, True, 1),
        ("cute_topk_defer_value_gathers", False, False, "true"),
    ],
)
def test_packed_rare_config(
    key: str,
    default: object,
    valid: object,
    invalid: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("helion.autotuner.config_spec.get_num_xcd", lambda device: 1)
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(10, 0), num_sm=1)
    with pytest.raises(exc.InvalidConfig, match="compatible top-k root"):
        spec.normalize({key: valid})
    spec.enable_cute_topk_search()
    assert spec.default_config()[key] == default
    config = helion.Config()
    config.config[key] = valid
    spec.normalize(config)
    generation = ConfigGeneration(spec)
    assert generation.unflatten(generation.flatten(config)) == config
    invalid_config = helion.Config()
    invalid_config.config[key] = invalid
    with pytest.raises(exc.InvalidConfig, match="must be one of"):
        spec.normalize(invalid_config)
    spec.normalize(invalid_config, _fix_invalid=True)
    assert invalid_config[key] == default


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("selection_codegen_stub")
@pytest.mark.parametrize(
    "width,k,lanes,vector,layout,key_dtype,rank_mode,value_mode,active",
    [
        (128, 32, 2, 4, "replicated", "int32", "ordinal", "decode", True),
        (128, 24, 2, 4, "replicated", "float32_bits", "ordinal", "decode", True),
        (128, 32, 4, 4, "distributed", "float32", "ordinal", "decode", True),
        (128, 12, 2, 4, "replicated", "int32", "ordinal", "decode", False),
        (64, 6, 2, 2, "distributed", "int32", "ordinal", "decode", False),
        (128, 32, 2, 1, "replicated", "int32", "ordinal", "decode", False),
        (128, 32, 2, 4, "replicated", "float32_native", "ordinal", "decode", False),
        (128, 32, 2, 4, "replicated", "int32", "ordinal", "gather", False),
        (128, 32, 2, 4, "replicated", "int32", "signed", "decode", False),
        (32768, 32, 2, 4, "replicated", "float32_native", "ordinal", "decode", True),
    ],
)
@pytest.mark.parametrize("key_encoder", ["dsl", "asm"])
@pytest.mark.parametrize("largest", [False, True])
def test_packed_rare_codegen_guard(
    width: int,
    k: int,
    lanes: int,
    vector: int,
    layout: str,
    key_dtype: str,
    rank_mode: str,
    value_mode: str,
    active: bool,
    key_encoder: str,
    largest: bool,
) -> None:
    code = _code(
        5,
        width,
        width + 2,
        k,
        vector,
        largest=largest,
        lanes=lanes,
        key_dtype=key_dtype,
        rank_mode=rank_mode,
        value_mode=value_mode,
        selection_layout=layout,
        key_encoder=key_encoder,
        defer_value_gathers=True,
    )
    assert ("topk_output_max_rank" in code) == active
    if not active:
        return
    branch = next(
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "topk_output_max_rank == cutlass.Int32(32767)"
    )
    text = ast.unparse(branch)
    assert (
        "cyclic_to_vector" not in text
        and "shuffle" not in text
        and "autovec_copy" not in text
    )
    assert "x[topk_row, topk_selected_index]" in text
    assert ("topk_value_rank = -topk_value_rank" in text) == (not largest)
    assert code.index("topk_selected =") < code.index("if topk_output_max_rank")
    if layout == "distributed":
        assert code.index("topk_output_keys = _cute_cyclic_to_vector") < code.index(
            "if topk_valid_row:"
        )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_packed_rare_wide_addresses(index_dtype: torch.dtype) -> None:
    code = _code(
        1,
        128,
        2**35,
        32,
        4,
        lanes=2,
        index_dtype=index_dtype,
        rank_mode="ordinal",
        value_mode="decode",
        defer_value_gathers=True,
    )
    assert "topk_output_max_rank" in code
    assert "topk_row = cutlass.Int64(" in code
    assert "topk_output_offset = topk_row * cutlass.Int64(32)" in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("key_encoder", ["dsl", "asm"])
@pytest.mark.parametrize(
    "width,k,lanes,vector,layout,key_dtype,offset",
    [
        (128, 32, 2, 4, "replicated", "int32", 0),
        (128, 128, 4, 8, "distributed", "float32_bits", 0),
        (65, 8, 2, 2, "distributed", "float32", 0),
        (64, 6, 2, 2, "distributed", "int32", 1),
    ],
)
def test_packed_rare_exact_bits(
    dtype: torch.dtype,
    index_dtype: torch.dtype,
    largest: bool,
    key_encoder: str,
    width: int,
    k: int,
    lanes: int,
    vector: int,
    layout: str,
    key_dtype: str,
    offset: int,
) -> None:
    rows = 514 if k == 128 else 5
    x, storage = _layout_input(rows, width, dtype, 2 if offset else 0, offset)
    if k == 128:
        x[2:].copy_(
            torch.arange(65536, dtype=torch.int32)
            .to(torch.int16)
            .view(dtype)
            .reshape(-1, width)
        )
    else:
        x[2].fill_(1)
        x[3].fill_(float("inf"))
        x[3, 1::2] = -float("inf")
        nan_words = [0x7FC1, -46] if dtype == torch.bfloat16 else [0x7E01, -478]
        x[4].copy_(
            torch.tensor(nan_words, dtype=torch.int16)
            .view(dtype)
            .repeat((width + 1) // 2)[:width]
        )
    original = storage.clone()
    value_storage = torch.full((rows * k + offset + 2,), 7, dtype=dtype, device=DEVICE)
    index_storage = torch.full(
        (rows * k + offset + 2,), -7, dtype=index_dtype, device=DEVICE
    )
    values = value_storage[offset : offset + rows * k].view(rows, k)
    indices = index_storage[offset : offset + rows * k].view(rows, k)
    code, _ = code_and_output(
        _extra_out_topk,
        (x, values, indices, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=vector,
        cute_topk_selection_layout=layout,
        cute_topk_value_mode="decode",
        cute_topk_key_dtype=key_dtype,
        cute_topk_rank_mode="ordinal",
        cute_topk_sort_network="compact_pruned",
        cute_topk_key_encoder=key_encoder,
        cute_topk_defer_value_gathers=True,
    )
    assert ("topk_output_max_rank" in code) == (k != 6)
    _assert_topk_output(x, values, indices, k, largest, index_dtype=index_dtype)
    assert torch.equal(storage.view(torch.int16), original.view(torch.int16))
    assert bool((value_storage[:offset] == 7).all())
    assert bool((value_storage[offset + rows * k :] == 7).all())
    assert bool((index_storage[:offset] == -7).all())
    assert bool((index_storage[offset + rows * k :] == -7).all())


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
@pytest.mark.parametrize("key_encoder", ["dsl", "asm"])
def test_packed_rare_alignment_transition(
    index_dtype: torch.dtype, layout: str, key_encoder: str
) -> None:
    rows, width, k = 5, 128, 32
    x, storage = _layout_input(rows, width, torch.bfloat16, 0, 0)
    original = storage.clone()
    value_storage = torch.empty(rows * k + 2, dtype=torch.bfloat16, device=DEVICE)
    index_storage = torch.empty(rows * k + 2, dtype=index_dtype, device=DEVICE)
    values = value_storage[: rows * k].view(rows, k)
    indices = index_storage[: rows * k].view(rows, k)
    bound = _extra_out_topk._bind_isolated((x, values, indices, k, True))
    bound.set_config(
        helion.Config(
            block_sizes=[1],
            cute_topk_lanes_per_row=4,
            cute_topk_rows_per_block=4,
            cute_topk_vector_width=8,
            cute_topk_output_vector_width=4,
            cute_topk_selection_layout=layout,
            cute_topk_key_dtype="int32",
            cute_topk_rank_mode="ordinal",
            cute_topk_key_encoder=key_encoder,
            cute_topk_value_mode="decode",
            cute_topk_defer_value_gathers=True,
        )
    )
    for value_offset, index_offset in ((0, 0), (0, 1), (1, 0), (1, 1), (0, 0)):
        value_storage.fill_(7)
        index_storage.fill_(-7)
        values = value_storage[value_offset : value_offset + rows * k].view(rows, k)
        indices = index_storage[index_offset : index_offset + rows * k].view(rows, k)
        bound(x, values, indices, k, True)
        _assert_topk_output(x, values, indices, k, index_dtype=index_dtype)
        assert bool((value_storage[:value_offset] == 7).all())
        assert bool((value_storage[value_offset + rows * k :] == 7).all())
        assert bool((index_storage[:index_offset] == -7).all())
        assert bool((index_storage[index_offset + rows * k :] == -7).all())
        assert torch.equal(storage.view(torch.int16), original.view(torch.int16))


# Endpoints top-k tests.


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize(
    "k,lanes,requested,layout,last",
    [
        (24, 2, 4, "replicated", 11),
        (32, 4, 4, "distributed", 7),
        (8, 2, 8, "distributed", 3),
    ],
)
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("key_dtype", ["int32", "float32", "float32_bits"])
@pytest.mark.parametrize("encoder", ["dsl", "asm"])
def test_packed_endpoint_codegen(
    k: int,
    lanes: int,
    requested: int,
    layout: str,
    last: int,
    largest: bool,
    key_dtype: str,
    encoder: str,
) -> None:
    code = _code(
        5,
        128,
        128,
        k,
        requested,
        lanes=lanes,
        largest=largest,
        key_dtype=key_dtype,
        key_encoder=encoder,
        rank_mode="ordinal",
        value_mode="decode",
        selection_layout=layout,
        defer_value_gathers=True,
    )
    assignments = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "topk_output_max_rank"
    ]
    assert len(assignments) == 1
    expression = assignments[0].value
    reads = [
        node
        for node in ast.walk(expression)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "topk_output_keys"
    ]
    assert len(reads) == 1
    assert ast.literal_eval(reads[0].slice) == (0 if largest else last)
    assert isinstance(expression, ast.UnaryOp) == (not largest)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "max"
        for node in ast.walk(expression)
    )
    assert "max(topk_output_max_rank" not in code
    if layout == "distributed" and key_dtype == "float32_bits":
        assert ".bitcast(cutlass.Int32) - cutlass.Int32(1073741824)" in ast.unparse(
            expression
        )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("encoder", ["dsl", "asm"])
@pytest.mark.parametrize("offset", [0, 1])
def test_packed_endpoint_non_power_two_k(
    dtype: torch.dtype,
    index_dtype: torch.dtype,
    largest: bool,
    encoder: str,
    offset: int,
) -> None:
    rows, width, k = 5, 128, 24
    x, storage = _layout_input(rows, width, dtype, 2 if offset else 0, offset)
    nan_word = -46 if dtype == torch.bfloat16 else -478
    nan = torch.tensor([nan_word], dtype=torch.int16, device=DEVICE).view(dtype)[0]
    # Only one output-owning lane needs a NaN repair. For smallest top-k,
    # exactly k-1 finite inputs put the selected NaN at the last valid rank.
    if largest:
        x[2].copy_(torch.arange(width, dtype=dtype, device=DEVICE))
        x[2, -1] = nan
    else:
        x[2].copy_(nan.expand(width))
        x[2, : k - 1].copy_(torch.arange(k - 1, dtype=dtype, device=DEVICE))
    x[3].fill_(float("inf"))
    x[3, 1::2] = -float("inf")
    original = storage.clone()
    values_storage = torch.full((rows * k + offset + 2,), 7, dtype=dtype, device=DEVICE)
    indices_storage = torch.full(
        (rows * k + offset + 2,), -7, dtype=index_dtype, device=DEVICE
    )
    values = values_storage[offset : offset + rows * k].view(rows, k)
    indices = indices_storage[offset : offset + rows * k].view(rows, k)
    code, _ = code_and_output(
        _extra_out_topk,
        (x, values, indices, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=2,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=4,
        cute_topk_key_dtype="int32",
        cute_topk_rank_mode="ordinal",
        cute_topk_value_mode="decode",
        cute_topk_selection_layout="replicated",
        cute_topk_key_encoder=encoder,
        cute_topk_defer_value_gathers=True,
    )
    assert "topk_output_keys[0]" in code if largest else "topk_output_keys[11]" in code
    _assert_topk_output(x, values, indices, k, largest, index_dtype=index_dtype)
    assert torch.equal(storage.view(torch.int16), original.view(torch.int16))
    assert bool((values_storage[:offset] == 7).all())
    assert bool((values_storage[offset + rows * k :] == 7).all())
    assert bool((indices_storage[:offset] == -7).all())
    assert bool((indices_storage[offset + rows * k :] == -7).all())


# Alias cache top-k tests.


@pytest.fixture
def cpu_codegen() -> Iterator[None]:
    pytest.importorskip("cutlass.cute")
    with (
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 0)),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 0),
        ),
        patch("helion.language.loops.use_tileir_tunables", return_value=False),
        patch("helion.language.loops._supports_warp_specialize", return_value=True),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
        patch("helion._compat._min_dot_size", return_value=(16, 16, 16)),
        patch("helion._compat._is_hip", return_value=False),
    ):
        yield


def _new_kernel() -> helion.Kernel[Any]:
    return helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")(
        _extra_out_topk.fn
    )


def _config(encoder: str) -> helion.Config:
    return helion.Config(
        block_sizes=[1],
        cute_topk_lanes_per_row=2,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=4,
        cute_topk_value_mode="decode",
        cute_topk_key_dtype="int32",
        cute_topk_rank_mode="ordinal",
        cute_topk_selection_layout="replicated",
        cute_topk_sort_network="compact_pruned",
        cute_topk_key_encoder=encoder,
        cute_topk_defer_value_gathers=True,
    )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("retain_input", [False, True])
def test_cached_topk_recompiles_after_construction_tensors_expire(
    retain_input: bool,
) -> None:
    kernel = _new_kernel()

    def first_binding() -> tuple[Any, torch.Tensor | None]:
        x = torch.empty((5, 128), dtype=torch.bfloat16)
        values = torch.empty((5, 32), dtype=x.dtype)
        indices = torch.empty((5, 32), dtype=torch.int32)
        bound = kernel.bind((x, values, indices, 32, True))
        assert "topk_output_max_rank" in bound.to_triton_code(_config("dsl"))
        return bound, x if retain_input else None

    bound, retained = first_binding()
    gc.collect()
    assert sum(
        ref() is not None for ref in bound._runtime_tensor_refs_by_name.values()
    ) == int(retain_input)
    assert bound.env.runtime_arg_values_by_name == {}
    x = torch.empty((5, 128), dtype=torch.bfloat16)
    values = torch.empty((5, 32), dtype=x.dtype)
    indices = torch.empty((5, 32), dtype=torch.int32)
    assert kernel.bind((x, values, indices, 32, True)) is bound
    code = bound.to_triton_code(_config("asm"))
    assert "topk_output_max_rank" in code
    assert "_cute_encode_ordered_topk" in code
    assert (retained is not None) == retain_input


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("alias_kind", ["view", "dlpack"])
@pytest.mark.parametrize("width", [1, 32])
def test_cached_topk_alias_binding_stays_rejected_after_release(
    alias_kind: str, width: int
) -> None:
    kernel = _new_kernel()

    def bind_pair() -> tuple[Any, Any]:
        x = torch.empty((5, width), dtype=torch.bfloat16)
        values = torch.empty_like(x)
        indices = torch.empty((5, width), dtype=torch.int32)
        disjoint = kernel.bind((x, values, indices, width, True))
        alias = x.view_as(x) if alias_kind == "view" else torch.from_dlpack(x)
        overlapping = kernel.bind((x, alias, indices, width, True))
        assert overlapping is not disjoint
        return disjoint, overlapping

    disjoint, overlapping = bind_pair()
    gc.collect()
    for bound, expected in ((disjoint, True), (overlapping, False)):
        assert all(ref() is None for ref in bound._runtime_tensor_refs_by_name.values())
        assert bound.config_spec.cute_topk_search_enabled == expected
        assert (
            "cute_topk_lanes_per_row" in bound.config_spec.default_config()
        ) == expected
        code = bound.to_triton_code(bound.config_spec.default_config())
        assert _has_register_selection(code) == expected


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("largest", [False, True])
def test_singleton_topk_search_and_recompile_after_input_release(
    dtype: torch.dtype, index_dtype: torch.dtype, largest: bool
) -> None:
    kernel = _new_kernel()
    config = helion.Config(
        block_sizes=[1],
        cute_topk_lanes_per_row=1,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=1,
    )

    def first_binding() -> Any:
        x = torch.empty((5, 1), dtype=dtype)
        values = torch.empty_like(x)
        indices = torch.empty((5, 1), dtype=index_dtype)
        bound = kernel.bind((x, values, indices, 1, largest))
        assert bound.config_spec.cute_topk_search_enabled
        _assert_register_selection(bound.to_triton_code(config))
        return bound

    bound = first_binding()
    gc.collect()
    assert all(ref() is None for ref in bound._runtime_tensor_refs_by_name.values())
    assert bound.env.runtime_arg_values_by_name == {}
    x = torch.empty((5, 1), dtype=dtype)
    values = torch.empty_like(x)
    indices = torch.empty((5, 1), dtype=index_dtype)
    assert kernel.bind((x, values, indices, 1, largest)) is bound
    config.config["cute_topk_vector_width"] = 4
    _assert_register_selection(bound.to_triton_code(config))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize(
    "damage",
    [
        "none",
        "missing",
        "identity",
        "sources",
        "properties",
        "length",
        "nonbool",
        "live_alias",
        "partial_live_alias",
    ],
)
def test_cached_topk_alias_fact_requires_matching_descriptor(damage: str) -> None:
    x = torch.empty((5, 32), dtype=torch.bfloat16)
    values = torch.empty_like(x)
    indices = torch.empty((5, 32), dtype=torch.int32)
    bound = _new_kernel().bind((x, values, indices, 32, True))
    env = bound.env
    key = _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
    descriptor = env.runtime_input_specializations[key]
    if damage == "missing":
        env.bound_runtime_input_specialization_results.clear()
    elif damage in ("identity", "sources", "properties"):
        changes: dict[str, Any] = {
            "identity": {"classifier_identity": "other"},
            "sources": {"sources": descriptor.sources[::-1]},
            "properties": {"reusable_tensor_properties": frozenset()},
        }[damage]
        env.runtime_input_specializations[key] = dataclasses.replace(
            descriptor, **changes
        )
    elif damage == "length":
        env.bound_runtime_input_specialization_results[key] = (True,)
    elif damage == "nonbool":
        env.bound_runtime_input_specialization_results[key] = (1, True, True)
    live_args: dict[str, object] = {}
    if damage in ("live_alias", "partial_live_alias"):
        live_args = {"x": x, "values": torch.from_dlpack(x)}
        if damage == "live_alias":
            live_args["indices"] = indices
    host = bound.host_function
    assert host is not None
    with env, host, env.use_runtime_arg_values(live_args):
        candidate = match_topk_root(
            host.device_ir.graphs,
            noncanonical_block_ids=host.device_ir.noncanonical_task_origin_block_ids,
        )
        assert candidate is not None
        assert topk_tensors_are_proven_disjoint(candidate, env) == (damage == "none")


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize(
    "damage",
    ["none", "alias", "missing_argument", "identity", "bound_false", "bound_none"],
)
def test_search_alias_proof_requires_registered_live_facts(damage: str) -> None:
    x = torch.empty((5, 32), dtype=torch.bfloat16)
    values = torch.empty_like(x)
    indices = torch.empty((5, 32), dtype=torch.int32)
    bound = _new_kernel().bind((x, values, indices, 32, True))
    env = bound.env
    key = _TENSOR_DISJOINT_MATRIX_SPECIALIZATION_KEY
    env.bound_runtime_input_specialization_results.clear()
    runtime_args: dict[str, object] = {"x": x, "values": values, "indices": indices}
    if damage == "alias":
        runtime_args["values"] = torch.from_dlpack(x)
    elif damage == "missing_argument":
        runtime_args.pop("values")
    elif damage == "identity":
        env.runtime_input_specializations[key] = dataclasses.replace(
            env.runtime_input_specializations[key], classifier_identity="other"
        )
    elif damage == "bound_false":
        env.bound_runtime_input_specialization_results[key] = (False, False, False)
    elif damage == "bound_none":
        env.bound_runtime_input_specialization_results[key] = None
    host = bound.host_function
    assert host is not None
    with env, host, env.use_runtime_arg_values(runtime_args):
        candidate = match_topk_root(
            host.device_ir.graphs,
            noncanonical_block_ids=host.device_ir.noncanonical_task_origin_block_ids,
        )
        assert candidate is not None
        assert not topk_tensors_are_proven_disjoint(candidate, env)
        assert topk_tensors_are_proven_disjoint(candidate, env, allow_unbound=True) == (
            damage == "none"
        )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("largest", [False, True])
def test_cached_topk_recompiles_exact_values_after_input_release(
    dtype: torch.dtype, index_dtype: torch.dtype, largest: bool
) -> None:
    kernel = _new_kernel()

    def run(encoder: str) -> Any:
        x, storage = _layout_input(5, 128, dtype, 0, 0)
        values = torch.empty((5, 32), dtype=dtype, device=DEVICE)
        indices = torch.empty((5, 32), dtype=index_dtype, device=DEVICE)
        args = (x, values, indices, 32, largest)
        bound = kernel.bind(args)
        code, _ = code_and_output(kernel, args, **_config(encoder).config)
        assert "topk_output_max_rank" in code
        _assert_topk_output(x, values, indices, 32, largest, index_dtype=index_dtype)
        assert torch.equal(x.view(torch.int16), storage.view(5, 128).view(torch.int16))
        return bound

    first = run("dsl")
    gc.collect()
    assert all(ref() is None for ref in first._runtime_tensor_refs_by_name.values())
    assert run("asm") is first


# Balanced top-k tests.


def _select(values: np.ndarray, k: int, network: str) -> np.ndarray:
    return selection_network(
        torch.from_numpy(values), k, network, "balanced", groups_per_result=1
    ).numpy()


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("network", ["batcher", "compact"])
def test_balanced_exhaustive_binary(size: int, network: str) -> None:
    values = (
        (np.arange(1 << size, dtype=np.uint32)[:, None] >> np.arange(size)) & 1
    ).astype(np.uint8)
    expected = np.sort(values, axis=1)[:, ::-1]
    for exponent in range(size.bit_length()):
        k = 1 << exponent
        np.testing.assert_array_equal(_select(values, k, network), expected[:, :k])


@pytest.mark.parametrize("size", [32, 64, 128, 256])
@pytest.mark.parametrize("network", ["batcher", "compact"])
@pytest.mark.parametrize("floating", [False, True])
def test_balanced_random_duplicates_padding(
    size: int, network: str, floating: bool
) -> None:
    generator = np.random.default_rng(20260926)
    values = generator.integers(-64, 64, size=(128, size), dtype=np.int32)
    values[0] = 0
    for row in range(1, len(values)):
        values[row, row % size :] = np.iinfo(np.int32).min
        generator.shuffle(values[row])
    if floating:
        values = values.astype(np.float32)
        values[values == np.float32(np.iinfo(np.int32).min)] = -np.inf
    expected = np.sort(values, axis=1)[:, ::-1]
    for exponent in range(size.bit_length()):
        k = 1 << exponent
        np.testing.assert_array_equal(_select(values, k, network), expected[:, :k])


def _operation_count_and_depth(
    size: int, k: int, network: str, balanced: bool
) -> tuple[int, int]:
    graph = trace_selection_network(
        1, size, k, torch.int32, network, "balanced" if balanced else "sequential"
    )
    depths = {}
    operations = 0
    for node in graph.graph.nodes:
        comparator = node.target in (
            torch.ops.aten.fmin.default,
            torch.ops.aten.fmax.default,
            torch.ops.aten.minimum.default,
            torch.ops.aten.maximum.default,
        )
        depths[node] = (
            max((depths[x] for x in node.all_input_nodes), default=0) + comparator
        )
        if comparator:
            operations += node.meta["val"].numel()
    return operations, depths[
        next(node for node in graph.graph.nodes if node.op == "output")
    ]


@pytest.mark.parametrize("size,k", [(32, 8), (64, 8), (128, 8), (32, 32), (128, 32)])
def test_balanced_same_operations_shorter_depth(size: int, k: int) -> None:
    # Inspect the current traced comparator DAG, not a second implementation
    # of the old handwritten schedule. These are tensor operations, not SASS.
    sequential_ops, sequential_depth = _operation_count_and_depth(
        size, k, "batcher", False
    )
    balanced_ops, balanced_depth = _operation_count_and_depth(size, k, "batcher", True)
    assert balanced_ops == sequential_ops
    assert balanced_depth <= sequential_depth
    if size > 2 * k:
        assert balanced_depth < sequential_depth


def test_balanced_config_roundtrip_and_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("helion.autotuner.config_spec.get_num_xcd", lambda device: 1)
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(10, 0), num_sm=1)
    with pytest.raises(exc.InvalidConfig):
        spec.normalize(helion.Config(cute_topk_merge_schedule="balanced"))
    spec.enable_cute_topk_search()
    assert spec.default_config()["cute_topk_merge_schedule"] == "sequential"
    config = helion.Config(cute_topk_merge_schedule="balanced")
    spec.normalize(config)
    generation = ConfigGeneration(spec)
    assert generation.unflatten(generation.flatten(config)) == config
    for invalid in (True, 1, "tree"):
        with pytest.raises(exc.InvalidConfig):
            spec.normalize(helion.Config(cute_topk_merge_schedule=invalid))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
@pytest.mark.parametrize("network", ["batcher", "compact", "compact_pruned"])
def test_balanced_schedule_reaches_codegen(layout: str, network: str) -> None:
    with patch(
        "helion._compiler.cute.selection_network.trace_selection_network",
        wraps=trace_selection_network,
    ) as trace:
        code = _code(
            5,
            128,
            130,
            8,
            selection_layout=layout,
            lanes=4,
            sort_network=network,
            merge_schedule="balanced",
        )
    assert trace.call_count == 1
    groups, _, k, _, selected_network, schedule, selected_layout = trace.call_args.args
    assert (groups, k, selected_network, schedule, selected_layout) == (
        4,
        8,
        network,
        "balanced",
        layout,
    )
    _assert_register_selection(code)
    assert _has_register_communication(code)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "width,k,lanes,network,key_dtype,layout,rank_mode,value_mode,padding,offset",
    [
        (128, 8, 4, "batcher", "float32_bits", "distributed", "signed", "gather", 0, 0),
        (1024, 8, 8, "batcher", "float32", "distributed", "ordinal", "decode", 2, 1),
        (
            65,
            6,
            2,
            "compact",
            "float32_native",
            "replicated",
            "ordinal",
            "decode",
            2,
            1,
        ),
        (
            128,
            32,
            4,
            "compact_pruned",
            "int32",
            "distributed",
            "signed",
            "gather",
            0,
            0,
        ),
        (33, 8, 16, "batcher", "int32", "distributed", "ordinal", "decode", 0, 0),
    ],
)
def test_balanced_gpu_exact_selected_bits(
    dtype: torch.dtype,
    index_dtype: torch.dtype,
    largest: bool,
    width: int,
    k: int,
    lanes: int,
    network: str,
    key_dtype: str,
    layout: str,
    rank_mode: str,
    value_mode: str,
    padding: int,
    offset: int,
) -> None:
    rows = 5
    x, storage = _layout_input(rows, width, dtype, padding, offset)
    original_storage = storage.clone()
    values_storage = torch.full((rows * k + 2,), 7, dtype=dtype, device=DEVICE)
    indices_storage = torch.full((rows * k + 2,), -7, dtype=index_dtype, device=DEVICE)
    values = values_storage[offset : offset + rows * k].view(rows, k)
    indices = indices_storage[offset : offset + rows * k].view(rows, k)
    code, _output = code_and_output(
        _extra_out_topk,
        (x, values, indices, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=4,
        cute_topk_value_mode=value_mode,
        cute_topk_key_dtype=key_dtype,
        cute_topk_rank_mode=rank_mode,
        cute_topk_selection_layout=layout,
        cute_topk_sort_network=network,
        cute_topk_merge_schedule="balanced",
    )
    _assert_register_selection(code)
    _assert_topk_output(x, values, indices, k, largest, index_dtype=index_dtype)
    assert torch.equal(storage.view(torch.int16), original_storage.view(torch.int16))
    assert bool((values_storage[offset + rows * k :] == 7).all())
    assert bool((indices_storage[offset + rows * k :] == -7).all())
    if offset:
        assert bool((values_storage[:offset] == 7).all())
        assert bool((indices_storage[:offset] == -7).all())


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("encoder", ["dsl", "asm"])
@pytest.mark.parametrize("largest", [False, True])
def test_balanced_endpoint_codegen_composition(encoder: str, largest: bool) -> None:
    # P=128 exceeds the compact table catalog, so compact_pruned selects its
    # chunked fallback and the balanced schedule is active. K/(L*V)=1 makes
    # every output group complete, activating the ordinal endpoint guard.
    with patch(
        "helion._compiler.cute.selection_network.trace_selection_network",
        wraps=trace_selection_network,
    ) as trace:
        code = _code(
            5,
            256,
            256,
            8,
            4,
            lanes=2,
            largest=largest,
            value_mode="decode",
            rank_mode="ordinal",
            key_encoder=encoder,
            sort_network="compact_pruned",
            merge_schedule="balanced",
            defer_value_gathers=True,
        )
    assert trace.call_count == 1
    assert trace.call_args.args[4:] == ("compact_pruned", "balanced", "replicated")
    _assert_register_selection(code)
    assert "topk_output_max_rank" in code
    assert (
        "_cute_encode_ordered_topk" in code
        if encoder == "asm"
        else "topk_magnitude" in code
    )
    assert "topk_output_keys[0]" in code if largest else "topk_output_keys[3]" in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
def test_balanced_asm_endpoint_exact_selected_bits(
    dtype: torch.dtype, largest: bool
) -> None:
    rows, width, k = 5, 256, 8
    x, storage = _layout_input(rows, width, dtype, 0, 0)
    original = storage.clone()
    value_storage = torch.full((rows * k + 2,), 7, dtype=dtype, device=DEVICE)
    index_storage = torch.full((rows * k + 2,), -7, dtype=torch.int32, device=DEVICE)
    values = value_storage[: rows * k].view(rows, k)
    indices = index_storage[: rows * k].view(rows, k)
    code, _ = code_and_output(
        _extra_out_topk,
        (x, values, indices, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=2,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=4,
        cute_topk_key_dtype="int32",
        cute_topk_rank_mode="ordinal",
        cute_topk_value_mode="decode",
        cute_topk_selection_layout="replicated",
        cute_topk_sort_network="compact_pruned",
        cute_topk_merge_schedule="balanced",
        cute_topk_key_encoder="asm",
        cute_topk_defer_value_gathers=True,
    )
    _assert_register_selection(code)
    assert "topk_output_max_rank" in code
    _assert_topk_output(x, values, indices, k, largest, index_dtype=torch.int32)
    assert torch.equal(storage.view(torch.int16), original.view(torch.int16))
    assert bool((value_storage[rows * k :] == 7).all())
    assert bool((index_storage[rows * k :] == -7).all())


# Seeds top-k tests.


@pytest.fixture
def _cpu_compile_environment() -> Iterator[None]:
    pytest.importorskip("cutlass.cute")
    with (
        patch("helion.runtime.kernel.target_device_capability", return_value=(10, 0)),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=(10, 0),
        ),
        patch("helion.language.loops.use_tileir_tunables", return_value=False),
        patch("helion.language.loops._supports_warp_specialize", return_value=True),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
        patch("helion._compat._min_dot_size", return_value=(16, 16, 16)),
        patch("helion._compat._is_hip", return_value=False),
    ):
        yield


def _is_small_block_integer_topk_seed(seed: helion.Config) -> bool:
    return (
        seed["cute_topk_key_dtype"] == "int32"
        and seed["cute_topk_selection_layout"] == "distributed"
        and seed["cute_topk_sort_network"] == "compact_pruned"
        and seed["cute_topk_key_encoder"] == "paired"
        and seed["cute_topk_lanes_per_row"] * seed["cute_topk_rows_per_block"]
        in (32, 64)
    )


def _is_scalar_store_integer_topk_seed(seed: helion.Config) -> bool:
    return (
        seed["cute_topk_key_dtype"] == "int32"
        and seed["cute_topk_selection_layout"] == "distributed"
        and seed["cute_topk_sort_network"] == "batcher"
        and seed["cute_topk_key_encoder"] == "paired"
        and seed["cute_topk_output_vector_width"] == 1
        and seed["cute_topk_lanes_per_row"] * seed["cute_topk_rows_per_block"]
        in (128, 256)
    )


def _is_large_fragment_integer_topk_seed(seed: helion.Config) -> bool:
    return (
        seed.get("cute_topk_key_dtype") == "int32"
        and seed["cute_topk_selection_layout"] == "distributed"
        and seed["cute_topk_sort_network"] == "compact_pruned"
        and seed["cute_topk_key_encoder"] == "paired"
        and seed["cute_topk_output_vector_width"] == 1
        and seed["cute_topk_lanes_per_row"] * seed["cute_topk_rows_per_block"] == 128
    )


@pytest.fixture
def _legacy_topk_seed_family() -> Iterator[None]:
    # Coverage-policy tests compare enabling a new coordinate independently
    # from compiler hints that now explicitly select that coordinate.
    original = CuteTopKHeuristic.get_seed_configs

    def legacy(cls: type, env: Any, device_ir: Any) -> list[helion.Config] | None:
        seeds = original(env, device_ir)
        if seeds is None:
            return None
        return [seed for seed in seeds if not seed.get("cute_topk_coarse_keys", False)]

    with patch.object(CuteTopKHeuristic, "get_seed_configs", classmethod(legacy)):
        yield


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("_cpu_compile_environment")
def test_topk_coarse_gather_seeds_retain_initial_and_finalist_coverage() -> None:
    from helion.autotuner.pattern_search import PatternSearch

    with FakeTensorMode():
        x = torch.empty((17, 256), dtype=torch.float32)
    args = (x, 8, True)
    bound = _allocating_topk._bind_isolated(args)
    spec = bound.config_spec
    with (
        bound.env,
        bound.host_function,
        patch.object(spec, "cute_topk_coarse_keys_available", False),
    ):
        legacy = CuteTopKHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    assert legacy is not None and len(legacy) == 8
    assert spec.compiler_seed_configs[:8] == legacy
    family = spec.compiler_seed_configs[8:]
    assert [
        (s["cute_topk_lanes_per_row"], s["cute_topk_rows_per_block"]) for s in family
    ] == [(16, 4), (8, 8)]
    expected = spec.default_config()
    expected.config.update(
        cute_topk_lanes_per_row=16,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=1,
        cute_topk_value_mode="gather",
        cute_topk_key_dtype="int64",
        cute_topk_rank_mode="ordinal",
        cute_topk_selection_layout="distributed",
        cute_topk_sort_network="compact_pruned",
        cute_topk_key_encoder="dsl",
        cute_topk_defer_value_gathers=False,
        cute_topk_merge_schedule="sequential",
        cute_topk_coarse_keys=True,
        cute_topk_key_recovery="packed",
    )
    assert family[0] == expected
    generation = ConfigGeneration(spec)
    for seed in family:
        normalized = copy.deepcopy(seed)
        spec.normalize(normalized)
        assert normalized == seed
        assert generation.strict_config_pair(seed)[1] == seed
    assert spec.compiler_default_config is None
    assert spec.default_config()["cute_topk_value_mode"] == "gather"
    assert "cute_topk_coarse_keys" not in spec.default_config()
    assert [g.key for g in spec.compiler_coverage_groups] == [
        "cute_topk_coarse_keys",
        "cute_topk_key_recovery",
    ]
    assert all(
        g.witnesses[0].carrier["cute_topk_value_mode"] == "decode"
        for g in spec.compiler_coverage_groups
    )
    with bound.env:
        search = PatternSearch(bound, args, initial_population=100)
        flats = search._generate_initial_population_flat()
        assert len(flats) == 102  # Ordinary 100 plus the two existing witnesses.
        initial = [search.config_gen.unflatten(flat) for flat in flats]
        assert all(seed in initial[:100] for seed in family)
        assert all(seed in search._pinned_finalist_configs for seed in family)
    _assert_register_selection(bound.to_code(expected))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("_cpu_compile_environment")
@pytest.mark.parametrize(
    "cols,k,expected_count",
    [(1, 1, 0), (17, 17, 0), (65, 8, 2), (1024, 32, 1), (1025, 32, 0)],
)
def test_topk_coarse_gather_seed_eligibility(
    cols: int, k: int, expected_count: int
) -> None:
    with FakeTensorMode():
        x = torch.empty((17, cols), dtype=torch.float32)
    bound = _allocating_topk._bind_isolated((x, k, True))
    family = [
        seed
        for seed in bound.config_spec.compiler_seed_configs
        if seed.get("cute_topk_coarse_keys", False)
    ]
    assert len(family) == expected_count
    for seed in family:
        normalized = copy.deepcopy(seed)
        bound.config_spec.normalize(normalized)
        assert normalized == seed
        assert seed["cute_topk_lanes_per_row"] * seed["cute_topk_rows_per_block"] == 64


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("_cpu_compile_environment")
@pytest.mark.parametrize(
    "cols,k,primary_lanes",
    [(1, 1, 1), (17, 3, 1), (64, 32, 2), (128, 32, 4), (1024, 32, 32), (32768, 1, 1)],
)
def test_topk_seeds_follow_fragment_geometry(
    cols: int, k: int, primary_lanes: int
) -> None:
    with FakeTensorMode():
        x = torch.empty((17, cols), dtype=torch.bfloat16)
    bound = _allocating_topk._bind_isolated((x, k, True))
    spec = bound.config_spec
    assert "cute_topk" in spec.autotuner_heuristics
    assert spec.compiler_default_config is None
    assert spec.default_config()["cute_topk_lanes_per_row"] == 16
    assert spec.default_config()["cute_topk_value_mode"] == "gather"
    assert spec.default_config()["cute_topk_sort_network"] == "batcher"
    seeds = [
        seed for seed in spec.compiler_seed_configs if "cute_topk_lanes_per_row" in seed
    ]
    small_block_seeds = [
        seed for seed in seeds if _is_small_block_integer_topk_seed(seed)
    ]
    scalar_store_seeds = [
        seed for seed in seeds if _is_scalar_store_integer_topk_seed(seed)
    ]
    large_fragment_seeds = [
        seed for seed in seeds if _is_large_fragment_integer_topk_seed(seed)
    ]
    original_seeds = [
        seed
        for seed in seeds
        if seed["cute_topk_key_dtype"] != "float32_bits"
        and not _is_small_block_integer_topk_seed(seed)
        and not _is_scalar_store_integer_topk_seed(seed)
        and not _is_large_fragment_integer_topk_seed(seed)
    ]
    legacy = [
        seed for seed in original_seeds if not seed["cute_topk_defer_value_gathers"]
    ]
    assert 4 < len(legacy) <= 8
    assert len(legacy) < len(original_seeds) <= 14
    assert seeds[: len(original_seeds)] == original_seeds
    assert (
        len(seeds)
        - len(original_seeds)
        - len(small_block_seeds)
        - len(scalar_store_seeds)
        - len(large_fragment_seeds)
        <= 4
    )
    assert len(small_block_seeds) <= 4
    assert len(scalar_store_seeds) <= 4
    assert len(large_fragment_seeds) <= 1
    if small_block_seeds:
        suffix = small_block_seeds + scalar_store_seeds + large_fragment_seeds
        assert seeds[-len(suffix) :] == suffix
    pairs = ConfigGeneration(spec).seed_flat_config_pairs()
    normalized = copy.deepcopy(seeds)
    for config in normalized:
        spec.normalize(config)
    assert [config for _flat, config in pairs[: len(seeds)]] == normalized
    primary = pairs[0][1]
    assert primary["cute_topk_lanes_per_row"] == primary_lanes
    assert primary["cute_topk_rows_per_block"] == 128 // primary_lanes
    assert primary["cute_topk_rank_mode"] == "ordinal"
    assert primary["cute_topk_value_mode"] == "decode"
    for _flat, config in pairs[:4]:
        spec.normalize(config)
        lanes = config["cute_topk_lanes_per_row"]
        rows = config["cute_topk_rows_per_block"]
        assert isinstance(lanes, int) and isinstance(rows, int)
        assert lanes <= min(32, 1 << (k - 1).bit_length())
        assert lanes * rows in (64, 128)
    assert pairs[3][1]["cute_topk_selection_layout"] == "distributed"
    assert all(
        seed["cute_topk_sort_network"] == "batcher" for seed in original_seeds[:4]
    )
    assert all(
        seed["cute_topk_sort_network"] == "compact_pruned"
        for seed in original_seeds[4:]
    )
    host_function = bound.host_function
    assert host_function is not None
    with bound.env, host_function:
        assert (
            CuteTopKHeuristic.get_seed_config(bound.env, host_function.device_ir)
            == spec.compiler_seed_configs[0]
        )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("_cpu_compile_environment")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_topk_integer_seeds_cover_scalar_and_paired_store_blocks(
    dtype: torch.dtype,
) -> None:
    from helion.autotuner.pattern_search import PatternSearch

    kernel = helion.kernel(
        backend="cute",
        static_shapes=True,
        autotune_effort="full",
        cute_structural_policy=_TOPK_STRUCTURAL_POLICY,
    )(_selected_topk)
    with FakeTensorMode():
        x = torch.empty((65536, 256), dtype=dtype)
    bound = kernel._bind_isolated((x, 32, True))
    spec = bound.config_spec
    family = [
        s for s in spec.compiler_seed_configs if _is_small_block_integer_topk_seed(s)
    ]
    assert [
        (
            s["cute_topk_lanes_per_row"],
            s["cute_topk_rows_per_block"],
            s["cute_topk_output_vector_width"],
        )
        for s in family
    ] == [
        (8, 4, 1),
        (4, 8, 1),
        (8, 8, 2),
        (4, 16, 2),
    ]
    prior = [
        seed
        for seed in spec.compiler_seed_configs
        if not _is_scalar_store_integer_topk_seed(seed)
        and not _is_large_fragment_integer_topk_seed(seed)
    ]
    assert prior[-4:] == family
    expected = spec.default_config()
    expected.config.update(
        cute_topk_lanes_per_row=4,
        cute_topk_rows_per_block=16,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=2,
        cute_topk_value_mode="decode",
        cute_topk_key_dtype="int32",
        cute_topk_rank_mode="ordinal",
        cute_topk_selection_layout="distributed",
        cute_topk_sort_network="compact_pruned",
        cute_topk_key_encoder="paired",
        cute_topk_defer_value_gathers=False,
        cute_topk_merge_schedule="balanced",
    )
    assert family[-1] == expected
    for seed in family:
        normalized = copy.deepcopy(seed)
        spec.normalize(normalized)
        assert normalized == seed
    assert spec.compiler_default_config is None
    assert spec.default_config()["cute_topk_lanes_per_row"] == 16
    generation = ConfigGeneration(spec)
    population = generation.random_population_flat(100)
    assert len(population) == 100
    assert all(
        seed in [generation.unflatten(flat) for flat in population] for seed in family
    )
    with bound.env:
        search = PatternSearch(bound, (x, 32, True), initial_population=100)
        assert len(search._generate_initial_population_flat()) == 100
        assert all(seed in search._pinned_finalist_configs for seed in family)
    _assert_register_selection(bound.to_code(expected))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("_cpu_compile_environment")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "cols,k,geometry",
    [
        (1, 1, [(1, 128)]),
        (17, 3, [(1, 128)]),
        (64, 32, [(2, 64), (2, 128), (1, 128)]),
        (65, 3, [(4, 32), (4, 64), (2, 64), (2, 128)]),
        (128, 8, [(4, 32), (4, 64), (2, 64), (2, 128)]),
        (256, 32, [(8, 16), (8, 32), (4, 32), (4, 64)]),
        (1024, 32, [(32, 4), (32, 8), (16, 8), (16, 16)]),
        (2048, 32, [(32, 4), (32, 8)]),
        (2049, 32, []),
        (32768, 1, []),
    ],
)
def test_topk_scalar_store_integer_seed_geometry(
    dtype: torch.dtype, cols: int, k: int, geometry: list[tuple[int, int]]
) -> None:
    with FakeTensorMode():
        x = torch.empty((17, cols), dtype=dtype)
    bound = _allocating_topk._bind_isolated((x, k, True))
    spec = bound.config_spec
    family = [
        seed
        for seed in spec.compiler_seed_configs
        if _is_scalar_store_integer_topk_seed(seed)
    ]
    assert len(family) <= 4
    assert [
        (seed["cute_topk_lanes_per_row"], seed["cute_topk_rows_per_block"])
        for seed in family
    ] == geometry
    if family:
        prior = [
            seed
            for seed in spec.compiler_seed_configs
            if not _is_large_fragment_integer_topk_seed(seed)
        ]
        assert prior[-len(family) :] == family
    assert len(set(spec.compiler_seed_configs)) == len(spec.compiler_seed_configs)
    generation = ConfigGeneration(spec)
    for seed in family:
        normalized = copy.deepcopy(seed)
        spec.normalize(normalized)
        assert normalized == seed
        _, strict = generation.strict_config_pair(seed)
        assert strict == seed
        assert (cols + seed["cute_topk_lanes_per_row"] - 1) // seed[
            "cute_topk_lanes_per_row"
        ] <= 64
        assert seed["cute_topk_vector_width"] <= 4
        assert seed["cute_topk_merge_schedule"] == "sequential"
        assert seed["cute_topk_rank_mode"] == "ordinal"
        assert seed["cute_topk_value_mode"] == "decode"
        assert seed["cute_topk_defer_value_gathers"] is False


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("_cpu_compile_environment")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_topk_scalar_store_integer_seeds_are_initial_pinned_hints(
    dtype: torch.dtype,
) -> None:
    from helion.autotuner.pattern_search import PatternSearch

    kernel = helion.kernel(
        backend="cute",
        static_shapes=True,
        autotune_effort="full",
        cute_structural_policy=_TOPK_STRUCTURAL_POLICY,
    )(_selected_topk)
    with FakeTensorMode():
        x = torch.empty((65536, 128), dtype=dtype)
    bound = kernel._bind_isolated((x, 8, True))
    spec = bound.config_spec
    family = [
        seed
        for seed in spec.compiler_seed_configs
        if _is_scalar_store_integer_topk_seed(seed)
    ]
    expected = spec.default_config()
    expected.config.update(
        cute_topk_lanes_per_row=4,
        cute_topk_rows_per_block=64,
        cute_topk_vector_width=4,
        cute_topk_output_vector_width=1,
        cute_topk_value_mode="decode",
        cute_topk_key_dtype="int32",
        cute_topk_rank_mode="ordinal",
        cute_topk_selection_layout="distributed",
        cute_topk_sort_network="batcher",
        cute_topk_key_encoder="paired",
        cute_topk_defer_value_gathers=False,
        cute_topk_merge_schedule="sequential",
    )
    assert expected in family
    assert spec.compiler_default_config is None
    assert spec.default_config()["cute_topk_lanes_per_row"] == 16
    with bound.env:
        search = PatternSearch(bound, (x, 8, True), initial_population=100)
        population = search._generate_initial_population_flat()
        assert len(population) == 100
        concrete = [search.config_gen.unflatten(flat) for flat in population]
        assert all(seed in concrete for seed in family)
        assert all(seed in search._pinned_finalist_configs for seed in family)
    code = bound.to_code(expected)
    _assert_register_selection(code)
    assert "cyclic_to_vector" not in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("_cpu_compile_environment")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "cols,k,geometry",
    [
        (1, 1, None),
        (64, 8, None),
        (65, 3, (1, 128)),
        (128, 8, (1, 128)),
        (129, 16, (2, 64)),
        (512, 8, (4, 32)),
        (513, 16, (8, 16)),
        (1024, 1, (8, 16)),
        (1024, 1024, (8, 16)),
        (4096, 16, (32, 4)),
        (4097, 16, None),
        (65536, 16, None),
    ],
)
def test_topk_large_fragment_integer_seed_scope(
    dtype: torch.dtype, cols: int, k: int, geometry: tuple[int, int] | None
) -> None:
    with FakeTensorMode():
        x = torch.empty((17, cols), dtype=dtype)
    bound = _allocating_topk._bind_isolated((x, k, True))
    spec = bound.config_spec
    family = [
        seed
        for seed in spec.compiler_seed_configs
        if _is_large_fragment_integer_topk_seed(seed)
    ]
    expected = [] if dtype == torch.float32 or geometry is None else [geometry]
    assert [
        (seed["cute_topk_lanes_per_row"], seed["cute_topk_rows_per_block"])
        for seed in family
    ] == expected
    assert len(family) <= 1
    assert len(set(spec.compiler_seed_configs)) == len(spec.compiler_seed_configs)
    assert spec.compiler_default_config is None
    assert spec.default_config().get("cute_topk_lanes_per_row") == (
        16 if spec.cute_topk_search_enabled else None
    )
    generation = ConfigGeneration(spec)
    for seed in family:
        _, normalized = generation.strict_config_pair(seed)
        assert normalized == seed
        assert (
            64
            < (cols + seed["cute_topk_lanes_per_row"] - 1)
            // seed["cute_topk_lanes_per_row"]
            <= 128
        )
        assert seed["cute_topk_vector_width"] == 8
        assert seed["cute_topk_merge_schedule"] == "balanced"
        assert seed["cute_topk_value_mode"] == "decode"
        assert seed["cute_topk_rank_mode"] == "ordinal"
        assert seed["cute_topk_defer_value_gathers"] is False


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("_cpu_compile_environment")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_topk_large_fragment_integer_seed_preserves_initial_budget(
    dtype: torch.dtype,
) -> None:
    from helion.autotuner.pattern_search import PatternSearch

    kernel = helion.kernel(
        backend="cute",
        static_shapes=True,
        autotune_effort="full",
        cute_structural_policy=_TOPK_STRUCTURAL_POLICY,
    )(_selected_topk)
    with FakeTensorMode():
        x = torch.empty((65536, 1024), dtype=dtype)
    args = (x, 16, False)
    bound = kernel._bind_isolated(args)
    spec = bound.config_spec
    family = [
        seed
        for seed in spec.compiler_seed_configs
        if _is_large_fragment_integer_topk_seed(seed)
    ]
    assert len(family) == 1
    assert family[0]["cute_topk_lanes_per_row"] == 8
    assert family[0]["cute_topk_rows_per_block"] == 16
    with bound.env:
        search = PatternSearch(bound, args, initial_population=100)
        population = search._generate_initial_population_flat()
        assert len(population) == 100
        for seed in spec.compiler_seed_configs:
            flat, normalized = search.config_gen.canonicalize_flat(
                search.config_gen.flatten(seed)
            )
            assert flat in population
            assert normalized in search._pinned_finalist_configs


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("_cpu_compile_environment")
def test_topk_large_fragment_integer_seed_respects_row_choices() -> None:
    with FakeTensorMode():
        x = torch.empty((17, 1024), dtype=torch.bfloat16)
    bound = _allocating_topk._bind_isolated((x, 16, False))
    spec = bound.config_spec
    assert any(
        _is_large_fragment_integer_topk_seed(s) for s in spec.compiler_seed_configs
    )
    host = bound.host_function
    assert host is not None
    with (
        bound.env,
        host,
        patch.dict(spec.cute_topk_choices, {"cute_topk_rows_per_block": (1, 2, 4, 8)}),
    ):
        seeds = CuteTopKHeuristic.get_seed_configs(bound.env, host.device_ir)
    assert seeds is not None
    assert not any(_is_large_fragment_integer_topk_seed(s) for s in seeds)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("_cpu_compile_environment")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "cols,k,distributed_lanes,native_lanes",
    [
        (3, 1, [1], [1]),
        (64, 32, [8, 4], [2, 1]),
        (65, 3, [16, 8], [4, 2]),
        (128, 32, [16, 8], [4, 2]),
        (256, 8, [32, 16], [8, 4]),
        (512, 8, [32], [8]),
        (32768, 1, [32], [1]),
    ],
)
def test_topk_new_seeds_cover_growing_and_native_fragments(
    dtype: torch.dtype,
    cols: int,
    k: int,
    distributed_lanes: list[int],
    native_lanes: list[int],
) -> None:
    with FakeTensorMode():
        x = torch.empty((17, cols), dtype=dtype)
    bound = _allocating_topk._bind_isolated((x, k, False))
    spec = bound.config_spec
    seeds = [
        seed
        for seed in spec.compiler_seed_configs
        if seed.get("cute_topk_sort_network") == "compact_pruned"
        and not seed["cute_topk_defer_value_gathers"]
        and not _is_small_block_integer_topk_seed(seed)
        and not _is_large_fragment_integer_topk_seed(seed)
    ]
    assert [
        seed["cute_topk_lanes_per_row"]
        for seed in seeds
        if seed["cute_topk_selection_layout"] == "distributed"
    ] == distributed_lanes
    assert [
        seed["cute_topk_lanes_per_row"]
        for seed in seeds
        if seed["cute_topk_key_dtype"] == "float32_native"
    ] == native_lanes
    for seed in seeds:
        spec.normalize(copy.deepcopy(seed))
        lanes = seed["cute_topk_lanes_per_row"]
        rows = seed["cute_topk_rows_per_block"]
        vector = seed["cute_topk_vector_width"]
        assert isinstance(lanes, int) and isinstance(rows, int)
        assert isinstance(vector, int)
        assert lanes * rows == 128
        assert vector <= min(8, (cols + lanes - 1) // lanes)
        assert seed["cute_topk_value_mode"] == "decode"
        assert seed["cute_topk_rank_mode"] == "ordinal"
        assert seed["cute_topk_output_vector_width"] == 4
        assert seed["cute_topk_key_dtype"] == (
            "int32"
            if seed["cute_topk_selection_layout"] == "distributed"
            else "float32_native"
        )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("_cpu_compile_environment")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "cols,k,geometry",
    [
        (1, 1, [(1, 128)]),
        (17, 3, [(1, 128)]),
        (64, 32, [(2, 64), (2, 128), (1, 128)]),
        (65, 3, [(4, 32), (4, 64), (2, 64), (2, 128)]),
        (256, 8, [(8, 16), (8, 32), (4, 32), (4, 64)]),
        (1024, 32, [(32, 4), (32, 8), (16, 8), (16, 16)]),
        (2048, 32, [(32, 4), (32, 8)]),
        (2049, 32, []),
        (32768, 1, []),
    ],
)
def test_topk_float_bit_seeds_preserve_bounded_geometry(
    dtype: torch.dtype, cols: int, k: int, geometry: list[tuple[int, int]]
) -> None:
    with FakeTensorMode():
        x = torch.empty((17, cols), dtype=dtype)
    bound = _allocating_topk._bind_isolated((x, k, True))
    spec = bound.config_spec
    seeds = [
        seed
        for seed in spec.compiler_seed_configs
        if seed.get("cute_topk_key_dtype") == "float32_bits"
    ]
    assert len(seeds) <= 4
    assert [
        (seed["cute_topk_lanes_per_row"], seed["cute_topk_rows_per_block"])
        for seed in seeds
    ] == geometry
    for seed in seeds:
        normalized = copy.deepcopy(seed)
        spec.normalize(normalized)
        assert normalized == seed
        lanes = seed["cute_topk_lanes_per_row"]
        rows = seed["cute_topk_rows_per_block"]
        assert isinstance(lanes, int) and isinstance(rows, int)
        assert lanes * rows in (128, 256)
        assert (cols + lanes - 1) // lanes <= 64
        assert seed["cute_topk_key_encoder"] == "paired"
        assert seed["cute_topk_selection_layout"] == "distributed"
        assert seed["cute_topk_sort_network"] == "batcher"
        assert seed["cute_topk_merge_schedule"] == "sequential"
        assert seed["cute_topk_rank_mode"] == "ordinal"
        assert seed["cute_topk_value_mode"] == "decode"
        assert seed["cute_topk_defer_value_gathers"] is False
    generation = ConfigGeneration(spec)
    pairs = generation.seed_flat_config_pairs()
    assert [
        config
        for _, config in pairs
        if config.get("cute_topk_key_dtype") == "float32_bits"
    ] == seeds
    for flat, config in pairs:
        assert generation.unflatten(flat) == config


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("_cpu_compile_environment")
def test_topk_float_bit_seeds_cover_distributed_256_threads() -> None:
    kernel = helion.kernel(
        backend="cute",
        static_shapes=True,
        autotune_effort="full",
        cute_structural_policy=_TOPK_STRUCTURAL_POLICY,
    )(_selected_topk)
    with FakeTensorMode():
        x = torch.empty((65536, 256), dtype=torch.bfloat16)
    bound = kernel._bind_isolated((x, 8, True))
    spec = bound.config_spec
    expected = spec.default_config()
    expected.config.update(
        cute_topk_defer_value_gathers=False,
        cute_topk_key_dtype="float32_bits",
        cute_topk_key_encoder="paired",
        cute_topk_lanes_per_row=4,
        cute_topk_merge_schedule="sequential",
        cute_topk_output_vector_width=2,
        cute_topk_rank_mode="ordinal",
        cute_topk_rows_per_block=64,
        cute_topk_selection_layout="distributed",
        cute_topk_sort_network="batcher",
        cute_topk_value_mode="decode",
        cute_topk_vector_width=4,
    )
    assert expected in spec.compiler_seed_configs
    normalized = copy.deepcopy(expected)
    spec.normalize(normalized)
    assert normalized == expected
    code = bound.to_code(expected)
    _assert_register_selection(code)
    assert "encode_ordinal_key_pair_16" in code
    assert "1073741824" in code
    # Seed hints replace random members without increasing the full search's
    # initial population or altering the default configuration.
    generation = ConfigGeneration(spec)
    population = generation.random_population_flat(100)
    assert len(population) == 100
    concrete = [generation.unflatten(flat) for flat in population]
    family = [
        seed
        for seed in spec.compiler_seed_configs
        if seed.get("cute_topk_key_dtype") == "float32_bits"
    ]
    assert len(family) == 4
    assert all(seed in concrete for seed in family)
    assert spec.compiler_default_config is None
    assert spec.default_config()["cute_topk_key_dtype"] == "int32"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_topk_float_bit_seed_encoding_range(dtype: torch.dtype) -> None:
    assert CuteTopKHeuristic._supports_float32_bit_keys(dtype, 16384)
    assert not CuteTopKHeuristic._supports_float32_bit_keys(dtype, 16385)
    for unsupported in (torch.float32, torch.float64, torch.int32, torch.bool):
        assert not CuteTopKHeuristic._supports_float32_bit_keys(unsupported, 256)
    index_bits = 14
    keys = torch.tensor(
        [
            -(32767 << index_bits),
            -(1 << index_bits),
            -1,
            0,
            1,
            (32767 << index_bits) | ((1 << index_bits) - 1),
        ],
        dtype=torch.int32,
    )
    encoded = (keys + 0x40000000).view(torch.float32)
    assert bool(torch.isfinite(encoded).all())
    assert bool((encoded >= torch.finfo(torch.float32).tiny).all())
    assert bool((encoded[1:] > encoded[:-1]).all())


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("_cpu_compile_environment")
def test_topk_float_bit_seeds_retain_strided_encoder_fallback() -> None:
    with FakeTensorMode():
        x = torch.empty_strided((17, 128), (256, 2), dtype=torch.bfloat16)
    bound = _allocating_topk._bind_isolated((x, 8, True))
    seeds = [
        seed
        for seed in bound.config_spec.compiler_seed_configs
        if seed.get("cute_topk_key_dtype") == "float32_bits"
    ]
    assert seeds
    code = bound.to_code(seeds[0])
    _assert_register_selection(code)
    assert "encode_ordinal_key_pair_16" not in code
    assert "1073741824" in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("_cpu_compile_environment")
@pytest.mark.parametrize(
    "dtype,inner_stride,supported",
    [
        (torch.float16, 1, True),
        (torch.bfloat16, 2, True),
        (torch.float32, 1, True),
        (torch.float64, 1, False),
        (torch.int32, 1, False),
    ],
)
def test_topk_seeds_follow_root_capabilities(
    dtype: torch.dtype, inner_stride: int, supported: bool
) -> None:
    with FakeTensorMode():
        x = torch.empty_strided(
            (17, 64), (64 * inner_stride, inner_stride), dtype=dtype
        )
    bound = _allocating_topk._bind_isolated((x, 32, True))
    assert ("cute_topk" in bound.config_spec.autotuner_heuristics) == supported
    seeds = [
        config
        for config in bound.config_spec.compiler_seed_configs
        if "cute_topk_lanes_per_row" in config
    ]
    assert bool(seeds) == supported
    float_bit_seeds = [
        seed for seed in seeds if seed["cute_topk_key_dtype"] == "float32_bits"
    ]
    assert bool(float_bit_seeds) == (dtype in (torch.float16, torch.bfloat16))
    if dtype == torch.float32:
        assert all(seed["cute_topk_key_dtype"] == "int64" for seed in seeds)
    if supported:
        for seed in seeds:
            bound.config_spec.normalize(copy.deepcopy(seed))
        config = bound.config_spec.default_config()
        config.config.update(seeds[0])
        code = bound.to_code(config)
        _assert_register_selection(code)
        assert "sort_rank" not in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("_cpu_compile_environment")
@pytest.mark.parametrize("alias_kind", ["separate", "view", "dlpack"])
def test_topk_seeds_require_final_runtime_alias_proof(alias_kind: str) -> None:
    x = torch.arange(32, dtype=torch.bfloat16).reshape(2, 16)
    values = (
        x.clone()
        if alias_kind == "separate"
        else x.view_as(x)
        if alias_kind == "view"
        else torch.from_dlpack(x)
    )
    indices = torch.empty_like(x, dtype=torch.int64)
    bound = _out_topk._bind_isolated((x, values, indices))
    runtime_args: dict[str, object] = {"x": x, "values": values, "indices": indices}
    host_function = bound.host_function
    assert host_function is not None
    with bound.env, host_function, bound.env.use_runtime_arg_values(runtime_args):
        seeds = CuteTopKHeuristic.get_seed_configs(bound.env, host_function.device_ir)
    assert bool(seeds) == (alias_kind == "separate")
    if seeds is not None:
        assert any(seed["cute_topk_key_dtype"] == "float32_bits" for seed in seeds)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("_cpu_compile_environment")
@pytest.mark.parametrize("alias_kind", ["separate", "view", "dlpack"])
@pytest.mark.parametrize("disable_heuristics", [False, True])
@pytest.mark.parametrize("width", [1, 32])
def test_topk_alias_fallback_retains_generic_search(
    alias_kind: str, disable_heuristics: bool, width: int
) -> None:
    x = torch.empty((17, width), dtype=torch.bfloat16)
    values = (
        torch.empty_like(x)
        if alias_kind == "separate"
        else x.view_as(x)
        if alias_kind == "view"
        else torch.from_dlpack(x)
    )
    indices = torch.empty_like(x, dtype=torch.int32)
    with patch.object(
        _out_topk.settings, "disable_autotuner_heuristics", disable_heuristics
    ):
        bound = _out_topk._bind_isolated((x, values, indices))
    spec = bound.config_spec
    specialized = alias_kind == "separate"
    assert spec.cute_topk_search_enabled == specialized
    config = spec.default_config()
    assert ("cute_topk_lanes_per_row" in config) == specialized
    assert _has_register_selection(bound.to_triton_code(config)) == specialized
    row = spec.block_sizes[0]
    if specialized:
        assert row.autotuner_min == row.max_size
        config.config.update(cute_topk_lanes_per_row=2, cute_topk_rows_per_block=64)
        spec.normalize(config)
        assert config["cute_topk_lanes_per_row"] == 2
        assert config["cute_topk_rows_per_block"] == 64
    else:
        assert row.max_size is None or row.autotuner_min < row.max_size
        small, large = copy.deepcopy(config), copy.deepcopy(config)
        small.config["block_sizes"] = [1]
        large.config["block_sizes"] = [16]
        spec.normalize(small)
        spec.normalize(large)
        assert small["block_sizes"] != large["block_sizes"]
        # Saved specialized configs can still be repaired for this fallback.
        large.config["cute_topk_lanes_per_row"] = 2
        spec.normalize(large, _fix_invalid=True)
        assert "cute_topk_lanes_per_row" not in large


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("_cpu_compile_environment")
def test_topk_seeds_registry_cache_and_disable_integration() -> None:
    assert CuteTopKHeuristic in get_heuristics("cute")
    assert CuteTopKHeuristic not in get_heuristics("triton")
    with FakeTensorMode():
        x = torch.empty((17, 64), dtype=torch.bfloat16)
    bound = _allocating_topk._bind_isolated((x, 32, True))
    spec = bound.config_spec
    structural_hash = spec.structural_fingerprint_hash()
    cache_hash = spec.cache_fingerprint_hash()
    spec.compiler_seed_configs = list(reversed(spec.compiler_seed_configs))
    assert spec.structural_fingerprint_hash() == structural_hash
    assert spec.cache_fingerprint_hash() != cache_hash
    with patch.object(_allocating_topk.settings, "disable_autotuner_heuristics", True):
        disabled = _allocating_topk._bind_isolated((x, 32, True))
    assert disabled.config_spec.compiler_seed_configs == []
    assert disabled.config_spec.cute_topk_search_enabled
    assert disabled.config_spec.compiler_default_config is None


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("_cpu_compile_environment")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "cols,k,expected_lanes",
    [
        (1, 1, [1]),
        (17, 3, [1]),
        (64, 32, [2, 1]),
        (128, 32, [4, 2]),
        (129, 24, [8, 4]),
        (1024, 8, [8]),
        (32768, 1, [1]),
    ],
)
def test_topk_endpoint_encoder_seeds_are_unpromoted(
    dtype: torch.dtype, cols: int, k: int, expected_lanes: list[int]
) -> None:
    with FakeTensorMode():
        x = torch.empty((17, cols), dtype=dtype)
    bound = _allocating_topk._bind_isolated((x, k, True))
    spec = bound.config_spec
    default = spec.default_config()
    assert default["cute_topk_key_encoder"] == "dsl"
    assert default["cute_topk_defer_value_gathers"] is False
    assert default["cute_topk_merge_schedule"] == "sequential"
    assert spec.compiler_default_config is None
    seeds = [
        seed
        for seed in spec.compiler_seed_configs
        if seed.get("cute_topk_defer_value_gathers", False)
    ]
    assert [
        (seed["cute_topk_lanes_per_row"], seed["cute_topk_key_encoder"])
        for seed in seeds
    ] == [
        (lanes, encoder)
        for lanes in expected_lanes
        for encoder in ("dsl", "asm", "paired")
    ]
    for seed in seeds:
        spec.normalize(copy.deepcopy(seed))
        lanes = seed["cute_topk_lanes_per_row"]
        assert isinstance(lanes, int)
        rows = seed["cute_topk_rows_per_block"]
        assert isinstance(rows, int)
        assert rows * lanes == 128
        assert seed["cute_topk_vector_width"] == 8
        assert seed["cute_topk_output_vector_width"] == 4
        assert seed["cute_topk_value_mode"] == "decode"
        assert seed["cute_topk_key_dtype"] == "int32"
        assert seed["cute_topk_rank_mode"] == "ordinal"
        assert seed["cute_topk_selection_layout"] == "replicated"
        assert seed["cute_topk_sort_network"] == "compact_pruned"
        assert seed["cute_topk_merge_schedule"] == "sequential"
    generation = ConfigGeneration(spec)
    flattened = generation.seed_flat_config_pairs()
    normalized = copy.deepcopy(seeds)
    for config in normalized:
        spec.normalize(config)
    assert [
        config
        for _, config in flattened
        if config.get("cute_topk_defer_value_gathers", False)
    ] == normalized
    for flat, config in flattened:
        assert generation.unflatten(flat) == config


# Paired top-k tests.


def _prmt(word: np.ndarray, selectors: int) -> np.ndarray:
    result = np.zeros_like(word)
    for output_byte in range(4):
        selector = (selectors >> (4 * output_byte)) & 15
        byte = (word >> (8 * (selector & 3))) & 255
        if selector & 8:
            byte = np.where(byte & 128, 255, 0).astype(np.uint32)
        result |= byte << (8 * output_byte)
    return result


def _lop3(a: np.ndarray, b: np.ndarray, c: int, table: int) -> np.ndarray:
    result = np.zeros_like(a)
    for index in range(8):
        if table & (1 << index):
            result |= (
                (a if index & 4 else ~a)
                & (b if index & 2 else ~b)
                & np.uint32(c if index & 1 else c ^ 0xFFFFFFFF)
            )
    return result


@pytest.mark.parametrize("infinity", [0x7C00, 0x7F80])
@pytest.mark.parametrize("bits", range(1, 16))
@pytest.mark.parametrize("largest", [False, True])
def test_paired_ordinal_exhaustive(infinity: int, bits: int, largest: bool) -> None:
    low = np.arange(65536, dtype=np.uint32)
    # A bijection exercises every possible word in both halves; additional
    # extremes catch cross-half carry and mixed finite/exceptional pairs.
    partners = (
        (low * 32771 + 1) & 65535,
        np.zeros_like(low),
        np.full_like(low, 65535),
        np.full_like(low, infinity),
    )
    bias = 32767 - infinity
    mask = (1 << bits) - 1
    for high in partners:
        words = low | (high << 16)
        sign = _prmt(words, 0xBB99)
        rank = _lop3(words, sign, 0x7FFF7FFF, 0x78)
        magnitude = words & 0x7FFF7FFF
        assert bool(((magnitude & 65535) + bias < 65536).all())
        assert bool(((magnitude >> 16) + bias < 65536).all())
        nan_mask = _prmt(magnitude + np.uint32(bias | (bias << 16)), 0xBB99)
        rank = _lop3(rank, nan_mask, 0x7FFF7FFF, 0xB8)
        ranks = (_prmt(rank, 0x9910).view(np.int32), _prmt(rank, 0xBB32).view(np.int32))
        for actual, word in zip(ranks, (low, high), strict=True):
            mag = (word & 32767).astype(np.int32)
            expected = np.where(word & 32768, -1 - mag, mag)
            expected = np.where(mag > infinity, 32767, expected).astype(np.int32)
            np.testing.assert_array_equal(actual, expected)
        for column in (0, (mask // 2) & ~1, mask - 1):
            for half, (rank, word) in enumerate(zip(ranks, (low, high), strict=True)):
                ordered = rank if largest else -rank
                packed = (ordered << bits) | (mask - column - half)
                restored = packed >> bits
                if not largest:
                    restored = -restored
                np.testing.assert_array_equal(restored, rank)
                np.testing.assert_array_equal(
                    mask - (packed & mask), np.full_like(rank, column + half)
                )
                recovered = (restored ^ ((restored >> 31) & 32767)) & 65535
                non_nan = (word & 32767) <= infinity
                np.testing.assert_array_equal(recovered[non_nan], word[non_nan])
                assert bool((restored[~non_nan] == 32767).all())
                if bits <= 9:
                    np.testing.assert_array_equal(
                        packed.astype(np.float32).astype(np.int32), packed
                    )
                if bits <= 14:
                    floating = (packed + 0x40000000).view(np.float32)
                    assert bool(np.isfinite(floating).all())
                    assert bool((floating > 0).all())


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize(
    "cols,stride,vector,rank,key,active",
    [
        (128, 128, 8, "ordinal", "int32", True),
        (128, 130, 2, "ordinal", "int32", True),
        (128, 130, 2, "ordinal", "float32_bits", True),
        (130, 132, 2, "ordinal", "int32", True),
        (130, 132, 2, "ordinal", "float32_bits", True),
        (128, 128, 4, "ordinal", "float32", True),
        (65, 66, 8, "ordinal", "int32", False),
        (128, 130, 8, "ordinal", "int32", False),
        (128, 128, 1, "ordinal", "int32", False),
        (128, 128, 8, "signed", "int32", False),
        (128, 128, 8, "ordinal", "float32_native", False),
        (1, 1, 8, "ordinal", "int32", False),
    ],
)
def test_paired_encoder_codegen_scope(
    cols: int, stride: int, vector: int, rank: str, key: str, active: bool
) -> None:
    code = _code(
        5,
        cols,
        stride,
        min(cols, 32),
        4,
        lanes=2,
        input_vector=vector,
        key_encoder="paired",
        rank_mode=rank,
        key_dtype=key,
        value_mode="decode",
        defer_value_gathers=True,
    )
    assert ("_cute_encode_ordinal_pair" in code) == active
    if cols > 1 and key != "float32_native":
        assert "_cute_encode_ordered_topk" in code
    if key == "float32_native":
        assert "_cute_encode_ordered_topk" not in code
    if active:
        if vector == 2:
            assert "ir.VectorType.get([1], cutlass.Uint32.mlir_type)" not in code
            assert "dtype=cutlass.Uint32), cutlass.Uint32)" in code
            assert "(topk_pairs, topk_col," in code
        else:
            assert (
                f"ir.VectorType.get([{vector // 2}], cutlass.Uint32.mlir_type)" in code
            )
        assert f"topk_input_bits.iterator.alignment >= {vector * 2}" in code
        assert "topk_keys[topk_i + 1]" in code
        assert "_cute_encode_ordered_topk" in code  # Scalar ABI fallback.


@skipUnlessCuteAvailable("requires CuTe DSL")
def test_paired_encoder_keeps_wide_address_math() -> None:
    code = _code(
        1, 128, 2**35, 32, 4, lanes=2, key_encoder="paired", rank_mode="ordinal"
    )
    assert "_cute_encode_ordinal_pair" in code
    assert "topk_row * cutlass.Int64(34359738368)" in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "width,k,padding,offset,vector,key,layout",
    [
        (128, 32, 0, 0, 8, "int32", "replicated"),
        (128, 24, 2, 0, 2, "int32", "distributed"),
        (128, 24, 2, 0, 2, "float32_bits", "distributed"),
        (130, 24, 2, 0, 2, "int32", "distributed"),
        (130, 24, 2, 0, 2, "float32_bits", "distributed"),
        (65, 6, 1, 1, 4, "float32", "replicated"),
        (128, 32, 0, 1, 8, "int32", "replicated"),
    ],
)
def test_paired_encoder_exact_selected_bits(
    dtype: torch.dtype,
    index_dtype: torch.dtype,
    largest: bool,
    width: int,
    k: int,
    padding: int,
    offset: int,
    vector: int,
    key: str,
    layout: str,
) -> None:
    rows = 5
    x, storage = _layout_input(rows, width, dtype, padding, offset)
    original = storage.clone()
    value_storage = torch.full((rows * k + offset + 2,), 7, dtype=dtype, device=DEVICE)
    index_storage = torch.full(
        (rows * k + offset + 2,), -7, dtype=index_dtype, device=DEVICE
    )
    values = value_storage[offset : offset + rows * k].view(rows, k)
    indices = index_storage[offset : offset + rows * k].view(rows, k)
    code, _ = code_and_output(
        _extra_out_topk,
        (x, values, indices, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=2,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=vector,
        cute_topk_output_vector_width=4,
        cute_topk_key_encoder="paired",
        cute_topk_key_dtype=key,
        cute_topk_rank_mode="ordinal",
        cute_topk_value_mode="decode",
        cute_topk_selection_layout=layout,
        cute_topk_sort_network="compact_pruned",
        cute_topk_defer_value_gathers=True,
    )
    _assert_register_selection(code)
    _assert_topk_output(x, values, indices, k, largest, index_dtype=index_dtype)
    assert torch.equal(storage.view(torch.int16), original.view(torch.int16))
    assert bool((value_storage[:offset] == 7).all()) and bool(
        (value_storage[offset + rows * k :] == 7).all()
    )
    assert bool((index_storage[:offset] == -7).all()) and bool(
        (index_storage[offset + rows * k :] == -7).all()
    )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [False, True])
def test_paired_encoder_same_bound_alignment_transition(
    dtype: torch.dtype, largest: bool
) -> None:
    rows, width, k = 5, 128, 32
    _, source = _layout_input(rows, width, dtype, 0, 0)
    storage = torch.cat((source, source[-1:]))
    values = torch.empty((rows, k), dtype=dtype, device=DEVICE)
    indices = torch.empty((rows, k), dtype=torch.int32, device=DEVICE)
    x = storage[: rows * width].view(rows, width)
    bound = _extra_out_topk._bind_isolated((x, values, indices, k, largest))
    bound.set_config(
        helion.Config(
            block_sizes=[1],
            cute_topk_lanes_per_row=2,
            cute_topk_rows_per_block=4,
            cute_topk_vector_width=8,
            cute_topk_output_vector_width=4,
            cute_topk_key_encoder="paired",
            cute_topk_rank_mode="ordinal",
            cute_topk_value_mode="decode",
            cute_topk_defer_value_gathers=True,
        )
    )
    for offset in (0, 1, 0):
        x = storage[offset : offset + rows * width].view(rows, width)
        bound(x, values, indices, k, largest)
        _assert_topk_output(x, values, indices, k, largest, index_dtype=torch.int32)


# Wide output top-k tests.


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize(
    "k,lanes,requested,effective,wide",
    [
        (32, 1, 8, 8, True),
        (32, 2, 4, 4, True),
        (32, 2, 8, 8, True),
        (32, 4, 8, 8, True),
        (24, 2, 8, 8, True),
        (12, 2, 8, 4, True),
        (6, 2, 8, 2, False),
        (32, 32, 8, 1, False),
    ],
)
def test_distributed_wide_output_codegen(
    index_dtype: torch.dtype,
    largest: bool,
    k: int,
    lanes: int,
    requested: int,
    effective: int,
    wide: bool,
) -> None:
    code = _code(
        1,
        128,
        2**35,
        k,
        requested,
        lanes=lanes,
        largest=largest,
        selection_layout="distributed",
        key_dtype="float32_bits",
        rank_mode="ordinal",
        value_mode="decode",
        defer_value_gathers=True,
        index_dtype=index_dtype,
    )
    assert (
        "from helion.runtime.cute.register_layout import cyclic_to_vector_wide" in code
    ) == wide
    assert "topk_row = cutlass.Int64(" in code
    assert "topk_selected[topk_j]" in code  # Misaligned ABI keeps cyclic scalar stores.
    if effective == 1:
        assert "cute.autovec_copy" not in code
        return
    assert f"cute.assume(topk_output_offset, divby={effective})" in code
    assert f"topk_row * cutlass.Int64({k})" in code
    assert f"values.iterator.alignment >= {2 * effective}" in code
    if k != 1 << (k - 1).bit_length():
        assert f"topk_output_col < cutlass.Int32({k})" in code
        assert "topk_output_max_rank" not in code
    else:
        endpoint = 0 if largest else k // lanes - 1
        assert f"topk_output_keys[{endpoint}]" in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "k,lanes,vector,offset,key_dtype,defer",
    [
        (32, 1, 8, 0, "int32", True),
        (32, 2, 4, 0, "int32", False),
        (32, 2, 8, 0, "float32_bits", True),
        (24, 2, 4, 1, "int32", True),
        (24, 4, 8, 0, "float32_native", True),
        (6, 2, 8, 0, "int32", False),
    ],
)
def test_distributed_wide_output_exact_bits_and_tails(
    dtype: torch.dtype,
    index_dtype: torch.dtype,
    largest: bool,
    k: int,
    lanes: int,
    vector: int,
    offset: int,
    key_dtype: str,
    defer: bool,
) -> None:
    rows, width = 9, 128
    x, storage = _layout_input(rows, width, dtype, 2 * offset, offset)
    original = storage.clone()
    value_storage = torch.full((rows * k + 2,), 7, dtype=dtype, device=DEVICE)
    index_storage = torch.full((rows * k + 2,), -7, dtype=index_dtype, device=DEVICE)
    values = value_storage[offset : offset + rows * k].view(rows, k)
    indices = index_storage[offset : offset + rows * k].view(rows, k)
    code_and_output(
        _extra_out_topk,
        (x, values, indices, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=vector,
        cute_topk_selection_layout="distributed",
        cute_topk_sort_network="compact_pruned",
        cute_topk_key_encoder="paired",
        cute_topk_key_dtype=key_dtype,
        cute_topk_rank_mode="ordinal",
        cute_topk_value_mode="decode",
        cute_topk_defer_value_gathers=defer,
    )
    _assert_topk_output(x, values, indices, k, largest, index_dtype=index_dtype)
    assert torch.equal(original.view(torch.int16), storage.view(torch.int16))
    assert bool((value_storage[:offset] == 7).all())
    assert bool((value_storage[offset + rows * k :] == 7).all())
    assert bool((index_storage[:offset] == -7).all())
    assert bool((index_storage[offset + rows * k :] == -7).all())


# Test autotune final benchmark top-k tests.


# Fused softmax of the selected logits.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _softmax_row_topk(
    x: torch.Tensor,
    k: int,
    largest: hl.constexpr,
    index_dtype: torch.dtype = torch.int64,
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    values = torch.empty((x.size(0), k), dtype=x.dtype, device=x.device)
    indices = torch.empty((x.size(0), k), dtype=index_dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        vals, idx = torch.topk(x[row, :], k, dim=-1, largest=largest, sorted=True)
        values[row, :] = torch.softmax(vals.to(torch.float32), dim=-1).to(x.dtype)
        indices[row, :] = idx
    return values, indices


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
@pytest.mark.parametrize(
    "k,lanes,vector", [(1, 16, 1), (3, 8, 1), (8, 16, 4), (32, 2, 8)]
)
def test_topk_softmax_codegen(layout: str, k: int, lanes: int, vector: int) -> None:
    code = _code(
        5,
        64,
        64,
        k,
        vector,
        kernel=_softmax_row_topk,
        selection_layout=layout,
        lanes=lanes,
        value_mode="decode",
        rank_mode="ordinal",
    )
    assert "import softmax_topk_values as" not in code
    assert "row_known_maximum" in code
    assert "sort_rank" not in code
    assert code.count("@cute.kernel") == 1


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("damage", ["none", "sum_axis", "max_axis"])
def test_topk_softmax_uses_row_local_epilogue(damage: str) -> None:
    x = torch.empty((5, 64), dtype=torch.bfloat16)
    bound = _softmax_row_topk._bind_isolated((x, 8, True))
    env, host = bound.env, bound.host_function
    assert host is not None
    with env, host:
        candidate = match_topk_root(host.device_ir.graphs, noncanonical_block_ids=set())
        assert candidate is not None and candidate.fragment_graph is not None
        assert (
            _match_direct_topk_root(host.device_ir.graphs, noncanonical_block_ids=set())
            is None
        )
        if damage != "none":
            target = (
                torch.ops.aten.amax.default
                if damage == "max_axis"
                else torch.ops.aten.sum.dim_IntList
            )
            node = next(
                node
                for graph in host.device_ir.graphs
                for node in graph.graph.nodes
                if node.target is target
            )
            node.args = (node.args[0], [0], *node.args[2:])
        result = match_topk_root(host.device_ir.graphs, noncanonical_block_ids=set())
        assert (result is not None) == (damage == "none")


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
@pytest.mark.parametrize(
    "n,k,lanes,vector", [(65, 3, 8, 1), (64, 8, 16, 4), (128, 32, 2, 8)]
)
def test_topk_softmax_fused_values(
    dtype: torch.dtype,
    largest: bool,
    layout: str,
    n: int,
    k: int,
    lanes: int,
    vector: int,
) -> None:
    storage = torch.randn((19, n + 2), dtype=dtype, device=DEVICE)
    x = storage[:, 1 : n + 1]
    x[0] = 0
    x[1] = -2
    x[2] = torch.linspace(-2, 0.05, n, device=DEVICE, dtype=dtype)
    x[3] = torch.finfo(dtype).max
    x[4] = -torch.finfo(dtype).max
    x[5, 0] = float("nan")
    x[6] = float("-inf")
    x[7] = float("inf")
    original = x.clone()
    code, (values, indices) = code_and_output(
        _softmax_row_topk,
        (x, k, largest, torch.int32),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=8,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=vector,
        cute_topk_selection_layout=layout,
        cute_topk_sort_network="compact_pruned",
        cute_topk_value_mode="decode",
        cute_topk_rank_mode="ordinal",
    )
    assert "import softmax_topk_values as" not in code
    assert values.dtype == dtype and indices.dtype == torch.int32
    selected = original.gather(1, indices.long())
    expected_logits = torch.topk(
        original, k, dim=-1, largest=largest, sorted=True
    ).values
    torch.testing.assert_close(
        selected, expected_logits, rtol=0, atol=0, equal_nan=True
    )
    expected = torch.softmax(selected.float(), dim=-1).to(dtype)
    torch.testing.assert_close(
        values,
        expected,
        rtol=0.008 if dtype == torch.bfloat16 else 0.001,
        atol=1e-6,
        equal_nan=True,
    )
    sorted_indices = indices.sort(dim=-1).values
    assert bool((sorted_indices[:, 1:] != sorted_indices[:, :-1]).all())
    assert torch.equal(x.view(torch.int16), original.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "k,lanes,layout,value_mode,key_dtype,largest",
    [
        (1, 16, "distributed", "gather", "int32", True),
        (64, 1, "replicated", "decode", "float32_native", False),
        (32, 4, "distributed", "decode", "float32_native", False),
        (8, 1, "distributed", "gather", "int32", True),
    ],
)
def test_topk_softmax_value_recovery(
    dtype: torch.dtype,
    k: int,
    lanes: int,
    layout: str,
    value_mode: str,
    key_dtype: str,
    largest: bool,
) -> None:
    x = torch.randn((5, 64), dtype=dtype, device=DEVICE)
    x[0, 0] = float("nan")
    x[1, 0] = float("inf")
    x[2] = -torch.finfo(x.dtype).max
    x[3] = 0
    original = x.clone()
    code, (values, indices) = code_and_output(
        _softmax_row_topk,
        (x, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=8,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=8,
        cute_topk_selection_layout=layout,
        cute_topk_sort_network="compact_pruned",
        cute_topk_value_mode=value_mode,
        cute_topk_key_dtype=key_dtype,
    )
    assert "import softmax_topk_values as" not in code
    assert indices.dtype == torch.int64
    logits = original.gather(1, indices)
    torch.testing.assert_close(
        logits,
        torch.topk(original, k, dim=-1, largest=largest).values,
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    torch.testing.assert_close(
        values,
        torch.softmax(logits.float(), dim=-1).to(x.dtype),
        rtol=0.008,
        atol=1e-6,
        equal_nan=True,
    )
    assert torch.equal(x.view(torch.int16), original.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("key_dtype", ["int32", "float32_native"])
@pytest.mark.parametrize("rank_mode", ["signed", "ordinal"])
def test_topk_softmax_decode_eliminates_value_gathers(
    key_dtype: str, rank_mode: str
) -> None:
    code = _code(
        5,
        64,
        64,
        8,
        4,
        kernel=_softmax_row_topk,
        value_mode="decode",
        key_dtype=key_dtype,
        rank_mode=rank_mode,
    )
    assert "import softmax_topk_values as" not in code
    assert "x[topk_row, topk_selected_index]" not in code


# Register selection composed with ordinary row expressions.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _pointwise_row_topk(
    x: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    values = torch.empty((x.size(0), k), dtype=torch.float32, device=x.device)
    indices = torch.empty((x.size(0), k), dtype=torch.int64, device=x.device)
    raw = torch.empty((x.size(0), k), dtype=x.dtype, device=x.device)
    for row in hl.tile(x.size(0)):
        vals, idx = torch.topk(x[row, :], k, dim=-1)
        values[row, :] = torch.sigmoid(vals.to(torch.float32) * 0.5 + 0.25)
        indices[row, :] = idx * 3 + 1
        raw[row, :] = vals
    return values, indices, raw


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _temperature_row_topk(
    x: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    values = torch.empty((x.size(0), k), dtype=torch.float32, device=x.device)
    indices = torch.empty((x.size(0), k), dtype=torch.int64, device=x.device)
    summary = torch.empty((x.size(0), 1), dtype=torch.float32, device=x.device)
    for row in hl.tile(x.size(0)):
        vals, idx = torch.topk(x[row, :], k, dim=-1)
        logits = vals.to(torch.float32) / 0.75
        values[row, :] = torch.softmax(logits, dim=-1)
        indices[row, :] = idx
        summary[row, :] = torch.sum(logits, dim=-1, keepdim=True)
    return values, indices, summary


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _preprocessed_row_topk(
    x: torch.Tensor, k: int, largest: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    values = torch.empty((x.size(0), k), dtype=x.dtype, device=x.device)
    indices = torch.empty((x.size(0), k), dtype=torch.int64, device=x.device)
    processed = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        source = (x[row, :].to(torch.float32) * 0.5 + -0.0).to(x.dtype)
        vals, idx = torch.topk(source, k, dim=-1, largest=largest)
        values[row, :] = vals
        indices[row, :] = idx
        processed[row, :] = source
    return values, indices, processed


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _broadcast_row_topk(
    x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    values = torch.empty((x.size(0), k), dtype=x.dtype, device=x.device)
    indices = torch.empty((x.size(0), k), dtype=torch.int64, device=x.device)
    for row in hl.tile(x.size(0)):
        source = (x[row, :].to(torch.float32) * scale[row, :] + bias[None, :]).to(
            x.dtype
        )
        vals, idx = torch.topk(source, k, dim=-1)
        values[row, :] = vals
        indices[row, :] = idx
    return values, indices


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _broadcast_reduction_row_topk(
    x: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    values = torch.empty((x.size(0), k), dtype=torch.float32, device=x.device)
    indices = torch.empty((x.size(0), k), dtype=torch.int64, device=x.device)
    for row in hl.tile(x.size(0)):
        vals, idx = torch.topk(x[row, :], k, dim=-1)
        values[row, :] = vals.to(torch.float32).sum(dim=-1, keepdim=True)
        indices[row, :] = idx
    return values, indices


def _composition_config(lanes: int, layout: str) -> dict[str, object]:
    return {
        "block_sizes": [1],
        "cute_topk_lanes_per_row": lanes,
        "cute_topk_rows_per_block": 4,
        "cute_topk_vector_width": 8,
        "cute_topk_output_vector_width": 4,
        "cute_topk_selection_layout": layout,
        "cute_topk_sort_network": "compact_pruned",
        "cute_topk_value_mode": "decode",
        "cute_topk_rank_mode": "ordinal",
    }


def _assert_composed_register_selection(code: str, layout: str) -> None:
    _assert_register_selection(code)
    assert "sort_rank" not in code
    assert code.count("@cute.kernel") == 1


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
@pytest.mark.parametrize(
    "kind", ["pointwise", "temperature", "preprocess", "broadcast", "broadcast_store"]
)
def test_topk_composition_codegen(kind: str, layout: str) -> None:
    x = torch.empty((5, 65), dtype=torch.bfloat16)
    kernel: helion.Kernel[Any]
    args: tuple[object, ...]
    if kind == "pointwise":
        kernel, args = _pointwise_row_topk, (x, 3)
    elif kind == "temperature":
        kernel, args = _temperature_row_topk, (x, 3)
    elif kind == "preprocess":
        kernel, args = _preprocessed_row_topk, (x, 3, True)
    elif kind == "broadcast_store":
        kernel, args = _broadcast_reduction_row_topk, (x, 3)
    else:
        scale = torch.empty((5, 1), dtype=torch.float32)
        bias = torch.empty((65,), dtype=torch.float32)
        kernel, args = _broadcast_row_topk, (x, scale, bias, 3)
    bound = kernel._bind_isolated(args)
    config = bound.config_spec.default_config()
    config.config.update(_composition_config(8, layout))
    _assert_composed_register_selection(bound.to_code(config), layout)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("destination_columns", [1, 3])
def test_topk_composition_rejects_mismatched_store_extent(
    destination_columns: int,
) -> None:
    x = torch.empty((5, 65), dtype=torch.bfloat16)
    bound = _temperature_row_topk._bind_isolated((x, 3))
    host = bound.host_function
    assert host is not None
    with bound.env, host:
        plan = match_topk_root(host.device_ir.graphs, noncanonical_block_ids=set())
        assert plan is not None and plan.fragment_graph is not None
        graph = plan.fragment_graph
        target = next(
            effect
            for effect in graph.stores
            if effect.args[0].meta["val"].dtype == torch.float32
            and effect.args[0].meta["val"].size(1) == destination_columns
        )
        replacement = (
            graph.source
            if destination_columns == 3
            else next(
                node
                for node in graph.root_graph.nodes
                if node.target is torch.ops.aten.div.Tensor and node.args[1] == 0.75
            )
        )
        target.args = (*target.args[:2], replacement, target.args[3])
        assert (
            match_topk_root(host.device_ir.graphs, noncanonical_block_ids=set()) is None
        )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "n,k,lanes,layout",
    [
        (65, 3, 8, "distributed"),
        (64, 8, 16, "replicated"),
        (64, 32, 2, "distributed"),
        (64, 32, 8, "distributed"),
    ],
)
def test_topk_composition_pointwise_outputs(
    n: int, k: int, lanes: int, layout: str
) -> None:
    x = torch.randn((9, n), dtype=torch.bfloat16, device=DEVICE)
    x[0, 0] = float("nan")
    x[1] = 0
    x[1, 1::2] = -0.0
    original = x.clone()
    code, (values, encoded_indices, raw) = code_and_output(
        _pointwise_row_topk, (x, k), **_composition_config(lanes, layout)
    )
    _assert_composed_register_selection(code, layout)
    assert values.dtype == torch.float32
    assert bool((encoded_indices.remainder(3) == 1).all())
    indices = (encoded_indices - 1) // 3
    _check_topk(x, original, (raw, indices), k, True)
    assert torch.equal(
        raw.view(torch.int16), original.gather(1, indices).view(torch.int16)
    )
    torch.testing.assert_close(
        values,
        torch.sigmoid(raw.float() * 0.5 + 0.25),
        rtol=2e-6,
        atol=1e-7,
        equal_nan=True,
    )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "n,k,lanes,layout",
    [
        (65, 3, 8, "replicated"),
        (64, 8, 16, "distributed"),
        (64, 32, 2, "distributed"),
        (64, 32, 8, "distributed"),
    ],
)
def test_topk_composition_temperature_and_row_reduction(
    n: int, k: int, lanes: int, layout: str
) -> None:
    x = torch.randn((5, n), dtype=torch.float16, device=DEVICE)
    x[0] = float("inf")
    x[1] = float("-inf")
    x[2, 0] = float("nan")
    original = x.clone()
    code, (values, indices, summary) = code_and_output(
        _temperature_row_topk, (x, k), **_composition_config(lanes, layout)
    )
    _assert_composed_register_selection(code, layout)
    assert values.dtype == summary.dtype == torch.float32
    selected = original.gather(1, indices)
    _check_topk(x, original, (selected, indices), k, True)
    logits = selected.float() / 0.75
    torch.testing.assert_close(
        values, torch.softmax(logits, dim=-1), rtol=2e-6, atol=1e-7, equal_nan=True
    )
    torch.testing.assert_close(
        summary, logits.sum(dim=-1, keepdim=True), equal_nan=True
    )
    # Inactive lanes must not contribute duplicate selected values or padding.
    finite_rows = torch.isfinite(logits).all(dim=-1)
    torch.testing.assert_close(
        values[finite_rows].sum(dim=-1), torch.ones_like(summary[finite_rows, 0])
    )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "k,lanes,layout", [(3, 8, "distributed"), (8, 16, "replicated")]
)
def test_topk_composition_broadcasts_row_reduction_store(
    k: int, lanes: int, layout: str
) -> None:
    x = torch.randn((5, 65), dtype=torch.float16, device=DEVICE)
    original = x.clone()
    code, (values, indices) = code_and_output(
        _broadcast_reduction_row_topk, (x, k), **_composition_config(lanes, layout)
    )
    _assert_composed_register_selection(code, layout)
    selected = original.gather(1, indices)
    _check_topk(x, original, (selected, indices), k, True)
    expected = selected.float().sum(dim=-1, keepdim=True).expand_as(values)
    torch.testing.assert_close(values, expected)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "largest,layout", [(True, "distributed"), (False, "replicated")]
)
def test_topk_composition_preprocess_preserves_selected_bits(
    dtype: torch.dtype, largest: bool, layout: str
) -> None:
    x = torch.randn((9, 65), dtype=dtype, device=DEVICE)
    x[0] = 0
    x[0, 1::2] = -0.0
    x[1] = float("nan")
    x[2, 0] = float("inf")
    x[2, 1] = float("-inf")
    original = x.clone()
    code, (values, indices, processed) = code_and_output(
        _preprocessed_row_topk, (x, 3, largest), **_composition_config(8, layout)
    )
    _assert_composed_register_selection(code, layout)
    expected = (original.float() * 0.5 + -0.0).to(dtype)
    torch.testing.assert_close(processed, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.signbit(processed[0]), torch.signbit(expected[0]))
    torch.testing.assert_close(
        values,
        torch.topk(expected, 3, dim=-1, largest=largest).values,
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    # Recovery must use the processed fragment, including its exceptional bits.
    assert torch.equal(
        values.view(torch.int16), processed.gather(1, indices).view(torch.int16)
    )
    assert torch.equal(x.view(torch.int16), original.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
def test_topk_composition_row_and_column_broadcast(layout: str) -> None:
    x = torch.randn((5, 65), dtype=torch.bfloat16, device=DEVICE)
    scale = torch.tensor([0.5, -2.0, 0.0, 1.0, 2.0], device=DEVICE).view(5, 1)
    bias = torch.arange(65, dtype=torch.float32, device=DEVICE) / 16
    original = x.clone()
    code, (values, indices) = code_and_output(
        _broadcast_row_topk, (x, scale, bias, 3), **_composition_config(8, layout)
    )
    _assert_composed_register_selection(code, layout)
    expected = (original.float() * scale + bias[None, :]).to(x.dtype)
    torch.testing.assert_close(
        values, torch.topk(expected, 3, dim=-1).values, rtol=0, atol=0
    )
    assert torch.equal(
        values.view(torch.int16), expected.gather(1, indices).view(torch.int16)
    )
    assert torch.equal(x.view(torch.int16), original.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "key_dtype,rank_mode,encoder,lanes",
    [
        ("float32_native", "ordinal", "dsl", 2),
        ("int32", "signed", "asm", 2),
        ("float32_bits", "ordinal", "paired", 8),
    ],
)
def test_topk_composition_encoding_preserves_processed_bits(
    key_dtype: str, rank_mode: str, encoder: str, lanes: int
) -> None:
    x = torch.randn((5, 64), dtype=torch.bfloat16, device=DEVICE)
    x[0] = float("nan")
    x[1] = 0
    x[1, 1::2] = -0.0
    x[2, :2] = torch.tensor([float("inf"), float("-inf")], device=DEVICE)
    config = _composition_config(lanes, "distributed")
    config.update(
        cute_topk_key_dtype=key_dtype,
        cute_topk_rank_mode=rank_mode,
        cute_topk_key_encoder=encoder,
        cute_topk_output_vector_width=8,
    )
    code, (values, indices, processed) = code_and_output(
        _preprocessed_row_topk, (x, 32, False), **config
    )
    _assert_composed_register_selection(code, "distributed")
    torch.testing.assert_close(
        values,
        torch.topk(processed, 32, dim=-1, largest=False).values,
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert torch.equal(
        values.view(torch.int16), processed.gather(1, indices).view(torch.int16)
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _composed_out_topk(
    x: torch.Tensor, values: torch.Tensor, indices: torch.Tensor, k: int
) -> None:
    k = hl.specialize(k)
    for row in hl.tile(x.size(0)):
        vals, idx = torch.topk(x[row, :], k, dim=-1)
        values[row, :] = torch.sigmoid(vals.float())
        indices[row, :] = idx + 1


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_topk_composition_strided_memory_preserves_guards() -> None:
    rows, n, k = 9, 65, 3
    storage = torch.randn((rows, 2 * n + 2), device=DEVICE, dtype=torch.bfloat16)
    x = storage[:, 1 : 2 * n + 1 : 2]
    original = storage.clone()
    value_storage = torch.full((rows, 2 * k + 2), -17.0, device=DEVICE)
    index_storage = torch.full((rows, 3 * k + 2), -42, device=DEVICE, dtype=torch.int32)
    values = value_storage[:, 1 : 2 * k + 1 : 2]
    indices = index_storage[:, 1 : 3 * k + 1 : 3]
    code, _output = code_and_output(
        _composed_out_topk,
        (x, values, indices, k),
        **_composition_config(8, "distributed"),
    )
    _assert_composed_register_selection(code, "distributed")
    selected_indices = indices.long() - 1
    selected = x.gather(1, selected_indices)
    torch.testing.assert_close(
        selected, torch.topk(x, k, dim=-1).values, rtol=0, atol=0
    )
    torch.testing.assert_close(
        values, torch.sigmoid(selected.float()), rtol=2e-6, atol=1e-7
    )
    for target, stride, sentinel in ((value_storage, 2, -17), (index_storage, 3, -42)):
        written = torch.zeros_like(target, dtype=torch.bool)
        written[:, 1 : stride * k + 1 : stride] = True
        assert bool((target[~written] == sentinel).all())
    assert torch.equal(storage.view(torch.int16), original.view(torch.int16))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _reused_prologue_row_topk(
    x: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    values = torch.empty((x.size(0), k), dtype=torch.float32, device=x.device)
    indices = torch.empty((x.size(0), k), dtype=torch.int64, device=x.device)
    for row in hl.tile(x.size(0)):
        processed = x[row, :] * 0.5
        selected, idx = torch.topk(processed, k, dim=-1)
        values[row, :] = selected.float() + processed.float()
        indices[row, :] = idx
    return values, indices


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
def test_topk_composition_reused_prologue_codegen(layout: str) -> None:
    x = torch.empty((5, 64), dtype=torch.bfloat16)
    bound = _reused_prologue_row_topk._bind_isolated((x, 64))
    config = bound.config_spec.default_config()
    config.config.update(_composition_config(4, layout))
    code = bound.to_code(config)
    _assert_composed_register_selection(code, layout)
    if layout == "distributed":
        assert "import subgroup_vectorize as" in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
def test_topk_composition_reuses_prologue_across_layouts(layout: str) -> None:
    x = torch.randn((5, 64), dtype=torch.bfloat16, device=DEVICE)
    original = x.clone()
    # Input vectors contain eight values; selected-output vectors contain four.
    # Distributed selection also permutes physical lanes for contiguous stores.
    code, (values, indices) = code_and_output(
        _reused_prologue_row_topk, (x, 64), **_composition_config(4, layout)
    )
    _assert_composed_register_selection(code, layout)
    processed = original * 0.5
    selected = processed.gather(1, indices)
    torch.testing.assert_close(
        selected, torch.topk(processed, 64, dim=-1).values, rtol=0, atol=0
    )
    torch.testing.assert_close(
        indices.sort(dim=-1).values,
        torch.arange(64, device=DEVICE).expand_as(indices),
    )
    torch.testing.assert_close(
        values, selected.float() + processed.float(), rtol=0, atol=0
    )
    assert torch.equal(x.view(torch.int16), original.view(torch.int16))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _expanded_row_topk(
    x: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    expanded_output = torch.empty_like(x, dtype=torch.float32)
    sums = torch.empty((x.size(0), 1), dtype=torch.float32, device=x.device)
    indices = torch.empty((x.size(0), k), dtype=torch.int64, device=x.device)
    for row in hl.tile(x.size(0)):
        source = x[row, :]
        values, idx = torch.topk(source, k, dim=-1)
        total = values.float().sum(-1, keepdim=True)
        expanded = total.expand_as(source)
        expanded_output[row, :] = expanded
        sums[row, :] = expanded.sum(-1, keepdim=True)
        indices[row, :] = idx
    return expanded_output, sums, indices


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _integer_boolean_reduction_row_topk(
    x: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    index_sum = torch.empty((x.size(0), 1), dtype=torch.int64, device=x.device)
    positive = torch.empty((x.size(0), 1), dtype=torch.bool, device=x.device)
    all_positive = torch.empty((x.size(0), 1), dtype=torch.bool, device=x.device)
    indices = torch.empty((x.size(0), k), dtype=torch.int64, device=x.device)
    for row in hl.tile(x.size(0)):
        values, idx = torch.topk(x[row, :], k, dim=-1)
        index_sum[row, :] = (idx * 2**40).sum(-1, keepdim=True)
        positive[row, :] = (values > 0).amax(-1, keepdim=True)
        all_positive[row, :] = (values > 0).amin(-1, keepdim=True)
        indices[row, :] = idx
    return index_sum, positive, all_positive, indices


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
@pytest.mark.parametrize("kind", ["expand", "integer_boolean"])
def test_topk_composition_reduction_shapes_codegen(kind: str, layout: str) -> None:
    kernel = (
        _expanded_row_topk if kind == "expand" else _integer_boolean_reduction_row_topk
    )
    bound = kernel._bind_isolated((torch.empty((5, 65), dtype=torch.bfloat16), 3))
    config = bound.config_spec.default_config()
    config.config.update(_composition_config(8, layout))
    _assert_composed_register_selection(bound.to_code(config), layout)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
def test_topk_composition_rejects_shape_query_arithmetic() -> None:
    bound = _expanded_row_topk._bind_isolated(
        (torch.empty((5, 65), dtype=torch.bfloat16), 3)
    )
    host = bound.host_function
    assert host is not None
    with bound.env, host:
        plan = match_topk_root(host.device_ir.graphs, noncanonical_block_ids=set())
        assert plan is not None and plan.fragment_graph is not None
        graph = plan.fragment_graph.root_graph
        query = next(
            node for node in graph.nodes if node.target is torch.ops.aten.sym_size.int
        )
        view = next(iter(query.users))
        with graph.inserting_before(view):
            arithmetic = graph.call_function(operator.add, (query, 1))
            arithmetic.meta["val"] = query.meta["val"] + 1
        view.update_arg(
            1, [arithmetic if value is query else value for value in view.args[1]]
        )
        assert (
            match_topk_root(host.device_ir.graphs, noncanonical_block_ids=set()) is None
        )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
def test_topk_composition_expand_as_uses_logical_extent(layout: str) -> None:
    x = torch.randn((5, 65), dtype=torch.bfloat16, device=DEVICE)
    original = x.clone()
    code, (expanded, sums, indices) = code_and_output(
        _expanded_row_topk, (x, 3), **_composition_config(8, layout)
    )
    _assert_composed_register_selection(code, layout)
    selected = original.gather(1, indices)
    torch.testing.assert_close(selected, torch.topk(original, 3, dim=-1).values)
    expected = selected.float().sum(-1, keepdim=True).expand_as(original)
    torch.testing.assert_close(expanded, expected, rtol=0, atol=0)
    # The reduction must visit 65 columns, excluding the padded tile entries.
    torch.testing.assert_close(sums, expected.sum(-1, keepdim=True))
    assert torch.equal(x.view(torch.int16), original.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
def test_topk_composition_integer_boolean_reductions(layout: str) -> None:
    x = torch.randn((5, 65), dtype=torch.bfloat16, device=DEVICE)
    x[0] = -x[0].abs() - 1
    x[1] = x[1].abs() + 1
    x[2] = 0
    x[4] = -x[4].abs() - 1
    x[4, 0] = 1
    original = x.clone()
    code, (index_sum, positive, all_positive, indices) = code_and_output(
        _integer_boolean_reduction_row_topk,
        (x, 3),
        **_composition_config(8, layout),
    )
    _assert_composed_register_selection(code, layout)
    selected = original.gather(1, indices)
    torch.testing.assert_close(selected, torch.topk(original, 3, dim=-1).values)
    torch.testing.assert_close(index_sum, (indices * 2**40).sum(-1, keepdim=True))
    torch.testing.assert_close(positive, (selected > 0).amax(-1, keepdim=True))
    torch.testing.assert_close(all_positive, (selected > 0).amin(-1, keepdim=True))
    assert torch.equal(x.view(torch.int16), original.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("output_vector", [1, 8])
def test_replicated_selection_output_routing_native(output_vector: int) -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(20261008)
    x = torch.randn((64, 64), dtype=torch.bfloat16, device=DEVICE, generator=generator)
    x[0].zero_()
    x[1] = (torch.arange(64, device=DEVICE) % 5 - 2).to(x.dtype)
    original = x.clone()
    kernel = helion.kernel(
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_structural_policy=_TOPK_STRUCTURAL_POLICY,
    )(_selected_topk)
    code, (values, indices) = code_and_output(
        kernel,
        (x, 16, False),
        block_sizes=[32],
        cute_topk_lanes_per_row=2,
        cute_topk_rows_per_block=16,
        cute_topk_vector_width=8,
        cute_topk_output_vector_width=output_vector,
        cute_topk_key_dtype="int32",
        cute_topk_key_encoder="paired",
        cute_topk_rank_mode="ordinal",
        cute_topk_value_mode="gather",
        cute_topk_selection_layout="replicated",
        cute_topk_sort_network="compact_pruned",
        cute_topk_merge_schedule="balanced",
    )
    _assert_register_selection(code)
    # Conditional rmem copies repeated the first eight results in both halves.
    expected = torch.argsort(original, dim=-1, descending=True, stable=True)[:, :16]
    assert torch.equal(indices.to(torch.int64), expected)
    assert torch.equal(values, original.gather(1, expected))
    assert torch.equal(x.view(torch.int16), original.view(torch.int16))


# Full sorting reuses the same ordered-key networks and subgroup layouts.


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_network_sort(
    x: torch.Tensor, descending: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.empty_like(x)
    indices = torch.empty(x.shape, dtype=torch.int64, device=x.device)
    for row in hl.tile(x.size(0)):
        vals, idx = torch.sort(x[row, :], dim=-1, descending=descending)
        values[row, :] = vals
        indices[row, :] = idx
    return values, indices


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("descending", [False, True])
def test_sort_uses_register_selection_network(descending: bool) -> None:
    with FakeTensorMode():
        x = torch.empty((17, 65), dtype=torch.bfloat16)
    bound = _row_network_sort._bind_isolated((x, descending))
    assert bound.config_spec.cute_topk_search_enabled
    config = bound.config_spec.default_config()
    config.config.update(_composition_config(8, "distributed"))
    code = bound.to_code(config)
    _assert_register_selection(code)
    assert "sort_rank" not in code
    assert code.count("@cute.kernel") == 1


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("width,layout", [(16, "replicated"), (65, "distributed")])
def test_sort_network_preserves_values_indices_and_ties(
    dtype: torch.dtype, descending: bool, width: int, layout: str
) -> None:
    x = _inputs(width, dtype)
    original = x.clone()
    config = _composition_config(4, layout)
    # Sorting must enforce first-index ties even when a top-k tuning choice
    # would otherwise distinguish signed zeros or use native float keys.
    config.update(cute_topk_rank_mode="ordinal", cute_topk_key_dtype="float32_native")
    code, (values, indices) = code_and_output(
        _row_network_sort, (x, descending), **config
    )
    assert "sort_rank" not in code
    expected_values, expected_indices = torch.sort(
        original, dim=-1, descending=descending, stable=True
    )
    torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
    assert torch.equal(values.view(torch.int16), expected_values.view(torch.int16))
    assert torch.equal(x.view(torch.int16), original.view(torch.int16))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _normalized_input_row_topk(
    x: torch.Tensor, k: int, mode: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    values = torch.empty((x.size(0), k), dtype=x.dtype, device=x.device)
    indices = torch.empty((x.size(0), k), dtype=torch.int64, device=x.device)
    normalized = torch.empty_like(x)
    for row in hl.tile(x.size(0)):
        source = x[row, :].to(torch.float32)
        if mode == "rms":
            variance = torch.sum(source * source, dim=-1, keepdim=True) / x.size(1)
            source = source * torch.rsqrt(variance + 1e-5)
        else:
            source = torch.softmax(source, dim=-1)
        source = source.to(x.dtype)
        vals, idx = torch.topk(source, k, dim=-1)
        values[row, :] = vals
        indices[row, :] = idx
        normalized[row, :] = source
    return values, indices, normalized


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("mode", ["rms", "softmax"])
@pytest.mark.parametrize("width", [64, 65])
def test_topk_composition_reduces_input_and_vectorizes(mode: str, width: int) -> None:
    x = torch.empty((5, width), dtype=torch.bfloat16)
    bound = _normalized_input_row_topk._bind_isolated((x, 8, mode))
    code = bound.to_code(_composition_config(4, "distributed"))
    _assert_composed_register_selection(code, "distributed")
    assert "row_reduce" in code
    assert ("cute.autovec_copy(row_transfer_memory" in code) == (width == 64)
    assert "subgroup_vectorize" in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ["rms", "softmax"])
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
@pytest.mark.parametrize("width", [64, 65])
def test_topk_composition_normalizes_before_selection(
    mode: str, layout: str, width: int
) -> None:
    x = torch.randn((5, width), dtype=torch.bfloat16, device=DEVICE)
    x[0].zero_()
    x[0, 1::2] = -0.0
    x[1, 1] = float("nan")
    x[2, 2] = float("inf")
    original = x.clone()
    code, (values, indices, normalized) = code_and_output(
        _normalized_input_row_topk, (x, 8, mode), **_composition_config(4, layout)
    )
    _assert_composed_register_selection(code, layout)
    selected = normalized.gather(1, indices)
    assert torch.equal(values.view(torch.int16), selected.view(torch.int16))
    torch.testing.assert_close(
        values, torch.topk(normalized, 8, dim=-1).values, rtol=0, atol=0, equal_nan=True
    )
    source = original.float()
    expected = (
        source * torch.rsqrt((source * source).sum(-1, keepdim=True) / width + 1e-5)
        if mode == "rms"
        else torch.softmax(source, dim=-1)
    ).to(x.dtype)
    torch.testing.assert_close(
        normalized, expected, rtol=0.004, atol=1e-7, equal_nan=True
    )
    assert torch.equal(x.view(torch.int16), original.view(torch.int16))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _ordered_extremum_row_topk(
    x: torch.Tensor, k: int, largest: hl.constexpr, scale: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    values = torch.empty((x.size(0), k), dtype=torch.float32, device=x.device)
    indices = torch.empty((x.size(0), k), dtype=torch.int64, device=x.device)
    for row in hl.tile(x.size(0)):
        vals, idx = torch.topk(x[row, :], k, dim=-1, largest=largest)
        logits = vals.to(torch.float32) * scale
        exponents = torch.exp(logits - torch.amax(logits, dim=-1, keepdim=True))
        values[row, :] = exponents / torch.sum(exponents, dim=-1, keepdim=True)
        indices[row, :] = idx
    return values, indices


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("k,lanes,scale", [(6, 4, 1.0), (32, 2, 1.0), (6, 4, -1.0)])
def test_topk_composition_order_fact_is_monotone(
    k: int, lanes: int, scale: float
) -> None:
    x = torch.empty((5, 64), dtype=torch.bfloat16)
    bound = _ordered_extremum_row_topk._bind_isolated((x, k, False, scale))
    code = bound.to_code(_composition_config(lanes, "distributed"))
    _assert_composed_register_selection(code, "distributed")
    assert ("row_known_maximum" in code) == (scale > 0)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("k,lanes,scale", [(6, 4, 1.0), (32, 2, 1.0), (6, 4, -1.0)])
@pytest.mark.parametrize("largest", [False, True])
def test_topk_composition_ordered_maximum_owner(
    k: int, lanes: int, scale: float, largest: bool
) -> None:
    x = torch.randn((5, 64), dtype=torch.bfloat16, device=DEVICE)
    x[0].zero_()
    x[0, 1::2] = -0.0
    x[1].fill_(float("nan"))
    x[2, 0] = float("inf")
    x[2, 1] = -float("inf")
    original = x.clone()
    code, (values, indices) = code_and_output(
        _ordered_extremum_row_topk,
        (x, k, largest, scale),
        **_composition_config(lanes, "distributed"),
    )
    _assert_composed_register_selection(code, "distributed")
    assert ("row_known_maximum" in code) == (scale > 0)
    selected = original.gather(1, indices)
    torch.testing.assert_close(
        selected,
        torch.topk(original, k, dim=-1, largest=largest).values,
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    expected = torch.softmax(selected.float() * scale, dim=-1)
    torch.testing.assert_close(values, expected, rtol=2e-6, atol=1e-7, equal_nan=True)
    assert torch.equal(x.view(torch.int16), original.view(torch.int16))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _semantic_row_topk(
    x: torch.Tensor, k: int, mode: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    values = torch.empty((x.size(0), k), dtype=torch.float32, device=x.device)
    indices = torch.empty((x.size(0), k), dtype=torch.int64, device=x.device)
    for row in hl.tile(x.size(0)):
        vals, idx = torch.topk(x[row, :], k, dim=-1)
        logits = vals.to(torch.float32) / 0.75
        if mode == "softmax":
            shifted = logits - torch.amax(logits, dim=-1, keepdim=True)
            exponents = torch.exp(shifted)
            output = exponents / torch.sum(exponents, dim=-1, keepdim=True)
        elif mode == "logsumexp":
            maximum = torch.amax(logits, dim=-1, keepdim=True)
            output = maximum + torch.log(
                torch.sum(torch.exp(logits - maximum), dim=-1, keepdim=True)
            )
        elif mode == "reciprocal":
            output = torch.reciprocal(logits)
        elif mode == "zero_sign":
            output = (torch.reciprocal(logits) < 0).to(torch.float32)
        elif mode == "predicate":
            output = (logits == 0).to(torch.float32)
        else:
            output = logits
        values[row, :] = output
        indices[row, :] = idx
    return values, indices


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _indices_only_composed_row_topk(x: torch.Tensor, k: int) -> torch.Tensor:
    k = hl.specialize(k)
    indices = torch.empty((x.size(0), k), dtype=torch.int64, device=x.device)
    for row in hl.tile(x.size(0)):
        vals, idx = torch.topk(x[row, :], k, dim=-1)
        indices[row, :] = idx * 2 + 1
    return indices


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize(
    "mode", ["softmax", "logsumexp", "reciprocal", "zero_sign", "predicate", "raw"]
)
def test_topk_composition_value_observability_codegen(mode: str) -> None:
    x = torch.empty((5, 64), dtype=torch.bfloat16)
    bound = _semantic_row_topk._bind_isolated((x, 8, mode))
    config = _composition_config(4, "distributed")
    config["cute_topk_rank_mode"] = "signed"
    code = bound.to_code(config)
    _assert_composed_register_selection(code, "distributed")
    assert ("row_known_maximum" in code) == (mode in ("softmax", "logsumexp"))
    assert ("if not topk_value_decodable" in code) == (
        mode not in ("softmax", "predicate")
    )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
def test_topk_composition_indices_only_avoids_value_recovery() -> None:
    x = torch.empty((5, 64), dtype=torch.bfloat16)
    bound = _indices_only_composed_row_topk._bind_isolated((x, 8))
    code = bound.to_code(_composition_config(4, "distributed"))
    _assert_composed_register_selection(code, "distributed")
    assert "row_selected_values" not in code
    assert "topk_value_decodable" not in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "mode", ["softmax", "logsumexp", "reciprocal", "zero_sign", "predicate", "raw"]
)
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
def test_topk_composition_value_observability(mode: str, layout: str) -> None:
    x = torch.randn((5, 64), dtype=torch.bfloat16, device=DEVICE)
    x[0].zero_()
    x[0, 1::2] = -0.0
    x[1, :8] = float("nan")
    x[2, :8] = float("inf")
    original = x.clone()
    config = _composition_config(4, layout)
    config["cute_topk_rank_mode"] = "signed"
    code, (values, indices) = code_and_output(
        _semantic_row_topk, (x, 8, mode), **config
    )
    _assert_composed_register_selection(code, layout)
    selected = original.gather(1, indices)
    torch.testing.assert_close(
        selected, torch.topk(original, 8, dim=-1).values, rtol=0, atol=0, equal_nan=True
    )
    logits = selected.float() / 0.75
    if mode == "softmax":
        expected = torch.softmax(logits, dim=-1)
    elif mode == "logsumexp":
        maximum = logits.amax(dim=-1, keepdim=True)
        expected = (
            maximum + torch.log(torch.exp(logits - maximum).sum(-1, keepdim=True))
        ).expand_as(logits)
    elif mode == "reciprocal":
        expected = logits.reciprocal()
    elif mode == "zero_sign":
        expected = (logits.reciprocal() < 0).float()
    elif mode == "predicate":
        expected = (logits == 0).float()
    else:
        expected = logits
    torch.testing.assert_close(values, expected, rtol=2e-6, atol=1e-7, equal_nan=True)
    if mode in ("raw", "reciprocal"):
        assert torch.equal(values[0].signbit(), expected[0].signbit())
    assert torch.equal(x.view(torch.int16), original.view(torch.int16))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.float32, torch.float16, torch.bfloat16, torch.int64]
)
@pytest.mark.parametrize("lanes,vector", [(2, 8), (4, 2), (32, 4)])
def test_subgroup_vector_layout_preserves_bits(
    dtype: torch.dtype, lanes: int, vector: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cutlass
    import cutlass.cute as cute

    from helion.runtime.cute.register_layout import subgroup_vectorize

    # CuTe resolves deferred annotations through the original function globals.
    monkeypatch.setitem(globals(), "cutlass", cutlass)

    @cute.kernel
    def _subgroup_vector_layout_kernel(
        x,
        output,
        lanes: cutlass.Constexpr,
        vector: cutlass.Constexpr,
        registers: cutlass.Constexpr,
    ):
        thread = cute.arch.lane_idx()
        row = thread // lanes
        lane = thread % lanes
        fragment = cute.make_rmem_tensor(registers, x.element_type)
        for index in cutlass.range_constexpr(registers):
            fragment[index] = x[row, index * lanes + lane]
        grouped, output_lane = subgroup_vectorize(fragment, vector, lanes)
        for index in cutlass.range_constexpr(registers):
            column = (index // vector * lanes + output_lane) * vector + index % vector
            output[row, column] = grouped[index]

    registers = 16
    shape = (32 // lanes, registers * lanes)
    if dtype in (torch.float16, torch.bfloat16):
        words = torch.arange(32 * registers, device=DEVICE, dtype=torch.int16)
        words[:8] = torch.tensor(
            [0, -32768, 0x7FC1, -47, 0x7F80, -128, 0x7E01, -511],
            device=DEVICE,
            dtype=torch.int16,
        )
        x = words.view(dtype).reshape(shape)
    else:
        x = torch.arange(32 * registers, device=DEVICE).to(dtype).reshape(shape)
    output = torch.empty_like(x)
    default_cute_launcher(
        _subgroup_vector_layout_kernel,
        (1,),
        x,
        output,
        lanes,
        vector,
        registers,
        block=(8, 4, 1),
    )
    assert torch.equal(x.view(torch.uint8), output.view(torch.uint8))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
@pytest.mark.parametrize(
    "value_mode,rank_mode", [("gather", "signed"), ("decode", "ordinal")]
)
def test_fp32_ordered_selection_preserves_full_precision(
    largest: bool, layout: str, value_mode: str, rank_mode: str
) -> None:
    x = _inputs(65, torch.float32)
    # Adjacent FP32 values share the same BF16/FP16 representation.
    x[8] = (torch.arange(65, device=x.device).float() * 2**-23 + 1).flip(0)
    x[9] = (torch.arange(65, device=x.device).float() + 1) * 2**-149
    x[10] = -x[9]
    x[11].view(torch.int32)[:8] = torch.tensor(
        [
            0x7FC00001,
            0x7FC00002,
            -4194303,
            -4194302,
            0,
            -2147483648,
            0x7F800000,
            -8388608,
        ],
        device=x.device,
        dtype=torch.int32,
    )
    original = x.clone()
    code, (values, indices) = code_and_output(
        _row_topk,
        (x, 8, largest),
        block_sizes=[8],
        cute_topk_selection_layout=layout,
        cute_topk_value_mode=value_mode,
        cute_topk_rank_mode=rank_mode,
        cute_topk_lanes_per_row=8,
        cute_topk_vector_width=4,
        cute_topk_output_vector_width=4,
    )
    assert "cutlass.Int64(-9223372036854775808)" in code
    _assert_topk_output(original, values, indices, 8, largest)
    assert torch.equal(x.view(torch.int32), original.view(torch.int32))


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fp32_masked_row_topk(
    x: torch.Tensor, lengths: torch.Tensor, k: int, next_n: int
) -> torch.Tensor:
    k = hl.specialize(k)
    next_n = hl.specialize(next_n)
    output = torch.empty((x.size(0), k), device=x.device, dtype=torch.int32)
    for row in hl.tile(x.size(0)):
        count = lengths[row.index // next_n] - next_n + row.index % next_n + 1
        col = hl.arange(x.size(1))
        masked = torch.where(col[None, :] < count[:, None], x[row, :], float("-inf"))
        _, indices = torch.topk(masked, k, dim=-1)
        output[row, :] = torch.where(indices < count[:, None], indices, -1).to(
            torch.int32
        )
    return output


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("next_n", [1, 2])
def test_fp32_ordered_selection_runtime_lengths(next_n: int) -> None:
    x = torch.randn((6 * next_n, 64), device=DEVICE)
    x[0] = float("-inf")
    x[1] = float("-inf")
    lengths = torch.tensor([0, 3, 8, 64, 65, 81], device=DEVICE, dtype=torch.int32)
    code, indices = code_and_output(
        _fp32_masked_row_topk,
        (x, lengths, 8, next_n),
        block_sizes=[8],
        cute_topk_selection_layout="distributed",
        cute_topk_lanes_per_row=8,
    )
    assert "cutlass.Int64(-9223372036854775808)" in code
    for r in range(x.size(0)):
        n = max(0, min(64, int(lengths[r // next_n]) - next_n + r % next_n + 1))
        real = indices[r][indices[r] >= 0].long()
        assert len(real) == min(8, n) and len(real.unique()) == len(real)
        assert bool((real < n).all())
        torch.testing.assert_close(
            x[r, real].sort().values,
            x[r, :n].topk(min(8, n)).values.sort().values,
            rtol=0,
            atol=0,
        )


def test_fp32_ordered_selection_config_legality(topk_spec: ConfigSpec) -> None:
    spec = topk_spec
    spec.enable_cute_topk_search(torch.float32)
    config = spec.default_config()
    assert config["cute_topk_key_dtype"] == "int64"
    assert spec.cute_topk_choices["cute_topk_key_encoder"] == ("dsl",)
    assert spec.cute_topk_choices["cute_topk_defer_value_gathers"] == (False,)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "layout,value_mode", [("distributed", "decode"), ("replicated", "gather")]
)
def test_fp32_ordered_selection_transformed_values(
    layout: str, value_mode: str
) -> None:
    x = _inputs(65, torch.float32)
    x[0] = 1 + torch.arange(65, device=x.device).float() * 2**-23
    code, (values, indices, processed) = code_and_output(
        _preprocessed_row_topk,
        (x, 8, True),
        block_sizes=[8],
        cute_topk_selection_layout=layout,
        cute_topk_value_mode=value_mode,
        cute_topk_lanes_per_row=8,
        cute_topk_rank_mode="ordinal",
    )
    assert "cutlass.Int64(-9223372036854775808)" in code
    expected = x * 0.5 + -0.0
    torch.testing.assert_close(processed, expected, rtol=0, atol=0, equal_nan=True)
    _assert_topk_output(expected, values, indices, 8)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _mixed_dtype_row_topk(
    x: torch.Tensor,
    bias: torch.Tensor,
    k: int,
    producer: hl.constexpr,
    largest: hl.constexpr,
    normalize_selected: hl.constexpr,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = hl.specialize(k)
    values = torch.empty((x.size(0), k), device=x.device, dtype=torch.float32)
    indices = torch.empty((x.size(0), k), device=x.device, dtype=torch.int64)
    processed = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    for row in hl.tile(x.size(0)):
        scores = x[row, :].float()
        if producer != "cast":
            scores = scores + bias[:]
        if producer == "rounded":
            scores = scores.to(x.dtype).float()
        elif producer == "sigmoid":
            scores = torch.sigmoid(scores)
        elif producer == "softmax":
            scores = torch.softmax(scores, dim=-1)
        selected, selected_indices = torch.topk(scores, k, dim=-1, largest=largest)
        if normalize_selected:
            selected = torch.softmax(selected, dim=-1)
        values[row, :] = selected
        indices[row, :] = selected_indices
        processed[row, :] = scores
    return values, indices, processed


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "layout,value_mode", [("distributed", "decode"), ("replicated", "gather")]
)
@pytest.mark.parametrize(
    "producer,largest,normalize_selected",
    [
        ("affine", True, False),
        ("affine", False, False),
        ("rounded", True, False),
        ("sigmoid", True, False),
        ("softmax", True, False),
        ("affine", True, True),
        ("cast", True, False),
    ],
)
def test_mixed_dtype_selection_preserves_producer_boundaries(
    dtype: torch.dtype,
    layout: str,
    value_mode: str,
    producer: str,
    largest: bool,
    normalize_selected: bool,
) -> None:
    x = (
        _inputs(65, dtype)
        if producer == "cast"
        else torch.randn(17, 65, device=DEVICE, dtype=dtype)
    )
    # These FP32 score differences vanish when rounded to the input dtype.
    x[0].fill_(1)
    bias = torch.arange(65, device=x.device, dtype=torch.float32) * 2**-23
    code, (values, indices, processed) = code_and_output(
        _mixed_dtype_row_topk,
        (x, bias, 8, producer, largest, normalize_selected),
        block_sizes=[8],
        cute_topk_selection_layout=layout,
        cute_topk_value_mode=value_mode,
        cute_topk_rank_mode="ordinal",
        cute_topk_lanes_per_row=8,
        cute_topk_output_vector_width=4,
    )
    assert "cutlass.Int64(-9223372036854775808)" in code
    expected = x.float()
    if producer != "cast":
        expected = expected + bias
    if producer == "rounded":
        expected = expected.to(dtype).float()
    elif producer == "sigmoid":
        expected = expected.sigmoid()
    elif producer == "softmax":
        expected = expected.softmax(-1)
    tolerance = 2e-6 if producer in ("sigmoid", "softmax") else 0
    torch.testing.assert_close(
        processed, expected, rtol=tolerance, atol=tolerance, equal_nan=True
    )
    if normalize_selected:
        expected_values = expected.topk(8, largest=largest).values.softmax(-1)
        torch.testing.assert_close(values, expected_values, rtol=3e-5, atol=2e-6)
        assert bool(((indices >= 0) & (indices < x.size(1))).all())
        sorted_indices = indices.sort(-1).values
        assert bool((sorted_indices[:, 1:] != sorted_indices[:, :-1]).all())
        torch.testing.assert_close(
            values, processed.gather(1, indices).softmax(-1), rtol=3e-5, atol=2e-6
        )
    else:
        # Exact recovery is relative to the actual typed producer, including
        # NaN payloads and signed zero; numeric producer error is checked above.
        _assert_topk_output(processed, values, indices, 8, largest)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _narrowed_row_topk(
    x: torch.Tensor, selection_dtype: hl.constexpr
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.empty((x.size(0), 8), device=x.device, dtype=selection_dtype)
    indices = torch.empty((x.size(0), 8), device=x.device, dtype=torch.int64)
    for row in hl.tile(x.size(0)):
        scores = x[row, :].to(selection_dtype)
        selected, selected_indices = torch.topk(scores, 8, dim=-1)
        values[row, :] = selected
        indices[row, :] = selected_indices
    return values, indices


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["replicated", "distributed"])
def test_mixed_dtype_selection_uses_narrowed_keys(
    dtype: torch.dtype, layout: str
) -> None:
    x = _inputs(65, torch.float32)
    x[0] = 1 + torch.arange(65, device=x.device).float() * 2**-23
    code, (values, indices) = code_and_output(
        _narrowed_row_topk,
        (x, dtype),
        block_sizes=[8],
        cute_topk_selection_layout=layout,
        cute_topk_value_mode="decode",
        cute_topk_rank_mode="ordinal",
        cute_topk_lanes_per_row=8,
    )
    assert "cutlass.Int64(-9223372036854775808)" not in code
    _assert_topk_output(x.to(dtype), values, indices, 8)


# Guarded coarse-rank selection: execute the current tensor algorithm over
# a complete physical warp, checking against independently sorted full keys.
def _coarse_rank_tensor_model(
    rows,
    k,
    lanes,
    vector,
    *,
    recovery="direct",
    rank_only=False,
    payload_only=False,
    active_rows=None,
    index_bits=None,
    traced=False,
):
    # Pack ordinary row values into the compiler's lane/register input layout.
    # Selection, guarding and recovery all execute the production tensor
    # program; Python's complete key sort is an independent numerical oracle.
    n = len(rows[0])
    assert len(rows) == 32 // lanes and all(len(row) == n for row in rows)
    bits = (n - 1).bit_length() if index_bits is None else index_bits
    assert n <= 1 << bits
    mask = (1 << bits) - 1
    if active_rows is None:
        active_rows = len(rows)
    size = max(1 << (((n + lanes - 1) // lanes) - 1).bit_length(), vector)
    dtype = torch.int32 if rank_only else torch.int64
    keys = torch.full((32, size), -(1 << (31 if rank_only else 63)), dtype=dtype)
    for lane in range(32):
        for register in range(size):
            column = (
                (register // vector) * lanes + lane % lanes
            ) * vector + register % vector
            if lane // lanes < active_rows and column < n:
                rank = int(rows[lane // lanes][column])
                keys[lane, register] = (
                    rank if rank_only else (rank << bits) | (mask - column)
                )

    events = []
    select = selection_coarse.selection_network
    recover = selection_coarse.coarse_rank_recover

    def observe_selection(inputs, *args, **kwargs):
        events.append(("sort", inputs.dtype == torch.int64, inputs.shape[1]))
        return select(inputs, *args, **kwargs)

    def observe_recovery(inputs, *args, **kwargs):
        events.append(("recover", recovery, inputs.shape[1]))
        return recover(inputs, *args, **kwargs)

    def eager_cond(predicate, true_fn, false_fn, operands):
        # CPU branch observation only. Separate graph tests below execute the
        # real traced torch.cond; existing codegen tests check register lowering.
        assert predicate.ndim == 0
        return (true_fn if predicate.item() else false_fn)(*operands)

    if traced:
        graph = trace_coarse_rank_selection(
            32,
            size,
            k,
            dtype,
            "batcher",
            "sequential",
            bits,
            vector,
            lanes,
            recovery,
            payload_only,
        )
        result = graph(keys)
    else:
        with (
            patch.object(
                selection_coarse, "selection_network", side_effect=observe_selection
            ),
            patch.object(
                selection_coarse, "coarse_rank_recover", side_effect=observe_recovery
            ),
            patch.object(torch, "cond", side_effect=eager_cond),
        ):
            result = selection_coarse.coarse_rank_selection(
                keys,
                k,
                "batcher",
                "sequential",
                bits,
                vector,
                lanes,
                recovery,
                payload_only,
            )
    actual = []
    for group in range(active_rows):
        row = [
            int(result[group * lanes + rank % lanes, rank // lanes])
            for rank in range(k)
        ]
        expected = sorted(
            [
                (int(value) << bits) | (mask - column)
                for column, value in enumerate(rows[group])
            ],
            reverse=True,
        )[:k]
        if payload_only:
            assert [value & mask for value in row] == [
                value & mask for value in expected
            ]
        else:
            assert row == expected
        actual.append(row)
    return events, actual


@pytest.mark.parametrize("rank_only", [False, True])
@pytest.mark.parametrize("recovery", ["direct", "packed"])
@pytest.mark.parametrize("payload_only", [False, True])
def test_coarse_rank_tensor_graph_matches_eager(rank_only, recovery, payload_only):
    for cause in ("unique", "tie", "below", "other_row"):
        rows = [[0x3E800000 + i * 8192 for i in range(16)] for _ in range(8)]
        if cause == "tie":
            rows[-1] = [0x3F000001] * 16
        elif cause == "below":
            rows[-1] = rows[-1][:3] + [-0x00800000] + [-1] * 12
        elif cause == "other_row":
            rows[-1][-1] = 0x7F800000
        _, eager = _coarse_rank_tensor_model(
            rows,
            4,
            4,
            1,
            rank_only=rank_only,
            recovery=recovery,
            payload_only=payload_only,
        )
        _, traced = _coarse_rank_tensor_model(
            rows,
            4,
            4,
            1,
            rank_only=rank_only,
            recovery=recovery,
            payload_only=payload_only,
            traced=True,
        )
        assert traced == eager


@pytest.mark.parametrize(
    "lanes,vector,n,k",
    [
        (1, 1, 17, 8),
        (2, 8, 65, 32),
        (4, 4, 64, 8),
        (8, 8, 256, 8),
        (16, 8, 256, 8),
        (32, 4, 65, 64),
        (32, 1, 128, 1),
    ],
)
def test_coarse_rank_topk_actual_helper_exact_membership(lanes, vector, n, k):
    rng = random.Random(61)
    rows = []
    for _ in range(32 // lanes):
        ranks = [0x3E800000 + i * 8192 for i in range(n)]
        rng.shuffle(ranks)
        rows.append(ranks)
    events, _ = _coarse_rank_tensor_model(rows, k, lanes, vector)
    assert events[-1] == ("sort", True, max(1, k // lanes))
    # Reverse the sub-bucket order of selected ranks. Refinement must restore it.
    for row in rows:
        row[:k] = [0x3F000000 + i for i in reversed(range(k))]
    _coarse_rank_tensor_model(rows, k, lanes, vector)


@pytest.mark.parametrize("exception", [0, -1, 1, 0x7F800000, 0x7FFFFFFF, -0x7FFFFFFF])
def test_coarse_rank_topk_exceptional_and_cross_subgroup_fallback(exception):
    rows = [[0x3E800000 + i * 8192 for i in range(65)] for _ in range(4)]
    # Above-range ranks require a whole-warp fallback. Below-range ranks
    # cannot displace the eight accepted ranks, whose keys remain exact.
    rows[-1][3] = exception
    events, _ = _coarse_rank_tensor_model(rows, 8, 8, 4)
    assert events[-1] == ("sort", True, 16 if exception >= 0x7F800000 else 1)


@pytest.mark.parametrize("k", [1, 8, 16, 32, 64])
def test_coarse_rank_topk_duplicate_cutoff_and_padding(k):
    rows = [[0x3F000000 + i for i in range(65)] for _ in range(2)]
    events, _ = _coarse_rank_tensor_model(rows, k, 16, 8)
    assert events[-1] == ("sort", True, 8)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
def test_coarse_rank_topk_codegen_default_and_eligibility():
    x = torch.empty((5, 65), dtype=torch.float32)
    bound = _preprocessed_row_topk._bind_isolated((x, 8, True))
    config = bound.config_spec.default_config()
    config.config.update(_composition_config(16, "distributed"))
    old = bound.to_code(config)
    config.config["cute_topk_coarse_keys"] = False
    assert old == bound.to_code(config)
    config.config["cute_topk_coarse_keys"] = True
    with patch(
        "helion._compiler.cute.selection_coarse.trace_coarse_rank_selection",
        wraps=trace_coarse_rank_selection,
    ) as trace:
        new = bound.to_code(config)
    assert trace.call_count == 1
    assert trace.call_args.args[:1] == (32,)
    assert trace.call_args.args[6:9] == (7, 8, 16)
    _assert_register_selection(new)
    assert "import coarse_rank_topk as" not in new
    assert "register_conditional" in new
    for invalid in ("replicated",):
        config.config["cute_topk_selection_layout"] = invalid
        with pytest.raises(exc.InvalidConfig, match="complete FP32 distributed"):
            bound.to_code(config)
    half = _preprocessed_row_topk._bind_isolated((x.to(torch.float16), 8, True))
    cfg = half.config_spec.default_config()
    cfg.config.update(_composition_config(16, "distributed"))
    cfg.config["cute_topk_coarse_keys"] = True
    with pytest.raises(exc.InvalidConfig, match="complete FP32 distributed"):
        half.to_code(cfg)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("seed", [31, 61, 97])
@pytest.mark.usefixtures("_legacy_topk_seed_family")
def test_coarse_rank_topk_old_population_prefix_and_rng(seed):
    from test.cute_population_contracts import checked_initial_population

    from helion.autotuner.pattern_search import PatternSearch

    args = (torch.empty((5, 256), dtype=torch.float32), 8, True)
    results = []
    states = []
    with (
        _mock_cuda_unavailable(),
        _target(),
        _forbid_native_compile(),
        patch(
            "helion._compiler.autotuner_heuristics.register_topk_key_recovery_coverage"
        ),
    ):
        for enabled in (False, True):
            kernel = helion.kernel(
                _row_topk.fn, backend="cute", static_shapes=True, autotune_effort="full"
            )
            context = patch(
                "helion._compiler.autotuner_heuristics.register_topk_coarse_keys_coverage"
            )
            if not enabled:
                with context:
                    bound = _cpu_bind(kernel, args)
            else:
                bound = _cpu_bind(kernel, args)
            spec = bound.config_spec
            assert "cute_topk_coarse_keys" not in spec.default_config()
            assert all(
                "cute_topk_coarse_keys" not in s for s in spec.compiler_seed_configs
            )
            with bound.env:
                random.seed(seed)
                search = PatternSearch(bound, args, initial_population=100)
                population = checked_initial_population(search)
                configs = [
                    dict(search.config_gen.canonicalize_flat(x)[1]) for x in population
                ]
                results.append(configs)
                states.append(random.getstate())
            if enabled:
                group = next(
                    g
                    for g in spec.compiler_coverage_groups
                    if g.key == "cute_topk_coarse_keys"
                )
                assert group.deferred and group.legacy is False
                assert configs[-1]["cute_topk_coarse_keys"] is True
        assert results[1][:-1] == results[0]
        assert states[0] == states[1]


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "n,k,lanes,vector,largest",
    [
        (17, 3, 1, 1, True),
        (65, 8, 8, 4, True),
        (256, 8, 16, 8, True),
        (65, 32, 32, 4, False),
    ],
)
def test_coarse_rank_topk_native_exact_special_values(n, k, lanes, vector, largest):
    x = torch.arange(5 * n, device=DEVICE, dtype=torch.float32).reshape(5, n) / 128 + 1
    x[1].fill_(1)
    x[2, 0] = float("nan")
    x[2, 1] = float("inf")
    x[2, 2] = -float("inf")
    x[3, 0] = -0.0
    x[3, 1] = 0.0
    x[3, 2] = torch.finfo(torch.float32).tiny / 2
    if not largest:
        x = -x
    if not largest:
        storage = torch.zeros((5, n * 3), device=x.device, dtype=x.dtype)
        storage[:, ::3] = x
        x = storage[:, ::3]
    before = x.clone()
    code, (values, indices) = code_and_output(
        _row_topk,
        (x, k, largest),
        block_sizes=[1],
        cute_topk_lanes_per_row=lanes,
        cute_topk_rows_per_block=4,
        cute_topk_vector_width=vector,
        cute_topk_selection_layout="distributed",
        cute_topk_key_dtype="int64",
        cute_topk_coarse_keys=True,
    )
    _assert_register_selection(code)
    assert "import coarse_rank_topk as" not in code
    _assert_topk_output(x, values, indices, k, largest)
    assert torch.equal(x.view(torch.int32), before.view(torch.int32))


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("producer", ["sigmoid", "softmax"])
def test_coarse_rank_topk_native_composed_producer(producer):
    x = torch.randn((17, 256), device=DEVICE, dtype=torch.float32)
    bias = torch.linspace(-0.25, 0.25, 256, device=DEVICE)
    before = (x.clone(), bias.clone())
    code, (values, indices, processed) = code_and_output(
        _mixed_dtype_row_topk,
        (x, bias, 8, producer, True, False),
        block_sizes=[1],
        cute_topk_lanes_per_row=16,
        cute_topk_rows_per_block=8,
        cute_topk_vector_width=8,
        cute_topk_selection_layout="distributed",
        cute_topk_key_dtype="int64",
        cute_topk_coarse_keys=True,
    )
    _assert_register_selection(code)
    assert "import coarse_rank_topk as" not in code
    _assert_topk_output(processed, values, indices, 8, True)
    expected = (x + bias).sigmoid() if producer == "sigmoid" else (x + bias).softmax(-1)
    torch.testing.assert_close(processed, expected, rtol=3e-5, atol=2e-6)
    assert torch.equal(before[0], x) and torch.equal(before[1], bias)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("n,k", [(1, 1), (65, 65)])
def test_coarse_rank_topk_rejects_unsupported_extent(n, k):
    bound = _row_topk._bind_isolated((torch.ones(3, n), k, True))
    cfg = bound.config_spec.default_config()
    cfg.config.update(
        cute_topk_selection_layout="distributed", cute_topk_coarse_keys=True
    )
    with pytest.raises(exc.InvalidConfig, match="complete FP32 distributed"):
        bound.to_code(cfg)


def test_coarse_rank_topk_model_detects_missing_ambiguity_guard():
    rows = [[0x3F000000 + i for i in range(65)] for _ in range(2)]
    _coarse_rank_tensor_model(rows, 8, 16, 8)
    with (
        patch.object(
            selection_coarse, "coarse_rank_guard", return_value=torch.tensor(False)
        ),
        pytest.raises(AssertionError),
    ):
        _coarse_rank_tensor_model(rows, 8, 16, 8)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("layout", ["strided", "transposed", "offset"])
def test_coarse_rank_topk_resident_mapping_ignores_host_strides(layout):
    import re

    if layout == "strided":
        x = torch.empty((5, 195))[:, ::3]
    elif layout == "transposed":
        x = torch.empty((65, 5)).t()
    else:
        x = torch.empty((7, 67))[1:6, 1:66]
    bound = _row_topk._bind_isolated((x, 8, True))
    cfg = bound.config_spec.default_config()
    cfg.config.update(_composition_config(16, "distributed"))
    before = bound.to_code(cfg)
    cfg.config["cute_topk_coarse_keys"] = True
    after = bound.to_code(cfg)

    # Remove only the generated register program, whose arithmetic differs
    # between exact and coarse selection. Restoring full-key packing at each
    # Int32 rank store must recover every surrounding host address, mask and
    # physical register mapping, including sliced bases and transposed storage.
    class SelectionOnly(ast.NodeTransformer):
        rank_buffers = 0
        rank_stores = 0
        rank_padding = 0
        register_statements = 0

        def visit_Module(self, node):
            # Scalar lowering introduces ordinary v_* temporaries before the
            # register assignment. Remove their dependency chains as part of
            # the program, but retain all topk_* address and input setup.
            temporaries = {
                target.id: statement.value
                for statement in ast.walk(node)
                if isinstance(statement, ast.Assign)
                for target in statement.targets
                if isinstance(target, ast.Name)
                and re.fullmatch(r"v(?:_\d+)?", target.id)
            }
            self.program_temporaries = set()
            pending = [
                value.id
                for statement in ast.walk(node)
                if isinstance(statement, ast.Assign)
                and any(self.register_target(target) for target in statement.targets)
                for value in ast.walk(statement.value)
                if isinstance(value, ast.Name)
            ]
            while pending:
                name = pending.pop()
                if name not in temporaries or name in self.program_temporaries:
                    continue
                self.program_temporaries.add(name)
                pending.extend(
                    value.id
                    for value in ast.walk(temporaries[name])
                    if isinstance(value, ast.Name)
                )
            node = self.generic_visit(node)
            used = {value.id for value in ast.walk(node) if isinstance(value, ast.Name)}
            body = []
            for statement in node.body:
                if isinstance(statement, (ast.Import, ast.ImportFrom)) and not (
                    isinstance(statement, ast.ImportFrom)
                    and statement.module == "__future__"
                ):
                    statement.names = [
                        alias
                        for alias in statement.names
                        if (alias.asname or alias.name.split(".")[0]) in used
                    ]
                    if not statement.names:
                        continue
                body.append(statement)
            node.body = body
            return node

        def register_target(self, node):
            while isinstance(node, ast.Subscript):
                node = node.value
            return isinstance(node, ast.Name) and (
                node.id.startswith("register_") or node.id in self.program_temporaries
            )

        def visit_Assign(self, node):
            if all(self.register_target(target) for target in node.targets):
                self.register_statements += 1
                return None
            if (
                len(node.targets) == 1
                and ast.unparse(node.targets[0]) == "topk_selected"
            ):
                assert self.register_target(node.value)
                node.value = ast.Name("selected_program", ast.Load())
            if (
                len(node.targets) == 1
                and ast.unparse(node.targets[0]) == "topk_keys"
                and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "cute.make_rmem_tensor"
                and ast.unparse(node.value.args[1]) == "cutlass.Int32"
            ):
                self.rank_buffers += 1
                node.value.args[1] = ast.parse("cutlass.Int64", mode="eval").body
            if (
                len(node.targets) == 1
                and ast.unparse(node.targets[0]) == "topk_keys[topk_i]"
                and ast.unparse(node.value) == "topk_ordered"
            ):
                self.rank_stores += 1
                return ast.parse(
                    "topk_packed = ((cutlass.Int64(topk_ordered) << cutlass.Int32(7)) "
                    "| cutlass.Int64(cutlass.Int32(127) - topk_col))\n"
                    "topk_keys[topk_i] = topk_packed"
                ).body
            return self.generic_visit(node)

        def visit_If(self, node):
            if any(self.register_target(value) for value in ast.walk(node.test)):
                self.register_statements += 1
                return None
            return self.generic_visit(node)

        def visit_ImportFrom(self, node):
            for alias in node.names:
                if alias.asname:
                    alias.asname = re.sub(r"_[0-9a-f]{16}$", "", alias.asname)
            return node

        def visit_Name(self, node):
            node.id = re.sub(r"_[0-9a-f]{16}$", "", node.id)
            return node

        def visit_Call(self, node):
            if (
                ast.unparse(node.func) == "topk_keys.fill"
                and ast.unparse(node.args[0]) == "cutlass.Int32(-2147483648)"
            ):
                self.rank_padding += 1
                node.args[0] = ast.parse(
                    "cutlass.Int64(-9223372036854775808)", mode="eval"
                ).body
            return self.generic_visit(node)

    restored = SelectionOnly()
    original = SelectionOnly()
    old = original.visit(ast.parse(before))
    new = restored.visit(ast.parse(after))
    assert original.register_statements > 0 and restored.register_statements > 0
    assert restored.rank_buffers == restored.rank_padding == 1
    assert restored.rank_stores > 0
    assert ast.dump(old) == ast.dump(new)


@pytest.mark.parametrize(
    "n,k,lanes,vector",
    [
        (2, 1, 1, 1),
        (17, 4, 4, 8),
        (65, 8, 16, 4),
        (256, 8, 16, 8),
        (257, 64, 16, 4),
        (1025, 16, 32, 8),
        (8193, 32, 32, 4),
        (32768, 8, 32, 8),
    ],
)
def test_coarse_rank_topk_packed_residues_exact(n, k, lanes, vector):
    mask = (1 << (n - 1).bit_length()) - 1
    rng = random.Random(173)
    rows = []
    for row in range(32 // lanes):
        ranks = [
            0x08000000 + i * (mask + 1) + ((i * 137 + row) & mask) for i in range(n)
        ]
        rng.shuffle(ranks)
        rows.append(ranks)
    events, packed = _coarse_rank_tensor_model(
        rows, k, lanes, vector, recovery="packed"
    )
    _, direct = _coarse_rank_tensor_model(rows, k, lanes, vector, recovery="direct")
    assert packed == direct
    assert sum(event[0] == "recover" for event in events) == 1
    assert events[-1] == ("sort", True, max(1, k // lanes))


def test_coarse_rank_topk_packed_residue_sign_bit_is_masked():
    # Every residue is 0xff, so each packed Int32 has its sign bit set.
    rows = [[0x3F0000FF + i * 256 for i in range(256)] for _ in range(2)]
    _, direct = _coarse_rank_tensor_model(rows, 8, 16, 8, recovery="direct")
    _, packed = _coarse_rank_tensor_model(rows, 8, 16, 8, recovery="packed")
    assert packed == direct


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("seed", [31, 61, 97])
@pytest.mark.usefixtures("_legacy_topk_seed_family")
def test_coarse_rank_topk_key_recovery_population(seed):
    from test.cute_population_contracts import checked_initial_population

    from helion.autotuner.pattern_search import PatternSearch

    args = (torch.empty((5, 256), dtype=torch.float32), 8, True)
    results, states = [], []
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        for enabled in (False, True):
            kernel = helion.kernel(
                _row_topk.fn, backend="cute", static_shapes=True, autotune_effort="full"
            )
            if enabled:
                bound = _cpu_bind(kernel, args)
            else:
                with patch(
                    "helion._compiler.autotuner_heuristics.register_topk_key_recovery_coverage"
                ):
                    bound = _cpu_bind(kernel, args)
            spec = bound.config_spec
            assert "cute_topk_key_recovery" not in spec.default_config()
            assert all(
                "cute_topk_key_recovery" not in x for x in spec.compiler_seed_configs
            )
            with bound.env:
                random.seed(seed)
                search = PatternSearch(bound, args, initial_population=100)
                population = checked_initial_population(search)
                results.append(
                    [
                        dict(search.config_gen.canonicalize_flat(x)[1])
                        for x in population
                    ]
                )
                states.append(random.getstate())
            if enabled:
                group = next(
                    g
                    for g in spec.compiler_coverage_groups
                    if g.key == "cute_topk_key_recovery"
                )
                assert group.deferred and group.legacy == "direct"
                assert group.dependencies[0].key == "cute_topk_coarse_keys"
                assert results[-1][-1]["cute_topk_key_recovery"] == "packed"
                assert results[-1][-1]["cute_topk_coarse_keys"] is True
        assert results[1][:-1] == results[0]
        assert states[0] == states[1]


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("mode", ["direct", "packed"])
def test_coarse_rank_topk_key_recovery_config_and_codegen(mode):
    bound = _row_topk._bind_isolated((torch.empty(5, 256), 8, True))
    spec = bound.config_spec
    generation = spec.create_config_generation()
    cfg = spec.default_config()
    cfg.config.update(
        cute_topk_selection_layout="distributed",
        cute_topk_key_dtype="int64",
        cute_topk_coarse_keys=True,
        cute_topk_key_recovery=mode,
    )
    _, normalized = generation.strict_config_pair(cfg)
    assert normalized.config.get("cute_topk_key_recovery", "direct") == mode
    _, repeated = generation.strict_config_pair(
        helion.Config.from_dict(normalized.config)
    )
    assert repeated.config == normalized.config
    with patch(
        "helion._compiler.cute.selection_coarse.trace_coarse_rank_selection",
        wraps=trace_coarse_rank_selection,
    ) as trace:
        code = bound.to_code(cfg)
    assert trace.call_count == 1
    assert trace.call_args.args[-2:] == (mode, True)
    _assert_register_selection(code)
    assert "import coarse_rank_topk as" not in code


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize(
    "mode,coarse,dtype",
    [
        ("packed", False, torch.float32),
        ("bad", True, torch.float32),
        (True, True, torch.float32),
        ("packed", True, torch.float16),
    ],
)
def test_coarse_rank_topk_key_recovery_rejects_invalid(mode, coarse, dtype):
    bound = _row_topk._bind_isolated((torch.empty(5, 256, dtype=dtype), 8, True))
    cfg = bound.config_spec.default_config()
    cfg.config.update(
        cute_topk_selection_layout="distributed",
        cute_topk_key_dtype="int64",
        cute_topk_coarse_keys=coarse,
        cute_topk_key_recovery=mode,
    )
    with pytest.raises(exc.InvalidConfig):
        bound.config_spec.create_config_generation().strict_config_pair(cfg)


@pytest.mark.parametrize(
    "n,k,lanes,vector",
    [(17, 8, 4, 4), (65, 32, 32, 4), (256, 8, 16, 8), (1025, 16, 32, 8)],
)
def test_coarse_rank_topk_key_recovery_modes_exact(n, k, lanes, vector):
    rng = random.Random(812)
    rows = []
    for _ in range(32 // lanes):
        ranks = [0x3E800000 + i * 4096 + rng.randrange(128) for i in range(n)]
        rng.shuffle(ranks)
        rows.append(ranks)
    direct_events, direct = _coarse_rank_tensor_model(
        rows, k, lanes, vector, recovery="direct"
    )
    packed_events, packed = _coarse_rank_tensor_model(
        rows, k, lanes, vector, recovery="packed"
    )
    assert direct == packed
    assert [event[0] for event in packed_events] == [
        event[0] for event in direct_events
    ]
    assert any(event[0] == "recover" for event in packed_events)
    rows[0] = [0x3F000001] * n
    for recovery in ("direct", "packed"):
        events, _ = _coarse_rank_tensor_model(rows, k, lanes, vector, recovery=recovery)
        assert not any(event[0] == "recover" for event in events)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", ["direct", "packed"])
def test_coarse_rank_topk_key_recovery_native(mode):
    x = (
        torch.arange(5 * 256, device=DEVICE, dtype=torch.float32).reshape(5, 256) / 128
        + 1
    )
    x[1].fill_(1)
    x[2, 0] = float("nan")
    x[3, 0] = -0.0
    x[4, 0] = float("inf")
    original = x.clone()
    cfg = _composition_config(16, "distributed")
    cfg.update(
        cute_topk_key_dtype="int64",
        cute_topk_coarse_keys=True,
        cute_topk_key_recovery=mode,
    )
    _, (values, indices) = code_and_output(_row_topk, (x, 8, True), **cfg)
    _assert_topk_output(x, values, indices, 8, True)
    assert torch.equal(original.view(torch.int32), x.view(torch.int32))


@pytest.mark.parametrize("recovery", ["direct", "packed"])
@pytest.mark.parametrize(
    "n,k,lanes,vector",
    [(17, 8, 1, 1), (65, 32, 8, 4), (256, 8, 16, 8), (257, 64, 32, 4)],
)
def test_coarse_rank_topk_rank_input_exact(n, k, lanes, vector, recovery):
    rng = random.Random(117)
    rows = []
    for _ in range(32 // lanes):
        row = [0x3E800000 + i * 16384 + rng.randrange(128) for i in range(n)]
        rng.shuffle(row)
        rows.append(row)
    for fallback in (False, True):
        if fallback:
            # One ambiguous subrow must force the same full-warp fallback even
            # while the other subrows could take the narrow path.
            rows[0] = [0x3F000001] * n
        old_events, old = _coarse_rank_tensor_model(
            rows, k, lanes, vector, recovery=recovery
        )
        events, actual = _coarse_rank_tensor_model(
            rows, k, lanes, vector, recovery=recovery, rank_only=True
        )
        assert actual == old and events == old_events


@pytest.mark.parametrize("recovery", ["direct", "packed"])
@pytest.mark.parametrize("rank_mode", ["signed", "ordinal"])
@pytest.mark.parametrize("largest", [False, True])
def test_coarse_rank_topk_rank_input_special_values(recovery, rank_mode, largest):
    # These are the complete classes surrounding both ends of the admitted
    # FP32 rank domain: NaNs, infinities, normals, subnormals and signed zeros.
    words = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x007FFFFF,
        0x807FFFFF,
        0x00800000,
        0x80800000,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7F800001,
        0xFF800001,
        0x7FFFFFFF,
        0xFFFFFFFF,
        0x3F800000,
    ]
    ranks = []
    for word in words:
        bits = word if word < 1 << 31 else word - (1 << 32)
        sign = bits >> 31
        magnitude = bits & 0x7FFFFFFF
        rank = (
            (bits ^ (sign & 0x7FFFFFFF))
            if rank_mode == "ordinal"
            else (magnitude ^ sign) - sign
        )
        if magnitude > 0x7F800000:
            rank = 0x7FFFFFFF
        ranks.append(rank if largest else -rank)
    assert all(-(1 << 31) < rank < (1 << 31) for rank in ranks)
    rows = [ranks, list(reversed(ranks))]
    old_events, old = _coarse_rank_tensor_model(rows, 16, 16, 8, recovery=recovery)
    events, actual = _coarse_rank_tensor_model(
        rows, 16, 16, 8, recovery=recovery, rank_only=True
    )
    assert actual == old and events == old_events
    assert events[-1] == ("sort", True, 8)  # Complete resident fallback, with tails.


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("rank_mode", ["signed", "ordinal"])
@pytest.mark.parametrize("largest", [False, True])
def test_coarse_rank_topk_rank_input_actual_encoder(rank_mode, largest):
    bound = _row_topk._bind_isolated((torch.empty(5, 65), 8, largest))
    cfg = bound.config_spec.default_config()
    cfg.config.update(_composition_config(16, "distributed"))
    cfg.config.update(cute_topk_rank_mode=rank_mode, cute_topk_key_dtype="int64")
    encoders = []
    for enabled in (False, True):
        cfg.config["cute_topk_coarse_keys"] = enabled
        tree = ast.parse(bound.to_code(cfg))
        body = next(
            node.body
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and any(
                isinstance(stmt, ast.Assign)
                and ast.unparse(stmt.targets[0]) == "topk_magnitude"
                for stmt in node.body
            )
        )
        start = next(
            i
            for i, stmt in enumerate(body)
            if isinstance(stmt, ast.Assign)
            and ast.unparse(stmt.targets[0]) == "topk_magnitude"
        )
        end = next(
            i
            for i, stmt in enumerate(body)
            if isinstance(stmt, ast.Assign)
            and ast.unparse(stmt.targets[0]) == "topk_keys[topk_i]"
        )
        encoders.append(
            compile(
                ast.Module(body=body[start : end + 1], type_ignores=[]),
                "<actual rank encoder>",
                "exec",
            )
        )
    rng = random.Random(113)
    words = [rng.getrandbits(32) for _ in range(4096)]
    # Both signs at every exponent boundary, including all non-finite classes.
    words += [
        sign | (exponent << 23) | mantissa
        for sign in (0, 1 << 31)
        for exponent in range(256)
        for mantissa in (0, 1, (1 << 23) - 1)
    ]
    for index, word in enumerate(words):
        bits = word if word < 1 << 31 else word - (1 << 32)
        column = index % 65
        values = []
        for encoder in encoders:
            local = {
                "topk_bits": bits,
                "topk_col": column,
                "topk_i": 0,
                "topk_keys": {},
            }
            exec(encoder, {"cutlass": SimpleNamespace(Int32=int, Int64=int)}, local)
            values.append(local["topk_keys"][0])
        full_key, rank = values
        assert -(1 << 31) < rank < (1 << 31)
        assert full_key == (rank << 7) | (127 - column)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rank_mode", ["signed", "ordinal"])
@pytest.mark.parametrize("largest", [False, True])
def test_coarse_rank_topk_rank_input_native(rank_mode, largest):
    # Narrow input padding must stay distinct from either extreme canonical
    # rank. Include scalar loads, row tails, subrows and both recovery paths.
    storage = torch.randn((7, 195), device=DEVICE, dtype=torch.float32)
    x = storage[1:6, ::3]
    x[0, :7] = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, 1e-40, -1e-40],
        device=x.device,
    )
    x[1].fill_(1)
    original = storage.clone()
    cfg = _composition_config(16, "distributed")
    cfg.update(
        cute_topk_coarse_keys=True,
        cute_topk_key_dtype="int64",
        cute_topk_rank_mode=rank_mode,
        cute_topk_key_recovery="direct" if largest else "packed",
    )
    _, (values, indices) = code_and_output(_row_topk, (x, 8, largest), **cfg)
    _assert_topk_output(x, values, indices, 8, largest)
    assert torch.equal(original.view(torch.int32), storage.view(torch.int32))


@pytest.mark.parametrize("rank_only", [False, True])
@pytest.mark.parametrize("recovery", ["direct", "packed"])
def test_coarse_rank_topk_payload_sorted_models(rank_only, recovery):
    rng = random.Random(701)
    for lanes, vector, n, k in (
        (1, 1, 17, 8),
        (2, 4, 33, 16),
        (4, 4, 65, 32),
        (8, 8, 65, 16),
        (16, 8, 129, 8),
        (32, 4, 65, 64),
        (32, 1, 33, 1),
    ):
        rows = []
        for group in range(32 // lanes):
            row = [0x3E800000 + column * 8192 + group for column in range(n)]
            rng.shuffle(row)
            rows.append(row)
        events, _ = _coarse_rank_tensor_model(
            rows,
            k,
            lanes,
            vector,
            recovery=recovery,
            rank_only=rank_only,
            payload_only=True,
        )
        assert [event for event in events if event[0] == "sort"] == [
            (
                "sort",
                False,
                max(vector, 1 << (((n + lanes - 1) // lanes) - 1).bit_length()),
            )
        ]
        # A padded row makes its whole physical warp take the exact fallback.
        events, _ = _coarse_rank_tensor_model(
            rows,
            k,
            lanes,
            vector,
            recovery=recovery,
            rank_only=rank_only,
            payload_only=True,
            active_rows=len(rows) - 1,
        )
        assert events[-1][0:2] == ("sort", True)


@pytest.mark.parametrize("rank_only", [False, True])
@pytest.mark.parametrize("recovery", ["direct", "packed"])
def test_coarse_rank_topk_payload_every_collision(rank_only, recovery):
    n, k, lanes = 65, 32, 8
    for position in range(k - 1):
        rows = [[0x3F000000 - column * 8192 for column in range(n)] for _ in range(4)]
        # Reverse the true order within this bucket, while coarse payload order
        # puts the earlier column first. Keep the final cutoff bucket unique.
        rows[-1][position + 1] = rows[-1][position] + 1
        events, _ = _coarse_rank_tensor_model(
            rows,
            k,
            lanes,
            4,
            recovery=recovery,
            rank_only=rank_only,
            payload_only=True,
        )
        assert events[-1] == ("sort", True, 16)
    for invalid in (0, -1, 1, 0x7F800000, 0x7FFFFFFF):
        rows = [[0x3F000000 - column * 8192 for column in range(n)] for _ in range(4)]
        rows[-1][1] = invalid
        events, _ = _coarse_rank_tensor_model(
            rows,
            k,
            lanes,
            4,
            recovery=recovery,
            rank_only=rank_only,
            payload_only=True,
        )
        sorts = [event for event in events if event[0] == "sort"]
        if invalid >= 0x7F800000:
            assert sorts[-1] == ("sort", True, 16)
        else:
            assert sorts == [("sort", False, 16)]


@pytest.mark.parametrize("position", [7, 15, 23])
def test_coarse_rank_topk_payload_guard_counterexamples(position):
    rows = [[0x3F000000 - column * 8192 for column in range(65)] for _ in range(4)]
    rows[-1][position + 1] = rows[-1][position] + 1
    events, _ = _coarse_rank_tensor_model(rows, 32, 8, 4, payload_only=True)
    assert events[-1] == ("sort", True, 16)
    guard = selection_coarse.coarse_rank_guard

    def omit_payload_guard(keys, selected, k, bits, groups, payload_only=False):
        return guard(keys, selected, k, bits, groups, False)

    with (
        patch.object(
            selection_coarse, "coarse_rank_guard", side_effect=omit_payload_guard
        ),
        pytest.raises(AssertionError),
    ):
        _coarse_rank_tensor_model(rows, 32, 8, 4, payload_only=True)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.usefixtures("cpu_codegen")
@pytest.mark.parametrize("recovery", ["direct", "packed"])
def test_coarse_rank_topk_payload_consumer_proof(recovery):
    x = torch.empty((5, 65), dtype=torch.float32)
    cases = (
        (_row_topk, (x, 8, True), False),
        (_preprocessed_row_topk, (x, 8, True), False),
        (_indices_only_composed_row_topk, (x, 8), True),
        (_semantic_row_topk, (x, 8, "softmax"), False),
    )
    for kernel, args, unused in cases:
        bound = kernel._bind_isolated(args)
        for mode in ("decode", "gather"):
            cfg = bound.config_spec.default_config()
            cfg.config.update(_composition_config(16, "distributed"))
            cfg.config.update(
                cute_topk_coarse_keys=True,
                cute_topk_key_recovery=recovery,
                cute_topk_value_mode=mode,
            )
            with patch(
                "helion._compiler.cute.selection_coarse.trace_coarse_rank_selection",
                wraps=trace_coarse_rank_selection,
            ) as trace:
                code = bound.to_code(cfg)
            assert trace.call_count == 1
            assert trace.call_args.args[-2:] == (recovery, unused or mode == "gather")
            _assert_register_selection(code)
            assert "import coarse_rank_topk as" not in code
            assert code.count("@cute.kernel") == 1


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("recovery", ["direct", "packed"])
@pytest.mark.parametrize("k,lanes", [(8, 16), (13, 8), (32, 8)])
def test_coarse_rank_topk_payload_sorted_native(recovery, k, lanes):
    # The first four rows form a complete fast-path warp for both layouts.
    # Other positive rows exercise
    # an internal collision, a register boundary, exact ties, and padding.
    storage = torch.empty((11, 195), dtype=torch.float32, device=DEVICE)
    x = storage[1:10, ::3]
    words = 0x3F000000 - torch.arange(65, device=x.device, dtype=torch.int32) * 8192
    x.copy_(words.view(torch.float32))
    x[4, 2] = (words[1] + 1).view(torch.float32)
    x[5, 8] = (words[7] + 1).view(torch.float32)
    x[6].fill_(1)
    x[7, 0] = float("nan")
    original = x.view(torch.int32).clone()
    cfg = _composition_config(lanes, "distributed")
    cfg.update(
        cute_topk_coarse_keys=True,
        cute_topk_key_recovery=recovery,
        cute_topk_value_mode="gather",
        cute_topk_key_dtype="int64",
    )
    with patch(
        "helion._compiler.cute.selection_coarse.trace_coarse_rank_selection",
        wraps=trace_coarse_rank_selection,
    ) as trace:
        code, (values, indices) = code_and_output(_row_topk, (x, k, True), **cfg)
    assert trace.call_args_list
    assert all(call.args[-2:] == (recovery, True) for call in trace.call_args_list)
    _assert_register_selection(code)
    _assert_topk_output(x, values, indices, k)
    assert torch.equal(original, x.view(torch.int32))


@pytest.mark.parametrize("rank_only", [False, True])
@pytest.mark.parametrize("recovery", ["direct", "packed"])
def test_coarse_rank_topk_payload_index_bit_boundaries(rank_only, recovery):
    for bits in range(1, 24):
        n = 2 if bits == 1 else 4
        rows = [
            [0x00800000 + (n - i) * (1 << bits) for i in range(n)] for _ in range(8)
        ]
        events, _ = _coarse_rank_tensor_model(
            rows,
            2,
            4,
            1,
            recovery=recovery,
            rank_only=rank_only,
            payload_only=True,
            index_bits=bits,
        )
        assert [event[1] for event in events if event[0] == "sort"] == [False]
        rows[-1][1] = rows[-1][0] + 1
        events, _ = _coarse_rank_tensor_model(
            rows,
            2,
            4,
            1,
            recovery=recovery,
            rank_only=rank_only,
            payload_only=True,
            index_bits=bits,
        )
        assert events[-1][0:2] == ("sort", True)


@pytest.mark.parametrize("rank_only", [False, True])
@pytest.mark.parametrize("recovery", ["direct", "packed"])
@pytest.mark.parametrize("payload_only", [False, True])
def test_coarse_rank_topk_below_domain_count_proof(rank_only, recovery, payload_only):
    rng = random.Random(809)
    for n, k, lanes, vector in (
        (17, 8, 1, 1),
        (65, 32, 8, 4),
        (129, 8, 16, 8),
        (65, 64, 32, 4),
    ):
        for accepted in (k - 1, k, k + 1):
            lower = [-0x7FFFFFFF, -0x00800000, -1, 0, 1, 0x007FFFFF]
            rows = []
            for group in range(32 // lanes):
                row = [0x3E800000 + i * 8192 + group for i in range(accepted)]
                row += [lower[i % len(lower)] for i in range(n - accepted)]
                rng.shuffle(row)
                rows.append(row)
            events, _ = _coarse_rank_tensor_model(
                rows,
                k,
                lanes,
                vector,
                recovery=recovery,
                rank_only=rank_only,
                payload_only=payload_only,
            )
            sorts = [event for event in events if event[0] == "sort"]
            if accepted < k:
                assert sorts[-1][1] and sorts[-1][2] == max(
                    vector, 1 << (((n + lanes - 1) // lanes) - 1).bit_length()
                )
            elif payload_only:
                assert len(sorts) == 1
            else:
                assert sorts[-1] == ("sort", True, max(1, k // lanes))


@pytest.mark.parametrize("rank_only", [False, True])
@pytest.mark.parametrize("recovery", ["direct", "packed"])
@pytest.mark.parametrize("payload_only", [False, True])
def test_coarse_rank_topk_below_domain_sentinel_counterexample(
    rank_only, recovery, payload_only
):
    # One live negative bucket equals the -inf sentinel bit pattern, while
    # only seven values are accepted. Uniqueness must not permit refinement.
    rows = [[0x3E800000 + i * 8192 for i in range(17)] for _ in range(4)]
    rows[-1] = rows[-1][:7] + [-0x00800000] + [-1] * 9
    events, _ = _coarse_rank_tensor_model(
        rows, 8, 8, 4, recovery=recovery, rank_only=rank_only, payload_only=payload_only
    )
    assert events[-1] == ("sort", True, 4)
    assert not any(event[0] == "recover" for event in events)


@pytest.mark.parametrize("rank_only", [False, True])
@pytest.mark.parametrize("recovery", ["direct", "packed"])
@pytest.mark.parametrize("payload_only", [False, True])
def test_coarse_rank_topk_below_domain_full_warp_fallback(
    rank_only, recovery, payload_only
):
    base = [[0x3E800000 + i * 8192 for i in range(65)] for _ in range(4)]
    for cause in ("above", "short", "tie", "inactive"):
        rows = [row.copy() for row in base]
        if cause == "above":
            rows[-1][-1] = 0x7F800000
        elif cause == "short":
            rows[-1] = rows[-1][:7] + [-1] * 58
        elif cause == "tie":
            rows[-1] = [0x3E800000] * 65
        events, _ = _coarse_rank_tensor_model(
            rows,
            8,
            8,
            4,
            recovery=recovery,
            rank_only=rank_only,
            payload_only=payload_only,
            active_rows=3 if cause == "inactive" else 4,
        )
        assert events[-1] == ("sort", True, 16)


@pytest.mark.parametrize("rank_only", [False, True])
@pytest.mark.parametrize("rank_mode", ["signed", "ordinal"])
@pytest.mark.parametrize("largest", [False, True])
def test_coarse_rank_topk_below_domain_fp32_classes(rank_only, rank_mode, largest):
    # Exercise each exceptional class separately so an above-range value does
    # not mask the opportunity to discard an unrelated below-range value.
    words = [
        0,
        0x80000000,
        1,
        0x80000001,
        0x007FFFFF,
        0x807FFFFF,
        0x00800000,
        0x80800000,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7F800001,
        0xFF800001,
        0x7FFFFFFF,
        0xFFFFFFFF,
    ]

    def rank(word):
        bits = word if word < 1 << 31 else word - (1 << 32)
        sign = bits >> 31
        magnitude = bits & 0x7FFFFFFF
        ordered = (
            (bits ^ (sign & 0x7FFFFFFF))
            if rank_mode == "ordinal"
            else (magnitude ^ sign) - sign
        )
        if magnitude > 0x7F800000:
            ordered = 0x7FFFFFFF
        return ordered if largest else -ordered

    for word in words:
        base = [
            rank((0x3E800000 + i * 8192) | (0 if largest else 0x80000000))
            for i in range(65)
        ]
        rows = [base.copy() for _ in range(4)]
        rows[-1][0] = rank(word)
        for recovery in ("direct", "packed"):
            events, _ = _coarse_rank_tensor_model(
                rows, 8, 8, 4, recovery=recovery, rank_only=rank_only, payload_only=True
            )
            sorts = [event for event in events if event[0] == "sort"]
            if rank(word) >= 0x7F800000:
                assert sorts[-1] == ("sort", True, 16)
            else:
                assert len(sorts) == 1


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("recovery", ["direct", "packed"])
@pytest.mark.parametrize("largest", [False, True])
def test_coarse_rank_topk_below_domain_native(recovery, largest):
    storage = torch.zeros((7, 195), device=DEVICE, dtype=torch.float32)
    x = storage[:, ::3]
    x[:] = (torch.arange(65, device=x.device).float() - 37) / 128
    x[0, :] = -1
    x[0, :7] = torch.arange(7, device=x.device).float() + 1
    x[0, 7] = -torch.finfo(torch.float32).tiny
    x[1, :] = -float("inf")
    x[1, :8] = torch.arange(8, device=x.device).float() + 1
    x[2, 0:4] = torch.tensor([0.0, -0.0, 1e-40, -1e-40], device=x.device)
    x[3, :3] = torch.tensor(
        [float("nan"), float("inf"), -float("inf")], device=x.device
    )
    x[4, :] = 1
    if not largest:
        x.neg_()
    before = x.clone()
    for value_mode in ("gather", "decode"):
        code, (values, indices) = code_and_output(
            _row_topk,
            (x, 8, largest),
            block_sizes=[1],
            cute_topk_selection_layout="distributed",
            cute_topk_lanes_per_row=8,
            cute_topk_rows_per_block=4,
            cute_topk_vector_width=4,
            cute_topk_key_dtype="int64",
            cute_topk_rank_mode="ordinal",
            cute_topk_value_mode=value_mode,
            cute_topk_coarse_keys=True,
            cute_topk_key_recovery=recovery,
        )
        _assert_register_selection(code)
        assert "import coarse_rank_topk as" not in code
        _assert_topk_output(x, values, indices, 8, largest)
        assert torch.equal(x.view(torch.int32), before.view(torch.int32))


@pytest.mark.parametrize("rank_only", [False, True])
@pytest.mark.parametrize("payload_only", [False, True])
def test_coarse_rank_topk_below_domain_index_bit_boundaries(rank_only, payload_only):
    for bits in range(1, 24):
        rows = [[-0x00800000, 0x3E800000] for _ in range(16)]
        events, _ = _coarse_rank_tensor_model(
            rows,
            2,
            2,
            1,
            recovery="direct" if bits % 2 else "packed",
            rank_only=rank_only,
            payload_only=payload_only,
            index_bits=bits,
        )
        assert events[-1] == ("sort", True, 1)


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _fragment_leading_axes_topk(x, k: hl.constexpr, largest: hl.constexpr):
    values = torch.empty((x.size(0), x.size(1), k), device=x.device, dtype=x.dtype)
    indices = torch.empty((x.size(0), x.size(1), k), device=x.device, dtype=torch.int64)
    for row in hl.tile(x.size(0)):
        selected, index = torch.topk(x[row, :, :], k, dim=-1, largest=largest)
        values[row, :, :] = selected
        indices[row, :, :] = index
    return values, indices


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _fragment_composed_topk_gather(x, k: hl.constexpr):
    values = torch.empty((x.size(0), k), device=x.device, dtype=x.dtype)
    gathered = torch.empty_like(values)
    for row in hl.tile(x.size(0)):
        first, _ = torch.topk(x[row, :], k + 2, dim=-1)
        second, index = torch.topk(first, k, dim=-1, largest=False)
        values[row, :] = second
        gathered[row, :] = torch.gather(first, -1, index)
    return values, gathered


def _simulate_topk_fragment(source, inputs, outputs, blocks):
    import inspect
    import math

    from test.test_cute_computed_fragment import _simulate_independent_fragment

    text = inspect.getsource(_simulate_independent_fragment)
    text = text.replace("Int32=int,", "Int32=np.int32,").replace(
        "Int64=int,", "Int64=np.int64,"
    )
    text = text.replace(
        '"operator": operator,',
        '"operator": operator, "_bitcast": _bitcast, '
        '"_cute_gather_registers": _gather_registers,',
    )
    text = text.replace(
        "Float32=np.float32,", "Float32=np.float32, Float16=np.float16, Int16=np.int16,"
    )

    def bitcast(value, dtype):
        return np.asarray(value).view(dtype)[()]

    class Bitcasts(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "bitcast":
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="_bitcast", ctx=ast.Load()),
                        args=[node.func.value, *node.args],
                        keywords=[],
                    ),
                    node,
                )
            return node

    namespace = {
        "ast": ast,
        "math": math,
        "torch": torch,
        "SimpleNamespace": SimpleNamespace,
        "_bitcast": bitcast,
        "_gather_registers": gather_registers,
    }
    exec(compile(text, "<existing-fragment-model-with-bitcast>", "exec"), namespace)
    modeled = ast.unparse(
        ast.fix_missing_locations(Bitcasts().visit(ast.parse(source)))
    )
    namespace["_simulate_independent_fragment"](modeled, inputs, outputs, blocks)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int64])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("width,k", [(17, 3), (7, 7), (1, 1)])
def test_fragment_leading_axes_topk_generated_values(dtype, largest, width, k):
    rows, groups = 5, 3
    x = (
        (torch.arange(rows * groups * width).reshape(rows, groups, width) * 7) % 13 - 6
    ).to(dtype)
    if dtype in (torch.float16, torch.float32) and width >= 7:
        x[:, :, :7] = torch.tensor(
            [
                float("nan"),
                float("inf"),
                -float("inf"),
                0.0,
                -0.0,
                1.401298464324817e-45,
                -1.401298464324817e-45,
            ]
        )
    if dtype == torch.int64 and width >= 2:
        x[:, :, 0] = torch.iinfo(torch.int64).min
        x[:, :, 1] = torch.iinfo(torch.int64).max
    snapshot = x.clone()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_leading_axes_topk, (x, k, largest))
        cfg = bound.config_spec.default_config()
        source = bound.to_code(cfg)
    values = torch.empty((rows, groups, k), dtype=dtype)
    indices = torch.full((rows, groups, k), -99, dtype=torch.int64)
    _simulate_topk_fragment(
        source,
        {"x": x},
        {"values": values, "indices": indices},
        (rows + cfg.block_sizes[0] - 1) // cfg.block_sizes[0],
    )
    expected = torch.argsort(x, dim=-1, descending=largest, stable=True)[:, :, :k]
    torch.testing.assert_close(indices, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        values, x.gather(-1, expected), rtol=0, atol=0, equal_nan=True
    )
    torch.testing.assert_close(x, snapshot, rtol=0, atol=0, equal_nan=True)
    if dtype in (torch.float16, torch.float32):
        bits = torch.int16 if dtype == torch.float16 else torch.int32
        assert torch.equal(values.view(bits), x.gather(-1, expected).view(bits))
    assert "fragment_topk_previous_rank" in source


def test_fragment_composed_topk_gather_generated_values():
    x = ((torch.arange(5 * 17).reshape(5, 17) * 3) % 11).float()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_composed_topk_gather, (x, 3))
        cfg = bound.config_spec.default_config()
        source = bound.to_code(cfg)
    values, gathered = torch.empty((5, 3)), torch.empty((5, 3))
    _simulate_topk_fragment(
        source,
        {"x": x},
        {"values": values, "gathered": gathered},
        (5 + cfg.block_sizes[0] - 1) // cfg.block_sizes[0],
    )
    first = x.sort(dim=-1, descending=True, stable=True).values[:, :5]
    expected = first.sort(dim=-1, stable=True).values[:, :3]
    torch.testing.assert_close(values, expected, rtol=0, atol=0)
    torch.testing.assert_close(gathered, expected, rtol=0, atol=0)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int32, torch.int64]
)
def test_fragment_leading_axes_topk_native(dtype):
    x = ((torch.arange(5 * 3 * 17, device=DEVICE).reshape(5, 3, 17) * 7) % 13 - 6).to(
        dtype
    )
    before = x.clone()
    for largest in (False, True):
        values, indices = _fragment_leading_axes_topk(x, 3, largest)
        expected = torch.argsort(x, dim=-1, descending=largest, stable=True)[:, :, :3]
        torch.testing.assert_close(indices, expected, rtol=0, atol=0)
        torch.testing.assert_close(values, x.gather(-1, expected), rtol=0, atol=0)
        torch.testing.assert_close(x, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.int32, torch.int64]
)
def test_fragment_leading_axes_topk_strided_generated_values(dtype):
    storage = torch.arange(3 * 17 * 10).reshape(3, 17, 10).to(dtype)
    x = storage[:, :, ::2].permute(2, 0, 1)
    before = storage.clone()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_leading_axes_topk, (x, 3, True))
        cfg = bound.config_spec.default_config()
        source = bound.to_code(cfg)
    values = torch.empty((5, 3, 3), dtype=dtype)
    indices = torch.full((5, 3, 3), -99, dtype=torch.int64)
    _simulate_topk_fragment(
        source,
        {"x": x},
        {"values": values, "indices": indices},
        (5 + cfg.block_sizes[0] - 1) // cfg.block_sizes[0],
    )
    expected = torch.argsort(x, dim=-1, descending=True, stable=True)[:, :, :3]
    torch.testing.assert_close(indices, expected, rtol=0, atol=0)
    torch.testing.assert_close(values, x.gather(-1, expected), rtol=0, atol=0)
    torch.testing.assert_close(storage, before, rtol=0, atol=0)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fragment_composed_topk_gather_native():
    x = ((torch.arange(5 * 17, device=DEVICE).reshape(5, 17) * 3) % 11).float()
    before = x.clone()
    values, gathered = _fragment_composed_topk_gather(x, 3)
    first = x.sort(dim=-1, descending=True, stable=True).values[:, :5]
    expected = first.sort(dim=-1, stable=True).values[:, :3]
    torch.testing.assert_close(values, expected, rtol=0, atol=0)
    torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _fragment_topk_padded_producer(x, largest: hl.constexpr, mode: hl.constexpr):
    tail_values = torch.empty_like(x)
    tail_indices = torch.empty(x.shape, device=x.device, dtype=torch.int64)
    for tail_row in hl.tile(x.size(0)):
        tail_columns = hl.arange(x.size(2))
        if mode == 0:
            tail_condition = tail_columns % 3 == 0
        elif mode == 1:
            tail_condition = tail_columns >= 0
        else:
            tail_condition = tail_columns < 0
        tail_computed = torch.where(
            tail_condition[None, None, :], x[tail_row, :, :] + 1, 1.0
        )
        tail_selected, tail_index = torch.topk(
            tail_computed, x.size(2), dim=-1, largest=largest
        )
        tail_values[tail_row, :, :] = tail_selected
        tail_indices[tail_row, :, :] = tail_index
    return tail_values, tail_indices


def _topk_padded_producer_reference(x, largest, mode):
    columns = torch.arange(x.size(2), device=x.device)
    condition = (
        (columns % 3 == 0)
        if mode == 0
        else (columns >= 0 if mode == 1 else columns < 0)
    )
    computed = torch.where(condition[None, None, :], x + 1, 1.0)
    indices = torch.argsort(computed, dim=-1, descending=largest, stable=True)
    return computed.gather(-1, indices), indices


@pytest.mark.parametrize("width,rows", [(17, 5), (33, 1)])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_fragment_topk_computed_logical_tail(width, rows, largest, mode):
    x = (torch.arange(rows * 3 * width).reshape(rows, 3, width) % 11 + 2).float()
    if largest:
        x = -x
    before = x.clone()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_topk_padded_producer, (x, largest, mode))
        cfg = bound.config_spec.default_config()
        source = bound.to_code(cfg)
    values, indices = torch.empty_like(x), torch.full(x.shape, -99, dtype=torch.int64)
    _simulate_topk_fragment(
        source,
        {"x": x},
        {"tail_values": values, "tail_indices": indices},
        (rows + cfg.block_sizes[0] - 1) // cfg.block_sizes[0],
    )
    expected_values, expected_indices = _topk_padded_producer_reference(
        x, largest, mode
    )
    assert torch.equal(values, expected_values)
    assert torch.equal(indices, expected_indices)
    assert torch.equal(x, before)
    scans = [
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name)
        and n.target.id.startswith("fragment_topk_column")
    ]
    assert len(scans) == 1 and ast.literal_eval(scans[0].iter.args[0]) == width


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _fragment_topk_ambiguous_padded_reshape(x):
    result = torch.empty((x.size(0), 3), device=x.device, dtype=x.dtype)
    for row in hl.tile(x.size(0)):
        columns = hl.arange(x.size(2))
        transformed = torch.where(columns[None, None, :] >= 0, x[row, :, :] + 1, 1.0)
        flattened = transformed.reshape(transformed.size(0), -1)
        first, _ = torch.topk(flattened, 5, dim=-1)
        second, _ = torch.topk(first, 3, dim=-1)
        result[row, :] = second
    return result


def test_fragment_topk_padded_reshape_without_coordinate_proof_declines():
    x = torch.ones((5, 3, 17))
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_topk_ambiguous_padded_reshape, (x,))
        with pytest.raises(helion.exc.BackendUnsupported):
            bound.to_code(bound.config_spec.default_config())


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fragment_topk_computed_logical_tail_native():
    for largest in (False, True):
        for mode in (0, 1, 2):
            x = (
                torch.arange(5 * 3 * 17, device=DEVICE).reshape(5, 3, 17) % 11 + 2
            ).float()
            if largest:
                x = -x
            before = x.clone()
            values, indices = _fragment_topk_padded_producer(x, largest, mode)
            expected_values, expected_indices = _topk_padded_producer_reference(
                x, largest, mode
            )
            torch.testing.assert_close(values, expected_values, rtol=0, atol=0)
            torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
            torch.testing.assert_close(x, before, rtol=0, atol=0)


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _fragment_topk_tiled_selected_axis(x):
    tile_result = torch.empty((x.size(0), x.size(1), 3), device=x.device, dtype=x.dtype)
    for tile_row in hl.tile(x.size(0)):
        for tile_col in hl.tile(x.size(2), block_size=8):
            tile_picked, _ = torch.topk(x[tile_row, :, tile_col], 3, dim=-1)
            tile_result[tile_row, :, :] = tile_picked
    return tile_result


def test_fragment_topk_tiled_selected_axis_declines():
    from helion._compiler.cute.computed_fragment import computed_fragment_supported

    x = torch.ones((5, 3, 17))
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_topk_tiled_selected_axis, (x,))
        with bound.env, bound.host_function:
            assert not computed_fragment_supported(
                bound.env, bound.host_function.device_ir.graphs
            )
        source = bound.to_code(bound.config_spec.default_config())
    assert "fragment_topk_slot" not in source
    assert "for sort_k in range" in source


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _fragment_masked_computed_row_topk(x, bias, k: hl.constexpr, largest: hl.constexpr):
    values = torch.empty((x.size(0), k), dtype=x.dtype, device=x.device)
    indices = torch.empty((x.size(0), k), dtype=torch.int64, device=x.device)
    for row in hl.tile(x.size(0)):
        data = x[row, :]
        valid = torch.isfinite(data) & torch.isfinite(bias[:])
        if largest:
            ranked = torch.where(valid, data + bias[:], float("-inf"))
        else:
            ranked = torch.where(valid, data + bias[:], float("inf"))
        _, index = torch.topk(ranked, k, dim=-1, largest=largest)
        values[row, :] = torch.gather(torch.where(valid, data, 0.0), -1, index)
        indices[row, :] = index
    return values, indices


@pytest.mark.parametrize("width,k", [(17, 3), (65, 8)])
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize("tile", [1, 4])
def test_fragment_single_computed_topk_fallback_values(width, k, largest, tile):
    storage = ((torch.arange(5 * width * 2).reshape(5, -1) * 3) % 11).float()
    x = storage[:, ::2]
    x[0] = 2
    x[1] = float("nan")
    x[2, :4] = torch.tensor([float("inf"), -float("inf"), 0.0, -0.0])
    bias = torch.zeros(width)
    before = storage.clone()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_masked_computed_row_topk, (x, bias, k, largest))
        config = bound.config_spec.default_config()
        config.config["block_sizes"] = [tile]
        source = bound.to_code(config)
    assert "fragment_topk_previous_rank" in source
    values = torch.empty((5, k))
    indices = torch.full((5, k), -1, dtype=torch.int64)
    _simulate_topk_fragment(
        source,
        {"x": x, "bias": bias},
        {"values": values, "indices": indices},
        (5 + tile - 1) // tile,
    )
    valid = torch.isfinite(x) & torch.isfinite(bias)
    ranked = torch.where(valid, x + bias, -torch.inf if largest else torch.inf)
    expected = torch.argsort(ranked, dim=-1, descending=largest, stable=True)[:, :k]
    assert torch.equal(indices, expected)
    assert torch.equal(
        values.view(torch.int32),
        torch.where(valid, x, 0.0).gather(-1, expected).view(torch.int32),
    )
    assert torch.equal(storage.view(torch.int32), before.view(torch.int32))


@pytest.mark.parametrize("kind", ["direct", "pointwise", "prologue", "softmax"])
def test_fragment_single_topk_preserves_optimized_owner(kind):
    from helion._compiler.cute.computed_fragment import computed_fragment_supported

    x = torch.ones(5, 64)
    kernel, args = {
        "direct": (_row_topk, (x, 8, True)),
        "pointwise": (_pointwise_row_topk, (x, 8)),
        "prologue": (_preprocessed_row_topk, (x, 8, True)),
        "softmax": (_temperature_row_topk, (x, 8)),
    }[kind]
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(kernel, args)
        with bound.env, bound.host_function:
            ir = bound.host_function.device_ir
            assert (
                match_topk_root(
                    ir.graphs,
                    noncanonical_block_ids=ir.noncanonical_task_origin_block_ids,
                )
                is not None
            )
            assert not computed_fragment_supported(bound.env, ir.graphs)
        config = bound.config_spec.default_config()
        source = bound.to_code(config)
    _assert_register_selection(source)
    assert "fragment_topk_slot" not in source


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width,k", [(17, 3), (65, 8)])
def test_fragment_single_computed_topk_fallback_native(width, k):
    x = ((torch.arange(5 * width, device=DEVICE).reshape(5, width) * 3) % 11).float()
    x[0] = 2
    x[1] = float("nan")
    bias = torch.zeros(width, device=DEVICE)
    before = x.clone()
    for largest in (False, True):
        _, (values, indices) = code_and_output(
            _fragment_masked_computed_row_topk, (x, bias, k, largest), block_sizes=[4]
        )
        valid = torch.isfinite(x)
        ranked = torch.where(valid, x + bias, -torch.inf if largest else torch.inf)
        expected = torch.argsort(ranked, dim=-1, descending=largest, stable=True)[:, :k]
        assert torch.equal(indices, expected)
        assert torch.equal(values, torch.where(valid, x, 0.0).gather(-1, expected))
        assert torch.equal(x.view(torch.int32), before.view(torch.int32))


def _simulate_network_topk_fragment(source, inputs, outputs, blocks, threads=128):
    """Execute scalar instructions or ordinary compact-plan tensor operations.

    Encoding, masks, physical row ownership, cyclic slots, payload gathers and
    shared lifetime also execute the emitted AST. Every compare-exchange and
    shuffle participates in the model, with no replacement selection oracle.
    SDK and native tests separately verify compact-plan TensorSSA lowering.
    """
    import inspect
    import math

    from test.test_cute_computed_fragment import _simulate_fragment_warp_reduction

    text = inspect.getsource(_simulate_fragment_warp_reduction)
    text = text.replace(
        "        def visit_Call(self, node):\n",
        "        def visit_Call(self, node):\n"
        "            if ast.unparse(node.func) == '_cute_execute_register_plan':\n"
        "                return ast.Yield(ast.Tuple([ast.Tuple(node.args, ast.Load()), ast.Constant(0), ast.Constant('register_plan')], ast.Load()))\n",
    )
    text = text.replace(
        '            if kind == "index":\n',
        '            if kind == "register_plan":\n'
        "                peers, count = _execute_register_plan_events(events)\n"
        "                state.exchanges += count\n"
        "                events = advance(peers)\n"
        "                continue\n"
        '            if kind == "index":\n',
    )
    text = text.replace(
        "min=np.minimum, max=np.maximum",
        "min=lambda a, b, propagate_nan=False: (np.minimum if propagate_nan else np.fmin)(a, b), "
        "max=lambda a, b, propagate_nan=False: (np.maximum if propagate_nan else np.fmax)(a, b), "
        "tanh=np.tanh",
    )
    text = text.replace(
        '"operator": operator,',
        '"operator": operator, "_bitcast": _bitcast, '
        '"_cute_gather_registers": _gather_registers,',
    )
    text = text.replace(
        "Float16=np.float16,",
        "Float16=np.float16, Int16=np.int16, range_constexpr=range,",
    )
    text = text.replace(
        "BFloat16=lambda value: torch.tensor(float(value)).bfloat16().item(),",
        "BFloat16=lambda value: torch.tensor(float(value)).bfloat16(),",
    )
    text = text.replace(
        "make_layout=lambda shape: shape,",
        "make_layout=lambda shape: shape, make_rmem_tensor=lambda shape, dtype: np.empty(shape, dtype=dtype),",
    )
    text = text.replace(
        "assert all(event[1:] == (offset, kind) for event in events)",
        "assert all(event[2] == kind for event in events)",
    )
    text = text.replace(
        """                assert 0 <= offset < 32
                peers = [events[offset][0] for lane in range(32)]""",
        """                assert all(0 <= event[1] < 32 for event in events)
                peers = [events[event[1]][0] for event in events]""",
    )
    text = text.replace(
        "assert offset in (16, 8, 4, 2, 1)",
        "assert 0 <= offset < 32 and all(event[1] == offset for event in events)",
    )
    text = text.replace(
        'ast.literal_eval(kwargs["mask"])',
        'ast.literal_eval(kwargs.get("mask", ast.Constant(0xFFFFFFFF)))',
    )
    text = text.replace(
        'ast.literal_eval(kwargs["mask_and_clamp"])',
        'ast.literal_eval(kwargs.get("mask_and_clamp", ast.Constant(0 if kind == "up" else 31)))',
    )
    text = text.replace(
        'assert len(node.args) == (2 if kind == "index" else 1)',
        "assert 1 <= len(node.args) <= 2",
    )
    text = text.replace(
        'offset = node.args[1] if kind == "index" else kwargs["offset"]',
        'offset = kwargs.get("offset", node.args[1] if len(node.args) == 2 else None)\n            assert offset is not None',
    )
    # The scalar model's Pointer is contiguous-only. Keep actual storage strides
    # and offsets here, exactly as the independent fragment pointer model does.
    text = text.replace(
        """            assert 0 <= self.offset < self.tensor.numel()
            return self.tensor.reshape(-1)[self.offset].item()""",
        """            storage = self.tensor.as_strided((self.tensor.untyped_storage().nbytes() // self.tensor.element_size(),), (1,), storage_offset=0)
            offset = self.tensor.storage_offset() + int(self.offset)
            assert 0 <= offset < storage.numel()
            return storage[offset].item()""",
    )
    namespace = {
        "ast": ast,
        "_execute_register_plan_events": execute_register_plan_events,
        "_gather_registers": gather_registers,
        "math": math,
        "torch": torch,
        "SimpleNamespace": SimpleNamespace,
        "Counter": __import__("collections").Counter,
        "itertools": __import__("itertools"),
    }

    def bitcast(value, dtype):
        if isinstance(value, torch.Tensor):
            assert value.dtype == torch.bfloat16 and dtype == np.int16
            return np.int16(value.view(torch.int16).item())
        return np.asarray(value).view(dtype)[()]

    class Bitcasts(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "bitcast":
                return ast.copy_location(
                    ast.Call(
                        ast.Name("_bitcast", ast.Load()),
                        [node.func.value, *node.args],
                        [],
                    ),
                    node,
                )
            return node

    namespace["_bitcast"] = bitcast
    exec(compile(text, "<existing-fragment-model-network>", "exec"), namespace)
    modeled = ast.unparse(
        ast.fix_missing_locations(Bitcasts().visit(ast.parse(source)))
    )
    return namespace["_simulate_fragment_warp_reduction"](
        modeled, inputs, outputs, blocks, threads, allow_lane_stores=True
    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int32]
)
@pytest.mark.parametrize("largest", [False, True])
@pytest.mark.parametrize(
    "width,k,threads", [(1, 1, 32), (17, 3, 128), (65, 33, 256), (129, 129, 64)]
)
def test_fragment_network_topk_values(
    dtype, largest, width, k, threads, block_rows=None
):
    x = ((torch.arange(5 * 3 * width).reshape(5, 3, width) * 7) % 13 - 6).to(dtype)
    if dtype.is_floating_point and width >= 7:
        x[:, :, :7] = torch.tensor(
            [
                float("nan"),
                float("inf"),
                -float("inf"),
                0.0,
                -0.0,
                1.401298464324817e-45,
                -1.401298464324817e-45,
            ]
        ).to(dtype)
    if dtype == torch.int32:
        x[:, :, 0] = torch.iinfo(dtype).min
        if width > 1:
            x[:, :, 1] = torch.iinfo(dtype).max
    original = x.clone()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        b = _cpu_bind(_fragment_leading_axes_topk, (x, k, largest))
        c = b.config_spec.default_config()
        c.config.update(cute_fragment_topk_network=True, cute_fragment_threads=threads)
        if block_rows is not None:
            c.config["block_sizes"] = [block_rows]
        source = b.to_code(c)
    actual = torch.empty((5, 3, k), dtype=dtype)
    indices = torch.full((5, 3, k), -99, dtype=torch.int64)
    state = _simulate_network_topk_fragment(
        source,
        {"x": x},
        {"values": actual, "indices": indices},
        (5 + c.block_sizes[0] - 1) // c.block_sizes[0],
        threads,
    )
    expected = torch.argsort(x, dim=-1, descending=largest, stable=True)[:, :, :k]
    assert torch.equal(indices, expected)
    bits = torch.int16 if dtype in (torch.float16, torch.bfloat16) else torch.int32
    assert torch.equal(actual.view(bits), x.view(bits).gather(-1, expected))
    assert torch.equal(x.view(bits), original.view(bits))
    assert state.exchanges > 0 and "fragment_topk_previous_rank" not in source


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("largest", [False, True])
def test_fragment_network_topk_masked_producer(mode, largest):
    x = ((torch.arange(5 * 3 * 17).reshape(5, 3, 17) * 7) % 13 - 20).float()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        b = _cpu_bind(_fragment_topk_padded_producer, (x, largest, mode))
        c = b.config_spec.default_config()
        before = b.to_code(c)
        c.config["cute_fragment_topk_network"] = True
        after = b.to_code(c)
    expected = (torch.empty_like(x), torch.empty(x.shape, dtype=torch.int64))
    actual = (torch.empty_like(expected[0]), torch.empty_like(expected[1]))
    blocks = (5 + c.block_sizes[0] - 1) // c.block_sizes[0]
    _simulate_topk_fragment(
        before,
        {"x": x},
        {"tail_values": expected[0], "tail_indices": expected[1]},
        blocks,
    )
    _simulate_network_topk_fragment(
        after, {"x": x}, {"tail_values": actual[0], "tail_indices": actual[1]}, blocks
    )
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(actual[0].view(torch.int32), expected[0].view(torch.int32))


def test_fragment_network_topk_composed_gather():
    x = ((torch.arange(5 * 17).reshape(5, 17) * 3) % 11).float()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        b = _cpu_bind(_fragment_composed_topk_gather, (x, 3))
        c = b.config_spec.default_config()
        c.config["cute_fragment_topk_network"] = True
        source = b.to_code(c)
    actual = (torch.empty((5, 3)), torch.empty((5, 3)))
    _simulate_network_topk_fragment(
        source,
        {"x": x},
        {"values": actual[0], "gathered": actual[1]},
        (5 + c.block_sizes[0] - 1) // c.block_sizes[0],
    )
    first = x.sort(dim=-1, descending=True, stable=True).values[:, :5]
    expected = first.sort(dim=-1, stable=True).values[:, :3]
    assert torch.equal(actual[0], expected) and torch.equal(actual[1], expected)


@pytest.mark.parametrize("layout", ["strided", "transposed", "offset"])
def test_fragment_network_topk_strides(layout):
    storage = (torch.arange(7 * 3 * 35).reshape(7, 3, 35) % 29).float()
    x = (
        storage[:5, :, :34:2]
        if layout == "strided"
        else storage[:5, :, :17].transpose(0, 1)
        if layout == "transposed"
        else storage[1:6, :, 1:18]
    )
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        b = _cpu_bind(_fragment_leading_axes_topk, (x, 3, True))
        c = b.config_spec.default_config()
        c.config["cute_fragment_topk_network"] = True
        source = b.to_code(c)
    actual = torch.empty((*x.shape[:-1], 3))
    indices = torch.full(actual.shape, -99, dtype=torch.int64)
    _simulate_network_topk_fragment(
        source,
        {"x": x},
        {"values": actual, "indices": indices},
        (x.shape[0] + c.block_sizes[0] - 1) // c.block_sizes[0],
    )
    expected = torch.argsort(x, dim=-1, descending=True, stable=True)[:, :, :3]
    assert torch.equal(indices, expected) and torch.equal(
        actual, x.gather(-1, expected)
    )


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize(
    "dtype,width,k,largest",
    [
        (torch.float16, 17, 3, True),
        (torch.bfloat16, 65, 33, False),
        (torch.float32, 129, 129, True),
        (torch.int32, 33, 7, False),
    ],
)
def test_fragment_network_topk_sdk(tmp_path, dtype, width, k, largest, block_rows=None):
    import importlib.util

    import cutlass
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import func
    import cutlass.cute as cute

    cuda_initialized = torch.cuda.is_initialized()
    x = torch.empty((5, 3, width), dtype=dtype)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_leading_axes_topk, (x, k, largest))
        config = bound.config_spec.default_config()
        config.config["cute_fragment_topk_network"] = True
        if block_rows is not None:
            config.config["block_sizes"] = [block_rows]
        source = bound.to_code(config)
    tree = ast.parse(source)
    kernel = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
    )
    kernel.name = "staged_network_topk"
    kernel.decorator_list = [ast.parse("cute.jit", mode="eval").body]
    module = ast.fix_missing_locations(
        ast.Module(
            body=[
                *[
                    n
                    for n in tree.body
                    if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign))
                ],
                kernel,
            ],
            type_ignores=[],
        )
    )
    path = tmp_path / "sdk_payload.py"
    path.write_text(ast.unparse(module) + "\n")
    spec = importlib.util.spec_from_file_location("network_sdk", path)
    sdk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sdk)
    ctype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
        torch.int32: cutlass.Int32,
    }[dtype]
    with ir.Context(), ir.Location.unknown():
        emitted = ir.Module.create()
        with ir.InsertionPoint(emitted.body):
            entry = func.FuncOp("entry", ([], []))
            block = entry.add_entry_block()
            with ir.InsertionPoint(block):
                tensors = [
                    cute.make_tensor(
                        cute.make_ptr(t, 0, cute.AddressSpace.gmem, assumed_align=16),
                        cute.make_layout(shape, stride=strides),
                    )
                    for t, shape, strides in [
                        (ctype, (5, 3, width), (3 * width, width, 1)),
                        (ctype, (5, 3, k), (3 * k, k, 1)),
                        (cutlass.Int64, (5, 3, k), (3 * k, k, 1)),
                    ]
                ]
                sdk.staged_network_topk(*tensors)
                func.ReturnOp([])
        assert emitted.operation.verify()
        text = str(emitted)
        (tmp_path / "sdk.mlir").write_text(text)
        assert "nvvm.shfl.sync" in text and "scf.for" in text
    assert torch.cuda.is_initialized() == cuda_initialized


@pytest.fixture
def _network_prefix_without_later_coverage():
    from contextlib import ExitStack

    # This unit test checks the network group's exact one-row extension.
    # Full combined populations are checked independently without suppression.
    with ExitStack() as stack:
        for registration in (
            "register_fragment_local_atomic_registers_coverage",
            "register_fragment_register_snapshots_coverage",
            "register_fragment_published_scalars_coverage",
            "register_fragment_skip_zero_atomics_coverage",
            "register_fragment_atomic_consumer_fusion_coverage",
            "register_integer_loop_reduction_coverage",
            "register_fragment_integer_atomic_epochs_coverage",
            "register_fragment_packet_loads_coverage",
            "register_fragment_register_producers_coverage",
        ):
            stack.enter_context(
                patch("helion._compiler.autotuner_heuristics." + registration)
            )
        yield


@pytest.mark.parametrize("seed", [31, 61, 97])
@pytest.mark.usefixtures("_network_prefix_without_later_coverage")
def test_fragment_network_topk_population_prefix(seed):
    from test.cute_population_contracts import checked_initial_population

    from helion.autotuner.pattern_search import PatternSearch

    args = (torch.empty((5, 3, 65)), 7, True)
    results = []
    states = []
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        for enabled in (False, True):
            kernel = helion.kernel(
                _fragment_leading_axes_topk.fn,
                backend="cute",
                static_shapes=True,
                autotune_effort="full",
            )
            if not enabled:
                with patch(
                    "helion._compiler.autotuner_heuristics.register_fragment_topk_network_coverage"
                ):
                    bound = _cpu_bind(kernel, args)
            else:
                bound = _cpu_bind(kernel, args)
            spec = bound.config_spec
            assert "cute_fragment_topk_network" not in spec.default_config()
            with bound.env:
                random.seed(seed)
                search = PatternSearch(bound, args, initial_population=100)
                population = checked_initial_population(search)
                configs = [
                    dict(search.config_gen.canonicalize_flat(x)[1]) for x in population
                ]
                results.append(configs)
                states.append(random.getstate())
            if enabled:
                assert configs[-1]["cute_fragment_topk_network"] is True
                group = next(
                    g
                    for g in spec.compiler_coverage_groups
                    if g.key == "cute_fragment_topk_network"
                )
                assert group.deferred and group.legacy is False
        assert results[1][:-1] == results[0] and states[0] == states[1]


@pytest.mark.parametrize("kind", ["ordinary", "dtype", "width"])
def test_fragment_network_topk_unsupported_request(kind):
    from helion.exc import InvalidConfig

    kernel = _row_topk if kind == "ordinary" else _fragment_leading_axes_topk
    x = (
        torch.empty((5, 65))
        if kind == "ordinary"
        else torch.empty(
            (5, 3, 2049 if kind == "width" else 17),
            dtype=torch.int64 if kind == "dtype" else torch.float32,
        )
    )
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(kernel, (x, 3, True))
        c = bound.config_spec.default_config()
        before = bound.to_code(c)
        c.config["cute_fragment_topk_network"] = False
        assert bound.to_code(c) == before
        c.config["cute_fragment_topk_network"] = True
        with pytest.raises(InvalidConfig, match="complete fragment top-k"):
            bound.to_code(c)


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _fragment_mixed_network_topk(x, y):
    output = torch.empty((x.size(0), x.size(1), 3), dtype=x.dtype, device=x.device)
    control = torch.empty((y.size(0), y.size(1), 3), dtype=y.dtype, device=y.device)
    for row in hl.tile(x.size(0)):
        a, _ = torch.topk(x[row, :, :], 3, dim=-1)
        b, _ = torch.topk(y[row, :, :], 3, dim=-1, largest=False)
        output[row, :, :] = a
        control[row, :, :] = b
    return output, control


@pytest.mark.parametrize("wide", [False, True])
def test_fragment_network_topk_mixed_fallback(wide):
    x = ((torch.arange(2 * 3 * 17).reshape(2, 3, 17) * 5) % 19).float()
    width = 1025 if wide else 17
    y = ((torch.arange(2 * 3 * width).reshape(2, 3, width) * 11) % 31).to(
        torch.float32 if wide else torch.int64
    )
    if not wide:
        y[:, :, 0] = torch.iinfo(torch.int64).min
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        b = _cpu_bind(_fragment_mixed_network_topk, (x, y))
        c = b.config_spec.default_config()
        c.config["cute_fragment_topk_network"] = True
        source = b.to_code(c)
    actual = (torch.empty((2, 3, 3)), torch.empty((2, 3, 3), dtype=y.dtype))
    _simulate_network_topk_fragment(
        source,
        {"x": x, "y": y},
        {"output": actual[0], "control": actual[1]},
        (2 + c.block_sizes[0] - 1) // c.block_sizes[0],
    )
    assert torch.equal(
        actual[0], x.sort(dim=-1, descending=True, stable=True).values[:, :, :3]
    )
    assert torch.equal(actual[1], y.sort(dim=-1, stable=True).values[:, :, :3])
    assert "fragment_topk_previous_rank" in source and _has_register_selection(source)


@helion.kernel(backend="cute", autotune_effort="none", static_shapes=True)
def _fragment_network_topk_reshaped(x):
    output = torch.empty((x.size(0), 2), device=x.device, dtype=x.dtype)
    output_indices = torch.empty((x.size(0), 2), device=x.device, dtype=torch.int64)
    for row in hl.tile(x.size(0)):
        values = x[row, :]
        grouped = values.reshape(values.size(0) * 4, x.size(1) // 4)
        selected, _ = torch.topk(grouped, 2, dim=-1)
        scores = selected.sum(dim=-1).reshape(values.size(0), 4)
        values, indices = torch.topk(scores, 2, dim=-1)
        output[row, :] = values
        output_indices[row, :] = indices
    return output, output_indices


def test_fragment_network_topk_resident_reshape():
    x = ((torch.arange(5 * 4 * 16).reshape(5, 64) * 7) % 31).float()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        b = _cpu_bind(_fragment_network_topk_reshaped, (x,))
        c = b.config_spec.default_config()
        before = b.to_code(c)
        c.config["cute_fragment_topk_network"] = True
        after = b.to_code(c)
    actual = (torch.empty((5, 2)), torch.empty((5, 2), dtype=torch.int64))
    expected = tuple(torch.empty_like(x) for x in actual)
    blocks = (5 + c.block_sizes[0] - 1) // c.block_sizes[0]
    _simulate_topk_fragment(
        before, {"x": x}, {"output": expected[0], "output_indices": expected[1]}, blocks
    )
    _simulate_network_topk_fragment(
        after, {"x": x}, {"output": actual[0], "output_indices": actual[1]}, blocks
    )
    assert all(itertools.starmap(torch.equal, zip(actual, expected, strict=True)))


@pytest.mark.parametrize("mutation", ["padding", "slots", "barrier", "comparisons"])
def test_fragment_network_topk_generated_mutations(mutation):
    x = -torch.arange(1, 5 * 3 * 17 + 1).reshape(5, 3, 17).float()
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        b = _cpu_bind(_fragment_leading_axes_topk, (x, 17, True))
        c = b.config_spec.default_config()
        c.config["cute_fragment_topk_network"] = True
        source = b.to_code(c)
    tree = ast.parse(source)
    if mutation == "padding":
        loop = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id.startswith("fragment_topk_col")
        )
        loop.value = ast.BinOp(loop.value, ast.Mod(), ast.Constant(17))
    elif mutation == "slots":
        slot = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id.startswith("fragment_topk_slot")
        )
        slot.value = ast.Constant(0)
    elif mutation == "comparisons":
        comparisons = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "cute.math.max"
        ]
        assert comparisons
        for comparison in comparisons:
            assert isinstance(comparison.func, ast.Attribute)
            comparison.func.attr = "min"
    else:
        function = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name.startswith("_helion_")
        )
        index = next(
            i
            for i, n in enumerate(function.body)
            if isinstance(n, ast.Expr) and ast.unparse(n) == "cute.arch.sync_threads()"
        )
        function.body.pop(index)
    actual = torch.empty((5, 3, 17))
    indices = torch.full(actual.shape, -99, dtype=torch.int64)
    with pytest.raises((AssertionError, IndexError)):
        _simulate_network_topk_fragment(
            ast.unparse(tree),
            {"x": x},
            {"values": actual, "indices": indices},
            (5 + c.block_sizes[0] - 1) // c.block_sizes[0],
        )
        expected = torch.argsort(x, dim=-1, descending=True, stable=True)
        assert torch.equal(indices, expected)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.int32]
)
def test_fragment_network_topk_native(dtype):
    x = ((torch.arange(5 * 3 * 65, device=DEVICE).reshape(5, 3, 65) * 7) % 31 - 15).to(
        dtype
    )
    original = x.clone()
    for largest in (False, True):
        _, (values, indices) = code_and_output(
            _fragment_leading_axes_topk,
            (x, 33, largest),
            block_sizes=[4],
            cute_fragment_topk_network=True,
        )
        expected = torch.argsort(x, dim=-1, descending=largest, stable=True)[:, :, :33]
        assert torch.equal(indices, expected) and torch.equal(
            values, x.gather(-1, expected)
        )
    assert torch.equal(x, original)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fragment_network_topk_resident_reshape_native():
    x = ((torch.arange(5 * 4 * 16, device=DEVICE).reshape(5, 64) * 7) % 31).float()
    original = x.clone()
    _, (values, indices) = code_and_output(
        _fragment_network_topk_reshaped,
        (x,),
        block_sizes=[4],
        cute_fragment_topk_network=True,
    )
    first = (
        x.reshape(5, 4, 16)
        .sort(dim=-1, descending=True, stable=True)
        .values[:, :, :2]
        .sum(dim=-1)
    )
    expected = torch.argsort(first, dim=-1, descending=True, stable=True)[:, :2]
    assert torch.equal(indices, expected) and torch.equal(
        values, first.gather(-1, expected)
    )
    assert torch.equal(x, original)


def test_fragment_network_topk_capacity_boundary():
    test_fragment_network_topk_values(
        torch.float32, False, 1024, 513, 512, block_rows=1
    )


@pytest.mark.parametrize("value", [1, "network", None])
def test_fragment_network_topk_strict_boolean(value):
    from helion.exc import InvalidConfig

    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        b = _cpu_bind(_fragment_leading_axes_topk, (torch.empty(5, 3, 17), 3, True))
        c = b.config_spec.default_config()
        c.config["cute_fragment_topk_network"] = value
        with pytest.raises(InvalidConfig, match="complete fragment top-k"):
            b.to_code(c)


@skipUnlessCuteAvailable("requires CuTe DSL")
@pytest.mark.parametrize("initialized", [False, True])
def test_fragment_network_sdk_preserves_existing_cuda_state(tmp_path, initialized):
    with (
        patch("torch.cuda.is_initialized", return_value=initialized),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU SDK only")),
    ):
        test_fragment_network_topk_sdk(tmp_path, torch.int32, 33, 7, False)


@pytest.mark.parametrize("seed", [31, 61, 97])
def test_fragment_network_full_population_keeps_later_coverage(seed):
    from test.cute_population_contracts import checked_initial_population

    from helion.autotuner.pattern_search import PatternSearch

    args = (torch.empty((5, 3, 65)), 7, True)
    with _mock_cuda_unavailable(), _target(), _forbid_native_compile():
        bound = _cpu_bind(_fragment_leading_axes_topk, args)
        keys = [group.key for group in bound.config_spec.compiler_coverage_groups]
        assert keys.index("cute_fragment_topk_network") < keys.index(
            "cute_fragment_packet_loads"
        )
        with bound.env:
            random.seed(seed)
            search = PatternSearch(bound, args, initial_population=100)
            # This helper verifies the whole base prefix, RNG state and every
            # declared appended witness, with no registration suppression.
            population = checked_initial_population(search)
            configs = [search.config_gen.canonicalize_flat(x)[1] for x in population]
        assert any(
            config.get("cute_fragment_topk_network", False) for config in configs
        )
        assert any(
            config.get("cute_fragment_packet_loads", False) for config in configs
        )
