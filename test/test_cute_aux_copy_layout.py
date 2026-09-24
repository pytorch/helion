from __future__ import annotations

import ast
from collections import Counter

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")

from test.test_cute_aux_read_fence import _rank_two_inputs
from test.test_cute_aux_read_fence import _rank_two_multi_aux
from test.test_cute_batched_aux_tma import _config
from test.test_cute_batched_aux_tma import _cpu_codegen

from helion._compiler.cute.aux_copy_layout import select_aux_copy_layout
from helion._compiler.cute.mma_support import get_cute_mma_support
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


@pytest.mark.parametrize("dtype_bits", [8, 16, 32, 64])
@pytest.mark.parametrize(
    "shape",
    [(128, 32), (128, 64), (64, 128), (32, 16), (16, 8), (128, 4), (8, 4), (96, 48)],
)
def test_aux_copy_layout_exact_ownership(
    shape: tuple[int, int], dtype_bits: int
) -> None:
    m, n = shape
    tm, tn, values = select_aux_copy_layout(m, n, dtype_bits)
    assert tm * tn == 32
    assert tm <= m and tn * values <= n
    assert values * dtype_bits <= 128
    owners: Counter[tuple[int, int]] = Counter()
    for lane in range(32):
        points = [
            (lane // tn + row * tm, lane % tn * values + col * tn * values + value)
            for row in range(m // tm)
            for col in range(n // (tn * values))
            for value in range(values)
        ]
        assert points and len(points) == m * n // 32
        owners.update(points)
    assert owners == Counter((row, col) for row in range(m) for col in range(n))


@pytest.mark.parametrize("dtype_bits", [8, 16, 32, 64])
def test_aux_copy_layout_preserves_old_safe_geometry(dtype_bits: int) -> None:
    assert select_aux_copy_layout(128, 128, dtype_bits) == (4, 8, 128 // dtype_bits)


def test_aux_copy_layout_narrow_vectors_and_mixed_dtypes() -> None:
    assert select_aux_copy_layout(128, 32, 16) == (8, 4, 8)
    assert select_aux_copy_layout(128, 32, 32) == (4, 8, 4)
    assert select_aux_copy_layout(128, 4, 16) == (32, 1, 4)
    assert select_aux_copy_layout(8, 4, 16) == (8, 4, 1)


@pytest.mark.parametrize(
    "args",
    [
        (0, 32, 16),
        (128, 0, 16),
        (-1, 32, 16),
        (128, 32, 0),
        (128, 32, 24),
        (128, 32, 256),
        (1, 1, 16),
        (3, 7, 16),
    ],
)
def test_aux_copy_layout_invalid_geometry(args: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        select_aux_copy_layout(*args)


def test_aux_copy_layout_last_row_last_subtile_bounds() -> None:
    # Actual failing multiwave coordinates: block103 + grid152 = work255.
    tm, tn, values = select_aux_copy_layout(128, 32, 16)
    last_vector_ends = []
    for lane in range(32):
        row = 384 + lane // tn + (128 // tm - 1) * tm
        column = 8064 + 3 * 32 + lane % tn * values
        last_vector_ends.append((row * 8192 + column + values) * 2)
    assert max(last_vector_ends) == 512 * 8192 * 2
    assert all(end <= 512 * 8192 * 2 for end in last_vector_ends)


@pytest.mark.parametrize("mixed", [False, True])
def test_aux_copy_codegen_uses_actual_subtile_per_descriptor(mixed: bool) -> None:
    args = list(_rank_two_inputs(2))
    if mixed:
        args[-1] = torch.empty_like(args[-1], dtype=torch.float32)
    with _cpu_codegen():
        code = _rank_two_multi_aux._bind_isolated(tuple(args)).to_code(
            _config(block_sizes=[128, 128, 64], tcgen05_aux_load_mode="simt")
        )
    calls = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_cute_aux_copy_layout"
    ]
    assert len(calls) == 2
    for node in calls:
        assert ast.unparse(node.args[0]) == "cute.size(tcgen05_epi_tile[0])"
        assert ast.unparse(node.args[1]) == "cute.size(tcgen05_epi_tile[1])"
    assert sorted(ast.literal_eval(node.args[2]) for node in calls) == (
        [16, 32] if mixed else [16, 16]
    )
    assert "CooperativeGroup(cutlass.pipeline.Agent.Thread, cutlass.Int32(32))" in code


def test_aux_copy_layout_not_imported_in_tma_codegen() -> None:
    with _cpu_codegen():
        code = _rank_two_multi_aux._bind_isolated(_rank_two_inputs(2)).to_code(
            _config(block_sizes=[128, 128, 64], tcgen05_aux_load_mode="tma")
        )
    assert "_cute_aux_copy_layout" not in code
    assert "tcgen05_aux_copy_values" not in code
    assert "PipelineTmaAsync.create" in code


@pytest.mark.parametrize("m", [512, 513])
def test_aux_copy_persistent_fringe_host_admission(m: int) -> None:
    with _cpu_codegen():
        code = _rank_two_multi_aux._bind_isolated(
            _rank_two_inputs(2, m=m, n=8192)
        ).to_code(_config(block_sizes=[128, 128, 64], tcgen05_aux_load_mode="simt"))
    host = next(
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name == "_rank_two_multi_aux"
    )
    guards = [
        (index, node)
        for index, node in enumerate(host.body)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "_helion_tcgen05_persistent_total_tiles > 1"
    ]
    if m == 512:
        assert not guards
        return
    assert len(guards) == 1
    guard_index, guard = guards[0]
    launches = [
        index
        for index, node in enumerate(host.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_launcher"
    ]
    assert len(launches) == 1 and guard_index < launches[0]
    assert len(guard.body) == 1 and isinstance(guard.body[0], ast.Raise)
    error = guard.body[0].exc
    assert isinstance(error, ast.Call) and ast.unparse(error.func) == "RuntimeError"
    assert "validated single-root static full tiles" in ast.unparse(error.args[0])
    total = host.body[guard_index - 1]
    assert isinstance(total, ast.Assign)
    assert ast.unparse(total.targets[0]) == "_helion_tcgen05_persistent_total_tiles"
    # Evaluate only the generated integer shape expression, never the launcher.
    assert (
        eval(
            compile(ast.Expression(total.value), "<host-tile-count>", "eval"),
            {"__builtins__": {}, "_BLOCK_SIZE_0": 128, "_BLOCK_SIZE_1": 128},
        )
        == 320
    )


@skipUnlessBackends(["cute"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("fringe", [False, True])
def test_aux_copy_last_column_multiwave_runtime(
    dtype: torch.dtype, mixed: bool, fringe: bool
) -> None:
    if not get_cute_mma_support().tcgen05_f16bf16:
        pytest.skip("requires TCgen05 F16/BF16 support")
    m, n = (513 if fringe else 512), 8192
    lhs = torch.zeros((m, 128), dtype=dtype, device=DEVICE)
    rhs = torch.ones((128, n), dtype=dtype, device=DEVICE)
    bias = (torch.arange(m * n, device=DEVICE) % 61).reshape(m, n).to(dtype)
    scale = torch.ones((m, n), dtype=torch.float32 if mixed else dtype, device=DEVICE)
    args = (lhs, rhs, bias, scale)
    before = tuple(value.clone() for value in args)
    bound = _rank_two_multi_aux._bind_isolated(args)
    bound.set_config(_config(block_sizes=[128, 128, 64], tcgen05_aux_load_mode="simt"))
    if fringe:
        # This persistent M-tail path is not runtime-admitted. Retain its
        # fail-closed contract without claiming numerical/graph fringe support.
        with pytest.raises(
            RuntimeError,
            match=r"validated single-root static full tiles:.*total_tiles=320,",
        ):
            bound(*args)
        for actual, original in zip(args, before, strict=True):
            assert torch.equal(actual, original)
        return
    first = bound(*args)
    second = bound(*args)
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first, second)
    # The dot is exactly zero. Distinct column values catch duplicated SIMD
    # owners without relying on floating-point tolerance or allocation padding.
    assert torch.equal(first, bias)
    assert torch.equal(first[-1, -32:], bias[-1, -32:])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = bound(*args)
    graph.replay()
    assert torch.equal(captured, first)
    for _ in range(3):
        captured.fill_(float("nan"))
        graph.replay()
        assert torch.equal(captured, first)
    for actual, original in zip(args, before, strict=True):
        assert torch.equal(actual, original)
