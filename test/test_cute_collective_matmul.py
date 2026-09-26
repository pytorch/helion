from __future__ import annotations

import ast
import itertools
from types import SimpleNamespace

from examples.moe_matmul_ogs import moe_matmul_ogs
import pytest
import torch

from test._cute_binding import _cpu_bind

import helion
from helion._compiler.cute.collective_matmul import _uniform_reuse_names
from helion._testing import skipUnlessBackends
import helion.language as hl

CUDA_DEVICE = "cuda"


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _computed_gathered_matmul(
    a: torch.Tensor, b: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    groups, k, n = b.shape
    m = rows.size(1)
    out = torch.empty((groups, m, n), device=a.device, dtype=a.dtype)
    for group in hl.grid(groups):
        for tile_m, tile_n in hl.tile([m, n]):
            source_rows = rows[group, tile_m]
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                left = a[source_rows, tile_k] * 0.5
                right = b[group, tile_k, tile_n]
                acc = torch.addmm(acc, left, right)
            out[group, tile_m, tile_n] = torch.relu(acc).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _aliased_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    groups, m, k = a.shape
    n = b.size(1)
    for group in hl.grid(groups):
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, a[group, tile_m, tile_k], b[tile_k, tile_n])
            a[group, tile_m, tile_n] = acc.to(a.dtype)
    return a


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _grid_computed_matmul(
    a: torch.Tensor, b: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    k, n = b.shape
    m = rows.numel()
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    for tile_m, tile_n in hl.tile([m, n]):
        source_rows = rows[tile_m]
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            left = a[source_rows, tile_k] * 0.5
            right = b[tile_k, tile_n]
            acc = torch.addmm(acc, left, right)
        out[tile_m, tile_n] = torch.relu(acc).to(out.dtype)
    return out


def _config(
    bm: int = 64, bn: int = 32, bk: int = 32, *, copy: str = "scalar"
) -> helion.Config:
    return helion.Config(
        block_sizes=[bm, bn, bk],
        num_threads=[128 // bn, bn, 1],
        cute_vector_widths=[1, 1, 1, 1],
        cute_collective_mma=True,
        cute_collective_copy=copy,
    )


def _grid_config(*, n_first: bool, copy: str = "scalar") -> helion.Config:
    config = dict(_config(bk=64, copy=copy))
    config["cute_vector_widths"] = [1, 1, 1]
    config["loop_orders"] = [[1, 0] if n_first else [0, 1]]
    return helion.Config.from_dict(config)


@pytest.mark.parametrize("n_first", [False, True])
@skipUnlessBackends(["cute"])
def test_grid_collective_uses_physical_thread_order(n_first: bool) -> None:
    inputs = (
        torch.empty((193, 78), dtype=torch.float16),
        torch.empty((78, 70), dtype=torch.float16),
        torch.empty(130, dtype=torch.int32),
    )
    bound = _cpu_bind(_grid_computed_matmul, inputs)
    source = bound.to_code(_grid_config(n_first=n_first))
    assert "cute.gemm(" in source
    assert f"block=({32 if n_first else 4}, {4 if n_first else 32}, 1)" in source
    assert (
        "collective_tid = cutlass.Int32(cute.arch.thread_idx()[0]) + "
        f"cutlass.Int32(cute.arch.thread_idx()[1]) * {32 if n_first else 4}"
    ) in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n_first", [False, True])
@pytest.mark.parametrize("copy", ["scalar", "async_cached"])
@skipUnlessBackends(["cute"])
def test_grid_collective_correctness(
    dtype: torch.dtype, n_first: bool, copy: str
) -> None:
    torch.manual_seed(58)
    a = torch.randn((193, 80), device=CUDA_DEVICE, dtype=dtype)[:, :78]
    b = torch.randn((78, 80), device=CUDA_DEVICE, dtype=dtype)[:, :70]
    rows = torch.randperm(193, device=CUDA_DEVICE)[:130].to(torch.int32)
    bound = _grid_computed_matmul._bind_isolated((a, b, rows))
    config = _grid_config(n_first=n_first, copy=copy)
    assert "cute.gemm(" in bound.to_code(config)
    bound.set_config(config)
    expected = torch.relu((a[rows.long()] * 0.5).float() @ b.float()).to(dtype)
    for _ in range(3):
        torch.testing.assert_close(bound(a, b, rows), expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bk", [16, 32, 64, 128])
@skipUnlessBackends(["cute"])
def test_computed_gathered_source_uses_collective(dtype: torch.dtype, bk: int) -> None:
    bound = _cpu_bind(
        _computed_gathered_matmul,
        (
            torch.empty((193, 78), dtype=dtype),
            torch.empty((3, 78, 70), dtype=dtype),
            torch.empty((3, 130), dtype=torch.int32),
        ),
    )
    code = bound.to_code(_config(bk=bk))
    ast.parse(code)
    assert "cute.gemm(" in code
    assert "_helion_pending_collective_mma" not in code
    assert "block=(32, 4, 1)" in code
    # Global stores remain in the original row/column epilogue. The scalar
    # recipe contains the indirect row load, scaling, and K-tail predicate.
    assert "rows.iterator" in code
    assert "for tile_offset_1 in range" in code
    assert "collective" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tile", [(32, 32, 16), (64, 32, 64), (128, 64, 32)])
@skipUnlessBackends(["cute"])
def test_computed_gathered_matmul_correctness(
    dtype: torch.dtype, tile: tuple[int, int, int]
) -> None:
    torch.manual_seed(41)
    a = torch.randn((193, 78), device=CUDA_DEVICE, dtype=dtype)
    b = torch.randn((3, 78, 70), device=CUDA_DEVICE, dtype=dtype)
    rows = torch.stack(
        [torch.randperm(193, device=CUDA_DEVICE)[:130] for _ in range(3)]
    ).to(torch.int32)
    bound = _computed_gathered_matmul._bind_isolated((a, b, rows))
    bound.set_config(_config(*tile))
    result = bound(a, b, rows)
    expected = torch.relu(torch.bmm((a[rows.long()] * 0.5).float(), b.float())).to(
        dtype
    )
    torch.testing.assert_close(result, expected, rtol=0.02, atol=0.02)


@skipUnlessBackends(["cute"])
def test_input_mutation_keeps_original_serial_schedule() -> None:
    bound = _cpu_bind(
        _aliased_matmul,
        (
            torch.empty((3, 64, 64), dtype=torch.float16),
            torch.empty((64, 64), dtype=torch.float16),
        ),
    )
    code = bound.to_code(_config())
    assert "cute.gemm(" not in code
    assert "_helion_pending_collective_mma" not in code


@skipUnlessBackends(["cute"])
def test_uniform_metadata_is_read_once_before_row_loop() -> None:
    kernel = helion.kernel(
        moe_matmul_ogs.fn, backend="cute", static_shapes=False, autotune_effort="none"
    )
    bound = _cpu_bind(
        kernel,
        (
            torch.empty((193, 78), dtype=torch.float16),
            torch.empty((3, 78, 70), dtype=torch.float16),
            torch.empty(3, dtype=torch.int32),
            torch.empty(4, dtype=torch.int32),
            torch.empty(193, dtype=torch.int32),
            130,
        ),
    )
    source = bound.to_code(_config(bk=64))
    assert source.count("expert_token_counts.iterator") == 1
    assert source.count("expert_token_offsets.iterator") == 1
    assert "existing_values =" not in source


def test_uniform_reuse_excludes_thread_values_and_modified_names() -> None:
    prefix = ast.parse(
        "uniform = (data.iterator + cutlass.Int32(cute.arch.block_idx()[0])).load()\n"
        "varying = (data.iterator + cutlass.Int32(cute.arch.thread_idx()[0])).load()\n"
        "copied_varying = varying + 1\n"
        "modified = 1\n"
    ).body
    loop_body = ast.parse("modified = 2\n").body
    counter = itertools.count()
    df = SimpleNamespace(new_var=lambda hint: f"fresh_{next(counter)}")
    assert _uniform_reuse_names(prefix, loop_body, {"data"}, df) == ({"uniform"}, set())


def test_uniform_reuse_does_not_cross_intervening_store() -> None:
    prefix = ast.parse(
        "uniform = (data.iterator + cutlass.Int32(0)).load()\n"
        "(other.iterator + cutlass.Int32(0)).store(1)\n"
    ).body
    counter = itertools.count()
    df = SimpleNamespace(new_var=lambda hint: f"fresh_{next(counter)}")
    assert _uniform_reuse_names(prefix, [], {"data", "other"}, df) == (set(), set())


def _padded_inputs(device: str, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    a = torch.randn((193, 80), device=device, dtype=dtype)[:, :78]
    b = torch.randn((3, 78, 80), device=device, dtype=dtype)[:, :, :70]
    counts = torch.tensor([0, 96, 97], device=device, dtype=torch.int32)
    offsets = torch.tensor([0, 0, 96, 193], device=device, dtype=torch.int32)
    rows = torch.randperm(193, device=device).to(torch.int32)
    return a, b, counts, offsets, rows


@skipUnlessBackends(["cute"])
def test_async_copies_have_cache_safe_alignment_and_stride_facts() -> None:
    kernel = helion.kernel(
        moe_matmul_ogs.fn, backend="cute", static_shapes=False, autotune_effort="none"
    )
    inputs = (*_padded_inputs("cpu", torch.float16), 97)
    bound = _cpu_bind(kernel, inputs)
    assert "input_tensor_metadata" in bound.env.compiler_fact_specialization_facts
    source = bound.to_code(_config(bk=64, copy="async"))
    assert source.count("cute.arch.cp_async_shared_global(") == 2
    assert "cute.arch.cp_async_wait_group(0)" in source
    assert "copy_tail_lane" in source


@skipUnlessBackends(["cute"])
def test_cached_copy_keeps_gather_loads_outside_reduction_loop() -> None:
    kernel = helion.kernel(
        moe_matmul_ogs.fn, backend="cute", static_shapes=False, autotune_effort="none"
    )
    inputs = (*_padded_inputs("cpu", torch.float16), 97)
    bound = _cpu_bind(kernel, inputs)
    source = bound.to_code(_config(bk=64, copy="async_cached"))
    assert source.count("cute.arch.cp_async_shared_global(") == 2
    assert "cute.make_rmem_tensor((4,), cutlass.Int32)" in source
    reduction = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "tile_offset_3"
    )
    assert "sorted_to_orig_token_idx.iterator" not in ast.unparse(reduction)
    assert source.index("prefill_slot") < source.index("for tile_offset_3")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("unaligned", [False, True])
@pytest.mark.parametrize("copy", ["async", "async_cached"])
@skipUnlessBackends(["cute"])
def test_async_gathered_matmul_preserves_tails_and_empty_groups(
    dtype: torch.dtype, unaligned: bool, copy: str
) -> None:
    torch.manual_seed(73)
    a, b, counts, offsets, rows = _padded_inputs("cuda", dtype)
    if unaligned:
        storage = torch.empty(193 * 80 + 1, device=CUDA_DEVICE, dtype=dtype)
        shifted = torch.as_strided(storage, a.shape, a.stride(), storage_offset=1)
        shifted.copy_(a)
        a = shifted
    kernel = helion.kernel(
        moe_matmul_ogs.fn, backend="cute", static_shapes=False, autotune_effort="none"
    )
    inputs = (a, b, counts, offsets, rows, 97)
    bound = kernel._bind_isolated(inputs)
    config = _config(bk=64, copy=copy)
    source = bound.to_code(config)
    assert source.count("cute.arch.cp_async_shared_global(") == (1 if unaligned else 2)
    bound.set_config(config)
    expected = torch.empty((193, 70), device=CUDA_DEVICE, dtype=dtype)
    expected[rows[:96].long()] = (a[rows[:96].long()].float() @ b[1].float()).to(dtype)
    expected[rows[96:].long()] = (a[rows[96:].long()].float() @ b[2].float()).to(dtype)
    for _ in range(3):
        torch.testing.assert_close(bound(*inputs), expected, rtol=0.02, atol=0.02)
