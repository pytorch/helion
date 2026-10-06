"""Integer boundary provenance without weakening address or mask proofs."""

from __future__ import annotations

import ast
import itertools
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from test._cute_binding import _cpu_bind
from test.test_cute_collective_tf32 import _config as _jagged_config
from test.test_cute_collective_tf32 import _jagged_inputs
from test.test_cute_collective_tf32 import _kernel as _jagged_kernel
from test.test_cute_contiguous_copy import _evaluate_copy
from test.test_cute_contiguous_copy import _plan

import helion
from helion._compiler.cute.collective_matmul import _uniform_reuse_names
from helion._compiler.cute.contiguous_copy import recipe_has_integer_result
from helion._compiler.cute.scalar_recipe import build_recipe
from helion._testing import skipUnlessBackends
import helion.language as hl


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


@pytest.mark.parametrize(
    "source,integer_tensors,expected",
    [
        ("value = (metadata.iterator + cutlass.Int32(row)).load()", {"metadata"}, True),
        (
            (
                "p = metadata.iterator + cutlass.Int64(row)\n"
                "base = p.load() if valid else cutlass.Int64(0)\n"
                "value = (base + cutlass.Int32(row)) * 24"
            ),
            {"metadata"},
            True,
        ),
        ("value = cutlass.Int32(unknown * 0.125)", set(), True),
        ("value = cutlass.Int64(unknown)", set(), True),
        ("value = (metadata.iterator + cutlass.Int32(row)).load()", set(), False),
        ("value = unknown * 24", {"metadata"}, False),
        (
            "value = cutlass.Float32((metadata.iterator + cutlass.Int32(row)).load())",
            {"metadata"},
            False,
        ),
        (
            (
                "base = (metadata.iterator + cutlass.Int32(row)).load()\n"
                "value = base + cutlass.Float32(0.125)"
            ),
            {"metadata"},
            False,
        ),
        (
            "base = (metadata.iterator + cutlass.Int32(row)).load()\nvalue = base / 8",
            {"metadata"},
            False,
        ),
        (
            (
                "value = (metadata.iterator + cutlass.Int32(row)).load() "
                "if valid else cutlass.Float32(0)"
            ),
            {"metadata"},
            False,
        ),
        (
            (
                "base = (metadata.iterator + cutlass.Int32(row)).load()\n"
                "base = cutlass.Float32(0.125)\n"
                "value = base * 24"
            ),
            {"metadata"},
            False,
        ),
    ],
)
def test_scalar_boundary_type_uses_reaching_definitions(
    source: str, integer_tensors: set[str], expected: bool
) -> None:
    recipe = build_recipe(
        _expr("value"),
        ast.parse(source).body,
        {"metadata", "row", "valid", "unknown"},
    )
    assert recipe is not None
    counter = itertools.count()
    statements, value = recipe.emit({}, lambda hint: f"r{next(counter)}")
    assert recipe_has_integer_result(statements, value, integer_tensors) == expected


def test_boundary_type_proof_remains_bounded() -> None:
    source = ["q0 = (metadata.iterator + cutlass.Int32(0)).load()"]
    source.extend(f"q{i} = q{i - 1} + q{i - 1}" for i in range(1, 80))
    statements = cast("list[ast.Assign]", ast.parse("\n".join(source)).body)
    assert not recipe_has_integer_result(statements, _expr("q79"), {"metadata"})


def test_only_immutable_uniform_integer_boundaries_supply_type_facts() -> None:
    prefix = ast.parse(
        "uniform = (metadata.iterator + cutlass.Int32(0)).load()\n"
        "floating = (data.iterator + cutlass.Int32(0)).load()\n"
        "varying = (metadata.iterator + cutlass.Int32(cute.arch.thread_idx()[0])).load()\n"
        "modified = uniform\n"
        "unknown_value = unknown * 24\n"
    ).body
    counter = itertools.count()
    df = SimpleNamespace(new_var=lambda hint: f"r{next(counter)}")
    names, integers = _uniform_reuse_names(
        prefix,
        ast.parse("modified = 1").body,
        {"metadata", "data", "unknown"},
        df,
        integer_tensor_names=frozenset({"metadata"}),
    )
    assert names == {"uniform", "floating", "unknown_value"}
    assert integers == {"uniform"}


@pytest.mark.parametrize("integer_cast", ["Int32", "Int64"])
@pytest.mark.parametrize("origin", [0, 1, 3])
@pytest.mark.parametrize("limit", [0, 10, 24])
def test_integer_boundary_vector_copy_preserves_values_and_tail_reads(
    integer_cast: str, origin: int, limit: int
) -> None:
    source = (
        f"value = (A.iterator + cutlass.{integer_cast}(origin * 24 + k)).load() "
        "if k < limit else cutlass.Float32(0)"
    )
    assert _plan(source, dtype="cutlass.Float32", aligned_names={"k": 4}) is None
    plan = _plan(
        source,
        dtype="cutlass.Float32",
        aligned_names={"k": 4, "origin": 1},
    )
    assert plan is not None
    output, loads, index_loads, copies = _evaluate_copy(
        plan, origin=origin, k=8, limit=limit
    )
    expected_indices = [origin * 24 + k for k in range(8, 12) if k < limit]
    assert output == [origin * 24 + k if k < limit else 0 for k in range(8, 12)]
    assert loads == ([] if limit >= 12 else expected_indices)
    assert index_loads == []
    assert copies == int(limit <= 8 or limit >= 12)


def test_integer_type_fact_does_not_evaluate_an_inactive_division() -> None:
    plan = _plan(
        "value = (A.iterator + cutlass.Int32(24 // (count - m)) * 8 + origin + k).load() "
        "if m < count else cutlass.Float16(0)",
        aligned_names={"k": 8, "origin": 8, "count": 1, "m": 1},
    )
    assert plan is not None
    output, loads, index_loads, copies = _evaluate_copy(
        plan, origin=0, k=0, limit=24, count=1
    )
    assert output == [0] * 8
    assert loads == [] and index_loads == [] and copies == 0


@helion.kernel(
    backend="cute", static_shapes=True, autotune_effort="none", dot_precision="tf32"
)
def _offset_rows_matmul(
    a: torch.Tensor, b: torch.Tensor, row_base: torch.Tensor
) -> torch.Tensor:
    m = a.size(0)
    groups, k, n = b.shape
    flattened = a.view(-1)
    out = torch.empty((m, n), device=a.device, dtype=torch.float32).view(-1)
    for group in hl.tile(groups):
        base = row_base[group]
        count = (row_base[group.index + 1] - base).to(torch.int32)
        for tile_m in hl.jagged_tile(count):
            rows = base[:, None] + tile_m.index[None, :]
            for tile_n in hl.tile(n):
                acc = hl.zeros([group, tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    index = rows[:, :, None] * k + tile_k.index[None, None, :]
                    left = hl.load(flattened, [index])
                    acc = acc + torch.matmul(left, b[group, tile_k, tile_n])
                target = rows[:, :, None] * n + tile_n.index[None, None, :]
                hl.store(out, [target], acc)
    return out.view(m, n)


@pytest.mark.parametrize(
    "offset_dtype", [torch.int32, torch.int64, torch.float32, torch.float64]
)
@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@skipUnlessBackends(["cute"])
def test_declared_metadata_dtype_controls_collective_vector_copy(
    offset_dtype: torch.dtype, compute: str
) -> None:
    inputs = (
        torch.empty((197, 64), dtype=torch.float32),
        torch.empty((3, 64, 128), dtype=torch.float32),
        torch.empty((4,), dtype=offset_dtype),
    )
    bound = _cpu_bind(_offset_rows_matmul, inputs)
    source = bound.to_code(
        helion.Config(
            block_sizes=[1, 64, 32, 32],
            num_threads=[1, 4, 32, 1],
            cute_vector_widths=[1, 1, 1, 1],
            cute_lane_layouts=["blocked"] * 4,
            cute_collective_mma=True,
            cute_collective_compute=compute,
            cute_collective_copy="scalar",
            cute_collective_recipe="vector_unrolled",
        )
    )
    integer_offset = offset_dtype in (torch.int32, torch.int64)
    assert ("cute.make_ptr(flattened.element_type" in source) == integer_offset
    if integer_offset:
        assert "cute.gemm(" in source
        assert ("tcgen05.CtaGroup.ONE" in source) == (compute == "tcgen05")
        assert "cute.make_ptr(b.element_type" in source
        assert "cute.arch.cvt_f32_tf32(" in source


@pytest.mark.parametrize(
    "shape", [(16, 319, 32, 32), (64, 4674, 64, 64), (256, 30856, 128, 128)]
)
@pytest.mark.parametrize("compute", ["tcgen05", "warp"])
@skipUnlessBackends(["cute"])
def test_original_jagged_async_copy_retains_alignment_guard(
    shape: tuple[int, int, int, int], compute: str
) -> None:
    batch, rows, k, n = shape
    arguments = _jagged_inputs(batch=batch, rows=rows, k=k, n=n)
    config = _jagged_config(
        128 if compute == "tcgen05" else 32,
        32 if n == 32 else 64,
        64,
        compute=compute,
        copy="async_cached",
    )
    source = _cpu_bind(_jagged_kernel(), arguments).to_code(config)
    assert "collective_a_raw" in source and "collective_b_raw" in source
    assert "cute.arch.cvt_f32_tf32(collective_a_raw[" in source
    assert "cute.arch.cvt_f32_tf32(collective_b_raw[" in source
    offsets, a, b, bias = arguments
    unaligned_a = torch.empty(a.numel() + 1, dtype=a.dtype)[1:].view_as(a)
    unaligned = _cpu_bind(_jagged_kernel(), (offsets, unaligned_a, b, bias)).to_code(
        config
    )
    assert "collective_a_raw" not in unaligned
    assert "collective_b_raw" in unaligned
