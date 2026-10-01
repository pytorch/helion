"""Typed address proofs and bounded analysis for vectorized operand loads."""

from __future__ import annotations

import ast
import itertools
import struct
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test.test_cute_collective_vector_recipe import _RECIPE
from test.test_cute_collective_vector_recipe import _run_recipe
from test.test_cute_fuse_mm_accumulation import _cpu_target

import helion
from helion._compiler.cute.collective_vector_recipe import emit_vector_recipe
from helion._compiler.cute.contiguous_copy import CopyTensorFacts
from helion._compiler.cute.contiguous_copy import _Analysis
from helion._compiler.cute.contiguous_copy import plan_contiguous_copy
from helion._testing import skipUnlessBackends
from helion.autotuner.benchmarking import _make_cudagraph_replay
import helion.language as hl

CUDA_DEVICE = "cuda"

if TYPE_CHECKING:
    from pathlib import Path


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _load_plan(source: str, dtype: str, width: int):
    return plan_contiguous_copy(
        cast("list[ast.Assign]", ast.parse(source).body),
        _expr("loaded"),
        coordinate="k",
        tensors={"A": CopyTensorFacts(dtype, (24, 1), 16)},
        aligned_names={"k": width},
    )


def _check_recipe(source: str, dtype: str, width: int) -> int:
    arguments = {
        "width": width,
        "m": 0,
        "k": 8,
        "origin": 0,
        "lower": 0,
        "upper": 24,
        "alignment": 16,
        "dtype": dtype,
    }
    expected, reads, scalar_copies = _run_recipe(source, optimized=False, **arguments)
    actual, actual_reads, copies = _run_recipe(source, optimized=True, **arguments)
    assert [struct.pack("d", value) for value in actual] == [
        struct.pack("d", value) for value in expected
    ]
    assert actual_reads == reads and scalar_copies == 0
    return copies


@pytest.mark.parametrize(
    "dtype,width",
    [("cutlass.Float16", 8), ("cutlass.BFloat16", 8), ("cutlass.Float32", 4)],
)
@pytest.mark.parametrize(
    "offset",
    [
        "cutlass.Float32(0.125) * 8",
        "0.125 * 8",
        "(scales.iterator + 2).load() * 8",
        "cutlass.Float32(-0.125) * 8",
    ],
)
@pytest.mark.parametrize("predicate", ["True", "k < 11", "False"])
def test_precast_float_arithmetic_preserves_scalar_addresses(
    dtype: str, width: int, offset: str, predicate: str
) -> None:
    source = f"""
extra = cutlass.Int32({offset})
loaded = (A.iterator + k + extra).load() if {predicate} else {dtype}(0.0)
value = {dtype}(loaded * 0.5)
"""
    assert _load_plan(source, dtype, width) is None
    assert _check_recipe(source, dtype, width) == 0


@pytest.mark.parametrize(
    "offset",
    [
        "cutlass.Int32(cutlass.Float32(0.125)) * 8",
        "cutlass.Int32((indices.iterator + m).load()) * 8",
    ],
)
def test_integer_arithmetic_after_cast_retains_vector_copy(offset: str) -> None:
    source = f"""
extra = {offset}
loaded = (A.iterator + k + extra).load()
value = cutlass.Float16(loaded * 0.5)
"""
    assert _load_plan(source, "cutlass.Float16", 8) is not None
    assert _check_recipe(source, "cutlass.Float16", 8) == 1


def test_unknown_gather_dtype_keeps_scalar_values_and_reads() -> None:
    # The fixture's explicit casts mirror generated pointer arithmetic. Without
    # a cast or source dtype fact, an ordinary .load() does not prove an integer.
    source = _RECIPE.replace("cutlass.Int32(row)", "row")
    assert _check_recipe(source, "cutlass.Float16", 8) == 0


def _doubling_recipe(depth: int) -> str:
    statements = ["q0 = (k + cutlass.Int32(m) * 8) - k"]
    statements.extend(f"q{i} = q{i - 1} + q{i - 1}" for i in range(1, depth + 1))
    statements.extend(
        [
            f"loaded = (A.iterator + k + q{depth}).load()",
            "value = cutlass.Float16(loaded * 0.5)",
        ]
    )
    return "\n".join(statements)


def test_small_affine_dag_preserves_values_and_vector_copy() -> None:
    source = _doubling_recipe(4)
    assert _load_plan(source, "cutlass.Float16", 8) is not None
    assert _check_recipe(source, "cutlass.Float16", 8) == 1


@pytest.mark.parametrize("depth", [8, 12, 40, 300])
def test_large_affine_dag_declines_without_expansion(depth: int) -> None:
    source = _doubling_recipe(depth)
    assert _load_plan(source, "cutlass.Float16", 8) is None
    counter = itertools.count()
    generated = emit_vector_recipe(
        cast("list[ast.Assign]", ast.parse(source).body),
        _expr("value"),
        coordinate="k",
        width=8,
        tensors={"A": CopyTensorFacts("cutlass.Float16", (24, 1), 16)},
        aligned_names={"k": 8},
        destination="shared",
        destination_indices=(_expr("m"), _expr("k")),
        fresh_name=lambda name: f"recipe_{next(counter)}_{name}",
    )
    emitted = ast.unparse(ast.Module(body=generated, type_ignores=[]))
    assert "autovec_copy" not in emitted
    assert _check_recipe(source, "cutlass.Float16", 8) == 0


def _predicate_recipe(depth: int) -> str:
    statements = ["p0 = k < 16"]
    statements.extend(f"p{i} = p{i - 1} and p{i - 1}" for i in range(1, depth + 1))
    statements.extend(
        [
            f"loaded = (A.iterator + k).load() if p{depth} else cutlass.Float16(0)",
            "value = cutlass.Float16(loaded * 0.5)",
        ]
    )
    return "\n".join(statements)


def test_small_predicate_dag_retains_masked_vector_copy() -> None:
    source = _predicate_recipe(2)
    assert _load_plan(source, "cutlass.Float16", 8) is not None
    assert _check_recipe(source, "cutlass.Float16", 8) == 1


@pytest.mark.parametrize("depth", [12, 24])
def test_large_predicate_dag_has_bounded_proof_work(depth: int) -> None:
    source = _predicate_recipe(depth)
    interval = _Analysis.interval
    calls = 0
    limit = 128 * len(ast.parse(source).body)

    def limited_interval(self: _Analysis, value: ast.expr) -> bool:
        nonlocal calls
        calls += 1
        # A deterministic work cap catches exponential traversal before an
        # adversarial test can itself hang. It does not require a specific
        # memoization layout or where a conservative rejection happens.
        assert calls <= limit
        return interval(self, value)

    with patch.object(_Analysis, "interval", limited_interval):
        assert _load_plan(source, "cutlass.Float16", 8) is None
        assert _check_recipe(source, "cutlass.Float16", 8) == 0


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _float_shift_matmul(
    a: torch.Tensor, b: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    m = a.size(0)
    k, n = b.shape
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    for tile_m, tile_n in hl.tile([m, n]):
        displacement = (shift[0] * 8).to(torch.int32)
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            source_k = tile_k.index + displacement
            left = a[tile_m, source_k] * 0.5
            acc = torch.addmm(acc, left, b[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(out.dtype)
    return out


def _shift_config(recipe: str, compute: str) -> helion.Config:
    return helion.Config(
        block_sizes=[64, 32, 32],
        num_threads=[4, 32, 1],
        cute_vector_widths=[1, 1, 1],
        cute_lane_layouts=["blocked"] * 3,
        cute_collective_mma=True,
        cute_collective_compute=compute,
        cute_collective_copy="scalar",
        cute_collective_recipe=recipe,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("recipe", ["vector", "vector_unrolled"])
@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@skipUnlessBackends(["cute"])
def test_float_shifted_matmul_keeps_only_aligned_operand_vectorized(
    dtype: torch.dtype, recipe: str, compute: str
) -> None:
    inputs = (
        torch.empty((65, 80), dtype=dtype),
        torch.empty((64, 48), dtype=dtype)[:, :37],
        torch.tensor([0.125], dtype=torch.float32),
    )
    with _cpu_target():
        bound = _float_shift_matmul._bind_isolated(inputs)
        source = bound.to_code(_shift_config(recipe, compute))
    assert "cute.gemm(" in source
    assert ("tcgen05.CtaGroup.ONE" in source) == (compute == "tcgen05")
    assert "cute.make_ptr(a.element_type" not in source
    assert "cute.make_ptr(b.element_type" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("recipe", ["vector", "vector_unrolled"])
@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@skipUnlessBackends(["cute"])
def test_float_shifted_matmul_cuda_mutations(
    dtype: torch.dtype, recipe: str, compute: str, tmp_path: Path
) -> None:
    if compute == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(20260912)
    # Powers-of-two inputs make each scaled operand and the FP32 dot exact.
    # A one-element displacement invalidates its 16-byte vector alignment;
    # B retains an aligned row stride and has a partial N tile/vector.
    a = torch.randint(-8, 9, (65, 80), device=CUDA_DEVICE).to(dtype) * 0.125
    b_storage = torch.randint(-8, 9, (64, 48), device=CUDA_DEVICE).to(dtype) * 0.125
    b = b_storage[:, :37]
    shift = torch.tensor([0.125], device=CUDA_DEVICE, dtype=torch.float32)
    inputs = (a, b, shift)
    reference = _float_shift_matmul._bind_isolated(inputs)
    scalar_config = _shift_config("scalar", compute)
    reference.set_config(scalar_config)
    bound = _float_shift_matmul._bind_isolated(inputs)
    config = _shift_config(recipe, compute)
    bound.set_config(config)
    source = bound.to_code(config)
    (tmp_path / "candidate.generated.py").write_text(source)
    (tmp_path / "control.generated.py").write_text(reference.to_code(scalar_config))
    assert "cute.gemm(" in source
    assert ("tcgen05.CtaGroup.ONE" in source) == (compute == "tcgen05")
    assert "cute.make_ptr(a.element_type" not in source
    assert "cute.make_ptr(b.element_type" in source

    def run() -> torch.Tensor:
        return bound(*inputs)

    replay = _make_cudagraph_replay(run)
    graph_output = replay()
    expected = torch.empty((65, 37), device=CUDA_DEVICE, dtype=dtype)
    for displacement in (1, 3, 7):
        a.mul_(0.5)
        b.mul_(-1)
        shift.fill_(displacement / 8)
        expected = (
            (a[:, displacement : displacement + 64] * 0.5).float() @ b.float()
        ).to(dtype)
        torch.testing.assert_close(reference(*inputs), expected, rtol=0, atol=0)
        torch.testing.assert_close(run(), expected, rtol=0, atol=0)
        graph_output.fill_(float("nan"))
        torch.testing.assert_close(replay(), expected, rtol=0, atol=0)
    # New allocations retain the original view's row stride and reuse the
    # compiled ordinary dynamic binding, including its alignment guards.
    replacements = (a.clone(), b_storage.clone()[:, :37], shift.clone())
    torch.testing.assert_close(reference(*replacements), expected, rtol=0, atol=0)
    torch.testing.assert_close(bound(*replacements), expected, rtol=0, atol=0)
