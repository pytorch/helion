from __future__ import annotations

import ast

from examples.matmul_split_k import matmul_split_k
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_collective_effects import _lower
from test.test_cute_fuse_mm_accumulation import _cpu_target

import helion
from helion import exc
from helion._compiler.cute.collective_matmul import _region_write_roots
from helion._testing import skipUnlessBackends
from helion.autotuner.benchmarking import _make_cudagraph_replay

CUDA_DEVICE = "cuda"


@pytest.mark.parametrize("scope", ["", ", scope='gpu'", ", scope='cta'"])
def test_unused_relaxed_atomic_add_has_known_output_effect(scope: str) -> None:
    body = ast.parse(
        "if active:\n"
        "    cute.arch.atomic_add("
        "(out.iterator + cute.crd2idx((m, n), out.layout)).llvm_ptr, "
        f"val=cutlass.Float32(value), sem='relaxed'{scope})\n"
    ).body
    original = ast.dump(ast.Module(body=body, type_ignores=[]))
    assert _region_write_roots(body, 0) == {"out"}
    assert ast.dump(ast.Module(body=body, type_ignores=[])) == original


@pytest.mark.parametrize(
    "source",
    [
        "cute.arch.atomic_add((out.iterator + i).llvm_ptr, val=v)",
        "cute.arch.atomic_add((out.iterator + i).llvm_ptr, val=v, sem='acquire')",
        "cute.arch.atomic_add((out.iterator + i).llvm_ptr, val=v, sem='release')",
        "cute.arch.atomic_add((out.iterator + i).llvm_ptr, val=v, sem='acq_rel')",
        "cute.arch.atomic_add((out.iterator + i).llvm_ptr, val=v, sem=ordering)",
        "cute.arch.atomic_add((out.iterator + i).llvm_ptr, val=v, sem='relaxed', scope=scope)",
        "cute.arch.atomic_add((out.iterator + i).llvm_ptr, val=v, sem='relaxed', other=True)",
        "cute.arch.atomic_add(ptr, val=v, sem='relaxed')",
        "cute.arch.atomic_add(ptr.llvm_ptr, val=v, sem='relaxed')",
        "cute.arch.atomic_add((out.iterator + other.iterator).llvm_ptr, val=v, sem='relaxed')",
        "cute.arch.atomic_add((out.iterator + unknown_index()).llvm_ptr, val=v, sem='relaxed')",
        "cute.arch.atomic_add((out.iterator + i).llvm_ptr, val=unknown_value(), sem='relaxed')",
        "old = cute.arch.atomic_add((out.iterator + i).llvm_ptr, val=v, sem='relaxed')",
    ],
)
def test_atomic_effect_requires_exact_relaxed_unused_form(source: str) -> None:
    assert _region_write_roots(ast.parse(source).body, 0) is None


@pytest.mark.parametrize(
    "pointer",
    [
        "out.iterator if active else ptr",
        "cute.where(active, out.iterator, ptr)",
        "ptr + cutlass.Int32(0)",
        "out.iterator + ptr",
    ],
)
def test_atomic_base_cannot_hide_pointer_alias(pointer: str) -> None:
    source = (
        "ptr = limits.iterator\n"
        f"cute.arch.atomic_add(({pointer}).llvm_ptr, "
        "val=cutlass.Int32(1), sem='relaxed')"
    )
    assert _region_write_roots(ast.parse(source).body, 0) is None
    with pytest.raises(exc.BackendUnsupported, match="unclassified effects"):
        _lower(source)


@pytest.mark.parametrize(
    "offset", ["cutlass.Int32(index)", "cutlass.Int64(index) * 8 + 2"]
)
def test_atomic_base_accepts_proven_integer_offsets(offset: str) -> None:
    body = ast.parse(
        f"cute.arch.atomic_add((out.iterator + {offset}).llvm_ptr, "
        "val=cutlass.Float32(value), sem='relaxed')"
    ).body
    assert _region_write_roots(body, 0) == {"out"}


def test_collective_staging_preserves_disjoint_atomic_statement() -> None:
    source = _lower(
        "cute.arch.atomic_add((out.iterator + 0).llvm_ptr, "
        "val=cutlass.Float16(0), sem='relaxed')"
    )
    assert "cute.gemm(" in source
    assert source.count("cute.arch.atomic_add(") == 1


@pytest.mark.parametrize("root", ["a", "b", "limits"])
def test_collective_staging_rejects_atomic_aliases(root: str) -> None:
    with pytest.raises(exc.BackendUnsupported, match="may alias row-loop writes"):
        _lower(
            f"cute.arch.atomic_add(({root}.iterator + 0).llvm_ptr, "
            "val=cutlass.Float16(0), sem='relaxed')"
        )


def test_collective_control_proof_includes_relaxed_atomics() -> None:
    with pytest.raises(exc.BackendUnsupported, match="control.flow"):
        _lower(
            root_prefix="count = (flags.iterator + 0).load()",
            outer_header="if count > 0",
            root_suffix="cute.arch.atomic_add((flags.iterator + 0).llvm_ptr, "
            "val=cutlass.Int32(1), sem='relaxed')",
        )


def _config(compute: str) -> helion.Config:
    return helion.Config(
        block_sizes=[64, 64, 32],
        num_threads=[2, 64, 1],
        cute_vector_widths=[1, 1, 1, 1],
        cute_collective_mma=True,
        cute_collective_compute=compute,
        cute_collective_copy="async",
        split_k=16,
    )


@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@skipUnlessBackends(["cute"])
def test_split_k_collective_keeps_atomic_epilogue(compute: str) -> None:
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    inputs = (
        torch.empty((64, 1024), dtype=torch.float16),
        torch.empty((1024, 128), dtype=torch.float16),
    )
    with _mock_cuda_unavailable(), _cpu_target():
        bound = kernel._bind_isolated(inputs)
        code = bound.to_code(_config(compute))
    assert "cute.gemm(" in code
    assert code.count("cute.arch.atomic_add(") == 1
    assert "cpasync.CopyG2SOp(" in code or "cp_async_shared_global(" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@pytest.mark.parametrize("bias", [False, True])
@skipUnlessBackends(["cute"])
def test_collective_atomic_tails_and_mutated_graphs(
    dtype: torch.dtype, compute: str, bias: bool
) -> None:
    if compute == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(4317)
    # Odd K/N test scalar tails while padded row strides admit vector copies.
    x = (torch.randn((97, 536), device=CUDA_DEVICE, dtype=dtype) * 0.025)[:, :530]
    y = (torch.randn((530, 80), device=CUDA_DEVICE, dtype=dtype) * 0.025)[:, :71]
    b = torch.randn((71,), device=CUDA_DEVICE, dtype=dtype) * 0.025
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    args = (x, y, lambda acc, tiles: acc + b[tiles[1]]) if bias else (x, y)
    bound = kernel._bind_isolated(args)
    config = _config(compute)
    code = bound.to_code(config)
    assert "cute.gemm(" in code
    assert code.count("cute.arch.atomic_add(") == 1
    bound.set_config(config)

    def run() -> torch.Tensor:
        return bound(*args)

    def expected() -> torch.Tensor:
        result = x.float() @ y.float()
        if bias:
            result += b.float()
        return result.to(dtype)

    torch.testing.assert_close(run(), expected(), atol=2e-3, rtol=1e-2)
    replay = _make_cudagraph_replay(run)
    for _iteration in range(3):
        x.uniform_(-0.05, 0.05)
        y.uniform_(-0.05, 0.05)
        b.uniform_(-0.05, 0.05)
        torch.testing.assert_close(run(), expected(), atol=2e-3, rtol=1e-2)
        torch.testing.assert_close(replay(), expected(), atol=2e-3, rtol=1e-2)
