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
import helion.language as hl

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


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _matmul_counting_tiles(
    a: torch.Tensor, b: torch.Tensor, tiles: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
        hl.atomic_add(tiles, [0], 1)
        hl.atomic_add(rows, [tile_m], 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _matmul_counting_into_its_output(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.zeros((m, n), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
        hl.atomic_add(out, [0, 0], 1.0)
    return out


def _counting_config(compute: str) -> helion.Config:
    return helion.Config(
        block_sizes=[64, 64, 32],
        num_threads=[2, 64, 1],
        cute_collective_mma=True,
        cute_collective_compute=compute,
        cute_collective_copy="async_cached",
        cute_collective_stages=2,
    )


def _atomic_guards(code: str) -> dict[str, str]:
    """The guard of each atomic call, by the tensor it updates."""
    guards: dict[str, str] = {}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.If):
            for call in ast.walk(node.body[0]):
                if isinstance(call, ast.Call) and ast.unparse(call.func) == (
                    "cute.arch.atomic_add"
                ):
                    tensor = ast.unparse(call.args[0]).split(".iterator")[0]
                    guards[tensor.strip("(")] = ast.unparse(node.test)
    return guards


@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@skipUnlessBackends(["cute"])
def test_tile_counter_beside_a_collective_matmul_runs_at_the_first_lane(
    compute: str,
) -> None:
    # The epilogue runs the rows of the output tile a thread holds in a lane
    # loop.  The collective matmul's placeholder leaves the body to its later
    # lowering, undistributed; the tile counter covers no tile axis, so it is
    # pinned to the first lane of that loop.  The row counter varies with the
    # loop and keeps every lane.
    inputs = (
        torch.empty((256, 512), dtype=torch.float16),
        torch.empty((512, 256), dtype=torch.float16),
        torch.zeros((1,), dtype=torch.int32),
        torch.zeros((256,), dtype=torch.int32),
    )
    with _mock_cuda_unavailable(), _cpu_target():
        bound = _matmul_counting_tiles._bind_isolated(inputs)
        code = bound.to_code(_counting_config(compute))
    assert "cute.gemm(" in code
    guards = _atomic_guards(code)
    assert "and lane_0 == 0" in guards["tiles"], code
    assert "lane_0" not in guards["rows"], code


@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@skipUnlessBackends(["cute"])
def test_tile_counter_into_the_collective_output_rejects_the_config(
    compute: str,
) -> None:
    # Issued at the first lane only, the counter would run before the other
    # lanes' stores of the output it adds into.
    inputs = (
        torch.empty((256, 512), dtype=torch.float16),
        torch.empty((512, 256), dtype=torch.float16),
    )
    with _mock_cuda_unavailable(), _cpu_target():
        bound = _matmul_counting_into_its_output._bind_isolated(inputs)
        with pytest.raises(
            exc.BackendUnsupported,
            match="lane-invariant atomic on out issued at the first lane_0 would "
            "run beside another access of its tensors",
        ):
            bound.to_code(_counting_config(compute))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@skipUnlessBackends(["cute"])
def test_tile_counter_beside_a_collective_matmul_counts_each_tile_once(
    compute: str,
) -> None:
    if compute == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(5203)
    a = torch.randn((256, 512), device=CUDA_DEVICE, dtype=torch.float16)
    b = torch.randn((512, 256), device=CUDA_DEVICE, dtype=torch.float16)
    tiles = torch.zeros((1,), device=CUDA_DEVICE, dtype=torch.int32)
    rows = torch.zeros((256,), device=CUDA_DEVICE, dtype=torch.int32)
    args = (a, b, tiles, rows)
    bound = _matmul_counting_tiles._bind_isolated(args)
    config = _counting_config(compute)
    assert "cute.gemm(" in bound.to_code(config)
    bound.set_config(config)
    out = bound(*args)
    torch.testing.assert_close(out, a.float() @ b.float(), atol=1e-1, rtol=1e-2)
    # 64 x 64 tiles of a 256 x 256 output.
    assert tiles.item() == 16
    assert rows.tolist() == [4] * 256
