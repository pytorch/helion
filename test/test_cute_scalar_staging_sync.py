from __future__ import annotations

import ast
from dataclasses import replace
from typing import TYPE_CHECKING

from examples.aot_example import matmul_custom_key
import pytest
import torch

from test import test_cute_lowerings
from test.test_cute_fuse_mm_accumulation import _cpu_target

import helion
from helion._compiler.cute.cute_mma import _build_kloop_non_pipeline_consumer_if
from helion._compiler.cute.cute_mma import _build_kloop_pipeline_consumer_if
from helion._compiler.cute.cute_mma import _Tcgen05ScalarSmemSync
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _native_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for tm, tn in hl.tile([m, n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(k):
            acc = hl.dot(a[tm, tk], b[tk, tn], acc=acc)
        out[tm, tn] = acc.to(out.dtype)
    return out


@pytest.fixture(autouse=True)
def _target(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("HELION_CUTE_MMA_IMPL", "auto")
    with _cpu_target():
        yield


def _assert_copy_order(source: str, first_store: str) -> None:
    commit = source.index("cute.nvgpu.tcgen05.commit(")
    wait = source.index("cute.arch.mbarrier_wait(", commit)
    store = source.index(first_store, wait)
    fence = source.index("cute.arch.fence_view_async_shared()", store)
    rendezvous = source.index("cute.arch.sync_threads()", fence)
    assert commit < wait < store < fence < rendezvous
    assert "with cute.arch.elect_one():" in source[:commit]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("aot", [False, True])
@pytest.mark.parametrize(
    "shape,bm", [((73, 41, 78), 64), ((73, 41, 162), 128), ((128, 128, 78), 64)]
)
@skipUnlessBackends(["cute"])
def test_edge_scalar_staging_is_ordered(
    dtype: torch.dtype, aot: bool, shape: tuple[int, int, int], bm: int
) -> None:
    m, n, k = shape
    kernel = (
        helion.kernel(
            matmul_custom_key.fn,
            key=matmul_custom_key._key_fn,
            backend="cute",
            static_shapes=False,
            autotune_effort="none",
            dot_precision="ieee",
        )
        if aot
        else _native_matmul
    )
    # Keep TensorMap strides valid while retaining partial M/N/K extents, so
    # the test reaches both TMA and scalar producers within the same kernel.
    bound = kernel._bind_isolated(
        (
            torch.empty((m, ((k + 7) // 8) * 8), dtype=dtype)[:, :k],
            torch.empty((k, ((n + 7) // 8) * 8), dtype=dtype)[:, :n],
        )
    )
    source = bound.to_code(helion.Config(block_sizes=[bm, 32, 32]))
    assert "make_trivial_tiled_mma(" in source and "cute.gemm(" in source
    assert "mbarrier_init(tcgen05_scalar_smem_mbar, 1)" in source
    root = ast.parse(source)
    copies = [
        node
        for node in ast.walk(root)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "tcgen05_tma_full_tile"
        and any(
            isinstance(child, ast.Call)
            and ast.unparse(child.func) == "cute.arch.mbarrier_wait"
            for stmt in node.orelse
            for child in ast.walk(stmt)
        )
    ]
    assert len(copies) == 1
    full = ast.unparse(ast.Module(body=copies[0].body, type_ignores=[]))
    edge = ast.unparse(ast.Module(body=copies[0].orelse, type_ignores=[]))
    assert "mbarrier_wait(" not in full
    _assert_copy_order(edge, "sA_mma[")
    assert edge.index("sB_mma[") < edge.index("fence_view_async_shared()")
    assert "tcgen05_scalar_smem_phase ^ cutlass.Int32(1)" in edge


@skipUnlessBackends(["cute"])
def test_all_tma_kernel_keeps_its_pipeline() -> None:
    kernel = helion.kernel(
        matmul_custom_key.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated(
        (
            torch.empty((128, 128), dtype=torch.float16),
            torch.empty((128, 128), dtype=torch.float16),
        )
    )
    source = bound.to_code(helion.Config(block_sizes=[64, 32, 32]))
    assert "cute.gemm(" in source
    assert "tcgen05_scalar_smem" not in source
    assert "cute.nvgpu.tcgen05.commit(" not in source


@pytest.mark.parametrize("tma_a", [False, True])
def test_asymmetric_tma_also_orders_the_scalar_operand(tma_a: bool) -> None:
    args = test_cute_lowerings.TestPerKiterTmaBuilders()._make_args(
        use_tma_a=tma_a, use_tma_b=not tma_a
    )
    args = replace(
        args,
        scalar_smem_sync=_Tcgen05ScalarSmemSync("barrier", "phase", "exec_active"),
    )
    consumer = _build_kloop_non_pipeline_consumer_if(args)
    assert isinstance(consumer, ast.If)
    full = ast.unparse(ast.Module(body=consumer.body, type_ignores=[]))
    edge = ast.unparse(ast.Module(body=consumer.orelse, type_ignores=[]))
    _assert_copy_order(full, "smem_b[" if tma_a else "smem_a[")
    _assert_copy_order(edge, "smem_a[")
    assert ("smem_a[" in full) != tma_a
    assert ("smem_b[" in full) == tma_a


def test_two_cta_and_role_local_builder_has_no_scalar_barrier() -> None:
    args = test_cute_lowerings.TestPerKiterTmaBuilders()._make_args(
        is_two_cta=True, static_full_tiles=True
    )
    source = ast.unparse(
        _build_kloop_pipeline_consumer_if(args, include_scalar_fallback=False)
    )
    assert "mbarrier_wait(" not in source
    assert "consumer_wait(" in source
