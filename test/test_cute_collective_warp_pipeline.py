from __future__ import annotations

import ast
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test._cute_binding import _cpu_bind
from test.test_cute_collective_chunk_seeds import _partitioned_matmul

import helion
from helion._compiler.cute.collective_warp_pipeline import pipeline_collective_loop
from helion._compiler.cute.collective_warp_pipeline import signed_int32_bound
from helion._testing import skipUnlessBackends
from helion.autotuner.benchmarking import _make_cudagraph_replay
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import InvalidConfig
import helion.language as hl

CUDA_DEVICE = "cuda"


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


@pytest.mark.parametrize("stages", [2, 3, 4])
@pytest.mark.parametrize("start,stop", [(0, 0), (7, 5), (0, 1), (3, 65), (-31, 311)])
def test_pipeline_preserves_reads_and_consume_order(
    stages: int, start: int, stop: int
) -> None:
    loop = ast.parse("for k in range(start, stop, 32):\n    pass").body[0]
    assert isinstance(loop, ast.For)
    copies = ast.parse("issue(k, p_stage_write)").body
    compute = ast.parse("consume(k, p_stage_read)").body
    schedule = pipeline_collective_loop(
        loop,
        copies,
        compute,
        prefix="p",
        start=_expr("cutlass.Int32(start)"),
        stop=_expr("cutlass.Int32(stop)"),
        block_k=32,
        stages=stages,
    )
    memory: dict[int, int] = {}
    pending: list[list[tuple[int, int]]] = []
    issued: list[tuple[int, int]] = []
    reads: list[int] = []
    consumed: list[int] = []

    def issue(k: int, stage: int) -> None:
        assert start <= k < stop
        assert 0 <= stage < stages
        reads.append(k)
        issued.append((stage, k))

    def commit() -> None:
        pending.append(issued.copy())
        issued.clear()

    def wait(distance: int) -> None:
        while len(pending) > distance:
            memory.update(pending.pop(0))

    def consume(k: int, stage: int) -> None:
        assert memory[stage] == k
        consumed.append(k)

    namespace = {
        "cutlass": SimpleNamespace(Int32=int, Int64=int, range_constexpr=range),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                cp_async_commit_group=commit,
                cp_async_wait_group=wait,
                sync_threads=lambda: None,
            )
        ),
        "start": start,
        "stop": stop,
        "issue": issue,
        "consume": consume,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=schedule, type_ignores=[])),
            "<pipeline-test>",
            "exec",
        ),
        namespace,
    )
    assert consumed == list(range(start, stop, 32))
    assert reads == consumed
    assert not pending and not issued


@pytest.mark.parametrize("stages", [2, 3, 4])
@pytest.mark.parametrize(
    "start,stop",
    [
        (2**30, 2**30 + 1),
        (-(2**30) - 1, -(2**30) + 64),
        (2**31 - 65, 2**31 - 2),
        (-(2**31), -(2**31) + 97),
        (2**31 - 2, 2**31 - 2),
        (-(2**31), -(2**31)),
        (2**31 - 2, 2**31 - 65),
        (-(2**31) + 97, -(2**31)),
    ],
)
def test_pipeline_preserves_recipe_integer_width(
    stages: int, start: int, stop: int
) -> None:
    loop = ast.parse(
        "for k in range(cutlass.Int32(start), cutlass.Int32(stop), 32):\n    pass"
    ).body[0]
    assert isinstance(loop, ast.For)
    schedule = pipeline_collective_loop(
        loop,
        ast.parse("issue(k, p_stage_write)").body,
        ast.parse("consume(k, p_stage_read)").body,
        prefix="p",
        start=_expr("cutlass.Int32(start)"),
        stop=_expr("cutlass.Int32(stop)"),
        block_k=32,
        stages=stages,
    )
    memory: dict[int, tuple[bool, bool]] = {}
    pending: list[list[tuple[int, tuple[bool, bool]]]] = []
    issued: list[tuple[int, tuple[bool, bool]]] = []
    reads: list[int] = []
    consumed: list[int] = []

    def predicates(k: np.int32 | np.int64) -> tuple[bool, bool]:
        # These integer masks intentionally overflow before comparison. A
        # widened copy coordinate changes them even when k itself fits Int32.
        with np.errstate(over="ignore"):
            return bool(k * np.int32(2) < 0), bool(k + np.int32(1024) < 0)

    def issue(k: np.int32 | np.int64, stage: int) -> None:
        assert start <= k < stop
        assert 0 <= stage < stages
        reads.append(int(k))
        issued.append((int(stage), predicates(k)))

    def commit() -> None:
        pending.append(issued.copy())
        issued.clear()

    def wait(distance: int) -> None:
        while len(pending) > distance:
            memory.update(pending.pop(0))

    def consume(k: np.int32, stage: int) -> None:
        assert memory[int(stage)] == predicates(k)
        consumed.append(int(k))

    namespace = {
        "cutlass": SimpleNamespace(
            Int32=np.int32, Int64=np.int64, range_constexpr=range
        ),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                cp_async_commit_group=commit,
                cp_async_wait_group=wait,
                sync_threads=lambda: None,
            )
        ),
        # Preserve the original signed Int32 induction-variable semantics.
        "range": lambda begin, end, step: (
            np.int32(k) for k in range(int(begin), int(end), step)
        ),
        "start": start,
        "stop": stop,
        "issue": issue,
        "consume": consume,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=schedule, type_ignores=[])),
            "<typed-pipeline-test>",
            "exec",
        ),
        namespace,
    )
    assert consumed == list(range(start, stop, 32))
    assert reads == consumed
    assert not pending and not issued


@pytest.mark.parametrize("source", ["limit", "cutlass.Int64(limit)", str(2**31)])
def test_wider_or_unknown_bounds_keep_single_stage(source: str) -> None:
    assert not signed_int32_bound(_expr(source))


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _gathered_gemm(
    a: torch.Tensor, b: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    m = rows.numel()
    k, n = b.shape
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for row, column in hl.tile([m, n]):
        source = rows[row]
        acc = hl.zeros([row, column], dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[source, inner], b[inner, column])
        out[row, column] = acc.to(out.dtype)
    return out


def _config(stages: int, *, partitions: int | None = None) -> helion.Config:
    values = {
        "block_sizes": [32, 64, 32],
        "num_threads": [2, 64, 1],
        "cute_collective_mma": True,
        "cute_collective_copy": "async_cached",
        "cute_collective_stages": stages,
    }
    if partitions is not None:
        values["partitions"] = partitions
    return helion.Config.from_dict(values)


@pytest.mark.parametrize("stages", [1, 2, 3, 4])
@skipUnlessBackends(["cute"])
def test_pipeline_codegen_and_config_transfer(stages: int) -> None:
    inputs = (
        torch.empty((64, 16384), dtype=torch.float16),
        torch.empty((16384, 128), dtype=torch.float16),
    )
    bound = _cpu_bind(_partitioned_matmul, inputs)
    config = _config(stages, partitions=64)
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
        generation = ConfigGeneration(bound.config_spec)
        transferred = generation.unflatten(generation.flatten(normalized))
    assert transferred.get("cute_collective_stages", 1) == stages
    code = bound.to_code(transferred)
    assert "cute.gemm(" in code
    if stages > 1:
        assert f"cute.arch.cp_async_wait_group({stages - 1})" in code
        assert "collective_prefetch_k" in code
        assert f"(32, 32, {stages})" in code
    else:
        assert "collective_prefetch_k" not in code


@skipUnlessBackends(["cute"])
def test_pipeline_seed_survives_chunk_seed_transfer() -> None:
    inputs = (
        torch.empty((64, 16384), dtype=torch.float16),
        torch.empty((16384, 128), dtype=torch.float16),
    )
    bound = _cpu_bind(_partitioned_matmul, inputs)
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        seeds = [config for _flat, config in generation.seed_flat_config_pairs()]
    assert any(
        config.get("cute_collective_stages") == 4
        and config.get("cute_collective_compute", "warp") == "warp"
        and config["partitions"] == 64
        and config.block_sizes == [32, 64, 32]
        for config in seeds
    )
    with pytest.raises(InvalidConfig, match="cute_collective_stages"):
        bound.config_spec.normalize(_config(True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("stages", [2, 3, 4])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@skipUnlessBackends(["cute"])
def test_pipeline_gathered_tails_and_graphs(
    stages: int, transposed: bool, dtype: torch.dtype
) -> None:
    torch.manual_seed(781)
    a = torch.randn((193, 88), device=CUDA_DEVICE, dtype=dtype)[:, :86] * 0.025
    # Preserve padded row strides, including a transposed K-major RHS.
    a_storage = torch.empty((193, 88), device=CUDA_DEVICE, dtype=dtype)
    a_storage[:, :86].copy_(a)
    a = a_storage[:, :86]
    if transposed:
        b = torch.randn((70, 88), device=CUDA_DEVICE, dtype=dtype)[:, :86].T
    else:
        b = torch.randn((86, 80), device=CUDA_DEVICE, dtype=dtype)[:, :70]
    rows = torch.randperm(193, device=CUDA_DEVICE)[:131].to(torch.int32)
    bound = _gathered_gemm._bind_isolated((a, b, rows))
    config = _config(stages)
    code = bound.to_code(config)
    assert f"cute.arch.cp_async_wait_group({stages - 1})" in code
    bound.set_config(config)

    def run() -> torch.Tensor:
        return bound(a, b, rows)

    def expected() -> torch.Tensor:
        return (a[rows.long()].float() @ b.float()).to(dtype)

    torch.testing.assert_close(run(), expected(), atol=2e-3, rtol=1e-2)
    replay = _make_cudagraph_replay(run)
    for _ in range(3):
        a.uniform_(-0.05, 0.05)
        b.uniform_(-0.05, 0.05)
        rows.copy_(torch.randperm(193, device=CUDA_DEVICE)[:131].to(torch.int32))
        torch.testing.assert_close(run(), expected(), atol=2e-3, rtol=1e-2)
        torch.testing.assert_close(replay(), expected(), atol=2e-3, rtol=1e-2)
