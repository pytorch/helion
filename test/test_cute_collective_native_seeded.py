from __future__ import annotations

import ast

from examples.mamba2_chunk_scan import helion_mamba2_chunk_scan_kernel
import pytest
import torch

from test._cute_binding import _cpu_bind
from test.test_cute_collective_dot import _invariant_seed_dot
from test.test_cute_collective_seeds import _independent_matmuls
from test.test_cute_collective_tcgen05 import _scalar_epilogues

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._compiler.cute.collective_tcgen05 import CollectiveTcgen05Plan
from helion._compiler.cute.collective_tcgen05 import CollectiveTmemResource
from helion._testing import skipUnlessBackends
from helion.autotuner.accuracy import _chunked_assert_close
from helion.autotuner.benchmarking import _make_cudagraph_replay
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

CUDA_DEVICE = "cuda"


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _seeded_contraction(
    a: torch.Tensor, b: torch.Tensor, seed: torch.Tensor, limit: int
) -> torch.Tensor:
    out = torch.empty_like(seed)
    for row, column in hl.tile([a.size(0), b.size(0)]):
        acc = seed[row, column]
        for reduction in hl.tile(limit):
            acc = hl.dot(a[row, reduction] * 0.5, b[column, reduction].T, acc=acc)
        out[row, column] = acc
    return out


@helion.kernel(backend="cute", static_shapes=False, autotune_effort="none")
def _seed_written_in_place(
    a: torch.Tensor, b: torch.Tensor, seed: torch.Tensor
) -> torch.Tensor:
    for row, column in hl.tile([a.size(0), b.size(0)]):
        acc = seed[row, column]
        for reduction in hl.tile(a.size(1)):
            acc = hl.dot(a[row, reduction] * 0.5, b[column, reduction].T, acc=acc)
        seed[row, column] = acc
    return seed


def _config(
    bm: int = 64,
    bn: int = 32,
    bk: int = 16,
    *,
    enabled: bool = True,
    compute: str = "tcgen05",
) -> helion.Config:
    return helion.Config(
        block_sizes=[bm, bn, bk],
        num_threads=[128 // bn, bn, 1],
        cute_vector_widths=[1, 1, 1],
        cute_collective_mma=True,
        cute_collective_copy="async_cached",
        cute_collective_compute=compute,
        cute_collective_native_seeded=enabled,
    )


def _inputs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    return (
        torch.empty((130, 80), dtype=dtype)[:, :78],
        torch.empty((70, 80), dtype=dtype)[:, :78],
        torch.empty((130, 70), dtype=torch.float32),
        78,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tile", [(64, 32, 16), (64, 64, 32), (128, 64, 128)])
@skipUnlessBackends(["cute"])
def test_native_uploads_float32_carry_before_accumulating(
    dtype: torch.dtype, tile: tuple[int, int, int]
) -> None:
    bound = _cpu_bind(_seeded_contraction, _inputs(dtype))
    source = bound.to_code(_config(*tile))
    warp = bound.to_code(_config(*tile, compute="warp"))
    assert "tcgen05.CtaGroup.ONE" in source
    assert "LdMatrix8x8x16bOp" not in source
    assert "cute.arch.fence_view_async_tmem_store()" in source
    assert "tcgen05.Field.ACCUMULATE, True" in source
    assert "seed.iterator" in source
    assert "St16x128bOp" in source if tile[0] == 64 else "St32x32bOp" in source
    assert _scalar_epilogues(source) == _scalar_epilogues(warp)
    assert source.index("fence_view_async_tmem_store") < source.index("cute.gemm(")


@skipUnlessBackends(["cute"])
def test_seeded_native_is_optional_and_survives_config_roundtrip() -> None:
    bound = _cpu_bind(_seeded_contraction, _inputs(torch.bfloat16))
    assert "tcgen05.CtaGroup.ONE" not in bound.to_code(_config(enabled=False))
    assert "tcgen05.CtaGroup.ONE" not in bound.to_code(_config(compute="warp"))
    with bound.env:
        generation = ConfigGeneration(bound.config_spec)
        _flat, config = generation.canonicalize_flat(generation.flatten(_config()))
    assert config["cute_collective_native_seeded"] is True
    assert "tcgen05.CtaGroup.ONE" in bound.to_code(config)


@pytest.mark.parametrize("seeded", [False, True])
@skipUnlessBackends(["cute"])
def test_seed_search_distinguishes_nonzero_carries(seeded: bool) -> None:
    if seeded:
        bound = _cpu_bind(_seeded_contraction, _inputs(torch.bfloat16))
    else:
        bound = _cpu_bind(
            _independent_matmuls,
            tuple(
                torch.empty(shape, dtype=torch.bfloat16)
                for shape in ((128, 256), (256, 64), (96, 64), (64, 256))
            ),
        )
    assert bound.host_function is not None
    with bound.env:
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        generation = ConfigGeneration(bound.config_spec)
        recovered = [
            generation.canonicalize_flat(generation.flatten(config))[1]
            for config in seeds
        ]
    native = [
        config
        for config in recovered
        if config.get("cute_collective_compute") == "tcgen05"
    ]
    assert native
    assert (
        any(config.get("cute_collective_native_seeded") is True for config in native)
        is seeded
    )
    assert any(
        config.get("cute_collective_native_seeded") is not True for config in native
    )
    if seeded:
        source = bound.to_code(
            next(
                config
                for config in native
                if config.get("cute_collective_native_seeded")
            )
        )
        assert "fence_view_async_tmem_store" in source


@pytest.mark.parametrize("reason", ["invariant", "written_seed", "narrow_seed"])
@skipUnlessBackends(["cute"])
def test_seeded_native_does_not_relax_carry_dtype_or_alias_proofs(reason: str) -> None:
    a, b, seed, _limit = _inputs(torch.float16)
    if reason == "invariant":
        bound = _cpu_bind(_invariant_seed_dot, (a, b.T, seed))
    elif reason == "written_seed":
        bound = _cpu_bind(_seed_written_in_place, (a, b, seed))
        for enabled in (False, True):
            with pytest.raises(
                helion.exc.BackendUnsupported,
                match="operand loads may alias row-loop writes",
            ):
                bound.to_code(_config(enabled=enabled))
        return
    else:
        bound = _cpu_bind(_seeded_contraction, (a, b, seed.half(), 78))
    source = bound.to_code(_config())
    assert "tcgen05.CtaGroup.ONE" not in source
    assert "fence_view_async_tmem_store" not in source


@pytest.mark.parametrize("bm", [64, 128])
@pytest.mark.parametrize("bn", [32, 64])
def test_seed_store_wait_precedes_cta_handoff(bm: int, bn: int) -> None:
    plan = CollectiveTcgen05Plan(
        "tile",
        "tid",
        bm,
        bn,
        16,
        "cutlass.Float16",
        CollectiveTmemResource("mem", bn),
        zero_seed=False,
    )
    source = ast.unparse(ast.Module(body=plan.seed_from_shared(), type_ignores=[]))
    assert source.index("cute.copy(") < source.index("fence_view_async_tmem_store")
    assert source.index("fence_view_async_tmem_store") < source.index("sync_threads")
    assert "cutlass.Float32" in source
    finish = plan.finish(ast.Constant(0), ast.Name(id="limit", ctx=ast.Load()))
    read = next(node for node in finish if isinstance(node, ast.If))
    assert ast.literal_eval(read.test) is True


@pytest.mark.parametrize("value", [0, 1, "true", None])
@skipUnlessBackends(["cute"])
def test_seeded_native_requires_boolean_config(value: object) -> None:
    bound = _cpu_bind(_seeded_contraction, _inputs(torch.float16))
    config = helion.Config.from_dict(
        _config().config | {"cute_collective_native_seeded": value}
    )
    with pytest.raises(helion.exc.InvalidConfig, match="must be a boolean"):
        bound.to_code(config)


@pytest.mark.parametrize("capability", [(8, 0), (9, 0), (12, 0), None])
@skipUnlessBackends(["cute"])
def test_seeded_native_is_available_only_on_sm100(
    capability: tuple[int, int] | None,
) -> None:
    bound = _cpu_bind(_seeded_contraction, _inputs(torch.bfloat16))
    bound.config_spec.target_device_capability = capability
    with pytest.raises(
        helion.exc.InvalidConfig,
        match="cute_collective_native_seeded requires SM100-family",
    ):
        bound.to_code(_config())


def _mamba_sizes(shape: int) -> tuple[tuple[int, ...], ...]:
    batch, heads, groups, sequence, chunk, head, state = (
        (2, 8, 2, 1024, 64, 64, 64),
        (2, 16, 4, 2048, 128, 64, 128),
    )[shape - 1]
    chunks = sequence // chunk
    return (
        (batch, chunks, groups, chunk, chunk),
        (batch, sequence, heads, head),
        (batch, heads, chunks, chunk),
        (batch, heads, chunks, chunk),
        (batch, sequence, groups, state),
        (batch, chunks, heads, head, state),
        (heads,),
    )


def _mamba_kernel(*, static_shapes: bool) -> helion.Kernel:
    return helion.kernel(
        helion_mamba2_chunk_scan_kernel.fn,
        backend="cute",
        static_shapes=static_shapes,
        cute_region_fission=True,
        cute_full_slice_matmul_tiling=True,
        autotune_effort="none",
    )


@pytest.mark.parametrize("shape", [1, 2])
@pytest.mark.parametrize("static_shapes", [False, True])
@skipUnlessBackends(["cute"])
def test_native_seed_search_covers_both_original_contractions(
    shape: int, static_shapes: bool
) -> None:
    inputs = tuple(
        torch.empty(size, dtype=torch.bfloat16) for size in _mamba_sizes(shape)
    )
    bound = _cpu_bind(_mamba_kernel(static_shapes=static_shapes), inputs)
    assert bound.host_function is not None
    with bound.env:
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    seeded = next(
        config for config in seeds if config.get("cute_collective_native_seeded")
    )
    source = bound.to_code(seeded)
    mixed = bound.to_code(
        helion.Config.from_dict(
            seeded.config | {"cute_collective_native_seeded": False}
        )
    )
    assert source.count("tcgen05.CtaGroup.ONE") == 2
    assert source.count("cute.gemm(") == 2
    assert "warp.MmaF16BF16Op" not in source
    assert source.count("mul.rn.f32") == mixed.count("mul.rn.f32") == 5
    assert _scalar_epilogues(source) == _scalar_epilogues(mixed)
    assert mixed.count("tcgen05.CtaGroup.ONE") == 1
    assert mixed.count("warp.MmaF16BF16Op") == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "tile,stride", [((64, 32, 16), 80), ((64, 64, 32), 78), ((128, 64, 128), 80)]
)
@skipUnlessBackends(["cute"])
def test_seeded_native_tails_empty_k_and_mutating_graphs(
    dtype: torch.dtype, tile: tuple[int, int, int], stride: int
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(1709)
    a = torch.randn((130, stride), device=CUDA_DEVICE, dtype=dtype)[:, :78]
    b = torch.randn((70, stride), device=CUDA_DEVICE, dtype=dtype)[:, :78]
    seed = torch.randn((130, 70), device=CUDA_DEVICE, dtype=torch.float32)
    inputs = (a, b, seed, 78)
    bound = _seeded_contraction._bind_isolated(inputs)
    config = _config(*tile)
    assert "fence_view_async_tmem_store" in bound.to_code(config)
    bound.set_config(config)

    def expected() -> torch.Tensor:
        return seed + (a * 0.5).float() @ b.float().T

    def run() -> torch.Tensor:
        return bound(*inputs)

    torch.testing.assert_close(run(), expected(), rtol=1e-4, atol=1e-4)
    replay = _make_cudagraph_replay(run)
    for _iteration in range(3):
        a.normal_()
        b.normal_()
        seed.normal_()
        torch.testing.assert_close(replay(), expected(), rtol=1e-4, atol=1e-4)
    new_a = torch.empty((130, stride), device=CUDA_DEVICE, dtype=dtype)[:, :78].copy_(a)
    new_b = torch.empty((70, stride), device=CUDA_DEVICE, dtype=dtype)[:, :78].copy_(b)
    torch.testing.assert_close(
        bound(new_a, new_b, seed.clone(), 78), expected(), rtol=1e-4, atol=1e-4
    )
    # An empty reduction must preserve all FP32 bits, including signed zero,
    # infinities, a quiet NaN payload, and the smallest positive subnormal.
    patterns = torch.tensor(
        [0, -2147483648, 2139095040, -8388608, 2143289345, 1],
        device=CUDA_DEVICE,
        dtype=torch.int32,
    )
    seed.view(torch.int32).flatten().copy_(
        patterns.repeat((seed.numel() + 5) // 6)[: seed.numel()]
    )
    actual = bound(a, b, seed, 0)
    torch.testing.assert_close(
        actual.view(torch.int32), seed.view(torch.int32), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@skipUnlessBackends(["cute"])
def test_native_accumulates_into_seed_before_each_k_instruction(
    dtype: torch.dtype,
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    a = torch.full((64, 32), 2.0, device=CUDA_DEVICE, dtype=dtype)
    b = torch.full((32, 32), 1.0 / 16, device=CUDA_DEVICE, dtype=dtype)
    seed = torch.full((64, 32), 2.0**24, device=CUDA_DEVICE, dtype=torch.float32)
    # Each k16 dot contributes 1, which rounds away at this FP32 seed.
    # Adding the complete product (2) afterwards would change the result.
    assert not torch.equal(seed + 2, seed)
    bound = _seeded_contraction._bind_isolated((a, b, seed, 32))
    bound.set_config(_config())
    torch.testing.assert_close(bound(a, b, seed, 32), seed, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [1, 2])
@pytest.mark.parametrize("tile", [(64, 32, 64), (64, 64, 64), (128, 64, 128)])
@skipUnlessBackends(["cute"])
def test_seeded_native_original_mamba_strict_accuracy_and_mutated_graphs(
    shape: int, dtype: torch.dtype, tile: tuple[int, int, int]
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    sizes = _mamba_sizes(shape)
    torch.manual_seed(20260911 + shape)
    x = torch.randn(sizes[1], device=CUDA_DEVICE, dtype=torch.bfloat16)
    dt = torch.rand(sizes[2], device=CUDA_DEVICE, dtype=x.dtype)
    decay = -torch.rand_like(dt).cumsum(-1)
    c = torch.randn(sizes[4], device=CUDA_DEVICE, dtype=x.dtype)
    cb = torch.randn(sizes[0], device=CUDA_DEVICE, dtype=x.dtype)
    previous = torch.randn(sizes[5], device=CUDA_DEVICE, dtype=x.dtype)
    residual = torch.randn(sizes[6], device=CUDA_DEVICE, dtype=x.dtype)
    arguments = tuple(
        value.to(dtype) for value in (cb, x, dt, decay, c, previous, residual)
    )
    kernel = _mamba_kernel(static_shapes=True)
    reference = kernel._bind_isolated(arguments)
    reference.set_config(reference.config_spec.autotune_reference_config())
    bound = kernel._bind_isolated(arguments)
    bm, bn, bk = tile
    config = helion.Config(
        block_sizes=[bm, bn, bk],
        num_threads=[128 // bn, bn, 1, 1],
        cute_vector_widths=[1] * 7,
        cute_lane_layouts=["blocked"] * 7,
        cute_collective_mma=True,
        cute_collective_compute="tcgen05",
        cute_collective_copy="async_cached",
        cute_collective_native_seeded=True,
    )
    source = bound.to_code(config)
    assert source.count("tcgen05.CtaGroup.ONE") == 2
    assert source.count("mul.rn.f32") == 5
    assert "warp.MmaF16BF16Op" not in source
    bound.set_config(config)

    def compare(actual: torch.Tensor, expected: torch.Tensor) -> None:
        _chunked_assert_close(
            actual, expected, atol=0.01, rtol=0.01, scale_atol_by_expected_rms=True
        )

    def run() -> torch.Tensor:
        return bound(*arguments)

    expected = reference(*arguments)
    for _iteration in range(3):
        compare(run(), expected)
    replay = _make_cudagraph_replay(run)
    for factor in (0.875, 1.125, 0.75):
        arguments[1].mul_(factor)
        arguments[5].mul_(factor)
        compare(replay(), reference(*arguments))
    replacement = tuple(value.clone() for value in arguments)
    compare(bound(*replacement), reference(*replacement))
