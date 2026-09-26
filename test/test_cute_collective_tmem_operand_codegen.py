from __future__ import annotations

import ast
from functools import partial
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test.test_cute_collective_dot import _config as _dot_config
from test.test_cute_collective_dot import _full_slice_dot
from test.test_cute_collective_native_seeded import _config as _seeded_config
from test.test_cute_collective_native_seeded import _inputs
from test.test_cute_collective_native_seeded import _mamba_kernel
from test.test_cute_collective_native_seeded import _mamba_sizes
from test.test_cute_collective_native_seeded import _seed_written_in_place
from test.test_cute_collective_native_seeded import _seeded_contraction
from test.test_cute_collective_tcgen05 import _scalar_epilogues
from test.test_cute_collective_tmem_seed import _dependent_config
from test.test_cute_collective_tmem_seed import _dependent_contractions
from test.test_cute_collective_tmem_seed import _dependent_inputs
from test.test_cute_collective_vector_epilogue_codegen import _mamba_config
from test.test_cute_collective_vector_epilogue_codegen import _residual_matmul
from test.test_cute_fuse_mm_accumulation import _cpu_target
from test.test_cute_packed_collective_matmul import _config as _packed_config
from test.test_cute_packed_collective_matmul import _original_bound

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._compiler.cute.collective_matmul import (
    has_collective_tmem_operand_candidate,
)
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._testing import skipUnlessBackends
from helion.autotuner.accuracy import _chunked_assert_close
from helion.autotuner.benchmarking import _make_cudagraph_replay
from helion.autotuner.config_generation import ConfigGeneration

pytestmark = skipUnlessBackends(["cute"])

CUDA_DEVICE = "cuda"


def _sizes(shape: int) -> tuple[tuple[int, ...], ...]:
    if shape == 0:
        return (
            (1, 4, 1, 64, 64),
            (1, 256, 4, 32),
            (1, 4, 4, 64),
            (1, 4, 4, 64),
            (1, 256, 1, 32),
            (1, 4, 4, 32, 32),
            (4,),
        )
    return _mamba_sizes(shape)


def _mamba_tmem_config(
    shape: int, *, enabled: bool = True, seed: bool = True
) -> helion.Config:
    bn, bk = (32, 32) if shape == 0 else (64, 64)
    return helion.Config.from_dict(
        _mamba_config("vector_unrolled").config
        | {
            "block_sizes": [64, bn, bk],
            "num_threads": [128 // bn, bn, 1, 1],
            "cute_collective_tmem_a": enabled,
            "cute_collective_tmem_seed": seed,
            "cute_proven_bounds": True,
        }
    )


def _loop(source: str, name: str) -> str:
    node = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == name
    )
    return ast.unparse(node)


@pytest.mark.parametrize("shape", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_original_three_mamba_sizes_keep_b_staging_and_seed_arithmetic(
    shape: int, dtype: torch.dtype, static_shapes: bool
) -> None:
    args = tuple(torch.empty(size, dtype=dtype) for size in _sizes(shape))
    bound = _cpu_bind(_mamba_kernel(static_shapes=static_shapes), args)
    control = bound.to_code(_mamba_tmem_config(shape, enabled=False))
    source = bound.to_code(_mamba_tmem_config(shape))
    assert source.count("tcgen05.CtaGroup.ONE") == 2
    assert source.count("OperandSource.TMEM") == 1
    assert source.count("OperandSource.SMEM") == 1
    assert "collective_1_a_values" in source
    assert "collective_1_ap =" not in source
    assert "collective_tmem_seed_values" in source
    assert source.count("mul.rn.f32") == control.count("mul.rn.f32")
    assert source.count(".allocate(") == source.count(".relinquish_alloc_permit(") == 1
    assert _loop(source, "collective_1_bi") == _loop(control, "collective_1_bi")
    assert "cute.arch.fence_view_async_tmem_store()" in source


@pytest.mark.parametrize("bm,bn,bk", [(64, 32, 16), (64, 64, 32), (128, 64, 128)])
@pytest.mark.parametrize("recipe", ["scalar", "vector_unrolled"])
def test_computed_a_tails_preserve_original_scalar_epilogue(
    bm: int, bn: int, bk: int, recipe: str
) -> None:
    bound = _cpu_bind(_seeded_contraction, _inputs(torch.bfloat16))
    config = helion.Config.from_dict(
        _seeded_config(bm, bn, bk).config | {"cute_collective_recipe": recipe}
    )
    control = bound.to_code(config)
    source = bound.to_code(
        helion.Config.from_dict(config.config | {"cute_collective_tmem_a": True})
    )
    assert "OperandSource.TMEM" in source
    assert _scalar_epilogues(source) == _scalar_epilogues(control)
    assert _loop(source, "collective_bi") == _loop(control, "collective_bi")
    assert "cutlass.range_constexpr" in _loop(source, "collective_ai")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_partial_synthetic_k_keeps_inactive_recipe_guard(dtype: torch.dtype) -> None:
    args = (torch.empty((130, 21), dtype=dtype), torch.empty((21, 70), dtype=dtype))
    bound = _cpu_bind(_full_slice_dot, args)
    source = bound.to_code(_full_slice_tmem_config())
    assert source.count("OperandSource.TMEM") == 1
    first = _loop(source, "collective_ai")
    assert "if " in first and "< 21" in first
    assert "collective_a_values[collective_ai]" in first
    dtype_name = "Float16" if dtype == torch.float16 else "BFloat16"
    assert f"= cutlass.{dtype_name}(0)" in first


def _full_slice_tmem_config() -> helion.Config:
    return helion.Config.from_dict(
        _dot_config(full_slice=True).config
        | {
            "cute_collective_compute": "tcgen05",
            "cute_collective_copy": "async_cached",
            "cute_collective_recipe": "vector_unrolled",
            "cute_collective_tmem_a": True,
        }
    )


def test_tmem_a_coupled_seeds_and_roundtrip_include_bounds_and_seed_reuse() -> None:
    args = tuple(torch.empty(size, dtype=torch.bfloat16) for size in _sizes(2))
    bound = _cpu_bind(_mamba_kernel(static_shapes=False), args)
    assert bound.host_function is not None
    with bound.env:
        assert has_collective_tmem_operand_candidate(
            bound.host_function.device_ir.graphs
        )
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        assert any(seed.get("cute_collective_tmem_a", False) for seed in seeds)
        coupled = next(
            seed
            for seed in seeds
            if seed.get("cute_collective_tmem_a")
            and seed.get("cute_collective_tmem_seed")
            and seed.get("cute_proven_bounds")
        )
        generation = ConfigGeneration(bound.config_spec)
        _flat, recovered = generation.canonicalize_flat(generation.flatten(coupled))
        for key in (
            "cute_collective_tmem_a",
            "cute_collective_tmem_seed",
            "cute_proven_bounds",
        ):
            assert recovered[key] is True
        assert recovered["cute_collective_compute"] == "tcgen05"
        assert recovered["cute_collective_recipe"] == "vector_unrolled"
    assert "OperandSource.TMEM" in bound.to_code(recovered)


def test_direct_load_operands_do_not_add_duplicate_tmem_seeds() -> None:
    inputs = tuple(
        torch.empty(shape, dtype=torch.bfloat16)
        for shape in ((128, 256), (256, 64), (128, 64), (128, 64))
    )
    bound = _cpu_bind(_residual_matmul, inputs)
    assert bound.host_function is not None
    assert not has_collective_tmem_operand_candidate(
        bound.host_function.device_ir.graphs
    )
    with bound.env:
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    assert all(not seed.get("cute_collective_tmem_a", False) for seed in seeds)


@pytest.mark.parametrize("flag", [None, 0, 1, "true"])
def test_tmem_a_requires_a_boolean(flag: object) -> None:
    bound = _cpu_bind(_seeded_contraction, _inputs(torch.float16))
    with pytest.raises(helion.exc.InvalidConfig, match="cute_collective_tmem_a"):
        bound.to_code(
            helion.Config.from_dict(
                _seeded_config().config | {"cute_collective_tmem_a": flag}
            )
        )


@pytest.mark.parametrize("capability", [None, (8, 0), (9, 0), (12, 0)])
def test_tmem_a_requires_sm100(capability: tuple[int, int] | None) -> None:
    bound = _cpu_bind(_seeded_contraction, _inputs(torch.float16))
    bound.config_spec.target_device_capability = capability
    config = helion.Config.from_dict(
        _seeded_config(compute="warp", enabled=False).config
        | {"cute_collective_tmem_a": True}
    )
    with pytest.raises(
        helion.exc.InvalidConfig, match="cute_collective_tmem_a requires SM100"
    ):
        bound.to_code(config)


def test_warp_compute_retains_original_operand_path() -> None:
    bound = _cpu_bind(_seeded_contraction, _inputs(torch.float16))
    config = helion.Config.from_dict(
        _seeded_config(compute="warp").config | {"cute_collective_tmem_a": True}
    )
    assert "OperandSource.TMEM" not in bound.to_code(config)


@pytest.mark.parametrize("packed", [False, True])
def test_tf32_and_packed_a_keep_their_existing_native_paths(
    packed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if packed:
        # This control exercises the original packed contraction. Materialized
        # operands have separate producer axes and a native consumer below.
        monkeypatch.setenv("HELION_CUTE_MATERIALIZE_TRANSFORMED_OPERANDS", "0")
        bound = _original_bound()
        assert bound.env.cute_fission_plan is None
        config = _packed_config()
    else:
        bound = _cpu_bind(_seeded_contraction, _inputs(torch.float32))
        config = _seeded_config()
    control = bound.to_code(config)
    source = bound.to_code(
        helion.Config.from_dict(config.config | {"cute_collective_tmem_a": True})
    )
    assert "OperandSource.TMEM" not in source
    assert "OperandSource.SMEM" in source
    assert _scalar_epilogues(source) == _scalar_epilogues(control)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_materialized_packed_operand_keeps_native_shared_a(
    dtype: torch.dtype,
) -> None:
    with (
        _cpu_target(),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
    ):
        bound = _original_bound(
            (128, 256, 128),
            dtype,
            cute_region_fission=True,
            cute_materialize_transformed_operands=True,
            cute_full_slice_matmul_tiling=True,
            cute_segmented_matmul_tiling=True,
            cute_flatten_nested_reductions=True,
        )
        plan = bound.env.cute_fission_plan
        assert plan is not None and plan.region_count == 2
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            selected = next(
                seed
                for seed in bound.config_spec.autotune_seed_configs()
                if seed.get("tcgen05_cta_group") in ("one", "two")
            )
            config = bound._normalized_config_copy(
                helion.Config.from_dict(
                    bound.config_spec.default_config().config | selected.config
                )
            )
        control = bound.to_code(config)
        source = bound.to_code(
            helion.Config.from_dict(config.config | {"cute_collective_tmem_a": True})
        )

    def stages(code: str) -> list[str]:
        result = []
        for node in ast.walk(ast.parse(code)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "PyCodeCache"
                and node.func.attr == "load"
            ):
                stage = ast.literal_eval(node.args[0])
                assert isinstance(stage, str)
                result.append(stage)
        return result

    before, after = stages(control), stages(source)
    assert len(before) == len(after) == 2
    assert "OperandSource.SMEM" in after[1]
    assert all("OperandSource.TMEM" not in stage for stage in after)
    assert [ast.dump(ast.parse(stage)) for stage in before] == [
        ast.dump(ast.parse(stage)) for stage in after
    ]


def test_tmem_a_preserves_operand_alias_rejection() -> None:
    a, b, seed, _limit = _inputs(torch.float16)
    bound = _cpu_bind(_seed_written_in_place, (a, b, seed))
    config = helion.Config.from_dict(
        _seeded_config().config | {"cute_collective_tmem_a": True}
    )
    with pytest.raises(
        helion.exc.BackendUnsupported, match="operand loads may alias row-loop writes"
    ):
        bound.to_code(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("static_shapes", [False, True])
def test_original_mamba_tmem_a_cuda(
    shape: int, dtype: torch.dtype, static_shapes: bool
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(20260913)
    args = tuple(
        torch.randn(size, device=CUDA_DEVICE, dtype=torch.bfloat16).to(dtype)
        for size in _sizes(shape)
    )
    args[2].uniform_()
    args[3].copy_(-torch.rand_like(args[2]).cumsum(-1))
    kernel = _mamba_kernel(static_shapes=static_shapes)
    bound = kernel._bind_isolated(args)
    config = _mamba_tmem_config(shape)
    assert "OperandSource.TMEM" in bound.to_code(config)
    bound.set_config(config)
    control = kernel._bind_isolated(args)
    control.set_config(
        helion.Config.from_dict(config.config | {"cute_collective_tmem_a": False})
    )
    reference = kernel._bind_isolated(args)
    reference.set_config(reference.config_spec.autotune_reference_config())
    replay = _make_cudagraph_replay(lambda: bound(*args))
    for factor in (1.0, 0.875, 1.125):
        args[1].mul_(factor)
        args[5].mul_(factor)
        expected = control(*args)
        _chunked_assert_close(
            expected,
            reference(*args),
            atol=0.01,
            rtol=0.01,
            scale_atol_by_expected_rms=True,
        )
        torch.testing.assert_close(bound(*args), expected, rtol=0, atol=0)
        torch.testing.assert_close(replay(), expected, rtol=0, atol=0)
    replacements = tuple(value.clone() for value in args)
    torch.testing.assert_close(
        bound(*replacements), control(*replacements), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tmem_a_partial_synthetic_k_cuda(dtype: torch.dtype) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(847)
    args = (
        torch.randn((130, 21), device=CUDA_DEVICE, dtype=dtype),
        torch.randn((21, 70), device=CUDA_DEVICE, dtype=dtype),
    )
    bound = _full_slice_dot._bind_isolated(args)
    config = _full_slice_tmem_config()
    assert "OperandSource.TMEM" in bound.to_code(config)
    bound.set_config(config)
    control = _full_slice_dot._bind_isolated(args)
    control.set_config(
        helion.Config.from_dict(config.config | {"cute_collective_tmem_a": False})
    )
    replay = _make_cudagraph_replay(lambda: bound(*args))
    for scale in (1.0, -0.5, 0.75):
        args[0].mul_(scale)
        expected = torch.relu((args[0] * 0.5).float() @ args[1].float()).to(dtype)
        torch.testing.assert_close(control(*args), expected, rtol=0.02, atol=0.02)
        torch.testing.assert_close(bound(*args), control(*args), rtol=0, atol=0)
        torch.testing.assert_close(replay(), control(*args), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "bm,bn,bk,recipe",
    [
        (64, 32, 16, "scalar"),
        (64, 64, 32, "vector_unrolled"),
        (128, 64, 128, "vector_unrolled"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tmem_a_runtime_k_tails_unaligned_and_empty_cuda(
    bm: int, bn: int, bk: int, recipe: str, dtype: torch.dtype
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(867)
    a = torch.randn((130, 88), device=CUDA_DEVICE, dtype=dtype)[:, 1:79]
    b = torch.randn((70, 88), device=CUDA_DEVICE, dtype=dtype)[:, 1:79]
    seed = torch.randn((130, 70), device=CUDA_DEVICE, dtype=torch.float32)
    config = helion.Config.from_dict(
        _seeded_config(bm, bn, bk).config
        | {"cute_collective_recipe": recipe, "cute_collective_tmem_a": True}
    )
    for limit in (78, 0, 33, 78):
        args = (a, b, seed, limit)
        bound = _seeded_contraction._bind_isolated(args)
        assert "OperandSource.TMEM" in bound.to_code(config)
        bound.set_config(config)
        control = _seeded_contraction._bind_isolated(args)
        control.set_config(
            helion.Config.from_dict(config.config | {"cute_collective_tmem_a": False})
        )
        replay = _make_cudagraph_replay(partial(bound, *args))
        for factor in (1.0, -0.5, 0.75):
            a.mul_(factor)
            seed.mul_(0.5)
            expected = seed + (a[:, :limit] * 0.5).float() @ b[:, :limit].float().T
            torch.testing.assert_close(control(*args), expected, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(bound(*args), control(*args), rtol=0, atol=0)
            torch.testing.assert_close(replay(), control(*args), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bm,bn", [(64, 32), (128, 64)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tmem_a_dynamic_empty_seed_bridge_cuda(
    bm: int, bn: int, dtype: torch.dtype
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    args = _dependent_inputs(dtype, "cuda")
    a, b, scale, lengths = args
    for value in (a, b, scale):
        value.copy_(torch.randint(-4, 5, value.shape, device=CUDA_DEVICE) * 0.25)
    lengths.fill_(78)
    config = helion.Config.from_dict(
        _dependent_config(bm, bn, True).config | {"cute_collective_tmem_a": True}
    )
    bound = _dependent_contractions._bind_isolated(args)
    assert "OperandSource.TMEM" in bound.to_code(config)
    assert "collective_tmem_seed_values" in bound.to_code(config)
    bound.set_config(config)
    control = _dependent_contractions._bind_isolated(args)
    control.set_config(_dependent_config(bm, bn, False))
    replay = _make_cudagraph_replay(lambda: bound(*args))
    for first, second in ((78, 78), (0, 78), (78, 0), (0, 0), (17, 37), (78, 78)):
        lengths.copy_(
            torch.tensor([first, second], device=CUDA_DEVICE, dtype=torch.int32)
        )
        a.mul_(-0.5)
        expected = control(*args)
        torch.testing.assert_close(bound(*args), expected, rtol=0, atol=0)
        torch.testing.assert_close(replay(), expected, rtol=0, atol=0)
