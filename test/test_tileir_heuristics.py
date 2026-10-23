from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _forbid_cuda_work
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs
from helion._compiler.autotuner_heuristics import get_heuristics
from helion._compiler.autotuner_heuristics.tileir import TileIRChunkReductionHeuristic
from helion._compiler.autotuner_heuristics.tileir import TileIRMatmulSeedHeuristic
from helion._compiler.autotuner_heuristics.tileir import TileIRRowReductionHeuristic
from helion._compiler.autotuner_heuristics.tileir import TileIRSmallReductionHeuristic
from helion._compiler.backend import TileIRBackend
from helion._testing import skipIfRefEager
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import MatmulFact
from helion.autotuner.config_spec import ReductionCategory
from helion.autotuner.config_spec import ReductionDescriptor
from helion.autotuner.config_spec import ReductionKernelFact
from helion.autotuner.config_spec import ReductionLoopSpec
from helion.autotuner.config_spec import TensorNumelConstraint
from helion.autotuner.llm.prompting import build_compiler_analysis_section
from helion.autotuner.llm_search import LLMGuidedSearch
import helion.language as hl
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("ENABLE_TILE", "1")
    with _mock_cuda_unavailable(), _forbid_cuda_work():
        yield


def _spec() -> ConfigSpec:
    spec = ConfigSpec(backend=TileIRBackend(), target_device_capability=(10, 3))
    spec.indexing.length = 2
    return spec


def _env(spec: ConfigSpec) -> SimpleNamespace:
    return SimpleNamespace(
        backend_name="tileir",
        config_spec=spec,
        settings=Settings(backend="tileir"),
        device=torch.device("cpu"),
    )


def _matmul_spec(dtype: torch.dtype = torch.bfloat16) -> ConfigSpec:
    spec = _spec()
    # Registered M/N axes can have unknown logical extents; the K axis's
    # nonconsecutive block id also catches accidental config-position mapping.
    for block_id in (0, 1, 4):
        spec.block_sizes.append(
            BlockSizeSpec(block_id=block_id, size_hint=2048, max_size=256)
        )
    spec.matmul_facts.append(MatmulFact(2, 2, 0, 1, 4, None, None, 2048, dtype, dtype))
    return spec


def _reduction_spec(
    category: ReductionCategory = ReductionCategory.FULL_SLICE,
    width: int = 7168,
    *,
    row: bool = True,
) -> ConfigSpec:
    spec = _spec()
    spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=1024))
    reduction_id = 1 if row else 0
    if row:
        spec.grid_block_ids.append(0)
        spec.reduction_loops.append(ReductionLoopSpec(block_id=1, size_hint=width))
    else:
        spec.grid_block_ids.extend((1, 2))
    spec.reduction_kernel_fact = ReductionKernelFact(
        (ReductionDescriptor(category, reduction_id, 0, width, 4),),
        grid_axis_block_ids=tuple(spec.grid_block_ids),
    )
    return spec


def test_tileir_seeds_are_candidates_in_their_own_backend() -> None:
    expected = (
        TileIRMatmulSeedHeuristic,
        TileIRRowReductionHeuristic,
        TileIRSmallReductionHeuristic,
        TileIRChunkReductionHeuristic,
    )
    assert get_heuristics("tileir") == expected
    assert all(h not in get_heuristics("triton") for h in expected)
    assert all(not h.promote_seed_to_default for h in expected)
    spec = _matmul_spec()
    env = _env(spec)
    seeds = compiler_seed_configs(env, SimpleNamespace())
    assert len(seeds) == 1
    assert spec.compiler_default_config is None
    assert spec.autotuner_heuristics == ["tileir_matmul"]
    env.settings.disable_autotuner_heuristics = True
    assert compiler_seed_configs(env, SimpleNamespace()) == []
    assert spec.autotuner_heuristics == []


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("ceiling", (64, 256))
def test_matmul_seed_honors_live_axis_bounds(dtype: torch.dtype, ceiling: int) -> None:
    spec = _matmul_spec(dtype)
    for block in spec.block_sizes:
        block.max_size = ceiling
    env = _env(spec)
    assert TileIRMatmulSeedHeuristic.is_eligible(env, SimpleNamespace())
    seed = TileIRMatmulSeedHeuristic.get_seed_config(env, SimpleNamespace())
    assert seed is not None
    assert seed.block_sizes == [min(128, ceiling)] * 3
    assert seed.num_stages == 3
    assert seed.num_warps == 4
    assert seed.config["num_ctas"] == seed.config["occupancy"] == 1
    assert seed.indexing == ["pointer", "pointer"]
    generation = spec.create_config_generation()
    assert generation.unflatten(generation.flatten(seed)) == seed


@pytest.mark.parametrize("dtype", (torch.float32, torch.float8_e4m3fn, torch.int8))
def test_matmul_seed_declines_other_dot_dtypes(dtype: torch.dtype) -> None:
    assert not TileIRMatmulSeedHeuristic.is_eligible(
        _env(_matmul_spec(dtype)), SimpleNamespace()
    )


def test_matmul_seed_declines_unmodeled_axes() -> None:
    spec = _matmul_spec()
    spec.block_sizes.append(BlockSizeSpec(block_id=5, size_hint=32))
    assert not TileIRMatmulSeedHeuristic.is_eligible(_env(spec), SimpleNamespace())
    spec = _matmul_spec()
    spec.matmul_facts[0] = spec.matmul_facts[0]._replace(n_block_id=0)
    assert not TileIRMatmulSeedHeuristic.is_eligible(_env(spec), SimpleNamespace())


def test_seed_shrinks_for_tensor_numel_constraints() -> None:
    spec = _matmul_spec()
    spec.tensor_numel_constraints.append(
        TensorNumelConstraint(lambda m, n: m * n <= 4096, (0, 1), "m*n <= 4096")
    )
    seed = TileIRMatmulSeedHeuristic.get_seed_config(_env(spec), SimpleNamespace())
    assert seed is not None
    assert seed.block_sizes[0] * seed.block_sizes[1] <= 4096
    for block in spec.block_sizes:
        block.autotuner_min = 128
    assert (
        TileIRMatmulSeedHeuristic.get_seed_config(_env(spec), SimpleNamespace()) is None
    )


def test_seed_declines_unsatisfiable_constraint_at_declared_minimum() -> None:
    spec = _matmul_spec()
    for block in spec.block_sizes:
        block.min_size = block.autotuner_min = block.max_size = 128
    spec.tensor_numel_constraints.append(
        TensorNumelConstraint(lambda m, n: m * n <= 4096, (0, 1), "m*n <= 4096")
    )
    assert (
        TileIRMatmulSeedHeuristic.get_seed_config(_env(spec), SimpleNamespace()) is None
    )


@pytest.mark.parametrize(
    "width, reductions",
    ((1024, [[None]]), (7168, [[None], [4096]]), (32768, [[4096]])),
)
def test_row_reduction_covers_resident_and_streamed_candidates(
    width: int, reductions: list[list[int | None]]
) -> None:
    spec = _reduction_spec(width=width)
    env = _env(spec)
    assert TileIRRowReductionHeuristic.is_eligible(env, SimpleNamespace())
    seeds = TileIRRowReductionHeuristic.get_seed_configs(env, SimpleNamespace())
    assert [s.reduction_loops for s in seeds] == reductions
    assert all(s.block_sizes == [1] for s in seeds)
    assert (
        TileIRRowReductionHeuristic.get_seed_config(env, SimpleNamespace()) == seeds[0]
    )
    generation = spec.create_config_generation()
    assert all(generation.unflatten(generation.flatten(s)) == s for s in seeds)


def test_row_reduction_respects_required_persistence_and_row_floor() -> None:
    spec = _reduction_spec()
    spec.allowed_pid_types = ("persistent_blocked",)
    seeds = TileIRRowReductionHeuristic.get_seed_configs(_env(spec), SimpleNamespace())
    assert all(s.pid_type == "persistent_blocked" for s in seeds)
    spec.block_sizes[0].autotuner_min = 2
    assert not TileIRRowReductionHeuristic.is_eligible(_env(spec), SimpleNamespace())


def test_small_reduction_batches_independent_groups() -> None:
    spec = _reduction_spec(width=32)
    spec.reduction_loops.clear()
    assert TileIRSmallReductionHeuristic.is_eligible(_env(spec), SimpleNamespace())
    seed = TileIRSmallReductionHeuristic.get_seed_config(_env(spec), SimpleNamespace())
    assert seed is not None
    assert seed.block_sizes == [128]
    assert seed.num_stages == 2
    spec.reduction_kernel_fact = spec.reduction_kernel_fact._replace(
        reductions=(ReductionDescriptor(ReductionCategory.FULL_SLICE, 1, 0, 256, 4),)
    )
    assert not TileIRSmallReductionHeuristic.is_eligible(_env(spec), SimpleNamespace())


@pytest.mark.parametrize(
    "category", (ReductionCategory.DECLINED, ReductionCategory.USER_TILE)
)
@pytest.mark.parametrize(
    "bounds, expected", (((256, 8192), 1024), ((2048, 8192), 2048), ((256, 512), 512))
)
def test_chunk_reduction_uses_declared_bounds(
    category: ReductionCategory, bounds: tuple[int, int], expected: int
) -> None:
    spec = _reduction_spec(category, width=0, row=False)
    block = spec.block_sizes[0]
    block.min_size, block.max_size = bounds
    block.autotuner_min = bounds[0]
    env = _env(spec)
    assert TileIRChunkReductionHeuristic.is_eligible(env, SimpleNamespace())
    seed = TileIRChunkReductionHeuristic.get_seed_config(env, SimpleNamespace())
    assert seed is not None
    assert seed.block_sizes == [expected]
    assert seed.num_stages == 2


def test_chunk_reduction_declines_unrelated_reduction_axes() -> None:
    spec = _reduction_spec(ReductionCategory.DECLINED, width=0, row=False)
    spec.reduction_kernel_fact = spec.reduction_kernel_fact._replace(
        reductions=(
            *spec.reduction_kernel_fact.reductions,
            ReductionDescriptor(ReductionCategory.DECLINED, 3, 0, 0),
        )
    )
    assert not TileIRChunkReductionHeuristic.is_eligible(_env(spec), SimpleNamespace())


def test_llm_round_zero_and_prompt_receive_compiler_seeds() -> None:
    spec = _reduction_spec()
    spec.compiler_seed_configs = compiler_seed_configs(_env(spec), SimpleNamespace())
    search = LLMGuidedSearch.__new__(LLMGuidedSearch)
    search.config_spec = spec
    search.config_gen = spec.create_config_generation()
    search.initial_random_configs = 3
    round_zero = search._build_seed_configs()
    assert all(s in round_zero for s in spec.compiler_seed_configs)
    prompt = build_compiler_analysis_section(spec)
    assert "tileir_row_reduction" in prompt
    assert '"reduction_loops":[null]' in prompt
    assert '"reduction_loops":[4096]' in prompt


@skipIfRefEager("requires compiler IR")
def test_live_cpu_binding_recognizes_registered_matmul_tiles() -> None:
    @helion.kernel(backend="tileir", static_shapes=True)
    def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        n, _ = y.shape
        bm = hl.register_block_size(16, 256)
        bn = hl.register_block_size(32, 256)
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
        for tm, tn in hl.grid([(m + bm - 1) // bm, (n + bn - 1) // bn]):
            rows = tm * bm + hl.arange(bm)
            cols = tn * bn + hl.arange(bn)
            acc = hl.zeros([bm, bn], dtype=torch.float32)
            for kk in hl.tile(k):
                a = hl.load(x, [rows, kk.index], extra_mask=(rows < m)[:, None])
                b = hl.load(y, [cols, kk.index], extra_mask=(cols < n)[:, None])
                acc = hl.dot(a, b.T, acc=acc)
            hl.store(
                out,
                [rows, cols],
                acc.to(x.dtype),
                extra_mask=(rows < m)[:, None] & (cols < n)[None, :],
            )
        return out

    bound = _cpu_bind(
        matmul,
        (
            torch.empty((256, 512), dtype=torch.bfloat16),
            torch.empty((512, 512), dtype=torch.bfloat16),
        ),
    )
    spec = bound.config_spec
    assert spec.matmul_facts[0].static_m is None
    assert spec.matmul_facts[0].static_n is None
    assert spec.autotuner_heuristics == ["tileir_matmul"]
    assert spec.compiler_seed_configs[0].block_sizes == [128, 128, 128]


@skipIfRefEager("requires compiler IR")
def test_live_cpu_binding_recognizes_declined_but_bounded_chunk() -> None:
    @helion.kernel(backend="tileir", static_shapes=True)
    def chunk_max(x: torch.Tensor) -> torch.Tensor:
        rows, cols = x.shape
        chunk = hl.register_block_size(256, 8192)
        parts = (cols + chunk - 1) // chunk
        out = torch.empty((rows, parts), dtype=x.dtype, device=x.device)
        flat = x.view(-1)
        for row, part in hl.grid([rows, parts]):
            column = part * chunk + hl.arange(chunk)
            value = hl.load(flat, [row * cols + column], extra_mask=column < cols)
            out[row, part] = value.amax()
        return out

    bound = _cpu_bind(chunk_max, (torch.empty((64, 32000)),))
    spec = bound.config_spec
    assert (
        spec.reduction_kernel_fact.reductions[0].category == ReductionCategory.DECLINED
    )
    assert spec.autotuner_heuristics == ["tileir_chunk_reduction"]
    assert spec.compiler_seed_configs[0].block_sizes == [1024]


@skipIfRefEager("requires compiler IR")
def test_live_cpu_binding_recognizes_fixed_width_group_reduction() -> None:
    @helion.kernel(backend="tileir", static_shapes=True)
    def group_norm(x: torch.Tensor) -> torch.Tensor:
        groups, _ = x.shape
        block_groups = hl.register_block_size(1, 1024)
        out = torch.empty_like(x)
        for group in hl.tile(groups, block_size=block_groups):
            column = hl.arange(32)
            value = x[group, column].float()
            maximum = value.abs().amax(-1)
            out[group, column] = (value / maximum[:, None]).to(x.dtype)
        return out

    bound = _cpu_bind(group_norm, (torch.empty((4096, 32), dtype=torch.bfloat16),))
    spec = bound.config_spec
    assert (
        spec.reduction_kernel_fact.reductions[0].category
        == ReductionCategory.FULL_SLICE
    )
    assert not spec.reduction_loops
    assert spec.autotuner_heuristics == ["tileir_small_reduction"]
    assert spec.compiler_seed_configs[0].block_sizes == [128]


@skipIfRefEager("requires compiler IR")
def test_live_cpu_binding_recognizes_canonical_row_reduction() -> None:
    @helion.kernel(backend="tileir", static_shapes=True)
    def row_norm(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for row in hl.tile(x.shape[0]):
            value = x[row, :].float()
            inv = torch.rsqrt((value * value).sum(-1) / x.shape[1] + 1e-5)
            out[row, :] = (value * inv[:, None]).to(x.dtype)
        return out

    bound = _cpu_bind(row_norm, (torch.empty((16, 7168), dtype=torch.float16),))
    spec = bound.config_spec
    assert spec.autotuner_heuristics == ["tileir_row_reduction"]
    seeds = spec.compiler_seed_configs
    assert [s.block_sizes for s in seeds] == [[1], [1]]
    assert [s.reduction_loops for s in seeds] == [[None], [4096]]
