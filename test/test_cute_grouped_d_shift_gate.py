"""The autotuner does not propose the bounded D shift; a config that names it works.

``grouped_d_shift.d_shift_searched`` says why.  The proposals behind the gate
(the shifted seeds, the M,N worklist tiles on the shift, the paired prefix
recipes) keep their own tests, which bind with the gate lifted.  What two of
them render of the lifted gate's palette, the M,N tiles of
test_cute_grouped_mn_device_seeds.py and the source-128 branches of
test_cute_grouped_source128.py, is rendered here as a search gets it,
without the shift, one seed per codegen branch.
"""

from __future__ import annotations

import ast
import concurrent.futures
import json
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from unittest.mock import patch

from packaging.version import Version
import pytest
import torch

from test._cute_grouped_gemm_split_sizes import _device_offsets_kernel
from test._cute_grouped_gemm_split_sizes import _device_split_sizes_kernel

import helion
from helion._compiler.cute.cutedsl_compat import _cute_backend_requirement_error
from helion._compiler.cute.cutedsl_compat import _CuteDSLVersionProbe
from helion._compiler.cute.cutedsl_compat import tcgen05_runtime_n_ptx_compatible
from helion._compiler.cute.grouped_d_shift import d_shift_searched
from helion._compiler.cute.tcgen05_config import surface
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_codegen import tcgen05_target
from helion._testing.cute_d_shift_gate import d_shift_proposals
from helion._testing.cute_dense_tcgen05 import wrapper_plans
from helion._testing.cute_flash import seeds_per_codegen_branch
from helion._testing.cute_grouped_configs import prefix_recipe_args as _offsets
from helion._testing.cute_grouped_offsets import fresh_grouped_offsets_k64 as _kernel
from helion._testing.cute_grouped_offsets import grouped_offsets_k64
from helion._testing.cute_grouped_offsets import observed_logical_k
from helion._testing.cute_population_contracts import checked_initial_population
from helion.autotuner.effort_profile import get_effort_profile
from helion.autotuner.llm.configs import parse_response_configs
from helion.autotuner.llm_search import LLMGuidedSearch
from helion.autotuner.local_cache import LocalAutotuneCache
from helion.autotuner.surrogate_pattern_search import LFBOTreeSearch
from helion.exc import InvalidConfig
from helion.runtime.settings import default_autotuner_fn

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterator
    from pathlib import Path

    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only

KEY = "tcgen05_grouped_d_shift"
ORIENTATION = "tcgen05_grouped_worklist_orientation"
PREFIX = "tcgen05_grouped_prefix_adaptive_n"
BALANCED = "tcgen05_grouped_balanced_workers"
PANEL = ("tcgen05_grouped_l2_swizzle_alternate", "tcgen05_grouped_l2_swizzle_mn_ratio")
MMA_N = "tcgen05_grouped_worklist_mma_n"
SOURCE_M = "tcgen05_grouped_worklist_source_m_tile"
CLUSTER_N = "tcgen05_cluster_n"
SWIZZLE = "tcgen05_l2_swizzle_size"
# Legal under the M,N orientation only.
MN_ONLY = (BALANCED, *PANEL)
# The two options a search holds only with the gate lifted, spelled out: a
# check that reads them from the hook's own list
# (``surface.D_SHIFT_PROPOSAL_KEYS``) passes with an option missing from it.
GATED = (KEY, PREFIX)


def _sizes(rows: int, columns: int, reduction: int):
    return (
        torch.empty((rows, reduction), dtype=torch.bfloat16),
        torch.empty((8, columns, reduction), dtype=torch.bfloat16),
        torch.full((8,), rows // 8, dtype=torch.int32),
    )


# One kernel per way the knob is offered: device offsets with a bf16 output
# (the paired prefix recipes exist for it), an fp32 output, device split sizes.
KERNELS: dict[str, tuple[Callable[[], object], tuple[torch.Tensor, ...]]] = {
    "offsets": (_kernel, _offsets(groups=16, m=2048, n=1536, k=2048)),
    "fp32": (lambda: observed_logical_k, _offsets(groups=4, m=2048, n=512, k=512)),
    "sizes": (lambda: _device_split_sizes_kernel, _sizes(2048, 512, 256)),
}


def _bind(name: str, *, proposed: bool = False) -> BoundKernel:
    kernel, args = KERNELS[name]
    if not proposed:
        return cpu_bind(kernel(), args)
    with d_shift_proposals():
        return cpu_bind(kernel(), args)


def _without_shift(config: helion.Config) -> helion.Config:
    return helion.Config.from_dict(
        {key: value for key, value in config.config.items() if key != KEY}
    )


def _holds(config: helion.Config, expected: helion.Config) -> bool:
    """Whether ``config`` has every option of ``expected`` at its value. A
    config that a search builds spells out defaults a normalized one omits."""
    return all(config.get(key) == value for key, value in expected.config.items())


def _explicit(bound: BoundKernel) -> list[helion.Config]:
    """One shifted config per family of the lifted gate's palette."""
    seeds = bound.config_spec.compiler_seed_configs
    picks = [
        next(
            seed
            for seed in seeds
            if seed.get(KEY) is True and seed.get(ORIENTATION, "nm") == "nm"
        ),
        next(
            seed
            for seed in seeds
            if seed.get(KEY) is True
            and seed.get(ORIENTATION) == "mn"
            and seed.get(BALANCED) is not True
            and seed.get(PREFIX) is not True
        ),
        next(
            seed
            for seed in seeds
            if seed.get(KEY) is True
            and seed.get(BALANCED) is True
            and seed.get(PREFIX) is not True
        ),
    ]
    # An M,N seed with the adaptive panel widths, where the kernel has them.
    picks.extend(
        [
            seed
            for seed in seeds
            if seed.get(PANEL[0]) is not None and seed.get(PREFIX) is not True
        ][:1]
    )
    picks.extend([seed for seed in seeds if seed.get(PREFIX) is True][:1])
    return picks


def test_the_gate_is_closed() -> None:
    assert d_shift_searched() is False
    with d_shift_proposals():
        assert d_shift_searched() is True
    assert d_shift_searched() is False


@pytest.mark.parametrize("name", list(KERNELS))
def test_a_search_gets_no_seed_that_stores_through_the_shift(name: str) -> None:
    with forbid_cuda_work(), tcgen05_target(), mock_cuda_unavailable():
        bound = _bind(name)
        proposed = _bind(name, proposed=True)
    model = bound.config_spec._cute_tcgen05_config
    # The kernel still offers the shift; only the proposals are gone.
    assert model.grouped_d_shift_supported
    assert not model.grouped_d_shift_search_enabled
    seeds = bound.config_spec.compiler_seed_configs
    assert seeds
    assert not any(seed.get(KEY) for seed in seeds)
    assert not any(seed.get(PREFIX) for seed in seeds)

    other = proposed.config_spec.compiler_seed_configs
    assert proposed.config_spec._cute_tcgen05_config.grouped_d_shift_search_enabled
    shifted = [seed for seed in other if seed.get(KEY) is True]
    assert shifted
    # The gate changes no seed that does not carry the shift: those of the
    # lifted gate's palette lead this one, in their order, and the compiler
    # default is the same config. The shifted entries follow every other
    # seed, so a search that proposes them starts from the same leading
    # seeds.
    kept = [seed for seed in other if seed.get(KEY) is not True]
    assert all(seed.get(ORIENTATION, "nm") == "nm" for seed in kept)
    assert seeds[: len(kept)] == kept
    assert other == [*kept, *shifted]
    # Behind them come the M,N tiles: each is the lifted gate's tile without
    # the shift, on the same stage count. The shifted copies of N,M seeds,
    # whose originals lead the palette, and the paired prefix recipes, which
    # are legal only on the shift, have no counterpart.
    tiles = seeds[len(kept) :]
    assert tiles
    assert tiles == [
        _without_shift(seed)
        for seed in shifted
        if seed.get(ORIENTATION) == "mn" and not seed.get(PREFIX)
    ]
    assert (
        bound.config_spec.compiler_default_config
        == proposed.config_spec.compiler_default_config
    )
    assert not bound.config_spec.compiler_default_config.get(KEY)


@pytest.mark.parametrize("name", list(KERNELS))
def test_the_knob_is_validated_but_not_a_search_coordinate(name: str) -> None:
    with forbid_cuda_work(), tcgen05_target(), mock_cuda_unavailable():
        bound = _bind(name)
        proposed = _bind(name, proposed=True)
        model = bound.config_spec._cute_tcgen05_config
        # Normalization reads the same fragments with the gate closed or
        # lifted: a config that names the knob is validated as before.
        other = proposed.config_spec._cute_tcgen05_config
        for for_search in (False, True):
            fragments = surface.optional_fragments(model, for_search=for_search)
            assert KEY in fragments
            assert set(fragments) == set(
                surface.optional_fragments(other, for_search=for_search)
            )
        assert {ORIENTATION, *MN_ONLY} <= set(surface.optional_fragments(model))
        fields = bound.config_spec.create_config_generation()._flat_fields()
        assert KEY not in fields
        assert PREFIX not in fields
        lifted = proposed.config_spec.create_config_generation()._flat_fields()
        assert KEY in lifted
        if name == "offsets":
            assert PREFIX in lifted
        # Nothing else leaves the surface. The M,N tiles stay seeds, so the
        # M,N orientation, its MMA width and the options that are legal
        # under it alone stay coordinates, each with the choices the lifted
        # gate gives it.
        assert set(lifted) - set(fields) == set(GATED) & set(lifted)
        assert set(fields) <= set(lifted)
        offered = set(surface.optional_fragments(other, for_search=True))
        searched = {ORIENTATION, MMA_N, *(set(MN_ONLY) & offered)}
        assert BALANCED in searched
        if name != "sizes":
            assert set(PANEL) <= searched
        assert searched <= set(fields)
        assert fields[ORIENTATION] == lifted[ORIENTATION]
        assert fields[MMA_N] == lifted[MMA_N]


def test_a_full_search_population_holds_no_shifted_config() -> None:
    kernel, args = KERNELS["offsets"]
    with forbid_cuda_work(), tcgen05_target(), mock_cuda_unavailable():
        bound = cpu_bind(kernel(), args)
        rng_state = random.getstate()
        try:
            random.seed(9927)
            search = LFBOTreeSearch(
                bound,
                args,
                **LFBOTreeSearch.get_kwargs_from_profile(
                    get_effort_profile("full"), bound.settings
                ),
            )
            generation = search.config_gen
            configs = [
                generation.unflatten(row) for row in checked_initial_population(search)
            ]
            configs.extend(generation.random_config() for _ in range(64))
        finally:
            random.setstate(rng_state)
    assert len(configs) >= search.initial_population
    for config in configs:
        assert not config.get(KEY)
        assert not config.get(PREFIX)
    # Every M,N tile the compiler seeds is in it, without the shift, as the
    # search transfers it onto its flat fields.
    tiles = [
        seed
        for seed in bound.config_spec.compiler_seed_configs
        if seed.get(ORIENTATION) == "mn"
    ]
    assert tiles
    for tile in tiles:
        transferred = generation.unflatten(generation.flatten(tile))
        assert transferred[ORIENTATION] == "mn"
        assert transferred[MMA_N] == tile[MMA_N]
        assert transferred in configs


def _output_stores(tree: ast.AST) -> list[ast.Call]:
    """The TMA stores of the output in a rendered kernel."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cute.copy"
        and ast.unparse(node.args[0]) == "tcgen05_tma_store_atom"
    ]


def _assigned(tree: ast.AST, name: str) -> list[str]:
    return [
        ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]


def _rendered_on_the_groups_descriptor(
    bound: BoundKernel, config: helion.Config
) -> ast.Module:
    """The source of ``config``, parsed, once it is checked that no plan is
    on the shift and that every store of the output names the group's own
    descriptor."""
    source = bound.to_code(config)
    tree = ast.parse(source)
    plans = wrapper_plans(source)
    assert not any("bounded_output_shift" in plan for plan in plans)
    assert plans[0]["dynamic_d_tensormap"] is True
    stores = _output_stores(tree)
    assert stores
    for store in stores:
        (named,) = [k.value for k in store.keywords if k.arg == "tma_desc_ptr"]
        assert ast.unparse(named) == "tcgen05_grouped_d_tensormap_desc_ptr"
    return tree


def test_an_mn_tile_without_the_shift_stores_through_its_groups_descriptor() -> None:
    """Why the M,N tiles stay seeds. The shift's store goes through the
    host-encoded descriptor, based M rows below the output. Without the
    shift every store of a tile names the group's own descriptor, which the
    kernel rebuilds per group at the group's first row, clamped into the
    output, over the group's rows: the descriptor every N,M seed has."""
    _, args = KERNELS["offsets"]
    rows = args[0].shape[0]
    with forbid_cuda_work(), tcgen05_target(), mock_cuda_unavailable():
        bound = _bind("offsets")
        spec = bound.config_spec
        seeds = spec.compiler_seed_configs
        tiles = [seed for seed in seeds if seed.get(ORIENTATION) == "mn"]
        # One tile per code path of the M,N worklist, and the compiler
        # default, an N,M worklist, beside them.
        picks = [
            next(t for t in tiles if not t.get(BALANCED) and t[SOURCE_M] == 128),
            next(t for t in tiles if not t.get(BALANCED) and t[SOURCE_M] == 256),
            next(t for t in tiles if t.get(BALANCED) is True),
            next(t for t in tiles if t.get(PANEL[0]) is not None),
            seeds[0],
        ]
        assert picks[-1].get(ORIENTATION, "nm") == "nm"
        for pick in picks:
            config = spec.normalized_config(pick)
            assert not config.get(KEY)
            tree = _rendered_on_the_groups_descriptor(bound, config)
            # The group's descriptor: based on its first row, over its rows.
            base = (
                "out.iterator + cutlass.Int32(tcgen05_grouped_global_m_start)"
                " * cutlass.Int32(out.layout.stride[0])"
            )
            assert _assigned(tree, "tcgen05_grouped_d_tensormap_base") == [base]
            start = (
                "min(max(tcgen05_grouped_raw_start, cutlass.Int32(0)), "
                f"cutlass.Int32({rows}))"
            )
            assert _assigned(tree, "tcgen05_grouped_visible_start") == [start]

        # The same tile with the shift, which a config may still name: its
        # stores name no descriptor and go through the host-encoded one.
        shifted = dict(spec.normalized_config(picks[0]).config)
        shifted[KEY] = True
        source = bound.to_code(helion.Config.from_dict(shifted))
        assert wrapper_plans(source)[0]["bounded_output_shift"] is True
        tree = ast.parse(source)
        stores = _output_stores(tree)
        assert stores
        assert not any(k.arg == "tma_desc_ptr" for s in stores for k in s.keywords)
        assert not _assigned(tree, "tcgen05_grouped_d_tensormap_base")


# The three shapes of test_cute_grouped_mn_device_seeds.py, which renders
# these tiles on the shift.
@pytest.mark.parametrize(
    "m,groups,n,k", [(2048, 16, 1536, 2048), (8192, 16, 1536, 2048), (512, 4, 512, 256)]
)
def test_the_mn_tiles_a_search_gets_normalize_and_render(
    m: int, groups: int, n: int, k: int
) -> None:
    """The M,N tiles of the palette a search gets, the ones it can hand a
    user: each normalizes without the shift, and each codegen branch reaches
    native source that stores through the group's descriptor."""
    with forbid_cuda_work(), tcgen05_target(), mock_cuda_unavailable():
        bound = cpu_bind(_kernel(), _offsets(groups=groups, m=m, n=n, k=k))
        spec = bound.config_spec
        tiles = [
            seed for seed in spec.compiler_seed_configs if seed.get(ORIENTATION) == "mn"
        ]
        assert {tile[SOURCE_M] for tile in tiles} == {128, 256}
        assert {tile[MMA_N] for tile in tiles} == {128, 192, 224, 256}
        assert {tile[CLUSTER_N] for tile in tiles} == {1, 2}
        assert {tile.get(BALANCED) for tile in tiles} == {None, True}
        for tile in tiles:
            normalized = spec.normalized_config(tile)
            assert normalized[ORIENTATION] == "mn"
            assert not set(GATED) & set(normalized.config)

        def geometry_and_schedule(tile: helion.Config) -> tuple[object, ...]:
            return tile[SOURCE_M], tile[MMA_N], tile[CLUSTER_N], tile.get(BALANCED)

        def raster(tile: helion.Config) -> tuple[object, ...]:
            return tile.get(SWIZZLE), *(tile.get(key) for key in PANEL)

        # Every tile is normalized above. Rendered are the first tile of each
        # worklist geometry and worker schedule, which decide the lines a
        # tile renders differently without the shift (its rewritten
        # descriptors), and the last tile of each panel raster (L2 swizzle
        # size, alternation, M/N ratio). A geometry is not rendered once per
        # raster: the raster's lines, which pick a group's next tile, are
        # the same with the shift and without it, and
        # test_cute_grouped_mn_device_seeds.py renders them per geometry on
        # the shift, in the slow tier. The pipeline depths only resize the
        # same source.
        for tile in seeds_per_codegen_branch(
            tiles, geometry_and_schedule, last_per=raster
        ):
            _rendered_on_the_groups_descriptor(bound, spec.normalized_config(tile))


@pytest.fixture
def isolated_cute_requirement_cache() -> Iterator[None]:
    """The backend's requirement check keeps its answer for the process
    (``_cute_backend_requirement_error``): cleared on both sides of a test
    that patches the DSL version, as test_cute_grouped_source128.py does, so
    that the answer read under the patch neither comes from another test nor
    outlives this one."""
    _cute_backend_requirement_error.cache_clear()
    try:
        yield
    finally:
        _cute_backend_requirement_error.cache_clear()


@pytest.mark.parametrize("kind", ["offsets", "split_sizes"])
@pytest.mark.usefixtures("isolated_cute_requirement_cache")
def test_the_source128_seeds_a_search_gets_render_on_a_newer_dsl(kind: str) -> None:
    """The typed device kernels of test_cute_grouped_source128.py on a CuTe
    DSL newer than the one the runtime-N PTX is validated on, bound as a
    search binds them. That module renders the branches of the lifted gate's
    palette; these are the branches of the palette a search gets."""
    metadata = torch.full((8,), 256, dtype=torch.int64)
    kernel = _device_split_sizes_kernel
    if kind == "offsets":
        metadata = torch.arange(9, dtype=torch.int64) * 256
        kernel = _device_offsets_kernel
    args = (
        torch.empty((2048, 256), dtype=torch.bfloat16),
        torch.empty((8, 256, 256), dtype=torch.bfloat16),
        metadata,
    )
    with (
        forbid_cuda_work(),
        tcgen05_target(),
        mock_cuda_unavailable(),
        patch(
            "helion._compiler.cute.cutedsl_compat._installed_cute_dsl_version",
            return_value=_CuteDSLVersionProbe(version=Version("4.8.2")),
        ),
    ):
        assert not tcgen05_runtime_n_ptx_compatible()
        bound = cpu_bind(kernel, args)
        seeds = bound.config_spec.compiler_seed_configs
        assert not any(seed.get(KEY) for seed in seeds)
        source128 = [seed for seed in seeds if seed.get(SOURCE_M) == 128]

        # The branches test_cute_grouped_source128.py renders by, less the
        # shift, which no seed here carries: the CTA group, the cluster N,
        # the orientation and the MMA N.
        def branch(seed: helion.Config) -> tuple[object, ...]:
            return (
                seed["tcgen05_cluster_m"],
                seed.get(CLUSTER_N),
                seed.get(ORIENTATION),
                seed.get(MMA_N),
            )

        rendered = seeds_per_codegen_branch(source128, branch)
        # The N,M seeds of both CTA groups, and the M,N tiles at each MMA N
        # and at cluster N 2, which the lifted gate has on the shift.
        assert {branch(seed) for seed in rendered} == {
            (1, 1, None, None),
            (2, 1, None, None),
            (2, 1, "mn", 128),
            (2, 2, "mn", 128),
            (2, 1, "mn", 192),
            (2, 1, "mn", 224),
            (2, 1, "mn", 256),
        }
        for seed in rendered:
            _rendered_on_the_groups_descriptor(bound, seed)


@pytest.mark.parametrize("name", list(KERNELS))
def test_an_explicit_config_keeps_the_shift_and_its_code(name: str) -> None:
    with forbid_cuda_work(), tcgen05_target(), mock_cuda_unavailable():
        proposed = _bind(name, proposed=True)
        bound = _bind(name)
        picks = _explicit(proposed)
        # Device split sizes have no adaptive panel widths, and only the
        # bf16 offsets kernel has the prefix recipes.
        assert len(picks) == {"offsets": 5, "fp32": 4, "sizes": 3}[name]
        assert any(pick.get(BALANCED) is True for pick in picks)
        for index, pick in enumerate(picks):
            # What a user saved: the config as a search once reported it.
            saved = helion.Config.from_json(
                proposed.config_spec.normalized_config(pick).to_json()
            )
            assert saved[KEY] is True
            config = bound.config_spec.normalized_config(saved)
            assert config == saved
            source = bound.to_code(config)
            assert "'bounded_output_shift': True" in source
            # The gate changes what a search proposes, not what a config means.
            assert source == proposed.to_code(saved)
            unshifted = _without_shift(saved)
            if saved.get(PREFIX) is True:
                # The paired recipes exist on the shift only.
                with pytest.raises(InvalidConfig):
                    bound.config_spec.normalized_config(unshifted)
            elif index == 0:
                assert "bounded_output_shift" not in bound.to_code(
                    bound.config_spec.normalized_config(unshifted)
                )


def test_a_cached_result_with_the_shift_is_used_as_saved(tmp_path: Path) -> None:
    _, args = KERNELS["offsets"]
    # A kernel that searches: the cache is the one a search reads and writes.
    kernel = helion.kernel(
        grouped_offsets_k64.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="quick",
    )
    with forbid_cuda_work(), tcgen05_target(), mock_cuda_unavailable():
        proposed = _bind("offsets", proposed=True)
        saved = proposed.config_spec.normalized_config(
            next(
                seed
                for seed in proposed.config_spec.compiler_seed_configs
                if seed.get(PREFIX) is True
            )
        )
        assert saved[KEY] is True
        expected = proposed.to_code(saved)

        bound = cpu_bind(kernel, args)
        entry = tmp_path / "entry.best_config"
        with (
            patch.object(LocalAutotuneCache, "_generate_key"),
            patch.object(LocalAutotuneCache, "_get_local_cache_path", lambda _: entry),
        ):
            with bound.env:
                cache = default_autotuner_fn(bound, args)
            assert isinstance(cache, LocalAutotuneCache)
            # The entry an earlier version's search wrote.
            cache.put(saved)
            assert json.loads(entry.read_text())["config"]
            with patch.object(
                cache.autotuner,
                "autotune",
                side_effect=AssertionError("a cache hit does not search"),
            ):
                hit = cache.autotune()
        assert hit == saved
        bound.set_config(hit)
        assert bound.to_code(hit) == expected

        # A search that takes the saved config as a starting point drops the
        # shift and the recipes' knob, the coordinates it does not have, and
        # keeps the recipe's M,N tile: no candidate of it stores through the
        # shift.
        generation = bound.config_spec.create_config_generation()
        _, transferred = generation.canonicalize_flat(generation.flatten(saved))
        assert not set(GATED) & set(transferred.config)
        assert transferred[ORIENTATION] == "mn"
        assert transferred[MMA_N] == saved[MMA_N]
        assert "bounded_output_shift" not in bound.to_code(transferred)


def _reply(search: LLMGuidedSearch, proposal: helion.Config) -> str:
    """An LLM's reply that proposes ``proposal``, by the options in which
    it differs from the reference config the reply's keys are merged over."""
    reference = search._default_config_dict
    named = {
        key: value
        for key, value in proposal.config.items()
        if reference.get(key) != value
    }
    return json.dumps({"configs": [named]})


def _parsed(search: LLMGuidedSearch, reply: str) -> list[helion.Config]:
    """The configs of ``reply`` as the parser reads them, ahead of the
    search's admission: what ``_parse_configs`` hands ``_admit_proposals``."""
    return parse_response_configs(
        reply,
        config_spec=search.config_spec,
        default_config_dict=search._default_config_dict,
        log=search.log,
    )


def test_an_llm_reply_that_names_a_gated_knob_is_held_to_the_search() -> None:
    kernel, args = KERNELS["offsets"]
    with forbid_cuda_work(), tcgen05_target(), mock_cuda_unavailable():
        proposed = _bind("offsets", proposed=True)
        bound = cpu_bind(kernel(), args)
        spec = bound.config_spec
        search = LLMGuidedSearch(bound, args)
        search._initialize_prompt_state()
        search.log = MagicMock()
        generation = search.config_gen
        assert not set(GATED) & set(generation._flat_fields())
        # One seed per family of the lifted gate's palette: the shift on an
        # N,M worklist, three M,N tiles and a paired prefix recipe.
        picks = _explicit(proposed)
        assert [pick.get(ORIENTATION, "nm") for pick in picks] == [
            "nm",
            "mn",
            "mn",
            "mn",
            "mn",
        ]
        for pick in picks:
            search.log.reset_mock()
            reply = _reply(search, spec.normalized_config(pick))
            (parsed,) = _parsed(search, reply)
            # The reply's keys are merged and normalized like any caller's
            # config: the parser keeps the shift.
            assert parsed[KEY] is True
            admitted = search._admit_proposals([parsed])
            messages = [call.args[0] for call in search.log.call_args_list]
            # What a round takes in is the admitted reply: the search's own
            # parse admits what the parser read.
            assert search._parse_configs(reply) == admitted
            (config,) = admitted
            named = f"{KEY}, {PREFIX}" if pick.get(PREFIX) else KEY
            assert messages[-1] == (
                "An LLM proposal names options this search does not propose "
                f"({named}); benchmarking it without them"
            )
            assert config == spec.normalized_config(config)
            # What the search benchmarks is the reply's worklist without the
            # shift: the same orientation and tile, and no option of the
            # gate, as its own candidates.
            assert not set(surface.D_SHIFT_PROPOSAL_KEYS) & set(config.config)
            assert not set(GATED) & set(config.config)
            assert config.get(ORIENTATION, "nm") == pick.get(ORIENTATION, "nm")
            assert config.get(MMA_N) == pick.get(MMA_N)
            if not pick.get(PREFIX):
                assert _holds(config, spec.normalized_config(_without_shift(pick)))
            assert "bounded_output_shift" not in bound.to_code(config)

        # Every M,N tile and recipe the compiler seeds with the gate lifted,
        # as a reply: each is benchmarked under M,N at its MMA width, without
        # the shift, and none is dropped.
        tiles = 0
        for seed in proposed.config_spec.compiler_seed_configs:
            if seed.get(ORIENTATION) != "mn":
                continue
            tiles += 1
            (parsed,) = _parsed(search, _reply(search, spec.normalized_config(seed)))
            (config,) = search._admit_proposals([parsed])
            assert config[ORIENTATION] == "mn"
            assert config[MMA_N] == seed[MMA_N]
            assert config["tcgen05_ab_stages"] == seed["tcgen05_ab_stages"]
            assert not set(GATED) & set(config.config)
        assert tiles == 165

        # A proposal that names none of the gated options is the search's
        # own kind and is benchmarked as parsed.
        search.log.reset_mock()
        plain = spec.normalized_config(spec.compiler_seed_configs[0])
        reply = _reply(search, plain)
        (parsed,) = _parsed(search, reply)
        search.log.reset_mock()
        (same,) = search._admit_proposals([parsed])
        assert same is parsed
        search.log.assert_not_called()
        assert search._parse_configs(reply) == [parsed]

        # With the gate lifted the options are coordinates of the search,
        # and a reply names them as a draw does.
        lifted = LLMGuidedSearch(proposed, args)
        lifted._initialize_prompt_state()
        lifted.log = MagicMock()
        for pick in picks:
            (parsed,) = _parsed(
                lifted, _reply(lifted, proposed.config_spec.normalized_config(pick))
            )
            assert parsed[KEY] is True
            lifted.log.reset_mock()
            (same,) = lifted._admit_proposals([parsed])
            assert same is parsed
            lifted.log.assert_not_called()


def _reply_naming(bound: BoundKernel, *proposals: dict[str, object]) -> str:
    """An LLM's reply that proposes ``proposals``, each as the whole config
    a search of ``bound`` would report for it."""
    spec = bound.config_spec
    return json.dumps(
        {
            "configs": [
                dict(spec.normalized_config(proposal).config) for proposal in proposals
            ]
        }
    )


def _benchmarked(
    search: LLMGuidedSearch, replies: tuple[str, str]
) -> list[tuple[str, list[helion.Config]]]:
    """The batches ``search`` hands to its benchmark, in order, when the LLM
    answers round 0 with ``replies[0]`` and the refinement round with
    ``replies[1]``.

    The loop and its two rounds are the search's own. What needs a device or
    a provider is a stand-in: the two LLM calls, the benchmark and the
    reading it would leave for the best member, the rebenchmark that ends a
    round and the final selection."""
    batches: list[tuple[str, list[helion.Config]]] = []

    def benchmark(configs: list[helion.Config], *, generation: int, desc: str) -> None:
        batches.append((desc, list(configs)))

    first: concurrent.futures.Future[str] = concurrent.futures.Future()
    first.set_result(replies[0])
    search.log = MagicMock()
    with (
        patch.object(search, "_call_llm_async", return_value=first),
        patch.object(search, "_call_llm", return_value=replies[1]),
        patch.object(search, "_benchmark_and_ingest", side_effect=benchmark),
        patch.object(search, "_finalize_round"),
        patch.object(search, "_finalize"),
        patch.object(
            LLMGuidedSearch,
            "best",
            new_callable=PropertyMock,
            return_value=SimpleNamespace(perf=1.0),
        ),
    ):
        search._autotune_inner()
    return batches


def test_neither_round_of_an_llm_search_benchmarks_a_gated_knob() -> None:
    """Round 0 and a refinement round each take in a reply of their own,
    through ``_parse_configs``, which admits what it parsed
    (``_admit_proposals``); a round that benchmarked its reply as the
    parser read it would store through the shift. So the search's own loop
    runs here over one reply per round that names the gated knobs, and
    nothing it hands to the benchmark names one."""
    kernel, args = KERNELS["offsets"]
    with forbid_cuda_work(), tcgen05_target(), mock_cuda_unavailable():
        proposed = _bind("offsets", proposed=True)
        bound = cpu_bind(kernel(), args)
        picks = _explicit(proposed)
        # Round 0 is answered with the shift on an N,M worklist and a paired
        # prefix recipe, the refinement round with an M,N tile: every family
        # of the lifted gate's palette, and a new config for each round,
        # which benchmarks no config twice.
        replies = (
            _reply_naming(bound, picks[0], picks[4]),
            _reply_naming(bound, picks[1]),
        )
        # The reference config, then each round's proposals.
        expected = [("Round 0 seed", 1), ("Round 0 LLM", 2), ("Round 1", 1)]

        search = LLMGuidedSearch(bound, args, initial_random_configs=0, max_rounds=2)
        batches = _benchmarked(search, replies)
        assert [(desc, len(configs)) for desc, configs in batches] == expected
        for _, configs in batches:
            for config in configs:
                assert not set(surface.D_SHIFT_PROPOSAL_KEYS) & set(config.config)
                assert not set(GATED) & set(config.config)
        # The worklists the replies name are benchmarked, without the shift:
        # the N,M one and the recipe's tile in round 0, the M,N tile in the
        # refinement round.
        (_, (unshifted, recipe_tile)), (_, (tile,)) = batches[1:]
        assert unshifted.get(ORIENTATION, "nm") == "nm"
        assert recipe_tile[ORIENTATION] == "mn"
        assert recipe_tile[MMA_N] == picks[4][MMA_N]
        assert _holds(
            tile, bound.config_spec.normalized_config(_without_shift(picks[1]))
        )
        # Each round's log says which of its proposals it changed.
        changed = [
            call.args[0]
            for call in search.log.call_args_list
            if call.args[0].startswith("An LLM proposal names options")
        ]
        assert len(changed) == 3

        # With the gate lifted the options are coordinates of the search, and
        # the same replies reach the benchmark as written: the stand-ins
        # hide no gated knob.
        lifted = LLMGuidedSearch(proposed, args, initial_random_configs=0, max_rounds=2)
        batches = _benchmarked(lifted, replies)
        assert [(desc, len(configs)) for desc, configs in batches] == expected
        (_, (shifted, recipe)), (_, (mn,)) = batches[1:]
        assert shifted[KEY] is True
        assert shifted[ORIENTATION] == "nm"
        assert recipe[KEY] is True
        assert recipe[PREFIX] is True
        assert mn[KEY] is True
        assert mn[ORIENTATION] == "mn"


def test_the_backend_holds_proposals_only() -> None:
    """The hook acts on a proposal a search takes from outside its flat
    fields. A config the user names is not one: it normalizes to itself and
    compiles as written (``test_an_explicit_config_keeps_the_shift_and_its_
    code``), whatever the hook would do with it as a proposal."""
    with forbid_cuda_work(), tcgen05_target(), mock_cuda_unavailable():
        proposed = _bind("offsets", proposed=True)
        bound = _bind("offsets")
        spec = bound.config_spec
        shifted = spec.normalized_config(_explicit(proposed)[0])
        assert shifted[KEY] is True
        # As a proposal the config loses the shift ...
        config, returned = spec.search_proposal(shifted)
        assert returned == (KEY,)
        assert not config.get(KEY)
        # ... and the caller's config is left as it was: the kernel still
        # normalizes it to itself and renders the shift.
        assert shifted[KEY] is True
        assert spec.normalized_config(shifted) == shifted
        assert "'bounded_output_shift': True" in bound.to_code(shifted)
        explicit = helion.kernel(
            grouped_offsets_k64.fn,
            backend="cute",
            static_shapes=True,
            config=shifted,
        )
        _, args = KERNELS["offsets"]
        named = cpu_bind(explicit, args)
        assert named.to_code(shifted) == bound.to_code(shifted)
