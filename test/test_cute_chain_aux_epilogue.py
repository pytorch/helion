"""The exact-shape auxiliary epilogue mechanisms on a materialized chain's GEMM launch.

CPU render pins (fake sm_100 target, the ``test_cute_phase_mma_roots``
fixtures): the GEMM launch of an ordered ``hl.barrier()`` chain
(``squeeze_and_excitation_net_bwd_dc``: a pointwise phase, then the
relu-backward masked GEMM ``grad_c = acc.to(fp16) * (c[tile] > 0)``) takes
the mechanisms that hide its auxiliary read behind the mainloop -- the
register prefetch ring (``tcgen05_aux_prefetch_subtiles``), the
whole-fragment hoist (``tcgen05_aux_fragment_hoist``), the epilogue-issued
aux ring (``tcgen05_aux_load_mode="epi_tma"``), the epilogue warp groups and
the configured subtile -- on its 256-row CtaGroup.TWO pair launch, judged by
the normalizer exactly as ``materialized_fission_codegen._stage_config``
renders it (``CuteTcgen05Config.region_launch_config``).  The region AB ring
is priced beside the rings each mechanism adds; the 128-row pair tile, the
producer-warp ring and a launch left flat are refused with the launch named;
search samples outside the envelope repair to fixed points; the chain's
seeds carry the mechanisms' 256-row twins; and the pointwise launch and the
mainloop are untouched (the aux read moves, the arithmetic does not).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from typing import cast

from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_bwd_dc
from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd
import pytest
import torch

from test.cute_population_contracts import with_flat_min_blocks_default
from test.test_cute_full_slice_matmul import _bind
from test.test_cute_materialized_fission import _se_args
from test.test_cute_materialized_fission import _sources
from test.test_cute_phase_mma_roots import _native_cpu_target  # noqa: F401

import helion
from helion._compiler.autotuner_heuristics.cute_materialized import (
    CuteMaterializedMmaHeuristic,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_REGION_TWO_CTA_LAUNCH_PROJECTION,
)
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
from helion.exc import InvalidConfig

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from typing import Any

    from helion.runtime.kernel import BoundKernel


# The campaign shape of squeeze_and_excitation_net 00_backward shape 2:
# grad_cb is a 4096x1024 pointwise phase, grad_c a 4096x512x1024 GEMM.
M, N, K = 4096, 1024, 512


def _bind_chain() -> BoundKernel[Any]:
    generator = torch.Generator().manual_seed(31)
    t = {
        name: torch.randn(dims, generator=generator, dtype=torch.float16) * 0.25
        for name, dims in (
            ("grad_out", (M, N)),
            ("x", (M, N)),
            ("b", (K, N)),
            ("c", (M, K)),
            ("d", (M, N)),
        )
    }
    return _bind(
        squeeze_and_excitation_net_bwd_dc.fn,
        (t["grad_out"], t["x"], t["b"], t["c"], t["d"]),
    )


def _pair_config(bound: BoundKernel[Any], **overrides: object) -> helion.Config:
    """The campaign winner's recipe (a persistent CtaGroup.TWO GEMM launch of
    the flat chain, the relu mask read ahead of the accumulator wait) on the
    256x128 pair tile at bk=64 with the deepest ring that fits (8)."""
    spec = bound.config_spec
    values = dict(spec.default_config().config)
    values.update(
        block_sizes=[8, 64, 256, 128, 64],
        num_threads=[8, 4, 0, 0, 0],
        cute_vector_widths=[1, 16, 1, 1, 1],
        cute_lane_layouts=["blocked", "strided", "blocked", "blocked", "blocked"],
        pid_type="flat",
        tcgen05_region_pid_types=["persistent_blocked"],
        tcgen05_region_ab_stages=[8],
        tcgen05_region_c_stages=[2],
        tcgen05_cta_group="two",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=1,
        tcgen05_aux_load_placement="pre_acc_wait",
        tcgen05_l2_swizzle_size=2,
    )
    values.update(overrides)
    return helion.Config.from_dict(values)


def _strict(bound: BoundKernel[Any], config: helion.Config) -> helion.Config:
    with bound.env:
        return bound.config_spec.normalized_config(config)


def _repaired(bound: BoundKernel[Any], config: helion.Config) -> helion.Config:
    generation = ConfigGeneration(bound.config_spec)
    with bound.env:
        return generation.unflatten(generation.flatten(config))


def _stages(bound: BoundKernel[Any], config: helion.Config) -> list[str]:
    return _sources(bound.to_code(config))


def _mainloop(source: str) -> str:
    """The GEMM launch's source up to its epilogue (the K loop included)."""
    return source[: source.index("tcgen05_kernel_desc")]


MECHANISMS: dict[str, dict[str, object]] = {
    "prefetch_ring": {"tcgen05_aux_prefetch_subtiles": 2},
    "fragment_hoist": {"tcgen05_aux_fragment_hoist": True},
    "epi_tma": {"tcgen05_aux_load_mode": "epi_tma"},
    "epi_warp_groups": {"tcgen05_epi_warp_groups": 2},
    "epi_subtile": {"tcgen05_epi_subtile_n": 32},
}
MARKERS = {
    "prefetch_ring": "tcgen05_aux_ring_",
    "fragment_hoist": "aux_rmem_full",
    "epi_tma": "tcgen05_aux_tma",
    "epi_warp_groups": "epi_group",
    "epi_subtile": "cute.make_layout(32)",
}


def test_chain_launch_config_is_what_the_renderer_sees() -> None:
    bound = _bind_chain()
    state = bound.config_spec._cute_tcgen05_config
    assert state.materialized_matmul_block_ids == ((2, 3, 4),)
    assert state.region_has_exact_aux(0)
    assert state.exact_shape_aux_kernel_detected and state.exact_aux_tma_provable
    config = _strict(bound, _pair_config(bound, tcgen05_aux_load_mode="epi_tma"))
    with bound.env:
        (launch,) = [cfg for _region, cfg in state.chain_launches(config.config)]
    # The persistent launch: its pid type, the paired pipeline's fixed
    # coordinates, its region's ring depth and the kernel's aux mode.
    assert launch["pid_type"] == "persistent_blocked"
    assert launch["tcgen05_ab_stages"] == 8 and launch["tcgen05_c_stages"] == 2
    assert launch["tcgen05_aux_load_mode"] == "epi_tma"
    assert all(
        launch[key] == value for key, value in TCGEN05_REGION_TWO_CTA_LAUNCH_PROJECTION
    )
    assert "tcgen05_region_pid_types" not in launch
    assert "tcgen05_region_ab_stages" not in launch
    # The aux load mode is the kernel's, not projected away.
    assert "tcgen05_aux_load_mode" not in dict(TCGEN05_REGION_TWO_CTA_LAUNCH_PROJECTION)


@pytest.mark.parametrize("mechanism", sorted(MECHANISMS))
def test_mechanism_admitted_on_the_pair_launch_and_rendered_there_only(
    mechanism: str,
) -> None:
    bound = _bind_chain()
    base = _strict(bound, _pair_config(bound))
    config = _strict(bound, _pair_config(bound, **MECHANISMS[mechanism]))
    for key, value in MECHANISMS[mechanism].items():
        assert config.config[key] == value, key
    # Strict normalization is a fixed point and the flat search schema
    # carries every coordinate.
    assert _strict(bound, helion.Config.from_dict(dict(config.config))) == config
    assert _repaired(bound, config) == with_flat_min_blocks_default(
        bound.config_spec, config
    )
    pointwise, gemm = _stages(bound, config)
    base_pointwise, base_gemm = _stages(bound, base)
    # The pointwise launch does not see the GEMM launch's epilogue knobs.
    assert pointwise == base_pointwise
    assert MARKERS[mechanism] in gemm and MARKERS[mechanism] not in base_gemm
    assert "CtaGroup.TWO" in gemm and "'use_pdl': True" in gemm
    if mechanism in ("prefetch_ring", "fragment_hoist"):
        # The aux read moves within the epilogue; the mainloop (tiles, K
        # order, MMAs) is the base launch's.  (The warp groups grow the role
        # launch to ``4G + 2`` warps, renumbering the TMA and MMA warps.)
        assert _mainloop(gemm) == _mainloop(base_gemm)


def test_mechanisms_compose_on_the_pair_launch() -> None:
    bound = _bind_chain()
    config = _strict(
        bound,
        _pair_config(
            bound,
            tcgen05_epi_subtile_n=32,
            tcgen05_aux_prefetch_subtiles=1,
            tcgen05_epi_warp_groups=2,
        ),
    )
    assert config.config["tcgen05_aux_prefetch_subtiles"] == 1
    assert config.config["tcgen05_epi_warp_groups"] == 2
    _pointwise, gemm = _stages(bound, config)
    assert "tcgen05_aux_ring_" in gemm and "epi_group" in gemm


def test_region_ring_is_priced_beside_the_epilogue_rings() -> None:
    bound = _bind_chain()
    # bk=64 on the 256x128 pair: 24 KiB of A/B per stage per CTA; eight
    # stages fit beside the 2-stage C ring, the 2-stage aux ring of
    # ``epi_tma`` and a second warp group's C stages, nine do not.
    assert _strict(bound, _pair_config(bound, tcgen05_region_ab_stages=[8])).config[
        "tcgen05_region_ab_stages"
    ] == [8]
    with pytest.raises(InvalidConfig, match="per-region SMEM budgets"):
        _strict(bound, _pair_config(bound, tcgen05_region_ab_stages=[9]))
    for extra in (
        {"tcgen05_aux_load_mode": "epi_tma"},
        {"tcgen05_epi_warp_groups": 2},
    ):
        deep = _pair_config(bound, tcgen05_region_ab_stages=[9], **extra)
        with pytest.raises(InvalidConfig, match="per-region SMEM budgets"):
            _strict(bound, deep)
        repaired = _repaired(bound, deep)
        assert repaired.config["tcgen05_region_ab_stages"] == [8]
        for key, value in extra.items():
            assert repaired.config[key] == value
    # bk=128 doubles the stage: four fit beside the aux ring, five do not,
    # and the epilogue-issued ring at the deepest fitting depth renders.
    wide = _pair_config(
        bound,
        block_sizes=[8, 64, 256, 128, 128],
        tcgen05_region_ab_stages=[5],
        tcgen05_aux_load_mode="epi_tma",
    )
    with pytest.raises(InvalidConfig, match="per-region SMEM budgets"):
        _strict(bound, wide)
    repaired = _repaired(bound, wide)
    assert repaired.config["tcgen05_region_ab_stages"] == [4]
    assert repaired.config["tcgen05_aux_load_mode"] == "epi_tma"
    assert "tcgen05_aux_tma" in _stages(bound, repaired)[1]


def test_outside_the_envelope_is_refused_by_launch_and_repaired() -> None:
    bound = _bind_chain()
    # The campaign winner's 128x256 pair tile (a 64-row per-CTA epilogue) is
    # outside the mechanisms' validated envelope.
    narrow = {"block_sizes": [8, 64, 128, 256, 128], "tcgen05_region_ab_stages": [4]}
    for mechanism, reason in (
        ("prefetch_ring", "256-row CtaGroup.TWO pair tile"),
        ("epi_warp_groups", "256-row CtaGroup.TWO pair tile"),
        ("epi_subtile", "256-row CtaGroup.TWO pair tile"),
        ("epi_tma", "bm=128 CtaGroup.TWO epilogue tile"),
    ):
        config = _pair_config(bound, **narrow, **MECHANISMS[mechanism])
        with pytest.raises(InvalidConfig, match=reason) as info:
            _strict(bound, config)
        assert "GEMM launch 0 of the materialized chain" in str(info.value)
        repaired = _repaired(bound, config)
        assert all(
            repaired.config[key] == bound.config_spec.default_config().config[key]
            for key in MECHANISMS[mechanism]
        )
        assert _strict(bound, helion.Config.from_dict(dict(repaired.config))) == (
            repaired
        )
    # The hoist is not a subtile-loop rewrite: the 128-row pair's 64-row
    # per-CTA fragment (256 B per epilogue thread) fits the register budget,
    # so the campaign winner's own tile takes it.
    hoisted = _strict(
        bound, _pair_config(bound, **narrow, tcgen05_aux_fragment_hoist=True)
    )
    assert hoisted.config["tcgen05_aux_fragment_hoist"] is True
    plain_narrow = _strict(bound, _pair_config(bound, **narrow))
    pointwise, gemm = _stages(bound, hoisted)
    plain_pointwise, plain_gemm = _stages(bound, plain_narrow)
    assert pointwise == plain_pointwise
    assert "aux_rmem_full" in gemm and "aux_rmem_full" not in plain_gemm
    assert _mainloop(gemm) == _mainloop(plain_gemm)
    # The producer-warp ring has no chain lowering.
    with pytest.raises(InvalidConfig, match="producer-warp auxiliary ring"):
        _strict(bound, _pair_config(bound, tcgen05_aux_load_mode="tma"))
    assert (
        _repaired(bound, _pair_config(bound, tcgen05_aux_load_mode="tma")).config[
            "tcgen05_aux_load_mode"
        ]
        == "simt"
    )
    # A GEMM launch left flat (the one-shot one-CTA kernel) keeps the
    # per-subtile register reads: the subtile-loop rewrites are refused.
    flat_values = dict(bound.config_spec.default_config().config)
    flat_values.update(
        block_sizes=[8, 64, 128, 128, 64],
        pid_type="flat",
        tcgen05_ab_stages=4,
        tcgen05_aux_load_placement="pre_acc_wait",
    )
    for mechanism in ("prefetch_ring", "epi_warp_groups", "epi_subtile"):
        with pytest.raises(InvalidConfig, match="launch left flat"):
            _strict(bound, helion.Config.from_dict(flat_values | MECHANISMS[mechanism]))
    with pytest.raises(InvalidConfig, match="tcgen05_aux_fragment_hoist=True requires"):
        _strict(
            bound,
            helion.Config.from_dict(flat_values | {"tcgen05_aux_fragment_hoist": True}),
        )
    # The epilogue-issued ring renders on the flat launch too, priced by the
    # one-CTA model: six stages of the 128x128x64 tile fit beside its rings.
    flat_epi_tma = _strict(
        bound,
        helion.Config.from_dict(
            flat_values | {"tcgen05_aux_load_mode": "epi_tma", "tcgen05_ab_stages": 6}
        ),
    )
    assert flat_epi_tma.config["tcgen05_ab_stages"] == 6
    assert "tcgen05_aux_tma" in _stages(bound, flat_epi_tma)[1]


def test_chain_seeds_carry_the_mechanism_twins() -> None:
    bound = _bind_chain()
    host = bound.host_function
    assert host is not None
    spec = bound.config_spec
    with bound.env, host:
        seeds = CuteMaterializedMmaHeuristic.get_seed_configs(bound.env, host.device_ir)
    twins = [
        seed
        for seed in seeds
        if seed.config.get("tcgen05_aux_prefetch_subtiles")
        or seed.config.get("tcgen05_aux_fragment_hoist")
        or seed.config.get("tcgen05_aux_load_mode") == "epi_tma"
    ]
    # Per K tile: the ring at its deepest depth (three (128, 32) subtile
    # fragments of the relu mask per thread), the hoist and the
    # epilogue-issued ring, on the 256x128 pair at the ring the budget fits,
    # plus the hoist on the geometry rule's own 128x256 pair.
    assert [
        (
            seed.block_sizes[2:],
            seed.config["tcgen05_region_ab_stages"],
            seed.config.get("tcgen05_aux_prefetch_subtiles", 0),
            seed.config.get("tcgen05_aux_fragment_hoist", False),
            seed.config.get("tcgen05_aux_load_mode", "simt"),
        )
        for seed in twins
    ] == [
        ([256, 128, 64], [8], 3, False, "simt"),
        ([256, 128, 64], [8], 0, True, "simt"),
        ([128, 256, 64], [8], 0, True, "simt"),
        ([256, 128, 64], [8], 0, False, "epi_tma"),
        ([256, 128, 128], [4], 3, False, "simt"),
        ([256, 128, 128], [4], 0, True, "simt"),
        ([128, 256, 128], [4], 0, True, "simt"),
        ([256, 128, 128], [4], 0, False, "epi_tma"),
    ]
    generation = ConfigGeneration(spec)
    for seed in twins:
        assert seed.config["tcgen05_aux_load_placement"] == "pre_acc_wait"
        assert seed.config["tcgen05_region_pid_types"] == ["persistent_blocked"]
        with bound.env:
            normalized = bound._normalized_config_copy(
                helion.Config.from_dict(spec.default_config().config | seed.config)
            )
        for key in (
            "tcgen05_aux_prefetch_subtiles",
            "tcgen05_aux_fragment_hoist",
            "tcgen05_aux_load_mode",
            "tcgen05_region_ab_stages",
        ):
            if key in seed.config:
                assert normalized.config[key] == seed.config[key], key
        assert generation.unflatten(
            generation.flatten(normalized)
        ) == with_flat_min_blocks_default(spec, normalized)


def test_two_gemm_fission_plan_without_a_single_output_aux_proof_stays_inert() -> None:
    # squeeze_and_excitation_net_fwd: c = relu(x @ a) (no auxiliary operand)
    # and the shared fanout d / out = x * d (two outputs: no single-output
    # allocation proof records the exact-shape ``x``).  No launch has a
    # subject for the mechanisms: refused strictly, repaired in search, and
    # the campaign winner's recipe renders unchanged.
    bound = _bind(
        squeeze_and_excitation_net_fwd.fn, _se_args((128, 256, 64)), fission=True
    )
    state = bound.config_spec._cute_tcgen05_config
    assert state.materialized_matmul_block_ids == ((0, 1, 2), (3, 4, 5))
    assert not any(state.region_has_exact_aux(region) for region in range(2))
    values = dict(bound.config_spec.default_config().config)
    for mechanism in ("prefetch_ring", "fragment_hoist", "epi_tma"):
        config = helion.Config.from_dict(values | MECHANISMS[mechanism])
        with pytest.raises(InvalidConfig):
            _strict(bound, config)
        repaired = _repaired(bound, config)
        assert all(repaired.config[key] == values[key] for key in MECHANISMS[mechanism])


def test_the_render_of_a_pinned_record_is_unchanged() -> None:
    # The campaign winner of squeeze_and_excitation_net 00_backward shape 2
    # (``bwd_dc``: the 128x256 pair at bk=128, four region stages, the relu
    # mask read ahead of the accumulator wait through registers) carries no
    # mechanism; its render is the pre-change one, byte for byte.
    bound = _bind_chain()
    winner = json.loads(
        '{"block_sizes": [8, 64, 128, 256, 128], "num_threads": [8, 4, 0, 0, 0],'
        ' "loop_orders": [[1, 0], [0, 1]], "l2_groupings": [1, 16],'
        ' "cute_proven_bounds": false, "cute_pdl": true,'
        ' "indexing": ["tensor_descriptor", "tensor_descriptor", "tensor_descriptor",'
        ' "tensor_descriptor", "tensor_descriptor", "tensor_descriptor",'
        ' "tensor_descriptor", "pointer", "tensor_descriptor"],'
        ' "pid_type": "flat", "tcgen05_cta_group": "two", "tcgen05_cluster_m": 2,'
        ' "tcgen05_cluster_n": 1, "tcgen05_ab_stages": 5, "tcgen05_acc_stages": 1,'
        ' "tcgen05_c_stages": 2, "tcgen05_num_epi_warps": 4, "tcgen05_l2_swizzle_size": 2,'
        ' "tcgen05_c_store_mode": "normal", "tcgen05_region_ab_stages": [4],'
        ' "tcgen05_region_c_stages": [2], "tcgen05_region_pid_types": ["persistent_blocked"],'
        ' "tcgen05_aux_load_placement": "pre_acc_wait", "tcgen05_aux_fragment_hoist": false,'
        ' "tcgen05_aux_prefetch_subtiles": 0, "tcgen05_epi_subtile_n": null,'
        ' "tcgen05_epi_warp_groups": 1, "tcgen05_strategy": "role_local_monolithic",'
        ' "tcgen05_layout_strategy": "default", "tcgen05_warp_spec_mma_warps": 1,'
        ' "tcgen05_warp_spec_ab_load_warps": 1, "tcgen05_warp_spec_epi_load_warps": 0,'
        ' "tcgen05_warp_spec_scheduler_warps": 0, "tcgen05_warp_spec_c_input_warps": 0,'
        ' "tcgen05_warp_spec_store_warps": 0, "tcgen05_warp_spec_register_decrease": 120,'
        ' "tcgen05_warp_spec_register_increase": 256, "tcgen05_aux_load_mode": "simt",'
        ' "cute_vector_widths": [1, 16, 1, 1, 1],'
        ' "cute_lane_layouts": ["blocked", "strided", "blocked", "blocked", "blocked"],'
        ' "cute_min_blocks_per_mp": 0, "cute_collective_mma": false,'
        ' "cute_collective_native_seeded": false, "cute_collective_tmem_seed": false,'
        ' "cute_collective_tmem_a": false, "cute_collective_copy": "scalar",'
        ' "cute_collective_compute": "warp", "tcgen05_persistence_model": "non_persistent",'
        ' "tcgen05_layout_overrides_epi_tile_m": null, "tcgen05_layout_overrides_epi_tile_n": null,'
        ' "tcgen05_layout_overrides_smem_swizzle_a": null,'
        ' "tcgen05_layout_overrides_smem_swizzle_b": null,'
        ' "tcgen05_layout_overrides_d_store_box_n": null}'
    )
    config = _strict(bound, helion.Config.from_dict(winner))
    assert dict(config.config) == winner
    pointwise, gemm = _stages(bound, config)
    assert "griddepcontrol_wait" in pointwise
    assert "CtaGroup.TWO" in gemm and "'use_pdl': True" in gemm
    assert not any(marker in gemm for marker in MARKERS.values())
    # Only the aux mechanisms project per launch; without them the launch
    # config equals the kernel's minus the region keys.
    state = bound.config_spec._cute_tcgen05_config
    with bound.env:
        (launch,) = [cfg for _region, cfg in state.chain_launches(config.config)]
    expected = {
        key: value
        for key, value in winner.items()
        if not key.startswith("tcgen05_region_")
    }
    expected.update(TCGEN05_REGION_TWO_CTA_LAUNCH_PROJECTION)
    expected.update(pid_type="persistent_blocked", tcgen05_ab_stages=4)
    assert launch == cast("dict[str, object]", expected)
