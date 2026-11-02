"""Per-phase tcgen05 planning for ``hl.barrier()`` kernels with several GEMMs.

CPU render pins: the barrier phases of the two-GEMM backward chains
(``test._cute_se_bwd_chains``, the pre-split ``squeeze_and_excitation_net``
``bwd_dx`` / ``bwd_da`` bodies) are separate ordered launches, so each GEMM
phase owns its own tile triple (``materialized_matmul_block_ids``) and seeds,
the pointwise phase keeps its SIMT layout knobs, and the relu-backward mask
epilogue ``acc.to(fp16) * (c[tile] > 0)`` splices into the tcgen05 epilogue
chain.
``tcgen05_region_pid_types`` runs chosen GEMM launches of the flat chain as
the static persistent two-CTA pipeline of the standalone single-GEMM surface,
and the launches left flat keep the one-CTA pipeline at the AB depth their
tile fits.  A single-GEMM chain (``squeeze_and_excitation_net_bwd_dc``) is
planned per phase as well: its GEMM launch owns the one region triple and the
per-launch pid types, and its matmul-free phase is a pointwise region of the
same kind: its layout knobs are searched, seeded with one 16-byte packet per
thread, and honored by the grid vectorizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_bwd_dc
from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test._cute_se_bwd_chains import se_bwd_da_chain
from test._cute_se_bwd_chains import se_bwd_dx_chain
from test.cute_population_contracts import with_flat_min_blocks_default
from test.test_cute_full_slice_matmul import _bind
from test.test_cute_full_slice_matmul import _cpu_target
from test.test_cute_materialized_fission import _se_args
from test.test_cute_materialized_fission import _sources

import helion
from helion._compiler.autotuner_heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._compiler.autotuner_heuristics.cute_materialized import (
    CuteMaterializedMmaHeuristic,
)
from helion._compiler.autotuner_heuristics.cute_materialized_operand import (
    CuteMaterializedOperandHeuristic,
)
from helion._compiler.cute.aux_tensor import analyze_tcgen05_matmul_store_chains
from helion._compiler.cute.mma_support import CuteMmaSupport
from helion._testing import skipUnlessBackends
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_fragment import ListOf
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any

    from helion.runtime.kernel import BoundKernel


@pytest.fixture(autouse=True)
def _native_cpu_target() -> Generator[None, None, None]:
    support = CuteMmaSupport(
        universal=True, warp_f16bf16=True, warpgroup_f16bf16=True, tcgen05_f16bf16=True
    )
    with (
        _cpu_target(),
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        patch(
            "helion._compiler.cute.mma_support.get_cute_mma_support",
            return_value=support,
        ),
        patch(
            "helion._compiler.cute.cute_mma.get_cute_mma_support", return_value=support
        ),
        patch(
            "helion._compiler.cute.tcgen05_config.CuteTcgen05Config.per_cta_smem_budget_bytes",
            return_value=232448,
        ),
        patch(
            "helion._compiler.cute.tcgen05_config.CuteTcgen05Config.per_cta_smem_capacity_bytes",
            return_value=232448,
        ),
    ):
        yield


def _bwd_args(
    shape: tuple[int, int, int] = (256, 256, 64), dtype: torch.dtype = torch.float16
) -> dict[str, torch.Tensor]:
    m, n, k = shape
    generator = torch.Generator().manual_seed(31)
    return {
        name: torch.randn(dims, generator=generator, dtype=dtype) * 0.25
        for name, dims in (
            ("grad_out", (m, n)),
            ("x", (m, n)),
            ("a", (n, k)),
            ("b", (k, n)),
            ("c", (m, k)),
            ("d", (m, n)),
        )
    }


def _bind_bwd_dx(**kwargs: object) -> BoundKernel[Any]:
    t = _bwd_args(**kwargs)  # type: ignore[arg-type]
    return _bind(
        se_bwd_dx_chain,
        (t["grad_out"], t["x"], t["a"], t["b"], t["c"], t["d"]),
    )


def _bind_bwd_da(**kwargs: object) -> BoundKernel[Any]:
    t = _bwd_args(**kwargs)  # type: ignore[arg-type]
    return _bind(
        se_bwd_da_chain,
        (t["grad_out"], t["x"], t["b"], t["c"], t["d"]),
    )


@pytest.mark.parametrize("bind", [_bind_bwd_dx, _bind_bwd_da], ids=["dx", "da"])
def test_barrier_phases_plan_one_tcgen05_triple_per_gemm(bind: object) -> None:
    bound = bind()  # type: ignore[operator]
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert spec.cute_tcgen05_search_enabled
    # No single-matrix projection: each GEMM phase has its own triple.
    assert spec._tcgen05_matmul_block_ids() is None
    assert state.materialized_matmul_block_ids == ((2, 3, 4), (5, 6, 7))
    assert state.materialized_matmul_region_indices == (1, 2)
    assert len(state.materialized_matmul_shapes) == 2
    # The pointwise phase keeps its SIMT layout knobs; GEMM axes are pinned.
    assert spec.cute_pointwise_region_block_ids == frozenset({0, 1})
    assert spec.cute_pointwise_region_grid_groups == ((0, 1),)
    # Ordered per-phase launches are the only lowering hosting tcgen05.
    assert spec.allowed_pid_types == ("flat",)
    assert spec.default_config().pid_type == "flat"
    # The search surface: tcgen05 knobs, the pointwise phase's SIMT layout
    # knobs and the launch chain's ``cute_pdl``; the pointwise pid-type knob
    # is inert when every launch is flat and stays off the surface.
    keys = set(ConfigGeneration(spec)._key_to_flat_indices)
    assert {"tcgen05_ab_stages", "num_threads", "cute_vector_widths"} <= keys
    assert "cute_pdl" in keys
    assert "cute_pointwise_pid_type" not in keys
    # (``cute_pdl`` reaches the chain through the tcgen05 config's
    # ``optional_fragments`` -- the chain is not a fission bundle.)
    # Every GEMM phase has its complete single-output allocation proof (the
    # exact-shape auxiliary epilogues load through registers), so the chain
    # offers ``tcgen05_region_pid_types``: a GEMM launch may run the static
    # persistent two-CTA pipeline while the chain itself stays flat.  The
    # two-CTA coordinates and the per-region ring depths come with it; the
    # fission plan's materialized-operand PDL edge does not.
    assert all(facts is not None for facts in state.materialized_pipeline_facts)
    assert state.region_pid_type_choices() == (
        "inherit",
        "persistent_blocked",
        "persistent_interleaved",
    )
    fragments = state.optional_fragments(for_search=True)
    assert fragments["tcgen05_cluster_m"].choices == (1, 2)  # type: ignore[attr-defined]
    assert fragments["tcgen05_cta_group"].choices == ("auto", "two")  # type: ignore[attr-defined]
    assert {
        "tcgen05_region_ab_stages",
        "tcgen05_region_c_stages",
        "tcgen05_region_pid_types",
    } <= keys
    assert "tcgen05_materialized_pdl" not in keys
    # Explicit configs still validate the full coordinate set.
    assert state.optional_fragments()["tcgen05_cluster_m"].choices == (1, 2)  # type: ignore[attr-defined]


def test_phase_seeds_round_trip_and_render_three_launches() -> None:
    bound = _bind_bwd_dx()
    host = bound.host_function
    assert host is not None
    spec = bound.config_spec
    with bound.env, host:
        seeds = CuteMaterializedMmaHeuristic.get_seed_configs(bound.env, host.device_ir)
        assert seeds
        generation = ConfigGeneration(spec)
        for seed in seeds:
            normalized = bound._normalized_config_copy(
                helion.Config.from_dict(spec.default_config().config | seed.config)
            )
            assert generation.unflatten(
                generation.flatten(normalized)
            ) == with_flat_min_blocks_default(spec, normalized)
    defaults = spec.default_config().block_sizes
    for seed in seeds:
        # The pointwise phase's axes keep their defaults in every seed.
        assert seed.block_sizes[:2] == defaults[:2]
    stages = _sources(bound.to_code(seeds[0]))
    assert len(stages) == 3
    pointwise, gemm_c, gemm_x = stages
    assert "cute.gemm(" not in pointwise and "tcgen05" not in pointwise
    assert all(
        "cute.gemm(" in stage and "tcgen05_" in stage for stage in (gemm_c, gemm_x)
    )
    # grad_c = acc.to(fp16) * (c[tile] > 0): the comparison and the boolean
    # promotion render inside the tcgen05 epilogue chain of the first GEMM.
    assert " > 0.0" in gemm_c and ".to(cutlass.Float32)" in gemm_c
    # grad_x = grad_out * d + acc.to(fp16): a product of two exact-shape aux
    # loads on the carrier's other side.
    assert "tcgen05_aux_product" in gemm_x


def test_relu_backward_mask_is_a_whitelisted_epilogue_chain() -> None:
    bound = _bind_bwd_dx()
    host = bound.host_function
    assert host is not None
    graphs = host.device_ir.graphs
    mma_nodes = [
        node
        for info in graphs
        for node in info.graph.nodes
        if node.target is torch.ops.aten.addmm.default
    ]
    assert len(mma_nodes) == 2
    chains = analyze_tcgen05_matmul_store_chains(graphs, mma_nodes[0])
    assert chains is not None and len(chains) == 1
    _store, chain = chains[0]
    (leaf,) = chain.auxiliary_tensor_loads
    names = iter(f"local_{i}" for i in range(16))
    prelude, final = chain.render_prelude_and_expr(
        "acc", lambda prefix: f"{prefix}_{next(names)}", "", {leaf: "aux_c"}
    )
    assert "(aux_c) > (0.0)" in prelude
    assert ".to(cutlass.Float32)" in prelude
    # fp16 rounding of the product, as the FX graph's mul.Tensor result dtype.
    assert ".to(cutlass.Float16).to(cutlass.Float32)" in prelude
    assert final.startswith("tcgen05_chain_step")


def test_single_gemm_barrier_chain_plans_per_phase() -> None:
    @helion.kernel(static_shapes=True)
    def scale_then_mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        m, k = x.size()
        n = w.size(1)
        scaled = torch.empty_like(x)
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        for row, col in hl.tile([m, k]):
            scaled[row, col] = x[row, col] * 2.0
        hl.barrier()
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, scaled[tile_m, tile_k], w[tile_k, tile_n])
            out[tile_m, tile_n] = acc.to(x.dtype)
        return out

    x = torch.randn(256, 128, dtype=torch.float16)
    w = torch.randn(128, 192, dtype=torch.float16)
    bound = _bind(scale_then_mm.fn, (x, w))
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert spec.cute_tcgen05_search_enabled
    # One GEMM phase beside a plain phase is an ordered chain like any
    # other: the GEMM launch owns the one region triple (not the standalone
    # single-matrix projection) and may run persistent / two-CTA.
    assert spec._tcgen05_matmul_block_ids() is None
    assert state.materialized_matmul_block_ids == ((2, 3, 4),)
    assert state.materialized_matmul_region_indices == (1,)
    assert state.materialized_matmul_shapes == ((256, 192, 128),)
    assert state.region_pid_type_choices() == (
        "inherit",
        "persistent_blocked",
        "persistent_interleaved",
    )
    # The matmul-free phase is its own flat launch: its SIMT layout knobs
    # stay on the surface while the GEMM axes keep their auto layout.
    assert spec.allowed_pid_types == ("flat",)
    assert spec.cute_pointwise_region_block_ids == frozenset({0, 1})
    assert spec.cute_pointwise_region_grid_groups == ((0, 1),)
    keys = set(ConfigGeneration(spec)._key_to_flat_indices)
    assert {
        "num_threads",
        "cute_vector_widths",
        "cute_lane_layouts",
        "tcgen05_region_pid_types",
        "tcgen05_region_ab_stages",
        "tcgen05_region_c_stages",
    } <= keys
    assert "tcgen05_materialized_pdl" not in keys
    for block_id in (2, 3, 4):
        assert spec.num_threads.block_id_lookup(block_id)._fragment(
            spec
        ).search_values() == [0]
        assert spec.cute_vector_widths.block_id_lookup(block_id)._fragment(
            spec
        ).search_values() == [1]


def _bind_bwd_dc(**kwargs: object) -> BoundKernel[Any]:
    t = _bwd_args(**kwargs)  # type: ignore[arg-type]
    return _bind(
        squeeze_and_excitation_net_bwd_dc.fn,
        (t["grad_out"], t["x"], t["b"], t["c"], t["d"]),
    )


def test_single_gemm_chain_pointwise_phase_seeds_and_renders_vector_packets() -> None:
    # grad_cb = grad_out * x * d * (1 - d) beside one native GEMM: the phase
    # is matmul-free, so the vectorized producer layout is seeded and the
    # pointwise launch renders 16-byte packets (the duplicate ``d`` load is
    # one hoisted packet) while the GEMM launch stays scalar-free of them.
    bound = _bind_bwd_dc()
    host = bound.host_function
    assert host is not None
    spec = bound.config_spec
    assert spec._cute_tcgen05_config.materialized_matmul_block_ids == ((2, 3, 4),)
    assert spec.cute_pointwise_region_block_ids == frozenset({0, 1})
    assert spec.cute_pointwise_region_grid_groups == ((0, 1),)
    with bound.env, host:
        assert CuteMaterializedOperandHeuristic.is_eligible(bound.env, host.device_ir)
        seeds = CuteMaterializedOperandHeuristic.get_seed_configs(
            bound.env, host.device_ir
        )
    assert seeds
    seed = seeds[0]
    # One 16-byte packet per thread along the contiguous ``n`` axis.
    assert (
        spec.cute_vector_widths.config_get(
            cast("list[int]", seed.config["cute_vector_widths"]), 1
        )
        == 8
    )
    assert (
        spec.cute_vector_widths.config_get(
            cast("list[int]", seed.config["cute_vector_widths"]), 0
        )
        == 1
    )
    num_threads = cast("list[int]", seed.config["num_threads"])
    column_threads = spec.num_threads.config_get(num_threads, 1)
    assert column_threads is not None
    assert seed.block_sizes[1] // column_threads == 8
    assert seed.block_sizes[0] == spec.num_threads.config_get(num_threads, 0)
    assert "cute_pointwise_pid_type" not in seed.config
    normalized = bound._normalized_config_copy(
        helion.Config.from_dict(spec.default_config().config | seed.config)
    )
    generation = ConfigGeneration(spec)
    assert generation.unflatten(
        generation.flatten(normalized)
    ) == with_flat_min_blocks_default(spec, normalized)
    pointwise, gemm = _sources(bound.to_code(normalized))
    assert "cute.gemm(" not in pointwise and "tcgen05" not in pointwise
    assert pointwise.count("ir.VectorType.get([8], cutlass.Uint16.mlir_type)") == 3
    assert pointwise.count("cute.arch.load(") == 3
    assert "_cute_store_u16_vec(" in pointwise
    assert ").load()" not in pointwise
    assert "cute.gemm(" in gemm and "VectorType" not in gemm


def test_chain_pointwise_phase_without_a_contiguous_axis_seeds_no_packets() -> None:
    @helion.kernel(static_shapes=True)
    def transpose_then_mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        m, k = x.size()
        n = w.size(1)
        scaled = torch.empty_like(x)
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        for row, col in hl.tile([m, k]):
            scaled[row, col] = x[col, row] * 2.0
        hl.barrier()
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, scaled[tile_m, tile_k], w[tile_k, tile_n])
            out[tile_m, tile_n] = acc.to(x.dtype)
        return out

    x = torch.randn(256, 256, dtype=torch.float16)
    w = torch.randn(256, 192, dtype=torch.float16)
    bound = _bind(transpose_then_mm.fn, (x, w))
    host = bound.host_function
    assert host is not None
    spec = bound.config_spec
    # The phase is still a separately tunable SIMT launch...
    assert spec.cute_pointwise_region_block_ids == frozenset({0, 1})
    with bound.env, host:
        seeds = CuteMaterializedOperandHeuristic.get_seed_configs(
            bound.env, host.device_ir
        )
    # ...but the load and the store disagree on the stride-1 axis, so no
    # packet layout is seeded (the per-site proof would reject one of them).
    assert seeds == []


def _one_gemm_chain_pair_config(
    bound: BoundKernel[Any], **overrides: object
) -> helion.Config:
    """The dc chain at the campaign shape: grad_c as a 256x128x64 CTA-pair
    launch with an 8-deep ring, the pointwise phase as 16-byte packets."""
    spec = bound.config_spec
    base = dict(spec.default_config().config)
    base.update(
        block_sizes=[1, 1024, 256, 128, 64],
        pid_type="flat",
        num_threads=[1, 128, 0, 0, 0],
        cute_vector_widths=[1, 8, 1, 1, 1],
        tcgen05_region_pid_types=["persistent_blocked"],
        tcgen05_cta_group="two",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=1,
        tcgen05_ab_stages=2,
        tcgen05_region_ab_stages=[8],
        tcgen05_region_c_stages=[2],
        tcgen05_aux_load_placement="pre_acc_wait",
    )
    base.update(overrides)
    return helion.Config.from_dict(base)


def test_one_gemm_chain_two_cta_launch_and_deep_flat_ring_at_campaign_shape() -> None:
    # squeeze_and_excitation_net backward at (4096, 512, 1024): grad_cb is a
    # 4096x1024 pointwise phase, grad_c a 4096x512x1024 GEMM with the
    # relu-backward mask.  The chain exposes the per-launch pid types, the
    # two-CTA coordinates and the pointwise layout knobs; the nvjet-like
    # pair program is strictly valid, seeded, and renders per launch.
    bound = _bind_bwd_dc(shape=(4096, 1024, 512))
    host = bound.host_function
    assert host is not None
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert state.materialized_matmul_shapes == ((4096, 512, 1024),)
    assert spec.allowed_pid_types == ("flat",)
    fragments = state.optional_fragments(for_search=True)
    cta_group = fragments["tcgen05_cta_group"]
    assert isinstance(cta_group, EnumFragment)
    assert cta_group.choices == ("auto", "two")
    ab_fragment = fragments["tcgen05_ab_stages"]
    assert isinstance(ab_fragment, IntegerFragment)
    # The flat launch searches the standalone surface's ring depth (12 for
    # 16-bit operands); its tile clamps it to the one-CTA SMEM fit.
    assert (ab_fragment.low, ab_fragment.high) == (1, 12)
    pid_types = fragments["tcgen05_region_pid_types"]
    assert isinstance(pid_types, ListOf) and pid_types.length == 1
    pair = _one_gemm_chain_pair_config(bound)
    normalized = _strict(bound, pair)
    for key in (
        "block_sizes",
        "num_threads",
        "cute_vector_widths",
        "tcgen05_region_pid_types",
        "tcgen05_region_ab_stages",
        "tcgen05_cta_group",
        "tcgen05_cluster_m",
    ):
        assert normalized.config[key] == pair.config[key], key
    assert _repaired(bound, pair).config == normalized.config
    generation = ConfigGeneration(spec)
    assert generation.unflatten(
        generation.flatten(normalized)
    ) == with_flat_min_blocks_default(spec, normalized)
    # The deep flat ring of the campaign winner (128x128x64, ab=5, c=4) stays
    # expressible; 7 stages overflow the one-CTA fit (6) and are refused
    # strictly, clamped by the search.  A flat launch has no tile scheduler,
    # so the L2 swizzle is dead on it and normalizes to 1; the persistent
    # pair launch rasters under it (four N tiles of 128 admit 4).
    flat = helion.Config.from_dict(
        spec.default_config().config
        | {
            "block_sizes": [512, 32, 128, 128, 64],
            "tcgen05_ab_stages": 5,
            "tcgen05_c_stages": 4,
            "tcgen05_l2_swizzle_size": 4,
        }
    )
    assert _strict(bound, flat).config["tcgen05_ab_stages"] == 5
    assert _strict(bound, flat).config["tcgen05_l2_swizzle_size"] == 1
    swizzled_pair = _one_gemm_chain_pair_config(bound, tcgen05_l2_swizzle_size=4)
    assert _strict(bound, swizzled_pair).config["tcgen05_l2_swizzle_size"] == 4
    deep = helion.Config.from_dict(flat.config | {"tcgen05_ab_stages": 7})
    with pytest.raises(helion.exc.InvalidConfig, match="maximum AB stages: 6"):
        _strict(bound, deep)
    assert _repaired(bound, deep).config["tcgen05_ab_stages"] == 6
    # Seeds: the chain's own two-CTA recipe at both K tiles (the geometry
    # rule's 128x256 pair tile, the deepest ring its budget fits), each
    # paired by the operand heuristic with the vectorized pointwise layout.
    with bound.env, host:
        native = CuteMaterializedMmaHeuristic.get_seed_configs(
            bound.env, host.device_ir
        )
        paired = CuteMaterializedOperandHeuristic.get_seed_configs(
            bound.env, host.device_ir
        )
    two_cta = [seed for seed in native if "tcgen05_region_pid_types" in seed.config]
    # Beside each pair seed, three twins move the masked GEMM onto the
    # 256-row pair tile the epilogue mechanisms render on and prefetch the
    # relu mask: the register ring at its deepest depth, the whole-fragment
    # hoist and the epilogue-issued aux ring, each at the deepest AB ring
    # the region budget fits beside the launch's rings; the hoist alone
    # also takes the geometry rule's own 128x256 pair.
    assert [seed.block_sizes[2:] for seed in two_cta] == [
        [128, 256, 64],
        [256, 128, 64],
        [256, 128, 64],
        [128, 256, 64],
        [256, 128, 64],
        [128, 256, 128],
        [256, 128, 128],
        [256, 128, 128],
        [128, 256, 128],
        [256, 128, 128],
    ]
    assert two_cta[0].config["tcgen05_region_ab_stages"] == [8]
    assert [
        (
            seed.config.get("tcgen05_aux_prefetch_subtiles", 0),
            seed.config.get("tcgen05_aux_fragment_hoist", False),
            seed.config.get("tcgen05_aux_load_mode", "simt"),
            seed.config["tcgen05_region_ab_stages"],
        )
        for seed in two_cta
    ] == [
        (0, False, "simt", [8]),
        (3, False, "simt", [8]),
        (0, True, "simt", [8]),
        (0, True, "simt", [8]),
        (0, False, "epi_tma", [8]),
        (0, False, "simt", [4]),
        (3, False, "simt", [4]),
        (0, True, "simt", [4]),
        (0, True, "simt", [4]),
        (0, False, "epi_tma", [4]),
    ]
    assert all(seed.config["tcgen05_cta_group"] == "two" for seed in two_cta)
    assert native[0].pid_type == "flat"
    assert "tcgen05_region_pid_types" not in native[0].config
    vectorized = [seed for seed in paired if "tcgen05_region_pid_types" in seed.config]
    assert len(vectorized) == len(two_cta)
    for seed in vectorized:
        assert seed.block_sizes[:2] == [1, 1024]
        assert seed.config["cute_vector_widths"] == [1, 8, 1, 1, 1]
        assert seed.config["num_threads"] == [1, 128, 0, 0, 0]
    # Rendering: the GEMM launch is the CtaGroup.TWO persistent pipeline
    # under a (2, 1, 1) cluster with the mask epilogue; the pointwise launch
    # is three 16-byte packets and a vector store.
    pointwise, gemm = _sources(bound.to_code(normalized))
    assert "CtaGroup.TWO" in gemm
    assert "_helion_cute_cluster_shape = (2, 1, 1)" in gemm
    assert " > 0.0" in gemm
    assert pointwise.count("ir.VectorType.get([8], cutlass.Uint16.mlir_type)") == 3
    assert "_cute_store_u16_vec(" in pointwise and ").load()" not in pointwise
    assert "VectorType" not in gemm
    flat_pointwise, flat_gemm = _sources(bound.to_code(_strict(bound, flat)))
    assert "CtaGroup.ONE" in flat_gemm and "CtaGroup.TWO" not in flat_gemm
    assert "_helion_cute_cluster_shape" not in flat_gemm
    assert "VectorType" not in flat_pointwise


def test_chain_flat_ring_admission_is_the_standalone_bound() -> None:
    # A flat GEMM launch of a chain admits the ring the standalone
    # single-GEMM surface admits for its tile (the AB ring within the per-CTA
    # capacity less the 28 KiB reservation: 4 stages of 128x32x128, 5 of
    # 128x16x128, 6 of 128x128x64 fp16), tightened by the one-CTA model where
    # the output ring matters (128x256x64: 4 beside a 2-stage ring, 3 beside
    # a 4-stage one).  At bm=128/bk=128 the deeper 16-bit rings the bare
    # capacity would hold (128x32x128 ab5, 128x16x128 ab6) changed the GEMM's
    # rounding on B200 and are refused.
    bound = _bind_bwd_dc(shape=(4096, 1024, 512))
    base = dict(bound.config_spec.default_config().config)

    def flat(tiles: list[int], **overrides: object) -> helion.Config:
        return helion.Config.from_dict(
            base | {"block_sizes": [512, 32, *tiles]} | overrides
        )

    for tiles, c_stages, limit in (
        ([128, 32, 128], 2, 4),
        ([128, 16, 128], 2, 5),
        ([128, 128, 64], 4, 6),
        ([128, 256, 64], 2, 4),
        ([128, 256, 64], 4, 3),
    ):
        fits = flat(tiles, tcgen05_ab_stages=limit, tcgen05_c_stages=c_stages)
        assert _strict(bound, fits).config["tcgen05_ab_stages"] == limit, tiles
        deep = flat(tiles, tcgen05_ab_stages=limit + 1, tcgen05_c_stages=c_stages)
        with pytest.raises(
            helion.exc.InvalidConfig, match=f"maximum AB stages: {limit}"
        ):
            _strict(bound, deep)
        assert _repaired(bound, deep).config["tcgen05_ab_stages"] == limit, tiles
    # A tile that fits no ring takes the standalone surface's handling.
    huge = flat([256, 256, 256], tcgen05_ab_stages=4)
    with pytest.raises(helion.exc.InvalidConfig, match="only supported"):
        _strict(bound, huge)
    # The kernel's knob is bounded only by launches that read it: with the
    # one GEMM launch persistent under its own region entry it is dead.
    pair = _one_gemm_chain_pair_config(bound, tcgen05_ab_stages=12)
    assert _strict(bound, pair).config["tcgen05_ab_stages"] == 12
    assert _repaired(bound, pair).config["tcgen05_ab_stages"] == 12
    # An inherit launch's region entries are priced as codegen applies them:
    # its ``tcgen05_region_c_stages`` entry sizes the output ring its
    # ``tcgen05_region_ab_stages`` entry fits beside (grad_x 128x256x64).
    dx = _bind_bwd_dx(shape=(256, 256, 128))
    blocks = list(
        cast(
            "list[int]",
            _two_cta_chain_config(dx, ["persistent_blocked", "inherit"]).config[
                "block_sizes"
            ],
        )
    )
    blocks[5:8] = [128, 256, 64]
    for c_counts, limit in (([2, 2], 4), ([2, 4], 3)):
        fits = _two_cta_chain_config(
            dx,
            ["persistent_blocked", "inherit"],
            block_sizes=blocks,
            tcgen05_region_ab_stages=[4, limit],
            tcgen05_region_c_stages=c_counts,
        )
        assert _strict(dx, fits).config["tcgen05_region_ab_stages"] == [4, limit]
        deep = _two_cta_chain_config(
            dx,
            ["persistent_blocked", "inherit"],
            block_sizes=blocks,
            tcgen05_region_ab_stages=[4, limit + 1],
            tcgen05_region_c_stages=c_counts,
        )
        with pytest.raises(helion.exc.InvalidConfig, match="per-region SMEM"):
            _strict(dx, deep)
        assert _repaired(dx, deep).config["tcgen05_region_ab_stages"] == [4, limit]


def test_same_shape_gemm_phases_plan_per_phase() -> None:
    # Two GEMM phases of one shape used to share the standalone surface's
    # single tile triple (one analysis key); planned per phase they own
    # independent triples and the per-launch pid types like any chain.
    @helion.kernel(static_shapes=True)
    def mm_twice(x: torch.Tensor, w: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        m, k = x.size()
        n = w.size(1)
        y = torch.empty([m, n], dtype=x.dtype, device=x.device)
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
            y[tile_m, tile_n] = acc.to(x.dtype)
        hl.barrier()
        for tile_m2, tile_n2 in hl.tile([m, n]):
            acc2 = hl.zeros([tile_m2, tile_n2], dtype=torch.float32)
            for tile_k2 in hl.tile(k):
                acc2 = torch.addmm(acc2, y[tile_m2, tile_k2], v[tile_k2, tile_n2])
            out[tile_m2, tile_n2] = acc2.to(x.dtype)
        return out

    x = torch.randn(256, 256, dtype=torch.float16)
    w = torch.randn(256, 256, dtype=torch.float16)
    v = torch.randn(256, 256, dtype=torch.float16)
    bound = _bind(mm_twice.fn, (x, w, v))
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert spec._tcgen05_matmul_block_ids() is None
    assert state.materialized_matmul_block_ids == ((0, 1, 2), (3, 4, 5))
    assert state.materialized_matmul_shapes == ((256, 256, 256),) * 2
    assert spec.allowed_pid_types == ("flat",)
    assert state.region_pid_type_choices() is not None
    assert spec.cute_pointwise_region_block_ids == frozenset()


def test_unsupported_gemm_epilogue_declines_the_phase_surface() -> None:
    @helion.kernel(static_shapes=True)
    def two_gemms_with_a_reduction_epilogue(
        x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
    ) -> torch.Tensor:
        m, k = x.size()
        n1 = w1.size(1)
        n2 = w2.size(1)
        h = torch.empty([m, n1], dtype=x.dtype, device=x.device)
        out = torch.empty([m, n2], dtype=x.dtype, device=x.device)
        for tile_m, tile_n in hl.tile([m, n1]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], w1[tile_k, tile_n])
            # A row reduction of the accumulator is outside the epilogue chain.
            h[tile_m, tile_n] = (acc - acc.amax(dim=1, keepdim=True)).to(x.dtype)
        hl.barrier()
        for tile_m, tile_n in hl.tile([m, n2]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(n1):
                acc = torch.addmm(acc, h[tile_m, tile_k], w2[tile_k, tile_n])
            out[tile_m, tile_n] = acc.to(x.dtype)
        return out

    x = torch.randn(256, 128, dtype=torch.float16)
    w1 = torch.randn(128, 256, dtype=torch.float16)
    w2 = torch.randn(256, 64, dtype=torch.float16)
    bound = _bind(two_gemms_with_a_reduction_epilogue.fn, (x, w1, w2))
    spec = bound.config_spec
    assert not spec.cute_tcgen05_search_enabled
    assert spec._cute_tcgen05_config.materialized_matmul_block_ids == ()
    # The SIMT lowering stays, with the barrier kernel's ordinary pid types.
    assert "persistent_blocked" in spec.allowed_pid_types


def test_gemm_phase_with_an_independent_store_declines() -> None:
    @helion.kernel(static_shapes=True)
    def gemm_phase_with_side_store(
        x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m, k = x.size()
        n1 = w1.size(1)
        n2 = w2.size(1)
        h = torch.empty([m, n1], dtype=x.dtype, device=x.device)
        side = torch.empty([m, n1], dtype=x.dtype, device=x.device)
        out = torch.empty([m, n2], dtype=x.dtype, device=x.device)
        for tile_m, tile_n in hl.tile([m, n1]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], w1[tile_k, tile_n])
            h[tile_m, tile_n] = acc.to(x.dtype)
            # A store the matmul does not reach: its load has no lowering
            # beside a collective matmul, so the phase has no tcgen05 form.
            side[tile_m, tile_n] = c[tile_m, tile_n] * 2.0
        hl.barrier()
        for tile_m, tile_n in hl.tile([m, n2]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(n1):
                acc = torch.addmm(acc, h[tile_m, tile_k], w2[tile_k, tile_n])
            out[tile_m, tile_n] = acc.to(x.dtype)
        return out, side

    x = torch.randn(256, 128, dtype=torch.float16)
    w1 = torch.randn(128, 256, dtype=torch.float16)
    w2 = torch.randn(256, 64, dtype=torch.float16)
    c = torch.randn(256, 256, dtype=torch.float16)
    bound = _bind(gemm_phase_with_side_store.fn, (x, w1, w2, c))
    spec = bound.config_spec
    assert not spec.cute_tcgen05_search_enabled
    assert spec._cute_tcgen05_config.materialized_matmul_block_ids == ()
    assert "persistent_blocked" in spec.allowed_pid_types
    # The SIMT lowering stays live: the default config renders.
    assert "cute.gemm(" not in bound.to_code(spec.default_config())


def test_gemm_phase_with_an_atomic_declines() -> None:
    @helion.kernel(static_shapes=True)
    def gemm_phase_with_atomic(
        x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m, k = x.size()
        n1 = w1.size(1)
        n2 = w2.size(1)
        h = torch.empty([m, n1], dtype=x.dtype, device=x.device)
        total = torch.zeros([m, n1], dtype=torch.float32, device=x.device)
        out = torch.empty([m, n2], dtype=x.dtype, device=x.device)
        for tile_m, tile_n in hl.tile([m, n1]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], w1[tile_k, tile_n])
            h[tile_m, tile_n] = acc.to(x.dtype)
            hl.atomic_add(total, [tile_m, tile_n], c[tile_m, tile_n].to(torch.float32))
        hl.barrier()
        for tile_m, tile_n in hl.tile([m, n2]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(n1):
                acc = torch.addmm(acc, h[tile_m, tile_k], w2[tile_k, tile_n])
            out[tile_m, tile_n] = acc.to(x.dtype)
        return out, total

    x = torch.randn(256, 128, dtype=torch.float16)
    w1 = torch.randn(128, 256, dtype=torch.float16)
    w2 = torch.randn(256, 64, dtype=torch.float16)
    c = torch.randn(256, 256, dtype=torch.float16)
    bound = _bind(gemm_phase_with_atomic.fn, (x, w1, w2, c))
    spec = bound.config_spec
    assert not spec.cute_tcgen05_search_enabled
    assert spec._cute_tcgen05_config.materialized_matmul_block_ids == ()


def test_materialized_fission_region_indices_are_the_launch_order() -> None:
    bound = _bind(
        squeeze_and_excitation_net_fwd.fn, _se_args((128, 256, 64)), fission=True
    )
    state = bound.config_spec._cute_tcgen05_config
    assert state.materialized_matmul_block_ids == ((0, 1, 2), (3, 4, 5))
    assert state.materialized_matmul_region_indices == (0, 1)
    # The fission surface is unchanged: ``cute_pdl`` is the tcgen05 search
    # surface's own fragment (``CuteTcgen05Config.optional_fragments``, the
    # renderer places the wait), offered here exactly as before; with its
    # persistent walks the paired two-CTA schedule and its region knobs stay.
    keys = set(ConfigGeneration(bound.config_spec)._key_to_flat_indices)
    assert "cute_pdl" in keys
    assert "persistent_blocked" in bound.config_spec.allowed_pid_types
    assert {
        "tcgen05_region_ab_stages",
        "tcgen05_region_c_stages",
        "tcgen05_materialized_pdl",
    } <= keys
    fragments = state.optional_fragments(for_search=True)
    assert fragments["tcgen05_cluster_m"].choices == (1, 2)  # type: ignore[attr-defined]
    assert fragments["tcgen05_cta_group"].choices == ("auto", "two")  # type: ignore[attr-defined]


def test_collective_seeds_ignore_axes_of_contraction_free_phases() -> None:
    # Without tcgen05 the collective family seeds the barrier kernel's GEMM
    # phases; the pointwise phase's axes are not singleton axes it must veto.
    support = CuteMmaSupport(
        universal=True,
        warp_f16bf16=True,
        warpgroup_f16bf16=True,
        tcgen05_f16bf16=False,
    )
    with (
        patch(
            "helion._compiler.cute.mma_support.get_cute_mma_support",
            return_value=support,
        ),
        patch(
            "helion._compiler.cute.cute_mma.get_cute_mma_support", return_value=support
        ),
    ):
        bound = _bind_bwd_dx()
    spec = bound.config_spec
    assert not spec.cute_tcgen05_search_enabled
    host = bound.host_function
    assert host is not None
    with bound.env, host:
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, host.device_ir
        )
    assert seeds
    default_blocks = [
        spec.block_sizes.block_id_lookup(i)._fragment(spec).default() for i in (0, 1)
    ]
    unset_threads = [
        spec.num_threads.block_id_lookup(i)._fill_missing() for i in (0, 1)
    ]
    assert default_blocks != [1, 1]
    for seed in seeds:
        assert seed.config["cute_collective_mma"] is True
        # The pointwise phase's blocks keep their fragment defaults and its
        # threads are left unset, never forced to the singleton geometry.
        assert seed.block_sizes[:2] == default_blocks
        threads = seed.config["num_threads"]
        assert isinstance(threads, list) and threads[:2] == unset_threads
        # The GEMM phases carry the collective geometry.
        assert seed.block_sizes[2] in (32, 64, 128)


def _two_cta_chain_config(
    bound: BoundKernel[Any], region_pid_types: list[str], **overrides: object
) -> helion.Config:
    """A dx chain config: grad_c 128x128x64, grad_x 128x128x64 CTA-pair tiles."""
    spec = bound.config_spec
    base = dict(spec.default_config().config)
    blocks = list(cast("list[int]", base["block_sizes"]))
    blocks[2:8] = [128, 128, 64, 128, 128, 64]
    base.update(
        block_sizes=blocks,
        pid_type="flat",
        loop_orders=[[1, 0], [0, 1], [0, 1]],
        tcgen05_region_pid_types=region_pid_types,
        tcgen05_cta_group="two",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=1,
        tcgen05_region_ab_stages=[4, 2],
        tcgen05_aux_load_placement="pre_acc_wait",
    )
    base.update(overrides)
    return helion.Config.from_dict(base)


def _strict(bound: BoundKernel[Any], config: helion.Config) -> helion.Config:
    copy = helion.Config.from_dict(dict(config.config))
    bound.config_spec.normalize(copy)
    return copy


def _repaired(bound: BoundKernel[Any], config: helion.Config) -> helion.Config:
    copy = helion.Config.from_dict(dict(config.config))
    bound.config_spec.normalize(copy, _fix_invalid=True)
    return copy


def test_two_cta_region_launches_render_per_gemm_launch() -> None:
    bound = _bind_bwd_dx(shape=(256, 256, 128))
    state = bound.config_spec._cute_tcgen05_config
    assert state.materialized_matmul_shapes == ((256, 128, 256), (256, 256, 128))
    # A mixed chain: grad_c persistent two-CTA, grad_x left flat.  Strict
    # normalization keeps every coordinate, including the pointwise launch's
    # non-default loop order (only clustered launches bind their raster).
    mixed = _two_cta_chain_config(bound, ["persistent_blocked", "inherit"])
    normalized = _strict(bound, mixed)
    for key in (
        "tcgen05_region_pid_types",
        "tcgen05_region_ab_stages",
        "tcgen05_cta_group",
        "tcgen05_cluster_m",
        "loop_orders",
        "block_sizes",
    ):
        assert normalized.config[key] == mixed.config[key], key
    assert normalized.pid_type == "flat"
    assert normalized.config["tcgen05_persistence_model"] == "non_persistent"
    pointwise, gemm_c, gemm_x = _sources(bound.to_code(normalized))
    # The persistent GEMM launch: the CtaGroup.TWO instruction under a
    # (2, 1, 1) cluster and the static persistent tile walk; its epilogue
    # still carries the relu-backward mask.
    assert "CtaGroup.TWO" in gemm_c
    assert "_helion_cute_cluster_shape = (2, 1, 1)" in gemm_c
    assert " > 0.0" in gemm_c
    # The flat GEMM launch keeps the one-CTA pipeline, no cluster.
    assert "CtaGroup.ONE" in gemm_x and "CtaGroup.TWO" not in gemm_x
    assert "_helion_cute_cluster_shape" not in gemm_x
    # The pointwise launch renders exactly as under the plain flat chain.
    flat = dict(normalized.config)
    for key in (
        "tcgen05_region_pid_types",
        "tcgen05_region_ab_stages",
        "tcgen05_aux_load_placement",
    ):
        flat.pop(key)
    flat.update(tcgen05_cta_group="auto", tcgen05_cluster_m=1)
    flat_pointwise, flat_c, _flat_x = _sources(
        bound.to_code(_strict(bound, helion.Config.from_dict(flat)))
    )
    assert flat_pointwise == pointwise
    assert "CtaGroup.ONE" in flat_c
    # Both GEMM launches persistent, under either persistent pid type.
    for pid_type in ("persistent_blocked", "persistent_interleaved"):
        both = _strict(bound, _two_cta_chain_config(bound, [pid_type] * 2))
        assert both.config["tcgen05_region_pid_types"] == [pid_type] * 2
        stages = _sources(bound.to_code(both))
        assert len(stages) == 3
        assert stages[0] == pointwise
        assert all("CtaGroup.TWO" in stage for stage in stages[1:])


def test_region_pid_types_fail_closed() -> None:
    bound = _bind_bwd_dx(shape=(256, 256, 128))
    spec = bound.config_spec
    # All-inherit is the implicit default: the key and the clusters leave.
    inherit = _two_cta_chain_config(bound, ["inherit", "inherit"])
    with pytest.raises(helion.exc.InvalidConfig):
        _strict(bound, inherit)
    repaired = _repaired(bound, inherit)
    assert "tcgen05_region_pid_types" not in repaired.config
    assert repaired.config["tcgen05_cluster_m"] == 1
    assert repaired.config["tcgen05_cta_group"] == "auto"
    # The knob has one subject, the two-CTA region schedule.
    # A malformed list is a schema error under both normalization modes.
    for malformed in (["persistent_blocked"], ["persistent_blocked", "flat"]):
        with pytest.raises(helion.exc.InvalidConfig, match="tcgen05_region_pid_types"):
            _strict(
                bound,
                _two_cta_chain_config(
                    bound,
                    ["persistent_blocked"] * 2,
                    tcgen05_region_pid_types=malformed,
                ),
            )
    for overrides in (
        {"tcgen05_cta_group": "auto"},
        {"tcgen05_cluster_m": 1},
        {"tcgen05_cluster_n": 2},
    ):
        config = _two_cta_chain_config(bound, ["persistent_blocked"] * 2, **overrides)
        with pytest.raises(helion.exc.InvalidConfig):
            _strict(bound, config)
        repaired = _repaired(bound, config)
        assert "tcgen05_region_pid_types" not in repaired.config
        assert repaired.config["tcgen05_cluster_m"] == 1
        assert repaired.config["tcgen05_cluster_n"] == 1
        assert repaired.config["tcgen05_cta_group"] == "auto"
    # A cluster without the knob: the chain's flat launches refuse clusters.
    clustered = dict(spec.default_config().config)
    clustered.update(tcgen05_cluster_m=2)
    with pytest.raises(helion.exc.InvalidConfig, match="ordered hl.barrier"):
        _strict(bound, helion.Config.from_dict(clustered))
    assert (
        "tcgen05_region_pid_types"
        not in _repaired(bound, helion.Config.from_dict(clustered)).config
    )
    assert (
        _repaired(bound, helion.Config.from_dict(clustered)).config["tcgen05_cluster_m"]
        == 1
    )
    # The per-region ring depth is bounded by the region's complete budget
    # (two-CTA) or, for an ``inherit`` launch, its one-CTA SMEM fit
    # (128x128x64 fp16 beside a 2-stage output ring: 6 stages).
    deep = _two_cta_chain_config(
        bound, ["persistent_blocked", "inherit"], tcgen05_region_ab_stages=[16, 9]
    )
    with pytest.raises(helion.exc.InvalidConfig, match="per-region SMEM budgets"):
        _strict(bound, deep)
    repaired = _repaired(bound, deep)
    stages = repaired.config["tcgen05_region_ab_stages"]
    assert isinstance(stages, list) and stages[0] < 16 and stages[1] == 6
    # A persistent launch binds its raster: its non-default loop order is
    # refused, and only that launch's entry is repaired.
    raster = _two_cta_chain_config(
        bound, ["persistent_blocked", "inherit"], loop_orders=[[1, 0], [1, 0], [1, 0]]
    )
    with pytest.raises(helion.exc.InvalidConfig, match="loop_orders"):
        _strict(bound, raster)
    assert _repaired(bound, raster).config["loop_orders"] == [[1, 0], [0, 1], [1, 0]]
    # The fission plan's materialized-operand PDL edge has no subject here.
    pdl = _two_cta_chain_config(
        bound, ["persistent_blocked"] * 2, tcgen05_materialized_pdl=True
    )
    with pytest.raises(helion.exc.InvalidConfig, match="materialized"):
        _strict(bound, pdl)
    assert "tcgen05_materialized_pdl" not in _repaired(bound, pdl).config


def _chain_with_vector_phase(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The dx chain behind a 1-D vector phase: its roots and the
    ``loop_orders`` entries (one per multi-dim tile loop) no longer align."""
    m, n = x.size()
    k = a.size(1)
    grad_cb = torch.empty([m, n], dtype=x.dtype, device=x.device)
    grad_c = torch.empty([m, k], dtype=x.dtype, device=x.device)
    grad_x = torch.empty([m, n], dtype=x.dtype, device=x.device)
    w = torch.empty_like(v)
    for pv in hl.tile(v.size(0)):
        w[pv] = v[pv] * 2.0
    hl.barrier()
    for pm, pn in hl.tile((m, n)):
        grad_cb[pm, pn] = grad_out[pm, pn] * x[pm, pn] * d[pm, pn] * (1.0 - d[pm, pn])
    hl.barrier()
    for rc, cc in hl.tile((m, k)):
        acc = hl.zeros([rc, cc], dtype=torch.float32)
        for rn in hl.tile(n):
            acc = torch.addmm(acc, grad_cb[rc, rn], b[cc, rn].T)
        grad_c[rc, cc] = acc.to(x.dtype) * (c[rc, cc] > 0)
    hl.barrier()
    for rx, cx in hl.tile((m, n)):
        acc = hl.zeros([rx, cx], dtype=torch.float32)
        for rk in hl.tile(k):
            acc = torch.addmm(acc, grad_c[rx, rk], a[cx, rk].T)
        grad_x[rx, cx] = grad_out[rx, cx] * d[rx, cx] + acc.to(x.dtype)
    return grad_x, w


def test_clustered_launch_loop_orders_bind_by_block_ids() -> None:
    """A clustered GEMM launch's raster guard finds its ``loop_orders`` entry
    by the launch's (m, n) block ids, not by its root position: behind a 1-D
    phase (no ``LoopOrderSpec``) the roots are (2, 3) while the GEMM entries
    are 1 and 2."""
    t = _bwd_args(shape=(256, 256, 128))
    v = torch.randn(777, generator=torch.Generator().manual_seed(5), dtype=t["x"].dtype)
    bound = _bind(
        _chain_with_vector_phase,
        (t["grad_out"], t["x"], t["a"], t["b"], t["c"], t["d"], v),
    )
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert state.region_pid_type_choices() is not None
    assert state.materialized_matmul_region_indices == (2, 3)
    assert state.materialized_matmul_block_ids == ((3, 4, 5), (6, 7, 8))
    assert [tuple(item.block_ids) for item in spec.loop_orders] == [
        (1, 2),
        (3, 4),
        (6, 7),
    ]
    base = dict(spec.default_config().config)
    blocks = list(cast("list[int]", base["block_sizes"]))
    blocks[3:9] = [128, 128, 64, 128, 128, 64]
    base.update(
        block_sizes=blocks,
        pid_type="flat",
        tcgen05_cta_group="two",
        tcgen05_cluster_m=2,
        tcgen05_cluster_n=1,
        tcgen05_region_ab_stages=[4, 4],
        tcgen05_aux_load_placement="pre_acc_wait",
    )

    def chain(pid_types: list[str], loop_orders: list[list[int]]) -> helion.Config:
        return helion.Config.from_dict(
            base | {"tcgen05_region_pid_types": pid_types, "loop_orders": loop_orders}
        )

    both = ["persistent_blocked", "persistent_blocked"]
    # Either clustered GEMM launch's swapped raster is refused, and the
    # repair resets exactly that entry; the pointwise phase keeps its order.
    for swapped in ([[1, 0], [1, 0], [0, 1]], [[1, 0], [0, 1], [1, 0]]):
        with pytest.raises(helion.exc.InvalidConfig, match="clustered GEMM launch"):
            _strict(bound, chain(both, swapped))
        assert _repaired(bound, chain(both, swapped)).config["loop_orders"] == [
            [1, 0],
            [0, 1],
            [0, 1],
        ]
    assert _strict(bound, chain(both, [[1, 0], [0, 1], [0, 1]])).config[
        "loop_orders"
    ] == [[1, 0], [0, 1], [0, 1]]
    # An ``inherit`` launch is a flat launch (shared ring depth, entry 0):
    # its raster stays free.
    mixed = helion.Config.from_dict(
        chain(["persistent_blocked", "inherit"], [[1, 0], [0, 1], [1, 0]]).config
        | {"tcgen05_region_ab_stages": [4, 0]}
    )
    assert _strict(bound, mixed).config["loop_orders"] == [[1, 0], [0, 1], [1, 0]]
    # The entries are the specs over each clustered launch's (m, n) block ids.
    free = chain(both, [[1, 0], [0, 1], [0, 1]])
    assert state._clustered_launch_loop_order_entries(free.config) == [1, 2]
    assert state._clustered_launch_loop_order_entries(mixed.config) == [1]
    # Fail closed: a clustered launch without a unique entry over its (m, n)
    # block ids (here grad_c's spec is missing) yields no binding, and the
    # whole config then keeps the default raster.
    with patch.object(spec, "loop_orders", [spec.loop_orders[0], spec.loop_orders[2]]):
        assert state._clustered_launch_loop_order_entries(free.config) is None
    with patch.object(
        type(state), "_clustered_launch_loop_order_entries", return_value=None
    ):
        with pytest.raises(helion.exc.InvalidConfig, match="loop_orders"):
            _strict(bound, free)
        assert _repaired(bound, free).config["loop_orders"] == [[0, 1]] * 3


def test_flat_launch_recipe_stays_with_the_inherit_launch() -> None:
    """The chain's whole-kernel config carries the flat recipe of its
    ``inherit`` launches (a scheduler warp, a one-slot accumulator ring,
    direct stores -- the campaign's flat winners); the persistent launch
    receives the paired recipe by projection and renders exactly as under
    the plain monolithic whole-kernel config."""
    bound = _bind_bwd_dx(shape=(256, 256, 128))
    flat_recipe = {
        "tcgen05_strategy": "role_local_with_scheduler",
        "tcgen05_warp_spec_scheduler_warps": 1,
        "tcgen05_acc_stages": 1,
        "tcgen05_c_store_mode": "direct",
    }
    mixed = _strict(
        bound,
        _two_cta_chain_config(bound, ["persistent_blocked", "inherit"], **flat_recipe),
    )
    for key, value in flat_recipe.items():
        assert mixed.config[key] == value, key
    assert mixed.config["tcgen05_region_pid_types"] == ["persistent_blocked", "inherit"]
    pointwise, gemm_c, gemm_x = _sources(bound.to_code(mixed))
    plain = _strict(
        bound, _two_cta_chain_config(bound, ["persistent_blocked", "inherit"])
    )
    _plain_pointwise, plain_c, _plain_x = _sources(bound.to_code(plain))
    # The persistent launch does not see the flat recipe.
    assert gemm_c == plain_c
    assert "CtaGroup.TWO" in gemm_c
    # The inherit launch renders exactly as the flat chain with that recipe
    # (every store of dx loads an auxiliary operand, so the flat chain keeps
    # the ``pre_acc_wait`` placement too).
    flat = dict(mixed.config)
    for key in ("tcgen05_region_pid_types", "tcgen05_region_ab_stages"):
        flat.pop(key)
    flat.update(tcgen05_cta_group="auto", tcgen05_cluster_m=1)
    flat_pointwise, _flat_c, flat_x = _sources(
        bound.to_code(_strict(bound, helion.Config.from_dict(flat)))
    )
    assert flat_pointwise == pointwise
    assert flat_x == gemm_x


def test_region_two_cta_seeds_round_trip() -> None:
    bound = _bind_bwd_dx(shape=(256, 256, 128))
    host = bound.host_function
    assert host is not None
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    seeds = state.region_two_cta_seed_configs()
    # Per K tile (64 and 128, both dividing every GEMM's K): the
    # all-persistent seed and one seed per GEMM launch left flat.
    assert [seed.block_sizes[2:8] for seed in seeds] == [
        [128, 64, 64, 128, 64, 64]
    ] * 3 + [[128, 64, 128, 128, 64, 128]] * 3
    assert [seed.config["tcgen05_region_pid_types"] for seed in seeds] == [
        ["persistent_blocked", "persistent_blocked"],
        ["inherit", "persistent_blocked"],
        ["persistent_blocked", "inherit"],
    ] * 2
    for seed in seeds:
        assert seed.config["tcgen05_cta_group"] == "two"
        assert seed.config["tcgen05_aux_load_placement"] == "pre_acc_wait"
        stages = seed.config["tcgen05_region_ab_stages"]
        pid_types = seed.config["tcgen05_region_pid_types"]
        assert isinstance(stages, list) and isinstance(pid_types, list)
        # A flat launch keeps the shared ring depth; persistent ones their own.
        assert all(
            (stage == 0) == (pid_type == "inherit") and 0 <= stage <= 16
            for stage, pid_type in zip(stages, pid_types, strict=True)
        )
    with bound.env, host:
        ranked = CuteMaterializedMmaHeuristic.get_seed_configs(
            bound.env, host.device_ir
        )
    assert ranked[0].pid_type == "flat"
    assert "tcgen05_region_pid_types" not in ranked[0].config
    two_cta = [seed for seed in ranked if "tcgen05_region_pid_types" in seed.config]
    assert len(two_cta) == len(seeds)
    generation = ConfigGeneration(spec)
    for seed in two_cta:
        # Strict normalization keeps the recipe, and the flat search schema
        # carries every coordinate.
        normalized = bound._normalized_config_copy(
            helion.Config.from_dict(spec.default_config().config | seed.config)
        )
        assert (
            normalized.config["tcgen05_region_pid_types"]
            == seed.config["tcgen05_region_pid_types"]
        )
        assert normalized.config["tcgen05_cta_group"] == "two"
        assert generation.unflatten(
            generation.flatten(normalized)
        ) == with_flat_min_blocks_default(spec, normalized)


def test_region_two_cta_tile_rule() -> None:
    from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
    from helion.autotuner.config_fragment import BlockSizeFragment

    fragments = (BlockSizeFragment(64, 256, 128), BlockSizeFragment(16, 256, 128))
    tile = CuteTcgen05Config._region_two_cta_tile
    # Wide outputs: the widest machine-filling tile (B200: 74 CTA pairs).
    assert tile(4096, 512, fragments, pairs_per_wave=74) == (128, 256)
    assert tile(4096, 1024, fragments, pairs_per_wave=74) == (256, 256)
    assert tile(2048, 1024, fragments, pairs_per_wave=74) == (128, 256)
    # A narrow long-K output: the fullest small tile.
    assert tile(1024, 512, fragments, pairs_per_wave=74) == (128, 64)
    # Whole tiles only; nothing divides a 96-row output.
    assert tile(96, 512, fragments, pairs_per_wave=74) is None
    assert tile(256, 64, fragments, pairs_per_wave=74) == (128, 64)


def test_fission_plan_keeps_its_whole_kernel_pid_types() -> None:
    bound = _bind(
        squeeze_and_excitation_net_fwd.fn, _se_args((128, 256, 64)), fission=True
    )
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert state.region_pid_type_choices() is None
    assert "tcgen05_region_pid_types" not in set(
        ConfigGeneration(spec)._key_to_flat_indices
    )
    config = dict(spec.default_config().config)
    config["tcgen05_region_pid_types"] = ["persistent_blocked", "persistent_blocked"]
    with pytest.raises(helion.exc.InvalidConfig, match="ordered hl.barrier"):
        _strict(bound, helion.Config.from_dict(config))
    # (The fission plan's own two-CTA schedule is unchanged: persistent pid
    # types, ``tcgen05_cta_group='two'`` and the region ring depths.)
    assert (
        "tcgen05_region_pid_types"
        not in _repaired(bound, helion.Config.from_dict(config)).config
    )
