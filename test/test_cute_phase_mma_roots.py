"""Per-phase tcgen05 planning for ``hl.barrier()`` kernels with several GEMMs.

CPU render pins: the barrier phases of ``squeeze_and_excitation_net_bwd_dx`` /
``_bwd_da`` are separate ordered launches, so each GEMM phase owns its own
tile triple (``materialized_matmul_block_ids``) and seeds, the pointwise phase
keeps its SIMT layout knobs, and the relu-backward mask epilogue
``acc.to(fp16) * (c[tile] > 0)`` splices into the tcgen05 epilogue chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_bwd_da
from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_bwd_dx
from examples.squeeze_and_excitation_net import squeeze_and_excitation_net_fwd
import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
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
from helion._compiler.cute.aux_tensor import analyze_tcgen05_matmul_store_chains
from helion._compiler.cute.mma_support import CuteMmaSupport
from helion._testing import skipUnlessBackends
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
        squeeze_and_excitation_net_bwd_dx.fn,
        (t["grad_out"], t["x"], t["a"], t["b"], t["c"], t["d"]),
    )


def _bind_bwd_da(**kwargs: object) -> BoundKernel[Any]:
    t = _bwd_args(**kwargs)  # type: ignore[arg-type]
    return _bind(
        squeeze_and_excitation_net_bwd_da.fn,
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
    # Clustered / two-CTA codegen and the per-region pipeline knobs whose
    # schedule requires them need a persistent launch: a flat-only surface
    # does not sample them (every such config fails at codegen).
    fragments = state.optional_fragments(for_search=True)
    assert fragments["tcgen05_cluster_m"].choices == (1,)  # type: ignore[attr-defined]
    assert fragments["tcgen05_cta_group"].choices == ("auto",)  # type: ignore[attr-defined]
    assert not keys & {
        "tcgen05_region_ab_stages",
        "tcgen05_region_c_stages",
        "tcgen05_materialized_pdl",
    }
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


def test_single_gemm_barrier_kernel_keeps_the_single_triple_surface() -> None:
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
    assert spec.cute_tcgen05_search_enabled
    assert spec._tcgen05_matmul_block_ids() == (2, 3, 4)
    assert spec._cute_tcgen05_config.materialized_matmul_block_ids == ()
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
