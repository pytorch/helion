"""CPU proofs for ``tcgen05_persistent_grid_cap`` on plain clustered GEMM grids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_binding import _forbid_cuda_work
from test._cute_dense_tcgen05 import cpu_codegen_target

import helion
from helion._compiler.cute.strategies import TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY
from helion._testing import skipUnlessBackends
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator


pytestmark = skipUnlessBackends(["cute"])


@pytest.fixture(autouse=True)
def _cpu_codegen() -> Iterator[None]:
    with cpu_codegen_target(), _forbid_cuda_work():
        yield


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def plain_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = hl.dot(a[rows, inner], b[inner, cols], acc=acc)
        out[rows, cols] = acc.to(a.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def bias_gelu_aux_mm(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The epilogue_subtiling 01 shape: pre-activation and gelu of one tile."""
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    aux = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        pre = acc * 1.25
        pre = pre + bias[cols]
        aux[rows, cols] = pre.to(a.dtype)
        out[rows, cols] = torch.nn.functional.gelu(pre).to(a.dtype)
    return out, aux


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def bias_residual_gelu_mm(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    """The epilogue_subtiling 00 shape: an exact-shape (source-C) aux load."""
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = torch.addmm(acc, a[rows, inner], b[inner, cols])
        val = acc * 1.25
        val = val + residual[rows, cols].to(torch.float32) * 0.5
        val = val + bias[cols]
        val = torch.nn.functional.gelu(val)
        out[rows, cols] = val.to(a.dtype)
    return out


def _args(m: int = 2048, n: int = 2048, k: int = 4096) -> tuple[torch.Tensor, ...]:
    # 2048x2048 outputs on 256x128 CtaGroup.TWO tiles: 64 2x2 clusters, more
    # than a B200 co-schedules (33), fewer than its 74 SM pairs.
    return (
        torch.empty((m, k), dtype=torch.float16),
        torch.empty((n, k), dtype=torch.float16).T,
    )


def _config(**overrides: object) -> helion.Config:
    values = {
        "block_sizes": [256, 128, 64],
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": 2,
        "tcgen05_cluster_n": 2,
        "tcgen05_ab_stages": 4,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 2,
    }
    values.update(overrides)
    return helion.Config(**values)


def test_resident_clusters_caps_the_launch_at_the_co_resident_cluster_count() -> None:
    bound = plain_mm._bind_isolated(_args())
    source = bound.to_code(
        _config(**{TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY: "resident_clusters"})
    )
    assert "_helion_cute_cluster_shape = (2, 2, 1)" in source
    assert (
        "_MAX_ACTIVE_CLUSTERS = helion.runtime.get_max_active_clusters(a.device, 4)"
        in source
    )
    assert ", _MAX_ACTIVE_CLUSTERS))" in source
    assert "_NUM_SM // 2" not in source
    # The persistent roles loop over the capped grid.
    assert "while tcgen05_role_local_0_work_tile.is_valid_tile:" in source


def test_sm_pairs_is_the_implicit_default() -> None:
    bound = plain_mm._bind_isolated(_args())
    for config in (_config(), _config(tcgen05_persistent_grid_cap="sm_pairs")):
        normalized = dict(config.config)
        bound.config_spec.normalize(normalized)
        # Validation fills the registered fragment with its default.
        assert normalized.get(TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY, "sm_pairs") == (
            "sm_pairs"
        )
        source = bound.to_code(config)
        assert "_MAX_ACTIVE_CLUSTERS" not in source
        assert "max(1, _NUM_SM // 2)" in source


@pytest.mark.parametrize(
    "override",
    (
        {"tcgen05_cluster_n": 1},
        {"pid_type": "flat"},
    ),
)
def test_resident_clusters_is_rejected_where_both_caps_agree(
    override: dict[str, object],
) -> None:
    bound = plain_mm._bind_isolated(_args())
    config = _config(tcgen05_persistent_grid_cap="resident_clusters", **override)
    with pytest.raises(helion.exc.InvalidConfig, match="tcgen05_persistent_grid_cap"):
        bound.to_code(config)
    fixed = dict(config.config)
    bound.config_spec.normalize(fixed, _fix_invalid=True)
    assert fixed.get(TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY, "sm_pairs") == "sm_pairs"


def test_search_space_carries_the_knob_with_cluster_n() -> None:
    bound = plain_mm._bind_isolated(_args())
    config = _config(tcgen05_persistent_grid_cap="resident_clusters")
    with bound.env:
        generator = ConfigGeneration(bound.config_spec)
        flat, canonical = generator.canonicalize_flat(generator.flatten(config))
        assert generator.unflatten(flat) == canonical
    assert (
        canonical.config[TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY] == "resident_clusters"
    )
    assert canonical.config["tcgen05_cluster_n"] == 2


def _is_multi_tile_seed(seed: helion.Config) -> bool:
    """The multi-tile two-CTA seeds: a narrow N pair tile on the interleaved walk.

    The paired-pipeline family seeds the same tile as ``cta_group="two"`` on
    ``persistent_blocked``; those are not under test here.
    """
    return (
        seed.block_sizes == [256, 128, 64]
        and seed.config.get("pid_type") == "persistent_interleaved"
        and seed.config.get("tcgen05_cta_group") == "auto"
        and seed.config.get("tcgen05_cluster_m") == 2
    )


def test_multi_tile_two_cta_seeds_cover_one_tile_per_cta_grids() -> None:
    # 2048x2048 on 256x256 pair tiles is 128 CTAs on the 152-SM target: one
    # tile per CTA.  The seeds add the 256x128 pair tile (256 CTAs) run
    # persistently, in a 2x2 cluster on the resident-cluster cap and as pairs.
    bound = plain_mm._bind_isolated(_args())
    seeds = [
        seed.config
        for seed in bound.config_spec.autotune_seed_configs()
        if _is_multi_tile_seed(seed)
    ]
    assert [seed["tcgen05_cluster_n"] for seed in seeds] == [2, 1]
    for seed in seeds:
        assert seed["tcgen05_cluster_m"] == 2
        assert seed["pid_type"] == "persistent_interleaved"
        assert seed["tcgen05_ab_stages"] >= 4
    assert seeds[0][TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY] == "resident_clusters"
    assert TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY not in seeds[1]
    with bound.env:
        generator = ConfigGeneration(bound.config_spec)
        for seed in bound.config_spec.autotune_seed_configs():
            if not _is_multi_tile_seed(seed):
                continue
            _flat, canonical = generator.canonicalize_flat(generator.flatten(seed))
            assert canonical.block_sizes == [256, 128, 64]
            assert (
                canonical.config["tcgen05_cluster_n"]
                == seed.config["tcgen05_cluster_n"]
            )
            # The pair seed carries the implicit default; validation fills it in.
            assert canonical.config.get(
                TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY, "sm_pairs"
            ) == seed.config.get(TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY, "sm_pairs")


def test_multi_tile_two_cta_seeds_skip_machine_filling_grids() -> None:
    # 4096x4096 on 256x256 pair tiles is 512 CTAs: already several tiles per
    # CTA, so the 256x256 seeds stand and no narrower multi-tile seed is added.
    bound = plain_mm._bind_isolated(_args(m=4096, n=4096))
    assert not [
        seed
        for seed in bound.config_spec.autotune_seed_configs()
        if _is_multi_tile_seed(seed)
    ]


def test_search_fix_returns_the_cap_to_default_when_cluster_n_is_repaired() -> None:
    # A one-CTA sample with N multicast is repaired to cluster_n=1 by the fix
    # pass; the knob it carried then names the same kernel twice and returns to
    # its default instead of surviving as a distinct search point.
    bound = plain_mm._bind_isolated(_args())
    config = _config(
        tcgen05_cluster_m=1,
        tcgen05_cluster_n=2,
        tcgen05_persistent_grid_cap="resident_clusters",
    )
    with bound.env:
        generator = ConfigGeneration(bound.config_spec)
        _flat, canonical = generator.canonicalize_flat(generator.flatten(config))
    assert canonical.config["tcgen05_cluster_n"] == 1
    assert canonical.config.get(TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY, "sm_pairs") == (
        "sm_pairs"
    )


def test_multi_tile_two_cta_seeds_share_a_two_output_epilogue() -> None:
    # The two-CTA fan-out is validated for the 256x128 16-bit tile, so the
    # seeds of a two-output kernel carry it (one TMEM read, one bias load).
    a, b = _args()
    bias = torch.empty((2048,), dtype=torch.float16)
    bound = bias_gelu_aux_mm._bind_isolated((a, b, bias))
    seeds = [
        seed.config
        for seed in bound.config_spec.autotune_seed_configs()
        if _is_multi_tile_seed(seed)
    ]
    assert [seed["tcgen05_cluster_n"] for seed in seeds] == [2, 1]
    assert all(seed["tcgen05_epilogue_fanout"] == "shared" for seed in seeds)
    normalized = dict(seeds[0])
    bound.config_spec.normalize(normalized)
    assert normalized["tcgen05_epilogue_fanout"] == "shared"
    assert normalized[TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY] == "resident_clusters"


def test_exact_shape_aux_kernels_keep_the_narrow_tile_under_simt_aux_loads() -> None:
    # A residual (exact-shape aux) epilogue reads its tile per subtile with
    # SIMT loads, so the 256x128 two-CTA tile is seeded and survives the search
    # projection; the aux-TMA producer keeps the validated 256x256 regime.
    a, b = _args()
    bias = torch.empty((2048,), dtype=torch.float16)
    residual = torch.empty((2048, 2048), dtype=torch.float16)
    bound = bias_residual_gelu_mm._bind_isolated((a, b, bias, residual))
    state = bound.config_spec._cute_tcgen05_config
    assert state.exact_shape_aux_kernel_detected
    seeds = [
        seed.config
        for seed in bound.config_spec.autotune_seed_configs()
        if _is_multi_tile_seed(seed)
    ]
    assert [seed["tcgen05_cluster_n"] for seed in seeds] == [2, 1]
    assert all("tcgen05_epilogue_fanout" not in seed for seed in seeds)
    config = _config(
        tcgen05_ab_stages=8, tcgen05_persistent_grid_cap="resident_clusters"
    )
    with bound.env:
        generator = ConfigGeneration(bound.config_spec)
        _flat, canonical = generator.canonicalize_flat(generator.flatten(config))
        assert canonical.block_sizes == [256, 128, 64]
        assert canonical.config["tcgen05_cluster_n"] == 2
        assert canonical.config[TCGEN05_PERSISTENT_GRID_CAP_CONFIG_KEY] == (
            "resident_clusters"
        )
        tma = dict(config.config)
        tma["tcgen05_aux_load_mode"] = "tma"
        _flat, projected = generator.canonicalize_flat(
            generator.flatten(helion.Config(**tma))
        )
        assert projected.block_sizes[:2] == [256, 256]
    source = bound.to_code(canonical)
    assert (
        "_MAX_ACTIVE_CLUSTERS = helion.runtime.get_max_active_clusters(a.device, 4)"
        in source
    )
