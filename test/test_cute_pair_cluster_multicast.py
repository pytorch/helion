"""The 2x2 (4-CTA) multicast cluster of the paired tcgen05 GEMM.

``tcgen05_cluster_n=2`` on the CTA pair (``tcgen05_cluster_m=2``) multicasts
each A half across the two pairs of a cluster that share an M tile and walk
adjacent N tiles; B stays split inside each pair.  The render is the pair's
with the cluster lines changed (cluster layout, empty-barrier arrive count,
V-leader predicate, scheduler cluster shape, launch grid).  These tests pin
the admission rules (the 256-row pair tile, whole N-tile pairs, the
co-resident cluster cap), the seeds, the CPU render facts and, on an SM100
GPU, the numerics against torch and bitwise against the 2x1 pair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _target

import helion
from helion._compiler.cute.epilogue_fanout import FANOUT_CONFIG_KEY
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
from helion.exc import InvalidConfig
import helion.language as hl

if TYPE_CHECKING:
    from torch import Tensor

pytestmark = skipUnlessBackends(["cute"])

PAIR = {
    "pid_type": "persistent_interleaved",
    "tcgen05_cta_group": "auto",
    "tcgen05_cluster_m": 2,
    "tcgen05_acc_stages": 2,
    "tcgen05_c_stages": 2,
    "tcgen05_num_epi_warps": 4,
    "tcgen05_l2_swizzle_size": 1,
    "tcgen05_strategy": "role_local_monolithic",
    "tcgen05_layout_strategy": "default",
    "tcgen05_persistence_model": "static_persistent",
}
CLUSTER_LINES = (
    "cluster",
    "CooperativeGroup",
    "PersistentTileSchedulerParams",
    "_launcher(",
    "_MAX_ACTIVE_CLUSTERS",
)


def gemm(x: Tensor, w: Tensor) -> Tensor:
    m, k = x.size()
    n = w.size(1)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


def two_outputs(x: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    m, k = x.size()
    n = w.size(1)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    s = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        sigmoid_result = torch.sigmoid(acc.to(x.dtype))
        s[tile_m, tile_n] = sigmoid_result
        out[tile_m, tile_n] = (2.0 * sigmoid_result.to(torch.float32)).to(x.dtype)
    return out, s


def _kernel(fn):
    return helion.kernel(fn, backend="cute", static_shapes=True, autotune_effort="full")


def _bind(fn, m: int, n: int, k: int):
    return _kernel(fn)._bind_isolated(
        (
            torch.empty((m, k), dtype=torch.float16),
            torch.empty((k, n), dtype=torch.float16),
        )
    )


def _cluster_lines_only(code_a: str, code_b: str) -> set[str]:
    """Lines on one side only, asserted to be cluster lines; returns them."""
    lines_a = {line.strip() for line in code_a.splitlines()}
    lines_b = {line.strip() for line in code_b.splitlines()}
    changed = lines_a ^ lines_b
    assert changed
    assert all(any(marker in line for marker in CLUSTER_LINES) for line in changed), (
        changed
    )
    return changed


def _assert_2x2_render(code: str) -> None:
    """The cluster facts of a 2x2 render: launch dims, masks, arrive count."""
    assert "cute.make_layout((2, 2, 1))" in code
    assert "_helion_cute_cluster_shape = (2, 2, 1)" in code
    # Each CTA issues its share of the A box and multicasts it across the N
    # pair (mcast_mode=2); B's mask covers the M pair, self-only at V=2.
    assert (
        "tcgen05_a_mcast_mask = cute.nvgpu.cpasync.create_tma_multicast_mask("
        "tcgen05_cluster_layout_vmnk, tcgen05_block_in_cluster_coord_vmnk, "
        "mcast_mode=2)"
    ) in code
    assert (
        "tcgen05_b_mcast_mask = cute.nvgpu.cpasync.create_tma_multicast_mask("
        "tcgen05_cluster_layout_vmnk, tcgen05_block_in_cluster_coord_vmnk, "
        "mcast_mode=1)"
    ) in code
    # The empty barrier collects one tcgen05.commit per pair that read the
    # slot: num_mcast_ctas_a + num_mcast_ctas_b - 1 = 2.
    assert (
        "tcgen05_ab_pipeline_consumer_group = cutlass.pipeline.CooperativeGroup("
        "cutlass.pipeline.Agent.Thread, cutlass.Int32(2))"
    ) in code
    # Each pair's V-leader issues the MMA and the release.
    assert "block_idx_in_cluster()) % cutlass.Int32(2) == cutlass.Int32(0)" in code
    assert "_launcher(" in code
    launch = code[code.index("_launcher(") :]
    assert ", (2, 2, min(" in launch
    # The scheduler walks whole clusters.
    assert "PersistentTileSchedulerParams(" in code
    assert ", 1), (2, 2, 1))" in code


# --------------------------------------------------------------------------
# CPU: admission, repair, seeds and render facts (fake sm100 target)
# --------------------------------------------------------------------------


def test_plain_2x2_cluster_caps_at_the_co_resident_cluster_count() -> None:
    """A plain 2x2 persistent launch sizes its grid in whole co-resident clusters.

    64 pair tiles make 32 clusters (below the cap); 128 pair tiles make 64,
    above the 33 four-CTA clusters a B200 co-schedules, so the launch must
    read the device's count instead of ``_NUM_SM // 2`` (two hardware waves:
    14.43 vs 11.90 us on the 4096x1024x1024 256x128 pair tile).
    """
    with _target(), _mock_cuda_unavailable():
        bound = _bind(gemm, 4096, 1024, 1024)
        config = helion.Config(
            block_sizes=[256, 128, 128],
            tcgen05_cluster_n=2,
            tcgen05_ab_stages=4,
            **PAIR,
        )
        code = bound.to_code(bound.config_spec.normalized_config(config))
    _assert_2x2_render(code)
    assert (
        "_MAX_ACTIVE_CLUSTERS = helion.runtime.get_max_active_clusters(x.device, 4)"
        in code
    )
    assert "_NUM_SM // 2" not in code


def test_pair_fanout_admits_the_2x2_cluster() -> None:
    """The shared fanout keeps the CTA pair's N-pair multicast: same epilogue,
    a 2x2 cluster layout and the two-producer empty barrier."""
    with _target(), _mock_cuda_unavailable():
        bound = _bind(two_outputs, 4096, 1024, 1024)
        codes = {}
        for cluster_n in (1, 2):
            config = helion.Config(
                block_sizes=[256, 128, 128],
                tcgen05_cluster_n=cluster_n,
                tcgen05_ab_stages=4,
                **PAIR,
            )
            config.config[FANOUT_CONFIG_KEY] = "shared"
            normalized = bound.config_spec.normalized_config(config)
            assert normalized.config[FANOUT_CONFIG_KEY] == "shared"
            assert normalized.config["tcgen05_cluster_n"] == cluster_n
            codes[cluster_n] = bound.to_code(normalized)
    _assert_2x2_render(codes[2])
    # Two TMA-store rings (one per output) on both cluster shapes: the
    # epilogue is untouched; every line that exists on one side only is a
    # cluster line.
    for code in codes.values():
        assert code.count("cutlass.pipeline.PipelineTmaStore.create(num_stages=2") == 2
        assert "tcgen05_tma_store_atom_1" in code
    _cluster_lines_only(codes[1], codes[2])


def test_pair_fanout_refuses_the_2x2_cluster_on_an_odd_n_tile_count() -> None:
    with _target(), _mock_cuda_unavailable():
        bound = _bind(two_outputs, 4096, 640, 1024)
        config = helion.Config(
            block_sizes=[256, 128, 128],
            tcgen05_cluster_n=2,
            tcgen05_ab_stages=4,
            **PAIR,
        )
        config.config[FANOUT_CONFIG_KEY] = "shared"
        with pytest.raises(InvalidConfig, match=r"cdiv\(640, 128\) = 5 tiles is odd"):
            bound.config_spec.normalize(dict(config.config))
        repaired = dict(config.config)
        bound.config_spec.normalize(repaired, _fix_invalid=True)
        # The cluster is repaired, the fanout and the pair stay.
        assert repaired["tcgen05_cluster_n"] == 1
        assert repaired["tcgen05_cluster_m"] == 2
        assert repaired[FANOUT_CONFIG_KEY] == "shared"
        bound.config_spec.normalize(dict(repaired))


def test_plain_2x2_cluster_refuses_an_odd_n_tile_count_before_render() -> None:
    """Every 2x2 family walks whole pairs of N tiles.  The plain pair used to
    pass normalization and fail at render (``cute_mma``'s BackendUnsupported)
    after the search had spent a compile; now it fails closed with the reason
    and the search repairs the cluster."""
    with _target(), _mock_cuda_unavailable():
        bound = _bind(gemm, 4096, 1280, 1280)
        odd = helion.Config(
            block_sizes=[256, 256, 64],
            tcgen05_cluster_n=2,
            tcgen05_ab_stages=6,
            **PAIR,
        )
        with pytest.raises(InvalidConfig, match=r"cdiv\(1280, 256\) = 5 tiles is odd"):
            bound.config_spec.normalize(dict(odd.config))
        repaired = dict(odd.config)
        bound.config_spec.normalize(repaired, _fix_invalid=True)
        assert repaired["tcgen05_cluster_n"] == 1
        assert repaired["tcgen05_cluster_m"] == 2
        assert repaired["block_sizes"] == [256, 256, 64]
        bound.config_spec.normalize(dict(repaired))
        # Ten 128-wide N tiles pair up: admitted and rendered as a 2x2.
        even = helion.Config(
            block_sizes=[256, 128, 64],
            tcgen05_cluster_n=2,
            tcgen05_ab_stages=8,
            **PAIR,
        )
        normalized = bound.config_spec.normalized_config(even)
        assert normalized.config["tcgen05_cluster_n"] == 2
        _assert_2x2_render(bound.to_code(normalized))


def test_paired_pipeline_admits_the_2x2_cluster_and_renders_the_auto_pair() -> None:
    """``tcgen05_cta_group='two'`` admits ``tcgen05_cluster_n=2`` on the
    256-row pair tile with whole N-tile pairs; its render is byte-identical
    to the ``'auto'`` pair's 2x2 (the pin had no codegen behind it)."""
    with _target(), _mock_cuda_unavailable():
        bound = _bind(gemm, 4096, 1024, 1024)
        codes = {}
        for cta_group in ("two", "auto"):
            for cluster_n in (1, 2):
                config = helion.Config(
                    block_sizes=[256, 256, 64],
                    tcgen05_cluster_n=cluster_n,
                    tcgen05_ab_stages=6,
                    **{**PAIR, "tcgen05_cta_group": cta_group},
                )
                normalized = bound.config_spec.normalized_config(config)
                assert normalized.config["tcgen05_cta_group"] == cta_group
                assert normalized.config["tcgen05_cluster_n"] == cluster_n
                assert normalized.config["tcgen05_cluster_m"] == 2
                codes[cta_group, cluster_n] = bound.to_code(normalized)
    assert codes["two", 2] == codes["auto", 2]
    assert codes["two", 1] == codes["auto", 1]
    _assert_2x2_render(codes["two", 2])
    _cluster_lines_only(codes["two", 1], codes["two", 2])


@pytest.mark.parametrize(
    "shape,block_sizes,ab_stages,reason",
    (
        # The 128-row pair tile runs a 64-row CTA tile outside the envelope.
        (
            (4096, 1024, 1024),
            [128, 128, 64],
            8,
            "two-CTA 256-row pair tile with an even N tile count of an unbatched GEMM",
        ),
        # 1280 // 256 = 5 N tiles cannot pair (every 2x2 family's rule).
        ((4096, 1280, 1280), [256, 256, 64], 6, r"cdiv\(1280, 256\) = 5 tiles is odd"),
    ),
    ids=("128_row_pair_tile", "odd_n_tiles"),
)
def test_paired_pipeline_refuses_the_2x2_cluster_outside_its_envelope(
    shape: tuple[int, int, int], block_sizes: list[int], ab_stages: int, reason: str
) -> None:
    """Outside the envelope the strict refusal names the rule and the repair
    drops the cluster, not the family (the search keeps its pair sample)."""
    with _target(), _mock_cuda_unavailable():
        bound = _bind(gemm, *shape)
        config = helion.Config(
            block_sizes=block_sizes,
            tcgen05_cluster_n=2,
            tcgen05_ab_stages=ab_stages,
            **{**PAIR, "tcgen05_cta_group": "two", "pid_type": "persistent_blocked"},
        )
        with pytest.raises(InvalidConfig, match=reason):
            bound.config_spec.normalize(dict(config.config))
        repaired = dict(config.config)
        bound.config_spec.normalize(repaired, _fix_invalid=True)
        assert repaired["tcgen05_cta_group"] == "two"
        assert repaired["tcgen05_cluster_m"] == 2
        assert repaired["tcgen05_cluster_n"] == 1
        assert repaired["block_sizes"] == block_sizes
        bound.config_spec.normalize(dict(repaired))


def _paired_seeds(bound):
    return [
        seed
        for seed in bound.config_spec.autotune_seed_configs()
        if seed.config.get("tcgen05_cta_group") == "two"
    ]


def test_paired_seeds_carry_2x2_twins_of_the_deep_rings() -> None:
    """nvjet stages its 2x2 clusters 64 deep in K: the deepest bk=64 ring of
    every 256-row pair tile with whole N-tile pairs gets a cluster_n=2 twin
    (and nothing else does); each twin is a strict fixed point."""
    with _target(), _mock_cuda_unavailable():
        bound = _bind(gemm, 4096, 1024, 1024)
        seeds = _paired_seeds(bound)
        twins = [seed for seed in seeds if seed.config["tcgen05_cluster_n"] == 2]
        assert {
            (tuple(seed.block_sizes), seed.config["tcgen05_ab_stages"])
            for seed in twins
        } == {((256, 64, 64), 10), ((256, 128, 64), 8), ((256, 256, 64), 6)}
        for twin in twins:
            sibling = dict(twin.config) | {"tcgen05_cluster_n": 1}
            assert any(seed.config == sibling for seed in seeds)
            normalized = dict(twin.config)
            bound.config_spec.normalize(normalized)
            assert normalized["tcgen05_cluster_n"] == 2
            assert normalized["tcgen05_cta_group"] == "two"
        # Five 256-wide N tiles cannot pair: that twin is withheld, the
        # 128- and 64-wide ones (10 and 20 tiles) stay.
        bound = _bind(gemm, 4096, 1280, 1280)
        twins = [
            seed
            for seed in _paired_seeds(bound)
            if seed.config["tcgen05_cluster_n"] == 2
        ]
        assert {tuple(seed.block_sizes) for seed in twins} == {
            (256, 64, 64),
            (256, 128, 64),
        }


# --------------------------------------------------------------------------
# GPU: numerics against torch, bitwise against the 2x1 pair, launch repeats
# --------------------------------------------------------------------------


def _requires_sm100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")


def _gpu_inputs(m: int, n: int, k: int, seed: int = 1105) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((m, k), generator=generator, dtype=torch.float16) * 0.5
    w = torch.randn((k, n), generator=generator, dtype=torch.float16) * 0.5
    return x.to(DEVICE), w.to(DEVICE)


def _compile(bound, **values: object):
    config = helion.Config(**values)
    bound.config_spec.normalize(config)
    return config, bound.compile_config(config, allow_print=False)


AUTO_AND_TWO = (("auto", 1), ("auto", 2), ("two", 2))
TWO_ONLY = (("two", 1), ("two", 2))
GPU_SHAPES = (
    # (m, n, k), pair tile, ring, families: 4x4 128-wide tiles -> 4 clusters.
    ((1024, 512, 512), [256, 128, 64], 8, AUTO_AND_TWO),
    # Non-square, 2x6 256-wide tiles -> 6 clusters.
    ((512, 1536, 256), [256, 256, 64], 6, AUTO_AND_TWO),
    # 8x20 128-wide tiles -> 80 clusters, above the 33 co-resident ones:
    # the capped persistent grid walks several clusters per CTA.
    ((2048, 2560, 512), [256, 128, 128], 4, AUTO_AND_TWO),
    # The 64-wide pair tile at its deep ring, 3x16 tiles -> 24 clusters:
    # the paired pipeline's own geometry (the plain family's direct-entry
    # envelope stops at three stages there).
    ((768, 1024, 1024), [256, 64, 64], 10, TWO_ONLY),
)


@pytest.mark.parametrize(
    "shape,block_sizes,ab_stages,families",
    GPU_SHAPES,
    ids=("4_clusters", "non_square", "80_clusters_capped", "bn64"),
)
def test_gpu_pair_2x2_matches_torch_and_the_2x1_pair_bitwise(
    shape: tuple[int, int, int],
    block_sizes: list[int],
    ab_stages: int,
    families: tuple[tuple[str, int], ...],
) -> None:
    """The 2x2 cluster (A multicast, pair-split B) computes the same K order
    per CTA as the 2x1 pair: bitwise equal, and ``'two'`` equals ``'auto'``."""
    _requires_sm100()
    x, w = _gpu_inputs(*shape)
    ref = (x.float() @ w.float()).to(torch.float16)
    bound = _kernel(gemm)._bind_isolated((x, w))
    outputs = {}
    for cta_group, cluster_n in families:
        config, fn = _compile(
            bound,
            block_sizes=block_sizes,
            tcgen05_cluster_n=cluster_n,
            tcgen05_ab_stages=ab_stages,
            **{**PAIR, "tcgen05_cta_group": cta_group},
        )
        assert config.config["tcgen05_cluster_n"] == cluster_n
        assert config.config["tcgen05_cta_group"] == cta_group
        outputs[cta_group, cluster_n] = fn(x, w)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            outputs[cta_group, cluster_n], ref, rtol=1e-2, atol=5e-2
        )
    first = outputs[families[0]]
    for family in families[1:]:
        assert torch.equal(outputs[family], first), family


@pytest.mark.parametrize(
    "shape",
    ((1024, 1024, 512), (2048, 2560, 512)),
    ids=("16_clusters", "80_clusters_capped"),
)
def test_gpu_pair_fanout_2x2_matches_the_2x1_fanout_bitwise(
    shape: tuple[int, int, int],
) -> None:
    """The two-output shared fanout on the 2x2 cluster: both outputs bitwise
    equal to the 2x1 fanout's, and right against torch."""
    _requires_sm100()
    x, w = _gpu_inputs(*shape)
    s_ref = torch.sigmoid((x.float() @ w.float()).to(torch.float16))
    out_ref = (2.0 * s_ref.float()).to(torch.float16)
    bound = _kernel(two_outputs)._bind_isolated((x, w))
    outputs = {}
    for cluster_n in (1, 2):
        values = dict(
            block_sizes=[256, 128, 128],
            tcgen05_cluster_n=cluster_n,
            tcgen05_ab_stages=4,
            **PAIR,
        )
        values[FANOUT_CONFIG_KEY] = "shared"
        config, fn = _compile(bound, **values)
        assert config.config[FANOUT_CONFIG_KEY] == "shared"
        assert config.config["tcgen05_cluster_n"] == cluster_n
        outputs[cluster_n] = fn(x, w)
        torch.cuda.synchronize()
        out, s = outputs[cluster_n]
        torch.testing.assert_close(s, s_ref, rtol=1e-2, atol=2e-2)
        torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=5e-2)
    assert torch.equal(outputs[2][0], outputs[1][0])
    assert torch.equal(outputs[2][1], outputs[1][1])


def test_gpu_2x2_cluster_launch_repeats() -> None:
    """Many back-to-back launches of the capped 2x2 kernel (several clusters
    per CTA) on fresh inputs: every result matches the pair's, no hang."""
    _requires_sm100()
    m, n, k = 2048, 2560, 512
    x, w = _gpu_inputs(m, n, k)
    bound = _kernel(gemm)._bind_isolated((x, w))
    fns = {
        cluster_n: _compile(
            bound,
            block_sizes=[256, 128, 128],
            tcgen05_cluster_n=cluster_n,
            tcgen05_ab_stages=4,
            **PAIR,
        )[1]
        for cluster_n in (1, 2)
    }
    for round_index in range(8):
        x, w = _gpu_inputs(m, n, k, seed=round_index)
        expected = fns[1](x, w)
        results = [fns[2](x, w) for _ in range(8)]
        torch.cuda.synchronize()
        for result in results:
            assert torch.equal(result, expected)
