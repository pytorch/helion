"""The flex forward search surface's structural coverage design is complete
for every head dim the CuTe flash backend supports, and its compiler seeds
carry the measured list-walk bodies.

The autotuner refuses a surface whose deterministic coverage design cannot
realize one of a knob's searched values (``AutotuneError: incomplete CuTe
flash structural coverage design``) before measuring anything, so a knob
whose legal values depend on the head dim must offer only the values some
row of the surface realizes.  The tail split (``cute_flash_flex_tail_split``)
hands each unit ``head_dim_v / corr_tile_size / splits`` correction chunks:
at head dim 96 a split of 8 divides no chunk count of the searched
correction tiles (12 / 6 / 3) and must not be searched there (pass 6 of the
2026-10-16 campaign ERRORed on causal-d96 that way).

The surface is built with the options ``flex_search_surface`` derives for
``examples/flex_attention.py`` (captured from real binds: the ``[B,H,S,D]``
and the ``[B,S,H,D].transpose(1, 2)`` layouts, and a causal or a noop
``mask_mod``, produce identical options, so they are not axes here; a
``score_mod`` adds the score region, grouped-query heads set ``kv_group``)
for every head dim 16..256 step 16, the 4K plan and the 128K cursor-walk
plan, and the design is validated on CPU.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from helion._compiler.autotuner_heuristics.cute import CuteFlashAttentionHeuristic
from helion._compiler.backend import CuteBackend
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute import cute_flash
from helion._compiler.device_ir import DeviceIR
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec

if TYPE_CHECKING:
    import helion

HEAD_DIMS = tuple(range(16, 257, 16))
SM_COUNT = 148  # the B200's streaming multiprocessors: the tail split's seed

# (q_seq, kv_seq, batch, heads): the campaign's 4K cell and its 128K
# cursor-walk cell (1024 KV tiles: several bitmask words, the cursor).
PLANS = {
    "s4096": (4096, 4096, 4, 16),
    "s131072": (131072, 131072, 1, 8),
}
# The score plan's four surfaces: a mask-only plan (any ``mask_mod``, the
# BlockMask tables walked), a plan with a score region (a ``score_mod``
# other than the identity: softcap / ALiBi run the hook on every tile),
# torch's empty BlockMask under a mask region (the trivial walk: one KV
# block over the whole sequence, no list walk, the flex body), and torch's
# empty BlockMask with the identity ``score_mod`` (the dense flex plan,
# ``AttentionScorePlan.flex_dense``: the dense attention surface over the
# flex kernel's two tiles, ``_flex_dense_search_surface``).
WALKS = ("mask", "score", "trivial", "dense")


def flex_forward_spec(
    head_dim: int,
    *,
    q_seq: int,
    kv_seq: int,
    num_bh: int,
    heads: int,
    kv_group: int,
    walk: str = "mask",
    device_sm_count: int = SM_COUNT,
) -> ConfigSpec:
    """The search surface ``examples/flex_attention.py`` binds, as
    ``device_ir`` passes ``flex_search_surface``'s options to
    ``enable_cute_flash_search`` (the two tunable block sizes are the query
    and KV tiles; the unit batch / head axes are fixed)."""
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(10, 0))
    for block_id in (0, 1):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=128))
    score_region = walk == "score"
    trivial_walk = walk == "trivial"
    dense = walk == "dense"
    spec.enable_cute_flash_search(
        head_dim=head_dim,
        head_dim_v=None,
        num_kv=kv_seq // 128,
        num_bh=num_bh,
        tensor_4d_heads=heads,
        dtype=torch.bfloat16,
        block_size_targets={0: 128, 1: 128},
        is_causal=False,
        has_kv_tile_pruning=False,
        requires_ws_overlap=False,
        small_biased_candidate=False,
        standard_dense_output=False,
        standard_causal_output=False,
        tmem_rowmax_compatible=not score_region,
        causal_resident_compatible=False,
        output_requires_tma=False,
        supports_tensor_4d_tma=False,
        has_row_epilogue=False,
        plain_row_body=dense,
        has_score_modifiers=score_region,
        score_modifier_kinds=("dense",) if dense else ("flex",),
        device_sm_count=device_sm_count,
        kv_group=kv_group,
        fast_math=True,
        has_list_walk=not (trivial_walk or dense),
        flex_has_score_region=score_region,
        flex_has_mask_region=not dense,
        flex_trivial_walk=trivial_walk,
        q_seq=q_seq,
        kv_seq=kv_seq,
        flex_aux_smem_candidates=(),
        flex_prefetch_regs=(),
        flex_coalesce_steps=(),
        flex_hook_ops=4 if score_region else 0,
    )
    return spec


def attention_spec(
    *, causal: bool, head_dim: int = 128, num_kv: int = 32
) -> ConfigSpec:
    """The dense / causal attention kernels' surface (``examples/attention.py``:
    the unit batch-head tile ahead of the 128-row query and 128-column KV
    tiles, (1, 128, 128))."""
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(10, 0))
    for block_id, target in enumerate((1, 128, 128)):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=target))
    spec.enable_cute_flash_search(
        head_dim=head_dim,
        num_kv=num_kv,
        num_bh=64,
        dtype=torch.bfloat16,
        block_size_targets={0: 1, 1: 128, 2: 128},
        is_causal=causal,
        standard_dense_output=not causal,
        standard_causal_output=causal,
        causal_resident_compatible=causal,
        device_sm_count=SM_COUNT,
        fast_math=True,
    )
    return spec


def flex_backward_spec(head_dim: int) -> ConfigSpec:
    """The flex backward surface (``helion_flex_attention_backward_kernel``
    at b4 h32 kv8 s4096, causal BlockMask, captured from a real bind): two
    tunable tiles; the 2-CTA pair family is legal at head dim 128 only."""
    spec = ConfigSpec(backend=CuteBackend(), target_device_capability=(10, 0))
    for block_id in (0, 1):
        spec.block_sizes.append(BlockSizeSpec(block_id=block_id, size_hint=128))
    two_cta = head_dim == 128
    spec.enable_cute_flash_bwd_search(
        block_size_targets={0: 128, 1: 128},
        two_cta_allowed=two_cta,
        two_cta_table_steps=160 if two_cta else 0,
        persistent_allowed=True,
        flex=True,
        tile_groups=(8, 0, 2, 4, 16, 32),
        fast_math=True,
        aux_smem_candidates=(),
        ds_tmem_allowed=not two_cta,
        stats_mma_allowed=True,
        s_first_allowed=True,
        kv_stage_allowed=not two_cta,
        table_step_choices=(0,),
        has_mask=True,
        has_score=False,
        additive_bias=False,
    )
    return spec


# surface -> (spec builder, the seeds' block sizes)
SURFACES = {
    "attention_dense": (lambda: attention_spec(causal=False), [1, 128, 128]),
    "attention_causal": (lambda: attention_spec(causal=True), [1, 128, 128]),
    "flex_mask_s4096": (
        lambda: flex_forward_spec(
            128, q_seq=4096, kv_seq=4096, num_bh=64, heads=16, kv_group=1
        ),
        [128, 128],
    ),
    "flex_score_s4096": (
        lambda: flex_forward_spec(
            128, q_seq=4096, kv_seq=4096, num_bh=64, heads=16, kv_group=1, walk="score"
        ),
        [128, 128],
    ),
    "flex_gqa4_s131072": (
        lambda: flex_forward_spec(
            128, q_seq=131072, kv_seq=131072, num_bh=8, heads=8, kv_group=4
        ),
        [128, 128],
    ),
    "flex_d64_s4096": (
        lambda: flex_forward_spec(
            64, q_seq=4096, kv_seq=4096, num_bh=64, heads=16, kv_group=1
        ),
        [128, 128],
    ),
    "flex_dense_s4096": (
        lambda: flex_forward_spec(
            128, q_seq=4096, kv_seq=4096, num_bh=64, heads=16, kv_group=1, walk="dense"
        ),
        [128, 128],
    ),
    "flex_backward_d128": (lambda: flex_backward_spec(128), [128, 128]),
}


def recorded_compiler_seeds(spec: ConfigSpec) -> list[helion.Config]:
    """The seeds a bind records on the spec: the flash heuristic's (the seed
    builder, legalized; None off the forward flash surface) plus the spec's
    own ``autotune_seed_configs`` (the backward pair seeds live there), in
    ``BoundKernel``'s order with duplicates folded."""
    env = Mock(spec=CompileEnvironment)
    env.config_spec = spec
    heuristic = CuteFlashAttentionHeuristic.get_seed_configs(env, Mock(spec=DeviceIR))
    assert (heuristic is None) == (not spec.cute_flash_search_enabled)
    seeds = list(heuristic or ())
    for seed in spec.autotune_seed_configs():
        if seed not in seeds:
            seeds.append(seed)
    spec.compiler_seed_configs = seeds
    return seeds


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_compiler_seeds_transfer_on_every_flash_surface(surface: str) -> None:
    """Every flash surface the heuristic serves gets compiler seeds, and the
    search admits each one: ``seed_flat_config_pairs`` transfers the whole
    recorded set without a failure.  (The seed builder once accepted only
    the attention kernels' three block sizes and dropped every flex seed.)"""
    build, block_sizes = SURFACES[surface]
    spec = build()
    with patch.dict(os.environ, {}, clear=True):
        seeds = recorded_compiler_seeds(spec)
        assert seeds
        assert all(seed.config["block_sizes"] == block_sizes for seed in seeds)
        messages: list[str] = []
        pairs = ConfigGeneration(spec).seed_flat_config_pairs(messages.append)
    assert not messages
    assert len(pairs) == len(seeds)


@pytest.mark.parametrize("plan", sorted(PLANS))
@pytest.mark.parametrize("kv_group", (1, 4), ids=("plain", "gqa4"))
@pytest.mark.parametrize("walk", WALKS)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_flex_forward_coverage_design_is_complete(
    head_dim: int, walk: str, kv_group: int, plan: str
) -> None:
    q_seq, kv_seq, batch, heads = PLANS[plan]
    spec = flex_forward_spec(
        head_dim,
        q_seq=q_seq,
        kv_seq=kv_seq,
        num_bh=batch * heads,
        heads=heads,
        kv_group=kv_group,
        walk=walk,
    )
    with patch.dict(os.environ, {}, clear=True):
        generation = ConfigGeneration(spec)
        generation.validate_flash_structural_coverage()
        assert generation.flash_structural_coverage_uncovered_values() == []
        assert generation.flash_structural_coverage_uncovered_interactions() == []


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_tail_split_offers_only_the_splits_a_correction_tile_realizes(
    head_dim: int,
) -> None:
    """The searched splits divide ``head_dim / corr_tile_size`` for some
    searched correction tile (8 / 16 / 32 where they divide the head dim),
    the seed is one of them, and the resolver honours every searched value
    on the one-item grid with that tile.  Past head dim 128 the flex walk
    runs on the cooperative pair (no fa4 family): the knob is pinned off."""
    corr_tiles = tuple(size for size in (8, 16, 32) if head_dim % size == 0)
    realizable = tuple(
        splits
        for splits in cute_flash.FLASH_FLEX_TAIL_SPLITS
        if splits == 0 or any((head_dim // size) % splits == 0 for size in corr_tiles)
    )
    assert cute_flash._flash_flex_tail_split_realizable(head_dim, corr_tiles) == (
        realizable
    )
    expected = realizable if head_dim <= 128 else (0,)
    spec = flex_forward_spec(
        head_dim, q_seq=4096, kv_seq=4096, num_bh=64, heads=16, kv_group=1
    )
    with patch.dict(os.environ, {}, clear=True):
        fragments = spec._cute_flash_autotune_fragments()
    fragment = fragments[cute_flash.FLASH_FLEX_TAIL_SPLIT_KEY]
    assert isinstance(fragment, EnumFragment)
    assert tuple(fragment.search_choices or fragment.choices) == expected
    assert fragment.default() in expected
    for splits in expected:
        if splits == 0:
            continue
        size = next(s for s in corr_tiles if (head_dim // s) % splits == 0)
        with patch.dict(os.environ, {}, clear=True):
            resolved = spec._resolve_cute_flash_config(
                {
                    **spec.default_config().config,
                    cute_flash.FLASH_FLEX_TAIL_SPLIT_KEY: splits,
                    cute_flash.FLASH_CORR_TILE_SIZE_KEY: size,
                    cute_flash.FLASH_PERSISTENT_KEY: False,
                    cute_flash.FLASH_KV_SPLIT_KEY: 1,
                }
            )
        assert (resolved.flex_tail_split, resolved.corr_tile_size) == (splits, size)


@pytest.mark.parametrize(
    "targets, expected",
    [
        ((1, 128, 128), [1, 128, 128]),
        ((128, 128), [128, 128]),
        ((1, 1, 128, 128), [1, 1, 128, 128]),
        ((2, 128, 128), None),
        ((64, 128), None),
        ((128,), None),
    ],
)
def test_seed_block_sizes_accept_the_flex_kernels_two_tiles(
    targets: tuple[int, ...], expected: list[int] | None
) -> None:
    """The seeds' block sizes are the 128-row query and 128-column KV tiles
    behind any unit batch / head tiles: the attention kernels' (1, 128, 128)
    and the flex kernel's (128, 128), whose batch / head axes are fixed
    outside the search."""
    assert cute_flash._flash_seed_block_sizes(targets) == expected


def test_dense_flex_surface_seeds_the_dense_families() -> None:
    """The dense flex plan's surface (``walk="dense"``) seeds the dense
    attention kernel's families instead of the list-walk bodies: on the
    campaign's small (1, 4, 256, 64) grid the row programs and the
    persistent fa4 / ws_overlap bodies, and no seed resolves to a list
    walk (``has_list_walk``) or carries the hook knobs' searched values."""
    spec = flex_forward_spec(
        64, q_seq=256, kv_seq=256, num_bh=4, heads=4, kv_group=1, walk="dense"
    )
    with patch.dict(os.environ, {}, clear=True):
        seeds = recorded_compiler_seeds(spec)
        assert seeds
        resolved = [spec._resolve_cute_flash_config(seed.config) for seed in seeds]
        fragments = spec._cute_flash_autotune_fragments()
        messages: list[str] = []
        pairs = ConfigGeneration(spec).seed_flat_config_pairs(messages.append)
    assert not messages
    # (Two of the dense surface's 21 seeds fold into one flat config on this
    # small grid, on the attention example's own surface as well.)
    assert len(pairs) >= len(seeds) - 1
    families = {(cfg.pipeline_family, cfg.persistent) for cfg in resolved}
    assert ("row_mma", False) in families
    assert ("fa4", True) in families
    assert ("ws_overlap", True) in families
    assert all(not cfg.flex_early_qk and cfg.flex_tail_split == 0 for cfg in resolved)
    family = fragments[cute_flash.FLASH_PIPELINE_FAMILY_KEY]
    assert isinstance(family, EnumFragment)
    assert "row_mma" in (family.search_choices or family.choices)
    tail = fragments[cute_flash.FLASH_FLEX_TAIL_SPLIT_KEY]
    assert isinstance(tail, EnumFragment)
    assert tuple(tail.search_choices or tail.choices) == (0,)


@pytest.mark.parametrize("plan", sorted(PLANS))
def test_flex_forward_surface_seeds_the_measured_list_walk_bodies(
    plan: str,
) -> None:
    """The flex forward surface receives the compiler seeds (the seed builder
    used to require the attention kernels' three block sizes, which left
    every flex search without a seed: two cold searches on the gqa-causal
    bshd cell ended 3-4% behind a structure a third had found), every seed
    transfers onto the spec, and the measured walk bodies are among them:
    the mid-length grouped-query persistent walk (the shared softmax region,
    the rolled chunk loop, the 8/2 cadence on 8x1 packets, 8-row correction
    tiles, 184/64/40 registers, a three-slot ring with two output tiles, no
    all-kept vote) and the long walk's pass-2c body."""
    q_seq, kv_seq, batch, heads = PLANS[plan]
    spec = flex_forward_spec(
        128, q_seq=q_seq, kv_seq=kv_seq, num_bh=batch * heads, heads=heads, kv_group=4
    )
    with patch.dict(os.environ, {}, clear=True):
        seeds = recorded_compiler_seeds(spec)
        assert seeds
        assert all(seed.config["block_sizes"] == [128, 128] for seed in seeds)
        resolved = [spec._resolve_cute_flash_config(seed.config) for seed in seeds]
    mid = [
        cfg
        for cfg in resolved
        if cfg.persistent
        and cfg.softmax_region == "shared"
        and cfg.flex_rolled_chunks
        and cfg.e2e_schedule == "8/2"
        and cfg.exp2_packet == "8x1"
        and cfg.corr_tile_size == 8
    ]
    assert len(mid) == 1
    (mid_walk,) = mid
    assert (
        mid_walk.softmax_regs,
        mid_walk.corr_regs,
        mid_walk.other_regs,
        mid_walk.kv_stage,
        mid_walk.o_stage,
        mid_walk.flex_hook_fast_full,
        mid_walk.causal_kv_order,
    ) == (184, 64, 40, 3, 2, False, "descending")
    assert any(
        not cfg.persistent and cfg.softmax_region == "shared" and cfg.flex_rolled_chunks
        for cfg in resolved
    )
    # The campaign-structure seeds other than the measured bodies carry the
    # all-kept vote on; the per-family defaults keep it off.
    assert {cfg.flex_hook_fast_full for cfg in resolved} == {False, True}
