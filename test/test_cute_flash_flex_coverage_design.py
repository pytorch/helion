"""The flex forward search surface's structural coverage design is complete
for every head dim the CuTe flash backend supports.

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
from unittest.mock import patch

import pytest
import torch

from helion._compiler.backend import CuteBackend
from helion._compiler.cute import cute_flash
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec

HEAD_DIMS = tuple(range(16, 257, 16))
SM_COUNT = 148  # the B200's streaming multiprocessors: the tail split's seed

# (q_seq, kv_seq, batch, heads): the campaign's 4K cell and its 128K
# cursor-walk cell (1024 KV tiles: several bitmask words, the cursor).
PLANS = {
    "s4096": (4096, 4096, 4, 16),
    "s131072": (131072, 131072, 1, 8),
}
# The score plan's two surfaces: a mask-only plan (any ``mask_mod``, the
# BlockMask tables walked) and a plan with a score region (a ``score_mod``
# other than the identity: softcap / ALiBi run the hook on every tile).
WALKS = ("mask", "score")


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
        plain_row_body=False,
        has_score_modifiers=score_region,
        score_modifier_kinds=("flex",),
        device_sm_count=device_sm_count,
        kv_group=kv_group,
        fast_math=True,
        has_list_walk=True,
        flex_has_score_region=score_region,
        flex_has_mask_region=True,
        flex_trivial_walk=False,
        q_seq=q_seq,
        kv_seq=kv_seq,
        flex_aux_smem_candidates=(),
        flex_prefetch_regs=(),
        flex_coalesce_steps=(),
        flex_hook_ops=4 if score_region else 0,
    )
    return spec


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
