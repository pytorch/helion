"""GPU numerics of the epilogue warp groups, the configured epilogue subtile
and the auxiliary prefetch ring (``tcgen05_epi_warp_groups``,
``tcgen05_epi_subtile_n``, ``tcgen05_aux_prefetch_subtiles``).

Every variant renders the same per-element chain in the same K order: the
outputs are bitwise equal to the one-group, per-subtile-read render of the
same tile, on the CTA pair and on the one-CTA tile, including a shape with
several tiles per CTA (the groups' ring index walks the persistent tile
counter) and a ragged three-group split at one and two C stages per group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pytest
import torch

import helion
from helion._compiler.cute.epilogue_fanout import FANOUT_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPI_SUBTILE_N_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl

if TYPE_CHECKING:
    from torch import Tensor

pytestmark = skipUnlessBackends(["cute"])

GROUPS = TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY
SUBTILE = TCGEN05_EPI_SUBTILE_N_CONFIG_KEY
PREFETCH = TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY
HOIST = TCGEN05_AUX_FRAGMENT_HOIST_CONFIG_KEY
PLACEMENT = TCGEN05_AUX_LOAD_PLACEMENT_CONFIG_KEY

BASE = {
    "pid_type": "persistent_interleaved",
    "tcgen05_cta_group": "auto",
    "tcgen05_cluster_n": 1,
    "tcgen05_acc_stages": 2,
    "tcgen05_num_epi_warps": 4,
    "tcgen05_l2_swizzle_size": 1,
    "tcgen05_strategy": "role_local_monolithic",
    "tcgen05_layout_strategy": "default",
    "tcgen05_persistence_model": "static_persistent",
    FANOUT_CONFIG_KEY: "shared",
    PLACEMENT: "pre_acc_wait",
}


def gated_pair(x: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    """``se_block_fwd``: ``s = sigmoid(x @ w)`` and ``out = 2 * x * s``."""
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    s = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(n):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        sigmoid_result = torch.sigmoid(acc.to(x.dtype))
        s[tile_m, tile_n] = sigmoid_result
        x_tile = x[tile_m, tile_n].to(torch.float32)
        out[tile_m, tile_n] = (2.0 * x_tile * sigmoid_result).to(x.dtype)
    return out, s


def _requires_sm100() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")


def _inputs(m: int, n: int, seed: int = 20261025) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((m, n), generator=generator, dtype=torch.float16) * 0.1
    w = torch.randn((n, n), generator=generator, dtype=torch.float16) * 0.1
    return x.to(DEVICE), w.to(DEVICE)


def _compile(bound: Any, **values: object) -> tuple[helion.Config, Any]:
    config = helion.Config(**values)
    bound.config_spec.normalize(config)
    return config, bound.compile_config(config, allow_print=False)


def _reference(x: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    s = torch.sigmoid((x.float() @ w.float()).to(torch.float16))
    return (2.0 * x.float() * s.float()).to(torch.float16), s


# (m, n), tile, ab stages, cluster_m, and the variants (groups, subtile,
# prefetch, hoist, c_stages) rendered beside the one-group per-subtile base.
PAIR_VARIANTS = (
    (2, 32, 0, True, 2),  # the measured recipe: per-group hoist
    (2, 32, 0, True, 1),  # one C stage per group
    (2, 32, 3, False, 2),  # two groups over the prefetch ring
    (1, 32, 3, False, 2),  # one group, prefetch ring
    (1, 32, 0, False, 2),  # the subtile alone
    (1, 64, 1, False, 2),  # the plan's subtile, a one-deep ring
    (2, 64, 0, True, 1),  # two groups at the (128, 64) subtile
    (3, 32, 2, False, 1),  # ragged: eight subtiles over three groups
    (4, 32, 1, False, 1),  # four groups, two subtiles each
)
# The one-CTA 128x128 tile's plan subtile is (128, 64): two subtiles, one per
# group; two groups keep one C stage each (two-stage rings of two outputs and
# four AB stages fill the arena otherwise).
ONE_CTA_VARIANTS = (
    (2, None, 0, True, 1),
    (2, None, 1, False, 1),
    (1, None, 1, False, 2),
    (2, 32, 0, True, 2),
)
# One C stage per group: the pair's four-stage AB ring leaves 32 KiB for the
# two outputs' rings (a third group's stage overflows the arena).
WINNER_VARIANTS = (
    (2, None, 0, True, 1),
    (1, None, 2, False, 2),
    (2, None, 1, False, 1),
)
# The ragged split with two C stages per group (three groups' rings at four
# AB stages): each group's ring index strides by its own subtile count per
# tile, so the stage a bulk store still reads is never the next r2s target.
RAGGED_VARIANTS = (
    (3, 32, 2, False, 2),
    (3, 32, 0, True, 2),
)
CASES = (
    # The CTA pair at one tile per CTA (the hillclimb cell) ...
    ((4096, 1024), [256, 256, 64], 5, 2, PAIR_VARIANTS),
    # ... and at several tiles per CTA (the ring index walks the tile counter).
    ((4096, 2048), [256, 256, 64], 5, 2, PAIR_VARIANTS[:4] + PAIR_VARIANTS[7:8]),
    ((4096, 2048), [256, 256, 64], 4, 2, RAGGED_VARIANTS),
    # The one-CTA 128x128 tile (its plan subtile is (128, 64)).
    ((1024, 1024), [128, 128, 64], 4, 1, ONE_CTA_VARIANTS),
    # The campaign winner's 256x128 pair tile at the plan's (128, 32) subtile
    # (four subtiles, two tiles per CTA): groups and the ring on the plan's
    # own subtile.
    ((4096, 1024), [256, 128, 128], 4, 2, WINNER_VARIANTS),
)


@pytest.mark.parametrize(
    "shape,block_sizes,ab_stages,cluster_m,variants",
    CASES,
    ids=(
        "pair_one_tile",
        "pair_two_tiles",
        "pair_two_tiles_ragged",
        "one_cta",
        "winner_pair",
    ),
)
def test_grouped_epilogues_match_torch_and_the_one_group_render_bitwise(
    shape: tuple[int, int],
    block_sizes: list[int],
    ab_stages: int,
    cluster_m: int,
    variants: tuple[tuple[int, int | None, int, bool, int], ...],
) -> None:
    _requires_sm100()
    x, w = _inputs(*shape)
    ref_out, ref_s = _reference(x, w)
    kernel = helion.kernel(
        gated_pair, backend="cute", static_shapes=True, autotune_effort="full"
    )
    bound = kernel._bind_isolated((x, w))
    common: dict[str, object] = {
        **BASE,
        "block_sizes": block_sizes,
        "tcgen05_cluster_m": cluster_m,
        "tcgen05_ab_stages": ab_stages,
        "tcgen05_c_stages": 2,
        HOIST: False,
    }
    _config, base = _compile(bound, **common)
    base_out, base_s = base(x, w)
    torch.cuda.synchronize()
    torch.testing.assert_close(base_out, ref_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(base_s, ref_s, rtol=1e-2, atol=1e-2)
    for groups, subtile, prefetch, hoist, c_stages in variants:
        values = {
            **common,
            GROUPS: groups,
            SUBTILE: subtile,
            PREFETCH: prefetch,
            HOIST: hoist,
            "tcgen05_c_stages": c_stages,
        }
        config, fn = _compile(bound, **values)
        for key in (GROUPS, SUBTILE, PREFETCH, HOIST, "tcgen05_c_stages"):
            assert config.config[key] == values[key], (key, config.config[key])
        out, s = fn(x, w)
        torch.cuda.synchronize()
        assert torch.equal(out, base_out), (groups, subtile, prefetch, hoist)
        assert torch.equal(s, base_s), (groups, subtile, prefetch, hoist)
        # Launch again: the persistent tile counter and the ring stay in step.
        out2, s2 = fn(x, w)
        torch.cuda.synchronize()
        assert torch.equal(out2, base_out) and torch.equal(s2, base_s)
