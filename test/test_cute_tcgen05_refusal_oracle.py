"""Reachability oracle for the tcgen05 config-layer refusals (CPU, fake B200).

Every compile of an explicit config runs the strict ``ConfigSpec.normalize``
(``BoundKernel.to_code`` -> ``_normalized_config_copy``), and the autotuner's
candidates pass the same strict pass (``has_compiled_config``,
``ConfigGeneration.strict_config_pair``) after their repair.  A codegen
refusal whose condition a strict tcgen05 validator judges from the same
facts is therefore unreachable on both paths.  Each :class:`Refusal` below is
an explicit config one of those validators refuses with its ``InvalidConfig``
text, built to trip one *conjunct* of a former codegen gate (named in the
comments); the conjuncts whose fact the validator reads from the same config
keys or bind-time facts were deleted.  A ``Refusal`` shows that its conjunct
is refused ahead of codegen for that kernel and config; it is not a proof over
every kernel, so a conjunct that reads a plan fact without a config-layer
twin stays in codegen and is pinned in ``LIVE`` instead: the plan's TMA-store
epilogue (``tcgen05_use_tma_store_epilogue`` needs the output's TMA legality
too), the composition of the lowered store's own chain (the validators'
``exact_shape_aux_kernel_detected`` is kernel-wide), the TMA legality of the
staged auxiliary descriptors, the grouped and edge families.

The remaining tests pin those live codegen refusals (the strict pass admits
the config, codegen refuses it), the knob values the search fragments refuse
ahead of any validator, and the bind whose tcgen05 search was declined: there
the strict normalizer refuses every non-default tcgen05 key outright, so the
``_normalize_epilogue_groups`` early return needs no codegen backstop.
"""

from __future__ import annotations

import dataclasses
from typing import Any
from typing import Callable

import pytest
import torch

from test import test_cute_rank3_rhs_b_tma as grouped
from test.test_autotuner_heuristics import _grouped_worklist_bind_patches
from test.test_cute_tcgen05_aux_fragment_hoist import ON
from test.test_cute_tcgen05_aux_fragment_hoist import PRE
from test.test_cute_tcgen05_aux_fragment_hoist import base_config
from test.test_cute_tcgen05_aux_fragment_hoist import bind
from test.test_cute_tcgen05_aux_fragment_hoist import cuda_trace  # noqa: F401
from test.test_cute_tcgen05_aux_fragment_hoist import gated_pair
from test.test_cute_tcgen05_aux_fragment_hoist import plain_single
from test.test_cute_tcgen05_aux_fragment_hoist import residual_single

import helion
from helion._compiler.cute.pipeline_smem import TCGEN05_CTA_GROUP_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_AUX_LOAD_MODE_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPI_SUBTILE_N_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import (
    TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY,
)
from helion._compiler.cute.tcgen05_constants import TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY
from helion._compiler.cute.tcgen05_constants import TCGEN05_SPLIT_K_CONFIG_KEY
from helion._testing.cute_codegen import cpu_target
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl

GROUPS = TCGEN05_EPI_WARP_GROUPS_CONFIG_KEY
SUBTILE = TCGEN05_EPI_SUBTILE_N_CONFIG_KEY
PREFETCH = TCGEN05_AUX_PREFETCH_SUBTILES_CONFIG_KEY
ROUND = TCGEN05_EPI_ROUND_SUBTILES_CONFIG_KEY
EPI_TMA = {
    TCGEN05_AUX_LOAD_MODE_CONFIG_KEY: "epi_tma",
    "tcgen05_ab_stages": 4,
    "tcgen05_c_stages": 2,
}
CTA_PAIR = {"tcgen05_cluster_m": 2, TCGEN05_CTA_GROUP_CONFIG_KEY: "two"}
LAYOUT_OVERRIDE = {
    "tcgen05_layout_strategy": "explicit_epi_tile",
    "tcgen05_layout_overrides_epi_tile_m": 128,
    "tcgen05_layout_overrides_epi_tile_n": 32,
    "tcgen05_layout_overrides_d_store_box_n": 32,
}
WIDE = {"block_sizes": [128, 256, 64], "tcgen05_ab_stages": 3, "tcgen05_c_stages": 2}
SCHEDULER = {
    "tcgen05_strategy": "role_local_with_scheduler",
    "tcgen05_warp_spec_scheduler_warps": 1,
}
C_INPUT_AB3 = {"tcgen05_warp_spec_c_input_warps": 1, "tcgen05_ab_stages": 3}
# The flat one-CTA split-K pipeline's config (``test_cute_tcgen05_split_k``).
FLAT_SPLIT = {
    "block_sizes": [128, 128, 64],
    "pid_type": "flat",
    "tcgen05_cluster_m": 1,
    "tcgen05_cluster_n": 1,
    "tcgen05_ab_stages": 4,
    TCGEN05_SPLIT_K_CONFIG_KEY: 2,
}
SPLIT_STRICT = "requires the flat one-CTA tcgen05 TMA pipeline over a plain rank-2 GEMM"
AB3_STRICT = "incompatible with a productive aux producer warp"


def fp32_out_single(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """fp16 operands accumulated into an fp32 output: no 16-bit epilogue."""
    m, n = x.size()
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(n):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


def bias_row_single(
    x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """A row-vector auxiliary operand only: no exact-shape auxiliary tile."""
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(n):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        out[tile_m, tile_n] = (acc + bias[tile_n].to(torch.float32)).to(x.dtype)
    return out


def two_distinct_aux(
    x: torch.Tensor, w: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two output stores, each reading its own exact-shape auxiliary tile."""
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    s = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(n):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        s[tile_m, tile_n] = (acc * y[tile_m, tile_n].to(torch.float32)).to(x.dtype)
        out[tile_m, tile_n] = (acc + x[tile_m, tile_n].to(torch.float32)).to(x.dtype)
    return out, s


def residual_aux(x: torch.Tensor, w: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """``acc + r`` with ``r`` an argument whose dtype or strides vary per test."""
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(n):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        out[tile_m, tile_n] = (acc + r[tile_m, tile_n]).to(x.dtype)
    return out


def two_store_mixed(
    x: torch.Tensor, w: torch.Tensor, r: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two stores: one reads the exact-shape residual, the other a row vector.

    The kernel-wide ``exact_shape_aux_kernel_detected`` is True, so the strict
    hoist / prefetch validators admit; the store lowering judges each store's
    own chain and refuses on the row-vector store.
    """
    m, n = x.size()
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    s = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(n):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        out[tile_m, tile_n] = (acc + r[tile_m, tile_n].to(torch.float32)).to(x.dtype)
        s[tile_m, tile_n] = (acc + bias[tile_n].to(torch.float32)).to(x.dtype)
    return out, s


def out_arg_plain(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """The output is an argument: a TMA-illegal view takes the SIMT store body."""
    m, n = x.size()
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(n):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


def out_arg_residual(
    x: torch.Tensor, w: torch.Tensor, r: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """``out_arg_plain`` with an exact-shape residual in the chain."""
    m, n = x.size()
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(n):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        out[tile_m, tile_n] = (acc + r[tile_m, tile_n].to(torch.float32)).to(x.dtype)
    return out


def side_store_single(
    x: torch.Tensor, w: torch.Tensor, side: torch.Tensor
) -> torch.Tensor:
    """A statement beside the warp roles (the residual shared loop)."""
    m, k = x.size()
    n = w.size(1)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
        side[tile_m.id, tile_n.id] = 1.0
    return out


def fp16(*shape: int) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.float16)


def bind_args(fn: Callable[..., object], *args: torch.Tensor) -> Any:
    kernel = helion.kernel(
        fn, backend="cute", static_shapes=True, autotune_effort="full"
    )
    return kernel._bind_isolated(args)


def strided_out() -> torch.Tensor:
    """A 16-bit output view with a 4-byte element stride: no TMA store."""
    return fp16(4096, 2048)[:, ::2]


def bind_two_store_mixed() -> Any:
    return bind_args(
        two_store_mixed,
        fp16(4096, 1024),
        fp16(1024, 1024),
        fp16(4096, 1024),
        fp16(1024),
    )


def bind_strided_out_plain() -> Any:
    return bind_args(out_arg_plain, fp16(4096, 1024), fp16(1024, 1024), strided_out())


def bind_strided_out_residual() -> Any:
    return bind_args(
        out_arg_residual,
        fp16(4096, 1024),
        fp16(1024, 1024),
        fp16(4096, 1024),
        strided_out(),
    )


def clc_pair(**overrides: Any) -> dict[str, Any]:
    """The validated two-stage scheduler mailbox envelope."""
    config: dict[str, Any] = {
        "block_sizes": [256, 256, 128],
        "pid_type": "persistent_interleaved",
        "tcgen05_cluster_m": 2,
        "tcgen05_cluster_n": 1,
        "tcgen05_ab_stages": 2,
        "tcgen05_acc_stages": 2,
        "tcgen05_c_stages": 2,
        "tcgen05_num_epi_warps": 4,
        "tcgen05_strategy": "role_local_with_scheduler",
        "tcgen05_warp_spec_scheduler_warps": 1,
        "tcgen05_persistence_model": "clc_persistent",
        TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY: 2,
    }
    config.update(overrides)
    return config


@dataclasses.dataclass(frozen=True)
class Refusal:
    name: str
    bind: Callable[[], Any]
    config: dict[str, Any]
    strict: str


HOIST_STRICT = "aux_fragment_hoist=True requires an exact-shape auxiliary operand"
EXACT_AUX_STRICT = "requires an exact-shape rank-2 auxiliary operand"

CORPUS: list[Refusal] = [
    # ``validate_strategy_invariants``: tcgen05_cluster_n=2 without cluster_m=2
    # (the planner keeps its own gate for the clustered single-CTA shapes).
    Refusal(
        "cluster_n_without_cluster_m",
        lambda: bind(plain_single),
        base_config(tcgen05_cluster_n=2),
        "strategy invariants violated: tcgen05_cluster_n=2 requires tcgen05_cluster_m=2",
    ),
    # ``epi_subtile_refusal``: the config conjuncts of the plan-level
    # ``tcgen05_epi_subtile_n`` gate ("needs a 16-bit role-local TMA-store
    # epilogue without explicit layout overrides"): static full tiles, a
    # 16-bit output, no layout override.  The plan's TMA-store epilogue is
    # the gate's remaining conjunct (``LIVE``).
    Refusal(
        "epi_subtile_edge_tiles",
        lambda: bind(plain_single, rows=4096 + 64),
        base_config({SUBTILE: 32}),
        "the configured epilogue subtile requires static full output tiles",
    ),
    Refusal(
        "epi_subtile_fp32_output",
        lambda: bind(fp32_out_single),
        base_config({SUBTILE: 32}),
        "the configured epilogue subtile needs 16-bit outputs",
    ),
    Refusal(
        "epi_subtile_layout_override",
        lambda: bind(plain_single),
        base_config({SUBTILE: 32}, **LAYOUT_OVERRIDE),
        "the configured epilogue subtile requires tcgen05_layout_strategy='default'",
    ),
    # ``epi_tma_refusal``: the reasons of the epilogue-issued auxiliary ring's
    # former chain with a config-layer twin (no exact-shape aux, several
    # stores, the producer warp, edge tiles, the clustered single-CTA family,
    # the bm=128 pair, an aux dtype mismatch).  The chain keeps the plan's
    # TMA-store epilogue (``LIVE``) and the descriptors' TMA legality.
    Refusal(
        "epi_tma_without_aux",
        lambda: bind(plain_single),
        base_config(EPI_TMA),
        EXACT_AUX_STRICT,
    ),
    Refusal(
        "epi_tma_two_aux_stores",
        lambda: bind_args(
            two_distinct_aux, fp16(4096, 1024), fp16(1024, 1024), fp16(4096, 1024)
        ),
        base_config(EPI_TMA),
        EXACT_AUX_STRICT,
    ),
    Refusal(
        "epi_tma_producer_warp",
        lambda: bind(residual_single),
        base_config(EPI_TMA, tcgen05_ab_stages=2, tcgen05_warp_spec_c_input_warps=1),
        "replaces the aux producer warp: requires tcgen05_warp_spec_c_input_warps=0",
    ),
    Refusal(
        "epi_tma_edge_tiles",
        lambda: bind(residual_single, rows=4096 + 64),
        base_config(EPI_TMA),
        "epi_tma' requires static full output tiles",
    ),
    Refusal(
        "epi_tma_clustered_single_cta",
        lambda: bind(residual_single),
        base_config(EPI_TMA, tcgen05_cluster_m=2),
        "epi_tma' requires the CtaGroup.TWO pair when tcgen05_cluster_m=2",
    ),
    Refusal(
        "epi_tma_bm128_pair",
        lambda: bind(residual_single),
        base_config(EPI_TMA, CTA_PAIR),
        "is not validated on the bm=128 CtaGroup.TWO epilogue tile",
    ),
    Refusal(
        "epi_tma_strided_aux",
        lambda: bind_args(
            residual_aux, fp16(4096, 1024), fp16(1024, 1024), fp16(4096, 2048)[:, ::2]
        ),
        base_config(EPI_TMA),
        "epi_tma' requires TMA-legal auxiliary tensors",
    ),
    # ``_validate_cta_group_config``: the former ``_tcgen05_use_2cta_instrs``
    # envelope of an explicit tcgen05_cta_group='two'.
    Refusal(
        "cta_group_two_unclustered",
        lambda: bind(plain_single),
        base_config(**{TCGEN05_CTA_GROUP_CONFIG_KEY: "two"}),
        "tcgen05_cta_group='two' requires one TMA-legal half/bfloat16 MMA",
    ),
    # ``pre_acc_wait_admissible``: the former store-lowering gate on a chain
    # without per-subtile auxiliary loads (and on an unsafe shared fanout).
    Refusal(
        "pre_acc_wait_aux_free_store",
        lambda: bind(gated_pair),
        base_config(PRE),
        "a store of the kernel has no per-subtile auxiliary loads to place",
    ),
    # ``aux_fragment_hoist_admissible``: the store-lowering gates on the
    # hoist's placement, the kernel's exact-shape operand and the per-thread
    # register budget.  The lowering keeps the plan's TMA-store epilogue and
    # the lowered store's own exact-shape read (``LIVE``).
    Refusal(
        "hoist_without_pre_acc_wait",
        lambda: bind(residual_single),
        base_config(ON),
        HOIST_STRICT,
    ),
    Refusal(
        "hoist_without_exact_aux",
        lambda: bind_args(
            bias_row_single, fp16(4096, 1024), fp16(1024, 1024), fp16(1024)
        ),
        base_config(ON, PRE),
        HOIST_STRICT,
    ),
    Refusal(
        "hoist_over_budget",
        lambda: bind(residual_single),
        base_config(ON, PRE, WIDE),
        "at most 256 bytes per epilogue thread",
    ),
    # ``aux_prefetch_refusal``: the store-lowering gates on the ring's
    # placement, the superseding hoist, the ring's register budget and the
    # whole-subtile partition of the CTA tile.  As for the hoist, the plan's
    # TMA-store epilogue and the store's own exact-shape read stay (``LIVE``).
    Refusal(
        "prefetch_without_pre_acc_wait",
        lambda: bind(residual_single),
        base_config({PREFETCH: 1}),
        "the auxiliary prefetch ring needs tcgen05_aux_load_placement='pre_acc_wait'",
    ),
    Refusal(
        "prefetch_with_hoist",
        lambda: bind(residual_single),
        base_config(PRE, ON, {PREFETCH: 1}),
        "superseded by the whole-fragment hoist",
    ),
    Refusal(
        "prefetch_over_budget",
        lambda: bind(residual_single),
        base_config(PRE, WIDE, {SUBTILE: 64, PREFETCH: 2}),
        "deepest fitting depth 1",
    ),
    Refusal(
        "prefetch_bm128_pair",
        lambda: bind(residual_single),
        base_config(PRE, CTA_PAIR, {PREFETCH: 1}, tcgen05_c_stages=2),
        "requires the 256-row CtaGroup.TWO pair tile when tcgen05_cluster_m=2",
    ),
    # ``epi_round_subtiles_refusal``: the former store-lowering gate on a round
    # that does not divide the C ring.
    Refusal(
        "round_not_dividing_the_ring",
        lambda: bind(plain_single),
        base_config({ROUND: 4}, tcgen05_c_stages=2),
        "must divide tcgen05_c_stages=2",
    ),
    # ``_normalize_split_k``: the config-level half of the split-K pipeline's
    # envelope -- the flat pid type, a body whose only side effect is the
    # tile store (``split_k_plain_body``), a K loop at least ``split`` tiles
    # deep.  The pipeline keeps the plan facts (the one-CTA TMA pipeline, the
    # tile family, one epilogue warp group).
    Refusal(
        "split_k_on_a_persistent_pid",
        lambda: bind(plain_single),
        base_config({TCGEN05_SPLIT_K_CONFIG_KEY: 2}),
        f"tcgen05_split_k=2 {SPLIT_STRICT}",
    ),
    Refusal(
        "split_k_beyond_the_k_tiles",
        lambda: bind(plain_single),
        # Two AB stages: the SMEM budget refuses four at bk=128 ahead of the
        # split; K=1024 over bk=128 is 8 tiles.
        {
            **FLAT_SPLIT,
            "block_sizes": [128, 128, 128],
            "tcgen05_ab_stages": 2,
            TCGEN05_SPLIT_K_CONFIG_KEY: 16,
        },
        "with at least 16 K tiles",
    ),
    Refusal(
        "split_k_beside_a_side_store",
        lambda: bind_args(
            side_store_single, fp16(4096, 1024), fp16(1024, 1024), torch.zeros(32, 8)
        ),
        FLAT_SPLIT,
        SPLIT_STRICT,
    ),
    # ``_normalize_aux_producer_ab_stages``: the C-input warp, or the store
    # warp staging the auxiliary ring through TMA, beside three AB stages on
    # a kernel with an exact-shape auxiliary operand of the output's dtype.
    Refusal(
        "ab_stages_3_with_a_c_input_warp",
        lambda: bind(residual_single),
        base_config(SCHEDULER, C_INPUT_AB3),
        AB3_STRICT,
    ),
    Refusal(
        "ab_stages_3_with_the_store_warp_aux_ring",
        lambda: bind(residual_single),
        base_config(
            SCHEDULER,
            {TCGEN05_AUX_LOAD_MODE_CONFIG_KEY: "tma"},
            tcgen05_warp_spec_store_warps=1,
            tcgen05_ab_stages=3,
        ),
        AB3_STRICT,
    ),
    # The exact-shape fact counts an auxiliary operand of any dtype (an
    # FP32 residual of an FP16 output too), so the strict pass refuses its
    # producer warp before codegen.
    Refusal(
        "ab_stages_3_with_a_c_input_warp_and_an_fp32_residual",
        lambda: bind_args(
            residual_aux,
            fp16(4096, 1024),
            fp16(1024, 1024),
            torch.empty(4096, 1024, dtype=torch.float32),
        ),
        base_config(SCHEDULER, C_INPUT_AB3),
        AB3_STRICT,
    ),
]


@pytest.mark.parametrize("refusal", CORPUS, ids=[r.name for r in CORPUS])
def test_strict_normalizer_refuses_before_codegen(refusal: Refusal) -> None:
    bound = refusal.bind()
    config = helion.Config(**refusal.config)
    with pytest.raises(InvalidConfig, match=refusal.strict):
        bound.config_spec.normalized_config(config)
    # The explicit path reports the normalizer's refusal: codegen never ran.
    with pytest.raises(InvalidConfig, match=refusal.strict):
        bound.to_code(config)


def test_split_k_within_the_k_tiles_is_admitted() -> None:
    # The control of ``split_k_beyond_the_k_tiles``: the same split over the
    # 64-deep K tile (K=1024 -> 16 tiles) passes the strict pass, so the
    # K-tile conjunct is what that corpus entry exercises.
    bound = bind(plain_single)
    config = helion.Config(
        **{**FLAT_SPLIT, "tcgen05_ab_stages": 2, TCGEN05_SPLIT_K_CONFIG_KEY: 16}
    )
    bound.config_spec.normalized_config(config)


HOIST_CODEGEN = (
    "aux_fragment_hoist=True requires tcgen05_aux_load_placement='pre_acc_wait', "
    "the static-full TMA-store epilogue and an exact-shape auxiliary load in the "
    "chain"
)
PREFETCH_CODEGEN = (
    "aux_prefetch_subtiles=1 requires tcgen05_aux_load_placement='pre_acc_wait', "
    "the static-full TMA-store epilogue, an exact-shape auxiliary load in the "
    "chain and no tcgen05_aux_fragment_hoist"
)
SUBTILE_CODEGEN = (
    "epi_subtile_n=32 needs a 16-bit role-local TMA-store epilogue without "
    "explicit layout overrides"
)

# Codegen refusals the normalizer does not shadow: the strict pass admits the
# config and the planner or store lowering refuses it.  They stay.
LIVE: list[tuple[str, Callable[[], Any], dict[str, Any], type[Exception], str]] = [
    # The lowered store's own chain: the validators' kernel-wide
    # ``exact_shape_aux_kernel_detected`` admits the hoist and the ring on a
    # kernel whose sibling store reads only a row vector.
    (
        "hoist_on_a_store_without_the_exact_aux",
        bind_two_store_mixed,
        base_config(PRE, ON),
        InvalidConfig,
        HOIST_CODEGEN,
    ),
    (
        "prefetch_ring_on_a_store_without_the_exact_aux",
        bind_two_store_mixed,
        base_config(PRE, {PREFETCH: 1}),
        InvalidConfig,
        PREFETCH_CODEGEN,
    ),
    # The plan's TMA-store epilogue: a TMA-illegal output argument takes the
    # SIMT store body; the validators take static full tiles for the TMA
    # store.
    (
        "epi_subtile_on_the_simt_store_epilogue",
        bind_strided_out_plain,
        base_config({SUBTILE: 32}),
        InvalidConfig,
        SUBTILE_CODEGEN,
    ),
    (
        "hoist_on_the_simt_store_epilogue",
        bind_strided_out_residual,
        base_config(PRE, ON),
        InvalidConfig,
        HOIST_CODEGEN,
    ),
    (
        "prefetch_ring_on_the_simt_store_epilogue",
        bind_strided_out_residual,
        base_config(PRE, {PREFETCH: 1}),
        InvalidConfig,
        PREFETCH_CODEGEN,
    ),
    (
        "epi_tma_on_the_simt_store_epilogue",
        bind_strided_out_residual,
        base_config(EPI_TMA),
        BackendUnsupported,
        "epi_tma' requires the TMA-store epilogue",
    ),
    (
        "cluster_n2_on_the_clustered_single_cta_family",
        lambda: bind(plain_single),
        base_config(tcgen05_cluster_m=2, tcgen05_cluster_n=2, tcgen05_c_stages=2),
        BackendUnsupported,
        r"tcgen05_cluster_n=2 requires tcgen05_cluster_m=2 with use_2cta=True \(bm=256\)",
    ),
    (
        "epi_warp_groups_on_a_flat_pid",
        lambda: bind(plain_single),
        base_config({GROUPS: 2}, pid_type="flat"),
        InvalidConfig,
        "epi_warp_groups=2 needs the standard role-local TMA-store epilogue",
    ),
    (
        "prefetch_ring_on_a_flat_pid",
        lambda: bind(residual_single),
        base_config(PRE, {PREFETCH: 1}, pid_type="flat"),
        InvalidConfig,
        "need the standard static-full role-local TMA-store epilogue",
    ),
    (
        "sched_stage_count_with_a_residual_shared_loop",
        lambda: bind_args(
            side_store_single,
            fp16(512, 257 * 128),
            fp16(257 * 128, 512),
            torch.zeros(8, 8),
        ),
        clc_pair(),
        InvalidConfig,
        "sched_stage_count=2 requires omitted shared-loop full role-local scheduler",
    ),
]


@pytest.mark.parametrize("case", LIVE, ids=[case[0] for case in LIVE])
def test_live_codegen_refusals_pass_the_strict_normalizer(
    case: tuple[str, Callable[[], Any], dict[str, Any], type[Exception], str],
) -> None:
    _name, make_bound, config, error, codegen = case
    bound = make_bound()
    requested = helion.Config(**config)
    bound.config_spec.normalized_config(requested)
    with pytest.raises(error, match=codegen):
        bound.to_code(requested)


GROUPED_SIGNATURE = [4, 128, 64, 32, 16, 128, 1536, 128, 64, 16, 16, 80, 16]
GROUPED_RASTER: list[tuple[str, dict[str, Any], str]] = [
    (
        "l2_swizzle",
        {"tcgen05_l2_swizzle_size": 2},
        "requires tcgen05_l2_swizzle_size=1",
    ),
    ("l2_groupings", {"l2_groupings": [2]}, "requires l2_groupings entries"),
]


@pytest.mark.parametrize("case", GROUPED_RASTER, ids=[c[0] for c in GROUPED_RASTER])
def test_grouped_static_signature_keeps_the_plain_raster(
    case: tuple[str, dict[str, Any], str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # ``_normalize_grouped_static_signature_raster``: the static grouped
    # scheduler has no L2 swizzle or grouping decode (the former ``cute_mma``
    # gates).  Bound the way ``test_cute_grouped_tail_store`` binds the
    # rank-3 grouped GEMM on the CPU.
    _name, part, text = case
    monkeypatch.setenv("HELION_CUTE_MMA_IMPL", "tcgen05")
    monkeypatch.setattr(grouped, "DEVICE", torch.device("cpu"))
    config = grouped._dynamic_bk64_config(direct=True)
    config.config.update(part)
    config.config[TCGEN05_GROUPED_STATIC_PROBLEM_SIGNATURE_CONFIG_KEY] = (
        GROUPED_SIGNATURE
    )
    with cpu_target(), _grouped_worklist_bind_patches():
        kernel = grouped._rank3_rhs_grouped_nt_with_mn_tails_and_k_sizes
        args = grouped._make_documented_mixed_k_args()
        bound = kernel.bind(args)
        bound.env.config_spec.tcgen05.search_enabled = True
        with pytest.raises(InvalidConfig, match=text):
            bound.config_spec.normalized_config(config)
        with pytest.raises(InvalidConfig, match=text):
            grouped._code_for(kernel, args, config)


# Values outside a knob's choices are refused by its search fragment, ahead of
# every validator (the former codegen type checks of these knobs).
FRAGMENT_REFUSALS: list[tuple[str, dict[str, Any], str]] = [
    (
        "subtile",
        {SUBTILE: 48},
        r"epi_subtile_n must be one of \(None, 32, 64\), got 48",
    ),
    ("groups", {GROUPS: 5}, r"epi_warp_groups must be one of \(1, 2, 3, 4\), got 5"),
    ("prefetch", {PREFETCH: -1}, r"aux_prefetch_subtiles must be in \[0, 4\], got -1"),
    ("round", {ROUND: 3}, r"epi_round_subtiles must be one of \(1, 2, 4\), got 3"),
]


@pytest.mark.parametrize(
    "case", FRAGMENT_REFUSALS, ids=[c[0] for c in FRAGMENT_REFUSALS]
)
def test_fragment_choices_refuse_before_any_validator(
    case: tuple[str, dict[str, Any], str],
) -> None:
    _name, part, text = case
    bound = bind(residual_single)
    with pytest.raises(InvalidConfig, match=text):
        bound.to_code(helion.Config(**base_config(part)))


KNOBS: list[tuple[str, dict[str, Any]]] = [
    ("subtile", {SUBTILE: 32}),
    ("groups", {GROUPS: 2}),
    ("prefetch", {PREFETCH: 1}),
    ("round", {ROUND: 2}),
    ("hoist", ON),
    ("placement", PRE),
    ("epi_tma", {TCGEN05_AUX_LOAD_MODE_CONFIG_KEY: "epi_tma"}),
    ("cta_group", {TCGEN05_CTA_GROUP_CONFIG_KEY: "two"}),
    ("cluster_n", {"tcgen05_cluster_n": 2}),
    ("sched_stage_count", {TCGEN05_SCHED_STAGE_COUNT_CONFIG_KEY: 2}),
]


@pytest.mark.parametrize("knob", KNOBS, ids=[knob[0] for knob in KNOBS])
def test_search_disabled_bind_refuses_every_tcgen05_knob(
    knob: tuple[str, dict[str, Any]],
) -> None:
    # ``_normalize_epilogue_groups`` returns early without the search; the
    # strict pass has already refused every non-default tcgen05 key by then.
    _name, part = knob
    bound = bind(residual_single)
    bound.config_spec._cute_tcgen05_config.search_enabled = False
    config = helion.Config(block_sizes=[128, 128, 64], **part)
    with pytest.raises(InvalidConfig, match="only supported for tcgen05-enabled"):
        bound.to_code(config)
