"""GPU tests of the FlexAttention envelope on the CuTe fa4 pair body.

The example kernel (``examples/flex_attention.py``) goes through the flex
matcher, the hook renderer and the list-walk emitter. Expectations:

* a dense BlockMask with the noop mask is bitwise the dense fa4 kernel at
  the same pinned config and walk order;
* a causal BlockMask with the causal ``mask_mod`` is bitwise the causal plan
  with the single-application mask (store-back) and the descending split;
* against ``torch.nn.attention.flex_attention`` (fp32 reference) within bf16
  tolerance: rows with no visible key (output 0, LSE ``-inf``), ``M != N``,
  ``M % 256 == 128``, ``N % 128 == 64``, one KV block (the trivial walk),
  grouped-query heads, a score mod with a table gather, a (256, 128) mask;
* a random table mask with empty rows, pairs and heads launched repeatedly
  (the statistics handoff under contention) gives stable output;
* the persistent list walk (host-sorted longest-first item order) is bitwise
  the one-item-per-CTA grid and follows in-place table rewrites.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Any

import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.cute")
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import flex_attention as torch_flex
from torch.nn.attention.flex_attention import noop_mask

import helion
from helion._compiler.cute import cute_flash
from helion._testing import DEVICE
from helion._testing import EXAMPLES_DIR
from helion._testing import import_path
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
import helion.language as hl
from helion.runtime.kernel import Kernel

if TYPE_CHECKING:
    from collections.abc import Callable

# The pruned-interval campaign structure the flex seeds carry.
PAIR_BODY: dict[str, object] = {
    cute_flash.FLASH_PIPELINE_FAMILY_KEY: "fa4",
    cute_flash.FLASH_EXP2_PACKET_KEY: "8x2",
    cute_flash.FLASH_E2E_SCHEDULE_KEY: "16/8",
    cute_flash.FLASH_S_LOAD_REP_KEY: 16,
    cute_flash.FLASH_P_STORE_REP_KEY: 16,
    cute_flash.FLASH_DISC_PIPE_KEY: 2,
    cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: True,
    cute_flash.FLASH_EPI_TMA_KEY: True,
    cute_flash.FLASH_PERSISTENT_KEY: False,
    cute_flash.FLASH_STAT_TRANSPORT_KEY: "ring2",
    cute_flash.FLASH_SOFTMAX_DISC_KEY: True,
    # The seeded default splits a small grid's KV walk over several CTAs
    # (``cute_flash_kv_split``, 4 for most of these shapes); the pinned
    # bodies keep the one-CTA walk, ``test_split_kv_walk_matches_the_unsplit_kernel``
    # covers the splits.
    cute_flash.FLASH_KV_SPLIT_KEY: 1,
}
PAIR_DESC: dict[str, object] = {
    **PAIR_BODY,
    cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "descending",
}
# The staged-store body with 32-column chunks and the ascending walk.
STAGED_ASC: dict[str, object] = {
    **PAIR_BODY,
    cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: "ascending",
    cute_flash.FLASH_EPI_TMA_KEY: False,
    cute_flash.FLASH_EPI_STG_KEY: True,
    cute_flash.FLASH_S_LOAD_REP_KEY: 32,
}


def _cute_kernel(kernel: Kernel) -> Kernel:
    return Kernel(
        kernel.fn, settings=kernel.settings.copy(backend="cute", autotune_effort="none")
    )


def _flex_kernel() -> Kernel:
    module = import_path(EXAMPLES_DIR / "flex_attention.py")
    return _cute_kernel(module.helion_flex_attention_kernel)


def _flex_args(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    score_mod: Callable[..., Any],
    block_mask: Any,
    *,
    mask_mod: Callable[..., Any] | None = None,
    enable_gqa: bool = False,
) -> tuple[Any, ...]:
    """The kernel arguments the example wrapper builds (both lists always)."""
    full_num = block_mask.full_kv_num_blocks
    full_idx = block_mask.full_kv_indices
    if full_num is None or full_idx is None:
        full_num = torch.zeros_like(block_mask.kv_num_blocks)
        full_idx = torch.zeros_like(block_mask.kv_indices)
    return (
        q,
        k,
        v,
        score_mod,
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        full_num,
        full_idx,
        block_mask.mask_mod if mask_mod is None else mask_mod,
        block_mask.BLOCK_SIZE[0],
        block_mask.BLOCK_SIZE[1],
        1.0 / math.sqrt(q.size(-1)),
        enable_gqa,
    )


def _compile_flex(
    args: tuple[Any, ...], overrides: dict[str, object] | None
) -> tuple[Any, cute_flash.FlashAttentionConfig]:
    """Compile the flex kernel at the default config plus ``overrides``,
    asserting the surface is the flex plan category and the body the walk."""
    bound = _flex_kernel().bind(args)
    spec = bound.config_spec
    assert spec.cute_flash_search_enabled
    assert spec._cute_flash_score_modifier_kinds == ("flex",)
    base = spec.default_config().config
    normalized = bound._normalized_config_copy(
        helion.Config(**{**base, **(overrides or {})})
    )
    resolved = spec._resolve_cute_flash_config(normalized.config)
    assert resolved.pipeline_family == "fa4"
    assert resolved.persistent == bool(
        (overrides or {}).get(cute_flash.FLASH_PERSISTENT_KEY, False)
    )
    code = bound.to_code(normalized)
    assert "flash_flex_status" in code
    return bound.compile_config(normalized), resolved


def _run_flex(
    args: tuple[Any, ...], overrides: dict[str, object] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    fn, _ = _compile_flex(args, overrides)
    out, lse = fn(*args)
    torch.cuda.synchronize()
    return out, lse


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    score_mod: Callable[..., Any] | None,
    block_mask: Any,
    *,
    enable_gqa: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, lse = torch_flex(
        q.float(),
        k.float(),
        v.float(),
        score_mod=score_mod,
        block_mask=block_mask,
        scale=1.0 / math.sqrt(q.size(-1)),
        enable_gqa=enable_gqa,
        return_lse=True,
    )
    return out, lse


def _assert_close_to_reference(
    out: torch.Tensor, lse: torch.Tensor, out_ref: torch.Tensor, lse_ref: torch.Tensor
) -> None:
    torch.testing.assert_close(out.float(), out_ref, atol=3e-2, rtol=2e-2)
    finite = torch.isfinite(lse_ref)
    torch.testing.assert_close(lse[finite].float(), lse_ref[finite], atol=2e-2, rtol=0)
    empty = ~finite
    if empty.any():
        # A row with no visible key: output 0 and LSE -inf (flex semantics).
        assert torch.all(out[empty] == 0)
        assert torch.all(lse[empty] == float("-inf"))


def _qkv(
    batch: int, heads: int, q_seq: int, kv_seq: int, *, kv_heads: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    kv_heads = heads if kv_heads is None else kv_heads
    q = torch.randn(batch, heads, q_seq, 64, dtype=torch.bfloat16, device=DEVICE)
    k = torch.randn(batch, kv_heads, kv_seq, 64, dtype=torch.bfloat16, device=DEVICE)
    v = torch.randn(batch, kv_heads, kv_seq, 64, dtype=torch.bfloat16, device=DEVICE)
    return q, k, v


def _identity(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    return score


def _causal(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    return q_idx >= kv_idx


def _sliding(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    return (q_idx >= kv_idx) & (q_idx - kv_idx <= 384)


def _empty_rows(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    # Every fourth 128-row block sees no key at all.
    return ((q_idx // 128) % 4 != 1) & (q_idx >= kv_idx)


def _band(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    return (kv_idx <= q_idx + 300) & (kv_idx + 700 >= q_idx)


@skipIfNotCUDA()
@onlyBackends(["cute"])
@pytest.mark.parametrize("order", ["descending", "ascending"])
def test_dense_blockmask_is_bitwise_the_dense_fa4_kernel(order: str) -> None:
    batch, heads, seq = 2, 4, 2048
    q, k, v = _qkv(batch, heads, seq, seq)
    block_mask = create_block_mask(
        noop_mask, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    args = _flex_args(q, k, v, _identity, block_mask)
    out, lse = _run_flex(
        args, {**PAIR_BODY, cute_flash.FLASH_CAUSAL_KV_ORDER_KEY: order}
    )
    dense = _cute_kernel(import_path(EXAMPLES_DIR / "attention.py").attention)
    dense_bound = dense.bind((q, k, v))
    dense_cfg = dense_bound._normalized_config_copy(
        helion.Config(
            **{
                **dense_bound.config_spec.default_config().config,
                **PAIR_BODY,
                cute_flash.FLASH_KV_ORDER_KEY: order,
            }
        )
    )
    out_dense, lse_dense = dense_bound.compile_config(dense_cfg)(q, k, v)
    torch.cuda.synchronize()
    assert torch.equal(out, out_dense.view_as(out))
    # Natural-log LSE against the dense kernel's base-2 LSE.
    torch.testing.assert_close(
        lse.float(), lse_dense.view_as(lse).float() * math.log(2.0), atol=1e-4, rtol=0
    )
    out_ref, lse_ref = _reference(q, k, v, None, block_mask)
    _assert_close_to_reference(out, lse, out_ref, lse_ref)


@skipIfNotCUDA()
@onlyBackends(["cute"])
def test_causal_blockmask_is_bitwise_the_causal_plan() -> None:
    batch, heads, seq = 2, 4, 2048
    q, k, v = _qkv(batch, heads, seq, seq)
    block_mask = create_block_mask(
        _causal, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    args = _flex_args(q, k, v, _identity, block_mask)
    out, lse = _run_flex(args, PAIR_DESC)
    out_ref, lse_ref = _reference(q, k, v, None, block_mask)
    _assert_close_to_reference(out, lse, out_ref, lse_ref)
    causal = _cute_kernel(import_path(EXAMPLES_DIR / "attention.py").causal_attention)
    causal_bound = causal.bind((q, k, v))
    causal_cfg = causal_bound._normalized_config_copy(
        helion.Config(
            **{
                **causal_bound.config_spec.default_config().config,
                **PAIR_DESC,
                cute_flash.FLASH_CAUSAL_LOOP_SPLIT_KEY: True,
                cute_flash.FLASH_CAUSAL_MASK_STOREBACK_KEY: True,
                cute_flash.FLASH_CAUSAL_SKIP_DEAD_TILE_KEY: False,
                cute_flash.FLASH_CAUSAL_STAGGER_KEY: False,
            }
        )
    )
    resolved = causal_bound.config_spec._resolve_cute_flash_config(causal_cfg.config)
    assert resolved.causal_mask_storeback and resolved.causal_loop_split
    out_causal, _ = causal_bound.compile_config(causal_cfg)(q, k, v)
    torch.cuda.synchronize()
    assert torch.equal(out, out_causal.view_as(out))


_REFERENCE_CASES = {
    # name: (M, N, mask, heads, kv_heads, block sizes)
    "empty_rows": (2048, 2048, _empty_rows, 4, None, (128, 128)),
    "m_smaller": (1024, 2048, _band, 4, None, (128, 128)),
    "kv_smaller": (2048, 1024, _band, 4, None, (128, 128)),
    "ragged_m": (2048 + 128, 2048 + 128, _causal, 4, None, (128, 128)),
    "ragged_n": (2048, 2048 + 64, _band, 4, None, (128, 128)),
    "one_kv_block": (512, 128, _sliding, 4, None, (128, 128)),
    "gqa": (2048, 2048, _causal, 8, 2, (128, 128)),
    "q256": (2048 + 128, 2048 + 128, _causal, 4, None, (256, 128)),
}


@skipIfNotCUDA()
@onlyBackends(["cute"])
@pytest.mark.parametrize("case", sorted(_REFERENCE_CASES))
@pytest.mark.parametrize(
    "overrides",
    [None, PAIR_DESC, STAGED_ASC],
    ids=["default", "pair_desc", "staged_asc"],
)
def test_flex_matches_torch_flex_attention(
    case: str, overrides: dict[str, object] | None
) -> None:
    q_seq, kv_seq, mask, heads, kv_heads, block_sizes = _REFERENCE_CASES[case]
    batch = 2
    enable_gqa = kv_heads is not None
    q, k, v = _qkv(batch, heads, q_seq, kv_seq, kv_heads=kv_heads)
    block_mask = create_block_mask(
        mask, batch, heads, q_seq, kv_seq, device=DEVICE, BLOCK_SIZE=block_sizes
    )
    args = _flex_args(q, k, v, _identity, block_mask, enable_gqa=enable_gqa)
    out, lse = _run_flex(args, overrides)
    out_ref, lse_ref = _reference(q, k, v, None, block_mask, enable_gqa=enable_gqa)
    _assert_close_to_reference(out, lse, out_ref, lse_ref)


@skipIfNotCUDA()
@onlyBackends(["cute"])
def test_score_mod_with_a_table_gather() -> None:
    batch, heads, seq = 2, 4, 2048
    q, k, v = _qkv(batch, heads, seq, seq)
    torch.manual_seed(1)
    table = torch.randn(2 * seq, device=DEVICE)
    offset = seq

    def score_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return score + hl.load(table, [kv_idx - q_idx + offset])

    def score_ref(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return score + table[kv_idx - q_idx + offset]

    block_mask = create_block_mask(
        _causal, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    args = _flex_args(q, k, v, score_mod, block_mask)
    out_ref, lse_ref = _reference(q, k, v, score_ref, block_mask)
    for overrides in (None, PAIR_DESC):
        out, lse = _run_flex(args, overrides)
        _assert_close_to_reference(out, lse, out_ref, lse_ref)


@skipIfNotCUDA()
@onlyBackends(["cute"])
@pytest.mark.parametrize(
    "overrides", [PAIR_DESC, STAGED_ASC], ids=["pair_desc", "staged_asc"]
)
def test_aux_smem_staging_is_bitwise_the_global_gather(
    overrides: dict[str, object],
) -> None:
    """``cute_flash_flex_aux_smem``: a data-dependent gather
    (``bias[h, idx[b, q, kv]]``, the graphormer-spatial shape of score_mod
    with out-of-range indices) read from the shared-memory copy through the
    sentinel select gives the same output and LSE as the clamped global
    gather, at 16- and 32-column chunks; both match torch (where
    out-of-range indices read 0, the ``hl.load`` convention)."""
    batch, heads, seq = 2, 4, 1024
    q, k, v = _qkv(batch, heads, seq, seq)
    torch.manual_seed(2)
    idx = torch.randint(-1, 9, (batch, seq, seq), device=DEVICE, dtype=torch.int32)
    bias = torch.randn(heads, 7, device=DEVICE)

    def score_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return score + hl.load(bias, [h, hl.load(idx, [b, q_idx, kv_idx])])

    def score_ref(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        i = idx[b, q_idx, kv_idx]
        ok = (i >= 0) & (i < 7)
        return score + torch.where(ok, bias[h, i.clamp(0, 6)], 0.0)

    block_mask = create_block_mask(
        _band, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    args = _flex_args(q, k, v, score_mod, block_mask)
    results: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for budget in (0, 4096):
        fn, resolved = _compile_flex(
            args, {**overrides, cute_flash.FLASH_FLEX_AUX_SMEM_KEY: budget}
        )
        assert resolved.flex_aux_smem == budget
        out, lse = fn(*args)
        torch.cuda.synchronize()
        results[budget] = (out, lse)
    out, lse = results[4096]
    assert torch.equal(out, results[0][0])
    assert torch.equal(lse, results[0][1])
    out_ref, lse_ref = _reference(q, k, v, score_ref, block_mask)
    _assert_close_to_reference(out, lse, out_ref, lse_ref)


@skipIfNotCUDA()
@onlyBackends(["cute"])
@pytest.mark.parametrize(
    "overrides", [PAIR_DESC, STAGED_ASC], ids=["pair_desc", "staged_asc"]
)
def test_strided_table_runs_match_torch(overrides: dict[str, object]) -> None:
    """The strided-run form: ``types[b, q, kv, n]`` (int32, three literal
    trailing indices sharing one run per four-lane group) feeding gathers
    from a staged table (budget 4096), the graphormer-edge score_mod shape."""
    batch, heads, seq = 2, 4, 1024
    q, k, v = _qkv(batch, heads, seq, seq)
    torch.manual_seed(3)
    types = torch.randint(0, 8, (batch, seq, seq, 3), device=DEVICE, dtype=torch.int32)
    lengths = torch.randint(0, 4, (batch, seq, seq), device=DEVICE, dtype=torch.int32)
    table = torch.randn(heads, 24, device=DEVICE)

    def score_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        length = hl.load(lengths, [b, q_idx, kv_idx])
        total = torch.zeros_like(score)
        for n in range(3):
            et = hl.load(types, [b, q_idx, kv_idx, n])
            c = hl.load(table, [h, et + n * 8])
            total = total + torch.where(length > n, c, 0.0)
        return score + total / torch.clamp(length, min=1).to(torch.float32)

    def score_ref(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        length = lengths[b, q_idx, kv_idx]
        total = torch.zeros_like(score)
        for n in range(3):
            c = table[h, types[b, q_idx, kv_idx, n] + n * 8]
            total = total + torch.where(length > n, c, 0.0)
        return score + total / torch.clamp(length, min=1).to(torch.float32)

    block_mask = create_block_mask(
        _causal, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    args = _flex_args(q, k, v, score_mod, block_mask)
    out_ref, lse_ref = _reference(q, k, v, score_ref, block_mask)
    fn, resolved = _compile_flex(
        args, {**overrides, cute_flash.FLASH_FLEX_AUX_SMEM_KEY: 4096}
    )
    assert resolved.flex_aux_smem == 4096
    out, lse = fn(*args)
    torch.cuda.synchronize()
    _assert_close_to_reference(out, lse, out_ref, lse_ref)


def _edge_like_score_args(seed: int) -> tuple[Any, ...]:
    """Flex arguments with the graphormer-edge shape of score_mod over the band
    mask: a strided int32 row-table run (``types``, three reads per element),
    a row-table vector (``lengths``) and a gathered bias table."""
    batch, heads, seq = 2, 4, 1024
    q, k, v = _qkv(batch, heads, seq, seq)
    torch.manual_seed(seed)
    types = torch.randint(0, 8, (batch, seq, seq, 3), device=DEVICE, dtype=torch.int32)
    lengths = torch.randint(0, 4, (batch, seq, seq), device=DEVICE, dtype=torch.int32)
    table = torch.randn(heads, 24, device=DEVICE)

    def score_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        length = hl.load(lengths, [b, q_idx, kv_idx])
        total = torch.zeros_like(score)
        for n in range(3):
            et = hl.load(types, [b, q_idx, kv_idx, n])
            c = hl.load(table, [h, et + n * 8])
            total = total + torch.where(length > n, c, 0.0)
        return score + total / torch.clamp(length, min=1).to(torch.float32)

    block_mask = create_block_mask(
        _band, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    return _flex_args(q, k, v, score_mod, block_mask)


def _spatial_score_args(seed: int) -> tuple[Any, ...]:
    """Flex arguments with the graphormer-spatial shape of score_mod over the
    band mask: one int32 row-table vector feeding a gathered bias table."""
    batch, heads, seq = 2, 4, 1024
    q, k, v = _qkv(batch, heads, seq, seq)
    torch.manual_seed(seed)
    dist = torch.randint(0, 7, (batch, seq, seq), device=DEVICE, dtype=torch.int32)
    bias = torch.randn(heads, 7, device=DEVICE)

    def spatial_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return score + hl.load(bias, [h, hl.load(dist, [b, q_idx, kv_idx])])

    block_mask = create_block_mask(
        _band, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    return _flex_args(q, k, v, spatial_mod, block_mask)


# Each case below compiles two or three fa4 kernels, which runs past the
# suite's 60 s per-test timeout under CI's eight xdist workers on one B200.
_MULTI_COMPILE_TIMEOUT = pytest.mark.timeout(300)


@skipIfNotCUDA()
@onlyBackends(["cute"])
@_MULTI_COMPILE_TIMEOUT
@pytest.mark.parametrize(
    "overrides", [PAIR_DESC, STAGED_ASC], ids=["pair_desc", "staged_asc"]
)
def test_next_chunk_table_prefetch_is_bitwise_the_in_chunk_loads(
    overrides: dict[str, object],
) -> None:
    """``cute_flash_flex_prefetch``: the row-table vector and the shared
    strided run issued ahead (budgets 32: the vector one chunk ahead, and
    128: both one ahead at 32 columns, two ahead at 16) give the same
    output and LSE as the in-chunk loads, on partial and full tiles."""
    args = _edge_like_score_args(4)
    results: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for budget in (0, 32, 128):
        fn, resolved = _compile_flex(
            args, {**overrides, cute_flash.FLASH_FLEX_PREFETCH_KEY: budget}
        )
        assert resolved.flex_prefetch == budget
        out, lse = fn(*args)
        torch.cuda.synchronize()
        results[budget] = (out, lse)
    for budget in (32, 128):
        assert torch.equal(results[budget][0], results[0][0])
        assert torch.equal(results[budget][1], results[0][1])


@skipIfNotCUDA()
@onlyBackends(["cute"])
@_MULTI_COMPILE_TIMEOUT
@pytest.mark.parametrize(
    "overrides", [PAIR_DESC, STAGED_ASC], ids=["pair_desc", "staged_asc"]
)
def test_deep_table_prefetch_lead_is_bitwise_the_in_chunk_loads(
    overrides: dict[str, object],
) -> None:
    """A single-vector score_mod at budget 128 runs three (32 columns) /
    seven (16 columns) chunks ahead and gives the same output and LSE as
    the in-chunk loads."""
    args = _spatial_score_args(4)
    deep: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for budget in (0, 128):
        fn, resolved = _compile_flex(
            args, {**overrides, cute_flash.FLASH_FLEX_PREFETCH_KEY: budget}
        )
        assert resolved.flex_prefetch == budget
        out, lse = fn(*args)
        torch.cuda.synchronize()
        deep[budget] = (out, lse)
    assert torch.equal(deep[128][0], deep[0][0])
    assert torch.equal(deep[128][1], deep[0][1])


# The TMA-epilogue pair body with 32-column chunks and the output staged in
# the dead Q stages: the only 32-column body whose shared memory holds the
# cooperative staging of the graphormer-edge pair (136 KB).
WIDE_OVER_Q: dict[str, object] = {
    **PAIR_DESC,
    cute_flash.FLASH_S_LOAD_REP_KEY: 32,
    cute_flash.FLASH_O_OVER_Q_KEY: True,
}


@skipIfNotCUDA()
@onlyBackends(["cute"])
@_MULTI_COMPILE_TIMEOUT
@pytest.mark.parametrize(
    ("overrides", "width"),
    [
        (PAIR_DESC, 64),
        (PAIR_DESC, 32),
        (WIDE_OVER_Q, 128),
        (WIDE_OVER_Q, 64),
        (WIDE_OVER_Q, 32),
    ],
    ids=[
        "pair_desc_16-64",
        "pair_desc_16-32",
        "over_q_32-128",
        "over_q_32-64",
        "over_q_32-32",
    ],
)
def test_cooperative_table_loads_are_bitwise_the_per_lane_vectors(
    overrides: dict[str, object], width: int
) -> None:
    """``cute_flash_flex_coalesce``: the row-table runs fetched
    warp-cooperatively (whole lines of 4 / 8 / 16 rows per request at 128 /
    64 / 32 bytes per row) and staged through the warp's shared-memory
    buffer give the same output and LSE as the per-lane vectors, with and
    without the ring prefetch, on partial and full tiles; at 16-column
    chunks the 64-byte spans cap the request at 64, so the pair_desc body
    offers no 128."""
    args = _edge_like_score_args(5)
    fn, resolved = _compile_flex(
        args, {**overrides, cute_flash.FLASH_FLEX_COALESCE_KEY: 0}
    )
    assert resolved.flex_coalesce == 0
    out_ref, lse_ref = fn(*args)
    torch.cuda.synchronize()
    for prefetch in (0, 128):
        fn, resolved = _compile_flex(
            args,
            {
                **overrides,
                cute_flash.FLASH_FLEX_COALESCE_KEY: width,
                cute_flash.FLASH_FLEX_PREFETCH_KEY: prefetch,
            },
        )
        assert resolved.flex_coalesce == width, (width, resolved.flex_coalesce)
        assert resolved.flex_prefetch == prefetch
        out, lse = fn(*args)
        torch.cuda.synchronize()
        assert torch.equal(out, out_ref), (width, prefetch)
        assert torch.equal(lse, lse_ref), (width, prefetch)


@skipIfNotCUDA()
@onlyBackends(["cute"])
@_MULTI_COMPILE_TIMEOUT
@pytest.mark.parametrize(
    ("overrides", "width"),
    [(PAIR_DESC, 64), (WIDE_OVER_Q, 128)],
    ids=["pair_desc_16", "over_q_32"],
)
def test_cooperative_loads_of_the_spatial_vector_table_are_bitwise(
    overrides: dict[str, object], width: int
) -> None:
    """The spatial shape (one int32 vector table, a 36 KB staging at 32
    columns): a 128-byte request resolves to the widest width the body
    offers and gives the same output and LSE as the per-lane vectors."""
    args = _spatial_score_args(5)
    fn, resolved = _compile_flex(
        args, {**overrides, cute_flash.FLASH_FLEX_COALESCE_KEY: 0}
    )
    assert resolved.flex_coalesce == 0
    out_ref, lse_ref = fn(*args)
    torch.cuda.synchronize()
    fn, resolved = _compile_flex(
        args, {**overrides, cute_flash.FLASH_FLEX_COALESCE_KEY: 128}
    )
    assert resolved.flex_coalesce == width
    out, lse = fn(*args)
    torch.cuda.synchronize()
    assert torch.equal(out, out_ref)
    assert torch.equal(lse, lse_ref)


@skipIfNotCUDA()
@onlyBackends(["cute"])
def test_random_table_mask_with_empty_rows_is_stable_under_repeated_launches() -> None:
    batch, heads, seq = 4, 8, 2048
    torch.manual_seed(1)
    table = (torch.rand(batch, heads, seq // 128, seq // 128, device=DEVICE) < 0.4).to(
        torch.int32
    )
    table[0, 1, 2:4] = 0  # an empty pair
    table[1, 3] = 0  # an empty head

    def table_mask(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return hl.load(table, [b, h, q_idx // 128, kv_idx // 128]) != 0

    def table_mask_ref(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return table[b, h, q_idx // 128, kv_idx // 128] != 0

    q, k, v = _qkv(batch, heads, seq, seq)
    block_mask = create_block_mask(
        table_mask_ref, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    args = _flex_args(q, k, v, _identity, block_mask, mask_mod=table_mask)
    fn, _ = _compile_flex(args, PAIR_DESC)
    out, lse = fn(*args)
    torch.cuda.synchronize()
    out_ref, lse_ref = _reference(q, k, v, None, block_mask)
    _assert_close_to_reference(out, lse, out_ref, lse_ref)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=DEVICE)
    for _ in range(100):
        flush.zero_()
        out_again, _ = fn(*args)
    torch.cuda.synchronize()
    assert torch.equal(out_again, out)


@skipIfNotCUDA()
@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("q_seq", "kv_seq", "body"),
    [
        pytest.param(2048, 2048, PAIR_DESC, id="asymmetric-desc"),
        pytest.param(2048, 2048, STAGED_ASC, id="asymmetric-asc"),
        pytest.param(128, 2176, STAGED_ASC, id="dead-second-stage"),
        # The TMA epilogue of a ragged query length stores no dead tile
        # (``test_ragged_query_tma_epilogue_is_bitwise_the_staged_store``).
        pytest.param(128, 2176, PAIR_DESC, id="dead-second-stage-tma-store"),
    ],
)
def test_absent_stage_mma_skip_is_bitwise_the_unskipped_kernel(
    q_seq: int, kv_seq: int, body: dict[str, object]
) -> None:
    """``cute_flash_flex_skip_absent``: a random table mask (the two 128-row
    stages of a pair rarely share a tile, so each stage is absent on most
    steps) and a 128-row query (one pair whose second stage is dead on every
    step) give the same bits with the MMA skip on and off: a skipped stage's
    softmax zero-stores P, and a zero P adds nothing."""
    batch, heads = 4, 8
    torch.manual_seed(2)
    table = (
        torch.rand(batch, heads, q_seq // 128, -(-kv_seq // 128), device=DEVICE) < 0.35
    ).to(torch.int32)
    table[0, 1, 0] = 0  # an empty row (a stage absent on every step)

    def table_mask(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return hl.load(table, [b, h, q_idx // 128, kv_idx // 128]) != 0

    def table_mask_ref(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return table[b, h, q_idx // 128, kv_idx // 128] != 0

    q, k, v = _qkv(batch, heads, q_seq, kv_seq)
    block_mask = create_block_mask(
        table_mask_ref,
        batch,
        heads,
        q_seq,
        kv_seq,
        device=DEVICE,
        BLOCK_SIZE=(128, 128),
    )
    args = _flex_args(q, k, v, _identity, block_mask, mask_mod=table_mask)
    off_fn, off_cfg = _compile_flex(
        args, {**body, cute_flash.FLASH_FLEX_SKIP_ABSENT_KEY: False}
    )
    on_fn, on_cfg = _compile_flex(
        args, {**body, cute_flash.FLASH_FLEX_SKIP_ABSENT_KEY: True}
    )
    assert not off_cfg.flex_skip_absent
    assert on_cfg.flex_skip_absent
    out_off, lse_off = off_fn(*args)
    out_on, lse_on = on_fn(*args)
    torch.cuda.synchronize()
    assert torch.equal(out_on, out_off)
    assert torch.equal(lse_on, lse_off)
    out_ref, lse_ref = _reference(q, k, v, None, block_mask)
    _assert_close_to_reference(out_on, lse_on, out_ref, lse_ref)


@skipIfNotCUDA()
@onlyBackends(["cute"])
@pytest.mark.parametrize("q_seq", [64, 128, 384, 1152])
def test_ragged_query_tma_epilogue_is_bitwise_the_staged_store(q_seq: int) -> None:
    """The TMA output store of a ragged query length (``M % 256 != 0``) stores
    only the live rows of the last pair: a full tile through TMA, a half-live
    tile through the row-guarded vector store, a dead tile not at all. Its
    128-row box over the coalesced (S, D, B) output view used to land in the
    next head's first rows (zeros for a dead second stage, racing that head's
    own store). Bitwise the staged vector store of the same body (the same
    sO contents, another store path) and close to torch at every head."""
    batch, heads, kv_seq = 4, 8, 2176
    torch.manual_seed(4)
    table = (
        torch.rand(batch, heads, -(-q_seq // 128), -(-kv_seq // 128), device=DEVICE)
        < 0.4
    ).to(torch.int32)
    table[0, 1, 0] = 0  # an empty row (zeros and -inf, nothing else)

    def table_mask(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return hl.load(table, [b, h, q_idx // 128, kv_idx // 128]) != 0

    def table_mask_ref(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return table[b, h, q_idx // 128, kv_idx // 128] != 0

    q, k, v = _qkv(batch, heads, q_seq, kv_seq)
    block_mask = create_block_mask(
        table_mask_ref,
        batch,
        heads,
        q_seq,
        kv_seq,
        device=DEVICE,
        BLOCK_SIZE=(128, 128),
    )
    args = _flex_args(q, k, v, _identity, block_mask, mask_mod=table_mask)
    staged = {
        **PAIR_DESC,
        cute_flash.FLASH_EPI_TMA_KEY: False,
        cute_flash.FLASH_EPI_STG_KEY: True,
    }
    out_tma, lse_tma = _run_flex(args, PAIR_DESC)
    out_stg, lse_stg = _run_flex(args, staged)
    assert torch.equal(out_tma, out_stg)
    assert torch.equal(lse_tma, lse_stg)
    out_ref, lse_ref = _reference(q, k, v, None, block_mask)
    _assert_close_to_reference(out_tma, lse_tma, out_ref, lse_ref)


@skipIfNotCUDA()
@onlyBackends(["cute"])
@pytest.mark.parametrize(
    ("q_seq", "kv_seq", "splits"),
    [
        pytest.param(128, 2176, True, id="128-row-query"),
        pytest.param(384, 2176, True, id="ragged-last-pair"),
        pytest.param(2048, 2048, False, id="two-live-stages"),
    ],
)
def test_dead_stage_split_merges_both_softmax_warpgroups(
    q_seq: int, kv_seq: int, splits: bool
) -> None:
    """``cute_flash_flex_dead_stage_split``: a pair whose second stage sees no
    tile walks the first stage's tiles on both softmax warpgroups and merges
    the two partial softmaxes; the result matches the unsplit kernel within
    the merge's rounding and the reference within bf16 tolerance. A pair with
    two live stages renders the split-capable body but never splits: bitwise
    the unsplit kernel."""
    batch, heads = 4, 8
    torch.manual_seed(3)
    table = (
        torch.rand(batch, heads, -(-q_seq // 128), -(-kv_seq // 128), device=DEVICE)
        < 0.4
    ).to(torch.int32)
    table[0, 1, 0] = 0  # an empty row: its stage is absent on every step

    def table_mask(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return hl.load(table, [b, h, q_idx // 128, kv_idx // 128]) != 0

    def table_mask_ref(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return table[b, h, q_idx // 128, kv_idx // 128] != 0

    q, k, v = _qkv(batch, heads, q_seq, kv_seq)
    block_mask = create_block_mask(
        table_mask_ref,
        batch,
        heads,
        q_seq,
        kv_seq,
        device=DEVICE,
        BLOCK_SIZE=(128, 128),
    )
    args = _flex_args(q, k, v, _identity, block_mask, mask_mod=table_mask)
    skip = {**STAGED_ASC, cute_flash.FLASH_FLEX_SKIP_ABSENT_KEY: True}
    off_fn, off_cfg = _compile_flex(
        args, {**skip, cute_flash.FLASH_FLEX_DEAD_STAGE_SPLIT_KEY: False}
    )
    on_fn, on_cfg = _compile_flex(
        args, {**skip, cute_flash.FLASH_FLEX_DEAD_STAGE_SPLIT_KEY: True}
    )
    assert not off_cfg.flex_dead_stage_split
    assert on_cfg.flex_dead_stage_split
    out_off, lse_off = off_fn(*args)
    out_on, lse_on = on_fn(*args)
    torch.cuda.synchronize()
    out_ref, lse_ref = _reference(q, k, v, None, block_mask)
    _assert_close_to_reference(out_on, lse_on, out_ref, lse_ref)
    if splits:
        # The split items' two partial softmaxes are merged in another
        # summation order: equal within fp32 rounding, not bitwise.
        torch.testing.assert_close(
            out_on.float(), out_off.float(), atol=1e-2, rtol=1e-2
        )
        finite = torch.isfinite(lse_off)
        torch.testing.assert_close(lse_on[finite], lse_off[finite], atol=1e-3, rtol=0)
        assert torch.equal(lse_on[~finite], lse_off[~finite])
    else:
        assert torch.equal(out_on, out_off)
        assert torch.equal(lse_on, lse_off)


@skipIfNotCUDA()
@onlyBackends(["cute"])
@pytest.mark.parametrize("kv_split", [2, 4])
@pytest.mark.parametrize(
    ("q_seq", "kv_seq", "body", "dead_split"),
    [
        pytest.param(64, 2176, STAGED_ASC, False, id="m64-staged"),
        pytest.param(128, 2176, PAIR_DESC, False, id="m128-tma"),
        pytest.param(128, 8320, PAIR_DESC, True, id="m128-dead-split-three-words"),
        pytest.param(384, 2176, STAGED_ASC, True, id="m384-ragged-dead-split"),
        pytest.param(2048, 2048, PAIR_DESC, False, id="aligned-tma"),
        pytest.param(
            2048 + 128,
            2048,
            {**PAIR_DESC, cute_flash.FLASH_PERSISTENT_KEY: True},
            False,
            id="ragged-persistent",
        ),
    ],
)
def test_split_kv_walk_matches_the_unsplit_kernel(
    kv_split: int, q_seq: int, kv_seq: int, body: dict[str, object], dead_split: bool
) -> None:
    """``cute_flash_kv_split``: every work item's KV walk over two or four
    CTAs, each walking a contiguous slice of the union walk and the last
    arriving warp of each 32-row group merging the partial outputs in split
    order. A random table mask with an empty row and an empty head (zeros
    and ``-inf``), ragged query lengths (M = 64, 128, 384, 2176), the
    dead-stage split on and off and the persistent item loop: within the
    merge's rounding of the unsplit kernel (another summation order, LSE
    within 1e-3; the empty rows bitwise) and close to torch; repeated
    launches agree bitwise (the arrival counters reset themselves)."""
    batch, heads = 4, 8
    torch.manual_seed(5)
    table = (
        torch.rand(batch, heads, -(-q_seq // 128), -(-kv_seq // 128), device=DEVICE)
        < 0.4
    ).to(torch.int32)
    table[0, 1, 0] = 0  # an empty row
    table[1, 3] = 0  # an empty head
    if q_seq >= 256:
        table[2, 5, 1] = 0  # a dead second stage of a live first stage

    def table_mask(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return hl.load(table, [b, h, q_idx // 128, kv_idx // 128]) != 0

    def table_mask_ref(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return table[b, h, q_idx // 128, kv_idx // 128] != 0

    q, k, v = _qkv(batch, heads, q_seq, kv_seq)
    block_mask = create_block_mask(
        table_mask_ref,
        batch,
        heads,
        q_seq,
        kv_seq,
        device=DEVICE,
        BLOCK_SIZE=(128, 128),
    )
    args = _flex_args(q, k, v, _identity, block_mask, mask_mod=table_mask)
    base = {
        **body,
        cute_flash.FLASH_FLEX_SKIP_ABSENT_KEY: dead_split,
        cute_flash.FLASH_FLEX_DEAD_STAGE_SPLIT_KEY: dead_split,
    }
    one_fn, one_cfg = _compile_flex(args, {**base, cute_flash.FLASH_KV_SPLIT_KEY: 1})
    split_fn, split_cfg = _compile_flex(
        args, {**base, cute_flash.FLASH_KV_SPLIT_KEY: kv_split}
    )
    assert one_cfg.kv_split == 1 and split_cfg.kv_split == kv_split
    assert split_cfg.flex_dead_stage_split is dead_split
    out_one, lse_one = one_fn(*args)
    out_split, lse_split = split_fn(*args)
    torch.cuda.synchronize()
    out_ref, lse_ref = _reference(q, k, v, None, block_mask)
    _assert_close_to_reference(out_split, lse_split, out_ref, lse_ref)
    torch.testing.assert_close(out_split.float(), out_one.float(), atol=1e-2, rtol=1e-2)
    finite = torch.isfinite(lse_one)
    assert torch.equal(finite, torch.isfinite(lse_split))
    torch.testing.assert_close(lse_split[finite], lse_one[finite], atol=1e-3, rtol=0)
    assert torch.equal(lse_split[~finite], lse_one[~finite])
    assert torch.all(out_split[~finite] == 0)
    for _ in range(3):
        out_again, lse_again = split_fn(*args)
        torch.cuda.synchronize()
        assert torch.equal(out_again, out_split)
        assert torch.equal(lse_again, lse_split)


def _copy_tables(args: tuple[Any, ...], block_mask: Any) -> None:
    """Rewrite the kernel's four tables in place from ``block_mask``."""
    full_num = block_mask.full_kv_num_blocks
    full_idx = block_mask.full_kv_indices
    if full_num is None or full_idx is None:
        full_num = torch.zeros_like(block_mask.kv_num_blocks)
        full_idx = torch.zeros_like(block_mask.kv_indices)
    for target, source in zip(
        args[4:8],
        (block_mask.kv_num_blocks, block_mask.kv_indices, full_num, full_idx),
        strict=True,
    ):
        target.copy_(source)


@skipIfNotCUDA()
@onlyBackends(["cute"])
def test_persistent_list_walk_is_bitwise_the_grid_and_follows_table_rewrites() -> None:
    batch, heads, seq = 2, 4, 2048 + 128  # a ragged last pair: 9 pairs, 17 rows
    rows = num_kv = -(-seq // 128)
    torch.manual_seed(2)
    table = (torch.rand(batch, heads, rows, num_kv, device=DEVICE) < 0.35).to(
        torch.int32
    )
    table[0, 1, 2:4] = 0  # an empty pair
    table[1, 3] = 0  # an empty head

    def table_mask(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return hl.load(table, [b, h, q_idx // 128, kv_idx // 128]) != 0

    def table_mask_ref(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return table[b, h, q_idx // 128, kv_idx // 128] != 0

    q, k, v = _qkv(batch, heads, seq, seq)
    block_mask = create_block_mask(
        table_mask_ref, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    args = _flex_args(q, k, v, _identity, block_mask, mask_mod=table_mask)
    fn_grid, _ = _compile_flex(args, PAIR_DESC)
    fn_persistent, resolved = _compile_flex(
        args, {**PAIR_DESC, cute_flash.FLASH_PERSISTENT_KEY: True}
    )
    assert resolved.persistent and resolved.persistent_loop == "while"

    def check() -> None:
        out, lse = fn_grid(*args)
        out_p, lse_p = fn_persistent(*args)
        torch.cuda.synchronize()
        assert torch.equal(out_p, out)
        finite = torch.isfinite(lse)
        assert torch.equal(finite, torch.isfinite(lse_p))
        assert torch.equal(lse_p[finite], lse[finite])
        out_ref, lse_ref = _reference(q, k, v, None, block_mask)
        _assert_close_to_reference(out_p, lse_p, out_ref, lse_ref)

    check()
    # The item order is rebuilt from the tables' version counters: a sparser
    # mask written in place (new counts, new longest-first order) is followed
    # without recompiling.
    table.copy_(torch.rand(batch, heads, rows, num_kv, device=DEVICE) < 0.15)
    table[0, 0] = 0
    block_mask = create_block_mask(
        table_mask_ref, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    _copy_tables(args, block_mask)
    check()


def _dilated(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    diff = torch.abs(q_idx - kv_idx)
    return (diff <= 512) & (diff % 2 == 0)


def _temporal(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    q_tok = (q_idx - 128) % 512
    kv_tok = (kv_idx - 128) % 512
    return (kv_idx < 128) | (q_idx < 128) | (torch.abs(q_tok - kv_tok) < 64)


@skipIfNotCUDA()
@onlyBackends(["cute"])
@pytest.mark.parametrize("mask", [_dilated, _temporal], ids=["dilated", "temporal"])
@pytest.mark.parametrize(
    "overrides", [PAIR_DESC, STAGED_ASC], ids=["pair_desc", "staged_asc"]
)
def test_keep_word_mask_form_is_bitwise_the_element_form(
    mask: Callable[..., Any], overrides: dict[str, object]
) -> None:
    """``cute_flash_flex_mask_form``: the per-chunk keep word (lane intervals,
    the parity pattern, lane-affine remainders, the tile-uniform ``kv < 128``)
    applied through the R2P select chain writes the same ``-inf`` pattern as
    the per-element predicate, at 16- and 32-column chunks and with a
    half-live last pair; both match torch."""
    batch, heads, seq = 2, 4, 2048 + 128
    q, k, v = _qkv(batch, heads, seq, seq)
    block_mask = create_block_mask(
        mask, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    args = _flex_args(q, k, v, _identity, block_mask)
    results: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for form in ("bitmask", "element"):
        fn, resolved = _compile_flex(
            args, {**overrides, cute_flash.FLASH_FLEX_MASK_FORM_KEY: form}
        )
        assert resolved.flex_mask_form == form
        out, lse = fn(*args)
        torch.cuda.synchronize()
        results[form] = (out, lse)
    out, lse = results["bitmask"]
    assert torch.equal(out, results["element"][0])
    assert torch.equal(lse, results["element"][1])
    out_ref, lse_ref = _reference(q, k, v, None, block_mask)
    _assert_close_to_reference(out, lse, out_ref, lse_ref)
