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
    args: tuple[Any, ...],
    overrides: dict[str, object] | None,
    *,
    expect_in_code: str | None = None,
    code_sink: list[str] | None = None,
) -> tuple[Any, cute_flash.FlashAttentionConfig]:
    """Compile the flex kernel at the default config plus ``overrides``,
    asserting the surface is the flex plan category and the body the walk
    (``code_sink`` receives the rendered source)."""
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
    # (The list walk is persistent by default; an override pins either way.)
    assert resolved.persistent == bool(
        {**base, **(overrides or {})}[cute_flash.FLASH_PERSISTENT_KEY]
    )
    code = bound.to_code(normalized)
    assert "flash_flex_status" in code
    if expect_in_code is not None:
        assert expect_in_code in code
    if code_sink is not None:
        code_sink.append(code)
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
    batch: int,
    heads: int,
    q_seq: int,
    kv_seq: int,
    *,
    kv_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    kv_heads = heads if kv_heads is None else kv_heads
    q = torch.randn(batch, heads, q_seq, 64, dtype=dtype, device=DEVICE)
    k = torch.randn(batch, kv_heads, kv_seq, 64, dtype=dtype, device=DEVICE)
    v = torch.randn(batch, kv_heads, kv_seq, 64, dtype=dtype, device=DEVICE)
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

# The prefetch and coalescing budgets below assume the unrolled PASS1 chunk
# loop: the rolled loop (``cute_flash_flex_rolled_chunks``, on by default for
# the edge-like hook's size) has no register prefetch ring.
UNROLLED: dict[str, object] = {cute_flash.FLASH_FLEX_ROLLED_CHUNKS_KEY: False}


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
    output and LSE as the in-chunk loads, on partial and full tiles; the
    body is pinned unrolled (``UNROLLED``)."""
    args = _edge_like_score_args(4)
    results: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for budget in (0, 32, 128):
        fn, resolved = _compile_flex(
            args, {**overrides, **UNROLLED, cute_flash.FLASH_FLEX_PREFETCH_KEY: budget}
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
            args, {**overrides, **UNROLLED, cute_flash.FLASH_FLEX_PREFETCH_KEY: budget}
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
    offers no 128.  The body is pinned unrolled (``UNROLLED``) so the ring
    prefetch cases exist."""
    args = _edge_like_score_args(5)
    fn, resolved = _compile_flex(
        args, {**overrides, **UNROLLED, cute_flash.FLASH_FLEX_COALESCE_KEY: 0}
    )
    assert resolved.flex_coalesce == 0
    out_ref, lse_ref = fn(*args)
    torch.cuda.synchronize()
    for prefetch in (0, 128):
        fn, resolved = _compile_flex(
            args,
            {
                **overrides,
                **UNROLLED,
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
        args, {**overrides, **UNROLLED, cute_flash.FLASH_FLEX_COALESCE_KEY: 0}
    )
    assert resolved.flex_coalesce == 0
    out_ref, lse_ref = fn(*args)
    torch.cuda.synchronize()
    fn, resolved = _compile_flex(
        args, {**overrides, **UNROLLED, cute_flash.FLASH_FLEX_COALESCE_KEY: 128}
    )
    assert resolved.flex_coalesce == width
    out, lse = fn(*args)
    torch.cuda.synchronize()
    assert torch.equal(out, out_ref)
    assert torch.equal(lse, lse_ref)


@skipIfNotCUDA()
@onlyBackends(["cute"])
@_MULTI_COMPILE_TIMEOUT
@pytest.mark.parametrize(
    ("overrides", "width"),
    [(PAIR_DESC, 64), (WIDE_OVER_Q, 128)],
    ids=["pair_desc_16", "over_q_32"],
)
def test_rolled_chunk_loop_is_bitwise_the_unrolled_kernel(
    overrides: dict[str, object], width: int
) -> None:
    """The rolled PASS1 chunk loop (``cute_flash_flex_rolled_chunks``), with
    and without the cp.async stream, gives the same output and LSE as the
    unrolled per-lane kernel."""
    args = _spatial_score_args(5)
    fn, resolved = _compile_flex(
        args, {**overrides, **UNROLLED, cute_flash.FLASH_FLEX_COALESCE_KEY: 0}
    )
    assert not resolved.flex_rolled_chunks
    out_ref, lse_ref = fn(*args)
    torch.cuda.synchronize()
    for coalesce in (0, width):
        fn, resolved = _compile_flex(
            args,
            {
                **overrides,
                cute_flash.FLASH_FLEX_COALESCE_KEY: coalesce,
                cute_flash.FLASH_FLEX_ROLLED_CHUNKS_KEY: True,
            },
        )
        assert resolved.flex_rolled_chunks and resolved.flex_prefetch == 0
        assert resolved.flex_coalesce == coalesce
        out, lse = fn(*args)
        torch.cuda.synchronize()
        assert torch.equal(out, out_ref), coalesce
        assert torch.equal(lse, lse_ref), coalesce


@skipIfNotCUDA()
@onlyBackends(["cute"])
@_MULTI_COMPILE_TIMEOUT
def test_item_words_are_bitwise_the_device_list_walk() -> None:
    """The host-built per-item bitmask words (``cute_flash_flex_item_words``,
    the default) give the same output and LSE as the device walk over the
    index lists, on the persistent loop and the plain grid."""
    batch, heads, seq = 2, 4, 2048
    q, k, v = _qkv(batch, heads, seq, seq)
    block_mask = create_block_mask(
        _band, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    args = _flex_args(q, k, v, _identity, block_mask)
    for persistent in (False, True):
        base = {**PAIR_DESC, cute_flash.FLASH_PERSISTENT_KEY: persistent}
        fn, resolved = _compile_flex(
            args,
            {**base, cute_flash.FLASH_FLEX_ITEM_WORDS_KEY: False},
            expect_in_code="for flash_flex_i in cutlass.range(flash_flex_cp0, unroll=1):",
        )
        assert not resolved.flex_item_words
        out_ref, lse_ref = fn(*args)
        torch.cuda.synchronize()
        fn, resolved = _compile_flex(
            args,
            {**base, cute_flash.FLASH_FLEX_ITEM_WORDS_KEY: True},
            expect_in_code="_flash_mFlexWords[flash_flex_wrow + cutlass.Int32(3)]",
        )
        assert resolved.flex_item_words
        out, lse = fn(*args)
        torch.cuda.synchronize()
        assert torch.equal(out, out_ref), persistent
        assert torch.equal(lse, lse_ref), persistent


def _half(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    # Every row sees the first half of the keys: equal counts (the identity
    # item order), full tiles only, not every tile.
    return kv_idx < 1024


@skipIfNotCUDA()
@onlyBackends(["cute"])
@_MULTI_COMPILE_TIMEOUT
def test_host_flags_skip_the_table_loads_bitwise() -> None:
    """The tables' host flags (``flex_item_flags``, a runtime scalar): the
    all-full bit binds the words as constants without loading them and the
    identity-order bit decodes the persistent item without the item table;
    both give the output and LSE of the table path (inference-tensor
    copies of the tables, which rebuild per call with the flags 0), on the
    persistent loop and the plain grid."""
    from helion.runtime.cute.flash_flex import FLEX_FLAG_ALL_FULL
    from helion.runtime.cute.flash_flex import FLEX_FLAG_IDENTITY_ORDER
    from helion.runtime.cute.flash_flex import flex_item_flags

    batch, heads, seq = 2, 4, 2048
    q, k, v = _qkv(batch, heads, seq, seq)
    for mask_mod, expect in (
        (noop_mask, FLEX_FLAG_ALL_FULL | FLEX_FLAG_IDENTITY_ORDER),
        (_half, FLEX_FLAG_IDENTITY_ORDER),
    ):
        block_mask = create_block_mask(
            mask_mod, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
        )
        args = _flex_args(q, k, v, _identity, block_mask)
        tables = args[4:8]
        geometry = {
            "q_block": 128,
            "batch": batch,
            "heads": heads,
            "num_pairs": seq // 256,
            "num_kv": seq // 128,
        }
        assert flex_item_flags(*tables, **geometry) == expect
        with torch.inference_mode():
            plain_tables = tuple(table.clone() for table in tables)
        assert flex_item_flags(*plain_tables, **geometry) == 0
        plain_args = (*args[:4], *plain_tables, *args[8:])
        for persistent in (False, True):
            fn, resolved = _compile_flex(
                args,
                {**PAIR_DESC, cute_flash.FLASH_PERSISTENT_KEY: persistent},
                expect_in_code="if _flash_flex_hflags & cutlass.Int32(1) == cutlass.Int32(0):",
            )
            assert resolved.flex_item_words
            out_ref, lse_ref = fn(*plain_args)
            torch.cuda.synchronize()
            out, lse = fn(*args)
            torch.cuda.synchronize()
            assert torch.equal(out, out_ref), (mask_mod.__name__, persistent)
            assert torch.equal(lse, lse_ref), (mask_mod.__name__, persistent)
            ref_out, ref_lse = _reference(q, k, v, None, block_mask)
            _assert_close_to_reference(out, lse, ref_out, ref_lse)


def _plain_table_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    """The same arguments with inference-tensor copies of the four tables:
    the uncached table path, host flags 0."""
    with torch.inference_mode():
        plain_tables = tuple(table.clone() for table in args[4:8])
    return (*args[:4], *plain_tables, *args[8:])


@skipIfNotCUDA()
@onlyBackends(["cute"])
@_MULTI_COMPILE_TIMEOUT
@pytest.mark.parametrize(
    ("q_seq", "block_rows", "persistent"),
    [(2048 + 128, 128, False), (2048 + 128, 128, True), (2048, 256, False)],
    ids=["odd_rows_grid", "odd_rows_persistent", "q_block_256"],
)
def test_host_flags_all_full_constants_on_odd_rows_and_256_row_tables(
    q_seq: int, block_rows: int, persistent: bool
) -> None:
    """The all-full constants where the device's word is not the plain full
    word of both stages: an odd row count (the last pair's missing
    second-stage row is the ``select_`` zero word) and 256-row BlockMask
    tables (one row, one word pair per item); bitwise the table path."""
    from helion.runtime.cute.flash_flex import FLEX_FLAG_ALL_FULL
    from helion.runtime.cute.flash_flex import FLEX_FLAG_IDENTITY_ORDER
    from helion.runtime.cute.flash_flex import flex_item_flags

    batch, heads, kv_seq = 2, 4, 2048
    q, k, v = _qkv(batch, heads, q_seq, kv_seq)
    block_mask = create_block_mask(
        noop_mask,
        batch,
        heads,
        q_seq,
        kv_seq,
        device=DEVICE,
        BLOCK_SIZE=(block_rows, 128),
    )
    args = _flex_args(q, k, v, _identity, block_mask)
    geometry = {
        "q_block": block_rows,
        "batch": batch,
        "heads": heads,
        "num_pairs": -(-q_seq // 256),
        "num_kv": kv_seq // 128,
    }
    assert (
        flex_item_flags(*args[4:8], **geometry)
        == FLEX_FLAG_ALL_FULL | FLEX_FLAG_IDENTITY_ORDER
    )
    full_word = (1 << (kv_seq // 128)) - 1
    rows = -(-q_seq // block_rows)
    if block_rows == 256:
        expect = f"flash_flex_f0_0 = cutlass.Int32({full_word})"
    else:
        expect = (
            f"flash_flex_f1_0 = cutlass.Int32(cutlass.select_(flash_m_tile1 < {rows}, "
            f"cutlass.Int32({full_word}), cutlass.Int32(0)))"
        )
    fn, resolved = _compile_flex(
        args,
        {**PAIR_DESC, cute_flash.FLASH_PERSISTENT_KEY: persistent},
        expect_in_code=expect,
    )
    assert resolved.flex_item_words
    out_ref, lse_ref = fn(*_plain_table_args(args))
    torch.cuda.synchronize()
    out, lse = fn(*args)
    torch.cuda.synchronize()
    assert torch.equal(out, out_ref)
    assert torch.equal(lse, lse_ref)
    ref_out, ref_lse = _reference(q, k, v, None, block_mask)
    _assert_close_to_reference(out, lse, ref_out, ref_lse)


@skipIfNotCUDA()
@onlyBackends(["cute"])
@_MULTI_COMPILE_TIMEOUT
def test_host_flags_follow_table_rewrites_and_graph_replay() -> None:
    """The flags live in the per-BlockMask cache behind the tables' version
    counters: rewriting a full mask's tables in place into a banded mask's
    flips them 3 -> 0 and the same compiled kernel (built with the band's
    mask hook, which full tiles never run) follows, bitwise the band's own
    tables; a CUDA-graph capture of the flags-3 mask rebuilds the tables in
    the graph with the flags 0 and replays the eager result bitwise."""
    from helion.runtime.cute.flash_flex import FLEX_FLAG_ALL_FULL
    from helion.runtime.cute.flash_flex import FLEX_FLAG_IDENTITY_ORDER
    from helion.runtime.cute.flash_flex import flex_item_flags

    batch, heads, seq = 2, 4, 2048
    q, k, v = _qkv(batch, heads, seq, seq)
    full = create_block_mask(
        noop_mask, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    band = create_block_mask(
        _band, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    args = _flex_args(q, k, v, _identity, full, mask_mod=_band)
    band_args = _flex_args(q, k, v, _identity, band)
    tables, band_tables = args[4:8], band_args[4:8]
    assert all(a.shape == b.shape for a, b in zip(tables, band_tables, strict=True))
    geometry = {
        "q_block": 128,
        "batch": batch,
        "heads": heads,
        "num_pairs": seq // 256,
        "num_kv": seq // 128,
    }
    fn, _ = _compile_flex(args, PAIR_DESC)
    assert (
        flex_item_flags(*tables, **geometry)
        == FLEX_FLAG_ALL_FULL | FLEX_FLAG_IDENTITY_ORDER
    )
    out_full, lse_full = fn(*args)
    torch.cuda.synchronize()
    ref_out, ref_lse = _reference(q, k, v, None, full)
    _assert_close_to_reference(out_full, lse_full, ref_out, ref_lse)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_lse = fn(*args)
    for _ in range(2):
        graph_out.fill_(0)
        graph_lse.fill_(0)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(graph_out, out_full)
        assert torch.equal(graph_lse, lse_full)
    # The in-place rewrite bumps the version counters: the flags follow.
    for table, source in zip(tables, band_tables, strict=True):
        table.copy_(source)
    assert flex_item_flags(*tables, **geometry) == 0
    out_band, lse_band = fn(*args)
    torch.cuda.synchronize()
    band_ref_out, band_ref_lse = _reference(q, k, v, None, band)
    _assert_close_to_reference(out_band, lse_band, band_ref_out, band_ref_lse)
    out_own, lse_own = fn(*band_args)
    torch.cuda.synchronize()
    assert torch.equal(out_band, out_own)
    assert torch.equal(lse_band, lse_own)


@skipIfNotCUDA()
@onlyBackends(["cute"])
@_MULTI_COMPILE_TIMEOUT
@pytest.mark.parametrize("case", ["gqa", "kv_split"])
def test_host_flags_with_grouped_queries_and_the_split_walk(case: str) -> None:
    """Grouped-query heads (two KV heads for four query heads) and the
    split-KV grid (``cute_flash_kv_split`` 2: two CTAs per pair item, the
    words guarded by the all-full bit on both) with the all-full flags:
    bitwise the table path, close to torch."""
    from helion.runtime.cute.flash_flex import FLEX_FLAG_ALL_FULL
    from helion.runtime.cute.flash_flex import FLEX_FLAG_IDENTITY_ORDER
    from helion.runtime.cute.flash_flex import flex_item_flags

    batch, heads, seq = 2, 4, 2048
    kv_heads = 2 if case == "gqa" else None
    q, k, v = _qkv(batch, heads, seq, seq, kv_heads=kv_heads)
    block_mask = create_block_mask(
        noop_mask, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    args = _flex_args(q, k, v, _identity, block_mask, enable_gqa=kv_heads is not None)
    geometry = {
        "q_block": 128,
        "batch": batch,
        "heads": heads,
        "num_pairs": seq // 256,
        "num_kv": seq // 128,
    }
    assert (
        flex_item_flags(*args[4:8], **geometry)
        == FLEX_FLAG_ALL_FULL | FLEX_FLAG_IDENTITY_ORDER
    )
    overrides: dict[str, object] = {**PAIR_DESC, cute_flash.FLASH_PERSISTENT_KEY: False}
    if case == "kv_split":
        overrides[cute_flash.FLASH_KV_SPLIT_KEY] = 2
    fn, resolved = _compile_flex(
        args,
        overrides,
        expect_in_code="_flash_flex_hflags = cutlass.Int32(flash_flex_host_flags)",
    )
    assert resolved.flex_item_words
    if case == "kv_split":
        assert resolved.kv_split == 2
    out_ref, lse_ref = fn(*_plain_table_args(args))
    torch.cuda.synchronize()
    out, lse = fn(*args)
    torch.cuda.synchronize()
    assert torch.equal(out, out_ref)
    assert torch.equal(lse, lse_ref)
    ref_out, ref_lse = _reference(
        q, k, v, None, block_mask, enable_gqa=kv_heads is not None
    )
    _assert_close_to_reference(out, lse, ref_out, ref_lse)


RESIDENT_REP32: dict[str, object] = {
    **PAIR_DESC,
    cute_flash.FLASH_S_LOAD_REP_KEY: 32,
    cute_flash.FLASH_P_STORE_REP_KEY: 32,
    cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: False,
    cute_flash.FLASH_SOFTMAX_REGS_KEY: 200,
    cute_flash.FLASH_CORR_REGS_KEY: 64,
    cute_flash.FLASH_OTHER_REGS_KEY: 48,
}


RESIDENT_REP16_SPLIT: dict[str, object] = {
    **RESIDENT_REP32,
    cute_flash.FLASH_P_STORE_REP_KEY: 16,
    cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: True,
}
RESIDENT_REP32_SPLIT: dict[str, object] = {
    **RESIDENT_REP32,
    cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: True,
}


@skipIfNotCUDA()
@onlyBackends(["cute"])
@_MULTI_COMPILE_TIMEOUT
@pytest.mark.parametrize(
    ("persistent", "store"),
    [
        (False, "rep32"),
        (True, "rep32"),
        (False, "rep16_split"),
        (True, "rep16_split"),
        (False, "rep32_split"),
        (True, "rep32_split"),
    ],
    ids=[
        "grid_rep32",
        "persistent_rep32",
        "grid_rep16_split",
        "persistent_rep16_split",
        "grid_rep32_split",
        "persistent_rep32_split",
    ],
)
def test_resident_pass2_is_bitwise_the_reload_pass2(
    persistent: bool, store: str
) -> None:
    """``cute_flash_s_resident``: PASS2 over the chunks PASS1 left in registers
    gives the same output and LSE as the TMEM-reload PASS2 (the per-chunk
    numerics are shared), on a mask-only walk with partial and full tiles,
    with Rep32 P stores and with Rep16 stores keeping the split P arrival."""
    batch, heads, seq = 2, 4, 2048
    q, k, v = _qkv(batch, heads, seq, seq)
    block_mask = create_block_mask(
        _band, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    args = _flex_args(q, k, v, _identity, block_mask)
    body = {
        "rep32": RESIDENT_REP32,
        "rep16_split": RESIDENT_REP16_SPLIT,
        "rep32_split": RESIDENT_REP32_SPLIT,
    }[store]
    base = {**body, cute_flash.FLASH_PERSISTENT_KEY: persistent}
    fn, resolved = _compile_flex(
        args,
        base,
        expect_in_code={
            "rep32": "fa4_disc_exp_convert_store_rep32_pipe(",
            "rep16_split": "fa4_disc_exp_convert_store_pipe(",
            "rep32_split": "fa4_disc_exp_convert_store_rep32_split(",
        }[store],
    )
    assert not resolved.s_resident
    out_ref, lse_ref = fn(*args)
    torch.cuda.synchronize()
    fn, resolved = _compile_flex(
        args,
        {**base, cute_flash.FLASH_S_RESIDENT_KEY: True},
        expect_in_code={
            "rep32": "fa4_disc_exp_convert_store_resident_rep32((flash_res_frg0, flash_res_frg1, flash_res_frg2, flash_res_frg3), ",
            "rep16_split": "fa4_disc_resident_chunk16(flash_res_frg0, ",
            "rep32_split": "fa4_disc_resident_pair32(flash_p_sum, flash_res_frg0, flash_res_frg1, ",
        }[store],
    )
    assert resolved.s_resident
    out, lse = fn(*args)
    torch.cuda.synchronize()
    assert torch.equal(out, out_ref)
    assert torch.equal(lse, lse_ref)


@skipIfNotCUDA()
@onlyBackends(["cute"])
@_MULTI_COMPILE_TIMEOUT
@pytest.mark.parametrize("resident", [False, True], ids=["reload", "resident"])
def test_tmem_rowmax_is_bitwise_the_software_rowmax_on_the_list_walk(
    resident: bool,
) -> None:
    """``cute_flash_rowmax="tmem"`` on the list walk's full tiles (the hardware
    per-load maxima) gives the same output and LSE as the software FMNMX tree
    (the maximum is exact in any order), with the reload and the resident PASS2."""
    batch, heads, seq = 2, 4, 2048
    q, k, v = _qkv(batch, heads, seq, seq)
    block_mask = create_block_mask(
        _band, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    args = _flex_args(q, k, v, _identity, block_mask)
    base = {**RESIDENT_REP32, cute_flash.FLASH_S_RESIDENT_KEY: resident}
    fn, resolved = _compile_flex(args, base)
    assert resolved.rowmax == "software" and resolved.s_resident is resident
    bound = _flex_kernel().bind(args)
    probe = bound.config_spec._resolve_cute_flash_config(
        bound._normalized_config_copy(
            helion.Config(
                **{
                    **bound.config_spec.default_config().config,
                    **base,
                    cute_flash.FLASH_ROWMAX_KEY: "tmem",
                }
            )
        ).config
    )
    if probe.rowmax != "tmem":
        pytest.skip("this device's policy has no TMEM row reduction (GB300 only)")
    out_ref, lse_ref = fn(*args)
    torch.cuda.synchronize()
    fn, resolved = _compile_flex(
        args,
        {**base, cute_flash.FLASH_ROWMAX_KEY: "tmem"},
        expect_in_code=(
            "(flash_res_frg3, flash_res_red3))" if resident else "disc_rowmax_ldred("
        ),
    )
    assert resolved.rowmax == "tmem"
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


@skipIfNotCUDA()
@onlyBackends(["cute"])
@_MULTI_COMPILE_TIMEOUT
@pytest.mark.parametrize(
    ("persistent", "seq"),
    [(False, 2048), (True, 2048), (False, 16384), (True, 16384)],
    ids=["grid_words", "persistent_words", "grid_cursor", "persistent_cursor"],
)
def test_shared_softmax_region_is_bitwise_the_per_stage_regions(
    persistent: bool, seq: int
) -> None:
    """``cute_flash_softmax_region="shared"``: one softmax region for both
    query stages (the stage a warp-uniform runtime value) gives the same
    output and LSE as the two per-stage regions with the same exp cadence,
    on a mask-only walk with partial, full and absent tiles, over the
    register words (2K) and the host words table's cursor (16K), on the
    grid and the persistent walks. (One CTA per pair: the split-KV walk
    the small 2K grid defaults to keeps the per-stage regions.)"""
    batch, heads = (2, 4) if seq == 2048 else (1, 2)
    q, k, v = _qkv(batch, heads, seq, seq)
    block_mask = create_block_mask(
        _band, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    args = _flex_args(q, k, v, _identity, block_mask)
    base = {
        cute_flash.FLASH_PERSISTENT_KEY: persistent,
        cute_flash.FLASH_KV_SPLIT_KEY: 1,
        cute_flash.FLASH_E2E_SCHEDULE_KEY: "16/6",
        cute_flash.FLASH_E2E_OFFSET0_KEY: 3,
        cute_flash.FLASH_E2E_OFFSET_KEY: 3,
    }
    fn, resolved = _compile_flex(args, base)
    assert resolved.softmax_region == "per_stage"
    out_ref, lse_ref = fn(*args)
    torch.cuda.synchronize()
    fn, resolved = _compile_flex(
        args,
        {**base, cute_flash.FLASH_SOFTMAX_REGION_KEY: "shared"},
        expect_in_code="flash_stage = cute.arch.make_warp_uniform(warp_idx // 4)",
    )
    assert resolved.softmax_region == "shared"
    out, lse = fn(*args)
    torch.cuda.synchronize()
    assert torch.equal(out, out_ref)
    assert torch.equal(lse, lse_ref)


def _score_table_mod(table: torch.Tensor) -> Callable[..., Any]:
    """A per-(row, column) score bias ``table[q, kv]`` (a row-strided table
    load in the score hook: a coalescible run, so the hook runs on every
    tile and stages through the per-warp cooperative ring)."""

    def score_mod(score: Any, b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return score + hl.load(table, [q_idx, kv_idx]).to(torch.float32)

    return score_mod


def _keep_table_masks(
    keep: torch.Tensor,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """A causal mask gated by a per-(row, column) table ``keep[q, kv]`` (a
    row-strided table load in the mask hook: the coalescible run the
    staging ring serves, and the prefetch's per-chunk table load): the
    Helion form for the kernel and the eager form for the BlockMask."""

    def helion_mask(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return (q_idx >= kv_idx) & (hl.load(keep, [q_idx, kv_idx]) != 0)

    def torch_mask(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return (q_idx >= kv_idx) & (keep[q_idx, kv_idx] != 0)

    return helion_mask, torch_mask


_SHARED_COVERAGE: dict[str, dict[str, object]] = {
    # The shapes and bodies the shared region must match per stage on.
    "ragged": {"seq": 1920},
    "fp16": {"dtype": torch.float16},
    "gqa": {"kv_heads": 2},
    "rep32_store": {
        "overrides": {
            cute_flash.FLASH_S_LOAD_REP_KEY: 32,
            cute_flash.FLASH_P_STORE_REP_KEY: 32,
            cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: False,
        }
    },
    "mixed_store": {
        "overrides": {
            cute_flash.FLASH_P_STORE_REP_KEY: 32,
            cute_flash.FLASH_SPLIT_P_ARRIVE_KEY: True,
        }
    },
    "resident": {
        "overrides": {**RESIDENT_REP32, cute_flash.FLASH_S_RESIDENT_KEY: True}
    },
    "coalesced_mask": {
        "table_mask": True,
        "overrides": {
            cute_flash.FLASH_FLEX_ROLLED_CHUNKS_KEY: False,
            cute_flash.FLASH_FLEX_COALESCE_KEY: 64,
        },
        "expect": ("flex_coalesce", 64),
    },
    "prefetched_mask": {
        "table_mask": True,
        "overrides": {
            cute_flash.FLASH_FLEX_ROLLED_CHUNKS_KEY: False,
            cute_flash.FLASH_FLEX_PREFETCH_KEY: 32,
        },
        "expect": ("flex_prefetch", 32),
    },
    "coalesced_score": {
        "score_table": True,
        "overrides": {
            cute_flash.FLASH_FLEX_ROLLED_CHUNKS_KEY: False,
            cute_flash.FLASH_FLEX_COALESCE_KEY: 64,
        },
        "expect": ("flex_coalesce", 64),
    },
}


@skipIfNotCUDA()
@onlyBackends(["cute"])
@_MULTI_COMPILE_TIMEOUT
@pytest.mark.parametrize("case", sorted(_SHARED_COVERAGE), ids=sorted(_SHARED_COVERAGE))
def test_shared_softmax_region_coverage_is_bitwise_the_per_stage_regions(
    case: str,
) -> None:
    """The shared softmax region matches the per-stage regions bitwise on a
    ragged query length (the live-rows guard), fp16, grouped-query heads,
    the Rep32 and the mixed Rep32/Rep16 P stores, the resident PASS2, and
    the table-gated mask whose hook stages its rows through the per-warp
    cooperative ring (``cute_flash_flex_coalesce``: the ring is addressed by
    the runtime stage) or prefetches them (``cute_flash_flex_prefetch``),
    and a score region (a coalesced score table: the hook on every tile)."""
    spec = _SHARED_COVERAGE[case]
    batch, heads = 2, 4
    seq = int(spec.get("seq", 2048))
    kv_heads = spec.get("kv_heads")
    q, k, v = _qkv(
        batch,
        heads,
        seq,
        seq,
        kv_heads=kv_heads,  # pyrefly: ignore[bad-argument-type]
        dtype=spec.get("dtype", torch.bfloat16),  # pyrefly: ignore[bad-argument-type]
    )
    if spec.get("table_mask"):
        torch.manual_seed(1)
        keep = (torch.rand(seq, seq, device=DEVICE) < 0.7).to(torch.int32)
        helion_mask, torch_mask = _keep_table_masks(keep)
    else:
        helion_mask = torch_mask = _band
    score_mod: Callable[..., Any] = _identity
    if spec.get("score_table"):
        torch.manual_seed(2)
        score_mod = _score_table_mod(
            torch.randn(seq, seq, dtype=torch.bfloat16, device=DEVICE)
        )
    block_mask = create_block_mask(
        torch_mask, batch, heads, seq, seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    args = _flex_args(
        q,
        k,
        v,
        score_mod,
        block_mask,
        mask_mod=helion_mask,
        enable_gqa=kv_heads is not None,
    )
    base = {
        cute_flash.FLASH_KV_SPLIT_KEY: 1,
        cute_flash.FLASH_E2E_OFFSET0_KEY: 3,
        cute_flash.FLASH_E2E_OFFSET_KEY: 3,
        **spec.get("overrides", {}),  # pyrefly: ignore[bad-argument-type]
    }
    fn, resolved = _compile_flex(args, base)
    assert resolved.softmax_region == "per_stage"
    expect = spec.get("expect")
    if expect is not None:
        name, value = expect  # pyrefly: ignore[bad-assignment]
        assert getattr(resolved, name) == value, (name, getattr(resolved, name))
    out_ref, lse_ref = fn(*args)
    torch.cuda.synchronize()
    fn, resolved = _compile_flex(
        args,
        {**base, cute_flash.FLASH_SOFTMAX_REGION_KEY: "shared"},
        expect_in_code="flash_stage = cute.arch.make_warp_uniform(warp_idx // 4)",
    )
    assert resolved.softmax_region == "shared"
    if expect is not None:
        name, value = expect  # pyrefly: ignore[bad-assignment]
        assert getattr(resolved, name) == value
    if case == "coalesced_mask":
        assert resolved.flex_coalesce_slots  # the staging ring is in use
    out, lse = fn(*args)
    torch.cuda.synchronize()
    assert torch.equal(out, out_ref)
    assert torch.equal(lse, lse_ref)


def _window(width: int) -> Callable[..., Any]:
    def window(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return (q_idx >= kv_idx) & (q_idx - kv_idx <= width)

    return window


def _documents_320(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
    # 320-token documents (not a multiple of the tile): every boundary tile
    # is partial with whole-warp kept rows on one side, but the keep-word
    # peephole has no lane word for the ``// 320`` equality (element form).
    return (q_idx // 320) == (kv_idx // 320)


# (name: batch, heads, kv_heads, q_seq, kv_seq, mask, config overrides,
# dtype, whether the knob renders the plain hook: a table gather or a
# predicate the keep-word peephole does not fold (``// 320``), or no mask).
_FAST_FULL_CASES: dict[str, tuple[Any, ...]] = {
    "ragged_causal_2000": (2, 2, 2, 2000, 2000, _causal, {}, torch.bfloat16, False),
    "ragged_causal_4000": (1, 2, 2, 4000, 4000, _causal, {}, torch.bfloat16, False),
    "ragged_causal_3001": (1, 2, 2, 3001, 3001, _causal, {}, torch.bfloat16, False),
    "ragged_causal_fp16": (2, 2, 2, 2000, 2000, _causal, {}, torch.float16, False),
    "ragged_causal_resident": (
        2,
        2,
        2,
        2000,
        2000,
        _causal,
        {cute_flash.FLASH_S_RESIDENT_KEY: True, cute_flash.FLASH_S_LOAD_REP_KEY: 32},
        torch.bfloat16,
        False,
    ),
    "ragged_causal_shared_region": (
        2,
        2,
        2,
        2000,
        2000,
        _causal,
        {cute_flash.FLASH_SOFTMAX_REGION_KEY: "shared"},
        torch.bfloat16,
        False,
    ),
    "causal_diagonal": (2, 2, 2, 2048, 2048, _causal, {}, torch.bfloat16, False),
    "window_256": (2, 2, 2, 2048, 2048, _window(256), {}, torch.bfloat16, False),
    "window_512": (2, 2, 2, 2048, 2048, _window(512), {}, torch.bfloat16, False),
    "window_1024": (2, 2, 2, 2048, 2048, _window(1024), {}, torch.bfloat16, False),
    "band": (2, 4, 4, 2048, 2048, _band, {}, torch.bfloat16, False),
    "band_gqa_persistent": (
        2,
        8,
        2,
        2048,
        2048,
        _band,
        {cute_flash.FLASH_PERSISTENT_KEY: True},
        torch.bfloat16,
        False,
    ),
    "documents_320": (2, 2, 2, 2048, 2048, _documents_320, {}, torch.bfloat16, True),
    "documents_table": (2, 2, 2, 2048, 2048, None, {}, torch.bfloat16, True),
    "whisper_noop_ragged": (
        2,
        2,
        2,
        448,
        1500,
        noop_mask,
        {},
        torch.bfloat16,
        True,
    ),
}


@skipIfNotCUDA()
@onlyBackends(["cute"])
@_MULTI_COMPILE_TIMEOUT
@pytest.mark.parametrize("case", list(_FAST_FULL_CASES))
def test_flex_hook_fast_full_is_bitwise_the_plain_hook(case: str) -> None:
    """``cute_flash_flex_hook_fast_full``: a warp whose rows keep every
    column of a partial tile takes the step's dense path, which is the same
    values in the same places: the same output and LSE as the plain hook on
    ragged causal lengths (every tile of the ragged query row is a partial
    tile for torch; two-pass, resident and shared-region bodies, fp16), the
    causal diagonal, sliding windows of three widths, a band, a GQA band on
    the persistent walk.  Document masks (a table gather, or ``// 320``
    which the keep-word peephole does not fold) and a mask-less ragged cross
    attention (Whisper's 448 x 1500) keep the plain hook: the knob renders
    the same code."""
    batch, heads, kv_heads, q_seq, kv_seq, mask, base, dtype, plain = _FAST_FULL_CASES[
        case
    ]
    mask_arg: Callable[..., Any] | None = None
    if mask is None:
        doc = (torch.arange(q_seq, device=DEVICE) // 320).to(torch.int32)

        def documents_table(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
            return hl.load(doc, [q_idx]) == hl.load(doc, [kv_idx])

        def documents_table_ref(b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
            return doc[q_idx] == doc[kv_idx]

        mask, mask_arg = documents_table_ref, documents_table
    q, k, v = _qkv(batch, heads, q_seq, kv_seq, kv_heads=kv_heads, dtype=dtype)
    block_mask = create_block_mask(
        mask, batch, heads, q_seq, kv_seq, device=DEVICE, BLOCK_SIZE=(128, 128)
    )
    args = _flex_args(
        q, k, v, _identity, block_mask, mask_mod=mask_arg, enable_gqa=kv_heads != heads
    )
    base = {**base, cute_flash.FLASH_KV_SPLIT_KEY: 1}
    plain_code: list[str] = []
    fn, resolved = _compile_flex(args, base, code_sink=plain_code)
    assert resolved.flex_hook_fast_full is False
    assert resolved.s_resident is (case == "ragged_causal_resident")
    out_ref, lse_ref = fn(*args)
    torch.cuda.synchronize()
    fast_code: list[str] = []
    fn, resolved = _compile_flex(
        args,
        {**base, cute_flash.FLASH_FLEX_HOOK_FAST_FULL_KEY: True},
        code_sink=fast_code,
    )
    assert resolved.flex_hook_fast_full is True
    vote = "flash_flex_needs_mask = cutlass.Boolean(cute.arch.vote_any_sync(flash_flex_masked))"
    if plain:
        assert fast_code == plain_code
        return
    assert vote in fast_code[0]
    out, lse = fn(*args)
    torch.cuda.synchronize()
    assert torch.equal(out, out_ref)
    assert torch.equal(lse, lse_ref)
