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
