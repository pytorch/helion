"""Native correctness of computed FP16/BF16 dots and their FP32 carries."""

from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import matchesBackends
from helion._testing.cute_gathered_bf16 import routed_transposed_product
from helion._testing.cute_gathered_bf16 import split_paged_attention
from helion._testing.cute_half_mma import batched_dot
from helion._testing.cute_half_mma import cast_dot
from helion._testing.cute_half_mma import computed_dot
from helion._testing.cute_half_mma import recurrent_dot

if matchesBackends(["cute"]):
    pytest.importorskip("cutlass.cute")
    from helion._testing.cute_weighted_merge import merge_one

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(32, 32, 32), (24, 19, 11)])
def test_half_mma_matches_fp32_accumulation(
    dtype: torch.dtype, shape: tuple[int, int, int]
) -> None:
    torch.manual_seed(42)
    m, k, n = shape
    x = torch.randn(2, m, k, device=DEVICE, dtype=dtype) * 0.2
    y = torch.randn(2, k, n, device=DEVICE, dtype=dtype) * 0.2
    acc = torch.randn(2, m, n, device=DEVICE, dtype=torch.float32)
    bound = computed_dot.bind((x, y, acc))
    config = helion.Config.from_dict(
        dict(bound.config_spec.default_config()) | {"cute_fragment_half_mma": True}
    )
    compiled = bound.compile_config(config)
    actual = compiled(x, y, acc)
    expected = torch.bmm(x.float() * 0.5, y.float()) + acc
    expected = expected + expected.amax(dim=1, keepdim=True)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


def test_half_mma_keeps_a_computed_loop_accumulator() -> None:
    torch.manual_seed(43)
    x = torch.randn(2, 3, 32, 32, device=DEVICE, dtype=torch.bfloat16) * 0.2
    y = torch.randn(2, 3, 32, 8, device=DEVICE, dtype=torch.bfloat16) * 0.2
    bound = recurrent_dot.bind((x, y))
    config = helion.Config.from_dict(
        dict(bound.config_spec.default_config()) | {"cute_fragment_half_mma": True}
    )
    actual = bound.compile_config(config)(x, y)
    expected = torch.zeros((2, 32, 8), device=DEVICE)
    for step in range(3):
        expected = expected * 0.5 + torch.bmm(
            x[:, step].float() * 0.5, y[:, step].float()
        )
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_mma_predicates_an_odd_native_contraction(dtype: torch.dtype) -> None:
    torch.manual_seed(44)
    x = torch.randn(2, 19, 19, device=DEVICE, dtype=dtype) * 0.2
    y = torch.randn(2, 19, 11, device=DEVICE, dtype=dtype) * 0.2
    bound = batched_dot.bind((x, y))
    config = helion.Config.from_dict(
        dict(bound.config_spec.default_config()) | {"cute_fragment_half_mma": True}
    )
    actual = bound.compile_config(config)(x, y)
    torch.testing.assert_close(
        actual, torch.bmm(x.float(), y.float()), rtol=1e-4, atol=1e-4
    )


def test_half_mma_partial_in_place_cast_matches_reference() -> None:
    torch.manual_seed(45)
    x = torch.randn(2, 8, 8, device=DEVICE, dtype=torch.bfloat16) * 0.2
    y = torch.randn(2, 8, 8, device=DEVICE, dtype=torch.float16) * 0.2
    bound = cast_dot.bind((x, y))
    config = helion.Config.from_dict(
        dict(bound.config_spec.default_config()) | {"cute_fragment_half_mma": True}
    )
    actual = bound.compile_config(config)(x, y)
    torch.testing.assert_close(
        actual, torch.bmm(x.half().float(), y.float()), rtol=1e-4, atol=1e-4
    )


def test_gathered_nt_signed_pages_and_repeated_routes() -> None:
    # The second rank-2 operand window is a signed physical page; invalid
    # entries must be zero-filled even when other slots use the same page.
    torch.manual_seed(45)
    # 65 native K steps cross the runtime-loop threshold.
    x = torch.randn((32, 1040), device=DEVICE, dtype=torch.bfloat16)
    bank = torch.randn((5, 32, 1040), device=DEVICE, dtype=torch.bfloat16)
    order = torch.tensor([3, -1, -5, -6, 5, 3, 0, 4], device=DEVICE, dtype=torch.int64)
    args = (x, bank, order, 0, 0, False)
    bound = routed_transposed_product.bind(args)
    config = helion.Config(
        block_sizes=bound.config_spec.default_config().block_sizes,
        cute_mma_dag="off",
        cute_fragment_half_mma=True,
    )
    assert "gathered_mma_page" in bound.to_code(config)
    got = bound.compile_config(config)(*args)
    key = bank[order.remainder(bank.size(0))].float()
    reference = torch.einsum("mk,bnk->bmn", x.float(), key)
    reference[(order < -bank.size(0)) | (order >= bank.size(0))] = 0
    torch.testing.assert_close(got, reference, rtol=3e-4, atol=5e-5)


def test_fused_gathered_page_keeps_causal_row_limits_and_online_carry() -> None:
    torch.manual_seed(46)
    query = torch.randn((1, 2, 8, 128), device=DEVICE, dtype=torch.bfloat16)
    bank = torch.randn((5, 1, 32, 128), device=DEVICE, dtype=torch.bfloat16)
    # A distinct final page has one live key; its high score makes accidentally
    # including/excluding that key in either query row easy to detect.
    query[0, 0, 0] = query[0, 1, 0]
    bank[4, 0, 0] = query[0, 1, 0] * 2
    pages = torch.tensor(
        [([2, -5, 3, 1] * 16) + [-1]], device=DEVICE, dtype=torch.int32
    )
    length = 2049  # Two 32-key pages per live split.
    seq = torch.tensor([length], device=DEVICE, dtype=torch.int32)
    scale = 128**-0.5
    args = (query, bank, pages, seq, scale)
    op = helion.kernel(backend="cute", static_shapes=False)(split_paged_attention)
    bound = op.bind(args)
    config = next(
        candidate
        for candidate in bound.config_spec.compiler_seed_configs
        if candidate.config.get("cute_fragment_half_mma")
    )
    assert ".gathered_softmax_page(" in bound.to_code(config)
    actual = bound.compile_config(config)(*args)
    # Independently flatten the permuted physical pages and normalize across
    # every visible key. This oracle does not split the context into pages.
    ordered = bank[pages[0].long(), 0].reshape(-1, 128)[:length].float()
    scores = torch.einsum("qhd,ld->qhl", query[0].float(), ordered) * scale
    limits = torch.tensor([length - 1, length], device=DEVICE)
    visible = torch.arange(length, device=DEVICE)[None, :] < limits[:, None]
    weights = scores.masked_fill(~visible[:, None, :], -torch.inf).softmax(-1)
    expected = torch.einsum("qhl,ld->qhd", weights, ordered[:, :64]).bfloat16()[None]
    assert (expected[0, 0, 0] - expected[0, 1, 0]).abs().max() > 1
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.008)


@pytest.mark.parametrize("pipelined", (False, True), ids=("direct", "async"))
def test_streaming_split_merge_shares_both_halves_of_the_weight_vector(
    pipelined: bool,
) -> None:
    torch.manual_seed(62)
    query = torch.randn((1, 1, 16, 192), device=DEVICE, dtype=torch.bfloat16)
    page = 64 if pipelined else 32
    keys = torch.randn((64, 1, page, 192), device=DEVICE, dtype=torch.bfloat16)
    pages = torch.randperm(64, device=DEVICE, dtype=torch.int32)[None, :]
    seq = torch.tensor([page], device=DEVICE, dtype=torch.int32)
    scale = 192**-0.5
    args = (query, keys, pages, seq, scale)
    op = helion.kernel(backend="cute", static_shapes=False)(split_paged_attention)
    bound = op.bind(args)
    assert not any(
        c.config.get("cute_gathered_page_stages") == 8
        for c in bound.config_spec.compiler_seed_configs
    )
    candidate = next(
        c
        for c in bound.config_spec.compiler_seed_configs
        if c.config.get("cute_fragment_half_mma")
        and c.pid_type == "persistent_interleaved"
        and (
            c.config.get("cute_gathered_page_stages") == 4
            if pipelined
            else not c.config.get("cute_gathered_page_pipeline")
        )
    )
    assert ".split_weighted_merge(" in bound.to_code(candidate)
    compiled = bound.compile_config(candidate)
    visible = keys[pages[0].long(), 0].reshape(-1, 192).float()
    # Zero, one occupied half-warp, both halves and every one of the 64 slots
    # run a single compilation and the same persistent kernel/allocations.
    for length in (0, page + 3, 32 * page + 1, 64 * page):
        seq.fill_(length)
        actual = compiled(*args).float()
        if length == 0:
            # The public program finishes with 0/0 when there are no splits.
            assert torch.isnan(actual).all()
            continue
        scores = query[0, 0].float() @ visible[:length].T * scale
        reference = scores.softmax(-1) @ visible[:length, :128]
        torch.testing.assert_close(actual[0, 0], reference, rtol=0.02, atol=0.003)


@pytest.mark.parametrize(
    ("stages", "fast_math"),
    [(2, False), (4, False), (8, False), (8, True)],
    ids=("2", "4", "8", "8-fast-math"),
)
def test_async_panel_ring_refills_and_reuses_slots(
    stages: int, fast_math: bool
) -> None:
    # Nine QK panels require at least one slot refill at every selected
    # depth. Two distinct pages also exercise a partial last page.
    torch.manual_seed(91)
    query = torch.randn((1, 1, 16, 576), device=DEVICE, dtype=torch.bfloat16)
    keys = torch.randn((2, 1, 64, 576), device=DEVICE, dtype=torch.bfloat16)
    pages = torch.tensor([[-1, -2]], device=DEVICE, dtype=torch.int32)
    length = torch.tensor([99], device=DEVICE, dtype=torch.int32)
    scale = 576**-0.5
    args = (query, keys, pages, length, scale)
    bound = helion.kernel(backend="cute", static_shapes=False, fast_math=fast_math)(
        split_paged_attention
    ).bind(args)
    cfg = next(
        c
        for c in bound.config_spec.compiler_seed_configs
        if c.config.get("cute_gathered_page_pipeline")
        and c.config.get("cute_gathered_page_stages", 2) == stages
    )
    result = bound.compile_config(cfg)(*args)[0, 0].float()
    ordered = keys[pages[0].long(), 0].reshape(-1, 576)[:99].float()
    expected = (query[0, 0].float() @ ordered.T * scale).softmax(-1) @ ordered[:, :512]
    torch.testing.assert_close(result, expected, rtol=0.02, atol=0.003)


def test_split_weighted_merge_respects_fast_math_for_subnormals() -> None:
    scores = torch.zeros((64, 1), device=DEVICE, dtype=torch.float32)
    partials = torch.zeros((64, 1, 128), device=DEVICE, dtype=torch.float32)
    partials[0].fill_(2.0**-130)
    out = torch.empty((1, 128), device=DEVICE, dtype=torch.bfloat16)
    for fast in (False, True):
        out.fill_(float("nan"))
        merge_one(scores, partials, out, 1, fast)
        # exp2(0) = 1. The single input has BF16 bits 0x0008. Approximate
        # FP32 division flushes it; the default quotient preserves it.
        torch.testing.assert_close(
            out.view(torch.int16),
            torch.full_like(out.view(torch.int16), 0 if fast else 8),
            atol=0,
            rtol=0,
        )

    # The quotient is now normal in strict math. The tiny weight times the
    # large partial isolates exp2 flushing, independently of division FTZ.
    scores[1].fill_(-130)
    partials.zero_()
    partials[1].fill_(2.0**126)
    for fast in (False, True):
        out.fill_(float("nan"))
        merge_one(scores, partials, out, 2, fast)
        torch.testing.assert_close(
            out, torch.full_like(out, 0.0 if fast else 0.0625), rtol=0, atol=0
        )
