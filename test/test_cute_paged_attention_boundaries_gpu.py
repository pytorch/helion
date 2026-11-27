from __future__ import annotations

from typing import cast

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing.cuda_graph import cuda_graph_for_replay
from helion._testing.cute_capture_replay import assert_identical
from helion._testing.cute_capture_replay import capture_launches
from helion._testing.cute_capture_replay import compile_multiwave_defaults
from helion._testing.cute_capture_replay import fill_nan
from helion._testing.cute_capture_replay import multiwave_attention_args
from helion._testing.cute_capture_replay import multiwave_lengths
from helion._testing.cute_folded_attention import folded_attention_args
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_optional_lse import optional_lse_fixture
from helion._testing.cute_optional_lse import optional_lse_reference
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_pair import paired_bounds_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args
from helion._testing.cute_paged_base2 import base2_attention_fixture
from helion._testing.cute_paged_key_clip import check_key_clip_replay
from helion._testing.cute_paged_key_clip import expected_key_shift
from helion._testing.cute_paged_key_clip import poison_unloaded_keys


@skipIfNotCUDA()
def test_empty_int64_attention_interval_precedes_subtraction() -> None:
    """Both original and replay intervals are empty; assert the native route.

    The sibling attention oracle covers nonempty intervals. This regression
    specifically proves the extreme ordered-comparison guard before subtract.
    """
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("the initial paged pipeline uses SM100 tensor memory")
    args = list(paged_attention_args(device=DEVICE))
    for value in args[:7]:
        assert isinstance(value, torch.Tensor)
        value.zero_()
    bounds = torch.tensor([[1, -(2**63)], [2**63 - 1, -1]], device=DEVICE)
    args.append(bounds)
    bound = paged_online_attention.bind(tuple(args))
    config = helion.Config(
        block_sizes=[64],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps="attention",
        cute_min_blocks_per_mp=1,
    )
    with bound.env:
        config = bound.config_spec.normalized_config(config)
    assert config.get("cute_mma_dag_pass_warps") == "attention"
    assert "_helion_paged_attention.paged_attention_body(" in bound.to_code(config)
    compiled = bound.compile_config(config)
    outputs = compiled(*args)
    with cuda_graph_for_replay() as graph:
        with torch.cuda.graph(graph):
            replay = compiled(*args)
        for changed in (False, True):
            if changed:
                bounds.copy_(torch.tensor([[0, 0], [5, 2]], device=DEVICE))
                outputs = compiled(*args)
            graph.replay()
            for output, lse in (outputs, replay):
                assert torch.count_nonzero(output).item() == 0
                assert torch.isneginf(lse).all().item()


@skipIfNotCUDA()
@pytest.mark.parametrize(
    "form,reference,base2,reverse,window,query_len,options,key_clip,lengths",
    [
        # The default clip under the guarded PDL chain's loader-tail release:
        # a window that starts inside the first page, a one-page range with
        # both ends hidden, an empty range and the last table page.
        (
            "mapped",
            True,
            True,
            False,
            31,
            4,
            {
                "cute_mma_dag_v_lag": 2,
                "cute_mma_dag_probability_schedule": "early",
                "cute_mma_dag_query_tma": True,
                "cute_pdl": True,
                "cute_pdl_capture_overlap": True,
                "cute_pdl_release": "loader_tail",
            },
            "leading",
            ((0, 100), (40, 300), (129, 256), (257, 2175), (164, 5)),
        ),
        # Reflected pages, the 40-column panel and denominators in TMEM: both
        # ends of one page, a last page holding one key, page-aligned lengths.
        # No dependent-launch option is named (cute_pdl None drops the
        # fixture's), so this is the paired default: the window's first page,
        # the leading clip's tile, loads last, after the loader's tail release
        # and with its V tiles in the drain.
        (
            "mapped",
            True,
            False,
            True,
            300,
            5,
            {
                "cute_mma_dag_v_lag": 3,
                "cute_mma_dag_probability_schedule": "early_tmem",
                "cute_pdl": None,
            },
            "both",
            ((0, 700), (1, 513), (128, 389), (2176, 1000), (333, 2049)),
        ),
        # Causal-only attention through the ordinary per-key softmax: only the
        # last page has hidden keys.
        (
            "per_key",
            False,
            False,
            False,
            None,
            4,
            {"cute_mma_dag_v_lag": 1},
            "trailing",
            ((0, 1), (129, 300), (128, 2049), (77, 640), (2176, 513)),
        ),
        # The per-key softmax with a window and the eight-column panel, under
        # sleeping waits and the final join: the other replays retire tensor
        # memory early, the paired schedule's default.
        (
            "mapped",
            False,
            False,
            False,
            127,
            1,
            {
                "cute_mma_dag_sleep_waits": True,
                "cute_mma_dag_early_tmem_retirement": False,
            },
            "both",
            ((0, 200), (128, 129), (1, 500), (333, 2176), (64, 1025)),
        ),
    ],
)
def test_key_clip_never_reads_keys_no_row_sees(
    form: str,
    reference: bool,
    base2: bool,
    reverse: bool,
    window: int | None,
    query_len: int,
    options: dict[str, object],
    key_clip: str,
    lengths: tuple[tuple[int, int], ...],
) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention uses SM100 tensor memory")
    if base2:
        kernel, initial = base2_attention_fixture(form, query_len, True, DEVICE)
        initial = initial[:-1]
    else:
        kernel, initial = optional_lse_fixture(
            form, True, DEVICE, query_len=query_len, window=window
        )
    values = list(initial)
    if form == "mapped":
        values[7] = window
        larger = folded_attention_args(
            query_len, device=DEVICE, pages=40, table_pages=17
        )
    else:
        larger = paged_attention_args(rows=query_len * 8, device=DEVICE, pages=40)
    # Thirty-four logical pages need their own physical pages.
    values[1:6] = larger[1:6]
    args = (*values, reference, reverse)
    bound = helion.kernel(
        kernel.fn,
        backend="cute",
        static_shapes=True,
        fast_math=True,
        index_dtype=torch.int64 if reverse else torch.int32,
    ).bind(args)
    config = optional_lse_config()
    config.config.update(
        block_sizes=[helion.next_power_of_2(query_len * 8)],
        cute_mma_dag_pass_warps="attention_paired",
        cute_mma_dag_key_clip=key_clip,
        **options,
    )
    if config.config["cute_pdl"] is None:
        # Leave the dependent launch to the proved paired kernel's default.
        del config.config["cute_pdl"]
    source = bound.to_code(config)
    assert "paged_attention_paired_body(" in source
    assert f"key_clip={key_clip!r}" in source
    assert ("capture_release_tail=True" in source) == ("cute_pdl" in options)
    assert ("early_tmem_retirement=True" in source) == (
        options.get("cute_mma_dag_early_tmem_retirement", True)
    )
    assert ("reflected_dtype=" in source) == reverse
    leading, trailing = check_key_clip_replay(
        bound,
        config,
        args,
        form,
        lengths,
        key_clip=key_clip,
        reference_scores=reference,
        reverse_pages=reverse,
        base2_scores=base2,
    )
    # Each side the clip names left keys out in some replay.
    assert (leading > 0) == (key_clip != "trailing")
    assert (trailing > 0) == (key_clip != "leading")


@skipIfNotCUDA()
def test_key_clip_leaves_out_tiles_no_row_sees() -> None:
    """Page ranges wider than the visible keys, under the default policies.

    A kernel that reads its page range from ``bounds`` can be handed pages
    before every row's window and after every row's causal bound. Such a tile
    takes the largest shift on its side (124 keys, four masked rows stay
    loaded), which the ranges derived from a length never produce. The config
    names no dependent-launch option, so the replay also runs the proved
    paired kernel's guarded overlap, whose loader-tail release counts pages:
    in the first replay the last four pages of both requests are such tiles.
    """
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention uses SM100 tensor memory")
    query_len, group, window, table_pages = 4, 8, 31, 17
    values = list(paged_attention_args(rows=query_len * group, device=DEVICE, pages=40))
    values[7], values[9] = query_len, window
    bounds = torch.zeros((2, 2), dtype=torch.int64, device=DEVICE)
    args = (*values, bounds, True, True, True)
    bound = helion.kernel(
        paired_bounds_attention("request").fn,
        backend="cute",
        static_shapes=True,
        fast_math=True,
    ).bind(args)
    config = optional_lse_config()
    del config.config["cute_pdl"]
    config.config.update(
        block_sizes=[query_len * group],
        cute_mma_dag_pass_warps="attention_paired",
        cute_mma_dag_key_clip="both",
        cute_mma_dag_v_lag=2,
        cute_mma_dag_probability_schedule="early",
    )
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    assert normalized.get("cute_pdl") is True
    assert normalized.get("cute_pdl_capture_overlap") is True
    assert normalized.get("cute_pdl_release") == "loader_tail"
    source = bound.to_code(config)
    assert "paged_attention_paired_body(" in source
    assert "key_clip='both'" in source
    assert "capture_release_tail=True" in source
    # Each range contains the pages its length makes visible, so the eager
    # reference over those pages is the result: every table page; one page
    # more on both sides or after the end; an empty request whose pages all
    # follow its causal bound; and two-page ranges, which release after the
    # loader's loop, starting or ending with such a tile.
    lengths = ((700, 5), (1300, 129), (0, 2176), (300, 1000))
    page_ranges = (
        ((0, table_pages), (0, table_pages)),
        ((8, 12), (0, 3)),
        ((0, 6), (15, table_pages)),
        ((1, 3), (7, 9)),
    )
    shifts = {
        expected_key_shift(
            tile * 128,
            length,
            query_len=query_len,
            group=group,
            query_columns=query_len * group,
            window=window,
            key_clip="both",
        )
        for request_lengths, ranges in zip(lengths, page_ranges, strict=True)
        for length, (first, last) in zip(request_lengths, ranges, strict=True)
        for tile in range(first, last)
    }
    # Whole tiles on each side, and partly hidden ones between them.
    assert {124, -124} < shifts
    leading, trailing = check_key_clip_replay(
        bound,
        config,
        args,
        "per_key",
        lengths,
        key_clip="both",
        reference_scores=True,
        page_ranges=page_ranges,
    )
    assert leading > 0 and trailing > 0


@skipIfNotCUDA()
def test_key_clip_under_the_default_policy_on_a_multiwave_grid() -> None:
    """A windowed fast_math kernel's defaults on more paired CTAs than SMs.

    The config names neither a clip nor a dependent-launch option. The kernel
    therefore clips leading keys and overlaps back-to-back captured calls,
    and past one wave the release moves from each loader's tail to the entry
    of every CTA. With nonfinite K and V in every key the clip
    leaves out, each request must equal the joined kernel without dependent
    launch exactly, eager and captured; requests of both waves are also held
    against the eager reference.
    """
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention uses SM100 tensor memory")
    # One physical page per logical page: poison never reaches a visible key.
    values = multiwave_attention_args(DEVICE, distinct_pages=True)
    # Eight rows are one query position's grouped heads.
    values[7], values[9] = 1, 300
    q, k, v, ks, vs, pages, lengths = (cast("torch.Tensor", x) for x in values[:7])
    requests = lengths.numel()
    assert pages.unique().numel() == pages.numel()
    args = (*values, None, True, True, True)
    bound = helion.kernel(
        paged_online_attention.fn, backend="cute", static_shapes=True, fast_math=True
    ).bind(args)
    # The eval kernel's schedule, and neither a clip nor a dependent-launch
    # option: the helper checks the default policy and its release at the
    # entry of every CTA.
    normalized, source, tracked, plain = compile_multiwave_defaults(
        bound, args, cute_mma_dag_probability_schedule="early"
    )
    assert normalized.get("cute_mma_dag_key_clip") == "leading"
    assert "key_clip='leading'" in source
    clean = [tensor.clone() for tensor in (k, v, ks, vs)]
    # Both ends of the grid, two heads per request: CTAs of the first wave
    # and CTAs of the last.
    sample = torch.tensor([1, 2, requests - 2, requests - 1], device=DEVICE)
    items = (sample[:, None] * 2 + torch.arange(2, device=DEVICE)).reshape(-1)
    left_out = 0
    with cuda_graph_for_replay() as graph:
        captured = capture_launches(graph, tracked, args, 3)
        for replay in range(3):
            for tensor, original in zip((k, v, ks, vs), clean, strict=True):
                tensor.copy_(original)
            lengths.copy_(multiwave_lengths(requests, replay, DEVICE))
            expected = plain(*args)
            reference = optional_lse_reference(
                (q[items], k, v, ks, vs, pages[sample], lengths[sample], *args[7:10]),
                "per_key",
                reference_scores=True,
            )
            for value, sampled in zip(expected, reference, strict=True):
                torch.testing.assert_close(
                    value[items], sampled, rtol=0.002, atol=0.002
                )
            leading, trailing = poison_unloaded_keys(
                args, "per_key", key_clip="leading"
            )
            assert trailing == 0
            left_out += leading
            assert_identical(tracked(*args), expected)
            fill_nan(captured)
            graph.replay()
            for output in captured:
                assert_identical(output, expected)
    assert left_out > 0
