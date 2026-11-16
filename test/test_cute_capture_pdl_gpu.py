"""Actual captured dependencies, not ready-input-only overlap benchmarks."""

from __future__ import annotations

from typing import cast

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import is_cuda
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing.cute_capture_replay import MULTIWAVE_LAST_WAVE
from helion._testing.cute_capture_replay import assert_identical
from helion._testing.cute_capture_replay import capture_launches
from helion._testing.cute_capture_replay import compile_multiwave_defaults
from helion._testing.cute_capture_replay import multiwave_attention_args
from helion._testing.cute_capture_replay import multiwave_lengths
from helion._testing.cute_capture_replay import observed_wait_modes
from helion._testing.cute_capture_replay import ramp_attention_operands
from helion._testing.cute_folded_attention import folded_attention_args
from helion._testing.cute_folded_attention import folded_paged_attention
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args
from helion._testing.cute_pdl import CONTROL_DELAY
from helion._testing.cute_pdl import produce
from helion.autotuner.benchmarking import do_bench_cuda_graph
from helion.autotuner.input_banks import make_input_bank_cycle
from helion.runtime.cute.capture_pdl import WaitMode
from helion.runtime.settings import default_autotuner_fn


def _capture_args(return_lse: bool) -> list[object]:
    args = list(paged_attention_args(device=DEVICE, rows=8, requests=2))
    ramp_attention_operands(args)
    args[3].fill_(126)
    args[4].fill_(126)
    args[5].copy_(
        (torch.arange(args[5].numel(), device=DEVICE) * 7 % args[1].size(0)).reshape(
            args[5].shape
        )
    )
    args[6].copy_(torch.tensor([0, 1025], dtype=torch.int32, device=DEVICE))
    args[-1] = -1
    args.extend((None, True, return_lse))
    return args


@pytest.fixture(
    scope="module",
    params=[
        (True, "entry", 0),
        (False, "entry", 3),
        (True, "loader_tail", 0),
        # No dependent-launch key at all: the proved paired kernel's default
        # policy, which is this guarded overlap with the loader-tail release.
        (False, None, 3),
    ],
)
def kernels(request):
    if not is_cuda():
        pytest.skip("capture PDL requires CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention requires SM100 tensor memory")
    return_lse, release, lag = request.param
    args = _capture_args(return_lse)
    bound = paged_online_attention.bind(tuple(args))
    config = helion.Config(
        block_sizes=[8],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps="attention_paired",
        cute_min_blocks_per_mp=1,
        # Reuse fixtures for both wrapper ABIs and both V-lag loader branches.
        cute_mma_dag_query_tma=(return_lse == (release == "entry")),
        cute_mma_dag_v_lag=lag,
    )
    if release is not None:
        config.config.update(
            cute_pdl=True, cute_pdl_capture_overlap=True, cute_pdl_release=release
        )
    normalized = bound.config_spec.normalized_config(config)
    assert normalized.get("cute_pdl") is True
    assert normalized.get("cute_pdl_capture_overlap") is True
    assert normalized.get("cute_pdl_release", "auto") == (release or "loader_tail")
    tracked = bound.compile_config(config)
    plain = bound.compile_config(
        helion.Config.from_dict(
            {
                **config.config,
                "cute_pdl": False,
                "cute_pdl_capture_overlap": False,
                "cute_pdl_release": "auto",
            }
        )
    )
    tracked(*args)
    plain(*args)
    torch.cuda.synchronize()
    return args, tracked, plain


@pytest.mark.cpu_only
def test_capture_fixture_skips_before_querying_cuda(monkeypatch):
    monkeypatch.setitem(kernels.__wrapped__.__globals__, "is_cuda", lambda: False)

    def unavailable():
        raise AssertionError("CUDA capability must not be queried")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unavailable)
    with pytest.raises(pytest.skip.Exception, match="capture PDL requires CUDA"):
        kernels.__wrapped__(None)


def close(actual, expected):
    for value, reference in zip(actual, expected, strict=True):
        if value is None:
            assert reference is None
        else:
            torch.testing.assert_close(value, reference, rtol=0.002, atol=0.002)


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_capture_overlap_retains_non_immediate_ancestor_and_replays(kernels):
    args, tracked, plain = kernels
    graph = torch.cuda.CUDAGraph()
    with observed_wait_modes() as modes, torch.cuda.graph(graph):
        output_a = tracked(*args)
        output_b = tracked(*args)
        dependent_args = [output_a[0], *args[1:]]
        output_c = tracked(*dependent_args)
        output_d = tracked(*args)
    assert modes == [
        WaitMode.EARLY,
        WaitMode.BEFORE_WRITES,
        WaitMode.EARLY,
        WaitMode.BEFORE_WRITES,
    ], modes
    for iteration in range(2):
        args[0].add_(0.03125)
        args[2].neg_()
        args[6].copy_(
            torch.tensor(
                [129, 0] if iteration else [0, 1025], dtype=torch.int32, device=DEVICE
            )
        )
        expected_a = plain(*args)
        expected_c = plain(expected_a[0], *args[1:])
        graph.replay()
        for output in (output_a, output_b, output_d):
            close(output, expected_a)
        close(output_c, expected_c)


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_autotuner_bank_cycle_graph_overlaps_behind_each_cycle_head(kernels):
    # The graph a search times with rotating input banks: an untimed priming
    # cycle, the start event node, the timed cycle and the end event node,
    # captured once and only replayed. Every bank's call is a node of its
    # own with that bank's addresses, so the guard grants the overlap to each
    # call behind another call of its cycle. The head of the priming cycle
    # has no predecessor and the head of the timed cycle follows the event
    # node: both keep the early wait. This is the benchmark in which the
    # search ranks the dependent-launch policy.
    args, tracked, plain = kernels
    cycle = make_input_bank_cycle(tracked, args, 3)
    assert len({id(bank) for bank in cycle.banks}) == 3
    cycle.warm(retained_cycles=2)
    torch.cuda.synchronize()
    early, late = WaitMode.EARLY, WaitMode.BEFORE_WRITES
    with observed_wait_modes() as modes:
        samples = do_bench_cuda_graph(
            cycle,
            return_mode="all",
            fixed_repetitions=3,
            pre_warmed=True,
            l2_flush_mode="off",
            pre_timing=cycle,
        )
        assert modes == [early, late, late, early, late, late], modes
        assert len(samples) == 3 and all(sample > 0 for sample in samples)
        # The cold regime has no priming cycle: a cache clear, then the events.
        modes.clear()
        do_bench_cuda_graph(
            cycle,
            return_mode="all",
            fixed_repetitions=1,
            pre_warmed=True,
            l2_flush_mode="replay",
        )
        assert modes == [early, late, late], modes
        # A kernel without the contract takes no part in the proof.
        modes.clear()
        plain_cycle = make_input_bank_cycle(plain, args, 3)
        plain_cycle.warm(retained_cycles=2)
        torch.cuda.synchronize()
        do_bench_cuda_graph(
            plain_cycle,
            return_mode="all",
            fixed_repetitions=1,
            pre_warmed=True,
            l2_flush_mode="off",
            pre_timing=plain_cycle,
        )
        assert modes == []


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_capture_overlap_ranges_from_empty_to_full(kernels):
    # The loader's tail release sits four pages before the end of the range:
    # cover ranges without that step (0 to 3 pages, released after the loop),
    # the step at the first page (4 pages) and later (5 and all 17 pages),
    # each beside a request of another length in the same replay.
    args, tracked, plain = kernels
    graph = torch.cuda.CUDAGraph()
    outputs = capture_launches(graph, tracked, args, 3)
    for lengths in (
        [0, 0],
        [1, 5],
        [128, 129],
        [384, 385],
        [512, 513],
        [2176, 641],
        [3, 2176],
    ):
        args[6].copy_(torch.tensor(lengths, dtype=torch.int32, device=DEVICE))
        expected = plain(*args)
        graph.replay()
        for output in outputs:
            assert_identical(output, expected)


@pytest.fixture(scope="module")
def multiwave():
    """The default policy on a grid of more than one wave of CTAs."""
    if not is_cuda():
        pytest.skip("capture PDL requires CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention requires SM100 tensor memory")
    args = multiwave_attention_args(DEVICE)
    args[-1] = -1
    args.extend((None, True, True))
    bound = paged_online_attention.bind(tuple(args))
    # No dependent-launch key and no occupancy key: the compiler's defaults,
    # whose release moves to the entry of every CTA.
    _normalized, _source, tracked, plain = compile_multiwave_defaults(bound, args)
    return args, tracked, plain


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_default_policy_on_a_multiwave_grid_with_mixed_ranges(multiwave):
    # The paired kernel's defaults (early retirement without the final join,
    # the loader ahead of the setup rendezvous, guarded overlap) with CTAs of
    # very different lifetimes retiring and starting across two waves, eager
    # and as overlapped captured calls, against the joined kernel without
    # dependent launch.
    args, tracked, plain = multiwave
    requests = args[6].numel()
    args[6].copy_(multiwave_lengths(requests, 0, DEVICE))
    assert_identical(tracked(*args), plain(*args))
    graph = torch.cuda.CUDAGraph()
    outputs = capture_launches(graph, tracked, args, 4)
    for replay in range(4):
        if replay == 3:
            args[6].zero_()
        else:
            args[6].copy_(multiwave_lengths(requests, replay + 1, DEVICE))
        args[0].add_(0.03125)
        expected = plain(*args)
        graph.replay()
        for output in outputs:
            assert_identical(output, expected)


# The aliasing scenarios on a grid of two waves: the folded fixture, whose
# output is an argument, with one query a request. A request of the launch
# ahead walks ``ALIAS_TABLE`` pages; the launches behind it walk one.
ALIAS_TABLE = 8
# The scenario that tells the two placements apart: the first CTAs of the
# launch ahead walk a table of 64 pages (under a window that holds them) and
# every other request one page, so those CTAs outlive several launches of
# one-page requests behind them.
ALIAS_LONG_TABLE = 64
ALIAS_PAGE_BYTES = 33792
# Physical pages no aliased output covers, which the page tables cycle over.
ALIAS_POOL = 13
# Replays of a captured graph in each scenario.
ALIAS_REPLAYS = 100


def _folded_bank(requests: int, pages: int, table: int) -> list[object]:
    """The folded fixture's arguments for one query a request, in its own
    layout (padded Q and O behind a pointer offset, K and V pages with
    their scales in one storage each), for ``requests`` requests, a pool of
    ``pages`` pages and a page table of ``table`` pages a request, under a
    window of 2047 keys or, for a longer table, the keys the table holds
    less one."""
    q, k, v, ks, vs, _table, _lengths, _window, block, lse, out = folded_attention_args(
        1
    )

    def grown(tensor: object, leading: int, storage: torch.Tensor | None = None):
        like = cast("torch.Tensor", tensor)
        if storage is None:
            elements = like.storage_offset() + leading * like.stride(0)
            storage = torch.empty(elements, dtype=like.dtype, device=DEVICE)
        return storage.view(like.dtype).as_strided(
            (leading, *like.shape[1:]), like.stride(), like.storage_offset()
        )

    assert cast("torch.Tensor", k).stride(0) == ALIAS_PAGE_BYTES
    pools = [
        torch.empty(pages * ALIAS_PAGE_BYTES, dtype=torch.int8, device=DEVICE)
        for _ in range(2)
    ]
    return [
        grown(q, requests),
        grown(k, pages, pools[0]),
        grown(v, pages, pools[1]),
        grown(ks, pages, pools[0]),
        grown(vs, pages, pools[1]),
        torch.empty((requests, table), dtype=torch.int32, device=DEVICE),
        torch.empty(requests, dtype=torch.int32, device=DEVICE),
        max(2047, table * 128 - 1),
        block,
        lse,
        grown(out, requests),
    ]


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """The tensor's elements as integers, so NaNs compare by their bits."""
    integer = torch.int16 if tensor.dtype == torch.bfloat16 else torch.int32
    return tensor.contiguous().view(integer).clone()


class _AliasedBanks:
    """Three sets of arguments of one bound kernel, ``first`` for the launch
    ahead and ``second`` and ``third`` for launches behind it, whose outputs
    are views of the V pool of ``first``.

    The pages an output covers are the ones the requests of the last wave
    of ``first`` read last: the top slots of their page tables, in the order
    the loader walks them. Those CTAs enter when the first wave's leave, so
    with the release at entry in every CTA the launches behind start while
    the last wave still has these pages ahead of it. A launch that stored
    its output ahead of its dependency wait would change them under it.

    ``third`` also reads what ``first`` writes: its Q is a view of the
    buffer the output of ``first`` lies in, an input that a launch two
    ahead of it stores.

    The last wave is resident when a launch behind starts under either
    placement of the release. With ``first_ctas`` the pages are instead the
    ones the first ``MULTIWAVE_LAST_WAVE`` requests of ``first`` read last,
    those requests walk a table of ``ALIAS_LONG_TABLE`` pages and every
    other request of ``first`` one page. Those are the CTAs the last wave's
    guard waits for: under it a launch behind starts when they have exited,
    and with the release at entry in every CTA it starts as soon as that
    many of the short CTAs have, while the long ones are still walking.
    Only that placement lets a launch behind run beside them.
    """

    def __init__(self, requests: int, *, first_ctas: bool = False) -> None:
        self.requests = requests
        slots = ALIAS_LONG_TABLE if first_ctas else ALIAS_TABLE
        # The requests of ``first`` that read the aliased pages.
        self.readers = (
            slice(0, MULTIWAVE_LAST_WAVE)
            if first_ctas
            else slice(requests - MULTIWAVE_LAST_WAVE, requests)
        )
        banks = [_folded_bank(requests, ALIAS_POOL, slots) for _ in range(3)]
        out = cast("torch.Tensor", banks[0][10])
        q = cast("torch.Tensor", banks[0][0])
        # The elements an output's view spans, behind its pointer offset.
        self.out_elements = out.storage_offset() + requests * out.stride(0)
        region_pages = -(-self.out_elements * 2 // ALIAS_PAGE_BYTES)
        self.region_pages = region_pages
        banks[0] = _folded_bank(requests, ALIAS_POOL + 2 * region_pages, slots)
        generator = torch.Generator().manual_seed(requests)
        for index, bank in enumerate(banks):
            q_bank, k, v, ks, vs, table, lengths = (
                cast("torch.Tensor", value) for value in bank[:7]
            )
            for storage in (k, v):
                backing = torch.empty(0, dtype=torch.int8, device=DEVICE).set_(
                    storage.untyped_storage()
                )
                backing.copy_(
                    torch.randint(
                        -8, 9, (backing.numel(),), generator=generator, dtype=torch.int8
                    )
                )
            for scale in (ks, vs):
                scale.copy_(
                    torch.randint(
                        125, 129, scale.shape, generator=generator, dtype=torch.uint8
                    )
                )
            q_bank.copy_(torch.randn(q_bank.shape, generator=generator) / 4)
            table.copy_(
                (torch.arange(table.numel()) * 7 % ALIAS_POOL).reshape(table.shape)
            )
            # The launch ahead walks every page of its table (as far as
            # the window reaches); the ones behind it one page, and reach
            # their stores while it runs.
            window = cast("int", bank[7])
            lengths.fill_(min(slots * 128, window) if index == 0 else 100 + index)
            if index == 0 and first_ctas:
                lengths[MULTIWAVE_LAST_WAVE:] = 100
        self.first, self.second, self.third = banks
        # The output regions in the V pool of ``first``, behind the pages
        # its tables cycle over, and the slots that read them: the top
        # slots of the reading requests, newest region page first.
        v_first = cast("torch.Tensor", self.first[2])
        v_bytes = torch.empty(0, dtype=torch.int8, device=DEVICE).set_(
            v_first.untyped_storage()
        )
        table = cast("torch.Tensor", self.first[5])
        self.regions = []
        for region in range(2):
            first_page = ALIAS_POOL + region * region_pages
            for page in range(region_pages):
                slot = region * region_pages + page
                request = slot % MULTIWAVE_LAST_WAVE
                if not first_ctas:
                    request = requests - 1 - request
                table[request, slots - 1 - slot // MULTIWAVE_LAST_WAVE] = (
                    first_page + page
                )
            begin = first_page * ALIAS_PAGE_BYTES
            view = v_bytes[begin : begin + self.out_elements * 2]
            self.regions.append((view, view.clone()))
            aliased = v_bytes.view(torch.bfloat16).as_strided(
                out.shape,
                out.stride(),
                first_page * (ALIAS_PAGE_BYTES // 2) + out.storage_offset(),
            )
            assert aliased.data_ptr() % 16 == 0
            (self.second, self.third)[region][10] = aliased
        assert 2 * region_pages <= MULTIWAVE_LAST_WAVE * (slots - 1)
        # One buffer for the output of ``first`` and the Q of ``third``: a
        # request's Q lies on the output of the request with twice its index.
        chain = torch.empty(
            q.storage_offset() + requests * q.stride(0),
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        assert q.stride(0) == 2 * out.stride(0) and q.stride()[2:] == out.stride()[2:]
        self.chain = chain
        self.first[10] = chain.as_strided(out.shape, out.stride(), out.storage_offset())
        self.chained_q = chain.as_strided(q.shape, q.stride(), q.storage_offset())

    def restore(self) -> None:
        """The aliased V pages as they were, and nothing left of an earlier
        output of ``first``."""
        for view, pristine in self.regions:
            view.copy_(pristine)
        self.chain.zero_()


def _aliased_kernels(placement: str, *, first_ctas: bool = False):
    """The banks of a scenario with the paired kernel's defaults on their
    grid of two waves, whose release is at entry in every CTA (``entry``),
    or the same kernel with the release of the last wave's CTAs alone
    (``last_wave``); and the joined kernel without dependent launch."""
    if not is_cuda():
        pytest.skip("capture PDL requires CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention requires SM100 tensor memory")
    sms = torch.cuda.get_device_properties(DEVICE).multi_processor_count
    banks = _AliasedBanks(sms + MULTIWAVE_LAST_WAVE, first_ctas=first_ctas)
    bound = helion.kernel(
        folded_paged_attention.fn, backend="cute", static_shapes=True
    ).bind(tuple(banks.first))
    banks.restore()
    _normalized, _source, tracked, plain = compile_multiwave_defaults(
        bound, banks.first, last_wave_guard=placement == "last_wave"
    )
    for bank in (banks.second, banks.third):
        tracked(*bank)
    torch.cuda.synchronize()
    return banks, tracked, plain


@pytest.fixture(scope="module", params=["entry", "last_wave"])
def aliased(request):
    """Outputs over pages the last wave of the launch ahead reads, under
    both placements of the release."""
    return _aliased_kernels(request.param)


@pytest.fixture(scope="module", params=["entry", "last_wave"])
def aliased_first_ctas(request):
    """Outputs over pages the first CTAs of the launch ahead read, which
    outlive every other CTA of their launch, under both placements."""
    return _aliased_kernels(request.param, first_ctas=True)


def _launched(kernel, bank) -> tuple[torch.Tensor, torch.Tensor]:
    """One eager call's output and LSE as bits, the launch complete."""
    output, lse = kernel(*bank)
    torch.cuda.synchronize()
    return _bits(output), _bits(lse)


def _check_an_output_over_pages_the_launch_ahead_reads(banks, tracked, plain) -> None:
    """Two captured calls back to back: the second stores its output into V
    pages that the reading requests of the first take at the end of their
    walk. It may read its own inputs while the first runs, and stores
    behind its dependency wait, so the first call has the bits of a call
    that ran alone over the pages as they were, in every replay."""
    banks.restore()
    first = _launched(plain, banks.first)
    second = _launched(plain, banks.second)
    banks.restore()
    for value, reference in zip(_launched(tracked, banks.first), first, strict=True):
        assert torch.equal(value, reference)
    # What a store ahead of the wait would do: over zeroed pages, and over
    # the second call's output, every reading request of the first call
    # gets another output, and no other request does. (The LSE reads K
    # alone.)
    for early in ("zeroes", "output"):
        banks.restore()
        if early == "zeroes":
            banks.regions[0][0].zero_()
        else:
            _launched(plain, banks.second)
        output, lse = _launched(plain, banks.first)
        rows = (output != first[0]).reshape(banks.requests, -1).any(dim=1)
        assert bool(rows[banks.readers].all()), early
        assert int(rows.sum()) == MULTIWAVE_LAST_WAVE, early
        assert torch.equal(lse, first[1]), early
    banks.restore()
    graph = torch.cuda.CUDAGraph()
    with observed_wait_modes() as modes, torch.cuda.graph(graph):
        captured = [tracked(*banks.first), tracked(*banks.second)]
    assert modes == [WaitMode.EARLY, WaitMode.BEFORE_WRITES], modes
    for replay in range(ALIAS_REPLAYS):
        banks.restore()
        for _output, lse in captured:
            lse.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        for call, expected in zip(captured, (first, second), strict=True):
            for value, reference in zip(call, expected, strict=True):
                assert torch.equal(_bits(value), reference), replay


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_an_output_over_pages_the_last_wave_of_the_launch_ahead_reads(aliased):
    # The last wave enters as the first wave's CTAs leave, so under either
    # placement the launch behind starts while the last wave still has
    # these pages ahead of it.
    _check_an_output_over_pages_the_launch_ahead_reads(*aliased)


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_an_output_over_pages_the_first_ctas_of_the_launch_ahead_read(
    aliased_first_ctas,
):
    # The window the release at entry in every CTA opens: the first CTAs of
    # the launch ahead outlive its other CTAs, and the launch behind starts
    # once as many CTAs as the last wave holds have exited, beside them.
    # Under the last wave's guard it starts when those first CTAs have
    # exited. A store ahead of the dependency wait would change their last
    # pages under the first placement alone.
    _check_an_output_over_pages_the_launch_ahead_reads(*aliased_first_ctas)


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_a_chain_of_three_launches_over_the_first_ones_pages_and_output(aliased):
    # Three captured calls. The second and the third store their outputs
    # into V pages the last wave of the first reads at the end of its walk,
    # and the third reads the first call's output as its Q: an input that a
    # launch two ahead of it stores. The second may read ahead; the third
    # may not, and waits at its entry for the second, which has waited for
    # the first. Every call has the bits of the three run one after another.
    banks, tracked, plain = aliased
    chained = [banks.chained_q, *banks.third[1:]]
    banks.restore()
    first = _launched(plain, banks.first)
    second = _launched(plain, banks.second)
    third = _launched(plain, chained)
    # The third call's result depends on the first call's output.
    banks.restore()
    assert not torch.equal(_launched(plain, chained)[0], third[0])
    banks.restore()
    graph = torch.cuda.CUDAGraph()
    with observed_wait_modes() as modes, torch.cuda.graph(graph):
        captured = [
            tracked(*banks.first),
            tracked(*banks.second),
            tracked(*chained),
        ]
    assert modes == [WaitMode.EARLY, WaitMode.BEFORE_WRITES, WaitMode.EARLY], modes
    for replay in range(ALIAS_REPLAYS):
        banks.restore()
        for _output, lse in captured:
            lse.fill_(float("nan"))
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        for call, expected in zip(captured, (first, second, third), strict=True):
            for value, reference in zip(call, expected, strict=True):
                assert torch.equal(_bits(value), reference), replay


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_unknown_delayed_producer_forces_early_wait(kernels):
    args, tracked, plain = kernels
    lengths = args[6]
    values = torch.tensor([0, 1025], dtype=torch.int32, device=DEVICE)
    produce(values, lengths, delay=CONTROL_DELAY, grid=1, block=32)
    tracked(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with observed_wait_modes() as modes, torch.cuda.graph(graph):
        output_a = tracked(*args)
        produce(values, lengths, delay=CONTROL_DELAY, grid=1, block=32)
        output_b = tracked(*args)
    assert modes == [WaitMode.EARLY, WaitMode.EARLY], modes
    for iteration in range(2):
        values.copy_(
            torch.tensor(
                [129, 0] if iteration else [0, 1025], dtype=torch.int32, device=DEVICE
            )
        )
        lengths.copy_(values)
        expected_b = plain(*args)
        lengths.fill_(0)
        expected_a = plain(*args)
        graph.replay()
        close(output_a, expected_a)
        close(output_b, expected_b)


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_capture_overlap_on_non_current_device(kernels):
    args, tracked, plain = kernels
    launch_device = args[0].device
    assert launch_device.index is not None
    other_device = (launch_device.index + 1) % torch.cuda.device_count()
    # Initialize the other context before starting global CUDA capture.
    with torch.cuda.device(other_device):
        torch.empty(1, device=other_device)
        torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with (
        observed_wait_modes() as modes,
        torch.cuda.device(launch_device),
        torch.cuda.graph(graph),
        torch.cuda.device(other_device),
    ):
        output_a = tracked(*args)
        output_b = tracked(*args)
        assert torch.cuda.current_device() == other_device
    assert modes == [WaitMode.EARLY, WaitMode.BEFORE_WRITES], modes
    for _ in range(2):
        args[0].add_(0.03125)
        args[2].neg_()
        expected = plain(*args)
        graph.replay()
        close(output_a, expected)
        close(output_b, expected)


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
def test_opt_out_override_reaches_a_result_cached_without_it(tmp_path, monkeypatch):
    # An application tunes the paired kernel once, and the cached result
    # carries the guarded default. It then declares that it edits its
    # captured graphs, with one of the two opt-out overrides. The cache
    # identity does not hold the overrides, so the kernel reads the same
    # entry back (a miss would raise), and what it compiles from it must not
    # read ahead of a predecessor: no launch of it takes part in the capture
    # proof, where the tuned kernel's second launch waits only before its
    # writes.
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention requires SM100 tensor memory")
    monkeypatch.setenv("HELION_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HELION_ASSERT_CACHE_HIT", "1")
    for name in ("HELION_SKIP_CACHE", "HELION_FORCE_AUTOTUNE"):
        monkeypatch.delenv(name, raising=False)
    args = _capture_args(True)

    def declared(overrides):
        # Without an automatic structural policy, the config schema of every
        # kernel that has an override: one cache key with and without them.
        return helion.kernel(
            paged_online_attention.fn,
            backend="cute",
            static_shapes=True,
            cute_structural_policy=None,
            autotune_config_overrides=overrides,
        )

    def captured(kernel):
        kernel(*args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with observed_wait_modes() as modes, torch.cuda.graph(graph):
            outputs = [kernel(*args), kernel(*args)]
        graph.replay()
        torch.cuda.synchronize()
        return modes, outputs

    tuned_kernel = declared({})
    bound = tuned_kernel.bind(tuple(args))
    tuned = bound.config_spec.normalized_config(
        helion.Config(
            block_sizes=[8],
            cute_mma_dag="tcgen05",
            cute_mma_dag_warps=4,
            cute_mma_dag_stages=2,
            cute_mma_dag_pass_warps="attention_paired",
            cute_min_blocks_per_mp=1,
            cute_mma_dag_query_tma=True,
        )
    )
    assert tuned.get("cute_pdl_capture_overlap") is True
    stored = default_autotuner_fn(bound, args)
    stored.put(tuned)
    modes, expected = captured(tuned_kernel)
    assert bound._config == tuned
    assert modes == [WaitMode.EARLY, WaitMode.BEFORE_WRITES], modes
    for overrides in ({"cute_pdl": False}, {"cute_pdl_capture_overlap": False}):
        kernel = declared(overrides)
        opted_out = kernel.bind(tuple(args))
        assert default_autotuner_fn(opted_out, args).key == stored.key
        modes, outputs = captured(kernel)
        assert modes == [], (overrides, modes)
        assert opted_out._config is not None
        assert opted_out._config.get("cute_pdl") is False
        for output in outputs:
            close(output, expected[0])
    # An application may call the kernel's autotuner itself and pass the hit
    # on: the autotuner returns what the kernel would compile.
    kernel = declared({"cute_pdl_capture_overlap": False})
    driven = kernel.bind(tuple(args))
    driven.set_config(default_autotuner_fn(driven, args).autotune())
    modes, outputs = captured(kernel)
    assert modes == [], modes
    for output in outputs:
        close(output, expected[0])
    # A kernel reads its overrides when it selects a config. Declared on the
    # tuned kernel, which already runs, the opt-out holds from ``reset`` on.
    tuned_kernel.settings.autotune_config_overrides = {"cute_pdl": False}
    tuned_kernel.reset()
    modes, outputs = captured(tuned_kernel)
    assert modes == [], modes
    for output in outputs:
        close(output, expected[0])
