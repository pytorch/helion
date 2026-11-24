"""The final verification's tie rule: among finalists it cannot tell apart,
the backend's preference decides (``Backend.finalist_preference``).

The CuTe backend ranks a kernel that renders the generic early release of
its programmatic dependents (``cute_pdl`` on the generic path, released at
entry or from the last wave) ahead of one that does not.  That difference
is invisible to the kernel's isolated timing whenever a plain launch
separates its back-to-back calls, and paid by its successor in the real call
sequence.  The lowering is read off the finalist's rendered module: on the
families that own their wait and release (tcgen05 here) every config ranks
alike, so their finalists stay a plain tie.

Behind those it ranks a kernel that flushes a packet with the streaming store
policy (``cute_store_policy='streaming'``): the evict-first lines of its
output are what the next kernel of the call sequence reads, and the kernel's
own timing cannot see that either.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from test.cute_population_contracts import tcgen05_mma_support
from test.test_cute_pdl import _COLUMN_SUM
from test.test_cute_tcgen05_fixed_chain import _bind_gemm
from test.test_cute_tcgen05_fixed_chain import _source
from test.test_cute_tcgen05_pdl import _persistent_config

import helion
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.packet_flush_names import PACKET_FLUSH_STREAMING
from helion._compiler.cute.packet_flush_names import streaming_store_rendered
from helion._compiler.cute.pdl_release import generic_release_rendered
from helion._compiler.triton.backend import TritonBackend
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_vloop_sink_kernels import col_reduce_sum_static
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.effort_profile import FinalSelectionConfig
from helion.runtime.settings import Settings

# The B200 model with tcgen05 MMA support, CUDA hidden, the inputs traced as
# ``cuda:0`` tensors of a 148-SM device, and the reduction SMEM budget (the
# ``test_cute_tcgen05_fixed_chain`` binders' target).
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.cpu_render(
        mma_support=tcgen05_mma_support(),
        forbid_native=False,
        fake_device_sm_count=148,
        reduction_smem_budget_bytes=232448,
    ),
]

_ENV = "HELION_AUTOTUNE_FINAL_SELECTION_PREFERENCE_TOLERANCE"
_WAIT = "    cute.arch.griddepcontrol_wait()\n"
_RELEASE = "    cute.arch.griddepcontrol_launch_dependents()\n"
_BODY = "    pid_0 = cutlass.Int32(cute.arch.block_idx()[0])\n    out.store(pid_0)\n"


def _module(kernel_body: str) -> str:
    return f"@cute.kernel\ndef _helion_k(x, out):\n{kernel_body}\n_helion_k._helion_cute_use_pdl = True\n"


# The shapes the generic path renders (test_cute_pdl.py) and the shapes the
# families render, as module sources.
_ENTRY = _module(_WAIT + _RELEASE + _BODY)
_GUARD = _module(
    _WAIT
    + "    pdl_cta = cutlass.Int32(cute.arch.block_idx()[0])\n"
    + "    if pdl_cta >= 4:\n    "
    + _RELEASE
    + "    else:\n        pass\n"
    + _BODY
)
_SPREAD = _module(
    _WAIT
    + _RELEASE
    + "    pdl_spread_pad = cute.arch.alloc_smem(cutlass.Uint8, 90113, alignment=128)\n"
    + "    _helion_pdl_spread_keep(pdl_spread_pad.toint())\n"
    + _BODY
)
_NONE = _module(_WAIT + _BODY)
# A kernel whose packet flush took the streaming policy, and the plain flush.
_STREAMING = _module(
    _BODY + "    _cute_store_u32x4_streaming(out.iterator + pid_0 * 4, vals)\n"
)
_PLAIN_FLUSH = _module(
    _BODY + "    _cute_store_u32_vec(out.iterator + pid_0 * 4, vals)\n"
)
_STREAMING_ENTRY = _module(
    _WAIT
    + _RELEASE
    + _BODY
    + "    _cute_store_u16x8_streaming(out.iterator + pid_0 * 8, vals)\n"
)
_EXIT = _module(_WAIT + _BODY + _RELEASE)
_FAMILY = _module(_WAIT + _BODY + "    if warp_idx == 0:\n    " + _RELEASE)
_OFF = _module(_BODY)


def _member(perf: float, source: str = _OFF, **config: object) -> SimpleNamespace:
    return SimpleNamespace(
        perf=perf,
        perfs=[perf],
        config=helion.Config(**config),
        source=source,
        fn=object(),
    )


def _search(backend: str, members: list[SimpleNamespace] = ()) -> PopulationBasedSearch:  # type: ignore[assignment]
    # The searches of the autotuner tests are light doubles with real
    # settings: the preference is reached through the backend's registered
    # name, and the finalist's source through the kernel's ``to_code``.
    search = PopulationBasedSearch.__new__(PopulationBasedSearch)
    search.settings = Settings(backend=backend)
    search.log = Mock()
    search.format_performance = lambda perf: f"{perf:.4f}ms"
    by_config = {member.config: member.source for member in members}
    renders: list[helion.Config] = []

    def to_code(config: helion.Config) -> str:
        renders.append(config)
        return by_config[config]

    search.kernel = SimpleNamespace(to_code=to_code, renders=renders)
    return search


def test_the_generic_early_release_is_read_off_the_rendered_module() -> None:
    assert generic_release_rendered(_ENTRY)
    assert generic_release_rendered(_GUARD)
    assert not generic_release_rendered(_NONE)
    assert not generic_release_rendered(_EXIT)
    assert not generic_release_rendered(_FAMILY)
    assert not generic_release_rendered(_OFF)
    assert not generic_release_rendered("def not_a_kernel(): pass\n")
    # (a module that does not parse)
    assert not generic_release_rendered("def kernel(:")


def test_cute_prefers_a_kernel_that_renders_the_generic_release() -> None:
    # The generic early release ranks ahead of none; the release with the
    # pad of ``cute_pdl_spread`` (an early launch of the kernel then never
    # packs its CTAs onto the SMs its primary left idle) ranks ahead of both.
    on = helion.Config(block_sizes=[64], cute_pdl=True)
    for source, rank in (
        (_SPREAD, 0),
        (_ENTRY, 1),
        (_GUARD, 1),
        (_NONE, 2),
        (_EXIT, 2),
        (_FAMILY, 2),
    ):
        assert CuteBackend.finalist_preference(on, lambda s=source: s) == rank, source
    # Without the knob the config ranks last and the module is never rendered.
    never = Mock(side_effect=AssertionError("rendered"))
    assert CuteBackend.finalist_preference(helion.Config(block_sizes=[64]), never) == 2
    assert (
        CuteBackend.finalist_preference(
            helion.Config(block_sizes=[64], cute_pdl=False), never
        )
        == 2
    )


@skipUnlessBackends(["cute"])
def test_the_streaming_flush_is_read_off_the_rendered_module() -> None:
    for helper in PACKET_FLUSH_STREAMING.values():
        assert streaming_store_rendered(_module(_BODY + f"    {helper}(p, vals)\n"))
    for helper in PACKET_FLUSH_STREAMING:
        assert not streaming_store_rendered(_module(_BODY + f"    {helper}(p, v)\n"))
    assert streaming_store_rendered(_STREAMING_ENTRY)
    assert not streaming_store_rendered(_PLAIN_FLUSH)
    assert not streaming_store_rendered(_OFF)
    # The import of a helper the kernel never calls is no streaming store.
    assert not streaming_store_rendered(
        "from helion._compiler.cute.l2_policy import store_u32x4_streaming"
        " as _cute_store_u32x4_streaming\n" + _OFF
    )


@skipUnlessBackends(["cute"])
def test_cute_ranks_a_streaming_store_behind_every_default_store() -> None:
    # A kernel that flushes a streaming packet ranks three behind the same
    # kernel with the default policy, so behind every kernel that stores
    # with the default policy whatever it releases.
    streaming = helion.Config(block_sizes=[64], cute_store_policy="streaming")
    assert CuteBackend.finalist_preference(streaming, lambda: _STREAMING) == 5
    both = helion.Config(block_sizes=[64], cute_store_policy="streaming", cute_pdl=True)
    assert CuteBackend.finalist_preference(both, lambda: _STREAMING_ENTRY) == 4
    # The policy on a kernel that flushes no such packet (scalar stores,
    # narrower packets, a staged store) is the default policy's kernel.
    assert CuteBackend.finalist_preference(streaming, lambda: _PLAIN_FLUSH) == 2
    assert CuteBackend.finalist_preference(streaming, lambda: _OFF) == 2
    assert CuteBackend.finalist_preference(both, lambda: _ENTRY) == 1
    # A config with the default policy is never rendered for it.
    never = Mock(side_effect=AssertionError("rendered"))
    for policy in ("default", "l2_evict_last"):
        config = helion.Config(block_sizes=[64], cute_store_policy=policy)
        assert CuteBackend.finalist_preference(config, never) == 2


@skipUnlessBackends(["cute"])
def test_a_default_store_within_tolerance_wins_over_a_streaming_one() -> None:
    # Three finalists of one layout 0.03% apart, two of them with streaming
    # stores (the final selection of the GELU backward this rule was measured
    # on): the default store wins.  One 2% behind does not, and a streaming
    # config whose kernel flushes no packet ties as a default store.
    fastest = _member(
        92.69, _STREAMING, block_sizes=[512], cute_store_policy="streaming"
    )
    second = _member(
        92.70,
        _STREAMING,
        block_sizes=[512],
        cute_store_policy="streaming",
        cute_min_blocks_per_mp=3,
    )
    default = _member(92.72, _PLAIN_FLUSH, block_sizes=[512])
    far_default = _member(94.60, _PLAIN_FLUSH, block_sizes=[1024])
    live = [fastest, second, default, far_default]
    search = _search("cute", live)
    assert search._prefer_backend_preference_within_tolerance(fastest, live) is default
    search.log.assert_called_once()
    # The band's streaming finalist was rendered; the default one was not.
    assert search.kernel.renders == [fastest.config, second.config]
    search = _search("cute", [fastest, second, far_default])
    assert (
        search._prefer_backend_preference_within_tolerance(
            fastest, [fastest, second, far_default]
        )
        is fastest
    )
    search.log.assert_not_called()
    inert = _member(92.60, _OFF, block_sizes=[32], cute_store_policy="streaming")
    search = _search("cute", [inert, default])
    assert (
        search._prefer_backend_preference_within_tolerance(inert, [inert, default])
        is inert
    )
    # The store ranks before the release: a default store that releases
    # nothing wins over a streaming store that releases at entry.
    releasing = _member(
        92.65,
        _STREAMING_ENTRY,
        block_sizes=[256],
        cute_store_policy="streaming",
        cute_pdl=True,
    )
    search = _search("cute", [releasing, default])
    assert (
        search._prefer_backend_preference_within_tolerance(
            releasing, [releasing, default]
        )
        is default
    )


@skipUnlessBackends(["cute"])
def test_a_followed_kernel_prefers_its_release_down_to_a_trace_tick() -> None:
    # A 5.2 us kernel whose releasing twin reads one trace tick (32 ns) slower,
    # 0.6% and outside the 0.5% band: alone the faster one stays; a kernel
    # that another kernel of its sequence runs right behind
    # (followed_in_sequence) takes the twin, as its successor launches 0.3-2
    # us earlier behind the release, and so it does two ticks behind (the
    # band is three).  A twin four ticks behind is no tie.
    fastest = _member(0.005184, _OFF, block_sizes=[64])
    twin = _member(0.005216, _ENTRY, block_sizes=[64], cute_pdl=True)
    two_ticks = _member(0.005248, _ENTRY, block_sizes=[48], cute_pdl=True)
    far = _member(0.005312, _ENTRY, block_sizes=[32], cute_pdl=True)
    for followed, live, chosen in (
        (False, [fastest, twin], fastest),
        (True, [fastest, twin], twin),
        (True, [fastest, two_ticks], two_ticks),
        (True, [fastest, far], fastest),
    ):
        search = _search("cute", live)
        search._followed_in_sequence = followed
        assert (
            search._prefer_backend_preference_within_tolerance(fastest, live) is chosen
        ), (followed, [m.perf for m in live])
    # An explicit tolerance stands as given: 0 turns the preference off; an
    # invalid one is ignored (logged), and the followed band applies.
    for value, chosen in (("0", fastest), ("not-a-number", twin), ("-1", twin)):
        with patch.dict("os.environ", {_ENV: value}):
            search = _search("cute", [fastest, twin])
            search._followed_in_sequence = True
            assert (
                search._prefer_backend_preference_within_tolerance(
                    fastest, [fastest, twin]
                )
                is chosen
            ), value


def test_a_finalist_is_rendered_once_however_often_the_backend_reads_it() -> None:
    # A rendering is a codegen pass: the hook gets the module on demand and
    # may read it more than once.
    class Rereads:
        @classmethod
        def finalist_preference(cls, config: helion.Config, source) -> int:
            return 1 + len(source()) - len(source())

    fastest = _member(3.640, block_sizes=[32])
    close = _member(3.651, block_sizes=[64])
    search = _search("triton", [fastest, close])
    with patch(
        "helion._compiler.backend_registry.get_backend_class", return_value=Rereads
    ):
        assert (
            search._prefer_backend_preference_within_tolerance(
                fastest, [fastest, close]
            )
            is fastest
        )
    assert search.kernel.renders == [fastest.config, close.config]


def test_the_default_backend_prefers_nothing() -> None:
    # The base implementation (every backend but cute) ranks every config
    # alike, so the fastest finalist always keeps its place.
    never = Mock(side_effect=AssertionError("rendered"))
    on = helion.Config(block_sizes=[64], cute_pdl=True)
    assert TritonBackend.finalist_preference(on, never) == 0
    assert (
        TritonBackend.finalist_preference(helion.Config(block_sizes=[64]), never) == 0
    )
    fastest = _member(3.640, block_sizes=[32])
    close_on = _member(3.651, _ENTRY, block_sizes=[64], cute_pdl=True)
    search = _search("triton", [fastest, close_on])
    assert (
        search._prefer_backend_preference_within_tolerance(fastest, [fastest, close_on])
        is fastest
    )
    assert search.kernel.renders == []


def test_a_preferred_finalist_within_tolerance_wins_the_tie() -> None:
    # Four finalists 0.2-0.4% apart (the steady windows' resolution): the
    # fastest has the knob off, the first knob-on finalist within 0.5% that
    # renders the generic release wins; a knob-on finalist 2% slower does
    # not, nor does one whose placement releases nowhere early.
    fastest = _member(3.640, block_sizes=[32])
    close_off = _member(3.645, block_sizes=[16])
    close_on = _member(3.651, _ENTRY, block_sizes=[64], cute_pdl=True)
    closer_on_none = _member(
        3.648, _NONE, block_sizes=[64], cute_pdl=True, cute_pdl_release="none"
    )
    far_on = _member(3.720, _ENTRY, block_sizes=[128], cute_pdl=True)
    live = [fastest, close_off, close_on, closer_on_none, far_on]
    search = _search("cute", live)
    with patch.dict("os.environ", {}, clear=False):
        import os

        os.environ.pop(_ENV, None)
        assert (
            search._prefer_backend_preference_within_tolerance(fastest, live)
            is close_on
        )
        search.log.assert_called_once()
        # Only the finalists in the band were rendered, each once.
        assert search.kernel.renders == [close_on.config, closer_on_none.config]
        # The fastest already preferred: nothing changes, nothing is logged.
        search = _search("cute", live)
        assert (
            search._prefer_backend_preference_within_tolerance(close_on, live)
            is close_on
        )
        search.log.assert_not_called()
        # Only the far one is on: the fastest keeps its place.
        search = _search("cute", live)
        assert (
            search._prefer_backend_preference_within_tolerance(
                fastest, [fastest, close_off, far_on]
            )
            is fastest
        )
        # Among several preferred finalists in the band the fastest of them wins.
        faster_on = _member(3.646, _GUARD, block_sizes=[8], cute_pdl=True)
        search = _search("cute", [*live, faster_on])
        assert (
            search._prefer_backend_preference_within_tolerance(
                fastest, [*live, faster_on]
            )
            is faster_on
        )
        # A finalist that also spreads its CTAs ranks ahead of the releasing
        # ones, so it wins the band although it is not the fastest of them.
        spread_on = _member(
            3.650, _SPREAD, block_sizes=[4], cute_pdl=True, cute_pdl_spread=True
        )
        search = _search("cute", [*live, faster_on, spread_on])
        assert (
            search._prefer_backend_preference_within_tolerance(
                fastest, [*live, faster_on, spread_on]
            )
            is spread_on
        )


def test_a_family_that_owns_its_release_stays_a_plain_tie() -> None:
    # Two finalists of a kernel whose family places its own wait and release
    # (the source shape of the tcgen05 path: the release under a role
    # predicate, never right after the top-level wait) differ only in
    # unrelated knobs: the fastest keeps its place whatever the knob says.
    fastest = _member(34.94, _FAMILY, block_sizes=[512, 256, 64], cute_pdl=True)
    other = _member(
        35.06, _FAMILY, block_sizes=[256, 256, 64], cute_pdl=True, l2_groupings=[2]
    )
    search = _search("cute", [fastest, other])
    assert (
        search._prefer_backend_preference_within_tolerance(fastest, [fastest, other])
        is fastest
    )
    search.log.assert_not_called()


@pytest.mark.parametrize(
    "raw,expected", [("0", 3.640), ("0.03", 3.720), ("bad", 3.640)]
)
def test_the_tolerance_is_an_environment_override(raw: str, expected: float) -> None:
    # A knob-on finalist 2.2% behind the fastest is outside the default band
    # (and the band an unreadable override falls back to) and inside a 3%
    # one; a zero band keeps the fastest.
    fastest = _member(3.640, block_sizes=[32])
    far_on = _member(3.720, _ENTRY, block_sizes=[128], cute_pdl=True)
    search = _search("cute", [fastest, far_on])
    with patch.dict("os.environ", {_ENV: raw}):
        chosen = search._prefer_backend_preference_within_tolerance(
            fastest, [fastest, far_on]
        )
    assert chosen.perf == expected
    if raw == "bad":
        search.log.warning.assert_called_once()


@skipUnlessBackends(["cute"])
def test_rendered_kernels_rank_as_their_lowering() -> None:
    # Real renders: a column reduction on the generic path releases at entry
    # (one wave) and ranks ahead of the families, first with the spread pad;
    # a tcgen05 GEMM on the entry-wait path gets
    # the generic wait but its MMA warp owns the release, so its configs
    # rank last whatever placement they carry: a pair of them is a tie.
    with mock_cuda_unavailable():
        bound = cpu_bind(
            col_reduce_sum_static, (torch.empty(1024, 1024, dtype=torch.bfloat16),)
        )
        generic = bound.config_spec.normalized_config(
            helion.Config(**_COLUMN_SUM, cute_pdl=True)
        )
        assert (
            CuteBackend.finalist_preference(generic, lambda: bound.to_code(generic))
            == 1
        )
        spread = bound.config_spec.normalized_config(
            helion.Config(**_COLUMN_SUM, cute_pdl=True, cute_pdl_spread=True)
        )
        assert (
            CuteBackend.finalist_preference(spread, lambda: bound.to_code(spread)) == 0
        )
    gemm = _bind_gemm((1024, 1024, 1024), torch.float16)
    sources = {}
    for placement in ("auto", "entry", "none"):
        config = _persistent_config(
            cute_pdl=True,
            tcgen05_strategy="role_local_with_scheduler",
            tcgen05_warp_spec_scheduler_warps=1,
            cute_pdl_release=placement,
        )
        with gemm.env:
            normalized = gemm.config_spec.normalized_config(config)
        if normalized.config.get("tcgen05_strategy") != "role_local_with_scheduler":
            pytest.skip("scheduler-warp strategy not admitted on this binding")
        sources[placement] = _source(gemm, normalized)
        assert "_helion_cute_use_pdl = True" in sources[placement]
        assert (
            CuteBackend.finalist_preference(normalized, lambda p=placement: sources[p])
            == 2
        )
    fastest = SimpleNamespace(
        perf=34.94,
        config=helion.Config(
            cute_pdl=True, cute_pdl_release="none", block_sizes=[512, 256, 64]
        ),
        source=sources["none"],
    )
    other = SimpleNamespace(
        perf=35.06,
        config=helion.Config(cute_pdl=True, block_sizes=[128, 128, 64]),
        source=sources["auto"],
    )
    search = _search("cute", [fastest, other])
    assert (
        search._prefer_backend_preference_within_tolerance(fastest, [fastest, other])
        is fastest
    )


@skipUnlessBackends(["cute"])
def test_cute_names_the_releasing_twin_where_the_search_flips_the_knob() -> None:
    # A column sum on the CPU stand-in's B200 model: the search flips
    # cute_pdl, so a config without it has a twin with it at the auto
    # placement, also where the knob-off config names an inert one (none); a
    # config with it has none, and neither does any config of a backend
    # without the knob.
    with mock_cuda_unavailable():
        bound = cpu_bind(
            col_reduce_sum_static, (torch.empty(1024, 1024, dtype=torch.bfloat16),)
        )
    spec = bound.config_spec
    for off in (
        helion.Config(**_COLUMN_SUM),
        helion.Config(**_COLUMN_SUM, cute_pdl_release="none"),
    ):
        twin = CuteBackend.releasing_twin(off, spec)
        assert twin is not None
        assert twin.config == {
            **off.config,
            "cute_pdl": True,
            "cute_pdl_release": "auto",
        }
        assert CuteBackend.releasing_twin(twin, spec) is None
        assert TritonBackend.releasing_twin(off, spec) is None


def _twin_search(
    finalists: list[SimpleNamespace], twin_perf: float, twin_fn: object = None
) -> PopulationBasedSearch:
    # A followed kernel's search double whose config generator and benchmark
    # pass the twin through: it comes back timed at ``twin_perf``.
    search = _search("cute", finalists)
    search._followed_in_sequence = True
    search.config_spec = SimpleNamespace()
    search._finalist_history_perf = lambda member: member.perf
    search.config_gen = SimpleNamespace(flatten=lambda config: config)

    def make_unbenchmarked(config: helion.Config) -> SimpleNamespace:
        # The canonical form the search keeps: a knob-on config holds its
        # placement (``_normalize_cute_pdl``).
        canonical = {**config.config}
        if canonical.get("cute_pdl") is True:
            canonical.setdefault("cute_pdl_release", "auto")
        return SimpleNamespace(
            config=helion.Config(**canonical), perfs=[], perf=float("inf"), fn=None
        )

    def benchmark_population(members, **_: object) -> None:
        for member in members:
            member.perfs.append(twin_perf)
            member.perf = twin_perf
            member.fn = twin_fn

    twin_fn = object() if twin_fn is None else twin_fn

    search.make_unbenchmarked = make_unbenchmarked
    search.benchmark_population = Mock(side_effect=benchmark_population)
    return search


def test_a_followed_kernel_times_the_releasing_twin_of_its_fastest_finalist() -> None:
    # The finalists of a followed kernel all leave cute_pdl off (the search's
    # region): the twin of the fastest is compiled, timed and added, so the
    # preference band can take it; a twin the search timed already joins as
    # timed, a twin that is a finalist comes back as that finalist (the screen
    # keeps it), and a backend without the knob adds nothing.
    fastest = _member(0.002464, block_sizes=[32], num_threads=[8, 16])
    other = _member(0.002496, block_sizes=[64], num_threads=[8, 16])
    twin_config = helion.Config(block_sizes=[32], num_threads=[8, 16], cute_pdl=True)
    canonical = helion.Config(
        block_sizes=[32], num_threads=[8, 16], cute_pdl=True, cute_pdl_release="auto"
    )
    with patch.object(CuteBackend, "releasing_twin", return_value=twin_config):
        search = _twin_search([fastest, other], 0.002496)
        (twin,) = search._releasing_twins([fastest, other], {})
        assert twin.config == canonical and twin.perf == 0.002496
        search.benchmark_population.assert_called_once()
        # The search timed that twin already, as its canonical form (the
        # comparison is made after canonicalizing): it joins the finalists as
        # the search timed it, and a finalist that is it comes back as that
        # finalist, which the caller does not add again; neither is timed
        # again.
        timed = _member(0.002528, **canonical.config)
        for known, finalists in (
            ({canonical: timed}, [fastest, other]),
            ({}, [fastest, other, timed]),
        ):
            search = _twin_search(finalists, 0.002496)
            (returned,) = search._releasing_twins(finalists, known)
            assert returned is timed
            search.benchmark_population.assert_not_called()
    with patch.object(CuteBackend, "releasing_twin", return_value=None):
        search = _twin_search([fastest, other], 0.002496)
        assert search._releasing_twins([fastest, other], {}) == []


@skipUnlessBackends(["cute"])
def test_a_pinned_seed_cannot_undo_a_followed_kernels_release() -> None:
    # The fastest finalist is a pinned seed without the release, its twin one
    # tick behind: a followed kernel takes the twin (the pinned rule first,
    # the preference last); any other kernel keeps the pinned seed.
    seed = _member(0.005184, _OFF, block_sizes=[64])
    twin = _member(0.005216, _ENTRY, block_sizes=[64], cute_pdl=True)
    live = [seed, twin]
    for followed, chosen in ((True, twin), (False, seed)):
        search = _search("cute", live)
        search._followed_in_sequence = followed
        search._final_rebenchmark_pinned_tolerance = lambda: 0.0
        assert search._select_finalist(live, {seed.config}) is chosen, followed


def test_a_twin_that_is_a_finalists_kernel_adds_nothing() -> None:
    # The twin's code is a finalist's (the knob changed nothing it renders):
    # the benchmark settles it as that finalist's kernel, and it is dropped
    # instead of timing that kernel twice.
    fastest = _member(0.002464, block_sizes=[32], num_threads=[8, 16])
    other = _member(0.002496, block_sizes=[64], num_threads=[8, 16])
    twin_config = helion.Config(block_sizes=[32], num_threads=[8, 16], cute_pdl=True)
    with patch.object(CuteBackend, "releasing_twin", return_value=twin_config):
        search = _twin_search([fastest, other], 0.002464, twin_fn=fastest.fn)
        assert search._releasing_twins([fastest, other], {}) == []
        search.benchmark_population.assert_called_once()


@skipUnlessBackends(["cute"])
def test_a_pinned_seed_keeps_the_band_measured_from_the_fastest() -> None:
    # A pinned seed 0.6% behind the fastest finalist wins its tie under a 1%
    # pinned tolerance; with the preference tolerance set to 0, a releasing
    # config between the two (behind the fastest) is not taken: the band is
    # measured from the fastest, not from the seed.
    fastest = _member(0.005184, _OFF, block_sizes=[64])
    seed = _member(0.005216, _OFF, block_sizes=[48])
    releasing = _member(0.005200, _ENTRY, block_sizes=[32], cute_pdl=True)
    live = [fastest, seed, releasing]
    with patch.dict("os.environ", {_ENV: "0"}):
        search = _search("cute", live)
        search._followed_in_sequence = True
        search._final_rebenchmark_pinned_tolerance = lambda: 0.01
        assert search._select_finalist(live, {seed.config}) is seed
    # A tolerance of 0 turns the preference off: a releasing fastest finalist
    # does not take the tie back from the pinned seed (the band, measured from
    # it, would hold it).
    fastest = _member(0.005184, _ENTRY, block_sizes=[64], cute_pdl=True)
    live = [fastest, seed]
    with patch.dict("os.environ", {_ENV: "0"}):
        search = _search("cute", live)
        search._followed_in_sequence = True
        search._final_rebenchmark_pinned_tolerance = lambda: 0.01
        assert search._select_finalist(live, {seed.config}) is seed


def test_the_budget_counts_a_twin_only_where_one_can_exist() -> None:
    # The rounds keep the pinned seed and the incumbent outside the top_k; a
    # followed kernel whose backend can name a releasing twin for its spec
    # keeps its twin too, and only it budgets for one more finalist.
    policy = FinalSelectionConfig(top_k=4, family_finalists=0)
    assert policy.finalists(9) == 6
    assert policy.finalists(9, twins=1) == 7
    assert policy.estimated_seconds(9, twins=1) > policy.estimated_seconds(9)
    for backend, followed, names, twins in (
        ("cute", True, True, 1),
        ("cute", True, False, 0),
        ("cute", False, True, 0),
        ("triton", True, True, 0),
    ):
        search = _search(backend)
        search._followed_in_sequence = followed
        search.config_spec = SimpleNamespace()
        with patch.object(CuteBackend, "names_releasing_twins", return_value=names):
            assert search._twin_survivors() == twins, (backend, followed, names)
