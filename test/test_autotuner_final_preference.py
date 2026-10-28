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
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable
from test._cute_vloop_sink_kernels import col_reduce_sum_static
from test.test_cute_pdl import _COLUMN_SUM
from test.test_cute_tcgen05_fixed_chain import _bind_gemm
from test.test_cute_tcgen05_fixed_chain import _cpu_only  # noqa: F401
from test.test_cute_tcgen05_fixed_chain import _source
from test.test_cute_tcgen05_pdl import _persistent_config

import helion
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.pdl_release import generic_release_rendered
from helion._compiler.triton.backend import TritonBackend
from helion.autotuner.base_search import PopulationBasedSearch
from helion.runtime.settings import Settings

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
_NONE = _module(_WAIT + _BODY)
_EXIT = _module(_WAIT + _BODY + _RELEASE)
_FAMILY = _module(_WAIT + _BODY + "    if warp_idx == 0:\n    " + _RELEASE)
_OFF = _module(_BODY)


def _member(perf: float, source: str = _OFF, **config: object) -> SimpleNamespace:
    return SimpleNamespace(perf=perf, config=helion.Config(**config), source=source)


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
    assert not generic_release_rendered("this is not python")


def test_cute_prefers_a_kernel_that_renders_the_generic_release() -> None:
    on = helion.Config(block_sizes=[64], cute_pdl=True)
    for source, rank in (
        (_ENTRY, 0),
        (_GUARD, 0),
        (_NONE, 1),
        (_EXIT, 1),
        (_FAMILY, 1),
    ):
        assert CuteBackend.finalist_preference(on, lambda s=source: s) == rank, source
    # Without the knob the config ranks last and the module is never rendered.
    never = Mock(side_effect=AssertionError("rendered"))
    assert CuteBackend.finalist_preference(helion.Config(block_sizes=[64]), never) == 1
    assert (
        CuteBackend.finalist_preference(
            helion.Config(block_sizes=[64], cute_pdl=False), never
        )
        == 1
    )


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


@pytest.mark.usefixtures("_cpu_only")
def test_rendered_kernels_rank_as_their_lowering() -> None:
    # Real renders: a column reduction on the generic path releases at entry
    # (one wave) and ranks first; a tcgen05 GEMM on the entry-wait path gets
    # the generic wait but its MMA warp owns the release, so its configs
    # rank last whatever placement they carry: a pair of them is a tie.
    with _mock_cuda_unavailable():
        bound = _cpu_bind(
            col_reduce_sum_static, (torch.empty(1024, 1024, dtype=torch.bfloat16),)
        )
        generic = bound.config_spec.normalized_config(
            helion.Config(**_COLUMN_SUM, cute_pdl=True)
        )
        assert (
            CuteBackend.finalist_preference(generic, lambda: bound.to_code(generic))
            == 0
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
            == 1
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
