from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

import helion
from helion.autotuner.base_search import PopulationBasedSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.logger import AutotuningLogger
from helion.runtime.settings import Settings

if TYPE_CHECKING:
    from collections.abc import Callable


def make_search(
    callback: Callable[..., list[float]] | None,
    original: torch.Tensor,
    *,
    mutated: bool,
    isolated: Mock,
) -> PopulationBasedSearch:
    settings = Settings(
        autotune_log_level=logging.CRITICAL,
        autotune_suspicious_rebenchmark_ratio=0,
        autotune_benchmark_fn=callback,
    )
    search = PopulationBasedSearch.__new__(PopulationBasedSearch)
    search.settings = settings
    search.args = (original,)
    search.log = AutotuningLogger(settings)
    search.best_perf_so_far = 100.0
    search.benchmark_provider = SimpleNamespace(
        mutated_arg_indices=[0] if mutated else [],
        benchmark_isolated=isolated,
    )
    search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
    search.config_spec = SimpleNamespace(backend=None)
    return search


def clone_args(
    args: tuple[torch.Tensor, ...],
    process_group_name: str | None,
    idx_to_clone: list[int] | None,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        value.clone() if idx_to_clone is None or i in idx_to_clone else value
        for i, value in enumerate(args)
    )


@pytest.mark.parametrize("mutated", [False, True])
@pytest.mark.parametrize("use_isolated", [False, True])
def test_custom_callback_honors_private_storage(
    mutated: bool, use_isolated: bool
) -> None:
    original = torch.arange(8, dtype=torch.float32)
    observed: list[torch.Tensor] = []
    forwarded = []

    def callback(
        fns: list[Callable[[], object]], *, repeat: int, desc: str | None = None
    ) -> list[float]:
        forwarded.append((repeat, desc))
        for fn in fns:
            fn()
        return [50.0, 51.0]

    isolated = Mock(side_effect=AssertionError("custom callback owns timing"))
    search = make_search(callback, original, mutated=mutated, isolated=isolated)
    members = [
        PopulationMember(observed.append, [100.0], (), helion.Config(num_warps=4)),
        PopulationMember(observed.append, [101.0], (), helion.Config(num_warps=8)),
    ]
    with (
        patch(
            "helion.autotuner.base_search._clone_args", side_effect=clone_args
        ) as clone,
        patch("helion.autotuner.base_search.clear_jit_fast_path_caches") as clear,
        patch("helion.autotuner.base_search.dist.is_initialized", return_value=False),
    ):
        search.rebenchmark(
            members,
            desc="cold callback",
            target_ms=1000.0,
            use_isolated=use_isolated,
            confirm_suspicious=False,
            candidate_private_args=True,
        )
    assert forwarded == [(9, "cold callback")]
    assert len(observed) == 2
    assert all(torch.equal(value, original) for value in observed)
    assert len({original.data_ptr(), *(value.data_ptr() for value in observed)}) == 3
    assert clone.call_count == 3  # A retained sacrificial clone plus both candidates.
    assert all(call.kwargs["idx_to_clone"] is None for call in clone.call_args_list)
    assert clear.call_count == 4
    isolated.assert_not_called()
    assert members[0].perfs == [100.0, 50.0]
    assert members[1].perfs == [101.0, 51.0]


def test_custom_callback_default_retains_shared_readonly_storage() -> None:
    original = torch.arange(8, dtype=torch.float32)
    observed = []

    def callback(
        fns: list[Callable[[], object]], *, repeat: int, desc: str | None = None
    ) -> list[float]:
        for fn in fns:
            fn()
        return [50.0, 51.0]

    search = make_search(callback, original, mutated=False, isolated=Mock())
    members = [
        PopulationMember(observed.append, [100.0], (), helion.Config(num_warps=4)),
        PopulationMember(observed.append, [101.0], (), helion.Config(num_warps=8)),
    ]
    with patch("helion.autotuner.base_search._clone_args") as clone:
        search.rebenchmark(members, confirm_suspicious=False)
    assert observed[0] is observed[1] is original
    clone.assert_not_called()


def test_successful_process_isolation_keeps_existing_forwarding() -> None:
    original = torch.arange(8, dtype=torch.float32)
    isolated = Mock(return_value=[50.0, 51.0])
    search = make_search(None, original, mutated=False, isolated=isolated)
    members = [
        PopulationMember(lambda value: None, [100.0], (), helion.Config(num_warps=4)),
        PopulationMember(lambda value: None, [101.0], (), helion.Config(num_warps=8)),
    ]
    with patch("helion.autotuner.base_search._clone_args") as clone:
        search.rebenchmark(
            members,
            desc="process isolation",
            target_ms=1000,
            candidate_private_args=True,
            confirm_suspicious=False,
        )
    isolated.assert_called_once()
    assert isolated.call_args.kwargs["fresh_process"] is True
    assert isolated.call_args.kwargs["warmup"] == 1
    assert isolated.call_args.kwargs["rep"] == 1000
    assert isolated.call_args.kwargs["desc"] == "process isolation"
    clone.assert_not_called()


def test_private_readonly_storage_when_process_worker_unavailable() -> None:
    original = torch.arange(8, dtype=torch.float32)
    observed = []
    isolated = Mock(return_value=None)
    search = make_search(None, original, mutated=False, isolated=isolated)

    def steady(
        fn: Callable[[], object],
        *,
        warmup: int,
        rep: int,
        return_mode: str,
        process_group_name: str | None = None,
    ) -> float:
        fn()
        return 50.0

    search.config_spec = SimpleNamespace(
        backend=SimpleNamespace(get_do_bench=lambda: steady)
    )
    members = [
        PopulationMember(observed.append, [100.0], (), helion.Config(num_warps=4)),
        PopulationMember(observed.append, [101.0], (), helion.Config(num_warps=8)),
    ]
    with patch("helion.autotuner.base_search._clone_args", side_effect=clone_args):
        search.rebenchmark(
            members,
            candidate_private_args=True,
            use_interleaved=False,
            confirm_suspicious=False,
        )
    assert isolated.call_args.kwargs["fresh_process"] is True
    assert all(torch.equal(value, original) for value in observed)
    assert len({original.data_ptr(), *(value.data_ptr() for value in observed)}) == 3


def test_distributed_custom_callback_keeps_existing_shared_storage() -> None:
    original = torch.arange(8, dtype=torch.float32)
    observed = []

    def callback(
        fns: list[Callable[[], object]], *, repeat: int, desc: str | None = None
    ) -> list[float]:
        for fn in fns:
            fn()
        return [50.0, 51.0]

    search = make_search(callback, original, mutated=False, isolated=Mock())
    members = [
        PopulationMember(observed.append, [100.0], (), helion.Config(num_warps=4)),
        PopulationMember(observed.append, [101.0], (), helion.Config(num_warps=8)),
    ]
    with (
        patch("helion.autotuner.base_search.dist.is_initialized", return_value=True),
        patch(
            "helion.autotuner.base_search.sync_object",
            side_effect=lambda value, **kwargs: value,
        ),
        patch("helion.autotuner.base_search._clone_args") as clone,
    ):
        search.rebenchmark(
            members, candidate_private_args=True, confirm_suspicious=False
        )
    assert observed[0] is observed[1] is original
    clone.assert_not_called()
