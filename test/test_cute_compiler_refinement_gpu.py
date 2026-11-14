from __future__ import annotations

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfRefEager
from helion.autotuner import PatternSearch
from helion.autotuner.search_space_logger import canonical_config_id
import helion.language as hl


@helion.kernel(autotune_effort="full", static_shapes=True, autotune_log_level=0)
def _add_one(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + 1
    return out


@pytest.mark.slow
@skipIfRefEager("compiles and benchmarks compiler-option variants")
def test_compiler_refinement_runs_before_real_final_selection(monkeypatch):
    # Ordinary tests disable final selection. A single initial candidate and
    # no neighbor generations leave its other compiler pipeline for refinement.
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_SELECTION_ROUNDS", "2")
    monkeypatch.setenv("HELION_AUTOTUNE_FINAL_REBENCHMARK_TOP_K", "2")
    args = (torch.randn(2048, device=DEVICE),)
    bound = _add_one.bind(args)
    search = PatternSearch(
        bound, args, initial_population=1, max_generations=0, copies=1
    )
    best = search.autotune(skip_cache=True)
    refinement = search._autotune_metrics.compiler_refinement
    assert refinement is not None and refinement["completed"] is True
    assert refinement["new_trials"] >= 1
    assert any(trial["outcome"] == "ok" for trial in refinement["trials"])
    selection = search._autotune_metrics.final_selection
    assert selection is not None and selection["policy"]["rounds"] == 2
    assert selection["selected_config_id"] == canonical_config_id(best)
    assert search._selected_member is not None
    assert search._selected_member.config == best
    torch.testing.assert_close(bound.compile_config(best)(*args), args[0] + 1)
