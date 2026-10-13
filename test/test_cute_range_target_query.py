from __future__ import annotations

from types import SimpleNamespace

import pytest

from helion.language import loops


@pytest.mark.parametrize("backend", ["cute", "pallas"])
def test_nontriton_range_choices_never_initialize_triton_driver(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fields = (
        "static_ranges",
        "range_unroll_factors",
        "range_warp_specialize",
        "range_num_stages",
        "range_multi_buffers",
        "range_flattens",
    )
    spec = SimpleNamespace(backend_name=backend, **{name: [] for name in fields})
    monkeypatch.setattr(
        loops.CompileEnvironment, "current", lambda: SimpleNamespace(config_spec=spec)
    )
    monkeypatch.setattr(
        loops, "use_tileir_tunables", lambda: pytest.fail("non-Triton queried driver")
    )
    monkeypatch.setattr(loops, "_supports_warp_specialize", lambda: False)
    loops._add_config_range_choice([7], allow_static_range=True)
    assert len(spec.static_ranges) == 1
    assert len(spec.range_unroll_factors) == 1
    assert not spec.range_warp_specialize


@pytest.mark.parametrize("backend", ["triton", "tileir"])
@pytest.mark.parametrize("tileir", [False, True])
def test_triton_retains_original_tileir_range_choice_gate(
    backend: str, tileir: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    fields = (
        "static_ranges",
        "range_unroll_factors",
        "range_warp_specialize",
        "range_num_stages",
        "range_multi_buffers",
        "range_flattens",
    )
    spec = SimpleNamespace(backend_name=backend, **{name: [] for name in fields})
    queries = []
    monkeypatch.setattr(
        loops.CompileEnvironment, "current", lambda: SimpleNamespace(config_spec=spec)
    )
    monkeypatch.setattr(
        loops, "use_tileir_tunables", lambda: queries.append(1) or tileir
    )
    monkeypatch.setattr(loops, "_supports_warp_specialize", lambda: False)
    loops._add_config_range_choice([7], allow_static_range=True)
    assert queries == [1]
    assert len(spec.static_ranges) == (0 if tileir else 1)
    assert len(spec.range_unroll_factors) == (0 if tileir else 1)
    if tileir:
        assert all(not getattr(spec, field) for field in fields)
