"""Field-layout phase stability of ``tcgen05_fanout_store_commit``.

The epilogue-fanout preparation pass runs before the first coverage group records
its field fingerprint. The registration pass then re-proves the fanout plans. The
store-commit search field is latched by preparation, so that re-proof can neither
remove the field (late proof loss) nor add it (late proof recovery). A value is
still admitted only under the current proof.

The phase boundary is observed, not simulated: ``register_cluster_coverage`` is the
first registration call after preparation, so the test records the flat field layout
on entry to it. The proof is withheld in one pass only. The real proof function runs
in both passes, and nothing is fabricated or unbound.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from test.test_cute_shared_dag_store_commit import KEY
from test.test_cute_shared_dag_store_commit import _bound_external
from test.test_cute_shared_dag_store_commit import _config
from test.test_cute_shared_dag_store_commit import _loop_lines
from test.test_cute_shared_dag_store_commit import check_grouped_protocol

import helion
from helion._compiler import autotuner_heuristics as heuristics
from helion._compiler.autotuner_heuristics import cute_epilogue_fanout as coverage
from helion._compiler.cute.epilogue_fanout import FANOUT_CONFIG_KEY
from helion._compiler.cute.shared_epilogue_dag import SharedDagFanoutPlan
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion.exc import InvalidConfig


@pytest.fixture(autouse=True)
def cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def _phase_bind(count: int, *, prepared: bool, registered: bool):
    """Bind with the real proof kept or withheld in the preparation/registration pass."""
    original_prove = coverage.prove_epilogue_fanout
    original_cluster = heuristics.register_cluster_coverage
    calls: list[str] = []
    boundary: list[tuple[str, ...]] = []

    def prove(*args, **kwargs):
        phase = "prepare" if not boundary else "register"
        calls.append(phase)
        result = original_prove(*args, **kwargs)
        keep = prepared if phase == "prepare" else registered
        return result if keep else None

    def cluster(env, device_ir):
        boundary.append(tuple(env.config_spec._flat_fields()))
        return original_cluster(env, device_ir)

    with (
        patch.object(coverage, "prove_epilogue_fanout", prove),
        patch.object(heuristics, "register_cluster_coverage", cluster),
    ):
        bound = _bound_external(count)
    assert calls == ["prepare", "register"], calls
    assert len(boundary) == 1
    return bound, boundary[0]


def _final_fields(bound) -> tuple[str, ...]:
    with bound.env:
        bound.config_spec.validate_compiler_coverage_groups()
        return tuple(bound.config_spec._flat_fields())


def _unowned(bound, fields: tuple[str, ...]) -> list[str]:
    """The layout the coverage registry holds fixed: fields no LATER group owns.

    A later group's own field (``cute_min_blocks_per_mp``) is legitimately added
    after the boundary; the registry compares every other row. The store-commit
    field is kept in the comparison: it must be fixed by the boundary.
    """
    owned = {group.key for group in bound.config_spec.compiler_coverage_groups} - {KEY}
    return [name for name in fields if name not in owned]


@pytest.mark.parametrize("count", (2, 4))
def test_proved_in_both_passes_publishes_the_field_at_the_boundary(count) -> None:
    bound, at_boundary = _phase_bind(count, prepared=True, registered=True)
    state = bound.config_spec._cute_tcgen05_config
    assert state.shared_dag_store_commit_search_enabled
    assert KEY in at_boundary
    assert _unowned(bound, _final_fields(bound)) == _unowned(bound, at_boundary)
    assert any(g.key == KEY for g in bound.config_spec.compiler_coverage_groups)


@pytest.mark.parametrize("count", (2, 4))
def test_late_proof_loss_keeps_the_published_layout_and_refuses_loudly(count) -> None:
    bound, at_boundary = _phase_bind(count, prepared=True, registered=False)
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    # The re-proof lost every plan; the latched field stays where it was
    # published, so the layout the first fingerprint records cannot move.
    assert state.epilogue_fanout_plans == ()
    assert state.shared_dag_store_commit_search_enabled
    assert KEY in at_boundary
    assert _unowned(bound, _final_fields(bound)) == _unowned(bound, at_boundary)
    if spec._compiler_coverage_groups:
        assert KEY in [row[0] for row in spec._compiler_coverage_fields]
    # No witness can be registered without the current proof.
    assert all(g.key != KEY for g in spec.compiler_coverage_groups)
    # The current proof is lost: grouped is refused loudly, never served serially.
    grouped = _config("grouped")
    assert state.fanout_store_commit_refusal(grouped.config) is not None
    with pytest.raises(InvalidConfig):
        spec.normalized_config(grouped)
    with pytest.raises(InvalidConfig):
        bound.to_code(grouped)
    alone = helion.Config.from_dict(
        {k: v for k, v in grouped.config.items() if k != FANOUT_CONFIG_KEY}
    )
    with pytest.raises(InvalidConfig, match=KEY):
        spec.normalized_config(alone)
    # The neutral value stays implicit on the stable surface.
    serial = helion.Config.from_dict(dict(alone.config) | {KEY: "serial"})
    assert KEY not in spec.normalized_config(serial).config


@pytest.mark.parametrize("count", (2, 4))
def test_late_proof_recovery_cannot_append_the_search_field(count) -> None:
    bound, at_boundary = _phase_bind(count, prepared=False, registered=True)
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    # The registration pass recovered a shared-DAG plan the preparation lacked.
    assert any(isinstance(p, SharedDagFanoutPlan) for p in state.epilogue_fanout_plans)
    assert not state.shared_dag_store_commit_search_enabled
    assert KEY not in at_boundary
    assert KEY not in _final_fields(bound)
    if spec._compiler_coverage_groups:
        assert KEY not in [row[0] for row in spec._compiler_coverage_fields]
    assert all(g.key != KEY for g in spec.compiler_coverage_groups)
    with bound.env:
        population = spec.create_config_generation().initial_population_view()
        assert all(KEY not in c.config for _f, c in population.seed_flat_config_pairs())
    # An explicit request is still judged by the current proof alone (the
    # validation surface), exactly as the fanout key's own explicit values are.
    grouped = _config("grouped")
    assert state.fanout_store_commit_refusal(grouped.config) is None
    assert spec.normalized_config(grouped).config[KEY] == "grouped"
    check_grouped_protocol(_loop_lines(bound.to_code(grouped)), count)
