"""Field-layout phase stability of ``tcgen05_fanout_store_commit_group_size``.

The bounded commit group size is latched by the epilogue-fanout preparation pass,
before the first coverage group records its field fingerprint, under the same rule
as the store commit mode. The registration pass's re-proof can neither remove the
field (late proof loss) nor add it (late proof recovery); a bounded size is still
admitted only under the current proof. The phase boundary is observed on entry to
``register_cluster_coverage``; the real proof runs in both passes and is withheld
in one only.
"""

from __future__ import annotations

import pytest

from test.test_cute_shared_dag_store_commit import KEY
from test.test_cute_shared_dag_store_commit import _config
from test.test_cute_shared_dag_store_commit import _loop_lines
from test.test_cute_shared_dag_store_commit_groups import GROUP
from test.test_cute_shared_dag_store_commit_groups import _sized
from test.test_cute_shared_dag_store_commit_groups import check_bounded_protocol
from test.test_cute_shared_dag_store_commit_phases import _final_fields
from test.test_cute_shared_dag_store_commit_phases import _phase_bind

import helion
from helion._compiler.cute.epilogue_fanout import FANOUT_CONFIG_KEY
from helion._compiler.cute.shared_epilogue_dag import SharedDagFanoutPlan
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion.exc import InvalidConfig


@pytest.fixture(autouse=True)
def cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def _unowned(bound, fields: tuple[str, ...]) -> list[str]:
    """Fields no LATER group owns; both store-commit fields stay in the comparison."""
    owned = {group.key for group in bound.config_spec.compiler_coverage_groups} - {
        KEY,
        GROUP,
    }
    return [name for name in fields if name not in owned]


@pytest.mark.parametrize("count", (3, 4))
def test_proved_in_both_passes_publishes_the_size_at_the_boundary(count) -> None:
    bound, at_boundary = _phase_bind(count, prepared=True, registered=True)
    state = bound.config_spec._cute_tcgen05_config
    assert state.shared_dag_store_commit_group_search_enabled
    assert list(at_boundary).index(KEY) + 1 == list(at_boundary).index(GROUP)
    assert _unowned(bound, _final_fields(bound)) == _unowned(bound, at_boundary)
    assert any(g.key == GROUP for g in bound.config_spec.compiler_coverage_groups)


@pytest.mark.parametrize("count", (3, 4))
def test_late_proof_loss_keeps_the_layout_and_refuses_a_bounded_size(count) -> None:
    bound, at_boundary = _phase_bind(count, prepared=True, registered=False)
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert state.epilogue_fanout_plans == ()
    assert state.shared_dag_store_commit_group_search_enabled
    assert GROUP in at_boundary
    assert _unowned(bound, _final_fields(bound)) == _unowned(bound, at_boundary)
    if spec._compiler_coverage_groups:
        assert GROUP in [row[0] for row in spec._compiler_coverage_fields]
    assert all(g.key != GROUP for g in spec.compiler_coverage_groups)
    bounded = _sized(2)
    assert state.fanout_store_commit_group_size_refusal(bounded.config) is not None
    with pytest.raises(InvalidConfig):
        spec.normalized_config(bounded)
    with pytest.raises(InvalidConfig):
        bound.to_code(bounded)
    alone = helion.Config.from_dict(
        {k: v for k, v in bounded.config.items() if k not in (FANOUT_CONFIG_KEY, KEY)}
    )
    with pytest.raises(InvalidConfig, match=GROUP):
        spec.normalized_config(alone)
    # The neutral value stays implicit on the stable surface.
    neutral = helion.Config.from_dict(dict(alone.config) | {GROUP: "all"})
    assert GROUP not in spec.normalized_config(neutral).config


@pytest.mark.parametrize("count", (3, 4))
def test_late_proof_recovery_cannot_append_the_size_field(count) -> None:
    bound, at_boundary = _phase_bind(count, prepared=False, registered=True)
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert any(isinstance(p, SharedDagFanoutPlan) for p in state.epilogue_fanout_plans)
    assert not state.shared_dag_store_commit_group_search_enabled
    assert GROUP not in at_boundary
    assert GROUP not in _final_fields(bound)
    if spec._compiler_coverage_groups:
        assert GROUP not in [row[0] for row in spec._compiler_coverage_fields]
    assert all(g.key != GROUP for g in spec.compiler_coverage_groups)
    with bound.env:
        population = spec.create_config_generation().initial_population_view()
        assert all(
            GROUP not in c.config for _f, c in population.seed_flat_config_pairs()
        )
    # An explicit request is judged by the current proof alone, like the
    # store commit mode's own explicit values after a recovery.
    bounded = _sized(2)
    assert state.fanout_store_commit_group_size_refusal(bounded.config) is None
    assert spec.normalized_config(bounded).config[GROUP] == 2
    check_bounded_protocol(_loop_lines(bound.to_code(bounded)), count)
    assert KEY in _config("grouped").config
