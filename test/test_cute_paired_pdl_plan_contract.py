"""Unbound builder contract: never infer unknown CLC, query bundle only if needed."""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from helion import exc
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.compile_environment import NoCurrentEnvironment
from helion._compiler.cute.device_state import CuteTcgen05MatmulPlan
from helion._compiler.cute.strategies import Tcgen05PersistenceModel
from helion._compiler.cute.tcgen05_lifecycle import tcgen05_launch_uses_pdl
from helion._compiler.program_id import Tcgen05PersistentProgramIDs


def splitter(plan):
    value = Tcgen05PersistentProgramIDs.__new__(Tcgen05PersistentProgramIDs)
    value._tcgen05_plan = lambda: plan
    return value


@pytest.mark.parametrize(
    "pdl,materialized,clc,fission", tuple(itertools.product((False, True), repeat=4))
)
def test_pair_builder_queries_only_relevant_bundle_fact_and_matches_launch_contract(
    pdl, materialized, clc, fission
):
    config = {"cute_pdl": pdl, "tcgen05_materialized_pdl": materialized}
    helper = splitter(SimpleNamespace(is_clc_persistent=clc))
    env = SimpleNamespace(cute_fission_plan=object() if fission else None)
    with patch.object(CompileEnvironment, "current", return_value=env) as current:
        value = helper._tcgen05_pair_launch_uses_pdl(SimpleNamespace(config=config))
    assert value == tcgen05_launch_uses_pdl(
        config, clc_persistent=clc, fission_bundle=fission
    )
    assert current.call_count == int(pdl and not materialized and not clc)


@pytest.mark.parametrize(
    "config,clc,want",
    [
        ({}, False, False),
        ({"cute_pdl": False}, False, False),
        ({"cute_pdl": False}, True, True),
        ({"cute_pdl": True}, True, True),
        ({"tcgen05_materialized_pdl": True}, False, True),
        ({"cute_pdl": True, "tcgen05_materialized_pdl": True}, False, True),
    ],
)
def test_unbound_plain_forced_CLC_or_handoff_does_not_fabricate_environment(
    config, clc, want
):
    helper = splitter(SimpleNamespace(is_clc_persistent=clc))
    with patch.object(
        CompileEnvironment, "current", side_effect=NoCurrentEnvironment
    ) as current:
        assert (
            helper._tcgen05_pair_launch_uses_pdl(SimpleNamespace(config=config)) is want
        )
    current.assert_not_called()


def test_general_PDL_still_requires_actual_bundle_environment():
    helper = splitter(SimpleNamespace(is_clc_persistent=False))
    with (
        patch.object(CompileEnvironment, "current", side_effect=NoCurrentEnvironment),
        pytest.raises(NoCurrentEnvironment),
    ):
        helper._tcgen05_pair_launch_uses_pdl(SimpleNamespace(config={"cute_pdl": True}))


@pytest.mark.parametrize(
    "config,clc",
    [
        ({"cute_pdl": 1}, False),
        ({"cute_pdl": "false"}, False),
        ({"tcgen05_materialized_pdl": 1}, False),
        ({"cute_pdl": True, "tcgen05_materialized_pdl": "false"}, False),
        ({}, None),
        ({}, 1),
    ],
)
def test_invalid_actual_launch_flags_remain_fail_closed_without_environment(
    config, clc
):
    helper = splitter(SimpleNamespace(is_clc_persistent=clc))
    with (
        patch.object(CompileEnvironment, "current", side_effect=NoCurrentEnvironment),
        pytest.raises(exc.InvalidConfig, match="proved booleans"),
    ):
        helper._tcgen05_pair_launch_uses_pdl(SimpleNamespace(config=config))


def test_missing_or_partial_plan_never_defaults_unknown_CLC_to_false():
    with pytest.raises(exc.BackendUnsupported, match="requires a tcgen05 launch plan"):
        splitter(None)._tcgen05_pair_launch_uses_pdl(SimpleNamespace(config={}))
    with pytest.raises(AttributeError, match="is_clc_persistent"):
        splitter(SimpleNamespace())._tcgen05_pair_launch_uses_pdl(
            SimpleNamespace(config={})
        )


@pytest.mark.parametrize(
    "model,want",
    [
        (Tcgen05PersistenceModel.STATIC_PERSISTENT.value, False),
        (Tcgen05PersistenceModel.CLC_PERSISTENT.value, True),
    ],
)
def test_fixture_CLC_fact_corresponds_to_real_plan_property(model, want):
    plan = CuteTcgen05MatmulPlan(
        bm=128,
        bn=128,
        bk=64,
        k_tile_count=16,
        cluster_m=1,
        is_two_cta=False,
        uses_role_local_persistent_body=True,
        uses_cluster_m2_one_cta_role_local_bridge=False,
        cta_thread_count=192,
        physical_m_threads=128,
        acc_stage_count=2,
        ab_stage_count=2,
        c_stage_count=2,
        epi_warp_count=4,
        persistence_model=model,
    )
    assert plan.is_clc_persistent is want
