from __future__ import annotations

from contextlib import nullcontext
import copy
import functools
import json
import operator
import random
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

from benchmarks.cute import compare_attention_backends as benchmark
import pytest

from ._cute_binding import _forbid_native_compile
from ._cute_binding import _mock_cuda_unavailable
from .test_conditional_parent_search import _fixture
from .test_owned_conditional_coordinates import _surface
import helion
from helion._compiler.cute.cute_flash import flash_structural_leaf_from_config
from helion.autotuner import LFBOPatternSearch
from helion.autotuner.base_search import PopulationMember
from helion.autotuner.config_generation import ConfigGeneration
from helion.autotuner.search_space_logger import canonical_config_id


@pytest.fixture(autouse=True)
def _no_native_compile():
    with _forbid_native_compile():
        yield


@pytest.fixture
def protected_cpu():
    with _mock_cuda_unavailable():
        yield


def _run(*, limit=6, values=(True, False), all_known=False):
    search, members, leaf, constraints = _fixture(values)
    known = {member.config for member in members}
    if all_known:
        known.add(search.config_gen.canonicalize_flat(["16/6", False, 2])[1])
    rng = random.getstate()
    try:
        random.seed(732)
        parent, children, ledger = search._flash_conditional_parent_candidates(
            members, leaf, constraints, known, limit
        )
    finally:
        random.setstate(rng)
    manifest = {canonical_config_id(config): dict(config.config) for config in known}
    manifest.update(
        {canonical_config_id(child.config): child.config.config for child in children}
    )
    kwargs = {
        "config_generation": search.config_gen,
        "ranked_parent_ids": [
            canonical_config_id(member.config)
            for member in sorted(members, key=search._flash_member_rank_key)
        ],
        "manifest_configs": manifest,
        "known_config_ids": {canonical_config_id(config) for config in known},
        "leaf": {
            "family": leaf.pipeline_family,
            "compound_packet": None,
            "softmax_disc": True,
        },
        "lane": ("cute_flash_s_stage", 2),
        "neighbor_limit": limit,
        "radius": 2,
        "selected_parent_id": None
        if parent is None
        else canonical_config_id(parent.config),
        "generated_ids": [canonical_config_id(child.config) for child in children],
        "trial_index": 1,
        "template_owned_coordinates": True,
    }
    return search, parent, children, json.loads(json.dumps(ledger)), kwargs


def test_saturated_best_parent_falls_back_with_one_shared_budget():
    search, parent, children, ledger, kwargs = _run()
    assert parent is not None and parent.config["cute_flash_e2e_schedule"] == "16/6"
    assert len(children) == 1
    assert children[0].config["cute_flash_packed_reduce"] is False
    assert [attempt["consumed"] for attempt in ledger["attempts"]] == [2, 2]
    assert ledger["consumed"] == 4
    assert sum(item["limit"] for item in ledger["allocations"]) == 6
    assert ledger["attempts"][0]["novel_config_ids"] == []
    assert ledger["attempts"][1]["novel_config_ids"] == kwargs["generated_ids"]
    search._surrogate_select.assert_called_once()
    rng = random.getstate()
    benchmark._validate_flash_conditional_parent_search(ledger, **kwargs)
    assert random.getstate() == rng


@pytest.mark.parametrize(
    "mutation",
    [
        "drop_draw",
        "increase_budget",
        "bad_allocation",
        "skip_parent",
        "unknown_set",
        "hide_known",
        "invent_child",
        "change_coordinate",
        "change_random",
        "change_rng",
        "change_normalized",
        "wrong_parent",
        "missing_attempt",
    ],
)
def test_strict_conditional_ledger_rejects_tampering(mutation):
    _search, _parent, _children, ledger, kwargs = _run()
    first = ledger["attempts"][0]
    last = ledger["attempts"][-1]
    if mutation == "drop_draw":
        first["proposals"].pop()
    elif mutation == "increase_budget":
        ledger["neighbor_limit"] += 1
    elif mutation == "bad_allocation":
        ledger["allocations"][0]["limit"] += 1
    elif mutation == "skip_parent":
        ledger["attempts"] = [last]
    elif mutation == "unknown_set":
        ledger["known_config_ids"].pop()
    elif mutation == "hide_known":
        first["proposals"][0]["outcome"] = "invalid"
    elif mutation == "invent_child":
        kwargs["generated_ids"] = ["f" * 16]
    elif mutation == "change_coordinate":
        first["proposals"][0]["raw_flat_values"][2] = 1
    elif mutation == "change_random":
        first["proposals"][1]["raw_flat_values"][0] = "16/10"
    elif mutation == "change_rng":
        first["random_state"][1][-1] = 625
    elif mutation == "change_normalized":
        first["proposals"][0]["config"]["cute_flash_s_stage"] = 1
    elif mutation == "wrong_parent":
        kwargs["selected_parent_id"] = kwargs["ranked_parent_ids"][0]
    elif mutation == "missing_attempt":
        ledger["attempts"] = []
    with pytest.raises(RuntimeError, match="invalid conditional parent search"):
        benchmark._validate_flash_conditional_parent_search(ledger, **kwargs)


def test_strict_replay_rejects_another_valid_rng_state():
    _search, _parent, _children, ledger, kwargs = _run()
    attempt = ledger["attempts"][0]
    actual_raw = attempt["proposals"][1]["raw_flat_values"]
    generation = kwargs["config_generation"]
    base = generation.flatten(
        helion.Config.from_dict(kwargs["manifest_configs"][attempt["parent_config_id"]])
    )
    for seed in range(100):
        state = json.loads(json.dumps(random.Random(seed).getstate()))
        proposed = benchmark._replay_flash_conditional_random_proposals(
            generation,
            base,
            radius=2,
            count=1,
            raw_state=state,
            fail=lambda detail: pytest.fail(detail),
        )
        if proposed[0] != actual_raw:
            attempt["random_state"] = state
            break
    else:
        pytest.fail("fixture did not exercise a different valid RNG draw")
    with pytest.raises(RuntimeError, match="random proposal replay mismatch"):
        benchmark._validate_flash_conditional_parent_search(ledger, **kwargs)


def test_strict_replay_rejects_continuation_after_a_productive_parent():
    _search, _parent, _children, ledger, kwargs = _run()
    ledger["attempts"].append(copy.deepcopy(ledger["attempts"][-1]))
    with pytest.raises(RuntimeError, match="skipped a productive parent"):
        benchmark._validate_flash_conditional_parent_search(ledger, **kwargs)


def test_fresh_outer_provenance_rejects_legacy_policy():
    from test import test_benchmarking as fixtures

    assert benchmark._CUTE_FLASH_LANE_POLICY_VERSION == 16
    with pytest.raises(
        SystemExit, match="terminal coordinate refinement policy is inconsistent"
    ):
        benchmark._validate_required_full_autotune(
            fixtures._full_autotune_trial_provenance()
        )


@functools.lru_cache(maxsize=2)
def _full_phase_fixture(*, template_owned_coordinates: bool):
    """Produce current or legacy proposals for fabricated full-phase measurements."""
    from test import test_benchmarking as fixtures

    trial = fixtures._full_autotune_trial()
    provenance = fixtures._full_autotune_trial_provenance()
    phase = trial["search_phase_metrics"]
    generation = fixtures._fresh_full_autotune_config_generation()
    search = LFBOPatternSearch.__new__(LFBOPatternSearch)
    search.config_spec = generation.config_spec
    search.config_gen = generation
    search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
    search.radius = 2
    search.num_neighbors_cap = -1
    search._surrogate_select = lambda candidates, count: candidates[:count]
    manifest = phase["config_manifest"]
    known = {
        helion.Config.from_dict(manifest[key]["config"])
        for key in phase["initial_config_ids"]
    }
    replacements = {}
    child_configs = {}
    ledgers = []
    parent_ids = []
    rng = random.getstate()
    try:
        random.seed(9182)
        for decision in phase["leaf_results"][0]["rounds"][1]["parent_decisions"]:
            members = [
                PopulationMember(
                    Mock(),
                    [record["selection_perf"]],
                    generation.flatten(
                        helion.Config.from_dict(manifest[record["config_id"]]["config"])
                    ),
                    helion.Config.from_dict(manifest[record["config_id"]]["config"]),
                    status=record["status"],
                )
                for record in decision["candidate_results"]
            ]
            leaf = flash_structural_leaf_from_config(members[0].config.config)
            lane = decision["pipeline_lane"]
            # v23 used the unfiltered iterator even for resident parents. Select
            # that producer behavior before generating proposals, rather than
            # relabeling a current filtered ledger as a legacy one afterward.
            with (
                nullcontext()
                if template_owned_coordinates
                else patch.object(
                    ConfigGeneration, "flash_owned_coordinate_indices", return_value=[]
                )
            ):
                parent, children, ledger = search._flash_conditional_parent_candidates(
                    members,
                    leaf,
                    (
                        *search._flash_leaf_constraints(leaf),
                        (lane["key"], lane["value"]),
                    ),
                    known,
                    100,
                )
            assert parent is not None and len(children) == 1
            child = children[0].config
            child_id = canonical_config_id(child)
            replacements[decision["generated_config_ids"][0]] = child_id
            child_configs[child_id] = dict(child.config)
            parent_ids.append(canonical_config_id(parent.config))
            ledgers.append(json.loads(json.dumps(ledger)))
            if not template_owned_coordinates:
                ledgers[-1]["schema_version"] = 1
                for attempt in ledgers[-1]["attempts"]:
                    assert attempt.pop("owned_coordinate_indices") == []
            known.add(child)
    finally:
        random.setstate(rng)

    def replace_ids(value):
        if isinstance(value, dict):
            return {
                replacements.get(key, key): replace_ids(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [replace_ids(item) for item in value]
        return replacements.get(value, value) if isinstance(value, str) else value

    phase = replace_ids(phase)
    trial["search_phase_metrics"] = phase
    for event in phase["measurement_timeline"]:
        event["updates"].sort(key=operator.itemgetter("config_id"))
    for child_id, config in child_configs.items():
        phase["config_manifest"][child_id]["config"] = config
    for decision, parent_id, ledger in zip(
        phase["leaf_results"][0]["rounds"][1]["parent_decisions"],
        parent_ids,
        ledgers,
        strict=True,
    ):
        decision["selected_config_id"] = parent_id
        decision["conditional_parent_search"] = ledger
    phase["phase"] = (
        "cute_flash_structural_qualification_v24"
        if template_owned_coordinates
        else "cute_flash_structural_qualification_v23"
    )
    phase["cute_flash_lane_policy_version"] = 16 if template_owned_coordinates else 15
    policy = provenance["flash_terminal_coordinate_refinement_policy"]
    policy["lane_policy_version"] = phase["cute_flash_lane_policy_version"]
    provenance["flash_terminal_coordinate_refinement_policy_sha256"] = (
        benchmark._canonical_json_sha256(policy)
    )
    return provenance, phase, generation, trial


def _validate_v23_phase(provenance, phase, generation, trial):
    from test import test_benchmarking as fixtures

    fixture_generation = fixtures._FixtureConfigGeneration(
        generation, provenance, trial
    )
    benchmark._validate_flash_structural_qualification_phase(
        provenance,
        phase,
        trial_index=1,
        expected_initial_config_ids=phase["initial_config_ids"],
        expected_initial_population_count=100,
        config_generation=fixture_generation,
    )


def _resident_attempt():
    spec, generation, base, parent = _surface()
    search = LFBOPatternSearch.__new__(LFBOPatternSearch)
    search.config_spec = spec
    search.config_gen = spec.create_config_generation()
    search.kernel = SimpleNamespace(env=SimpleNamespace(process_group_name=None))
    search.radius = 2
    search.num_neighbors_cap = -1
    search._surrogate_select = lambda candidates, count: candidates[:count]
    flat, global_parent = search.config_gen.canonicalize_flat(
        search.config_gen.flatten(parent)
    )
    member = PopulationMember(Mock(), [1.0], flat, global_parent, status="ok")
    leaf = flash_structural_leaf_from_config(parent.config)
    constraints = (("cute_flash_kv_stage", 2),)
    selected, children, ledger = search._flash_conditional_parent_candidates(
        [member], leaf, constraints, {global_parent}, 12
    )
    manifest = {canonical_config_id(global_parent): dict(global_parent.config)}
    manifest.update(
        {
            canonical_config_id(child.config): dict(child.config.config)
            for child in children
        }
    )
    kwargs = {
        "config_generation": search.config_gen,
        "ranked_parent_ids": [canonical_config_id(global_parent)],
        "manifest_configs": manifest,
        "known_config_ids": {canonical_config_id(global_parent)},
        "leaf": {"family": "fa4", "compound_packet": None, "softmax_disc": False},
        "lane": ("cute_flash_kv_stage", 2),
        "neighbor_limit": 12,
        "radius": 2,
        "selected_parent_id": canonical_config_id(selected.config),
        "generated_ids": [canonical_config_id(child.config) for child in children],
        "trial_index": 1,
        "template_owned_coordinates": True,
    }
    return generation, base, global_parent, children, ledger, kwargs


def test_resident_owned_prefix_yields_a_child_without_extra_raw_proposals(
    protected_cpu,
):
    generation, base, parent, children, ledger, kwargs = _resident_attempt()
    old = generation.coordinate_neighbor_projections(base, radius=2, limit=12)
    assert len(old) == 12 and all(item.outcome == "incumbent_alias" for item in old)
    assert len(children) == 1 and children[0].config != parent
    assert ledger["schema_version"] == 2
    assert ledger["consumed"] == 12
    assert ledger["allocations"] == [
        {"parent_config_id": canonical_config_id(parent), "limit": 12}
    ]
    assert len(ledger["attempts"][0]["proposals"]) == 12
    assert ledger["attempts"][0]["owned_coordinate_indices"]
    benchmark._validate_flash_conditional_parent_search(ledger, **kwargs)


@pytest.mark.parametrize(
    "mutation", ("drop", "extra", "reorder", "boolean", "mode", "schema")
)
def test_owned_mask_is_derived_and_not_trusted(mutation, protected_cpu):
    _generation, _base, _parent, _children, ledger, kwargs = _resident_attempt()
    ledger = copy.deepcopy(ledger)
    owned = ledger["attempts"][0]["owned_coordinate_indices"]
    if mutation == "drop":
        owned.pop()
    elif mutation == "extra":
        owned.append(999)
    elif mutation == "reorder":
        owned.reverse()
    elif mutation == "boolean":
        owned[0] = True
    elif mutation == "mode":
        kwargs["manifest_configs"][kwargs["ranked_parent_ids"][0]][
            "cute_flash_softmax_lowering"
        ] = "standard"
    elif mutation == "schema":
        ledger["schema_version"] = 1
    with pytest.raises(RuntimeError, match="invalid conditional parent search"):
        benchmark._validate_flash_conditional_parent_search(ledger, **kwargs)


@pytest.mark.parametrize(
    "mutation", (None, "quota", "retry", "budget", "incomplete", "policy", "old_schema")
)
def test_v24_full_phase_preserves_strict_shared_contract(mutation):
    provenance, original_phase, generation, original_trial = _full_phase_fixture(
        template_owned_coordinates=True
    )
    provenance = copy.deepcopy(provenance)
    phase = copy.deepcopy(original_phase)
    trial = copy.deepcopy(original_trial)
    trial["search_phase_metrics"] = phase
    assert phase["phase"] == "cute_flash_structural_qualification_v24"
    assert phase["cute_flash_lane_policy_version"] == 16
    search = LFBOPatternSearch.__new__(LFBOPatternSearch)
    search.config_spec = generation.config_spec
    search.config_gen = generation
    ledgers = []
    for leaf in phase["leaf_results"]:
        for round_data in leaf["rounds"]:
            for decision in round_data["parent_decisions"]:
                if "conditional_parent_search" not in decision:
                    continue
                ledger = decision["conditional_parent_search"]
                ledgers.append(ledger)
                assert ledger["schema_version"] == 2
                for attempt in ledger["attempts"]:
                    parent = helion.Config.from_dict(
                        phase["config_manifest"][attempt["parent_config_id"]]["config"]
                    )
                    parent_leaf = flash_structural_leaf_from_config(parent.config)
                    lane = decision["pipeline_lane"]
                    conditional_generation = search._flash_leaf_config_generation(
                        parent_leaf,
                        (
                            *search._flash_leaf_constraints(parent_leaf),
                            (lane["key"], lane["value"]),
                        ),
                    )
                    assert conditional_generation is not None
                    assert attempt["owned_coordinate_indices"] == (
                        conditional_generation.flash_owned_coordinate_indices(parent)
                    )
    if mutation == "quota":
        phase["conditional_candidates_per_pipeline_lane"] = 0
    elif mutation == "retry":
        phase["qualification_failure_retries"] = 0
    elif mutation == "budget":
        phase["budget_exhausted"] = True
    elif mutation == "incomplete":
        phase["completed"] = False
    elif mutation == "policy":
        phase["cute_flash_lane_policy_version"] = 15
    elif mutation == "old_schema":
        ledgers[0]["schema_version"] = 1
    if mutation is None:
        _validate_v23_phase(provenance, phase, generation, trial)
    else:
        with pytest.raises(RuntimeError):
            _validate_v23_phase(provenance, phase, generation, trial)
