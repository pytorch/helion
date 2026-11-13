"""The graph-free metadata transport choice never supplies its own proof."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import dataclasses
import functools

import pytest
import torch

import helion
from helion._compiler.cute.autotune.mma_dag_surface import enable_cute_mma_dag_search
from helion._compiler.cute.cute_mma_dag import MMA_DAG_V_LAG_CHOICES
from helion._compiler.cute.cute_mma_dag import (
    MMA_DAG_VALUE_METADATA_READ_OVERLAP_KEY as OVERLAP,
)
from helion._compiler.cute.cute_mma_dag import MMA_DAG_VALUE_METADATA_SKIP_KEY as KEY
from helion._compiler.cute.cute_mma_dag_seeds import MmaDagSeedPlanner
from helion._compiler.cute.paged_attention_config import paged_attention_config_reason
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_optional_lse import optional_lse_fixture
from helion._testing.cute_paged_positive_zero import positive_zero_attention_fixture
from helion._testing.cute_paged_v_lag import paired_v_lag_bound
from helion._testing.cute_paged_v_lag import paired_v_lag_config
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@functools.cache
def _mapped_bound():
    kernel, args = optional_lse_fixture("mapped", True, query_len=5)
    with cpu_render_target(fake_device_sm_count=148):
        return cpu_bind(kernel, args)


@contextmanager
def _surface(*, available=True, paired=True, tma_policies=True):
    bound = _mapped_bound()
    assert bound.host_function is not None
    spec = copy.copy(bound.config_spec)
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    # This unit consumes a transport-owned proof. Supply only its graph-free
    # interface here; the separate transport integration supplies eligibility.
    facts = dataclasses.replace(
        facts,
        attention=dataclasses.replace(
            facts.attention,
            paired_items=paired,
            value_metadata_skip_available=available,
            tma_load_slots=(
                facts.attention.tma_load_slots
                if tma_policies
                else (None,) * len(facts.attention.tma_load_slots)
            ),
        ),
    )
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        enable_cute_mma_dag_search(spec, facts)
        yield bound, spec


def _config(value=False, *, family="attention_paired"):
    config = paired_v_lag_config(0)
    config.config.update({KEY: value, "cute_mma_dag_pass_warps": family})
    return config


def _assert_idempotent(spec, config, *, repair=False):
    again = helion.Config.from_dict(config.config)
    spec.normalize(again, _fix_invalid=repair)
    assert again == config


@pytest.mark.parametrize("value", [False, True])
def test_metadata_fragment_roundtrip_and_neighbors(value):
    with _surface() as (bound, spec):
        fragment = spec.cute_mma_dag_value_metadata_skip
        assert fragment is not None
        assert spec._flat_fields()[KEY] is fragment
        assert fragment.choices == (False, True)
        assert fragment.default() is False
        config = spec.normalized_config(_config(value))
        assert config.get(KEY) is value
        _assert_idempotent(spec, config)
        assert bound.env.backend.autotune_config_is_viable(spec, config)
        generation = spec.create_config_generation()
        flat, _ = generation.canonicalize_flat(generation.flatten(config))
        neighbors = generation.coordinate_neighbor_projections(flat)
        move = next(x for x in neighbors if x.key == KEY)
        assert move.outcome == "candidate"
        assert move.config.get(KEY) is not value


@pytest.mark.parametrize("value", [0, 1, 0.0, "True", None])
def test_metadata_requires_exact_bool_and_repairs_to_default(value):
    with _surface() as (bound, spec):
        config = _config(value)
        with pytest.raises(helion.exc.InvalidConfig, match=KEY):
            spec.normalized_config(config)
        assert not bound.env.backend.autotune_config_is_viable(spec, config)
        spec.normalize(config, _fix_invalid=True)
        assert config.get(KEY) is False
        _assert_idempotent(spec, config, repair=True)


@pytest.mark.parametrize("available,paired", [(False, True), (True, False)])
def test_unavailable_transport_rejects_true_but_repairs_to_default(available, paired):
    with _surface(available=available, paired=paired) as (_, spec):
        assert spec.cute_mma_dag_value_metadata_skip is None
        assert KEY not in spec._flat_fields()
        config = _config(True, family="attention")
        with pytest.raises(helion.exc.InvalidConfig, match=KEY):
            spec.normalized_config(config)
        spec.normalize(config, _fix_invalid=True)
        assert config.get(KEY, False) is False
        default = _config(False, family="attention")
        assert spec.normalized_config(default) == config
        _assert_idempotent(spec, config)


def test_ordinary_source_without_positive_zero_has_no_metadata_fragment():
    bound = paired_v_lag_bound()
    assert bound.host_function is not None
    spec = bound.config_spec
    assert spec.cute_mma_dag_facts is not None
    assert spec.cute_mma_dag_facts.attention is not None
    assert not spec.cute_mma_dag_facts.attention.value_metadata_skip_available
    assert spec.cute_mma_dag_value_metadata_skip is None
    with bound.env, bound.host_function:
        config = _config(True)
        with pytest.raises(helion.exc.InvalidConfig, match=KEY):
            spec.normalized_config(config)
        spec.normalize(config, _fix_invalid=True)
        assert config == spec.normalized_config(_config(False))
        _assert_idempotent(spec, config)


@pytest.mark.parametrize(
    "lost_proof",
    [
        "attention",
        "paired_items",
        "reference_scores",
        "query_tma_available",
        "value_metadata_skip_available",
    ],
)
def test_reregistered_facts_remove_stale_paired_fragments(lost_proof):
    keys = (
        "cute_mma_dag_v_lag",
        "cute_mma_dag_sleep_waits",
        "cute_mma_dag_query_tma",
        "cute_mma_dag_probability_schedule",
        KEY,
        OVERLAP,
    )
    with _surface() as (_, spec):
        facts = spec.cute_mma_dag_facts
        assert facts is not None and facts.attention is not None
        facts = dataclasses.replace(
            facts,
            attention=dataclasses.replace(
                facts.attention,
                reference_scores=True,
                query_tma_available=True,
            ),
        )
        enable_cute_mma_dag_search(spec, facts)
        assert all(getattr(spec, key) is not None for key in keys)
        original_choices = {key: getattr(spec, key).choices for key in keys}
        assert set(keys) <= spec._flat_fields().keys()

        attention = (
            None
            if lost_proof == "attention"
            else dataclasses.replace(facts.attention, **{lost_proof: False})
        )
        reduced = dataclasses.replace(facts, attention=attention)
        removed = (
            set(keys)
            if lost_proof in ("attention", "paired_items")
            else set(
                {
                    "reference_scores": ("cute_mma_dag_probability_schedule",),
                    "query_tma_available": ("cute_mma_dag_query_tma",),
                    "value_metadata_skip_available": (KEY, OVERLAP),
                }[lost_proof]
            )
        )
        # Repeat registration to pin idempotence after the proof is lost.
        for _ in range(2):
            enable_cute_mma_dag_search(spec, reduced)
            for key in keys:
                assert (getattr(spec, key) is None) == (key in removed)
                assert (key not in spec._flat_fields()) == (key in removed)

        # Restoring the proof restores exactly the prior search domains.
        enable_cute_mma_dag_search(spec, facts)
        assert {key: getattr(spec, key).choices for key in keys} == original_choices


@pytest.mark.parametrize("mode,family", [("tcgen05", "attention"), ("off", "off")])
def test_nonpaired_and_off_paths_pin_metadata_false(mode, family):
    with _surface() as (_, spec):
        config = _config(True, family=family)
        config.config["cute_mma_dag"] = mode
        normalized = spec.normalized_config(config)
        assert normalized.get(KEY) is False
        _assert_idempotent(spec, normalized)


def test_failed_mapped_repair_clears_metadata_before_early_return():
    with _surface() as (_, spec):
        config = _config(True)
        config.config["cute_min_blocks_per_mp"] = 2
        spec.normalize(config, _fix_invalid=True)
        assert config.get("cute_mma_dag") == "off"
        assert config.get("cute_mma_dag_pass_warps") == "off"
        assert config.get(KEY) is False
        _assert_idempotent(spec, config, repair=True)


def test_missing_key_keeps_old_config_default():
    with _surface() as (_, spec):
        old = _config()
        old.config.pop(KEY)
        assert spec.normalized_config(old) == spec.normalized_config(_config(False))


def test_metadata_seeds_preserve_prefix_and_cover_tma_policy_anchors():
    with _surface(available=False) as (_, spec):
        facts = spec.cute_mma_dag_facts
        assert facts is not None
        planner = MmaDagSeedPlanner(facts, spec, occupancy_key="cute_min_blocks_per_mp")
        old = planner.configs(planner.ranked())
    with _surface() as (bound, spec):
        facts = spec.cute_mma_dag_facts
        assert facts is not None and facts.attention is not None
        planner = MmaDagSeedPlanner(facts, spec, occupancy_key="cute_min_blocks_per_mp")
        seeds = planner.configs(planner.ranked())
        assert seeds[: len(old)] == old
        tail = seeds[len(old) :]
        extra = [seed for seed in tail if not seed.get(OVERLAP, False)]
        overlap = [seed for seed in tail if seed.get(OVERLAP, False)]
        assert tail == [*extra, *overlap]
        assert overlap == [
            helion.Config.from_dict({**seed.config, OVERLAP: True}) for seed in extra
        ]
        for seed in overlap:
            normalized = spec.normalized_config(seed)
            assert normalized.get(OVERLAP) is True
            assert bound.env.backend.autotune_config_is_viable(spec, normalized)
        # Preserve one old metadata/default-policy seed per lag before adding
        # exactly two cache anchors per lag, without a scheduling cross product.
        lag_count = len(MMA_DAG_V_LAG_CHOICES)
        defaults = extra[:lag_count]
        assert [s.get("cute_mma_dag_v_lag", 0) for s in defaults] == list(
            MMA_DAG_V_LAG_CHOICES
        )
        assert all(not any(s.get("load_eviction_policies")) for s in defaults)
        assert all(
            helion.Config.from_dict({k: v for k, v in s.config.items() if k != KEY})
            in old
            for s in defaults
        )
        assert len(extra) == 3 * lag_count
        assert len(set(extra)) == len(extra)
        tma_slots = {s for s in facts.attention.tma_load_slots if s is not None}
        assert tma_slots
        expected_pairs = {
            (lag, policy)
            for lag in MMA_DAG_V_LAG_CHOICES
            for policy in ("", "streaming", "l2_last")
        }
        pairs = set()
        for seed in extra:
            policies = seed.get("load_eviction_policies")
            policy = policies[next(iter(tma_slots))]
            assert all(
                value == (policy if slot in tma_slots else "")
                for slot, value in enumerate(policies)
            )
            lag = seed.get("cute_mma_dag_v_lag", 0)
            pairs.add((lag, policy))
            # Cache anchors differ from the corresponding old metadata seed
            # only at its already-proved TMA load slots.
            default = defaults[list(MMA_DAG_V_LAG_CHOICES).index(lag)]
            assert seed == helion.Config.from_dict(
                {**default.config, "load_eviction_policies": policies}
            )
            assert seed.get(KEY) is True
            assert seed.get("cute_mma_dag_pass_warps") == "attention_paired"
            normalized = spec.normalized_config(seed)
            assert normalized.get(KEY) is True
            assert bound.env.backend.autotune_config_is_viable(spec, normalized)
        assert pairs == expected_pairs


@pytest.mark.parametrize("paired,tma_policies", [(False, True), (True, False)])
def test_metadata_policy_seeds_require_paired_tma_anchors(paired, tma_policies):
    with _surface(paired=paired, tma_policies=tma_policies) as (_, spec):
        facts = spec.cute_mma_dag_facts
        assert facts is not None
        planner = MmaDagSeedPlanner(facts, spec, occupancy_key="cute_min_blocks_per_mp")
        metadata = [
            s
            for s in planner.configs(planner.ranked())
            if s.get(KEY, False) and not s.get(OVERLAP, False)
        ]
        if not paired:
            assert not metadata
        else:
            assert len(metadata) == len(MMA_DAG_V_LAG_CHOICES)
            assert all(not any(s.get("load_eviction_policies")) for s in metadata)


def test_written_metadata_and_early_probability_have_joint_cache_anchors():
    kernel, args = positive_zero_attention_fixture()
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(kernel, args)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            spec = bound.config_spec
            facts = spec.cute_mma_dag_facts
            assert facts is not None and facts.attention is not None
            attention = facts.attention
            assert (
                attention.reference_scores and attention.value_metadata_skip_available
            )
            planner = MmaDagSeedPlanner(
                facts, spec, occupancy_key="cute_min_blocks_per_mp"
            )
            seeds = planner.configs(planner.ranked())
            combined = [
                seed
                for seed in seeds
                if seed.get(KEY, False)
                and seed.get("cute_mma_dag_probability_schedule", "serial") != "serial"
            ]
            # P scheduling, V lag and cache policy interact. Cover their bounded
            # CPU seed product without multiplying native kernel compilations.
            expected = {
                (schedule, lag, policy)
                for schedule in ("early", "early_tmem")
                for lag in MMA_DAG_V_LAG_CHOICES
                for policy in ("", "streaming", "l2_last")
            }
            assert len(combined) == len(expected)
            assert seeds[-len(combined) :] == combined
            slots = {slot for slot in attention.tma_load_slots if slot is not None}
            assert slots
            actual = set()
            normalized_configs = set()
            selected = None
            for seed in combined:
                policies = seed.get("load_eviction_policies")
                policy = policies[next(iter(slots))]
                assert all(
                    value == (policy if slot in slots else "")
                    for slot, value in enumerate(policies)
                )
                axes = (
                    seed.get("cute_mma_dag_probability_schedule"),
                    seed.get("cute_mma_dag_v_lag", 0),
                    policy,
                )
                actual.add(axes)
                assert seed.get("cute_mma_dag_query_tma") is True
                assert not seed.get("cute_pdl_capture_overlap", False)
                assert not seed.get(OVERLAP, False)
                normalized = spec.normalized_config(seed)
                normalized_configs.add(normalized)
                assert normalized.get(KEY) is True
                assert normalized.get("cute_pdl_capture_overlap", False) is False
                assert bound.env.backend.autotune_config_is_viable(spec, normalized)
                if axes == ("early_tmem", 3, "streaming"):
                    selected = normalized
            assert actual == expected
            assert len(normalized_configs) == len(expected)
            # Removing TMA slots keeps the probability/lag anchors but must
            # not multiply them into identical empty-policy combinations.
            no_tma_facts = dataclasses.replace(
                facts,
                attention=dataclasses.replace(
                    attention, tma_load_slots=(None,) * len(attention.tma_load_slots)
                ),
            )
            no_tma_spec = copy.copy(spec)
            enable_cute_mma_dag_search(no_tma_spec, no_tma_facts)
            no_tma_planner = MmaDagSeedPlanner(
                no_tma_facts, no_tma_spec, occupancy_key="cute_min_blocks_per_mp"
            )
            no_tma_seeds = no_tma_planner.configs(no_tma_planner.ranked())
            no_tma_combined = [
                seed
                for seed in no_tma_seeds
                if seed.get(KEY, False)
                and seed.get("cute_mma_dag_probability_schedule", "serial") != "serial"
            ]
            assert len(no_tma_combined) == 2 * len(MMA_DAG_V_LAG_CHOICES)
            assert all(
                not any(seed.get("load_eviction_policies")) for seed in no_tma_combined
            )
            assert len(
                {no_tma_spec.normalized_config(seed) for seed in no_tma_combined}
            ) == len(no_tma_combined)
        assert selected is not None
        source = bound.to_code(selected)
    assert "probability_schedule='early_tmem'" in source
    assert "value_metadata_scales=" in source


def test_selected_contract_refuses_true_without_both_transport_facts():
    for available, paired in ((False, True), (True, False)):
        with _surface(available=available, paired=paired) as (_, spec):
            facts = spec.cute_mma_dag_facts
            assert facts is not None
            # Gate checks precede geometry; no manufactured tile proof is needed.
            assert paged_attention_config_reason(facts, {}, _config(True).config)


@pytest.mark.parametrize("key", [KEY, OVERLAP])
def test_generic_root_discards_default_but_rejects_true_and_wrong_type(key):
    @helion.kernel(backend="cute")
    def copy_kernel(x):
        y = torch.empty_like(x)
        for tile in hl.tile(x.shape[0]):
            y[tile] = x[tile]
        return y

    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(copy_kernel, (torch.empty(8),))
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        spec = bound.config_spec
        assert spec.cute_mma_dag is None
        old = spec.default_config()
        config = helion.Config.from_dict({**old.config, key: False})
        assert spec.normalized_config(config) == spec.normalized_config(old)
        config.config[key] = True
        with pytest.raises(helion.exc.InvalidConfig, match=key):
            spec.normalized_config(config)
        spec.normalize(config, _fix_invalid=True)
        assert config == spec.normalized_config(old)
        config.config[key] = 1
        with pytest.raises(helion.exc.InvalidConfig, match=key):
            spec.normalized_config(config)


@pytest.mark.parametrize("enabled", [False, True])
def test_read_overlap_roundtrip_and_neighbor_requires_metadata(enabled):
    with _surface() as (bound, spec):
        fragment = spec.cute_mma_dag_value_metadata_read_overlap
        assert fragment is not None and fragment.choices == (False, True)
        assert fragment.default() is False
        config = _config(True)
        config.config[OVERLAP] = enabled
        normalized = spec.normalized_config(config)
        assert normalized.get(OVERLAP) is enabled
        _assert_idempotent(spec, normalized)
        generation = spec.create_config_generation()
        flat, _ = generation.canonicalize_flat(generation.flatten(normalized))
        move = next(
            p
            for p in generation.coordinate_neighbor_projections(flat)
            if p.key == OVERLAP
        )
        assert move.outcome == "candidate" and move.config.get(OVERLAP) is not enabled
        assert bound.env.backend.autotune_config_is_viable(spec, move.config)
        config.config[KEY] = False
        if enabled:
            with pytest.raises(helion.exc.InvalidConfig, match=OVERLAP):
                spec.normalized_config(config)
        else:
            assert spec.normalized_config(config).get(OVERLAP) is False
        spec.normalize(config, _fix_invalid=True)
        assert config.get(OVERLAP) is False
        broken = helion.Config.from_dict(normalized.config)
        broken.config[KEY] = False
        _, repaired = generation.canonicalize_flat(generation.flatten(broken))
        assert repaired.get(KEY) is False and repaired.get(OVERLAP) is False
        _assert_idempotent(spec, repaired)
        facts = spec.cute_mma_dag_facts
        assert facts is not None
        assert paged_attention_config_reason(
            facts, {}, {**config.config, OVERLAP: True}
        )


@pytest.mark.parametrize("value", [0, 1, 0.0, "True", None])
def test_read_overlap_requires_exact_bool(value):
    with _surface() as (_, spec):
        config = _config(True)
        config.config[OVERLAP] = value
        with pytest.raises(helion.exc.InvalidConfig, match=OVERLAP):
            spec.normalized_config(config)
        spec.normalize(config, _fix_invalid=True)
        assert config.get(OVERLAP) is False
        _assert_idempotent(spec, config, repair=True)


def test_read_overlap_unavailable_and_repaired_paths_keep_default_identity():
    for available, paired in [(False, True), (True, False)]:
        with _surface(available=available, paired=paired) as (_, spec):
            assert spec.cute_mma_dag_value_metadata_read_overlap is None
            old = _config(False, family="attention")
            supplied = helion.Config.from_dict({**old.config, OVERLAP: True})
            with pytest.raises(helion.exc.InvalidConfig, match=OVERLAP):
                spec.normalized_config(supplied)
            spec.normalize(supplied, _fix_invalid=True)
            assert supplied == spec.normalized_config(old)
    with _surface() as (_, spec):
        old = _config(True)
        supplied = helion.Config.from_dict({**old.config, OVERLAP: False})
        assert spec.normalized_config(supplied) == spec.normalized_config(old)
        for family, mode in [("attention", "tcgen05"), ("off", "off")]:
            supplied = _config(True, family=family)
            supplied.config.update({OVERLAP: True, "cute_mma_dag": mode})
            assert spec.normalized_config(supplied).get(OVERLAP) is False
        supplied = _config(True)
        supplied.config.update({OVERLAP: True, "cute_min_blocks_per_mp": 2})
        spec.normalize(supplied, _fix_invalid=True)
        assert supplied.get("cute_mma_dag") == "off"
        assert supplied.get(OVERLAP) is False
        _assert_idempotent(spec, supplied, repair=True)
