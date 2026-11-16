from __future__ import annotations

import ast
import dataclasses
import functools
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import helion
from helion._compiler.cute.autotune.heuristics.cute_mma_dag import CuteMmaDagHeuristic
from helion._compiler.cute.cute_mma_dag import AXIS_LEAD
from helion._compiler.cute.cute_mma_dag import match_mma_dag
from helion._compiler.cute.cute_mma_dag import mma_dag_facts
from helion._compiler.cute.cute_mma_dag_seeds import MmaDagSeedPlanner
from helion._compiler.cute.paged_attention_config import paged_attention_config_reason
from helion._compiler.cute.paged_attention_config import paged_attention_required_knobs
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import stage_sources
from helion._testing.cute_paged_attention import ordered_paged_online_attention
from helion._testing.cute_paged_attention import paged_online_attention
from helion._testing.cute_paged_attention_schedule import paged_attention_args

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def bound():
    with cpu_render_target():
        return cpu_bind(paged_online_attention, paged_attention_args())


def _attention_config():
    return helion.Config(
        block_sizes=[64],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps="attention",
        cute_min_blocks_per_mp=1,
    )


@functools.cache
def _facts(bound):
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        program = match_mma_dag(bound.host_function.device_ir.graphs)
        assert program is not None
        return mma_dag_facts(program)


def test_attention_configuration_and_surface(bound):
    facts = _facts(bound)
    assert facts.attention is not None
    assert bound.config_spec.cute_mma_dag_pass_warps is not None
    assert "attention" in bound.config_spec.cute_mma_dag_pass_warps.choices
    sizes = {
        axis.block_id: 1 if axis.kind == AXIS_LEAD else axis.extent
        for axis in facts.axes
    }
    assert paged_attention_config_reason(facts, sizes, _attention_config()) is None
    with bound.env, bound.host_function:
        normalized = bound.config_spec.normalized_config(_attention_config())
    assert normalized.get("cute_mma_dag_pass_warps") == "attention"
    assert bound.env.backend.autotune_config_is_viable(bound.config_spec, normalized)


# Each mutation breaks a different obligation of this one physical schedule;
# these are contract negatives rather than a product of tuning dimensions.
@pytest.mark.parametrize(
    "changes",
    [
        {"cute_mma_dag": "warp"},
        {"cute_mma_dag_warps": 8},
        {"cute_mma_dag_stages": 4},
        {"cute_mma_dag_cluster": 2},
        {"cute_mma_dag_persistent": 1},
        {"cute_mma_dag_loads": "lane"},
        {"cute_mma_dag_derived": "pass"},
        {"cute_mma_dag_epilogue": "split"},
        {"cute_mma_dag_exchange": "smem"},
        {"cute_pdl": True, "cute_pdl_spread": True},
        {"cute_min_blocks_per_mp": 2},
        {"load_eviction_policies": ["last"]},
    ],
)
def test_attention_contract_declines_unimplemented_choices(bound, changes):
    facts = _facts(bound)
    sizes = {
        axis.block_id: 1 if axis.kind == AXIS_LEAD else axis.extent
        for axis in facts.axes
    }
    config = _attention_config().config | changes
    assert paged_attention_config_reason(facts, sizes, config) is not None


def test_attention_requires_both_proof_and_full_query_tile(bound):
    facts = _facts(bound)
    sizes = {
        axis.block_id: 1 if axis.kind == AXIS_LEAD else axis.extent
        for axis in facts.axes
    }
    assert paged_attention_config_reason(
        dataclasses.replace(facts, attention=None), sizes, _attention_config()
    )
    assert facts.attention is not None
    sizes[facts.axes[facts.attention.query_axis].block_id] = 32
    assert paged_attention_config_reason(facts, sizes, _attention_config())


def test_ordinary_schedule_is_preserved(bound):
    config = _attention_config()
    config.config["cute_mma_dag_pass_warps"] = "off"
    with bound.env, bound.host_function:
        # The generic renderer's derived state still exceeds its budget at
        # this geometry; only the proved attention topology bypasses it.
        with pytest.raises(helion.exc.InvalidConfig, match="derived value holds 256"):
            bound.config_spec.normalized_config(config)
        config.config["cute_mma_dag"] = "warp"
        config.config["cute_mma_dag_warps"] = 8
        normalized = bound.config_spec.normalized_config(config)
    assert normalized.get("cute_mma_dag_pass_warps") == "off"


def test_attention_seed_uses_the_same_contract(bound, monkeypatch):
    facts = _facts(bound)
    with bound.env, bound.host_function:
        planner = MmaDagSeedPlanner(
            facts, bound.config_spec, occupancy_key="cute_min_blocks_per_mp"
        )
        seeds = planner.configs([])
        assert len(seeds) == 3
        for key, value in paged_attention_required_knobs(facts).items():
            assert seeds[0].get(key) == value
        supports_key = bound.config_spec.supports_config_key
        with monkeypatch.context() as context:
            context.setattr(
                bound.config_spec,
                "supports_config_key",
                lambda key: key != "cute_min_blocks_per_mp" and supports_key(key),
            )
            assert planner.configs([]) == []
        policies = []
        for seed in seeds:
            normalized = bound.config_spec.normalized_config(seed)
            assert normalized.get("cute_mma_dag_pass_warps") == "attention"
            assert bound.env.backend.autotune_config_is_viable(
                bound.config_spec, normalized
            )
            policies.append(tuple(normalized.get("load_eviction_policies")))
        assert len(set(policies)) == 3
        declined = MmaDagSeedPlanner(
            dataclasses.replace(facts, attention=None),
            bound.config_spec,
            occupancy_key="cute_min_blocks_per_mp",
        )
        assert declined.configs([]) == []


def test_attention_source_owns_the_launch(bound):
    with bound.env, bound.host_function:
        normalized = bound.config_spec.normalized_config(_attention_config())
    source = bound.to_code(normalized)
    assert "_helion_paged_attention.paged_attention_body(" in source
    assert "helion_paged_attention" in source
    assert "block=(416, 1, 1)" in source


@pytest.mark.parametrize("rows,width", [(64, 64), (7, 128)])
def test_unsupported_transport_has_no_attention_choice(rows, width):
    with cpu_render_target():
        result = cpu_bind(
            paged_online_attention, paged_attention_args(rows=rows, width=width)
        )
    facts = _facts(result)
    assert facts.attention is None
    assert result.config_spec.cute_mma_dag_pass_warps is not None
    assert "attention" not in result.config_spec.cute_mma_dag_pass_warps.choices


def test_paired_maps_retain_proved_leading_stride(monkeypatch):
    from helion._compiler.cute import _paged_attention_runtime as runtime

    records = []

    def tensor_map(view, **options):
        # TMA requires a statically known contiguous leading mode even
        # when the generic launcher passes the remaining strides dynamically.
        assert type(view.layout.stride[0]) is int and view.layout.stride[0] == 1
        records.append((view, options))
        return view

    monkeypatch.setattr(
        runtime.cute,
        "make_layout",
        lambda shape, stride: SimpleNamespace(shape=shape, stride=stride),
    )
    monkeypatch.setattr(
        runtime.cute,
        "make_tensor",
        lambda iterator, layout: SimpleNamespace(iterator=iterator, layout=layout),
    )
    monkeypatch.setattr(runtime.cute, "recast_ptr", lambda pointer, dtype: pointer)
    monkeypatch.setattr(runtime.cuda, "create_tensor_map_tiled_from_view", tensor_map)
    raw_strides = tuple(object() for _ in range(5))
    scale_strides = tuple(object() for _ in range(4))
    raw, scale = SimpleNamespace(iterator=object()), SimpleNamespace(iterator=object())
    inspect.unwrap(runtime.make_paged_tma_operands)(
        raw,
        scale,
        (128, 128, 2, 1, 13),
        raw_strides,
        (128, 2, 1, 13),
        scale_strides,
        128,
    )
    assert len(records) == 2
    assert records[0][0].layout.stride[1:] == raw_strides[1:]
    assert records[1][0].layout.stride[1:] == scale_strides[1:]
    assert records[0][0].iterator is raw.iterator
    assert records[1][0].iterator is scale.iterator


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"cute_mma_dag_warps": 8}, "requires cute_mma_dag_warps=4"),
        ({"cute_mma_dag_stages": 4}, "requires cute_mma_dag_stages=2"),
        ({"cute_min_blocks_per_mp": 2}, "requires cute_min_blocks_per_mp=1"),
        ({"block_sizes": [32]}, "requires block size 64"),
    ],
)
def test_explicit_attention_reports_its_contract(bound, changes, reason):
    config = _attention_config()
    config.config.update(changes)
    with (
        bound.env,
        bound.host_function,
        pytest.raises(helion.exc.InvalidConfig, match=reason),
    ):
        bound.config_spec.normalized_config(config)


def test_attention_preserves_live_tma_policies_and_pins_inert_globals(bound):
    with bound.env, bound.host_function:
        config = bound.config_spec.normalized_config(_attention_config())
        config.config["pid_type"] = "persistent_blocked"
        config.config["load_eviction_policies"] = [
            "l2_last"
        ] * bound.config_spec.load_eviction_policies.length
        assert bound.env.backend.autotune_config_is_viable(bound.config_spec, config)
        normalized = bound.config_spec.normalized_config(config)
    assert normalized.get("cute_mma_dag_pass_warps") == "attention"
    assert normalized.get("pid_type") == "flat"
    slots = set(_facts(bound).attention.tma_load_slots)
    assert normalized.get("load_eviction_policies") == [
        "l2_last" if slot in slots else ""
        for slot in range(bound.config_spec.load_eviction_policies.length)
    ]


@pytest.mark.parametrize(
    "policy,canonical",
    [
        ("", ""),
        ("l2_last", "l2_last"),
        ("l1_l2_first", "streaming"),
        ("first", ""),
        ("last", ""),
    ],
)
def test_attention_scalar_policy_canonicalizes(bound, policy, canonical):
    config = _attention_config()
    config.config["load_eviction_policies"] = policy
    with bound.env, bound.host_function:
        assert bound.env.backend.autotune_config_is_viable(bound.config_spec, config)
        normalized = bound.config_spec.normalized_config(config)
    assert normalized.get("cute_mma_dag_pass_warps") == "attention"
    slots = set(_facts(bound).attention.tma_load_slots)
    assert normalized.get("load_eviction_policies") == [
        canonical if slot in slots else ""
        for slot in range(bound.config_spec.load_eviction_policies.length)
    ]


def test_attention_late_repair_revalidates_ordinary_budget(bound):
    config = _attention_config()
    # A malformed saved policy survives entry normalization and is rejected
    # only by the final topology contract. Repair must then check the ordinary
    # body's larger derived-value budget, rather than returning it unchecked.
    config.config["load_eviction_policies"] = 42
    with bound.env, bound.host_function:
        bound.config_spec.normalize(config, _fix_invalid=True)
        assert config.get("cute_mma_dag_pass_warps") == "off"
        assert bound.env.backend.autotune_config_is_viable(bound.config_spec, config)


@functools.cache
def _phase_bound():
    with cpu_render_target():
        return cpu_bind(ordered_paged_online_attention(), paged_attention_args())


@pytest.mark.parametrize("ordered", [False, True])
def test_attention_facts_use_the_current_phase(ordered):
    with cpu_render_target():
        bound = cpu_bind(
            ordered_paged_online_attention() if ordered else paged_online_attention,
            paged_attention_args()
            if ordered
            else (*paged_attention_args(), None, True),
        )
        spec = bound.config_spec
        # Re-registration must replace stale whole-root/phase metadata before
        # facts consult it. A whole-root pair is unavailable under a stale phase.
        spec.cute_mma_dag_phase_index = None if ordered else 1
        observed = []

        def inspect_facts(program):
            observed.append(spec.cute_mma_dag_phase_index)
            return mma_dag_facts(program)

        with patch("helion._compiler.cute.cute_mma_dag.mma_dag_facts", inspect_facts):
            facts = CuteMmaDagHeuristic._facts(bound.env, bound.host_function.device_ir)
        assert observed == [1 if ordered else None]
        assert facts is not None


@pytest.mark.parametrize("scalar", [False, True])
def test_attention_phase_projects_policies_before_contract_validation(scalar):
    bound = _phase_bound()
    spec = bound.config_spec
    assert spec.cute_mma_dag_phase_index == 1
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    owned = spec.cute_mma_dag_phase_load_slots
    siblings = set(range(spec.load_eviction_policies.length)) - owned
    assert siblings
    tma_slots = tuple(
        spec.cute_mma_dag_phase_eviction_map[slot]
        for slot in facts.attention.tma_load_slots
        if slot is not None
    )
    assert len(tma_slots) == len(set(tma_slots)) == 4
    expected_tma = (
        ("l2_last",) * 4 if scalar else ("streaming", "l2_last", "", "streaming")
    )
    selected = ["l2_last"] * spec.load_eviction_policies.length
    for slot, policy in zip(tma_slots, expected_tma, strict=True):
        selected[slot] = policy
    config = _attention_config()
    config.config["load_eviction_policies"] = "l2_last" if scalar else selected
    with bound.env, bound.host_function:
        normalized = spec.normalized_config(config)
    assert normalized.get("cute_mma_dag_pass_warps") == "attention"
    policies = normalized.get("load_eviction_policies")
    assert all(policies[slot] == "l2_last" for slot in siblings)
    tma = dict(zip(tma_slots, expected_tma, strict=True))
    assert tma.keys() <= owned
    assert all(policies[slot] == tma.get(slot, "") for slot in owned)
    sizes = {
        axis.block_id: 1 if axis.kind == AXIS_LEAD else axis.extent
        for axis in facts.axes
    }
    # The same final config fails the unprojected attention contract because
    # the sibling's live hint is still present. Successful normalization
    # above therefore depends on the phase-local policy view.
    assert (
        paged_attention_config_reason(facts, sizes, normalized)
        == "the attention schedule accepts L2 hints only on its TMA load slots"
    )
    source = bound.to_code(normalized)
    expected_hints = (
        ("evict_last",) * 4
        if scalar
        else ("evict_first", "evict_last", "", "evict_first")
    )
    assert _emitted_tma_hints(stage_sources(source)[1]) == expected_hints


def test_attention_phase_seeds_use_global_load_slots():
    bound = _phase_bound()
    spec = bound.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    tma = {
        spec.cute_mma_dag_phase_eviction_map[slot]
        for slot in facts.attention.tma_load_slots
        if slot is not None
    }
    assert tma != set(facts.attention.tma_load_slots)
    with bound.env, bound.host_function:
        seeds = CuteMmaDagHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
        assert seeds is not None
        seeds = [
            seed for seed in seeds if seed.get("cute_mma_dag_pass_warps") == "attention"
        ]
        assert len(seeds) == 3
        for seed, policy in zip(seeds, ("", "streaming", "l2_last"), strict=True):
            expected = [
                policy if slot in tma else ""
                for slot in range(spec.load_eviction_policies.length)
            ]
            assert seed.get("load_eviction_policies") == expected
            normalized = spec.normalized_config(seed)
            assert normalized.get("load_eviction_policies") == expected


def _emitted_tma_hints(source):
    call = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "paged_attention_body"
    )
    hints = next(
        keyword.value for keyword in call.keywords if keyword.arg == "tma_load_hints"
    )
    return ast.literal_eval(hints)


def test_attention_codegen_keeps_each_raw_and_scale_policy(bound):
    facts = _facts(bound)
    assert facts.attention is not None
    slots = facts.attention.tma_load_slots
    assert len(slots) == 4 and len(set(slots)) == 4 and None not in slots
    policies = ["last"] * bound.config_spec.load_eviction_policies.length
    for slot, policy in zip(
        slots, ("l1_l2_first", "l2_last", "", "streaming"), strict=True
    ):
        policies[slot] = policy
    config = _attention_config()
    config.config["load_eviction_policies"] = policies
    source = bound.to_code(config)
    assert _emitted_tma_hints(source) == (
        "evict_first",
        "evict_last",
        "",
        "evict_first",
    )
