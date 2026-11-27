"""The paired schedule's per-role register quotas as a config option.

``cute_mma_dag_register_quotas`` names the ``setmaxnreg`` count of the
compute, K conversion, V conversion and control warps. The option changes no
result and its default renders the kernel a config without it renders; the
admitted splits are measured ones, each must fit the CTA's register pool, and
the search picks among them. A persistent work partition keeps the roles and
their warps, so every split runs under every partition.
"""

from __future__ import annotations

import inspect
import traceback
from unittest.mock import patch

import pytest

import helion
from helion._compiler.cute import _paged_attention_paired_pipeline as paired_pipeline
from helion._compiler.cute import paged_attention_registers as registers
from helion._compiler.cute import paged_attention_transport
from helion._compiler.cute.cute_mma_dag import MMA_DAG_KEYS
from helion._compiler.cute.cute_mma_dag import paged_attention_register_quota_choices
from helion._compiler.cute.cute_mma_dag_seeds import MmaDagSeedPlanner
from helion._compiler.cute.paged_attention_registers import (
    MMA_DAG_REGISTER_QUOTAS_CHOICES as CHOICES,
)
from helion._compiler.cute.paged_attention_registers import (
    MMA_DAG_REGISTER_QUOTAS_DEFAULT as DEFAULT,
)
from helion._compiler.cute.paged_attention_registers import (
    MMA_DAG_REGISTER_QUOTAS_KEY as KEY,
)
from helion._compiler.cute.paged_attention_registers import (
    paired_register_quotas_reason,
)
from helion._compiler.cute.paged_attention_registers import parse_register_quotas
from helion._compiler.cute.paged_attention_transport import paged_attention_launch_warps
from helion._testing import cpu_render_target
from helion._testing.cute_offline import compile_offline
from helion._testing.cute_offline import partition_bound
from helion._testing.cute_paged_partition import partition_config
from helion._testing.cute_paged_register_quotas import decode_bound
from helion._testing.cute_paged_register_quotas import decode_config
from helion._testing.cute_paged_register_quotas import decode_kernel
from helion._testing.cute_paged_register_quotas import one_unit_quota_order
from helion._testing.cute_paged_register_quotas import partition_kernel
from helion._testing.cute_paged_register_quotas import persistent_quota_order
from helion._testing.cute_paged_register_quotas import sass_quota_order
from helion._testing.cute_paged_register_quotas import sass_quotas
from helion._testing.cute_paged_v_lag import paired_v_lag_bound
from helion._testing.cute_paged_v_lag import paired_v_lag_config
from helion._testing.cute_paged_v_lag import without_paired_head
from helion.runtime.cute.host_wrappers import HOST_WRAPPER_EMITTERS

pytestmark = pytest.mark.cpu_only
PASS_WARPS = "cute_mma_dag_pass_warps"
PARTITION = "cute_mma_dag_partition"
# Draft and verify steps of the decode kernel: 32 and 40 query columns.
DRAFT, VERIFY = 4, 5
# The call of the paired body ends with this argument in every render.
BODY_CALL_END = "coordinate_dtype=cutlass.Int32"


def _paired_config(quotas: object) -> helion.Config:
    config = paired_v_lag_config(0)
    config.config[KEY] = quotas
    return config


def test_admitted_splits_fit_the_launch_pool():
    # The pool is what the CTA launches with: twenty warps at the largest
    # setmaxnreg count that lets them share an SM's register file.
    assert registers.PAIRED_ROLE_WARPS == (8, 4, 4, 4)
    assert registers.paired_launch_threads() == 640
    assert paged_attention_launch_warps(per_key=True, paired_items=True) * 32 == 640
    assert registers.SM_REGISTERS // 640 == 102
    assert registers.launch_registers(640, 1) == 96
    assert registers.paired_register_pool() == 640 * 96 == 61440
    # The scoped schedule's launch counts, stated where it picks its occupancy.
    assert registers.launch_registers(384, 2) == 80
    assert registers.launch_registers(384, 1) == 168
    assert registers.launch_registers(32, 1) == registers.SETMAXNREG_MAX

    assert CHOICES[0] == DEFAULT and len(set(CHOICES)) == len(CHOICES)
    assert parse_register_quotas(DEFAULT) == registers.PAIRED_DEFAULT_REGISTER_QUOTAS
    # Today's split and the three admitted candidates, nothing else.
    assert CHOICES == (
        "144/72/72/48",
        "136/80/80/48",
        "136/72/72/48",
        "120/96/96/48",
    )
    used = {}
    for quotas in CHOICES:
        counts = parse_register_quotas(quotas)
        assert counts is not None
        assert registers.format_register_quotas(counts) == quotas
        assert paired_register_quotas_reason(counts) is None
        used[quotas] = sum(
            warps * 32 * count
            for warps, count in zip(registers.PAIRED_ROLE_WARPS, counts, strict=True)
        )
        assert used[quotas] <= registers.paired_register_pool()
    # The default uses the whole pool; one candidate leaves part of it unused.
    assert used[DEFAULT] == used["136/80/80/48"] == used["120/96/96/48"] == 61440
    assert used["136/72/72/48"] == 59392


@pytest.mark.parametrize(
    "counts,reason",
    [
        # One more step for either converter, or for compute, exceeds the pool.
        ((144, 80, 72, 48), "need 62464 registers and the CTA's pool holds 61440"),
        ((152, 72, 72, 48), "need 63488 registers and the CTA's pool holds 61440"),
        # setmaxnreg's own domain.
        ((140, 72, 72, 48), "compute quota 140 is not a multiple of 8"),
        ((144, 72, 72, 16), "control quota 16 is not a multiple of 8 in [24, 256]"),
        ((264, 24, 24, 24), "compute quota 264 is not a multiple of 8 in [24, 256]"),
        # The body raises compute and lowers the rest from the launch count.
        ((96, 96, 96, 96), "compute quota 96 is not above the 96 registers"),
        ((144, 104, 40, 48), "K conversion quota 104 is above the 96 registers"),
        ((144, 72, 72), "expected 4 quotas"),
    ],
)
def test_quotas_outside_the_pool_or_the_instruction_have_a_reason(counts, reason):
    assert reason in (paired_register_quotas_reason(counts) or "")


@pytest.mark.parametrize(
    "value",
    # One entry per way to miss the canonical spelling: the field count, a
    # leading zero, white space, another separator, a sign, nothing at all,
    # and each type that is not a string.
    [
        "144/72/72",
        "144/72/72/48/24",
        "0144/72/72/48",
        "144/72/72/48 ",
        "144,72,72,48",
        "144/-72/72/48",
        "",
        144,
        None,
        True,
        (144, 72, 72, 48),
    ],
)
def test_only_the_canonical_spelling_parses(value):
    assert parse_register_quotas(value) is None


@pytest.mark.parametrize("quotas", CHOICES)
def test_quotas_roundtrip_and_codegen(quotas):
    bound = paired_v_lag_bound()
    assert bound.host_function is not None
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        fragment = bound.config_spec.cute_mma_dag_register_quotas
        assert fragment is not None
        assert bound.config_spec._flat_fields()[KEY] is fragment
        # Searched in full: no timing regime narrows it.
        assert fragment.choices == CHOICES and fragment.search_choices is None
        assert fragment.default() == DEFAULT
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None
        assert paged_attention_register_quota_choices(
            facts.attention, "attention_paired"
        ) == (CHOICES)
        config = bound.config_spec.normalized_config(_paired_config(quotas))
        assert config.get(KEY) == quotas and type(config.get(KEY)) is str
        assert bound.env.backend.autotune_config_is_viable(bound.config_spec, config)
        again = bound.config_spec.normalized_config(
            helion.Config.from_dict(config.config)
        )
        assert again == config
        absent = bound.config_spec.normalized_config(paired_v_lag_config(0))
        assert absent.get(KEY) == DEFAULT
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
        without = bound.to_code(absent)
    assert "paged_attention_paired_body(" in source
    counts = parse_register_quotas(quotas)
    assert (f"register_quotas={counts!r}" in source) == (quotas != DEFAULT)
    # The default renders what a config without the option renders.
    assert "register_quotas" not in without
    assert (source == without) == (quotas == DEFAULT)


@pytest.mark.parametrize(
    "value",
    # Splits that fit the pool and are not admitted, spellings the parser
    # refuses and every scalar type a config can carry.
    [
        "152/64/64/48",
        "128/72/72/48",
        "128/80/80/48",
        "144/72/72",
        " 144/72/72/48",
        "default",
        144,
        0,
        True,
        None,
        1.5,
    ],
)
def test_quotas_outside_the_admitted_splits_are_invalid(value):
    bound = paired_v_lag_bound()
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        with pytest.raises(helion.exc.InvalidConfig, match=KEY):
            bound.config_spec.normalized_config(_paired_config(value))
        assert not bound.env.backend.autotune_config_is_viable(
            bound.config_spec, _paired_config(value)
        )
        repaired = _paired_config(value)
        bound.config_spec.normalize(repaired, _fix_invalid=True)
        assert repaired.get(KEY) == DEFAULT


def test_the_planner_admits_the_splits_the_shared_rule_admits():
    # The planner takes the admitted splits from the function the option's
    # choices, the config contract and the seeds take them from
    # (``paged_attention_register_quota_choices``), so a split that
    # normalizes always has a plan. Shown the default alone, the planner
    # declines a split the other three admit, and the kernel is left without
    # an attention schedule to run.
    bound = paired_v_lag_bound()
    assert bound.host_function is not None
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        config = bound.config_spec.normalized_config(_paired_config("136/80/80/48"))
    with cpu_render_target(fake_device_sm_count=148):
        assert "register_quotas=(136, 80, 80, 48)" in bound.to_code(config)
        with (
            patch.object(
                paged_attention_transport,
                "paged_attention_register_quota_choices",
                return_value=(DEFAULT,),
            ) as asked,
            pytest.raises(
                helion.exc.BackendUnsupported,
                match="cannot run the attention schedule",
            ),
        ):
            bound.to_code(config)
    assert asked.call_args.args[1] == "attention_paired"


@pytest.mark.parametrize("heads,per_key", [(3, True), (2, False)])
def test_quotas_require_the_paired_transport(heads, per_key):
    bound = paired_v_lag_bound(heads=heads, per_key=per_key)
    assert bound.config_spec.cute_mma_dag_register_quotas is None
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None
    assert paged_attention_register_quota_choices(
        facts.attention, "attention_paired"
    ) == (DEFAULT,)
    assert bound.host_function is not None
    with bound.env, bound.host_function:
        assert KEY not in bound.config_spec._flat_fields()
        config = _paired_config("136/80/80/48")
        config.config[PASS_WARPS] = "attention"
        with pytest.raises(helion.exc.InvalidConfig, match=KEY):
            bound.config_spec.normalized_config(config)
        # Naming the default is naming nothing: one identity with or without.
        absent = paired_v_lag_config(0)
        absent.config.update(
            {PASS_WARPS: "attention", "block_sizes": [64 if per_key else 32]}
        )
        named = helion.Config.from_dict({**absent.config, KEY: DEFAULT})
        normalized = bound.config_spec.normalized_config(named)
        assert KEY not in normalized.config
        assert normalized == bound.config_spec.normalized_config(absent)


@pytest.mark.parametrize("pass_warps", ["attention", "attention_scoped", "off"])
def test_other_schedules_pin_the_quotas(pass_warps):
    # Bodies without these roles keep one autotune identity whatever split a
    # search candidate carries, like the other paired knobs.
    bound = paired_v_lag_bound()
    assert bound.host_function is not None
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None
    assert paged_attention_register_quota_choices(facts.attention, pass_warps) == (
        DEFAULT,
    )
    config = _paired_config("120/96/96/48")
    config.config[PASS_WARPS] = pass_warps
    if pass_warps == "attention_scoped":
        config.config["cute_min_blocks_per_mp"] = 2
    elif pass_warps == "off":
        config.config["cute_mma_dag"] = "off"
    with bound.env, bound.host_function:
        normalized = bound.config_spec.normalized_config(config)
        assert normalized.get(KEY) == DEFAULT
        assert bound.env.backend.autotune_config_is_viable(
            bound.config_spec, normalized
        )
        default = _paired_config(DEFAULT)
        default.config.update(
            {key: value for key, value in config.config.items() if key != KEY}
        )
        assert normalized == bound.config_spec.normalized_config(default)


def test_one_seed_per_other_split_follows_the_leading_seed():
    bound = paired_v_lag_bound()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and bound.host_function is not None
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        seeds = MmaDagSeedPlanner(
            facts, bound.config_spec, occupancy_key="cute_min_blocks_per_mp"
        ).configs([])
        head = seeds[0]
        # The leading seed and every seed behind the counterparts name no
        # split: they run the default.
        named = [seed for seed in seeds if KEY in seed.config]
        assert named == seeds[1 : len(CHOICES)]
        assert [seed.get(KEY) for seed in named] == list(CHOICES[1:])
        for seed in named:
            # The leading seed with another split, nothing else changed.
            assert {k: v for k, v in seed.config.items() if k != KEY} == head.config
            normalized = bound.config_spec.normalized_config(seed)
            assert normalized.get(KEY) == seed.get(KEY)
            assert bound.env.backend.autotune_config_is_viable(
                bound.config_spec, normalized
            )
        # The shared seed-order helper checks and skips the counterparts.
        assert without_paired_head(seeds, facts) == seeds[len(CHOICES) :]
        assert bound.config_spec.normalized_config(head).get(KEY) == DEFAULT
        # A search that samples the dependent-launch policy seeds its
        # diagonal on the leading seed first; the splits follow it.
        with patch.object(bound.config_spec, "cute_pdl_capture_overlap_searched", True):
            searched = MmaDagSeedPlanner(
                facts, bound.config_spec, occupancy_key="cute_min_blocks_per_mp"
            ).configs([])
        assert searched[0] == head
        assert [seed for seed in searched if KEY in seed.config] == named
        assert searched.index(named[0]) == 3


def test_a_kernel_without_the_pair_proof_has_no_split_seed():
    bound = paired_v_lag_bound(heads=3)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and bound.host_function is not None
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        seeds = MmaDagSeedPlanner(
            facts, bound.config_spec, occupancy_key="cute_min_blocks_per_mp"
        ).configs([])
    assert seeds and all(KEY not in seed.config for seed in seeds)


def test_the_option_joins_the_family_and_the_compile_cache_identity():
    assert KEY in MMA_DAG_KEYS
    # The template asserts the pool with this module's arithmetic, so a
    # cached kernel is keyed by its source as well.
    dependencies = HOST_WRAPPER_EMITTERS["helion_paged_attention"].source_dependencies
    assert "_compiler/cute/paged_attention_registers.py" in dependencies
    assert "_compiler/cute/_paged_attention_paired_pipeline.py" in dependencies


def _render_with_quotas(query_len: int, counts: tuple[int, ...]) -> str:
    """The decode kernel's default render with ``counts`` passed to the
    paired body directly, as no config can."""
    bound, _ = decode_bound(query_len)
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(decode_config(bound))
    assert source.count(BODY_CALL_END) == 1 and "register_quotas" not in source
    return source.replace(BODY_CALL_END, f"{BODY_CALL_END}, register_quotas={counts!r}")


@pytest.mark.parametrize("query_len", [DRAFT, VERIFY])
def test_default_and_absent_quotas_render_the_same_source(query_len):
    bound, _ = decode_bound(query_len)
    with cpu_render_target(fake_device_sm_count=148):
        absent = bound.to_code(decode_config(bound))
        named = bound.to_code(decode_config(bound, **{KEY: DEFAULT}))
    # Byte-identical source: the default is the template's own parameter
    # default and the render names no split.
    assert named == absent and "register_quotas" not in absent
    assert absent.count(f"query_columns={query_len * 8}") == 1


# Both decode kernels (32 and 40 query columns) with every admitted split:
# each pair is another kernel to the assembler, and each is compiled once.
@pytest.mark.parametrize("quotas", CHOICES)
@pytest.mark.parametrize("query_len", [DRAFT, VERIFY])
def test_admitted_splits_reach_the_assembler_and_keep_the_launch_pool(
    query_len, quotas
):
    counts = parse_register_quotas(quotas)
    assert counts is not None
    # Compiled as a launch compiles it, under the launcher's options.
    kernel = decode_kernel(query_len, quotas)
    assert f"query_columns={query_len * 8}" in kernel.source
    assert (f"register_quotas={counts!r}" in kernel.source) == (quotas != DEFAULT)
    # The quotas reach the assembler in the roles' order.
    assert kernel.ptx_quotas() == one_unit_quota_order(counts)
    # The pool is derived from the CTA's threads and the count the assembler
    # has every one of them launch with, whatever part of it the split uses.
    assert kernel.threads == registers.paired_launch_threads() == 640
    assert kernel.launch_registers == registers.launch_registers(640, 1) == 96
    if kernel.sass is None:
        pytest.skip("nvdisasm is not installed")
    assert sass_quotas(kernel.sass) == sass_quota_order(one_unit_quota_order(counts))


def test_each_role_issues_its_own_quota(tmp_path):
    # Four different counts that fit the pool. The admitted splits give both
    # conversions one count, so only a split like this shows that the K and
    # the V conversion warps each issue their own.
    counts = (136, 80, 72, 48)
    assert paired_register_quotas_reason(counts) is None
    bound, args = decode_bound(DRAFT)
    kernel = compile_offline(bound, args, _render_with_quotas(DRAFT, counts), tmp_path)
    assert kernel.ptx_quotas() == one_unit_quota_order(counts)
    assert kernel.launch_registers == 96
    if kernel.sass is None:
        pytest.skip("nvdisasm is not installed")
    assert sass_quotas(kernel.sass) == sass_quota_order(one_unit_quota_order(counts))


# Both persistent partitions with every admitted split: each pair is another
# kernel to the assembler (the balanced one adds the exchange to the compute
# groups and the spare warp's role), and each is compiled once.
@pytest.mark.parametrize("quotas", CHOICES)
@pytest.mark.parametrize("partition", ["whole", "balanced"])
def test_persistent_partitions_run_every_split_from_the_same_pool(partition, quotas):
    counts = parse_register_quotas(quotas)
    assert counts is not None
    bound, _ = partition_bound()
    assert bound.host_function is not None
    named = partition_config(
        partition, cute_mma_dag_v_lag=2, cute_mma_dag_register_quotas=quotas
    )
    with (
        cpu_render_target(fake_device_sm_count=148, num_sm=148),
        bound.env,
        bound.host_function,
    ):
        # A partition pins no split: the config keeps both, and is viable.
        config = bound.config_spec.normalized_config(named)
        assert (config.get(PARTITION), config.get(KEY)) == (partition, quotas)
        assert bound.env.backend.autotune_config_is_viable(bound.config_spec, config)
    kernel = partition_kernel(partition, quotas)
    assert f"partition={partition!r}" in kernel.source
    assert (f"register_quotas={counts!r}" in kernel.source) == (quotas != DEFAULT)
    # Every role of the persistent CTA issues the quota of the role it
    # extends: the four control warps behind the scan, then the walking
    # compute groups and the conversion warps (one instruction for both
    # conversions, whose quotas are one count in every admitted split).
    assert kernel.ptx_quotas() == persistent_quota_order(counts)
    assert len(kernel.ptx_quotas()) == 3
    # The pool is the one-unit CTA's: the same threads, launched with the
    # same count, so the split's fit carries over.
    assert kernel.threads == registers.paired_launch_threads() == 640
    assert kernel.launch_registers == registers.launch_registers(640, 1) == 96
    if kernel.sass is None:
        pytest.skip("nvdisasm is not installed")
    assert sass_quotas(kernel.sass) == sass_quota_order(persistent_quota_order(counts))


def test_a_persistent_cta_refuses_conversions_with_two_counts(tmp_path):
    # A persistent CTA's K and V conversion warps take their quota from one
    # instruction, and no admitted split gives them two counts. A render
    # that named such a split, one the pool holds, fails when it is traced.
    counts = (136, 80, 72, 48)
    assert registers.paired_register_quotas_reason(counts) is None
    assert all(
        key == value
        for _compute, key, value, _control in map(
            registers.parse_register_quotas, registers.MMA_DAG_REGISTER_QUOTAS_CHOICES
        )
    )
    bound, args = partition_bound()
    with cpu_render_target(fake_device_sm_count=148, num_sm=148):
        source = bound.to_code(partition_config("balanced", cute_mma_dag_v_lag=2))
    assert source.count(BODY_CALL_END) == 1 and "register_quotas" not in source
    source = source.replace(
        BODY_CALL_END, f"{BODY_CALL_END}, register_quotas={counts!r}"
    )
    with pytest.raises(AssertionError) as refused:
        compile_offline(bound, args, source, tmp_path)
    # The assertion is the template's own of the two counts, not another
    # one that tracing the render happened to fail first.
    statements = [
        frame.line
        for frame in traceback.extract_tb(refused.value.__traceback__)
        if frame.filename == inspect.getsourcefile(paired_pipeline)
    ]
    assert statements[-1] == (
        "assert cutlass.const_expr(key_registers == value_registers)"
    )


def test_the_template_refuses_quotas_the_pool_cannot_hold(tmp_path):
    # No config reaches the template with such a split. A render that named
    # one would wait forever on the device, so tracing it fails with the
    # reason instead.
    bound, args = decode_bound(DRAFT)
    with pytest.raises(
        AssertionError,
        match="the roles need 63488 registers and the CTA's pool holds 61440",
    ):
        compile_offline(
            bound, args, _render_with_quotas(DRAFT, (144, 80, 80, 48)), tmp_path
        )
