"""Check the paired attention work partition's config contract and rendering."""

from __future__ import annotations

import ast
import contextlib
import copy
import ctypes
import dataclasses
import functools
import importlib
from pathlib import Path
import random
import re
from types import SimpleNamespace

from cutlass.base_dsl.compiler import DumpDir
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
import pytest

import helion
from helion import exc
from helion._compiler.cute import paged_attention_partition
from helion._compiler.cute import paged_attention_partition_bounds
from helion._compiler.cute._paged_attention_paired_pipeline import _pipeline_parts
from helion._compiler.cute._paged_attention_paired_pipeline import (
    _pipeline_shared_memory,
)
from helion._compiler.cute._paged_attention_partition import SKEW_SHARE
from helion._compiler.cute._paged_attention_partition import partial_slot_bytes
from helion._compiler.cute._paged_attention_partition import plan_geometry
from helion._compiler.cute._paged_attention_partition import plan_grid_bound
from helion._compiler.cute.autotune.mma_dag_surface import PARTITION_TRANSPORT
from helion._compiler.cute.autotune.mma_dag_surface import enable_cute_mma_dag_search
from helion._compiler.cute.cute_mma_dag_seeds import MmaDagSeedPlanner
from helion._compiler.cute.paged_attention_config import (
    paged_attention_partition_reason,
)
from helion._compiler.cute.paged_attention_config import (
    paged_attention_partition_skew_reason,
)
from helion._compiler.cute.paged_attention_partition import (
    PARTITION_SMEM_BYTES_ONE_STAGE,
)
from helion._compiler.cute.paged_attention_partition import (
    PARTITION_SMEM_BYTES_TWO_STAGES,
)
from helion._compiler.cute.paged_attention_partition import partition_q_stages
from helion._compiler.cute.paged_attention_partition import partition_skew_choices
from helion._compiler.cute.paged_attention_partition import partition_slot_bits
from helion._compiler.cute.paged_attention_partition_bounds import (
    partition_record_bytes,
)
from helion._testing import cpu_render_target
from helion._testing.cute_capture_gate import CaptureOnAnotherThread
from helion._testing.cute_capture_gate import capture_stand_ins
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import load_generated_module
from helion._testing.cute_paged_bounded import bounded_partial_fixture
from helion._testing.cute_paged_partition import SHORT_WINDOW
from helion._testing.cute_paged_partition import partition_attention_args
from helion._testing.cute_paged_partition import partition_attention_kernel
from helion._testing.cute_paged_partition import partition_config
from helion._testing.cute_paged_partition_plan import PARTITIONS
from helion._testing.cute_paged_partition_plan import active_ctas
from helion._testing.cute_paged_partition_plan import applied_skew
from helion._testing.cute_paged_partition_plan import fold_order
from helion._testing.cute_paged_partition_plan import planner_plans
from helion._testing.cute_paged_partition_plan import planner_slot_cta
from helion._testing.cute_paged_partition_plan import planner_tables
from helion._testing.cute_paged_partition_plan import planner_units
from helion._testing.cute_paged_partition_plan import reference_partition
from helion._testing.cute_paged_partition_plan import simulate_merges
from helion._testing.cute_paged_partition_plan import slot_ranges
from helion._testing.cute_paged_partition_plan import split_units
from helion._testing.cute_paged_partition_plan import unit_headers
from helion.runtime.cute import graph_lease
from helion.runtime.cute import launch_caches
from helion.runtime.cute import launcher
from helion.runtime.cute import paged_partition
from helion.runtime.cute.capture_pdl import CaptureGuard
from helion.runtime.cute.capture_pdl import CaptureKey
from helion.runtime.cute.capture_pdl import Dependency
from helion.runtime.cute.capture_pdl import LaunchEffects
from helion.runtime.cute.capture_pdl import StorageRange
from helion.runtime.cute.capture_pdl import WaitMode
from helion.runtime.cute.capture_pdl_cuda import CudaCaptureDriver
from helion.runtime.cute.host_wrappers import HOST_WRAPPER_EMITTERS

pytestmark = pytest.mark.cpu_only
KEY = "cute_mma_dag_partition"
SKEW = "cute_mma_dag_partition_skew"
PAIRED = "attention_paired"
QUERY = "cute_mma_dag_query_tma"
SKIP = "cute_mma_dag_value_metadata_skip"
TEMPLATE = (
    Path(__file__).parents[1]
    / "helion/_compiler/cute/_paged_attention_paired_pipeline.py"
)
DEVICE_HELPERS = TEMPLATE.with_name("_paged_attention_partition.py")
# 160 units of at most 17 pages on 11 SMs: mean shares of up to 261 slots,
# which take a skew of up to eight.
WIDE_SHARES = {"units": 160, "sms": 11}
WIDE_SKEWS = (0, 1, 2, 4, 8)


@functools.cache
def _bind(
    units: int,
    reserved_sms: int,
    capacity: int | None,
    window: int | None,
    fast_math: bool,
    input_banks: int,
    sms: int,
):
    facts = {} if capacity is None else {"smem_capacity_bytes": capacity}
    shape = {} if window is None else {"window": window}
    with cpu_render_target(fake_device_sm_count=sms, num_sm=sms, **facts):
        return cpu_bind(
            partition_attention_kernel(
                reserved_sms=reserved_sms, fast_math=fast_math, input_banks=input_banks
            ),
            partition_attention_args(units=units, **shape),
            num_sm=sms,
        )


def _bound(
    units: int = 17,
    reserved_sms: int = 0,
    capacity: int | None = None,
    window: int | None = None,
    fast_math: bool = True,
    input_banks: int = 1,
    sms: int = 148,
):
    """The module's bind of one shape. ``sms`` is the modelled capacity: the
    SMs a persistent launch may use, which a real bind reads less the
    reserved ones. The defaults are filled in ahead of the cache, so a
    shape is bound once however a test spells it."""
    return _bind(units, reserved_sms, capacity, window, fast_math, input_banks, sms)


def _render(
    config: helion.Config,
    units: int = 17,
    reserved_sms: int = 0,
    window: int | None = None,
    sms: int = 148,
) -> str:
    with cpu_render_target(fake_device_sm_count=sms, num_sm=sms):
        return _bound(units, reserved_sms, None, window, sms=sms).to_code(config)


def test_a_shape_is_bound_once_however_it_is_spelled():
    """With the defaults, by keyword, and by position as ``_render`` names
    it: one bind."""
    default = _bound()
    binds = _bind.cache_info().misses
    assert _bound(units=17) is default
    assert _bound(17, 0, None, None) is default
    assert _bind.cache_info().misses == binds


def _function(tree: ast.Module, prefix: str) -> ast.FunctionDef:
    (found,) = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith(prefix)
    ]
    return found


def _body_call(kernel: ast.FunctionDef) -> ast.Call:
    (call,) = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func).endswith("paged_attention_paired_body")
    ]
    return call


def test_whole_renders_a_unit_helper_and_one_cta_per_sm():
    source = _render(partition_config("whole", cute_mma_dag_v_lag=2), reserved_sms=140)
    tree = ast.parse(source)
    scanned = _function(tree, "_helion_pa_unit_bounds_scanned")
    (helper,) = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("_helion_pa_unit_bounds")
        and node is not scanned
    ]
    kernel = _function(tree, "_helion_paged_online_attention")
    # The prelude is straight-line code of the unit: the tracer runs it as
    # plain Python at every call, and nothing in it names this CTA.
    *assignments, returned = helper.body
    assert all(isinstance(node, ast.Assign) for node in assignments)
    assert isinstance(returned, ast.Return)
    assert not any(
        isinstance(node, (ast.If, ast.For, ast.While, ast.IfExp))
        for node in ast.walk(helper)
    )
    helper_text = ast.unparse(helper)
    assert not re.search(r"\b(pid_\d|num_blocks_\d|block_idx)", helper_text)
    parameters = [argument.arg for argument in helper.args.args]
    assert parameters[0] == "tile_offset_0" and "lengths" in parameters
    assert ast.unparse(returned.value).startswith("(cutlass.Int32(tile_offset_0 * 2")
    # The kernel keeps no work item of its own: the only length load is the
    # helper's, and the body gets typed zeros.
    assert helper_text.count("ldg_s32") == 2
    assert "ldg_" not in ast.unparse(kernel)
    # The source reads the length twice. Both loads are the same Int32 word,
    # so a second helper takes that word, as the plan scan kept it, in place
    # of each load and is the first helper otherwise, statement for statement.
    loads = [
        index
        for index, node in enumerate(helper.body)
        if "ldg_s32" in ast.unparse(node)
    ]
    assert len(loads) == 2
    assert [argument.arg for argument in scanned.args.args] == [
        parameters[0],
        "scanned_length",
        *parameters[1:],
    ]
    assert len(scanned.body) == len(helper.body)
    for index, (kept, loaded) in enumerate(zip(scanned.body, helper.body, strict=True)):
        if index in loads:
            assert isinstance(kept, ast.Assign) and isinstance(loaded, ast.Assign)
            assert ast.unparse(kept.targets[0]) == ast.unparse(loaded.targets[0])
            assert ast.unparse(kept.value) == "cutlass.Int32(scanned_length)"
        else:
            assert ast.unparse(kept) == ast.unparse(loaded)
    assert "ldg_" not in ast.unparse(scanned)
    call = _body_call(kernel)
    assert [ast.unparse(argument) for argument in call.args[4:8]] == [
        "cutlass.Int32(0)",
        "cutlass.Int64(0)",
        "cutlass.Int64(0)",
        "cutlass.Int32(0)",
    ]
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
    assert keywords["partition"] == "'whole'"
    assert (keywords["units"], keywords["page_limit"]) == ("17", "17")
    assert keywords["q_stages"] == "2"
    assert keywords["slot_bits"] == "32"
    assert keywords["unit_bounds"] == helper.name
    assert keywords["unit_bounds_scanned"] == scanned.name
    # Every kernel argument the prelude reads is passed, block sizes included,
    # which also keeps them from being dropped as unused.
    assert keywords["unit_bounds_args"] == f"({', '.join(parameters[1:])})"
    assert "_BLOCK_SIZE_0" in parameters and "\n_BLOCK_SIZE_0 = 1\n" in source
    # One CTA per unit up to one per SM of the launch device.
    assert (
        "_PARTITION_CTAS = helion.runtime.get_num_sm(q.device, reserved_sms=140)"
        in source
    )
    assert "(min(((34,)[0] // 2,)[0], _PARTITION_CTAS),)" in source
    assert "block=(640, 1, 1)" in source
    assert "get_num_sm(q.device)" in _render(partition_config("whole"))


def test_item_and_a_missing_key_share_config_and_source():
    absent = partition_config(cute_mma_dag_v_lag=2)
    explicit = partition_config("item", cute_mma_dag_v_lag=2)
    bound = _bound()
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        normalized = bound.config_spec.normalized_config(explicit)
        assert normalized == bound.config_spec.normalized_config(absent)
        assert KEY not in normalized.config
        whole = bound.config_spec.normalized_config(partition_config("whole"))
        assert whole.config[KEY] == "whole"
        assert bound.config_spec.normalized_config(whole) == whole
    source = _render(absent)
    assert source == _render(explicit)
    assert "unit_bounds" not in source and "_PARTITION_CTAS" not in source
    assert "((34,)[0] // 2,)" in source


@pytest.mark.parametrize(
    "change,reason",
    [
        ({QUERY: False}, f"{KEY}='whole' requires {QUERY}=True"),
        (
            {"cute_mma_dag_pass_warps": "attention"},
            (
                f"{KEY}='whole' needs the paired attention schedule "
                f"(cute_mma_dag_pass_warps={PAIRED!r} with {QUERY}=True), and "
                "this config selects cute_mma_dag_pass_warps='attention', whose "
                "only partition is 'item'"
            ),
        ),
        (
            {KEY: "balanced_fill"},
            f"{KEY} must be one of ('item', 'whole', 'balanced') for {PAIRED}",
        ),
        ({KEY: 1}, f"{KEY} must be one of ('item', 'whole', 'balanced') for {PAIRED}"),
    ],
)
def test_illegal_partition_is_rejected_and_repaired_to_item(change, reason):
    """Each case is one rule; the repair changes the partition and nothing else."""
    bound = _bound()
    config = partition_config("whole")
    config.config.update(change)
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        with pytest.raises(exc.InvalidConfig, match=re.escape(reason)):
            bound.config_spec.normalized_config(config)
        bound.config_spec.normalize(config, _fix_invalid=True)
        assert KEY not in config.config
        expected = partition_config()
        expected.config.update(
            {key: value for key, value in change.items() if key != KEY}
        )
        assert config == bound.config_spec.normalized_config(expected)


def test_partition_needs_the_paired_proof_and_no_value_metadata():
    facts = _bound().config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    assert facts.attention.partition_choices == ("item", "whole", "balanced")
    assert facts.attention.partition_units == 17
    assert facts.attention.partition_page_limit == 17
    assert facts.attention.partition_slot_bits == 32
    legal = {KEY: "whole", "cute_mma_dag_pass_warps": PAIRED, QUERY: True}
    assert paged_attention_partition_reason(facts, legal) is None
    assert paged_attention_partition_reason(facts, {**legal, SKIP: True}) == (
        f"{KEY}='whole' requires {SKIP}=False"
    )
    unpaired = dataclasses.replace(
        facts, attention=dataclasses.replace(facts.attention, paired_items=False)
    )
    assert paged_attention_partition_reason(unpaired, legal) == (
        f"{KEY} must be one of ('item',) for {PAIRED}"
    )
    # One unit per CTA is every schedule's partition, with or without the key.
    assert paged_attention_partition_reason(unpaired, {}) is None
    assert paged_attention_partition_reason(unpaired, {KEY: "item"}) is None
    # A partition on another schedule: the reason names the schedule the
    # partition needs and the one the config selects.
    reason = paged_attention_partition_reason(facts, {KEY: "whole"})
    assert reason is not None and reason.startswith(
        f"{KEY}='whole' needs the paired attention schedule "
        f"(cute_mma_dag_pass_warps={PAIRED!r} with {QUERY}=True), and this "
        "config selects cute_mma_dag_pass_warps='attention', whose only "
        "partition is 'item'. A key that a config leaves out takes that key's "
        "own default, not the kernel's default config"
    )
    # A value no schedule of the kernel offers keeps the plain reason.
    assert paged_attention_partition_reason(facts, {KEY: "halves"}) == (
        f"{KEY} must be one of ('item',) for attention"
    )


def test_a_config_that_names_only_the_partition_is_told_what_it_lacks():
    """A config that names a partition and nothing else does not get the
    kernel's default schedule with it: its other keys take their own
    defaults, which select another body. The error says so instead of
    listing that body's one partition."""
    bound = _bound()
    config = helion.Config(block_sizes=[64], **{KEY: "whole"})
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        with pytest.raises(
            exc.InvalidConfig,
            match=rf"{KEY}='whole' needs the paired attention schedule "
            rf"\(cute_mma_dag_pass_warps='{PAIRED}' with {QUERY}=True\), and "
            r"this config selects cute_mma_dag_pass_warps='\w+', whose only "
            r"partition is 'item'\. A key that a config leaves out takes that "
            r"key's own default, not the kernel's default config: add the "
            r"partition to a complete config of the paired schedule",
        ):
            bound.config_spec.normalized_config(config)
        bound.config_spec.normalize(config, _fix_invalid=True)
        assert KEY not in config.config
        assert config.config["cute_mma_dag_pass_warps"] != PAIRED


def test_query_stages_follow_the_shared_memory_capacity():
    assert [
        partition_q_stages(capacity)
        for capacity in (0, 232448, 208896, 208895, 188416, 188415)
    ] == [2, 2, 2, 1, 1, 0]
    for capacity, stages in ((200000, 1), (186000, 0)):
        bound = _bound(capacity=capacity)
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None and facts.attention is not None
        assert facts.attention.paired_items
        assert facts.attention.partition_q_stages == stages
        assert facts.attention.partition_choices == (
            ("item", "whole", "balanced") if stages else ("item",)
        )
        if stages:
            with cpu_render_target(
                fake_device_sm_count=148, smem_capacity_bytes=capacity
            ):
                assert "q_stages=1" in bound.to_code(partition_config("whole"))


def _seeds(bound, spec=None) -> list[helion.Config]:
    """The planner's seeds for ``bound``, under its config spec or ``spec``."""
    spec = bound.config_spec if spec is None else spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        planner = MmaDagSeedPlanner(facts, spec, occupancy_key="cute_min_blocks_per_mp")
        return planner.configs(planner.ranked())


def test_the_search_weighs_one_unit_per_cta_against_balanced():
    bound = _bound()
    spec = bound.config_spec
    fragment = spec._flat_fields()[KEY]
    assert fragment is spec.cute_mma_dag_partition
    assert fragment.choices == ("item", "whole", "balanced")
    # Whole units stay a value a config can name; the search does not draw it.
    assert fragment._active_choices() == ("item", "balanced")
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        default = spec.default_config()
        assert KEY not in default.config and SKEW not in default.config
        generation = spec.create_config_generation()
        for config in (
            partition_config(),
            partition_config("whole"),
            partition_config("balanced"),
        ):
            expected = spec.normalized_config(config)
            _flat, transferred = generation.canonicalize_flat(
                generation.flatten(expected)
            )
            for key in (KEY, SKEW, QUERY, "cute_mma_dag_v_lag"):
                assert transferred.config.get(key) == expected.config.get(key)
    # Shares of 18 slots take no skew: the bind has no such field.
    assert SKEW not in spec._flat_fields()
    # Without fast math no unit may be split: nothing is left to search.
    exact = _bound(fast_math=False).config_spec
    assert exact._flat_fields()[KEY]._active_choices() == ("item",)
    assert SKEW not in exact._flat_fields()


def test_the_skew_is_searched_only_over_bank_cycles():
    eager = _bound(**WIDE_SHARES)
    assert eager.config_spec._flat_fields()[SKEW].choices == WIDE_SKEWS
    assert eager.config_spec._flat_fields()[SKEW]._active_choices() == (0,)
    banks = _bound(**WIDE_SHARES, input_banks=2).config_spec
    assert banks.cute_mma_dag_partition_skew_searched
    assert banks._flat_fields()[SKEW]._active_choices() == WIDE_SKEWS
    # Searched or not, a skewed config keeps its skew through a search's row.
    with (
        cpu_render_target(fake_device_sm_count=11, num_sm=11),
        eager.env,
        eager.host_function,
    ):
        generation = eager.config_spec.create_config_generation()
        expected = eager.config_spec.normalized_config(
            partition_config("balanced", **{SKEW: 2})
        )
        _flat, transferred = generation.canonicalize_flat(generation.flatten(expected))
    assert (transferred.config[KEY], transferred.config[SKEW]) == ("balanced", 2)
    # A unit per SM, or 160 units on 148: shares of 18 and 19 slots take no
    # skew, and the bind has no such field, whatever the benchmark.
    assert SKEW not in _bound(160, input_banks=2).config_spec._flat_fields()


def test_seeds_balance_only_more_units_than_sms():
    """160 units on 148 SMs run in two waves; 17 have an SM each."""
    seeds = _seeds(_bound())
    assert any(seed.get("cute_mma_dag_pass_warps") == PAIRED for seed in seeds)
    assert not any(KEY in seed.config or SKEW in seed.config for seed in seeds)
    # One unit per SM is still one wave.
    assert not any(KEY in seed.config for seed in _seeds(_bound(148)))

    release = "cute_pdl_release"
    # The SMs, the window and the benchmark's input banks, then the skew and
    # the dependent release of each balanced seed.
    for sms, window, banks, wanted in (
        (148, None, 1, [(None, "loader_tail")]),
        (148, SHORT_WINDOW, 1, [(None, "loader_tail")]),
        # No window, but 148 CTAs share at most 19 slots each: the seeds'
        # skew of eight pages is not a choice of the bind.
        (148, -1, 1, [(None, "loader_tail")]),
        # On 11 SMs the shares reach 261 slots and take it, but an eager
        # search does not sample the skew and its benchmark cannot rank a
        # skewed seed against the balanced one: it gets none.
        (11, -1, 1, [(None, "loader_tail")]),
        (11, None, 1, [(None, "loader_tail")]),
        # A search over a cycle of input banks samples the skew and gets
        # the seed, which releases its dependents at entry there, where the
        # benchmark observes the release too. A window still gets none.
        (11, -1, 2, [(None, "loader_tail"), (8, "auto")]),
        (11, None, 2, [(None, "loader_tail")]),
    ):
        bound = _bound(160, 0, None, window, True, banks, sms)
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None and facts.attention is not None
        assert facts.attention.windowed is (window != -1)
        assert bound.config_spec.cute_pdl_capture_overlap_searched is (banks > 1)
        assert bound.config_spec.cute_mma_dag_partition_skew_searched is (banks > 1)
        # The seeds' skew is a choice of the bind on 11 SMs, whatever the
        # benchmark: only the search's sampling decides the seed there.
        assert (8 in facts.attention.partition_skew_choices) is (sms == 11)
        seeds = _seeds(bound)
        # The seeds of one unit per CTA keep their order; the leading family
        # follows them on the balanced partition.
        named = [seed for seed in seeds if KEY in seed.config]
        assert named == seeds[-len(wanted) :]
        assert [(seed.config.get(SKEW), seed.config[release]) for seed in named] == (
            wanted
        )
        head = seeds[0]
        assert head.get("cute_mma_dag_pass_warps") == PAIRED and head.get(QUERY)
        assert head.config[release] == "loader_tail"
        spec = bound.config_spec
        generation = spec.create_config_generation()
        with (
            cpu_render_target(fake_device_sm_count=sms, num_sm=sms),
            bound.env,
            bound.host_function,
        ):
            for seed, (skew, released) in zip(named, wanted, strict=True):
                expected = {**head.config, KEY: "balanced", release: released}
                if skew is not None:
                    expected[SKEW] = skew
                assert seed.config == expected
                # A seed is a complete legal config, and the search keeps it.
                normalized = spec.normalized_config(seed)
                assert normalized.config[KEY] == "balanced"
                assert normalized.config.get(SKEW) == skew
                assert normalized.config[release] == released
            transferred = [
                config
                for _flat, config in generation.seed_flat_config_pairs()
                if KEY in config.config
            ]
        assert [
            (config.config.get(SKEW), config.config[release]) for config in transferred
        ] == wanted
    # What gates the skewed seed is that the search samples the skew, not
    # that it samples the dependent-launch policy: a search over a bank
    # cycle that does not sample that policy (a kernel without it) still
    # gets the seed, with the release the balanced seed has.
    bound = _bound(160, 0, None, -1, True, 2, 11)
    spec = copy.copy(bound.config_spec)
    spec.cute_pdl_capture_overlap_searched = False
    assert spec.cute_mma_dag_partition_skew_searched
    named = [seed for seed in _seeds(bound, spec) if KEY in seed.config]
    assert [(seed.config.get(SKEW), seed.config[release]) for seed in named] == [
        (None, "loader_tail"),
        (8, "loader_tail"),
    ]


def _paired_parent(**knobs: object) -> helion.Config:
    return helion.Config(
        block_sizes=[64],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps=PAIRED,
        cute_min_blocks_per_mp=1,
        cute_mma_dag_v_lag=3,
        cute_pdl=False,
        **knobs,
    )


def _step(spec, parent: helion.Config, overrides=None, **moves: object):
    """The configs one requested step from ``parent`` gives a search: with
    the prerequisites the backend adds, and as requested."""
    generation = spec.create_config_generation(overrides=overrides)
    base, _config = generation.canonicalize_flat(generation.flatten(parent))
    slots = {key: entry[0][0] for key, entry in generation._key_to_flat_indices.items()}
    requested = list(base)
    for key, value in moves.items():
        requested[slots[key]] = value
    neighbor = generation.conditional_neighbor(base, requested)
    return generation.unflatten(list(neighbor)), generation.unflatten(requested)


def test_a_step_onto_a_partition_takes_the_transport_it_needs():
    bound = _bound(**WIDE_SHARES, input_banks=2)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    # The fixture has no value-metadata transport: supply its graph-free
    # interface, the option a partition must leave off.
    spec = copy.copy(bound.config_spec)
    facts = dataclasses.replace(
        facts,
        attention=dataclasses.replace(
            facts.attention, value_metadata_skip_available=True
        ),
    )
    overlap = "cute_mma_dag_value_metadata_read_overlap"
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        enable_cute_mma_dag_search(spec, facts)
        plain = _paired_parent()
        metadata = _paired_parent(**{SKIP: True, overlap: True})
        assert spec.normalized_config(metadata).config[SKIP] is True
        # One mapping states the transport, for the step and for a fixed
        # override: the options the legality rule rules on, at the values
        # it accepts, and the read overlap of the transport it refuses.
        assert dict(PARTITION_TRANSPORT) == {QUERY: True, SKIP: False, overlap: False}
        on_partition = {KEY: "balanced", "cute_mma_dag_pass_warps": PAIRED}
        for key in (QUERY, SKIP):
            refused = {**on_partition, **PARTITION_TRANSPORT}
            assert paged_attention_partition_reason(facts, refused) is None
            refused[key] = not PARTITION_TRANSPORT[key]
            assert paged_attention_partition_reason(facts, refused) == (
                f"{KEY}='balanced' requires {key}={PARTITION_TRANSPORT[key]}"
            )
        for parent in (plain, metadata):
            assert spec.normalized_config(parent).get(QUERY, False) is False
            reached, alone = _step(spec, parent, **{KEY: "balanced"})
            # Alone the step is where it started: normalization repairs a
            # partition without its transport to one unit per CTA.
            assert KEY not in alone.config
            assert reached.config[KEY] == "balanced"
            assert {key: reached.config[key] for key in PARTITION_TRANSPORT} == dict(
                PARTITION_TRANSPORT
            )
            assert reached.config["cute_mma_dag_v_lag"] == 3
            assert paged_attention_partition_reason(facts, reached.config) is None
            # A skew belongs to the balanced partition and brings it along.
            reached, alone = _step(spec, parent, **{SKEW: 4})
            assert KEY not in alone.config and SKEW not in alone.config
            assert (reached.config[KEY], reached.config[SKEW]) == ("balanced", 4)
            assert reached.config[QUERY] is True and not reached.config[SKIP]
        balanced = partition_config("balanced", cute_mma_dag_v_lag=3)
        # On the partition a skew is a step of its own, and so is leaving it.
        reached, alone = _step(spec, balanced, **{SKEW: 8})
        assert reached == alone and reached.config[SKEW] == 8
        skewed = partition_config("balanced", **{SKEW: 8})
        reached, alone = _step(spec, skewed, **{KEY: "item"})
        assert reached == alone
        assert KEY not in reached.config and SKEW not in reached.config
        # Another schedule pins the partition: its step asks for nothing.
        other = _paired_parent()
        other.config["cute_mma_dag_pass_warps"] = "attention"
        other.config["cute_mma_dag_v_lag"] = 0
        reached, alone = _step(spec, other, **{KEY: "balanced"})
        assert reached == alone and KEY not in reached.config
        # A fixed override is never changed: the step stays an alias.
        reached, alone = _step(
            spec, plain, overrides={QUERY: False}, **{KEY: "balanced"}
        )
        assert reached == alone and KEY not in reached.config


def _seeded_spec(bound):
    """The bind's surface with the value-metadata transport on offer, which a
    partition must leave off, and the compiler seeds of that surface."""
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    spec = copy.copy(bound.config_spec)
    facts = dataclasses.replace(
        facts,
        attention=dataclasses.replace(
            facts.attention, value_metadata_skip_available=True
        ),
    )
    enable_cute_mma_dag_search(spec, facts)
    planner = MmaDagSeedPlanner(facts, spec, occupancy_key="cute_min_blocks_per_mp")
    return spec, planner.configs(planner.ranked())


def _transferred(spec, seeds, overrides):
    """What the seeds become under ``overrides``; None for a dropped seed."""
    generation = spec.create_config_generation(overrides=overrides)
    kept = []
    for seed in seeds:
        try:
            kept.append(generation.canonicalize_flat(generation.flatten(seed))[1])
        except exc.InvalidConfig:
            kept.append(None)
    return kept


# The binds: a unit per SM (no seed names a partition), more units than SMs
# on a window and without one (a balanced seed each), all three with shares
# of 18 or 19 slots, which take no skew, and shares that take the seeds'
# skew, in a search over two input banks, which samples the skew and gets
# the skewed seed.
@pytest.mark.parametrize(
    "units,sms,window,banks",
    [(17, 148, None, 1), (160, 148, None, 1), (160, 148, -1, 1), (160, 11, -1, 2)],
)
def test_an_overridden_partition_or_skew_keeps_every_paired_seed(
    units, sms, window, banks
):
    """A fixed ``cute_mma_dag_partition`` or range skew is layered over
    complete candidates. Rejected wherever a candidate's own transport
    differed, a partition kept 6 to 9 of 27 to 55 paired seeds and a skew
    alone the balanced ones: none of them at 17 units, one or two at 160.
    Every paired candidate now takes what the search's step onto the
    partition takes."""
    bound = _bound(units, 0, None, window, input_banks=banks, sms=sms)
    overlap = "cute_mma_dag_value_metadata_read_overlap"
    with (
        cpu_render_target(fake_device_sm_count=sms, num_sm=sms),
        bound.env,
        bound.host_function,
    ):
        spec, seeds = _seeded_spec(bound)
        paired = [seed.get("cute_mma_dag_pass_warps") == PAIRED for seed in seeds]
        assert sum(paired) > 20 and not all(paired)
        assert any(seed.get(SKIP) for seed in seeds)
        assert any(
            not seed.get(QUERY)
            for seed, pair in zip(seeds, paired, strict=True)
            if pair
        )
        # Shares of 18 and 19 slots take no skew; those of 261 up to eight.
        fragment = spec.cute_mma_dag_partition_skew
        skews = (0,) if fragment is None else fragment.choices
        assert skews == (WIDE_SKEWS if sms == 11 else (0,))
        for override in (
            {KEY: "balanced"},
            {KEY: "whole"},
            *({SKEW: skew} for skew in skews[1:]),
            *({KEY: "balanced", SKEW: skew} for skew in skews[-1:] if skew),
        ):
            partition = override.get(KEY, "balanced")
            kept = _transferred(spec, seeds, override)
            for seed, pair, config in zip(seeds, paired, kept, strict=True):
                if not pair:
                    # One unit per CTA is every other body's partition.
                    assert config is None
                    continue
                assert config is not None, (override, seed)
                assert config.config[KEY] == partition
                # A seed's own skew stays where the partition is balanced.
                own = seed.config.get(SKEW) if partition == "balanced" else None
                assert config.config.get(SKEW) == override.get(SKEW, own)
                assert config.config[QUERY] is True
                assert not config.config.get(SKIP) and not config.config.get(overlap)
                # Nothing else moved.
                for key in ("cute_mma_dag_v_lag", "cute_pdl_release", "block_sizes"):
                    assert config.config.get(key) == spec.normalized_config(seed).get(
                        key
                    )
        # An overridden option is never changed: a partition named together
        # with a transport that cannot run it keeps no seed.
        for override in ({KEY: "balanced", QUERY: False}, {KEY: "whole", SKIP: True}):
            assert not any(_transferred(spec, seeds, override))
        if skews[-1]:
            assert not any(_transferred(spec, seeds, {KEY: "item", SKEW: skews[-1]}))
            assert not any(_transferred(spec, seeds, {KEY: "whole", SKEW: skews[-1]}))
        assert not any(_transferred(spec, seeds, {SKEW: 3}))
        # One unit per CTA and no skew are what the seeds already are; a
        # balanced seed's own skew does not survive another partition.
        named = [index for index, seed in enumerate(seeds) if SKEW in seed.config]
        assert bool(named) == (sms == 11)
        for override in ({KEY: "item"}, {KEY: "whole"}, {SKEW: 0}):
            kept = _transferred(spec, seeds, override)
            assert all(config is not None for config in kept) or KEY in override
            for index, config in enumerate(kept):
                if config is None:
                    assert not paired[index] and override == {KEY: "whole"}
                    continue
                assert SKEW not in config.config
                wanted = override.get(KEY, seeds[index].config.get(KEY, "item"))
                assert config.config.get(KEY, "item") == wanted


def test_whole_is_one_wave_for_the_dependent_release():
    # 160 units exceed the 148 SMs: one CTA per unit runs in two waves, on
    # which the guard of the last wave's CTAs exists for the config that
    # names it; the persistent grid is one wave.
    waves = _render(
        partition_config(cute_pdl=True, cute_pdl_release="last_wave"), units=160
    )
    assert "pdl_cta >= 12" in waves
    assert "pdl_cta" not in _render(partition_config(cute_pdl=True), units=160)
    for release in ("auto", "entry", "last_wave"):
        source = _render(
            partition_config("whole", cute_pdl=True, cute_pdl_release=release),
            units=160,
        )
        kernel = _function(ast.parse(source), "_helion_paged_online_attention")
        assert [ast.unparse(node) for node in kernel.body[:2]] == [
            "cute.arch.griddepcontrol_wait()",
            "cute.arch.griddepcontrol_launch_dependents()",
        ]
        assert "pdl_cta" not in source
    # Guarded capture overlap keeps its wrapper ABI and its contract.
    for release in ("auto", "loader_tail"):
        source = _render(
            partition_config(
                "whole",
                cute_pdl=True,
                cute_pdl_capture_overlap=True,
                cute_pdl_release=release,
            ),
            units=160,
        )
        kernel = _function(ast.parse(source), "_helion_paged_online_attention")
        assert ast.unparse(kernel.body[0]) == (
            "if capture_pdl_wait_mode == 0:\n    cute.arch.griddepcontrol_wait()"
        )
        keywords = {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in _body_call(kernel).keywords
        }
        assert keywords["capture_pdl_wait_mode"] == "capture_pdl_wait_mode"
        tail = release == "loader_tail"
        assert (keywords.get("capture_release_tail") == "True") == tail
        assert ("griddepcontrol_launch_dependents" in ast.unparse(kernel)) != tail
        assert "'capture_pdl_contract': 'paired-output-stores-v1'" in source
        assert "'capture_pdl_parameter_sizes': (128, 128, 128, 128, 128, 4)" in source
        assert source.count("'kind': 'helion_") == 2


def test_default_policies_follow_the_persistent_grid():
    """The schedule's defaults on 160 units and 148 SMs, for both partitions.

    Both get early retirement without a final join, the leading key clip and
    guarded capture overlap. One CTA per unit runs in two waves, so its
    dependents are released at entry in every CTA; the persistent grid
    is one wave by construction and releases from each loader's tail.
    """
    sources = {
        partition: _render(
            partition_config(partition, default_policies=True),
            units=160,
            window=SHORT_WINDOW,
        )
        for partition in (None, "whole")
    }
    for partition, source in sources.items():
        kernel = _function(ast.parse(source), "_helion_paged_online_attention")
        assert ast.unparse(kernel.body[0]) == (
            "if capture_pdl_wait_mode == 0:\n    cute.arch.griddepcontrol_wait()"
        )
        keywords = {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in _body_call(kernel).keywords
        }
        assert keywords["early_tmem_retirement"] == "True"
        assert keywords["key_clip"] == "'leading'"
        assert "join_before_exit" not in keywords
        assert keywords["capture_pdl_wait_mode"] == "capture_pdl_wait_mode"
        assert "'capture_pdl_contract': 'paired-output-stores-v1'" in source
        whole = partition == "whole"
        assert (keywords.get("capture_release_tail") == "True") == whole
        assert "pdl_cta" not in source
        assert ("griddepcontrol_launch_dependents" in ast.unparse(kernel)) != whole
        if not whole:
            assert ast.unparse(kernel.body[1]) == (
                "cute.arch.griddepcontrol_launch_dependents()"
            )
    bound = _bound(160, 0, None, SHORT_WINDOW)
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        normalized = bound.config_spec.normalized_config(
            partition_config("whole", default_policies=True)
        )
    assert normalized.config[KEY] == "whole"
    assert normalized.config["cute_pdl_release"] == "loader_tail"
    assert normalized.config["cute_pdl_capture_overlap"] is True


def test_units_stream_with_their_own_steps_and_the_cta_total():
    """Design contract of the loader's walk, read from the template.

    A unit's pages take the unit's own steps from zero, so page coordinates
    and a key clip's offsets see its first and last page as window edges and
    never the boundary between two units; ring slots follow the pages the
    CTA streamed before; the tail release counts the CTA's whole stream; a
    lagged V tile keeps its own page, key offset and first head.
    """
    tree = ast.parse(TEMPLATE.read_text())

    def calls(node: ast.AST, name: str) -> list[ast.Call]:
        return [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call) and ast.unparse(call.func) == name
        ]

    pages = _function(tree, "_load_pages")
    parameters = [argument.arg for argument in pages.args.args]
    walk = _function(tree, "_load_units")
    (stream,) = calls(walk, "_load_pages")
    passed = dict(
        zip(parameters, (ast.unparse(value) for value in stream.args), strict=True)
    )
    # A whole unit starts at its step zero; the balanced partition clips
    # the unit to the CTA's slots and starts where the clip does. The loader
    # takes that range from the helper every walking role calls, for the
    # slots it planned, and moves the header slot on by the unit.
    assert passed["g0"] == "streamed" and passed["step0"] == "step0"

    def balanced_arm(node: ast.AST) -> ast.If:
        (arm,) = [
            test
            for test in ast.walk(node)
            if isinstance(test, ast.If)
            and ast.unparse(test.test) == "cutlass.const_expr(partition == 'balanced')"
        ]
        return arm

    (taken,) = [
        node
        for node in ast.walk(walk)
        if isinstance(node, ast.Assign) and calls(node, "_unit_range")
    ]
    assert ast.unparse(taken) == (
        "item, first_page, last_page, length, unit_pages, step0, count = "
        "_unit_range(unit, header, begin, stop, None, lengths, unit_bounds, "
        "unit_bounds_scanned, unit_bounds_args, page_limit, partition)"
    )
    advance = balanced_arm(walk)
    assert [ast.unparse(node) for node in advance.body] == [
        "header = header + cutlass.Int32(unit_pages) + 1"
    ]
    assert taken.lineno < advance.lineno < stream.lineno
    ranged = _function(tree, "_unit_range")
    (first_step,) = [
        node
        for node in ranged.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "step0"
    ]
    assert ast.unparse(first_step.value) == "cutlass.Int64(0)"
    clip = balanced_arm(ranged)
    assert ast.unparse(clip.body[-1]) == (
        "count, step0 = unit_segment(header, unit_pages, begin, stop)"
    )
    assert first_step.lineno < clip.lineno
    assert ast.unparse(ranged.body[-1]) == (
        "return (item, first_page, last_page, length, unit_pages, step0, count)"
    )
    assert (passed["count"], passed["total"]) == ("count", "total")
    assert (passed["first_page"], passed["last_page"]) == ("first_page", "last_page")
    assert (passed["request"], passed["first_head"]) == (
        "item // heads",
        "item % heads",
    )
    assert passed["length"] == "length" and passed["key_clip"] == "key_clip"
    lag = (
        "previous_page",
        "older_page",
        "previous_shift",
        "older_shift",
        "previous_head",
        "older_head",
    )
    assert all(passed[name] == name for name in lag)
    assert ast.unparse(pages.body[-1]) == f"return ({', '.join(lag)})"
    (drain,) = calls(walk, "_drain_values")
    drained = [ast.unparse(value) for value in drain.args]
    assert drained[8:15] == ["streamed", *lag]
    # Coordinates and the key offset use the unit's step; slots the stream's.
    coordinates = calls(pages, "attention_page_coordinates")
    assert len(coordinates) == 2
    assert all(ast.unparse(call.args[2]) == "step0 + step" for call in coordinates)
    shifts = calls(pages, "logical_attention_key_shift")
    assert len(shifts) == 2
    assert all(
        [ast.unparse(value) for value in call.args[:3]]
        == ["logical_key_base", "128", "length"]
        for call in shifts
    )
    slots = {
        ast.unparse(call.args[6]) for call in calls(pages, "_enqueue_operand_tile")
    }
    assert slots == {"g0 + step", "g0 + step - distance"}
    releases = [
        node
        for node in ast.walk(pages)
        if isinstance(node, ast.If)
        and ast.unparse(node.body[0]) == "cute.arch.griddepcontrol_launch_dependents()"
    ]
    assert len(releases) == 2
    assert all(ast.unparse(node.test) == "g0 + step == total - 4" for node in releases)
    # The one-unit body passes the item itself as the stream.
    body = _function(tree, "paged_attention_paired_body")
    (single,) = calls(body, "_load_pages")
    one = dict(
        zip(parameters, (ast.unparse(value) for value in single.args), strict=True)
    )
    assert (one["g0"], one["step0"]) == ("cutlass.Int64(0)", "cutlass.Int64(0)")
    assert (one["count"], one["total"]) == ("count", "count")
    assert (one["previous_head"], one["older_head"]) == ("first_head", "first_head")


def _shared_bytes(
    query_columns: int,
    q_stages: int,
    partition: str,
    units: int = 17,
    scanned: bool = True,
) -> int:
    """Replay the paired body's shared-memory table as the allocator places
    it, in order and with each array's alignment, for a kernel with the
    query map and no value metadata. ``scanned``: the prelude has the helper
    that takes a scanned length."""
    # What the body passes for its table and its optional parts.
    body = _function(ast.parse(TEMPLATE.read_text()), "paged_attention_paired_body")
    (allocation,) = [
        call
        for call in ast.walk(body)
        if isinstance(call, ast.Call)
        and ast.unparse(call.func) == "_allocate_pipeline_shared_memory"
    ]
    _allocator, table, parts = allocation.args
    assert [ast.unparse(argument) for argument in table.args] == [
        "query_columns",
        "value_metadata_scales is not None",
        "q_stages if partition != 'item' else 0",
        "units if partition != 'item' else 0",
        "partition == 'balanced'",
    ]
    assert [ast.unparse(argument) for argument in parts.args] == [
        "query_map is not None",
        "value_metadata_scales is not None",
        "units if partition != 'item' else 0",
        "unit_bounds_scanned is not None",
        "partition == 'balanced'",
    ]
    persistent = partition != "item"
    balanced = partition == "balanced"
    arrays = _pipeline_shared_memory(
        query_columns,
        False,
        q_stages if persistent else 0,
        units if persistent else 0,
        balanced,
    )
    present = _pipeline_parts(
        True, False, units if persistent else 0, scanned, balanced
    )
    offset = 0
    for array in arrays:
        if array.guard is not None and array.guard not in present:
            continue
        width = array.dtype.width // 8
        alignment = max(array.alignment, width)
        offset = -(-offset // alignment) * alignment + width * array.count
    return offset


def test_persistent_shared_memory_stays_inside_its_budgets():
    """A persistent CTA adds query stages, their barriers (the pipeline's
    block grows from 32 to 64 slots), the compute groups' walk records, the
    plan tables (a word per scanning thread and 32 for the warps), the four
    words of the plan and, with one unit per scanning thread, a word per
    scanning thread for the scanned lengths. ``partition_q_stages`` admits
    two stages from 208,896 bytes of shared memory per CTA and one from
    188,416; the layouts must fit those with all twenty warps scanning, as
    for 609 to 640 units."""
    assert _shared_bytes(40, 1, "item") == 178960
    assert [plan_geometry(units) for units in (17, 608, 609, 640, 641, 10**6)] == [
        (1, 1),
        (1, 19),
        (1, 20),
        (1, 20),
        (2, 11),
        (1563, 20),
    ]
    assert max(plan_geometry(units)[1] for units in range(1, 5000)) == 20
    small = [_shared_bytes(40, stages, "whole") for stages in (1, 2)]
    assert small == [179240 + 272 + 128, 199720 + 272 + 128]
    one, two = (_shared_bytes(40, stages, "whole", 640) for stages in (1, 2))
    assert (one, two) == (179240 + 2704 + 2560, 199720 + 2704 + 2560)
    # The totals the budgets' comment states, and its second query panel.
    assert (one, two) == (184504, 204984) and two - one == 20480
    assert one <= PARTITION_SMEM_BYTES_ONE_STAGE < two
    assert two <= PARTITION_SMEM_BYTES_TWO_STAGES
    # The lengths are kept only with one unit per scanning thread and a
    # prelude that can take them.
    assert _shared_bytes(40, 2, "whole", 640, scanned=False) == 199720 + 2704
    assert _shared_bytes(40, 2, "whole", 641) == 199720 + 16 + 4 * 32 * 12
    assert _shared_bytes(40, 2, "whole", 641) == _shared_bytes(
        40, 2, "whole", 641, scanned=False
    )
    # Narrower query panels need less.
    assert all(
        _shared_bytes(columns, stages, "whole", 640) <= (one, two)[stages - 1]
        for columns in (8, 32)
        for stages in (1, 2)
    )
    assert _shared_bytes(32, 2, "whole") == 186920 + 272 + 128
    # The balanced partition adds twelve words to each compute group's
    # record, eight to the plan and eight for the groups' posts.
    for stages, limit in (
        (1, PARTITION_SMEM_BYTES_ONE_STAGE),
        (2, PARTITION_SMEM_BYTES_TWO_STAGES),
    ):
        whole = _shared_bytes(40, stages, "whole", 640)
        balanced = _shared_bytes(40, stages, "balanced", 640)
        assert balanced == whole + 4 * (2 * 12 + 8 + 8) and balanced <= limit
        # The totals the budgets' comment states.
        assert balanced == {1: 184664, 2: 205144}[stages]


def _check_partition(pages: list[int], grid: int) -> None:
    """The properties every partition of ``pages`` over ``grid`` CTAs has."""
    headers = unit_headers(pages)
    total = headers[-1]
    ranges = slot_ranges(pages, grid)
    active = min(grid, total)
    # The active CTAs tile the slots with equal shares of at least one slot.
    assert [begin for begin, _end in ranges[:active]] == [
        cta * total // active for cta in range(active)
    ]
    assert all(
        ranges[cta][1] == (ranges[cta + 1][0] if cta + 1 < active else total)
        for cta in range(active)
    )
    sizes = [end - begin for begin, end in ranges[:active]]
    assert min(sizes) >= 1 and max(sizes) - min(sizes) <= 1
    assert ranges[active:] == [(total, total)] * (grid - active)
    for partition in PARTITIONS:
        ctas = reference_partition(pages, grid, partition)
        assert len(ctas) == grid and not any(ctas[active:])
        owners: dict[int, list[tuple[int, object]]] = {}
        for cta, segments in enumerate(ctas):
            assert [s.unit for s in segments] == sorted({s.unit for s in segments})
            for segment in segments:
                owners.setdefault(segment.unit, []).append((cta, segment))
        assert sorted(owners) == list(range(len(pages)))
        for unit, count in enumerate(pages):
            parts = owners[unit]
            where = [cta for cta, _segment in parts]
            # Consecutive CTAs; the lowest holds the last steps, so the steps
            # ascend from the highest CTA down and cover the unit once.
            assert where == list(range(where[0], where[0] + len(parts)))
            steps = [
                step
                for _cta, segment in reversed(parts)
                for step in range(segment.step, segment.step + segment.count)
            ]
            assert steps == list(range(count))
            assert all(segment.pages == count for _cta, segment in parts)
            if partition == "whole" or count == 0:
                # One owner: the CTA of the header slot.
                ((cta, segment),) = parts
                assert segment.whole
                assert ranges[cta][0] <= headers[unit] < ranges[cta][1]
                continue
            if len(parts) == 1:
                assert parts[0][1].whole
                continue
            assert all(segment.count > 0 for _cta, segment in parts)
            assert parts[0][1].tail and parts[-1][1].head
            for index, (cta, segment) in enumerate(parts):
                assert not segment.whole
                carrying = [s for s in ctas[cta] if s.count]
                # A CTA ends with the part whose unit it is the lowest CTA
                # of, and starts with a part of a unit begun below it.
                assert (carrying[-1] if index == 0 else carrying[0]) is segment
        if partition == "balanced":
            # A CTA's pages are its slots less the headers among them.
            for cta, (begin, end) in enumerate(ranges):
                inside = sum(begin <= header < end for header in headers[:-1])
                assert sum(s.count for s in ctas[cta]) == end - begin - inside


def test_reference_partition_covers_every_unit_once():
    """The slot definitions, on the shapes that have bitten: a unit longer
    than the launch has CTAs, fewer slots than CTAs, empty units at every
    position, one unit per SM and one more or less, and seeded random
    lengths."""
    cases = [
        ([17], 148),
        ([1, 0, 0], 148),
        ([0], 148),
        ([0, 0, 0], 2),
        ([391], 148),
        ([0, 5, 300], 148),
        ([17, 0, 0], 148),
        ([8192, *[17] * 127], 148),
        ([0] * 320, 148),
        ([3, 0, 0, 0, 7, 0, 1], 3),
    ]
    cases += [([17] * units, 148) for units in (1, 147, 148, 149, 300)]
    cases += [([211] * 128, 148), ([17] * 200, 148), ([17] * 128, 128)]
    generator = random.Random(7)
    for _ in range(60):
        units = generator.choice([1, 2, 5, 17, 40, 149, 320])
        limit = generator.choice([1, 4, 17, 60])
        pages = [
            generator.choice([0, generator.randint(0, limit), limit])
            for _unit in range(units)
        ]
        cases.append((pages, generator.choice([1, 2, 8, 148, 149, units + 3])))
    for pages, grid in cases:
        _check_partition(pages, grid)
    # Fewer slots than CTAs: one unit of 17 pages has 18 slots. The first
    # CTA holds only its header: the unit itself under ``whole``, nothing
    # under ``balanced``, whose next 17 CTAs take a page each.
    assert [len(s) for s in reference_partition([17], 148, "whole")[:3]] == [1, 0, 0]
    split = reference_partition([17], 148, "balanced")
    assert [len(segments) for segments in split[:19]] == [0, *[1] * 17, 0]
    assert [segments[0].step for segments in split[1:18]] == list(range(16, -1, -1))
    empties = reference_partition([1, 0, 0], 148, "balanced")
    assert [[(s.unit, s.count) for s in segments] for segments in empties[:5]] == [
        [],
        [(0, 1)],
        [(1, 0)],
        [(2, 0)],
        [],
    ]
    # The eval's shapes: 200 and 128 requests of a 17-page window and 128 of
    # 211 pages on 148 SMs.
    for pages, per_cta, segments, shared in (
        ([17] * 200, (22, 24), {2: 116, 3: 32}, {1: 72, 2: 128}),
        ([17] * 128, (14, 16), {1: 40, 2: 108}, {2: 128}),
        ([211] * 128, (182, 184), {1: 24, 2: 124}, {2: 112, 3: 16}),
    ):
        ctas = reference_partition(pages, 148, "balanced")
        counts = [sum(s.count for s in segments) for segments in ctas]
        assert (min(counts), max(counts)) == per_cta
        assert {
            n: sum(len(s) == n for s in ctas) for n in {len(s) for s in ctas}
        } == segments
        parts = [
            sum(any(s.unit == u for s in c) for c in ctas) for u in range(len(pages))
        ]
        assert {n: parts.count(n) for n in set(parts)} == shared
    # ``whole`` balances the same slots with whole units: a long unit's CTA
    # gets no more, and the ranges it spans get nothing.
    ragged = reference_partition([60, 1, 1, 1, 1, 1, 1], 4, "whole")
    assert [[s.unit for s in segments] for segments in ragged] == [
        [0],
        [],
        [],
        [1, 2, 3, 4, 5, 6],
    ]


def test_whole_is_one_unit_per_cta_only_for_equal_lengths():
    """With no more units than CTAs ``whole`` is not ``item``. The slot rule
    gives every unit a CTA of its own when the lengths are equal; with ragged
    lengths the CTAs inside a long unit's slots own nothing and short units
    share a CTA, which then streams more than any unit has pages."""
    units = 128
    equal = reference_partition([17] * units, units, "whole")
    assert [[s.unit for s in segments] for segments in equal] == [
        [unit] for unit in range(units)
    ]
    cycle = (17, 0, 11, 6, 2)
    pages = [cycle[unit % len(cycle)] for unit in range(units)]
    ragged = reference_partition(pages, units, "whole")
    owned = [len(segments) for segments in ragged]
    assert sum(owned) == units
    assert owned.count(0) == 44 and max(owned) == 2
    busiest = max(sum(s.count for s in segments) for segments in ragged)
    assert busiest == 19 and max(pages) == 17
    # The device plan is that partition: no rule maps it to one unit per CTA.
    plans = planner_units(pages, units, 32, 17)
    assert [last - first for first, last, _count in plans] == owned


def test_planner_mirror_matches_the_reference_partition():
    """The device's scan and decode, lane by lane in Python, find the units
    the definitions give each CTA: with one unit per thread and with several
    (more than 640 units), with both widths of the slot products, with more
    CTAs than slots and with units longer than a CTA's share."""
    generator = random.Random(11)
    cases = [
        ([17] * 200, 148, 17),
        ([17], 148, 17),
        ([1, 0, 0], 148, 17),
        ([0] * 33, 148, 17),
        ([8192, *[17] * 127], 148, 8192),
        ([2] * 641, 148, 2),
        ([3] * 46341, 148, 17),
    ]
    for _ in range(80):
        units = generator.choice([1, 2, 31, 32, 33, 64, 148, 300, 640, 641, 1300])
        limit = generator.choice([1, 3, 17, 40])
        pages = [
            generator.choice([0, generator.randint(0, limit), limit])
            for _unit in range(units)
        ]
        if generator.randrange(4) == 0:
            pages = [0] * units
            pages[generator.randrange(units)] = limit
        cases.append((pages, generator.choice([1, 7, 148, 300, units + 3]), limit))
    for pages, grid, limit in cases:
        units = len(pages)
        prefix, totals = planner_tables(pages)
        chunk, warps = plan_geometry(units)
        assert len(prefix) == 32 * warps and len(totals) == warps
        assert sum(totals) == unit_headers(pages)[-1]
        assert chunk * 32 * warps >= units > chunk * 32 * (warps - 1)
        for slot_bits in (32, 64):
            bound = plan_grid_bound(units, limit) if slot_bits == 32 else None
            expected = reference_partition(pages, grid, "whole", grid_bound=bound)
            plans = planner_units(pages, grid, slot_bits, limit)
            for (first, last, count), segments in zip(plans, expected, strict=True):
                assert list(range(first, last)) == [s.unit for s in segments]
                assert count == sum(s.count for s in segments)


def test_slot_products_widen_before_they_can_overflow():
    """The eval's widest bind keeps Int32 products; a bind whose slots times
    twice the SM count pass 2**31 takes Int64 ones; one whose slots alone do
    not fit has no persistent partition. The bound has one definition: the
    device code caps a launch's active CTAs with the function the
    compile-time side decides the width by."""
    assert (
        paged_attention_partition.plan_grid_bound
        is paged_attention_partition_bounds.plan_grid_bound
        is plan_grid_bound
    )
    for units, limit in ((1, 1), (320, 8192), (885, 8192), (262112, 8192)):
        bound = (2**31 - 1) // (units * (limit + 1))
        assert plan_grid_bound(units, limit) == bound
        # The largest grid whose products stay Int32, and the next does not.
        assert bound * units * (limit + 1) < 2**31
        assert (bound + 1) * units * (limit + 1) >= 2**31
        for num_sm in (1, 148, bound // 2, bound // 2 + 1):
            assert partition_slot_bits(units, limit, num_sm) == (
                32 if bound >= 2 * num_sm else 64
            )
    # The template's kernels are keyed by the module that states it.
    assert (
        "_compiler/cute/paged_attention_partition_bounds.py"
        in HOST_WRAPPER_EMITTERS["helion_paged_attention"].source_dependencies
    )
    assert plan_grid_bound(320, 8192) == 819
    assert partition_slot_bits(320, 8192, 148) == 32
    assert partition_slot_bits(885, 8192, 148) == 32
    assert partition_slot_bits(886, 8192, 148) == 64
    assert partition_slot_bits(262112, 8192, 148) == 64
    assert partition_slot_bits(262113, 8192, 148) == 0
    assert partition_slot_bits(17, 0, 148) == 0
    # With Int32 products the device caps the active CTAs at the bound, so
    # a launch on more SMs than expected idles CTAs instead of overflowing.
    pages = [8192] * 885
    bound = plan_grid_bound(885, 8192)
    assert bound == 296
    capped = planner_units(pages, 400, 32, 8192)
    assert capped[bound:] == [(885, 885, 0)] * (400 - bound)
    assert sum(last - first for first, last, _count in capped) == 885


def test_balanced_renders_a_workspace_the_capture_proof_counts_as_stored():
    """The balanced partition's kernel takes one more tensor, the launch
    workspace, which the host function asks the runtime for on every call.
    The alias plan lists it as stored and not as read ahead of the
    dependency wait, so a captured launch may still overlap its predecessor
    and no other argument may overlap it; the wrapper-only suffix the
    capture proof patches is what it was."""
    source = _render(
        partition_config("balanced", default_policies=True),
        units=160,
        window=SHORT_WINDOW,
    )
    tree = ast.parse(source)
    kernel = _function(tree, "_helion_paged_online_attention")
    parameters = [argument.arg for argument in kernel.args.args]
    assert parameters[-7:] == [
        "pa_partition_ws",
        "pa_map",
        "pa_map_1",
        "pa_map_2",
        "pa_map_3",
        "pa_query_map",
        "capture_pdl_wait_mode",
    ]
    keywords = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in _body_call(kernel).keywords
    }
    assert keywords["partition"] == "'balanced'"
    assert keywords["partition_workspace"] == "pa_partition_ws"
    # Forty query columns: ten 16-byte vectors per thread and forty row
    # pairs per slot, four slots and the counters' line per CTA.
    assert keywords["partition_record_bytes"] == "83584"
    assert (keywords["units"], keywords["q_stages"]) == ("160", "2")
    assert "unit_bounds_scanned" in keywords
    # The same grid and defaults as the whole-unit partition: one wave, so
    # the loaders release the dependents from their tails.
    assert keywords["capture_release_tail"] == "True"
    assert keywords["capture_pdl_wait_mode"] == "capture_pdl_wait_mode"
    assert "(min(((320,)[0] // 2,)[0], _PARTITION_CTAS),)" in source
    assert (
        "from helion.runtime.cute.paged_partition import paged_partition_workspace "
        "as _helion_paged_partition_workspace" in source
    )
    # The capacity is defined once, ahead of the launch that sizes both the
    # grid and the workspace with it.
    assert source.count("_PARTITION_CTAS = helion.runtime.get_num_sm(q.device)") == 1
    assert (
        "_helion_paged_partition_workspace(q, ctas=_PARTITION_CTAS, "
        "record_bytes=83584), block=(640, 1, 1))" in source
    )
    (plans,) = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]).endswith("._helion_cute_wrapper_plans")
    ]
    attention, alias = plans
    assert attention["capture_pdl_contract"] == "paired-output-stores-v1"
    assert attention["capture_pdl_parameter_sizes"] == (128, 128, 128, 128, 128, 4)
    names = alias["tensor_names"]
    assert names[-1] == "pa_partition_ws" and len(set(names)) == len(names)
    assert alias["stored"][-1] is True and alias["capture_loaded"][-1] is False
    stored = {name for name, flag in zip(names, alias["stored"], strict=True) if flag}
    assert stored == {"out", "lse", "pa_partition_ws"}
    assert alias["tensor_idx"][-1] == parameters.index("pa_partition_ws")
    # Without the partition there is no workspace anywhere.
    assert "partition_ws" not in _render(partition_config("whole"))


def test_balanced_needs_fast_math_and_the_normalized_epilogue():
    """Merging the parts of a split unit reassociates the source's sums:
    only fast math offers the partition. Without it the choice does not
    exist, an explicit config is rejected and the repair is one unit per
    CTA; whole units remain available."""
    exact = _bound(fast_math=False)
    facts = exact.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    assert facts.attention.partition_choices == ("item", "whole")
    assert facts.attention.partition_record_bytes == 0
    config = partition_config("balanced")
    with cpu_render_target(fake_device_sm_count=148), exact.env, exact.host_function:
        with pytest.raises(
            exc.InvalidConfig,
            match=re.escape(f"{KEY} must be one of ('item', 'whole') for {PAIRED}"),
        ):
            exact.config_spec.normalized_config(config)
        exact.config_spec.normalize(config, _fix_invalid=True)
        assert KEY not in config.config
        assert config.config["cute_mma_dag_pass_warps"] == PAIRED
        whole = exact.config_spec.normalized_config(partition_config("whole"))
        assert whole.config[KEY] == "whole"
    fast = _bound()
    with cpu_render_target(fake_device_sm_count=148), fast.env, fast.host_function:
        kept = fast.config_spec.normalized_config(partition_config("balanced"))
        assert kept.config[KEY] == "balanced"
        for change, reason in (
            ({QUERY: False}, f"{KEY}='balanced' requires {QUERY}=True"),
            ({SKIP: True}, f"{KEY}='balanced' requires {SKIP}=False"),
        ):
            facts = fast.config_spec.cute_mma_dag_facts
            assert facts is not None
            illegal = {KEY: "balanced", "cute_mma_dag_pass_warps": PAIRED, QUERY: True}
            assert paged_attention_partition_reason(facts, {**illegal, **change}) == (
                reason
            )
    assert (
        partition_record_bytes(40)
        == fast.config_spec.cute_mma_dag_facts.attention.partition_record_bytes
    )


def test_raw_statistics_offer_whole_units_only():
    """The raw-statistics epilogue stores each unit's unnormalized panel,
    maximum and denominator, and has no merge: a bind of it offers whole
    units and no balanced partition, with fast math too, and has no
    workspace record."""
    kernel, initial = bounded_partial_fixture(4)
    args = (*initial[:7], 31, *initial[8:])
    fast = helion.kernel(kernel.fn, backend="cute", static_shapes=True, fast_math=True)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(fast, args)
    assert bound.env.settings.fast_math
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    assert facts.attention.partial_statistics and facts.attention.paired_items
    assert facts.attention.partition_choices == ("item", "whole")
    assert facts.attention.partition_record_bytes == 0
    assert facts.attention.partition_skew_choices == (0,)
    paired = {"cute_mma_dag_pass_warps": PAIRED, QUERY: True}
    assert paged_attention_partition_reason(facts, {KEY: "whole", **paired}) is None
    assert paged_attention_partition_reason(facts, {KEY: "balanced", **paired}) == (
        f"{KEY} must be one of ('item', 'whole') for {PAIRED}"
    )


def test_a_record_holds_two_counters_and_four_partials():
    """The compile-time record size is the device layout's: a line for the
    counters and four slots, each one head's panel as a 16-byte vector per
    thread and row quad followed by the rows' pairs, in whole lines."""
    for columns, panel, slot, record in (
        (8, 4096, 4224, 17024),
        (32, 16384, 16640, 66688),
        (40, 20480, 20864, 83584),
    ):
        assert partial_slot_bytes(columns) == (panel, slot)
        assert panel == columns // 4 * 128 * 16 and slot % 128 == 0
        assert slot >= panel + columns * 8
        assert partition_record_bytes(columns) == 128 + 4 * slot == record
    # One definition of the layout: the device template and the compile-time
    # side take the header, the slot and the record from the module that
    # states them, and define none of them again.
    bounds = paged_attention_partition_bounds
    assert partial_slot_bytes is bounds.partial_slot_bytes
    names = {"RECORD_HEADER_BYTES", "partial_slot_bytes", "partition_record_bytes"}
    device = importlib.import_module("helion._compiler.cute._paged_attention_partition")
    transport = importlib.import_module(
        "helion._compiler.cute.paged_attention_transport"
    )
    assert device.RECORD_HEADER_BYTES is bounds.RECORD_HEADER_BYTES
    assert transport.partition_record_bytes is bounds.partition_record_bytes
    # The host, which zeroes the counters of a workspace it allocates,
    # takes their size from there as well: the two words lead the header.
    assert bounds.RECORD_COUNTER_BYTES == 8 <= bounds.RECORD_HEADER_BYTES
    assert paged_partition.RECORD_COUNTER_BYTES is bounds.RECORD_COUNTER_BYTES
    names.add("RECORD_COUNTER_BYTES")
    for module in (paged_attention_partition, device, transport, paged_partition):
        assert module.__file__ is not None
        tree = ast.parse(Path(module.__file__).read_text())
        # The module's own definitions (a schedule's field of the record's
        # size is no definition of the layout).
        defined = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        } | {
            target.id
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        assert not names & defined, module.__name__
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in names:
                        assert node.module is not None
                        assert node.module.rpartition(".")[2] == (
                            "paged_attention_partition_bounds"
                        )


def _check_balanced(pages, grid, skew, workspace_ctas, limit) -> int:
    """The balanced partition of ``pages``: the definitions' properties, the
    device plan's mirror and the device's arithmetic for a part's unit.
    Returns the units it split."""
    units = len(pages)
    headers = unit_headers(pages)
    total = headers[-1]
    bound = plan_grid_bound(units, limit)
    active = active_ctas(pages, grid, grid_bound=bound, workspace_ctas=workspace_ctas)
    ranges = slot_ranges(
        pages, grid, grid_bound=bound, workspace_ctas=workspace_ctas, skew=skew
    )
    # The active CTAs tile the slots; a skew, at most a thirty-second of the
    # mean share, takes no more than that off a share and puts no more on
    # top (and the slot a floor rounds away).
    assert ranges[0][0] == 0 and ranges[active - 1][1] == total
    assert all(ranges[c][1] == ranges[c + 1][0] for c in range(active - 1))
    sizes = [end - begin for begin, end in ranges[:active]]
    assert min(sizes) >= 1 and ranges[active:] == [(total, total)] * (grid - active)
    share = total // active
    applied_cap = min(skew, share // SKEW_SHARE)
    assert applied_skew(total, active, skew) == (applied_cap if active > 1 else 0)
    assert min(sizes) >= share - share // SKEW_SHARE
    assert max(sizes) <= share + 1 + share // SKEW_SHARE
    ctas = reference_partition(
        pages,
        grid,
        "balanced",
        grid_bound=bound,
        workspace_ctas=workspace_ctas,
        skew=skew,
    )
    seen = {}
    for cta, segments in enumerate(ctas):
        for segment in segments:
            for step in range(segment.step, segment.step + segment.count):
                assert (segment.unit, step) not in seen
                seen[segment.unit, step] = cta
    assert len(seen) == sum(pages)
    assert sorted(s.unit for segments in ctas for s in segments if not s.pages) == [
        unit for unit, count in enumerate(pages) if not count
    ]
    parts = split_units(ctas)
    for cta, segments in enumerate(ctas):
        # At most one part that is not its unit's lowest, the CTA's first
        # page-carrying segment and its first unit, and at most one lowest
        # part, its last: two slots and one counter per CTA and head.
        higher = [s for s in segments if not s.whole and parts[s.unit][0] != cta]
        lowest = [s for s in segments if not s.whole and parts[s.unit][0] == cta]
        carrying = [s for s in segments if s.count]
        assert len(higher) <= 1 and len(lowest) <= 1
        assert not higher or (higher[0] is carrying[0] is segments[0])
        assert not lowest or lowest[0] is carrying[-1]
    plans = planner_plans(pages, grid, 32, limit, "balanced", skew, workspace_ctas)
    for cta, (plan, segments) in enumerate(zip(plans, ctas, strict=True)):
        first, last, count, begin, end, header, slots, planned, applied = plan
        assert (slots, planned) == (total, active)
        assert list(range(first, last)) == [s.unit for s in segments]
        assert count == sum(s.count for s in segments)
        assert applied == applied_skew(total, active, skew)
        if cta < active:
            assert (begin, end) == ranges[cta]
            assert not applied or end - begin >= 31
        if segments:
            assert header == headers[first]
            # A compute group posts the part that does not end with its
            # unit's last step, as the thread that decides its role sees it
            # (``split_arrival``). Only the CTA's first unit can be one: the
            # one whose first page slot lies ahead of the range.
            posts = [
                index
                for index, segment in enumerate(segments)
                if not segment.whole and segment.step + segment.count != segment.pages
            ]
            assert posts == [0] * (begin > header + 1)
            assert bool(posts) == (not segments[0].whole and not segments[0].tail)
        for segment in segments:
            if segment.whole:
                continue
            where = parts[segment.unit]
            slot = headers[segment.unit]
            low = (
                cta
                if segment.tail
                else planner_slot_cta(slot + 1, slots, planned, applied, 32)
            )
            high = (
                cta
                if segment.head
                else planner_slot_cta(slot + segment.pages, slots, planned, applied, 32)
            )
            assert (low, high) == (where[0], where[-1])
            # The owner's walk down from the equal shares' owner is short:
            # the applied skew moves a boundary by at most a quarter of
            # ``applied * (active + 1)`` slots, and is itself at most a
            # thirty-second of the share (two steps on 148 CTAs).
            for target in (slot + 1, slot + segment.pages):
                equal = ((target + 1) * planned - 1) // slots
                owner = planner_slot_cta(target, slots, planned, applied, 32)
                steps = equal - owner
                assert 0 <= steps <= 1 + (applied * (planned + 1) / 4 + 1) / share
                assert planned > 148 or steps <= 2
    return len(parts)


def test_balanced_plan_and_split_arithmetic_match_the_definitions():
    """The balanced plan on the shapes that have bitten and seeded random
    ones, with range skews up to past the legal bound and workspaces smaller
    than the launch: what each CTA runs, what it posts, and where a part's
    unit begins and ends."""
    split = 0
    for pages, grid in (
        ([17], 148),
        ([1, 0, 0], 148),
        ([391], 148),
        ([0, 5, 300], 148),
        ([8192, *[17] * 127], 148),
        ([17] * 200, 148),
        ([17] * 320, 148),
        ([211] * 128, 148),
        # Shares of 458 and 529 slots: a skew of eight whole, of sixteen
        # as fourteen and whole.
        ([211] * 320, 148),
        ([391] * 200, 148),
    ):
        for skew in (0, 2, 8, 16):
            split += _check_balanced(pages, grid, skew, 148, max(pages))
    generator = random.Random(23)
    for _ in range(300):
        units = generator.choice([1, 2, 3, 17, 40, 147, 148, 149, 300, 320])
        # Up to 600 pages: shares long enough for the skews to apply whole.
        limit = generator.choice([3, 17, 64, 600])
        pages = [
            generator.choice([0, generator.randint(0, limit), limit])
            for _unit in range(units)
        ]
        if generator.randrange(5) == 0:
            pages = [limit, *[generator.randint(0, 1) for _unit in range(units - 1)]]
        split += _check_balanced(
            pages,
            generator.choice([1, 2, 3, 8, 20, 148, 149, 400]),
            generator.choice([0, 0, 1, 2, 4, 8, 64]),
            generator.choice([8, 148, 148, 1000]),
            limit,
        )
    assert split > 3000
    # A skew at the mean share would empty the last ranges, on which a launch
    # hangs: 200 units of one page and one of 60 on 148 CTAs share three
    # slots each. A launch applies at most a thirty-second of its mean
    # share, so that one applies none, as does one request alone.
    hung = [1] * 200 + [60]
    assert slot_ranges(hung, 148, skew=4) == slot_ranges(hung, 148)
    assert min(end - begin for begin, end in slot_ranges(hung, 148, skew=4)) == 3
    assert slot_ranges([17], 148, skew=4) == slot_ranges([17], 148)
    # 320 units of a 17-page window on 148 CTAs share 38 slots: a skew of
    # eight, a fifth of the share, is applied as one. (Applied whole it
    # cost 7% of the launch on a B200.) 200 such units share 24: none.
    window = slot_ranges([17] * 320, 148, skew=8)
    assert window == slot_ranges([17] * 320, 148, skew=1)
    sizes = [end - begin for begin, end in window]
    assert sizes[0] - sizes[-1] in (1, 2, 3) and min(sizes) >= 37
    assert slot_ranges([17] * 200, 148, skew=8) == slot_ranges([17] * 200, 148)
    # The same units with all their 211 pages share 458 slots and take it
    # whole: the first CTA runs sixteen slots more than the last. A skew of
    # sixteen is applied as fourteen.
    wide = slot_ranges([211] * 320, 148, skew=8)
    sizes = [end - begin for begin, end in wide]
    assert sizes[0] - sizes[-1] in (15, 16, 17) and min(sizes) >= 449
    assert slot_ranges([211] * 320, 148, skew=16) == slot_ranges(
        [211] * 320, 148, skew=14
    )


def test_any_arrival_order_merges_every_split_unit_once():
    """The exchange between the parts of split units, as interleaved step
    machines: for random lengths, grids, dispatch orders and numbers of
    resident CTAs down to one, every unit is stored once, by a part that
    read only published partials, in the one fold order, and every counter
    is zero afterwards. No machine waits for another CTA's."""
    generator = random.Random(5)
    merged = orders = 0
    for trial in range(400):
        units = generator.choice([1, 2, 3, 17, 40, 149, 300])
        limit = generator.choice([3, 17, 64])
        kind = generator.randrange(4)
        pages = [limit] * units
        if kind == 1:
            pages = [generator.randint(0, limit) for _unit in range(units)]
        elif kind == 2:
            pages = [generator.choice([0, 0, 1, limit]) for _unit in range(units)]
        elif kind == 3:
            pages = [limit, *[generator.randint(0, 1) for _unit in range(units - 1)]]
        grid = generator.choice([1, 2, 3, 8, 20, 148, 400])
        results = simulate_merges(
            pages,
            grid,
            generator.choice([1, 2, 8, 148, grid]),
            trial,
            workspace_ctas=generator.choice([8, 148]),
            skew=generator.choice([0, 2, 8]),
        )
        split = [order for order in results.values() if len(order) > 1]
        merged += len(split)
        orders += sum(len(order) > 2 for order in split)
    assert merged > 1500 and orders > 20
    # One unit over eighteen CTAs: seventeen parts, folded from the unit's
    # first steps down, whichever part arrives last.
    for seed in range(20):
        (order,) = simulate_merges([17], 148, 3, seed, workspace_ctas=148).values()
        assert order == fold_order(list(range(1, 18))) == tuple(range(17, 0, -1))


def test_the_workspace_is_a_write_the_capture_proof_keeps_behind_the_wait():
    """Two back-to-back captured launches share the workspace. Listed with
    the writes only, as the alias plan lists it, the second launch still
    reads its inputs ahead of the dependency wait; listed with the reads as
    well, it would have to wait first, on every second call."""
    key = CaptureKey(context=1, thread=2, stream=3, capture_id=4, graph=5)
    inputs = StorageRange(0, 100, 200)
    outputs = StorageRange(0, 300, 400)
    workspace = StorageRange(0, 500, 600)

    def modes(reads):
        guard = CaptureGuard()
        chain = []
        for node in (1, 2, 3):
            decision = guard.decide(
                key,
                node,
                () if node == 1 else (Dependency(node - 1, 1, 0, 1),),
                LaunchEffects(reads, (outputs, workspace)),
            )
            guard.record(key, node, decision)
            chain.append(decision.mode)
        return chain

    assert modes((inputs,)) == [
        WaitMode.EARLY,
        WaitMode.BEFORE_WRITES,
        WaitMode.BEFORE_WRITES,
    ]
    assert modes((inputs, workspace)) == [WaitMode.EARLY] * 3


def _unlocked() -> None:
    """Another thread's launch must not wait for this one's device work: no
    allocation, fill, stream synchronization, graph bookkeeping or release
    of a buffer holds the module's lock."""
    assert not paged_partition._WORKSPACES.lock.locked()


class _FakeBuffer:
    """What the workspace helper needs of a tensor; ``stream`` is the one
    that was current when the buffer was allocated and filled. ``held``
    takes the stream of a buffer that is released under the module's lock."""

    def __init__(self, nbytes, zeroed, log, stream, held):
        _unlocked()
        self.nbytes, self.zeroed, self.log = nbytes, zeroed, log
        self.stream, self.held = stream, held

    def __del__(self):
        # The last reference returns the buffer to the allocator. Recorded
        # for the fakes to assert: an exception raised here is only printed.
        if paged_partition._WORKSPACES.lock.locked():
            self.held.append(self.stream)

    def data_ptr(self):
        return 4096

    def view(self, ctas, record_bytes):
        assert ctas * record_bytes == self.nbytes
        return self

    def __getitem__(self, index):
        assert index == (
            slice(None),
            slice(None, paged_attention_partition_bounds.RECORD_COUNTER_BYTES),
        )
        return self

    def zero_(self):
        _unlocked()
        self.log.append("counters zeroed in the capture")


class _FakeWorkspaces:
    """The workspace helper on fake streams, captures and buffers."""

    def __init__(self, monkeypatch):
        self.log = []
        # The buffers that were released while the helper held its lock.
        self.held = []
        self.stream = 7
        self.capture = None
        # The fill streams the helper took, and the stream that is current
        # while one of them is selected.
        self.fill_streams = []
        self.selected = None
        self.synchronized = []
        # What runs inside the wait for a spare's fill, if anything.
        self.during_synchronize = None
        # The calling thread's stream capture interaction mode as the
        # driver keeps it (0 is GLOBAL, a thread's default; 1 THREAD_LOCAL),
        # the modes the helper set it to, and the status of an exchange.
        self.capture_mode = 0
        self.exchanges = []
        self.exchange_status = 0
        # The devices of the process, those it has a context on, and the
        # other devices on which the thread's current stream captures.
        self.devices = 1
        self.contexts = {0}
        self.capturing_elsewhere = set()
        self.context_queries = []
        self.tracked = []
        # The cute_cuda_graph that captures, if one does; otherwise the
        # leases raw captures took, by capture, and those whose graphs CUDA
        # has destroyed since, with the looks the helper took at leases.
        self.managed = None
        self.leases = {}
        self.destroyed = set()
        self.looks = 0
        self.like = type(
            "Like", (), {"device": type("Device", (), {"type": "cuda", "index": 0})()}
        )()
        fakes = self
        monkeypatch.setattr(
            paged_partition, "_WORKSPACES", paged_partition._Workspaces()
        )

        def zeroed(device, geometry):
            # A spare is allocated and zeroed with the captures of other
            # threads ignored.
            assert self.selected is None or self.capture_mode == 1
            return _FakeBuffer(
                geometry[0] * geometry[1], True, self.log, self.current(), self.held
            )

        def exchange(mode):
            """``cuThreadExchangeStreamCaptureMode``: swap the thread's mode
            with the one the caller passes by reference."""
            _unlocked()
            if self.exchange_status:
                return self.exchange_status
            self.capture_mode, mode._obj.value = mode._obj.value, self.capture_mode
            self.exchanges.append(self.capture_mode)
            return 0

        self.exchange = exchange
        library = SimpleNamespace(cuThreadExchangeStreamCaptureMode=exchange)
        monkeypatch.setattr(paged_partition, "_zeroed", zeroed)
        monkeypatch.setattr(
            paged_partition, "cuda_driver", lambda: SimpleNamespace(library=library)
        )
        monkeypatch.setattr(
            paged_partition.torch,
            "empty",
            lambda nbytes, dtype, device: _FakeBuffer(
                nbytes, False, self.log, self.current(), self.held
            ),
        )

        def capturing(index):
            if index == self.like.device.index:
                return self.capture is not None
            # Of the other devices only one with a context is asked: the
            # question would create one.
            assert index in self.contexts
            return index in self.capturing_elsewhere

        def has_context(index):
            self.context_queries.append(index)
            return index in self.contexts

        monkeypatch.setattr(paged_partition, "_capturing", capturing)
        monkeypatch.setattr(
            paged_partition.torch.cuda, "device_count", lambda: self.devices
        )
        monkeypatch.setattr(
            paged_partition.torch._C,
            "_cuda_hasPrimaryContext",
            has_context,
            raising=False,
        )
        monkeypatch.setattr(
            paged_partition.torch._C,
            "_cuda_getCurrentRawStream",
            lambda index: self.stream,
            raising=False,
        )
        monkeypatch.setattr(
            paged_partition,
            "cuda_stream_capture_context",
            lambda device: (self.stream, self.capture),
        )

        class FillStream:
            """A stream CUDA creates for the fills of the helper's spares."""

            def __init__(self, device, pointer=None):
                _unlocked()
                assert device is fakes.like.device and fakes.capture_mode == 1
                self.pointer = pointer
                fakes.fill_streams.append(self)

            def synchronize(self):
                _unlocked()
                # Behind the fill, with the launching stream current again
                # and the captures of other threads still ignored.
                assert fakes.selected is None and fakes.capture_mode == 1
                if fakes.during_synchronize is not None:
                    fakes.during_synchronize()
                fakes.synchronized.append(self)

        def pool_stream(device):
            # Other code can hold a stream of PyTorch's pool, and capture a
            # graph on it.
            raise AssertionError("the helper took a stream of PyTorch's pool")

        self.FillStream = FillStream

        @contextlib.contextmanager
        def select(stream):
            _unlocked()
            assert fakes.selected is None and stream in fakes.fill_streams
            fakes.selected = stream
            try:
                yield
            finally:
                fakes.selected = None

        def launching_stream(device):
            raise AssertionError("a launch waited for the stream it launches on")

        monkeypatch.setattr(paged_partition, "_private_stream", FillStream)
        monkeypatch.setattr(paged_partition.torch.cuda, "Stream", pool_stream)
        monkeypatch.setattr(paged_partition.torch.cuda, "stream", select)
        monkeypatch.setattr(
            paged_partition.torch.cuda, "current_stream", launching_stream
        )

        def track(owner, attribute, key, value, tensors):
            _unlocked()
            assert fakes.managed is not None
            fakes.tracked.append((attribute, key))

        def lease(device):
            _unlocked()
            assert device is fakes.like.device and fakes.managed is None
            fakes.leases.setdefault(fakes.capture, []).append(1000 + fakes.capture)
            return fakes.leases[fakes.capture][-1]

        def ended(lease):
            fakes.looks += 1
            if lease not in fakes.destroyed:
                return False
            # Once: the lease is gone with the answer.
            fakes.destroyed.remove(lease)
            return True

        monkeypatch.setattr(paged_partition, "track_cuda_graph_cache_entry", track)
        monkeypatch.setattr(paged_partition, "lease_capturing_graph", lease)
        monkeypatch.setattr(paged_partition, "graph_lease_ended", ended)
        monkeypatch.setattr(
            paged_partition,
            "ACTIVE_CUDA_GRAPH",
            SimpleNamespace(get=lambda: fakes.managed),
        )

    def current(self):
        """The current stream: the launch's, or the fill stream selected."""
        return self.stream if self.selected is None else self.selected

    def workspace(self, ctas=148, record_bytes=83584):
        workspace = paged_partition.paged_partition_workspace(
            self.like, ctas=ctas, record_bytes=record_bytes
        )
        # What the launch dropped (the buffer of a stream it evicted, of a
        # graph CUDA destroyed, one it made and did not keep) went back to
        # the allocator with the lock free.
        assert not self.held
        # The thread has the capture mode it came with.
        assert self.capture_mode == 0
        return workspace

    def kept(self, geometry=(148, 83584)):
        return [key for key in paged_partition._WORKSPACES.eager if key[2] == geometry]


def test_workspace_buffers_follow_streams_and_captures(monkeypatch):
    """One zeroed buffer per device, geometry and eager stream, at most
    sixteen of them; a capture gets its own, the zeroed spare of the last
    eager launch if there is one, which the next eager launch replaces, and
    otherwise a buffer whose counters it zeroes inside the graph. A spare is
    zeroed on the device's fill stream and the launch that makes it waits
    for that stream alone: never for the one it launches on, behind
    whatever the caller queued there (the fakes refuse that wait)."""
    fakes = _FakeWorkspaces(monkeypatch)
    workspace, log, synchronized = fakes.workspace, fakes.log, fakes.synchronized
    eager = workspace()
    assert eager.nbytes == 148 * 83584 and eager.zeroed
    # The launch's own buffer is filled on its stream, ahead of the launch.
    assert eager.stream == 7
    # The first eager launch also zeroed a spare, and waited for its fill.
    (fill,) = fakes.fill_streams
    assert synchronized == [fill]
    assert workspace() is eager and len(synchronized) == 1
    assert workspace(record_bytes=66688) is not eager
    fakes.stream = 8
    other = workspace()
    assert other is not eager and other.zeroed and other.stream == 8
    # A capture adopts the spare: no fill in the graph, and the same buffer
    # for every launch of that capture.
    fakes.capture = 1
    captured = workspace()
    assert captured.zeroed and captured not in (eager, other) and not log
    assert captured.stream is fill
    assert workspace() is captured
    # The capture's buffer is leased on its graph, once.
    assert fakes.leases == {1: [1001]} and not fakes.tracked
    # A second capture finds no spare and zeroes its counters in the graph.
    fakes.capture = 2
    late = workspace()
    assert not late.zeroed and log == ["counters zeroed in the capture"]
    assert workspace() is late and len(log) == 1
    assert fakes.leases == {1: [1001], 2: [1002]} and not fakes.tracked
    # The next eager launch has its buffer and replaces the spare.
    fakes.capture = None
    assert workspace() is other and len(synchronized) == 3
    fakes.capture = 3
    assert workspace().zeroed and len(log) == 1
    fakes.capture = None
    # Eager buffers are bounded per device and geometry. A stream that loses
    # its buffer loses it behind the lock (the fakes refuse a release under
    # it).
    for stream in range(100, 100 + 2 * paged_partition.EAGER_STREAMS):
        fakes.stream = stream
        workspace()
    kept = fakes.kept()
    assert len(kept) == paged_partition.EAGER_STREAMS
    assert kept[-1] == (0, 100 + 2 * paged_partition.EAGER_STREAMS - 1, (148, 83584))
    # The streams used least recently lose theirs: one that launches between
    # the launches of more other streams than are kept keeps its buffer
    # (by order of first use it was evicted after sixteen newer streams and
    # zero-filled a new buffer every time).
    fakes.stream = 50
    busy = workspace()
    fills = len(synchronized)
    for stream in range(200, 200 + 2 * paged_partition.EAGER_STREAMS):
        fakes.stream = stream
        workspace()
        fakes.stream = 50
        assert workspace() is busy
    kept = fakes.kept()
    assert len(kept) == paged_partition.EAGER_STREAMS and kept[-1][1] == 50
    assert kept[-2][1] == 200 + 2 * paged_partition.EAGER_STREAMS - 1
    # The spare is there: no launch waited for another fill.
    assert len(synchronized) == fills
    # One fill stream serves every spare of the device.
    assert fakes.fill_streams == [fill] and set(synchronized) == {fill}


def test_spares_are_zeroed_on_a_stream_cuda_creates_for_the_helper(monkeypatch):
    """PyTorch hands the streams of its pool to whoever asks, so a spare
    zeroed on one could become a node of a graph another thread captures
    there, and the wait for the fill would invalidate that capture. The
    helper's fill stream is one the driver creates for it (the fakes refuse
    a stream of the pool): when the first spare is made, in the device's
    context, non-blocking, one per device and kept for every later spare.
    Where CUDA creates none the launch goes on without a spare, a capture
    zeroes its counters inside its graph, and the next eager launch tries
    again."""
    private_stream = paged_partition._private_stream
    fakes = _FakeWorkspaces(monkeypatch)
    workspace, log, synchronized = fakes.workspace, fakes.log, fakes.synchronized
    spares = paged_partition._WORKSPACES.spares
    events, flags = [], []
    refused = [True]

    class Library:
        cuThreadExchangeStreamCaptureMode = staticmethod(fakes.exchange)

        def cuStreamCreate(self, handle, flag):
            _unlocked()
            # With the captures of other threads ignored, in the device's
            # context, which a memory query made current.
            assert fakes.capture_mode == 1
            assert events[-2:] == [
                f"device {fakes.like.device.index}",
                "memory query",
            ], events
            events.append("create")
            if refused[0]:
                return 1
            flags.append(flag)
            handle._obj.value = 0x5000 + len(flags)
            return 0

    @contextlib.contextmanager
    def device(selected):
        assert selected is fakes.like.device
        events.append(f"device {selected.index}")
        yield
        events.append("device restored")

    cuda = paged_partition.torch.cuda
    monkeypatch.setattr(paged_partition, "_private_stream", private_stream)
    monkeypatch.setattr(
        paged_partition, "cuda_driver", lambda: SimpleNamespace(library=Library())
    )
    monkeypatch.setattr(cuda, "device", device)
    monkeypatch.setattr(cuda, "mem_get_info", lambda: events.append("memory query"))
    monkeypatch.setattr(
        cuda,
        "ExternalStream",
        lambda pointer, device: fakes.FillStream(device, pointer),
        raising=False,
    )
    # CUDA creates no stream: the launch has its buffer and no spare, and a
    # capture zeroes its counters in its graph.
    eager = workspace()
    assert eager.zeroed and eager.stream == 7
    assert events == ["device 0", "memory query", "create", "device restored"]
    assert not fakes.fill_streams and not synchronized and not spares
    fakes.capture = 1
    assert not workspace().zeroed and log == ["counters zeroed in the capture"]
    fakes.capture = None
    # The next eager launch asks again, and zeroes the spare on the stream
    # it gets.
    refused[0] = False
    assert workspace() is eager
    (fill,) = fakes.fill_streams
    assert fill.pointer == 0x5001 and flags == [1]
    assert synchronized == [fill] and len(spares) == 1
    # A capture adopts the spare; the next eager launch makes another on
    # the same stream, which is not created again.
    fakes.capture = 2
    adopted = workspace()
    assert adopted.zeroed and adopted.stream is fill and len(log) == 1
    fakes.capture = None
    assert not spares and workspace() is eager
    assert synchronized == [fill, fill] and len(spares) == 1
    assert fakes.fill_streams == [fill] and events.count("create") == 2
    # While a cute_cuda_graph capture is underway in the process, CUDA
    # refuses a stream synchronization from every thread in the default
    # capture mode: no spare is made, and the next eager launch makes it.
    fakes.capture = 3
    assert workspace().stream is fill and not spares
    fakes.capture = None
    monkeypatch.setattr(launch_caches, "_CAPTURES_UNDERWAY", 1)
    assert workspace() is eager
    assert synchronized == [fill, fill] and not spares
    monkeypatch.setattr(launch_caches, "_CAPTURES_UNDERWAY", 0)
    assert workspace() is eager
    assert synchronized == [fill, fill, fill] and len(spares) == 1
    assert fakes.fill_streams == [fill] and events.count("create") == 2
    # Another device gets a stream of its own.
    fakes.like = type(fakes.like)()
    fakes.like.device = type(fakes.like.device)()
    fakes.like.device.index = 1
    fakes.devices, fakes.contexts = 2, {0, 1}
    monkeypatch.setattr(
        paged_partition.torch._C,
        "_cuda_getCurrentRawStream",
        lambda index: (index, fakes.stream),
        raising=False,
    )
    assert workspace() is not eager
    assert [stream.pointer for stream in fakes.fill_streams] == [0x5001, 0x5002]
    assert events[-4:] == ["device 1", "memory query", "create", "device restored"]
    assert synchronized[-1] is fakes.fill_streams[1] and flags == [1, 1]


def test_a_spare_is_made_with_other_threads_captures_ignored(monkeypatch):
    """The launch that makes a spare puts its thread into CUDA's
    ``THREAD_LOCAL`` capture mode for the spare's stream, allocation, fill
    and wait, and gives the thread back the mode it had: also when the wait
    raises. In the thread's default ``GLOBAL`` mode CUDA refuses those calls
    while any other thread captures a graph in that mode, and the refused
    call invalidates that capture. A launch that makes no spare leaves the
    mode alone, and where CUDA refuses the exchange no spare is made."""
    fakes = _FakeWorkspaces(monkeypatch)
    spares = paged_partition._WORKSPACES.spares
    # The first eager launch makes a spare: THREAD_LOCAL (1) for the fill
    # stream's creation, the fill and the wait, then the thread's own mode.
    eager = fakes.workspace()
    assert fakes.exchanges == [1, 0] and len(spares) == 1
    assert len(fakes.fill_streams) == 1 and len(fakes.synchronized) == 1
    # A launch that finds its buffer and a spare exchanges nothing.
    assert fakes.workspace() is eager and fakes.exchanges == [1, 0]
    # Nor does a capture, which adopts the spare ...
    fakes.capture = 1
    assert fakes.workspace().zeroed and not spares
    fakes.capture = None
    assert fakes.exchanges == [1, 0]
    # ... or a launch while a cute_cuda_graph capture is underway, which
    # makes no spare.
    monkeypatch.setattr(launch_caches, "_CAPTURES_UNDERWAY", 1)
    assert fakes.workspace() is eager and not spares
    monkeypatch.setattr(launch_caches, "_CAPTURES_UNDERWAY", 0)
    assert fakes.exchanges == [1, 0]

    # The wait for the fill raises: the launch raises, publishes no spare
    # and still leaves the thread in the mode it had.
    def refused():
        raise RuntimeError("the wait raised")

    fakes.during_synchronize = refused
    with pytest.raises(RuntimeError, match="the wait raised"):
        paged_partition.paged_partition_workspace(
            fakes.like, ctas=148, record_bytes=83584
        )
    fakes.during_synchronize = None
    assert fakes.capture_mode == 0 and fakes.exchanges == [1, 0, 1, 0]
    assert not spares and len(fakes.synchronized) == 1
    # CUDA refuses the exchange (here CUDA_ERROR_NOT_INITIALIZED): the
    # launch goes on with its buffer, without a spare and without a fill.
    fakes.exchange_status = 3
    assert fakes.workspace() is eager and not spares
    assert fakes.exchanges == [1, 0, 1, 0] and len(fakes.synchronized) == 1
    # The next eager launch asks again and makes the spare.
    fakes.exchange_status = 0
    assert fakes.workspace() is eager and len(spares) == 1
    assert fakes.exchanges == [1, 0, 1, 0, 1, 0] and len(fakes.synchronized) == 2


def test_no_spare_is_made_while_the_thread_captures_on_another_device(monkeypatch):
    """A thread that captures a graph on one device and launches eagerly on
    another makes no spare there: CUDA refuses a thread with a capture of
    its own the spare's zeroing or the wait for it in either capture mode,
    and the refused call invalidates its capture. The launch asks every device the
    process has a context on, and no other: the question would create a
    context. The next eager launch outside the capture makes the spare."""
    fakes = _FakeWorkspaces(monkeypatch)
    spares = paged_partition._WORKSPACES.spares
    fakes.devices = 3
    fakes.contexts = {0, 2}
    eager = fakes.workspace()
    assert len(spares) == 1 and fakes.exchanges == [1, 0]
    # Every device was asked for its context, and device 1, which has none,
    # for nothing else (the fake refuses the question).
    assert fakes.context_queries == [0, 1, 2]
    # A capture adopts the spare; the thread then captures on device 2 and
    # launches eagerly on device 0.
    fakes.capture = 1
    assert fakes.workspace().zeroed and not spares
    fakes.capture = None
    fakes.capturing_elsewhere = {2}
    assert fakes.workspace() is eager
    assert not spares and fakes.exchanges == [1, 0]
    assert len(fakes.fill_streams) == 1 and len(fakes.synchronized) == 1
    # That capture has ended: the next eager launch makes the spare.
    fakes.capturing_elsewhere = set()
    assert fakes.workspace() is eager
    assert len(spares) == 1 and fakes.exchanges == [1, 0, 1, 0]
    assert len(fakes.synchronized) == 2


def test_a_capture_that_starts_while_a_spare_is_made_begins_behind_the_fill(
    monkeypatch,
):
    """The launch that makes a spare asks whether a capture is underway,
    then fills the spare and waits for the fill. A ``cute_cuda_graph``
    capture that another thread starts in between (here inside the wait)
    begins behind the wait: the question, the fill and the wait hold the
    capture gate, which the capture's count waits for. Begun inside, the
    capture would be invalidated by the wait in CUDA's global capture mode.
    The launch itself never waits for the gate: while another thread holds
    it no spare is made, and the next eager launch makes it."""
    fakes = _FakeWorkspaces(monkeypatch)
    spares = paged_partition._WORKSPACES.spares
    events = []

    def synchronized_behind(capture):
        def during():
            capture.start_inside_a_driver_call()
            events.append("fill synchronized")

        return during

    with capture_stand_ins(events) as (gate, graph):
        capture = CaptureOnAnotherThread(gate, graph)
        fakes.during_synchronize = synchronized_behind(capture)
        try:
            eager = fakes.workspace()
        finally:
            fakes.during_synchronize = None
            capture.finish()
        assert events == ["fill synchronized", "capture_begin", "capture_end"]
        assert len(spares) == 1 and len(fakes.synchronized) == 1
        # A capture adopts that spare. With the gate held elsewhere (another
        # thread counts its capture, unloads a library or makes a spare) the
        # next eager launch makes none and does not wait.
        fakes.capture = 1
        assert fakes.workspace().zeroed and not spares
        fakes.capture = None
        gate.waited.clear()
        with gate:
            assert fakes.workspace() is eager
        assert not gate.deadlocks and not gate.waited.is_set()
        assert not spares and len(fakes.synchronized) == 1
        # ... and the launch after that makes it.
        assert fakes.workspace() is eager
        assert len(spares) == 1 and len(fakes.synchronized) == 2
    # A capture whose entry fails is counted no longer: spares are made again.
    fakes.capture = 2
    assert fakes.workspace().zeroed and not spares
    fakes.capture = None
    with capture_stand_ins([], fails=True):
        with (
            pytest.raises(RuntimeError, match="capture_begin failed"),
            launch_caches.cute_cuda_graph(),
        ):
            pytest.fail("the body ran")
        assert not launch_caches.capture_may_be_active()
        assert fakes.workspace() is eager
        assert len(spares) == 1 and len(fakes.synchronized) == 3


def test_a_capture_that_starts_inside_the_refills_question_begins_behind_the_fill(
    monkeypatch,
):
    """The launch that makes a spare asks whether a capture is underway
    under the capture gate: a ``cute_cuda_graph`` capture that another
    thread starts while the question is being answered waits for the gate,
    and begins behind the fill and the wait for it. Asked ahead of the gate,
    the answer would be stale by the time the fill is awaited."""
    fakes = _FakeWorkspaces(monkeypatch)
    spares = paged_partition._WORKSPACES.spares
    events = []
    asked = []
    fakes.during_synchronize = lambda: events.append("fill synchronized")

    with capture_stand_ins(events) as (gate, graph):
        capture = CaptureOnAnotherThread(gate, graph)

        def question():
            asked.append(None)
            if len(asked) == 1:
                capture.start_inside_a_driver_call()
            return False

        cuda = launch_caches.torch.cuda
        monkeypatch.setattr(cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(cuda, "is_current_stream_capturing", question)
        try:
            fakes.workspace()
        finally:
            fakes.during_synchronize = None
            capture.finish()
        assert len(asked) == 1
        assert events == ["fill synchronized", "capture_begin", "capture_end"]
        assert len(spares) == 1 and len(fakes.synchronized) == 1


def test_a_graphs_workspace_is_released_with_the_graph(monkeypatch):
    """A captured graph holds its buffer's address, so the buffer lives as
    long as the graph and no longer. A raw ``torch.cuda.graph`` leases it on
    the graph; when CUDA has destroyed the graph, the next launch that
    captures, or that makes an eager buffer or a spare, returns the buffer
    to the allocator, and a launch that makes nothing takes no look. A
    ``cute_cuda_graph`` owns its buffer instead and takes no lease."""
    fakes = _FakeWorkspaces(monkeypatch)
    workspace, state = fakes.workspace, paged_partition._WORKSPACES
    eager = workspace()

    def capture(number):
        fakes.capture = number
        buffer = workspace()
        assert workspace() is buffer
        fakes.capture = None
        return buffer

    def held():
        return sorted(key[2] for key in state.captured)

    first, second = capture(1), capture(2)
    assert held() == [1, 2] and fakes.leases == {1: [1001], 2: [1002]}
    assert sorted(state.leases) == [1001, 1002]
    # Both graphs live: the eager launch that makes the next spare looks at
    # their leases and releases nothing.
    assert workspace() is eager and fakes.looks > 0 and held() == [1, 2]
    # The first graph is destroyed. A launch that finds its buffer and the
    # spare takes no look and makes nothing.
    fakes.destroyed.add(1001)
    looks = fakes.looks
    assert workspace() is eager and fakes.looks == looks and held() == [1, 2]
    # The next capture releases the destroyed graph's buffer before it takes
    # its own, and a capture again and again holds one buffer, not one each.
    third = capture(3)
    assert third not in (first, second) and held() == [2, 3]
    assert sorted(state.leases) == [1002, 1003]
    for number in range(4, 12):
        fakes.destroyed.add(999 + number)
        capture(number)
        assert held() == [2, number] and len(state.leases) == 2
    # So does an eager launch that makes a buffer: here the spare the first
    # of those captures adopted, and then the buffer of a new stream.
    fakes.destroyed.add(1011)
    assert workspace() is eager and held() == [2]
    fakes.destroyed.add(1002)
    fakes.stream = 9
    assert workspace() is not eager and held() == [] and not state.leases
    # A cute_cuda_graph releases its buffer itself: no lease.
    fakes.managed = object()
    leases = dict(fakes.leases)
    managed = capture(12)
    assert fakes.tracked == [("captured", (0, 9, 12, (148, 83584)))]
    assert fakes.leases == leases and not state.leases
    assert state.captured == {(0, 9, 12, (148, 83584)): managed}


def test_a_lease_ends_once_when_its_semaphore_is_posted():
    """A lease is a semaphore that libc allocated. CUDA posts it from the
    destructor of the user object the captured graph retains, which is
    libc's ``sem_post`` itself, and the poll takes that post once and frees
    the semaphore with it: no CUDA, no interpreter lock and no wait."""
    libc = graph_lease._libc()
    lease = libc.malloc(graph_lease._SEMAPHORE_BYTES)
    assert lease and not libc.sem_init(lease, 0, 0)
    assert not graph_lease.graph_lease_ended(lease)
    assert not graph_lease.graph_lease_ended(lease)
    # What CUDA calls when the graph is gone.
    assert not libc.sem_post(ctypes.c_void_p(lease))
    assert graph_lease.graph_lease_ended(lease)


class _FakeGraphDriver:
    """The driver calls of a lease on a stream that captures a graph: the
    user object created, what the graph retained, and CUDA's destruction of
    the object, which calls its destructor on the pointer it was given."""

    def __init__(self):
        self.objects = []
        self.retained = []

    def cuStreamGetCaptureInfo_v3(
        self, stream, status, _id, graph, _nodes, _edges, _count
    ):
        assert stream == 7
        status._obj.value = 1
        graph._obj.value = 0x6000
        return 0

    def cuUserObjectCreate(self, holder, pointer, destructor, references, flags):
        assert (references, flags) == (1, 1)
        self.objects.append((pointer, destructor.value))
        holder._obj.value = 0x7000 + len(self.objects)
        return 0

    def cuGraphRetainUserObject(self, graph, holder, count, flags):
        assert (count, flags) == (1, 1)
        self.retained.append((graph, holder.value))
        return 0

    def destroy(self):
        pointer, destructor = self.objects.pop()
        ctypes.CFUNCTYPE(None, ctypes.c_void_p)(destructor)(pointer)


# 0: as imported. 1: as ``python -O`` compiles the module, without its
# ``assert`` statements.
@pytest.mark.parametrize("optimize", [0, 1])
def test_a_lease_starts_unposted_whatever_its_memory_held(optimize, monkeypatch):
    """A lease that reads as ended returns its graph's workspace to the
    allocator, under a graph that still uses it. The semaphore is therefore
    initialized whatever the block that libc recycled for it held, and in
    every interpreter mode: inside an ``assert``, which ``python -O``
    drops with all it calls, the initialization left a recycled block that
    was not zero as a lease already ended. A block libc does not give, or a
    semaphore it does not initialize, is an error and not a lease."""
    source = Path(graph_lease.__file__)
    module = {"__name__": graph_lease.__name__, "__package__": graph_lease.__package__}
    exec(compile(source.read_text(), str(source), "exec", optimize=optimize), module)
    libc, driver = module["_libc"](), _FakeGraphDriver()
    capture_driver = object.__new__(CudaCaptureDriver)
    capture_driver.library = driver
    allocate, release = libc.malloc, libc.free
    blocks, freed = [], []

    def recycled(size):
        blocks.append(allocate(size))
        ctypes.memset(blocks[-1], 0xFF, size)
        return blocks[-1]

    def free(block):
        freed.append(block)
        release(block)

    monkeypatch.setattr(libc, "malloc", recycled)
    monkeypatch.setattr(libc, "free", free)
    module["cuda_driver"] = lambda: capture_driver
    module["torch"] = SimpleNamespace(
        cuda=SimpleNamespace(
            device=contextlib.nullcontext,
            current_stream=lambda: SimpleNamespace(cuda_stream=7),
        )
    )
    take, ended = module["lease_capturing_graph"], module["graph_lease_ended"]
    lease = take("cuda:0")
    assert blocks == [lease] and driver.retained == [(0x6000, 0x7001)]
    assert not ended(lease)
    assert not ended(lease) and not freed
    driver.destroy()
    assert ended(lease) and freed == [lease]
    # No semaphore: the block goes back and nothing reaches the driver.
    monkeypatch.setattr(libc, "sem_init", lambda *arguments: -1)
    with pytest.raises(OSError, match="sem_init"):
        take("cuda:0")
    assert freed == blocks and len(blocks) == 2
    monkeypatch.setattr(libc, "malloc", lambda size: None)
    with pytest.raises(MemoryError):
        take("cuda:0")
    assert not driver.objects and len(driver.retained) == 1


# What an ``assert`` of these modules may call: reads that change nothing.
ASSERT_CALLS = {"const_expr", "data_ptr"}


@pytest.mark.parametrize(
    "module",
    [
        "runtime/cute/graph_lease.py",
        "runtime/cute/paged_partition.py",
        "_compiler/cute/paged_attention_partition.py",
        "_compiler/cute/_paged_attention_partition.py",
    ],
)
def test_no_assert_of_the_partition_modules_does_work(module):
    """``python -O`` removes an ``assert`` with everything it calls, so a
    call that does something (an allocation, an initialization, a release)
    may not sit in one: the partition's runtime and device modules assert
    only on values they already have, or on reads that change nothing."""
    tree = ast.parse((TEMPLATE.parents[2] / module).read_text())
    called = {
        ast.unparse(call.func).rpartition(".")[2]
        for node in ast.walk(tree)
        if isinstance(node, ast.Assert)
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    }
    assert called <= ASSERT_CALLS, called


def test_a_workspace_is_shared_by_geometry_not_by_size(monkeypatch):
    """Two kernels whose records differ in size can need the same bytes:
    with 17 and 61 reserved SMs of 148, 131 records of 16 query columns and
    87 of 24. One's counters lie where the other's partials do, which no
    launch leaves zero, so neither an eager stream, a spare nor a capture
    may hand one kernel the other's buffer."""
    narrow = (131, partition_record_bytes(16))
    wide = (87, partition_record_bytes(24))
    assert narrow[0] * narrow[1] == wide[0] * wide[1] and narrow[1] != wide[1]
    fakes = _FakeWorkspaces(monkeypatch)
    eager = [fakes.workspace(*narrow), fakes.workspace(*wide)]
    assert eager[0] is not eager[1] and eager[0].nbytes == eager[1].nbytes
    # Each geometry keeps its buffer on the stream and has a spare of its
    # own, zeroed on the device's one fill stream.
    assert [fakes.workspace(*narrow), fakes.workspace(*wide)] == eager
    assert len(fakes.synchronized) == 2 and len(fakes.fill_streams) == 1
    fakes.capture = 1
    captured = [fakes.workspace(*narrow), fakes.workspace(*wide)]
    assert captured[0] is not captured[1] and not fakes.log
    assert all(buffer.zeroed and buffer not in eager for buffer in captured)
    # Without spares too: each capture zeroes the counters of its own layout.
    fakes.capture = 2
    late = [fakes.workspace(*narrow), fakes.workspace(*wide)]
    assert late[0] is not late[1] and len(fakes.log) == 2
    assert [fakes.workspace(*narrow), fakes.workspace(*wide)] == late


def _call_names(node: ast.AST) -> list[str]:
    return [
        ast.unparse(call.func).rpartition(".")[2]
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    ]


def test_parts_touch_the_workspace_only_behind_the_dependency_wait():
    """The workspace is global memory that the launch before this one may
    still write: a captured launch that overlaps its predecessor reads only
    its inputs ahead of the dependency wait. Read from the template: every
    access to the workspace is in one of six helpers; a compute group
    calls them for a part behind the wait it executes at the part's end;
    the spare warp waits before it arrives; and a group merges a posted
    unit only at the end of its walk, behind the spare warp's reply."""
    helpers = ast.parse(DEVICE_HELPERS.read_text())
    loads = {"_load_pair": "ldcg_v2_f32", "_load_quad": "ldcg_v4_f32"}
    for name, row in loads.items():
        assert row in _call_names(_function(helpers, name))
    accesses = {
        "ld_relaxed_gpu_u32",
        "atom_add_acq_rel_gpu_u32",
        "stg_u32_l2",
        "stg_v2_f32_l2",
        "stg_v4_f32_l2",
        *loads,
        *loads.values(),
    }
    # The counter of a split unit is read, added to and reset in one helper,
    # which the three arrivals call.
    counting = {
        node.name
        for node in helpers.body
        if isinstance(node, ast.FunctionDef)
        and {"ld_relaxed_gpu_u32", "atom_add_acq_rel_gpu_u32", "stg_u32_l2"}
        & set(_call_names(node))
    }
    assert counting == {"_count_arrival"}
    accesses |= counting
    touching = {
        node.name
        for node in helpers.body
        if isinstance(node, ast.FunctionDef)
        and node.name not in {*loads, *counting}
        and accesses & set(_call_names(node))
    }
    assert touching == {
        "split_arrival",
        "store_part",
        "arrive_part",
        "pair_partial",
        "merge_parts",
        "post_parts",
    }
    assert {
        node.name
        for node in helpers.body
        if isinstance(node, ast.FunctionDef) and counting & set(_call_names(node))
    } == {"split_arrival", "arrive_part", "post_parts"}
    template = ast.parse(TEMPLATE.read_text())
    assert not accesses & set(_call_names(template))
    callers = {
        node.name: touching & set(_call_names(node))
        for node in template.body
        if isinstance(node, ast.FunctionDef) and touching & set(_call_names(node))
    }
    assert callers == {
        "_reduce_part": {"pair_partial"},
        "_compute_units": {"split_arrival", "store_part", "arrive_part", "merge_parts"},
        "paged_attention_paired_body": {"post_parts"},
    }
    compute = _function(template, "_compute_units")
    (walk,) = [node for node in compute.body if isinstance(node, ast.While)]
    # A part's epilogue starts with the wait; the arrival's read, the
    # reduction that loads another part, the stores and the merge follow.
    (part,) = [
        node
        for node in ast.walk(walk)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "split"
        and "split_arrival" in _call_names(node)
    ]
    waited, arrival = part.body
    assert ast.unparse(waited) == (
        "if capture_pdl_wait_mode != 0:\n    cute.arch.griddepcontrol_wait()"
    )
    assert ast.unparse(arrival.test) == "thread == 0"
    for name in ("_reduce_part", "store_part", "arrive_part", "merge_parts"):
        (call,) = [
            node
            for node in ast.walk(walk)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == name
        ]
        assert call.lineno > waited.end_lineno
        if name != "_reduce_part":
            # These run for parts only, whose epilogue waited.
            assert any(
                isinstance(node, ast.If)
                and ast.unparse(node.test) == "split"
                and call in ast.walk(node)
                for node in ast.walk(walk)
            )
    # A whole unit's reduction loads no other part: the pair is a part's role.
    reduce = _function(template, "_reduce_part")
    (pairing,) = [
        node
        for node in reduce.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "split"
    ]
    assert ast.unparse(pairing.body[0]) == (
        "pair = cutlass.Int32(walk[WALK_ROLE]) == PART_MERGE_PAIR"
    )
    assert all(
        ast.unparse(node.test) == "pair"
        for node in ast.walk(reduce)
        if isinstance(node, ast.If) and "pair_partial" in _call_names(node)
    )
    # The spare warp's lane waits before its arrival.
    post = _function(helpers, "post_parts")
    (posted,) = [
        node
        for node in ast.walk(post)
        if isinstance(node, ast.If)
        and "_count_arrival" in _call_names(node)
        and ast.unparse(node.test).endswith("== MAIL_PART")
    ]
    assert ast.unparse(posted.body[0]) == (
        "if capture_pdl_wait_mode != 0:\n    cute.arch.griddepcontrol_wait()"
    )
    # The late merge of a posted unit: behind the spare warp's reply, in a
    # group whose every path to the end of its walk waited.
    tail = compute.body[compute.body.index(walk) + 1 :]
    assert sum("merge_parts" in _call_names(statement) for statement in tail) == 1
    (late,) = [
        node
        for statement in tail
        for node in ast.walk(statement)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.Int32(mail[MAIL_KIND]) == MAIL_LAST"
    ]
    assert "merge_parts" in _call_names(late)
    reply = [
        node
        for statement in tail
        for node in ast.walk(statement)
        if isinstance(node, ast.Expr) and "barrier_cta_sync" in _call_names(node)
    ]
    assert len(reply) == 1 and reply[0].lineno < late.lineno
    assert "POSTED_BARRIER + pair_head" in ast.unparse(reply[0])


def test_a_group_posts_by_one_decision():
    """Whether a compute group has posted a part to the spare warp is
    decided once, by the thread that decides the part's role: in the branch
    that takes ``PART_POST`` it notes the post in the group's record. The
    walk arrives at the post's barrier by that role, and the end of the
    walk posts for a group without a part by that note. Decided again
    there, from the plan, the two could drift apart: a group would post
    twice or never, and its CTA would hang at the post's barriers."""
    helpers = ast.parse(DEVICE_HELPERS.read_text())
    template = ast.parse(TEMPLATE.read_text())

    def notes(tree: ast.AST) -> list[str]:
        return [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and "[WALK_POSTED]" in ast.unparse(node.targets[0])
        ]

    decide = _function(helpers, "split_arrival")
    (branch,) = [
        node
        for node in decide.body
        if isinstance(node, ast.If) and "PART_POST" in ast.unparse(node)
    ]
    assert ast.unparse(branch.test) == "step0 + count != pages"
    taken = [ast.unparse(statement) for statement in branch.body]
    for statement in (
        "role = cutlass.Int32(PART_POST)",
        "walk[WALK_POSTED] = cutlass.Int32(1)",
        "mail[MAIL_KIND] = cutlass.Int32(MAIL_PART)",
    ):
        assert statement in taken
    assert not any(
        name in ast.unparse(statement)
        for statement in branch.orelse
        for name in ("PART_POST", "WALK_POSTED", "MAIL_PART")
    )
    # No other helper notes a post, and the template only clears the note:
    # thread 0, in its record of the CTA's first unit, ahead of the
    # barrier that publishes the record.
    assert notes(helpers) == ["walk[WALK_POSTED] = cutlass.Int32(1)"]
    assert notes(template) == ["walk[WALK_POSTED] = cutlass.Int32(0)"]
    compute = _function(template, "_compute_units")
    (walk,) = [node for node in compute.body if isinstance(node, ast.While)]
    (record,) = [
        node
        for node in walk.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "thread == 0"
        and notes(node)
    ]
    (cleared,) = [
        node
        for node in ast.walk(record)
        if isinstance(node, ast.If) and notes(node) and node is not record
        if "const_expr" not in ast.unparse(node.test)
    ]
    assert ast.unparse(cleared.test) == "unit == cutlass.Int32(plan[PLAN_FIRST])"
    following = walk.body[walk.body.index(record) + 1 :]
    assert ast.unparse(following[0]).startswith("cute.arch.barrier(")
    # In the walk the group arrives for a part by its role.
    arrivals = [
        node
        for node in ast.walk(walk)
        if isinstance(node, ast.If)
        and isinstance(node.body[0], ast.Expr)
        and "POST_BARRIER" in ast.unparse(node.body[0])
    ]
    assert [ast.unparse(node.test) for node in arrivals] == ["role == PART_POST"]
    # Behind it the group reads the note back, and nothing of the plan's
    # range decides again what it posts.
    tail = compute.body[compute.body.index(walk) + 1 :]
    text = "\n".join(ast.unparse(statement) for statement in tail)
    assert "posted = cutlass.Int32(walk[WALK_POSTED]) != 0" in text
    assert "PLAN_BEGIN" not in text and "PLAN_HEADER" not in text
    (unposted,) = [
        node
        for statement in tail
        for node in ast.walk(statement)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "not posted"
    ]
    assert "POST_BARRIER + pair_head" in ast.unparse(unposted)
    assert "mail[MAIL_KIND] = cutlass.Int32(MAIL_NONE)" in ast.unparse(unposted)


def _own_calls(function: ast.FunctionDef) -> set[str]:
    """What ``function`` calls, as the source spells it."""
    return {
        ast.unparse(call.func)
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
    }


def test_the_lane_that_decides_a_parts_role_waits_for_no_other_cta():
    """One thread decides what a part of a split unit does, and the
    group's next barrier publishes the role. The decision waits for
    nothing, as no part of the exchange waits for another CTA, and has to
    stay so: a variant whose thread 0 polled the unit's counter there until
    the other parts had arrived never stored the unit (300 units on 148
    CTAs, odd units ragged and even ones empty), while the same poll by all
    32 lanes of warp 0 completed. Why is not established.

    Read from the template: the decision is one call in that thread's
    block; everything the call can run is pinned here by name, so a helper
    that polls cannot be added to it unseen; it looks at the counter once,
    without arriving; and its only loop is the walk to a part's owner under
    a skew, which reads no memory. The spare warp, one lane of which does
    wait (for the launch's dependencies), arrives through the non-aligned
    form only."""
    helpers = ast.parse(DEVICE_HELPERS.read_text())
    arrival = _function(helpers, "split_arrival")
    assert _own_calls(arrival) == {
        "cute.arch.block_idx",
        "cutlass.Int32",
        "_slot_cta",
        "_count_arrival",
    }
    assert not any(isinstance(node, (ast.While, ast.For)) for node in ast.walk(arrival))
    (look,) = [
        call
        for call in ast.walk(arrival)
        if isinstance(call, ast.Call) and ast.unparse(call.func) == "_count_arrival"
    ]
    assert [ast.unparse(keyword.value) for keyword in look.keywords] == ["False"]
    assert [keyword.arg for keyword in look.keywords] == ["arrive"]
    # The counter's helper: one access to read or to add, one to reset, a
    # fence, and no loop. Without ``arrive`` the access is the relaxed load.
    count = _function(helpers, "_count_arrival")
    assert _own_calls(count) == {
        "cutlass.const_expr",
        "cutlass.Int32",
        "cutlass.Int64",
        "fence_acq_rel_gpu",
        "atom_add_acq_rel_gpu_u32",
        "ld_relaxed_gpu_u32",
        "stg_u32_l2",
        "l2_policy_evict_last",
    }
    assert not any(isinstance(node, (ast.While, ast.For)) for node in ast.walk(count))
    for name in ("atom_add_acq_rel_gpu_u32", "ld_relaxed_gpu_u32", "stg_u32_l2"):
        assert _call_names(count).count(name) == 1
    adding, looking = next(
        (node.body, node.orelse)
        for node in count.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.const_expr(arrive)"
    )
    assert "atom_add_acq_rel_gpu_u32" in _call_names(ast.Module(adding, []))
    assert _call_names(ast.Module(looking, [])) == ["Int32", "ld_relaxed_gpu_u32"]
    # The owner walk and the boundary it evaluates are arithmetic.
    owner = _function(helpers, "_slot_cta")
    (walk,) = [
        node for node in ast.walk(owner) if isinstance(node, (ast.While, ast.For))
    ]
    assert ast.unparse(walk.test) == "boundary > slot"
    boundary = _function(helpers, "_slot_boundary")
    assert not any(
        isinstance(node, (ast.While, ast.For)) for node in ast.walk(boundary)
    )
    for function in (owner, boundary):
        assert _own_calls(function) <= {
            "_slot_boundary",
            "cutlass.const_expr",
            "cutlass.Int32",
            "cutlass.Uint32",
            "cutlass.Int64",
        }
    compute = _function(ast.parse(TEMPLATE.read_text()), "_compute_units")
    (deciding,) = [
        node
        for node in ast.walk(compute)
        if isinstance(node, ast.If) and "split_arrival" in _call_names(node.body[0])
    ]
    assert ast.unparse(deciding.test) == "thread == 0" and not deciding.orelse
    (call,) = deciding.body
    assert isinstance(call, ast.Expr) and _call_names(call)[0] == "split_arrival"
    # The group's barriers are ``cute.arch.barrier`` (``bar.sync``); the
    # spare warp's are the non-aligned primitives.
    assert "barrier" in _call_names(
        _function(ast.parse(TEMPLATE.read_text()), "_reduce_part")
    )
    posting = _call_names(_function(helpers, "post_parts"))
    assert "griddepcontrol_wait" in posting and "barrier" not in posting
    assert {"barrier_cta_sync", "barrier_cta_arrive"} <= set(posting)


def test_range_skew_is_a_knob_of_the_balanced_partition():
    """The skew moves the balanced partition's range boundaries: no other
    partition has it, zero and a missing key are one identity, and a bind
    offers only the values its largest possible mean share can apply: up to
    a thirty-second of it, the bound of the device and of the compile-time
    choices alike."""
    assert SKEW_SHARE == 32 == paged_attention_partition_bounds.SKEW_SHARE
    # One definition: the compile-time module and the device template bind
    # the share by importing it from the module that states it, and neither
    # assigns the name (an equal int of their own would be the same object).
    device = importlib.import_module("helion._compiler.cute._paged_attention_partition")
    for module in (paged_attention_partition, device):
        assert module.__file__ is not None
        tree = ast.parse(Path(module.__file__).read_text())
        imported = {
            alias.asname or alias.name: node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert imported.get("SKEW_SHARE") == "paged_attention_partition_bounds"
        assigned = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
        }
        assert "SKEW_SHARE" not in assigned
    every = (0, 1, 2, 4, 8, 16)
    # Units of at most 17 pages, one CTA each: a mean share of 18 slots.
    assert partition_skew_choices(every, 17, 17, 148) == (0,)
    narrow = _bound()
    assert narrow.config_spec.cute_mma_dag_facts.attention.partition_skew_choices == (
        0,
    )
    assert partition_skew_choices(every, 320, 8192, 148) == every
    assert partition_skew_choices(every, 300, 3, 148) == (0,)
    assert partition_skew_choices(every, 17, 1, 148) == (0,)
    # 148 CTAs: 320 requests of 211 pages and 200 of 391 share 458 and 529
    # slots, 128 requests of 211 pages on 128 CTAs 212.
    assert partition_skew_choices(every, 320, 211, 148) == (0, 1, 2, 4, 8)
    assert partition_skew_choices(every, 200, 391, 148) == every
    assert partition_skew_choices(every, 128, 211, 148) == (0, 1, 2, 4)
    choices = WIDE_SKEWS
    sms = WIDE_SHARES["sms"]
    bound = _bound(**WIDE_SHARES)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    assert facts.attention.partition_skew_choices == choices
    legal = {KEY: "balanced", "cute_mma_dag_pass_warps": PAIRED, QUERY: True}
    assert paged_attention_partition_skew_reason(facts, {**legal, SKEW: 8}) is None
    assert paged_attention_partition_skew_reason(facts, legal) is None
    with (
        cpu_render_target(fake_device_sm_count=sms, num_sm=sms),
        bound.env,
        bound.host_function,
    ):
        skewed = bound.config_spec.normalized_config(
            partition_config("balanced", **{SKEW: 8})
        )
        assert (skewed.config[KEY], skewed.config[SKEW]) == ("balanced", 8)
        equal = bound.config_spec.normalized_config(
            partition_config("balanced", **{SKEW: 0})
        )
        assert SKEW not in equal.config
        assert equal == bound.config_spec.normalized_config(
            partition_config("balanced")
        )
        for partition, skew, allowed in (
            ("whole", 8, (0,)),
            (None, 2, (0,)),
            ("balanced", 3, choices),
            ("balanced", True, choices),
            ("balanced", -1, choices),
        ):
            config = partition_config(partition, **{SKEW: skew})
            reason = (
                f"{SKEW} must be one of {allowed!r} for {KEY}={partition or 'item'!r}"
            )
            with pytest.raises(exc.InvalidConfig, match=re.escape(reason)):
                bound.config_spec.normalized_config(config)
            # The repair drops the skew and keeps the partition.
            bound.config_spec.normalize(config, _fix_invalid=True)
            assert SKEW not in config.config
            assert config.config.get(KEY) == partition
    # A skew that no launch of the bind applies whole is not a config of it.
    with cpu_render_target(fake_device_sm_count=148), narrow.env, narrow.host_function:
        reason = f"{SKEW} must be one of (0,) for {KEY}='balanced'"
        with pytest.raises(exc.InvalidConfig, match=re.escape(reason)):
            narrow.config_spec.normalized_config(
                partition_config("balanced", **{SKEW: 2})
            )
    kernel = _function(
        ast.parse(_render(partition_config("balanced", **{SKEW: 8}), **WIDE_SHARES)),
        "_helion_paged_online_attention",
    )
    keywords = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in _body_call(kernel).keywords
    }
    assert keywords["partition_skew"] == "8"
    assert "partition_skew" not in _render(partition_config("balanced"), **WIDE_SHARES)


def test_a_persistent_bind_is_specialized_on_the_sm_count():
    """An attention bind's facts read the SM count the bind saw: whether a
    persistent launch has more than one CTA to offer a partition for, and
    the width of its plan's slot products. The bound kernel is therefore
    specialized on it, and not reused on a device with another count."""
    bound = _bound()
    assert "config_num_sm" in bound.env.compiler_fact_specialization_facts
    assert partition_slot_bits(886, 8192, 148) == 64
    assert partition_slot_bits(886, 8192, 74) == 32


def _quota_sites(partition: str) -> dict[int, list[tuple[int, str]]]:
    """The register quotas each of a CTA's twenty warps takes in the template
    under ``partition``, in order: the line and the call of each.

    A quota is a ``setmaxregister`` statement of the body, or the one a role
    helper starts with, at the helper's call in the body: the tracer inlines
    a helper at every call, so each call is an instruction of its own. A
    role helper may take its quota through one entry helper it starts with
    (the compute roles' ``_compute_group``). The tests around a quota may
    read the warp and the partition only, and are evaluated here for each
    warp."""
    tree = ast.parse(TEMPLATE.read_text())
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    body = functions["paged_attention_paired_body"]

    def quotas(node: ast.AST) -> list[ast.Call]:
        return [
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call) and "setmaxregister" in ast.unparse(call.func)
        ]

    # A helper's quota is one unconditional statement of the helper: the
    # instruction, or the call of the entry helper that holds it.
    helpers = {}
    for name, function in functions.items():
        if function is not body and quotas(function):
            (quota,) = quotas(function)
            assert any(
                isinstance(node, ast.Expr) and node.value is quota
                for node in function.body
            ), name
            helpers[name] = ast.unparse(quota)
    entries = dict(helpers)
    through: list[ast.Call] = []
    for name, function in functions.items():
        entered = [
            call
            for call in ast.walk(function)
            if isinstance(call, ast.Call) and ast.unparse(call.func) in entries
        ]
        if function is not body and entered:
            (taken,) = entered
            assert any(
                isinstance(node, ast.Assign) and node.value is taken
                for node in function.body
            ), name
            helpers[name] = entries[ast.unparse(taken.func)]
            through.append(taken)
    sites: list[tuple[list[tuple[str, bool]], int, str]] = []

    def visit(statements: list[ast.stmt], guards: list[tuple[str, bool]]) -> None:
        for node in statements:
            if isinstance(node, ast.If):
                test = ast.unparse(node.test)
                visit(node.body, [*guards, (test, True)])
                visit(node.orelse, [*guards, (test, False)])
                continue
            taken = []
            for call in ast.walk(node):
                if isinstance(call, ast.Call):
                    name = ast.unparse(call.func)
                    if "setmaxregister" in name:
                        taken.append((call.lineno, ast.unparse(call)))
                    elif name in helpers:
                        taken.append((call.lineno, helpers[name]))
            # Never in a loop or under another kind of block.
            assert not taken or isinstance(node, (ast.Expr, ast.Assign)), taken
            sites.extend((guards, line, quota) for line, quota in taken)

    visit(body.body, [])
    # Every quota of the template is one of those: no other function takes
    # one, and a helper is called from the body alone, or as the one
    # statement through which a role helper takes its quota.
    assert len(sites) == len(quotas(body)) + sum(
        ast.unparse(call.func) in helpers and call not in through
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
    )

    def taken_by(guards: list[tuple[str, bool]], warp: int) -> bool:
        scope = {
            "warp": warp,
            "partition": partition,
            "cutlass": SimpleNamespace(const_expr=lambda value: value),
        }
        return all(bool(eval(test, scope)) is expected for test, expected in guards)

    return {
        warp: [(line, quota) for guards, line, quota in sites if taken_by(guards, warp)]
        for warp in range(20)
    }


def _ptx(config: helion.Config, monkeypatch, directory: Path) -> str:
    """The PTX of the module's default shape under ``config``: the generated
    kernel behind the launcher's own host wrapper, compiled without a GPU
    and without a JIT engine, as the DSL hands it to its assembler."""
    driver = importlib.import_module("cuda.bindings.driver")
    # The launcher's pointer adapter without its CUDA checks: the wrapper is
    # compiled and never called, so CPU addresses only stand in for the ABI.
    monkeypatch.setattr(
        launcher,
        "_get_cute_launcher_imports",
        lambda: (cute.AddressSpace.gmem, make_ptr, None),
    )
    monkeypatch.setattr(launcher, "_validate_cute_launcher_tensor", lambda tensor: None)
    source = directory / "kernel.py"
    source.write_text(_render(config))
    launches = []

    def record(entry, grid, *arguments, block):
        launches.append((entry, grid, arguments, block))

    kernel = partition_attention_kernel()
    with cpu_render_target(fake_device_sm_count=148, forbid_native=False):
        getattr(load_generated_module(source), kernel.name)(
            *partition_attention_args(), _launcher=record
        )
    ((entry, grid, arguments, block),) = launches
    built = launcher._build_cute_schema_and_args(
        entry, tuple(arguments), (*grid, 1, 1)[:3]
    )
    wrapper = launcher._create_cute_wrapper(entry, built.schema, block, num_sm=148)
    with cpu_target(forbid_native=False):
        cute.compile[cute.KeepPTX(True), DumpDir(str(directory))](
            wrapper,
            *built.launch_args,
            driver.CUstream(0),
            no_jit_engine=True,
            no_cache=True,
        )
    (kept,) = directory.glob("*.ptx")
    return kept.read_text()


@pytest.mark.parametrize("partition", ["item", *PARTITIONS])
def test_a_warpgroup_takes_its_quota_from_one_instruction(
    partition, monkeypatch, tmp_path
):
    """``setmaxnreg`` is ``.sync.aligned``: every warp of a warpgroup has to
    execute the same instruction, so a quota belongs to a warpgroup, never
    to one warp's role. Each of a CTA's five warpgroups takes one quota, its
    four warps from the same statement of the template, and the kernel's
    PTX has one instruction per statement and no other: with one unit per
    CTA the two conversion warpgroups have a statement each, and a
    persistent CTA's share one."""
    sites = _quota_sites(partition)
    groups = []
    for group in range(5):
        taken = {tuple(sites[warp]) for warp in range(4 * group, 4 * group + 4)}
        assert len(taken) == 1, (group, taken)
        ((site,),) = taken
        groups.append(site)
    assert [quota for _line, quota in groups] == [
        "cute.arch.setmaxregister_increase(144)",
        "cute.arch.setmaxregister_increase(144)",
        "cute.arch.setmaxregister_decrease(72)",
        "cute.arch.setmaxregister_decrease(72)",
        "cute.arch.setmaxregister_decrease(48)",
    ]
    assert len(set(groups)) == (4 if partition == "item" else 3)
    statements = sorted(
        (action[:3], int(count))
        for _line, quota in set(groups)
        for action, count in re.findall(r"setmaxregister_(\w+)\((\d+)\)", quota)
    )
    instructions = re.findall(
        r"setmaxnreg\.(inc|dec)\.sync\.aligned\.u32\s+(\d+);",
        _ptx(partition_config(partition), monkeypatch, tmp_path),
    )
    assert sorted((action, int(count)) for action, count in instructions) == statements


def _aligned_barriers() -> dict[str, list[str]]:
    """The aligned barriers of each function of the template and of the
    partition's device helpers, as the source spells them."""
    found: dict[str, list[str]] = {}
    for path in (TEMPLATE, DEVICE_HELPERS):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef):
                barriers = [
                    ast.unparse(call)
                    for call in ast.walk(node)
                    if isinstance(call, ast.Call)
                    and ast.unparse(call.func) == "cute.arch.barrier"
                ]
                if barriers:
                    found[node.name] = barriers
    return found


def test_only_the_compute_groups_and_the_join_meet_at_aligned_barriers():
    """The compute groups meet at named barriers of the aligned form
    (``cute.arch.barrier``: ``bar.sync``), which all 128 threads of a group
    reach at one instruction, and the body's all-thread join is that form
    too. A barrier instruction "causes executing thread to wait for all
    non-exited threads from its warp" (PTX ISA 9.5, section 9.7.15.1), so a
    block that only some lanes of a warp run ahead of one (the row owners'
    state, statistics, weights and stores, one thread's record, role
    decision or arrival) needs no warp synchronization of its own. Every
    other role of a persistent CTA publishes through the non-aligned form
    (the plan at barriers 7 and 6, the retirement at barrier 5, the spare
    warp's posts): their participants reach them at different
    instructions."""
    first = "cute.arch.barrier(barrier_id=2 + pair_head * 2, number_of_threads=128)"
    second = "cute.arch.barrier(barrier_id=1 + pair_head * 2, number_of_threads=128)"
    assert _aligned_barriers() == {
        "_compute_begin": [first],
        "_reduce_rows": [second],
        "_compute_reduce": [second],
        "_reduce_part": [second],
        "_compute_units": [first],
        "arrive_part": [second, second],
        "merge_parts": [second, second],
        "paged_attention_paired_body": ["cute.arch.barrier()"],
    }
