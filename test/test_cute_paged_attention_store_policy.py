"""Check that ``cute_store_policy`` reaches the paired attention schedule's
output stores: the default renders the kernel without the option,
``streaming`` renders the same stores with the streaming cache operator and
``l2_evict_last`` under the L2 evict-last hint, with one unit per CTA and on
the persistent partitions, for an output addressed by its own layout and by
proved storage strides; and where a search is offered the three. The
option is the output stores' alone: a balanced launch stores the partials
of its split units into its workspace under the L2 evict-first hint
whatever the option says."""

from __future__ import annotations

import ast
from collections import Counter
import enum
import inspect
from pathlib import Path
import re
import tempfile
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.cute import _paged_attention_io as io
from helion._compiler.cute import _paged_attention_paired_pipeline as template
from helion._compiler.cute import _paged_attention_partition as partition_template
from helion._compiler.cute.cute_mma_dag import paired_output_stores
from helion._compiler.cute.packet_flush_names import streaming_store_rendered
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_folded_attention import folded_attention_args
from helion._testing.cute_offline import compile_offline
from helion._testing.cute_paged_bounded import bounded_partial_fixture
from helion._testing.cute_paged_headwide import headwide_config
from helion._testing.cute_paged_headwide import headwide_kernel
from helion._testing.cute_paged_partition import partition_attention_args
from helion._testing.cute_paged_stream import NO_WINDOW
from helion._testing.cute_paged_stream import SMS
from helion._testing.cute_paged_stream import body_keywords
from helion._testing.cute_paged_stream import stream_bound
from helion._testing.cute_paged_stream import stream_config
from helion._testing.cute_paged_stream import stream_render
from helion._testing.cute_paged_stream import stream_seeds
from helion._testing.cute_ptx_addresses import store_addresses
from helion._testing.cute_template_source import source_function
import helion.language as hl

pytestmark = pytest.mark.cpu_only
KEY = "cute_store_policy"
PARTITION = "cute_mma_dag_partition"
PASS_WARPS = "cute_mma_dag_pass_warps"
# Units on 148 SMs: a grid of more than two waves, which has the persistent
# partitions.
WAVES = 300
# The policies that change the stores, and what each names in the source.
POLICIES = ("streaming", "l2_evict_last")
NAMED = {policy: f", store_policy={policy!r}" for policy in POLICIES}
# A store under a policy as it is in the PTX: the conversion and the store,
# one instruction pair, with the streaming cache operator or with the L2
# cache hint and its policy word.
STORED = {
    "streaming": re.compile(
        r"\{ \.reg \.b16 h; cvt\.rn\.bf16\.f32 h, %\w+; "
        r"st\.global\.cs\.b16 \[%\w+\], h; \}"
    ),
    "l2_evict_last": re.compile(
        r"\{ \.reg \.b16 h; cvt\.rn\.bf16\.f32 h, %\w+; "
        r"st\.global\.L2::cache_hint\.b16 \[%\w+\], h, %\w+; \}"
    ),
}
# The operator or hint a policy's output stores carry, and the instruction
# that makes the ``l2_evict_last`` policy's word. A balanced kernel's
# workspace stores carry the L2 hint under every policy, with policy words
# of their own: its partials' vector stores an evict-first word, its
# counters' resets evict-last ones. The store's kind after the mark picks
# out the output stores (``.b16``, or ``.f32`` for a raw fp32 panel).
MARK = {"streaming": "st.global.cs", "l2_evict_last": "st.global.L2::cache_hint"}
POLICY_WORD = "createpolicy.fractional.L2::evict_last.b64"
# A policy word as the PTX makes it, and a store that carries one: its kind
# (``b16`` an output element, ``v4.f32`` and ``v2.f32`` a partial's panel
# quad and row pair, ``u32`` a counter) and the word's register.
_WORD_MADE = re.compile(r"createpolicy\.fractional\.L2::(evict_\w+)\.b64 (%\w+), 1\.0;")
_HINTED = re.compile(r"st\.global\.L2::cache_hint\.([\w.]+) \[%\w+\], .*, (%\w+);")


def _hinted_stores(ptx: str) -> Counter[tuple[str, str]]:
    """The stores of a kernel under an L2 cache hint, counted by the kind of
    store and the eviction priority of the policy word each is handed."""
    words = {}
    for priority, register in _WORD_MADE.findall(ptx):
        # The assembler's input names a register once: one word, one policy.
        assert register not in words
        words[register] = priority
    stores = _HINTED.findall(ptx)
    assert len(stores) == ptx.count("st.global.L2::cache_hint")
    return Counter((kind, words[register]) for kind, register in stores)


def _opcodes(ptx: str) -> Counter[str]:
    """Instructions of a kernel by opcode, the inline blocks apart."""
    counts: Counter[str] = Counter()
    for line in ptx.splitlines():
        line = line.strip()
        if (
            not line
            or line.startswith((".", "//", "{", "}", "$", "%"))
            or line.endswith(":")
        ):
            continue
        counts[line.split()[0].rstrip(";")] += 1
    return counts


# A template function's definition (``source_function``), by the name this
# module's tests call it.
_function = source_function


def test_the_default_policy_renders_the_kernel_without_the_option():
    body = _function("paged_attention_paired_body")
    defaults = dict(
        zip(
            [argument.arg for argument in body.args.args][-len(body.args.defaults) :],
            [ast.unparse(default) for default in body.args.defaults],
            strict=True,
        )
    )
    assert defaults["store_policy"] == "'default'"
    bound = stream_bound(WAVES, rows=32)
    for partition in ("item", "whole", "balanced"):
        knobs = {} if partition == "item" else {PARTITION: partition}
        plain = stream_render(bound, stream_config(32, **knobs))
        assert "store_policy" not in plain
        assert stream_render(bound, stream_config(32, **knobs, **{KEY: "default"})) == (
            plain
        )
        # Either other policy names itself and changes nothing else of the
        # source.
        for policy in POLICIES:
            source = stream_render(bound, stream_config(32, **knobs, **{KEY: policy}))
            assert body_keywords(source)["store_policy"] == repr(policy)
            assert source.count(NAMED[policy]) == 1
            assert source.replace(NAMED[policy], "") == plain
    # A config keeps the policy it names on the paired schedule: nothing
    # pins it, and it is a field of the search as before.
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        spec = bound.config_spec
        for policy in POLICIES:
            kept = spec.normalized_config(stream_config(32, **{KEY: policy}))
            assert kept.config[KEY] == policy
        assert spec._flat_fields()[KEY].default() == "default"
    # The other attention bodies make their stores as before.
    plain = stream_render(bound, stream_config(32, **OTHER_BODY))
    assert "paged_attention_paired_body" not in plain
    for policy in POLICIES:
        assert stream_render(
            bound, stream_config(32, **OTHER_BODY, **{KEY: policy})
        ) == (plain)


# Another attention body of the stream fixture's kernel, which the paired
# schedule's options do not reach.
OTHER_BODY = {
    PASS_WARPS: "attention",
    "cute_mma_dag_query_tma": False,
    "cute_mma_dag_v_lag": 0,
}


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _pointwise(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + y[tile]
    return out


def test_evict_last_is_offered_wherever_the_kernel_has_the_paired_schedule():
    """The hint's policy word is made at run time, so the fixed descriptor
    that gates the packet flushes' ``l2_evict_last`` does not gate the
    paired schedule's: a kernel with that schedule has the three policies
    as its field on a target without the descriptor, a config on the
    schedule keeps the choice, and a config of the same kernel on another
    body takes the default for it, as every kernel without the schedule
    does."""
    bound = stream_bound(WAVES, rows=32)
    spec = bound.config_spec
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        # The modelled B200 has no descriptor for the packet flushes.
        assert not spec.cute_l2_evict_last_store_policy_supported
        fragment = spec._flat_fields()[KEY]
        assert fragment.choices == ("default", "streaming", "l2_evict_last")
        assert fragment.default() == "default"
        for partition in ("item", "whole", "balanced"):
            knobs = {} if partition == "item" else {PARTITION: partition}
            config = stream_config(32, **knobs, **{KEY: "l2_evict_last"})
            assert spec.normalized_config(config).config[KEY] == "l2_evict_last"
        # A search steps onto the hint from the plain stores and keeps the
        # schedule it is on.
        generation = spec.create_config_generation()
        base, parent = generation.canonicalize_flat(
            generation.flatten(spec.normalized_config(stream_config(32)))
        )
        steps = {
            projection.to_value: projection
            for projection in generation.coordinate_neighbor_projections(base)
            if projection.key == KEY
        }
        assert set(steps) == {"streaming", "l2_evict_last"}
        reached = steps["l2_evict_last"].config
        assert reached is not None and reached.config[KEY] == "l2_evict_last"
        assert reached.config[PASS_WARPS] == parent.config[PASS_WARPS]
        # On another body the choice has only the packet flushes to apply
        # to: it is repaired to the default, strictly as well.
        other = spec.normalized_config(
            stream_config(32, **OTHER_BODY, **{KEY: "l2_evict_last"})
        )
        assert other.config[PASS_WARPS] == "attention"
        assert other.config[KEY] == "default"
        # With the descriptor the choice is the packet flushes' there too.
        with patch.object(
            type(spec),
            "cute_l2_evict_last_store_policy_supported",
            new=property(lambda spec: True),
        ):
            assert spec._flat_fields()[KEY].choices == fragment.choices
            other = spec.normalized_config(
                stream_config(32, **OTHER_BODY, **{KEY: "l2_evict_last"})
            )
            assert other.config[KEY] == "l2_evict_last"
    # The same attention kernel on a device whose shared memory does not
    # hold the paired schedule: it has MMA-DAG facts and other attention
    # bodies, no output stores of this kind, and the packet flushes' field.
    small = stream_bound(128, smem_capacity_bytes=150_000)
    spec = small.config_spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    assert paired_output_stores(facts) is False
    with (
        cpu_render_target(
            fake_device_sm_count=SMS, num_sm=SMS, smem_capacity_bytes=150_000
        ),
        small.env,
        small.host_function,
    ):
        assert not spec.cute_l2_evict_last_store_policy_supported
        assert spec._flat_fields()[KEY].choices == ("default", "streaming")
    # A kernel without the paired schedule: the field is the packet
    # flushes', gated by the descriptor as before.
    arguments = (torch.zeros(64), torch.zeros(64))
    with cpu_render_target(fake_device_sm_count=SMS, num_sm=SMS):
        pointwise = cpu_bind(_pointwise, arguments, num_sm=SMS)
        spec = pointwise.config_spec
        assert spec.cute_mma_dag_facts is None
        with pointwise.env, pointwise.host_function:
            assert spec._flat_fields()[KEY].choices == ("default", "streaming")
            repaired = spec.normalized_config(
                helion.Config(block_sizes=[32], **{KEY: "l2_evict_last"})
            )
            assert repaired.config[KEY] == "default"


# A policy as a member of an enum whose members are strings: of a class
# with the ``str`` mixin and, where the interpreter has it (3.11), of a
# ``StrEnum``. A member compares equal to the policy's name, which is all
# normalization asks, and its repr is not the name's.
_POLICY_MEMBERS = {"STREAMING": "streaming", "L2_EVICT_LAST": "l2_evict_last"}
ENUM_KINDS = {"str-mixin": enum.Enum("StorePolicy", _POLICY_MEMBERS, type=str)}
if hasattr(enum, "StrEnum"):
    ENUM_KINDS["StrEnum"] = enum.StrEnum("StorePolicy", _POLICY_MEMBERS)


@pytest.mark.parametrize("kind", list(ENUM_KINDS))
def test_a_policy_given_as_an_enum_member_renders_the_policys_name(kind):
    bound = stream_bound(128)
    for policy in POLICIES:
        member = ENUM_KINDS[kind](policy)
        assert member == policy and type(member) is not str
        assert repr(member) != repr(policy)
        source = stream_render(bound, stream_config(8, **{KEY: member}))
        assert body_keywords(source)["store_policy"] == repr(policy)
        assert source == stream_render(bound, stream_config(8, **{KEY: policy}))
        # Normalization leaves the policy's name, a plain string, which is
        # what code generation renders.
        target = cpu_render_target(fake_device_sm_count=SMS)
        with target, bound.env, bound.host_function:
            named = bound.config_spec.normalized_config(
                stream_config(8, **{KEY: member})
            )
        assert type(named.config[KEY]) is str and named.config[KEY] == policy


def test_a_windowed_kernel_seeds_the_hint_on_the_top_seed_of_each_partition():
    """Behind every other seed: the leading family under the hint, and
    where the units outnumber the SMs its balanced form, in source order
    and in the reversed order that partition is seeded with. A kernel
    without a window has no such seed."""

    def hinted(seeds: list[helion.Config]) -> list[helion.Config]:
        named = [seed for seed in seeds if seed.config.get(KEY) == "l2_evict_last"]
        assert seeds[len(seeds) - len(named) :] == named
        return named

    def without(seed: helion.Config) -> dict[str, object]:
        return {key: value for key, value in seed.config.items() if key != KEY}

    # One wave: one unit per CTA is the only seeded partition.
    seeds = stream_seeds(stream_bound(128, rows=32))
    (item,) = hinted(seeds)
    assert without(item) == seeds[0].config
    assert KEY not in seeds[0].config and PARTITION not in seeds[0].config
    # More units than SMs: the balanced seeds of the leading family as
    # well, the one in source order and the one in reversed order.
    seeds = stream_seeds(stream_bound(WAVES, rows=32))
    item, balanced, walked = hinted(seeds)
    assert without(item) == seeds[0].config
    partitioned, reversed_order = [
        seed
        for seed in seeds
        if seed.config.get(PARTITION) == "balanced" and KEY not in seed.config
    ]
    assert without(balanced) == partitioned.config
    assert without(balanced) == {**seeds[0].config, PARTITION: "balanced"}
    assert without(walked) == reversed_order.config
    assert without(walked) == {
        **partitioned.config,
        "cute_mma_dag_page_order": "reversed",
        "cute_mma_dag_clipped_page_lead": 8,
    }
    # Every seed renders, the new ones with the hint.
    bound = stream_bound(WAVES, rows=32)
    for seed in (item, balanced, walked):
        with (
            cpu_render_target(fake_device_sm_count=SMS),
            bound.env,
            bound.host_function,
        ):
            strict = bound.config_spec.normalized_config(seed)
        assert strict.config[KEY] == "l2_evict_last"
        keywords = body_keywords(stream_render(bound, strict))
        assert keywords["store_policy"] == "'l2_evict_last'"
    # No window: no seed names a store policy.
    for units in (128, WAVES):
        seeds = stream_seeds(stream_bound(units, rows=32, window=NO_WINDOW))
        assert not any(KEY in seed.config for seed in seeds)


def test_the_policy_reaches_every_output_store_of_the_template():
    """One function stores a work item's O panel, for the one-unit CTA and
    for a persistent CTA's units and merged units, and every caller hands
    it the body's policy."""
    store = _function("_compute_store")
    assert [argument.arg for argument in store.args.args][-1] == "store_policy"
    calls = [
        call
        for call in ast.walk(ast.parse(Path(inspect.getfile(template)).read_text()))
        if isinstance(call, ast.Call) and ast.unparse(call.func) == "_compute_store"
    ]
    assert len(calls) == 3
    assert all(ast.unparse(call.args[-1]) == "store_policy" for call in calls)
    for name in ("_compute_head", "_compute_units"):
        function = _function(name)
        assert [argument.arg for argument in function.args.args][-1] == "store_policy"
        (call,) = [
            call
            for call in ast.walk(_function("paged_attention_paired_body"))
            if isinstance(call, ast.Call) and ast.unparse(call.func) == name
        ]
        assert ast.unparse(call.args[-1]) == "store_policy"
    # The plain stores are the function they were, and the ones under a
    # policy address the same element and are handed the policy.
    streamed = [
        call
        for call in ast.walk(store)
        if isinstance(call, ast.Call) and ast.unparse(call.func) == "store_output_rows"
    ]
    plain = [
        call
        for call in ast.walk(store)
        if isinstance(call, ast.Call)
        and ast.unparse(call.func) == "store_output_element"
    ]
    assert len(streamed) == len(plain) == 1
    assert ast.unparse(streamed[0].args[-1]) == "store_policy"
    assert "store_policy" not in ast.unparse(_function("store_output_element", io))
    # One policy word a thread, made ahead of its stores and not per store.
    rows = _function("store_output_rows", io)
    words = [
        node
        for node in ast.walk(rows)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "l2_policy_evict_last"
    ]
    assert len(words) == 1
    # One loop over the rows for both output layouts and both policies.
    (loop,) = [node for node in ast.walk(rows) if isinstance(node, ast.For)]
    assert words[0] not in ast.walk(loop)
    stores = [
        ast.unparse(node.func)
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) in ("streamed", "hinted")
    ]
    assert sorted(stores) == ["hinted", "streamed"]
    # The per-row stores carry no policy.
    assert "store_policy" not in ast.unparse(_function("store_row_element", io))


# The stream fixture's kernels: one unit per CTA at 8 and 40 query columns
# and with 16 query rows, fewer than the 32 columns their block has, and the
# persistent partitions at 32, whose balanced one stores a unit from two
# places (its own units and the units it merges).
KERNELS = {
    "item-8": (8, None, 1),
    "item-16": (16, None, 1),
    "item-40": (40, None, 1),
    "whole-32": (32, "whole", 1),
    "balanced-32": (32, "balanced", 2),
}


@pytest.mark.parametrize("kernel", list(KERNELS))
def test_policy_stores_differ_by_the_cache_operator_or_hint_alone(kernel):
    """The output stores under each policy, and with them, from the same
    compiles, what the kernel stores into a balanced launch's workspace
    (``_workspace_stores_keep_their_own_policies``)."""
    rows, partition, sites = KERNELS[kernel]
    units = 128 if partition is None else WAVES
    bound = stream_bound(units, rows=rows)
    arguments = partition_attention_args(units=units, rows=rows)
    knobs = {} if partition is None else {PARTITION: partition}
    compiled = {}
    with tempfile.TemporaryDirectory(prefix="helion-cute-stores-") as dump_dir:
        for policy in ("default", *POLICIES):
            source = stream_render(bound, stream_config(rows, **knobs, **{KEY: policy}))
            compiled[policy] = compile_offline(
                bound, arguments, source, Path(dump_dir) / policy
            ).ptx
    _workspace_stores_keep_their_own_policies(kernel, compiled)
    plain = compiled["default"]
    stores = rows * sites
    # The plain kernel converts and stores each O element of a thread with
    # the default operator and has neither a streaming nor a hinted store.
    assert plain.count("st.global.b16") == stores
    assert not any(f"{mark}.b16" in plain for mark in MARK.values())
    assert len(re.findall(r"cvt\.rn\.bf16\.f32", plain)) == stores
    before = _opcodes(plain)
    for policy in POLICIES:
        ptx = compiled[policy]
        # Each of them is one conversion and store with the policy's
        # operator or hint, and no plain 16-bit store is left.
        assert len(STORED[policy].findall(ptx)) == stores
        assert ptx.count(f"{MARK[policy]}.b16") == stores
        assert not any(
            f"{MARK[other]}.b16" in ptx for other in POLICIES if other != policy
        )
        assert "st.global.b16" not in ptx
        assert len(re.findall(r"cvt\.rn\.bf16\.f32", ptx)) == stores
        # Store for store they go to the plain kernel's addresses, which
        # are distinct: the address arithmetic of both, evaluated.
        addresses = store_addresses(plain, "st.global.b16")
        assert len(set(addresses)) == stores
        assert store_addresses(ptx, f"{MARK[policy]}.b16") == addresses
        # The per-row LSE stores are the plain ones in both.
        for instruction in ("st.global.b32", "st.global.f32"):
            assert ptx.count(instruction) == plain.count(instruction)
        after = _opcodes(ptx)
        moved = {
            opcode: after[opcode] - before[opcode]
            for opcode in before.keys() | after.keys()
            if after[opcode] != before[opcode]
        }
        # Outside the inline pairs the conversions and stores are gone.
        assert moved.pop("cvt.rn.bf16.f32") == moved.pop("st.global.b16") == -stores
        # The hint's policy word: one a thread at each place that stores a
        # unit, ahead of its stores (the balanced partition's workspace
        # stores make their own in the plain kernel too).
        assert moved.pop(POLICY_WORD, 0) == (sites if policy == "l2_evict_last" else 0)
        if sites == 1:
            # Nothing else differs but the stores' addresses, which the
            # plain form folds into its operand: one addition a store but
            # the first. (Around them the assembler may form one index in
            # the other width: an addition for a conversion.)
            assert moved.pop("add.s64") == stores - 1, (policy, moved)
            assert moved in ({}, {"add.s32": -1, "cvt.u64.u32": 1}), (policy, moved)
        else:
            # A merging kernel also forms a row's address once where the
            # plain stores form it per element: integer arithmetic only.
            assert all(
                opcode.split(".")[0]
                in ("add", "sub", "and", "or", "not", "shl", "shr", "mul", "cvt", "mov")
                for opcode in moved
            ), (policy, moved)
            assert not any(".f32" in opcode or "f32" in opcode for opcode in moved)


# What a balanced kernel stores into its launch workspace at 32 query
# columns: a part's partial, one quad of its panel a thread and row quad
# and a row's pair, once in the kernel, and the reset of a unit's counter
# wherever a thread reads or adds to one (the lowest part's look, a part's
# own arrival, and the spare warp's arrival for each head's posted part).
PARTIAL_STORES = {("v4.f32", "evict_first"): 8, ("v2.f32", "evict_first"): 1}
COUNTER_RESETS = {("u32", "evict_last"): 4}


def _workspace_stores_keep_their_own_policies(
    kernel: str, compiled: dict[str, str]
) -> None:
    """A part's partial takes a hint of its own, a trade that the backend's
    README states with ``cute_mma_dag_partition``: its stores into the
    workspace carry the L2 evict-first hint under every
    ``cute_store_policy``, with one policy word a thread. The output
    stores carry the config's policy, and the counters' resets the
    evict-last hint. A kernel that splits no unit, with one unit per CTA
    or on the whole-units partition, stores nothing of the kind."""
    rows, partition, sites = KERNELS[kernel]
    for policy in ("default", *POLICIES):
        ptx = compiled[policy]
        expected: dict[tuple[str, str], int] = {}
        made: dict[str, int] = {}
        if policy == "l2_evict_last":
            # One word at each place that stores a unit's outputs.
            expected["b16", "evict_last"] = rows * sites
            made["evict_last"] = sites
        if partition == "balanced":
            assert rows == 32
            expected.update(PARTIAL_STORES)
            expected.update(COUNTER_RESETS)
            # One word for the partial's nine stores, and one a reset.
            made["evict_first"] = 1
            made["evict_last"] = made.get("evict_last", 0) + 4
        assert _hinted_stores(ptx) == expected, (kernel, policy)
        words = Counter(priority for priority, _register in _WORD_MADE.findall(ptx))
        assert words == made, (kernel, policy)
        assert ptx.count("createpolicy") == sum(made.values())
        # The streaming operator is the output stores' alone.
        streamed = rows * sites if policy == "streaming" else 0
        assert ptx.count("st.global.cs") == streamed, (kernel, policy)


def test_two_functions_of_the_template_write_the_workspace():
    """Every store into a balanced launch's workspace is in two functions
    of the partition template: ``store_part``, a part's partial, and
    ``_count_arrival``, a counter's arrival (an atomic add, which takes no
    hint) and its reset, under evict-last. Neither reads the body's store
    policy. The partial's policy word is counted in the compiled kernels
    (``_workspace_stores_keep_their_own_policies``)."""
    tree = ast.parse(Path(inspect.getfile(partition_template)).read_text())
    # What the template can write global memory with.
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "device.ptx"
        for alias in node.names
    }
    writers = {name for name in imported if name.startswith(("stg_", "atom_"))}
    assert writers == {
        "atom_add_acq_rel_gpu_u32",
        "stg_u32_l2",
        "stg_v2_f32_l2",
        "stg_v4_f32_l2",
    }
    called: dict[str, list[tuple[str, ast.Call]]] = {}
    for function in tree.body:
        if isinstance(function, ast.FunctionDef):
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and ast.unparse(node.func) in writers:
                    called.setdefault(function.name, []).append(
                        (ast.unparse(node.func), node)
                    )
    assert sorted(called) == ["_count_arrival", "store_part"]
    # The partial: a quad of its panel and a row's pair.
    assert sorted(name for name, _call in called["store_part"]) == [
        "stg_v2_f32_l2",
        "stg_v4_f32_l2",
    ]
    # The counter: the arrival and the reset.
    assert sorted(name for name, _call in called["_count_arrival"]) == [
        "atom_add_acq_rel_gpu_u32",
        "stg_u32_l2",
    ]
    (reset,) = [call for name, call in called["_count_arrival"] if name == "stg_u32_l2"]
    assert ast.unparse(reset.args[-1]) == "l2_policy_evict_last()"
    assert "store_policy" not in ast.unparse(tree)


def test_a_folded_output_takes_a_policy_by_its_storage_strides():
    """The fixture whose outputs are proved from explicit storage indices
    (the layout of the SDG kernel's): forty stores a thread, each with the
    policy's cache operator or hint."""
    query_len = 5
    arguments = folded_attention_args(query_len)
    with cpu_render_target(fake_device_sm_count=SMS, num_sm=SMS):
        bound = cpu_bind(headwide_kernel(), arguments, num_sm=SMS)
    compiled = {}
    with tempfile.TemporaryDirectory(prefix="helion-cute-stores-") as dump_dir:
        for policy in ("default", *POLICIES):
            config = headwide_config(query_len)
            config.config[KEY] = policy
            with cpu_render_target(fake_device_sm_count=SMS, num_sm=SMS):
                source = bound.to_code(config)
            keywords = body_keywords(source)
            assert "io_layouts" in keywords
            assert keywords.get("store_policy") == (
                None if policy == "default" else repr(policy)
            )
            compiled[policy] = compile_offline(
                bound, arguments, source, Path(dump_dir) / policy
            ).ptx
    stores = 8 * query_len
    assert compiled["default"].count("st.global.b16") == stores
    assert not any(f"{mark}.b16" in compiled["default"] for mark in MARK.values())
    addresses = store_addresses(compiled["default"], "st.global.b16")
    assert len(set(addresses)) == stores
    for policy in POLICIES:
        assert len(STORED[policy].findall(compiled[policy])) == stores
        assert "st.global.b16" not in compiled[policy]
        # The same padded rows, by the storage's strides.
        assert store_addresses(compiled[policy], f"{MARK[policy]}.b16") == addresses
    assert compiled["l2_evict_last"].count(POLICY_WORD) == 1
    assert POLICY_WORD not in compiled["streaming"]


def test_a_raw_statistics_epilogue_stores_its_fp32_panel_under_a_policy():
    """The epilogue that stores each unit's unnormalized fp32 panel beside
    its row maxima and denominators: the panel's stores take the operator
    or the hint, the per-row ones do not."""
    kernel, initial = bounded_partial_fixture(4)
    arguments = (*initial[:7], 31, *initial[8:])
    fast = helion.kernel(kernel.fn, backend="cute", static_shapes=True, fast_math=True)
    with cpu_render_target(fake_device_sm_count=SMS, num_sm=SMS):
        bound = cpu_bind(fast, arguments, num_sm=SMS)
    compiled = {}
    with tempfile.TemporaryDirectory(prefix="helion-cute-stores-") as dump_dir:
        for policy in ("default", *POLICIES):
            config = stream_config(32, **{KEY: policy})
            with cpu_render_target(fake_device_sm_count=SMS, num_sm=SMS):
                source = bound.to_code(config)
            keywords = body_keywords(source)
            assert "maximum_output" in keywords
            assert ("store_policy" in keywords) is (policy != "default")
            compiled[policy] = compile_offline(
                bound, arguments, source, Path(dump_dir) / policy
            ).ptx
    plain = compiled["default"]
    assert not any(f"{mark}.f32" in plain for mark in MARK.values())
    # Thirty-two fp32 elements a thread, which the plain kernel stores as
    # 32-bit words, and the two statistics of a row.
    assert plain.count("st.global.b32") == 34
    # The panel's elements and the two statistics, four bytes each, at
    # distinct addresses.
    addresses = sorted(store_addresses(plain, "st.global.b32"))
    assert len(set(addresses)) == 34
    for policy in POLICIES:
        ptx = compiled[policy]
        assert ptx.count(f"{MARK[policy]}.f32") == ptx.count(MARK[policy]) == 32
        assert not any(MARK[other] in ptx for other in POLICIES if other != policy)
        assert ptx.count("st.global.b32") == 2
        # The kernel under a policy stores to the same 34 places.
        assert (
            sorted(
                store_addresses(ptx, f"{MARK[policy]}.f32")
                + store_addresses(ptx, "st.global.b32")
            )
            == addresses
        )
    assert compiled["l2_evict_last"].count(POLICY_WORD) == 1
    assert POLICY_WORD not in compiled["streaming"]


def test_the_final_selections_tie_rule_covers_the_bodys_streaming_stores():
    """Among finalists it cannot tell apart, the final selection ranks a
    kernel that renders streaming stores three ranks back
    (``CuteBackend.finalist_preference``). The paired body flushes no
    packet, whose helper names say so for other kernels: its kernel is
    known by the policy that its call of the body names
    (``streaming_store_rendered``). A paired finalist under ``streaming``
    ranks three behind the same config under ``default``, with the early
    release and without it, and one under ``l2_evict_last`` ranks as under
    ``default``."""
    named = "store_policy='streaming'"
    # One wave, whose release is the loader's tail, and more than two, whose
    # release is the generic one; ``cute_pdl=False`` declines either.
    for units in (128, WAVES):
        bound = stream_bound(units, rows=32)
        for release, rank in (({}, 1), ({"cute_pdl": False}, 2)):
            ranks = {}
            for policy in ("default", *POLICIES):
                config = stream_config(32, **release, **{KEY: policy})
                source = stream_render(bound, config)
                assert body_keywords(source).get("store_policy") == (
                    None if policy == "default" else repr(policy)
                )
                assert streaming_store_rendered(source) is (policy == "streaming")
                with (
                    cpu_render_target(fake_device_sm_count=SMS),
                    bound.env,
                    bound.host_function,
                ):
                    strict = bound.config_spec.normalized_config(config)
                assert strict.config.get(KEY, "default") == policy
                assert strict.config["cute_pdl"] is not bool(release)
                ranks[policy] = bound.env.backend.finalist_preference(
                    strict, lambda source=source: source
                )
                if policy != "streaming":
                    continue
                # The reader takes the policy from the body's own call:
                # another policy there, or the keyword on another call, is
                # no streaming store.
                assert source.count(named) == 1
                for other in (
                    source.replace(named, "store_policy='l2_evict_last'"),
                    source.replace(named, "store_policy=policy"),
                    source.replace("paged_attention_paired_body(", "another_body("),
                ):
                    assert not streaming_store_rendered(other)
            assert ranks == {
                "default": rank,
                "l2_evict_last": rank,
                "streaming": rank + 3,
            }, (units, release, ranks)
