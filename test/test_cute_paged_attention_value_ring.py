"""Check the depth of the paired schedule's raw V ring: the shared-memory
table a depth gives, which depths a device admits, what a config may name
and what it renders, and where a search samples and seeds it."""

from __future__ import annotations

import ast
import copy
import dataclasses
from pathlib import Path
import re
import tempfile

import cutlass
import pytest

from helion import exc
from helion._compiler.cute import _paged_attention_paired_pipeline as template
from helion._compiler.cute.cute_mma_dag import paged_attention_value_ring_choices
from helion._compiler.cute.cute_mma_dag import paged_attention_value_ring_facts
from helion._compiler.cute.cute_mma_dag_seeds import VALUE_RING_SEED_PAGES
from helion._compiler.cute.paged_attention_config import (
    paged_attention_value_ring_reason,
)
from helion._compiler.cute.paged_attention_shared_memory import KEY_RING_PAGES
from helion._compiler.cute.paged_attention_shared_memory import (
    MMA_DAG_VALUE_RING_CHOICES,
)
from helion._compiler.cute.paged_attention_shared_memory import MMA_DAG_VALUE_RING_KEY
from helion._compiler.cute.paged_attention_shared_memory import (
    PAIRED_DEFAULT_VALUE_RING,
)
from helion._compiler.cute.paged_attention_shared_memory import RAW_TILE_BYTES
from helion._compiler.cute.paged_attention_shared_memory import SCALE_TILE_BYTES
from helion._compiler.cute.paged_attention_shared_memory import (
    paired_shared_memory_bytes,
)
from helion._compiler.cute.paged_attention_shared_memory import pipeline_barrier_plan
from helion._compiler.cute.paged_attention_shared_memory import pipeline_parts
from helion._compiler.cute.paged_attention_shared_memory import pipeline_shared_bytes
from helion._compiler.cute.paged_attention_shared_memory import pipeline_shared_memory
from helion._compiler.cute.paged_attention_shared_memory import value_ring_reason
from helion._compiler.cute.target_facts import B200
from helion._testing import cpu_render_target
from helion._testing.cute_offline import compile_offline
from helion._testing.cute_offline import partition_bound
from helion._testing.cute_paged_stream import NO_WINDOW
from helion._testing.cute_paged_stream import PAIRED
from helion._testing.cute_paged_stream import SMS
from helion._testing.cute_paged_stream import body_keywords
from helion._testing.cute_paged_stream import seeds_ahead_of_the_reversed_order
from helion._testing.cute_paged_stream import stream_bound
from helion._testing.cute_paged_stream import stream_config
from helion._testing.cute_paged_stream import stream_render
from helion._testing.cute_paged_stream import stream_seeds
from helion._testing.cute_paged_stream import stream_step
from helion._testing.cute_paged_stream import transferred_seeds
from helion._testing.cute_template_source import source_function

pytestmark = pytest.mark.cpu_only
KEY = MMA_DAG_VALUE_RING_KEY
LEAD = "cute_mma_dag_release_lead"
PARTITION = "cute_mma_dag_partition"
SKEW = "cute_mma_dag_partition_skew"
SKIP = "cute_mma_dag_value_metadata_skip"
V_LAG = "cute_mma_dag_v_lag"
# Shared memory one CTA of the modelled B200 may request.
CAPACITY = 232448
# Units on 148 SMs: a grid of one wave, and one of more than two, whose
# seeds include the balanced partition.
ONE_WAVE = 128
WAVES = 300
# A device that grants a CTA 256 KiB: a persistent CTA holds a third page.
ROOMY = 262144
# One that grants 200,000 bytes: the paired schedule fits, a third page not.
TIGHT = 200000


def _allocated(
    query_columns: int,
    *,
    value_ring: int = 2,
    query_map: bool = False,
    query_stages: int = 0,
    units: int = 0,
    balanced: bool = False,
    scanned_bounds: bool = False,
    prefetch_signal: bool = False,
) -> int:
    """Bytes the DSL's allocator places the template's table in: each
    array at the next offset of its alignment, by its element type."""
    arrays = pipeline_shared_memory(
        query_columns,
        False,
        query_stages,
        units,
        balanced,
        prefetch_signal,
        value_ring,
    )
    parts = pipeline_parts(query_map, False, units, scanned_bounds, balanced)
    offset = 0
    for array in arrays:
        if array.guard is not None and array.guard not in parts:
            continue
        width = getattr(cutlass, array.dtype).width // 8
        alignment = max(array.alignment, width)
        offset = -(-offset // alignment) * alignment + width * array.count
    return offset


def test_the_table_the_contract_replays_is_the_one_the_body_allocates():
    assert B200.smem_capacity_bytes == CAPACITY
    # Every layout a body can have, at both depths: the byte count the
    # config contract computes is what the template's typed table occupies.
    layouts = [
        {},
        {"query_map": True},
        {"prefetch_signal": True},
        {"query_map": True, "prefetch_signal": True},
        {"query_stages": 1, "units": 17, "scanned_bounds": True},
        {"query_stages": 2, "units": 17, "scanned_bounds": True},
        {"query_stages": 2, "units": 640, "scanned_bounds": True, "balanced": True},
        {"query_stages": 2, "units": 641, "balanced": True},
    ]
    for query_columns in (8, 32, 40):
        for layout in layouts:
            for ring in MMA_DAG_VALUE_RING_CHOICES:
                assert paired_shared_memory_bytes(
                    query_columns, value_ring=ring, **layout
                ) == _allocated(query_columns, value_ring=ring, **layout)
    # Each entry names its element type as the DSL does, with its width.
    for ring in MMA_DAG_VALUE_RING_CHOICES:
        for array in pipeline_shared_memory(40, False, 2, 17, True, False, ring):
            assert getattr(cutlass, array.dtype).width == 8 * array.width
    # One more page per head: two raw tiles with their scales and, in a
    # block of 32 barriers, a second round for the four ring barriers. A
    # block that already has a second round keeps it.
    tiles = 2 * (RAW_TILE_BYTES + SCALE_TILE_BYTES)
    assert tiles == 33792
    assert paired_shared_memory_bytes(40) == 178952
    assert paired_shared_memory_bytes(40, value_ring=3) == 178952 + tiles + 32 * 8
    assert paired_shared_memory_bytes(32, value_ring=3) == 204296
    assert (
        paired_shared_memory_bytes(
            40, value_ring=3, query_map=True, prefetch_signal=True
        )
        == paired_shared_memory_bytes(40, query_map=True, prefetch_signal=True) + tiles
    )
    persistent = {"query_stages": 2, "units": 17, "scanned_bounds": True}
    assert paired_shared_memory_bytes(40, value_ring=3, **persistent) == (
        paired_shared_memory_bytes(40, **persistent) + tiles
    )
    for ring, slots in ((2, 32), (3, 64)):
        offsets, count, _rounds = pipeline_barrier_plan(False, 0, False, ring)
        assert count == slots
        assert offsets["raw_v_free"] - offsets["raw_v_ready"] == 2 * ring
        assert offsets["key_ready"] - offsets["raw_v_free"] == 2 * ring
        assert offsets["raw_v_ready"] == 8
        # The K ring's barriers of each kind: its own depth's, two heads.
        assert offsets["raw_k_ready"] == 0
        assert offsets["raw_k_free"] == 2 * KEY_RING_PAGES
        assert offsets["raw_v_ready"] - offsets["raw_k_free"] == 2 * KEY_RING_PAGES
    # The K ring keeps its two pages per head, the depth at which the
    # template's K roles take their slots, at every depth of the V ring.
    assert KEY_RING_PAGES == 2
    for ring in MMA_DAG_VALUE_RING_CHOICES:
        arrays = {a.name: a for a in pipeline_shared_memory(40, False, value_ring=ring)}
        assert arrays["raw_k"].count == 2 * KEY_RING_PAGES * RAW_TILE_BYTES
        assert arrays["scale_k"].count == 2 * KEY_RING_PAGES * SCALE_TILE_BYTES
        assert arrays["raw_v"].count == 2 * ring * RAW_TILE_BYTES
        assert arrays["scale_v"].count == 2 * ring * SCALE_TILE_BYTES
    assert pipeline_shared_bytes(
        pipeline_shared_memory(40, False), pipeline_parts(False, False)
    ) == paired_shared_memory_bytes(40)


def test_a_device_admits_a_depth_its_shared_memory_holds():
    assert KEY == "cute_mma_dag_value_ring"
    assert MMA_DAG_VALUE_RING_CHOICES == (2, 3)
    assert PAIRED_DEFAULT_VALUE_RING == MMA_DAG_VALUE_RING_CHOICES[0] == 2
    # One unit per CTA: with every optional part such a body can have (the
    # query map's barrier, a bucketed page order's tables), the widest panel
    # takes 214,936 bytes of the B200's 232,448.
    assert value_ring_reason(2, CAPACITY, 40) is None
    assert value_ring_reason(3, CAPACITY, 40) is None
    assert value_ring_reason(3, 214936, 40) is None
    short = value_ring_reason(3, 214935, 40)
    assert short is not None and "needs 214936 bytes" in short and "214935" in short
    # At 32 query columns the same parts take 206,232 bytes.
    assert value_ring_reason(3, 206232, 32) is None
    assert value_ring_reason(3, 206231, 32) is not None
    assert (
        paired_shared_memory_bytes(
            40, value_ring=3, query_map=True, prefetch_signal=True, page_order=True
        )
        == 214936
    )
    # An unknown capacity (a render without a device) admits every depth,
    # as it admits the schedule; no other value is a depth.
    assert value_ring_reason(3, 0, 40) is None
    for value in (4, 1, 0):
        assert "not one of (2, 3)" in str(value_ring_reason(value, CAPACITY, 40))
    # A persistent CTA: two query stages leave no room on a B200, whatever
    # the unit count, and one stage would.
    for units in (17, 640, 641, 5000):
        for balanced in (False, True):
            two = value_ring_reason(
                3, CAPACITY, 40, query_stages=2, units=units, balanced=balanced
            )
            assert two is not None and "with 2 query stages" in two
            assert (
                value_ring_reason(
                    3, CAPACITY, 40, query_stages=1, units=units, balanced=balanced
                )
                is None
            )
            assert (
                value_ring_reason(
                    3, ROOMY, 40, query_stages=2, units=units, balanced=balanced
                )
                is None
            )

    def persistent(columns: int, units: int, balanced: bool) -> int:
        return paired_shared_memory_bytes(
            columns,
            value_ring=3,
            query_stages=2,
            units=units,
            balanced=balanced,
            scanned_bounds=True,
        )

    # What two query stages take beside three pages per head. At 40 query
    # columns the least is 233,912 bytes (whole units) and 234,072
    # (balanced), more than the B200 grants; at 32 columns the most is
    # 225,976 and 226,136, at 640 units, the last with one unit per scanning
    # thread, and a grid of 128 units takes 221,880 and 222,040.
    counts = (1, 17, 128, 200, 320, 640, 641, 1280, 100000)
    for balanced, least_40, most_32, at_128 in (
        (False, 233912, 225976, 221880),
        (True, 234072, 226136, 222040),
    ):
        assert min(persistent(40, units, balanced) for units in counts) == least_40
        assert most_32 == persistent(32, 640, balanced)
        assert most_32 == max(persistent(32, units, balanced) for units in counts)
        assert persistent(32, 128, balanced) == at_128
        assert least_40 > CAPACITY > most_32
        for units in counts:
            assert (
                value_ring_reason(
                    3, CAPACITY, 32, query_stages=2, units=units, balanced=balanced
                )
                is None
            )
    # The bind's facts carry the capacity, and the choices follow it and
    # the config's partition and value transport: here for 40 query columns.
    bound = stream_bound(WAVES, rows=40)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    attention = facts.attention
    assert attention.smem_capacity == CAPACITY and attention.query_columns == 40
    assert attention.partition_choices == ("item", "whole", "balanced")
    assert attention.partition_q_stages == 2
    assert paged_attention_value_ring_choices(attention, PAIRED) == (2, 3)
    assert paged_attention_value_ring_choices(attention, PAIRED, False, "item") == (
        2,
        3,
    )
    for partition in ("whole", "balanced", "other"):
        assert paged_attention_value_ring_choices(
            attention, PAIRED, False, partition
        ) == (2,)
    assert paged_attention_value_ring_choices(attention, PAIRED, True) == (2,)
    assert paged_attention_value_ring_choices(attention, "attention") == (2,)
    assert paged_attention_value_ring_choices(None, PAIRED) == (2,)
    assert paged_attention_value_ring_choices(
        dataclasses.replace(attention, paired_items=False), PAIRED
    ) == (2,)
    # Narrower query panels leave a persistent CTA room for the third page:
    # at 32 columns it takes 223,416 of the B200's bytes.
    for rows, needed in ((32, 223416), (8, 185016)):
        narrower = stream_bound(WAVES, rows=rows).config_spec.cute_mma_dag_facts
        assert narrower is not None and narrower.attention is not None
        assert narrower.attention.query_columns == rows
        for partition in ("item", "whole", "balanced"):
            assert paged_attention_value_ring_choices(
                narrower.attention, PAIRED, False, partition
            ) == (2, 3)
        assert (
            paired_shared_memory_bytes(
                rows, value_ring=3, query_stages=2, units=WAVES, scanned_bounds=True
            )
            == needed
        )
    # Room is not enough: the partition has to be one the bind offers. A
    # kernel without fast_math has no balanced partition, so a config that
    # names one has no ring whose depth it could name.
    exact = stream_bound(ONE_WAVE, fast_math=False).config_spec.cute_mma_dag_facts
    assert exact is not None and exact.attention is not None
    assert exact.attention.partition_choices == ("item", "whole")
    assert (
        paged_attention_value_ring_facts(exact.attention, PAIRED, False, "balanced")
        is None
    )
    for partition, depths in (("whole", (2, 3)), ("balanced", (2,))):
        assert (
            paged_attention_value_ring_choices(
                exact.attention, PAIRED, False, partition
            )
            == depths
        )
    roomy = stream_bound(WAVES, rows=40, smem_capacity_bytes=ROOMY).config_spec
    assert roomy.cute_mma_dag_facts is not None
    wide = roomy.cute_mma_dag_facts.attention
    assert wide is not None and wide.smem_capacity == ROOMY
    for partition in ("item", "whole", "balanced"):
        assert paged_attention_value_ring_choices(wide, PAIRED, False, partition) == (
            2,
            3,
        )
    tight = stream_bound(ONE_WAVE, rows=40, smem_capacity_bytes=TIGHT).config_spec
    assert tight.cute_mma_dag_facts is not None
    narrow = tight.cute_mma_dag_facts.attention
    assert narrow is not None and narrow.paired_items
    assert narrow.smem_capacity == TIGHT
    assert paged_attention_value_ring_choices(narrow, PAIRED) == (2,)
    # A device without room has no field and no seed.
    assert tight.cute_mma_dag_value_ring is None
    assert KEY not in tight._flat_fields()


def test_the_default_depth_renders_the_kernel_without_the_option():
    body = source_function("paged_attention_paired_body")
    defaults = dict(
        zip(
            [argument.arg for argument in body.args.args][-len(body.args.defaults) :],
            [ast.unparse(default) for default in body.args.defaults],
            strict=True,
        )
    )
    assert defaults["value_ring"] == "PAIRED_DEFAULT_VALUE_RING"
    assert defaults["shared_memory_capacity"] == "0"
    bound = stream_bound(ONE_WAVE)
    plain = stream_render(bound, stream_config())
    assert stream_render(bound, stream_config(**{KEY: 2})) == plain
    assert "value_ring" not in plain and "shared_memory_capacity" not in plain
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        spec = bound.config_spec
        named = spec.normalized_config(stream_config(**{KEY: 2}))
        assert KEY not in named.config
        assert named == spec.normalized_config(stream_config())
        assert KEY not in spec.default_config().config


@pytest.mark.parametrize("v_lag", [0, 1, 2, 3])
def test_a_deeper_ring_renders_its_depth_and_the_capacity_it_was_admitted_on(v_lag):
    bound = stream_bound(ONE_WAVE)
    plain = stream_render(bound, stream_config(**{V_LAG: v_lag}))
    source = stream_render(bound, stream_config(**{KEY: 3, V_LAG: v_lag}))
    keywords = body_keywords(source)
    assert keywords["value_ring"] == "3"
    assert keywords["shared_memory_capacity"] == str(CAPACITY)
    assert keywords.get("v_lag", "0") == str(v_lag)
    # Nothing else of the kernel changes: the depth runs under every V lag.
    assert (
        source.replace(f", value_ring=3, shared_memory_capacity={CAPACITY}", "")
        == plain
    )


def test_the_loader_and_the_v_conversion_take_the_ring_in_turns_of_its_depth():
    slot, phase = template._ring_slot, template._ring_phase
    for depth in (2, 3, 4):
        uses: dict[int, int] = {}
        for step in range(8 * depth):
            for head in (0, 1):
                ring = slot(step, head, depth)
                # The pages take the depth's slots in turn, two heads each,
                # and a slot's barriers alternate phase from use to use.
                assert ring == 2 * (step % depth) + head
                assert phase(step, depth) == uses.get(ring, 0) % 2
                uses[ring] = uses.get(ring, 0) + 1
        assert sorted(uses) == list(range(2 * depth))
        assert len(set(uses.values())) == 1
    # The default depth keeps the expressions the body always traced.
    assert [slot(step, 1, 2) for step in range(4)] == [1, 3, 1, 3]
    assert [phase(step, 2) for step in range(6)] == [0, 0, 1, 1, 0, 0]
    helpers = {
        name: ast.unparse(source_function(name))
        for name in ("_ring_slot", "_ring_phase")
    }
    assert "return 2 * (step & 1) + head" in helpers["_ring_slot"]
    assert "return step // 2 & 1" in helpers["_ring_phase"]

    def text(name: str) -> str:
        return ast.unparse(source_function(name))

    # One request helper serves K and V; each request names its ring's depth.
    enqueue = text("_enqueue_operand_tile")
    assert "ring = _ring_slot(step, pair_head, depth)" in enqueue
    assert "_wait_non_mma(free + ring, _ring_phase(step, depth) ^ 1, sleep_waits)" in (
        enqueue
    )
    requests = [
        call
        for name in ("_load_pages", "_drain_values")
        for call in ast.walk(source_function(name))
        if isinstance(call, ast.Call)
        and ast.unparse(call.func) == "_enqueue_operand_tile"
    ]
    by_ring = {"raw_k": [], "raw_v": []}
    for call in requests:
        depth = {k.arg: ast.unparse(k.value) for k in call.keywords}.get("depth")
        by_ring[ast.unparse(call.args[0])].append(depth)
    assert by_ring == {"raw_k": ["KEY_RING_PAGES"] * 2, "raw_v": ["value_ring"] * 3}
    values = text("_convert_values")
    assert "ring = _ring_slot(step, head, value_ring)" in values
    assert "_ring_phase(step, value_ring)" in values
    # The K ring keeps the default depth: its conversion takes its slot and
    # phase from the same helpers, at that depth. The depth is the K ring's
    # own constant, as in the K requests above: the V ring's default names
    # no slot of the K ring.
    keys = text("_convert_keys")
    assert "ring = _ring_slot(step, head, KEY_RING_PAGES)" in keys
    assert "_ring_phase(step, KEY_RING_PAGES)" in keys
    assert "value_ring" not in keys and "VALUE_RING" not in keys
    # The value-metadata actor walks the ring at its default depth, and the
    # body refuses a deeper one beside it.
    assert "ring = 2 * (step & 1) + pair_head" in text("_prepare_value_metadata")
    body = text("paged_attention_paired_body")
    assert "assert cutlass.const_expr(value_metadata_scales is None)" in body
    # Every role that walks the ring gets the body's depth.
    tree = source_function("paged_attention_paired_body")
    for callee, count in (
        ("_load_pages", 1),
        ("_load_units", 1),
        ("_drain_values", 1),
        ("_convert_values", 2),
    ):
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == callee
        ]
        assert len(calls) == count
        for call in calls:
            passed = dict(
                zip(
                    [argument.arg for argument in source_function(callee).args.args],
                    [ast.unparse(argument) for argument in call.args],
                    strict=False,
                )
            )
            assert passed["value_ring"] == "value_ring", callee


def test_a_depth_must_fit_the_partition_and_the_value_transport():
    bound = stream_bound(WAVES, rows=40)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    legal = {KEY: 3, "cute_mma_dag_pass_warps": PAIRED}
    assert paged_attention_value_ring_reason(facts, legal) is None
    assert paged_attention_value_ring_reason(facts, {}) is None
    assert paged_attention_value_ring_reason(facts, {KEY: 2}) is None
    for value in (4, 1, 0, True, "3", 3.0):
        reason = paged_attention_value_ring_reason(facts, {**legal, KEY: value})
        assert reason is not None and "(2, 3)" in reason
    # A persistent partition's CTA: the reason says what it would need.
    for partition in ("whole", "balanced"):
        reason = paged_attention_value_ring_reason(
            facts, {**legal, PARTITION: partition}
        )
        assert reason is not None
        assert f"{KEY} must be one of (2,) for" in reason
        assert re.search(
            rf"needs 23\d\d\d\d bytes of shared memory per CTA with 2 query stages "
            rf"and the device grants {CAPACITY}",
            reason,
        )
    for change in ({"cute_mma_dag_pass_warps": "attention"}, {SKIP: True}):
        reason = paged_attention_value_ring_reason(facts, {**legal, **change})
        assert reason is not None and "must be one of (2,) for" in reason
        assert "bytes" not in reason
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        spec = bound.config_spec
        assert spec.normalized_config(stream_config(40, **{KEY: 3})).config[KEY] == 3
        # Each is rejected, and repaired to the same config without the
        # depth: no other option changes.
        for ring, change in (
            (3, {PARTITION: "whole"}),
            (3, {PARTITION: "balanced"}),
            (4, {}),
            (
                3,
                {
                    "cute_mma_dag_pass_warps": "attention",
                    "cute_mma_dag_query_tma": False,
                },
            ),
        ):
            config = stream_config(40, **{KEY: ring}, **change)
            with pytest.raises(exc.InvalidConfig, match=re.escape(KEY)):
                spec.normalized_config(config)
            spec.normalize(config, _fix_invalid=True)
            assert KEY not in config.config
            assert config == spec.normalized_config(stream_config(40, **change))
        off = spec.normalized_config(stream_config(40, **{KEY: 3}, cute_mma_dag="off"))
        assert KEY not in off.config
        assert off == spec.normalized_config(stream_config(40, cute_mma_dag="off"))
    # A device whose shared memory does not hold the depth: the config is
    # rejected where it is normalized, with the bytes it would take.
    tight = stream_bound(ONE_WAVE, rows=40, smem_capacity_bytes=TIGHT)
    with (
        cpu_render_target(fake_device_sm_count=SMS, smem_capacity_bytes=TIGHT),
        tight.env,
        tight.host_function,
    ):
        config = stream_config(40, **{KEY: 3})
        with pytest.raises(
            exc.InvalidConfig,
            match=rf"needs 214936 bytes of shared memory per CTA and the device "
            rf"grants {TIGHT}",
        ):
            tight.config_spec.normalized_config(config)
        tight.config_spec.normalize(config, _fix_invalid=True)
        assert KEY not in config.config
    # A device with room for it renders the depth on a persistent partition.
    roomy = stream_bound(WAVES, rows=40, smem_capacity_bytes=ROOMY)
    with cpu_render_target(
        fake_device_sm_count=SMS, num_sm=SMS, smem_capacity_bytes=ROOMY
    ):
        keywords = body_keywords(
            roomy.to_code(stream_config(40, **{KEY: 3, PARTITION: "balanced"}))
        )
    assert keywords["partition"] == "'balanced'"
    assert keywords["value_ring"] == "3"
    assert keywords["shared_memory_capacity"] == str(ROOMY)


def test_the_body_refuses_a_table_its_capacity_does_not_hold():
    """The render's own gate: the body replays the table it allocates
    against the capacity it is given, so a kernel that reaches it with a
    depth the device does not hold fails where it is compiled, not where it
    is launched."""
    bound, args = partition_bound()
    config = stream_config(**{KEY: 3, "block_sizes": [64]})
    with cpu_render_target(fake_device_sm_count=SMS, num_sm=SMS):
        source = bound.to_code(config)
    capacity = f"shared_memory_capacity={CAPACITY}"
    assert source.count(capacity) == 1
    with tempfile.TemporaryDirectory(prefix="helion-cute-ring-") as dump_dir:
        deeper = compile_offline(bound, args, source, Path(dump_dir) / "fits")
        # This body has the query map and no prefetch barrier: 213,008 bytes.
        exact = source.replace(capacity, "shared_memory_capacity=213008")
        compile_offline(bound, args, exact, Path(dump_dir) / "exact")
        short = source.replace(capacity, "shared_memory_capacity=213007")
        with pytest.raises(AssertionError, match=r"\(213008, 213007\)"):
            compile_offline(bound, args, short, Path(dump_dir) / "short")
        plain = compile_offline(
            bound,
            args,
            source.replace(f", value_ring=3, {capacity}", ""),
            Path(dump_dir) / "plain",
        )
    # Six raw V tiles where the default has four: the ring's slots are
    # taken modulo three.
    assert deeper.ptx != plain.ptx
    assert deeper.threads == plain.threads == 640


def test_the_depth_is_searched_everywhere_and_seeded_beside_the_longer_lead():
    eager = stream_bound(ONE_WAVE, window=NO_WINDOW)
    fragment = eager.config_spec._flat_fields()[KEY]
    assert fragment is eager.config_spec.cute_mma_dag_value_ring
    # The depth changes the loader's own schedule, which every timing of
    # the kernel shows: a search samples it with or without bank cycles.
    assert fragment.choices == fragment._active_choices() == (2, 3)
    assert fragment.default() == 2
    banks = stream_bound(ONE_WAVE, window=NO_WINDOW, input_banks=2)
    assert banks.config_spec._flat_fields()[KEY]._active_choices() == (2, 3)
    # No seed names it without the longer lead of the tail release, which
    # only a search over bank cycles samples: alone the depth gains nothing.
    assert not any(KEY in seed.config for seed in stream_seeds(eager))
    assert VALUE_RING_SEED_PAGES == 3
    seeds = stream_seeds(banks)
    head = seeds[0]
    assert head.get("cute_mma_dag_pass_warps") == PAIRED
    assert KEY not in head.config and head.config[V_LAG] == 2
    # The leading paired family with the longer lead, then once more with
    # three pages per head and V issued with its K, behind every other seed.
    named = [seed for seed in seeds if KEY in seed.config]
    assert named == seeds[-1:]
    assert named[0].config == {**head.config, LEAD: 16, KEY: 3, V_LAG: 0}
    assert seeds[-2].config == {**head.config, LEAD: 16}
    # Neither for a windowed kernel nor past one wave, where no seed names
    # the lead. (A windowed kernel's only seeds with the depth are in
    # reversed page order, with the clipped page ahead of the walk's end
    # and V two tiles behind K: test_cute_paged_attention_reversed_order.)
    for other in (
        stream_bound(ONE_WAVE, input_banks=2),
        stream_bound(WAVES, window=NO_WINDOW, input_banks=2),
    ):
        assert other.config_spec._flat_fields()[KEY]._active_choices() == (2, 3)
        assert not any(
            KEY in seed.config
            for seed in seeds_ahead_of_the_reversed_order(stream_seeds(other))
        )
        assert all(
            (seed.config["cute_mma_dag_clipped_page_lead"], seed.config[V_LAG])
            == (8, 2)
            for seed in stream_seeds(other)
            if KEY in seed.config
        )
    for bound in (eager, banks):
        spec = bound.config_spec
        with (
            cpu_render_target(fake_device_sm_count=SMS),
            bound.env,
            bound.host_function,
        ):
            assert KEY not in spec.default_config().config
            generation = spec.create_config_generation()
            normalized = spec.normalized_config(named[0])
            assert normalized.config[KEY] == 3
            _flat, transferred = generation.canonicalize_flat(
                generation.flatten(normalized)
            )
            assert transferred.config[KEY] == 3
    assert body_keywords(stream_render(banks, named[0]))["value_ring"] == "3"


def test_a_step_onto_a_depth_takes_one_unit_per_cta():
    bound = stream_bound(WAVES, window=NO_WINDOW, rows=40)
    spec = bound.config_spec
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        plain = stream_config(40)
        balanced = stream_config(40, **{PARTITION: "balanced"})
        for parent in (plain, balanced):
            reached, alone = stream_step(spec, parent, **{KEY: 3})
            assert reached.config[KEY] == 3
            assert PARTITION not in spec.normalized_config(reached).config
            assert reached.config[V_LAG] == 2
            # Alone the step is an alias of where it started from a
            # partition whose CTA has no room: normalization repairs the
            # depth and changes nothing else.
            if parent is plain:
                assert alone == reached
            else:
                assert KEY not in alone.config
                assert alone.config[PARTITION] == "balanced"
        on = spec.normalized_config(stream_config(40, **{KEY: 3}))
        # A step back is the knob's own.
        reached, alone = stream_step(spec, on, **{KEY: 2})
        assert reached == alone and KEY not in spec.normalized_config(reached).config
        # A step onto a partition from a deeper ring is the partition's: it
        # drops the depth its CTA has no room for.
        reached, _alone = stream_step(spec, on, **{PARTITION: "balanced"})
        moved = spec.normalized_config(reached)
        assert moved.config[PARTITION] == "balanced" and KEY not in moved.config
        # The depth pins no V lag and takes none: the search crosses them.
        reached, alone = stream_step(spec, on, **{V_LAG: 0})
        assert reached == alone and spec.normalized_config(reached).config[KEY] == 3
    # Where the partition's CTA holds the depth, the step takes one unit per
    # CTA all the same: a persistent CTA is slower with a deeper ring, so a
    # search samples the depth with one unit per CTA alone.
    roomy = stream_bound(WAVES, window=NO_WINDOW, rows=40, smem_capacity_bytes=ROOMY)
    with (
        cpu_render_target(fake_device_sm_count=SMS, smem_capacity_bytes=ROOMY),
        roomy.env,
        roomy.host_function,
    ):
        spec = roomy.config_spec
        balanced = stream_config(40, **{PARTITION: "balanced"})
        reached, alone = stream_step(spec, balanced, **{KEY: 3})
        moved = spec.normalized_config(reached)
        assert moved.config[KEY] == 3 and PARTITION not in moved.config
        # Alone the step is the row of the partition at the default depth:
        # an alias of where it started.
        _same, unmoved = stream_step(spec, balanced)
        assert alone == unmoved
        assert KEY not in alone.config and alone.config[PARTITION] == "balanced"
        # A step onto a partition from a deeper ring is the partition's
        # here as well, and so is a draw that moves both.
        on = spec.normalized_config(stream_config(40, **{KEY: 3}))
        for partition in ("whole", "balanced"):
            reached, _alone = stream_step(spec, on, **{PARTITION: partition})
            assert reached.config[PARTITION] == partition and KEY not in reached.config
            reached, _alone = stream_step(
                spec, stream_config(40), **{KEY: 3, PARTITION: partition}
            )
            assert reached.config[PARTITION] == partition and KEY not in reached.config


def test_a_search_samples_a_deeper_ring_with_one_unit_per_cta_alone():
    """A persistent CTA that holds three pages per head is slower with them
    (the README has the measurement). The config contract still admits the
    depth there, for a config or a fixed override that names it; a search's
    rows on a persistent partition hold the default depth, so no draw, step,
    seed or cached config makes it time one."""
    # Thirty-two query columns on a B200, where a persistent CTA holds the
    # depth, and forty on a device that has room for it there too.
    for bound, rows, capacity in (
        (stream_bound(WAVES, window=NO_WINDOW, rows=32), 32, CAPACITY),
        (
            stream_bound(WAVES, window=NO_WINDOW, rows=40, smem_capacity_bytes=ROOMY),
            40,
            ROOMY,
        ),
    ):
        spec = bound.config_spec
        attention = spec.cute_mma_dag_facts.attention
        partitions = ("item", "whole", "balanced")
        assert attention.partition_choices == partitions
        with (
            cpu_render_target(fake_device_sm_count=SMS, smem_capacity_bytes=capacity),
            bound.env,
            bound.host_function,
        ):
            for partition in partitions:
                # The contract: every partition's CTA holds the depth.
                assert paged_attention_value_ring_choices(
                    attention, PAIRED, False, partition
                ) == (2, 3)
                named = stream_config(rows, **{KEY: 3, PARTITION: partition})
                kept = spec.normalized_config(named)
                assert kept.config[KEY] == 3
                assert kept.config.get(PARTITION, "item") == partition
            generation = spec.create_config_generation()
            slots = {
                key: entry[0][0]
                for key, entry in generation._key_to_flat_indices.items()
            }
            base = generation.canonicalize_flat(
                generation.flatten(stream_config(rows))
            )[0]
            # Every row a search can make of the two knobs: the depth stays
            # with one unit per CTA, and a persistent partition's row is the
            # one at the default depth.
            sampled = set()
            for partition in partitions:
                rows_of = []
                for ring in (2, 3):
                    flat = list(base)
                    flat[slots[PARTITION]] = partition
                    flat[slots[KEY]] = ring
                    config = generation.canonicalize_flat(flat)[1]
                    assert config.config.get(PARTITION, "item") == partition
                    rows_of.append(config)
                    sampled.add((partition, config.config.get(KEY, 2)))
                assert (rows_of[0] == rows_of[1]) is (partition != "item")
            assert sampled == {("item", 2), ("item", 3), ("whole", 2), ("balanced", 2)}
            # What counts is the partition a row decodes to, not the one its
            # slot names: a row whose persistent partition lacks the query
            # map it needs decodes to one unit per CTA, and keeps its depth.
            for partition in partitions[1:]:
                flat = list(base)
                flat[slots[PARTITION]] = partition
                flat[slots[KEY]] = 3
                flat[slots["cute_mma_dag_query_tma"]] = False
                config = generation.canonicalize_flat(flat)[1]
                assert config.config.get(PARTITION, "item") == "item"
                assert config.config[KEY] == 3
            # A seed or a cached config that names the depth beside a
            # persistent partition starts a search at that partition's row.
            for partition in partitions[1:]:
                named = stream_config(rows, **{KEY: 3, PARTITION: partition})
                pair = generation.transfer_seed(named)
                assert pair is not None
                assert KEY not in pair[1].config
                assert pair[1].config[PARTITION] == partition
                assert (
                    pair[1]
                    == generation.transfer_seed(
                        stream_config(rows, **{PARTITION: partition})
                    )[1]
                )
                # A step onto the depth from there takes one unit per CTA.
                reached, alone = stream_step(
                    spec, stream_config(rows, **{PARTITION: partition}), **{KEY: 3}
                )
                assert reached.config[KEY] == 3 and PARTITION not in reached.config
                assert alone == pair[1]
            # A fixed override keeps the depth it names on a partition whose
            # CTA holds it, in a seed's row and across a step onto one.
            forced = spec.create_config_generation(overrides={KEY: 3})
            for partition in partitions:
                named = stream_config(rows, **{PARTITION: partition})
                row = forced.canonicalize_flat(forced.flatten(named))[1]
                assert row.config[KEY] == 3
                assert row.config.get(PARTITION, "item") == partition
            # Under a fixed persistent partition no row names a depth.
            for partition in partitions[1:]:
                fixed = spec.create_config_generation(overrides={PARTITION: partition})
                row = fixed.canonicalize_flat(
                    fixed.flatten(stream_config(rows, **{KEY: 3}))
                )[1]
                assert KEY not in row.config and row.config[PARTITION] == partition
            fixed = spec.create_config_generation(overrides={PARTITION: "item"})
            row = fixed.canonicalize_flat(
                fixed.flatten(stream_config(rows, **{KEY: 3, PARTITION: "whole"}))
            )[1]
            assert row.config[KEY] == 3 and row.config.get(PARTITION, "item") == "item"
            # Nor under a fixed range skew other than zero, which moves
            # every paired row onto the balanced partition whatever its
            # slot names: the row that names one unit per CTA as well.
            # (The partition's own overrides alone do this. A seed that a
            # fixed range-ahead distance or a fixed look of the spare warp
            # moves onto the balanced partition keeps a depth that
            # partition holds, which
            # test_cute_paged_attention_range_ahead.py pins.)
            skewed = spec.create_config_generation(overrides={SKEW: 1})
            layout = skewed._key_to_flat_indices
            start = skewed.canonicalize_flat(skewed.flatten(stream_config(rows)))[0]
            for partition in partitions:
                flat = list(start)
                flat[layout[PARTITION][0][0]] = partition
                flat[layout[KEY][0][0]] = 3
                row = skewed.canonicalize_flat(flat)[1]
                assert KEY not in row.config, partition
                assert (row.config[PARTITION], row.config[SKEW]) == ("balanced", 1)
            pair = skewed.transfer_seed(stream_config(rows, **{KEY: 3}))
            assert pair is not None and KEY not in pair[1].config
            assert (pair[1].config[PARTITION], pair[1].config[SKEW]) == ("balanced", 1)
            # A fixed skew of zero moves no row: the depth stays where the
            # search samples it.
            level = spec.create_config_generation(overrides={SKEW: 0})
            row = level.canonicalize_flat(
                level.flatten(stream_config(rows, **{KEY: 3}))
            )[1]
            assert row.config[KEY] == 3 and PARTITION not in row.config


def test_overrides_keep_the_paired_seeds_with_or_without_their_depth():
    bound = stream_bound(ONE_WAVE, window=NO_WINDOW, input_banks=2, rows=40)
    spec = bound.config_spec
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        seeded, failures = transferred_seeds(spec, None)
        assert not failures
        paired = [
            seed for seed in seeded if seed.get("cute_mma_dag_pass_warps") == PAIRED
        ]
        assert [seed.config[KEY] for seed in seeded if KEY in seed.config] == [3]
        # A fixed depth is layered over complete candidates: every paired
        # one takes it and keeps its place; a candidate on another schedule
        # is left to that body's rule, which has no such ring.
        forced, failures = transferred_seeds(spec, {KEY: 3})
        kept = [
            seed for seed in forced if seed.get("cute_mma_dag_pass_warps") == PAIRED
        ]
        assert kept and forced[0] == kept[0] and len(forced) == len(kept)
        assert all(seed.config[KEY] == 3 for seed in kept)
        assert len(failures) == len(seeded) - len(paired)
        assert all(KEY in failure for failure in failures)
        # The depth pins neither a V lag nor a dependent-launch policy.
        assert {seed.config[V_LAG] for seed in kept} == {0, 1, 2, 3}
        assert {seed.config["cute_pdl"] for seed in kept} == {True, False}
        # An override that takes away what a seed's own depth needs takes
        # the depth, and the seed stays in the search.
        for overrides, drops in (
            ({PARTITION: "balanced"}, True),
            ({PARTITION: "whole"}, True),
            ({"cute_mma_dag_pass_warps": "attention"}, True),
            ({PARTITION: "item"}, False),
            ({V_LAG: 3}, False),
            ({"cute_pdl": False}, False),
        ):
            moved, failures = transferred_seeds(spec, overrides)
            assert not [failure for failure in failures if KEY in failure], overrides
            rings = [seed.config[KEY] for seed in moved if KEY in seed.config]
            assert rings == ([] if drops else [3]), overrides
        # An override that names both the depth and a partition whose CTA
        # has no room is never repaired: the candidates are rejected.
        moved, failures = transferred_seeds(spec, {KEY: 3, PARTITION: "balanced"})
        assert not moved and len(failures) == len(seeded)
        assert sum(KEY in failure for failure in failures) == len(paired)
    # Under a fixed depth a balanced seed whose CTA has no room for it takes
    # one unit per CTA, where the depth fits.
    waves = stream_bound(WAVES, window=NO_WINDOW, input_banks=2, rows=40)
    with cpu_render_target(fake_device_sm_count=SMS), waves.env, waves.host_function:
        seeded, _failures = transferred_seeds(waves.config_spec, None)
        assert any(seed.config.get(PARTITION) == "balanced" for seed in seeded)
        forced, _failures = transferred_seeds(waves.config_spec, {KEY: 3})
        assert forced and all(seed.config[KEY] == 3 for seed in forced)
        assert not any(PARTITION in seed.config for seed in forced)
    # Where its CTA holds the depth (32 query columns), the seed keeps its
    # partition under the override: the depth is the override's, not a
    # sample of the search.
    narrow = stream_bound(WAVES, window=NO_WINDOW, input_banks=2, rows=32)
    with cpu_render_target(fake_device_sm_count=SMS), narrow.env, narrow.host_function:
        seeded, _failures = transferred_seeds(narrow.config_spec, None)
        balanced = sum(seed.config.get(PARTITION) == "balanced" for seed in seeded)
        assert balanced and not any(KEY in seed.config for seed in seeded)
        forced, _failures = transferred_seeds(narrow.config_spec, {KEY: 3})
        assert forced and all(seed.config[KEY] == 3 for seed in forced)
        assert balanced == sum(
            seed.config.get(PARTITION) == "balanced" for seed in forced
        )


def test_a_marked_kernel_keeps_the_depth():
    # The depth needs no dependent-launch policy: a kernel that does not
    # take the guarded one still has the field.
    bound = stream_bound(ONE_WAVE, window=NO_WINDOW)
    marked = copy.copy(bound.config_spec)
    marked.has_explicit_pdl = True
    assert marked.cute_mma_dag_value_ring is not None
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        kept = marked.normalized_config(stream_config(**{KEY: 3}, cute_pdl=False))
    assert kept.config[KEY] == 3
