"""Check paired retirement configuration and allocation-wide synchronization.

The same module pins the template's setup rendezvous: the loader warp
initializes the pipeline barriers and only arrives at it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import helion
from helion._compiler.cute._paged_attention_paired_pipeline import (
    _allocate_pipeline_shared_memory,
)
from helion._compiler.cute._paged_attention_paired_pipeline import (
    _pipeline_barrier_plan,
)
from helion._compiler.cute._paged_attention_paired_pipeline import _pipeline_barriers
from helion._compiler.cute._paged_attention_paired_pipeline import _pipeline_parts
from helion._compiler.cute._paged_attention_paired_pipeline import (
    _pipeline_shared_memory,
)
from helion._compiler.cute.cute_mma_dag_seeds import MmaDagSeedPlanner
from helion._compiler.cute.pdl_release import CUTE_PDL_RELEASE_CHOICES
from helion._testing import cpu_render_target
from helion._testing.cute_paged_v_lag import paired_v_lag_bound
from helion._testing.cute_paged_v_lag import paired_v_lag_config
from helion.autotuner.config_generation import ConfigGeneration

pytestmark = pytest.mark.cpu_only
KEY = "cute_mma_dag_early_tmem_retirement"
SOURCE = (
    Path(__file__).parents[1]
    / "helion/_compiler/cute/_paged_attention_paired_pipeline.py"
)


def _specialize_retirement(enabled, join=False):
    class Specialize(ast.NodeTransformer):
        def visit_If(self, node):
            test = ast.unparse(node.test)
            if test == "cutlass.const_expr(early_tmem_retirement)":
                return node.body if enabled else node.orelse
            if test == "cutlass.const_expr(not early_tmem_retirement)":
                return node.orelse if enabled else node.body
            if test == (
                "cutlass.const_expr(join_before_exit or not early_tmem_retirement)"
            ):
                return node.body if join or not enabled else node.orelse
            return self.generic_visit(node)

    return ast.fix_missing_locations(Specialize().visit(ast.parse(SOURCE.read_text())))


def calls(node, name):
    return [
        n
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and ast.unparse(n.func).split(".")[-1] == name
    ]


def expression_matches(node, source):
    return ast.dump(node, include_attributes=False) == ast.dump(
        ast.parse(source, mode="eval").body, include_attributes=False
    )


def final_joins(body):
    """The all-thread joins anywhere in the body, each a top-level statement."""
    joins = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Expr)
        and expression_matches(node.value, "cute.arch.barrier()")
    ]
    assert all(node in body.body for node in joins)
    return joins


def retirement_calls(node, name):
    return [call for call in calls(node, name) if ast.unparse(call.args[0]) == "5"]


def assert_protocol(source, *, join):
    tree = ast.parse(source)
    compute = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_compute_head"
    )
    body = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "paged_attention_paired_body"
    )
    arrivals = retirement_calls(tree, "barrier_cta_arrive")
    assert len(arrivals) == 1
    arrive = arrivals[0]
    assert ast.unparse(arrive) == "prims.barrier_cta_arrive(5, thread_count=288)"
    arrival_index = next(
        i
        for i, n in enumerate(compute.body)
        if isinstance(n, ast.Expr) and n.value is arrive
    )
    result_index = next(
        i
        for i, n in enumerate(compute.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "result" for t in n.targets)
    )
    assert result_index < arrival_index
    assert calls(compute.body[result_index], "load_query_panel")
    assert "Tcgen05Wait.LOAD" in ast.unparse(compute.body[result_index + 1])
    assert "Tcgen05Fence.BEFORE_THREAD_SYNC" in ast.unparse(
        compute.body[arrival_index - 1]
    )
    # The actual nonserial final-PV drain must dominate the final O load.
    drains = [
        n
        for n in compute.body[:result_index]
        if isinstance(n, ast.If) and "probability_schedule !=" in ast.unparse(n.test)
    ]
    assert len(drains) == 1
    assert any(
        expression_matches(c, "_wait_non_mma(pv_done, (count - 1) & 1, sleep_waits)")
        for c in calls(drains[0], "_wait_non_mma")
    )
    # The denominators are reduced inside that wait: after the page loop,
    # ahead of the drain, without touching shared P or the O panel that the
    # last PV still owns.
    drain_index = compute.body.index(drains[0])
    loop_index = max(
        i
        for i, node in enumerate(compute.body[:drain_index])
        if isinstance(node, ast.For)
    )
    reduction = compute.body[loop_index + 1 : drain_index]
    assert len([node for node in reduction if calls(node, "warp_row_sum")]) == 2
    assert len([node for node in reduction if calls(node, "barrier")]) == 2
    names = {
        node.id
        for statement in reduction
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }
    assert not names & {"probability", "output_address", "result"}
    assert not any(calls(statement, "store_query_panel") for statement in reduction)
    # The seeded serial schedule drains each PV inside the loop before the
    # final output-panel read. Non-reference softmax uses this same path.
    loops = [
        node
        for node in compute.body[:result_index]
        if isinstance(node, ast.For) and calls(node, "online_softmax_per_key")
    ]
    assert len(loops) == 1
    serial = [
        node
        for node in ast.walk(loops[0])
        if isinstance(node, ast.If)
        and expression_matches(
            node.test, 'cutlass.const_expr(probability_schedule == "serial")'
        )
    ]
    assert len(serial) == 1 and calls(loops[0], "online_softmax_per_key")
    assert any(
        expression_matches(c, "_wait_non_mma(pv_done, step & 1, sleep_waits)")
        for c in calls(serial[0], "_wait_non_mma")
    )
    tail = ast.Module(body=compute.body[arrival_index + 1 :], type_ignores=[])
    assert not any(
        calls(tail, name)
        for name in [
            "load_query_panel",
            "store_query_panel",
            "tcgen05_ld",
            "tcgen05_st",
            "tcgen05_mma",
        ]
    )
    wait = calls(tail, "griddepcontrol_wait")
    stores = calls(tail, "store_output_element") + calls(tail, "store_row_element")
    assert len(wait) == 1 and stores and all(wait[0].lineno < n.lineno for n in stores)
    owner = next(
        n
        for n in ast.walk(body)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "warp < 8"
    )
    assert len(calls(owner, "_compute_head")) == 1
    assert "head = cute.arch.make_warp_uniform(warp // 4)" in ast.unparse(owner)
    frees = calls(tree, "tcgen05_dealloc")
    assert len(frees) == 1 and len(calls(tree, "tcgen05_alloc")) == 1
    issuer = next(
        n
        for n in ast.walk(body)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "warp == 16"
        and calls(n, "tcgen05_dealloc")
    )
    assert (
        isinstance(issuer.body[0], ast.For)
        and ast.unparse(issuer.body[0].iter) == "cutlass.range(count)"
    )
    sync = next(n for n in issuer.body if calls(n, "barrier_cta_sync"))
    assert ast.unparse(sync) == "prims.barrier_cta_sync(5, thread_count=288)"
    assert sync.lineno < frees[0].lineno
    # No role waits for the others after the retirement, unless the caller
    # asked for one common end; that join then follows the release.
    joins = final_joins(body)
    assert len(joins) == int(join)
    assert all(node.lineno > frees[0].lineno for node in joins)
    assert (
        ast.unparse(frees[0])
        == "prims.tcgen05_dealloc(cutlass.inttoptr(base, 6, cutlass.Int32), 512, group=prims.CTAGroup.CTA_1)"
    )


@pytest.mark.parametrize("join", [False, True])
def test_enabled_retirement_preserves_complete_protocol(join):
    assert_protocol(ast.unparse(_specialize_retirement(True, join)), join=join)


def test_disabled_retirement_keeps_final_join_before_deallocation():
    tree = _specialize_retirement(False)
    assert not retirement_calls(tree, "barrier_cta_arrive")
    assert not retirement_calls(tree, "barrier_cta_sync")
    body = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "paged_attention_paired_body"
    )
    defaults = dict(
        zip(
            (arg.arg for arg in body.args.args[-len(body.args.defaults) :]),
            body.args.defaults,
            strict=True,
        )
    )
    assert ast.literal_eval(defaults["early_tmem_retirement"]) is False
    assert ast.literal_eval(defaults["join_before_exit"]) is False
    frees = calls(tree, "tcgen05_dealloc")
    (join,) = final_joins(body)
    assert len(frees) == 1 and join.lineno < frees[0].lineno
    owners = [
        node
        for node in body.body
        if isinstance(node, ast.If)
        and expression_matches(node.test, "warp == 16")
        and frees[0] in ast.walk(node)
    ]
    assert len(owners) == 1
    assert (
        ast.unparse(frees[0])
        == "prims.tcgen05_dealloc(cutlass.inttoptr(base, 6, cutlass.Int32), 512, group=prims.CTAGroup.CTA_1)"
    )


def _loader_statements(statements):
    """The statements among ``statements`` that the loader warp executes:
    both arms of a compile-time test, its own arm of a role test, and every
    other statement whole."""
    for node in statements:
        test = ast.unparse(node.test) if isinstance(node, ast.If) else None
        if test is None:
            yield node
        elif test.startswith("cutlass.const_expr("):
            yield from _loader_statements([*node.body, *node.orelse])
        elif test == "warp == 17":
            yield from _loader_statements(node.body)
        elif test in ("warp == 16", "thread < 2 * query_columns"):
            # Warp 17 holds threads 544-575; the row state belongs to the
            # first 2 * query_columns <= 80 threads.
            yield from _loader_statements(node.orelse)
        else:
            yield node


def _shared_memory_names(body):
    """Names bound to the CTA's shared memory: its allocations (the body
    takes them from ``_allocate_pipeline_shared_memory``) and every pointer
    or tensor derived from them."""
    shared = set()
    assignments = [
        (node.targets[0].id, node.value)
        for node in ast.walk(body)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    ]
    grew = True
    while grew:
        grew = False
        for name, value in assignments:
            if name in shared:
                continue
            if calls(value, "_allocate_pipeline_shared_memory") or shared & {
                node.id for node in ast.walk(value) if isinstance(node, ast.Name)
            }:
                shared.add(name)
                grew = True
    return shared


def test_setup_rendezvous_follows_the_loader_initialization():
    tree = ast.parse(SOURCE.read_text())
    body = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "paged_attention_paired_body"
    )
    # One rendezvous: the loader warp arrives, every other thread waits for
    # all 640 arrivals. No all-thread barrier precedes the role dispatch.
    (rendezvous,) = [
        node
        for node in body.body
        if isinstance(node, ast.If) and calls(node, "barrier_cta_arrive")
    ]
    assert ast.unparse(rendezvous.test) == "warp == 17"
    assert [ast.unparse(node) for node in rendezvous.body] == [
        "prims.barrier_cta_arrive(6, thread_count=640)"
    ]
    # The QK role's allocation writes the TMEM slot: only the side that
    # waited reads it.
    assert [ast.unparse(node) for node in rendezvous.orelse] == [
        "prims.barrier_cta_sync(6, thread_count=640)",
        "base = cute.make_tensor(slot, cute.make_layout((1,)))[0]",
    ]
    index = body.body.index(rendezvous)
    assert not any(
        expression_matches(node.value, "cute.arch.barrier()")
        for statement in body.body[:index]
        for node in ast.walk(statement)
        if isinstance(node, ast.Expr)
    )
    # The loader initializes every pipeline barrier, one instruction per 32
    # with a lane-indexed address, and fences before it arrives. Each lane
    # initialized other barriers than lane zero later uses, so the warp
    # synchronizes behind the fence. Only the query barrier belongs to
    # another warp, which also issues its copies.
    initializers = [node for node in body.body[:index] if calls(node, "mbarrier_init")]
    loader, owner = initializers
    assert isinstance(loader, ast.If) and ast.unparse(loader.test) == "warp == 17"
    (pipeline,) = calls(loader, "mbarrier_init")
    assert ast.unparse(pipeline.args[0]) == "barriers + (32 * init_round + lane)"
    assert [ast.unparse(node) for node in loader.body[-2:]] == [
        "prims.fence_mbarrier_init()",
        "cute.arch.sync_warp()",
    ]
    assert not any(
        isinstance(node, ast.If) and "lane" in ast.unparse(node.test)
        for node in ast.walk(loader)
    )
    # The QK warp allocates TMEM as one warp before lane zero alone
    # initializes the query barrier, fences and issues the copies.
    assert isinstance(owner, ast.If) and ast.unparse(owner.test) == "warp == 16"
    assert [ast.unparse(node.value.func) for node in owner.body[:2]] == [
        "prims.tcgen05_alloc",
        "prims.tcgen05_relinquish_alloc_permit",
    ]
    (lane_zero,) = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "lane == 0"
    ]
    order = [
        ast.unparse(node.value.func).split(".")[-1]
        for node in lane_zero.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert order == ["mbarrier_init", "fence_mbarrier_init", "enqueue_paged_query"]
    # The loader's role reads nothing the other roles prepare before that
    # rendezvous: not the TMEM base, the row state or shared Q.
    role = next(
        node
        for node in ast.walk(body)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "warp == 17"
        and calls(node, "_enqueue_operand_tile")
    )
    names = {
        node.id
        for statement in role.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }
    assert "raw_k_ready" in names
    assert not names & {"base", "state_raw", "scratch_raw", "shared_q_raw", "slot"}
    # Nor does it in the statements every role runs ahead of the role
    # dispatch, where the loader has waited for nothing: there it allocates
    # and derives addresses, initializes its barriers and stores its share of
    # Q, but reads no shared memory and hands none to a helper that could.
    (dispatch,) = [
        position
        for position, node in enumerate(body.body)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "warp < 8"
    ]
    assert index < dispatch
    shared = _shared_memory_names(body)
    assert {"slot", "barriers", "state_raw", "shared_q", "raw_k_ready"} <= shared
    unwaited = list(_loader_statements(body.body[:dispatch]))
    assert rendezvous.body[0] in unwaited
    assert not any(node in unwaited for node in rendezvous.orelse)

    def touches_shared(node):
        return bool(shared & {n.id for n in ast.walk(node) if isinstance(n, ast.Name)})

    assert not [
        ast.unparse(node)
        for statement in unwaited
        for node in ast.walk(statement)
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Load)
        and touches_shared(node.value)
    ]
    assert {
        ast.unparse(call.func)
        for statement in unwaited
        for call in ast.walk(statement)
        if isinstance(call, ast.Call) and any(map(touches_shared, call.args))
    } == {"cute.make_tensor", "prims.mbarrier_init"}


@pytest.mark.parametrize("value_metadata", [False, True])
def test_pipeline_barrier_block_follows_its_table(value_metadata):
    table = _pipeline_barriers(value_metadata)
    offsets, count, rounds = _pipeline_barrier_plan(value_metadata)
    assert count % 32 == 0 and len(rounds) == count // 32
    # What the lane masks of the initialization rounds give each slot.
    initialized = [
        1 + sum(extra for extra, lanes in terms if lanes >> lane & 1)
        for terms in rounds
        for lane in range(32)
    ]
    # The entries follow one another, each slot of an entry expecting the
    # entry's arrivals; the padding to whole rounds expects one.
    slot = 0
    for name, width, arrivals in table:
        assert offsets[name] == slot
        assert initialized[slot : slot + width] == [arrivals] * width
        slot += width
    assert len(offsets) == len(table) and set(initialized[slot:]) <= {1}


class _RecordingAllocator:
    """What ``cutlass.utils.SmemAllocator`` does with the body's calls: each
    array starts at the next offset of its alignment."""

    def __init__(self) -> None:
        self.size = 0
        self.calls: list[tuple[object, int, int]] = []

    def allocate_array(self, dtype, count: int, *, byte_alignment: int) -> int:
        self.calls.append((dtype, count, byte_alignment))
        alignment = max(byte_alignment, dtype.width // 8)
        start = -(-self.size // alignment) * alignment
        self.size = start + dtype.width // 8 * count
        return start


def _paired_shared_memory(query_columns: int, *parts: str):
    """The body's allocations for one panel width and set of optional
    parts: the allocator after them and what the body gets by name."""
    allocator = _RecordingAllocator()
    arrays = _pipeline_shared_memory(query_columns, "value_metadata" in parts)
    shared = _allocate_pipeline_shared_memory(allocator, arrays, frozenset(parts))
    return allocator, arrays, vars(shared)


def test_paired_body_allocates_its_shared_memory_table_in_order():
    for parts in ((), ("query_map",), ("value_metadata",)):
        allocator, arrays, shared = _paired_shared_memory(40, *parts)
        present = [a for a in arrays if a.guard is None or a.guard in parts]
        # One allocation per entry the body has, in table order, with the
        # entry's own element type, count and alignment.
        assert allocator.calls == [(a.dtype, a.count, a.alignment) for a in present]
        # The body reads every array by its entry's name; an array of a part
        # it lacks is ``None``.
        assert list(shared) == [a.name for a in arrays]
        assert len(set(shared)) == len(arrays)
        assert {name for name, pointer in shared.items() if pointer is None} == {
            a.name for a in arrays if a not in present
        }
        # Each array starts where the one before it ends, rounded up.
        assert shared["raw_k"] == 0 and shared["raw_v"] == 4 * 16384
    arrays = _pipeline_shared_memory(40, True)
    # Every optional part an entry names is one a body can have, and the
    # barrier block is the one ``_pipeline_barrier_plan`` lays out.
    assert {a.guard for a in arrays} - {None} == _pipeline_parts(True, True)
    assert _pipeline_parts(False, False) == frozenset()
    for value_metadata in (False, True):
        (block,) = (
            a
            for a in _pipeline_shared_memory(40, value_metadata)
            if a.name == "barriers"
        )
        assert block.count == _pipeline_barrier_plan(value_metadata)[1]


def test_paired_shared_memory_stays_inside_its_budget():
    # The transport admits the paired schedule on 180 KiB of shared memory
    # (``capacity >= 184320``). Replay the body's allocations, in order and
    # with their alignment, for every panel width and both optional parts.
    def layout(query_columns, *parts):
        return _paired_shared_memory(query_columns, *parts)[0].size

    assert layout(40) == 178952
    assert layout(40, "query_map") == 178960
    assert layout(40, "value_metadata") == 179232
    largest = max(
        layout(query_columns, *parts)
        for query_columns in (8, 32, 40)
        for parts in (
            (),
            ("query_map",),
            ("value_metadata",),
            ("query_map", "value_metadata"),
        )
    )
    assert largest == 179232
    gate = (SOURCE.parent / "paged_attention_transport.py").read_text()
    assert largest <= 184320 and "capacity >= 184320" in gate


def test_missing_retirement_is_enabled_only_for_the_paired_schedule():
    bound = paired_v_lag_bound()
    absent = paired_v_lag_config(0)
    enabled = helion.Config.from_dict({**absent.config, KEY: True})
    disabled = helion.Config.from_dict({**absent.config, KEY: False})
    unpaired = helion.Config.from_dict(
        {
            **absent.config,
            "cute_mma_dag_pass_warps": "attention",
            "cute_mma_dag_v_lag": 0,
        }
    )
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        normalized = bound.config_spec.normalized_config(absent)
        assert normalized.get(KEY) is True
        assert normalized == bound.config_spec.normalized_config(enabled)
        assert bound.config_spec.normalized_config(disabled).get(KEY) is False
        assert bound.config_spec.normalized_config(unpaired).get(KEY) is False
        # Flattening a config that was not normalized first means the same.
        generation = ConfigGeneration(bound.config_spec)
        for config, expected in ((absent, True), (disabled, False), (unpaired, False)):
            _, round_trip = generation.canonicalize_flat(generation.flatten(config))
            assert round_trip.get(KEY) is expected
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(absent)
        assert source == bound.to_code(enabled)
        assert "early_tmem_retirement=True" in source
        assert "join_before_exit" not in source
        assert "early_tmem_retirement" not in bound.to_code(disabled)


def test_an_invalid_retirement_is_repaired_to_what_the_missing_key_gives():
    """A value that is no bool is an ``InvalidConfig``, and its repair is the
    config without the key: early retirement on the paired schedule, the
    final join on another body of the same kernel."""
    bound = paired_v_lag_bound()
    paired = paired_v_lag_config(0).config
    unpaired = {
        **paired,
        "cute_mma_dag_pass_warps": "attention",
        "cute_mma_dag_v_lag": 0,
    }
    spec = bound.config_spec
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        for config, default in ((paired, True), (unpaired, False)):
            assert KEY not in config
            omitted = dict(config)
            spec.normalize(omitted)
            assert omitted[KEY] is default
            for value in ("early", 1, 0, None):
                with pytest.raises(helion.exc.InvalidConfig, match=KEY):
                    spec.normalize({**config, KEY: value})
                repaired = {**config, KEY: value}
                spec.normalize(repaired, _fix_invalid=True)
                assert repaired == omitted


# Every placement the release knob offers (the loader's tail belongs to the
# guarded capture overlap), and the knob on a kernel without ``cute_pdl``.
@pytest.mark.parametrize(
    "options",
    [
        *(
            {"cute_pdl": True, "cute_pdl_release": release}
            for release in CUTE_PDL_RELEASE_CHOICES
            if release != "loader_tail"
        ),
        {
            "cute_pdl": True,
            "cute_pdl_capture_overlap": True,
            "cute_pdl_release": "loader_tail",
        },
        {"cute_pdl": False, "cute_pdl_release": "exit"},
    ],
    ids=lambda options: (
        "-".join(str(value).lower() for value in options.values() if value is not True)
        or "pdl"
    ),
)
def test_only_the_exit_release_keeps_a_common_end(options):
    # The generic exit release follows the body in every thread, so the
    # retired schedule keeps its final join exactly where the dependent-launch
    # pass puts the release behind the body (``exit_release_applies``).
    bound = paired_v_lag_bound()
    config = paired_v_lag_config(0)
    config.config.update(options)
    with cpu_render_target(fake_device_sm_count=148):
        source = bound.to_code(config)
    assert "early_tmem_retirement=True" in source
    kernel = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    released_at_exit = (
        ast.unparse(kernel.body[-1]) == "cute.arch.griddepcontrol_launch_dependents()"
    )
    assert released_at_exit == (
        options == {"cute_pdl": True, "cute_pdl_release": "exit"}
    )
    assert ("join_before_exit=True" in source) == released_at_exit


def test_retirement_seeds_are_four_independent_paired_counterparts():
    bound = paired_v_lag_bound()
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None
    with cpu_render_target(fake_device_sm_count=148), bound.env, bound.host_function:
        planner = MmaDagSeedPlanner(
            facts, bound.config_spec, occupancy_key="cute_min_blocks_per_mp"
        )
        seeds = planner.configs(planner.ranked())
        late = [s for s in seeds if s.get(KEY) is False]
        assert len(late) == 4 and seeds[-4:] == late
        # Every other paired seed spells out the schedule's default: the
        # late counterparts differ from them in this key alone.
        paired = [
            s
            for s in seeds[:-4]
            if s.get("cute_mma_dag_pass_warps") == "attention_paired"
        ]
        assert paired and all(s.get(KEY) is True for s in paired)
        assert all(KEY not in s.config for s in seeds[:-4] if s not in paired)
        # A promoted seed has the default spelled out or not: the base config
        # it is layered on gives up its pin of the knob.
        unspelled = {k: v for k, v in paired[0].config.items() if k != KEY}
        for values in (paired[0].config, unspelled):
            promoted = helion.Config.from_dict(
                bound.config_spec.promoted_default_values(values)
            )
            assert bound.config_spec.normalized_config(promoted).get(KEY) is True
        assert [s.get("cute_mma_dag_v_lag", 0) for s in late] == [0, 1, 2, 3]
        for seed in late:
            assert seed.get("cute_mma_dag_pass_warps") == "attention_paired"
            assert seed.get("cute_pdl") is False
            assert not any(seed.get("load_eviction_policies"))
            counterpart = helion.Config.from_dict({**seed.config, KEY: True})
            assert counterpart in paired
            normalized = bound.config_spec.normalized_config(seed)
            assert normalized.get(KEY) is False
            assert bound.env.backend.autotune_config_is_viable(
                bound.config_spec, normalized
            )
