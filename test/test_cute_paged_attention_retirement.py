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
from helion._compiler.cute._paged_attention_partition import plan_geometry
from helion._compiler.cute.cute_mma_dag_seeds import MmaDagSeedPlanner
from helion._compiler.cute.pdl_release import CUTE_PDL_CAPTURE_DEFAULT
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
PARTITION_HELPERS = SOURCE.with_name("_paged_attention_partition.py")


def _specialize_retirement(enabled, join=False, balanced=False):
    """The template with its retirement tests decided and, inside the role
    helpers, the balanced partition's arms taken or left out."""

    class Specialize(ast.NodeTransformer):
        def visit_If(self, node):
            test = ast.unparse(node.test)
            if test == "cutlass.const_expr(early_tmem_retirement)":
                return self.visit_all(node.body if enabled else node.orelse)
            if test == "cutlass.const_expr(not early_tmem_retirement)":
                return self.visit_all(node.orelse if enabled else node.body)
            if test == (
                "cutlass.const_expr(join_before_exit or not early_tmem_retirement)"
            ):
                return self.visit_all(node.body if join or not enabled else node.orelse)
            if test == "cutlass.const_expr(partition == 'balanced')":
                return self.visit_all(node.body if balanced else node.orelse)
            node = self.generic_visit(node)
            # An arm that only held a decided test keeps its place.
            node.body = node.body or [ast.Pass()]
            return node

        def visit_all(self, statements):
            result = []
            for statement in statements:
                visited = self.visit(statement)
                result.extend(visited if isinstance(visited, list) else [visited])
            return result

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


def function(tree, name):
    return next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def bound_arguments(tree, call):
    """The arguments of ``call`` by the parameter names of the template
    helper it calls, as source text: an audit that names the parameters it
    reads keeps reading them when the helper gains, drops or reorders
    others."""
    helper = function(tree, call.func.id)
    bound = dict(zip([arg.arg for arg in helper.args.args], call.args, strict=False))
    bound.update({keyword.arg: keyword.value for keyword in call.keywords})
    return {name: ast.unparse(value) for name, value in bound.items()}


# What reads or writes the tensor-memory allocation.
TMEM_ACCESSES = (
    "load_query_panel",
    "store_query_panel",
    "tcgen05_ld",
    "tcgen05_st",
    "tcgen05_mma",
)


def assert_drain(tree, reduce):
    """``reduce`` waits for its range's last PV once, as its own statement:
    returns that statement. The wait itself is one helper's, which both
    reductions call with the range ``_compute_pages`` ran."""
    drain = function(tree, "_drain_last_pv")
    (waits,) = [node for node in drain.body if isinstance(node, ast.If)]
    assert ast.unparse(waits.test) == (
        "cutlass.const_expr(probability_schedule != 'serial')"
    )
    assert any(
        expression_matches(
            c, "_wait_non_mma(pv_done, (phase + count - 1) & 1, sleep_waits)"
        )
        for c in calls(waits, "_wait_non_mma")
    )
    assert not any(calls(drain, name) for name in TMEM_ACCESSES)
    (drained,) = [node for node in reduce.body if calls(node, "_drain_last_pv")]
    assert isinstance(drained, ast.Expr) and expression_matches(
        drained.value,
        "_drain_last_pv(pv_done, phase, count, sleep_waits, probability_schedule)",
    )
    return drained


def assert_reduction_overlaps_the_last_pv(tree, reduction):
    """The statements a reduction runs ahead of its wait for the last PV,
    with the helper that reduces the rows: they touch neither shared P nor
    the O panel that PV still owns."""
    rows = function(tree, "_reduce_rows")
    (shared,) = [node for node in reduction if calls(node, "_reduce_rows")]
    assert all(
        isinstance(node, ast.Assert) for node in reduction[: reduction.index(shared)]
    )
    body = rows.body[1:] if ast.get_docstring(rows) is not None else rows.body
    statements = [*body, *reduction]
    assert len([node for node in statements if calls(node, "warp_row_sum")]) == 2
    assert len([node for node in statements if calls(node, "barrier")]) == 2
    names = {
        node.id
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }
    assert not names & {"probability", "output_address", "result"}
    # The denominators of the TMEM schedule are the one panel it reads.
    panels = [call for node in statements for call in calls(node, "load_query_panel")]
    assert [ast.unparse(call.args[0]) for call in panels] == ["denominator_address"]
    assert not any(
        calls(statement, name)
        for statement in statements
        for name in TMEM_ACCESSES
        if name != "load_query_panel"
    )


def assert_retirement_helper(tree):
    """The QK role's early retirement, one helper for the one-unit role and
    the persistent walk: it joins barrier 5 with the two compute groups and
    frees the allocation, and it holds the template's only early release."""
    retire = function(tree, "_retire_tensor_memory")
    body = retire.body[1:] if ast.get_docstring(retire) is not None else retire.body
    assert [ast.unparse(node) for node in body] == [
        "prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)",
        "prims.barrier_cta_sync(5, thread_count=288)",
        "prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)",
        "prims.tcgen05_dealloc(cutlass.inttoptr(base, 6, cutlass.Int32), 512, group=prims.CTAGroup.CTA_1)",
    ]
    assert calls(tree, "tcgen05_dealloc") == calls(retire, "tcgen05_dealloc")
    assert retirement_calls(tree, "barrier_cta_sync") == calls(
        retire, "barrier_cta_sync"
    )
    return retire


def assert_arrival_helper(tree):
    """A compute thread's arrival at the early retirement, one helper for
    the one-unit role and the persistent walk: the fence behind the thread's
    last use of the slab and the arrival at barrier 5, the template's only
    one."""
    arrival = function(tree, "_arrive_at_retirement")
    body = arrival.body[1:] if ast.get_docstring(arrival) is not None else arrival.body
    assert [ast.unparse(node) for node in body] == [
        "prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)",
        "prims.barrier_cta_arrive(5, thread_count=288)",
    ]
    assert retirement_calls(tree, "barrier_cta_arrive") == calls(
        arrival, "barrier_cta_arrive"
    )
    return arrival


def assert_protocol(source, *, join):
    tree = ast.parse(source)
    compute = function(tree, "_compute_head")
    pages = function(tree, "_compute_pages")
    reduce = function(tree, "_compute_reduce")
    store = function(tree, "_compute_store")
    body = function(tree, "paged_attention_paired_body")
    arrivals = calls(compute, assert_arrival_helper(tree).name)
    assert len(arrivals) == 1
    arrive = arrivals[0]
    arrival_index = next(
        i
        for i, n in enumerate(compute.body)
        if isinstance(n, ast.Expr) and n.value is arrive
    )
    # One group runs its helpers in order: its entry, pages, the reduction
    # that ends with the final O read, the arrival, the dependency wait, the
    # stores.
    order = [
        ast.unparse(n.value.func)
        for n in compute.body
        if isinstance(n, (ast.Assign, ast.Expr))
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
    ]
    assert order == [
        "_compute_group",
        "_compute_begin",
        "_compute_pages",
        "_compute_reduce",
        "_arrive_at_retirement",
        "_compute_store",
    ]
    result_index = next(
        i
        for i, n in enumerate(compute.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "result" for t in n.targets)
    )
    assert result_index == arrival_index - 1
    assert calls(compute.body[result_index], "_compute_reduce")
    # The reduction ends with the final O read and returns the register copy.
    load, wait, returned = reduce.body[-3:]
    assert isinstance(load, ast.Assign) and ast.unparse(load.targets[0]) == "result"
    assert calls(load, "load_query_panel")
    assert "Tcgen05Wait.LOAD" in ast.unparse(wait)
    assert ast.unparse(returned) == "return result"
    # The actual nonserial final-PV drain must dominate the final O load. The
    # page helper leaves it to the reduction, which both get the same range.
    assert not any(
        isinstance(n, ast.If) and "probability_schedule !=" in ast.unparse(n.test)
        for n in pages.body
    )
    drained = assert_drain(tree, reduce)
    assert drained is reduce.body[-4]
    ranges = {
        call.func.id: bound_arguments(tree, call)
        for call in (
            calls(compute, "_compute_pages") + calls(compute, "_compute_reduce")
        )
    }
    assert [ranges["_compute_pages"][name] for name in ("phase", "step0", "count")] == [
        "cutlass.Int32(0)",
        "cutlass.Int64(0)",
        "count",
    ]
    assert [ranges["_compute_reduce"][name] for name in ("phase", "count")] == [
        "cutlass.Int32(0)",
        "count",
    ]
    # The denominators are reduced inside that wait: after the page loop,
    # ahead of the drain, without touching shared P or the O panel that the
    # last PV still owns.
    assert isinstance(pages.body[-1], ast.For)
    reduction = reduce.body[: reduce.body.index(drained)]
    if ast.get_docstring(reduce) is not None:
        reduction = reduction[1:]
    assert_reduction_overlaps_the_last_pv(tree, reduction)
    # The seeded serial schedule drains each PV inside the loop before the
    # final output-panel read. Non-reference softmax uses this same path.
    loops = [
        node
        for node in pages.body
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
        expression_matches(c, "_wait_non_mma(pv_done, (phase + step) & 1, sleep_waits)")
        for c in calls(serial[0], "_wait_non_mma")
    )
    # Nothing after the arrival touches TMEM: neither the rest of the group's
    # driver nor the store helper it calls.
    tail = ast.Module(
        body=[*compute.body[arrival_index + 1 :], *store.body], type_ignores=[]
    )
    assert not any(calls(tail, name) for name in TMEM_ACCESSES)
    wait = calls(tail, "griddepcontrol_wait")
    stored = calls(compute, "_compute_store")
    stores = calls(store, "store_output_element") + calls(store, "store_row_element")
    assert len(wait) == 1 and len(stored) == 1 and stores
    assert arrive.lineno < wait[0].lineno < stored[0].lineno
    owner = next(
        n
        for n in ast.walk(body)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "warp < 8"
        and calls(n, "_compute_head")
    )
    assert len(calls(owner, "_compute_head")) == 1
    assert "head = cute.arch.make_warp_uniform(warp // 4)" in ast.unparse(owner)
    # The QK role retires the allocation through the one helper both
    # drivers call: it joins barrier 5 and frees, and nothing else does.
    retire = assert_retirement_helper(tree)
    assert not calls(body, "tcgen05_dealloc")
    assert len(calls(tree, "tcgen05_alloc")) == 1
    issuer = next(
        n
        for n in ast.walk(body)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "warp == 16"
        and calls(n, retire.name)
    )
    # The QK role issues every page of the range before it joins barrier 5.
    assert isinstance(issuer.body[0], ast.Expr) and calls(issuer.body[0], "_issue_qk")
    (issue,) = function(tree, "_issue_qk").body[-1:]
    assert isinstance(issue, ast.For)
    assert ast.unparse(issue.iter) == "cutlass.range(count)"
    assert [ast.unparse(node) for node in issuer.body[1:]] == [
        "_retire_tensor_memory(base)"
    ]
    # No role waits for the others after the retirement, unless the caller
    # asked for one common end; that join then follows the release.
    joins = final_joins(body)
    assert len(joins) == int(join)
    assert all(node.lineno > issuer.body[1].lineno for node in joins)


def _called_helpers(trees, statements):
    """The template and partition helpers ``statements`` can run: those they
    call, and those these call in turn."""
    defined = {
        node.name: node
        for tree in trees
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    found: dict[str, ast.FunctionDef] = {}
    pending = list(statements)
    while pending:
        for call in ast.walk(pending.pop()):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                helper = defined.get(call.func.id)
                if helper is not None and helper.name not in found:
                    found[helper.name] = helper
                    pending.append(helper)
    return found


def assert_unit_protocol(source, *, balanced):
    """A persistent CTA's roles keep the protocol over all of its units:
    whole units, or under the balanced partition the parts of units too."""
    tree = ast.parse(source)
    helpers = ast.parse(PARTITION_HELPERS.read_text())
    compute = function(tree, "_compute_units")
    (walk,) = [node for node in compute.body if isinstance(node, ast.While)]
    assert ast.unparse(walk.test) == "unit < end"
    # The unit and the end of the range wait in shared memory while the
    # unit's pages run: one group thread parks them between the unit's two
    # group barriers and every thread reloads them after the pages.
    parked = next(
        i for i, n in enumerate(walk.body) if "walk[WALK_UNIT] = unit" in ast.unparse(n)
    )
    assert ast.unparse(walk.body[parked]).startswith("if thread == 0:")
    assert "walk[WALK_END] = end" in ast.unparse(walk.body[parked])
    assert "cute.arch.barrier(" in ast.unparse(walk.body[parked + 1])
    assert calls(walk.body[parked + 2], "_compute_pages")
    assert [ast.unparse(n) for n in walk.body[parked + 3 : parked + 5]] == [
        "unit = cutlass.Int32(walk[WALK_UNIT])",
        "end = cutlass.Int32(walk[WALK_END])",
    ]
    assert ast.unparse(walk.body[-1]) == "unit = unit + 1"
    # Every unit runs the one-unit driver's helpers in its order, behind
    # the unit's range. The balanced partition reduces with the helper that
    # can merge a unit in two parts.
    reducer = "_reduce_part" if balanced else "_compute_reduce"
    order = [
        ast.unparse(n.value.func)
        for n in walk.body
        if isinstance(n, (ast.Assign, ast.Expr))
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
    ]
    assert order == ["_unit_range", "_compute_begin", "_compute_pages", reducer]
    assert len(calls(compute, "_compute_reduce")) == int(not balanced)
    assert len(calls(compute, "_reduce_part")) == int(balanced)
    # The reduction waits for the last PV of the range's own pages: both
    # helpers get the range's count and the phase the CTA's stream is in.
    ranges = {
        call.func.id: bound_arguments(tree, call)
        for call in (calls(walk, "_compute_pages") + calls(walk, reducer))
    }
    assert [ranges["_compute_pages"][name] for name in ("phase", "step0", "count")] == [
        "phase",
        "step0",
        "count",
    ]
    assert [ranges[reducer][name] for name in ("phase", "count")] == ["phase", "count"]
    # The range is the one every walking role takes. Whole units run from
    # their first step; the balanced partition's range is the unit's steps
    # in the CTA's slots, which a compute group leaves in the plan: the
    # helper reads them from there, behind the unit's bounds and pages.
    assert ast.unparse(walk.body[0]) == (
        "item, first_page, last_page, length, unit_pages, step0, count = "
        "_unit_range(unit, header, None, None, plan, lengths, unit_bounds, "
        "unit_bounds_scanned, unit_bounds_args, page_limit, partition)"
    )
    ranged = function(tree, "_unit_range")
    assigned = sorted(
        (
            n
            for n in ast.walk(ranged)
            if isinstance(n, ast.Assign)
            and {"step0", "begin", "stop"}
            & {t.id for t in ast.walk(n.targets[0]) if isinstance(t, ast.Name)}
        ),
        key=lambda n: n.lineno,
    )
    assert [ast.unparse(n) for n in assigned] == [
        "step0 = cutlass.Int64(0)",
        *(
            [
                "begin = cutlass.Int32(plan[PLAN_BEGIN])",
                "stop = cutlass.Int32(plan[PLAN_END])",
                "count, step0 = unit_segment(header, unit_pages, begin, stop)",
            ]
            if balanced
            else []
        ),
    ]
    assert ast.unparse(ranged.body[-1]) == (
        "return (item, first_page, last_page, length, unit_pages, step0, count)"
    )
    advance = walk.body[-2]
    assert ast.unparse(advance) == "phase = phase + cutlass.Int32(count & 1) & 1"
    # Each thread arrives once: after the O read of its last unit and before
    # that unit's dependency wait and stores, or after a walk without units.
    arrival = assert_arrival_helper(tree).name
    assert len(calls(compute, arrival)) == 2
    (stored,) = [
        node
        for node in walk.body
        if isinstance(node, ast.If) and calls(node, "_compute_store")
    ]
    assert ast.unparse(stored.test) == "stores" and not stored.orelse
    assert len(stored.body) == 1 and len(calls(walk, "_compute_store")) == 1
    index = {
        "reduce": next(i for i, n in enumerate(walk.body) if calls(n, reducer)),
        "arrive": next(i for i, n in enumerate(walk.body) if calls(n, arrival)),
        # The wait every unit executes, as its own statement.
        "wait": next(
            i
            for i, n in enumerate(walk.body)
            if ast.unparse(n)
            == "if capture_pdl_wait_mode != 0:\n    cute.arch.griddepcontrol_wait()"
        ),
        "store": walk.body.index(stored),
    }
    assert list(index.values()) == sorted(index.values())
    assert len(set(index.values())) == 4
    last = walk.body[index["arrive"]]
    assert isinstance(last, ast.If) and not last.orelse
    assert ast.unparse(last.test) == "unit == end - 1"
    assert [ast.unparse(node) for node in last.body] == [f"{arrival}()"]
    # The exchange of a part of a split unit lies between that wait and the
    # stores, and nowhere else in the walk.
    exchange = [
        i
        for i, n in enumerate(walk.body)
        if any(calls(n, name) for name in ("store_part", "arrive_part", "merge_parts"))
    ]
    assert exchange == ([index["store"] - 1] if balanced else [])
    assert index["wait"] < index["store"] - int(balanced)
    # Nothing a thread runs after its arrival touches tensor memory: not the
    # rest of its last unit (the exchange of a part and the stores), not what
    # follows the walk (the spare warp's reply and the merge of a posted
    # unit), and none of the helpers these call.
    after_walk = compute.body[compute.body.index(walk) + 1 :]
    tail = [*walk.body[index["arrive"] + 1 :], *after_walk]
    reached = _called_helpers((tree, helpers), tail)
    assert "_compute_store" in reached
    assert ({"store_part", "arrive_part", "merge_parts"} <= set(reached)) == balanced
    assert not {"_reduce_part", "_compute_reduce", "_compute_pages"} & set(reached)
    for node in (*tail, *reached.values()):
        assert not any(calls(node, name) for name in TMEM_ACCESSES), ast.unparse(node)
    if balanced:
        # A part's reduction keeps the one-unit order around the wait for
        # its last PV. Ahead of it: the rows, the other part's loads from
        # the workspace and its weights, none of which is tensor memory.
        # Behind it: every read of the O panel, eight columns at a time for
        # a unit in two parts and whole otherwise.
        reduce = function(tree, "_reduce_part")
        drained = assert_drain(tree, reduce)
        position = reduce.body.index(drained)
        reduction = reduce.body[:position]
        if ast.get_docstring(reduce) is not None:
            reduction = reduction[1:]
        assert_reduction_overlaps_the_last_pv(tree, reduction)
        loaded = _called_helpers((tree, helpers), reduction)
        assert {"pair_partial", "merge_pair_rows", "merge_pair_other"} <= set(loaded)
        for helper in loaded.values():
            if helper.name != "_reduce_rows":
                assert not any(calls(helper, name) for name in TMEM_ACCESSES)
        (read,) = reduce.body[position + 1 : -1]
        assert ast.unparse(reduce.body[-1]) == "return panel"
        assert isinstance(read, ast.If) and ast.unparse(read.test) == "pair"
        (chunks,) = read.body
        (columns,) = calls(chunks, "tcgen05_ld")
        assert "output_address + chunk * 8" in ast.unparse(columns)
        assert "Tcgen05Wait.LOAD" in ast.unparse(chunks)
        own, wait, _copy = read.orelse
        assert ast.unparse(own) == (
            "own = load_query_panel(output_address, query_columns)"
        )
        assert "Tcgen05Wait.LOAD" in ast.unparse(wait)
        assert sum(len(calls(reduce, name)) for name in TMEM_ACCESSES) == 2
    idle, *planned, empty = after_walk[: 3 if balanced else 2]
    assert ast.unparse(idle) == "no_units = first_unit >= last_unit"
    assert [ast.unparse(node) for node in planned] == (
        ["no_units = cutlass.Int32(plan[PLAN_FIRST]) >= cutlass.Int32(plan[PLAN_LAST])"]
        if balanced
        else []
    )
    assert isinstance(empty, ast.If) and not empty.orelse
    assert ast.unparse(empty.test) == "no_units"
    assert ast.unparse(empty.body[0]) == f"{arrival}()"
    assert calls(empty.body[1], "griddepcontrol_wait") and len(empty.body) == 2
    # The QK role waits for every query copy the loader requested and issues
    # every unit before it joins barrier 5 and frees: without a final join,
    # nothing of the CTA is outstanding when it returns.
    issuer = function(tree, "_issue_qk_units")
    (issue,) = [node for node in issuer.body if isinstance(node, ast.For)]
    assert ast.unparse(issue.iter) == "cutlass.range(first_unit, last_unit, 1)"
    assert len(calls(issue, "_issue_qk")) == 1
    (taken,) = [node for node in issue.body if isinstance(node, ast.If)]
    assert ast.unparse(taken.test) == "count > 0"
    assert any(
        expression_matches(
            call, "mbar_wait(query_ready + stage, loaded // q_stages & 1)"
        )
        for call in calls(taken, "mbar_wait")
    )
    # The first unit with pages gets its queries from this role, one lane's
    # request into the first stage, ahead of the wait for them.
    own, landed = (
        next(i for i, node in enumerate(taken.body) if calls(node, name))
        for name in ("enqueue_paged_query", "mbar_wait")
    )
    assert own < landed
    first = taken.body[own]
    assert isinstance(first, ast.If) and ast.unparse(first.test) == "loaded == 0"
    (lane_zero,) = first.body
    assert ast.unparse(lane_zero.test) == "lane == 0" and not lane_zero.orelse
    (request,) = calls(lane_zero, "enqueue_paged_query")
    assert expression_matches(
        request,
        "enqueue_paged_query(query_map, query_raw, query_ready, item // heads, "
        "item % heads, query_columns)",
    )
    assert len(calls(issuer, "enqueue_paged_query")) == 1
    retire = issuer.body[issuer.body.index(issue) + 1 :]
    assert [ast.unparse(node) for node in retire] == ["_retire_tensor_memory(base)"]
    # The loader requests the queries of the other units with pages, each
    # into a stage the QK role has released, and counts the first one as
    # that role does: together they request exactly the queries the QK role
    # waits for.
    loader = function(tree, "_load_units")
    (stream,) = [node for node in loader.body if isinstance(node, ast.For)]
    assert ast.unparse(stream.iter) == "cutlass.range(first_unit, last_unit, 1)"
    (paged,) = [node for node in stream.body if isinstance(node, ast.If)]
    assert ast.unparse(paged.test) == "count > 0"
    assert len(calls(loader, "enqueue_paged_query")) == 1
    (later,) = [node for node in paged.body if calls(node, "enqueue_paged_query")]
    assert isinstance(later, ast.If) and ast.unparse(later.test) == "loaded > 0"
    assert not later.orelse
    released, request = later.body
    assert expression_matches(
        released.value,
        "_wait_non_mma(query_free + stage, loaded // q_stages & 1 ^ 1, sleep_waits)",
    )
    assert calls(request, "enqueue_paged_query")
    counted = paged.body[paged.body.index(later) + 1]
    assert ast.unparse(counted) == "loaded = loaded + 1"
    # Every walking role takes a unit's range from the one helper, which
    # evaluates the unit through the one helper that takes the unit's length
    # from the scan where the scan kept it: no role calls the prelude, counts
    # a unit's pages or clips it itself, so none loads a length the scan
    # already loaded and all three count the same pages for every unit.
    scanned = function(tree, "_scanned_bounds")
    assert ast.unparse(scanned.body[-2].test) == (
        "cutlass.const_expr(lengths is not None)"
    )
    assert ast.unparse(scanned.body[-2].body[-1]) == (
        "return unit_bounds_scanned(unit, cutlass.Int32(lengths[unit]), "
        "*unit_bounds_args)"
    )
    assert ast.unparse(scanned.body[-1]) == (
        "return unit_bounds(unit, *unit_bounds_args)"
    )
    (evaluated,) = calls(function(tree, "_unit_range"), "_scanned_bounds")
    assert [ast.unparse(argument) for argument in evaluated.args] == [
        "unit",
        "lengths",
        "unit_bounds",
        "unit_bounds_scanned",
        "unit_bounds_args",
    ]
    assert calls(tree, "_scanned_bounds") == [evaluated]
    for name in ("_compute_units", "_issue_qk_units", "_load_units"):
        role = function(tree, name)
        (taken,) = calls(role, "_unit_range")
        assert [ast.unparse(argument) for argument in taken.args[:2]] == [
            "unit",
            "header",
        ]
        assert [ast.unparse(argument) for argument in taken.args[5:]] == [
            "lengths",
            "unit_bounds",
            "unit_bounds_scanned",
            "unit_bounds_args",
            "page_limit",
            "partition",
        ]
        for evaluator in (
            "unit_bounds",
            "unit_bounds_scanned",
            "unit_page_count",
            "unit_segment",
        ):
            assert not calls(role, evaluator), (name, evaluator)
    # Both drivers free the allocation in the QK role, through the one
    # retirement helper, and nowhere else.
    body = function(tree, "paged_attention_paired_body")
    assert len(calls(body, "_issue_qk_units")) == 1
    assert len(calls(body, "_compute_units")) == 1
    helper = assert_retirement_helper(tree)
    assert len(calls(tree, helper.name)) == 2
    assert len(calls(body, helper.name)) == len(calls(issuer, helper.name)) == 1
    # The compute threads arrive there through the one arrival helper: in
    # the one-unit role, and twice in the walk.
    assert len(calls(tree, arrival)) == 3


@pytest.mark.parametrize("join", [False, True])
@pytest.mark.parametrize("balanced", [False, True])
def test_enabled_retirement_preserves_complete_protocol(join, balanced):
    source = ast.unparse(_specialize_retirement(True, join, balanced))
    assert_protocol(source, join=join)
    assert_unit_protocol(source, balanced=balanced)


def test_disabled_retirement_keeps_final_join_before_deallocation():
    tree = _specialize_retirement(False)
    # Nothing calls the compute threads' arrival, the only one at barrier 5.
    assert not calls(tree, assert_arrival_helper(tree).name)
    # Nothing calls the early retirement, the only wait at barrier 5.
    retire = function(tree, "_retire_tensor_memory")
    assert not calls(tree, retire.name)
    assert retirement_calls(tree, "barrier_cta_sync") == calls(
        retire, "barrier_cta_sync"
    )
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
    # The body's release at the common end; the only other one in the
    # template is the uncalled helper's.
    frees = calls(body, "tcgen05_dealloc")
    assert len(calls(tree, "tcgen05_dealloc")) == len(frees) + 1
    assert len(calls(retire, "tcgen05_dealloc")) == 1
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
        elif test == "warp != 17":
            yield from _loader_statements(node.orelse)
        elif test in ("warp == 16", "thread < 2 * query_columns"):
            # Warp 17 holds threads 544-575; the row state belongs to the
            # first 2 * query_columns <= 80 threads.
            yield from _loader_statements(node.orelse)
        else:
            yield node


def _shared_memory_names(body):
    """Names bound to the CTA's shared memory: its allocations (the body
    takes them from ``_allocate_pipeline_shared_memory``) and every pointer
    or tensor derived from them. A value read out of shared memory (an
    element of one of them) is not one."""
    shared = set()
    assignments = [
        (node.targets[0].id, node.value)
        for node in ast.walk(body)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    ]

    def names(node):
        return {part.id for part in ast.walk(node) if isinstance(part, ast.Name)}

    grew = True
    while grew:
        grew = False
        for name, value in assignments:
            if name in shared:
                continue
            if any(
                isinstance(part, ast.Subscript) and shared & names(part.value)
                for part in ast.walk(value)
            ):
                continue
            if calls(value, "_allocate_pipeline_shared_memory") or shared & names(
                value
            ):
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
    # A one-unit CTA's loader arrives here; a persistent CTA's arrives from
    # its role, once it has planned (below).
    (rendezvous,) = [
        node
        for node in body.body
        if isinstance(node, ast.If)
        and any(
            ast.unparse(call.args[0]) == "6" for call in calls(node, "barrier_cta_sync")
        )
    ]
    assert ast.unparse(rendezvous.test) == "warp == 17"
    (one_unit,) = rendezvous.body
    assert ast.unparse(one_unit.test) == "cutlass.const_expr(partition == 'item')"
    assert not one_unit.orelse
    assert [ast.unparse(node) for node in one_unit.body] == [
        "prims.barrier_cta_arrive(6, thread_count=640)"
    ]
    # The other arrival at barrier 6 is the persistent loader's.
    assert (
        sum(
            ast.unparse(call.args[0]) == "6"
            for call in calls(body, "barrier_cta_arrive")
        )
        == 2
    )
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
    loader_init = loader
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
    # Its page helpers only get what the role passes them.
    role = next(
        node
        for node in ast.walk(body)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "warp == 17"
        and calls(node, "_load_pages")
    )
    assert calls(role, "_drain_values")
    names = {
        node.id
        for statement in role.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }
    assert "raw_k_ready" in names
    assert not names & {"base", "state_raw", "scratch_raw", "shared_q_raw", "slot"}
    for helper in ("_load_pages", "_drain_values"):
        loader = function(tree, helper)
        assert calls(loader, "_enqueue_operand_tile")
        used = {node.id for node in ast.walk(loader) if isinstance(node, ast.Name)}
        assert used & {"raw_v_ready"} and not used & {"base", "state", "slot"}
    # A persistent CTA's loader also requests the queries of every unit but
    # the first and owns the stages' barriers with the pipeline's. Which
    # units the CTA runs is the
    # loader's plan. Right after the barrier initialization every thread
    # scans its share of the units into the plan tables and the scanning
    # warps arrive at barrier 7; the loader's warp does not wait there yet.
    planning = body.body[body.body.index(loader_init) + 1]
    assert ast.unparse(planning.test) == "cutlass.const_expr(partition != 'item')"
    assert len(calls(body, "partition_scan")) == 1
    assert len(calls(body, "partition_plan")) == 1
    scan = next(
        i for i, node in enumerate(planning.body) if calls(node, "partition_scan")
    )
    assert isinstance(planning.body[scan], ast.Expr)
    others, quota = planning.body[scan + 1 :]
    assert isinstance(others, ast.If) and ast.unparse(others.test) == "warp != 17"
    assert not others.orelse
    (scanners,) = others.body
    assert ast.unparse(scanners.test) == "warp < plan_warps" and not scanners.orelse
    assert [ast.unparse(node) for node in scanners.body] == [
        "prims.barrier_cta_arrive(7, thread_count=plan_threads)"
    ]
    # Behind its scan the control warpgroup takes its register quota: one
    # instruction that all four of its warps execute, as ``setmaxnreg`` asks
    # of a warpgroup, ahead of the rendezvous, so that the loader's role
    # runs behind it from its first statement and no warp waits for another
    # on its way there. The partition module follows every quota of the
    # template to its warps.
    assert ast.unparse(quota) == (
        "if warp >= 16:\n    cute.arch.setmaxregister_decrease(48)"
    )
    # Barrier 7 counts the scanning warps and the loader's, once.
    threads = next(
        node.value
        for node in planning.body
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "plan_threads"
    )
    for warps, expected in ((1, 64), (17, 576), (18, 576), (20, 640)):
        assert eval(ast.unparse(threads), {"plan_warps": warps}) == expected
    # The loader's warp is itself a scanning warp exactly when the scan
    # takes more than seventeen warps: more than 544 units for each unit a
    # thread scans.
    for units in (544, 545, 640, 641, 1088, 1089, 1280, 1281, 1633, 2000):
        chunk, warps = plan_geometry(units)
        assert (warps > 17) == (units > 544 * chunk), units
    # The roles' dispatch is one statement: a persistent partition's roles,
    # the loader's first, else the one-unit roles from the compute groups on.
    (position,) = [
        i
        for i, node in enumerate(body.body)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.const_expr(partition != 'item')"
        and node.orelse
        and ast.unparse(node.orelse[0].test) == "warp < 8"
    ]
    planned = body.body[position]
    assert calls(planned, "_load_units")
    assert body.body.index(planning) < index < position
    dispatch = planned.body[-1]
    assert isinstance(dispatch, ast.If) and ast.unparse(dispatch.test) == "warp == 17"
    assert all(isinstance(node, ast.Assert) for node in planned.body[:-1])
    # The loader's role, in order, behind the control warps' quota: the wait
    # for the scans at barrier 7, the plan, lane zero's record of it, the
    # arrival at the rendezvous that publishes the record, and the stream,
    # which starts from the role's own copy of the plan.
    role = dispatch.body
    assert ast.unparse(role[0]) == (
        "prims.barrier_cta_sync(7, thread_count=plan_threads)"
    )
    assert isinstance(role[1], ast.Assign) and calls(role[1], "partition_plan")
    assert ast.unparse(role[1].targets[0]) == (
        "(planned_first, planned_last, planned_pages, planned_begin, planned_end, "
        "planned_header, planned_slots, planned_active, planned_skew)"
    )
    written = role[2]
    assert isinstance(written, ast.If) and ast.unparse(written.test) == "lane == 0"
    *record, slots = written.body
    assert [ast.unparse(node) for node in record] == [
        "plan[PLAN_FIRST] = planned_first",
        "plan[PLAN_LAST] = planned_last",
        "plan[PLAN_PAGES] = cutlass.Int32(planned_pages)",
    ]
    # The balanced partition's roles also clip their units to the CTA's
    # slots, and a part of a split unit finds the unit's other CTAs from the
    # slots of all units, the active CTAs and the applied skew.
    assert ast.unparse(slots.test) == "cutlass.const_expr(partition == 'balanced')"
    assert [ast.unparse(node) for node in slots.body] == [
        "plan[PLAN_BEGIN] = planned_begin",
        "plan[PLAN_END] = planned_end",
        "plan[PLAN_HEADER] = planned_header",
        "plan[PLAN_SLOTS] = planned_slots",
        "plan[PLAN_ACTIVE] = planned_active",
        "plan[PLAN_SKEW] = planned_skew",
    ]
    assert ast.unparse(role[3]) == "prims.barrier_cta_arrive(6, thread_count=640)"
    stream = role[4]
    assert len(role) == 5
    assert isinstance(stream, ast.If) and ast.unparse(stream.test) == "lane == 0"
    (started,) = stream.body
    (walk,) = calls(started, "_load_units")
    passed = [ast.unparse(argument) for argument in walk.args]
    assert {"planned_first", "planned_last", "planned_pages"} <= set(passed)
    assert {"query_ready", "query_free", "shared_q_raw"} <= set(passed)
    assert not set(passed) & {"base", "state_raw", "scratch_raw", "slot", "plan"}
    used = {
        node.id
        for node in ast.walk(function(tree, "_load_units"))
        if isinstance(node, ast.Name)
    }
    assert not used & {"base", "state", "slot", "plan", "load_query_panel"}
    # Every other role waited at the rendezvous and reads the record behind
    # it, as its first statement. No role of the persistent dispatch takes a
    # quota for a part of a warpgroup: the conversion warps' is one
    # statement of their eight warps, and the control warps hold theirs.
    reader = function(tree, "_planned_units")
    assert ast.unparse(reader.body[-1]) == (
        "return (cutlass.Int32(plan[PLAN_FIRST]), cutlass.Int32(plan[PLAN_LAST]), "
        "cutlass.Int64(cutlass.Int32(plan[PLAN_PAGES])))"
    )
    (compute_arm,) = dispatch.orelse
    assert ast.unparse(compute_arm.test) == "warp < 8"
    (convert_arm,) = compute_arm.orelse
    assert ast.unparse(convert_arm.test) == "warp < 16"
    control_arm = convert_arm.orelse
    assert [
        ast.unparse(arm[0]) for arm in (compute_arm.body, convert_arm.body, control_arm)
    ] == [
        "first_unit, last_unit, _pages = _planned_units(plan)",
        "_first, _last, count = _planned_units(plan)",
        "first_unit, last_unit, count = _planned_units(plan)",
    ]
    assert len(calls(body, "_planned_units")) == 3
    assert ast.unparse(convert_arm.body[1]) == "cute.arch.setmaxregister_decrease(72)"
    assert [
        ast.unparse(call)
        for call in ast.walk(dispatch)
        if isinstance(call, ast.Call) and "setmaxregister" in ast.unparse(call.func)
    ] == ["cute.arch.setmaxregister_decrease(72)"]
    assert not any(
        isinstance(node, ast.If) and ast.unparse(node.test) == "warp == 17"
        for node in ast.walk(ast.Module(body=control_arm, type_ignores=[]))
    )
    assignments = {
        node.targets[0].id: ast.unparse(node.value)
        for node in ast.walk(body)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    assert assignments["query_free"] == "barriers + barrier_offsets['query_free']"
    # The QK role issues no query copy of its own there, ahead of the
    # rendezvous: a persistent CTA's first copy is issued in its role, behind
    # the plan (``assert_unit_protocol``).
    (copies,) = [
        node
        for node in owner.body
        if isinstance(node, ast.If) and calls(node, "enqueue_paged_query")
    ]
    assert ast.unparse(copies.test) == (
        "cutlass.const_expr(query_map is not None and partition == 'item')"
    )
    # Nor does the loader read what the other roles prepare in the statements
    # it runs without having waited at the rendezvous, which for a persistent
    # CTA is everything up to its stream: there it allocates and derives
    # addresses, initializes its barriers, stores its share of Q, scans, and
    # plans. It reads no element of shared memory itself: the plan tables
    # are read inside ``partition_plan``, behind barrier 7, where every
    # thread that wrote them (``partition_scan``) arrived or is of the
    # loader's own warp, and the plan record is only written. It hands shared
    # memory to no other helper.
    shared = _shared_memory_names(body)
    assert {"slot", "barriers", "state_raw", "shared_q", "raw_k_ready"} <= shared
    assert {"plan_prefix", "plan_totals", "plan", "plan_lengths"} <= shared
    assert not shared & {"base", "first_unit", "last_unit", "count"}
    assert not shared & {"planned_first", "planned_last", "planned_pages"}
    unwaited = list(_loader_statements([*body.body[:position], *role[:-1]]))
    assert one_unit.body[0] in unwaited and role[3] in unwaited
    assert quota in unwaited
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

    def hands_shared(node):
        # An element read passes a value on, not shared memory.
        if isinstance(node, ast.Subscript) and touches_shared(node.value):
            return False
        if isinstance(node, ast.Name):
            return node.id in shared
        return any(map(hands_shared, ast.iter_child_nodes(node)))

    assert {
        ast.unparse(call.func)
        for statement in unwaited
        for call in ast.walk(statement)
        if isinstance(call, ast.Call) and any(map(hands_shared, call.args))
    } == {
        "cute.make_tensor",
        "prims.mbarrier_init",
        "partition_scan",
        "partition_plan",
    }


# The metadata barriers and a persistent partition's query barriers (one or
# two stages) are alternative tails of the block.
@pytest.mark.parametrize(
    "value_metadata,query_stages", [(False, 0), (True, 0), (False, 1), (False, 2)]
)
def test_pipeline_barrier_block_follows_its_table(value_metadata, query_stages):
    table = _pipeline_barriers(value_metadata, query_stages)
    offsets, count, rounds = _pipeline_barrier_plan(value_metadata, query_stages)
    names = [name for name, _width, _arrivals in table]
    assert names[12:] == (
        ["metadata_ready"] * value_metadata
        + ["query_ready", "query_free"] * bool(query_stages)
    )
    # One barrier per stage for the copies, then one per stage for its release.
    assert [width for _name, width, _arrivals in table[12:]] == (
        [4] * value_metadata + [query_stages] * (2 * bool(query_stages))
    )
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
    parts: the allocator after them and what the body gets by name. A
    persistent body is one of 17 units with two query stages."""
    allocator = _RecordingAllocator()
    persistent = "persistent" in parts
    arrays = _pipeline_shared_memory(
        query_columns,
        "value_metadata" in parts,
        2 if persistent else 0,
        17 if persistent else 0,
        "balanced" in parts,
    )
    shared = _allocate_pipeline_shared_memory(allocator, arrays, frozenset(parts))
    return allocator, arrays, vars(shared)


def test_paired_body_allocates_its_shared_memory_table_in_order():
    for parts in (
        (),
        ("query_map",),
        ("value_metadata",),
        ("persistent",),
        ("persistent", "scanned_lengths"),
        ("persistent", "balanced", "scanned_lengths"),
    ):
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
    # Every optional part an entry names is one a body can have: one unit
    # per CTA with a query map and value metadata, or a persistent partition,
    # whose query barriers lie in the block and which has no value metadata.
    # The barrier block is the one ``_pipeline_barrier_plan`` lays out.
    persistent = _pipeline_parts(True, False, 17, True)
    assert persistent == {"persistent", "scanned_lengths"}
    balanced = _pipeline_parts(True, False, 17, True, True)
    assert balanced == persistent | {"balanced"}
    # The scan keeps its lengths only with one unit per scanning thread and
    # a prelude that takes them.
    assert _pipeline_parts(True, False, 17) == {"persistent"}
    assert _pipeline_parts(True, False, 640, True) == persistent
    assert _pipeline_parts(True, False, 641, True) == {"persistent"}
    assert {a.guard for a in arrays} - {None} == (
        _pipeline_parts(True, True) | balanced
    )
    assert _pipeline_parts(False, False) == frozenset()
    assert _pipeline_parts(False, False, 0, True) == frozenset()
    for value_metadata, query_stages in ((False, 0), (True, 0), (False, 1), (False, 2)):
        (block,) = (
            a
            for a in _pipeline_shared_memory(
                40, value_metadata, query_stages, 17 if query_stages else 0
            )
            if a.name == "barriers"
        )
        assert block.count == _pipeline_barrier_plan(value_metadata, query_stages)[1]


def test_paired_shared_memory_stays_inside_its_budget():
    # The transport admits the paired schedule on 180 KiB of shared memory
    # (``capacity >= 184320``). Replay the body's allocations, in order and
    # with their alignment, for every panel width and both optional parts.
    # A persistent partition's own arrays (the compute groups' walk records
    # and posts, the plan tables, the plan and the scanned lengths) are
    # replayed against that partition's budgets in
    # test_cute_paged_attention_partition.py.
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
            # The guarded dependent-launch default of every paired seed.
            assert {key: seed.get(key) for key in CUTE_PDL_CAPTURE_DEFAULT} == dict(
                CUTE_PDL_CAPTURE_DEFAULT
            )
            assert not any(seed.get("load_eviction_policies"))
            counterpart = helion.Config.from_dict({**seed.config, KEY: True})
            assert counterpart in paired
            normalized = bound.config_spec.normalized_config(seed)
            assert normalized.get(KEY) is False
            assert bound.env.backend.autotune_config_is_viable(
                bound.config_spec, normalized
            )
