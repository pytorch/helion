"""Check the paired attention schedule's boundary L2 prefetch: which grids
have a form, the config contract, what a positive knob renders and the
structure of the device role."""

from __future__ import annotations

import ast
import copy
import dataclasses
from fractions import Fraction
import functools
from pathlib import Path
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute._paged_attention_prefetch import HEAD_START_SHIFT
from helion._compiler.cute._paged_attention_prefetch import SIBLING_LANES
from helion._compiler.cute.autotune import search_surface
from helion._compiler.cute.autotune.mma_dag_surface import enable_cute_mma_dag_search
from helion._compiler.cute.cute_mma_dag import MMA_DAG_BOUNDARY_PREFETCH_CHOICES
from helion._compiler.cute.cute_mma_dag import paged_attention_boundary_prefetch_choices
from helion._compiler.cute.cute_mma_dag_seeds import BOUNDARY_PREFETCH_SEED_PAGES
from helion._compiler.cute.cute_mma_dag_seeds import MmaDagSeedPlanner
from helion._compiler.cute.paged_attention_config import (
    paged_attention_boundary_prefetch_reason,
)
from helion._compiler.cute.paged_attention_prefetch import BOUNDARY_PREFETCH_IDLE_LIMIT
from helion._compiler.cute.paged_attention_prefetch import BOUNDARY_PREFETCH_WAVE_LIMIT
from helion._compiler.cute.paged_attention_prefetch import boundary_prefetch_form
from helion._compiler.cute.paged_attention_prefetch import boundary_prefetch_layout
from helion._compiler.cute.paged_attention_registers import (
    MMA_DAG_REGISTER_QUOTAS_CHOICES,
)
from helion._compiler.cute.paged_attention_registers import parse_register_quotas
from helion._compiler.cute.pdl_release import capture_policy_available
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_folded_attention import folded_attention_args
from helion._testing.cute_offline import compile_offline
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_paged_partition import partition_attention_args
from helion._testing.cute_paged_partition import partition_attention_kernel
from helion._testing.cute_paged_partition import partition_config
from helion._testing.cute_paged_positive_zero import positive_zero_attention_fixture
from helion._testing.cute_paged_stream import seeds_ahead_of_the_reversed_order
from helion._testing.cute_template_source import source_calls
from helion._testing.cute_template_source import tree_function
from helion.runtime.cute.capture_pdl import WAIT_MODE_HEAD_START_SHIFT

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion._compiler.cute.paged_attention_transport import PagedAttentionTransport

pytestmark = pytest.mark.cpu_only
KEY = "cute_mma_dag_boundary_prefetch"
PAIRED = "attention_paired"
SKIP = "cute_mma_dag_value_metadata_skip"
PARTITION = "cute_mma_dag_partition"
QUOTAS = "cute_mma_dag_register_quotas"
SMS = 148
TEMPLATE = (
    Path(__file__).parents[1]
    / "helion/_compiler/cute/_paged_attention_paired_pipeline.py"
)
DEVICE_HELPERS = TEMPLATE.with_name("_paged_attention_prefetch.py")
PREFETCH_INSTRUCTION = "cp.async.bulk.prefetch.L2.global"
# Units on 148 SMs: a third of the SMs idle (none), at most a fifth of them
# idle (the first CTAs of an overlapped launch serve their siblings), a grid
# that exactly fills the device (none), more than one wave and at most two
# and a half (a unit serves the one a capacity ahead) and three waves (none).
FORMS = {
    100: "",
    128: "sibling",
    147: "sibling",
    148: "",
    149: "wave_ahead",
    160: "wave_ahead",
    300: "wave_ahead",
    444: "",
}


@functools.cache
def _bound(
    units: int,
    sms: int = SMS,
    heads: int = 2,
    input_banks: int = 1,
    bounds: bool = False,
):
    """More than one input bank is the benchmark that times a search's
    candidates back to back in one captured graph; ``bounds`` gives the
    kernel its page range per request in a tensor instead of the length."""
    args = list(partition_attention_args(units=units, rows=8, heads=heads))
    if bounds:
        args[10] = torch.zeros(units, 2, dtype=torch.int32)
    with cpu_render_target(fake_device_sm_count=sms, num_sm=sms):
        return cpu_bind(
            partition_attention_kernel(input_banks=input_banks),
            tuple(args),
            num_sm=sms,
        )


def _config(pages: int | None = None, **overrides: object) -> helion.Config:
    """The paired schedule under its default policies (the guarded capture
    overlap, released from the loader's tail on one wave), V two tiles
    behind K; ``pages`` None leaves the knob absent."""
    knobs: dict[str, object] = {"block_sizes": [8], "cute_mma_dag_v_lag": 2}
    knobs.update(overrides)
    if pages is not None:
        knobs[KEY] = pages
    return partition_config(default_policies="cute_pdl" not in knobs, **knobs)


def _render(
    units: int, config: helion.Config, sms: int = SMS, bounds: bool = False
) -> str:
    with cpu_render_target(fake_device_sm_count=sms, num_sm=sms):
        return _bound(units, sms, bounds=bounds).to_code(config)


# A function of a parsed source by the start of its name (``tree_function``)
# and the calls of a name in a tree (``source_calls``), by the names this
# module's tests call them.
_function = tree_function
_calls = source_calls


def _body_call(kernel: ast.FunctionDef) -> ast.Call:
    (call,) = _calls(kernel, "paged_attention_paired_body")
    return call


def _wrapper_plans(source: str) -> list[dict[str, object]]:
    (line,) = [
        line for line in source.splitlines() if "._helion_cute_wrapper_plans = " in line
    ]
    return ast.literal_eval(line.split(" = ", 1)[1])


def test_a_grid_has_a_form_with_few_idle_sms_or_up_to_two_and_a_half_waves():
    assert Fraction(1, 5) == BOUNDARY_PREFETCH_IDLE_LIMIT
    assert Fraction(5, 2) == BOUNDARY_PREFETCH_WAVE_LIMIT
    for units, form in FORMS.items():
        assert boundary_prefetch_form(units, SMS) == form
    # The sibling form begins where a fifth of the SMs stay idle, 29 of 148,
    # and ends below the capacity, where neither form applies; the wave-ahead
    # form begins above it and ends with two and a half waves, 370 units.
    edges = {
        1: "",
        90: "",
        118: "",
        119: "sibling",
        370: "wave_ahead",
        371: "",
        400: "",
        443: "",
    }
    for units, form in edges.items():
        assert boundary_prefetch_form(units, SMS) == form
    # Both limits belong to the form: exactly a fifth idle, exactly two and
    # a half waves.
    assert [boundary_prefetch_form(units, 100) for units in (79, 80, 250, 251)] == [
        "",
        "sibling",
        "wave_ahead",
        "",
    ]
    # An SM count the target does not report is at most one: no form.
    assert not any(
        boundary_prefetch_form(units, sms) for units in (1, 128, 300) for sms in (0, 1)
    )
    assert MMA_DAG_BOUNDARY_PREFETCH_CHOICES == (0, 8, 12, 4)


@pytest.mark.parametrize("units", sorted(FORMS))
def test_requests_render_exactly_where_the_knob_is_positive_and_a_form_applies(units):
    form = FORMS[units]
    bound = _bound(units)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    assert facts.attention.boundary_prefetch_form == form
    # Off, by a zero or a missing key, is the kernel without the role.
    off = _render(units, _config())
    assert _render(units, _config(0)) == off
    for trace in ("boundary_prefetch", "unit_bounds", "capture_pdl_head_start"):
        assert trace not in off
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        spec = bound.config_spec
        assert KEY not in spec.normalized_config(_config(0)).config
        assert spec.normalized_config(_config(0)) == spec.normalized_config(_config())
        for pages in (8, 12, 4):
            config = _config(pages)
            if form:
                assert spec.normalized_config(config).config[KEY] == pages
                continue
            with pytest.raises(exc.InvalidConfig, match=re.escape(KEY)):
                spec.normalized_config(config)
            spec.normalize(config, _fix_invalid=True)
            assert config == spec.normalized_config(_config())
    if not form:
        return
    source = _render(units, _config(8))
    tree, plain = ast.parse(source), ast.parse(off)
    kernel = _function(tree, "_helion_paged_online_attention")
    helper = _function(tree, "_helion_pa_unit_bounds")
    call = _body_call(kernel)
    added = {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in call.keywords
        if keyword.arg
        and (keyword.arg.startswith("boundary_prefetch") or "unit" in keyword.arg)
    }
    assert added == {
        "boundary_prefetch": "8",
        "boundary_prefetch_form": repr(form),
        "boundary_prefetch_capacity": str(SMS),
        "boundary_prefetch_key": "k",
        "boundary_prefetch_key_scales": "k_scale",
        # A page of raw rows and its scales share one stride of the pool's
        # thirteen pages.
        "boundary_prefetch_layout": "(33792, 33792, 13)",
        "units": str(units),
        "unit_bounds": helper.name,
        "unit_bounds_args": "(lengths, _BLOCK_SIZE_0)",
    }
    # Nothing else of the kernel changes: without those keywords its
    # statements are the ones of the kernel without the knob.
    call.keywords = [keyword for keyword in call.keywords if keyword.arg not in added]
    assert ast.unparse(kernel) == ast.unparse(
        _function(plain, "_helion_paged_online_attention")
    )
    # The module gains the unit function, and only a sibling grid asks the
    # host for its head start.
    assert [node.name for node in tree.body if isinstance(node, ast.FunctionDef)] == [
        helper.name,
        *(node.name for node in plain.body if isinstance(node, ast.FunctionDef)),
    ]
    asked = {**_wrapper_plans(off)[0], "capture_pdl_head_start": True}
    assert _wrapper_plans(source)[0] == (
        asked if form == "sibling" else _wrapper_plans(off)[0]
    )
    assert _wrapper_plans(source)[1] == _wrapper_plans(off)[1]
    for pages in (12, 4):
        assert _render(units, _config(pages)) == source.replace(
            "boundary_prefetch=8", f"boundary_prefetch={pages}"
        )


def test_the_role_evaluates_a_unit_through_the_kernels_own_prelude():
    source = _render(300, _config(8))
    tree = ast.parse(source)
    kernel = _function(tree, "_helion_paged_online_attention")
    helper = _function(tree, "_helion_pa_unit_bounds")
    # Straight-line code of the unit: the tracer runs it as plain Python at
    # every call, and nothing in it names this CTA.
    *assignments, returned = helper.body
    assert isinstance(returned, ast.Return)
    assert all(
        isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        for node in assignments
    )
    assert not re.search(r"\b(pid_\d|num_blocks_\d|block_idx)", ast.unparse(helper))
    assert [argument.arg for argument in helper.args.args] == [
        "tile_offset_0",
        "lengths",
        "_BLOCK_SIZE_0",
    ]
    # Every statement is one of the kernel's own prelude, in its order, and
    # every one of them feeds the unit's item, bounds or length. This kernel
    # derives its page range from the length, so the function reads the
    # length and nothing else.
    prelude = [ast.unparse(node) for node in kernel.body]
    statements = [ast.unparse(node) for node in assignments]
    positions = [prelude.index(statement) for statement in statements]
    assert positions == sorted(positions) and len(set(positions)) == len(positions)
    names: set[str] = {
        node.id for node in ast.walk(returned) if isinstance(node, ast.Name)
    }
    for node in reversed(assignments):
        assert isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        assert node.targets[0].id in names
        names |= {
            part.id for part in ast.walk(node.value) if isinstance(part, ast.Name)
        }
    assert ast.unparse(helper).count("ldg_") == 2
    assert all("lengths.iterator" in ast.unparse(node) for node in assignments[:1])
    # The kernel passes its own item and bounds as before, and the unit
    # function returns the same four expressions.
    call = _body_call(kernel)
    assert ast.unparse(returned.value) == (
        "(" + ", ".join(ast.unparse(argument) for argument in call.args[4:8]) + ")"
    )


# Both forms on a kernel whose page range follows from the length, and one
# on a kernel that takes it from a tensor, which the unit function then
# loads for the served unit as well.
@pytest.mark.parametrize(("units", "bounds"), [(128, False), (300, False), (300, True)])
def test_the_role_names_only_tensors_the_capture_proof_lists_as_loaded(units, bounds):
    source = _render(units, _config(8), bounds=bounds)
    tree = ast.parse(source)
    kernel = _function(tree, "_helion_paged_online_attention")
    helper = _function(tree, "_helion_pa_unit_bounds")
    call = _body_call(kernel)
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    _contract, alias = _wrapper_plans(source)
    tensors = alias["tensor_names"]
    assert isinstance(tensors, list)
    loaded = {
        name
        for name, read, stored in zip(
            tensors, alias["capture_loaded"], alias["stored"], strict=True
        )
        if read and not stored
    }
    inputs = {"q", "k", "k_scale", "v", "v_scale", "lengths", "pages"}
    assert loaded == inputs | ({"bounds"} if bounds else set())
    # What the role gets: the page table (the body's second argument), the K
    # tensors and the arguments of the unit function, which are the tensors
    # the item, bounds and length are read from.
    named = {
        ast.unparse(call.args[1]),
        ast.unparse(keywords["boundary_prefetch_key"]),
        ast.unparse(keywords["boundary_prefetch_key_scales"]),
        *(
            node.id
            for node in ast.walk(keywords["unit_bounds_args"])
            if isinstance(node, ast.Name)
        ),
        *(
            node.id
            for node in ast.walk(helper)
            if isinstance(node, ast.Name) and node.id in tensors
        ),
    }
    read = {"lengths", "bounds"} if bounds else {"lengths"}
    assert named & set(tensors) == {"pages", "k", "k_scale"} | read
    assert named & set(tensors) <= loaded
    # The unit function loads from exactly those: the served unit's length
    # and, where the kernel has them, its two page bounds.
    assert {argument.arg for argument in helper.args.args} & set(tensors) == read
    loads = [
        ast.unparse(node.value)
        for node in helper.body
        if isinstance(node, ast.Assign) and ".iterator" in ast.unparse(node.value)
    ]
    assert sorted({load.split(".iterator")[0] for load in loads}) == sorted(read)
    assert ast.unparse(helper).count("ldg_s32") == len(loads) == (3 if bounds else 2)
    # The device role passes those on and touches no other tensor of the
    # body: the template hands it exactly these arguments.
    body = _function(ast.parse(TEMPLATE.read_text()), "paged_attention_paired_body")
    (role,) = _calls(body, "prefetch_boundary_units")
    assert [ast.unparse(argument) for argument in role.args[:9]] == [
        "pages",
        "boundary_prefetch_key",
        "boundary_prefetch_key_scales",
        "tile // 2",
        "lane",
        "capture_pdl_wait_mode",
        "prefetch_signal",
        "unit_bounds",
        "unit_bounds_args",
    ]
    for name in ("boundary_prefetch_key", "boundary_prefetch_key_scales"):
        uses = [
            node
            for node in ast.walk(body)
            if isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, ast.Load)
        ]
        # An assertion that the tensor was given, and the role's argument.
        assert len(uses) == 2


def test_the_device_role_reads_page_ids_in_range_and_requests_pages_in_the_pool():
    tree = ast.parse(DEVICE_HELPERS.read_text())
    role = _function(tree, "prefetch_boundary_units")
    request = _function(tree, "_request_key_page")
    # One global read beside the unit function's: a page id of the served
    # unit's own range, inside the page table.
    loads = [
        node
        for node in ast.walk(role)
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Load)
        and ast.unparse(node.value) in ("pages", "key", "key_scales")
    ]
    assert [ast.unparse(node) for node in loads] == ["pages[request, logical_page]"]
    guards = [
        node
        for node in ast.walk(role)
        if isinstance(node, ast.If) and any(part is loads[0] for part in ast.walk(node))
    ]
    assert ast.unparse(guards[-1].test) == (
        "(index < count) & (logical_page >= 0) & (logical_page < table_pages)"
    )
    assert "table_pages = cute.size(pages, mode=[1])" in ast.unparse(role)
    # A request is formed only for an id inside the page pool: the entries a
    # range does not have keep the id -1, and a table may hold anything.
    (issue,) = _calls(role, "_request_key_page")
    (bounded,) = [
        node
        for node in ast.walk(role)
        if isinstance(node, ast.If)
        and any(part is issue for part in ast.walk(node))
        and "physical" in ast.unparse(node.test)
    ]
    assert ast.unparse(bounded.test) == (
        "(physical[index] >= 0) & (physical[index] < pool)"
    )
    assert "page = cutlass.Int32(-1)" in ast.unparse(role)
    assert "_raw_page, _scale_page, pool = layout" in ast.unparse(role)
    # The K tensors are only addresses of requests: two byte ranges a page,
    # sizes that are multiples of sixteen bytes, nothing loaded or stored.
    for name in ("key", "key_scales"):
        uses = [
            node
            for node in ast.walk(role)
            if isinstance(node, ast.Name) and node.id == name
        ]
        assert len(uses) == 1 and any(use is uses[0] for use in issue.args)
    prefetches = _calls(request, "cp_async_bulk_prefetch")
    assert [ast.unparse(call.args[0]).split(".iterator")[0] for call in prefetches] == [
        "key",
        "key_scales",
    ]
    assert [ast.unparse(call.args[1]) for call in prefetches] == ["kept * 256", "1024"]
    assert not any(
        keyword.arg == "l2_cache_hint"
        for call in prefetches
        for keyword in call.keywords
    )
    called = {
        ast.unparse(call.func).rpartition(".")[2]
        for function in (role, request)
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
    }
    assert not [
        name
        for name in called
        if re.search(r"^(ldg|stg|ld_|st_|atom|red)|load|store", name)
    ]
    assert not any(
        isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
        for function in (role, request)
        for node in ast.walk(function)
    )


def _key_pool(raw_page: int, scale_page: int) -> tuple[torch.Tensor, torch.Tensor]:
    """K rows and their scales in the fixture's shapes, each in a storage
    of its own with its own page stride."""
    pages, width, heads, dim = 13, 128, 2, 128
    raw = torch.empty(pages * raw_page, dtype=torch.int8)
    scales = torch.empty(pages * scale_page, dtype=torch.uint8)
    return (
        raw.as_strided(
            (pages, 1, width, heads, dim),
            (raw_page, width * heads * dim, heads * dim, dim, 1),
        ),
        scales.as_strided(
            (pages, 1, width, heads, 4),
            (scale_page, width * heads * 4, 4, width * 4, 1),
        ),
    )


def test_a_request_holds_the_rows_the_loader_reads_and_the_pages_scales():
    """``_request_key_page`` on the host, its statements run over integers
    against K tensors on the CPU: the raw request is the byte range of the
    rows of both heads of the keys the loader's box holds, a positive key
    shift leaving out leading keys and a negative one trailing keys, and
    the scale request the page's scale words of both heads. A request
    changes no result, so no launch tells a range beside its page from the
    page's own."""
    request = _function(ast.parse(DEVICE_HELPERS.read_text()), "_request_key_page")
    # The helper without its decorator and annotations, the device's integer
    # types Python's, and a request recorded where the device issues it.
    request.decorator_list = []
    request.returns = None
    for argument in request.args.args:
        argument.annotation = None
    issued: list[tuple[int, int]] = []
    scope: dict[str, object] = {
        "cutlass": SimpleNamespace(Int32=int, Int64=int),
        "prims": SimpleNamespace(
            cp_async_bulk_prefetch=lambda address, size: issued.append((address, size))
        ),
    }
    exec(compile(ast.Module([request], []), str(DEVICE_HELPERS), "exec"), scope)
    request_key_page = cast("Callable[..., None]", scope["_request_key_page"])

    args = partition_attention_args(units=128, rows=8)
    fixture = (cast("torch.Tensor", args[1]), cast("torch.Tensor", args[3]))
    # The fixture's pool, whose raw rows and scales share a page stride (the
    # layout its kernel is rendered with), and a pool that tells the two
    # strides apart.
    assert (fixture[0].stride(0), fixture[1].stride(0)) == (33792, 33792)
    for key, scales in (fixture, _key_pool(34048, 2048)):
        layout = (key.stride(0), scales.stride(0), key.size(0))
        tensors = (
            SimpleNamespace(iterator=key.data_ptr()),
            SimpleNamespace(iterator=scales.data_ptr()),
        )
        for page in (0, 5, 12):
            # A whole box, leading keys left out, trailing keys left out.
            for shift in (0, 1, 37, 127, -1, -90, -127):
                first, last = max(shift, 0), 128 + min(shift, 0)
                rows = key[page, 0, first:last]
                words = scales[page, 0]
                # Each is one run of bytes: a key's two heads are adjacent
                # rows, and a head's scale words are adjacent to the next's.
                assert rows.is_contiguous() and words.transpose(0, 1).is_contiguous()
                issued.clear()
                request_key_page(*tensors, page, 0, shift, layout)
                assert issued == [
                    (rows.data_ptr(), rows.numel()),
                    (words.data_ptr(), words.numel()),
                ], (layout, page, shift)
                # The role's pairs begin at head zero. The helper takes any
                # first head: its ranges begin at that head's row and at its
                # scale words.
                issued.clear()
                request_key_page(*tensors, page, 1, shift, layout)
                assert [address for address, _size in issued] == [
                    key[page, 0, first, 1].data_ptr(),
                    scales[page, 0, 0, 1].data_ptr(),
                ], (layout, page, shift)


def test_the_serving_rule_gives_a_waiting_unit_one_server():
    assert SIBLING_LANES == 6 and HEAD_START_SHIFT == WAIT_MODE_HEAD_START_SHIFT
    role = _function(ast.parse(DEVICE_HELPERS.read_text()), "prefetch_boundary_units")
    (sibling,) = [
        node
        for node in role.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.const_expr(form == 'sibling')"
    ]
    rules = {
        form: {
            ast.unparse(node.targets[0]): ast.unparse(node.value)
            for node in statements
            if isinstance(node, ast.Assign)
        }
        for form, statements in (("wave_ahead", role.body), ("sibling", sibling.body))
    }
    assert rules["wave_ahead"]["target"] == "unit + capacity"
    assert rules["wave_ahead"]["serving"] == "lane == 0"
    assert rules["sibling"] == {
        "head": "capture_pdl_wait_mode >> HEAD_START_SHIFT",
        "target": "unit + head * (lane + 1)",
        "serving": "(unit < head) & (lane < SIBLING_LANES)",
    }
    (served,) = [
        node
        for node in role.body
        if isinstance(node, ast.If) and "serving" in ast.unparse(node.test)
    ]
    assert ast.unparse(served.test) == "serving & (target < units)"

    def servers(form: str, units: int, **values: int) -> dict[int, list[int]]:
        """Unit -> the units that request its pages, by the template's rule."""
        found: dict[int, list[int]] = {}
        for unit in range(units):
            for lane in range(32):
                scope = {"unit": unit, "lane": lane, "SIBLING_LANES": 6, **values}
                target = eval(rules[form]["target"], {}, scope)
                if eval(rules[form]["serving"], {}, scope) and target < units:
                    found.setdefault(target, []).append(unit)
        return found

    # Wave ahead: every unit behind the first wave is served once, by the
    # unit whose SM it takes over; the first wave by nobody.
    for units in (149, 160, 300, 443):
        assert servers("wave_ahead", units, capacity=SMS) == {
            unit: [unit - SMS] for unit in range(SMS, units)
        }
    # Siblings: behind a launch of the same kernel the first ``SMS - units``
    # CTAs have a head start. Each of at most six lanes of such a CTA serves
    # one waiting unit, each waiting unit at most once, in dispatch order.
    for units in (119, 120, 128, 136, 140, 144, 147):
        head = SMS - units
        found = servers("sibling", units, head=head)
        waiting = min(units - head, SIBLING_LANES * head)
        assert sorted(found) == list(range(head, head + waiting))
        assert all(len(units_serving) == 1 for units_serving in found.values())
        assert all(server < head for (server,) in found.values())
        requests: dict[int, int] = {}
        for (server,) in found.values():
            requests[server] = requests.get(server, 0) + 1
        assert max(requests.values()) <= SIBLING_LANES
    # Without a head start (an eager launch, an unproved predecessor, one
    # that left no SM idle) nobody is served, and with more idle SMs than
    # units every unit starts at once.
    for head in (0, 128, 500):
        assert servers("sibling", 128, head=head) == {}


def test_the_wave_ahead_requests_wait_for_the_loaders_last_request():
    tree = ast.parse(TEMPLATE.read_text())
    body = _function(tree, "paged_attention_paired_body")
    # The loader's role with one unit per CTA: the pages, then the V tiles
    # that still lag, then the tail release, then the arrival, as the last
    # statement of the lane that issues the stream.
    (loader,) = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "warp == 17"
        and _calls(node, "_load_pages")
        and not _calls(node, "_load_units")
    ]
    (stream,) = loader.body
    assert isinstance(stream, ast.If) and ast.unparse(stream.test) == "lane == 0"
    pages, drain, release, arrival = stream.body
    # In source order the pages are one call; a bucketed page order splits
    # them around its handoff (test_cute_paged_attention_page_order.py).
    assert isinstance(pages, ast.If)
    assert ast.unparse(pages.test) == "cutlass.const_expr(bucketed)"
    (pages,) = pages.orelse
    assert isinstance(pages, ast.Assign) and _calls(pages, "_load_pages")
    assert isinstance(drain, ast.Expr) and _calls(drain, "_drain_values")
    assert ast.unparse(release.test) == "cutlass.const_expr(capture_release_tail)"
    assert ast.unparse(arrival) == (
        "if cutlass.const_expr(prefetch_signal is not None):\n"
        "    prims.mbarrier_arrive(prefetch_signal)"
    )
    # Every K and V request of the stream is issued inside those two calls,
    # for every V lag: the lagged V tiles of the last pages by the drain.
    issuers = {
        function.name
        for function in tree.body
        if isinstance(function, ast.FunctionDef)
        and _calls(function, "_enqueue_operand_tile")
    }
    assert issuers == {"_load_pages", "_drain_values"}
    drained = _function(tree, "_drain_values")
    assert "for tail in cutlass.range_constexpr(v_lag):" in ast.unparse(drained)
    for call in (*_calls(pages, "_load_pages"), *_calls(drain, "_drain_values")):
        assert "v_lag" in [ast.unparse(argument) for argument in call.args]
    # The barrier has one arrival, this one, and one waiter: the role, which
    # resolves its unit first and requests behind the wait.
    assert [
        ast.unparse(node)
        for node in ast.walk(body)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "prefetch_signal"
    ] == [
        "prefetch_signal = None",
        "prefetch_signal = barriers + barrier_offsets['prefetch_signal']",
    ]
    readers = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Name)
        and node.id == "prefetch_signal"
        and isinstance(node.ctx, ast.Load)
    ]
    (waiter,) = _calls(body, "prefetch_boundary_units")
    # The arrival and its compile-time test, and the role's argument.
    assert len(readers) == 3
    assert sum(any(part is node for part in ast.walk(arrival)) for node in readers) == 2
    assert sum(any(part is node for part in waiter.args) for node in readers) == 1
    role = _function(ast.parse(DEVICE_HELPERS.read_text()), "prefetch_boundary_units")
    (served,) = [
        node
        for node in role.body
        if isinstance(node, ast.If) and "serving" in ast.unparse(node.test)
    ]
    order = [
        "load"
        if "pages[request" in text
        else "wait"
        if "mbar_wait" in text
        else "issue"
        for text in map(ast.unparse, served.body)
        if "pages[request" in text or "mbar_wait" in text or "_request_key_page" in text
    ]
    assert order == ["load", "wait", "issue"]
    (wait,) = _calls(role, "mbar_wait_sleep")
    assert ast.unparse(wait) == "mbar_wait_sleep(signal, cutlass.Int32(0))"
    assert (
        "assert cutlass.const_expr((signal is not None) == (form == 'wave_ahead'))"
        in (ast.unparse(role))
    )


def test_the_bodys_join_waits_for_the_serving_lanes():
    """Only the serving lanes of the warp run the role, in the wave-ahead
    form lane zero behind a wait, and no other role waits for their
    requests. The body's all-thread join (``bar.sync``) stands behind the
    role, and a barrier instruction "causes executing thread to wait for
    all non-exited threads from its warp" (PTX ISA 9.5, section 9.7.15.1):
    the join waits for the requests. The role itself reaches no barrier of
    either form and synchronizes nothing: a body without the join waits for
    no serving lane."""
    body = _function(ast.parse(TEMPLATE.read_text()), "paged_attention_paired_body")
    (join,) = [
        node
        for node in body.body
        if isinstance(node, ast.If) and _calls(node, "barrier")
    ]
    assert [ast.unparse(node) for node in join.body] == ["cute.arch.barrier()"]
    (role,) = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "cutlass.const_expr(boundary_prefetch > 0)"
        and _calls(node, "prefetch_boundary_units")
    ]
    *checks, issue = role.body
    assert all(isinstance(node, ast.Assert) for node in checks)
    assert isinstance(issue, ast.Expr) and _calls(issue, "prefetch_boundary_units")
    assert role.end_lineno is not None and role.end_lineno < join.lineno
    # The role's own code is the one block of some lanes and joins nothing.
    helpers = ast.parse(DEVICE_HELPERS.read_text())
    called = {
        ast.unparse(call.func)
        for call in ast.walk(helpers)
        if isinstance(call, ast.Call)
    }
    assert "mbar_wait_sleep" in called
    assert not [
        name for name in called if "barrier" in name or name.endswith("sync_warp")
    ]


# One config per form, compiled once each beside the same kernel without
# pages: about four seconds a compile.
@pytest.mark.parametrize("units,form", [(128, "sibling"), (300, "wave_ahead")])
def test_the_role_compiles_without_a_device(units, form, tmp_path):
    """The other tests of this module read generated source and the
    template's statements, so a role the DSL cannot lower would show only
    on a device. Compiled the way a launch compiles it, offline (the
    launcher's own wrapper and options, no JIT engine), each form's kernel
    holds the role's requests: two prefetch instructions a page, a page's
    two heads of K and their packed scales. The same kernel without pages
    holds none."""
    pages = 8
    bound = _bound(units)
    args = tuple(partition_attention_args(units=units, rows=8, heads=2))
    source = _render(units, _config(pages))
    assert f"boundary_prefetch_form={form!r}" in source
    kernel = compile_offline(bound, args, source, tmp_path / "pages")
    assert kernel.ptx.count(PREFETCH_INSTRUCTION) == 2 * pages
    assert kernel.threads == 640 and kernel.launch_registers == 96
    without = compile_offline(
        bound, args, _render(units, _config(0)), tmp_path / "none"
    )
    assert "prefetch" not in without.ptx
    # The role adds no barrier and no warp synchronization of its own.
    for instruction in ("bar.warp.sync", "bar.sync", "barrier.sync"):
        assert kernel.ptx.count(instruction) == without.ptx.count(instruction)


# One render per V lag: the loader drains zero to three lagging V tiles
# behind its last page, and the same kernel waits behind all of them.
@pytest.mark.parametrize("v_lag", [0, 1, 2, 3])
def test_every_v_lag_renders_the_wave_ahead_role(v_lag):
    source = _render(300, _config(8, cute_mma_dag_v_lag=v_lag))
    call = _body_call(_function(ast.parse(source), "_helion_paged_online_attention"))
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
    assert keywords.get("v_lag", "0") == str(v_lag)
    assert keywords["boundary_prefetch_form"] == "'wave_ahead'"
    assert "capture_release_tail" not in keywords


def test_the_knob_is_pinned_to_zero_for_configs_that_cannot_run_it():
    bound = _bound(300)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    legal = {KEY: 8, "cute_mma_dag_pass_warps": PAIRED}
    assert paged_attention_boundary_prefetch_reason(facts, legal) is None
    assert paged_attention_boundary_prefetch_reason(facts, {}) is None
    assert paged_attention_boundary_prefetch_reason(facts, {KEY: 0}) is None
    # Another schedule, the value-metadata transport (its actor is the warp
    # that prefetches) and a persistent partition (no unit waits for an SM,
    # and the balanced one's merge helper is that warp) have no role.
    for change in (
        {"cute_mma_dag_pass_warps": "attention"},
        {SKIP: True},
        {PARTITION: "whole"},
        {PARTITION: "balanced"},
    ):
        reason = paged_attention_boundary_prefetch_reason(facts, {**legal, **change})
        assert reason is not None and reason.startswith(
            f"{KEY} must be one of (0,) for"
        )
    for value in (16, 2, -8, True, "8", 8.0):
        reason = paged_attention_boundary_prefetch_reason(facts, {**legal, KEY: value})
        assert reason is not None and "(0, 8, 12, 4)" in reason
    assert paged_attention_boundary_prefetch_choices(facts.attention, PAIRED) == (
        0,
        8,
        12,
        4,
    )
    assert paged_attention_boundary_prefetch_choices(
        dataclasses.replace(facts.attention, paired_items=False), PAIRED
    ) == (0,)
    # Normalization rejects each of them, and repairs it to the same config
    # without the knob: no other option changes.
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        spec = bound.config_spec
        for pages, change in (
            (8, {PARTITION: "whole"}),
            (8, {PARTITION: "balanced"}),
            (16, {}),
            (
                8,
                {
                    "cute_mma_dag_pass_warps": "attention",
                    "cute_mma_dag_query_tma": False,
                },
            ),
        ):
            config = _config(pages, **change)
            with pytest.raises(exc.InvalidConfig, match=re.escape(KEY)):
                spec.normalized_config(config)
            spec.normalize(config, _fix_invalid=True)
            assert KEY not in config.config
            assert config == spec.normalized_config(_config(**change))
        # The knob is one of the paired schedule's: a config whose schedule
        # is decided only after the partition's rules ran, by a mode that is
        # off or by a repair, keeps the identity of the config without pages.
        named = spec.normalized_config(_config(8, cute_mma_dag="off"))
        assert KEY not in named.config
        assert named == spec.normalized_config(_config(cute_mma_dag="off"))
        repaired = _config(8, cute_mma_dag_warps=8)
        spec.normalize(repaired, _fix_invalid=True)
        assert repaired.config["cute_mma_dag_pass_warps"] == "off"
        assert KEY not in repaired.config
        # A kernel with more roots than the paired one has neither form. No
        # kernel of today gets there (the paired schedule owns a kernel's
        # only root), so the test supplies the flag.
        other_roots = copy.copy(spec)
        other_roots.cute_pdl_capture_overlap_available = False
        with pytest.raises(exc.InvalidConfig, match="whose only root"):
            other_roots.normalized_config(_config(8, cute_pdl=False))


# Every admitted register split on a grid past one wave, and one of them on
# a one-wave grid: a split is another kernel to the role, whose warp keeps
# the control quota under each, and the two forms differ in what they render.
@pytest.mark.parametrize(
    ("units", "quotas"),
    [
        *((300, quotas) for quotas in MMA_DAG_REGISTER_QUOTAS_CHOICES),
        (128, MMA_DAG_REGISTER_QUOTAS_CHOICES[-1]),
    ],
)
def test_pages_run_under_every_register_split_of_one_unit_per_cta(units, quotas):
    """The role's warp is a control warp, and every admitted split leaves the
    control quota where the default has it: pages and a split are
    independent options of one unit per CTA. A persistent partition runs
    every split and no pages, whatever the split."""
    counts = parse_register_quotas(quotas)
    assert counts is not None
    assert counts[3] == parse_register_quotas(MMA_DAG_REGISTER_QUOTAS_CHOICES[0])[3]
    bound = _bound(units)
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        spec = bound.config_spec
        split = spec.normalized_config(_config(**{QUOTAS: quotas}))
        both = spec.normalized_config(_config(8, **{QUOTAS: quotas}))
        # Pages add their key to the split's config and change nothing else.
        assert both.config == {**split.config, KEY: 8}
        assert both.config[QUOTAS] == quotas
        for partition in ("whole", "balanced"):
            persistent = {PARTITION: partition, QUOTAS: quotas}
            config = _config(8, **persistent)
            with pytest.raises(exc.InvalidConfig, match=re.escape(KEY)):
                spec.normalized_config(config)
            spec.normalize(config, _fix_invalid=True)
            assert config == spec.normalized_config(_config(**persistent))
            assert config.config[QUOTAS] == quotas and KEY not in config.config
    # The kernel of the split with the role: the split's own kernel with the
    # role's keywords, as under the default split.
    source = _render(units, _config(8, **{QUOTAS: quotas}))
    plain = _render(units, _config(**{QUOTAS: quotas}))
    kernel = _function(ast.parse(source), "_helion_paged_online_attention")
    call = _body_call(kernel)
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
    assert keywords["boundary_prefetch"] == "8"
    assert keywords["boundary_prefetch_form"] == repr(FORMS[units])
    default = quotas == MMA_DAG_REGISTER_QUOTAS_CHOICES[0]
    assert keywords.get("register_quotas") == (None if default else repr(counts))
    call.keywords = [
        keyword
        for keyword in call.keywords
        if not (
            keyword.arg
            and (keyword.arg.startswith("boundary_prefetch") or "unit" in keyword.arg)
        )
    ]
    assert ast.unparse(kernel) == ast.unparse(
        _function(ast.parse(plain), "_helion_paged_online_attention")
    )


# SMs of the device the kernel with the value-metadata transport binds for.
METADATA_SMS = 4


@functools.cache
def _metadata_bound():
    """A kernel that offers the value-metadata transport on a grid with a
    form: five requests on four SMs, a wave and a unit."""
    sms = METADATA_SMS
    kernel, args = positive_zero_attention_fixture(query_len=4, post_scale=-22)
    wide = folded_attention_args(4, batch=5)
    args = (*wide[:7], *args[7:10], wide[10], *args[11:])
    with cpu_render_target(fake_device_sm_count=sms, num_sm=sms):
        bound = cpu_bind(kernel, args, num_sm=sms)
    facts = bound.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    assert facts.attention.value_metadata_skip_available
    assert facts.attention.boundary_prefetch_form == "wave_ahead"
    return bound


def _metadata_config(**knobs: object) -> helion.Config:
    config = optional_lse_config()
    del config.config["cute_pdl"]
    config.config.update(cute_mma_dag_pass_warps=PAIRED, **knobs)
    return config


def test_the_value_metadata_transport_and_the_prefetch_exclude_each_other():
    """The transport's actor is the warp that prefetches."""
    sms = METADATA_SMS
    bound = _metadata_bound()
    spec = bound.config_spec
    config = _metadata_config
    with cpu_render_target(fake_device_sm_count=sms), bound.env, bound.host_function:
        assert spec.normalized_config(config(**{KEY: 8})).config[KEY] == 8
        transport = spec.normalized_config(config(**{SKIP: True}))
        assert transport.config[SKIP] is True
        both = config(**{KEY: 8, SKIP: True})
        with pytest.raises(exc.InvalidConfig, match=re.escape(KEY)):
            spec.normalized_config(both)
        # The repair takes the pages and keeps the transport.
        spec.normalize(both, _fix_invalid=True)
        assert both == transport
    # Each renders alone: the role without the metadata actor, the metadata
    # actor without the role.
    with cpu_render_target(fake_device_sm_count=sms, num_sm=sms):
        role = bound.to_code(config(**{KEY: 8}))
        metadata = bound.to_code(config(**{SKIP: True}))
    assert "boundary_prefetch=8" in role and "value_metadata_scales" not in role
    assert "value_metadata_scales" in metadata and "boundary_prefetch" not in metadata


# What the form of a one-wave grid needs of the dependent-launch policy: each
# case takes one part of the capture contract with the tail release away.
@pytest.mark.parametrize(
    "policy",
    [
        {"cute_pdl": False},
        {"cute_pdl": True},
        {"cute_pdl": True, "cute_pdl_capture_overlap": True},
        {
            "cute_pdl": True,
            "cute_pdl_capture_overlap": True,
            "cute_pdl_release": "entry",
        },
    ],
)
def test_the_sibling_form_needs_the_capture_contract_and_the_tail_release(policy):
    with cpu_render_target(fake_device_sm_count=SMS):
        for units, form in ((128, "sibling"), (300, "wave_ahead")):
            bound = _bound(units)
            with bound.env, bound.host_function:
                spec = bound.config_spec
                config = _config(8, **policy)
                if form == "wave_ahead":
                    # A unit that waits for another unit's SM does so in
                    # every launch, with or without dependent launch.
                    assert spec.normalized_config(config).config[KEY] == 8
                    continue
                with pytest.raises(
                    exc.InvalidConfig,
                    match="cute_pdl=True, cute_pdl_capture_overlap=True and "
                    "cute_pdl_release='loader_tail'",
                ):
                    spec.normalized_config(config)
                spec.normalize(config, _fix_invalid=True)
                assert config == spec.normalized_config(_config(**policy))
                # Spelled out, the default policy keeps the knob.
                kept = spec.normalized_config(
                    _config(
                        8,
                        cute_pdl=True,
                        cute_pdl_capture_overlap=True,
                        cute_pdl_release="loader_tail",
                    )
                )
                assert kept == spec.normalized_config(_config(8))
                assert kept.config[KEY] == 8
    # Past one wave the role renders without the contract too.
    source = _render(300, _config(8, **policy))
    assert "boundary_prefetch_form='wave_ahead'" in source
    assert "capture_pdl_head_start" not in source


def test_the_render_leaves_the_sibling_role_out_without_the_tail_release(monkeypatch):
    """The config contract rejects pages on a one-wave grid without the
    capture contract and the tail release; the render repeats the test
    against the launch it emits. With the contract's check out of the way,
    such a config renders the kernel without the role."""
    monkeypatch.setattr(
        search_surface, "boundary_prefetch_policy_reason", lambda *_: None
    )
    for policy in (
        {"cute_pdl": False},
        {"cute_pdl": True},
        {
            "cute_pdl": True,
            "cute_pdl_capture_overlap": True,
            "cute_pdl_release": "entry",
        },
    ):
        with cpu_render_target(fake_device_sm_count=SMS), _bound(128).env:
            kept = _bound(128).config_spec.normalized_config(_config(8, **policy))
        assert kept.config[KEY] == 8
        assert _render(128, _config(8, **policy)) == _render(128, _config(**policy))
    # The control: with the release the same grid renders the role.
    assert "boundary_prefetch_form='sibling'" in _render(128, _config(8))


def test_one_request_per_page_needs_two_adjacent_heads_of_byte_rows():
    """Descriptor coordinates are [dimension, key, head, subpage, page] for
    the raw rows and [key, head, subpage, page] in packed words for the
    scales (``PagedTmaOperand``)."""

    def layout(**changes: object) -> tuple[int, int, int] | None:
        fields: dict[str, object] = {
            "heads": 2,
            "subpages": 1,
            "key_width": 128,
            "raw_dtype": torch.int8,
            "scale_dtype": torch.uint8,
            "raw_shape": (128, 128, 2, 1, 13),
            "raw_strides": (1, 256, 128, 32768, 33792),
            "scale_shape": (128, 2, 1, 13),
            "scale_strides": (1, 128, 128, 8448),
        }
        assert set(changes) <= set(fields)
        fields.update(changes)
        key = SimpleNamespace(
            raw=SimpleNamespace(dtype=fields["raw_dtype"]),
            scale=SimpleNamespace(dtype=fields["scale_dtype"]),
            raw_shape=fields["raw_shape"],
            raw_strides=fields["raw_strides"],
            scale_shape=fields["scale_shape"],
            scale_strides=fields["scale_strides"],
        )
        transport = SimpleNamespace(
            key=key,
            **{name: fields[name] for name in ("heads", "subpages", "key_width")},
        )
        return boundary_prefetch_layout(cast("PagedAttentionTransport", transport))

    # The fixture's layout: a page of raw rows and its scales share a stride.
    assert layout() == (33792, 33792, 13)
    assert layout(raw_strides=(1, 256, 128, 32768, 34048)) == (34048, 33792, 13)
    for change in (
        {"heads": 4},
        {"subpages": 2},
        {"key_width": 64},
        {"raw_dtype": torch.int16},
        {"scale_dtype": torch.int32},
        # Heads that are not adjacent rows of a key, keys that are not two
        # rows apart.
        {"raw_strides": (1, 256, 256, 32768, 33792)},
        {"raw_strides": (1, 512, 128, 32768, 33792)},
        {"raw_strides": (2, 256, 128, 32768, 33792)},
        # A head's scale words that are not one run of 128, heads whose
        # runs are not adjacent.
        {"scale_strides": (2, 128, 128, 8448)},
        {"scale_strides": (1, 4, 128, 8448)},
        # A scale pool of another size than the raw one.
        {"scale_shape": (128, 2, 1, 12)},
    ):
        assert layout(**change) is None, change


def test_a_layout_or_device_without_a_form_offers_only_zero():
    # Four heads per request: a pair's rows are not one byte range per page.
    with cpu_render_target(fake_device_sm_count=SMS, num_sm=SMS):
        four = cpu_bind(
            partition_attention_kernel(),
            partition_attention_args(units=128, rows=8, heads=4),
            num_sm=SMS,
        )
    facts = four.config_spec.cute_mma_dag_facts
    assert facts is not None and facts.attention is not None
    assert facts.attention.paired_items
    assert facts.attention.boundary_prefetch_form == ""
    # A device of 100 SMs holds 128 units in two waves; one of 40 in four.
    for sms, form in ((100, "wave_ahead"), (40, "")):
        facts = _bound(128, sms).config_spec.cute_mma_dag_facts
        assert facts is not None and facts.attention is not None
        assert facts.attention.boundary_prefetch_form == form
    with cpu_render_target(fake_device_sm_count=SMS), four.env, four.host_function:
        config = _config(8)
        with pytest.raises(exc.InvalidConfig, match=re.escape(KEY)):
            four.config_spec.normalized_config(config)
        four.config_spec.normalize(config, _fix_invalid=True)
        assert KEY not in config.config
    assert "capacity=100" in _render(128, _config(8), sms=100)


def _seeds(bound, spec=None) -> list[helion.Config]:
    spec = bound.config_spec if spec is None else spec
    facts = spec.cute_mma_dag_facts
    assert facts is not None
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        planner = MmaDagSeedPlanner(facts, spec, occupancy_key="cute_min_blocks_per_mp")
        # A windowed kernel under fast math also has the leading family and
        # the prefetch seeds in reversed page order, behind every seed
        # these tests are about.
        return seeds_ahead_of_the_reversed_order(planner.configs(planner.ranked()))


@pytest.mark.parametrize("units", [128, 300])
def test_the_prefetch_is_searched_and_seeded_only_over_bank_cycles(units):
    eager = _bound(units)
    fragment = eager.config_spec._flat_fields()[KEY]
    assert fragment is eager.config_spec.cute_mma_dag_boundary_prefetch
    # Off leads and is all a search without bank cycles draws; every search
    # that draws the knob can draw off.
    assert fragment.choices == (0, 8, 12, 4) and fragment.default() == 0
    assert fragment._active_choices() == (0,)
    assert not eager.config_spec.cute_mma_dag_boundary_prefetch_searched
    banks = _bound(units, input_banks=2)
    assert banks.config_spec.cute_mma_dag_boundary_prefetch_searched
    searched = banks.config_spec._flat_fields()[KEY]
    assert searched.choices == searched._active_choices() == (0, 8, 12, 4)
    # No seed names the knob where it is not searched, and nothing the
    # autotune-off default: it stays the kernel without the role.
    assert not any(KEY in seed.config for seed in _seeds(eager))
    assert BOUNDARY_PREFETCH_SEED_PAGES == (8, 12)
    seeds = _seeds(banks)
    named = [seed for seed in seeds if KEY in seed.config]
    # The leading paired family once more per page count, behind every
    # other seed: 0, the head itself, then 8 and 12.
    head = seeds[0]
    assert head.get("cute_mma_dag_pass_warps") == PAIRED and KEY not in head.config
    assert named == seeds[-2:]
    assert [seed.config for seed in named] == [
        {**head.config, KEY: pages} for pages in (8, 12)
    ]
    # Every other seed keeps its place: they are the seeds of the same
    # benchmark without the knob's search.
    unsearched = copy.copy(banks.config_spec)
    unsearched.cute_mma_dag_boundary_prefetch_searched = False
    assert seeds[:-2] == _seeds(banks, unsearched)
    for bound in (eager, banks):
        spec = bound.config_spec
        with (
            cpu_render_target(fake_device_sm_count=SMS),
            bound.env,
            bound.host_function,
        ):
            default = spec.default_config()
            assert KEY not in default.config
            # A seed is a complete legal config, and a search's row keeps a
            # prefetch, searched or not.
            generation = spec.create_config_generation()
            for seed in named:
                normalized = spec.normalized_config(seed)
                assert normalized.config[KEY] == seed.config[KEY]
                _flat, transferred = generation.canonicalize_flat(
                    generation.flatten(normalized)
                )
                assert transferred.config[KEY] == seed.config[KEY]
            _flat, plain = generation.canonicalize_flat(generation.flatten(default))
            assert KEY not in plain.config
        with cpu_render_target(fake_device_sm_count=SMS, num_sm=SMS):
            assert "boundary_prefetch" not in bound.to_code(default)
    with (
        cpu_render_target(fake_device_sm_count=SMS),
        banks.env,
        banks.host_function,
    ):
        # (Ahead of the same seeds in reversed page order.)
        transferred = [
            config.config[KEY]
            for config in seeds_ahead_of_the_reversed_order(
                [
                    config
                    for _flat, config in banks.config_spec.create_config_generation().seed_flat_config_pairs()
                ]
            )
            if KEY in config.config
        ]
    assert transferred == [8, 12]


def test_a_bind_without_a_form_has_no_prefetch_field():
    for units in (100, 148, 444):
        for banks in (1, 2):
            bound = _bound(units, input_banks=banks)
            assert bound.config_spec.cute_mma_dag_boundary_prefetch is None
            assert KEY not in bound.config_spec._flat_fields()
            assert not any(KEY in seed.config for seed in _seeds(bound))


def test_a_one_wave_grid_offers_pages_only_with_the_capture_policy():
    # On a grid of fewer units than SMs the pages need the guarded capture
    # policy, which a kernel with explicit dependent-launch markers does not
    # take: no config of it could keep pages, so it has no field and no
    # seeds. Past one wave the role needs no dependent launch, and both
    # stay. Markers in the source leave a kernel without the attention
    # schedules altogether (``_paged_attention_schedule_facts``): this is
    # the policy's second gate, and the markers are stated on the spec of a
    # paired bind.
    for units, form in ((128, "sibling"), (300, "wave_ahead")):
        bound = _bound(units, input_banks=2)
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None and facts.attention is not None
        assert facts.attention.boundary_prefetch_form == form
        for markers in (False, True):
            spec = copy.copy(bound.config_spec)
            spec.has_explicit_pdl = markers
            # Under a bank cycle the policy is searched where it is available.
            spec.cute_pdl_capture_overlap_searched = capture_policy_available(spec)
            assert spec.cute_pdl_capture_overlap_searched is not markers
            with (
                cpu_render_target(fake_device_sm_count=SMS),
                bound.env,
                bound.host_function,
            ):
                enable_cute_mma_dag_search(spec, facts)
            offered = form == "wave_ahead" or not markers
            assert (spec.cute_mma_dag_boundary_prefetch is not None) is offered
            assert (KEY in spec._flat_fields()) is offered
            seeds = _seeds(bound, spec)
            assert [seed.config[KEY] for seed in seeds if KEY in seed.config] == (
                list(BOUNDARY_PREFETCH_SEED_PAGES) if offered else []
            )


def _step(spec, parent: helion.Config, **moves: object):
    """The configs one requested step from ``parent`` gives a search: with
    the prerequisites the backend adds, and as requested."""
    generation = spec.create_config_generation()
    base, _config = generation.canonicalize_flat(generation.flatten(parent))
    slots = {key: entry[0][0] for key, entry in generation._key_to_flat_indices.items()}
    requested = list(base)
    for key, value in moves.items():
        requested[slots[key]] = value
    neighbor = generation.conditional_neighbor(base, requested)
    return generation.unflatten(list(neighbor)), generation.unflatten(requested)


def test_a_step_onto_a_prefetch_takes_what_its_role_needs():
    overlap = "cute_mma_dag_value_metadata_read_overlap"
    policy = {
        "cute_pdl": True,
        "cute_pdl_capture_overlap": True,
        "cute_pdl_release": "loader_tail",
    }
    for units, form in ((128, "sibling"), (300, "wave_ahead")):
        bound = _bound(units, input_banks=2)
        facts = bound.config_spec.cute_mma_dag_facts
        assert facts is not None and facts.attention is not None
        # The fixture has no value-metadata transport: supply its graph-free
        # interface, the option a prefetch must leave off.
        spec = copy.copy(bound.config_spec)
        with (
            cpu_render_target(fake_device_sm_count=SMS),
            bound.env,
            bound.host_function,
        ):
            enable_cute_mma_dag_search(
                spec,
                dataclasses.replace(
                    facts,
                    attention=dataclasses.replace(
                        facts.attention, value_metadata_skip_available=True
                    ),
                ),
            )
            assert spec.cute_mma_dag_boundary_prefetch is not None
            plain = _config()
            metadata = _config(
                **{SKIP: True, overlap: True, "cute_mma_dag_key_clip": "off"}
            )
            balanced = _config(**{PARTITION: "balanced"})
            unlaunched = _config(cute_pdl=False)
            assert spec.normalized_config(metadata).config[SKIP] is True
            for parent in (plain, metadata, balanced, unlaunched):
                reached, alone = _step(spec, parent, **{KEY: 8})
                assert reached.config[KEY] == 8
                assert not reached.config.get(SKIP) and not reached.config.get(overlap)
                assert PARTITION not in reached.config
                assert reached.config["cute_mma_dag_v_lag"] == 2
                if form == "sibling":
                    # The units that wait for the previous launch's SMs: the
                    # step takes the capture contract and the tail release.
                    assert {key: reached.config[key] for key in policy} == policy
                else:
                    assert reached.config["cute_pdl"] is (parent is not unlaunched)
                # Alone the step is an alias of where it started wherever
                # the parent lacks a prerequisite: normalization repairs a
                # prefetch without it to none and changes nothing else.
                if parent is plain or (parent is unlaunched and form == "wave_ahead"):
                    assert alone == reached
                else:
                    assert KEY not in alone.config
            # A step back to none, and one between page counts, is the knob's
            # own: nothing else moves.
            on = spec.normalized_config(_config(8))
            for pages in (0, 12):
                reached, alone = _step(spec, on, **{KEY: pages})
                assert reached == alone
                moved = spec.normalized_config(reached)
                assert moved.config.get(KEY, 0) == pages
                assert all(
                    moved.config[key] == value
                    for key, value in on.config.items()
                    if key != KEY
                )
            # A step onto a partition from a prefetching config is the
            # partition's: it drops the prefetch, which has no role there.
            reached, _alone = _step(spec, on, **{PARTITION: "balanced"})
            assert reached.config[PARTITION] == "balanced" and KEY not in reached.config
            # The role needs no register split and takes none: a step onto
            # pages keeps the parent's, and a step between splits the pages.
            split = spec.normalized_config(_config(**{QUOTAS: "120/96/96/48"}))
            assert split.config[QUOTAS] == "120/96/96/48"
            for parent, move in (
                (split, {KEY: 8}),
                (on, {QUOTAS: "136/80/80/48"}),
            ):
                reached, alone = _step(spec, parent, **move)
                assert reached == alone
                moved = spec.normalized_config(reached)
                assert all(
                    moved.config[key] == value
                    for key, value in {**parent.config, **move}.items()
                )
    # The same step on a kernel that offers the transport itself: from a
    # config with the transport and its read overlap, the step drops both,
    # and alone it is an alias of where it started.
    bound = _metadata_bound()
    spec = bound.config_spec
    with (
        cpu_render_target(fake_device_sm_count=METADATA_SMS),
        bound.env,
        bound.host_function,
    ):
        parent = spec.normalized_config(_metadata_config(**{SKIP: True, overlap: True}))
        assert parent.config[SKIP] is True and parent.config[overlap] is True
        reached, alone = _step(spec, parent, **{KEY: 8})
        assert reached.config[KEY] == 8
        assert not reached.config.get(SKIP) and not reached.config.get(overlap)
        assert KEY not in alone.config and alone.config[SKIP] is True


@pytest.mark.parametrize("units", [128, 300])
def test_overrides_keep_the_paired_seeds_with_or_without_their_prefetch(units):
    sibling = FORMS[units] == "sibling"
    bound = _bound(units, input_banks=2)
    spec = bound.config_spec
    policy = {
        "cute_pdl": True,
        "cute_pdl_capture_overlap": True,
        "cute_pdl_release": "loader_tail",
    }

    def transferred(overrides):
        failures: list[str] = []
        generation = spec.create_config_generation(overrides=overrides)
        pairs = generation.seed_flat_config_pairs(failures.append)
        return [config for _flat, config in pairs], failures

    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        seeded, failures = transferred(None)
        assert not failures
        paired = [
            seed for seed in seeded if seed.get("cute_mma_dag_pass_warps") == PAIRED
        ]
        # (The prefetch seeds follow once more in reversed page order, twice
        # each: test_cute_paged_attention_reversed_order.)
        assert [
            seed.config[KEY]
            for seed in seeds_ahead_of_the_reversed_order(seeded)
            if KEY in seed.config
        ] == [8, 12]
        # A fixed prefetch is layered over complete candidates: every paired
        # one takes what the role needs and keeps its place, and a candidate
        # on another schedule is left to that body's rule, which has no role.
        forced, failures = transferred({KEY: 12})
        kept = [
            seed for seed in forced if seed.get("cute_mma_dag_pass_warps") == PAIRED
        ]
        assert kept and all(seed.config[KEY] == 12 for seed in kept)
        assert forced[0] == kept[0] and len(forced) == len(kept)
        # The leading seed's counterparts on the other register splits keep
        # theirs: pages pin no split.
        assert {seed.config[QUOTAS] for seed in paired} == set(
            MMA_DAG_REGISTER_QUOTAS_CHOICES
        )
        assert {seed.config[QUOTAS] for seed in kept} == set(
            MMA_DAG_REGISTER_QUOTAS_CHOICES
        )
        assert len(failures) == len(seeded) - len(paired)
        assert all(KEY in failure for failure in failures)
        if sibling:
            # The units that wait for the previous launch's SMs: also the
            # seeds of the other dependent-launch policies take the guarded
            # one, and fold into the seeds that had it.
            assert all(
                {key: seed.config[key] for key in policy} == policy for seed in kept
            )
        else:
            assert {seed.config["cute_pdl"] for seed in kept} == {True, False}
        # An override that takes away what a seed's own prefetch needs takes
        # its pages, and the seed stays in the search: a partition, either
        # opt-out of the capture policy or another release on a one-wave
        # grid. Past one wave the role needs no dependent launch.
        for overrides, drops in (
            ({PARTITION: "balanced"}, True),
            ({"cute_pdl": False}, sibling),
            ({"cute_pdl_capture_overlap": False}, sibling),
            ({"cute_pdl_release": "entry"}, sibling),
            ({"cute_pdl_spread": True}, sibling),
            ({"cute_pdl": True}, False),
            ({"cute_mma_dag_v_lag": 1}, False),
            ({QUOTAS: MMA_DAG_REGISTER_QUOTAS_CHOICES[-1]}, False),
        ):
            moved, failures = transferred(overrides)
            # (A partition is declined by the seeds of the other schedules.)
            assert not [failure for failure in failures if KEY in failure], overrides
            assert not failures or PARTITION in overrides
            pages = [
                seed.config[KEY]
                for seed in seeds_ahead_of_the_reversed_order(moved)
                if KEY in seed.config
            ]
            assert pages == ([] if drops else [8, 12]), overrides
            # The paired seeds are all still there, the leading one first.
            still = [
                seed for seed in moved if seed.get("cute_mma_dag_pass_warps") == PAIRED
            ]
            assert moved[0] == still[0]
            assert len({repr(sorted(seed.config.items())) for seed in still}) == len(
                still
            )
        # An override that names both the pages and an option their role
        # cannot have is never repaired: the candidates are rejected.
        moved, failures = transferred({KEY: 8, PARTITION: "balanced"})
        assert not moved and len(failures) == len(seeded)
        assert sum(KEY in failure for failure in failures) == len(paired)


@pytest.mark.parametrize("units", [128, 300])
def test_a_selected_config_keeps_its_pages_only_with_the_policy_they_need(units):
    """The kernel also selects configs without a search (an autotune cache
    hit, AOT tuning data), and overrides that decline the guarded capture
    policy decline it for such a config too (``unsearched_config``). On a
    one-wave grid the pages go with the policy, as they do in a search
    candidate: the config stays legal. Past one wave the role needs no
    dependent launch and the pages stay."""
    sibling = FORMS[units] == "sibling"
    bound = _bound(units)
    spec = bound.config_spec
    with cpu_render_target(fake_device_sm_count=SMS), bound.env, bound.host_function:
        selected = spec.normalized_config(_config(8))
        plain = spec.normalized_config(_config())
        for overrides in (
            {"cute_pdl": False},
            {"cute_pdl_capture_overlap": False},
        ):
            compiled = spec.unsearched_config(selected, overrides)
            assert compiled is not selected
            declined = spec.normalized_config(spec.unsearched_config(plain, overrides))
            assert declined.config["cute_pdl"] is False
            pages = {} if sibling else {KEY: 8}
            assert spec.normalized_config(compiled).config == {
                **declined.config,
                **pages,
            }
        # Overrides the policy keeps act on a search only, the knob's own
        # and a partition among them, and so does another schedule.
        for overrides in (
            {"cute_pdl": True},
            {"cute_pdl_release": "entry"},
            {KEY: 0},
            {PARTITION: "balanced"},
            {"cute_mma_dag_pass_warps": "attention"},
        ):
            assert spec.unsearched_config(selected, overrides) is selected
