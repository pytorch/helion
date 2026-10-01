from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

from benchmarks.cute.kda_prefill_fused_bt32 import kda_prefill_native_math_bt32
import pytest
import torch

from test.test_cute_state_transfer_cap import _cpu
from test.test_cute_state_transfer_cap import _inputs
from test.test_cute_state_transfer_cap import _plan

import helion
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chunk_prefill_prepared_state_copy as adapter
from helion._compiler.cute.prepared_state_copy import raw_state_copy_eligible
from helion._compiler.device_function import DeviceFunction
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
from helion.runtime.cute.chunk_prefill import validate_plan


@pytest.fixture(scope="module", params=("tma", "tma_pipelined", "tma_planar"))
def captured_copy_data(request):
    records = []
    contexts = []
    build = adapter.bind_fast_state_copy

    def observe(recurrence, root, *, pipelined=False, planes_per_transfer=1):
        result = build(
            recurrence,
            root,
            pipelined=pipelined,
            planes_per_transfer=planes_per_transfer,
        )
        records.extend(
            (
                result,
                build(
                    recurrence,
                    root,
                    pipelined=pipelined,
                    planes_per_transfer=planes_per_transfer,
                ),
            )
        )
        contexts.append(DeviceFunction.current())
        return result

    with _cpu(), patch.object(adapter, "bind_fast_state_copy", observe):
        bound = kda_prefill_native_math_bt32._bind_isolated(_inputs())
        old = bound.to_code(helion.Config(block_sizes=[64]))
        explicit = bound.to_code(
            helion.Config(block_sizes=[64], cute_state_transfer_transport="register")
        )
        new = bound.to_code(
            helion.Config(block_sizes=[64], cute_state_transfer_transport=request.param)
        )
    assert old == explicit
    assert records  # The public emitter invokes the real graph/lease binder.
    return records, _plan(old), _plan(new), bound, contexts[-1]


@pytest.fixture
def captured_copy(captured_copy_data):
    records, old, new, bound, context = captured_copy_data
    with _cpu(), bound.env, bound.host_function, context:
        yield records, old, new


def test_public_transport_preserves_original_programs_and_default(captured_copy):
    records, old, new = captured_copy
    new = dict(new)
    payload = new.pop("prepared_state_copy_program")
    assert new == old
    assert payload in (
        (1, 128, 128, 32, 210944, 64, 536),
        (2, 128, 128, 32, 210944, 64, 536, 168960, 28, 32),
        (3, 128, 128, 32, 168960, 64, 536, 2, 28, 32),
    )
    for plan in records:
        plan.check()
        assert plan.payload() == payload
        assert [r.source_view.offset for r in plan.schedule.requests] == [0, 32, 64, 96]
        assert plan.schedule.cuts[0].anchor is plan.lease.cycle.owner.loop
        assert plan.lease.publication.packed_cut.anchor.owner is plan.lease.recurrence
        assert plan.lease.output.store.anchor.owner is plan.lease.recurrence


@pytest.mark.parametrize(
    "field,value",
    (
        ("byte_offset", 1024),
        ("byte_size", 8192),
        ("alignment", 16),
        ("live_from", 2),
        ("live_until", 6),
    ),
)
def test_scratch_cannot_escape_original_output_lease(captured_copy, field, value):
    plan = captured_copy[0][0]
    with pytest.raises(chain._UnsupportedChain, match="scratch lease"):
        replace(plan, scratch=replace(plan.scratch, **{field: value}))


@pytest.mark.parametrize(
    "field,value", (("byte_offset", 528), ("stages", 2), ("arrivals", 4))
)
def test_copy_cannot_reuse_foreign_completion(captured_copy, field, value):
    plan = captured_copy[0][0]
    with pytest.raises(chain._UnsupportedChain, match="scratch lease"):
        replace(plan, barriers=replace(plan.barriers, **{field: value}))


def test_replaced_owner_and_cut_are_rejected(captured_copy):
    first, other = captured_copy[0][:2]
    with pytest.raises(chain._UnsupportedChain, match="scratch lease"):
        replace(first, schedule=other.schedule)
    with pytest.raises(chain._UnsupportedChain):
        replace(first, tmem_column=0)
    original = first.lease.cycle.mode
    try:
        first.lease.cycle.mode = 1
        with pytest.raises(chain._UnsupportedChain):
            first.check()
    finally:
        first.lease.cycle.mode = original
    first.check()


@pytest.mark.parametrize("port_index", range(4))
@pytest.mark.parametrize("field", ("descriptor", "ready", "complete"))
def test_physical_port_changed_before_binding_is_rejected(
    captured_copy, port_index, field
):
    plan = captured_copy[0][0]
    recurrence = plan.lease.recurrence
    ports = list(recurrence.ports)
    port = ports[port_index]
    original = getattr(port, field)
    changed = (original[0] + 64, *original[1:]) if original else (64,)
    ports[port_index] = replace(port, **{field: changed})
    with pytest.raises(chain._UnsupportedChain, match="scratch lifetime"):
        adapter.bind_fast_state_copy(
            replace(recurrence, ports=tuple(ports)), plan.lease.cycle.owner.root
        )


@pytest.mark.parametrize("index", range(7))
def test_serialized_copy_resource_mutations_rejected(captured_copy, index):
    plan = dict(captured_copy[2])
    payload = list(plan["prepared_state_copy_program"])
    payload[index] += 1
    plan["prepared_state_copy_program"] = tuple(payload)
    with pytest.raises(ValueError, match="raw state-copy program"):
        validate_plan(plan)


def test_copy_requires_original_external_state_program(captured_copy):
    plan = dict(captured_copy[2])
    plan.pop("prepared_state_abi_program")
    with pytest.raises(BackendUnsupported, match="bound state ABI"):
        validate_plan(plan)


@pytest.mark.parametrize("alignment", (1, 2, 4, 8, 16, 32, 256))
def test_pointer_alignment_is_a_real_transport_precondition(alignment):
    assert raw_state_copy_eligible(
        (2, 8, 128, 128), (131072, 16384, 128, 1), alignment
    ) is (alignment >= 16)


@pytest.mark.parametrize(
    "strides,accepted",
    (
        ((135168, 16896, 132, 1), True),  # padded rows/heads/sequences
        ((131072, 16384, 128, 2), False),
        ((131072, 16384, 129, 1), False),
        ((131072, 16385, 128, 1), False),
        ((0, 16384, 128, 1), False),
        ((-131072, 16384, 128, 1), False),
        ((2**38, 16384, 128, 1), False),
    ),
)
def test_unsupported_strides_keep_register_transport(strides, accepted):
    assert raw_state_copy_eligible((2, 8, 128, 128), strides, 16) is accepted


@pytest.mark.parametrize("layout", ("offset1", "offset2", "stride2"))
def test_public_bind_does_not_admit_tma_for_unsupported_storage(layout):
    values = list(_inputs())
    shape = values[7].shape
    count = values[7].numel()
    if layout == "stride2":
        values[7] = torch.empty((*shape[:-1], shape[-1] * 2), dtype=torch.float32)[
            ..., ::2
        ]
    else:
        offset = 1 if layout == "offset1" else 2
        values[7] = torch.empty(count + offset, dtype=torch.float32)[offset:].view(
            shape
        )
        assert values[7].data_ptr() % 16 == offset * 4
    with _cpu():
        bound = kda_prefill_native_math_bt32._bind_isolated(tuple(values))
        assert bound.config_spec.cute_state_transfer_transport is None
        with (
            bound.env,
            bound.host_function,
            pytest.raises(InvalidConfig, match="requires a bound"),
        ):
            bound.config_spec.normalized_config(
                helion.Config(cute_state_transfer_transport="tma")
            )
        source = bound.to_code(helion.Config(block_sizes=[64]))
    assert "'kind': 'chunk_prefill_sm100'" not in source
    assert "prepared_state_copy_program" not in source


@pytest.mark.parametrize(
    "field,value", (("group", 3), ("groups", 4), ("byte_size", 8192))
)
def test_startup_owner_cannot_be_replaced(captured_copy, field, value):
    plan = captured_copy[0][0]
    if plan.startup_reuse is None:
        with pytest.raises(chain._UnsupportedChain, match="scratch lease"):
            replace(plan, startup_reuse=adapter.fast_startup_reuse())
    else:
        with pytest.raises(chain._UnsupportedChain, match="scratch lease"):
            replace(plan, startup_reuse=replace(plan.startup_reuse, **{field: value}))


@pytest.mark.parametrize("index", (7, 8, 9))
def test_serialized_startup_owner_mutations_rejected(captured_copy, index):
    plan = dict(captured_copy[2])
    payload = list(plan["prepared_state_copy_program"])
    if payload[0] == 1:
        payload += [168960, 28, 32]
    payload[index] += 1
    plan["prepared_state_copy_program"] = tuple(payload)
    with pytest.raises(ValueError, match="raw state-copy program"):
        validate_plan(plan)


@pytest.mark.parametrize("planes", (0, True, 3, 4))
def test_copy_rejects_unproved_panel_grouping(captured_copy, planes):
    plan = captured_copy[0][0]
    with pytest.raises(chain._UnsupportedChain):
        replace(plan, planes_per_transfer=planes)


def test_planar_scratch_is_only_borrowed_producer_region(captured_copy):
    plan = captured_copy[0][0]
    if plan.planes_per_transfer != 2:
        return
    revised, borrowed = plan.startup_reuse.resources(plan.pipeline)
    assert plan.scratch == borrowed
    assert plan.scratch.byte_size == 32768
    assert plan.barriers.stages == 3
    assert plan.startup_reuse.first_warp == 28
    assert plan.startup_reuse.last_warp == 32
    assert revised.shared_buffer("factor_stages_4").live_from == borrowed.live_until
    assert revised.shared_buffer("output_stages") == plan.pipeline.shared_buffer(
        "output_stages"
    )
    with pytest.raises(chain._UnsupportedChain):
        replace(plan, scratch=replace(borrowed, name="foreign_copy"))
    with pytest.raises(chain._UnsupportedChain):
        replace(plan, scratch=replace(borrowed, byte_size=16384))
    with pytest.raises(chain._UnsupportedChain):
        adapter.bind_fast_state_copy(
            plan.lease.recurrence,
            plan.lease.cycle.owner.root,
            pipelined=True,
            planes_per_transfer=2,
        )
