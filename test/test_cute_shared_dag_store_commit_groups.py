"""CPU checks of the bounded commit group size of a grouped shared-DAG store commit.

``tcgen05_fanout_store_commit_group_size=2`` splits the grouped protocol's ONE bulk
commit group per subtile into contiguous groups of two outputs (three outputs:
2+1, four: 2+2). Each group closes right after its last output's ring write with
its own fence, all-thread barrier, issuer TMA copies and commit; the published
acquire, every ring, drain, release and all arithmetic stay as in the grouped
protocol. ``all`` (the default) is the grouped protocol byte for byte.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import replace
import itertools
from unittest.mock import Mock

import pytest
import torch

from test.test_cute_shared_dag_store_commit import KEY
from test.test_cute_shared_dag_store_commit import SCHEDULES
from test.test_cute_shared_dag_store_commit import SHAPE
from test.test_cute_shared_dag_store_commit import _arithmetic
from test.test_cute_shared_dag_store_commit import _bound_external
from test.test_cute_shared_dag_store_commit import _captured_stores
from test.test_cute_shared_dag_store_commit import _config
from test.test_cute_shared_dag_store_commit import _config_from
from test.test_cute_shared_dag_store_commit import _index
from test.test_cute_shared_dag_store_commit import _loop_lines
from test.test_cute_shared_dag_store_commit import _no_dag_cases
from test.test_cute_shared_dag_store_commit import _without_loop
from test.test_cute_shared_epilogue_dag import bound_original
from test.test_cute_shared_epilogue_dag import configuration

import helion
from helion._compiler.cute.epilogue_fanout import FANOUT_CONFIG_KEY
from helion._compiler.cute.epilogue_fanout import (
    FANOUT_STORE_COMMIT_BOUNDED_STORE_COUNTS,
)
from helion._compiler.cute.epilogue_fanout import (
    FANOUT_STORE_COMMIT_GROUP_SIZE_CONFIG_KEY,
)
from helion._compiler.cute.epilogue_fanout import FANOUT_STORE_COMMIT_GROUP_SIZES
from helion._compiler.cute.shared_epilogue_dag import SharedDagFanoutPlan
from helion._compiler.cute.shared_epilogue_dag import grouped_commit_groups
from helion._compiler.cute.shared_epilogue_dag import grouped_store_commit
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_cuda_work
from helion._testing.cute_dense_tcgen05 import cpu_codegen_target
from helion.autotuner.compiler_coverage import append_compiler_coverage
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
import helion.language as hl

GROUP = FANOUT_STORE_COMMIT_GROUP_SIZE_CONFIG_KEY


@pytest.fixture(autouse=True)
def cpu_only():
    with cpu_codegen_target(), forbid_cuda_work():
        yield


def _sized(size: object, *, placement: str = "post_acc_wait") -> helion.Config:
    config = _config("grouped", placement=placement)
    config.config[GROUP] = size
    return config


def _expected_groups(count: int) -> list[list[int]]:
    return [list(range(i, min(i + 2, count))) for i in range(0, count, 2)]


def check_bounded_protocol(lines: list[str], count: int) -> None:
    """The bounded subtile body, in order: each group's R2S, fence, barrier, TMA, commit."""
    text = "\n".join(lines)
    groups = _expected_groups(count)
    assert text.count(".producer_acquire()") == 1
    assert text.count(".producer_commit()") == len(groups)
    assert text.count("tcgen05_epilog_sync_barrier.arrive_and_wait()") == 1 + len(
        groups
    )
    assert text.count(".arrive_and_wait()") == 1 + len(groups)
    assert text.count("cute.arch.fence_view_async_shared()") == len(groups)
    assert text.count("cute.copy(tcgen05_tiled_copy_r2s") == count
    assert text.count("cute.copy(tcgen05_tma_store_atom") == count
    assert text.count("cute.copy(tcgen05_tiled_copy_t2r") == 1
    acquire = _index(lines, r"tcgen05_c_pipeline\.producer_acquire\(\)")
    tmem = _index(lines, r"cute\.copy\(tcgen05_tiled_copy_t2r", acquire)
    position = _index(lines, r"epilog_sync_barrier\.arrive_and_wait\(\)", tmem)
    for group in groups:
        for k in group:
            suffix = "" if k == 0 else f"_{k}"
            buffer = _index(lines, rf"tcgen05_c_buffer{suffix} = ", position)
            position = _index(
                lines, rf"cute\.copy\(tcgen05_tiled_copy_r2s{suffix}, ", buffer
            )
        # The group's fence, barrier, issuer TMA copies and commit, contiguous.
        assert lines[position + 1] == "cute.arch.fence_view_async_shared()"
        assert lines[position + 2] == "tcgen05_epilog_sync_barrier.arrive_and_wait()"
        assert lines[position + 3].startswith("if tcgen05_warp_idx == cutlass.Int32(0)")
        for offset, k in enumerate(group):
            suffix = "" if k == 0 else f"_{k}"
            assert lines[position + 4 + offset].startswith(
                f"cute.copy(tcgen05_tma_store_atom{suffix}, tcgen05_bSG_sD{suffix}"
                f"[None, tcgen05_c_buffer{suffix}]"
            )
        position += 4 + len(group)
        assert lines[position] == "tcgen05_c_pipeline.producer_commit()"
    assert position == len(lines) - 1 or "advance" in lines[position + 1]
    release = _index(lines, r"acc_pipeline\.consumer_release")
    assert tmem < release


# ---------------------------------------------------------------- bulk-group FIFO model


def _fifo_violations(count: int, stages: int, groups, wait_depth: int) -> int:
    """Simulate one issuer's bulk-group FIFO over many subtiles (in-order completion).

    Each subtile: acquire = ``wait_group.read(wait_depth)`` (complete the oldest
    groups until at most ``wait_depth`` are pending), then write ring ``r``'s slot
    ``u % stages`` for every output and commit each group. A violation is a write
    to a slot that a still-pending group reads.
    """
    pending: list[set[tuple[int, int]]] = []
    violations = 0
    for u in range(4 * stages + 7):
        while len(pending) > wait_depth:
            pending.pop(0)
        for group in groups:
            reads = set()
            for ring in group:
                slot = (ring, u % stages)
                violations += any(slot in group_reads for group_reads in pending)
                reads.add(slot)
            pending.append(reads)
    return violations


@pytest.mark.parametrize("stages", (1, 2, 3, 4))
@pytest.mark.parametrize("count", FANOUT_STORE_COMMIT_BOUNDED_STORE_COUNTS)
def test_bulk_fifo_model_admits_bounded_groups_at_every_ring_depth(count, stages):
    for groups in ([list(range(count))], _expected_groups(count)):
        # The SDK acquire, wait_group.read(num_stages - 1), never lets a write
        # overtake a pending reader, for one group or two groups per subtile.
        assert _fifo_violations(count, stages, groups, stages - 1) == 0
        # With k groups per subtile the newest k * (S - 1) groups are still from
        # the S - 1 previous subtiles, so even that deeper wait is safe: the
        # SDK's S - 1 has margin when k = 2. One group more reaches back to
        # subtile u - S and the model detects the overwrite (negative control).
        k = len(groups)
        assert _fifo_violations(count, stages, groups, k * (stages - 1)) == 0
        assert _fifo_violations(count, stages, groups, k * (stages - 1) + 1) > 0


@pytest.mark.parametrize("count", FANOUT_STORE_COMMIT_BOUNDED_STORE_COUNTS)
def test_commit_groups_are_contiguous_pairs_in_output_order(count):
    stores = [replace(s, commit_group_size=2) for s in _captured_stores(count)]
    assert grouped_commit_groups(stores, True) == tuple(
        tuple(group) for group in _expected_groups(count)
    )
    default = _captured_stores(count)
    assert grouped_commit_groups(default, True) == (tuple(range(count)),)
    assert grouped_commit_groups(default, False) == ()


@pytest.mark.parametrize(
    "fault", ("mixed_sizes", "unknown_size", "bool_size", "serial", "two_outputs")
)
def test_emission_refuses_any_unproved_group_size(fault):
    stores = [replace(s, commit_group_size=2) for s in _captured_stores(4)]
    grouped = grouped_store_commit(stores)
    assert grouped
    if fault == "mixed_sizes":
        stores[-1] = replace(stores[-1], commit_group_size="all")
    elif fault == "unknown_size":
        stores = [replace(s, commit_group_size=3) for s in stores]
    elif fault == "bool_size":
        stores = [replace(s, commit_group_size=True) for s in stores]
    elif fault == "serial":
        stores = [replace(s, commit_mode="serial") for s in stores]
        grouped = grouped_store_commit(stores)
    else:
        stores = stores[:2]
    with pytest.raises(BackendUnsupported):
        grouped_commit_groups(stores, grouped)


# ---------------------------------------------------------------- emission


@pytest.mark.parametrize("placement", ("post_acc_wait", "pre_acc_wait"))
@pytest.mark.parametrize("count", FANOUT_STORE_COMMIT_BOUNDED_STORE_COUNTS)
def test_bounded_groups_render_in_order_with_only_copied_control_lines(
    count, placement
):
    bound = _bound_external(count)
    grouped = bound.to_code(_config("grouped", placement=placement))
    bounded = bound.to_code(_sized(2, placement=placement))
    serial = bound.to_code(_config(None, placement=placement))
    ast.parse(bounded)
    check_bounded_protocol(_loop_lines(bounded), count)
    # Only copies of the grouped protocol's own control lines are added: one
    # fence, barrier, issuer header and commit for the early group.
    added = Counter(_loop_lines(bounded)) - Counter(_loop_lines(grouped))
    removed = Counter(_loop_lines(grouped)) - Counter(_loop_lines(bounded))
    assert removed == Counter()
    assert added == Counter(
        {
            "cute.arch.fence_view_async_shared()": 1,
            "tcgen05_epilog_sync_barrier.arrive_and_wait()": 1,
            "if tcgen05_warp_idx == cutlass.Int32(0):": 1,
            "tcgen05_c_pipeline.producer_commit()": 1,
        }
    )
    assert _arithmetic(_loop_lines(bounded)) == _arithmetic(_loop_lines(grouped))
    assert _arithmetic(_loop_lines(bounded)) == _arithmetic(_loop_lines(serial))
    assert _without_loop(bounded) == _without_loop(grouped) == _without_loop(serial)
    for token in ("PipelineTmaStore.create(", "producer_tail()"):
        assert bounded.count(token) == grouped.count(token) == count, token
    for token in (
        ".consumer_release(tcgen05_acc_consumer_state)",
        "tcgen05_acc_consumer_state.advance()",
        "tcgen05_tmem_allocator.free(",
    ):
        assert bounded.count(token) == grouped.count(token) == 1, token


@pytest.mark.parametrize("count", (2, 3, 4))
def test_all_is_the_grouped_protocol_byte_for_byte_and_stays_implicit(count):
    bound = _bound_external(count)
    spec = bound.config_spec
    grouped = _config("grouped")
    explicit_all = _sized("all")
    assert bound.to_code(explicit_all) == bound.to_code(grouped)
    normalized = spec.normalized_config(explicit_all)
    assert GROUP not in normalized.config
    assert normalized == spec.normalized_config(grouped)
    serial_all = _config(None)
    serial_all.config[GROUP] = "all"
    assert GROUP not in spec.normalized_config(serial_all).config
    assert bound.to_code(serial_all) == bound.to_code(_config(None))
    assert GROUP not in spec.default_config().config
    assert all(GROUP not in seed.config for seed in spec.compiler_seed_configs)


def test_original_bool_modular_four_output_kernel_renders_bounded_groups():
    bound = bound_original()
    config = configuration()
    config.config[KEY] = "grouped"
    grouped = bound.to_code(config)
    config.config[GROUP] = 2
    bounded = bound.to_code(config)
    check_bounded_protocol(_loop_lines(bounded), 4)
    assert _arithmetic(_loop_lines(bounded)) == _arithmetic(_loop_lines(grouped))
    assert _without_loop(bounded) == _without_loop(grouped)
    assert bounded.count("cutlass.Uint8(0)") == grouped.count("cutlass.Uint8(0)") == 1


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="full")
def _bf16_three(
    a: torch.Tensor,
    b: torch.Tensor,
    residual: torch.Tensor,
    out0: torch.Tensor,
    out1: torch.Tensor,
    out2: torch.Tensor,
) -> None:
    rows, reduction = a.shape
    columns = b.shape[1]
    for row, col in hl.tile((rows, columns)):
        acc = hl.zeros([row, col], dtype=torch.float32)
        for inner in hl.tile(reduction):
            acc = torch.addmm(acc, a[row, inner], b[inner, col])
        value = acc + residual[row, col]
        out0[row, col] = value.to(torch.bfloat16)
        out1[row, col] = (value * 2.0).to(torch.bfloat16)
        out2[row, col] = (value - 1.0).to(torch.bfloat16)


def test_bf16_cast_outputs_render_bounded_groups_with_identical_casts():
    bound = cpu_bind(
        _bf16_three,
        (
            torch.empty(SHAPE, dtype=torch.bfloat16),
            torch.empty(SHAPE, dtype=torch.bfloat16),
            torch.empty(SHAPE, dtype=torch.float32),
            *(torch.empty(SHAPE, dtype=torch.bfloat16) for _ in range(3)),
        ),
    )
    (plan,) = bound.config_spec._cute_tcgen05_config.epilogue_fanout_plans
    assert isinstance(plan, SharedDagFanoutPlan) and plan.output_dtype is torch.bfloat16
    grouped = bound.to_code(_config("grouped"))
    bounded = bound.to_code(_sized(2))
    check_bounded_protocol(_loop_lines(bounded), 3)
    assert _arithmetic(_loop_lines(bounded)) == _arithmetic(_loop_lines(grouped))
    assert bounded.count(".to(cutlass.BFloat16)") == grouped.count(
        ".to(cutlass.BFloat16)"
    )
    assert bounded.count(".to(cutlass.BFloat16)") >= 3
    assert _without_loop(bounded) == _without_loop(grouped)


@pytest.mark.parametrize("c_stages", (2, 3, 4))
@pytest.mark.parametrize("schedule", sorted(SCHEDULES))
@pytest.mark.parametrize("count", FANOUT_STORE_COMMIT_BOUNDED_STORE_COUNTS)
def test_bounded_groups_follow_every_admitted_grouped_schedule_and_budget(
    count, schedule, c_stages
):
    bound = _bound_external(count)
    spec = bound.config_spec
    grouped = _config("grouped")
    bounded = _sized(2)
    for config in (grouped, bounded):
        config.config.update(SCHEDULES[schedule], tcgen05_c_stages=c_stages)
    # The bounded size never widens or narrows what the grouped commit admits.
    try:
        grouped_n = spec.normalized_config(grouped)
    except InvalidConfig:
        with pytest.raises(InvalidConfig):
            spec.normalized_config(bounded)
        return
    bounded_n = spec.normalized_config(bounded)
    assert bounded_n.config[GROUP] == 2
    bounded_code = bound.to_code(bounded_n)
    grouped_code = bound.to_code(grouped_n)
    check_bounded_protocol(_loop_lines(bounded_code), count)
    assert _arithmetic(_loop_lines(bounded_code)) == _arithmetic(
        _loop_lines(grouped_code)
    )
    assert _without_loop(bounded_code) == _without_loop(grouped_code)
    assert f"% cutlass.Int32({c_stages})" in bounded_code
    assert f"PipelineTmaStore.create(num_stages={c_stages}," in bounded_code


def test_admitted_schedule_grid_is_pinned():
    admitted = []
    for count, schedule, c_stages in itertools.product(
        FANOUT_STORE_COMMIT_BOUNDED_STORE_COUNTS, sorted(SCHEDULES), (2, 3, 4)
    ):
        config = _sized(2)
        config.config.update(SCHEDULES[schedule], tcgen05_c_stages=c_stages)
        try:
            _bound_external(count).config_spec.normalized_config(config)
        except InvalidConfig:
            continue
        admitted.append((count, schedule, c_stages))
    # c_stages=3 is refused by the shared fanout itself and the four-output c4
    # ring by the AB-stage SMEM budget, exactly as for the grouped commit.
    assert set(admitted) == {
        (count, schedule, c_stages)
        for schedule in SCHEDULES
        for count, c_stages in ((3, 2), (3, 4), (4, 2))
    }


# ---------------------------------------------------------------- normalization


@pytest.mark.parametrize("count", FANOUT_STORE_COMMIT_BOUNDED_STORE_COUNTS)
def test_bounded_size_requires_the_grouped_commit(count):
    bound = _bound_external(count)
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert spec.normalized_config(_sized(2)).config[GROUP] == 2
    for mode in (None, "serial"):
        config = _config(mode)
        config.config[GROUP] = 2
        assert state.fanout_store_commit_group_size_refusal(config.config) is not None
        with pytest.raises(InvalidConfig, match=GROUP):
            spec.normalized_config(config)
        with pytest.raises(InvalidConfig):
            bound.to_code(config)


def test_two_outputs_never_gain_the_field_and_an_explicit_two_is_refused():
    bound = _bound_external(2)
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert state.shared_dag_store_commit_search_enabled
    assert not state.shared_dag_store_commit_group_search_enabled
    with bound.env:
        assert GROUP not in spec.create_config_generation()._flat_fields()
    assert all(group.key != GROUP for group in spec.compiler_coverage_groups)
    with pytest.raises(InvalidConfig, match=GROUP):
        spec.normalized_config(_sized(2))
    with pytest.raises(InvalidConfig):
        bound.to_code(_sized(2))
    raw = dict(_sized(2).config)
    spec.normalize(raw, _fix_invalid=True)
    assert GROUP not in raw and raw[KEY] == "grouped"


@pytest.mark.parametrize("value", (True, False, 1, 3, "2", "bogus", None, 2.0))
def test_mistyped_size_is_refused(value):
    bound = _bound_external(4)
    config = _sized(value)
    with pytest.raises(InvalidConfig, match=GROUP):
        bound.config_spec.normalized_config(config)


@pytest.mark.parametrize(
    "field,value",
    (
        (FANOUT_CONFIG_KEY, "off"),
        ("tcgen05_epi_round_subtiles", 2),
        ("tcgen05_epi_warp_groups", 2),
        ("tcgen05_aux_load_mode", "tma"),
        ("tcgen05_c_store_mode", "direct"),
    ),
)
def test_bounded_size_on_an_unsupported_schedule_fails_loudly(field, value):
    bound = bound_original()
    config = configuration()
    config.config[KEY] = "grouped"
    config.config[GROUP] = 2
    config.config[field] = value
    state = bound.config_spec._cute_tcgen05_config
    assert state.fanout_store_commit_group_size_refusal(config.config) is not None
    with pytest.raises(InvalidConfig):
        bound.config_spec.normalized_config(config)


def test_search_repair_drops_the_size_with_its_grouped_commit():
    bound = _bound_external(4)
    raw = dict(_sized(2).config)
    raw[FANOUT_CONFIG_KEY] = "off"
    bound.config_spec.normalize(raw, _fix_invalid=True)
    assert KEY not in raw and GROUP not in raw
    raw = dict(_config(None).config)
    raw[GROUP] = 2
    bound.config_spec.normalize(raw, _fix_invalid=True)
    assert GROUP not in raw
    raw = dict(_sized(2).config)
    bound.config_spec.normalize(raw, _fix_invalid=True)
    assert raw[GROUP] == 2 and raw[KEY] == "grouped"


def test_strict_round_trip_keeps_the_bounded_size():
    bound = _bound_external(3)
    spec = bound.config_spec
    with bound.env:
        generation = spec.create_config_generation()
        flat, effective = generation.strict_config_pair(_sized(2))
        assert effective.config[GROUP] == 2 and effective.config[KEY] == "grouped"
        assert generation.unflatten(flat) == effective
        assert generation.strict_config_pair(effective) == (flat, effective)


# ---------------------------------------------------------------- schema and coverage


@pytest.mark.parametrize("count", FANOUT_STORE_COMMIT_BOUNDED_STORE_COUNTS)
def test_field_is_prepared_before_coverage_and_reachable_by_canonical_coverage(count):
    bound = _bound_external(count)
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert state.shared_dag_store_commit_group_search_enabled
    with bound.env:
        fields = list(spec.create_config_generation()._flat_fields())
    assert fields.index(KEY) + 1 == fields.index(GROUP)
    with bound.env:
        fragment = spec.create_config_generation()._flat_fields()[GROUP]
    assert tuple(fragment.choices) == FANOUT_STORE_COMMIT_GROUP_SIZES
    assert any(row[0] == GROUP for row in spec._compiler_coverage_fields)
    keys = [group.key for group in spec.compiler_coverage_groups]
    assert keys.index(FANOUT_CONFIG_KEY) < keys.index(KEY) < keys.index(GROUP)
    (group,) = [g for g in spec.compiler_coverage_groups if g.key == GROUP]
    assert group.mechanism == "cute.shared_dag_store_commit_group_size"
    assert tuple(group.domain) == FANOUT_STORE_COMMIT_GROUP_SIZES
    assert group.legacy == "all"
    assert [(d.mechanism, d.key, d.value) for d in group.dependencies] == [
        ("cute.epilogue_fanout", FANOUT_CONFIG_KEY, "shared"),
        ("cute.shared_dag_store_commit", KEY, "grouped"),
    ]
    assert {w.value for w in group.witnesses} == {"all", 2}
    assert all(
        w.carrier.config[FANOUT_CONFIG_KEY] == "shared"
        and w.carrier.config[KEY] == "grouped"
        and GROUP not in w.carrier.config
        for w in group.witnesses
    )
    with bound.env:
        _rows, outcomes = append_compiler_coverage(
            [], spec.create_config_generation(), cached_configs=lambda: (), pin=Mock()
        )
    added = [
        outcome
        for outcome in outcomes
        if outcome.mechanism == group.mechanism and outcome.outcome == "added"
    ]
    assert any(
        outcome.effective is not None and outcome.effective.config.get(GROUP) == 2
        for outcome in added
    )


@pytest.mark.parametrize(
    "case",
    (
        "single_output",
        "masked_store",
        "mixed_dtype",
        "overlapping_outputs",
        "overlapping_residual",
    ),
)
def test_kernels_without_a_shared_dag_plan_never_gain_the_size(case):
    bound = _no_dag_cases()[case]()
    spec = bound.config_spec
    state = spec._cute_tcgen05_config
    assert not state.shared_dag_store_commit_group_search_enabled
    assert all(group.key != GROUP for group in spec.compiler_coverage_groups)
    if state.search_enabled:
        with bound.env:
            assert GROUP not in spec.create_config_generation()._flat_fields()
    assert GROUP not in spec.default_config().config
    config = (
        _config_from(spec.compiler_seed_configs[0])
        if spec.compiler_seed_configs
        else helion.Config(block_sizes=[128, 32, 64])
    )
    config.config[GROUP] = 2
    with pytest.raises(InvalidConfig, match=GROUP):
        spec.normalized_config(config)
