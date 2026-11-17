"""Neither store policy changes an output bit of the paired attention
schedule: under ``streaming`` and under ``l2_evict_last`` its output stores
write the same values to the same addresses with another cache operator or
under a cache hint, eager and as captured calls that overlap their
predecessors, with one unit per CTA and on the persistent partitions, whose
compute groups store between two units and after a merge.

These tests launch the stores; they cannot tell which cache operator or
hint a store carries. The instruction is pinned by
``test_cute_paged_attention_store_policy.py``."""

from __future__ import annotations

from typing import cast

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipIfRefEager
from helion._testing.cute_folded_attention import folded_attention_args
from helion._testing.cute_optional_lse import initialize_paged_attention_inputs
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_paged_bounded import bounded_partial_fixture
from helion._testing.cute_paged_headwide import configure_headwide_case
from helion._testing.cute_paged_headwide import headwide_config
from helion._testing.cute_paged_headwide import headwide_kernel
from helion._testing.cute_paged_partition import PAGE
from helion._testing.cute_paged_partition import PAGES
from helion._testing.cute_paged_partition import SHORT_WINDOW
from helion._testing.cute_paged_rowwise_native import ROW_CASES
from helion._testing.cute_paged_stream import IDLE_SMS
from helion._testing.cute_paged_stream import assert_same_bits
from helion._testing.cute_paged_stream import body_keywords
from helion._testing.cute_paged_stream import check_same_bits
from helion._testing.cute_paged_stream import one_wave_units
from helion._testing.cute_paged_stream import poison_results
from helion._testing.cute_paged_stream import replayed_bits
from helion._testing.cute_paged_stream import stream_pair
from helion._testing.cute_paged_stream import stream_patterns
from helion._testing.poisoned_outputs import poisoned_launch

KEY = "cute_store_policy"
PARTITION = "cute_mma_dag_partition"
# The policies that change the stores: the streaming cache operator, and
# the L2 evict-last hint with a policy word each thread makes.
POLICIES = ("streaming", "l2_evict_last")


def _check(
    policy: str,
    window: int,
    calls: int,
    *,
    rows: int = 8,
    reserved_sms: int = 0,
    **schedule: object,
) -> None:
    """The kernel with the default stores against the kernel with the
    stores of ``policy``, over streams of every length the table has."""
    args, default, candidate, plain, source = stream_pair(
        {KEY: policy},
        schedule=schedule,
        window=window,
        rows=rows,
        reserved_sms=reserved_sms,
    )
    assert "store_policy" not in body_keywords(plain)
    assert body_keywords(source)["store_policy"] == repr(policy)
    units = cast("torch.Tensor", args[6]).numel()
    check_same_bits(
        args,
        default,
        candidate,
        stream_patterns(units, window),
        calls=calls,
        what=(policy, rows, schedule),
    )


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("rows", [8, 16, 40])
def test_policy_stores_keep_every_output_bit(rows, policy):
    """One unit per CTA, the whole table as the window: units of zero to
    seventeen pages, a unit without pages among them, whose stores are its
    only work. Eight query rows store 8 KiB a unit, forty 40 KiB; sixteen
    are fewer rows than their block's 32 query columns, so a thread has
    columns without a row to store."""
    _check(policy, PAGES * PAGE - 1, 3, rows=rows)


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("partition", ["whole", "balanced"])
def test_policy_stores_keep_every_bit_on_a_persistent_partition(partition, policy):
    """A third of the grid's units per SM left to the partition: every CTA
    runs three units or parts of them, so a compute group stores between
    two units while the other roles stage the next one, and under the
    balanced partition a unit in parts is stored by the part that merges
    it."""
    units = one_wave_units()
    _check(
        policy,
        SHORT_WINDOW,
        3,
        rows=32,
        # (The grid's units and the SMs it leaves idle are the device's SMs.)
        reserved_sms=units + IDLE_SMS - units // 3,
        **{PARTITION: partition},
    )


# One compile pair each: the late retirement, whose stores precede the
# body's final join, sleeping waits, V issued with its K, and a raw V ring
# of three pages beside a boundary prefetch, the other occupants of a
# one-unit CTA's epilogue.
SCHEDULES = {
    "joined": {"cute_mma_dag_early_tmem_retirement": False},
    "lag3-sleep": {"cute_mma_dag_v_lag": 3, "cute_mma_dag_sleep_waits": True},
    "lag0": {"cute_mma_dag_v_lag": 0},
    "ring3-prefetch8": {
        "cute_mma_dag_value_ring": 3,
        "cute_mma_dag_boundary_prefetch": 8,
    },
}


@pytest.mark.slow
@pytest.mark.timeout(300)
@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("schedule", list(SCHEDULES))
def test_policy_stores_keep_every_bit_across_schedule_knobs(schedule, policy):
    """A short window: a long request starts at a later page and streams at
    most four pages, so most of a launch's time is its stores."""
    _check(policy, SHORT_WINDOW, 2, **SCHEDULES[schedule])


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("query_len", [1, 5])
def test_policy_stores_write_a_folded_output_at_its_own_addresses(query_len, policy):
    """Outputs proved from explicit storage indices: padded rows behind a
    pointer offset, which the kernel addresses by the storage's strides.
    The kernel under a policy leaves every byte of the output's storage as
    the default kernel leaves it, the padding between the rows included,
    and the same LSE, over the fixture's cases."""
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention uses SM100 tensor memory")
    values = list(folded_attention_args(query_len, device=DEVICE))
    values[7] = 2047
    args = tuple(values)
    configure_headwide_case(args, "original")
    bound = headwide_kernel().bind(args)
    config = headwide_config(query_len)
    assert config.config.get(KEY, "default") == "default"
    candidate = headwide_config(query_len)
    candidate.config[KEY] = policy
    plain, source = bound.to_code(config), bound.to_code(candidate)
    assert "store_policy" not in body_keywords(plain)
    assert body_keywords(source)["store_policy"] == repr(policy)
    assert "io_layouts" in body_keywords(source)
    kernels = [bound.compile_config(each) for each in (config, candidate)]
    supplied = cast("torch.Tensor", args[10])
    storage = supplied.untyped_storage()
    backing = torch.empty(0, device=supplied.device, dtype=torch.uint8).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )
    for case in ROW_CASES:
        configure_headwide_case(args, case)
        written = []
        for kernel in kernels:
            backing.fill_(0x5A)
            results = cast("tuple[torch.Tensor, ...]", kernel(*args))
            torch.cuda.synchronize()
            written.append(
                (
                    backing.clone(),
                    *(value.clone().view(torch.int32) for value in results[1:]),
                )
            )
        # Some byte of the storage is an output, and some is padding.
        assert bool((written[0][0] != 0x5A).any()), case
        assert bool((written[0][0] == 0x5A).any()), case
        for value, reference in zip(written[1], written[0], strict=True):
            assert torch.equal(value, reference), case


@skipIfNotCUDA()
@skipIfRefEager("capture dependency proof needs native CUDA graph nodes")
@pytest.mark.parametrize("policy", POLICIES)
def test_policy_stores_keep_every_bit_of_a_raw_statistics_epilogue(policy):
    """The epilogue that stores each unit's unnormalized fp32 panel beside
    its row maxima and denominators: under a policy the panel's stores, four
    bytes an element, leave every bit of the three outputs as the default
    kernel leaves it, eager and as a captured call replayed over changed
    inputs."""
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("paired attention uses SM100 tensor memory")
    kernel, initial = bounded_partial_fixture(4, DEVICE)
    values = (*initial[:7], 31, *initial[8:])
    q, pages, lengths = (cast("torch.Tensor", values[index]) for index in (0, 5, 6))
    panel = cast("torch.Tensor", values[10])
    assert panel.dtype == torch.float32
    initialize_paged_attention_inputs(values)
    bound = helion.kernel(
        kernel.fn, backend="cute", static_shapes=True, fast_math=True
    ).bind(values)
    kernels = []
    for candidate in ("default", policy):
        config = optional_lse_config()
        config.config.update(
            block_sizes=[32],
            cute_mma_dag_pass_warps="attention_paired",
            cute_min_blocks_per_mp=1,
            cute_mma_dag_query_tma=True,
            **{KEY: candidate},
        )
        keywords = body_keywords(bound.to_code(config))
        assert "maximum_output" in keywords and "denominator_output" in keywords
        assert keywords.get("store_policy") == (
            None if candidate == "default" else repr(policy)
        )
        kernels.append(bound.compile_config(config))

    def prepare(case: str) -> None:
        initialize_paged_attention_inputs(values)
        lengths.copy_(lengths.new_tensor([256, 257]))
        if case == "changed":
            q.neg_()
            pages.copy_(pages.flip(1))

    def launched(compiled: object) -> list[torch.Tensor]:
        return poisoned_launch(lambda: compiled(*values), preallocated=[panel])  # type: ignore[operator]

    expected = {}
    for case in ("ordinary", "changed"):
        prepare(case)
        expected[case] = launched(kernels[0])
        # The panel, the maxima and the denominators.
        assert len(expected[case]) == 3
        assert_same_bits(launched(kernels[1]), expected[case], (policy, case, "eager"))
    assert not torch.equal(expected["ordinary"][0], expected["changed"][0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = kernels[1](*values)
    for case in expected:
        prepare(case)
        poison_results([captured])
        graph.replay()
        torch.cuda.synchronize()
        assert_same_bits(replayed_bits(captured), expected[case], (policy, case))
