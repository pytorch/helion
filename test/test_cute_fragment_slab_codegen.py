"""CPU code generation checks for fragment slabs with a runtime offset."""

from __future__ import annotations

import re

import pytest

from helion._compiler.cute.fragment_slab_loads import plan_slab_loads
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import tcgen05_mma_support
from helion._testing.cute_fragment_slab import _config
from helion._testing.cute_fragment_slab import _varlen_anchored_gram
from helion._testing.cute_fragment_slab import _varlen_contiguous_dim_windows
from helion._testing.cute_fragment_slab import _varlen_contiguous_inputs
from helion._testing.cute_fragment_slab import _varlen_gram_inputs

pytestmark = pytest.mark.cpu_only

VEC = "cute_fragment_vector_io"
BATCH = "cute_fragment_batched_loads"
SLAB = "cute_fragment_slab_loads"
SYNC = "cute.arch.sync_threads()"
PACKET = "num_bits_per_copy=128"
# Runtime row addition in the promoted vector dtype, before pointer widening.
SLAB_ROW = (
    r"cutlass\.Int32\(fragment_slab_scalar \+ cutlass\.Int32\(0 \+ [^)\n]*% 32\)\)"
)


@pytest.fixture
def _cpu_only():
    with cpu_render_target(mma_support=tcgen05_mma_support()):
        yield


def test_varlen_windows_at_one_runtime_offset_share_a_slab(_cpu_only: None) -> None:
    bound = cpu_bind(_varlen_anchored_gram, _varlen_gram_inputs())
    spec = bound.config_spec
    # q (two row tiers) and k (rows, then columns) are read at the runtime
    # offset token_base[j] under the valid_len[j] limit; gc's windows are
    # static at the uniform chunk row tile_r.begin: three slabs.
    assert spec.cute_fragment_slab_loads_root_ids
    with bound.env, bound.host_function:
        plan = plan_slab_loads(bound.env, bound.host_function.device_ir.graphs)
    groups = {id(group): group for group in plan.values()}
    assert len(groups) == 3 and len(plan) == 10
    by_tensor = {group.tensor.args[0]: group for group in groups.values()}
    assert sorted(by_tensor) == ["gc", "k", "q"]
    for name in ("q", "k"):
        group = by_tensor[name]
        assert group.offset is not None and group.offset.tensor == "token_base"
        assert group.limit is not None and group.limit.tensor == "valid_len"
        assert (group.start, group.extent) == (0, 32)
    assert by_tensor["gc"].offset is None and by_tensor["gc"].limit is None
    assert by_tensor["gc"].descriptors[0][0] == "uniform"
    assert sorted(by_tensor["q"].members.values()) == [(0, 16), (16, 16)]
    assert sorted(by_tensor["k"].members.values()) == [(0, 16), (0, 32)]
    windows = bound.to_code(_config(bound))
    slabs = bound.to_code(_config(bound, **{SLAB: True}))
    # Ten window loads + the two 0-d table loads become the three slab loads
    # staged ahead of the body, behind the offset and the limit read once
    # per CTA (the body's own 0-d loads of them stay).
    assert windows.count(".load()") == 12
    assert slabs.count(".load()") == 7
    assert slabs.count("fragment_slab_scalar = cutlass.Int32(0)") == 1
    assert slabs.count("fragment_slab_scalar_1 = cutlass.Int32(0)") == 1
    assert "(token_base.iterator + " in slabs and "(valid_len.iterator + " in slabs
    # Every slab row is predicated by the limit (in the window's own
    # coordinates) and the tensor's end at the shifted row.
    assert len(re.findall(r"% 32\) < fragment_slab_scalar_1\n", slabs)) == 2
    assert len(re.findall(SLAB_ROW + r" < \d+\n", slabs)) == 2
    assert slabs.count(SYNC) < windows.count(SYNC)
    # The members read views at their static offsets: tier 1's q rows 16..
    # of the q slab; k's columns the whole k slab.
    assert re.search(r"\(fragment_coordinate_\d+ \+ 16\) \* 128", slabs)
    # With the packets the q / k slabs move as eight-lane bf16 packets under
    # one row predicate each; the gc slab as four-lane fp32 packets.
    everything = bound.to_code(_config(bound, **{SLAB: True, VEC: True, BATCH: True}))
    assert everything.count(".load()") == 4  # the two tables, twice
    assert everything.count("cute.copy(") == 3 * 2 + 2
    assert "cute.make_layout((8,)), cutlass.BFloat16" in everything
    assert "cute.make_layout((4,)), cutlass.Float32" in everything
    assert everything.count(SYNC) < slabs.count(SYNC)


def test_varlen_windows_along_the_contiguous_dim_share_a_slab(
    _cpu_only: None,
) -> None:
    bound = cpu_bind(_varlen_contiguous_dim_windows, _varlen_contiguous_inputs())
    spec = bound.config_spec
    assert spec.cute_fragment_slab_loads_root_ids
    with bound.env, bound.host_function:
        plan = plan_slab_loads(bound.env, bound.host_function.device_ir.graphs)
    groups = {id(group): group for group in plan.values()}
    assert len(groups) == 1 and len(plan) == 2
    (group,) = groups.values()
    # The head is the uniform scalar tile_r.begin // NT; the windows along
    # the contiguous token dim share token_base[j] under valid_len[j].
    assert [kind for kind, _ in group.descriptors] == ["uniform", "window", "window"]
    assert group.window_dim == 2 and (group.start, group.extent) == (0, 32)
    assert group.offset is not None and group.limit is not None
    assert group.offset.tensor == "token_base" and group.limit.tensor == "valid_len"
    windows = bound.to_code(_config(bound))
    slabs = bound.to_code(_config(bound, **{SLAB: True}))
    # Two window loads + two 0-d table loads -> one slab load + the offset
    # and limit read once ahead of it + the body's two table loads.
    assert windows.count(".load()") == 4 and slabs.count(".load()") == 5
    assert slabs.count("fragment_slab_scalar = cutlass.Int32(0)") == 1
    assert slabs.count("fragment_slab_scalar_1 = cutlass.Int32(0)") == 1
    assert "(token_base.iterator + " in slabs and "(valid_len.iterator + " in slabs
    # Each slab column is live below the limit (in the window's own
    # coordinates) and inside the tensor's 128 tokens; the offset shifts
    # the unit-stride token index.
    assert len(re.findall(r"% 32\) < fragment_slab_scalar_1\n", slabs)) == 1
    assert len(re.findall(SLAB_ROW + r" < 128\n", slabs)) == 1
    assert re.search(
        rf"cutlass\.Int64\({SLAB_ROW}\) \* cutlass\.Int64\(1\)",
        slabs,
    )
    assert re.search(r"\+ 16\) \* 1\]", slabs)
    assert slabs.count(SYNC) < windows.count(SYNC)
    # Varying along the contiguous dim, the slab is walked under every
    # transport knob.
    everything = bound.to_code(_config(bound, **{SLAB: True, VEC: True, BATCH: True}))
    assert everything.count(".load()") == 5
    assert everything.count(PACKET) == 1 and everything.count("cute.copy(") == 1
