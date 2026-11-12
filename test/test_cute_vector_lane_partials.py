"""A cross-thread reduce that ``hoist_warp_reduce`` leaves in a constexpr
V-loop combines one V lane of every thread: a partial of the reduction over
the loop's block.  Values the same in every lane and later reduces may fold
the partials; per-element values computed from one may not, and such a
config is refused (``_per_lane_reduce_escape``)."""

from __future__ import annotations

import ast
import textwrap

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.hoist_warp_reduce import hoist_warp_reduce_from_vloop
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _hoist(src: str) -> str:
    body = hoist_warp_reduce_from_vloop(ast.parse(textwrap.dedent(src)).body)
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def test_a_partial_stored_per_element_is_refused() -> None:
    """``v - v.sum()``: each element would be stored minus one lane's sum."""
    src = """
    for vec_lane_1 in cutlass.range_constexpr(4):
        v = cutlass.Float32(_v0[vec_lane_1])
        s = cutlass.Float32(cute.arch.warp_reduction_sum(v, threads_in_group=4))
        d = v - s
        vals.append(d)
    """
    with pytest.raises(exc.BackendUnsupported, match="other than an assignment reads"):
        _hoist(src)


def test_a_partial_carried_across_the_lanes_is_refused() -> None:
    """A carry updated in every lane with one lane's sum adds the partials
    where the program adds the whole sum once per element."""
    src = """
    for vec_lane_1 in cutlass.range_constexpr(4):
        v = cutlass.Float32(_v0[vec_lane_1])
        s = cutlass.Float32(cute.arch.warp_reduction_sum(v, threads_in_group=4))
        acc = acc + cute.math.exp(v) + s
    """
    with pytest.raises(exc.BackendUnsupported, match="acc carries one V lane"):
        _hoist(src)


def test_a_reduce_in_a_branch_of_the_lane_body_is_refused() -> None:
    """A branch around a reduce of the lane's element writes values that mix
    the two; the check does not look into it and refuses the loop."""
    src = """
    for vec_lane_1 in cutlass.range_constexpr(4):
        acc_copy = acc
        v = cutlass.Float32(_v0[vec_lane_1])
        if beta:
            acc = acc_copy + cute.math.exp(v)
        else:
            s = cutlass.Float32(cute.arch.warp_reduction_sum(v, threads_in_group=4))
            acc = acc_copy + s
    """
    with pytest.raises(exc.BackendUnsupported, match="other than an assignment reads"):
        _hoist(src)


def test_partials_folded_by_later_reduces_and_lane_uniform_merges_stay() -> None:
    """Welford over a chunk per V lane: the centered squares fold into the
    second reduce and the merge reads values the same in every lane, so the
    loop stays as it is (the count's Int64 reduce input keeps the hoist out,
    as in the emitted Welford of test_cute_sibling_layout_safety.py)."""
    src = textwrap.dedent(
        """
        for vec_lane_1 in cutlass.range_constexpr(4):
            chunk = cutlass.Float32(_v0[vec_lane_1])
            valid = cutlass.Int32(lane_base + vec_lane_1 < n)
            n_c = cutlass.Int64(cute.arch.warp_reduction_sum(cutlass.Int64(valid), threads_in_group=4))
            total = cutlass.Float32(cute.arch.warp_reduction_sum(chunk, threads_in_group=4))
            mean_c = total / cutlass.Float32(n_c)
            centered = chunk - mean_c
            square = centered * centered
            m2_c = cutlass.Float32(cute.arch.warp_reduction_sum(square, threads_in_group=4))
            delta = mean_c - mean
            count = cnt + cutlass.Float32(n_c)
            mean = mean + delta * cutlass.Float32(n_c) / count
            m2 = m2 + m2_c + delta * delta * cnt * cutlass.Float32(n_c) / count
            cnt = count
        """
    )
    assert _hoist(src) == ast.unparse(ast.parse(src))


@helion.kernel(backend="cute", static_shapes=True)
def _center_rows(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    bn = hl.register_block_size(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        for tile_n in hl.tile(n, block_size=bn):
            v = x[tile_m, tile_n]
            out[tile_m, tile_n] = v - v.sum(-1, keepdim=True)
    return out


@pytest.mark.parametrize("vector", [1, 4])
def test_a_tile_sum_read_per_element_is_not_split_by_vector_lanes(vector: int) -> None:
    """Four threads hold a 16-column tile: as a lane loop (one element per
    lane) the sum is complete before its elements read it; as four vector
    lanes the reduce would stay in the V-loop and sum 4 of the 16."""
    x = torch.empty(64, 64, dtype=torch.float32)
    config = helion.Config(
        block_sizes=[16, 1], num_threads=[4, 1], cute_vector_widths=[vector, 1]
    )
    with mock_cuda_unavailable(), cpu_render_target():
        bound = cpu_bind(_center_rows, (x,))
        config = bound.config_spec.normalized_config(config)
        if vector == 1:
            assert "warp_reduction_sum(sum_1_lane_acc" in bound.to_code(config)
            return
        with pytest.raises(exc.BackendUnsupported, match="vector lane loop"):
            bound.to_code(config)
