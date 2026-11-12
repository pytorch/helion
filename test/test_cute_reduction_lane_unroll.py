"""CPU codegen tests: a looped reduction's lane loops, the barrier pass and the unroll.

A looped reduction walks a thread's share of each chunk one vector per lane:
a reduction lane loop, the lane's base index, then a constexpr V-loop over
the vector's elements (``reduction_emitter.emit_lane_loop_body``).

- The barrier pass (``cute/lane_loops/barriers.py``, run from
  ``nest_check.check_lane_loop_nest``) takes those loops as iterations of the
  device loop and reads the base index from its definition.  A per-element
  store of the consume sweep is then one thread's element; when the base index
  was a name of unknown origin the store looked like one element every thread
  writes, and a barrier closed the body of every element.
- ``cute_lane_unroll`` applies to these lanes: normalization keeps the knob
  when a chunk spans more than one vector per thread, and the unroll
  (``cute/unroll_lane_loads.py``) issues every lane's loads, the packets above
  the V-loop and the element loads in it, before the first lane computes and
  stores (``HELION_REDUCTION_VLOOP_ATTR``).
"""

from __future__ import annotations

import ast
import functools
from typing import TYPE_CHECKING
from typing import cast

import pytest
import torch

import helion
from helion._testing import cpu_render_target
from helion._testing.cute_chunked_rows import row_shift
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import is_barrier
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_codegen import roll_loops
from helion._testing.cute_sweep_kernels import workspace_chunk_config
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.autotuner.config_spec import ConfigSpec
    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only

# An odd row length: ``y`` and ``out`` rows start at 2-byte offsets, so their
# accesses stay per element while the padded ``x`` rows load packets.
_ROWS, _N = 8, 4099


@functools.cache
def _bound(
    fn: Callable[..., torch.Tensor] = row_shift, y_dtype: torch.dtype = torch.bfloat16
) -> BoundKernel:
    """``fn`` over the padded bf16 ``x`` and a ``y_dtype`` ``y`` (fp32: its
    per-element loads take the scalar cache hints)."""
    kernel = helion.kernel(
        fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    args = (
        torch.empty(_ROWS, _N + 5, dtype=torch.bfloat16)[:, :_N],
        torch.empty(_ROWS, _N, dtype=y_dtype),
    )
    with mock_cuda_unavailable():
        return cpu_bind(kernel, args)


def _config(
    chunk: int, threads: int, vec: int, unroll: int = 1, **extra: object
) -> helion.Config:
    spec = _bound().config_spec
    (rdim,) = spec.reduction_loops.valid_block_ids()
    return helion.Config(
        block_sizes=[1],
        reduction_loops=[chunk],
        num_threads=[
            threads if block_id == rdim else 1
            for block_id in spec.num_threads.valid_block_ids()
        ],
        cute_vector_widths=[
            vec if block_id == rdim else 1
            for block_id in spec.cute_vector_widths.valid_block_ids()
        ],
        cute_lane_unroll=unroll,
        **extra,
    )


@functools.cache
def _code(chunk: int, threads: int, vec: int, unroll: int = 1) -> str:
    with mock_cuda_unavailable():
        return _bound().to_code(_config(chunk, threads, vec, unroll))


def _gathered_add(
    x: torch.Tensor, idx: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    rows, _n = x.shape
    for tile_r in hl.tile(rows):
        total = torch.sum(x[tile_r, :].to(torch.float32), dim=-1, keepdim=True)
        columns = idx[tile_r, :]
        old = hl.load(out, [tile_r.index[:, None], columns])
        hl.store(
            out,
            [tile_r.index[:, None], columns],
            (old.to(torch.float32) + total).to(out.dtype),
        )
    return out


def _gathered_into_rows(
    x: torch.Tensor, idx: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    rows, _n = x.shape
    for tile_r in hl.tile(rows):
        total = torch.sum(x[tile_r, :].to(torch.float32), dim=-1, keepdim=True)
        columns = idx[tile_r, :]
        old = hl.load(out, [tile_r.index[:, None], columns])
        out[tile_r, :] = (old.to(torch.float32) + total).to(out.dtype)
    return out


def _scattered_back(
    x: torch.Tensor, idx: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    rows, _n = x.shape
    for tile_r in hl.tile(rows):
        total = torch.sum(x[tile_r, :].to(torch.float32), dim=-1, keepdim=True)
        old = out[tile_r, :]
        out[tile_r, :] = (old.to(torch.float32) + total).to(out.dtype)
        hl.store(out, [tile_r.index[:, None], idx[tile_r, :]], old)
    return out


def _added_in_place(
    x: torch.Tensor, idx: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    rows, _n = x.shape
    for tile_r in hl.tile(rows):
        total = torch.sum(x[tile_r, :].to(torch.float32), dim=-1, keepdim=True)
        out[tile_r, :] = (out[tile_r, :].to(torch.float32) + total).to(out.dtype)
    return out


# A read-modify-write through gathered columns, a gathered load beside a store
# of the whole row, and a load of the whole row beside a scatter of it.
_GATHERED = {
    "read_modify_write": _gathered_add,
    "row_store": _gathered_into_rows,
    "scatter": _scattered_back,
}


@functools.cache
def _gathered_code(
    kernel_name: str, chunk: int, threads: int, vec: int, aligned: bool = False
) -> str:
    """``_GATHERED[kernel_name]`` (``_added_in_place`` for ``"in_place"``)
    rendered at ``_config``: it has ``_bound``'s blocks.  ``aligned`` rows
    of 4096 elements move ``out`` in packets, loaded before a lane's V-loop
    and flushed after it; the odd rows of ``_N`` keep its accesses per
    element."""
    kernel = helion.kernel(
        {**_GATHERED, "in_place": _added_in_place}[kernel_name],
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    n = 4096 if aligned else _N
    args = (
        torch.empty(_ROWS, n + 5, dtype=torch.bfloat16)[:, :n],
        torch.empty(_ROWS, n, dtype=torch.int32),
        torch.empty(_ROWS, n, dtype=torch.bfloat16),
    )
    with mock_cuda_unavailable():
        return cpu_bind(kernel, args).to_code(_config(chunk, threads, vec))


# The scatter is refused before the lanes are looked at: its store through
# ``idx`` races between the threads sharing the tile axis, inside the sweep
# the reduction pass rewrites, where no barrier can order them.
_REFUSALS = {"scatter": "threads sharing a tile axis race on a tensor"}


# Nine cases: each kernel pairs a gather with another access of the row (a
# per-element read-modify-write, a row store, a scatter of the row it read),
# at scalar and two-wide lanes over odd rows, whose accesses stay per
# element, and at two-wide lanes over aligned rows, whose packets of ``out``
# the lanes load and flush around their V-loops.
@pytest.mark.parametrize(
    ("vec", "aligned"), [(1, False), (2, False), (2, True)], ids=["1", "2", "2-aligned"]
)
@pytest.mark.parametrize("kernel_name", list(_GATHERED))
def test_gathered_accesses_across_a_threads_lanes_are_rejected(
    kernel_name: str, vec: int, aligned: bool
) -> None:
    """1024-element chunks over 64 threads: 16 elements per thread, in 16
    lanes or in 8 lanes of two-wide vectors.  The columns ``idx`` names may
    be an element another lane of the same thread stores, or loads
    (``idx`` swapping columns 0 and 128, both thread 0's): the lane storing
    it first runs before the lane loading it, which reads the new value
    where the program loads every column before storing any, and no
    barrier orders a thread's own lanes.  A packet loaded before the V-loop
    or flushed after it is such an access of every element of the vector."""
    with pytest.raises(
        helion.exc.BackendUnsupported, match=_REFUSALS.get(kernel_name, "two lanes")
    ):
        _gathered_code(kernel_name, 1024, 64, vec, aligned)


@pytest.mark.parametrize("kernel_name", list(_GATHERED))
def test_gathered_accesses_across_chunks_are_rejected(kernel_name: str) -> None:
    """64-element chunks over 64 threads: one lane per thread, but 65
    chunks a row.  The columns ``idx`` names may be an element an earlier
    chunk stored: the sweep runs each chunk's store before the next chunk's
    load, where the program loads every column before storing any, and a
    barrier orders threads, not chunks."""
    with pytest.raises(
        helion.exc.BackendUnsupported, match=_REFUSALS.get(kernel_name, "two chunks")
    ):
        _gathered_code(kernel_name, 64, 64, 1)


def _shifted_load(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    for tile_r in hl.tile(rows):
        total = torch.sum(x[tile_r, :].to(torch.float32), dim=-1, keepdim=True)
        old = out[tile_r, :n]
        out[tile_r, 64:] = (old.to(torch.float32) + total).to(out.dtype)
    return out


def _shifted_store(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    for tile_r in hl.tile(rows):
        total = torch.sum(x[tile_r, :].to(torch.float32), dim=-1, keepdim=True)
        out[tile_r, :n] = (x[tile_r, :].to(torch.float32) + total).to(out.dtype)
        out[tile_r, 64:] = (x[tile_r, :].to(torch.float32) - total).to(out.dtype)
    return out


@pytest.mark.parametrize("fn", [_shifted_load, _shifted_store], ids=["load", "store"])
def test_overlapping_slices_across_chunks_are_rejected(fn) -> None:
    """64-element chunks over 64 threads, one lane per thread, over rows of
    4096 elements and ``out`` rows 64 longer: chunk i stores ``out[64 +
    rindex_1]``, which chunk i + 1 then loads (or stores again) at
    ``out[rindex_1]``, where the program loads (or stores) the first slice
    whole before storing the second.  No barrier orders chunks, whatever
    the addresses."""
    kernel = helion.kernel(
        fn, backend="cute", static_shapes=True, autotune_effort="none"
    )
    args = (
        torch.empty(_ROWS, 4096, dtype=torch.bfloat16),
        torch.empty(_ROWS, 4160, dtype=torch.bfloat16),
    )
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, args)
        with pytest.raises(helion.exc.BackendUnsupported, match="two chunks"):
            bound.to_code(_config(64, 64, 1))


def _shifted_back(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    rows, n = x.shape
    for tile_r in hl.tile(rows):
        total = torch.sum(x[tile_r, :].to(torch.float32), dim=-1, keepdim=True)
        old = out[tile_r, 64:]
        out[tile_r, :n] = (old.to(torch.float32) + total).to(out.dtype)
    return out


def test_a_slice_loaded_before_later_chunks_store_it_needs_nothing() -> None:
    """The other way round: chunk i loads ``out[64 + rindex_1]`` and
    stores ``out[rindex_1]``, so each element is loaded one chunk before it
    is stored, as the program orders the two: neither a rejection nor a
    barrier."""
    kernel = helion.kernel(
        _shifted_back, backend="cute", static_shapes=True, autotune_effort="none"
    )
    args = (
        torch.empty(_ROWS, 4096, dtype=torch.bfloat16),
        torch.empty(_ROWS, 4160, dtype=torch.bfloat16),
    )
    with mock_cuda_unavailable():
        code = cpu_bind(kernel, args).to_code(_config(64, 64, 1))
    assert len(roll_loops(code)) == 2, code
    assert "cute.arch.sync_threads()" not in code


def _gather_rows_into_other_tile(
    src: torch.Tensor, idx: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    for tile_m, tile_n in hl.tile([idx.size(0), out.size(0)]):
        out[tile_n, :] = src[idx[tile_m], :]
    return out


def test_a_buffer_each_chunk_allocates_needs_no_chunk_order() -> None:
    """The rows gathered by ``tile_m`` and stored by ``tile_n`` move through
    shared memory (``rebind_smem``), which each chunk of the row's rolled
    columns (8 in chunks of 4) allocates, fills and reads back: no chunk
    sees another's values, and the barriers order the threads' uses of it,
    the one closing the body included."""
    kernel = helion.kernel(
        _gather_rows_into_other_tile, backend="cute", static_shapes=True
    )
    args = (torch.empty(64, 8), torch.zeros(16, dtype=torch.int64), torch.empty(16, 8))
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, args)
        config = bound.config_spec.normalized_config(
            helion.Config(block_sizes=[16, 16])
        )
        assert config.reduction_loops == [4]
        code = bound.to_code(config)
    (chunks,) = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For) and ast.unparse(node.target).startswith("roffset")
    ]
    body = ast.unparse(chunks)
    assert "rebind_smem = cute.make_tensor(rebind_smem_ptr" in body, body
    assert body.count("cute.arch.sync_threads()") == 2, body


@pytest.mark.cuda
@helion._testing.skipIfNotCUDA()
def test_shifted_slices_match_eager() -> None:
    """On a GPU, over rows of 128 elements: the slice loaded a chunk before
    it is stored matches eager at 64-element chunks over 64 threads.  The
    slices stored a chunk before they are loaded or stored again are
    rejected there (above), as a chunk would read or overwrite what an
    earlier one stored; over whole rows, one element per thread, a barrier
    orders the threads, and they match eager."""
    torch.manual_seed(0)
    x = torch.randn(_ROWS, 128, device=helion._testing.DEVICE)
    out = torch.randn(_ROWS, 192, device=helion._testing.DEVICE)
    total = x.sum(-1, keepdim=True)
    shifted_back = out.clone()
    shifted_back[:, :128] = out[:, 64:] + total
    shifted_load = out.clone()
    shifted_load[:, 64:] = out[:, :128] + total
    shifted_store = out.clone()
    shifted_store[:, :128] = x + total
    shifted_store[:, 64:] = x - total
    rows = {**_config(64, 128, 1).config, "reduction_loops": [None]}
    for fn, expected, config in (
        (_shifted_back, shifted_back, _config(64, 64, 1).config),
        (_shifted_load, shifted_load, rows),
        (_shifted_store, shifted_store, rows),
    ):
        kernel = helion.kernel(
            fn, backend="cute", static_shapes=True, autotune_effort="none"
        )
        _code, result = helion._testing.code_and_output(
            kernel, (x, out.clone()), **config
        )
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-3)


def test_a_lanes_own_packets_need_no_barrier() -> None:
    """An in-place add of the row's sum over aligned rows: each lane loads
    its packet of ``out`` before the V-loop and flushes it after, its own
    elements in every chunk, which needs neither a barrier nor a rejection."""
    code = _gathered_code("in_place", 1024, 64, 2, aligned=True)
    assert "= cute.arch.load(out.iterator" in code
    assert "_cute_store_u16_vec(out.iterator" in code
    assert "cute.arch.sync_threads()" not in code


def _element_stores(loop: ast.For, tensor: str) -> list[ast.Call]:
    """The per-element stores of ``tensor`` in ``loop``: ``(tensor.iterator +
    ...).store(...)``, not a packet's ``cute.arch.store``."""
    return [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "store"
        and f"{tensor}.iterator" in ast.unparse(node.func.value)
    ]


@pytest.mark.parametrize("vec", [1, 2])
def test_consume_sweep_stores_need_no_barrier(vec: int) -> None:
    # 1024-element chunks over 64 threads: 16 / vec lanes per thread, each
    # storing ``out`` per element in the innermost loop (the V-loop's body
    # at two-wide vectors).
    code = _code(1024, 64, vec)
    _reduce, consume = roll_loops(code)
    innermost = "reduction_vec_lane" if vec > 1 else "reduction_lane"
    (elements,) = [
        node
        for node in ast.walk(consume)
        if isinstance(node, ast.For) and ast.unparse(node.target).startswith(innermost)
    ]
    assert _element_stores(elements, "out"), ast.unparse(consume)
    assert "cute.arch.sync_threads()" not in code


@pytest.mark.parametrize(
    ("chunk", "threads", "vec", "kept"),
    [
        (1024, 64, 2, True),
        (1024, 64, 1, True),
        # One vector per thread per chunk: no lane loop, the knob is inert.
        (128, 64, 2, False),
    ],
)
def test_lane_unroll_survives_normalization_with_reduction_lanes(
    chunk: int, threads: int, vec: int, kept: bool
) -> None:
    normalized = _bound().config_spec.normalized_config(
        _config(chunk, threads, vec, unroll=8)
    )
    assert normalized.config["cute_lane_unroll"] == (8 if kept else 1)


def _row_argmax(x: torch.Tensor) -> torch.Tensor:
    rows, _n = x.shape
    out = torch.empty([rows], dtype=torch.int64, device=x.device)
    for tile_r in hl.tile(rows):
        out[tile_r] = torch.argmax(x[tile_r, :], dim=-1)
    return out


@functools.cache
def _bound_argmax() -> BoundKernel:
    kernel = helion.kernel(
        _row_argmax, backend="cute", static_shapes=True, autotune_effort="none"
    )
    with mock_cuda_unavailable():
        return cpu_bind(kernel, (torch.empty(_ROWS, 4096, dtype=torch.float32),))


# Configs whose lane loops are settled by the steps of normalization after
# the knobs: the reduction-loop cap, the rolling of a persistent reduction
# and the reduction's thread fit.  (kernel, config, unroll kept)
_SETTLED_LATE = {
    # argmax reduces within a warp: the chunk caps to 32 elements on 32
    # threads, one lane each.
    "capped": (
        _bound_argmax,
        {"block_sizes": [1], "reduction_loops": [2048], "num_threads": [1, 32]},
        1,
    ),
    # The 4099-element row rolls into 1024-element chunks: 16 lanes on 64
    # threads.
    "rolled": (_bound, {"block_sizes": [1], "num_threads": [1, 64]}, 8),
    # Four rows of 1024 threads exceed the CTA: the reduction fits on 256
    # threads, 4 lanes each.
    "fitted": (
        _bound,
        {"block_sizes": [4], "num_threads": [4, 0], "reduction_loops": [1024]},
        8,
    ),
}


@pytest.mark.parametrize("case", sorted(_SETTLED_LATE))
def test_lane_unroll_follows_the_final_reduction_threads(case: str) -> None:
    bound, values, kept = _SETTLED_LATE[case]
    spec = bound().config_spec
    normalized = spec.normalized_config(helion.Config(**values, cute_lane_unroll=8))
    assert normalized.config["cute_lane_unroll"] == kept
    # A normalized config normalizes to itself.
    assert spec.normalized_config(normalized) == normalized
    with mock_cuda_unavailable():
        code = bound().to_code(normalized)
    assert ("_lane_unroll_loads_" in code) is (kept > 1)


def test_lane_unroll_issues_the_consume_sweeps_loads_first() -> None:
    code = _code(1024, 64, 2, unroll=8)
    reduce, consume = roll_loops(code)
    for sweep in (reduce, consume):
        phases = [node for node in sweep.body if isinstance(node, ast.For)]
        assert len(phases) == 2, ast.unparse(sweep)
        load_phase, compute_phase = map(ast.unparse, phases)
        assert "cutlass.range_constexpr(8)" in load_phase
        assert ".store(" not in load_phase
        assert "_lane_unroll_loads_" in compute_phase
    load_phase, compute_phase = (
        ast.unparse(node) for node in consume.body if isinstance(node, ast.For)
    )
    # The per-element loads of ``y`` in the V-loop move to the load phase
    # with the V-loop; the stores stay in the compute phase.
    assert "y.iterator" in load_phase
    assert "y.iterator" not in compute_phase
    assert "for reduction_vec_lane_2 in cutlass.range_constexpr(2):" in load_phase
    assert ".store(" in compute_phase
    # The reduce sweep's packets of ``x`` load first too; its sum still
    # accumulates lane by lane, element by element, in the compute phase.
    reduce_load, reduce_compute = (
        ast.unparse(node) for node in reduce.body if isinstance(node, ast.For)
    )
    assert "x.iterator" in reduce_load
    assert "total_acc = total_acc +" in reduce_compute
    assert "total_acc" not in reduce_load


def test_lane_unroll_off_keeps_the_rolled_lane_loops() -> None:
    code = _code(1024, 64, 2)
    assert "_lane_unroll_loads_" not in code
    assert code.count("for reduction_lane_1 in range(8):") == 2


@pytest.mark.parametrize("policy", ["first", "last", "streaming"])
def test_lane_unroll_issues_hinted_scalar_loads_first(policy: str) -> None:
    """A hinted 4-byte scalar load lowers to exact-policy PTX
    (``lower_scalar_policy_loads``), so the unroll takes the lane body that
    holds it: the consume sweep's ``y`` loads, hint kept,
    move to the load phase."""
    config = _config(1024, 64, 2, unroll=8, load_eviction_policies=["", policy])
    with mock_cuda_unavailable():
        code = _bound(y_dtype=torch.float32).to_code(config)
    _reduce, consume = roll_loops(code)
    phases = [node for node in consume.body if isinstance(node, ast.For)]
    assert len(phases) == 2, ast.unparse(consume)
    load_phase, compute_phase = map(ast.unparse, phases)
    assert "_cute_scalar_policy_load(y.iterator" in load_phase
    assert f"'{policy}')" in load_phase
    assert "y.iterator" not in compute_phase
    assert ".store(" in compute_phase


def _row_shift_evict_normal(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``row_shift`` reading ``y`` with an L1 hint that the scalar lowering
    keeps as an extended load (``evict_normal``)."""
    rows, _n = y.shape
    out = torch.empty_like(y)
    for tile_r in hl.tile(rows):
        total = torch.sum(x[tile_r, :].to(torch.float32), dim=-1, keepdim=True)
        y_row = hl.load(y, [tile_r, slice(None)], eviction_policy="evict_normal")
        out[tile_r, :] = (y_row.to(torch.float32) - total).to(out.dtype)
    return out


def test_lane_unroll_declines_a_hinted_scalar_load_left_unlowered() -> None:
    """A hinted scalar load ``lower_scalar_policy_loads`` leaves to NVVM as
    an extended load still declines the unroll: the consume sweep keeps its
    rolled lane loop, while the reduce sweep, whose loads carry no hint,
    unrolls."""
    with mock_cuda_unavailable():
        code = _bound(_row_shift_evict_normal, torch.float32).to_code(
            _config(1024, 64, 2, unroll=8)
        )
    reduce, consume = roll_loops(code)
    assert "_lane_unroll_loads_" in ast.unparse(reduce)
    (lane_loop,) = [node for node in consume.body if isinstance(node, ast.For)]
    assert "level1_eviction_priority='evict_normal'" in ast.unparse(lane_loop)
    assert "_lane_unroll_loads_" not in ast.unparse(consume)


def _weighted_square_sums(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """``acc += sum(x * w) ** 2``: the cross-lane fold unrolls the tile
    reduction's vectorized lanes completely, whatever ``cute_lane_unroll``
    says."""
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n):
            s = torch.sum(
                x[tile_m, tile_n].to(torch.float32) * w[tile_m, tile_n], dim=-1
            )
            acc = acc + s * s
        out[tile_m] = acc
    return out


@pytest.mark.parametrize("policy", ["last", "streaming"])
def test_forced_fold_unroll_keeps_hinted_scalar_loads_in_place(policy: str) -> None:
    """Without the knob a hinted scalar load stays where it is: the fold's
    forced unroll copies its lane body in place (``_force_constexpr``)
    rather than issuing the loads first.  ``w``'s odd row stride keeps its
    loads per element, so they take the scalar hint."""
    kernel = helion.kernel(_weighted_square_sums, backend="cute", static_shapes=True)
    args = (
        torch.empty(_ROWS, 16384, dtype=torch.bfloat16),
        torch.empty(_ROWS, 16385, dtype=torch.float32)[:, :16384],
    )
    config = helion.Config(
        block_sizes=[1, 4096],
        num_threads=[1, 128],
        cute_vector_widths=[1, 16],
        load_eviction_policies=["", policy],
    )
    # The tile reduction sizes its shared-memory exchange from the device's
    # budget (a B200's).
    with cpu_render_target(reduction_smem_budget_bytes=232448):
        code = cpu_bind(kernel, args).to_code(config)
    assert "_cute_scalar_policy_load(w.iterator" in code
    assert f"'{policy}')" in code
    assert "cutlass.range_constexpr(2)" in code
    assert "_lane_unroll_loads_" not in code


def _row_sum(x: torch.Tensor) -> torch.Tensor:
    rows, _n = x.shape
    out = torch.empty([rows], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(rows):
        out[tile_r] = torch.sum(x[tile_r, :].to(torch.float32), dim=-1)
    return out


def _loads_first_seeds(spec: ConfigSpec) -> list[dict[str, object]]:
    return [
        seed.config
        for seed in spec.compiler_seed_configs
        if seed.config.get("cute_lane_unroll") == 16
    ]


def test_row_store_kernels_seed_loads_first_lanes() -> None:
    """``CuteRolledRowLoadsFirstHeuristic``: a row kernel with a sweep that
    stores the row gets the 64-thread, 2048-element seed with the 16-lane
    loads-first unroll, its lanes unvectorized on rows at odd offsets; a row
    kernel storing one value per row does not."""
    spec = _bound().config_spec
    assert "cute_rolled_row_loads_first" in spec.autotuner_heuristics
    (rdim,) = spec.reduction_loops.valid_block_ids()
    (seed,) = _loads_first_seeds(spec)
    assert seed["reduction_loops"] == [2048]
    assert spec.num_threads.config_get(seed["num_threads"], rdim) == 64
    assert "cute_vector_widths" not in seed
    # The seed's unroll survives normalization: 32 unvectorized lanes.
    assert (
        spec.normalized_config(helion.Config(**seed)).config["cute_lane_unroll"] == 16
    )
    kernel = helion.kernel(
        _row_sum, backend="cute", static_shapes=True, autotune_effort="none"
    )
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, (torch.empty(_ROWS, _N, dtype=torch.bfloat16),))
    assert "cute_rolled_row_loads_first" not in bound.config_spec.autotuner_heuristics


def _row_shift_prefix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``row_shift`` summing the first ``n`` elements of wider ``x`` rows."""
    rows, n = y.shape
    out = torch.empty_like(y)
    for tile_r in hl.tile(rows):
        total = torch.sum(x[tile_r, :n].to(torch.float32), dim=-1, keepdim=True)
        out[tile_r, :] = (y[tile_r, :].to(torch.float32) - total).to(out.dtype)
    return out


def _row_shift_from_one(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``row_shift`` summing ``x`` rows from their second element."""
    rows, n = y.shape
    out = torch.empty_like(y)
    for tile_r in hl.tile(rows):
        total = torch.sum(x[tile_r, 1 : n + 1].to(torch.float32), dim=-1, keepdim=True)
        out[tile_r, :] = (y[tile_r, :].to(torch.float32) - total).to(out.dtype)
    return out


def _row_double(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_r in hl.tile(x.size(0)):
        out[tile_r, :] = x[tile_r, :] * 2
    return out


def _seed_lane_vector(
    fn: Callable[..., torch.Tensor], *args: torch.Tensor, static_shapes: bool = True
) -> tuple[ConfigSpec, dict[str, object], int]:
    """``fn``'s config spec, its loads-first seed (the first, where another
    comes under ``cute_proven_bounds``) and the seed's lane vector."""
    kernel = helion.kernel(
        fn, backend="cute", static_shapes=static_shapes, autotune_effort="none"
    )
    with mock_cuda_unavailable():
        spec = cpu_bind(kernel, args).config_spec
    (rdim,) = spec.reduction_loops.valid_block_ids()
    seed = _loads_first_seeds(spec)[0]
    widths = cast("list[int]", seed.get("cute_vector_widths", []))
    return spec, seed, spec.cute_vector_widths.config_get(widths, rdim, 1)


@pytest.mark.parametrize(
    ("fn", "width", "vec"),
    [
        # The prefix of an odd-length row: every row starts at a 2-byte
        # offset.
        (_row_shift_prefix, 4099, 1),
        (_row_shift_prefix, 4100, 2),
        # Even rows read from their second element.
        (_row_shift_from_one, 4100, 1),
    ],
)
def test_loads_first_seed_reads_rows_through_slices(
    fn: Callable[..., torch.Tensor], width: int, vec: int
) -> None:
    """A row the sweeps read through a slice counts with the slice's start:
    the lanes are two-wide only where both the start and the row stride
    keep them aligned."""
    x = torch.empty(_ROWS, width, dtype=torch.bfloat16)
    y = torch.empty(_ROWS, 4098, dtype=torch.bfloat16)
    assert _seed_lane_vector(fn, x, y)[2] == vec


def test_overlapping_rows_take_unvectorized_lanes() -> None:
    """Rows one element apart (a unit stride beside the reduced dim's) start
    at every offset: only the reduced dim's unit stride is exempt."""
    x = torch.empty(4105, dtype=torch.bfloat16).as_strided((_ROWS, 4098), (1, 1))
    y = torch.empty(_ROWS, 4098, dtype=torch.bfloat16)
    assert _seed_lane_vector(row_shift, x, y)[2] == 1


@pytest.mark.parametrize("static_shapes", [True, False])
def test_loads_first_seed_ignores_the_storage_offset(static_shapes: bool) -> None:
    """The seed reads no base pointer, which the bind keys only after the
    seeds are made: rows at storage offsets 0 and 1 get the same seed, also
    when the offset is dynamic.  A seed that read it would differ between
    the two, and with it the autotune cache key."""
    y = torch.empty(_ROWS, 4098, dtype=torch.bfloat16)
    seeds = [
        _seed_lane_vector(
            row_shift,
            torch.empty(_ROWS, 4106, dtype=torch.bfloat16)[:, offset : offset + 4098],
            y,
            static_shapes=static_shapes,
        )[1]
        for offset in (0, 1)
    ]
    assert seeds[0] == seeds[1]


def test_a_short_row_keeps_the_lane_vector_but_not_the_unroll() -> None:
    """A row shorter than the 2048-element chunk normalizes the seed to a
    persistent row of 64 threads: a 64-element row keeps its two-wide lanes
    and loses the unroll, which has no second vector per thread to issue."""
    spec, seed, vec = _seed_lane_vector(
        row_shift,
        torch.empty(_ROWS, 64, dtype=torch.bfloat16),
        torch.empty(_ROWS, 64, dtype=torch.bfloat16),
    )
    normalized = spec.normalized_config(helion.Config(**seed)).config
    assert vec == 2
    assert normalized["reduction_loops"] == [None]
    assert normalized["cute_lane_unroll"] == 1
    assert normalized["cute_vector_widths"] == seed["cute_vector_widths"]


def test_pointwise_row_kernels_get_the_loads_first_seed() -> None:
    """The gate needs a row-store sweep, not a reduction: a row stored as a
    function of itself walks the row as a cross-entropy backward whose
    one-hot sum is folded away does."""
    spec, seed, vec = _seed_lane_vector(
        _row_double, torch.empty(_ROWS, 4098, dtype=torch.bfloat16)
    )
    assert seed["reduction_loops"] == [2048]
    assert vec == 2


def test_loads_first_seed_guards_no_dynamic_shape() -> None:
    """The lane vector reads a dynamic row's strides without comparing them:
    the seed adds no shape guard, which would become a dispatch fact that no
    generated code needs."""
    kernel = helion.kernel(
        row_shift,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        disable_autotuner_heuristics=True,
    )
    args = (
        torch.empty(_ROWS, 4104, dtype=torch.bfloat16)[:, :4098],
        torch.empty(_ROWS, 4098, dtype=torch.bfloat16),
    )
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, args)
    (heuristic,) = (
        heuristic
        for heuristic in bound.env.backend.autotuner_heuristics()
        if heuristic.name == "cute_rolled_row_loads_first"
    )
    host = bound.host_function
    assert host is not None
    guards = len(bound.env.shape_env.guards)
    with bound.env, host:
        assert heuristic.is_eligible(bound.env, host.device_ir)
        assert heuristic.get_seed_config(bound.env, host.device_ir) is not None
    assert len(bound.env.shape_env.guards) == guards


def test_loads_first_seed_vectorizes_rows_aligned_for_it() -> None:
    """Rows every tensor of the sweeps keeps 4-byte aligned (an even row
    length and stride) take two-wide lanes, the widest the seed uses."""
    kernel = helion.kernel(
        row_shift, backend="cute", static_shapes=True, autotune_effort="none"
    )
    n = _N - 1
    args = (
        torch.empty(_ROWS, n + 8, dtype=torch.bfloat16)[:, :n],
        torch.empty(_ROWS, n, dtype=torch.bfloat16),
    )
    with mock_cuda_unavailable():
        spec = cpu_bind(kernel, args).config_spec
    (rdim,) = spec.reduction_loops.valid_block_ids()
    (seed,) = _loads_first_seeds(spec)
    assert spec.cute_vector_widths.config_get(seed["cute_vector_widths"], rdim) == 2
    # 16 two-wide lanes per thread: the unroll survives normalization.
    assert (
        spec.normalized_config(helion.Config(**seed)).config["cute_lane_unroll"] == 16
    )


def _workspace_rescale(x: torch.Tensor) -> torch.Tensor:
    # The second sweep also writes the workspace it reads back.
    rows, n = x.shape
    ws = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(rows):
        v = x[tile_r, :].to(torch.float32)
        ws[tile_r, :] = v * v
        total = torch.sum(v, dim=-1, keepdim=True)
        ws[tile_r, :] = ws[tile_r, :] - total
    return ws


def _workspace_code(fn: object) -> str:
    kernel = helion.kernel(fn, backend="cute", static_shapes=True)
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, (torch.empty(8, 4096, dtype=torch.bfloat16),))
        return bound.to_code(workspace_chunk_config(bound.config_spec))


def _lane_loop_barriers(sweep: ast.For) -> list[ast.For]:
    """The loop directly holding each barrier inside the sweep's lane loops."""
    return [
        loop
        for loop in ast.walk(sweep)
        if isinstance(loop, ast.For) and loop is not sweep
        for stmt in loop.body
        if is_barrier(stmt)
    ]


def test_barrier_stays_where_the_lanes_write_what_it_orders() -> None:
    code = _workspace_code(_workspace_rescale)
    _reduce, consume = roll_loops(code)
    # The lanes store into the workspace: the barrier stays in each lane.
    (holder,) = _lane_loop_barriers(consume)
    assert ast.unparse(holder.target).startswith("reduction_lane_")
