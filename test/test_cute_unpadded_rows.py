"""``cute_unpadded_rows``: a masked persistent row run unpadded across warps.

A persistent row whose length is not a power of two pads to the next one and
is masked, which keeps it on one warp walking scalar lanes.  The knob runs it
unpadded when a whole number of warps, one vector per thread, covers it
exactly: 768 columns as 96 threads of 8 elements become the multi-warp vector
row, whose packets load before the arithmetic and whose sums combine across
warps through shared memory.  The rows render on CPU; the ``cuda`` test runs
the unpadded row against the padded one: masks bit for bit, floats within
the rounding the other summation order and contraction give.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.autotune import search_surface
from helion._compiler.cute.autotune.heuristics import (
    CutePersistentUnpaddedRowsHeuristic,
)
from helion._compiler.cute.reduction_emitter import CutePersistentReductionEmitter
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion._testing.cute_dropout_rows import dropout_residual_norm
from helion._testing.cute_dropout_rows import dropout_residual_norm_args
import helion.language as hl

if TYPE_CHECKING:
    from helion.autotuner.block_id_sequence import BlockIdSequence
    from helion.runtime.kernel import BoundKernel

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


_dropout_residual_norm = helion.kernel(
    dropout_residual_norm,
    backend="cute",
    static_shapes=True,
    autotune_effort="none",
    cute_rng_stream="word0",
)


def _layout(
    bound: BoundKernel,
    *,
    rows: int,
    threads: int,
    vec: int,
    unpadded: bool,
) -> helion.Config:
    """``rows`` rows per CTA of ``threads`` threads with ``vec`` elements each
    (each sequence spelled in its own slot order)."""
    spec = bound.config_spec
    (reduction,) = spec.reduction_block_ids

    def per_block(
        sequence: BlockIdSequence[Any], values: dict[int, object], default: object
    ) -> list[object]:
        return [values.get(item.block_ids[0], default) for item in sequence]

    (row,) = (b for b in spec.num_threads.valid_block_ids() if b != reduction)
    return spec.normalized_config(
        helion.Config(
            block_sizes=[rows],
            num_threads=per_block(spec.num_threads, {row: rows, reduction: threads}, 0),
            cute_vector_widths=per_block(spec.cute_vector_widths, {reduction: vec}, 1),
            cute_reduction_reloads=["register"],
            cute_unpadded_rows=unpadded,
        )
    )


@functools.cache
def _bind() -> BoundKernel:
    with mock_cuda_unavailable(), forbid_native_compile():
        return cpu_bind(_dropout_residual_norm, dropout_residual_norm_args(768))


def _render(*, rows: int, threads: int, vec: int, unpadded: bool) -> str:
    bound = _bind()
    config = _layout(bound, rows=rows, threads=threads, vec=vec, unpadded=unpadded)
    with mock_cuda_unavailable(), forbid_native_compile():
        return bound.to_code(config)


@pytest.mark.parametrize(
    ("rows", "threads", "vec"), [(1, 96, 8), (2, 96, 8), (1, 192, 4)]
)
def test_an_exact_layout_runs_the_row_unpadded(
    rows: int, threads: int, vec: int
) -> None:
    code = _render(rows=rows, threads=threads, vec=vec, unpadded=True)
    assert f"block=({threads}, {rows}, 1)" in code, code
    # One packet per thread and input, no mask, and the cross-warp combine of
    # whole warps.
    assert "< 768" not in code, code
    assert f"ir.VectorType.get([{vec}]" in code, code
    assert f"group_span={threads}" in code, code
    assert "_cute_grouped_reduce_shared_two_stage(" in code, code


def test_the_knob_off_keeps_the_padded_row() -> None:
    code = _render(rows=1, threads=96, vec=8, unpadded=False)
    # The masked row stays on one warp walking scalar lanes.
    assert "block=(32, 1, 1)" in code, code
    assert "< 768" in code, code
    assert "range(24)" in code, code


@pytest.mark.parametrize(
    ("rows", "threads", "vec", "kept"),
    [
        pytest.param(2, 96, 8, True, id="exact"),
        pytest.param(2, 64, 8, False, id="short"),
        pytest.param(2, 96, 4, False, id="narrow"),
        pytest.param(16, 96, 8, False, id="beyond-the-thread-budget"),
    ],
)
def test_normalization_keeps_the_knob_only_for_an_exact_layout(
    rows: int, threads: int, vec: int, kept: bool
) -> None:
    # No exact layout leaves a partial warp: a thread count below a warp is a
    # power of two (``num_threads`` rejects 48), as is a vector width, so
    # their product is never the length of a masked row.
    config = _layout(_bind(), rows=rows, threads=threads, vec=vec, unpadded=True)
    assert config.config["cute_unpadded_rows"] is kept


def _softmax(x: torch.Tensor) -> torch.Tensor:
    rows, _n = x.shape
    out = torch.empty_like(x)
    for row in hl.tile(rows):
        v = x[row, :].to(torch.float32)
        e = torch.exp(v - torch.amax(v, dim=-1, keepdim=True))
        out[row, :] = (e / torch.sum(e, dim=-1, keepdim=True)).to(out.dtype)
    return out


def _row_argmax(x: torch.Tensor) -> torch.Tensor:
    rows, _n = x.shape
    out = torch.empty([rows], dtype=torch.int64, device=x.device)
    for row in hl.tile(rows):
        out[row] = torch.argmax(x[row, :], dim=-1)
    return out


def _add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:  # noqa: FURB118
    return left + right


def _padded_count(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    for row in hl.tile(x.size(0)):
        out[row] = hl.reduce(_add, x[row, :].float(), dim=1, other=1.0)
    return out


@functools.cache
def _bind_row(kernel: str, static_shapes: bool = True) -> BoundKernel:
    fn = {"softmax": _softmax, "argmax": _row_argmax, "count": _padded_count}[kernel]
    dtype = torch.bfloat16 if kernel == "softmax" else torch.float32
    bound = helion.kernel(
        fn, backend="cute", static_shapes=static_shapes, autotune_effort="none"
    )
    with mock_cuda_unavailable(), forbid_native_compile():
        return cpu_bind(bound, (torch.empty(19, 768, dtype=dtype),))


def _row_config(
    bound: BoundKernel, threads: int, vec: int, **extra: object
) -> helion.Config:
    """One row per CTA of ``threads`` threads with ``vec`` elements each,
    the knob on."""
    spec = bound.config_spec
    (reduction,) = spec.reduction_block_ids
    return helion.Config(
        block_sizes=[1],
        num_threads=[
            threads if block_id == reduction else 1
            for block_id in spec.num_threads.valid_block_ids()
        ],
        cute_vector_widths=[
            vec if block_id == reduction else 1
            for block_id in spec.cute_vector_widths.valid_block_ids()
        ],
        cute_unpadded_rows=True,
        **extra,
    )


@pytest.mark.parametrize("reduction_loops", [None, [None], [2048]])
def test_every_spelling_of_a_persistent_row_runs_it_unpadded(
    reduction_loops: list[int | None] | None,
) -> None:
    # A chunk past the 768 columns is the persistent row too.  The decision
    # reads the normalized reduction loops, so the spellings agree, and a
    # normalized config normalizes to itself.
    bound = _bind_row("softmax")
    spec = bound.config_spec
    extra = {} if reduction_loops is None else {"reduction_loops": reduction_loops}
    normalized = spec.normalized_config(_row_config(bound, 96, 8, **extra))
    assert normalized.config["cute_unpadded_rows"] is True
    assert normalized == spec.normalized_config(_row_config(bound, 96, 8))
    assert spec.normalized_config(normalized) == normalized


@pytest.mark.parametrize(("unpadded", "unroll"), [(True, 1), (False, 8)])
def test_an_unpadded_row_leaves_no_lane_loop_to_unroll(
    unpadded: bool, unroll: int
) -> None:
    # Unpadded, each thread's lanes are its vector's elements, a constexpr
    # loop: ``cute_lane_unroll`` resets.  The padded row on the same threads
    # walks 24 live lanes on one warp, which the unroll takes.
    bound = _bind_row("softmax")
    config = _row_config(bound, 96, 8, cute_lane_unroll=8)
    config.config["cute_unpadded_rows"] = unpadded
    normalized = bound.config_spec.normalized_config(config)
    assert normalized.config["cute_unpadded_rows"] is unpadded
    assert normalized.config["cute_lane_unroll"] == unroll


@pytest.mark.parametrize(
    ("kernel", "static_shapes", "threads", "vec"),
    [
        # A dynamic extent is planned from its hint: no row runs unpadded.
        pytest.param("softmax", False, 96, 8, id="dynamic-extent"),
        # argmax reduces within a warp: its 768 columns roll into 32-element
        # chunks, and a rolled row is not a persistent one.
        pytest.param("argmax", True, 192, 4, id="rolled-argmax"),
    ],
)
def test_the_knob_resets_where_no_row_runs_unpadded(
    kernel: str, static_shapes: bool, threads: int, vec: int
) -> None:
    spec = _bind_row(kernel, static_shapes).config_spec
    normalized = spec.normalized_config(
        _row_config(_bind_row(kernel, static_shapes), threads, vec)
    )
    assert normalized.config["cute_unpadded_rows"] is False
    assert spec.normalized_config(normalized) == normalized


def test_a_row_an_hl_reduce_reduces_keeps_its_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # hl.reduce folds its ``other`` into every padded slot, and under a sum
    # other=1.0 counts them: the slots are part of the result, so the row
    # never runs unpadded.  Normalization turns the knob off, the seeds leave
    # it out, and with the knob kept past normalization the emitter still
    # keeps the row's mask and the 1.0 (unpadded, a zero row summed to 0
    # instead of 256).
    bound = _bind_row("count")
    spec = bound.config_spec
    config = _row_config(bound, 192, 4)
    assert spec.normalized_config(config).config["cute_unpadded_rows"] is False
    assert not any(
        seed.config.get("cute_unpadded_rows") for seed in spec.compiler_seed_configs
    )
    monkeypatch.setattr(search_surface, "_normalize_cute_lane_knobs", lambda *_: None)
    with mock_cuda_unavailable(), forbid_native_compile():
        code = bound.to_code(config)
    assert "< 768" in code, code
    assert "cutlass.Float32(1.0)" in code, code


def _two_roots(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sums_x = torch.empty([x.size(0)], dtype=torch.float32, device=x.device)
    sums_y = torch.empty([y.size(0)], dtype=torch.float32, device=y.device)
    for row in hl.tile(x.size(0)):
        sums_x[row] = torch.sum(x[row, :].to(torch.float32), dim=-1)
    for row in hl.tile(y.size(0)):
        sums_y[row] = torch.sum(y[row, :].to(torch.float32), dim=-1)
    return sums_x, sums_y


def test_a_row_the_thread_fit_trims_turns_the_knob_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two root loops of one masked row each, in exact layouts: 768 = 96 x 8
    # and 640 = 160 x 4.  The first runs unpadded, and the fit trims the
    # second's threads to 10 beside its 96.  The emitter takes that row
    # unpadded from its layout and refuses it (``_unpadded_lanes``: "10
    # threads do not cover the 640-wide row exactly"), so normalization
    # turns the knob off and both rows render padded.
    kernel = helion.kernel(
        _two_roots, backend="cute", static_shapes=True, autotune_effort="none"
    )
    args = (
        torch.empty(19, 768, dtype=torch.bfloat16),
        torch.empty(19, 640, dtype=torch.bfloat16),
    )
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(kernel, args)
        spec = bound.config_spec
        layouts = {768: (96, 8), 640: (160, 4)}
        rows = {
            block_id: spec.num_threads.block_id_lookup(block_id).size_hint
            for block_id in spec.reduction_block_ids
        }
        requested = helion.Config(
            block_sizes=[1, 1],
            num_threads=[
                layouts[rows[block_id]][0] if block_id in rows else 1
                for block_id in spec.num_threads.valid_block_ids()
            ],
            cute_vector_widths=[
                layouts[rows[block_id]][1] if block_id in rows else 1
                for block_id in spec.cute_vector_widths.valid_block_ids()
            ],
            cute_unpadded_rows=True,
        )
        config = spec.normalized_config(requested)
        assert config.config["cute_unpadded_rows"] is False
        code = bound.to_code(config)
        monkeypatch.setattr(
            search_surface, "_normalize_cute_lane_knobs", lambda *_: None
        )
        with pytest.raises(
            exc.BackendUnsupported,
            match="10 threads do not cover the 640-wide row exactly",
        ):
            bound.to_code(requested)
    assert "< 768" in code, code
    assert "< 640" in code, code


@pytest.mark.parametrize(
    ("threads", "requested", "resident", "shared_axis", "refusal"),
    [
        pytest.param(96, 96, False, False, None, id="exact"),
        pytest.param(64, 96, False, False, "64 threads", id="trimmed-by-the-budget"),
        pytest.param(32, 96, False, False, "32 threads", id="capped-at-a-warp"),
        pytest.param(96, 96, True, False, "resident in registers", id="resident"),
        pytest.param(
            96, 96, False, True, "runs on another block's threads", id="shared-axis"
        ),
    ],
)
def test_the_emitter_refuses_a_row_its_threads_do_not_cover(
    threads: int,
    requested: int,
    resident: bool,
    shared_axis: bool,
    refusal: str | None,
) -> None:
    # What normalization admits the emitter checks again on the threads that
    # survived the budget: anything short of one exact vector per thread
    # (threads trimmed by the tile axes, an indexed reduction's warp cap, a
    # resident or shared-axis row) would address past the row.
    emitter = cast("Any", object.__new__(CutePersistentReductionEmitter))
    emitter.strategy = SimpleNamespace(
        _thread_count=threads,
        _cute_requested_threads=requested,
        _cute_resident_reduction=resident,
    )
    if refusal is None:
        assert emitter._unpadded_lanes(768, shared_axis) == 8
    else:
        with pytest.raises(exc.BackendUnsupported, match=refusal):
            emitter._unpadded_lanes(768, shared_axis)


def test_the_seeds_offer_the_knob_to_the_search() -> None:
    spec = _bind().config_spec
    seeds = [dict(seed) for seed in spec.compiler_seed_configs]
    (reduction,) = spec.reduction_block_ids
    layouts = [
        (
            seed["block_sizes"],
            spec.num_threads.config_get(seed["num_threads"], reduction),
            spec.cute_vector_widths.config_get(seed["cute_vector_widths"], reduction),
        )
        for seed in seeds
        if seed.get("cute_unpadded_rows") is True
    ]
    # One row per CTA: bf16 x fills a 16-byte packet with 8 elements (96
    # threads), the alternate takes 4 (192 threads).
    rows = [CutePersistentUnpaddedRowsHeuristic.ROW_BLOCK_SIZE]
    assert layouts == [(rows, 96, 8), (rows, 192, 4)], seeds
    field = spec._flat_fields()["cute_unpadded_rows"]
    assert set(field.search_choices) == {False, True}


@functools.cache
def _outputs(unpadded: bool) -> tuple[torch.Tensor, ...]:
    args = dropout_residual_norm_args(768, DEVICE)
    bound = _dropout_residual_norm._bind_isolated(args)
    config = _layout(bound, rows=1, threads=96, vec=8, unpadded=unpadded)
    assert ("< 768" not in bound.to_code(config)) is unpadded
    return bound.compile_config(config)(*args)


@pytest.mark.cuda
@skipIfNotCUDA()
def test_cuda_unpadded_rows_match_the_padded_rows() -> None:
    keep, centered, norm = _outputs(True)
    padded_keep, padded_centered, padded_norm = _outputs(False)
    assert torch.equal(keep.view(torch.uint8), padded_keep.view(torch.uint8))
    # Another summation order, and the contraction of ``residual + keep * x * c``.
    torch.testing.assert_close(centered, padded_centered, rtol=0, atol=1e-5)
    torch.testing.assert_close(
        norm.float(), padded_norm.float(), rtol=0.016, atol=0.016
    )
