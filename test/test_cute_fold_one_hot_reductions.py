"""The closed form of a sum over a one-hot row
(``helion/_compiler/cute/fold_one_hot_reductions.py``)."""

from __future__ import annotations

import re
from unittest.mock import patch

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import cpu_render_target
from helion._testing import onlyBackends
from helion._testing import skipIfNotCUDA
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only

_FOLD = "helion._compiler.cute.fold_one_hot_reductions.fold_one_hot_reductions"


def _ce_backward_fn(
    labels: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    grad_in: torch.Tensor,
    logits: torch.Tensor,
    lse: torch.Tensor,
) -> torch.Tensor:
    """better-benchmark's cross-entropy backward: the NLL gradient's one-hot
    row, summed, fed back through ``log_softmax`` and added to ``grad_in``."""
    rows, vocab = logits.shape
    out = torch.empty_like(grad_in)
    for tile_r in hl.tile(rows):
        label = labels[tile_r]
        valid = label != -100
        target = torch.where(valid, label, 0)
        scale = torch.where(valid, numerator[()] / denominator[()], 0.0)
        x = logits[tile_r, :].to(torch.float32)
        classes = hl.arange(x.size(1), dtype=torch.int64)
        grad = torch.where(target[:, None] == classes[None, :], -1.0, 0.0)
        grad = grad * scale[:, None]
        grad_sum = torch.sum(grad, dim=-1, keepdim=True)
        probs = torch.exp(x - lse[tile_r][:, None])
        total = grad_in[tile_r, :].to(torch.float32) + (grad - probs * grad_sum)
        out[tile_r, :] = total.to(out.dtype)
    return out


def _ce_backward_flat_sum_fn(
    labels: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    grad_in: torch.Tensor,
    logits: torch.Tensor,
    lse: torch.Tensor,
) -> torch.Tensor:
    """``_ce_backward_fn`` summing without ``keepdim``, the sum broadcast
    back with ``[:, None]``."""
    rows, vocab = logits.shape
    out = torch.empty_like(grad_in)
    for tile_r in hl.tile(rows):
        label = labels[tile_r]
        valid = label != -100
        target = torch.where(valid, label, 0)
        scale = torch.where(valid, numerator[()] / denominator[()], 0.0)
        x = logits[tile_r, :].to(torch.float32)
        classes = hl.arange(x.size(1), dtype=torch.int64)
        grad = torch.where(target[:, None] == classes[None, :], -1.0, 0.0)
        grad = grad * scale[:, None]
        grad_sum = torch.sum(grad, dim=-1)
        probs = torch.exp(x - lse[tile_r][:, None])
        total = grad_in[tile_r, :].to(torch.float32) + (
            grad - probs * grad_sum[:, None]
        )
        out[tile_r, :] = total.to(out.dtype)
    return out


def _smoothed_sum_fn(labels: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A one-hot row whose misses are not zero (label smoothing): the sum adds
    rounding errors, which no closed form reproduces."""
    rows, vocab = x.shape
    out = torch.empty([rows, 1], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(rows):
        row = x[tile_r, :].to(torch.float32)
        classes = hl.arange(row.size(1), dtype=torch.int64)
        hit = labels[tile_r][:, None] == classes[None, :]
        weights = torch.where(hit, 0.9, 0.1 / (vocab - 1))
        out[tile_r, :] = torch.sum(weights * row, dim=-1, keepdim=True)
    return out


def _picked_logit_fn(labels: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The selected value varies along the row: a gather, not a closed form."""
    rows, vocab = x.shape
    out = torch.empty([rows, 1], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(rows):
        row = x[tile_r, :].to(torch.float32)
        classes = hl.arange(row.size(1), dtype=torch.int64)
        hit = labels[tile_r][:, None] == classes[None, :]
        out[tile_r, :] = torch.sum(torch.where(hit, row, 0.0), dim=-1, keepdim=True)
    return out


def _tiled_sum_fn(labels: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A one-hot sum over each tile of a tiled dim: its iota counts within
    the tile, and the mask drops the partial last tile."""
    out = torch.empty([labels.size(0)], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(labels.size(0)):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(x.size(0)):
            idx = hl.arange(tile_n.block_size)
            hit = labels[tile_m][None, :] == idx[:, None]
            acc = acc + torch.sum(torch.where(hit, -1.0, 0.0), dim=0)
        out[tile_m] = acc
    return out


def _tiled_fixed_block_sum_fn(labels: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``_tiled_sum_fn`` with a literal block size and iota length."""
    out = torch.empty([labels.size(0)], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(labels.size(0)):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for _tile_n in hl.tile(x.size(0), block_size=32):
            hit = labels[tile_m][None, :] == hl.arange(32)[:, None]
            acc = acc + torch.sum(torch.where(hit, -1.0, 0.0), dim=0)
        out[tile_m] = acc
    return out


def _labelled_row_fn(labels: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A one-hot row summed and added back to the row."""
    rows, vocab = x.shape
    out = torch.empty([rows, vocab], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(rows):
        row = x[tile_r, :].to(torch.float32)
        classes = hl.arange(row.size(1), dtype=torch.int64)
        hit = labels[tile_r][:, None] == classes[None, :]
        grad_sum = torch.sum(torch.where(hit, -1.0, 0.0), dim=-1, keepdim=True)
        out[tile_r, :] = row + grad_sum
    return out


def _int8_iota_row_fn(labels: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``_labelled_row_fn`` with an int8 iota."""
    rows, vocab = x.shape
    out = torch.empty([rows, vocab], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(rows):
        row = x[tile_r, :].to(torch.float32)
        classes = hl.arange(row.size(1), dtype=torch.int8)
        hit = labels[tile_r][:, None] == classes[None, :]
        grad_sum = torch.sum(torch.where(hit, -1.0, 0.0), dim=-1, keepdim=True)
        out[tile_r, :] = row + grad_sum
    return out


def _host_extent_row_fn(labels: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``_labelled_row_fn`` counting to the host's ``vocab``."""
    rows, vocab = x.shape
    out = torch.empty([rows, vocab], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(rows):
        row = x[tile_r, :].to(torch.float32)
        classes = hl.arange(vocab, dtype=torch.int64)
        hit = labels[tile_r][:, None] == classes[None, :]
        grad_sum = torch.sum(torch.where(hit, -1.0, 0.0), dim=-1, keepdim=True)
        out[tile_r, :] = row + grad_sum
    return out


def _add(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:  # noqa: FURB118
    return left + right


def _reduced_row_fn(labels: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``_labelled_row_fn`` summing with ``hl.reduce``."""
    rows, vocab = x.shape
    out = torch.empty([rows, vocab], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(rows):
        row = x[tile_r, :].to(torch.float32)
        classes = hl.arange(row.size(1), dtype=torch.int64)
        hit = labels[tile_r][:, None] == classes[None, :]
        grad = torch.where(hit, -1.0, 0.0)
        out[tile_r, :] = row + hl.reduce(_add, grad, dim=1, keep_dims=True)
    return out


def _folded_wide_transpose_fn(labels: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Wide rows plus a one-hot sum, transposed: the load walks each row
    along its stride-1 dim, the store walks as wide a dim along the tile."""
    n, w = x.shape
    out = torch.empty([w, n], dtype=x.dtype, device=x.device)
    for tile in hl.tile(n):
        row = x[tile, :]
        hit = labels[tile][:, None] == hl.arange(w)[None, :]
        grad_sum = torch.sum(torch.where(hit, -1.0, 0.0), dim=-1, keepdim=True)
        out[:, tile] = (row + grad_sum).T.to(x.dtype)
    return out


def _rank0_sum_fn(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile] = x[tile] + s[()].sum(dim=0)
    return out


def _one_hot_sum_only_fn(labels: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A one-hot row summed, the sum the only use of the row's dim."""
    rows, vocab = x.shape
    out = torch.empty([rows], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(rows):
        hit = labels[tile_r][:, None] == hl.arange(vocab)[None, :]
        out[tile_r] = torch.sum(torch.where(hit, -1.0, 0.0), dim=-1)
    return out


def _one_hot_scaled_sum_fn(
    labels: torch.Tensor, s: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """``_labelled_row_fn`` with the one-hot row scaled per row."""
    out = torch.empty_like(x)
    for tile_r in hl.tile(x.size(0)):
        row = x[tile_r, :]
        classes = hl.arange(row.size(1), dtype=torch.int64)
        hit = labels[tile_r][:, None] == classes[None, :]
        grad = torch.where(hit, -1.0, 0.0) * s[tile_r][:, None]
        out[tile_r, :] = row + torch.sum(grad, dim=-1, keepdim=True)
    return out


def _row_scale_fn(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Rows scaled by a factor per row, with no one-hot sum to fold."""
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        out[tile, :] = x[tile, :] * s[tile][:, None]
    return out


def _one_hot_scaled_fn(
    labels: torch.Tensor, s: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """A one-hot row of per-row scales summed, the sum feeding more of the
    rows' arithmetic: ``r * s - x``."""
    rows, vocab = x.shape
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        hit = labels[tile_r][:, None] == hl.arange(vocab)[None, :]
        r = torch.sum(torch.where(hit, s[tile_r][:, None], 0.0), dim=-1)
        out[tile_r, :] = r[:, None] * s[tile_r][:, None] - x[tile_r, :]
    return out


def _kernel(fn: object, **settings: object) -> helion.Kernel:
    return helion.kernel(
        fn, backend="cute", static_shapes=True, autotune_effort="none", **settings
    )


def _ce_args(
    rows: int = 16, vocab: int = 1000, device: str = "cpu"
) -> tuple[torch.Tensor, ...]:
    labels = torch.randint(0, vocab, (rows,), device=device)
    logits = torch.randn(rows, vocab, device=device).to(torch.bfloat16)
    lse = torch.logsumexp(logits.float(), dim=-1)
    return (
        labels,
        torch.tensor(1.0, device=device),
        torch.tensor(float(rows), device=device),
        torch.randn(rows, vocab, device=device).to(torch.bfloat16),
        logits,
        lse,
    )


def _render(
    kernel: helion.Kernel, args: tuple[torch.Tensor, ...], **config: object
) -> str:
    with cpu_render_target():
        bound = cpu_bind(kernel, args)
        spec = bound.config_spec
        chosen = spec.default_config() if not config else helion.Config(**config)
        return bound.to_code(spec.normalized_config(chosen))


@pytest.mark.parametrize(
    ("fn", "config"),
    [
        pytest.param(_ce_backward_fn, {}, id="default"),
        pytest.param(
            _ce_backward_fn,
            {"block_sizes": [1], "reduction_loops": [256]},
            id="looped",
        ),
        pytest.param(_ce_backward_flat_sum_fn, {}, id="dropped_dim"),
    ],
)
def test_ce_backward_sums_its_one_hot_row_in_closed_form(
    fn: object, config: dict[str, object]
) -> None:
    """No sweep sums the one-hot row: the row is read and written once, and
    the sum is the label's term, kept only for an in-range label."""
    code = _render(_kernel(fn), _ce_args(), **config)
    assert "_cute_grouped_reduce" not in code, code
    assert "grad_sum_acc" not in code
    assert "operator.le(target, v_" in code
    assert code.count("logits.iterator") == 1


def _folds(fn: object, args: tuple[object, ...]) -> int:
    """The sums ``fold_one_hot_reductions`` folds while ``fn`` binds."""
    with cpu_render_target():
        bound = cpu_bind(_kernel(fn), args)
    return bound.host_function.device_ir.folded_one_hot_sums


_LABELS = torch.randint(0, 100, (16,))


@pytest.mark.parametrize(
    ("fn", "args", "folds"),
    [
        # The fold needs the whole row: a tile's sum is no closed form of the
        # dim's full extent.
        pytest.param(_tiled_sum_fn, (_LABELS, torch.empty(100)), 0, id="tiled"),
        pytest.param(
            _tiled_fixed_block_sum_fn, (_LABELS, torch.empty(100)), 0, id="tiled_32"
        ),
        # An int8 iota of 512 wraps and counts some labels twice.
        pytest.param(
            _int8_iota_row_fn, (_LABELS, torch.empty(16, 512)), 0, id="wrapping_iota"
        ),
        pytest.param(
            _int8_iota_row_fn, (_LABELS, torch.empty(16, 128)), 1, id="int8_iota"
        ),
        pytest.param(
            _host_extent_row_fn, (_LABELS, torch.empty(16, 1000)), 1, id="host_extent"
        ),
        # A 0-d sum has no dim to fold (``dims[0] % rank`` needs a rank).
        pytest.param(
            _rank0_sum_fn, (torch.empty(64), torch.tensor(2.0)), 0, id="rank0"
        ),
    ],
)
def test_only_a_whole_row_counted_by_a_distinct_iota_folds(
    fn: object, args: tuple[object, ...], folds: int
) -> None:
    assert _folds(fn, args) == folds


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_a_half_fold_stays_in_the_sums_dtype(dtype: torch.dtype) -> None:
    """The closed form's zero takes the sum's dtype, as the traced scalars
    do: a float32 one promoted the folded sum and what it feeds, which then
    skipped the rounding the sweep's half sum takes after each op."""
    args = (_LABELS, torch.empty(16, dtype=dtype), torch.empty(16, 100, dtype=dtype))
    assert _folds(_one_hot_scaled_fn, args) == 1
    code = _render(_kernel(_one_hot_scaled_fn), args)
    assert "Float32" not in code, code


def test_an_hl_reduce_sum_of_a_one_hot_row_folds() -> None:
    """``canonicalize_reductions`` turns the ``hl.reduce`` into a sum of the
    row masked to the identity, which names the dtype it sums in."""
    assert _folds(_reduced_row_fn, (_LABELS, torch.empty(16, 1000))) == 1


def test_narrow_labels_are_range_checked_in_the_compare_dtype() -> None:
    """``int8_labels == int64_iota`` compares in int64: the range check must
    too, where ``int8(1000)`` would be -24 and drop every label."""
    args = (torch.randint(0, 100, (16,), dtype=torch.int8), torch.empty(16, 1000))
    code = _render(_kernel(_labelled_row_fn), args)
    assert "_cute_grouped_reduce" not in code
    assert "cutlass.Int8(-24)" not in code
    assert re.search(r"(v_\d+) = cutlass\.Int64\(load\w*\)", code), code
    assert re.search(r"= operator\.le\(v_\d+, v_\d+\)", code)
    assert re.search(r"v_\d+ = 999$", code, re.MULTILINE)


def test_a_dim_only_the_folded_sum_used_retires() -> None:
    """With its sum folded, nothing uses the row's dim: it retires, as merged
    dims do, so it claims no thread axis (whose threads each repeated the
    label's load and the store) and no reduction knobs."""
    with cpu_render_target():
        bound = cpu_bind(
            _kernel(_one_hot_sum_only_fn), (_LABELS, torch.empty(16, 1000))
        )
        assert not any(block.reduction for block in bound.env.block_sizes)
        code = bound.to_code(helion.Config(block_sizes=[16]))
    assert "block=(16, 1, 1)" in code, code
    assert "cute_reduction_reloads" not in bound.config_spec.default_config().config


def test_the_closed_form_does_not_claim_the_sums_schema() -> None:
    """Its nodes take the sum's source location, but not its
    ``original_aten``, the schema ``ReductionLowering`` reads a reduction's
    dims from."""
    with cpu_render_target():
        bound = cpu_bind(_kernel(_labelled_row_fn), (_LABELS, torch.empty(16, 1000)))
    graphs = bound.host_function.device_ir.graphs
    nodes = [node for info in graphs for node in info.graph.nodes]
    assert not any(node.target is torch.ops.aten.sum.dim_IntList for node in nodes)
    assert not any(
        node.meta.get("original_aten") is torch.ops.aten.sum.dim_IntList
        for node in nodes
    )


def test_folded_ce_backward_gets_no_strided_pointwise_tiles() -> None:
    """Without its reduction the kernel is pointwise, and its stride-1 dim is
    the row: the pointwise tile seeds, which vectorize a tiled axis, had only
    the rows to tile (1024-row tiles, every thread walking whole rows)."""
    with cpu_render_target():
        bound = cpu_bind(_kernel(_ce_backward_fn), _ce_args(vocab=4096))
    assert "cute_pointwise_vec" not in bound.config_spec.autotuner_heuristics


def test_a_wide_transpose_keeps_its_strided_pointwise_tiles() -> None:
    """Beside a folded sum, a store that moves the wide dim along the tile
    keeps the tile seeds, which still vectorize its traffic."""
    args = (
        torch.randint(0, 4096, (4096,)),
        torch.empty(4096, 4096, dtype=torch.bfloat16),
    )
    with cpu_render_target():
        bound = cpu_bind(_kernel(_folded_wide_transpose_fn), args)
    assert bound.host_function.device_ir.folded_one_hot_sums == 1
    assert "cute_pointwise_vec" in bound.config_spec.autotuner_heuristics


def test_unfolded_wide_rows_keep_their_pointwise_tiles() -> None:
    """Only a kernel whose one-hot sum folded drops the pointwise tile
    seeds over rows wider than a CTA of packets: rows scaled per row, 4096
    bfloat16 wide, keep them beside the row seeds."""
    args = (
        torch.empty(8192, 4096, dtype=torch.bfloat16),
        torch.empty(8192, dtype=torch.bfloat16),
    )
    with cpu_render_target():
        bound = cpu_bind(_kernel(_row_scale_fn), args)
    heuristics = bound.config_spec.autotuner_heuristics
    assert "cute_pointwise_vec" in heuristics, heuristics
    assert "cute_full_chunk_row" in heuristics, heuristics


@pytest.mark.parametrize("fn", [_smoothed_sum_fn, _picked_logit_fn])
def test_sums_that_are_no_one_hot_closed_form_still_reduce(fn: object) -> None:
    args = (
        torch.randint(0, 1000, (16,)),
        torch.randn(16, 1000).to(torch.bfloat16),
    )
    assert _folds(fn, args) == 0


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """The bits of ``tensor``'s elements, which tell ``-0`` from ``+0``."""
    return tensor.view({2: torch.int16, 4: torch.int32}[tensor.element_size()])


@pytest.mark.cuda
@skipIfNotCUDA()
@onlyBackends(["cute"])
class TestFoldOneHotReductions(TestCase):
    def test_folded_ce_backward_is_bitwise_the_sweep(self) -> None:
        """Ignored and out-of-range labels, and numerators and denominators
        making ``scale`` zero, infinite and NaN: every bit of the output
        matches the kernel that sums the one-hot row in a sweep."""
        torch.manual_seed(0)
        labels, _, _, grad_in, logits, lse = _ce_args(32, 1000, DEVICE)
        labels[::5] = -100
        labels[1] = 1000
        labels[2] = 1003
        labels[3] = -7
        grad_in[0, :8] = -0.0
        scales = ((1.0, 32.0), (0.0, 32.0), (-1.5, 0.0), (float("nan"), 1.0))
        for fn, config in (
            (_ce_backward_fn, {}),
            (_ce_backward_fn, {"block_sizes": [1], "reduction_loops": [256]}),
            (_ce_backward_flat_sum_fn, {}),
        ):
            settings = {"config": helion.Config(**config)} if config else {}
            folded = _kernel(fn, **settings)
            with patch(_FOLD, return_value=0):
                swept = _kernel(fn, **settings)
                # The fold runs while the kernel binds.
                swept.bind(_ce_args(32, 1000, DEVICE))
            for numerator, denominator in scales:
                args = (
                    labels,
                    torch.tensor(numerator, device=DEVICE),
                    torch.tensor(denominator, device=DEVICE),
                    grad_in,
                    logits,
                    lse,
                )
                code, folded_out = code_and_output(folded, args)
                self.assertNotIn("_cute_grouped_reduce", code)
                code, swept_out = code_and_output(swept, args)
                self.assertIn("_cute_grouped_reduce", code)
                torch.testing.assert_close(
                    _bits(folded_out), _bits(swept_out), rtol=0, atol=0
                )

    def test_a_folded_half_sum_is_bitwise_the_sweep(self) -> None:
        """``r * s - x`` with ``s = 1 + eps`` and ``x = 1 + 2 * eps``: the
        sweep rounds ``r * s`` to ``1 + 2 * eps`` in the half dtype and
        outputs 0, where a float32 folded sum kept ``eps**2`` (``2**-20`` in
        float16)."""
        for dtype, eps in ((torch.float16, 2.0**-10), (torch.bfloat16, 2.0**-7)):
            labels = torch.randint(0, 100, (16,), device=DEVICE)
            s = torch.full((16,), 1 + eps, dtype=dtype, device=DEVICE)
            x = torch.full((16, 100), 1 + 2 * eps, dtype=dtype, device=DEVICE)
            args = (labels, s, x)
            folded = _kernel(_one_hot_scaled_fn)
            with patch(_FOLD, return_value=0):
                swept = _kernel(_one_hot_scaled_fn)
                # The fold runs while the kernel binds.
                swept.bind(args)
            _, folded_out = code_and_output(folded, args)
            _, swept_out = code_and_output(swept, args)
            torch.testing.assert_close(
                _bits(folded_out), _bits(swept_out), rtol=0, atol=0
            )
            self.assertFalse(folded_out.any())

    def test_a_zero_sum_of_negative_zeros_is_positive_as_in_eager(self) -> None:
        """Missing labels beside a negative scale make every term ``-0``.
        With a thread per element, the sweep's warp reduction of them
        returned ``-0``; the folded sum is eager ``torch.sum``'s ``+0``.
        Added to rows of ``-0``, the sum keeps its sign."""
        labels = torch.tensor([-100, 32, 40, -1] * 4, device=DEVICE)
        s = torch.full((16,), -1.5, device=DEVICE)
        x = torch.full((16, 32), -0.0, device=DEVICE)
        config = helion.Config(
            block_sizes=[16], num_threads=[4, 32], cute_vector_widths=[1, 1]
        )
        kernel = _kernel(_one_hot_scaled_sum_fn, config=config)
        code, out = code_and_output(kernel, (labels, s, x))
        self.assertNotIn("warp_reduction", code)
        hit = labels[:, None] == torch.arange(32, device=DEVICE)[None, :]
        grad = torch.where(hit, -1.0, 0.0) * s[:, None]
        expected = x + torch.sum(grad, dim=-1, keepdim=True)
        self.assertFalse(expected.signbit().any())
        torch.testing.assert_close(_bits(out), _bits(expected), rtol=0, atol=0)
