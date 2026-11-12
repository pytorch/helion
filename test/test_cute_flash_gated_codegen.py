"""Which gated-attention kernels the fused bodies of ``cute_flash_gated`` take,
and what they render (on CPU; the GPU tests are marked ``cuda``).  The HSTU
kernels and helpers shared with the GPU tests stay in test_cute_flash_gated.py,
where test_cute_flash_gated_row_mma.py imports them from too."""

from __future__ import annotations

import ast
import re

import pytest
import torch

from test.test_cute_flash_gated import _APPROX_GATE_DIVISION
from test.test_cute_flash_gated import _IEEE_GATE_DIVISION
from test.test_cute_flash_gated import _cpu_codegen_patches
from test.test_cute_flash_gated import _division_gate_kernel
from test.test_cute_flash_gated import _extrema_gate_kernel
from test.test_cute_flash_gated import _gated_markers
from test.test_cute_flash_gated import _hstu_args
from test.test_cute_flash_gated import _hstu_module
from test.test_cute_flash_gated import _jagged_inputs
from test.test_cute_flash_gated import _poisoned_output_kernel
from test.test_cute_flash_gated import _sigmoid_gate_kernel
from test.test_cute_flash_gated import _with_fast_math

import helion
from helion._compiler.cute import cute_flash_gated
from helion._compiler.cute.math_templates import SIGMOID_TEMPLATE
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_flash import compile_and_run
from helion._testing.cute_flash import gated_attention_match
import helion.language as hl

pytestmark = pytest.mark.cpu_only

pytest.importorskip("cutlass")


@helion.kernel(backend="cute", static_shapes=True)
def _lane_sequences(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    masked_store: hl.constexpr = False,
) -> torch.Tensor:
    """The HSTU example's dataflow with one sequence per head: the offsets are
    indexed by the output lane's own axis.  It stores into a poisoned output,
    where rows the program does not store stay NaN, unmasked or, with
    ``masked_store``, as the example does, masked by the length."""
    scale = 1.0 / max_seq_len
    num_heads = hl.specialize(q.size(1))
    dim_v = hl.specialize(v.size(2))
    out = torch.full_like(v, float("nan"))
    for tile_h, tile_q in hl.tile([num_heads, max_seq_len], block_size=[1, None]):
        starts = seq_offsets[tile_h.begin]
        ends = seq_offsets[tile_h.begin + 1]
        seq_len = ends - starts
        if tile_q.begin < seq_len:
            mask_q = tile_q.index < seq_len
            q_blk = q[tile_q.index + starts, tile_h.begin, :]
            acc = hl.zeros([tile_q, dim_v], dtype=torch.float32)
            for tile_kv in hl.tile(0, tile_q.end, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]
                scores = (
                    torch.nn.functional.silu(torch.matmul(q_blk, k_blk.T) * alpha)
                    * scale
                )
                scores = torch.where(
                    (tile_q.index.unsqueeze(1) > tile_kv.index.unsqueeze(0))
                    & mask_q[:, None]
                    & mask_kv[None, :],
                    scores,
                    0.0,
                )
                acc += torch.matmul(scores.to(v.dtype), v_blk)
            if masked_store:
                hl.store(
                    out,
                    [tile_q.index + starts, tile_h.begin, slice(None)],
                    acc.to(out.dtype),
                    extra_mask=mask_q[:, None],
                )
            else:
                out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _scaled_row_offset(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    row_alpha: hl.constexpr,
) -> torch.Tensor:
    """``_lane_sequences`` (unmasked) reading its query rows at
    ``torch.add(tile_q.index, starts, alpha=row_alpha)``."""
    scale = 1.0 / max_seq_len
    num_heads = hl.specialize(q.size(1))
    dim_v = hl.specialize(v.size(2))
    out = torch.full_like(v, float("nan"))
    for tile_h, tile_q in hl.tile([num_heads, max_seq_len], block_size=[1, None]):
        starts = seq_offsets[tile_h.begin]
        ends = seq_offsets[tile_h.begin + 1]
        seq_len = ends - starts
        if tile_q.begin < seq_len:
            mask_q = tile_q.index < seq_len
            rows = torch.add(tile_q.index, starts, alpha=row_alpha)
            q_blk = q[rows, tile_h.begin, :]
            acc = hl.zeros([tile_q, dim_v], dtype=torch.float32)
            for tile_kv in hl.tile(0, tile_q.end, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]
                scores = (
                    torch.nn.functional.silu(torch.matmul(q_blk, k_blk.T) * alpha)
                    * scale
                )
                scores = torch.where(
                    (tile_q.index.unsqueeze(1) > tile_kv.index.unsqueeze(0))
                    & mask_q[:, None]
                    & mask_kv[None, :],
                    scores,
                    0.0,
                )
                acc += torch.matmul(scores.to(v.dtype), v_blk)
            out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _batches_from_one(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """The HSTU example's dataflow over batches 1 to n - 1: the root grid's
    batch axis starts at 1."""
    scale = 1.0 / max_seq_len
    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dim_v = hl.specialize(v.size(2))
    out = torch.full_like(v, float("nan"))
    for tile_b, tile_h, tile_q in hl.tile(
        [1, 0, 0], [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
    ):
        starts = seq_offsets[tile_b.begin]
        ends = seq_offsets[tile_b.begin + 1]
        seq_len = ends - starts
        if tile_q.begin < seq_len:
            mask_q = tile_q.index < seq_len
            q_blk = q[tile_q.index + starts, tile_h.begin, :]
            acc = hl.zeros([tile_q, dim_v], dtype=torch.float32)
            for tile_kv in hl.tile(0, tile_q.end, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]
                scores = (
                    torch.nn.functional.silu(torch.matmul(q_blk, k_blk.T) * alpha)
                    * scale
                )
                scores = torch.where(
                    (tile_q.index.unsqueeze(1) > tile_kv.index.unsqueeze(0))
                    & mask_q[:, None]
                    & mask_kv[None, :],
                    scores,
                    0.0,
                )
                acc += torch.matmul(scores.to(v.dtype), v_blk)
            out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _even_sequences(
    max_seq_len: int,
    alpha: float,
    attn_scale: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """``jagged_hstu_attn_2``'s sequence grid over the even sequences."""
    heads = hl.specialize(q.size(1))
    head_dim = hl.specialize(q.size(2))
    num_sequences = seq_offsets.size(0) - 1
    out = torch.empty_like(v)
    for seq_idx in hl.grid(0, num_sequences, 2):
        start = seq_offsets[seq_idx]
        end = seq_offsets[seq_idx + 1]
        for tile_q in hl.tile(start, end):
            q_blk = q[tile_q, :, :].transpose(0, 1)
            acc = hl.zeros([heads, tile_q, head_dim], dtype=torch.float32)
            for tile_kv in hl.tile(start, end):
                k_blk = k[tile_kv, :, :].transpose(0, 1)
                v_blk = v[tile_kv, :, :].transpose(0, 1)
                scores = torch.bmm(q_blk, k_blk.transpose(-2, -1)) * alpha
                scores = torch.nn.functional.silu(scores) * attn_scale
                causal_mask = tile_q.index.unsqueeze(1) >= tile_kv.index.unsqueeze(0)
                scores = torch.where(causal_mask[None, :, :], scores, 0.0)
                acc = acc + torch.bmm(scores.to(v.dtype), v_blk)
            out[tile_q, :, :] = acc.transpose(0, 1).to(out.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _sequences_from_one(
    max_seq_len: int,
    alpha: float,
    attn_scale: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """``jagged_hstu_attn_2``'s sequence grid over sequences 1 to n - 1."""
    heads = hl.specialize(q.size(1))
    head_dim = hl.specialize(q.size(2))
    num_sequences = seq_offsets.size(0) - 1
    out = torch.empty_like(v)
    for seq_idx in hl.grid(1, num_sequences):
        start = seq_offsets[seq_idx]
        end = seq_offsets[seq_idx + 1]
        for tile_q in hl.tile(start, end):
            q_blk = q[tile_q, :, :].transpose(0, 1)
            acc = hl.zeros([heads, tile_q, head_dim], dtype=torch.float32)
            for tile_kv in hl.tile(start, end):
                k_blk = k[tile_kv, :, :].transpose(0, 1)
                v_blk = v[tile_kv, :, :].transpose(0, 1)
                scores = torch.bmm(q_blk, k_blk.transpose(-2, -1)) * alpha
                scores = torch.nn.functional.silu(scores) * attn_scale
                causal_mask = tile_q.index.unsqueeze(1) >= tile_kv.index.unsqueeze(0)
                scores = torch.where(causal_mask[None, :, :], scores, 0.0)
                acc = acc + torch.bmm(scores.to(v.dtype), v_blk)
            out[tile_q, :, :] = acc.transpose(0, 1).to(out.dtype)
    return out


@helion.kernel()
def _sequence_scale_gate_kernel(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """Accepted: an fp32 per-sequence scale (a 0-d tensor, not exact in bf16)
    beside the bf16 scores, in a compare and a product, and a literal bound
    (not exact in bf16 either) in a compare."""
    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dimV = hl.specialize(v.size(2))
    out = torch.zeros_like(v)
    for tile_b, tile_h, tile_q in hl.tile(
        [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
    ):
        starts = seq_offsets[tile_b.begin]
        ends = seq_offsets[tile_b.begin + 1]
        seq_len = ends - starts
        if tile_q.begin < seq_len:
            mask_q = tile_q.index < seq_len
            q_blk = q[tile_q.index + starts, tile_h.begin, :]
            acc = hl.zeros([tile_q, dimV], dtype=torch.float32)
            for tile_kv in hl.tile(0, tile_q.end, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]
                qk = torch.matmul(q_blk, k_blk.T) * alpha
                scale = 1.0 / seq_len
                keep = (qk > scale) & (qk < 4.1)
                scores = torch.where(keep, torch.nn.functional.silu(qk) * scale, 0.0)
                scores = torch.where(
                    (tile_q.index.unsqueeze(1) > tile_kv.index.unsqueeze(0))
                    & mask_q[:, None]
                    & mask_kv[None, :],
                    scores,
                    0.0,
                )
                acc += torch.matmul(scores.to(v.dtype), v_blk)
            out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)
    return out


@helion.kernel()
def _remainder_gate_kernel(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """Accepted: a remainder of the bf16 scores by a literal not exact in
    bf16, which eager converts to bf16 first."""
    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dimV = hl.specialize(v.size(2))
    out = torch.zeros_like(v)
    for tile_b, tile_h, tile_q in hl.tile(
        [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
    ):
        starts = seq_offsets[tile_b.begin]
        ends = seq_offsets[tile_b.begin + 1]
        seq_len = ends - starts
        if tile_q.begin < seq_len:
            mask_q = tile_q.index < seq_len
            q_blk = q[tile_q.index + starts, tile_h.begin, :]
            acc = hl.zeros([tile_q, dimV], dtype=torch.float32)
            for tile_kv in hl.tile(0, tile_q.end, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]
                qk = torch.matmul(q_blk, k_blk.T) * alpha
                scores = torch.where(
                    (tile_q.index.unsqueeze(1) > tile_kv.index.unsqueeze(0))
                    & mask_q[:, None]
                    & mask_kv[None, :],
                    torch.remainder(qk, 0.3),
                    0.0,
                )
                acc += torch.matmul(scores.to(v.dtype), v_blk)
            out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)
    return out


@helion.kernel()
def _non_finite_bound_gate_kernel(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """Accepted: gate bounds that are not finite, a where's -inf branch, a
    -inf fill and an fp16 1e5 fill (inf once rounded to fp16)."""
    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dimV = hl.specialize(v.size(2))
    out = torch.zeros_like(v)
    for tile_b, tile_h, tile_q in hl.tile(
        [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
    ):
        starts = seq_offsets[tile_b.begin]
        ends = seq_offsets[tile_b.begin + 1]
        seq_len = ends - starts
        if tile_q.begin < seq_len:
            mask_q = tile_q.index < seq_len
            q_blk = q[tile_q.index + starts, tile_h.begin, :]
            acc = hl.zeros([tile_q, dimV], dtype=torch.float32)
            for tile_kv in hl.tile(0, tile_q.end, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]
                qk = torch.matmul(q_blk, k_blk.T) * alpha
                floor = hl.full([tile_q, tile_kv], float("-inf"), dtype=torch.float32)
                ceiling = hl.full([tile_q, tile_kv], 1e5, dtype=torch.float16)
                bounded = torch.minimum(
                    torch.maximum(torch.where(qk > 0.0, qk, float("-inf")), floor),
                    ceiling,
                )
                scores = torch.where(
                    (tile_q.index.unsqueeze(1) > tile_kv.index.unsqueeze(0))
                    & mask_q[:, None]
                    & mask_kv[None, :],
                    bounded,
                    0.0,
                )
                acc += torch.matmul(scores.to(v.dtype), v_blk)
            out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)
    return out


@helion.kernel()
def _row_index_gate_kernel(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
) -> torch.Tensor:
    """Accepted: the query row index beside bf16 values per element (in a
    product with the scores) and per row (in a compare with a bf16 0-d
    tensor), rounded to bf16 in both; the product comes first."""
    scale = 1.0 / max_seq_len
    num_heads = hl.specialize(q.size(1))
    num_batches = hl.specialize(seq_offsets.size(0) - 1)
    dimV = hl.specialize(v.size(2))
    out = torch.zeros_like(v)
    for tile_b, tile_h, tile_q in hl.tile(
        [num_batches, num_heads, max_seq_len], block_size=[1, 1, None]
    ):
        starts = seq_offsets[tile_b.begin]
        ends = seq_offsets[tile_b.begin + 1]
        seq_len = ends - starts
        if tile_q.begin < seq_len:
            mask_q = tile_q.index < seq_len
            q_blk = q[tile_q.index + starts, tile_h.begin, :]
            acc = hl.zeros([tile_q, dimV], dtype=torch.float32)
            for tile_kv in hl.tile(0, tile_q.end, block_size=None):
                mask_kv = tile_kv.index < seq_len
                k_blk = k[tile_kv.index + starts, tile_h.begin, :]
                v_blk = v[tile_kv.index + starts, tile_h.begin, :]
                qk = torch.matmul(q_blk, k_blk.T) * alpha
                scores = torch.nn.functional.silu(qk * tile_q.index.unsqueeze(1))
                limit = (seq_len * 0.5).to(torch.bfloat16)
                keep = (tile_q.index < limit).unsqueeze(1)
                scores = scores * scale * keep.to(scores.dtype)
                scores = torch.where(
                    (tile_q.index.unsqueeze(1) > tile_kv.index.unsqueeze(0))
                    & mask_q[:, None]
                    & mask_kv[None, :],
                    scores,
                    0.0,
                )
                acc += torch.matmul(scores.to(v.dtype), v_blk)
            out[tile_q.index + starts, tile_h.begin, :] = acc.to(out.dtype)
    return out


def _root_args() -> tuple[object, ...]:
    """Arguments of the root-form kernels: 512 query rows, eight 64-wide
    heads."""
    q = torch.empty((3044, 8, 64), dtype=torch.bfloat16)
    offsets = torch.zeros((9,), dtype=torch.int64)
    return (512, 1.0 / 64**2, q, torch.empty_like(q), torch.empty_like(q), offsets)


def _sequence_args() -> tuple[object, ...]:
    """Arguments of ``_sequences_from_one``: 64 sequences, four fp32 heads."""
    q = torch.empty((3913, 4, 64), dtype=torch.float32)
    offsets = torch.zeros((65,), dtype=torch.int32)
    return (
        128,
        1.0 / 64**2,
        1.0 / 128,
        q,
        torch.empty_like(q),
        torch.empty_like(q),
        offsets,
    )


def _match(kernel: helion.Kernel, args: tuple[object, ...]) -> object:
    """The config-independent gated-attention matcher on a CPU bind."""
    with cpu_target():
        kernel.reset()
        return gated_attention_match(kernel.bind(args))


# Each body's store condition: one in the tcgen05 body (128-row query tiles),
# one per row half of a warp's 16-row m-tile in a register-MMA row program.
_BODIES = (
    (
        "helion_flash_gated",
        helion.Config(block_sizes=[128, 128], cute_flash_kv_stage=2),
        "flash_store_row_ok",
        1,
    ),
    (
        "helion_flash_gated_row_mma",
        helion.Config(
            block_sizes=[32, 32], cute_flash_kv_stage=2, cute_flash_gated_row_warps=4
        ),
        "grm_store_ok",
        2,
    ),
)


@pytest.mark.parametrize(
    ("kind", "config", "store_ok", "count"), _BODIES, ids=["tcgen05", "row_mma"]
)
def test_a_sequence_per_lane_stores_the_rows_past_its_end(
    kind: str, config: helion.Config, store_ok: str, count: int
) -> None:
    """With one sequence per output lane no later sequence stores the rows
    the length masks, so an unmasked store writes them (the program's zeros)
    and only a store masked by the length skips them."""
    line = re.compile(rf"^ *{store_ok} = (.*)$", re.MULTILINE)
    for skips in (False, True):
        with cpu_target():
            _lane_sequences.reset()
            code = _lane_sequences.bind((*_root_args(), skips)).to_code(config)
        assert f"'kind': '{kind}'" in code
        stores = line.findall(code)
        assert len(stores) == count
        for store in stores:
            assert bool(re.search(r"\bflash_g\d+\b", store)) == skips, store


def test_a_row_index_that_scales_its_offset_stays_off_the_fused_bodies() -> None:
    """The fused bodies read and store row ``tile_q.index + starts``: a row
    index adding the offset with another ``alpha`` stays on the general
    path."""
    assert _match(_scaled_row_offset, (*_root_args(), 1)) is not None
    assert _match(_scaled_row_offset, (*_root_args(), 2)) is None


def test_root_grids_that_do_not_start_at_zero_stay_off_the_fused_bodies() -> None:
    """The fused bodies number the root grid's work items from 0 in steps of
    1, so a batch axis or a sequence grid starting at 1, and a sequence grid
    in steps of 2, stay on the general path; with that check patched out each
    of them matches."""
    assert _match(_lane_sequences, _root_args()) is not None
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            cute_flash_gated, "root_grid_origin_is_canonical", lambda ir: True
        )
        assert _match(_batches_from_one, _root_args()) is not None
        assert _match(_sequences_from_one, _sequence_args()) is not None
        assert _match(_even_sequences, _sequence_args()) is not None
    assert _match(_even_sequences, _sequence_args()) is None
    assert _match(_batches_from_one, _root_args()) is None
    assert _match(_sequences_from_one, _sequence_args()) is None


def test_a_non_finite_mask_fill_is_a_float_call() -> None:
    """The fill of a ``_mask_to`` the gate reads is spelled like every other
    constant: an infinity is ``float('-inf')``, where ``repr`` gives a bare
    name the generated module does not define.  The kernels' fills are
    zeros, so the test sets the scores' fill to -inf."""
    with cpu_target():
        _lane_sequences.reset()
        bound = _lane_sequences.bind(_root_args())
        (mask,) = [
            node
            for info in bound.host_function.device_ir.graphs
            for node in info.graph.nodes
            if getattr(node.target, "__name__", None) == "_mask_to"
        ]
        mask.args = (mask.args[0], float("-inf"))
        try:
            code = bound.to_code(_BODIES[0][1])
        finally:
            # The edited kernel stays in the bind cache.
            _lane_sequences.reset()
    assert "'kind': 'helion_flash_gated'" in code
    assert "flash_elem_in_tile_m" in code
    assert ", cutlass.Float32(float('-inf')))" in code
    assert "cutlass.Float32(-inf)" not in code


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("kind", "config", "declined_config"),
    [
        (
            "helion_flash_gated",
            helion.Config(block_sizes=[128, 128], cute_flash_kv_stage=2),
            helion.Config(block_sizes=[128, 256]),
        ),
        (
            "helion_flash_gated_row_mma",
            helion.Config(
                block_sizes=[32, 32],
                cute_flash_kv_stage=2,
                cute_flash_gated_row_warps=4,
            ),
            helion.Config(block_sizes=[32, 256]),
        ),
    ],
    ids=["tcgen05", "row_mma"],
)
def test_a_sequence_per_lane_gets_the_programs_zeros(
    kind: str, config: helion.Config, declined_config: helion.Config
) -> None:
    """Head 0 holds a 173-row sequence, head 1 a 256-row one.  The rows of
    head 0 from 173 to the end of its last query tile get the program's
    zeros, as at a config the bodies decline with the same query tile."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family gated attention required")
    torch.manual_seed(5)
    # One sequence per head: as many heads as sequences.
    q, k, v, seq_offsets = _jagged_inputs([173, 256], 2, 64, padding=128)
    args = (256, 0.05, q, k, v, seq_offsets)
    _lane_sequences.reset()
    code, fused = compile_and_run(_lane_sequences, args, config)
    assert f"'kind': '{kind}'" in code
    code, declined = compile_and_run(_lane_sequences, args, declined_config)
    assert "'kind': 'helion_flash_gated" not in code
    q_tile = config.config["block_sizes"][0]
    tiles_end = -(-173 // q_tile) * q_tile
    assert (fused[173:tiles_end, 0] == 0).all()
    assert torch.isnan(fused[tiles_end:, 0]).all()
    tol = 2e-2 * declined.nan_to_num().abs().max().item()
    torch.testing.assert_close(fused, declined, atol=tol, rtol=2e-2, equal_nan=True)


def test_gate_extrema_render_the_pointwise_nan_forms() -> None:
    """The gate's relu, maximum and minimum render the pointwise lowering's
    NaN-propagating ``cute.math.max`` / ``min`` in the fused body, as on the
    scalar path of a config the body declines."""
    with cpu_target():
        _extrema_gate_kernel.reset()
        bound = _extrema_gate_kernel.bind(_hstu_args())
        fused = bound.to_code(
            helion.Config(block_sizes=[128, 128], cute_flash_kv_stage=2)
        )
        # A tile neither fused body takes.
        scalar = bound.to_code(helion.Config(block_sizes=[64, 16]))
    assert _gated_markers(fused)
    assert not _gated_markers(scalar)
    # relu (``maximum(x, 0)``) and maximum; minimum.
    for code in (fused, scalar):
        assert (
            len(re.findall(r"cute\.math\.max\([^\n]*propagate_nan=True\)", code)) == 2
        )
        assert (
            len(re.findall(r"cute\.math\.min\([^\n]*propagate_nan=True\)", code)) == 1
        )


def test_fast_math_setting_selects_approximate_gate_math() -> None:
    """The existing fast_math setting (never a config knob) switches the gate's
    division and exp to the approximate forms; without it the gate stays IEEE
    and its exp the exactly reduced ``cute.math.exp``.  Either way the exp is
    the one a config the fused body declines (the scalar path) computes.  The
    shipped example enables the setting; its exact twin is built here."""
    mod = _hstu_module()
    fast_kernel = mod._helion_jagged_attention_kernel
    assert fast_kernel.settings.fast_math
    exact_kernel = _with_fast_math(fast_kernel, False)
    config = helion.Config(block_sizes=[128, 128], cute_flash_kv_stage=2)
    # A tile neither fused body takes.
    scalar_config = helion.Config(block_sizes=[64, 16])
    args = _hstu_args()
    with _cpu_codegen_patches():
        exact_kernel.reset()
        exact_bound = exact_kernel.bind(args)
        exact = exact_bound.to_code(config)
        exact_scalar = exact_bound.to_code(scalar_config)
        fast_bound = fast_kernel.bind(args)
        fast = fast_bound.to_code(config)
        fast_scalar = fast_bound.to_code(scalar_config)
    assert _gated_markers(exact) and _gated_markers(fast)
    assert not _gated_markers(exact_scalar) and not _gated_markers(fast_scalar)
    assert "approx=True" not in exact
    # The ``1 + exp(v)`` site, in the fused body as on the scalar path: the
    # exactly reduced exp without the setting, ``ex2.approx`` of the rounded
    # ``v * log2(e)`` with it (the gate's only ``fastmath=True``).
    gate, scalar = r"flash_g\d+", r"cutlass\.Float32\(v_\d+\)"
    exact_exp = r"= cute\.math\.exp\({}\)$"
    fast_exp = r"= cute\.math\.exp2\({} \* 1\.4426950408889634, fastmath=True\)$"
    for pattern, code in (
        (exact_exp.format(gate), exact),
        (exact_exp.format(scalar), exact_scalar),
        (fast_exp.format(gate), fast),
        (fast_exp.format(scalar), fast_scalar),
    ):
        assert re.search(pattern, code, re.MULTILINE), pattern
    assert "cute.math.exp2(" not in exact
    assert "fastmath=True" not in exact
    assert "cute.math.exp(" not in fast
    assert fast.count("fastmath=True") == 1
    assert _IEEE_GATE_DIVISION.search(exact)
    assert not _APPROX_GATE_DIVISION.search(exact)
    assert _APPROX_GATE_DIVISION.search(fast)
    assert not _IEEE_GATE_DIVISION.search(fast)


def test_sigmoid_gate_computes_the_pointwise_sigmoid() -> None:
    """A ``torch.sigmoid`` gate renders the pointwise lowering's sigmoid
    (``SIGMOID_TEMPLATE``: ``rcp.approx.ftz`` of ``ex2.approx.ftz``) in the
    fused body, of a score vector and of a scalar, as on the scalar path of a
    config the body declines, with and without fast_math: tuned configs must
    not change the gate's numerics."""
    # The template as the code printer spells it, around a placeholder operand.
    contract = re.escape(
        ast.unparse(ast.parse(SIGMOID_TEMPLATE.format(x="operand"), mode="eval"))
    )
    gate = contract.replace("operand", r"flash_g\d+")
    scalar = contract.replace("operand", r"cutlass\.Float32\(\w+\)")
    config = helion.Config(block_sizes=[128, 128], cute_flash_kv_stage=2)
    # A tile neither fused body takes.
    scalar_config = helion.Config(block_sizes=[64, 16])
    args = _hstu_args()
    for fast_math in (False, True):
        kernel = _with_fast_math(_sigmoid_gate_kernel, fast_math)
        with cpu_target():
            kernel.reset()
            bound = kernel.bind(args)
            fused_code = bound.to_code(config)
            scalar_code = bound.to_code(scalar_config)
        assert _gated_markers(fused_code)
        assert not _gated_markers(scalar_code)
        # The per-sequence weight: a scalar, fp32 like its operand.
        assert re.search(rf"^ *flash_g\d+ = {gate}$", fused_code, re.MULTILINE)
        assert re.search(
            rf"= _helion_flash_rt\.gated_round_vec\({gate}, "
            r"cutlass\.BFloat16\)$",
            fused_code,
            re.MULTILINE,
        )
        assert re.search(
            rf"= cutlass\.BFloat16\({scalar}\)$", scalar_code, re.MULTILINE
        )


def test_a_wider_tensor_operand_is_rounded_to_the_common_dtype() -> None:
    """Eager and the pointwise lowering round a 0-d fp32 tensor beside bf16
    scores to bf16 before a compare or a product (the op's common dtype is
    bf16), and a compare's literal too (``qk < 4.1`` tests against
    ``bf16(4.1) = 4.09375``); the fused body does the same, rounding the
    scale once, where it is computed."""
    with _cpu_codegen_patches():
        _sequence_scale_gate_kernel.reset()
        bound = _sequence_scale_gate_kernel.bind(_hstu_args())
        code = bound.to_code(
            helion.Config(block_sizes=[128, 128], cute_flash_kv_stage=2)
        )
    assert _gated_markers(code)
    rounded = re.search(
        r"^ *(flash_g\d+) = cutlass\.Float32\(cutlass\.BFloat16\((flash_g\d+)\)\)$",
        code,
        re.MULTILINE,
    )
    assert rounded
    name, scale = rounded.groups()
    operand = rf"cute\.full\(\(32,\), {name}, cutlass\.Float32\)"
    assert re.search(rf"= flash_g\d+ > {operand}$", code, re.MULTILINE)
    assert re.search(
        rf"gated_round_vec\(flash_g\d+ \* {operand}, cutlass\.BFloat16\)$",
        code,
        re.MULTILINE,
    )
    assert f"cute.full((32,), {scale}," not in code
    assert re.search(
        r"= flash_g\d+ < cute\.full\(\(32,\), cutlass\.Float32\(4\.09375\), "
        r"cutlass\.Float32\)$",
        code,
        re.MULTILINE,
    )


def test_a_remainder_rounds_its_scalar_to_the_common_dtype() -> None:
    """Eager remainder and fmod convert a Python scalar to the tensor's dtype
    first: bf16 scores remainder 0.3 divide by bf16(0.3) = 0.30078125, and so
    does the fused body."""
    with _cpu_codegen_patches():
        _remainder_gate_kernel.reset()
        bound = _remainder_gate_kernel.bind(_hstu_args())
        code = bound.to_code(
            helion.Config(block_sizes=[128, 128], cute_flash_kv_stage=2)
        )
    assert _gated_markers(code)
    assert re.search(
        r"= flash_g\d+ % cute\.full\(\(32,\), cutlass\.Float32\(0\.30078125\), ",
        code,
    )
    assert "cutlass.Float32(0.3)" not in code


def test_non_finite_gate_constants_are_float_calls() -> None:
    """``repr`` spells an infinity as a bare name the generated module does
    not define: a where's ``-inf`` branch, a ``-inf`` fill and an fp16 fill
    rounded to inf render as ``float('-inf')`` and ``float('inf')``."""
    with _cpu_codegen_patches():
        _non_finite_bound_gate_kernel.reset()
        bound = _non_finite_bound_gate_kernel.bind(_hstu_args())
        code = bound.to_code(
            helion.Config(block_sizes=[128, 128], cute_flash_kv_stage=2)
        )
    assert _gated_markers(code)
    assert "cutlass.Float32(float('-inf'))" in code
    assert "cutlass.Float32(float('inf'))" in code
    names = {
        node.id for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Name)
    }
    assert not names & {"inf", "nan"}


def test_an_index_rounded_in_two_sections_is_bound_in_each() -> None:
    """The row index, an expression rather than a bound name, is rounded to
    bf16 per element first and per row second: each section binds its own
    rounding, since the row section runs first and cannot read the per-element
    one.  No gate name is read above its binding."""
    with cpu_target():
        _row_index_gate_kernel.reset()
        bound = _row_index_gate_kernel.bind(_hstu_args())
        code = bound.to_code(
            helion.Config(block_sizes=[128, 128], cute_flash_kv_stage=2)
        )
    assert _gated_markers(code)
    rounding = r"^ *(flash_g\d+) = cutlass\.Float32\(cutlass\.BFloat16\("
    roundings = re.findall(
        rounding + r"cutlass\.Float32\(flash_gate_row\)\)\)$", code, re.MULTILINE
    )
    assert len(roundings) == 2
    bound_at: dict[str, int] = {}
    for number, line in enumerate(code.splitlines()):
        binding = re.match(r" *(flash_g\d+) = ", line)
        for name in re.findall(
            r"\bflash_g\d+\b", line[binding.end() :] if binding else line
        ):
            assert name in bound_at, (name, line)
        if binding:
            bound_at.setdefault(binding.group(1), number)


def test_fast_math_gate_divisions_follow_the_pointwise_lowering() -> None:
    """Under fast_math the pointwise lowering divides fp32 with div.approx.ftz
    (``rcp.approx.ftz`` and a flush-to-zero product) and other dtypes exactly;
    the fused body and the register-MMA row program do the same, so the
    configs they decline compute the same gate."""
    kernel = _with_fast_math(_division_gate_kernel, True)
    with cpu_target():
        kernel.reset()
        bound = kernel.bind(_hstu_args())
        fused = bound.to_code(
            helion.Config(block_sizes=[128, 128], cute_flash_kv_stage=2)
        )
        row = bound.to_code(helion.Config(block_sizes=[16, 64]))
        # A tile neither fused body takes.
        scalar = bound.to_code(helion.Config(block_sizes=[64, 16]))
    assert _gated_markers(fused)
    assert "'kind': 'helion_flash_gated_row_mma'" in row
    assert not _gated_markers(scalar)
    # The fp32 ``1 / seq_len``, ``seq_len / max_seq_len``, F.silu's
    # ``x / (1 + exp(-x))`` and its quotient by ``seq_len``: div.approx.ftz on
    # the scalar path.
    scalar_divisions = re.findall(
        r"= cute\.math\.div\(v_\d+, v_\d+, approx=True, ftz=True\)$",
        scalar,
        re.MULTILINE,
    )
    assert len(scalar_divisions) == 4
    assert re.search(r"= qk / \w+$", scalar, re.MULTILINE)
    rcp = r"cute\.math\.rcp\({}, approx=True, ftz=True\)"
    scalar_rcp = rcp.format(r"cutlass\.Float32\(flash_g\d+\)")
    # ``gated_mul_ftz_vec``: the flush-to-zero product div.approx.ftz takes.
    vector_division = r"= _helion_flash_rt\.gated_mul_ftz_vec\(flash_g\d+, {}\)$"
    for code, width in ((fused, 32), (row, 8)):
        # The bf16 quotient ``qk / (1 + exp(-qk))``: IEEE, as on the scalar
        # path.
        assert re.search(
            r"_helion_flash_rt\.gated_round_vec\(flash_g\d+ / flash_g\d+, "
            r"cutlass\.BFloat16\)$",
            code,
            re.MULTILINE,
        )
        # The scalar fp32 quotients: the same division, and a flush-to-zero
        # approximate reciprocal.
        assert re.search(rf"= {scalar_rcp}$", code, re.MULTILINE)
        assert re.search(
            r"= cute\.math\.div\(cutlass\.Float32\(flash_g\d+\), "
            r"cutlass\.Float32\(flash_g\d+\), approx=True, ftz=True\)$",
            code,
            re.MULTILINE,
        )
        # The vector ones: the divisor's reciprocal (a vector, or a scalar's
        # broadcast) times the dividend, in the flush-to-zero multiply of
        # div.approx.ftz.
        assert re.search(
            vector_division.format(rcp.format(r"flash_g\d+")), code, re.MULTILINE
        )
        broadcast = rf"cute\.full\(\({width},\), {scalar_rcp}, cutlass\.Float32\)"
        assert re.search(vector_division.format(broadcast), code, re.MULTILINE)
        assert "approx=True)" not in code


@pytest.mark.parametrize(
    ("kind", "config", "store_ok", "count"), _BODIES, ids=["tcgen05", "row_mma"]
)
def test_only_an_unmasked_store_writes_rows_past_the_last_sequence(
    kind: str, config: helion.Config, store_ok: str, count: int
) -> None:
    """A store without a row mask writes the rows from the last sequence's end
    on (the program's zeros): its condition compares the row with ``off[n]``,
    the offsets' entry 8 for eight batches.  A store masked by the length
    writes none of them, as in the program."""
    line = re.compile(rf"^ *{store_ok} = (.*)$", re.MULTILINE)
    for masked in (False, True):
        with cpu_target():
            _poisoned_output_kernel.reset()
            bound = _poisoned_output_kernel.bind((*_root_args(), masked))
            code = bound.to_code(config)
        assert f"'kind': '{kind}'" in code
        stores = line.findall(code)
        assert len(stores) == count
        for store in stores:
            rows_end = re.search(r"\| \(flash_o_row >= (flash_g\d+)\)\)", store)
            assert (rows_end is None) == masked, store
            if rows_end is not None:
                last_offset = r"\(seq_offsets\.iterator \+ cutlass\.Int64\(8\) \*"
                load = re.compile(
                    rf"^ *{rows_end[1]}_raw = {last_offset}", re.MULTILINE
                )
                assert load.search(code), store
