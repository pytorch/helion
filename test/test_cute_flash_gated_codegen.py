"""Which gated-attention kernels the fused bodies of ``cute_flash_gated`` take,
and the store each one renders (on CPU; the GPU tests are marked ``cuda``)."""

from __future__ import annotations

import re

import pytest
import torch

import helion
from helion._compiler.cute import cute_flash_gated
from helion._testing import DEVICE
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_flash import compile_and_run
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


def _root_args(
    heads: int = 8, total_rows: int = 3044, device: str = "cpu"
) -> tuple[object, ...]:
    """Arguments of the root-form kernels: 512 query rows, 64-wide heads."""
    q = torch.empty((total_rows, heads, 64), dtype=torch.bfloat16, device=device)
    offsets = torch.zeros((heads + 1,), dtype=torch.int64, device=device)
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
        bound = kernel.bind(args)
        env, host_function = bound.env, bound.host_function
        assert env is not None and host_function is not None
        with env, host_function:
            return cute_flash_gated.match_gated_attention(host_function.device_ir)


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


def _jagged_inputs(
    lengths: list[int], dim: int, *, padding: int
) -> tuple[torch.Tensor, ...]:
    """One sequence per head, ``padding`` rows after the last."""
    total = sum(lengths) + padding
    q, k, v = (
        torch.randn(total, len(lengths), dim, device=DEVICE, dtype=torch.bfloat16)
        for _ in range(3)
    )
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return q, k, v, torch.tensor(offsets, device=DEVICE, dtype=torch.int64)


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
    q, k, v, seq_offsets = _jagged_inputs([173, 256], 64, padding=128)
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
