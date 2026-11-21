"""The decode-step paged attention kernel of the per-key schedule's launch
stress (``test_cute_paged_attention_handoff_stress.py``).

The kernel is the windowed grouped-query decode step of a serving workload:
flat Q and O storage with explicit offsets, forward page order, base-2 scores
relative to the previous maximum and a head-wide lazy rescale. Compiled with
the per-key ``attention`` schedule at two CTAs per SM, it hangs when a
compute warp other than the PV issuer waits for the value-ready barrier and
misses a phase. The other paged fixtures (reverse page order, post-scaled
operands) do not reproduce that in the same stress, so the source is kept in
this form.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

QUERY_HEADS = 16
DIM = 128
WINDOW = 2047
# A 2,048-key window ending four keys short of a page boundary: the first
# page keeps its last seven keys, the fifteen pages between are interior.
PAGES = 17


@helion.kernel(backend="cute", static_shapes=True, fast_math=True)
def paged_decode_attention(
    q_flat: torch.Tensor,
    k5: torch.Tensor,
    v5: torch.Tensor,
    ks5: torch.Tensor,
    vs5: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    out_flat: torch.Tensor,
    lse_flat: torch.Tensor,
    qlen: hl.constexpr,
    window: hl.constexpr,
    with_lse: hl.constexpr,
    rows_block: hl.constexpr,
) -> None:
    # Literal integers only: Helion traces module-level ints as symbols.
    # Rows are (query, grouped head) pairs: row = query * 8 + head_in_group.
    items = lengths.size(0) * 2
    for tile_i, tile_r, tile_d in hl.tile(  # pyrefly: ignore [no-matching-overload]
        [items, qlen * 8, 128], block_size=[1, rows_block, 128]
    ):
        item = tile_i.begin
        req = item // 2
        head = item % 2
        r = tile_r.index
        qrow = r // 8
        hrow = r % 8
        q_off = (
            req * (qlen * 4608)
            + head * 1024
            + qrow[:, None] * 4608
            + hrow[:, None] * 128
            + tile_d.index
        )
        q_tile = q_flat[q_off]
        length = lengths[req]
        qpos = (length - qlen) + qrow  # pyrefly: ignore [unsupported-operation]
        end = (length + 127) // 128
        if window < 0:  # pyrefly: ignore [unsupported-operation]
            begin = 0
        else:
            # pyrefly: ignore [unsupported-operation]
            begin = torch.clamp(length - qlen - window, min=0) // 128
        m_i = hl.full([tile_r], float("-inf"), dtype=torch.float32)
        l_i = hl.zeros([tile_r, 128], dtype=torch.float32)
        acc = hl.zeros([tile_r, tile_d], dtype=torch.float32)
        for it in hl.grid(begin, end):
            page = page_table[req, it]
            k_raw = k5[page, 0, :, head, tile_d]
            k_sc = ks5[page, 0, :, head, tile_d.index // 32]
            k_dq = (
                k_raw.to(torch.float32) * torch.exp2(k_sc.to(torch.float32) - 127)
            ).to(torch.bfloat16)
            qk = hl.dot(q_tile, k_dq.permute(1, 0))
            kpos = hl.arange(128) + it * 128
            visible = kpos[None, :] <= qpos[:, None]
            if window >= 0:  # pyrefly: ignore [unsupported-operation]
                visible = visible & (kpos[None, :] >= qpos[:, None] - window)
            old_ok = (m_i >= 0) & (m_i < float("inf"))
            ref = torch.where(old_ok, m_i, 0.0)
            # Base-2 scores relative to the previous (nonnegative finite) max.
            score = torch.where(
                visible, qk * 0.12751743082459868 - ref[:, None], float("-inf")
            )
            tile_max = torch.amax(score, dim=1)
            proposed = torch.maximum(m_i, tile_max + ref)
            # Keep the stale reference unless some row's maximum grows by more
            # than 4 (log2 units); all rows of the head share one decision.
            grow = (tile_max > 4.0) | ~old_ok | torch.isnan(tile_max)
            head_grow = torch.amax(grow.to(torch.float32), dim=0) > 0.0
            m_new = torch.where(head_grow, proposed, m_i)
            safe = torch.where(m_new == float("-inf"), 0.0, m_new)
            p = torch.exp2(score + (ref - safe)[:, None])
            alpha = torch.exp2(m_i - safe)
            l_i = l_i * alpha[:, None] + p
            v_raw = v5[page, 0, :, head, tile_d]
            v_sc = vs5[page, 0, :, head, tile_d.index // 32]
            v_dq = (
                v_raw.to(torch.float32) * torch.exp2(v_sc.to(torch.float32) - 127)
            ).to(torch.bfloat16)
            acc = hl.dot(p.to(torch.bfloat16), v_dq, acc=acc * alpha[:, None])
            m_i = m_new
        l_sum = torch.sum(l_i, dim=1)
        safe_l = torch.where(l_sum == 0, 1.0, l_sum)
        o_off = (
            req * (qlen * 2048)
            + head * 1024
            + qrow[:, None] * 2048
            + hrow[:, None] * 128
            + tile_d.index
        )
        out_flat[o_off] = (acc * torch.reciprocal(safe_l)[:, None]).to(torch.bfloat16)
        if with_lse:
            lse_off = req * (qlen * 16) + head * 8 + qrow * 16 + hrow
            lse_flat[lse_off] = torch.log(l_sum) + m_i * 0.6931471805599453


def paged_decode_args(
    batch: int, query_len: int, device: torch.device | str
) -> tuple[object, ...]:
    """One decode step per request over ``PAGES`` zero K/V pages.

    Q is a strided view of a wider per-token projection (4,608 elements per
    token); each request owns distinct pages behind a permuted page table.
    """
    pool = batch * PAGES + 1
    storage = [
        torch.zeros(pool * 33792, dtype=torch.int8, device=device) for _ in range(2)
    ]
    k, v = [x.as_strided((pool, 128, 2, 128), (33792, 256, 128, 1)) for x in storage]
    ks, vs = [
        x.view(torch.uint8).as_strided((pool, 128, 2, 4), (33792, 4, 512, 1), 32768)
        for x in storage
    ]
    generator = torch.Generator(device=device).manual_seed(0)
    projection = torch.empty(
        (batch, query_len, 4608), dtype=torch.bfloat16, device=device
    )
    q = projection[..., : QUERY_HEADS * DIM].view(batch, query_len, QUERY_HEADS, DIM)
    q.normal_(generator=generator)
    table = torch.zeros((batch, PAGES), dtype=torch.int32, device=device)
    table.copy_(
        (
            torch.randperm(pool - 1, generator=generator, device=device)[
                : batch * PAGES
            ]
            + 1
        ).view(batch, PAGES)
    )
    lengths = torch.full((batch,), PAGES * 128 - 4, dtype=torch.int32, device=device)
    out = torch.zeros(q.shape, dtype=torch.bfloat16, device=device)
    q_flat = q.as_strided(
        ((batch * query_len - 1) * q.stride(1) + QUERY_HEADS * DIM,), (1,)
    )
    return (
        q_flat,
        k.unsqueeze(1),
        v.unsqueeze(1),
        ks.unsqueeze(1),
        vs.unsqueeze(1),
        table,
        lengths,
        out.view(-1),
        torch.empty((1,), dtype=torch.float32, device=device),
        query_len,
        WINDOW,
        False,
        helion.next_power_of_2(query_len * 8),
    )


def per_key_two_cta_config() -> helion.Config:
    """The per-key ``attention`` schedule at two CTAs per SM."""
    return helion.Config(
        block_sizes=[],
        cute_mma_dag="tcgen05",
        cute_mma_dag_warps=4,
        cute_mma_dag_stages=2,
        cute_mma_dag_pass_warps="attention",
        cute_min_blocks_per_mp=2,
        cute_pdl=False,
    )


class LaunchWatchdog:
    """When a launch loop has stopped making progress.

    ``stall`` bounds the time between two progress reports. The first call
    is not a launch alone: it also JIT-compiles the kernel, which a cold
    compile cache on a loaded machine can stretch well past what a launch
    may stall, and nothing reports the end of the compile. So until the
    first report the bound is ``first_call``; a first launch that hangs is
    still reported, when that longer bound expires.
    """

    def __init__(
        self,
        *,
        first_call: float,
        stall: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._first_call = first_call
        self._stall = stall
        # (time of the last report, launches completed by it; None before
        # the first call returns). One tuple, so the watching thread never
        # pairs one report's time with another's count.
        self._state: tuple[float, int | None] = (clock(), None)

    @property
    def launches(self) -> int:
        """The launches the last progress report counted."""
        return self._state[1] or 0

    def progress(self, launches: int) -> None:
        """``launches`` launches have completed (0: the first call has)."""
        self._state = (self._clock(), launches)

    def expired(self) -> str | None:
        """The report of a hang, or None while the loop is within its bound."""
        since, launches = self._state
        limit = self._first_call if launches is None else self._stall
        if self._clock() - since <= limit:
            return None
        if launches is None:
            return (
                f"HANG: no progress for {limit}s in the first call, which also compiles"
            )
        return f"HANG: no progress for {limit}s after {launches} launches"
