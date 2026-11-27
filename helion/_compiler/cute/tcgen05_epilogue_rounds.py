"""The TMA-store epilogue in rounds of several subtiles.

``tcgen05_epi_round_subtiles`` (``K``) groups the subtiles of the role-local
TMA-store epilogue: each round loads ``K`` subtiles from TMEM back to back,
converts and stages them into ``K`` distinct buffers of the C ring, then pays
ONE async-proxy fence, ONE named barrier and one bulk commit for the ``K``
TMA stores.  The per-subtile loop (``K = 1``) pays a TMEM-load wait, the
fence, two barriers and a bulk-group wait for every subtile; with one output
tile per CTA that chain is exposed at the end of the kernel.  nvjet stores
its 128x80 bf16 tile in three rounds (32, 32, 16 columns); the grouped
row-union kernel of the same geometry (five 128x16 subtiles) measured 3.25 us
per call in the catalogue's back-to-back graph with five rounds, 3.10 with
``K = 2`` (rounds of 2, 2 and 1 subtiles), 3.26 with a single round (the one
bulk store drains unoverlapped).

The C ring keeps ``c_stages`` subtile buffers (the SMEM layout, the kernel
descriptor and the wrapper's TMA store box are unchanged); the bulk-group
pipeline counts rounds, so it has ``c_stages // K`` stages.  Subtile ``u`` of
the CTA's sequence (``tile * subtiles_per_tile + s``) stages in buffer ``u %
c_stages``; a buffer is reused ``c_stages`` subtiles later, which is at
least ``c_stages // K`` rounds later, and the round's acquire
(``cp.async.bulk.wait_group.read(c_stages // K - 1)``) proves the group that
read it complete.  The acquire and the barrier that publishes it are emitted
only where a reuse is possible: unconditionally for the rounds past the first
``c_stages // K`` of a tile, under a ``tile_counter`` guard for the earlier
ones, so a CTA's first tile pays none of them.

``render_last_tile_staging`` renders the same plain store's CTA's last tile
in one round of all its subtiles, staged in the A ring that tile's MMAs have
drained (``cute_mma``'s ``tcgen05_last_tile_staging``).
"""

from __future__ import annotations

import dataclasses
import math
import textwrap


@dataclasses.dataclass(frozen=True)
class Tcgen05EpilogueRoundNames:
    """Generated-source names the rounds renderer composes."""

    epi_active: str
    epi_leader_predicate: str
    acc_pipeline: str
    acc_consumer_state: str
    tiled_copy_t2r: str
    tiled_copy_r2s: str
    ttr_tacc: str
    ttr_tacc_mn: str
    ttr_racc: str
    trs_racc: str
    ttr_rd: str
    trs_rd: str
    trs_sd: str
    acc_vec: str
    c_buffer: str
    epilog_sync_barrier: str
    tma_store_atom: str
    bsg_sd: str
    bsg_gd: str
    c_pipeline: str
    target_dtype: str
    tile_counter: str
    tma_desc_arg: str = ""


def round_groups(subtile_count: int, round_subtiles: int) -> list[tuple[int, ...]]:
    """The subtile indices of each round: ``K`` per round, the remainder last."""
    assert subtile_count >= 1 and round_subtiles >= 1
    return [
        tuple(range(start, min(start + round_subtiles, subtile_count)))
        for start in range(0, subtile_count, round_subtiles)
    ]


def slot_name(base: str, slot: int) -> str:
    return base if slot == 0 else f"{base}_r{slot}"


def render_round_subtile_loop(
    names: Tcgen05EpilogueRoundNames,
    *,
    subtile_count: int,
    round_subtiles: int,
    c_stage_count: int,
    acc_rhs: str,
    acc_wait: bool,
    acc_release: bool,
) -> str:
    """Render the per-tile TMA-store epilogue as rounds (see the module doc).

    ``acc_rhs`` is the identity epilogue's register conversion with the
    slot-0 carrier (``names.trs_racc``) in it; every other slot substitutes
    its own carrier.  ``acc_wait`` puts the accumulator ``consumer_wait``
    ahead of the first round, ``acc_release`` the TMEM-load fence and the
    accumulator release after the last subtile's conversion.
    """
    assert names.trs_racc in acc_rhs, (names.trs_racc, acc_rhs)
    assert c_stage_count % round_subtiles == 0 and c_stage_count >= round_subtiles
    round_stages = c_stage_count // round_subtiles
    groups = round_groups(subtile_count, round_subtiles)
    rounds_per_tile = len(groups)
    body: list[str] = []
    for slot in range(1, max(len(group) for group in groups)):
        body.extend(
            [
                f"{slot_name(names.ttr_racc, slot)} = cute.make_rmem_tensor({names.ttr_racc}.shape, cutlass.Float32)",
                f"{slot_name(names.ttr_rd, slot)} = cute.make_rmem_tensor({names.ttr_racc}.shape, {names.target_dtype})",
                f"{slot_name(names.trs_rd, slot)} = {names.tiled_copy_r2s}.retile({slot_name(names.ttr_rd, slot)})",
                f"{slot_name(names.trs_racc, slot)} = {names.tiled_copy_r2s}.retile({slot_name(names.ttr_racc, slot)})",
            ]
        )
    if acc_wait:
        body.append(f"{names.acc_pipeline}.consumer_wait({names.acc_consumer_state})")
    for round_index, group in enumerate(groups):
        if round_index >= round_stages:
            guard = None
        else:
            # Global round ``tile * rounds_per_tile + round_index`` may reuse a
            # buffer once it is at least ``round_stages``.
            guard = (
                f"if {names.tile_counter} >= cutlass.Int32("
                f"{math.ceil((round_stages - round_index) / rounds_per_tile)}):"
            )
        acquire = [
            f"if {names.epi_leader_predicate}:",
            f"    {names.c_pipeline}.producer_acquire()",
            f"{names.epilog_sync_barrier}.arrive_and_wait()",
        ]
        if guard is None:
            body.extend(acquire)
        else:
            body.append(guard)
            body.extend(f"    {line}" for line in acquire)
        for slot, subtile in enumerate(group):
            body.extend(
                [
                    f"{slot_name(names.ttr_tacc_mn, slot)} = {names.ttr_tacc}[(None, None, None, cutlass.Int32({subtile}))]",
                    f"cute.copy({names.tiled_copy_t2r}, {slot_name(names.ttr_tacc_mn, slot)}, {slot_name(names.ttr_racc, slot)})",
                ]
            )
        for slot, subtile in enumerate(group):
            rhs = acc_rhs.replace(names.trs_racc, slot_name(names.trs_racc, slot))
            body.append(f"{slot_name(names.acc_vec, slot)} = {rhs}")
            if acc_release and subtile == subtile_count - 1:
                body.extend(
                    [
                        "cute.arch.fence_view_async_tmem_load()",
                        "with cute.arch.elect_one():",
                        f"    {names.acc_pipeline}.consumer_release({names.acc_consumer_state})",
                    ]
                )
            body.append(
                f"{slot_name(names.trs_rd, slot)}.store({slot_name(names.acc_vec, slot)})"
            )
        for slot, subtile in enumerate(group):
            body.extend(
                [
                    f"{slot_name(names.c_buffer, slot)} = ({names.tile_counter} * cutlass.Int32({subtile_count}) + cutlass.Int32({subtile})) % cutlass.Int32({c_stage_count})",
                    f"cute.copy({names.tiled_copy_r2s}, {slot_name(names.trs_rd, slot)}, {names.trs_sd}[(None, None, None, {slot_name(names.c_buffer, slot)})])",
                ]
            )
        body.extend(
            [
                "cute.arch.fence_view_async_shared()",
                f"{names.epilog_sync_barrier}.arrive_and_wait()",
                f"if {names.epi_leader_predicate}:",
            ]
        )
        for slot, subtile in enumerate(group):
            body.append(
                f"    cute.copy({names.tma_store_atom}, {names.bsg_sd}[(None, {slot_name(names.c_buffer, slot)})], "
                f"{names.bsg_gd}[(None, cutlass.Int32({subtile}))]{names.tma_desc_arg})"
            )
        body.append(f"    {names.c_pipeline}.producer_commit()")
    return f"if {names.epi_active}:\n" + "".join(f"    {line}\n" for line in body)


@dataclasses.dataclass(frozen=True)
class Tcgen05LastTileStagingNames:
    """Generated-source names the last-tile staging renderer composes,
    beside the rounds renderer's (``Tcgen05EpilogueRoundNames``)."""

    rounds: Tcgen05EpilogueRoundNames
    last_tile_next_work: str
    staging_smem: str
    smem_layout: str
    smem: str
    trs_sd: str
    bsg_sd: str
    epi_tidx: str
    tcgc_epi: str
    d_store_layout: str
    epi_tile: str
    subtile_count: str


def render_last_tile_staging(
    names: Tcgen05LastTileStagingNames, *, acc_rhs: str, default_loop: str
) -> str:
    """The plain TMA-store epilogue with its CTA's last tile staged at once.

    On the last tile (no valid ``names.last_tile_next_work``, the work tile
    one step ahead of the role's) every subtile goes from TMEM
    through registers into its own stage of a C tile carved from the A ring
    (``names.staging_smem``), which the tile's MMAs have drained; one fence
    and one barrier publish them, and the leader issues every TMA store
    behind one commit. Every other tile runs ``default_loop``, the C-ring
    loop, whose stores never touch the A ring. ``acc_rhs`` is the identity
    epilogue's conversion of ``names.rounds.trs_racc``.
    """
    r = names.rounds
    assert r.trs_racc in acc_rhs, (r.trs_racc, acc_rhs)
    body = [
        (
            f"{names.smem_layout} = cutlass.utils.blackwell_helpers.make_smem_layout_epi("
            f"{r.target_dtype}, {names.d_store_layout}, {names.epi_tile}, {names.subtile_count})"
        ),
        (
            f"{names.smem} = cute.make_tensor(cute.recast_ptr({names.staging_smem}, "
            f"{names.smem_layout}.inner, dtype={r.target_dtype}), {names.smem_layout}.outer)"
        ),
        # (the C ring's R2S copy: CuTe builds it from the T2R copy and the
        # register tile alone, so only the SMEM partition is new)
        (
            f"{names.trs_sd} = {r.tiled_copy_r2s}.get_slice({names.epi_tidx})"
            f".partition_D({names.smem})"
        ),
        (
            f"{names.bsg_sd}, _ = cute.nvgpu.cpasync.tma_partition("
            f"{r.tma_store_atom}, 0, cute.make_layout(1), cute.group_modes({names.smem}, 0, 2), "
            f"cute.group_modes({names.tcgc_epi}, 0, 2))"
        ),
        f"for _tcgen05_subtile in cutlass.range({names.subtile_count}, unroll_full=True):",
        "    if _tcgen05_subtile == 0:",
        f"        {r.acc_pipeline}.consumer_wait({r.acc_consumer_state})",
        f"    {r.ttr_tacc_mn} = {r.ttr_tacc}[(None, None, None, cutlass.Int32(_tcgen05_subtile))]",
        f"    cute.copy({r.tiled_copy_t2r}, {r.ttr_tacc_mn}, {r.ttr_racc})",
        f"    {r.acc_vec} = {acc_rhs}",
        f"    if _tcgen05_subtile == {names.subtile_count} - 1:",
        "        cute.arch.fence_view_async_tmem_load()",
        "        with cute.arch.elect_one():",
        f"            {r.acc_pipeline}.consumer_release({r.acc_consumer_state})",
        f"    {r.trs_rd}.store({r.acc_vec})",
        (
            f"    cute.copy({r.tiled_copy_r2s}, {r.trs_rd}, "
            f"{names.trs_sd}[(None, None, None, cutlass.Int32(_tcgen05_subtile))])"
        ),
        "cute.arch.fence_view_async_shared()",
        f"{r.epilog_sync_barrier}.arrive_and_wait()",
        f"if {r.epi_leader_predicate}:",
        f"    for _tcgen05_subtile in cutlass.range({names.subtile_count}, unroll_full=True):",
        (
            f"        cute.copy({r.tma_store_atom}, {names.bsg_sd}[(None, cutlass.Int32(_tcgen05_subtile))], "
            f"{r.bsg_gd}[(None, cutlass.Int32(_tcgen05_subtile))]{r.tma_desc_arg})"
        ),
        f"    {r.c_pipeline}.producer_commit()",
    ]
    return (
        f"if not {names.last_tile_next_work}.is_valid_tile:\n"
        + "".join(f"    {line}\n" for line in body)
        + "else:\n"
        + textwrap.indent(default_loop, "    ")
    )
