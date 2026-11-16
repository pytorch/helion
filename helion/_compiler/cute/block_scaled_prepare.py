"""Generate exact packed preparation from compiler-proved integer recipes.

The workspace is owned by the generated host function, not this factory. Every
launch rewrites all packed bytes and scale bytes, including physical K padding.
This is required for negative scales, NaNs, graph replay and pointer replacement.
"""

from __future__ import annotations

import hashlib
import importlib
import linecache
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from .block_scaled_config import block_scaled_workspace_shape

if TYPE_CHECKING:
    from collections.abc import Mapping


# A row-contiguous side (``*_row_fast``: the source's unit stride is the
# output row) transposes through shared memory: a block gathers a tile of
# TILE_ROWS rows x TILE_GROUPS groups with row-contiguous byte loads, then
# writes the qwords with consecutive threads on consecutive groups (128-byte
# runs per row) and the scales as aligned four-byte packs.  The one-thread-
# per-(row, group) form kept each row's qwords a kilobyte apart (one 32-byte
# sector per 8-byte store) and scattered single scale bytes: 5x L2 write
# amplification and a long-scoreboard-bound kernel.
# 32 x 16 keeps a block's gather at sixteen independent byte loads per thread
# (one warp reads one 32-byte sector of a k-byte row) and leaves the small
# problems enough blocks; 128-row tiles ran latency-bound on 16-64 blocks.
TILE_ROWS = 32
TILE_GROUPS = 16
_PREPARE_THREADS = 256
# ``cute_scaled_prepare_rows``: rows of a tile.  Sixteen-row tiles double the
# block count; at 1024x2048x1024 (RHS 1024 x 128 groups: 512 blocks instead of
# 256) that is 6.63 -> 6.39 us end to end, neutral at the smaller shapes.
PREPARE_ROWS_CHOICES: tuple[int, ...] = (32, 16)


class _TileGeometry:
    """Derived sizes of a ``rows`` x TILE_GROUPS tile."""

    def __init__(self, rows: int) -> None:
        assert rows in PREPARE_ROWS_CHOICES
        self.rows = rows
        # Shared-memory pitch of one k-byte row: rows + 1 keeps the sixteen
        # threads reading sixteen different k-byte rows of one output row off
        # a common bank.
        self.pitch = rows + 1
        self.scales = TILE_GROUPS * 8 * self.pitch
        self.smem = self.scales + rows * TILE_GROUPS
        # Gather: a thread owns one row and TILE_GROUPS // gather_slices groups.
        self.gather_slices = _PREPARE_THREADS // rows
        assert TILE_GROUPS % self.gather_slices == 0
        # Scatter: consecutive threads on consecutive groups, rows * TILE_GROUPS
        # qwords per block; four scale bytes per 32-bit store.
        self.scatter_passes = rows * TILE_GROUPS // _PREPARE_THREADS
        self.scale_stores = rows * TILE_GROUPS // 4
        assert rows * TILE_GROUPS % _PREPARE_THREADS == 0


_BYTE_REPLICATE = 0x0101010101010101


def _scale_position(padded_groups: int) -> str:
    return (
        f"(row // 128 * {padded_groups // 4} + group // 4) * 512"
        " + (row % 32) * 16 + ((row % 128) // 32) * 4 + group % 4"
    )


def preparation_source(plan: Mapping[str, object]) -> str:
    """Emit the two-operand byte permutation without any floating arithmetic."""
    m, n, k, bn, bk = (cast("int", plan[key]) for key in ("m", "n", "k", "bn", "bk"))
    mp, np, kp = block_scaled_workspace_shape(m, n, k, bn, bk)
    groups, padded_groups = k // 16, kp // 16
    tile = _TileGeometry(int(cast("int", plan.get("prepare_rows", TILE_ROWS))))
    if plan.get("fused") is not None:
        # No preparation kernel: the consumer reads the sources directly.
        return (
            "@cute.jit\n"
            "def _entry(a: cute.Tensor, b: cute.Tensor, sa: cute.Tensor, sb: cute.Tensor, "
            "workspace: cute.Tensor, output: cute.Tensor, alpha: cutlass.Float32, stream: CUstream):\n"
            "    _consumer.fused_entry(a, b, sa, sb, output, alpha, stream)\n"
        )
    qa_bytes, qb_bytes = mp * padded_groups * 8, np * padded_groups * 8
    sa_bytes = mp * padded_groups
    sides = (
        ("lhs", "a", "sa", m, mp, 0, qa_bytes + qb_bytes),
        ("rhs", "b", "sb", n, np, qa_bytes, qa_bytes + qb_bytes + sa_bytes),
    )
    # Block ranges: tiled sides take TILE_ROWS x TILE_GROUPS tiles, the
    # others one thread per (row, group) with consecutive threads on
    # consecutive groups (their qwords are already contiguous).
    layout: list[tuple[str, int, int]] = []
    first_block = 0
    for side, _source, _scale, _rows, padded_rows, _q, _sf in sides:
        if plan[side + "_row_fast"] and plan.get("prepare", "tiled") == "tiled":
            assert padded_rows % tile.rows == 0
            count = (padded_rows // tile.rows) * (
                (padded_groups + TILE_GROUPS - 1) // TILE_GROUPS
            )
            mode = "tiled"
        else:
            count = (padded_rows * padded_groups + _PREPARE_THREADS - 1) // (
                _PREPARE_THREADS
            )
            mode = "linear"
        layout.append((mode, first_block, count))
        first_block += count
    total_blocks = first_block
    workspace_bytes = cast("int", plan["workspace_bytes"])
    lines = [
        "@cute.kernel",
        "def _prepare(a: cute.Tensor, b: cute.Tensor, sa: cute.Tensor, sb: cute.Tensor, workspace: cute.Tensor):",
        "    bid = cutlass.Int32(cute.arch.block_idx()[0])",
        "    tid = cutlass.Int32(cute.arch.thread_idx()[0])",
        # Let the consumer's prologue start now; its TMA producer waits for
        # this kernel's completion before reading the workspace.
        *(
            ("    cute.arch.griddepcontrol_launch_dependents()",)
            if plan.get("pdl")
            else ()
        ),
        f"    words = cute.make_tensor(cute.recast_ptr(workspace.iterator, dtype=cutlass.Uint64), cute.make_layout(({workspace_bytes // 8},)))",
        f"    sf_words = cute.make_tensor(cute.recast_ptr(workspace.iterator, dtype=cutlass.Uint32), cute.make_layout(({workspace_bytes // 4},)))",
    ]
    for side, source, scale, *_rest in sides:
        lines.extend(
            (
                f"    {source}_bytes = cute.make_tensor(cute.recast_ptr({source}.iterator, dtype=cutlass.Uint8), cute.make_layout(({plan[side + '_span']},)))",
                f"    {scale}_bytes = cute.make_tensor(cute.recast_ptr({scale}.iterator, dtype=cutlass.Uint8), cute.make_layout(({plan[side + '_scale_span']},)))",
            )
        )
        if plan.get(side + "_qword_address") is not None:
            lines.append(
                f"    {source}_words = cute.make_tensor(cute.recast_ptr({source}.iterator, dtype=cutlass.Uint64), cute.make_layout(({cast('int', plan[side + '_span']) // 8},)))"
            )
    scale_position = _scale_position(padded_groups)
    for (side, source, scale, rows, padded_rows, q_offset, sf_offset), (
        mode,
        first,
        count,
    ) in zip(sides, layout, strict=True):
        if count == 0:
            continue
        addresses = cast("tuple[str, ...]", plan[side + "_byte_addresses"])
        assert len(addresses) == 8
        scale_address = cast("str", plan[side + "_scale_address"])
        qword_address = cast("str | None", plan.get(side + "_qword_address"))
        lines.append(f"    {'if' if first == 0 else 'elif'} bid < {first + count}:")
        if mode == "linear":
            lines.extend(
                (
                    f"        linear = cutlass.Int64(bid - {first}) * {_PREPARE_THREADS} + cutlass.Int64(tid)",
                    f"        if linear < {padded_rows * padded_groups}:",
                )
            )
            # Coalesce the source's contiguous axis across consecutive threads.
            if plan[side + "_row_fast"]:
                lines.extend(
                    (
                        f"            row = linear % {padded_rows}",
                        f"            group = linear // {padded_rows}",
                    )
                )
            else:
                lines.extend(
                    (
                        f"            row = linear // {padded_groups}",
                        f"            group = linear % {padded_groups}",
                    )
                )
            lines.extend(
                (
                    "            word = cutlass.Uint64(0)",
                    "            sf = cutlass.Uint8(0)",
                    f"            if row < {rows} and group < {groups}:",
                    f"                signed_sf = {scale}_bytes[{scale_address}]",
                    "                sf = signed_sf & cutlass.Uint8(127)",
                    "                sign = cutlass.Uint8(((cutlass.Uint32(signed_sf) >> 7) & 1) * 136)",
                )
            )
            if qword_address is not None:
                lines.append(
                    f"                word = cutlass.Uint64({source}_words[cutlass.Int64({qword_address}) // 8]) ^ (cutlass.Uint64(sign) * cutlass.Uint64({_BYTE_REPLICATE}))"
                )
            else:
                for byte, address in enumerate(addresses):
                    lines.extend(
                        (
                            f"                q{byte} = {source}_bytes[{address}] ^ sign",
                            f"                word = word | ((cutlass.Uint64(q{byte}) & 255) << {8 * byte})",
                        )
                    )
            lines.extend(
                (
                    f"            words[{q_offset // 8} + row * {padded_groups} + group] = word",
                    f"            sf_position = {sf_offset} + {scale_position}",
                    "            workspace[sf_position] = sf",
                )
            )
            continue
        row_tiles = padded_rows // tile.rows
        lines.extend(
            (
                f"        tile = bid - {first}",
                f"        row0 = (tile % {row_tiles}) * {tile.rows}",
                f"        group0 = (tile // {row_tiles}) * {TILE_GROUPS}",
                f"        smem = cute.make_tensor(cute.arch.alloc_smem(cutlass.Uint8, {tile.smem}), cute.make_layout(({tile.smem},)))",
                # Stage 2's scale bytes do not depend on the tile: issue those
                # loads first so their latency overlaps the gather.
                f"        j = tid % {TILE_GROUPS}",
                f"        row_base = tid // {TILE_GROUPS}",
                "        group = group0 + j",
            )
        )
        for p in range(tile.scatter_passes):
            lines.extend(
                (
                    f"        row = row0 + row_base + {_PREPARE_THREADS // TILE_GROUPS * p}",
                    f"        signed_sf_{p} = cutlass.Uint8(0)",
                    f"        if row < {rows} and group < {groups}:",
                    # The DSL reads a Uint8 buffer back as Int8: convert so
                    # both paths of the guard agree on the type.
                    f"            signed_sf_{p} = cutlass.Uint8(cutlass.Uint32({scale}_bytes[{scale_address}]) & 255)",
                )
            )
        lines.extend(
            (
                # Stage 1: row-contiguous byte loads into the k-byte-major tile.
                f"        row_local = tid % {tile.rows}",
                f"        slice = tid // {tile.rows}",
                "        row = row0 + row_local",
                f"        for gi in cutlass.range_constexpr({TILE_GROUPS // tile.gather_slices}):",
                f"            group_local = slice + {tile.gather_slices} * gi",
                "            group = group0 + group_local",
                f"            if row < {rows} and group < {groups}:",
            )
        )
        for byte, address in enumerate(addresses):
            lines.append(
                f"                smem[(group_local * 8 + {byte}) * {tile.pitch} + row_local] = cutlass.Uint8(cutlass.Uint32({source}_bytes[{address}]) & 255)"
            )
        lines.append("            else:")
        for byte in range(8):
            lines.append(
                f"                smem[(group_local * 8 + {byte}) * {tile.pitch} + row_local] = cutlass.Uint8(0)"
            )
        lines.extend(
            (
                "        cute.arch.sync_threads()",
                # Stage 2: consecutive threads write consecutive qwords of a row.
                "        group = group0 + j",
            )
        )
        for p in range(tile.scatter_passes):
            lines.extend(
                (
                    f"        row_local = row_base + {_PREPARE_THREADS // TILE_GROUPS * p}",
                    "        row = row0 + row_local",
                    "        word = cutlass.Uint64(0)",
                    "        sf = cutlass.Uint8(0)",
                    f"        if row < {rows} and group < {groups}:",
                    f"            sf = signed_sf_{p} & cutlass.Uint8(127)",
                    f"            sign = ((cutlass.Uint32(signed_sf_{p}) >> 7) & 1) * 136",
                )
            )
            for byte in range(8):
                lines.append(
                    f"            word = word | (cutlass.Uint64((cutlass.Uint32(smem[(j * 8 + {byte}) * {tile.pitch} + row_local]) & 255) ^ sign) << {8 * byte})"
                )
            lines.extend(
                (
                    f"        if group < {padded_groups}:",
                    f"            words[{q_offset // 8} + row * {padded_groups} + group] = word",
                    f"        smem[{tile.scales} + row_local * {TILE_GROUPS} + j] = sf",
                )
            )
        lines.extend(
            (
                "        cute.arch.sync_threads()",
                # Stage 3: four scale bytes of a row share one aligned word.
                f"        for p in cutlass.range_constexpr({(tile.scale_stores + _PREPARE_THREADS - 1) // _PREPARE_THREADS}):",
                f"            index = tid + {_PREPARE_THREADS} * p",
                "            row_local = index // 4",
                "            quad = index % 4",
                "            row = row0 + row_local",
                "            group = group0 + quad * 4",
                f"            if index < {tile.scale_stores} and group < {padded_groups}:",
                f"                packed = cutlass.Uint32(smem[{tile.scales} + row_local * {TILE_GROUPS} + quad * 4]) & 255",
            )
        )
        for byte in range(1, 4):
            lines.append(
                f"                packed = packed | ((cutlass.Uint32(smem[{tile.scales} + row_local * {TILE_GROUPS} + quad * 4 + {byte}]) & 255) << {8 * byte})"
            )
        lines.extend(
            (
                f"                sf_words[({sf_offset} + {scale_position}) // 4] = packed",
            )
        )
    lines.extend(
        (
            "",
            "@cute.jit",
            "def _entry(a: cute.Tensor, b: cute.Tensor, sa: cute.Tensor, sb: cute.Tensor, workspace: cute.Tensor, output: cute.Tensor, alpha: cutlass.Float32, stream: CUstream):",
            f"    _prepare(a, b, sa, sb, workspace).launch(grid=({total_blocks}, 1, 1), block=({_PREPARE_THREADS}, 1, 1), stream=stream)",
            "    _consumer(workspace, output, alpha, stream)",
        )
    )
    return "\n".join(lines) + "\n"


def make_block_scaled_entry(plan: Mapping[str, object]) -> object:
    if plan.get("format") == "mxfp8":
        from .mxfp8_prepare import make_mxfp8_entry

        return make_mxfp8_entry(plan)
    import cutlass
    import cutlass.cute as cute

    from .block_scaled_runtime import Sm100BlockScaledGemm

    source = preparation_source(plan)
    filename = f"<helion_block_scaled:{hashlib.sha256(source.encode()).hexdigest()}>"
    namespace: dict[str, Any] = {
        "cutlass": cutlass,
        "cute": cute,
        "CUstream": importlib.import_module("cuda.bindings.driver").CUstream,
        "_consumer": Sm100BlockScaledGemm(
            cast("int", plan["m"]),
            cast("int", plan["n"]),
            cast("int", plan["k"]),
            cast("int", plan["bn"]),
            cast("int", plan["bk"]),
            cast("int", plan["stages"]),
            cast("int", plan["cluster_m"]),
            cast("bool", plan["persistent"]),
            cast("int", plan["num_sm"]),
            tuple(
                {
                    "torch.float16": cutlass.Float16,
                    "torch.bfloat16": cutlass.BFloat16,
                    "torch.float32": cutlass.Float32,
                }[name]
                for name in cast("tuple[str, ...]", plan["round_dtypes"])
            ),
            pdl=bool(plan.get("pdl", False)),
            fused=cast("dict[str, tuple[str, int]] | None", plan.get("fused")),
            epilogue_n=int(cast("int", plan.get("epilogue_n", 64))),
        ),
    }
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    try:
        exec(compile(source, filename, "exec"), namespace)
    except BaseException:
        linecache.cache.pop(filename, None)
        raise
    return namespace["_entry"]
