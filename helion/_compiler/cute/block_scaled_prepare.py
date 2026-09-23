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


def preparation_source(plan: Mapping[str, object]) -> str:
    """Emit the two-operand byte permutation without any floating arithmetic."""
    m, n, k, bn, bk = (cast("int", plan[key]) for key in ("m", "n", "k", "bn", "bk"))
    mp, np, kp = block_scaled_workspace_shape(m, n, k, bn, bk)
    groups, padded_groups = k // 16, kp // 16
    qa_bytes, qb_bytes = mp * padded_groups * 8, np * padded_groups * 8
    sa_bytes = mp * padded_groups
    lhs_count = mp * padded_groups
    total_count = (mp + np) * padded_groups
    lines = [
        "@cute.kernel",
        "def _prepare(a: cute.Tensor, b: cute.Tensor, sa: cute.Tensor, sb: cute.Tensor, workspace: cute.Tensor):",
        "    linear = cutlass.Int64(cute.arch.block_idx()[0]) * 256 + cutlass.Int64(cute.arch.thread_idx()[0])",
        "    words = cute.make_tensor(cute.recast_ptr(workspace.iterator, dtype=cutlass.Uint64), cute.make_layout(("
        + str(cast("int", plan["workspace_bytes"]) // 8)
        + ",)))",
    ]
    for side, source, scale in (("lhs", "a", "sa"), ("rhs", "b", "sb")):
        lines.extend(
            (
                f"    {source}_bytes = cute.make_tensor(cute.recast_ptr({source}.iterator, dtype=cutlass.Uint8), cute.make_layout(({plan[side + '_span']},)))",
                f"    {scale}_bytes = cute.make_tensor(cute.recast_ptr({scale}.iterator, dtype=cutlass.Uint8), cute.make_layout(({plan[side + '_scale_span']},)))",
            )
        )
    for (
        side,
        source,
        scale,
        rows,
        padded_rows,
        q_offset,
        sf_offset,
        predicate,
        slot,
    ) in (
        (
            "lhs",
            "a",
            "sa",
            m,
            mp,
            0,
            qa_bytes + qb_bytes,
            f"linear < {lhs_count}",
            "linear",
        ),
        (
            "rhs",
            "b",
            "sb",
            n,
            np,
            qa_bytes,
            qa_bytes + qb_bytes + sa_bytes,
            f"linear < {total_count}",
            f"linear - {lhs_count}",
        ),
    ):
        lines.extend(
            (
                f"    {'if' if side == 'lhs' else 'elif'} {predicate}:",
                f"        slot = {slot}",
            )
        )
        if plan[side + "_row_fast"]:
            lines.extend(
                (
                    f"        row = slot % {padded_rows}",
                    f"        group = slot // {padded_rows}",
                )
            )
        else:
            lines.extend(
                (
                    f"        row = slot // {padded_groups}",
                    f"        group = slot % {padded_groups}",
                )
            )
        lines.extend(
            (
                "        word = cutlass.Uint64(0)",
                "        sf = cutlass.Uint8(0)",
                f"        if row < {rows} and group < {groups}:",
                f"            signed_sf = {scale}_bytes[{plan[side + '_scale_address']}]",
                "            sf = signed_sf & cutlass.Uint8(127)",
                "            sign = cutlass.Uint8(((cutlass.Uint32(signed_sf) >> 7) & 1) * 136)",
            )
        )
        addresses = cast("tuple[str, ...]", plan[side + "_byte_addresses"])
        assert len(addresses) == 8
        for byte, address in enumerate(addresses):
            lines.extend(
                (
                    f"            q{byte} = {source}_bytes[{address}] ^ sign",
                    f"            word = word | ((cutlass.Uint64(q{byte}) & 255) << {8 * byte})",
                )
            )
        lines.extend(
            (
                f"        words[{q_offset // 8} + row * {padded_groups} + group] = word",
                f"        sf_position = {sf_offset} + (row // 128 * {padded_groups // 4} + group // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + group % 4",
                "        workspace[sf_position] = sf",
            )
        )
    lines.extend(
        (
            "",
            "@cute.jit",
            "def _entry(a: cute.Tensor, b: cute.Tensor, sa: cute.Tensor, sb: cute.Tensor, workspace: cute.Tensor, output: cute.Tensor, alpha: cutlass.Float32, stream: CUstream):",
            f"    _prepare(a, b, sa, sb, workspace).launch(grid=({(total_count + 255) // 256}, 1, 1), block=(256, 1, 1), stream=stream)",
            "    _consumer(workspace, output, alpha, stream)",
        )
    )
    return "\n".join(lines) + "\n"


def make_block_scaled_entry(plan: Mapping[str, object]) -> object:
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
