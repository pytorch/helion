"""Pack live E8M0 scale bytes while native MXFP8 TMA reads original operands."""

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
    m, n, k, bn, bk = [cast("int", plan[key]) for key in ("m", "n", "k", "bn", "bk")]
    mp, np, kp = block_scaled_workspace_shape(m, n, k, bn, bk)
    groups, padded_groups = k // 32, kp // 32
    lhs_count = mp * padded_groups
    total = (mp + np) * padded_groups
    lines = [
        "@cute.kernel",
        "def _prepare(sa: cute.Tensor, sb: cute.Tensor, workspace: cute.Tensor):",
        "    linear = cutlass.Int64(cute.arch.block_idx()[0]) * 256 + cutlass.Int64(cute.arch.thread_idx()[0])",
    ]
    for side, source, rows, offset, slot in (
        ("lhs", "sa", m, 0, "linear"),
        ("rhs", "sb", n, lhs_count, f"linear - {lhs_count}"),
    ):
        lines += [
            f"    {'if' if side == 'lhs' else 'elif'} linear < {lhs_count if side == 'lhs' else total}:",
            f"        slot = {slot}",
            f"        row = slot // {padded_groups}",
            f"        group = slot % {padded_groups}",
            "        sf = cutlass.Uint8(0)",
            f"        if row < {rows} and group < {groups}:",
            f"            sf = cutlass.Uint8(cute.recast_ptr({source}.iterator, dtype=cutlass.Uint8)[{plan[side + '_scale_address']}])",
            f"        position = {offset} + (row // 128 * {padded_groups // 4} + group // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + group % 4",
            "        workspace[position] = sf",
        ]
    rhs = (
        f"cute.make_tensor(b.iterator, cute.make_layout(({n}, {k}), stride=(b.stride[1], b.stride[0])))"
        if plan["rhs_transposed"]
        else "b"
    )
    lines += [
        "",
        "@cute.jit",
        "def _entry(a: cute.Tensor, b: cute.Tensor, sa: cute.Tensor, sb: cute.Tensor, workspace: cute.Tensor, output: cute.Tensor, alpha: cutlass.Float32, stream: CUstream):",
        f"    _prepare(sa, sb, workspace).launch(grid=({(total + 255) // 256}, 1, 1), block=(256, 1, 1), stream=stream)",
        f"    rhs = {rhs}",
        "    _consumer.mxfp8(a, rhs, workspace, output, alpha, stream)",
    ]
    return "\n".join(lines) + "\n"


def make_mxfp8_entry(plan: Mapping[str, object]) -> object:
    import cutlass
    import cutlass.cute as cute

    from .block_scaled_runtime import Sm100BlockScaledGemm

    source = preparation_source(plan)
    filename = f"<helion_mxfp8:{hashlib.sha256(source.encode()).hexdigest()}>"
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
                for name in cast("tuple[str,...]", plan["round_dtypes"])
            ),
            operand_format="mxfp8",
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
