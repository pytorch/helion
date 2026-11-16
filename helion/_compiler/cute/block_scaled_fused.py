"""Eligibility of the fused block-scaled preparation (``cute_scaled_prepare="fused"``).

The fused consumer reads the packed sides and their scales straight from
the source tensors: a K-contiguous side by TMA into its K-major stage
("direct"), a row-contiguous side by TMA into a raw ``[k-byte][row]``
staging tile that the transform warps transpose into K-major words
("transpose"), and the scales from their swizzled source layout, whose
sign bits the transform warps clear and fold into the data.  That needs
the compiler-proved byte recipes of each side to be the plain affine forms
below, the scale recipes to be exactly the 32x4x4 swizzle with the padded
group count the consumer's descriptors use, and the scale spans to cover
the padded extents.  Everything else keeps the preparation kernel.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from .block_scaled_prepare import _scale_position

if TYPE_CHECKING:
    from collections.abc import Callable


def _compile(expression: str) -> Callable[[int, int], int]:
    namespace = {"cutlass": SimpleNamespace(Int64=int, Int32=int)}
    code = compile(expression, "<recipe>", "eval")
    return lambda row, group: int(eval(code, namespace, {"row": row, "group": group}))


def _sample(rows: int, groups: int) -> tuple[list[int], list[int]]:
    """Rows and groups covering every residue of the swizzle and the block
    multipliers of both formulas (periods 128 rows and 4 groups)."""
    row_samples = sorted({*range(min(rows, 384)), rows - 1, max(rows - 129, 0)})
    group_samples = sorted({*range(min(groups, 12)), groups - 1, max(groups - 5, 0)})
    return row_samples, group_samples


def _side_mode(
    byte_addresses: tuple[str, ...], rows: int, groups: int
) -> tuple[str, int] | None:
    """``("direct", row stride)`` when byte ``b`` of ``(row, group)`` sits at
    ``row * S + group * 8 + b`` with ``S`` a multiple of 16, ``("transpose",
    k-byte stride)`` when it sits at ``(group * 8 + b) * S + row`` with ``S`` a
    multiple of 16, else ``None``."""
    if len(byte_addresses) != 8:
        return None
    address = [_compile(expression) for expression in byte_addresses]
    base = address[0](0, 0)
    if base != 0:
        return None
    row_samples, group_samples = _sample(rows, groups)

    def matches(expected: Callable[[int, int, int], int]) -> bool:
        return all(
            address[byte](row, group) == expected(row, group, byte)
            for row in row_samples
            for group in group_samples
            for byte in range(8)
        )

    row_stride = address[0](1, 0) if rows > 1 else 16
    if row_stride > 0 and row_stride % 16 == 0:

        def direct(row: int, group: int, byte: int) -> int:
            return row * row_stride + group * 8 + byte

        if matches(direct):
            return "direct", row_stride
    kbyte_stride = address[1](0, 0)
    if kbyte_stride > 0 and kbyte_stride % 16 == 0:

        def transpose(row: int, group: int, byte: int) -> int:
            return (group * 8 + byte) * kbyte_stride + row

        if matches(transpose):
            return "transpose", kbyte_stride
    return None


def _swizzled_scales(
    scale_address: str, scale_span: int, rows: int, groups: int, padded_groups: int
) -> bool:
    if scale_span < (rows + 127) // 128 * 128 * padded_groups:
        return False
    position = _compile(_scale_position(padded_groups))
    recipe = _compile(scale_address)
    row_samples, group_samples = _sample(rows, groups)
    return all(
        recipe(row, group) == position(row, group)
        for row in row_samples
        for group in group_samples
    )


def fused_geometry(
    *,
    lhs_byte_addresses: tuple[str, ...],
    rhs_byte_addresses: tuple[str, ...],
    lhs_scale_address: str,
    rhs_scale_address: str,
    lhs_scale_span: int,
    rhs_scale_span: int,
    m: int,
    n: int,
    k: int,
    bn: int,
    bk: int,
    cluster_m: int,
) -> dict[str, tuple[str, int]] | None:
    """``{"lhs": (mode, stride), "rhs": (mode, stride)}`` or ``None``."""
    groups = k // 16
    padded_groups = (k + bk - 1) // bk * bk // 16
    if cluster_m != 1 or k % 16 or padded_groups != (groups + 3) // 4 * 4:
        return None
    lhs = _side_mode(lhs_byte_addresses, m, groups)
    rhs = _side_mode(rhs_byte_addresses, n, groups)
    if lhs is None or rhs is None:
        return None
    # A transposed side stages a [k-byte][rows] tile under the TMA swizzle of
    # its row width, 64 or 128 bytes, and its eight transform warps own whole
    # sixteen-row-by-quad units (the 128-row M tile always qualifies).
    if rhs[0] == "transpose" and (bn not in (64, 128) or bn * bk % 8192):
        return None
    if not _swizzled_scales(
        lhs_scale_address, lhs_scale_span, m, groups, padded_groups
    ):
        return None
    if not _swizzled_scales(
        rhs_scale_address, rhs_scale_span, n, groups, padded_groups
    ):
        return None
    return {"lhs": lhs, "rhs": rhs}
