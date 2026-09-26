from __future__ import annotations

import ast

from helion._compiler.cute.fuse_fma import fuse_fma


def test_fuses_loop_carried_sum_of_product() -> None:
    body = ast.parse(
        """
acc = cutlass.Float32(0)
for lane in cutlass.range_constexpr(8):
    x = cutlass.Float32(values[lane])
    square = x * x
    acc = acc + square
result = acc
"""
    ).body

    result = fuse_fma(body)
    code = ast.unparse(ast.Module(body=result, type_ignores=[]))

    assert "square =" not in code
    assert "acc = cute.math.fma(x, x, acc)" in code
