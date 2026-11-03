"""Select exact native FP32 warp extrema through the grouped helper API."""

from __future__ import annotations

import ast

from .reduce_call import FP32_REDUX_REDUCES


def enable_fp32_warp_redux(body: list[ast.stmt]) -> list[ast.stmt]:
    for statement in body:
        for node in ast.walk(statement):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in FP32_REDUX_REDUCES
            ):
                continue
            if not any(keyword.arg == "fp32_redux" for keyword in node.keywords):
                node.keywords.append(
                    ast.keyword(arg="fp32_redux", value=ast.Constant(True))
                )
    return body
