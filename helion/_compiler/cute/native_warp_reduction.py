"""Select exact native FP32 warp extrema through the grouped helper API."""

from __future__ import annotations

import ast

_HELPERS = {
    "_cute_grouped_reduce_warp",
    "_cute_grouped_reduce_shared_serial",
    "_cute_grouped_reduce_shared_two_stage",
    "_cute_grouped_reduce_cluster",
    "_cute_grouped_reduce_block",
    "_cute_grouped_reduce_cluster_online_pair",
    "_cute_grouped_reduce_cluster_warp_online_pair",
}


def enable_fp32_warp_redux(body: list[ast.stmt]) -> list[ast.stmt]:
    for statement in body:
        for node in ast.walk(statement):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _HELPERS
            ):
                continue
            if not any(keyword.arg == "fp32_redux" for keyword in node.keywords):
                node.keywords.append(
                    ast.keyword(arg="fp32_redux", value=ast.Constant(True))
                )
    return body
