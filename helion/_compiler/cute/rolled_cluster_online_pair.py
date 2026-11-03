"""Warp-local online-softmax pairs for a proved rolled cluster reduction."""

from __future__ import annotations

import ast
from typing import cast

from . import cluster_online_pair as pair
from ._ast_pass_utils import is_name_assign
from .analysis.online_pair import _PAIR_REDUCE
from .analysis.online_pair import OnlinePairMatcher
from .analysis.online_pair import _assigned_name
from .analysis.online_pair import _wrap_sites
from .analysis.online_pair import rolled_pair_gate

_WARP_PAIR = "_cute_grouped_reduce_cluster_warp_online_pair"


def _unwrap(working: list[ast.stmt], owners: set[int]) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    for statement in working:
        result.extend(
            cast("ast.For", statement).body if id(statement) in owners else [statement]
        )
    return result


def _fuse_cached_pair(
    working: list[ast.stmt],
    owners: set[int],
    constexpr_values: dict[str, int] | None,
    rename_groups: dict[str, str] | None,
) -> bool:
    """Run the explicit two-pass pair rewrite on the owner-wrapped rolled body.

    With every cluster-reduce site inside a one-trip owner the rolled form
    has the shape ``cluster_online_pair`` understands: under ``fast_math``
    it caches sweep B's exponentials (computed in the CTA's own maximum
    frame), exchanges the packed ``(max, sum)`` pair once at the CTA level
    and rescales the cached values in the write sweep.  Returns whether a
    pair was formed; the body is unchanged otherwise.
    """
    sites = OnlinePairMatcher(working).sites
    if not sites or len(sites) < 2:
        return False
    if any(id(site.for_node) not in owners for site in sites):
        return False
    pair.fuse_cluster_online_pair(
        working, constexpr_values, rename_groups=rename_groups, fast_math=True
    )
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == _PAIR_REDUCE
        for statement in working
        for node in ast.walk(statement)
    )


def fuse_rolled_cluster_online_pair(
    body: list[ast.stmt],
    constexpr_values: dict[str, int] | None = None,
    *,
    rename_groups: dict[str, str] | None = None,
    fast_math: bool = False,
) -> list[ast.stmt]:
    """One packed cluster exchange for a rolled online-softmax pair.

    Under ``fast_math`` the CTA-level pair with cached exponentials is tried
    first (one exchange, no second ``exp2`` sweep); otherwise, and as the
    fallback, the exact warp-level pair keeps every exponential as written
    and only moves sweep B into the frame of the warp's own maximum.
    """
    working, owners = _wrap_sites(body)
    if fast_math and _fuse_cached_pair(
        working, owners, constexpr_values, rename_groups
    ):
        return _unwrap(working, owners)
    for match in OnlinePairMatcher(working, rolled_pair_gate(owners)).proved():
        site_a, site_b, maximum, scale = (
            match.site_a,
            match.site_b,
            match.maximum,
            match.scale,
        )
        frame = pair._unique_name(working, lambda name: name, "_rolled_pair_frame")
        after_a = working.index(site_a.for_node) + 1
        working.insert(
            after_a,
            pair._stmt(
                f"{frame} = cutlass.Float32(0) if {maximum} == "
                f"cutlass.Float32(float('-inf')) else {maximum}"
            ),
        )
        for node in match.subtractions:
            node.id = frame
        call = site_a.call
        site_a.assign.value = cast(
            "ast.Assign",
            pair._stmt(
                f"_x = _cute_grouped_reduce_warp("
                f"{', '.join(ast.unparse(argument) for argument in call.args[:4])}, "
                "pre=1, group_span=32)"
            ),
        ).value
        pair._pair_site_b(site_b, maximum, scale, False, 0)
        new_call = cast("ast.Assign", site_b.for_node.body[site_b.stmt_idx]).value
        assert isinstance(new_call, ast.Call)
        new_call.func = ast.Name(id=_WARP_PAIR, ctx=ast.Load())
        pair._rebuffer_pair(working, site_a, site_b)
        for statement in working:
            if (
                is_name_assign(statement)
                and _assigned_name(statement) == site_b.buf_name
            ):
                statement.value = cast(
                    "ast.Assign",
                    pair._stmt(
                        f"_x = cute.arch.alloc_smem(cutlass.Int64, "
                        f"{cast('int', site_b.group_span) // 32 * cast('int', site_b.cluster_n)})"
                    ),
                ).value
        break
    return _unwrap(working, owners)
