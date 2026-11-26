"""Fold a proven FP32 row sum across an entire, converged physical CTA."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ... import exc
from ._ast_pass_utils import assigned_name
from ._ast_pass_utils import block_uniform_scalar

if TYPE_CHECKING:
    from ..device_function import DeviceFunction


@dataclass(frozen=True)
class RowSumAnchor:
    thread: str
    coordinate: str
    offset: int
    collective: tuple[str, ...]


def parallel_sum(
    fn: DeviceFunction,
    setup: ast.stmt,
    output: ast.For,
    acc: str,
    statistic: str,
    terms: list[ast.stmt],
    threads: int,
) -> list[ast.stmt] | None:
    # No scratch for one warp, at most 128 static bytes otherwise. The final
    # target-specific shared-memory check accounts for the entire kernel.
    if threads % 32 or not 1 <= (warps := threads // 32) <= 32:
        return None
    tid = assigned_name(setup)
    assert tid is not None
    # The store coordinate is a bijection from (thread, sweep) onto the full
    # row. Reuse precisely that walk for the addends, not the store's math.
    sweep = copy.copy(output)
    sweep.target = copy.deepcopy(output.target)
    sweep.iter = copy.deepcopy(output.iter)
    sweep.orelse = []
    sweep.body = [copy.deepcopy(output.body[0]), *terms]
    lane, partial, pointer, shared, pick = [
        fn.new_var("publication_" + x, dce=False)
        for x in ("lane", "partial", "pointer", "shared", "pick")
    ]
    fold = ast.parse(
        f"{lane} = cutlass.Int32(cute.arch.lane_idx())\n{partial} = cutlass.Float32(cute.arch.warp_reduction_sum({acc}, threads_in_group=32))"
    ).body
    if warps == 1:
        fold = fold[1:]  # No lane chooses a scratch slot in a single warp.
        finish = ast.parse(f"{statistic} = {partial}").body
    else:
        # A warp in every output cohort folds the partials by butterfly.
        # The grouped serial reducer folds them in a different FP32 order;
        # its two-stage alternative lets only warp zero compute the result
        # and publishes that result through a separate shared slot.
        finish = ast.parse(f"""
{pointer} = cute.arch.alloc_smem(cutlass.Float32, {warps}, alignment=16)
{shared} = cute.make_tensor({pointer}, ({warps},))
if {lane} == 0:
    {shared}[{tid} // 32] = {partial}
cute.arch.sync_threads()
{pick} = cutlass.Float32(0)
if {lane} < {warps}:
    {pick} = {shared}[{lane}]
{statistic} = cutlass.Float32(cute.arch.warp_reduction_sum({pick}, threads_in_group=32))
cute.arch.sync_threads()
""").body
    protocol = [*fold, *finish] if warps > 1 else [fold[-1]]
    fn.cute_state.lanes.parallel_row_sum_anchors[partial] = RowSumAnchor(
        tid, ast.dump(setup), len(fold) - 1, tuple(map(_protocol_stmt, protocol))
    )
    return [setup, sweep, *fold, *finish]


def _protocol_stmt(stmt: ast.stmt) -> str:
    # The AST finalizer can insert an empty else branch without changing
    # control flow. Everything that affects the scratch protocol must match.
    if isinstance(stmt, ast.If) and all(isinstance(s, ast.Pass) for s in stmt.orelse):
        stmt = copy.copy(stmt)
        stmt.orelse = []
    return ast.dump(stmt)


def validate_parallel_sum(
    definition: ast.FunctionDef, anchors: dict[str, RowSumAnchor]
) -> None:
    """Require the final collectives to be reached in order by the full CTA."""
    parents = {n: p for p in ast.walk(definition) for n in ast.iter_child_nodes(p)}
    # Warp leaders derive the destination slot from the same unique physical
    # thread coordinate as the complete-row sweep. Check its origin and all
    # definitions: a later pass may not replace it with a lane-local index.
    coordinates: dict[str, list[ast.AST]] = {a.thread: [] for a in anchors.values()}
    for node, parent in parents.items():
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            and node.id in coordinates
        ):
            coordinates[node.id].append(parent)
    marked: set[ast.stmt] = set()
    seen: set[str] = set()
    for node in ast.walk(definition):
        if (
            not isinstance(node, ast.Assign)
            or (name := assigned_name(node)) not in anchors
        ):
            continue
        assert name is not None
        if name in seen:
            raise exc.InvalidConfig("duplicated full-CTA row sum")
        seen.add(name)
        marked.add(node)
        suite = next(
            body
            for _, body in ast.iter_fields(parents[node])
            if isinstance(body, list) and node in body
        )
        # The first warp result is our stable anchor. In a multiwarp fold
        # its lane-id assignment is immediately before it. Match every
        # publication, barrier, load and fold in order, in this same suite;
        # a readiness fence before the leader stores is not a valid fence.
        anchor = anchors[name]
        expected = anchor.collective
        start = suite.index(node) - anchor.offset
        actual = suite[start : start + len(expected)]
        if start < 0 or tuple(map(_protocol_stmt, actual)) != expected:
            raise exc.InvalidConfig("changed block-uniform row sum protocol")
        origins = coordinates[anchor.thread]
        if not (
            len(origins) == 1
            and ast.dump(origins[0]) == anchor.coordinate
            and origins[0] in suite[: suite.index(node)]
        ):
            raise exc.InvalidConfig("changed physical row sum coordinate")
    if not anchors or seen != anchors.keys():
        raise exc.InvalidConfig("missing full-CTA row sum")
    path = set()
    for stmt in marked:
        node: ast.AST = stmt
        while node is not definition:
            path.add(node)
            node = parents[node]

    allowed = {"range", "cutlass.range_constexpr", "cutlass.Int32", "cutlass.Int64"}

    def uniform(expr: ast.AST, known: set[str]) -> bool:
        return block_uniform_scalar(
            expr, known.__contains__, allowed, integer_literals=True
        )

    def visit(body: list[ast.stmt], known: set[str]) -> None:
        for s in body:
            if isinstance(s, ast.Assign) and (lhs := assigned_name(s)) is not None:
                known.difference_update(
                    n.target.id
                    for n in ast.walk(s.value)
                    if isinstance(n, ast.NamedExpr) and isinstance(n.target, ast.Name)
                )
                if uniform(s.value, known):
                    known.add(lhs)
                else:
                    known.discard(lhs)
                continue
            targets = {
                n.id
                for n in ast.walk(s)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            }
            branch_facts: set[str] | None = None
            if isinstance(s, ast.If) and uniform(s.test, known):
                taken, other = known.copy(), known.copy()
                visit(s.body, taken)
                visit(s.orelse, other)
                # Every lane makes the same choice: a variable uniform on
                # both exits stays uniform even if its value differs by path.
                branch_facts = taken & other
            elif s in path and s not in marked:
                if (
                    isinstance(s, ast.For)
                    and isinstance(s.target, ast.Name)
                    and uniform(s.iter, known)
                    and not s.orelse
                ):
                    visit(s.body, (known - targets) | {s.target.id})
                else:
                    raise exc.InvalidConfig("row sum requires block-uniform dispatch")
            if any(
                isinstance(n, (ast.Return, ast.Break, ast.Continue))
                for n in ast.walk(s)
            ):
                raise exc.InvalidConfig(
                    "unsupported nonlocal control around CTA row sum"
                )
            if branch_facts is None:
                known.difference_update(targets)
            else:
                known.clear()
                known.update(branch_facts)

    visit(definition.body, {a.arg for a in definition.args.args})
